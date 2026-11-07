"""The Bench2Drive ingest pools recorded route results without copying a report.

Pure: a synthetic results tree in ``tmp_path`` stands in for the site's
recordings. What it must get right: per route the latest attempt that scored
wins, a crashed attempt never shadows its rerun, a requested route with no
valid attempt marks the run partial, the aggregate is recomputed from the
per-route records, a route that was attempted and never scored is reported as
lost rather than dropped, and nothing written into ``results/`` names this
machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive.harness import SET_ASIDE
from paddock.config import BenchmarkOverride, ConfigError, SiteConfig
from paddock.results import legacy, store
from paddock.results.ingest import bench2drive


def _results_json(route: str, ds: float, rc: float, status: str = "Completed", collisions: int = 0) -> dict:
    return {
        "_checkpoint": {
            "records": [
                {
                    "route_id": f"RouteScenario_{route}",
                    "status": status,
                    "scores": {
                        "score_composed": ds,
                        "score_route": rc,
                        "score_penalty": ds / rc if rc else 0.0,
                    },
                    "infractions": {"collisions_vehicle": ["hit"] * collisions, "min_speed_infractions": []},
                    "meta": {"duration_system": 100.0},
                }
            ]
        }
    }


def _write(root: Path, run_id: str, folder: str, payload: dict) -> None:
    path = root / run_id / folder / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


def _unscored(route: str) -> dict:
    """What CARLA leaves behind when the route never ran: an empty checkpoint."""
    return {"_checkpoint": {"records": []}}


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "bench2drive" / "modelx"
    first = "20260901T000000-aaaaaa"
    _write(root, first, "1711-Town12-ParkingCutIn", _results_json("1711", 60.0, 100.0, collisions=1))
    _write(root, first, "2802-Town12-InvadingTurn", _results_json("2802", 100.0, 100.0))
    # 24333: first attempt crashed, later attempt drove
    crashed = _results_json("24333", 0.0, 0.0, status="Failed - Agent crashed")
    _write(root, "20260902T000000-bbbbbb", "24333-Town15-DynamicObjectCrossing", crashed)
    blocked = _results_json("24333", 16.59, 44.06, status="Failed - Agent got blocked")
    _write(root, "20260903T000000-cccccc", "24333-Town15-DynamicObjectCrossing", blocked)
    # an earlier, worse attempt of 2802 must lose to the later one above
    _write(root, "20260831T000000-000000", "2802-Town12-InvadingTurn", _results_json("2802", 10.0, 50.0))
    return root


def test_latest_scoring_attempt_wins_and_crashes_never_shadow(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    ((run, metrics, samples),) = bench2drive.ingest(
        root, model="modelx", site="test-site", run_label="pooled"
    )
    by_route = {s["native_id"]["route_id"]: s for s in samples}
    assert by_route["2802"]["native_id"]["run_id"] == "20260901T000000-aaaaaa"
    assert by_route["2802"]["metrics"]["driving_score"] == 100.0
    assert by_route["24333"]["native_id"]["run_id"] == "20260903T000000-cccccc"
    assert by_route["24333"]["metrics"]["driving_score"] == 16.59
    assert metrics.n == 3
    assert metrics.metrics["driving_score"]["mean"] == round((60.0 + 100.0 + 16.59) / 3, 6)
    assert metrics.metrics["collision_rate"]["mean"] == round(1 / 3, 6)
    assert set(metrics.by_split) == {"ParkingCutIn", "InvadingTurn", "DynamicObjectCrossing"}
    assert run.status == "complete" and run.valid
    assert run.config["source_runs"] == ["20260901T000000-aaaaaa", "20260903T000000-cccccc"]
    assert run.started_utc.startswith("2026-09-01T00:00:00")
    assert run.run_id == (
        "pooled-"
        + store.sequence_hash([s["sample_id"] for s in samples])
        + "-"
        + store.config_hash(run.config)
    )


def test_requested_route_without_a_measurement_marks_the_run_partial(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    routes = ["1711", "2802", "24333", "9999"]
    ((run, metrics, _),) = bench2drive.ingest(root, model="modelx", site="s", routes=routes)
    assert run.status == "partial" and not run.valid
    assert run.dataset == {
        "name": "bench2drive220",
        "n_samples": 3,
        "sequences_sha256": metrics.run_id.split("-")[1],
        "n_expected": 4,
        "n_failed": 1,
    }
    assert "9999" in run.notes


def test_written_run_record_names_no_machine_path(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    env = {"PADDOCK_CV_SPEED_M_S": "6"}
    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s", model_env=env)
    out = store.write_run(
        run, metrics, samples, root=tmp_path / "results", artifacts_root=tmp_path / "artifacts"
    )
    written = json.loads((out / "run.json").read_text())
    assert str(tmp_path) not in json.dumps(written)
    assert written["ingested_from"].endswith("bench2drive/modelx") and not written[
        "ingested_from"
    ].startswith("/")
    assert written["config"]["model_env"] == {"PADDOCK_CV_SPEED_M_S": "6"}
    assert json.loads((out / "metrics.json").read_text())["primary_metric"] == "driving_score"
    samples_file = tmp_path / "artifacts" / "bench2drive" / "modelx" / run.run_id / "samples.jsonl"
    lines = samples_file.read_text().splitlines()
    assert len(lines) == 3 and all("sha256" in line for line in lines)


def test_a_route_that_never_scored_is_expected_not_absent(tmp_path: Path) -> None:
    """The 23695 case: every attempt crashed, and `routes` is unset.

    Before this was fixed the expected set was built from the samples that
    survived, so `missing` was empty by construction: the route left no trace in
    `routes_subset`, `n_failed` stayed 0, the run claimed `complete`/`valid`, and
    the mean silently came from a smaller route set.
    """
    root = _tree(tmp_path)
    # Route 23695 in this repo's real trees: several attempts, every one of them
    # with an empty `_checkpoint.records`. Not set aside, so it is lost.
    for run in ("20260904T000000-dddddd", "20260905T000000-eeeeee"):
        _write(root, run, "23695-Town13-InterurbanAdvancedActorFlow", _unscored("23695"))

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert "23695" not in {s["native_id"]["route_id"] for s in samples}, "it did not score"
    assert "23695" in run.config["routes_subset"], "but it was attempted, so it was expected"
    assert run.dataset["n_expected"] == 4 and run.dataset["n_failed"] == 1
    assert run.status == "partial" and not run.valid
    assert "23695" in run.notes and "did not score" in run.notes
    assert metrics.n == 3, "the three routes that scored are what the mean is over"


def test_a_scenario_name_with_a_digit_or_an_underscore_is_scored_not_skipped(tmp_path: Path) -> None:
    """Ten of Bench2Drive-220's routes are MergerIntoSlowTrafficV2 or T_Junction.

    A folder pattern that read only letters in the scenario skipped them before
    `parse_results` ever saw them: not scored, not rejected, not expected, so the
    pooled run lost ten routes and said nothing.
    """
    root = _tree(tmp_path)
    run_id = "20260904T000000-dddddd"
    _write(root, run_id, "26401-Town06-MergerIntoSlowTrafficV2", _results_json("26401", 100.0, 100.0))
    _write(root, run_id, "28154-Town10HD-T_Junction", _results_json("28154", 20.0, 40.0))

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    scenarios = {s["native_id"]["route_id"]: s["native_id"]["scenario"] for s in samples}
    assert scenarios["26401"] == "MergerIntoSlowTrafficV2"
    assert scenarios["28154"] == "T_Junction"
    assert metrics.n == 5 and run.dataset["n_expected"] == 5


def test_a_route_excluded_by_the_subset_and_never_scored_is_still_recorded(tmp_path: Path) -> None:
    """11715 exactly: excluded for every model because CARLA crashes on it.

    An explicit `routes` list is a choice and keeps the run complete, but
    "excluded" and "attempted and lost" are different facts and `results/` must
    hold both -- otherwise the only place the loss exists is a prose sentence in
    a benchmark doc.
    """
    root = _tree(tmp_path)
    for run in ("20260904T000000-dddddd", "20260905T000000-eeeeee"):
        _write(root, run, "11715-Town12-EnterActorFlow", _unscored("11715"))

    ((run, metrics, _),) = bench2drive.ingest(
        root, model="modelx", site="s", routes=["1711", "2802", "24333"]
    )

    assert run.status == "complete" and run.valid, "the subset it was asked for is complete"
    assert run.dataset["n_expected"] == 3 and run.dataset["n_failed"] == 0
    assert "11715" not in run.config["routes_subset"]
    assert list(run.config["routes_never_scored"]) == ["11715"]
    assert "20260905T000000-eeeeee" in run.config["routes_never_scored"]["11715"]
    assert "attempted outside the configured route subset" in run.notes
    # The 50-route pools' case: a set aside route the list asked for none of is not
    # "set aside" in this pool, so its config -- and run id -- are what they were.
    assert "routes_set_aside" not in run.config


def test_a_route_set_aside_is_neither_expected_nor_lost_and_says_why(tmp_path: Path) -> None:
    """11715 fails in the simulator whichever model drives, so it measures no model.

    It leaves the pool -- not expected, not lost, in no mean -- and the record names it
    with its reason, while its attempts stay on the record as evidence.
    """
    root = _tree(tmp_path)
    for run in ("20260904T000000-dddddd", "20260905T000000-eeeeee"):
        _write(root, run, "11715-Town12-EnterActorFlow", _unscored("11715"))

    ((run, metrics, _),) = bench2drive.ingest(root, model="modelx", site="s")

    assert run.status == "complete" and run.valid
    assert run.dataset["n_expected"] == 3 and run.dataset["n_failed"] == 0
    assert run.config["routes_set_aside"] == ["11715"]
    assert "11715" not in run.config["routes_subset"]
    assert "11715" in run.config["routes_never_scored"], "its attempts are still evidence"
    assert SET_ASIDE["11715"] in run.notes and "set aside for every model" in run.notes
    assert "attempted outside the configured route subset" not in run.notes
    assert metrics.n == 3


def test_a_set_aside_route_that_scored_is_left_out(tmp_path: Path) -> None:
    """A route set aside because it fails for another model is out of this one's pool too:
    neither its sample nor its run's config enters it, so every mean is over the same routes.
    That run was driven through another camera rig: had its config entered the pool, the pool
    would not rank."""
    root = _tree(tmp_path)
    _write(root, "20260906T000000-ffffff", "11715-Town12-EnterActorFlow", _results_json("11715", 90.0, 100.0))
    other = _source_run_json("modelx", [], {})
    other["config"]["camera_rig"] = "kitscenes"
    (root / "20260906T000000-ffffff" / "run.json").write_text(json.dumps(other))

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert "11715" not in {s["native_id"]["route_id"] for s in samples}
    assert "20260906T000000-ffffff" not in run.config["source_runs"]
    assert run.config["routes_set_aside"] == ["11715"]
    assert metrics.n == 3 and run.valid
    assert "11715 left out although it scored here" in run.notes


def test_a_full_set_pool_names_a_set_aside_route_no_sweep_drove(tmp_path: Path) -> None:
    """With every route of the file asked for, a route set aside is named although no attempt of
    it exists, and a route never attempted is lost, not absent."""
    root = _tree(tmp_path)
    routes = ["1711", "2802", "24333", "11715", "9999"]
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s", routes=routes)
    assert run.config["routes_set_aside"] == ["11715"]
    assert "routes_never_scored" not in run.config
    assert run.dataset["n_expected"] == 4 and run.dataset["n_failed"] == 1
    assert run.status == "partial" and "9999" in run.notes


ROUTES_XML = """<routes>
<route id="1711" town="Town12"><scenarios><scenario name="a" type="ParkingCutIn"/></scenarios></route>
<route id="11715" town="Town12"><scenarios><scenario name="b" type="EnterActorFlow"/></scenarios></route>
</routes>
"""


def test_routes_all_is_every_route_of_the_sites_routes_file(tmp_path: Path) -> None:
    """`routes = "all"` asks for the routes file's every id; a bare string is refused by name, as
    iterating it once asked for routes 1, 7, 1 and 1."""
    data = tmp_path / "Bench2Drive" / "leaderboard" / "data"
    data.mkdir(parents=True)
    (data / "routes.xml").write_text(ROUTES_XML)
    settings = {"repo": str(tmp_path / "Bench2Drive"), "routes_file": "routes.xml"}
    cfg = _site(tmp_path, "", benchmarks={"bench2drive": BenchmarkOverride(settings)})
    assert legacy._bench2drive_routes(cfg, {"routes": "all"}) == ["1711", "11715"]
    assert legacy._bench2drive_routes(cfg, {"routes": ["1711", 2802]}) == ["1711", "2802"]
    assert legacy._bench2drive_routes(cfg, {}) is None
    with pytest.raises(ConfigError, match='list of route ids or "all"'):
        legacy._bench2drive_routes(cfg, {"routes": "1711"})
    with pytest.raises(ConfigError, match="needs \\[benchmarks.bench2drive\\] repo"):
        legacy._bench2drive_routes(_site(tmp_path, ""), {"routes": "all"})


def test_a_failure_record_gives_the_reason_a_route_never_scored(tmp_path: Path) -> None:
    """A CARLA crash and an attempt cut off by a restart leave the same empty checkpoint;
    the harness's failure.json says which."""
    root = _tree(tmp_path)
    folder = "23695-Town13-InterurbanAdvancedActorFlow"
    _write(root, "20260904T000000-dddddd", folder, _unscored("23695"))
    failure = {"kind": "carla_crash", "signal": 11, "game_time_s": 36.4, "returncode": -15}
    (root / "20260904T000000-dddddd" / folder / "failure.json").write_text(json.dumps(failure))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    said = run.config["routes_never_scored"]["23695"]
    assert said == "20260904T000000-dddddd: CARLA crashed (signal 11) at 36.4 s of game time"


def test_an_attempt_cut_off_later_does_not_hide_the_failures_on_record(tmp_path: Path) -> None:
    """Five crashes and then a cut-off attempt (route 11715's real trees): the reason is the crash."""
    root = _tree(tmp_path)
    folder = "23695-Town13-InterurbanAdvancedActorFlow"
    failure = {"kind": "carla_crash", "signal": 11, "game_time_s": 36.4, "returncode": -15}
    for run in ("20260904T000000-dddddd", "20260905T000000-eeeeee"):
        _write(root, run, folder, _unscored("23695"))
        (root / run / folder / "failure.json").write_text(json.dumps(failure))
    _write(root, "20260906T000000-ffffff", folder, _unscored("23695"))  # cut off: no failure record
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    said = run.config["routes_never_scored"]["23695"]
    assert said == "20260905T000000-eeeeee: CARLA crashed (signal 11) at 36.4 s of game time"


def test_without_a_failure_record_the_reason_is_the_parsers_unchanged(tmp_path: Path) -> None:
    """routes_never_scored is hashed into the run id, so a stored record ingested again must read
    exactly as it did."""
    root = _tree(tmp_path)
    _write(root, "20260904T000000-dddddd", "23695-Town13-InterurbanAdvancedActorFlow", _unscored("23695"))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    said = run.config["routes_never_scored"]["23695"]
    assert said == "20260904T000000-dddddd: no records in _checkpoint (the route did not score)"


def test_the_rejection_reason_names_the_run_and_no_machine_path(tmp_path: Path) -> None:
    """`notes` is tracked, so the reason may name a run id but never a disk."""
    root = _tree(tmp_path)
    _write(root, "20260904T000000-dddddd", "23695-Town13-InterurbanAdvancedActorFlow", _unscored("23695"))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    assert "20260904T000000-dddddd" in run.notes
    assert str(tmp_path) not in run.notes and "/" not in run.notes.split("23695", 1)[1]


def test_an_unreadable_results_file_is_a_rejection_not_a_crash(tmp_path: Path) -> None:
    """One truncated file on a network mount used to abort the whole walk."""
    root = _tree(tmp_path)
    truncated = root / "20260904T000000-dddddd" / "23695-Town13-InterurbanAdvancedActorFlow" / "results.json"
    truncated.parent.mkdir(parents=True)
    truncated.write_text('{"_checkpoint": {"records": [')

    ((run, metrics, _),) = bench2drive.ingest(root, model="modelx", site="s")

    assert metrics.n == 3, "the routes that did score are still ingested"
    assert run.status == "partial" and "unreadable" in run.notes


def test_duration_ignores_the_repetitions_the_parser_dropped(tmp_path: Path) -> None:
    """QwenDrive's real 24333: two crashed attempts at 1.0 s, one drive at 1245.3."""
    root = tmp_path / "bench2drive" / "modely"
    payload = _results_json("24333", 16.59, 44.06, status="Failed - Agent got blocked")
    crashed = _results_json("24333", 0.0, 0.0, status="Failed - Agent crashed")["_checkpoint"]["records"][0]
    crashed["meta"]["duration_system"] = 1.002
    payload["_checkpoint"]["records"][0]["meta"]["duration_system"] = 1245.267
    payload["_checkpoint"]["records"].insert(0, crashed)
    _write(root, "20260903T000000-cccccc", "24333-Town15-DynamicObjectCrossing", payload)

    ((_, _, samples),) = bench2drive.ingest(root, model="modely", site="s")

    assert samples[0]["metrics"]["n_runs"] == 1
    assert samples[0]["metrics"]["duration_s"] == 1245.3


# --- the run id and the config describe the run, not the operator -------------


def _source_run_json(
    model: str, routes: list[str], env: dict, routes_file: str = "bench2drive220.xml"
) -> dict:
    """What the harness writes beside the route folders of a run it drove."""
    return {
        "run_id": "20260901T000000-aaaaaa",
        "site": "a-machine",
        "benchmark": "bench2drive",
        "model": model,
        "config": {
            "routes_file": routes_file,
            "routes_subset": routes,
            "waypoint_dt": 0.25,
            "model_env": env,
        },
    }


def test_two_speed_bands_of_one_model_do_not_share_a_run_directory(tmp_path: Path) -> None:
    """`write_run` overwrites, so a run id that ignores the config loses a band.

    Both entries carry the documented `label = "pooled50"` and the same routes;
    only `model_env` differs, which for constant velocity is the entire
    experiment. With the route set alone in the id they landed in one directory
    holding whichever ran last, with no error.
    """
    root = _tree(tmp_path)
    written = []
    for speed in ("3", "15"):
        ((run, metrics, samples),) = bench2drive.ingest(
            root,
            model="constant_velocity",
            site="s",
            run_label="pooled50",
            model_env={"PADDOCK_CV_SPEED_M_S": speed},
        )
        written.append(store.write_run(run, metrics, samples, root=tmp_path / "results"))

    assert written[0] != written[1], "the two bands must not be one directory"
    assert len({p.name for p in (tmp_path / "results" / "bench2drive" / "constant_velocity").iterdir()}) == 2
    for path, speed in zip(written, ("3", "15"), strict=True):
        record = json.loads((path / "run.json").read_text())
        assert record["config"]["model_env"]["PADDOCK_CV_SPEED_M_S"] == speed
    # The route set is still legible in the id, and identical for the two.
    assert written[0].name.split("-")[1] == written[1].name.split("-")[1]


def test_routes_recorded_for_another_model_are_refused(tmp_path: Path) -> None:
    """A root one component too high pooled two models into one row, silently."""
    root = tmp_path / "bench2drive"
    for model, route, ds in (("modelx", "1711", 10.0), ("othermodel", "2802", 90.0)):
        _write(
            root / model,
            "20260901T000000-aaaaaa",
            f"{route}-Town12-InvadingTurn",
            _results_json(route, ds, 100.0),
        )
        (root / model / "20260901T000000-aaaaaa" / "run.json").write_text(
            json.dumps(_source_run_json(model, [route], {}))
        )

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert [s["native_id"]["route_id"] for s in samples] == ["1711"]
    assert metrics.metrics["driving_score"]["mean"] == 10.0, "not the mean of two models"
    assert run.status == "partial" and not run.valid
    assert "othermodel" in run.notes and "2802" in run.notes


def test_config_is_read_from_the_source_runs(tmp_path: Path) -> None:
    """`routes_file` is a site setting and `waypoint_dt` a model capability."""
    root = _tree(tmp_path)
    for run_id in ("20260901T000000-aaaaaa", "20260903T000000-cccccc"):
        (root / run_id / "run.json").write_text(
            json.dumps(_source_run_json("modelx", [], {"OMP_NUM_THREADS": "2"}, routes_file="other220.xml"))
        )

    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s", model_env={"PADDOCK_X": "1"})

    assert run.config["routes_file"] == "other220.xml", "not the ingest's constant"
    assert run.config["waypoint_dt"] == 0.25
    assert run.config["model_env"] == {"OMP_NUM_THREADS": "2", "PADDOCK_X": "1"}
    assert run.config["source_sites"] == ["a-machine"]
    assert run.config["source_config_read"] == "2/2 source runs"
    assert "run.json" not in run.notes, "nothing was assumed, so nothing to warn about"


def _pooled_under(tmp_path: Path, declared: tuple[dict | None, dict | None]):
    """Pool the two contributing source runs of `_tree`, each recorded under `declared`."""
    root = _tree(tmp_path)
    for run_id, env in zip(("20260901T000000-aaaaaa", "20260903T000000-cccccc"), declared, strict=True):
        record = _source_run_json("modelx", [], {})
        if env is not None:
            record["config"]["declared_env"] = env
        (root / run_id / "run.json").write_text(json.dumps(record))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s", run_label="pooled")
    return run


def test_the_model_yaml_env_of_the_source_runs_reaches_the_pooled_record(tmp_path: Path) -> None:
    """The declared-env check, one level up: the pooled config was rebuilt without `declared_env`."""
    on = {"DRIVEJEPA_MOMENTUM": "on"}
    run = _pooled_under(tmp_path / "a", (on, on))
    assert run.config["declared_env"] == on
    assert run.valid, run.notes


def test_source_runs_under_different_model_yaml_env_do_not_pool_into_one_ranked_number(
    tmp_path: Path,
) -> None:
    """Momentum off and on pooled into one valid record, with the id of an all-on pool."""
    off, on = {"DRIVEJEPA_MOMENTUM": "off"}, {"DRIVEJEPA_MOMENTUM": "on"}
    mixed = _pooled_under(tmp_path / "mixed", (off, on))
    same = _pooled_under(tmp_path / "same", (on, on))

    assert mixed.config["declared_env"] == [off, on], "recorded disagreeing, like model_env"
    assert not mixed.valid
    assert "model.yaml env" in mixed.notes
    assert mixed.run_id != same.run_id


def test_a_source_run_that_recorded_no_model_yaml_env_is_not_read_as_agreeing(tmp_path: Path) -> None:
    """`_agreed` skips a missing value, so a run from before 21 Oct 2026 would take the others' env."""
    on = {"DRIVEJEPA_MOMENTUM": "on"}
    run = _pooled_under(tmp_path / "half", (None, on))
    assert not run.valid
    assert "1 of 2 source runs recorded no declared_env" in run.notes

    # Every source run from before 21 Oct 2026: nothing to compare, and the record is as it was.
    old = _pooled_under(tmp_path / "old", (None, None))
    assert "declared_env" not in old.config
    assert old.valid, old.notes


def test_source_runs_that_disagree_are_recorded_disagreeing(tmp_path: Path) -> None:
    """Collapsing to the first value is how a record asserts a config half its routes ran under."""
    root = _tree(tmp_path)
    for run_id, routes_file in (("20260901T000000-aaaaaa", "a.xml"), ("20260903T000000-cccccc", "b.xml")):
        (root / run_id / "run.json").write_text(
            json.dumps(_source_run_json("modelx", [], {}, routes_file=routes_file))
        )

    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")

    assert run.config["routes_file"] == ["a.xml", "b.xml"]


def test_a_tree_without_run_records_says_so(tmp_path: Path) -> None:
    """The oldest campaigns kept neither file; the record must not pretend otherwise."""
    root = _tree(tmp_path)
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    assert run.config["source_config_read"] == "0/2 source runs"
    assert run.config["source_sites"] == []
    assert "kept no run.json" in run.notes and "0/2 source runs recorded" in run.notes


# --- a pooled run says which setup each route was driven under ---------------


def test_recording_setups_land_in_by_group_and_stop_the_row_ranking(tmp_path: Path) -> None:
    """OpenPilot's row: 16 routes with human resume, 26 on a different upstream
    longitudinal setup, 8 plain. `paddock report` prints metrics, never `notes`,
    so the prose caveat was invisible to it and the pooled mean ranked second.
    """
    root = _tree(tmp_path)
    setups = {"assisted": ["1711"], "native-long": ["2802"], "plain": ["24333"]}

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s", setups=setups)

    assert {s["native_id"]["route_id"]: s["group"] for s in samples} == {
        "1711": "assisted",
        "2802": "native-long",
        "24333": "plain",
    }
    assert set(metrics.by_group) == {"assisted", "native-long", "plain"}
    assert metrics.by_group["assisted"]["driving_score"]["mean"] == 60.0
    assert metrics.by_group["native-long"]["driving_score"]["mean"] == 100.0
    assert not run.valid, "a mean over three setups is a measurement of none of them"
    assert run.status == "complete", "every route scored; it is the pooling that is invalid"
    assert "assisted, native-long, plain" in run.notes and "by_group" in run.notes


def test_one_setup_over_every_route_still_ranks(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    ((run, metrics, _),) = bench2drive.ingest(
        root, model="modelx", site="s", setups={"unassisted": ["1711", "2802", "24333"]}
    )
    assert run.valid and set(metrics.by_group) == {"unassisted"}


def test_without_setups_nothing_changes(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    ((run, metrics, _),) = bench2drive.ingest(root, model="modelx", site="s")
    assert run.valid and metrics.by_group == {}


# --- the rate: 20 Hz, or the model's own declared rate -------------------------


def test_a_run_driven_at_the_models_declared_rate_ranks(tmp_path: Path, monkeypatch) -> None:
    """`paddock run` drives a model at its planning_hz and ranks it; the ingest agrees.

    The native-rate campaign's records were made valid by editing run.json, which a
    re-ingest silently undid. The rule is here instead, and the note names the key.
    """
    monkeypatch.setattr(bench2drive, "_registered_native_hz", lambda model: (2.0, "planning_hz"))
    env = {"PADDOCK_INFER_HZ": "2"}
    ((run, _, _),) = bench2drive.ingest(_tree(tmp_path), model="modelx", site="s", model_env=env)
    assert run.valid, run.notes
    assert "driven at 2 Hz, the rate modelx/model.yaml declares as planning_hz" in run.notes


@pytest.mark.parametrize(
    "declared", [(10.0, "native_input_hz"), None], ids=["declares another rate", "declares none"]
)
def test_a_rate_the_model_does_not_declare_does_not_rank(tmp_path: Path, monkeypatch, declared) -> None:
    """Every route scored; it is the rate that keeps the row off the board."""
    monkeypatch.setattr(bench2drive, "_registered_native_hz", lambda model: declared)
    env = {"PADDOCK_INFER_HZ": "2"}
    ((run, _, _),) = bench2drive.ingest(_tree(tmp_path), model="modelx", site="s", model_env=env)
    assert run.status == "complete" and not run.valid
    assert "driven at 2 Hz, neither the 20 Hz" in run.notes and "does not rank" in run.notes


def test_ranked_ranks_a_run_as_recorded_and_says_why(tmp_path: Path, monkeypatch) -> None:
    """Ranking an off-rate, partial run is an ingest input, not a run.json edit."""
    monkeypatch.setattr(bench2drive, "_registered_native_hz", lambda model: None)
    routes = ["1711", "2802", "24333", "9999"]
    env = {"PADDOCK_INFER_HZ": "2"}
    ((plain, _, _),) = bench2drive.ingest(
        _tree(tmp_path / "a"), model="modelx", site="s", routes=routes, model_env=env
    )
    ((ranked, _, _),) = bench2drive.ingest(
        _tree(tmp_path / "b"),
        model="modelx",
        site="s",
        routes=routes,
        model_env=env,
        ranked=True,
    )
    assert not plain.valid and ranked.valid
    assert ranked.status == "partial" and ranked.run_id == plain.run_id
    assert "ranked = true in the ingest config; the mean is over the 3 routes that scored." in ranked.notes


def test_the_declared_rate_comes_from_the_registry() -> None:
    """planning_hz first; native_input_hz stands in for a model that declares no planning rate."""
    assert bench2drive._registered_native_hz("drivejepa") == (2.0, "planning_hz")
    assert bench2drive._registered_native_hz("alpamayo15") == (10.0, "native_input_hz")
    assert bench2drive._registered_native_hz("no-such-model") is None


# --- the site config is read by a stranger, so it fails by name --------------


def _site(tmp_path: Path, toml: str, benchmarks: dict | None = None) -> SiteConfig:
    source = tmp_path / "paddock.toml"
    source.write_text(toml)
    return SiteConfig(
        site="s",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={},
        models={},
        benchmarks=benchmarks or {},
        gpu=None,  # type: ignore[arg-type]
        source=source,
    )


def test_a_single_bracket_ingest_table_is_named_not_a_typeerror(tmp_path: Path) -> None:
    """`[ingest.bench2drive]` iterated the dict's keys; `entry["runs"]` then raised
    `TypeError: string indices must be integers`, an hour from its cause."""
    cfg = _site(tmp_path, '[ingest.bench2drive]\nmodel = "drivejepa"\nruns = "/nowhere"\n')
    with pytest.raises(ConfigError, match=r"\[\[ingest.bench2drive\]\]"):
        legacy.ingest_all(cfg, dry_run=True)


def test_a_missing_key_names_the_key(tmp_path: Path) -> None:
    cfg = _site(tmp_path, '[[ingest.bench2drive]]\nmodel = "drivejepa"\n')
    with pytest.raises(ConfigError, match="missing 'runs'"):
        legacy.ingest_all(cfg, dry_run=True)


def _with_early_stop(record: dict, rules: dict) -> dict:
    record["config"]["early_stop"] = rules
    return record


def test_an_early_stop_is_read_from_the_source_runs_and_the_row_still_ranks(tmp_path: Path) -> None:
    """A route the early stop ended is a failed drive; each route says whether one was, and the notes why."""
    root = _tree(tmp_path)
    rules = {"stuck_s": 30.0, "expert_factor": 4.0, "progress_m": 3.0}
    for run_id in ("20260901T000000-aaaaaa", "20260903T000000-cccccc"):
        record = _with_early_stop(_source_run_json("modelx", [], {}), rules)
        (root / run_id / "run.json").write_text(json.dumps(record))
    route_dir = root / "20260903T000000-cccccc" / "24333-Town15-DynamicObjectCrossing"
    (route_dir / "early_stop.json").write_text(json.dumps({"reason": "stuck", "game_time_s": 41.2}))
    (root / "20260901T000000-aaaaaa" / "1711-Town12-ParkingCutIn" / "early_stop.json").write_text(
        json.dumps({"reason": None})
    )

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert run.config["early_stop"] == rules
    assert run.valid and run.status == "complete"
    assert "early stop (30 s without 3 m of route progress or 4x the expert's time" in run.notes
    assert "expected to move" in run.notes and "failed drive" in run.notes
    assert "ran without it" not in run.notes
    by_route = {s["native_id"]["route_id"]: s["metrics"] for s in samples}
    assert by_route["24333"]["ended_stuck"] == 1.0
    assert by_route["1711"]["ended_stuck"] == 0.0 and by_route["1711"]["ended_over_time"] == 0.0
    assert metrics.metrics["ended_stuck"]["n"] == 2, "2802's run wrote no record, so says nothing"


def test_the_routes_without_an_expert_drive_are_counted_among_the_pooled_routes(tmp_path: Path) -> None:
    """The runs record the site's routes the over-time rule cannot end; the pooled record covers fewer."""
    rules = {"expert_factor": 4.0, "no_expert": ["1711", "3364", "3373"]}  # 1711 alone is pooled here
    run = _pooled_with(tmp_path, ({"early_stop": rules}, {"early_stop": rules}))
    assert run.config["early_stop"] == rules, "the config keeps what the runs recorded"
    assert "(none on the 1 route(s) here with no expert drive)" in run.notes


def test_routes_driven_before_the_early_stop_are_counted_out_loud(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    rules = {"stuck_s": 30.0, "progress_m": 3.0}
    (root / "20260903T000000-cccccc" / "run.json").write_text(
        json.dumps(_with_early_stop(_source_run_json("modelx", [], {}), rules))
    )
    (root / "20260901T000000-aaaaaa" / "run.json").write_text(json.dumps(_source_run_json("modelx", [], {})))

    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")

    assert run.config["early_stop"] == rules
    assert "1 of 2 source runs ran without it" in run.notes
    assert not run.valid, "with and without the rule are two experiments, as two rules are"


def _pooled_with(tmp_path: Path, configs: tuple[dict | None, dict | None]):
    """Pool `_tree`'s two contributing source runs, each with these keys in its recorded config;
    None is a run that kept no config at all."""
    root = _tree(tmp_path)
    for run_id, extra in zip(("20260901T000000-aaaaaa", "20260903T000000-cccccc"), configs, strict=True):
        record = _source_run_json("modelx", [], {})
        if extra is None:
            record.pop("config")
        else:
            record["config"].update(extra)
        (root / run_id / "run.json").write_text(json.dumps(record))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    return run


def test_a_pool_whose_runs_override_a_setting_the_model_declares_does_not_rank(tmp_path: Path) -> None:
    """What makes a `paddock run` valid: false makes the pool so too. AutoE2E's sweep set three of
    its declared settings (10 Hz, a steady ego history, no route); its record directory is tuning,
    which model.yaml does not declare, and a key restating its default is no override."""
    declared = {"AUTOE2E_EGO_HISTORY": "real", "AUTOE2E_REPLAN_S": "0", "AUTOE2E_ROUTE": "1"}
    env = {"AUTOE2E_EGO_HISTORY": "steady", "AUTOE2E_REPLAN_S": "0.1", "AUTOE2E_RECORD_DIR": "<site path:1>"}
    sweep = {"declared_env": declared, "model_env": {**env, "AUTOE2E_ROUTE": "0"}}
    run = _pooled_with(tmp_path / "a", (sweep, sweep))
    assert not run.valid
    assert "AUTOE2E_EGO_HISTORY, AUTOE2E_REPLAN_S, AUTOE2E_ROUTE" in run.notes
    assert "AUTOE2E_RECORD_DIR" not in run.notes
    restated = {"declared_env": declared, "model_env": {"AUTOE2E_ROUTE": "1", "AUTOE2E_RECORD_DIR": "<x>"}}
    assert _pooled_with(tmp_path / "b", (restated, restated)).valid


def test_a_pool_keeps_ranking_on_a_setting_its_model_ranks_with_a_note(tmp_path: Path, monkeypatch) -> None:
    """The same exception a `paddock run` has. Any other declared setting still stops the pool."""
    monkeypatch.setattr(bench2drive, "_registered_ranked_with_note", lambda _model: ("X_VIEWS",))
    declared = {"X_VIEWS": "ring", "X_MODE": "a"}
    noted = {"declared_env": declared, "model_env": {"X_VIEWS": "replicate"}}
    assert _pooled_with(tmp_path / "a", (noted, noted)).valid
    both = {"declared_env": declared, "model_env": {"X_VIEWS": "replicate", "X_MODE": "b"}}
    run = _pooled_with(tmp_path / "b", (both, both))
    assert not run.valid
    assert "X_MODE" in run.notes and "X_VIEWS" not in run.notes


def test_a_pool_driven_on_another_camera_rig_names_it_and_does_not_rank(tmp_path: Path) -> None:
    """The rig is in each run's config only when it is not paddock's, so a pool of default-rig runs
    keeps its run id; another rig, or a mix, is recorded and does not rank."""
    kit = {"camera_rig": "kitscenes"}
    run = _pooled_with(tmp_path / "a", (kit, kit))
    assert run.config["camera_rig"] == "kitscenes" and not run.valid
    assert "kitscenes" in run.notes
    mixed = _pooled_with(tmp_path / "b", (kit, {}))
    assert mixed.config["camera_rig"] == ["kitscenes", "paddock"] and not mixed.valid
    default = _pooled_with(tmp_path / "c", ({}, {}))
    assert "camera_rig" not in default.config and default.valid
    assert run.run_id != default.run_id


def test_a_pool_asked_at_the_rate_its_model_declared_records_it_and_ranks(tmp_path: Path) -> None:
    """Every run since 23 Oct 2026 records how often the node was asked (`infer_hz`). AutoE2E's 50 routes
    pooled without it: the record named no rate and took the run id of the same routes asked every
    tick. A recorded rate with no `planning_hz` beside it is the one the model declared, so the pool
    ranks -- judged from the runs, not from model.yaml as it is when the ingest runs."""
    run = _pooled_with(tmp_path / "a", ({"infer_hz": 10.0}, {"infer_hz": 10}))
    assert run.config["infer_hz"] == 10.0 and run.valid, "10 and 10.0 are one rate"
    assert "10 Hz, the planning_hz the model's model.yaml declared" in run.notes
    every = _pooled_with(tmp_path / "b", ({}, {}))
    assert "infer_hz" not in every.config and every.valid, "a pool of runs asked every tick keeps its id"
    assert run.run_id != every.run_id
    at_20 = _pooled_with(tmp_path / "c", ({"infer_hz": 20.0}, {"infer_hz": 20.0}))
    assert at_20.valid and at_20.run_id == every.run_id, "20 Hz on a 20 Hz simulation is every tick"


def test_a_pool_at_the_hosts_rate_at_two_rates_or_at_a_rate_not_known_does_not_rank(tmp_path: Path) -> None:
    """The host's PADDOCK_INFER_HZ in place of the declared rate is recorded with the declared one
    beside it: an operator's rate. One mean over two rates describes neither, and a run that kept no
    config may have been asked at any rate. None of them ranks, and the notes say which."""
    host = _pooled_with(tmp_path / "a", ({"infer_hz": 2.0, "planning_hz": 10.0},) * 2)
    assert (host.config["infer_hz"], host.config["planning_hz"]) == (2.0, 10.0) and not host.valid
    assert "the rate the host's PADDOCK_INFER_HZ set" in host.notes
    forced = _pooled_with(tmp_path / "b", ({"infer_hz": 0.0, "planning_hz": 10.0},) * 2)
    assert forced.config["infer_hz"] == 0.0 and not forced.valid, "every tick, forced on a 10 Hz model"
    two = _pooled_with(tmp_path / "c", ({"infer_hz": 10.0}, {"infer_hz": 2.0}))
    assert sorted(two.config["infer_hz"]) == [2.0, 10.0] and not two.valid
    with_every_tick = _pooled_with(tmp_path / "d", ({"infer_hz": 10.0}, {}))
    assert sorted(with_every_tick.config["infer_hz"]) == [0.0, 10.0] and not with_every_tick.valid
    unknown = _pooled_with(tmp_path / "e", ({"infer_hz": 10.0}, None))
    assert not unknown.valid and "1 of 2 source runs kept no config" in unknown.notes
    garbled = _pooled_with(tmp_path / "f", ({"infer_hz": "fast"}, {"infer_hz": "fast"}))
    assert not garbled.valid and "'fast', which is not a rate" in garbled.notes


def test_the_old_rate_rule_reads_only_runs_that_recorded_no_rate(tmp_path: Path) -> None:
    """Before 23 Oct 2026 a rate reached the record only as PADDOCK_INFER_HZ in `model_env` (the September
    native-rate records). A run that recorded `infer_hz` is judged by that alone, so one record never
    says both that it was asked at the model's own rate and that it was off the protocol."""
    september = _pooled_with(tmp_path / "a", ({"model_env": {"PADDOCK_INFER_HZ": "2"}},) * 2)
    assert not september.valid and "driven at 2 Hz" in september.notes
    both = _pooled_with(tmp_path / "b", ({"model_env": {"PADDOCK_INFER_HZ": "2"}, "infer_hz": 2.0},) * 2)
    assert "driven at 2 Hz" not in both.notes and "2 Hz, the planning_hz" in both.notes


def test_a_pool_of_routes_ended_at_a_collision_does_not_rank(tmp_path: Path) -> None:
    rules = {"stuck_s": 30.0, "collision": True, "progress_m": 3.0}
    run = _pooled_with(tmp_path, ({"early_stop": rules}, {"early_stop": rules}))
    assert run.config["early_stop"] == rules and not run.valid
    assert "stop_on_collision" in run.notes


def test_a_pool_without_an_early_stop_records_none(tmp_path: Path) -> None:
    """No key in the config, so the config hash and run id of every earlier pool are unchanged."""
    ((run, _, samples),) = bench2drive.ingest(_tree(tmp_path), model="modelx", site="s")
    assert "early_stop" not in run.config
    assert not any("ended_stuck" in s["metrics"] for s in samples)


def test_routes_stopped_under_different_rules_do_not_rank(tmp_path: Path) -> None:
    """Two thresholds are two experiments; one mean over both describes neither."""
    root = _tree(tmp_path)
    for run_id, stuck_s in (("20260901T000000-aaaaaa", 30.0), ("20260903T000000-cccccc", 60.0)):
        rules = {"stuck_s": stuck_s, "progress_m": 3.0}
        (root / run_id / "run.json").write_text(
            json.dumps(_with_early_stop(_source_run_json("modelx", [], {}), rules))
        )

    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")

    assert run.config["early_stop"] == [
        {"progress_m": 3.0, "stuck_s": 30.0},
        {"progress_m": 3.0, "stuck_s": 60.0},
    ]
    assert not run.valid
    assert "early stop settings differ between source runs" in run.notes


def test_a_route_ended_by_hand_keeps_its_score_and_the_pool_does_not_rank(tmp_path: Path) -> None:
    """A stop request is an operator's choice, not a rule: the route stays, named; the row does not rank."""
    root = _tree(tmp_path)
    route_dir = root / "20260903T000000-cccccc" / "24333-Town15-DynamicObjectCrossing"
    (route_dir / "early_stop.json").write_text(json.dumps({"reason": "manual", "game_time_s": 12.0}))

    ((run, _metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert not run.valid
    assert "1 route(s) ended by hand" in run.notes
    by_route = {s["native_id"]["route_id"]: s["metrics"] for s in samples}
    assert by_route["24333"]["ended_manual"] == 1.0
