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

from paddock.config import ConfigError, SiteConfig
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
    """The 11715 case: every attempt crashed, and `routes` is unset.

    Before this was fixed the expected set was built from the samples that
    survived, so `missing` was empty by construction: the route left no trace in
    `routes_subset`, `n_failed` stayed 0, the run claimed `complete`/`valid`, and
    the mean silently came from a smaller route set.
    """
    root = _tree(tmp_path)
    # Route 11715 in this repo's real trees: several attempts, every one of them
    # with an empty `_checkpoint.records`.
    for run in ("20260904T000000-dddddd", "20260905T000000-eeeeee"):
        _write(root, run, "11715-Town12-EnterActorFlow", _unscored("11715"))

    ((run, metrics, samples),) = bench2drive.ingest(root, model="modelx", site="s")

    assert "11715" not in {s["native_id"]["route_id"] for s in samples}, "it did not score"
    assert "11715" in run.config["routes_subset"], "but it was attempted, so it was expected"
    assert run.dataset["n_expected"] == 4 and run.dataset["n_failed"] == 1
    assert run.status == "partial" and not run.valid
    assert "11715" in run.notes and "did not score" in run.notes
    assert metrics.n == 3, "the three routes that scored are what the mean is over"


def test_the_rejection_reason_names_the_run_and_no_machine_path(tmp_path: Path) -> None:
    """`notes` is tracked, so the reason may name a run id but never a disk."""
    root = _tree(tmp_path)
    _write(root, "20260904T000000-dddddd", "11715-Town12-EnterActorFlow", _unscored("11715"))
    ((run, _, _),) = bench2drive.ingest(root, model="modelx", site="s")
    assert "20260904T000000-dddddd" in run.notes
    assert str(tmp_path) not in run.notes and "/" not in run.notes.split("11715", 1)[1]


def test_an_unreadable_results_file_is_a_rejection_not_a_crash(tmp_path: Path) -> None:
    """One truncated file on a network mount used to abort the whole walk."""
    root = _tree(tmp_path)
    truncated = root / "20260904T000000-dddddd" / "11715-Town12-EnterActorFlow" / "results.json"
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
    assert "kept no run.json" in run.notes


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


# --- the site config is read by a stranger, so it fails by name --------------


def _site(tmp_path: Path, toml: str) -> SiteConfig:
    source = tmp_path / "paddock.toml"
    source.write_text(toml)
    return SiteConfig(
        site="s",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={},
        models={},
        benchmarks={},
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
