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

from paddock.results import store
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
    assert run.run_id == "pooled-" + store.sequence_hash([s["sample_id"] for s in samples])


def test_requested_route_without_a_measurement_marks_the_run_partial(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    routes = ["1711", "2802", "24333", "9999"]
    ((run, metrics, _),) = bench2drive.ingest(root, model="modelx", site="s", routes=routes)
    assert run.status == "partial" and not run.valid
    assert run.dataset == {
        "name": "bench2drive220",
        "n_samples": 3,
        "sequences_sha256": metrics.run_id.split("-", 1)[1],
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
