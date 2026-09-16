"""The Bench2Drive ingest pools recorded route results without copying a report.

Pure: a synthetic results tree in ``tmp_path`` stands in for the site's
recordings. What it must get right: per route the latest attempt that scored
wins, a crashed attempt never shadows its rerun, a requested route with no
valid attempt marks the run partial, the aggregate is recomputed from the
per-route records, and nothing written into ``results/`` names this machine.
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
