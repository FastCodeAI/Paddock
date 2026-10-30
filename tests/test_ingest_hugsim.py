"""The HUGSIM ingest reads a recorded campaign without copying a report.

Pure: a synthetic campaign in ``tmp_path`` stands in for the site's recordings. What it must get right: scores
come from each scenario's ``eval.json`` and must agree with ``results.jsonl``; a scenario attempted and never
scored marks the run partial and is named; the release's own broken scenarios are left out for every model;
the protocol (controller fix, call rate, cameras) is read from the recordings; and nothing written names this
machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddock.results import store
from paddock.results.ingest import hugsim
from paddock.results.ingest.hugsim import CONTROLLER_NOTE, call_interval

NOTE = (
    f"{CONTROLLER_NOTE} traj2control heading, not on main)\n"
    "NOTE: --render_cams ['CAM_FRONT']: other cameras returned black\n"
)


def _scenario(
    root: Path, scenario: str, ds: str, scene: str, mode: str, scores: dict, steps: int, ending: str
) -> dict:
    run_dir = root / f"{ds}_bridge" / f"{scene}_{mode}"
    run_dir.mkdir(parents=True)
    (run_dir / "eval.json").write_text(json.dumps({**scores, "details": {}}))
    (run_dir / "agent_log.jsonl").write_text(
        "".join(json.dumps({"step": i, "called": i % 2 == 0}) + "\n" for i in range(steps))
    )
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / f"{scenario}.log").write_text(f"{NOTE}Ready for simulation\n{ending}\n")
    return {
        "scenario": scenario,
        "dataset": ds,
        "scene": scene,
        "mode": mode,
        "difficulty": mode.split("_")[0],
        "render_cams": "CAM_FRONT",
        "exit": 0,
        "run_dir": str(run_dir),
        **scores,
        "steps": steps,
    }


def _scores(hd: float, pdms: float, rc: float) -> dict:
    return {"nc": 1.0, "dac": 1.0, "ttc": 1.0, "c": 1.0, "pdms": pdms, "rc": rc, "hdscore": hd}


def _campaign(tmp_path: Path) -> Path:
    root = tmp_path / "hugsim-eval" / "modelx"
    root.mkdir(parents=True)
    rows = [
        _scenario(
            root,
            "scene-0013-easy-00",
            "nuscenes",
            "scene-0013",
            "easy_00",
            _scores(0.9, 0.9, 1.0),
            120,
            "Complete",
        ),
        _scenario(
            root,
            "scene-021-hard-00",
            "pandaset",
            "021",
            "hard_00",
            _scores(0.1, 0.5, 0.2),
            30,
            "Collision with foreground",
        ),
        _scenario(
            root,
            "scene-163453191685-medium-00",
            "waymo",
            "163453191685_0_200",
            "medium_00",
            _scores(0.2, 1.0, 0.2),
            401,
            "simulation finished in 50.0 s",
        ),
    ]
    # attempted twice, crashed both times
    (root / "logs" / "scene-0166-medium-01.log").write_text(
        "Traceback (most recent call last):\nIndexError: index is out of bounds for dimension with size 0\n"
    )
    rows += [
        {
            "scenario": "scene-0166-medium-01",
            "dataset": "nuscenes",
            "scene": "scene-0166",
            "mode": "medium_01",
            "exit": 1,
            "steps": 60,
        }
    ] * 2
    # a release bug: fails for every model and is not part of the benchmark
    rows.append(
        {
            "scenario": "scene-021-medium-02",
            "dataset": "pandaset",
            "scene": "021",
            "mode": "medium_02",
            "exit": 1,
        }
    )
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_ingest_reads_the_campaign(tmp_path: Path) -> None:
    [(run, metrics, samples)] = hugsim.ingest(
        _campaign(tmp_path), model="modelx", site="test-site", run_label="campaign"
    )
    assert run.benchmark == "hugsim" and run.model == "modelx"
    assert [s["native_id"]["scenario"] for s in samples] == [
        "scene-0013-easy-00",
        "scene-021-hard-00",
        "scene-163453191685-medium-00",
    ]
    assert run.dataset["n_samples"] == 3 and run.dataset["n_expected"] == 4 and run.dataset["n_failed"] == 1
    assert run.status == "partial" and run.valid is False
    assert "scene-0166-medium-01" in run.notes and "IndexError" in run.notes
    assert "scene-021-medium-02" not in run.notes  # excluded, not failed
    assert run.config["excluded_attempted"] == 1

    assert run.config["infer_every"] == 2 and run.config["infer_hz"] == 2.0
    assert run.config["render_cams"] == "CAM_FRONT"
    assert run.config["controller"] == "traj2control with upstream PR #57"

    assert metrics.primary_metric == "hd_score"
    assert metrics.metrics["hd_score"]["mean"] == pytest.approx((0.9 + 0.1 + 0.2) / 3)
    assert metrics.metrics["route_completion"]["mean"] == pytest.approx(100 * (1.0 + 0.2 + 0.2) / 3)
    assert metrics.metrics["collision_rate"]["mean"] == pytest.approx(1 / 3)
    assert set(metrics.by_split) == {"nuscenes", "pandaset", "waymo"}
    assert set(metrics.by_group) == {"easy", "hard", "medium"}
    endings = {s["native_id"]["scenario"]: s["extra"]["ending"] for s in samples}
    assert endings == {
        "scene-0013-easy-00": "completed",
        "scene-021-hard-00": "hit a vehicle",
        "scene-163453191685-medium-00": "step cap",
    }
    assert all(len(s["native_id"]["sha256"]) == 64 for s in samples)


def test_a_complete_campaign_is_valid(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["scenario"] != "scene-0166-medium-01"]
    (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    [(run, _, _)] = hugsim.ingest(root, model="modelx", site="test-site")
    assert run.status == "complete" and run.valid is True


def test_results_line_and_eval_json_must_agree(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    (root / "nuscenes_bridge" / "scene-0013_easy_00" / "eval.json").write_text(
        json.dumps({**_scores(0.5, 0.9, 1.0), "details": {}})
    )
    [(run, _, samples)] = hugsim.ingest(root, model="modelx", site="test-site")
    assert "scene-0013-easy-00" not in {s["native_id"]["scenario"] for s in samples}
    assert "results.jsonl says hdscore" in run.notes and run.valid is False


def test_a_missing_controller_note_does_not_rank(tmp_path: Path) -> None:
    root = _campaign(tmp_path)
    log = root / "logs" / "scene-021-hard-00.log"
    log.write_text(log.read_text().replace(CONTROLLER_NOTE, "NOTE: something else"))
    [(run, _, _)] = hugsim.ingest(root, model="modelx", site="test-site")
    assert run.config["controller"].startswith("PR #57 announced on 2 of 3")
    assert "does not rank" in run.notes


def test_model_env_is_the_operators_statement(tmp_path: Path) -> None:
    [(run, _, _)] = hugsim.ingest(
        _campaign(tmp_path), model="modelx", site="test-site", model_env={"MODELX_SWITCH": "off"}
    )
    assert run.config["model_env"] == {"MODELX_SWITCH": "off"}
    assert "operator's statement" in run.notes


def test_call_interval() -> None:
    assert call_interval([(400, 200, 0), (21, 11, 0), (1, 1, 0)]) == 2
    assert call_interval([(10, 10, 0), (5, 5, 0)]) == 1
    assert call_interval([(10, 10, 0), (10, 5, 0)]) is None
    assert call_interval([(10, 3, 1)]) is None  # a bridge that failed a step stopped counting early


def test_nothing_written_names_this_machine(tmp_path: Path) -> None:
    [(run, metrics, samples)] = hugsim.ingest(_campaign(tmp_path), model="modelx", site="test-site")
    out = tmp_path / "results"
    store.write_run(run, metrics, samples, root=out)
    for path in out.rglob("*.json"):
        assert str(tmp_path) not in path.read_text(), path
