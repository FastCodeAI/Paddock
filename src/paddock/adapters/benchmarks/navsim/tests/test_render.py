"""Tests for the NAVSIM comparison figure's selection logic.

The drawing itself needs the devkit and the dataset, so it is not tested here.
What is tested is everything that decides *what* gets drawn -- which runs supply
plans, which scenarios are chosen, what a panel is captioned -- because those
are the parts that silently produce a misleading figure rather than an error.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from paddock.adapters.benchmarks.navsim import render


def _run(model: str, run_id: str = "r1", config: dict | None = None):
    return types.SimpleNamespace(
        model=model,
        benchmark="navsim",
        run_id=run_id,
        run={"config": config or {}},
        metrics={},
        samples=[],
    )


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_plans_come_from_the_dumped_scored_poses(tmp_path) -> None:
    cfg = types.SimpleNamespace(runs=tmp_path)
    _dump(
        tmp_path / "navsim/m/r1/chunks/000/trajectories.jsonl",
        [{"token": "t1", "log_name": "log-a", "navsim_poses": [[1.0, 0.0, 0.0]]}],
    )
    plans, logs = render._plans(_run("m"), cfg)
    assert plans == {"t1": [[1.0, 0.0, 0.0]]}
    assert logs == {"t1": "log-a"}


def test_a_dump_without_a_token_is_skipped(tmp_path) -> None:
    """Runs recorded before the proxy labelled its plans cannot be placed."""
    cfg = types.SimpleNamespace(runs=tmp_path)
    _dump(tmp_path / "navsim/m/r1/trajectories.jsonl", [{"navsim_poses": [[1.0, 0.0, 0.0]]}])
    plans, _ = render._plans(_run("m"), cfg)
    assert plans == {}


def test_a_resumed_run_reads_the_run_it_merged_from(tmp_path) -> None:
    """Its earlier chunks' plans live in the earlier run's directory, and the
    later run's plan -- the one that was scored -- wins the token."""
    cfg = types.SimpleNamespace(runs=tmp_path)
    _dump(
        tmp_path / "navsim/m/old/chunks/000/trajectories.jsonl",
        [
            {"token": "t1", "log_name": "log-a", "navsim_poses": [[1.0, 0.0, 0.0]]},
            {"token": "t2", "log_name": "log-b", "navsim_poses": [[2.0, 0.0, 0.0]]},
        ],
    )
    _dump(
        tmp_path / "navsim/m/new/chunks/001/trajectories.jsonl",
        [{"token": "t2", "log_name": "log-b", "navsim_poses": [[9.0, 0.0, 0.0]]}],
    )
    run = _run("m", "new", {"merged_from": "old"})
    plans, _ = render._plans(run, cfg)
    assert sorted(plans) == ["t1", "t2"]
    assert plans["t2"] == [[9.0, 0.0, 0.0]], "the scored run's plan must win"


def test_scenarios_rank_by_disagreement_and_spread_across_logs() -> None:
    plans = {
        "a": {"agree": [[10.0, 0.0]], "differ": [[10.0, 0.0]], "same_log": [[10.0, 0.0]]},
        "b": {"agree": [[10.0, 0.1]], "differ": [[10.0, 9.0]], "same_log": [[10.0, 8.0]]},
    }
    logs = {"agree": "log-a", "differ": "log-b", "same_log": "log-b"}
    assert render._scenarios(plans, logs, 2) == ["differ", "agree"]


def test_the_grid_is_filled_when_there_are_fewer_logs_than_panels() -> None:
    plans = {
        "a": {"x": [[10.0, 0.0]], "y": [[10.0, 0.0]]},
        "b": {"x": [[10.0, 5.0]], "y": [[10.0, 3.0]]},
    }
    logs = {"x": "one", "y": "one"}
    assert sorted(render._scenarios(plans, logs, 2)) == ["x", "y"]


def test_a_caption_names_only_the_models_on_the_panel() -> None:
    """A score for a model whose line is absent reads as a line not found."""
    payload = {"models": ["meteor"], "scores": {"t1": {"meteor": 0.5, "drivejepa": 0.9}}}
    assert render.caption(payload, "t1") == "t1   meteor 0.50"


def test_a_caption_without_scores_is_just_the_scenario() -> None:
    assert render.caption({"models": ["meteor"], "scores": {}}, "abcdef1234") == "abcdef12"


def test_written_paths_keeps_a_path_containing_a_space() -> None:
    """Splitting the child's output on whitespace would make this two paths
    that do not exist, and the caller would report that nothing was drawn."""
    output = "/runs/my figures/navsim_trajectories.png\n"
    assert render.written_paths(output) == [Path("/runs/my figures/navsim_trajectories.png")]


def test_written_paths_ignores_chatter_around_the_filenames() -> None:
    """The devkit's loader prints progress bars to the same stream."""
    output = "Loading logs: 100%|####| 136/136\n/runs/a.png\n  /runs/b.png  \ndone\n"
    assert render.written_paths(output) == [Path("/runs/a.png"), Path("/runs/b.png")]


def test_the_drawing_timeout_is_overridable(monkeypatch) -> None:
    monkeypatch.delenv(render.ENV_TIMEOUT, raising=False)
    assert render.render_timeout_s() == float(render.DEFAULT_TIMEOUT_S)
    monkeypatch.setenv(render.ENV_TIMEOUT, "45")
    assert render.render_timeout_s() == 45.0
    for bad in ("0", "-1", "soon"):
        monkeypatch.setenv(render.ENV_TIMEOUT, bad)
        assert render.render_timeout_s() == float(render.DEFAULT_TIMEOUT_S)
