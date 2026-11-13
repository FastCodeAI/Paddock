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

import numpy as np

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


# ---------------------------------------------------------------------------
# videos for the website
# ---------------------------------------------------------------------------


def _scored(model: str, cfg, rows: dict[str, tuple[float, list, list]], floor: bool = False):
    """A run with dumps and records. `rows`: token -> (EPDMS, driving command, end point)."""
    _dump(
        cfg.runs / f"navsim/{model}/r1/chunks/000/trajectories.jsonl",
        [
            {
                "token": token,
                "log_name": f"log-{token}",
                "driving_command": command,
                "navsim_poses": [[end[0] / 2, end[1] / 2, 0.0], [end[0], end[1], 0.0]],
            }
            for token, (_, command, end) in rows.items()
        ],
    )
    run = _run(model)
    run.floor = floor
    # As the scorer's CSV writes them: Python reprs, not JSON.
    run.samples = [
        {"sample_id": f"navsim/navtest/{t}", "native_id": repr({"token": t}), "metrics": repr({"epdms": s})}
        for t, (s, _, _) in rows.items()
    ]
    return run


LEFT, STRAIGHT = [1, 0, 0, 0], [0, 1, 0, 0]


def _one_frame_logs(monkeypatch) -> None:
    """Every scenario in a log of its own, and one frame enough to be picked."""
    monkeypatch.setattr(render, "_order", lambda cfg, log: [log.removeprefix("log-")])
    monkeypatch.setattr(render, "MIN_FRAMES", 1)


def test_showcase_picks_each_kinds_widest_split_breaking_ties_by_where_plans_end(
    tmp_path, monkeypatch
) -> None:
    _one_frame_logs(monkeypatch)
    cfg = types.SimpleNamespace(runs=tmp_path)
    a = _scored(
        "a", cfg, {"t1": (1.0, LEFT, [10, 5]), "t2": (1.0, LEFT, [10, 5]), "t3": (0.0, LEFT, [10, 5])}
    )
    # t1 and t2 split the same way; b's plan ends further from a's on t2.
    b = _scored(
        "b", cfg, {"t1": (0.0, LEFT, [10, 4]), "t2": (0.0, LEFT, [10, -9]), "t3": (0.0, LEFT, [10, 5])}
    )
    picks = render.showcase([a, b], n=1, cfg=cfg)
    assert picks == [
        {
            "sample": "navsim/navtest/t2",
            "title": "Left turn, scenario t2",
            "caption": "The left turn the models split on most, at 0.0 s: "
            "1 of 2 score above 0.9 and 1 score below 0.1.",
        }
    ]


def test_showcase_leaves_the_floor_out_of_the_picking(tmp_path, monkeypatch) -> None:
    _one_frame_logs(monkeypatch)
    cfg = types.SimpleNamespace(runs=tmp_path)
    a = _scored("a", cfg, {"t1": (0.5, STRAIGHT, [10, 0]), "t2": (0.5, STRAIGHT, [10, 0])})
    b = _scored("b", cfg, {"t1": (0.5, STRAIGHT, [10, 0]), "t2": (0.4, STRAIGHT, [10, 0])})
    floor = _scored("cv", cfg, {"t1": (0.0, STRAIGHT, [10, 0]), "t2": (1.0, STRAIGHT, [10, 0])}, floor=True)
    assert render.showcase([a, b, floor], n=1, cfg=cfg)[0]["sample"] == "navsim/navtest/t2"


def test_a_run_whose_dumps_name_no_scenario_cannot_be_drawn(tmp_path) -> None:
    cfg = types.SimpleNamespace(runs=tmp_path)
    _dump(tmp_path / "navsim/old/r1/trajectories.jsonl", [{"navsim_poses": [[1.0, 0.0, 0.0]]}])
    assert not render.drawable(_run("old"), cfg)
    assert render.drawable(_scored("new", cfg, {"t1": (1.0, LEFT, [5, 0])}), cfg)


def test_the_recorded_path_is_in_the_scored_frames_ego_frame() -> None:
    import math

    def frame(x, y, heading):
        half = heading / 2
        return {
            "ego2global_translation": [x, y, 0.0],
            "ego2global_rotation": [math.cos(half), 0.0, 0.0, math.sin(half)],  # w, x, y, z
        }

    # The car faces north (+y in the world) and drives on north, then 2 m west (its left).
    frames = [
        frame(100.0, 200.0, math.pi / 2),
        frame(100.0, 210.0, math.pi / 2),
        frame(98.0, 220.0, math.pi / 2),
    ]
    recorded = render._recorded(frames, 0, 8)
    assert np.allclose(recorded, [[10.0, 0.0], [20.0, 2.0]])
    assert render._recorded(frames, 2, 8) is None


def test_a_scenario_is_picked_only_inside_a_long_enough_run_of_scored_frames(monkeypatch) -> None:
    # One log: frames 0-9 all scored, a gap at 10, then 11-13.
    order = [f"f{k}" for k in range(14)]
    monkeypatch.setattr(render, "_order", lambda cfg, log: order)
    tokens = {f"f{k}": f"navsim/navtest/f{k}" for k in [*range(10), 11, 12, 13]}
    starts = render._starts(None, {"log": tokens})
    assert set(starts) == {f"navsim/navtest/f{k}" for k in range(10)}  # 11-13 is 1.5 s, under MIN_FRAMES
    # The clip plays from the run's start, so the pick is that far in.
    assert starts["navsim/navtest/f6"] == 6 * render.FRAME_S


def test_a_clip_is_capped_before_and_after_its_scenario() -> None:
    always = lambda k: True  # noqa: E731
    assert render._stretch(always, 50, 100) == (50 - render.BEFORE, 51 + render.AFTER)
    assert render._stretch(always, 2, 5) == (0, 5)
    assert render._stretch(lambda k: k != 48, 50, 100) == (49, 51 + render.AFTER)
