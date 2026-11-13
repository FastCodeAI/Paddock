"""The website's KITScenes videos: which keyframes are picked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from paddock.adapters.benchmarks.kitscenes import render as R
from paddock.render import LoadedRun


def _path(lateral_at_3s: float) -> list[list[float]]:
    """Eight poses 0.5 s apart; the sixth, at 3 s, is `lateral_at_3s` to the left."""
    return [[5.0 * (i + 1), lateral_at_3s if i == 5 else 0.0, 0.0] for i in range(8)]


def _run(model: str, rows: dict[str, tuple[float, float]], floor: bool = False) -> LoadedRun:
    """`rows`: scene -> (ADE, where the recorded path is at 3 s, sideways)."""
    samples = [
        {
            "sample_id": f"kitscenes/val/{scene}/40",
            "native_id": {"scene_id": scene, "keyframe_index": 40},
            "metrics": {R.SCORE: ade},
            "prediction": [[5.0, 0.0]],
            "ground_truth": _path(lateral),
        }
        for scene, (ade, lateral) in rows.items()
    ]
    run = LoadedRun(model, "kitscenes", f"{model}-run", {}, {}, samples)
    run.floor = floor
    return run


def test_showcase_reads_the_kind_off_the_recorded_path() -> None:
    rows = {
        "aaaaaaaa-1": (1.0, 3.0),  # left
        "bbbbbbbb-2": (2.0, -3.0),  # right
        "cccccccc-3": (3.0, 0.5),  # straight on
    }
    picks = R.showcase([_run("a", rows), _run("b", rows)], n=3)
    assert [p["title"] for p in picks] == [
        "Left turn, scene aaaaaaaa",
        "Right turn, scene bbbbbbbb",
        "Straight on, scene cccccccc",
    ]


def test_showcase_leaves_the_floor_out_of_the_picking() -> None:
    rows = {"aaaaaaaa-1": (1.0, 3.0), "bbbbbbbb-2": (2.0, 3.0)}
    floor = _run("constant_velocity", {"aaaaaaaa-1": (40.0, 3.0), "bbbbbbbb-2": (0.1, 3.0)}, floor=True)
    picks = R.showcase([_run("a", rows), floor], n=1)
    assert picks[0]["sample"] == "kitscenes/val/bbbbbbbb-2/40"


def test_showcase_says_which_runs_it_looked_in() -> None:
    with pytest.raises(LookupError, match="a/a-run"):
        R.showcase([_run("a", {})], n=1)


def test_render_sample_names_the_shape_of_an_id() -> None:
    with pytest.raises(LookupError, match="kitscenes/val/<scene>/40"):
        R.render_sample([], "kitscenes/scene", "out", SimpleNamespace())


def test_a_held_plan_stays_where_it_lies_on_the_road() -> None:
    import numpy as np

    plan = [[5.0, 0.0], [10.0, 1.0]]
    # The car has driven 5 m on, still facing the same way: the plan is now 5 m nearer.
    held = R._held(
        plan, np.array([100.0, 50.0, 0.3]), np.array([100.0 + 5 * np.cos(0.3), 50.0 + 5 * np.sin(0.3), 0.3])
    )
    assert np.allclose(held, [[-5.0, 0.0], [0.0, 0.0], [5.0, 1.0]])
    # Unmoved, it is the plan itself, starting where the car is.
    assert np.allclose(R._held(plan, np.zeros(3), np.zeros(3)), [[0.0, 0.0], *plan])


def test_a_caption_says_when_in_the_clip_the_moment_is() -> None:
    pick = {"why": "disagree", "kind": "straight", "mean": 2.0, "low": 0.2, "high": 11.9}
    assert R._caption(pick, 13.0).startswith("Where the models disagree most, at 13.0 s:")
