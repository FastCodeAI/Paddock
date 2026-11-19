"""The website's nuScenes videos: which keyframes are picked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from paddock.adapters.benchmarks.nuscenes import render as R
from paddock.policy.protocol import Command
from paddock.render import LoadedRun


def _run(model: str, errors: dict[tuple[str, int, Command], float], floor: bool = False) -> LoadedRun:
    samples = [
        {
            "sample_id": f"nuscenes/val/{scene}/{index}",
            "native_id": {"scene": scene, "index_in_scene": index, "command": int(command)},
            "metrics": {R.SCORE: error},
            "prediction": [[1.0, 0.0]],
        }
        for (scene, index, command), error in errors.items()
    ]
    run = LoadedRun(model, "nuscenes", f"{model}-run", {}, {}, samples)
    run.floor = floor  # what tools/site/videos.py sets on the table's floor row
    return run


LEFT, RIGHT, STRAIGHT = Command.LEFT, Command.RIGHT, Command.STRAIGHT


def test_showcase_skips_a_scenes_first_two_seconds() -> None:
    errors = {
        ("scene-0001", 0, LEFT): 9.0,  # a first keyframe: the models have no past yet
        ("scene-0002", 3, LEFT): 8.0,  # still inside the first 2 s
        ("scene-0003", 4, LEFT): 2.0,
        ("scene-0004", 10, LEFT): 1.0,
    }
    runs = [_run("a", errors), _run("b", errors)]
    picks = R.showcase(runs, n=1)
    assert picks[0]["sample"] == "nuscenes/val/scene-0003/4"
    assert picks[0]["title"] == "Left turn, scene-0003"
    # The clip starts at the scene's first keyframe, 0.5 s apart.
    assert picks[0]["caption"].startswith(
        "The hardest left turn, at 2.0 s: on average the models' plans are 2.0 m"
    )


def test_showcase_leaves_the_floor_out_of_the_picking() -> None:
    errors = {("scene-0010", 5, RIGHT): 1.0, ("scene-0011", 5, RIGHT): 2.0}
    # Constant velocity is far off on scene-0010, which would make it the hardest.
    floor = _run(
        "constant_velocity", {("scene-0010", 5, RIGHT): 30.0, ("scene-0011", 5, RIGHT): 0.5}, floor=True
    )
    picks = R.showcase([_run("a", errors), _run("b", errors), floor], n=1)
    assert picks[0]["sample"] == "nuscenes/val/scene-0011/5"


def test_showcase_one_of_each_kind_then_the_widest_spread() -> None:
    a = {("scene-0001", 5, LEFT): 1.0, ("scene-0002", 5, RIGHT): 1.0, ("scene-0003", 5, STRAIGHT): 1.0}
    a[("scene-0004", 5, STRAIGHT)] = 0.1
    b = dict(a)
    b[("scene-0004", 5, STRAIGHT)] = 4.1
    picks = R.showcase([_run("a", a), _run("b", b)], n=4)
    assert [p["sample"].split("/")[2] for p in picks] == [
        "scene-0001",
        "scene-0002",
        "scene-0004",
        "scene-0003",
    ]
    assert picks[3]["caption"].startswith("Where the models disagree most")


def test_showcase_says_which_runs_it_looked_in() -> None:
    with pytest.raises(LookupError, match="a/a-run"):
        R.showcase([_run("a", {})], n=1)


def test_render_sample_names_the_shape_of_an_id() -> None:
    with pytest.raises(LookupError, match="nuscenes/val/scene-0103/12"):
        R.render_sample([], "nuscenes/scene-0103", "out", SimpleNamespace())
