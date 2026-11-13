"""viz.moment: the rule that picks the open-loop moments, and how one is drawn."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from paddock.viz import moment, video


def test_choose_takes_the_hardest_of_each_kind_then_the_widest_spread() -> None:
    scores = {
        "l1": {"a": 3.0, "b": 3.0},  # the hardest left turn
        "l2": {"a": 1.0, "b": 1.0},
        "r1": {"a": 2.0, "b": 2.0},  # the only right turn
        "s1": {"a": 0.5, "b": 0.5},
        "s2": {"a": 0.1, "b": 5.1},  # the widest spread, and the hardest straight one
        "s3": {"a": 0.2, "b": 4.2},  # the widest spread once s2 is taken
    }
    kinds = {"l1": "left", "l2": "left", "r1": "right", "s1": "straight", "s2": "straight", "s3": "straight"}
    picks = moment.choose(scores, kinds, {}, 4, higher_is_better=False)
    assert [(p["sample"], p["why"], p["of_kind"]) for p in picks] == [
        ("l1", "hardest", True),
        ("r1", "hardest", True),
        ("s2", "hardest", True),
        ("s3", "disagree", False),
    ]
    assert picks[0]["scores"] == {"a": 3.0, "b": 3.0}
    assert (picks[3]["low"], picks[3]["high"]) == (0.2, 4.2)


def test_choose_shows_no_place_twice_and_only_moments_every_model_scored() -> None:
    scores = {
        "x1": {"a": 9.0},  # b never scored it, so it is never shown
        "p1": {"a": 3.0, "b": 3.0},
        "p2": {"a": 2.0, "b": 2.0},  # same place as p1
        "q1": {"a": 1.0, "b": 1.0},
    }
    kinds = dict.fromkeys(scores, "left")
    places = {"x1": "x", "p1": "p", "p2": "p", "q1": "q"}
    picks = moment.choose(scores, kinds, places, 4, higher_is_better=False)
    assert [p["sample"] for p in picks] == ["p1", "q1"]


def test_higher_is_better_flips_hardest() -> None:
    scores = {"a": {"m": 0.9, "n": 0.9}, "b": {"m": 0.1, "n": 0.1}}
    picks = moment.choose(scores, dict.fromkeys(scores, "left"), {}, 1, higher_is_better=True)
    assert picks[0]["sample"] == "b"


def test_first_disagree_picks_each_kind_by_spread_and_breaks_ties_as_asked() -> None:
    # Scores of 0 or 1: t1 and t2 split the same way, so only the tie-break tells them apart.
    scores = {
        "t1": {"a": 1.0, "b": 0.0},
        "t2": {"a": 1.0, "b": 0.0},
        "t3": {"a": 0.0, "b": 0.0},
    }
    kinds = dict.fromkeys(scores, "left")
    picks = moment.choose(
        scores, kinds, {}, 1, higher_is_better=True, first="disagree", tiebreak={"t1": 2.0, "t2": 9.0}
    )
    assert [(p["sample"], p["why"], p["of_kind"]) for p in picks] == [("t2", "disagree", True)]
    with pytest.raises(ValueError, match="first="):
        moment.choose(scores, kinds, {}, 1, higher_is_better=True, first="worst")


def test_kind_from_path_reads_the_lateral_offset_at_three_seconds() -> None:
    def path(y):
        return [[5.0 * i, y if i == 5 else 0.0] for i in range(6)]

    assert moment.kind_from_path(path(2.5)) == "left"
    assert moment.kind_from_path(path(-2.5)) == "right"
    assert moment.kind_from_path(path(1.0)) == "straight"


def _forward_camera(height: float = 1.5) -> moment.Camera:
    """A pinhole camera looking straight ahead from `height` above the road."""
    cam2ego = np.eye(4)
    # OpenCV x right, y down, z forward -> ego x forward, y left, z up.
    cam2ego[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cam2ego[2, 3] = height
    return moment.Camera(np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]]), cam2ego)


def test_a_straight_path_runs_up_the_middle_below_the_horizon() -> None:
    camera = _forward_camera()
    (line,) = moment.project([[10.0, 0.0], [20.0, 0.0]], camera)
    assert np.allclose(line[:, 0], 640)  # dead ahead is the principal point's column
    assert (line[:, 1] > 360).all()  # the road is below the horizon
    assert line[-1, 1] < line[0, 1]  # and further away is higher up
    # 20 m ahead, 1.5 m down: 1000 * 1.5 / 20 = 75 px below the horizon.
    assert moment.end_point([[20.0, 0.0]], camera) == pytest.approx([640, 435])
    # Left in the ego frame is left in the picture.
    assert moment.end_point([[20.0, 2.0]], camera)[0] < 640


def test_a_path_behind_the_camera_draws_nothing() -> None:
    camera = _forward_camera()
    assert moment.project([[-5.0, 0.0], [-10.0, 0.0]], camera) == []
    assert moment.end_point([[-10.0, 0.0]], camera) is None


def test_a_plan_too_short_to_see_says_so() -> None:
    assert (
        moment._note(moment.Plan("a", "#000000", [[0.1, 0.0], [0.2, 0.0]], "6.02 m"))
        == "barely moves · 6.02 m"
    )
    assert moment._note(moment.Plan("a", "#000000", [[5.0, 0.0], [10.0, 0.0]], "0.3 m")) == "0.3 m"


FFMPEG = video.find_ffmpeg()


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
def test_write_plays_every_keyframe_with_its_plans_in_order(tmp_path: Path) -> None:
    from PIL import Image

    steps = []
    for k in range(3):
        path = tmp_path / f"{k}.jpg"
        Image.new("RGB", (640, 360), (60 * k, 60 * k, 60 * k)).save(path)
        plans = [moment.Plan("a", "#d62728", np.array([[5.0, 0.0], [10.0, 0.5 * k]]), f"{k}.0 m")]
        steps.append(
            moment.Step(
                path, _forward_camera(), plans, np.array([[5.0, 0.0]]), 0.5 * k, "picked" if k == 1 else ""
            )
        )
    written = moment.write(
        tmp_path / "out" / "clip", steps, step_s=0.5, title="test", score_name="error", ffmpeg=FFMPEG
    )
    assert [p.name for p in written] == ["clip.mp4", "clip.jpg"]
    raw = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(written[0]),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            "96x38",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 38, 96)
    # Three keyframes, each held 0.5 s at 10 frames a second, and nothing else.
    assert len(frames) == 3 * round(0.5 * moment.FPS)
    # In order: the camera view (left two thirds) brightens keyframe by keyframe.
    brightness = [frames[i * 5 + 2, :30, :60].mean() for i in range(3)]
    assert brightness == sorted(brightness)
    assert Image.open(written[1]).size == (moment.WIDTH, moment.HEIGHT)


def test_write_needs_a_keyframe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no keyframes"):
        moment.write(tmp_path / "clip", [], step_s=0.5, title="t", score_name="s", ffmpeg="ffmpeg")


def test_the_view_from_above_keeps_one_scale_for_the_whole_clip() -> None:
    near, far, half = moment._extent([np.array([[5.0, 0.0], [40.0, 2.0]]), np.array([[3.0, -12.0]])])
    assert (near, far, half) == (-2.0, 43.0, 15.0)


def test_a_path_held_on_the_road_starts_at_its_own_first_point() -> None:
    camera = _forward_camera()
    ahead = [[5.0, 0.0], [15.0, 0.0]]
    # Held on the road, the line starts 5 m ahead: 1000 * 1.5 / 5 = 300 px below the horizon.
    (held,) = moment.project(ahead, camera, from_car=False)
    assert held[0, 1] == pytest.approx(360 + 300)
    # From the car, the same plan is joined to the car, so its line starts much nearer.
    (joined,) = moment.project(ahead, camera)
    assert joined[0, 1] > held[0, 1] + 100
    # A held plan that never left its start barely moves, wherever the car is now.
    still = [[-5.0, 0.0], [-4.0, 0.0]]
    assert moment._note(moment.Plan("a", "#000000", still, "1 m"), from_car=False) == "barely moves · 1 m"
    assert moment._note(moment.Plan("a", "#000000", ahead, "1 m"), from_car=False) == "1 m"
