"""The website's AlpaSim videos: which scenes are picked, and how one is drawn."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.alpasim import render as R
from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
from paddock.adapters.benchmarks.neuroncap.render import _decode, _encode
from paddock.render import LoadedRun


def _scene(tag: str) -> str:
    return f"clipgt-{tag}-0000-0000-0000-000000000000"


def _metrics(score: float, scored_s: float = 16.0, **extra) -> dict:
    return {"scene_score": score, "duration_frac_20s": scored_s / 20, **extra}


def _run(model: str, scenes: dict[str, dict]) -> LoadedRun:
    samples = [
        {"sample_id": f"alpasim/{_scene(tag)}", "native_id": {"scene": _scene(tag)}, "metrics": metrics}
        for tag, metrics in scenes.items()
    ]
    return LoadedRun(model, "alpasim", f"{model}-run", {}, {}, samples)


def _picked(runs, n: int = 4) -> list[str]:
    return [p["sample"].removeprefix("alpasim/clipgt-")[:8] for p in R.showcase(runs, n)]


def test_render_draws_the_scores_so_paddock_render_has_something_to_call(tmp_path: Path) -> None:
    """`paddock render --bench alpasim` calls `render`; without one it ended in an AttributeError."""
    pytest.importorskip("matplotlib")
    from paddock import render as rendering

    runs = [_run(m, {"aaaaaaa1": _metrics(0.2 * i), "aaaaaaa2": _metrics(1.0)}) for i, m in enumerate("ab")]
    written = R.render(runs, tmp_path, SimpleNamespace(runs=tmp_path))
    assert [p.name for p in written] == ["scores_by_scene.png"] and written[0].stat().st_size > 0
    assert R.render([_run("a", {})], tmp_path, SimpleNamespace(runs=tmp_path)) == []
    # Every hook `paddock render` reaches for is there.
    assert all(callable(getattr(R, hook)) for hook in ("render", "render_sample", "showcase"))
    assert rendering._adapter("alpasim", lambda _: None) is R


def test_showcase_picks_the_two_hardest_then_the_two_most_split() -> None:
    scores = {  # scene: one score per model
        "aaaaaaa1": [0.0, 0.0, 0.2],  # average 0.07: the hardest
        "aaaaaaa2": [0.0, 0.3, 0.3],  # average 0.20: the second hardest
        "aaaaaaa3": [0.0, 1.0, 0.0],  # spread 0.47: the most split of the rest
        "aaaaaaa4": [1.0, 0.0, 1.0],  # the same spread, but easier: next
        "aaaaaaa5": [0.5, 0.5, 0.6],  # average 0.53, spread 0.05: neither
        "aaaaaaa6": [1.0, 1.0, 1.0],
    }
    runs = [_run(m, {tag: _metrics(v[i]) for tag, v in scores.items()}) for i, m in enumerate("abc")]
    picks = R.showcase(runs)
    assert _picked(runs) == ["aaaaaaa1", "aaaaaaa2", "aaaaaaa3", "aaaaaaa4"]
    assert picks[0] == {
        "sample": f"alpasim/{_scene('aaaaaaa1')}",
        "title": "Scene aaaaaaa1",
        "caption": "The hardest scene: the models average 0.07 of 1. One drive per model.",
    }
    assert picks[1]["caption"] == (
        "The second hardest scene: the models average 0.20 of 1. One drive per model."
    )
    assert picks[2]["caption"] == (
        "Where the models split most: 1 of 3 score above 0.9 and 2 score below 0.1. One drive per model."
    )
    assert picks[3]["caption"].startswith("Where the models split next most: 2 of 3 score above 0.9")
    assert all(p["caption"].endswith("One drive per model.") for p in picks)
    # No scene twice: aaaaaaa1 has the third-widest spread and is already shown as the hardest.
    assert len(set(_picked(runs, 6))) == 6


def test_showcase_skips_a_scene_with_nothing_to_watch() -> None:
    def runs(scored: dict[str, list[float]]) -> list[LoadedRun]:
        rest = {t: _metrics(0.5 + 0.1 * k) for k, t in enumerate(["bbbbbbb1", "bbbbbbb2", "bbbbbbb3"])}
        return [
            _run(m, {**{tag: _metrics(0.0, s[i]) for tag, s in scored.items()}, **rest})
            for i, m in enumerate("abc")
        ]

    # Every model is out within a second: skipped, though it is the hardest by score.
    assert "aaaaaaa1" not in _picked(runs({"aaaaaaa1": [0.0, 0.5, 0.9]}))
    # One model is scored for a full second, but only two for 5 seconds or more: skipped.
    assert "aaaaaaa1" not in _picked(runs({"aaaaaaa1": [0.2, 5.0, 9.0]}))
    # Three models are scored for 5 seconds or more: kept, and the hardest.
    assert _picked(runs({"aaaaaaa1": [5.0, 5.0, 9.0]}))[0] == "aaaaaaa1"


def test_showcase_breaks_ties_by_the_harder_scene_then_the_lower_id() -> None:
    scores = {
        "cccccccc": [0.2, 0.2, 0.2],  # the same average as the next two: the lower id goes first
        "bbbbbbbb": [0.2, 0.2, 0.2],
        "aaaaaaaa": [0.2, 0.2, 0.2],
        "dddddddd": [0.0, 1.0, 0.5],  # the same spread as eeeeeeee, and as hard: the lower id
        "eeeeeeee": [1.0, 0.0, 0.5],
        "ffffffff": [0.1, 0.9, 0.5],  # a narrower spread
    }
    runs = [_run(m, {tag: _metrics(v[i]) for tag, v in scores.items()}) for i, m in enumerate("abc")]
    assert _picked(runs) == ["aaaaaaaa", "bbbbbbbb", "dddddddd", "eeeeeeee"]


def test_showcase_considers_only_scenes_every_run_scored() -> None:
    a = _run("a", {"aaaaaaa1": _metrics(0.0), "aaaaaaa2": _metrics(0.5)})
    b = _run("b", {"aaaaaaa1": _metrics(0.0), "aaaaaaa2": _metrics(0.5)})
    c = _run("c", {"aaaaaaa2": _metrics(0.5)})
    assert _picked([a, b, c]) == ["aaaaaaa2"]
    with pytest.raises(LookupError, match="no AlpaSim scene was scored by every run"):
        R.showcase([a, b, _run("c", {"aaaaaaa3": _metrics(0.5)})])


def test_render_sample_says_what_it_looked_for(tmp_path: Path) -> None:
    run = _run("a", {"aaaaaaa1": _metrics(1.0)})
    cfg = SimpleNamespace(runs=tmp_path)
    with pytest.raises(LookupError, match="they look like alpasim/clipgt-01d503d4"):
        R.render_sample([run], "aaaaaaa1", tmp_path / "out", cfg)
    with pytest.raises(LookupError, match=r"camera_front_wide_120fov_default\.mp4.*a/a-run"):
        R.render_sample([run], f"alpasim/{_scene('aaaaaaa1')}", tmp_path / "out", cfg)


def _alpasim_frame(box: tuple[int, int, int, int], value: int = 128) -> np.ndarray:
    """One frame laid out like AlpaSim's video: white, a framed map on top, the framed camera picture."""
    frame = np.full((1000, 900, 3), 255, np.uint8)
    frame[30:504, 62:536] = 0  # the map's frame is about half the width
    frame[32:502, 64:534] = 255
    frame[120:122, 590:885] = 0  # a rule of the metrics table
    left, top, right, bottom = box
    frame[top:bottom, left:right] = 0
    frame[top + 2 : bottom - 2, left + 2 : right - 2] = value
    return frame


def test_the_camera_picture_is_found_from_the_frame_not_from_fixed_numbers() -> None:
    cases = (((29, 516, 872, 992), 16 / 9), ((8, 519, 894, 989), 16 / 9), ((150, 560, 850, 960), 1.0))
    for box, aspect in cases:
        left, top, right, bottom = R._camera_box(_alpasim_frame(box), aspect)
        # Inside the frame line, never on it, and of the shape asked for.
        assert box[0] < left < right < box[2] and box[1] < top < bottom < box[3]
        assert abs((right - left) / (bottom - top) - aspect) < 0.01
        assert (_alpasim_frame(box)[top:bottom, left:right] == 128).all()
        # Trimmed about the centre, and only one way: the other side keeps all but the frame line.
        assert abs((left + right) - (box[0] + box[2])) <= 2 and abs((top + bottom) - (box[1] + box[3])) <= 2
        assert max(right - left - (box[2] - box[0]), bottom - top - (box[3] - box[1])) >= -8
    # A night scene is dark all over; the frame line is still the first and last dark row.
    assert R._camera_box(_alpasim_frame((29, 516, 872, 992), value=20), 16 / 9)[1] in range(517, 523)
    with pytest.raises(RuntimeError, match="no camera picture found"):
        R._camera_box(np.full((1000, 900, 3), 255, np.uint8), 16 / 9)


def test_the_bar_says_how_the_scored_part_ended_and_every_speed_is_the_cars_own() -> None:
    crash = _metrics(0.0, 8.0, collision_rate=1.0, collision_at_fault=1.0)
    stop = R._stop_frame(crash)
    assert stop == round((3.0 + 8.0) * 20) + 1  # the event is one step after the last scored one
    assert R._outcome(crash, stop, 32.2) == ("score 0.00", "collided at 11 s, driving 32 km/h")
    assert R._outcome(crash, stop, None) == ("score 0.00", "collided at 11 s")
    bumped = _metrics(0.99, 12.6, collision_rate=1.0, collision_at_fault=0.0)
    assert R._outcome(bumped, R._stop_frame(bumped), 0.4) == ("score 0.99", "was hit at 16 s, standing still")
    off = _metrics(0.38, 3.0, offroad=1.0)
    assert R._outcome(off, R._stop_frame(off), 50.0) == ("score 0.38", "left the road at 6 s")
    astray = _metrics(0.75, 11.6, dist_to_gt_trajectory=4.02, progress=0.6)
    assert R._outcome(astray, R._stop_frame(astray), 48.0) == ("score 0.75", "4 m off the route at 15 s")
    done = _metrics(1.0, 16.6, dist_to_gt_trajectory=4.1, progress=1.0)
    assert R._outcome(done, R._stop_frame(done), 62.0)[1] == "passed the end of the route at 20 s"
    clean = _metrics(1.0, 16.85, dist_to_gt_trajectory=0.9)
    assert R._stop_frame(clean) is None
    assert R._outcome(clean, None, None) == ("score 1.00", "scored to the end")


def _rollout(tmp_path: Path, rows: int, step_us: int = 50_000) -> Path:
    """A rollout folder with a controller log beside it; the video is only a name here."""
    rollout = tmp_path / "scene" / "rollouts" / "scene" / "r1"
    rollout.mkdir(parents=True)
    log = tmp_path / "scene" / "controller" / "alpasim_controller_r1.csv"
    log.parent.mkdir()
    lines = ["timestamp_us,vx,vy"] + [f"{1_000_000 + k * step_us},{k / 10},0.0" for k in range(rows)]
    log.write_text("\n".join(lines) + "\n")
    return rollout / "scene_r1_camera_front_wide_120fov_default.mp4"


def test_the_speed_is_the_cars_own_at_the_stop_two_steps_into_the_log(tmp_path: Path) -> None:
    video = _rollout(tmp_path, rows=100)
    # Frame 52 of the video is row 50 of the log: 5.0 m/s, 18 km/h.
    assert R._speed_kmh(video, 52) == pytest.approx(18.0)
    assert R._speed_kmh(video, None) is None
    assert R._speed_kmh(video, 150) is None  # the log ends first: no speed, never the last row's
    assert R._speed_kmh(tmp_path / "a" / "b" / "c" / "d" / "none.mp4", 52) is None  # no log at all


def test_a_drive_on_another_clock_is_refused(tmp_path: Path) -> None:
    def run(**config) -> LoadedRun:
        return LoadedRun("a", "alpasim", "a-run", {"config": config}, {}, [])

    video = _rollout(tmp_path / "ok", rows=396)
    R._check_clock(run(), video, 398)  # 20 steps a second, one frame a step
    R._check_clock(run(control_timestep_us=50_000, force_gt_duration_us=3_000_000), video, 398)
    R._check_clock(run(bridge_args=["--force-gt-us", "3000000"]), video, 398)
    R._check_clock(run(), tmp_path / "a" / "b" / "c" / "d" / "none.mp4", 10)  # nothing recorded: no check
    with pytest.raises(R.NotDrawable, match="video_every_nth_frame = 1"):
        R._check_clock(run(), video, 199)  # every second frame only
    with pytest.raises(R.NotDrawable, match="steps every 0.1 s"):
        R._check_clock(run(), _rollout(tmp_path / "slow", rows=198, step_us=100_000), 200)
    with pytest.raises(R.NotDrawable, match="steps every 0.1 s"):
        R._check_clock(run(control_timestep_us=100_000), video, 398)
    with pytest.raises(R.NotDrawable, match="hands over to the model at 2 s"):
        R._check_clock(run(bridge_args=["--force-gt-us", "2000000"]), video, 398)
    assert issubclass(R.NotDrawable, LookupError)  # so `paddock render` prints it and exits 1


def test_a_tile_freezes_dimmed_from_the_moment_its_scoring_stopped() -> None:
    pytest.importorskip("PIL")
    w, h = 480, 270
    ramp = [np.full((h, w, 3), 100 + 50 * k, np.uint8) for k in range(4)]  # 100, 150, 200, 250
    tiles = [
        R._Tile("Stopped", "#ff0000", "score 0.00", "left the road at 0 s", ramp, 1),
        R._Tile("Scored", "#00ff00", "score 1.00", "scored to the end", ramp, None),
    ]
    height = R.BAR + h + R.STRIP
    body = slice(R.BAR + 10, R.BAR + h - 10)
    before = R._compose(tiles, 0, 4, w, h, height, "test")
    assert before.shape == (height, R.WIDTH, 3)
    assert before[body, 10 : w - 10].mean() == 100  # still scored: not dimmed
    after = R._compose(tiles, 3, 4, w, h, height, "test")
    # Frozen on the frame where scoring stopped (150), dimmed; the other tile plays on.
    assert abs(after[body, 10 : w - 10].mean() - 150 * 0.45) < 2
    assert after[body, w + 10 : 2 * w - 10].mean() == 250


FFMPEG = resolve_ffmpeg(None)


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
def test_render_sample_writes_one_video_on_one_clock(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    scene = _scene("aaaaaaa1")
    runs = []
    # a: scored for 0.5 s after the hand-over, then off the road. b: scored to the end of its 4 s.
    for model, metrics in (("a", _metrics(0.0, 0.5, offroad=1.0)), ("b", _metrics(1.0, 16.85))):
        rollout = tmp_path / "alpasim" / model / f"{model}-run" / scene / "rollouts" / scene / "r1"
        rollout.mkdir(parents=True)
        frames = (_alpasim_frame((29, 516, 872, 992), value=60 + 2 * k) for k in range(80))
        _encode(frames, rollout / f"{scene}_r1_camera_front_wide_120fov_default.mp4", 900, 1000, fps=20)
        (rollout / "_complete").touch()
        runs.append(_run(model, {"aaaaaaa1": metrics}))
    written = R.render_sample(runs, f"alpasim/{scene}", tmp_path / "out", SimpleNamespace(runs=tmp_path))
    assert [p.name for p in written] == ["scene-aaaaaaa1.mp4", "scene-aaaaaaa1.jpg"]
    frames = _decode(written[0], 192, 39)
    # b is scored to its last frame (80 frames), then the last frame holds for 1.5 s at 20 a second.
    assert len(frames) == 80 + round(R.HOLD_S * R.FPS)
    # a froze, dimmed, at frame 71: by the end its tile is darker than b's, which played on.
    assert frames[-1][10:30, 5:40].mean() < frames[-1][10:30, 53:88].mean() - 30
