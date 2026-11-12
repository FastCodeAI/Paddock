"""The website's neuroNCAP videos: which episodes are picked, and how one is drawn."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.neuroncap import render as R
from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
from paddock.render import LoadedRun


def _run(model: str, scores: dict[tuple[str, str], list[float]]) -> LoadedRun:
    samples = []
    for (kind, scene), runs in scores.items():
        for i, score in enumerate(runs):
            samples.append(
                {
                    "sample_id": f"neuroncap/{scene}/{kind}/run_{i}",
                    "native_id": {"scenario": kind, "scene": scene, "run": i},
                    "metrics": {"ncap_score": score, "collision_rate": float(score < 5), "impact_speed": 5.0},
                }
            )
    return LoadedRun(model, "neuroncap", f"{model}-run", {}, {}, samples)


def test_showcase_picks_the_hardest_of_each_kind_then_the_widest_disagreement() -> None:
    a = _run(
        "a",
        {
            ("frontal", "0001"): [4, 1, 2],  # with b: the hardest frontal; run 2 is a's most typical
            ("frontal", "0002"): [3, 3, 3],
            ("side", "0001"): [0, 0, 0],  # harder still, but scene 0001 is already shown
            ("side", "0003"): [2, 2, 2],
            ("stationary", "0004"): [5, 5, 5],
            ("side", "0005"): [0, 0, 3],  # a averages 1, b 5: the widest gap
        },
    )
    b = _run(
        "b",
        {
            ("frontal", "0001"): [1, 1, 1],
            ("frontal", "0002"): [3, 3, 3],
            ("side", "0001"): [0, 0, 0],
            ("side", "0003"): [3, 3, 3],
            ("stationary", "0004"): [4, 4, 4],
            ("side", "0005"): [5, 5, 5],
        },
    )
    picks = R.showcase([a, b])
    assert [p["sample"] for p in picks] == [
        "neuroncap/0001/frontal/run_2",
        "neuroncap/0003/side/run_0",
        "neuroncap/0004/stationary/run_0",
        "neuroncap/0005/side/run_0",
    ]
    assert picks[0]["title"] == "Oncoming car, scene 0001"
    assert picks[0]["caption"].startswith("The hardest frontal scenario: the models average 1.7 of 5.")
    assert "Run 3 of 3, the most typical one." in picks[0]["caption"]
    assert picks[3]["caption"].startswith(
        "Where the models disagree most: their averages run from 1.0 to 5.0"
    )


def test_showcase_considers_only_scenarios_every_run_scored() -> None:
    a = _run("a", {("frontal", "0001"): [1]})
    b = _run("b", {("frontal", "0002"): [1]})
    with pytest.raises(LookupError, match="no neuroNCAP scenario was scored by every run"):
        R.showcase([a, b])


def test_render_sample_says_what_it_looked_for(tmp_path: Path) -> None:
    run = _run("a", {("side", "0103"): [5]})
    cfg = SimpleNamespace(runs=tmp_path)
    with pytest.raises(LookupError, match="they look like neuroncap/0103/frontal/run_0"):
        R.render_sample([run], "0103-side", tmp_path / "out", cfg)
    with pytest.raises(LookupError, match=r"side-0103/run_0/FC_TRAJ\.mp4.*a/a-run"):
        R.render_sample([run], "neuroncap/0103/side/run_0", tmp_path / "out", cfg)


def test_a_finished_episode_freezes_dimmed_on_its_last_frame() -> None:
    pytest.importorskip("PIL")
    w, h = 640, 360
    white = [np.full((h, w, 3), 255, np.uint8)]
    tiles = [
        R._Tile("Short", "#ff0000", {"ncap_score": 0.0, "collision_rate": 1.0, "impact_speed": 10.0}, white),
        R._Tile("Long", "#00ff00", {"ncap_score": 5.0, "collision_rate": 0.0}, white * 3),
    ]
    height = R.BAR + h + R.STRIP
    frame = R._compose(tiles, 2, 3, w, h, height, "test")
    assert frame.shape == (height, R.WIDTH, 3)
    body = slice(R.BAR + 10, R.BAR + h - 10)
    assert frame[body, 10 : w - 10].mean() < 140  # the short episode is over: dimmed
    assert frame[body, w + 10 : 2 * w - 10].mean() > 250  # the long one is still running
    # impact_speed is the two cars' speed toward each other, never the model's own speed.
    assert R._outcome(tiles[0].metrics) == "NCAP 0.0 · the cars met at 36 km/h"
    assert R._outcome(tiles[1].metrics) == "NCAP 5.0 · no collision"


FFMPEG = resolve_ffmpeg(None)


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
def test_render_sample_writes_one_video_on_one_clock(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    runs = []
    for model, frames in (("a", 10), ("b", 20)):  # 1 s and 2 s at neuroNCAP's 10 frames a second
        episode = tmp_path / "neuroncap" / model / f"{model}-run" / "frontal-0103" / "run_0"
        episode.mkdir(parents=True)
        grey = (np.full((180, 320, 3), 8 * k, np.uint8) for k in range(frames))
        R._encode(grey, episode / "FC_TRAJ.mp4", 320, 180)
        runs.append(_run(model, {("frontal", "0103"): [5]}))
    written = R.render_sample(
        runs, "neuroncap/0103/frontal/run_0", tmp_path / "out", SimpleNamespace(runs=tmp_path)
    )
    assert [p.name for p in written] == ["frontal-0103-run0.mp4", "frontal-0103-run0.jpg"]
    frames = R._decode(written[0], 192, 82)
    # The longer episode sets the length (20 frames), then the last frame holds for 1.5 s.
    assert len(frames) == 20 + round(R.HOLD_S * R.FPS)
