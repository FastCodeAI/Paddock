"""Bench2Drive's website videos: the route pick rule and the side-by-side render. No GPU, no CARLA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive import render as R
from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
from paddock.render import LoadedRun

#: route id -> (town, scenario)
ROUTES = {
    "100": ("Town01", "Accident"),  # Overtaking
    "101": ("Town01", "ConstructionObstacle"),  # Overtaking
    "200": ("Town02", "ParkingExit"),  # Merging
    "300": ("Town03", "HardBreakRoute"),  # Emergency braking
    "400": ("Town04", "VanillaSignalizedTurnEncounterGreenLight"),  # no listed ability
    "500": ("Town05", "StaticCutIn"),  # Emergency braking
}


def _run(model: str, scores: dict[str, float]) -> LoadedRun:
    samples = [
        {
            "sample_id": f"bench2drive/{r}-{ROUTES[r][0]}-{ROUTES[r][1]}",
            "native_id": {"route_id": r, "town": ROUTES[r][0], "scenario": ROUTES[r][1]},
            "metrics": {"driving_score": ds},
        }
        for r, ds in scores.items()
    ]
    return LoadedRun(
        model=model, benchmark="bench2drive", run_id=f"{model}-220", run={}, metrics={}, samples=samples
    )


def test_showcase_picks_the_hardest_of_each_ability_then_the_widest_disagreement() -> None:
    runs = [
        _run("a", {"100": 10, "101": 30, "200": 0, "300": 20, "400": 100, "500": 50}),
        _run("b", {"100": 20, "101": 10, "200": 2, "300": 10, "400": 0, "500": 60}),
    ]
    picks = R.showcase(runs, 4)
    routes = [p["sample"].split("/")[1].split("-")[0] for p in picks]
    # Overtaking: 100 (mean 15) beats 101 (20); Merging: 200; Emergency braking: 300 (15) beats 500 (55);
    # then the widest gap among the rest: 400 (100 vs 0).
    assert routes == ["100", "200", "300", "400"]
    assert picks[0]["title"].startswith("Overtaking: Accident, route 100")
    assert "15.0" in picks[0]["caption"] and "recorded for this video" in picks[0]["caption"]
    assert "from 0.0 to 100.0" in picks[3]["caption"]


def test_showcase_never_shows_a_scenario_type_twice_and_needs_every_run() -> None:
    runs = [_run("a", {"100": 10, "101": 10}), _run("b", {"100": 10})]
    # 101 lacks b's score, so only 100 is eligible, and Overtaking is used once.
    assert [p["sample"] for p in R.showcase(runs, 4)] == ["bench2drive/100-Town01-Accident"]
    with pytest.raises(LookupError, match="no Bench2Drive route was scored by every run"):
        R.showcase([_run("a", {"100": 1}), _run("b", {"101": 1})], 4)


def test_front_project_puts_a_point_ahead_on_the_centre_line_below_the_horizon() -> None:
    cam = (800, 450, 70.0, 1.5, 1.5)
    u, v = R.front_project(np.array([[11.5, 0.0]]), cam)[0]
    assert u == pytest.approx(400) and v > 225
    left = R.front_project(np.array([[11.5, 2.0]]), cam)[0]
    assert left[0] < 400  # y is to the left, so left of centre
    assert np.isnan(R.front_project(np.array([[1.0, 0.0]]), cam)).all()  # behind the camera


def _record(root: Path, model: str, route: str, ticks: int, ds: float) -> Path:
    town, scenario = ROUTES[route]
    d = root / "bench2drive" / model / f"2026{model}" / f"{route}-{town}-{scenario}"
    (d / "front").mkdir(parents=True)
    from PIL import Image

    lines = []
    for t in range(ticks):
        Image.fromarray(np.full((90, 160, 3), 10 * (t % 20), np.uint8)).save(d / "front" / f"{t:05d}.jpg")
        lines.append(json.dumps({"t": t, "plan": [[2.0 + k, 0.0] for k in range(6)]}))
    (d / "ticks.jsonl").write_text("\n".join(lines))
    (d / "rig.json").write_text(
        json.dumps(
            {"cameras": {"CAM_FRONT": {"pose": [1.5, 0, 1.5, 0, 0, 0], "fov": 70.0, "size": [160, 90]}}}
        )
    )
    record = {"status": "Completed", "scores": {"score_composed": ds, "score_route": 100.0}}
    (d / "results.json").write_text(json.dumps({"_checkpoint": {"records": [record]}}))
    return d


def test_render_sample_says_where_it_looked(tmp_path: Path) -> None:
    runs = [_run("a", {"100": 10})]
    with pytest.raises(LookupError, match="no recorded drive of route 100"):
        R.render_sample(
            runs, "bench2drive/100-Town01-Accident", tmp_path / "out", SimpleNamespace(runs=tmp_path)
        )


FFMPEG = resolve_ffmpeg(None)


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
def test_render_sample_writes_one_video_on_one_clock(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _record(tmp_path, "a", "100", 20, 40.0)  # 1 s of ticks
    _record(tmp_path, "b", "100", 40, 80.0)  # 2 s
    runs = [_run("a", {"100": 10}), _run("b", {"100": 90})]
    written = R.render_sample(
        runs, "bench2drive/100-Town01-Accident", tmp_path / "out", SimpleNamespace(runs=tmp_path)
    )
    assert [p.name for p in written] == ["route-100.mp4", "route-100.jpg"]
    from paddock.adapters.benchmarks.neuroncap.render import _decode

    frames = _decode(written[0], 192, 80, FFMPEG)
    # One frame every second tick: the longer drive's 40 ticks give 20 frames, then the last holds 1.5 s.
    assert len(frames) == 40 // R.STEP + round(R.HOLD_S * R.FPS)


def test_a_finished_drive_freezes_dimmed_on_its_last_frame(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    short = _record(tmp_path, "a", "100", 4, 40.0)
    tiles = [R._tile(_run("a", {"100": 10}), "100", tmp_path)]
    assert tiles[0].this == "DS 40.0 · Completed" and tiles[0].ranked == "ranked 10.0"
    live = R._compose(tiles, 1, 3, 640, 360, 48 + 360 + 46, "t")
    over = R._compose(tiles, 5, 3, 640, 360, 48 + 360 + 46, "t")
    assert over[48:408, :640].mean() < live[48:408, :640].mean()
    assert short.exists()
