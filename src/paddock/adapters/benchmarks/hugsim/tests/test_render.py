"""HUGSIM's website videos: the episode pick rule and the side-by-side render. No GPU, no simulator."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.hugsim import render as R
from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg
from paddock.render import LoadedRun


def _sample(dataset: str, scene: str, hd: float) -> dict:
    return {
        "sample_id": f"hugsim/{dataset}/scene-{scene}-easy-00",
        "native_id": {
            "dataset": dataset,
            "scene": scene,
            "source": f"{dataset}_bridge/{scene}_easy_00/eval.json",
        },
        "group": "easy",
        "metrics": {"hd_score": hd, "route_completion": 100 * hd},
    }


def _run(model: str, rows: list[tuple[str, str, float]]) -> LoadedRun:
    return LoadedRun(model=model, benchmark="hugsim", run_id=f"{model}-run", run={}, metrics={},
                     samples=[_sample(*r) for r in rows])  # fmt: skip


def test_showcase_takes_each_datasets_widest_disagreement() -> None:
    a = _run("a", [("nuscenes", "1", 0.0), ("nuscenes", "2", 0.0), ("waymo", "3", 0.9), ("waymo", "4", 0.5)])
    b = _run("b", [("nuscenes", "1", 1.0), ("nuscenes", "2", 0.1), ("waymo", "3", 0.0), ("waymo", "4", 0.5)])
    picks = R.showcase([a, b], 4)
    # Each dataset's widest gap first (nuScenes 1: 1.0, Waymo 3: 0.9), then the widest left (2: 0.1, 4: 0).
    assert [p["sample"].split("/")[-1] for p in picks] == [
        "scene-1-easy-00",
        "scene-3-easy-00",
        "scene-2-easy-00",
        "scene-4-easy-00",
    ]
    assert "from 0.00 to 1.00" in picks[0]["caption"] and picks[0]["title"].startswith("nuScenes, easy")


def test_showcase_needs_every_run() -> None:
    with pytest.raises(LookupError, match="no HUGSIM episode was scored by every run"):
        R.showcase([_run("a", [("nuscenes", "1", 1.0)]), _run("b", [("waymo", "2", 1.0)])], 4)


def test_front_project_puts_a_straight_plan_on_the_centre_line() -> None:
    camera_in_vehicle = np.array([[1, 0, 0, 1.7], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]])  # 1.5 m up
    intrinsic = {"W": 800, "H": 450, "cx": 400.0, "cy": 225.0, "fovx": 1.0, "fovy": 0.6}
    cam = {"intrinsic": intrinsic, "v2c": np.linalg.inv(camera_in_vehicle)}
    (u, v), (u2, _), far = R.front_project(np.array([[0.0, 10.0], [2.0, 10.0], [0.0, 0.2]]), cam)
    assert u == pytest.approx(400) and v > 225  # ahead on the ground: centre, below the horizon
    assert u2 > u  # x is to the right
    assert np.isnan(far).all()  # too close to be ahead of the camera


def _episode(root: Path, model: str, scene: str, steps: int) -> None:
    from paddock.adapters.benchmarks.neuroncap.render import _encode

    d = root / model / f"nuscenes_bridge/{scene}_easy_00"
    d.mkdir(parents=True)
    _encode((np.full((900, 2400, 3), 9 * k, np.uint8) for k in range(steps)), d / "video.mp4", 2400, 900)
    lines = [json.dumps({"step": k, "plan": [[0.0, 5.0 + j] for j in range(8)]}) for k in range(steps)]
    (d / "agent_log.jsonl").write_text("\n".join(lines))
    cam = {
        "intrinsic": {"W": 800, "H": 450, "cx": 400.0, "cy": 225.0, "fovx": 1.0, "fovy": 0.6},
        "v2c": np.eye(4),
    }
    with open(d / "infos.pkl", "wb") as f:
        pickle.dump([{"cam_params": {"CAM_FRONT": cam}} for _ in range(steps)], f)


def test_render_sample_says_where_it_looked(tmp_path: Path) -> None:
    run = _run("a", [("nuscenes", "1", 1.0)])
    with pytest.raises(LookupError, match="no run here has the episode folder"):
        R.render_sample([run], "hugsim/nuscenes/scene-1-easy-00", tmp_path, SimpleNamespace(runs=tmp_path))


FFMPEG = resolve_ffmpeg(None)


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
def test_render_sample_writes_one_video_on_one_clock(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    _episode(tmp_path, "a", "1", 4)  # 1 s at 0.25 s a step
    _episode(tmp_path, "b", "1", 8)  # 2 s
    runs = [_run("a", [("nuscenes", "1", 0.0)]), _run("b", [("nuscenes", "1", 1.0)])]
    cfg = SimpleNamespace(runs=tmp_path / "none", benchmarks={"hugsim": {"episodes": str(tmp_path)}})
    written = R.render_sample(runs, "hugsim/nuscenes/scene-1-easy-00", tmp_path / "out", cfg)
    assert [p.name for p in written] == ["scene-1-easy-00.mp4", "scene-1-easy-00.jpg"]
    from paddock.adapters.benchmarks.neuroncap.render import _decode

    # The longer episode's 2 s at 10 fps, then the 1.5 s hold.
    assert len(_decode(written[0], 192, 100, FFMPEG)) == round(8 * R.STEP_S * R.FPS) + round(R.HOLD_S * R.FPS)
