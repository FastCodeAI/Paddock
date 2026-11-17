"""nuScenes' streamed mode: one stream per scene, at each keyframe's own time and global pose.

No dataset: a synthetic index of two scenes, with nuScenes' 0.4-0.6 s keyframe
gaps and a global frame far from the origin, goes through the harness's own
`samples()` and `_sample`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes import harness as NH
from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.adapters.models.drivejepa import momentum
from paddock.config import SiteConfig
from paddock.registry import Registry

#: A scene's clock, absolute microseconds, and its keyframe gaps in seconds.
T0 = 1_533_151_603_547_590
SCENES = {
    "scene-0003": [0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5],
    "scene-0012": [0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5],
}


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:3, 3] = [x, y, 0.0]
    return pose


def index() -> list[list[L.Keyframe]]:
    """Two scenes of keyframes in the loader's own shape, scene then time order."""
    scenes = []
    for s, (scene, gaps) in enumerate(SCENES.items()):
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        scenes.append(
            [
                L.Keyframe(
                    token=f"{scene}-{i}",
                    scene=scene,
                    index_in_scene=i,
                    timestamp_us=T0 + 3_600_000_000 * s + round(t * 1e6),
                    # Global frame: Boston-seaport-sized offsets, heading turning.
                    ego2world=_pose(600.0 + 5.0 * t, 1600.0 + 2.0 * t, 0.3 + 0.01 * i),
                )
                for i, t in enumerate(times)
            ]
        )
    return scenes


def harness(tmp_path, monkeypatch):
    """The real harness on a throwaway site, its index replaced by `index()`."""
    for sub in ("repos", "runs", "weights", "nuscenes"):
        (tmp_path / sub).mkdir(exist_ok=True)
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n[datasets.nuscenes]\npath = "{tmp_path / "nuscenes"}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    bench = NH.BENCHMARK(Registry.load().benchmarks["nuscenes"], SiteConfig.load())
    bench._index_cache = index()
    return bench


def expected_fed(samples) -> list[bool]:
    """Drive-JEPA's rule over the samples in the order they are sent: fed when the
    previous sample of the same scene was planned 0.45-0.55 s earlier."""
    fed, last = [], {}
    for sample in samples:
        key, now = sample.stream.key, sample.stream.timestamp_us
        before = last.get(key)
        fed.append(before is not None and momentum.MIN_AGE_US <= now - before <= momentum.MAX_AGE_US)
        last[key] = now
    return fed


def test_off_the_samples_carry_no_stream_and_the_run_config_is_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(NH.ENV_STREAMED, raising=False)
    bench = harness(tmp_path, monkeypatch)
    samples = list(bench.samples())
    assert samples and all(s.stream is None and s.warmup is None for s in samples)
    assert "streamed" not in bench.run_config(), "no new key, so no new run id"
    assert bench.stream_mode() is None and bench.serial_streams() is False
    assert bench.remarks() == []


def test_on_each_sample_names_its_scene_at_its_own_time_and_global_pose(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(NH.ENV_STREAMED, "1")
    bench = harness(tmp_path, monkeypatch)
    frames = {f"nuscenes/val/{f.scene}/{f.index_in_scene}": f for scene in index() for f in scene}
    samples = list(bench.samples())
    # Every keyframe with six keyframes of future: 11 - 6 and 10 - 6.
    assert len(samples) == 5 + 4
    for sample in samples:
        cur = frames[sample.sample_id]
        assert sample.stream.key == cur.scene
        assert sample.stream.timestamp_us == cur.timestamp_us
        assert sample.stream.pose == pytest.approx(
            (cur.ego2world[0, 3], cur.ego2world[1, 3], math.atan2(cur.ego2world[1, 0], cur.ego2world[0, 0]))
        )
        assert sample.warmup is None, "nuScenes streams across samples; no warm-up is planned"
        assert sample.observations[-1].timestamp_us == cur.timestamp_us
    # In index order within a scene, which the runner keeps (serial_streams).
    assert [s.sample_id for s in samples] == sorted(
        (s.sample_id for s in samples), key=lambda i: (i.rsplit("/", 2)[1], int(i.rsplit("/", 1)[1]))
    )
    assert bench.serial_streams() is True and bench.stream_mode() == NH.STREAMED_PER_SCENE
    assert bench.run_config()["streamed"] == NH.STREAMED_PER_SCENE
    (remark,) = bench.remarks()
    assert NH.ENV_STREAMED in remark and "0.45-0.55 s" in remark
    # 0.4 / 0.6 s gaps are outside the window; the first keyframe of a scene has none.
    assert expected_fed(samples) == [False, True, True, False, True, False, True, False, True]


def test_the_stream_does_not_change_what_the_model_is_given(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(NH.ENV_STREAMED, raising=False)
    plain = [[o.to_wire() for o in s.observations] for s in harness(tmp_path, monkeypatch).samples()]
    monkeypatch.setenv(NH.ENV_STREAMED, "on")
    streamed = [[o.to_wire() for o in s.observations] for s in harness(tmp_path, monkeypatch).samples()]
    assert plain == streamed


def test_preflight_refuses_a_misspelt_switch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(NH.ENV_STREAMED, "per-scene")
    problems = harness(tmp_path, monkeypatch).preflight()
    assert any(p.startswith(f"{NH.ENV_STREAMED}='per-scene'") for p in problems)
