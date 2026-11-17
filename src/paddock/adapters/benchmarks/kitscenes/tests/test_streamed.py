"""KITScenes' streamed mode: a stream per sample and a warm-up 0.5 s before each scored keyframe.

The synthetic scene of test_harness.py (10 Hz poses, straight ahead at 5 m/s,
keyframe 50 with 48 frames of history), through the harness's own `samples()`.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.benchmarks.kitscenes import harness as H
from paddock.adapters.benchmarks.kitscenes.harness import POD_HEIGHT_M, RING
from paddock.adapters.benchmarks.kitscenes.tests.test_harness import KEYFRAME, _benchmark, _write_scene
from paddock.benchmark import STREAMED_WARMUP
from paddock.policy.protocol import Command


def _sample(tmp_path, monkeypatch, **site):
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive], **site)
    (sample,) = list(bench.samples())
    return bench, sample


def test_off_there_is_no_stream_no_warmup_and_no_new_run_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(H.ENV_STREAMED, raising=False)
    bench, sample = _sample(tmp_path, monkeypatch)
    assert sample.stream is None and sample.warmup is None
    assert bench.run_config() == {"ground_offset_m": POD_HEIGHT_M}, "no new key, so no new run id"
    assert bench.stream_mode() is None and bench.serial_streams() is False and bench.remarks() == []


def test_on_the_warmup_is_the_frame_half_a_second_back_and_its_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(H.ENV_STREAMED, "1")
    bench, sample = _sample(tmp_path, monkeypatch)
    key = f"kitscenes/val/{sample.native_id['scene_id']}/{KEYFRAME}"

    assert sample.stream.key == key == sample.sample_id
    assert sample.stream.timestamp_us == KEYFRAME * 100_000
    # poses.txt drives straight ahead at 0.5 m a frame, in the scene's frame.
    np.testing.assert_allclose(sample.stream.pose, (KEYFRAME * 0.5, 0.0, 0.0), atol=1e-9)

    warm = sample.warmup
    assert warm.stream.key == key
    assert sample.stream.timestamp_us - warm.stream.timestamp_us == 500_000
    np.testing.assert_allclose(warm.stream.pose, ((KEYFRAME - 5) * 0.5, 0.0, 0.0), atol=1e-9)
    # A prefix of the sample's own history, the same frames: index keyframe - 5 last.
    assert len(warm.observations) == len(sample.observations) - 5
    assert all(a is b for a, b in zip(warm.observations, sample.observations, strict=False))
    assert warm.observations[-1].image_paths["CAM_FRONT"].endswith(f"{KEYFRAME - 5:010d}.jpg")

    assert bench.stream_mode() == STREAMED_WARMUP and bench.serial_streams() is False
    assert bench.run_config() == {"ground_offset_m": POD_HEIGHT_M, "streamed": STREAMED_WARMUP}
    (remark,) = bench.remarks()
    assert "A Paddock choice" in remark and "never scored" in remark


def test_the_scored_request_is_unchanged_by_the_switch(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    wires = []
    for switch in ("", "1"):
        monkeypatch.setenv(H.ENV_STREAMED, switch)
        (sample,) = list(_benchmark(tmp_path, monkeypatch, [archive]).samples())
        wires.append([o.to_wire() for o in sample.observations])
    assert wires[0] == wires[1]


def test_under_a_route_hint_the_warmup_frame_gets_the_keyframes_command(tmp_path, monkeypatch) -> None:
    """A copy: the same frame inside the scored request keeps STRAIGHT, as every history frame does."""
    monkeypatch.setenv(H.ENV_STREAMED, "1")
    _, sample = _sample(
        tmp_path,
        monkeypatch,
        settings='command_source = "gt"\n',
        row={"driving_command": np.array([1, 0, 0, 0])},
    )
    assert sample.observations[-1].command == Command.LEFT
    assert sample.warmup.observations[-1].command == Command.LEFT
    assert sample.observations[-6].command == Command.STRAIGHT
    assert sample.warmup.observations[-1].timestamp_us == sample.observations[-6].timestamp_us
    assert all(o.command == Command.STRAIGHT for o in sample.warmup.observations[:-1])


def test_preflight_refuses_a_misspelt_switch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(H.ENV_STREAMED, "warmup")
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    problems = _benchmark(tmp_path, monkeypatch, [archive]).preflight()
    assert any(p.startswith(f"{H.ENV_STREAMED}='warmup'") for p in problems)
