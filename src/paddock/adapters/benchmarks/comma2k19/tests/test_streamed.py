"""comma2k19's streamed mode: a stream per sample and a warm-up 0.5 s before each keyframe.

A synthetic segment (straight ahead at 25 m/s, 20 Hz) through the harness's own
`_segment_samples`, with decoded frames stood in by `_synthetic_frames`.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19.tests.synthetic import straight_pose, write_segment
from paddock.adapters.benchmarks.comma2k19.tests.test_harness import _AlwaysValid, _bench, _synthetic_frames
from paddock.benchmark import STREAMED_WARMUP

SPEED = 25.0
HISTORY = 16


def _samples(tmp_path, monkeypatch, **settings):
    bench = _bench(history_frames=HISTORY, keyframes_per_segment=2, **settings)
    bench.cfg.runs = tmp_path / "runs"
    frames = _synthetic_frames(260)

    def fake_decode(video, limit=None, **kwargs):
        yield from frames[: limit if limit is not None else len(frames)]

    monkeypatch.setattr(L, "decode_frames", fake_decode)
    seg = write_segment(tmp_path / "seg", straight_pose(speed=SPEED, n=260))
    segment = L.SegmentRef(chunk=1, route_id="dongle|route", segment=4, path=seg)
    return bench, list(bench._segment_samples(segment, _AlwaysValid()))


def test_off_there_is_no_stream_no_warmup_and_no_new_run_config(tmp_path, monkeypatch, rotations) -> None:
    monkeypatch.delenv(H.ENV_STREAMED, raising=False)
    bench, samples = _samples(tmp_path, monkeypatch)
    assert len(samples) == 2
    assert all(s.stream is None and s.warmup is None for s in samples)
    assert "streamed" not in bench.run_config()
    assert bench.stream_mode() is None and bench.remarks() == []


def test_on_the_warmup_is_ten_frames_back_in_the_windows_frame(tmp_path, monkeypatch, rotations) -> None:
    monkeypatch.setenv(H.ENV_STREAMED, "1")
    bench, samples = _samples(tmp_path, monkeypatch)
    assert len(samples) == 2
    for sample in samples:
        assert sample.stream.key == sample.sample_id
        assert sample.stream.timestamp_us == sample.observations[-1].timestamp_us
        # The window's first frame is the origin; the keyframe is 15 frames on.
        np.testing.assert_allclose(sample.stream.pose, ((HISTORY - 1) / 20 * SPEED, 0.0, 0.0), atol=1e-6)

        warm = sample.warmup
        assert warm.stream.key == sample.sample_id
        assert sample.stream.timestamp_us - warm.stream.timestamp_us == 500_000
        assert len(warm.observations) == HISTORY - 10
        assert all(a is b for a, b in zip(warm.observations, sample.observations, strict=False))
        np.testing.assert_allclose(warm.stream.pose, ((HISTORY - 11) / 20 * SPEED, 0.0, 0.0), atol=1e-6)
    assert bench.stream_mode() == STREAMED_WARMUP and bench.serial_streams() is False
    assert bench.run_config()["streamed"] == STREAMED_WARMUP
    (remark,) = bench.remarks()
    assert "A Paddock choice" in remark


def test_the_scored_request_is_unchanged_by_the_switch(tmp_path, monkeypatch, rotations) -> None:
    def wire(sample):
        return [{k: v for k, v in o.to_wire().items() if k != "image_paths"} for o in sample.observations]

    monkeypatch.delenv(H.ENV_STREAMED, raising=False)
    _, plain = _samples(tmp_path / "off", monkeypatch)
    monkeypatch.setenv(H.ENV_STREAMED, "1")
    _, streamed = _samples(tmp_path / "on", monkeypatch)
    # The staged frames live under each run's own ring; their names match.
    assert [wire(s) for s in plain] == [wire(s) for s in streamed]
    assert [[o.image_paths["CAM_FRONT"].rsplit("/", 1)[1] for o in s.observations] for s in plain] == [
        [o.image_paths["CAM_FRONT"].rsplit("/", 1)[1] for o in s.observations] for s in streamed
    ]
