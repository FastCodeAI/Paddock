import json
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes.history import before, table_rows
from paddock.adapters.benchmarks.nuscenes.loader import Keyframe


def test_history_never_includes_scored_or_future_exposures():
    frames = [Keyframe(str(t), "scene", -1, t, np.eye(4)) for t in (0, 50_000, 140_000, 500_000, 550_000)]
    assert [f.timestamp_us for f in before(frames, 500_000, 0.4)] == [140_000]
    assert [f.timestamp_us for f in before(frames, 500_000, 0.5)] == [0, 50_000, 140_000]


def test_streamed_table_crosses_chunk_boundary(tmp_path: Path):
    rows = [{"text": "x" * 1_100_000}, {"timestamp": 42}, {"nested": [1, 2]}]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(rows))
    assert list(table_rows(path)) == rows
    path.write_text('[{"truncated":')
    with pytest.raises(ValueError, match="incomplete"):
        list(table_rows(path))


def test_dense_history_preserves_anchor_and_ground_truth(monkeypatch):
    from paddock.adapters.benchmarks.nuscenes.harness import NuScenesBenchmark

    harness = object.__new__(NuScenesBenchmark)
    settings = {}
    monkeypatch.setattr(harness, "setting", lambda key, default=None: settings.get(key, default))
    harness._boxes = {}
    frames = []
    for i in range(8):
        pose = np.eye(4)
        pose[0, 3] = i * 2
        frames.append(Keyframe(str(i), "scene", i, i * 500_000, pose))
    baseline = harness._sample(frames, 1)
    exposures = [Keyframe(str(t), "scene", -1, t, np.eye(4)) for t in (100_000, 400_000, 600_000)]
    harness._camera_history_cache = {"scene": exposures}
    settings.update(camera_history="CAM_FRONT", history_seconds=5.0)
    dense = harness._sample(frames, 1)
    assert dense.sample_id == baseline.sample_id
    np.testing.assert_array_equal(dense.ground_truth, baseline.ground_truth)
    assert [o.timestamp_us for o in dense.observations] == [100_000, 400_000, 500_000]
    np.testing.assert_array_equal(dense.observations[-1].ego2world, baseline.observations[-1].ego2world)
    assert harness.run_config() == {"camera_history": "CAM_FRONT", "history_seconds": 5.0, "scoring_hz": 2.0}


def test_openpilot_rejects_keyframe_only_history(monkeypatch):
    from types import SimpleNamespace

    from paddock.adapters.benchmarks.nuscenes.harness import NuScenesBenchmark

    harness = object.__new__(NuScenesBenchmark)
    monkeypatch.setattr(harness, "setting", lambda key, default=None: default)
    assert harness.input_problems(SimpleNamespace(name="openpilot", min_input_hz=10))
    assert not harness.input_problems(SimpleNamespace(name="other", min_input_hz=2))
