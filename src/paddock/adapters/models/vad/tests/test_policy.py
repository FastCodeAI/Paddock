"""VAD's own inputs, in paddock's env: its command one-hot and the pipeline steps it keeps.

The inputs it shares with the other three planners are checked in uniad/tests/test_bev_planners.py, and
the forward against VAD's own test pipeline in uniad/tests/test_golden.py.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.models.vad.policy import POLICY, reset_vad_state


def test_the_command_is_vads_int_one_hot_of_right_left_straight() -> None:
    """vad_nuscenes_converter writes np.array([1, 0, 0]) for right, [0, 1, 0] left, [0, 0, 1] straight."""
    policy = POLICY()
    for command, row in enumerate(np.eye(3, dtype=np.int64)):
        onehot = policy._extra_results(command)["ego_fut_cmd"]
        assert onehot.dtype == np.int64
        np.testing.assert_array_equal(onehot, row)


def test_only_the_dataset_readers_are_dropped() -> None:
    """Normalisation, the 0.8 scale and the padding stay; what read nuScenes from disk goes."""
    assert set(POLICY.drop_steps) == {
        "LoadMultiViewImageFromFiles",
        "LoadPointsFromFile",
        "LoadAnnotations3D",
        "CustomObjectRangeFilter",
        "CustomObjectNameFilter",
    }
    assert POLICY.collect_keys == ["img", "ego_fut_cmd"]


def test_reset_gives_the_temporal_state_of_a_fresh_model() -> None:
    class Model:
        prev_frame_info = {"prev_bev": object(), "scene_token": "x", "prev_pos": [1.0], "prev_angle": 3.0}
        prev_frame_infos = [1, 2]

    model = Model()
    reset_vad_state(model)
    assert model.prev_frame_info == {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    assert model.prev_frame_infos == []
