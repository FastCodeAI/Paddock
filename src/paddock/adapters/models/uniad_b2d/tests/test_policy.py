"""The Bench2Drive UniAD's own settings, in paddock's env.

Shared inputs (CAN-bus layout, l2g, lidar2img on Bench2Drive's rig, the six commands) are checked in
uniad/tests/test_bev_planners.py; the forward against Bench2DriveZoo's own path in uniad/tests/test_golden.py.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.uniad.policy import reset_uniad_state
from paddock.adapters.models.uniad_b2d.policy import POLICY


def test_it_reads_bench2drives_cameras_and_lidar() -> None:
    assert POLICY.cameras == bp.B2D_CAMERAS
    assert POLICY.lidar_from_observation is False
    np.testing.assert_array_equal(POLICY.nominal_lidar2ego, bp.B2D_LIDAR2EGO)


def test_reset_gives_the_temporal_state_of_a_fresh_model() -> None:
    class Tracker:
        cleared = False

        def clear(self) -> None:
            self.cleared = True

    class Model:
        prev_frame_info = {"prev_bev": object(), "scene_token": "x", "prev_pos": [1.0], "prev_angle": 3.0}
        prev_frame_infos = [1]
        test_track_instances = object()
        l2g_r_mat = l2g_t = prev_bev = timestamp = scene_token = object()
        track_base = Tracker()

    model = Model()
    reset_uniad_state(model)
    assert model.prev_frame_info == {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    assert model.prev_frame_infos == []
    assert model.test_track_instances is model.l2g_r_mat is model.prev_bev is model.timestamp is None
    assert model.track_base.cleared
