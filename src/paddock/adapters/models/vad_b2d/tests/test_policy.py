"""The Bench2Drive VAD's own inputs, in paddock's env: the six-way command one-hot that picks its plan.

Shared inputs are checked in uniad/tests/test_bev_planners.py; the forward against Bench2DriveZoo's own path
in uniad/tests/test_golden.py.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.vad_b2d.policy import POLICY


def test_the_command_is_the_agents_one_hot_of_six() -> None:
    policy = POLICY()
    for command in range(6):
        np.testing.assert_array_equal(policy._extra_results(command)["ego_fut_cmd"], np.eye(6)[command])


def test_it_reads_bench2drives_cameras_and_lidar() -> None:
    assert POLICY.cameras == bp.B2D_CAMERAS
    assert POLICY.lidar_from_observation is False
