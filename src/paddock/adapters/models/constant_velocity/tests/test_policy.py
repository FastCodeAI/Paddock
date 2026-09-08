"""Contract tests for the constant-velocity baseline. No GPU, no weights."""

from __future__ import annotations

import numpy as np

from paddock.adapters.models.constant_velocity.policy import POLICY
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation, Policy


def _obs(speed: float) -> Observation:
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )


def test_policy_imports_and_declares_no_cameras() -> None:
    p = POLICY()
    assert isinstance(p, Policy)
    assert p.cameras == ()
    assert p.waypoint_dt == 0.5


def test_projects_straight_at_the_current_speed() -> None:
    traj = POLICY().infer(_obs(10.0))  # 10 m/s
    assert traj.waypoints.shape[0] == 6
    # x forward advances by speed * dt each step; y stays ~0 (straight)
    assert abs(traj.waypoints[0, 0] - 5.0) < 1e-6  # 10 m/s * 0.5 s
    assert abs(traj.waypoints[-1, 0] - 30.0) < 1e-6  # 6 * 5 m
    assert np.allclose(traj.waypoints[:, 1], 0.0)


def test_stationary_stays_put() -> None:
    traj = POLICY().infer(_obs(0.0))
    assert np.allclose(traj.waypoints[:, :2], 0.0)
