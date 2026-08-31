"""Fixture test for the template adapter.

Every model adapter ships one of these. It must run in paddock's own
environment -- no GPU, no dataset, no model weights -- because it is what a
reviewer can actually execute on a PR.

It cannot check that the model is good. It checks that the adapter obeys the
contract: right frame, right shape, no NaNs, and a plan that is not accidentally
an emergency stop.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.models._template.policy import POLICY
from paddock.policy.protocol import N_WAYPOINTS, Command, Observation


def make_observation(speed: float = 10.0) -> Observation:
    """A minimal synthetic observation. Replace with a recorded one where you can."""
    return Observation(
        images={"CAM_FRONT": np.zeros((256, 512, 3), dtype=np.uint8)},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={"CAM_FRONT": np.array([[500.0, 0, 256.0], [0, 500.0, 128.0], [0, 0, 1.0]])},
        camera2ego={"CAM_FRONT": np.eye(4)},
    )


def test_infer_obeys_the_contract() -> None:
    policy = POLICY()
    policy.load()
    policy.reset()

    obs = make_observation(speed=10.0)
    obs.canbus[13] = 10.0  # forward velocity

    traj = policy.infer(obs)

    assert traj.waypoints.shape[1] in (2, 3), "waypoints must be (N,2) or (N,3)"
    assert len(traj.waypoints) >= N_WAYPOINTS, f"closed-loop needs >= {N_WAYPOINTS} waypoints"
    assert np.isfinite(traj.waypoints).all(), "waypoints must be finite"
    assert traj.xy[:, 0].max() > 0, "x is forward; a moving ego should plan ahead of itself"
    assert not np.allclose(traj.xy, 0.0), "all-zero means emergency stop, not 'no plan'"


def test_reset_is_idempotent() -> None:
    """Two resets then one infer must equal one reset then one infer."""
    policy = POLICY()
    policy.load()

    obs = make_observation()
    obs.canbus[13] = 10.0

    policy.reset()
    first = policy.infer(obs)
    policy.reset()
    policy.reset()
    second = policy.infer(obs)

    np.testing.assert_allclose(first.waypoints, second.waypoints)
