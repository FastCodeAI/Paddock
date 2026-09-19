"""Fixture test for the template adapter.

Every model adapter ships one of these. It must run in paddock's own
environment -- no GPU, no dataset, no model weights -- because it is what a
reviewer can actually execute on a PR.

It cannot check that the model is good. It checks that the adapter obeys the
contract, and most of that is delegated to `paddock.conformance`, which is the
same set of checks `paddock verify --model <name>` runs. Keep
`test_conformance` and add adapter-specific tests beside it; do not re-write the
generic ones by hand.
"""

from __future__ import annotations

import numpy as np

from paddock import conformance
from paddock.adapters.models._template.policy import POLICY
from paddock.policy.protocol import CANBUS_VELOCITY, N_WAYPOINTS, Command, Observation


def make_observation(speed: float = 10.0) -> Observation:
    """A minimal synthetic observation. Replace with a recorded one where you can.

    Note `camera2ego`: it is **not** the identity. Paddock's camera frame is
    [right, down, forward] and its ego frame is [forward, left, up], so a camera
    with no mounting rotation is

        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

    and the identity means "the camera's axes are the ego's axes" -- a 90 degree
    error. That exact mistake, written in a benchmark loader that had copied it
    from here, sent openpilot a sideways image on comma2k19 and produced a
    published ADE of 38.8 m that was read as a modelling result for weeks.
    `conformance.check_camera_mounting` now rejects it, and
    `conformance.synthetic_observation()` builds this correctly if you would
    rather not write it out.
    """
    camera2ego = np.eye(4)
    camera2ego[:3, :3] = conformance.UNROTATED_CAMERA2EGO
    camera2ego[2, 3] = 1.5  # camera height above the ground, metres
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={"CAM_FRONT": np.zeros((256, 512, 3), dtype=np.uint8)},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={"CAM_FRONT": np.array([[500.0, 0, 256.0], [0, 500.0, 128.0], [0, 0, 1.0]])},
        camera2ego={"CAM_FRONT": camera2ego},
    )


def test_conformance() -> None:
    """The shared Policy contract. Equivalent to `paddock verify --model _template`.

    Failures name the check and explain why it exists, so read the output rather
    than the assertion.
    """
    findings = conformance.policy_findings(POLICY())
    failures = [f for f in findings if f.failed]
    assert not failures, "\n" + "\n".join(f.render() for f in failures)


def test_infer_obeys_the_contract() -> None:
    """The same properties spelled out, for a reader who wants them explicit."""
    policy = POLICY()
    policy.load()
    policy.reset()

    traj = policy.infer(make_observation(speed=10.0))

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

    policy.reset()
    first = policy.infer(obs)
    policy.reset()
    policy.reset()
    second = policy.infer(obs)

    np.testing.assert_allclose(first.waypoints, second.waypoints)
