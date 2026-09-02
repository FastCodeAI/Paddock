"""Translating between NAVSIM's conventions and paddock's, with no vendor imports.

Split out of `agent.py` so the part that can silently corrupt a score is
testable in paddock's own environment. `agent.py` needs navsim and nuplan and
runs on python 3.9 inside the vendor's conda env; nothing here imports either,
so `tests/test_navsim_adapter.py` can pin the geometry and the resampling
without a 4 GB checkpoint or a second interpreter.

Same reasoning as CameraFit in the openpilot adapter: the arithmetic that
decides whether a number is right belongs where it can be checked.
"""

from __future__ import annotations

import numpy as np

# The contracts, imported rather than restated. paddock.policy.protocol imports
# on the vendor's python 3.9 (it needs only numpy), so nothing here is a copy
# that can drift from its definition. See docs/reviewing.md §2.
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    encode_image,
)

__all__ = ["encode_image"]  # re-exported for agent.py, which is 3.9-only code

#: NAVSIM's frame spacing and planning grid. Both are fixed by the benchmark:
#: the scorer's simulator expects eight poses over four seconds.
NAVSIM_INTERVAL_S = 0.5
NAVSIM_NUM_POSES = 8

#: paddock camera name -> NAVSIM slot. The nuPlan rig has eight cameras to
#: nuScenes' six; cam_l1/cam_r1 are true side views with no nuScenes equivalent,
#: so nothing maps onto them and no model asks for them.
CAMERA_SLOTS = {
    "CAM_FRONT": "cam_f0",
    "CAM_FRONT_LEFT": "cam_l0",
    "CAM_FRONT_RIGHT": "cam_r0",
    "CAM_BACK": "cam_b0",
    "CAM_BACK_LEFT": "cam_l2",
    "CAM_BACK_RIGHT": "cam_r2",
}

#: NAVSIM's one-hot [left, straight, right, unknown], by position, -> paddock's
#: Command. Built from the enum so a renumbering there cannot leave a stale
#: literal here. Inverse of the mapping in adapters/models/drivejepa/policy.py.
COMMAND_FROM_ONE_HOT = (
    int(Command.LEFT),
    int(Command.STRAIGHT),
    int(Command.RIGHT),
    int(Command.FOLLOW_REFERENCE),
)

#: In OpenScene the lidar frame IS the ego frame -- every log frame carries
#: lidar2ego_translation [0,0,0] and an identity rotation -- so a camera's
#: sensor2lidar extrinsics are already camera2ego. That matters: openpilot and
#: alpamayo read camera2ego[2,3] as the camera's height above the road (1.52 m
#: for CAM_F0) to build their ground-plane warp. Were this not identity, the
#: warp would silently sample the wrong plane.
LIDAR_IS_EGO = True


def pose_matrix(pose) -> np.ndarray:
    """(x, y, heading) -> a 4x4 rigid transform."""
    x, y, heading = float(pose[0]), float(pose[1]), float(pose[2])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0], matrix[0, 1] = cos_h, -sin_h
    matrix[1, 0], matrix[1, 1] = sin_h, cos_h
    matrix[0, 3], matrix[1, 3] = x, y
    return matrix


def canbus(ego_status, yaw_rate: float) -> list[float]:
    """The 18-float nuScenes CAN bus layout paddock's Observation expects.

    NAVSIM has no CAN bus; every field here is reconstructed from EgoStatus.
    Yaw rate is the one quantity EgoStatus does not carry at all, so it is
    differenced from consecutive headings by the caller.
    """
    x, y, heading = (float(v) for v in ego_status.ego_pose)
    velocity = np.asarray(ego_status.ego_velocity, dtype=np.float64).ravel()
    accel = np.asarray(ego_status.ego_acceleration, dtype=np.float64).ravel()
    canbus = [0.0] * 18
    canbus[0:3] = [x, y, 0.0]
    # quaternion about z, w-first, matching nuScenes' ordering
    canbus[3:7] = [np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)]
    # Offsets from the protocol, not literals: these are what every model reads.
    canbus[CANBUS_ACCEL : CANBUS_ACCEL + 3] = [
        float(accel[0]),
        float(accel[1]) if accel.size > 1 else 0.0,
        0.0,
    ]
    canbus[CANBUS_YAW_RATE] = float(yaw_rate)
    canbus[CANBUS_VELOCITY : CANBUS_VELOCITY + 3] = [
        float(velocity[0]),
        float(velocity[1]) if velocity.size > 1 else 0.0,
        0.0,
    ]
    return [float(v) for v in canbus]


def derive_headings(xy: np.ndarray) -> np.ndarray:
    """Heading from the chord between consecutive waypoints.

    Only openpilot, alpamayo and visionpilot need this -- none of the three
    predicts yaw, and PDM orients the ego footprint by heading when it tests
    collisions and drivable-area compliance, so a missing heading is not a
    detail the scorer shrugs off.

    Below a millimetre of travel the chord direction is numerical noise rather
    than a heading. A stopped ego would otherwise be handed a random yaw and
    fail drivable-area checks it should pass, so the last real heading is
    carried forward instead.
    """
    previous = np.vstack([np.zeros((1, 2), dtype=np.float64), xy[:-1]])
    delta = xy - previous
    step = np.hypot(delta[:, 0], delta[:, 1])
    chord = np.arctan2(delta[:, 1], delta[:, 0])

    headings = np.empty(len(xy), dtype=np.float64)
    last = 0.0
    for index in range(len(xy)):
        if step[index] < 1e-3:
            headings[index] = last
        else:
            headings[index] = chord[index]
            last = chord[index]
    return headings


def to_navsim_poses(waypoints, dt: float) -> np.ndarray:
    """A paddock trajectory -> NAVSIM's eight poses at 0.5 s as (x, y, heading).

    Resamples onto NAVSIM's grid rather than assuming a model already plans on
    it. t=0 is prepended as an exact anchor -- the ego is at the origin of its
    own frame -- so interpolation is correct for any waypoint spacing, not just
    0.5 s.

    A model whose horizon is shorter than four seconds raises instead of being
    extrapolated. Extrapolating would fabricate a plan the model never made and
    score it as though it had; refusing names the gap. VisionPilot plans 3 s
    today and will land here until its planner horizon is extended.
    """
    array = np.asarray(waypoints, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"waypoints must be (N, 2) or (N, 3), got {array.shape!r}")
    if not np.isfinite(array).all():
        raise ValueError("trajectory contains NaN or inf")

    times = np.concatenate([[0.0], dt * np.arange(1, len(array) + 1)])
    target = NAVSIM_INTERVAL_S * np.arange(1, NAVSIM_NUM_POSES + 1)
    if target[-1] > times[-1] + 1e-9:
        raise ValueError(
            f"model plans {times[-1]:.2f} s ({len(array)} waypoints at {dt:.2f} s) but NAVSIM "
            f"scores {target[-1]:.1f} s; extend the model's horizon rather than extrapolating"
        )

    xs = np.interp(target, times, np.concatenate([[0.0], array[:, 0]]))
    ys = np.interp(target, times, np.concatenate([[0.0], array[:, 1]]))
    if array.shape[1] == 3:
        # unwrapped so interpolation never crosses the +/-pi seam
        unwrapped = np.unwrap(np.concatenate([[0.0], array[:, 2]]))
        headings = np.interp(target, times, unwrapped)
    else:
        headings = derive_headings(np.stack([xs, ys], axis=1))
    return np.stack([xs, ys, headings], axis=1)
