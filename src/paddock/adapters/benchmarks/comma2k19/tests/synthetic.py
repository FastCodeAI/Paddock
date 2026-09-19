"""Fixtures whose right answer is known by construction.

Shared by `test_harness.py` and `test_overlay.py`, which need the same fake
segment for different reasons -- one to check the maths, the other to check the
drawing. A module rather than `conftest.py` because these are helpers to call,
not fixtures to inject, and because two copies of a synthetic pose is exactly
how two tests end up agreeing with each other and with nothing else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L


def reference_rot_from_quat(quats) -> np.ndarray:
    """Hamilton [w,x,y,z] -> R, written out from the standard expansion.

    The textbook algebra, not a call into the toolkit -- that is the whole
    point. `test_the_toolkit_rotations_match_an_independent_reference` asserts
    the toolkit reproduces this, and `conftest.rotations` stands this in for the
    toolkit where no clone exists, so the two uses hold each other honest: if
    this drifts, the agreement test fails at any site that has a clone.

    Accepts a single quaternion or an (N, 4) stack, like the toolkit's.
    """
    q = np.asarray(quats, dtype=np.float64)
    flat = np.atleast_2d(q)
    w, x, y, z = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    r = np.empty((flat.shape[0], 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r[0] if q.ndim == 1 else r


def reference_rot_from_euler(rpy) -> np.ndarray:
    """[roll, pitch, yaw] -> Rz(yaw) @ Ry(pitch) @ Rx(roll).

    openpilot's calibration order, written out rather than delegated. See
    `reference_rot_from_quat` for why this lives here.
    """
    roll, pitch, yaw = (float(v) for v in np.asarray(rpy, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


#: device [forward, right, down] <- view [right, down, forward]. The toolkit's
#: `camera.device_frame_from_view_frame`, transcribed so the stand-in can carry
#: it; `test_the_camera_model_paddock_holds_locally_still_matches_the_toolkit`
#: is the drift check for the literals in `loader.py` and this one moves with it.
REFERENCE_DEVICE_FROM_VIEW = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def rot2quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> Hamilton [w, x, y, z]. The toolkit's is unrunnable.

    `utils/orientation.rot2quat` needs `xrange` and a numpy that accepts an
    array in a scalar slot; `toolkit.py` has the detail. This is only ever fed
    the proper rotations built here, so the trace branch is enough.
    """
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    q = np.array(
        [
            w,
            (r[2, 1] - r[1, 2]) / (4 * w),
            (r[0, 2] - r[2, 0]) / (4 * w),
            (r[1, 0] - r[0, 1]) / (4 * w),
        ]
    )
    return q if q[0] >= 0 else -q


def straight_pose(n: int = 200, speed: float = 20.0, hz: float = 20.0) -> L.SegmentPose:
    """A car driving straight east at constant speed, at the equator.

    ECEF at (lat 0, lon 0) has x out of the ground, y east, z north -- so
    "driving east" is +y in ECEF, and the device frame [forward, right, down]
    maps forward->+y_ecef, right->-z_ecef, down->-x_ecef.
    """
    t = np.arange(n) / hz
    pos = np.zeros((n, 3))
    pos[:, 0] = 6378137.0
    pos[:, 1] = speed * t
    vel = np.zeros((n, 3))
    vel[:, 1] = speed
    ecef_from_device = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    # rot2quat of that matrix, by hand: it is a 120 deg rotation about
    # (1,-1,-1)/sqrt(3); w = 0.5, and the vector part follows.
    quat = np.tile(rot2quat(ecef_from_device), (n, 1))
    return L.SegmentPose(times=t, positions=pos, orientations=quat, velocities=vel)


def write_segment(seg: Path, pose: L.SegmentPose) -> Path:
    """Write `pose` where `loader.load_pose` will find it, and return the dir.

    comma2k19's arrays are extension-less files that happen to be .npy, so the
    save-and-rename is not a quirk of the test -- it is the layout the loader
    reads, and writing `frame_times.npy` here would make the fixture pass
    against a loader that could not read the real dataset.
    """
    (seg / "global_pose").mkdir(parents=True, exist_ok=True)
    for name, array in (
        ("frame_times", pose.times),
        ("frame_positions", pose.positions),
        ("frame_orientations", pose.orientations),
        ("frame_velocities", pose.velocities),
    ):
        np.save(seg / "global_pose" / name, array)
        (seg / "global_pose" / name).with_suffix(".npy").rename(seg / "global_pose" / name)
    return seg


def reset_pose_cache(pose: L.SegmentPose) -> L.SegmentPose:
    """Drop what `cached_property` derived from `orientations` before they moved."""
    for attr in ("ecef_from_device", "device_from_ecef", "velocity_device", "yaw_rate"):
        pose.__dict__.pop(attr, None)
    return pose
