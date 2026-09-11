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
