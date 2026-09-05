"""Estimating comma2k19's camera mount rotation, the way openpilot does.

openpilot's driving model never sees a raw frame. `modeld` warps every frame
with `get_warp_matrix(rpyCalib, intrinsics, ...)`, where `rpyCalib` comes from
`selfdrive/locationd/calibrationd.py` and describes how the phone is mounted.
The *calibrated* frame is defined (in openpilot's
`common/transformations/README.md`) as aligned with the car in pitch and yaw and
with the device in roll -- so the model always sees the road from the same
apparent viewpoint no matter how the phone sits.

comma2k19 has no `liveCalibration` stream, so it has to be estimated. This
module reimplements calibrationd's formula with one deliberate substitution.

## The substitution

calibrationd derives its estimate from `cameraOdometry`, which is **the model's
own `pose` output**. Offline that is circular: the calibration would depend on
the model under evaluation, and a model that mispredicts its own direction of
travel would be handed a warp that hides the error. comma2k19 ships something
strictly better -- `global_pose/frame_velocities`, an INS/GNSS/vision-fused ECEF
velocity per frame -- so `trans` comes from ground truth. The gating conditions,
the residual formula, and the rotation composition are calibrationd's:

    straight_and_fast = v > 15 mph and trans[0] > 15 mph and |yaw rate| < 2 deg/s
    observed_rpy      = [0, -atan2(trans[2], trans[0]), atan2(trans[1], trans[0])]
    new_rpy           = euler_from_rot(rot_from_euler(rpy) @ rot_from_euler(observed))

In openpilot `trans` is expressed in the *calibrated* frame (the model predicts
in the frame it was shown), which is what makes `observed_rpy` a shrinking
residual and the composition convergent. Ground-truth velocity is in the
*device* frame, so it is rotated into the current calibrated frame each
iteration; without that the residual would be constant and the composition would
run away. The fixed point is the one openpilot converges to.

## Why the estimate pools a whole route

All the one-minute segments of a route were recorded in one drive with one
phone mounting, so their qualifying frames are samples of the same quantity.
Pooling them is the openpilot equivalent of having a persisted
`CalibrationParams` to load, and it is required for correctness rather than
being a refinement: in a 220-segment sample, **5 segments have no frame at all
that passes the `straight_and_fast` gate** -- stop-and-go traffic where the car
never holds 15 mph straight. Estimating per segment leaves those at `rpy = 0`,
silently, and a zero calibration costs roughly 3x in cross-track error.
openpilot never hits this because it does not forget its calibration when the
car slows down.

When a whole route has no qualifying frames, `Calibration.n_valid` is 0 and the
caller can see it rather than inheriting a plausible-looking zero.

## Sanity, expressed as a guard rather than a hope

`plausible` rejects a mount rotation no windscreen bracket could produce. The
previous implementation's `rpy = [-90, 0, +90] deg` would fail it, which is the
point: this class of defect must be loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19.loader import (
    SegmentPose,
    euler_from_rot,
    load_pose,
    rot_from_euler,
)

__all__ = ["Calibration", "RouteCalibrations", "estimate", "plausible"]

#: calibrationd.MIN_SPEED_FILTER = 15 * CV.MPH_TO_MS
MIN_SPEED_FILTER = 15 * 0.44704
#: calibrationd.MAX_YAW_RATE_FILTER = radians(2)
MAX_YAW_RATE_FILTER = np.radians(2.0)

#: How far from level a real windscreen mount can plausibly be, degrees.
#: Wider than openpilot's own PITCH/YAW limits, which are tici-specific and
#: exist to keep the model frame inside a different sensor. The job of this
#: bound is only to catch a coordinate-convention error, and a 90 degree
#: mistake is two orders of magnitude outside it.
MAX_PLAUSIBLE_DEG = 15.0


@dataclass(frozen=True)
class Calibration:
    """An `rpyCalib` estimate plus enough context to judge it."""

    rpy: np.ndarray
    n_valid: int
    n_total: int
    spread: np.ndarray
    scope: str

    @property
    def rpy_deg(self) -> np.ndarray:
        return np.degrees(self.rpy)

    @property
    def is_valid(self) -> bool:
        return self.n_valid > 0 and plausible(self.rpy)

    def as_dict(self) -> dict:
        return {
            "rpy": self.rpy.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "spread_deg": np.degrees(self.spread).tolist(),
            "scope": self.scope,
            "plausible": plausible(self.rpy),
        }


def plausible(rpy) -> bool:
    """Whether `rpy` could be a real camera mounting.

    Used as a guard, not a filter: a caller that gets False should say so
    loudly. The failure this exists for -- a coordinate-convention slip
    producing a 90 degree rotation -- is invisible in a trajectory score.
    """
    return bool(np.all(np.abs(np.degrees(np.asarray(rpy, dtype=np.float64))) <= MAX_PLAUSIBLE_DEG))


def _qualifying(pose: SegmentPose) -> np.ndarray:
    """calibrationd's `straight_and_fast` gate, on recorded pose."""
    trans = pose.velocity_device
    return (
        (pose.speed > MIN_SPEED_FILTER)
        & (trans[:, 0] > MIN_SPEED_FILTER)
        & (np.abs(pose.yaw_rate) < MAX_YAW_RATE_FILTER)
        & np.isfinite(trans).all(axis=1)
    )


def _observed_rpy(trans_calib: np.ndarray) -> np.ndarray:
    """calibrationd's residual, verbatim."""
    return np.array(
        [
            0.0,
            -np.arctan2(trans_calib[2], trans_calib[0]),
            np.arctan2(trans_calib[1], trans_calib[0]),
        ]
    )


def _fixed_point(trans_device: np.ndarray, n_total: int, scope: str) -> Calibration:
    rpy = np.zeros(3)
    if len(trans_device) == 0:
        return Calibration(rpy, 0, n_total, np.zeros(3), scope)
    per_frame = np.zeros((len(trans_device), 3))
    for _ in range(8):
        calib_from_device = rot_from_euler(rpy).T
        trans_calib = trans_device @ calib_from_device.T
        per_frame = np.array(
            [euler_from_rot(rot_from_euler(rpy) @ rot_from_euler(_observed_rpy(t))) for t in trans_calib]
        )
        new_rpy = per_frame.mean(axis=0)
        if np.allclose(new_rpy, rpy, atol=1e-9):
            rpy = new_rpy
            break
        rpy = new_rpy
    return Calibration(rpy, len(trans_device), n_total, per_frame.max(axis=0) - per_frame.min(axis=0), scope)


def estimate(segment_paths: list[Path] | list[str], scope: str = "route") -> Calibration:
    """Pooled `rpyCalib` over the given segment directories."""
    samples: list[np.ndarray] = []
    n_total = 0
    for path in segment_paths:
        try:
            pose = load_pose(Path(path))
        except Exception:  # noqa: BLE001 -- one broken sibling must not lose the route
            continue
        n_total += len(pose)
        mask = _qualifying(pose)
        if mask.any():
            samples.append(pose.velocity_device[mask])
    pooled = np.concatenate(samples) if samples else np.zeros((0, 3))
    return _fixed_point(pooled, n_total, scope)


class RouteCalibrations:
    """Memoised route-level calibrations, keyed by (chunk, route_id).

    A route can hold ~70 segments and each of its segments would otherwise
    re-pool the whole route; loading one segment's pose arrays is cheap but
    doing it 70^2 times is not.
    """

    def __init__(self, segments) -> None:
        self._by_route: dict[tuple[int, str], list] = {}
        for s in segments:
            self._by_route.setdefault(s.route_key, []).append(s)
        self._cache: dict[tuple[int, str], Calibration] = {}

    def get(self, segment) -> Calibration:
        key = segment.route_key
        if key not in self._cache:
            siblings = self._by_route.get(key) or [segment]
            self._cache[key] = estimate([s.path for s in siblings])
        return self._cache[key]

    def summary(self) -> dict:
        """Per-route estimates, for a run's `metrics.json` and for eyeballing.

        Mount rotation is a property of the *rig*, so consistency within a
        vehicle is itself evidence the estimate is right: on this dataset the
        RAV4 routes cluster at 3.5-3.8 deg of pitch and the Civic routes at
        1.7-1.8 deg.
        """
        return {f"{chunk}/{route}": cal.as_dict() for (chunk, route), cal in sorted(self._cache.items())}
