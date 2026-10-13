"""A 32-path swerve vocabulary for Drive-JEPA, built only from what the model has at drive time.

    paths, meta = vocabulary(natives, pick_index, speed, accel)

Inputs: the 32 native plans (32, 8, 3), the index of the released pick (argmax pdm_score), ego
speed (m/s) and ego acceleration (m/s^2; accepted for the interface, unused). No map, lane,
drivable area or actor is read. Output: (32, 8, 3) plans at t = 0.5 .. 4.0 s, (x forward, y left,
heading) in the ego rear-axle frame, and one dict per path.

Design (measured on round0_v5 renders, see the review report):

* Two bases driven at the CURRENT speed. Rescue ticks happen after the released model has
  already braked (median 3.1 m/s), and a car can only move sideways as fast as it moves forward:
  a 2.5 m swerve on a braking base needs a turn radius of about 1 m. Holding speed is what makes
  a drivable swerve exist at all.
    - "pick":     the pick's own path (its heading channel gives the direction), at current speed,
                  continued straight past its end.
    - "straight": straight ahead along the ego heading, at current speed. Robust to a pick whose
                  shape is noise at low speed.
* Offsets of +-1.0 .. 4.0 m in 0.5 m steps, applied NORMAL to the base (not along ego y), both
  sides: the safe side flips with the country (Singapore oncoming lane is on the right, Boston on
  the left) and nothing the model has at drive time says which country it is in.
* Plus "brake": the pick's path decelerating at 3 m/s^2 from current speed, nudged +-1.0 / 1.5 m.
  This is the slot pair that covers stationary obstacles when no native brake is safe.
* Lateral profile: a cosine S-curve that moves over and STAYS (no return: a return is almost
  never safe against an oncoming car and, as previously drawn, was undrivable). It advances with
  distance travelled, with the ramp as short as two limits allow: 4.0 m/s^2 lateral accel and a
  5 m turning radius at the base speed. Every path is drivable by construction; meta reports the
  peak added lateral acceleration and the tightest added turn radius of each path.
"""

from __future__ import annotations

import numpy as np

DT = 0.5
T = np.arange(1, 9) * DT
FINE_DT = 0.05
TF = np.round(np.arange(1, 81) * FINE_DT, 3)  # 0.05 .. 4.0 s
WP = np.arange(10, 81, 10) - 1  # indices of 0.5 .. 4.0 s in TF
A_LAT_MAX = 4.0  # m/s^2
MIN_RADIUS = 5.0  # m
BRAKE_DECEL = 3.0  # m/s^2
STILL_M = 0.05

OFFSETS = (1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5, 4.0, -4.0)
LAYOUT = (("pick", OFFSETS), ("straight", OFFSETS), ("brake", (1.0, -1.0, 1.5, -1.5)))


def _ease(u):
    return (1 - np.cos(np.pi * np.clip(u, 0.0, 1.0))) / 2


def _geometry(pick):
    """The pick as a curve parametrised by arc length: (arc knots, xy knots, heading knots)."""
    poly = np.vstack([[0.0, 0.0], pick[:, :2]])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] < 0.5:  # a stopped pick has no direction of its own: straight ahead
        return _straight()
    head = np.unwrap(np.r_[0.0, pick[:, 2]])
    keep = np.r_[True, seg > 1e-3]
    poly, arc, head = poly[keep], arc[keep], head[keep]
    far = poly[-1] + 1e4 * np.array([np.cos(head[-1]), np.sin(head[-1])])
    return np.r_[arc, arc[-1] + 1e4], np.vstack([poly, far]), np.r_[head, head[-1]]


def _straight():
    return np.array([0.0, 1e4]), np.array([[0.0, 0.0], [1e4, 0.0]]), np.array([0.0, 0.0])


def _distance(kind, v0):
    if kind == "brake":
        tt = np.minimum(TF, v0 / BRAKE_DECEL)
        return v0 * tt - 0.5 * BRAKE_DECEL * tt**2
    return v0 * TF


def _ramp_length(offset, v_max):
    """Shortest distance over which a cosine S-curve moves `offset` sideways, under both limits.

    Peak curvature of the S-curve is pi^2 |offset| / (2 L^2). The turning-radius limit is purely
    geometric; the lateral-acceleration limit (curvature * v^2) binds at the highest speed on the
    ramp, which for every base here is the current speed (they hold speed or slow down).
    """
    by_radius = np.pi * np.sqrt(abs(offset) * MIN_RADIUS / 2)
    by_accel = v_max * np.pi * np.sqrt(abs(offset) / (2 * A_LAT_MAX))
    return max(by_radius, by_accel, 1e-3)


def _lateral(offset, s, v_max):
    """Cosine move-over-and-stay, advancing with distance travelled (a car cannot crab)."""
    length = _ramp_length(offset, v_max)
    ramp_s = float(np.interp(length, s, TF)) if s[-1] >= length else float("inf")
    return offset * _ease(s / length), ramp_s


def _waypoints(fine_xy):
    """(P, 80, 2) positions on the fine grid -> (P, 8, 3) with the tangent heading at each waypoint."""
    full = np.concatenate([np.zeros((len(fine_xy), 1, 2)), fine_xy], 1)
    d = np.gradient(full, FINE_DT, axis=1)
    moving = np.linalg.norm(d, axis=-1) * FINE_DT > STILL_M / 5
    h = np.arctan2(d[..., 1], d[..., 0])
    h[:, 0] = np.where(moving[:, 0], h[:, 0], 0.0)
    moving[:, 0] = True
    idx = np.maximum.accumulate(np.where(moving, np.arange(full.shape[1])[None], 0), axis=1)
    h = np.unwrap(np.take_along_axis(h, idx, 1), axis=1)
    return np.concatenate([full[:, 1:][:, WP], h[:, 1:][:, WP][..., None]], -1)


def _added_metrics(s, y):
    """Peak lateral acceleration (curvature * v^2, m/s^2) and tightest turn radius (m) of the offset
    manoeuvre relative to its base, on the fine grid."""
    s, y = np.r_[0.0, s], np.r_[0.0, y]
    v = np.gradient(s, FINE_DT)
    moving = v > 0.3
    dyds = np.gradient(y, FINE_DT) / np.maximum(v, 1e-6)
    d2 = np.gradient(dyds, FINE_DT) / np.maximum(v, 1e-6)
    kappa = np.where(moving, np.abs(d2) / (1 + dyds**2) ** 1.5, 0.0)[2:-2]
    acc = float((kappa * v[2:-2] ** 2).max())
    return acc, float(1 / max(kappa.max(), 1e-9))


def vocabulary(natives, pick_index, speed, accel=0.0):
    """32 swerve plans from the released pick and the ego speed. `accel` is accepted and unused."""
    natives = np.asarray(natives, dtype=float)
    pick = natives[int(pick_index)]
    v0 = max(float(speed), 0.0)
    geom_pick = _geometry(pick)
    fines, meta = [], []
    for kind, offsets in LAYOUT:
        arc, xy_k, head_k = _straight() if kind == "straight" else geom_pick
        s = _distance(kind, v0)
        xy = np.stack([np.interp(s, arc, xy_k[:, 0]), np.interp(s, arc, xy_k[:, 1])], -1)
        h = np.interp(s, arc, head_k)
        normal = np.stack([-np.sin(h), np.cos(h)], -1)
        for off in offsets:
            lat, ramp = _lateral(off, s, v0)
            fines.append(xy + lat[:, None] * normal)
            acc, radius = _added_metrics(s, lat)
            meta.append(
                {
                    "base": kind,
                    "offset_m": off,
                    "ramp_s": round(ramp, 2),  # inf: still ramping at 4 s (low speed)
                    "peak_lat_accel": round(acc, 2),
                    "min_radius_m": round(radius, 1),
                }
            )
    return _waypoints(np.stack(fines)), meta
