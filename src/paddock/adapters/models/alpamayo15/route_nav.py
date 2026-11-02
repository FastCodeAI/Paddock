# Portions adapted from NVIDIA AlpaSim (https://github.com/NVlabs/alpasim),
# src/driver/src/alpasim_driver/models/nav_text.py and models/alpamayo_base.py
# (`route_in_rig_at_t0`), Copyright (c) 2026 NVIDIA Corporation, Apache-2.0.
"""The navigation text AlpaSim's own Alpamayo driver builds from its route.

On AlpaSim the model's route instruction comes from the route polyline, not
from a discrete command: AlpaSim's driver moves the route into the rig frame of
the newest pose and classifies the bend ahead by heading change. This module is
that classifier and that move, ported number for number, so the adapter shows
the model the same "Turn left in 23m" the native driver did. Every other
benchmark keeps `policy.nav_text_for`, whose phrasing differs at the edges
(it can say "in 0m"; this never says less than "in 1m").

Turns are classified by heading change rather than lateral offset: a fixed
lateral threshold means ~24 degrees of heading at 5 m but only ~3 degrees at
40 m, so it reads gentle drift and parallel lane offsets as turns up close and
misses real turns further out.
"""

from __future__ import annotations

import numpy as np

#: Shortest segment that still defines a bearing. Resampled routes can repeat
#: their last waypoint as padding.
MIN_SEGMENT_LENGTH_M = 1e-3
#: Turns nearer than this are ignored; the ego is already in them.
MIN_LOOKAHEAD_M = 5.0
#: Heading change that separates a turn from lane curvature.
TURN_ANGLE_DEG = 30.0
#: Turns beyond this are ignored; a later call reports them as the ego nears.
DISTANCE_LOOKAHEAD_M = 40.0


def route_in_frame(
    route_xyz: np.ndarray, frame_to_world: np.ndarray, route_frame_to_world: np.ndarray
) -> np.ndarray:
    """Route points expressed in *route_frame_to_world*, re-expressed in *frame_to_world*.

    AlpaSim sends the route in the rig frame at the route's own timestamp; its
    driver re-expresses it in the rig frame at the newest pose before
    classifying it. Both poses are 4x4 rig-to-world. NaN rows stay NaN.
    """
    relative = np.linalg.inv(frame_to_world) @ route_frame_to_world
    return route_xyz @ relative[:3, :3].T + relative[:3, 3]


def route_to_nav_text(route_in_rig: np.ndarray) -> str | None:
    """Classify the route ahead as a left turn, a right turn, or straight.

    *route_in_rig* is (N, 2) or (N, 3) in the rig frame (+x forward, +y left);
    rows may be NaN. Returns "Turn left in {N}m", "Turn right in {N}m" or
    "Continue straight", or None if the route holds no usable waypoint, in
    which case the model runs without a route instruction.
    """
    route_in_rig = np.asarray(route_in_rig, dtype=np.float64)
    if route_in_rig.size == 0:
        return None

    xy = route_in_rig[:, :2]
    points = xy[~np.isnan(xy).any(axis=1)]
    if len(points) == 0:
        return None

    if len(points) >= 2:
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > MIN_SEGMENT_LENGTH_M
        points = points[keep]

    # Measuring a heading change needs two segments. With one, the route is
    # indistinguishable from a straight road at an angle to the ego.
    if len(points) < 3:
        return "Continue straight"

    distances = np.linalg.norm(points, axis=1)

    # The initial road heading is the chord across the near zone rather than
    # the first segment alone, so a constant yaw offset between ego and road
    # does not read as a turn.
    near = np.flatnonzero(distances <= MIN_LOOKAHEAD_M)
    start, end = (near[0], near[-1]) if len(near) >= 2 else (0, 1)
    base = points[end] - points[start]
    if np.linalg.norm(base) < MIN_SEGMENT_LENGTH_M:
        base = points[1] - points[0]
    initial_heading = np.arctan2(base[1], base[0])

    # Segment k runs points[k] -> points[k + 1]; a turn is attributed to the far
    # endpoint, where the heading change has been realised.
    segments = np.diff(points, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    deviations = np.remainder(headings - initial_heading + np.pi, 2 * np.pi) - np.pi
    endpoint_distances = distances[1:]

    turning = (
        (endpoint_distances >= MIN_LOOKAHEAD_M)
        & (endpoint_distances <= DISTANCE_LOOKAHEAD_M)
        & (np.abs(deviations) > np.radians(TURN_ANGLE_DEG))
    )
    if not turning.any():
        return "Continue straight"

    index = int(np.argmax(turning))
    direction = "left" if deviations[index] > 0 else "right"
    return f"Turn {direction} in {max(1, round(endpoint_distances[index]))}m"
