"""The KITScenes route hint reproduces the reference Drive-JEPA `--command gt` rule."""

import numpy as np

from paddock.adapters.benchmarks.kitscenes.harness import ROUTE_LOOKAHEAD_FRAMES, route_hint
from paddock.policy.protocol import Command


def _poses(path_xy):
    xy = np.asarray(path_xy, dtype=float)
    yaw = np.zeros(len(xy))
    return np.column_stack([xy, yaw])


def test_straight_road_is_straight_with_no_maneuver():
    poses = _poses([(x, 0.0) for x in range(0, 120)])  # 1 m per frame, heading +x
    assert route_hint(poses, 10) == (Command.STRAIGHT, None, None)


def test_left_and_right_offsets_beyond_5m_are_turns_with_distance():
    n = ROUTE_LOOKAHEAD_FRAMES
    left = _poses([(i, 0.0) for i in range(20)] + [(20 + i, 0.3 * i) for i in range(1, n + 20)])
    cmd, man, dist = route_hint(left, 10)
    assert (cmd, man) == (Command.LEFT, "TURN_LEFT") and dist is not None and dist > 0
    right = _poses([(i, 0.0) for i in range(20)] + [(20 + i, -0.3 * i) for i in range(1, n + 20)])
    assert route_hint(right, 10)[:2] == (Command.RIGHT, "TURN_RIGHT")


def test_lookahead_is_clamped_at_the_end_of_the_log():
    poses = _poses([(x, 0.0) for x in range(0, 30)])
    assert route_hint(poses, 29) == (Command.STRAIGHT, None, None)
    assert route_hint(poses, 5)[0] == Command.STRAIGHT
