"""The next lane change on the route, without CARLA.

Route 17598 (SequentialLaneChange) is the layout used throughout: two
``CHANGELANELEFT`` points at 32 m and two more at 67 m, on a 1 m-spaced route.
Before the agent sent these, both lane changes reached the model as STRAIGHT.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.benchmarks.bench2drive.navigation import LANE_CHANGE_HORIZON_M, lane_change_ahead

OPTIONS = (
    ["LANEFOLLOW"] * 32
    + ["CHANGELANELEFT"] * 2
    + ["LANEFOLLOW"] * 33
    + ["CHANGELANELEFT"] * 2
    + ["LANEFOLLOW"] * 25
)
S = np.arange(len(OPTIONS), dtype=float)


def test_the_first_lane_change_is_reported_with_its_side_and_distance() -> None:
    assert lane_change_ahead(OPTIONS, S, 0) == ("LEFT", 32.0)
    assert lane_change_ahead(OPTIONS, S, 20) == ("LEFT", 12.0)


def test_a_lane_change_reads_zero_while_the_ego_is_on_it_then_the_next_one_is_reported() -> None:
    assert lane_change_ahead(OPTIONS, S, 33) == ("LEFT", 0.0)
    assert lane_change_ahead(OPTIONS, S, 34) == ("LEFT", 33.0), "past the first: the second, 67 m along"
    assert lane_change_ahead(OPTIONS, S, 69) == (None, None), "past the last: nothing ahead"


def test_the_side_follows_the_road_option() -> None:
    right = ["LANEFOLLOW"] * 10 + ["CHANGELANERIGHT"] + ["LANEFOLLOW"] * 5
    assert lane_change_ahead(right, np.arange(len(right), dtype=float), 3) == ("RIGHT", 7.0)


def test_junction_maneuvers_are_not_lane_changes() -> None:
    """LEFT/RIGHT/STRAIGHT are junction maneuvers, which ``nav_maneuver`` already carries."""
    junction = ["LANEFOLLOW"] * 25 + ["RIGHT"] * 13 + ["STRAIGHT"] * 5 + ["LEFT"] * 5
    assert lane_change_ahead(junction, np.arange(len(junction), dtype=float), 0) == (None, None)


def test_a_lane_change_beyond_the_horizon_is_not_reported_yet() -> None:
    far = ["LANEFOLLOW"] * 150 + ["CHANGELANELEFT"]
    s = np.arange(len(far), dtype=float)
    assert lane_change_ahead(far, s, 0) == (None, None)
    assert lane_change_ahead(far, s, int(150 - LANE_CHANGE_HORIZON_M)) == ("LEFT", LANE_CHANGE_HORIZON_M)


def test_no_route_or_no_position_is_no_lane_change() -> None:
    assert lane_change_ahead([], [], 0) == (None, None)
    assert lane_change_ahead(OPTIONS, S, None) == (None, None)
