"""decisions.py's geometry-only lateral/longitudinal classification.

Waypoints are ego-frame [x, y] at t = dt, 2*dt, ... (decisions.py's DT = 0.5s),
with the origin implicitly prepended as the sample's current pose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "models" / "alpamayo15"))

from decisions import classify_lateral, classify_longitudinal, geometry_features, label  # noqa: E402


def _waypoints_from_steps(dx: list[float], dy: list[float]) -> list[list[float]]:
    assert len(dx) == len(dy)
    xs, ys, x, y = [], [], 0.0, 0.0
    for ddx, ddy in zip(dx, dy, strict=True):
        x, y = x + ddx, y + ddy
        xs.append(x)
        ys.append(y)
    return [[x, y] for x, y in zip(xs, ys, strict=True)]


def _constant_lateral_drift(dy_per_step: float, dx_per_step: float = 2.0, n: int = 6) -> list[list[float]]:
    return _waypoints_from_steps([dx_per_step] * n, [dy_per_step] * n)


def test_straight_constant_speed() -> None:
    waypoints = _constant_lateral_drift(dy_per_step=0.0)
    lat, lon, feats = label(waypoints)
    assert lat == "straight"
    assert lon == "keep_speed"
    assert feats["lateral_offset_m"] == 0.0


def test_turn_left_from_lateral_offset() -> None:
    waypoints = _constant_lateral_drift(dy_per_step=0.4)  # final y = 2.4m >= LATERAL_TURN_M
    lat, _, feats = label(waypoints)
    assert lat == "turn_left"
    assert feats["lateral_offset_m"] > 2.0


def test_turn_right_from_lateral_offset() -> None:
    waypoints = _constant_lateral_drift(dy_per_step=-0.4)  # final y = -2.4m
    lat, _, _ = label(waypoints)
    assert lat == "turn_right"


def test_lane_change_below_turn_threshold() -> None:
    waypoints = _constant_lateral_drift(dy_per_step=0.2)  # final y = 1.2m, in [1.0, 2.0)
    lat, _, _ = label(waypoints)
    assert lat == "lane_change"


def test_nudge_below_lane_change_threshold() -> None:
    waypoints = _constant_lateral_drift(dy_per_step=0.1)  # final y = 0.6m, in [0.3, 1.0)
    lat, _, _ = label(waypoints)
    assert lat == "nudge"


def test_stop_when_final_speed_drops_below_threshold() -> None:
    # speeds (m/s): 5, 4, 3, 2, 1, 0.2 -- vf < STOP_SPEED_MPS (0.5)
    speeds = [5.0, 4.0, 3.0, 2.0, 1.0, 0.2]
    dx = [v * 0.5 for v in speeds]  # DT = 0.5s
    waypoints = _waypoints_from_steps(dx, [0.0] * len(dx))
    _, lon, feats = label(waypoints)
    assert lon == "stop"
    assert feats["vf"] < 0.5


def test_speed_up_when_final_speed_exceeds_initial_by_threshold() -> None:
    speeds = [2.0, 2.8, 3.6, 4.4, 5.2, 6.0]  # v0=2.0, vf=6.0, delta=4.0 >= SPEED_DELTA_MPS
    dx = [v * 0.5 for v in speeds]
    waypoints = _waypoints_from_steps(dx, [0.0] * len(dx))
    _, lon, feats = label(waypoints)
    assert lon == "speed_up"
    assert feats["min_speed"] >= 0.5


def test_slow_down_without_crossing_the_stop_threshold() -> None:
    speeds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.0]  # v0=5.0, vf=2.0, delta=3.0 >= SPEED_DELTA_MPS, min=2.0
    dx = [v * 0.5 for v in speeds]
    waypoints = _waypoints_from_steps(dx, [0.0] * len(dx))
    _, lon, feats = label(waypoints)
    assert lon == "slow_down"
    assert feats["min_speed"] >= 0.5


def test_label_returns_none_triple_for_too_few_waypoints() -> None:
    assert label([]) == (None, None, None)
    assert label([[1.0, 0.0]]) == (None, None, None)


def test_classify_lateral_and_longitudinal_are_pure_functions_of_features() -> None:
    feats = geometry_features(_constant_lateral_drift(dy_per_step=0.4))
    assert classify_lateral(feats) == "turn_left"
    assert classify_longitudinal(feats) == "keep_speed"
