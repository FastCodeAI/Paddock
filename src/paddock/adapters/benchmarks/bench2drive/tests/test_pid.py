"""The vendored dual-PID tracker, checked without CARLA.

Pins the sign conventions -- the one place a frame mix-up would send the car
the wrong way with a perfectly valid-looking score -- and the brake rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive.pid import (
    BRAKE_SPEED,
    CONTROL_HORIZON_S,
    MAX_THROTTLE,
    WaypointTracker,
)


def _plan(speed_mps: float, curve: float = 0.0, n: int = 6, dt: float = 0.5) -> np.ndarray:
    """Waypoints in paddock's frame (x fwd, y LEFT) at `speed_mps`, bending by `curve` m per step^2."""
    t = np.arange(1, n + 1)
    x = speed_mps * dt * t
    y = curve * t**2
    return np.stack([x, y], axis=1)


def test_straight_plan_steers_straight_and_accelerates_from_rest() -> None:
    steer, throttle, brake, desired = WaypointTracker(0.5).control(_plan(5.0), speed=0.0)
    assert desired == pytest.approx(5.0)
    assert steer == pytest.approx(0.0)
    assert 0.0 < throttle <= MAX_THROTTLE
    assert brake == 0.0


def test_desired_speed_is_waypoint_spacing_over_waypoint_dt() -> None:
    """The same geometry read on a 1 s grid is half the speed of a 0.5 s grid."""
    plan = _plan(4.0, dt=0.5)
    assert WaypointTracker(0.5).control(plan, 0.0)[3] == pytest.approx(4.0)
    assert WaypointTracker(1.0).control(plan, 0.0)[3] == pytest.approx(2.0)


def test_a_plan_to_the_left_steers_left_and_to_the_right_steers_right() -> None:
    """paddock's y is LEFT; CARLA's steer is +right. The tracker crosses them."""
    left = WaypointTracker().control(_plan(5.0, curve=+0.4), speed=5.0)[0]
    right = WaypointTracker().control(_plan(5.0, curve=-0.4), speed=5.0)[0]
    assert left < 0.0 < right
    assert left == pytest.approx(-right)


def test_a_stopped_plan_brakes() -> None:
    steer, throttle, brake, desired = WaypointTracker().control(np.zeros((6, 2)), speed=3.0)
    assert desired < BRAKE_SPEED
    assert brake == 1.0
    assert throttle == 0.0


def test_overspeed_brakes_rather_than_coasting() -> None:
    _, throttle, brake, desired = WaypointTracker().control(_plan(5.0), speed=8.0)
    assert desired == pytest.approx(5.0)
    assert brake == 1.0
    assert throttle == 0.0


def test_a_longer_plan_does_not_change_the_control_it_produces() -> None:
    """The desired speed is a mean, so an unfixed window makes throttle a
    function of how far the model chose to plan.

    openpilot's grid was extended from 4 s to supercombo's native 10 s; without
    `CONTROL_HORIZON_S` that alone would have moved every recorded Bench2Drive
    score, with nothing in the Bench2Drive diff to explain it. Both plans below
    are the same drive -- 5 m/s, straight -- read to different depths.
    """
    short = WaypointTracker(0.5).control(_plan(5.0, n=8), speed=3.0)
    long = WaypointTracker(0.5).control(_plan(5.0, n=20), speed=3.0)
    assert short == long

    # ... and the window is a truncation, not a resample: a plan that stops
    # inside it is still read whole.
    assert WaypointTracker(0.5).control(_plan(5.0, n=6), speed=0.0)[3] == pytest.approx(5.0)


def test_the_control_window_covers_what_the_gains_were_tuned_on() -> None:
    """TCP's constants assume roughly four seconds of plan; a much shorter
    window would be a different controller wearing the same numbers."""
    assert CONTROL_HORIZON_S == pytest.approx(4.0)
    assert int(round(CONTROL_HORIZON_S / 0.5)) == 8


def test_yaw_column_is_ignored_and_short_plans_are_rejected() -> None:
    tracker = WaypointTracker()
    with_yaw = np.hstack([_plan(5.0), np.zeros((6, 1))])
    assert tracker.control(with_yaw, 0.0)[3] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        tracker.control(np.array([[2.5, 0.0]]), 0.0)
