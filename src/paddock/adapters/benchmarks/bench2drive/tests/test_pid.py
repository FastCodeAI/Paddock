"""The reimplemented dual-PID tracker, checked without CARLA.

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
    grid_stride,
    on_grid,
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


def test_a_plan_is_read_at_its_own_points_every_half_second() -> None:
    """A plan 0.5 s apart is read point for point, one 0.1 s apart at every fifth point, and a
    spacing that does not divide 0.5 s not at all: nothing is interpolated."""
    assert grid_stride(0.5, 0.5) == 1
    assert grid_stride(0.1, 0.5) == 5
    assert grid_stride(0.25, 0.5) == 2
    for off_grid in (0.2, 0.3, 0.75, 1.0, 0.0, -0.1):
        assert grid_stride(off_grid, 0.5) is None, off_grid

    coarse = _plan(5.0, curve=0.3, n=8)
    assert np.array_equal(on_grid(coarse, 0.5, 0.5, 6), coarse[:6]), "the first six, as the wire sends"

    t = 0.1 * np.arange(1, 65)  # alpamayo2edge: 64 poses 0.1 s apart, with heading
    fine = np.stack([5.0 * t, 0.2 * t**2, np.arctan2(0.4 * t, 5.0)], axis=1)
    grid = on_grid(fine, 0.1, 0.5, 6)
    assert np.array_equal(grid, fine[4:30:5, :2]), "its points at 0.5, 1.0, ... 3.0 s; the yaw is not read"
    assert np.allclose(grid[:, 0], 5.0 * 0.5 * np.arange(1, 7))


def test_a_fine_plan_read_on_the_grid_keeps_its_speed() -> None:
    """The bug this guards: a 0.1 s plan's first six points, read 0.5 s apart, ask for a fifth
    of the planned speed."""
    t = 0.1 * np.arange(1, 65)
    fine = np.stack([6.0 * t, np.zeros_like(t)], axis=1)
    assert WaypointTracker(0.5).control(fine[:6], 6.0)[3] == pytest.approx(6.0 / 5)
    assert WaypointTracker(0.5).control(on_grid(fine, 0.1, 0.5, 6), 6.0)[3] == pytest.approx(6.0)


def test_a_plan_off_the_grid_or_too_short_is_refused() -> None:
    with pytest.raises(ValueError, match="no points on the 0.5 s grid"):
        on_grid(_plan(5.0, n=20, dt=0.2), 0.2, 0.5, 6)
    with pytest.raises(ValueError, match="ends at 2.5 s"):
        on_grid(_plan(5.0, n=25, dt=0.1), 0.1, 0.5, 6)
    with pytest.raises(ValueError, match="ends at 2.5 s"):
        on_grid(_plan(5.0, n=5), 0.5, 0.5, 6)
