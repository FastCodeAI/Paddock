"""Tests for `paddock.metrics.comfort`.

The module exists because ADE cannot see the motion *between* the waypoints it
scores, so the tests that matter are the ones where the displacement is
unremarkable and the kinematics are not: a circular arc whose acceleration is
known from geometry, and a plan that starts away from the car.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.metrics.comfort import BOUNDS, COMFORT_KEYS, MIN_WAYPOINTS, comfort

DT = 0.5


def _straight(speed: float = 20.0, n: int = 8, dt: float = DT) -> np.ndarray:
    x = speed * dt * np.arange(1, n + 1)
    return np.stack([x, np.zeros(n)], axis=1)


def _arc(radius: float, speed: float, n: int = 8, dt: float = DT) -> np.ndarray:
    """A constant-speed left turn: yaw rate is v/R and lateral accel v^2/R."""
    theta = (speed / radius) * dt * np.arange(1, n + 1)
    return np.stack([radius * np.sin(theta), radius * (1 - np.cos(theta))], axis=1)


def test_every_call_returns_exactly_the_declared_keys():
    """A key that appears only sometimes breaks aggregation across models."""
    for xy in (_straight(), np.zeros((2, 2)), np.full((6, 2), np.nan)):
        assert set(comfort(xy, DT)) == set(COMFORT_KEYS)


def test_a_straight_constant_speed_plan_is_comfortable():
    m = comfort(_straight(), DT)
    for name in BOUNDS:
        assert abs(m[f"comfort_{name}_max"]) < 1e-9, name
        assert m[f"comfort_{name}_within_bounds"] == 1.0, name
    assert m["comfort_within_bounds"] == 1.0


def test_a_constant_radius_turn_reports_the_geometric_yaw_rate_and_lateral_accel():
    """The check that the finite differences mean what their names say.

    Chord differencing under-reads a circular arc by `sinc(w*dt/2)`, which is
    0.04 % here, so 1 % tolerance tests the formula and not the discretisation.
    """
    radius, speed = 100.0, 20.0
    m = comfort(_arc(radius, speed), DT)
    assert m["comfort_yaw_rate_max"] == pytest.approx(speed / radius, rel=0.01)
    assert m["comfort_lat_accel_max"] == pytest.approx(speed**2 / radius, rel=0.01)
    # 4 m/s^2 is inside the 4.89 bound; the longitudinal channels see a constant
    # speed, so a turn this gentle is comfortable in every channel.
    assert m["comfort_within_bounds"] == 1.0


def test_a_sharp_turn_fails_the_lateral_bound_and_the_overall_flag():
    m = comfort(_arc(radius=25.0, speed=20.0), DT)  # 16 m/s^2
    assert m["comfort_lat_accel_max"] > BOUNDS["lat_accel"]
    assert m["comfort_lat_accel_within_bounds"] == 0.0
    assert m["comfort_within_bounds"] == 0.0
    # One failing channel must not contaminate the others.
    assert m["comfort_lon_accel_within_bounds"] == 1.0


def test_braking_is_reported_as_a_positive_magnitude_against_its_own_bound():
    """`lon_decel` is a separate, looser bound than `lon_accel` -- braking hard
    is allowed where accelerating hard is not, and the sign convention is what
    keeps the two from being compared against each other."""
    # 20 m/s to a stop within one 0.5 s interval: -40 m/s^2.
    xy = np.array([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    m = comfort(xy, DT)
    assert m["comfort_lon_decel_max"] == pytest.approx(40.0)
    assert m["comfort_lon_decel_within_bounds"] == 0.0
    assert m["comfort_lon_accel_max"] <= 0.0


def test_a_plan_that_starts_away_from_the_car_is_not_smooth():
    """The origin is prepended, so the first commanded step is measurable.

    Without it a plan whose first waypoint jumps 20 m ahead -- the classic
    "model has not warmed up" artefact -- would difference to a perfectly
    constant speed and score as the most comfortable plan in the run. With it,
    the jump is a 42 m/s first interval falling to 2 m/s: the car is commanded
    to teleport and then brake, and both halves are out of bounds.
    """
    jump = _straight(speed=2.0) + np.array([20.0, 0.0])
    m = comfort(jump, DT)
    assert m["comfort_lon_decel_max"] > BOUNDS["lon_decel"]
    assert m["comfort_lon_jerk_max"] > BOUNDS["lon_jerk"]
    assert m["comfort_within_bounds"] == 0.0


def test_a_stopped_plan_has_no_yaw_rate_rather_than_a_zero_one():
    """`arctan2(0, 0)` is 0, not NaN, so a parked plan would otherwise report a
    heading and be credited with holding it steady."""
    m = comfort(np.zeros((8, 2)), DT)
    assert np.isnan(m["comfort_yaw_rate_max"])
    assert np.isnan(m["comfort_yaw_rate_within_bounds"])
    # And an unmeasurable channel makes the overall flag unmeasurable too,
    # rather than passing on the strength of the channels that did resolve.
    assert np.isnan(m["comfort_within_bounds"])
    assert m["comfort_lon_accel_max"] == 0.0


@pytest.mark.parametrize(
    "xy, dt",
    [
        (np.zeros((MIN_WAYPOINTS - 1, 2)), DT),  # too short for a jerk
        (_straight(), 0.0),  # no grid to difference on
        (np.array([[1.0, 0.0], [np.nan, 0.0], [3.0, 0.0], [4.0, 0.0]]), DT),
        (np.zeros(4), DT),  # not (N, 2)
    ],
)
def test_an_unusable_plan_is_nan_throughout_rather_than_partly_scored(xy, dt):
    m = comfort(xy, dt)
    assert all(np.isnan(v) for v in m.values()), m
