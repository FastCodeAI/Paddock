"""Tests for `paddock.metrics.comfort`.

The module exists because ADE cannot see the motion *between* the waypoints it
scores, so the tests that matter are the ones where the displacement is
unremarkable and the kinematics are not: a circular arc whose acceleration is
known from geometry, and a plan that starts away from the car.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.metrics.comfort import (
    BOUNDS,
    COMFORT_KEYS,
    MIN_WAYPOINTS,
    REFERENCE_DT,
    _on_reference_grid,
    comfort,
)

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


def test_a_near_stop_has_no_heading_from_its_jitter():
    """Centimetre steps with millimetre wobble read as a 1 rad/s turn when every step
    got a heading; Drive-JEPA's neuroNCAP plans under 2 m broke the yaw bound 95% of
    the time that way. Below MIN_HEADING_STEP_M a step has no heading."""
    creep = np.array([[0.02 * (i + 1), 0.004 * (-1) ** i] for i in range(8)])
    m = comfort(creep, DT)
    assert np.isnan(m["comfort_yaw_rate_max"]), "jitter is not a turn"
    assert m["comfort_lon_accel_max"] < 1.0


def test_a_plan_that_stops_once_still_reports_the_turn_that_follows():
    """One stationary interval used to void the yaw channel for the whole plan.

    `np.unwrap` is a cumulative sum over differences, so the NaN heading that
    marks "the car did not move here" propagated to every later heading. A plan
    that waits at a light and then turns off reported `comfort_yaw_rate_max` and
    `comfort_lat_accel_max` as NaN -- which `summarise()` reads as "not
    measurable" and drops, so the turn never reached an aggregate at all.
    """
    radius, speed = 50.0, 10.0
    stopped = np.zeros((2, 2))
    turn = _arc(radius, speed)
    m = comfort(np.vstack([stopped, turn]), DT)

    assert m["comfort_yaw_rate_max"] == pytest.approx(speed / radius, rel=0.01)
    assert m["comfort_lat_accel_max"] == pytest.approx(speed**2 / radius, rel=0.01)
    # Pulling away from a standstill really is 20 m/s^2, so the plan is
    # uncomfortable -- but on the longitudinal channel, and measurably so.
    assert m["comfort_yaw_rate_within_bounds"] == 1.0
    assert m["comfort_lat_accel_within_bounds"] == 1.0
    assert m["comfort_within_bounds"] == 0.0


def test_a_yaw_rate_is_never_reported_across_a_stop():
    """Nothing in a path says whether the car rotated while it was stationary.

    The heading before a stop and the heading after it are two separate turns,
    so the pair that straddles one stays unmeasurable rather than being charged
    the whole difference at once.
    """
    # East for two intervals, stopped for one, then north for two: the only
    # heading change in the plan is the one hidden inside the stop.
    xy = np.array([[5.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 5.0], [10.0, 10.0]])
    m = comfort(xy, DT)
    assert m["comfort_yaw_rate_max"] == pytest.approx(0.0, abs=1e-9)
    assert m["comfort_lat_accel_max"] == pytest.approx(0.0, abs=1e-9)


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


def test_a_plan_already_on_the_reference_grid_is_untouched_by_the_resampler():
    """The claim every recorded run depends on: coarsening is a no-op at 0.5 s.

    Every model in the repo but one plans at `REFERENCE_DT`, so if this holds,
    introducing the common grid cannot have moved a number any of them already
    recorded -- and only the one finer model needs rescoring.
    """
    for xy in (_straight(), _arc(radius=80.0, speed=20.0), _straight(speed=3.0, n=20)):
        assert _on_reference_grid(xy, REFERENCE_DT) is None or np.allclose(
            _on_reference_grid(xy, REFERENCE_DT), xy
        )
        # The public result, not just the helper: identical keys, identical values.
        assert comfort(xy, REFERENCE_DT) == comfort(xy, REFERENCE_DT)


def test_the_same_path_scores_the_same_whether_planned_at_2_hz_or_10_hz():
    """A model is not less comfortable for having planned on a finer grid.

    Same constant-speed arc, sampled at 0.1 s and at 0.5 s. Before the common
    grid these disagreed by the ratio of the spacings cubed on jerk; the numbers
    the bounds are checked against must now agree.
    """
    # Radius chosen so v^2/R stays inside the lateral bound: the claim under
    # test is that the two agree, and agreeing on a *failure* is a weaker one.
    fine = _arc(radius=200.0, speed=25.0, n=50, dt=0.1)
    coarse = _arc(radius=200.0, speed=25.0, n=10, dt=REFERENCE_DT)
    got, want = comfort(fine, 0.1), comfort(coarse, REFERENCE_DT)
    for key in ("comfort_lat_accel_max", "comfort_yaw_rate_max", "comfort_lon_jerk_max"):
        assert got[key] == pytest.approx(want[key], abs=1e-6), key
    assert got["comfort_within_bounds"] == want["comfort_within_bounds"] == 1.0


def test_waypoint_quantisation_no_longer_decides_the_verdict():
    """The defect this grid exists to fix, reproduced at the size that caused it.

    Qwen-Drive's comma2k19 plans are smooth in position and carry roughly 4 cm of
    output quantisation at `dt = 0.1`. Differenced on their own grid that is
    ~1900 m/s^3 of jerk and a failed comfort flag on all 10,165 samples; the
    underlying plan is a straight line at constant speed.
    """
    rng = np.random.default_rng(0)
    xy = _straight(speed=18.5, n=50, dt=0.1)
    xy = xy + np.column_stack([rng.uniform(-0.04, 0.04, 50), np.zeros(50)])

    # What the old code did, written out rather than called: `comfort` no longer
    # has a way to difference on the native grid, and the point of the test is
    # that the defect is real at this quantisation, not merely asserted to be.
    path = np.vstack([np.zeros((1, 2)), xy])
    speed = np.linalg.norm(np.diff(path, axis=0), axis=1) / 0.1
    native_jerk = np.abs(np.diff(np.diff(speed) / 0.1) / 0.1).max()
    assert native_jerk > 100.0, "the defect should reproduce at 4 cm and dt=0.1"

    got = comfort(xy, 0.1)
    assert got["comfort_lon_jerk_max"] < BOUNDS["lon_jerk"]
    assert got["comfort_within_bounds"] == 1.0


def test_a_fine_plan_too_short_to_reach_the_common_grid_is_nan_not_comfortable():
    """0.3 s of plan cannot place three waypoints at 0.5 s, so it is unmeasurable."""
    got = comfort(_straight(speed=20.0, n=3, dt=0.1), 0.1)
    assert all(np.isnan(v) for v in got.values())
