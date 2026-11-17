"""The floor must be exactly predictable, or it cannot serve as a reference.

Every other model's number on a closed-loop benchmark is read as a difference
from this one, so a change here silently moves every GAP in the leaderboard.
These pin the arithmetic rather than trusting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.constant_velocity.policy import (
    CANBUS_VELOCITY,
    N_WAYPOINTS,
    POLICY,
)
from paddock.policy.protocol import Command, Observation


def _obs(speed: float | None) -> Observation:
    canbus = np.zeros(18, dtype=np.float64)
    if speed is not None:
        canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=Command.FOLLOW_REFERENCE,
        camera2image={},
        camera2ego={},
    )


def test_the_plan_is_the_speed_times_the_elapsed_time() -> None:
    """10 m/s at dt 0.5 reaches 5, 10, ... 30 m. Checkable by hand on purpose."""
    plan = POLICY().infer(_obs(10.0))
    assert plan.dt == 0.5
    assert plan.waypoints.shape == (N_WAYPOINTS, 2)
    np.testing.assert_allclose(plan.waypoints[:, 0], [5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    # Straight ahead: y is exactly zero, not nearly.
    assert not plan.waypoints[:, 1].any()


def test_it_plans_six_waypoints_because_closed_loop_needs_six() -> None:
    """`Trajectory.to_wire` refuses fewer, and the engine indexes positionally."""
    assert POLICY().infer(_obs(3.0)).to_wire()["trajectory"]


def test_a_stationary_ego_plans_to_stay_put() -> None:
    plan = POLICY().infer(_obs(0.0))
    assert not plan.waypoints.any()


def test_it_is_deterministic() -> None:
    """Any difference between two runs of the floor is the harness, not the policy."""
    policy = POLICY()
    first = policy.infer(_obs(7.25)).waypoints
    policy.reset()
    np.testing.assert_array_equal(first, policy.infer(_obs(7.25)).waypoints)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -4.0])
def test_an_unusable_speed_becomes_a_stationary_plan_not_a_guess(bad: float) -> None:
    """A floor that invented a speed would be driving on its own invention.

    NaN would also propagate into the waypoints and be rejected downstream by
    Trajectory, failing somewhere that says nothing about calibration.
    """
    plan = POLICY().infer(_obs(bad))
    assert not plan.waypoints.any()


def test_a_short_canbus_is_read_as_no_speed_rather_than_crashing() -> None:
    """Not every source fills 18 channels; a floor should still answer."""
    obs = _obs(5.0)
    obs.canbus = np.zeros(4, dtype=np.float64)
    assert not POLICY().infer(obs).waypoints.any()


def test_it_does_not_self_prime() -> None:
    """A manufactured history would change nothing and only obscure that."""
    assert POLICY().self_prime is False


def test_a_fixed_speed_ignores_the_measured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Bench2Drive route starts at rest; the speed bands drive at a set speed from tick one."""
    monkeypatch.setenv("PADDOCK_CV_SPEED_M_S", "3")
    plan = POLICY().infer(_obs(0.0))
    np.testing.assert_allclose(plan.waypoints[:, 0], [1.5, 3.0, 4.5, 6.0, 7.5, 9.0])
    assert not plan.waypoints[:, 1].any()


def test_without_the_variable_it_holds_the_measured_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDOCK_CV_SPEED_M_S", raising=False)
    assert not POLICY().infer(_obs(0.0)).waypoints.any()


@pytest.mark.parametrize("bad", ["0", "-3", "nan", "inf", "fast"])
def test_an_unusable_fixed_speed_refuses_to_start(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A typo must not quietly become the measured-speed floor, which scores 0 on every route."""
    monkeypatch.setenv("PADDOCK_CV_SPEED_M_S", bad)
    with pytest.raises(ValueError):
        POLICY()
