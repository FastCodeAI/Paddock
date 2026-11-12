"""The policy node: `/observe` takes a frame of history in and plans nothing, and the
heartbeat says how far the plan reaches."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from paddock.policy.protocol import Command, Observation, Policy, Trajectory

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paddock.policy.server import _plan_reach, build_app  # noqa: E402


def _body() -> dict:
    return Observation(
        images={},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=1_000_000,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    ).to_wire()


class _Counting(Policy):
    """Plans every call; does not override `observe`."""

    cameras = ()

    def __init__(self) -> None:
        self.infers = 0

    def infer(self, obs: Observation) -> Trajectory:
        self.infers += 1
        return Trajectory(np.ones((6, 2)) * self.infers)


class _Recording(_Counting):
    """Overrides `observe` to record without planning, as Drive-JEPA does."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded = 0

    def observe(self, obs: Observation) -> None:
        self.recorded += 1


def test_observe_answers_with_no_plan_and_leaves_infer_unchanged() -> None:
    """The answer has nothing a simulator could drive; a later /infer is the
    one that would have followed a thrown-away /infer."""
    observed, thrown_away = _Counting(), _Counting()
    with TestClient(build_app(observed)) as a, TestClient(build_app(thrown_away)) as b:
        answer = a.post("/observe", json=_body())
        assert answer.status_code == 200 and answer.json() == {"observed": True}
        b.post("/infer", json=_body())
        assert a.post("/infer", json=_body()).content == b.post("/infer", json=_body()).content
    assert observed.infers == thrown_away.infers == 2


def test_a_model_that_only_records_is_not_asked_to_plan() -> None:
    policy = _Recording()
    with TestClient(build_app(policy)) as client:
        client.post("/observe", json=_body())
    assert (policy.recorded, policy.infers) == (1, 0)


# -- the heartbeat's plan reach --------------------------------------------------------


def _steady(dt: float, horizon_s: float = 6.4) -> Trajectory:
    """6 m/s straight ahead, `dt` apart, to `horizon_s`."""
    t = dt * np.arange(1, int(round(horizon_s / dt)) + 1)
    return Trajectory(np.stack([6.0 * t, np.zeros_like(t)], axis=1), dt=dt)


@pytest.mark.parametrize("dt", [0.5, 0.25, 0.1])
def test_the_heartbeat_reads_every_plan_half_a_second_and_three_seconds_ahead(dt: float) -> None:
    """Whatever the spacing. It read `trajectory`, a plan's first six points: 3 s of a plan
    0.5 s apart, but 0.6 s of alpamayo2edge's, 0.1 s apart, logged as barely moving (#200)."""
    reach = _plan_reach(_steady(dt).to_wire())
    assert [value for pair in reach for value in pair] == pytest.approx([3.0, 0.5, 18.0, 3.0])


def test_a_plan_half_a_second_apart_reads_as_it_did_before() -> None:
    """The first and last of the six points `trajectory` holds, to the bit."""
    result = Trajectory(np.random.default_rng(0).normal(size=(8, 3)), dt=0.5).to_wire()
    (near, _), (far, _) = _plan_reach(result)
    (x0, y0), (x5, y5) = result["trajectory"][0], result["trajectory"][-1]
    assert (near, far) == ((x0**2 + y0**2) ** 0.5, (x5**2 + y5**2) ** 0.5)


def test_a_plan_ending_before_three_seconds_says_where_it_ends() -> None:
    (near, near_s), (far, far_s) = _plan_reach(_steady(0.1, horizon_s=0.6).to_wire())
    assert (near_s, far_s) == (pytest.approx(0.5), pytest.approx(0.6))
    assert far == pytest.approx(3.6)
    assert all(np.isnan(m) for m, _ in _plan_reach({"trajectory": [[1.0, 0.0]] * 6}))


class _Fine(Policy):
    """alpamayo2edge's shape of plan: 64 points 0.1 s apart."""

    cameras = ()

    def infer(self, obs: Observation) -> Trajectory:
        return _steady(0.1)


def test_the_node_log_names_the_times_it_read(caplog) -> None:
    caplog.set_level(logging.INFO, logger="paddock.policy.server")
    with TestClient(build_app(_Fine())) as client:
        client.post("/infer", json=_body())
    assert "plan reach 3.00 m at 0.5 s / 18.00 m at 3.0 s" in caplog.text
