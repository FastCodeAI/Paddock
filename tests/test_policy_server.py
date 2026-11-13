"""`/observe` on the policy node: a frame of history in, no plan out."""

from __future__ import annotations

import numpy as np
import pytest

from paddock.policy.protocol import Command, Observation, Policy, Trajectory

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paddock.policy.server import build_app  # noqa: E402


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


class _Stepping(Policy):
    """Counts frames since its last reset; the plan says which frame of the episode it came from."""

    cameras = ()

    def __init__(self) -> None:
        self.resets, self.since_reset, self.primed = 0, 0, []

    def reset(self) -> None:
        self.resets += 1
        self.since_reset = 0

    def infer(self, obs: Observation) -> Trajectory:
        self.since_reset += 1
        self.primed.append(self.self_prime)
        return Trajectory(np.ones((6, 2)) * self.since_reset, extras={"ts": obs.timestamp_us})


def test_infer_each_resets_once_and_answers_every_frame_in_order() -> None:
    """One episode per request: a plan per frame, the state carried across them, self-priming off."""
    policy = _Stepping()
    frames = []
    for i in range(4):
        body = _body()
        body["timestamp"] = 1_000_000 + 500_000 * i
        frames.append(body)
    with TestClient(build_app(policy)) as client:
        answer = client.post("/infer_each", json={"observations": frames})
    assert answer.status_code == 200
    plans = answer.json()["plans"]
    assert [p["trajectory"][0][0] for p in plans] == [1.0, 2.0, 3.0, 4.0]
    assert [p["extras"]["ts"] for p in plans] == [1_000_000, 1_500_000, 2_000_000, 2_500_000]
    assert all(p["dt"] == 0.5 for p in plans)
    assert policy.resets == 1 and policy.primed == [False] * 4
    assert policy.self_prime is True, "restored for the closed-loop calls that follow"
