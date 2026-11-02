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
