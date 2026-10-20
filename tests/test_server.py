"""`policy/server.py`'s FastAPI wrapper: the one node every model's infer loop shares."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from paddock.policy.protocol import Command, Observation, Policy, constant_velocity
from paddock.policy.server import build_app


class _Constant(Policy):
    def infer(self, obs):
        return constant_velocity(1.0)


def _wire_body() -> dict:
    obs = Observation(
        images={},
        ego2world=np.eye(4),
        canbus=np.arange(18, dtype=np.float64),
        timestamp_us=1,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )
    return obs.to_wire()


def test_padock_capture_infer_appends_one_wire_body_line_per_call(tmp_path, monkeypatch) -> None:
    """A debug-only replay dump: every /infer request body, one JSON line each."""
    capture_path = tmp_path / "capture.jsonl"
    monkeypatch.setenv("PADDOCK_CAPTURE_INFER", str(capture_path))

    client = TestClient(build_app(_Constant()))
    body = _wire_body()
    for _ in range(2):
        resp = client.post("/infer", json=body)
        assert resp.status_code == 200

    lines = capture_path.read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line) == body for line in lines)


def test_infer_does_not_write_a_capture_file_when_unset(tmp_path, monkeypatch) -> None:
    """Opt-in only -- the ordinary case must not leave a file behind."""
    monkeypatch.delenv("PADDOCK_CAPTURE_INFER", raising=False)
    client = TestClient(build_app(_Constant()))
    assert client.post("/infer", json=_wire_body()).status_code == 200
    assert list(tmp_path.iterdir()) == []
