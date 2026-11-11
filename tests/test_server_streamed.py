"""`/infer_sequence` with and without a `stream`: the node half of navhard's streamed mode."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from paddock.policy.protocol import Command, Observation, Policy, StreamFrame, Trajectory

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from paddock.policy.server import build_app  # noqa: E402

STREAM = StreamFrame(
    key="2021.06.28.15.02.02_veh-38_02398_02848", timestamp_us=1_624_895_253_999_326, pose=(1.5, -2.25, 0.3)
)


def _observations(n: int = 4) -> list[dict]:
    return [
        Observation(
            images={},
            ego2world=np.eye(4),
            canbus=np.zeros(18),
            timestamp_us=-500_000 * (n - 1 - i),
            command=Command.STRAIGHT,
            camera2image={},
            camera2ego={},
        ).to_wire()
        for i in range(n)
    ]


class _Plain(Policy):
    """Keeps nothing between samples and does not override `infer_sequence_streamed`."""

    cameras = ()

    def infer(self, obs: Observation) -> Trajectory:
        return Trajectory(np.full((6, 2), obs.timestamp_us / 1e6))


class _Streams(_Plain):
    """Records which entry point each request reached."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, StreamFrame | None, int]] = []

    def infer_sequence(self, observations: list[Observation]) -> Trajectory:
        self.calls.append(("infer_sequence", None, len(observations)))
        return super().infer_sequence(observations)

    def infer_sequence_streamed(self, observations: list[Observation], stream: StreamFrame) -> Trajectory:
        self.calls.append(("infer_sequence_streamed", stream, len(observations)))
        return super().infer_sequence(observations)


def test_a_request_without_a_stream_reaches_infer_sequence_as_before() -> None:
    policy = _Streams()
    with TestClient(build_app(policy)) as client:
        assert client.post("/infer_sequence", json={"observations": _observations()}).status_code == 200
    assert policy.calls == [("infer_sequence", None, 4)]


def test_a_stream_reaches_infer_sequence_streamed_as_a_stream_frame() -> None:
    policy = _Streams()
    with TestClient(build_app(policy)) as client:
        body = {"observations": _observations(), "stream": STREAM.to_wire()}
        assert client.post("/infer_sequence", json=body).status_code == 200
    assert policy.calls[0] == ("infer_sequence_streamed", STREAM, 4)


def test_a_model_that_keeps_nothing_ignores_the_stream() -> None:
    with TestClient(build_app(_Plain())) as client:
        plain = client.post("/infer_sequence", json={"observations": _observations()})
        streamed = client.post(
            "/infer_sequence", json={"observations": _observations(), "stream": STREAM.to_wire()}
        )
    assert plain.status_code == streamed.status_code == 200
    assert plain.content == streamed.content


MALFORMED = (
    {"key": "", "timestamp_us": 1, "pose": [0, 0, 0]},
    {"key": "a", "timestamp_us": 1.5, "pose": [0, 0, 0]},
    {"key": "a", "timestamp_us": True, "pose": [0, 0, 0]},
    {"key": "a", "timestamp_us": 1, "pose": [0, 0]},
    "log_a",
)


def test_a_malformed_stream_fails_the_sample_rather_than_keying_a_book_wrongly() -> None:
    policy = _Streams()
    with TestClient(build_app(policy)) as client:
        for stream in MALFORMED:
            with pytest.raises(ValueError, match="stream"):
                client.post("/infer_sequence", json={"observations": _observations(), "stream": stream})
    assert policy.calls == []


def test_the_handler_runs_each_request_to_completion_before_the_next() -> None:
    """A model's per-drive state (Drive-JEPA's momentum books) relies on it: an
    `async def` with no `await` holds the event loop for the whole sample. A
    plain `def` would be run on a thread pool, two samples at once."""
    app = build_app(_Plain())
    route = next(r for r in app.routes if getattr(r, "path", None) == "/infer_sequence")
    assert inspect.iscoroutinefunction(route.endpoint)
    tree = ast.parse(textwrap.dedent(inspect.getsource(route.endpoint)))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Await)]


def test_a_stream_frame_survives_the_wire_and_refuses_what_it_cannot_key() -> None:
    wire = STREAM.to_wire()
    assert wire == {"key": STREAM.key, "timestamp_us": STREAM.timestamp_us, "pose": [1.5, -2.25, 0.3]}
    assert StreamFrame.from_wire(wire) == STREAM
    with pytest.raises(ValueError, match="pose"):
        StreamFrame.from_wire({**wire, "pose": [0.0, float("nan"), 0.0]})
