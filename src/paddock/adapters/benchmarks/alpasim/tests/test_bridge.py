"""End-to-end test of the gRPC-to-HTTP translation, on CPU.

The bridge is the only novel code in this adapter -- every other closed-loop
benchmark paddock drives speaks HTTP to the agent already -- so it is exercised
for real here rather than only through a stand-in: a gRPC client that talks the
EgodriverService contract, the actual `DriverScaffold` + `PolicyPlanner`, and a
local HTTP stub standing in for the policy node. No GPU, no simulator, no model.

**Skipped where AlpaSim is not installed.** `grpc` and `alpasim_grpc` belong to
the simulator's environment, which paddock's own deliberately lacks (the bridge
runs under `[benchmarks.alpasim] python`, not paddock's interpreter). So CI
skips this file and a reviewer with a site runs it -- the arrangement
CONTRIBUTING describes for tests that genuinely need a site.

    .../alpasim/.venv/bin/python -m pytest \\
        src/paddock/adapters/benchmarks/alpasim/tests/test_bridge.py -q
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

grpc = pytest.importorskip("grpc", reason="AlpaSim's environment supplies grpc")
pytest.importorskip("alpasim_grpc", reason="AlpaSim's environment supplies alpasim_grpc")

from alpasim_grpc.v0.common_pb2 import Empty  # noqa: E402
from alpasim_grpc.v0.egodriver_pb2 import (  # noqa: E402
    DriveRequest,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    RolloutEgoTrajectory,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import EgodriverServiceStub  # noqa: E402

from paddock.adapters.benchmarks.alpasim.bridge import PolicyPlanner  # noqa: E402
from paddock.adapters.benchmarks.alpasim.scaffold import DriverScaffold  # noqa: E402

#: What the stub node plans: 2 m/s straight ahead, six waypoints half a second
#: apart -- a constant-velocity plan, so the arithmetic is checkable by hand.
STUB_DT = 0.5
STUB_TRAJECTORY = [[1.0 * i, 0.0] for i in range(1, 7)]


class _StubNode(BaseHTTPRequestHandler):
    """The smallest thing that answers /infer like a paddock policy node."""

    bodies: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler's spelling)
        length = int(self.headers.get("Content-Length", 0))
        _StubNode.bodies.append(json.loads(self.rfile.read(length)))
        payload = json.dumps({"trajectory": STUB_TRAJECTORY, "dt": STUB_DT}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return


@pytest.fixture
def stub_node():
    _StubNode.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _StubNode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _StubNode
    server.shutdown()


@pytest.fixture
def bridge(stub_node):
    """The real scaffold and planner behind a real gRPC channel."""
    url, _ = stub_node
    scaffold = DriverScaffold(planner=PolicyPlanner(url), version_id="test-bridge-0.0.0", keep_images=True)
    server = grpc.server(__import__("concurrent.futures", fromlist=["x"]).ThreadPoolExecutor(4))
    from alpasim_grpc.v0.egodriver_pb2_grpc import add_EgodriverServiceServicer_to_server

    add_EgodriverServiceServicer_to_server(scaffold, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield EgodriverServiceStub(channel), stub_node[1]
    channel.close()
    server.stop(None)


def _pose(timestamp_us: int, x: float = 0.0):
    """One ego pose at the origin, facing +x."""
    from alpasim_grpc.v0.common_pb2 import PoseAtTime

    pose = PoseAtTime()
    pose.timestamp_us = timestamp_us
    pose.pose.vec.x = x
    pose.pose.quat.w = 1.0
    return pose


def _open_session(stub, session="s-1", cameras=("camera_front_wide_120fov",)):
    """start_session as the runtime actually sends it.

    The camera declarations are nested three deep --
    rollout_spec.vehicle.available_cameras -- and the scene id is an optional
    DebugInfo that "will not be present in benchmark runs to avoid any
    potential data leakage" (egodriver.proto). Both matter: this is the only
    RPC that ever carries calibration.
    """
    request = DriveSessionRequest()
    request.session_uuid = session
    request.debug_info.scene_id = "clipgt-test"
    for logical_id in cameras:
        camera = request.rollout_spec.vehicle.available_cameras.add()
        camera.logical_id = logical_id
        # A real declaration carries a valid rig_to_camera. An all-zero
        # quaternion is not merely empty -- geometry rejects it as corrupt.
        camera.rig_to_camera.quat.w = 1.0
    return stub.start_session(request)


def _push_egomotion(stub, session="s-1", timestamp_us=1_000_000):
    request = RolloutEgoTrajectory()
    request.session_uuid = session
    request.trajectory.poses.append(_pose(timestamp_us))
    # No scalar speed exists on the wire: DynamicState carries Vec3 velocities
    # only, so 5 m/s forward is what the bridge must derive a speed from.
    request.dynamic_states.add().linear_velocity.x = 5.0
    return stub.submit_egomotion_observation(request)


def _drive(stub, session="s-1", now_us=1_000_000, query_us=1_000_000):
    request = DriveRequest()
    request.session_uuid = session
    request.time_now_us = now_us
    request.time_query_us = query_us
    return stub.drive(request)


def test_the_version_the_runtime_probes_is_the_one_we_declared(bridge) -> None:
    """The runtime probes this once at connect; it is what identifies the run."""
    stub, _ = bridge
    assert stub.get_version(Empty()).version_id == "test-bridge-0.0.0"


def test_a_drive_call_reaches_the_node_and_comes_back_as_a_trajectory(bridge) -> None:
    """The whole path: gRPC in, HTTP out, HTTP back, gRPC out."""
    stub, node = bridge
    _open_session(stub)
    _push_egomotion(stub)
    response = _drive(stub)

    assert len(node.bodies) == 1, "the node should have been asked exactly once"
    assert response.trajectory.poses, "the bridge returned no waypoints"


def test_the_node_is_asked_with_what_alpasim_actually_sent(bridge) -> None:
    """Translation, not invention: ego pose, timestamp and speed must survive."""
    stub, node = bridge
    _open_session(stub)
    _push_egomotion(stub, timestamp_us=7_654_321)
    _drive(stub, now_us=7_700_000)

    body = node.bodies[0]
    assert body["timestamp"] == 7_654_321, "the node must be told the pose's time, not the request's"
    assert np.asarray(body["ego2world"]).shape == (4, 4)
    assert body["canbus"][13] == pytest.approx(5.0), "ego speed must reach the node"
    # FOLLOW_REFERENCE: AlpaSim's route is a path, not a discrete manoeuvre.
    assert body["command"] == 3


def test_the_plan_is_anchored_to_the_observation_it_was_built_from(bridge) -> None:
    """`model_t0_us` is the pose's time, never the request's.

    The fallback convention stamps the anchor with *now* while the pose behind
    it is older, moving the anchor forward in time without moving it in space.
    Inference here takes hundreds of milliseconds, so that gap is always real.
    """
    stub, _ = bridge
    _open_session(stub)
    _push_egomotion(stub, timestamp_us=1_000_000)
    response = _drive(stub, now_us=1_450_000)
    anchor_us = response.trajectory.poses[0].timestamp_us
    assert anchor_us == 1_000_000, "the anchor was stamped with `now`, not the pose it came from"


def test_waypoint_spacing_matches_the_dt_the_node_reported(bridge) -> None:
    stub, _ = bridge
    _open_session(stub)
    _push_egomotion(stub, timestamp_us=1_000_000)
    poses = _drive(stub).trajectory.poses
    # poses[0] is the anchor; the plan follows at the node's dt.
    spacing = poses[2].timestamp_us - poses[1].timestamp_us
    assert spacing == int(STUB_DT * 1e6)


def test_driving_before_any_egomotion_returns_the_contracts_not_ready(bridge) -> None:
    """An empty trajectory, and the node is never asked.

    An empty Trajectory is the egodriver contract's legal "not ready yet"
    (INTERFACE.md 4.8, 7.5), not an error -- the runtime pushes observations and
    calls drive() on its own schedule, and the first call can legitimately
    arrive before any egomotion has. What matters for correctness is that no
    plan is invented from no pose: the node must not be asked at all, because a
    trajectory returned here would be the bridge's driving, not the model's.
    """
    stub, node = bridge
    _open_session(stub)
    response = _drive(stub)
    assert list(response.trajectory.poses) == []
    assert node.bodies == [], "the node was asked for a plan with no pose to anchor it"


def test_a_dead_node_fails_the_rpc_rather_than_driving_straight_ahead(stub_node) -> None:
    """Silently substituting a plan would score the model on the bridge."""
    url, _ = stub_node
    planner = PolicyPlanner(url.replace(url.rsplit(":", 1)[1], "1"))  # nothing listening
    with pytest.raises(RuntimeError, match="did not answer"):
        planner.post_infer({"images": {}})


def test_a_camera_with_an_unusable_extrinsic_is_omitted_not_fatal(bridge) -> None:
    """One bad calibration must not kill a session the model could still drive."""
    stub, node = bridge
    request = DriveSessionRequest()
    request.session_uuid = "bad-cal"
    camera = request.rollout_spec.vehicle.available_cameras.add()
    camera.logical_id = "camera_front_wide_120fov"
    # left uninitialised: an all-zero quaternion, which geometry rejects
    stub.start_session(request)
    _push_egomotion(stub, session="bad-cal")
    _drive(stub, session="bad-cal")

    assert node.bodies, "the session should still have driven"
    assert node.bodies[-1]["calibration"]["camera2ego"] == {}


def test_two_sessions_are_kept_apart(bridge) -> None:
    """AlpaSim runs concurrent rollouts against one driver."""
    stub, node = bridge
    _open_session(stub, session="a")
    _open_session(stub, session="b")
    _push_egomotion(stub, session="a", timestamp_us=111_000)
    _push_egomotion(stub, session="b", timestamp_us=222_000)
    _drive(stub, session="a", now_us=111_000)
    _drive(stub, session="b", now_us=222_000)

    assert [b["timestamp"] for b in node.bodies] == [111_000, 222_000]
    stub.close_session(DriveSessionCloseRequest(session_uuid="a"))
    stub.close_session(DriveSessionCloseRequest(session_uuid="b"))


# ---------------------------------------------------------------------------
# the bridge as a subprocess, the way launcher.py starts it
# ---------------------------------------------------------------------------


def test_the_bridge_starts_as_a_subprocess_and_tears_down() -> None:
    """The launch path, not just the servicer.

    `serve()` is called across a process boundary with a different interpreter
    and a constructed PYTHONPATH, so an argument-order slip or a missing path
    shows up only here -- and did: the first live attempt died on
    `serve(scaffold, port=port)` when the real signature is
    `serve(host, port, driver)`. That is invisible to every in-process test.
    """
    import tempfile
    from pathlib import Path as _Path

    from paddock.adapters.benchmarks.alpasim import launcher
    from paddock.config import SiteConfig
    from paddock.runtime import ports

    try:
        cfg = SiteConfig.load()
    except Exception:  # noqa: BLE001
        pytest.skip("no site configuration")
    python = (cfg.benchmarks.get("alpasim") or {}).get("python")
    if not python or not _Path(python).exists():
        pytest.skip("[benchmarks.alpasim] python is not configured here")

    tmp = _Path(tempfile.mkdtemp())
    port = ports.find_free(9200)
    with launcher.BridgeProcess(
        python=python,
        policy_url="http://127.0.0.1:1",
        port=port,
        version_id="test-subprocess-0",
        log_path=tmp / "bridge.log",
    ) as bridge:
        assert bridge.proc is not None and bridge.proc.poll() is None
    assert bridge.proc.poll() is not None, "the bridge outlived its context manager"
