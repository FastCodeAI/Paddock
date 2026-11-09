"""The bridge's opt-in switches: off, every existing profile sends what it sent before.

Each profile's planner is built exactly as `paddock run` builds it -- `bridge.main` on
the profile's own `bridge_args` -- and driven through 4.5 s of a scene (warm-up,
handover, replans and the ticks between). Everything it sends the node, /reset,
/observe and /infer, in order, is hashed. The hashes were taken from `origin/main`
before the switches existed, so a profile whose bodies move by a single byte fails
here. The switches themselves are then checked one by one.

Runs in AlpaSim's environment, which supplies grpc and alpasim_utils for the bridge.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("grpc", reason="AlpaSim's environment supplies grpc")
pytest.importorskip("alpasim_utils", reason="AlpaSim's environment supplies alpasim_utils")

from paddock.adapters.benchmarks.alpasim import (
    bridge,  # noqa: E402
    harness,  # noqa: E402
)

STEP_US = 50_000
#: AlpaSim's ids for the six cameras any profile renames.
CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_front_tele_30fov",
    "camera_rear_right_70fov",
)


def _png(tick: int, camera_index: int) -> bytes:
    """A small, exactly reproducible frame. PNG, not JPEG: lossless, so the decoded
    pixels cannot depend on the libjpeg build the test runs on."""
    from PIL import Image

    rows = np.arange(12, dtype=np.uint16)[:, None, None]
    cols = np.arange(16, dtype=np.uint16)[None, :, None]
    chan = np.arange(3, dtype=np.uint16)[None, None, :]
    pixels = ((rows * 7 + cols * 3 + chan * 11 + tick * 5 + camera_index * 31) % 256).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


def _state(t_s: float) -> tuple[float, float, float]:
    """(x, y, yaw) of a car at 8 m/s turning left at 0.1 rad/s, far from the origin."""
    yaw = 0.3 + 0.1 * t_s
    return (
        1234.5 + 80.0 * (math.sin(yaw) - math.sin(0.3)),
        -567.25 - 80.0 * (math.cos(yaw) - math.cos(0.3)),
        yaw,
    )


def _quat(yaw: float) -> SimpleNamespace:
    return SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))


def _pose(t_us: int) -> SimpleNamespace:
    x, y, yaw = _state(t_us / 1e6)
    return SimpleNamespace(
        timestamp_us=t_us, pose=SimpleNamespace(quat=_quat(yaw), vec=SimpleNamespace(x=x, y=y, z=0.25))
    )


def _dynamic(t_us: int, warmup_until_us: int) -> SimpleNamespace:
    """What AlpaSim reports: zero through the forced replay, then the car's motion."""
    vec = SimpleNamespace
    if t_us <= warmup_until_us:
        return vec(
            linear_velocity=vec(x=0.0, y=0.0, z=0.0),
            angular_velocity=vec(x=0.0, y=0.0, z=0.0),
            linear_acceleration=vec(x=0.0, y=0.0, z=0.0),
        )
    return vec(
        linear_velocity=vec(x=8.0, y=0.05, z=0.01),
        angular_velocity=vec(x=0.0, y=0.0, z=0.1),
        linear_acceleration=vec(x=0.3, y=0.8, z=0.0),
    )


class _Intrinsics:
    def __init__(self, index: int) -> None:
        self.opencv_pinhole_param = SimpleNamespace(
            focal_length_x=500.0 + index,
            focal_length_y=501.0 + index,
            principal_point_x=8.0,
            principal_point_y=6.0,
        )

    @staticmethod
    def WhichOneof(_: str) -> str:  # noqa: N802  (protobuf's spelling)
        return "opencv_pinhole_param"


def _cameras() -> dict:
    return {
        camera: SimpleNamespace(
            intrinsics=_Intrinsics(index),
            rig_to_camera=SimpleNamespace(
                quat=_quat(0.1 * index), vec=SimpleNamespace(x=1.5 + index, y=0.1 * index, z=1.6)
            ),
        )
        for index, camera in enumerate(CAMERAS)
    }


def _route(t_us: int) -> SimpleNamespace:
    """A route bending left ahead, in the rig frame at its own time."""
    return SimpleNamespace(
        timestamp_us=t_us,
        waypoints=[SimpleNamespace(x=5.0 * k, y=0.02 * (5.0 * k) ** 2 / 10.0, z=0.0) for k in range(-1, 12)],
    )


def _view(now_us: int, warmup_until_us: int = 3_050_000) -> SimpleNamespace:
    times = list(range(max(0, now_us - 33 * STEP_US), now_us + 1, STEP_US))
    poses = [_pose(t) for t in times]
    return SimpleNamespace(
        session_uuid="session-1",
        debug_scene_id="clipgt-0123abcd-test",
        poses=poses,
        latest_pose=poses[-1],
        dynamic_states=[(t, _dynamic(t, warmup_until_us)) for t in times],
        latest_dynamic_state=_dynamic(times[-1], warmup_until_us),
        image_cache={
            camera: tuple((t, _png(t // STEP_US, i)) for t in times) for i, camera in enumerate(CAMERAS)
        },
        cameras=_cameras(),
        route=_route(now_us),
        drive_count=now_us // STEP_US + 1,
    )


def _build(args: tuple[str, ...], monkeypatch) -> bridge.PolicyPlanner:
    """The planner `bridge.main` builds for these flags, without serving it."""
    built = {}
    monkeypatch.setattr(
        bridge, "scaffold_serve", lambda host, port, scaffold: built.setdefault("s", scaffold)
    )
    bridge.main(["--policy-url", "http://127.0.0.1:1", "--port", "1", *args])
    return built["s"]._planner


def _drive(planner: bridge.PolicyPlanner, until_us: int = 4_500_000) -> list:
    """Every request the planner sends the node, in order, over ticks 0..until_us."""
    sent: list = []
    dt = planner._waypoint_dt
    plan = [[2.0 * (k + 1), 0.05 * (k + 1), 0.01 * (k + 1)] for k in range(int(round(6.0 / dt)))]

    def post(path, body):
        sent.append((path, json.loads(json.dumps(body))))
        return {"observed": True}

    def post_infer(body):
        sent.append(("/infer", json.loads(json.dumps(body))))
        return {"trajectory": [p[:2] for p in plan[:6]], "plan": {"waypoints": plan, "dt": dt}}

    planner._post = post
    planner.post_infer = post_infer
    planner.on_session_start(SimpleNamespace(session_uuid="session-1"))
    for t in range(0, until_us + 1, STEP_US):
        planner.plan(_view(t), time_now_us=t, time_query_us=t + STEP_US)
    return sent


def _digest(sent: list) -> str:
    return hashlib.sha256(json.dumps(sent, sort_keys=True).encode()).hexdigest()


#: sha256 of everything each profile sent, computed on `origin/main` (9819e36)
#: before any of the switches existed. A change here is a change to what that
#: model's node receives.
BEFORE = {
    "default": "d2cb4e34e5831301de403a7e6b1bf18b9698e7e004b72473476efd69cf162c40",
    "drivejepa": "273b00c85ac84499389875c731e4f5aa77a426b2e09f920cde928e348a3df35d",
    "autoe2e": "b3640279eff26e6cbc6d31b46ff09139eb1b87307d671951a1162ba6002c53a9",
}


@pytest.mark.parametrize("profile", sorted(BEFORE))
def test_existing_profiles_send_what_they_sent_before_the_switches(profile, monkeypatch) -> None:
    args = harness.profile_for(profile).bridge_args if profile != "default" else ()
    sent = _drive(_build(args, monkeypatch))
    assert sent, "the planner sent nothing"
    assert _digest(sent) == BEFORE[profile]


#: sha256 of everything each profile sent, computed at 1fa1448, before the Alpamayo 1.5
#: profile and its two switches (--send-route-time, --send-plan-dt) existed.
BEFORE_ALPAMAYO = {
    **BEFORE,
    "qwendrive": "a6fc5fcec45834b51a1272dcb39fbd0cafb826606a4973021119297a64d2ae4a",
}


@pytest.mark.parametrize("profile", sorted(BEFORE_ALPAMAYO))
def test_existing_profiles_send_what_they_sent_before_the_alpamayo_profile(profile, monkeypatch) -> None:
    args = harness.profile_for(profile).bridge_args if profile != "default" else ()
    assert _digest(_drive(_build(args, monkeypatch))) == BEFORE_ALPAMAYO[profile]


@pytest.mark.parametrize("profile", ["default", *harness.profiled_models()])
def test_only_the_alpamayo_profile_sends_a_plan_spacing_or_a_route_time(profile, monkeypatch) -> None:
    """Every profile, including any added later: the two keys reach a node only under alpamayo15."""
    args = harness.profile_for(profile).bridge_args if profile != "default" else ()
    keys = {key for path, body in _drive(_build(args, monkeypatch)) for key in body}
    sent = {"plan_dt", "route_timestamp_us"} & keys
    assert sent == ({"plan_dt", "route_timestamp_us"} if profile == "alpamayo15" else set())


# ---------------------------------------------------------------------------
# the switches, one by one, through the qwendrive profile
# ---------------------------------------------------------------------------


def _qwendrive(monkeypatch) -> bridge.PolicyPlanner:
    return _build(harness.profile_for("qwendrive").bridge_args, monkeypatch)


def test_the_qwendrive_profile_selects_the_replan_rhythm_with_its_switches(monkeypatch) -> None:
    from paddock.adapters.benchmarks.alpasim.replan_rhythm import ReplanRhythmPlanner

    planner = _qwendrive(monkeypatch)
    assert isinstance(planner, ReplanRhythmPlanner)
    assert planner._replan_every_us == 1_500_000 and planner._observe_between
    assert planner._waypoint_dt == 0.1
    assert planner._canbus_ego_frame and planner._send_route and planner._send_scene_id
    assert planner._named_cameras_only and not planner._canbus_acceleration


def test_every_step_reaches_the_node_and_the_model_is_asked_every_1_5_s(monkeypatch) -> None:
    sent = _drive(_qwendrive(monkeypatch), until_us=6_000_000)
    assert sent[0][0] == "/reset"
    infer_times = [body["timestamp"] for path, body in sent if path == "/infer"]
    assert infer_times == [3_000_000, 4_500_000, 6_000_000]
    times = [body["timestamp"] for path, body in sent if path != "/reset"]
    assert times == list(range(0, 6_000_001, STEP_US)), "one body per 50 ms step, none missing"


def test_only_the_three_named_cameras_are_sent_under_the_nodes_names(monkeypatch) -> None:
    sent = _drive(_qwendrive(monkeypatch), until_us=3_000_000)
    for _, body in sent[1:]:
        assert sorted(body["images"]) == ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
        assert sorted(body["calibration"]["camera2ego"]) == ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
        # No intrinsics without --pinhole-intrinsics: the lenses are f-theta, and the model reads none.
        assert body["calibration"]["camera2image"] == {}


def test_canbus_carries_forward_and_sideways_speed_and_the_warmup_fix(monkeypatch) -> None:
    from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, CANBUS_YAW_RATE

    sent = {
        body["timestamp"]: body
        for path, body in _drive(_qwendrive(monkeypatch), 4_500_000)
        if path != "/reset"
    }
    # After the forced replay: what the simulator reports, verbatim.
    after = sent[4_500_000]["canbus"]
    assert (after[CANBUS_VELOCITY], after[CANBUS_VELOCITY + 1], after[CANBUS_YAW_RATE]) == (8.0, 0.05, 0.1)
    assert after[CANBUS_ACCEL] == after[CANBUS_ACCEL + 1] == 0.0, "acceleration stays off in this profile"
    # During it: velocity and yaw rate from the last two poses (8 m/s on an 80 m circle).
    during = sent[2_000_000]["canbus"]
    assert during[CANBUS_VELOCITY] == pytest.approx(8.0, abs=1e-3)
    assert during[CANBUS_VELOCITY + 1] == pytest.approx(8.0 * math.sin(0.0025), abs=1e-4)
    assert during[CANBUS_YAW_RATE] == pytest.approx(0.1, abs=1e-6)
    # The first tick has one pose: nothing to derive from, so the reported zero stays.
    assert sent[0]["canbus"][CANBUS_VELOCITY] == 0.0


def test_a_car_standing_still_in_the_warmup_keeps_its_zero() -> None:
    still = _view(1_000_000)
    for pose in still.poses:
        pose.pose.vec.x, pose.pose.vec.y = 10.0, 20.0
    assert bridge.ego_frame_motion(still)[:3] == (0.0, 0.0, 0.0)
    crawling = _view(1_000_000)
    for k, pose in enumerate(crawling.poses):
        pose.pose.vec.x, pose.pose.vec.y = 10.0 + 0.04 * k, 20.0  # 0.8 m/s
    assert bridge.ego_frame_motion(crawling)[3] is False


def test_the_acceleration_switch_sends_the_simulators_value() -> None:
    from paddock.policy.protocol import CANBUS_ACCEL

    view = _view(4_000_000)
    canbus = bridge.canbus_from(view, acceleration=True)
    assert (canbus[CANBUS_ACCEL], canbus[CANBUS_ACCEL + 1]) == (0.3, 0.8)
    assert not bridge.canbus_from(view)[CANBUS_ACCEL : CANBUS_ACCEL + 2].any()


def test_the_route_and_scene_id_ride_along_as_alpasim_sent_them(monkeypatch) -> None:
    sent = _drive(_qwendrive(monkeypatch), 3_000_000)
    body = sent[-1][1]
    assert body["scene_id"] == "clipgt-0123abcd-test"
    assert body["route_xy"] == [[w.x, w.y] for w in _route(3_000_000).waypoints]
    view = _view(0)
    view.route = None
    assert bridge.route_xy(view.route) is None


@pytest.mark.parametrize(
    "args",
    [
        ["--observe-between-replans"],
        ["--named-cameras-only"],
        [
            "--planner",
            "drivejepa-0.1.4b",
            "--front-camera",
            "c",
            "--frame-interval-us",
            "50000",
            "--force-gt-us",
            "3000000",
            "--send-route",
        ],
    ],
)
def test_switches_that_would_do_nothing_are_refused(args, monkeypatch) -> None:
    monkeypatch.setattr(bridge, "serve", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        bridge.main(["--policy-url", "http://x", "--port", "1", *args])


def test_the_qwendrive_node_reads_these_bodies_as_the_native_driver_built_its_inputs(monkeypatch) -> None:
    """Bridge -> wire -> Qwen-Drive's node, with a stand-in model: no refusals, frames 0.5 s
    apart, the sideways speed of a turning car, the warm-up speed fixed, and 50 points at 0.1 s
    accepted by the bridge."""
    from paddock.adapters.models.qwendrive.policy import OUTPUT_POINTS, POLICY
    from paddock.policy.protocol import Observation

    node = POLICY(planner="planner-sft", mode="direct", num_samples=1, attn="sdpa", views="ring")
    calls = []

    class Model:
        def generate_trajectory(self, scene, mode, num_samples, seed):
            calls.append((scene, seed))
            t = (np.arange(OUTPUT_POINTS) + 1) * 0.1
            return SimpleNamespace(trajectories=np.stack([8.0 * t, 0 * t, 0 * t], 1)[None], reasoning=None)

    node.model, node._mode = Model(), "direct_planning"
    node._scene = lambda obs, inputs: inputs
    planner = _qwendrive(monkeypatch)

    def post(path, body):
        if path == "/reset":
            node.reset()
        else:
            node.observe(Observation.from_wire(json.loads(json.dumps(body)), cameras=node.cameras))
        return {"observed": True}

    def post_infer(body):
        return node.infer(Observation.from_wire(json.loads(json.dumps(body)), cameras=node.cameras)).to_wire()

    planner._post, planner.post_infer = post, post_infer
    planner.on_session_start(SimpleNamespace(session_uuid="session-1"))
    results = [planner.plan(_view(t), t, t + STEP_US) for t in range(0, 6_000_001, STEP_US)]
    assert [r is not None for r in results].count(True) == 61, "a plan at every step from handover (3.0 s) on"
    assert len(calls) == 3
    first, last = calls[0][0], calls[-1][0]
    assert first["frame_gaps_s"] == [0.0, 0.0, 0.0, 0.0] and last["history_fill"] == 0
    # Handover plan: every row is a warm-up row, each with the pose-derived speed.
    np.testing.assert_allclose(first["history_velocity"][:, 0], 8.0, atol=1e-3)
    # Two plans later the rows are what the simulator reported: sideways 0.05 m/s, and the
    # turn's 0.1 rad/s * 8 m/s = 0.8 m/s^2 sideways acceleration.
    np.testing.assert_allclose(last["history_velocity"][:, 1], 0.05)
    np.testing.assert_allclose(last["history_acceleration"][1:, 1], 0.8, atol=1e-9)
    assert last["driving_command"] in ([0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    from paddock.adapters.models.qwendrive.policy import scene_seed

    assert {seed for _, seed in calls} == {scene_seed("clipgt-0123abcd-test")}
    assert results[-1].step_us == 100_000


# -- Alpamayo 1.5 ---------------------------------------------------------------


def _alpamayo(monkeypatch) -> bridge.PolicyPlanner:
    return _build(harness.profile_for("alpamayo15").bridge_args, monkeypatch)


def test_alpamayo_asks_every_100_ms_after_the_warmup_and_observes_every_other_step(monkeypatch) -> None:
    """60 warm-up steps observed, then a replan on every other step and the step between observed."""
    sent = _drive(_alpamayo(monkeypatch))
    paths = [path for path, _ in sent]
    assert paths[0] == "/reset"
    infers = [body["timestamp"] for path, body in sent if path == "/infer"]
    assert infers == list(range(3_000_000, 4_500_001, 100_000))
    observed = [body["timestamp"] for path, body in sent if path == "/observe"]
    assert observed[:60] == list(range(0, 3_000_000, STEP_US))
    assert observed[60:] == list(range(3_050_000, 4_500_000, 100_000))


def test_alpamayo_gets_its_four_cameras_by_name_with_no_intrinsics(monkeypatch) -> None:
    """The four AlpaSim cameras its native driver reads, renamed, and no pinhole K: they are
    f-theta lenses, which the node passes on as rendered."""
    (_, body) = next(item for item in _drive(_alpamayo(monkeypatch)) if item[0] == "/infer")
    assert sorted(body["images"]) == ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_FRONT_TELE"]
    assert body["calibration"]["camera2image"] == {}
    assert sorted(body["calibration"]["camera2ego"]) == sorted(body["images"])


def test_alpamayo_gets_the_route_its_time_and_the_plan_spacing(monkeypatch) -> None:
    """The route as AlpaSim sent it, the time of its frame, and plan_dt 0.1; no canbus switch
    and no scene id, which the model never reads."""
    (_, body) = next(item for item in _drive(_alpamayo(monkeypatch)) if item[0] == "/infer")
    route = _route(body["timestamp"])
    assert body["route_xy"] == [[w.x, w.y] for w in route.waypoints]
    assert body["route_timestamp_us"] == route.timestamp_us == body["timestamp"]
    assert body["plan_dt"] == 0.1
    assert "scene_id" not in body
    default = next(item for item in _drive(_build((), monkeypatch)) if item[0] == "/infer")[1]
    assert body["canbus"] == default["canbus"]


def test_the_alpamayo_node_reads_these_bodies_with_every_alpasim_field(monkeypatch) -> None:
    from paddock.policy.protocol import Observation

    (_, body) = next(item for item in _drive(_alpamayo(monkeypatch)) if item[0] == "/infer")
    cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_FRONT_TELE")
    obs = Observation.from_wire(body, cameras=cameras)
    assert sorted(obs.images) == sorted(cameras)
    assert obs.camera2image == {}
    assert obs.plan_dt == 0.1
    assert obs.route_timestamp_us == body["timestamp"]
    assert obs.route_xy.shape == (13, 2)


@pytest.mark.parametrize(
    "args",
    [
        ("--send-route-time",),
        ("--replan-every-us", "100000", "--frame-interval-us", "50000", "--force-gt-us", "3000000")
        + ("--send-route-time",),
    ],
)
def test_a_route_time_without_a_route_is_refused(args, monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _build(args, monkeypatch)


def test_the_new_switches_leave_a_body_alone_until_turned_on(monkeypatch) -> None:
    """Off, no plan_dt and no route_timestamp_us; on, exactly those two keys are added."""
    plain = next(item for item in _drive(_build(("--send-route",), monkeypatch)) if item[0] == "/infer")[1]
    both = ("--send-route", "--send-route-time", "--send-plan-dt")
    extended = next(item for item in _drive(_build(both, monkeypatch)) if item[0] == "/infer")[1]
    assert "plan_dt" not in plain and "route_timestamp_us" not in plain
    assert {k: v for k, v in extended.items() if k not in ("plan_dt", "route_timestamp_us")} == plain
    assert extended["plan_dt"] == 0.5
