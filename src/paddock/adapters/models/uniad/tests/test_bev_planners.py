"""What the four BEVFormer planners are fed and how their plans come back, checked in paddock's own env.

No model, no GPU, no weights. The inputs are checked against what the original tools themselves fed each
model on the golden frames (``fixtures/golden_<model>.json``, written by ``golden_reference.py`` from the
repos' own converters, datasets and pipelines): lidar2img, the CAN bus, l2g and the command must come out the
same from a paddock Observation. The frame of the plan, the commands, the 0.5 s history grid and the policy
contract are checked on synthetic cases. ``test_golden.py`` then runs the models themselves.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from paddock import conformance
from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.uniad.tests import golden_driver
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_YAW_RATE, Command, Observation
from paddock.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ("uniad", "vad")
#: A lidar in exactly LIDAR_TOP's axes (x right, y forward, z up), 0.4 m behind the ego origin and 1.8 m up:
#: the lever-arm and axis tests' mount, where no calibration tilt blurs the arithmetic.
AXIS_ALIGNED_LIDAR2EGO = np.array(
    [
        [0.0, 1.0, 0.0, -0.4],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.8],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _fixture(model: str) -> dict:
    return json.loads((FIXTURES / f"golden_{model}.json").read_text())


def _observations(model: str, with_quaternion: bool) -> list[Observation]:
    command = Command.STRAIGHT
    return golden_driver.observations(_fixture(model), "/nonexistent", with_quaternion, command)


# -- inputs, against the original tools' own ----------------------------------------------------------


@pytest.mark.parametrize("model", ["uniad", "vad"])
@pytest.mark.parametrize("variant", ["stored_quaternion", "canonical_quaternion"])
def test_nuscenes_inputs_equal_the_datasets(model: str, variant: str) -> None:
    """lidar2img, can_bus, l2g, timestamp and command as get_data_info + the test pipeline gave them."""
    seen = _fixture(model)["variants"][variant]["frames"]
    previous = None
    for obs, want in zip(_observations(model, variant == "stored_quaternion"), seen, strict=True):
        lidar2ego, source = bp.model_lidar2ego(obs.lidar2ego, bp.NUSC_LIDAR2EGO)
        assert source == "observation", "nuScenes' own LIDAR_TOP is in LIDAR_TOP's axes"
        # VAD's test pipeline scales every camera by 0.8 after normalising, lidar2img with it.
        scale = np.diag([0.8, 0.8, 1.0, 1.0]) if model == "vad" else np.eye(4)
        projections = [
            scale @ bp.lidar2img(obs.camera2image[c], obs.camera2ego[c], lidar2ego) for c in bp.NUSC_CAMERAS
        ]
        np.testing.assert_allclose(projections, want["lidar2img"], rtol=1e-9, atol=1e-6)

        given = bp.source_quaternion(obs.canbus, obs.ego2world)
        assert (given is not None) == (variant == "stored_quaternion")
        quaternion = given if given is not None else bp.continuous_quaternion(obs.ego2world[:3, :3], previous)
        previous = quaternion
        can_bus, gravity = bp.nuscenes_can_bus(obs.ego2world, obs.canbus, quaternion)
        assert not gravity, "nuScenes' CAN-bus accel already holds gravity"
        np.testing.assert_allclose(can_bus, want["can_bus"], rtol=1e-12, atol=1e-9)

        if model == "uniad":
            l2g_r, l2g_t = bp.nuscenes_l2g(obs.ego2world, lidar2ego)
            np.testing.assert_array_equal(l2g_r, np.asarray(want["l2g_r_mat"], dtype=np.float32))
            np.testing.assert_array_equal(l2g_t, np.asarray(want["l2g_t"], dtype=np.float32))
            assert obs.timestamp_us / 1e6 == want["timestamp"]
            assert bp.nuscenes_command(obs) == want["command"]
        else:
            assert bp.nuscenes_command(obs) == int(np.argmax(want["ego_fut_cmd"]))


def test_the_nuscenes_can_bus_holds_w_four_times() -> None:
    """The datasets assign a pyquaternion Quaternion into a numpy slice, which broadcasts its w."""
    pose = np.eye(4)
    pose[:3, :3] = bp.quaternion_matrix([0.9, 0.0, 0.0, math.sqrt(1 - 0.81)])
    can_bus, _ = bp.nuscenes_can_bus(pose, np.zeros(18), bp.matrix_quaternion(pose[:3, :3]))
    np.testing.assert_allclose(can_bus[3:7], [0.9] * 4)


@pytest.mark.parametrize(("z", "added"), [(0.0, True), (0.3, True), (9.57, False), (9.81, False)])
def test_gravity_is_added_only_where_the_source_left_it_out(z: float, added: bool) -> None:
    out, was_added = bp.accel_with_gravity([0.4, -0.1, z])
    assert was_added is added
    assert out[2] == pytest.approx(z + (bp.GRAVITY if added else 0.0))


def test_a_source_quaternion_is_kept_only_when_it_is_the_ego_poses() -> None:
    q = np.array([-0.96275, -0.0074, -0.0062, 0.27022])
    q /= np.linalg.norm(q)
    pose = np.eye(4)
    pose[:3, :3] = bp.quaternion_matrix(q)
    canbus = np.zeros(18)
    assert bp.source_quaternion(canbus, pose) is None, "zeros (HUGSIM, Bench2Drive) are no quaternion"
    canbus[3:7] = q
    np.testing.assert_array_equal(bp.source_quaternion(canbus, pose), q)  # w < 0 kept
    canbus[3:7] = [0.9547, 0.0, 0.0, -0.2975]  # nuScenes' CAN-bus orientation: another rotation
    assert bp.source_quaternion(canbus / np.linalg.norm(canbus[3:7]), pose) is None


def test_a_derived_quaternion_keeps_its_sign_through_an_episode() -> None:
    previous = None
    for yaw in np.linspace(2.0, 4.5, 30):  # through 180 degrees, where w crosses zero
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
        )
        q = bp.continuous_quaternion(rotation, previous)
        np.testing.assert_allclose(bp.quaternion_matrix(q), rotation, atol=1e-12)
        if previous is None:
            assert q[0] >= 0
        else:
            assert float(q @ previous) > 0
        previous = q


# -- the lidar and the cameras ------------------------------------------------------------------------


def test_paddocks_identity_lidar_is_replaced_by_the_models_mount() -> None:
    lidar2ego, source = bp.model_lidar2ego(np.eye(4), bp.NUSC_LIDAR2EGO)
    assert source == "nominal"
    np.testing.assert_array_equal(lidar2ego, bp.NUSC_LIDAR2EGO)
    assert bp.model_lidar2ego(None, bp.NUSC_LIDAR2EGO)[1] == "nominal"
    assert bp.has_lidar_top_axes(bp.NUSC_LIDAR2EGO) and bp.has_lidar_top_axes(AXIS_ALIGNED_LIDAR2EGO)
    assert not bp.has_lidar_top_axes(np.eye(4))


def test_intrinsics_scale_with_the_resize() -> None:
    k = np.array([[630.0, 0, 400.0], [0, 630.0, 225.0], [0, 0, 1.0]])
    np.testing.assert_allclose(bp.scaled_intrinsics(k, 2.0, 2.0), [[1260, 0, 800], [0, 1260, 450], [0, 0, 1]])
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    same, same_k = bp.to_load_size(image, k)  # already the load size: no cv2 needed, nothing changes
    assert same is image
    np.testing.assert_array_equal(same_k, k)


# -- the plan ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lidar2ego", [AXIS_ALIGNED_LIDAR2EGO, bp.NUSC_LIDAR2EGO], ids=["axis_aligned", "nuscenes"]
)
def test_a_straight_plan_changes_only_axes(lidar2ego: np.ndarray) -> None:
    """On a straight road the lever arm cancels: lidar (x right, y forward) -> ego (x forward, y left)."""
    plan = np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]  # 8 m/s along the lidar's own line
    ego = bp.lidar_plan_to_ego(plan, lidar2ego)
    # Only the lidar's tilt (~2 degrees on nuScenes) moves anything; an axis-aligned mount is exact.
    tolerance = 1e-12 if lidar2ego is AXIS_ALIGNED_LIDAR2EGO else 0.06
    np.testing.assert_allclose(ego, np.c_[4.0 * np.arange(1, 7), np.zeros(6)], atol=tolerance)


def test_right_in_the_lidar_frame_is_negative_y() -> None:
    """lidar x is right (nuScenes LIDAR_TOP is yawed -90 degrees), paddock's y is left."""
    plan = np.c_[0.1 * np.arange(1, 7), 4.0 * np.arange(1, 7)]
    assert (bp.lidar_plan_to_ego(plan, AXIS_ALIGNED_LIDAR2EGO)[:, 1] < 0).all()


@pytest.mark.parametrize("turn", [-1.0, 1.0], ids=["right", "left"])
def test_through_a_turn_the_ego_origin_is_recovered(turn: float) -> None:
    """A rear axle on a 25 m arc: its lidar's positions, as the model would plan them, give the arc back."""
    radius, speed, arm = 25.0, 8.0, AXIS_ALIGNED_LIDAR2EGO[:2, 3]
    times = 0.5 * np.arange(1, 7)
    heading = turn * speed * times / radius
    rear = np.c_[radius * np.sin(np.abs(heading)), turn * radius * (1 - np.cos(heading))]
    rotations = np.stack([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]).transpose(
        2, 0, 1
    )
    lidar_in_ego = rear + rotations @ arm
    lidar_frame = (np.c_[lidar_in_ego, np.zeros(6)] - AXIS_ALIGNED_LIDAR2EGO[:3, 3]) @ AXIS_ALIGNED_LIDAR2EGO[
        :3, :3
    ]
    recovered = bp.lidar_plan_to_ego(lidar_frame[:, :2], AXIS_ALIGNED_LIDAR2EGO)
    # The heading comes from the path (a chord at the last waypoint), so a few centimetres remain...
    np.testing.assert_allclose(recovered, rear, atol=0.05)
    # ...where shifting by the lever arm without turning it is off by a third of a metre at 3 s.
    static = lidar_in_ego - arm
    assert np.abs(static - rear).max() > 0.25
    assert np.sign(rear[-1, 1]) == turn, "y is left"


# -- commands ------------------------------------------------------------------------------------------


def _with_command(
    command: Command, lane_change: str | None = None, metres: float | None = None, route: bool = True
) -> Observation:
    obs = conformance.synthetic_observation(cameras=("CAM_FRONT",))
    obs.command = command
    obs.nav_lane_change, obs.nav_lane_change_m = lane_change, metres
    obs.route_xy = np.c_[np.arange(0.0, 50.0, 5.0), np.zeros(10)] if route else None
    return obs


def test_nuscenes_commands_follow_the_label_code() -> None:
    """UniAD get_sdc_planning_label and VAD's converter: 0 right, 1 left, 2 forward -- paddock's own order."""
    assert [bp.nuscenes_command(_with_command(c)) for c in Command] == [0, 1, 2, 2]


def test_nuscenes_commands_ignore_the_route() -> None:
    assert [bp.nuscenes_command(_with_command(c, route=False)) for c in Command] == [0, 1, 2, 2]


# -- the command from AlpaSim's route ------------------------------------------------------------------


def _path(radius: float, side: int, straight_m: float, distance_m: float) -> tuple[float, float, float]:
    """(x, y, heading) after `distance_m` along a path straight for `straight_m`, then a circle of `radius`
    turning left (`side` +1) or right (-1); ego frame, y left."""
    if distance_m <= straight_m:
        return distance_m, 0.0, 0.0
    angle = (distance_m - straight_m) / radius
    return straight_m + radius * math.sin(angle), side * radius * (1 - math.cos(angle)), side * angle


def _route(radius: float, side: int, straight_m: float, step_m: float = 4.0, points: int = 20) -> np.ndarray:
    """A route like AlpaSim's: `points` waypoints `step_m` apart along `_path`, from the car."""
    return np.array([_path(radius, side, straight_m, k * step_m)[:2] for k in range(points)])


def _label_lateral(radius: float, side: int, straight_m: float, distance_m: float) -> float:
    """What UniAD's get_sdc_planning_label and VAD's converter compute for a car that covers `distance_m`
    along `_path` in 3 s: the lidar's future pose (the ego pose there, then lidar2ego), its origin read in
    the current lidar frame, x (+ right)."""
    x, y, heading = _path(radius, side, straight_m, distance_m)
    ego = np.eye(4)
    ego[:2, :2] = [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
    ego[:2, 3] = x, y
    future_lidar_origin = (ego @ bp.NUSC_LIDAR2EGO)[:, 3]
    return float((np.linalg.inv(bp.NUSC_LIDAR2EGO) @ future_lidar_origin)[0])


#: (radius m, side, straight before the curve m): a junction turn starting 5 m ahead, a gentle 200 m bend,
#: a turn 30 m ahead, and no turn at all.
SHAPES = {
    "left_turn": (15.0, 1, 5.0),
    "right_turn": (15.0, -1, 5.0),
    "gentle_left": (200.0, 1, 0.0),
    "turn_30m_ahead": (15.0, -1, 30.0),
    "straight": (1e9, 1, 0.0),
}


@pytest.mark.parametrize("speed", [0.0, 2.0, 5.0, 10.0, 15.0])
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_route_command_is_the_label_codes_rule_applied_to_the_route(shape: str, speed: float) -> None:
    """The lidar 3 s ahead (speed * 3 s along the route, at least the minimum lookahead), in the current
    lidar frame, +-2 m: on a densely sampled route, the same number the label code gets from the path."""
    radius, side, straight = SHAPES[shape]
    record = bp.route_command(
        _route(radius, side, straight, step_m=0.25, points=400), speed, bp.NUSC_LIDAR2EGO
    )
    distance = max(speed * 3.0, bp.COMMAND_MIN_LOOKAHEAD_M)
    want = _label_lateral(radius, side, straight, distance)
    assert record["lookahead_m"] == pytest.approx(distance)
    assert record["lateral_m"] == pytest.approx(want, abs=0.01)
    expected = bp.NUSC_RIGHT if want >= 2.0 else bp.NUSC_LEFT if want <= -2.0 else bp.NUSC_FORWARD
    assert record["command"] == expected
    assert not record["route_end"] and record["route_offset_m"] == pytest.approx(0.0, abs=1e-9)


#: On AlpaSim's own route (20 points 4 m apart), the command by speed, m/s: 0, 2, 5, 10, 15.
EXPECTED = {
    "straight": ["FORWARD"] * 5,
    "left_turn": ["FORWARD", "FORWARD", "LEFT", "LEFT", "LEFT"],
    "right_turn": ["FORWARD", "FORWARD", "RIGHT", "RIGHT", "RIGHT"],
    "gentle_left": ["FORWARD", "FORWARD", "FORWARD", "LEFT", "LEFT"],
    "turn_30m_ahead": ["FORWARD", "FORWARD", "FORWARD", "FORWARD", "RIGHT"],
}


@pytest.mark.parametrize("shape", sorted(EXPECTED))
def test_turns_on_alpasims_route_by_speed(shape: str) -> None:
    """Left is negative x in the lidar frame, right positive; a turn is called once 3 s of travel reaches
    2 m into it, so a faster car hears of it sooner and a standing car not at all."""
    radius, side, straight = SHAPES[shape]
    route = _route(radius, side, straight)
    names = [bp.route_command(route, v, bp.NUSC_LIDAR2EGO)["name"] for v in (0.0, 2.0, 5.0, 10.0, 15.0)]
    assert names == EXPECTED[shape]


def test_the_car_is_placed_on_the_route_and_the_route_may_run_out() -> None:
    """A route starting behind the car gives what the route from the car gives; one shorter than the
    lookahead is read at its last point, as the label code reads the last future keyframe a scene has."""
    radius, side, straight = SHAPES["left_turn"]
    ahead = _route(radius, side, straight)
    behind = np.vstack([[[-8.0, 0.0], [-4.0, 0.0]], ahead])
    for speed in (0.0, 5.0, 10.0):
        a, b = (bp.route_command(r, speed, bp.NUSC_LIDAR2EGO) for r in (ahead, behind))
        assert a == b
    short = bp.route_command(ahead[:4], 15.0, bp.NUSC_LIDAR2EGO)
    assert short["route_end"] and short["lookahead_m"] == 45.0
    assert short == {
        **bp.route_command(ahead[:4], 4.0, bp.NUSC_LIDAR2EGO),
        "lookahead_m": 45.0,
        "speed_m_s": 15.0,
    }


def test_alpasims_nan_padding_is_dropped() -> None:
    route = _route(*SHAPES["left_turn"])
    padded = np.vstack([route[:6], np.full((14, 2), np.nan)])
    assert bp.route_command(padded, 5.0, bp.NUSC_LIDAR2EGO) == bp.route_command(
        route[:6], 5.0, bp.NUSC_LIDAR2EGO
    )
    assert bp.route_command(np.full((20, 2), np.nan), 5.0, bp.NUSC_LIDAR2EGO) is None
    assert bp.route_command(route[:1], 5.0, bp.NUSC_LIDAR2EGO) is None


def _alpasim(
    command: Command, route: np.ndarray | None, stamped: bool = True, speed: float = 8.0
) -> Observation:
    obs = conformance.synthetic_observation(
        speed=speed, cameras=bp.NUSC_CAMERAS, width=1600, height=900, timestamp_us=3_500_000
    )
    obs.command = command
    obs.route_xy = route
    obs.route_timestamp_us = obs.timestamp_us if stamped and route is not None else None
    return obs


def test_only_follow_reference_with_a_stamped_route_is_decided_by_the_route() -> None:
    """AlpaSim's bridge sends FOLLOW_REFERENCE, the route and its frame's time; the route decides. Without
    the time (Bench2Drive's client) or without a route, FOLLOW_REFERENCE stays FORWARD; a command the source
    gives (LEFT, RIGHT, STRAIGHT) is never overridden."""
    left = _route(*SHAPES["left_turn"])
    record = bp.command_record(_alpasim(Command.FOLLOW_REFERENCE, left))
    assert (record["source"], record["name"], record["given"]) == ("route", "LEFT", "FOLLOW_REFERENCE")
    assert bp.nuscenes_command(_alpasim(Command.FOLLOW_REFERENCE, left)) == bp.NUSC_LEFT
    for obs in (
        _alpasim(Command.FOLLOW_REFERENCE, left, stamped=False),
        _alpasim(Command.FOLLOW_REFERENCE, None),
        _alpasim(Command.FOLLOW_REFERENCE, np.full((20, 2), np.nan)),
    ):
        record = bp.command_record(obs)
        assert (record["source"], record["command"]) == ("given", bp.NUSC_FORWARD)
    for given, want in ((Command.RIGHT, 0), (Command.LEFT, 1), (Command.STRAIGHT, 2)):
        assert bp.command_record(_alpasim(given, _route(*SHAPES["right_turn"]))) == {
            "source": "given",
            "command": want,
            "name": bp.NUSC_COMMAND_NAMES[want],
            "given": given.name,
        }


def test_a_route_in_another_frame_is_refused() -> None:
    obs = _alpasim(Command.FOLLOW_REFERENCE, _route(*SHAPES["left_turn"]))
    obs.route_timestamp_us = obs.timestamp_us - 50_000
    with pytest.raises(ValueError, match="route_xy is in the ego frame at 3450000 us"):
        bp.command_record(obs)


@pytest.mark.parametrize("model", ["uniad", "vad"])
def test_the_nuscenes_datasets_input_carries_the_routes_command(model: str) -> None:
    """Through the policy's own `_results`: UniAD's ``command`` and VAD's ``ego_fut_cmd`` one-hot
    [right, left, straight], and the record in ``last`` that the diagnostics serve."""
    import importlib

    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY()
    policy._box_type = policy._box_mode = None  # set by _build in the model's environment; unread here
    obs = _alpasim(Command.FOLLOW_REFERENCE, _route(*SHAPES["right_turn"]))
    results = policy._results(obs, bp.NUSC_LIDAR2EGO)
    assert results["command"] == bp.NUSC_RIGHT
    if model == "vad":
        assert results["ego_fut_cmd"].tolist() == [1, 0, 0]
    assert policy.last["command"] == bp.NUSC_RIGHT
    assert policy.last["command_record"]["source"] == "route"


class CommandStub(bp.BevPlannerPolicy):
    """The shared machinery with the command recorded as `Mmdet3dPlannerPolicy._results` records it."""

    cameras = bp.NUSC_CAMERAS

    def _results(self, obs, lidar2ego):
        record = bp.command_record(obs, lidar2ego)
        self.last.update(command=record["command"], command_record=record)
        return {}

    def _forward(self, results):
        return np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]

    def _reset_model(self) -> None:
        pass


def test_a_command_the_route_decided_travels_with_the_plan() -> None:
    """Under ``extras["route_command"]``, which AlpaSim's bridge writes to the run's per-replan log; any
    other command leaves the answer as it was, with no extras."""
    policy = CommandStub()
    policy.model = object()
    routed = policy.infer(_alpasim(Command.FOLLOW_REFERENCE, _route(*SHAPES["left_turn"])))
    assert routed.extras == {"route_command": policy.last["command_record"]}
    assert routed.extras["route_command"]["name"] == "LEFT"
    policy.reset()
    plain = policy.infer(_alpasim(Command.FOLLOW_REFERENCE, _route(*SHAPES["left_turn"]), stamped=False))
    assert plain.extras is None and "extras" not in plain.to_wire()


# -- history -------------------------------------------------------------------------------------------


def test_history_is_fed_on_the_half_second_grid() -> None:
    at_2hz = [i * 500_000 for i in range(5)]
    assert bp.history_on_grid(at_2hz[:-1], at_2hz[-1], None) == [0, 1, 2, 3]
    at_10hz = [i * 100_000 for i in range(21)]
    assert bp.history_on_grid(at_10hz[:-1], at_10hz[-1], None) == [0, 5, 10, 15]
    assert bp.history_on_grid(at_10hz[:-1], at_10hz[-1], after_us=1_000_000) == [15]
    assert bp.history_on_grid([], 2_000_000, None) == []


# -- the policy contract, on a stub model --------------------------------------------------------------


class StubPlanner(bp.BevPlannerPolicy):
    """The shared machinery with the model replaced: plans 8 m/s straight ahead, records what it was fed."""

    cameras = bp.NUSC_CAMERAS
    config_env, checkpoint_env = "STUB_CONFIG", "STUB_CHECKPOINT"

    def __init__(self) -> None:
        super().__init__()
        self.config = self.checkpoint = __file__
        self.fed: list[int] = []

    def _build(self) -> None:
        self.model = object()

    def _results(self, obs, lidar2ego):
        return {"t": obs.timestamp_us}

    def _forward(self, results):
        self.fed.append(results["t"])
        return np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]

    def _reset_model(self) -> None:
        self.fed = []


def test_the_shared_machinery_obeys_the_policy_contract() -> None:
    findings = conformance.policy_findings(StubPlanner())
    failures = [f for f in findings if f.failed]
    assert not failures, "\n" + "\n".join(f.render() for f in failures)


def test_an_open_loop_sequence_steps_the_model_on_the_grid_only() -> None:
    policy = StubPlanner()
    policy.load()
    frames = [
        conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=i * 100_000)
        for i in range(11)
    ]
    trajectory = policy.infer_sequence(frames)
    assert policy.fed == [0, 500_000, 1_000_000]
    np.testing.assert_allclose(trajectory.xy[:, 0], 4.0 * np.arange(1, 7), atol=0.06)


def test_a_missing_camera_is_an_error_naming_it() -> None:
    """A camera the source calibrated but sent no image for is a lost frame, refused by name.

    One the source has neither an image nor a calibration for is a camera it does not have, and is fed
    black instead (`_absent`; the one-camera tests below).
    """
    policy = StubPlanner()
    policy.load()
    obs = conformance.synthetic_observation(cameras=("CAM_FRONT",))
    obs.camera2ego["CAM_BACK"] = np.eye(4)
    obs.camera2image["CAM_BACK"] = np.eye(3)
    with pytest.raises(ValueError, match="CAM_BACK"):
        policy.infer(obs)


@pytest.mark.parametrize("model", MODELS)
def test_load_without_its_settings_names_them(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    module = importlib.import_module(f"paddock.adapters.models.{model}.policy")
    monkeypatch.delenv(module.POLICY.config_env, raising=False)
    monkeypatch.delenv(module.POLICY.checkpoint_env, raising=False)
    policy = module.POLICY()
    with pytest.raises(RuntimeError, match=module.POLICY.config_env):
        policy.load()
    with pytest.raises(RuntimeError, match="load"):
        policy.infer(conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS))


@pytest.mark.parametrize("model", MODELS)
def test_the_declaration_matches_the_policy(model: str) -> None:
    """model.yaml's cameras, plan grid, rate and settings are the ones the policy reads."""
    import importlib

    spec = Registry.load().models[model]
    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY
    assert tuple(spec.cameras) == policy.cameras
    assert spec.waypoint_dt == bp.PLAN_DT and spec.horizon_s == bp.PLAN_DT * bp.PLAN_STEPS
    assert spec.planning_hz == 1 / bp.PLAN_DT, "the temporal BEV trained on 0.5 s steps"
    assert {policy.config_env, policy.checkpoint_env} <= set(spec.runtime.env)


def test_the_bgr_frames_reach_the_nuscenes_pipeline_as_the_loader_reads_them() -> None:
    """mmcv.imread gives BGR, and the loader casts to float32: an RGB observation is fed reversed."""
    from paddock.adapters.models.uniad.policy import UniADPolicy

    policy = UniADPolicy()
    policy._box_type = policy._box_mode = None  # set by _build in the model's env
    obs = conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS)
    for camera in bp.NUSC_CAMERAS:
        image = np.zeros((900, 1600, 3), dtype=np.uint8)
        image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30  # R, G, B
        obs.images[camera] = image
        obs.camera2image[camera] = np.array([[1260.0, 0, 800.0], [0, 1260.0, 450.0], [0, 0, 1.0]])
    results = policy._results(obs, bp.NUSC_LIDAR2EGO)
    assert results["img"][0].dtype == np.float32
    np.testing.assert_array_equal(results["img"][0][0, 0], [30.0, 20.0, 10.0])
    assert results["img_shape"] == (900, 1600, 3, 6)


# -- the configs the adapters load ---------------------------------------------------------------------

#: (img_norm_cfg mean, std, to_rgb) each config must hold; the adapters feed BGR, as the datasets' loaders do.
NORMALISATION = {
    "uniad": ([103.530, 116.280, 123.675], [1.0, 1.0, 1.0], False),
    "vad": ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True),
}


@pytest.mark.parametrize("model", MODELS)
def test_the_config_normalises_and_reads_the_can_bus_as_documented(model: str) -> None:
    """Read from the site's checkout of the model's repo; skipped where there is none (CI)."""
    from paddock.config import ConfigError, SiteConfig

    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site paddock.toml, so no checkout of the model's repo")
    spec = Registry.load().models[model]
    policy = __import__(f"paddock.adapters.models.{model}.policy", fromlist=["POLICY"]).POLICY
    path = Path(cfg.resolve(spec.runtime.env[policy.config_env], **spec.context(cfg)))
    if not path.is_file():
        pytest.skip(f"{path} is not checked out at this site")
    text = path.read_text()
    norm = re.search(
        r"img_norm_cfg\s*=\s*dict\(\s*mean=\[([^\]]*)\],\s*std=\[([^\]]*)\],\s*to_rgb=(\w+)\)", text
    )
    assert norm, f"no img_norm_cfg in {path}"
    mean, std, to_rgb = NORMALISATION[model]
    np.testing.assert_allclose([float(v) for v in norm.group(1).split(",")], mean)
    np.testing.assert_allclose([float(v) for v in norm.group(2).split(",")], std)
    assert (norm.group(3) == "True") is to_rgb
    assert re.search(r"use_can_bus\s*=\s*True", text), f"{path} does not use the CAN bus"
    assert not re.search(r"use_can_bus\s*=\s*False", text)


def test_yaw_rate_sits_where_both_layouts_read_it() -> None:
    assert CANBUS_YAW_RATE == CANBUS_ACCEL + 5


# -- clients that ask faster than the plan steps (neuroNCAP), and benchmarks that score 4 s (NAVSIM) ------


def _moving(obs: Observation, x: float) -> Observation:
    obs.ego2world = np.eye(4)
    obs.ego2world[0, 3] = x
    return obs


def test_a_client_at_10_hz_gets_a_model_step_every_half_second_and_the_held_plan_between() -> None:
    policy = StubPlanner()
    policy.load()
    plans = {}
    for i in range(13):  # 0.0 .. 1.2 s, the ego driving the plan's own 8 m/s
        t = i * 100_000
        obs = _moving(
            conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=t), 8.0 * t / 1e6
        )
        plans[t] = policy.infer(obs).xy
    assert policy.fed == [0, 500_000, 1_000_000], "the temporal BEV is stepped on the 0.5 s grid only"
    assert policy.last["held"] == 10  # 13 calls, 3 steps
    # Held fixed in the world: 0.2 s later, from 1.6 m further on, the same 8 m/s plan ahead of the car.
    np.testing.assert_allclose(plans[200_000], plans[0], atol=0.01)


def test_a_held_plan_is_resampled_in_time_and_in_the_current_ego_frame() -> None:
    plan = np.c_[4.0 * np.arange(1, 7), np.zeros(6)]  # 8 m/s straight, made at t = 0 from the origin
    # 0.2 s later the ego has not moved: the plan's points from t = 0.7 s on, 1.6 m further each.
    still = bp.held_plan(plan, np.eye(4), 0, np.eye(4), 200_000)
    np.testing.assert_allclose(still[:, 0], 1.6 + 4.0 * np.arange(1, 7), atol=1e-9)
    # The same from a pose turned 90 degrees left in the world: the plan stays where it was in the world.
    turned = np.eye(4)
    turned[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    np.testing.assert_allclose(
        bp.held_plan(plan, np.eye(4), 0, turned, 200_000)[:, 1], -still[:, 0], atol=1e-9
    )


def test_in_scene_order_every_keyframe_is_a_step_however_soon_it_follows_the_last() -> None:
    """nuScenes keyframes come 0.40-0.65 s apart and the repos' test loops step on every one; `infer` would
    answer the 0.40 s one from the held plan, so scene order (`infer_each`) steps every frame instead."""
    stamps = [0, 400_000, 850_000, 1_350_000, 1_950_000, 2_400_000]
    frames = [conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=t) for t in stamps]
    policy = StubPlanner()
    policy.load()
    plans = policy.infer_each(frames)
    assert policy.fed == stamps and len(plans) == len(stamps)
    assert "held" not in policy.last and policy.self_prime is True
    np.testing.assert_allclose(plans[1].xy[:, 0], 4.0 * np.arange(1, 7), atol=0.06)
    policy.reset()
    for frame in frames:
        policy.infer(frame)
    assert policy.fed == [0, 850_000, 1_350_000, 1_950_000], (
        "frame by frame, the 0.40 and 0.45 s ones are held"
    )


def test_clients_that_honour_planning_hz_are_never_held() -> None:
    policy = StubPlanner()
    policy.load()
    for i in range(5):
        policy.infer(conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=i * 500_000))
    assert policy.fed == [i * 500_000 for i in range(5)]
    assert "held" not in policy.last


def test_the_horizon_is_the_models_own_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    class Extended(StubPlanner):
        horizon_env = "STUB_HORIZON_S"

    obs = conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS)
    monkeypatch.delenv("STUB_HORIZON_S", raising=False)
    policy = Extended()
    policy.load()
    assert len(policy.infer(obs).xy) == bp.PLAN_STEPS
    monkeypatch.setenv("STUB_HORIZON_S", "4.0")
    policy = Extended()
    policy.load()
    xy = policy.infer(obs).xy
    assert len(xy) == 8, "NAVSIM's eight poses"
    np.testing.assert_allclose(np.diff(xy[5:, 0]), 4.0, atol=0.06), "the last step held"
    monkeypatch.setenv("STUB_HORIZON_S", "2.0")
    with pytest.raises(ValueError, match="shorter"):
        Extended()


@pytest.mark.parametrize("model", MODELS)
def test_the_horizon_setting_is_off_by_default_and_named_in_model_yaml(model: str) -> None:
    import importlib

    spec = Registry.load().models[model]
    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY
    assert policy.horizon_env not in spec.runtime.env, "unset by default: the plan is the model's own"
    assert policy.horizon_env in (spec.directory / "model.yaml").read_text(), "a site must find its name"


# -- preparing the next frame while the model runs ----------------------------------------------------


class SlowStub(StubPlanner):
    """Records what each step was fed -- frame index, quaternion branch, which frame -- from `_results`,
    which runs on the preparing thread, and takes a little while to run, so preparing overlaps it."""

    def _results(self, obs, lidar2ego):
        q = self._nuscenes_quaternion(obs)
        self.order.append(int(obs.timestamp_us))
        return {"t": obs.timestamp_us, "frame": self._prepared, "w": float(q[0])}

    def _run(self, prepared, results):
        import time

        time.sleep(0.01)
        self.fed.append((results["t"], results["frame"], results["w"]))
        return np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]

    def _reset_model(self) -> None:
        self.fed, self.order = [], []


@pytest.mark.parametrize("threads", ["1", "2", "5"])
def test_preparing_ahead_feeds_every_frame_what_it_is_fed_one_at_a_time(
    threads: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next frame is prepared on another thread while the model runs, strictly in frame order: the
    frame index each step carries and the quaternion's sign branch (held from frame to frame) come out the
    same at any thread count. The poses turn through 180 degrees so the branch has somewhere to go."""
    monkeypatch.setenv(bp.THREADS_ENV, threads)
    policy = SlowStub()
    policy.load()
    assert policy.threads == int(threads)
    frames = []
    for i in range(9):
        obs = conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=i * 500_000)
        yaw = math.radians(150 + 10 * i)
        obs.ego2world[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        frames.append(obs)
    policy.infer_sequence(frames)
    stamps = [i * 500_000 for i in range(9)]
    assert policy.order == stamps, "prepared in frame order, each once"
    assert [t for t, _, _ in policy.fed] == stamps
    assert [frame for _, frame, _ in policy.fed] == list(range(9)), "the frame index is the step's own"
    q = None
    expected = []
    for obs in frames:
        q = bp.continuous_quaternion(obs.ego2world[:3, :3], q)
        expected.append(float(q[0]))
    assert [w for _, _, w in policy.fed] == expected
    assert policy.last["steps"] == 9 and policy.diagnostics()["frames_fed"] == 9


def test_a_step_that_fails_leaves_nothing_being_prepared_behind_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `reset` after a failed request must not meet the next frame still being prepared."""

    class Failing(SlowStub):
        def _run(self, prepared, results):
            if results["t"] == 1_000_000:
                raise RuntimeError("the model failed on this frame")
            return super()._run(prepared, results)

    monkeypatch.setenv(bp.THREADS_ENV, "3")
    policy = Failing()
    policy.load()
    frames = [
        conformance.synthetic_observation(cameras=bp.NUSC_CAMERAS, timestamp_us=i * 500_000) for i in range(5)
    ]
    with pytest.raises(RuntimeError, match="failed on this frame"):
        policy.infer_sequence(frames)
    assert policy.order == [0, 500_000, 1_000_000, 1_500_000], "the frame after the failure was finished"
    policy.reset()
    assert policy.order == [] and policy._prepared == 0


# -- a source with one forward camera (comma2k19) --------------------------------------------------------


def _front_only() -> Observation:
    """A comma2k19-shaped observation: CAM_FRONT with an image and a calibration, nothing for the others."""
    obs = conformance.synthetic_observation(cameras=("CAM_FRONT",))
    obs.images["CAM_FRONT"] = np.full((874, 1164, 3), 77, dtype=np.uint8)
    obs.camera2image["CAM_FRONT"] = np.array([[910.0, 0, 582.0], [0, 910.0, 437.0], [0, 0, 1.0]])
    return obs


def test_the_nuscenes_rig_is_the_golden_fixtures_calibration() -> None:
    """NUSC_RIG, which a black slot keeps, is scene-0103's calibration as the fixtures recorded it."""
    observed = _fixture("uniad")["observations"][0]
    assert set(bp.NUSC_RIG) == set(bp.NUSC_CAMERAS)
    for camera, (intrinsics, pose) in bp.NUSC_RIG.items():
        np.testing.assert_array_equal(intrinsics, observed["camera2image"][camera])
        np.testing.assert_array_equal(pose, observed["camera2ego"][camera])


def test_a_camera_the_source_does_not_have_is_absent_and_one_it_lost_is_not() -> None:
    obs = _front_only()
    assert bp.BevPlannerPolicy._absent(obs, "CAM_BACK")
    assert not bp.BevPlannerPolicy._absent(obs, "CAM_FRONT")
    # A calibration without its image is a lost frame, not a missing camera.
    obs.camera2ego["CAM_BACK"] = np.eye(4)
    assert not bp.BevPlannerPolicy._absent(obs, "CAM_BACK")
    # A source that does not describe its forward camera cannot say any camera is absent.
    bare = _front_only()
    del bare.camera2image["CAM_FRONT"]
    assert not bp.BevPlannerPolicy._absent(bare, "CAM_BACK")


@pytest.mark.parametrize("model", ["uniad", "vad"])
def test_a_one_camera_source_is_fed_five_black_slots_at_nuscenes_calibration(monkeypatch, model: str) -> None:
    """The road frame stays CAM_FRONT; the other five reach `_results` black, at NUSC_RIG, and are listed."""
    import importlib

    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY()
    seen = {}
    monkeypatch.setattr(policy, "_results", lambda obs, lidar2ego: seen.setdefault("obs", obs) and {})
    monkeypatch.setattr(policy, "_prepare", lambda results: None)
    obs = _front_only()
    policy._prepare_frame(obs)
    fed = seen["obs"]
    blank = [c for c in bp.NUSC_CAMERAS if c != "CAM_FRONT"]
    assert policy.diagnostics()["blank_views"] == blank
    assert fed.images["CAM_FRONT"] is obs.images["CAM_FRONT"]
    for camera in blank:
        assert fed.images[camera].shape == (900, 1600, 3) and not fed.images[camera].any()
        np.testing.assert_array_equal(fed.camera2image[camera], bp.NUSC_RIG[camera][0])
        np.testing.assert_array_equal(fed.camera2ego[camera], bp.NUSC_RIG[camera][1])
    assert set(obs.images) == {"CAM_FRONT"}, "the source's own observation is not written to"
    fed.images["CAM_BACK"][0, 0] = 255
    assert not fed.images["CAM_BACK_LEFT"].any(), "each slot has its own zeros"


def test_a_camera_lost_from_a_described_rig_is_still_refused(monkeypatch) -> None:
    from paddock.adapters.models.uniad.policy import POLICY

    policy = POLICY()
    monkeypatch.setattr(policy, "_prepare", lambda results: None)
    obs = _front_only()
    obs.camera2ego["CAM_BACK"] = np.eye(4)
    obs.camera2image["CAM_BACK"] = np.eye(3)
    with pytest.raises(ValueError, match="reads all six surround cameras"):
        policy._prepare_frame(obs)


# -- the black views' backbone features, computed once (BlankViewFeatures) -----------------------------


def _views_backbone(calls: list[int]):
    """A stand-in backbone that maps each view on its own, as ResNet does in eval: two feature levels."""

    def backbone(x):
        calls.append(len(x))
        flat = x.reshape(len(x), -1)
        return (flat.sum(axis=1, keepdims=True) * 2.0, flat.max(axis=1, keepdims=True) - 1.0)

    return backbone


def _numpy_ops():
    from types import SimpleNamespace

    return SimpleNamespace(equal=np.array_equal, cat=np.concatenate, clone=np.copy)


def test_black_views_are_answered_from_one_run_and_the_output_is_the_full_runs() -> None:
    from paddock.adapters.models.uniad.bev_planners import BlankViewFeatures

    rng = np.random.default_rng(0)
    black = np.full((3, 4, 5), -1.5)
    calls: list[int] = []
    wrapped = BlankViewFeatures(_views_backbone(calls), _numpy_ops())
    reference = _views_backbone([])
    for _ in range(3):
        x = np.stack([rng.normal(size=(3, 4, 5))] + [black] * 5)
        wrapped.blank = [1, 2, 3, 4, 5]
        out = wrapped(x)
        expected = reference(x)
        assert isinstance(out, tuple) and len(out) == 2
        for got, want in zip(out, expected):  # noqa: B905 - adapters run on Python 3.9
            np.testing.assert_array_equal(got, want)
    # the first step runs one black view and the real one; later steps the real view alone
    assert calls == [1, 1, 1, 1] and wrapped.reused == 15


def test_a_slot_marked_black_whose_input_differs_runs_the_backbone_on_every_view() -> None:
    from paddock.adapters.models.uniad.bev_planners import BlankViewFeatures

    black = np.zeros((3, 4, 5))
    calls: list[int] = []
    wrapped = BlankViewFeatures(_views_backbone(calls), _numpy_ops())
    wrapped.blank = [1, 2]
    wrapped(np.stack([np.ones((3, 4, 5)), black, black]))
    not_black = np.stack([np.ones((3, 4, 5)), black, black + 0.25])
    calls.clear()
    out = wrapped(not_black)
    for got, want in zip(out, _views_backbone([])(not_black)):  # noqa: B905
        np.testing.assert_array_equal(got, want)
    assert calls == [3]


def test_with_no_black_slot_the_backbone_runs_as_it_always_did() -> None:
    from paddock.adapters.models.uniad.bev_planners import BlankViewFeatures

    calls: list[int] = []
    wrapped = BlankViewFeatures(_views_backbone(calls), _numpy_ops())
    x = np.random.default_rng(1).normal(size=(6, 3, 4, 5))
    for got, want in zip(wrapped(x), _views_backbone([])(x)):  # noqa: B905
        np.testing.assert_array_equal(got, want)
    assert calls == [6] and wrapped.reused == 0


@pytest.mark.parametrize(("model", "installed"), [("uniad", True), ("vad", False)])
def test_only_uniad_runs_its_backbone_once_for_the_black_views(model: str, installed: bool) -> None:
    """UniAD's backbone is most of a step (1.35x with the wrapper); VAD's is not (1.02x), and the batch of
    one moves VAD's plans by up to 2.2 mm, so VAD's backbone is left as it is."""
    import importlib
    from types import SimpleNamespace

    policy = importlib.import_module(f"paddock.adapters.models.{model}.policy").POLICY()
    original = _views_backbone([])
    model_stub = SimpleNamespace(img_backbone=SimpleNamespace(forward=original))
    policy._install_view_features(model_stub, _numpy_ops())
    assert (policy.view_features is not None) is installed
    assert (model_stub.img_backbone.forward is not original) is installed


def test_the_black_slots_reach_the_backbone_wrapper_in_camera_order(monkeypatch) -> None:
    """A one-camera source's five black slots are named by their index in the model's camera order,
    the order the views are batched in, and handed to the wrapper before the step runs."""
    from types import SimpleNamespace

    from paddock.adapters.models.uniad.policy import POLICY

    policy = POLICY()
    monkeypatch.setattr(policy, "_results", lambda obs, lidar2ego: {})
    monkeypatch.setattr(policy, "_prepare", lambda results: None)
    prepared = policy._prepare_frame(_front_only())
    slots = prepared[2]["blank_slots"]
    assert slots == [1, 2, 3, 4, 5]
    assert [bp.NUSC_CAMERAS[i] for i in slots] == policy.diagnostics()["blank_views"]

    seen = {}
    policy.view_features = SimpleNamespace(blank=[], reused=0)

    def step(batch, results):
        seen["blank"] = list(policy.view_features.blank)
        policy.view_features.reused += len(policy.view_features.blank)
        return np.zeros((6, 2))

    monkeypatch.setattr(policy, "_run", step)
    policy._run_prepared(_front_only(), prepared)
    assert seen["blank"] == [1, 2, 3, 4, 5]
    # what the wrapper answered from its cache that step, in the diagnostics a run can keep
    assert policy.diagnostics()["views_from_cache"] == 5
