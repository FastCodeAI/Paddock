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
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, CANBUS_YAW_RATE, Command, Observation
from paddock.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = ("uniad", "vad", "uniad_b2d", "vad_b2d")


def _fixture(model: str) -> dict:
    return json.loads((FIXTURES / f"golden_{model}.json").read_text())


def _observations(model: str, with_quaternion: bool) -> list[Observation]:
    command = Command.FOLLOW_REFERENCE if model.endswith("_b2d") else Command.STRAIGHT
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


@pytest.mark.parametrize("model", ["uniad_b2d", "vad_b2d"])
def test_bench2drive_inputs_equal_bench2drivezoos(model: str) -> None:
    """lidar2img, can_bus, l2g and command as B2D_E2E_Dataset's maths and the agents' pipeline gave them."""
    seen = _fixture(model)["variants"]["dataset_colour"]["frames"]
    for obs, want in zip(_observations(model, False), seen, strict=True):
        # The agents' inference_only_pipeline: VAD scales by 0.8 after its first pad; UniAD does not scale.
        scale = np.diag([0.8, 0.8, 1.0, 1.0]) if model == "vad_b2d" else np.eye(4)
        projections = [
            scale @ bp.lidar2img(obs.camera2image[c], obs.camera2ego[c], bp.B2D_LIDAR2EGO)
            for c in bp.B2D_CAMERAS
        ]
        np.testing.assert_allclose(projections, want["lidar2img"], rtol=1e-9, atol=1e-6)
        can_bus, _ = bp.b2d_can_bus(obs.ego2world, obs.canbus)
        np.testing.assert_allclose(can_bus, want["can_bus"], rtol=1e-12, atol=1e-9)
        l2g_r, l2g_t = bp.b2d_l2g(obs.ego2world)
        np.testing.assert_allclose(l2g_r, want["l2g_r_mat"], atol=1e-12)
        np.testing.assert_allclose(l2g_t, want["l2g_t"], atol=1e-9)
        assert bp.b2d_command(obs) == want["command"] == bp.B2D_LANEFOLLOW


def test_the_nuscenes_can_bus_holds_w_four_times() -> None:
    """The datasets assign a pyquaternion Quaternion into a numpy slice, which broadcasts its w."""
    pose = np.eye(4)
    pose[:3, :3] = bp.quaternion_matrix([0.9, 0.0, 0.0, math.sqrt(1 - 0.81)])
    can_bus, _ = bp.nuscenes_can_bus(pose, np.zeros(18), bp.matrix_quaternion(pose[:3, :3]))
    np.testing.assert_allclose(can_bus[3:7], [0.9] * 4)


def test_bench2drive_can_bus_layout_and_heading() -> None:
    """Velocity at 7, acceleration at 10:13, rotation rate at 13:16, the heading wrapped at 16."""
    yaw = 2.5  # (pi/2, pi]: Bench2Drive's ego_yaw is yaw - 2pi there, and wrapped back at 16
    pose = np.eye(4)
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:2, 3] = [12.0, -3.0]
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = 7.0
    canbus[CANBUS_ACCEL : CANBUS_ACCEL + 3] = [0.5, -0.2, 0.0]
    canbus[CANBUS_ACCEL + 3 : CANBUS_VELOCITY] = [0.01, 0.02, 0.3]
    out, gravity = bp.b2d_can_bus(pose, canbus)
    assert gravity
    np.testing.assert_allclose(out[0:3], [12.0, -3.0, 0.0])
    unwrapped = yaw - 2 * math.pi
    np.testing.assert_allclose(out[3:7], [math.cos(unwrapped / 2), 0, 0, math.sin(unwrapped / 2)])
    np.testing.assert_allclose(out[7:10], [7.0, 0.0, 0.0])
    np.testing.assert_allclose(out[10:13], [0.5, -0.2, bp.GRAVITY])
    np.testing.assert_allclose(out[13:16], [0.01, -0.02, 0.3])
    np.testing.assert_allclose(out[16:18], [yaw, math.degrees(yaw)])


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
    assert bp.has_lidar_top_axes(bp.NUSC_LIDAR2EGO) and bp.has_lidar_top_axes(bp.B2D_LIDAR2EGO)
    assert not bp.has_lidar_top_axes(np.eye(4))


def _carla_camera2ego(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """paddock's Bench2Drive agent, ``_cam2ego``: CARLA's Transform.get_matrix (roll = pitch = 0), y-flipped
    to right-handed, then body -> optical (agent.py, ``_rh`` and ``R_OPT``)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    carla = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.0]])
    flip = np.diag([1.0, -1.0, 1.0, 1.0])
    out = flip @ carla @ flip
    out[:3, :3] = out[:3, :3] @ np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return out


#: Bench2DriveZoo's agents hard-code these (``self.lidar2img`` in team_code/uniad_b2d_agent.py and
#: vad_b2d_agent.py at 498c1f7) for Bench2Drive's own rig.
B2D_AGENT_LIDAR2IMG = {
    "CAM_FRONT": [
        [1142.51841, 800.0, 0.0, -952.0],
        [0.0, 450.0, -1142.51841, -809.704417],
        [0.0, 1.0, 0.0, -1.19],
    ],
    "CAM_FRONT_LEFT": [
        [6.03961325e-14, 1394.75744, 0.0, -920.539908],
        [-368.61842, 258.109396, -1142.51841, -647.29675],
        [-0.819152044, 0.573576436, 0.0, -0.829094072],
    ],
    "CAM_BACK": [
        [-560.166031, -800.0, 0.0, -1288.0],
        [0.0, -450.0, -560.166031, -858.939847],
        [0.0, -1.0, 0.0, -1.61],
    ],
    "CAM_BACK_RIGHT": [
        [360.989788, -1347.23223, 0.0, -104.238127],
        [422.861679, -153.909064, -1142.51841, -496.004706],
        [0.939692621, -0.342020143, 0.0, -0.492889531],
    ],
}


@pytest.mark.parametrize("camera", sorted(B2D_AGENT_LIDAR2IMG))
def test_bench2drive_rig_projects_as_bench2drivezoos_agents_do(camera: str) -> None:
    """On paddock's ``bench2drive`` rig, lidar2img from the calibration paddock sends equals the agents'."""
    from paddock.adapters.benchmarks.bench2drive import rig

    cam = rig.cameras("bench2drive")[camera]
    x, y, z, _roll, _pitch, yaw = cam["pose"]
    width, height = cam["size"]
    f = width / (2.0 * math.tan(math.radians(cam["fov"]) / 2.0))
    k = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    projection = bp.lidar2img(k, _carla_camera2ego(x, y, z, yaw), bp.B2D_LIDAR2EGO)
    np.testing.assert_allclose(projection[:3], B2D_AGENT_LIDAR2IMG[camera], rtol=1e-6, atol=1e-5)


def test_intrinsics_scale_with_the_resize() -> None:
    k = np.array([[630.0, 0, 400.0], [0, 630.0, 225.0], [0, 0, 1.0]])
    np.testing.assert_allclose(bp.scaled_intrinsics(k, 2.0, 2.0), [[1260, 0, 800], [0, 1260, 450], [0, 0, 1]])
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    same, same_k = bp.to_load_size(image, k)  # already the load size: no cv2 needed, nothing changes
    assert same is image
    np.testing.assert_array_equal(same_k, k)


# -- the plan ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lidar2ego", [bp.B2D_LIDAR2EGO, bp.NUSC_LIDAR2EGO], ids=["b2d", "nuscenes"])
def test_a_straight_plan_changes_only_axes(lidar2ego: np.ndarray) -> None:
    """On a straight road the lever arm cancels: lidar (x right, y forward) -> ego (x forward, y left)."""
    plan = np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]  # 8 m/s along the lidar's own line
    ego = bp.lidar_plan_to_ego(plan, lidar2ego)
    # Only the lidar's tilt (~2 degrees on nuScenes) moves anything; Bench2Drive's is exact.
    tolerance = 1e-12 if lidar2ego is bp.B2D_LIDAR2EGO else 0.06
    np.testing.assert_allclose(ego, np.c_[4.0 * np.arange(1, 7), np.zeros(6)], atol=tolerance)


def test_right_in_the_lidar_frame_is_negative_y() -> None:
    """lidar x is right (nuScenes LIDAR_TOP is yawed -90 degrees), paddock's y is left."""
    plan = np.c_[0.1 * np.arange(1, 7), 4.0 * np.arange(1, 7)]
    assert (bp.lidar_plan_to_ego(plan, bp.B2D_LIDAR2EGO)[:, 1] < 0).all()


@pytest.mark.parametrize("turn", [-1.0, 1.0], ids=["right", "left"])
def test_through_a_turn_the_ego_origin_is_recovered(turn: float) -> None:
    """A rear axle on a 25 m arc: its lidar's positions, as the model would plan them, give the arc back."""
    radius, speed, arm = 25.0, 8.0, bp.B2D_LIDAR2EGO[:2, 3]
    times = 0.5 * np.arange(1, 7)
    heading = turn * speed * times / radius
    rear = np.c_[radius * np.sin(np.abs(heading)), turn * radius * (1 - np.cos(heading))]
    rotations = np.stack([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]).transpose(
        2, 0, 1
    )
    lidar_in_ego = rear + rotations @ arm
    lidar_frame = (np.c_[lidar_in_ego, np.zeros(6)] - bp.B2D_LIDAR2EGO[:3, 3]) @ bp.B2D_LIDAR2EGO[:3, :3]
    recovered = bp.lidar_plan_to_ego(lidar_frame[:, :2], bp.B2D_LIDAR2EGO)
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


def test_bench2drive_commands_are_carlas_road_options_minus_one() -> None:
    """On Bench2Drive's own client, which sends a route: CARLA's road option minus one."""
    expected = {Command.RIGHT: 1, Command.LEFT: 0, Command.STRAIGHT: 2, Command.FOLLOW_REFERENCE: 3}
    assert {c: bp.b2d_command(_with_command(c)) for c in Command} == expected


def test_without_a_route_straight_is_bench2drives_lanefollow() -> None:
    """HUGSIM / nuScenes STRAIGHT means "not turning"; Bench2Drive's STRAIGHT is junction-only."""
    expected = {Command.RIGHT: 1, Command.LEFT: 0, Command.STRAIGHT: 3, Command.FOLLOW_REFERENCE: 3}
    assert {c: bp.b2d_command(_with_command(c, route=False)) for c in Command} == expected
    # A lane change on the route within the agent's 4 m lookahead is restored as one.
    assert bp.b2d_command(_with_command(Command.STRAIGHT, "LEFT", 0.0)) == bp.B2D_CHANGELANELEFT
    assert bp.b2d_command(_with_command(Command.FOLLOW_REFERENCE, "RIGHT", 3.5)) == bp.B2D_CHANGELANERIGHT
    assert bp.b2d_command(_with_command(Command.FOLLOW_REFERENCE, "RIGHT", 12.0)) == bp.B2D_LANEFOLLOW
    assert bp.b2d_command(_with_command(Command.LEFT, "RIGHT", 1.0)) == bp.B2D_LEFT, "a turn is kept"


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
    policy = StubPlanner()
    policy.load()
    obs = conformance.synthetic_observation(cameras=("CAM_FRONT",))
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
    "uniad_b2d": ([103.530, 116.280, 123.675], [1.0, 1.0, 1.0], False),
    "vad": ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True),
    "vad_b2d": ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], True),
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
