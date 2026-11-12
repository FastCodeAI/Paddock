"""policy.py with the model stubbed out: the contract, the sequence it builds, the reads it makes.

These run in paddock's own environment, which has neither torch nor the vendor
package; the stub replaces the one method that needs them. What the vendor makes
of the same sequence is tests/test_vendor_inputs.py, in the model's environment.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from paddock import conformance
from paddock.adapters.models.alpamayo2edge import inputs
from paddock.adapters.models.alpamayo2edge.policy import (
    MAX_DRAWS,
    NOMINAL,
    POLICY,
    Alpamayo2EdgePolicy,
    Settings,
    draw_until_complete,
)
from paddock.policy import protocol
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation

#: Where NAVSIM's cameras point, degrees left of forward (navtest log pickles).
BEARINGS = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 55,
    "CAM_FRONT_RIGHT": -55,
    "CAM_BACK_LEFT": 141,
    "CAM_BACK_RIGHT": -141,
}
#: NAVSIM's 1920x1080 intrinsics at a tenth of the size, so the tests stay fast.
WIDTH, HEIGHT, FX, CX, CY = 192, 108, 154.5, 96.0, 56.0
SPEED, YAW_RATE = 6.0, 0.1  # m/s, rad/s: a gentle left curve


class _Stub(Alpamayo2EdgePolicy):
    """The real policy with the model replaced: straight ahead at 8 m/s, draw i shifted i x 0.1 m left."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings or Settings())
        self.runs: list = []

    def load(self) -> None:
        self.model = "stub"

    def _run_model(self, sample):
        self.runs.append(sample)
        draws = self.settings.samples
        xyz = np.zeros((draws, inputs.PLAN_STEPS, 3))
        xyz[:, :, 0] = 8.0 * inputs.PLAN_DT_S * np.arange(1, inputs.PLAN_STEPS + 1)
        xyz[:, :, 1] = 0.1 * np.arange(draws)[:, None]
        rot = np.broadcast_to(np.eye(3), (draws, inputs.PLAN_STEPS, 3, 3)).copy()
        return xyz, rot, [NOMINAL] * draws


def _camera2ego(bearing_deg: float) -> np.ndarray:
    yaw = math.radians(bearing_deg)
    turn = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = turn @ conformance.UNROTATED_CAMERA2EGO
    matrix[2, 3] = 1.5
    return matrix


def _pose(t: float) -> np.ndarray:
    """The car on a circle, at time t, in the frame of the car at t = 0 -- NAVSIM's convention."""
    radius = SPEED / YAW_RATE
    pose = np.eye(4)
    yaw = YAW_RATE * t
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:2, 3] = [radius * math.sin(yaw), radius * (1 - math.cos(yaw))]
    return pose


def _sequence(
    tmp_path: Path,
    times_s=(-1.5, -1.0, -0.5, 0.0),
    cameras=inputs.CAMERAS,
    command: Command = Command.LEFT,
) -> list[Observation]:
    """NAVSIM's 2 Hz path: four frames, each camera an image file, poses relative to now."""
    rng = np.random.default_rng(0)
    intrinsics = np.array([[FX, 0.0, CX], [0.0, FX, CY], [0.0, 0.0, 1.0]])
    sequence = []
    for t in times_s:
        paths = {}
        for camera in cameras:
            path = tmp_path / f"{camera}_{t:+.1f}.png"
            if not path.exists():
                Image.fromarray(rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)).save(path)
            paths[camera] = str(path)
        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = SPEED
        sequence.append(
            Observation(
                images={},
                image_paths=paths,
                ego2world=_pose(t),
                canbus=canbus,
                timestamp_us=int(round(t * 1e6)),
                command=command,
                camera2image={camera: intrinsics for camera in cameras},
                camera2ego={camera: _camera2ego(BEARINGS[camera]) for camera in cameras},
            )
        )
    return sequence


def _loaded(settings: Settings | None = None) -> _Stub:
    policy = _Stub(settings)
    policy.load()
    return policy


# -- the contract ---------------------------------------------------------------------


def test_the_policy_meets_the_contract_with_its_model_stubbed() -> None:
    findings = conformance.policy_findings(_Stub(), stochastic=True)
    failed = [f"{f.check}: {f.detail}" for f in findings if f.failed]
    assert not failed, failed


def test_planning_before_load_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="load"):
        _Stub().infer(_sequence(tmp_path)[-1])


def test_the_wire_keeps_every_camera_edge_reads(tmp_path: Path) -> None:
    """`from_wire` drops cameras the policy does not declare, so a slot camera
    missing from `cameras` would arrive as a missing image."""
    obs = _sequence(tmp_path)[-1]
    again = Observation.from_wire(obs.to_wire(), POLICY.cameras)
    assert set(again.image_paths) == set(inputs.CAMERAS)


# -- the sequence it builds -----------------------------------------------------------


def test_a_navsim_sequence_becomes_edges_input(tmp_path: Path) -> None:
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    sample = policy.runs[0]

    assert sample.frames.shape == (22, 1080, 1920, 3) and sample.frames.dtype == np.uint8
    assert sample.camera_ids.tolist() == [0] * 3 + [1] * 5 + [2] * 3 + [3] * 3 + [5] * 3 + [6] * 5
    assert sample.frame_times_s[1] == sample.frame_times_s[6] == [-1.5, -1.0, -0.5, -0.5, 0.0]
    assert sample.frame_times_s[0] == sample.frame_times_s[3] == [-1.0, -0.5, 0.0]
    assert sample.timestamps_us[3:8].tolist() == [-1_500_000, -1_000_000, -500_000, -500_000, 0]
    assert sample.nav_text == "Turn left"
    assert sample.extrapolated_s == pytest.approx(0.4)


def test_the_history_follows_the_curve_the_car_drove(tmp_path: Path) -> None:
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    sample = policy.runs[0]
    np.testing.assert_allclose(sample.history_xyz[-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(sample.history_rot[-1], np.eye(3), atol=1e-12)
    steps = np.linalg.norm(np.diff(sample.history_xyz, axis=0), axis=1)
    np.testing.assert_allclose(steps, SPEED * inputs.HISTORY_DT_S, rtol=1e-3)
    yaw = np.arctan2(sample.history_rot[:, 1, 0], sample.history_rot[:, 0, 0])
    np.testing.assert_allclose(yaw, YAW_RATE * -0.1 * np.arange(19, -1, -1), atol=1e-9)


def test_the_telephoto_slot_is_a_crop_of_the_front_camera(tmp_path: Path) -> None:
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    frames, ids = policy.runs[0].frames, policy.runs[0].camera_ids
    front, tele = frames[ids == 1][-1], frames[ids == 6][-1]
    assert not np.array_equal(front, tele)
    geometry = policy.diagnostics()["slot_geometry"]
    assert geometry[6]["hfov_deg"] == pytest.approx(30.0, abs=0.5)
    assert geometry[0]["bearing_deg"] == pytest.approx(55.0) and geometry[5]["bearing_deg"] == pytest.approx(
        -141.0
    )


def test_a_camera_the_source_never_sent_is_named(tmp_path: Path) -> None:
    cameras = tuple(c for c in inputs.CAMERAS if c != "CAM_BACK_LEFT")
    with pytest.raises(ValueError, match="CAM_BACK_LEFT"):
        _loaded().infer_sequence(_sequence(tmp_path, cameras=cameras))


def test_absent_cameras_black_feeds_their_slots_black_at_the_times_they_ask_for(tmp_path: Path) -> None:
    """comma2k19's one camera: the front and telephoto slots are real, the other four black,
    each black frame stamped like a real one would be, and the four named."""
    policy = _loaded(Settings(absent_cameras="black"))
    policy.infer_sequence(_sequence(tmp_path, cameras=("CAM_FRONT",)))
    sample = policy.runs[-1]
    assert sorted(sample.black_slots) == [0, 2, 3, 5]
    assert len(sample.frames) == len(sample.camera_ids) == len(sample.timestamps_us) == 22
    for camera_id, frame in zip(sample.camera_ids, sample.frames, strict=True):
        assert bool(frame.max() == 0) is (int(camera_id) in sample.black_slots)
    # A black slot's frames are taken at the source's own frame times, as a real one's are.
    assert sample.frame_times_s[0] == sample.frame_times_s[3] == sample.frame_times_s[5]
    assert policy.diagnostics()["last"]["black_slots"] == sample.black_slots


def test_with_every_camera_present_black_changes_nothing(tmp_path: Path) -> None:
    refuse, black = _loaded(), _loaded(Settings(absent_cameras="black"))
    refuse.infer_sequence(_sequence(tmp_path))
    black.infer_sequence(_sequence(tmp_path))
    assert black.runs[-1].black_slots == []
    np.testing.assert_array_equal(black.runs[-1].frames, refuse.runs[-1].frames)


def test_a_straight_command_is_continue_straight(tmp_path: Path) -> None:
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path, command=Command.STRAIGHT))
    assert policy.runs[0].nav_text == "Continue straight"


def test_a_nominal_distance_is_added_only_when_set(tmp_path: Path) -> None:
    policy = _loaded(Settings(nav_distance_m=20.0))
    policy.infer_sequence(_sequence(tmp_path))
    assert policy.runs[0].nav_text == "Turn left in 20m"


# -- what it reads and how often it runs the model ------------------------------------


def test_one_sequence_is_one_model_call(tmp_path: Path) -> None:
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    assert len(policy.runs) == 1


def test_each_file_is_read_once_and_the_next_scenario_reads_only_its_new_frame(
    tmp_path: Path, monkeypatch
) -> None:
    """At 2 Hz the front camera uses all four frames and the other four cameras the
    last three: 16 reads. NAVSIM's next scenario in a log is 0.5 s later and shares
    three frames with this one, so it reads one new frame per camera: 5."""
    reads: list[str] = []
    original = protocol.load_frame
    monkeypatch.setattr(protocol, "load_frame", lambda path: reads.append(str(path)) or original(path))

    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    assert len(reads) == 16 and len(set(reads)) == 16

    reads.clear()
    later = _sequence(tmp_path, times_s=(-1.0, -0.5, 0.0, 0.5))
    policy.infer_sequence(later)
    assert sorted(Path(p).name for p in reads) == sorted(f"{c}_+0.5.png" for c in inputs.CAMERAS)


# -- the plan -------------------------------------------------------------------------


def test_the_plan_is_all_64_poses_a_tenth_of_a_second_apart(tmp_path: Path) -> None:
    trajectory = _loaded().infer_sequence(_sequence(tmp_path))
    assert trajectory.waypoints.shape == (64, 3) and trajectory.dt == pytest.approx(0.1)
    assert trajectory.extras is None


def test_several_draws_return_their_medoid(tmp_path: Path) -> None:
    trajectory = _loaded(Settings(samples=3)).infer_sequence(_sequence(tmp_path))
    np.testing.assert_allclose(trajectory.waypoints[:, 1], 0.1)


def test_the_rationale_is_kept_only_when_asked(tmp_path: Path) -> None:
    trajectory = _loaded(Settings(save_reasoning=True)).infer_sequence(_sequence(tmp_path))
    assert trajectory.extras == {"reasoning": "<NOMINAL>"}


def test_a_plan_spacing_other_than_the_models_is_refused(tmp_path: Path) -> None:
    obs = _sequence(tmp_path)[-1]
    obs.plan_dt = 0.5
    with pytest.raises(ValueError, match="plan_dt"):
        _loaded().infer(obs)


def test_diagnostics_answer_before_load_and_report_the_last_plan(tmp_path: Path) -> None:
    assert _Stub().diagnostics()["last"] is None
    policy = _loaded()
    policy.infer_sequence(_sequence(tmp_path))
    diagnostics = policy.diagnostics()
    assert diagnostics["plans"] == 1 and diagnostics["nominal_rationales"] == 1
    assert diagnostics["last"]["nav_text"] == "Turn left"
    assert diagnostics["last"]["history_extrapolated_s"] == pytest.approx(0.4)


# -- settings -------------------------------------------------------------------------


def test_settings_default_to_the_vendors_and_read_the_environment() -> None:
    default = Settings("nearest", "hermite", 1, 43, None, False, "refuse", 4)
    assert Settings.from_env({}) == Settings() == default
    env = {
        "ALPAMAYO2_EDGE_FRAMES": "distinct",
        "ALPAMAYO2_EDGE_HISTORY": "linear",
        "ALPAMAYO2_EDGE_SAMPLES": "6",
        "ALPAMAYO2_EDGE_SEED": "7",
        "ALPAMAYO2_EDGE_NAV_DISTANCE_M": "25",
        "ALPAMAYO2_EDGE_SAVE_REASONING": "1",
        "ALPAMAYO2_EDGE_ABSENT_CAMERAS": "black",
        "ALPAMAYO2_EDGE_THREADS": "16",
    }
    assert Settings.from_env(env) == Settings("distinct", "linear", 6, 7, 25.0, True, "black", 16)


@pytest.mark.parametrize(
    "name,value",
    [
        ("FRAMES", "grid"),
        ("HISTORY", "clamp"),
        ("SAMPLES", "0"),
        ("NAV_DISTANCE_M", "-5"),
        ("SEED", "x"),
        ("ABSENT_CAMERAS", "grey"),
        ("THREADS", "0"),
    ],
)
def test_a_bad_setting_stops_the_node(name: str, value: str) -> None:
    with pytest.raises(ValueError):
        Settings.from_env({f"ALPAMAYO2_EDGE_{name}": value})


# -- closed loop: Bench2Drive's client ---------------------------------------------------

#: Where paddock's Bench2Drive cameras point, degrees left of forward (bench2drive/rig.py);
#: each is 70 deg wide. Rendered at a tenth of their 800x450.
B2D_BEARINGS = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 55,
    "CAM_FRONT_RIGHT": -55,
    "CAM_BACK_LEFT": 110,
    "CAM_BACK_RIGHT": -110,
}
B2D_WIDTH, B2D_HEIGHT, B2D_FOV = 80, 45, 70.0


def _b2d_frame(t: float, rng, speed: float = SPEED) -> Observation:
    """One request of Bench2Drive's client: each camera's pixels in the body, no file, and the
    route's next maneuver and how far along the route it is."""
    fx = B2D_WIDTH / 2 / math.tan(math.radians(B2D_FOV / 2))
    intrinsics = np.array([[fx, 0.0, B2D_WIDTH / 2], [0.0, fx, B2D_HEIGHT / 2], [0.0, 0.0, 1.0]])
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={c: rng.integers(0, 255, (B2D_HEIGHT, B2D_WIDTH, 3), dtype=np.uint8) for c in inputs.CAMERAS},
        ego2world=_pose(t),
        canbus=canbus,
        timestamp_us=int(round(t * 1e6)),
        command=Command.FOLLOW_REFERENCE,
        nav_maneuver="TURN_LEFT",
        nav_distance_m=37.4,
        camera2image={c: intrinsics for c in inputs.CAMERAS},
        camera2ego={c: _camera2ego(B2D_BEARINGS[c]) for c in inputs.CAMERAS},
    )


@pytest.fixture
def no_resize(monkeypatch):
    """Resizing to 1920x1080 is 30-40 ms a frame; these tests are about which frames are read."""
    monkeypatch.setattr(inputs, "warp", lambda image, fx, cx, cy, hfov=None: np.zeros((9, 16, 3), np.uint8))


def test_a_closed_loop_client_at_ten_hertz_fills_every_slot_with_its_own_frame(no_resize) -> None:
    """Asked every 0.1 s (`planning_hz`), the client hands Edge exactly the frames its slots ask
    for once 2 s have passed, and the route's next turn with its distance."""
    policy, rng = _loaded(), np.random.default_rng(0)
    for k in range(21):  # Bench2Drive's first tick is at 0.05 s; a plan every 0.1 s, 2 s of them
        plan = policy.infer(_b2d_frame(0.05 + 0.1 * k, rng))
    sample = policy.runs[-1]
    for camera_id in (1, 6):
        assert sample.frame_times_s[camera_id] == [-2.0, -1.0, -0.5, -0.1, 0.0]
    for camera_id in (0, 2, 3, 5):
        assert sample.frame_times_s[camera_id] == [-1.0, -0.1, 0.0]
    assert sample.extrapolated_s == 0.0, "the pose history is the one recorded, none of it continued"
    assert sample.nav_text == "Turn left in 37m"
    assert plan.dt == 0.1 and len(plan.waypoints) == inputs.PLAN_STEPS


def test_a_frame_sent_as_pixels_is_resized_once_not_once_per_plan_that_reads_it(no_resize) -> None:
    """Each plan reads 22 frames, 6 of them new; the other 16 are resized frames kept from an
    earlier plan. Before, with no file to know a frame by, all 22 were resized again."""
    policy, rng = _loaded(), np.random.default_rng(0)
    for k in range(25):
        policy.infer(_b2d_frame(0.05 + 0.1 * k, rng))
        assert policy._warp_misses == 6 * (k + 1), f"plan {k + 1}"
    assert policy._warp_hits == 22 * 25 - 6 * 25


def test_a_route_starts_from_its_first_frame_with_the_car_at_rest(no_resize) -> None:
    """The first plan of a route has one frame: every slot reads it, and the 1.9 s of history
    before it is the car standing where it is. A reset between routes forgets the last one,
    though the next route's clock starts again at the same time."""
    policy, rng = _loaded(), np.random.default_rng(0)
    for k in range(5):
        policy.infer(_b2d_frame(0.05 + 0.1 * k, rng))
    policy.reset()
    policy.infer(_b2d_frame(0.05, rng, speed=0.0))
    sample = policy.runs[-1]
    assert all(times == [0.0] * len(times) for times in sample.frame_times_s.values())
    assert sample.extrapolated_s == pytest.approx(1.9)
    assert np.allclose(sample.history_xyz, 0.0) and np.allclose(sample.history_rot, np.eye(3))
    assert policy._warp_misses == 6 * 6, "the new route's first frame is resized, not the last route's taken"


# -- model.yaml agrees with the policy ------------------------------------------------


def _spec():
    from paddock.registry import Registry

    return Registry.load().models["alpamayo2edge"]


def test_model_yaml_declares_the_cameras_and_spacing_the_policy_uses() -> None:
    spec = _spec()
    assert tuple(spec.cameras) == POLICY.cameras
    assert spec.waypoint_dt == POLICY.waypoint_dt == inputs.PLAN_DT_S
    assert spec.horizon_s == pytest.approx(inputs.PLAN_STEPS * inputs.PLAN_DT_S)
    assert spec.ego_history_steps == inputs.HISTORY_STEPS
    # The README's models picture draws the camera frames at the card's times.
    assert spec.camera_frame_times_s == inputs.FRONT_OFFSETS_S
    assert spec.history_frames == len(inputs.FRONT_OFFSETS_S)
    # A closed-loop client asks once per frame of the 0.1 s grid the slots and poses are on.
    assert spec.planning_hz == pytest.approx(1 / inputs.HISTORY_DT_S)


def test_model_yaml_declares_the_policys_defaults_and_passes_every_setting_through() -> None:
    """A declared value that differed from the code's default would record one setting
    in run.json and run another; a setting missing from env_passthrough never reaches
    the node from the launching shell."""
    spec = _spec()
    declared = {k: v for k, v in spec.runtime.env.items() if k.startswith("ALPAMAYO2_EDGE_")}
    assert Settings.from_env(declared) == Settings()
    knobs = {f"ALPAMAYO2_EDGE_{name.upper()}" for name in Settings.__dataclass_fields__}
    assert knobs == set(spec.runtime.env_passthrough)
    assert set(spec.runtime.tuning_env) == {"ALPAMAYO2_EDGE_SAVE_REASONING", "ALPAMAYO2_EDGE_THREADS"}
    # The one setting a site may change and stay ranked, with the line the report prints.
    assert spec.ranked_with_note == ("ALPAMAYO2_EDGE_ABSENT_CAMERAS",)
    assert "{value}" in spec.score_notes["ALPAMAYO2_EDGE_ABSENT_CAMERAS"]


# -- a rationale that never reaches the trajectory ------------------------------------


def _attempts(refused: int, error: str = "No <traj_future_start> token found in generated rows [0]."):
    seeds: list[int] = []

    def attempt(seed: int) -> str:
        seeds.append(seed)
        if len(seeds) <= refused:
            raise ValueError(error)
        return f"plan from seed {seed}"

    return attempt, seeds


def test_a_refused_draw_is_drawn_again_with_the_next_seed() -> None:
    """The vendor refuses a plan whose rationale has not reached the trajectory token in
    256 tokens. With the seed fixed, asking again would repeat the same draw, as NAVSIM's
    retry did for 2 navtest scenarios; the next seed is a new draw."""
    attempt, seeds = _attempts(refused=1)
    assert draw_until_complete(attempt, 43) == ("plan from seed 44", 44)
    assert seeds == [43, 44]


def test_a_draw_the_vendor_accepts_is_used_as_it_is() -> None:
    attempt, seeds = _attempts(refused=0)
    assert draw_until_complete(attempt, 43) == ("plan from seed 43", 43)
    assert seeds == [43]


def test_the_scenario_fails_when_every_draw_is_refused() -> None:
    attempt, seeds = _attempts(refused=MAX_DRAWS)
    with pytest.raises(ValueError, match="traj_future_start"):
        draw_until_complete(attempt, 43)
    assert seeds == list(range(43, 43 + MAX_DRAWS))


def test_any_other_error_is_not_drawn_again() -> None:
    attempt, seeds = _attempts(refused=1, error="Model produced nonfinite trajectory values")
    with pytest.raises(ValueError, match="nonfinite"):
        draw_until_complete(attempt, 43)
    assert seeds == [43]
