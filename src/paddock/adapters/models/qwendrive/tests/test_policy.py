"""Contract tests for the Qwen-Drive-1.0 adapter. No torch, no weights, no GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from paddock.adapters.models.qwendrive.policy import (
    CROP_SPAN,
    DRIVING_COMMAND,
    FRAME_DT,
    FRAME_TOLERANCE_S,
    GRID_INDICES,
    HISTORY_DT,
    KEEP_S,
    MAX_RECORDED,
    NAV_COMMAND,
    NEEDED_CAMERAS,
    NUM_CAM_FRAMES,
    NUM_HISTORY_STEPS,
    OUTPUT_POINTS,
    POLICY,
    VIEW_MODES,
    crop_box,
    medoid_index,
    pose_xy_yaw,
    resample_history,
    select_frame_indices,
)
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    N_WAYPOINTS,
    Command,
    Observation,
    Policy,
    Trajectory,
)

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def make_observation(
    t_us: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, speed: float = 10.0
) -> Observation:
    ego2world = np.eye(4)
    ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    ego2world[:2, 3] = [x, y]
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={},
        image_paths={c: f"/frames/{c}/{t_us}.jpg" for c in CAMERAS},
        ego2world=ego2world,
        canbus=canbus,
        timestamp_us=t_us,
        command=Command.STRAIGHT,
        camera2image={c: np.eye(3) for c in CAMERAS},
        camera2ego={c: np.eye(4) for c in CAMERAS},
    )


def _policy(**kwargs) -> POLICY:
    """Every knob named, so a site's own QWENDRIVE_* cannot move a test."""
    kwargs.setdefault("views", "ring")
    return POLICY(planner="planner-sft", mode="direct", num_samples=1, attn="sdpa", **kwargs)


# ---- configuration ----------------------------------------------------------


def test_policy_imports_without_the_model_environment() -> None:
    policy = _policy()
    assert isinstance(policy, Policy)
    assert policy.waypoint_dt == 0.5
    assert tuple(policy.cameras) == CAMERAS


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        _policy().infer(make_observation(0))


def test_planner_rl_refuses_direct_mode() -> None:
    """The vendor reward-optimised planner-rl under reasoning rollouts only."""
    with pytest.raises(ValueError, match="reasoning"):
        POLICY(planner="planner-rl", mode="direct")
    assert POLICY(planner="planner-rl", mode="reasoning").mode == "reasoning"
    with pytest.raises(ValueError, match="QWENDRIVE_MODE"):
        POLICY(planner="planner-sft", mode="vqa")


def test_defaults_come_from_the_environment(monkeypatch) -> None:
    for key in (
        "QWENDRIVE_PLANNER",
        "QWENDRIVE_MODE",
        "QWENDRIVE_NUM_SAMPLES",
        "QWENDRIVE_ATTN",
        "QWENDRIVE_SEED",
        "QWENDRIVE_VIEWS",
    ):
        monkeypatch.delenv(key, raising=False)
    p = POLICY()
    assert (p.planner, p.mode, p.num_samples, p.attn, p.seed) == ("planner-sft", "direct", 1, "sdpa", None)
    assert p.views == "ring"
    monkeypatch.setenv("QWENDRIVE_NUM_SAMPLES", "6")
    monkeypatch.setenv("QWENDRIVE_SEED", "7")
    q = POLICY()
    assert (q.num_samples, q.seed) == (6, 7)


# ---- the output grid --------------------------------------------------------


def test_grid_indices_land_on_the_half_second_grid() -> None:
    """The model emits 50 poses at 10 Hz; pose i is at (i+1)/10 s. Ten points,
    the whole 5 s: NAVSIM scores the first eight, neuroNCAP the first six and
    KITScenes three seconds, so the tail costs them nothing and comma2k19 scores
    ADE@5s on the same spacing as everything else on that board."""
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
    assert list(GRID_INDICES) == [int(round(t / 0.1)) - 1 for t in grid]
    assert len(GRID_INDICES) >= N_WAYPOINTS
    assert GRID_INDICES[-1] == OUTPUT_POINTS - 1


def test_the_grid_reaches_the_horizon_the_card_declares() -> None:
    """`horizon_s` in model.yaml is what the dashboard's time axis reads. A grid
    that stopped short would draw a 5 s plan over 4 s of waypoints."""
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "model.yaml").read_text())
    caps = spec["capabilities"]
    assert len(GRID_INDICES) * caps["waypoint_dt"] == pytest.approx(caps["horizon_s"])


def test_medoid_picks_the_consensus_draw() -> None:
    a = np.zeros((50, 3))
    b = np.ones((50, 3)) * 0.1
    outlier = np.ones((50, 3)) * 100.0
    assert medoid_index(np.stack([outlier, a, b])) in (1, 2)
    assert medoid_index(np.stack([a])) == 0


# ---- frame selection --------------------------------------------------------


def test_frame_slots_are_half_a_second_apart() -> None:
    assert (NUM_CAM_FRAMES, FRAME_DT, NUM_HISTORY_STEPS, HISTORY_DT) == (4, 0.5, 16, 0.1)


def test_a_10hz_source_contributes_every_fifth_frame() -> None:
    times = np.arange(20) * 0.1
    indices, gaps = select_frame_indices(times)
    assert indices == [4, 9, 14, 19]
    assert max(gaps) < 1e-9


def test_a_2hz_source_contributes_every_frame() -> None:
    """NAVSIM's four OpenScene keyframes are exactly the model's four slots."""
    times = np.arange(4) * 0.5
    indices, gaps = select_frame_indices(times)
    assert indices == [0, 1, 2, 3]
    assert max(gaps) < 1e-9


def test_a_short_history_repeats_its_oldest_frame_visibly() -> None:
    times = np.array([0.0, 0.1])
    indices, gaps = select_frame_indices(times)
    assert indices == [0, 0, 0, 1]
    assert gaps[0] > FRAME_TOLERANCE_S and gaps[-1] < 1e-9


def test_jitter_within_tolerance_is_not_a_stand_in() -> None:
    times = np.arange(20) * 0.1 + np.array([0.004 * (i % 2) for i in range(20)])
    _, gaps = select_frame_indices(times)
    assert max(gaps) < FRAME_TOLERANCE_S


# ---- pose history -----------------------------------------------------------


def _straight(n: int, speed: float, yaw: float, hz: float = 10.0):
    """n recorded frames driving straight at `speed` along heading `yaw`."""
    t = np.arange(n) / hz
    poses = np.stack([speed * t * np.cos(yaw), speed * t * np.sin(yaw), np.full(n, yaw)], axis=1)
    motion = np.tile([speed, 0.0, 0.0, 0.0], (n, 1))
    return t, poses, motion


def test_history_is_expressed_in_the_current_ego_frame() -> None:
    """Driving north in the world must read as driving along +x for the model."""
    t, poses, motion = _straight(20, speed=10.0, yaw=np.pi / 2)
    history, velocity, acceleration = resample_history(t, poses, motion)
    assert history.shape == (16, 3) and velocity.shape == (16, 2) and acceleration.shape == (16, 2)
    np.testing.assert_allclose(history[:, 0], (np.arange(16) - 15) * HISTORY_DT * 10.0, atol=1e-9)
    np.testing.assert_allclose(history[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(history[:, 2], 0.0, atol=1e-9)
    np.testing.assert_array_equal(history[-1], 0.0)
    np.testing.assert_allclose(velocity[:, 0], 10.0)


def test_a_left_turn_puts_the_past_behind_and_to_the_left_with_negative_heading() -> None:
    """Past poses on a left turn had a smaller yaw than now, so their heading
    relative to now is negative. The arc's centre is on the left, so the past
    positions lie behind and to the LEFT of the current axis (y > 0) -- the
    easy sign to get backwards, which is why it is pinned."""
    t = np.arange(20) * 0.1
    yaw = 0.3 * t  # a circle turned left at 0.3 rad/s and 10 m/s
    radius = 10.0 / 0.3
    poses = np.stack([radius * np.sin(yaw), radius * (1 - np.cos(yaw)), yaw], axis=1)
    motion = np.tile([10.0, 0.0, 0.0, 0.0], (20, 1))
    history, _, _ = resample_history(t, poses, motion)
    assert (history[:-1, 2] < 0).all()
    assert (history[:-1, 1] > 0).all()
    assert (history[:-1, 0] < 0).all()


def test_yaw_wraparound_does_not_spike_the_history() -> None:
    t, poses, motion = _straight(20, speed=10.0, yaw=np.pi - 0.01)
    poses[10:, 2] = -np.pi + 0.01  # the same heading, wrapped the other way
    history, _, _ = resample_history(t, poses, motion)
    assert np.abs(history[:, 2]).max() < 0.05


def test_acceleration_is_differenced_when_the_source_records_none() -> None:
    """KITScenes fills canbus with speed only."""
    t = np.arange(20) * 0.1
    poses = np.stack([0.5 * 2.0 * t**2, np.zeros(20), np.zeros(20)], axis=1)
    motion = np.stack([2.0 * t, np.zeros(20), np.zeros(20), np.zeros(20)], axis=1)
    _, velocity, acceleration = resample_history(t, poses, motion)
    np.testing.assert_allclose(acceleration[:, 0], 2.0, atol=1e-6)
    recorded = motion.copy()
    recorded[:, 2] = 3.0
    _, _, from_source = resample_history(t, poses, recorded)
    np.testing.assert_allclose(from_source[:, 0], 3.0)


def test_how_much_history_is_kept_does_not_decide_how_acceleration_is_derived() -> None:
    """`KEEP_S` bounds memory. It must not also pick a modelling branch.

    The decision "did the source record acceleration, or must it be differenced"
    used to be read off the retained rows, so a stream whose source filled the
    channel only in older records answered 2.0 trimmed and 0.0 untrimmed from
    the same input. The policy now tracks it as records arrive.
    """
    p = _policy()
    accel = 2.0
    for i in range(60):  # 6 s at 10 Hz, well past KEEP_S
        obs = make_observation(i * 100_000, x=0.5 * accel * (i * 0.1) ** 2, speed=accel * i * 0.1)
        # Only records well outside the retained window carry the accel channel,
        # so every row the trimmed history can see has a zero there.
        if i * 0.1 < 6.0 - KEEP_S - 1.0:
            obs.canbus[CANBUS_ACCEL] = accel
        p._record(obs)

    assert p.source_has_accel is True, "the channel was populated, even if those rows are gone"
    inputs = p.scene_inputs(Command.STRAIGHT)
    # Trusted as recorded, which for the retained rows is 0.0 -- not differenced.
    assert float(inputs["ego_acceleration"][0]) == 0.0

    # And the sticky flag is per episode, not per process.
    p.reset()
    assert p.source_has_accel is False


def test_a_source_that_never_records_acceleration_still_gets_it_differenced() -> None:
    """KITScenes fills speed only; the differenced path must survive the change."""
    p = _policy()
    for i in range(30):
        p._record(make_observation(i * 100_000, x=0.5 * 2.0 * (i * 0.1) ** 2, speed=2.0 * i * 0.1))
    assert p.source_has_accel is False
    inputs = p.scene_inputs(Command.STRAIGHT)
    assert float(inputs["ego_acceleration"][0]) == pytest.approx(2.0, abs=1e-6)


def test_a_sparse_early_history_is_a_standing_start_not_a_crash() -> None:
    t, poses, motion = _straight(1, speed=5.0, yaw=0.0)
    history, velocity, acceleration = resample_history(t, poses, motion)
    np.testing.assert_array_equal(history, 0.0)
    np.testing.assert_allclose(velocity[:, 0], 5.0)
    assert np.isfinite(acceleration).all()


def test_pose_xy_yaw_reads_the_rotated_x_axis() -> None:
    m = np.eye(4)
    m[:2, :2] = [[0, -1], [1, 0]]
    m[:3, 3] = [3.0, 4.0, 9.0]
    np.testing.assert_allclose(pose_xy_yaw(m), [3.0, 4.0, np.pi / 2])


# ---- commands ---------------------------------------------------------------


def test_command_mapping_round_trips_navsims_one_hot() -> None:
    assert (
        NAV_COMMAND[Command.STRAIGHT] == 0
        and NAV_COMMAND[Command.LEFT] == 1
        and NAV_COMMAND[Command.RIGHT] == 2
    )
    assert DRIVING_COMMAND[Command.LEFT] == (1, 0, 0, 0)
    assert DRIVING_COMMAND[Command.STRAIGHT] == (0, 1, 0, 0)
    assert DRIVING_COMMAND[Command.RIGHT] == (0, 0, 1, 0)
    assert DRIVING_COMMAND[Command.FOLLOW_REFERENCE] == (0, 0, 0, 1)
    assert NAV_COMMAND[Command.FOLLOW_REFERENCE] == 0, "the text vocabulary has no 'unknown'"
    assert set(NAV_COMMAND) == set(DRIVING_COMMAND) == set(Command)


# ---- recording --------------------------------------------------------------


def test_repeated_frames_are_recorded_once() -> None:
    p = _policy()
    for t in (0, 0, 0, 100_000, 100_000, 200_000):
        p._record(make_observation(t))
    assert p.timestamps == [0, 100_000, 200_000] and len(p.frames) == 3


def test_old_records_are_dropped_without_changing_what_the_model_sees() -> None:
    """Retention is bounded by time, because a record holds decoded pixels.

    The model never reads past 1.5 s, so trimming to KEEP_S must leave the four
    selected frames and the 16-pose history bit-for-bit what an untrimmed buffer
    would have given. Checked here by asserting the selected frames land exactly
    on the grid and the history spans exactly 1.5 s.
    """
    p = _policy()
    for i in range(60):  # 6 s at 10 Hz, well past KEEP_S
        p._record(make_observation(i * 100_000, x=i * 1.0))

    assert len(p.timestamps) <= int(KEEP_S * 10) + 1 < 60, "old records must be dropped"

    inputs = p.scene_inputs(Command.STRAIGHT)
    now = p.timestamps[-1] / 1e6
    chosen = [p.timestamps[i] / 1e6 for i in inputs["frame_indices"]]
    assert chosen == [now - 1.5, now - 1.0, now - 0.5, now]
    assert max(inputs["frame_gaps_s"]) < 1e-9, "every slot lands on its own time"
    assert inputs["history"].shape == (16, 3)
    np.testing.assert_array_equal(inputs["history"][-1], 0.0)
    # 1.5 s at 10 m/s: the oldest pose sits 15 m behind the current one.
    assert inputs["history"][0][0] == pytest.approx(-15.0)


def test_a_slow_source_keeps_enough_frames_to_fill_four_slots() -> None:
    """A 2 Hz source spans 1.5 s in four records; a 1 Hz one spans more than
    KEEP_S. Neither may be trimmed below four, or a slot has nothing to take."""
    p = _policy()
    for i in range(6):
        p._record(make_observation(i * 1_000_000))  # 1 Hz, 5 s of span
    assert len(p.timestamps) >= NUM_CAM_FRAMES
    indices, _ = select_frame_indices(np.array(p.timestamps) / 1e6)
    assert len(indices) == NUM_CAM_FRAMES


def test_the_record_count_is_capped_whatever_the_cadence() -> None:
    p = _policy()
    for i in range(MAX_RECORDED * 2):
        p._record(make_observation(i * 1_000))  # 1 kHz: KEEP_S would keep them all
    assert len(p.timestamps) == MAX_RECORDED


def test_a_drifting_cadence_is_counted_even_though_no_frame_repeats() -> None:
    """The quiet failure the tolerance exists for.

    A source ticking at 0.2 s fills all four slots with distinct frames that are
    each up to 0.1 s from the time they asked for, so `frames_repeated_total`
    stays 0 while the history is not the 1.5 s the model was told it is.
    """
    p = _loaded()
    for i in range(20):
        p.observe(make_observation(i * 200_000, x=i * 2.0))  # 5 Hz
    p.infer(make_observation(20 * 200_000, x=40.0))

    d = p.diagnostics()
    assert d["frames_repeated_total"] == 0, "distinct frames, so nothing repeats"
    assert d["frames_off_grid_total"] > 0, "but they are not on the 0.5 s grid"


def test_a_missing_camera_is_a_clear_error() -> None:
    p = _policy()
    obs = make_observation(0)
    obs.image_paths.pop("CAM_FRONT_LEFT")
    with pytest.raises(KeyError, match="CAM_FRONT_LEFT"):
        p._record(obs)


def test_pixels_and_paths_are_both_kept_as_given() -> None:
    p = _policy()
    obs = make_observation(0)
    pixels = np.zeros((4, 4, 3), np.uint8)
    obs.image_paths.pop("CAM_FRONT")
    obs.images["CAM_FRONT"] = pixels
    p._record(obs)
    assert p.frames[0]["CAM_FRONT"] is pixels
    assert p.frames[0]["CAM_FRONT_LEFT"] == "/frames/CAM_FRONT_LEFT/0.jpg"


def test_motion_is_read_at_the_protocols_declared_offsets() -> None:
    """The CAN-bus layout is imported, not re-typed.

    Built here through `CANBUS_VELOCITY` and `CANBUS_ACCEL` rather than through
    13 and 7, so if the protocol ever renumbers the layout this test moves with
    it and an adapter still reading channel 13 fails here instead of silently
    conditioning its plan on whatever now sits there.
    """
    p = _policy()
    obs = make_observation(0)
    obs.canbus[:] = 0.0
    obs.canbus[CANBUS_VELOCITY] = 7.5
    obs.canbus[CANBUS_VELOCITY + 1] = -0.25
    obs.canbus[CANBUS_ACCEL] = 1.5
    obs.canbus[CANBUS_ACCEL + 1] = 0.75
    p._record(obs)

    inputs = p.scene_inputs(Command.STRAIGHT)
    assert inputs["ego_velocity"] == (7.5, -0.25)
    assert inputs["ego_acceleration"] == (1.5, 0.75)


def test_a_renumbered_canbus_layout_moves_this_adapter_with_it(monkeypatch) -> None:
    """The test above passes either way today, because CANBUS_VELOCITY is 13.

    This one does not: it renumbers the layout under the adapter and checks the
    adapter follows. A hard-coded `canbus[13]` keeps reading the old channel and
    fails here, which is the whole point of importing the constant.
    """
    from paddock.adapters.models.qwendrive import policy as mod

    monkeypatch.setattr(mod, "CANBUS_VELOCITY", 3)
    monkeypatch.setattr(mod, "CANBUS_ACCEL", 9)

    p = _policy()
    obs = make_observation(0)
    obs.canbus[:] = 0.0
    obs.canbus[3] = 11.0
    obs.canbus[4] = 0.5
    obs.canbus[9] = 2.0
    obs.canbus[10] = -1.0
    p._record(obs)

    inputs = p.scene_inputs(Command.STRAIGHT)
    assert inputs["ego_velocity"] == (11.0, 0.5)
    assert inputs["ego_acceleration"] == (2.0, -1.0)


def test_scene_inputs_from_a_10hz_stream() -> None:
    p = _policy()
    for i in range(20):
        p._record(make_observation(i * 100_000, x=i * 1.0, speed=10.0))
    inputs = p.scene_inputs(Command.LEFT)
    assert inputs["frame_indices"] == [4, 9, 14, 19]
    assert inputs["history"].shape == (16, 3)
    assert inputs["ego_velocity"] == (10.0, 0.0)
    assert inputs["driving_command"] == [1.0, 0.0, 0.0, 0.0] and inputs["nav_command"] == 1


# ---- the infer path, with a stub model ------------------------------------


class _StubModel:
    def __init__(self, draws: int = 1):
        self.draws = draws
        self.calls: list[dict] = []

    def generate_trajectory(self, scene, mode, num_samples, seed):
        self.calls.append({"scene": scene, "mode": mode, "num_samples": num_samples, "seed": seed})
        t = (np.arange(OUTPUT_POINTS) + 1) * 0.1
        one = np.stack([10.0 * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        return NS(trajectories=np.stack([one] * self.draws), reasoning=None)


def _loaded(draws: int = 1) -> POLICY:
    p = _policy()
    p.model = _StubModel(draws)
    p._mode = "direct_planning"
    p._scene = lambda obs, inputs: inputs  # no qwen_drive in this env
    return p


def test_infer_obeys_the_contract() -> None:
    p = _loaded()
    p.reset()
    for i in range(20):
        p.observe(make_observation(i * 100_000, x=i * 1.0))
    traj = p.infer(make_observation(20 * 100_000, x=20.0))
    assert isinstance(traj, Trajectory) and traj.dt == 0.5
    assert traj.waypoints.shape == (10, 3)
    np.testing.assert_allclose(traj.xy[:, 0], [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    assert np.isfinite(traj.waypoints).all()
    assert not np.allclose(traj.xy, 0.0)
    assert p.model.calls[0]["num_samples"] == 1


def test_several_draws_execute_the_medoid() -> None:
    p = _loaded(draws=3)
    p.num_samples = 3
    traj = p.infer(make_observation(0))
    assert traj.waypoints.shape == (10, 3)


def test_infer_sequence_records_the_history_and_runs_the_model_once() -> None:
    p = _loaded()
    frames = [make_observation(i * 500_000, x=i * 5.0) for i in range(4)]
    traj = p.infer_sequence(frames)
    assert len(p.model.calls) == 1, "exactly one forward pass, on the scored frame"
    assert p.timestamps == [f.timestamp_us for f in frames]
    assert p.self_prime is True
    assert traj.waypoints.shape == (10, 3)


def test_diagnostics_report_cadence_and_stand_ins() -> None:
    p = _loaded()
    assert p.diagnostics() == {"status": "no frame seen yet"}
    p.infer(make_observation(0))
    p.infer(make_observation(100_000))
    d = p.diagnostics()
    assert d["request_dt_s"] == 0.1
    assert d["frames_repeated_total"] == 3 + 2, "one frame fills four slots, then two fill four"
    assert d["last_command"] == "STRAIGHT" and d["infer_seconds_mean"] is not None


# ---- filling three view slots from one camera ------------------------------


def _mono(t_us: int, **kwargs) -> Observation:
    """One observation from a source with a road camera and nothing else."""
    obs = make_observation(t_us, **kwargs)
    for camera in CAMERAS[1:]:
        obs.image_paths.pop(camera)
        obs.camera2image.pop(camera)
        obs.camera2ego.pop(camera)
    return obs


def test_a_monocular_mode_declares_the_one_camera_it_reads() -> None:
    """`cameras` is what a simulator renders and what `paddock verify` checks a
    benchmark can supply, so it has to follow the view mode rather than stay at
    the three the model was trained on."""
    assert tuple(_policy(views="ring").cameras) == CAMERAS
    for mode in ("replicate", "crop"):
        assert tuple(_policy(views=mode).cameras) == (NEEDED_CAMERAS[0],)


def test_an_unknown_view_mode_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="QWENDRIVE_VIEWS"):
        _policy(views="stereo")
    assert set(VIEW_MODES) == {"ring", "replicate", "crop"}


def test_the_view_mode_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("QWENDRIVE_VIEWS", raising=False)
    assert POLICY().views == "ring"
    monkeypatch.setenv("QWENDRIVE_VIEWS", "replicate")
    assert POLICY().views == "replicate"


def test_a_monocular_source_records_without_a_missing_camera_error() -> None:
    """The same observation that `ring` refuses, `replicate` accepts. Both are
    right: the refusal is what stops a ring model reading a third of its input
    and calling the result a number."""
    p = _policy(views="replicate")
    p._record(_mono(0))
    assert p.frames[0] == {"CAM_FRONT": "/frames/CAM_FRONT/0.jpg"}
    with pytest.raises(KeyError, match="CAM_FRONT_LEFT"):
        _policy(views="ring")._record(_mono(0))


def test_crop_boxes_are_overlapping_thirds_of_one_frame() -> None:
    size = (1164, 874)
    left = crop_box(size, "<FRONT LEFT VIEW>")
    front = crop_box(size, "<FRONT VIEW>")
    right = crop_box(size, "<FRONT RIGHT VIEW>")
    widths = {box[2] - box[0] for box in (left, front, right)}
    assert widths == {int(1164 * CROP_SPAN)}, "one width, so the three read at one scale"
    assert all(box[1] == 0 and box[3] == 874 for box in (left, front, right)), "full height"
    assert left[0] == 0 and right[2] == 1164, "the sides keep the shoulders"
    assert left[2] > front[0] and right[0] < front[2], "and overlap the centre"
    assert front[0] + front[2] == 1164, "which keeps the vanishing point centred"


class _FakeQwenDrive:
    """Just enough of the vendor package to build a scene without importing it.

    `_scene` is the one method the other tests stub out, and it is where the
    view slots are actually filled -- so it gets a fake rather than a skip.
    """

    class CameraFrame:
        def __init__(self, image, target_size=None):
            self.image = image
            self.target_size = target_size

    class DrivingScene:
        def __init__(self, views, token="", **rest):
            self.views = views
            self.token = token
            self.rest = rest


@pytest.fixture
def qwen_drive(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "qwen_drive", _FakeQwenDrive)
    return _FakeQwenDrive


def _scene_of(policy: POLICY, obs: Observation):
    policy._record(obs)
    return policy._scene(obs, policy.scene_inputs(obs.command))


def test_replicate_puts_the_true_forward_frame_in_all_three_slots(qwen_drive) -> None:
    """Measured against the alternative: 1.47 / 2.00 ADE on comma2k19's city and
    highway segments, against crop's 1.78 / 2.13."""
    scene = _scene_of(_policy(views="replicate"), _mono(0))
    assert sorted(scene.views) == ["<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>", "<FRONT VIEW>"]
    sources = {frame.image for frames in scene.views.values() for frame in frames}
    assert sources == {"/frames/CAM_FRONT/0.jpg"}, "the path, unopened: the processor decides when"


def test_crop_defers_the_decode_and_cuts_the_slot_it_was_asked_for(qwen_drive, tmp_path) -> None:
    """A loader, not an image. comma2k19 hands over 160 frames and the model
    reads four; opening on scene build would decode all of them."""
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (1164, 874), (10, 20, 30)).save(path)
    obs = _mono(0)
    obs.image_paths["CAM_FRONT"] = str(path)

    scene = _scene_of(_policy(views="crop"), obs)
    for view, frames in scene.views.items():
        assert all(callable(frame.image) for frame in frames), "nothing opened yet"
        cut = frames[-1].image()
        assert cut.size == (int(1164 * CROP_SPAN), 874)
        assert cut.tobytes() == Image.open(path).convert("RGB").crop(crop_box((1164, 874), view)).tobytes()


def test_a_staged_raw_frame_is_read_as_an_array_and_still_deferred(qwen_drive, tmp_path) -> None:
    """comma2k19 stages `.npy`, because PNG decode was 59% of a node's wall
    clock. PIL cannot open one, so the path cannot go through as a path -- but
    it must still not be read until the processor asks for it."""
    pixels = np.random.default_rng(0).integers(0, 255, (874, 1164, 3), dtype=np.uint8)
    path = tmp_path / "000966.npy"
    np.save(path, pixels)
    obs = _mono(0)
    obs.image_paths["CAM_FRONT"] = str(path)

    scene = _scene_of(_policy(views="replicate"), obs)
    for frames in scene.views.values():
        assert all(callable(frame.image) for frame in frames), "nothing read yet"
        assert np.array_equal(np.asarray(frames[-1].image()), pixels)


def test_a_ring_source_still_gets_one_real_camera_per_slot(qwen_drive) -> None:
    scene = _scene_of(_policy(views="ring"), make_observation(0))
    assert {view: frames[-1].image for view, frames in scene.views.items()} == {
        "<FRONT VIEW>": "/frames/CAM_FRONT/0.jpg",
        "<FRONT LEFT VIEW>": "/frames/CAM_FRONT_LEFT/0.jpg",
        "<FRONT RIGHT VIEW>": "/frames/CAM_FRONT_RIGHT/0.jpg",
    }


def test_diagnostics_name_the_view_mode_and_the_cameras_it_implies() -> None:
    p = _loaded()
    p.views = "replicate"
    p.cameras = (NEEDED_CAMERAS[0],)
    p.infer(_mono(0))
    d = p.diagnostics()
    assert d["views"] == "replicate" and d["cameras"] == ["CAM_FRONT"]
