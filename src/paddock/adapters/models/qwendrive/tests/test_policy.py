"""Contract tests for the Qwen-Drive-1.0 adapter. No torch, no weights, no GPU."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from paddock.adapters.models.qwendrive.policy import (
    DRIVING_COMMAND,
    FRAME_DT,
    FRAME_TOLERANCE_S,
    GRID_INDICES,
    HISTORY_DT,
    KEEP_S,
    MAX_RECORDED,
    NAV_COMMAND,
    NUM_CAM_FRAMES,
    NUM_HISTORY_STEPS,
    OUTPUT_POINTS,
    POLICY,
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
    ):
        monkeypatch.delenv(key, raising=False)
    p = POLICY()
    assert (p.planner, p.mode, p.num_samples, p.attn, p.seed) == ("planner-sft", "direct", 1, "sdpa", None)
    monkeypatch.setenv("QWENDRIVE_NUM_SAMPLES", "6")
    monkeypatch.setenv("QWENDRIVE_SEED", "7")
    q = POLICY()
    assert (q.num_samples, q.seed) == (6, 7)


# ---- the output grid --------------------------------------------------------


def test_grid_indices_land_on_the_half_second_grid() -> None:
    """The model emits 50 poses at 10 Hz; pose i is at (i+1)/10 s. Eight points,
    4 s, because NAVSIM scores 4 s; neuroNCAP takes six, KITScenes three seconds."""
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert list(GRID_INDICES) == [int(round(t / 0.1)) - 1 for t in grid]
    assert len(GRID_INDICES) >= N_WAYPOINTS
    assert GRID_INDICES[-1] < OUTPUT_POINTS


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
    assert traj.waypoints.shape == (8, 3)
    np.testing.assert_allclose(traj.xy[:, 0], [5, 10, 15, 20, 25, 30, 35, 40])
    assert np.isfinite(traj.waypoints).all()
    assert not np.allclose(traj.xy, 0.0)
    assert p.model.calls[0]["num_samples"] == 1


def test_several_draws_execute_the_medoid() -> None:
    p = _loaded(draws=3)
    p.num_samples = 3
    traj = p.infer(make_observation(0))
    assert traj.waypoints.shape == (8, 3)


def test_infer_sequence_records_the_history_and_runs_the_model_once() -> None:
    p = _loaded()
    frames = [make_observation(i * 500_000, x=i * 5.0) for i in range(4)]
    traj = p.infer_sequence(frames)
    assert len(p.model.calls) == 1, "exactly one forward pass, on the scored frame"
    assert p.timestamps == [f.timestamp_us for f in frames]
    assert p.self_prime is True
    assert traj.waypoints.shape == (8, 3)


def test_diagnostics_report_cadence_and_stand_ins() -> None:
    p = _loaded()
    assert p.diagnostics() == {"status": "no frame seen yet"}
    p.infer(make_observation(0))
    p.infer(make_observation(100_000))
    d = p.diagnostics()
    assert d["request_dt_s"] == 0.1
    assert d["frames_repeated_total"] == 3 + 2, "one frame fills four slots, then two fill four"
    assert d["last_command"] == "STRAIGHT" and d["infer_seconds_mean"] is not None
