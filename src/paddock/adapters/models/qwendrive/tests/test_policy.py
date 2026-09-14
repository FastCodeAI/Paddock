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
from paddock.policy.protocol import N_WAYPOINTS, Command, Observation, Policy, Trajectory

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def make_observation(
    t_us: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, speed: float = 10.0
) -> Observation:
    ego2world = np.eye(4)
    ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    ego2world[:2, 3] = [x, y]
    canbus = np.zeros(18)
    canbus[13] = speed
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
