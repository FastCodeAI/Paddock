"""Qwen-Drive on AlpaSim: the native driver's inputs, built in paddock's node. No torch, no GPU.

A body carrying a scene id is AlpaSim's (the qwendrive profile's bridge). These
tests feed the node 20 Hz streams the way that bridge sends them and check what
the model would be handed: the ego rows and their acceleration, the route's
command and its hold, the frames, the seed, and the plan that goes back.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from paddock.adapters.models.qwendrive.policy import (
    ALPASIM_WAYPOINT_DT,
    BASE_NOISE_SEED,
    NUM_HISTORY_STEPS,
    OUTPUT_POINTS,
    POLICY,
    ROUTE_COMMANDS,
    STOP_REACH_M,
    CommandHold,
    alpasim_frame_indices,
    alpasim_history,
    route_command,
    scene_seed,
)
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    Observation,
)

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
STEP_US = 50_000
SCENE = "clipgt-16b83346-test"


def _matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    m[:3, 3] = [x, y, z]
    return m


def _turning(t_s: float, speed: float = 10.0, yaw_rate: float = 0.2) -> tuple[float, float, float]:
    """(x, y, yaw) on a circle at constant *speed* and *yaw_rate*."""
    radius = speed / yaw_rate
    yaw = yaw_rate * t_s
    return radius * math.sin(yaw), radius * (1.0 - math.cos(yaw)), yaw


def _observation(t_us: int, route_xy=None, scene_id: str | None = SCENE, **motion) -> Observation:
    """One body as the qwendrive profile's bridge sends it, at *t_us* on the turning car."""
    speed, yaw_rate = motion.get("speed", 10.0), motion.get("yaw_rate", 0.2)
    x, y, yaw = _turning(t_us / 1e6, speed, yaw_rate)
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = motion.get("vx", speed)
    canbus[CANBUS_VELOCITY + 1] = motion.get("vy", 0.0)
    canbus[CANBUS_YAW_RATE] = yaw_rate
    canbus[CANBUS_ACCEL] = motion.get("ax", 0.0)
    return Observation(
        images={c: np.full((4, 6, 3), (t_us // STEP_US) % 256, dtype=np.uint8) for c in CAMERAS},
        ego2world=_matrix(x, y, yaw),
        canbus=canbus,
        timestamp_us=t_us,
        command=Command.FOLLOW_REFERENCE,
        camera2image={},
        camera2ego={},
        route_xy=route_xy,
        scene_id=scene_id,
    )


def _policy() -> POLICY:
    return POLICY(planner="planner-sft", mode="direct", num_samples=1, attn="sdpa", views="ring")


class _StubModel:
    """Plans *reach_m* straight ahead over 5 s, 50 points; records what it was given."""

    def __init__(self, reach_m: float = 50.0) -> None:
        self.reach_m = reach_m
        self.calls: list[dict] = []

    def generate_trajectory(self, scene, mode, num_samples, seed):
        self.calls.append({"scene": scene, "seed": seed})
        t = (np.arange(OUTPUT_POINTS) + 1) / OUTPUT_POINTS
        plan = np.stack([self.reach_m * t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        return NS(trajectories=plan[None], reasoning=None)


def _loaded(reach_m: float = 50.0) -> POLICY:
    p = _policy()
    p.model = _StubModel(reach_m)
    p._mode = "direct_planning"
    p._scene = lambda obs, inputs: inputs  # no qwen_drive in this env
    return p


def _drive(
    policy: POLICY, until_us: int, route_xy=None, infer_every_us: int = 1_500_000, start_us: int = 3_000_000
):
    """Observe every 50 ms step and infer on the replan grid from *start_us*, as the profile does."""
    plans = []
    for t in range(0, until_us + 1, STEP_US):
        obs = _observation(t, route_xy)
        if t >= start_us and (t - start_us) % infer_every_us == 0:
            plans.append(policy.infer(obs))
        else:
            policy.observe(obs)
    return plans


# ---- acceleration ---------------------------------------------------------


def test_acceleration_rows_follow_the_native_formula_with_the_turning_term() -> None:
    """a_fwd = dv_fwd/dt - w v_lat, a_lat = dv_lat/dt + w v_fwd, at 0.1 s, row 0 from 0.1 s older."""
    times = np.arange(0, 2.05, 0.05)
    rng = np.random.default_rng(3)
    velocity = rng.normal(8.0, 1.0, (len(times), 2))
    yaw_rate = rng.normal(0.0, 0.2, len(times))
    matrices = np.stack([_matrix(*_turning(t)) for t in times])
    _, rows_v, accel, fill = alpasim_history(times, matrices, velocity, yaw_rate)
    assert fill == 0
    picked = [len(times) - 1 - 2 * (NUM_HISTORY_STEPS - 1 - row) for row in range(NUM_HISTORY_STEPS)]
    for row, index in enumerate(picked):
        before = index - 2
        dv = (velocity[index] - velocity[before]) / 0.1
        w = yaw_rate[index]
        expected = (dv[0] - w * velocity[index, 1], dv[1] + w * velocity[index, 0])
        np.testing.assert_allclose(accel[row], expected, atol=1e-12)
        np.testing.assert_array_equal(rows_v[row], velocity[index])


def test_a_steady_turn_reads_as_centripetal_acceleration_not_zero() -> None:
    """Constant speed on a circle: no forward acceleration, w*v sideways (here 0.2 * 10 = 2 m/s^2)."""
    times = np.arange(0, 2.05, 0.05)
    matrices = np.stack([_matrix(*_turning(t)) for t in times])
    velocity = np.tile([10.0, 0.0], (len(times), 1))
    _, _, accel, _ = alpasim_history(times, matrices, velocity, np.full(len(times), 0.2))
    np.testing.assert_allclose(accel[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(accel[:, 1], 2.0, atol=1e-12)


def test_history_rows_are_the_samples_in_the_current_frame() -> None:
    times = np.arange(0, 2.05, 0.05)
    matrices = np.stack([_matrix(*_turning(t)) for t in times])
    history, _, _, _ = alpasim_history(times, matrices, np.zeros((len(times), 2)), np.zeros(len(times)))
    np.testing.assert_array_equal(history[-1], 0.0)
    # 1.5 s back on a left-hand circle: behind and to the left, heading -0.3 rad.
    x, y, heading = history[0]
    assert x < -14.0 and y > 2.0 and heading == pytest.approx(-0.3, abs=1e-12)


def test_a_short_buffer_fills_from_the_oldest_sample_and_says_so() -> None:
    times = np.arange(0, 0.55, 0.05)
    matrices = np.stack([_matrix(*_turning(t)) for t in times])
    _, _, _, fill = alpasim_history(times, matrices, np.zeros((len(times), 2)), np.zeros(len(times)))
    # Rows at now-0.6 s and older are past the 0.5 s buffer (the one at now-0.6 s is 0.1 s
    # from its oldest sample, within reach): nine fills, plus row 0's missing predecessor.
    assert fill == 9 + 1


def test_the_node_ignores_the_simulators_acceleration_on_alpasim() -> None:
    p = _policy()
    for t in range(0, 2_000_001, STEP_US):
        p._record(_observation(t, ax=5.0))
    inputs = p.alpasim_inputs("STRAIGHT")
    np.testing.assert_allclose(inputs["history_acceleration"][:, 0], 0.0, atol=1e-9)


# ---- route -> command -----------------------------------------------------


def _arc(curvature: float, length: float = 60.0, step: float = 2.0) -> np.ndarray:
    s = np.arange(-4.0, length, step)
    if curvature == 0.0:
        return np.stack([s, np.zeros_like(s)], axis=1)
    return np.stack([np.sin(curvature * s) / curvature, (1 - np.cos(curvature * s)) / curvature], axis=1)


@pytest.mark.parametrize(
    "route, expected",
    [
        (None, "AMBIGUOUS"),
        (np.zeros((0, 2)), "AMBIGUOUS"),
        (_arc(0.0), "STRAIGHT"),
        (_arc(0.05), "LEFT"),
        (_arc(-0.05), "RIGHT"),
        (_arc(0.005), "STRAIGHT"),  # a gentle bend is not a turn
        (_arc(0.0, length=10.0), "STRAIGHT"),  # too short to see a turn in
    ],
)
def test_route_command_three_answer_rule(route, expected) -> None:
    assert route_command(route) == expected


def test_a_route_that_contradicts_itself_is_ambiguous() -> None:
    """Off to the left at 20 m, yet the heading turned right on the way: the answers disagree."""
    route = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 3.0], [6.0, 6.0], [8.0, 9.0], [12.0, 10.0], [20.0, 9.0], [30.0, 4.0]]
    )
    assert route_command(route) == "AMBIGUOUS"


def test_ambiguous_is_unknown_in_the_one_hot_and_straight_in_the_text() -> None:
    assert ROUTE_COMMANDS["AMBIGUOUS"] == (0, (0.0, 0.0, 0.0, 1.0))
    assert ROUTE_COMMANDS["LEFT"] == (1, (1.0, 0.0, 0.0, 0.0))
    assert ROUTE_COMMANDS["RIGHT"] == (2, (0.0, 0.0, 1.0, 0.0))


# ---- the hold ---------------------------------------------------------------


def test_a_new_command_reaches_the_model_on_its_third_call() -> None:
    hold = CommandHold()
    served = [hold.update(c) for c in ["STRAIGHT", "STRAIGHT", "LEFT", "LEFT", "LEFT", "LEFT"]]
    assert served == ["STRAIGHT", "STRAIGHT", "STRAIGHT", "STRAIGHT", "LEFT", "LEFT"]
    assert hold.transitions == 1 and hold.held_back == 2


def test_a_flicker_shorter_than_the_dwell_never_reaches_the_model() -> None:
    hold = CommandHold()
    served = [hold.update(c) for c in ["STRAIGHT", "LEFT", "LEFT", "STRAIGHT", "RIGHT", "STRAIGHT"]]
    assert set(served) == {"STRAIGHT"} and hold.transitions == 0


def test_the_hold_runs_per_model_call_so_a_turn_reaches_the_model_3_s_after_it_appears() -> None:
    """Updated on /infer only, every 1.5 s: the route turns at 6.0 s, the model is told LEFT at 9.0 s,
    the native driver's behaviour, kept on purpose. /observe frames do not advance it."""
    p = _loaded()
    commands = []
    for t in range(0, 12_000_001, STEP_US):
        obs = _observation(t, _arc(0.05) if t >= 6_000_000 else _arc(0.0))
        if t >= 3_000_000 and (t - 3_000_000) % 1_500_000 == 0:
            p.infer(obs)
            commands.append((t, p.model.calls[-1]["scene"]["nav_command"]))
        else:
            p.observe(obs)
    assert commands == [
        (3_000_000, 0),
        (4_500_000, 0),
        (6_000_000, 0),
        (7_500_000, 0),
        (9_000_000, 1),
        (10_500_000, 1),
        (12_000_000, 1),
    ]
    assert p.diagnostics()["command_hold"] == {"held": "LEFT", "transitions": 1, "calls_held_back": 2}


def test_reset_clears_the_hold() -> None:
    p = _loaded()
    _drive(p, 4_500_000, route_xy=_arc(0.05))
    p.reset()
    assert p.hold.held is None and p.alpasim is None


# ---- the plan --------------------------------------------------------------


def test_alpasim_gets_all_50_points_at_0_1_s() -> None:
    plans = _drive(_loaded(), 4_500_000)
    assert len(plans) == 2
    for plan in plans:
        assert plan.dt == ALPASIM_WAYPOINT_DT == 0.1
        assert plan.waypoints.shape == (50, 3)
        assert plan.to_wire()["plan"]["dt"] == 0.1


def test_other_benchmarks_still_get_ten_points_at_0_5_s() -> None:
    p = _loaded()
    for t in range(0, 2_000_001, 100_000):
        obs = _observation(t, scene_id=None)
        p.observe(obs) if t < 2_000_000 else None
    plan = p.infer(_observation(2_100_000, scene_id=None))
    assert plan.dt == 0.5 and plan.waypoints.shape == (10, 3) and plan.extras is None


@pytest.mark.parametrize("reach_m", [0.0, 0.4, -2.0])
def test_a_stop_plan_is_sent_as_it_is_and_counted_never_raised(reach_m) -> None:
    p = _loaded(reach_m=reach_m)
    plans = _drive(p, 4_500_000)
    assert len(plans) == 2
    np.testing.assert_allclose(plans[-1].waypoints[-1, 0], reach_m)
    d = p.diagnostics()
    assert d["short_plans_total"] == 2 and len(d["short_plans_last"]) == 2
    assert d["short_plans_last"][0][2] == pytest.approx(10.0, abs=0.01)  # the speed it was planned at
    assert reach_m < STOP_REACH_M


def test_the_seed_is_the_native_per_scene_seed() -> None:
    expected = 42 + int.from_bytes(hashlib.sha256(SCENE.encode()).digest()[:8], "big") % 1_000_000
    assert scene_seed(SCENE) == expected and BASE_NOISE_SEED == 42
    p = _loaded()
    _drive(p, 4_500_000)
    assert {c["seed"] for c in p.model.calls} == {expected}
    assert scene_seed("clipgt-other") != expected


def test_each_call_reports_what_it_was_given() -> None:
    plans = _drive(_loaded(), 4_500_000, route_xy=_arc(0.0))
    block = plans[-1].extras["qwendrive"]
    assert block["route_command"] == "STRAIGHT" and block["command"] == "STRAIGHT"
    assert block["seed"] == scene_seed(SCENE) and block["history_fill"] == 0
    assert block["frame_gaps_ms"] == [0.0, 0.0, 0.0, 0.0]


# ---- refusals ----------------------------------------------------------------


def test_a_frame_off_its_slot_is_refused() -> None:
    times = np.arange(0, 1.55, 0.05)
    assert alpasim_frame_indices(times)[0] == [0, 10, 20, 30]
    gappy = np.delete(times, [9, 10, 11])  # nothing within 50 ms of now-1.0 s
    with pytest.raises(ValueError, match="observe-between-replans"):
        alpasim_frame_indices(gappy)


def test_only_replan_ticks_after_handover_are_refused_on_the_second_plan() -> None:
    """The profile without --observe-between-replans: the node hears every 1.5 s after the warm-up."""
    p = _loaded()
    for t in range(0, 3_000_001, STEP_US):
        (p.infer if t == 3_000_000 else p.observe)(_observation(t))
    with pytest.raises(ValueError, match="arrive 1500 ms apart"):
        p.infer(_observation(4_500_000))


def test_a_10_hz_stream_is_refused_on_alpasim() -> None:
    p = _policy()
    p._record(_observation(0))
    with pytest.raises(ValueError, match="100 ms apart"):
        p._record(_observation(100_000))


def test_a_missing_view_is_refused() -> None:
    p = _policy()
    obs = _observation(0)
    del obs.images["CAM_FRONT_RIGHT"]
    with pytest.raises(KeyError, match="CAM_FRONT_RIGHT"):
        p._record(obs)


def test_a_different_scene_without_reset_is_refused() -> None:
    p = _policy()
    p._record(_observation(0))
    with pytest.raises(ValueError, match="without a /reset"):
        p._record(_observation(STEP_US, scene_id="clipgt-other"))


def test_the_scene_id_crosses_the_wire() -> None:
    obs = _observation(0)
    again = Observation.from_wire(obs.to_wire(), cameras=CAMERAS)
    assert again.scene_id == SCENE
    assert "scene_id" not in _observation(0, scene_id=None).to_wire()
