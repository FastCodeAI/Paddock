"""Contract tests for the Qwen-Drive adapter. No torch, no qwen_drive, no weights.

Everything here is about the two things that would make a Qwen-Drive number
wrong without making it fail: a frame-convention error, which still produces
smooth mirrored trajectories, and a history that does not describe the motion
the poses describe, which the reference's ablation showed is this model's most
load-bearing input by a wide margin (a stationary history scored ADE 70.74 on a
highway segment against a 2.00 baseline).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from paddock.adapters.models.qwendrive.policy import (
    CAMERA_VIEWS,
    COMMANDS,
    HISTORY_DT,
    HISTORY_KEEP,
    NUM_HISTORY_STEPS,
    POLICY,
    VIEW_OFFSETS_S,
    interp_angle,
    smooth,
    synthesize_views,
    wrap_angle,
)
from paddock.policy.protocol import Command, Observation, Policy

SRC_HZ = 20.0
#: Small enough that a whole test sequence of them is cheap, large enough that
#: the `crop` boxes are distinguishable.
_IMAGE = np.zeros((16, 40, 3), dtype=np.uint8)


def _observation(t_s: float, x: float, y: float, yaw: float, speed: float, yaw_rate: float = 0.0):
    pose = np.eye(4)
    pose[0, 3], pose[1, 3] = x, y
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    canbus = np.zeros(18)
    canbus[13], canbus[12] = speed, yaw_rate
    return Observation(
        images={"CAM_FRONT": _IMAGE.copy()},
        ego2world=pose,
        canbus=canbus,
        timestamp_us=int(round(t_s * 1e6)),
        command=Command.STRAIGHT,
        camera2image={"CAM_FRONT": np.eye(3)},
        camera2ego={"CAM_FRONT": np.eye(4)},
    )


def _drive(policy, n: int = 60, speed: float = 20.0, omega: float = 0.0):
    """Feed `n` frames at 20 Hz along an arc, and return the last observation.

    `omega` is the yaw rate in rad/s, positive for a left turn -- paddock's ego
    frame is x forward, y left, so that is the direction the sign convention has
    to survive.
    """
    dt = 1.0 / SRC_HZ
    x = y = yaw = 0.0
    last = None
    for i in range(n):
        last = _observation(i * dt, x, y, yaw, speed, omega)
        policy.observe(last)
        x += speed * np.cos(yaw) * dt
        y += speed * np.sin(yaw) * dt
        yaw += omega * dt
    return last


def _history(policy):
    """(poses, velocity, acceleration) exactly as `_scene` would build them."""
    offsets, wanted = policy._grid()
    local = policy._poses_in_current_frame(wanted)
    velocity, acceleration = policy._ego_series(local, wanted, -offsets)
    return local, velocity, acceleration


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_policy_imports_without_torch() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.cameras == ("CAM_FRONT",)


def test_the_waypoint_rate_is_the_model_s_own() -> None:
    """50 waypoints at 10 Hz. Declaring 0.5 here would stretch a 5 s plan to 25 s.

    Nothing raises: the harness pairs prediction and ground truth by time, so a
    wrong `dt` scores a correct plan against the wrong part of the future and
    the error looks like a model that cannot drive.
    """
    assert POLICY().waypoint_dt == pytest.approx(0.1)


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(object())  # type: ignore[arg-type]


def test_an_unknown_view_mode_is_refused_at_construction(monkeypatch) -> None:
    """Not at the first frame, 40 minutes into a sweep."""
    monkeypatch.setenv("QWENDRIVE_VIEWS", "panorama")
    with pytest.raises(ValueError, match="panorama"):
        POLICY()


def test_reset_clears_every_cross_sample_channel() -> None:
    """comma2k19 scores 10,165 independent samples through one node.

    Any of these surviving would make sample N's history start with sample
    N-1's ending, and `infer_sequence` would stop being a self-contained query.
    """
    policy = POLICY()
    _drive(policy, n=5)
    policy.reset()
    assert policy.timestamps == []
    assert policy.poses == []
    assert policy.speeds == []
    assert policy.yaw_rates == []
    assert policy.frames == []


def test_the_seed_is_ours_not_the_checkpoint_s(monkeypatch) -> None:
    """`stochastic: false` in model.yaml is a claim about this file.

    The flow sampler falls back to the checkpoint's `noise_seed` when none is
    passed, so relying on that would make a recorded number depend on a config
    value nothing in paddock pins.
    """
    assert POLICY().seed == 42
    monkeypatch.setenv("QWENDRIVE_SEED", "7")
    assert POLICY().seed == 7


# ---------------------------------------------------------------------------
# the history the expert reads
# ---------------------------------------------------------------------------


def test_history_is_sixteen_poses_ending_at_the_origin() -> None:
    """The prompt asserts the last pose is (0, 0, 0). It has to be true."""
    policy = POLICY()
    _drive(policy)
    local, velocity, acceleration = _history(policy)
    assert local.shape == (NUM_HISTORY_STEPS, 3)
    assert velocity.shape == (NUM_HISTORY_STEPS, 2)
    assert acceleration.shape == (NUM_HISTORY_STEPS, 2)
    assert local[-1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_history_runs_oldest_first_and_spans_one_and_a_half_seconds() -> None:
    """Reversed, the model is told the car is driving backwards at 20 m/s."""
    policy = POLICY()
    _drive(policy, speed=20.0)
    local, _, _ = _history(policy)
    assert np.all(np.diff(local[:, 0]) > 0), "x must increase towards the present"
    assert local[0, 0] == pytest.approx(-20.0 * 1.5, abs=0.05)


def test_the_history_grid_is_ten_hertz_whatever_the_source_runs_at() -> None:
    """comma2k19 feeds 20 Hz. Taking the last 16 requests would span 0.8 s.

    The model differences exactly this grid to recover its own speed, so a
    history spanning 0.8 s labelled 1.5 s reads as a car going half speed --
    confident, smooth and wrong.
    """
    policy = POLICY()
    _drive(policy, speed=10.0)
    local, _, _ = _history(policy)
    steps = np.hypot(np.diff(local[:, 0]), np.diff(local[:, 1]))
    assert steps == pytest.approx(10.0 * HISTORY_DT, abs=0.01)


def test_a_left_turn_is_left_in_every_channel() -> None:
    """The one sign that a mirrored adapter would still pass every other test.

    Paddock's ego frame and Qwen-Drive's are both x forward, y left, so nothing
    in this policy flips anything -- comma2k19's y-right device frame is
    converted in the loader, before the observation exists. If a flip were
    reintroduced here the plans would stay smooth and score roughly twice the
    lateral error on every curve.

    On a left-turning arc the car came from behind and to the *right* of where
    it now points, so the recorded history sits at positive y with a negative
    (clockwise-of-now) heading, and the lateral acceleration points left.
    """
    policy = POLICY()
    _drive(policy, speed=15.0, omega=+0.2)
    local, _, acceleration = _history(policy)
    assert local[0, 1] > 0.05, "history of a left turn lies to the left of the ego x axis"
    assert local[0, 2] < -0.05, "the car was pointing right of its current heading"
    assert acceleration[-1, 1] > 0.0, "centripetal acceleration in a left turn points left"

    right = POLICY()
    _drive(right, speed=15.0, omega=-0.2)
    mirrored, _, right_accel = _history(right)
    assert mirrored[0, 1] < -0.05
    assert mirrored[0, 2] > 0.05
    assert right_accel[-1, 1] < 0.0


def test_velocity_comes_from_the_odometer_not_a_difference() -> None:
    """A one-sided difference at t=0 lags; the recorded speed does not.

    t=0 is the sample the expert conditions on most strongly and the only one a
    policy cannot centre a difference on, because the frames after it have not
    happened.
    """
    policy = POLICY()
    _drive(policy, speed=27.5)
    _, velocity, _ = _history(policy)
    assert velocity[-1, 0] == pytest.approx(27.5, abs=1e-6)
    assert velocity[-1, 1] == pytest.approx(0.0, abs=0.01), "no sideslip on a straight line"
    assert policy._speed_source == "recorded"


def test_a_source_with_no_odometer_falls_back_to_the_poses() -> None:
    """And says so, rather than reporting a moving car as parked.

    comma2k19 fills the speed slot from CAN; a benchmark that does not would
    otherwise hand this model the one input the reference measured as
    catastrophic to get wrong.
    """
    policy = POLICY()
    dt = 1.0 / SRC_HZ
    for i in range(60):
        policy.observe(_observation(i * dt, 18.0 * i * dt, 0.0, 0.0, speed=0.0))
    _, velocity, _ = _history(policy)
    assert policy._speed_source == "pose"
    assert velocity[-1, 0] == pytest.approx(18.0, abs=0.2)


def test_braking_shows_up_as_negative_longitudinal_acceleration() -> None:
    policy = POLICY()
    dt, x, speed = 1.0 / SRC_HZ, 0.0, 25.0
    for i in range(80):
        policy.observe(_observation(i * dt, x, 0.0, 0.0, speed))
        x += speed * dt
        speed = max(0.0, speed - 3.0 * dt)
    _, _, acceleration = _history(policy)
    assert acceleration[-1, 0] == pytest.approx(-3.0, abs=0.3)


def test_a_standstill_history_is_zero_rather_than_noise() -> None:
    """The failure mode the reference's ablation named: it plans to stay put."""
    policy = POLICY()
    for i in range(60):
        policy.observe(_observation(i / SRC_HZ, 0.0, 0.0, 0.0, 0.0))
    local, velocity, acceleration = _history(policy)
    assert np.allclose(local, 0.0, atol=1e-9)
    assert np.allclose(velocity, 0.0, atol=1e-9)
    assert np.allclose(acceleration, 0.0, atol=1e-9)


def test_a_short_history_is_clipped_not_extrapolated() -> None:
    """A sequence that starts 0.4 s ago must not invent 1.1 s of travel."""
    policy = POLICY()
    _drive(policy, n=9, speed=20.0)  # 0.4 s at 20 Hz
    local, _, _ = _history(policy)
    assert local[0, 0] == pytest.approx(-20.0 * 0.4, abs=0.05)
    assert local[-1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_a_single_frame_still_produces_a_usable_scene() -> None:
    """Closed-loop hands over one frame with no past. It must not raise."""
    policy = POLICY()
    policy.observe(_observation(0.0, 0.0, 0.0, 0.0, 12.0))
    local, velocity, acceleration = _history(policy)
    assert np.allclose(local, 0.0)
    assert velocity[-1, 0] == pytest.approx(12.0)
    assert np.all(np.isfinite(acceleration))


def test_a_repeated_timestamp_is_dropped() -> None:
    """np.interp needs a strictly increasing x, and a warm-up that clamps at the
    start of a sequence hands the same frame over several times."""
    policy = POLICY()
    policy.observe(_observation(0.0, 0.0, 0.0, 0.0, 10.0))
    policy.observe(_observation(0.0, 5.0, 0.0, 0.0, 10.0))
    policy.observe(_observation(-1.0, 9.0, 0.0, 0.0, 10.0))
    assert len(policy.timestamps) == 1


def test_old_frames_are_dropped_and_their_pixels_released() -> None:
    """160 observations at 3 MB each is half a gigabyte of images nobody reads."""
    policy = POLICY()
    first = _observation(0.0, 0.0, 0.0, 0.0, 10.0)
    policy.observe(first)
    first.image()  # force the decode cache
    _drive(policy, n=HISTORY_KEEP + 20)
    assert len(policy.frames) == HISTORY_KEEP
    assert len(policy.timestamps) == HISTORY_KEEP
    assert "CAM_FRONT" not in first.images


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def test_replicate_fills_all_three_slots_with_the_true_forward_frame() -> None:
    """DrivingScene raises on a missing view, so all three have to be filled.

    Measured rather than assumed: replicate scored 1.47/2.00 ADE against crop's
    1.78/2.13. Each replicated view really is a forward camera, where a crop is
    not a rotated one, and the model's learned cross-view extrinsics notice.
    """
    image = Image.fromarray(_IMAGE)
    views = synthesize_views(image, "replicate")
    assert set(views) == set(CAMERA_VIEWS)
    assert all(v is image for v in views.values())


def test_crop_gives_three_overlapping_thirds_of_the_same_width() -> None:
    views = synthesize_views(Image.fromarray(np.zeros((16, 100, 3), np.uint8)), "crop")
    assert set(views) == set(CAMERA_VIEWS)
    assert {v.size for v in views.values()} == {(55, 16)}


def test_frames_are_decoded_only_when_the_processor_asks() -> None:
    """comma2k19 sends 160 frames and the model looks at four.

    Those frames are PNGs and the decode is the dominant cost on this
    benchmark; decoding on record would spend it on frames read for their pose
    alone.
    """
    policy = POLICY()
    _drive(policy, n=40)
    assert all(frame._views is None for frame in policy.frames)
    loader = policy.frames[-1].loader(CAMERA_VIEWS[0])
    assert policy.frames[-1]._views is None, "building the loader must not decode"
    assert loader().size == (40, 16)


def test_view_frames_are_chosen_by_timestamp_not_by_index() -> None:
    """comma2k19's camera runs at 19.93 Hz, not 20.

    Index arithmetic drifts about 0.2 s over a one-minute segment, which is
    comparable to the 0.5 s spacing the prompt tells the model these frames
    have.
    """
    policy = POLICY()
    dt = 1.0 / 19.93
    for i in range(60):
        policy.observe(_observation(i * dt, 20.0 * i * dt, 0.0, 0.0, 20.0))
    frames, ages = policy._view_frames()
    assert len(frames) == len(VIEW_OFFSETS_S)
    assert ages == pytest.approx(list(VIEW_OFFSETS_S), abs=dt)
    assert ages[0] > ages[-1], "oldest first, matching the t-1.5s .. t-0s labels"


# ---------------------------------------------------------------------------
# the navigation command
# ---------------------------------------------------------------------------


def test_the_command_mapping_matches_the_model_s_vocabulary() -> None:
    """Cross-checked against the released Waymo scene files, where a left turn
    has future y = +11.98 m and a right turn y = -11.0 m.

    The one-hot is [left, straight, right, unknown] and the index selects from
    NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT") -- two different
    orderings of the same three things, which is exactly the kind of pair that
    gets crossed.
    """
    assert COMMANDS[Command.STRAIGHT] == (0, (0.0, 1.0, 0.0, 0.0))
    assert COMMANDS[Command.LEFT] == (1, (1.0, 0.0, 0.0, 0.0))
    assert COMMANDS[Command.RIGHT] == (2, (0.0, 0.0, 1.0, 0.0))


def test_every_command_a_benchmark_can_send_is_mapped() -> None:
    """A KeyError here would be a crash mid-sweep on the first junction."""
    assert set(COMMANDS) == {int(c) for c in Command}


def test_follow_reference_reads_as_go_straight() -> None:
    """The default for a source with no route, which is what comma2k19 is."""
    assert COMMANDS[Command.FOLLOW_REFERENCE] == COMMANDS[Command.STRAIGHT]


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------


def test_angles_interpolate_through_the_wrap() -> None:
    """Linear interpolation of the angle itself turns a crossing of +-pi into
    most of a lap, which on a 10 Hz history reads as a spin that never happened."""
    source = np.array([0.0, 1.0])
    angle = np.array([3.0, -3.0])  # 0.28 rad apart across the wrap
    out = interp_angle(np.array([0.5]), source, angle)
    assert abs(wrap_angle(out - np.pi))[0] < 0.01


def test_smoothing_preserves_a_ramp_and_its_endpoints() -> None:
    """The edges continue the slope rather than tapering to the last value.

    Edge replication -- the obvious padding -- cost a third of a steady
    deceleration at t=0, which is the sample the expert plans from. This is the
    test that caught it.
    """
    ramp = np.arange(10, dtype=float)
    assert smooth(ramp, 3) == pytest.approx(ramp)
    assert smooth(-2.5 * ramp, 3) == pytest.approx(-2.5 * ramp)


def test_diagnostics_before_the_first_frame_do_not_raise() -> None:
    diag = POLICY().diagnostics()
    assert diag["status"] == "no frame seen yet"
    assert diag["view_mode"] == "replicate"


def test_diagnostics_expose_the_pose_speed_beside_the_recorded_one() -> None:
    """The reference's own conversion check, kept as a live one.

    A disagreement here means the pose frame, the timestamps and the odometer
    are not describing the same motion -- and every one of those errors
    produces a smooth, plausible, wrong trajectory.
    """
    policy = POLICY()
    _drive(policy, speed=22.0)
    policy._scene  # noqa: B018 - the scene needs qwen_drive; diagnostics do not
    diag = policy.diagnostics()
    assert diag["speed_recorded"] == pytest.approx(22.0)
    assert diag["speed_pose"] == pytest.approx(22.0, rel=0.02)
    assert diag["history_span_s"] == pytest.approx(59 / SRC_HZ, abs=0.01)
    assert diag["nav_command"] == "GO STRAIGHT"
