"""inputs.py: which frame fills each slot, the crops, the 0.1 s history and the route sentence.

numpy and PIL only, so these run in paddock's own environment and in CI.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock.adapters.benchmarks.navsim.convert import CAMERA_SLOTS, to_navsim_poses
from paddock.adapters.models.alpamayo2edge import inputs
from paddock.policy.protocol import Command

#: NAVSIM's four frames on the 2 Hz path, seconds before now.
NAVSIM_TIMES = (-1.5, -1.0, -0.5, 0.0)


def _yaw_pose(x: float, y: float, yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:2, 3] = [x, y]
    return pose


def _straight(times, speed: float, accel: float = 0.0):
    """Poses and world velocities of a car on the x axis: x(t) = v t + a t^2 / 2, v(t) = v + a t."""
    times = np.asarray(times, dtype=np.float64)
    poses = np.stack([_yaw_pose(speed * t + 0.5 * accel * t * t, 0.0, 0.0) for t in times])
    velocities = np.stack([[speed + accel * t, 0.0, 0.0] for t in times])
    return poses, velocities


# -- slots and frames -----------------------------------------------------------------


def test_the_slots_are_the_vendors_six_cameras_and_22_frames() -> None:
    counts = {camera_id: len(inputs.offsets_for(camera_id)) for camera_id, _, _ in inputs.SLOTS}
    # CAMERA_FRAME_COUNTS in the vendor's sample.py; tests/test_vendor_inputs.py checks it there.
    assert counts == {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
    assert sum(counts.values()) == 22
    assert {camera for _, camera, _ in inputs.SLOTS} == set(inputs.CAMERAS)
    assert inputs.CAMERAS[0] == "CAM_FRONT"


def test_every_camera_is_one_navsim_sends() -> None:
    assert set(inputs.CAMERAS) <= set(CAMERA_SLOTS)


def test_at_2hz_each_slot_gets_the_newest_frame_at_or_before_its_time() -> None:
    assert inputs.select_frames(NAVSIM_TIMES, inputs.FRONT_OFFSETS_S) == [0, 1, 2, 2, 3]
    assert inputs.select_frames(NAVSIM_TIMES, inputs.OTHER_OFFSETS_S) == [1, 2, 3]


def test_distinct_takes_the_most_recent_frames_and_repeats_the_oldest() -> None:
    assert inputs.select_frames(NAVSIM_TIMES, inputs.FRONT_OFFSETS_S, "distinct") == [0, 0, 1, 2, 3]
    assert inputs.select_frames(NAVSIM_TIMES, inputs.OTHER_OFFSETS_S, "distinct") == [1, 2, 3]


def test_at_10hz_the_native_frames_are_chosen_despite_jitter() -> None:
    rng = np.random.default_rng(0)
    times = np.round(np.arange(-2.5, 0.0001, 0.1), 6) + rng.uniform(-0.005, 0.005, 26)
    times[-1] = 0.0
    picked = [times[i] for i in inputs.select_frames(times, inputs.FRONT_OFFSETS_S)]
    np.testing.assert_allclose(picked, inputs.FRONT_OFFSETS_S, atol=0.006)


def test_a_camera_seen_only_on_keyframes_picks_among_its_own_frames() -> None:
    """keyframe_cameras: the rear views exist at 2 Hz inside a 10 Hz sequence, and
    "now" is the newest observation, not the newest frame of that camera."""
    assert inputs.select_frames((-1.5, -1.0, -0.5), inputs.OTHER_OFFSETS_S, now_s=0.0) == [1, 2, 2]


def test_an_unknown_frame_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="nearest"):
        inputs.select_frames(NAVSIM_TIMES, inputs.OTHER_OFFSETS_S, "grid")


# -- crops ------------------------------------------------------------------------------

#: NAVSIM's cameras, every one of the eight: 1920x1080, 64 deg wide, the principal
#: point 20 px below the centre (navtest log pickles, `cam_intrinsic`).
FX, CX, CY = 1545.0, 960.0, 560.0


def test_crops_are_16_by_9_and_centred_on_the_principal_point() -> None:
    for hfov in (None, 30.0):
        left, top, right, bottom = inputs.crop_box(1920, 1080, FX, CX, CY, hfov)
        assert abs((right - left) / (bottom - top) - 16 / 9) < 0.01
        assert abs((left + right) / 2 - CX) <= 0.5 and abs((top + bottom) / 2 - CY) <= 0.5


def test_the_telephoto_crop_is_30_degrees_and_the_wide_one_all_the_sensor_allows() -> None:
    """The wide window is held to 62 deg of the sensor's 64 by the 519 px between the
    principal point and the bottom edge: a 16:9 window centred there is 1845 px wide."""
    assert inputs.crop_hfov_deg(1920, 1080, FX, CX, CY, 30.0) == pytest.approx(30.0, abs=0.1)
    assert inputs.crop_box(1920, 1080, FX, CX, CY) == (38, 41, 1883, 1079)
    assert inputs.crop_hfov_deg(1920, 1080, FX, CX, CY, None) == pytest.approx(61.7, abs=0.1)


def test_a_principal_point_at_the_edge_cannot_be_cropped_around() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        inputs.crop_box(1920, 1080, FX, 0.0, CY)


def test_every_frame_comes_out_at_the_vendors_camera_size() -> None:
    image = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for hfov in (None, 30.0):
        warped = inputs.warp(image, FX, CX, CY, hfov)
        assert warped.shape == (1080, 1920, 3) and warped.dtype == np.uint8


# -- history ----------------------------------------------------------------------------


def _steps(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)


@pytest.mark.parametrize("mode", inputs.HISTORY_MODES)
def test_a_steady_car_at_2hz_gives_20_poses_at_its_speed(mode: str) -> None:
    poses, velocities = _straight(NAVSIM_TIMES, 8.0)
    xyz, rot = inputs.ego_history(NAVSIM_TIMES, poses, velocities, mode)
    assert xyz.shape == (20, 3) and rot.shape == (20, 3, 3)
    np.testing.assert_allclose(xyz[-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(rot[-1], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(_steps(xyz), 0.8, atol=1e-9)
    np.testing.assert_allclose(xyz[0], [-15.2, 0.0, 0.0], atol=1e-9)


def test_hermite_reconstructs_steady_braking_and_linear_lags_it() -> None:
    """Braking at 2 m/s^2 from 8 m/s: the true last 0.1 s step is 0.81 m. Straight
    lines between NAVSIM's frames give it the average of the last 0.5 s, 0.85 m,
    i.e. the speed of 0.25 s ago; the cubic through the measured velocities is exact."""
    poses, velocities = _straight(NAVSIM_TIMES, 8.0, accel=-2.0)
    truth_t = -0.1 * np.arange(19, -1, -1)
    truth = 8.0 * truth_t - truth_t**2

    cubic, _ = inputs.ego_history(NAVSIM_TIMES, poses, velocities, "hermite")
    covered = truth_t >= -1.5
    np.testing.assert_allclose(cubic[covered, 0], truth[covered], atol=1e-9)
    assert _steps(cubic)[-1] == pytest.approx(0.81, abs=1e-9)

    straight, _ = inputs.ego_history(NAVSIM_TIMES, poses, velocities, "linear")
    assert _steps(straight)[-1] == pytest.approx(0.85, abs=1e-9)


@pytest.mark.parametrize("mode", inputs.HISTORY_MODES)
def test_the_time_before_navsims_first_frame_is_continued_not_parked(mode: str) -> None:
    """-1.9 to -1.5 s has no frame. Repeating the first pose would write four parked
    steps into the history tokens; it is continued at the first frame's velocity."""
    poses, velocities = _straight(NAVSIM_TIMES, 8.0, accel=-2.0)
    xyz, _ = inputs.ego_history(NAVSIM_TIMES, poses, velocities, mode)
    first_speed = 8.0 + 2.0 * 1.5 if mode == "hermite" else (poses[1, 0, 3] - poses[0, 0, 3]) / 0.5
    np.testing.assert_allclose(_steps(xyz)[:4], first_speed * 0.1, atol=1e-9)
    assert inputs.history_before_first_frame_s(NAVSIM_TIMES) == pytest.approx(0.4)
    assert inputs.history_before_first_frame_s(np.arange(-2.0, 0.01, 0.1)) == 0.0


@pytest.mark.parametrize("mode", inputs.HISTORY_MODES)
def test_a_steady_turn_turns_at_a_steady_rate(mode: str) -> None:
    rate, speed = 0.2, 6.0  # rad/s, m/s: a circle of radius 30 m

    def on_circle(t: float) -> np.ndarray:
        radius = speed / rate
        return _yaw_pose(radius * math.sin(rate * t), radius * (1 - math.cos(rate * t)), rate * t)

    poses = np.stack([on_circle(t) for t in NAVSIM_TIMES])
    velocities = np.stack([poses[i, :3, :3] @ [speed, 0.0, 0.0] for i in range(4)])
    _, rot = inputs.ego_history(NAVSIM_TIMES, poses, velocities, mode)
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    np.testing.assert_allclose(yaw, rate * (-0.1 * np.arange(19, -1, -1)), atol=1e-9)


def test_full_rotations_stay_rotations() -> None:
    rng = np.random.default_rng(2)
    poses = np.stack([np.eye(4)] * 4)
    for i in range(4):
        poses[i, :3, :3] = inputs.rotation_exp(rng.normal(scale=0.1, size=3))
        poses[i, :3, 3] = rng.normal(size=3)
    _, rot = inputs.ego_history(NAVSIM_TIMES, poses, None, "hermite")
    identity = np.broadcast_to(np.eye(3), rot.shape)
    np.testing.assert_allclose(rot @ np.transpose(rot, (0, 2, 1)), identity, atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_rotation_log_and_exp_are_inverse() -> None:
    for vector in (np.array([0.3, -0.2, 0.1]), np.array([1e-12, 0.0, 0.0]), np.array([0.0, 0.0, 3.0])):
        np.testing.assert_allclose(inputs.rotation_log(inputs.rotation_exp(vector)), vector, atol=1e-9)


def test_one_frame_alone_is_continued_at_its_velocity_or_held() -> None:
    pose, velocity = _straight([0.0], 5.0)
    moving, _ = inputs.ego_history([0.0], pose, velocity)
    np.testing.assert_allclose(_steps(moving), 0.5, atol=1e-12)
    held, _ = inputs.ego_history([0.0], pose, None)
    np.testing.assert_allclose(held, 0.0, atol=1e-12)


def test_times_that_do_not_increase_are_refused() -> None:
    poses, velocities = _straight([0.0, 0.0], 5.0)
    with pytest.raises(ValueError, match="increase"):
        inputs.ego_history([0.0, 0.0], poses, velocities)


# -- route ------------------------------------------------------------------------------


def test_the_driving_command_becomes_edges_route_sentence() -> None:
    assert inputs.nav_text(Command.LEFT) == "Turn left"
    assert inputs.nav_text(Command.RIGHT) == "Turn right"
    assert inputs.nav_text(Command.STRAIGHT) == "Continue straight"
    assert inputs.nav_text(Command.FOLLOW_REFERENCE) == "Continue straight"
    assert inputs.nav_text(Command.LEFT, nominal_m=20) == "Turn left in 20m"
    assert inputs.nav_text(Command.STRAIGHT, nominal_m=20) == "Continue straight"


def test_a_sources_own_next_maneuver_wins_over_the_command() -> None:
    assert inputs.nav_text(Command.LEFT, "TURN_RIGHT", 31.6) == "Turn right in 32m"
    assert inputs.nav_text(Command.LEFT, "STRAIGHT", 31.6) == "Continue straight"
    assert inputs.nav_text(Command.STRAIGHT, "TURN_LEFT", None, nominal_m=15) == "Turn left in 15m"


# -- output -----------------------------------------------------------------------------


def _edge_plan(speed: float = 8.0, rate: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """64 poses 0.1 s apart on a steady left curve, the shape Edge returns."""
    t = inputs.PLAN_DT_S * np.arange(1, inputs.PLAN_STEPS + 1)
    radius = speed / rate
    xyz = np.column_stack([radius * np.sin(rate * t), radius * (1 - np.cos(rate * t)), np.zeros_like(t)])
    rot = np.stack([_yaw_pose(0.0, 0.0, rate * s)[:3, :3] for s in t])
    return xyz, rot


def test_the_plan_keeps_all_64_poses_and_takes_yaw_from_the_rotations() -> None:
    xyz, rot = _edge_plan()
    plan = inputs.plan_from_poses(xyz, rot)
    assert plan.shape == (64, 3)
    np.testing.assert_allclose(plan[:, 2], 0.1 * inputs.PLAN_DT_S * np.arange(1, 65), atol=1e-12)


def test_navsim_reads_the_plan_at_its_eight_half_second_poses() -> None:
    plan = inputs.plan_from_poses(*_edge_plan())
    scored = to_navsim_poses(plan, inputs.PLAN_DT_S)
    np.testing.assert_allclose(scored, plan[4:40:5], atol=1e-9)


def test_the_medoid_is_the_draw_nearest_the_others() -> None:
    draws = np.zeros((3, 4, 2))
    draws[1] += 0.1
    draws[2] += 5.0
    assert inputs.medoid_index(draws) == 1
    assert inputs.medoid_index(draws[:1]) == 0
