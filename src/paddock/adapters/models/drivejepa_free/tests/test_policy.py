"""Contract tests for the Drive-JEPA perception-free adapter.

No navsim, no nuplan, no torch, no GPU: everything here runs in paddock's own
interpreter, which is the point -- the parts that go wrong on a cross-dataset
adapter are the crop, the frame ordering and the derived ego state, and none of
those need the model to check.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.drivejepa_free.policy import (
    CAMERA,
    COMMAND_ONE_HOT,
    HISTORY_DT,
    NAVSIM_CROP_MARGIN,
    NUM_KEYVAL,
    NUPLAN_TAN_HALF_H,
    NUPLAN_TAN_HALF_V,
    POLICY,
    POSE_DT,
    crop_box,
    cropped_intrinsics,
    relative_pose,
    to_model_image,
)
from paddock.policy.protocol import Command, Observation, Pinhole, Policy

#: The EON camera comma2k19 records with.
EON = Pinhole(fx=910.0, fy=910.0, cx=582.0, cy=437.0, width=1164, height=874)


def _observation(t_s: float, x: float = 0.0, speed: float = 20.0, yaw_rate: float = 0.0) -> Observation:
    ego2world = np.eye(4)
    ego2world[0, 3] = x
    canbus = np.zeros(18)
    canbus[13] = speed
    canbus[12] = yaw_rate
    image = np.zeros((EON.height, EON.width, 3), dtype=np.uint8)
    # A row ramp, so a crop that takes the wrong rows is visible in the values
    # rather than only in the shape.
    image[:, :, 0] = np.arange(EON.height, dtype=np.uint8)[:, None]
    return Observation(
        images={CAMERA: image},
        image_paths={},
        ego2world=ego2world,
        canbus=canbus,
        timestamp_us=int(round(t_s * 1e6)),
        command=Command.STRAIGHT,
        camera2image={CAMERA: np.array([[910.0, 0, 582.0], [0, 910.0, 437.0], [0, 0, 1.0]])},
        camera2ego={CAMERA: np.eye(4)},
    )


def _drive(policy, n: int = 40, hz: float = 20.0, speed: float = 20.0, accel: float = 0.0):
    """Feed a straight run at `hz`, returning the last observation."""
    obs = None
    for i in range(n):
        t = i / hz
        v = speed + accel * t
        obs = _observation(t, x=v * t, speed=v)
        policy._record(obs)
    return obs


# -- wiring --------------------------------------------------------------------


def test_policy_imports_without_navsim_installed() -> None:
    policy = POLICY(checkpoint="/nonexistent.ckpt", pretrain="/nonexistent.pt")
    assert isinstance(policy, Policy)
    assert policy.cameras == (CAMERA,)
    assert policy.waypoint_dt == POSE_DT


def test_load_without_weights_names_the_missing_setting() -> None:
    """Both files are required; the error must say which one is absent."""
    with pytest.raises(RuntimeError, match="DRIVEJEPA_FREE_CHECKPOINT"):
        POLICY(checkpoint=None, pretrain="/x.pt").load()
    with pytest.raises(RuntimeError, match="DRIVEJEPA_VJEPA_WEIGHTS"):
        POLICY(checkpoint="/x.ckpt", pretrain=None).load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY(checkpoint="/x.ckpt", pretrain="/x.pt").infer(object())  # type: ignore[arg-type]


def test_the_released_key_value_count_is_the_evaluation_one() -> None:
    """129, not the 257 the agent's config yaml defaults to.

    The eval script overrides it on the command line and the checkpoint only
    fits the overridden shape, so a "sensible default" here fails to load.
    """
    assert NUM_KEYVAL == 129


def test_reset_clears_the_history_and_the_cached_crop() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=5)
    policy._crop = (0, 1, 0, 1)
    policy.reset()
    assert policy.frames == []
    assert policy._crop is None
    assert policy._intrinsics is None


# -- the crop ------------------------------------------------------------------


def test_the_crop_reproduces_nuplans_field_of_view() -> None:
    """Within a degree on both axes, derived from focal length alone."""
    top, bottom, left, right = crop_box(EON)
    hfov = np.degrees(2 * np.arctan((right - left) / (2 * EON.fx)))
    vfov = np.degrees(2 * np.arctan((bottom - top) / (2 * EON.fy)))
    assert abs(hfov - np.degrees(2 * np.arctan(NUPLAN_TAN_HALF_H))) < 1.0
    assert abs(vfov - np.degrees(2 * np.arctan(NUPLAN_TAN_HALF_V))) < 1.0


def test_the_crop_keeps_the_aspect_the_resize_expects() -> None:
    """1.875, nuPlan's own post-crop aspect.

    The feature builder resizes to 512x256 unconditionally. A crop of a
    different aspect does not fail -- it stretches the image, which is a far
    bigger distribution shift than the degree of field of view it would buy.
    """
    top, bottom, left, right = crop_box(EON)
    assert abs((right - left) / (bottom - top) - 960.0 / 512.0) < 0.02


def test_the_crop_is_centred_on_the_principal_point() -> None:
    top, bottom, left, right = crop_box(EON)
    assert abs((top + bottom) / 2 - EON.cy) <= 1
    assert abs((left + right) / 2 - EON.cx) <= 1


def test_a_narrow_camera_is_clamped_not_padded() -> None:
    """Padding would widen a field of view with pixels that were never imaged."""
    narrow = Pinhole(fx=910.0, fy=910.0, cx=300.0, cy=200.0, width=600, height=400)
    top, bottom, left, right = crop_box(narrow)
    assert (top, bottom, left, right) == (0, 400, 0, 600)


def test_the_builders_own_crop_recovers_exactly_what_was_selected() -> None:
    """The 28-row pre-pad is the whole trick; without it 56 rows vanish."""
    box = crop_box(EON)
    image = _observation(0.0).image(CAMERA)
    padded = to_model_image(image, box)
    assert padded.shape[0] == (box[1] - box[0]) + 2 * NAVSIM_CROP_MARGIN
    selected = image[box[0] : box[1], box[2] : box[3]]
    np.testing.assert_array_equal(padded[NAVSIM_CROP_MARGIN:-NAVSIM_CROP_MARGIN], selected)


def test_the_intrinsics_follow_the_crop() -> None:
    box = crop_box(EON)
    k = cropped_intrinsics(EON, box)
    assert k[0, 0] == EON.fx and k[1, 1] == EON.fy
    assert k[0, 2] == EON.cx - box[2]
    assert k[1, 2] == EON.cy - box[0]


# -- history -------------------------------------------------------------------


def test_the_pair_is_half_a_second_apart_at_twenty_hertz() -> None:
    """Taking the last two requests would hand the encoder 50 ms of context."""
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=40, hz=20.0)
    older, now = policy._select_history()
    assert abs((now.timestamp - older.timestamp) / 1e6 - HISTORY_DT) < 0.03


def test_the_first_of_the_pair_is_the_older_frame() -> None:
    """`cameras[-2]` is t-0.5 s. Reversed, nothing raises and the plan moves 1 m."""
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=40, hz=20.0)
    older, now = policy._select_history()
    assert older.timestamp < now.timestamp
    assert now is policy.frames[-1]


def test_a_short_history_clamps_rather_than_extrapolating() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    policy._record(_observation(0.0))
    older, now = policy._select_history()
    assert older is now


def test_a_repeated_timestamp_is_dropped() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    policy._record(_observation(0.0))
    policy._record(_observation(0.0))
    assert len(policy.frames) == 1


def test_old_frames_are_dropped_and_their_pixels_released() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=200, hz=20.0)
    assert len(policy.frames) <= 64
    # 0.5 s at 20 Hz is ten frames; the kept window must still cover the pair.
    older, _ = policy._select_history()
    assert older in policy.frames


def test_pixels_are_only_released_where_a_path_can_produce_them_again() -> None:
    """An observation that carries pixels and no path holds the only copy.

    `paddock verify` caught this: it infers, calls `reset()`, and infers again
    over the same observation objects, and the first pass had emptied them --
    `KeyError: observation has no image for 'CAM_FRONT'`. Closed-loop sends
    pixels over the wire the same way, so this was a fault in the simulator
    path, not only in the checker.
    """
    from paddock.adapters.models.drivejepa_free.policy import Frame

    held = _observation(0.0)
    Frame(held).release()
    assert held.image(CAMERA).shape == (EON.height, EON.width, 3)

    staged = _observation(0.0)
    staged.image_paths[CAMERA] = "/staged/frame.npy"
    Frame(staged).release()
    assert CAMERA not in staged.images


# -- derived ego state ---------------------------------------------------------


def test_acceleration_comes_from_the_speed_slope_when_the_source_has_no_channel() -> None:
    """comma2k19 fills speed and yaw rate and leaves the accel slot at zero.

    A constant zero there is a different claim from "not measured", and zeroing
    the acceleration fields moved the reference's endpoint by 0.63 m.
    """
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=40, hz=20.0, speed=20.0, accel=-3.0)
    ax, _ = policy._acceleration()
    assert policy._accel_source == "speed_slope"
    assert abs(ax - (-3.0)) < 0.05


def test_a_recorded_acceleration_channel_is_preferred() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    for i in range(20):
        obs = _observation(i / 20.0, speed=20.0)
        obs.canbus[7] = 1.75
        policy._record(obs)
    ax, _ = policy._acceleration()
    assert policy._accel_source == "canbus"
    assert ax == pytest.approx(1.75)


def test_lateral_acceleration_is_the_centripetal_term_and_points_left_on_a_left_turn() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    for i in range(20):
        policy._record(_observation(i / 20.0, speed=20.0, yaw_rate=0.1))
    _, ay = policy._acceleration()
    assert ay == pytest.approx(2.0, abs=1e-5)
    policy.reset()
    for i in range(20):
        policy._record(_observation(i / 20.0, speed=20.0, yaw_rate=-0.1))
    _, ay = policy._acceleration()
    assert ay == pytest.approx(-2.0, abs=1e-5)


def test_a_standstill_accelerates_at_exactly_zero() -> None:
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    for i in range(20):
        policy._record(_observation(i / 20.0, speed=0.0))
    np.testing.assert_allclose(policy._acceleration(), [0.0, 0.0], atol=1e-9)


# -- commands and poses --------------------------------------------------------


def test_every_command_maps_to_a_one_hot() -> None:
    """A missing command KeyErrors mid-run, minutes into a sweep."""
    for command in Command:
        one_hot = COMMAND_ONE_HOT[int(command)]
        assert sum(one_hot) == 1, f"{command} is not one-hot: {one_hot}"


def test_the_one_hot_order_is_left_straight_right_unknown() -> None:
    assert COMMAND_ONE_HOT[int(Command.LEFT)] == (1, 0, 0, 0)
    assert COMMAND_ONE_HOT[int(Command.STRAIGHT)] == (0, 1, 0, 0)
    assert COMMAND_ONE_HOT[int(Command.RIGHT)] == (0, 0, 1, 0)
    assert COMMAND_ONE_HOT[int(Command.FOLLOW_REFERENCE)] == (0, 0, 0, 1)


def test_relative_pose_is_identity_against_itself() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 5.0, 0.0]
    np.testing.assert_allclose(relative_pose(pose, pose), [0.0, 0.0, 0.0], atol=1e-12)


def test_relative_pose_puts_the_past_behind_the_present() -> None:
    past, now = np.eye(4), np.eye(4)
    now[0, 3] = 10.0
    assert relative_pose(past, now)[0] < 0


# -- diagnostics ---------------------------------------------------------------


def test_diagnostics_before_the_first_frame_say_so() -> None:
    assert POLICY(checkpoint="/x.ckpt", pretrain="/x.pt").diagnostics() == {"status": "no frame seen yet"}


def test_diagnostics_report_the_history_span_against_its_target() -> None:
    """If the span collapses the encoder is reading a static scene."""
    policy = POLICY(checkpoint="/x.ckpt", pretrain="/x.pt")
    _drive(policy, n=40, hz=20.0)
    report = policy.diagnostics()
    assert report["history_target_s"] == HISTORY_DT
    assert abs(report["history_span_s"] - HISTORY_DT) < 0.03
    assert report["speed_ms"] == pytest.approx(20.0)
