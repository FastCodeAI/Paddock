"""Golden tests for the comma2k19 adapter's pose math.

Runs in paddock's own environment: no ffmpeg, no GPU, no dataset on disk. It
exercises the one part of this adapter that is easy to get quietly backwards
-- the ecef_from_device vs. device_from_ecef direction of the pose quaternion
-- with a synthetic straight-line segment where the right answer is obvious by
construction, so a refactor that flips the transform fails loudly here instead
of silently doubling everyone's ADE.
"""

from __future__ import annotations

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L


def _straight_line_pose(n: int, speed_mps: float = 20.0) -> L.SegmentPose:
    """A vehicle driving at `speed_mps` along the ECEF x-axis, facing it exactly.

    Identity orientation means device frame == ECEF frame, so every transform
    this module applies should be a no-op: ground truth in the device frame
    must equal the raw ECEF displacement.
    """
    step = speed_mps / L.SRC_HZ
    positions = np.stack([np.arange(n) * step, np.zeros(n), np.zeros(n)], axis=1)
    velocities = np.tile([speed_mps, 0.0, 0.0], (n, 1))
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))  # identity quaternion
    times = np.arange(n) / L.SRC_HZ
    return L.SegmentPose(times=times, positions=positions, velocities=velocities, orientations=orientations)


def test_quat2rot_identity_quaternion_is_identity_rotation() -> None:
    r = L._quat2rot(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)


def test_choose_keyframe_targets_the_middle_of_a_typical_segment() -> None:
    assert L.choose_keyframe(1200) == L.KEYFRAME_TARGET


def test_choose_keyframe_none_when_segment_too_short() -> None:
    assert L.choose_keyframe(50) is None


def test_frame_indices_clamp_at_the_segment_start() -> None:
    """Near the start of a segment, warm-up repeats frame 0 rather than going negative.

    Length stays `history + 1` regardless -- it is the count of repeats at the
    front that changes, matching KITScenes' `frame_indices`.
    """
    indices = L.frame_indices(keyframe=10, history=96)
    assert len(indices) == 97
    assert indices[0] == 0
    assert indices[-1] == 10
    assert indices == sorted(indices)


def test_frame_indices_full_history_when_available() -> None:
    indices = L.frame_indices(keyframe=600, history=96)
    assert indices == list(range(504, 601))


def test_vehicle_detection_from_dongle_id() -> None:
    assert L._vehicle(f"{L.RAV4_DONGLE}|2018-08-02--08-34-47") == "rav4"
    assert L._vehicle(f"{L.CIVIC_DONGLE}|2018-08-02--08-34-47") == "civic"
    assert L._vehicle("deadbeefdeadbeef|2018-08-02--08-34-47") == "unknown"


def test_ego_speed_is_the_recorded_velocity_magnitude() -> None:
    pose = _straight_line_pose(700, speed_mps=20.0)
    assert L.ego_speed(pose, 600) == 20.0


def test_ground_truth_on_a_straight_line_matches_the_raw_displacement() -> None:
    """A model that predicted this exact line would score zero ADE.

    Ground truth at 0.5 s steps for a 20 m/s straight-line drive is 10, 20, ...
    80 m ahead, 0 lateral -- if the ecef_from_device direction is flipped this
    still passes with identity orientation (the whole point of testing with
    it), so `test_ground_truth_relative_to_a_rotated_keyframe` below is the one
    that actually catches that bug.
    """
    pose = _straight_line_pose(700, speed_mps=20.0)
    keyframe = L.choose_keyframe(len(pose), n_gt=8)
    gt = L.ground_truth(pose, keyframe, n=8)

    expected_x = np.arange(1, 9) * 10.0  # 0.5s * 20 m/s = 10 m per step
    np.testing.assert_allclose(gt[:, 0], expected_x)
    np.testing.assert_allclose(gt[:, 1], np.zeros(8), atol=1e-9)
    np.testing.assert_allclose(gt[:, 2], np.zeros(8), atol=1e-9)


def test_ground_truth_relative_to_a_rotated_keyframe() -> None:
    """The keyframe facing +90 deg (yawed into ECEF +y) must see its own future as 'forward'.

    Positions still advance along ECEF x (unchanged from the straight-line
    fixture), but orientation at every frame is rotated 90 deg so the device's
    forward axis points along ECEF y. Getting ecef_from_device backwards flips
    this into a hard left/right sign error instead of leaving it as forward
    motion, unlike the identity-orientation case above.
    """
    n = 700
    pose = _straight_line_pose(n, speed_mps=20.0)
    # Rotate device frame 90 deg about the down (z) axis, at every frame:
    # forward (ECEF x) -> right becomes ECEF y, i.e. the quaternion below is
    # ecef_from_device with device-x mapped to ecef-y.
    half = np.pi / 4
    pose.orientations = np.tile([np.cos(half), 0.0, 0.0, np.sin(half)], (n, 1))

    keyframe = L.choose_keyframe(len(pose), n_gt=8)
    gt = L.ground_truth(pose, keyframe, n=8)

    # The vehicle still only moves along ECEF +x, and forward is now ECEF +y,
    # so its own future position is entirely off to one side, not ahead -- and
    # on a specific side: swapping ecef_from_device for its transpose (the bug
    # this test exists to catch) would flip this exact sign, not just move it
    # off-axis, since a +/-90 deg rotation is its own inverse up to sign.
    expected = np.arange(1, 9) * 10.0  # 0.5s * 20 m/s = 10 m per step
    np.testing.assert_allclose(gt[:, 0], np.zeros(8), atol=1e-9)
    np.testing.assert_allclose(gt[:, 1], expected)
