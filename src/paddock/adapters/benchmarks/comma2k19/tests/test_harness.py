"""Tests for the comma2k19 adapter.

Two kinds, deliberately:

* **Synthetic** tests build a fixture whose right answer is known by
  construction, so they run anywhere and fail loudly on a regression.
* **Real-data** tests are skipped when the dataset is absent. They exist
  because the defects this adapter was written to fix are all of the kind that
  a synthetic identity fixture cannot see -- a mirrored rotation and a 90 degree
  frame slip both look fine when there is no rotation to get wrong.

The `camera2ego` round trip is the single most important test here. The
removed implementation's one-line defect (`CAMERA2EGO = np.eye(4)`) would fail
it, and nothing else in the pipeline would have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import calibration as CAL
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.adapters.benchmarks.comma2k19.tests.synthetic import (
    reset_pose_cache,
    rot2quat,
    straight_pose,
    write_segment,
)

# --------------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------------


def test_rot_euler_round_trip(toolkit):
    for rpy in ([0, 0, 0], [0.0, 0.06, 0.02], [0.01, -0.1, 0.05]):
        got = L.euler_from_rot(L.rot_from_euler(rpy))
        assert np.allclose(got, rpy, atol=1e-12), (rpy, got)


def test_rot_from_euler_is_a_proper_rotation(toolkit):
    r = L.rot_from_euler([0.02, -0.07, 0.11])
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_quat2rot_matches_a_known_rotation(toolkit):
    # 90 deg about z: forward -> left in a right-handed frame.
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    r = L._quat2rot(q)
    assert np.allclose(r @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_quat2rot_matches_openpilot(toolkit):
    """openpilot's converter, if it is importable in this environment."""
    op = pytest.importorskip("openpilot.common.transformations.orientation")
    rng = np.random.default_rng(0)
    q = rng.standard_normal((8, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    assert np.allclose(L._quat2rot(q), np.asarray(op.rot_from_quat(q)), atol=1e-12)


def test_the_toolkit_rotations_match_an_independent_reference(toolkit):
    """Check the *toolkit*, not paddock's agreement with itself.

    Every other rotation test here now passes trivially if the toolkit is
    wrong, because paddock delegates to it. This one writes the algebra out
    again from the textbook definitions and asserts the toolkit reproduces it,
    which is the check that survives delegation -- and the one that has to stand
    in for the pull-request run these tests no longer get (see conftest.py).
    """
    rng = np.random.default_rng(11)

    def reference_rot_from_quat(w, x, y, z):
        """Hamilton [w,x,y,z] -> R, written from the standard expansion."""
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def reference_rot_from_euler(roll, pitch, yaw):
        """Rz(yaw) @ Ry(pitch) @ Rx(roll) -- the order openpilot's calibration uses."""
        cr, sr, cp, sp, cy, sy = (
            np.cos(roll),
            np.sin(roll),
            np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw),
            np.sin(yaw),
        )
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    for _ in range(200):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        assert np.allclose(toolkit.rot_from_quat(q), reference_rot_from_quat(*q), atol=1e-12)

        rpy = rng.uniform(-np.pi / 3, np.pi / 3, 3)
        assert np.allclose(toolkit.rot_from_euler(rpy), reference_rot_from_euler(*rpy), atol=1e-12)

    # And paddock's one remaining local piece inverts the toolkit's forward map
    # -- `toolkit.py` explains why `euler_from_rot` could not be delegated.
    for _ in range(200):
        rpy = rng.uniform(-1.0, 1.0, 3)
        assert np.allclose(L.euler_from_rot(toolkit.rot_from_euler(rpy)), rpy, atol=1e-12)


def test_the_camera_model_paddock_holds_locally_still_matches_the_toolkit(toolkit):
    """The EON geometry is duplicated on purpose, so it needs a drift check.

    `EON_WIDTH`/`EON_HEIGHT_PX`/`EON_FOCAL` are literals in `loader.py` so that
    importing the module needs no site config -- `paddock list` runs on a
    machine with no clone. That is a second source of truth, and this is what
    keeps it honest.
    """
    assert np.array_equal(L.EON_INTRINSICS, toolkit.eon_intrinsics)
    assert L.EON_FOCAL == toolkit.eon_intrinsics[0, 0] == toolkit.eon_intrinsics[1, 1]
    assert L.EON_WIDTH == 2 * toolkit.eon_intrinsics[0, 2]
    assert L.EON_HEIGHT_PX == 2 * toolkit.eon_intrinsics[1, 2]


def test_loading_the_toolkit_leaves_sys_path_alone(toolkit):
    """The clone must not stay importable after the import that needed it.

    Its root contains a directory called `openpilot` -- a partial vendored copy
    of comma's stack, not the model paddock evaluates. While the root sat on
    `sys.path`, `import openpilot` found the stub, and openpilot's own adapter
    tests stopped skipping and started failing against it. Removing the entry
    after the import is the fix; this is what keeps it removed.
    """
    import sys

    before = list(sys.path)
    TK._load.cache_clear()  # the fixture already loaded it; make the import happen again
    TK.load()
    assert sys.path == before, "the toolkit root must not outlive its own import"
    assert str(TK.root()) not in sys.path


def test_the_toolkit_is_pinned_in_repos_yaml():
    """A floating toolkit is a floating score. Runs without a clone."""
    from paddock.registry import load_repos

    entry = load_repos().get(TK.REPO)
    assert entry is not None, "the toolkit is a hard requirement; repos.yaml must list it"
    assert entry.sha, "pin a sha -- see the header of repos.yaml"
    assert entry.url.endswith("comma2k19.git")


# --------------------------------------------------------------------------
# camera2ego -- the defect that produced the previous result
# --------------------------------------------------------------------------


def test_camera2ego_for_zero_rpy_is_not_the_identity(toolkit):
    """The core of the removed implementation's bug, as an assertion.

    `rpy = 0` means "the camera is aligned with the direction of travel", which
    in paddock's conventions (camera [right, down, forward], ego
    [forward, left, up]) is a permutation, not the identity. Passing `np.eye(4)`
    instead is a 90 degree error.
    """
    m = L.camera2ego_from_rpy(np.zeros(3))
    expected = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert np.allclose(m[:3, :3], expected, atol=1e-12)
    assert not np.allclose(m[:3, :3], np.eye(3))
    assert np.isclose(m[2, 3], L.EON_HEIGHT)


def test_camera2ego_round_trips_through_openpilot(toolkit):
    """`camera2ego_from_rpy` must invert `CameraAdapter`'s rpy derivation.

    Both sides are needed for this to mean anything, so it is skipped when
    openpilot is not importable -- but where it does run, it is the test that
    pins the two adapters together.
    """
    op_cam = pytest.importorskip("openpilot.common.transformations.camera")
    op_orient = pytest.importorskip("openpilot.common.transformations.orientation")

    def derive_rpy(camera2ego):
        r_dev_cam = np.diag([1.0, -1.0, -1.0]) @ np.asarray(camera2ego)[:3, :3]
        return np.asarray(op_orient.rot2euler(op_cam.view_frame_from_device_frame.T @ r_dev_cam.T))

    for rpy in (np.zeros(3), np.array([0.0, 0.0613, 0.0234]), np.array([0.0, 0.0305, 0.0093])):
        assert np.allclose(derive_rpy(L.camera2ego_from_rpy(rpy)), rpy, atol=1e-9)

    # And the identity, which the removed loader passed, must come out as the
    # 90 degree error it is.
    assert np.allclose(np.degrees(derive_rpy(np.eye(4))), [-90.0, 0.0, 90.0], atol=1e-6)


def test_identity_camera2ego_is_rejected_by_the_policy_guard(toolkit):
    """The guard must refuse the value that produced the previous result."""
    policy_mod = pytest.importorskip("paddock.adapters.models.openpilot.policy")
    pytest.importorskip("openpilot.common.transformations.model")
    from paddock.policy.protocol import Pinhole

    pinhole = Pinhole(fx=910.0, fy=910.0, cx=582.0, cy=437.0, width=1164, height=874)
    with pytest.raises(policy_mod.ImplausibleMounting) as exc:
        policy_mod.CameraAdapter(pinhole, np.eye(4), resample=False)
    assert "90" in str(exc.value) or "identity" in str(exc.value)

    # And must accept a real one.
    ok = policy_mod.CameraAdapter(pinhole, L.camera2ego_from_rpy([0.0, 0.061, 0.023]), resample=False)
    assert np.allclose(np.degrees(ok.rpy), [0.0, 3.4952, 1.3178], atol=1e-2)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def test_synthetic_pose_velocity_is_forward_dominant(toolkit):
    pose = straight_pose()
    vd = pose.velocity_device
    assert np.allclose(vd[:, 0], 20.0, atol=1e-6)
    assert np.allclose(vd[:, 1:], 0.0, atol=1e-6)


def test_future_in_ego_frame_advances_forward_and_stays_centred(toolkit):
    pose = straight_pose(speed=20.0)
    fut = L.future_in_ego_frame(pose, 20, [0.5, 1.0, 1.5])
    assert np.allclose(fut[:, 0], [10.0, 20.0, 30.0], atol=1e-4)
    assert np.allclose(fut[:, 1], 0.0, atol=1e-6)


def test_ground_truth_relative_to_a_rotated_keyframe(toolkit):
    """The check a zero-rotation fixture cannot make.

    With the quaternion direction reversed, a relative position is mirrored
    through the direction of travel -- which is invisible when the vehicle's
    orientation is the identity. So the keyframe is yawed, and the answer is
    known: after a 90 deg left yaw, what was 30 m ahead is 30 m to the right.
    """
    pose = straight_pose(speed=20.0, n=200)
    # Yaw the keyframe 90 deg to the left (device z is down, so left is -z).
    yaw_left = L.rot_from_euler([0.0, 0.0, -np.pi / 2])
    idx = 20
    pose.orientations = pose.orientations.copy()
    rotated = pose.ecef_from_device[idx] @ yaw_left
    pose.orientations[idx] = rot2quat(rotated)
    reset_pose_cache(pose)  # the cached rotations predate the edit above

    fut = L.future_in_ego_frame(pose, idx, [1.5])  # 30 m of travel
    # Travel is still east; the car now faces north, so east is on its right,
    # and paddock's y is LEFT -- so y must be -30.
    assert np.isclose(fut[0, 0], 0.0, atol=1e-4), fut
    assert np.isclose(fut[0, 1], -30.0, atol=1e-4), fut


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_recovers_a_synthetic_mount_pitch(toolkit, tmp_path):
    """Pitch the camera down and check the estimate finds it.

    The pose stream describes the camera, so a camera pitched down by `p` sees
    the direction of travel as tilted *up* by `p` in its own frame. The estimate
    must return +p, and applying it must remove the residual.
    """
    pitch = np.radians(3.5)
    pose = straight_pose(speed=25.0, n=400)
    # device_from_calib = rot_from_euler([0, pitch, 0]); the recorded
    # orientation is the camera's, so rotate ecef_from_device by its inverse.
    ecef_from_device = pose.ecef_from_device[0] @ L.rot_from_euler([0.0, pitch, 0.0]).T
    pose.orientations = np.tile(rot2quat(ecef_from_device), (len(pose), 1))
    reset_pose_cache(pose)

    seg = write_segment(tmp_path / "seg", pose)

    cal = CAL.estimate([seg])
    assert cal.n_valid > 300
    assert np.isclose(cal.rpy[1], pitch, atol=1e-6), np.degrees(cal.rpy)
    assert CAL.plausible(cal.rpy)


def test_plausible_rejects_the_prior_bug():
    assert not CAL.plausible([-np.pi / 2, 0.0, np.pi / 2])
    assert CAL.plausible([0.0, np.radians(3.5), np.radians(1.3)])


# --------------------------------------------------------------------------
# Frames on the wire
# --------------------------------------------------------------------------


class _AlwaysValid:
    """A calibration survey that answers for any segment, so a frame test is a
    frame test and not a calibration test."""

    def get(self, segment):
        return CAL.Calibration(rpy=np.zeros(3), n_valid=10, n_total=10, spread=np.zeros(3), scope="route")


def _synthetic_frames(n: int = 64) -> list[np.ndarray]:
    """Frames whose pixels identify their index, so a misordered cache shows."""
    rng = np.random.default_rng(3)
    return [rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8) for _ in range(n)]


def _frame_bench(tmp_path, monkeypatch, frames, **settings):
    bench = _bench(history_frames=4, keyframes_per_segment=2, **settings)
    bench.cfg.runs = tmp_path / "runs"

    def fake_decode(video, limit=None, **kwargs):
        yield from frames[: limit if limit is not None else len(frames)]

    monkeypatch.setattr(L, "decode_frames", fake_decode)
    pose = straight_pose(speed=25.0, n=260)
    seg = write_segment(tmp_path / "seg", pose)
    return bench, L.SegmentRef(chunk=1, route_id="dongle|route", segment=4, path=seg)


def test_observations_carry_frame_paths_rather_than_pixels(tmp_path, monkeypatch):
    """paddock's open-loop convention, which this harness used to break.

    A 160-frame window base64'd into the request body is ~650 MB *per sample*;
    the node runs on this machine and can read the same disk.
    """
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)

    samples = list(bench._segment_samples(segment, _AlwaysValid()))
    assert samples
    for sample in samples:
        for obs in sample.observations:
            assert obs.images == {}
            assert obs.image_paths["CAM_FRONT"].endswith(".png")


def test_a_cached_frame_is_the_decoded_frame_bit_for_bit(tmp_path, monkeypatch):
    """Lossless, or two models' numbers stop being comparable across the change."""
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames, keep_frames=True)

    sample = next(iter(bench._segment_samples(segment, _AlwaysValid())))
    first = sample.native_id["first_frame_index"]
    for offset, obs in enumerate(sample.observations):
        loaded = obs.image("CAM_FRONT")
        assert np.array_equal(loaded, frames[first + offset])


def test_the_frames_are_released_once_the_segment_has_been_scored(tmp_path, monkeypatch):
    """800 frames a segment across 2035 segments is terabytes; they are scratch."""
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)

    samples = list(bench._segment_samples(segment, _AlwaysValid()))
    path = Path(samples[0].observations[0].image_paths["CAM_FRONT"])
    assert not path.exists()
    assert not bench._segment_frame_dir(segment).exists()


def test_keep_frames_leaves_them_for_the_next_run(tmp_path, monkeypatch):
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames, keep_frames=True)

    samples = list(bench._segment_samples(segment, _AlwaysValid()))
    path = Path(samples[0].observations[0].image_paths["CAM_FRONT"])
    assert path.exists()
    # And the kept directory is the shared one, not a per-run temporary, since
    # sharing between runs is the entire point of keeping.
    assert bench._frame_root == bench.cfg.runs / "comma2k19" / "_frames"


def test_a_second_pass_reuses_the_frames_instead_of_decoding_again(tmp_path, monkeypatch):
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames, keep_frames=True)
    list(bench._segment_samples(segment, _AlwaysValid()))

    def explode(*args, **kwargs):
        raise AssertionError("decoded again despite a full cache")
        yield  # pragma: no cover - generator, never reached

    monkeypatch.setattr(L, "decode_frames", explode)
    assert list(bench._segment_samples(segment, _AlwaysValid()))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_a_perfect_prediction_scores_zero():
    gt = np.stack([np.arange(1, 11) * 10.0, np.zeros(10)], axis=1)
    m = MC.score_trajectory(gt, gt)
    assert m["ade"] == pytest.approx(0.0)
    assert m["cross_track_mae"] == pytest.approx(0.0)
    assert m["lateral_at_arclength_mae"] == pytest.approx(0.0)
    assert m["path_length_ratio"] == pytest.approx(1.0)


def test_a_stationary_prediction_is_not_flattered_by_the_lateral_metric():
    """The failure mode the previous comma2k19 report misread.

    A model predicting a stationary car has a *small* axis-aligned lateral
    error, because its prediction sits at the origin and the ground truth's own
    lateral excursion is small. The along-track error is the whole distance
    travelled, and `path_length_ratio` is ~0. Both must say so.
    """
    gt = np.stack([np.arange(1, 11) * 10.0, np.linspace(0, 1.0, 10)], axis=1)
    stationary = np.zeros((10, 2))
    m = MC.score_trajectory(stationary, gt)
    assert m["lateral_mae"] < 1.0  # looks fine, and is not
    assert m["along_track_mae"] > 25.0  # the honest signal
    assert m["path_length_ratio"] < 0.01  # unmistakable
    assert np.isnan(m["lateral_at_arclength_mae"])  # the plan never got there


def test_speed_error_lands_in_along_track_not_cross_track_on_a_curve():
    """The reason the path-tangent split exists.

    On a curve, an axis-aligned split puts part of a pure speed error into y and
    it reads as a lane-keeping failure. Projecting onto the ground-truth tangent
    keeps it where it belongs.
    """
    s = np.linspace(0, 60, 11)[1:]
    radius = 200.0
    gt = np.stack([radius * np.sin(s / radius), radius * (1 - np.cos(s / radius))], axis=1)
    slow = np.stack([radius * np.sin(0.8 * s / radius), radius * (1 - np.cos(0.8 * s / radius))], axis=1)
    m = MC.score_trajectory(slow, gt, eval_horizon_s=5.0)
    assert m["cross_track_mae"] < 0.2 * m["along_track_mae"]
    assert m["lateral_at_arclength_mae"] < 0.05
    # The axis-aligned split, by contrast, smears it.
    assert m["lateral_mae"] > 5 * m["cross_track_mae"]


def test_lateral_at_arclength_catches_a_real_lane_offset():
    """And must *not* be blind to an actual lateral error."""
    gt = np.stack([np.arange(1, 11) * 10.0, np.zeros(10)], axis=1)
    offset = gt + np.array([0.0, 1.5])
    m = MC.score_trajectory(offset, gt)
    assert m["lateral_at_arclength_mae"] == pytest.approx(1.5, abs=0.05)
    assert m["along_track_mae"] < 0.2


# --------------------------------------------------------------------------
# Keyframe selection
# --------------------------------------------------------------------------


class _Cfg:
    def __init__(self, **kw):
        self._kw = kw

    def get(self, key, default=None):
        return self._kw.get(key, default)


def _bench(**settings):
    from paddock.adapters.benchmarks.comma2k19.harness import Comma2k19Benchmark

    bench = Comma2k19Benchmark.__new__(Comma2k19Benchmark)
    bench.spec = type("S", (), {"name": "comma2k19", "dataset": "comma2k19"})()
    bench.cfg = type("C", (), {"benchmarks": {"comma2k19": _Cfg(**settings)}, "datasets": {}})()
    return bench


def test_keyframes_leave_room_for_history_and_future():
    from paddock.adapters.benchmarks.comma2k19.harness import (
        DEFAULT_HISTORY_FRAMES,
        INTERVAL_S,
        N_GT_POSES,
    )

    bench = _bench()
    future = int(round(N_GT_POSES * INTERVAL_S * L.SRC_HZ))
    for n in (1197, 1200):
        keyframes = bench.keyframe_indices(n)
        assert keyframes
        for k in keyframes:
            # The keyframe is the *last* frame of its own history window, so a
            # full 160-frame window ends at index 159. The earliest keyframe
            # sits exactly there, and did before the default rose above one
            # keyframe per segment -- it was just never selected.
            assert k >= DEFAULT_HISTORY_FRAMES - 1
            assert k + future < n
            # The invariant the constant above stands in for, stated directly:
            # every frame the model is shown was actually recorded.
            assert bench.grid.window(k)[0] >= 0


def test_short_segments_yield_no_keyframes():
    # The real dataset has one 194-frame final segment; it cannot support both
    # a 160-frame history and a 5 s future.
    assert _bench().keyframe_indices(194) == []


def test_keyframe_selection_is_deterministic_and_scales_with_the_setting():
    a = _bench(keyframes_per_segment=4).keyframe_indices(1200)
    b = _bench(keyframes_per_segment=4).keyframe_indices(1200)
    assert a == b
    assert len(a) == 4
    assert a == sorted(a)
    assert len(_bench(keyframes_per_segment=1).keyframe_indices(1200)) == 1


def test_run_config_records_what_changes_the_number():
    cfg = _bench(history_frames=200, keyframes_per_segment=2).run_config()
    assert cfg["history_frames"] == 200
    assert cfg["keyframes_per_segment"] == 2
    assert "calibration" in cfg


def test_a_stationary_prediction_reproduces_the_published_numbers(toolkit):
    """The previous comma2k19 result, reproduced from a plan of literally zero.

    This is the load-bearing evidence for the claim in
    `docs/benchmarks/comma2k19.md` that the earlier run was measuring a model
    predicting a stationary car, rather than a model with a large longitudinal
    error.

    The published table (100 segments, openpilot) was:

        ade 38.84 | fde@1s 22.29 | fde@2s 44.43 | fde@3s 66.36
        longitudinal_mae 38.82 | lateral_mae 0.96

    Scoring an all-zero trajectory against a straight-highway ground truth at
    the recorded ~22.3 m/s reproduces every one of those to within a few per
    cent, because at zero prediction the error simply *is* the distance
    travelled. No model behaviour is needed to explain the numbers.

    The ground truth below is the real recorded future of
    `Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3` at frame 628, rounded --
    a typical straight-ish highway segment, not one chosen to fit.
    """
    gt = np.array(
        [
            [11.2, -0.13],
            [22.5, -0.23],
            [33.7, -0.32],
            [45.0, -0.41],
            [56.3, -0.51],
            [67.5, -0.63],
            [78.6, -0.76],
            [89.6, -0.88],
            [100.5, -0.98],
            [111.2, -1.11],
        ]
    )
    m = MC.score_trajectory(np.zeros((10, 2)), gt, eval_horizon_s=3.0)

    published = {
        "ade": 38.84,
        "fde@1s": 22.29,
        "fde@2s": 44.43,
        "fde@3s": 66.36,
        "longitudinal_mae": 38.82,
        "lateral_mae": 0.96,
    }
    for key, want in published.items():
        got = m[key]
        # 6 % covers the difference between one segment's speed and the mean of
        # the hundred the published run averaged; lateral is absolute because
        # it is sub-metre either way.
        tol = 0.06 * want if key != "lateral_mae" else 1.0
        assert abs(got - want) <= tol, f"{key}: reproduced {got:.2f}, published {want}"

    # And the metrics added since must make the cause unmistakable.
    assert m["path_length_ratio"] == pytest.approx(0.0)
    assert np.isnan(m["lateral_at_arclength_mae"])
    assert m["cross_track_mae"] < 0.1


# --------------------------------------------------------------------------
# Evaluation rate
# --------------------------------------------------------------------------


def test_the_native_rate_is_stride_one_and_touches_nothing():
    from paddock.adapters.benchmarks.comma2k19.harness import EvalGrid

    grid = EvalGrid.build(L.SRC_HZ, history_frames=160)
    assert grid.stride == 1
    assert grid.history_span_frames == 160
    assert grid.window(700) == list(range(541, 701))


def test_lowering_the_rate_widens_the_window_without_moving_the_keyframe():
    """The scored moment must not move with the rate.

    A rate sweep that also shifted the keyframe would vary two things at once,
    and the resulting curve would not be a statement about observation rate.
    """
    from paddock.adapters.benchmarks.comma2k19.harness import EvalGrid

    for hz, stride in ((20.0, 1), (10.0, 2), (5.0, 4), (4.0, 5), (2.0, 10)):
        grid = EvalGrid.build(hz, history_frames=40)
        assert grid.stride == stride
        window = grid.window(900)
        assert window[-1] == 900, "the keyframe is always the last observation"
        assert len(window) == 40, "the model sees the same number of frames at every rate"
        assert set(np.diff(window)) == {stride}


def test_an_unreachable_rate_is_refused_with_the_nearest_reachable_ones():
    from paddock.adapters.benchmarks.comma2k19.harness import EvalGrid, UnreachableRate

    with pytest.raises(UnreachableRate) as exc:
        EvalGrid.build(15.0, history_frames=160)
    # 15 Hz would need a stride of 1.33; silently rounding it to 1 would report
    # a 15 Hz run that fed the model 20 Hz.
    assert "20/n" in str(exc.value)
    assert "10" in str(exc.value) and "6.66667" in str(exc.value)


def test_input_hz_comes_from_the_environment_before_the_site_config(monkeypatch):
    from paddock.adapters.benchmarks.comma2k19.harness import ENV_INPUT_HZ

    monkeypatch.delenv(ENV_INPUT_HZ, raising=False)
    assert _bench().input_hz == L.SRC_HZ
    assert _bench(input_hz=4.0).input_hz == 4.0
    monkeypatch.setenv(ENV_INPUT_HZ, "2")
    assert _bench(input_hz=4.0).input_hz == 2.0


def test_run_config_separates_two_rates(monkeypatch):
    """Two rates must not share a config hash, or one overwrites the other."""
    from paddock.adapters.benchmarks.comma2k19.harness import ENV_INPUT_HZ

    monkeypatch.delenv(ENV_INPUT_HZ, raising=False)
    fast = _bench().run_config()
    slow = _bench(input_hz=4.0).run_config()
    assert fast != slow
    assert (fast["input_hz"], fast["stride"]) == (20.0, 1)
    assert (slow["input_hz"], slow["stride"]) == (4.0, 5)
    assert slow["history_span_s"] > fast["history_span_s"]


def test_a_window_longer_than_a_segment_is_refused_loudly(monkeypatch):
    """Zero samples with no explanation is the failure mode this prevents."""
    from paddock.adapters.benchmarks.comma2k19.harness import ENV_INPUT_HZ

    monkeypatch.delenv(ENV_INPUT_HZ, raising=False)
    assert _bench(input_hz=4.0)._rate_problems() == []

    problems = _bench(input_hz=2.0)._rate_problems()
    assert len(problems) == 1 and "exceeds a 60 s comma2k19 segment" in problems[0]
    # The number it names must be one that actually works.
    capped = int(problems[0].split("history_frames to at most ")[1].split(",")[0])
    bench = _bench(input_hz=2.0, history_frames=capped)
    assert bench._rate_problems() == []
    assert bench.keyframe_indices(int(60 * L.SRC_HZ)) != []
    assert _bench(input_hz=2.0, history_frames=capped + 1)._rate_problems() != []


def test_keyframes_leave_room_for_the_subsampled_window():
    for hz, history in ((20.0, 160), (10.0, 160), (4.0, 160), (2.0, 90)):
        bench = _bench(input_hz=hz, history_frames=history)
        grid = bench.grid
        for n in (1197, 1200):
            for k in bench.keyframe_indices(n):
                window = grid.window(k)
                assert window[0] >= 0 and window[-1] == k
                assert k + grid.future_frames < n


def test_a_model_below_the_configured_rate_is_refused_before_it_loads(monkeypatch):
    from paddock.adapters.benchmarks.comma2k19.harness import ENV_ALLOW_SLOW, ENV_INPUT_HZ

    monkeypatch.delenv(ENV_INPUT_HZ, raising=False)
    monkeypatch.delenv(ENV_ALLOW_SLOW, raising=False)
    model = type("M", (), {"name": "openpilot", "min_input_hz": 10.0})()

    assert _bench().model_problems(model) == []
    assert _bench(input_hz=10.0).model_problems(model) == []

    problems = _bench(input_hz=4.0).model_problems(model)
    assert len(problems) == 1
    assert "needs 10 Hz" in problems[0] and "supplies 4 Hz" in problems[0]
    assert ENV_ALLOW_SLOW in problems[0]

    # The override runs the pairing and marks the run unusable for a leaderboard.
    monkeypatch.setenv(ENV_ALLOW_SLOW, "1")
    assert _bench(input_hz=4.0).model_problems(model) == []
    assert _bench(input_hz=4.0).invalid_reasons() != []


# --------------------------------------------------------------------------
# Horizons and time alignment
# --------------------------------------------------------------------------


def test_a_horizon_the_plan_does_not_reach_is_nan_not_a_shorter_horizon():
    """The defect this replaced: `fde@5s` was FDE at 4.0 s wearing a 5 s label.

    openpilot returned 8 waypoints on a 0.5 s grid -- 4.0 s -- and the metric
    clamped its end index to what was available and emitted anyway.
    `benchmark.yaml` then advertised the result as a 5 s number.
    """
    gt = np.stack([20.0 * 0.5 * np.arange(1, 21), np.zeros(20)], axis=1)
    pred = gt[:8].copy()  # a perfect 4.0 s plan

    m = MC.score_trajectory(pred, gt)
    assert m["ade"] == pytest.approx(0.0)
    assert m["fde@3s"] == pytest.approx(0.0)
    assert m["ade@2s"] == pytest.approx(0.0)
    for key in ("ade@5s", "fde@5s", "ade@10s", "fde@10s"):
        assert np.isnan(m[key]), f"{key} was reported for a plan that stops at 4.0 s"
    assert m["scored_horizon_s"] == pytest.approx(4.0)


def test_a_longer_plan_reaches_the_long_horizons():
    gt = np.stack([20.0 * 0.5 * np.arange(1, 21), np.zeros(20)], axis=1)
    m = MC.score_trajectory(gt.copy(), gt)
    assert m["scored_horizon_s"] == pytest.approx(10.0)
    for key in ("ade@5s", "fde@5s", "ade@10s", "fde@10s"):
        assert m[key] == pytest.approx(0.0)


def test_scoring_pairs_by_time_not_by_index():
    """A model on a grid other than 0.5 s must still be scored against 0.5 s truth.

    The runner stamps a Trajectory with the *model's* `waypoint_dt` while this
    harness lays the recorded future out on its own `INTERVAL_S`. Pairing them
    elementwise is correct only while those two numbers coincide, which is a
    fact about today's model roster and not a promise either side makes.
    """
    from paddock.adapters.benchmarks.comma2k19.harness import INTERVAL_S, N_GT_POSES
    from paddock.benchmark import Sample
    from paddock.policy.protocol import Trajectory

    speed = 20.0
    gt_t = INTERVAL_S * np.arange(1, N_GT_POSES + 1)
    gt = np.stack([speed * gt_t, np.zeros(N_GT_POSES)], axis=1)
    sample = Sample(sample_id="t", observations=[], ground_truth=gt)

    # Same motion, quarter-second grid: perfect, and index pairing would have
    # scored waypoint k of the plan against 2k's worth of ground truth.
    fine_t = 0.25 * np.arange(1, 41)
    fine = Trajectory(np.stack([speed * fine_t, np.zeros(40)], axis=1), dt=0.25)
    m = _bench().score(sample, fine)
    assert m["ade"] == pytest.approx(0.0, abs=1e-9)
    assert m["fde@3s"] == pytest.approx(0.0, abs=1e-9)
    assert m["scored_horizon_s"] == pytest.approx(10.0)

    # Index pairing on the same input would have looked like a large error.
    naive = float(np.linalg.norm(fine.xy[:20] - gt, axis=1).mean())
    assert naive > 10.0


def test_a_plan_past_the_recorded_future_is_trimmed_not_extrapolated():
    from paddock.adapters.benchmarks.comma2k19.harness import INTERVAL_S, N_GT_POSES
    from paddock.benchmark import Sample
    from paddock.policy.protocol import Trajectory

    gt_t = INTERVAL_S * np.arange(1, N_GT_POSES + 1)
    gt = np.stack([20.0 * gt_t, np.zeros(N_GT_POSES)], axis=1)
    sample = Sample(sample_id="t", observations=[], ground_truth=gt)

    long_t = INTERVAL_S * np.arange(1, 41)  # 20 s of plan against 10 s of truth
    plan = Trajectory(np.stack([20.0 * long_t, np.zeros(40)], axis=1), dt=INTERVAL_S)
    m = _bench().score(sample, plan)
    assert m["scored_horizon_s"] == pytest.approx(10.0)
    assert m["ade@10s"] == pytest.approx(0.0, abs=1e-9)


def test_score_joins_the_auxiliary_metrics_onto_the_trajectory_ones():
    """One `score()` call must answer for every key `benchmark.yaml` declares.

    The trajectory metrics and the auxiliary ones are computed by two modules
    and merged here; a model publishing extras and one publishing none must
    still produce the same key set, or the run's metrics stop being comparable.
    """
    from paddock.adapters.benchmarks.comma2k19 import auxiliary as AUX
    from paddock.adapters.benchmarks.comma2k19.harness import INTERVAL_S, N_GT_POSES
    from paddock.benchmark import Sample
    from paddock.policy.protocol import Trajectory

    gt_t = INTERVAL_S * np.arange(1, N_GT_POSES + 1)
    gt = np.stack([20.0 * gt_t, np.zeros(N_GT_POSES)], axis=1)
    sample = Sample(
        sample_id="t",
        observations=[],
        ground_truth=gt,
        native_id={"reference": {"speed": 20.0, "yaw_rate": 0.0}},
    )
    plan = Trajectory(gt.copy(), dt=INTERVAL_S)
    bare = _bench().score(sample, plan)
    assert set(AUX.ALL_KEYS) <= set(bare)
    assert np.isnan(bare["speed_abs_err"])  # no extras, so nothing to compare

    plan.extras = {"pose": {"velocity": [20.5, 0.0, 0.0], "rotation_rate": [0.0, 0.0, 0.0]}}
    rich = _bench().score(sample, plan)
    assert set(rich) == set(bare)
    assert rich["speed_abs_err"] == pytest.approx(0.5)
