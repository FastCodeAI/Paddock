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

import os
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import calibration as CAL
from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.adapters.benchmarks.comma2k19.tests import synthetic as S
from paddock.adapters.benchmarks.comma2k19.tests.synthetic import (
    reset_pose_cache,
    rot2quat,
    straight_pose,
    write_segment,
)
from paddock.config import BenchmarkOverride

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

    The reference lives in `synthetic.py` rather than here because the
    `rotations` fixture stands it in for the toolkit where no clone exists. That
    gives it a second job -- and makes this test the thing that stops the
    stand-in drifting away from the toolkit it substitutes for.
    """
    rng = np.random.default_rng(11)

    for _ in range(200):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        assert np.allclose(toolkit.rot_from_quat(q), S.reference_rot_from_quat(q), atol=1e-12)

        rpy = rng.uniform(-np.pi / 3, np.pi / 3, 3)
        assert np.allclose(toolkit.rot_from_euler(rpy), S.reference_rot_from_euler(rpy), atol=1e-12)

    # The view-frame basis the stand-in carries, against the clone's own.
    assert np.array_equal(np.asarray(toolkit.device_frame_from_view_frame), S.REFERENCE_DEVICE_FROM_VIEW)

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


def test_observations_carry_frame_paths_rather_than_pixels(tmp_path, monkeypatch, rotations):
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
            # Raw, not PNG: the node decompressing what the runner just
            # compressed was two thirds of this benchmark's wall clock.
            assert obs.image_paths["CAM_FRONT"].endswith(".npy")


def test_a_cached_frame_is_the_decoded_frame_bit_for_bit(tmp_path, monkeypatch, rotations):
    """Lossless, or two models' numbers stop being comparable across the change."""
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames, keep_frames=True)

    sample = next(iter(bench._segment_samples(segment, _AlwaysValid())))
    first = sample.native_id["first_frame_index"]
    for offset, obs in enumerate(sample.observations):
        loaded = obs.image("CAM_FRONT")
        assert np.array_equal(loaded, frames[first + offset])


def test_the_frames_are_released_once_enough_segments_have_gone_by(tmp_path, monkeypatch, rotations):
    """800 frames a segment across 2035 segments is terabytes; they are scratch.

    Released on a ring rather than immediately, because the runner holds
    samples this generator has already passed -- see `_retire`. The property
    that matters is that the high-water mark is bounded, not that any one
    segment is gone the instant it is scored.
    """
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)

    samples = list(bench._segment_samples(segment, _AlwaysValid()))
    path = Path(samples[0].observations[0].image_paths["CAM_FRONT"])
    assert path.exists(), "released while a consumer could still be reading it"

    for _ in range(bench._retain_segments):
        bench._retire(tmp_path / "somewhere-else")
    assert not path.exists()
    assert not bench._segment_frame_dir(segment).exists()


def test_the_staged_frames_of_a_sweep_stay_bounded(tmp_path, monkeypatch):
    """The ring is a budget, not a leak: N segments retire to N directories."""
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)

    for index in range(3 * bench._retain_segments):
        directory = tmp_path / f"staged-{index}"
        directory.mkdir()
        bench._retire(directory)
        assert len(bench._retiring) <= bench._retain_segments


def test_the_ring_is_deep_enough_for_every_sample_the_runner_holds():
    """The ring has to cover the consumer's pipeline, not a fixed four segments.

    The runner may hold `PIPELINE_BATCHES` batches of `node_count` samples at
    once, and those samples are consecutive, so a site that raises `nodes`
    raises how far behind the newest staged segment the oldest live sample can
    be. At `nodes = 16` and five keyframes that is ten segments, and a fixed
    four would delete frames a node is still reading -- which surfaces as a
    file-not-found from inside the model, the hardest place to read it from.
    """
    for nodes, keyframes in [(1, 5), (4, 5), (16, 5), (16, 1)]:
        bench = _bench(nodes=nodes, keyframes_per_segment=keyframes)
        in_flight = H.PIPELINE_BATCHES * nodes
        assert bench._retain_segments >= in_flight / keyframes
        assert bench._retain_segments >= H.RETAIN_SEGMENTS


def _model(vram_gb: float):
    return type("M", (), {"vram_gb": vram_gb, "name": "stub"})()


@pytest.mark.parametrize(
    ("free_gb", "vram_gb", "nodes", "expected"),
    [
        (90.0, 18.0, 4, 4),  # a roomy card runs the site's full width
        (24.0, 18.0, 4, 1),  # only one 18 GB node fits in 24 GB
        (24.0, 10.0, 4, 2),
        (90.0, 26.0, 8, 3),  # VRAM binds below the site's ceiling
        (90.0, 0.0, 8, 8),  # a CPU model is not divided up
        (4.0, 18.0, 4, 1),  # never zero: one node that waits beats none at all
    ],
)
def test_node_count_is_capped_by_what_the_card_will_hold(monkeypatch, free_gb, vram_gb, nodes, expected):
    """Four nodes of an 18 GB model into 24 GB is not a slow run, it is no run.

    `runtime.gpu.wait_for` gates each launch on declared VRAM, so the three
    that do not fit poll for two hours and then raise. Dividing up front is
    what makes the sweep finish.
    """
    monkeypatch.setattr(H.gpu, "free_gb", lambda: free_gb)
    assert _bench(nodes=nodes).node_count(_model(vram_gb)) == expected


def test_node_count_falls_back_to_the_site_budget_without_nvidia_smi(monkeypatch):
    """No usable nvidia-smi is not a licence to start eight 18 GB nodes."""
    monkeypatch.setattr(H.gpu, "free_gb", lambda: None)
    assert _bench(nodes=8).node_count(_model(18.0)) == 1  # 24 GB budget // 18
    assert _bench(nodes=8).node_count(_model(0.0)) == 8


def test_node_count_without_a_model_is_the_site_setting(monkeypatch):
    """`_retain_segments` calls this with None and must keep sizing the ring."""
    monkeypatch.setattr(H.gpu, "free_gb", lambda: 8.0)
    assert _bench(nodes=6).node_count(None) == 6


def test_keep_frames_leaves_them_for_the_next_run(tmp_path, monkeypatch, rotations):
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames, keep_frames=True)

    samples = list(bench._segment_samples(segment, _AlwaysValid()))
    path = Path(samples[0].observations[0].image_paths["CAM_FRONT"])
    assert path.exists()
    # And the kept directory is the shared one, not a per-run temporary, since
    # sharing between runs is the entire point of keeping.
    assert bench._frame_root == bench.cfg.runs / "comma2k19" / "_frames"


def test_a_ring_left_by_a_dead_run_is_swept_and_a_live_one_is_not(tmp_path, monkeypatch):
    """`atexit` does not run on SIGTERM, which is how a sweep usually ends.

    This box was found carrying an 888 MB ring whose pid had been dead for five
    days. On the tmpfs the docs recommend, that is unaccounted RAM; the pid is
    in the directory name precisely so the next run can tell.
    """
    bench = _bench()
    bench.cfg.runs = tmp_path / "runs"
    cache = bench._frame_cache
    cache.mkdir(parents=True)

    dead = cache / "run-999999999-aaaaaaaa"  # above pid_max; cannot be running
    live = cache / f"run-{os.getpid()}-bbbbbbbb"
    stranger = cache / "not-a-ring"
    for d in (dead, live, stranger):
        d.mkdir()
        (d / "frame.npy").write_bytes(b"x")

    mine = bench._frame_root

    assert not dead.exists()
    assert live.exists(), "swept a ring whose process is still staging into it"
    assert stranger.exists(), "swept something this cache did not create"
    assert mine.exists() and mine.parent == cache


def test_preflight_says_so_when_the_frame_cache_cannot_be_written(tmp_path):
    """Staging starts after four nodes are up and have waited for VRAM.

    So an unwritable cache -- the ordinary case being `/dev/shm/...` owned by
    whoever on this machine ran the sweep first -- surfaced minutes and one
    checkpoint read in, rather than in a dry run.
    """
    blocked = tmp_path / "theirs"
    blocked.mkdir(mode=0o500)
    bench = _bench(frame_cache=str(blocked / "mine"))
    bench.cfg.runs = tmp_path / "runs"

    problems = bench._frame_cache_problems(n_segments=2033)
    assert len(problems) == 1
    assert "not writable" in problems[0]
    assert str(blocked / "mine") in problems[0]


def test_preflight_refuses_a_volume_that_cannot_hold_the_frames(tmp_path, monkeypatch):
    """`keep_frames` on the 4.2 TB volume this site uses would fill it in minutes.

    The gate is the high-water mark, not the total written: without
    `keep_frames` a run holds a ring of four segments, so the same 649 GB that
    refuses a retained sweep must let an ordinary one through.
    """
    bench = _bench(frame_cache=str(tmp_path / "cache"), keep_frames=True)
    bench.cfg.runs = tmp_path / "runs"
    monkeypatch.setattr(
        H.shutil, "disk_usage", lambda p: type("U", (), {"free": 649e9, "total": 4.2e12, "used": 0})()
    )

    (problem,) = bench._frame_cache_problems(n_segments=2033)
    assert "649 GB free" in problem and "4879 GB" in problem

    roomy = _bench(frame_cache=str(tmp_path / "cache"))
    roomy.cfg.runs = tmp_path / "runs"
    assert roomy._frame_cache_problems(n_segments=2033) == []


def _sweepable(bench, segment):
    """Point `samples()` at one segment, with a calibration that always answers."""
    bench.__dict__["_segments"] = [segment]
    bench.__dict__["_calibrations"] = _AlwaysValid()
    return bench


def test_a_segment_a_resume_has_finished_is_never_opened(tmp_path, monkeypatch, rotations):
    """The whole value of `skip` for this benchmark.

    A sample costs a minute of HEVC decoded and hundreds of frames staged,
    before the first sample of that segment exists. Filtering only the yielded
    samples would pay that price for every segment a resume is skipping, so a
    resume at 8000 of 10165 would save nothing at all.
    """
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)
    _sweepable(bench, segment)

    ids = [s.sample_id for s in bench.samples()]
    assert len(ids) == 2

    def explode(*args, **kwargs):
        raise AssertionError("decoded a segment whose every sample was already scored")
        yield  # pragma: no cover - generator, never reached

    monkeypatch.setattr(L, "decode_frames", explode)
    assert list(bench.samples(skip=set(ids))) == []


def test_the_segment_an_interruption_landed_in_yields_only_what_is_left(tmp_path, monkeypatch, rotations):
    """The partial segment is the one the outer skip cannot help with: its ids
    are not all in `skip`, so it is decoded, and the per-sample test after it
    is what keeps the already-scored keyframe out of the record."""
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)
    _sweepable(bench, segment)

    ids = [s.sample_id for s in bench.samples()]
    remaining = [s.sample_id for s in _sweepable(bench, segment).samples(skip={ids[0]})]
    assert remaining == ids[1:]


def test_the_last_samples_of_a_stream_keep_their_frames_after_it_ends(tmp_path, monkeypatch, rotations):
    """The end of the stream is not the end of the scoring, and this cost a run.

    `--limit` stops the iterator while the runner is still holding the last
    `PIPELINE_BATCHES` batches, and exhausting the segments does the same. The
    generator used to delete the whole run cache in its `finally`, so a
    four-node `--limit 100` lost its last eight samples to file-not-found
    raised *inside the model* and returned as an HTTP 500 -- and an eight-node
    one lost thirty-two. Nothing about that error says "staging".
    """
    frames = _synthetic_frames()
    bench, segment = _frame_bench(tmp_path, monkeypatch, frames)
    _sweepable(bench, segment)

    for taken in (bench.samples(limit=1), bench.samples()):
        samples = list(taken)
        assert samples
        for obs in samples[-1].observations:
            path = Path(obs.image_paths["CAM_FRONT"])
            assert path.exists(), f"{path} was deleted while the runner still held the sample"


def test_the_frame_cache_can_be_moved_off_the_run_store(tmp_path):
    """`{runs}` is a network mount at some sites, and these frames are scratch:
    written once and read once, seconds apart, by two processes on this machine."""
    bench = _bench(frame_cache=str(tmp_path / "shm"))
    bench.cfg.runs = tmp_path / "runs"
    assert bench._frame_root.parent == tmp_path / "shm"

    default = _bench()
    default.cfg.runs = tmp_path / "runs"
    assert default._frame_root.parent == tmp_path / "runs" / "comma2k19" / "_frames"


def test_the_node_count_is_a_site_setting_with_a_default():
    """More nodes is the speedup; the site says how many cores it can spare."""
    assert _bench().node_count(None) == H.DEFAULT_NODES
    assert _bench(nodes=12).node_count(None) == 12
    assert _bench(nodes=0).node_count(None) == 1, "zero nodes scores nothing"


def test_a_second_pass_reuses_the_frames_instead_of_decoding_again(tmp_path, monkeypatch, rotations):
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


def test_the_path_lengths_are_the_metres_the_car_actually_covered():
    """They start at the ego origin, which is where the car is at `t = 0`.

    Differencing the waypoints alone drops the first leg -- the longest one --
    so a car holding 30 m/s through a 3 s horizon reported 75 m of a 90 m drive.
    The ratio survived it (both sides lost the same leg) but the two lengths are
    published in metres and were short by one interval's worth.
    """
    gt = np.stack([np.arange(1, 7) * 15.0, np.zeros(6)], axis=1)  # 30 m/s, 0.5 s
    m = MC.score_trajectory(gt, gt, eval_horizon_s=3.0)
    assert m["recorded_path_length"] == pytest.approx(90.0)
    assert m["predicted_path_length"] == pytest.approx(90.0)
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


def _Cfg(**kw):
    """The site's `[benchmarks.comma2k19]` block, as the harness really sees it.

    The real `BenchmarkOverride`, not a double with the same two methods --
    and stringified here because `SiteConfig.load` stringifies every setting.
    The hand-written double stored raw Python values, which is exactly why
    `keep_frames = false` reading as **true** in production went unnoticed by a
    green suite: no test in this file ever exercised the string path.
    """
    return BenchmarkOverride(settings={k: str(v) for k, v in kw.items()})


def _bench(**settings):
    from paddock.adapters.benchmarks.comma2k19.harness import Comma2k19Benchmark

    bench = Comma2k19Benchmark.__new__(Comma2k19Benchmark)
    bench.spec = type("S", (), {"name": "comma2k19", "dataset": "comma2k19"})()
    bench.cfg = type(
        "C",
        (),
        {
            "benchmarks": {"comma2k19": _Cfg(**settings)},
            "datasets": {},
            # What `node_count` falls back to when nvidia-smi cannot be read.
            "gpu": type("G", (), {"budget_gb": 24.0, "serialize": True})(),
            # `frame_cache` is expanded for template variables by the real
            # SiteConfig; here a path is already a path.
            "resolve": lambda self, value: Path(value),
        },
    )()
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


def test_a_stationary_prediction_reproduces_the_published_numbers():
    """The previous comma2k19 result, reproduced from a plan of literally zero.

    Takes no `toolkit` fixture: it is numpy and a table of recorded metres, with
    no pose maths in it. It used to ask for one anyway, which meant the evidence
    this docstring calls load-bearing was skipped on every machine without a
    clone -- including CI, which is every PR.

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


# --------------------------------------------------------------------------
# Scene sets
# --------------------------------------------------------------------------

#: What this pretend site holds, as `SegmentRef.sample_key` spells it.
SITE_KEYS = (
    "1/aaa|2018-01-01--00-00-00/3",
    "1/aaa|2018-01-01--00-00-00/4",
    "2/bbb|2018-02-02--00-00-00/7",
)
ONE_SCENE = "keyframes_per_segment: 85\nscenes:\n- 2/bbb|2018-02-02--00-00-00/7\n"


def _ref(key: str) -> L.SegmentRef:
    """The `SegmentRef` whose `sample_key` is `key` -- the parse `list_segments` does."""
    chunk, route_id, segment = key.split("/")
    return L.SegmentRef(chunk=int(chunk), route_id=route_id, segment=int(segment), path=Path(key))


def _scene_bench(monkeypatch, tmp_path, name, body: str | None, keys=SITE_KEYS, **settings):
    """A bench asking for scene set `name`, with `scenes/` redirected to disk.

    Written out rather than pointed at the tracked `viewer10.yaml`: whether
    pinning works is a separate fact from which ten scenes this repo ships, and
    a test that asserts both fails for two unrelated reasons.
    """
    monkeypatch.setattr(H, "SCENES_DIR", tmp_path)
    monkeypatch.setenv(H.ENV_SCENE_SET, name)
    if body is not None:
        (tmp_path / f"{name}.yaml").write_text(body)
    bench = _bench(**settings)
    bench.__dict__["_all_segments"] = [_ref(k) for k in keys]
    return bench


def test_a_scene_set_restricts_the_segments_and_names_their_density(monkeypatch, tmp_path):
    """The one selection `--limit N` cannot express: score *these* segments."""
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE, keyframes_per_segment=5)
    assert [s.sample_key for s in bench._segments] == ["2/bbb|2018-02-02--00-00-00/7"]
    # The set wins over the site setting: it names the segments and the density
    # that working set means, which is the more specific of the two statements.
    assert bench.keyframes_per_segment == 85

    monkeypatch.delenv(H.ENV_SCENE_SET)
    plain = _bench(keyframes_per_segment=5)
    assert plain.scene_set is None and plain.keyframes_per_segment == 5


def test_a_dry_run_counts_the_scene_set_rather_than_the_dataset(monkeypatch, tmp_path):
    """`coverage()`, `preflight()` and `status()` must all describe the run."""
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    monkeypatch.setattr(type(bench), "_n_frames", staticmethod(lambda segment: 1200))

    rows, missing = bench.coverage()
    assert missing == []
    assert {r["sample_key"].rsplit("/", 1)[0] for r in rows} == {"2/bbb|2018-02-02--00-00-00/7"}
    assert len(rows) == 85  # the set's own density, not the default five


def test_a_scene_set_naming_an_absent_segment_is_a_preflight_problem(monkeypatch, tmp_path):
    """Nine of ten, quietly, is the failure pinning a set exists to prevent."""
    body = ONE_SCENE + "- 9/nope|2019-09-09--00-00-00/1\n"
    bench = _scene_bench(monkeypatch, tmp_path, "ten", body)
    problems = bench._scene_set_problems()
    assert len(problems) == 1 and "9/nope|2019-09-09--00-00-00/1" in problems[0]


def test_a_scene_set_that_is_not_here_says_so_instead_of_raising(monkeypatch, tmp_path):
    """Reported beside every other problem, not as a traceback out of `_segments`."""
    (tmp_path / "ten.yaml").write_text(ONE_SCENE)
    bench = _scene_bench(monkeypatch, tmp_path, "absent", None)
    assert bench._segments == []
    problem = bench._scene_set_problems()[0]
    assert "no scene set 'absent'" in problem and "have: ten" in problem


def test_a_scene_set_makes_the_run_invalid(monkeypatch, tmp_path):
    """Ten chosen segments is a working set to look at, not comma2k19.

    `_validity` turns this into `valid: false`, which is what keeps these runs
    out of `paddock report` and out of `render.load_runs` unless asked for by id.
    """
    monkeypatch.delenv(H.ENV_ALLOW_SLOW, raising=False)
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    reasons = bench.invalid_reasons()
    assert len(reasons) == 1 and "'ten' (1 of 3 segments)" in reasons[0]

    monkeypatch.delenv(H.ENV_SCENE_SET)
    assert _bench().invalid_reasons() == []


def test_a_run_with_no_scene_set_configures_exactly_as_it_did(monkeypatch, tmp_path):
    """The regression that would break `--resume` on every recorded sweep.

    A `scene_set` key present with the value `None` would change the config
    JSON of every configuration that has ever run, and so its run id -- the
    runner hashes `run_config()` verbatim into it. The literal below is the
    default configuration as the recorded sweeps in `results/comma2k19/` carry
    it, so a new key here fails this rather than silently orphaning them.
    """
    monkeypatch.delenv(H.ENV_SCENE_SET, raising=False)
    monkeypatch.delenv(H.ENV_INPUT_HZ, raising=False)
    monkeypatch.setattr(H, "SCENES_DIR", tmp_path)  # nothing on disk to find

    assert _bench().run_config() == {
        "history_frames": 160,
        "keyframes_per_segment": 5,
        "eval_horizon_s": 3.0,
        "interval_s": 0.5,
        "gt_horizon_s": 10.0,
        "input_hz": 20.0,
        "calibration": "route-pooled, calibrationd formula on recorded pose",
        "stride": 1,
        "history_span_s": 7.95,
    }


# --------------------------------------------------------------------------
# Reaching one keyframe: `paddock explain`
# --------------------------------------------------------------------------


def test_a_scene_names_its_keyframes_without_opening_a_video(monkeypatch, tmp_path):
    """`scene_sample_ids` is the bridge from the viewer to `paddock explain`.

    The viewer lists segments by `<chunk>/<route_id>/<segment>` and sample ids
    carry the benchmark name in front, so both spellings have to reach the same
    segment -- trimming the keyframe off an id is the obvious way to reach for
    this and must not silently return nothing.
    """
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    monkeypatch.setattr(type(bench), "_n_frames", staticmethod(lambda segment: 1200))

    ids = bench.scene_sample_ids("2/bbb|2018-02-02--00-00-00/7")
    assert len(ids) == 85  # the set's own density
    assert ids[0].startswith("comma2k19/2/bbb|2018-02-02--00-00-00/7/")
    assert bench.scene_sample_ids("comma2k19/2/bbb|2018-02-02--00-00-00/7") == ids
    assert bench.scene_sample_ids("2/bbb|2018-02-02--00-00-00/9") == []


def test_reaching_outside_the_scene_set_is_refused_rather_than_answered(monkeypatch, tmp_path):
    """The quiet way to explain a *different* sample than the one that was scored.

    `RouteCalibrations.get` pools every active segment of a route to estimate
    its calibration and falls back to the segment alone when it finds none, so
    which segments are in `_segments` decides what `camera2ego` a sample gets.
    Answering from `_all_segments` here would hand back an observation built
    from a different calibration under the id of one that was scored -- the
    exact class of defect this command exists to find. So it names the scene
    set instead.
    """
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    outside = "comma2k19/1/aaa|2018-01-01--00-00-00/3/40"

    with pytest.raises(ValueError) as caught:
        list(bench.samples_by_id([outside]))
    message = str(caught.value)
    assert "1/aaa|2018-01-01--00-00-00/3" in message
    assert "outside the active scene set 'ten'" in message
    assert H.ENV_SCENE_SET in message
    assert "different observation under the same sample id" in message


def test_an_id_from_another_dataset_is_simply_not_ours(monkeypatch, tmp_path):
    """Malformed and foreign ids yield nothing; only a *reachable* comma2k19
    segment held out by the scene set is worth raising about."""
    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    assert list(bench.samples_by_id(["nuscenes/scene-0103/12", "comma2k19/nope", ""])) == []
    assert list(bench.samples_by_id([])) == []

    # On disk but not in any scene set: no set is active, so nothing to say.
    monkeypatch.delenv(H.ENV_SCENE_SET)
    plain = _bench()
    plain.__dict__["_all_segments"] = [_ref(k) for k in SITE_KEYS]
    monkeypatch.setattr(type(plain), "_segment_samples", lambda self, segment, cals: iter(()))
    assert list(plain.samples_by_id(["comma2k19/9/zzz|2020-01-01--00-00-00/1/40"])) == []


def test_only_the_named_keyframes_of_only_the_named_segments_are_built(monkeypatch, tmp_path):
    """One keyframe in segment 1,900 must not decode the 1,899 before it."""
    from paddock.benchmark import Sample

    bench = _scene_bench(monkeypatch, tmp_path, "ten", ONE_SCENE)
    key = "2/bbb|2018-02-02--00-00-00/7"
    opened: list[str] = []

    def _fake(self, segment, cals):
        opened.append(segment.sample_key)
        for keyframe in (40, 80, 120):
            yield Sample(sample_id=self._sample_id(segment, keyframe), observations=[])

    monkeypatch.setattr(type(bench), "_segment_samples", _fake)
    got = list(bench.samples_by_id([f"comma2k19/{key}/80", f"comma2k19/{key}/999"]))

    assert [s.sample_id for s in got] == [f"comma2k19/{key}/80"]
    assert opened == [key], "one segment opened, once"
