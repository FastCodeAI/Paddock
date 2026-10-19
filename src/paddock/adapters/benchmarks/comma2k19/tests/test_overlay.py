"""Tests for the overlay: the frame conversion, the drawing, and the video.

The one that earns its place is `test_a_plan_to_the_left_is_drawn_to_the_left`.
A plan arrives in paddock's ego frame (y **left**) and the toolkit projects
comma2k19's device frame (y **right**), so the conversion is a sign -- and a
wrong sign mirrors the drawn path about the direction of travel, raises nothing,
and on a straight road is invisible. It would make every overlay quietly lie
about which way a model wanted to go, which is the one thing this feature exists
to answer.
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import overlay as OV
from paddock.adapters.benchmarks.comma2k19 import render as R
from paddock.adapters.benchmarks.comma2k19.tests.synthetic import straight_pose, write_segment

# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


def test_the_ego_device_conversion_is_its_own_inverse():
    """Needs no toolkit: it is `diag(1, -1, -1)` twice.

    Worth asserting anyway, because `device_from_ego` exists *only* because
    reading `ego_from_device` at an overlay call site and having to remember it
    is an involution is how a direction gets flipped during a later edit.
    """
    points = np.array([[10.0, 2.0, 0.5], [0.0, -3.0, -1.0]])
    assert np.allclose(L.device_from_ego(L.ego_from_device(points)), points)
    assert np.allclose(L.device_from_ego(points), points * [1.0, -1.0, -1.0])


def test_a_plan_on_the_road_plane_is_dropped_to_the_road(toolkit):
    """Ego z = 0 is at the camera; the road is `EON_HEIGHT` below it."""
    road = OV.ego_to_road([[10.0, 0.0]])
    assert np.allclose(road, [[10.0, 0.0, L.EON_HEIGHT]])
    # device z is *down*, so the road plane is a positive z. Getting this
    # backwards draws the plan on the ceiling.
    assert road[0, 2] > 0


def test_a_plan_to_the_left_is_drawn_to_the_left(toolkit):
    """The sign test. See the module docstring for why it is the important one."""
    centre = OV.project(OV.ego_to_road([[20.0, 0.0]]))[0]
    left = OV.project(OV.ego_to_road([[20.0, 3.0]]))[0]
    right = OV.project(OV.ego_to_road([[20.0, -3.0]]))[0]

    assert np.isclose(centre[0], L.EON_WIDTH / 2, atol=1.0), centre
    assert left[0] < centre[0] < right[0], (left, centre, right)
    # And symmetric about the centre, which a stray transpose would break.
    assert np.isclose(centre[0] - left[0], right[0] - centre[0], atol=1e-6)


def test_the_road_is_drawn_below_the_horizon(toolkit):
    """A point on the road ahead is below the image centre, and nearer is lower."""
    near = OV.project(OV.ego_to_road([[8.0, 0.0]]))[0]
    far = OV.project(OV.ego_to_road([[60.0, 0.0]]))[0]
    assert near[1] > L.EON_HEIGHT_PX / 2, near
    assert near[1] > far[1] > L.EON_HEIGHT_PX / 2, (near, far)


def test_a_radar_range_is_drawn_at_the_range_it_reports(toolkit):
    """No radar-to-camera offset -- see `radar_to_device` for the measurement.

    Pinned because openpilot's `RADAR_TO_CAMERA = 1.52` and opendbc's "from
    front of car" comment both argue for adding one, so this is the kind of
    correction someone reads the source and puts back. On real frames it makes
    every box about 1.5 m too far: 244 px of predicted lead width becomes 199 px
    against a car that measures 244.
    """
    box = OV.radar_to_device(40.0, 0.0, rpy=None)
    assert np.allclose(box[:, 0], 40.0)
    # The box stands on the road, so its base is the road plane and its top is
    # 1.5 m above it -- device z is down, so the top is the *smaller* z.
    assert box[:, 2].max() == pytest.approx(L.EON_HEIGHT)
    assert box[:, 2].min() == pytest.approx(L.EON_HEIGHT - 1.5)
    # Left-positive in, right-positive out: a return to the left of the car
    # must not be drawn to its right.
    assert OV.radar_to_device(40.0, 2.0, rpy=None)[:, 1].mean() == pytest.approx(-2.0)


def test_the_mount_rotation_moves_a_radar_box_by_the_pitch_it_encodes(toolkit):
    """The bug this was written for: radar is car-frame, the camera is not.

    comma2k19's mount pitch runs 1.0-3.8 deg across the two vehicles. At fl 910
    that is 16-61 px of vertical error on every box -- far more than the gap
    between a lead vehicle's bumper and the road under it, which is why the
    box looked "slightly off" against the car it was drawn on.

    The check is the first-order one: a pure pitch moves the projected box by
    `focal * pitch` rows and leaves the columns alone.
    """
    pitch = np.radians(3.0)
    uncorrected = OV.project(OV.radar_to_device(40.0, 0.0, rpy=None), clip=False)
    corrected = OV.project(OV.radar_to_device(40.0, 0.0, rpy=[0.0, pitch, 0.0]), clip=False)

    shift = np.mean(corrected[:, 1] - uncorrected[:, 1])
    assert abs(shift) == pytest.approx(L.EON_FOCAL * pitch, rel=0.05), shift
    assert np.allclose(corrected[:, 0], uncorrected[:, 0], atol=1.0), "pitch must not move columns"

    # And a route with no estimate draws what it always drew, rather than
    # silently applying a zero that looks like a measurement.
    assert np.allclose(OV.calib_to_device([[1.0, 2.0, 3.0]], None), [[1.0, 2.0, 3.0]])
    assert np.allclose(OV.calib_to_device([[1.0, 2.0, 3.0]], [0.0, 0.0, 0.0]), [[1.0, 2.0, 3.0]], atol=1e-12)


def test_a_plan_is_drawn_in_the_frame_the_radar_is_drawn_in(toolkit):
    """Plans are car-frame too: `camera2ego` carries the mount rotation.

    `loader.camera2ego_from_rpy` builds paddock's ego frame on this benchmark
    out of the route calibration, so a model plans in the same car-aligned frame
    a radar return is measured in, and needs the same rotation to be drawn.

    Stated as the invariant rather than a pixel count: a waypoint and a radar
    return at the same place on the road are the same device-frame point,
    whatever the mount.
    """
    rpy = np.radians([0.4, 3.6, -1.8])
    waypoint = OV.plan_to_device([[40.0, 2.0]], rpy)[0]
    # The first two corners are the box's base edge, whose midpoint is the
    # return itself, on the road plane.
    base = OV.radar_to_device(40.0, 2.0, rpy=rpy)[:2].mean(axis=0)
    assert np.allclose(waypoint, base, atol=1e-9), (waypoint, base)
    # With no estimate, a plan is drawn exactly where it always was.
    assert np.allclose(OV.plan_to_device([[40.0, 2.0]], None), OV.ego_to_road([[40.0, 2.0]]))


def test_a_plan_behind_the_camera_is_not_drawn(toolkit):
    """`img_from_device` NaNs negative depth, and the drawing must honour it.

    Without this a point behind the car projects through the origin and lands
    somewhere plausible-looking in the upper half of the image.
    """
    assert not np.isfinite(OV.project(OV.ego_to_road([[-10.0, 0.0]]))).all()


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


@pytest.fixture
def segment(tmp_path):
    """A synthetic segment with pose and nothing else -- no CAN, no GNSS.

    Deliberately incomplete: a partial segment is the common case at a site that
    fetched one chunk, and every optional log the HUD reads must degrade to
    "n/a" rather than raising halfway through a video.
    """
    return write_segment(tmp_path / "seg", straight_pose(n=60, speed=25.0))


def test_a_frame_is_drawn_without_changing_its_shape(toolkit, segment):
    pytest.importorskip("PIL")
    overlay = OV.SegmentOverlay(segment)
    frame = np.zeros((L.EON_HEIGHT_PX, L.EON_WIDTH, 3), dtype=np.uint8)
    drawn = overlay.draw(frame, 10, plans=[OV.Plan.for_model("openpilot", [[5.0, 0.0], [10.0, 0.5]])])

    assert drawn.shape == frame.shape
    assert drawn.dtype == np.uint8
    assert drawn.any(), "nothing was drawn at all"
    # The recorded path, the plan, the HUD and the minimap each land in a
    # different part of the frame; if any layer silently did nothing the count
    # of touched columns collapses.
    assert (drawn.any(axis=(0, 2))).sum() > L.EON_WIDTH // 2


def test_the_overlay_draws_a_plan_through_the_mount_rotation(toolkit, segment):
    """`draw` must use the rotation, not merely have `plan_to_device` exist.

    Drawn without it, every plan sat off the road it was planned on by the
    mount's yaw and pitch, which reads as a model that drifts or plans short.

    A pure yaw moves a point straight ahead sideways by `focal * tan(yaw)`
    whatever its depth, so the whole drawn plan moves by that many columns --
    rightwards for a positive yaw, the sign `calib_to_device` gives the radar.
    """
    pytest.importorskip("PIL")
    frame = np.zeros((L.EON_HEIGHT_PX, L.EON_WIDTH, 3), dtype=np.uint8)
    ahead = [[x, 0.0] for x in range(10, 50, 5)]
    plans_only = OV.Layers(path=False, radar=False, hud=False, minimap=False)
    yaw = np.radians(2.0)

    def column(overlay_rpy, plan_rpy=None) -> float:
        overlay = OV.SegmentOverlay(segment, layers=plans_only, rpy=overlay_rpy)
        drawn = overlay.draw(frame, 10, plans=[OV.Plan.for_model("openpilot", ahead, rpy=plan_rpy)])
        return float(np.nonzero(drawn.any(axis=2))[1].mean())

    level = column(np.zeros(3))
    assert level == pytest.approx(L.EON_WIDTH / 2, abs=1.0)
    assert column([0.0, 0.0, yaw]) - level == pytest.approx(L.EON_FOCAL * np.tan(yaw), abs=1.0)
    # A plan that says which calibration its model was handed is drawn with
    # that, whatever the overlay estimated for the radar.
    assert column(np.zeros(3), plan_rpy=[0.0, 0.0, yaw]) == pytest.approx(column([0.0, 0.0, yaw]), abs=0.5)


def test_a_missing_log_degrades_to_n_a_rather_than_raising(toolkit, segment):
    pytest.importorskip("PIL")
    overlay = OV.SegmentOverlay(segment)
    lines = dict(overlay._hud_lines(5, float(overlay.pose.times[5]), np.empty((0, 7)), None))
    assert lines["CAN speed"] == "n/a"
    assert lines["lead"] == "none in lane"
    # The pose is always there, so this one is a number even on a bare segment.
    assert "m/s (pose)" in lines["ego speed"]


def test_the_model_keeps_its_colour_here_too():
    """A plan is `viz.palette`'s colour, not a colour this module chose."""
    from paddock.viz import palette

    plan = OV.Plan.for_model("openpilot", [[1.0, 0.0]])
    assert plan.color == palette.color("openpilot")
    assert plan.label == "openpilot"


def test_the_font_needs_no_path_into_this_machine(monkeypatch):
    """`tests/test_portability.py` bans the hardcoded .ttf the script used."""
    monkeypatch.delenv(OV.ENV_FONT, raising=False)
    pytest.importorskip("PIL")
    assert OV.font(15) is not None


def test_radar_keeps_one_row_per_track(toolkit):
    """A track is re-reported at ~15 Hz; the HUD count must be tracks, not rows."""
    times = np.array([0.98, 0.99, 1.00, 1.01])
    values = np.array(
        [
            [30.0, 0.0, -1.0, np.nan, np.nan, 7.0, 0.0],
            [22.0, 3.0, +0.5, np.nan, np.nan, 8.0, 0.0],
            [29.5, 0.0, -1.0, np.nan, np.nan, 7.0, 0.0],  # track 7 again
            [21.8, 3.0, +0.5, np.nan, np.nan, 8.0, 0.0],
        ]
    )
    rows = OV.radar_tracks(times, values, 1.0, window=0.05)
    assert len(rows) == 2
    assert OV.lead_track(rows)[5] == 7.0  # the in-lane one, not the closer off-lane one


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


def test_encode_frames_writes_something_a_decoder_can_read(tmp_path):
    try:
        L.probe_decoder()
    except RuntimeError as exc:
        pytest.skip(f"no video backend here: {exc}")

    frames = [np.full((64, 96, 3), i * 20, dtype=np.uint8) for i in range(1, 6)]
    out = L.encode_frames(frames, tmp_path / "clip.mp4", fps=10.0)
    assert out.exists() and out.stat().st_size > 0
    assert len(list(L.decode_frames(out, width=96, height=64))) == len(frames)


def test_encoding_nothing_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        L.encode_frames([], tmp_path / "empty.mp4")


# --------------------------------------------------------------------------
# render_sample
# --------------------------------------------------------------------------


class _Run:
    def __init__(self, model, samples):
        self.model, self.samples = model, samples


def test_a_sample_id_that_is_not_there_names_what_is(tmp_path):
    runs = [_Run("openpilot", [{"sample_id": "comma2k19/1/abc|t/40/300", "metrics": {}}])]
    with pytest.raises(LookupError) as exc:
        R.render_sample(runs, "comma2k19/1/abc|t/40/999", tmp_path, cfg=None)
    assert "comma2k19/1/abc|t/40/300" in str(exc.value)


def test_a_run_with_no_stored_samples_says_where_they_went(tmp_path):
    with pytest.raises(LookupError, match="cleaned up"):
        R.render_sample([_Run("openpilot", [])], "comma2k19/1/abc|t/40/300", tmp_path, cfg=None)


def test_the_sample_id_may_be_given_without_the_benchmark_prefix():
    record = {"sample_id": "comma2k19/1/abc|t/40/300"}
    run = _Run("openpilot", [record])
    assert R._record(run, "1/abc|t/40/300") is record
    assert R._record(run, "comma2k19/1/abc|t/40/300") is record
    assert R._record(run, "1/abc|t/40/301") is None


def test_a_plan_is_drawn_with_the_calibration_its_run_was_given():
    """The record's `rpy_calib_deg`, not a second estimate of the route's.

    The harness handed each model a calibration and stored it with the sample,
    and the plan is in the frame that calibration defines -- so it is the one to
    draw the plan back with, run by run. Estimating again would also be a
    pooled pass over the whole route before the first frame is drawn.
    """
    assert np.allclose(R._calibration({"rpy_calib_deg": [0.1, 3.6, -1.8]}), np.radians([0.1, 3.6, -1.8]))
    # A record from before the harness stored it leaves the overlay to estimate.
    assert R._calibration({"chunk": 1}) is None


def test_the_legend_carries_the_number_the_plan_scored():
    label = R._plan_label(_Run("openpilot", []), {"metrics": {"ade": 1.234, "fde@3s": 2.0, "ade@5s": 9.9}})
    assert label.startswith("openpilot plan")
    assert "ade 1.23" in label and "fde@3s 2.00" in label
    # Only the two headline numbers: a legend entry is one line on an image.
    assert "ade@5s" not in label


def _chatty_ffmpeg(tmp_path, frames: int, noise_kb: int) -> str:
    """A stand-in ffmpeg that talks as much as a damaged HEVC stream does.

    Real ffmpeg emits a line per undecodable frame, so a segment's worth is
    roughly 96 KB -- comfortably past the 64 KB a pipe holds. This writes its
    complaints first and its frames second, which is the order that matters: a
    decoder blocked on stderr never reaches stdout, and a reader waiting on
    stdout never drains stderr.
    """
    script = tmp_path / "ffmpeg-stub"
    script.write_text(
        "#!/bin/sh\n"
        f"exec {sys.executable} -c '\n"
        "import sys\n"
        f'sys.stderr.buffer.write(b"[hevc @ 0x0] Could not find ref with POC\\n" * {noise_kb * 25})\n'
        "sys.stderr.flush()\n"
        f"sys.stdout.buffer.write(bytes({frames} * 8 * 4 * 3))\n"
        "'\n"
    )
    script.chmod(0o755)
    return str(script)


def test_a_decoder_that_complains_about_every_frame_does_not_wedge_the_sweep(tmp_path, monkeypatch):
    """The deadlock that had no timeout anywhere to end it.

    `stderr=PIPE` was drained only once stdout reached EOF. This harness expects
    truncated video, a damaged stream complains once per frame, and ~96 KB does
    not fit in a 64 KB pipe -- so ffmpeg blocked writing, `stdout.read()` blocked
    waiting, and four pinned nodes sat there until somebody noticed.
    """
    monkeypatch.setattr(L, "probe_decoder", lambda: "ffmpeg")
    monkeypatch.setattr(L, "_ffmpeg", lambda: _chatty_ffmpeg(tmp_path, frames=3, noise_kb=256))

    decoded: list[int] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            decoded.append(len(list(L.decode_frames(tmp_path / "video.hevc", width=8, height=4))))
        except BaseException as exc:  # noqa: BLE001 - reported through `error`, not swallowed
            error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30.0)
    assert not worker.is_alive(), "decode_frames deadlocked on ffmpeg's stderr"
    assert not error, error
    assert decoded == [3]
