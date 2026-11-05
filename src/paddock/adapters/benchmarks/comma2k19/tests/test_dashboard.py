"""Tests for `comma2k19.dashboard`.

The dashboard exists because numbers painted on a driving frame are hard to
read, so most of what is worth testing is whether a number it draws is the same
number the run recorded. `_horizon_error` is the sharp case: it re-aligns the
stored plan against the stored future, and if it aligns it even slightly
differently from `harness.score` then the curve on the video and the `fde@3s` in
the report are about different things. The test below pins them together.

The rest guards the two structural promises in the module docstring -- a fixed
layout regardless of what data exists, and a panel that is padded or cropped
rather than resampled -- plus the gap handling in the lead trace, where
interpolating would draw a car that was not there.

The pure helpers need only numpy and run everywhere; anything that builds a
figure asks for matplotlib first, as `tests/test_viz.py` does.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import dashboard as DB
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.metrics import trajectory as T

HZ = 20.0


def _radar_row(d: float, y: float = 0.0, addr: float = 1.0) -> list[float]:
    """One radar row in the dataset's column order; see `loader.lead_track`."""
    return [d, y, 0.0, 0.0, 0.0, addr, 0.0]


class _Overlay:
    """Just the attributes `Dashboard` reads, so a test needs no segment.

    A real `SegmentOverlay` needs a route on disk, a toolkit clone and a video;
    none of that says anything about whether the panels draw the right numbers.
    """

    def __init__(self, n: int = 60, speed: float = 20.0, can: bool = True, radar=(None, None)):
        t = np.arange(n) / HZ
        self.pose = types.SimpleNamespace(times=t, speed=np.full(n, speed))
        self.title = "b0c9 seg 3"
        self.speed = (t, np.full((n, 1), speed)) if can else (None, None)
        self.steer = (t, np.zeros((n, 1))) if can else (None, None)
        self.radar = radar
        self.radar_window_s = 0.1
        self.n = n

    def future_device(self, index: int):
        """Straight ahead at 1 m per row, device frame [forward, right, down]."""
        ahead = np.arange(20, dtype=np.float64)
        return np.column_stack([ahead, np.full_like(ahead, 0.5), np.zeros_like(ahead)])


def _scored(model="openpilot", plan=None, gt=None, metrics=None, dt=0.5) -> DB.Scored:
    return DB.Scored(
        model=model,
        plan=np.zeros((0, 2)) if plan is None else np.asarray(plan, dtype=np.float64),
        ground_truth=None if gt is None else np.asarray(gt, dtype=np.float64),
        metrics=metrics or {},
        dt=dt,
    )


def _straight(n: int, step: float) -> np.ndarray:
    """`n` waypoints straight ahead, `step` metres apart."""
    return np.column_stack([step * np.arange(1, n + 1), np.zeros(n)])


# --------------------------------------------------------------------------
# The numbers, which must be the run's numbers
# --------------------------------------------------------------------------


def test_the_error_curve_is_the_scored_number_and_not_a_near_miss():
    """The coupling that matters: the curve's value at 3 s *is* `fde@3s`.

    Computed here through `harness.score`'s own alignment -- origin at t = 0,
    paired by time -- so a change to either side that drifts from the other
    fails rather than quietly putting two different numbers on one screen.
    """
    gt = _straight(20, 10.0)
    plan = _straight(20, 9.0)  # consistently short, so the error grows
    item = _scored(plan=plan, gt=gt, dt=0.5)

    pred_a, gt_a, _ = T.align(
        plan,
        0.5 * np.arange(1, len(plan) + 1),
        np.vstack([np.zeros((1, 2)), gt]),
        np.concatenate([[0.0], 0.5 * np.arange(1, len(gt) + 1)]),
    )
    expected = MC.score_trajectory(pred_a, gt_a, interval_s=0.5, eval_horizon_s=3.0)

    t, err = DB._horizon_error(item)
    at_3s = err[int(np.argmin(np.abs(t - 3.0)))]
    assert math.isclose(at_3s, expected["fde@3s"], rel_tol=1e-9)
    assert math.isclose(float(np.mean(err[t <= 3.0 + 1e-9])), expected["ade@3s"], rel_tol=1e-9)


def test_the_error_curve_stops_where_the_recorded_future_does():
    """A 10 s plan against a 3 s future is scored over 3 s, not padded to 10."""
    t, err = DB._horizon_error(_scored(plan=_straight(20, 10.0), gt=_straight(6, 10.0)))
    assert len(t) == len(err) == 6
    assert math.isclose(t[-1], 3.0)


def test_a_sample_with_no_recorded_future_draws_nothing_rather_than_zeros():
    """An all-zero error curve would read as a perfect plan."""
    t, err = DB._horizon_error(_scored(plan=_straight(4, 10.0), gt=None))
    assert len(t) == len(err) == 0


def test_a_plans_speed_is_measured_from_where_the_car_is():
    """Without the prepended origin, a plan opening with a 10 m step reads as a
    constant speed from the second waypoint on and says nothing about the first."""
    t, v = DB._plan_speed(_straight(4, 10.0), dt=0.5)
    assert v == pytest.approx(np.full(4, 20.0))
    # Midpoints: a difference of two positions measures a speed between them.
    assert t == pytest.approx([0.25, 0.75, 1.25, 1.75])


def test_a_plan_that_brakes_shows_a_falling_speed():
    t, v = DB._plan_speed(np.column_stack([[10.0, 18.0, 22.0], [0.0, 0.0, 0.0]]), dt=0.5)
    assert list(v) == pytest.approx([20.0, 16.0, 8.0])
    assert len(t) == 3


@pytest.mark.parametrize("dt", [0.0, -0.5])
def test_a_plan_with_no_time_base_has_no_speed_profile(dt):
    t, v = DB._plan_speed(_straight(3, 10.0), dt=dt)
    assert len(t) == len(v) == 0


# --------------------------------------------------------------------------
# The CAN and radar traces
# --------------------------------------------------------------------------


def test_a_missing_can_channel_is_none_rather_than_an_empty_plot():
    assert DB._series((None, None)) is None
    assert DB._series((np.empty(0), np.empty((0, 1)))) is None


def test_a_can_series_takes_the_first_column_because_that_is_the_signal():
    t, v = DB._series((np.array([0.0, 0.5]), np.array([[3.0, 99.0], [4.0, 99.0]])))
    assert list(t) == [0.0, 0.5]
    assert list(v) == [3.0, 4.0]


def test_the_lead_trace_leaves_gaps_where_there_was_no_lead():
    """Interpolating across them would draw a car that was not there, which is
    the error this panel exists to make visible."""
    overlay = _Overlay(n=40)
    times = overlay.pose.times
    # A lead for the first second only, closing from 40 m.
    radar_t = times[:20]
    radar_v = np.array([_radar_row(40.0 - i * 0.5) for i in range(20)])
    ranges = DB._lead_range(_Overlay(n=40, radar=(radar_t, radar_v)), times)
    # 39.5 and not 40.0: `loader.radar_tracks` keeps the *latest* row per track
    # address in the window, which is what makes a count of them a track count.
    assert math.isclose(ranges[0], 39.5)
    assert np.isfinite(ranges[:19]).all()
    assert np.isnan(ranges[-5:]).all()


def test_a_track_outside_the_ego_lane_is_not_the_lead():
    """One definition of "the lead", shared with the overlay and `auxiliary`."""
    overlay = _Overlay(n=10)
    times = overlay.pose.times
    radar_v = np.array([_radar_row(30.0, y=L.HALF_LANE_M + 0.5) for _ in times])
    ranges = DB._lead_range(_Overlay(n=10, radar=(times, radar_v)), times)
    assert np.isnan(ranges).all()


def test_a_segment_with_no_radar_gives_an_all_nan_trace_rather_than_raising():
    overlay = _Overlay(n=10)
    assert np.isnan(DB._lead_range(overlay, overlay.pose.times)).all()


# --------------------------------------------------------------------------
# The legend label
# --------------------------------------------------------------------------


def test_the_label_carries_the_recorded_numbers_when_there_are_any():
    label = DB._label(_scored(metrics={"ade": 1.234, "fde@3s": 2.5}))
    assert label.startswith("openpilot")
    assert "ade 1.23" in label and "fde@3s 2.50" in label


def test_a_model_with_no_finite_metrics_is_named_and_nothing_more():
    assert DB._label(_scored(metrics={"ade": float("nan")})) == "openpilot"
    assert DB._label(_scored(metrics={})) == "openpilot"


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


def _dash(overlay=None, scored=(), keyframe=40, **kwargs):
    overlay = overlay if overlay is not None else _Overlay(n=60)
    return DB.Dashboard(overlay, scored, keyframe=keyframe, window_s=2.0, **kwargs)


def test_the_layout_is_the_same_whether_or_not_there_is_data_to_fill_it(mpl):
    """The module's own claim: a panel that appeared only when its data did
    would move every other panel between videos."""
    bare = _dash(overlay=_Overlay(n=60, can=False))
    full = _dash(scored=[_scored(plan=_straight(20, 10.0), gt=_straight(20, 10.0))])
    try:
        assert len(bare.fig.axes) == len(full.fig.axes) == 6
        assert bare.fig.get_size_inches().tolist() == full.fig.get_size_inches().tolist()
    finally:
        bare.close()
        full.close()


def test_a_composed_frame_is_the_road_plus_one_panel_width(mpl):
    dash = _dash()
    try:
        drawn = np.zeros((L.EON_HEIGHT_PX, L.EON_WIDTH, 3), dtype=np.uint8)
        out = dash.compose(drawn, 40, show_plans=True)
        assert out.shape == (L.EON_HEIGHT_PX, L.EON_WIDTH + DB.PANEL_W_PX, 3)
        assert out.dtype == np.uint8
        # The road is untouched: composition adds beside, it does not draw over.
        assert (out[:, : L.EON_WIDTH] == drawn).all()
    finally:
        dash.close()


@pytest.mark.parametrize("height", [400, 1000])
def test_an_odd_sized_frame_pads_or_crops_the_panel_rather_than_scaling_it(mpl, height):
    """Resampling would make the panel's text soft, which is the one thing the
    panel is for."""
    dash = _dash()
    try:
        out = dash.compose(np.zeros((height, 100, 3), dtype=np.uint8), 40, show_plans=False)
        assert out.shape == (height, 100 + DB.PANEL_W_PX, 3)
    finally:
        dash.close()


def test_the_playhead_follows_the_frame_and_reads_zero_at_the_scored_one(mpl):
    dash = _dash(keyframe=40)
    try:
        frame = np.zeros((L.EON_HEIGHT_PX, 8, 3), dtype=np.uint8)
        dash.compose(frame, 20, show_plans=False)
        early = [float(p.get_xdata()[0]) for p in dash._playheads]
        assert early == pytest.approx([-1.0] * len(early))
        dash.compose(frame, 40, show_plans=True)
        assert [float(p.get_xdata()[0]) for p in dash._playheads] == pytest.approx([0.0] * len(early))
        assert "scored frame" in dash._status.get_text()
        assert "20.0 m/s" in dash._status.get_text()
    finally:
        dash.close()


def test_a_plan_is_drawn_only_once_the_model_has_made_it(mpl):
    """The plans belong to the scored frame; drawing them over the history would
    claim the model predicted that road before it had seen it."""
    item = _scored(plan=_straight(3, 10.0), gt=_straight(3, 10.0))
    dash = _dash(scored=[item])
    frame = np.zeros((L.EON_HEIGHT_PX, 8, 3), dtype=np.uint8)
    try:
        dash.compose(frame, 20, show_plans=False)
        assert len(dash._plan_lines["openpilot"].get_xdata()) == 0
        dash.compose(frame, 40, show_plans=True)
        x, y = dash._plan_lines["openpilot"].get_data()
        # BEV convention: lateral (y-left) across, forward up.
        assert list(y) == pytest.approx([10.0, 20.0, 30.0])
        assert list(x) == pytest.approx([0.0, 0.0, 0.0])
    finally:
        dash.close()


def test_the_recorded_future_is_flipped_out_of_the_device_frame(mpl):
    """Device is y-right; every ego-frame quantity in paddock is y-left."""
    dash = _dash()
    try:
        dash.compose(np.zeros((L.EON_HEIGHT_PX, 8, 3), dtype=np.uint8), 40, show_plans=True)
        x, y = dash._gt_line.get_data()
        assert list(x) == pytest.approx([-0.5] * len(x))  # 0.5 m right -> -0.5 left
        assert list(y) == pytest.approx(list(range(len(y))))
    finally:
        dash.close()


def test_the_recorded_future_is_drawn_in_the_cars_frame_like_the_plans(mpl, rotations):
    """The plans beside it, and the ground truth they are scored against, are
    in the car's frame; a path drawn in the camera's leans off by the mount yaw.

    A camera yawed 1.5 deg off the car's axis sees a straight drive bear to its
    right. Rotated by the overlay's calibration, the drawn path is straight.
    """
    yaw = np.radians(1.5)

    class _Mounted(_Overlay):
        rpy = np.array([0.0, 0.0, yaw])

        def future_device(self, index: int):
            ahead = np.arange(20, dtype=np.float64)
            return np.column_stack([ahead * np.cos(yaw), ahead * np.sin(yaw), np.zeros_like(ahead)])

    dash = _dash(overlay=_Mounted(n=60))
    try:
        dash.compose(np.zeros((L.EON_HEIGHT_PX, 8, 3), dtype=np.uint8), 40, show_plans=True)
        x, y = dash._gt_line.get_data()
        assert list(x) == pytest.approx([0.0] * len(x), abs=1e-9)
        assert list(y) == pytest.approx(list(range(len(y))))
    finally:
        dash.close()


def test_frames_composes_one_output_per_input(mpl):
    dash = _dash()
    try:
        drawn = ((i, np.zeros((L.EON_HEIGHT_PX, 8, 3), dtype=np.uint8), i == 40) for i in (20, 30, 40))
        out = list(DB.frames(dash, drawn))
        assert len(out) == 3
        assert {frame.shape for frame in out} == {(L.EON_HEIGHT_PX, 8 + DB.PANEL_W_PX, 3)}
    finally:
        dash.close()
