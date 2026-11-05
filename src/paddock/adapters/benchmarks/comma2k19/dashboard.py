"""One sample as a dashboard: the road on the left, the instruments on the right.

The previous video drew everything on the image -- eleven rows of HUD text, a
legend and a minimap, all painted over the road the viewer was trying to look
at. Text on a driving frame is hard to read wherever it is put: it sits on
moving, textured, mid-grey pixels, and the frames where it matters most are the
ones with the most going on underneath it.

So the image keeps only what is *spatial* -- the recorded path, the radar
boxes, each model's plan -- and everything numeric moves into a panel beside it.
That also buys what an overlay cannot give: the numbers get a **time axis**. A
HUD says the car is doing 18.5 m/s; the speed panel says it has been slowing for
two seconds, which is the fact that explains the plan.

Every panel shares one x axis, in seconds relative to the scored frame, so a
single vertical playhead reads across all of them. Negative is history -- the
frames the model was actually shown -- and the shaded positive half is the
future the plan is scored against, drawn from the recorded pose so that a
viewer can see what the model was reaching for.

Layout is fixed rather than data-driven. A panel that appears only when its
data does would move every other panel between videos, and two runs of the same
model would stop being comparable by eye, which is the whole purpose of drawing
them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.metrics import trajectory as T
from paddock.viz import palette

#: Width of the instrument column, in pixels. The frame beside it is 1164 wide,
#: so this puts the road on about 60 % of a 1924-wide video -- enough panel to
#: read a trend off, without the driving image becoming the small half.
PANEL_W_PX = 760
#: Rendered at the same dpi the figures use, so text is the same size in a still
#: and in a video and the video does not look like a scaled-down figure.
DPI = 120

#: Seconds of recorded future drawn on the time panels, past the scored frame.
#: Matches `harness.N_GT_POSES * INTERVAL_S`, which is what a plan is scored
#: against -- drawing further would show road no metric ever looked at.
FUTURE_S = 10.0

_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#8f8e88"
_SURFACE = "#fcfcfb"
_GRID = "#e5e4e0"
_FUTURE = "#f2f1ec"


@dataclass(frozen=True)
class Scored:
    """One model's stored answer for this sample, as the panels need it.

    `dt` is the model's `waypoint_dt`, which the per-sample record does not
    carry -- it comes from the registry, and defaults to paddock's 0.5 s. It is
    used only to put the plan on a time axis for drawing; the metrics shown were
    computed at run time by `harness.score` and are read back, never recomputed
    here. A figure that recomputed its own numbers could disagree with the
    report beside it.
    """

    model: str
    plan: np.ndarray
    ground_truth: np.ndarray | None
    metrics: dict
    dt: float


def _series(pair) -> tuple[np.ndarray, np.ndarray] | None:
    """A `(t, first column)` CAN series, or None if the segment lacks it."""
    times, values = pair
    if times is None or values is None or not len(times):
        return None
    col = np.asarray(values, dtype=np.float64).reshape(len(values), -1)[:, 0]
    return np.asarray(times, dtype=np.float64), col


def _lead_range(overlay, times: np.ndarray) -> np.ndarray:
    """Range to the closest in-lane radar track at each frame time, else NaN.

    The same `loader.lead_track` definition the overlay draws and `auxiliary`
    scores against, so the trace under the playhead is the same object as the
    red box on the image.
    """
    out = np.full(len(times), np.nan)
    radar_t, radar_v = overlay.radar
    if radar_t is None or radar_v is None:
        return out
    for i, t in enumerate(times):
        lead = L.lead_track(L.radar_tracks(radar_t, radar_v, float(t), overlay.radar_window_s))
        if lead is not None:
            out[i] = float(lead[0])
    return out


def _plan_speed(xy: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """A plan's implied speed profile: waypoint spacing over `dt`.

    The origin is prepended because the first commanded step is measured from
    where the car *is*; without it a plan that opens with a 20 m jump reads as a
    constant speed. Samples land at interval midpoints, which is where a
    difference of two positions actually measures a speed.
    """
    if dt <= 0 or len(xy) < 1:
        return np.empty(0), np.empty(0)
    path = np.vstack([np.zeros((1, 2)), np.asarray(xy, dtype=np.float64)[:, :2]])
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt
    return dt * (np.arange(len(steps)) + 0.5), steps


def _horizon_error(scored: Scored) -> tuple[np.ndarray, np.ndarray]:
    """Displacement error against horizon, on the pair the score was computed on.

    Aligned exactly as `harness.score` aligns it -- by time, with the ego origin
    supplied at t = 0 -- so the curve's value at 3 s is the sample's own
    `fde@3s` and not a near miss of it.
    """
    if scored.ground_truth is None or not len(scored.plan):
        return np.empty(0), np.empty(0)
    gt = np.asarray(scored.ground_truth, dtype=np.float64)[:, :2]
    if not len(gt):
        return np.empty(0), np.empty(0)
    # The recorded future is laid out on the harness's own 0.5 s grid; see
    # `harness.INTERVAL_S`, which the stored record does not repeat.
    gt_t = 0.5 * np.arange(1, len(gt) + 1)
    pred_t = scored.dt * np.arange(1, len(scored.plan) + 1)
    pred, aligned, n = T.align(
        scored.plan,
        pred_t,
        np.vstack([np.zeros((1, 2)), gt]),
        np.concatenate([[0.0], gt_t]),
    )
    if not n:
        return np.empty(0), np.empty(0)
    return scored.dt * np.arange(1, n + 1), np.linalg.norm(pred - aligned, axis=1)


class Dashboard:
    """The instrument column for one sample, drawn once and updated per frame.

    The figure, its axes and every artist are built in `__init__`; `compose`
    moves the playhead and redraws. Rebuilding the figure per frame would be
    correct and about ten times slower, and a 200-frame clip is 200 redraws.
    """

    def __init__(
        self,
        overlay,
        scored: Sequence[Scored],
        keyframe: int,
        window_s: float,
        note: str = "",
        legend: Sequence[tuple[str, str]] = (),
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.overlay = overlay
        self.scored = list(scored)
        self.keyframe = keyframe
        self.t0 = float(overlay.pose.times[keyframe])
        self.times = np.asarray(overlay.pose.times, dtype=np.float64)
        #: Frame times relative to the scored frame: the x axis of every panel.
        self.rel = self.times - self.t0
        self.span = (-max(window_s, 1.0), FUTURE_S)

        height_px = L.EON_HEIGHT_PX
        self.fig = plt.figure(figsize=(PANEL_W_PX / DPI, height_px / DPI), dpi=DPI, facecolor=_SURFACE)
        grid = self.fig.add_gridspec(
            6, 1, height_ratios=[1.0, 2.0, 1.0, 1.0, 1.0, 1.25], hspace=0.55,
            left=0.16, right=0.97, top=0.98, bottom=0.06,
        )  # fmt: skip
        self._header = self.fig.add_subplot(grid[0])
        self._bev = self.fig.add_subplot(grid[1])
        self._speed = self.fig.add_subplot(grid[2])
        self._steer = self.fig.add_subplot(grid[3])
        self._lead = self.fig.add_subplot(grid[4])
        self._error = self.fig.add_subplot(grid[5])

        self._build_header(note, legend)
        self._build_bev()
        self._playheads = [
            self._build_speed(),
            self._build_steer(),
            self._build_lead(),
        ]
        self._build_error()

    # ---- construction ---------------------------------------------------

    def _style(self, ax, ylabel: str) -> None:
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_INK2, labelsize=8)
        ax.set_ylabel(ylabel, color=_INK2, fontsize=9)

    def _time_axis(self, ax, ylabel: str, last: bool = False):
        """A time panel, shaded past t = 0, with a playhead returned."""
        self._style(ax, ylabel)
        ax.set_xlim(*self.span)
        ax.axvspan(0.0, self.span[1], color=_FUTURE, zorder=0)
        ax.axvline(0.0, color=_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        if last:
            ax.set_xlabel("seconds relative to the scored frame", color=_INK2, fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        return ax.axvline(self.span[0], color=_INK, lw=1.4, zorder=6)

    def _build_header(self, note: str, legend: Sequence[tuple[str, str]]) -> None:
        """Title, the frame's own numbers, and a key to what is on the image.

        The key belongs here and not on the frame: `overlay.legend_entries` is
        the same list the drawn-on legend used, so nothing about what the image
        claims has changed -- only where the claim is written.
        """
        ax = self._header
        ax.axis("off")
        ax.set_facecolor(_SURFACE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 1.0, self.overlay.title, color=_INK, fontsize=11, va="top", weight="bold")
        self._status = ax.text(0, 0.76, "", color=_INK2, fontsize=9, va="top", family="monospace")
        row = 0.50
        for label, colour in legend:
            ax.text(0, row, "■", color=colour, fontsize=9, va="top")
            ax.text(0.035, row, label, color=_INK2, fontsize=8.5, va="top")
            row -= 0.175
        if note:
            ax.text(0, row - 0.05, note, color=_MUTED, fontsize=8, va="top")

    def _build_bev(self) -> None:
        """Bird's eye: recorded path ahead, and the plans once they are made.

        Drawn here rather than through `viz.bev` because this panel is updated
        per frame: `viz.bev.draw` creates artists, and creating 200 sets of them
        into one axes is a leak with a picture attached. The conventions it
        fixes -- x forward upward, y left leftward, ground truth underneath --
        are kept deliberately, so the inset reads the same as every BEV in the
        repo.
        """
        ax = self._bev
        self._style(ax, "forward (m)")
        ax.set_xlabel("lateral (m, left positive)", color=_INK2, fontsize=9)
        ax.set_xlim(12, -12)  # inverted: y-left positive reads on the left
        ax.set_ylim(0, 90)
        band, line = palette.ground_truth_style()
        (self._gt_band,) = ax.plot([], [], **band)
        (self._gt_line,) = ax.plot([], [], **line)
        self._plan_lines = {}
        for item in self.scored:
            (artist,) = ax.plot([], [], **palette.style(item.model), label=item.model)
            self._plan_lines[item.model] = artist
        ax.plot(0, 0, "ks", ms=6, zorder=8)
        if self._plan_lines:
            leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
            for text in leg.get_texts():
                text.set_color(_INK2)

    def _build_speed(self):
        """CAN speed leading up to the frame, and each plan's own speed profile.

        The two are different kinds of statement -- one measured, one commanded
        -- so the plan is dashed. A plan that slows where the recorded speed
        does not is the single most legible model error on this benchmark, and
        it is invisible in a HUD that shows one number at a time.
        """
        playhead = self._time_axis(self._speed, "speed (m/s)")
        can = _series(self.overlay.speed)
        if can is not None:
            self._speed.plot(can[0] - self.t0, can[1], color=_INK2, lw=1.6, zorder=4, label="CAN (recorded)")
        else:
            self._speed.plot(self.rel, self.overlay.pose.speed, color=_INK2, lw=1.6, zorder=4, label="pose")
        for item in self.scored:
            t, v = _plan_speed(item.plan, item.dt)
            if len(t):
                self._speed.plot(t, v, **palette.style(item.model, ls=(0, (3, 2)), zorder=5))
        leg = self._speed.legend(frameon=False, fontsize=8, loc="lower left")
        for text in leg.get_texts():
            text.set_color(_INK2)
        return playhead

    def _build_steer(self):
        playhead = self._time_axis(self._steer, "steering (deg)")
        steer = _series(self.overlay.steer)
        if steer is not None:
            self._steer.plot(steer[0] - self.t0, steer[1], color=_INK2, lw=1.4, zorder=4)
        return playhead

    def _build_lead(self):
        """Range to the closest in-lane radar track: the thing an ACC must see.

        Gaps are real and are left as gaps. Interpolating across a stretch with
        no lead would draw a car that was not there, which is precisely the
        error this panel exists to make visible.
        """
        playhead = self._time_axis(self._lead, "lead range (m)", last=True)
        ranges = _lead_range(self.overlay, self.times)
        self._lead.plot(self.rel, ranges, color=_INK2, lw=1.4, zorder=4)
        if np.isfinite(ranges).any():
            self._lead.set_ylim(0, float(np.nanmax(ranges)) * 1.15 + 1.0)
        else:
            self._lead.set_ylim(0, 1)
            self._lead.text(
                0.5, 0.5, "no lead in lane", transform=self._lead.transAxes,
                ha="center", va="center", color=_MUTED, fontsize=9,
            )  # fmt: skip
        return playhead

    def _build_error(self) -> None:
        """Displacement error against horizon -- the scored number, as a curve.

        Static: it is a statement about the scored frame and does not change as
        the clip plays. It is on screen throughout anyway, because the point of
        the video is to watch the road that produced it.
        """
        ax = self._error
        self._style(ax, "error (m)")
        ax.set_xlabel("plan horizon (s)", color=_INK2, fontsize=9)
        drew = False
        for item in self.scored:
            t, err = _horizon_error(item)
            if not len(t):
                continue
            drew = True
            ax.plot(t, err, **palette.style(item.model), label=_label(item))
        if drew:
            ax.set_xlim(0, FUTURE_S)
            leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
            for text in leg.get_texts():
                text.set_color(_INK2)
        else:
            ax.set_xlim(0, FUTURE_S)
            ax.text(
                0.5, 0.5, "no scored plan for this sample", transform=ax.transAxes,
                ha="center", va="center", color=_MUTED, fontsize=9,
            )  # fmt: skip

    # ---- per frame ------------------------------------------------------

    def compose(self, drawn: np.ndarray, index: int, show_plans: bool) -> np.ndarray:
        """The drawn frame with this frame's instrument column beside it."""
        rel = float(self.times[index] - self.t0)
        for playhead in self._playheads:
            playhead.set_xdata([rel, rel])
        self._status.set_text(self._status_text(index, rel))
        self._update_bev(index, show_plans)
        return np.hstack([np.asarray(drawn, dtype=np.uint8), self._panel(drawn.shape[0])])

    def _status_text(self, index: int, rel: float) -> str:
        speed = float(self.overlay.pose.speed[index])
        when = "scored frame" if index == self.keyframe else f"{rel:+.2f} s"
        return f"frame {index:5d}   {when:>14}   {speed:5.1f} m/s"

    def _update_bev(self, index: int, show_plans: bool) -> None:
        future = self.overlay.future_device(index)
        if future is not None and len(future):
            # The plans beside it are in the car's frame, and so is the ground
            # truth they are scored against (`loader.future_in_ego_frame`), so
            # the recorded path is rotated out of the camera's frame by the same
            # calibration before it is drawn. Without it the path leans off by
            # the mount yaw and a plan that matches the score looks wrong here.
            rpy = getattr(self.overlay, "rpy", None)
            if rpy is not None:
                calib_from_device = L.rot_from_euler(np.asarray(rpy, dtype=np.float64)).T
                future = np.asarray(future, dtype=np.float64) @ calib_from_device.T
            # Device frame is y-right; the BEV, like every ego-frame quantity in
            # paddock, is y-left.
            self._gt_band.set_data(-future[:, 1], future[:, 0])
            self._gt_line.set_data(-future[:, 1], future[:, 0])
        else:
            self._gt_band.set_data([], [])
            self._gt_line.set_data([], [])
        for item in self.scored:
            artist = self._plan_lines[item.model]
            if show_plans and len(item.plan):
                xy = np.asarray(item.plan, dtype=np.float64)
                artist.set_data(xy[:, 1], xy[:, 0])
            else:
                artist.set_data([], [])

    def _panel(self, height: int) -> np.ndarray:
        self.fig.canvas.draw()
        rgb = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
        if rgb.shape[0] == height:
            return rgb
        # Only reachable if a frame is not the EON's own height; pad or crop
        # rather than resample, so the panel's text never goes soft.
        if rgb.shape[0] > height:
            return rgb[:height]
        pad = np.full((height - rgb.shape[0], rgb.shape[1], 3), 252, dtype=np.uint8)
        return np.vstack([rgb, pad])

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def _label(item: Scored) -> str:
    """Model name plus the numbers the run recorded for this sample."""
    parts = [
        f"{key} {item.metrics[key]:.2f}"
        for key in ("ade", "fde@3s")
        if isinstance(item.metrics.get(key), (int, float)) and np.isfinite(item.metrics[key])
    ]
    return f"{item.model}   " + "  ".join(parts) if parts else item.model


def frames(
    dash: Dashboard,
    drawn: Iterator[tuple[int, np.ndarray, bool]],
) -> Iterator[np.ndarray]:
    """Compose an already-overlaid stream of `(index, rgb, show_plans)`."""
    for index, rgb, show_plans in drawn:
        yield dash.compose(rgb, index, show_plans)
