"""Draw a comma2k19 frame with its own labels -- and with what a model planned.

One drawing engine, two callers that want the same picture for different
reasons:

* `render.render_sample` -- one *scored* sample, with each run's stored plan over
  the recorded future, to see why a number came out the way it did.
* `tools/benchmarks/comma2k19/overlay_video.py` -- any segment, no model and no
  run, to see what the dataset actually holds.

Adapted from the toolkit's own `visualize_segment.py`, keeping its layers (the
future-path ribbon, optional lane rails, CAN radar boxes with the in-lane lead
picked out, a HUD, a north-up minimap) and adding the one it has no concept of:
a **planned** trajectory. Projection is the toolkit's throughout --
`img_from_device` then `denormalize` -- so what is drawn and what is scored go
through one camera model rather than two that agree until they do not.

Colour comes from `viz.palette`, so the recorded future is the same blue it is
in every other figure in the repo and a model keeps its own colour here too.

## The frame trap

A plan arrives in paddock's ego frame -- **x forward, y left, z up**, origin at
the camera. The toolkit projects comma2k19's device frame, **[forward, right,
down]**. `ego_to_road` does both halves: the axis flip (`loader.device_from_ego`)
and the drop onto the road plane, because a trajectory is a path over the ground
while the ego origin sits `EON_HEIGHT` above it.

Getting the sign wrong mirrors the drawn path about the direction of travel and
raises nothing at all -- on a straight road it is invisible, and a curve just
looks like a bad prediction. `tests/test_overlay.py` pins it down instead.

## Fonts

The script this came from named a DejaVu .ttf under `/usr/share`, which
`tests/test_portability.py` rejects on sight and which half the containers in
the world do not have. Pillow bundles a scalable default; `PADDOCK_FONT` names
a .ttf for anyone who wants a different one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import toolkit as TK
from paddock.viz import palette

#: Half width of the recorded-path ribbon, metres. Wide enough to read as a path
#: rather than a line, narrow enough that the lane markings stay visible under it.
PATH_HALF_W = 0.9
#: Half a US highway lane. The rails are **derived**, not recorded -- comma2k19
#: stores no lane geometry -- so they are off by default and labelled as such.
LANE_HALF_W = 1.85
#: How far ahead the recorded path is drawn, and how much of the near field is
#: skipped because the hood hides it anyway.
PATH_HORIZON_S = 5.0
PATH_SKIP_S = 0.3
#: Radar rows kept either side of a frame's timestamp. A track is re-reported at
#: ~15 Hz, so this is a little over one reporting period: wide enough to catch
#: every live track, narrow enough not to draw a stale one.
RADAR_WINDOW_S = 0.06
#: Blend applied after the ribbon so the road stays readable underneath it.
#: PIL polygons have no alpha of their own, so the whole layer is faded at once.
RIBBON_ALPHA = 0.55

#: An override for a site whose Pillow has no usable bundled font.
ENV_FONT = "PADDOCK_FONT"

_RADAR = "#ffc400"
_LEAD = "#ff4040"
_WHITE = "#ffffff"
_TEXT = "#dcdcdc"
_MINIMAP = "#5aa0ff"


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def project(points_device, clip: bool = True) -> np.ndarray:
    """Device-frame points -> (N, 2) pixels, NaN where they cannot be drawn.

    `clip=True` is the toolkit's `denormalize`, which NaNs anything outside the
    frame -- right for a path, which should simply stop at the edge of the
    image. `clip=False` keeps off-image pixels so a box straddling the edge can
    still be clamped and drawn. Points behind the camera are NaN either way;
    that is `img_from_device`, not this.
    """
    tk = TK.load()
    pts = np.atleast_2d(np.asarray(points_device, dtype=np.float64))
    norm = np.atleast_2d(np.asarray(tk.img_from_device(pts)))
    if clip:
        return np.atleast_2d(np.asarray(tk.denormalize(norm)))
    homogeneous = np.hstack([norm, np.ones((len(norm), 1))])
    return (np.asarray(tk.eon_intrinsics) @ homogeneous.T).T[:, :2]


def ego_to_road(xy, height: float = L.EON_HEIGHT) -> np.ndarray:
    """paddock ego [x fwd, y left] -> device points on the road plane.

    Accepts (N, 2) or (N, 3); a third column is taken as ego z (up). See the
    module docstring for why both halves of this matter.
    """
    pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"need (N, 2) or (N, 3) ego points, got shape {pts.shape}")
    ego = np.zeros((len(pts), 3))
    ego[:, : min(3, pts.shape[1])] = pts[:, : min(3, pts.shape[1])]
    device = L.device_from_ego(ego)
    device[:, 2] += height  # device z is *down*: the road is below the camera
    return device


def _finite(pixels: np.ndarray) -> np.ndarray:
    return pixels[np.isfinite(pixels).all(axis=1)]


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


@cache
def _font(size: int, override: str | None):
    from PIL import ImageFont

    if override:
        return ImageFont.truetype(override, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: one fixed bitmap size, and that is that
        return ImageFont.load_default()


def font(size: int = 15):
    """A font of roughly `size`, with no path to any one machine's font tree."""
    return _font(size, os.environ.get(ENV_FONT) or None)


@dataclass(frozen=True)
class Plan:
    """A planned path to draw, in paddock ego frame [x forward, y left].

    `label` reaches the legend, so it is the place to put the number the plan
    scored -- the picture and the metric belong side by side or the reader has
    to hold one of them in their head.
    """

    label: str
    xy: np.ndarray
    color: str

    @classmethod
    def for_model(cls, model: str, xy, label: str | None = None) -> Plan:
        return cls(label=label or model, xy=np.asarray(xy, dtype=np.float64), color=palette.color(model))


@dataclass(frozen=True)
class Layers:
    """Which of the recorded layers to draw. Plans are always drawn."""

    path: bool = True
    rails: bool = False
    radar: bool = True
    hud: bool = True
    minimap: bool = True


def draw_ribbon(
    draw, path_device, half_width: float, height: float, fill: str, outline: str | None = None
) -> None:
    """The toolkit's `draw_path`, as PIL polygons rather than cv2 fills."""
    left = project(path_device + np.array([0.0, -half_width, height]))
    right = project(path_device + np.array([0.0, half_width, height]))
    ok = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left, right = left[ok], right[ok]
    for i in range(1, len(left)):
        quad = [tuple(left[i - 1]), tuple(right[i - 1]), tuple(right[i]), tuple(left[i])]
        draw.polygon(quad, fill=fill, outline=outline)


def draw_rails(draw, path_device, half_width: float, height: float, color: str) -> None:
    """Two thin lines at +-`half_width`: the lane the car is holding, derived."""
    for sign in (-1.0, 1.0):
        pts = _finite(project(path_device + np.array([0.0, sign * half_width, height])))
        if len(pts) > 1:
            draw.line([tuple(p) for p in pts], fill=color, width=3)


def draw_plan(draw, plan_device, color: str, width: int = 5, dot: float = 5.0) -> None:
    """A planned path as a line with its waypoints marked.

    A line rather than a ribbon, so it stays legible *over* the recorded path
    instead of hiding it; the dots because waypoint spacing is the plan's speed
    profile, and a smooth line drawn through them hides the one failure mode
    (a plan that goes nowhere) that looks most like a modelling result.
    """
    pts = _finite(project(plan_device))
    if len(pts) > 1:
        draw.line([tuple(p) for p in pts], fill=color, width=width, joint="curve")
    for p in pts:
        draw.ellipse([p[0] - dot, p[1] - dot, p[0] + dot, p[1] + dot], fill=color)


def radar_tracks(radar_t, radar_v, t: float, window: float = RADAR_WINDOW_S) -> np.ndarray:
    """Live tracks at time `t`: the most recent row per track address.

    A track (columns `[fwd, left, rel_speed, nan, nan, address, new_track]`) is
    re-reported every ~15 Hz, so a window wide enough to catch every track also
    catches several copies of each. Keeping the last per address is what makes
    the count in the HUD a track count rather than a row count.
    """
    if radar_t is None or radar_v is None or not len(radar_t):
        return np.empty((0, 7))
    lo = int(np.searchsorted(radar_t, t - window))
    hi = int(np.searchsorted(radar_t, t + window))
    latest = {float(row[5]): row for row in radar_v[lo:hi]}
    return np.array(list(latest.values())) if latest else np.empty((0, 7))


def lead_track(rows, half_lane: float = 1.8):
    """The closest track inside the ego lane -- what an ACC model must predict."""
    in_lane = [r for r in rows if abs(float(r[1])) < half_lane and 1.0 < float(r[0]) < 200.0]
    return min(in_lane, key=lambda r: float(r[0])) if in_lane else None


def _place_label(draw, taken, x: float, y: float, text: str) -> tuple[float, float]:
    """Where to put `text` so it does not sit on a label already placed.

    Every radar return at highway distance projects to within a few pixels of
    the horizon, so the labels all want the same row and overlap into an
    unreadable smear -- which is what the original script produced. Walking the
    label upward until it clears keeps all of them rather than dropping any,
    and callers place the near tracks first so the closest car keeps the row
    nearest its own box.
    """
    line = 18
    width = draw.textlength(text, font=font(15))

    def clear(top: float) -> bool:
        return not any(x < t[2] and t[0] < x + width and top < t[3] and t[1] < top + line for t in taken)

    for _ in range(8):
        if y < 0 or clear(y):
            break
        y -= line
    taken.append((x, y, x + width, y + line))
    return x, y


def draw_radar(draw, rows, lead=None, size=(L.EON_WIDTH, L.EON_HEIGHT_PX)) -> int:
    """Radar returns as boxes standing on the road plane. Returns how many."""
    drawn = 0
    taken: list[tuple[float, float, float, float]] = []
    # Nearest first, so a label pushed clear of a collision is the one for the
    # car further away -- the near one is what a viewer is looking for.
    rows = sorted(rows, key=lambda r: float(r[0]))
    for row in rows:
        fwd, left, rel_v = float(row[0]), float(row[1]), float(row[2])
        if not np.isfinite(fwd) or not 1.0 < fwd <= 200.0:
            continue
        is_lead = lead is not None and float(row[5]) == float(lead[5])
        color = _LEAD if is_lead else _RADAR
        y = -left  # the radar reports left-positive; device y is right-positive
        box = np.array(
            [
                [fwd, y - 0.9, L.EON_HEIGHT],
                [fwd, y + 0.9, L.EON_HEIGHT],
                [fwd, y + 0.9, L.EON_HEIGHT - 1.5],
                [fwd, y - 0.9, L.EON_HEIGHT - 1.5],
            ]
        )
        pts = project(box, clip=False)
        if not np.isfinite(pts).all():
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        if xs.max() < 0 or xs.min() > size[0] or ys.max() < 0 or ys.min() > size[1]:
            continue
        draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=color, width=3 if is_lead else 2)
        text = f"{fwd:.0f} m {rel_v:+.1f}"
        draw.text(_place_label(draw, taken, xs.min(), ys.min() - 18, text), text, fill=color, font=font(15))
        drawn += 1
    return drawn


def draw_hud(draw, title: str, lines, note: str | None = None) -> None:
    width, height = 340, 22 * len(lines) + (66 if note else 44)
    draw.rectangle([12, 12, 12 + width, 12 + height], outline=_WHITE)
    draw.text((24, 22), title, fill=_WHITE, font=font(16))
    for i, (key, value) in enumerate(lines):
        draw.text((24, 50 + 22 * i), f"{key:<16}{value}", fill=_TEXT, font=font(15))
    if note:
        draw.text((24, 56 + 22 * len(lines)), note, fill=_WHITE, font=font(16))


def draw_legend(draw, size, entries) -> None:
    """Swatch and label per drawn layer, bottom left, in drawing order."""
    for n, (label, color) in enumerate(entries):
        y = size[1] - 30 - 22 * (len(entries) - n)
        draw.rectangle([20, y, 38, y + 14], fill=color)
        draw.text((46, y - 2), label, fill=_WHITE, font=font(15))


def draw_minimap(draw, ned: np.ndarray, index: int, size, box: int = 190, margin: int = 14) -> None:
    """The whole segment's path, north up, with the current position marked.

    The one layer that answers "where in the segment am I" -- which is exactly
    the question a keyframe drawn out of context raises.
    """
    x0, y0 = size[0] - box - margin, size[1] - box - margin
    draw.rectangle([x0, y0, x0 + box, y0 + box], outline=_WHITE)
    north, east = ned[:, 0], ned[:, 1]
    span = max(np.ptp(north), np.ptp(east), 1.0)
    cn, ce = (north.max() + north.min()) / 2, (east.max() + east.min()) / 2

    def to_px(i: int) -> tuple[float, float]:
        return (
            x0 + box / 2 + (east[i] - ce) / span * (box - 20),
            y0 + box / 2 - (north[i] - cn) / span * (box - 20),
        )

    step = max(1, len(ned) // 400)
    draw.line([to_px(i) for i in range(0, len(ned), step)], fill=_MINIMAP, width=2)
    px, py = to_px(min(index, len(ned) - 1))
    draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=_LEAD)
    draw.text((x0 + 6, y0 + 4), "N ^  segment path", fill=_WHITE, font=font(15))


# --------------------------------------------------------------------------
# One segment, drawable frame by frame
# --------------------------------------------------------------------------


def _load(seg_path: Path, rel: str):
    path = seg_path / rel
    return np.load(path) if path.exists() else None


def _nearest(times, t: float) -> int | None:
    """Index of the sample nearest `t`. Every log in a segment shares boot time."""
    if times is None or not len(times):
        return None
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


class SegmentOverlay:
    """One segment's labels, loaded once, drawn onto frames one at a time.

    Everything except the video: the pose stream, the CAN/GNSS/IMU logs and the
    minimap path. Constructed per segment because that is the unit all of them
    are stored in, and because `draw` is called once per frame.
    """

    def __init__(
        self,
        seg_path: Path,
        layers: Layers | None = None,
        path_horizon_s: float = PATH_HORIZON_S,
        path_skip_s: float = PATH_SKIP_S,
        radar_window_s: float = RADAR_WINDOW_S,
    ) -> None:
        self.path = Path(seg_path)
        self.layers = layers or Layers()
        self.path_horizon_s = path_horizon_s
        self.path_skip_s = path_skip_s
        self.radar_window_s = radar_window_s
        self.pose = L.load_pose(self.path)
        self.title = f"{self.path.parent.name}  seg {self.path.name}"

        can = self.path / "processed_log" / "CAN"
        self.speed = (_load(can, "speed/t"), _load(can, "speed/value"))
        self.steer = (_load(can, "steering_angle/t"), _load(can, "steering_angle/value"))
        self.wheel = (_load(can, "wheel_speed/t"), _load(can, "wheel_speed/value"))
        self.radar = (_load(can, "radar/t"), _load(can, "radar/value"))
        log = self.path / "processed_log"
        self.gnss = (_load(log, "GNSS/live_gnss_ublox/t"), _load(log, "GNSS/live_gnss_ublox/value"))
        self.accel = (_load(log, "IMU/accelerometer/t"), _load(log, "IMU/accelerometer/value"))
        self.gps_times = _load(self.path, "global_pose/frame_gps_times")

    @property
    def fps(self) -> float:
        """The segment's own rate, from `frame_times` rather than the container.

        The HEVC header claims 25 fps and a frame count of zero; both are wrong.
        """
        times = self.pose.times
        if len(times) < 2:
            return L.SRC_HZ
        return float((len(times) - 1) / (times[-1] - times[0]))

    @cached_property
    def ned(self) -> np.ndarray:
        """(N, 2) segment path in a local frame, north and east, for the minimap."""
        local_from_ecef = self.pose.ecef_from_device[0].T
        return np.einsum("ij,kj->ki", local_from_ecef, self.pose.positions - self.pose.positions[0])[:, :2]

    def future_device(self, index: int) -> np.ndarray | None:
        """The recorded path ahead of frame `index`, in that frame's device frame."""
        fps = self.fps
        start = index + int(round(self.path_skip_s * fps))
        stop = min(index + int(round(self.path_horizon_s * fps)), len(self.pose))
        if stop <= start:
            return None
        rel = self.pose.positions[start:stop] - self.pose.positions[index]
        return rel @ self.pose.device_from_ecef[index].T

    def draw(self, rgb: np.ndarray, index: int, plans=(), note: str | None = None) -> np.ndarray:
        """One frame with every enabled layer on it. Returns a new RGB array."""
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        pen = ImageDraw.Draw(img)
        t = float(self.pose.times[index])

        if self.layers.path:
            base = img.copy()
            future = self.future_device(index)
            if future is not None and len(future):
                draw_ribbon(pen, future, PATH_HALF_W, L.EON_HEIGHT, fill=palette.GROUND_TRUTH)
                if self.layers.rails:
                    draw_rails(pen, future, LANE_HALF_W, L.EON_HEIGHT, _WHITE)
            # Fade the ribbon back into the road, then keep drawing at full
            # strength: a plan must not be dimmed by the layer underneath it.
            img = Image.blend(base, img, RIBBON_ALPHA)
            pen = ImageDraw.Draw(img)

        for plan in plans:
            draw_plan(pen, ego_to_road(plan.xy), plan.color)

        tracks = radar_tracks(*self.radar, t, self.radar_window_s)
        lead = lead_track(tracks)
        if self.layers.radar and len(tracks):
            draw_radar(pen, tracks, lead, size=img.size)

        if self.layers.hud:
            draw_hud(pen, self.title, self._hud_lines(index, t, tracks, lead), note)
            draw_legend(pen, img.size, self._legend(plans))
        if self.layers.minimap:
            draw_minimap(pen, self.ned, index, img.size)
        return np.asarray(img)

    def _hud_lines(self, index: int, t: float, tracks, lead) -> list[tuple[str, str]]:
        def at(pair, fmt, *, take=None):
            times, values = pair
            i = _nearest(times, t)
            if i is None:
                return "n/a"
            row = np.ravel(values[i])
            return fmt(row if take is None else row[take])

        gps = "n/a"
        if self.gps_times is not None and index < len(self.gps_times):
            gps = f"week {self.gps_times[index][0]:.0f}  tow {self.gps_times[index][1]:.1f} s"
        ahead = "none in lane"
        if lead is not None:
            ahead = f"{float(lead[0]):5.1f} m  {float(lead[2]):+4.1f} m/s"
        return [
            ("t (boot)", f"{t:9.2f} s   frame {index}"),
            ("gps time", gps),
            ("CAN speed", at(self.speed, lambda v: f"{v[0]:5.1f} m/s  ({v[0] * 3.6:5.1f} km/h)")),
            ("steer angle", at(self.steer, lambda v: f"{v[0]:+6.1f} deg")),
            ("wheel speeds", at(self.wheel, lambda v: " ".join(f"{x:4.1f}" for x in v))),
            ("ego speed", f"{self.pose.speed[index]:5.1f} m/s (pose)"),
            ("lat, lon", at(self.gnss, lambda v: f"{v[0]:.5f}, {v[1]:.5f}")),
            ("altitude", at(self.gnss, lambda v: f"{v[4]:7.1f} m")),
            ("accel f/r/d", at(self.accel, lambda v: " ".join(f"{x:+5.2f}" for x in v[:3]))),
            ("radar tracks", f"{len(tracks)}"),
            ("lead", ahead),
        ]

    def _legend(self, plans) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        if self.layers.path:
            entries.append((f"recorded path (next {self.path_horizon_s:.0f} s)", palette.GROUND_TRUTH))
            if self.layers.rails:
                entries.append((f"lane rails +-{LANE_HALF_W} m (derived, not recorded)", _WHITE))
        if self.layers.radar:
            entries.append(("radar track (CAN)", _RADAR))
            entries.append(("lead track (closest in lane)", _LEAD))
        entries.extend((plan.label, plan.color) for plan in plans)
        return entries
