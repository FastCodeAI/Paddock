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
#: Re-exported: reading radar rows is data access, not drawing, and the harness
#: needs the same lead definition this module draws in order to score against it.
RADAR_WINDOW_S = L.RADAR_WINDOW_S
radar_tracks = L.radar_tracks
lead_track = L.lead_track
#: Blend applied after the ribbon so the road stays readable underneath it.
#: PIL polygons have no alpha of their own, so the whole layer is faded at once.
#:
#: Low, because a 1.8 m corridor two metres from the camera is most of the frame:
#: at 0.55 the recorded path was the brightest thing in the image and hid the
#: predicted one drawn on top of it, which inverts what the comparison is for.
RIBBON_ALPHA = 0.32

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


def calib_to_device(points_calib, rpy) -> np.ndarray:
    """Calibrated-frame points -> the device frame `project` expects.

    The *calibrated* frame is aligned with the car in pitch and yaw; the device
    frame is however the phone happens to sit on the windscreen. Anything
    measured by the car -- the radar above all -- arrives in the first and has
    to be rotated into the second before it can be projected, because the camera
    model knows only where the camera is pointing.

    `calibration.estimate` defines `p_calib = p_device @ rot_from_euler(rpy)`.
    The rotation is orthonormal, so the inverse is a right-multiply by its
    transpose; `rpy = None` means "no estimate", and the points pass through.
    """
    pts = np.atleast_2d(np.asarray(points_calib, dtype=np.float64))
    if rpy is None:
        return pts
    return pts @ L.rot_from_euler(np.asarray(rpy, dtype=np.float64)).T


def radar_to_device(fwd: float, left: float, rpy=None, height: float = L.EON_HEIGHT) -> np.ndarray:
    """One radar return -> the four corners of a car-sized box, device frame.

    ## The correction that was missing

    **The radar is bolted to the car; the camera is stuck to the windscreen.**
    A radar return arrives in the car-aligned (calibrated) frame and `project`
    consumes the device frame, so without `calib_to_device` the mount rotation
    is simply left out. On this dataset the mount pitch is 1.0-3.8 deg, which at
    fl 910 is **16-61 px** of vertical error on every box -- far more than the
    gap between a lead vehicle's bumper and the road under it, which is why the
    box looked slightly low against the car it was drawn on. The reference
    script has the same defect: it rotates lanes, edges and paths, and not radar.

    ## The correction that turned out not to be needed

    openpilot's `RADAR_TO_CAMERA = 1.52` (`selfdrive/controls/radard.py:26`)
    converts between the camera frame and the radar frame, and `dRel` is
    documented as measured "from front of car"
    (`opendbc/car/honda/radar_interface.py`). That reads like this column needs
    +1.52 m before it can be projected. **Measured on real frames, it does
    not** -- comma2k19's recorded `processed_log/CAN/radar` behaves as a
    camera-frame range already, and adding the offset puts every box about 1.5 m
    too far away.

    Checked two ways on `Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/4`, whose
    route calibration is a 2.60 deg pitch, at leads of 6.70 m and 8.42 m:

    * *Apparent width.* `width_px = 910 * 1.8 / d`, which is 244 px at 6.70 m and
      199 px at 8.22 m. Verticals at 244 px sit on the lead's tail lights; at
      199 px they cut well inside its body. Same at the second range (195 vs
      165 px). This is the decisive one -- it uses no assumption about camera
      height or pitch.
    * *Road plane.* A ground line at the raw range lands under the rear bumper;
      at range + 1.52 m it lands up on the bumper. Making the offset hypothesis
      fit would need a camera height of 1.48 m at one lead and 1.43 m at the
      other -- inconsistent, and both implausible for a windscreen mount --
      whereas the raw range fits comma's published 1.22 m at both.

    Consistent with the reference's own Track B number, which measured the
    model's camera-frame lead range as 0.8 m *short* of this column, not 1.5 m
    long. The likely provenance is that comma's 2018 logger published
    `liveTracks` already shifted into the camera frame, but that is inference;
    the geometry above is the measurement.

    ## What the box claims

    The box is *nominal*, not measured: 1.8 m wide and 1.5 m tall, standing on
    the road plane. A radar return is a point on the rear of the vehicle, so the
    face drawn is its rear bumper and neither the width nor the height says
    anything about the car that is actually there.
    """
    x = fwd
    y = -left  # the radar reports left-positive; the calibrated frame is y-right
    box = np.array(
        [
            [x, y - 0.9, height],
            [x, y + 0.9, height],
            [x, y + 0.9, height - 1.5],
            [x, y - 0.9, height - 1.5],
        ]
    )
    return calib_to_device(box, rpy)


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


def draw_radar(draw, rows, lead=None, size=(L.EON_WIDTH, L.EON_HEIGHT_PX), rpy=None) -> int:
    """Radar returns as boxes standing on the road plane. Returns how many.

    `rpy` is the route's estimated mount rotation; see `radar_to_device` for
    what it corrects and for what the box does and does not claim.
    """
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
        pts = project(radar_to_device(fwd, left, rpy), clip=False)
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
        rpy=None,
    ) -> None:
        self.path = Path(seg_path)
        self.layers = layers or Layers()
        #: The route's mount rotation, for anything measured in the car's frame
        #: rather than the camera's. Estimated from the route if not supplied --
        #: a caller that already holds a `RouteCalibrations` should pass its
        #: answer rather than paying for a second pooled estimate.
        self.rpy = self._rpy(rpy)
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

    def _rpy(self, given):
        """The mount rotation to draw car-frame measurements with.

        Falls back to the route's own pooled estimate, and to None -- draw
        uncorrected -- if the route has no frame that passes calibrationd's
        gate or produces a rotation no windscreen bracket could hold. Drawing
        nothing would hide the radar entirely; drawing an implausible rotation
        would move every box by more than the error it was meant to remove.
        """
        if given is not None:
            return np.asarray(given, dtype=np.float64)
        from paddock.adapters.benchmarks.comma2k19 import calibration as CAL

        cal = CAL.for_segment(self.path)
        return cal.rpy if cal.is_valid else None

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
            draw_radar(pen, tracks, lead, size=img.size, rpy=self.rpy)

        if self.layers.hud:
            draw_hud(pen, self.title, self._hud_lines(index, t, tracks, lead), note)
            draw_legend(pen, img.size, self.legend_entries(plans))
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

    def legend_entries(self, plans=()) -> list[tuple[str, str]]:
        """`(label, colour)` for every layer this overlay will actually draw.

        Public because the legend does not have to be drawn *on* the frame:
        `dashboard` puts the same entries in its header, where they sit on paper
        rather than on road. It reflects `self.layers`, so a legend can never
        promise a layer that is switched off.
        """
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
