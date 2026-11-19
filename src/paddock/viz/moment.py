"""A stretch of scored open-loop keyframes as a video, drawn the same way for every benchmark.

In open loop every model is shown the same recorded frames, so tiles side by side
(neuroNCAP's videos) would show one picture eight times. There is one camera view
instead. Every frame of the video is a scored keyframe with every model's plan for
it and the recorded path drawn on the road, the keyframes in order, each held for
the time to the next. A panel beside it shows the same lines from above, where
lengths and bends are easier to compare, with each model's score on that keyframe.

Knows neither models nor benchmarks, like the rest of viz/: a benchmark's render.py
supplies the frames, the camera that took them, and the named plans. `choose` is the
rule every open-loop benchmark uses to pick the moments the website shows.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.viz import palette, video

WIDTH = 1920
CAMERA_W, CAMERA_H = 1280, 720  # the camera view; the panel takes the rest of the width
PANEL_W = WIDTH - CAMERA_W
STRIP = 46  # the clock along the bottom, as in the neuroNCAP videos
HEIGHT = CAMERA_H + STRIP
FPS = 10

#: Nothing nearer the camera than this is drawn. A path starts at the car, behind or
#: below the lens; this is where its line enters the picture.
NEAR_M = 0.5
#: Lines are cut into pieces this long before they are projected, so a line bends
#: with the lens and stops where it leaves the view.
STEP_M = 0.25
#: Beyond about 45 degrees off the lens's axis, a fitted distortion polynomial can
#: fold back on itself. No plan needs drawing that far out.
MAX_R2 = 1.0
#: The view from above shows at most this far behind the car: a plan held on the road
#: falls behind as the car drives past it.
BEHIND_M = 10.0
#: A plan that never gets this far from the car is too short to see in the camera
#: view, which shows no road this close, or from above, where it hides under the
#: car. The legend says so instead.
SHORT_M = 3.0

BACKGROUND = (14, 15, 18)  # the neuroNCAP videos'
INK, INK2, MUTED = (242, 242, 245), (205, 207, 214), (150, 152, 160)
GRID = (52, 54, 62)
EDGE = (0, 0, 0, 150)  # under every line on the road, so it reads on light asphalt too

KIND_NAMES = {"left": "Left turn", "right": "Right turn", "straight": "Straight on"}


@dataclass
class Camera:
    """A camera on the car, enough to draw a path on the road into its picture.

    `cam2ego` takes the camera's own frame (OpenCV: x right, y down, z forward)
    into the ego frame (x forward, y left, z up, metres). `ground_z` is the road's
    height in the ego frame. `distortion` is OpenCV's k1, k2, p1, p2, k3, for a
    picture that was not undistorted.
    """

    intrinsic: np.ndarray
    cam2ego: np.ndarray
    ground_z: float = 0.0
    distortion: Sequence[float] | None = None


@dataclass
class Plan:
    """One model's plan: its name and colour, ego-frame x/y in metres, and its score here."""

    label: str
    colour: str
    xy: np.ndarray
    note: str = ""


@dataclass
class Step:
    """One frame of a clip: its picture, the camera that took it, every model's plan
    and the recorded path (ego frame, or None). `t` is seconds into the clip; `note`
    goes along the bottom, for the keyframe the clip was picked for.

    A plan starts at the car (`from_car`), unless it was made at an earlier frame
    and is shown where it lies on the road now: then each path's own first point is
    where the car was when the plan was made.
    """

    picture: Path
    camera: Camera
    plans: list[Plan]
    truth: np.ndarray | None
    t: float
    note: str = ""
    from_car: bool = True
    poster: bool = False  # the frame for the still; else the first with a note, else the middle one


def project(xy, camera: Camera, from_car: bool = True) -> list[np.ndarray]:
    """A path on the road, as pixel polylines in the camera's own picture.

    The path is drawn from the car itself (the ego origin), so a line enters at
    the bottom of the view, or with `from_car=False` from its own first point.
    It is split wherever it leaves what the lens can show.
    """
    px, ok = _pixels(_densify(_path(xy, from_car), STEP_M), camera)
    return [px[a:b] for a, b in _spans(ok) if b - a >= 2]


def _path(xy, from_car: bool) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64)[:, :2]
    return np.vstack([np.zeros((1, 2)), pts]) if from_car else pts


def end_point(xy, camera: Camera) -> np.ndarray | None:
    """Where a path ends, in pixels, or None when its end is out of view."""
    pts = np.asarray(xy, dtype=np.float64)[:, :2]
    if not len(pts):
        return None
    px, ok = _pixels(pts[-1:], camera)
    return px[0] if ok[0] else None


def _pixels(pts: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Ground points (ego x, y) as pixels, and which of them the lens can show."""
    ego = np.column_stack([pts, np.full(len(pts), camera.ground_z), np.ones(len(pts))])
    cam = (np.linalg.inv(np.asarray(camera.cam2ego, dtype=np.float64)) @ ego.T).T[:, :3]
    ok = cam[:, 2] > NEAR_M
    depth = np.where(ok, cam[:, 2], 1.0)
    u, v = cam[:, 0] / depth, cam[:, 1] / depth
    if camera.distortion is not None:
        k1, k2, p1, p2, k3 = (list(camera.distortion) + [0.0] * 5)[:5]
        r2 = u * u + v * v
        ok &= r2 < MAX_R2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        u, v = (
            u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u),
            v * radial + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v,
        )
    k = np.asarray(camera.intrinsic, dtype=np.float64)
    return np.column_stack([k[0, 0] * u + k[0, 1] * v + k[0, 2], k[1, 1] * v + k[1, 2]]), ok


def write(
    out: Path,
    steps: Sequence[Step],
    *,
    step_s: float,
    title: str,
    score_name: str,
    ffmpeg: str,
) -> list[Path]:
    """`out`.mp4 of `steps` in order, each held `step_s` (the time between keyframes,
    so the video runs in real time), and a poster `out`.jpg of the step with a note,
    the keyframe the clip was picked for, or else the middle one.

    The view from above keeps one scale for the whole clip, so it does not jump
    from one keyframe to the next.
    """
    from PIL import Image

    steps = list(steps)
    if not steps:
        raise ValueError(f"no keyframes to draw for {out}")
    extent = _extent(
        [p.xy for st in steps for p in st.plans] + [st.truth for st in steps if st.truth is not None]
    )
    geometry = _fit(Image.open(steps[0].picture).size)
    hold = max(1, round(step_s * FPS))

    def draw(step: Step) -> np.ndarray:
        view = _scaled(Image.open(step.picture).convert("RGB"), geometry)
        view = _road(view, step.camera, geometry, step.plans, step.truth, step.from_car)
        panel = _panel(step.plans, step.truth, score_name, extent, step.from_car)
        return _frame(view, panel, step.t, title, step.note)

    poster_step = next(
        (st for st in steps if st.poster), next((st for st in steps if st.note), steps[len(steps) // 2])
    )

    def frames():
        for step in steps:
            frame = draw(step)
            for _ in range(hold):
                yield frame

    out.parent.mkdir(parents=True, exist_ok=True)
    mp4, jpg = out.with_suffix(".mp4"), out.with_suffix(".jpg")
    video.encode(frames(), mp4, WIDTH, HEIGHT, FPS, ffmpeg)
    Image.fromarray(draw(poster_step)).save(jpg, quality=85)
    return [mp4, jpg]


def choose(
    scores: dict[str, dict[str, float]],
    kinds: dict[str, str],
    places: dict[str, str],
    n: int,
    *,
    higher_is_better: bool,
    first: str = "hardest",
    tiebreak: dict[str, float] | None = None,
) -> list[dict]:
    """The moments to show people: hard ones, and of different kinds.

    `scores` is sample -> model -> that model's score on it; only samples every
    model scored are considered, so no line is ever missing for want of a score.
    `kinds` names what the car did there (left, right, straight) and `places`
    where (a scene, a log): no place is shown twice.

    First one moment of each kind: the hardest, where hardest means the worst
    score averaged over the models, or with `first="disagree"` the one the models
    disagree on most. Then, until there are `n`, the moment where the models
    disagree most. Disagreement is the spread (standard deviation) of the models'
    scores: on a score that is mostly 0 or 1, as NAVSIM's is, the gap between the
    best and the worst model is 1 on thousands of moments and tells none apart.
    Ties go to the harder moment, then the larger `tiebreak` (a benchmark whose
    scores tie often passes how far apart the plans end), then the lower sample id.

    Returns [{"sample", "kind", "why", "of_kind", "mean", "low", "high", "scores"}],
    in the order to show them; `why` is "hardest" or "disagree", `of_kind` is true
    for a moment picked as the one of its kind, and `scores` is model -> score.
    """
    if first not in ("hardest", "disagree"):
        raise ValueError(f"first={first!r}: expected 'hardest' or 'disagree'")
    models = set().union(*scores.values()) if scores else set()
    full = {s: by for s, by in scores.items() if set(by) == models}
    if not full:
        return []
    sign = -1.0 if higher_is_better else 1.0

    def mean(s):
        return statistics.fmean(full[s].values())

    def spread(s):
        return statistics.pstdev(full[s].values())

    extra = tiebreak or {}

    def hardest(s):
        return (-sign * mean(s), -extra.get(s, 0.0), s)

    def disagree(s):
        return (-spread(s), -sign * mean(s), -extra.get(s, 0.0), s)

    picks: list[tuple[str, str, bool]] = []
    shown: set[str] = set()
    for kind in KIND_NAMES:
        left = sorted(
            (s for s in full if kinds.get(s) == kind and places.get(s, s) not in shown),
            key=hardest if first == "hardest" else disagree,
        )
        if left:
            picks.append((left[0], first, True))
            shown.add(places.get(left[0], left[0]))
    while len(picks) < n:
        left = sorted((s for s in full if places.get(s, s) not in shown), key=disagree)
        if not left:
            break
        picks.append((left[0], "disagree", False))
        shown.add(places.get(left[0], left[0]))
    return [
        {
            "sample": s,
            "kind": kinds.get(s, ""),
            "why": why,
            "of_kind": of_kind,
            "mean": mean(s),
            "low": min(full[s].values()),
            "high": max(full[s].values()),
            "scores": dict(full[s]),
        }
        for s, why, of_kind in picks[:n]
    ]


def kind_from_path(xy, threshold_m: float = 2.0) -> str:
    """Left, right or straight, from where a path is after 3 s.

    VAD's rule, the one nuScenes uses to give its models a command: more than
    2 m to one side at 3 s is a turn. For a dataset that records no command.
    """
    pts = np.asarray(xy, dtype=np.float64)
    lateral = float(pts[min(5, len(pts) - 1), 1])
    return "left" if lateral > threshold_m else "right" if lateral < -threshold_m else "straight"


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


@dataclass
class _Geometry:
    """How the source picture maps into the camera view: scale, then shift down by `dy`."""

    scale: float
    dy: float


def _fit(size: tuple[int, int]) -> _Geometry:
    """Fill the view's width. A taller picture loses its top, the sky; the road stays."""
    w, h = size
    scale = CAMERA_W / w
    return _Geometry(scale, CAMERA_H - h * scale)


def _scaled(image, geometry: _Geometry):
    from PIL import Image

    w, h = image.size
    scaled = image.resize((CAMERA_W, round(h * geometry.scale)), Image.BILINEAR)
    view = Image.new("RGB", (CAMERA_W, CAMERA_H), BACKGROUND)
    view.paste(scaled, (0, round(geometry.dy)))
    return view


def _road(view, camera: Camera, geometry: _Geometry, plans: Sequence[Plan], truth, from_car: bool = True):
    """The scored frame with the recorded path and every plan drawn on the road."""
    from PIL import Image, ImageDraw

    def place(line):
        return [(x * geometry.scale, y * geometry.scale + geometry.dy) for x, y in line]

    layer = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if truth is not None:
        gt = _rgb(palette.GROUND_TRUTH)
        for line in project(truth, camera, from_car):
            draw.line(place(line), fill=(*gt, 95), width=24, joint="curve")
            draw.line(place(line), fill=(*gt, 255), width=3, joint="curve")
    lines = [(plan, place(line)) for plan in plans for line in project(plan.xy, camera, from_car)]
    for _, line in lines:  # every edge first, so no edge cuts through another model's line
        draw.line(line, fill=EDGE, width=9, joint="curve")
    for plan, line in lines:
        draw.line(line, fill=(*_rgb(plan.colour), 255), width=5, joint="curve")
    # Where each plan ends, over every line: two straight plans in one lane differ
    # only in length, and the shorter one is otherwise hidden under the longer.
    for plan in plans:
        end = end_point(plan.xy, camera)
        if end is not None:
            x, y = place([end])[0]
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(*_rgb(plan.colour), 255), outline=EDGE, width=3)
    return Image.alpha_composite(view.convert("RGBA"), layer).convert("RGB")


def _panel(
    plans: Sequence[Plan],
    truth,
    score_name: str,
    extent: tuple[float, float, float] | None = None,
    from_car: bool = True,
) -> np.ndarray:
    """The panel beside the camera: the same lines from above, then who scored what."""
    from PIL import Image, ImageDraw

    rows = len(plans) + (truth is not None)
    row_h = 30 if rows <= 8 else 26
    legend_h = 40 + rows * row_h
    panel = Image.new("RGB", (PANEL_W, CAMERA_H), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    font, small = _font(22), _font(18)
    draw.text((24, 26), "From above", font=small, fill=MUTED, anchor="lm")
    _bev(draw, (24, 48, PANEL_W - 24, CAMERA_H - legend_h - 16), plans, truth, small, extent, from_car)

    y = CAMERA_H - legend_h + 14
    draw.text((PANEL_W - 24, y), score_name, font=small, fill=MUTED, anchor="rm")
    y += 26
    entries = [(p.label, p.colour, _note(p, from_car)) for p in plans]
    if truth is not None:
        entries.append(("Recorded path", palette.GROUND_TRUTH, "what the driver did"))
    for label, colour, note in entries:
        mid = y + row_h // 2
        draw.rounded_rectangle([24, mid - 4, 56, mid + 4], radius=4, fill=colour)
        draw.text((70, mid), label, font=font, fill=INK, anchor="lm")
        draw.text((PANEL_W - 24, mid), note, font=font, fill=INK2, anchor="rm")
        y += row_h
    return np.asarray(panel)


def _note(plan: Plan, from_car: bool = True) -> str:
    pts = _path(plan.xy, from_car)
    if len(pts) > 1 and float(np.hypot(*(pts - pts[0]).T).max()) < SHORT_M:
        return f"barely moves · {plan.note}" if plan.note else "barely moves"
    return plan.note


def _extent(paths) -> tuple[float, float, float]:
    """(nearest x, furthest x, widest |y|) the view from above shows, for these paths."""
    pts = np.vstack([np.zeros((1, 2)), *[np.asarray(p, dtype=np.float64)[:, :2] for p in paths]])
    pts = pts[pts[:, 0] >= -BEHIND_M]
    return (
        min(-2.0, pts[:, 0].min() - 2),
        max(10.0, pts[:, 0].max() + 3),
        max(8.0, np.abs(pts[:, 1]).max() + 3),
    )


def _bev(draw, box, plans: Sequence[Plan], truth, font, extent=None, from_car: bool = True) -> None:
    """Forward is up and left is left, as through the windscreen, with one scale both ways."""
    left, top, right, bottom = box
    if extent is None:
        extent = _extent([p.xy for p in plans] + ([truth] if truth is not None else []))
    near, far, half = extent
    scale = min((right - left) / (2 * half), (bottom - top) / (far - near))
    cx, base = (left + right) / 2, bottom - (0 - near) * scale

    def at(p):
        return [(cx - y * scale, base - x * scale) for x, y in np.asarray(p)[:, :2]]

    def ahead(xy):
        """The path from the car on, or from its first point still inside the view."""
        pts = _path(xy, from_car)
        inside = np.flatnonzero(pts[:, 0] >= near)
        return pts[inside[0] :] if len(inside) else pts[:0]

    step = 10 if far - near <= 60 else 20
    for metres in range(step, int(far) + 1, step):
        y = base - metres * scale
        draw.line([(left, y), (right, y)], fill=GRID, width=1)
        draw.text((left + 4, y - 4), f"{metres} m", font=font, fill=MUTED, anchor="lb")
    if truth is not None and len(gt := at(ahead(truth))) > 1:
        draw.line(gt, fill=_mix(palette.GROUND_TRUTH, 0.45), width=14, joint="curve")
        draw.line(gt, fill=palette.GROUND_TRUTH, width=2, joint="curve")
    lines = [(plan, line) for plan in plans if len(line := at(ahead(plan.xy))) > 1]
    for plan, line in lines:
        draw.line(line, fill=plan.colour, width=3, joint="curve")
    # One dot every waypoint, so a plan's speed shows in its spacing; over every
    # line, so a plan lying under another still shows where it ends.
    for plan, line in lines:
        for x, y in line[1:]:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=plan.colour)
        x, y = line[-1]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=plan.colour, width=2)
    car_w, car_l = 1.9 * scale / 2, 4.5 * scale
    draw.rectangle([cx - car_w, base - car_l * 0.75, cx + car_w, base + car_l * 0.25], outline=INK, width=2)


def _frame(view, panel: np.ndarray, t: float, title: str, note: str = "") -> np.ndarray:
    """The camera view and the panel over a strip with the clock, the title and the step's note."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    canvas.paste(view, (0, 0))
    canvas.paste(Image.fromarray(panel), (CAMERA_W, 0))
    draw = ImageDraw.Draw(canvas)
    font, small = _font(28), _font(23)
    mid = HEIGHT - STRIP // 2
    draw.rectangle([0, CAMERA_H, WIDTH, HEIGHT], fill=(0, 0, 0))
    draw.text((18, mid), f"t = {t:.1f} s", font=font, fill=INK, anchor="lm")
    draw.text((200, mid), title, font=small, fill=(190, 192, 200), anchor="lm")
    if note:
        draw.text((WIDTH - 18, mid), note, font=font, fill=INK, anchor="rm")
    else:
        draw.text(
            (WIDTH - 18, mid), "every model's plan at each keyframe", font=small, fill=MUTED, anchor="rm"
        )
    return np.asarray(canvas)


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def _rgb(colour: str) -> tuple[int, int, int]:
    h = colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _mix(colour: str, amount: float) -> tuple[int, int, int]:
    """`colour` laid over the background at `amount`, for a band that must not hide the lines on it."""
    return tuple(round(b + (c - b) * amount) for c, b in zip(_rgb(colour), BACKGROUND, strict=True))


def _densify(pts: np.ndarray, step: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        pieces = max(1, math.ceil(float(np.linalg.norm(b - a)) / step))
        out.extend(a + (b - a) * (i / pieces) for i in range(1, pieces + 1))
    return np.asarray(out)


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    spans, start = [], None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans
