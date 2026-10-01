"""What `openpilot_drv_supcombo_kitscenes_image_tweak.ipynb` runs, so the notebook stays short.

The notebook asks one question of five KITScenes turns: does a change to the image, made
before openpilot sees it, plan them better? Everything it calls is a method of `Lab`:

- `recorded_table`, `show_recorded`, `show_warps` (part 1): the turns as the recorded run
  planned them, the four images the model was handed, and the warp that cut them;
- `show_measurements` (part 2): how the camera heights and the speeds are measured;
- `checks`: that the node is the stock adapter, and that what is drawn is what it fed;
- `tweak` (part 3): one config on one turn, drawn beside stock, with what half-pixel shifts alone change;
- `compare` (part 4): one config on every turn, scored against stock and a noise floor.

Paddock never imports a model, and neither does this module. The model runs in
`openpilot_drv_supcombo_image_node.py`, beside it: launched in openpilot's own interpreter
as `paddock run` launches the adapter, and driven over localhost HTTP. Here a config
becomes the 3x3 matrices and lookup tables the node applies, and the node's own `View`
rebuilds what the model was handed, so it can be drawn.

The knobs and their geometry are the image-calibration notebook's, without the two that
are not image changes (the tick rate and the desire).
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import Markdown, display
from matplotlib import patheffects
from PIL import Image

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.adapters.benchmarks.kitscenes.harness import INTERVAL_S, POD_HEIGHT_M, _read_calibration
from paddock.adapters.models.openpilot.policy import MODEL_HZ
from paddock.adapters.models.openpilot.supercombo import rgb_to_yuv420
from paddock.config import REPO_ROOT, SiteConfig
from paddock.policy.protocol import Command, Trajectory, load_frame
from paddock.registry import Registry
from paddock.render import load_runs
from paddock.runner import load_benchmark
from paddock.runtime import env as env_builder
from paddock.runtime import ports, proc
from paddock.viz import palette

BENCH, MODEL = "kitscenes", "openpilot"
HERE = REPO_ROOT / "tools/models/openpilot/analysis"
TAGS = HERE / "openpilot_drv_supcombo_tags.yaml"
NODE_PY = HERE / "openpilot_drv_supcombo_image_node.py"
# Every file whose code decides a result: editing one re-runs everything cached, and nothing else does.
CODE_FILES = (NODE_PY,) + tuple(
    REPO_ROOT / "src/paddock" / p
    for p in (
        "adapters/models/openpilot/policy.py",
        "adapters/models/openpilot/supercombo.py",
        "policy/protocol.py",
        "adapters/benchmarks/kitscenes/harness.py",
        "adapters/benchmarks/kitscenes/loader.py",
    )
)
DT = INTERVAL_S  # the plan's and the GT's step, 0.5 s
N_LONG = 20  # the plan's waypoints: 10 s
H_M = 1.22  # the height of comma's camera over the road, the one the model was trained at (the note's Q4)
# `img` holds the newest frame and the one FRAME_SKIP model ticks before it: openpilot's
# MODEL_RUN_FREQ // MODEL_CONTEXT_FREQ = 20 // 5 (item 1 of supercombo.py's docstring). Restated
# because openpilot's ModelConstants cannot be imported in paddock's environment.
FRAME_SKIP = 4
SPEED_EXPONENT = 0.68  # the note's Q4: across keyframes, head / pose speed goes as (cam_h / map height)^0.68
FILL_Y = 126  # a blanked view's luma: mid-grey; its chroma is 128, no colour
TOL_M = 0.25  # checks(): the most a GPU and a CPU may disagree on one waypoint within H (metres)
ROAD_X = (10, 20, 30, 40, 60)  # the road grid's distances ahead of the camera (m)
ROAD_Y = (-6, -3, 0, 3, 6)  # and its lines along the road (m, left +)

GT_C = palette.GROUND_TRUTH
STOCK_C = palette.color(MODEL)  # the recorded plan, and stock run here
CONFIG_C = "#9467bd"  # the config's plan
GREY = "#8c8c8c"
VIEW_STYLE = {"img": ("#ffd92f", "-"), "big_img": ("#00e5ff", "--")}  # each view's footprint on the camera
OUTLINE = [patheffects.withStroke(linewidth=3.5, foreground="black")]  # a path stays visible on any road
# The four model inputs, in the order drawn: each view's older frame, then its keyframe.
PANELS = (("img", True), ("img", False), ("big_img", True), ("big_img", False))


def _load_node():
    """The node module, by path. Importing it does not import openpilot; only the node's process does."""
    spec = importlib.util.spec_from_file_location("openpilot_image_node", NODE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up while the class is being built
    spec.loader.exec_module(mod)
    return mod


NODE = _load_node()
VIEWS = NODE.VIEWS


# --------------------------------------------------------------------------
# A config
# --------------------------------------------------------------------------

GEOMETRY = (
    "height_scale",
    "zoom",
    "hstretch",
    "yaw_deg",
    "turn_yaw_deg",
    "pitch_deg",
    "roll_deg",
    "ahead_m",
    "left_m",
    "dx_px",
    "dy_px",
)


@dataclass(frozen=True)
class Setting:
    """One image change, from a config dict. Every default leaves the image alone, so Setting() is stock.

    The notebook's part 3 says what each knob does to the image and what it should do to the plan.
    """

    height_scale: float = 1.0
    zoom: float = 1.0
    hstretch: float = 1.0
    yaw_deg: float = 0.0
    turn_yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    ahead_m: float = 0.0
    left_m: float = 0.0
    dx_px: float = 0.0
    dy_px: float = 0.0
    views: str = "both"
    supersample: int = 1
    ss_views: str = "big_img"
    sharpen: float = 0.0
    gamma: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    saturation: float = 1.0
    blank: str = ""

    def __post_init__(self):
        # One type per field, so {"zoom": 1} and {"zoom": 1.0} are one config and one cache entry.
        for f in fields(self):
            if isinstance(f.default, float):
                object.__setattr__(self, f.name, float(getattr(self, f.name)))
        object.__setattr__(self, "supersample", int(self.supersample))
        for name, allowed in (
            ("views", ("both", *VIEWS)),
            ("ss_views", ("both", *VIEWS)),
            ("blank", ("", *VIEWS)),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} = {getattr(self, name)!r}: it must be one of {allowed}")
        if min(self.height_scale, self.zoom, self.hstretch) <= 0 or self.supersample < 1:
            raise ValueError("height_scale, zoom and hstretch must be positive, and supersample 1 or more")

    def changed(self):
        """{knob: value} of the knobs that change anything; empty for stock."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != f.default}
        if not any(k in GEOMETRY for k in out):
            out.pop("views", None)
        if self.supersample == 1:
            out.pop("ss_views", None)
        return out

    def geometric(self):
        return any(k in GEOMETRY for k in self.changed())

    def label(self):
        c = self.changed()
        return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()) or "stock"


STOCK = Setting()
# Eight shifts of half a model pixel: no geometry to speak of (0.03 deg in img), but every model pixel comes
# from a different camera pixel. What they do to a score is what any change at all does: compare()'s noise floor.
SHIFTS = tuple(
    Setting(dx_px=x, dy_px=y)
    for x, y in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))
)
# checks(): one config down every path the node's `View` has (shift, supersampling, filter, tables, fill).
EVERY_PATH = Setting(dx_px=0.5, supersample=3, sharpen=0.5, gamma=0.8, saturation=1.3, blank="img")


def as_setting(x=None, **knobs):
    """A Setting from a Setting, a config dict or keywords. An unknown knob raises a TypeError."""
    if isinstance(x, Setting):
        return replace(x, **knobs)
    return Setting(**((x or {}) | knobs))


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def calib(st, sc):
    """The config's geometry on rays in openpilot's calibrated frame (x forward, y right, z down): the ray a
    model pixel casts -> the ray the camera is sampled along. calib = T @ R @ D, as part 3 says."""
    D = np.diag([1.0, 1.0 / (st.zoom * st.hstretch), st.height_scale / st.zoom])
    psi = np.radians(st.yaw_deg + sc.sign * st.turn_yaw_deg)  # to the left
    R = _rz(-psi) @ _ry(np.radians(st.pitch_deg)) @ _rx(np.radians(st.roll_deg))
    T = np.array([[1.0, 0.0, st.ahead_m / sc.h], [0.0, 1.0, -st.left_m / sc.h], [0.0, 0.0, 1.0]])
    return T @ R @ D


def lut(st):
    """(3, 256) uint8: the Y, U and V tables (gamma, contrast, brightness, saturation)."""
    x = np.arange(256.0)
    y = x if st.gamma == 1 else 16.0 + 219.0 * np.clip((x - 16.0) / 219.0, 0.0, 1.0) ** st.gamma
    y = 125.5 + st.contrast * (y - 125.5) + st.brightness
    uv = 128.0 + st.saturation * (x - 128.0)
    return np.clip(np.rint(np.stack([y, uv, uv])), 0, 255).astype(np.uint8)


def kernel(alpha):
    """3 x 3: identity + alpha * (identity - binomial blur). -1 is the blur itself."""
    blur = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]) / 16.0
    delta = np.zeros((3, 3))
    delta[1, 1] = 1.0
    return delta + alpha * (delta - blur)


def _applies(which, view):
    return which in ("both", view)


def view_specs(st, sc, g):
    """{view: spec} for the node's `View`, for the views the config touches. Raises ValueError if a view
    would look behind the camera."""
    out = {}
    for v in VIEWS:
        spec = {}
        if st.geometric() and _applies(st.views, v):
            spec["calib"] = calib(st, sc).tolist()
            if st.dx_px or st.dy_px:
                spec["post"] = NODE._shift(st.dx_px, st.dy_px).tolist()
            # Every corner's ray must stay in front of the camera, or the homography folds the image over.
            w, h = g["model_size"][v]
            corners = np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], float)
            rays = (
                calib(st, sc) @ np.array(g["calib_from_model"][v]) @ NODE._shift(st.dx_px, st.dy_px) @ corners
            )
            fwd = rays[0] / np.linalg.norm(rays, axis=0)
            if fwd.min() <= 0.05:
                raise ValueError(
                    f"{st.label()}: {v}'s corners look {np.degrees(np.arccos(fwd.min())):.0f} deg off axis"
                )
        if st.supersample > 1 and _applies(st.ss_views, v):
            spec["supersample"] = int(st.supersample)
        if st.sharpen:
            spec["kernel"] = kernel(st.sharpen).tolist()
        if (st.gamma, st.contrast, st.brightness, st.saturation) != (1.0, 1.0, 0.0, 1.0):
            spec["lut"] = lut(st).tolist()
        if st.blank == v:
            spec["fill"] = FILL_Y
        if spec:
            out[v] = spec
    return out


def request(st, sc, g):
    """The /run body for one config on one sample. Every command is STRAIGHT, as the recorded run sent."""
    wire = [dict(o) for o in sc.wire]
    for o in wire:
        o["command"] = int(Command.STRAIGHT)
    body = {"observations": wire}
    specs = view_specs(st, sc, g)
    if specs:
        body["views"] = specs
    return body


def warp(st, sc, view, g):
    """The 3x3 warp (model pixel -> camera pixel) the node cuts this view with: composed in the node's order,
    and float32, as it is applied."""
    if not (st.geometric() and _applies(st.views, view)):
        return np.array(g["stock"][view], np.float32).astype(np.float64)
    C, M = np.array(g["camera_from_calib"]), np.array(g["calib_from_model"][view])
    return (C @ calib(st, sc) @ M @ NODE._shift(st.dx_px, st.dy_px)).astype(np.float32).astype(np.float64)


def apply_h(H, xy):
    """(n, 2) points through a 3x3 homography; NaN where they cross its horizon."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    p = np.c_[xy, np.ones(len(xy))] @ np.asarray(H, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p[:, :2] / p[:, 2:]
    out[~(p[:, 2] > 0)] = np.nan
    return out


def _to_cam(xy, sc, E=None):
    """(n, 2) road points (ego frame, metres) -> full-resolution camera pixels, through the real camera at its
    real height, without those under 0.5 m in front of it. E is the camera's pose: sc.E, on the rig, as the
    benchmark scores, unless given; sc.E0 is the camera over the origin."""
    E = sc.E if E is None else E
    xy = np.asarray(xy, float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(1)]
    pts = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ np.linalg.inv(E).T
    pts = pts[pts[:, 2] > 0.5, :3]
    uv = pts @ sc.K.T
    return uv[:, :2] / uv[:, 2:]


def _road_to_px(sc):
    """G: a road point (x, y, 1), camera-centred, metres -> camera pixel, at the camera's real height."""
    return sc.K @ np.linalg.inv(sc.E0)[:3, [0, 1, 3]]


def road_map(st, sc, g):
    """F: a road point (camera-centred, metres) -> the point stock shows at the model pixel where this config
    shows it. So the model sees the road at F(road). Exact on the road plane for the camera at its real
    height; img's, unless the config touches big_img alone."""
    if not st.geometric():
        return np.eye(3)
    view = "img" if st.views == "both" else st.views
    G = _road_to_px(sc)
    F = np.linalg.inv(G) @ warp(STOCK, sc, view, g) @ np.linalg.inv(warp(st, sc, view, g)) @ G
    return F / (F @ [10.0, 0.0, 1.0])[2]


def _plan_in_view(xy, g, view, h_b):
    """Road points where the model would put them in its own view: on a road h_b below the camera, through
    openpilot's own geometry (calibrated frame: x forward, y right, z down)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(1) & (xy[:, 0] > 0.5)]
    p = (
        np.c_[xy[:, 0], -xy[:, 1], np.full(len(xy), h_b)]
        @ np.linalg.inv(np.array(g["calib_from_model"][view])).T
    )
    return p[:, :2] / p[:, 2:]


_HORIZON_Y = np.linspace(-4, 4, 81)  # the horizon's calibrated rays, (1, y, 0), from 76 deg left to 76 right


def _rays_to(H, y=_HORIZON_Y):
    """The horizon, calibrated rays (1, y, 0), through H to pixels; NaN where they fall behind."""
    p = np.c_[np.ones_like(y), y, np.zeros_like(y)] @ np.asarray(H, float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p[:, :2] / p[:, 2:]
    out[~(p[:, 2] > 0)] = np.nan
    return out


def _outline(w, h, n=24):
    t = np.linspace(0, 1, n)
    return np.vstack(
        [
            np.c_[t * w, 0 * t],
            np.c_[w + 0 * t, t * h],
            np.c_[(1 - t) * w, h + 0 * t],
            np.c_[0 * t, (1 - t) * h],
        ]
    )


def _inside(uv, w, h):
    """Which of (n, 2) pixels fall inside a w x h image."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    return np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w) & (uv[:, 1] >= 0) & (uv[:, 1] <= h)


def _label_at(ax, uv, text, w, h, end=0, **kw):
    """Label a line at its first (end=0) or last (end=-1) point inside the image, and not at all when none is:
    a label placed outside the axes stretches the saved figure to reach it."""
    inside = np.flatnonzero(_inside(uv, w, h))
    if len(inside):
        ax.text(*np.asarray(uv)[inside[end]], text, clip_on=True, **kw)


def _lattice(W, w, h, step=64):
    """The model's pixel grid every `step` pixels, through W: a list of (n, 2) lines in camera pixels."""
    lines = [apply_h(W, np.c_[np.full(17, x), np.linspace(0, h, 17)]) for x in range(step, w, step)]
    return lines + [apply_h(W, np.c_[np.linspace(0, w, 33), np.full(33, y)]) for y in range(step, h, step)]


def _yuv_rgb(t):
    """(6, h, w) model tensor -> (2h, 2w, 3) RGB, as BT.601 limited range, for drawing."""
    y = NODE.full_luma(t).astype(np.float32)
    u, v = (np.repeat(np.repeat(t[i].astype(np.float32) - 128.0, 2, 0), 2, 1) for i in (4, 5))
    c = 1.164 * (y - 16.0)
    return np.clip(np.stack([c + 1.596 * v, c - 0.392 * u - 0.813 * v, c + 2.017 * u], -1), 0, 255).astype(
        np.uint8
    )


def _frame_half(path):
    """A camera frame at half size (JPEG draft mode), and that scale."""
    img = Image.open(path)
    full = img.width
    img.draft("RGB", (img.width // 2, img.height // 2))
    img = img.convert("RGB")
    return np.asarray(img), img.width / full


def _show_photo(fig, quality=80):
    """Display a figure as JPEG: its camera panels are photos, which PNG stores several times larger."""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", bbox_inches="tight", pil_kwargs={"quality": quality})
    plt.close(fig)
    display({"image/jpeg": base64.b64encode(buf.getvalue()).decode(), "text/plain": "<figure>"}, raw=True)


# --------------------------------------------------------------------------
# Paths, speeds and headings
# --------------------------------------------------------------------------


def plan(r):
    return np.asarray(r["trajectory"], float)[:, :2]


def _head(ex):
    """The pose head's forward speed (m/s)."""
    return float(((ex.get("pose") or {}).get("velocity") or [np.nan])[0])


def _speeds(xy):
    """Each 0.5 s step's mean speed, from the origin on."""
    xy = np.vstack([[0.0, 0.0], np.asarray(xy, float).reshape(-1, 2)])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT


def _headings(xy, sign, min_step):
    """Each step's heading (deg), positive toward the route's side; NaN on steps under min_step metres."""
    xy = np.vstack([[0.0, 0.0], np.asarray(xy, float).reshape(-1, 2)])
    d = np.diff(xy, axis=0)
    psi = np.degrees(np.arctan2(d[:, 1], d[:, 0])) * (sign or 1.0)
    psi[np.linalg.norm(d, axis=1) < min_step] = np.nan
    return psi


def _dpsi(xy, n, min_step):
    """Heading change (deg, left +) over a path's first n waypoints: the last usable step's heading minus the
    first's, a step being usable when it is at least min_step long. NaN without two usable steps."""
    pts = np.vstack([[0.0, 0.0], np.asarray(xy, float)[:n]])
    step = np.diff(pts, axis=0)
    ds = np.linalg.norm(step, axis=1)
    if len(ds) < n or not np.isfinite(ds).all():
        return np.nan
    ok = ds >= min_step
    if not ok.any():
        return np.nan
    first, last = ok.argmax(), n - 1 - ok[::-1].argmax()
    if last <= first:
        return np.nan
    psi = np.arctan2(step[:, 1], step[:, 0])
    return float(np.degrees(np.angle(np.exp(1j * (psi[last] - psi[first])))))


def _off_path(xy, gt, n):
    """The mean distance (m) from a plan's first n waypoints to the GT's path, the polyline from the origin through
    every GT waypoint. Unlike the benchmark's lateral error, which compares waypoints at the same time, it ignores
    when: a plan on the GT's line scores 0 however slowly it drives it."""
    poly = np.vstack([[0.0, 0.0], np.asarray(gt, float).reshape(-1, 2)])
    poly = poly[np.isfinite(poly).all(1)]
    a, ab = poly[:-1], np.diff(poly, axis=0)
    p = np.asarray(xy, float)[:n, None, :2]  # (n, 1, 2) against (segments, 2)
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab**2).sum(-1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(1).mean())


def _mode(dpsi_10s, curv_ratio, p):
    """The open-loop notebook's turn classes, first match wins."""
    if not np.isfinite(dpsi_10s) or abs(dpsi_10s) < p["turn_pred_deg"]:
        return "not_planned"
    if dpsi_10s <= -p["turn_pred_deg"]:
        return "wrong_way"
    if not np.isfinite(curv_ratio):
        return "gt_straight"  # the GT turns less than curve_min_deg within H
    if curv_ratio < 0:
        return "wrong_dir_H"
    if curv_ratio < p["curv_under"]:
        return "understeer"
    if curv_ratio >= p["curv_over"]:
        return "oversteer"
    return "matched"


# compare()'s criteria, each better when lower: (name, what it judges, the value from a metrics() dict).
CRITERIA = (
    ("ADE over 3 s (m)", "ADE", lambda m: m["ade"]),
    ("ADE over the GT's length (m)", "ADE", lambda m: m["ade_long"]),
    ("lateral error over 3 s (m)", "alignment", lambda m: m["lat"]),
    ("distance from the GT's path over 3 s (m)", "alignment", lambda m: m["path_err"]),
    ("|heading error| over 3 s (deg)", "alignment", lambda m: abs(m["dpsi_err"])),
    ("speed RMSE over 3 s (m/s)", "speed profile", lambda m: m["speed_rmse"]),
    ("|1 - start speed / the GT's|", "speed profile", lambda m: abs(1.0 - m["v0_ratio"])),
    ("|1 - pose head / pose speed|", "speed profile", lambda m: abs(1.0 - m["head_ratio"])),
)


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------

_CMDS = ["left", "straight", "right", "other"]  # kitscenes' one-hot driving_command, in that order


def _cmd(row):
    c = [int(v) for v in row["driving_command"]]
    return _CMDS[c.index(1)] if c.count(1) == 1 and c.count(0) == len(c) - 1 else "other"


def _label(sid):
    """kitscenes/<split>/<scene_id>/<keyframe> -> '<scene_id[:8]>/<keyframe>'."""
    _, _, scene_id, kf = sid.split("/")
    return f"{scene_id[:8]}/{kf}"


def _matches(sid, key):
    return sid == key or _label(sid) == key or any(sid.split("/", k)[-1].startswith(key) for k in range(3))


def _mined_heights(spec, cfg):
    """{"<split>/<scene_id>": metres}: scenario mining's front-camera height, where the tags file points."""
    if "json" not in spec:
        return {}
    root = Path(cfg.resolve(spec["json"]))
    out = {}
    for name in spec.get("files", []):
        path = root / name
        if path.exists():
            for k, v in json.loads(path.read_text())["scenes"].items():
                if v.get("height_ring_front_m") is not None:
                    out[k] = float(v["height_ring_front_m"])
    return out


def _repeats(stamps):
    """Model ticks per observation, as the adapter's `_repeats` counts them under `infer_sequence`: a frame's
    step times MODEL_HZ, the first frame taking the sequence's median step."""
    steps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    seed = float(np.median(steps)) / 1e6 if steps else None
    out, prev = [], None
    for t in stamps:
        dt = (t - prev) / 1e6 if prev is not None else seed
        out.append(max(1, round(dt * MODEL_HZ)) if dt is not None and dt > 0 else 1)
        prev = t
    return out


def _older(reps, back=FRAME_SKIP):
    """The observation `img`'s older frame comes from: the one shown `back` ticks before the last tick."""
    i, k = len(reps) - 1, back
    while k >= reps[i]:
        k -= reps[i]
        i -= 1
    return i


# --------------------------------------------------------------------------
# The model nodes
# --------------------------------------------------------------------------


@dataclass
class Node:
    proc: subprocess.Popen
    port: int
    client: httpx.Client
    ex: ThreadPoolExecutor  # one worker: requests to one node run in order, nodes run in parallel

    def post(self, route, body):
        r = self.client.post(f"http://127.0.0.1:{self.port}{route}", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{route}: HTTP {r.status_code}: {r.text[:400]}")
        return r.json()


class Pool:
    """The model nodes: started on first use, each sample pinned to one so its frames are decoded once."""

    def __init__(self, n, device, threads, cfg, logs, base_port):
        self.n, self.device, self.threads, self.cfg, self.logs = n, device, threads, cfg, logs
        self.base_port = int(os.environ.get("PADDOCK_BASE_PORT", base_port))
        self.nodes, self.home = [None] * n, {}

    def _launch(self, i):
        spec = Registry.load().models[MODEL]
        extra = {"PADDOCK_OPENPILOT_THREADS": str(self.threads)}
        if self.device == "cpu":
            extra["CUDA_VISIBLE_DEVICES"] = ""
        env = env_builder.build(spec, self.cfg, extra=extra)
        for k in [k for k in env if k.startswith("PADDOCK_OPENPILOT_") and k != "PADDOCK_OPENPILOT_THREADS"]:
            env.pop(k)  # a site-wide height scale or desire would make "stock" something else
        port = ports.find_free(self.base_port + 10 * i)
        nonce = uuid.uuid4().hex[:12]
        log = self.logs / f"node{i}.log"
        with open(log, "w") as fh:
            p = subprocess.Popen(
                [str(spec.runtime.interpreter(self.cfg, **spec.context(self.cfg))), "-u", str(NODE_PY)]
                + ["--port", str(port), "--nonce", nonce, "--parent", str(os.getpid())],
                env=env,
                cwd=env_builder.working_dir(spec, self.cfg),
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        return p, port, nonce, log

    def start(self, which):
        """Start the nodes in `which` that are not running, all at once, and wait for them."""
        need = [
            i for i in sorted(set(which)) if self.nodes[i] is None or self.nodes[i].proc.poll() is not None
        ]
        if not need:
            return
        t0 = time.time()
        launched = {i: self._launch(i) for i in need}
        for i, (p, port, nonce, log) in launched.items():
            proc.wait_alive(port, 600, p, log, expect_model=NODE.NAME, expect_nonce=nonce)
            self.nodes[i] = Node(p, port, httpx.Client(timeout=3600), ThreadPoolExecutor(1))
        print(f"started {len(need)} node(s) on {self.device} in {time.time() - t0:.0f} s")

    def assign(self, sid):
        """The node a sample runs on: the least loaded, once, so its frames stay in one node's cache."""
        if sid not in self.home:
            load = [sum(h == i for h in self.home.values()) for i in range(self.n)]
            self.home[sid] = int(np.argmin(load))
        return self.home[sid]

    def node(self, sid):
        self.start([self.assign(sid)])
        return self.nodes[self.home[sid]]

    def close(self):
        for i, nd in enumerate(self.nodes):
            if nd is not None:
                nd.ex.shutdown(wait=False, cancel_futures=True)
                nd.proc.terminate()
                try:
                    nd.proc.wait(30)
                except subprocess.TimeoutExpired:
                    nd.proc.kill()
                nd.client.close()
                self.nodes[i] = None


_OPEN = []  # the labs with nodes: one per kernel, so a new Lab stops the last one's


# --------------------------------------------------------------------------
# The lab
# --------------------------------------------------------------------------


class Lab:
    """The turns, the recorded run that planned them, and the model nodes that plan them again.

    scenes: sample ids, labels like "03ed2585/40", or unique prefixes. run_id: the recorded stock run.
    device, nodes, threads: where the model runs. height: the camera's real height, for drawing and for the
    knobs that move the camera over the road: "context" (scenario mining's, per scene), "harness" or metres.
    overrides: tag parameters for this lab only. recompute: ignore the results cached on disk.
    """

    def __init__(
        self,
        scenes,
        run_id,
        *,
        device="cpu",
        nodes=5,
        threads=8,
        height="context",
        overrides=None,
        recompute=False,
        base_port=9520,
    ):
        while _OPEN:
            _OPEN.pop().close()
        self.cfg = SiteConfig.load()
        reg = Registry.load()
        tags = yaml.safe_load(TAGS.read_text())
        bench_params = tags.get("bench_params", {}).get(BENCH, {})
        overrides = dict(overrides or {})
        unknown = sorted((set(overrides) | set(bench_params)) - set(tags["params"]))
        if unknown:
            raise KeyError(f"not tag params: {unknown}")
        self.p = tags["params"] | bench_params | overrides
        self.n_h = round(self.p["horizon_s"] / DT)  # waypoints within H = 3 s, the benchmark's horizon
        self.harness = load_benchmark(reg.benchmarks[BENCH], self.cfg)
        self._rows = {self.harness._sample_id(r): r for r in self.harness.coverage()[0]}
        self.mined = _mined_heights(tags.get("context", {}).get(BENCH, {}), self.cfg)
        self.run_id, self.height, self.device, self.recompute = run_id, height, device, recompute
        self.keys = []
        self.keys = [self.sid(k) for k in scenes]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("scenes names a keyframe twice")
        self.cache = self.cfg.runs / BENCH / "_analysis" / MODEL / "image_tweak"
        for sub in ("results", "geometry", "logs"):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)
        sha = reg.repos["openpilot"].sha
        self.code = hashlib.sha1(b"".join(p.read_bytes() for p in CODE_FILES) + sha.encode()).hexdigest()[:10]
        self.pool = Pool(nodes, device, threads, self.cfg, self.cache / "logs", base_port)
        atexit.register(self.pool.close)
        _OPEN.append(self)
        self._sc, self._geo, self._planes, self._res, self._rec, self._floor = {}, {}, {}, {}, None, {}
        print(
            f"{len(self.keys)} turns; code {self.code} (openpilot {sha[:9]}); model on {device}, {nodes} node(s); "
            f"cache {self._rel(self.cache)}"
        )

    def _rel(self, path):
        """A path under the site's runs root, as {runs}/..., so no machine path reaches the outputs."""
        return "{runs}/" + path.relative_to(self.cfg.runs).as_posix()

    def close(self):
        """Stop the nodes. They also stop with the kernel, and start again on the next call that needs one."""
        self.pool.close()

    # ---- samples ----

    def sid(self, key):
        """The sample id a key names: a sample id, a label like "03ed2585/40", or a unique prefix of either.
        The lab's turns are searched first, then every sample."""
        key = str(key)
        if key in self._rows:
            return key
        for pool in (self.keys, list(self._rows)):
            hit = [s for s in pool if _matches(s, key)]
            if len(hit) == 1:
                return hit[0]
            if len(hit) > 1:
                raise KeyError(f"{key!r} names {len(hit)} samples, e.g. {[_label(s) for s in hit[:4]]}")
        raise KeyError(f"{key!r} names no {BENCH} sample")

    def _real_height(self, row, E):
        """(metres, how): the camera's height over the road, per `height`."""
        if self.height == "context":
            h = self.mined.get(f"{row['split']}/{row['scene_id']}")
            return (h, "the map's") if h is not None else (float(E[2, 3]), "the harness's; no map height")
        if self.height == "harness":
            return float(E[2, 3]), "the harness's"
        return float(self.height), "set"

    def scene(self, key):
        """Everything about one keyframe that no config changes, read once."""
        sid = self.sid(key)
        if sid in self._sc:
            return self._sc[sid]
        row = self._rows[sid]
        path = self.harness._scene_path(row["scene_id"], row["split_dir"])
        kf = int(row["keyframe_index"])
        poses = L.load_poses(path)
        K, E = _read_calibration(path, ground_offset=self.harness._ground_offset())
        K, E = np.asarray(K["CAM_FRONT"], float)[:3, :3], np.array(E["CAM_FRONT"], float)
        h, how = self._real_height(row, E)
        Eh = E.copy()
        Eh[2, 3] = h
        E0 = Eh.copy()
        E0[:2, 3] = 0.0  # the camera over the origin: openpilot plans from the camera, not the rig's origin
        gt = L.ground_truth(poses, kf, n=N_LONG)[:, :2]
        side = _cmd(row)
        nav = re.fullmatch(r"Turn (left|right) in ([\d.]+)\s*m", str(row["nav_text"]).strip())
        sample = self.harness._sample(row)
        obs = sample.observations
        stamps = [int(o.timestamp_us) for o in obs]
        reps = _repeats(stamps)
        older = _older(reps)
        pair_dt = (stamps[-1] - stamps[older]) / 1e6
        self._sc[sid] = SimpleNamespace(
            sid=sid,
            label=_label(sid),
            split=row["split"],
            scene_id=row["scene_id"],
            kf=kf,
            side=side,
            sign={"left": 1.0, "right": -1.0}.get(side, 0.0),  # heading toward the route is + for both sides
            nav=str(row["nav_text"]).strip(),
            nav_m=float(nav.group(2)) if nav else np.nan,
            v=float(L.ego_speed(poses, kf)),  # the pose speed: "the right speed" (the note's Q4)
            poses=poses,
            sample=sample,
            wire=[o.to_wire() for o in obs],
            frame=obs[-1].image_paths["CAM_FRONT"],
            older=older,  # the observation img's older frame comes from
            older_frame=obs[older].image_paths["CAM_FRONT"],
            older_kf=kf - round(pair_dt * L.SRC_HZ),
            pair_dt=pair_dt,  # seconds between img's two frames
            ticks=sum(reps),  # model ticks over the sequence, as the adapter counts them
            gt=np.vstack([gt, np.full((N_LONG - len(gt), 2), np.nan)]),
            gt_n=len(gt),
            K=K,
            E=Eh,
            E0=E0,
            h=h,
            h_how=how,
            h_harness=float(E[2, 3]),
            h_map=self.mined.get(f"{row['split']}/{row['scene_id']}", np.nan),
        )
        return self._sc[sid]

    def recorded(self, key):
        """(plan, extras) as the recorded run planned this turn."""
        if self._rec is None:
            runs = load_runs(self.cfg, BENCH, models=[MODEL], run_ids=[self.run_id])
            self._rec = {s["sample_id"]: s for s in runs[0].samples} if runs else {}
        sid = self.sid(key)
        if sid not in self._rec:
            raise LookupError(f"no per-sample record of {_label(sid)} in {self.run_id} on this machine")
        s = self._rec[sid]
        return np.asarray(s["prediction"], float)[:, :2], s.get("extras") or {}

    def metrics(self, key, xy, extras=None):
        """One plan on one turn, as a dict; the notebook's part 1 lists the keys."""
        sc, p, n_h = self.scene(key), self.p, self.n_h
        xy = np.asarray(xy, float)[:, :2]
        ex = extras or {}
        s = self.harness.score(sc.sample, Trajectory(xy, dt=DT))
        out = {
            "ade": s["ade"],
            "fde_3s": s.get("fde@3s", np.nan),
            "lat": s["lateral_mae"],
            "lon": s["longitudinal_mae"],
        }
        n, step = sc.gt_n, p["min_step_m"]
        out["ade_long"] = float(np.mean(np.linalg.norm(xy[:n] - sc.gt[:n], axis=1)))
        out["path_err"] = _off_path(xy, sc.gt[:n], n_h)
        d, g = _dpsi(xy, n_h, step) * sc.sign, _dpsi(sc.gt, n_h, step) * sc.sign
        out |= {"dpsi": d, "gt_dpsi": g, "dpsi_err": d - g}
        out["curv_ratio"] = d / g if abs(g) >= p["curve_min_deg"] else np.nan
        out["dpsi_10s"] = _dpsi(xy, N_LONG, step) * sc.sign
        out["mode"] = _mode(out["dpsi_10s"], out["curv_ratio"], p)
        out["planned"] = out["mode"] not in ("not_planned", "wrong_way")
        g_long = _dpsi(sc.gt, n, step) * sc.sign
        out["turn_frac"] = (
            _dpsi(xy, n, step) * sc.sign / g_long if abs(g_long) >= p["curve_min_deg"] else np.nan
        )
        k = min(n_h, n)
        out["speed_rmse"] = float(np.sqrt(np.mean((_speeds(xy[:k]) - _speeds(sc.gt[:k])) ** 2)))
        v_plan, v_gt = _speeds(xy[:1])[0], _speeds(sc.gt[:1])[0]
        out["v0_ratio"] = v_plan / v_gt if v_gt >= p["v_go"] else np.nan
        out["head_ratio"] = _head(ex) / sc.v if sc.v >= p["v_go"] else np.nan
        out["plan_v0"] = float(ex.get("plan_v0", np.nan))
        out["cam_h"] = float((ex.get("road") or {}).get("camera_height", np.nan))
        return out

    # ---- the model ----

    def _geo_path(self, sid):
        return (
            self.cache / "geometry" / (hashlib.sha1(f"{sid} {self.code}".encode()).hexdigest()[:16] + ".json")
        )

    def geometry(self, key):
        """The camera as openpilot sees it for one sample (`/geometry`), cached on disk. Also decodes the
        sample's frames in its node."""
        sid = self.sid(key)
        if sid not in self._geo:
            path = self._geo_path(sid)
            if path.exists():
                self._geo[sid] = json.loads(path.read_text())
            else:
                g = self.pool.node(sid).post("/geometry", {"observations": self.scene(sid).wire})
                path.write_text(json.dumps(g))
                self._geo[sid] = g
        return self._geo[sid]

    def _prefetch(self, sids):
        """`/geometry` for each sample not yet on disk, in parallel across nodes."""
        need = [s for s in dict.fromkeys(sids) if s not in self._geo and not self._geo_path(s).exists()]
        if need:
            self.pool.start({self.pool.assign(s) for s in need})
            for f in [self.pool.node(s).ex.submit(self.geometry, s) for s in need]:
                f.result()

    def planes(self, key, older=False):
        """The keyframe's (Y, U, V) planes, or (older) those of the frame `img` pairs it with, converted as
        the adapter converts them."""
        sc = self.scene(key)
        k = (sc.sid, older)
        if k not in self._planes:
            self._planes[k] = rgb_to_yuv420(load_frame(sc.older_frame if older else sc.frame))
        return self._planes[k]

    def model_input(self, config, key, view, older=False):
        """(6, 128, 256) uint8: what the model is handed for this view under a config, on the keyframe or
        (older) the frame `img` pairs it with. Built with the node's own `View`, so it can be drawn without
        running the model; every fresh result checks it against what the node fed."""
        st, sc = as_setting(config), self.scene(key)
        g = self.geometry(sc.sid)
        spec = view_specs(st, sc, g).get(view, {})
        v = NODE.View.from_warp(
            warp(st, sc, view, g).astype(np.float32),
            spec,
            tuple(g["model_size"][view]),
            tuple(g["source_size"]),
        )
        return v.apply(*self.planes(sc.sid, older))

    def _job(self, sid, body):
        """A result's cache key: the exact matrices and tables sent, the sample, the device and the code."""
        blob = {"sample": sid, "views": body.get("views"), "device": self.device, "code": self.code}
        return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]

    def fetch(self, jobs, inputs=False):
        """[result] for [(config, key)]: from this session, the disk cache, or the nodes, in parallel.

        A result is /run's reply (trajectory, extras, warps, in_bounds, ticks, ms), or {"error": ...} for a
        config that cannot be shown. inputs=True runs afresh and keeps the model's keyframe tensors."""
        jobs = [(as_setting(st), self.sid(k)) for st, k in jobs]
        self._prefetch([sid for _, sid in jobs])
        out, where, todo = [None] * len(jobs), {}, {}
        for i, (st, sid) in enumerate(jobs):
            try:
                body = request(st, self.scene(sid), self.geometry(sid))
            except ValueError as e:
                out[i] = {"error": str(e)}
                continue
            h = self._job(sid, body)
            where.setdefault(h, []).append(i)
            have = self._res.get(h)
            if inputs:
                if have is None or "inputs" not in have:
                    todo[h] = (st, sid, body)
            elif have is None:
                path = self.cache / "results" / f"{h}.json"
                if path.exists() and not self.recompute:
                    self._res[h] = json.loads(path.read_text())
                else:
                    todo[h] = (st, sid, body)
        if todo:
            self._run(todo, keep_inputs=inputs)
        for h, idx in where.items():
            for i in idx:
                out[i] = self._res[h]
        return out

    def _run(self, todo, keep_inputs):
        """Run {hash: (setting, sid, body)} on the nodes. Each fresh result is checked on the way in: the
        keyframe tensors `model_input` builds, and so draws, must be the ones the model was fed."""
        t0 = time.time()
        self.pool.start({self.pool.assign(sid) for _, sid, _ in todo.values()})
        futs = {}
        for h, (_, sid, body) in todo.items():
            nd = self.pool.node(sid)
            futs[h] = nd.ex.submit(nd.post, "/run", body | {"inputs": True})
        failed = 0
        for h, f in futs.items():
            st, sid, _ = todo[h]
            try:
                r = f.result()
            except Exception as e:  # recorded, not raised: one bad config should not lose the rest
                self._res[h], failed = {"error": f"{type(e).__name__}: {e}"}, failed + 1
                continue
            for v in VIEWS:
                w, hh = self.geometry(sid)["model_size"][v]
                fed = np.frombuffer(base64.b64decode(r["inputs"][v]), np.uint8).reshape(6, hh // 2, w // 2)
                if not np.array_equal(self.model_input(st, sid, v), fed):
                    raise AssertionError(
                        f"{_label(sid)}, {st.label()}, {v}: what is drawn is not what the model was fed"
                    )
            kept = {k: v for k, v in r.items() if k != "inputs"}
            (self.cache / "results" / f"{h}.json").write_text(json.dumps(kept))
            self._res[h] = r if keep_inputs else kept
        ms = [self._res[h].get("ms", np.nan) for h in futs]
        print(
            f"ran {len(futs)} on {len({self.pool.home[todo[h][1]] for h in futs})} node(s) in {time.time() - t0:.0f} s "
            f"(median {np.nanmedian(ms) / 1000:.1f} s each){f'; {failed} FAILED' if failed else ''}"
        )

    # ---- drawing ----

    def _headline(self, sc):
        return f"{sc.label}: {sc.nav} at {sc.v:.1f} m/s; camera {sc.h:.2f} m over the road ({sc.h_how})"

    @staticmethod
    def _summary(name, m):
        return (
            f"{name}: ADE {m['ade']:.2f} m, lateral {m['lat']:.2f} m, off the GT's path {m['path_err']:.2f} m; "
            f"heading over 3 s {m['dpsi']:+.0f} of the GT's "
            f"{m['gt_dpsi']:+.0f} deg ({m['mode']}); start speed x{m['v0_ratio']:.2f} of the GT's, pose head "
            f"x{m['head_ratio']:.2f} of the pose speed; camera height read {m['cam_h']:.2f} m"
        )

    def _camera(self, ax, sc, g, layers, road=True, plans=()):
        """The keyframe at half size: each view's footprint (img solid, big_img dashed) and, for the last layer,
        its model-pixel grid every 64 px; the camera's horizon (dotted); the road at the camera's real height.
        layers: [(setting, colour)], colour None for each view's own. plans: the GT and these drawn on it."""
        photo, s = _frame_half(sc.frame)
        ax.imshow(photo)
        if plans:
            for xy, c, lw in [(sc.gt[: sc.gt_n], GT_C, 2.5)] + [(p[0], p[2], 2.0) for p in plans]:
                uv = _to_cam(xy, sc) * s
                ax.plot(uv[:, 0], uv[:, 1], "o-", color=c, lw=lw, ms=3, path_effects=OUTLINE)
        if road:
            for x in ROAD_X:
                uv = _to_cam(np.c_[np.full(33, x), np.linspace(-8, 8, 33)], sc, sc.E0) * s
                ax.plot(uv[:, 0], uv[:, 1], color="white", lw=0.8, alpha=0.7)
                if len(uv):
                    ax.text(
                        *uv[-1], f"{x} m", color="white", fontsize=8, ha="right", va="bottom", clip_on=True,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
                    )  # fmt: skip
            for y in ROAD_Y:
                uv = _to_cam(np.c_[np.linspace(5, max(ROAD_X), 56), np.full(56, y)], sc, sc.E0) * s
                ax.plot(uv[:, 0], uv[:, 1], color="white", lw=0.6, alpha=0.5)
        for i, (st, colour) in enumerate(layers):
            last = i == len(layers) - 1
            for v in VIEWS:
                c, ls = VIEW_STYLE[v]
                c = colour or c
                w, h = g["model_size"][v]
                W = warp(st, sc, v, g)
                uv = apply_h(W, _outline(w, h)) * s
                ax.plot(uv[:, 0], uv[:, 1], ls, color=c, lw=2.0 if last else 1.2)
                if last:
                    for line in _lattice(W, w, h):
                        ax.plot(line[:, 0] * s, line[:, 1] * s, color=c, lw=0.5, alpha=0.6)
        hz = _rays_to(np.array(g["camera_from_calib"])) * s
        ax.plot(hz[:, 0], hz[:, 1], ":", color="white", lw=1.2)
        ax.set_xlim(0, photo.shape[1])
        ax.set_ylim(photo.shape[0], 0)
        ax.set_axis_off()

    def _view(self, ax, sc, g, st, view, older, overlay=False, plans=(), horizons=False, title=None):
        """One model input as the model was handed it, YUV drawn as RGB at its full 512 x 256. overlay: the GT
        and each plan on it, through the real camera at its real height, then the view's own warp; the title
        counts the waypoints in view, since a sharp turn leaves img's 31 deg within a few metres."""
        w, h = g["model_size"][view]
        ax.imshow(_yuv_rgb(self.model_input(st, sc.sid, view, older)), extent=(0, w, h, 0))
        Wi = np.linalg.inv(warp(st, sc, view, g))
        counts = []
        if overlay:
            paths = [("GT", sc.gt[: sc.gt_n], GT_C, 2.5)] + [("plan", p[0], p[2], 2.0) for p in plans]
            for name, xy, c, lw in paths:
                uv = apply_h(Wi, _to_cam(xy, sc))
                ax.plot(uv[:, 0], uv[:, 1], "o-", color=c, lw=lw, ms=3, path_effects=OUTLINE)
                counts.append(f"{name} {int(_inside(uv, w, h).sum())}/{len(xy)}")
        if horizons:
            real = apply_h(Wi, _rays_to(np.array(g["camera_from_calib"])))
            ax.plot(real[:, 0], real[:, 1], "--", color="white", lw=1)
            trained = _rays_to(np.linalg.inv(np.array(g["calib_from_model"][view])))
            ax.plot(trained[:, 0], trained[:, 1], ":", color="white", lw=1)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        when = f"t - {sc.pair_dt:.1f} s (frame {sc.older_kf})" if older else f"t, the keyframe ({sc.kf})"
        title = title or f"{view}: {when}"
        if counts:
            title += "\nwaypoints in view: " + ", ".join(counts)
        ax.set_title(title, fontsize=10)

    def _bev(self, ax, sc, plans):
        n_h, gt = self.n_h, sc.gt[: sc.gt_n]
        ax.plot(gt[:, 1], gt[:, 0], "o-", color=GT_C, lw=3, ms=3, label="GT")
        if sc.gt_n >= n_h:
            ax.plot(gt[n_h - 1, 1], gt[n_h - 1, 0], "o", ms=12, mfc="none", mec=GT_C)
        for xy, _, c, label in plans:
            ax.plot(xy[:, 1], xy[:, 0], "o-", color=c, lw=2, ms=3, label=label)
            ax.plot(xy[n_h - 1, 1], xy[n_h - 1, 0], "o", ms=12, mfc="none", mec=c)
        ax.plot(0, 0, "ks", ms=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("lateral (m, left +)")
        ax.set_ylabel("forward (m)")
        ax.set_title("BEV: GT over the scene, plans over 10 s;\neach circled at 3 s", fontsize=10)

    def _speed(self, ax, sc, plans, legend=True):
        t = DT * (np.arange(N_LONG) + 0.5)  # each step's mean speed, at its midpoint
        ax.plot(t[: sc.gt_n], _speeds(sc.gt[: sc.gt_n]), "o-", color=GT_C, lw=2, ms=3, label="GT")
        ax.plot(0, sc.v, ">", color="black", ms=10, label="▶ pose speed")
        for xy, ex, c, label in plans:
            ax.plot(t, _speeds(xy), "o-", color=c, lw=2, ms=3, label=label)
            ax.plot(0, _head(ex), "<", color=c, mec="black", ms=10)
            ax.plot(0, ex.get("plan_v0", np.nan), "D", color=c, mec="black", ms=6)
        ax.axvline(self.p["horizon_s"], color="black", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
        ax.set_xlabel("t (s)")
        ax.set_title(
            "speed per 0.5 s step (m/s)\nat t = 0: ▶ pose speed, ◀ pose head, ◆ plan_v0", fontsize=10
        )
        if legend:
            ax.legend(fontsize=8, loc="best")

    def _heading(self, ax, sc, plans):
        t, step = DT * (np.arange(N_LONG) + 0.5), self.p["min_step_m"]
        ax.plot(t[: sc.gt_n], _headings(sc.gt[: sc.gt_n], sc.sign, step), "o-", color=GT_C, lw=2, ms=3)
        for xy, _, c, _ in plans:
            ax.plot(t, _headings(xy, sc.sign, step), "o-", color=c, lw=2, ms=3)
        ax.axvline(self.p["horizon_s"], color="black", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
        ax.set_xlabel("t (s)")
        ax.set_title(f"heading per step, toward the route ({sc.side}) (deg)", fontsize=10)

    def _road_map(self, ax, sc, g, st):
        """Where the config moves the road in the model's eyes, relative to stock (`road_map`): a grid of 5 m
        squares to 60 m, grey as it is and red as the model sees it."""
        F = road_map(st, sc, g)
        for x in np.arange(5, 61, 5):
            seg = np.c_[np.full(41, x), np.linspace(-10, 10, 41)]
            ax.plot(seg[:, 1], seg[:, 0], color="#cccccc", lw=0.8)
            m = apply_h(F, seg)
            ax.plot(m[:, 1], m[:, 0], color="#d62728", lw=0.9)
        for y in np.arange(-10, 11, 5):
            seg = np.c_[np.linspace(3, 60, 58), np.full(58, y)]
            ax.plot(seg[:, 1], seg[:, 0], color="#cccccc", lw=0.8)
            m = apply_h(F, seg)
            ax.plot(m[:, 1], m[:, 0], color="#d62728", lw=0.9)
        ax.set_xlim(15, -15)
        ax.set_ylim(0, 75)
        ax.set_xlabel("lateral (m, left +)")
        ax.set_ylabel("forward (m)")
        ax.set_title("the road (grey), as the config shows it to the model (red)", fontsize=10)

    # ---- part 1 ----

    def recorded_table(self, keys=None):
        """Part 1: each turn as the recorded run planned it, with metrics()."""
        rows = []
        for sid in [self.sid(k) for k in (keys or self.keys)]:
            sc = self.scene(sid)
            rows.append(
                {"key": sc.label, "route": f"{sc.side} in {sc.nav_m:g} m", "v": sc.v, "h_map": sc.h_map}
                | self.metrics(sid, *self.recorded(sid))
            )
        cols = ["route", "v", "mode", "ade", "ade_long", "lat", "path_err", "dpsi", "gt_dpsi", "turn_frac"]
        cols += ["speed_rmse"]
        cols += ["v0_ratio", "head_ratio", "plan_v0", "cam_h", "h_map"]
        return pd.DataFrame(rows).set_index("key")[cols]

    def show_recorded(self, key, overlay=("img",), quality=80):
        """Part 1: one turn as the recorded run planned it. overlay: where the GT and the plan are drawn: "img"
        and "big_img", on that view's keyframe; "camera", on the keyframe as the camera took it."""
        sc = self.scene(key)
        g = self.geometry(sc.sid)
        xy, ex = self.recorded(sc.sid)
        plans = [(xy, ex, STOCK_C, f"recorded plan, {self.run_id}")]
        fig = plt.figure(figsize=(22, 11.5))
        gs = fig.add_gridspec(
            2, 4, height_ratios=[2.4, 0.95], hspace=0.22, wspace=0.2, top=0.89, bottom=0.02, left=0.02, right=0.98
        )  # fmt: skip
        ax = fig.add_subplot(gs[0, 0:2])
        self._camera(ax, sc, g, [(STOCK, None)], plans=plans if "camera" in overlay else ())
        ax.set_title(
            "the keyframe: the footprints of img (yellow) and big_img (cyan, dashed), their model pixels every 64;\n"
            "the camera's horizon (dotted) and the road, 10-60 m ahead and every 3 m across",
            fontsize=10,
        )
        self._bev(fig.add_subplot(gs[0, 2]), sc, plans)
        right = gs[0, 3].subgridspec(2, 1, hspace=0.4)
        self._speed(fig.add_subplot(right[0]), sc, plans, legend=False)
        self._heading(fig.add_subplot(right[1]), sc, plans)
        for j, (v, older) in enumerate(PANELS):
            self._view(
                fig.add_subplot(gs[1, j]),
                sc,
                g,
                STOCK,
                v,
                older,
                overlay=v in overlay and not older,
                plans=plans,
            )
        fig.suptitle(
            self._headline(sc) + "\n" + self._summary("recorded", self.metrics(sc.sid, xy, ex)), fontsize=12
        )
        _show_photo(fig, quality)

    def show_warps(self, key=None, quality=80):
        """Part 1: the stock warps and the road's homography into each view. The matrices on one turn (key, the
        first by default); then, on every turn, the road drawn through its homography into both views; then a
        table of each turn's camera."""
        sc = self.scene(key or self.keys[0])
        g = self.geometry(sc.sid)
        C, G = np.array(g["camera_from_calib"]), _road_to_px(sc)
        mats = {"camera_from_calib: calibrated ray -> camera pixel (the camera's K, rotated by its rpy)": C}
        for v in VIEWS:
            mats[
                f"calib_from_model[{v}]: model pixel -> calibrated ray (openpilot's own; the same for every car)"
            ] = np.array(g["calib_from_model"][v])
        for v in VIEWS:
            mats[f"W[{v}] = camera_from_calib @ calib_from_model[{v}]: model pixel -> camera pixel"] = warp(
                STOCK, sc, v, g
            )
        mats[f"G: road point (x forward, y left, 1), metres -> camera pixel, the camera {sc.h:.2f} m up"] = G
        for v in VIEWS:
            mats[f"inv(W[{v}]) @ G: road point -> {v} pixel"] = np.linalg.inv(warp(STOCK, sc, v, g)) @ G
        fmt = {"float_kind": lambda x: f"{x:11.4g}"}
        print(f"{sc.label}, the stock warps and the road's homographies:\n")
        for name, M in mats.items():
            print(f"{name}\n{np.array2string(np.asarray(M), formatter=fmt)}\n")

        stroke = [patheffects.withStroke(linewidth=2, foreground="black")]
        fig, axes = plt.subplots(len(self.keys), 2, figsize=(18, 4.6 * len(self.keys)), squeeze=False)
        for row, sid in zip(axes, self.keys):
            s, gs_ = self.scene(sid), self.geometry(sid)
            for ax, v in zip(row, VIEWS):
                w, h = gs_["model_size"][v]
                ax.imshow(_yuv_rgb(self.model_input(STOCK, sid, v)), extent=(0, w, h, 0))
                Hv = np.linalg.inv(warp(STOCK, s, v, gs_)) @ _road_to_px(s)
                for x in (*ROAD_X, 80):
                    uv = apply_h(Hv, np.c_[np.full(41, x), np.linspace(-10, 10, 41)])
                    ax.plot(uv[:, 0], uv[:, 1], color="white", lw=1)
                    # the line's rightmost point in view (it runs right to left), the label to its left
                    _label_at(
                        ax, uv, f"{x} m ", w, h, color="white", fontsize=8, ha="right", va="bottom",
                        path_effects=stroke,
                    )  # fmt: skip
                for y in (-10, -6, -3, 0, 3, 6, 10):
                    uv = apply_h(Hv, np.c_[np.linspace(3, 80, 78), np.full(78, y)])
                    ax.plot(uv[:, 0], uv[:, 1], color="white", lw=0.7, alpha=0.7)
                trained = _rays_to(np.linalg.inv(np.array(gs_["calib_from_model"][v])))
                ax.plot(trained[:, 0], trained[:, 1], ":", color="white", lw=1)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{s.label}: {v} at t, the road through inv(W[{v}]) @ G", fontsize=10)
        fig.suptitle(
            "the road plane in each model view: lines across at 10-80 m, along at 0, ±3, ±6 and ±10 m; "
            "dotted, the horizon",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))  # the suptitle's 0.8 inch
        _show_photo(fig, quality)

        rows = []
        for sid in self.keys:
            s, gs_ = self.scene(sid), self.geometry(sid)
            row = {
                "key": s.label,
                "roll, pitch, yaw (deg)": ", ".join(f"{np.degrees(a):+.2f}" for a in gs_["rpy"]),
            }
            row["focal length (px)"] = s.K[0, 0]
            for v in VIEWS:
                (w, h), W = gs_["model_size"][v], warp(STOCK, s, v, gs_)
                c = [apply_h(W, [[w / 2 + dx, h / 2]])[0] for dx in (0, 1)]
                row[f"{v}: camera px per model px"] = float(np.linalg.norm(c[1] - c[0]))
            rows.append(row)
        span = []
        for v in VIEWS:
            (w, h), M = g["model_size"][v], np.array(g["calib_from_model"][v])
            r = M @ np.array([[0, w, w / 2, w / 2], [h / 2, h / 2, 0, h], [1, 1, 1, 1]], float)
            lr = [np.degrees(np.arctan2(r[1, i], r[0, i])) for i in (0, 1)]
            tb = [np.degrees(np.arctan2(r[2, i], r[0, i])) for i in (2, 3)]
            span.append(f"{v} {lr[0]:+.1f} to {lr[1]:+.1f} deg across, {tb[0]:+.1f} to {tb[1]:+.1f} down")
        print("every camera's views span (right and down +): " + "; ".join(span))
        display(pd.DataFrame(rows).set_index("key").round(2))

    # ---- part 2 ----

    def show_measurements(self, key=None, quality=80):
        """Part 2: on one turn, each camera height and each speed, with how it is measured; then the frames the
        model reads its motion from, and the road's distances as the model reads them."""
        sc = self.scene(key or self.keys[0])
        g = self.geometry(sc.sid)
        xy, ex = self.recorded(sc.sid)
        h_b = float((ex.get("road") or {}).get("camera_height", np.nan))
        head, v0 = _head(ex), float(ex.get("plan_v0", np.nan))
        n_pair = round(sc.pair_dt * L.SRC_HZ)
        v_pair = float(np.linalg.norm(sc.poses[sc.kf, :2] - sc.poses[sc.kf - n_pair, :2]) / sc.pair_dt)
        v_gt, v_plan = _speeds(sc.gt[:1])[0], _speeds(xy[:1])[0]

        def table(title, head_row, rows):
            lines = [f"**{title}**", "", f"| {' | '.join(head_row)} |", "|" + "---|" * len(head_row)]
            return "\n".join(lines + [f"| {' | '.join(r)} |" for r in rows])

        ratio = h_b / sc.h_map
        heights = [
            ("the harness's", f"{sc.h_harness:.3f}", f"calib.json's camera z in the rig's base frame, plus `POD_HEIGHT_M` = {POD_HEIGHT_M} m, a fixed guess at the base frame's height over the road (`_read_calibration`). openpilot never reads it: its warp is a rotation, and has no height in it."),
            ("the map's", f"{sc.h_map:.3f}", "scenario mining's `height_ring_front_m`: on each of the scene's frames on the map, the front camera's height over the Lanelet2 road surface beneath it, from the INS pose; then the median over those frames. This is the real height the notebook draws the road at."),
            ("**the model's** (`cam_h`)", f"**{h_b:.3f}**", "`extras.road.camera_height`: the z of supercombo's `road_transform` head, where it puts the road plane below the camera, on the keyframe's tick. Nothing the model is given holds a height: it infers this from the images."),
            ("openpilot's own", f"{H_M:.2f}", "comma's camera over the road, the height supercombo was trained at."),
        ]  # fmt: skip
        speeds = [
            ("▶ **the pose speed**", f"**{sc.v:.2f}**", f"`loader.ego_speed`: \\|p(k+1) − p(k−1)\\| / 0.2 s, the INS poses at 10 Hz either side of the keyframe k = {sc.kf}. The right speed."),
            ("the pose speed over `img`'s pair", f"{v_pair:.2f}", f"\\|p(k) − p(k−{n_pair})\\| / {sc.pair_dt:.1f} s, between exactly the two frames `img` holds, {sc.older_kf} and {sc.kf}."),
            ("the GT's first step", f"{v_gt:.2f}", "\\|GT(0.5 s)\\| / 0.5 s: where the poses put the car half a second on."),
            ("◀ **the pose head**", f"**{head:.2f}**", "`extras.pose.velocity[0]`: the forward speed from supercombo's `pose` head, on the keyframe's tick. Nothing the model is given holds a speed: it reads its motion from the images (`img` and `big_img` each hold two frames 0.2 s apart) and from its own features of earlier ticks."),
            ("◆ `plan_v0`", f"{v0:.2f}", "`extras.plan_v0`: the plan's own velocity at t = 0, a separate output from the pose head. It is the one openpilot's longitudinal control reads."),
            ("the plan's first step", f"{v_plan:.2f}", "\\|plan(0.5 s)\\| / 0.5 s: what the score sees."),
        ]  # fmt: skip
        ratios = [
            ("`head_ratio` = ◀ / ▶", f"**{head / sc.v:.3f}**", "0.877 is the note's median over 205 keyframes: 0.817 where the model reads its height low (near 1.39 m), 0.915 where it reads it high (near 1.62 m)."),
            ("`v0_ratio` = the plan's first step / the GT's", f"{v_plan / v_gt:.3f}", "median 0.909"),
            ("◆ / ◀", f"{v0 / head:.3f}", "median 1.024: the plan starts at about the speed the pose head reads."),
            ("the model's height / the map's", f"{ratio:.3f}", "where `head_ratio` would be if the model measured distance by geometry alone: a road point X ahead is read at X · cam_h / h."),
            (f"(the model's height / the map's)^{SPEED_EXPONENT}", f"{ratio**SPEED_EXPONENT:.3f}", "where the note's fit across keyframes puts it (exponent CI 0.62–0.75): the speed is misread less than the height."),
        ]  # fmt: skip
        md = [
            f"**{self._headline(sc)}.** From the recorded run, {self.run_id}.",
            table("The camera's height over the road (m)", ("height", "m", "how it is measured"), heights),
            table("The speeds at t = 0 (m/s)", ("speed", "m/s", "how it is measured"), speeds),
            table("Ratios", ("ratio", "this turn", "for reference"), ratios),
        ]
        display(Markdown("\n\n".join(md)))

        t_old, t_key = (
            self.model_input(STOCK, sc.sid, "img", older=True),
            self.model_input(STOCK, sc.sid, "img"),
        )
        fig = plt.figure(figsize=(22, 12.5))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1, 1.75], hspace=0.2, wspace=0.12, top=0.94, bottom=0.04, left=0.03, right=0.98
        )  # fmt: skip
        w, h = g["model_size"]["img"]
        for j, (t, when) in enumerate(
            ((t_old, f"t - {sc.pair_dt:.1f} s (frame {sc.older_kf})"), (t_key, f"t ({sc.kf})"))
        ):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(_yuv_rgb(t), extent=(0, w, h, 0))
            ax.set_title(f"img at {when}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 2])
        d = np.abs(NODE.full_luma(t_key).astype(np.int16) - NODE.full_luma(t_old).astype(np.int16))
        ax.imshow(d, cmap="magma", vmin=0, vmax=max(1.0, float(np.percentile(d, 99.5))), extent=(0, w, h, 0))
        ax.set_title("|luma(t) - luma(t - 0.2 s)|: the change between img's two frames", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[1, 0:2])
        ax.imshow(_yuv_rgb(t_key), extent=(0, w, h, 0))
        Wi, lat = np.linalg.inv(warp(STOCK, sc, "img", g)), np.linspace(-6, 6, 25)
        stroke = [patheffects.withStroke(linewidth=2, foreground="black")]
        for x in (10, 20, 30, 40):
            true = apply_h(Wi, _to_cam(np.c_[np.full(25, x), lat], sc, sc.E0))
            read = _plan_in_view(np.c_[np.full(25, x), lat], g, "img", h_b)
            ax.plot(true[:, 0], true[:, 1], color="#4dac26", lw=2)
            ax.plot(read[:, 0], read[:, 1], "--", color="#d62728", lw=2)
            # Both rows run right to left: the true one is labelled at its right end in view, the read one at
            # its left, so the two labels never sit on each other.
            _label_at(
                ax, true, f"{x} m ", w, h, 0, color="#4dac26", fontsize=10, ha="right", va="bottom",
                path_effects=stroke,
            )  # fmt: skip
            _label_at(
                ax, read, f" reads {x} m", w, h, -1, color="#d62728", fontsize=10, ha="left", va="bottom",
                path_effects=stroke,
            )  # fmt: skip

        def in_view(uv, n):
            return f"{int(_inside(uv, w, h).sum())}/{n} waypoints in view"

        gt = apply_h(Wi, _to_cam(sc.gt[: sc.gt_n], sc))
        ax.plot(
            gt[:, 0], gt[:, 1], "o-", color=GT_C, lw=2.5, ms=3, path_effects=OUTLINE,
            label=f"GT ({in_view(gt, sc.gt_n)})",
        )  # fmt: skip
        pl = apply_h(Wi, _to_cam(xy, sc))
        ax.plot(
            pl[:, 0], pl[:, 1], "o-", color=STOCK_C, lw=2, ms=3, path_effects=OUTLINE,
            label=f"plan, on the real road ({in_view(pl, len(xy))})",
        )  # fmt: skip
        pb = _plan_in_view(xy, g, "img", h_b)
        ax.plot(
            pb[:, 0], pb[:, 1], "o:", color=STOCK_C, lw=2, ms=3,
            label=f"plan, on the road the model reads, {h_b:.2f} m down ({in_view(pb, len(xy))})",
        )  # fmt: skip
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(
            f"img at t. Green: the road 10-40 m ahead, the camera {sc.h:.2f} m up.\nRed: the rows the model reads "
            f"as those distances, at its {h_b:.2f} m. So it reads every distance x{h_b / sc.h:.2f}",
            fontsize=10,
        )
        self._speed(fig.add_subplot(gs[1, 2]), sc, [(xy, ex, STOCK_C, "recorded plan")])
        fig.suptitle(self._headline(sc), fontsize=12)
        _show_photo(fig, quality)

    # ---- checks ----

    def checks(self, key=None, tol_m=TOL_M):
        """Raise unless: (1) the node is the stock adapter, bit for bit, with no config and with an identity
        config sent as every config is (calib and post); (2) what this module draws is what the model was fed,
        on every path through `View`; (3) the node's stock reproduces the recorded run on every turn, within
        tol_m; (4) the ticks counted here, from which the older frame is found, are the node's."""
        sc = self.scene(key or self.keys[0])
        self._prefetch(self.keys)
        nd, g = self.pool.node(sc.sid), self.geometry(sc.sid)
        body = request(STOCK, sc, g)
        ref = nd.post("/stock", {"observations": body["observations"]})
        eye = np.eye(3).tolist()
        identity = body | {"views": {v: {"calib": eye, "post": eye} for v in VIEWS}}
        for what, b in (("no config", body), ("an identity config, sent as configs are", identity)):
            r = nd.post("/run", b)
            same = {
                "plan": r["trajectory"] == ref["trajectory"],
                "extras": json.dumps(r["extras"], sort_keys=True)
                == json.dumps(ref["extras"], sort_keys=True),
                "warps": r["warps"] == ref["warps"],
                "ticks": r["ticks"] == ref["ticks"],
            }
            print(f"1. {sc.label}: the node with {what}, against the untouched adapter, the same: {same}")
            assert all(same.values()), f"the node with {what} is not the stock adapter"

        for st in (STOCK, EVERY_PATH):
            r = self.fetch([(st, sc.sid)], inputs=True)[0]
            for v in VIEWS:
                fed = np.frombuffer(base64.b64decode(r["inputs"][v]), np.uint8).reshape(6, 128, 256)
                ok = np.array_equal(self.model_input(st, sc.sid, v), fed)
                print(f"2. {sc.label}, {st.label()}, {v}: drawn == fed: {ok}")
                assert ok, f"model_input() is not what the model was fed ({st.label()}, {v})"

        rows = []
        for sid, r in zip(self.keys, self.fetch([(STOCK, s) for s in self.keys])):
            s, (rec, rex) = self.scene(sid), self.recorded(sid)
            mine = plan(r)
            rows.append(
                {
                    "key": s.label,
                    "largest difference within 3 s (m)": np.abs(mine[: self.n_h] - rec[: self.n_h]).max(),
                    "over 10 s (m)": np.abs(mine - rec).max(),
                    "ADE here - recorded (m)": self.metrics(sid, mine)["ade"]
                    - self.metrics(sid, rec, rex)["ade"],
                    "ticks, node": r["ticks"],
                    "ticks, counted here": s.ticks,
                    "img's older frame": f"{s.older_kf} (observation {s.older - len(s.wire)})",
                }
            )
        df = pd.DataFrame(rows).set_index("key")
        print(f"3 and 4. stock, run here on {self.device}, against {self.run_id}, recorded on a GPU:")
        display(df.round(3))
        assert df["largest difference within 3 s (m)"].max() < tol_m, (
            f"the node does not reproduce {self.run_id}"
        )
        assert (df["ticks, node"] == df["ticks, counted here"]).all(), (
            "the ticks counted here are not the node's"
        )

    # ---- part 3 ----

    def tweak(self, key, config=None, run=True, overlay=("img",), noise=True, quality=80):
        """Part 3: one config on one turn, beside stock, both run here. run=False draws only what the model would
        be shown, in a second or two. overlay: where the GT and the plans are drawn: "img" and "big_img", on that
        view's keyframe, with the camera's horizon (dashed) and the one the model was trained to expect (dotted);
        "camera", on the keyframe as the camera took it. noise: add a row of the largest change any of the eight
        half-pixel SHIFTS makes on this turn, which a change must beat to be the config's doing."""
        st = as_setting(config)
        sc = self.scene(key)
        g = self.geometry(sc.sid)
        view_specs(st, sc, g)  # a view that would look behind the camera raises here, before anything runs
        shifts = list(SHIFTS) if run and noise else []
        res = self.fetch([(STOCK, sc.sid)] + ([(st, sc.sid)] if run else []) + [(s, sc.sid) for s in shifts])
        plans = [(plan(res[0]), res[0]["extras"], STOCK_C, "stock")]
        if run and "error" in res[1]:
            print(f"{st.label()}: {res[1]['error']}")
        elif run:
            plans.append((plan(res[1]), res[1]["extras"], CONFIG_C, "config"))
        fig = plt.figure(figsize=(22, 18.5))
        gs = fig.add_gridspec(
            3, 4, height_ratios=[2.35, 1.1, 2.0], hspace=0.25, wspace=0.2, top=0.94, bottom=0.01, left=0.02,
            right=0.98,
        )  # fmt: skip
        views = gs[2, :].subgridspec(2, 4, hspace=0.25, wspace=0.2)  # stock's four inputs over the config's
        ax = fig.add_subplot(gs[0, 0:2])
        self._camera(ax, sc, g, [(STOCK, GREY), (st, None)], plans=plans if "camera" in overlay else ())
        ax.set_title(
            "the keyframe: stock's footprints grey; the config's img (yellow) and big_img (cyan, dashed),\n"
            "their model pixels every 64; the camera's horizon (dotted) and the road",
            fontsize=10,
        )
        self._road_map(fig.add_subplot(gs[0, 2]), sc, g, st)
        self._bev(fig.add_subplot(gs[0, 3]), sc, plans)
        self._speed(fig.add_subplot(gs[1, 0:2]), sc, plans)
        self._heading(fig.add_subplot(gs[1, 2:4]), sc, plans)
        for i, (which, name, pl) in enumerate(((STOCK, "stock", plans[:1]), (st, "config", plans[1:]))):
            for j, (v, older) in enumerate(PANELS):
                when = f"t - {sc.pair_dt:.1f} s" if older else "t"
                drawn = v in overlay and not older
                self._view(
                    fig.add_subplot(views[i, j]), sc, g, which, v, older, overlay=drawn, plans=pl,
                    horizons=drawn, title=f"{name}: {v} at {when}",
                )  # fmt: skip
        fig.suptitle(f"{self._headline(sc)}\nconfig: {st.label()}", fontsize=12)
        _show_photo(fig, quality)

        rows = {"stock": self.metrics(sc.sid, plans[0][0], plans[0][1])}
        if len(plans) > 1:
            rows["config"] = self.metrics(sc.sid, plans[1][0], plans[1][1])
            rows["change"] = {
                k: v - rows["stock"][k] for k, v in rows["config"].items() if isinstance(v, float)
            }
            rows["change"]["mode"] = f"{rows['stock']['mode']} -> {rows['config']['mode']}"
        if shifts:  # res[2:], after stock's and the config's
            ms = [self.metrics(sc.sid, plan(r), r["extras"]) for r in res[2:] if "error" not in r]
            d = {k: [abs(m[k] - v) for m in ms] for k, v in rows["stock"].items() if isinstance(v, float)}
            rows["noise"] = {k: max((x for x in xs if np.isfinite(x)), default=np.nan) for k, xs in d.items()}
            rows["noise"]["mode"] = ", ".join(sorted({m["mode"] for m in ms}))
        cols = [
            "mode",
            "ade",
            "ade_long",
            "lat",
            "lon",
            "path_err",
            "dpsi",
            "gt_dpsi",
            "dpsi_err",
            "turn_frac",
            "speed_rmse",
        ]
        cols += ["v0_ratio", "head_ratio", "plan_v0", "cam_h"]
        display(pd.DataFrame(rows).T[cols].infer_objects().round(3))
        if shifts:
            print(
                f"noise: the largest change that any of the {len(ms)} half-pixel shifts (T.SHIFTS) makes on this "
                "turn, and every class they give; a change within it is not the config's doing"
            )
        lam, mu = st.height_scale / st.zoom, st.height_scale * st.hstretch
        line = f"the config shows a road point (X, Y) about where stock shows (X x {lam:.3f}, Y x {mu:.3f})"
        if len(plans) > 1:
            inb = res[1].get("in_bounds") or {}
            line += "; share of each view inside the camera frame: " + ", ".join(
                f"{v} {f:.2f}" for v, f in inb.items()
            )
        print(line)

    # ---- part 4 ----

    def _crit(self, sid, r):
        m = self.metrics(sid, plan(r), r["extras"])
        return np.array([f(m) for _, _, f in CRITERIA], float), m

    def noise_floor(self, keys=None):
        """{criterion: (per turn, for the mean)}: what the eight half-pixel SHIFTS do to each criterion. Per turn,
        the median |change| on one turn; for the mean, the largest |change| of the mean over the turns."""
        keys = tuple(self.sid(k) for k in (keys or self.keys))
        if keys not in self._floor:
            res = self.fetch([(st, k) for st in (STOCK, *SHIFTS) for k in keys])
            n = len(keys)
            base = [self._crit(k, r)[0] for k, r in zip(keys, res[:n])]
            d = np.array(
                [
                    [self._crit(k, r)[0] - b for k, r, b in zip(keys, res[n * (i + 1) : n * (i + 2)], base)]
                    for i in range(len(SHIFTS))
                ]
            )  # (shift, turn, criterion)
            self._floor[keys] = {
                name: (
                    float(np.nanmedian(np.abs(d[:, :, i]))),
                    float(np.nanmax(np.abs(np.nanmean(d[:, :, i], axis=1)))),
                )
                for i, (name, _, _) in enumerate(CRITERIA)
            }
        return self._floor[keys]

    def compare(self, config, keys=None, noise=True):
        """Part 4: the config against stock on every turn, both run here. noise: judge each change against what
        the half-pixel SHIFTS do (noise_floor); without it any change at all counts."""
        st = as_setting(config)
        keys = [self.sid(k) for k in (keys or self.keys)]
        n = len(keys)
        res = self.fetch([(STOCK, k) for k in keys] + [(st, k) for k in keys])
        bad = [(self.scene(k).label, r["error"]) for k, r in zip(keys, res[n:]) if "error" in r]
        if bad:
            raise ValueError(f"{st.label()} cannot be shown: {bad}")
        crit = [[self._crit(k, r) for k, r in zip(keys, part)] for part in (res[:n], res[n:])]
        (cs, ms), (cc, mc) = ([np.array([c for c, _ in part]), [m for _, m in part]] for part in crit)

        cols = (
            "ade",
            "ade_long",
            "lat",
            "path_err",
            "dpsi_err",
            "turn_frac",
            "speed_rmse",
            "v0_ratio",
            "head_ratio",
            "cam_h",
        )
        per = {}
        for c in cols:
            a, b = np.array([m[c] for m in ms], float), np.array([m[c] for m in mc], float)
            per |= {(c, "stock"): a, (c, "config"): b, (c, "change"): b - a}
        per[("mode", "stock -> config")] = [f"{a['mode']} -> {b['mode']}" for a, b in zip(ms, mc)]
        per_df = pd.DataFrame(per, index=[self.scene(k).label for k in keys])
        per_df.columns = pd.MultiIndex.from_tuples(per_df.columns)
        display(Markdown(f"**Each turn, stock and config: {st.label()}** (both run here on {self.device})"))
        display(per_df.round(3))

        floor = self.noise_floor(keys) if noise else {name: (0.0, 0.0) for name, _, _ in CRITERIA}
        rows, groups = [], {}
        for i, (name, group, _) in enumerate(CRITERIA):
            d = cc[:, i] - cs[:, i]
            per_turn, for_mean = floor[name]
            better, worse = int((d < -per_turn).sum()), int((d > per_turn).sum())
            same = int(np.isfinite(d).sum()) - better - worse
            mean = float(np.nanmean(d))
            if mean < -for_mean and better > worse:
                verdict = "better"
            elif mean > for_mean and worse > better:
                verdict = "worse"
            else:
                verdict = "no clear change"
            groups.setdefault(group, []).append(verdict)
            rows.append(
                {
                    "criterion": name,
                    "judges": group,
                    "stock": float(np.nanmean(cs[:, i])),
                    "config": float(np.nanmean(cc[:, i])),
                    "change": mean,
                    "noise, mean": for_mean,
                    "noise, per turn": per_turn,
                    "better / same / worse": f"{better} / {same} / {worse}",
                    "verdict": verdict,
                }
            )
        planned = (sum(m["planned"] for m in ms), sum(m["planned"] for m in mc))
        verdict = (
            "better" if planned[1] > planned[0] else "worse" if planned[1] < planned[0] else "no clear change"
        )
        groups["alignment"].append(verdict)
        rows.append(
            {
                "criterion": "turns planned toward the route (count, higher is better)",
                "judges": "alignment",
                "stock": planned[0],
                "config": planned[1],
                "change": planned[1] - planned[0],
                "better / same / worse": "",
                "verdict": verdict,
            }
        )
        display(Markdown("**Means over the turns, and the verdict**"))
        display(pd.DataFrame(rows).set_index("criterion").round(3))
        lines = []
        for group, vs in groups.items():
            word = (
                "better"
                if "better" in vs and "worse" not in vs
                else "worse"
                if "worse" in vs and "better" not in vs
                else "mixed"
                if "better" in vs
                else "no clear change"
            )
            lines.append(f"- **{group}: {word}** ({', '.join(vs)})")
        how = "beyond what eight half-pixel shifts do" if noise else "any change counts (noise=False)"
        display(Markdown(f"**{st.label()}**, against stock on {n} turns ({how}):\n\n" + "\n".join(lines)))

        fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 13), squeeze=False)
        for j, k in enumerate(keys):
            sc = self.scene(k)
            plans = [
                (plan(res[j]), res[j]["extras"], STOCK_C, "stock"),
                (plan(res[n + j]), res[n + j]["extras"], CONFIG_C, "config"),
            ]
            self._bev(axes[0, j], sc, plans)
            self._speed(axes[1, j], sc, plans, legend=j == 0)
            self._heading(axes[2, j], sc, plans)
            axes[0, j].set_title(
                f"{sc.label}: {sc.side} in {sc.nav_m:g} m\nADE {ms[j]['ade']:.2f} -> {mc[j]['ade']:.2f} m",
                fontsize=10,
            )
            axes[1, j].set_title("speed (m/s); t = 0: ▶ pose, ◀ head, ◆ plan_v0", fontsize=9)
            axes[2, j].set_title("heading toward the route (deg)", fontsize=9)
        fig.suptitle(f"stock (orange) and the config (purple): {st.label()}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
        plt.show()
