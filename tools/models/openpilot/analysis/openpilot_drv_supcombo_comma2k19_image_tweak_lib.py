"""What `openpilot_drv_supcombo_comma2k19_image_tweak.ipynb` runs, so the notebook stays short.

The KITScenes image-tweak notebook's question, asked of five comma2k19 turns: does a change to the image,
made before openpilot sees it, plan them better? Everything it calls is a method of `Lab`:

- `stock_table`, `show_stock`, `show_warps` (part 1): the turns as stock plans them, the four images the model
  is handed, and the warp that cut them;
- `show_cameras` (part 1): the camera, and the row openpilot expects the horizon on in each view;
- `show_measurements` (part 2): how the camera height and the speeds are measured;
- `checks`: that the node is the stock adapter, and that what is drawn is what it fed;
- `tweak` (part 3): one config on one turn, drawn beside stock (or any other config), with what half-pixel
  shifts alone change;
- `compare` (part 4): one config on every turn, scored against stock (or any other config) and a noise floor;
- `arms` (part 5): several configs on every turn, side by side.

The camera is comma's own EON road camera, the one supercombo was trained on, so unlike KITScenes there is
no second camera to feed and comma's 65 deg cut is the whole frame: `fov_*` can still cut it narrower.
"Stock" is stock run here, with the extras the recorded run predates (`plan_v0`, the desire state); `checks`
shows it reproduces the recorded run. One thing the notebook says where it matters: the benchmark's ground
truth is the device's path, not the calibrated frame the plan is in, so the route's mount yaw, a degree or
two, is part of every score.

Paddock never imports a model, and neither does this module. The model runs in
`openpilot_drv_supcombo_image_node.py`, beside it: launched in openpilot's own interpreter
as `paddock run` launches the adapter, and driven over localhost HTTP. Here a config
becomes the 3x3 matrices and lookup tables the node applies, and the node's own `View`
rebuilds what the model was handed, so it can be drawn.

The frames are decoded once from each segment's video, as the harness decodes them, and kept as raw arrays
under the cache's frames/: the harness's own frame cache is scratch that a run retires.
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

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19.harness import DEFAULT_HISTORY_FRAMES, INTERVAL_S
from paddock.adapters.models.openpilot.policy import (
    DESIRE_MODES,
    DESIRE_TURN_LEFT,
    DESIRE_TURN_RIGHT,
    MODEL_HZ,
)
from paddock.adapters.models.openpilot.supercombo import rgb_to_yuv420
from paddock.benchmark import Sample
from paddock.config import REPO_ROOT, SiteConfig
from paddock.policy.protocol import Command, Trajectory, load_frame
from paddock.registry import Registry
from paddock.render import load_runs
from paddock.runner import load_benchmark
from paddock.runtime import env as env_builder
from paddock.runtime import ports, proc
from paddock.viz import palette

BENCH, MODEL = "comma2k19", "openpilot"
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
        "adapters/benchmarks/comma2k19/harness.py",
        "adapters/benchmarks/comma2k19/loader.py",
        "adapters/benchmarks/comma2k19/calibration.py",
    )
)
DT = INTERVAL_S  # the plan's and the GT's step, 0.5 s
N_LONG = 20  # the plan's waypoints: 10 s
H_M = 1.22  # the height of comma's camera over the road, the one the model was trained at (the note's Q4)
# `img` holds the newest frame and the one FRAME_SKIP model ticks before it: openpilot's
# MODEL_RUN_FREQ // MODEL_CONTEXT_FREQ = 20 // 5 (item 1 of supercombo.py's docstring). Restated
# because openpilot's ModelConstants cannot be imported in paddock's environment.
FRAME_SKIP = 4
FILL_Y = 126  # a blanked view's luma: mid-grey; its chroma is 128, no colour
TOL_M = 0.25  # checks(): the most a GPU and a CPU may disagree on one waypoint within H (metres)
ROAD_X = (10, 20, 30, 40, 60)  # the road grid's distances ahead of the camera (m)
ROAD_Y = (-6, -3, 0, 3, 6)  # and its lines along the road (m, left +)

GT_C = palette.GROUND_TRUTH
STOCK_C = palette.color(MODEL)  # stock, run here
CONFIG_C = "#7700bb"  # the config's plan
# arms(): one colour per arm, in this order, never cycled. With GT_C, every pair is >= 8 dE apart under
# protan and deutan simulation and >= 15 in normal vision (the dataviz skill's validator, all pairs);
# the two lightest sit under 3:1 against white, so every plot that uses them has a legend.
ARM_C = (STOCK_C, CONFIG_C, "#dd0099", "#00bb77", "#bb99dd")
GREY = "#8c8c8c"
VIEW_STYLE = {"img": ("#ffd92f", "-"), "big_img": ("#00e5ff", "--")}  # each view's footprint on the camera
OUTLINE = [patheffects.withStroke(linewidth=3.5, foreground="black")]  # a path stays visible on any road
# The four model inputs, in the order drawn: each view's older frame, then its keyframe.
PANELS = (("img", True), ("img", False), ("big_img", True), ("big_img", False))
# comma's EON road camera, the one supercombo's views were defined on and the one comma2k19 recorded with:
# 1164 x 874 at a 910 px focal length, so img (medmodel, 910 px) is one camera pixel per model pixel.
EON_W, EON_H, EON_F = L.EON_WIDTH, L.EON_HEIGHT_PX, L.EON_FOCAL
EON_FOV = float(np.degrees(2 * np.arctan(EON_W / 2 / EON_F)))  # 65.2 deg across
# checks(): a cut that fits the frame. The EON's own 65.2 deg is the whole frame, which a cut centred on the
# principal point overruns by half a pixel, so the check cuts narrower.
CHECK_CUT_DEG = 50.0
# The seconds of frames a sample holds before its keyframe, at the harness's defaults (160 frames at 20 Hz),
# which are the recorded run's; Lab refuses a site that sets others.
HISTORY_S = (DEFAULT_HISTORY_FRAMES - 1) / L.SRC_HZ
FILTERS = {"area": Image.BOX, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS, "nearest": Image.NEAREST}


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
    # the source: a cut of the camera's frame (see `cut`)
    fov_deg: float = 0.0
    fov_aspect: float = EON_W / EON_H
    fov_width: int = EON_W
    fov_told_deg: float = 0.0
    fov_filter: str = "area"
    # the route's turn, sent as openpilot's desire
    desire: str = "off"
    desire_lead_s: float = 0.0

    def __post_init__(self):
        # One type per field, so {"zoom": 1} and {"zoom": 1.0} are one config and one cache entry.
        for f in fields(self):
            if isinstance(f.default, float):
                object.__setattr__(self, f.name, float(getattr(self, f.name)))
        object.__setattr__(self, "supersample", int(self.supersample))
        object.__setattr__(self, "fov_width", int(self.fov_width))
        for name, allowed in (
            ("views", ("both", *VIEWS)),
            ("ss_views", ("both", *VIEWS)),
            ("blank", ("", *VIEWS)),
            ("fov_filter", tuple(FILTERS)),
            ("desire", tuple(DESIRE_MODES)),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} = {getattr(self, name)!r}: it must be one of {allowed}")
        if min(self.height_scale, self.zoom, self.hstretch) <= 0 or self.supersample < 1:
            raise ValueError("height_scale, zoom and hstretch must be positive, and supersample 1 or more")
        if not (0 <= self.fov_deg < 170 and 0 <= self.fov_told_deg < 170):
            raise ValueError("fov_deg and fov_told_deg are degrees across, 0 (as recorded) or under 170")
        if self.fov_aspect <= 0 or self.fov_width < 0 or 0 < self.fov_width < 64:
            raise ValueError("fov_aspect must be positive, and fov_width 0 (no resample) or 64 px or more")
        if not 0 <= self.desire_lead_s <= HISTORY_S:
            # The sample holds 7.95 s of frames before the keyframe; the desire can start on none earlier.
            raise ValueError(f"desire_lead_s must be 0 to {HISTORY_S:g} s")

    def changed(self):
        """{knob: value} of the knobs that change anything; empty for stock."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != f.default}
        if not any(k in GEOMETRY for k in out):
            out.pop("views", None)
        if self.supersample == 1:
            out.pop("ss_views", None)
        if not self.fov_deg:  # the frame as recorded: the cut's own knobs do nothing
            for k in ("fov_aspect", "fov_width", "fov_told_deg", "fov_filter"):
                out.pop(k, None)
        elif not self.fov_width:  # a crop, not resampled
            out.pop("fov_filter", None)
        if self.desire == "off":
            out.pop("desire_lead_s", None)
        return out

    def source(self):
        """What the model's frames are made from: the camera's frame as recorded, or the cut's knobs. Two
        configs with one source share their frames and their `/geometry`."""
        if not self.fov_deg:
            return ("front",)
        filt = self.fov_filter if self.fov_width else ""
        return ("front", self.fov_deg, self.fov_aspect, self.fov_width, self.fov_told_deg, filt)

    def geometric(self):
        return any(k in GEOMETRY for k in self.changed())

    def label(self):
        c = self.changed()
        return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()) or "stock"


STOCK = Setting()
STOCK_SOURCE = STOCK.source()
# Eight shifts of half a model pixel: no geometry to speak of (0.03 deg in img), but every model pixel comes
# from a different camera pixel. What they do to a score is what any change at all does: compare()'s noise floor.
SHIFTS = tuple(
    Setting(dx_px=x, dy_px=y)
    for x, y in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))
)
# checks(): one config down every path the node's `View` has (shift, supersampling, filter, tables, fill).
EVERY_PATH = Setting(dx_px=0.5, supersample=3, sharpen=0.5, gamma=0.8, saturation=1.3, blank="img")


def shifted(base, shift):
    """`base` with a half-pixel SHIFTS entry added to its own grid shift: the noise around any config. On
    STOCK it is the shift itself, so stock's noise runs are shared."""
    return replace(base, dx_px=base.dx_px + shift.dx_px, dy_px=base.dy_px + shift.dy_px)


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
    """The /run body for one config on one sample. Every command is STRAIGHT, as the recorded run sent, unless
    the config sends a desire: then the frames within desire_lead_s of the keyframe carry the route's side
    (LEFT or RIGHT), and the node sends it as `DESIRE_MODES[desire]`. openpilot pulses a desire on its rising
    edge, so it reaches the model once, on the first of those frames, and its queue carries it for 5 s."""
    wire = [dict(o) for o in sc.wire]
    turn = {"left": Command.LEFT, "right": Command.RIGHT}.get(sc.side)
    send = st.desire != "off" and turn is not None
    t_k = wire[-1]["timestamp"]
    for o in wire:
        on = send and t_k - o["timestamp"] <= st.desire_lead_s * 1e6
        o["command"] = int(turn if on else Command.STRAIGHT)
    body = {"observations": wire}
    if send:
        body["desire"] = st.desire
    specs = view_specs(st, sc, g)
    if specs:
        body["views"] = specs
    return body


def cut(st, K, size):
    """The cut the fov knobs make of a camera frame (size = w, h; intrinsics K), or None for the frame as
    recorded. Centred on the principal point, fov_deg across, at fov_aspect.

    fov_width w > 0 resamples the cut to w px across with `fov_filter` (PIL's `resize` over the float box: an
    output pixel's centre i + 0.5 is the box's x0 + (i + 0.5) * s, s the cut's px per output px). Its true
    intrinsics are then f / s and (c + 0.5 - x0) / s - 0.5. fov_width 0 crops whole pixels, resampling
    nothing: f as recorded, c - x0.

    The model is told the cut's true intrinsics, unless fov_told_deg says what field of view to tell it: then
    its focal length is that view's, (w / 2) / tan(told / 2), and only the warp openpilot computes changes.
    Raises ValueError if the cut does not fit in the frame."""
    if not st.fov_deg:
        return None
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    wc = 2.0 * f * np.tan(np.radians(st.fov_deg) / 2)  # the cut's width in camera pixels
    if st.fov_width:
        w = 2 * round(st.fov_width / 2)  # even: openpilot's YUV 4:2:0 halves both sides
        h = 2 * round(w / st.fov_aspect / 2)
        s = wc / w
        x0, y0 = cx + 0.5 - wc / 2, cy + 0.5 - h * s / 2
        box = (x0, y0, x0 + wc, y0 + h * s)
        Kc = np.array(
            [[f / s, 0.0, (cx + 0.5 - x0) / s - 0.5], [0.0, f / s, (cy + 0.5 - y0) / s - 0.5], [0, 0, 1]]
        )
        filt = st.fov_filter
    else:
        w = 2 * round(wc / 2)
        h = 2 * round(w / st.fov_aspect / 2)
        s, x0, y0 = 1.0, round(cx - w / 2), round(cy - h / 2)
        box = (x0, y0, x0 + w, y0 + h)
        Kc = np.array([[f, 0.0, cx - x0], [0.0, f, cy - y0], [0.0, 0.0, 1.0]])
        filt = None
    if box[0] < 0 or box[1] < 0 or box[2] > size[0] or box[3] > size[1]:
        raise ValueError(
            f"a {st.fov_deg:g} deg cut at aspect {st.fov_aspect:.3f} is {box[2] - box[0]:.0f} x "
            f"{box[3] - box[1]:.0f} px; it does not fit in the {size[0]} x {size[1]} frame about its centre"
        )
    told = Kc.copy()
    if st.fov_told_deg:
        told[0, 0] = told[1, 1] = (w / 2) / np.tan(np.radians(st.fov_told_deg) / 2)
    return SimpleNamespace(box=box, size=(w, h), scale=s, K=Kc, K_told=told, filter=filt)


def _stage(src, dst, c):
    """One frame, cut as `c` says, saved as the raw (h, w, 3) uint8 array `load_frame` reads: written aside,
    then renamed, so a frame half-written by an interrupted run is never read."""
    if dst.exists():
        return
    img = Image.fromarray(np.load(src)) if str(src).endswith(".npy") else Image.open(src).convert("RGB")
    if c.filter is None:
        arr = np.asarray(img.crop(tuple(int(v) for v in c.box)))
    else:
        arr = np.asarray(img.resize(c.size, FILTERS[c.filter], box=c.box))
    tmp = dst.with_name(f"{dst.stem}.{uuid.uuid4().hex[:8]}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, dst)


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


def road_map(st, sc, g, base=STOCK, sc0=None, g0=None):
    """F: a road point (camera-centred, metres) -> the point `base` shows at the model pixel where this config
    shows it. So the model sees the road at F(road). Exact on the road plane for each camera at its real
    height; img's, unless the configs touch big_img alone. sc0, g0: base's scene and geometry, when its source
    is not the config's."""
    sc0, g0 = sc0 or sc, g0 or g
    if sc0 is sc and not st.geometric() and not base.geometric():
        return np.eye(3)
    which = {s.views for s in (st, base) if s.geometric()}
    view = which.pop() if len(which) == 1 and "both" not in which else "img"
    G, G0 = _road_to_px(sc), _road_to_px(sc0)
    F = np.linalg.inv(G0) @ warp(base, sc0, view, g0) @ np.linalg.inv(warp(st, sc, view, g)) @ G
    return F / (F @ [10.0, 0.0, 1.0])[2]


def real_camera(sc, g):
    """camera_from_calib with the camera's true intrinsics: the same as /geometry's unless the config tells the
    model a field of view the frame does not have (fov_told_deg), when the model's warp is cut with the told
    one and the camera still sees through the true one."""
    C = np.array(g["camera_from_calib"])
    return C if sc.K_told is sc.K else sc.K @ np.linalg.inv(sc.K_told) @ C


def expected_row(g, view):
    """The row of `view`'s horizon in the model's own pixels, as openpilot defines the view: 47.6 of img's 256
    rows, 151.8 of big_img's. Where a camera at openpilot's calibration puts the horizon."""
    p = np.linalg.inv(np.array(g["calib_from_model"][view])) @ [1.0, 0.0, 0.0]
    return float(p[1] / p[2])


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
    """A camera frame at half size (JPEG draft mode; a staged cut, every other pixel), and that scale."""
    if str(path).endswith(".npy"):
        return np.load(path)[::2, ::2], 0.5
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

_SID = re.compile(r'\{"sample_id": "([^"]+)"')  # a samples.jsonl line's id, without parsing the line


def _parse(sid):
    """comma2k19/<chunk>/<dongle>|<route>/<segment>/<keyframe> -> (chunk, route_id, segment, keyframe)."""
    _, chunk, route_id, segment, kf = sid.split("/")
    return chunk, route_id, int(segment), int(kf)


def _label(sid):
    """A sample id -> '<MM-DD--hh-mm-ss>/<segment>/<keyframe>': the route's start, without its year and car."""
    _, route_id, segment, kf = _parse(sid)
    return f"{route_id.split('|')[-1][5:]}/{segment}/{kf}"


def _matches(sid, key):
    return (
        sid == key
        or _label(sid).startswith(key)
        or any(sid.split("/", k)[-1].startswith(key) for k in (1, 2))
    )


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

    scenes: sample ids, labels like "07-11--15-13-12/14/789", or unique prefixes. run_id: the recorded stock
    run, which `checks` holds stock here to, and whose sample ids are the ones a key is searched among. device, nodes, threads: where the model runs. height: the camera's
    real height, for drawing and for the knobs that move the camera over the road: "harness" (comma's mount,
    the 1.22 m `camera2ego_from_rpy` puts it at) or metres. overrides: tag parameters for this lab only.
    recompute: ignore the results cached on disk.
    """

    def __init__(
        self,
        scenes,
        run_id,
        *,
        device="cpu",
        nodes=5,
        threads=8,
        height="harness",
        overrides=None,
        recompute=False,
        base_port=9620,
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
        self.grid = self.harness.grid
        self.run_id, self.height, self.device, self.recompute = run_id, height, device, recompute
        run_dir = self.cfg.runs / BENCH / MODEL / run_id
        if not (run_dir / "samples.jsonl").exists():
            raise FileNotFoundError(f"no per-sample records of {run_id} on this machine")
        # A window of another length or rate would be another measurement than the recorded one.
        rc = json.loads((run_dir / "run_config.json").read_text())
        mine = {"history_frames": self.grid.history_frames, "input_hz": self.grid.input_hz}
        theirs = {k: rc[k] for k in mine}
        if theirs != mine:
            raise ValueError(f"this site's comma2k19 window is {mine}; {run_id}'s was {theirs}")
        # Each record kept as its line and parsed only when drawn: all 10,165 parsed would hold most of a
        # gigabyte for five turns.
        with (run_dir / "samples.jsonl").open() as fh:
            self._lines = {_SID.match(line).group(1): line for line in fh if line.strip()}
        self.keys = []
        self.keys = [self.sid(k) for k in scenes]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("scenes names a keyframe twice")
        self.cache = self.cfg.runs / BENCH / "_analysis" / MODEL / "image_tweak"
        for sub in ("results", "geometry", "logs", "frames"):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)
        sha = reg.repos["openpilot"].sha
        self.code = hashlib.sha1(b"".join(p.read_bytes() for p in CODE_FILES) + sha.encode()).hexdigest()[:10]
        self.pool = Pool(nodes, device, threads, self.cfg, self.cache / "logs", base_port)
        atexit.register(self.pool.close)
        _OPEN.append(self)
        self._sc, self._geo, self._planes, self._res, self._rec, self._floor = {}, {}, {}, {}, {}, {}
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
        """The sample id a key names: a sample id, a label like "07-11--15-13-12/14/789", or a unique prefix of
        either. The lab's turns are searched first, then every sample of the recorded run."""
        key = str(key)
        if key in self._lines:
            return key
        for pool in (self.keys, list(self._lines)):
            hit = [s for s in pool if _matches(s, key)]
            if len(hit) == 1:
                return hit[0]
            if len(hit) > 1:
                raise KeyError(f"{key!r} names {len(hit)} samples, e.g. {[_label(s) for s in hit[:4]]}")
        raise KeyError(f"{key!r} names no sample of {self.run_id}")

    def _record(self, sid):
        if sid not in self._rec:
            self._rec[sid] = json.loads(self._lines[sid])
        return self._rec[sid]

    def _segment(self, sid):
        chunk, route_id, segment, _ = _parse(sid)
        hit = [
            s
            for s in self.harness._segments
            if (str(s.chunk), s.route_id, int(s.segment)) == (chunk, route_id, segment)
        ]
        if len(hit) != 1:
            raise FileNotFoundError(f"{_label(sid)}: its segment is not in this site's comma2k19 data")
        return hit[0]

    def _real_height(self, E):
        """(metres, how): the camera's height over the road, per `height`."""
        if self.height == "harness":
            return float(E[2, 3]), "the harness's, comma's mount"
        return float(self.height), "set"

    def scene(self, key, config=None):
        """Everything about one keyframe that no image knob changes, read once per source: the camera's frames
        as recorded by default, or the cut a config feeds the model (its `fov_*` knobs)."""
        if isinstance(key, SimpleNamespace):
            return key
        sid = self.sid(key)
        src = STOCK_SOURCE if config is None else as_setting(config).source()
        if (sid, src) not in self._sc:
            self._sc[(sid, src)] = (
                self._front_scene(sid) if src == STOCK_SOURCE else self._source(sid, config)
            )
        return self._sc[(sid, src)]

    def _frames(self, seg, idx):
        """The window's frames, decoded from the segment's video as the harness decodes them and kept as raw
        arrays under frames/, each written aside and renamed (the harness's `_write_frame`). The video cannot be
        seeked, so a segment's first call decodes it from its start to the keyframe."""
        d = self.cache / "frames" / f"{seg.chunk}_{seg.route_id.replace('|', '_')}_{seg.segment}"
        d.mkdir(exist_ok=True)
        out = {i: d / f"{i:06d}.npy" for i in idx}
        need = {i for i, p in out.items() if not p.exists()}
        if need:
            t0 = time.time()
            for i, rgb in enumerate(L.decode_frames(seg.video, limit=max(need) + 1)):
                if i in need:
                    self.harness._write_frame(out[i], rgb)
            short = sorted(i for i in need if not out[i].exists())
            if short:
                raise FileNotFoundError(f"{seg.sample_key}: the video ends before frame {short[0]}")
            print(f"decoded {len(need)} frame(s) of {seg.sample_key} in {time.time() - t0:.0f} s")
        return [str(out[i]) for i in idx]

    def _front_scene(self, sid):
        """The keyframe as the harness builds it: the window's frames, the route's calibration, CAN and radar,
        through the harness's own `_observation` and `_reference`. Raises if the calibration is not the one
        the recorded run was given."""
        rec, seg, kf = self._record(sid), self._segment(sid), _parse(sid)[3]
        pose = L.load_pose(seg.path)
        cal = self.harness._calibrations.get(seg)
        rpy = [round(v, 4) for v in cal.rpy_deg]
        if not cal.is_valid or rpy != rec["native_id"]["rpy_calib_deg"]:
            raise AssertionError(
                f"{_label(sid)}: the calibration is {rpy} deg (valid: {cal.is_valid}), "
                f"{rec['native_id']['rpy_calib_deg']} in {self.run_id}"
            )
        E = L.camera2ego_from_rpy(cal.rpy)
        can, radar = L.load_can_speed(seg.path), L.load_radar(seg.path)
        idx = self.grid.window(kf)
        if idx[0] != rec["native_id"]["first_frame_index"]:
            raise AssertionError(
                f"{_label(sid)}: the window starts at frame {idx[0]}, not the recorded run's"
            )
        frames = self._frames(seg, idx)
        obs = [self.harness._observation(Path(f), pose, can, i, idx[0], E) for f, i in zip(frames, idx)]
        ref = self.harness._reference(pose, can, radar, kf)
        sample = Sample(
            sample_id=sid,
            observations=obs,
            ground_truth=L.future_in_ego_frame(pose, kf, self.grid.gt_horizons_s),
            split=seg.vehicle,
            native_id=rec["native_id"] | {"reference": ref},
        )
        gt = np.asarray(sample.ground_truth, float)[:N_LONG, :2]
        gt = gt[: int(np.isfinite(gt).all(1).cumprod().sum())]
        # The route's side is the GT's: comma2k19 records no navigation, and the run sent every frame STRAIGHT.
        turn = _dpsi(gt, len(gt), self.p["min_step_m"])
        side = "left" if turn > 0 else "right" if turn < 0 else "straight"
        stamps = [int(o.timestamp_us) for o in obs]
        reps = _repeats(stamps)
        older = _older(reps)
        h, how = self._real_height(E)
        Eh = E.copy()
        Eh[2, 3] = h
        E0 = Eh.copy()
        E0[:2, 3] = 0.0  # the camera over the origin: openpilot plans from the camera, not the car's origin
        K = np.asarray(L.EON_INTRINSICS, float)[:3, :3]
        return SimpleNamespace(
            sid=sid,
            src=STOCK_SOURCE,
            source="the EON road camera, as recorded",
            cut=None,
            idx=idx,  # each observation's frame index
            frames=frames,
            size=(EON_W, EON_H),
            label=_label(sid),
            segment=seg,
            kf=kf,
            side=side,
            sign={"left": 1.0, "right": -1.0}.get(side, 0.0),  # heading toward the route is + for both sides
            turn_deg=turn,
            route=f"{side}, the GT turns {turn:+.0f} deg over {len(gt) * DT:g} s",
            v=float(ref["speed"]),  # the harness's reference: CAN, the car's own odometer
            v_how=ref["speed_source"],
            v_pose=float(pose.speed[kf]),
            pose=pose,
            rpy_deg=cal.rpy_deg,
            cal_scope=cal.scope,
            sample=sample,
            wire=[o.to_wire() for o in obs],
            frame=frames[-1],
            older=older,  # the observation img's older frame comes from
            older_frame=frames[older],
            older_kf=idx[older],
            pair_dt=(stamps[-1] - stamps[older]) / 1e6,  # seconds between img's two frames
            ticks=sum(reps),  # model ticks over the sequence, as the adapter counts them
            gt=np.vstack([gt, np.full((N_LONG - len(gt), 2), np.nan)]),
            gt_n=len(gt),
            K=K,
            K_told=K,  # the intrinsics the model is told: the same object unless a config tells it others
            E=Eh,
            E_harness=E,
            E0=E0,
            h=h,
            h_how=how,
            h_harness=float(E[2, 3]),
        )

    def _source(self, sid, config):
        """The keyframe as a config's cut feeds it: the scene with each frame cut, and the cut's intrinsics (or
        the told ones) in the observations. A cut is staged under the cache's frames/ once, as raw arrays."""
        st, base = as_setting(config), self.scene(sid)
        c = cut(st, base.K, base.size)
        frames = self._stage_cut(base, c)
        K_told = c.K if not st.fov_told_deg else c.K_told
        how = f"resampled to {c.size[0]} x {c.size[1]} ({c.filter})" if c.filter else "cropped, not resampled"
        what = f"the EON road camera, cut to {st.fov_deg:g} deg across and {how}"
        if st.fov_told_deg:
            what += f"; the model is told {st.fov_told_deg:g} deg"
        wire = []
        for o, frame in zip(base.wire, frames):
            o = json.loads(json.dumps(o))  # a deep copy: the recorded scene's wire is shared
            o["image_paths"]["CAM_FRONT"] = str(frame)
            o["calibration"]["camera2image"]["CAM_FRONT"] = K_told.tolist()
            wire.append(o)
        return SimpleNamespace(
            **(
                vars(base)
                | dict(
                    src=st.source(),
                    source=what,
                    cut=c,
                    frames=frames,
                    size=c.size,
                    wire=wire,
                    frame=frames[-1],
                    older_frame=frames[base.older],
                    K=c.K,
                    K_told=K_told,
                )
            )
        )

    def _stage_cut(self, base, c):
        """The cut of each frame, as .npy under frames/, keyed by the segment and the cut itself (its box, size
        and filter), so a change to how a cut is computed never reuses an old one's frames."""
        key = f"{base.segment.sample_key} {np.round(c.box, 6).tolist()} {c.size} {c.filter}"
        d = self.cache / "frames" / ("cut_" + hashlib.sha1(key.encode()).hexdigest()[:16])
        d.mkdir(exist_ok=True)
        out = [d / f"{i:06d}.npy" for i in base.idx]
        todo = {o: f for o, f in zip(out, base.frames) if not o.exists()}
        if todo:
            t0 = time.time()
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda kv: _stage(kv[1], kv[0], c), todo.items()))
            print(f"staged {len(todo)} frame(s) of {base.label}'s cut in {time.time() - t0:.0f} s")
        return [str(o) for o in out]

    def recorded(self, key):
        """(plan, extras) as the recorded run planned this turn."""
        s = self._record(self.sid(key))
        return np.asarray(s["prediction"], float)[:, :2], s.get("extras") or {}

    def stock(self, key):
        """(plan, extras) as stock plans this turn here: the baseline every part compares with."""
        r = self.fetch([(STOCK, key)])[0]
        return plan(r), r["extras"]

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
        # The model's own desire_state: the probability it gives the route's turn, sent or not.
        state = (ex.get("desire") or {}).get("state")
        slot = {"left": DESIRE_TURN_LEFT, "right": DESIRE_TURN_RIGHT}.get(sc.side)
        out["p_turn"] = float(state[slot]) if state and slot else np.nan
        return out

    # ---- the model ----

    def _geo_path(self, sc):
        # The ring's own key is the one it had before there were sources, so its cache still applies.
        what = f"{sc.sid} {self.code}" if sc.src == STOCK_SOURCE else f"{sc.sid} {list(sc.src)} {self.code}"
        return self.cache / "geometry" / (hashlib.sha1(what.encode()).hexdigest()[:16] + ".json")

    def geometry(self, key, config=None):
        """The camera as openpilot sees it for one sample and source (`/geometry`), cached on disk. Also decodes
        the source's frames in the sample's node. key: a key, or a scene."""
        sc = self.scene(key, config)
        k = (sc.sid, sc.src)
        if k not in self._geo:
            path = self._geo_path(sc)
            if path.exists():
                self._geo[k] = json.loads(path.read_text())
            else:
                g = self.pool.node(sc.sid).post("/geometry", {"observations": sc.wire})
                path.write_text(json.dumps(g))
                self._geo[k] = g
        return self._geo[k]

    def _prefetch(self, scenes):
        """`/geometry` for each scene not yet on disk, in parallel across nodes."""
        need = {(s.sid, s.src): s for s in scenes}
        need = [s for k, s in need.items() if k not in self._geo and not self._geo_path(s).exists()]
        if need:
            self.pool.start({self.pool.assign(s.sid) for s in need})
            for f in [self.pool.node(s.sid).ex.submit(self.geometry, s) for s in need]:
                f.result()

    def planes(self, key, older=False):
        """The keyframe's (Y, U, V) planes, or (older) those of the frame `img` pairs it with, converted as
        the adapter converts them. key: a key (the recorded camera), or a scene."""
        sc = self.scene(key)
        k = (sc.sid, sc.src, older)
        if k not in self._planes:
            self._planes[k] = rgb_to_yuv420(load_frame(sc.older_frame if older else sc.frame))
        return self._planes[k]

    def model_input(self, config, key, view, older=False):
        """(6, 128, 256) uint8: what the model is handed for this view under a config, on the keyframe or
        (older) the frame `img` pairs it with. Built with the node's own `View`, so it can be drawn without
        running the model; every fresh result checks it against what the node fed."""
        st = as_setting(config)
        sc = self.scene(key, st)
        g = self.geometry(sc)
        spec = view_specs(st, sc, g).get(view, {})
        v = NODE.View.from_warp(
            warp(st, sc, view, g).astype(np.float32),
            spec,
            tuple(g["model_size"][view]),
            tuple(g["source_size"]),
        )
        return v.apply(*self.planes(sc, older))

    def _job(self, sc, body):
        """A result's cache key: the exact matrices and tables sent, the sample, its source, the desire and the
        commands it was sent on, the device and the code. A key adds nothing for the recorded camera and no
        desire, so results cached before those knobs existed still apply."""
        blob = {"sample": sc.sid, "views": body.get("views"), "device": self.device, "code": self.code}
        if sc.src != STOCK_SOURCE:
            blob["source"] = list(sc.src)
        if "desire" in body:
            blob["desire"] = [body["desire"], [o["command"] for o in body["observations"]]]
        return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16]

    def fetch(self, jobs, inputs=False):
        """[result] for [(config, key)]: from this session, the disk cache, or the nodes, in parallel.

        A result is /run's reply (trajectory, extras, warps, in_bounds, ticks, ms), or {"error": ...} for a
        config that cannot be shown. inputs=True runs afresh and keeps the model's keyframe tensors."""
        jobs = [(as_setting(st), self.sid(k)) for st, k in jobs]
        out, where, todo, scenes = [None] * len(jobs), {}, {}, {}
        for i, (st, sid) in enumerate(jobs):
            try:
                scenes[i] = self.scene(sid, st)
            except ValueError as e:  # a cut that does not fit the frame
                out[i] = {"error": str(e)}
        self._prefetch(scenes.values())
        for i, sc in scenes.items():
            st, sid = jobs[i]
            try:
                body = request(st, sc, self.geometry(sc))
            except ValueError as e:
                out[i] = {"error": str(e)}
                continue
            h = self._job(sc, body)
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
                w, hh = self.geometry(sid, st)["model_size"][v]
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
        out = f"{sc.label}: {sc.route}, at {sc.v:.1f} m/s; camera {sc.h:.2f} m over the road ({sc.h_how})"
        return out if sc.src == STOCK_SOURCE else f"{out}\nfed {sc.source}"

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
        hz = _rays_to(real_camera(sc, g)) * s
        ax.plot(hz[:, 0], hz[:, 1], ":", color="white", lw=1.2)
        ax.set_xlim(0, photo.shape[1])
        ax.set_ylim(photo.shape[0], 0)
        ax.set_axis_off()
        return s

    def _view(self, ax, sc, g, st, view, older, overlay=False, plans=(), horizons=False, title=None):
        """One model input as the model was handed it, YUV drawn as RGB at its full 512 x 256. overlay: the GT
        and each plan on it, through the real camera at its real height, then the view's own warp; the title
        counts the waypoints in view, since a sharp turn leaves img's 31 deg within a few metres. horizons: the
        row openpilot expects the horizon on (`expected_row`, solid, labelled) and the camera's real horizon
        through this config's warp (dashed)."""
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
            real = apply_h(Wi, _rays_to(real_camera(sc, g)))
            ax.plot(real[:, 0], real[:, 1], "--", color="white", lw=1)
            row = expected_row(g, view)
            stroke = [patheffects.withStroke(linewidth=2.5, foreground="black")]
            ax.axhline(row, color="white", lw=1.4, path_effects=stroke)
            ax.text(
                4, row - 2, f"expected horizon, row {row:.1f} of {h}", color="white", fontsize=8, va="bottom",
                path_effects=stroke, clip_on=True,
            )  # fmt: skip
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

    def _road_map(self, ax, sc, g, st, base=STOCK, sc0=None, g0=None, name0="stock"):
        """Where the config moves the road in the model's eyes, relative to the baseline (`road_map`): a grid of
        5 m squares to 60 m, grey as it is and red as the model sees it."""
        F = road_map(st, sc, g, base, sc0, g0)
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
        ax.set_title(f"the road as {name0} shows it (grey), as the config does (red)", fontsize=10)

    # ---- part 1 ----

    def stock_table(self, keys=None):
        """Part 1: each turn as stock plans it here, with metrics()."""
        keys = [self.sid(k) for k in (keys or self.keys)]
        rows = []
        for sid, r in zip(keys, self.fetch([(STOCK, k) for k in keys])):
            sc = self.scene(sid)
            rows.append(
                {"key": sc.label, "route": f"{sc.side}, {sc.turn_deg:+.0f} deg", "v": sc.v}
                | self.metrics(sid, plan(r), r["extras"])
            )
        cols = ["route", "v", "mode", "ade", "ade_long", "lat", "path_err", "dpsi", "gt_dpsi", "turn_frac"]
        cols += ["speed_rmse"]
        cols += ["v0_ratio", "head_ratio", "plan_v0", "cam_h"]
        return pd.DataFrame(rows).set_index("key")[cols]

    def show_stock(self, key, overlay=("img",), quality=80):
        """Part 1: one turn as stock plans it here. overlay: where the GT and the plan are drawn: "img" and
        "big_img", on that view's keyframe; "camera", on the keyframe as the camera took it."""
        sc = self.scene(key)
        g = self.geometry(sc.sid)
        xy, ex = self.stock(sc.sid)
        plans = [(xy, ex, STOCK_C, "stock")]
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
            self._headline(sc) + "\n" + self._summary("stock", self.metrics(sc.sid, xy, ex)), fontsize=12
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

    def show_cameras(self, key=None, quality=80):
        """Part 1: the camera the model is fed, on one turn, the first by default: the keyframe with its views'
        footprints, then img and big_img at the keyframe with the row openpilot expects the horizon on, then a
        table. comma2k19 recorded with comma's own EON camera, so there is no second camera to feed, and the
        EON's 65.2 deg cut is the whole frame. Runs no model: it reads the camera's /geometry."""
        sc = self.scene(key or self.keys[0])
        g = self.geometry(sc)
        fig = plt.figure(figsize=(22, 6.6))
        gs = fig.add_gridspec(
            1, 3, width_ratios=[1.45, 1, 1], wspace=0.08, top=0.86, bottom=0.02, left=0.02, right=0.98
        )  # fmt: skip
        ax = fig.add_subplot(gs[0, 0])
        self._camera(ax, sc, g, [(STOCK, None)])
        ax.set_title(
            "the EON road camera: the keyframe, img (yellow) and big_img (cyan, dashed) as cut from it;\n"
            "dotted, the camera's horizon",
            fontsize=10,
        )
        for j, v in enumerate(VIEWS):
            self._view(
                fig.add_subplot(gs[0, 1 + j]), sc, g, STOCK, v, False, horizons=True, title=f"{v} at t"
            )
        w, h = sc.size
        row = {
            "camera": "the EON road camera",
            "frame (px)": f"{w} x {h}",
            "focal length (px)": sc.K[0, 0],
            "field of view (deg)": f"{np.degrees(2 * np.arctan(w / 2 / sc.K[0, 0])):.1f} x "
            f"{np.degrees(2 * np.arctan(h / 2 / sc.K[1, 1])):.1f}",
            "height over the road (m)": sc.h,
            "roll, pitch, yaw (deg)": ", ".join(f"{np.degrees(a):+.2f}" for a in g["rpy"]),
            "calibration": sc.cal_scope,
        }
        for v in VIEWS:
            (mw, mh), W = g["model_size"][v], warp(STOCK, sc, v, g)
            pc = [apply_h(W, [[mw / 2 + dx, mh / 2]])[0] for dx in (0, 1)]
            row[f"{v}: camera px per model px"] = float(np.linalg.norm(pc[1] - pc[0]))
        fig.suptitle(
            f"{sc.label}: the EON road camera, as openpilot cuts its views from it. White, solid: the row "
            "openpilot expects the horizon on; dashed: the camera's own",
            fontsize=12,
        )
        _show_photo(fig, quality)
        display(pd.DataFrame([row]).set_index("camera").round(3))

    # ---- part 2 ----

    def show_measurements(self, key=None, quality=80):
        """Part 2: on one turn, each camera height and each speed, with how it is measured; then the frames the
        model reads its motion from, and the road's distances as the model reads them. From stock, run here."""
        sc = self.scene(key or self.keys[0])
        g = self.geometry(sc.sid)
        xy, ex = self.stock(sc.sid)
        h_b = float((ex.get("road") or {}).get("camera_height", np.nan))
        head, v0 = _head(ex), float(ex.get("plan_v0", np.nan))
        n_pair = sc.kf - sc.older_kf
        pos = sc.pose.positions  # ECEF metres
        v_pair = float(np.linalg.norm(pos[sc.kf] - pos[sc.older_kf]) / sc.pair_dt)
        v_gt, v_plan = _speeds(sc.gt[:1])[0], _speeds(xy[:1])[0]

        def table(title, head_row, rows):
            lines = [f"**{title}**", "", f"| {' | '.join(head_row)} |", "|" + "---|" * len(head_row)]
            return "\n".join(lines + [f"| {' | '.join(r)} |" for r in rows])

        roll, pitch, yaw = sc.rpy_deg
        heights = [
            ("the harness's", f"{sc.h_harness:.3f}", f"`camera2ego_from_rpy`'s z: comma's mount, `EON_HEIGHT` = {L.EON_HEIGHT} m, for every car. Nothing in comma2k19 measures it. openpilot never reads it: its warp is a rotation, and has no height in it."),
            ("**the model's** (`cam_h`)", f"**{h_b:.3f}**", "`extras.road.camera_height`: the z of supercombo's `road_transform` head, where it puts the road plane below the camera, on the keyframe's tick. Nothing the model is given holds a height: it infers this from the images."),
            ("openpilot's own", f"{H_M:.2f}", "comma's camera over the road, the height supercombo was trained at."),
        ]  # fmt: skip
        speeds = [
            ("▶ **the reference speed**", f"**{sc.v:.2f}**", f"the harness's `reference.speed`, the one it scores against: {'CAN, the car' + chr(39) + 's own odometer, at the keyframe' if sc.v_how == 'can' else 'the pose speed; this segment recorded no CAN'}. The right speed."),
            ("the pose speed", f"{sc.v_pose:.2f}", f"`SegmentPose.speed` at the keyframe k = {sc.kf}: the recorded device pose's velocity."),
            ("the pose speed over `img`'s pair", f"{v_pair:.2f}", f"\\|p(k) − p(k−{n_pair})\\| / {sc.pair_dt:.1f} s, between exactly the two frames `img` holds, {sc.older_kf} and {sc.kf}."),
            ("the GT's first step", f"{v_gt:.2f}", "\\|GT(0.5 s)\\| / 0.5 s: where the poses put the car half a second on."),
            ("◀ **the pose head**", f"**{head:.2f}**", "`extras.pose.velocity[0]`: the forward speed from supercombo's `pose` head, on the keyframe's tick. Nothing the model is given holds a speed: it reads its motion from the images (`img` and `big_img` each hold two frames 0.2 s apart) and from its own features of earlier ticks."),
            ("◆ `plan_v0`", f"{v0:.2f}", "`extras.plan_v0`: the plan's own velocity at t = 0, a separate output from the pose head. It is the one openpilot's longitudinal control reads."),
            ("the plan's first step", f"{v_plan:.2f}", "\\|plan(0.5 s)\\| / 0.5 s: what the score sees."),
        ]  # fmt: skip
        ratios = [
            ("`head_ratio` = ◀ / ▶", f"**{head / sc.v:.3f}**", "1 if the pose head reads the car's speed."),
            ("`v0_ratio` = the plan's first step / the GT's", f"{v_plan / v_gt:.3f}", "1 if the plan starts at the GT's speed."),
            ("◆ / ◀", f"{v0 / head:.3f}", "1 if the plan starts at the speed the pose head reads."),
            ("the model's height / the harness's", f"{h_b / sc.h:.3f}", "where `head_ratio` would be if the model measured distance by geometry alone: a road point X ahead is read at X · cam_h / h."),
        ]  # fmt: skip
        md = [
            f"**{self._headline(sc)}.** From stock, run here on {self.device}.",
            table("The camera's height over the road (m)", ("height", "m", "how it is measured"), heights),
            table("The speeds at t = 0 (m/s)", ("speed", "m/s", "how it is measured"), speeds),
            table("Ratios", ("ratio", "this turn", "what it would be"), ratios),
            f"**The ground truth's frame.** The GT is the device's own path (`future_in_ego_frame`), while the "
            f"plan is in the frame openpilot's calibration defines: this route's mount is {pitch:+.2f} deg pitch "
            f"and {yaw:+.2f} deg yaw ({sc.cal_scope}). A plan straight down the calibrated road therefore leaves "
            f"the GT at {abs(yaw):.2f} deg, {np.radians(abs(yaw)) * 30:.2f} m off at 30 m: part of every score here, "
            "as in the benchmark's, and the same for every config.",
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
        self._speed(fig.add_subplot(gs[1, 2]), sc, [(xy, ex, STOCK_C, "stock, here")])
        fig.suptitle(self._headline(sc), fontsize=12)
        _show_photo(fig, quality)

    # ---- checks ----

    def checks(self, key=None, tol_m=TOL_M):
        """Raise unless: (1) the node is the stock adapter, bit for bit, with no config and with an identity
        config sent as every config is (calib and post); (2) what this module draws is what the model was fed,
        on every path through `View`; (3) the sample rebuilt here scores the recorded plan as the recorded run
        scored it, so it is the sample that run was given, and the node's stock reproduces that run within tol_m;
        (4) the ticks counted here, from which the older frame is found, are the node's; (5) a cut of the frame
        (CHECK_CUT_DEG across) shows the model every pixel along the ray the recorded frame does."""
        sc = self.scene(key or self.keys[0])
        self._prefetch([self.scene(k) for k in self.keys])
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
            rescored = self.metrics(sid, rec, rex)
            run = self._record(sid)["metrics"]
            rows.append(
                {
                    "key": s.label,
                    "recorded plan's ADE, rescored here - the run's (m)": rescored["ade"] - run["ade"],
                    "and its lateral error's (m)": rescored["lat"] - run["lateral_mae"],
                    "stock - recorded, largest within 3 s (m)": np.abs(
                        mine[: self.n_h] - rec[: self.n_h]
                    ).max(),
                    "and over 10 s (m)": np.abs(mine - rec).max(),
                    "ADE, stock here - recorded (m)": self.metrics(sid, mine)["ade"] - rescored["ade"],
                    "ticks, node": r["ticks"],
                    "ticks, counted here": s.ticks,
                    "img's older frame": f"{s.older_kf} (observation {s.older - len(s.wire)})",
                }
            )
        df = pd.DataFrame(rows).set_index("key")
        print(
            f"3 and 4. the samples rebuilt here, and stock, run here on {self.device}, against {self.run_id}, "
            "recorded on a GPU:"
        )
        display(df.round(3))
        rescore = df.iloc[:, :2].abs().to_numpy().max()
        assert rescore < 1e-9, (
            f"the samples rebuilt here are not {self.run_id}'s: rescored {rescore:.3g} m off"
        )
        assert df["stock - recorded, largest within 3 s (m)"].max() < tol_m, (
            f"the node does not reproduce {self.run_id}"
        )
        assert (df["ticks, node"] == df["ticks, counted here"]).all(), (
            "the ticks counted here are not the node's"
        )

        # 5. A cut's geometry: openpilot, told the cut's intrinsics, must see each model pixel along the ray it
        # saw it along in the frame as recorded. A maps a recorded pixel to the cut's, so A @ W = W_cut.
        sc5 = self.scene(sc.sid, Setting(fov_deg=CHECK_CUT_DEG))
        g5, c = self.geometry(sc5), sc5.cut
        A = np.array(
            [
                [1 / c.scale, 0.0, (0.5 - c.box[0]) / c.scale - 0.5],
                [0.0, 1 / c.scale, (0.5 - c.box[1]) / c.scale - 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
        same_rpy = np.allclose(g5["rpy"], g["rpy"], atol=1e-6)
        for v in VIEWS:
            w, h = g["model_size"][v]
            grid = np.stack(np.meshgrid(np.linspace(0, w, 17), np.linspace(0, h, 9)), -1).reshape(-1, 2)
            err = np.abs(
                apply_h(warp(STOCK, sc5, v, g5), grid) - apply_h(A @ warp(STOCK, sc, v, g), grid)
            ).max()
            print(
                f"5. {sc.label}, {v}: a {CHECK_CUT_DEG:g} deg cut ({c.size[0]} x {c.size[1]}, {c.scale:.3f} camera px "
                f"each) sees every model pixel where the recorded frame does, to {err:.4f} cut px; rpy the same: "
                f"{same_rpy}"
            )
            assert err < 0.01 and same_rpy, f"the cut's geometry is not the recorded frame's ({v})"

    # ---- part 3 ----

    def tweak(self, key, config=None, run=True, overlay=("img",), noise=True, quality=80, against=None):
        """Part 3: one config on one turn, beside stock (or `against`, any other config), both run here.
        run=False draws only what the model would be shown, in a second or two. overlay: where the GT and the
        plans are drawn: "img" and "big_img", on that view's keyframe; "camera", on the keyframe as the
        camera took it. Every keyframe panel draws the row openpilot expects the horizon on (solid) and the
        camera's horizon (dashed). noise: add a row of the largest change any of the eight half-pixel SHIFTS
        of the baseline makes on this turn, which a change must beat to be the config's doing."""
        st = as_setting(config)
        base = STOCK if against is None else as_setting(against)
        name0 = "stock" if base == STOCK else "against"
        sc, sc0 = self.scene(key, st), self.scene(key, base)
        g, g0 = self.geometry(sc), self.geometry(sc0)
        view_specs(st, sc, g)  # a view that would look behind the camera raises here, before anything runs
        view_specs(base, sc0, g0)
        shifts = [shifted(base, x) for x in SHIFTS] if run and noise else []
        res = self.fetch([(base, sc.sid)] + ([(st, sc.sid)] if run else []) + [(x, sc.sid) for x in shifts])
        if "error" in res[0]:
            raise ValueError(f"{base.label()}: {res[0]['error']}")
        plans = [(plan(res[0]), res[0]["extras"], STOCK_C, name0)]
        if run and "error" in res[1]:
            print(f"{st.label()}: {res[1]['error']}")
        elif run:
            plans.append((plan(res[1]), res[1]["extras"], CONFIG_C, "config"))
        fig = plt.figure(figsize=(22, 18.5))
        gs = fig.add_gridspec(
            3, 4, height_ratios=[2.35, 1.1, 2.0], hspace=0.25, wspace=0.2, top=0.94, bottom=0.01, left=0.02,
            right=0.98,
        )  # fmt: skip
        views = gs[2, :].subgridspec(
            2, 4, hspace=0.25, wspace=0.2
        )  # the baseline's four inputs over the config's
        ax = fig.add_subplot(gs[0, 0:2])
        same = sc0.src == sc.src
        layers = ([(base, GREY)] if same else []) + [(st, None)]
        self._camera(ax, sc, g, layers, plans=plans if "camera" in overlay else ())
        grey = f"{name0}'s footprints grey; " if same else f"the config's source ({name0} is fed another); "
        ax.set_title(
            f"the keyframe: {grey}the config's img (yellow) and big_img (cyan, dashed),\n"
            "their model pixels every 64; the camera's horizon (dotted) and the road",
            fontsize=10,
        )
        self._road_map(fig.add_subplot(gs[0, 2]), sc, g, st, base, sc0, g0, name0)
        self._bev(fig.add_subplot(gs[0, 3]), sc, plans)
        self._speed(fig.add_subplot(gs[1, 0:2]), sc, plans)
        self._heading(fig.add_subplot(gs[1, 2:4]), sc, plans)
        rows_ = ((base, sc0, g0, name0, plans[:1]), (st, sc, g, "config", plans[1:]))
        for i, (which, s_, g_, name, pl) in enumerate(rows_):
            for j, (v, older) in enumerate(PANELS):
                when = f"t - {sc.pair_dt:.1f} s" if older else "t"
                self._view(
                    fig.add_subplot(views[i, j]), s_, g_, which, v, older, overlay=v in overlay and not older,
                    plans=pl, horizons=not older, title=f"{name}: {v} at {when}",
                )  # fmt: skip
        vs = "" if base == STOCK else f"\nagainst: {base.label()}"
        fig.suptitle(f"{self._headline(sc)}\nconfig: {st.label()}{vs}", fontsize=12)
        _show_photo(fig, quality)

        rows = {name0: self.metrics(sc.sid, plans[0][0], plans[0][1])}
        if len(plans) > 1:
            rows["config"] = self.metrics(sc.sid, plans[1][0], plans[1][1])
            rows["change"] = {
                k: v - rows[name0][k] for k, v in rows["config"].items() if isinstance(v, float)
            }
            rows["change"]["mode"] = f"{rows[name0]['mode']} -> {rows['config']['mode']}"
        if shifts:  # res[2:], after the baseline's and the config's
            ms = [self.metrics(sc.sid, plan(r), r["extras"]) for r in res[2:] if "error" not in r]
            d = {k: [abs(m[k] - v) for m in ms] for k, v in rows[name0].items() if isinstance(v, float)}
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
        cols += ["v0_ratio", "head_ratio", "plan_v0", "cam_h", "p_turn"]
        display(pd.DataFrame(rows).T[cols].infer_objects().round(3))
        if shifts:
            print(
                f"noise: the largest change that any of the {len(ms)} half-pixel shifts (T.SHIFTS) of {name0} makes "
                "on this turn, and every class they give; a change within it is not the config's doing"
            )
        F = road_map(st, sc, g, base, sc0, g0)
        p0, p1 = apply_h(F, [[20.0, 0.0], [20.0, 1.0]])
        line = (
            f"the config shows a road point (X, Y) about where {name0} shows (X x {p0[0] / 20:.3f}, "
            f"Y x {p1[1] - p0[1]:.3f}), measured 20 m ahead"
        )
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

    def noise_floor(self, keys=None, base=None):
        """{criterion: (per turn, for the mean)}: what the eight half-pixel SHIFTS of `base` (stock by default)
        do to each criterion. Per turn, the median |change| on one turn; for the mean, the largest |change| of
        the mean over the turns."""
        keys = tuple(self.sid(k) for k in (keys or self.keys))
        base = STOCK if base is None else as_setting(base)
        if (keys, base) not in self._floor:
            res = self.fetch([(st, k) for st in (base, *(shifted(base, x) for x in SHIFTS)) for k in keys])
            bad = sorted({r["error"] for r in res if "error" in r})
            if bad:
                raise ValueError(f"{base.label()}'s shifts cannot all be shown: {bad}")
            n = len(keys)
            ref = [self._crit(k, r)[0] for k, r in zip(keys, res[:n])]
            d = np.array(
                [
                    [self._crit(k, r)[0] - b for k, r, b in zip(keys, res[n * (i + 1) : n * (i + 2)], ref)]
                    for i in range(len(SHIFTS))
                ]
            )  # (shift, turn, criterion)
            self._floor[(keys, base)] = {
                name: (
                    float(np.nanmedian(np.abs(d[:, :, i]))),
                    float(np.nanmax(np.abs(np.nanmean(d[:, :, i], axis=1)))),
                )
                for i, (name, _, _) in enumerate(CRITERIA)
            }
        return self._floor[(keys, base)]

    @staticmethod
    def _verdict(d, per_turn, for_mean):
        """(verdict, better, same, worse) for per-turn changes d of a lower-is-better criterion: a turn counts
        when its change beats per_turn; the verdict needs the mean's change to beat for_mean, and more turns
        to move that way than the other."""
        better, worse = int((d < -per_turn).sum()), int((d > per_turn).sum())
        same = int(np.isfinite(d).sum()) - better - worse
        mean = float(np.nanmean(d))
        if mean < -for_mean and better > worse:
            return "better", better, same, worse
        if mean > for_mean and worse > better:
            return "worse", better, same, worse
        return "no clear change", better, same, worse

    def _run_all(self, settings, keys):
        """[[result per turn] per setting], raising if a setting cannot be shown on some turn."""
        n = len(keys)
        res = self.fetch([(st, k) for st in settings for k in keys])
        out = [res[i * n : (i + 1) * n] for i in range(len(settings))]
        for st, part in zip(settings, out):
            bad = [(self.scene(k).label, r["error"]) for k, r in zip(keys, part) if "error" in r]
            if bad:
                raise ValueError(f"{st.label()} cannot be shown: {bad}")
        return out

    def compare(self, config, keys=None, noise=True, against=None):
        """Part 4: the config against stock (or `against`, any other config) on every turn, both run here.
        noise: judge each change against what the half-pixel SHIFTS of the baseline do (noise_floor); without it
        any change at all counts."""
        st = as_setting(config)
        base = STOCK if against is None else as_setting(against)
        name0 = "stock" if base == STOCK else "against"
        keys = [self.sid(k) for k in (keys or self.keys)]
        n = len(keys)
        r0, r1 = self._run_all([base, st], keys)
        res = r0 + r1
        crit = [[self._crit(k, r) for k, r in zip(keys, part)] for part in (r0, r1)]
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
            "p_turn",
        )
        per = {}
        for c in cols:
            a, b = np.array([m[c] for m in ms], float), np.array([m[c] for m in mc], float)
            per |= {(c, name0): a, (c, "config"): b, (c, "change"): b - a}
        per[("mode", f"{name0} -> config")] = [f"{a['mode']} -> {b['mode']}" for a, b in zip(ms, mc)]
        per_df = pd.DataFrame(per, index=[self.scene(k).label for k in keys])
        per_df.columns = pd.MultiIndex.from_tuples(per_df.columns)
        vs = "" if base == STOCK else f", against {base.label()}"
        display(
            Markdown(f"**Each turn, {name0} and config: {st.label()}**{vs} (both run here on {self.device})")
        )
        display(per_df.round(3))

        floor = self.noise_floor(keys, base) if noise else {name: (0.0, 0.0) for name, _, _ in CRITERIA}
        rows, groups = [], {}
        for i, (name, group, _) in enumerate(CRITERIA):
            d = cc[:, i] - cs[:, i]
            per_turn, for_mean = floor[name]
            verdict, better, same, worse = self._verdict(d, per_turn, for_mean)
            groups.setdefault(group, []).append(verdict)
            rows.append(
                {
                    "criterion": name,
                    "judges": group,
                    name0: float(np.nanmean(cs[:, i])),
                    "config": float(np.nanmean(cc[:, i])),
                    "change": float(np.nanmean(d)),
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
                name0: planned[0],
                "config": planned[1],
                "change": planned[1] - planned[0],
                "better / same / worse": "",
                "verdict": verdict,
            }
        )
        display(Markdown("**Means over the turns, and the verdict**"))
        display(pd.DataFrame(rows).set_index("criterion").round(3))
        lines = []
        for group, vs_ in groups.items():
            word = (
                "better"
                if "better" in vs_ and "worse" not in vs_
                else "worse"
                if "worse" in vs_ and "better" not in vs_
                else "mixed"
                if "better" in vs_
                else "no clear change"
            )
            lines.append(f"- **{group}: {word}** ({', '.join(vs_)})")
        how = "beyond what eight half-pixel shifts do" if noise else "any change counts (noise=False)"
        display(
            Markdown(f"**{st.label()}**, against {name0}{vs} on {n} turns ({how}):\n\n" + "\n".join(lines))
        )

        fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 13), squeeze=False)
        for j, k in enumerate(keys):
            sc = self.scene(k)
            plans = [
                (plan(res[j]), res[j]["extras"], STOCK_C, name0),
                (plan(res[n + j]), res[n + j]["extras"], CONFIG_C, "config"),
            ]
            self._bev(axes[0, j], sc, plans)
            self._speed(axes[1, j], sc, plans, legend=j == 0)
            self._heading(axes[2, j], sc, plans)
            axes[0, j].set_title(
                f"{sc.label}: {sc.side}, {sc.turn_deg:+.0f} deg\nADE {ms[j]['ade']:.2f} -> {mc[j]['ade']:.2f} m",
                fontsize=10,
            )
            axes[1, j].set_title("speed (m/s); t = 0: ▶ pose, ◀ head, ◆ plan_v0", fontsize=9)
            axes[2, j].set_title("heading toward the route (deg)", fontsize=9)
        fig.suptitle(f"{name0} (orange) and the config (purple): {st.label()}{vs}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
        plt.show()

    # ---- part 5 ----

    def arms(self, arms, keys=None, noise=True):
        """Part 5: several configs on every turn, side by side. arms: {name: config}, at most five, the first
        the reference: each other arm's change is judged against it, beyond what the half-pixel SHIFTS of the
        first arm do (noise=False: any change counts). Each arm keeps its colour (ARM_C) by its place in the
        dict, so reordering the dict repaints it; adding an arm at the end does not."""
        if not 1 <= len(arms) <= len(ARM_C):
            raise ValueError(f"arms takes 1 to {len(ARM_C)} configs; split a longer list over several calls")
        names = list(arms)
        sts = [as_setting(c) for c in arms.values()]
        keys = [self.sid(k) for k in (keys or self.keys)]
        n = len(keys)
        res = self._run_all(sts, keys)
        crit = [[self._crit(k, r) for k, r in zip(keys, part)] for part in res]
        cs = [np.array([c for c, _ in part]) for part in crit]
        ms = [[m for _, m in part] for part in crit]

        display(
            Markdown("**The arms**\n\n" + "\n".join(f"- **{a}**: {st.label()}" for a, st in zip(names, sts)))
        )
        cols = (
            "ade",
            "path_err",
            "dpsi_err",
            "turn_frac",
            "speed_rmse",
            "v0_ratio",
            "head_ratio",
            "p_turn",
            "mode",
        )
        per = {(c, a): [m[c] for m in ma] for c in cols for a, ma in zip(names, ms)}
        per_df = pd.DataFrame(per, index=[self.scene(k).label for k in keys])
        per_df.columns = pd.MultiIndex.from_tuples(per_df.columns)
        display(Markdown(f"**Each turn** (run here on {self.device})"))
        display(per_df.round(3))

        floor = self.noise_floor(keys, sts[0]) if noise else {name: (0.0, 0.0) for name, _, _ in CRITERIA}
        rows = []
        for i, (name, group, _) in enumerate(CRITERIA):
            row = {"criterion": name, "judges": group}
            for a, c in zip(names, cs):
                row[a] = float(np.nanmean(c[:, i]))
            row["noise, mean"] = floor[name][1]
            for a, c in zip(names[1:], cs[1:]):
                v, b, sm, w = self._verdict(c[:, i] - cs[0][:, i], *floor[name])
                row[f"{a}: verdict"] = f"{v} ({b} / {sm} / {w})"
            rows.append(row)
        planned = {
            "criterion": "turns planned toward the route (count, higher is better)",
            "judges": "alignment",
        }
        planned |= {a: sum(m["planned"] for m in ma) for a, ma in zip(names, ms)}
        p_turn = {
            "criterion": "p_turn: the model's probability of the route's turn (mean)",
            "judges": "desire",
        }
        p_turn |= {a: float(np.nanmean([m["p_turn"] for m in ma])) for a, ma in zip(names, ms)}
        rows += [planned, p_turn]
        how = "beyond what eight half-pixel shifts of it do" if noise else "any change counts (noise=False)"
        display(
            Markdown(
                f"**Means over the {n} turns.** Each verdict is that arm against **{names[0]}**, {how}; in "
                "brackets, the turns better / the same / worse."
            )
        )
        display(pd.DataFrame(rows).set_index("criterion").round(3))

        colours = ARM_C[: len(names)]
        fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 13), squeeze=False)
        for j, k in enumerate(keys):
            sc = self.scene(k)
            plans = [(plan(part[j]), part[j]["extras"], c, a) for part, c, a in zip(res, colours, names)]
            self._bev(axes[0, j], sc, plans)
            self._speed(axes[1, j], sc, plans, legend=j == 0)
            self._heading(axes[2, j], sc, plans)
            axes[0, j].set_title(
                f"{sc.label}: {sc.side}, {sc.turn_deg:+.0f} deg\nADE "
                + ", ".join(f"{m[j]['ade']:.2f}" for m in ms)
                + " m",
                fontsize=10,
            )
            axes[1, j].set_title("speed (m/s); t = 0: ▶ pose, ◀ head, ◆ plan_v0", fontsize=9)
            axes[2, j].set_title("heading toward the route (deg)", fontsize=9)
        fig.suptitle("the arms, in the legend's order: " + ", ".join(names), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
        plt.show()
