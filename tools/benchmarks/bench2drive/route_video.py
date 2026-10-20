"""One video and one summary image per Bench2Drive route.

    python -m tools.benchmarks.bench2drive.route_video <route_dir> --expert <dir> --out <dir>
    python -m tools.benchmarks.bench2drive.route_video --runs <run_or_runs_dir> --expert <dir> --out <dir> --jobs 6

``<route_dir>`` is one route's paddock artifact directory recorded with
``record = "true"`` (``front/``, ``bev/``, ``model_road/``, ``model_wide/``,
``ticks.jsonl``, ``route_plan.json``, ``results.json``); ``--runs`` renders every
route found below a run directory or a directory of runs (the latest scored
attempt per route). ``--expert`` is the directory ``expert.py`` fetched
Think2Drive's drives into; a route with no drive there is drawn against its
route alone. Writes ``<out>/<route_id>/{video.mp4, summary.png, route.json}``.

Every frame is one CARLA tick in two (10 fps, so the video plays in real time)::

    +---------------------------+------------------+--------------+
    | front camera              | BEV camera       | model's wide |
    |  plan + lane lines        |  plan, expert,   | model's road |
    |  projected, HUD           |  route, actors   |              |
    +---------------------------+------------------+--------------+
    | route map: route, expert, | speed | progress | controls     |
    |  driven path, infractions | ...   score card                |
    +---------------------------+---------------------------------+

Frames: the front and BEV cameras are the agent's recording cameras (CAM_FRONT,
800x450, 70 deg at x=1.5 z=1.5; BEV 512x512, 60 deg, 30 m up, looking down),
fixed to the car, so ego-frame points -- x forward, y left, on the ground --
project with those mounts alone. The route map is CARLA's world, drawn as seen
from above (CARLA's y points down the page). The expert's "ghost" is where
Think2Drive was at the same time since its start: ahead of the ghost is faster
than the expert, behind it slower.

Needs opencv-python-headless and an ``ffmpeg`` with libx264 on PATH (or
``PADDOCK_FFMPEG``). Encodes on the CPU on purpose: the GPU is CARLA's.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paddock.adapters.benchmarks.bench2drive.recorder import to_ego  # noqa: E402
from tools.benchmarks.bench2drive import expert as expert_mod  # noqa: E402
from tools.benchmarks.bench2drive import runs  # noqa: E402

W, H = 1920, 1080
FPS = 10
TICK_HZ = 20.0

# Recording cameras (agent.py: CAM_TF / CAM_FOV / CAM_RES, and the BEV sensor).
FRONT_W, FRONT_H, FRONT_FOV, FRONT_X, FRONT_Z = 800, 450, 70.0, 1.5, 1.5
BEV_PX, BEV_FOV, BEV_Z = 512, 60.0, 30.0
BEV_PX_PER_M = BEV_PX / (2 * BEV_Z * math.tan(math.radians(BEV_FOV / 2)))

# BGR
GREEN, ORANGE, BLUE, WHITE, RED = (60, 220, 60), (0, 150, 255), (255, 160, 40), (255, 255, 255), (60, 60, 230)
YELLOW, CYAN, MAGENTA, GRAY = (40, 230, 230), (230, 220, 40), (220, 80, 220), (150, 150, 150)
PANEL_BG = (24, 24, 28)
SENT_ROUTE = (255, 215, 140)  # light blue: the sparse route the node was sent

#: Captions of the model-camera thumbnails that are not just their camera's name.
VIEW_LABELS = {"model_wide": "model input: wide (big_img)", "model_road": "model input: road (img)"}


# --------------------------------------------------------------------------- geometry


def front_project(xy: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Ego-frame ground points -> CAM_FRONT pixels; NaN behind the camera."""
    f = FRONT_W / (2 * math.tan(math.radians(FRONT_FOV / 2)))
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    depth = pts[:, 0] - FRONT_X
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FRONT_W / 2 + f * (-pts[:, 1]) / depth
        v = FRONT_H / 2 + f * (FRONT_Z - z) / depth
    out = np.stack([u, v], axis=-1)
    out[depth < 1.0] = np.nan
    return out


def bev_project(xy: np.ndarray, size: int) -> np.ndarray:
    """Ego-frame points -> pixels of the BEV camera image scaled to ``size``: forward is up, left is left."""
    s = BEV_PX_PER_M * size / BEV_PX
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.stack([size / 2 - s * pts[:, 1], size / 2 - s * pts[:, 0]], axis=-1)


def polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int = 2, closed: bool = False) -> None:
    pts = np.asarray(pts, dtype=float)
    keep = np.isfinite(pts).all(axis=1)
    # Split at NaNs so a run of points behind the camera does not draw a line across the image.
    runs_, current = [], []
    for p, k in zip(pts, keep, strict=True):
        if k and abs(p[0]) < 1e5 and abs(p[1]) < 1e5:
            current.append(p)
        elif current:
            runs_.append(current)
            current = []
    if current:
        runs_.append(current)
    for run in runs_:
        if len(run) > 1:
            cv2.polylines(
                img, [np.round(np.array(run)).astype(np.int32)], closed, color, thickness, cv2.LINE_AA
            )


def text(img, s: str, org, scale: float = 0.55, color=WHITE, thickness: int = 1) -> None:
    x, y = int(org[0]), int(org[1])
    # A 1 px dark outline under the glyphs, not an offset shadow: the offset one came
    # through x264 as a faint second copy of the text beside the first.
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def rotated_box(img, center, yaw_deg: float, half_l: float, half_w: float, s: float, color) -> None:
    """A box in the BEV: ``center`` in pixels, ``yaw_deg`` ego-relative (left-positive), half sizes in metres."""
    a = math.radians(yaw_deg)
    fwd = np.array([-math.sin(a), -math.cos(a)])  # ego +x is up, ego +y is left (pixel -x)
    left = np.array([-math.cos(a), math.sin(a)])
    c = np.asarray(center)
    corners = [
        c + s * (fx * half_l * fwd + lx * half_w * left) for fx, lx in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    cv2.polylines(img, [np.round(np.array(corners)).astype(np.int32)], True, color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- data

#: Panel order of the model-camera thumbnails (recorder.model_view_dir names):
#: openpilot's pair stacked; a surround rig two abreast with each side's cameras
#: in its own column, front and back first. Anything else follows, by name.
VIEW_ORDER = (
    "model_wide", "model_road",
    "model_front", "model_back",
    "model_front_left", "model_front_right",
    "model_back_left", "model_back_right",
)  # fmt: skip


def model_views(route_dir: Path) -> list[str]:
    """The model-camera thumbnail directories a route recorded, in panel order."""
    found = {p.name for p in Path(route_dir).glob("model_*") if p.is_dir()}
    return [v for v in VIEW_ORDER if v in found] + sorted(found - set(VIEW_ORDER))


def model_state(ticks: list[dict], model: str) -> list[dict]:
    """Per tick, the scalars the model reported under its own name in ``extras``.

    A node that replans less often than it is called answers the ticks between
    with a held plan and fewer fields (AutoE2E: only ``held_s``); those ticks show
    the last full report with the held fields over it, so the panel does not
    blank on every other frame. Nested values (lane lines, lead lists) are drawn
    elsewhere or not at all, and are left out.
    """
    out: list[dict] = []
    last_full: dict = {}
    for rec in ticks:
        own = (rec.get("extras") or {}).get(model)
        own = (
            {k: v for k, v in own.items() if not isinstance(v, (dict, list))} if isinstance(own, dict) else {}
        )
        if len(own) > 1:
            last_full = own
            out.append(own)
        else:
            out.append({**last_full, **own})
    return out


class RouteData:
    """Everything one route's frames are drawn from, loaded once."""

    def __init__(
        self, route_dir: Path, expert_root: Path | None, meta: dict | None, model: str | None = None
    ) -> None:
        self.dir = Path(route_dir)
        # <runs>/bench2drive/<model>/<run_id>/<route>: the layout every paddock run writes.
        self.model = model or self.dir.parent.parent.name
        name = runs._ROUTE_DIR.match(self.dir.name)
        self.route_id = name["rid"] if name else self.dir.name
        self.meta = meta or {
            "town": name["town"] if name else "",
            "scenario": name["scenario"] if name else "",
        }
        self.result = runs.load_result(self.dir)
        self.ticks = runs.load_ticks(self.dir)
        self.plan = runs.load_json(self.dir / "route_plan.json") or {}
        self.route_xy = np.asarray(self.plan.get("xy") or [], dtype=float).reshape(-1, 2)
        self.route_s = np.asarray(self.plan.get("s") or [], dtype=float)
        self.route_len = float(self.plan.get("length_m") or (self.route_s[-1] if len(self.route_s) else 0.0))
        self.actor_meta = {}
        actors_path = self.dir / "actors.jsonl"
        if actors_path.exists():
            for line in actors_path.read_text().splitlines():
                try:
                    a = json.loads(line)
                    self.actor_meta[a["id"]] = a
                except ValueError:
                    pass
        self.expert = expert_mod.load_expert(expert_root, self.route_id) if expert_root else None
        self.expert_xy = np.zeros((0, 2))
        self.expert_t = np.zeros(0)
        self.expert_v = np.zeros(0)
        self.expert_s = np.zeros(0)
        if self.expert:
            fr = self.expert["frames"]
            self.expert_xy = np.array([[f["x"], f["y"]] for f in fr], dtype=float)
            self.expert_t = np.arange(len(fr)) / float(self.expert.get("hz", 10.0))
            self.expert_v = np.array([f.get("speed") or 0.0 for f in fr], dtype=float)
            self.expert_s = self.progress(self.expert_xy)
        # Ego series.
        tk = self.ticks
        self.t = np.array([r["time"] for r in tk], dtype=float) if tk else np.zeros(0)
        if len(self.t):
            self.t = self.t - self.t[0]
        self.ego_xy = np.array([[r["ego"]["x"], r["ego"]["y"]] for r in tk], dtype=float).reshape(-1, 2)
        self.speed = np.array([r["ego"]["speed"] for r in tk], dtype=float)
        self.v_des = np.array([r["ctrl"].get("v_des") or 0.0 for r in tk], dtype=float)
        self.limit = np.array([r["ego"].get("speed_limit") or np.nan for r in tk], dtype=float)
        self.ego_s = np.array([(r.get("route") or {}).get("s", np.nan) for r in tk], dtype=float)
        self.throttle = np.array([r["ctrl"]["throttle"] for r in tk], dtype=float)
        self.brake = np.array([r["ctrl"]["brake"] for r in tk], dtype=float)
        self.steer = np.array([r["ctrl"]["steer"] for r in tk], dtype=float)
        self.lead_prob = np.array(
            [((r.get("extras") or {}).get("lead") or [{}])[0].get("prob", np.nan) for r in tk], dtype=float
        )
        self.views = model_views(self.dir)
        self.model_state = model_state(tk, self.model)
        self.infractions = runs.infraction_points(self.result) if self.result else []
        # The tick at which each located infraction happened: the ego's nearest approach.
        for inf in self.infractions:
            inf["tick"] = None
            if inf["x"] is not None and len(self.ego_xy):
                d = np.hypot(self.ego_xy[:, 0] - inf["x"], self.ego_xy[:, 1] - inf["y"])
                inf["tick"] = int(np.argmin(d))

    def progress(self, xy: np.ndarray) -> np.ndarray:
        """Along-route metres of world points: their nearest point on the 1 m route."""
        if not len(self.route_xy) or not len(xy):
            return np.full(len(xy), np.nan)
        out = np.empty(len(xy))
        for k, (x, y) in enumerate(xy):
            out[k] = self.route_s[int(np.argmin(np.hypot(self.route_xy[:, 0] - x, self.route_xy[:, 1] - y)))]
        return out

    def expert_speed_at_progress(self, s: float) -> float:
        if not len(self.expert_s) or not np.isfinite(s):
            return np.nan
        k = int(np.argmin(np.abs(self.expert_s - s)))
        return float(self.expert_v[k])

    def scores(self) -> dict:
        r = self.result or {}
        sc = r.get("scores") or {}
        return {
            "status": r.get("status", "running"),
            "ds": sc.get("score_composed"),
            "rc": sc.get("score_route"),
            "penalty": sc.get("score_penalty"),
            "game_s": (r.get("meta") or {}).get("duration_game"),
            "route_length": (r.get("meta") or {}).get("route_length"),
        }


# --------------------------------------------------------------------------- static layers


def _fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def route_map(rd: RouteData, w: int, h: int):
    """The world-fixed map as an image, and a world -> pixel function for per-frame marks."""
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#18181c")
    ax = fig.add_axes((0.06, 0.08, 0.9, 0.84), facecolor="#18181c")
    pts = [p for p in (rd.route_xy, rd.ego_xy, rd.expert_xy) if len(p)]
    allp = np.concatenate(pts) if pts else np.zeros((1, 2))
    lo, hi = allp.min(axis=0) - 15, allp.max(axis=0) + 15
    if len(rd.route_xy):
        ax.plot(
            rd.route_xy[:, 0], rd.route_xy[:, 1], color="#4a90e2", lw=7, alpha=0.35, solid_capstyle="round"
        )
        ax.plot(rd.route_xy[:, 0], rd.route_xy[:, 1], color="#4a90e2", lw=1.2, label="route")
        ax.plot(*rd.route_xy[0], "o", color="#4a90e2", ms=7)
        ax.plot(*rd.route_xy[-1], "s", color="#4a90e2", ms=7)
    if len(rd.expert_xy):
        ax.plot(
            rd.expert_xy[:, 0],
            rd.expert_xy[:, 1],
            color="#ff9600",
            lw=1.6,
            ls="--",
            label="expert (Think2Drive)",
        )
    for light in rd.plan.get("traffic_lights") or []:
        ax.plot(light["x"], light["y"], marker="^", color="#e6e600", ms=8, ls="none")
    for stop in rd.plan.get("stop_signs") or []:
        ax.plot(stop["x"], stop["y"], marker="8", color="#e63c3c", ms=8, ls="none")
    for inf in rd.infractions:
        if inf["x"] is not None:
            ax.plot(inf["x"], inf["y"], "x", color="#ff3030", ms=12, mew=3)
            ax.annotate(inf["kind"].replace("collisions_", "coll. "), (inf["x"], inf["y"]), color="#ff7070",
                        fontsize=8, xytext=(5, 5), textcoords="offset points")  # fmt: skip
    ax.plot([], [], color="white", lw=2, label=f"{rd.model} (driven)")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # CARLA's y points down the page seen from above
    ax.tick_params(colors="#909090", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#404040")
    ax.legend(loc="upper right", fontsize=7, facecolor="#18181c", labelcolor="white", framealpha=0.7)
    ax.set_title("route map (world, metres)  ^ light  o stop  x infraction", color="white", fontsize=9)
    fig.canvas.draw()
    trans = ax.transData

    def to_px(xy: np.ndarray) -> np.ndarray:
        p = trans.transform(np.asarray(xy, dtype=float).reshape(-1, 2))
        return np.stack([p[:, 0], h - p[:, 1]], axis=-1)

    img = _fig_to_bgr(fig)
    plt.close(fig)
    return img, to_px


def timeseries(rd: RouteData, w: int, h: int):
    """Speed, progress and controls against time, with an x -> pixel function for the cursor."""
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#18181c")
    axes = [fig.add_axes((0.09, 0.69 - k * 0.3, 0.88, 0.26), facecolor="#202026") for k in range(3)]
    t_end = max(
        float(rd.t[-1]) if len(rd.t) else 1.0, float(rd.expert_t[-1]) if len(rd.expert_t) else 1.0, 1.0
    )
    ax = axes[0]
    ax.plot(rd.t, rd.speed, color="white", lw=1.3, label="ego speed")
    ax.plot(rd.t, rd.v_des, color="#3cdc3c", lw=0.9, alpha=0.8, label="target (tracker)")
    if np.isfinite(rd.limit).any():
        ax.plot(rd.t, rd.limit, color="#909090", lw=0.8, ls=":", label="limit")
    if len(rd.expert_t):
        ax.plot(rd.expert_t, rd.expert_v, color="#ff9600", lw=1.1, ls="--", label="expert speed")
    ax.set_ylabel("m/s", color="#c0c0c0", fontsize=8)
    ax = axes[1]
    if rd.route_len > 0:
        ax.plot(rd.t, 100 * rd.ego_s / rd.route_len, color="white", lw=1.3, label="ego progress")
        if len(rd.expert_t):
            ax.plot(
                rd.expert_t,
                100 * rd.expert_s / rd.route_len,
                color="#ff9600",
                lw=1.1,
                ls="--",
                label="expert",
            )
    rc = rd.scores()["rc"]
    if rc is not None:
        ax.axhline(rc, color="#4a90e2", lw=1, ls=":", label=f"scored RC {rc:.1f}%")
    ax.set_ylim(-2, 105)
    ax.set_ylabel("% route", color="#c0c0c0", fontsize=8)
    ax = axes[2]
    ax.plot(rd.t, rd.throttle, color="#3cdc3c", lw=0.9, label="throttle")
    ax.plot(rd.t, rd.brake, color="#e63c3c", lw=0.9, label="brake")
    ax.plot(rd.t, rd.steer, color="#40c8ff", lw=0.9, label="steer")
    if np.isfinite(rd.lead_prob).any():
        ax.plot(rd.t, rd.lead_prob, color="#e650e6", lw=0.8, alpha=0.8, label="lead prob")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time since route start (s)", color="#c0c0c0", fontsize=8)
    for ax in axes:
        ax.set_xlim(0, t_end)
        ax.tick_params(colors="#909090", labelsize=7)
        ax.grid(color="#33333a", lw=0.5)
        ax.legend(
            loc="upper left", fontsize=6.5, facecolor="#202026", labelcolor="white", framealpha=0.7, ncol=4
        )
        for inf in rd.infractions:
            if inf["tick"] is not None and inf["tick"] < len(rd.t):
                ax.axvline(rd.t[inf["tick"]], color="#ff3030", lw=1, alpha=0.7)
    fig.canvas.draw()
    top = axes[0].transData

    def x_px(t: float) -> int:
        return int(round(top.transform((t, 0))[0]))

    y0 = int(h - axes[0].get_window_extent().y1)
    y1 = int(h - axes[2].get_window_extent().y0)
    img = _fig_to_bgr(fig)
    plt.close(fig)
    return img, x_px, (y0, y1)


def score_card(rd: RouteData, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), PANEL_BG, np.uint8)
    sc = rd.scores()
    y = 28
    text(img, f"route {rd.route_id}  {rd.meta.get('town', '')}", (12, y), 0.62)
    y += 24
    text(img, rd.meta.get("scenario", ""), (12, y), 0.55, YELLOW)
    y += 22
    ab = ", ".join(rd.meta.get("abilities") or runs.abilities(rd.meta.get("scenario", "")))
    text(img, f"ability: {ab}", (12, y), 0.45, GRAY)
    y += 30
    status = sc["status"]
    color = GREEN if status in ("Perfect", "Completed") else RED if status.startswith("Failed") else WHITE
    text(img, status, (12, y), 0.5, color)
    y += 30

    def fmt(v, f="{:.2f}"):
        return "-" if v is None else f.format(v)

    text(img, f"Driving score  {fmt(sc['ds'])}", (12, y), 0.6)
    y += 26
    text(img, f"= route completion {fmt(sc['rc'])} %", (12, y), 0.5)
    y += 22
    text(img, f"x infraction penalty {fmt(sc['penalty'], '{:.3f}')}", (12, y), 0.5)
    y += 28
    rl = sc["route_length"] or rd.route_len
    text(img, f"route {fmt(rl, '{:.0f}')} m   game time {fmt(sc['game_s'], '{:.1f}')} s", (12, y), 0.45, GRAY)
    y += 26
    counts: dict[str, int] = {}
    for inf in rd.infractions:
        counts[inf["kind"]] = counts.get(inf["kind"], 0) + 1
    if counts:
        text(img, "infractions (multiplier each):", (12, y), 0.45)
        y += 20
        for kind, n in counts.items():
            mult = runs.PENALTY.get(kind)
            text(img, f"  {n} x {kind}" + (f"  x{mult}" if mult else ""), (12, y), 0.42, (120, 120, 255))
            y += 18
    else:
        text(img, "no penalised infractions", (12, y), 0.45, GREEN)
        y += 20
    devs = (rd.result or {}).get("infractions", {})
    n_min = len(devs.get("min_speed_infractions") or [])
    if n_min:
        text(img, f"  ({n_min} min-speed entries, not penalised)", (12, y), 0.4, GRAY)
        y += 18
    if rd.expert:
        y += 8
        text(img, f"expert: {rd.expert['duration_s']:.1f} s in the dataset clip", (12, y), 0.42, ORANGE)
    else:
        y += 8
        text(img, "expert: no Think2Drive drive of this route", (12, y), 0.42, GRAY)
    return img


# --------------------------------------------------------------------------- frames


def _read(path: Path, size) -> np.ndarray:
    img = cv2.imread(str(path)) if path.exists() else None
    if img is None:
        img = np.full((size[1], size[0], 3), PANEL_BG, np.uint8)
        text(img, "no frame", (10, 30))
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


COMMANDS = {0: "RIGHT", 1: "LEFT", 2: "STRAIGHT", 3: "FOLLOW"}


def draw_frame(rd: RouteData, k: int, static: dict) -> np.ndarray:
    canvas = np.full((H, W, 3), PANEL_BG, np.uint8)
    rec = rd.ticks[k]
    ext = rec.get("extras") or {}
    plan = np.asarray(rec.get("plan") or [], dtype=float).reshape(-1, 2)
    ego = rec["ego"]

    # -- front camera
    fw, fh = 960, 540
    front = _read(rd.dir / "front" / f"{rec['t']:05d}.jpg", (FRONT_W, FRONT_H))
    lane = ext.get("lane") or {}
    for line, prob in zip(lane.get("lines") or [], lane.get("line_probs") or [], strict=False):
        if prob > 0.3:
            col = tuple(int(c * (0.4 + 0.6 * prob)) for c in YELLOW)
            polyline(front, front_project(np.asarray(line)), col, 1)
    for edge in lane.get("edges") or []:
        polyline(front, front_project(np.asarray(edge)), RED, 1)
    if len(plan):
        p = front_project(np.vstack([[FRONT_X + 1.0, 0.0], plan]))
        polyline(front, p, GREEN, 3)
        for q in p[1:]:
            if np.isfinite(q).all():
                cv2.circle(front, (int(q[0]), int(q[1])), 4, GREEN, -1, cv2.LINE_AA)
    front = cv2.resize(front, (fw, fh))
    canvas[0:fh, 0:fw] = front
    sent = rec.get("sent") or {}
    hud = [
        f"t {rec['time']:.1f}s  tick {rec['t']}  speed {3.6 * ego['speed']:.1f} km/h  target {3.6 * (rec['ctrl'].get('v_des') or 0):.1f}",
        f"cmd {COMMANDS.get(sent.get('command'), sent.get('command'))}  nav {sent.get('nav_maneuver') or '-'} {sent.get('nav_distance_m') if sent.get('nav_distance_m') is not None else ''}"
        f"  limit {3.6 * (ego.get('speed_limit') or 0):.0f} km/h",
        f"steer {rec['ctrl']['steer']:+.2f}  thr {rec['ctrl']['throttle']:.2f}  brk {rec['ctrl']['brake']:.2f}  infer {rec.get('infer_ms', 0):.0f} ms",
    ]
    lead = (ext.get("lead") or [None])[0]
    if lead:
        hud.append(f"lead p={lead['prob']:.2f} d={lead['d']:.1f}m v_rel={lead['v_rel']:+.1f}")
    lights = rec.get("lights") or {}
    nxt = lights.get("next")
    if nxt:
        hud.append(
            f"next light {str(nxt.get('state')).split('.')[-1]} in {nxt['d']:.0f} m"
            + ("  (at light)" if lights.get("at") else "")
        )
    route = rec.get("route") or {}
    hud.append(f"route {100 * route.get('frac', 0):.1f}%  off-route {route.get('dev', 0):.1f} m")
    for j, line in enumerate(hud):
        text(canvas, line, (10, 24 + 22 * j), 0.55)

    # -- BEV camera
    bs = 540
    bev = _read(rd.dir / "bev" / f"{rec['t']:05d}.jpg", (bs, bs))
    ex, ey, eyaw = ego["x"], ego["y"], ego["yaw"]
    if len(rd.route_xy):
        polyline(bev, bev_project(to_ego(rd.route_xy, ex, ey, eyaw), bs), BLUE, 2)
    if len(rd.expert_xy):
        polyline(bev, bev_project(to_ego(rd.expert_xy, ex, ey, eyaw), bs), ORANGE, 2)
        tt = rec["time"] - rd.ticks[0]["time"]
        g = int(np.clip(np.searchsorted(rd.expert_t, tt), 0, len(rd.expert_xy) - 1))
        gp = bev_project(to_ego(rd.expert_xy[g : g + 1], ex, ey, eyaw), bs)[0]
        if 0 <= gp[0] < bs and 0 <= gp[1] < bs:
            cv2.circle(bev, (int(gp[0]), int(gp[1])), 8, ORANGE, 2, cv2.LINE_AA)
    trail = rd.ego_xy[max(0, k - 400) : k + 1]
    if len(trail) > 1:
        polyline(bev, bev_project(to_ego(trail, ex, ey, eyaw), bs), WHITE, 1)
    s = BEV_PX_PER_M * bs / BEV_PX
    for row in rec.get("actors") or []:
        aid, ax_, ay_, ayaw, _v = row
        meta = rd.actor_meta.get(aid, {})
        typ = meta.get("type", "")
        col = MAGENTA if typ.startswith("walker") else GRAY if typ.startswith("static") else CYAN
        half = meta.get("ext") or [2.2, 1.0]
        c = bev_project(np.array([[ax_, ay_]]), bs)[0]
        rotated_box(bev, c, ayaw, half[0], half[1], s, col)
    for line, prob in zip(lane.get("lines") or [], lane.get("line_probs") or [], strict=False):
        if prob > 0.3:
            polyline(bev, bev_project(np.asarray(line), bs), YELLOW, 1)
    # The route as the node got it (agent.py `route_xy`, ego frame): the leaderboard's
    # sparse plan, whose chords a route-rasterising model (AutoE2E) followed.
    sent_route = np.asarray((rec.get("sent") or {}).get("route_xy") or [], dtype=float).reshape(-1, 2)
    if len(sent_route):
        q = bev_project(sent_route, bs)
        polyline(bev, q, SENT_ROUTE, 1)
        for u, v in q:
            if 0 <= u < bs and 0 <= v < bs:
                cv2.circle(bev, (int(u), int(v)), 4, SENT_ROUTE, 1, cv2.LINE_AA)
    if len(plan):
        p = bev_project(np.vstack([[0.0, 0.0], plan]), bs)
        polyline(bev, p, GREEN, 3)
    rotated_box(bev, (bs / 2, bs / 2), 0.0, 2.4, 1.05, s, WHITE)
    text(
        bev,
        "BEV  plan  expert  route" + ("  o sent" if len(sent_route) else "") + "  driven",
        (8, bs - 10),
        0.45,
    )
    canvas[0:bs, fw : fw + bs] = bev

    # -- model cameras (10 Hz thumbnails; nearest even tick), then what the model reported
    col_x, col_w = fw + bs, W - fw - bs
    t_even = rec["t"] - rec["t"] % 2
    if len(rd.views) <= 2:  # openpilot's pair, stacked at the column's width
        cols, cw, ch, gap = 1, col_w, 263, 14
    else:  # a surround rig, two abreast
        cols, cw, gap = 2, col_w // 2, 4
        ch = int(cw * FRONT_H / FRONT_W)
    y_end = 0
    for j, view in enumerate(rd.views):
        row, col = divmod(j, cols)
        y0 = row * (ch + gap)
        if y0 + ch > bs:
            break
        img = _read(rd.dir / view / f"{t_even:05d}.jpg", (cw, ch))
        text(img, VIEW_LABELS.get(view, view.removeprefix("model_").replace("_", " ")), (6, 18), 0.45)
        canvas[y0 : y0 + ch, col_x + col * cw : col_x + (col + 1) * cw] = img
        y_end = y0 + ch
    state = rd.model_state[k]
    if state and bs - y_end > 60:
        y = y_end + 26
        text(
            canvas,
            f"{rd.model} reported" + (" (plan held)" if "held_s" in state else ""),
            (col_x + 8, y),
            0.5,
        )
        line = ""
        for key, value in state.items():
            item = f"{key} {value:.3g}" if isinstance(value, float) else f"{key} {value}"
            if line and len(line) + len(item) > 44:
                y += 20
                if y > bs - 8:
                    break
                text(canvas, line, (col_x + 8, y), 0.45, GRAY)
                line = ""
            line = f"{line}  {item}" if line else item
        if line and y + 20 <= bs - 8:
            text(canvas, line, (col_x + 8, y + 20), 0.45, GRAY)

    # -- route map
    mp = static["map"].copy()
    to_px = static["map_px"]
    driven = to_px(rd.ego_xy[: k + 1])
    if len(driven) > 1:
        polyline(mp, driven, WHITE, 2)
    e = to_px(rd.ego_xy[k : k + 1])[0]
    cv2.circle(mp, (int(e[0]), int(e[1])), 6, GREEN, -1, cv2.LINE_AA)
    if len(rd.expert_xy):
        tt = rec["time"] - rd.ticks[0]["time"]
        g = int(np.clip(np.searchsorted(rd.expert_t, tt), 0, len(rd.expert_xy) - 1))
        gp = to_px(rd.expert_xy[g : g + 1])[0]
        cv2.circle(mp, (int(gp[0]), int(gp[1])), 6, ORANGE, 2, cv2.LINE_AA)
    canvas[540:1080, 0:960] = mp

    # -- time series + score card
    ts = static["ts"].copy()
    x = static["ts_x"](rec["time"] - rd.ticks[0]["time"])
    y0, y1 = static["ts_y"]
    cv2.line(ts, (x, y0), (x, y1), (255, 255, 255), 1, cv2.LINE_AA)
    canvas[540:1080, 960 : 960 + ts.shape[1]] = ts
    canvas[540:1080, 960 + ts.shape[1] : W] = static["card"]
    for inf in rd.infractions:
        if inf["tick"] is not None and 0 <= k - inf["tick"] < 3 * TICK_HZ:
            text(canvas, f"INFRACTION: {inf['kind']}", (300, 510), 0.9, RED, 2)
    return canvas


# --------------------------------------------------------------------------- outputs


def summary_png(rd: RouteData, static: dict, path: Path) -> None:
    """The last frame's bottom half -- map, series, card -- as a still for the dashboard table."""
    img = np.full((540, W, 3), PANEL_BG, np.uint8)
    mp = static["map"].copy()
    to_px = static["map_px"]
    if len(rd.ego_xy) > 1:
        polyline(mp, to_px(rd.ego_xy), WHITE, 2)
    img[:, 0:960] = mp
    img[:, 960 : 960 + static["ts"].shape[1]] = static["ts"]
    img[:, 960 + static["ts"].shape[1] :] = static["card"]
    cv2.imwrite(str(path), img)


def route_summary(rd: RouteData) -> dict[str, Any]:
    """Per-route numbers the dashboard and the analysis read (route.json)."""
    sc = rd.scores()
    out: dict[str, Any] = {
        "route_id": rd.route_id,
        "model": rd.model,
        **{k: rd.meta.get(k) for k in ("town", "scenario", "abilities", "length_m")},
        **sc,
        "ticks": len(rd.ticks),
        "infractions": [
            {k: v for k, v in i.items() if k != "text"} | {"text": i["text"]} for i in rd.infractions
        ],
        "has_expert": bool(rd.expert),
    }
    if len(rd.speed):
        out["mean_speed"] = round(float(rd.speed.mean()), 2)
        out["max_speed"] = round(float(rd.speed.max()), 2)
        out["stopped_frac"] = round(float((rd.speed < 0.3).mean()), 3)
        out["max_route_dev"] = round(
            max(float((r.get("route") or {}).get("dev") or 0.0) for r in rd.ticks), 2
        )
        out["mean_infer_ms"] = round(float(np.mean([r.get("infer_ms") or 0 for r in rd.ticks])), 1)
    if rd.expert:
        out["expert_duration_s"] = rd.expert["duration_s"]
        out["expert_mean_speed"] = round(float(rd.expert_v.mean()), 2)
        if rd.route_len > 0 and len(rd.expert_s):
            out["expert_progress_pct"] = round(100 * float(np.nanmax(rd.expert_s)) / rd.route_len, 1)
    return out


def render_route(
    route_dir: Path,
    expert_root: Path | None,
    out_root: Path,
    meta: dict | None = None,
    every: int = 2,
    model: str | None = None,
) -> dict:
    rd = RouteData(route_dir, expert_root, meta, model)
    out = Path(out_root) / rd.route_id
    out.mkdir(parents=True, exist_ok=True)
    if not rd.ticks:
        raise RuntimeError(f"{route_dir}: no ticks.jsonl -- was the route recorded with record = true?")
    mp, to_px = route_map(rd, 960, 540)
    ts, ts_x, ts_y = timeseries(rd, 640, 540)
    static = {
        "map": mp,
        "map_px": to_px,
        "ts": ts,
        "ts_x": ts_x,
        "ts_y": ts_y,
        "card": score_card(rd, W - 960 - 640, 540),
    }
    summary_png(rd, static, out / "summary.png")

    ffmpeg = os.environ.get("PADDOCK_FFMPEG", "ffmpeg")
    tmp = out / "video.partial.mp4"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
        "-r", f"{TICK_HZ / every:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(tmp),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k in range(0, len(rd.ticks), every):
            proc.stdin.write(draw_frame(rd, k, static).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code} for {route_dir}")
    tmp.replace(out / "video.mp4")
    summary = route_summary(rd) | {"source": str(route_dir), "frames": len(range(0, len(rd.ticks), every))}
    (out / "route.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("route_dir", nargs="?", type=Path)
    ap.add_argument("--runs", type=Path, nargs="*", help="run directories, or directories of runs")
    ap.add_argument("--expert", type=Path)
    ap.add_argument("--routes-xml", type=Path, help="Bench2Drive routes file, for town/scenario/length")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-render routes whose video exists")
    ap.add_argument("--model", help="the driver's name on the frames (default: the run directory's model)")
    args = ap.parse_args(argv)
    table = runs.route_table(args.routes_xml) if args.routes_xml else {}
    if args.route_dir:
        m = runs._ROUTE_DIR.match(args.route_dir.name)
        print(
            json.dumps(
                render_route(
                    args.route_dir,
                    args.expert,
                    args.out,
                    table.get(m["rid"]) if m else None,
                    model=args.model,
                ),
                indent=1,
            )
        )
        return 0
    chosen = runs.latest(runs.attempts(args.runs or []))
    todo = [
        a
        for a in chosen.values()
        if a.result is not None and (args.force or not (args.out / a.route_id / "video.mp4").exists())
    ]
    print(f"{len(todo)} route(s) to render")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                render_route, a.path, args.expert, args.out, table.get(a.route_id), model=args.model
            ): a
            for a in todo
        }
        for fut in as_completed(futures):
            a = futures[fut]
            try:
                s = fut.result()
                print(f"  {a.route_id}: {s['frames']} frames, DS {s['ds']}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"  {a.route_id}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
