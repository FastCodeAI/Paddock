"""Bookkeeping and drawing for the AutoE2E x Bench2Drive deep-dive notebook.

The notebook runs on paddock's own kernel; anything that needs the network runs in
AutoE2E's interpreter, through ``autoe2e_bench2drive_probe.py`` beside this file
(:func:`run_probe`), as every node does -- paddock never imports a model. What the
probe keeps is cached under ``{runs}/bench2drive/_analysis/autoe2e/deep_dive/<route>/``,
so only a route's first run costs model time.

Frames: CARLA's world for maps (``route_plan.json``, the recorded ego and expert
poses); paddock's ego frame (x forward, y left, metres) for everything the model was
given or returned, and for the recorded actors. The model's navigation raster and
its BEV segmentation share one grid, ``NAV``: 450 rows from 120 m ahead to 60 m behind,
300 columns from 60 m left to 60 m right, 0.4 m a pixel (upstream's
``AUTOE2E_NAVIGATION_GEOMETRY``, mirrored in the adapter as ``NAV_*``).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from paddock.adapters.benchmarks.bench2drive import pid, recorder
from paddock.config import REPO_ROOT, SiteConfig
from paddock.policy.protocol import N_WAYPOINTS
from paddock.registry import Registry
from paddock.runtime import env as env_builder

# `tools` is a package at the repo root, which neither a notebook kernel (started in this
# directory) nor a script elsewhere has on its path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.benchmarks.bench2drive import route_video, runs  # noqa: E402

MODEL = "autoe2e"
BENCH = "bench2drive"
PROBE = Path(__file__).with_name("autoe2e_bench2drive_probe.py")
VIEWS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
SEG_CLASSES = (
    "drivable_area", "lane_boundary", "intersection", "crosswalk",
    "stop_line", "vehicle", "vulnerable_road_user", "other_obstacle",
)  # fmt: skip
NAV_H, NAV_W, NAV_MPP, NAV_X_MAX, NAV_Y_MAX = 450, 300, 0.4, 120.0, 60.0
#: imshow extent of a NAV grid, (left, right, bottom, top) in ego (y, x): left of the car on the left.
NAV_EXTENT = (NAV_Y_MAX, NAV_Y_MAX - NAV_W * NAV_MPP, NAV_X_MAX - NAV_H * NAV_MPP, NAV_X_MAX)
EGO_STEPS, EGO_DT, TICK_S = 64, 0.1, 0.05
#: Waypoints the node sends the agent: the wire contract's six (``protocol.N_WAYPOINTS``), 3 s of the plan.
N_SENT = N_WAYPOINTS
VARIANT_LABEL = {
    "live": "as fed in CARLA",
    "ego_zero_turn": "ego rows: no turning",
    "ego_steady": "ego rows: steady straight past",
    "ego_prefill_now": "ego rows: pre-start = today's speed",
    "ego_standstill": "ego rows: standing still",
    "cam_static": "camera history: no motion",
    "route_off": "no route raster",
    "route_dense": "route from the 1 m route",
    "refh_1p5": "reference 1.5 m up",
    "refh_0": "reference on the ground",
}
VARIANT_COLOR = {
    "live": "black", "ego_zero_turn": "tab:green", "ego_steady": "tab:blue", "ego_prefill_now": "tab:cyan",
    "ego_standstill": "tab:gray", "cam_static": "tab:red", "route_off": "tab:olive", "route_dense": "tab:purple",
    "refh_1p5": "tab:brown", "refh_0": "tab:pink",
}  # fmt: skip


# --------------------------------------------------------------------------- the routes


@dataclass
class Route:
    """One recorded drive: what the leaderboard scored and what every tick held."""

    dir: Path
    route_id: str = ""
    town: str = ""
    scenario: str = ""
    ticks: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    early_stop: dict = field(default_factory=dict)
    actor_meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, route_dir: Path) -> Route:
        d = Path(route_dir)
        rid, town, scenario = d.name.split("-", 2)
        meta = {}
        for line in (d / "actors.jsonl").read_text().splitlines() if (d / "actors.jsonl").exists() else []:
            a = json.loads(line)
            meta[a["id"]] = a
        return cls(
            dir=d, route_id=rid, town=town, scenario=scenario, ticks=runs.load_ticks(d),
            plan=runs.load_json(d / "route_plan.json") or {}, result=runs.load_result(d) or {},
            early_stop=runs.load_json(d / "early_stop.json") or {}, actor_meta=meta,
        )  # fmt: skip

    def by_tick(self, tick: int) -> dict:
        return next(r for r in self.ticks if r["t"] == tick)

    def series(self) -> pd.DataFrame:
        """One row per tick: time, speed, controls, lane offset, the model's own report."""
        rows = []
        t0 = self.ticks[0]["time"]
        for r in self.ticks:
            a = (r.get("extras") or {}).get("autoe2e") or {}
            lane = r.get("lane") or {}
            rows.append(
                {
                    "tick": r["t"], "t": r["time"] - t0, "speed": r["ego"]["speed"], "v_des": r["ctrl"].get("v_des"),
                    "steer": r["ctrl"]["steer"], "throttle": r["ctrl"]["throttle"], "brake": r["ctrl"]["brake"],
                    "infer_ms": r.get("infer_ms"), "forward": "held_s" not in a, "held_s": a.get("held_s"),
                    "model_s": a.get("model_s"), "history_repeated": a.get("history_repeated"),
                    "ego_rows_filled": a.get("ego_rows_filled"), "kappa_0": a.get("kappa_0"),
                    "accel_0": a.get("accel_0"), "lane_offset": lane.get("offset"), "lane_heading": lane.get("heading"),
                    "yaw_rate": r["ego"].get("yaw_rate"), "x": r["ego"]["x"], "y": r["ego"]["y"], "yaw": r["ego"]["yaw"],
                }
            )  # fmt: skip
        df = pd.DataFrame(rows)
        # Held ticks report no model fields: carry the last forward's forward (the plan they hold).
        for c in ("history_repeated", "ego_rows_filled", "kappa_0", "accel_0"):
            df[c] = df[c].ffill()
        return df

    def describe(self) -> dict:
        r, es = self.result, self.early_stop
        inf = {
            k: len(v) for k, v in (r.get("infractions") or {}).items() if v and k != "min_speed_infractions"
        }
        return {
            "route": self.route_id, "scenario": self.scenario, "town": self.town,
            "status": r.get("status"), "DS": (r.get("scores") or {}).get("score_composed"),
            "RC": (r.get("scores") or {}).get("score_route"), "ended": es.get("reason"),
            "at_s": es.get("game_time_s"), "infractions": inf, "ticks": len(self.ticks),
        }  # fmt: skip

    def focus_ticks(self) -> dict[str, int]:
        """Three moments worth taking apart, each on a forward tick (one the node ran the model on).

        ``settled``: 1.5 s in, the car rolling and nothing yet wrong; ``onset``: 0.5 s
        before the car first leaves its lane line by a metre or turns 8 degrees off it
        (or 1 s before the end, if it never does); ``late``: the last forward tick, a
        few ticks before the route ended.
        """
        df = self.series()
        fwd = df[df["forward"]]

        def snap(t):
            return int(fwd.iloc[int(np.argmin(np.abs(fwd["t"].to_numpy() - t)))]["tick"])

        off = df[(df["lane_offset"].abs() > 1.0) | (df["lane_heading"].abs() > 8.0)]
        onset_t = (float(off["t"].iloc[0]) - 0.5) if len(off) else float(df["t"].iloc[-1]) - 1.0
        return {
            "settled": snap(1.5),
            "onset": snap(max(onset_t, 0.2)),
            "late": snap(float(df["t"].iloc[-1]) - 0.3),
        }


def site(config: str | None = None) -> SiteConfig:
    """The site config the deep-dive drives were recorded under (``PADDOCK_CONFIG`` by default)."""
    if config:
        import os

        os.environ["PADDOCK_CONFIG"] = config
    return SiteConfig.load()


def recorded_routes(cfg: SiteConfig, route_ids: list[str]) -> dict[str, Route]:
    """The latest attempt of each route that was driven with its inputs recorded."""
    root = cfg.runs / BENCH / MODEL
    found = {}
    for att in sorted(runs.attempts([root]), key=lambda a: a.run_id):
        if att.route_id in route_ids and (att.path / recorder.INPUTS_DIR).is_dir() and att.result is not None:
            found[att.route_id] = att.path
    missing = [r for r in route_ids if r not in found]
    if missing:
        raise SystemExit(
            f"no recorded drive with inputs under {root} for {missing}: drive them with record_inputs"
        )
    return {rid: Route.load(found[rid]) for rid in route_ids}


def cache_dir(cfg: SiteConfig, route: Route) -> Path:
    return cfg.runs / BENCH / "_analysis" / MODEL / "deep_dive" / f"{route.route_id}-{route.dir.parent.name}"


def run_probe(
    cfg: SiteConfig, route: Route, focus: list[int], every: int = 4, recompute: bool = False
) -> dict:
    """The probe's ``series.json`` for ``route``, running it in AutoE2E's interpreter if not cached."""
    out = cache_dir(cfg, route)
    done = out / "series.json"
    if done.exists() and not recompute:
        cached = json.loads(done.read_text())
        if {int(t) for t in cached["focus"]} >= set(focus):
            return cached
    out.mkdir(parents=True, exist_ok=True)
    spec = Registry.load().models[MODEL]
    paired = cfg.paired(MODEL, BENCH)
    # As the node ran, but a forward whenever the probe asks, and no per-call log of its own.
    env = env_builder.build(spec, paired, extra={"AUTOE2E_REPLAN_S": "0"})
    env.pop("AUTOE2E_RECORD_DIR", None)
    python = spec.runtime.interpreter(paired, **spec.context(paired))
    cmd = [str(python), "-u", str(PROBE), "--route", str(route.dir), "--out", str(out)]
    cmd += ["--focus", ",".join(str(t) for t in sorted(focus)), "--series-every", str(every)]
    with (out / "probe.log").open("w") as log:
        subprocess.run(cmd, env=env, cwd=env_builder.working_dir(spec, paired), stdout=log, stderr=subprocess.STDOUT, check=True)  # fmt: skip
    return json.loads(done.read_text())


def focus_data(cfg: SiteConfig, route: Route, tick: int) -> dict:
    with np.load(cache_dir(cfg, route) / f"focus_{tick:05d}.npz") as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- geometry helpers


def plan_rollout(controls: np.ndarray, v0: float) -> np.ndarray:
    """The adapter's own integration (``policy.integrate_controls``), restated in numpy for drawing.

    dt 0.1, speed first and clamped at zero, heading from the new speed: [64, 3] (x, y, heading).
    """
    v, out, speed, heading, x, y = float(v0), np.zeros((EGO_STEPS, 3)), np.zeros(EGO_STEPS), 0.0, 0.0, 0.0
    for k in range(EGO_STEPS):
        v = max(v + float(controls[k, 0]) * EGO_DT, 0.0)
        speed[k] = v
    heading_k = np.cumsum(speed * controls[:, 1] * EGO_DT)
    out[:, 0] = np.cumsum(speed * np.cos(heading_k) * EGO_DT)
    out[:, 1] = np.cumsum(speed * np.sin(heading_k) * EGO_DT)
    out[:, 2] = heading_k
    del heading, x, y
    return out


def actor_boxes(route: Route, tick: int) -> list[dict]:
    """The recorded actors at ``tick`` in the ego frame, each with its class and box corners."""
    rec = route.by_tick(tick)
    boxes = []
    for aid, ax, ay, ayaw, _v in rec.get("actors") or []:
        meta = route.actor_meta.get(aid, {})
        typ = meta.get("type", "")
        hl, hw = (meta.get("ext") or [2.2, 1.0])[:2]
        a = math.radians(ayaw)
        fwd, left = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        c = np.array([ax, ay])
        corners = np.array(
            [c + fx * hl * fwd + lx * hw * left for fx, lx in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        )
        kind = "walker" if typ.startswith("walker") else "static" if typ.startswith("static") else "vehicle"
        boxes.append({"id": aid, "type": typ, "kind": kind, "xy": c, "corners": corners})
    return boxes


def dense_route_ego(route: Route, tick: int, behind: float = 10.0, ahead: float = 120.0) -> np.ndarray:
    """The 1 m route near the ego at ``tick``, ego frame."""
    rec = route.by_tick(tick)
    e = rec["ego"]
    xy = np.asarray(route.plan.get("xy") or [], dtype=float).reshape(-1, 2)
    ego = recorder.to_ego(xy, e["x"], e["y"], e["yaw"])
    s = np.asarray(route.plan.get("s") or np.zeros(len(xy)))
    i = int(np.argmin(np.hypot(ego[:, 0], ego[:, 1])))
    keep = (s >= s[i] - behind) & (s <= s[i] + ahead)
    return ego[keep]


def nav_axes(ax, title: str = "", xlim=(-30, 60), ylim=(25, -25)) -> None:
    """A BEV axes in the NAV convention: forward up, left to the left; metres."""
    ax.set_xlim(*ylim)
    ax.set_ylim(*xlim)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward)")
    if title:
        ax.set_title(title, fontsize=10)


def draw_scene(ax, route: Route, tick: int, sent_route=True, dense_route=True, actors=True) -> None:
    """The privileged truth at ``tick`` on a NAV axes: actors, the 1 m route, the route the model was sent."""
    if dense_route:
        d = dense_route_ego(route, tick)
        ax.plot(d[:, 1], d[:, 0], color="dimgray", lw=1.0, ls="--", label="route (1 m, truth)")
    if sent_route:
        sr = np.asarray(route.by_tick(tick)["sent"].get("route_xy") or [], dtype=float).reshape(-1, 2)
        if len(sr):
            ax.plot(sr[:, 1], sr[:, 0], "o-", color="deepskyblue", lw=1.0, ms=3, label="route sent (sparse)")
    if actors:
        colors = {"vehicle": "tab:cyan", "walker": "magenta", "static": "dimgray"}
        for b in actor_boxes(route, tick):
            ax.add_patch(
                Polygon(b["corners"][:, ::-1], closed=True, fill=False, ec=colors[b["kind"]], lw=1.2)
            )
    ax.add_patch(Rectangle((-1.0, -2.4), 2.0, 4.8, fill=True, color="gold", alpha=0.8))


# --------------------------------------------------------------------------- figures


def overview(route: Route, focus: dict[str, int], expert_root: Path | None = None) -> plt.Figure:
    """The route as driven, the route itself, the expert's drive, and the front camera at each focus tick."""
    df = route.series()
    fig = plt.figure(figsize=(16, 5.5))
    ax = fig.add_axes((0.03, 0.08, 0.3, 0.84))
    xy = np.asarray(route.plan.get("xy") or [], dtype=float).reshape(-1, 2)
    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=4, alpha=0.3, label="route")
    ax.plot(df["x"], df["y"], color="black", lw=1.5, label="AutoE2E (driven)")
    if expert_root is not None:
        from tools.benchmarks.bench2drive import expert as expert_mod

        ex = expert_mod.load_expert(expert_root, route.route_id)
        if ex:
            exy = np.array([[f["x"], f["y"]] for f in ex["frames"]])
            ax.plot(exy[:, 0], exy[:, 1], color="tab:orange", lw=1.2, ls="--", label="expert (Think2Drive)")
    for name, t in focus.items():
        r = df[df["tick"] == t].iloc[0]
        ax.plot(r["x"], r["y"], "o", ms=7, label=f"{name} (t={r['t']:.2f} s)")
    ax.plot(df["x"].iloc[-1], df["y"].iloc[-1], "x", color="red", ms=12, mew=3, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # CARLA's y points down the page seen from above
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{route.route_id} {route.scenario}: world, metres", fontsize=10)
    for k, (name, t) in enumerate(focus.items()):
        img = plt.imread(route.dir / "front" / f"{t:05d}.jpg")
        a = fig.add_axes((0.36 + k * 0.215, 0.1, 0.205, 0.8))
        a.imshow(img)
        r = df[df["tick"] == t].iloc[0]
        a.set_title(
            f"{name}: t={r['t']:.2f} s, {r['speed']:.1f} m/s, lane offset {r['lane_offset']:+.2f} m",
            fontsize=8,
        )
        a.axis("off")
    return fig


def tick_flow_diagram(route: Route) -> plt.Figure:
    """One 0.05 s tick, CARLA to model and back, with this route's measured costs."""
    df = route.series()
    fw, hd = df[df["forward"]], df[~df["forward"]]
    model_s = float(np.nanmedian(fw["model_s"])) if len(fw) else float("nan")
    infer_fw = float(np.nanmedian(fw["infer_ms"])) / 1000 if len(fw) else float("nan")
    infer_hd = float(np.nanmedian(hd["infer_ms"])) / 1000 if len(hd) else float("nan")
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.axis("off")
    boxes = [
        (0.0, "CARLA world tick\nsynchronous, 0.05 s of game\ntime; nothing moves\nuntil the agent returns"),
        (1.0, "sensors\n6 cameras 800x450\n+ front/BEV recording\n+ speedometer"),
        (2.0, "agent.run_step\nego pose, speed, route\n(50 m plan points),\ncommand, calibration"),
        (3.0, "POST /infer\n6 raw frames, ~8.6 MB\nJSON, to the node"),
        (4.0, f"node: _push(frame)\nforward if 0.1 s since the last\n(else hold the last plan,\nre-expressed: {infer_hd:.2f} s)"),
        (5.0, f"AutoE2E forward\n8 frames x 6 views -> BEV,\nGRU -> 64 x (accel, curvature)\nmodel {model_s:.2f} s, /infer {infer_fw:.2f} s"),
        (6.0, "rollout\n64 x 0.1 s integrated from\nv0 = history speed;\n12 waypoints at 0.5 s"),
        (7.0, "WaypointTracker (PID)\naim at the waypoint ~4 m out;\ntarget speed = mean spacing\nover 4 s / 0.5 s"),
        (8.0, "VehicleControl\nsteer, throttle, brake\napplied on the next\nCARLA tick"),
    ]  # fmt: skip
    for x, label in boxes:
        ax.add_patch(Rectangle((x * 1.6, 0.35), 1.45, 0.55, fc="#eef3fb", ec="#4a6fa5"))
        ax.text(x * 1.6 + 0.725, 0.625, label, ha="center", va="center", fontsize=7.5)
        if x < 8:
            ax.add_patch(
                FancyArrowPatch(
                    (x * 1.6 + 1.45, 0.625), (x * 1.6 + 1.6, 0.625), arrowstyle="->", mutation_scale=12
                )
            )
    ax.add_patch(FancyArrowPatch((8 * 1.6 + 0.7, 0.35), (0.7, 0.35), connectionstyle="arc3,rad=0.25", arrowstyle="->", mutation_scale=12, color="#b04040"))  # fmt: skip
    ax.text(7.2, 0.02, "next tick: the control moves the car; the new pose and frames go into the next request,\nand into the model's own 6.4 s ego history and 3.5 s camera history", ha="center", fontsize=8, color="#b04040")  # fmt: skip
    ax.set_xlim(-0.1, 9 * 1.6)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Route {route.route_id}: one tick of the handover between CARLA and the model", fontsize=11)
    return fig


def handover_timeline(route: Route, events: dict[str, float] | None = None) -> plt.Figure:
    """Tick by tick: which ticks ran the model, how the car moved, and how much real past the model had."""
    df = route.series()
    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    ax = axes[0]
    fw = df[df["forward"]]
    ax.plot(df["t"], df["infer_ms"] / 1000, ".", ms=3, color="gray", label="/infer round trip (s, wall)")
    ax.plot(fw["t"], fw["model_s"], "o", ms=3, color="tab:red", label="model forward (s, wall)")
    ax.set_ylabel("wall s")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "1  the model runs on every second tick (0.1 s); the tick between gets the held plan", fontsize=10
    )
    ax = axes[1]
    ax.plot(df["t"], df["speed"], color="black", label="speed (m/s)")
    ax.plot(df["t"], df["v_des"], color="tab:green", lw=1, label="tracker's target speed from the plan")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("2  speed: what the plan asked for and what the car did", fontsize=10)
    ax = axes[2]
    ax.plot(df["t"], df["steer"], color="tab:blue", label="steer (+ right)")
    ax.plot(df["t"], df["throttle"], color="tab:green", label="throttle")
    ax.plot(df["t"], df["brake"], color="tab:red", label="brake")
    ax.set_ylabel("control")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("3  what the tracker sent CARLA", fontsize=10)
    ax = axes[3]
    ax.plot(df["t"], df["lane_offset"], color="black", label="lane offset (m, left +)")
    ax2 = ax.twinx()
    ax2.plot(df["t"], -df["lane_heading"], color="tab:purple", lw=1, label="heading vs lane (deg, left +)")
    ax2.plot(
        df["t"], 100 * df["kappa_0"], color="tab:orange", lw=1, label="plan's first curvature x100 (left +)"
    )
    ax.set_ylabel("m")
    ax2.set_ylabel("deg / 100/m")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title("4  where the car is in its lane, and which way the plan turns", fontsize=10)
    ax = axes[4]
    ax.plot(df["t"], df["history_repeated"], color="tab:red", label="camera history: frames repeated (of 7)")
    ax3 = ax.twinx()
    ax3.plot(
        df["t"], df["ego_rows_filled"], color="tab:blue", label="ego history: rows before the start (of 64)"
    )
    ax.set_ylabel("frames")
    ax3.set_ylabel("rows")
    ax.set_xlabel("game time since the route started (s)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title(
        "5  how much of the model's past is real: camera history fills by 3.5 s, ego history by 6.4 s",
        fontsize=10,
    )
    for name, t in (events or {}).items():
        for a in axes:
            a.axvline(t, color="gray", ls=":", lw=1)
        axes[0].text(
            t,
            axes[0].get_ylim()[1],
            f" {name}",
            fontsize=7,
            va="top",
            ha="right",
            rotation=90,
            color="dimgray",
        )
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def cameras(route: Route, tick: int, focus: dict) -> plt.Figure:
    """The six frames CARLA rendered, and the six tiles (plus the 1024 front) the network got."""
    images, _ = recorder.read_inputs(str(route.dir / recorder.INPUTS_DIR / f"{tick:05d}.npz"))
    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), gridspec_kw={"height_ratios": [0.56, 1, 1]})
    for k, v in enumerate(VIEWS):
        axes[0, k].imshow(images[v])
        axes[0, k].set_title(f"{v}\nas rendered, {images[v].shape[1]}x{images[v].shape[0]}", fontsize=8)
        axes[1, k].imshow(focus["tiles"][k])
        axes[1, k].set_title("network tile, 512x512", fontsize=8)
    axes[2, 0].imshow(focus["front_1024"])
    axes[2, 0].set_title("front at 1024x1024", fontsize=8)
    for a in axes.flat:
        a.axis("off")
    for k in range(1, 6):
        axes[2, k].remove()
    fig.suptitle(
        f"Route {route.route_id}, tick {tick}: what the model was shown (top: rendered; middle: network tiles)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def history(route: Route, tick: int, focus: dict) -> plt.Figure:
    """The seven history frames, oldest first, with their age and whether each is a stand-in."""
    th, dt, gap = focus["history_thumbs"], focus["history_dt_s"], focus["history_gap_s"]
    fig, axes = plt.subplots(6, 8, figsize=(17, 13))
    cur = [focus["tiles"][k] for k in range(6)]
    for j in range(8):
        for k in range(6):
            a = axes[k, j]
            if j < 7:
                a.imshow(th[j, k])
                if k == 0:
                    rep = np.isnan(gap[j])
                    a.set_title(
                        f"{dt[j]:+.2f} s" + ("  REPEATED" if rep else ""),
                        fontsize=9,
                        color="red" if rep else "black",
                    )
            else:
                a.imshow(cur[k])
                if k == 0:
                    a.set_title("now", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
            if j == 0:
                a.set_ylabel(VIEWS[k].replace("CAM_", ""), fontsize=8)
    fig.suptitle(
        f"Route {route.route_id}, tick {tick}: the camera history, t-3.5 .. t-0.5 s and now. A frame older than the route "
        "is a stand-in: the oldest real frame, repeated with its own pose",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def geometry(route: Route, tick: int, focus: dict) -> plt.Figure:
    """Where each camera looks, and which ground cells of the model's grid any camera sees."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5))
    c2e, K, sizes = focus["camera2ego"], focus["camera2image"], focus["sizes"]
    colors = plt.cm.tab10(np.arange(6))
    for k, v in enumerate(VIEWS):
        pos = c2e[k][:3, 3]
        axis = c2e[k][:3, 2]  # optical z, the viewing direction, in the ego frame
        hfov = 2 * math.atan(sizes[k][0] / (2 * K[k][0, 0]))
        yaw = math.atan2(axis[1], axis[0])
        for s in (-1, 1):
            d = np.array([math.cos(yaw + s * hfov / 2), math.sin(yaw + s * hfov / 2)]) * 40
            a1.plot([pos[1], pos[1] + d[1]], [pos[0], pos[0] + d[0]], color=colors[k], lw=1)
        a1.annotate(f"{v.replace('CAM_', '')}\n{math.degrees(hfov):.0f} deg", (pos[1] + 20 * math.sin(yaw), pos[0] + 20 * math.cos(yaw)), fontsize=7, color=colors[k], ha="center")  # fmt: skip
    draw_scene(a1, route, tick)
    nav_axes(a1, "camera footprints (40 m) and the scene", xlim=(-45, 45), ylim=(45, -45))
    # Ground cells (z = 0 in the ego frame, z = -ref_height in the reference frame) each camera projects in.
    rows, cols = np.meshgrid(np.arange(NAV_H), np.arange(NAV_W), indexing="ij")
    gx = NAV_X_MAX - (rows + 0.5) * NAV_MPP
    gy = NAV_Y_MAX - (cols + 0.5) * NAV_MPP
    pts = np.stack([gx, gy, np.full_like(gx, -float(focus["ref_height"])), np.ones_like(gx)], -1).reshape(
        -1, 4
    )
    seen = np.zeros(len(pts))
    for k in range(6):
        p = pts @ focus["proj_now"][k].T
        w = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = p[:, 0] / w, p[:, 1] / w
        seen += (w > 0.1) & (u >= 0) & (u < 512) & (v >= 0) & (v < 512)
    a2.imshow(seen.reshape(NAV_H, NAV_W), extent=NAV_EXTENT, cmap="viridis", vmin=0, vmax=3)
    draw_scene(a2, route, tick)
    nav_axes(a2, "how many cameras see each ground cell of the model's grid", xlim=(-60, 120), ylim=(60, -60))
    a2.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def ego_rows(route: Route, tick: int, focus: dict) -> plt.Figure:
    """The 64 rows of ego history, t-6.4 .. t-0.1 s, as the network got them."""
    rows = focus["ego_rows"]
    t = -EGO_DT * np.arange(EGO_STEPS, 0, -1)
    rec = route.by_tick(tick)
    filled = int((rec.get("extras") or {}).get("autoe2e", {}).get("ego_rows_filled") or 0)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    for k, (name, unit) in enumerate(
        (
            ("speed", "m/s"),
            ("acceleration", "m/s2"),
            ("yaw rate", "rad/s, left +"),
            ("curvature", "1/m, left +"),
        )
    ):
        axes[k].plot(t, rows[:, k], ".-", ms=3)
        if filled:
            axes[k].axvspan(
                t[0] - 0.05,
                t[filled - 1] + 0.05,
                color="tab:blue",
                alpha=0.12,
                label=f"{filled} rows before the start: filled",
            )
        axes[k].set_title(f"{name} ({unit})", fontsize=9)
        axes[k].set_xlabel("s before now")
        axes[k].grid(alpha=0.3)
    axes[1].annotate("newest acceleration\nmasked to 0", (t[-1], rows[-1, 1]), fontsize=7, xytext=(-80, 15), textcoords="offset points", arrowprops={"arrowstyle": "->"})  # fmt: skip
    if filled:
        axes[0].legend(fontsize=7)
    fig.suptitle(f"Route {route.route_id}, tick {tick}: the ego history the network got", fontsize=10)
    fig.tight_layout()
    return fig


def route_raster(route: Route, tick: int, focus: dict) -> plt.Figure:
    """The navigation input: the route corridor and destination, on the model's grid, over the truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    rr = focus["route_raster"]
    for k, (a, name) in enumerate(zip(axes, ("corridor (1.75 m either side of the sent route)", "destination disk"), strict=True)):  # fmt: skip
        a.imshow(rr[k], extent=NAV_EXTENT, cmap="Greys", vmin=0, vmax=1.5, alpha=0.9)
        draw_scene(a, route, tick)
        nav_axes(a, name, xlim=(-20, 100), ylim=(40, -40))
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"Route {route.route_id}, tick {tick}: the route the model was given (no map: map_valid false)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def outputs(route: Route, tick: int, focus: dict, variants=("live",)) -> plt.Figure:
    """What the network returned, how the adapter turns it into waypoints, and how the agent reads them."""
    v0 = float(focus["v0"])
    fig = plt.figure(figsize=(17, 6.5))
    a_acc, a_kap = fig.add_axes((0.04, 0.56, 0.26, 0.38)), fig.add_axes((0.04, 0.08, 0.26, 0.38))
    a_bev, a_cam = fig.add_axes((0.33, 0.06, 0.27, 0.9)), fig.add_axes((0.63, 0.25, 0.36, 0.6))
    t = EGO_DT * np.arange(1, EGO_STEPS + 1)
    rec = route.by_tick(tick)
    for name in variants:
        c = focus["controls_" + name]
        col = VARIANT_COLOR.get(name, "black")
        a_acc.plot(
            t, c[:, 0], color=col, lw=1.5 if name == "live" else 1, label=VARIANT_LABEL.get(name, name)
        )
        a_kap.plot(t, c[:, 1], color=col, lw=1.5 if name == "live" else 1)
        roll = plan_rollout(c, v0)
        a_bev.plot(roll[:, 1], roll[:, 0], color=col, lw=2 if name == "live" else 1)
        if name == "live":
            # The adapter samples 12 waypoints to 6 s; the wire format sends the first N_WAYPOINTS (6, to 3 s).
            wps = roll[4::5][:N_SENT]
            a_bev.plot(
                wps[:, 1],
                wps[:, 0],
                "o",
                color=col,
                ms=4,
                label=f"the {N_SENT} waypoints sent (0.5 s .. 3 s)",
            )
            # The tracker's reading of the plan: aim point and target speed (pid.WaypointTracker, stateless parts).
            w = wps[: int(round(pid.CONTROL_HORIZON_S / 0.5)), :2]
            mid = np.linalg.norm((w[1:] + w[:-1]) / 2, axis=1)
            aim = w[int(np.argmin(np.abs(mid - pid.AIM_DIST)))]
            v_des = float(np.linalg.norm(np.diff(w, axis=0), axis=1).mean() / 0.5)
            a_bev.plot(
                aim[1],
                aim[0],
                "*",
                ms=14,
                color="tab:red",
                label=f"aim point; target {v_des:.1f} m/s (car {v0:.1f})",
            )
            proj = route_video.front_project(wps[:, :2])
            img = plt.imread(route.dir / "front" / f"{tick:05d}.jpg")
            a_cam.imshow(img)
            ok = np.isfinite(proj).all(1)
            a_cam.plot(proj[ok, 0], proj[ok, 1], "o-", color="lime", ms=4)
    a_acc.set_title("acceleration (m/s2), 64 steps of 0.1 s", fontsize=9)
    a_kap.set_title("curvature (1/m, left +)", fontsize=9)
    a_kap.set_xlabel("s ahead")
    a_acc.legend(fontsize=7)
    for a in (a_acc, a_kap):
        a.grid(alpha=0.3)
    draw_scene(a_bev, route, tick)
    nav_axes(a_bev, "the plan, integrated (0.1 s rollout) over the truth", xlim=(-10, 70), ylim=(25, -25))
    a_bev.legend(fontsize=7, loc="lower left")
    ctrl = rec["ctrl"]
    a_cam.set_title(f"waypoints on the front camera; the car sent steer {ctrl['steer']:+.2f}, throttle {ctrl['throttle']:.2f}, brake {ctrl['brake']:.0f}", fontsize=9)  # fmt: skip
    a_cam.axis("off")
    fig.suptitle(
        f"Route {route.route_id}, tick {tick}: the network's output and how it becomes a control",
        fontsize=10,
        y=1.04,
    )
    return fig


def segmentation(route: Route, tick: int, focus: dict, classes=SEG_CLASSES) -> plt.Figure:
    """The BEV segmentation head's 8 class maps (sigmoid) over the truth: what the frozen camera BEV encodes."""
    seg = focus["seg"].astype(np.float32)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))
    for k, name in enumerate(classes):
        a = axes.flat[k]
        a.imshow(seg[SEG_CLASSES.index(name)], extent=NAV_EXTENT, cmap="magma", vmin=0, vmax=1)
        draw_scene(a, route, tick, sent_route=False)
        nav_axes(a, name, xlim=(-30, 70), ylim=(40, -40))
    fig.suptitle(
        f"Route {route.route_id}, tick {tick}: what the model's BEV says is where (truth: boxes, dashed route)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def seen_actors(route: Route, tick: int, focus: dict, max_dist: float = 50.0) -> pd.DataFrame:
    """For each recorded actor in view of the grid: the BEV's vehicle / VRU / obstacle score inside its box."""
    seg = focus["seg"].astype(np.float32)
    rows, cols = np.meshgrid(np.arange(NAV_H), np.arange(NAV_W), indexing="ij")
    gx = NAV_X_MAX - (rows + 0.5) * NAV_MPP
    gy = NAV_Y_MAX - (cols + 0.5) * NAV_MPP
    from matplotlib.path import Path as MplPath

    out = []
    for b in actor_boxes(route, tick):
        d = float(np.hypot(*b["xy"]))
        if d > max_dist:
            continue
        inside = (
            MplPath(b["corners"])
            .contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
            .reshape(NAV_H, NAV_W)
        )
        if not inside.any():
            continue
        out.append(
            {
                "actor": b["type"], "x_m": round(float(b["xy"][0]), 1), "y_m": round(float(b["xy"][1]), 1),
                "dist_m": round(d, 1), "vehicle": round(float(seg[5][inside].max()), 2),
                "vru": round(float(seg[6][inside].max()), 2), "other_obstacle": round(float(seg[7][inside].max()), 2),
                "drivable": round(float(seg[0][inside].mean()), 2),
            }
        )  # fmt: skip
    return pd.DataFrame(out).sort_values("dist_m") if out else pd.DataFrame()


def variants_table(series: dict, tick: int) -> pd.DataFrame:
    """Every counterfactual at a focus tick: the plan's first curvature, lateral offset, heading, speed."""
    f = series["focus"][str(tick)]
    rows = []
    for name, nums in f.items():
        if isinstance(nums, dict):
            rows.append({"variant": VARIANT_LABEL.get(name, name), **nums})
    return pd.DataFrame(rows).set_index("variant")


def series_plot(route: Route, series: dict, events: dict[str, float] | None = None) -> plt.Figure:
    """Along the drive: the plan's turn, as fed and under each counterfactual, beside what the car did."""
    df = route.series()
    t0 = route.ticks[0]["time"]
    s = series["series"]
    t = np.array([r["t_s"] for r in s]) - t0
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for name in ("live", "ego_zero_turn", "ego_steady", "cam_static", "route_dense"):
        if name not in s[0]:
            continue
        kw = {"color": VARIANT_COLOR[name], "lw": 2 if name == "live" else 1.2, "label": VARIANT_LABEL[name]}
        axes[0].plot(t, [r[name]["kappa_1s"] for r in s], ".-", **kw)
        axes[1].plot(t, [r[name]["y_2s"] for r in s], ".-", **kw)
    axes[0].set_ylabel("1/m")
    axes[0].set_title("plan's mean curvature over its first second (left +)", fontsize=10)
    axes[1].set_ylabel("m")
    axes[1].set_title("plan's lateral offset 2 s ahead (left +)", fontsize=10)
    axes[0].legend(fontsize=8, ncol=3)
    axes[2].plot(df["t"], df["lane_offset"], color="black", label="lane offset (m, left +)")
    axes[2].plot(
        df["t"], -df["lane_heading"] / 10, color="tab:purple", label="heading vs lane (deg/10, left +)"
    )
    axes[2].plot(df["t"], df["speed"] / 5, color="tab:green", label="speed (m/s / 5)")
    axes[2].plot(
        df["t"], df["history_repeated"] / 7, color="tab:red", lw=1, label="camera history repeated (/7)"
    )
    axes[2].set_xlabel("game time since the route started (s)")
    axes[2].legend(fontsize=8, ncol=4)
    axes[2].set_title("what the car did", fontsize=10)
    for name, tt in (events or {}).items():
        for a in axes:
            a.axvline(tt, color="gray", ls=":", lw=1)
        axes[0].text(
            tt,
            axes[0].get_ylim()[1],
            f" {name}",
            fontsize=7,
            va="top",
            ha="right",
            rotation=90,
            color="dimgray",
        )
    for a in axes:
        a.grid(alpha=0.3)
        a.axhline(0, color="gray", lw=0.6)
    fig.tight_layout()
    return fig


def replay_fidelity(series: dict) -> pd.DataFrame:
    """How exactly the offline replay reproduced the plans the node returned in CARLA."""
    rows = [
        {"tick": r["tick"], "max |waypoint diff| (m)": r.get("replay_vs_carla_m")} for r in series["series"]
    ]
    return pd.DataFrame(rows).describe().T


def events(route: Route) -> dict[str, float]:
    """The moments the timelines mark, read off the recording (game seconds since the start)."""
    df = route.series()
    out = {}
    real_cam = df[df["history_repeated"] == 0]
    if len(real_cam):
        out["camera history all real"] = float(real_cam["t"].iloc[0])
    real_ego = df[df["ego_rows_filled"] == 0]
    if len(real_ego):
        out["ego history all real"] = float(real_ego["t"].iloc[0])
    off = df[(df["lane_offset"].abs() > 1.0) | (df["lane_heading"].abs() > 8.0)]
    if len(off):
        out["leaves the lane line"] = float(off["t"].iloc[0])
    out[f"end ({route.early_stop.get('reason') or route.result.get('status')})"] = float(df["t"].iloc[-1])
    return out
