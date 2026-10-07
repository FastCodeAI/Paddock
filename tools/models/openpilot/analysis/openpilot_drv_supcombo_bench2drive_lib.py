"""Why supercombo scores what it scores on Bench2Drive: per-route features, events, and a primary cause.

    python -m tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_lib \\
        --runs <runs dir> --expert <expert dir> --routes-xml <bench2drive220.xml> --out <analysis dir>

Reads, for the attempt that counts at each route (``runs.latest``), what the
agent recorded -- ``ticks.jsonl`` (every node reply beside privileged world
state), ``route_plan.json``, ``actors.jsonl``, ``results.json`` -- and
Think2Drive's drive of the same route (``expert.py``), and writes

    route_features.csv    one row per route: the leaderboard's numbers, features, ``cause`` and ``patterns``
    patterns.csv          every failure pattern that holds for a route, with the evidence it held on
    pattern_catalogue.csv the patterns themselves: name, family, priority, what they mean
    standstills.csv       every stretch the car stood still: what was ahead when it stopped and when it
                          moved (or the route ended), the plan it held, the wait after green
    lights.csv            every traffic light on the route the car reached: its state as the car crossed
    turns.csv             every junction manoeuvre on the route: taken, or where the car left the route
    collisions.csv        every collision: speed, the other actor, what the model had seen of it

The question is not "how well" -- the leaderboard answers that -- but *why*,
and the per-tick record makes most of the answers mechanical:

* **standstill** -- supercombo has no speed input and infers motion from frame
  to frame; standing still, its plan collapses to centimetres. A stretch is
  ``v < 0.3 m/s`` for at least a second, described by what was ahead when it
  began and when it ended (``Signals.ahead``): a red light on the route, an
  actor in the ego's lane (privileged state, typed), the model's own lead, or
  nothing. A standstill the route ended in, with a clear lane at the end, is the
  trap; one with a stopped car or cones still ahead is a missing overtake.
* **patterns** -- ``PATTERNS`` is the catalogue of failure (and success) modes,
  each a named test over the route that returns its evidence. Every pattern that
  holds is recorded; the one with the lowest ``primary`` is the route's
  ``cause``. Adding a mode is adding one ``Pattern`` -- see
  docs/design/driving-failure-analysis.md.

Frames: ``ticks.jsonl`` actors are in paddock's ego frame (x forward, y left);
world positions are CARLA's (see ``recorder.py``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.benchmarks.bench2drive import expert as expert_mod  # noqa: E402
from tools.benchmarks.bench2drive import runs  # noqa: E402

TICK_HZ = 20.0
STILL_V = 0.3  # m/s: standing still
MOVED_V = 1.0  # m/s: has pulled away
MIN_STILL_S = 1.0
LIGHT_NEAR_M = 30.0  # a red light on the route this close explains a stop
AHEAD_M = 40.0  # a stationary vehicle / prop / pedestrian in the lane this close explains a stop
AHEAD_HALF_M = 2.5  # "in the lane": half a lane plus a cone at its edge
MOVING_V = 0.5  # m/s: an actor faster than this is moving
MOVING_NEAR_M = 15.0  # a moving vehicle this close is a lead to queue behind, further away it is traffic
SLOW_GREEN_S = 5.0  # standing longer than this after green is a slow restart

_TYPE = re.compile(r"type=([\w.]+)")


# --------------------------------------------------------------------------- per-tick signals


def actor_kind(meta: dict, speed: float) -> str:
    """What an actor is, for a reader: pedestrian / prop / moving vehicle / stationary vehicle.

    Props are cones, warning signs and the parked cars Bench2Drive spawns as static meshes.
    """
    typ = (meta or {}).get("type", "")
    if typ.startswith("walker"):
        return "pedestrian"
    if typ.startswith("static"):
        return "prop"
    return "moving vehicle" if speed > MOVING_V else "stationary vehicle"


@dataclass
class Signals:
    """``ticks.jsonl`` as arrays, one entry per tick."""

    t: np.ndarray
    v: np.ndarray
    v_des: np.ndarray
    brake: np.ndarray
    throttle: np.ndarray
    steer: np.ndarray
    s: np.ndarray
    frac: np.ndarray
    dev: np.ndarray
    x: np.ndarray
    y: np.ndarray
    plan_end_x: np.ndarray
    plan_end_y: np.ndarray
    lead_prob: np.ndarray
    lead_d: np.ndarray
    lead_vrel: np.ndarray
    light_d: np.ndarray
    light_state: list
    junction: np.ndarray
    lane: list
    lane_offset: np.ndarray
    ahead_d: np.ndarray  # nearest actor in the ego's lane ahead within AHEAD_M (privileged), metres
    ahead_v: np.ndarray
    ahead_kind: list  # actor_kind() of that actor
    lane_probs: np.ndarray  # the model's two ego lane lines, mean probability
    desire_turn: np.ndarray  # the model's own turnLeft + turnRight desire state
    infer_ms: np.ndarray
    movers_ahead: np.ndarray  # actors in the forward half-plane within 60 m moving > 1 m/s

    @classmethod
    def from_ticks(cls, ticks: list[dict], actor_meta: dict | None = None) -> Signals:
        n = len(ticks)
        actor_meta = actor_meta or {}
        f = lambda: np.full(n, np.nan)  # noqa: E731
        out = cls(
            *(f() for _ in range(17)),
            [None] * n,
            f(),
            [None] * n,
            f(),
            f(),
            f(),
            [None] * n,
            f(),
            f(),
            f(),
            f(),
        )
        t0 = ticks[0]["time"] if n else 0.0
        for k, r in enumerate(ticks):
            ego, ctrl, ext = r["ego"], r["ctrl"], r.get("extras") or {}
            route, lane, lights = r.get("route") or {}, r.get("lane") or {}, r.get("lights") or {}
            out.t[k] = r["time"] - t0
            out.v[k] = ego["speed"]
            out.v_des[k] = ctrl.get("v_des") or 0.0
            out.brake[k], out.throttle[k], out.steer[k] = ctrl["brake"], ctrl["throttle"], ctrl["steer"]
            out.s[k], out.frac[k], out.dev[k] = (
                route.get("s", np.nan),
                route.get("frac", np.nan),
                route.get("dev", np.nan),
            )
            out.x[k], out.y[k] = ego["x"], ego["y"]
            plan = r.get("plan") or []
            if plan:
                out.plan_end_x[k], out.plan_end_y[k] = plan[-1]
            lead = (ext.get("lead") or [None])[0]
            if lead:
                out.lead_prob[k], out.lead_d[k], out.lead_vrel[k] = lead["prob"], lead["d"], lead["v_rel"]
            nxt = lights.get("next")
            if nxt:
                out.light_d[k] = nxt["d"]
                out.light_state[k] = str(nxt.get("state") or "").split(".")[-1]
            out.junction[k] = 1.0 if lane.get("junction") else 0.0
            out.lane[k] = (lane.get("road"), lane.get("lane")) if lane else None
            out.lane_offset[k] = lane.get("offset", np.nan) if lane else np.nan
            best, movers = None, 0
            for aid, ax, ay, _ayaw, av in r.get("actors") or []:
                if 0.0 < ax and math.hypot(ax, ay) < 60.0 and av > 1.0:
                    movers += 1
                if 0.0 < ax < AHEAD_M and abs(ay) < AHEAD_HALF_M and (best is None or ax < best[1]):
                    best = (aid, ax, av)
            out.movers_ahead[k] = movers
            if best:
                out.ahead_d[k], out.ahead_v[k] = best[1], best[2]
                out.ahead_kind[k] = actor_kind(actor_meta.get(best[0]), best[2])
            ln = ext.get("lane") or {}
            probs = ln.get("line_probs") or []
            if len(probs) >= 3:
                out.lane_probs[k] = (probs[1] + probs[2]) / 2
            ds = (ext.get("desire") or {}).get("state") or []
            if len(ds) >= 3:
                out.desire_turn[k] = ds[1] + ds[2]
            out.infer_ms[k] = r.get("infer_ms") or np.nan
        return out

    def ahead(self, k: int) -> str:
        """What blocked the lane at tick ``k``, the way a reader would name it.

        A red or yellow light on the route within LIGHT_NEAR_M first; then the nearest actor in the
        ego's lane -- a stationary vehicle, prop or pedestrian anywhere within AHEAD_M, a moving
        vehicle only within MOVING_NEAR_M (further than that it is traffic, not an obstacle); then
        the model's own lead if it saw one within AHEAD_M that the privileged state does not put in
        the lane; else ``nothing``.
        """
        if (
            self.light_state[k] in ("Red", "Yellow")
            and np.isfinite(self.light_d[k])
            and self.light_d[k] < LIGHT_NEAR_M
        ):
            return "red light"
        kind = self.ahead_kind[k]
        if kind is not None and (kind != "moving vehicle" or self.ahead_d[k] < MOVING_NEAR_M):
            return kind
        if np.isfinite(self.lead_prob[k]) and self.lead_prob[k] > 0.5 and self.lead_d[k] < AHEAD_M:
            return "model lead only"
        return "nothing"

    def ahead_m(self, k: int) -> float | None:
        """Distance to what ``ahead(k)`` names, when it is an actor."""
        what = self.ahead(k)
        if what in ("red light", "nothing"):
            return None
        d = self.lead_d[k] if what == "model lead only" else self.ahead_d[k]
        return round(float(d), 1) if np.isfinite(d) else None


# --------------------------------------------------------------------------- events


@dataclass
class Standstill:
    route_id: str
    start_s: float
    end_s: float | None  # None: still standing when the route ended
    duration_s: float
    progress_pct: float
    ahead_at_start: str  # Signals.ahead() when the car stopped
    ahead_at_end: str  # ... and when it moved again, or when the route ended
    ahead_m_at_start: float | None  # how far ahead that actor was, metres (None for a light or nothing)
    ahead_m_at_end: float | None
    released: bool
    lead_prob: float | None
    plan_reach_m: float  # mean |last waypoint| while standing: how far the model planned to go
    green_wait_s: float | None  # standing after the light ahead turned green
    movers_ahead: float  # mean number of moving actors in view over the stop's last 2 s


def standstills(route_id: str, sg: Signals) -> list[Standstill]:
    still = sg.v < STILL_V
    events: list[Standstill] = []
    k, n = 0, len(sg.t)
    while k < n:
        if not still[k]:
            k += 1
            continue
        j = k
        while j < n and still[j]:
            j += 1
        dur = sg.t[j - 1] - sg.t[k] + 1 / TICK_HZ
        if dur >= MIN_STILL_S:
            ended = j < n
            green_wait = None
            if sg.ahead(k) == "red light":
                greens = [m for m in range(k, j) if sg.light_state[m] == "Green"]
                if greens:
                    green_wait = round(float(sg.t[j - 1] - sg.t[greens[0]]), 2)
            reach = np.hypot(sg.plan_end_x[k:j], sg.plan_end_y[k:j])
            last2 = slice(max(k, j - int(2 * TICK_HZ)), j)
            events.append(
                Standstill(
                    route_id,
                    round(float(sg.t[k]), 2),
                    round(float(sg.t[j - 1]), 2) if ended else None,
                    round(float(dur), 2),
                    round(100 * float(np.nan_to_num(sg.frac[k])), 1),
                    sg.ahead(k),
                    sg.ahead(j - 1),
                    sg.ahead_m(k),
                    sg.ahead_m(j - 1),
                    ended,
                    round(float(sg.lead_prob[k]), 3) if np.isfinite(sg.lead_prob[k]) else None,
                    round(float(np.nanmean(reach)), 3) if len(reach) else float("nan"),
                    green_wait,
                    round(float(np.nanmean(sg.movers_ahead[last2])), 2),
                )
            )
        k = j
    return events


def light_events(route_id: str, sg: Signals, plan: dict) -> list[dict]:
    """Each light on the route the car reached: the state the moment its along-route position passed the stop line."""
    out = []
    s_route = np.asarray(plan.get("s") or [], dtype=float)
    for light in plan.get("traffic_lights") or []:
        if light["i"] >= len(s_route):
            continue
        s_light = s_route[light["i"]]
        crossed = np.where(sg.s >= s_light)[0]
        approached = np.where(np.abs(sg.s - s_light) < LIGHT_NEAR_M)[0]
        if not len(approached):
            continue
        k = int(crossed[0]) if len(crossed) else None
        state_at_cross = None
        if k is not None:
            # The record's "next light" moves on once the car is past this one; read the tick before.
            prev = max(0, k - 1)
            state_at_cross = sg.light_state[prev] if np.isfinite(sg.light_d[prev]) else None
        window = approached[approached < (k if k is not None else len(sg.t))]
        red_seen = any(sg.light_state[m] == "Red" for m in window)
        stopped_for_red = any(
            sg.v[m] < STILL_V and sg.light_state[m] == "Red" and sg.light_d[m] < LIGHT_NEAR_M for m in window
        )
        out.append(
            {
                "route_id": route_id,
                "light_id": light["id"],
                "crossed": k is not None,
                "cross_s": round(float(sg.t[k]), 2) if k is not None else None,
                "state_at_cross": state_at_cross,
                "speed_at_cross": round(float(sg.v[k]), 2) if k is not None else None,
                "red_seen_on_approach": red_seen,
                "stopped_for_red": stopped_for_red,
            }
        )
    return out


def turn_events(route_id: str, sg: Signals, plan: dict, deviated: bool) -> list[dict]:
    """Each LEFT / RIGHT / STRAIGHT junction manoeuvre on the route: taken, or where the car left the route.

    ``left_route_here``: the leaderboard ended the route on a deviation and this is the manoeuvre the car
    reached and did not complete. The deviation's own location cannot say so -- the leaderboard flags it
    only once the car is 30 m off the route, by which time it is 30 m from any turn.
    """
    opts = plan.get("options") or []
    s_route = np.asarray(plan.get("s") or [], dtype=float)
    out = []
    i, missed = 0, False
    while i < len(opts):
        if opts[i] in ("LEFT", "RIGHT", "STRAIGHT"):
            j = i
            while j + 1 < len(opts) and opts[j + 1] == opts[i]:
                j += 1
            s0, s1 = s_route[i], s_route[j]
            reached = bool(np.nanmax(sg.s) >= s0) if len(sg.s) else False
            done = bool(np.nanmax(sg.s) >= s1) if len(sg.s) else False
            left_here = deviated and reached and not done and not missed
            missed = missed or left_here
            inside = (sg.s >= s0 - 5) & (sg.s <= s1 + 5)
            out.append(
                {
                    "route_id": route_id,
                    "maneuver": opts[i],
                    "start_m": round(float(s0), 1),
                    "length_m": round(float(s1 - s0), 1),
                    "reached": reached,
                    "completed": done,
                    "left_route_here": left_here,
                    "mean_speed_in": round(float(np.nanmean(sg.v[inside])), 2) if inside.any() else None,
                    "model_turn_desire_in": round(float(np.nanmean(sg.desire_turn[inside])), 3)
                    if inside.any()
                    else None,
                }
            )
            i = j + 1
        else:
            i += 1
    return out


def collision_events(route_id: str, sg: Signals, result: dict, actor_meta: dict) -> list[dict]:
    out = []
    for inf in runs.infraction_points(result):
        if not inf["kind"].startswith("collisions") or inf["x"] is None or not len(sg.x):
            continue
        k = int(np.argmin(np.hypot(sg.x - inf["x"], sg.y - inf["y"])))
        before = slice(max(0, k - int(2 * TICK_HZ)), k + 1)
        m = _TYPE.search(inf["text"])
        out.append(
            {
                "route_id": route_id,
                "kind": inf["kind"],
                "other": m.group(1) if m else "",
                "t_s": round(float(sg.t[k]), 2),
                "speed": round(float(sg.v[k]), 2),
                "in_junction": bool(sg.junction[k]),
                "lead_prob_2s_before": round(float(np.nanmax(sg.lead_prob[before])), 3)
                if np.isfinite(sg.lead_prob[before]).any()
                else None,
                "lead_d_2s_before": round(float(np.nanmin(sg.lead_d[before])), 2)
                if np.isfinite(sg.lead_d[before]).any()
                else None,
                "brake_2s_before": round(float(np.nanmax(sg.brake[before])), 2),
                "actor_ahead_min_m": round(float(np.nanmin(sg.ahead_d[before])), 2)
                if np.isfinite(sg.ahead_d[before]).any()
                else None,
            }
        )
    return out


# --------------------------------------------------------------------------- the per-route row


@dataclass
class RouteFeatures:
    route_id: str
    town: str
    scenario: str
    abilities: str
    status: str
    ds: float | None
    rc: float | None
    penalty: float | None
    success: bool
    cause: str = ""  # the primary pattern's label
    cause_family: str = ""
    patterns: str = ""  # every pattern that holds, ';'-separated names
    game_s: float | None = None
    launch_s: float | None = None  # first time above MOVED_V
    plan_reach_at_start_m: float | None = None
    max_speed: float | None = None
    mean_speed: float | None = None
    mean_speed_moving: float | None = None
    mean_speed_limit: float | None = None
    stopped_frac: float | None = None
    n_standstills: int = 0
    standstill_reasons: str = ""  # ahead_at_start of each standstill, ';'-separated
    final_standstill_start_s: float | None = None  # a standstill the route ended in
    final_standstill_s: float | None = None
    final_ahead_at_start: str | None = None
    final_ahead_at_end: str | None = None
    red_light_waits: int = 0
    mean_green_wait_s: float | None = None
    max_route_dev_m: float | None = None
    lane_changes: int = 0
    mean_abs_lane_offset_m: float | None = None
    lights_crossed: int = 0
    lights_crossed_red: int = 0
    turns_on_route: int = 0
    turns_completed: int = 0
    left_route_at_turn: bool = False
    deviated: bool = False
    collisions: int = 0
    collision_kinds: str = ""
    lead_detected_frac: float | None = None
    mean_ego_lane_prob: float | None = None
    mean_infer_ms: float | None = None
    expert_available: bool = False
    expert_duration_s: float | None = None
    expert_mean_speed: float | None = None
    expert_mean_speed_moving: float | None = None  # over frames above STILL_V, like mean_speed_moving
    expert_time_to_rc_s: float | None = None  # expert's time to reach the route completion the ego scored
    ego_time_to_rc_s: float | None = None
    infractions: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- failure patterns


@dataclass
class RouteContext:
    """Everything a pattern may look at for one route."""

    features: RouteFeatures
    status: str
    result: dict
    standstills: list[Standstill]
    lights: list[dict]
    turns: list[dict]
    collisions: list[dict]

    @property
    def final(self) -> Standstill | None:
        """The standstill the route ended in, if it ended standing."""
        st = self.standstills
        return st[-1] if st and not st[-1].released else None

    @property
    def failed(self) -> bool:
        return not self.status.startswith(("Perfect", "Completed"))

    def infractions(self, kind: str) -> int:
        events = (self.result.get("infractions") or {}).get(kind) or []
        return len(events) if isinstance(events, list) else int(events)


@dataclass(frozen=True)
class Pattern:
    """One failure (or success) mode, as a test over a route's recorded ticks.

    ``test`` returns a short evidence string when the pattern holds, else None. Every pattern that
    holds is written to ``patterns.csv`` with its evidence; the one with the lowest ``primary``
    becomes the route's ``cause``. ``primary=None`` makes a pattern a secondary label only -- true of
    the route, but never the reason it scored what it did.
    """

    name: str
    label: str
    family: str
    primary: int | None
    summary: str
    test: Callable[[RouteContext], str | None]


def _final_ahead(ctx: RouteContext, *kinds: str, at_end: bool = True) -> str | None:
    fin = ctx.final
    if fin is None or not ctx.failed:
        return None
    what = fin.ahead_at_end if at_end else fin.ahead_at_start
    if what in kinds:
        dist = fin.ahead_m_at_end if at_end else fin.ahead_m_at_start
        where = f" {dist:.0f} m ahead" if dist is not None else ""
        return f"stood {fin.duration_s:.0f} s from t={fin.start_s:.0f} s to the end; {what}{where}"
    return None


def _clear_road_trap(ctx: RouteContext, began_with: tuple[str, ...]) -> str | None:
    """Stood until the route ended with nothing in its lane at the end, having stopped for ``began_with``."""
    fin = ctx.final
    if (
        fin is None
        or not ctx.failed
        or fin.start_s < 2.0
        or fin.ahead_at_end not in ("nothing", "model lead only")
    ):
        return None
    if fin.ahead_at_start not in began_with:
        return None
    wait = f", {fin.green_wait_s:.0f} s of it after green" if fin.green_wait_s else ""
    return (
        f"stopped at t={fin.start_s:.0f} s ({fin.ahead_at_start} ahead), stood {fin.duration_s:.0f} s{wait} "
        f"with a clear lane at the end; plan reach {fin.plan_reach_m:.2f} m"
    )


def _deviated_at_turn(ctx: RouteContext) -> str | None:
    missed = [t for t in ctx.turns if t["left_route_here"] and t["maneuver"] in ("LEFT", "RIGHT")]
    if not missed:
        return None
    t = missed[0]
    return f"reached the {t['maneuver']} turn at {t['start_m']:.0f} m and left the route there; model turn desire {t['model_turn_desire_in']}"


def _collided(ctx: RouteContext, kind: str) -> str | None:
    hits = [c for c in ctx.collisions if c["kind"] == kind]
    if not hits:
        return None
    c = hits[0]
    return f"{len(hits)} x {c['other'] or kind} at {c['speed']:.1f} m/s (t={c['t_s']:.0f} s); lead prob 2 s before {c['lead_prob_2s_before']}"


def _ran_red_after_stopping(ctx: RouteContext) -> str | None:
    lights = [x for x in ctx.lights if x["crossed"] and x["state_at_cross"] == "Red" and x["stopped_for_red"]]
    if not lights or not ctx.infractions("red_light"):
        return None
    return f"stopped for red, then crossed {len(lights)} light(s) still on red at {lights[0]['speed_at_cross']} m/s"


def _slow_restart(ctx: RouteContext) -> str | None:
    slow = [e for e in ctx.standstills if e.released and e.green_wait_s and e.green_wait_s > SLOW_GREEN_S]
    if not slow:
        return None
    return f"{len(slow)} red-light stop(s) left a median {np.median([e.green_wait_s for e in slow]):.0f} s after green"


#: The catalogue. Order within a family is only for reading; `primary` decides the cause.
PATTERNS: list[Pattern] = [
    # -- outcome
    Pattern("not_scored", "crashed (not scored)", "not scored", 0,
            "the simulator or the agent crashed; the leaderboard's number is not a measurement",
            lambda c: c.status if any(c.status.endswith(x) for x in runs.CRASH_SUFFIXES) else None),
    Pattern("success", "success", "success", 1,
            "completed with no penalised infraction (Bench2Drive's success)",
            lambda c: "completed, no penalised infraction" if c.features.success else None),
    Pattern("completed_with_collision", "completed, with a collision", "completed with an infraction", 2,
            "reached the end of the route but hit something on the way",
            lambda c: f"{c.features.collisions} collision(s)" if not c.failed and c.features.collisions else None),
    Pattern("completed_ran_red", "completed, ran a red light", "completed with an infraction", 3,
            "reached the end of the route but crossed a red light",
            lambda c: f"{c.infractions('red_light')} red light(s)" if not c.failed and c.infractions("red_light") else None),
    Pattern("completed_other", "completed, other infraction", "completed with an infraction", 4,
            "reached the end with a stop-sign, lane or emergency-vehicle infraction",
            lambda c: c.features.infractions if not c.failed and not c.features.success else None),
    # -- standstill: the car will not (re)start
    Pattern("never_pulled_away", "never pulled away", "standstill trap", 10,
            "stood still from the first tick to the end; supercombo planned centimetres throughout",
            lambda c: f"plan reach at start {c.features.plan_reach_at_start_m} m; ahead: {c.final.ahead_at_start if c.final else '?'}"
            if c.features.launch_s is None else None),
    Pattern("trap_after_red_light", "stopped at a red light and never restarted", "standstill trap", 14,
            "stopped correctly for red, the light turned green, and the car stood until the route ended",
            lambda c: _clear_road_trap(c, ("red light",))),
    Pattern("trap_after_lead_left", "stopped behind traffic and never restarted after it left", "standstill trap", 15,
            "stopped behind a moving vehicle; it drove off, the car did not",
            lambda c: _clear_road_trap(c, ("moving vehicle", "stationary vehicle", "prop", "pedestrian"))),
    Pattern("trap_clear_road", "stopped with a clear road and never restarted", "standstill trap", 16,
            "stopped with nothing in its lane and stood until the route ended",
            lambda c: _clear_road_trap(c, ("nothing", "model lead only"))),
    # -- blocked: something really is in the lane, and openpilot does not go around it
    Pattern("blocked_by_stationary_vehicle", "stuck behind a stationary vehicle", "blocked, no overtake", 12,
            "a stopped car or accident in its lane, still there when the route ended; openpilot does not change lanes to pass",
            lambda c: _final_ahead(c, "stationary vehicle")),
    Pattern("blocked_by_prop", "stuck behind cones / a parked mesh", "blocked, no overtake", 12,
            "construction cones, warning signs or a parked car in its lane; no lane change to pass",
            lambda c: _final_ahead(c, "prop")),
    Pattern("blocked_by_pedestrian", "stuck behind a pedestrian", "blocked, no overtake", 12,
            "a pedestrian in its lane until the route ended",
            lambda c: _final_ahead(c, "pedestrian")),
    Pattern("stuck_in_traffic", "stuck in queued traffic", "blocked, no overtake", 13,
            "a moving vehicle close ahead at the end: queued traffic it neither followed off nor passed",
            lambda c: _final_ahead(c, "moving vehicle")),
    Pattern("stuck_at_red_light", "still at a red light when the route ended", "standstill trap", 13,
            "the route ended while the light ahead was still red",
            lambda c: _final_ahead(c, "red light")),
    # -- route following: no route input at the pinned adapter defaults
    Pattern("missed_junction_turn", "missed a junction turn", "route following", 11,
            "went straight (or the wrong way) where the route turned; the adapter sends no route desire by default",
            lambda c: _deviated_at_turn(c) if c.features.deviated else None),
    Pattern("left_route_elsewhere", "left the route away from a turn", "route following", 11.5,
            "deviated from the route away from any junction manoeuvre (lane change, exit, drift)",
            lambda c: f"max {c.features.max_route_dev_m} m off route" if c.features.deviated and not _deviated_at_turn(c) else None),
    # -- contact
    Pattern("collision_then_stuck", "collided, then stood until the end", "collision", 11.8,
            "a collision, after which the car never moved again",
            lambda c: f"{c.features.collision_kinds}; stood {c.final.duration_s:.0f} s after"
            if c.failed and c.features.collisions and c.final and not c.features.deviated else None),
    Pattern("collision", "collision", "collision", 20,
            "failed with a collision that did not end in a standstill",
            lambda c: c.features.collision_kinds if c.failed and c.features.collisions else None),
    Pattern("too_slow", "ran out of time while moving", "too slow", 30,
            "hit the 4000-tick cap without a final standstill",
            lambda c: c.status if c.failed and ("TickRuntime" in c.status or "timed out" in c.status) and not c.final else None),
    Pattern("other_failure", "other failure", "other", 99, "none of the above",
            lambda c: c.status if c.failed else None),
    # -- secondary labels (never the primary cause)
    Pattern("ran_red_light", "ran a red light", "traffic rules", None, "the leaderboard recorded a red-light infraction",
            lambda c: f"{c.infractions('red_light')} red light(s)" if c.infractions("red_light") else None),
    Pattern("ran_red_after_stopping", "stopped for red, then went on red", "traffic rules", None,
            "stood at the red light, then crossed it while it was still red", _ran_red_after_stopping),
    Pattern("stopped_for_red", "stopped for a red light", "traffic rules", None,
            "stood at a red light on the route (correct)",
            lambda c: f"{sum(1 for x in c.lights if x['stopped_for_red'])} light(s)" if any(x["stopped_for_red"] for x in c.lights) else None),
    Pattern("slow_restart_after_green", "slow to restart after green", "standstill trap", None,
            f"stood more than {SLOW_GREEN_S:.0f} s after the light turned green, then went", _slow_restart),
    Pattern("ran_stop_sign", "ran a stop sign", "traffic rules", None, "stop-sign infraction",
            lambda c: f"{c.infractions('stop_infraction')}" if c.infractions("stop_infraction") else None),
    Pattern("collision_vehicle", "hit a vehicle", "collision", None, "collision with a vehicle",
            lambda c: _collided(c, "collisions_vehicle")),
    Pattern("collision_static", "hit a static object", "collision", None, "collision with layout or a prop",
            lambda c: _collided(c, "collisions_layout")),
    Pattern("collision_pedestrian", "hit a pedestrian", "collision", None, "collision with a pedestrian",
            lambda c: _collided(c, "collisions_pedestrian")),
    Pattern("turn_taken_unprompted", "took a junction turn with no route input", "route following", None,
            "completed a LEFT/RIGHT manoeuvre although the model was never told the route",
            lambda c: ", ".join(t["maneuver"] for t in c.turns if t["maneuver"] in ("LEFT", "RIGHT") and t["completed"]) or None),
    Pattern("lane_change", "changed lanes", "route following", None, "the lane id changed outside a junction",
            lambda c: f"{c.features.lane_changes}" if c.features.lane_changes else None),
    Pattern("outside_route_lanes", "drove outside the route lanes", "traffic rules", None, "outside-route-lanes infraction",
            lambda c: f"{c.infractions('outside_route_lanes')}" if c.infractions("outside_route_lanes") else None),
    Pattern("did_not_yield_emergency", "did not yield to an emergency vehicle", "traffic rules", None,
            "yield-to-emergency-vehicle infraction",
            lambda c: "yes" if c.infractions("yield_emergency_vehicle_infractions") else None),
]  # fmt: skip


#: Holds only when no other primary pattern does: the catalogue's admission that it has no name for a route.
FALLBACK = "other_failure"


def match(ctx: RouteContext) -> list[tuple[Pattern, str]]:
    """Every pattern that holds for the route, with its evidence."""
    out = []
    for pattern in PATTERNS:
        if pattern.name == FALLBACK:
            continue
        evidence = pattern.test(ctx)
        if evidence:
            out.append((pattern, evidence))
    if not any(p.primary is not None for p, _ in out):
        fallback = next(p for p in PATTERNS if p.name == FALLBACK)
        evidence = fallback.test(ctx)
        if evidence:
            out.append((fallback, evidence))
    return out


def primary(matches: list[tuple[Pattern, str]]) -> Pattern | None:
    ranked = [p for p, _ in matches if p.primary is not None]
    return min(ranked, key=lambda p: p.primary) if ranked else None


def analyse_route(att: runs.Attempt, meta: dict, expert_root: Path | None) -> tuple[RouteFeatures, dict]:
    result = att.result or {}
    scores = result.get("scores") or {}
    ticks = runs.load_ticks(att.path)
    plan = runs.load_json(att.path / "route_plan.json") or {}
    actor_meta = {}
    path = att.path / "actors.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                a = json.loads(line)
                actor_meta[a["id"]] = a
            except ValueError:
                pass
    status = result.get("status", "")
    f = RouteFeatures(
        route_id=att.route_id,
        town=att.town,
        scenario=att.scenario,
        abilities=",".join(meta.get("abilities") or runs.abilities(att.scenario)),
        status=status,
        ds=scores.get("score_composed"),
        rc=scores.get("score_route"),
        penalty=scores.get("score_penalty"),
        success=status in ("Perfect", "Completed")
        and not any(v for k, v in (result.get("infractions") or {}).items() if k != "min_speed_infractions"),
        game_s=(result.get("meta") or {}).get("duration_game"),
    )
    f.infractions = ";".join(
        f"{k}:{len(v)}"
        for k, v in (result.get("infractions") or {}).items()
        if v and k != "min_speed_infractions"
    )
    events: dict[str, list] = {"standstills": [], "lights": [], "turns": [], "collisions": [], "patterns": []}
    f.deviated = "deviated" in status
    if not ticks:
        _label(f, RouteContext(f, status, result, [], [], [], []), events)
        return f, events
    sg = Signals.from_ticks(ticks, actor_meta)
    moved = np.where(sg.v > MOVED_V)[0]
    f.launch_s = round(float(sg.t[moved[0]]), 2) if len(moved) else None
    first = sg.t < 3.0
    f.plan_reach_at_start_m = round(
        float(np.nanmean(np.hypot(sg.plan_end_x[first], sg.plan_end_y[first]))), 3
    )
    f.max_speed = round(float(np.nanmax(sg.v)), 2)
    f.mean_speed = round(float(np.nanmean(sg.v)), 2)
    f.mean_speed_moving = (
        round(float(np.nanmean(sg.v[sg.v > STILL_V])), 2) if (sg.v > STILL_V).any() else None
    )
    limits = np.array([(r["ego"].get("speed_limit") or np.nan) for r in ticks], dtype=float)
    f.mean_speed_limit = round(float(np.nanmean(limits)), 2) if np.isfinite(limits).any() else None
    f.stopped_frac = round(float((sg.v < STILL_V).mean()), 3)
    st = standstills(att.route_id, sg)
    events["standstills"] = [asdict(e) for e in st]
    f.n_standstills = len(st)
    f.standstill_reasons = ";".join(e.ahead_at_start for e in st)
    if st and not st[-1].released:
        fin = st[-1]
        f.final_standstill_start_s, f.final_standstill_s = fin.start_s, fin.duration_s
        f.final_ahead_at_start, f.final_ahead_at_end = fin.ahead_at_start, fin.ahead_at_end
    waits = [e.green_wait_s for e in st if e.green_wait_s is not None]
    f.red_light_waits = sum(1 for e in st if e.ahead_at_start == "red light")
    f.mean_green_wait_s = round(float(np.mean(waits)), 2) if waits else None
    f.max_route_dev_m = round(float(np.nanmax(sg.dev)), 2)
    lanes = [ln for ln, j in zip(sg.lane, sg.junction, strict=True) if ln is not None and not j]
    f.lane_changes = sum(1 for a, b in zip(lanes, lanes[1:], strict=False) if a[0] == b[0] and a[1] != b[1])
    f.mean_abs_lane_offset_m = round(float(np.nanmean(np.abs(sg.lane_offset))), 3)
    lights = light_events(att.route_id, sg, plan)
    events["lights"] = lights
    f.lights_crossed = sum(1 for e in lights if e["crossed"])
    f.lights_crossed_red = sum(1 for e in lights if e["crossed"] and e["state_at_cross"] == "Red")
    turns = turn_events(att.route_id, sg, plan, f.deviated)
    events["turns"] = turns
    f.turns_on_route = sum(1 for e in turns if e["maneuver"] in ("LEFT", "RIGHT"))
    f.turns_completed = sum(1 for e in turns if e["maneuver"] in ("LEFT", "RIGHT") and e["completed"])
    f.left_route_at_turn = any(e["left_route_here"] for e in turns)
    cols = collision_events(att.route_id, sg, result, actor_meta)
    events["collisions"] = cols
    f.collisions = len(cols)
    f.collision_kinds = ";".join(sorted({c["kind"] for c in cols}))
    f.lead_detected_frac = round(float(np.nanmean(sg.lead_prob > 0.5)), 3)
    f.mean_ego_lane_prob = (
        round(float(np.nanmean(sg.lane_probs)), 3) if np.isfinite(sg.lane_probs).any() else None
    )
    f.mean_infer_ms = round(float(np.nanmean(sg.infer_ms)), 1)
    ex = expert_mod.load_expert(expert_root, att.route_id) if expert_root else None
    if ex and plan.get("xy"):
        f.expert_available = True
        f.expert_duration_s = ex["duration_s"]
        ev = np.array([fr.get("speed") or 0.0 for fr in ex["frames"]], dtype=float)
        f.expert_mean_speed = round(float(ev.mean()), 2)
        if (ev > STILL_V).any():
            f.expert_mean_speed_moving = round(float(ev[ev > STILL_V].mean()), 2)
        rxy = np.asarray(plan["xy"], dtype=float)
        rs = np.asarray(plan["s"], dtype=float)
        es = np.array(
            [rs[int(np.argmin(np.hypot(rxy[:, 0] - fr["x"], rxy[:, 1] - fr["y"])))] for fr in ex["frames"]]
        )
        target = (f.rc or 0) / 100 * float(plan.get("length_m") or rs[-1])
        reach_e = np.where(es >= target - 2.0)[0]
        reach_o = np.where(sg.s >= target - 2.0)[0]
        if target > 5 and len(reach_e):
            f.expert_time_to_rc_s = round(float(reach_e[0] / ex.get("hz", 10.0)), 2)
        if target > 5 and len(reach_o):
            f.ego_time_to_rc_s = round(float(sg.t[reach_o[0]]), 2)
    _label(f, RouteContext(f, status, result, st, lights, turns, cols), events)
    return f, events


def _label(f: RouteFeatures, ctx: RouteContext, events: dict) -> None:
    """Match every pattern; the primary one becomes the route's cause."""
    matches = match(ctx)
    top = primary(matches)
    f.cause = top.label if top else ""
    f.cause_family = top.family if top else ""
    f.patterns = ";".join(p.name for p, _ in matches)
    events["patterns"] = [
        {
            "route_id": f.route_id,
            "pattern": p.name,
            "label": p.label,
            "family": p.family,
            "is_primary": p is top,
            "evidence": evidence,
        }
        for p, evidence in matches
    ]


def write_csv(path: Path, rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return 0
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def analyse(
    run_dirs: list[Path], expert_root: Path | None, routes_xml: Path, out: Path
) -> list[RouteFeatures]:
    table = runs.route_table(routes_xml)
    chosen = runs.latest(runs.attempts(run_dirs))
    out.mkdir(parents=True, exist_ok=True)
    features, events = [], {"standstills": [], "lights": [], "turns": [], "collisions": [], "patterns": []}
    for rid in sorted(chosen, key=int):
        att = chosen[rid]
        if att.result is None:
            continue
        f, ev = analyse_route(att, table.get(rid, {}), expert_root)
        features.append(f)
        for k in events:
            events[k] += ev[k]
    rows = []
    for f in features:
        row = asdict(f)
        row.pop("extra")
        rows.append(row)
    write_csv(out / "route_features.csv", rows)
    for k, v in events.items():
        write_csv(out / f"{k}.csv", v)
    (out / "causes.json").write_text(json.dumps({f.route_id: f.cause for f in features}, indent=0))
    write_csv(
        out / "pattern_catalogue.csv",
        [
            {"name": p.name, "label": p.label, "family": p.family, "primary": p.primary, "summary": p.summary}
            for p in PATTERNS
        ],
    )
    return features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--expert", type=Path)
    ap.add_argument("--routes-xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    features = analyse(args.runs, args.expert, args.routes_xml, args.out)
    from collections import Counter

    print(f"{len(features)} routes analysed -> {args.out}")
    for cause, n in Counter(f.cause for f in features).most_common():
        print(f"  {n:4d}  {cause}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
