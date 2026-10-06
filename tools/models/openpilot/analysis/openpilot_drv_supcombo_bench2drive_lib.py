"""Why supercombo scores what it scores on Bench2Drive: per-route features, events, and a primary cause.

    python -m tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_lib \\
        --runs <runs dir> --expert <expert dir> --routes-xml <bench2drive220.xml> --out <analysis dir>

Reads, for the attempt that counts at each route (``runs.latest``), what the
agent recorded -- ``ticks.jsonl`` (every node reply beside privileged world
state), ``route_plan.json``, ``actors.jsonl``, ``results.json`` -- and
Think2Drive's drive of the same route (``expert.py``), and writes

    route_features.csv    one row per route: the leaderboard's numbers, the features below, ``cause``
    standstills.csv       every stretch the car stood still: what was ahead, what ended it, and how long it took
    lights.csv            every traffic light on the route the car reached: its state as the car crossed
    turns.csv             every junction manoeuvre on the route: taken, or where the car left the route
    collisions.csv        every collision: speed, the other actor, what the model had seen of it

The question is not "how well" -- the leaderboard answers that -- but *why*,
and the per-tick record makes most of the answers mechanical:

* **standstill** -- supercombo has no speed input and infers motion from frame
  to frame; standing still, its plan collapses to a few centimetres and stays
  there until something in view moves. A stretch is ``v < 0.3 m/s`` for at
  least a second; its *reason* is what was ahead when it began (a red light on
  the route within 30 m, an actor in the ego's lane within 15 m by privileged
  state, the model's own lead within 20 m, or nothing), and its *release* is
  what was true when the car moved again (nothing ahead any more, the light
  green, or -- the trap -- the route ended first).
* **cause** -- one primary reason per route, in this order: crashed (not
  scored) / success / completed with an infraction / never pulled away / left
  the route at a junction turn / left the route elsewhere / blocked behind an
  actor in its lane / blocked at a red light / blocked with nothing ahead /
  collision / ran out of time.

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
from collections.abc import Iterable
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
LIGHT_NEAR_M = 30.0
ACTOR_NEAR_M = 15.0
LEAD_NEAR_M = 20.0
LANE_HALF_M = 1.75
TURN_NEAR_M = 30.0

_TYPE = re.compile(r"type=([\w.]+)")


# --------------------------------------------------------------------------- per-tick signals


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
    ahead_d: np.ndarray  # nearest actor in the ego's lane ahead (privileged), metres
    ahead_v: np.ndarray
    ahead_id: list
    lane_probs: np.ndarray  # the model's two ego lane lines, mean probability
    desire_turn: np.ndarray  # the model's own turnLeft + turnRight desire state
    infer_ms: np.ndarray

    @classmethod
    def from_ticks(cls, ticks: list[dict]) -> Signals:
        n = len(ticks)
        f = lambda: np.full(n, np.nan)  # noqa: E731
        out = cls(
            *(f() for _ in range(17)), [None] * n, f(), [None] * n, f(), f(), f(), [None] * n, f(), f(), f()
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
            best = None
            for aid, ax, ay, _ayaw, av in r.get("actors") or []:
                if 0.0 < ax < 60.0 and abs(ay) < LANE_HALF_M and (best is None or ax < best[1]):
                    best = (aid, ax, av)
            if best:
                out.ahead_id[k], out.ahead_d[k], out.ahead_v[k] = best
            ln = ext.get("lane") or {}
            probs = ln.get("line_probs") or []
            if len(probs) >= 3:
                out.lane_probs[k] = (probs[1] + probs[2]) / 2
            ds = (ext.get("desire") or {}).get("state") or []
            if len(ds) >= 3:
                out.desire_turn[k] = ds[1] + ds[2]
            out.infer_ms[k] = r.get("infer_ms") or np.nan
        return out


# --------------------------------------------------------------------------- events


@dataclass
class Standstill:
    route_id: str
    start_s: float
    end_s: float | None  # None: still standing when the route ended
    duration_s: float
    progress_pct: float
    reason: str  # red_light | actor_ahead | lead | nothing
    release: str  # moved: <what changed> | route ended
    light_state_start: str | None
    actor_ahead_m: float | None
    lead_prob: float | None
    plan_reach_m: float  # mean |last waypoint| while standing: how far the model planned to go
    green_wait_s: float | None  # standing after the light ahead turned green


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
            light = sg.light_state[k] if np.isfinite(sg.light_d[k]) and sg.light_d[k] < LIGHT_NEAR_M else None
            actor = sg.ahead_d[k] if np.isfinite(sg.ahead_d[k]) and sg.ahead_d[k] < ACTOR_NEAR_M else None
            lead = sg.lead_prob[k] if np.isfinite(sg.lead_prob[k]) else None
            if light == "Red" or light == "Yellow":
                reason = "red_light"
            elif actor is not None:
                reason = "actor_ahead"
            elif lead is not None and lead > 0.5 and sg.lead_d[k] < LEAD_NEAR_M:
                reason = "lead"
            else:
                reason = "nothing"
            ended = j < n
            if ended:
                ahead_now = np.isfinite(sg.ahead_d[j]) and sg.ahead_d[j] < ACTOR_NEAR_M
                if reason == "red_light" and sg.light_state[j] != "Red":
                    release = "moved: light changed"
                elif reason == "actor_ahead" and not ahead_now:
                    release = "moved: actor ahead cleared"
                elif reason == "actor_ahead":
                    release = "moved: actor ahead still there"
                else:
                    release = "moved"
            else:
                release = "route ended"
            green_wait = None
            if reason == "red_light":
                greens = [m for m in range(k, j) if sg.light_state[m] == "Green"]
                if greens:
                    green_wait = round(float(sg.t[j - 1] - sg.t[greens[0]]), 2)
            reach = np.hypot(sg.plan_end_x[k:j], sg.plan_end_y[k:j])
            events.append(
                Standstill(
                    route_id,
                    round(float(sg.t[k]), 2),
                    round(float(sg.t[j - 1]), 2) if ended else None,
                    round(float(dur), 2),
                    round(100 * float(np.nan_to_num(sg.frac[k])), 1),
                    reason,
                    release,
                    light,
                    round(float(actor), 2) if actor is not None else None,
                    round(float(lead), 3) if lead is not None else None,
                    round(float(np.nanmean(reach)), 3) if len(reach) else float("nan"),
                    green_wait,
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


def turn_events(
    route_id: str, sg: Signals, plan: dict, deviated_at: tuple[float, float] | None
) -> list[dict]:
    """Each LEFT / RIGHT / STRAIGHT junction manoeuvre on the route: taken, or where the car left the route."""
    opts = plan.get("options") or []
    s_route = np.asarray(plan.get("s") or [], dtype=float)
    xy = np.asarray(plan.get("xy") or [], dtype=float).reshape(-1, 2)
    out = []
    i = 0
    while i < len(opts):
        if opts[i] in ("LEFT", "RIGHT", "STRAIGHT"):
            j = i
            while j + 1 < len(opts) and opts[j + 1] == opts[i]:
                j += 1
            s0, s1 = s_route[i], s_route[j]
            reached = bool(np.nanmax(sg.s) >= s0) if len(sg.s) else False
            done = bool(np.nanmax(sg.s) >= s1) if len(sg.s) else False
            left_here = False
            if deviated_at is not None and len(xy):
                d = np.hypot(xy[i : j + 1, 0] - deviated_at[0], xy[i : j + 1, 1] - deviated_at[1])
                left_here = bool(len(d) and d.min() < TURN_NEAR_M)
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
    cause: str = ""
    game_s: float | None = None
    launch_s: float | None = None  # first time above MOVED_V
    plan_reach_at_start_m: float | None = None
    max_speed: float | None = None
    mean_speed: float | None = None
    mean_speed_moving: float | None = None
    mean_speed_limit: float | None = None
    stopped_frac: float | None = None
    n_standstills: int = 0
    standstill_reasons: str = ""
    final_standstill_reason: str | None = None
    final_standstill_s: float | None = None
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
    collisions: int = 0
    collision_kinds: str = ""
    lead_detected_frac: float | None = None
    mean_ego_lane_prob: float | None = None
    mean_infer_ms: float | None = None
    expert_available: bool = False
    expert_duration_s: float | None = None
    expert_mean_speed: float | None = None
    expert_time_to_rc_s: float | None = None  # expert's time to reach the route completion the ego scored
    ego_time_to_rc_s: float | None = None
    infractions: str = ""
    extra: dict = field(default_factory=dict)


def classify(f: RouteFeatures, status: str) -> str:
    """One primary cause per route; see the module docstring for the order."""
    if any(status.endswith(s) for s in runs.CRASH_SUFFIXES):
        return "crashed (not scored)"
    if f.success:
        return "success"
    if status in ("Perfect", "Completed"):
        if f.collisions:
            return "completed, collision"
        if f.lights_crossed_red or "red_light" in f.infractions:
            return "completed, ran a red light"
        return "completed, other infraction"
    if f.launch_s is None:
        return "never pulled away"
    if "deviated" in status:
        return (
            "left the route at a junction turn" if f.left_route_at_turn else "left the route (not at a turn)"
        )
    if "blocked" in status:
        if f.collisions:
            return "collision, then blocked"
        return {
            "actor_ahead": "blocked behind an actor in its lane",
            "lead": "blocked behind a detected lead",
            "red_light": "blocked at a red light",
            "nothing": "blocked with nothing ahead (standstill trap)",
        }.get(f.final_standstill_reason or "", "blocked")
    if f.collisions:
        return "collision"
    if "TickRuntime" in status or "timed out" in status:
        return "ran out of time"
    return "other failure"


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
    events: dict[str, list] = {"standstills": [], "lights": [], "turns": [], "collisions": []}
    if not ticks:
        f.cause = classify(f, status)
        return f, events
    sg = Signals.from_ticks(ticks)
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
    f.standstill_reasons = ";".join(e.reason for e in st)
    if st and st[-1].end_s is None:
        f.final_standstill_reason, f.final_standstill_s = st[-1].reason, st[-1].duration_s
    waits = [e.green_wait_s for e in st if e.green_wait_s is not None]
    f.red_light_waits = sum(1 for e in st if e.reason == "red_light")
    f.mean_green_wait_s = round(float(np.mean(waits)), 2) if waits else None
    f.max_route_dev_m = round(float(np.nanmax(sg.dev)), 2)
    lanes = [ln for ln, j in zip(sg.lane, sg.junction, strict=True) if ln is not None and not j]
    f.lane_changes = sum(1 for a, b in zip(lanes, lanes[1:], strict=False) if a[0] == b[0] and a[1] != b[1])
    f.mean_abs_lane_offset_m = round(float(np.nanmean(np.abs(sg.lane_offset))), 3)
    lights = light_events(att.route_id, sg, plan)
    events["lights"] = lights
    f.lights_crossed = sum(1 for e in lights if e["crossed"])
    f.lights_crossed_red = sum(1 for e in lights if e["crossed"] and e["state_at_cross"] == "Red")
    deviated_at = None
    for inf in runs.infraction_points(result):
        if inf["kind"] == "route_dev" and inf["x"] is not None:
            deviated_at = (inf["x"], inf["y"])
    turns = turn_events(att.route_id, sg, plan, deviated_at)
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
    f.cause = classify(f, status)
    return f, events


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
    features, events = [], {"standstills": [], "lights": [], "turns": [], "collisions": []}
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
