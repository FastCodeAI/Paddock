"""Two openpilot scores from one Bench2Drive run.

openpilot's model does not launch from a standstill without a driver, so the
adapter emulates a driver-resume nudge (``PADDOCK_OPENPILOT_RESUME_S``) and logs
each nudge in the per-route ``log.csv`` ``resume`` column. From that *single* run
this derives two scores per route, so the assisted number never stands alone:

* **unassisted** -- the score had the car stayed stopped at the FIRST nudge, as
  the leaderboard would have recorded it ("Agent got blocked" there): route
  completion at the nudge times the infraction penalty earned up to that point
  along the route.
* **assisted** -- the full run's Driving Score (what the leaderboard records),
  marked ``*`` with the nudge count.

A route with no nudge scores identically in both. Route completion at the nudge
is the ego's progress along the route polyline (an approximation; see Fidelity
below) divided by the route length from ``results.json``.

This is the shipped, reproducible form of the two-case protocol documented in
``docs/benchmarks/bench2drive.md``; it reads only the artefacts a normal run
writes (``results.json`` + ``log.csv`` per route).

**Fidelity.** The *assisted* score is the leaderboard's own composed score, read
straight from ``results.json`` -- exact. The *unassisted* score is an
approximate reconstruction of a counterfactual ("had the car stayed stopped at
the first nudge"), rebuilt from the log, and is not expected to match the
leaderboard's stateful scorer to the digit. Its known approximations:

* route completion at the nudge is the ego's projection onto the route polyline
  (2-D), where the leaderboard tracks a stateful, windowed 3-D distance;
* the penalty earned before the nudge multiplies only infractions whose logged
  position falls before it -- infractions with no position (e.g. scenario
  timeout, emergency-vehicle yield) cannot be placed and are omitted;
* ``outside_route_lanes`` uses its coefficient here rather than the leaderboard's
  route-percentage formula.

A row is dropped when the route is an infrastructure failure (agent/sim crash,
setup failure), and flagged in a ``note`` column when the resume telemetry or the
route polyline needed for the unassisted case is missing. One repetition per
route is assumed. Run::

    python -m paddock.adapters.benchmarks.bench2drive.tally <run_dir> \\
        --routes-xml <repo>/leaderboard/data/bench2drive220.xml

``<run_dir>`` is a directory holding ``<route_id>-Town*`` per-route
subdirectories (a paddock run's artefact directory). Pure functions below take
paths only -- no machine location is baked in.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.bench2drive.harness import CRASH_STATUSES

#: Bench2Drive / CARLA leaderboard infraction penalty coefficients (the
#: multipliers the official Driving Score applies per event). Kept here beside
#: the tally that uses them so a change to the scoring shows up in a diff; they
#: match leaderboard ``statistics_manager`` and the values under which the
#: reported numbers were produced.
INFRACTION_PENALTY: dict[str, float] = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "scenario_timeouts": 0.70,
    "yield_emergency_vehicle_infractions": 0.70,
    "stop_infraction": 0.80,
    "outside_route_lanes": 0.90,
}

#: The simulator logs one row per tick at 20 Hz.
_LOG_DT_S = 0.05
_XY_RE = re.compile(r"x=(-?[\d.]+), y=(-?[\d.]+)")


def _is_infra_failure(status: str) -> bool:
    """True for a route that did not run (crash / bad sensors / failed setup).

    Reuses the harness's own ``CRASH_STATUSES`` (the leaderboard's
    ``FAILURE_MESSAGES``) so this list never drifts from the one the runner uses;
    the manager writes them as a ``"Failed - <message>"`` suffix. "Agent got
    blocked" is deliberately NOT one of them -- it is a legitimate scored outcome.
    """
    return any(status.endswith(msg) for msg in CRASH_STATUSES)


def route_polyline(routes_xml: Path, route_id: str) -> np.ndarray | None:
    """The route's waypoint polyline as ``(N, 2)`` ego-log XY, or None.

    CARLA's ``y`` is negated to match the ``ego_y`` the agent logs, so a route
    point and a logged position share one frame.
    """
    root = ET.parse(routes_xml).getroot()
    for route in root.findall(".//route"):
        if route.get("id") != route_id:
            continue
        pts = route.findall(".//waypoints/position") or route.findall(".//position")
        if not pts:
            return None
        return np.array([[float(p.get("x")), -float(p.get("y"))] for p in pts], dtype=np.float64)
    return None


def progress_along(poly: np.ndarray, pt: np.ndarray) -> tuple[float, float]:
    """``(distance_along, total_length)`` at the projection of ``pt`` onto ``poly``.

    The along-track distance is the arc length to the nearest point of the 2-D
    route polyline. This is an *approximation* of route completion; Bench2Drive's
    own definition tracks a stateful, windowed 3-D distance, so the two can
    differ (see the module docstring's fidelity note).
    """
    seg = np.diff(poly, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    best_dist, best_along = 1e18, 0.0
    for i in range(len(seg)):
        a, d, ll = poly[i], seg[i], seg_len[i]
        t = 0.0 if ll < 1e-6 else float(np.clip(np.dot(pt - a, d) / (ll * ll), 0.0, 1.0))
        proj = a + t * d
        dist = float(np.hypot(*(pt - proj)))
        if dist < best_dist:
            best_dist, best_along = dist, cum[i] + t * ll
    return best_along, float(cum[-1])


def _first_nudge_index(log_csv: Path) -> tuple[int, int | None, list[list[str]], dict[str, int]]:
    """``(nudge_count, first_nudge_row, body_rows, header_index)`` from a log.csv.

    A nudge is a rising edge of the ``resume`` column. Returns ``first`` = None
    when the column is absent or never rises.
    """
    rows = list(csv.reader(log_csv.open()))
    header = rows[0]
    ix = {k: i for i, k in enumerate(header)}
    body = [r for r in rows[1:] if len(r) > ix.get("ego_y", 1_000_000)]
    nudges, first, prev = 0, None, 0
    if "resume" in ix:
        ri = ix["resume"]
        for i, r in enumerate(body):
            v = int(float(r[ri])) if len(r) > ri and r[ri] != "" else 0
            if v and not prev:
                nudges += 1
                if first is None:
                    first = i
            prev = v
    return nudges, first, body, ix


def _penalty_before(rec: dict[str, Any], poly: np.ndarray, along_at_nudge: float) -> float:
    """Product of penalty coefficients for infractions logged at or before the
    nudge point along the route. Events without a parseable position are ignored
    (they cannot be placed relative to the nudge)."""
    penalty = 1.0
    for key, events in (rec.get("infractions") or {}).items():
        coeff = INFRACTION_PENALTY.get(key)
        if coeff is None or not isinstance(events, list):
            continue
        for text in events:
            m = _XY_RE.search(str(text))
            if not m:
                continue
            p = np.array([float(m.group(1)), -float(m.group(2))], dtype=np.float64)
            along, _ = progress_along(poly, p)
            if along <= along_at_nudge + 1.0:
                penalty *= coeff
    return penalty


def two_case_rows(run_dir: Path, routes_xml: Path) -> list[dict[str, Any]]:
    """One row per scored route with its unassisted and assisted scores.

    Rows carry a ``note`` when the unassisted reconstruction is degraded (no
    resume telemetry, or the route polyline is missing); infrastructure-failure
    routes are dropped entirely.
    """
    rows: list[dict[str, Any]] = []
    for d in sorted(p for p in run_dir.glob("*-Town*") if p.is_dir()):
        rid = d.name.split("-")[0]
        try:
            records = json.loads((d / "results.json").read_text())["_checkpoint"]["records"]
        except (OSError, KeyError, IndexError, ValueError):
            continue
        if not records:
            continue
        rec = records[0]
        status = rec.get("status") or ""
        # Drop infra failures; they carry no scored outcome (N1).
        if not status or _is_infra_failure(status):
            continue
        scores = rec["scores"]
        ds, rc, pen = scores["score_composed"], scores["score_route"], scores["score_penalty"]
        route_length = float(rec.get("meta", {}).get("route_length") or 0.0) or None

        notes: list[str] = []
        # `unassisted_ok` gates a row's unassisted score from the mean: the
        # assisted score is always exact, but the unassisted reconstruction is
        # only trustworthy when the resume telemetry, the log rows and (for a
        # nudged route) the polyline are all present, and there is one repetition.
        unassisted_ok = True
        if len(records) > 1:
            # A multi-rep run would need the log matched to the right repetition,
            # which this tool does not do (N5): don't average its unassisted score.
            notes.append(f"{len(records)} repetitions; used records[0]")
            unassisted_ok = False

        log_csv = d / "log.csv"
        first = None
        nudges = 0
        body: list[list[str]] = []
        ix: dict[str, int] = {}
        if log_csv.is_file():
            nudges, first, body, ix = _first_nudge_index(log_csv)
        # A header-only log (resume column but no data rows) is not telemetry.
        if "resume" not in ix or not body:
            notes.append("no resume telemetry")
            unassisted_ok = False

        una_rc, una_ds = rc, ds
        first_nudge_s = None
        if first is not None:
            first_nudge_s = round(first * _LOG_DT_S, 1)
            poly = route_polyline(routes_xml, rid)
            if poly is None:
                notes.append("route polyline missing; unassisted unavailable")
                unassisted_ok = False
            else:
                pt = np.array(
                    [float(body[first][ix["ego_x"]]), float(body[first][ix["ego_y"]])],
                    dtype=np.float64,
                )
                along, total = progress_along(poly, pt)
                total = route_length or total
                una_rc = round(100.0 * along / total, 2) if total else 0.0
                una_ds = round(una_rc * _penalty_before(rec, poly, along), 2)

        rows.append(
            {
                "route": rid,
                "scenario": d.name.split("-", 2)[-1],
                "status": status[:22],
                "nudges": nudges,
                "first_nudge_s": first_nudge_s,
                "unassisted_RC": una_rc,
                "unassisted_DS": una_ds,
                "unassisted_ok": unassisted_ok,
                "assisted_RC": rc,
                "assisted_DS": round(ds, 2),
                "penalty": pen,
                "note": "; ".join(notes),
            }
        )
    return rows


def _print(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'route':>6} {'scenario':<28} {'nudges':>6} {'t_nudge':>8} "
        f"{'unassisted DS':>14} {'assisted DS':>12}  status"
    )
    for r in rows:
        assisted = str(r["assisted_DS"]) + ("*" if r["nudges"] else "")
        tail = f"  {r['status']}" + (f"  [{r['note']}]" if r.get("note") else "")
        print(
            f"{r['route']:>6} {r['scenario']:<28} {r['nudges']:>6} "
            f"{str(r['first_nudge_s']):>8} {r['unassisted_DS']:>14} {assisted:>12}{tail}"
        )
    print("\nassisted DS is exact; unassisted DS is an approximate reconstruction (see module docstring)")
    if rows:
        # Assisted is exact for every row; unassisted only where it is trustworthy.
        usable = [r for r in rows if r.get("unassisted_ok")]
        mean_ass = sum(r["assisted_DS"] for r in rows) / len(rows)
        with_nudge = sum(1 for r in rows if r["nudges"])
        excluded = len(rows) - len(usable)
        una = (
            f"mean unassisted DS {sum(r['unassisted_DS'] for r in usable) / len(usable):.2f} "
            f"(n={len(usable)}" + (f", {excluded} excluded as unavailable" if excluded else "") + ")"
            if usable
            else "mean unassisted DS n/a (no route had a usable reconstruction)"
        )
        print(f"\n{una} | mean assisted DS {mean_ass:.2f} (n={len(rows)}, routes with nudges: {with_nudge})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Two openpilot scores from one Bench2Drive run.")
    ap.add_argument("run_dir", type=Path, help="run artefact dir holding <route>-Town* subdirs")
    ap.add_argument(
        "--routes-xml",
        type=Path,
        required=True,
        help="leaderboard routes XML (e.g. <repo>/leaderboard/data/bench2drive220.xml)",
    )
    args = ap.parse_args(argv)
    _print(two_case_rows(args.run_dir, args.routes_xml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
