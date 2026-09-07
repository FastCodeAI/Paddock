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
is the ego's progress along the route polyline (Bench2Drive's own RC definition)
divided by the route length from ``results.json``.

This is the shipped, reproducible form of the two-case protocol documented in
``docs/benchmarks/bench2drive.md``; it reads only the artefacts a normal run
writes (``results.json`` + ``log.csv`` per route). Run::

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

    The along-track distance is measured to the nearest point of the polyline,
    which is Bench2Drive's own route-completion definition.
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
    """One row per completed route with its unassisted and assisted scores."""
    rows: list[dict[str, Any]] = []
    for d in sorted(p for p in run_dir.glob("*-Town*") if p.is_dir()):
        rid = d.name.split("-")[0]
        try:
            rec = json.loads((d / "results.json").read_text())["_checkpoint"]["records"][0]
        except (OSError, KeyError, IndexError, ValueError):
            continue
        if not rec.get("status"):
            continue
        scores = rec["scores"]
        ds, rc, pen = scores["score_composed"], scores["score_route"], scores["score_penalty"]
        route_length = float(rec.get("meta", {}).get("route_length") or 0.0) or None

        log_csv = d / "log.csv"
        first = None
        nudges = 0
        if log_csv.is_file():
            nudges, first, body, ix = _first_nudge_index(log_csv)

        una_rc, una_ds = rc, ds
        first_nudge_s = None
        if first is not None:
            poly = route_polyline(routes_xml, rid)
            if poly is not None:
                pt = np.array(
                    [float(body[first][ix["ego_x"]]), float(body[first][ix["ego_y"]])],
                    dtype=np.float64,
                )
                along, total = progress_along(poly, pt)
                total = route_length or total
                una_rc = round(100.0 * along / total, 2) if total else 0.0
                una_ds = round(una_rc * _penalty_before(rec, poly, along), 2)
            first_nudge_s = round(first * _LOG_DT_S, 1)

        rows.append(
            {
                "route": rid,
                "scenario": d.name.split("-", 2)[-1],
                "status": rec["status"][:22],
                "nudges": nudges,
                "first_nudge_s": first_nudge_s,
                "unassisted_RC": una_rc,
                "unassisted_DS": una_ds,
                "assisted_RC": rc,
                "assisted_DS": round(ds, 2),
                "penalty": pen,
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
        print(
            f"{r['route']:>6} {r['scenario']:<28} {r['nudges']:>6} "
            f"{str(r['first_nudge_s']):>8} {r['unassisted_DS']:>14} {assisted:>12}  {r['status']}"
        )
    if rows:
        mean_una = sum(r["unassisted_DS"] for r in rows) / len(rows)
        mean_ass = sum(r["assisted_DS"] for r in rows) / len(rows)
        with_nudge = sum(1 for r in rows if r["nudges"])
        print(
            f"\nmean unassisted DS {mean_una:.2f} | mean assisted DS {mean_ass:.2f}  "
            f"(n={len(rows)}, routes with nudges: {with_nudge})"
        )


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
