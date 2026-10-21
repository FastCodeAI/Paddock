#!/usr/bin/env python3
"""Turn every trajectory (predicted and ground-truth) into a decision label.

For each sample, classifies both the predicted and ground-truth trajectory
into a lateral decision (straight / turn_left / turn_right / lane_change /
nudge) and a longitudinal decision (keep_speed / slow_down / stop /
speed_up), purely from waypoint geometry — no map or annotation data needed.
This turns raw L2 error into "right decision" vs. "wrong decision", which is
the thing failure taxonomy actually cares about.

Samples at index_in_scene == 0 (the cold-start frame -- no real pose history
yet, per the egomotion-warmstart discussion) are skipped entirely, so they
never enter decisions.csv and therefore never show up in decision_confusion.py
or anything else built on top of it.

Writes decisions.csv into the run directory (join key: sample_id).

Usage:
    python tools/models/alpamayo15/decisions.py [run_id]
    python tools/models/alpamayo15/decisions.py --stats   # print lateral-offset / speed-delta
                                                  # histograms instead of writing
                                                  # decisions.csv, to help calibrate
                                                  # the thresholds below against real
                                                  # data before trusting them at scale.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paddock.adapters.benchmarks.nuscenes.loader import TURN_THRESHOLD_M
from paddock.config import SiteConfig

# ---------------------------------------------------------------------------
# Tunable thresholds.
#
# LATERAL_TURN_M reuses nuScenes' own nav command derivation cutoff
# (adapters/benchmarks/nuscenes/loader.py: TURN_THRESHOLD_M, the 3 s horizon),
# so "turn" here stays comparable to the existing native_command-based
# analysis. The rest are first-pass defaults — run with --stats on your
# baseline and look at the actual lateral-offset / speed-delta distributions
# before trusting these at scale; they are not derived from data.
DT = 0.5  # seconds, matches Trajectory.dt

LATERAL_TURN_M = TURN_THRESHOLD_M
LATERAL_LANE_CHANGE_M = 1.0
LATERAL_NUDGE_M = 0.3
HEADING_TURN_DEG = 20.0

STOP_SPEED_MPS = 0.5
SPEED_DELTA_MPS = 1.0  # slow_down / speed_up cutoff on (v_end - v_start) over the horizon
PEAK_DECEL_MPS2 = 1.0  # per-step deceleration that alone counts as "slowing down"


def _heading_deg(dx: float, dy: float) -> float | None:
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx))


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def geometry_features(waypoints: list) -> dict | None:
    """From a list of [x, y] ego-frame waypoints (t = dt, 2*dt, ...), compute
    the raw geometric features the decision classifiers use. Returns None if
    there aren't enough points to say anything (there is no waypoint at t=0,
    so the origin is prepended to get the first step's speed/heading)."""
    if not waypoints or len(waypoints) < 2:
        return None

    pts = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in waypoints]
    steps = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        speed = math.hypot(dx, dy) / DT
        steps.append((speed, _heading_deg(dx, dy)))

    speeds = [s for s, _ in steps]
    headings = [h for _, h in steps if h is not None]

    v0, vf = speeds[0], speeds[-1]
    min_speed = min(speeds)
    peak_decel = (
        max((speeds[i] - speeds[i + 1]) / DT for i in range(len(speeds) - 1)) if len(speeds) > 1 else 0.0
    )
    heading_change = _wrap_deg(headings[-1] - headings[0]) if len(headings) >= 2 else 0.0
    lateral_offset = pts[-1][1]  # y of the final waypoint, ego frame (left is positive)

    return dict(
        v0=v0,
        vf=vf,
        min_speed=min_speed,
        peak_decel=peak_decel,
        heading_change_deg=heading_change,
        lateral_offset_m=lateral_offset,
    )


def classify_lateral(feats: dict) -> str:
    lat, hdg = feats["lateral_offset_m"], feats["heading_change_deg"]
    if abs(lat) >= LATERAL_TURN_M or abs(hdg) >= HEADING_TURN_DEG:
        return "turn_left" if lat > 0 else "turn_right"
    if abs(lat) >= LATERAL_LANE_CHANGE_M:
        return "lane_change"
    if abs(lat) >= LATERAL_NUDGE_M:
        return "nudge"
    return "straight"


def classify_longitudinal(feats: dict) -> str:
    v0, vf = feats["v0"], feats["vf"]
    if feats["min_speed"] < STOP_SPEED_MPS or vf < STOP_SPEED_MPS:
        return "stop"
    if (vf - v0) >= SPEED_DELTA_MPS:
        return "speed_up"
    if (v0 - vf) >= SPEED_DELTA_MPS or feats["peak_decel"] >= PEAK_DECEL_MPS2:
        return "slow_down"
    return "keep_speed"


def label(waypoints: list) -> tuple[str | None, str | None, dict | None]:
    feats = geometry_features(waypoints)
    if feats is None:
        return None, None, None
    return classify_lateral(feats), classify_longitudinal(feats), feats


def _print_histogram(title: str, values: list, edges: list) -> None:
    print(f"\n{title}  (n={len(values)})")
    buckets = [0] * (len(edges) + 1)
    for v in values:
        buckets[bisect.bisect_right(edges, v)] += 1
    lo = "-inf"
    for hi, count in zip(list(edges) + [None], buckets):
        hi_s = "inf" if hi is None else hi
        print(f"  [{str(lo):>6}, {str(hi_s):>6}) : {count}")
        lo = hi


def _resolve_run_dir(args, cfg) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id:
        return cfg.runs / args.bench / args.model / args.run_id
    cache = DIR / "baseline_run_id.txt"
    if not cache.exists():
        sys.exit("No run_id given and baseline_run_id.txt not found.")
    return cfg.runs / args.bench / args.model / cache.read_text().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Print lateral-offset/speed-delta histograms instead of writing "
        "decisions.csv, to help calibrate the thresholds at the top of this file.",
    )
    args = ap.parse_args()

    cfg = SiteConfig.load()
    run_dir = _resolve_run_dir(args, cfg)

    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        sys.exit(f"samples.jsonl not found in {run_dir}")

    rows = []
    lateral_offsets = []
    speed_deltas = []
    with samples_path.open() as f:
        for line in f:
            s = json.loads(line)
            native_id = s.get("native_id", {})
            if native_id.get("index_in_scene") == 0:
                continue  # cold-start frame: no real pose history, not a genuine decision
            pred, gt = s.get("prediction"), s.get("ground_truth")
            if not pred or not gt:
                continue
            n = min(len(pred), len(gt))
            pred, gt = pred[:n], gt[:n]

            lat_pred, lon_pred, feats_pred = label(pred)
            lat_gt, lon_gt, feats_gt = label(gt)
            if feats_pred is None or feats_gt is None:
                continue

            lateral_offsets.append(feats_pred["lateral_offset_m"])
            speed_deltas.append(feats_pred["vf"] - feats_pred["v0"])

            rows.append(
                dict(
                    sample_id=s.get("sample_id", ""),
                    native_scene=native_id.get("scene", s.get("group", "")),
                    native_index_in_scene=native_id.get("index_in_scene"),
                    native_command=native_id.get("command"),
                    horizon_n=n,
                    decision_lateral_pred=lat_pred,
                    decision_lateral_gt=lat_gt,
                    decision_longitudinal_pred=lon_pred,
                    decision_longitudinal_gt=lon_gt,
                    lateral_match=lat_pred == lat_gt,
                    longitudinal_match=lon_pred == lon_gt,
                    lateral_offset_m_pred=feats_pred["lateral_offset_m"],
                    lateral_offset_m_gt=feats_gt["lateral_offset_m"],
                    heading_change_deg_pred=feats_pred["heading_change_deg"],
                    heading_change_deg_gt=feats_gt["heading_change_deg"],
                    v0_pred=feats_pred["v0"],
                    vf_pred=feats_pred["vf"],
                    v0_gt=feats_gt["v0"],
                    vf_gt=feats_gt["vf"],
                    min_speed_pred=feats_pred["min_speed"],
                    min_speed_gt=feats_gt["min_speed"],
                    peak_decel_pred=feats_pred["peak_decel"],
                    peak_decel_gt=feats_gt["peak_decel"],
                )
            )

    if not rows:
        sys.exit("No samples had both prediction and ground_truth.")

    if args.stats:
        _print_histogram(
            "lateral offset at horizon (pred, m)",
            lateral_offsets,
            [-4, -2, -1, -0.3, 0.3, 1, 2, 4],
        )
        _print_histogram(
            "speed delta over horizon (pred, m/s)",
            speed_deltas,
            [-3, -1.5, -0.5, 0.5, 1.5, 3],
        )
        return

    out_path = run_dir / "decisions.csv"
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_lat_mismatch = sum(1 for r in rows if not r["lateral_match"])
    n_lon_mismatch = sum(1 for r in rows if not r["longitudinal_match"])
    print(f"Run: {run_dir.name}  (n={len(rows)} labeled samples)")
    print(f"  lateral mismatch:      {n_lat_mismatch} ({100 * n_lat_mismatch / len(rows):.1f}%)")
    print(f"  longitudinal mismatch: {n_lon_mismatch} ({100 * n_lon_mismatch / len(rows):.1f}%)")
    print(f"  written {out_path}")


if __name__ == "__main__":
    main()
