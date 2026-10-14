#!/usr/bin/env python3
"""Compare Alpamayo's implied initial speed against nuScenes ground truth.

Alpamayo's speed at each frame is not logged directly, but it is recoverable
from the first predicted waypoint: at t=0.5s (GRID_INDICES[0]) the ego-frame
displacement divided by waypoint_dt is the average speed the model is
"committing to" for the first half-second. Ground truth is derived the same
way from the recorded trajectory, so both numbers come from an identical
displacement/dt calculation and are directly comparable.

A large gap means the model's initial-speed estimate (derived from its pose
history, see `_history()` / `_sample_poses()` in policy.py) diverged sharply
from what the car actually did next — the egomotion-bootstrap failure mode
this experiment exists to catch.

Usage:
    python tools/analysis/speed_discrepancy.py [run_id]
    python tools/analysis/speed_discrepancy.py --run-dir /path/to/run
    python tools/analysis/speed_discrepancy.py --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paddock.config import SiteConfig

WAYPOINT_DT = 0.5  # AlpamayoPolicy.waypoint_dt


def cmd_int(native_id: dict) -> int:
    try:
        return int(native_id.get("command", 2))
    except Exception:
        return 2


def tag_row(native_id: dict, desc: str) -> dict:
    d = desc.lower()
    cmd = cmd_int(native_id)

    if any(w in d for w in ("switch lane", "change lane", "lane change", "merge left", "merge right")):
        maneuver = "lane_change"
    elif any(w in d for w in ("standstill", "stationary")) or (
        cmd in (2, 3) and any(w in d for w in ("wait", "red light", "stop sign"))
    ):
        maneuver = "stop"
    elif any(w in d for w in ("pulling out", "start from", "begin moving", "accelerat")):
        maneuver = "start"
    elif cmd == 0:
        maneuver = "turn_right"
    elif cmd == 1:
        maneuver = "turn_left"
    else:
        maneuver = "straight"

    intersect = (
        "intersection"
        if any(w in d for w in ("intersection", "junction", "crossroads"))
        else "non-intersection"
    )

    return dict(maneuver=maneuver, intersection=intersect)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument("--top", type=int, default=25, help="Number of worst discrepancies to list")
    ap.add_argument(
        "--exclude-frame-0",
        action="store_true",
        help="Skip index_in_scene == 0 (the cold-start frame) when listing worst discrepancies",
    )
    args = ap.parse_args()

    cfg = SiteConfig.load()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = cfg.runs / args.bench / args.model / args.run_id
    else:
        cache = DIR / "baseline_run_id.txt"
        if not cache.exists():
            sys.exit("No run_id given and baseline_run_id.txt not found.")
        run_dir = cfg.runs / args.bench / args.model / cache.read_text().strip()

    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        sys.exit(f"samples.jsonl not found in {run_dir}")

    root = cfg.datasets["nuscenes"].resolve()
    version = cfg.datasets["nuscenes"].version or "v1.0-trainval"
    descriptions = {
        s["name"]: s.get("description", "") for s in json.loads((root / version / "scene.json").read_text())
    }

    records = []
    with samples_path.open() as f:
        for line in f:
            s = json.loads(line)
            pred = s.get("prediction")
            gt = s.get("ground_truth")
            if pred is None or gt is None:
                continue
            pred0 = np.asarray(pred[0][:2], dtype=float)
            gt0 = np.asarray(gt[0][:2], dtype=float)
            pred_speed = float(np.linalg.norm(pred0) / WAYPOINT_DT)
            gt_speed = float(np.linalg.norm(gt0) / WAYPOINT_DT)

            native_id = s.get("native_id", {})
            scene = native_id.get("scene", s.get("group", ""))
            desc = descriptions.get(scene, "")
            tags = tag_row(native_id, desc)

            records.append(
                dict(
                    sample_id=s.get("sample_id", ""),
                    scene=scene,
                    index_in_scene=native_id.get("index_in_scene"),
                    pred_speed=pred_speed,
                    gt_speed=gt_speed,
                    diff=pred_speed - gt_speed,
                    abs_diff=abs(pred_speed - gt_speed),
                    **tags,
                )
            )

    if not records:
        sys.exit("No samples with both prediction and ground_truth found.")

    all_abs = np.array([r["abs_diff"] for r in records])
    overall_mean_abs = float(all_abs.mean())
    overall_std_abs = float(all_abs.std())

    print(f"\nRun: {run_dir.name}  (n={len(records)})")
    print(f"Speed derived from waypoint[0] / {WAYPOINT_DT}s (t=0.5s displacement)")
    print(
        f"\nOverall |pred_speed - gt_speed|: mean={overall_mean_abs:.3f} m/s  std={overall_std_abs:.3f} m/s"
    )
    print(f"Overall pred_speed mean: {np.mean([r['pred_speed'] for r in records]):.3f} m/s")
    print(f"Overall gt_speed mean:   {np.mean([r['gt_speed'] for r in records]):.3f} m/s")

    # ── Averages split by category ────────────────────────────────────────────
    for dim_key, dim_label in [("maneuver", "Maneuver"), ("intersection", "Intersection")]:
        by_cat: dict[str, list] = {}
        for r in records:
            by_cat.setdefault(r[dim_key], []).append(r)

        cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
        W = max(len(c) for c in cats)
        print(f"\n  [{dim_label}]")
        print(
            f"  {'Category':<{W}} | {'n':>5} | {'pred spd':>8} | {'gt spd':>8} | "
            f"{'mean |Δ|':>9} | {'std |Δ|':>8} | {'max |Δ|':>8}"
        )
        print(f"  {'-' * W}-+-------+----------+----------+-----------+----------+---------")
        for cat in cats:
            sub = by_cat[cat]
            pred_m = np.mean([r["pred_speed"] for r in sub])
            gt_m = np.mean([r["gt_speed"] for r in sub])
            abs_d = np.array([r["abs_diff"] for r in sub])
            print(
                f"  {cat:<{W}} | {len(sub):>5} | {pred_m:>8.3f} | {gt_m:>8.3f} | "
                f"{abs_d.mean():>9.3f} | {abs_d.std():>8.3f} | {abs_d.max():>8.3f}"
            )

    # ── Worst individual discrepancies ────────────────────────────────────────
    pool = records
    label = "all frames"
    if args.exclude_frame_0:
        pool = [r for r in records if r["index_in_scene"] != 0]
        label = "excluding index_in_scene == 0"

    worst = sorted(pool, key=lambda r: -r["abs_diff"])[: args.top]
    print(f"\n  Top {len(worst)} largest speed discrepancies ({label}, n={len(pool)}):")
    print(f"  {'scene':<14} | {'idx':>4} | {'pred spd':>8} | {'gt spd':>8} | {'diff':>7} | maneuver")
    print(f"  {'-' * 14}-+------+----------+----------+---------+---------")
    for r in worst:
        print(
            f"  {r['scene']:<14} | {r['index_in_scene']:>4} | {r['pred_speed']:>8.3f} | "
            f"{r['gt_speed']:>8.3f} | {r['diff']:>+7.3f} | {r['maneuver']}"
        )

    out_name = "speed_discrepancy_no_frame0.json" if args.exclude_frame_0 else "speed_discrepancy.json"
    out_path = run_dir / out_name
    out_path.write_text(json.dumps(worst, indent=2))
    print(f"\n  Written: {out_path}")


if __name__ == "__main__":
    main()
