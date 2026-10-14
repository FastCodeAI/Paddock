#!/usr/bin/env python3
"""For each scenario category, show L2 split by whether the sample is a turn.

Answers: is the bad performance in intersection/high-traffic/pedestrians etc.
driven by those scenes happening to also be turns, or is there a separate
non-turn effect?

Usage:
    python tools/analysis/turn_overlap.py [run_id]
    python tools/analysis/turn_overlap.py              # uses baseline_run_id.txt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT / "src"))
from scenario_tags import cmd_int as _coerce_cmd
from scenario_tags import is_turn as _is_turn
from scenario_tags import load_scene_descriptions, tag_categories

from paddock.config import SiteConfig


def mmean(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def fmt(v):
    return f"{v:.3f}" if not np.isnan(v) else "  — "


def cmd_int(r):
    v = r.get("command") or r.get("native_command") or "2"
    return _coerce_cmd(v)


def is_turn(r):
    return _is_turn(cmd_int(r))


def main():
    cfg = SiteConfig.load()

    # resolve run ID
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        cache = DIR / "baseline_run_id.txt"
        if not cache.exists():
            sys.exit("No run_id given and baseline_run_id.txt not found.")
        run_id = cache.read_text().strip()

    run_dir = cfg.runs / "nuscenes" / "alpamayo15" / run_id
    csv_path = run_dir / "metrics_samples.csv"
    if not csv_path.exists():
        sys.exit(f"metrics_samples.csv not found in {run_dir}")

    nuscenes_root = cfg.datasets["nuscenes"].resolve()
    version = cfg.datasets["nuscenes"].version or "v1.0-trainval"
    descriptions = load_scene_descriptions(nuscenes_root, version)

    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        scene = r.get("native_scene") or r.get("group", "")
        desc = descriptions.get(scene, "")
        r["_tags"] = tag_categories(desc, cmd_int(r))
        r["_l2"] = float(r.get("l2_avg") or "nan")
        r["_turn"] = is_turn(r)

    overall = mmean([r["_l2"] for r in rows])
    n_turn = sum(1 for r in rows if r["_turn"])
    n_straight = len(rows) - n_turn
    l2_turn = mmean([r["_l2"] for r in rows if r["_turn"]])
    l2_straight = mmean([r["_l2"] for r in rows if not r["_turn"]])

    print(f"\nRun: {run_id}")
    print(f"Overall L2: {overall:.3f}  (n={len(rows)})")
    print(f"  turns (cmd 0/1):      L2={l2_turn:.3f}  n={n_turn}")
    print(f"  non-turns (cmd 2/3):  L2={l2_straight:.3f}  n={n_straight}")

    # Dimensions to check (skip maneuver — that IS the turn split)
    DIMS = [
        ("intersection", "Intersection"),
        ("traffic_density", "Traffic density"),
        ("dynamic_agents_ped", "Dynamic agents — ped"),
        ("dynamic_agents_veh", "Dynamic agents — veh"),
        ("road_type", "Road type"),
    ]

    BAD_DELTA = 0.10
    MIN_N = 10

    for dim_key, dim_label in DIMS:
        by_cat: dict[str, list] = {}
        for r in rows:
            cat = r["_tags"][dim_key]
            by_cat.setdefault(cat, []).append(r)

        # only print categories that were flagged bad OR are interesting
        cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))

        W = max(len(c) for c in cats)
        print(f"\n  [{dim_label}]")
        print(
            f"  {'Category':<{W}} | {'n':>4} | {'L2 all':>6} | "
            f"{'n↺':>4} | {'L2 turn':>7} | {'n→':>4} | {'L2 non-turn':>11} | flag"
        )
        print(f"  {'-' * W}-+------+--------+------+---------+------+-------------+----")
        for cat in cats:
            sub = by_cat[cat]
            n = len(sub)
            l2_all = mmean([r["_l2"] for r in sub])
            turns = [r for r in sub if r["_turn"]]
            straights = [r for r in sub if not r["_turn"]]
            l2_t = mmean([r["_l2"] for r in turns])
            l2_s = mmean([r["_l2"] for r in straights])
            bad = not np.isnan(l2_all) and l2_all > overall + BAD_DELTA and n >= MIN_N
            flag = "◄" if bad else ""
            print(
                f"  {cat:<{W}} | {n:>4} | {fmt(l2_all):>6} | "
                f"{len(turns):>4} | {fmt(l2_t):>7} | {len(straights):>4} | {fmt(l2_s):>11} | {flag}"
            )


if __name__ == "__main__":
    main()
