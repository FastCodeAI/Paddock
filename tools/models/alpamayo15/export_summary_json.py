#!/usr/bin/env python3
"""Generate a summary metrics.json from a run's metrics_samples.csv.

Produces the same structure as paddock's own metrics.json but re-derived
from the CSV so it can be regenerated at any time without the original
samples.jsonl.

Usage:
    python tools/analysis/export_summary_json.py [run_id]
    python tools/analysis/export_summary_json.py --run-dir /path/to/run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paddock.config import SiteConfig

METRICS = [
    "l2_avg",
    "l2_uniad_avg",
    "l2_1s",
    "l2_uniad_1s",
    "l2_2s",
    "l2_uniad_2s",
    "l2_3s",
    "l2_uniad_3s",
    "col_avg",
    "col_uniad_avg",
    "col_1s",
    "col_3s",
    "ade_3s",
]


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "n_missing": 0, "mean": None}
    a = np.array(vals)
    return {
        "n": len(a),
        "n_missing": 0,
        "mean": round(float(a.mean()), 6),
        "median": round(float(np.median(a)), 6),
        "p90": round(float(np.percentile(a, 90)), 6),
        "max": round(float(a.max()), 6),
        "std": round(float(a.std()), 6),
    }


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if r.get(m) not in (None, "", "nan", "NaN")]
        if vals:
            out[m] = _stats(vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
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

    csv_path = run_dir / "metrics_samples.csv"
    if not csv_path.exists():
        sys.exit(f"metrics_samples.csv not found in {run_dir}")

    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)

    summary = {
        "run_id": run_dir.name,
        "benchmark": args.bench,
        "model": args.model,
        "n": n,
        "primary_metric": "l2_uniad_avg",
        "metrics": summarise(rows),
    }

    out_path = run_dir / "metrics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    uniad = summary["metrics"].get("l2_uniad_avg", {}).get("mean")
    l2 = summary["metrics"].get("l2_avg", {}).get("mean")
    print(f"  -> {out_path}  (n={n}, l2_uniad_avg={uniad:.4f}, l2_avg={l2:.4f})")


if __name__ == "__main__":
    main()
