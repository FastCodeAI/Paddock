#!/usr/bin/env python3
"""Clear every analysis artifact this tools/analysis/ pipeline generates for
a run, so you can rerun the whole thing from scratch without stale files
lying around.

Removes (only if present): decisions.csv, flicker_frames.json,
flicker_scenes.json, worst_frames.json, worst_frames_no_frame0.json,
worst_scenes.json, worst_scenes_for_video.json, metrics_samples.csv,
reasoning.json, reasoning_summary.csv, reasoning_frames/, category_frames/,
scenario_frames/ (decision_confusion.py's per-category worst-5 renders),
scenario_report.json, taxonomy.csv, taxonomy_summary.json (the last two are
from the retired failure_taxonomy.py -- harmless to also clear if an old run
directory still has them).

Deliberately does NOT touch: samples.jsonl, metrics.json, run.json -- the
actual model output. Everything this script removes is cheaply regenerable
by rerunning decisions.py / decision_confusion.py / decision_flicker.py /
worst_cases.py / print_reasoning.py / render_category.py /
export_metrics_csv.py against the untouched samples.jsonl.

Usage:
    python tools/analysis/clear_reasoning_and_taxonomy.py <run_id>
    python tools/analysis/clear_reasoning_and_taxonomy.py <run_id> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from paddock.config import SiteConfig

TARGETS = [
    "decisions.csv",
    "flicker_frames.json",
    "flicker_scenes.json",
    "worst_frames.json",
    "worst_frames_no_frame0.json",
    "worst_scenes.json",
    "worst_scenes_for_video.json",
    "metrics_samples.csv",
    "reasoning.json",
    "reasoning_summary.csv",
    "reasoning_frames",
    "category_frames",
    "scenario_frames",
    "scenario_report.json",
    "taxonomy.csv",
    "taxonomy_summary.json",
]

KEPT = ["samples.jsonl", "metrics.json", "run.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be removed")
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

    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    print(f"Run: {run_dir}")
    for name in TARGETS:
        path = run_dir / name
        if not path.exists():
            print(f"  (not present) {name}")
            continue
        if args.dry_run:
            print(f"  would remove: {name}")
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  removed: {name}")

    print(f"\nLeft untouched: {', '.join(KEPT)}")


if __name__ == "__main__":
    main()
