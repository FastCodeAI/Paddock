#!/usr/bin/env python3
"""Print a Markdown table of experiment results from the log written by run_experiments.sh.

Usage:
    summarize_results.py experiment_log.txt
    summarize_results.py experiment_log.txt --bench nuscenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get(metrics: dict, key: str, stat: str) -> str:
    val = metrics.get(key, {}).get(stat)
    return f"{val:.3f}" if val is not None else "—"


def load_metrics(bench: str, model: str, run_id: str) -> dict:
    path = REPO_ROOT / "results" / bench / model / run_id / "metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("metrics", {})


def _kitscenes_row(label: str, model: str, run_id: str) -> str:
    m = load_metrics("kitscenes", model, run_id)
    if not m:
        return f"| {label} | — | (metrics not found: {run_id}) | | | | |"
    n = m.get("ade", {}).get("n", "?")
    return (
        f"| {label} | {n}"
        f" | {get(m, 'ade', 'mean')}"
        f" | {get(m, 'ade', 'median')}"
        f" | {get(m, 'fde@3s', 'mean')}"
        f" | {get(m, 'longitudinal_mae', 'mean')}"
        f" | {get(m, 'lateral_mae', 'mean')}"
        " |"
    )


def _nuscenes_row(label: str, model: str, run_id: str) -> str:
    m = load_metrics("nuscenes", model, run_id)
    if not m:
        return f"| {label} | — | (metrics not found: {run_id}) | | | | | |"
    n = m.get("l2_avg", {}).get("n", "?")
    return (
        f"| {label} | {n}"
        f" | {get(m, 'l2_avg', 'mean')}"
        f" | {get(m, 'l2_1s', 'mean')}"
        f" | {get(m, 'l2_2s', 'mean')}"
        f" | {get(m, 'l2_3s', 'mean')}"
        f" | {get(m, 'ade_3s', 'mean')}"
        f" | {get(m, 'col_avg', 'mean')}"
        " |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise paddock experiment results")
    parser.add_argument("log", help="experiment_log.txt written by run_experiments.sh")
    parser.add_argument(
        "--bench", default="kitscenes", help="benchmark the run was scored on (default: kitscenes)"
    )
    args = parser.parse_args()

    log = Path(args.log)
    if not log.exists():
        raise SystemExit(f"Log not found: {log}")

    rows = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) == 4:
            label, bench, model, run_id = parts
            if bench != args.bench:
                continue
        elif len(parts) == 3:
            label, model, run_id = parts
            bench = args.bench
        else:
            print(f"  skipping malformed line: {line!r}", file=sys.stderr)
            continue
        rows.append((label, model, run_id))

    if not rows:
        raise SystemExit(f"No experiments found for bench={args.bench!r} in log.")

    print()
    if args.bench == "nuscenes":
        header = "| Experiment | n | L2 avg ↓ | L2@1s | L2@2s | L2@3s | ADE@3s | Col avg |"
        sep = "|---|---|---|---|---|---|---|---|"
        print(header)
        print(sep)
        for label, model, run_id in rows:
            print(_nuscenes_row(label, model, run_id))
        print()
        print("(n samples; L2 = displacement error in metres; Col = collision rate; lower is better)")
    else:
        header = "| Experiment | n | ADE mean | ADE median | FDE@3s | Long MAE | Lat MAE |"
        sep = "|---|---|---|---|---|---|---|"
        print(header)
        print(sep)
        for label, model, run_id in rows:
            print(_kitscenes_row(label, model, run_id))
        print()
        print("(n samples; lower is better for all metrics)")


if __name__ == "__main__":
    main()
