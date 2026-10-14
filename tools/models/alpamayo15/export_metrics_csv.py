#!/usr/bin/env python3
"""Export per-sample metrics from a paddock run to CSV.

Reads {run_dir}/samples.jsonl and writes {run_dir}/metrics_samples.csv with
one row per sample. The 'metrics' dict is flattened into columns; native_id
fields are prefixed with 'native_'. Prediction/ground_truth arrays are omitted.

Usage:
    export_metrics_csv.py <run_id>
    export_metrics_csv.py <run_id> --bench nuscenes --model alpamayo15
    export_metrics_csv.py --run-dir /full/path/to/run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paddock.config import SiteConfig


def _flatten(sample: dict) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample.get("sample_id", ""),
        "split": sample.get("split", ""),
        "group": sample.get("group", ""),
    }
    for k, v in (sample.get("native_id") or {}).items():
        row[f"native_{k}"] = v
    for k, v in (sample.get("metrics") or {}).items():
        row[k] = v
    for k, v in (sample.get("ego") or {}).items():
        row[f"ego_{k}"] = v
    return row


def export(run_dir: Path) -> Path:
    samples_path = run_dir / "samples.jsonl"
    if not samples_path.exists():
        raise SystemExit(
            f"No samples.jsonl in {run_dir}\nMake sure the run dir is in {{runs}}, not in results/."
        )

    rows = [_flatten(json.loads(ln)) for ln in samples_path.read_text().splitlines() if ln.strip()]
    if not rows:
        raise SystemExit(f"samples.jsonl in {run_dir} is empty")

    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                cols.append(k)
                seen.add(k)

    out_path = run_dir / "metrics_samples.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> {out_path}  ({len(rows)} rows, {len(cols)} columns)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample metrics from a paddock run to CSV")
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--model", default="alpamayo15")
    parser.add_argument("--bench", default="kitscenes")
    parser.add_argument("--run-dir", help="Full path to run dir (overrides run_id lookup)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        cfg = SiteConfig.load()
        run_dir = cfg.runs / args.bench / args.model / args.run_id
    else:
        cfg = SiteConfig.load()
        base = cfg.runs / args.bench / args.model
        candidates = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(f"No runs found in {base}")
        run_dir = candidates[0]
        print(f"Using most recent run: {run_dir.name}")

    export(run_dir)


if __name__ == "__main__":
    main()
