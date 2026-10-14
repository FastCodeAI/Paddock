#!/usr/bin/env python3
"""Slice per-sample nuScenes metrics by scene condition.

Joins metrics_samples.csv (from export_metrics_csv.py) to nuScenes scene.json
and tags each sample by weather, lighting, road type, and maneuver, then prints
a Markdown table per tag group.

Usage:
    slice_by_scenario.py <run_dir>
    slice_by_scenario.py <run_dir> --nuscenes-root /data/nuscenes --version v1.0-trainval
    slice_by_scenario.py --model alpamayo15 --bench nuscenes   # most-recent run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from scenario_tags import (
    load_scene_descriptions as _load_scene_descriptions,
)
from scenario_tags import (
    tag_command_maneuver as _tag_maneuver,
)
from scenario_tags import (
    tag_lighting as _tag_light,
)
from scenario_tags import (
    tag_scenario_bucket as _tag_scenario,
)
from scenario_tags import (
    tag_weather as _tag_weather,
)

METRICS = ["l2_avg", "l2_1s", "l2_2s", "l2_3s", "ade_3s"]
METRIC_LABELS = ["L2 avg", "L2@1s", "L2@2s", "L2@3s", "ADE@3s"]


def _load_csv(path: Path) -> tuple[list[str], list[dict]]:
    import csv

    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def _print_table(groups: dict[str, list[dict]], label: str) -> None:
    header = f"{'Group':<20} {'n':>5} " + " ".join(f"{m:>8}" for m in METRIC_LABELS)
    print(f"\n### {label}\n")
    print("| " + " | ".join(["Group", "n"] + METRIC_LABELS) + " |")
    print("|" + "|".join(["-" * 22, "-" * 7] + ["-" * 10] * len(METRIC_LABELS)) + "|")
    for group_name in sorted(groups):
        rows = groups[group_name]
        n = len(rows)
        vals = []
        for m in METRICS:
            nums = [float(r[m]) for r in rows if r.get(m) not in (None, "", "nan")]
            vals.append(f"{np.mean(nums):.3f}" if nums else "—")
        print("| " + " | ".join([f"{group_name:<20}", f"{n:>5}"] + [f"{v:>8}" for v in vals]) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice nuScenes metrics by scene condition")
    parser.add_argument("run_dir", nargs="?", help="Path to run directory containing metrics_samples.csv")
    parser.add_argument("--model", default="alpamayo15")
    parser.add_argument("--bench", default="nuscenes")
    parser.add_argument("--nuscenes-root", help="Root of nuScenes dataset (overrides paddock.toml)")
    parser.add_argument("--version", default="v1.0-trainval")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        from paddock.config import SiteConfig

        cfg = SiteConfig.load()
        base = cfg.runs / args.bench / args.model
        candidates = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(f"No runs found in {base}")
        run_dir = candidates[0]
        print(f"Using most recent run: {run_dir.name}")

    csv_path = run_dir / "metrics_samples.csv"
    if not csv_path.exists():
        raise SystemExit(f"No metrics_samples.csv in {run_dir}\nRun export_metrics_csv.py on this run first.")

    _, rows = _load_csv(csv_path)
    print(f"Loaded {len(rows)} samples from {csv_path}")

    # Resolve nuScenes root
    if args.nuscenes_root:
        nuscenes_root = Path(args.nuscenes_root)
    else:
        from paddock.config import SiteConfig

        cfg = SiteConfig.load()
        dataset = cfg.datasets.get("nuscenes")
        if dataset is None:
            raise SystemExit("nuscenes dataset not configured; pass --nuscenes-root explicitly")
        nuscenes_root = dataset.resolve()

    descriptions = _load_scene_descriptions(nuscenes_root, args.version)
    print(f"Loaded {len(descriptions)} scene descriptions")

    # Tag each row
    by_scenario: dict[str, list] = {}
    by_weather: dict[str, list] = {}
    by_light: dict[str, list] = {}
    by_maneuver: dict[str, list] = {}

    missing = 0
    for row in rows:
        scene_name = row.get("native_scene") or row.get("group", "")
        desc = descriptions.get(scene_name, "")
        if not desc and scene_name not in descriptions:
            missing += 1

        sc = _tag_scenario(desc)
        w = _tag_weather(desc)
        l = _tag_light(desc)
        m = _tag_maneuver(row.get("native_command", ""))

        by_scenario.setdefault(sc, []).append(row)
        by_weather.setdefault(w, []).append(row)
        by_light.setdefault(l, []).append(row)
        by_maneuver.setdefault(m, []).append(row)

    if missing:
        print(f"  warning: {missing} rows had no matching scene description")

    _print_table(by_scenario, "Scenario type (from nuScenes descriptions)")
    _print_table(by_weather, "Weather")
    _print_table(by_light, "Lighting")
    _print_table(by_maneuver, "Command (maneuver)")


if __name__ == "__main__":
    main()
