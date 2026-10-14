#!/usr/bin/env python3
"""Detect decision flicker: within a scene, does the predicted decision flip
back and forth in a way the ground truth doesn't (e.g. keep_speed -> stop ->
keep_speed while the real drive never stopped)? A model that flickers is
unsafe even when each individual frame's error looks small — this is the
temporal-consistency category of the failure taxonomy.

Requires decisions.csv (from decisions.py) in the run directory, plus the raw
geometric features it stores per sample. Writes flicker_frames.json /
flicker_scenes.json in the same shape as worst_cases.py's worst_frames.json /
worst_scenes.json, so they plug directly into build_failure_sample_ids.py.

Why "excess transitions vs. ground truth" instead of a naive same-scene
flip-flop check: checked empirically against a real 5119-sample baseline,
counting any 3-frame reversal in the raw discrete decision label (e.g.
straight -> nudge -> straight, or turn_left -> not_turning -> turn_left)
flagged 90%+ of scenes. Almost all of that turned out to be either
threshold-boundary discretization noise (adjacent frames' overlapping 3s
windows crossing a fixed cutoff back and forth) or, worse, the model
correctly entering and exiting a single real turn/stop — which LOOKS like a
"reversion" under a naive check but is exactly the right behavior. The fix:
count decision transitions across the whole scene for both the prediction and
the ground truth, and only flag predicted transitions that have no matching
ground-truth transition nearby. A real turn/stop shows up in both sequences
and cancels out; genuine model indecision doesn't. On that same baseline this
produced a cleanly ranked list (top scene: 17 unmatched transitions; median
~2; only ~5% of scenes at zero) instead of an undifferentiated wall of flags.

Usage:
    python tools/analysis/decision_flicker.py [run_id]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from scenario_tags import load_scene_descriptions

from paddock.config import SiteConfig

# Same turn cutoff as decisions.py's classify_lateral/classify_longitudinal,
# collapsed to the two safety-relevant states per axis (see docstring above
# for why the full 5-way/4-way labels are too noisy for this check).
LATERAL_TURN_M = 2.0
HEADING_TURN_DEG = 20.0
STOP_SPEED_MPS = 0.5

# A predicted transition only counts as "excess" if no ground-truth
# transition happens within this many frames of it (0.5s per frame).
MATCH_TOLERANCE_FRAMES = 1


def _coarse_lateral(lat_offset_m: float, heading_change_deg: float) -> str:
    if abs(lat_offset_m) >= LATERAL_TURN_M or abs(heading_change_deg) >= HEADING_TURN_DEG:
        return "turn_left" if lat_offset_m > 0 else "turn_right"
    return "not_turning"


def _coarse_longitudinal(vf: float, min_speed: float) -> str:
    return "stop" if (min_speed < STOP_SPEED_MPS or vf < STOP_SPEED_MPS) else "moving"


def _transitions(labels: list[str]) -> list[int]:
    """Frame indices i where labels[i] != labels[i+1]."""
    return [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]


def _excess_transitions(pred_labels: list[str], gt_labels: list[str]) -> list[int]:
    gt_t = set(_transitions(gt_labels))
    return [
        i for i in _transitions(pred_labels) if not any(abs(i - g) <= MATCH_TOLERANCE_FRAMES for g in gt_t)
    ]


def _resolve_run_dir(args, cfg) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id:
        return cfg.runs / args.bench / args.model / args.run_id
    cache = DIR / "baseline_run_id.txt"
    if not cache.exists():
        sys.exit("No run_id given and baseline_run_id.txt not found.")
    return cfg.runs / args.bench / args.model / cache.read_text().strip()


def _to_float(v: str) -> float:
    return float(v) if v not in (None, "", "nan") else math.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument("--top-scenes", type=int, default=30)
    args = ap.parse_args()

    cfg = SiteConfig.load()
    run_dir = _resolve_run_dir(args, cfg)

    csv_path = run_dir / "decisions.csv"
    if not csv_path.exists():
        sys.exit(f"decisions.csv not found in {run_dir} — run decisions.py first.")
    rows = list(csv.DictReader(csv_path.open()))

    root = cfg.datasets["nuscenes"].resolve()
    version = cfg.datasets["nuscenes"].version or "v1.0-trainval"
    descriptions = load_scene_descriptions(root, version)

    by_scene: dict[str, list[dict]] = {}
    for r in rows:
        by_scene.setdefault(r["native_scene"], []).append(r)

    flicker_frames = []
    scene_summaries = []
    for scene, srows in by_scene.items():
        srows.sort(key=lambda r: int(r["native_index_in_scene"]))

        pred_lat = [
            _coarse_lateral(_to_float(r["lateral_offset_m_pred"]), _to_float(r["heading_change_deg_pred"]))
            for r in srows
        ]
        gt_lat = [
            _coarse_lateral(_to_float(r["lateral_offset_m_gt"]), _to_float(r["heading_change_deg_gt"]))
            for r in srows
        ]
        pred_lon = [
            _coarse_longitudinal(_to_float(r["vf_pred"]), _to_float(r["min_speed_pred"])) for r in srows
        ]
        gt_lon = [_coarse_longitudinal(_to_float(r["vf_gt"]), _to_float(r["min_speed_gt"])) for r in srows]

        n_flicker_total = 0
        for pred_labels, gt_labels, axis_name in (
            (pred_lat, gt_lat, "lateral"),
            (pred_lon, gt_lon, "longitudinal"),
        ):
            for i in _excess_transitions(pred_labels, gt_labels):
                n_flicker_total += 1
                r = srows[i + 1]  # the frame where the unmatched transition lands
                flicker_frames.append(
                    dict(
                        sample_id=r["sample_id"],
                        scene=scene,
                        index_in_scene=int(r["native_index_in_scene"]),
                        axis=axis_name,
                        pattern=f"{pred_labels[i]}→{pred_labels[i + 1]} (no matching GT transition nearby)",
                        description=descriptions.get(scene, ""),
                    )
                )
        scene_summaries.append(
            dict(
                scene=scene,
                description=descriptions.get(scene, ""),
                n_frames=len(srows),
                n_flicker_points=n_flicker_total,
                flicker_rate=n_flicker_total / len(srows) if srows else 0.0,
            )
        )

    flicker_frames.sort(key=lambda r: (r["scene"], r["index_in_scene"]))
    scene_summaries.sort(key=lambda r: -r["n_flicker_points"])
    worst_scenes = scene_summaries[: args.top_scenes]

    out_frames = run_dir / "flicker_frames.json"
    out_frames.write_text(json.dumps(flicker_frames, indent=2))
    out_scenes = run_dir / "flicker_scenes.json"
    out_scenes.write_text(json.dumps(worst_scenes, indent=2))

    print(f"Run: {run_dir.name}  (n={len(rows)} decision-labeled samples across {len(by_scene)} scenes)")
    print(f"Flicker points found: {len(flicker_frames)}")
    print(f"\nTop {len(worst_scenes)} scenes by excess-transition count:")
    print(f"  {'scene':<14} | {'n':>4} | {'flicker':>7} | {'rate':>6} | description")
    print(f"  {'-' * 14}-+------+---------+--------+" + "-" * 30)
    for r in worst_scenes[:25]:
        desc_short = (r["description"] or "")[:40]
        print(
            f"  {r['scene']:<14} | {r['n_frames']:>4} | {r['n_flicker_points']:>7} | "
            f"{r['flicker_rate']:>6.1%} | {desc_short}"
        )
    print(f"\n  ... written {out_frames} and {out_scenes}")


if __name__ == "__main__":
    main()
