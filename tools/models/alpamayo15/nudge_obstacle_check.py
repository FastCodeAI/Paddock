#!/usr/bin/env python3
"""Does a predicted or ground-truth "nudge" actually correspond to a real
object being there?

decisions.py's lateral "nudge" label is a pure geometry proxy (0.3-1.0m of
final lateral offset) -- it can't tell a deliberate small avoidance maneuver
from ordinary lane-keeping/road-curvature noise (see the straight<->nudge
confusion in decision_confusion.py's overall table: a large fraction of
truly-straight samples get labeled "nudge"). This script adds the one signal
that actually distinguishes them: nuScenes' own annotated object boxes
(position + size only -- class is not available; see loader.py, which
discards it when building the collision-scoring box list). For every nudge
sample, it checks whether an annotated box sits ahead of the ego and on the
same side as the nudge -- i.e. whether there's plausibly something being
avoided.

Reuses the box-loading and world<->ego coordinate transforms the harness
itself already uses for collision scoring
(paddock.adapters.benchmarks.nuscenes.loader.build_index / .boxes_to_ego),
via a small cached keyframe index loader (below) -- no new geometry code, no
touching paddock core.

CAVEAT: this only tells you an object was annotated somewhere in the
relevant corridor at that keyframe, not that it caused the nudge, and object
class is unavailable (a parked car and a pedestrian look identical to this
check). The corridor size (CORRIDOR_AHEAD_M, MAX_LATERAL_M below) is a
first-pass guess, not validated against real data -- check the hit-rate
numbers against a handful of real frames before trusting them.

Requires decisions.csv (from decisions.py) in the run directory.

Usage:
    python tools/models/alpamayo15/nudge_obstacle_check.py [run_id]
    python tools/models/alpamayo15/nudge_obstacle_check.py [run_id] --side pred   # or gt, or both (default)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO_ROOT = DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DIR))

from paddock.adapters.benchmarks.nuscenes import loader as NL
from paddock.adapters.benchmarks.nuscenes.harness import VAL_SCENES
from paddock.config import SiteConfig

CORRIDOR_AHEAD_M = 25.0
MAX_LATERAL_M = 3.5  # roughly one lane width -- boxes farther than this aren't
# plausibly what a small (<1m) nudge is avoiding


def _load_nuscenes_index(cfg, scene_names: list[str]) -> dict[str, list]:
    """Keyframe index for the requested scenes (2 Hz), keyed by scene name.

    Loads ALL 150 val scenes so the cache key matches the one the harness
    builds, avoiding a 1.3 GB re-read every time this is called for a subset.
    """
    dataset = cfg.datasets.get("nuscenes")
    if dataset is None:
        raise SystemExit("nuscenes dataset not configured in paddock.toml")
    root, version = dataset.resolve(), (dataset.version or "v1.0-trainval")
    cache_path = REPO_ROOT / ".cache" / f"nuscenes_{version}_val_index.pkl"

    all_val_scenes = json.loads(VAL_SCENES.read_text())["val"]
    wanted = set(scene_names)
    print(
        f"  loading nuscenes keyframe index (all {len(all_val_scenes)} val scenes, "
        f"selecting {len(scene_names)}) ..."
    )
    all_scenes = NL.load_or_build_index(root, version, all_val_scenes, cache_path)
    return {frames[0].scene: frames for frames in all_scenes if frames and frames[0].scene in wanted}


def _resolve_run_dir(args, cfg) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id:
        return cfg.runs / args.bench / args.model / args.run_id
    cache = DIR / "baseline_run_id.txt"
    if not cache.exists():
        sys.exit("No run_id given and baseline_run_id.txt not found.")
    return cfg.runs / args.bench / args.model / cache.read_text().strip()


def _has_nearby_obstacle(boxes_ego, nudge_sign: float) -> bool:
    """boxes_ego: (N, 5) [x, y, yaw, length, width] in ego frame (x forward,
    y left). True if any box is ahead of the ego and on the same side as the
    nudge (sign of y matching sign of the nudge's lateral offset)."""
    for box in boxes_ego:
        x, y = float(box[0]), float(box[1])
        if 0.0 <= x <= CORRIDOR_AHEAD_M and 0.0 < abs(y) <= MAX_LATERAL_M and (y > 0) == (nudge_sign > 0):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--bench", default="nuscenes")
    ap.add_argument("--model", default="alpamayo15")
    ap.add_argument("--run-dir")
    ap.add_argument(
        "--side",
        choices=["pred", "gt", "both"],
        default="both",
        help="Check nudges from the predicted trajectory, ground truth, or both (default).",
    )
    args = ap.parse_args()

    cfg = SiteConfig.load()
    run_dir = _resolve_run_dir(args, cfg)

    csv_path = run_dir / "decisions.csv"
    if not csv_path.exists():
        sys.exit(f"decisions.csv not found in {run_dir} — run decisions.py first.")
    rows = list(csv.DictReader(csv_path.open()))

    sides = ["pred", "gt"] if args.side == "both" else [args.side]
    targets = {side: [r for r in rows if r[f"decision_lateral_{side}"] == "nudge"] for side in sides}

    scenes_needed = sorted({r["native_scene"] for side in sides for r in targets[side]})
    if not scenes_needed:
        sys.exit("No nudge samples found in decisions.csv for the requested side(s).")
    print(f"Loading keyframe index for {len(scenes_needed)} scene(s) with nudge samples...")
    scene_index = _load_nuscenes_index(cfg, scenes_needed)

    for side in sides:
        rows_side = targets[side]
        offset_key = f"lateral_offset_m_{side}"
        n_checked = 0
        n_with_obstacle = 0
        examples_no_obstacle = []
        for r in rows_side:
            frames = scene_index.get(r["native_scene"])
            if not frames:
                continue
            idx = int(r["native_index_in_scene"])
            kf = next((f for f in frames if f.index_in_scene == idx), None)
            if kf is None:
                continue
            boxes_ego = NL.boxes_to_ego(kf.ego2world, kf.boxes_world)
            nudge_sign = float(r[offset_key])
            has_obstacle = _has_nearby_obstacle(boxes_ego, nudge_sign)
            n_checked += 1
            if has_obstacle:
                n_with_obstacle += 1
            elif len(examples_no_obstacle) < 10:
                examples_no_obstacle.append(r["sample_id"])

        label = "predicted" if side == "pred" else "ground truth"
        print(f"\n[{label} nudges]  n={n_checked}")
        if n_checked:
            print(
                f"  with a plausible obstacle nearby: {n_with_obstacle} ({n_with_obstacle / n_checked:.0%})"
            )
            print(
                f"  with NO annotated object nearby:  {n_checked - n_with_obstacle} "
                f"({(n_checked - n_with_obstacle) / n_checked:.0%})"
            )
            if examples_no_obstacle:
                print(f"  example sample_ids with no nearby object: {', '.join(examples_no_obstacle)}")


if __name__ == "__main__":
    main()
