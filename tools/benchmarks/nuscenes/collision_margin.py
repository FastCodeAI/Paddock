"""Rescore a stored nuScenes run for signed collision margin, without re-running it.

The benchmark records collisions as a boolean, so a clean miss and a miss by four
centimetres are the same number. Over the 14 scenes neuroNCAP stages its
collisions in, that left more than half of a per-scene collision column at
exactly 0.0 -- nothing to correlate against the closed-loop outcome.

Nothing here needs a model. Inference is the expensive half of a nuScenes run
(19 to 108 minutes each) and its output is already on disk: every row of
`samples.jsonl` carries the plan the model produced and the sample token it was
produced for. This pairs those plans back up with the annotated boxes from the
dataset and measures the distance the original scoring pass threw away.

    python -m tools.benchmarks.nuscenes.collision_margin \
        --run .../nuscenes/drivejepa/<run_id>/samples.jsonl \
        --out margins/

Writes one `<model>.csv` per run: sample_id, scene, and the signed margin at each
horizon plus the worst over the plan. Positive is clearance in metres, negative
is penetration depth; `inf` means the scene had no annotated object to hit and is
written as an empty cell so it cannot be averaged in as a number.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tools import paths  # noqa: F401  (puts src/ on the path for the imports below)

# isort: split
from paddock.adapters.benchmarks.nuscenes import loader as L  # noqa: E402
from paddock.adapters.benchmarks.nuscenes.harness import N_STEPS, plan_headings  # noqa: E402

VAL_SCENES = Path(L.__file__).with_name("val_scenes.json")
HORIZONS = {"1s": 2, "2s": 4, "3s": 6}


def future_boxes_by_sample(root: Path, version: str, cache: Path | None) -> dict[str, list]:
    """`sample_id` -> the annotated boxes at each future step, in the plan's frame.

    The same construction the harness does at scoring time
    (`boxes_to_ego(cur.ego2world, f.boxes_world)` over the six future keyframes),
    rebuilt here so a stored run can be rescored. Keyframes without a full 3 s of
    contiguous future are skipped, exactly as `samples()` skips them, so the keys
    line up with what was scored.
    """
    scenes = json.loads(VAL_SCENES.read_text())["val"]
    index = L.load_or_build_index(root, version, scenes, cache)
    out: dict[str, list] = {}
    for frames in index:
        for i, cur in enumerate(frames):
            if i + N_STEPS >= len(frames):
                continue
            future = frames[i + 1 : i + 1 + N_STEPS]
            if future[-1].index_in_scene - cur.index_in_scene != N_STEPS:
                continue
            sample_id = f"nuscenes/val/{cur.scene}/{cur.index_in_scene}"
            out[sample_id] = [L.boxes_to_ego(cur.ego2world, f.boxes_world) for f in future]
    return out


def margins_for_plan(xy: np.ndarray, fut_boxes: list) -> dict[str, float]:
    """Signed margin at each horizon, and the worst point of the whole plan."""
    n = min(len(xy), len(fut_boxes), N_STEPS)
    yaws = plan_headings(xy[:n])
    per_step = [
        L.margin_to([xy[k, 0], xy[k, 1], yaws[k], L.EGO_LENGTH_M, L.EGO_WIDTH_M], fut_boxes[k])
        for k in range(n)
    ]
    out = {f"margin_{name}": per_step[k - 1] for name, k in HORIZONS.items() if k <= n}
    out["margin_min"] = min(per_step) if per_step else float("inf")
    return out


def _cell(value: float) -> str:
    """`inf` is written empty: a scene with nothing to hit has no measured distance."""
    return "" if value is None or math.isinf(value) else f"{value:.6f}"


def rescore(samples: Path, boxes: dict[str, list], out_csv: Path) -> tuple[int, int]:
    rows, skipped = 0, 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "scene", "margin_1s", "margin_2s", "margin_3s", "margin_min"]
    with samples.open() as fh, out_csv.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for line in fh:
            record = json.loads(line)
            fut = boxes.get(record["sample_id"])
            if fut is None or not record.get("prediction"):
                skipped += 1
                continue
            m = margins_for_plan(np.asarray(record["prediction"], dtype=np.float64), fut)
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "scene": record["native_id"]["scene"].replace("scene-", ""),
                    **{k: _cell(m.get(k)) for k in fields[2:]},
                }
            )
            rows += 1
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, type=Path, help="a samples.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="directory for the per-run csv")
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--cache", type=Path, default=None, help="index pickle; built if absent")
    args = ap.parse_args()

    root = args.dataset or paths.nuscenes()
    print(f"dataset {root} ({args.version}) -- building the val index, this is the slow part")
    boxes = future_boxes_by_sample(Path(root), args.version, args.cache)
    print(f"  {len(boxes)} scoreable keyframes indexed")

    for run in args.run:
        model = run.parent.parent.name
        rows, skipped = rescore(run, boxes, args.out / f"{model}.csv")
        note = f", {skipped} unmatched" if skipped else ""
        print(f"  {model:<18} {rows} samples rescored{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
