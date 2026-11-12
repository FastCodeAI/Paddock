"""Add Drive-JEPA's route command, `driving_command`, to a KITScenes keyframe list.

    python tools/benchmarks/kitscenes/driving_command.py \\
        --src <kitscenes_e2e_final.pkl> --out <kitscenes_e2e_final_gtcommand.pkl> \\
        --tier <KITScenes tier> [--tier <another tier> ...]

`[benchmarks.kitscenes] command_source = "gt"` gives each scored keyframe the
list's `driving_command`. A copy of the list without that column is refused in
preflight; this rebuilds it. The rule is Drive-JEPA's `--command gt`
(`adapters/models/drivejepa/kitscenes_adapter.py::_driving_command`), restated
because that module imports navsim, which paddock's interpreter lacks: take
where the car is 8 x 5 = 40 frames (4 s) after the keyframe, in the keyframe's
frame, and call it a turn if that is more than 5 m to one side.

On the 223-keyframe list it gives 21 left, 18 right and 184 straight, the counts
docs/benchmarks/kitscenes.md records for Drive-JEPA's row. Looking 50 frames
(5 s) ahead gives 47 turns instead. A run records the counts and a hash of the
commands (`run_config()["commands"]`), so a list built another way shows up as
a different measurement.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.kitscenes import loader as L

#: Drive-JEPA's `build_sample` default: 8 future frames, `HISTORY_STRIDE` 5 apart, at 10 Hz.
HORIZON_FRAMES = 8 * 5
#: Drive-JEPA's `TURN_THRESHOLD_M`.
TURN_THRESHOLD_M = 5.0
#: NAVSIM's one-hot order, `[left, straight, right, unknown]`, as the harness reads it.
LEFT, STRAIGHT, RIGHT = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]
SPLITS = ("val", "overlap_train_val", "train", "test")


def scene_dir(tiers: list[Path], scene_id: str, split: str) -> Path:
    """The first tier holding the scene's poses, its own split searched first."""
    for root in tiers:
        for name in (split, *SPLITS):
            if (root / name / scene_id / "poses.txt").is_file():
                return root / name / scene_id
    raise FileNotFoundError(f"no tier has {scene_id}/poses.txt: {', '.join(map(str, tiers))}")


def driving_command(poses: np.ndarray, keyframe: int) -> list[int]:
    lateral = L.to_relative_poses(poses[keyframe], poses[keyframe + HORIZON_FRAMES][None, :])[0, 1]
    if lateral > TURN_THRESHOLD_M:
        return LEFT
    if lateral < -TURN_THRESHOLD_M:
        return RIGHT
    return STRAIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="the keyframe list to read")
    parser.add_argument("--out", type=Path, required=True, help="where to write it with driving_command")
    parser.add_argument("--tier", type=Path, action="append", required=True, help="a KITScenes root")
    args = parser.parse_args()
    if args.out.resolve() == args.src.resolve():
        parser.error("--out is --src; write a new list, so the original stays as it was")

    rows = pickle.loads(args.src.read_bytes())
    poses: dict[str, np.ndarray] = {}
    out = []
    for row in rows:
        scene = row["scene_id"]
        if scene not in poses:
            poses[scene] = L.load_poses(scene_dir(args.tier, scene, row["split"]))
        out.append(row | {"driving_command": driving_command(poses[scene], int(row["keyframe_index"]))})

    counts = Counter(tuple(row["driving_command"]) for row in out)
    args.out.write_bytes(pickle.dumps(out))
    print(
        f"wrote {args.out}: {len(out)} keyframes, {counts[tuple(LEFT)]} left, "
        f"{counts[tuple(RIGHT)]} right, {counts[tuple(STRAIGHT)]} straight"
    )


if __name__ == "__main__":
    main()
