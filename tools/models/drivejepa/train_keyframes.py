"""List the KITScenes train keyframes the drivejepa_rerank checkpoint was trained on.

Writes the `--rows` pickle that `paddock_candidates.py` dumps candidates for:
one ``{scene_id, split, keyframe_index}`` per keyframe, for every scene in
`rerank_train_scenes.txt`.

    python tools/models/drivejepa/train_keyframes.py -o train_samples.pkl

Keyframes are the devkit KITScenes loader's `valid_keyframes(num_future_frames=8,
stride=10)`: every 10th 10 Hz frame, from the first with 1.5 s of history (frame
15) to the last with 4 s of future. That loader imports navsim, which paddock's
env lacks, so the rule is restated below and must be kept equal to it. On the
373 listed scenes this reproduces the list the checkpoint was trained on:
5,711 keyframes, identical.

Runs in paddock's env; scenes are found under the site's `[datasets.kitscenes]`
tiers, in their `train` directory.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.adapters.benchmarks.kitscenes import loader as L  # noqa: E402
from paddock.config import SiteConfig  # noqa: E402

# kitscenes_adapter.valid_keyframes, restated: HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1)
# frames of history, NUM_FUTURE_FRAMES poses of future at HISTORY_STRIDE apart.
HISTORY_STRIDE = 5
NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 8
STRIDE = 10

SCENES = Path(__file__).with_name("rerank_train_scenes.txt")


def keyframes(num_frames: int) -> list[int]:
    first = HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1)
    last = num_frames - NUM_FUTURE_FRAMES * HISTORY_STRIDE - 1
    return list(range(first, last + 1, STRIDE)) if last >= first else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, default=SCENES, help="one scene id per line; # comments")
    args = parser.parse_args()

    ids = [
        line.strip()
        for line in args.scenes.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    roots = SiteConfig.load().datasets["kitscenes"].roots
    rows, missing = [], []
    for scene_id in ids:
        scene = next(
            (root / "train" / scene_id for root in roots if (root / "train" / scene_id).is_dir()), None
        )
        if scene is None:
            missing.append(scene_id)
            continue
        rows += [
            {"scene_id": scene_id, "split": "train", "keyframe_index": k}
            for k in keyframes(len(L.load_poses(scene)))
        ]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ids)} listed scenes are under no kitscenes tier's train/: "
            f"{', '.join(missing[:5])} ..."
        )
    args.output.write_bytes(pickle.dumps(rows))
    print(f"wrote {args.output}: {len(rows)} keyframes from {len(ids)} scenes")


if __name__ == "__main__":
    main()
