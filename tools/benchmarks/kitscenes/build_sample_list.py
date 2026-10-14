"""Build a KITScenes sample-list .pkl from the scenes on disk.

Each scene contributes one keyframe.  By default the keyframe is chosen near
the middle of the sequence (far enough from both ends that 48 history frames
and 6 future waypoints are available).  Pass --keyframe-offset to override.

Usage:
    python tools/benchmarks/kitscenes/build_sample_list.py \\
        --out /path/to/my_sample_list.pkl

    # Override which splits to include (default: val only, matching the
    # canonical kitscenes_e2e_final.pkl used by the benchmark table)
    python tools/benchmarks/kitscenes/build_sample_list.py \\
        --out /path/to/my_sample_list.pkl --splits val train

    # Dry-run: print the rows without writing anything
    python tools/benchmarks/kitscenes/build_sample_list.py --dry-run

The dataset root is read from paddock.toml ([datasets.kitscenes]).
Run `paddock doctor` if this script cannot find it.

Constraints enforced here (same as harness.py):
  - keyframe_index >= HISTORY_FRAMES (48): the harness feeds 48 preceding frames
    before the scored frame; earlier indices would under-run the sequence.
  - keyframe_index < n_frames - FUTURE_FRAMES (6): leaves room for 6 waypoints
    (3 s at 0.5 s/waypoint) to be read from the recorded future.
  - poses.txt must be present; scenes without it are skipped with a warning.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Allow running directly without `python -m`
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tools import paths  # noqa: E402

HISTORY_FRAMES = 48  # infer_sequence feeds this many frames before the keyframe
FUTURE_FRAMES = 6  # waypoints scored after the keyframe (3 s at 0.5 s/step)
KNOWN_SPLITS = ("val", "overlap_train_val", "train", "test")


def _n_frames(poses_txt: Path) -> int:
    return sum(1 for line in poses_txt.read_text().splitlines() if line.strip())


def build_rows(
    data_root: Path,
    splits: list[str],
    keyframe_offset: int | None,
) -> list[dict]:
    rows: list[dict] = []
    skipped = 0

    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"  split {split!r} not found at {split_dir} — skipping", file=sys.stderr)
            continue

        scenes = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for scene_dir in scenes:
            poses = scene_dir / "poses.txt"
            if not poses.exists():
                print(f"  warning: no poses.txt in {scene_dir} — skipping", file=sys.stderr)
                skipped += 1
                continue

            n = _n_frames(poses)
            min_idx = HISTORY_FRAMES
            max_idx = n - FUTURE_FRAMES - 1

            if max_idx < min_idx:
                print(
                    f"  warning: {scene_dir.name} has only {n} frames "
                    f"(need ≥ {HISTORY_FRAMES + FUTURE_FRAMES + 1}) — skipping",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            if keyframe_offset is not None:
                idx = keyframe_offset
                if not (min_idx <= idx <= max_idx):
                    print(
                        f"  warning: {scene_dir.name}: --keyframe-offset {idx} is out of "
                        f"range [{min_idx}, {max_idx}] for {n}-frame scene — skipping",
                        file=sys.stderr,
                    )
                    skipped += 1
                    continue
            else:
                # midpoint, clamped to the valid window
                idx = max(min_idx, min(max_idx, n // 2))

            rows.append(
                {
                    "scene_id": scene_dir.name,
                    "keyframe_index": idx,
                    "split": split,
                }
            )

    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="destination .pkl path (required unless --dry-run)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        metavar="SPLIT",
        help="which split directories to scan (default: val)",
    )
    parser.add_argument(
        "--keyframe-offset",
        type=int,
        default=None,
        metavar="N",
        help="fixed keyframe index to use for every scene instead of the midpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print rows to stdout without writing the .pkl",
    )
    args = parser.parse_args()

    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run is set")

    data_root = paths.kitscenes()
    print(f"scanning {data_root}")

    rows, skipped = build_rows(data_root, args.splits, args.keyframe_offset)

    if args.dry_run:
        for r in rows:
            print(f"  {r['split']}/{r['scene_id']}  keyframe={r['keyframe_index']}")
        print(f"\n{len(rows)} rows ({skipped} skipped)")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(rows))
    print(f"wrote {len(rows)} rows -> {args.out}  ({skipped} skipped)")


if __name__ == "__main__":
    main()
