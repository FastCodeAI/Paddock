"""The sample list, taken from the Drive-JEPA run so the comparison is row-for-row.

The reference run scored 223 keyframes across 140 scenes and pickled one dict per
sample.  Reading that pickle back gives the exact sample set *and* the reference
values to join against, which is strictly better than re-deriving the keyframes: it
stays correct even if a scene's frame count were to differ, and it carries
``human_poses`` along so the ground truth can be cross-checked rather than trusted.

``derive`` mode re-runs the same keyframe rule from the scenes themselves and exists
only so the self-test can prove the two agree.
"""

from __future__ import annotations

import csv
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _from_env(name: str, description: str) -> Path:
    """A path supplied by model.yaml, or a clear error naming what to set.

    These used to be one machine's absolute paths. Reading them from the
    environment lets paddock resolve them per site -- and makes an unset value
    fail as "KITSCENES_ROOT is not set" rather than as a missing directory
    nobody else ever had.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. It should be supplied by "
            f"adapters/models/alpamayo15/model.yaml ({description}). "
            f"Run through `paddock run`, or export it for a standalone script."
        )
    return Path(value)


def kitscenes_root() -> Path:
    """Parent of the dataset's data/ directory."""
    return _from_env("KITSCENES_ROOT", "the KITScenes dataset root")


def kitscenes_repo() -> Path:
    """The kitscenes devkit checkout, for its split lists."""
    return _from_env("KITSCENES_REPO", "the kitscenes repo, from {repos}/kitscenes")


def drive_jepa_pkl() -> Path:
    """The shared 223-keyframe sample list every KITScenes run is scored on."""
    return _from_env("KITSCENES_SAMPLE_LIST", "the shared sample list pickle")


SPLIT_SUBDIR = "kitscenes/split/generated_splits/default_geo_split_v1_0"
SPLIT_FILES = {
    "train": "train.txt",
    "val": "validation.txt",
    "test": "test.txt",
    "test-e2e": "test-e2e.txt",
    "overlap_train_val": "overlap_train_val.txt",
}


@dataclass
class Sample:
    """One keyframe to score, with the reference run's result attached."""

    scene_id: str
    split: str
    keyframe: int
    reference: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return (self.scene_id, self.keyframe)


def load_drive_jepa_rows(pkl: Path | None = None) -> list[dict[str, Any]]:
    """The reference run's per-sample dicts, as pickled by ``run_kitscenes_inference``."""
    pkl = pkl or drive_jepa_pkl()
    with open(pkl, "rb") as handle:
        return pickle.load(handle)


def samples_from_pickle(pkl: Path | None = None) -> list[Sample]:
    """The 223 reference samples, in scene order."""
    pkl = pkl or drive_jepa_pkl()
    samples = [
        Sample(
            scene_id=row["scene_id"],
            split=row["split"],
            keyframe=int(row["keyframe_index"]),
            reference={
                "ade": float(row["ade"]),
                "fde@1s": float(row.get("fde@1s", np.nan)),
                "fde@2s": float(row.get("fde@2s", np.nan)),
                "fde@3s": float(row.get("fde@3s", np.nan)),
                "ego_speed": float(row["ego_speed"]),
                "driving_command": np.asarray(row["driving_command"]).tolist(),
                "human_poses": np.asarray(row["human_poses"], dtype=np.float64),
                "predicted_poses": np.asarray(row["predicted_poses"], dtype=np.float64),
                "interval_length": float(row["interval_length"]),
            },
        )
        for row in load_drive_jepa_rows(pkl)
    ]
    samples.sort(key=lambda s: (s.scene_id, s.keyframe))
    return samples


def split_scene_ids(split: str, split_dir: Path | None = None) -> list[str]:
    """Scene UUIDs for *split*, from the toolkit's generated split lists."""
    split_dir = split_dir or (kitscenes_repo() / SPLIT_SUBDIR)
    text = (split_dir / SPLIT_FILES[split]).read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


def archive_sizes(root: Path | None = None) -> dict[str, int]:
    """Scene UUID -> tar size in bytes, from the dataset manifest.

    Used to sort scenes smallest-first and to refuse a download that would not fit --
    the largest archive in the val pool is 13.6 GB, several times the mean.
    """
    root = root or kitscenes_root()
    manifest = root / "data/sequence_archives.csv"
    with open(manifest, newline="") as handle:
        return {row["sequence_id"]: int(row["archive_size_bytes"]) for row in csv.DictReader(handle)}


def group_by_scene(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Samples bucketed per scene, so a downloaded scene is scored in one visit."""
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scene_id, []).append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda s: s.keyframe)
    return grouped


def order_scenes(scene_ids: list[str], sizes: dict[str, int]) -> list[str]:
    """Smallest archive first: results accumulate early and the 13.6 GB outliers land last."""
    return sorted(scene_ids, key=lambda scene: (sizes.get(scene, 0), scene))
