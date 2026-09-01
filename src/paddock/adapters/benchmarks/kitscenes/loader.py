"""KITScenes -> openpilot ModelState inputs.

Ground truth and the sample list come from the Alpamayo/Drive-JEPA harness
unmodified, so the numbers are row-for-row comparable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

# Their package is also called `eval_kitscenes`, so importing it normally would be
# shadowed by this one. Load the two modules we reuse straight off disk instead --
# reusing them unmodified is what keeps the numbers row-for-row comparable.
from paddock.metrics.pose import to_relative_poses
from paddock.metrics.trajectory import displacement_errors

__all__ = ["to_relative_poses", "displacement_errors"]

# Both of the above used to be reached by loading a sibling repo's module off
# disk -- and, for to_relative_poses, by extracting its source with ast/exec to
# avoid importing torch. They now live in paddock.metrics, which is torch-free,
# so a normal import does the job and there is one definition rather than two
# that can drift.


def _roots() -> list[Path]:
    """Dataset tiers to search, fastest first.

    Supplied as a colon-separated search path by the model's model.yaml, from
    {data.kitscenes.tiers}. No default: a wrong guess here silently evaluates a
    different set of scenes than the one intended, which is far worse than a
    clear error.
    """
    raw = os.environ.get("KITSCENES_ROOTS")
    if not raw:
        raise RuntimeError(
            "KITSCENES_ROOTS is not set. It is supplied by the model's model.yaml "
            "from {data.kitscenes.tiers}; run through `paddock run`, or export it "
            "for a standalone script. See docs/data-setup.md."
        )
    return [Path(p) for p in raw.split(":") if p]


KITSCENES_ROOTS: list[Path] = _roots() if os.environ.get("KITSCENES_ROOTS") else []


def _ffmpeg() -> str:
    """ffmpeg, from PADDOCK_FFMPEG or PATH. See adapters/models/openpilot/nv12.py."""
    import shutil

    return os.environ.get("PADDOCK_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


SRC_HZ = 10.0  # KITScenes native rate
GT_STEP = 5  # 5 frames @10Hz == the 0.5 s scoring grid


def scene_path(scene_id: str, split: str | None = None) -> Path:
    """Locate a scene across every configured root, ignoring which split it sits in."""
    splits = [split] if split else []
    splits += [s for s in ("val", "overlap_train_val", "train", "test") if s != split]
    for root in _roots():
        for sp in splits:
            p = root / sp / scene_id
            if p.is_dir():
                return p
    raise FileNotFoundError(f"scene {scene_id} not found under {[str(r) for r in _roots()]}")


def scene_split(scene_id: str) -> str | None:
    for root in _roots():
        for sp in ("val", "overlap_train_val", "train", "test"):
            if (root / sp / scene_id).is_dir():
                return sp
    return None


def load_poses(scene: Path) -> np.ndarray:
    """poses.txt (TUM: t x y z qx qy qz qw) -> (N, 3) of (x, y, yaw)."""
    raw = np.loadtxt(scene / "poses.txt")
    x, y = raw[:, 1], raw[:, 2]
    qx, qy, qz, qw = raw[:, 4], raw[:, 5], raw[:, 6], raw[:, 7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return np.stack([x, y, yaw], axis=-1)


def ground_truth(poses: np.ndarray, keyframe: int, n: int = 8) -> np.ndarray:
    """(n, 3) future poses on the 0.5 s grid, in the keyframe's yaw frame."""
    idx = [keyframe + GT_STEP * (i + 1) for i in range(n)]
    idx = [i for i in idx if i < len(poses)]
    return to_relative_poses(poses[keyframe], poses[idx])


def ego_speed(poses: np.ndarray, keyframe: int) -> float:
    a = max(keyframe - 1, 0)
    b = min(keyframe + 1, len(poses) - 1)
    if b == a:
        return 0.0
    return float(np.linalg.norm(poses[b, :2] - poses[a, :2]) / ((b - a) / SRC_HZ))


def rgb_to_nv12_batch(frames: np.ndarray, w: int, h: int) -> np.ndarray:
    """(N,H,W,3) uint8 RGB -> (N, y+uv) tightly packed NV12, via ffmpeg.

    Going through ffmpeg rather than a hand-rolled matrix keeps the colour
    conversion identical to the path real openpilot frames take (hevc -> nv12).
    """
    n = len(frames)
    p = subprocess.run(
        [
            _ffmpeg(),
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "nv12",
            "pipe:1",
        ],
        input=frames.tobytes(),
        stdout=subprocess.PIPE,
        check=True,
    )
    per = w * h * 3 // 2
    out = np.frombuffer(p.stdout, dtype=np.uint8)
    assert out.size == n * per, f"expected {n * per} nv12 bytes, got {out.size}"
    return out.reshape(n, per)


def pack_nv12(dst: np.ndarray, packed: np.ndarray, w: int, h: int, stride: int, y_height: int) -> None:
    """Copy tightly packed NV12 into openpilot's strided VisionBuf layout, in place."""
    y = packed[: w * h].reshape(h, w)
    uv = packed[w * h :].reshape(h // 2, w)
    dst[: y_height * stride].reshape(y_height, stride)[:h, :w] = y
    off = stride * y_height
    dst[off : off + (h // 2) * stride].reshape(h // 2, stride)[:, :w] = uv


def frame_indices(keyframe: int, history: int) -> list[int]:
    """Source frame indices for warm-up, oldest first, clamped at the scene start."""
    return [max(0, keyframe - history + i) for i in range(history + 1)]


def frame_path(scene: Path, index: int, sensor: str = "camera_ring_front") -> Path:
    """Where one source frame lives, without opening it.

    Used where a frame only needs to be referred to -- a 24 MB decode per frame
    is worth avoiding when the consumer is a subprocess that can open it itself.
    """
    return scene / sensor / f"{index:010d}.jpg"


def load_frames(scene: Path, idxs: list[int], sensor: str = "camera_ring_front") -> list[Image.Image]:
    return [Image.open(frame_path(scene, i, sensor)) for i in idxs]
