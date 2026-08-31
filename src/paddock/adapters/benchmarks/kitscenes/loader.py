"""KITScenes -> openpilot ModelState inputs.

Ground truth and the sample list come from the Alpamayo/Drive-JEPA harness
unmodified, so the numbers are row-for-row comparable.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Their package is also called `eval_kitscenes`, so importing it normally would be
# shadowed by this one. Load the two modules we reuse straight off disk instead --
# reusing them unmodified is what keeps the numbers row-for-row comparable.
import importlib.util

ALPAMAYO_EVAL = Path("/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/eval_kitscenes")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_alpamayo_{name}", ALPAMAYO_EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _borrow(module: str, func: str):
    """Pull one function out of a reference module without importing the module.

    ``kitscenes_alpamayo_loader`` imports torch, which openpilot's venv has no
    reason to carry. Extracting the function's own source keeps it byte-identical
    to the reference implementation without dragging in the dependency, and
    without a copy that could silently drift.
    """
    import ast
    src = (ALPAMAYO_EVAL / f"{module}.py").read_text()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == func)
    ns: dict = {"np": np, "npt": np.typing}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"{module}.py", "exec"), ns)
    return ns[func]


_ref_metrics = _load("metrics")            # torch-free, import normally
displacement_errors = _ref_metrics.displacement_errors
to_relative_poses = _borrow("kitscenes_alpamayo_loader", "to_relative_poses")

# Scenes live in two places: the original 13 on the shakaal HDD, and everything
# fetched later under /mnt/p4 (NVMe, pruned to the 6% the eval reads). Search both.
KITSCENES_ROOTS = [Path(p) for p in os.environ.get(
    "KITSCENES_ROOTS",
    "/shakaal_hdd_1/workspace/naren/renesas/kitscenes_data/data:/mnt/p4/naren/kitscenes/data"
).split(":")]
KITSCENES_ROOT = KITSCENES_ROOTS[0]   # back-compat for callers that expect one
FFMPEG = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/.venv/bin/ffmpeg"
SRC_HZ = 10.0          # KITScenes native rate
GT_STEP = 5            # 5 frames @10Hz == the 0.5 s scoring grid


def scene_path(scene_id: str, split: str | None = None) -> Path:
    """Locate a scene across every configured root, ignoring which split it sits in."""
    splits = [split] if split else []
    splits += [s for s in ("val", "overlap_train_val", "train", "test") if s != split]
    for root in KITSCENES_ROOTS:
        for sp in splits:
            p = root / sp / scene_id
            if p.is_dir():
                return p
    raise FileNotFoundError(f"scene {scene_id} not found under {[str(r) for r in KITSCENES_ROOTS]}")


def scene_split(scene_id: str) -> str | None:
    for root in KITSCENES_ROOTS:
        for sp in ("val", "overlap_train_val", "train", "test"):
            if (root / sp / scene_id).is_dir():
                return sp
    return None


def load_poses(scene: Path) -> np.ndarray:
    """poses.txt (TUM: t x y z qx qy qz qw) -> (N, 3) of (x, y, yaw)."""
    raw = np.loadtxt(scene / "poses.txt")
    x, y = raw[:, 1], raw[:, 2]
    qx, qy, qz, qw = raw[:, 4], raw[:, 5], raw[:, 6], raw[:, 7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2))
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
        [FFMPEG, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"],
        input=frames.tobytes(), stdout=subprocess.PIPE, check=True)
    per = w * h * 3 // 2
    out = np.frombuffer(p.stdout, dtype=np.uint8)
    assert out.size == n * per, f"expected {n*per} nv12 bytes, got {out.size}"
    return out.reshape(n, per)


def pack_nv12(dst: np.ndarray, packed: np.ndarray, w: int, h: int,
              stride: int, y_height: int) -> None:
    """Copy tightly packed NV12 into openpilot's strided VisionBuf layout, in place."""
    y = packed[: w * h].reshape(h, w)
    uv = packed[w * h:].reshape(h // 2, w)
    dst[: y_height * stride].reshape(y_height, stride)[:h, :w] = y
    off = stride * y_height
    dst[off: off + (h // 2) * stride].reshape(h // 2, stride)[:, :w] = uv


def frame_indices(keyframe: int, history: int) -> list[int]:
    """Source frame indices for warm-up, oldest first, clamped at the scene start."""
    return [max(0, keyframe - history + i) for i in range(history + 1)]


def load_frames(scene: Path, idxs: list[int], sensor: str = "camera_ring_front") -> list[Image.Image]:
    return [Image.open(scene / sensor / f"{i:010d}.jpg") for i in idxs]
