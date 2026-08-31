"""NV12 packing for openpilot's model input.

Rescued from `openpilot/eval_kitscenes/loader.py`, where it sat in the
open-loop harness and was imported across repos by the closed-loop node. It is
model-specific, so it belongs to the model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _ffmpeg() -> str:
    """Locate ffmpeg.

    openpilot's own venv ships one, and using it keeps the colour conversion
    identical to the binary the recorded runs used. Fall back to PATH, and say
    so clearly if neither exists rather than failing inside a subprocess pipe.
    """
    explicit = os.environ.get("PADDOCK_FFMPEG")
    if explicit:
        return explicit
    repo = os.environ.get("PADDOCK_REPO")
    if repo:
        candidate = os.path.join(repo, ".venv", "bin", "ffmpeg")
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg not found. Set PADDOCK_FFMPEG, or install ffmpeg into openpilot's venv "
        "(it is used to match the colour conversion of the real hevc -> nv12 path)."
    )


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
    if out.size != n * per:
        raise RuntimeError(f"expected {n * per} nv12 bytes from ffmpeg, got {out.size}")
    return out.reshape(n, per)


def pack_nv12(dst: np.ndarray, packed: np.ndarray, w: int, h: int, stride: int, y_height: int) -> None:
    """Copy tightly packed NV12 into openpilot's strided VisionBuf layout, in place.

    In place is not an optimisation: ModelState caches its device tensors by
    buffer address, so `dst` must be written, never reallocated.
    """
    y = packed[: w * h].reshape(h, w)
    uv = packed[w * h :].reshape(h // 2, w)
    dst[: y_height * stride].reshape(y_height, stride)[:h, :w] = y
    off = stride * y_height
    dst[off : off + (h // 2) * stride].reshape(h // 2, stride)[:, :w] = uv
