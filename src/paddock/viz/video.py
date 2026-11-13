"""Writing frames as a video the website can play.

Knows neither models nor benchmarks: it takes RGB frames and writes an .mp4.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def find_ffmpeg() -> str | None:
    """`PADDOCK_FFMPEG`, then PATH. None when neither has one."""
    named = os.environ.get("PADDOCK_FFMPEG")
    if named and Path(named).exists():
        return named
    return shutil.which("ffmpeg")


def encode(
    frames: Iterable[np.ndarray], path: Path, width: int, height: int, fps: float, ffmpeg: str
) -> None:
    """H.264 for the web: `+faststart` puts the index first, so a browser starts playing at once.

    Two encoder threads, because shakaal degrades above 90% CPU and nothing here is urgent.
    Written to a .partial file and renamed, so a failed encode leaves no playable half.
    """
    partial = path.with_suffix(".partial.mp4")
    cmd = [
        ffmpeg, "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-threads", "2", str(partial),
    ]  # fmt: skip
    # stderr to a file, not a pipe: nothing reads it until the end, and a full pipe would
    # block ffmpeg while this process blocks writing frames to it.
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            log.seek(0)
            raise RuntimeError(
                f"ffmpeg exited {code} writing {path}: {log.read().decode(errors='replace').strip()}"
            )
    partial.replace(path)
