"""GPU headroom gating.

One card, shared with other users. Starting a 26 GB model into 2 GB of headroom
does not fail fast -- it fails several minutes into a checkpoint read, with a
CUDA OOM traceback that says nothing about the actual cause, which is that
somebody else has the GPU.

So paddock waits for the headroom a model declares in its `model.yaml`, and says
what it is waiting for.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable


class GpuUnavailable(RuntimeError):
    """The card never had room within the timeout."""


def free_gb() -> float | None:
    """Free VRAM on device 0, or None where there is no usable nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for(
    needed_gb: float,
    timeout_s: float = 7200,
    poll_s: float = 60.0,
    on_wait: Callable[[float, float], None] | None = None,
) -> None:
    """Block until the card has `needed_gb` free, or raise.

    A model declaring 0 GB (openpilot runs on tinygrad's CPU backend; VisionPilot
    on onnxruntime CPU) does not wait at all.
    """
    if needed_gb <= 0:
        return
    available = free_gb()
    if available is None:
        return  # no GPU to gate on; a CPU-only model still runs

    deadline = time.time() + timeout_s
    announced = False
    while (available := free_gb() or 0.0) < needed_gb:
        if time.time() > deadline:
            raise GpuUnavailable(
                f"need {needed_gb:.0f} GB but only {available:.1f} GB free after "
                f"{timeout_s / 60:.0f} min; the GPU is held by other processes"
            )
        if not announced and on_wait:
            on_wait(needed_gb, available)
            announced = True
        time.sleep(poll_s)
