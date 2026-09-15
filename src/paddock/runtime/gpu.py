"""GPU headroom gating.

One card, shared with other users. Starting a 26 GB model into 2 GB of headroom
does not fail fast -- it fails several minutes into a checkpoint read, with a
CUDA OOM traceback that says nothing about the actual cause, which is that
somebody else has the GPU.

So paddock waits for the headroom a model declares in its `model.yaml`, and says
what it is waiting for.

Where there is more than one card, the one that matters is the one the model
will land on -- `CUDA_VISIBLE_DEVICES` -- not the first one nvidia-smi lists.
See `visible_device`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable


class GpuUnavailable(RuntimeError):
    """The card never had room within the timeout."""


def visible_device() -> int:
    """The physical device index a launched model will actually use.

    `CUDA_VISIBLE_DEVICES` is in `runtime.env.PASSTHROUGH`, so a model node does
    honour it -- which means gating must honour it too. It did not: this read
    `nvidia-smi`'s first row unconditionally, so on a two-card box a run pinned
    to card 1 waited on card 0's headroom. That is not a slow path, it is the
    wrong answer twice over: it blocks on a full card the run is not using, and
    it green-lights a launch into an empty one that is not there.

    Only the first entry matters. paddock starts one process per model and
    `wait_for` asks a single "is there room to load this", so a list selects
    which card that process lands on; a model sharding itself across several is
    outside what a single free-VRAM number can describe anyway.

    Falls back to device 0 for an unset, empty or non-numeric value (`MIG-...`
    UUIDs are legal there and are not an index we can look up by position).
    """
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    return int(raw) if raw.isdigit() else 0


def free_gb() -> float | None:
    """Free VRAM on the visible device, or None where there is no usable nvidia-smi.

    The whole table is queried and indexed rather than asking nvidia-smi for one
    device: `-i` takes the *physical* index, and re-deriving that from a
    `CUDA_VISIBLE_DEVICES` that may itself be a remapping is a second chance to
    be wrong. A device index past the end of the table returns None -- "cannot
    tell", which `wait_for` treats as "do not block" -- rather than silently
    reporting some other card's headroom.
    """
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
        return int(out.stdout.strip().splitlines()[visible_device()]) / 1024
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
