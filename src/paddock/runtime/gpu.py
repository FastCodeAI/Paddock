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
from collections.abc import Callable, Mapping


class GpuUnavailable(RuntimeError):
    """The card never had room within the timeout."""


def visible_device(env: Mapping[str, str] | None = None) -> int | None:
    """The physical device index a launched model will actually use.

    `env` is **the environment that model will be launched with**, which is not
    paddock's own: `model.yaml` may carry `env.CUDA_VISIBLE_DEVICES = "1"`, and
    `runtime.env.build` puts it there. Reading `os.environ` instead gated a
    card-1 model on card 0's headroom -- the wrong answer twice over, since it
    blocks on a card the run is not using and green-lights a launch into an
    empty one that is not there. That is the defect this function was added to
    fix, and it reproduced one level up. Callers with no launch in hand (
    `paddock doctor`) pass nothing and get paddock's own environment, which for
    them is the right one.

    Only the first entry matters. paddock starts one process per model and
    `wait_for` asks a single "is there room to load this", so a list selects
    which card that process lands on; a model sharding itself across several is
    outside what a single free-VRAM number can describe anyway.

    Returns None for an explicitly empty value -- `CUDA_VISIBLE_DEVICES=""` is
    how CUDA is told there is no device, and gating on card 0 for a model that
    has deliberately been pushed onto the CPU is a wait for headroom it will
    never use. Falls back to device 0 when the variable is unset or non-numeric
    (`MIG-...` UUIDs are legal there and are not an index we can look up by
    position).
    """
    source = os.environ if env is None else env
    if source.get("CUDA_VISIBLE_DEVICES", "__unset__").strip() == "":
        return None
    raw = (source.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    return int(raw) if raw.isdigit() else 0


def free_gb(env: Mapping[str, str] | None = None) -> float | None:
    """Free VRAM on the visible device, or None where there is none to read.

    None covers three cases, all of which `wait_for` treats the same way -- do
    not block: no usable nvidia-smi, a device index past the end of the table
    (better "cannot tell" than some other card's headroom), and a model pinned
    to no device at all.

    The whole table is queried and indexed rather than asking nvidia-smi for one
    device: `-i` takes the *physical* index, and re-deriving that from a
    `CUDA_VISIBLE_DEVICES` that may itself be a remapping is a second chance to
    be wrong.
    """
    device = visible_device(env)
    if device is None or not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return int(out.stdout.strip().splitlines()[device]) / 1024
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for(
    needed_gb: float,
    timeout_s: float = 7200,
    poll_s: float = 60.0,
    on_wait: Callable[[float, float], None] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Block until the card has `needed_gb` free, or raise.

    `env` is the environment the model is about to be launched with; see
    `visible_device` for why that is not paddock's own.

    A model declaring 0 GB (openpilot runs on tinygrad's CPU backend; VisionPilot
    on onnxruntime CPU) does not wait at all.
    """
    if needed_gb <= 0:
        return
    available = free_gb(env)
    if available is None:
        return  # no GPU to gate on; a CPU-only model still runs

    deadline = time.time() + timeout_s
    announced = False
    while (available := free_gb(env) or 0.0) < needed_gb:
        if time.time() > deadline:
            raise GpuUnavailable(
                f"need {needed_gb:.0f} GB but only {available:.1f} GB free after "
                f"{timeout_s / 60:.0f} min; the GPU is held by other processes"
            )
        if not announced and on_wait:
            on_wait(needed_gb, available)
            announced = True
        time.sleep(poll_s)
