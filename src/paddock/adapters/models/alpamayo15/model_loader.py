"""Load Alpamayo 1.5 from local weights, without touching gated HuggingFace repos.

Two things make this less trivial than ``Alpamayo1_5.from_pretrained(path)``.

*Gating.*  ``config.json`` names ``nvidia/Cosmos-Reason2-8B`` as ``vlm_name_or_path``;
that repo is gated, and the value is consumed *during* ``ReasoningVLAConfig.__init__``
(``_initialize_vlm_config`` -> ``_build_processor``) and again in ``_build_tokenizer``
and ``_initialize_qwenvl3_vlm``.  ``PretrainedConfig.from_dict`` applies leftover
kwargs only *after* ``cls(**config_dict)``, so passing ``vlm_name_or_path=`` to
``from_pretrained`` is too late.  The config has to be constructed by hand with the
local path already in the dict.  Only the Qwen3VL *config* and *tokenizer* are read
from there -- the VLM weights come from the Alpamayo checkpoint itself.

*Attention backend.*  The checkpoint requests ``flash_attention_2``, and
``ReasoningVLAConfig.__init__`` defaults to it whenever the kwarg is ``None``.  The
local CUDA toolkit is 12.0, too old to emit sm_120, so ``flash-attn`` is deliberately
not installed and SDPA must be selected explicitly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from alpamayo1_5 import helper
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def _weights(env_var: str, default_name: str) -> Path:
    """Where a checkpoint lives here.

    Supplied by model.yaml from {weights}; the default is a bare directory name
    so a missing setting fails as "Alpamayo-1.5-10B not found" rather than
    pointing confidently at one machine's disk.
    """
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)
    root = os.environ.get("PADDOCK_WEIGHTS")
    return Path(root) / default_name if root else Path(default_name)


CKPT = _weights("ALPAMAYO_CHECKPOINT", "Alpamayo-1.5-10B")
COSMOS = _weights("ALPAMAYO_COSMOS", "Cosmos-Reason2-8B")
#: The Qwen3-VL-2B-Instruct processor files (image resize, chat template). Upstream
#: loads them by hub name (helper.BASE_PROCESSOR_NAME), which needs a HuggingFace cache
#: and, without HF_HUB_OFFLINE, a network check that can pull a newer revision.
PROCESSOR = _weights("ALPAMAYO_PROCESSOR", "Qwen3-VL-2B-Instruct")


def get_processor(tokenizer: Any, local: Path = PROCESSOR) -> Any:
    """helper.get_processor, read from *local* when that directory exists.

    Same files, same resize limits; only where they are read from changes. A site that
    keeps the processor in its HuggingFace cache instead gets upstream's call unchanged.
    """
    if not local.is_dir():
        return helper.get_processor(tokenizer)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(local), min_pixels=helper.MIN_PIXELS, max_pixels=helper.MAX_PIXELS
    )
    processor.tokenizer = tokenizer
    return processor


def build_config(ckpt: Path = CKPT, cosmos: Path = COSMOS, attn: str = "sdpa") -> Alpamayo1_5Config:
    """Config with the gated VLM path and the attention backend pinned locally."""
    cfg_dict: dict[str, Any] = json.loads((ckpt / "config.json").read_text())
    cfg_dict.pop("model_type", None)  # set by the class, rejected as a kwarg
    cfg_dict["vlm_name_or_path"] = str(cosmos)
    cfg_dict["attn_implementation"] = attn
    return Alpamayo1_5Config(**cfg_dict)


def reserve_vram(gigabytes: float, chunk_mb: int = 512) -> list[torch.Tensor]:
    """Claim VRAM now so a slow load cannot lose it to another process.

    Reading the 22 GB checkpoint off disk takes ~12 minutes, during which we hold no
    GPU memory at all; on a shared GPU another user can easily take it in that window
    and the ``.to("cuda")`` at the end then fails.  Allocating first and releasing the
    references just before the transfer closes that gap: PyTorch's caching allocator
    keeps freed blocks in this process's pool rather than returning them to the driver,
    so the model's own allocations are served from what we already hold.

    Chunks rather than one slab because the allocator splits cached large-pool blocks
    to satisfy smaller requests, and the model asks for thousands of varied sizes.
    """
    chunks: list[torch.Tensor] = []
    elements = chunk_mb * 1024 * 1024
    for _ in range(int(gigabytes * 1024 / chunk_mb)):
        try:
            chunks.append(torch.empty(elements, dtype=torch.uint8, device="cuda"))
        except torch.OutOfMemoryError:
            break
    if len(chunks) * chunk_mb / 1024 < gigabytes:
        # Hand back everything, including to the driver -- holding a partial
        # reservation would block both us and whoever else is waiting.
        del chunks
        torch.cuda.empty_cache()
        return []
    return chunks


def acquire_vram(gigabytes: float, timeout_s: float, poll_s: float = 30.0) -> list[torch.Tensor]:
    """Poll until the full reservation succeeds.

    Checking free memory and then allocating is a race on a shared GPU -- the check can
    pass and the memory be gone a moment later.  Attempting the allocation *is* the
    check, so this retries the whole grab rather than trusting a prior reading.
    """
    deadline = time.perf_counter() + timeout_s
    announced = False
    while True:
        chunks = reserve_vram(gigabytes)
        if chunks:
            print(f"reserved {gigabytes:.1f} GB of VRAM", flush=True)
            return chunks
        if time.perf_counter() > deadline:
            free, _ = torch.cuda.mem_get_info()
            raise SystemExit(
                f"could not reserve {gigabytes:.1f} GB within {timeout_s / 60:.0f} min "
                f"({free / 1e9:.1f} GB free; the GPU is shared with other users)"
            )
        if not announced:
            free, _ = torch.cuda.mem_get_info()
            print(
                f"waiting for {gigabytes:.1f} GB of VRAM ({free / 1e9:.1f} GB free, "
                f"other users hold the rest)",
                flush=True,
            )
            announced = True
        time.sleep(poll_s)


def load_model(
    ckpt: Path = CKPT,
    cosmos: Path = COSMOS,
    attn: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    reserve_gb: float | None = None,
    reserve_timeout_s: float = 7200.0,
) -> tuple[Alpamayo1_5, Any]:
    """Return ``(model, processor)`` ready for inference, with the backend verified."""
    reservation = acquire_vram(reserve_gb, reserve_timeout_s) if reserve_gb and device == "cuda" else None

    config = build_config(ckpt, cosmos, attn)
    model = Alpamayo1_5.from_pretrained(ckpt, config=config, dtype=dtype)

    # Hand the reservation back to our own allocator pool, then move in.  No
    # empty_cache() here -- that would return the blocks to the driver and undo it.
    del reservation
    model = model.to(device).eval()

    # A silent fall-back to flash_attention_2 would fail at the first forward pass on
    # sm_120, so make the mismatch loud here instead.
    assert model.config.attn_implementation == attn, (
        f"attn_implementation is {model.config.attn_implementation!r}, expected {attn!r}"
    )
    if attn != "flash_attention_2":
        text_attn = model.vlm.config.text_config._attn_implementation
        assert text_attn != "flash_attention_2", f"VLM text config still on {text_attn!r}"

    processor = get_processor(model.tokenizer)
    return model, processor


def main() -> None:
    """Smoke test: load the model and report what came up."""
    try:
        import flash_attn  # noqa: F401

        raise SystemExit("flash_attn is installed; this venv is meant to run without it")
    except ImportError:
        pass

    capability = torch.cuda.get_device_capability(0)
    print(f"device      : {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}")

    model, processor = load_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters  : {params / 1e9:.2f} B")
    print(f"attention   : {model.config.attn_implementation}")
    print(f"vlm path    : {model.config.vlm_name_or_path}")
    print(f"peak VRAM   : {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"processor   : {type(processor).__name__}")


if __name__ == "__main__":
    main()
