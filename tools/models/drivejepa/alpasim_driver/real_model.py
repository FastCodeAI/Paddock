# SPDX-License-Identifier: Apache-2.0
"""The REAL Drive-JEPA agent behind the planner's ``model_fn`` seam.

Mirrors :class:`fake_model.FakeDriveJEPA` exactly: ``forward(features) -> dict``
taking the numpy feature dict and returning ``proposals`` / ``pdm_score`` /
``trajectory``. The planner is unchanged -- swapping fake for real is a change
to the injected callable and nothing else.

**Runs only inside the Drive-JEPA conda env** (python 3.9 / torch 2.1 / cu121 /
mmcv-full 1.7.2). The AlpaSim venv is python 3.12 / torch 2.8 and cannot
construct this model; importing this module there fails at the torch/mmcv
import, which is the intended, loud outcome.

Three things upstream requires that are easy to miss, all handled here:

1. ``NAVSIM_EXP_ROOT`` must be set BEFORE the encoder is constructed -- the
   V-JEPA ViT-L weights are loaded from ``$NAVSIM_EXP_ROOT/Drive-JEPA-cache/``
   by an f-string at import/construction time
   (``bevformer/simple_image_encoder.py:25``).
2. The agent does ``np.load("./data/8192.npy")`` **relative to cwd**, so cwd
   must be ``navsim_v2`` while the agent is built (Paddock's model.yaml calls
   this "load-bearing").
3. ``.eval()`` is load-bearing, not hygiene: GridMask is only a no-op when not
   training (INTERFACE_DRIVEJEPA.md 6.3). A model left in train mode silently
   augments the input.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from typing import Any

import numpy as np

# --- where the weights and the upstream checkout live -----------------------
# The site says, through the same two variables paddock's drivejepa model.yaml
# sets: NAVSIM_EXP_ROOT ({data.navsim.exp}) and NAVSIM_DEVKIT_ROOT
# ({repo}/navsim_v2, the Drive-JEPA checkout repos.yaml pins). Unset, the
# placeholder below does not exist and _check_weights says so by name.
NAVSIM_EXP_ROOT = pathlib.Path(os.environ.get("NAVSIM_EXP_ROOT", "NAVSIM_EXP_ROOT-is-not-set"))
CACHE_DIR = NAVSIM_EXP_ROOT / "Drive-JEPA-cache"
DEFAULT_CHECKPOINT = CACHE_DIR / "drive_jepa_perception_based_agent_vitl_v2.ckpt"
ENCODER_WEIGHTS = CACHE_DIR / "vitl_merge_3dataset_e50.pt"

#: Verified 2026-09-04 against HF `LinhanWang/Drive-JEPA` AND against the copies
#: in a second running environment -- both sha256 matched exactly.
EXPECTED = {
    DEFAULT_CHECKPOINT.name: (
        3_735_352_913,
        "740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05",
    ),
    ENCODER_WEIGHTS.name: (
        5_127_748_765,
        "4649182770ef68f84a001780c6579435345948fd80f186fa3616ab078ced668f",
    ),
}

NAVSIM_V2 = pathlib.Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "NAVSIM_DEVKIT_ROOT-is-not-set"))

#: Keys the planner's Translation 1 produces (fake_model._validate).
REQUIRED_FEATURES = (
    "camera_feature_1",
    "camera_feature_2",
    "ego_status",
    "img_shape",
    "lidar2img",
)
#: Keys Translation 2 reads back.
RETURNED = ("proposals", "pdm_score", "trajectory")


class RealDriveJEPAError(RuntimeError):
    """Raised when the environment or weights cannot support the real model."""


def _check_weights() -> None:
    for path in (DEFAULT_CHECKPOINT, ENCODER_WEIGHTS):
        if not path.exists():
            raise RealDriveJEPAError(
                f"missing weights: {path}\n"
                "Fetch with: huggingface-cli download LinhanWang/Drive-JEPA "
                "--repo-type dataset --local-dir $NAVSIM_EXP_ROOT/Drive-JEPA-cache"
            )
        size, _ = EXPECTED[path.name]
        actual = path.stat().st_size
        if actual != size:
            raise RealDriveJEPAError(
                f"{path.name}: size {actual} != expected {size}. "
                "Wrong or truncated file -- refusing to load."
            )


class RealDriveJEPA:
    """Load the real agent once; call it per tick."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike | None = None,
        device: str = "cpu",
    ) -> None:
        _check_weights()
        if not NAVSIM_V2.is_dir():
            raise RealDriveJEPAError(
                f"Drive-JEPA checkout not found: {NAVSIM_V2}\n"
                "Set NAVSIM_DEVKIT_ROOT to <Drive-JEPA>/navsim_v2 at the sha repos.yaml pins"
            )
        # Must be set before the encoder is constructed.
        os.environ.setdefault("NAVSIM_EXP_ROOT", str(NAVSIM_EXP_ROOT))
        os.environ.setdefault("DRIVE_JEPA_MSDA", "auto")

        import torch  # imported here so the ImportError names the real cause

        self._torch = torch
        self._device = torch.device(device)
        ckpt = str(checkpoint_path or DEFAULT_CHECKPOINT)

        import sys

        if str(NAVSIM_V2) not in sys.path:
            sys.path.insert(0, str(NAVSIM_V2))

        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import (
            DriveJEPAAgent,
        )
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import (
            DriveJEPAConfig,
        )

        # np.load("./data/8192.npy") is relative to cwd -- build under navsim_v2.
        with _chdir(NAVSIM_V2):
            # lr is required by the constructor but is a training-only knob;
            # it never affects a forward pass. Any value works at inference.
            self._agent = DriveJEPAAgent(
                config=DriveJEPAConfig(), lr=1e-4, checkpoint_path=ckpt
            )
            self._agent.initialize()

        self._agent.to(self._device)
        # Load-bearing: GridMask is a no-op only when not training (6.3).
        self._agent.eval()
        if getattr(self._agent, "training", False):
            raise RealDriveJEPAError(".eval() did not clear training mode")

    # ------------------------------------------------------------------
    def forward(self, features: dict) -> dict:
        """numpy in, numpy out -- the seam the planner sees."""
        torch = self._torch
        missing = [k for k in REQUIRED_FEATURES if k not in features]
        if missing:
            raise RealDriveJEPAError(f"features missing keys: {missing}")

        tensors: dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.from_numpy(np.ascontiguousarray(value)).to(
                    self._device
                )
            else:
                tensors[key] = value

        with torch.no_grad():
            out = self._agent.forward(tensors)

        result = {}
        for key, value in out.items():
            result[key] = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else value
            )
        absent = [k for k in RETURNED if k not in result]
        if absent:
            raise RealDriveJEPAError(f"model returned no {absent}; got {list(result)}")
        return result

    __call__ = forward


@contextlib.contextmanager
def _chdir(path: pathlib.Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)
