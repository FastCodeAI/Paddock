"""`alpamayo15`, with `model.expert`/`action_in_proj`/`action_out_proj` replaced
by the overlay a Gallops training script (`rl_training/sft_train_diffusion_head.py
--save-overlay`, not part of this repo -- see docs/models/alpamayo15_overfit.md
for the contract) wrote after overfitting on cached rollouts from two scenes
(0053, 0058) -- only the diffusion head; `model.vlm` is untouched and
identical to plain `alpamayo15`.

Registered as its own model rather than an env-toggled flag on `alpamayo15`
for the same reason `alpamayo15_teacher_sft` is its own entry rather than a
flag on `alpamayo15_teacher`: paddock's report/leaderboard keys a result by
model name, and a flag
some runs set would silently fold a materially different model into
`alpamayo15`'s own row.
"""

from __future__ import annotations

import os

from paddock.adapters.models.alpamayo15.policy import AlpamayoPolicy

#: Written by Gallops' `rl_training/sft_train_diffusion_head.py --save-overlay`
#: (not part of this repo -- see docs/models/alpamayo15_overfit.md).
OVERLAY_CHECKPOINT = os.environ.get("ALPAMAYO_OVERLAY_CHECKPOINT")


class OverfitAlpamayoPolicy(AlpamayoPolicy):
    def load(self) -> None:
        if not OVERLAY_CHECKPOINT:
            raise RuntimeError(
                "ALPAMAYO_OVERLAY_CHECKPOINT must be set to the overlay .pt file "
                "sft_train_diffusion_head.py's --save-overlay wrote"
            )
        # Read before the 20 GB Alpamayo + 8B Cosmos VLM load `super().load()`
        # pays for, so a bad or missing overlay file fails in seconds rather
        # than after the expensive checkpoint read -- same reasoning as
        # drivejepa_rerank's own weights check ahead of its model load.
        if not os.path.exists(OVERLAY_CHECKPOINT):
            raise RuntimeError(f"ALPAMAYO_OVERLAY_CHECKPOINT does not exist: {OVERLAY_CHECKPOINT}")

        super().load()

        import torch

        device = next(self.model.parameters()).device
        overlay = torch.load(OVERLAY_CHECKPOINT, map_location=device, weights_only=False)
        self.model.expert.load_state_dict(overlay["expert"])
        self.model.action_in_proj.load_state_dict(overlay["action_in_proj"])
        self.model.action_out_proj.load_state_dict(overlay["action_out_proj"])


POLICY = OverfitAlpamayoPolicy
