"""Drive-JEPA with a learned re-ranker choosing among its 32 candidates.

Everything up to the 32 candidates is `drivejepa` unchanged: same checkpoint,
same input construction, same forward pass (`DriveJepaPolicy.forward`). Only
the last step differs. Drive-JEPA drives the candidate its NAVSIM-trained
scorer rates highest; this drives the one the re-ranker rates highest.

It is a different model from `drivejepa`, not a setting of it. The re-ranker
was fitted to KITScenes train, so its rows are adapted results and sit beside
the zero-shot ones, never in place of them. `model.yaml` declares both
training sets.

The re-ranker was trained on candidates this same path produced, with the
cameras cropped onto the OpenScene pinhole (`DRIVEJEPA_IMAGE_GEOMETRY`, set in
`model.yaml`). Running it on candidates made another way would score inputs it
never saw.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from paddock.adapters.models.drivejepa.policy import HISTORY_DT, DriveJepaPolicy
from paddock.adapters.models.drivejepa_rerank import ranker
from paddock.policy.protocol import Observation, Trajectory


class DriveJepaRerankPolicy(DriveJepaPolicy):
    def __init__(self, weights: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The checkpoint directory gallops wrote: reranker.npz + checkpoint.json.
        self.weights = weights or os.environ.get("DRIVEJEPA_RERANK_WEIGHTS")
        self.reranker: ranker.Reranker | None = None
        self.sidecar: dict[str, Any] = {}
        self.last: dict[str, Any] = {}

    def load(self) -> None:
        if not self.weights:
            raise RuntimeError(
                "no re-ranker: set DRIVEJEPA_RERANK_WEIGHTS (model.yaml supplies it from {weights})"
            )
        # Read before the 1.2 GB Drive-JEPA checkpoint, so a bad or drifted
        # weights file fails in seconds rather than after the model loads.
        self._load_reranker()
        super().load()

    def _load_reranker(self) -> None:
        self.reranker = ranker.load(self.weights)
        sidecar = Path(self.weights) / "checkpoint.json"
        self.sidecar = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    def infer(self, obs: Observation) -> Trajectory:
        if self.reranker is None:
            raise RuntimeError("load() must be called before infer()")
        output = self.forward(obs)
        proposals = output["proposals"][0].float().cpu().numpy()
        logits = output["pred_logit"][0].float().cpu().numpy()
        choice = self.reranker.choose(proposals, logits, obs.speed)
        self.last = {
            "chosen": choice,
            "model_pick": int(output["pdm_score"][0].float().argmax()),
        }
        return Trajectory(proposals[choice], dt=HISTORY_DT)

    def diagnostics(self) -> dict[str, Any]:
        """The base model's history check, plus which candidate was driven."""
        out = super().diagnostics()
        out.update(self.last)
        out["reranker"] = {
            key: self.sidecar.get(key) for key in ("train_run", "step", "train_git_sha", "selection_split")
        }
        return out


POLICY = DriveJepaRerankPolicy
