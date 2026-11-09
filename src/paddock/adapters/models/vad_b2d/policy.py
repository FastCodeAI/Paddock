"""VAD-Base trained on Bench2Drive (Bench2DriveZoo's released ``vad_b2d_base.pth``) as a paddock Policy.

The same architecture as `vad` with Bench2Drive's classes, trained by the Bench2Drive authors on their
CARLA dataset, planning six trajectories -- one per Bench2Drive command -- instead of three. It runs on
Bench2DriveZoo's own code and is fed as Bench2DriveZoo's CARLA agent (``team_code/vad_b2d_agent.py``)
feeds it, apart from the two places the agent and the training data disagree
(`bev_planners.B2DPlannerPolicy`). What is shared with the other three checkpoints is in `bev_planners`;
what follows is this one's own.

*Inputs*: BGR frames after a JPEG quality-20 round trip, then the config's ``inference_only_pipeline``:
``img_norm_cfg`` (mean 123.675, 116.28, 103.53; std 58.395, 57.12, 57.375; ``to_rgb=True``), padded to a
multiple of 32, scaled by 0.8 with ``lidar2img`` alike, padded again. That first pad is the agent's
pipeline, not the training one, which scales 1600x900 straight to 1280x720: the frame content is the same
to within a pixel row, and the extra rows are zero padding past the image. ``ego_fut_cmd`` is a one-hot
of the six commands (`bev_planners.b2d_command`), and selects which plan is driven. The agent also builds
``ego_lcf_feat``; this checkpoint never reads it (``ego_lcf_feat_idx=None``), so it is not built here.

*Ground truth.* Bench2DriveZoo's ``forward_test`` plans without it, so none is stubbed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.vad.policy import reset_vad_state


class VADB2DPolicy(bp.B2DPlannerPolicy):
    config_env = "VAD_B2D_CONFIG"
    checkpoint_env = "VAD_B2D_CHECKPOINT"

    def _extra_results(self, command: int) -> dict[str, Any]:
        # The agent's command_onehot: np.zeros(6) with a 1 at the command.
        onehot = np.zeros(6)
        onehot[command] = 1
        return {"ego_fut_cmd": onehot}

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        # (6 commands, 6 steps, 2) offsets; the agent sums them and takes the commanded one.
        offsets = output[0]["pts_bbox"]["ego_fut_preds"][int(results["command"])]
        return offsets.cumsum(dim=-2).detach().cpu().numpy()

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        reset_vad_state(self.model)


POLICY = VADB2DPolicy
