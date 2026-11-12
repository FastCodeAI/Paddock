"""UniAD-Base trained on Bench2Drive (Bench2DriveZoo's released ``uniad_base_b2d.pth``) as a paddock Policy.

The same architecture as `uniad` with Bench2Drive's nine classes and six commands, trained by the
Bench2Drive authors on their CARLA dataset. It runs on Bench2DriveZoo's own code, which vendors mmcv,
mmdet and mmdet3d 0.17.1 as one ``mmcv`` package, and is fed as Bench2DriveZoo's CARLA agent
(``team_code/uniad_b2d_agent.py``) feeds it, apart from the two places the agent and the training data
disagree (`bev_planners.B2DPlannerPolicy`). What is shared with the other three checkpoints is in
`bev_planners`; what follows is this one's own.

*Inputs*: BGR frames after a JPEG quality-20 round trip, ``img_norm_cfg`` caffe-style (mean 103.530,
116.280, 123.675; std 1; no channel swap), padded to 1600x928; ``lidar2img`` from Bench2DriveZoo's virtual
lidar; the CAN bus in Bench2Drive's layout (`bev_planners.b2d_can_bus`) with ``use_can_bus=True``;
``l2g_r_mat``/``l2g_t`` untransposed; and one of six commands (`bev_planners.b2d_command`).

*Ground truth.* Bench2DriveZoo's heads already plan without it (the agent passes none), so none is stubbed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.adapters.models.uniad.policy import reset_uniad_state


class UniADB2DPolicy(bp.B2DPlannerPolicy):
    config_env = "UNIAD_B2D_CONFIG"
    checkpoint_env = "UNIAD_B2D_CHECKPOINT"
    horizon_env = "UNIAD_B2D_HORIZON_S"

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        return output[0]["planning"]["result_planning"]["sdc_traj"][0].detach().cpu().numpy()

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        reset_uniad_state(self.model)


POLICY = UniADB2DPolicy
