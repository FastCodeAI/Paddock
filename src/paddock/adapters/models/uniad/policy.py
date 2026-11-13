"""UniAD-Base (OpenDriveLab, CVPR 2023), the released nuScenes checkpoint, as a paddock Policy.

UniAD stacks tracking, online mapping, motion forecasting, occupancy prediction and planning on one
BEVFormer grid (200 x 200 cells of 0.512 m, ResNet-101 with DCNv2, six cameras at 1600x900). The planner
reads the ego track query, the motion queries and a learned embedding of the command, then nudges its six
waypoints off predicted occupancy with a small nonlinear optimiser (``use_col_optim``). What it is fed and
how its plan comes back is shared with VAD (`bev_planners`); what follows is UniAD's own.

*Code.* The upstream repo at its v1 release (``OpenDriveLab/UniAD`` ``main``, pinned in repos.yaml), not
HUGSIM's client fork ``UniAD_SIM``. That fork changes the model as well as its inputs: its motion head
keeps every class's queries instead of the vehicles' (``vehicle_mask = ones``), it skips the collision
optimiser when nothing is tracked, and its ``forward_test`` tests ``prev_frame_info['prev_bev']`` -- which
UniAD never sets -- so it zeroes the ego-motion delta on every frame and fuses the previous grid unaligned.
Its client also feeds RGB to a model trained on BGR and turns ``use_can_bus`` off.

*Inputs*, as ``NuScenesE2EDataset.get_data_info`` and the config's ``test_pipeline`` build them: BGR float
frames, ``img_norm_cfg`` caffe-style (mean 103.530, 116.280, 123.675; std 1; no channel swap); ``lidar2img``
per camera; the 18-float CAN bus with ``use_can_bus=True``, as configured; ``l2g_r_mat``/``l2g_t`` and the
timestamp, which the tracker uses to carry its queries forward; and the command, 0 right, 1 left,
2 forward.

*Ground truth the forward reads.* UniAD's ``forward_test`` computes its own map IoU and occupancy metrics,
and dereferences ground truth to do it. The adapter hands it zero lane masks and empty occupancy labels
(`_gt_stubs`): they reach the metrics, which are discarded, and -- with nothing tracked -- the shape of the
all-zero occupancy forecast, which leaves the collision optimiser nothing to avoid, as with real labels.

*The motion anchors.* The motion head reads ``motion_anchor_infos_mode6.pkl``, a release asset of UniAD
v1.0 that the repo does not track; its config names it relative to the working directory, so model.yaml
points it at the weights folder instead (``UNIAD_MOTION_ANCHORS``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp

#: model.yaml's setting for the motion head's anchor file (see the module docstring).
ANCHORS_ENV = "UNIAD_MOTION_ANCHORS"


class UniADPolicy(bp.Mmdet3dPlannerPolicy):
    cameras = bp.NUSC_CAMERAS
    config_env = "UNIAD_CONFIG"
    checkpoint_env = "UNIAD_CHECKPOINT"
    horizon_env = "UNIAD_HORIZON_S"
    #: The steps of ``test_pipeline`` that read nuScenes from disk: the images and the annotations.
    drop_steps = ("LoadMultiViewImageFromFilesInCeph", "LoadAnnotations3D_E2E", "GenerateOccFlowLabels")
    #: What ``forward_test`` plans from; Bench2DriveZoo's ``inference_only_pipeline`` keeps the same five.
    collect_keys = ["img", "timestamp", "l2g_r_mat", "l2g_t", "command"]

    def _configure(self, cfg) -> None:
        anchors = os.environ.get(ANCHORS_ENV)
        if not anchors or not Path(anchors).is_file():
            raise FileNotFoundError(
                f"{ANCHORS_ENV}={anchors!r}: UniAD's motion head needs motion_anchor_infos_mode6.pkl "
                f"(a UniAD v1.0 release asset; docs/models/uniad.md)"
            )
        cfg.model.motion_head.anchor_info_path = anchors

    def _gt_stubs(self) -> dict[str, Any]:  # pragma: no cover - model environment only
        """Zero ground truth in the shapes ``forward_test`` indexes (module docstring).

        One lane mask (the IoU code takes the last as drivable, the rest as lanes); seven occupancy frames,
        the current one and six future (``occ_n_future_max``), of which the head keeps five; nine validity
        flags (``occ_receptive_field`` 3 + 6). Canvas 200 x 200, the config's ``canvas_size``.
        """
        torch = self._torch
        return {
            "gt_lane_labels": [torch.zeros((1, 1), dtype=torch.long)],
            "gt_lane_masks": [torch.zeros((1, 1, 200, 200), dtype=torch.uint8)],
            "gt_segmentation": [torch.zeros((1, 7, 200, 200), dtype=torch.long)],
            "gt_instance": [torch.zeros((1, 7, 200, 200), dtype=torch.long)],
            "gt_occ_img_is_valid": [torch.ones((1, 9), dtype=torch.bool)],
        }

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        return output[0]["planning"]["result_planning"]["sdc_traj"][0].detach().cpu().numpy()

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        """Every temporal attribute back to its value in ``UniADTrack.__init__`` (or unset)."""
        reset_uniad_state(self.model)


def reset_uniad_state(model) -> None:
    """UniAD's temporal state as a freshly built model has it: no previous grid, no tracks, no ego motion.

    ``prev_frame_info`` holds the ego pose the CAN-bus delta is taken from; the tracker keeps its queries,
    the previous grid, the lidar pose and the time it last ran; ``track_base`` numbers new tracks.
    """
    model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    if hasattr(model, "prev_frame_infos"):
        model.prev_frame_infos = []
    model.test_track_instances = None
    model.l2g_r_mat = None
    model.l2g_t = None
    model.prev_bev = None
    model.timestamp = None
    model.scene_token = None
    model.track_base.clear()


POLICY = UniADPolicy
