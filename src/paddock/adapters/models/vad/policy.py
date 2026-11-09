"""VAD-Base (HUST Vision Lab, ICCV 2023), the released nuScenes checkpoint, as a paddock Policy.

VAD is vectorised: a BEVFormer grid (200 x 200 cells, 30 m across and 60 m along, ResNet-50, six cameras
scaled to 1280x720) feeds agent queries with their own motion forecasts and map queries for lane dividers,
crossings and boundaries, and one ego query attends to both. It plans three trajectories, one per
command, as six 0.5 s offsets each, and the command picks which one is driven. What it is fed and how
its plan comes back is shared with UniAD and the Bench2Drive checkpoints (`bev_planners`); what follows
is VAD's own.

*Code.* The upstream repo (``hustvl/VAD``, pinned in repos.yaml), not HUGSIM's client fork ``VAD_SIM``,
whose client never calls ``model.eval()`` (dropout stays on, so a plan is a draw) and normalises the frames
the caffe way: means R/B-swapped and no division by the std, so the model reads inputs about 58 times its
training scale. Its config also turns ``use_can_bus`` off.

*Inputs*, as ``VADCustomNuScenesDataset.get_data_info`` and the config's ``test_pipeline`` build them: the
frames read as BGR, then ``img_norm_cfg`` (mean 123.675, 116.28, 103.53; std 58.395, 57.12, 57.375;
``to_rgb=True``), scaled by 0.8 to 1280x720 with ``lidar2img`` scaled alike and padded to a multiple of
32; the 18-float CAN bus with ``use_can_bus=True``; and ``ego_fut_cmd``, the command as a one-hot of
[right, left, straight]. VAD-Base reads no ego history (``ego_his_encoder`` and ``ego_lcf_feat_idx`` are
None), so neither is fed.

*Ground truth the forward reads.* VAD's ``forward_test`` scores its own detections, motion and plan, and
dereferences ground truth to do it. The adapter hands it no boxes and a future marked invalid
(`_gt_stubs`), so the metrics have nothing to compare and the planner metric is skipped. They work on a
copy of the predictions, which reach the plan untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp


class VADPolicy(bp.Mmdet3dPlannerPolicy):
    cameras = bp.NUSC_CAMERAS
    config_env = "VAD_CONFIG"
    checkpoint_env = "VAD_CHECKPOINT"
    #: The steps of ``test_pipeline`` that read nuScenes from disk: images, the lidar sweep (which
    #: ``forward_test`` never reads), the annotations and the two filters over them.
    drop_steps = (
        "LoadMultiViewImageFromFiles",
        "LoadPointsFromFile",
        "LoadAnnotations3D",
        "CustomObjectRangeFilter",
        "CustomObjectNameFilter",
    )
    #: What ``forward_test`` plans from; Bench2DriveZoo's ``inference_only_pipeline`` keeps the same two.
    collect_keys = ["img", "ego_fut_cmd"]

    def _extra_results(self, command: int) -> dict[str, Any]:
        # vad_nuscenes_converter writes the command as an int one-hot of [right, left, straight].
        return {"ego_fut_cmd": np.eye(3, dtype=np.int64)[command]}

    def _gt_stubs(self) -> dict[str, Any]:  # pragma: no cover - model environment only
        """No ground-truth boxes and an invalid future, in the nesting ``forward_test`` indexes.

        Boxes carry nine numbers (centre, size, yaw, velocity), as VAD's dataset builds them; an attribute
        row is 34 (future offsets, masks, goal, features, yaws). The ego history and local features are
        never read by VAD-Base but are indexed, and the ground-truth future is summed before the invalid
        flag skips its metric.
        """
        torch = self._torch
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        return {
            "gt_bboxes_3d": [[LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9)]],
            "gt_labels_3d": [[torch.zeros(0, dtype=torch.long)]],
            "gt_attr_labels": [[torch.zeros((0, 34))]],
            "fut_valid_flag": [torch.tensor([False])],
            "ego_his_trajs": [torch.zeros((1, 1, 2, 2))],
            "ego_fut_trajs": [torch.zeros((1, 1, 6, 2))],
            "ego_lcf_feat": [torch.zeros((1, 1, 1, 9))],
        }

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        # (3 commands, 6 steps, 2) offsets; VAD's own simple_test_pts picks by the command and sums so.
        offsets = output[0]["pts_bbox"]["ego_fut_preds"][int(np.argmax(results["ego_fut_cmd"]))]
        return offsets.cumsum(dim=-2).detach().cpu().numpy()

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        """VAD's temporal state as ``VAD.__init__`` leaves it: no previous grid, no ego pose to diff."""
        reset_vad_state(self.model)


def reset_vad_state(model) -> None:
    """Shared with the Bench2Drive checkpoint, whose code keeps the same dict plus a queue of past frames."""
    model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    if hasattr(model, "prev_frame_infos"):
        model.prev_frame_infos = []


POLICY = VADPolicy
