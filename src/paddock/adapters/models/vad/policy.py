"""VAD-Base (HUST Vision Lab, ICCV 2023), the released nuScenes checkpoint, as a paddock Policy.

VAD is vectorised: a BEVFormer grid (200 x 200 cells, 30 m across and 60 m along, ResNet-50, six cameras
scaled to 1280x720) feeds agent queries with their own motion forecasts and map queries for lane dividers,
crossings and boundaries, and one ego query attends to both. It plans three trajectories, one per
command, as six 0.5 s offsets each, and the command picks which one is driven. What it is fed and how
its plan comes back is shared with UniAD (`bev_planners`); what follows is VAD's own.

*Code.* The upstream repo (``hustvl/VAD``, pinned in repos.yaml), not HUGSIM's client fork ``VAD_SIM``,
whose client never calls ``model.eval()`` (dropout stays on, so a plan is a draw) and normalises HUGSIM's RGB
renders with the caffe mean meant for BGR, so red and blue are swapped. Its scale is right: the caffe
normalisation, with no division by a std, is the one the released weights were trained with. Its config
also turns ``use_can_bus`` off.

*Inputs*, as ``VADCustomNuScenesDataset.get_data_info`` and the config's ``test_pipeline`` build them,
with one change: the normalisation the released weights were trained with (`RELEASED_IMG_NORM`), not the
config's. The frames are read as BGR, then normalised caffe-style (mean 103.530, 116.280, 123.675; std 1;
``to_rgb=False``, so they stay BGR), scaled by 0.8 to 1280x720 with ``lidar2img`` scaled alike and padded
to a multiple of 32; the 18-float CAN bus with ``use_can_bus=True``; and ``ego_fut_cmd``, the command as a
one-hot of [right, left, straight]. VAD-Base reads no ego history (``ego_his_encoder`` and
``ego_lcf_feat_idx`` are None), so neither is fed.

*Normalisation.* The config at the pinned sha holds torchvision's ``img_norm_cfg`` (mean 123.675, 116.28,
103.53; std 58.395, 57.12, 57.375; ``to_rgb=True``), a later change (hustvl/VAD issue #9) the released
checkpoint was not trained with. VAD's own docs say so (``docs/train_eval.md``, "Reproduce results with
pre-trained weights"): to evaluate the released weights, set the caffe values above, and "If you use the new
``img_norm_cfg`` config, you will get wrong metric results". Fed the config's values, the model reads red
and blue swapped and about 1/58 of its training scale; every VAD result paddock made before this override
(`_configure`) was made that way.

*Ground truth the forward reads.* VAD's ``forward_test`` scores its own detections, motion and plan, and
dereferences ground truth to do it. The adapter hands it no boxes and a future marked invalid
(`_gt_stubs`), so the metrics have nothing to compare and the planner metric is skipped. They work on a
copy of the predictions, which reach the plan untouched.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp

#: How the released VAD-Base weights were trained to see a frame: BGR, the caffe mean, no division by a
#: std. VAD's docs at the pinned sha, ``docs/train_eval.md`` ("Reproduce results with pre-trained
#: weights"), prescribe exactly this ``img_norm_cfg`` for evaluating them; the configs' own, torchvision's
#: mean and std with ``to_rgb=True``, came later (hustvl/VAD issue #9) and the same page says it gives
#: "wrong metric results".
RELEASED_IMG_NORM = {"mean": [103.530, 116.280, 123.675], "std": [1.0, 1.0, 1.0], "to_rgb": False}
#: The pipeline step that applies ``img_norm_cfg`` (``projects/mmdet3d_plugin/datasets/pipelines``).
NORMALIZE_STEP = "NormalizeMultiviewImage"


class VADPolicy(bp.Mmdet3dPlannerPolicy):
    cameras = bp.NUSC_CAMERAS
    config_env = "VAD_CONFIG"
    checkpoint_env = "VAD_CHECKPOINT"
    horizon_env = "VAD_HORIZON_S"
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

    def _configure(self, cfg) -> None:
        """The released weights' normalisation in place of the config's (`RELEASED_IMG_NORM`).

        The config spreads ``img_norm_cfg`` into its pipelines' steps when it is read, so changing it as
        VAD's docs say means changing those steps; ``img_norm_cfg`` itself is set too, for anything that
        reads it from the config.
        """
        cfg.data.test.pipeline = with_normalisation(cfg.data.test.pipeline, RELEASED_IMG_NORM)
        cfg.img_norm_cfg = copy.deepcopy(RELEASED_IMG_NORM)

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


def with_normalisation(pipeline: list, norm: dict[str, Any]) -> list:
    """A copy of a config pipeline with every ``NormalizeMultiviewImage`` step set to ``norm``.

    Steps nested in a wrapper's ``transforms`` (``MultiScaleFlipAug3D``) are reached too; every other step,
    and the pipeline given, are left as they are. A pipeline with no such step raises: there would be
    nothing to set, and the model would read whatever the config does.
    """
    out = copy.deepcopy(list(pipeline))

    def visit(steps) -> int:
        count = 0
        for step in steps:
            if step.get("type") == NORMALIZE_STEP:
                step.update(copy.deepcopy(norm))
                count += 1
            count += visit(step.get("transforms", []))
        return count

    if not visit(out):
        raise ValueError(f"no {NORMALIZE_STEP} step in the pipeline, so no normalisation to set")
    return out


def reset_vad_state(model) -> None:
    """Clear what VAD carries from one frame to the next: the previous grid and the pose it was taken at."""
    model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
    if hasattr(model, "prev_frame_infos"):
        model.prev_frame_infos = []


POLICY = VADPolicy
