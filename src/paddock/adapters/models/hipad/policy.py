"""HiP-AD (Nullmax, ICCV 2025), the released Bench2Drive checkpoint, as a paddock Policy.

HiP-AD is a Sparse4D-style planner: six cameras (ResNet-50, 640x352) feed one decoder that updates agent,
map and planning queries together, and its planning queries read image features around their own
waypoints (planning deformable attention). It plans several kinds of trajectory at once -- by time at 5 Hz
and 2 Hz, by distance every 2 m and 5 m, and by time again with a speed-matched mode chosen -- each six
points in its lidar frame.

*Code.* ``nullmax-vision/HiP-AD`` at the pinned sha, built and fed as its own CARLA agent does
(``bench2drive/leaderboard/team_code/hipad_b2d_agent.py``): the config's ``inference_only_pipeline``
without its file loader, the model in fp16 where the config asks for it, its plugin imported by
``plugin_dir``. What the agent feeds is reproduced in `inputs`; the frames are its own: JPEG at quality 20
and back (`inputs.JPEG_QUALITY`), BGR, at 1600x900.

*Rate.* The agent runs the model on every CARLA tick (20 Hz) with ``with_close_loop`` on, which keeps ten
copies of the temporal memory and steps each every tenth call, so each still sees frames 0.5 s apart, as it
trained (``sparse_onedecoder.py``, ``init_instance_bank_list``). Paddock runs HiP-AD's own open-loop mode
instead -- one memory, stepped every 0.5 s -- through the same machinery as UniAD and VAD
(`bev_planners.BevPlannerPolicy`: ``planning_hz: 2``, the plan held between steps). Each memory sees the
same 0.5 s steps either way; the plan is replaced every 0.5 s instead of every 0.05 s.

*Plan.* ``plan_speed_2hz``: six points 0.5 s apart (3 s), the mode matched to speed, the 2 Hz twin of the
``plan_speed_5hz`` the agent's PID controller follows (six points 0.2 s apart, 1.2 s). Paddock's wire
carries 0.5 s points, and its closed-loop clients each drive a plan with their own controller.

*Not the agent's.* The target point where a source has no route (`inputs.command_target_point`); lane
changes recovered from ``nav_lane_change``, because paddock's Bench2Drive client sends them as STRAIGHT.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

from paddock.adapters.models.hipad import inputs as hi
from paddock.adapters.models.uniad import bev_planners as bp
from paddock.policy.protocol import Observation

#: The config's absolute ``project_dir`` at the pinned sha, the authors' own path; its ``data/kmeans``
#: anchors ship in the repo, so the prefix is moved to the checkout (`repoint`).
AUTHORS_PROJECT_DIR = "/opt/data/private/project/HiP-AD"
#: The plan paddock drives (module docstring, *Plan*).
PLAN_KEY = "plan_speed_2hz"


def repoint(value: Any, old: str, new: str) -> Any:
    """`value` with every string under it that starts with `old` starting with `new` instead."""
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old) else value
    if isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = repoint(value[key], old, new)
        return value
    if isinstance(value, list):
        return [repoint(v, old, new) for v in value]
    if isinstance(value, tuple):
        return tuple(repoint(v, old, new) for v in value)
    return value


class HiPADPolicy(bp.BevPlannerPolicy):
    cameras = hi.CAMERAS
    config_env = "HIPAD_CONFIG"
    checkpoint_env = "HIPAD_CHECKPOINT"
    horizon_env = "HIPAD_HORIZON_S"
    nominal_lidar2ego = hi.LIDAR2EGO

    def _begin_episode(self) -> None:
        super()._begin_episode()
        self.route = hi.RouteTracker()
        self._t0_us: int | None = None

    def _prepare_frame(self, obs: Observation):
        # HiP-AD trained in one lidar mount only (`inputs.LIDAR2EGO`): a source's own lidar is not used.
        return super()._prepare_frame(dataclasses.replace(obs, lidar2ego=None))

    # -- the model's code base ----------------------------------------------

    def _build(self) -> None:  # pragma: no cover - model environment only
        import importlib

        import cv2
        import torch
        from mmcv import Config
        from mmcv.parallel import DataContainer
        from mmcv.parallel.collate import collate
        from mmcv.runner import load_checkpoint, wrap_fp16_model

        repo = os.environ.get("HIPAD_REPO") or os.getcwd()
        cfg = Config.fromfile(self.config)
        # In place: mmcv's Config turns attribute assignment into a config key.
        repoint(cfg._cfg_dict, AUTHORS_PROJECT_DIR, repo)
        cfg.model.head.onedecoder_head.with_close_loop = False  # module docstring, *Rate*
        if hasattr(cfg.model.head, "evaluate_bench2dive"):
            cfg.model.head.evaluate_bench2dive = False
        # The backbone's ImageNet start is for training; the checkpoint below holds every weight.
        cfg.model.img_backbone.pretrained = None
        module = bp.plugin_module(cfg)
        if module:
            importlib.import_module(module)
        from mmdet.datasets.pipelines import Compose
        from mmdet.models import build_detector

        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, self.checkpoint, map_location="cpu", strict=True)
        model.cuda()
        self.model = model.eval()
        steps = [
            step
            for step in cfg.inference_only_pipeline
            if step["type"] not in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles")
        ]
        self.pipeline = Compose(steps)
        aug = cfg.data_aug_conf
        self.aug = hi.aug_config((aug["W"], aug["H"]), tuple(aug["final_dim"][::-1]))
        self._torch, self._collate, self._dc, self._cv2 = torch, collate, DataContainer, cv2

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        reset_hipad_state(self.model)

    def _frame(self, rgb: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """One camera as the agent hands it on: JPEG at quality 20 and back, then BGR (``tick``,
        ``run_step``). The agent encodes the RGB array as cv2 reads arrays, BGR, and decodes it the same."""
        cv2 = self._cv2
        ok, encoded = cv2.imencode(
            ".jpg", np.ascontiguousarray(rgb), [int(cv2.IMWRITE_JPEG_QUALITY), hi.JPEG_QUALITY]
        )
        if not ok:
            raise RuntimeError("cv2 could not JPEG-encode a frame")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR), k

    def _results(self, obs: Observation, lidar2ego: np.ndarray) -> dict[str, Any]:  # pragma: no cover
        """What the agent's ``run_step`` hands its pipeline, from a paddock observation."""
        views = self._views(obs, self._frame)
        images, projections, lidar2cams = [], [], []
        for camera, (image, k) in zip(self.cameras, views):  # noqa: B905 - runs on Python 3.9
            images.append(image)
            projections.append(bp.lidar2img(k, obs.camera2ego[camera], lidar2ego))
            lidar2cams.append(np.linalg.inv(obs.camera2ego[camera]) @ lidar2ego)
        index = hi.command_index(obs.command, obs.nav_lane_change, obs.nav_lane_change_m)
        target = hi.target_record(
            obs.route_xy, obs.route_timestamp_us, obs.timestamp_us, obs.ego2world, index, self.route
        )
        if self._t0_us is None:
            self._t0_us = int(obs.timestamp_us)
        self.last.update(
            command=hi.COMMAND_NAMES[index],
            target_source=target["source"],
            target_point=np.round(target["target_point"], 3).tolist(),
        )
        lidar2global = np.asarray(obs.ego2world, dtype=np.float64) @ lidar2ego
        return {
            "img": images,
            "folder": "",
            "scene_token": self.scene_token,
            "frame_idx": self._prepared,
            # Seconds from the episode's first frame, as the agent's ``step / frame_rate``: the memory reads
            # only differences, and an absolute clock (nuScenes, ~1.5e9 s) would lose them in float32.
            "timestamp": (int(obs.timestamp_us) - self._t0_us) / 1e6,
            "lidar2img": projections,
            "lidar2cam": np.stack(lidar2cams, axis=0),
            "aug_config": dict(self.aug),
            "custom_status": np.zeros(
                6, dtype=np.float32
            ),  # set by the agent, read by no layer of this config
            "command": index,
            "gt_ego_fut_cmd": hi.one_hot(index),
            "target_point": target["target_point"],
            "l2g_r_mat": lidar2global[:3, :3],
            "l2g_t": lidar2global[:3, 3],
            "lidar2global": lidar2global,
            "image_wh": np.array([hi.FINAL_SIZE for _ in self.cameras], dtype=np.float32),
        }

    def _prepare(self, results: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - model env only
        """The pipeline and collate, then each value unwrapped as the agent unwraps it (still on the CPU)."""
        batch = self._collate([self.pipeline(results)], samples_per_gpu=1)
        out = {}
        for key, value in batch.items():
            if isinstance(value, self._dc):
                out[key] = value.data[0]
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], self._dc):
                out[key] = value[0].data
            else:
                out[key] = value
        return out

    def _run(self, batch: dict[str, Any], results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        device = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.model(
                img=device["img"],
                img_metas=device["img_metas"],
                projection_mat=device["projection_mat"],
                gt_ego_fut_cmd=device["gt_ego_fut_cmd"],
                image_wh=device["image_wh"],
                timestamp=device["timestamp"],
                target_point=device["target_point"],
                rescale=True,
                return_loss=False,
            )
        plan = outputs[0]["img_bbox"][PLAN_KEY]
        return plan.float().cpu().numpy().reshape(-1, 2)


def reset_hipad_state(model) -> None:  # pragma: no cover - model environment only
    """Clear what HiP-AD carries between frames: every instance bank's cache, and its step count."""
    head = model.head.onedecoder_head
    for name in dir(head):
        banks = getattr(head, name)
        if name.endswith("_instance_bank_list") and isinstance(banks, list):
            for bank in banks:
                bank.reset()
    head.run_step = 0


POLICY = HiPADPolicy
