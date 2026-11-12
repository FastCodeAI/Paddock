"""Write the golden fixture for UniAD or VAD (nuScenes checkpoints), from the models' own data path.

Not a test, and not paddock code: this is the *original* tools' path, run once in the model's environment
to produce the numbers the adapter is held to (`test_golden.py`). It does what each repo's data converter,
dataset and test script do for one nuScenes keyframe -- ``obtain_sensor2top`` and ``_get_can_bus_info``
from the converters, ``get_data_info`` and ``pre_pipeline`` from the datasets, the config's own pipeline
with its own file loader reading the JPEGs, ``prepare_test_data``, collate, and the forward -- and records
what the model was given and what it planned. Nothing here imports the adapter.

The inputs are four consecutive nuScenes val keyframes of scene-0103 (the phase-1 sample: six cameras,
calibration, ego poses) and the scene's CAN-bus ``pose`` messages.

Each model is run under named *variants*: what the adapter must reproduce exactly, and two that measure an
input paddock's wire cannot carry (recorded, not asserted):

``stored_quaternion`` -- each camera placed with the lidar's ego pose and the ego pose's
  quaternion as nuScenes stores it (the adapter's input when a source sends that quaternion);
  ``canonical_quaternion`` -- the same with the quaternion's sign chosen w >= 0 on the first frame and held
  continuous (the adapter's input when a source sends none); ``camera_poses`` -- each camera placed with its
*own* ego pose, as the converter does (an Observation has one ego pose per frame).

Run from the model's repo root, paddock's src on PYTHONPATH after it::

    python -m paddock.adapters.models.uniad.tests.golden_reference --model uniad \\
        --config <cfg> --checkpoint <ckpt> [--anchors <pkl>] \\
        --meta <nusc_sample>/meta.json --data <nusc_sample> --can-bus <scene-0103_pose.json> \\
        --out <fixtures>/golden_uniad.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

NUSC_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
#: Command fed: nuScenes 2 = FORWARD (the scene drives straight).
NUSC_COMMAND = 2


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def converter_can_bus(messages: list[dict], sample_timestamp: int) -> np.ndarray:
    """``_get_can_bus_info`` of the BEVFormer-style converters (UniAD, VAD), verbatim in effect.

    pos and orientation come from the last message at or before the sample; accel, rotation_rate and vel
    from ``pose``, which after the loop is the first message *after* it (upstream reads the loop variable).
    """
    messages = [dict(m) for m in messages]
    last_pose = messages[0]
    pose = messages[0]
    for pose in messages:
        if pose["utime"] > sample_timestamp:
            break
        last_pose = pose
    last_pose = dict(last_pose)
    last_pose.pop("utime")
    can_bus = list(last_pose.pop("pos")) + list(last_pose.pop("orientation"))
    for key in last_pose.keys():  # accel, rotation_rate, vel: the JSON's key order
        can_bus.extend(pose[key])
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def canonical_signs(quaternions: list[list[float]]) -> list[np.ndarray]:
    """w >= 0 on the first, then each on the sign branch of the one before (the adapter's documented rule)."""
    out, previous = [], None
    for q in quaternions:
        q = np.asarray(q, dtype=np.float64)
        if previous is None:
            q = q if q[0] >= 0 else -q
        elif float(q @ previous) < 0:
            q = -q
        out.append(q)
        previous = q
    return out


def nusc_info(meta: dict, index: int, messages: list[dict], camera_poses: bool, rotation=None) -> dict:
    """The info-file record for one keyframe, as ``_fill_trainval_infos`` builds it (the parts read later)."""
    from pyquaternion import Quaternion

    frame = meta["frames"][index]
    lidar = frame["lidar"]
    l2e_t = np.array(lidar["calib"]["translation"])
    l2e_r = lidar["calib"]["rotation"]
    e2g_t = np.array(lidar["ego_pose"]["translation"])
    e2g_r = rotation if rotation is not None else lidar["ego_pose"]["rotation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    cams = OrderedDict()
    for cam in NUSC_CAMERAS:
        c = frame["cams"][cam]
        pose = c["ego_pose"] if camera_poses else lidar["ego_pose"]
        # obtain_sensor2top: sweep -> ego -> global -> ego' -> lidar
        l2e_r_s = c["calib"]["rotation"]
        l2e_t_s = np.array(c["calib"]["translation"])
        e2g_r_s = pose["rotation"]
        e2g_t_s = np.array(pose["translation"])
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= (
            e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            + l2e_t @ np.linalg.inv(l2e_r_mat).T
        )
        cams[cam] = {
            "data_path": c["filename"],
            "sensor2lidar_rotation": R.T,
            "sensor2lidar_translation": T,
            "cam_intrinsic": np.array(c["calib"]["camera_intrinsic"]),
        }
    return {
        "token": frame["sample_token"],
        "lidar_path": "",
        "sweeps": [],
        "cams": cams,
        "lidar2ego_translation": l2e_t,
        "lidar2ego_rotation": l2e_r,
        "ego2global_translation": e2g_t,
        "ego2global_rotation": e2g_r,
        "timestamp": frame["timestamp"],
        "scene_token": meta["scene_token"],
        "can_bus": converter_can_bus(messages, frame["timestamp"]),
        "frame_idx": index,
        "prev": "",
        "next": "",
    }


def nusc_data_info(info: dict, model: str, box_type, box_mode) -> dict:
    """``get_data_info`` of NuScenesE2EDataset (UniAD) / VADCustomNuScenesDataset (VAD), image and pose parts.

    The map and box annotations it also loads feed only evaluation; their place is taken by zero stubs of
    the shapes the pipeline collects (`nusc_gt_stubs`).
    """
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion

    input_dict = {
        "sample_idx": info["token"],
        "pts_filename": info["lidar_path"],
        "sweeps": info["sweeps"],
        "ego2global_translation": info["ego2global_translation"],
        "ego2global_rotation": info["ego2global_rotation"],
        "prev_idx": info["prev"],
        "next_idx": info["next"],
        "scene_token": info["scene_token"],
        "can_bus": info["can_bus"],
        "frame_idx": info["frame_idx"],
        "timestamp": info["timestamp"] / 1e6,
    }
    l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
    l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
    l2g_t = info["lidar2ego_translation"] @ e2g_r_mat.T + info["ego2global_translation"]
    if model == "uniad":
        input_dict.update(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32))
    image_paths, lidar2img_rts, lidar2cam_rts, cam_intrinsics = [], [], [], []
    for _cam_type, cam_info in info["cams"].items():
        image_paths.append(cam_info["data_path"])
        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
        lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info["cam_intrinsic"]
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rts.append(viewpad @ lidar2cam_rt.T)
        cam_intrinsics.append(viewpad)
        lidar2cam_rts.append(lidar2cam_rt.T)
    input_dict.update(
        img_filename=image_paths,
        lidar2img=lidar2img_rts,
        cam_intrinsic=cam_intrinsics,
        lidar2cam=lidar2cam_rts,
    )
    rotation = Quaternion(input_dict["ego2global_rotation"])
    translation = input_dict["ego2global_translation"]
    can_bus = input_dict["can_bus"]
    can_bus[:3] = translation
    can_bus[3:7] = rotation
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    can_bus[-2] = patch_angle / 180 * np.pi
    can_bus[-1] = patch_angle
    # Custom3DDataset.pre_pipeline
    input_dict.update(
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[],
        box_type_3d=box_type,
        box_mode_3d=box_mode,
    )
    return input_dict


def nusc_gt_stubs(model: str, command: int) -> dict:
    """Zero ground truth in the form the skipped annotation steps would have put in ``results``."""
    import torch

    if model == "uniad":
        t = 7  # current + occ_n_future_max (6)
        return {
            "gt_lane_labels": torch.zeros(1, dtype=torch.long),
            "gt_lane_bboxes": torch.zeros((1, 4)),
            "gt_lane_masks": torch.zeros((1, 200, 200), dtype=torch.uint8),
            "gt_segmentation": torch.zeros((t, 200, 200), dtype=torch.long),
            "gt_instance": torch.zeros((t, 200, 200), dtype=torch.long),
            "gt_centerness": torch.zeros((t, 1, 200, 200)),
            "gt_offset": torch.zeros((t, 2, 200, 200)),
            "gt_flow": torch.zeros((t, 2, 200, 200)),
            "gt_backward_flow": torch.zeros((t, 2, 200, 200)),
            "gt_occ_has_invalid_frame": False,
            "gt_occ_img_is_valid": np.ones(9, dtype=bool),
            "sdc_planning": np.zeros((1, 6, 3)),
            "sdc_planning_mask": np.zeros((1, 6, 2)),
            "command": command,
        }
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    return {
        "gt_bboxes_3d": LiDARInstance3DBoxes(np.zeros((0, 9)), box_dim=9, origin=(0.5, 0.5, 0.5)),
        "gt_labels_3d": np.zeros(0, dtype=np.int64),
        "gt_attr_labels": np.zeros((0, 34), dtype=np.float32),
        "fut_valid_flag": False,
        "ego_his_trajs": np.zeros((2, 2)),
        "ego_fut_trajs": np.zeros((6, 2)),
        "ego_fut_masks": np.zeros(6),
        "ego_fut_cmd": np.eye(3, dtype=np.int64)[command],
        "ego_lcf_feat": np.zeros(9),
        "bbox3d_fields": ["gt_bboxes_3d"],
    }


def one_gpu(batch):
    """What ``MMDataParallel.scatter`` hands the module on one GPU (mmcv 1.7.2's own fails under torch 2.x).

    Tensors to the device; each ``DataContainer`` unwrapped to its first share, moved unless ``cpu_only``.
    """
    import torch
    from mmcv.parallel import DataContainer

    if torch.is_tensor(batch):
        return batch.cuda()
    if isinstance(batch, DataContainer):
        return batch.data[0] if batch.cpu_only else one_gpu(batch.data[0])
    if isinstance(batch, (list, tuple)):
        return type(batch)(one_gpu(item) for item in batch)
    if isinstance(batch, dict):
        return {key: one_gpu(value) for key, value in batch.items()}
    return batch


def run_nuscenes(args, meta: dict, messages: list[dict]) -> dict:
    import torch
    from mmcv import Config
    from mmcv.parallel import collate
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(args.config)
    if args.model == "uniad":
        cfg.model.motion_head.anchor_info_path = args.anchors
    plugin = ".".join(Path(cfg.plugin_dir).parts)
    importlib.import_module(plugin)
    from mmdet.datasets.pipelines import Compose, to_tensor
    from mmdet3d.core.bbox import get_box_type
    from mmdet3d.models import build_model

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    model = model.cuda().eval()
    pristine = copy.deepcopy(model.prev_frame_info)
    box_type, box_mode = get_box_type("LiDAR")

    # The config's own test pipeline, its file loader reading the JPEGs from --data; only the steps that read
    # the dataset's annotations are left out (their output is stubbed in `nusc_gt_stubs`).
    skip = (
        "LoadAnnotations3D_E2E",
        "GenerateOccFlowLabels",
        "LoadPointsFromFile",
        "LoadAnnotations3D",
        "CustomObjectRangeFilter",
        "CustomObjectNameFilter",
    )
    steps = []
    for step in copy.deepcopy(cfg.data.test.pipeline):
        if step["type"] in skip:
            continue
        if step["type"] == "LoadMultiViewImageFromFilesInCeph":
            step["img_root"] = args.data
        for inner in step.get("transforms", []):
            if inner["type"] == "CustomCollect3D":
                inner["keys"] = [k for k in inner["keys"] if k != "points"]
        steps.append(step)
    pipeline = Compose(steps)

    stored = [f["lidar"]["ego_pose"]["rotation"] for f in meta["frames"]]
    variants = {
        "stored_quaternion": dict(camera_poses=False, rotations=stored),
        "canonical_quaternion": dict(
            camera_poses=False, rotations=[q.tolist() for q in canonical_signs(stored)]
        ),
        "camera_poses": dict(camera_poses=True, rotations=stored),
    }
    out = {}
    for name, variant in variants.items():
        # A fresh sequence, as a new scene in the val loop: the model's own scene-change handling plus its
        # initial ego-pose memory, so every variant starts where a freshly built model does.
        model.prev_frame_info = copy.deepcopy(pristine)
        if hasattr(model, "test_track_instances"):
            model.test_track_instances = None
            model.track_base.clear()
        frames = []
        for i in range(len(meta["frames"])):
            info = nusc_info(meta, i, messages, variant["camera_poses"], variant["rotations"][i])
            info["scene_token"] = f"{meta['scene_token']}-{name}"
            input_dict = nusc_data_info(info, args.model, box_type, box_mode)
            if args.model == "vad":
                input_dict["img_filename"] = [os.path.join(args.data, p) for p in input_dict["img_filename"]]
            input_dict.update(nusc_gt_stubs(args.model, NUSC_COMMAND))
            example = pipeline(input_dict)
            data = {
                k: (to_tensor(v[0]) if "l2g" in k else v) for k, v in example.items()
            }  # prepare_test_data
            batch = collate([data], samples_per_gpu=1)
            metas = batch["img_metas"][0].data[0][0]
            seen = {
                "lidar2img": np.asarray(metas["lidar2img"]).tolist(),
                "can_bus": np.asarray(metas["can_bus"]).tolist(),
                "img_shape": [list(s) for s in metas["img_shape"]],
            }
            if args.model == "uniad":
                seen["l2g_r_mat"] = batch["l2g_r_mat"][0].numpy().tolist()
                seen["l2g_t"] = batch["l2g_t"][0].numpy().tolist()
                seen["timestamp"] = float(batch["timestamp"][0][0])
                seen["command"] = int(batch["command"][0][0])
            else:
                command = batch["ego_fut_cmd"][0]
                command = command.data[0] if hasattr(command, "data") else command
                seen["ego_fut_cmd"] = command.numpy().reshape(-1).tolist()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **one_gpu(batch))
            if args.model == "uniad":
                plan = result[0]["planning"]["result_planning"]["sdc_traj"][0].cpu().numpy()
            else:
                preds = result[0]["pts_bbox"]["ego_fut_preds"]
                plan = preds[NUSC_COMMAND].cumsum(dim=-2).cpu().numpy()
            seen["plan"] = plan.tolist()
            frames.append(seen)
            print(f"{name} frame {i}: last waypoint {np.round(plan[-1], 3).tolist()}", flush=True)
        out[name] = {"frames": frames}
    return out


def observation_inputs(meta: dict, messages: list[dict]) -> list[dict]:
    """What paddock's nuScenes harness sends for each keyframe (as loader.py builds it), plus the CAN fields.

    camera2ego and lidar2ego are the calibration records as 4x4s, ego2world the LIDAR_TOP sample's ego
    pose; the CAN-bus values are those of the message the converter reads (`converter_can_bus`).
    """
    from pyquaternion import Quaternion

    def matrix(record):
        m = np.eye(4)
        m[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
        m[:3, 3] = record["translation"]
        return m.tolist()

    frames = []
    for frame in meta["frames"]:
        message = converter_can_bus(messages, frame["timestamp"])
        frames.append(
            {
                "sample_token": frame["sample_token"],
                "timestamp_us": int(frame["timestamp"]),
                "images": {cam: frame["cams"][cam]["filename"] for cam in NUSC_CAMERAS},
                "camera2image": {
                    cam: frame["cams"][cam]["calib"]["camera_intrinsic"] for cam in NUSC_CAMERAS
                },
                "camera2ego": {cam: matrix(frame["cams"][cam]["calib"]) for cam in NUSC_CAMERAS},
                "lidar2ego": matrix(frame["lidar"]["calib"]),
                "ego2world": matrix(frame["lidar"]["ego_pose"]),
                "ego_quaternion": list(frame["lidar"]["ego_pose"]["rotation"]),
                "can_accel": message[7:10].tolist(),
                "can_rotation_rate": message[10:13].tolist(),
                "can_velocity": message[13:16].tolist(),
            }
        )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["uniad", "vad"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchors", default=None, help="UniAD's motion_anchor_infos_mode6.pkl")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--data", required=True, help="the directory the meta's image filenames are under")
    parser.add_argument("--can-bus", required=True, help="the scene's CAN-bus <scene>_pose.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    meta = json.load(open(args.meta))
    messages = sorted(json.load(open(args.can_bus)), key=lambda m: m["utime"])
    variants = run_nuscenes(args, meta, messages)
    import mmcv

    fixture = {
        "provenance": {
            "written_by": "paddock.adapters.models.uniad.tests.golden_reference",
            "date": time.strftime("%Y-%m-%d", time.gmtime()),
            "model": args.model,
            "repo": os.getcwd().split(os.sep)[-1],
            "repo_head": git_head(os.getcwd()),
            "config": os.path.relpath(os.path.abspath(args.config), os.getcwd()),
            "checkpoint": os.path.basename(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "data": f"nuScenes {meta['scene']} keyframes, sample tokens "
            + ", ".join(f["sample_token"] for f in meta["frames"]),
            "can_bus": os.path.basename(args.can_bus),
            "torch": torch.__version__,
            "mmcv": mmcv.__version__,
            "python": sys.version.split()[0],
            "device": torch.cuda.get_device_name(0),
            "command": NUSC_COMMAND,
        },
        "observations": observation_inputs(meta, messages),
        "variants": variants,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(fixture, handle, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
