"""Write what UniAD's or VAD's own converter and dataset give every keyframe of the nuScenes val scenes.

Not a test, and not paddock code: run once in the models' environment, on the CPU (no model is built), from
the repo's root. For each val keyframe it calls the repo's own converter functions -- ``_get_can_bus_info``,
``obtain_sensor2top`` and, for the command, UniAD's ``NuScenesTraj.get_sdc_planning_label`` or VAD's
``get_global_sensor_pose`` -- on a stand-in for the devkit's ``NuScenes`` that holds the val records they
read, with the devkit's own ``NuScenesCanBus`` reading the CAN-bus files. The lines of ``get_data_info``
that turn an info record into the model's ``can_bus``, ``lidar2img`` and ``l2g``, and the lines of VAD's
``_fill_trainval_infos`` that turn the future poses into its command (both inline in larger functions that
also load the map and the annotations), are repeated here verbatim. `test_scene_golden.py` holds what
paddock's harness sends every val keyframe to these (``UNIAD_VAD_INFOS_REFERENCE``); the two golden scenes,
run through the datasets themselves (`scene_reference.py`), check the repeated lines.

    cd <UniAD or VAD root>
    python -m paddock.adapters.models.uniad.tests.infos_reference --model uniad \\
        --root <nuScenes root: v1.0-trainval/*.json, can_bus/> --out <dir>/infos_reference_uniad.npz
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "v1.0-trainval"
CAMERAS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
VAL_SCENES = Path(__file__).resolve().parents[3] / "benchmarks" / "nuscenes" / "val_scenes.json"


class ValTables:
    """The part of the devkit's ``NuScenes`` the converter functions read, for the val scenes.

    ``get`` and ``get_sample_data_path`` as the devkit has them; each sample's ``data`` (channel -> keyframe
    sample_data token) and ``anns`` as its reverse index adds them; ``scene`` for UniAD's ``NuScenesTraj``,
    which walks every scene; no annotations (``PredictHelper`` indexes them, the command never reads them).
    """

    def __init__(self, root: Path, scene_names: list[str]) -> None:
        def load(name: str) -> list[dict]:
            with open(root / VERSION / f"{name}.json") as handle:
                return json.load(handle)

        self.dataroot = str(root)
        self.scene = [s for s in load("scene") if s["name"] in set(scene_names)]
        scene_tokens = {s["token"] for s in self.scene}
        samples = {
            s["token"]: dict(s, data={}, anns=[]) for s in load("sample") if s["scene_token"] in scene_tokens
        }
        sample_data = {}
        for record in load("sample_data"):
            if record["is_key_frame"] and record["sample_token"] in samples:
                sample_data[record["token"]] = record
        calibrated = {c["token"]: c for c in load("calibrated_sensor")}
        sensors = {s["token"]: s for s in load("sensor")}
        for record in sample_data.values():
            channel = sensors[calibrated[record["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            samples[record["sample_token"]]["data"][channel] = record["token"]
        poses = {r["ego_pose_token"] for r in sample_data.values()}
        self.sample_annotation: list[dict] = []
        self._tables = {
            "scene": {s["token"]: s for s in self.scene},
            "sample": samples,
            "sample_data": sample_data,
            "calibrated_sensor": calibrated,
            "sensor": sensors,
            "ego_pose": {p["token"]: p for p in load("ego_pose") if p["token"] in poses},
        }

    def get(self, table: str, token: str) -> dict:
        return self._tables[table][token]

    def get_sample_data_path(self, token: str) -> str:
        return os.path.join(self.dataroot, self.get("sample_data", token)["filename"])


def vad_command(converter, nusc, sample: dict, fut_ts: int = 6) -> int:
    """VAD's ``_fill_trainval_infos``, the lines that make ``gt_ego_fut_cmd``, verbatim from the future on."""
    from pyquaternion import Quaternion

    sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    ego_fut_trajs = np.zeros((fut_ts + 1, 3))
    ego_fut_masks = np.zeros(fut_ts + 1)  # upstream: np.zeros((fut_ts+1))
    sample_cur = sample
    for i in range(fut_ts + 1):
        pose_mat = converter.get_global_sensor_pose(sample_cur, nusc, inverse=False)
        ego_fut_trajs[i] = pose_mat[:3, 3]
        ego_fut_masks[i] = 1
        if sample_cur["next"] == "":
            ego_fut_trajs[i + 1 :] = ego_fut_trajs[i]
            break
        else:
            sample_cur = nusc.get("sample", sample_cur["next"])
    ego_fut_trajs = ego_fut_trajs - np.array(pose_record["translation"])
    rot_mat = Quaternion(pose_record["rotation"]).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    ego_fut_trajs = ego_fut_trajs - np.array(cs_record["translation"])
    rot_mat = Quaternion(cs_record["rotation"]).inverse.rotation_matrix
    ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
    if ego_fut_trajs[-1][0] >= 2:
        command = np.array([1, 0, 0])  # Turn Right
    elif ego_fut_trajs[-1][0] <= -2:
        command = np.array([0, 1, 0])  # Turn Left
    else:
        command = np.array([0, 0, 1])  # Go Straight
    return int(np.argmax(command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["uniad", "vad"], required=True)
    parser.add_argument("--root", required=True, help="a nuScenes root: v1.0-trainval/*.json and can_bus/")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion

    # vad_nuscenes_converter parses a command line of its own when it is imported (scene_reference.py).
    argv, sys.argv = sys.argv, [sys.argv[0], "nuscenes", "--out-dir", "/nonexistent"]
    try:
        converter = importlib.import_module(f"tools.data_converter.{args.model}_nuscenes_converter")
    finally:
        sys.argv = argv
    scene_names = json.loads(VAL_SCENES.read_text())["val"]
    nusc = ValTables(Path(args.root), scene_names)
    can = NuScenesCanBus(dataroot=args.root)
    if args.model == "uniad":
        from mmdet3d.core.bbox import get_box_type
        from projects.mmdet3d_plugin.datasets.data_utils.trajectory_api import NuScenesTraj

        # NuScenesE2EDataset's arguments at test time (projects/configs/stage2_e2e/base_e2e.py, the dataset's
        # defaults): planning_steps 6, velocity in the boxes, the LiDAR box mode; the rest feed other labels.
        classes = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle"]
        classes += ["bicycle", "pedestrian", "traffic_cone"]
        traj = NuScenesTraj(nusc, 12, 6, 4, 4, True, classes, get_box_type("LiDAR")[1], True)

    rows: dict[str, list] = {
        k: [] for k in ("token", "can_bus", "lidar2img", "l2g_r_mat", "l2g_t", "timestamp", "command")
    }
    for scene in sorted(nusc.scene, key=lambda s: s["name"]):
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            token = sample["next"]
            if "LIDAR_TOP" not in sample["data"]:
                continue
            # _fill_trainval_infos: the info record's pose, calibration, CAN bus and cameras.
            sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            can_bus = converter._get_can_bus_info(nusc, can, sample)
            l2e_r, l2e_t = cs_record["rotation"], cs_record["translation"]
            e2g_r, e2g_t = pose_record["rotation"], pose_record["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            cams = {}
            for cam in CAMERAS:
                cam_info = converter.obtain_sensor2top(
                    nusc, sample["data"][cam], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(
                    cam_intrinsic=np.array(
                        nusc.get(
                            "calibrated_sensor",
                            nusc.get("sample_data", sample["data"][cam])["calibrated_sensor_token"],
                        )["camera_intrinsic"]
                    )
                )
                cams[cam] = cam_info
            # get_data_info, verbatim: l2g (UniAD), lidar2img, can_bus.
            l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
            l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
            lidar2img_rts = []
            for _cam_type, cam_info in cams.items():
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
            rotation = Quaternion(e2g_r)
            translation = e2g_t
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            if args.model == "uniad":
                command = traj.get_sdc_planning_label(sample["token"])[2]
            else:
                command = vad_command(converter, nusc, sample)
            rows["token"].append(sample["token"])
            rows["can_bus"].append(np.asarray(can_bus, dtype=np.float64))
            rows["lidar2img"].append(np.stack(lidar2img_rts))
            rows["l2g_r_mat"].append(l2g_r_mat.astype(np.float32))
            rows["l2g_t"].append(np.asarray(l2g_t).astype(np.float32))
            rows["timestamp"].append(sample["timestamp"] / 1e6)
            rows["command"].append(int(command))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **{key: np.asarray(value) for key, value in rows.items()},
        provenance=np.asarray(
            json.dumps(
                {
                    "written_by": "paddock.adapters.models.uniad.tests.infos_reference",
                    "date": time.strftime("%Y-%m-%d", time.gmtime()),
                    "model": args.model,
                    "converter": converter.__name__,
                    "root": str(Path(args.root).resolve()),
                }
            )
        ),
    )
    print(f"wrote {args.out}: {len(rows['token'])} keyframes of {len(nusc.scene)} scenes")


if __name__ == "__main__":
    main()
