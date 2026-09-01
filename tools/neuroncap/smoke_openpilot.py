"""Feed the openpilot node one real nuScenes frame and check what comes back.

Exercises the camera adaptation (letterbox + warp), the NV12 path, ModelState's
buffer contract, and the plan -> waypoint conversion, without needing NeuRAD.
The field-of-view margins printed here are the number that decides whether the
adaptation is honest: both model views must fit inside what the sensor saw.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools import paths

sys.path.insert(0, str(paths.repo("openpilot")))
sys.path.insert(0, str(paths.ncap() / "servers"))

import ncap_io
from nuscenes import NuScenes
from openpilot_server import OpenpilotNode
from pyquaternion import Quaternion

DATAROOT = str(paths.nuscenes())


def encode(image: np.ndarray) -> str:
    buff = io.BytesIO()
    torch.save(torch.from_numpy(image.copy()), buff)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def pose_matrix(rec: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    m[:3, 3] = rec["translation"]
    return m


def main() -> None:
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == "scene-0103")
    sample = nusc.get("sample", scene["first_sample_token"])
    cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    calib = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    ego = pose_matrix(nusc.get("ego_pose", cam["ego_pose_token"]))

    image = np.asarray(Image.open(Path(DATAROOT) / cam["filename"]).convert("RGB"))
    canbus = np.zeros(18)
    canbus[ncap_io.CANBUS_VELOCITY] = 8.0

    body = {
        "images": {"CAM_FRONT": encode(image)},
        "ego2world": ego.tolist(),
        "canbus": canbus.tolist(),
        "timestamp": cam["timestamp"],
        "command": 2,
        "calibration": {
            "camera2image": {"CAM_FRONT": np.asarray(calib["camera_intrinsic"]).tolist()},
            "camera2ego": {"CAM_FRONT": pose_matrix(calib).tolist()},
            "lidar2ego": np.eye(4).tolist(),
        },
    }

    node = OpenpilotNode()
    out = node.infer(body)
    cam_adapter = node.camera
    print(
        f"source: {cam_adapter.src.width}x{cam_adapter.src.height}  "
        f"hFOV {cam_adapter.src.hfov_deg:.1f} deg  vFOV {cam_adapter.src.vfov_deg:.1f} deg"
    )
    print(f"fov margin after the wide model view is carved out: {cam_adapter.margin_deg()}")
    print(f"rpy (device_from_calib): {np.round(cam_adapter.rpy, 4).tolist()}")

    traj = np.asarray(out["trajectory"])
    print("\nwaypoints (x forward, y left), 0.5 s apart:")
    for i, (x, y) in enumerate(traj, 1):
        print(f"  t={i * 0.5:.1f}s  x={x:7.2f}  y={y:6.2f}")


if __name__ == "__main__":
    main()
