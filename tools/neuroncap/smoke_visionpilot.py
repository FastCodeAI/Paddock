"""Feed the VisionPilot node one real nuScenes frame and check what comes back.

This exercises the parts that have no other test: the ONNX pre-processing
contracts, the ground projection built from NeuroNCAP's calibration, and the
IDM speed profile. It does not need the renderer -- a recorded keyframe stands
in for a rendered one, which is exactly what --spoof-renderer does.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, "/mnt/p4/naren/neuroncap/servers")
import ncap_io
from visionpilot_server import VisionPilotNode

DATAROOT = "/shakaal_hdd_1/workspace/NeurIPS/Dataset/raw_data/nuscenes"
WEIGHTS = Path("/shakaal_hdd_1/workspace/naren/renesas/vision_pilot/VisionPilot/modules/models/weights")


def encode(image: np.ndarray) -> str:
    buff = io.BytesIO()
    torch.save(torch.from_numpy(image), buff)
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
    canbus[ncap_io.CANBUS_VELOCITY] = 8.0  # 8 m/s, a plausible urban speed

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

    node = VisionPilotNode(WEIGHTS, ["CPUExecutionProvider"])
    ph = ncap_io.Request.parse(body).pinhole("CAM_FRONT")
    print(f"camera: {ph.width}x{ph.height}  hFOV {ph.hfov_deg:.1f} deg  vFOV {ph.vfov_deg:.1f} deg")
    print(f"mounting height: {ncap_io.Request.parse(body).camera_height():.2f} m\n")

    # Two calls: AutoDrive is a two-frame model, so the first has no history.
    node.infer(body)
    out = node.infer(body)

    traj = np.asarray(out["trajectory"])
    print("debug:", out["debug"])
    print("\nwaypoints (x forward, y left), 0.5 s apart:")
    for i, (x, y) in enumerate(traj, 1):
        print(f"  t={i*0.5:.1f}s  x={x:7.2f}  y={y:6.2f}")
    step = np.linalg.norm(np.diff(np.vstack([[0, 0], traj]), axis=0), axis=1)
    print(f"\nimplied speed over horizon: {step / 0.5}")


if __name__ == "__main__":
    main()
