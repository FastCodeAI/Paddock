"""Write one NeuroNCAP /infer request to disk, for smoke-testing model nodes.

Each node lives in its own environment -- openpilot has no torch, Drive-JEPA is
on python 3.9, Alpamayo on 3.12 -- so they cannot share a test harness. They can
share the *input*: this writes a real request built from a nuScenes keyframe and
that scene's true calibration, and every node's smoke test reads the same file.
Identical bytes in means any difference in what comes out is the node's.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion

from paddock.policy.protocol import CANBUS_VELOCITY
from tools import paths

DATAROOT = str(paths.nuscenes())
OUT = paths.ncap() / "fixtures"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
EGO_SPEED = 8.0  # m/s, roughly what the logged driver was doing here
N_FRAMES = 8  # keyframes of history, 0.5 s apart


def pose_matrix(rec: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    m[:3, 3] = rec["translation"]
    return m


def encode(image: np.ndarray) -> str:
    buff = io.BytesIO()
    torch.save(torch.from_numpy(image.copy()), buff)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def build_request(nusc: NuScenes, sample: dict) -> dict:
    images, cam2image, cam2ego = {}, {}, {}
    timestamp = None
    for name in CAMERAS:
        sd = nusc.get("sample_data", sample["data"][name])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        images[name] = encode(np.asarray(Image.open(Path(DATAROOT) / sd["filename"]).convert("RGB")))
        cam2image[name] = np.asarray(calib["camera_intrinsic"]).tolist()
        cam2ego[name] = pose_matrix(calib).tolist()
        if name == "CAM_FRONT":
            timestamp = sd["timestamp"]
            ego = pose_matrix(nusc.get("ego_pose", sd["ego_pose_token"]))

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar2ego = pose_matrix(nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"]))

    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = EGO_SPEED  # forward velocity, ego frame

    return {
        "images": images,
        "ego2world": ego.tolist(),
        "canbus": canbus.tolist(),
        "timestamp": timestamp,
        "command": 2,  # STRAIGHT
        "calibration": {"camera2image": cam2image, "camera2ego": cam2ego, "lidar2ego": lidar2ego.tolist()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == "scene-0103")

    # A sequence, not a single frame. Three of the four nodes infer their own
    # speed from how the world moves between frames, so a repeated still frame
    # reads as a parked car and every one of them plans a stop -- which looks
    # like a broken adapter but is the correct response to the input.
    token = nusc.scene[[s["name"] for s in nusc.scene].index("scene-0103")]["first_sample_token"]
    requests = []
    for _ in range(N_FRAMES):
        sample = nusc.get("sample", token)
        requests.append(build_request(nusc, sample))
        token = sample["next"]
        if not token:
            break

    path = OUT / "infer_sequence_0103.json"
    path.write_text(json.dumps(requests))
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(requests)} frames, {len(CAMERAS)} cameras each)"
    )


if __name__ == "__main__":
    main()
