"""NeuroNCAP model node for Drive-JEPA (perception-based, v2).

    NAVSIM_EXP_ROOT=/shakaal_hdd_1/workspace/naren/navsim_workspace/exp \\
    PYTHONPATH=/shakaal_hdd_1/workspace/naren/renesas/Drive-JEPA/navsim_v2:\\
/mnt/p4/naren/neuroncap/servers \\
      conda run -p .../drive-jepa python drivejepa_server.py --port 9003

Of the four models this is the closest fit to NeuroNCAP out of the box: it is a
NAVSIM agent, and NAVSIM's planning grid -- eight poses at 0.5 s, ego frame,
x forward and y left -- is exactly the grid NeuroNCAP drives on. The first six
poses are taken and nothing is resampled.

Rather than reimplement the feature pipeline, this builds the ``AgentInput``
dataclass NAVSIM already knows how to consume and calls the shipped feature
builder and agent. That keeps the BEVFormer projection maths (lidar2img from
each camera's sensor2lidar and intrinsics) in the code that was trained with it.

Two adaptations:

*Cameras.* Drive-JEPA's perception branch reads four views -- front, front-left,
front-right and back -- which map onto nuScenes cameras of the same names. The
resolutions differ (NAVSIM is 1920x1080, nuScenes 1600x900) but the geometry
travels in the intrinsics and the pipeline rescales, so this is a genuine
adaptation rather than a fudge. The one hard-coded number that does not travel
is the ``[28:-28]`` crop the stitching path applies to reach a 2:1 aspect: on
1080 rows that yields 1024, on 900 it yields 844. Both are then resized to
512x256, so the effect is a slightly different vertical crop, not a distortion.

*Rate.* Drive-JEPA is a 2 Hz agent with a four-frame history. NeuroNCAP may be
stepping at 10 Hz to suit the other models, so the history is selected by
timestamp at ~0.5 s spacing rather than by taking the last four requests --
otherwise the model would see 0.3 s of history and read the scene as nearly
static.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI

import ncap_io

from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus

# Drive-JEPA camera slot -> nuScenes camera. The perception branch reads these
# four; the remaining slots stay empty and are skipped by the loader.
CAMERA_MAP = {
    "cam_f0": "CAM_FRONT",
    "cam_l0": "CAM_FRONT_LEFT",
    "cam_r0": "CAM_FRONT_RIGHT",
    "cam_b0": "CAM_BACK",
}
NEEDED_CAMERAS = tuple(CAMERA_MAP.values())

HISTORY_FRAMES = 4  # NAVSIM agent input depth
HISTORY_DT = 0.5  # seconds between history frames, NAVSIM's rate

# NeuroNCAP's Command enum -> NAVSIM's one-hot [left, straight, right, unknown].
COMMAND_ONE_HOT = {
    0: [0, 0, 1, 0],  # RIGHT
    1: [1, 0, 0, 0],  # LEFT
    2: [0, 1, 0, 0],  # STRAIGHT
    3: [0, 0, 0, 1],  # FOLLOW_REFERENCE -> unknown
}


def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(x, y, heading) of *pose* expressed in *reference*'s frame."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], yaw_of(delta)], dtype=np.float64)


class DriveJepaNode:
    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        config = DriveJEPAConfig()
        self.agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=checkpoint)
        self.agent.initialize()
        self.agent.eval().to(device)
        self.feature_builder = self.agent.get_feature_builders()[0]
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Drop the frame history between runs."""
        self.history: list[dict[str, Any]] = []

    def _record(self, req: ncap_io.Request) -> None:
        self.history.append(
            {
                "timestamp": req.timestamp,
                "ego2world": req.ego2world,
                "images": req.images,
                "camera2image": req.camera2image,
                "camera2ego": req.camera2ego,
                "speed": req.speed,
                "accel": req.accel,
                "command": req.command,
            }
        )
        # 12 s of 10 Hz history is far more than the 1.5 s ever used; trimming
        # keeps six decoded camera images per entry from growing without bound.
        if len(self.history) > 128:
            self.history = self.history[-128:]

    def _select_history(self) -> list[dict[str, Any]]:
        """Four frames at ~0.5 s spacing, oldest first, clamped at the run start.

        Selected by timestamp rather than by position, so the same node behaves
        correctly whether NeuroNCAP is stepping at 0.5 s or 0.1 s.
        """
        now = self.history[-1]["timestamp"]
        chosen = []
        for step in range(HISTORY_FRAMES - 1, -1, -1):
            target = now - int(step * HISTORY_DT * 1e6)
            nearest = min(self.history, key=lambda f: abs(f["timestamp"] - target))
            chosen.append(nearest)
        return chosen

    def _cameras(self, frame: dict[str, Any], lidar2ego: np.ndarray) -> Cameras:
        slots: dict[str, Camera] = {}
        ego2lidar = np.linalg.inv(lidar2ego)
        for slot, camera in CAMERA_MAP.items():
            sensor2lidar = ego2lidar @ frame["camera2ego"][camera]
            slots[slot] = Camera(
                image=frame["images"][camera],
                sensor2lidar_rotation=sensor2lidar[:3, :3].astype(np.float32),
                sensor2lidar_translation=sensor2lidar[:3, 3].astype(np.float32),
                intrinsics=frame["camera2image"][camera].astype(np.float32),
            )
        empty = Camera()
        return Cameras(
            cam_f0=slots["cam_f0"],
            cam_l0=slots["cam_l0"],
            cam_l1=empty,
            cam_l2=empty,
            cam_r0=slots["cam_r0"],
            cam_r1=empty,
            cam_r2=empty,
            cam_b0=slots["cam_b0"],
        )

    def _agent_input(self, lidar2ego: np.ndarray) -> AgentInput:
        frames = self._select_history()
        current = frames[-1]["ego2world"]

        ego_statuses = []
        for frame in frames:
            ego_statuses.append(
                EgoStatus(
                    ego_pose=relative_pose(frame["ego2world"], current),
                    ego_velocity=np.array([frame["speed"], 0.0], dtype=np.float32),
                    ego_acceleration=np.array([frame["accel"], 0.0], dtype=np.float32),
                    driving_command=np.array(COMMAND_ONE_HOT[frame["command"]], dtype=np.int64),
                )
            )
        cameras = [self._cameras(frame, lidar2ego) for frame in frames]
        return AgentInput(ego_statuses=ego_statuses, cameras=cameras, lidars=[])

    def infer(self, body: dict[str, Any]) -> dict[str, Any]:
        req = ncap_io.Request.parse(body, cameras=NEEDED_CAMERAS)
        self._record(req)
        lidar2ego = np.asarray(body["calibration"]["lidar2ego"], dtype=np.float64)

        agent_input = self._agent_input(lidar2ego)
        features = self.feature_builder.compute_features(agent_input)
        features = {
            k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v) for k, v in features.items()
        }

        with torch.no_grad():
            output = self.agent.forward(features)

        # NAVSIM plans eight poses at 0.5 s as (x, y, heading), ego frame,
        # x forward and y left -- the grid NeuroNCAP drives, so no resampling.
        trajectory = output["trajectory"][0].float().cpu().numpy()
        return ncap_io.response(trajectory[:6, :2]) | {
            "debug": {
                "history_span_s": round(
                    (self.history[-1]["timestamp"] - self._select_history()[0]["timestamp"]) / 1e6, 2
                ),
                "poses_returned": int(trajectory.shape[0]),
            }
        }


def build_app(node: DriveJepaNode) -> FastAPI:
    app = FastAPI()

    @app.get("/alive")
    async def alive() -> bool:
        return True

    @app.post("/infer")
    async def infer(body: dict) -> dict:
        return node.infer(body)

    @app.post("/reset")
    async def reset(body: dict | None = None) -> bool:
        node.reset()
        return True

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--checkpoint",
        default="/shakaal_hdd_1/workspace/naren/navsim_workspace/exp/Drive-JEPA-cache/"
        "drive_jepa_perception_based_agent_vitl_v2.ckpt",
    )
    args = parser.parse_args()
    uvicorn.run(build_app(DriveJepaNode(args.checkpoint)), host=args.host, port=args.port, log_level="warning")
