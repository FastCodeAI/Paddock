"""Drive-JEPA (perception-based, v2) as a paddock Policy.

Adapted from the NeuroNCAP model node. The model-specific reasoning is
unchanged; the seams now implement `Policy`, and every path arrives through
`model.yaml` rather than a hardcoded default.

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

import os
from typing import Any

import numpy as np

from paddock.policy.protocol import Observation, Policy, Trajectory

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


class DriveJepaPolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = HISTORY_DT

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        # Resolved from model.yaml (DRIVEJEPA_CHECKPOINT) rather than defaulted
        # to one machine's path, so a site with the checkpoint elsewhere needs
        # no code change.
        self.checkpoint = checkpoint or os.environ.get("DRIVEJEPA_CHECKPOINT")
        self.device = device
        self.agent = None
        self.feature_builder = None
        self.history: list[dict[str, Any]] = []

    def load(self) -> None:
        # navsim only exists inside Drive-JEPA's own environment; this module is
        # imported by `paddock list` and the test suite, which run in paddock's.
        if not self.checkpoint:
            raise RuntimeError(
                "no checkpoint: set DRIVEJEPA_CHECKPOINT (model.yaml supplies it "
                "from {data.navsim.exp}) or pass one to DriveJepaPolicy"
            )
        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

        config = DriveJEPAConfig()
        self.agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=self.checkpoint)
        self.agent.initialize()
        self.agent.eval().to(self.device)
        self.feature_builder = self.agent.get_feature_builders()[0]
        self.reset()

    def reset(self) -> None:
        """Drop the frame history between runs."""
        self.history: list[dict[str, Any]] = []

    def _record(self, obs: Observation) -> None:
        self.history.append(
            {
                "timestamp": obs.timestamp_us,
                "ego2world": obs.ego2world,
                "images": {c: obs.image(c) for c in self.cameras},
                "camera2image": obs.camera2image,
                "camera2ego": obs.camera2ego,
                "speed": obs.speed,
                "accel": obs.accel,
                "command": int(obs.command),
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

    def _cameras(self, frame: dict[str, Any], lidar2ego: np.ndarray):
        from navsim.common.dataclasses import Camera, Cameras

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

    def _agent_input(self, lidar2ego: np.ndarray):
        from navsim.common.dataclasses import AgentInput, EgoStatus

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

    def infer(self, obs: Observation) -> Trajectory:
        if self.agent is None:
            raise RuntimeError("load() must be called before infer()")

        import torch

        self._record(obs)
        lidar2ego = obs.require_lidar2ego()

        agent_input = self._agent_input(lidar2ego)
        features = self.feature_builder.compute_features(agent_input)
        features = {
            k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v) for k, v in features.items()
        }

        with torch.no_grad():
            output = self.agent.forward(features)

        # NAVSIM plans eight poses at 0.5 s as (x, y, heading), ego frame,
        # x forward and y left -- exactly the grid paddock uses, so nothing is
        # resampled. All eight are returned: closed-loop takes the first six,
        # and open-loop scoring wants the full horizon for FDE@3s.
        trajectory = output["trajectory"][0].float().cpu().numpy()
        return Trajectory(trajectory, dt=HISTORY_DT)

    def diagnostics(self) -> dict[str, Any]:
        """History depth actually achieved.

        The reason this matters: history is selected by timestamp, so a
        simulator stepping faster than 0.5 s still yields ~1.5 s of context. If
        this collapses toward zero the model is reading a near-static scene and
        its plans will look plausible while being badly under-informed.
        """
        if not self.history:
            return {"status": "no frame seen yet"}
        chosen = self._select_history()
        return {
            "frames_recorded": len(self.history),
            "history_span_s": round((self.history[-1]["timestamp"] - chosen[0]["timestamp"]) / 1e6, 2),
            "history_target_s": round((HISTORY_FRAMES - 1) * HISTORY_DT, 2),
        }


POLICY = DriveJepaPolicy
