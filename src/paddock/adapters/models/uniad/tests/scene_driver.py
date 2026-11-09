"""Drive the uniad or vad adapter over a scene-order golden fixture, inside the model's own environment.

Started by `test_scene_golden.py` under the interpreter, environment and working directory the model's
``model.yaml`` gives a policy node. It builds one paddock Observation per keyframe of the fixture's scene,
carrying what the repo's dataset gave that keyframe (`observations`), sends the whole scene through the
adapter's own ``infer_each`` -- scene order, one reset, every keyframe in turn -- and prints one JSON object:
the trajectory returned at each keyframe (paddock's ego frame) and the lidar mount the plans are in.

    python -m paddock.adapters.models.uniad.tests.scene_driver --model uniad --fixture <json> --data <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np

from paddock.adapters.models.uniad import bev_planners as bp
from paddock.policy.protocol import CANBUS_ACCEL, Command, Observation

#: The datasets' command index -> paddock's (`bev_planners.NUSC_COMMAND`, inverted).
COMMANDS = {0: Command.RIGHT, 1: Command.LEFT, 2: Command.STRAIGHT}


def observations(fixture: dict, data: str) -> list[Observation]:
    """Each keyframe as an Observation carrying what the repo's dataset gave the model there.

    Two of those are things paddock's nuScenes harness does not send today, and both are on the wire:
    each camera is placed where the converter placed it -- with its *own* ego pose, folded into
    ``camera2ego`` relative to the lidar's (``sensor2lidar``) -- and the ego pose's quaternion as nuScenes
    stores it goes in canbus 3:7. The CAN-bus accel, rotation rate and velocity are the converter's message's,
    and the command is the dataset's.
    """
    out = []
    for frame in fixture["frames"]:
        lidar2ego = bp.pose_matrix(frame["lidar2ego_translation"], frame["lidar2ego_rotation"])
        camera2ego = {}
        for camera, cam in frame["cams"].items():
            cam2lidar = np.eye(4)
            cam2lidar[:3, :3] = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            cam2lidar[:3, 3] = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
            camera2ego[camera] = lidar2ego @ cam2lidar
        canbus = np.zeros(18)
        canbus[0:3] = frame["ego2global_translation"]
        canbus[3:7] = frame["ego2global_rotation"]
        canbus[CANBUS_ACCEL:16] = np.asarray(frame["can_bus"], dtype=np.float64)[CANBUS_ACCEL:16]
        out.append(
            Observation(
                images={},
                image_paths={
                    camera: os.path.join(data, cam["file"]) for camera, cam in frame["cams"].items()
                },
                ego2world=bp.pose_matrix(frame["ego2global_translation"], frame["ego2global_rotation"]),
                canbus=canbus,
                timestamp_us=int(frame["timestamp_us"]),
                command=COMMANDS[int(frame["command"])],
                camera2image={
                    camera: np.asarray(cam["intrinsic"], dtype=np.float64)
                    for camera, cam in frame["cams"].items()
                },
                camera2ego=camera2ego,
                lidar2ego=lidar2ego,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["uniad", "vad"], required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument(
        "--data", required=True, help="a nuScenes root holding the scene's samples/CAM_* JPEGs"
    )
    args = parser.parse_args()

    fixture = json.load(open(args.fixture))
    policy = importlib.import_module(f"paddock.adapters.models.{args.model}.policy").POLICY()
    policy.load()
    trajectories = policy.infer_each(observations(fixture, args.data))
    result = {
        "trajectories": [t.waypoints.tolist() for t in trajectories],
        "lidar2ego": np.asarray(policy.lidar2ego).tolist(),
        "diagnostics": policy.diagnostics(),
    }
    print("SCENE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
