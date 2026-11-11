"""Drive one of the four adapters over a golden fixture's frames, inside the model's own environment.

Started by `test_golden.py` with the interpreter, environment and working directory the model's
``model.yaml`` declares -- what a policy node gets -- so it also proves the declared runtime. It builds
paddock Observations from the fixture's recorded inputs, the way paddock's nuScenes harness would send them
(image paths, calibration, the LIDAR_TOP ego pose, the CAN-bus values), runs the adapter's own
``infer`` frame by frame, and prints one JSON object: per variant, the plan the model made (in its lidar
frame) and the trajectory the adapter returned (paddock's ego frame), then the same sequence once more after
a ``reset``.

    python -m paddock.adapters.models.uniad.tests.golden_driver --model uniad --fixture <json> --data <dir>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np

from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command, Observation

#: The variants the adapter reproduces, and how each one's observations differ (see golden_reference.py).
VARIANTS = {
    "uniad": {"stored_quaternion": True, "canonical_quaternion": False},
    "vad": {"stored_quaternion": True, "canonical_quaternion": False},
    "uniad_b2d": {"dataset_colour": False},
    "vad_b2d": {"dataset_colour": False},
}


def observations(fixture: dict, data: str, with_quaternion: bool, command: Command) -> list[Observation]:
    out = []
    for frame in fixture["observations"]:
        ego2world = np.asarray(frame["ego2world"], dtype=np.float64)
        canbus = np.zeros(18)
        canbus[0:3] = ego2world[:3, 3]
        if with_quaternion:
            canbus[3:7] = frame["ego_quaternion"]
        canbus[CANBUS_ACCEL : CANBUS_ACCEL + 3] = frame["can_accel"]
        canbus[CANBUS_ACCEL + 3 : CANBUS_VELOCITY] = frame["can_rotation_rate"]
        canbus[CANBUS_VELOCITY : CANBUS_VELOCITY + 3] = frame["can_velocity"]
        out.append(
            Observation(
                images={},
                image_paths={cam: os.path.join(data, path) for cam, path in frame["images"].items()},
                ego2world=ego2world,
                canbus=canbus,
                timestamp_us=int(frame["timestamp_us"]),
                command=command,
                camera2image={
                    cam: np.asarray(k, dtype=np.float64) for cam, k in frame["camera2image"].items()
                },
                camera2ego={cam: np.asarray(m, dtype=np.float64) for cam, m in frame["camera2ego"].items()},
                lidar2ego=np.asarray(frame["lidar2ego"], dtype=np.float64),
            )
        )
    return out


def run(policy, frames: list[Observation]) -> list[dict]:
    rows = []
    for obs in frames:
        trajectory = policy.infer(obs)
        rows.append(
            {
                "plan_lidar": np.asarray(policy.plan_lidar).tolist(),
                "trajectory": trajectory.waypoints.tolist(),
                "lidar2ego": np.asarray(policy.lidar2ego).tolist(),
                "diagnostics": policy.diagnostics(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--data", required=True, help="the directory the fixture's image paths are under")
    args = parser.parse_args()

    fixture = json.load(open(args.fixture))
    command = Command.STRAIGHT if fixture["provenance"]["command"] == 2 else Command.FOLLOW_REFERENCE
    policy = importlib.import_module(f"paddock.adapters.models.{args.model}.policy").POLICY()
    policy.load()
    result: dict = {"variants": {}}
    for variant, with_quaternion in VARIANTS[args.model].items():
        policy.reset()
        first = run(policy, observations(fixture, args.data, with_quaternion, command))
        policy.reset()
        again = run(policy, observations(fixture, args.data, with_quaternion, command))
        result["variants"][variant] = {"frames": first, "after_reset": again}
    print("GOLDEN " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
