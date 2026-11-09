"""Drive the uniad or vad adapter through one scene-order request, inside the model's own environment.

Started by `test_scene_golden.py` under the interpreter, environment and working directory the model's
``model.yaml`` gives a policy node. The request is the body paddock's runner posts to a node's
``/infer_each`` in scene order -- ``{"observations": [...]}``, each the wire form of an Observation the
nuScenes harness built -- and it is handled as the node handles it: each observation decoded for the
model's cameras (``Observation.from_wire``), then the adapter's own ``infer_each`` (one reset, every
keyframe in turn). It prints one JSON object: the trajectory returned at each keyframe (paddock's ego
frame), the lidar mount the plans are in, and the adapter's diagnostics.

    python -m paddock.adapters.models.uniad.tests.scene_driver --model uniad --request <json>
"""

from __future__ import annotations

import argparse
import importlib
import json

import numpy as np

from paddock.policy.protocol import Observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["uniad", "vad"], required=True)
    parser.add_argument("--request", required=True, help="an /infer_each body: {'observations': [wire, ...]}")
    args = parser.parse_args()

    with open(args.request) as handle:
        body = json.load(handle)
    policy = importlib.import_module(f"paddock.adapters.models.{args.model}.policy").POLICY()
    policy.load()
    observations = [Observation.from_wire(item, cameras=policy.cameras) for item in body["observations"]]
    trajectories = policy.infer_each(observations)
    result = {
        "trajectories": [t.waypoints.tolist() for t in trajectories],
        "lidar2ego": np.asarray(policy.lidar2ego).tolist(),
        "diagnostics": policy.diagnostics(),
    }
    print("SCENE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
