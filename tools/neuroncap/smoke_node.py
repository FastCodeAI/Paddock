"""Run one model node against the shared fixture and print what it produced.

Usage (each in its own environment):
    python smoke_node.py openpilot
    python smoke_node.py visionpilot
    python smoke_node.py alpamayo
    python smoke_node.py drivejepa

Prints the waypoints and the implied speed profile. A node that is wired up
correctly produces waypoints that march forward at roughly the ego speed; a
sign error or a unit error shows up here as a car that plans to drive backwards,
sideways, or at 300 m/s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from tools import paths

sys.path.insert(0, str(paths.ncap() / "servers"))

FIXTURE = paths.ncap() / "fixtures/infer_sequence_0103.json"


def build(kind: str):
    if kind == "openpilot":
        from openpilot_server import OpenpilotNode

        return OpenpilotNode()
    if kind == "visionpilot":
        from visionpilot_server import VisionPilotNode

        return VisionPilotNode(
            paths.repo("vision_pilot") / "VisionPilot/modules/models/weights",
            ["CPUExecutionProvider"],
        )
    if kind == "alpamayo":
        from alpamayo_server import AlpamayoNode

        return AlpamayoNode()
    if kind == "drivejepa":
        from drivejepa_server import DriveJepaNode

        return DriveJepaNode(
            str(
                paths.dataset("navsim").parent
                / "exp/Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
            )
        )
    raise SystemExit(f"unknown node: {kind}")


def main() -> None:
    kind = sys.argv[1]
    sequence = json.loads(FIXTURE.read_text())
    node = build(kind)

    # Feed the whole sequence; report the last answer. Nodes that derive speed
    # from apparent motion need genuinely different frames, not a repeated one.
    for body in sequence:
        out = node.infer(body)
    print(
        f"fed {len(sequence)} frames, {(sequence[-1]['timestamp'] - sequence[0]['timestamp']) / 1e6:.1f} s apart"
    )

    traj = np.asarray(out["trajectory"])
    print(f"\n{kind}: {traj.shape[0]} waypoints")
    if "debug" in out:
        print(f"debug: {out['debug']}")
    print("\n  t (s)      x (m)    y (m)   implied speed (m/s)")
    prev = np.zeros(2)
    for i, point in enumerate(traj, 1):
        speed = np.linalg.norm(point - prev) / 0.5
        print(f"  {i * 0.5:4.1f}   {point[0]:8.2f} {point[1]:8.2f}   {speed:8.2f}")
        prev = point


if __name__ == "__main__":
    main()
