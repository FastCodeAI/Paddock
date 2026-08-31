"""Replay the rewritten node over the frames the old runs recorded.

This is not a closed-loop re-run -- the ego poses are the ones the *old* node
drove to -- but it answers the question a re-run cannot answer cheaply: given
exactly the same pictures, what does the new node now decide? Three things are
compared against the logged truth and against the old node's numbers in
``validation/visionpilot/planner_analysis.json``:

    fused distance   the particle filter's posterior, vs the target's true range
    fused velocity   the filter's velocity state, vs the true closing speed
    planned travel   how far the 3 s trajectory goes, against how far away the
                     target actually is -- a plan that runs past the target is
                     the failure that put VisionPilot into the stationary car

Run with the renderer down; it reads recorded JPEGs.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/p4/naren/neuroncap/servers")
sys.path.insert(0, "/mnt/p4/naren/neuroncap")

import diag_visionpilot as diag  # noqa: E402
from visionpilot_server import VisionPilotNode  # noqa: E402

BASE = Path("/mnt/p4/naren/neuroncap")
RUNS = BASE / "outputs/runs1/visionpilot"
OUT = BASE / "validation/visionpilot"
SCENARIOS = ("stationary", "frontal", "side")
CANBUS_VELOCITY = 13


def encode(image: np.ndarray) -> str:
    buff = io.BytesIO()
    torch.save(torch.from_numpy(image.copy()), buff)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def replay(node: VisionPilotNode, scenario: str) -> list[dict]:
    run = RUNS / f"{scenario}-0103"
    K = np.asarray(diag.CALIB["CAM_FRONT"]["K"])
    cam2ego = np.asarray(diag.CALIB["CAM_FRONT"]["cam2ego"])

    actors = json.loads((run / "actors.json").read_text())
    target = next(a for a in actors if a["is_target_actor"])
    at = np.asarray(target["timestamps"], dtype=np.int64)
    ap_ = np.asarray(target["poses"])
    dims = np.asarray(target["dims"])
    ego_poses = json.loads((run / "ego_poses.json").read_text())

    frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))
    stamps = [int(p.stem) for p in frames]
    positions = np.array([np.asarray(ego_poses[str(t)])[:2, 3] for t in stamps])
    dts = np.diff(np.asarray(stamps, dtype=np.float64)) / 1e6
    speeds = np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1) / dts]
    speeds[0] = speeds[1]

    node.reset()
    rows = []
    for i, path in enumerate(frames):
        t = stamps[i]
        image = np.asarray(Image.open(path).convert("RGB"))
        ego2world = np.asarray(ego_poses[str(t)])

        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = speeds[i]
        body = {
            "images": {"CAM_FRONT": encode(image)},
            "ego2world": ego2world.tolist(),
            "canbus": canbus.tolist(),
            "timestamp": t,
            "command": 2,
            "calibration": {
                "camera2image": {"CAM_FRONT": K.tolist()},
                "camera2ego": {"CAM_FRONT": cam2ego.tolist()},
                "lidar2ego": np.eye(4).tolist(),
            },
        }
        out = node.infer(body)
        traj = np.asarray(out["trajectory"])
        dbg = out["debug"]

        actor2world = diag.actor_pose_at(at, ap_, t)
        scale3 = np.array([dims[1], dims[0], dims[2]])
        pts = np.c_[diag.CORNERS * scale3, np.ones(8)]
        pts_ego = (np.linalg.inv(ego2world) @ actor2world @ pts.T).T
        gt_range = float(np.linalg.norm(pts_ego[:, :2].mean(axis=0)))

        rows.append(
            {
                "t": round((t - stamps[0]) / 1e6, 2),
                "ego_v": round(float(speeds[i]), 2),
                "gt_range_m": round(gt_range, 2),
                "fused_m": dbg["fused_distance_m"],
                "fused_v": dbg["fused_velocity_ms"],
                "has_cipo": dbg["has_cipo"],
                "autospeed_cipo_m": dbg["autospeed_cipo_m"],
                "autodrive_m": dbg["autodrive_m"],
                "planned_3s_m": round(float(traj[-1, 0]), 2),
                "overruns_target": bool(traj[-1, 0] > gt_range),
            }
        )
        print(f"  {scenario} {rows[-1]}", flush=True)
    return rows


def main() -> None:
    node = VisionPilotNode(
        Path("/shakaal_hdd_1/workspace/naren/renesas/vision_pilot/VisionPilot/modules/models/weights"),
        ["CPUExecutionProvider"],
    )
    old = {r["scenario"]: r["rows"] for r in json.loads((OUT / "planner_analysis.json").read_text())}

    results = {}
    for scenario in SCENARIOS:
        results[scenario] = replay(node, scenario)
    (OUT / "replay_fixed.json").write_text(json.dumps(results, indent=1))

    print("\n=== new node vs old, on identical frames ===")
    print(f"{'':>11}  {'range MAE (m)':>22}  {'closing MAE (m/s)':>22}  {'frames planning past target':>28}")
    print(f"{'':>11}  {'old -> new':>22}  {'old -> new':>22}  {'old -> new':>28}")
    for scenario in SCENARIOS:
        new, prev = results[scenario], old[scenario]
        gt = np.array([r["gt_range_m"] for r in new])
        new_err = np.abs(np.array([r["fused_m"] for r in new]) - gt).mean()
        old_err = np.abs(np.array([r["autodrive_d_m"] for r in prev]) - np.array([r["gt_range_m"] for r in prev])).mean()

        true_closing = np.array([r["closing_true"] for r in prev])
        old_close = np.abs(np.array([r["closing_est"] for r in prev]) - true_closing).mean()
        # the filter reports velocity as negative-when-approaching
        new_close = np.abs(-np.array([r["fused_v"] for r in new]) - true_closing).mean()

        old_over = sum(r["reach_shipped"] > r["gt_range_m"] for r in prev)
        new_over = sum(r["overruns_target"] for r in new)
        print(
            f"{scenario:>11}  {old_err:9.1f} -> {new_err:8.1f}  {old_close:9.2f} -> {new_close:8.2f}  "
            f"{old_over:11d}/{len(prev)} -> {new_over:6d}/{len(new)}"
        )


if __name__ == "__main__":
    main()
