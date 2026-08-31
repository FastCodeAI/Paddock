"""Draw the target actor's 3D box onto the rendered camera, as an independent check.

The scenario logs and the rendered pixels come from two different processes --
NeuroNCAP computes actor poses, NeuRAD renders them -- so projecting the logged
box into the logged image tests that they agree. If the box lands on the car in
the picture, the render, the calibration and the actor bookkeeping are all
consistent, and any remaining oddity is in how the scene looks rather than in
how it was produced.

Calibration comes from nuScenes rather than the run directory, which does not
log it; it is fixed per scene, so this is the same data the engine used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nuscenes import NuScenes
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

DATAROOT = "/shakaal_hdd_1/workspace/NeurIPS/Dataset/raw_data/nuscenes"
ROOT = Path(__import__("os").environ.get("NCAP_RUNS", "/mnt/p4/naren/neuroncap/outputs/runs1"))
OUT = Path("/mnt/p4/naren/neuroncap/validation")

# The 8 corners of a unit box centred at the origin, x forward, y left, z up.
CORNERS = np.array([[sx, sy, sz] for sx in (0.5, -0.5) for sy in (0.5, -0.5) for sz in (0.5, -0.5)])
EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def pose_matrix(rec: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
    m[:3, 3] = rec["translation"]
    return m


def calibration(scene_name: str) -> tuple[np.ndarray, np.ndarray]:
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    return pose_matrix(calib), np.asarray(calib["camera_intrinsic"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openpilot")
    ap.add_argument("--scenario", default="frontal")
    ap.add_argument("--tiles", type=int, default=8)
    args = ap.parse_args()

    run = ROOT / args.model / f"{args.scenario}-0103"
    actors = json.loads((run / "actors.json").read_text())
    ego = json.loads((run / "ego_poses.json").read_text())
    target = next(a for a in actors if a["is_target_actor"])

    cam2ego, K = calibration("scene-0103")
    ego2cam = np.linalg.inv(cam2ego)

    at = np.array(target["timestamps"], dtype=np.int64)
    ap_ = np.array(target["poses"])
    dims = np.array(target["dims"])  # width, length, height

    frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))
    picks = np.linspace(0, len(frames) - 1, args.tiles).astype(int)

    tiles = []
    for i in picks:
        path = frames[i]
        t = int(path.stem)
        actor2world = ap_[int(np.argmin(np.abs(at - t)))]
        ego2world = np.array(ego[str(t)])

        # Box corners: actor frame -> world -> ego -> camera.
        scale = np.array([dims[1], dims[0], dims[2]])  # length(x), width(y), height(z)
        pts = np.c_[CORNERS * scale, np.ones(8)]
        pts_cam = (ego2cam @ np.linalg.inv(ego2world) @ actor2world @ pts.T).T

        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        infront = pts_cam[:, 2] > 0.1
        if infront.all():
            uv = (K @ pts_cam[:, :3].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            for a, b in EDGES:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=(0, 255, 0), width=4)
            label = f"target @ {np.linalg.norm(pts_cam[:, :3].mean(axis=0)):.1f} m"
        else:
            label = "target behind camera"
        draw.rectangle([0, 0, 460, 40], fill=(0, 0, 0))
        draw.text((8, 12), f"{label}   t={(t - int(frames[0].stem)) / 1e6:.1f}s", fill=(0, 255, 128))
        tiles.append(image)

    w, h = tiles[0].size
    tw = 640
    th = int(tw * h / w)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (16, 16, 18))
    for i, tile in enumerate(tiles):
        sheet.paste(tile.resize((tw, th), Image.BILINEAR), ((i % cols) * tw, (i // cols) * th))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"projcheck_{args.model}_{args.scenario}.jpg"
    sheet.save(path, quality=90)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
