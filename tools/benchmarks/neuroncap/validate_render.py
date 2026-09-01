"""Check NeuRAD's renders against the real nuScenes photographs.

Two things are on trial here.  First, that the renderer is wired up correctly at
all -- right pose convention, right timestamp units, right camera.  Second, and
the reason this exists: ``run_renderer.py`` feeds NeuRAD empty point clouds
wherever a lidar sweep is missing from this nuScenes copy.  If those points
mattered to image rendering, the reconstruction would be visibly wrong.

The test re-renders keyframes from the pose the real camera occupied and
compares against the recorded JPEG.  A faithful reconstruction lands around
20-25 dB PSNR against held-out keyframes; single digits mean the scene is
broken, and anything in between deserves a look at the saved side-by-sides.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import httpx
import numpy as np
import torch
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion

from tools import paths

CAM = "CAM_FRONT"


def pose_matrix(record: dict) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    m[:3, 3] = np.asarray(record["translation"])
    return m


def render(client: httpx.Client, url: str, pose: np.ndarray, timestamp: int) -> np.ndarray:
    resp = client.post(
        f"{url}/render_image",
        json={"pose": pose.tolist(), "timestamp": int(timestamp), "camera_name": CAM},
        timeout=600.0,
    )
    resp.raise_for_status()
    tensor = torch.load(io.BytesIO(base64.b64decode(resp.text)), map_location="cpu", weights_only=True)
    return tensor.numpy().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--dataroot", default=str(paths.nuscenes()))
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default=str(paths.ncap() / "validation"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == args.scene)

    # Walk the keyframes, sampling a few spread through the clip.
    tokens, token = [], scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    picks = np.linspace(0, len(tokens) - 1, args.frames).astype(int)

    client = httpx.Client()
    scores = []
    for i in picks:
        sample = nusc.get("sample", tokens[i])
        cam_data = nusc.get("sample_data", sample["data"][CAM])
        ego = pose_matrix(nusc.get("ego_pose", cam_data["ego_pose_token"]))
        cam2ego = pose_matrix(nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"]))

        rendered = render(client, args.url, ego @ cam2ego, cam_data["timestamp"])
        real = np.asarray(Image.open(Path(args.dataroot) / cam_data["filename"]).convert("RGB"))
        if rendered.shape != real.shape:
            real = np.asarray(
                Image.fromarray(real).resize((rendered.shape[1], rendered.shape[0]), Image.BILINEAR)
            )

        score = psnr(rendered, real)
        scores.append(score)
        Image.fromarray(np.concatenate([real, rendered], axis=0)).save(
            out_dir / f"frame_{i:02d}.jpg", quality=90
        )
        print(f"keyframe {i:2d}  PSNR {score:5.2f} dB   (top: real, bottom: rendered)", flush=True)

    print(f"\nmean PSNR over {len(scores)} keyframes: {np.mean(scores):.2f} dB")
    print(f"side-by-sides written to {out_dir}")


if __name__ == "__main__":
    main()
