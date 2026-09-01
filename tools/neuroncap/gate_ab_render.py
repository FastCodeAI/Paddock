"""Decisive check that the renderer honours a commanded actor pose.

Render one camera pose twice: once with the target actor where the scenario
puts it, once with the same actor teleported 500 m away. If the renderer is
placing it, the pixels inside the projected box change a lot and the pixels
elsewhere barely move. If it is ignoring /update_actors, both renders are
identical and the difference is zero everywhere.

This replaces an earlier single-image heuristic that compared the box against
neighbouring road: that failed its own control, calling the known-broken runs
"rendered", because a city scene has plenty of contrast whether or not the
target is in it. Comparing against a counterfactual render has no such
ambiguity.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion

from tools import paths

URL = "http://localhost:8000"
SCENE = os.environ.get("NCAP_SCENE", "0103")
DATAROOT = str(paths.nuscenes())


def pm(r):
    m = np.eye(4)
    m[:3, :3] = Quaternion(r["rotation"]).rotation_matrix
    m[:3, 3] = r["translation"]
    return m


def render(pose, ts):
    r = httpx.post(
        f"{URL}/render_image",
        json={"pose": pose.tolist(), "timestamp": int(ts), "camera_name": "CAM_FRONT"},
        timeout=600,
    )
    r.raise_for_status()
    return (
        torch.load(io.BytesIO(base64.b64decode(r.text)), map_location="cpu", weights_only=True)
        .numpy()
        .astype(np.float32)
    )


# This check teleports the target actor and back, which rewrites the renderer's
# actor state. Doing that while a simulation is mid-run replaces the actors out
# from under it and kills the run -- which is exactly what happened the first
# time. Refuse rather than corrupt someone's evaluation.
_busy = subprocess.run(["pgrep", "-af", "main.py --engine"], capture_output=True, text=True).stdout.strip()
if _busy and not os.environ.get("NCAP_GATE_FORCE"):
    sys.exit(
        "refusing to run: a NeuroNCAP simulation is active and this check mutates\n"
        "renderer actor state. Wait for it to finish, or set NCAP_GATE_FORCE=1 if\n"
        "you are deliberately pointing at an idle renderer.\n\n" + _busy
    )

run = Path(sys.argv[1])
actors = json.loads((run / "actors.json").read_text())
ego = json.loads((run / "ego_poses.json").read_text())
target = next(a for a in actors if a["is_target_actor"])

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
scene = next(s for s in nusc.scene if s["name"] == f"scene-{SCENE}")
sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"])
c = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
cam2ego, K = pm(c), np.asarray(c["camera_intrinsic"])

at, ap = np.array(target["timestamps"]), np.array(target["poses"])
frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))

# Pick the frame where the target is around 10 m -- close enough to occupy a
# decent patch, far enough to be fully in view.
best = None
for f in frames:
    t = int(f.stem)
    e2w = np.array(ego[str(t)])
    cpt = (
        np.linalg.inv(cam2ego)
        @ np.linalg.inv(e2w)
        @ ap[int(np.argmin(np.abs(at - t)))]
        @ np.array([0, 0, 0, 1.0])
    )
    if cpt[2] > 3 and (best is None or abs(cpt[2] - 10) < abs(best[3] - 10)):
        best = (t, e2w, cpt[:3], cpt[2])
t, e2w, cpt, rng = best
u, v = (K @ cpt)[:2] / cpt[2]
half = max(20, int(900 / rng))

send = dict(timestamps=target["timestamps"], dims=target["dims"], uuid=target["uuid"], is_target_actor=True)
httpx.post(f"{URL}/update_actors", json=[{**send, "poses": target["poses"]}], timeout=300).raise_for_status()
here = render(e2w @ cam2ego, t)

far = ap.copy()
far[:, 0, 3] += 500.0
httpx.post(f"{URL}/update_actors", json=[{**send, "poses": far.tolist()}], timeout=300).raise_for_status()
gone = render(e2w @ cam2ego, t)

httpx.post(f"{URL}/update_actors", json=[{**send, "poses": target["poses"]}], timeout=300).raise_for_status()

diff = np.abs(here - gone).mean(axis=-1)
H, W = diff.shape
y0, y1 = max(0, int(v - half)), min(H, int(v + half))
x0, x1 = max(0, int(u - half)), min(W, int(u + half))
inside = diff[y0:y1, x0:x1].mean()
mask = np.ones_like(diff, bool)
mask[y0:y1, x0:x1] = False
outside = diff[mask].mean()

print(f"{run.parent.name}/{run.name}: target at {rng:.1f} m, box {2 * half}px")
print(f"  mean |diff| inside projected box : {inside:7.2f}")
print(f"  mean |diff| everywhere else      : {outside:7.2f}")
print(f"  ratio                            : {inside / max(outside, 1e-6):7.1f}x")
print(
    "  VERDICT:",
    "actor IS rendered at the commanded pose"
    if inside > 5 * max(outside, 0.5)
    else "actor NOT rendered -- results invalid",
)
