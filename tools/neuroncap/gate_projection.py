"""Refuse to trust a run whose target actor was not actually rendered.

This exists because a whole evaluation was already produced against an
invisible target: /update_actors was failing with a 500, NeuroNCAP does not
check HTTP status, and every score came out plausible. The only reliable check
is to look at the pixels -- so the target's box is projected into the rendered
camera and the image content inside it is compared against a patch of road
beside it. A rendered vehicle differs from empty tarmac; nothing does not.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion

DATAROOT = "/shakaal_hdd_1/workspace/NeurIPS/Dataset/raw_data/nuscenes"
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs1")

def pm(r):
    m = np.eye(4); m[:3,:3] = Quaternion(r["rotation"]).rotation_matrix; m[:3,3] = r["translation"]; return m

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
scene = next(s for s in nusc.scene if s["name"] == "scene-0103")
sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"])
c = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
ego2cam, K = np.linalg.inv(pm(c)), np.asarray(c["camera_intrinsic"])

ok_all = True
for model_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    for run in sorted(model_dir.glob("*-0103")):
        actors = json.loads((run / "actors.json").read_text())
        ego = json.loads((run / "ego_poses.json").read_text())
        target = next((a for a in actors if a["is_target_actor"]), None)
        if target is None:
            print(f"{model_dir.name:12s} {run.name:18s} NO TARGET"); ok_all = False; continue
        at, ap = np.array(target["timestamps"]), np.array(target["poses"])
        frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))

        scores = []
        for f in frames:
            t = int(f.stem)
            cpt = ego2cam @ np.linalg.inv(np.array(ego[str(t)])) @ ap[int(np.argmin(np.abs(at - t)))] @ np.array([0,0,0,1.0])
            if not (3.0 < cpt[2] < 30.0):
                continue
            u, v = (K @ cpt[:3])[:2] / cpt[2]
            half = max(12, int(700 / cpt[2]))
            img = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
            H, W = img.shape
            if not (half < u < W - half and half < v < H - half):
                continue
            box = img[int(v-half):int(v+half), int(u-half):int(u+half)]
            # Road beside the target, same height in frame: the null hypothesis.
            off = int(u + 3.5 * half)
            if off + half >= W:
                off = int(u - 3.5 * half)
            ref = img[int(v-half):int(v+half), max(0,off-half):off+half]
            if box.size and ref.size:
                scores.append(abs(box.std() - ref.std()) + abs(box.mean() - ref.mean()))
        verdict = "RENDERED" if scores and np.median(scores) > 6.0 else "NOT RENDERED"
        if verdict != "RENDERED":
            ok_all = False
        print(f"{model_dir.name:12s} {run.name:18s} {verdict:12s} "
              f"(n={len(scores):3d}, median contrast vs road {np.median(scores) if scores else 0:.1f})")
sys.exit(0 if ok_all else 1)
