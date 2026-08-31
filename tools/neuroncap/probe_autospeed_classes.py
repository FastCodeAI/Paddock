"""Do AutoSpeed's threat classes fire on the target, and how early?

`LongitudinalFusion::select_cipo` keeps only `class_id` 1 and 2 -- CIPO Level 1
(same lane) and Level 2 (adjacent lane, potential cut-in) -- and takes the
closest L2 over the closest L1 when the L2 is nearer, which it calls a cut-in.
Everything else the detector emits is ignored.

That rule is only worth porting if those classes actually fire on the staged
target, and early enough to brake for. This prints, per frame, the class of the
detection matching the logged target and the nearest L1/L2 range, so the answer
is measured rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import diag_visionpilot as diag

RUNS = Path("/mnt/p4/naren/neuroncap/outputs/runs1/visionpilot")
CLASS_NAME = {0: "other", 1: "L1", 2: "L2", 3: "L3"}


def main() -> None:
    K = np.asarray(diag.CALIB["CAM_FRONT"]["K"])
    cam2ego = np.asarray(diag.CALIB["CAM_FRONT"]["cam2ego"])
    ego2cam = np.linalg.inv(cam2ego)
    net = ort.InferenceSession(str(diag.WEIGHTS / "autospeed_fp32.onnx"), providers=["CPUExecutionProvider"])

    for scenario in ("stationary", "frontal", "side"):
        run = RUNS / f"{scenario}-0103"
        actors = json.loads((run / "actors.json").read_text())
        target = next(a for a in actors if a["is_target_actor"])
        at = np.asarray(target["timestamps"], dtype=np.int64)
        ap_ = np.asarray(target["poses"])
        dims = np.asarray(target["dims"])
        ego_poses = json.loads((run / "ego_poses.json").read_text())
        frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))

        print(f"\n=== {scenario} ===")
        print("   t   gt_rng  target_cls  score   nearest_L1  nearest_L2  cut_in")
        first_threat = None
        for path in frames:
            t = int(path.stem)
            image = np.asarray(Image.open(path).convert("RGB"))
            dets = diag.autospeed(net, image, 0.6, 0.45)

            actor2world = diag.actor_pose_at(at, ap_, t)
            ego2world = np.asarray(ego_poses[str(t)])
            scale3 = np.array([dims[1], dims[0], dims[2]])
            pts = np.c_[diag.CORNERS * scale3, np.ones(8)]
            pts_ego = (np.linalg.inv(ego2world) @ actor2world @ pts.T).T
            gt_range = float(np.linalg.norm(pts_ego[:, :2].mean(axis=0)))
            pts_cam = (ego2cam @ pts_ego.T).T
            visible = bool((pts_cam[:, 2] > 0.1).all())

            target_cls, target_score = "-", 0.0
            if visible and len(dets):
                uv = (K @ pts_cam[:, :3].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                gt_box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
                ious = np.array([diag.iou_matrix(gt_box, dets[j:j + 1, :4])[0] for j in range(len(dets))])
                if ious.max() > 0.3:
                    j = int(ious.argmax())
                    target_cls = CLASS_NAME.get(int(dets[j, 5]), "?")
                    target_score = float(dets[j, 4])

            best = {1: np.inf, 2: np.inf}
            if len(dets):
                bottom = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], axis=1)
                ground, ok = diag.project_to_ground(bottom, K, cam2ego)
                for j in range(len(dets)):
                    cls = int(dets[j, 5])
                    if cls in (1, 2) and ok[j] and 0 < ground[j, 0] < 60:
                        best[cls] = min(best[cls], float(np.hypot(*ground[j])))

            l1 = f"{best[1]:9.1f}" if np.isfinite(best[1]) else "        -"
            l2 = f"{best[2]:9.1f}" if np.isfinite(best[2]) else "        -"
            cut_in = np.isfinite(best[2]) and best[2] < best[1]
            if first_threat is None and (np.isfinite(best[1]) or np.isfinite(best[2])):
                first_threat = ((t - int(frames[0].stem)) / 1e6, gt_range)
            print(f"{(t - int(frames[0].stem)) / 1e6:5.1f} {gt_range:7.1f}  {target_cls:>9}  {target_score:5.2f}"
                  f"  {l1}  {l2}   {'yes' if cut_in else ''}")
        if first_threat:
            print(f"  first L1/L2 threat at t={first_threat[0]:.1f}s, target then {first_threat[1]:.1f} m away")
        else:
            print("  no L1/L2 detection at any point in this run")


if __name__ == "__main__":
    main()
