"""Three-way trajectory comparison, built so overlapping predictions stay readable.

Three things make coincident trajectories legible:

* **Per-panel zoom.** One shared scale is set by the longest trajectory (~55 m) and
  flattens every slow scene into a dot. Each panel is scaled to its own content;
  a metre grid plus the axis ticks carry the scale instead.
* **Ground truth as a band, not a line.** GT is drawn thick and translucent
  underneath, so a model sitting exactly on it still reads as a distinct line.
Alpamayo is plotted as its SINGLE DRAW (`predicted_poses`) -- the apples-to-apples
line against openpilot's one trajectory.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from tools import paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(paths.repo("openpilot")))
from eval_kitscenes import loader as L

DJ_PKL = paths.dataset("navsim").parent / "exp" / "inference/kitscenes_e2e_final.pkl"
ALP = paths.repo("alpamayo1.5") / "eval_kitscenes/runs/full/results.jsonl"
OURS = Path("eval_kitscenes/runs/full223/results.jsonl")
VP_JSON = Path("eval_kitscenes/visionpilot_results.json")

PANELS = [
    ("f0b1b61b", 130),
    ("a724baf7", 130),
    ("b9c27408", 40),
    ("85dbe6d6", 130),
    ("45ac20f8", 130),
    ("be5138b4", 40),
    ("eca7bf3a", 220),
    ("950dda14", 40),
]

GT_BAND = dict(color="#1f77b4", lw=7.0, alpha=0.28, zorder=2, solid_capstyle="round")
GT_LINE = dict(color="#1f77b4", lw=1.4, ls="-", zorder=3, label="human (GT)")
MODELS = [
    ("openpilot", "#16a34a", "-", "o", 5.0, 2.3),
    ("Alpamayo 1.5", "#7c3aed", "--", "s", 4.4, 2.0),
    ("Drive-JEPA", "#ea580c", ":", "^", 4.6, 2.2),
]
VP = dict(color="#0891b2", lw=2.2, ls=(0, (5, 1, 1, 1)), zorder=4)


def load_visionpilot(ours):
    """VisionPilot paths, de-rotated into the reference frame.

    Two things differ from the other three models and are handled here:

    * Its output is a PATH, not a trajectory -- `longitudinal` is a fixed 0..60 m
      grid identical in every scene (including one recorded at 0.4 m/s). There is
      no timing, so it cannot be placed on the 0.5 s grid or scored with ADE.
    * Its frame carries a constant per-scene yaw offset. Its ground truth is the
      same trajectory as the reference -- arc lengths agree to 1e-4 m at every
      point -- rotated by a fixed angle, so that angle is measured off the GT and
      undone on the predicted path.
    """
    if not VP_JSON.exists():
        return {}
    raw = json.load(open(VP_JSON))
    out = {}
    for sid, v in raw.items():
        key = (sid, int(v["frame_index"]))
        if key not in ours:
            continue
        gt_them = np.array([v["ground_truth"]["longitudinal"], v["ground_truth"]["lateral"]]).T[5::5][:8]
        gt_ref = np.asarray(ours[key]["gt"])[: len(gt_them)]
        th = float(np.mean(np.arctan2(gt_ref[:, 1], gt_ref[:, 0]) - np.arctan2(gt_them[:, 1], gt_them[:, 0])))
        path = np.array([v["prediction"]["longitudinal"], v["prediction"]["lateral"]]).T
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out[key] = {"path": path @ R.T, "rot_deg": np.degrees(th)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nav")
    ap.add_argument("--shared-scale", action="store_true", help="one scale for all panels")
    ap.add_argument("--out", default="openpilot/model_output/kitscenes_three_way.png")
    args = ap.parse_args()

    ours = {(r["scene_id"][:8], r["keyframe_index"]): r for r in map(json.loads, open(OURS))}
    dj = {(r["scene_id"][:8], int(r["keyframe_index"])): r for r in pickle.load(open(DJ_PKL, "rb"))}
    alp = {}
    for r in map(json.loads, open(ALP)):
        if r.get("arm") == args.arm:
            alp[(r["scene_id"][:8], int(r["keyframe_index"]))] = r

    vp = load_visionpilot(ours)

    ncol = 4
    nblock = (len(PANELS) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nblock * 2,
        ncol,
        figsize=(4.9 * ncol, 7.2 * nblock),
        gridspec_kw={"height_ratios": [0.72, 1.45] * nblock},
    )

    gmax = 0.0
    for k in PANELS:
        for a in (ours[k]["gt"], ours[k]["pred"], dj[k]["predicted_poses"], alp[k]["predicted_poses"]):
            gmax = max(gmax, np.abs(np.asarray(a)[:, :2]).max())

    for i, k in enumerate(PANELS):
        blk, col = divmod(i, ncol)
        cam, bev = axes[blk * 2, col], axes[blk * 2 + 1, col]
        o = ours[k]
        kf = o["keyframe_index"]
        scene = L.scene_path(o["scene_id"], o["split"])

        cam.imshow(Image.open(scene / "camera_ring_front" / f"{kf:010d}.jpg"))
        cam.set_title(f"{k[0]}  f{kf}   v={o['ego_speed']:.1f} m/s", fontsize=11)
        cam.axis("off")

        gt = np.asarray(o["gt"])[:, :2]
        preds = [
            np.asarray(o["pred"])[:, :2],
            np.asarray(alp[k]["predicted_poses"])[:, :2],
            np.asarray(dj[k]["predicted_poses"])[:, :2],
        ]
        ades = [o["ade"], alp[k]["ade"], dj[k]["ade"]]

        # Alpamayo's K=6 candidates are saved (`pred_xyz_10hz`); Drive-JEPA's 32 are
        # not -- its results keep only the selected `predicted_poses`, so there is
        # nothing to draw for it without re-running that inference.
        k6 = alp[k].get("pred_xyz_10hz")
        if k6 is not None:
            for t in np.asarray(k6):
                bev.plot(t[:, 1], t[:, 0], color="#c4b5fd", lw=0.8, alpha=0.55, zorder=1)
        bev.plot(gt[:, 1], gt[:, 0], **GT_BAND)
        bev.plot(gt[:, 1], gt[:, 0], **GT_LINE)
        for (name, c, ls, mk, ms, lw), p, ade in zip(MODELS, preds, ades):
            bev.plot(
                p[:, 1],
                p[:, 0],
                color=c,
                ls=ls,
                lw=lw,
                marker=mk,
                ms=ms,
                mfc="white",
                mew=1.3,
                zorder=4,
                label=f"{name}  {ade:.2f}",
            )
        if k in vp:
            pth = vp[k]["path"]
            bev.plot(pth[:, 1], pth[:, 0], **VP, label=f"VisionPilot (path)  {vp[k]['rot_deg']:+.1f}°")
        bev.plot(0, 0, "ks", ms=7, zorder=8)

        if args.shared_scale:
            lim = float(np.ceil(gmax / 5) * 5)
            bev.set_xlim(lim, -lim)
            bev.set_ylim(-3, lim * 1.05)
        else:  # per-panel zoom, padded and squared
            allp = np.vstack([gt] + preds)  # candidates excluded: they can fly far off
            cx, cy = allp[:, 1].mean(), allp[:, 0].mean()
            half = max(np.ptp(allp[:, 1]), np.ptp(allp[:, 0])) / 2 * 1.30 + 1.5
            bev.set_xlim(cx + half, cx - half)
            bev.set_ylim(cy - half, cy + half)
        bev.grid(alpha=0.3, lw=0.6)
        bev.set_xlabel("lateral y (m), left ←", fontsize=8.5)
        if col == 0:
            bev.set_ylabel("longitudinal x (m)", fontsize=9)
        bev.legend(fontsize=7.0, loc="best", framealpha=0.88, title="ADE (m)", title_fontsize=7.0)
        bev.tick_params(labelsize=8)

    fig.suptitle(
        "openpilot vs Alpamayo 1.5 vs Drive-JEPA on KITScenes — same 8 keyframes\n"
        f"per-panel zoom · GT = blue band · Alpamayo `{args.arm}` single draw, "
        "faint purple = its K=6 candidates (Drive-JEPA saves none)\n"
        "VisionPilot is a PATH over 0–60 m, not a trajectory — no timing, "
        "so no ADE; de-rotated into the reference frame by the angle shown",
        fontsize=11.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
