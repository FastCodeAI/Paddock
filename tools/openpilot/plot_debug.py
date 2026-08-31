"""Debug sheet: source camera | what openpilot's network sees | BEV of all three models.

The middle column is the point of this plot -- it is the actual `img` tensor fed to
the supercombo, so a bad trajectory can be traced to a bad input rather than guessed at.

Vehicle frame is +x forward, +y left, so the BEV's horizontal axis is negated to put
left on the left. All four trajectories live on the same 0.5 s grid.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/shakaal_hdd_1/workspace/naren/renesas/openpilot")
from eval_kitscenes.camera_adapt import load_camera, warp_to_model_frame, MODEL_VIEWS
from eval_kitscenes import loader as L

DJ_PKL = Path("/shakaal_hdd_1/workspace/naren/navsim_workspace/exp/inference/kitscenes_e2e_final.pkl")
ALP = Path("/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/eval_kitscenes/runs/full/results.jsonl")
OURS = Path("eval_kitscenes/runs/local20/results.jsonl")

GT_S   = dict(color="#111111", lw=3.0, zorder=6, label="ground truth", marker="o", ms=4)
OP_S   = dict(color="#16a34a", lw=2.6, zorder=5, label="openpilot", marker="o", ms=4)
ALP_S  = dict(color="#1f6feb", lw=2.2, zorder=4, label="Alpamayo (medoid)", marker="s", ms=3.5)
DJ_S   = dict(color="#d97706", lw=2.0, ls="--", zorder=3, label="Drive-JEPA", marker="^", ms=3.5)
K_S    = dict(color="#79b8ff", lw=0.8, alpha=0.55, zorder=2)

# spread across the interesting disagreements rather than the best rows
DEFAULT = ["f0b1b61b:130", "1df9975b:130", "67abb4e2:130", "ab1cafcc:40",
           "c34c778f:40", "b8002a79:130"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=DEFAULT, help="scene8:keyframe")
    ap.add_argument("--arm", default="nonav")
    ap.add_argument("--out", default="openpilot/model_output/kitscenes_bev_debug.png")
    args = ap.parse_args()

    ours = {(r["scene_id"][:8], r["keyframe_index"]): r for r in map(json.loads, open(OURS))}
    dj = {(r["scene_id"][:8], int(r["keyframe_index"])): r for r in pickle.load(open(DJ_PKL, "rb"))}
    alp = {}
    for r in map(json.loads, open(ALP)):
        if r.get("arm") == args.arm:
            alp[(r["scene_id"][:8], int(r["keyframe_index"]))] = r

    picks = [(s.split(":")[0], int(s.split(":")[1])) for s in args.scenes]
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(19, 4.5 * n),
                             gridspec_kw={"width_ratios": [1.35, 1.35, 1.0]})
    axes = np.atleast_2d(axes)

    for i, key in enumerate(picks):
        o = ours[key]
        scene = L.scene_path(o["scene_id"], o["split"] if (L.KITSCENES_ROOT / o["split"] / o["scene_id"]).is_dir()
                             else ("val" if (L.KITSCENES_ROOT / "val" / o["scene_id"]).is_dir() else "overlap_train_val"))
        kf = o["keyframe_index"]
        cam = load_camera(scene)
        src = Image.open(scene / "camera_ring_front" / f"{kf:010d}.jpg")

        axes[i, 0].imshow(src)
        axes[i, 0].set_title(f"{key[0]}  kf{kf}  ({o['split']})  v={o['ego_speed']:.1f} m/s\n"
                             f"KITScenes camera_ring_front (87 deg)", fontsize=10)
        axes[i, 0].axis("off")

        prep = np.asarray(cam.prepare(src))
        size, big = MODEL_VIEWS["img"]
        axes[i, 1].imshow(warp_to_model_frame(prep, cam.warp_matrix(big), size))
        lp = np.mean(o["lane_line_probs"][:4])
        axes[i, 1].set_title(f"what openpilot's network sees: `img` 512x256\n"
                             f"mean laneProb {lp:.2f}", fontsize=10)
        axes[i, 1].axis("off")

        ax = axes[i, 2]
        gt = np.array(o["gt"])
        op = np.array(o["pred"])
        ax.plot(-gt[:, 1], gt[:, 0], **GT_S)
        ax.plot(-op[:, 1], op[:, 0], **OP_S)
        lbl = [f"openpilot {o['ade']:.2f}"]
        if key in alp:
            a = np.array(alp[key].get("predicted_poses_medoid") or alp[key]["predicted_poses"])
            ax.plot(-a[:, 1], a[:, 0], **ALP_S)
            lbl.append(f"Alpamayo {alp[key]['ade']:.2f}")
            k = alp[key].get("pred_xyz_10hz")
            if k is not None:
                for t in np.asarray(k):
                    ax.plot(-t[:, 1], t[:, 0], **K_S)
        if key in dj:
            d = np.asarray(dj[key]["predicted_poses"])
            ax.plot(-d[:, 1], d[:, 0], **DJ_S)
            lbl.append(f"Drive-JEPA {dj[key]['ade']:.2f}")
        # travel ratio at 3 s -- separates "wrong path" from "right path, wrong speed"
        d3 = lambda a: float(np.linalg.norm(np.asarray(a)[5, :2]))
        ratio = d3(op) / max(d3(gt), 1e-6)
        lon = np.abs(op[:6, 0] - gt[:6, 0]).mean()
        lat = np.abs(op[:6, 1] - gt[:6, 1]).mean()
        ax.set_title("BEV  ADE: " + "  |  ".join(lbl) +
                     f"\nopenpilot travel@3s {ratio:.2f}x GT   "
                     f"err lon {lon:.2f} / lat {lat:.2f} m", fontsize=9)
        ax.set_xlabel("y left (m)"); ax.set_ylabel("x forward (m)")
        ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("KITScenes trajectory debug — 0.5 s grid, 0–4 s horizon "
                 "(ADE scored over 0–3 s)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  ({n} scenes, arm={args.arm})")


if __name__ == "__main__":
    main()
