"""Slide-ready version of the three-way comparison: 16:9, 4 panels, large type.

Deliberately fewer panels than the debug figure -- at slide distance eight BEVs plus
per-panel legends are unreadable. One shared legend, ADE in each panel title, and
every font/line sized for projection.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/shakaal_hdd_1/workspace/naren/renesas/openpilot")
from eval_kitscenes import loader as L
from eval_kitscenes.plot_three_way import load_visionpilot, DJ_PKL, ALP, OURS

DEFAULT = ["f0b1b61b:130", "85dbe6d6:130", "be5138b4:40", "eca7bf3a:220"]

GT_BAND = dict(color="#1f77b4", lw=13.0, alpha=0.25, zorder=2, solid_capstyle="round")
GT_LINE = dict(color="#1f77b4", lw=2.6, zorder=3)
STYLE = {                      # colour, linestyle, marker, markersize, linewidth
    "openpilot":   ("#16a34a", "-",              "o", 9.5, 4.0),
    "Alpamayo 1.5": ("#7c3aed", "--",             "s", 8.5, 3.6),
    "Drive-JEPA":  ("#ea580c", ":",              "^", 9.0, 3.8),
    "VisionPilot": ("#0891b2", (0, (5, 1, 1, 1)), None, 0,  3.6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=DEFAULT)
    ap.add_argument("--arm", default="nav")
    ap.add_argument("--out", default="openpilot/model_output/kitscenes_slide.png")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    ours = {(r["scene_id"][:8], r["keyframe_index"]): r for r in map(json.loads, open(OURS))}
    dj = {(r["scene_id"][:8], int(r["keyframe_index"])): r for r in pickle.load(open(DJ_PKL, "rb"))}
    alp = {}
    for r in map(json.loads, open(ALP)):
        if r.get("arm") == args.arm:
            alp[(r["scene_id"][:8], int(r["keyframe_index"]))] = r
    vp = load_visionpilot(ours)

    picks = [(p.split(":")[0], int(p.split(":")[1])) for p in args.panels]
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(16, 9), dpi=args.dpi,
                             gridspec_kw={"height_ratios": [1.30, 1.22], "hspace": 0.10, "wspace": 0.06})
    axes = np.atleast_2d(axes)

    # One common frame for every BEV so panels are directly comparable, with equal
    # x/y spans so trajectory shape is not distorted.
    pts = []
    for k in picks:
        pts.append(np.asarray(ours[k]["gt"])[:, :2])
        pts.append(np.asarray(ours[k]["pred"])[:, :2])
        pts.append(np.asarray(alp[k]["predicted_poses"])[:, :2])
        pts.append(np.asarray(dj[k]["predicted_poses"])[:, :2])
    P = np.vstack(pts)
    ylo, yhi = min(0.0, P[:, 0].min()) - 2.0, P[:, 0].max() + 3.0
    span = max(yhi - ylo, np.ptp(P[:, 1]) + 4.0)
    yhi = ylo + span
    xc = P[:, 1].mean()
    XLIM, YLIM = (xc + span / 2, xc - span / 2), (ylo, yhi)

    for i, k in enumerate(picks):
        cam, bev = axes[0, i], axes[1, i]
        o = ours[k]; kf = o["keyframe_index"]
        scene = L.scene_path(o["scene_id"], o["split"])
        im = Image.open(scene / "camera_ring_front" / f"{kf:010d}.jpg")
        # Full frame, uncropped. Any vertical crop widens the aspect, and at a fixed
        # column width a wider image is a SHORTER one -- the opposite of what is wanted.
        cam.imshow(im)
        cam.set_title(f"{k[0]}  ·  {o['ego_speed']:.1f} m/s", fontsize=18, pad=7)
        cam.axis("off")

        gt = np.asarray(o["gt"])[:, :2]
        series = [("openpilot", np.asarray(o["pred"])[:, :2], o["ade"]),
                  ("Alpamayo 1.5", np.asarray(alp[k]["predicted_poses"])[:, :2], alp[k]["ade"]),
                  ("Drive-JEPA", np.asarray(dj[k]["predicted_poses"])[:, :2], dj[k]["ade"])]

        bev.plot(gt[:, 1], gt[:, 0], **GT_BAND)
        bev.plot(gt[:, 1], gt[:, 0], **GT_LINE)
        for name, p, _ in series:
            c, ls, mk, ms, lw = STYLE[name]
            bev.plot(p[:, 1], p[:, 0], color=c, ls=ls, lw=lw, marker=mk, ms=ms,
                     mfc="white", mew=2.0, zorder=5)
        if k in vp:
            c, ls, mk, ms, lw = STYLE["VisionPilot"]
            bev.plot(vp[k]["path"][:, 1], vp[k]["path"][:, 0], color=c, ls=ls, lw=lw, zorder=4)
        bev.plot(0, 0, marker="s", ms=13, mfc="none", mec="black", mew=2.5, zorder=1)
        # a model that predicts almost no motion vanishes under the ego marker; say so
        for name, pth, _ in series:
            if np.linalg.norm(pth, axis=1).max() < 0.15 * np.linalg.norm(gt, axis=1).max():
                bev.annotate(f"{name}\nstops at {np.linalg.norm(pth,axis=1).max():.1f} m",
                             xy=(0, 0.4), xytext=(0.52, 0.22), textcoords="axes fraction",
                             fontsize=12.5, color=STYLE[name][0], fontweight="bold",
                             ha="center", arrowprops=dict(arrowstyle="->", lw=2.2,
                             color=STYLE[name][0]))

        bev.set_xlim(*XLIM); bev.set_ylim(*YLIM)
        bev.grid(alpha=0.3, lw=0.9)
        bev.tick_params(labelsize=13)
        # ADE coloured per model -- three bare numbers cannot be attributed at slide distance
        bev.text(0.02, 1.035, "ADE", transform=bev.transAxes, fontsize=14,
                 color="#444444", va="bottom")
        for j, (name, _, a) in enumerate(series):
            bev.text(0.30 + j * 0.24, 1.035, f"{a:.2f}", transform=bev.transAxes,
                     fontsize=15, fontweight="bold", color=STYLE[name][0], va="bottom")
        bev.set_xlabel("lateral y (m), left ←", fontsize=14)
        if i == 0:
            bev.set_ylabel("longitudinal x (m)", fontsize=15)
        else:
            bev.set_yticklabels([])

    handles = [Line2D([], [], color="#1f77b4", lw=7, alpha=0.45, label="human (ground truth)")]
    for name in ("openpilot", "Alpamayo 1.5", "Drive-JEPA"):
        c, ls, mk, ms, lw = STYLE[name]
        handles.append(Line2D([], [], color=c, ls=ls, lw=lw, marker=mk, ms=ms,
                              mfc="white", mew=2.0, label=name))
    c, ls, _, _, lw = STYLE["VisionPilot"]
    handles.append(Line2D([], [], color=c, ls=ls, lw=lw, label="VisionPilot (path — no timing)"))
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=16,
               frameon=False, bbox_to_anchor=(0.5, 0.965))

    fig.suptitle("Trajectory prediction on KITScenes — zero-shot", fontsize=25, y=0.995)
    fig.tight_layout(rect=[0.005, 0.01, 0.998, 0.90], h_pad=0.8, w_pad=0.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}  {int(16*args.dpi)}x{int(9*args.dpi)} px")


if __name__ == "__main__":
    main()
