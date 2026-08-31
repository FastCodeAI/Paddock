"""Bird's-eye-view of predicted vs ground-truth trajectories, next to the camera view.

    python eval_kitscenes/plot_bev.py --run runs/full/results.jsonl --arm nav --out bev.png

One row per sample: the front camera at t0 on the left, the BEV on the right.  The BEV
shows every sampled trajectory (Alpamayo draws K of them), the medoid that the report
quotes, the ground truth, and the Drive-JEPA prediction for the same keyframe, so the
two systems can be compared by eye on identical axes.

Vehicle frame: +x forward, +y left, so the horizontal axis is negated to put left on
the left of the figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_kitscenes import manifest as manifest_mod  # noqa: E402
from eval_kitscenes.camera_warp import Pinhole, load_and_warp  # noqa: E402
from eval_kitscenes.kitscenes_alpamayo_loader import KITScenesAlpamayoScene  # noqa: E402

GT_STYLE = dict(color="#111111", lw=2.6, zorder=5, label="ground truth")
MEDOID_STYLE = dict(color="#1f6feb", lw=2.4, zorder=4, label="Alpamayo (medoid)")
DRAW_STYLE = dict(color="#79b8ff", lw=0.9, alpha=0.75, zorder=2)
JEPA_STYLE = dict(color="#d97706", lw=2.0, ls="--", zorder=3, label="Drive-JEPA")


def find_scene(root: Path, scene_id: str) -> Path | None:
    for split in ("val", "overlap_train_val"):
        path = root / "data" / split / scene_id
        if path.is_dir():
            return path
    return None


def front_image(scene_path: Path, keyframe: int) -> np.ndarray | None:
    """The front camera at t0, warped exactly as the model received it."""
    sensor = "camera_ring_front"
    try:
        scene = KITScenesAlpamayoScene(scene_path)
        pinhole: Pinhole = scene._pinhole(sensor)
        return load_and_warp(scene_path / sensor / f"{keyframe:010d}.jpg", pinhole, "fill")
    except Exception:  # noqa: BLE001 - the plot is still useful without the photo
        return None


def draw_bev(ax, row: dict[str, Any], reference: dict[str, Any] | None) -> None:
    gt = np.asarray(row["human_poses"], dtype=float)
    draws = np.asarray(row["pred_xyz_10hz"], dtype=float)  # (K, 64, 3)
    medoid = draws[int(row.get("medoid_index", 0))]

    for k, draw in enumerate(draws):
        ax.plot(-draw[:, 1], draw[:, 0], **DRAW_STYLE, label="Alpamayo draws" if k == 0 else None)
    ax.plot(-medoid[:, 1], medoid[:, 0], **MEDOID_STYLE)
    ax.plot(-gt[:, 1], gt[:, 0], marker="o", ms=3.5, **GT_STYLE)
    if reference is not None:
        jepa = np.asarray(reference["predicted_poses"], dtype=float)
        ax.plot(-jepa[:, 1], jepa[:, 0], **JEPA_STYLE)

    ax.plot(0, 0, marker="^", ms=9, color="#111111", zorder=6)  # ego at t0
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("lateral (m)   <- left | right ->")
    ax.set_ylabel("forward (m)")

    # Frame the 0-3 s scoring window plus a little headroom.
    span = max(np.abs(gt[:6, :2]).max(), np.abs(medoid[:30, :2]).max()) * 1.15 + 2
    ax.set_xlim(-span * 0.55, span * 0.55)
    ax.set_ylim(-3, span)

    ref_ade = f"{reference['ade']:.2f}" if reference is not None else "n/a"
    ax.set_title(
        f"{row['scene_id'][:8]}  k={row['keyframe_index']}  v0={row['ego_speed']:.1f} m/s\n"
        f"Alpamayo ADE {row['ade']:.2f} / medoid {row['medoid_ade']:.2f} / min {row['min_ade']:.2f}"
        f"   ·   Drive-JEPA {ref_ade}",
        fontsize=9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", default="nav")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kitscenes-root", type=Path, default=manifest_mod.KITSCENES_ROOT)
    parser.add_argument("--drive-jepa-pkl", type=Path, default=manifest_mod.DRIVE_JEPA_PKL)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.run) if line.strip()]
    rows = [r for r in rows if r["arm"] == args.arm]
    rows.sort(key=lambda r: (r["scene_id"], r["keyframe_index"]))
    rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows for arm {args.arm!r} in {args.run}")

    reference = {
        (r["scene_id"], int(r["keyframe_index"])): r
        for r in manifest_mod.load_drive_jepa_rows(args.drive_jepa_pkl)
    }

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4.1 * len(rows)),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    axes = np.atleast_2d(axes)

    for i, row in enumerate(rows):
        scene_path = find_scene(args.kitscenes_root, row["scene_id"])
        image = front_image(scene_path, row["keyframe_index"]) if scene_path else None
        ax_img = axes[i, 0]
        if image is not None:
            ax_img.imshow(image)
            ax_img.set_title(
                f"front camera, t0   ·   {(row.get('cot') or '')[:78]}", fontsize=8, loc="left"
            )
        else:
            ax_img.text(0.5, 0.5, "scene not on disk", ha="center", va="center", fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        draw_bev(axes[i, 1], row, reference.get((row["scene_id"], row["keyframe_index"])))
        if i == 0:
            axes[i, 1].legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"Alpamayo-1.5 vs Drive-JEPA on KITScenes  ·  arm '{args.arm}'  "
        f"·  scored 0-3 s on the 0.5 s grid",
        fontsize=11,
        y=0.997,
    )
    # Reserve a fixed 0.7 inch for the suptitle regardless of how many rows there are,
    # otherwise it collides with the first row's axes title.
    top = 1.0 - 0.7 / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=95)
    print(f"wrote {args.out}  ({len(rows)} samples, arm '{args.arm}')")


if __name__ == "__main__":
    main()
