"""Per-sample figures: the six ring cameras plus a BEV of predicted vs. recorded path.

Complements visualize_kitscenes.py, which summarises the whole run.  This renders
individual samples the way a scene viewer would, so a specific success or failure
can be inspected rather than inferred from an aggregate.

    python scripts/inference/visualize_kitscenes_samples.py \
        --results <pkl> --kitscenes-root <root> --outdir <dir> \
        --samples <scene_id>:<keyframe> ...

Requires the scene to still be on disk.  The sweep streams and deletes scenes, so
re-fetch any scene you want to plot:

    python -m kitscenes.download <root> --scenes <scene_id>

All six ring cameras are shown, but the model only consumes four of them
(front, front-left, front-right, rear).  Those four are drawn with the crop
box that ``kitscenes_adapter`` actually feeds the network -- the OpenScene
field of view, which is much narrower than the KITScenes sensor -- so what the
model saw is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# Same validated categorical slots as the summary figures.
HUMAN_COLOR = "#2a78d6"
PREDICTED_COLOR = "#eb6834"
CROP_COLOR = "#eda100"
SURFACE = "#fcfcfb"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#d8d8d4"

# Ring layout as it appears on the vehicle: forward row, then rear row.
CAMERA_GRID = [
    ["camera_ring_front_left", "camera_ring_front", "camera_ring_front_right"],
    ["camera_ring_rear_left", "camera_ring_rear", "camera_ring_rear_right"],
]
# The four the perception-based agent actually reads, and the slot each fills.
MODEL_CAMERAS = {
    "camera_ring_front": "cam_f0",
    "camera_ring_front_left": "cam_l0",
    "camera_ring_front_right": "cam_r0",
    "camera_ring_rear": "cam_b0",
}
DISPLAY_WIDTH = 720  # downscale for file size; the crop box is drawn in these coords

TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
TARGET_FOCAL, TARGET_CU, TARGET_CV = 1545.0, 960.0, 560.0
EGO_LENGTH, EGO_WIDTH = 4.8, 2.0


def crop_box(intrinsics: Dict[str, float]) -> tuple:
    """The window kitscenes_adapter feeds the model, in source pixels."""
    scale = TARGET_FOCAL / float(intrinsics["focal_length"])
    left = intrinsics["principal_point_u"] - TARGET_CU / scale
    top = intrinsics["principal_point_v"] - TARGET_CV / scale
    return left, top, TARGET_WIDTH / scale, TARGET_HEIGHT / scale


def draw_cameras(axes, scene_path: Path, frame_index: int, calibration: Dict[str, Any]) -> None:
    """Six ring cameras, with the model's crop outlined on the four it consumes."""
    for row, names in enumerate(CAMERA_GRID):
        for column, sensor in enumerate(names):
            ax = axes[row][column]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            image_path = scene_path / sensor / f"{frame_index:010d}.jpg"
            if not image_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color=MUTED, fontsize=9)
                continue

            image = Image.open(image_path)
            source_width = image.width
            scale = DISPLAY_WIDTH / source_width
            ax.imshow(image.resize((DISPLAY_WIDTH, int(image.height * scale)), Image.BILINEAR))

            slot = MODEL_CAMERAS.get(sensor)
            if slot:
                left, top, width, height = crop_box(calibration[sensor]["intrinsics"])
                ax.add_patch(
                    Rectangle((left * scale, top * scale), width * scale, height * scale,
                              fill=False, edgecolor=CROP_COLOR, linewidth=2.0, zorder=3)
                )
            label = f"{sensor.replace('camera_ring_', '')}" + (f"  → {slot}" if slot else "  (unused)")
            ax.set_title(label, fontsize=8, color=INK if slot else MUTED, pad=3)
            for spine in ax.spines.values():
                spine.set_color(GRID)


def draw_bev(ax, sample: Dict[str, Any], horizon: int) -> None:
    """Predicted vs. recorded path in the ego frame at the keyframe."""
    human = sample["human_poses"][:horizon, :2]
    predicted = sample["predicted_poses"][:horizon, :2]

    ax.set_facecolor(SURFACE)
    ax.add_patch(
        Rectangle((-EGO_WIDTH / 2, -EGO_LENGTH / 2), EGO_WIDTH, EGO_LENGTH,
                  facecolor="#c9ccd1", edgecolor=INK, linewidth=1.2, zorder=2)
    )
    ax.plot(human[:, 1], human[:, 0], color=HUMAN_COLOR, linewidth=2.4,
            marker="o", markersize=6, label="recorded", zorder=4)
    ax.plot(predicted[:, 1], predicted[:, 0], color=PREDICTED_COLOR, linewidth=2.4,
            linestyle="--", marker="o", markersize=6, label="Drive-JEPA", zorder=5)

    # Scale the two axes independently. Tying the lateral range to the longitudinal
    # reach makes a 50 m straight run render as a sliver.  Aspect stays equal so the
    # trajectory shape is not distorted.
    lateral = max(np.abs(np.concatenate([human, predicted])[:, 1]).max() * 1.4 + 3.0, 8.0)
    forward = max(np.concatenate([human, predicted])[:, 0].max() * 1.12, 12.0)
    ax.set_xlim(lateral, -lateral)  # inverted so left-of-ego is on the left
    ax.set_ylim(-6, forward)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("lateral y (m), left ←", fontsize=9, color=MUTED)
    ax.set_ylabel("longitudinal x (m)", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")


def render_sample(
    sample: Dict[str, Any], kitscenes_root: Path, output_path: Path, horizon: int
) -> bool:
    scene_path = kitscenes_root / "data" / sample["split"] / sample["scene_id"]
    if not scene_path.is_dir():
        print(f"  ! {sample['scene_id']} not on disk -- re-fetch it to plot this sample")
        return False
    calibration = json.loads((scene_path / "calibration/calib.json").read_text())

    # Explicit placement: tight_layout cannot reconcile the image axes with the
    # equal-aspect BEV, and silently let the BEV labels run under the cameras.
    figure = plt.figure(figsize=(16.5, 6.4))
    figure.patch.set_facecolor(SURFACE)
    camera_spec = figure.add_gridspec(
        2, 3, left=0.012, right=0.635, top=0.885, bottom=0.055, wspace=0.03, hspace=0.16
    )
    bev_spec = figure.add_gridspec(1, 1, left=0.705, right=0.985, top=0.885, bottom=0.095)
    camera_axes = [[figure.add_subplot(camera_spec[r, c]) for c in range(3)] for r in range(2)]
    bev_ax = figure.add_subplot(bev_spec[0, 0])

    draw_cameras(camera_axes, scene_path, sample["keyframe_index"], calibration)
    draw_bev(bev_ax, sample, horizon)

    figure.suptitle(
        f"{sample['scene_id']}  ({sample['split']})   keyframe {sample['keyframe_index']} "
        f"= t {sample['keyframe_index'] / 10:.1f} s        "
        f"ADE@3s {sample['ade']:.2f} m · FDE@3s {sample.get('fde@3s', float('nan')):.2f} m · "
        f"ego {sample['ego_speed']:.1f} m/s",
        fontsize=11, color=INK, y=0.965,
    )
    figure.text(0.012, 0.012,
                "orange box = the OpenScene-FOV crop actually fed to the model",
                fontsize=8, color=MUTED)
    figure.savefig(output_path, dpi=120, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--kitscenes-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples", nargs="*", default=None,
                        help="scene_id:keyframe pairs; default = every sample whose scene is on disk")
    parser.add_argument("--horizon", type=int, default=6, help="waypoints to draw (6 = 3 s)")
    args = parser.parse_args()

    results = pickle.load(open(args.results, "rb"))
    if args.samples:
        wanted = {(s.split(":")[0], int(s.split(":")[1])) for s in args.samples}
        chosen = [r for r in results if (r["scene_id"], r["keyframe_index"]) in wanted]
    else:
        chosen = [
            r for r in results
            if (args.kitscenes_root / "data" / r["split"] / r["scene_id"]).is_dir()
        ]
    if not chosen:
        raise SystemExit("no matching samples (are the scenes still on disk?)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sample in chosen:
        name = f"sample_{sample['scene_id'][:8]}_f{sample['keyframe_index']}.png"
        written += render_sample(sample, args.kitscenes_root, args.outdir / name, args.horizon)
    print(f"\n{written}/{len(chosen)} figures -> {args.outdir}")


if __name__ == "__main__":
    main()
