"""Show all of Drive-JEPA's trajectory proposals for one sample, and which it picked.

The agent does not regress a trajectory: it generates a set of proposals, predicts
a NAVSIM PDM score for each with a learned head, and takes the argmax
(drive_jepa_model.py:85-90).  Plotting every proposal separates two very different
failure modes -- "could not produce a good trajectory" from "produced one and did
not select it".

    python scripts/inference/visualize_proposals.py \
        --scene <scene_id> --split <split> --keyframe <n> \
        --kitscenes-root <root> --output <png>

Needs a GPU and the scene on disk.  The predicted score is a learned proxy: at
inference the scorer reads only BEV features, so no map, annotations or simulator
are required -- which is why it runs on KITScenes at all.

LAYOUT NOTE
-----------
In BEV the proposals pile onto one another, because on a straight road they differ
almost entirely in how far they travel rather than in shape.  Two things fix that:
the BEV panel exaggerates the lateral axis (marked on the panel, since it breaks
the 1:1 aspect), and a distance-against-time panel shows the axis the proposals
actually differ on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

DEVKIT_ROOT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
os.chdir(DEVKIT_ROOT)
sys.path.insert(0, ".")

from scripts.inference.kitscenes_adapter import KITScenesSceneAdapter  # noqa: E402
from scripts.inference.run_kitscenes_inference import DEFAULT_CKPT, build_agent  # noqa: E402

HUMAN_COLOR = "#2a78d6"  # categorical slot 1
PICKED_COLOR = "#eb6834"  # slot 2
BEST_COLOR = "#1baf7a"  # slot 3
CROP_COLOR = "#eda100"
SURFACE = "#fcfcfb"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#d8d8d4"

CAMERA_STRIP = [
    ("camera_ring_front_left", "cam_l0"),
    ("camera_ring_front", "cam_f0"),
    ("camera_ring_front_right", "cam_r0"),
    ("camera_ring_rear_left", None),
    ("camera_ring_rear", "cam_b0"),
    ("camera_ring_rear_right", None),
]
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
TARGET_FOCAL, TARGET_CU, TARGET_CV = 1545.0, 960.0, 560.0
DISPLAY_WIDTH = 620


def collect(scene_path: Path, keyframe: int, horizon: int):
    """Run the agent once and return proposals, predicted scores and ground truth."""
    adapter = KITScenesSceneAdapter(scene_path)
    sample = adapter.build_sample(keyframe, num_future_frames=8, command_mode="gt")

    agent = build_agent(DEFAULT_CKPT, "drive_jepa_perception_based_agent")
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(sample.agent_input))
    features = {k: v.unsqueeze(0).cuda() for k, v in features.items()}
    features.update(past_ego_simulated_states=None, metric_cache=None, observation_interval=None)

    with torch.no_grad():
        output = agent.forward(features)

    proposals = output["proposals"][0, :, :horizon, :2].float().cpu().numpy()
    scores = output["pdm_score"][0].float().cpu().numpy()
    truth = sample.future_poses[:horizon, :2]
    errors = np.linalg.norm(proposals - truth[None], axis=2).mean(1)
    return proposals, scores, truth, errors


def draw_camera_strip(axes, scene_path: Path, keyframe: int) -> None:
    calibration = json.loads((scene_path / "calibration/calib.json").read_text())
    for ax, (sensor, slot) in zip(axes, CAMERA_STRIP):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        path = scene_path / sensor / f"{keyframe:010d}.jpg"
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center", color=MUTED, fontsize=8)
            continue
        image = Image.open(path)
        scale = DISPLAY_WIDTH / image.width
        ax.imshow(image.resize((DISPLAY_WIDTH, int(image.height * scale)), Image.BILINEAR))
        if slot:
            intr = calibration[sensor]["intrinsics"]
            zoom = TARGET_FOCAL / float(intr["focal_length"])
            left = (intr["principal_point_u"] - TARGET_CU / zoom) * scale
            top = (intr["principal_point_v"] - TARGET_CV / zoom) * scale
            ax.add_patch(
                Rectangle(
                    (left, top),
                    TARGET_WIDTH / zoom * scale,
                    TARGET_HEIGHT / zoom * scale,
                    fill=False,
                    edgecolor=CROP_COLOR,
                    linewidth=1.6,
                    zorder=3,
                )
            )
        ax.set_title(
            sensor.replace("camera_ring_", "") + (f" → {slot}" if slot else "  (unused)"),
            fontsize=7.5,
            color=INK if slot else MUTED,
            pad=2,
        )
        for spine in ax.spines.values():
            spine.set_color(GRID)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", default="overlap_train_val")
    parser.add_argument("--keyframe", type=int, required=True)
    parser.add_argument("--kitscenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=6, help="waypoints (6 = 3 s)")
    args = parser.parse_args()

    scene_path = args.kitscenes_root / "data" / args.split / args.scene
    if not scene_path.is_dir():
        raise SystemExit(f"scene not on disk: {scene_path}")

    proposals, scores, truth, errors = collect(scene_path, args.keyframe, args.horizon)
    picked, best = int(scores.argmax()), int(errors.argmin())
    print(
        f"{len(proposals)} proposals | picked #{picked} ADE {errors[picked]:.2f} "
        f"| best #{best} ADE {errors[best]:.2f} | rank of picked "
        f"{(errors < errors[picked]).sum()}/{len(proposals)}"
    )

    figure = plt.figure(figsize=(18.5, 8.2))
    figure.patch.set_facecolor(SURFACE)
    strip = figure.add_gridspec(1, 6, left=0.012, right=0.988, top=0.905, bottom=0.645, wspace=0.025)
    lower = figure.add_gridspec(1, 3, left=0.045, right=0.985, top=0.565, bottom=0.085, wspace=0.24)
    draw_camera_strip([figure.add_subplot(strip[0, i]) for i in range(6)], scene_path, args.keyframe)
    bev, profile, scatter = (figure.add_subplot(lower[0, i]) for i in range(3))

    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    ramp = plt.get_cmap("Greys")
    times = np.arange(1, args.horizon + 1) * 0.5

    # --- BEV, lateral axis exaggerated so the proposals do not collapse -------
    for i, proposal in enumerate(proposals):
        if i in (picked, best):
            continue
        bev.plot(
            proposal[:, 1], proposal[:, 0], color=ramp(0.25 + 0.55 * norm(scores[i])), linewidth=1.1, zorder=2
        )
    bev.plot(
        truth[:, 1],
        truth[:, 0],
        color=HUMAN_COLOR,
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="recorded",
        zorder=6,
    )
    bev.plot(
        proposals[best][:, 1],
        proposals[best][:, 0],
        color=BEST_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=5,
        label=f"lowest ADE ({errors[best]:.2f} m)",
        zorder=5,
    )
    bev.plot(
        proposals[picked][:, 1],
        proposals[picked][:, 0],
        color=PICKED_COLOR,
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=5,
        label=f"picked ({errors[picked]:.2f} m)",
        zorder=7,
    )
    span = np.abs(np.concatenate([proposals.reshape(-1, 2), truth])[:, 1]).max()
    bev.set_xlim(span * 1.2 + 0.5, -(span * 1.2 + 0.5))
    bev.set_ylim(-2, np.concatenate([proposals.reshape(-1, 2), truth])[:, 0].max() * 1.08)
    style(bev)
    bev.set_xlabel("lateral y (m), left ←   [axis exaggerated]", fontsize=9, color=MUTED)
    bev.set_ylabel("longitudinal x (m)", fontsize=9, color=MUTED)
    bev.set_title(
        f"all {len(proposals)} proposals, shaded by predicted score", fontsize=10, color=INK, loc="left"
    )
    bev.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")

    # --- Distance against time: the axis the proposals actually differ on -----
    for i, proposal in enumerate(proposals):
        if i in (picked, best):
            continue
        profile.plot(
            times,
            np.linalg.norm(proposal, axis=1),
            color=ramp(0.25 + 0.55 * norm(scores[i])),
            linewidth=1.1,
            zorder=2,
        )
    profile.plot(
        times,
        np.linalg.norm(truth, axis=1),
        color=HUMAN_COLOR,
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="recorded",
        zorder=6,
    )
    profile.plot(
        times,
        np.linalg.norm(proposals[best], axis=1),
        color=BEST_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="lowest ADE",
        zorder=5,
    )
    profile.plot(
        times,
        np.linalg.norm(proposals[picked], axis=1),
        color=PICKED_COLOR,
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=5,
        label="picked",
        zorder=7,
    )
    style(profile)
    profile.set_xlabel("time (s)", fontsize=9, color=MUTED)
    profile.set_ylabel("distance travelled (m)", fontsize=9, color=MUTED)
    profile.set_title("distance profile — where the proposals separate", fontsize=10, color=INK, loc="left")
    profile.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")

    # --- Does the predicted score track accuracy? -----------------------------
    scatter.scatter(
        scores, errors, s=44, color="#9aa0a6", edgecolor=SURFACE, linewidth=1.0, zorder=3, label="proposal"
    )
    scatter.scatter(
        [scores[picked]],
        [errors[picked]],
        s=155,
        color=PICKED_COLOR,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=5,
        label="picked (argmax score)",
    )
    scatter.scatter(
        [scores[best]],
        [errors[best]],
        s=155,
        color=BEST_COLOR,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=5,
        label="lowest ADE",
    )
    style(scatter)
    correlation = float(np.corrcoef(scores, errors)[0, 1])
    scatter.set_xlabel("predicted PDM score (what the model maximises)", fontsize=9, color=MUTED)
    scatter.set_ylabel("actual ADE@3s (m)", fontsize=9, color=MUTED)
    scatter.set_title(f"score vs. accuracy   r = {correlation:+.2f}", fontsize=10, color=INK, loc="left")
    scatter.legend(frameon=False, fontsize=8.5, labelcolor=INK)

    figure.suptitle(
        f"{args.scene}  ({args.split})  keyframe {args.keyframe} = t {args.keyframe / 10:.1f} s"
        f"   —   orange box on the cameras is the crop fed to the model",
        fontsize=11.5,
        color=INK,
        y=0.972,
    )
    figure.savefig(args.output, dpi=140, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
