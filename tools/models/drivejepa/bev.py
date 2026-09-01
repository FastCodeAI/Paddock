"""Render BEV plots of Drive-JEPA predicted vs. human trajectories.

Reads the pickle written by run_inference.py, so plots can be restyled without
re-running the model.

    python scripts/inference/visualize_bev.py                  # one PNG per scene
    python scripts/inference/visualize_bev.py --grid           # + combined contact sheet
    python scripts/inference/visualize_bev.py --cameras        # + 8-camera reference view
    python scripts/inference/visualize_bev.py --velocities     # + velocity arrows on traffic
"""

import argparse
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from matplotlib.lines import Line2D
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.common.enums import BoundingBoxIndex
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_annotations_to_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, CAMERAS_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_all_ax, configure_ax, configure_bev_ax
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf

DEVKIT_ROOT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
CONFIG_ROOT = DEVKIT_ROOT / "navsim/planning/script/config/common"
DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])

# Local copies of navsim's configs (never mutate the shared dicts -- other navsim
# plotting code reads them). Both trajectories default to identical styling, so a
# good prediction hides the ground truth underneath it: draw the human thicker and
# below, the prediction dashed and on top, so both stay readable when they coincide.
HUMAN_STYLE: Dict[str, Any] = {**TRAJECTORY_CONFIG["human"], "line_width": 3.5, "marker_size": 6, "zorder": 3}
AGENT_STYLE: Dict[str, Any] = {
    **TRAJECTORY_CONFIG["agent"],
    "line_style": "--",
    "line_width": 2.0,
    "marker_size": 4,
    "zorder": 4,
}
VELOCITY_COLOR = "#333333"


def _cameras_only_sensor_config(frame_idx: int) -> SensorConfig:
    """All 8 cameras for one frame, and explicitly no LiDAR.

    SensorConfig.build_all_sensors() would also request lidar_pc, which the
    perception-based agent never reads (it declares lidar_pc=[]) and which is
    therefore not downloaded -- asking for it raises FileNotFoundError.
    """
    frames = [frame_idx]
    return SensorConfig(
        cam_f0=frames,
        cam_l0=frames,
        cam_l1=frames,
        cam_l2=frames,
        cam_r0=frames,
        cam_r1=frames,
        cam_r2=frames,
        cam_b0=frames,
        lidar_pc=[],
    )


def build_scene_loader(split: str, sensors: bool) -> SceneLoader:
    """BEV needs map + annotations only; cameras are loaded only for the reference view."""
    scene_filter: SceneFilter = instantiate(
        OmegaConf.load(CONFIG_ROOT / f"train_test_split/scene_filter/{split}.yaml")
    )
    sensor_config = (
        _cameras_only_sensor_config(scene_filter.num_history_frames - 1)
        if sensors
        else SensorConfig.build_no_sensors()
    )
    return SceneLoader(
        data_path=DATA_ROOT / "navsim_logs/test",
        sensor_blobs_path=DATA_ROOT / "sensor_blobs/test",
        synthetic_scenes_path=DATA_ROOT / "synthetic_scenes",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def _to_trajectory(poses, sampling: Dict[str, float]) -> Trajectory:
    return Trajectory(
        poses,
        TrajectorySampling(num_poses=int(sampling["num_poses"]), interval_length=sampling["interval_length"]),
    )


def _legend(ax: plt.Axes, velocities: bool = False, fontsize: int = 9, loc: str = "upper right") -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=HUMAN_STYLE["line_color"],
            lw=3.5,
            marker="o",
            markeredgecolor="black",
            markersize=6,
            label="Human (ground truth)",
        ),
        Line2D(
            [0],
            [0],
            color=AGENT_STYLE["line_color"],
            lw=2,
            ls="--",
            marker="o",
            markeredgecolor="black",
            markersize=4,
            label="Drive-JEPA (predicted)",
        ),
    ]
    if velocities:
        handles.append(Line2D([0], [0], color=VELOCITY_COLOR, lw=1.5, label="Velocity (1 s of travel)"))
    ax.legend(handles=handles, loc=loc, framealpha=0.9, fontsize=fontsize)


def add_velocity_arrows(ax: plt.Axes, annotations, seconds: float = 1.0, min_speed: float = 0.5) -> None:
    """Draw each tracked object's velocity as an arrow spanning `seconds` of travel.

    navsim's own BEV draws a fixed-length heading stub on every box, so a parked
    car and a fast-moving one look identical. velocity_3d is stored in the ego
    frame (navsim_scenario_utils rotates it by the ego heading to globalise it),
    so its x/y components are used directly here. Plot axes are (ego_y, ego_x):
    horizontal is sideways, vertical is forward.
    """
    for box, velocity in zip(annotations.boxes, annotations.velocity_3d):
        vx, vy = float(velocity[0]), float(velocity[1])
        if math.hypot(vx, vy) < min_speed:  # skip parked/static objects
            continue
        x, y = float(box[BoundingBoxIndex.X]), float(box[BoundingBoxIndex.Y])
        ax.arrow(
            y,
            x,
            vy * seconds,
            vx * seconds,
            width=0.12,
            head_width=0.7,
            head_length=0.9,
            color=VELOCITY_COLOR,
            alpha=0.9,
            length_includes_head=True,
            zorder=5,
        )


def draw_bev(
    ax: plt.Axes,
    scene,
    result: Dict[str, Any],
    legend: bool = True,
    match_horizon: bool = True,
    velocities: bool = False,
) -> None:
    """Map + annotations + both trajectories, in the ego frame."""
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    add_configured_bev_on_ax(ax, scene.map_api, frame)

    human_poses = result["human_poses"]
    human_sampling = dict(result["human_sampling"])
    if match_horizon:
        # the scene's future runs 5s but the agent predicts 4s -- truncate so both
        # curves cover the same window and can be compared point-for-point
        human_poses = human_poses[: len(result["predicted_poses"])]
        human_sampling["num_poses"] = len(human_poses)

    add_trajectory_to_bev_ax(ax, _to_trajectory(human_poses, human_sampling), HUMAN_STYLE)
    add_trajectory_to_bev_ax(
        ax, _to_trajectory(result["predicted_poses"], result["predicted_sampling"]), AGENT_STYLE
    )
    if velocities:
        add_velocity_arrows(ax, frame.annotations)

    configure_bev_ax(ax)
    configure_ax(ax)
    if legend:
        _legend(ax, velocities)


# The 3x3 arrangement mirrors the physical camera rig -- left column looks left,
# top row looks forward. The centre holds the BEV, as in navsim's own layout, but
# with the predicted/human trajectories drawn on it rather than a bare map.
CAMERA_LAYOUT = [
    [("cam_l0", "front-left"), ("cam_f0", "front"), ("cam_r0", "front-right")],
    [("cam_l1", "left"), None, ("cam_r1", "right")],
    [("cam_l2", "rear-left"), ("cam_b0", "rear"), ("cam_r2", "rear-right")],
]


def plot_cameras_with_bev(
    scene, frame_idx: int, result: Dict[str, Any], match_horizon: bool = True, velocities: bool = False
):
    """All 8 cameras with projected 3D boxes, around a BEV showing both trajectories."""
    frame = scene.frames[frame_idx]
    fig, axes = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    for row_idx, row in enumerate(CAMERA_LAYOUT):
        for col_idx, entry in enumerate(row):
            if entry is None:
                continue
            attr, _ = entry
            add_annotations_to_camera_ax(
                axes[row_idx, col_idx], getattr(frame.cameras, attr), frame.annotations
            )

    configure_all_ax(axes)
    # titles are applied after configure_all_ax so they survive it
    for row_idx, row in enumerate(CAMERA_LAYOUT):
        for col_idx, entry in enumerate(row):
            if entry is not None:
                axes[row_idx, col_idx].set_title(entry[1], fontsize=9)

    # centre cell: the same BEV as the standalone figure, drawn last so
    # configure_all_ax cannot strip its axis settings
    centre = axes[1, 1]
    centre.clear()
    draw_bev(centre, scene, result, legend=False, match_horizon=match_horizon, velocities=velocities)
    centre.set_title("BEV", fontsize=9)
    # the centre cell is small, so keep the legend compact and low-left where the
    # trajectories (which run forward, i.e. upward) will not be covered
    _legend(centre, velocities, fontsize=5.5, loc="lower left")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.08, left=0.01, right=0.99, top=0.96, bottom=0.01)
    return fig, axes


def displacement_errors(result: Dict[str, Any]) -> Dict[str, float]:
    """Final and average displacement error, in metres.

    The prediction covers 4s and the human future 5s, so both are truncated to
    the overlapping horizon -- comparing the two final poses directly would be
    measuring different points in time. Both use a 0.5s interval, so index i
    refers to the same timestamp in each.
    """
    import numpy as np

    pred, human = result["predicted_poses"], result["human_poses"]
    assert result["predicted_sampling"]["interval_length"] == result["human_sampling"]["interval_length"], (
        "trajectories are sampled at different intervals; index-wise comparison is invalid"
    )
    n = min(len(pred), len(human))
    dists = np.linalg.norm(pred[:n, :2] - human[:n, :2], axis=1)
    return {
        "fde": float(dists[-1]),
        "ade": float(dists.mean()),
        "horizon_s": n * result["predicted_sampling"]["interval_length"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    exp_root = Path(os.environ["NAVSIM_EXP_ROOT"])
    parser.add_argument("--results", type=Path, default=exp_root / "inference/results.pkl")
    parser.add_argument("--output-dir", type=Path, default=exp_root / "inference/plots")
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--grid", action="store_true", help="also write one combined contact sheet")
    parser.add_argument("--cameras", action="store_true", help="also write the 8-camera + BEV reference view")
    parser.add_argument("--velocities", action="store_true", help="draw velocity arrows on moving traffic")
    parser.add_argument(
        "--full-human",
        action="store_true",
        help="plot the human's full 5s future instead of truncating to the agent's 4s",
    )
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()
    match_horizon = not args.full_human

    with open(args.results, "rb") as f:
        results: List[Dict[str, Any]] = pickle.load(f)
    print(f"loaded {len(results)} results from {args.results}")

    scene_loader = build_scene_loader(args.split, sensors=False)
    # sensors are expensive to load, so build this loader once and only if needed
    camera_loader = build_scene_loader(args.split, sensors=True) if args.cameras else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for i, result in enumerate(results, 1):
        scene = scene_loader.get_scene_from_token(result["token"])
        err = displacement_errors(result)
        errors.append(err)

        fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        draw_bev(ax, scene, result, match_horizon=match_horizon, velocities=args.velocities)
        ax.set_title(
            f"{result['token']}\nFDE {err['fde']:.2f} m · ADE {err['ade']:.2f} m  (over {err['horizon_s']:.1f}s)",
            fontsize=10,
        )
        out = args.output_dir / f"bev_{result['token']}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i}/{len(results)}] {out.name}  FDE {err['fde']:.2f} m  ADE {err['ade']:.2f} m")

        if camera_loader is not None:
            cam_scene = camera_loader.get_scene_from_token(result["token"])
            cfig, _ = plot_cameras_with_bev(
                cam_scene,
                cam_scene.scene_metadata.num_history_frames - 1,
                result,
                match_horizon=match_horizon,
                velocities=args.velocities,
            )
            cam_out = args.output_dir / f"cameras_{result['token']}.png"
            cfig.savefig(cam_out, dpi=args.dpi, bbox_inches="tight")
            plt.close(cfig)
            print(f"          {cam_out.name}")

    if args.grid and results:
        cols = min(4, len(results))
        rows = (len(results) + cols - 1) // cols
        w, h = BEV_PLOT_CONFIG["figure_size"]
        fig, axes = plt.subplots(rows, cols, figsize=(w * cols / 1.6, h * rows / 1.6))
        axes = [axes] if rows * cols == 1 else list(axes.flat)
        for ax, result in zip(axes, results):
            scene = scene_loader.get_scene_from_token(result["token"])
            draw_bev(ax, scene, result, legend=False, match_horizon=match_horizon, velocities=args.velocities)
            ax.set_title(
                f"{result['token'][:10]}…  FDE {displacement_errors(result)['fde']:.1f} m", fontsize=8
            )
        for ax in axes[len(results) :]:
            ax.axis("off")
        _legend(axes[0], args.velocities)
        fig.tight_layout()
        grid_path = args.output_dir / "bev_grid.png"
        fig.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\ncontact sheet -> {grid_path}")

    if errors:
        mean_fde = sum(e["fde"] for e in errors) / len(errors)
        mean_ade = sum(e["ade"] for e in errors) / len(errors)
        print(f"\nover {len(errors)} scenes: mean FDE {mean_fde:.2f} m · mean ADE {mean_ade:.2f} m")
    print(f"plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
