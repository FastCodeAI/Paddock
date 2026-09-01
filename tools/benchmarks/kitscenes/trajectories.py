"""Plot Drive-JEPA's KITScenes predictions against the recorded human trajectory.

Reads the pickle written by run_kitscenes_inference.py, so plots can be restyled
without re-running the model.  Deliberately free of navsim/nuplan imports -- the
KITScenes results carry everything needed, and the navsim BEV helpers assume
nuPlan map layers that do not exist here.

    python scripts/inference/visualize_kitscenes.py --results <pkl> --outdir <dir>

Produces two figures:

* ``kitscenes_trajectories.png`` -- a contact sheet of BEV panels spanning the
  error distribution (best / median / worst), so the failure mode is visible
  rather than summarised.
* ``kitscenes_error_growth.png`` -- displacement error against horizon for the
  model and a constant-velocity baseline, which is the chart that actually
  carries the finding.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots 1 and 2 (light surface #fcfcfb): worst adjacent
# CVD separation dE 24.7, normal-vision 33.6, both >= 3:1 on the surface.
HUMAN_COLOR = "#2a78d6"
PREDICTED_COLOR = "#eb6834"
BASELINE_COLOR = "#6b7280"
SURFACE = "#fcfcfb"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#d8d8d4"


def constant_velocity(sample: Dict[str, Any], num_poses: int) -> np.ndarray:
    """Straight-ahead dead reckoning at the keyframe speed.

    The simplest possible baseline: hold the current speed, hold the current
    heading.  A constant-turn-rate variant would be stronger on curves, so this
    is a floor rather than a competitive reference -- which is what makes it a
    meaningful bar for the model to clear.
    """
    speed = sample["ego_speed"]
    interval = sample["interval_length"]
    return np.stack([speed * interval * np.arange(1, num_poses + 1), np.zeros(num_poses)], axis=-1)


def _style_bev(ax: plt.Axes) -> None:
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=3)


def plot_contact_sheet(results: List[Dict[str, Any]], output_path: Path, columns: int = 4) -> None:
    """BEV panels sampled across the error distribution, worst first."""
    ordered = sorted(results, key=lambda r: r["ade"], reverse=True)
    # Span the distribution rather than showing only the tail: worst four, then
    # evenly spaced ranks through to the best.
    picks = list(range(4)) + list(np.linspace(4, len(ordered) - 1, columns * 2 - 4, dtype=int))
    chosen = [ordered[i] for i in dict.fromkeys(picks)]

    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.4 * rows))
    figure.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes).ravel()

    # One shared scale across panels. Per-panel autoscaling would silently rescale
    # every panel to its own extent, making a 50 m run and a 0.3 m creep look alike.
    extent_x = max(
        np.abs(np.concatenate([s["human_poses"][:, 0], s["predicted_poses"][:, 0]])).max() for s in chosen
    )
    extent_y = max(
        np.abs(np.concatenate([s["human_poses"][:, 1], s["predicted_poses"][:, 1]])).max() for s in chosen
    )
    limit_y = max(extent_y, extent_x * 0.35) * 1.15

    for ax, sample in zip(axes, chosen):
        human = sample["human_poses"][:, :2]
        predicted = sample["predicted_poses"][: len(human), :2]
        horizon = min(len(human), len(predicted))

        # BEV convention: x forward on the vertical axis, y left on the
        # horizontal, with the axis inverted so left-of-ego draws to the left.
        ax.plot(
            human[:horizon, 1],
            human[:horizon, 0],
            color=HUMAN_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="human",
            zorder=3,
        )
        ax.plot(
            predicted[:horizon, 1],
            predicted[:horizon, 0],
            color=PREDICTED_COLOR,
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=4,
            label="Drive-JEPA",
            zorder=4,
        )
        ax.scatter([0], [0], marker="s", s=26, color=INK, zorder=5)

        _style_bev(ax)
        ax.set_xlim(limit_y, -limit_y)  # inverted: left-of-ego draws to the left
        ax.set_ylim(-2.0, extent_x * 1.1)
        ax.set_title(
            f"{sample['scene_id'][:8]}  f{sample['keyframe_index']}\n"
            f"ADE {sample['ade']:.2f} m · {sample['ego_speed']:.1f} m/s",
            fontsize=8,
            color=INK,
            pad=6,
        )

    for ax in axes[len(chosen) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "Drive-JEPA zero-shot on KITScenes — predicted vs. recorded trajectory\n"
        "shared scale across panels · worst-first, then spanning the distribution",
        fontsize=12,
        color=INK,
        y=0.985,
    )
    # Legend under the title rather than at the foot, where it collided with the
    # shared x label.
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        bbox_to_anchor=(0.5, 0.928),
    )
    figure.supxlabel("lateral y (m), left ←", fontsize=9, color=MUTED, y=0.008)
    figure.supylabel("longitudinal x (m)", fontsize=9, color=MUTED, x=0.006)
    figure.tight_layout(rect=(0.02, 0.025, 1, 0.905))
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_error_growth(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Mean displacement error against horizon, model vs. dead reckoning."""
    interval = results[0]["interval_length"]
    horizon = min(len(r["human_poses"]) for r in results)
    times = interval * np.arange(1, horizon + 1)

    model_errors, baseline_errors = [], []
    for sample in results:
        human = sample["human_poses"][:horizon, :2]
        predicted = sample["predicted_poses"][:horizon, :2]
        model_errors.append(np.linalg.norm(predicted - human, axis=-1))
        baseline_errors.append(np.linalg.norm(constant_velocity(sample, horizon) - human, axis=-1))
    model = np.array(model_errors)
    baseline = np.array(baseline_errors)

    figure, ax = plt.subplots(figsize=(7.2, 4.4))
    figure.patch.set_facecolor(SURFACE)

    for values, color, label, style in (
        (model, PREDICTED_COLOR, "Drive-JEPA", "-"),
        (baseline, BASELINE_COLOR, "constant velocity", "--"),
    ):
        mean = values.mean(axis=0)
        # Standard error, not a percentile band: the claim is about the mean.
        error = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
        ax.fill_between(times, mean - error, mean + error, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            times,
            mean,
            color=color,
            linewidth=2.0,
            linestyle=style,
            marker="o",
            markersize=5,
            label=label,
            zorder=3,
        )

    # Direct labels at the horizon end instead of numbers on every point.
    for values, color, offset in ((model, PREDICTED_COLOR, 8), (baseline, BASELINE_COLOR, -14)):
        ax.annotate(
            f"{values.mean(axis=0)[-1]:.2f} m",
            (times[-1], values.mean(axis=0)[-1]),
            textcoords="offset points",
            xytext=(6, offset),
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("horizon (s)", fontsize=10, color=MUTED)
    ax.set_ylabel("mean displacement error (m)", fontsize=10, color=MUTED)
    ax.set_title(
        f"Drive-JEPA does not beat dead reckoning on KITScenes  ({len(results)} samples)",
        fontsize=12,
        color=INK,
        pad=12,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    results = pickle.load(open(args.results, "rb"))
    if not results:
        raise SystemExit(f"no samples in {args.results}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_contact_sheet(results, args.outdir / "kitscenes_trajectories.png", args.columns)
    plot_error_growth(results, args.outdir / "kitscenes_error_growth.png")
    print(f"{len(results)} samples -> {args.outdir}/kitscenes_trajectories.png")
    print(f"{len(results)} samples -> {args.outdir}/kitscenes_error_growth.png")


if __name__ == "__main__":
    main()
