"""Draw the two VisionPilot-on-KITScenes diagnostic figures from a stages dump.

    python tools/benchmarks/kitscenes/visionpilot_stages.py --out traj.json   # slow, needs the model
    python tools/benchmarks/kitscenes/visionpilot_figures.py traj.json --out-dir figs

Figure 1 puts the timed trajectory next to the bare path geometry, per scene.
Figure 2 walks one sample through the four stages and lists what each assumes.

Split from the extractor on purpose: redrawing is a second, and re-running the
model over the history of eight samples is minutes, so a caption fix should not
cost an inference pass.

Colour comes from `paddock.viz.palette`, which fixes it per model *name*. Do not
recolour by hand -- these panels are meant to sit beside every other figure in
the project and be read without re-checking the legend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is chosen)

from paddock.viz.palette import ASSIGNED, GROUND_TRUTH  # noqa: E402

#: KITScenes scores the 3 s horizon: six waypoints at 0.5 s. The trajectory
#: carries eight (4 s); averaging all eight is a different, larger metric.
SCORED_WAYPOINTS = 6
WAYPOINT_DT = 0.5

VP = ASSIGNED["visionpilot"]
DJ = ASSIGNED["drivejepa"]


def ade(prediction, ground_truth, n: int = SCORED_WAYPOINTS) -> float:
    p = np.asarray(prediction, dtype=float)
    g = np.asarray(ground_truth, dtype=float)[:n]
    if len(p) < n:
        return float("nan")
    return float(np.linalg.norm(p[:n] - g, axis=1).mean())


def trajectories(data: dict, out: Path, reference: dict | None = None) -> None:
    """One panel per sample: human, Drive-JEPA, timed VisionPilot, geometry-only."""
    keys = list(data)
    cols = min(4, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.9 * cols, 5.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        v = data[key]
        human = np.asarray(v["human"], dtype=float)
        jepa = np.asarray(v["drivejepa"], dtype=float)
        timed = np.asarray(v["vp_fixed"], dtype=float)
        bare = np.asarray(v["rawpath_fixed"], dtype=float)

        if len(bare):
            ax.plot(
                bare[:, 1],
                bare[:, 0],
                "--^",
                color=VP,
                ms=5,
                lw=2.0,
                alpha=0.5,
                zorder=3,
                label="VisionPilot — geometry only,\nno speed schedule",
            )
        ax.plot(
            human[:, 1],
            human[:, 0],
            "-o",
            color=GROUND_TRUTH,
            ms=5.5,
            lw=2.6,
            zorder=6,
            label="human (recorded)",
        )
        ax.plot(jepa[:, 1], jepa[:, 0], "-s", color=DJ, ms=5, lw=2.2, zorder=5, label="Drive-JEPA")
        ax.plot(
            timed[:, 1],
            timed[:, 0],
            "-D",
            color=VP,
            ms=5,
            lw=2.6,
            zorder=7,
            label="VisionPilot — timed (paddock)",
        )
        ax.plot(0, 0, "ks", ms=9, zorder=8)

        title = f"{key}   ego {v['speed']:.1f} m/s\ntimed {ade(timed, human):.2f}   ·   geometry-only {ade(bare, human):.2f}"
        if reference and key in reference:
            title += f"   ·   reported {reference[key]:.2f}"
        ax.set_title(title, fontsize=10.5)
        if v["speed"] < 1.0:
            ax.annotate(
                "ego almost stopped:\nit travels 0.1 m in 3 s", xy=(-19, 30), fontsize=8.5, color="0.25"
            )
        ax.grid(alpha=0.3)
        ax.set_xlim(-21, 21)
        ax.set_ylim(-2, 55)
        ax.axhline(0, color="0.7", lw=0.8)

    for ax in axes[len(keys) :]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xlabel("lateral y (m)   —   positive = LEFT")
    for ax in axes[::cols]:
        ax.set_ylabel("forward x (m)")
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.94)

    fig.suptitle(
        "VisionPilot on KITScenes — a timed trajectory vs the bare path geometry\n"
        "ADE is the mean over the first six waypoints (3 s). Timed markers are 0.5 s apart, so tight "
        "spacing is a slow plan.  Ego frame: x forward, y positive to the left.",
        fontsize=13.5,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")


def pipeline(data: dict, out: Path, key: str) -> None:
    """One sample walked through the four stages, plus the assumption table."""
    v = data[key]
    fig = plt.figure(figsize=(19.5, 9.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.24)

    # 1 -- pixels to ground, both origins.
    ax = fig.add_subplot(grid[0, 0])
    on_road = np.asarray(v["points_fixed"], dtype=float)
    on_pod = np.asarray(v["points_podorigin"], dtype=float)
    ax.axhspan(-50, 0, color="#b2182b", alpha=0.07, zorder=0)
    ax.scatter(
        on_road[:, 1],
        on_road[:, 0],
        s=26,
        color=VP,
        zorder=4,
        label=f"origin on the road ({len(on_road)} pts)",
    )
    if len(on_pod):
        ax.scatter(
            on_pod[:, 1],
            on_pod[:, 0],
            s=34,
            marker="x",
            color="#b2182b",
            zorder=5,
            label=f"origin on the roof pod ({len(on_pod)} pts)",
        )
    ax.axhline(0, color="0.55", lw=1.4)
    ax.plot(0, 0, "ks", ms=9, zorder=6)
    ax.annotate(
        "BEHIND the car — the whole path",
        xy=(0.05, 0.30),
        xycoords="axes fraction",
        fontsize=10.5,
        color="#b2182b",
        fontweight="bold",
    )
    ax.annotate(
        "AHEAD", xy=(0.03, 0.80), xycoords="axes fraction", fontsize=10.5, color="#1b7837", fontweight="bold"
    )
    ax.set_title(
        "1 · Project AutoSteer pixels onto the road plane\n"
        "Assumes: flat road, and z = 0 in the ego frame IS the road",
        fontsize=10.5,
    )
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_ylim(-45, 45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    # 2 -- the fitted curve, and where its evidence stops.
    ax = fig.add_subplot(grid[0, 1])
    bare = np.asarray(v["rawpath_fixed"], dtype=float)
    ax.scatter(on_road[:, 1], on_road[:, 0], s=22, color=VP, alpha=0.45, zorder=3, label="projected points")
    ax.plot(
        bare[:, 1],
        bare[:, 0],
        "--^",
        color=VP,
        lw=2.2,
        ms=5,
        zorder=4,
        label=f"fitted path, evidence to {v['x_max_fixed']:.0f} m",
    )
    ax.axhline(v["x_max_fixed"], color="0.6", ls=":", lw=1.2)
    ax.annotate(
        "past here the curve is a\nstraight tangent, not evidence",
        xy=(-9, v["x_max_fixed"] + 1.5),
        fontsize=8.5,
        color="0.3",
    )
    ax.plot(0, 0, "ks", ms=9)
    ax.set_title(
        "2 · Fit y = ax² + bx + c, near points weighted\n"
        "Assumes: the path is a parabola within 50 m. NO timing yet",
        fontsize=10.5,
    )
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_ylim(-6, 40)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    # 3 -- the speed schedule, and what the cruise target does to it.
    ax = fig.add_subplot(grid[0, 2])
    human = np.asarray(v["human"], dtype=float)
    timed = np.asarray(v["vp_fixed"], dtype=float)
    stale = np.asarray(v["vp_runstart"], dtype=float)
    t = np.arange(1, len(timed) + 1) * WAYPOINT_DT
    ax.plot(
        np.arange(1, len(human) + 1) * WAYPOINT_DT,
        human[:, 0],
        "-o",
        color=GROUND_TRUTH,
        lw=2.4,
        ms=5,
        label="human",
    )
    ax.plot(
        t,
        timed[:, 0],
        "-D",
        color=VP,
        lw=2.4,
        ms=5,
        label=f"IDM, cruise target {v['set_speed_fixed']:.1f} m/s\n(latched at the scored frame)",
    )
    ax.plot(
        t,
        stale[:, 0],
        "--v",
        color="#7f7f7f",
        lw=2.0,
        ms=5,
        label=f"IDM, cruise target {v['set_speed_runstart']:.1f} m/s\n(latched 4.8 s earlier)",
    )
    ax.axvline(3.0, color="0.6", ls=":", lw=1.2)
    ax.annotate("scored to 3 s", xy=(3.02, 2), fontsize=8.5, color="0.35")
    ax.set_title(
        "3 · Walk an IDM speed schedule forward\n"
        "Assumes: a cruise target. KITScenes posts no limit, so it is latched",
        fontsize=10.5,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("distance travelled s(t) (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # 4 -- the result.
    ax = fig.add_subplot(grid[1, 0])
    jepa = np.asarray(v["drivejepa"], dtype=float)
    ax.plot(human[:, 1], human[:, 0], "-o", color=GROUND_TRUTH, lw=2.6, ms=5.5, zorder=6, label="human")
    ax.plot(jepa[:, 1], jepa[:, 0], "-s", color=DJ, lw=2.2, ms=5, zorder=5, label="Drive-JEPA")
    ax.plot(bare[:, 1], bare[:, 0], "--^", color=VP, lw=2.0, ms=5, alpha=0.5, zorder=3, label="geometry only")
    ax.plot(timed[:, 1], timed[:, 0], "-D", color=VP, lw=2.6, ms=5, zorder=7, label="timed (paddock)")
    ax.plot(0, 0, "ks", ms=9)
    ax.set_title(f"4 · waypoint(t) = [ s(t) , fitted_lateral(s(t)) ]\n{key}", fontsize=10.5)
    ax.set_xlabel("lateral y (m)")
    ax.set_ylabel("forward x (m)")
    ax.set_xlim(-12, 12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")

    # 5 -- every assumption, and whether it holds here.
    ax = fig.add_subplot(grid[1, 1:])
    ax.axis("off")
    table = [
        (
            "What the network gives",
            "64 lateral positions at fixed image rows, + confidence.",
            "no metres, no time",
            "ok",
        ),
        (
            "Which points to keep",
            "confidence >= 0.5  (about 30 of 64 survive)",
            "a validity flag, not 'road is clear'",
            "ok",
        ),
        (
            "Where the road is",
            "flat plane at z = 0 of the EGO frame",
            "KITScenes' ego frame is the ROOF POD, 2.2 m up",
            "bad",
        ),
        (
            "Camera height",
            "read from calib.json T_to_reference",
            "z = -0.183 m as shipped -> rays run backwards",
            "bad",
        ),
        (
            "Path shape",
            "quadratic, near points weighted, capped 50 m",
            "tangent past the last real point",
            "ok",
        ),
        ("Current speed", "from the recorded ego poses", "measured, not assumed", "ok"),
        (
            "Cruise target",
            "no posted limit in KITScenes -> max(speed, 1.0)",
            "latched 4.8 s early pinned 39/223 at 1 m/s",
            "bad",
        ),
        (
            "Horizon",
            "stack plans 1 s; we sample to 4 s, score to 3 s",
            "two thirds of the score is extrapolation",
            "warn",
        ),
    ]
    colour = {"ok": "#1b7837", "bad": "#b2182b", "warn": "#b8860b"}
    ax.text(0.0, 1.02, "Every assumption behind that number", fontsize=12.5, fontweight="bold", va="top")
    y = 0.90
    for name, how, note, kind in table:
        ax.text(0.005, y, "●", color=colour[kind], fontsize=13, va="top")
        ax.text(0.035, y, name, fontsize=9.8, fontweight="bold", va="top")
        ax.text(0.235, y, how, fontsize=9.4, va="top")
        ax.text(0.635, y, note, fontsize=9.4, va="top", color=colour[kind])
        y -= 0.118
    ax.text(
        0.005,
        y - 0.01,
        "red = broken on KITScenes until the ground-plane and set-point fixes   ·   "
        "amber = a real limit that remains",
        fontsize=9,
        color="0.3",
        va="top",
    )

    fig.suptitle(
        "How VisionPilot's KITScenes trajectory is actually built — and what each step assumes",
        fontsize=14,
        y=0.975,
    )
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path, help="JSON written by visionpilot_stages.py")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--pipeline-sample", default=None, help="which sample the stage walkthrough uses (default: the first)"
    )
    parser.add_argument(
        "--reference", type=Path, default=None, help="optional JSON of {sample: ADE} to print alongside ours"
    )
    args = parser.parse_args()

    data = json.loads(args.stages.read_text())
    reference = json.loads(args.reference.read_text()) if args.reference else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajectories(data, args.out_dir / "visionpilot_kitscenes_trajectories.png", reference)
    pipeline(
        data, args.out_dir / "visionpilot_kitscenes_pipeline.png", args.pipeline_sample or next(iter(data))
    )


if __name__ == "__main__":
    main()
