"""One video per scenario showing every VisionPilot output at once.

The comparison videos answer "did it crash". This answers "what did it see and
what did it decide", which is the only view that makes a braking failure
explicable. Everything drawn here is an actual output of the node -- nothing is
recomputed or approximated for the picture.

    camera     AutoSpeed boxes, coloured by the stack's own threat classes
               (L1 red / L2 yellow / L3 blue / other green, matching
               visualization.cpp), AutoSteer's lane points, the fitted ego path
               and the planned trajectory projected back onto the road
    bev        the same path, lane points and plan from above, plus every box
               put on the ground plane and the fused CIPO range
    readouts   commanded acceleration with VisionPilot's own FCW/AEB bands,
               curvature, fused distance and closing speed, CIPO probability
    history    acceleration, speed and range against time, so "it braked at
               t = 2.6 s" is readable off the strip

Frames come from the recorded run, so this is a replay of what the node did,
not a re-simulation: the ego trajectory is the one that was actually driven.

    envs/ncap/bin/python make_visionpilot_video.py --runs outputs/runs1_fixed
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "/mnt/p4/naren/neuroncap/servers")
sys.path.insert(0, "/mnt/p4/naren/neuroncap")

from visionpilot_server import VisionPilotNode  # noqa: E402

BASE = Path("/mnt/p4/naren/neuroncap")
WEIGHTS = Path("/shakaal_hdd_1/workspace/naren/renesas/vision_pilot/VisionPilot/modules/models/weights")
FFMPEG = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/.venv/bin/ffmpeg"
SCENARIOS = ("stationary", "frontal", "side")
CANBUS_VELOCITY = 13
STEP_S = 0.1

# VisionPilot's own palette for the threat classes, from visualization.cpp.
CLASS_COLOR = {0: "#50c850", 1: "#ff2020", 2: "#ffd200", 3: "#00c8dc"}
CLASS_NAME = {0: "other", 1: "L1 same lane", 2: "L2 cut-in", 3: "L3"}

BG = "#0c0c0e"
FG = "#e8e8ea"
DIM = "#8a8a92"


def encode(image: np.ndarray) -> str:
    buff = io.BytesIO()
    torch.save(torch.from_numpy(image.copy()), buff)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def ground_to_image(points: np.ndarray, K: np.ndarray, cam2ego: np.ndarray) -> np.ndarray:
    """(x forward, y left) on the road -> pixels, dropping anything behind the camera."""
    if len(points) == 0:
        return np.empty((0, 2))
    ego = np.c_[points, np.zeros(len(points)), np.ones(len(points))]
    cam = (np.linalg.inv(cam2ego) @ ego.T).T
    infront = cam[:, 2] > 0.5
    uv = (K @ cam[infront, :3].T).T
    return uv[:, :2] / uv[:, 2:3]


def style(ax, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#2a2a30")
    ax.tick_params(colors=DIM, labelsize=7)
    if title:
        ax.set_title(title, color=FG, fontsize=9, loc="left", pad=4)


def replay(node: VisionPilotNode, run: Path, calib: dict) -> list[dict]:
    """Drive the node through the run's frames, keeping every output."""
    K = np.asarray(calib["CAM_FRONT"]["K"])
    cam2ego = np.asarray(calib["CAM_FRONT"]["cam2ego"])

    ego_poses = json.loads((run / "ego_poses.json").read_text())
    frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))
    stamps = [int(p.stem) for p in frames]
    positions = np.array([np.asarray(ego_poses[str(t)])[:2, 3] for t in stamps])
    dts = np.diff(np.asarray(stamps, dtype=np.float64)) / 1e6
    speeds = np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1) / dts]
    speeds[0] = speeds[1]

    node.reset()
    node.debug_detail = True
    out = []
    for i, path in enumerate(frames):
        image = np.asarray(Image.open(path).convert("RGB"))
        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = speeds[i]
        body = {
            "images": {"CAM_FRONT": encode(image)},
            "ego2world": np.asarray(ego_poses[str(stamps[i])]).tolist(),
            "canbus": canbus.tolist(),
            "timestamp": stamps[i],
            "command": 2,
            "calibration": {
                "camera2image": {"CAM_FRONT": K.tolist()},
                "camera2ego": {"CAM_FRONT": cam2ego.tolist()},
                "lidar2ego": np.eye(4).tolist(),
            },
        }
        result = node.infer(body)
        out.append(
            {
                "t": (stamps[i] - stamps[0]) / 1e6,
                "image": path,
                "speed": float(speeds[i]),
                "trajectory": np.asarray(result["trajectory"]),
                **result["debug"],
            }
        )
        print(f"    t={out[-1]['t']:4.1f}s  a={out[-1]['accel_cmd']:6.2f}  "
              f"d={out[-1]['fused_distance_m']:6.1f}  {out[-1]['warning'] or ''}", flush=True)
    return out


def render(rows: list[dict], scenario: str, calib: dict, out_dir: Path) -> None:
    K = np.asarray(calib["CAM_FRONT"]["K"])
    cam2ego = np.asarray(calib["CAM_FRONT"]["cam2ego"])
    times = np.array([r["t"] for r in rows])
    accels = np.array([r["accel_cmd"] for r in rows])
    speeds = np.array([r["speed"] for r in rows])
    ranges = np.array([r["fused_distance_m"] for r in rows])

    for i, row in enumerate(rows):
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
        gs = GridSpec(2, 3, figure=fig, width_ratios=[2.45, 1.12, 0.92],
                      height_ratios=[1.0, 0.40], hspace=0.34, wspace=0.15,
                      left=0.022, right=0.988, top=0.930, bottom=0.065)

        # ── camera ────────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        image = np.asarray(Image.open(row["image"]).convert("RGB"))
        ax.imshow(image)
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        style(ax, "CAM_FRONT — AutoSpeed boxes, AutoSteer lane points, planned path")

        for det, rng in zip(row["detections"], row["box_ranges"]):
            x1, y1, x2, y2, score, cls = det
            colour = CLASS_COLOR.get(int(cls), "#888888")
            lw = 2.6 if int(cls) in (1, 2) else 1.2
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, lw=lw))
            if int(cls) in (1, 2):
                label = f"{CLASS_NAME[int(cls)]} {score:.2f}"
                if rng > 0:
                    label += f"  {rng:.0f} m"
                ax.text(min(max(x1, 4), image.shape[1] - 190), max(y1 - 8, 14), label,
                        color=colour, fontsize=9, weight="bold")

        lane_uv = np.asarray(row["lane_uv"])
        if len(lane_uv):
            ax.plot(lane_uv[:, 0], lane_uv[:, 1], ".", color="#ffe000", ms=5, label="AutoSteer points")

        traj = row["trajectory"]
        traj_uv = ground_to_image(traj, K, cam2ego)
        if len(traj_uv):
            ax.plot(traj_uv[:, 0], traj_uv[:, 1], "-o", color="#00ff96", lw=2.4, ms=5,
                    label="planned trajectory")
        if row["path_coeffs"] is not None:
            xs = np.linspace(2, max(6.0, row["path_x_max"]), 40)
            fit_uv = ground_to_image(np.stack([xs, np.polyval(row["path_coeffs"], xs)], axis=1), K, cam2ego)
            if len(fit_uv):
                ax.plot(fit_uv[:, 0], fit_uv[:, 1], "--", color="#ff9a3c", lw=1.6, label="fitted ego path")
        leg = ax.legend(loc="lower left", fontsize=8, facecolor="#141418", edgecolor="#2a2a30")
        for text in leg.get_texts():
            text.set_color(FG)

        # ── bird's eye ────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        style(ax, "Bird's eye (ego at origin)")
        ax.set_xlim(-12, 12); ax.set_ylim(-2, 55)
        ax.set_xlabel("left (m)", color=DIM, fontsize=8)
        ax.set_ylabel("forward (m)", color=DIM, fontsize=8)
        ax.grid(color="#1e1e24", lw=0.6)
        ax.add_patch(Rectangle((-0.9, -2.0), 1.8, 4.5, color="#3a6ea5", alpha=0.85))

        lane_ground = np.asarray(row["lane_ground"])
        if len(lane_ground):
            ax.plot(lane_ground[:, 1], lane_ground[:, 0], ".", color="#ffe000", ms=4)
        if row["path_coeffs"] is not None:
            xs = np.linspace(0, max(6.0, row["path_x_max"]), 60)
            ax.plot(np.polyval(row["path_coeffs"], xs), xs, "--", color="#ff9a3c", lw=1.6)
        for det, ground in zip(row["detections"], row["box_ground"]):
            if ground[0] < -900:  # ray never met the road plane
                continue
            cls = int(det[5])
            ax.plot(ground[1], ground[0], "o", color=CLASS_COLOR.get(cls, "#888"),
                    ms=10 if cls in (1, 2) else 5, mec="#000", mew=0.5)
        if row["has_cipo"] and row["fused_distance_m"] < 80:
            ax.axhline(row["fused_distance_m"], color="#ff4d4d", lw=1.4, ls=":")
            ax.text(-11.4, row["fused_distance_m"] + 0.9,
                    f"fused CIPO {row['fused_distance_m']:.1f} m", color="#ff7a7a", fontsize=8)
        ax.plot(traj[:, 1], traj[:, 0], "-o", color="#00ff96", lw=2.2, ms=5)

        # ── readouts ──────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
        warn = row["warning"]
        warn_colour = {"AEB": "#ff3b3b", "FCW": "#ffb02e"}.get(warn, "#4b8b5a")
        lines = [
            ("LONGITUDINAL", None, None),
            ("commanded accel", f"{row['accel_cmd']:+.2f} m/s²", warn_colour),
            ("warning", warn or "none", warn_colour),
            ("ego speed", f"{row['speed']:.2f} m/s", FG),
            ("set speed", f"{row['set_speed_ms']:.2f} m/s", DIM),
            ("", None, None),
            ("PERCEPTION", None, None),
            ("fused CIPO range", f"{row['fused_distance_m']:.1f} m", FG),
            ("fused closing", f"{-row['fused_velocity_ms']:+.2f} m/s", FG),
            ("fusion ±1σ", f"{row['fused_stddev_m']:.1f} m", DIM),
            ("CIPO present", "yes" if row["has_cipo"] else "no", FG),
            ("cut-in (L2 nearer)", "yes" if row["cut_in"] else "no",
             "#ffd200" if row["cut_in"] else DIM),
            ("", None, None),
            ("NETWORKS", None, None),
            ("AutoSpeed boxes", f"{row['autospeed_boxes']}", FG),
            ("AutoSpeed CIPO", f"{row['autospeed_cipo_m']:.1f} m"
             if row["autospeed_cipo_m"] is not None else "none (no L1/L2)",
             FG if row["autospeed_cipo_m"] is not None else "#ff7a7a"),
            ("AutoDrive range", f"{row['autodrive_m']:.1f} m", FG),
            ("AutoDrive conf", f"{row['autodrive_prob']:.2f}", FG),
            ("AutoSteer kept", f"{row['path_points']}/64", FG),
            ("", None, None),
            ("LATERAL", None, None),
            ("AutoDrive curvature", f"{row['curvature']:+.4f} 1/m", FG),
            ("path extent", f"{row['path_extent_m']:.0f} m", DIM),
            ("lateral @ 3 s", f"{traj[-1, 1]:+.2f} m", FG),
        ]
        y = 0.985
        for label, value, colour in lines:
            if value is None:
                if label:
                    ax.text(0.0, y, label, color="#5f8fbf", fontsize=8.5, weight="bold",
                            transform=ax.transAxes)
                y -= 0.036
                continue
            ax.text(0.0, y, label, color=DIM, fontsize=9, transform=ax.transAxes)
            ax.text(1.0, y, value, color=colour, fontsize=10, weight="bold",
                    ha="right", transform=ax.transAxes)
            y -= 0.040

        # ── history strips ────────────────────────────────────────────────────
        for col, (series, label, colour, extra) in enumerate(
            [
                (accels, "commanded acceleration (m/s²)", "#ff9a3c", "bands"),
                (speeds, "ego speed (m/s)", "#5fd0ff", None),
                (ranges, "fused CIPO range (m)", "#ff6f6f", None),
            ]
        ):
            ax = fig.add_subplot(gs[1, col])
            style(ax, label)
            if extra == "bands":
                lo = min(-6.0, float(series.min()) * 1.15)
                hi = max(2.0, float(series.max()) * 1.15)
                ax.set_ylim(lo, hi)
                ax.axhspan(-5.0, -3.0, color="#ffb02e", alpha=0.18)
                ax.axhspan(lo, -5.0, color="#ff3b3b", alpha=0.18)
                ax.text(0.015, 0.06, "FCW band / AEB band", color=DIM, fontsize=7,
                        transform=ax.transAxes)
            ax.plot(times, series, color=colour, lw=1.0, alpha=0.35)
            ax.plot(times[: i + 1], series[: i + 1], color=colour, lw=2.0)
            ax.plot(times[i], series[i], "o", color=colour, ms=6)
            ax.set_xlim(times[0], times[-1])
            ax.set_xlabel("t (s)", color=DIM, fontsize=8)

        fig.suptitle(
            f"VisionPilot  ·  scene-0103  {scenario}  ·  t = {row['t']:.1f} s",
            color=FG, fontsize=15, x=0.028, ha="left", y=0.975,
        )
        fig.text(0.985, 0.975,
                 "all values are node outputs — replay of the recorded run",
                 color=DIM, fontsize=9, ha="right")
        fig.savefig(out_dir / f"{i:05d}.png", facecolor=BG)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/runs1_fixed")
    ap.add_argument("--out", default="videos_visionpilot")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    args = ap.parse_args()

    calib = json.loads((BASE / "validation/calib_0103.json").read_text())
    out_dir = BASE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    node = VisionPilotNode(WEIGHTS, ["CPUExecutionProvider"])

    for scenario in args.scenarios:
        run = BASE / args.runs / "visionpilot" / f"{scenario}-0103"
        if not (run / "CAM_FRONT").is_dir():
            print(f"skip {scenario}: no frames under {run}")
            continue
        print(f"=== {scenario} ===", flush=True)
        rows = replay(node, run, calib)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            render(rows, scenario, calib, tmp)
            files = sorted(tmp.glob("*.png"))
            listing = tmp / "list.txt"
            # Hold the last frame so the outcome is readable before the loop ends.
            listing.write_text("\n".join([f"file '{f}'" for f in files] + [f"file '{files[-1]}'"] * 20))
            path = out_dir / f"visionpilot_detail_{scenario}_0103.mp4"
            subprocess.run(
                [FFMPEG, "-loglevel", "error", "-y", "-r", "10", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
                check=True,
            )
            print(f"wrote {path}  ({len(files)} frames, {len(files) * STEP_S:.1f}s + hold)")


if __name__ == "__main__":
    main()
