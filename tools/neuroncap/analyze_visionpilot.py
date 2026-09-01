"""Would AutoSpeed have saved the run, and is the IDM port doing what it claims?

`diag_visionpilot.py` showed the three networks see the scene correctly. This
asks the next question: given what they saw, was the trajectory that came out
the right one? Three things are measured, per frame:

*What AutoSpeed knows.*  Every box's bottom edge is intersected with the road
plane, which turns a 2D detection into a range and a lateral offset. The box
matching the target actor is compared against the logged truth. If that range
is good, the boxes the adapter discards were carrying the information the
adapter is missing -- particularly for the side scenario, where AutoDrive by
construction cannot see a vehicle that is not in the ego's path.

*What the closing-speed estimate is worth.*  The node differences AutoDrive's
distance between frames. Frames are 0.1 s apart, so the estimate is the network's
frame-to-frame jitter divided by 0.1 -- amplified tenfold. It is compared here
against the true closing speed from the logs.

*What IDM then commands.*  The node's own acceleration is recomputed, and beside
it two counterfactuals: the same IDM given the true closing speed, and the same
IDM with the horizon's gap update corrected (the shipped one shrinks the gap by
the lead vehicle's speed rather than by the closing speed, so against a parked
car it believes the gap never closes at all).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from tools import paths

matplotlib.use("Agg")
import diag_visionpilot as diag  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

BASE = paths.ncap()
RUNS = BASE / "outputs/runs1/visionpilot"
OUT = BASE / "validation/visionpilot"
SCENARIOS = ("stationary", "frontal", "side")

CIPO_PROB_THRESHOLD = 0.4
CORRIDOR_HALF_WIDTH = 2.5  # m; a lane's worth either side of the ego path
STEP_DT, HORIZON_STEPS = 0.5, 8


class IDM:
    """The node's planner, verbatim, plus a switch for the gap-update fix."""

    def __init__(self, speed_limit: float) -> None:
        self.speed_limit, self.a, self.b = speed_limit, 1.5, 3.0
        self.T, self.s0, self.delta = 1.5, 2.0, 4.0
        self.mu, self.g = 0.2, 9.81

    def acceleration(self, kappa, ego_v, has_cipo, closing, gap):
        curv_v_max = np.sqrt(self.mu * self.g / abs(kappa)) if abs(kappa) > 1e-9 else np.inf
        v0 = min(self.speed_limit, curv_v_max)
        dynamic = (ego_v * closing) / (2.0 * np.sqrt(self.a * self.b))
        s_star = self.s0 + max(0.0, ego_v * self.T + dynamic)
        s = max(0.5, gap)
        free_road = (ego_v / v0) ** self.delta if v0 > 0 else 1.0
        interaction = (s_star / s) ** 2 if has_cipo else 0.0
        return float(self.a * (1.0 - free_road - interaction))

    def profile(self, speed, kappa, has_cipo, closing, gap, fixed_gap_update: bool):
        v, s, travelled, out = speed, gap, 0.0, []
        for _ in range(HORIZON_STEPS):
            accel = self.acceleration(kappa, v, has_cipo, closing, s)
            v_next = max(0.0, v + accel * STEP_DT)
            travelled += 0.5 * (v + v_next) * STEP_DT
            if fixed_gap_update:
                # gap closes at the closing speed, by definition of closing speed
                s = max(0.5, s - closing * STEP_DT)
            else:
                # shipped: ego travel is subtracted and closing*dt added back,
                # which leaves the gap shrinking at the *lead's* speed
                s = max(0.5, s - (0.5 * (v + v_next) * STEP_DT - closing * STEP_DT))
            v = v_next
            out.append(travelled)
        return np.asarray(out)


def ego_speeds(ego_poses: dict, stamps: list[int]) -> np.ndarray:
    """Forward speed from the logged ego poses, m/s."""
    pos = np.array([np.asarray(ego_poses[str(t)])[:2, 3] for t in stamps])
    dt = np.diff(np.asarray(stamps, dtype=np.float64)) / 1e6
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    return np.r_[v[0], v]


def analyse(scenario: str, nets: dict) -> dict:
    run = RUNS / f"{scenario}-0103"
    K = np.asarray(diag.CALIB["CAM_FRONT"]["K"])
    cam2ego = np.asarray(diag.CALIB["CAM_FRONT"]["cam2ego"])
    ego2cam = np.linalg.inv(cam2ego)

    actors = json.loads((run / "actors.json").read_text())
    target = next(a for a in actors if a["is_target_actor"])
    at = np.asarray(target["timestamps"], dtype=np.int64)
    ap_ = np.asarray(target["poses"])
    dims = np.asarray(target["dims"])
    ego_poses = json.loads((run / "ego_poses.json").read_text())

    frames = sorted((run / "CAM_FRONT").iterdir(), key=lambda p: int(p.stem))
    stamps = [int(p.stem) for p in frames]
    speeds = ego_speeds(ego_poses, stamps)

    trace = json.loads((OUT / scenario / "trace.json").read_text())
    idm = IDM(speed_limit=max(float(speeds[0]), 1.0))

    rows = []
    prev_d = prev_t = None
    for i, path in enumerate(frames):
        t = stamps[i]
        image = np.asarray(Image.open(path).convert("RGB"))
        dets = diag.autospeed(nets["speed"], image, 0.6, 0.45)

        # Ground truth target, in the ego frame.
        actor2world = diag.actor_pose_at(at, ap_, t)
        ego2world = np.asarray(ego_poses[str(t)])
        scale3 = np.array([dims[1], dims[0], dims[2]])
        pts = np.c_[diag.CORNERS * scale3, np.ones(8)]
        pts_ego = (np.linalg.inv(ego2world) @ actor2world @ pts.T).T
        gt_xy = pts_ego[:, :2].mean(axis=0)
        gt_range = float(np.linalg.norm(gt_xy))
        pts_cam = (ego2cam @ pts_ego.T).T
        gt_visible = bool((pts_cam[:, 2] > 0.1).all())

        # Every box's bottom edge, put on the road plane.
        matched_range = matched_lat = np.nan
        nearest_in_corridor = np.inf
        if len(dets):
            bottom = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], axis=1)
            ground, ok = diag.project_to_ground(bottom, K, cam2ego)
            for j in range(len(dets)):
                if not ok[j]:
                    continue
                x, y = ground[j]
                if 0 < x < 60 and abs(y) < CORRIDOR_HALF_WIDTH:
                    nearest_in_corridor = min(nearest_in_corridor, float(np.hypot(x, y)))
            if gt_visible:
                uv = (K @ pts_cam[:, :3].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                gt_box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
                ious = np.array([diag.iou_matrix(gt_box, dets[j : j + 1, :4])[0] for j in range(len(dets))])
                if ious.max() > 0.3 and ok[int(ious.argmax())]:
                    x, y = ground[int(ious.argmax())]
                    matched_range, matched_lat = float(np.hypot(x, y)), float(y)

        d, p, kappa = trace[i]["autodrive_d_m"], trace[i]["autodrive_p"], trace[i]["curvature"]

        closing_est = 0.0
        if prev_d is not None:
            dt = (t - prev_t) / 1e6
            if dt > 0:
                closing_est = float(np.clip((prev_d - d) / dt, -30.0, 30.0))
        prev_d, prev_t = d, t

        closing_true = 0.0 if i == 0 else (rows[-1]["gt_range_m"] - gt_range) / ((t - stamps[i - 1]) / 1e6)

        v = float(speeds[i])
        has_cipo = p >= CIPO_PROB_THRESHOLD
        rows.append(
            {
                "t": round((t - stamps[0]) / 1e6, 2),
                "ego_v": round(v, 2),
                "gt_range_m": round(gt_range, 2),
                "gt_visible": gt_visible,
                "autodrive_d_m": d,
                "autodrive_p": p,
                "autospeed_range_m": round(matched_range, 2) if np.isfinite(matched_range) else None,
                "autospeed_lat_m": round(matched_lat, 2) if np.isfinite(matched_lat) else None,
                "autospeed_nearest_corridor_m": round(nearest_in_corridor, 2)
                if np.isfinite(nearest_in_corridor)
                else None,
                "closing_est": round(closing_est, 2),
                "closing_true": round(closing_true, 2),
                "accel_shipped": round(idm.acceleration(kappa, v, has_cipo, closing_est, d), 2),
                "accel_true_closing": round(idm.acceleration(kappa, v, has_cipo, closing_true, d), 2),
                "accel_gt_gap": round(idm.acceleration(kappa, v, True, closing_true, gt_range), 2),
                "reach_shipped": round(float(idm.profile(v, kappa, has_cipo, closing_est, d, False)[-1]), 1),
                "reach_fixed": round(float(idm.profile(v, kappa, has_cipo, closing_est, d, True)[-1]), 1),
            }
        )
        print(f"  {scenario} {rows[-1]}", flush=True)
    return {"scenario": scenario, "rows": rows}


def plot(results: list[dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(19, 11))
    for col, res in enumerate(results):
        rows = res["rows"]
        t = [r["t"] for r in rows]
        gt = [r["gt_range_m"] for r in rows]

        ax = axes[0, col]
        ax.plot(t, gt, "k-", lw=2.5, label="ground truth")
        ax.plot(t, [r["autodrive_d_m"] for r in rows], "-", color="tab:red", label="AutoDrive (used)")
        asr = [r["autospeed_range_m"] for r in rows]
        ax.plot(
            t,
            [np.nan if v is None else v for v in asr],
            "o",
            ms=5,
            color="tab:blue",
            label="AutoSpeed box (discarded)",
        )
        nc = [r["autospeed_nearest_corridor_m"] for r in rows]
        ax.plot(
            t,
            [np.nan if v is None else v for v in nc],
            "--",
            color="tab:green",
            lw=1.2,
            label="nearest AutoSpeed box in corridor",
        )
        ax.set_ylim(0, 70)
        ax.set_title(f"{res['scenario']}  —  range to target")
        ax.set_ylabel("distance (m)")
        ax.legend(fontsize=7.5)

        ax = axes[1, col]
        ax.plot(t, [r["closing_true"] for r in rows], "k-", lw=2.5, label="true closing speed")
        ax.plot(
            t,
            [r["closing_est"] for r in rows],
            "-",
            color="tab:red",
            lw=1.2,
            label="differenced AutoDrive (used)",
        )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylabel("closing speed (m/s)")
        ax.set_title("closing speed estimate")
        ax.legend(fontsize=7.5)

        ax = axes[2, col]
        ax.plot(t, [r["accel_shipped"] for r in rows], "-", color="tab:red", label="IDM as shipped")
        ax.plot(
            t,
            [r["accel_true_closing"] for r in rows],
            "-",
            color="tab:orange",
            label="IDM, true closing speed",
        )
        ax.plot(t, [r["accel_gt_gap"] for r in rows], "-", color="tab:green", label="IDM, true gap + closing")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylabel("commanded accel (m/s²)")
        ax.set_xlabel("time (s)")
        ax.set_title("longitudinal command")
        ax.legend(fontsize=7.5)

    fig.suptitle("Vision Pilot on scene-0103: what the nets saw vs what the planner did", fontsize=14)
    fig.tight_layout()
    path = OUT / "planner_analysis.png"
    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


def main() -> None:
    providers = ["CPUExecutionProvider"]
    nets = {"speed": ort.InferenceSession(str(diag.WEIGHTS / "autospeed_fp32.onnx"), providers=providers)}
    results = [analyse(s, nets) for s in SCENARIOS]
    (OUT / "planner_analysis.json").write_text(json.dumps(results, indent=1))
    plot(results)

    print("\n=== summary ===")
    for res in results:
        rows = res["rows"]
        ad = np.array([abs(r["autodrive_d_m"] - r["gt_range_m"]) for r in rows])
        m = [(r["autospeed_range_m"], r["gt_range_m"]) for r in rows if r["autospeed_range_m"] is not None]
        as_err = np.array([abs(a - g) for a, g in m]) if m else np.array([np.nan])
        ce = np.array([r["closing_est"] for r in rows])
        ct = np.array([r["closing_true"] for r in rows])
        print(
            f"{res['scenario']:>11}: AutoDrive range MAE {ad.mean():5.1f} m | "
            f"AutoSpeed box range MAE {as_err.mean():5.1f} m over {len(m)}/{len(rows)} frames | "
            f"closing-speed MAE {np.abs(ce - ct).mean():5.2f} m/s (true mean {ct.mean():.2f})"
        )


if __name__ == "__main__":
    main()
