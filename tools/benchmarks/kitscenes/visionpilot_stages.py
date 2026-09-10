"""Dump every stage of VisionPilot's KITScenes trajectory, for one set of samples.

VisionPilot emits no trajectory. The adapter builds one: project AutoSteer's
image-space path onto the road, fit a curve, then walk an IDM speed schedule
along it. When a reported ADE looks wrong, the useful question is *which stage*
produced it, and that needs the intermediates rather than the final number.

Three variants are recorded per sample so the two failure modes that cost a week
are visible side by side:

    vp_fixed       what the merged adapter does: ego origin on the road, ACC
                   set-point latched at the scored frame
    vp_podorigin   origin left on the roof pod, as KITScenes ships it -- every
                   projected point then lands *behind* the car
    vp_runstart    road-level origin, set-point latched at the first history
                   frame, 4.8 s early

plus `rawpath_*`, the fitted curve with no speed schedule at all. Scoring that
is the single best explanation for a set of externally reported numbers that did
not reconcile with ours (correlation 0.94-0.96, against 0.28 for `vp_fixed`).

Runs in the *model's* interpreter, since onnxruntime lives there and paddock's
own environment deliberately has no model dependencies:

    KITSCENES_ROOTS=... VISIONPILOT_WEIGHTS=... \
    PYTHONPATH=<repo>/src:<vision_pilot> \
        <ncap-env>/bin/python tools/benchmarks/kitscenes/visionpilot_stages.py --out traj.json

`paddock run` sets those for a normal run; this script is driven by hand, so it
reads them from the environment and says which is missing rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from paddock.adapters.benchmarks.kitscenes import loader as L
from paddock.adapters.models.visionpilot.policy import (
    NET_H,
    NET_W,
    PLAN_HORIZON_S,
    STEER_MASK_THRESHOLD,
    WAYPOINT_DT,
    VisionPilotPolicy,
)
from paddock.policy.protocol import Command, Observation

#: Roof-pod height above the road. The benchmark's own default; see
#: `adapters/benchmarks/kitscenes/harness.py::POD_HEIGHT_M`.
POD_HEIGHT_M = 2.2

#: Source frames of history, matching the harness.
HISTORY_FRAMES = 48

#: Eight samples spanning the speed range, worst-first. Chosen to match a
#: figure that was circulating with different numbers on it.
DEFAULT_SAMPLES = [
    "f0b1b61b/130",
    "eca7bf3a/220",
    "be5138b4/40",
    "b9c27408/40",
    "a724baf7/130",
    "950dda14/40",
    "85dbe6d6/130",
    "45ac20f8/130",
]


def _sample_list() -> Path:
    """The shared keyframe list, from `[benchmarks.kitscenes] sample_list`."""
    from tools import paths

    settings = paths.config().benchmarks.get("kitscenes") or {}
    value = settings.get("sample_list")
    if not value:
        raise RuntimeError("[benchmarks.kitscenes] sample_list is not set in paddock.toml")
    return Path(value)


def _calibration(scene: Path, ground_offset: float) -> tuple[np.ndarray, np.ndarray]:
    entry = json.loads((scene / "calibration" / "calib.json").read_text())["camera_ring_front"]
    k = entry["intrinsics"]
    focal = float(k["focal_length"])
    intrinsics = np.array(
        [
            [focal, 0.0, float(k["principal_point_u"])],
            [0.0, focal, float(k["principal_point_v"])],
            [0.0, 0.0, 1.0],
        ]
    )
    cam2ego = np.asarray(entry["T_to_reference"], dtype=np.float64).copy()
    cam2ego[2, 3] += ground_offset
    return intrinsics, cam2ego


def _observation(scene: Path, index: int, poses, intrinsics, cam2ego) -> Observation:
    canbus = np.zeros(18)
    canbus[13] = L.ego_speed(poses, index)
    return Observation(
        images={},
        image_paths={"CAM_FRONT": str(L.frame_path(scene, index))},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=int(index / 10 * 1e6),
        command=Command.STRAIGHT,
        camera2image={"CAM_FRONT": intrinsics},
        camera2ego={"CAM_FRONT": cam2ego},
    )


def stages(policy: VisionPilotPolicy, row: dict) -> dict:
    """Every intermediate for one sample, as plain JSON-able lists."""
    scene = L.scene_path(row["scene_id"], row.get("split"))
    keyframe = int(row["keyframe_index"])
    poses = L.load_poses(scene)
    indices = L.frame_indices(keyframe, HISTORY_FRAMES)

    record = {
        "scene": row["scene_id"],
        "keyframe": keyframe,
        "speed": float(L.ego_speed(poses, keyframe)),
        "speed_at_history_start": float(L.ego_speed(poses, indices[0])),
        "human": np.asarray(row["human_poses"])[:, :2].tolist(),
        "drivejepa": np.asarray(row["predicted_poses"])[:, :2].tolist(),
        "drivejepa_ade": float(row["ade"]),
    }

    for label, offset, setpoint in (
        ("fixed", POD_HEIGHT_M, "scored"),
        ("podorigin", 0.0, "scored"),
        ("runstart", POD_HEIGHT_M, "run_start"),
    ):
        # Read inside observe(), so it must be set before the history replay.
        os.environ["VISIONPILOT_SETPOINT"] = setpoint
        intrinsics, cam2ego = _calibration(scene, offset)
        policy.reset()
        for index in indices[:-1]:
            policy.observe(_observation(scene, index, poses, intrinsics, cam2ego))
        last = _observation(scene, keyframe, poses, intrinsics, cam2ego)

        record[f"vp_{label}"] = np.asarray(policy.infer(last).xy).tolist()
        record[f"set_speed_{label}"] = None if policy.set_speed is None else float(policy.set_speed)

        # The geometry alone: AutoSteer's points on the ground and the curve
        # fitted through them, sampled over its own extent. No timing anywhere.
        image = last.image("CAM_FRONT")
        resized = np.asarray(Image.fromarray(image).resize((NET_W, NET_H), Image.BILINEAR))
        xp, confidence = policy._run_steer(resized)
        ground = policy._path_points(xp, confidence, last)
        fit = policy._fit_path(ground)
        record[f"points_{label}"] = np.asarray(ground).tolist()
        record[f"x_max_{label}"] = float(fit.x_max)
        record[f"kept_{label}"] = int((confidence >= STEER_MASK_THRESHOLD).sum())

        count = int(round(PLAN_HORIZON_S / WAYPOINT_DT))
        if fit.coeffs is not None and fit.x_max > 0:
            xs = np.linspace(fit.x_max / count, fit.x_max, count)
            record[f"rawpath_{label}"] = np.stack([xs, fit.lateral_at(xs)], axis=1).tolist()
        else:
            # x_max <= 0 means the fit had no forward evidence at all, which is
            # exactly what the roof-pod origin produces. Recorded as empty
            # rather than silently drawn as a flat line.
            record[f"rawpath_{label}"] = []
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="where to write the JSON")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=DEFAULT_SAMPLES,
        help="scene-prefix/keyframe pairs, e.g. b9c27408/40. Default: the eight in DEFAULT_SAMPLES.",
    )
    args = parser.parse_args()

    rows = pickle.loads(_sample_list().read_bytes())
    by_key = {(r["scene_id"][:8], int(r["keyframe_index"])): r for r in rows}

    policy = VisionPilotPolicy()
    policy.load()

    out = {}
    for name in args.samples:
        prefix, keyframe = name.split("/")
        key = (prefix, int(keyframe))
        if key not in by_key:
            raise SystemExit(f"{name} is not in the shared sample list")
        out[name] = stages(policy, by_key[key])
        print(f"  {name}", flush=True)

    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
