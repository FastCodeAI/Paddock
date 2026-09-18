"""Dump every trajectory candidate Drive-JEPA scores, for a whole KITScenes sample list.

Drive-JEPA does not regress one trajectory. It generates 32 proposals, predicts a
NAVSIM PDM score for each with a learned head, and returns the argmax
(`drive_jepa_model.py:85-91`). `proposals.py` draws that for one sample; this
writes it for every sample in a list, so the selection can be measured -- the
best of the 32 against the one picked -- instead of inspected three plots at a time.

    NAVSIM_DEVKIT_ROOT=<Drive-JEPA>/navsim_v2 NAVSIM_EXP_ROOT=<navsim exp> \\
    python tools/models/drivejepa/candidates.py \\
        --samples <kitscenes_e2e_final.pkl> --tier <root> [--tier <root> ...] \\
        --output <dir>/candidates.jsonl [--accel as_is|zero|smoothed] [--command gt|straight|unknown]

Runs in the Drive-JEPA environment (Python 3.9, torch, a GPU). Resumable: rows
already in --output are skipped.

`--accel` and `--command` change one model input at a time, for the input checks
that come before any claim about the scorer:

- `as_is` is the adapter's acceleration: a central difference of central-difference
  velocities over 0.1 s steps of the 10 Hz roof-pod poses
  (`kitscenes_adapter._body_frame_dynamics`). NAVSIM's comes from the vehicle.
- `smoothed` differences the same poses over +-0.5 s windows instead.
- `unknown` is the one-hot NAVSIM uses when there is no route command.

The model reads only the last ego status (`drive_jepa_model.py:51`); every
history status is changed anyway, so no other reader can see the original.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEVKIT_ROOT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
os.chdir(DEVKIT_ROOT)
sys.path.insert(0, ".")

from scripts.inference.kitscenes_adapter import (  # noqa: E402  (needs the chdir above)
    HISTORY_STRIDE,
    NUM_HISTORY_FRAMES,
    KITScenesSceneAdapter,
)
from scripts.inference.run_kitscenes_inference import DEFAULT_CKPT, build_agent  # noqa: E402

COMMAND_UNKNOWN = np.array([0, 0, 0, 1], dtype=np.int64)
SMOOTH_HALF_WINDOW = 5  # frames at 10 Hz: +-0.5 s

# OpenScene's front-camera intrinsics and lens distortion, read from a navtest
# frame's cam metadata. The adapter warps KITScenes onto these intrinsics but
# hands over a clean pinhole image; `--distort` re-applies the distortion the
# checkpoint saw in training, if that training image was in fact distorted.
TARGET_FOCAL, TARGET_CU, TARGET_CV = 1545.0, 960.0, 560.0
TARGET_W, TARGET_H = 1920, 1080
OPENSCENE_DISTORTION = np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231], dtype=np.float64)


def distortion_maps() -> tuple[np.ndarray, np.ndarray]:
    """Remap that turns the pinhole crop into what OpenScene's distorted lens saw.

    Treat each output pixel as a distorted-image pixel and ask `undistortPoints`
    which ideal pixel it corresponds to; sampling the pinhole image there adds the
    distortion (barrel, since k1 < 0). Intrinsics are constant, so build once.
    """
    import cv2

    k = np.array([[TARGET_FOCAL, 0, TARGET_CU], [0, TARGET_FOCAL, TARGET_CV], [0, 0, 1]], dtype=np.float64)
    grid = np.stack(np.meshgrid(np.arange(TARGET_W), np.arange(TARGET_H)), axis=-1).astype(np.float32)
    ideal = cv2.undistortPoints(grid.reshape(-1, 1, 2), k, OPENSCENE_DISTORTION, P=k).reshape(
        TARGET_H, TARGET_W, 2
    )
    return ideal[..., 0], ideal[..., 1]


def resolve_scene(tiers: list[Path], split: str, scene_id: str) -> Path:
    """First tier holding the scene with its poses and front camera.

    Tiers hold different scene sets, not nested copies (docs/data-setup.md), so
    this is resolved per scene.
    """
    for tier in tiers:
        scene = tier / "data" / split / scene_id
        if (scene / "poses.txt").is_file() and (scene / "camera_ring_front").is_dir():
            return scene
    raise FileNotFoundError(f"{split}/{scene_id} is in no tier: {[str(t) for t in tiers]}")


def smoothed_acceleration(adapter: KITScenesSceneAdapter, index: int) -> np.ndarray:
    """Body-frame acceleration from velocities differenced over +-0.5 s."""
    timestamps, poses = adapter.pose_timestamps, adapter.poses
    last = len(timestamps) - 1

    def span(i: int) -> tuple[int, int]:
        return max(i - SMOOTH_HALF_WINDOW, 0), min(i + SMOOTH_HALF_WINDOW, last)

    def velocity(i: int) -> np.ndarray:
        lo, hi = span(i)
        dt = timestamps[hi] - timestamps[lo]
        return (poses[hi, :2] - poses[lo, :2]) / dt if dt > 0 else np.zeros(2)

    lo, hi = span(index)
    dt = timestamps[hi] - timestamps[lo]
    acceleration = (velocity(hi) - velocity(lo)) / dt if dt > 0 else np.zeros(2)
    cos_y, sin_y = np.cos(-poses[index, 2]), np.sin(-poses[index, 2])
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (rotation @ acceleration).astype(np.float32)


def apply_variant(sample, adapter: KITScenesSceneAdapter, keyframe: int, accel: str, command: str) -> None:
    history = [keyframe - HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1 - i) for i in range(NUM_HISTORY_FRAMES)]
    statuses = sample.agent_input.ego_statuses
    for position, frame_index in enumerate(history):
        changes = {}
        if accel == "zero":
            changes["ego_acceleration"] = np.zeros(2, dtype=np.float32)
        elif accel == "smoothed":
            changes["ego_acceleration"] = smoothed_acceleration(adapter, frame_index)
        if command == "unknown":
            changes["driving_command"] = COMMAND_UNKNOWN
        if changes:
            statuses[position] = dataclasses.replace(statuses[position], **changes)


def apply_geometry(sample, origin_offset: float, distort_maps) -> None:
    """Move the ego origin back and/or re-distort the images, in place.

    `origin_offset` shifts the ego origin backward along the forward axis by that
    many metres. The checkpoint places its footprint `rear_axle_to_center`=1.461 m
    ahead of each waypoint (`encoder.compute_corners`), so the trajectory frame is
    the rear axle; OpenScene's origin is the rear axle (front camera at x=1.67),
    while the adapter leaves it at the roof pod (front camera at x=0.21). An offset
    of ~1.46 m makes the two agree. Only camera extrinsics change here; the caller
    shifts the ground-truth future to match, and the model's output lands in the
    same shifted frame.

    Extrinsics come from the adapter's lru-cached `_camera_extrinsics`, so the
    translation is replaced with a fresh array rather than mutated in place.
    """
    import cv2

    offset = np.array([origin_offset, 0.0, 0.0], dtype=np.float32)
    for cameras in sample.agent_input.cameras:
        for name in ("cam_f0", "cam_l0", "cam_r0", "cam_b0"):
            cam = getattr(cameras, name)
            if cam.image is None:
                continue
            if origin_offset:
                cam.sensor2lidar_translation = (
                    np.asarray(cam.sensor2lidar_translation, dtype=np.float32) + offset
                )
            if distort_maps is not None:
                cam.image = cv2.remap(cam.image, distort_maps[0], distort_maps[1], cv2.INTER_LINEAR)


def rear_axle_future(future_poses: np.ndarray, origin_offset: float) -> np.ndarray:
    """Ground truth for the model's output when the origin is shifted back by L.

    The adapter records the future of the roof pod (x, y, heading). With the origin
    at the pod (L=0) that is what the model plans against. With the origin shifted
    back by L, the model plans the rear-axle path, which is the pod path pulled L
    behind the car's own heading at each step -- not a constant shift, because in a
    turn the heading rotates. Re-expressed in the new (rear-axle-at-keyframe) frame:

        x' = x - L*cos(heading) + L
        y' = y - L*sin(heading)

    At L=0 this is the identity, so the offset-0 run reproduces the baseline.
    """
    poses = np.asarray(future_poses, dtype=np.float64).copy()
    heading = poses[:, 2]
    poses[:, 0] = poses[:, 0] - origin_offset * np.cos(heading) + origin_offset
    poses[:, 1] = poses[:, 1] - origin_offset * np.sin(heading)
    return poses


def rounded(array: np.ndarray, digits: int = 4) -> list:
    return np.round(array.astype(np.float64), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--samples", type=Path, required=True, help="pickle of dicts with scene_id, split, keyframe_index"
    )
    parser.add_argument(
        "--tier", type=Path, action="append", required=True, help="KITScenes root; repeat, fastest first"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accel", choices=["as_is", "zero", "smoothed"], default="as_is")
    parser.add_argument("--command", choices=["gt", "straight", "unknown"], default="gt")
    parser.add_argument(
        "--origin-offset",
        type=float,
        default=0.0,
        help="shift the ego origin back toward the rear axle by this many metres (0 = adapter's pod frame)",
    )
    parser.add_argument(
        "--distort", action="store_true", help="re-apply OpenScene lens distortion to each crop"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    distort_maps = distortion_maps() if args.distort else None

    with open(args.samples, "rb") as handle:
        samples = [(s["scene_id"], s["split"], int(s["keyframe_index"])) for s in pickle.load(handle)]
    if args.limit:
        samples = samples[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        with open(args.output) as handle:
            for line in handle:
                row = json.loads(line)
                done.add((row["scene_id"], row["keyframe_index"]))
    todo = [s for s in samples if (s[0], s[2]) not in done]
    print(
        f"{len(samples)} samples, {len(done)} already written, {len(todo)} to run | accel={args.accel} command={args.command}"
    )
    if not todo:
        return

    agent = build_agent(args.checkpoint, "drive_jepa_perception_based_agent")
    num_future = agent._trajectory_sampling.num_poses
    adapters: dict[Path, KITScenesSceneAdapter] = {}
    started = time.perf_counter()

    with open(args.output, "a") as out:
        for n, (scene_id, split, keyframe) in enumerate(todo, 1):
            scene_path = resolve_scene(args.tier, split, scene_id)
            if scene_path not in adapters:
                adapters.clear()  # one scene's calibration at a time; it is per-session state
                adapters[scene_path] = KITScenesSceneAdapter(scene_path)
            adapter = adapters[scene_path]
            # `unknown` is not an adapter mode: build with `gt`, then overwrite the one-hot.
            sample = adapter.build_sample(
                keyframe,
                num_future_frames=num_future,
                command_mode="gt" if args.command == "unknown" else args.command,
            )
            apply_variant(sample, adapter, keyframe, args.accel, args.command)
            apply_geometry(sample, args.origin_offset, distort_maps)
            human_poses = rear_axle_future(sample.future_poses, args.origin_offset)

            features: dict = {}
            for builder in agent.get_feature_builders():
                features.update(builder.compute_features(sample.agent_input))
            features = {k: v.unsqueeze(0).cuda() for k, v in features.items()}
            features.update(past_ego_simulated_states=None, metric_cache=None, observation_interval=None)
            with torch.no_grad():
                output = agent.forward(features)

            proposals = output["proposals"][0].float().cpu().numpy()
            logits = output["pred_logit"][0].float().cpu().numpy()
            scores = output["pdm_score"][0].float().cpu().numpy()
            last = sample.agent_input.ego_statuses[-1]
            row = {
                "scene_id": scene_id,
                "split": split,
                "keyframe_index": keyframe,
                "tier": str(scene_path.parents[2]),
                "variant": {
                    "accel": args.accel,
                    "command": args.command,
                    "origin_offset": args.origin_offset,
                    "distort": args.distort,
                },
                "ego_velocity": rounded(np.asarray(last.ego_velocity)),
                "ego_acceleration": rounded(np.asarray(last.ego_acceleration)),
                "driving_command": np.asarray(last.driving_command).astype(int).tolist(),
                "interval_s": sample.future_interval_s,
                "human_poses": rounded(human_poses),
                "picked": int(scores.argmax()),
                "trajectory": rounded(output["trajectory"][0].float().cpu().numpy()),
                "scores": rounded(scores, 6),
                "logits": rounded(logits, 5),
                "proposals": rounded(proposals),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.perf_counter() - started)
                print(f"  {n}/{len(todo)}  {rate:.2f} samples/s", flush=True)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
