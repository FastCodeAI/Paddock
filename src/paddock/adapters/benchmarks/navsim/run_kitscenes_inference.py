"""Run Drive-JEPA zero-shot on KITScenes Multimodal scenes and score ADE/FDE.

    python scripts/inference/run_kitscenes_inference.py \
        --kitscenes-root /path/to/kitscenes_data --split val --limit 1

WHAT THIS DOES AND DOES NOT MEASURE
-----------------------------------
The checkpoint is trained on nuPlan/OpenScene (US cities, windshield-mounted
camera rig).  KITScenes is European urban driving recorded from a roof-mounted
sensor pod.  ``kitscenes_adapter`` reconciles the frames, the field of view and
the ego dynamics exactly, but it cannot undo the rig and domain differences, so
these numbers are a **zero-shot transfer sanity check**, not a benchmark result
comparable to the paper's KITScenes e2e baselines (which are trained on it).

PROTOCOL
--------
The paper's Table 6 reports on 200 nine-second samples drawn from
``val`` + ``overlap_train_val`` -- each "a non-overlapping 9 s window of 4 s past
observation and up to 5 s of future trajectory anchored at a keyframe" -- with
every metric evaluated at the 3 s horizon.  ``--protocol e2e`` (the default)
reproduces that sampling: keyframes start at 4 s and repeat every 9 s, requiring
a full 5 s of future, and ADE/FDE are scored over 0..3 s.

Two gaps remain against the published numbers.  The list identifying the exact
200 samples is not shipped with the toolkit and the paper does not say it will
be released, so this enumerates the same population rather than the same rows.
And Drive-JEPA ingests a 1.5 s history (four frames at 2 Hz, fixed by NAVSIM),
not the 4 s the benchmark offers -- a model constraint, not an adapter choice.

The map-grounded metrics (drivable-surface survival, collision-free rate,
centerline distance) need the Lanelet2 map plus a lidar occupancy layer and are
not implemented here; this reports displacement error only.  NAVSIM's PDM score
is out of reach entirely -- it needs nuPlan maps and the NAVSIM simulator.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

DEVKIT_ROOT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
os.chdir(DEVKIT_ROOT)  # the agent's __init__ does np.load("./data/8192.npy")

from scripts.inference.kitscenes_adapter import (  # noqa: E402 - needs the chdir above
    KITScenesSceneAdapter,
)

CONFIG_ROOT = DEVKIT_ROOT / "navsim/planning/script/config/common"
DEFAULT_CKPT = (
    Path(os.environ["NAVSIM_EXP_ROOT"]) / "Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"
)


def split_scene_ids(kitscenes_repo: Path, split: str) -> list[str]:
    """Scene UUIDs for *split*, read from the toolkit's split files."""
    filenames = {
        "train": "train.txt",
        "val": "validation.txt",
        "test": "test.txt",
        "test-e2e": "test-e2e.txt",
        "overlap_train_val": "overlap_train_val.txt",
    }
    split_file = kitscenes_repo / "kitscenes/split/generated_splits/default_geo_split_v1_0" / filenames[split]
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def fetch_scene(kitscenes_root: Path, kitscenes_repo: Path, scene_id: str) -> bool:
    """Download and extract one scene tar.  Returns False if the fetch failed.

    Streaming one scene at a time keeps peak disk near a single scene (~5 GB)
    instead of the 594 GB the full val + overlap_train_val pool would need.
    """
    result = subprocess.run(
        ["python", "-m", "kitscenes.download", str(kitscenes_root), "--scenes", scene_id],
        cwd=kitscenes_repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    ! download failed: {result.stderr.strip().splitlines()[-1:] or result.stdout[-200:]}")
    return result.returncode == 0


def move_all_modules_to_gpu(instance: nn.Module) -> None:
    """Mirrors run_inference.py so the forward pass matches the eval exactly."""
    for attr_name in dir(instance):
        attr = getattr(instance, attr_name)
        if isinstance(attr, nn.Module):
            setattr(instance, attr_name, attr.cuda())


def build_agent(checkpoint_path: Path, agent_name: str):
    cfg = OmegaConf.load(CONFIG_ROOT / f"agent/{agent_name}.yaml")
    cfg.checkpoint_path = str(checkpoint_path)
    cfg.config.latent = False
    agent = instantiate(cfg)
    agent.initialize()
    agent.eval()
    move_all_modules_to_gpu(agent)
    return agent


def displacement_errors(
    predicted: np.ndarray, ground_truth: np.ndarray, interval_s: float, eval_horizon_s: float
) -> dict[str, float]:
    """ADE up to *eval_horizon_s* plus FDE at each whole second within it.

    The paper evaluates every Table 6 metric at the 3 s horizon, so ADE must be
    averaged over 0..3 s -- averaging over the model's full 4 s output would not
    be comparable.
    """
    available = min(len(predicted), len(ground_truth))
    horizon = min(available, int(round(eval_horizon_s / interval_s)))
    error = np.linalg.norm(predicted[:horizon, :2] - ground_truth[:horizon, :2], axis=-1)
    metrics: dict[str, float] = {"ade": float(error.mean())}
    for seconds in range(1, int(eval_horizon_s) + 1):
        index = int(round(seconds / interval_s)) - 1
        if 0 <= index < horizon:
            metrics[f"fde@{seconds}s"] = float(error[index])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kitscenes-root", type=Path, default=Path(os.environ.get("KITSCENES_ROOT", "")))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "overlap_train_val"],
        help="the paper draws its 200 e2e samples from val + overlap_train_val",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="scene UUIDs (default: the whole split)")
    parser.add_argument("--limit", type=int, default=0, help="max scenes (0 = all)")
    parser.add_argument(
        "--stream-download",
        action="store_true",
        help="fetch each scene, score it, delete it -- keeps peak disk at ~5 GB "
        "instead of the 594 GB val + overlap_train_val would occupy at once",
    )
    parser.add_argument(
        "--kitscenes-repo",
        type=Path,
        default=Path(os.environ["KITSCENES_REPO"]) if os.environ.get("KITSCENES_REPO") else None,
        help="clone of the kitscenes toolkit, for split lists and the downloader",
    )
    parser.add_argument(
        "--protocol",
        choices=["e2e", "sweep"],
        default="e2e",
        help="e2e: the paper's non-overlapping 9 s windows (4 s past + 5 s future, "
        "scored at 3 s). sweep: dense keyframes at --keyframe-stride, for debugging.",
    )
    parser.add_argument(
        "--keyframe-stride", type=int, default=10, help="sweep protocol only, in 10 Hz frames"
    )
    parser.add_argument("--eval-horizon", type=float, default=3.0, help="seconds scored, paper uses 3")
    parser.add_argument("--max-samples-per-scene", type=int, default=0, help="0 = all keyframes")
    parser.add_argument("--command", choices=["gt", "straight"], default="gt")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--agent", default="drive_jepa_perception_based_agent")
    parser.add_argument(
        "--output", type=Path, default=Path(os.environ["NAVSIM_EXP_ROOT"]) / "inference/kitscenes_results.pkl"
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "compute_trajectory() moves features to CUDA; a GPU is required"

    # (split, scene_id) pairs.  Streaming works off the toolkit's split lists so a
    # run covers the full population; otherwise only what is already extracted.
    targets: list[tuple] = []
    for split in args.splits:
        split_root = args.kitscenes_root / "data" / split
        if args.stream_download:
            scene_ids = split_scene_ids(args.kitscenes_repo, split)
        elif split_root.is_dir():
            scene_ids = sorted(d.name for d in split_root.iterdir() if d.is_dir())
        else:
            print(f"! skipping {split}: {split_root} not present and --stream-download not set")
            continue
        targets.extend((split, scene_id) for scene_id in scene_ids)

    if args.scenes:
        targets = [t for t in targets if t[1] in args.scenes]
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("no scenes to evaluate")

    print(f"device     : {torch.cuda.get_device_name(0)}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"scenes     : {len(targets)} across {', '.join(args.splits)}")
    print(f"streaming  : {args.stream_download}\n")

    agent = build_agent(args.checkpoint, args.agent)
    horizon = agent._trajectory_sampling
    num_future_frames = horizon.num_poses
    print(f"agent horizon: {horizon.num_poses} poses x {horizon.interval_length}s = {horizon.time_horizon}s")

    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for scene_number, (split, scene_id) in enumerate(targets, 1):
        scene_path = args.kitscenes_root / "data" / split / scene_id
        fetched_here = False
        if not scene_path.is_dir():
            if not args.stream_download:
                print(f"[{scene_number}/{len(targets)}] {scene_id}  ! not on disk, skipping")
                continue
            print(f"[{scene_number}/{len(targets)}] {scene_id}  downloading ...", flush=True)
            download_started = time.perf_counter()
            if not fetch_scene(args.kitscenes_root, args.kitscenes_repo, scene_id):
                continue
            fetched_here = True
            print(f"    fetched in {time.perf_counter() - download_started:.0f}s")

        adapter = KITScenesSceneAdapter(scene_path)
        if args.protocol == "e2e":
            # 4 s of past observation, 9 s windows that do not overlap, and a full
            # 5 s of future so keyframes land where the benchmark's would.
            keyframes = adapter.valid_keyframes(num_future_frames=10, stride=90, first_frame=40)
        else:
            keyframes = adapter.valid_keyframes(num_future_frames, stride=args.keyframe_stride)
        if args.max_samples_per_scene:
            keyframes = keyframes[: args.max_samples_per_scene]
        print(
            f"[{scene_number}/{len(targets)}] {scene_id} ({split})  "
            f"{adapter.num_frames} frames, pod {adapter.ground_offset:.2f} m above road, "
            f"{len(keyframes)} keyframes"
        )

        for keyframe in keyframes:
            sample = adapter.build_sample(
                keyframe,
                num_future_frames=num_future_frames,
                command_mode=args.command,
            )
            t0 = time.perf_counter()
            predicted = agent.compute_trajectory(sample.agent_input)
            elapsed = time.perf_counter() - t0

            metrics = displacement_errors(
                predicted.poses, sample.future_poses, sample.future_interval_s, args.eval_horizon
            )
            results.append(
                {
                    "scene_id": scene_id,
                    "split": split,
                    "keyframe_index": keyframe,
                    "predicted_poses": predicted.poses,
                    "human_poses": sample.future_poses,
                    "interval_length": sample.future_interval_s,
                    "driving_command": sample.agent_input.ego_statuses[-1].driving_command,
                    "ego_speed": float(np.linalg.norm(sample.agent_input.ego_statuses[-1].ego_velocity)),
                    "inference_seconds": elapsed,
                    **metrics,
                }
            )
            print(
                f"    frame {keyframe:4d} (t={keyframe / 10.0:5.1f}s)  "
                f"ADE {metrics['ade']:5.2f}  FDE@3s {metrics.get('fde@3s', float('nan')):5.2f}  "
                f"{elapsed:.2f}s"
            )

        # Checkpoint after every scene: a full streaming sweep runs for hours and
        # should survive an interruption without losing what it already scored.
        with open(args.output, "wb") as handle:
            pickle.dump(results, handle)

        if fetched_here and args.stream_download:
            shutil.rmtree(scene_path, ignore_errors=True)

    with open(args.output, "wb") as handle:
        pickle.dump(results, handle)

    print(f"\n{len(results)} samples in {time.perf_counter() - started:.1f}s")
    if results:
        print("\nzero-shot transfer (nuPlan-trained checkpoint -> KITScenes):")
        for key in ("ade", "fde@1s", "fde@2s", "fde@3s", "fde@4s"):
            values = [r[key] for r in results if key in r]
            if values:
                print(f"  {key:8s} mean {np.mean(values):6.2f} m   median {np.median(values):6.2f} m")
        print(f"  mean ego speed {np.mean([r['ego_speed'] for r in results]):.1f} m/s")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
