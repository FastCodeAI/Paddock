"""Run Drive-JEPA inference over navsim scenes and dump trajectories to disk.

Kept separate from visualization so the (GPU-bound) forward passes run once and
the (cheap) plotting can be re-run and restyled freely.

    python scripts/inference/run_inference.py --limit 8

Output is a pickle holding, per token, the predicted and human trajectories in
the ego frame, plus the trajectory sampling needed to rebuild them.
"""

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

# The agent's __init__ does np.load("./data/8192.npy"), a path relative to the
# devkit root, so anchor the process there no matter where it was invoked from.
DEVKIT_ROOT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
os.chdir(DEVKIT_ROOT)

CONFIG_ROOT = DEVKIT_ROOT / "navsim/planning/script/config/common"
DEFAULT_CKPT = Path(os.environ["NAVSIM_EXP_ROOT"]) / "Drive-JEPA-cache/drive_jepa_perception_based_agent_vitl_v2.ckpt"


def move_all_modules_to_gpu(instance: nn.Module) -> None:
    """Mirrors run_pdm_score_sequential.py so inference matches the eval exactly."""
    for attr_name in dir(instance):
        attr = getattr(instance, attr_name)
        if isinstance(attr, nn.Module):
            setattr(instance, attr_name, attr.cuda())


def build_agent(checkpoint_path: Path, agent_name: str):
    """Instantiate the agent from the same yaml the eval script uses."""
    cfg = OmegaConf.load(CONFIG_ROOT / f"agent/{agent_name}.yaml")
    # the two overrides eval_drive_jepa_perception_based.sh passes on the CLI
    cfg.checkpoint_path = str(checkpoint_path)
    cfg.config.latent = False

    agent = instantiate(cfg)
    agent.initialize()
    agent.eval()
    move_all_modules_to_gpu(agent)
    return agent


def build_scene_loader(agent, split: str) -> SceneLoader:
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    scene_filter: SceneFilter = instantiate(OmegaConf.load(CONFIG_ROOT / f"train_test_split/scene_filter/{split}.yaml"))
    return SceneLoader(
        data_path=data_root / "navsim_logs/test",
        sensor_blobs_path=data_root / "sensor_blobs/test",
        synthetic_scenes_path=data_root / "synthetic_scenes",
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--agent", default="drive_jepa_perception_based_agent")
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--limit", type=int, default=8, help="number of scenes (0 = all available)")
    parser.add_argument("--tokens", nargs="*", default=None, help="specific tokens instead of the first --limit")
    parser.add_argument("--output", type=Path, default=Path(os.environ["NAVSIM_EXP_ROOT"]) / "inference/results.pkl")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "compute_trajectory() moves features to CUDA; a GPU is required"
    print(f"device     : {torch.cuda.get_device_name(0)} (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
    print(f"checkpoint : {args.checkpoint}")

    agent = build_agent(args.checkpoint, args.agent)
    scene_loader = build_scene_loader(agent, args.split)

    available = scene_loader.tokens
    print(f"scenes available: {len(available)}")
    if args.tokens:
        missing = [t for t in args.tokens if t not in available]
        if missing:
            raise SystemExit(f"tokens not present in the loaded split: {missing}")
        tokens = list(args.tokens)
    else:
        tokens = available if args.limit == 0 else available[: args.limit]
    print(f"running inference on {len(tokens)} scenes\n")

    results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for i, token in enumerate(tokens, 1):
        scene = scene_loader.get_scene_from_token(token)
        t0 = time.perf_counter()
        predicted = agent.compute_trajectory(scene.get_agent_input())
        elapsed = time.perf_counter() - t0
        human = scene.get_future_trajectory()

        results.append(
            {
                "token": token,
                "log_name": scene.scene_metadata.log_name,
                "predicted_poses": predicted.poses,  # (num_poses, 3) -> x, y, heading in ego frame
                "human_poses": human.poses,
                # the two differ: the agent predicts a 4s horizon while the scene's
                # future covers 5s, so each trajectory carries its own sampling
                "predicted_sampling": {
                    "num_poses": predicted.trajectory_sampling.num_poses,
                    "interval_length": predicted.trajectory_sampling.interval_length,
                },
                "human_sampling": {
                    "num_poses": human.trajectory_sampling.num_poses,
                    "interval_length": human.trajectory_sampling.interval_length,
                },
                "inference_seconds": elapsed,
            }
        )
        print(f"  [{i}/{len(tokens)}] {token}  {elapsed:.2f}s")

    total = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)

    per_scene = sum(r["inference_seconds"] for r in results) / max(len(results), 1)
    print(f"\ndone in {total:.1f}s ({per_scene:.2f}s/scene forward pass)")
    print(f"wrote {len(results)} results -> {args.output}")


if __name__ == "__main__":
    main()
