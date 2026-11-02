"""WA-JEPA (AFARI Research) as a paddock Policy.

WA-JEPA is a V-JEPA 2.1 ViT-L/16 encoder with a joint flow predictor that
denoises future scene tokens and the ego trajectory together. It is a NAVSIM
agent: four cameras, four history frames at 2 Hz, eight (x, y, heading)
waypoints at 0.5 s in the ego frame, x forward and y left -- the grid paddock
already uses, so nothing is resampled.

The adapter follows upstream's own closed-loop client,
``close_loop/hugsim_planner.py``, rather than its NAVSIM agent: that client
already turns a simulator's rendered nuScenes-style cameras into model input
without NAVSIM's dataclasses, and every conversion in it was measured against a
simulator by the authors. Where paddock's wire differs from HUGSIM's, the
difference is named below.

*Cameras.* The model tells cameras apart by a learned embedding indexed by list
position in ``model.camera_names`` ([cam_l0, cam_f0, cam_r0, cam_b0]), so the
order of `CAMERA_MAP` is load-bearing: a swapped pair raises nothing and plans
from the wrong view. The mapping onto nuScenes names is upstream's HUGSIM one.
Frames are resized straight to 512x256 with ``INTER_AREA``, no crop -- what the
training loader does to NAVSIM's 1920x1080, and nuScenes' 1600x900 has the same
16:9 aspect and nearly the same front field of view (64.5 vs 63.7 degrees).

*Rate.* NeuroNCAP steps at 0.1 s; the model reads history 0.5 s apart. History
is selected by timestamp, as in the drivejepa adapter, so four frames span 1.5 s
whatever the step. Feeding the last four requests would show the model a world
moving at a fifth of its speed. Until 1.5 s have passed the oldest frame is
repeated, which is upstream's warm-up too; `diagnostics` counts those calls.

*Ego status.* ``[command one-hot (4), vx, vy, ax, ay]`` with NAVSIM's command
order (left, straight, right, unknown). paddock's command is 0=RIGHT 1=LEFT
2=STRAIGHT 3=FOLLOW_REFERENCE -- HUGSIM's order for the first three, so
`COMMAND_TO_NAVSIM` is upstream's ``command_map`` [2, 0, 1] plus "unknown" for
FOLLOW_REFERENCE. vy and ay are zero, as upstream's client sends them.

*Precision.* fp32 weights under bf16 autocast, which is how the model trained.
Casting the weights to bf16 is a different model (upstream's note).

*Determinism.* The flow sampler draws its noise from a generator re-seeded with
``flow_inference_seed`` on every call, so identical input gives an identical
plan. The released 91.7 EPDMS is a ten-seed mean; one seed is one draw from it.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from paddock.policy.protocol import Command, Observation, Policy, Trajectory

#: Model camera slot -> paddock (nuScenes) camera, in the model's slot order.
CAMERA_MAP = {
    "cam_l0": "CAM_FRONT_LEFT",
    "cam_f0": "CAM_FRONT",
    "cam_r0": "CAM_FRONT_RIGHT",
    "cam_b0": "CAM_BACK",
}

#: paddock Command -> NAVSIM driving-command index (0 left, 1 straight, 2 right, 3 unknown).
COMMAND_TO_NAVSIM = {
    Command.RIGHT: 2,
    Command.LEFT: 0,
    Command.STRAIGHT: 1,
    Command.FOLLOW_REFERENCE: 3,
}

HISTORY_FRAMES = 4
HISTORY_DT = 0.5  # seconds between history frames and between waypoints
INPUT_HW = (256, 512)
EGO_STATUS_DIM = 8


def pose_se2(ego2world: np.ndarray) -> np.ndarray:
    """(x, y, yaw) of a 4x4 ego->world pose."""
    return np.array(
        [ego2world[0, 3], ego2world[1, 3], np.arctan2(ego2world[1, 0], ego2world[0, 0])],
        dtype=np.float64,
    )


def ego_status(obs: Observation) -> np.ndarray:
    status = np.zeros(EGO_STATUS_DIM, dtype=np.float32)
    status[COMMAND_TO_NAVSIM[Command(int(obs.command))]] = 1.0
    status[4] = obs.speed  # vx; vy (5) stays zero
    status[6] = obs.accel  # ax; ay (7) stays zero
    return status


def image_tensor_input(image: np.ndarray) -> np.ndarray:
    """uint8 RGB HWC -> float32 CHW in [-1, 1] at the model's 512x256.

    The training loader's transform (``datasets/navsim_official_dataset.py``),
    minus its BGR->RGB swap: paddock frames are already RGB.
    """
    import cv2

    resized = cv2.resize(image, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_AREA)
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return np.clip(chw * 2.0 - 1.0, -1.0, 1.0)


class WaJepaPolicy(Policy):
    cameras = tuple(CAMERA_MAP.values())
    waypoint_dt = HISTORY_DT

    def __init__(
        self, checkpoint: str | None = None, config: str | None = None, device: str = "cuda"
    ) -> None:
        self.checkpoint = checkpoint or os.environ.get("WAJEPA_CHECKPOINT")
        self.config = config or os.environ.get("WAJEPA_CONFIG")
        self.device = device
        self.model = None
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.warmup_calls = 0

    def load(self) -> None:
        if not self.checkpoint or not self.config:
            raise RuntimeError(
                "set WAJEPA_CHECKPOINT and WAJEPA_CONFIG (model.yaml supplies both) "
                "or pass them to WaJepaPolicy"
            )
        # torch and the WA-JEPA package exist only in the model's own env; this
        # module is imported by `paddock list` and the test suite.
        import torch
        from omegaconf import OmegaConf

        repo = os.path.dirname(os.path.dirname(os.path.abspath(self.config)))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from models.factory import build_world_model
        from training.checkpoint import load_checkpoint

        cfg = OmegaConf.load(self.config)
        OmegaConf.set_struct(cfg, False)
        # The full trained state dict is restored below, so the V-JEPA 2.1
        # pretrained encoder is never read (upstream's HUGSIM preset does the same).
        cfg.model.require_pretrained = False
        cfg.model.vjepa2_ckpt = None
        cfg.model.init_from_checkpoint = None
        cfg.model.require_init_from_checkpoint = False
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        model_cfg = cfg_dict["model"]
        expected = list(CAMERA_MAP)
        if [str(c).lower() for c in model_cfg["camera_names"]] != expected:
            raise RuntimeError(f"config camera_names {model_cfg['camera_names']} != adapter slots {expected}")
        for key, want in (
            ("num_history_frames", HISTORY_FRAMES),
            ("ego_status_dim", EGO_STATUS_DIM),
            ("input_hw", list(INPUT_HW)),
        ):
            if model_cfg[key] != want:
                raise RuntimeError(
                    f"config model.{key}={model_cfg[key]!r}; the adapter is built for {want!r}"
                )

        self.torch = torch
        self.cfg = cfg_dict
        model = build_world_model(cfg_dict).to(self.device)
        # Strict by default: a partly restored trajectory head still emits
        # plausible-looking waypoints.
        load_checkpoint(self.checkpoint, model, map_location=self.device)
        self.model = model.eval()
        self.reset()

    def reset(self) -> None:
        self.history = []
        self.calls = 0
        self.warmup_calls = 0

    def _record(self, obs: Observation) -> None:
        # Pixels are converted once, on arrival: a frame is picked for up to
        # four consecutive plans' history at 2 Hz, and fifteen more times at 10 Hz.
        self.history.append(
            {
                "timestamp": obs.timestamp_us,
                "pose": pose_se2(obs.ego2world),
                "images": np.stack([image_tensor_input(obs.image(c)) for c in self.cameras]),
            }
        )
        # 1.5 s at 10 Hz is 16 frames; keep a margin, not the whole episode.
        if len(self.history) > 32:
            self.history = self.history[-32:]

    def _select_history(self) -> list[dict[str, Any]]:
        """Four frames at ~0.5 s spacing, oldest first, clamped at the run start."""
        now = self.history[-1]["timestamp"]
        chosen = []
        for step in range(HISTORY_FRAMES - 1, -1, -1):
            target = now - int(step * HISTORY_DT * 1e6)
            chosen.append(min(self.history, key=lambda f: abs(f["timestamp"] - target)))
        return chosen

    def features(self, obs: Observation) -> dict[str, np.ndarray]:
        """Record *obs* and build the model's input arrays (batch dim excluded)."""
        from close_loop.hugsim_planner import convert_absolute_to_relative_se2_array

        self._record(obs)
        frames = self._select_history()
        poses = np.stack([f["pose"] for f in frames])
        return {
            "history_images": np.stack([f["images"] for f in frames]),  # [T, V, 3, H, W]
            "ego_status": ego_status(obs),
            "history_trajectory": convert_absolute_to_relative_se2_array(poses[-1], poses).astype(np.float32),
        }

    def observe(self, obs: Observation) -> None:
        """Record only: the plan is a pure function of the recorded history."""
        self._record(obs)

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        torch = self.torch
        arrays = self.features(obs)
        span_us = self.history[-1]["timestamp"] - self.history[0]["timestamp"]
        if span_us < (HISTORY_FRAMES - 1) * HISTORY_DT * 1e6:
            self.warmup_calls += 1
        self.calls += 1
        batch = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in arrays.items()}
        autocast = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.no_grad(), torch.autocast(autocast, dtype=torch.bfloat16):
            pred = self.model.predict_trajectory(batch)
        # [8, 3] (x, y, heading), ego frame, x forward, y left, t = 0.5 .. 4.0 s.
        # All eight are returned: closed loop takes six, open loop scores 4 s.
        return Trajectory(pred[0].float().cpu().numpy().astype(np.float64), dt=HISTORY_DT)

    def diagnostics(self) -> dict[str, Any]:
        """History depth achieved and how many plans ran on a padded history."""
        if not self.history:
            return {"status": "no frame seen yet"}
        chosen = self._select_history()
        return {
            "frames_recorded": len(self.history),
            "history_span_s": round((self.history[-1]["timestamp"] - chosen[0]["timestamp"]) / 1e6, 2),
            "history_target_s": (HISTORY_FRAMES - 1) * HISTORY_DT,
            "plans": self.calls,
            "plans_on_padded_history": self.warmup_calls,
        }


POLICY = WaJepaPolicy
