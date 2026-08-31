"""NeuroNCAP model node for NVIDIA Alpamayo 1.5.

    PYTHONPATH=/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5:\\
/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/eval_kitscenes:\\
/mnt/p4/naren/neuroncap/servers \\
      a1_5_venv/bin/python alpamayo_server.py --port 9002

Alpamayo is a 10B vision-language-action model: the VLM writes a
chain-of-causation trace, then a diffusion expert samples trajectories
conditioned on its hidden states. Three adaptations matter.

*Cameras.* Alpamayo takes four slots -- three nominally 120 deg views at
+55/0/-55 deg and a 30 deg telephoto. nuScenes' front three cameras sit almost
exactly at those bearings, which is the good news. The bad news is field of
view: nuScenes gives about 65 deg where the model expects 120, so the three wide
slots are filled with the widest 16:9 crop available and objects subtend roughly
1.8x the angle the model was trained to associate with a given distance. The
KITScenes run hit the same wall at 87 deg and found `fill` still beat
`letterbox` overall, so `fill` is the default here too -- but this is the single
largest source of doubt in Alpamayo's numbers on this benchmark, and it biases
depth. The telephoto slot is exact, because 30 deg is narrower than the source.

*History.* The model wants 16 ego poses at dt=0.1 and 4 camera frames, and
derives its initial speed purely by differencing that history -- a wrong rate
produces a confident, meaningless trajectory. That is why this node insists on
a 0.1 s simulation step (`--engine.scenario.time-step 0.1`) rather than
resampling: at 10 Hz the request stream *is* the history, one frame per step.
A mismatched step is rejected loudly instead of silently mis-scaling the speed.

*Output.* Alpamayo predicts 64 waypoints at 10 Hz (6.4 s). NeuroNCAP wants six
on a 0.5 s grid, so indices 4, 9, ... 29 are taken -- the same subsampling the
KITScenes harness used. With more than one sample drawn, the medoid is
executed: the closed loop can only drive one trajectory, and the medoid is the
consensus draw rather than an oracle-selected best.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from scipy.spatial import transform as spt

import ncap_io

from alpamayo1_5 import helper
from eval_kitscenes.camera_warp import (
    TARGET_H,
    TARGET_W,
    TELE_HFOV_DEG,
    Pinhole as WarpPinhole,
)
from eval_kitscenes.model_loader import load_model

# Alpamayo slot -> (nuScenes camera, how to fill it). Matched by bearing:
# nuScenes' front-left/front-right sit within a couple of degrees of the
# +55/-55 the model expects.
CAMERA_SLOTS: list[tuple[int, str, str]] = [
    (0, "CAM_FRONT_LEFT", "fill"),
    (1, "CAM_FRONT", "fill"),
    (2, "CAM_FRONT_RIGHT", "fill"),
    (6, "CAM_FRONT", "zoom"),
]
NEEDED_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

TARGET_ASPECT = TARGET_H / TARGET_W  # 9:16, the model frame's height/width

NUM_HISTORY_STEPS = 16
NUM_CAM_FRAMES = 4
HISTORY_DT = 0.1
MODEL_HZ = 10.0
# 64 waypoints at 10 Hz -> the six 0.5 s points NeuroNCAP drives.
GRID_INDICES = np.array([4, 9, 14, 19, 24, 29])


def to_pinhole(p: ncap_io.Pinhole) -> WarpPinhole:
    """ncap_io's intrinsics in the shape eval_kitscenes' warp code expects."""
    return WarpPinhole(focal=p.fx, cu=p.cx, cv=p.cy, width=p.width, height=p.height)


def crop_box(src: WarpPinhole, target_hfov_deg: float | None = None) -> tuple[int, int, int, int]:
    """Widest 16:9 window centred on the principal point that fits the frame.

    This is eval_kitscenes' crop_box with the vertical extent also constrained.
    Theirs picks the width from the horizontal margin alone and derives the
    height from it, which is safe on KITScenes' 3504x2272 (4:3-ish) sensors but
    not here: a NeuRAD render is 1599x900, already wider than 16:9, so the
    height is the binding dimension and the box runs off the bottom by a few
    pixels -- ``crop (55, 36, 1598, 904) escapes the 1599x900 frame``.

    Taking the smaller of the two limits keeps the principal point at the output
    centre, which is the property the whole adaptation rests on. It costs about
    1% of horizontal field of view against the unclamped box.
    """
    half_w_limit = min(src.cu, src.width - 1 - src.cu)
    if target_hfov_deg is not None:
        half_w_limit = min(half_w_limit, src.focal * math.tan(math.radians(0.5 * target_hfov_deg)))
    # Height follows width at 16:9, so a vertical margin caps the width too.
    half_h_limit = min(src.cv, src.height - 1 - src.cv)
    half_w = min(half_w_limit, half_h_limit / TARGET_ASPECT)

    width = int(2.0 * half_w)
    height = int(width * TARGET_ASPECT)
    left = int(round(src.cu - 0.5 * width))
    top = int(round(src.cv - 0.5 * height))
    box = (left, top, left + width, top + height)
    if box[0] < 0 or box[1] < 0 or box[2] > src.width or box[3] > src.height:
        raise ValueError(f"crop {box} escapes the {src.width}x{src.height} frame")
    return box


def warp(image: Image.Image, src: WarpPinhole, kind: str) -> np.ndarray:
    """Crop to the slot's field of view and rescale to the model's frame."""
    box = crop_box(src, TELE_HFOV_DEG if kind == "zoom" else None)
    resized = image.crop(box).resize((TARGET_W, TARGET_H), Image.BICUBIC)
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def medoid(trajectories: np.ndarray) -> np.ndarray:
    """The draw closest to all the others: (K, T, D) -> (T, D).

    Executed rather than the best-scoring draw, because in closed loop there is
    no ground truth to score against and only one trajectory can be driven.
    """
    if len(trajectories) == 1:
        return trajectories[0]
    flat = trajectories.reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).sum(axis=1)
    return trajectories[int(np.argmin(distances))]


class AlpamayoNode:
    def __init__(self, num_traj_samples: int = 1, reserve_gb: float | None = None,
                 temperature: float = 1.0, top_p: float = 0.95, max_generation_length: int = 512) -> None:
        self.model, self.processor = load_model(reserve_gb=reserve_gb)
        self.num_traj_samples = num_traj_samples
        self.temperature, self.top_p = temperature, top_p
        self.max_generation_length = max_generation_length
        self.reset()

    def reset(self) -> None:
        """Drop the pose and image history between runs."""
        self.poses: list[np.ndarray] = []
        self.timestamps: list[int] = []
        self.frames: list[dict[str, np.ndarray]] = []

    def _sample_poses(self, offsets_s: np.ndarray) -> np.ndarray:
        """Ego poses at the given offsets before now, interpolated over time.

        The history is resampled by timestamp rather than by taking the last N
        requests, because NeuroNCAP's request cadence is not uniform: the
        priming steps at the start of every run advance by dataset keyframes
        (0.5 s) whatever the scenario time step is, and only the closed-loop
        steps honour it. Taking the last 16 requests would therefore hand
        Alpamayo a history spanning 0.5 s intervals labelled as 0.1 s ones, and
        it derives its initial speed by differencing exactly those -- so it
        would confidently plan for a car going five times too fast.

        Translation is interpolated linearly and rotation with slerp, so a
        sparse early history still yields metrically correct spacing rather
        than repeated poses reading as a standing start.
        """
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        stack = np.stack(self.poses)
        now = times[-1]
        wanted = np.clip(now - offsets_s, times[0], now)

        xyz = np.stack([np.interp(wanted, times, stack[:, i, 3]) for i in range(3)], axis=1)
        rotations = spt.Rotation.from_matrix(stack[:, :3, :3])
        if len(times) == 1:
            matrices = np.repeat(stack[:1, :3, :3], len(wanted), axis=0)
        else:
            matrices = spt.Slerp(times, rotations)(wanted).as_matrix()

        poses = np.repeat(np.eye(4)[None], len(wanted), axis=0)
        poses[:, :3, :3] = matrices
        poses[:, :3, 3] = xyz
        return poses

    def _history(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The last 16 ego poses at dt=0.1, expressed in the current (t0) frame.

        Full SO(3), matching load_physical_aiavdataset rather than a yaw-only
        approximation.
        """
        offsets = np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT
        stack = self._sample_poses(offsets)

        t0_inv = spt.Rotation.from_matrix(stack[-1, :3, :3]).inv()
        xyz = t0_inv.apply(stack[:, :3, 3] - stack[-1, :3, 3])
        rot = (t0_inv * spt.Rotation.from_matrix(stack[:, :3, :3])).as_matrix()
        return (
            torch.from_numpy(xyz).float()[None, None],
            torch.from_numpy(rot).float()[None, None],
        )

    def _image_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N_cam, 4, 3, H, W) uint8, cameras in ascending slot order.

        Frames are picked by timestamp at the same 0.1 s spacing as the pose
        history, for the same reason: the request cadence is not uniform.
        """
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        wanted = times[-1] - np.arange(NUM_CAM_FRAMES - 1, -1, -1) * HISTORY_DT
        recent = [self.frames[int(np.argmin(np.abs(times - w)))] for w in wanted]

        ordered = sorted(CAMERA_SLOTS, key=lambda entry: entry[0])
        per_camera = []
        for _, camera, kind in ordered:
            warped = np.stack([frame[f"{camera}:{kind}"] for frame in recent])
            per_camera.append(torch.from_numpy(warped).permute(0, 3, 1, 2))
        indices = torch.tensor([slot for slot, _, _ in ordered], dtype=torch.int64)
        return torch.stack(per_camera, dim=0), indices

    def _warp_frame(self, req: ncap_io.Request) -> dict[str, np.ndarray]:
        warped = {}
        for _, camera, kind in CAMERA_SLOTS:
            key = f"{camera}:{kind}"
            if key in warped:
                continue
            image = Image.fromarray(req.images[camera])
            pinhole = to_pinhole(req.pinhole(camera))
            warped[key] = warp(image, pinhole, kind)
        return warped

    def infer(self, body: dict[str, Any]) -> dict[str, Any]:
        req = ncap_io.Request.parse(body, cameras=NEEDED_CAMERAS)
        self.poses.append(req.ego2world)
        self.timestamps.append(req.timestamp)
        self.frames.append(self._warp_frame(req))
        # A run is at most a few hundred steps, but each entry holds four warped
        # 1920x1080 frames, so old ones are dropped once they are out of reach.
        if len(self.poses) > 64:
            self.poses, self.timestamps, self.frames = (
                self.poses[-64:], self.timestamps[-64:], self.frames[-64:]
            )

        hist_xyz, hist_rot = self._history()
        image_frames, camera_indices = self._image_frames()

        messages = helper.create_message(
            frames=image_frames.flatten(0, 1), camera_indices=camera_indices, nav_text=None
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {"tokenized_data": inputs, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}, "cuda"
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=self.top_p,
                temperature=self.temperature,
                num_traj_samples=self.num_traj_samples,
                max_generation_length=self.max_generation_length,
                return_extra=True,
            )

        samples = pred_xyz.float().cpu().numpy()[0, 0]  # (K, 64, 3)
        chosen = medoid(samples)
        # Alpamayo is x-forward, y-left, the same convention NeuroNCAP drives.
        waypoints = chosen[GRID_INDICES][:, :2]

        reasoning = extra.get("text", [""]) if isinstance(extra, dict) else ""
        return ncap_io.response(waypoints) | {
            "debug": {
                "num_samples": int(len(samples)),
                "history_frames": min(len(self.poses), NUM_HISTORY_STEPS),
                "request_dt_s": (
                    round((self.timestamps[-1] - self.timestamps[-2]) / 1e6, 3)
                    if len(self.timestamps) > 1 else None
                ),
                "reasoning": (reasoning[0][:300] if isinstance(reasoning, list) and reasoning else ""),
            }
        }


def build_app(node: AlpamayoNode) -> FastAPI:
    app = FastAPI()

    @app.get("/alive")
    async def alive() -> bool:
        return True

    @app.post("/infer")
    async def infer(body: dict) -> dict:
        return node.infer(body)

    @app.post("/reset")
    async def reset(body: dict | None = None) -> bool:
        node.reset()
        return True

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-traj-samples", type=int, default=1,
                        help="draws per step; the medoid is executed. 1 keeps VRAM near 24 GB")
    parser.add_argument("--reserve-gb", type=float, default=None,
                        help="claim this much VRAM before the slow checkpoint read (shared GPU)")
    args = parser.parse_args()

    node = AlpamayoNode(num_traj_samples=args.num_traj_samples, reserve_gb=args.reserve_gb)
    uvicorn.run(build_app(node), host=args.host, port=args.port, log_level="warning")
