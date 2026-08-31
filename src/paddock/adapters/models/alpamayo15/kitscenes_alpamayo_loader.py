"""Turn a KITScenes scene plus a keyframe into the dict Alpamayo 1.5 consumes.

``load_physical_aiavdataset()`` returns a plain dict and nothing downstream needs the
gated PhysicalAI-AV package, so running Alpamayo on KITScenes means building the same
dict from ``poses.txt`` and a directory of JPEGs.  Only three keys ever reach the model
(see ``test_inference.py``): ``tokenized_data``, ``ego_history_xyz`` and
``ego_history_rot``.  The rest is bookkeeping, kept so the dict stays a drop-in.

Two conventions have to line up exactly with the Drive-JEPA run this is compared
against, and both are asserted rather than assumed:

*Frames.*  KITScenes is natively 10 Hz and ``timestamp.reference.txt`` agrees with
``poses.txt`` to under 3e-5 s, so frame index == pose index and no resampling happens
anywhere.  That is a better fit than Drive-JEPA got -- NAVSIM forced it to 2 Hz.

*Ego frame.*  Poses are TUM ``x y z qx qy qz qw`` in ``base_frame``, which ``calib.json``
shows is ``lidar_top`` (the roof pod).  The transform into the t0 frame mirrors
``load_physical_aiavdataset.py`` exactly, using full SO(3) rather than yaw only.  The
scoring GT keeps Drive-JEPA's yaw-only convention; the two differ by under 2 cm over
4 s (the pod's roll and pitch are small), and ``metrics.py`` reconciles them explicitly.

The scale of the history is the one input that can be wrong silently and fatally:
Alpamayo derives its initial speed purely from finite differences of the 16 history
waypoints at dt=0.1, so a wrong rate produces a plausible-looking but meaningless
trajectory.  ``build_sample`` checks it against an independently computed speed.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange
from eval_kitscenes.camera_warp import Pinhole, load_and_warp
from scipy.spatial import transform as spt

# Alpamayo slot id -> (KITScenes sensor, warp mode for the fill/letterbox family).
# Matched by yaw: the KITScenes ring sits at +60.9 / +0.97 / -58.5 deg, within 6 deg of
# Alpamayo's canonical +55 / 0 / -55.  Slot 6 is a digital telephoto off the front camera.
CAMERA_SLOTS: list[tuple[int, str, str]] = [
    (0, "camera_ring_front_left", "wide"),
    (1, "camera_ring_front", "wide"),
    (2, "camera_ring_front_right", "wide"),
    (6, "camera_ring_front", "zoom"),
]

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
NUM_CAM_FRAMES = 4
DT = 0.1

# Drive-JEPA's NAVSIM-imposed sampling, reproduced so the keyframes and the scoring
# grid are identical (``kitscenes_adapter.py``).
HISTORY_STRIDE = 5
NUM_HISTORY_FRAMES = 4
TURN_THRESHOLD_M = 5.0

POSES_FILENAME = "poses.txt"
REFERENCE_TIMESTAMP_FILENAME = "timestamp.reference.txt"
CALIBRATION_FILENAME = "calibration/calib.json"


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of a TUM-ordered quaternion.  Ported from ``kitscenes_adapter``."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def to_relative_poses(
    origin: npt.NDArray[np.float64], poses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Express absolute (x, y, yaw) poses in the frame of *origin*.

    Byte-for-byte the Drive-JEPA function, because the ground truth it produces is what
    both models are scored against.
    """
    cos_o, sin_o = np.cos(origin[2]), np.sin(origin[2])
    delta = poses[:, :2] - origin[:2]
    rotated = np.stack(
        [cos_o * delta[:, 0] + sin_o * delta[:, 1], -sin_o * delta[:, 0] + cos_o * delta[:, 1]],
        axis=-1,
    )
    heading = np.arctan2(np.sin(poses[:, 2] - origin[2]), np.cos(poses[:, 2] - origin[2]))
    return np.concatenate([rotated, heading[:, None]], axis=-1)


class KITScenesAlpamayoScene:
    """One KITScenes scene, sampled into Alpamayo input dicts."""

    def __init__(self, scene_path: Path) -> None:
        self.scene_path = Path(scene_path)
        self.scene_id = self.scene_path.name

        raw = np.loadtxt(self.scene_path / POSES_FILENAME)
        if raw.ndim == 1:
            raw = raw[None, :]
        self.pose_timestamps = raw[:, 0]
        self.xyz = raw[:, 1:4].astype(np.float64)
        self.quat = raw[:, 4:8].astype(np.float64)  # TUM xyzw == scipy xyzw, no reorder
        self.planar = np.stack(
            [raw[:, 1], raw[:, 2], [quaternion_to_yaw(*row) for row in raw[:, 4:8]]], axis=-1
        )

        self.reference_timestamps = np.loadtxt(self.scene_path / REFERENCE_TIMESTAMP_FILENAME)
        self.calibration = json.loads((self.scene_path / CALIBRATION_FILENAME).read_text())
        self.num_frames = len(self.reference_timestamps)

        assert len(self.xyz) == self.num_frames, (
            f"{self.scene_id}: {len(self.xyz)} poses vs {self.num_frames} reference timestamps"
        )

    # -- geometry ----------------------------------------------------------- #

    @cache
    def _pinhole(self, sensor: str) -> Pinhole:
        """Intrinsics for *sensor*, with the frame size read off frame 0."""
        from PIL import Image

        with Image.open(self.scene_path / sensor / f"{0:010d}.jpg") as image:
            width, height = image.size
        return Pinhole.from_calib(self.calibration[sensor], width, height)

    # -- sampling ----------------------------------------------------------- #

    def valid_keyframes(
        self, num_future_frames: int = 10, stride: int = 90, first_frame: int | None = 40
    ) -> list[int]:
        """Keyframes with a full history behind and a full future ahead.

        Reproduces ``KITScenesSceneAdapter.valid_keyframes``.  The defaults are the
        paper's e2e protocol: non-overlapping 9 s windows of 4 s past and 5 s future.
        """
        minimum = HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1)
        first = minimum if first_frame is None else max(first_frame, minimum)
        last = self.num_frames - num_future_frames * HISTORY_STRIDE - 1
        return list(range(first, last + 1, stride)) if last >= first else []

    def planar_future(self, k: int, n: int = 8, stride: int = HISTORY_STRIDE) -> npt.NDArray:
        """Drive-JEPA's ground truth: (n, 3) x/y/yaw at 0.5 s, in the yaw-only t0 frame."""
        future = [k + stride * (i + 1) for i in range(n)]
        return to_relative_poses(self.planar[k], self.planar[future])

    def body_velocity(self, k: int) -> npt.NDArray[np.float64]:
        """Central-difference velocity at *k*, in the body frame.

        Ported from ``_body_frame_dynamics`` so ``ego_speed`` matches the reference run
        exactly -- it is one of the join keys used to prove the samples line up.
        """
        last = len(self.pose_timestamps) - 1
        lo, hi = max(k - 1, 0), min(k + 1, last)
        dt = self.pose_timestamps[hi] - self.pose_timestamps[lo]
        velocity = np.zeros(2) if dt <= 0 else (self.planar[hi, :2] - self.planar[lo, :2]) / dt
        yaw = self.planar[k, 2]
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        return np.array([[cos_y, -sin_y], [sin_y, cos_y]]) @ velocity

    def nav_text(self, k: int, num_future_frames: int = 10) -> str:
        """The route hint Drive-JEPA received, rendered as text for Alpamayo.

        ``run_kitscenes_inference.py`` defaults to ``--command gt``, which reads the
        recorded future at ``k + 50`` and hands the planner a left/straight/right
        one-hot on 39 of the 223 samples.  Withholding the equivalent from Alpamayo
        would bias the comparison against it, so this mirrors the same threshold and
        the same lookahead frame.
        """
        horizon = k + num_future_frames * HISTORY_STRIDE
        lateral = to_relative_poses(self.planar[k], self.planar[[horizon]])[0, 1]
        distance = float(np.linalg.norm(self.xyz[horizon, :2] - self.xyz[k, :2]))
        if lateral > TURN_THRESHOLD_M:
            return f"Turn left in {distance:.0f}m"
        if lateral < -TURN_THRESHOLD_M:
            return f"Turn right in {distance:.0f}m"
        return "Continue straight"

    def driving_command(self, k: int, num_future_frames: int = 10) -> list[int]:
        """NAVSIM's [left, straight, right, unknown] one-hot, for the paired join."""
        horizon = k + num_future_frames * HISTORY_STRIDE
        lateral = to_relative_poses(self.planar[k], self.planar[[horizon]])[0, 1]
        if lateral > TURN_THRESHOLD_M:
            return [1, 0, 0, 0]
        if lateral < -TURN_THRESHOLD_M:
            return [0, 0, 1, 0]
        return [0, 1, 0, 0]

    # -- the sample --------------------------------------------------------- #

    def build_sample(
        self,
        k: int,
        fov_mode: str = "fill",
        slots: list[tuple[int, str, str]] | None = None,
        num_future_steps: int = NUM_FUTURE_STEPS,
        load_images: bool = True,
    ) -> dict[str, Any]:
        """Assemble the Alpamayo input dict for keyframe *k*.

        *fov_mode* selects how the 120 deg slots are filled (``fill`` or ``letterbox``);
        the telephoto slot is always an exact 30 deg digital zoom.
        """
        slots = CAMERA_SLOTS if slots is None else slots
        assert k - (NUM_HISTORY_STEPS - 1) >= 0, f"keyframe {k} lacks {NUM_HISTORY_STEPS} history"
        assert k - (NUM_CAM_FRAMES - 1) >= 0, f"keyframe {k} lacks {NUM_CAM_FRAMES} camera frames"

        history = np.arange(k - NUM_HISTORY_STEPS + 1, k + 1)
        num_future_valid = int(min(num_future_steps, self.num_frames - 1 - k))
        future = np.arange(k + 1, k + 1 + num_future_valid)

        # Into the t0 ego frame, exactly as load_physical_aiavdataset does it.
        t0_rot = spt.Rotation.from_quat(self.quat[k])
        t0_inv = t0_rot.inv()
        hist_xyz = t0_inv.apply(self.xyz[history] - self.xyz[k])
        hist_rot = (t0_inv * spt.Rotation.from_quat(self.quat[history])).as_matrix()
        fut_xyz = t0_inv.apply(self.xyz[future] - self.xyz[k])
        fut_rot = (t0_inv * spt.Rotation.from_quat(self.quat[future])).as_matrix()

        speed = float(np.linalg.norm(self.body_velocity(k)))
        self._check_history(hist_xyz, hist_rot, speed, k)

        image_frames, camera_indices = self._load_images(k, slots, fov_mode, load_images)

        absolute_us = np.rint(self.reference_timestamps[k - NUM_CAM_FRAMES + 1 : k + 1] * 1e6).astype(
            np.int64
        )
        absolute_timestamps = torch.from_numpy(np.tile(absolute_us, (len(slots), 1)))
        relative = (absolute_timestamps - absolute_timestamps.min()).float() * 1e-6

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": torch.from_numpy(hist_xyz).float()[None, None],
            "ego_history_rot": torch.from_numpy(hist_rot).float()[None, None],
            "ego_future_xyz": torch.from_numpy(fut_xyz).float()[None, None],
            "ego_future_rot": torch.from_numpy(fut_rot).float()[None, None],
            "relative_timestamps": relative,
            "absolute_timestamps": absolute_timestamps,
            "t0_us": int(absolute_us[-1]),
            "clip_id": f"{self.scene_id}:{k}",
            # KITScenes-specific bookkeeping, consumed by the driver and the metrics.
            "kits_scene_id": self.scene_id,
            "kits_keyframe": int(k),
            "kits_num_future_valid": num_future_valid,
            "kits_planar_gt": self.planar_future(k),
            "kits_ego_speed": speed,
            "kits_t0_rot": t0_rot.as_matrix(),
            "kits_t0_yaw": float(self.planar[k, 2]),
            "kits_fov_mode": fov_mode,
        }

    def _check_history(self, hist_xyz: npt.NDArray, hist_rot: npt.NDArray, speed: float, k: int) -> None:
        """Guard the three ways the ego history can be silently wrong."""
        assert np.abs(hist_xyz[-1]).max() < 1e-9, f"history does not end at the origin: {hist_xyz[-1]}"
        assert np.abs(hist_rot[-1] - np.eye(3)).max() < 1e-9, "history does not end at identity"

        # Alpamayo reads v0 off these differences at dt=0.1; a rate or scale error here
        # corrupts every waypoint while still looking like a valid trajectory.
        step_speed = float(np.linalg.norm(hist_xyz[-1] - hist_xyz[-2]) / DT)
        if speed > 1.0:
            assert abs(step_speed - speed) <= 0.15 * speed + 0.1, (
                f"frame {k}: history implies {step_speed:.2f} m/s but poses give {speed:.2f} m/s"
            )
        # Moving forward means the past sits behind us on +x.
        if speed > 2.0:
            assert hist_xyz[0, 0] < -0.5, f"frame {k}: history starts at x={hist_xyz[0, 0]:.2f}"

    def _load_images(
        self, k: int, slots: list[tuple[int, str, str]], fov_mode: str, load_images: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack (N_cam, 4, 3, H, W) uint8 frames, ordered by ascending camera index."""
        ordered = sorted(slots, key=lambda entry: entry[0])
        frame_indices = range(k - NUM_CAM_FRAMES + 1, k + 1)
        camera_indices = torch.tensor([slot for slot, _, _ in ordered], dtype=torch.int64)

        if not load_images:  # CPU self-tests exercise the pose path without the JPEGs
            return torch.zeros((len(ordered), NUM_CAM_FRAMES, 3, 8, 8), dtype=torch.uint8), camera_indices

        per_camera = []
        for _, sensor, kind in ordered:
            pinhole = self._pinhole(sensor)
            mode = "zoom" if kind == "zoom" else fov_mode
            frames = np.stack(
                [
                    load_and_warp(self.scene_path / sensor / f"{i:010d}.jpg", pinhole, mode)
                    for i in frame_indices
                ]
            )
            per_camera.append(rearrange(torch.from_numpy(frames), "t h w c -> t c h w"))

        image_frames = torch.stack(per_camera, dim=0)
        assert image_frames.dtype == torch.uint8
        assert bool((camera_indices[1:] > camera_indices[:-1]).all()), "camera indices not ascending"
        return image_frames, camera_indices
