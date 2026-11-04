"""Adapt KITScenes Multimodal scenes into NAVSIM ``AgentInput`` samples.

Drive-JEPA consumes nuPlan/OpenScene conventions: a 2 Hz history of four frames,
cameras in the ``cam_f0 / cam_l0 / cam_r0 / cam_b0`` layout with intrinsics and
``sensor2lidar`` extrinsics, and an ego status in the lidar frame.  KITScenes
stores 10 Hz per-sensor directories, a roof-pod reference frame, and TUM poses.
This module bridges the two.

Three conventions have to be reconciled; each is handled explicitly below.

1. **Reference frame.**  KITScenes calibrates everything against ``base_frame``,
   which ``calib.json`` shows is identical to ``lidar_top`` -- an origin on the
   roof pod, ~2.2 m above the road.  OpenScene's lidar frame sits at road level
   (its cameras are at z = +1.52).  We estimate the pod height from the ground
   return of the lidar sweep and drop the origin to the road, so the extrinsics
   we hand the model are geometrically truthful.

2. **Field of view.**  The KITScenes ring cameras are 3504x2272 at 87 deg hFOV;
   OpenScene is 1920x1080 at 64 deg.  We centre-crop about the principal point
   and rescale so the resulting image carries *exactly* the OpenScene intrinsics
   (f=1545, c=(960, 560)), which is what the model was trained on.

3. **Ego dynamics.**  OpenScene ships ``ego_dynamic_state``; KITScenes ships
   poses only, so velocity and acceleration are finite-differenced off the 10 Hz
   pose stream and rotated into the body frame.

The rig geometries still differ (roof pod vs. windshield mounts), which no
transform can undo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar
from PIL import Image

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

# NAVSIM logs are 2 Hz with four history frames; KITScenes is 10 Hz.
HISTORY_STRIDE = 5
NUM_HISTORY_FRAMES = 4

# OpenScene camera geometry that the checkpoint was trained against.
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080
TARGET_FOCAL = 1545.0
TARGET_CU, TARGET_CV = 960.0, 560.0

# KITScenes ring camera -> NAVSIM slot.  Yaw angles measured from calib.json are
# 1 / 61 / -58 / -179 deg against OpenScene's 0 / 55 / -55 / 180.
CAMERA_MAP: dict[str, str] = {
    "cam_f0": "camera_ring_front",
    "cam_l0": "camera_ring_front_left",
    "cam_r0": "camera_ring_front_right",
    "cam_b0": "camera_ring_rear",
}

POSES_FILENAME = "poses.txt"
REFERENCE_TIMESTAMP_FILENAME = "timestamp.reference.txt"
CALIBRATION_FILENAME = "calibration/calib.json"

# Lateral offset (m) over the prediction horizon above which a sample counts as
# a turn.  Only used by the ``gt`` driving-command mode.
TURN_THRESHOLD_M = 5.0


@dataclass
class KITScenesSample:
    """One keyframe turned into model input plus its ground-truth future."""

    scene_id: str
    keyframe_index: int
    agent_input: AgentInput
    # (num_poses, 3) x, y, heading in the keyframe ego frame
    future_poses: npt.NDArray[np.float64]
    future_interval_s: float


# --------------------------------------------------------------------------- #
# Pose handling
# --------------------------------------------------------------------------- #


def _quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw of a (qx, qy, qz, qw) quaternion, the TUM ordering used by poses.txt."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def load_poses(scene_path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load ``poses.txt`` as (timestamps_s, (N, 3) x/y/yaw in the local UTM frame)."""
    raw = np.loadtxt(scene_path / POSES_FILENAME)
    if raw.ndim == 1:
        raw = raw[None, :]
    timestamps = raw[:, 0]
    poses = np.stack(
        [raw[:, 1], raw[:, 2], [_quaternion_to_yaw(*row) for row in raw[:, 4:8]]],
        axis=-1,
    )
    return timestamps, poses


def to_relative_poses(
    origin: npt.NDArray[np.float64], poses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Express absolute (x, y, yaw) poses in the frame of *origin*."""
    cos_o, sin_o = np.cos(origin[2]), np.sin(origin[2])
    delta = poses[:, :2] - origin[:2]
    rotated = np.stack(
        [cos_o * delta[:, 0] + sin_o * delta[:, 1], -sin_o * delta[:, 0] + cos_o * delta[:, 1]],
        axis=-1,
    )
    heading = np.arctan2(np.sin(poses[:, 2] - origin[2]), np.cos(poses[:, 2] - origin[2]))
    return np.concatenate([rotated, heading[:, None]], axis=-1)


def _body_frame_dynamics(
    timestamps: npt.NDArray[np.float64], poses: npt.NDArray[np.float64], index: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Central-difference velocity and acceleration at *index*, in the body frame.

    OpenScene's ``ego_dynamic_state`` is ``[vx, vy, ax, ay]`` in the ego frame;
    KITScenes only ships poses, so we differentiate the 10 Hz stream.
    """
    last = len(timestamps) - 1
    prev_i, next_i = max(index - 1, 0), min(index + 1, last)

    def velocity_at(i: int) -> npt.NDArray[np.float64]:
        lo, hi = max(i - 1, 0), min(i + 1, last)
        dt = timestamps[hi] - timestamps[lo]
        if dt <= 0:
            return np.zeros(2)
        return (poses[hi, :2] - poses[lo, :2]) / dt

    velocity_global = velocity_at(index)
    dt_outer = timestamps[next_i] - timestamps[prev_i]
    acceleration_global = (
        (velocity_at(next_i) - velocity_at(prev_i)) / dt_outer if dt_outer > 0 else np.zeros(2)
    )

    yaw = poses[index, 2]
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (
        (rotation @ velocity_global).astype(np.float32),
        (rotation @ acceleration_global).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Camera handling
# --------------------------------------------------------------------------- #


def _crop_box(intrinsics: dict) -> tuple[int, int, int, int, float]:
    """Crop window that maps the source pinhole onto the OpenScene intrinsics.

    Choosing ``scale = TARGET_FOCAL / f`` matches the field of view, and anchoring
    the window on the principal point puts it at (TARGET_CU, TARGET_CV) after the
    resize -- so every adapted camera ends up with identical, exact intrinsics.
    """
    focal = float(intrinsics["focal_length"])
    scale = TARGET_FOCAL / focal
    left = intrinsics["principal_point_u"] - TARGET_CU / scale
    top = intrinsics["principal_point_v"] - TARGET_CV / scale
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + TARGET_WIDTH / scale)),
        int(round(top + TARGET_HEIGHT / scale)),
        scale,
    )


def target_intrinsics() -> npt.NDArray[np.float32]:
    """The OpenScene camera matrix every adapted image is warped onto."""
    return np.array(
        [[TARGET_FOCAL, 0.0, TARGET_CU], [0.0, TARGET_FOCAL, TARGET_CV], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class KITScenesSceneAdapter:
    """Builds NAVSIM ``AgentInput`` samples from one KITScenes scene directory."""

    def __init__(self, scene_path: Path, ground_offset: float | None = None) -> None:
        self.scene_path = Path(scene_path)
        self.scene_id = self.scene_path.name
        self._calibration = json.loads((self.scene_path / CALIBRATION_FILENAME).read_text())
        self.reference_timestamps = np.loadtxt(self.scene_path / REFERENCE_TIMESTAMP_FILENAME)
        self.pose_timestamps, self.poses = load_poses(self.scene_path)
        self.num_frames = len(self.reference_timestamps)
        self.ground_offset = ground_offset if ground_offset is not None else self.estimate_ground_offset()

    # -- geometry ----------------------------------------------------------- #

    def estimate_ground_offset(self, frame_index: int = 0) -> float:
        """Height of the roof-pod origin above the road, from the lidar ground return.

        The road is the dominant surface in the -3.5 m .. -1.5 m band below a roof
        pod, so its median z is a stable estimate without needing a plane fit.
        """
        try:
            import pyarrow.parquet as pq

            sweep_path = self.scene_path / "lidar_top" / f"{frame_index:010d}.parquet"
            table = pq.read_table(sweep_path, columns=["x", "y", "z"])
            # Coordinates are stored as int32 ticks; the metre scale lives in the
            # parquet metadata as ``discretization_resolution``.
            resolution = float(table.schema.metadata[b"discretization_resolution"])
            points = np.stack([table[axis].to_numpy() for axis in ("x", "y", "z")], axis=-1) * resolution
            near = points[(np.abs(points[:, 0]) < 15.0) & (np.abs(points[:, 1]) < 8.0)]
            ground = near[(near[:, 2] > -3.5) & (near[:, 2] < -1.5)]
            if len(ground) < 100:
                raise ValueError(f"only {len(ground)} ground candidates")
            return float(-np.median(ground[:, 2]))
        except Exception as exc:  # noqa: BLE001 - fall back rather than abort a run
            fallback = 2.2
            print(f"  ! ground estimate failed ({exc}); falling back to {fallback} m")
            return fallback

    @cache
    def _camera_extrinsics(self, sensor: str) -> tuple[npt.NDArray, npt.NDArray]:
        """``sensor2lidar`` rotation/translation with the origin dropped to the road."""
        transform = np.array(self._calibration[sensor]["T_to_reference"], dtype=np.float64)
        rotation = transform[:3, :3]
        translation = transform[:3, 3] + np.array([0.0, 0.0, self.ground_offset])
        return rotation.astype(np.float32), translation.astype(np.float32)

    def _load_camera(self, sensor: str, frame_index: int) -> Camera:
        entry = self._calibration[sensor]
        image_path = self.scene_path / sensor / f"{frame_index:010d}.jpg"
        left, top, right, bottom, _ = _crop_box(entry["intrinsics"])
        image = Image.open(image_path).convert("RGB").crop((left, top, right, bottom))
        image = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)

        rotation, translation = self._camera_extrinsics(sensor)
        return Camera(
            image=np.asarray(image),
            sensor2lidar_rotation=rotation,
            sensor2lidar_translation=translation,
            intrinsics=target_intrinsics(),
            distortion=np.zeros(5, dtype=np.float32),
            camera_path=image_path,
        )

    # -- sample construction ------------------------------------------------ #

    def valid_keyframes(
        self, num_future_frames: int, stride: int = 10, first_frame: int | None = None
    ) -> list[int]:
        """Keyframes with a full history behind them and a full future ahead.

        *first_frame* and *stride* are in 10 Hz frames.  The paper's e2e protocol
        anchors each keyframe after 4 s of past observation and spaces the 9 s
        windows so they do not overlap, i.e. ``first_frame=40, stride=90``.
        """
        minimum = HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1)
        first = minimum if first_frame is None else max(first_frame, minimum)
        last = self.num_frames - num_future_frames * HISTORY_STRIDE - 1
        return list(range(first, last + 1, stride)) if last >= first else []

    def _driving_command(self, keyframe: int, num_future_frames: int, mode: str) -> npt.NDArray[np.int64]:
        """NAVSIM's 4-way one-hot: [left, straight, right, unknown].

        Verified empirically against the navtest logs -- index 0 correlates with
        positive future lateral offset, index 2 with negative.  KITScenes has no
        route annotation, so ``gt`` infers the command from the recorded future
        (a route hint the benchmark would normally supply) and ``straight``
        pins it, which is the honest choice when no route is available.
        """
        if mode == "straight":
            return np.array([0, 1, 0, 0], dtype=np.int64)

        horizon = keyframe + num_future_frames * HISTORY_STRIDE
        lateral = to_relative_poses(self.poses[keyframe], self.poses[horizon][None, :])[0, 1]
        if lateral > TURN_THRESHOLD_M:
            return np.array([1, 0, 0, 0], dtype=np.int64)
        if lateral < -TURN_THRESHOLD_M:
            return np.array([0, 0, 1, 0], dtype=np.int64)
        return np.array([0, 1, 0, 0], dtype=np.int64)

    def build_sample(
        self,
        keyframe: int,
        num_future_frames: int = 8,
        camera_frames: tuple[int, ...] = (2, 3),
        command_mode: str = "gt",
    ) -> KITScenesSample:
        """Assemble the ``AgentInput`` for *keyframe* plus its ground-truth future.

        *camera_frames* are history indices, matching the agent's ``SensorConfig``:
        Drive-JEPA reads ``cam_f0`` at [2, 3] and the side/rear cameras at [3].
        """
        history = [
            keyframe - HISTORY_STRIDE * (NUM_HISTORY_FRAMES - 1 - i) for i in range(NUM_HISTORY_FRAMES)
        ]
        local_history = to_relative_poses(self.poses[keyframe], self.poses[history])
        command = self._driving_command(keyframe, num_future_frames, command_mode)

        ego_statuses: list[EgoStatus] = []
        cameras: list[Cameras] = []
        lidars: list[Lidar] = []

        for position, frame_index in enumerate(history):
            velocity, acceleration = _body_frame_dynamics(self.pose_timestamps, self.poses, frame_index)
            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_history[position].astype(np.float32),
                    ego_velocity=velocity,
                    ego_acceleration=acceleration,
                    driving_command=command,
                )
            )

            slots = {name: Camera() for name in CAMERA_MAP}
            if position in camera_frames:
                # cam_f0 is needed at both history indices, the rest only at the last.
                wanted = CAMERA_MAP if position == max(camera_frames) else {"cam_f0": CAMERA_MAP["cam_f0"]}
                for slot, sensor in wanted.items():
                    slots[slot] = self._load_camera(sensor, frame_index)

            cameras.append(
                Cameras(
                    cam_f0=slots["cam_f0"],
                    cam_l0=slots["cam_l0"],
                    cam_l1=Camera(),
                    cam_l2=Camera(),
                    cam_r0=slots["cam_r0"],
                    cam_r1=Camera(),
                    cam_r2=Camera(),
                    cam_b0=slots["cam_b0"],
                )
            )
            lidars.append(Lidar())  # the perception-based agent leaves lidar_pc empty

        future = [keyframe + HISTORY_STRIDE * (i + 1) for i in range(num_future_frames)]
        future_poses = to_relative_poses(self.poses[keyframe], self.poses[future])

        return KITScenesSample(
            scene_id=self.scene_id,
            keyframe_index=keyframe,
            agent_input=AgentInput(ego_statuses, cameras, lidars),
            future_poses=future_poses,
            future_interval_s=HISTORY_STRIDE / 10.0,
        )
