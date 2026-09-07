"""nuScenes v1.0 tables -> a compact per-keyframe index, plus the geometry the
open-loop planning protocol needs.

No devkit: the tables are plain JSON and the three big ones (sample_data 1.3 GB,
ego_pose 0.6 GB, sample_annotation 0.6 GB) are each loaded once, reduced to the
val keyframes, and dropped. The result -- ~6k keyframes with camera paths,
calibration, ego pose, CAN-bus status and every annotated box -- is cached as a
pickle so later runs start in seconds.

Frames and conventions follow UniAD/VAD's nuScenes planning evaluation:
* the ego frame of a keyframe is the LIDAR_TOP ego_pose (x forward, y left);
* the ground-truth future is the next six keyframes (0.5 s apart, 3 s);
* the CAN-bus vector is the 18-float layout [translation 3, rotation 4,
  accel 3, rotation_rate 3, velocity 3, patch angle rad, deg];
* the high-level command is inferred from the future, VAD's rule: lateral
  offset of the 3 s pose beyond 2 m -> left / right, else straight.
"""

from __future__ import annotations

import bisect
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
KEYFRAME_DT_S = 0.5
FUTURE_STEPS = 6  # 3 s at 2 Hz
TURN_THRESHOLD_M = 2.0  # VAD's gt_ego_fut_cmd rule at the 3 s horizon
#: UniAD's ego footprint for the collision test (length x width), metres.
EGO_LENGTH_M, EGO_WIDTH_M = 4.084, 1.85


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def quat_to_rot(q) -> np.ndarray:
    """nuScenes quaternion (w, x, y, z) -> 3x3 rotation."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_matrix(translation, rotation) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_rot(rotation)
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def yaw_of(rotation) -> float:
    """Heading about +z of a nuScenes quaternion."""
    r = quat_to_rot(rotation)
    return float(math.atan2(r[1, 0], r[0, 0]))


def to_ego_xy(ego2world: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """World (N, 2|3) -> the ego frame of `ego2world`, planar (N, 2)."""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    world2ego = np.linalg.inv(ego2world)
    local = (world2ego[:3, :3] @ pts.T).T + world2ego[:3, 3]
    return local[:, :2]


def boxes_to_ego(ego2world: np.ndarray, boxes_world: np.ndarray) -> np.ndarray:
    """(N, 5) boxes [x, y, yaw, length, width] world -> the same in the ego frame."""
    if len(boxes_world) == 0:
        return np.zeros((0, 5))
    xy = to_ego_xy(ego2world, boxes_world[:, :2])
    ego_yaw = math.atan2(ego2world[1, 0], ego2world[0, 0])
    yaw = (boxes_world[:, 2] - ego_yaw + math.pi) % (2 * math.pi) - math.pi
    return np.column_stack([xy, yaw, boxes_world[:, 3], boxes_world[:, 4]])


def box_corners(box) -> np.ndarray:
    """[x, y, yaw, length, width] -> (4, 2) corners, counter-clockwise."""
    x, y, yaw, length, width = (float(v) for v in box[:5])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def obb_intersects(box_a, box_b) -> bool:
    """Separating-axis test for two oriented rectangles in the plane."""
    a, b = box_corners(box_a), box_corners(box_b)
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def collides(ego_box, boxes) -> bool:
    """Does the ego footprint at `ego_box` overlap any of `boxes` (N, 5)?"""
    if len(boxes) == 0:
        return False
    # cheap circle reject before the exact test
    ex, ey = float(ego_box[0]), float(ego_box[1])
    reach = 0.5 * math.hypot(EGO_LENGTH_M, EGO_WIDTH_M)
    for box in boxes:
        if math.hypot(float(box[0]) - ex, float(box[1]) - ey) > reach + 0.5 * math.hypot(
            float(box[3]), float(box[4])
        ):
            continue
        if obb_intersects(ego_box, box):
            return True
    return False


def command_from_future(future_xy_ego: np.ndarray) -> tuple[int, str | None, float | None]:
    """VAD's rule -> (paddock Command int, nav_maneuver, nav_distance_m).

    Command ints: 0=RIGHT 1=LEFT 2=STRAIGHT (paddock.policy.protocol.Command).
    """
    if len(future_xy_ego) == 0:
        return 2, None, None
    end = future_xy_ego[-1]
    distance = float(np.hypot(end[0], end[1]))
    if end[1] >= TURN_THRESHOLD_M:
        return 1, "TURN_LEFT", round(distance, 1)
    if end[1] <= -TURN_THRESHOLD_M:
        return 0, "TURN_RIGHT", round(distance, 1)
    return 2, None, None


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@dataclass
class Keyframe:
    token: str
    scene: str
    index_in_scene: int
    timestamp_us: int
    ego2world: np.ndarray  # LIDAR_TOP ego pose
    cameras: dict[str, dict] = field(default_factory=dict)  # name -> {path, intrinsic, cam2ego}
    lidar2ego: np.ndarray | None = None
    boxes_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))  # x y yaw l w
    canbus: np.ndarray = field(default_factory=lambda: np.zeros(18))


def _load(root: Path, version: str, table: str):
    with (root / version / f"{table}.json").open() as handle:
        return json.load(handle)


def build_index(root: Path, version: str, scene_names: list[str], log=print) -> list[list[Keyframe]]:
    """Every keyframe of the named scenes, in scene order then time order."""
    scenes = {s["name"]: s for s in _load(root, version, "scene")}
    wanted = [scenes[n] for n in scene_names if n in scenes]
    missing = [n for n in scene_names if n not in scenes]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split scenes absent from {version}/scene.json, e.g. {missing[:3]}"
        )

    samples = {s["token"]: s for s in _load(root, version, "sample")}
    ordered: list[tuple[str, list[str]]] = []
    for scene in wanted:
        tokens, tok = [], scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = samples[tok]["next"]
        ordered.append((scene["name"], tokens))
    val_tokens = {t for _, toks in ordered for t in toks}
    log(f"nuscenes index: {len(ordered)} scenes, {len(val_tokens)} keyframes")

    sensors = {s["token"]: s["channel"] for s in _load(root, version, "sensor")}
    calibrated = {c["token"]: c for c in _load(root, version, "calibrated_sensor")}

    # sample_data: keyframe records of the val samples only, then drop the table
    log("nuscenes index: reading sample_data.json (1.3 GB) ...")
    per_sample: dict[str, dict[str, dict]] = {t: {} for t in val_tokens}
    for sd in _load(root, version, "sample_data"):
        if not sd["is_key_frame"] or sd["sample_token"] not in per_sample:
            continue
        channel = sensors[calibrated[sd["calibrated_sensor_token"]]["sensor_token"]]
        per_sample[sd["sample_token"]][channel] = {
            "filename": sd["filename"],
            "ego_pose_token": sd["ego_pose_token"],
            "calibrated_sensor_token": sd["calibrated_sensor_token"],
        }
    needed_poses = {rec["ego_pose_token"] for recs in per_sample.values() for rec in recs.values()}

    log("nuscenes index: reading ego_pose.json ...")
    poses = {
        p["token"]: (p["translation"], p["rotation"])
        for p in _load(root, version, "ego_pose")
        if p["token"] in needed_poses
    }

    log("nuscenes index: reading sample_annotation.json ...")
    boxes: dict[str, list] = {t: [] for t in val_tokens}
    for ann in _load(root, version, "sample_annotation"):
        tok = ann["sample_token"]
        if tok in boxes:
            w, length, _h = ann["size"]
            x, y, _z = ann["translation"]
            boxes[tok].append([x, y, yaw_of(ann["rotation"]), length, w])

    out: list[list[Keyframe]] = []
    for scene_name, tokens in ordered:
        can = _CanBus.for_scene(root, scene_name)
        frames: list[Keyframe] = []
        for i, tok in enumerate(tokens):
            recs = per_sample[tok]
            if "LIDAR_TOP" not in recs:
                continue
            lidar = recs["LIDAR_TOP"]
            ego2world = pose_matrix(*poses[lidar["ego_pose_token"]])
            lidar_cal = calibrated[lidar["calibrated_sensor_token"]]
            cams = {}
            for cam in CAMERAS:
                if cam not in recs:
                    continue
                cal = calibrated[recs[cam]["calibrated_sensor_token"]]
                cams[cam] = {
                    "path": str(root / recs[cam]["filename"]),
                    "intrinsic": np.asarray(cal["camera_intrinsic"], dtype=np.float64),
                    "cam2ego": pose_matrix(cal["translation"], cal["rotation"]),
                }
            ts = int(samples[tok]["timestamp"])
            frames.append(
                Keyframe(
                    token=tok,
                    scene=scene_name,
                    index_in_scene=i,
                    timestamp_us=ts,
                    ego2world=ego2world,
                    cameras=cams,
                    lidar2ego=pose_matrix(lidar_cal["translation"], lidar_cal["rotation"]),
                    boxes_world=np.asarray(boxes[tok], dtype=np.float64).reshape(-1, 5),
                    canbus=can.vector(ts, ego2world) if can else np.zeros(18),
                )
            )
        out.append(frames)
    return out


class _CanBus:
    """Nearest CAN-bus pose message per timestamp, one scene at a time."""

    def __init__(self, messages: list[dict]) -> None:
        self.messages = sorted(messages, key=lambda m: m["utime"])
        self.times = [m["utime"] for m in self.messages]

    @classmethod
    def for_scene(cls, root: Path, scene_name: str):
        path = root / "can_bus" / f"{scene_name}_pose.json"
        if not path.exists():
            return None
        with path.open() as handle:
            return cls(json.load(handle))

    def vector(self, timestamp_us: int, ego2world: np.ndarray) -> np.ndarray:
        i = bisect.bisect_left(self.times, timestamp_us)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
        j = min(cands, key=lambda k: abs(self.times[k] - timestamp_us))
        m = self.messages[j]
        v = np.zeros(18)
        v[0:3] = m["pos"]
        v[3:7] = m["orientation"]
        v[7:10] = m["accel"]
        v[10:13] = m["rotation_rate"]
        v[13:16] = m["vel"]
        yaw = math.atan2(ego2world[1, 0], ego2world[0, 0])
        v[16], v[17] = yaw, math.degrees(yaw)
        return v


def load_or_build_index(root: Path, version: str, scene_names: list[str], cache: Path | None, log=print):
    if cache is not None and cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)
    index = build_index(root, version, scene_names, log=log)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump(index, handle)
    return index
