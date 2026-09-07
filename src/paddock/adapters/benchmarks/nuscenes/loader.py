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

from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command

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
    """World (N, 2|3) -> the ego BEV frame of `ego2world`, planar (N, 2).

    A bird's-eye (SE(2)) transform: only the ego yaw and its XY translation, so a
    point's height never enters -- and never has to be fabricated as Z=0 before a
    full 3-D rotation that carries the pose's pitch/roll. This is the UniAD/VAD
    convention, and it keeps annotation boxes (stored without Z) and future ego
    poses (which do carry a real Z) in exactly one frame; a full 3-D transform
    would place them in slightly different planes and shift near-boundary
    collisions.
    """
    pts = np.asarray(points_world, dtype=np.float64)[:, :2]
    yaw = math.atan2(ego2world[1, 0], ego2world[0, 0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ (pts - ego2world[:2, 3]).T).T


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
    """VAD's rule -> (paddock Command, nav_maneuver, nav_distance_m).

    The command follows VAD's 3 s lateral-offset rule; `nav_distance_m` is the
    along-track distance *from the ego* to the first future point whose lateral
    offset crosses the threshold in the commanded direction, measured from the
    ego origin (0, 0) so the leg to the first future point is included, and only
    crossings in the turn's own direction count.

    This is a PROXY for the protocol's "distance to the next maneuver": nuScenes
    carries no map junctions, so onset is inferred from the future trajectory. A
    lateral threshold can be crossed only once a turn is already under way, so
    the value can understate the true lead-in distance; it is a best-effort hint
    for the model's prompt, not a surveyed junction distance.
    """
    if len(future_xy_ego) == 0:
        return int(Command.STRAIGHT), None, None
    lateral = future_xy_ego[-1][1]
    if abs(lateral) < TURN_THRESHOLD_M:
        return int(Command.STRAIGHT), None, None
    if lateral > 0:
        command, maneuver = Command.LEFT, "TURN_LEFT"
        crossed = np.flatnonzero(future_xy_ego[:, 1] >= TURN_THRESHOLD_M)
    else:
        command, maneuver = Command.RIGHT, "TURN_RIGHT"
        crossed = np.flatnonzero(future_xy_ego[:, 1] <= -TURN_THRESHOLD_M)
    # Along-track distance from the ego origin to the crossing point. Prepending
    # (0, 0) makes cumulative[k] the distance to future_xy_ego[k].
    path = np.vstack([[0.0, 0.0], future_xy_ego[:, :2]])
    cumulative = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    distance = float(cumulative[crossed[0]]) if len(crossed) else float(cumulative[-1])
    return int(command), maneuver, round(distance, 1)


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
        if can is None:
            raise FileNotFoundError(
                f"CAN-bus pose file missing for scene {scene_name!r}: "
                f"{root / 'can_bus' / f'{scene_name}_pose.json'}. The nuScenes CAN-bus "
                "expansion supplies the ego speed the models read; without it scoring "
                "would silently treat the ego as stationary. Download the can_bus expansion."
            )
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
                    canbus=can.vector(ts, ego2world),
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
        v[CANBUS_ACCEL : CANBUS_ACCEL + 3] = m["accel"]  # 7:10
        v[CANBUS_ACCEL + 3 : CANBUS_VELOCITY] = m["rotation_rate"]  # 10:13, yaw rate at CANBUS_YAW_RATE
        v[CANBUS_VELOCITY : CANBUS_VELOCITY + 3] = m["vel"]  # 13:16
        yaw = math.atan2(ego2world[1, 0], ego2world[0, 0])
        v[16], v[17] = yaw, math.degrees(yaw)
        return v


#: Bump when the cached index layout (Keyframe fields, geometry conventions) or
#: the build ORDER changes, so an old pickle is rebuilt rather than silently
#: reused. 3: scene order canonicalized (sorted) at build time -- a schema-2
#: cache may hold an unsorted build under a now-sorted key, so it must not match.
_CACHE_SCHEMA = 3


def _cache_key(root: Path, version: str, scene_names: list[str]) -> dict:
    return {
        "schema": _CACHE_SCHEMA,
        "root": str(Path(root).resolve()),
        "version": version,
        "scenes": sorted(scene_names),
    }


def load_or_build_index(root: Path, version: str, scene_names: list[str], cache: Path | None, log=print):
    # Canonicalize the scene order so a cache hit and a cold build return the
    # SAME ordering (the key is order-independent, so the build must be too).
    scene_names = sorted(scene_names)
    key = _cache_key(root, version, scene_names)
    if cache is not None and cache.exists():
        try:
            with cache.open("rb") as handle:
                blob = pickle.load(handle)
            if isinstance(blob, dict) and blob.get("key") == key:
                return blob["index"]
            log(f"nuscenes: index cache {cache} does not match this dataset/version/scenes; rebuilding")
        except Exception as exc:  # noqa: BLE001 -- a bad cache must never be fatal; rebuild
            log(f"nuscenes: could not read index cache {cache} ({exc}); rebuilding")
    index = build_index(root, version, scene_names, log=log)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump({"key": key, "index": index}, handle)
    return index
