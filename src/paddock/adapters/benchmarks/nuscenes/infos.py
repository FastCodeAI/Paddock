"""What UniAD's and VAD's own nuScenes datasets give the model at a keyframe, for the models that ask for it.

Paddock's index (`loader`) builds one set of inputs for every model paired with nuScenes. UniAD and VAD were
trained, and are evaluated by their authors, on what their converters write into an info file
(``create_nuscenes_infos`` in ``uniad_nuscenes_converter`` / ``vad_nuscenes_converter``) and their datasets
build from it (``NuScenesE2EDataset`` / ``VADCustomNuScenesDataset``, ``get_data_info``), and four of those
inputs are not what the index sends:

* **The ego quaternion as nuScenes stores it.** q and -q are one rotation, and the tables store either; the
  index keeps the rotation. The datasets write the stored quaternion into ``can_bus[3:7]``, which then holds
  its w four times (`bev_planners.nuscenes_can_bus`), and the CAN-bus MLP reads that number, sign and all.
* **Each camera where it was when it fired.** The converter's ``obtain_sensor2top`` takes a camera to the
  lidar through the camera's *own* ego pose -- camera, its ego, global, the keyframe's LIDAR_TOP ego,
  lidar -- where the index places every camera with the lidar's ego pose (`sensor2lidar`).
* **The CAN-bus message.** ``_get_can_bus_info`` reads accel, rotation rate and velocity from the first
  ``pose`` message *after* the keyframe, in the file's order; the index takes the nearest
  (`converter_can_bus`).
* **The command**, each repo's own rule in the lidar frame (`uniad_command`, `vad_command`); the index's is
  VAD's threshold applied in the ego frame (`loader.command_from_future`).

A model asks for these in its model.yaml (``inputs: {nuscenes: uniad}`` or ``vad``; `CONVENTIONS`), the run
records it, and only those models get them: every other model's inputs are built exactly as before. The
numbers come from the tables once -- ``sample_data`` and ``ego_pose`` reduced to the val keyframes' LIDAR_TOP
and camera records, and each val scene's CAN-bus ``pose`` file -- and are cached beside the index.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.policy.protocol import Command

#: The input conventions a model may ask nuScenes for, by the repo whose converter and dataset define them.
#: Both build the same CAN bus and camera geometry (the two converters share that code verbatim); they differ
#: in the command (`uniad_command`, `vad_command`).
CONVENTIONS = ("uniad", "vad")

#: The datasets' command index (UniAD ``get_sdc_planning_label``, VAD ``gt_ego_fut_cmd``) -> paddock's.
DATASET_COMMAND = {0: Command.RIGHT, 1: Command.LEFT, 2: Command.STRAIGHT}
#: The 3 s pose's offset to the right (lidar x) beyond which the command turns, metres: both repos' rule.
TURN_M = 2.0
#: Future keyframes the command reads: UniAD ``planning_steps`` (6 in its configs), VAD ``fut_ts``.
FUTURE_STEPS = 6

#: Bump when what `build` stores, or how, changes, so an old cache is rebuilt rather than reused.
_SCHEMA = 2


@dataclass(frozen=True)
class KeyframeInfo:
    """One keyframe as the converters record it, each number as the tables store it."""

    #: The LIDAR_TOP sample_data's ego pose: translation, and quaternion (w, x, y, z) with its stored sign.
    ego_translation: tuple[float, ...]
    ego_rotation: tuple[float, ...]
    #: ``_get_can_bus_info``'s 18 floats: pos, orientation, accel, rotation rate, vel, two zeros.
    can_bus: tuple[float, ...]
    #: Each camera's own keyframe sample_data ego pose, (translation, quaternion), as stored.
    camera_poses: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=dict)
    #: The LIDAR_TOP's and each camera's calibrated_sensor, (translation, quaternion), as stored.
    lidar_calibration: tuple[tuple[float, ...], tuple[float, ...]] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    camera_calibrations: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the converters' and datasets' computations
# ---------------------------------------------------------------------------


def converter_can_bus(messages: list[dict], timestamp_us: int) -> np.ndarray:
    """``_get_can_bus_info`` for one keyframe: the two converters' function, which is the same code.

    It walks the scene's ``pose`` messages in the file's order and stops at the first one after the keyframe.
    pos and orientation come from the last message it passed (the first message, when the first is already
    after the keyframe); accel, rotation_rate and vel -- in the message's own key order -- from the loop
    variable where it stopped: the first message after the keyframe, or the file's last when none is after it.
    """
    if not messages:
        raise ValueError("no CAN-bus pose messages")
    last = pose = messages[0]
    for pose in messages:
        if pose["utime"] > timestamp_us:
            break
        last = pose
    values = [*last["pos"], *last["orientation"]]
    for key in last:
        if key not in ("utime", "pos", "orientation"):
            values.extend(pose[key])
    values.extend([0.0, 0.0])
    if len(values) != 18:
        raise ValueError(
            f"a CAN-bus pose message with keys {sorted(last)} gives {len(values)} floats, not 18"
        )
    return np.asarray(values, dtype=np.float64)


def rotation_matrix(quaternion) -> np.ndarray:
    """pyquaternion's ``Quaternion(q).rotation_matrix``, operation for operation: how the converters and
    datasets build every rotation (normalised only when ``|1 - q.q| >= 1e-14``, then ``Q @ conj(Qbar).T``).

    The converter takes each camera to the lidar through global coordinates (``obtain_sensor2top``), a couple
    of kilometres from the origin, where a rotation equal to the last bit or so -- `loader.quat_to_rot`'s
    closed form -- moves ``lidar2img`` by up to 1.4e-9 over val; built this way it is within 1.3e-12.
    """
    q = np.array([float(v) for v in quaternion])
    if not abs(1.0 - np.dot(q, q)) < 1e-14:
        norm = math.sqrt(np.dot(q, q))
        if norm > 0:
            q = q / norm
    left = np.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]],
        ]
    )
    right = np.array(
        [
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], q[3], -q[2]],
            [q[2], -q[3], q[0], q[1]],
            [q[3], q[2], -q[1], q[0]],
        ]
    )
    return np.dot(left, right.conj().transpose())[1:][:, 1:]


def _pose(translation, quaternion) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation_matrix(quaternion)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def sensor2lidar(camera2ego, camera_ego2world, lidar2ego, ego2world) -> np.ndarray:
    """4x4 camera -> the keyframe's LIDAR_TOP, as ``obtain_sensor2top`` computes it (``sensor2lidar_*``).

    sweep -> its ego -> global -> the lidar's ego -> lidar, with the camera's own ego pose `camera_ego2world`
    and the lidar's `ego2world`, in the converter's own order of operations (row vectors,
    ``points @ R.T + T``).
    """
    l2e_r_mat, l2e_t = np.asarray(lidar2ego, dtype=np.float64)[:3, :3], np.asarray(lidar2ego)[:3, 3]
    e2g_r_mat, e2g_t = np.asarray(ego2world, dtype=np.float64)[:3, :3], np.asarray(ego2world)[:3, 3]
    l2e_r_s_mat, l2e_t_s = np.asarray(camera2ego, dtype=np.float64)[:3, :3], np.asarray(camera2ego)[:3, 3]
    e2g_r_s_mat = np.asarray(camera_ego2world, dtype=np.float64)[:3, :3]
    e2g_t_s = np.asarray(camera_ego2world, dtype=np.float64)[:3, 3]
    to_lidar = np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ to_lidar
    translation = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ to_lidar
    translation -= e2g_t @ to_lidar + l2e_t @ np.linalg.inv(l2e_r_mat).T
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = translation
    return out


def camera_in_keyframe(frame: L.Keyframe, camera: str, info: KeyframeInfo) -> np.ndarray:
    """The camera's pose in the keyframe's ego frame (its LIDAR_TOP ego pose), where it was when it fired.

    ``lidar2ego @ sensor2lidar``: what an Observation's ``camera2ego`` must be for ``K @ inv(camera2ego) @
    lidar2ego`` -- how every BEVFormer planner here builds ``lidar2img`` (`bev_planners.lidar2img`) -- to be
    the dataset's ``lidar2img``, ``viewpad @ inv(sensor2lidar)``. ``sensor2lidar`` is built from the stored
    quaternions with the converter's own rotations (`rotation_matrix`); the index's ``lidar2ego``, the one
    the observation carries, then cancels against itself in the adapter.
    """
    sensor2lidar_ = sensor2lidar(
        _pose(*info.camera_calibrations[camera]),
        _pose(*info.camera_poses[camera]),
        _pose(*info.lidar_calibration),
        _pose(info.ego_translation, info.ego_rotation),
    )
    return np.asarray(frame.lidar2ego, dtype=np.float64) @ sensor2lidar_


def dataset_can_bus(frame: L.Keyframe, info: KeyframeInfo) -> np.ndarray:
    """The 18 floats ``get_data_info`` leaves in ``can_bus``, with the quaternion whole at 3:7.

    0:3 the ego translation, 3:7 the stored ego quaternion (the dataset's assignment then broadcasts its
    w; the adapter does that, `bev_planners.nuscenes_can_bus`), 7:16 the converter's accel, rotation rate
    and vel, 16:18 the heading wrapped to [0, 360) degrees, radians then degrees (``patch_angle``).
    """
    out = np.asarray(info.can_bus, dtype=np.float64).copy()
    out[0:3] = np.asarray(frame.ego2world, dtype=np.float64)[:3, 3]
    out[3:7] = info.ego_rotation
    heading = np.dot(rotation_matrix(info.ego_rotation), np.array([1, 0, 0]))  # nuScenes' quaternion_yaw
    patch_angle = np.arctan2(heading[1], heading[0]) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    out[16] = patch_angle / 180 * np.pi
    out[17] = patch_angle
    return out


def _turn(x: float) -> int:
    """Both repos' rule on the 3 s pose's offset to the right (lidar x): 0 RIGHT, 1 LEFT, 2 straight."""
    if x >= TURN_M:
        return 0
    if x <= -TURN_M:
        return 1
    return 2


def vad_offset(frames: list[L.Keyframe], i: int) -> float:
    """Lidar x of the lidar's pose 3 s on, in keyframe `i`'s lidar frame: VAD's converter (``ego_fut_trajs``).

    The future poses are the LIDAR_TOP's (``get_global_sensor_pose``), held at the last keyframe when the
    scene ends first; then global -> the keyframe's ego -> its lidar, in float64.
    """
    future = frames[i : i + FUTURE_STEPS + 1]
    last = np.asarray(future[-1].ego2world) @ np.asarray(future[-1].lidar2ego)
    here = frames[i]
    point = last[:3, 3] - np.asarray(here.ego2world)[:3, 3]
    point = np.asarray(here.ego2world)[:3, :3].T @ point
    point = point - np.asarray(here.lidar2ego)[:3, 3]
    point = np.asarray(here.lidar2ego)[:3, :3].T @ point
    return float(point[0])


def uniad_offset(frames: list[L.Keyframe], i: int) -> float | None:
    """Lidar x of UniAD's planning label 3 s on (``NuScenesTraj.get_sdc_planning_label``); None if no future.

    The label is the ego box -- centred on the future LIDAR_TOP, its bottom 0.78 m below
    (``LiDARInstance3DBoxes`` with origin (0.5, 0.5, 0.5) and height 1.56) -- moved lidar -> ego -> global at
    that keyframe and global -> ego -> lidar at this one, as the box methods do it: each matrix and
    translation cast to float32 and applied to float32 row vectors (``rotate(m)`` is ``p @ m``).
    """
    future = frames[i + 1 : i + 1 + FUTURE_STEPS]
    if not future:
        return None
    last, here = future[-1], frames[i]

    def rotate(point: np.ndarray, matrix) -> np.ndarray:
        return point @ np.asarray(matrix, dtype=np.float64).astype(np.float32)

    def translate(point: np.ndarray, vector) -> np.ndarray:
        return point + np.asarray(vector, dtype=np.float64).astype(np.float32)

    point = np.zeros((1, 3), dtype=np.float32)
    point[0, 2] += np.float32(1.56) * (np.float32(0.0) - np.float32(0.5))
    point = rotate(point, np.asarray(last.lidar2ego)[:3, :3].T)
    point = translate(point, np.asarray(last.lidar2ego)[:3, 3])
    point = rotate(point, np.asarray(last.ego2world)[:3, :3].T)
    point = translate(point, np.asarray(last.ego2world)[:3, 3])
    point = translate(point, -np.asarray(here.ego2world)[:3, 3])
    point = rotate(point, np.linalg.inv(np.asarray(here.ego2world)[:3, :3]).T)
    point = translate(point, -np.asarray(here.lidar2ego)[:3, 3])
    point = rotate(point, np.linalg.inv(np.asarray(here.lidar2ego)[:3, :3]).T)
    return float(point[0, 0])


def dataset_command(convention: str, frames: list[L.Keyframe], i: int) -> int:
    """Keyframe `i`'s command as `convention`'s dataset gives it: 0 RIGHT, 1 LEFT, 2 straight."""
    if convention == "uniad":
        offset = uniad_offset(frames, i)
        return 2 if offset is None else _turn(offset)
    if convention == "vad":
        return _turn(vad_offset(frames, i))
    raise ValueError(f"no nuScenes input convention {convention!r} (one of {', '.join(CONVENTIONS)})")


def command(convention: str, frames: list[L.Keyframe], i: int) -> tuple[int, str | None, None]:
    """(paddock Command, nav_maneuver, nav_distance_m) for the dataset's command.

    The maneuver names the turn the command is; the datasets carry no distance to it, so none is sent.
    """
    value = DATASET_COMMAND[dataset_command(convention, frames, i)]
    maneuver = {Command.LEFT: "TURN_LEFT", Command.RIGHT: "TURN_RIGHT"}.get(value)
    return int(value), maneuver, None


# ---------------------------------------------------------------------------
# from the tables, cached
# ---------------------------------------------------------------------------


def build(root: Path, version: str, scene_names: list[str], log=print) -> dict[str, KeyframeInfo]:
    """Every keyframe of the named scenes that the index holds (one with a LIDAR_TOP), by sample token."""
    scenes = {s["name"]: s for s in L._load(root, version, "scene")}
    missing = [name for name in scene_names if name not in scenes]
    if missing:
        raise FileNotFoundError(f"{len(missing)} scenes absent from {version}/scene.json, e.g. {missing[:3]}")
    samples = {s["token"]: s for s in L._load(root, version, "sample")}
    ordered: list[tuple[str, list[str]]] = []
    for name in scene_names:
        tokens, token = [], scenes[name]["first_sample_token"]
        while token:
            tokens.append(token)
            token = samples[token]["next"]
        ordered.append((name, tokens))
    sensors = {s["token"]: s["channel"] for s in L._load(root, version, "sensor")}
    calibrated = {c["token"]: c for c in L._load(root, version, "calibrated_sensor")}
    channel_of = {token: sensors[c["sensor_token"]] for token, c in calibrated.items()}
    channels = {"LIDAR_TOP", *L.CAMERAS}

    log("nuscenes infos: reading sample_data.json (1.3 GB) ...")
    pose_tokens: dict[str, dict[str, str]] = {token: {} for _, tokens in ordered for token in tokens}
    calibration: dict[str, dict[str, tuple]] = {token: {} for token in pose_tokens}
    for record in L._load(root, version, "sample_data"):
        if not record["is_key_frame"] or record["sample_token"] not in pose_tokens:
            continue
        channel = channel_of[record["calibrated_sensor_token"]]
        if channel in channels:
            pose_tokens[record["sample_token"]][channel] = record["ego_pose_token"]
            cal = calibrated[record["calibrated_sensor_token"]]
            calibration[record["sample_token"]][channel] = (
                tuple(float(v) for v in cal["translation"]),
                tuple(float(v) for v in cal["rotation"]),
            )
    needed = {token for by_channel in pose_tokens.values() for token in by_channel.values()}
    log("nuscenes infos: reading ego_pose.json ...")
    poses = {
        p["token"]: (tuple(float(v) for v in p["translation"]), tuple(float(v) for v in p["rotation"]))
        for p in L._load(root, version, "ego_pose")
        if p["token"] in needed
    }

    out: dict[str, KeyframeInfo] = {}
    for name, tokens in ordered:
        path = Path(root) / "can_bus" / f"{name}_pose.json"
        if not path.exists():
            raise FileNotFoundError(f"CAN-bus pose file missing for scene {name!r}: {path}")
        with path.open() as handle:
            messages = json.load(handle)
        for token in tokens:
            by_channel = pose_tokens[token]
            if "LIDAR_TOP" not in by_channel:
                continue  # the index skips it too
            translation, rotation = poses[by_channel["LIDAR_TOP"]]
            out[token] = KeyframeInfo(
                ego_translation=translation,
                ego_rotation=rotation,
                can_bus=tuple(converter_can_bus(messages, int(samples[token]["timestamp"])).tolist()),
                camera_poses={
                    camera: poses[by_channel[camera]] for camera in L.CAMERAS if camera in by_channel
                },
                lidar_calibration=calibration[token]["LIDAR_TOP"],
                camera_calibrations={
                    camera: calibration[token][camera] for camera in L.CAMERAS if camera in by_channel
                },
            )
    return out


def _cache_key(root: Path, version: str, scene_names: list[str]) -> dict:
    return {
        "kind": "infos",
        "schema": _SCHEMA,
        "root": str(Path(root).resolve()),
        "version": version,
        "scenes": sorted(scene_names),
    }


def load_or_build(root: Path, version: str, scene_names: list[str], cache: Path | None, log=print):
    """`build`'s infos: from `cache` when it holds this root's, version's and scenes', else built, cached."""
    scene_names = sorted(scene_names)
    return L.cached(
        cache,
        _cache_key(root, version, scene_names),
        lambda: build(root, version, scene_names, log=log),
        log=log,
        what="infos",
    )
