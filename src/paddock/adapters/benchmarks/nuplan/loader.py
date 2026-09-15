"""Reading nuPlan v1.1 logs without importing nuplan-devkit.

Each nuPlan log is one SQLite database: ego poses at 100 Hz, eight camera
streams at 10 Hz, and scenario tags attached to the 20 Hz lidar keyframes.
Paddock's own environment deliberately holds no driving-stack dependencies, so
everything here is stdlib plus numpy.

The one part of that which is not obvious is calibration. nuPlan stores camera
intrinsics and extrinsics as *pickled devkit objects* --
`nuplan.database.common.data_types.CameraIntrinsic` and friends -- so a plain
`pickle.loads` raises ModuleNotFoundError on any machine without the devkit
installed. Those classes are all `list` subclasses holding plain floats, so
`_CalibrationUnpickler` maps them onto `list` and reads the numbers straight
out. That keeps the devkit, and the torch/hydra tree it drags in, out of
paddock's interpreter -- which is the same reason model deps are imported inside
`load()`.
"""

from __future__ import annotations

import io
import pickle
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

#: The camera every model is scored from, and the one every model declares.
FRONT_CAMERA = "CAM_F0"

#: paddock's camera names -> nuPlan's channels. nuPlan ships eight: CAM_F0
#: forward, CAM_B0 back, and three down each side. The mapping is OpenScene's,
#: so a model that was trained or evaluated against NAVSIM sees the same view
#: under the same name here.
#:
#: Load-bearing for multi-camera models: drivejepa declares four cameras and
#: alpamayo15 three, and `Observation.image` raises rather than substituting a
#: view it was not given -- correctly, since a model indexing its projection
#: matrices positionally would otherwise read each camera through its
#: neighbour's geometry without anything failing.
CHANNELS = {
    "CAM_FRONT": "CAM_F0",
    "CAM_FRONT_LEFT": "CAM_L0",
    "CAM_FRONT_RIGHT": "CAM_R0",
    "CAM_LEFT": "CAM_L1",
    "CAM_RIGHT": "CAM_R1",
    "CAM_BACK_LEFT": "CAM_L2",
    "CAM_BACK_RIGHT": "CAM_R2",
    "CAM_BACK": "CAM_B0",
}

#: Ego poses are logged at 100 Hz, cameras at 10 Hz. The scored anchor is a
#: camera frame, so the harness works in camera time and reads poses by
#: timestamp rather than by row index.
EGO_HZ = 100.0
CAMERA_HZ = 10.0


# ---- calibration ---------------------------------------------------------


class _Row(list):
    """Stand-in for the devkit's list-subclass calibration types.

    `CameraIntrinsic`, `Translation` and `Rotation` all pickle as
    NEWOBJ-then-APPENDS over a list subclass, so an empty `list` subclass
    reconstructs them exactly. Nothing about their behaviour is needed -- only
    the numbers they carry.
    """


_DEVKIT_MODULE = "nuplan.database.common.data_types"
_DEVKIT_TYPES = frozenset({"CameraIntrinsic", "Translation", "Rotation", "Quaternion"})


class _CalibrationUnpickler(pickle.Unpickler):
    """Unpickler that will only build the handful of types a calibration needs.

    A blob out of a downloaded database is untrusted input, and unpickling is
    arbitrary code execution by design. Allowing exactly the devkit's four
    container types and numpy's scalar reconstructors keeps that closed, and
    fails loudly rather than silently returning something odd if nuPlan ever
    changes the encoding.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == _DEVKIT_MODULE and name in _DEVKIT_TYPES:
            return _Row
        # numpy.core.multiarray.scalar + numpy.dtype rebuild the float64 scalars
        # inside Translation and Rotation.
        if module.startswith("numpy"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from a nuPlan calibration blob; "
            "expected only devkit calibration containers and numpy scalars"
        )


def decode_calibration(blob: bytes) -> np.ndarray:
    """One pickled calibration column as a float array."""
    return np.asarray(_CalibrationUnpickler(io.BytesIO(blob)).load(), dtype=np.float64)


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation from a (w, x, y, z) quaternion, nuPlan's storage order."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_of(q: np.ndarray) -> float:
    """Heading in radians from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def camera_calibration(
    conn: sqlite3.Connection, channel: str = FRONT_CAMERA
) -> tuple[np.ndarray, np.ndarray]:
    """(3x3 intrinsics, 4x4 camera-to-ego) for one channel of this log.

    Read from the log rather than assumed: openpilot derives its warp from the
    focal length, so a guessed intrinsic would quietly change what the model
    sees and therefore its score, with nothing failing to say so.
    """
    row = conn.execute(
        "SELECT intrinsic, translation, rotation FROM camera WHERE channel = ?", (channel,)
    ).fetchone()
    if row is None:
        raise LookupError(f"log has no camera channel {channel!r}")
    intrinsics = decode_calibration(row[0]).reshape(3, 3)
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = quaternion_matrix(decode_calibration(row[2]))
    cam2ego[:3, 3] = decode_calibration(row[1])
    return intrinsics, cam2ego


def lidar_calibration(conn: sqlite3.Connection) -> np.ndarray:
    """4x4 lidar-to-ego for this log's merged point cloud.

    Needed even by a camera-only run: BEVFormer-style models build their BEV
    grid in the lidar frame and project the cameras into it, so `lidar2ego` is
    the transform that ties the two together. drivejepa calls
    `Observation.require_lidar2ego()`, which raises rather than assuming
    identity -- right, because an identity here would silently shift the whole
    BEV grid by the sensor offset and still produce a plausible trajectory.

    nuPlan stores it in the same pickled devkit types as the cameras.

    Expect the **identity matrix**, and do not treat that as a bug: nuPlan's
    channel is `MergedPointCloud`, already expressed in the ego frame, so its
    stored translation is [0, 0, 0] and its rotation [1, 0, 0, 0]. The decoder
    is fine -- CAM_F0 in the same log decodes to [1.67, -0.03, 1.52].
    """
    row = conn.execute("SELECT translation, rotation FROM lidar LIMIT 1").fetchone()
    if row is None:
        raise LookupError("log has no lidar calibration")
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = quaternion_matrix(decode_calibration(row[1]))
    lidar2ego[:3, 3] = decode_calibration(row[0])
    return lidar2ego


# ---- locating logs -------------------------------------------------------

#: Where the split's .db files sit under the dataset root. nuPlan's own
#: documentation says `nuplan-v1.1/splits/<split>`, but the mini archive on S3
#: unpacks to `nuplan-v1.1/splits/data/cache/<split>` -- both are in the wild, so
#: both are searched rather than making one site's layout the tracked one.
_SPLIT_DIRS = (
    "nuplan-v1.1/splits/{split}",
    "nuplan-v1.1/splits/data/cache/{split}",
    "nuplan-v1.1/{split}",
)


def split_dirs(root: Path, split: str) -> list[Path]:
    """Candidate directories holding this split's databases, in search order."""
    return [root / pattern.format(split=split) for pattern in _SPLIT_DIRS]


def log_dbs(root: Path, split: str) -> list[Path]:
    """Every log database for *split*, sorted by name.

    Sorted explicitly: `--limit 5` has to choose the same five samples at both
    sites, and the order a filesystem hands back is not a promise.
    """
    for directory in split_dirs(root, split):
        if directory.is_dir():
            found = sorted(directory.glob("*.db"))
            if found:
                return found
    return []


def sensor_root(root: Path) -> Path:
    """Where the camera JPEGs live. `image.filename_jpg` is relative to this."""
    return root / "nuplan-v1.1" / "sensor_blobs"


def log_locations(root: Path) -> dict[str, Path]:
    """Log name -> the directory `image.filename_jpg` should be joined to.

    nuPlan splits its ~450 GB of camera data into nine zips, and each unpacks
    into a directory of its own: `sensor_blobs/nuplan-v1.1_mini_camera_8/<log>/
    CAM_F0/...`. The database only knows `<log>/CAM_F0/<token>.jpg`, so
    something has to know which shard holds which log. Sites that flatten the
    shards, or that hold only one, are equally normal -- so both layouts are
    detected rather than one being declared correct.

    A log directory is recognised by containing camera channels; anything else
    one level down is treated as a shard and searched one level deeper. Built
    once per run: it is a few dozen `listdir` calls, against tens of thousands
    of `exists` checks if resolved per frame.
    """
    sensors = sensor_root(root)
    if not sensors.is_dir():
        return {}
    found: dict[str, Path] = {}
    for entry in sorted(sensors.iterdir()):
        if not entry.is_dir():
            continue  # the LICENSE file each shard ships
        if any((entry / channel).is_dir() for channel in (FRONT_CAMERA, "CAM_B0")):
            found.setdefault(entry.name, sensors)
            continue
        for log_dir in sorted(entry.iterdir()):
            if log_dir.is_dir():
                found.setdefault(log_dir.name, entry)
    return found


def open_log(path: Path) -> sqlite3.Connection:
    """Read-only connection to one log.

    Opened through a file: URI with mode=ro so a run can never write to the
    dataset, including the -wal and -shm files sqlite would otherwise create in
    a directory that is often read-only or shared.
    """
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


# ---- anchors -------------------------------------------------------------


def scenario_anchors(conn: sqlite3.Connection) -> Iterator[tuple[str, int, bytes]]:
    """(scenario type, timestamp_us, ego_pose_token) for every tagged keyframe.

    Ordered by (type, timestamp) so a per-type selection made downstream is
    stable regardless of how sqlite chose to return rows.
    """
    query = """
        SELECT scenario_tag.type, lidar_pc.timestamp, lidar_pc.ego_pose_token
        FROM scenario_tag
        JOIN lidar_pc ON lidar_pc.token = scenario_tag.lidar_pc_token
        ORDER BY scenario_tag.type, lidar_pc.timestamp
    """
    yield from conn.execute(query)


def camera_frames(conn: sqlite3.Connection, channel: str = FRONT_CAMERA) -> list[tuple[int, str, bytes]]:
    """(timestamp_us, filename, ego_pose_token) for one channel, oldest first."""
    query = """
        SELECT image.timestamp, image.filename_jpg, image.ego_pose_token
        FROM image
        JOIN camera ON camera.token = image.camera_token
        WHERE camera.channel = ?
        ORDER BY image.timestamp
    """
    return list(conn.execute(query, (channel,)))


#: Column offsets into the array `ego_poses` returns.
POSE_T, POSE_X, POSE_Y = 0, 1, 2
POSE_QW = 3  # ..QW+4 is the quaternion
POSE_VX, POSE_VY = 7, 8
POSE_AX, POSE_AY = 9, 10
POSE_YAW_RATE = 11


def ego_poses(conn: sqlite3.Connection) -> np.ndarray:
    """All ego poses as (N, 12); see the POSE_* offsets above.

    Acceleration and yaw rate are read, not left at zero. A model reading all
    of speed/accel/yaw-rate cannot distinguish a zeroed channel from a car
    travelling at constant velocity in a straight line, so omitting them is not
    a missing feature but a wrong input that nothing reports -- drivejepa reads
    `Observation.accel` on every frame.

    Returned as one array rather than queried per sample: a log holds ~40 000
    poses and a sample needs a 4 s window of them, so a single read plus
    `searchsorted` beats a query per waypoint by a wide margin.
    """
    rows = conn.execute(
        "SELECT timestamp, x, y, qw, qx, qy, qz, vx, vy, "
        "acceleration_x, acceleration_y, angular_rate_z FROM ego_pose ORDER BY timestamp"
    ).fetchall()
    return np.asarray(rows, dtype=np.float64)


def pose_at(poses: np.ndarray, timestamp_us: int) -> int:
    """Index of the pose nearest *timestamp_us*.

    Nearest rather than preceding: at 100 Hz the two differ by at most 5 ms, and
    a camera frame's own timestamp falls between pose samples about half the
    time.
    """
    index = int(np.searchsorted(poses[:, POSE_T], timestamp_us))
    if index <= 0:
        return 0
    if index >= len(poses):
        return len(poses) - 1
    before, after = poses[index - 1, POSE_T], poses[index, POSE_T]
    return index - 1 if (timestamp_us - before) <= (after - timestamp_us) else index


def speed_at(poses: np.ndarray, index: int) -> float:
    """Ego speed in m/s. Magnitude of the logged velocity, so it does not matter
    whether nuPlan recorded vx/vy in the body or the global frame."""
    return float(np.hypot(poses[index, POSE_VX], poses[index, POSE_VY]))


def accel_at(poses: np.ndarray, index: int) -> float:
    """Forward acceleration in m/s^2, **signed**.

    The longitudinal component alone, not the magnitude: `hypot` would report a
    braking car as accelerating, and braking is most of what the scored nuPlan
    moments are doing. This takes nuPlan's `acceleration_x` as body-frame
    longitudinal, which is how the devkit documents it.
    """
    return float(poses[index, POSE_AX])


def yaw_rate_at(poses: np.ndarray, index: int) -> float:
    """Yaw rate in rad/s about the vertical axis."""
    return float(poses[index, POSE_YAW_RATE])


def ego_frame(poses: np.ndarray, origin: int, indices: np.ndarray) -> np.ndarray:
    """Poses at *indices* expressed in the ego frame at *origin*: x fwd, y left."""
    x0, y0 = poses[origin, POSE_X], poses[origin, POSE_Y]
    yaw = yaw_of(poses[origin, POSE_QW : POSE_QW + 4])
    dx = poses[indices, POSE_X] - x0
    dy = poses[indices, POSE_Y] - y0
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.stack([dx * cos + dy * sin, -dx * sin + dy * cos], axis=1)


def ground_truth(poses: np.ndarray, origin: int, interval_s: float, n: int) -> np.ndarray | None:
    """The recorded future at *origin*, sampled every *interval_s* for *n* steps.

    Returns None when the log ends before the horizon does. Dropping such an
    anchor is deliberate: padding it with the last known pose would read as the
    car stopping, and score a model that predicted the truth as though it were
    wrong.
    """
    times = poses[origin, POSE_T] + np.arange(1, n + 1) * interval_s * 1e6
    if times[-1] > poses[-1, POSE_T]:
        return None
    indices = np.array([pose_at(poses, t) for t in times])
    return ego_frame(poses, origin, indices)
