"""A 10 Hz observation history for a NAVSIM scenario, from nuPlan's own sensor data.

OpenScene, the dataset NAVSIM scores on, keeps one frame per 0.5 s of nuPlan's
10 Hz cameras. A model that reads ego motion off consecutive frames (openpilot,
which has no speed input at all) sees a parked car at that rate, and a model
built for 0.1 s frames (alpamayo) sees the world moving five times too fast.
The frames in between were never lost: they are in nuPlan's public sensor
blobs, under the same log names and image tokens OpenScene uses, and each
log's nuPlan database says which image was taken when and where the ego was.

So, given the keyframe NAVSIM is scoring -- identified by the path of one of
its images -- this module returns the preceding `history_s` of frames at
10 Hz, with the ego pose, velocity, acceleration and yaw rate recorded at
each, expressed in the keyframe's own frame the way NAVSIM expresses its
history. The scorer, the metric cache and the trajectory contract are
untouched: only what the model is shown changes.

**Runs inside the vendor's Python 3.9 interpreter** (see agent.py): stdlib,
numpy, and `convert` only -- no 3.10+ syntax evaluated at import.
"""

from __future__ import annotations

import bisect
import math
import os
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace

from paddock.adapters.benchmarks.navsim.convert import CAMERA_SLOTS

#: nuPlan's camera rate. Its lidar runs at 20 Hz; images at 10 Hz
#: (nuplan_scenario_queries.py in the devkit says so in as many words).
NUPLAN_HZ = 10.0

#: A camera on the rig fires within ~8 ms of the front one and consecutive
#: frames are 100 ms apart, so anything farther than this from a target time is
#: a neighbouring frame, not jitter. Measured on a navtest log.
MATCH_TOLERANCE_S = 0.05

_EGO_QUERY = """
select i.timestamp, i.filename_jpg,
       e.x, e.y, e.qw, e.qx, e.qy, e.qz,
       e.vx, e.vy, e.acceleration_x, e.acceleration_y, e.angular_rate_z
from image i
join camera c on i.camera_token = c.token
join ego_pose e on i.ego_pose_token = e.token
where c.channel = ?
order by i.timestamp
"""


@dataclass
class DenseFrame:
    """One observation of the dense history, in the scored keyframe's frame."""

    #: Microseconds relative to the scored keyframe; 0 for the keyframe itself.
    time_us: int
    #: paddock camera name -> absolute jpg path.
    paths: dict[str, str]
    #: x, y, heading in the keyframe's frame -- the origin for the keyframe.
    pose: tuple[float, float, float]
    #: Ego-frame velocity and acceleration as nuPlan recorded them, m/s and m/s^2.
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    #: rad/s, nuPlan's angular_rate_z -- measured, not differenced.
    yaw_rate: float
    #: The log's first frame, repeated: the scenario sits closer to the log's
    #: start than the history asked for. Same as KITScenes' clamp at a scene
    #: start, and what a model's own warm-up would have shown it.
    clamped: bool = False


def quaternion_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Heading about z of a w-first quaternion; nuPlan's ego_pose stores one."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def relative_se2(
    pose: tuple[float, float, float], origin: tuple[float, float, float]
) -> tuple[float, float, float]:
    """`pose` expressed in the frame whose global pose is `origin`.

    What NAVSIM's `convert_absolute_to_relative_se2_array` does for its own
    history, written out so this module needs nothing from nuplan.
    """
    dx, dy = pose[0] - origin[0], pose[1] - origin[1]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    heading = (pose[2] - origin[2] + math.pi) % (2.0 * math.pi) - math.pi
    return (c * dx + s * dy, -s * dx + c * dy, heading)


def as_ego_status(frame: DenseFrame) -> SimpleNamespace:
    """The three fields `convert.canbus` reads off NAVSIM's EgoStatus."""
    return SimpleNamespace(
        ego_pose=list(frame.pose),
        ego_velocity=list(frame.velocity),
        ego_acceleration=list(frame.acceleration),
    )


class _LogIndex:
    """Every image of the wanted channels in one log, with the ego pose it was taken at.

    Loaded once per log per process -- a log is a few thousand frames per
    camera, and a scenario's history is looked up by bisection after that.
    """

    def __init__(self, db_path: str, channels: list[str]) -> None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"no nuPlan database at {db_path}")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            self.times: dict[str, list[int]] = {}
            self.names: dict[str, list[str]] = {}
            self.ego: dict[str, list[tuple[float, ...]]] = {}
            self._by_name: dict[str, dict[str, int]] = {}
            for channel in channels:
                rows = conn.execute(_EGO_QUERY, (channel,)).fetchall()
                if not rows:
                    raise RuntimeError(f"{db_path}: no {channel} images")
                self.times[channel] = [int(r[0]) for r in rows]
                self.names[channel] = [os.path.basename(str(r[1])) for r in rows]
                self.ego[channel] = [
                    (
                        float(r[2]),
                        float(r[3]),
                        quaternion_yaw(float(r[4]), float(r[5]), float(r[6]), float(r[7])),
                        float(r[8]),
                        float(r[9]),
                        float(r[10]),
                        float(r[11]),
                        float(r[12]),
                    )
                    for r in rows
                ]
                self._by_name[channel] = {name: i for i, name in enumerate(self.names[channel])}
        finally:
            conn.close()

    def index_of(self, channel: str, name: str) -> int | None:
        return self._by_name[channel].get(name)

    def nearest(self, channel: str, t_us: float) -> int:
        times = self.times[channel]
        i = bisect.bisect_left(times, t_us)
        if i <= 0:
            return 0
        if i >= len(times):
            return len(times) - 1
        return i if times[i] - t_us < t_us - times[i - 1] else i - 1


class DenseHistory:
    """Dense frames for scenarios, from a nuPlan root laid out as nuPlan lays it out.

    `root/splits/<split>/<log>.db` and `root/sensor_blobs/<split>/<log>/<CHANNEL>/<token>.jpg`.
    `cameras` are paddock names; each maps to a nuPlan channel through the same
    table the proxy uses for OpenScene, so a model gets the same views either way.
    """

    def __init__(self, root: str, split: str, cameras: list[str], hz: float = NUPLAN_HZ) -> None:
        self.root = str(root)
        self.split = str(split)
        unknown = [c for c in cameras if c not in CAMERA_SLOTS]
        if unknown:
            raise ValueError(f"no nuPlan channel for {unknown!r}")
        self.channels = {camera: CAMERA_SLOTS[camera].upper() for camera in cameras}
        self.hz = float(hz)
        self._logs: dict[str, _LogIndex] = {}

    def db_path(self, log: str) -> str:
        return os.path.join(self.root, "splits", self.split, log + ".db")

    def image_path(self, log: str, channel: str, name: str) -> str:
        return os.path.join(self.root, "sensor_blobs", self.split, log, channel, name)

    def _index(self, log: str) -> _LogIndex:
        index = self._logs.get(log)
        if index is None:
            index = self._logs[log] = _LogIndex(self.db_path(log), sorted(set(self.channels.values())))
        return index

    def frames(self, keyframe_image: str, history_s: float) -> list[DenseFrame]:
        """`history_s` of frames at `hz` ending at the keyframe, oldest first.

        `keyframe_image` is the path of one of the keyframe's images as NAVSIM
        names it -- `<log>/<CHANNEL>/<token>.jpg`, absolute or relative -- and
        that camera anchors the timing: every other camera contributes its
        frame nearest to each of the anchor's target times.
        """
        parts = str(keyframe_image).replace("\\", "/").split("/")
        if len(parts) < 3:
            raise ValueError(f"not a <log>/<channel>/<image> path: {keyframe_image!r}")
        log, anchor, name = parts[-3], parts[-2], parts[-1]
        if anchor not in self.channels.values():
            raise ValueError(f"{anchor} is not one of this model's channels {sorted(self.channels.values())}")
        index = self._index(log)
        at = index.index_of(anchor, name)
        if at is None:
            raise RuntimeError(f"{name} is not a {anchor} image of {log} in {self.db_path(log)}")

        t0 = index.times[anchor][at]
        count = int(round(history_s * self.hz)) + 1
        step_us = 1e6 / self.hz
        tolerance_us = MATCH_TOLERANCE_S * 1e6
        origin = index.ego[anchor][at][:3]
        first = index.times[anchor][0]

        frames: list[DenseFrame] = []
        for k in range(count):
            target = t0 - (count - 1 - k) * step_us
            clamped = target < first - tolerance_us
            paths: dict[str, str] = {}
            for camera, channel in self.channels.items():
                i = 0 if clamped else index.nearest(channel, target)
                if not clamped and abs(index.times[channel][i] - target) > step_us:
                    raise RuntimeError(
                        f"{log}: no {channel} frame within {step_us / 1e6:.1f} s of "
                        f"t0{(target - t0) / 1e6:+.1f} s; the sensor blobs are missing frames"
                    )
                paths[camera] = self.image_path(log, channel, index.names[channel][i])
            a = 0 if clamped else index.nearest(anchor, target)
            x, y, heading, vx, vy, ax, ay, yaw_rate = index.ego[anchor][a]
            frames.append(
                DenseFrame(
                    time_us=int(index.times[anchor][a] - t0),
                    paths=paths,
                    pose=relative_se2((x, y, heading), origin),
                    velocity=(vx, vy),
                    acceleration=(ax, ay),
                    yaw_rate=yaw_rate,
                    clamped=clamped,
                )
            )
        return frames
