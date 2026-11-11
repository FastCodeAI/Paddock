"""Paddock observations as Alpamayo 2 Edge's input, in numpy.

Edge reads one sample, shaped the way the vendor's PhysicalAI-AV loader shapes it
(`load_physical_aiavdataset.py` in NVlabs/alpamayo2-edge at 02f55a7):

- **22 camera frames from six cameras.** Front wide (ID 1) and front telephoto
  (ID 6) at -2, -1, -0.5, -0.1 and 0 s; cross left (0), cross right (2), rear left
  (3) and rear right (5) at -1, -0.1 and 0 s. Each frame keeps the time it was
  taken, and the prompt prints that time beside it (`<t=-0.5s>`).
- **20 ego poses** 0.1 s apart, from -1.9 s to now, in the frame of the newest.
- **One route sentence**, "Turn left in 25m" or "Continue straight".

This module decides which recorded frame fills each slot, crops it, rebuilds the
pose history at 0.1 s from whatever rate the source records, and writes the
sentence. It needs numpy and PIL only, so its tests run in paddock's own
environment; torch and the vendor package are used only in policy.py.

*Cameras.* Five paddock cameras fill Edge's IDs by where they look; the sixth
slot, the 30 deg telephoto, is a crop of CAM_FRONT, as in alpamayo15. Every frame
is the widest 16:9 window around its camera's principal point, resized to one
common size, because the vendor packs all 22 into one tensor. NAVSIM's cameras are
64 deg wide, and the 16:9 window around their principal point 62 deg, where Edge's
front and cross cameras are 120 deg: in those three slots things look nearer than
they are. The rear slots (70 deg) are close, and the telephoto crop is exact.

*Frames.* `nearest`, the default, is the vendor's decoder rule: for each slot, the
newest frame at or before its time. At NAVSIM's 2 Hz that gives the front cameras
the frames at -1.5, -1.0, -0.5, -0.5 and 0 s and the other four -1.0, -0.5 and 0 s;
the prompt labels each with the time it was taken, so the model is told which
frame is repeated and how old it is.

*History.* NAVSIM sends four poses, 0.5 s apart. Straight lines between them give
every 0.1 s step of a segment the segment's average speed, so the newest steps
carry the speed of 0.25 s ago, off by the acceleration times 0.25 s, and Edge
reads its starting speed from those steps. `hermite`, the default, joins the poses
with cubics that also pass through each frame's measured velocity, which is exact
for a car speeding up or braking steadily. The oldest 0.4 s lie before NAVSIM's
first frame and are continued backwards at that frame's velocity. Repeating the
first pose there would instead write four parked steps into the history tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from paddock.policy.protocol import Command

#: Edge camera ID -> (paddock camera, field of view to crop to in degrees, None for
#: the widest 16:9 window). ID 4, the rear telephoto, is not one of Edge's inputs.
SLOTS: tuple[tuple[int, str, float | None], ...] = (
    (0, "CAM_FRONT_LEFT", None),
    (1, "CAM_FRONT", None),
    (2, "CAM_FRONT_RIGHT", None),
    (3, "CAM_BACK_LEFT", None),
    (5, "CAM_BACK_RIGHT", None),
    (6, "CAM_FRONT", 30.0),
)
#: The cameras the slots read, CAM_FRONT first: NAVSIM's 10 Hz path anchors its
#: history on the first declared camera.
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

#: Seconds before now each camera's frames are taken at, oldest first
#: (`_FRONT_OFFSETS_US` and `_OTHER_OFFSETS_US` in the vendor's loader).
FRONT_OFFSETS_S = (-2.0, -1.0, -0.5, -0.1, 0.0)
OTHER_OFFSETS_S = (-1.0, -0.1, 0.0)
HISTORY_STEPS = 20
HISTORY_DT_S = 0.1
PLAN_STEPS = 64
PLAN_DT_S = 0.1
#: (width, height) every frame is resized to: the vendor's camera size, so the
#: processor's own resize sees what it sees in the vendor's pipeline.
FRAME_SIZE = (1920, 1080)
#: A frame this much later than the time a slot asks for still counts as at or
#: before it: a 10 Hz source's frames sit a few milliseconds off the 0.1 s grid.
JITTER_S = 0.02

FRAME_MODES = ("nearest", "distinct")
HISTORY_MODES = ("hermite", "linear")


def offsets_for(camera_id: int) -> tuple[float, ...]:
    return FRONT_OFFSETS_S if camera_id in (1, 6) else OTHER_OFFSETS_S


# -- frames ---------------------------------------------------------------------


def select_frames(
    times_s: Sequence[float], offsets_s: Sequence[float], mode: str = "nearest", now_s: float | None = None
) -> list[int]:
    """Indices into `times_s` of the frames that fill one camera's slots, oldest slot first.

    `times_s` are the times of the frames that carry this camera, increasing;
    `now_s` is the newest observation's time, by default the last of them.

    `nearest`: for each offset, the newest frame at or before `now_s + offset`,
    or the oldest frame when none is that old -- the vendor's rule, which on a
    10 Hz source returns exactly its frames.

    `distinct`: the most recent frames whatever their spacing, the oldest
    repeated when there are fewer frames than slots. This is alpamayo15's rule
    for a source slower than the model's grid, kept to compare against: Edge is
    told each frame's real time either way.
    """
    times = np.asarray(times_s, dtype=np.float64)
    if times.size == 0:
        raise ValueError("no frame to select from")
    now = float(times[-1]) if now_s is None else float(now_s)
    if mode == "nearest":
        picks = []
        for offset in offsets_s:
            eligible = np.flatnonzero(times <= now + offset + JITTER_S)
            picks.append(int(eligible[-1]) if eligible.size else 0)
        return picks
    if mode == "distinct":
        recent = list(range(max(0, times.size - len(offsets_s)), times.size))
        return [recent[0]] * (len(offsets_s) - len(recent)) + recent
    raise ValueError(f"frame selection {mode!r} is not one of {FRAME_MODES}")


def crop_box(
    width: int, height: int, fx: float, cx: float, cy: float, hfov_deg: float | None = None
) -> tuple[int, int, int, int]:
    """A 16:9 window centred on the principal point: (left, top, right, bottom).

    `hfov_deg=None` is the widest window the sensor allows; a value narrows it to
    that field of view. alpamayo15's `camera_warp.crop_box` with the same rule,
    restated here because one adapter does not import another.
    """
    half_w = min(cx, width - 1 - cx)
    if hfov_deg is not None:
        half_w = min(half_w, fx * math.tan(math.radians(0.5 * hfov_deg)))
    half_w = min(half_w, min(cy, height - 1 - cy) * 16.0 / 9.0)
    box_w = int(round(2.0 * half_w))
    box_h = int(round(box_w * 9.0 / 16.0))
    left, top = int(round(cx - 0.5 * box_w)), int(round(cy - 0.5 * box_h))
    box = (left, top, left + box_w, top + box_h)
    if box_w <= 0 or left < 0 or top < 0 or box[2] > width or box[3] > height:
        raise ValueError(
            f"crop {box} does not fit the {width}x{height} frame "
            f"(principal point {cx:.1f},{cy:.1f}, field of view {hfov_deg})"
        )
    return box


def warp(image: np.ndarray, fx: float, cx: float, cy: float, hfov_deg: float | None = None) -> np.ndarray:
    """One camera frame cropped to its slot and resized to FRAME_SIZE: (1080, 1920, 3) uint8."""
    from PIL import Image

    height, width = image.shape[:2]
    box = crop_box(width, height, fx, cx, cy, hfov_deg)
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).crop(box).resize(FRAME_SIZE, Image.BICUBIC)
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def crop_hfov_deg(width: int, height: int, fx: float, cx: float, cy: float, hfov_deg: float | None) -> float:
    """The field of view a slot actually receives, for diagnostics."""
    left, _, right, _ = crop_box(width, height, fx, cx, cy, hfov_deg)
    return math.degrees(2.0 * math.atan(0.5 * (right - left) / fx))


# -- history --------------------------------------------------------------------


def _hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """The rotation vector (axis times angle) of a rotation of less than a half turn."""
    cos = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if angle < 1e-9:
        return 0.5 * axis
    if angle > math.pi - 1e-6:
        raise ValueError("a half turn between two frames has no unique interpolation")
    return angle / (2.0 * math.sin(angle)) * axis


def rotation_exp(vector: np.ndarray) -> np.ndarray:
    """The rotation matrix of a rotation vector (Rodrigues' formula)."""
    angle = float(np.linalg.norm(vector))
    k = _hat(np.asarray(vector, dtype=np.float64))
    if angle < 1e-9:
        return np.eye(3) + k
    return np.eye(3) + math.sin(angle) / angle * k + (1.0 - math.cos(angle)) / angle**2 * (k @ k)


def resample_poses(
    times_s: Sequence[float],
    poses: np.ndarray,
    query_s: Sequence[float],
    velocities: np.ndarray | None = None,
    mode: str = "hermite",
) -> tuple[np.ndarray, np.ndarray]:
    """Poses at `query_s`, from poses recorded at `times_s`: world positions (Q, 3), rotations (Q, 3, 3).

    `poses` are (N, 4, 4) ego-to-world at strictly increasing times; `velocities`
    (N, 3) are each frame's measured velocity in the world frame, or None.
    Between frames, rotations turn at a constant rate (slerp) and positions follow
    `mode`: `hermite` the cubic through both frames' poses and velocities,
    `linear` the straight line, which is also what `hermite` falls back to when
    there are no velocities. Before the first frame both continue at that frame's
    velocity -- measured, or the first segment's for `linear` -- and the first
    segment's turn rate. One frame alone is continued at its velocity, or held.
    """
    if mode not in HISTORY_MODES:
        raise ValueError(f"history {mode!r} is not one of {HISTORY_MODES}")
    times = np.asarray(times_s, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    query = np.asarray(query_s, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != times.size or times.size == 0:
        raise ValueError(f"need (N, 4, 4) poses at N times, got {poses.shape} at {times.size}")
    if np.any(np.diff(times) <= 0):
        raise ValueError("pose times must increase strictly")
    positions, rotations = poses[:, :3, 3], poses[:, :3, :3]
    measured = None if velocities is None else np.asarray(velocities, dtype=np.float64).reshape(-1, 3)

    xyz = np.empty((query.size, 3))
    rot = np.empty((query.size, 3, 3))
    if times.size == 1:
        start_velocity = measured[0] if measured is not None else np.zeros(3)
        for q, t in enumerate(query):
            xyz[q] = positions[0] + start_velocity * (t - times[0])
            rot[q] = rotations[0]
        return xyz, rot

    durations = np.diff(times)
    # Turn rate of each segment, in the body frame of its first pose.
    rates = [rotation_log(rotations[i].T @ rotations[i + 1]) / durations[i] for i in range(times.size - 1)]
    cubic = mode == "hermite" and measured is not None
    start_velocity = measured[0] if cubic else (positions[1] - positions[0]) / durations[0]
    for q, t in enumerate(query):
        if t <= times[0]:
            xyz[q] = positions[0] + start_velocity * (t - times[0])
            rot[q] = rotations[0] @ rotation_exp(rates[0] * (t - times[0]))
            continue
        i = min(int(np.searchsorted(times, t, side="right")) - 1, times.size - 2)
        h = durations[i]
        s = (t - times[i]) / h
        if cubic:
            xyz[q] = (
                (2 * s**3 - 3 * s**2 + 1) * positions[i]
                + (s**3 - 2 * s**2 + s) * h * measured[i]
                + (-2 * s**3 + 3 * s**2) * positions[i + 1]
                + (s**3 - s**2) * h * measured[i + 1]
            )
        else:
            xyz[q] = (1 - s) * positions[i] + s * positions[i + 1]
        rot[q] = rotations[i] @ rotation_exp(rates[i] * (t - times[i]))
    return xyz, rot


def ego_history(
    times_s: Sequence[float],
    poses: np.ndarray,
    velocities: np.ndarray | None = None,
    mode: str = "hermite",
) -> tuple[np.ndarray, np.ndarray]:
    """Edge's 20 poses, -1.9 s to the newest frame, in the newest pose's frame: (20, 3), (20, 3, 3).

    As the vendor's loader writes them: positions R0^T (p - p0) and rotations R0^T R.
    """
    now = float(np.asarray(times_s, dtype=np.float64)[-1])
    query = now - HISTORY_DT_S * np.arange(HISTORY_STEPS - 1, -1, -1)
    xyz, rot = resample_poses(times_s, poses, query, velocities, mode)
    origin, frame = np.asarray(poses)[-1, :3, 3], np.asarray(poses)[-1, :3, :3]
    return (xyz - origin) @ frame, frame.T[None] @ rot


def history_before_first_frame_s(times_s: Sequence[float]) -> float:
    """How much of the 1.9 s history lies before the oldest recorded frame and is continued, not read."""
    times = np.asarray(times_s, dtype=np.float64)
    return max(0.0, round((HISTORY_STEPS - 1) * HISTORY_DT_S - float(times[-1] - times[0]), 6))


# -- route ----------------------------------------------------------------------


def nav_text(
    command: int,
    maneuver: str | None = None,
    distance_m: float | None = None,
    nominal_m: float | None = None,
) -> str:
    """The route sentence in Edge's phrasing: "Turn left in 25m", "Turn right", "Continue straight".

    A source's own next maneuver and distance (`nav_maneuver`, `nav_distance_m`)
    win where it sends them. Otherwise the 4-way driving command every source
    carries decides: LEFT and RIGHT turn, anything else goes straight. NAVSIM gives
    no distance, so its turns have none -- the sentence alpamayo15 is given there
    too -- unless `nominal_m` supplies one.
    """
    if maneuver is not None:
        if maneuver not in ("TURN_LEFT", "TURN_RIGHT"):
            return "Continue straight"
        side = "left" if maneuver == "TURN_LEFT" else "right"
    elif int(command) in (Command.LEFT, Command.RIGHT):
        side = "left" if int(command) == Command.LEFT else "right"
        distance_m = None
    else:
        return "Continue straight"
    distance = distance_m if distance_m is not None else nominal_m
    if distance is None:
        return f"Turn {side}"
    return f"Turn {side} in {max(0, int(round(distance)))}m"


# -- output ---------------------------------------------------------------------


def plan_from_poses(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Edge's 64 future poses as paddock waypoints: (64, 3) rows of x, y and yaw."""
    xyz, rot = np.asarray(xyz, dtype=np.float64), np.asarray(rot, dtype=np.float64)
    yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    return np.column_stack([xyz[:, :2], yaw])


def medoid_index(trajectories: np.ndarray) -> int:
    """Index of the draw closest to all the others: (K, T, D) -> int."""
    if len(trajectories) == 1:
        return 0
    flat = np.asarray(trajectories, dtype=np.float64).reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).sum(axis=1)
    return int(np.argmin(distances))
