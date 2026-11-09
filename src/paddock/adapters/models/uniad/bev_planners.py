"""Inputs and outputs of the BEVFormer-based planners: UniAD and VAD, nuScenes and Bench2Drive checkpoints.

Four models share one shape. Six surround cameras are lifted into a bird's-eye-view grid laid out in a
*lidar* frame (x right, y forward, z up); the previous grid is warped onto the current one with the ego
motion read from an 18-float CAN-bus vector; and the plan is six (x, y) waypoints 0.5 s apart in that same
lidar frame. What differs is the code base (OpenMMLab's mmdet3d for the nuScenes checkpoints,
Bench2DriveZoo's vendored copy for the Bench2Drive ones) and three conventions, each of which is a silent
failure when crossed:

* **The CAN-bus layout.** nuScenes puts acceleration at 7:10, rotation rate at 10:13 and velocity at 13:16
  (the order of the CAN-bus expansion's ``pose`` message); Bench2DriveZoo puts velocity at 7:10,
  acceleration at 10:13 and rotation rate at 13:16 (``prepare_B2D.py``). Both keep translation at 0:3 and
  the heading, in radians then degrees, at 16:18; at 3:7 Bench2DriveZoo has the heading's quaternion and the
  nuScenes datasets the ego quaternion's w four times (`nuscenes_can_bus`). Paddock's own canbus is the
  nuScenes layout, so for Bench2Drive it is re-ordered here.
* **``l2g_r_mat``.** The nuScenes datasets store the *transpose* of the lidar-to-global rotation (their
  tracker multiplies row vectors); Bench2DriveZoo stores the rotation itself. Each model is fed its own.
* **The lidar.** Every model's grid and plan live in the lidar frame it was trained with. nuScenes'
  LIDAR_TOP sits ~0.94-0.99 m ahead of the rear axle and 1.84 m up, yawed -90 degrees; Bench2DriveZoo's
  virtual lidar sits 0.39 m behind the CARLA vehicle origin and 1.84 m up, with the same axes. Paddock's
  closed-loop clients send ``lidar2ego`` as the identity ("lidar = ego"), which is the wrong axes for these
  models -- a grid turned a quarter turn, as Drive-JEPA's adapter once found -- so a lidar is used only where
  the source sends one in LIDAR_TOP's axes, and the model's training mount is used everywhere else
  (`model_lidar2ego`).

The plan comes back as the *lidar's* future positions in the current lidar frame -- both code bases build
the planning target from the lidar origin's future poses (UniAD ``get_sdc_planning_label``, VAD
``vad_nuscenes_converter``, Bench2DriveZoo ``get_ego_future_xy``) -- so `lidar_plan_to_ego` turns it into
the ego origin's path, lever arm included (``paddock.policy.protocol.Trajectory`` is the ego origin's).

Everything in this module above `BevPlannerPolicy` is numpy only, so paddock's own tests check it without
the models. The policy itself imports the model's code base inside `load()`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    Command,
    Observation,
    Policy,
    Trajectory,
)

#: Spacing and count of the plan's waypoints: UniAD ``planning_steps = 6``, VAD ``fut_ts = 6``, both 0.5 s
#: apart (nuScenes keyframes; Bench2DriveZoo samples its 10 Hz logs every 5th frame, ``sample_interval=5``).
PLAN_DT = 0.5
PLAN_STEPS = 6

#: Camera order of the nuScenes info files (``camera_types`` in the UniAD and VAD data converters). The
#: spatial cross-attention adds a learned embedding per camera *index*, so the order is part of the model.
NUSC_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
#: Camera order of Bench2DriveZoo (``CAMERAS`` in ``mmcv/datasets/prepare_B2D.py``, and its agents).
B2D_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

#: (width, height) every config loads its cameras at: nuScenes' 1600x900, which Bench2Drive's rig
#: reproduces. Neither test pipeline resizes before normalising (VAD scales by 0.8 after), so a source at
#: another size is brought to this one first, intrinsics with it (`to_load_size`).
LOAD_SIZE = (1600, 900)

#: Standard gravity, m/s^2: the vertical term of an accelerometer reading at rest.
GRAVITY = 9.81


def pose_matrix(translation, quaternion) -> np.ndarray:
    """4x4 from a translation and a (w, x, y, z) quaternion, as nuScenes stores both."""
    out = np.eye(4)
    out[:3, :3] = quaternion_matrix(quaternion)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def quaternion_matrix(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation, normalising first, as pyquaternion's ``rotation_matrix`` does."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quaternion(rotation) -> np.ndarray:
    """3x3 rotation -> a (w, x, y, z) unit quaternion with w >= 0 (Shepperd's method)."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out = np.asarray(q, dtype=np.float64)
    out /= np.linalg.norm(out)
    return out if out[0] >= 0 else -out


def yaw_of(rotation) -> float:
    """Heading of a rotation's x axis in the xy plane, radians: nuScenes' ``quaternion_yaw``."""
    m = np.asarray(rotation, dtype=np.float64)
    return float(math.atan2(m[1, 0], m[0, 0]))


# --------------------------------------------------------------------------
# The lidar frame and the cameras
# --------------------------------------------------------------------------

#: nuScenes LIDAR_TOP as calibrated on the log of the phase-1 fixture (scene-0103; calibrated_sensor
#: d051cafdd9fe4d999b413462364d44a0): the frame UniAD's and VAD's grids are laid out in when a source has no
#: lidar of its own in those axes. A recorded calibration rather than an idealised one, so every number is
#: one the checkpoints trained with; nuScenes' other LIDAR_TOP calibration differs by 4 cm and ~1.5 degrees.
NUSC_LIDAR2EGO = pose_matrix(
    [0.985793, 0.0, 1.84019],
    [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719],
)

#: Bench2DriveZoo's lidar in the CARLA vehicle frame (right-handed), as both of its agents hard-code it
#: (``self.lidar2ego`` in ``team_code/uniad_b2d_agent.py`` and ``vad_b2d_agent.py``) and its converter
#: builds it (``lidar_to_righthand_ego``): 0.39 m behind the vehicle origin, 1.84 m up, x right, y forward.
B2D_LIDAR2EGO = np.array(
    [
        [0.0, 1.0, 0.0, -0.39],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.84],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

#: How far a source's lidar axes may sit from LIDAR_TOP's (x right, y forward, z up) and still be read as
#: one, degrees. nuScenes' own calibrations are within 2.5.
LIDAR_AXES_TOLERANCE_DEG = 10.0


def has_lidar_top_axes(lidar2ego) -> bool:
    """Whether a lidar points x right, y forward and z up in the ego frame, as nuScenes' LIDAR_TOP does."""
    rotation = np.asarray(lidar2ego, dtype=np.float64)[:3, :3]
    target = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # columns: lidar x, y, z in ego
    cos_limit = math.cos(math.radians(LIDAR_AXES_TOLERANCE_DEG))
    return all(float(rotation[:, i] @ target[:, i]) >= cos_limit for i in range(3))


def model_lidar2ego(source_lidar2ego, nominal: np.ndarray) -> tuple[np.ndarray, str]:
    """The lidar frame the model's grid is laid out in, and where it came from ("observation" | "nominal").

    A source lidar in LIDAR_TOP's axes is used as it is: on nuScenes data it is the very sensor whose frame
    the checkpoint trained in, per log. Anything else -- no lidar, or paddock's identity ("lidar = ego",
    which HUGSIM, Bench2Drive and NAVSIM send) -- gets the model's training mount instead. The identity
    itself would lay the grid out x forward, y left: a quarter turn from training, so every sample point
    of the spatial cross-attention would miss its camera.
    """
    if source_lidar2ego is not None and has_lidar_top_axes(source_lidar2ego):
        return np.asarray(source_lidar2ego, dtype=np.float64), "observation"
    return np.asarray(nominal, dtype=np.float64), "nominal"


def lidar2img(camera2image, camera2ego, lidar2ego) -> np.ndarray:
    """4x4 lidar -> pixel projection: the padded intrinsics times lidar -> camera.

    The nuScenes datasets build it from ``sensor2lidar`` and Bench2DriveZoo's from ``cam2ego`` and
    ``lidar2ego``; both are ``K @ inv(camera2ego) @ lidar2ego`` once the camera's pose and the lidar's are
    in one ego frame. Paddock's camera frame is the optical one (x right, y down, z forward), the same as
    both datasets'.
    """
    pad = np.eye(4)
    pad[:3, :3] = np.asarray(camera2image, dtype=np.float64)
    return pad @ np.linalg.inv(np.asarray(camera2ego, dtype=np.float64)) @ np.asarray(lidar2ego, np.float64)


def scaled_intrinsics(camera2image, sx: float, sy: float) -> np.ndarray:
    """K for an image resized by (sx, sy): the models' own convention, ``diag(s, s, 1) @ K``.

    VAD's ``RandomScaleImageMultiViewImage`` scales ``lidar2img`` exactly so, without a half-pixel term; the
    HUGSIM clients double K the same way for their 800x450 -> 1600x900 resize.
    """
    return np.diag([sx, sy, 1.0]) @ np.asarray(camera2image, dtype=np.float64)


def to_load_size(image: np.ndarray, camera2image, size=LOAD_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """An RGB frame at the config's load size, and its intrinsics. Unchanged when it is already that size.

    Shrinking uses area interpolation, enlarging bilinear (cv2's own advice for each). Needs cv2, so it is
    called only in the model's environment. A source of another aspect ratio is scaled anisotropically and
    its intrinsics with it: the projection stays exact, the pixels are no longer square.
    """
    height, width = image.shape[:2]
    if (width, height) == tuple(size):
        return image, np.asarray(camera2image, dtype=np.float64)
    import cv2

    shrink = width > size[0] or height > size[1]
    resized = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return resized, scaled_intrinsics(camera2image, size[0] / width, size[1] / height)


# --------------------------------------------------------------------------
# The CAN bus
# --------------------------------------------------------------------------


def accel_with_gravity(accel) -> tuple[np.ndarray, bool]:
    """(ax, ay, az) as an accelerometer reads it, and whether gravity had to be added.

    Both training sets hold specific force, gravity included: nuScenes' CAN-bus ``accel`` reads z ~ +9.6 to
    9.8, and Bench2Drive's comes from CARLA's IMU, z +9.81 at rest. Most paddock sources send the kinematic
    acceleration, z = 0, which no accelerometer on a car reports (it would be free fall). So a z within
    g/2 of zero is taken to be one without gravity, and g is added.
    """
    out = np.asarray(accel, dtype=np.float64).copy()
    if abs(out[2]) < GRAVITY / 2:
        out[2] += GRAVITY
        return out, True
    return out, False


def continuous_quaternion(rotation, previous=None) -> np.ndarray:
    """A quaternion for `rotation` on the same sign branch as `previous`, w >= 0 on the first.

    q and -q are one rotation, and nuScenes stores either (57% of the ego poses in paddock's nuScenes
    subset have w < 0), but the CAN-bus MLP reads numbers, not the rotation -- w, for the nuScenes
    checkpoints (`nuscenes_can_bus`). A pose source gives a smooth quaternion within a log, so the sign is
    held continuous across an episode rather than flipped wherever w crosses zero.
    """
    q = matrix_quaternion(rotation)
    if previous is not None and float(q @ np.asarray(previous, dtype=np.float64)) < 0:
        q = -q
    return q


def source_quaternion(canbus, rotation) -> np.ndarray | None:
    """The source's own quaternion at canbus 3:7, if it is a unit quaternion of exactly `rotation`, else None.

    A source that fills the slot the way the nuScenes info files do (the ego pose's quaternion) keeps its
    sign; HUGSIM and Bench2Drive leave it zero, and nuScenes' CAN-bus orientation is not the ego pose's.
    """
    q = np.asarray(canbus, dtype=np.float64)[3:7]
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-6:
        return None
    if not np.allclose(quaternion_matrix(q), np.asarray(rotation, dtype=np.float64)[:3, :3], atol=1e-6):
        return None
    return q.copy()


def nuscenes_can_bus(ego2world, canbus, quaternion) -> tuple[np.ndarray, bool]:
    """The 18 floats UniAD's and VAD's nuScenes datasets put in ``can_bus``, and whether gravity was added.

    ``get_data_info`` overwrites 0:3 with the ego translation, 3:7 with the ego rotation and 16:18 with the
    heading (``patch_angle``, wrapped to [0, 360) degrees: radians, then degrees); 7:16 stay the CAN-bus
    ``pose`` message's accel, rotation_rate and vel, which is paddock's own layout.

    3:7 holds the quaternion's *w four times*, not the quaternion. The datasets write
    ``can_bus[3:7] = Quaternion(...)``, and a pyquaternion ``Quaternion`` has ``__float__`` (its w) but no
    ``__len__``, so numpy assigns it as a scalar and broadcasts it. Every checkpoint trained on that, so it
    is fed that. (Bench2DriveZoo wraps the quaternion in ``list()`` and gets all four.)
    """
    pose = np.asarray(ego2world, dtype=np.float64)
    source = np.asarray(canbus, dtype=np.float64)
    out = np.zeros(18)
    out[0:3] = pose[:3, 3]
    out[3:7] = float(np.asarray(quaternion, dtype=np.float64)[0])
    out[CANBUS_ACCEL : CANBUS_ACCEL + 3], added = accel_with_gravity(source[CANBUS_ACCEL : CANBUS_ACCEL + 3])
    out[CANBUS_ACCEL + 3 : CANBUS_VELOCITY] = source[CANBUS_ACCEL + 3 : CANBUS_VELOCITY]
    out[CANBUS_VELOCITY : CANBUS_VELOCITY + 3] = source[CANBUS_VELOCITY : CANBUS_VELOCITY + 3]
    patch_angle = yaw_of(pose) / np.pi * 180
    if patch_angle < 0:
        patch_angle += 360
    out[16] = patch_angle / 180 * np.pi
    out[17] = patch_angle
    return out, added


def nuscenes_l2g(ego2world, lidar2ego) -> tuple[np.ndarray, np.ndarray]:
    """(``l2g_r_mat``, ``l2g_t``) as the nuScenes datasets compute them: the rotation *transposed*, float32.

    ``l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T`` and ``l2g_t = l2e_t @ e2g_r_mat.T + e2g_t``: the tracker
    moves its reference points as row vectors, ``p @ l2g_r_mat + l2g_t``.
    """
    pose = np.asarray(ego2world, dtype=np.float64)
    lidar = np.asarray(lidar2ego, dtype=np.float64)
    rotation = lidar[:3, :3].T @ pose[:3, :3].T
    translation = lidar[:3, 3] @ pose[:3, :3].T + pose[:3, 3]
    return rotation.astype(np.float32), translation.astype(np.float32)


def b2d_yaw(ego2world) -> float:
    """The heading in Bench2Drive's ``ego_yaw`` range, (-3pi/2, pi/2] radians.

    Bench2Drive computes it as pi/2 - compass from CARLA's IMU, compass in [0, 2pi); paddock's right-handed
    heading is the same angle, in (-pi, pi]. The range matters because the quaternion at 3:7 is built from
    it unwrapped.
    """
    yaw = yaw_of(np.asarray(ego2world, dtype=np.float64))
    return yaw - 2 * np.pi if yaw > np.pi / 2 else yaw


def b2d_can_bus(ego2world, canbus) -> tuple[np.ndarray, bool]:
    """The 18 floats Bench2DriveZoo's datasets put in ``can_bus`` (``B2D_E2E_Dataset.get_data_info``).

    0:3 ego translation with z = 0; 3:7 ``Quaternion(axis=z, radians=ego_yaw)``; 7:10 ``ego_vel`` =
    (speed, 0, 0); 10:13 ``ego_accel``, CARLA's IMU accelerometer y-flipped to right-handed (gravity in z);
    13:16 ``ego_rotation_rate`` = -gyroscope, which in right-handed terms is (roll rate, -pitch rate, yaw
    rate); 16:18 the heading wrapped to [0, 2pi), radians then degrees.

    Two places where the dataset and Bench2DriveZoo's own agents disagree follow the dataset the checkpoints
    were trained on: the agents put the *unwrapped* heading at 16, which differs by 2pi for three quarters
    of all headings, and they never wrap it.
    """
    source = np.asarray(canbus, dtype=np.float64)
    yaw = b2d_yaw(ego2world)
    out = np.zeros(18)
    out[0:2] = np.asarray(ego2world, dtype=np.float64)[0:2, 3]
    out[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    out[7] = source[CANBUS_VELOCITY]
    out[10:13], added = accel_with_gravity(source[CANBUS_ACCEL : CANBUS_ACCEL + 3])
    roll_rate, pitch_rate, yaw_rate = source[CANBUS_ACCEL + 3 : CANBUS_VELOCITY]
    out[13:16] = [roll_rate, -pitch_rate, yaw_rate]
    wrapped = yaw + 2 * np.pi if yaw < 0 else yaw
    out[16] = wrapped
    out[17] = wrapped / np.pi * 180
    return out, added


def b2d_l2g(ego2world) -> tuple[np.ndarray, np.ndarray]:
    """(``l2g_r_mat``, ``l2g_t``) as Bench2DriveZoo computes them: the rotation *not* transposed.

    ``ego2world`` is rebuilt from the heading alone with z = 0, as the dataset and the agents build it
    (``Quaternion(axis=z, radians=ego_yaw)``, ``ego_translation`` with z = 0), and the lidar placed on it.
    """
    yaw = b2d_yaw(ego2world)
    pose = np.eye(4)
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[0:2, 3] = np.asarray(ego2world, dtype=np.float64)[0:2, 3]
    lidar2global = pose @ B2D_LIDAR2EGO
    return lidar2global[:3, :3], lidar2global[:3, 3]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

#: paddock Command -> the nuScenes models' index. UniAD's ``get_sdc_planning_label`` and VAD's converter
#: both derive it from the 3 s pose's lateral offset in the lidar frame (x right): x >= 2 m -> 0 "RIGHT",
#: x <= -2 m -> 1 "LEFT", else 2 "FORWARD"/"Go Straight". UniAD embeds it (``navi_embed``), VAD selects one
#: of its three plans with it. FOLLOW_REFERENCE is "keep going", which is FORWARD in that vocabulary.
NUSC_COMMAND = {
    Command.RIGHT: 0,
    Command.LEFT: 1,
    Command.STRAIGHT: 2,
    Command.FOLLOW_REFERENCE: 2,
}

#: Bench2DriveZoo's six commands: CARLA's RoadOption minus one (``command_near``; VOID reads as LANEFOLLOW).
B2D_LEFT, B2D_RIGHT, B2D_STRAIGHT, B2D_LANEFOLLOW, B2D_CHANGELANELEFT, B2D_CHANGELANERIGHT = range(6)

#: paddock Command -> Bench2DriveZoo's, before the lane-change refinement in `b2d_command`.
B2D_COMMAND = {
    Command.RIGHT: B2D_RIGHT,
    Command.LEFT: B2D_LEFT,
    Command.STRAIGHT: B2D_STRAIGHT,
    Command.FOLLOW_REFERENCE: B2D_LANEFOLLOW,
}

#: Metres ahead within which a lane change on the route becomes the command. Bench2DriveZoo's agents
#: take the command of the first route node beyond ``RoutePlanner(4.0, 50.0)``'s 4 m.
B2D_ROUTE_LOOKAHEAD_M = 4.0


def nuscenes_command(obs: Observation) -> int:
    return NUSC_COMMAND[Command(int(obs.command))]


def b2d_command(obs: Observation) -> int:
    """Bench2DriveZoo's command for this observation.

    Paddock's Bench2Drive client sends CARLA's road option at the route point nearest the ego, with lane
    changes folded into STRAIGHT, and the next lane change separately (``nav_lane_change``). Bench2Drive's
    models were trained with the lane changes as commands of their own, so a lane change within
    `B2D_ROUTE_LOOKAHEAD_M` is restored as one -- unless the ego is mid-turn, where the turn is kept.

    A source without a route (HUGSIM, nuScenes, NAVSIM: ``route_xy`` is None) speaks the three-way
    vocabulary, where STRAIGHT means "not turning". Bench2Drive's STRAIGHT is the junction command; keeping
    along the road is LANEFOLLOW in its training data, so that is what STRAIGHT becomes there. Bench2Drive's
    own client sends a route, and its STRAIGHT is CARLA's.
    """
    command = Command(int(obs.command))
    near = obs.nav_lane_change_m is not None and obs.nav_lane_change_m <= B2D_ROUTE_LOOKAHEAD_M
    if near and command in (Command.STRAIGHT, Command.FOLLOW_REFERENCE):
        if obs.nav_lane_change == "LEFT":
            return B2D_CHANGELANELEFT
        if obs.nav_lane_change == "RIGHT":
            return B2D_CHANGELANERIGHT
    if command == Command.STRAIGHT and obs.route_xy is None:
        return B2D_LANEFOLLOW
    return B2D_COMMAND[command]


# --------------------------------------------------------------------------
# The plan, and the history it is planned from
# --------------------------------------------------------------------------

#: A waypoint step shorter than this carries no usable heading (the HUGSIM bridge's threshold too).
MIN_STEP_M = 0.3


def lidar_plan_to_ego(plan, lidar2ego) -> np.ndarray:
    """The model's plan -- the lidar origin's future (x, y) in the current lidar frame -- as the ego's path.

    Each waypoint is placed in the ego frame (z = 0 in the lidar frame: the lidar keeps its height), which
    gives where the *lidar* will be. The ego origin is the lidar's lever arm behind that, turned by the
    heading the car will have then, which is taken from the path itself (central differences; a step under
    `MIN_STEP_M` keeps the previous heading, and the first starts from the current one, 0). On a straight
    road the two corrections cancel and only the axes change; through a turn the arm swings with the car.
    """
    plan = np.asarray(plan, dtype=np.float64)[:, :2]
    lidar = np.asarray(lidar2ego, dtype=np.float64)
    points = np.c_[plan, np.zeros(len(plan))] @ lidar[:3, :3].T + lidar[:3, 3]
    path = np.vstack([lidar[:2, 3], points[:, :2]])  # the lidar now, then at each waypoint
    arm = lidar[:2, 3]
    out = np.empty_like(plan)
    heading = 0.0
    for k in range(1, len(path)):
        ahead = path[min(k + 1, len(path) - 1)]
        step = ahead - path[k - 1]
        if math.hypot(step[0], step[1]) >= MIN_STEP_M:
            heading = math.atan2(step[1], step[0])
        c, s = math.cos(heading), math.sin(heading)
        out[k - 1] = path[k] - np.array([c * arm[0] - s * arm[1], s * arm[0] + c * arm[1]])
    return out


def history_on_grid(times_us, now_us: int, after_us: int | None, dt_s: float = PLAN_DT) -> list[int]:
    """Indices of the frames to feed before `now_us`, oldest first: one per `dt_s` step back from it.

    The temporal BEV fuses the previous grid as if it were `dt_s` old -- every checkpoint trained on 0.5 s
    steps. An open-loop sequence at 10 Hz would otherwise show it a world moving at a fifth of its speed.
    Each step takes the frame nearest its target within a quarter step and stops at the first gap, at the
    frame the model last stepped on (`after_us`), or when the frames run out.
    """
    tolerance = 0.25 * dt_s * 1e6
    chosen: list[int] = []
    k = 1
    while True:
        target = now_us - k * dt_s * 1e6
        if after_us is not None and target <= after_us + tolerance:
            break
        candidates = [i for i, t in enumerate(times_us) if abs(t - target) <= tolerance and i not in chosen]
        if not candidates:
            break
        chosen.append(min(candidates, key=lambda i: abs(times_us[i] - target)))
        k += 1
    return chosen[::-1]


#: A call less than one plan step (0.5 s, less this slack for clock rounding) after the model's last step
#: is answered from the held plan (`held_plan`) instead of a new step: a client asking every 0.1 s
#: (neuroNCAP) would otherwise have the temporal BEV fuse grids 0.1 s apart as if they were 0.5 s. Clients
#: that honour ``planning_hz`` (HUGSIM, Bench2Drive) call 0.5 s apart, within a millisecond, and never
#: reach it.
STEP_SLACK_S = 0.02


def held_plan(
    plan_xy, planned_from: np.ndarray, planned_us: int, ego2world: np.ndarray, now_us: int
) -> np.ndarray:
    """A plan made at `planned_us` from pose `planned_from`, sampled at now + 0.5 k in the current ego frame.

    The plan is held fixed in the world, as the HUGSIM and Bench2Drive clients hold one between replans:
    its waypoints and the ego's position when it was made are interpolated in time, and past the plan's end
    the last segment's velocity continues. Planar: the poses' headings and xy only.
    """
    plan = np.asarray(plan_xy, dtype=np.float64)[:, :2]
    start = np.asarray(planned_from, dtype=np.float64)
    c0, s0 = start[0, 0], start[1, 0]
    norm0 = math.hypot(c0, s0) or 1.0
    c0, s0 = c0 / norm0, s0 / norm0
    world = np.vstack([start[:2, 3], plan @ np.array([[c0, s0], [-s0, c0]]) + start[:2, 3]])
    times = planned_us / 1e6 + PLAN_DT * np.arange(len(world))
    targets = now_us / 1e6 + PLAN_DT * np.arange(1, len(plan) + 1)
    sampled = np.empty((len(targets), 2))
    for i, t in enumerate(targets):
        if t <= times[-1]:
            sampled[i] = [np.interp(t, times, world[:, 0]), np.interp(t, times, world[:, 1])]
        else:
            velocity = (world[-1] - world[-2]) / (times[-1] - times[-2])
            sampled[i] = world[-1] + velocity * (t - times[-1])
    now = np.asarray(ego2world, dtype=np.float64)
    c, s = now[0, 0], now[1, 0]
    norm = math.hypot(c, s) or 1.0
    c, s = c / norm, s / norm
    return (sampled - now[:2, 3]) @ np.array([[c, -s], [s, c]])


def extend_horizon(waypoints, dt: float, horizon_s: float) -> np.ndarray:
    """Continue the last segment until the plan reaches `horizon_s`: METEOR's rule (meteor/policy.py).

    Constant displacement per step from the last two waypoints, the model's speed and heading at the end of
    its plan held. Nothing inside the model's own horizon changes. An extrapolation the model never made,
    so it is opt-in (`BevPlannerPolicy.horizon_env`).
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    needed = int(round(horizon_s / dt))
    if len(waypoints) >= needed:
        return waypoints
    step = waypoints[-1] - waypoints[-2]
    return np.vstack([waypoints, waypoints[-1] + step * np.arange(1, needed - len(waypoints) + 1)[:, None]])


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


class BevPlannerPolicy(Policy):
    """What the four adapters share: settings, the history grid, the temporal reset, the frame of the plan.

    A subclass names its cameras, its two settings and how its code base builds, feeds and reads the model
    (`_build`, `_results`, `_forward`, `_reset_model`). One `infer` is one model step on the current
    frame, preceded by a step on each frame recorded since the last one that lies on the 0.5 s grid
    (`history_on_grid`): closed loop, at `planning_hz: 2`, there are none; an open-loop sequence brings its
    past this way.
    """

    waypoint_dt = PLAN_DT
    #: The environment variables naming the model's config and checkpoint (model.yaml sets both).
    config_env = ""
    checkpoint_env = ""
    #: The lidar mount used when the source has no lidar in LIDAR_TOP's axes (`model_lidar2ego`).
    nominal_lidar2ego = NUSC_LIDAR2EGO
    #: Whether a source's own lidar may stand in for the model's (only on nuScenes-trained models).
    lidar_from_observation = True
    #: The environment variable that continues the plan past its own 3 s (`extend_horizon`); unset, the plan
    #: is the model's. NAVSIM scores 4 s and refuses a shorter plan; nothing else needs it.
    horizon_env = ""

    def __init__(self) -> None:
        self.config = os.environ.get(self.config_env)
        self.checkpoint = os.environ.get(self.checkpoint_env)
        self.model: Any = None
        self.pipeline: Any = None
        raw = os.environ.get(self.horizon_env, "").strip() if self.horizon_env else ""
        self.horizon_s = float(raw) if raw else PLAN_DT * PLAN_STEPS
        if self.horizon_s < PLAN_DT * PLAN_STEPS:
            raise ValueError(
                f"{self.horizon_env}={raw} is shorter than the model's {PLAN_DT * PLAN_STEPS} s plan"
            )
        self.episode = 0
        self._begin_episode()

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        if self.model is not None:
            return
        missing = [
            env
            for env, value in ((self.config_env, self.config), (self.checkpoint_env, self.checkpoint))
            if not value
        ]
        if missing:
            raise RuntimeError(f"{type(self).__name__} needs {' and '.join(missing)} (model.yaml sets them)")
        for path in (self.config, self.checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{type(self).__name__}: no such file {path}")
        self._build()
        self.reset()

    def reset(self) -> None:
        """A new episode: the model's temporal state back to how a freshly built model has it."""
        self.episode += 1
        self._begin_episode()
        if self.model is not None:
            self._reset_model()

    def _begin_episode(self) -> None:
        #: The ``scene_token`` both code bases compare to detect a new sequence; unique per episode.
        self.scene_token = f"paddock-episode-{self.episode}"
        self._pending: list[Observation] = []
        self._last_step_us: int | None = None
        self._quaternion: np.ndarray | None = None
        self._steps = 0
        self.last: dict[str, Any] = {}
        self.plan_lidar: np.ndarray | None = None
        self.lidar2ego: np.ndarray | None = None
        #: The last step's plan in the ego frame, the pose and the time it was made from (`held_plan`).
        self._held: tuple[np.ndarray, np.ndarray, int] | None = None

    def observe(self, obs: Observation) -> None:
        """Keep a frame of history; the next `infer` steps the model on it if it lies on the 0.5 s grid."""
        self._pending.append(obs)
        if len(self._pending) > 256:
            self._pending = self._pending[-256:]

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        now = int(obs.timestamp_us)
        since = None if self._last_step_us is None else (now - self._last_step_us) / 1e6
        if self._held is not None and since is not None and 0 < since < PLAN_DT - STEP_SLACK_S:
            # Asked again before a plan step has passed (neuroNCAP's 0.1 s): hold the last plan.
            self.last["held"] = self.last.get("held", 0) + 1
            xy = held_plan(*self._held, np.asarray(obs.ego2world, dtype=np.float64), now)
            return Trajectory(extend_horizon(xy, PLAN_DT, self.horizon_s), dt=PLAN_DT)
        times = [int(o.timestamp_us) for o in self._pending]
        history = [self._pending[i] for i in history_on_grid(times, now, self._last_step_us)]
        self._pending = []
        for past in history:
            self._step(past)
        plan, lidar2ego = self._step(obs)
        self.last["frames_fed"] = len(history) + 1
        xy = lidar_plan_to_ego(plan, lidar2ego)
        self._held = (xy, np.asarray(obs.ego2world, dtype=np.float64).copy(), now)
        return Trajectory(extend_horizon(xy, PLAN_DT, self.horizon_s), dt=PLAN_DT)

    def _step(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """One model step on `obs`: (the plan in the model's lidar frame, that lidar in the ego frame)."""
        if self.lidar_from_observation:
            lidar2ego, lidar_source = model_lidar2ego(obs.lidar2ego, self.nominal_lidar2ego)
        else:
            lidar2ego, lidar_source = np.asarray(self.nominal_lidar2ego, dtype=np.float64), "nominal"
        missing = [
            camera for camera in self.cameras if camera not in obs.images and camera not in obs.image_paths
        ]
        if missing:
            raise ValueError(
                f"{type(self).__name__} reads all six surround cameras; this observation has no {missing}"
            )
        results = self._results(obs, lidar2ego)
        plan = np.asarray(self._forward(results), dtype=np.float64).reshape(-1, 2)
        self._last_step_us = int(obs.timestamp_us)
        self._steps += 1
        #: The last step's plan as the model gave it, and the lidar frame it is in (full precision).
        self.plan_lidar, self.lidar2ego = plan, lidar2ego
        self.last.update(lidar_source=lidar_source, plan_lidar=plan.round(3).tolist(), steps=self._steps)
        return plan, lidar2ego

    def _camera(self, obs: Observation, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """One camera at the config's load size, RGB, and its intrinsics."""
        image = obs.image(camera)
        self.last.setdefault("source_size", {})[camera] = [int(image.shape[1]), int(image.shape[0])]
        return to_load_size(np.ascontiguousarray(image[:, :, :3]), obs.camera2image[camera])

    def _nuscenes_quaternion(self, obs: Observation) -> np.ndarray:
        """canbus 3:7: the source's own quaternion when it is the ego pose's, else one held continuous."""
        given = source_quaternion(obs.canbus, obs.ego2world)
        q = given if given is not None else continuous_quaternion(obs.ego2world[:3, :3], self._quaternion)
        self._quaternion = q
        self.last["quaternion_source"] = "canbus" if given is not None else "ego2world"
        return q

    # -- what a code base supplies -----------------------------------------

    def _build(self) -> None:  # pragma: no cover - model environment only
        raise NotImplementedError

    def _results(self, obs: Observation, lidar2ego: np.ndarray) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def _forward(self, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover - model environment only
        raise NotImplementedError

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        raise NotImplementedError

    def diagnostics(self) -> dict[str, Any]:
        """Where this episode's inputs came from, and the last plan as the model gave it.

        ``lidar_source`` is "observation" where a source's LIDAR_TOP framed the grid and "nominal" where the
        model's training mount did; ``quaternion_source`` (nuScenes checkpoints) is "canbus" where the
        source's own quaternion was used; ``gravity_added`` says the source's acceleration had no gravity
        term; ``source_size`` is each camera as it arrived, before the resize to `LOAD_SIZE`.
        """
        if not self.last:
            return {"status": "no frame seen yet", "episode": self.episode}
        return {"episode": self.episode, **self.last}


def plugin_module(cfg) -> str | None:
    """The module an OpenMMLab config's ``plugin_dir`` names, as the repos' ``tools/test.py`` imports it."""
    if not getattr(cfg, "plugin", False) or not hasattr(cfg, "plugin_dir"):
        return None
    return ".".join(Path(cfg.plugin_dir).parts)


def without_loaders(pipeline: list, drop: tuple[str, ...], keys: list[str] | None = None) -> list:
    """A config pipeline with the steps that read the dataset removed, and its collected keys set.

    The steps dropped load images or annotations from disk; the adapter hands over the images the loader
    would have produced and stubs the evaluation-only annotations itself. ``keys`` replaces the inner
    ``CustomCollect3D``'s, as Bench2DriveZoo's own ``inference_only_pipeline`` does for the same models.
    """
    import copy

    out = []
    for step in copy.deepcopy(list(pipeline)):
        if step["type"] in drop:
            continue
        for inner in step.get("transforms", []):
            if keys is not None and inner["type"] == "CustomCollect3D":
                inner["keys"] = list(keys)
        if keys is not None and step["type"] == "CustomCollect3D":
            step["keys"] = list(keys)
        out.append(step)
    return out


def loaded_images(images: list[np.ndarray]) -> dict[str, Any]:
    """The keys the configs' multi-view loaders set, for frames already in memory.

    ``LoadMultiViewImageFromFilesInCeph`` (UniAD) and mmdet3d's ``LoadMultiViewImageFromFiles`` (VAD) both
    stack the views, set the shapes of the stack, a unit scale factor and a placeholder ``img_norm_cfg``,
    and hand the views on as a list.
    """
    stacked = np.stack(images, axis=-1)
    return {
        "filename": [f"view{i}" for i in range(len(images))],
        "img": [stacked[..., i] for i in range(stacked.shape[-1])],
        "img_shape": stacked.shape,
        "ori_shape": stacked.shape,
        "pad_shape": stacked.shape,
        "scale_factor": 1.0,
        "img_norm_cfg": {
            "mean": np.zeros(stacked.shape[2], dtype=np.float32),
            "std": np.ones(stacked.shape[2], dtype=np.float32),
            "to_rgb": False,
        },
    }


# --------------------------------------------------------------------------
# The two code bases
# --------------------------------------------------------------------------


class Mmdet3dPlannerPolicy(BevPlannerPolicy):
    """The nuScenes checkpoints, on their own repos' mmdet3d code: ``tools/test.py``, frame by frame.

    The model is built and loaded as the repos' test scripts build it (``pretrained`` and ``train_cfg``
    cleared, strict checkpoint, eval mode, one GPU) and fed by the config's own test pipeline, from
    normalisation on. Two things stand in for the dataset: the frames arrive already decoded (`loaded_images`
    sets what the file loader would), and the evaluation-only annotations each ``forward_test`` dereferences
    before it plans -- ground truth for its own metrics -- are zero stubs (`_gt_stubs`). Neither reaches the
    plan: the stubs feed IoU and L2 numbers the adapter discards.
    """

    #: Pipeline steps that read the dataset from disk, dropped (`without_loaders`).
    drop_steps: tuple[str, ...] = ()
    #: What the inner ``CustomCollect3D`` keeps: the inputs ``forward_test`` plans from.
    collect_keys: list[str] = []

    def _configure(self, cfg) -> None:
        """Adjust the config before the model is built; the default changes nothing."""

    def _build(self) -> None:  # pragma: no cover - model environment only
        import importlib

        import torch
        from mmcv import Config
        from mmcv.parallel import collate
        from mmcv.runner import load_checkpoint

        cfg = Config.fromfile(self.config)
        self._configure(cfg)
        module = plugin_module(cfg)
        if module:
            importlib.import_module(module)
        from mmdet.datasets.pipelines import Compose, to_tensor
        from mmdet3d.core.bbox import get_box_type
        from mmdet3d.models import build_model

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        # strict: a checkpoint that does not fill every weight is the wrong checkpoint, not a smaller model.
        checkpoint = load_checkpoint(model, self.checkpoint, map_location="cpu", strict=True)
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        model.cuda()
        self.model = model.eval()
        self.pipeline = Compose(without_loaders(cfg.data.test.pipeline, self.drop_steps, self.collect_keys))
        self._torch, self._collate, self._to_tensor = torch, collate, to_tensor
        self._box_type, self._box_mode = get_box_type("LiDAR")

    def _results(self, obs: Observation, lidar2ego: np.ndarray) -> dict[str, Any]:
        """What ``get_data_info``, ``pre_pipeline`` and the image loader would hand the pipeline."""
        images, projections, intrinsics, lidar2cams = [], [], [], []
        for camera in self.cameras:
            rgb, k = self._camera(obs, camera)
            # mmcv.imread decodes to BGR and the loader casts to float32 (to_float32=True in both configs).
            images.append(np.ascontiguousarray(rgb[:, :, ::-1]).astype(np.float32))
            projections.append(lidar2img(k, obs.camera2ego[camera], lidar2ego))
            pad = np.eye(4)
            pad[:3, :3] = k
            intrinsics.append(pad)
            lidar2cams.append(np.linalg.inv(obs.camera2ego[camera]) @ lidar2ego)
        can_bus, gravity = nuscenes_can_bus(obs.ego2world, obs.canbus, self._nuscenes_quaternion(obs))
        l2g_r_mat, l2g_t = nuscenes_l2g(obs.ego2world, lidar2ego)
        command = nuscenes_command(obs)
        self.last.update(command=command, gravity_added=gravity)
        results: dict[str, Any] = {
            "sample_idx": f"{self.scene_token}-{self._steps}",
            "pts_filename": "",
            "sweeps": [],
            "prev_idx": "",
            "next_idx": "",
            "scene_token": self.scene_token,
            "can_bus": can_bus,
            "frame_idx": self._steps,
            "timestamp": int(obs.timestamp_us) / 1e6,
            "l2g_r_mat": l2g_r_mat,
            "l2g_t": l2g_t,
            "img_filename": list(self.cameras),
            "lidar2img": projections,
            "cam_intrinsic": intrinsics,
            "lidar2cam": lidar2cams,
            "lidar2ego": np.asarray(lidar2ego, dtype=np.float32),
            "command": command,
            # Custom3DDataset.pre_pipeline
            "img_fields": [],
            "bbox3d_fields": [],
            "pts_mask_fields": [],
            "pts_seg_fields": [],
            "bbox_fields": [],
            "mask_fields": [],
            "seg_fields": [],
            "box_type_3d": self._box_type,
            "box_mode_3d": self._box_mode,
        }
        results.update(self._extra_results(command))
        results.update(loaded_images(images))
        return results

    def _extra_results(self, command: int) -> dict[str, Any]:
        return {}

    def _forward(self, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover - model environment only
        example = self.pipeline(results)
        # NuScenesE2EDataset.prepare_test_data turns the l2g matrices into tensors before collating.
        data = {k: (self._to_tensor(v[0]) if "l2g" in k else v) for k, v in example.items()}
        batch = self._collate([data], samples_per_gpu=1)
        batch.update(self._gt_stubs())
        with self._torch.no_grad():
            output = self.model(return_loss=False, rescale=True, **to_one_gpu(batch))
        return self._plan(output, results)

    def _gt_stubs(self) -> dict[str, Any]:  # pragma: no cover - model environment only
        raise NotImplementedError

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def to_one_gpu(batch):
    """A collated batch as ``MMDataParallel`` hands it to the model on one GPU.

    Its ``scatter`` moves tensors to the device, unwraps each ``DataContainer`` to its first GPU's share
    (moved too unless ``cpu_only``) and keeps every container's nesting. Called directly because mmcv
    1.7.2's ``Scatter`` passes an int where torch 2.x's ``_get_stream`` takes a ``torch.device``, so
    ``MMDataParallel`` fails on any batch holding a ``DataContainer`` under the torch this env needs for
    Blackwell kernels.
    """
    import torch
    from mmcv.parallel import DataContainer

    def move(obj):
        if torch.is_tensor(obj):
            return obj.cuda()
        if isinstance(obj, DataContainer):
            return obj.data[0] if obj.cpu_only else move(obj.data[0])
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(item) for item in obj)
        if isinstance(obj, dict):
            return {key: move(value) for key, value in obj.items()}
        return obj

    return move(batch)


#: Image quality Bench2Drive's dataset was written at (``tools/data_collect.py``: ``cv2.imwrite`` of
#: CARLA's BGR frames with IMWRITE_JPEG_QUALITY 20). Bench2DriveZoo's agents put every frame through the same
#: round trip before the model sees it.
B2D_JPEG_QUALITY = 20

#: Bench2DriveZoo's file loaders, dropped from ``inference_only_pipeline`` as its agents drop them.
B2D_LOADERS = ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles")


class B2DPlannerPolicy(BevPlannerPolicy):
    """The Bench2Drive checkpoints, on Bench2DriveZoo's code, fed as its CARLA agents feed them.

    ``team_code/{uniad,vad}_b2d_agent.py`` build the model (``build_model``, strict ``load_checkpoint``,
    ``.cuda().eval()``), run the config's ``inference_only_pipeline`` without its file loader, collate,
    move tensors to the GPU and call ``model(batch, return_loss=False, rescale=True)``. This does the same.
    Where the agents and the dataset the checkpoints trained on disagree, the dataset wins, and each case is
    named where it is handled: the colour order (`_results`), the heading at canbus 16 (`b2d_can_bus`).
    """

    cameras = B2D_CAMERAS
    nominal_lidar2ego = B2D_LIDAR2EGO
    #: The lidar is Bench2DriveZoo's virtual one, wherever the source's is.
    lidar_from_observation = False

    def _build(self) -> None:  # pragma: no cover - model environment only
        import mmcv
        import torch

        # Bench2DriveZoo ships its own `mmcv` (mmcv + mmdet + mmdet3d 0.17.1 in one package), built in place,
        # and it must shadow the env's mmcv-full: model.yaml puts the checkout first on PYTHONPATH.
        vendored = Path(mmcv.__file__).resolve().parent.parent
        if not (vendored / "adzoo").is_dir():
            raise RuntimeError(
                f"imported mmcv from {mmcv.__file__}, not Bench2DriveZoo's vendored package: the "
                f"Bench2DriveZoo checkout must come first on PYTHONPATH (model.yaml runtime.pythonpath)"
            )
        from mmcv import Config
        from mmcv.core.bbox import get_box_type
        from mmcv.datasets.pipelines import Compose
        from mmcv.models import build_model
        from mmcv.parallel.collate import collate
        from mmcv.utils import load_checkpoint

        cfg = Config.fromfile(self.config)
        anchors = cfg.model.get("motion_head", {}).get("anchor_info_path")
        if anchors and not Path(anchors).is_file():
            raise FileNotFoundError(
                f"{anchors} not found from {os.getcwd()}: Bench2DriveZoo's config names it relative to the "
                f"checkout, so the node must start there (model.yaml runtime.cwd)"
            )
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, self.checkpoint, map_location="cpu", strict=True)
        model.cuda()
        model.eval()
        self.model = model
        self.pipeline = Compose([p for p in cfg.inference_only_pipeline if p["type"] not in B2D_LOADERS])
        self._torch, self._collate = torch, collate
        self._box_type, _ = get_box_type("LiDAR")

    def _results(self, obs: Observation, lidar2ego: np.ndarray) -> dict[str, Any]:
        """The ``results`` dict of the agents' ``run_step``, from an observation instead of CARLA's sensors.

        Colour: the dataset's frames were CARLA's BGR written by ``cv2.imwrite`` at JPEG quality 20 and read
        back by ``mmcv.imread`` -- BGR -- and the configs normalise accordingly (UniAD's caffe mean expects
        BGR; VAD's ``to_rgb=True`` converts from it). Bench2DriveZoo's agents convert CARLA's frames to RGB
        before the same JPEG round trip, so they hand the model red and blue swapped. Here each frame goes
        through the dataset's own round trip, in BGR.
        """
        import cv2

        images, projections, lidar2cams = [], [], []
        for camera in self.cameras:
            rgb, k = self._camera(obs, camera)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), B2D_JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed for {camera}")
            images.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
            projections.append(lidar2img(k, obs.camera2ego[camera], lidar2ego))
            lidar2cams.append(np.linalg.inv(obs.camera2ego[camera]) @ lidar2ego)
        can_bus, gravity = b2d_can_bus(obs.ego2world, obs.canbus)
        l2g_r_mat, l2g_t = b2d_l2g(obs.ego2world)
        command = b2d_command(obs)
        self.last.update(command=command, gravity_added=gravity)
        stacked_shape = np.stack(images, axis=-1).shape
        results: dict[str, Any] = {
            "lidar2img": np.stack(projections, axis=0),
            "lidar2cam": np.stack(lidar2cams, axis=0),
            "img": images,
            "folder": " ",
            "scene_token": self.scene_token,
            "frame_idx": 0,
            "timestamp": int(obs.timestamp_us) / 1e6,
            "box_type_3d": self._box_type,
            "can_bus": can_bus,
            "command": command,
            "l2g_r_mat": l2g_r_mat,
            "l2g_t": l2g_t,
            "img_shape": stacked_shape,
            "ori_shape": stacked_shape,
            "pad_shape": stacked_shape,
        }
        results.update(self._extra_results(command))
        return results

    def _extra_results(self, command: int) -> dict[str, Any]:
        return {}

    def _forward(self, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover - model environment only
        torch = self._torch
        example = self.pipeline(results)
        batch = self._collate([example], samples_per_gpu=1)
        for key, data in batch.items():
            if key != "img_metas" and torch.is_tensor(data[0]):
                data[0] = data[0].to("cuda")
        with torch.no_grad():
            output = self.model(batch, return_loss=False, rescale=True)
        return self._plan(output, results)

    def _plan(self, output, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
