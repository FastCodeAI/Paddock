"""Inputs and outputs of the BEVFormer-based planners UniAD and VAD (their nuScenes checkpoints).

The two models share one shape. Six surround cameras are lifted into a bird's-eye-view grid laid out in a
*lidar* frame (x right, y forward, z up); the previous grid is warped onto the current one with the ego
motion read from an 18-float CAN-bus vector; and the plan is six (x, y) waypoints 0.5 s apart in that same
lidar frame. Both run on OpenMMLab's mmdet3d, and three conventions of their datasets are each a silent
failure when crossed:

* **The CAN-bus layout.** nuScenes puts acceleration at 7:10, rotation rate at 10:13 and velocity at 13:16
  (the order of the CAN-bus expansion's ``pose`` message), translation at 0:3 and the heading, in radians
  then degrees, at 16:18; at 3:7 the datasets hold the ego quaternion's w four times (`nuscenes_can_bus`).
  Paddock's own canbus is the nuScenes layout.
* **``l2g_r_mat``.** The nuScenes datasets store the *transpose* of the lidar-to-global rotation (their
  tracker multiplies row vectors).
* **The lidar.** Every model's grid and plan live in the lidar frame it was trained with. nuScenes'
  LIDAR_TOP sits ~0.94-0.99 m ahead of the rear axle and 1.84 m up, yawed -90 degrees. Paddock's
  closed-loop clients send ``lidar2ego`` as the identity ("lidar = ego"), which is the wrong axes for these
  models -- a grid turned a quarter turn, as Drive-JEPA's adapter once found -- so a lidar is used only where
  the source sends one in LIDAR_TOP's axes, and the model's training mount is used everywhere else
  (`model_lidar2ego`).

The plan comes back as the *lidar's* future positions in the current lidar frame -- both code bases build
the planning target from the lidar origin's future poses (UniAD ``get_sdc_planning_label``, VAD
``vad_nuscenes_converter``) -- so `lidar_plan_to_ego` turns it into the ego origin's path, lever arm
included (``paddock.policy.protocol.Trajectory`` is the ego origin's).

Everything in this module above `BevPlannerPolicy` is numpy only, so paddock's own tests check it without
the models. The policy itself imports the model's code base inside `load()`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    Command,
    Observation,
    Policy,
    Trajectory,
)

#: Spacing and count of the plan's waypoints: UniAD ``planning_steps = 6``, VAD ``fut_ts = 6``, both 0.5 s
#: apart (nuScenes keyframes).
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

#: (width, height) every config loads its cameras at: nuScenes' 1600x900, which Bench2Drive's rig
#: reproduces. Neither test pipeline resizes before normalising (VAD scales by 0.8 after), so a source at
#: another size is brought to this one first, intrinsics with it (`to_load_size`).
LOAD_SIZE = (1600, 900)

#: Standard gravity, m/s^2: the vertical term of an accelerometer reading at rest.
GRAVITY = 9.81

#: Threads a node may use beside the one that runs the model, to read and prepare frames: one prepares the
#: next frame (its six images, the config's test pipeline, the collated batch) while the model runs on this
#: one, the rest decode and convert the six cameras side by side, and OpenCV's own pool is held to as many.
#: Speed only: every frame is prepared in order and fed what it would be fed one at a time, so the plans
#: are the same at any count (the golden tests, `make check-uniad-vad`). 1 or less prepares on the model's
#: thread, one camera after another. model.yaml declares it `tuning_env`.
THREADS_ENV = "UNIAD_VAD_THREADS"
DEFAULT_THREADS = 5


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

#: nuScenes' six cameras on the same log as `NUSC_LIDAR2EGO` (scene-0103's calibrated_sensor records, the
#: golden fixtures' calibration): each camera's (intrinsics at 1600x900, camera2ego in optical axes). A slot a
#: source has no camera for (comma2k19 sends its forward camera alone) is fed black (`_absent`), and the
#: spatial cross-attention still projects every BEV point through that slot's lidar2img, so the slot keeps
#: the geometry the checkpoints trained with. test_bev_planners checks every number against
#: golden_uniad.json.
NUSC_RIG: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "CAM_FRONT": (
        np.array(
            [
                [1252.8131021185304, 0.0, 826.588114781398],
                [0.0, 1252.8131021185304, 469.9846626224581],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.010260207775400836, 0.00843344807166727, 0.9999117986552754, 1.72200568478],
                [-0.9998725753702894, 0.012316255772487425, 0.010155927635204, 0.00475453292289],
                [-0.0122295199738352, -0.9998885871922777, 0.008558740785910231, 1.49491291905],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
    "CAM_FRONT_RIGHT": (
        np.array(
            [
                [1256.7485116440405, 0.0, 817.7887570959712],
                [0.0, 1256.7485116440403, 451.9541780095127],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [-0.8439797263539992, 0.016455514361927055, 0.5361225956350965, 1.58082565783],
                [-0.5361413772261798, 0.0036210747121631953, -0.8441204365752224, -0.499078711449],
                [-0.015831775940933233, -0.9998580418564496, 0.005766368488289917, 1.51749368405],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
    "CAM_FRONT_LEFT": (
        np.array(
            [
                [1257.8625342125129, 0.0, 827.2410631095686],
                [0.0, 1257.8625342125129, 450.915498205774],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.8225460431338631, 0.0064783206158985016, 0.5686616201985344, 1.5752559464],
                [-0.5686843280499335, 0.01643406515767573, 0.8223916685696594, 0.500519383135],
                [-0.004017705218174177, -0.999843964258619, 0.01720189467439451, 1.50696032589],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
    "CAM_BACK": (
        np.array(
            [
                [796.8910634503094, 0.0, 857.7774326863696],
                [0.0, 796.8910634503094, 476.8848988407415],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.009215410429862864, -0.006800497022728537, -0.9999344125745713, 0.05524611077],
                [0.9998947902096122, 0.011264881076691956, 0.009138433454562552, 0.0107882366898],
                [0.011201996352644218, -0.9999134240995926, 0.006903592046427567, 1.56794286957],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
    "CAM_BACK_LEFT": (
        np.array(
            [
                [1254.9860565800168, 0.0, 829.5769333630991],
                [0.0, 1254.9860565800168, 467.1680561863987],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [0.9478787070617002, -0.008923243758269323, -0.31850640875792396, 1.04852047718],
                [0.31862179229696586, 0.01878971058568336, 0.9476956791341696, 0.483058131052],
                [-0.0024718763133260322, -0.9997836378421765, 0.020653506495383587, 1.56210154484],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
    "CAM_BACK_RIGHT": (
        np.array(
            [
                [1249.9629280788233, 0.0, 825.3768045375984],
                [0.0, 1249.9629280788233, 462.54816385708756],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [-0.9237361237050898, -0.0026030300067295302, -0.38302062346181753, 1.05945173053],
                [0.383024352207656, -0.011446056573421563, -0.923667328319468, -0.46720294852],
                [-0.0019797419530799, -0.9999311036384969, 0.0115701598129463, 1.55050857555],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ),
}

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


def with_blank_views(obs: Observation, cameras: list[str]) -> Observation:
    """`obs` with each of `cameras` black, at `LOAD_SIZE` and nuScenes' own calibration of it (`NUSC_RIG`).

    A new observation: the source's own tables are not written to, and each slot gets its own zeros, so a
    pipeline that writes in place cannot reach another. Diagnostics list the slots as ``blank_views``, so a
    run records which of its views were real.
    """
    images, camera2image, camera2ego = dict(obs.images), dict(obs.camera2image), dict(obs.camera2ego)
    for camera in cameras:
        intrinsics, pose = NUSC_RIG[camera]
        images[camera] = np.zeros((LOAD_SIZE[1], LOAD_SIZE[0], 3), dtype=np.uint8)
        camera2image[camera], camera2ego[camera] = intrinsics.copy(), pose.copy()
    return dataclasses.replace(obs, images=images, camera2image=camera2image, camera2ego=camera2ego)


def lidar2img(camera2image, camera2ego, lidar2ego) -> np.ndarray:
    """4x4 lidar -> pixel projection: the padded intrinsics times lidar -> camera.

    The nuScenes datasets build it from ``sensor2lidar``, which is ``K @ inv(camera2ego) @ lidar2ego`` once
    the camera's pose and the lidar's are in one ego frame. Paddock's camera frame is the optical one
    (x right, y down, z forward), the same as the datasets'.
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

    The training set holds specific force, gravity included: nuScenes' CAN-bus ``accel`` reads z ~ +9.6 to
    9.8. Most paddock sources send the kinematic
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

    q and -q are one rotation, and nuScenes stores either (w < 0 on 41.7% of the val keyframes' ego poses),
    but the CAN-bus MLP reads numbers, not the rotation -- w, for the nuScenes checkpoints
    (`nuscenes_can_bus`). A pose source gives a smooth quaternion within a log, so the sign is held continuous
    across an episode rather than flipped wherever w crosses zero. The nuScenes harness sends these models
    the stored one (`source_quaternion`), so this is for every other source.
    """
    q = matrix_quaternion(rotation)
    if previous is not None and float(q @ np.asarray(previous, dtype=np.float64)) < 0:
        q = -q
    return q


def source_quaternion(canbus, rotation) -> np.ndarray | None:
    """The source's own quaternion at canbus 3:7, if it is a unit quaternion of exactly `rotation`, else None.

    A source that fills the slot the way the nuScenes info files do (the ego pose's quaternion) keeps its
    sign: paddock's nuScenes harness does, for the models that declare ``inputs: {nuscenes: ...}``
    (`benchmarks/nuscenes/infos`). HUGSIM and Bench2Drive leave it zero, and the CAN-bus message's own
    orientation, which nuScenes sends every other model there, is not the ego pose's.
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
    is fed that.
    """
    pose = np.asarray(ego2world, dtype=np.float64)
    source = np.asarray(canbus, dtype=np.float64)
    out = np.zeros(18)
    out[0:3] = pose[:3, 3]
    out[3:7] = float(np.asarray(quaternion, dtype=np.float64)[0])
    # The checkpoint's slots are fixed by training, so they are literals; paddock's are read by its constants.
    # The two layouts coincide today, which is why this is a copy and not a re-order.
    out[7:10], added = accel_with_gravity(source[CANBUS_ACCEL : CANBUS_ACCEL + 3])
    out[10:13] = source[CANBUS_YAW_RATE - 2 : CANBUS_YAW_RATE + 1]  # roll, pitch, yaw rate
    out[13:16] = source[CANBUS_VELOCITY : CANBUS_VELOCITY + 3]
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


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

#: paddock Command -> the nuScenes models' index. UniAD's ``get_sdc_planning_label`` and VAD's converter
#: both derive it from the 3 s pose's lateral offset in the lidar frame (x right): x >= 2 m -> 0 "RIGHT",
#: x <= -2 m -> 1 "LEFT", else 2 "FORWARD"/"Go Straight". UniAD embeds it (``navi_embed``), VAD selects one
#: of its three plans with it. FOLLOW_REFERENCE is "keep going", which is FORWARD in that vocabulary --
#: unless it comes with a route whose frame time is stated, when the route decides (`command_record`).
NUSC_COMMAND = {
    Command.RIGHT: 0,
    Command.LEFT: 1,
    Command.STRAIGHT: 2,
    Command.FOLLOW_REFERENCE: 2,
}
NUSC_RIGHT, NUSC_LEFT, NUSC_FORWARD = 0, 1, 2
NUSC_COMMAND_NAMES = ("RIGHT", "LEFT", "FORWARD")

#: The label rule `route_command` reproduces: the lidar's position at the last of the six future keyframes
#: (3 s; the last one the scene has if it ends sooner), in the current lidar frame, x >= +2 m RIGHT,
#: x <= -2 m LEFT, else FORWARD. UniAD ``data_utils/trajectory_api.py`` ``get_sdc_planning_label`` (the
#: pseudo ego box at each future lidar origin, moved into the initial lidar frame; ``planning_steps = 6``);
#: VAD ``tools/data_converter/vad_nuscenes_converter.py`` ("drive command according to final fut step
#: offset from lcf"; the lidar's global pose at each of ``fut_ts = 6`` keyframes, in the current lidar frame).
COMMAND_HORIZON_S = PLAN_DT * PLAN_STEPS
COMMAND_LATERAL_M = 2.0
#: The shortest distance along a route at which `route_command` takes the 3 s position: what a car pulling
#: away from rest at a moderate 1.5 m/s^2 covers in 3 s (0.5 * 1.5 * 3^2). Speed times 3 s alone shrinks to
#: nothing at a standstill, so a car waiting at a junction would be told FORWARD until it had rolled into the
#: turn; the label code tells a car that pulls away into a turn within 3 s that turn. Short enough that a car
#: stopped on a straight lane is still told FORWARD.
COMMAND_MIN_LOOKAHEAD_M = 6.75


def route_command(route_xy, speed_m_s: float, lidar2ego) -> dict[str, Any] | None:
    """The nuScenes command for a route, by the datasets' own rule (`COMMAND_HORIZON_S`), and what decided it.

    The route -- ego frame at the observation's time, x forward, y left; AlpaSim pads it with NaN -- stands
    in for the future the label code reads. The car is placed on it at the route's nearest point, and its
    3 s position is the point `lookahead_m` further along: what it covers in 3 s at its current speed, at
    least `COMMAND_MIN_LOOKAHEAD_M`, the route's last point if the route ends sooner. The lidar is placed
    there facing along the route, its lever arm swung with it, and read in the current lidar frame as the
    label code reads it. None when fewer than two finite points remain.

    The record: ``command`` (the index) and its ``name``; ``lateral_m``, the 3 s lidar position's x in the
    current lidar frame (+ right; the number the thresholds apply to); ``lookahead_m``, ``speed_m_s``;
    ``route_offset_m``, how far the car is from the route; ``route_end``, whether the route ran out first.
    """
    points = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 1:
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    if len(points) < 2:
        return None
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    starts = np.r_[0.0, np.cumsum(lengths)[:-1]]
    # The car on the route: each segment's nearest point to the ego origin, then the nearest of those.
    along = np.clip(-np.einsum("ij,ij->i", points[:-1], segments) / lengths**2, 0.0, 1.0)
    distance = np.linalg.norm(points[:-1] + along[:, None] * segments, axis=1)
    nearest = int(np.argmin(distance))
    speed = max(float(speed_m_s), 0.0)
    lookahead = max(speed * COMMAND_HORIZON_S, COMMAND_MIN_LOOKAHEAD_M)
    target = starts[nearest] + along[nearest] * lengths[nearest] + lookahead
    at_end = bool(target >= starts[-1] + lengths[-1])
    index = len(segments) - 1 if at_end else int(np.searchsorted(starts, target, side="right")) - 1
    fraction = 1.0 if at_end else (target - starts[index]) / lengths[index]
    position = points[index] + fraction * segments[index]
    heading = math.atan2(segments[index][1], segments[index][0])
    lidar = np.asarray(lidar2ego, dtype=np.float64)
    arm = lidar[:3, 3]
    c, s = math.cos(heading), math.sin(heading)
    future = np.array(
        [position[0] + c * arm[0] - s * arm[1], position[1] + s * arm[0] + c * arm[1], arm[2], 1.0]
    )
    lateral = float((np.linalg.inv(lidar) @ future)[0])
    if lateral >= COMMAND_LATERAL_M:
        command = NUSC_RIGHT
    elif lateral <= -COMMAND_LATERAL_M:
        command = NUSC_LEFT
    else:
        command = NUSC_FORWARD
    return {
        "source": "route",
        "command": command,
        "name": NUSC_COMMAND_NAMES[command],
        "lateral_m": round(lateral, 3),
        "lookahead_m": round(lookahead, 3),
        "speed_m_s": round(speed, 3),
        "route_offset_m": round(float(distance[nearest]), 3),
        "route_end": at_end,
    }


def command_record(obs: Observation, lidar2ego=NUSC_LIDAR2EGO) -> dict[str, Any]:
    """The nuScenes command for this observation and where it came from (``source``: "given" or "route").

    The source's command, mapped by `NUSC_COMMAND`, except FOLLOW_REFERENCE with a route whose frame time is
    stated (``route_timestamp_us``; AlpaSim's bridge with ``--send-route-time``, the only source that sends
    one): there the route decides (`route_command`), as the datasets would have labelled where it goes.
    Bench2Drive's client also sends FOLLOW_REFERENCE with a route, without a time, and keeps FORWARD. The
    route must be in the observation's own frame: one stamped at another time is refused rather than read
    in the wrong frame.
    """
    given = Command(int(obs.command))
    if given == Command.FOLLOW_REFERENCE and obs.route_xy is not None and obs.route_timestamp_us is not None:
        if int(obs.route_timestamp_us) != int(obs.timestamp_us):
            raise ValueError(
                f"route_xy is in the ego frame at {int(obs.route_timestamp_us)} us, the observation's pose "
                f"at {int(obs.timestamp_us)} us; a route command needs both in one frame"
            )
        record = route_command(obs.route_xy, obs.speed, lidar2ego)
        if record is not None:
            return {**record, "given": given.name}
    command = NUSC_COMMAND[given]
    return {"source": "given", "command": command, "name": NUSC_COMMAND_NAMES[command], "given": given.name}


def nuscenes_command(obs: Observation, lidar2ego=NUSC_LIDAR2EGO) -> int:
    return int(command_record(obs, lidar2ego)["command"])


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
    """What the two adapters share: settings, the history grid, the temporal reset, the frame of the plan.

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
        self.threads = 1
        self._prepare_pool: ThreadPoolExecutor | None = None
        self._views_pool: ThreadPoolExecutor | None = None
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
        self._start_threads()
        self.reset()

    def _start_threads(self) -> None:
        """The preparing thread and the camera threads (`THREADS_ENV`); OpenCV's pool held to the latter."""
        raw = os.environ.get(THREADS_ENV, "").strip()
        threads = int(raw) if raw else DEFAULT_THREADS
        self.threads = max(1, threads)
        with contextlib.suppress(
            ImportError
        ):  # cv2 is in every model env; paddock's own tests run without it
            import cv2

            cv2.setNumThreads(max(1, self.threads - 1))
        if self.threads > 1:
            self._prepare_pool = ThreadPoolExecutor(1, thread_name_prefix="bev-prepare")
            self._views_pool = ThreadPoolExecutor(self.threads - 1, thread_name_prefix="bev-views")

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
        #: Frames prepared this episode (`_prepare_frame`): the frame index a frame's inputs carry. Equal to
        #: `_steps` one frame at a time; ahead of it by one while the next frame is prepared during a step.
        self._prepared = 0
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
        plan, lidar2ego = self._step_through([*history, obs])[-1]
        self.last["frames_fed"] = len(history) + 1
        xy = lidar_plan_to_ego(plan, lidar2ego)
        self._held = (xy, np.asarray(obs.ego2world, dtype=np.float64).copy(), now)
        # A command the route decided travels with the plan, so a run's per-replan log records it; with any
        # other command the answer is what it always was.
        record = self.last.get("command_record")
        extras = {"route_command": record} if record is not None and record["source"] == "route" else None
        return Trajectory(extend_horizon(xy, PLAN_DT, self.horizon_s), dt=PLAN_DT, extras=extras)

    def infer_each(self, observations: list[Observation]) -> list[Trajectory]:
        """Scene order: one reset, then a model step on every frame, as the repos' test loops step.

        `infer` answers a call less than a plan step after the last one from the held plan, for clients
        that ask faster than 2 Hz. nuScenes keyframes are 0.40-0.65 s apart -- 462 of val's 5,869 intervals
        are 0.40 or 0.45 s -- and UniAD's and VAD's own test loops step on every one, so here no frame is
        held, and the next frame is prepared while the model runs on this one (`_step_through`).
        """
        if not observations:
            raise ValueError("infer_each needs at least one observation")
        previous, self.self_prime = self.self_prime, False
        try:
            self.reset()
            out = []
            steps = self._step_through(list(observations))
            for plan, lidar2ego in steps:
                xy = lidar_plan_to_ego(plan, lidar2ego)
                out.append(Trajectory(extend_horizon(xy, PLAN_DT, self.horizon_s), dt=PLAN_DT))
            last = observations[-1]
            self._held = (xy, np.asarray(last.ego2world, dtype=np.float64).copy(), int(last.timestamp_us))
            self.last["frames_fed"] = 1
            return out
        finally:
            self.self_prime = previous

    def _step(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """One model step on `obs`: (the plan in the model's lidar frame, that lidar in the ego frame)."""
        return self._step_through([obs])[-1]

    def _step_through(self, frames: list[Observation]) -> list[tuple[np.ndarray, np.ndarray]]:
        """A model step on each frame in turn; `_step`'s pair for each.

        The model runs on this thread, frame after frame. With threads to spare (`THREADS_ENV`) the next
        frame is prepared on another while the model runs on this one: its images read and decoded, put
        through the config's test pipeline and collated, all CPU work the model's step does not wait on.
        Preparing happens strictly in frame order, one frame ahead, so each frame's inputs -- its frame
        index and the quaternion's sign branch among them -- are what they would be one at a time, and
        nothing is still being prepared when this returns, so a `reset` never meets a frame in flight.
        """
        if self._prepare_pool is None or len(frames) == 1:
            return [self._run_prepared(obs, self._prepare_frame(obs)) for obs in frames]
        out = []
        pending: Future | None = self._prepare_pool.submit(self._prepare_frame, frames[0])
        try:
            for k, obs in enumerate(frames):
                prepared = pending.result()
                pending = (
                    self._prepare_pool.submit(self._prepare_frame, frames[k + 1])
                    if k + 1 < len(frames)
                    else None
                )
                out.append(self._run_prepared(obs, prepared))
        finally:
            if pending is not None:
                with contextlib.suppress(Exception):  # the step failed; its error is the one that matters
                    pending.result()
        return out

    def _prepare_frame(self, obs: Observation) -> tuple[np.ndarray, str, dict[str, Any], Any]:
        """All a step needs before the model: (lidar2ego, where it came from, results, prepared input)."""
        lidar2ego, lidar_source = model_lidar2ego(obs.lidar2ego, self.nominal_lidar2ego)
        missing = [
            camera for camera in self.cameras if camera not in obs.images and camera not in obs.image_paths
        ]
        absent = [camera for camera in missing if self._absent(obs, camera)]
        if len(absent) < len(missing):
            raise ValueError(
                f"{type(self).__name__} reads all six surround cameras; this observation has no {missing}. "
                "A slot is left black only when the source describes CAM_FRONT and sends neither an image "
                "nor a calibration for that camera"
            )
        if absent:
            obs = with_blank_views(obs, absent)
        self.last["blank_views"] = absent
        results = self._results(obs, lidar2ego)
        self._prepared += 1
        return lidar2ego, lidar_source, results, self._prepare(results)

    def _run_prepared(self, obs: Observation, prepared) -> tuple[np.ndarray, np.ndarray]:
        lidar2ego, lidar_source, results, batch = prepared
        plan = np.asarray(self._run(batch, results), dtype=np.float64).reshape(-1, 2)
        self._last_step_us = int(obs.timestamp_us)
        self._steps += 1
        #: The last step's plan as the model gave it, and the lidar frame it is in (full precision).
        self.plan_lidar, self.lidar2ego = plan, lidar2ego
        self.last.update(lidar_source=lidar_source, plan_lidar=plan.round(3).tolist(), steps=self._steps)
        return plan, lidar2ego

    @staticmethod
    def _absent(obs: Observation, camera: str) -> bool:
        """Is `camera` one this source does not have, as opposed to one that went missing?

        Qwen-Drive's rule (its policy's ``_absent``): only a source that describes its cameras can say one
        is absent. comma2k19 calibrates its forward camera and sends nothing at all for the other five, so a
        camera counts as absent only when CAM_FRONT has an image and a calibration and this camera has
        neither. A source that sends a calibration without its image has lost a frame, and is refused.
        """
        front = "CAM_FRONT"
        if camera == front:
            return False
        front_described = (
            (front in obs.images or front in obs.image_paths)
            and front in obs.camera2ego
            and front in obs.camera2image
        )
        sent_anything = any(camera in table for table in (obs.camera2ego, obs.camera2image))
        return front_described and not sent_anything

    def _camera(self, obs: Observation, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """One camera at the config's load size, RGB, and its intrinsics."""
        image = obs.image(camera)
        self.last.setdefault("source_size", {})[camera] = [int(image.shape[1]), int(image.shape[0])]
        return to_load_size(np.ascontiguousarray(image[:, :, :3]), obs.camera2image[camera])

    def _views(self, obs: Observation, convert: Callable[[np.ndarray, np.ndarray], Any]) -> list[Any]:
        """`convert(rgb, intrinsics)` of each camera (`_camera`), in the model's camera order.

        On the camera threads when there are any: decoding a 1600x900 JPEG and casting it to float32 are
        each done by one camera's thread with the GIL released (Pillow's decoder, numpy's copies), so six
        take about as long as one. What each thread computes is what the loop computed.
        """
        if self._views_pool is None:
            return [convert(*self._camera(obs, camera)) for camera in self.cameras]
        return list(self._views_pool.map(lambda camera: convert(*self._camera(obs, camera)), self.cameras))

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
        """One frame's results to the plan as the model gives it."""
        raise NotImplementedError

    def _prepare(self, results: dict[str, Any]) -> Any:
        """The CPU half of a step, what the model is then called with (`_run`); by default `results` as is.

        A code base splits its `_forward` here so that the next frame can be prepared while the model runs
        (`_step_through`).
        """
        return results

    def _run(self, prepared: Any, results: dict[str, Any]) -> np.ndarray:
        """The model on `_prepare`'s output. By default `_forward`, for a code base that does not split it."""
        return self._forward(prepared)

    def _reset_model(self) -> None:  # pragma: no cover - model environment only
        raise NotImplementedError

    def diagnostics(self) -> dict[str, Any]:
        """Where this episode's inputs came from, and the last plan as the model gave it.

        ``lidar_source`` is "observation" where a source's LIDAR_TOP framed the grid and "nominal" where the
        model's training mount did; ``quaternion_source`` (nuScenes checkpoints) is "canbus" where the
        source's own quaternion was used; ``gravity_added`` says the source's acceleration had no gravity
        term; ``source_size`` is each camera as it arrived, before the resize to `LOAD_SIZE`; ``blank_views``
        the slots the source has no camera for, fed black (`_absent`);
        ``command_record`` (nuScenes checkpoints) is the last step's command and what decided it
        (`command_record`), also sent with the plan as ``extras["route_command"]`` when a route decided it.
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
    ``CustomCollect3D``'s.
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
        # mmcv.imread decodes to BGR and the loader casts to float32 (to_float32=True in both configs).
        views = self._views(obs, lambda rgb, k: (np.ascontiguousarray(rgb[:, :, ::-1]).astype(np.float32), k))
        for camera, (image, k) in zip(self.cameras, views):  # noqa: B905 - runs on Python 3.9, no strict=
            images.append(image)
            projections.append(lidar2img(k, obs.camera2ego[camera], lidar2ego))
            pad = np.eye(4)
            pad[:3, :3] = k
            intrinsics.append(pad)
            lidar2cams.append(np.linalg.inv(obs.camera2ego[camera]) @ lidar2ego)
        can_bus, gravity = nuscenes_can_bus(obs.ego2world, obs.canbus, self._nuscenes_quaternion(obs))
        l2g_r_mat, l2g_t = nuscenes_l2g(obs.ego2world, lidar2ego)
        record = command_record(obs, lidar2ego)
        command = record["command"]
        self.last.update(command=command, command_record=record, gravity_added=gravity)
        results: dict[str, Any] = {
            "sample_idx": f"{self.scene_token}-{self._prepared}",
            "pts_filename": "",
            "sweeps": [],
            "prev_idx": "",
            "next_idx": "",
            "scene_token": self.scene_token,
            "can_bus": can_bus,
            "frame_idx": self._prepared,
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

    def _prepare(self, results: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - model env only
        """The config's test pipeline, ``prepare_test_data`` and collate: the batch, still on the CPU."""
        example = self.pipeline(results)
        # NuScenesE2EDataset.prepare_test_data turns the l2g matrices into tensors before collating.
        data = {k: (self._to_tensor(v[0]) if "l2g" in k else v) for k, v in example.items()}
        batch = self._collate([data], samples_per_gpu=1)
        batch.update(self._gt_stubs())
        return batch

    def _run(self, batch: dict[str, Any], results: dict[str, Any]) -> np.ndarray:  # pragma: no cover
        with self._torch.no_grad():
            output = self.model(return_loss=False, rescale=True, **to_one_gpu(batch))
        return self._plan(output, results)

    def _forward(self, results: dict[str, Any]) -> np.ndarray:  # pragma: no cover - model environment only
        return self._run(self._prepare(results), results)

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
