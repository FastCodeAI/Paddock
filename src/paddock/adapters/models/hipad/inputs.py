"""Paddock observations as HiP-AD's inputs, in numpy, so paddock's own tests check them without the model.

HiP-AD's own CARLA agent (``bench2drive/leaderboard/team_code/hipad_b2d_agent.py`` at the pinned sha) is
the reference for every input here; its training dataset (``projects/mmdet3d_plugin/datasets/
bench2drive_dataset.py``) for the frames they are in. Three of them are the model's own and are fed as it
was trained:

* **The lidar frame.** Plans, target points and the camera projections are all in Bench2Drive's LIDAR_TOP
  frame: x right, y forward, z up, 0.39 m behind the ego origin and 1.84 m up (`LIDAR2EGO`, the agent's
  ``LIDAR2EGO``). The model trained only in that mount, so it is used whatever lidar a source sends.
* **The command**, one of Bench2Drive's six road options (`command_index`).
* **The target point**, the next point of the route the car is driving, in the lidar's axes and measured
  from the ego origin (`target_point`).

The fourth, where a source has no route, is paddock's (`command_target_point`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from paddock.policy.protocol import Command

#: HiP-AD's camera order (the agent's ``CAMERA``). Its deformable attention projects into each view by
#: index, so the order is part of the model.
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

#: Bench2Drive's LIDAR_TOP in the ego frame (the agent's ``LIDAR2EGO``): x right, y forward, z up.
LIDAR2EGO = np.array(
    [
        [0.0, 1.0, 0.0, -0.39],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.84],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

#: The agent JPEG-encodes every camera at this quality and decodes it again before the model sees it
#: (``tick``: ``cv2.IMWRITE_JPEG_QUALITY, 20``), to match the compressed frames of the training set.
JPEG_QUALITY = 20

#: The agent's route planner, ``RoutePlanner(4.0, 50.0)``: a route point within 4 m of the car is passed,
#: and the search for one stops 50 m along the route.
ROUTE_MIN_DISTANCE_M = 4.0
ROUTE_MAX_DISTANCE_M = 50.0
#: A held route point found again in a later tick's route within this distance is the same point.
SAME_POINT_M = 1.0

#: Bench2Drive's road options as HiP-AD indexes them: the agent maps CARLA's ``RoadOption`` (LEFT 1, RIGHT
#: 2, STRAIGHT 3, LANEFOLLOW 4, CHANGELANELEFT 5, CHANGELANERIGHT 6; VOID -1 read as 4) to ``option - 1``
#: and one-hot encodes it over six.
LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT = range(6)
COMMAND_NAMES = ("LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW", "CHANGELANELEFT", "CHANGELANERIGHT")
N_COMMANDS = 6
#: Paddock's command -> HiP-AD's road option. Paddock's Bench2Drive client sends a lane change as STRAIGHT
#: (its ``CMD``); `command_index` recovers it from ``nav_lane_change``.
FROM_PADDOCK = {
    Command.LEFT: LEFT,
    Command.RIGHT: RIGHT,
    Command.STRAIGHT: STRAIGHT,
    Command.FOLLOW_REFERENCE: LANEFOLLOW,
}
#: A lane change this close along the route is the one being driven: the agent's planner holds a route
#: point's option once the car is within `ROUTE_MIN_DISTANCE_M` of it.
LANE_CHANGE_NEAR_M = ROUTE_MIN_DISTANCE_M

#: Where a source sends no route, paddock's stand-in for the route point: `COMMAND_TARGET_M` ahead, turned
#: `COMMAND_TURN_DEG` towards a turn, or moved one lane (`LANE_WIDTH_M`) aside for a lane change. Paddock's
#: choice, not the model's: HiP-AD trained only with CARLA's route, whose next point lies 4-50 m ahead.
COMMAND_TARGET_M = 20.0
COMMAND_TURN_DEG = 45.0
LANE_WIDTH_M = 3.5

#: HiP-AD's image pipeline (the config's ``data_aug_conf`` and ``input_shape``): frames loaded at 1600x900,
#: scaled to cover 640x352 and cut from the bottom (``bot_pct_lim`` 0).
LOAD_SIZE = (1600, 900)
FINAL_SIZE = (640, 352)


def command_index(obs_command: int, lane_change: str | None, lane_change_m: float | None) -> int:
    """HiP-AD's road option for a paddock command, a lane change within reach read as the change."""
    command = Command(int(obs_command))
    if command in (Command.STRAIGHT, Command.FOLLOW_REFERENCE) and lane_change_m is not None:
        if lane_change in ("LEFT", "RIGHT") and float(lane_change_m) <= LANE_CHANGE_NEAR_M:
            return CHANGELANELEFT if lane_change == "LEFT" else CHANGELANERIGHT
    return FROM_PADDOCK[command]


def one_hot(index: int) -> np.ndarray:
    out = np.zeros(N_COMMANDS, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def ego_to_lidar_axes(xy) -> np.ndarray:
    """(x forward, y left) -> HiP-AD's (x right, y forward), same origin."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack([-xy[..., 1], xy[..., 0]], axis=-1)


def command_target_point(index: int) -> np.ndarray:
    """Paddock's stand-in target point for a source with no route, in HiP-AD's axes (`COMMAND_TARGET_M`)."""
    if index in (LEFT, RIGHT):
        angle = math.radians(COMMAND_TURN_DEG) * (1.0 if index == LEFT else -1.0)
        ego = (COMMAND_TARGET_M * math.cos(angle), COMMAND_TARGET_M * math.sin(angle))
    elif index in (CHANGELANELEFT, CHANGELANERIGHT):
        ego = (COMMAND_TARGET_M, LANE_WIDTH_M if index == CHANGELANELEFT else -LANE_WIDTH_M)
    else:
        ego = (COMMAND_TARGET_M, 0.0)
    return ego_to_lidar_axes(np.asarray(ego)).astype(np.float32)


def _ego_to_world(xy: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    pose = np.asarray(ego2world, dtype=np.float64)
    return xy @ pose[:2, :2].T + pose[:2, 3]


def _world_to_ego(xy: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    pose = np.asarray(ego2world, dtype=np.float64)
    return (xy - pose[:2, 3]) @ pose[:2, :2]


class RouteTracker:
    """The agent's ``RoutePlanner``, fed paddock's route each tick instead of holding the plan once.

    The agent keeps the whole route and pops each point the car comes within `ROUTE_MIN_DISTANCE_M` of,
    searching no further than `ROUTE_MAX_DISTANCE_M` along it, never below two points; its target point is
    then the route's second point. Paddock's Bench2Drive client sends the same downsampled route
    (``Observation.route_xy``), a window of it around the car each tick, so the point the agent would hold
    at the front is remembered in the world and found again in the next window.
    """

    def __init__(self) -> None:
        self.front: np.ndarray | None = None

    def reset(self) -> None:
        self.front = None

    def _start(self, world: np.ndarray, ego_xy: np.ndarray) -> int:
        if self.front is not None:
            d = np.linalg.norm(world - self.front, axis=1)
            if d.min() <= SAME_POINT_M:
                return int(d.argmin())
        # First tick, or the held point left the window: the route starts where the car is, so begin at the
        # last point not ahead of it.
        behind = np.flatnonzero(ego_xy[:, 0] <= 0.0)
        return int(behind[-1]) if len(behind) else 0

    def target(self, route_xy, ego2world) -> np.ndarray:
        """The agent's target point, in **paddock's** ego frame (x forward, y left)."""
        route_xy = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
        route_xy = route_xy[np.isfinite(route_xy).all(axis=1)]
        if len(route_xy) == 0:
            raise ValueError("the route (Observation.route_xy) is empty")
        world = _ego_to_world(route_xy, ego2world)
        ego = np.asarray(ego2world, dtype=np.float64)[:2, 3]
        route = world[self._start(world, route_xy) :]
        to_pop = 0
        if len(route) > 2:
            farthest = -np.inf
            cumulative = 0.0
            for i in range(1, len(route)):
                if cumulative > ROUTE_MAX_DISTANCE_M:
                    break
                cumulative += float(np.linalg.norm(route[i] - route[i - 1]))
                distance = float(np.linalg.norm(route[i] - ego))
                if farthest < distance <= ROUTE_MIN_DISTANCE_M:
                    farthest = distance
                    to_pop = i
            to_pop = min(to_pop, len(route) - 2)
        route = route[to_pop:]
        self.front = route[0].copy()
        chosen = route[1] if len(route) > 1 else route[0]
        return _world_to_ego(chosen[None], ego2world)[0]


def target_record(
    obs_route_xy, route_timestamp_us, timestamp_us, ego2world, index, tracker
) -> dict[str, Any]:
    """HiP-AD's target point (its axes, from the ego origin) and where it came from ("route" | "command").

    The route is used where it is in the observation's own frame: Bench2Drive's, which carries no stamp, or
    one stamped at the observation's time. AlpaSim's route, in the rig frame at another time, is not used.
    """
    stamped_elsewhere = route_timestamp_us is not None and int(route_timestamp_us) != int(timestamp_us)
    if obs_route_xy is not None and not stamped_elsewhere:
        route = np.asarray(obs_route_xy, dtype=np.float64).reshape(-1, 2)
        if np.isfinite(route).all(axis=1).sum() >= 1:
            ego = tracker.target(route, ego2world)
            point = ego_to_lidar_axes(ego).astype(np.float32)
            return {"source": "route", "target_point": point, "ego_xy": np.round(ego, 3).tolist()}
    return {"source": "command", "target_point": command_target_point(index)}


def aug_config(load_size=LOAD_SIZE, final_size=FINAL_SIZE) -> dict[str, Any]:
    """The agent's ``get_augmentation``: scale to cover `final_size`, cut from the bottom, no flip or turn."""
    width, height = load_size
    final_w, final_h = final_size
    resize = max(final_h / height, final_w / width)
    new_w, new_h = int(width * resize), int(height * resize)
    crop_h = int((1 - 0.0) * new_h) - final_h
    crop_w = int(max(0, new_w - final_w) / 2)
    return {
        "resize": resize,
        "resize_dims": (new_w, new_h),
        "crop": (crop_w, crop_h, crop_w + final_w, crop_h + final_h),
        "flip": False,
        "rotate": 0,
        "rotate_3d": 0,
    }
