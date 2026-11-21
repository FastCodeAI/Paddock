"""Paddock observations as SimLingo's input, and SimLingo's output as a paddock plan, in numpy.

What SimLingo's own CARLA agent does (``team_code/agent_simlingo.py`` at 743b243), and what of it
this module reproduces so that its tests run in paddock's own environment:

*Route.* The agent keeps the leaderboard's sparse route plan in world coordinates
(``nav_planner.RoutePlanner``, ``min_distance`` 7.5 m, ``max_distance`` 50 m). Every tick it
drops the points up to the farthest one within 7.5 m of the ego, looking no further than 50 m
along the route, and keeps at least two. The next two points are its two target points.
Paddock sends the same plan every tick as ``Observation.route_xy``, in the ego frame and
windowed around the ego, so :class:`RouteTracker` keeps the same state by position: the world
point the agent would hold at the front of its route.

*Frames.* SimLingo's ego frame is CARLA's: x forward, y **right**. Paddock's is x forward,
y **left**. Target points go in with y flipped, and the plan comes back with y flipped.

*Image.* The agent cuts the bottom 4.8/16 of the frame (the bonnet), as training did:
``rgb[: int(H - (H * 4.8) // 16)]``, 359 of 512 rows.

*Camera, open loop.* The agent's camera is 1024x512 at 110 deg (team_code/config_simlingo.py);
paddock's Bench2Drive rig renders it as ``CAM_SIMLINGO``. A recorded dataset has its own, narrower
``CAM_FRONT`` (64-88 deg). :func:`to_canvas` places that frame at its true angular scale inside a
black 1024x512, 110 deg frame, principal point on the centre: objects keep the size and bearing
SimLingo learned, and what the real camera did not see stays black. Stretching it to fill 110 deg
would make everything look nearer than it is. The mount (height, offset) is not changed.

*Route, open loop.* A recorded dataset sends no route, only a command. SimLingo was trained with
the route given either as target points or as a sentence (``route_as: target_point_command`` in its
config), and its agent has the sentence form built in (``eval_route_as = 'command'``):
:func:`command_sentence` writes it as the agent does.

*Plan.* SimLingo predicts ten positions 0.25 s apart (``speed_wps``, t = 0.25 ... 2.5 s) and
a path of twenty points along the road (``route``). Paddock's closed-loop clients need six
points on the 0.5 s grid, 3 s, so :func:`plan` pads the plan past 2.5 s with its last
position: nothing is predicted beyond what the model gave.
"""

from __future__ import annotations

import numpy as np

from paddock.policy.protocol import Command

#: The agent's RoutePlanner(min_distance, max_distance), agent_simlingo.py.
ROUTE_MIN_DISTANCE_M = 7.5
ROUTE_MAX_DISTANCE_M = 50.0

#: SimLingo's position spacing: data_save_freq 5 frames of CARLA's 20 Hz.
PLAN_DT = 0.25
#: Ten speed waypoints (adaptors.py: future_speed_waypoints = 10).
NATIVE_POINTS = 10
NATIVE_HORIZON_S = PLAN_DT * NATIVE_POINTS

#: A route point counts as the one the tracker held if it is this close to it.
SAME_POINT_M = 1.0

#: SimLingo's camera, as its agent mounts and renders it.
CANVAS_W, CANVAS_H = 1024, 512
CANVAS_FOV_DEG = 110.0
CANVAS_F = CANVAS_W / (2.0 * np.tan(np.radians(CANVAS_FOV_DEG) / 2.0))

#: The agent's map_command (agent_simlingo.py), keyed by paddock's Command. Paddock's STRAIGHT is
#: "no turn ahead", the agent's lane-following command 4, "follow the road"; its "go straight at the
#: next intersection" (3) names a junction, which a dataset's command does not say there is.
COMMAND_TEXT = {
    Command.RIGHT: "go right at the next intersection",
    Command.LEFT: "go left at the next intersection",
    Command.STRAIGHT: "follow the road",
    Command.FOLLOW_REFERENCE: "follow the road",
}
#: The junction maneuver (Observation.nav_maneuver) each turning command is.
COMMAND_MANEUVER = {Command.RIGHT: "TURN_RIGHT", Command.LEFT: "TURN_LEFT"}
FOLLOW = "follow the road"


def flip_y(xy: np.ndarray) -> np.ndarray:
    """Between paddock's ego frame (y left) and SimLingo's (y right). Its own inverse."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 1] = -out[..., 1]
    return out


def ego_to_world(xy: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    """(N, 2) ego-frame points to the world frame's x, y."""
    pts = np.c_[np.asarray(xy, dtype=np.float64), np.zeros(len(xy)), np.ones(len(xy))]
    return (pts @ np.asarray(ego2world, dtype=np.float64).T)[:, :2]


def world_to_ego(xy: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    """(N, 2) world-frame x, y to the ego frame, on the ego's ground plane."""
    world2ego = np.linalg.inv(np.asarray(ego2world, dtype=np.float64))
    pts = np.c_[np.asarray(xy, dtype=np.float64), np.zeros(len(xy)), np.ones(len(xy))]
    # z is not known in the world; take the ego's own height so the drop is only planar.
    pts[:, 2] = np.asarray(ego2world, dtype=np.float64)[2, 3]
    return (pts @ world2ego.T)[:, :2]


def cropped_rows(height: int) -> int:
    """Rows kept by the agent's bonnet cut: ``int(H - (H * 4.8) // 16)``."""
    return int(height - (height * 4.8) // 16)


class RouteTracker:
    """SimLingo's RoutePlanner, fed paddock's route each tick instead of holding the plan once."""

    def __init__(self) -> None:
        self.front: np.ndarray | None = None  # world x, y of the point at the front of the route

    def reset(self) -> None:
        self.front = None

    def _start(self, world: np.ndarray, ego_xy: np.ndarray) -> int:
        """Index of the point the agent would hold at the front of its route."""
        if self.front is not None:
            d = np.linalg.norm(world - self.front, axis=1)
            if d.min() <= SAME_POINT_M:
                return int(d.argmin())
        # First tick, or the held point left paddock's window: the agent's route starts where
        # the car does, so begin at the last point not ahead of the ego.
        behind = np.flatnonzero(ego_xy[:, 0] <= 0.0)
        return int(behind[-1]) if len(behind) else 0

    def targets(self, route_xy: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
        """The agent's two target points, (2, 2), in **paddock's** ego frame (x forward, y left)."""
        route_xy = np.asarray(route_xy, dtype=np.float64)
        if route_xy.ndim != 2 or len(route_xy) == 0:
            raise ValueError("SimLingo needs the route (Observation.route_xy); this source sends none")
        world = ego_to_world(route_xy, ego2world)
        ego = np.asarray(ego2world, dtype=np.float64)[:2, 3]
        start = self._start(world, route_xy)
        route = world[start:]
        to_pop = 0
        if len(route) > 2:  # run_step returns the route untouched when it has two points or fewer
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
            to_pop = min(to_pop, len(route) - 2)  # popleft only while more than two remain
        route = route[to_pop:]
        self.front = route[0].copy()
        if len(route) > 2:
            chosen = route[1:3]
        elif len(route) > 1:
            chosen = route[[1, 1]]
        else:
            chosen = route[[0, 0]]
        return world_to_ego(chosen, ego2world)


def plan(speed_wps: np.ndarray, horizon_s: float) -> np.ndarray:
    """SimLingo's positions as paddock waypoints 0.25 s apart, x forward, y left, out to ``horizon_s``.

    ``speed_wps`` is (10, 2) in SimLingo's frame (y right). The first ten points are the model's own;
    every point past 2.5 s repeats the last of them.
    """
    wps = np.asarray(speed_wps, dtype=np.float64)
    if wps.shape != (NATIVE_POINTS, 2):
        raise ValueError(f"expected ({NATIVE_POINTS}, 2) speed waypoints, got {wps.shape}")
    n = int(round(horizon_s / PLAN_DT))
    if abs(n * PLAN_DT - horizon_s) > 1e-6 or n < NATIVE_POINTS:
        raise ValueError(
            f"horizon {horizon_s} s must be a multiple of {PLAN_DT} s "
            f"and at least the model's {NATIVE_HORIZON_S} s"
        )
    padded = np.vstack([wps, np.repeat(wps[-1:], n - NATIVE_POINTS, axis=0)])
    return flip_y(padded)


def to_canvas(rgb: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """``rgb`` from a pinhole camera, placed at its true angular scale in SimLingo's 110 deg frame.

    Each pixel keeps the bearing it had: the frame is scaled by the ratio of focal lengths and its
    principal point put on the canvas centre. Where the source camera saw nothing, black.
    """
    from PIL import Image

    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    sx, sy = CANVAS_F / fx, CANVAS_F / fy
    new_w, new_h = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    method = Image.Resampling.BOX if (sx < 1 and sy < 1) else Image.Resampling.BILINEAR
    scaled = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), method), dtype=np.uint8)
    ox = int(round(CANVAS_W / 2 - cx * sx))
    oy = int(round(CANVAS_H / 2 - cy * sy))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + new_w, CANVAS_W), min(oy + new_h, CANVAS_H)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = scaled[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return canvas


def command_sentence(command: int, distance_m: float | None = None) -> str:
    """The route as the agent writes it in command mode: ``Command: ... .``

    The agent adds the distance to a turn when it knows it ("in 25 meter"); a dataset's command
    rarely comes with one, and then the sentence is the form the agent uses without a distance.
    """
    text = COMMAND_TEXT[Command(int(command))]
    if text != FOLLOW and distance_m is not None:
        return f"Command: {text} in {int(distance_m)} meter."
    return f"Command: {text}."


def turn_distance(command: int, maneuver: str | None, distance_m: float | None) -> float | None:
    """The distance to the command's own turn: only when the next junction maneuver is that turn."""
    wanted = COMMAND_MANEUVER.get(Command(int(command)))
    return distance_m if wanted is not None and maneuver == wanted else None
