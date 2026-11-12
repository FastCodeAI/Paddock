"""The per-tick analysis record the Bench2Drive agent writes beside ``log.csv``.

``log.csv`` keeps six waypoints and the actuation, which is enough to replay
what the car did and nothing about why. Explaining a route needs what the node
actually answered -- every waypoint and every extra head (lead, lanes, pose,
sigma, desire, longitudinal intent) -- next to what the car could have known
and the model could not: where the other actors were, which lane it was in,
how far along the route it had got and what the next light showed. So when
recording is on, the agent also writes, per route:

* ``route_plan.json`` -- the leaderboard's world-coordinate route (1 m spacing),
  its cumulative length, and the traffic lights and stop signs whose trigger
  volume lies on it, each with the plan index it sits at;
* ``ticks.jsonl`` -- one JSON object per tick, see :func:`tick_record`;
* ``actors.jsonl`` -- one line per actor the first time it comes within
  :data:`ACTOR_RADIUS_M`: id, blueprint, role and half-extent, so the per-tick
  rows can stay five numbers an actor.

Nothing here reaches the model: it is privileged state, read for analysis only,
after the node has answered.

Frames. World positions are CARLA's own (left-handed, metres, yaw in degrees),
the frame the routes file and Bench2Drive's dataset annotations use, so a
route, a run and an expert drive overlay without conversion. Everything
relative to the ego -- actors, the plan -- is paddock's ego frame, x forward
and y LEFT, the frame the node's waypoints already arrive in.

Runs inside the CARLA client's interpreter (Python 3.8 at the one site that has
one) and takes plain numbers rather than ``carla`` objects, so it imports
without CARLA and is tested that way. Keep the syntax to what 3.8 accepts.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

#: Actors further than this from the ego are not written. Twice the longest
#: plan supercombo returns at urban speed, and a little more than the BEV
#: camera's half-width at its 30 m mount.
ACTOR_RADIUS_M = 60.0

#: A traffic light or stop sign belongs to the route when its trigger location
#: lies this close to the route polyline (a lane is ~3.5 m wide).
ON_ROUTE_M = 4.0

#: Downscaled copies of the model's own cameras are kept every this many ticks
#: (CARLA ticks at 20 Hz, so 2 is 10 Hz) at 1/MODEL_VIEW_SCALE of each side.
MODEL_VIEW_EVERY = 2
MODEL_VIEW_SCALE = 4

#: Where each tick's whole request goes when the inputs are recorded (``record_inputs``):
#: one compressed ``.npz`` per tick, the model's camera frames as rendered (uint8,
#: lossless; a frame :func:`wire_codec` sends as JPEG is encoded the same way again on
#: replay) beside the rest of the request as JSON, so a route can be fed to a model
#: again offline (``tools/benchmarks/bench2drive/replay.py``). ~2.5-3 MB a tick for six
#: 800x450 views, so it is its own setting, off unless asked for.
INPUTS_DIR = "inputs"

#: Frames of this many pixels or more go to the node as JPEG (q95), the rest as raw
#: ``.npy``: raw, a multi-megapixel frame is ~20 MB a tick. Every view of the
#: ``bench2drive`` and ``kitscenes`` rigs is over it; paddock's 800x450 views are not.
JPEG_FROM_PX = 1_000_000


def wire_codec(frame) -> str | None:
    """How the agent sends ``frame`` (H, W, 3) to the node: ``"jpeg"`` or raw (None).

    The agent and a replay of its record both decide by this, so a replayed request
    carries each frame the way the live one did.
    """
    return "jpeg" if frame.shape[0] * frame.shape[1] >= JPEG_FROM_PX else None


#: Thumbnail directories named before any model but openpilot was recorded, and
#: which its dashboard reads; every other camera gets :func:`model_view_dir`'s name.
MODEL_VIEW_DIRS = {"CAM_OP_ROAD": "model_road", "CAM_OP_WIDE": "model_wide"}

#: How far either side of the last match the route search looks before falling
#: back to the whole plan, in plan points (1 m apart).
SEARCH_WINDOW = 60


def rounded(value: Any, digits: int = 4) -> Any:
    """``value`` with every float rounded, recursively; JSON-ready.

    The node's extras are nested dicts and lists of floats, written at 20 Hz
    for up to 4000 ticks a route: full precision would triple the file for
    digits nobody can read off a simulator.
    """
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v, digits) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return rounded(value.item(), digits)
    return value


def to_ego(points_xy: np.ndarray, ego_x: float, ego_y: float, ego_yaw_deg: float) -> np.ndarray:
    """CARLA world points -> paddock ego frame (x forward, y left), (N, 2).

    CARLA is left-handed: at yaw 0 the car faces +x and +y is to its right, so
    forward is (cos yaw, sin yaw) and right is (-sin yaw, cos yaw). Left is the
    negation of right.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    yaw = math.radians(ego_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pts[:, 0] - ego_x, pts[:, 1] - ego_y
    return np.stack([dx * c + dy * s, dx * s - dy * c], axis=-1)


def model_view_dir(camera: str) -> str:
    """Where a camera's thumbnails go: ``model_front_left`` for ``CAM_FRONT_LEFT``.

    Every camera the model declares is kept, so a six-view model's record shows
    all six views it planned from, not the two openpilot reads.
    """
    if camera in MODEL_VIEW_DIRS:
        return MODEL_VIEW_DIRS[camera]
    return "model_" + (camera[4:] if camera.startswith("CAM_") else camera).lower()


def write_inputs(path: str, images: dict, body: dict) -> None:
    """One tick's request: ``images`` (camera id -> uint8 HxWx3, as sent) and the rest of the body.

    Written to a temporary name and renamed, so a route cut off mid-write leaves
    no torn file for a replay to trip on.
    """
    arrays = {"image_" + cid: np.ascontiguousarray(img, dtype=np.uint8) for cid, img in images.items()}
    text = np.frombuffer(json.dumps(body, separators=(",", ":")).encode(), dtype=np.uint8)
    partial = path + ".partial.npz"
    np.savez_compressed(partial, body=text, **arrays)
    os.replace(partial, path)


def read_inputs(path: str) -> tuple[dict, dict]:
    """``(images, body)`` as :func:`write_inputs` wrote them."""
    with np.load(path) as stored:
        body = json.loads(bytes(stored["body"]).decode())
        images = {name[len("image_") :]: stored[name] for name in stored.files if name.startswith("image_")}
    return images, body


def wrap_deg(angle: float) -> float:
    """``angle`` in (-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


class Route:
    """The leaderboard's route as arrays, and where on it the ego is.

    ``xy`` is (N, 2) CARLA world metres, ``s`` the along-route distance of each
    point from the first. The search remembers its last match and looks in a
    window around it first, so a route that doubles back near itself cannot
    jump the ego to the wrong pass, and a 4000-tick route does not rescan a
    thousand points per tick.
    """

    def __init__(self, xy: np.ndarray, options: list[str]) -> None:
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        self.options = list(options)
        steps = np.linalg.norm(np.diff(self.xy, axis=0), axis=1) if len(self.xy) > 1 else np.zeros(0)
        self.s = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.s[-1]) if len(self.s) else 0.0
        self._last = 0

    def locate(self, x: float, y: float) -> tuple[int, float]:
        """(index of the nearest route point, distance to it in metres)."""
        if not len(self.xy):
            return 0, float("nan")
        lo, hi = max(0, self._last - SEARCH_WINDOW), min(len(self.xy), self._last + SEARCH_WINDOW + 1)
        d = np.hypot(self.xy[lo:hi, 0] - x, self.xy[lo:hi, 1] - y)
        i, best = lo + int(np.argmin(d)), float(np.min(d))
        if best > 10.0:  # lost the thread (a respawn, a long excursion): search everything
            d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
            i, best = int(np.argmin(d)), float(np.min(d))
        self._last = i
        return i, best

    def on_route(self, x: float, y: float, radius: float = ON_ROUTE_M) -> int | None:
        """The index of the route point nearest ``(x, y)`` if within ``radius``, else None."""
        if not len(self.xy):
            return None
        d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
        i = int(np.argmin(d))
        return i if d[i] <= radius else None

    def plan_json(self, lights: list[dict], stops: list[dict]) -> dict:
        return {
            "frame": "carla_world",
            "xy": rounded(self.xy.tolist(), 2),
            "s": rounded(self.s.tolist(), 2),
            "options": self.options,
            "length_m": round(self.length, 2),
            "traffic_lights": lights,
            "stop_signs": stops,
        }


def next_on_route(marks: list[dict], index: int, route: Route) -> dict | None:
    """The first mark (light or stop) at or ahead of route ``index``, with its distance along the route."""
    ahead = [m for m in marks if m["i"] >= index]
    if not ahead:
        return None
    mark = min(ahead, key=lambda m: m["i"])
    return {"id": mark["id"], "d": round(float(route.s[mark["i"]] - route.s[index]), 2)}


def actor_row(
    actor_id: int, x: float, y: float, yaw_deg: float, vx: float, vy: float, ego: dict
) -> list[float] | None:
    """``[id, x, y, yaw, speed]`` in the ego frame, or None beyond :data:`ACTOR_RADIUS_M`.

    ``x, y, yaw_deg, vx, vy`` are the actor's CARLA world pose and velocity;
    ``ego`` holds the ego's ``x``, ``y`` and ``yaw`` the same way.
    """
    if math.hypot(x - ego["x"], y - ego["y"]) > ACTOR_RADIUS_M:
        return None
    ex, ey = to_ego(np.array([[x, y]]), ego["x"], ego["y"], ego["yaw"])[0]
    return [
        int(actor_id),
        round(float(ex), 2),
        round(float(ey), 2),
        # Relative heading, positive to the left like the ego frame's y.
        round(-wrap_deg(yaw_deg - ego["yaw"]), 1),
        round(math.hypot(vx, vy), 2),
    ]


def tick_record(
    tick: int,
    game_time: float,
    infer_ms: float,
    ego: dict,
    sent: dict,
    reply: dict,
    control: dict,
    route: dict | None = None,
    lane: dict | None = None,
    lights: dict | None = None,
    actors: list | None = None,
    asked: bool = True,
    plan=None,
    dt: float | None = None,
) -> dict:
    """One line of ``ticks.jsonl``.

    ``asked``    whether this tick asked the node for a plan. A model that declares
                 ``planning_hz`` is asked once per 1/planning_hz s, and the ticks
                 between follow the last plan: on those nothing is sent, ``infer_ms``
                 is 0, and ``sent`` and ``reply`` are what this tick would have sent
                 and the reply still followed. Read it with ``tick_asked``.
    ``ego``      CARLA world pose (x, y, z, yaw, pitch, roll), speedometer speed,
                 ego-frame velocity/acceleration, yaw rate, posted limit (m/s)
    ``sent``     what the node was given besides pixels: command, navigation
                 (the route polyline included), the agent's acceleration estimate
    ``reply``    the node's JSON exactly as returned (trajectory, dt, extras)
    ``plan``     the points the controller followed, ``dt`` s apart (``pid.on_grid``):
                 the reply's own points every 0.5 s. For a plan 0.5 s apart they are the
                 reply's ``trajectory``, which is recorded when ``plan`` is None
    ``control``  steer / throttle / brake and the tracker's target speed
    ``route``    nearest route index, along-route metres, distance off the route
    ``lane``     the map lane under the ego and the ego's offset and heading in it
    ``lights``   the light the ego is at, and the next light / stop sign on the route
    ``actors``   ``[id, x, y, yaw, speed]`` rows within the radius, ego frame
    """
    record = {
        "t": tick,
        "time": round(game_time, 3),
        "asked": bool(asked),
        "infer_ms": round(infer_ms, 1),
        "ego": rounded(ego, 4),
        "sent": rounded(sent, 4),
        "plan": rounded(reply.get("trajectory") if plan is None else plan, 3),
        "dt": reply.get("dt") if dt is None else dt,
        "extras": rounded(reply.get("extras") or {}, 3),
        "ctrl": rounded(control, 4),
    }
    if route is not None:
        record["route"] = rounded(route, 3)
    if lane is not None:
        record["lane"] = rounded(lane, 3)
    if lights is not None:
        record["lights"] = lights
    if actors is not None:
        record["actors"] = actors
    return record


def tick_asked(record: dict) -> bool:
    """Whether a ``ticks.jsonl`` line is a tick that asked the node for a plan.

    Its ``asked`` field. Lines written before that field (28-29 Oct 2026, when the
    client's own rate was new) marked a tick that did not ask only by an /infer time of
    exactly 0, which no request takes; every earlier line asked.
    """
    if "asked" in record:
        return bool(record["asked"])
    infer_ms = record.get("infer_ms")
    return infer_ms is None or infer_ms > 0


def dumps(record: dict) -> str:
    """One JSONL line, compact."""
    return json.dumps(record, separators=(",", ":"), allow_nan=False)
