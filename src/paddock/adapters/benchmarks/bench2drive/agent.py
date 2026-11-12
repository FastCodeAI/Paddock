#!/usr/bin/env python
"""Model-agnostic Bench2Drive / CARLA-leaderboard AutonomousAgent for paddock.

The model runs in its own env behind paddock's HTTP policy node; this agent is
the CARLA *client*. Each tick it renders the cameras the model declares, packs
them into the paddock wire format, POSTs ``/infer``, receives waypoints (ego
frame, x forward, y LEFT) and turns them into a CARLA ``VehicleControl`` via
the Bench2Drive dual-PID tracker, reimplemented in ``pid.py``.

One agent serves every model: which cameras to render and which node to call
come from the environment, set by ``harness.run_episode``:

    PADDOCK_NODE_URL      policy node base url
    PADDOCK_CAMERAS       comma-separated camera ids       (default CAM_FRONT)
    PADDOCK_CAMERA_RIG    where those cameras sit and what they render: a rig
                          named in ``rig.py``          (default paddock)
    PADDOCK_WAYPOINT_DT   the grid the controller reads a plan on, s (default 0.5):
                          the plan's own points at every multiple of it, the
                          first six (``pid.on_grid``), whatever its spacing
    PADDOCK_INFER_HZ      ask for a plan this often and follow the last one in
                          between (default 0: every tick); from `planning_hz`
    PADDOCK_RECORD_DIR    if set, write front/BEV frames + a per-tick log.csv
                          here; the harness sets SAVE_PATH alongside it so the
                          leaderboard allows the BEV camera's 30 m mount. Also
                          the analysis record (``recorder.py``): every tick's
                          full node reply beside privileged world state, the
                          route plan, and thumbnails of the model's cameras;
                          and ``rig.json``, the cameras as rendered
    PADDOCK_RECORD_INPUTS "1" (with PADDOCK_RECORD_DIR): also keep each tick's
                          whole request -- the model cameras' frames as sent and
                          the rest of the body -- under inputs/, for offline replay
    PADDOCK_METRICS_DIR   if set, write <dir>/<save_name>/metric_info.json: the
                          per-tick ego kinematics Bench2Drive's
                          efficiency_smoothness_benchmark.py scores. The
                          leaderboard hands the agent its `save_name` by
                          appending "+<save_name>" to the agent-config string
                          (leaderboard_evaluator.py:363 at 7ec25d1)
    PADDOCK_STOP_STUCK_S  if set, end the route once the ego has gone this many
                          game seconds without getting 3 m further along it
    PADDOCK_STOP_AFTER_S  if set, end the route at this game time (the harness
                          sets it to a multiple of the expert's time on the route)
    PADDOCK_STOP_ON_COLLISION  "1": end the route at the first collision the
                          leaderboard records (and so penalises)
    PADDOCK_STOP_REQUEST  a file whose appearance ends the route by hand (the
                          dashboard's "Stop route" writes it)
    PADDOCK_STOP_RECORD   where to write why and when a route was ended early;
                          see ``early_stop.py``
    PADDOCK_SEGMENT       a sim2real diagnostic, never a scored run: the spec of a
                          recorded segment to capture or replay; see ``segment.py``

It runs inside the CARLA client's interpreter (Python 3.8 at the one site that
has one), with ``REPO_ROOT/src`` on PYTHONPATH like every process paddock
launches -- so the navigation commands, canbus layout, image encoding and
trajectory validation are imported from ``paddock.policy.protocol`` rather than
restated here. Keep the syntax to what 3.8 accepts.

Coordinate note: CARLA is left-handed (x fwd, y RIGHT); paddock uses x fwd,
y LEFT. Mapped both ways with a y-flip (F @ M @ F on the way in; the tracker
takes paddock's frame directly on the way out).
"""

from __future__ import annotations

import csv
import json
import math
import os
import time

import carla
import cv2
import numpy as np
import requests
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from paddock.adapters.benchmarks.bench2drive import early_stop, navigation, recorder, rig
from paddock.adapters.benchmarks.bench2drive.pid import WaypointTracker, on_grid
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    N_WAYPOINTS,
    Command,
    Trajectory,
    encode_image,
)

SERVER = os.environ.get("PADDOCK_NODE_URL", "http://127.0.0.1:9000")
CAMS = [c.strip() for c in os.environ.get("PADDOCK_CAMERAS", "CAM_FRONT").split(",") if c.strip()]
WAYPOINT_DT = float(os.environ.get("PADDOCK_WAYPOINT_DT", "0.5"))
RECORD_DIR = os.environ.get("PADDOCK_RECORD_DIR") or None
#: "1": also keep each tick's whole request under RECORD_DIR/inputs/ (recorder.write_inputs)
RECORD_INPUTS = os.environ.get("PADDOCK_RECORD_INPUTS") == "1" and RECORD_DIR is not None
METRICS_DIR = os.environ.get("PADDOCK_METRICS_DIR") or None
STOP_STUCK_S = float(os.environ["PADDOCK_STOP_STUCK_S"]) if os.environ.get("PADDOCK_STOP_STUCK_S") else None
STOP_AFTER_S = float(os.environ["PADDOCK_STOP_AFTER_S"]) if os.environ.get("PADDOCK_STOP_AFTER_S") else None
STOP_ON_COLLISION = os.environ.get("PADDOCK_STOP_ON_COLLISION") == "1"
STOP_REQUEST = os.environ.get("PADDOCK_STOP_REQUEST") or None
STOP_RECORD = os.environ.get("PADDOCK_STOP_RECORD") or None
SEGMENT = os.environ.get("PADDOCK_SEGMENT") or None
#: Ask the node for a plan at most this often, Hz, and follow the last plan
#: unchanged in between: 2 Hz is one /infer per 10 frames of the 20 Hz
#: simulation. 0 asks every tick. The harness sets it from the model's
#: `planning_hz`. Unchanged means in the ego frame it was planned in, as the
#: native-rate campaign drove it: its log.csv holds identical waypoints for
#: 10-frame runs (docs/benchmarks/bench2drive-native-rate.md).
INFER_HZ = float(os.environ.get("PADDOCK_INFER_HZ") or 0)
#: Slack on "a plan is due", us: a 0.5 s step can read as 499999 us from floats.
DUE_SLACK_US = 1000

#: The rig the cameras are placed by (``rig.py``); the harness passes a site's ``camera_rig``.
CAMERA_RIG = os.environ.get("PADDOCK_CAMERA_RIG") or rig.DEFAULT
#: Every camera a model might ask for: placement in the CARLA ego frame, field of view, size.
RIG = rig.cameras(CAMERA_RIG)
#: Rendered for the frames/BEV recording even when the model does not consume it.
RECORD_CAM = "CAM_FRONT"


def _rendered() -> list:
    """The cameras this agent renders: the model's own, and the recording camera when recording."""
    return list(CAMS) + ([RECORD_CAM] if RECORD_DIR and RECORD_CAM not in CAMS else [])


# CARLA's route option -> paddock's navigation command. Lane changes are
# "keep going" from the model's point of view.
CMD = {
    RoadOption.LEFT: Command.LEFT,
    RoadOption.RIGHT: Command.RIGHT,
    RoadOption.STRAIGHT: Command.STRAIGHT,
    RoadOption.LANEFOLLOW: Command.FOLLOW_REFERENCE,
    RoadOption.CHANGELANELEFT: Command.STRAIGHT,
    RoadOption.CHANGELANERIGHT: Command.STRAIGHT,
}
# Junction maneuvers, for policies that take a navigation instruction (Alpamayo's
# "Turn left in 40m"). Lane-follow and lane changes are not maneuvers here.
NAV_MANEUVER = {
    RoadOption.LEFT: "TURN_LEFT",
    RoadOption.RIGHT: "TURN_RIGHT",
    RoadOption.STRAIGHT: "STRAIGHT",
}

F = np.diag([1.0, -1.0, 1.0, 1.0])  # y-flip: CARLA (left-handed) <-> paddock (right-handed)

# Body (x-fwd, y-left, z-up) -> camera-optical (x-right, y-down, z-fwd). paddock's
# camera2ego convention is the optical frame (as nuScenes/neuroncap supply); without
# this the geometric warp openpilot builds is rotated 90 deg and it never sees the
# road (derived mounting rpy [-90,0,90] instead of [0,0,0]). Right-multiplied onto
# each camera's ego-frame rotation so mounting yaw/pitch are preserved.
R_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

#: Columns of waypoints written to log.csv when recording.
LOGGED_WAYPOINTS = 6

#: The model's own cameras -- every one it declares -- kept as thumbnails in the
#: analysis record, each in its own directory.
MODEL_VIEWS = {cid: recorder.model_view_dir(cid) for cid in CAMS}

#: Ticks between refreshes of the candidate actor list. Poses are read every
#: tick from the client's own copy of the last snapshot; only *which* actors
#: exist is re-listed, so a new one appears in the record within half a second.
ACTOR_REFRESH_TICKS = 10

#: Blueprint prefixes of what the analysis record tracks: cars and bikes,
#: pedestrians, and the props scenarios place in the lane (cones, warning
#: signs, and the parked cars Bench2Drive spawns as static meshes).
TRACKED = ("vehicle.", "walker.pedestrian.", "static.prop.")

#: Seconds to wait for one /infer reply. Under heavy GPU contention a single
#: DriveJEPA step can stall well past 120 s; raise via PADDOCK_INFER_TIMEOUT_S.
INFER_TIMEOUT_S = int(os.environ.get("PADDOCK_INFER_TIMEOUT_S", "120"))


def get_entry_point():
    return "PaddockB2DAgent"


def _rh(matrix4x4) -> np.ndarray:
    """CARLA local->world 4x4 -> the same transform in the y-flipped frame."""
    m = np.array(matrix4x4, dtype=np.float64)
    return F @ m @ F


def _intrinsics(width, height, fov_deg) -> list:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]


class PaddockB2DAgent(AutonomousAgent):
    #: Segment mode's driver (``segment.py``) when PADDOCK_SEGMENT is set; None on every other run.
    _seg = None

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._t = 0
        # {tick: kinematics}, the shape efficiency_smoothness_benchmark.py reads.
        # Keyed by save_name so a sweep's routes (and repetitions) never collide.
        self._metric_info = {}
        self._metric_path = None
        conf = str(path_to_conf_file or "")
        save_name = conf.rsplit("+", 1)[-1] if "+" in conf else ""
        if METRICS_DIR and save_name:
            self._metric_path = os.path.join(METRICS_DIR, save_name, "metric_info.json")
        self._prev_speed = 0.0
        self._prev_us = None
        self._payload = None
        self._plan_us = None
        self._hero = None
        self._tracker = WaypointTracker(WAYPOINT_DT)
        # One connection to the loopback node for the whole route.
        self._http = requests.Session()
        self._cam2ego = {}
        self._cam2img = {}
        for cid in CAMS:
            w, h = RIG[cid]["size"]
            x, y, z, roll, pitch, yaw = RIG[cid]["pose"]
            tf = carla.Transform(
                carla.Location(x=x, y=y, z=z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
            )
            m = _rh(tf.get_matrix())
            m[:3, :3] = m[:3, :3] @ R_OPT  # body -> camera-optical (paddock convention)
            self._cam2ego[cid] = m.tolist()
            self._cam2img[cid] = _intrinsics(w, h, RIG[cid]["fov"])
        try:
            self._http.post(f"{SERVER}/reset", json={}, timeout=30)
        except Exception as exc:  # a fresh node has nothing to reset; not fatal
            print("[paddock-b2d] reset warning:", exc)
        # Segment mode (PADDOCK_SEGMENT): capture a recorded run tick by tick, or replay it to
        # T0 and then hand over with converted frames. Off for every standard run; the spec
        # names the cameras it touches. See segment.py.
        if SEGMENT:
            # Imported only here: it needs Pillow, which preflight does not check and a scored
            # run's client interpreter need not have.
            from paddock.adapters.benchmarks.bench2drive import segment

            self._seg = segment.Segment(
                SEGMENT,
                {c: RIG[c]["pose"] for c in CAMS},
                {c: RIG[c]["size"] for c in CAMS},
                {c: RIG[c]["fov"] for c in CAMS},
            )
        self._log = None
        if RECORD_DIR:
            os.makedirs(os.path.join(RECORD_DIR, "front"), exist_ok=True)
            os.makedirs(os.path.join(RECORD_DIR, "bev"), exist_ok=True)
            self._write_rig_record()
            self._log = open(os.path.join(RECORD_DIR, "log.csv"), "w", newline="")
            self._w = csv.writer(self._log)
            self._w.writerow(
                ["frame", "speed", "steer", "throttle", "brake", "v_des", "path_end"]
                + [f"wp{i}{ax}" for i in range(LOGGED_WAYPOINTS) for ax in ("x", "y")]
                + ["ego_x", "ego_y", "ego_yaw", "resume", "nav_maneuver", "nav_distance_m"]
            )
        self._nav_route = self._dense_route()
        self._early_stop = None
        # What early_stop.json says: written when a rule ends the route, or at the
        # end with "reason": null, so a route's record also names its own budget.
        self._stop_record = None
        if STOP_STUCK_S is not None or STOP_AFTER_S is not None or STOP_ON_COLLISION:
            self._early_stop = early_stop.EarlyStop(STOP_STUCK_S, STOP_AFTER_S, STOP_ON_COLLISION)
            self._stop_record = {
                "reason": None,
                "stuck_s": STOP_STUCK_S,
                "after_s": STOP_AFTER_S,
                "collision": STOP_ON_COLLISION,
            }
        # A stop request is honoured whether or not any rule is on; once is enough.
        self._stop_requested = False
        # The lights on the route, so a car waiting short of a red one is not stuck.
        self._lights_by_id = {}
        self._stop_lights = []
        if self._early_stop is not None and self._nav_route is not None:
            try:
                self._stop_lights, _ = self._route_marks(self._nav_route)
            except Exception as exc:  # the hold falls back to CARLA's own "at a light"
                print("[paddock-b2d] route lights unavailable to the early stop:", repr(exc))
        self._ticks = None
        if RECORD_DIR:
            self._open_analysis_record()
        self._inputs_failed = False
        if RECORD_INPUTS:
            os.makedirs(os.path.join(RECORD_DIR, recorder.INPUTS_DIR), exist_ok=True)

    def _dense_route(self):
        """The leaderboard's own route as a ``recorder.Route``: what the early stop measures progress on,
        and where the lane-change lookup finds the next lane change.

        Built whether or not the analysis record is on: it feeds the node, not
        the record. None when there is no route, which leaves the early stop
        without a measure and sends "no lane change", rather than stopping the
        route at setup.
        """
        plan = getattr(self, "_dense_plan", None) or []
        if not plan:
            return None
        try:
            return recorder.Route(
                [[tf.location.x, tf.location.y] for tf, _ in plan], [opt.name for _, opt in plan]
            )
        except Exception as exc:  # a malformed plan costs the measure, not the route
            print("[paddock-b2d] dense route unavailable:", repr(exc))
            return None

    def _write_rig_record(self):
        """``rig.json``: the rig's name and every camera rendered, so a video of this route
        draws through the cameras it was driven with rather than paddock's default."""
        try:
            with open(os.path.join(RECORD_DIR, rig.RECORD), "w") as handle:
                json.dump({"name": CAMERA_RIG, "cameras": {cid: RIG[cid] for cid in _rendered()}}, handle)
        except OSError as exc:  # for the videos only; the route goes on without it
            print("[paddock-b2d] rig record not written:", repr(exc))

    # ---- the analysis record (recorder.py): never allowed to stop a route ----

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Keep the route at the leaderboard's own spacing for the analysis record.

        The base class stores only a 50 m downsample of it
        (``autonomous_agent.py:132`` at 7ec25d1), which is what ``_command`` and
        ``_nav`` read and go on reading; a 133 m route is four points there, too
        few to say how far along it the car got. The route handed in here is
        the RouteScenario's own, interpolated at 1 m.
        """
        self._dense_plan = list(global_plan_world_coord)
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def _open_analysis_record(self):
        """route_plan.json, ticks.jsonl and actors.jsonl; see ``recorder.py``.

        Called from ``setup``, which the leaderboard runs after
        ``set_global_plan`` (leaderboard_evaluator.py:362-364 at 7ec25d1), so
        the route is already known. A failure here costs the record, not the
        route: an exception out of ``setup`` would score as "Agent couldn't be
        set up".
        """
        self._record_failed = False
        self._tracked = []
        self._seen = set()
        try:
            plan = getattr(self, "_dense_plan", None) or []
            self._route = recorder.Route(
                [[tf.location.x, tf.location.y] for tf, _ in plan], [opt.name for _, opt in plan]
            )
            try:
                self._route_lights, self._route_stops = self._route_marks(self._route)
            except Exception as exc:  # the lights are context; the rest of the record still stands
                print("[paddock-b2d] route lights not recorded:", repr(exc))
                self._route_lights, self._route_stops = [], []
            with open(os.path.join(RECORD_DIR, "route_plan.json"), "w") as handle:
                json.dump(self._route.plan_json(self._route_lights, self._route_stops), handle)
            for view in MODEL_VIEWS.values():
                os.makedirs(os.path.join(RECORD_DIR, view), exist_ok=True)
            self._ticks = open(os.path.join(RECORD_DIR, "ticks.jsonl"), "w")
            self._actors_log = open(os.path.join(RECORD_DIR, "actors.jsonl"), "w")
        except Exception as exc:
            print("[paddock-b2d] analysis record disabled:", repr(exc))
            self._ticks = None

    def _route_marks(self, route):
        """Traffic lights and stop signs on ``route`` (a ``recorder.Route``), by route index.

        A light is on the route when one of the stop-line waypoints it governs
        (``get_affected_lane_waypoints``) lies on the route *and* points the way
        the route does there -- the stop line of the opposing lane is a lane
        width from the route too, and without the heading test it would be
        counted. Stop signs carry no such list, so theirs is the trigger
        volume's centre within ``recorder.ON_ROUTE_M``.
        """
        world = CarlaDataProvider.get_world()
        lights, stops = [], []
        if world is None or not len(route.xy):
            return lights, stops
        lo, hi = route.xy.min(axis=0) - 50.0, route.xy.max(axis=0) + 50.0
        for light in world.get_actors().filter("traffic.traffic_light*"):
            loc = light.get_location()
            if not (lo[0] <= loc.x <= hi[0] and lo[1] <= loc.y <= hi[1]):
                continue
            for wp in light.get_affected_lane_waypoints():
                tl = wp.transform.location
                i = route.on_route(tl.x, tl.y, radius=2.5)
                if i is None:
                    continue
                j = min(i + 1, len(route.xy) - 1)
                k = max(j - 2, 0)
                heading = math.degrees(
                    math.atan2(route.xy[j, 1] - route.xy[k, 1], route.xy[j, 0] - route.xy[k, 0])
                )
                if abs(recorder.wrap_deg(wp.transform.rotation.yaw - heading)) < 60.0:
                    lights.append({"id": light.id, "i": i, "x": round(tl.x, 2), "y": round(tl.y, 2)})
                    self._lights_by_id[light.id] = light
                    break
        for stop in world.get_actors().filter("traffic.stop"):
            loc = stop.get_transform().transform(stop.trigger_volume.location)
            i = route.on_route(loc.x, loc.y)
            if i is not None:
                stops.append({"id": stop.id, "i": i, "x": round(loc.x, 2), "y": round(loc.y, 2)})
        return lights, stops

    def _write_analysis_record(
        self, input_data, game_time, infer_ms, sent, payload, control, desired_speed, asked=True, plan=None
    ):
        """One line of ticks.jsonl, plus model-camera thumbnails every few ticks. ``plan`` is
        what the controller followed (``decide``), recorded in place of the reply's
        ``trajectory``, which for a plan finer than 0.5 s is not what the car drove."""
        hero = self._hero_actor()
        if hero is None:
            return
        tf = hero.get_transform()
        yaw = tf.rotation.yaw
        rad = math.radians(yaw)
        c, s = math.cos(rad), math.sin(rad)
        v, a, w = hero.get_velocity(), hero.get_acceleration(), hero.get_angular_velocity()
        ego = {
            "x": tf.location.x,
            "y": tf.location.y,
            "z": tf.location.z,
            "yaw": yaw,
            "pitch": tf.rotation.pitch,
            "roll": tf.rotation.roll,
            "speed": sent["speed"],
            # World vectors into the ego frame (x forward, y left), as recorder.to_ego.
            "v": [v.x * c + v.y * s, v.x * s - v.y * c],
            "a": [a.x * c + a.y * s, a.x * s - a.y * c],
            # CARLA's yaw grows clockwise seen from above; left-positive here.
            "yaw_rate": -w.z,
            "speed_limit": sent["speed_limit"],
        }
        index, off = self._route.locate(tf.location.x, tf.location.y)
        route = {
            "i": index,
            "s": float(self._route.s[index]) if len(self._route.s) else 0.0,
            "dev": off,
            "frac": float(self._route.s[index]) / self._route.length if self._route.length > 0 else 0.0,
        }
        lane = None
        wp = CarlaDataProvider.get_map().get_waypoint(tf.location)
        if wp is not None:
            wyaw = math.radians(wp.transform.rotation.yaw)
            dx, dy = tf.location.x - wp.transform.location.x, tf.location.y - wp.transform.location.y
            lane = {
                "road": wp.road_id,
                "lane": wp.lane_id,
                "junction": bool(wp.is_junction),
                "width": wp.lane_width,
                # Left-positive offset from the lane centre, and heading against it.
                "offset": dx * math.sin(wyaw) - dy * math.cos(wyaw),
                "heading": recorder.wrap_deg(yaw - wp.transform.rotation.yaw),
            }
        lights = {"at": bool(hero.is_at_traffic_light()), "state": str(hero.get_traffic_light_state())}
        nxt = recorder.next_on_route(self._route_lights, index, self._route)
        if nxt is not None:
            light = self._lights_by_id.get(nxt["id"])
            nxt["state"] = str(light.get_state()) if light is not None else None
            lights["next"] = nxt
        stop = recorder.next_on_route(self._route_stops, index, self._route)
        if stop is not None:
            lights["stop"] = stop

        if self._t % ACTOR_REFRESH_TICKS == 0 or not self._tracked:
            world = CarlaDataProvider.get_world()
            self._tracked = [
                actor
                for actor in world.get_actors()
                if actor.type_id.startswith(TRACKED) and actor.id != hero.id
            ]
        rows = []
        for actor in self._tracked:
            try:
                atf, av = actor.get_transform(), actor.get_velocity()
            except RuntimeError:  # destroyed since the last refresh
                continue
            row = recorder.actor_row(
                actor.id, atf.location.x, atf.location.y, atf.rotation.yaw, av.x, av.y, ego
            )
            if row is None:
                continue
            rows.append(row)
            if actor.id not in self._seen:
                self._seen.add(actor.id)
                ext = actor.bounding_box.extent
                meta = {
                    "id": actor.id,
                    "type": actor.type_id,
                    "role": actor.attributes.get("role_name", ""),
                    "mesh": actor.attributes.get("mesh_path", ""),
                    "ext": [round(ext.x, 2), round(ext.y, 2), round(ext.z, 2)],
                    "t": self._t,
                }
                self._actors_log.write(recorder.dumps(meta) + "\n")
                self._actors_log.flush()  # rare, and a live dashboard reads it mid-route

        record = recorder.tick_record(
            self._t,
            game_time,
            infer_ms,
            ego,
            sent,
            payload,
            {
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
                "v_des": desired_speed,
            },
            route=route,
            lane=lane,
            lights=lights,
            actors=rows,
            plan=None if plan is None else plan.xy.tolist(),
            dt=None if plan is None else plan.dt,
            asked=asked,
        )
        self._ticks.write(recorder.dumps(record) + "\n")
        self._ticks.flush()

        if self._t % recorder.MODEL_VIEW_EVERY == 0:
            for cid, view in MODEL_VIEWS.items():
                if cid in input_data:
                    bgr = input_data[cid][1][:, :, :3]
                    h, w = bgr.shape[:2]
                    small = cv2.resize(
                        bgr,
                        (w // recorder.MODEL_VIEW_SCALE, h // recorder.MODEL_VIEW_SCALE),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(
                        os.path.join(RECORD_DIR, view, f"{self._t:05d}.jpg"),
                        small,
                        [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )

    def destroy(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        if self._stop_record is not None and self._stop_record["reason"] is None:
            self._write_stop_record()  # on, and the route went to the leaderboard's own end
        if self._log is not None:
            self._log.close()
            self._log = None
        for handle in ("_ticks", "_actors_log"):
            if getattr(self, handle, None) is not None:
                getattr(self, handle).close()
                setattr(self, handle, None)
        if self._metric_path and self._metric_info:
            os.makedirs(os.path.dirname(self._metric_path), exist_ok=True)
            with open(self._metric_path, "w") as handle:
                json.dump(self._metric_info, handle)
        self._http.close()

    def _record_metric_info(self):
        """One tick of ego kinematics, exactly the six fields the upstream
        efficiency/comfort tool consumes (AutonomousAgent.get_metric_info at
        7ec25d1), read from the hero this agent already tracks."""
        hero = self._hero_actor()
        if hero is None:
            return
        tf = hero.get_transform()
        acc, ang = hero.get_acceleration(), hero.get_angular_velocity()
        fwd, right, loc, rot = tf.get_forward_vector(), tf.get_right_vector(), tf.location, tf.rotation
        self._metric_info[self._t] = {
            "acceleration": [acc.x, acc.y, acc.z],
            "angular_velocity": [ang.x, ang.y, ang.z],
            "forward_vector": [fwd.x, fwd.y, fwd.z],
            "right_vector": [right.x, right.y, right.z],
            "location": [loc.x, loc.y, loc.z],
            "rotation": [rot.roll, rot.pitch, rot.yaw],
        }

    def sensors(self):
        s = []
        for cid in _rendered():
            w, h = RIG[cid]["size"]
            x, y, z, roll, pitch, yaw = RIG[cid]["pose"]
            s.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "width": w,
                    "height": h,
                    "fov": RIG[cid]["fov"],
                    "id": cid,
                }
            )
        # Top-down BEV -- recording/visualisation only, never sent to the model.
        # Its z=30 mount exceeds the 3m radius limit, allowed only when SAVE_PATH is set.
        if RECORD_DIR and os.environ.get("SAVE_PATH"):
            s.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 30.0,
                    "roll": 0.0,
                    "pitch": -90.0,
                    "yaw": 0.0,
                    "width": 512,
                    "height": 512,
                    "fov": 60,
                    "id": "BEV",
                }
            )
        s.append({"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"})
        return s

    def _hero_actor(self):
        """The ego, found once per route rather than by an actor scan per tick."""
        if self._hero is None:
            self._hero = CarlaDataProvider.get_hero_actor()
        world = CarlaDataProvider.get_world()
        if self._hero is None and world is not None:  # not in the pool: fall back to a scan
            for actor in world.get_actors():
                if actor.attributes.get("role_name") == "hero":
                    self._hero = actor
                    break
        return self._hero

    def _ego2world(self) -> np.ndarray:
        hero = self._hero_actor()
        if hero is None:
            return np.eye(4)
        m = _rh(hero.get_transform().get_matrix())
        m[2, 3] = 0.0  # ego frame sits on the ground plane (paddock convention)
        return m

    def _command(self) -> int:
        plan = getattr(self, "_global_plan_world_coord", None)
        hero = self._hero_actor()
        if not plan or hero is None:
            return int(Command.FOLLOW_REFERENCE)
        hero_loc = hero.get_location()
        best, best_d = RoadOption.LANEFOLLOW, 1e9
        for tf, opt in plan:
            d = tf.location.distance(hero_loc)
            if d < best_d:
                best_d, best = d, opt
        return int(CMD.get(best, Command.FOLLOW_REFERENCE))

    def _nav(self) -> tuple:
        """The next junction maneuver on the route and the along-route metres to it.

        Walks the leaderboard's world-coordinate plan (1 m spacing, see
        ``set_global_plan``) forward from the point nearest the ego to the first
        LEFT/RIGHT/STRAIGHT option and sums the segment lengths on the way. Inside
        the maneuver the distance is 0; with no junction left before the goal it
        is (None, None), which a policy reads as "continue straight".
        """
        plan = getattr(self, "_global_plan_world_coord", None)
        hero = self._hero_actor()
        if not plan or hero is None:
            return None, None
        hero_loc = hero.get_location()
        start, best_d = 0, 1e9
        for i, (tf, _opt) in enumerate(plan):
            d = tf.location.distance(hero_loc)
            if d < best_d:
                best_d, start = d, i
        dist = 0.0
        for j in range(start, len(plan)):
            name = NAV_MANEUVER.get(plan[j][1])
            if name is not None:
                return name, round(dist, 1)
            if j + 1 < len(plan):
                dist += plan[j][0].location.distance(plan[j + 1][0].location)
        return None, None

    def _route_xy(self, behind_m: float = 10.0, ahead_m: float = 200.0):
        """The leaderboard's global plan near the ego as an ego-frame polyline (x forward,
        y left, metres): from the first plan point at least ``behind_m`` before the one
        nearest the ego to the first at least ``ahead_m`` after it. None without a plan.
        A policy that rasterises the route (AutoE2E's navigation tile) reads it; a
        polyline ending short of ``ahead_m`` is the route's end.

        The points are the plan's own, and the plan is sparse: the leaderboard hands
        ``set_global_plan`` a 1 m route and the base class keeps
        ``downsample_route(route, 50)`` of it, a point at each road-option change and
        otherwise every ~50 m. A straight segment between two of them cuts a bend by its
        sagitta, ~3 m at a 100 m radius."""
        plan = getattr(self, "_global_plan_world_coord", None)
        hero = self._hero_actor()
        if not plan or hero is None:
            return None
        hero_loc = hero.get_location()
        start = int(np.argmin([tf.location.distance(hero_loc) for tf, _opt in plan]))
        lo, acc = start, 0.0
        while lo > 0 and acc < behind_m:
            acc += plan[lo][0].location.distance(plan[lo - 1][0].location)
            lo -= 1
        hi, acc = start, 0.0
        while hi < len(plan) - 1 and acc < ahead_m:
            acc += plan[hi][0].location.distance(plan[hi + 1][0].location)
            hi += 1
        # CARLA world -> paddock world is the same y-flip `_rh` applies to the ego pose.
        world = np.array([[tf.location.x, -tf.location.y, 0.0, 1.0] for tf, _opt in plan[lo : hi + 1]])
        ego = (np.linalg.inv(self._ego2world()) @ world.T).T
        return np.round(ego[:, :2], 3).tolist()

    def _lane_change(self) -> tuple:
        """The next lane change on the dense route and the along-route metres to it.

        See ``navigation.lane_change_ahead``; (None, None) with no route or no
        lane change within its horizon.
        """
        route, hero = self._nav_route, self._hero_actor()
        if route is None or hero is None:
            return None, None
        loc = hero.get_location()
        index, _ = route.locate(loc.x, loc.y)
        return navigation.lane_change_ahead(route.options, route.s, index)

    def tick_inputs(self, input_data, timestamp, frames=True):
        """Everything one tick contributes to the wire body, with the camera frames still
        raw (RGB arrays), so a caller may swap a frame before ``infer`` encodes it. With
        ``frames`` false the cameras are left out: a tick that asks the node nothing."""
        us = int(timestamp * 1e6)
        # No speedometer is a broken sensor setup, not a stationary car: let the
        # KeyError surface, so the leaderboard records the route as crashed.
        speed = float(input_data["SPEED"][1]["speed"])
        if self._prev_us is None:
            accel = 0.0
        else:
            dt = (us - self._prev_us) / 1e6
            accel = (speed - self._prev_speed) / dt if dt > 0 else 0.0
        self._prev_speed, self._prev_us = speed, us

        images = {}
        front_bgr = None
        for cid in CAMS:
            if frames and cid in input_data:
                bgra = input_data[cid][1]
                images[cid] = bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        if RECORD_DIR and RECORD_CAM in input_data:
            front_bgr = input_data[RECORD_CAM][1][:, :, :3].copy()

        canbus = [0.0] * 18
        canbus[CANBUS_ACCEL] = accel
        canbus[CANBUS_VELOCITY] = speed

        # Posted road limit at the ego (CARLA reports km/h). Sent so a policy with a
        # cruise set-point can use the real limit instead of guessing one.
        hero = self._hero_actor()
        try:
            lim_kmh = float(hero.get_speed_limit()) if hero is not None else 0.0
        except Exception:
            lim_kmh = 0.0
        speed_limit = lim_kmh / 3.6 if lim_kmh > 0 else None
        nav_maneuver, nav_distance_m = self._nav()
        nav_lane_change, nav_lane_change_m = self._lane_change()
        return {
            "us": us,
            "speed": speed,
            "accel": accel,
            "images": images,
            "front_bgr": front_bgr,
            "canbus": canbus,
            "ego2world": self._ego2world().tolist(),
            "command": self._command(),
            "speed_limit": speed_limit,
            "nav_maneuver": nav_maneuver,
            "nav_distance_m": nav_distance_m,
            "nav_lane_change": nav_lane_change,
            "nav_lane_change_m": nav_lane_change_m,
            "route_xy": self._route_xy(),
        }

    def wire_body(self, inputs, override_images=None):
        """The /infer body for one tick. ``override_images`` replaces the raw frame of the
        cameras it names (segment mode) before encoding; everything else is untouched."""
        images = {}
        for cid, rgb in inputs["images"].items():
            if override_images and cid in override_images:
                rgb = override_images[cid]
            # raw .npy for the small frames the multi-camera models stack; JPEG (q95)
            # for multi-megapixel frames (recorder.wire_codec)
            images[cid] = encode_image(rgb, codec=recorder.wire_codec(rgb))
        return {
            "images": images,
            "ego2world": inputs["ego2world"],
            "canbus": inputs["canbus"],
            "timestamp": inputs["us"],
            "command": inputs["command"],
            "speed_limit": inputs["speed_limit"],
            "nav_maneuver": inputs["nav_maneuver"],
            "nav_distance_m": inputs["nav_distance_m"],
            "nav_lane_change": inputs["nav_lane_change"],
            "nav_lane_change_m": inputs["nav_lane_change_m"],
            "route_xy": inputs["route_xy"],
            "calibration": {
                "camera2image": self._cam2img,
                "camera2ego": self._cam2ego,
                "lidar2ego": np.eye(4).tolist(),  # lidar coincident with ego
            },
        }

    def infer(self, inputs, override_images=None):
        """POST one tick to the node and return its reply; see ``wire_body``."""
        return self._post(self.wire_body(inputs, override_images))

    def _post(self, body):
        r = self._http.post(f"{SERVER}/infer", json=body, timeout=INFER_TIMEOUT_S)
        r.raise_for_status()
        return r.json()

    def decide(self, payload, speed):
        """Node reply -> (control, desired_speed, plan, native). Validated here: a NaN
        from a mis-warmed model is an error, not a steer value CARLA accepts silently.

        ``plan`` is what the controller follows: the reply's whole plan (``plan``, every
        point at the model's own spacing) read at every WAYPOINT_DT s, six points
        (``pid.on_grid``). Those are the same six points the reply's ``trajectory``
        carries (``Trajectory.to_wire``). That field once held the plan's first six
        points whatever their spacing, which for alpamayo2edge's, 0.1 s apart, was 0.6 s
        of plan that read as 3 s would drive at a fifth of the planned speed; reading
        ``plan`` keeps the client right against a node that still sends that."""
        whole = payload["plan"]
        points = on_grid(whole["waypoints"], float(whole["dt"]), WAYPOINT_DT, N_WAYPOINTS)
        plan = Trajectory(points, dt=WAYPOINT_DT)
        native = (payload.get("extras") or {}).get("native_long")
        if native:
            control, desired_speed = self._control_native(plan.xy, speed, native)
        else:
            control, desired_speed = self._control(plan.xy, speed)
        return control, desired_speed, plan, native

    def run_step(self, input_data, timestamp):
        if self._seg is not None:
            control = self._seg.step(
                self, self._hero_actor(), CarlaDataProvider.get_world(), input_data, timestamp
            )
            if self._metric_path:
                self._record_metric_info()
            self._t += 1
            return control

        us = int(timestamp * 1e6)
        due = self._payload is None or INFER_HZ <= 0 or us - self._plan_us >= 1e6 / INFER_HZ - DUE_SLACK_US
        inputs = self.tick_inputs(input_data, timestamp, frames=due)
        speed, accel, front_bgr = inputs["speed"], inputs["accel"], inputs["front_bgr"]
        nav_maneuver, nav_distance_m = inputs["nav_maneuver"], inputs["nav_distance_m"]
        body = self.wire_body(inputs)

        if due:
            # A tick that sends no request has none to record.
            if RECORD_INPUTS and not self._inputs_failed:
                try:
                    recorder.write_inputs(
                        os.path.join(RECORD_DIR, recorder.INPUTS_DIR, f"{self._t:05d}.npz"),
                        inputs["images"],  # the frames as sent
                        {k: v for k, v in body.items() if k != "images"},
                    )
                except Exception as exc:  # for analysis only; the route goes on without it
                    self._inputs_failed = True
                    print("[paddock-b2d] inputs record stopped at tick", self._t, ":", repr(exc))

            sent_at = time.monotonic()
            self._payload = self._post(body)
            infer_ms = (time.monotonic() - sent_at) * 1000.0
            self._plan_us = us
        else:
            # No /infer this tick: the analysis record says so (`asked` false) and holds the
            # plan in force, the last reply, with no time spent asking (log.csv repeats its
            # waypoints likewise).
            infer_ms = 0.0
        payload = self._payload
        control, desired_speed, plan, native = self.decide(payload, speed)

        if self._metric_path:
            self._record_metric_info()

        if self._ticks is not None and not self._record_failed:
            sent = {
                "command": body["command"],
                "nav_maneuver": nav_maneuver,
                "nav_distance_m": nav_distance_m,
                "nav_lane_change": inputs["nav_lane_change"],
                "nav_lane_change_m": inputs["nav_lane_change_m"],
                "speed": speed,
                "accel": accel,
                "speed_limit": inputs["speed_limit"],
                # The leaderboard's sparse plan as the node got it (ego frame): a
                # model that rasterises it (AutoE2E) planned from these chords, not
                # from route_plan.json's dense 1 m route.
                "route_xy": body["route_xy"],
            }
            try:
                self._write_analysis_record(
                    input_data,
                    timestamp,
                    infer_ms,
                    sent,
                    payload,
                    control,
                    desired_speed,
                    asked=due,
                    plan=plan,
                )
            except Exception as exc:  # the record is for analysis; the route must go on without it
                self._record_failed = True
                print("[paddock-b2d] analysis record stopped at tick", self._t, ":", repr(exc))

        if self._log is not None and front_bgr is not None:
            cv2.imwrite(
                os.path.join(RECORD_DIR, "front", f"{self._t:05d}.jpg"),
                front_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            if "BEV" in input_data:
                cv2.imwrite(
                    os.path.join(RECORD_DIR, "bev", f"{self._t:05d}.jpg"),
                    input_data["BEV"][1][:, :, :3],
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
            logged = np.full((LOGGED_WAYPOINTS, 2), np.nan)
            head = plan.xy[:LOGGED_WAYPOINTS]
            logged[: len(head)] = head
            e2w = self._ego2world()  # ego world pose, for the traversed-path / failure-location footprint
            self._w.writerow(
                [
                    self._t,
                    round(speed, 3),
                    round(control.steer, 4),
                    round(control.throttle, 3),
                    round(control.brake, 3),
                    round(desired_speed, 3),
                    round(float(np.linalg.norm(plan.xy[-1])), 3),
                ]
                + [round(float(v), 3) for v in logged.flatten()]
                + [
                    round(float(e2w[0, 3]), 3),
                    round(float(e2w[1, 3]), 3),
                    round(float(math.atan2(e2w[1, 0], e2w[0, 0])), 4),
                    # 1 while the driver-resume emulation is steering the speed
                    # (openpilot native mode only): the "nudge" that marks a route *
                    1 if (native and "+resume" in str(native.get("source", ""))) else 0,
                    # the route instruction the node was given this tick (Alpamayo nav text)
                    nav_maneuver or "",
                    "" if nav_distance_m is None else nav_distance_m,
                ]
            )
            self._log.flush()
        if self._early_stop is not None or (STOP_REQUEST and not self._stop_requested):
            self._check_early_stop(timestamp)
        self._t += 1
        return control

    def _check_early_stop(self, game_time):
        """Feed this tick to ``early_stop``; when a rule fires, or a stop is requested, end the route.

        After the tick's control is decided and recorded, so a rule changes nothing
        about a drive until the tick it fires on (``early_stop.replay`` relies on it).
        """
        route, hero = self._nav_route, self._hero_actor()
        along = None
        if route is not None and hero is not None:
            loc = hero.get_location()
            index, _ = route.locate(loc.x, loc.y)
            along = float(route.s[index])
        if STOP_REQUEST and not self._stop_requested and os.path.exists(STOP_REQUEST):
            self._stop_requested = True
            self._end_early(early_stop.MANUAL, game_time, along)
            return
        if self._early_stop is None or along is None:
            return
        self._stop_record.update(
            game_time_s=round(float(game_time), 2), tick=self._t, route_m=round(along, 1)
        )
        # Waiting at a red light is not being stuck: inside its trigger box, or short of the
        # next light on the route (early_stop.waiting_at_light).
        try:
            ahead = recorder.next_on_route(self._stop_lights, index, route)
            if ahead is not None:
                light = self._lights_by_id.get(ahead["id"])
                ahead["state"] = str(light.get_state()) if light is not None else None
            waiting = early_stop.waiting_at_light(
                bool(hero.is_at_traffic_light()), str(hero.get_traffic_light_state()), ahead
            )
        except RuntimeError:
            waiting = False
        collisions = (early_stop.scenario_collisions() or 0) if self._early_stop.collision else 0
        reason = self._early_stop.update(game_time, along, collisions=collisions, waiting=waiting)
        if reason is not None:
            self._end_early(reason, game_time, along)

    def _end_early(self, reason, game_time, along):
        """Have the leaderboard end the route now, scored as it stands, and say why in early_stop.json."""
        self._early_stop = None  # once: the manager ends the route before calling us again
        if not early_stop.end_route():
            print(f"[paddock-b2d] early stop ({reason}): no ScenarioManager to end the route; driving on")
            return
        where = "" if along is None else f", {along:.0f} m along"
        print(f"[paddock-b2d] early stop at {game_time:.2f} s{where}: {reason}", flush=True)
        if self._stop_record is None:  # a stop request with no rule on
            self._stop_record = {"reason": None}
        self._stop_record.update(reason=reason, game_time_s=round(float(game_time), 2), tick=self._t)
        if along is not None:
            self._stop_record["route_m"] = round(along, 1)
        self._write_stop_record()

    def _write_stop_record(self):
        """This repetition's verdict, at the top of the route's early_stop.json and added to its
        ``repetitions``: the evaluator builds a new agent for every repetition, and the harness
        averages them (``ended_early``). The top stays the latest, which the videos read."""
        if not STOP_RECORD or self._stop_record is None:
            return
        try:
            with open(STOP_RECORD) as handle:
                earlier = json.load(handle).get("repetitions") or []
        except (OSError, ValueError, AttributeError):  # the first repetition (the harness clears it)
            earlier = []
        try:
            with open(STOP_RECORD, "w") as handle:
                json.dump({**self._stop_record, "repetitions": [*earlier, dict(self._stop_record)]}, handle)
        except OSError as exc:  # the route still ends; the harness just cannot say why
            print("[paddock-b2d] early stop record not written:", repr(exc))

    def _control(self, wps, speed):
        """Waypoints (ego frame, x fwd, y LEFT) -> CARLA control via the dual-PID."""
        steer, throttle, brake, desired_speed = self._tracker.control(wps, speed)
        c = carla.VehicleControl()
        c.steer = steer
        c.throttle = throttle
        c.brake = brake
        return c, desired_speed

    def _control_native(self, wps, speed, native):
        """Model-native longitudinal control (openpilot), structured like its
        LongControl: steer from the standard tracker on the plan positions; the
        speed PID (TCP's proven actuator loop) is driven by the model's own
        v_target at action_t instead of the tracker's 4 s mean; braking is the
        model's desired deceleration (proportional), and should_stop holds the
        car -- no all-or-nothing brake below 0.4 m/s, creep allowed."""
        from paddock.adapters.benchmarks.bench2drive.pid import CLIP_DELTA, MAX_THROTTLE

        # Steering only: the native path drives the speed PID itself just below,
        # so calling the full tracker here would step that PID twice per tick.
        steer = self._tracker.steer_only(wps)
        a = float(native.get("accel", 0.0))
        stop = bool(native.get("should_stop", False))
        v_target = float(native.get("v_target", speed + a * 0.5))
        if stop:
            throttle, brake = 0.0, 0.5  # hold, as openpilot does at a stop
        else:
            delta = float(np.clip(v_target - speed, 0.0, CLIP_DELTA))
            throttle = float(np.clip(self._tracker.speed.step(delta), 0.0, MAX_THROTTLE))
            brake = float(np.clip(-a / 5.0, 0.0, 1.0)) if a < -0.1 else 0.0
            if brake > 0.0:
                throttle = 0.0
        c = carla.VehicleControl()
        c.steer = steer
        c.throttle = throttle
        c.brake = brake
        return c, v_target
