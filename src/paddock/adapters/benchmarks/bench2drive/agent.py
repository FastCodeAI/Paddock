#!/usr/bin/env python
"""Model-agnostic Bench2Drive / CARLA-leaderboard AutonomousAgent for paddock.

The model runs in its own env behind paddock's HTTP policy node; this agent is
the CARLA *client*. Each tick it renders the cameras the model declares, packs
them into the paddock wire format, POSTs ``/infer``, receives waypoints (ego
frame, x forward, y LEFT) and turns them into a CARLA ``VehicleControl`` via
the Bench2Drive dual-PID tracker vendored in ``pid.py``.

One agent serves every model: which cameras to render and which node to call
come from the environment, set by ``harness.run_episode``:

    PADDOCK_NODE_URL      policy node base url
    PADDOCK_CAMERAS       comma-separated camera ids       (default CAM_FRONT)
    PADDOCK_WAYPOINT_DT   spacing of the returned waypoints, s   (default 0.5)
    PADDOCK_RECORD_DIR    if set, write front/BEV frames + a per-tick log.csv
                          here; the harness sets SAVE_PATH alongside it so the
                          leaderboard allows the BEV camera's 30 m mount
    PADDOCK_METRICS_DIR   if set, write <dir>/<save_name>/metric_info.json: the
                          per-tick ego kinematics Bench2Drive's
                          efficiency_smoothness_benchmark.py scores. The
                          leaderboard hands the agent its `save_name` by
                          appending "+<save_name>" to the agent-config string
                          (leaderboard_evaluator.py:363 at 7ec25d1)

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

import carla
import cv2
import numpy as np
import requests
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from paddock.adapters.benchmarks.bench2drive.pid import WaypointTracker
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command, Trajectory, encode_image

SERVER = os.environ.get("PADDOCK_NODE_URL", "http://127.0.0.1:9000")
CAMS = [c.strip() for c in os.environ.get("PADDOCK_CAMERAS", "CAM_FRONT").split(",") if c.strip()]
WAYPOINT_DT = float(os.environ.get("PADDOCK_WAYPOINT_DT", "0.5"))
RECORD_DIR = os.environ.get("PADDOCK_RECORD_DIR") or None
METRICS_DIR = os.environ.get("PADDOCK_METRICS_DIR") or None

# Placement (x, y, z, roll, pitch, yaw) in the CARLA ego frame for every nuScenes
# camera a model might ask for; radius < 3m (the leaderboard sensor limit).
CAM_TF = {
    "CAM_FRONT": (1.5, 0.0, 1.5, 0.0, 0.0, 0.0),
    "CAM_FRONT_LEFT": (1.0, -0.5, 1.5, 0.0, 0.0, -55.0),
    "CAM_FRONT_RIGHT": (1.0, 0.5, 1.5, 0.0, 0.0, 55.0),
    "CAM_BACK": (-1.5, 0.0, 1.5, 0.0, 0.0, 180.0),
    "CAM_BACK_LEFT": (0.0, -0.6, 1.5, 0.0, 0.0, -110.0),
    "CAM_BACK_RIGHT": (0.0, 0.6, 1.5, 0.0, 0.0, 110.0),
}
CAM_FOV = {
    "CAM_FRONT": 70.0,
    "CAM_FRONT_LEFT": 70.0,
    "CAM_FRONT_RIGHT": 70.0,
    "CAM_BACK": 110.0,
    "CAM_BACK_LEFT": 70.0,
    "CAM_BACK_RIGHT": 70.0,
}
# One resolution for every camera: model feature builders np.stack the images, so
# unequal sizes raise. Half-res (vs 1600x900) is a big render + payload win and the
# front is cropped/resized downstream anyway.
CAM_RES = (800, 450)

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

F = np.diag([1.0, -1.0, 1.0, 1.0])  # y-flip: CARLA (left-handed) <-> paddock (right-handed)

# Body (x-fwd, y-left, z-up) -> camera-optical (x-right, y-down, z-fwd). paddock's
# camera2ego convention is the optical frame (as nuScenes/neuroncap supply); without
# this the geometric warp openpilot builds is rotated 90 deg and it never sees the
# road (derived mounting rpy [-90,0,90] instead of [0,0,0]). Right-multiplied onto
# each camera's ego-frame rotation so mounting yaw/pitch are preserved.
R_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

#: Columns of waypoints written to log.csv when recording.
LOGGED_WAYPOINTS = 6


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
        self._hero = None
        self._tracker = WaypointTracker(WAYPOINT_DT)
        # One connection to the loopback node for the whole route.
        self._http = requests.Session()
        self._cam2ego = {}
        self._cam2img = {}
        w, h = CAM_RES
        for cid in CAMS:
            x, y, z, roll, pitch, yaw = CAM_TF[cid]
            tf = carla.Transform(
                carla.Location(x=x, y=y, z=z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
            )
            m = _rh(tf.get_matrix())
            m[:3, :3] = m[:3, :3] @ R_OPT  # body -> camera-optical (paddock convention)
            self._cam2ego[cid] = m.tolist()
            self._cam2img[cid] = _intrinsics(w, h, CAM_FOV[cid])
        try:
            self._http.post(f"{SERVER}/reset", json={}, timeout=30)
        except Exception as exc:  # a fresh node has nothing to reset; not fatal
            print("[paddock-b2d] reset warning:", exc)
        self._log = None
        if RECORD_DIR:
            os.makedirs(os.path.join(RECORD_DIR, "front"), exist_ok=True)
            os.makedirs(os.path.join(RECORD_DIR, "bev"), exist_ok=True)
            self._log = open(os.path.join(RECORD_DIR, "log.csv"), "w", newline="")
            self._w = csv.writer(self._log)
            self._w.writerow(
                ["frame", "speed", "steer", "throttle", "brake", "v_des", "path_end"]
                + [f"wp{i}{ax}" for i in range(LOGGED_WAYPOINTS) for ax in ("x", "y")]
            )

    def destroy(self):
        if self._log is not None:
            self._log.close()
            self._log = None
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
        w, h = CAM_RES
        s = []
        for cid in CAMS:
            x, y, z, roll, pitch, yaw = CAM_TF[cid]
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
                    "fov": CAM_FOV[cid],
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

    def run_step(self, input_data, timestamp):
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
            if cid in input_data:
                bgra = input_data[cid][1]
                rgb = bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
                images[cid] = encode_image(rgb)
                if cid == "CAM_FRONT" and RECORD_DIR:
                    front_bgr = bgra[:, :, :3].copy()

        canbus = [0.0] * 18
        canbus[CANBUS_ACCEL] = accel
        canbus[CANBUS_VELOCITY] = speed

        body = {
            "images": images,
            "ego2world": self._ego2world().tolist(),
            "canbus": canbus,
            "timestamp": us,
            "command": self._command(),
            "calibration": {
                "camera2image": self._cam2img,
                "camera2ego": self._cam2ego,
                "lidar2ego": np.eye(4).tolist(),  # lidar coincident with ego
            },
        }

        r = self._http.post(f"{SERVER}/infer", json=body, timeout=120)
        r.raise_for_status()
        # Validated here: a NaN from a mis-warmed model is an error, not a steer
        # value CARLA accepts silently.
        plan = Trajectory(r.json()["trajectory"], dt=WAYPOINT_DT)
        control, desired_speed = self._control(plan.xy, speed)

        if self._metric_path:
            self._record_metric_info()

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
            )
            self._log.flush()
        self._t += 1
        return control

    def _control(self, wps, speed):
        """Waypoints (ego frame, x fwd, y LEFT) -> CARLA control via the dual-PID."""
        steer, throttle, brake, desired_speed = self._tracker.control(wps, speed)
        c = carla.VehicleControl()
        c.steer = steer
        c.throttle = throttle
        c.brake = brake
        return c, desired_speed
