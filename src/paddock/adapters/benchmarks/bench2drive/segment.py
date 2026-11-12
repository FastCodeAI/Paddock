"""Segment mode for the Bench2Drive agent: replay a recorded run to T=0, then run the model closed
loop with every camera frame converted, tick by tick:

    carla_frame_t (+ depth_t + wsm_t) -> converted_frame_t -> model -> action -> CARLA moves -> t+1

Nothing is predicted and nothing is held: the converter (Cosmos 3, resident in a service) turns
the frame of the tick that just happened into a real-looking one before the model sees it, and
CARLA waits (synchronous mode). Diagnostic only; never part of a standard benchmark run. Enabled
by ``PADDOCK_SEGMENT=<spec.json>``; every other run is untouched. Spec keys:

    mode        "live" (converted frames) | "control" (CARLA frames through the same path)
    poses       metric_info.json of the recorded run (per-tick CARLA location/rotation)
    t0          tick where the model takes over (the tick the ego crossed the scenario trigger)
    tp_ticks    past window: the model is shown converted frames for [t0 - tp_ticks, t0) while the
                car is still on the recorded rails, so its history/filters are warm at T=0
    cam         the model camera being converted (one id, e.g. CAM_FRONT)
    out_dir     frames + a per-tick csv go here
    wsm_url     map renderer service (Ludus): POST /init {map, cam}, POST /wsm {state} -> png
    cosmos_url  converter service: POST /frame {rgb, depth, wsm, tick, timestamp_us} ->
                {"frames": [png...], "ticks": [...]} (may return 0 frames while it collects its
                first clip, several once, then exactly one per call)

Timeline (20 Hz):
    [0, t_anchor)        replay by teleport (physics off); nothing shown to the model
    t_anchor             export the map -> wsm/init; spawn a depth camera on the world at the
                         model camera's mount/fov/resolution (the leaderboard forbids depth sensors)
    (t_anchor, t0)       replay continues; each tick's frame is converted and fed to the model with
                         the speed the recorded poses imply; plans are discarded
    t0                   handover: physics on, recorded velocity
    [t0, ...)            closed loop, one converted frame and one plan per tick

Python 3.8 (runs inside the CARLA client interpreter).
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import queue
import time

import carla
import numpy as np
import requests
from PIL import Image

from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY

ENV = "PADDOCK_SEGMENT"
DRIVABLE = (
    carla.LaneType.Driving,
    carla.LaneType.Shoulder,
    carla.LaneType.Parking,
    carla.LaneType.Bidirectional,
)
HZ = 20.0


def _category(type_id):
    t = type_id.lower()
    if t.startswith("walker"):
        return "pedestrian"
    if any(
        k in t
        for k in (
            "bike",
            "crossbike",
            "omafiets",
            "diamondback",
            "gazelle",
            "vespa",
            "yamaha",
            "harley",
            "kawasaki",
            "ninja",
            "low_rider",
            "zx125",
        )
    ):
        return "cyclist"
    if any(
        k in t
        for k in (
            "truck",
            "bus",
            "carlacola",
            "firetruck",
            "ambulance",
            "sprinter",
            "volkswagen.t2",
            "european_hgv",
            "fuso",
            "mitsubishi",
        )
    ):
        return "truck"
    return "car"


def _edge_pt(b, side):
    t = b.transform
    r = t.get_right_vector()
    loc = t.location
    return [
        loc.x + side * r.x * b.lane_width / 2,
        loc.y + side * r.y * b.lane_width / 2,
        loc.z + 0.03,
    ]


def _outermost(wp, side):
    cur = wp
    while True:
        n = cur.get_right_lane() if side > 0 else cur.get_left_lane()
        if n is None or n.lane_type not in DRIVABLE or n.lane_id * wp.lane_id < 0:
            return cur
        cur = n


def export_map_layers(world, center, radius):
    """The static layers the Ludus renderer draws (CARLA coords)."""
    m = world.get_map()

    def near(loc):
        return loc.distance(center) <= radius

    lane_lines, boundaries, wait_lines = [], [], []
    seen_b = set()
    for a in m.generate_waypoints(2):
        if not near(a.transform.location):
            continue
        nxt = a.next(2)
        if not nxt or nxt[0].road_id != a.road_id:
            continue
        b = nxt[0]
        if (not a.is_junction) and b.is_junction and a.lane_type == carla.LaneType.Driving:
            wait_lines.append({"points": [_edge_pt(a, -1), _edge_pt(a, 1)]})
        if a.lane_type not in DRIVABLE:
            continue
        for side, name in [(-1, "left"), (1, "right")]:
            marking = getattr(a, name + "_lane_marking")
            kind = str(marking.type)
            color = str(marking.color)
            if not kind.endswith("NONE") and a.lane_type == carla.LaneType.Driving:
                lane_lines.append(
                    {
                        "chain": [a.road_id, a.section_id, a.lane_id, side],
                        "s": a.s,
                        "points": [_edge_pt(a, side), _edge_pt(b, side)],
                        "style": "DASHED" if "Broken" in kind else "SOLID",
                        "color": "YELLOW" if "Yellow" in color else "WHITE",
                        "kind": kind,
                    }
                )
        for side in (-1, 1):
            o = _outermost(a, side)
            ob = _outermost(b, side)
            key = (a.road_id, a.section_id, side, round(a.s, 1))
            if key in seen_b:
                continue
            seen_b.add(key)
            boundaries.append(
                {
                    "chain": [a.road_id, a.section_id, o.lane_id, side],
                    "s": a.s,
                    "points": [_edge_pt(o, side), _edge_pt(ob, side)],
                }
            )
    cws, poly = [], []
    for p in m.get_crosswalks():
        pt = [p.x, p.y, p.z + 0.03]
        if poly and abs(pt[0] - poly[0][0]) < 1e-3 and abs(pt[1] - poly[0][1]) < 1e-3:
            if len(poly) >= 3 and near(carla.Location(x=poly[0][0], y=poly[0][1], z=poly[0][2])):
                cws.append(poly)
            poly = []
        else:
            poly.append(pt)
    lights, signs, poles = [], [], []
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        t = tl.get_transform()
        if not near(t.location):
            continue
        bb = tl.bounding_box
        cen = t.transform(carla.Location(x=bb.location.x, y=bb.location.y, z=bb.location.z))
        lights.append(
            {
                "center": [cen.x, cen.y, cen.z],
                "yaw": t.rotation.yaw,
                "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
            }
        )
        poles.append(
            {
                "points": [
                    [t.location.x, t.location.y, t.location.z],
                    [t.location.x, t.location.y, t.location.z + max(3.0, cen.z - t.location.z)],
                ]
            }
        )
    for lm in m.get_all_landmarks():
        t = lm.transform
        if not near(t.location) or lm.type == "1000001":
            continue
        signs.append(
            {
                "center": [t.location.x, t.location.y, t.location.z + 2.0],
                "yaw": t.rotation.yaw,
                "type": lm.type,
                "name": lm.name,
            }
        )
        poles.append(
            {
                "points": [
                    [t.location.x, t.location.y, t.location.z],
                    [t.location.x, t.location.y, t.location.z + 3.0],
                ]
            }
        )
    for lbl, dest in (("Poles", "poles"), ("TrafficSigns", "signs")):
        try:
            for o in world.get_environment_objects(getattr(carla.CityObjectLabel, lbl)):
                loc = o.transform.location
                if not near(loc):
                    continue
                e = o.bounding_box.extent
                if dest == "poles":
                    poles.append({"points": [[loc.x, loc.y, loc.z - e.z], [loc.x, loc.y, loc.z + e.z]]})
                else:
                    signs.append(
                        {
                            "center": [loc.x, loc.y, loc.z],
                            "yaw": o.transform.rotation.yaw,
                            "type": "env",
                            "name": o.name,
                        }
                    )
        except Exception as exc:  # large maps may report odd frames; layers are best effort
            print("[segment] env objects", lbl, "skipped:", exc)
    return {
        "lane_lines": lane_lines,
        "road_boundaries": boundaries,
        "wait_lines": wait_lines,
        "crosswalks": cws,
        "traffic_lights": lights,
        "traffic_signs": signs,
        "poles": poles,
    }


def actor_states(world, hero, radius):
    """Every vehicle/walker near the hero (the ego included, flagged): what the obstacle layer draws."""
    hero_loc = hero.get_location()
    out = []
    for actor in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.*")):
        tf = actor.get_transform()
        if tf.location.distance(hero_loc) > radius:
            continue
        box = actor.bounding_box
        center = tf.transform(carla.Location(x=box.location.x, y=box.location.y, z=box.location.z))
        out.append(
            {
                "id": actor.id,
                "type": actor.type_id,
                "category": "ego" if actor.id == hero.id else _category(actor.type_id),
                "matrix": tf.get_matrix(),
                "center": [center.x, center.y, center.z],
                "extent": [box.extent.x, box.extent.y, box.extent.z],
                "yaw": tf.rotation.yaw,
            }
        )
    return out


def _depth_to_inverse(dm):
    raw = np.frombuffer(dm.raw_data, np.uint8).reshape(dm.height, dm.width, 4).astype(np.float64)
    depth_m = (raw[:, :, 2] + raw[:, :, 1] * 256.0 + raw[:, :, 0] * 65536.0) / 16777215.0 * 1000.0
    inv = 1.0 / np.maximum(depth_m, 0.5)
    inv = inv / np.percentile(inv, 99.5)
    return (np.clip(inv, 0, 1) ** 0.6 * 255).astype(np.uint8)


def _b64png(arr):
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _decode(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


class Segment:
    def __init__(self, spec_path, mounts, model_res, model_fovs):
        """``mounts`` / ``model_res`` / ``model_fovs``: dicts keyed by camera id for every camera the
        model declares; the spec's ``cams`` (or single ``cam``) picks the ones to convert."""
        self.spec = json.load(open(spec_path))
        self.mode = self.spec["mode"]
        self.t0 = int(self.spec.get("t0", 0))
        self.tp = int(self.spec.get("tp_ticks", 0))
        self.t_anchor = self.t0 - self.tp
        # segment length after T0 (ticks); 0 = until the leaderboard ends the route. Past it the ego is
        # placed on the recording's final pose so the route completes and the run ends quickly.
        self.t_end = int(self.spec.get("t_end_ticks", 0))
        self.cams = list(self.spec.get("cams") or [self.spec["cam"]])
        self.cam = self.cams[0]  # the camera used for the pose/log columns
        self.mounts = {c: mounts[c] for c in self.cams}
        self.res = {c: tuple(model_res[c]) for c in self.cams}
        self.fovs = {c: float(model_fovs[c]) for c in self.cams}
        self.cosmos_urls = self.spec.get("cosmos_urls") or {c: self.spec.get("cosmos_url") for c in self.cams}
        self.out = self.spec["out_dir"]
        os.makedirs(self.out, exist_ok=True)
        mi = json.load(open(self.spec["poses"]))
        ticks = sorted(mi, key=lambda k: int(k))
        self.poses = []
        for t in ticks:
            loc, rot = mi[t]["location"], mi[t]["rotation"]
            self.poses.append(
                carla.Transform(
                    carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                    carla.Rotation(roll=rot[0], pitch=rot[1], yaw=rot[2]),
                )
            )
        self.n_poses = len(self.poses)
        if self.mode == "capture":
            # open-loop capture: replay EVERY recorded pose, never hand over; from tick 0 save each
            # camera's RGB,
            # inverse depth, camera pose, the actor boxes and the model inputs, so the whole recording can be
            # converted offline and replayed through the model with no CARLA in the loop.
            self.t0, self.tp, self.t_anchor, self.t_end = self.n_poses, self.n_poses, 0, 0
        self.radius = float(self.spec.get("radius_m", 300))
        self._depth = {}  # cam -> (sensor, queue)
        self._held = None  # newest node reply
        self._n_converted = 0
        self._log = open(os.path.join(self.out, "segment_log.csv"), "w", newline="")
        self._w = csv.writer(self._log)
        self._w.writerow(
            [
                "tick",
                "phase",
                "speed",
                "steer",
                "throttle",
                "brake",
                "v_des",
                "ego_x",
                "ego_y",
                "ego_yaw",
                "wsm_ms",
                "cosmos_ms",
                "frames_returned",
            ]
        )
        for c in self.cams:
            for sub in ("carla", "depth", "wsm", "model"):
                os.makedirs(os.path.join(self.out, "frames", c, sub), exist_ok=True)
        rig = {c: (self.res[c], self.fovs[c]) for c in self.cams}
        print(
            f"[segment] v3 mode={self.mode} t0={self.t0} tp={self.tp} "
            f"(anchor tick {self.t_anchor}) poses={self.n_poses} cams={rig}",
            flush=True,
        )

    # ---------------------------------------------------------------- replay
    def replay(self, hero, t):
        if hero is None or t >= self.n_poses:
            return
        hero.set_simulate_physics(False)
        hero.set_transform(self.poses[t])

    def replay_speed(self, t):
        if t <= 0 or t >= self.n_poses:
            return 0.0
        a, b = self.poses[t - 1].location, self.poses[t].location
        return math.hypot(b.x - a.x, b.y - a.y) * HZ

    def handover(self, hero):
        p0, p1 = self.poses[max(0, self.t0 - 1)], self.poses[self.t0]
        v = carla.Vector3D((p1.location.x - p0.location.x) * HZ, (p1.location.y - p0.location.y) * HZ, 0.0)
        hero.set_transform(p1)
        hero.set_simulate_physics(True)
        hero.set_target_velocity(v)
        hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        print(f"[segment] handover at tick {self.t0}, v={math.hypot(v.x, v.y):.2f} m/s", flush=True)

    # ------------------------------------------------------------ services
    def _post(self, url, path, payload, timeout=900):
        r = requests.post(url + path, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def start_services(self, world, hero):
        """At t_anchor: map -> wsm service; depth camera at the model camera (for the converter)."""
        m = export_map_layers(world, hero.get_location(), self.radius)
        print("[segment] map layers:", {k: len(v) for k, v in m.items()}, flush=True)
        if self.mode == "capture":
            json.dump(
                {
                    "map": m,
                    "cams": [
                        {
                            "id": c,
                            "w": self.res[c][0],
                            "h": self.res[c][1],
                            "fov": self.fovs[c],
                            "mount": list(self.mounts[c]),
                        }
                        for c in self.cams
                    ],
                },
                open(os.path.join(self.out, "map.json"), "w"),
            )
        if self.mode == "live":
            cams = [
                {"id": c, "w": self.res[c][0], "h": self.res[c][1], "fov": self.fovs[c]} for c in self.cams
            ]
            info = self._post(self.spec["wsm_url"], "/init", {"map": m, "cams": cams})
            print("[segment] wsm init:", info, flush=True)
        if self.mode in ("live", "capture"):
            for c in self.cams:
                x, y, z, roll, pitch, yaw = self.mounts[c]
                bp = world.get_blueprint_library().find("sensor.camera.depth")
                for k, v in [
                    ("image_size_x", str(self.res[c][0])),
                    ("image_size_y", str(self.res[c][1])),
                    ("fov", str(self.fovs[c])),
                    ("sensor_tick", "0"),
                ]:
                    bp.set_attribute(k, v)
                sensor = world.spawn_actor(
                    bp,
                    carla.Transform(
                        carla.Location(x=x, y=y, z=z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
                    ),
                    attach_to=hero,
                )
                q = queue.Queue()
                sensor.listen(q.put)
                self._depth[c] = (sensor, q)

    def _convert_cam(self, ci, c, t, us, rgb, depth, actors):
        """One camera through the map renderer and its converter.

        Returns (cam, [(tick, frame)], wsm_ms, cosmos_ms).
        """
        state = {
            "timestamp_us": us,
            "camera_to_world": self._depth[c][0].get_transform().get_matrix(),
            "actors": actors,
            "cam": ci,
        }
        t0 = time.monotonic()
        wsm = _decode(self._post(self.spec["wsm_url"], "/wsm", state)["wsm"])
        t1 = time.monotonic()
        rep = self._post(
            self.cosmos_urls[c],
            "/frame",
            {
                "tick": t,
                "timestamp_us": us,
                "rgb": _b64png(rgb),
                "depth": _b64png(depth),
                "wsm": _b64png(wsm),
            },
        )
        t2 = time.monotonic()
        Image.fromarray(depth).save(os.path.join(self.out, "frames", c, "depth", f"{t:05d}.png"))
        Image.fromarray(wsm).save(os.path.join(self.out, "frames", c, "wsm", f"{t:05d}.jpg"), quality=90)
        w, h = self.res[c]
        frames = []
        # zip(strict=) is 3.10+; this module runs in the CARLA client's Python 3.8. The service
        # returns the two lists pairwise by construction.
        for tk, fb in zip(rep["ticks"], rep["frames"]):  # noqa: B905
            f = _decode(fb)
            if f.shape[0] != h or f.shape[1] != w:
                f = np.array(Image.fromarray(f).resize((w, h), Image.LANCZOS))
            frames.append((int(tk), f))
        return c, frames, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    def convert(self, world, hero, t, timestamp, rgbs):
        """All converted cameras for one tick (in parallel).

        Returns ({cam: [(tick, frame)]}, wsm_ms, cosmos_ms).
        """
        us = int(timestamp * 1e6)
        if self.mode == "control":
            return {c: [(t, rgbs[c])] for c in self.cams}, 0.0, 0.0
        depths = {c: _depth_to_inverse(self._depth[c][1].get(timeout=30)) for c in self.cams}
        actors = actor_states(world, hero, self.radius)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(self.cams)) as ex:
            results = list(
                ex.map(
                    lambda ci_c: self._convert_cam(
                        ci_c[0], ci_c[1], t, us, rgbs[ci_c[1]], depths[ci_c[1]], actors
                    ),
                    enumerate(self.cams),
                )
            )
        out = {c: frames for c, frames, _, _ in results}
        self._n_converted += sum(len(f) for f in out.values())
        return out, max(r[2] for r in results), max(r[3] for r in results)

    def _by_tick(self, per_cam):
        """{cam: [(tick, frame)]} -> [(tick, {cam: frame})] for the ticks every camera returned."""
        ticks = None
        for c in self.cams:
            tk = [t for t, _ in per_cam[c]]
            ticks = tk if ticks is None else [t for t in ticks if t in tk]
        return [(tk, {c: dict(per_cam[c])[tk] for c in self.cams}) for tk in (ticks or [])]

    # ------------------------------------------------------------------ main
    def step(self, agent, hero, world, input_data, timestamp):
        """One leaderboard tick. Returns the VehicleControl to apply."""
        t = agent._t
        e2w = agent._ego2world()
        pose = (
            round(float(e2w[0, 3]), 3),
            round(float(e2w[1, 3]), 3),
            round(float(math.atan2(e2w[1, 0], e2w[0, 0])), 4),
        )
        speed = float(input_data["SPEED"][1]["speed"])

        if self.mode == "capture" and t >= self.n_poses:
            # every recorded pose captured: hold the ego on the final pose until the
            # leaderboard closes the route
            hero.set_simulate_physics(False)
            hero.set_transform(self.poses[-1])
            if t == self.n_poses:
                print(f"CAPTURE_END tick {t}: all {self.n_poses} recorded poses captured", flush=True)
            self._w.writerow([t, "capture-end", 0, 0, 0, 0, "", pose[0], pose[1], pose[2], "", "", ""])
            self._log.flush()
            return carla.VehicleControl()

        if t < self.t0:
            self.replay(hero, t)
            if t == self.t_anchor:
                self.start_services(world, hero)
            if t <= self.t_anchor:
                self._w.writerow(
                    [t, "replay", round(speed, 3), 0, 0, 0, "", pose[0], pose[1], pose[2], "", "", ""]
                )
                self._log.flush()
                return carla.VehicleControl()
            inputs = agent.tick_inputs(input_data, timestamp)
            rgbs = {c: inputs["images"][c] for c in self.cams}
            if self.mode == "capture":
                if t >= self.n_poses - 1:
                    hero.set_simulate_physics(False)
                    hero.set_transform(self.poses[self.n_poses - 1])
                depths = {c: _depth_to_inverse(self._depth[c][1].get(timeout=30)) for c in self.cams}
                for c in self.cams:
                    Image.fromarray(rgbs[c]).save(
                        os.path.join(self.out, "frames", c, "carla", f"{t:05d}.jpg"), quality=92
                    )
                    Image.fromarray(depths[c]).save(
                        os.path.join(self.out, "frames", c, "depth", f"{t:05d}.png")
                    )
                with open(os.path.join(self.out, "ticks.jsonl"), "a") as tf:
                    tf.write(
                        json.dumps(
                            {
                                "tick": t,
                                "timestamp_us": int(timestamp * 1e6),
                                "speed": self.replay_speed(t),
                                "ego2world": inputs["ego2world"],
                                "canbus": list(inputs["canbus"]),
                                "command": inputs["command"],
                                "nav_maneuver": inputs["nav_maneuver"],
                                "nav_distance_m": inputs["nav_distance_m"],
                                "speed_limit": inputs.get("speed_limit"),
                                "camera_to_world": {
                                    c: self._depth[c][0].get_transform().get_matrix() for c in self.cams
                                },
                                "actors": actor_states(world, hero, self.radius),
                            }
                        )
                        + "\n"
                    )
                self._w.writerow(
                    [
                        t,
                        "capture",
                        round(self.replay_speed(t), 3),
                        0,
                        0,
                        0,
                        "",
                        pose[0],
                        pose[1],
                        pose[2],
                        "",
                        "",
                        "",
                    ]
                )
                self._log.flush()
                return carla.VehicleControl()
            # past window: the car is on rails, the model watches converted frames and warms up
            per_cam, wsm_ms, cosmos_ms = self.convert(world, hero, t, timestamp, rgbs)
            for c in self.cams:
                Image.fromarray(rgbs[c]).save(
                    os.path.join(self.out, "frames", c, "carla", f"{t:05d}.jpg"), quality=90
                )
            frames = self._by_tick(per_cam)
            for (
                tk,
                f,
            ) in frames:  # 0 while the converters collect their first clip, then a burst, then one per tick
                inp = dict(inputs)
                inp["us"] = int((timestamp - (t - tk) / HZ) * 1e6)
                v_rep = self.replay_speed(tk)
                inp["speed"] = v_rep
                inp["canbus"] = list(inputs["canbus"])
                inp["canbus"][CANBUS_VELOCITY] = v_rep
                inp["canbus"][CANBUS_ACCEL] = (v_rep - self.replay_speed(tk - 1)) * HZ
                self._held = agent.infer(inp, f)
                for c in self.cams:
                    Image.fromarray(f[c]).save(
                        os.path.join(self.out, "frames", c, "model", f"{tk:05d}.jpg"), quality=90
                    )
            self._w.writerow(
                [
                    t,
                    "past-" + self.mode,
                    round(self.replay_speed(t), 3),
                    0,
                    0,
                    0,
                    "",
                    pose[0],
                    pose[1],
                    pose[2],
                    round(wsm_ms),
                    round(cosmos_ms),
                    len(frames),
                ]
            )
            self._log.flush()
            return carla.VehicleControl()

        if t == self.t0:
            self.handover(hero)
        if self.t_end and t >= self.t0 + self.t_end:
            # segment over: walk the ego to the recorded end pose so the route completes
            if t == self.t0 + self.t_end:
                print(
                    f"SEGMENT_END tick {t} (T0 + {self.t_end}): ending the route via the recorded final pose",
                    flush=True,
                )
            hero.set_simulate_physics(False)
            hero.set_transform(
                self.poses[min(self.n_poses - 1, self.n_poses - 1 - max(0, 40 - (t - self.t0 - self.t_end)))]
            )
            self._w.writerow(
                [t, "segment-end", round(speed, 3), 0, 0, 0, "", pose[0], pose[1], pose[2], "", "", ""]
            )
            self._log.flush()
            return carla.VehicleControl()
        inputs = agent.tick_inputs(input_data, timestamp)
        rgbs = {c: inputs["images"][c] for c in self.cams}
        per_cam, wsm_ms, cosmos_ms = self.convert(world, hero, t, timestamp, rgbs)
        for c in self.cams:
            Image.fromarray(rgbs[c]).save(
                os.path.join(self.out, "frames", c, "carla", f"{t:05d}.jpg"), quality=90
            )
        frames = self._by_tick(per_cam)
        if not frames:
            raise RuntimeError(
                f"converter returned no frame at tick {t} (past window too short for its first clip?)"
            )
        for tk, f in frames:
            self._held = agent.infer(
                inputs if tk == t else dict(inputs, us=int((timestamp - (t - tk) / HZ) * 1e6)), f
            )
            for c in self.cams:
                Image.fromarray(f[c]).save(
                    os.path.join(self.out, "frames", c, "model", f"{tk:05d}.jpg"), quality=90
                )
        control, v_des, plan, _native = agent.decide(self._held, inputs["speed"])
        # the model's plan for this tick (waypoints, ego frame x fwd / y left) as the controller
        # read it (agent.decide) -- kept so offline analysis can see what the model intended,
        # not only what the car did
        with open(os.path.join(self.out, "plans.jsonl"), "a") as pf:
            pf.write(
                json.dumps(
                    {
                        "tick": t,
                        "timestamp_us": inputs["us"],
                        "speed": inputs["speed"],
                        "ego2world": inputs["ego2world"],
                        "trajectory": plan.xy.tolist(),
                        "extras": self._held.get("extras"),
                    }
                )
                + "\n"
            )
        self._w.writerow(
            [
                t,
                self.mode,
                round(speed, 3),
                round(control.steer, 4),
                round(control.throttle, 3),
                round(control.brake, 3),
                round(v_des, 3),
                pose[0],
                pose[1],
                pose[2],
                round(wsm_ms),
                round(cosmos_ms),
                len(frames),
            ]
        )
        self._log.flush()
        return control

    def close(self):
        for sensor, _ in self._depth.values():
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        self._log.close()
