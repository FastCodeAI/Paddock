"""HUGSIM <-> paddock bridge: the AD client HUGSIM talks to, driving any paddock model node, command only.

Runs in HUGSIM's interpreter, started by ``launcher.py`` once per scenario. It imports the wire contract from
``paddock.policy.protocol`` (``REPO_ROOT/src`` is on its path) rather than restating it.

HUGSIM side (upstream ``closed_loop.py``): each 0.25 s step the simulator writes ``pickle((obs, info))`` to
``<output>/obs_pipe`` and reads ``pickle(plan)`` from ``<output>/plan_pipe``; plan = (N, 2) waypoints at 0.5 s
from t + 0.5 s, in HUGSIM's "lidar" axes (x right, y forward) about HUGSIM's ego point. That ego point is the
*front camera's* calibrated pose (``hug_sim._get_obs``: c2w = ego @ v2front @ inv(v2c) @ cam_rect, so ego =
the front camera; the kinematic model moves it). ``pickle('Done')`` ends the episode.

Paddock side: one POST /infer per model call with the Observation wire body (``paddock.policy.protocol``):

* ``images`` -- the six rendered cameras, uint8 RGB, raw .npy (exact pixels). On Waymo and KITTI-360 HUGSIM
  renders the three BACK cameras black (it has no rear data there); a camera the run did not render
  (``render_cams``) is black too. Both are sent as rendered.
* ``camera2image`` -- K from the camera yaml's fovx / fovy / cx / cy (``sim_utils.create_cam``'s own
  construction).
* ``camera2ego`` -- optical camera -> vehicle, = inv(v2c) @ cam_rect: HUGSIM renders every camera cam_rect
  away from its calibrated pose, and the model must be told where the pixels were actually rendered from.
* ``lidar2ego`` -- the identity ("lidar = ego"), as the Bench2Drive client sends; see ``Rig.calibration``.
* ``ego2world`` -- the vehicle frame (nuScenes ego: x forward, y left, z up, rear axle on the ground; the
  cameras' v2c are given in it), yawed so x is HUGSIM's direction of travel (see ``Rig``), in HUGSIM's z-up
  "IMU" world (the frame of ``info['ego_box']`` / ``['obj_boxes']``).
* ``canbus`` -- speed (``info['ego_velo']``), acceleration from successive speeds, yaw rate from successive
  headings (as the Bench2Drive client: measured, not the controller's command).
* ``command`` -- ``info['command']``: HUGSIM 0 right, 1 left, 2 straight, which are paddock's ``Command``
  values (checked on route geometry). No ``route_xy``: command only.

The model's trajectory (paddock ego frame, rear axle, at the reply's own dt) is held in the world and
re-sampled each step at t_now + 0.5 k (k = 1..N) for the front-camera point, so a model called every
``--infer_every`` steps
(Drive-JEPA: 2, its native 2 Hz) is tracked between calls, as the Bench2Drive client does with a held plan.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import traceback

import numpy as np
import yaml

from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, CANBUS_YAW_RATE, encode_image

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
PLAN_DT = 0.5
#: Seconds per simulator step (upstream ``HUGSimEnv``; the kinematic model integrates at 0.25 s).
STEP_DT = 0.25
MIN_STEP_M = 0.3  # a waypoint step shorter than this carries no usable heading
#: Waypoints sent to HUGSIM at most: 0.5 s apart, so 4 s, NAVSIM's planning horizon (UniAD and VAD plan
#: 3 s). Below that, the model's own horizon, never padded. HUGSIM scores every step's sent plan (collision,
#: drivable area, TTC, comfort over its length), so a plan cut short is judged over less time than the model
#: planned: the reply's `plan` (the whole plan) is read, not its `trajectory`, which the wire cuts at six
#: waypoints for neuroNCAP.
N_MAX = 8


def plan_of(reply: dict) -> tuple[np.ndarray, float]:
    """The model's whole plan from an /infer reply, and its spacing.

    ``plan`` when the node sends it (every node since ``Trajectory.to_wire`` carried it), else
    ``trajectory``, which holds at most six waypoints.
    """
    plan = reply.get("plan") or {}
    if plan.get("waypoints"):
        return np.asarray(plan["waypoints"], dtype=np.float64)[:, :2], float(plan.get("dt") or PLAN_DT)
    return np.asarray(reply["trajectory"], dtype=np.float64), float(reply.get("dt") or PLAN_DT)


def euler_xyz_deg(rot_deg) -> np.ndarray:
    """``scipy.spatial.transform.Rotation.from_euler("XYZ", rot_deg, degrees=True).as_matrix()``, in numpy.

    Upper-case ``XYZ`` is scipy's *intrinsic* convention: R = Rx(a) @ Ry(b) @ Rz(c). HUGSIM's own ``rt2pose``
    builds every camera pose with that call; written out here so paddock's environment, which has no scipy,
    can import and test this module. ``tests/test_bridge.py`` pins it to scipy's matrix.
    """
    a, b, c = (math.radians(float(v)) for v in rot_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0], [-math.sin(b), 0.0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0.0], [math.sin(c), math.cos(c), 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def rt2pose(rot_deg, trans) -> np.ndarray:
    """``sim_utils.rt2pose``: XYZ Euler degrees + translation -> 4x4."""
    m = np.eye(4)
    m[:3, :3] = euler_xyz_deg(rot_deg)
    m[:3, 3] = trans
    return m


def se2(x: float, y: float, yaw: float) -> np.ndarray:
    return np.array([[math.cos(yaw), -math.sin(yaw), x], [math.sin(yaw), math.cos(yaw), y], [0.0, 0.0, 1.0]])


def infer_every(planning_hz: float | None) -> int:
    """Simulator steps between model calls: every step unless the model declares a slower native rate.

    HUGSIM steps at 4 Hz. Drive-JEPA replans at 2 Hz (``planning_hz: 2.0``), so every second step, the last
    plan held in between; a model that declares no rate is asked every step.
    """
    if not planning_hz or planning_hz <= 0:
        return 1
    return max(1, round(1.0 / (STEP_DT * planning_hz)))


def observes(native_input_hz: float | None, every: int) -> bool:
    """Whether the model is handed a frame on the steps between its plans (POST /observe).

    A model asked slower than the rate its history is built on (``native_input_hz``) is, so its history
    holds every frame HUGSIM renders (one each 0.25 s, the densest there is), not only the planning rate's.
    Qwen-Drive plans at 2 Hz and reads poses on a 0.1 s grid; asked every second step with nothing between,
    its history came at 2 Hz. A model asked every step has no step between.
    """
    return every > 1 and bool(native_input_hz) and float(native_input_hz) > 1.0 / (STEP_DT * every) + 1e-9


def step_action(step: int, have_plan: bool, every: int, observe: bool) -> str | None:
    """What the bridge asks the node at this step: "infer" (a plan), "observe" (a frame, no plan) or None."""
    if not have_plan or step % every == 0:
        return "infer"
    return "observe" if observe else None


class Rig:
    """Camera matrices for paddock, from HUGSIM's camera yaml (``load_camera_cfg``'s own reading of it)."""

    def __init__(self, camera_yaml: str) -> None:
        with open(camera_yaml) as fh:
            cfg = yaml.safe_load(fh)
        rect = np.eye(4)
        if "cam_rect" in cfg:
            rect[:3, :3] = euler_xyz_deg(cfg["cam_rect"]["rot"])
            rect[:3, 3] = cfg["cam_rect"]["trans"]
        self.camera2image, self.camera2ego = {}, {}
        for c in CAMS:
            cam = cfg["cams"][c]
            v2c = rt2pose(cam["extrinsics"]["v2c_rot"], cam["extrinsics"]["v2c_trans"])
            l2c = rt2pose(cam["extrinsics"]["l2c_rot"], cam["extrinsics"]["l2c_trans"])
            i = cam["intrinsics"]
            fx = i["W"] / (2 * math.tan(math.radians(i["fovx"]) / 2))
            fy = i["H"] / (2 * math.tan(math.radians(i["fovy"]) / 2))
            self.camera2image[c] = [[fx, 0.0, i["cx"]], [0.0, fy, i["cy"]], [0.0, 0.0, 1.0]]
            self.camera2ego[c] = np.linalg.inv(v2c) @ rect
            self.lidar2ego_raw = np.linalg.inv(v2c) @ l2c  # identical from every camera (one rig)
        # HUGSIM moves its ego point (the front camera) along the *camera's* heading, which is cam_yaw (~0.58
        # deg on nuScenes) left of the calibration's vehicle x axis. A model told the vehicle axis would plan
        # "straight" 0.58 deg right of HUGSIM's direction of travel on every call, and the held-plan loop
        # turns that into a steady right drift. So the vehicle frame sent to models is the calibration's,
        # yawed by cam_yaw about the rear axle: its x axis is HUGSIM's direction of travel. The whole rig
        # turns with it (camera2ego, lidar2ego), so every projection is unchanged.
        c2v = np.linalg.inv(
            rt2pose(
                cfg["cams"]["CAM_FRONT"]["extrinsics"]["v2c_rot"],
                cfg["cams"]["CAM_FRONT"]["extrinsics"]["v2c_trans"],
            )
        )
        self.cam_yaw = math.atan2(
            c2v[1, 2], c2v[0, 2]
        )  # optical axis heading in the calibration's vehicle frame
        rz = np.eye(4)
        rz[:2, :2] = [
            [math.cos(-self.cam_yaw), -math.sin(-self.cam_yaw)],
            [math.sin(-self.cam_yaw), math.cos(-self.cam_yaw)],
        ]
        self.camera2ego = {c: rz @ m for c, m in self.camera2ego.items()}
        self.lidar2ego = rz @ self.lidar2ego_raw
        front = (
            rz @ c2v
        )  # calibrated (unrectified) front camera = HUGSIM's ego point, in the travel-aligned frame
        self.cam_xy = front[:2, 3].copy()
        self.cam_z = float(front[2, 3])
        self.T_cam = se2(
            self.cam_xy[0], self.cam_xy[1], 0.0
        )  # front-camera point; heading = the frame's x axis

    def calibration(self) -> dict:
        # lidar2ego is sent as the identity ("lidar = ego"), as the Bench2Drive client does, not HUGSIM's real
        # lidar (x right, y forward: yawed -90 deg). Drive-JEPA places each camera on the ego frame and
        # AutoE2E never reads it, but an adapter that places cameras relative to the lidar (sensor2lidar =
        # inv(lidar2ego) @ camera2ego) turns every camera a quarter turn off with HUGSIM's lidar and drives
        # camera-blind -- the bug Drive-JEPA's adapter had on nuScenes until 13 Oct 2026.
        return {
            "camera2image": self.camera2image,
            "camera2ego": {c: m.tolist() for c, m in self.camera2ego.items()},
            "lidar2ego": np.eye(4).tolist(),
        }


def vehicle_pose(info: dict, rig: Rig) -> tuple[np.ndarray, float]:
    """Vehicle (rear axle) planar pose in the IMU world, and ground height, from HUGSIM's ego_box.

    HUGSIM's ego_box is the front-camera point.
    """
    x, y, z = info["ego_box"][0], info["ego_box"][1], info["ego_box"][2]
    T_world_cam = se2(x, y, info["ego_box"][6])
    T = T_world_cam @ np.linalg.inv(rig.T_cam)
    return T, z - rig.cam_z


def pose_xyyaw(T: np.ndarray) -> tuple[float, float, float]:
    return T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])


class HeldPlan:
    """A model trajectory fixed in the world when it was planned, sampled later for the front-camera point."""

    def __init__(self, t0: float, T_world_veh: np.ndarray, traj, dt: float) -> None:
        traj = np.asarray(traj, dtype=np.float64)
        xy = traj[:, :2]
        # Heading of each waypoint from the path itself, not a predicted heading column: the heading only
        # places the front-camera point (1.7 m ahead of the rear axle), and a near-stationary plan's predicted
        # headings are noise (Drive-JEPA's stop plan on KITTI-360 swung that point 2.3 m sideways and HUGSIM
        # reversed the car). A segment shorter than MIN_STEP_M keeps the previous heading; the first starts
        # from the current one (0).
        yaw, prev = [], 0.0
        for a, b in zip(np.vstack([[0.0, 0.0], xy[:-1]]), xy, strict=True):
            d = b - a
            if math.hypot(d[0], d[1]) >= MIN_STEP_M:
                prev = prev + math.remainder(math.atan2(d[1], d[0]) - prev, 2 * math.pi)
            yaw.append(prev)
        yaw = np.asarray(yaw)
        x0, y0, yaw0 = pose_xyyaw(T_world_veh)
        pts = (T_world_veh @ np.c_[xy, np.ones(len(xy))].T).T[:, :2]
        # How many 0.5 s waypoints the model's own horizon holds: what HUGSIM is sent, at most N_MAX.
        self.n = max(1, min(N_MAX, int(math.floor(len(xy) * dt / PLAN_DT + 1e-9))))
        self.t = t0 + dt * np.arange(0, len(xy) + 1)
        self.xy = np.vstack([[x0, y0], pts])
        self.yaw = np.unwrap(np.r_[yaw0, yaw0 + yaw])

    def sample(self, times) -> list[tuple[float, float, float]]:
        """Rear-axle world poses at `times`.

        Past the plan's end (only a held plan's last 0.25 s) the last segment's velocity continues.
        """
        out = []
        for t in times:
            if t <= self.t[-1]:
                x = np.interp(t, self.t, self.xy[:, 0])
                y = np.interp(t, self.t, self.xy[:, 1])
                h = np.interp(t, self.t, self.yaw)
            else:
                v = (self.xy[-1] - self.xy[-2]) / (self.t[-1] - self.t[-2])
                x, y = self.xy[-1] + v * (t - self.t[-1])
                h = self.yaw[-1]
            out.append((x, y, h))
        return out


def to_hugsim(held: HeldPlan, t_now: float, T_world_veh_now: np.ndarray, rig: Rig) -> np.ndarray:
    """(held.n, 2) waypoints for HUGSIM, in (x right, y forward).

    The front-camera point at t_now + 0.5 k, about the current camera pose.
    """
    T_world_cam_now = T_world_veh_now @ rig.T_cam
    inv_now = np.linalg.inv(T_world_cam_now)
    plan = []
    for x, y, h in held.sample(t_now + PLAN_DT * np.arange(1, held.n + 1)):
        cam = se2(x, y, h) @ rig.T_cam
        rel = inv_now @ cam
        fwd, left = rel[0, 2], rel[1, 2]
        plan.append((-left, fwd))
    return np.asarray(plan, dtype=np.float64)


def main() -> None:
    # HUGSIM's interpreter has it; paddock's own env, which only imports this module, does not
    import requests

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--server", required=True, help="paddock model node, e.g. http://127.0.0.1:9600")
    ap.add_argument(
        "--camera_yaml", required=True, help="the HUGSIM camera config this scenario renders with"
    )
    ap.add_argument("--infer_every", type=int, default=1, help="call the model every N steps (0.25 s each)")
    ap.add_argument(
        "--observe", action="store_true", help="hand the node a frame (POST /observe) on steps between plans"
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--dump_steps",
        default="",
        help="comma list of steps whose request + HUGSIM boxes go to dump_<step>.npz",
    )
    args = ap.parse_args()
    dump = {int(s) for s in args.dump_steps.split(",") if s.strip()}

    obs_pipe, plan_pipe = os.path.join(args.output, "obs_pipe"), os.path.join(args.output, "plan_pipe")
    for p in (obs_pipe, plan_pipe):
        if not os.path.exists(p):
            os.mkfifo(p)
    rig = Rig(args.camera_yaml)
    http = requests.Session()
    http.post(f"{args.server}/reset", json={}, timeout=60).raise_for_status()
    log = open(os.path.join(args.output, "agent_log.jsonl"), "w")  # noqa: SIM115 - one line per step, until 'Done'
    print("bridge ready", vars(args), flush=True)

    step, held, prev = 0, None, None
    black_cache: dict = {}
    while True:
        with open(obs_pipe, "rb") as f:
            msg = pickle.loads(f.read())
        if isinstance(msg, str) and msg == "Done":
            print(f"simulator finished after {step} steps", flush=True)
            break
        obs, info = msg
        t = float(info["timestamp"])
        T_veh, ground_z = vehicle_pose(info, rig)
        vx, vy, yaw = pose_xyyaw(T_veh)
        speed = float(info["ego_velo"])
        accel = yaw_rate = 0.0
        if prev is not None and t > prev[0]:
            accel = (speed - prev[1]) / (t - prev[0])
            yaw_rate = math.remainder(yaw - prev[2], 2 * math.pi) / (t - prev[0])
        prev = (t, speed, yaw)
        rec = {
            "step": step,
            "t": t,
            "command": int(info["command"]),
            "speed": round(speed, 3),
            "ego": [round(vx, 3), round(vy, 3), round(yaw, 4)],
            "called": False,
        }
        plan = None
        try:
            action = step_action(step, held is not None, args.infer_every, args.observe)
            if action is not None:
                e2w = np.eye(4)
                e2w[:2, :2], e2w[:2, 3], e2w[2, 3] = T_veh[:2, :2], T_veh[:2, 2], ground_z
                canbus = [0.0] * 18
                canbus[CANBUS_VELOCITY], canbus[CANBUS_ACCEL], canbus[CANBUS_YAW_RATE] = (
                    speed,
                    accel,
                    yaw_rate,
                )
                images = {}
                for c in CAMS:
                    img = obs["rgb"][c]
                    if (
                        not img.any()
                    ):  # an unrendered (or HUGSIM-blanked rear) camera: encode the black frame once
                        if img.shape not in black_cache:
                            black_cache[img.shape] = encode_image(img)
                        images[c] = black_cache[img.shape]
                    else:
                        images[c] = encode_image(img)
                body = {
                    "images": images,
                    "ego2world": e2w.tolist(),
                    "canbus": canbus,
                    "timestamp": int(round(t * 1e6)),
                    "command": int(info["command"]),
                    "route_xy": None,
                    "calibration": rig.calibration(),
                }
                if step in dump:  # geometry check: exactly what the model is sent, next to HUGSIM's own boxes
                    np.savez_compressed(
                        os.path.join(args.output, f"dump_{step}.npz"),
                        **{f"img_{c}": obs["rgb"][c] for c in CAMS},
                        camera2image=np.array([rig.camera2image[c] for c in CAMS]),
                        camera2ego=np.array([rig.camera2ego[c] for c in CAMS]),
                        ego2world=e2w,
                        obj_boxes=np.array(info["obj_boxes"], dtype=np.float64).reshape(-1, 7),
                        ego_box=np.array(info["ego_box"], dtype=np.float64),
                        cams=np.array(CAMS),
                    )
                t0 = time.time()
                if action == "observe":
                    # A frame of history, no plan: the plan in force stays the one held.
                    http.post(f"{args.server}/observe", json=body, timeout=args.timeout).raise_for_status()
                    rec.update(observed=True, observe_ms=round(1000 * (time.time() - t0)))
                else:
                    r = http.post(f"{args.server}/infer", json=body, timeout=args.timeout)
                    r.raise_for_status()
                    out = r.json()
                    traj, dt = plan_of(out)
                    if not np.isfinite(traj).all() or len(traj) < 2:
                        raise ValueError(f"bad trajectory from the model: {traj.tolist()}")
                    held = HeldPlan(t, T_veh, traj, dt)
                    rec.update(
                        called=True,
                        infer_ms=round(1000 * (time.time() - t0)),
                        n_wp=len(traj),
                        dt=dt,
                        n_sent=held.n,
                        model_last=[round(float(v), 2) for v in traj[-1, :2]],
                        model_traj=np.round(traj, 3).tolist(),
                        extras=out.get("extras"),
                    )
            plan = to_hugsim(held, t, T_veh, rig)
            rec["plan"] = np.round(
                plan, 3
            ).tolist()  # exactly what HUGSIM was sent (x right, y forward, 0.5 s)
            rec["plan_first"] = [round(float(v), 2) for v in plan[0]]
            rec["plan_last"] = [round(float(v), 2) for v in plan[-1]]
        except Exception:
            rec["error"] = traceback.format_exc()[-800:]
            print(rec["error"], file=sys.stderr, flush=True)
            plan = (
                None  # tells HUGSIM the AD side failed: upstream then fails the episode, which is not scored
            )
        log.write(json.dumps(rec) + "\n")
        log.flush()
        with open(plan_pipe, "wb") as f:
            f.write(pickle.dumps(plan))
        step += 1


if __name__ == "__main__":
    main()
