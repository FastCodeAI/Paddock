"""HiP-AD's own CARLA agent and paddock's adapter, fed the same frames: their inputs and plans compared.

Run in HiP-AD's environment, on the GPU, with the checkpoint (``make check-hipad``)::

    python -m paddock.adapters.models.hipad.tests.agent_reference REPO CHECKPOINT FRAMES_DIR

``FRAMES_DIR`` holds, for each of the six cameras, a folder of 1600x900 frames named by the camera
(``CAM_FRONT/``, ...); the first ``--frames`` of each, in name order, are the drive's frames. Any real
camera footage serves: what is compared is what each side makes of the same pixels.

The agent is ``bench2drive/leaderboard/team_code/hipad_b2d_agent.py`` at the pinned sha, imported with
CARLA and the leaderboard's base class stubbed (neither is reached by what it computes for the model). It
drives a straight route east at 5 m/s on Bench2Drive's camera rig, from its own sensor readings: GNSS,
compass, speedometer and the six cameras, as CARLA would hand them. The adapter is given the same drive
as paddock's Bench2Drive client sends it: the rig's calibration as ``agent.py`` builds it, the pose, and
the route around the car.

Compared on every frame: the camera projections, the image tensor, the command, the target point, the
frame's pose and its timestamp; and on every frame, the plan. HiP-AD's agent keeps ten temporal memories
(``with_close_loop``), each stepped every tenth call; the adapter keeps one, stepped every call (policy.py,
*Rate*). Here the agent
is switched to one memory after it is built, so both step one memory 0.5 s at a time and every plan is
comparable: the frames after the first check what the memory is fed, its timestamps and poses.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
EARTH_RADIUS = 6378137.0
SPEED = 5.0
TICKS_PER_FRAME = 10  # 0.5 s at CARLA's 20 Hz


def _stub_carla() -> None:
    class VehicleControl:
        steer = throttle = brake = 0.0

    carla = types.ModuleType("carla")
    carla.VehicleControl = VehicleControl
    sys.modules["carla"] = carla

    class AutonomousAgent:
        def get_metric_info(self):
            return {}

    track = types.SimpleNamespace(SENSORS="SENSORS")
    leaderboard = types.ModuleType("leaderboard")
    autoagents = types.ModuleType("leaderboard.autoagents")
    base = types.ModuleType("leaderboard.autoagents.autonomous_agent")
    base.AutonomousAgent, base.Track = AutonomousAgent, track
    leaderboard.autoagents, autoagents.autonomous_agent = autoagents, base
    sys.modules.update(
        {
            "leaderboard": leaderboard,
            "leaderboard.autoagents": autoagents,
            "leaderboard.autoagents.autonomous_agent": base,
        }
    )


def _gps(x: float, y: float) -> np.ndarray:
    """CARLA (x, y) -> (lat, lon), the inverse of the agent's ``gps_to_location`` at lat_ref = lon_ref = 0."""
    lat = 2.0 * math.degrees(math.atan(math.exp(y / EARTH_RADIUS))) - 90.0
    lon = math.degrees(x / EARTH_RADIUS)
    return np.array([lat, lon])


def rig_calibration(rig: dict, camera: str) -> tuple[np.ndarray, np.ndarray]:
    """(intrinsics, camera2ego) as ``bench2drive/agent.py`` builds them: CARLA's transform matrix for the
    mount, y-flipped (``_rh``), then optical axes (``R_OPT``). Bench2Drive's cameras have no pitch or roll."""
    x, y, z, roll, pitch, yaw = rig[camera]["pose"]
    assert roll == 0 and pitch == 0, "this reference handles yaw-only mounts"
    width, height = rig[camera]["size"]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    carla_matrix = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.0]])
    flip = np.diag([1.0, -1.0, 1.0, 1.0])
    pose = flip @ carla_matrix @ flip
    r_opt = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pose[:3, :3] = pose[:3, :3] @ r_opt
    f = width / (2.0 * math.tan(math.radians(rig[camera]["fov"]) / 2.0))
    k = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    return k, pose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repo")
    parser.add_argument("checkpoint")
    parser.add_argument("frames_dir")
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args(argv)

    import cv2
    import torch

    from paddock.adapters.benchmarks.bench2drive import rig as rigs
    from paddock.adapters.models.hipad.policy import HiPADPolicy
    from paddock.policy.protocol import Command, Observation

    repo = Path(args.repo)
    frames = {camera: sorted((Path(args.frames_dir) / camera).iterdir())[: args.frames] for camera in CAMERAS}
    config = repo / "projects/configs/hipad_b2d_stage2.py"

    # -- the agent ------------------------------------------------------------------------------------
    _stub_carla()
    sys.path[:0] = [str(repo / "bench2drive/leaderboard"), str(repo)]
    os.chdir(repo)
    scratch = tempfile.mkdtemp(prefix="hipad-agent-")
    os.environ.update(SAVE_PATH=scratch, ROUTES="reference.xml", IS_BENCH2DRIVE="1")
    import team_code.hipad_b2d_agent as agent_module  # noqa: E402

    # The authors' project_dir, as policy.py repoints it.
    from mmcv import Config

    from paddock.adapters.models.hipad.policy import AUTHORS_PROJECT_DIR, repoint

    original_fromfile = Config.fromfile

    def fromfile(path, *a, **k):
        cfg = original_fromfile(path, *a, **k)
        repoint(cfg._cfg_dict, AUTHORS_PROJECT_DIR, str(repo))
        return cfg

    Config.fromfile = staticmethod(fromfile)
    agent_module.Config.fromfile = staticmethod(fromfile)
    agent = agent_module.SparseAgent()
    agent.setup(f"{config}+{args.checkpoint}+reference")
    # One temporal memory, as the adapter runs HiP-AD (policy.py, *Rate*): HiP-AD's own open-loop mode.
    head = agent.model.head.onedecoder_head
    head.with_close_loop = False
    head.init_instance_bank_list()
    agent.lat_ref, agent.lon_ref = 0.0, 0.0
    # A straight route east (CARLA +x), points 10 m behind to 200 m ahead, every 50 m, LANEFOLLOW.
    route_x = [-10.0, 0.0, 50.0, 100.0, 150.0, 200.0]
    lanefollow = 4
    agent._route_planner = agent_module.RoutePlanner(4.0, 50.0, lat_ref=0.0, lon_ref=0.0)
    agent._route_planner.route.extend((np.array([x, 0.0]), lanefollow) for x in route_x)
    agent.initialized = True
    agent.metric_info = {}

    seen: list[dict] = []
    agent_model = agent.model
    original_forward = agent_model.forward

    def recording_forward(*a, **kwargs):
        out = original_forward(*a, **kwargs)
        seen.append({"inputs": kwargs, "outputs": out})
        return out

    agent_model.forward = recording_forward

    # -- the adapter ----------------------------------------------------------------------------------
    os.environ.update(HIPAD_CONFIG=str(config), HIPAD_CHECKPOINT=args.checkpoint, HIPAD_REPO=str(repo))
    os.environ.setdefault("UNIAD_VAD_THREADS", "1")
    policy = HiPADPolicy()
    policy.load()
    fed: list[dict] = []
    adapter_model = policy.model
    adapter_forward = adapter_model.forward

    def adapter_recording(*a, **kwargs):
        out = adapter_forward(*a, **kwargs)
        fed.append({"inputs": kwargs, "outputs": out})
        return out

    adapter_model.forward = adapter_recording
    calibration = {c: rig_calibration(rigs._BENCH2DRIVE, c) for c in CAMERAS}

    failures: list[str] = []
    for i in range(args.frames):
        step = i * TICKS_PER_FRAME
        x = SPEED * step / 20.0
        rgb = {c: cv2.cvtColor(cv2.imread(str(frames[c][i])), cv2.COLOR_BGR2RGB) for c in CAMERAS}
        # CARLA hands cameras over as BGRA; IMU: accel, gyro, compass (pi/2 = facing +x, east).
        input_data = {
            c: (step, np.dstack([rgb[c][:, :, ::-1], np.full(rgb[c].shape[:2], 255, np.uint8)]))
            for c in CAMERAS
        }
        input_data.update(
            {
                "GPS": (step, np.r_[_gps(x, 0.0), 0.0]),
                "SPEED": (step, {"speed": np.float64(SPEED)}),  # CARLA hands a numpy scalar
                "IMU": (step, np.array([0, 0, 9.81, 0, 0, 0, math.pi / 2])),
                "bev": (step, np.zeros((512, 512, 4), np.uint8)),
            }
        )
        agent.step = step - 1
        agent.run_step(input_data, step / 20.0)

        ego2world = np.eye(4)
        ego2world[0, 3] = x
        route = np.array([[rx - x, 0.0] for rx in route_x])
        obs = Observation(
            images=rgb,
            ego2world=ego2world,
            canbus=np.zeros(18),
            timestamp_us=int(step / 20.0 * 1e6),
            command=Command.FOLLOW_REFERENCE,
            camera2image={c: calibration[c][0] for c in CAMERAS},
            camera2ego={c: calibration[c][1] for c in CAMERAS},
            route_xy=route,
        )
        policy.infer(obs)

        a, b = seen[-1]["inputs"], fed[-1]["inputs"]

        def compare(name, left, right, atol, frame=i):
            left = left.detach().float().cpu().numpy() if torch.is_tensor(left) else np.asarray(left, float)
            right = (
                right.detach().float().cpu().numpy() if torch.is_tensor(right) else np.asarray(right, float)
            )
            diff = float(np.abs(left - right).max()) if left.shape == right.shape else float("inf")
            ok = diff <= atol
            print(f"frame {frame} {name:18s} max |agent - adapter| = {diff:.3g}  {'ok' if ok else 'DIFFERS'}")
            if not ok:
                failures.append(f"frame {frame} {name}")

        compare("projection_mat", a["projection_mat"], b["projection_mat"], 1e-2)
        compare("img", a["img"], b["img"], 1e-4)
        compare("gt_ego_fut_cmd", a["gt_ego_fut_cmd"], b["gt_ego_fut_cmd"], 0)
        compare("target_point", a["target_point"], b["target_point"], 1e-3)
        compare("T_global", a["img_metas"][0]["T_global"], b["img_metas"][0]["T_global"], 1e-6)
        compare("timestamp", a["timestamp"], b["timestamp"], 1e-9)
        compare(
            "plan_speed_2hz",
            seen[-1]["outputs"][0]["img_bbox"]["plan_speed_2hz"],
            fed[-1]["outputs"][0]["img_bbox"]["plan_speed_2hz"],
            # Equal inputs, but not bit-equal plans: fp16 and flash-attention are not bit-reproducible,
            # and the memory carries a frame's rounding into the next. Two runs of one model on one
            # KITScenes set differed by up to 0.17 m; a wrong timestamp or pose moves a plan by metres.
            0.1,
        )
    print("PASS" if not failures else f"FAIL: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
