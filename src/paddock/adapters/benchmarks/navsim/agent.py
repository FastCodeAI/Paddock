"""NAVSIM agents that paddock injects into the PDM scorer.

NAVSIM's scorer owns its own iteration: it loads scenes, re-simulates each
predicted trajectory through a kinematic model and scores the result against a
cached ground truth. Paddock does not reimplement any of that. Instead it hands
the scorer an agent that forwards to a paddock policy node over HTTP -- the same
node closed-loop benchmarks drive -- so NAVSIM stays authoritative about scoring
and paddock stays authoritative about model environments.

That split is what makes four models possible at all: the scorer runs in
Drive-JEPA's python 3.9 conda environment, while openpilot runs under uv on
3.12 and cannot share a process with it.

**This module deliberately does not import paddock.** It is loaded inside the
vendor's environment, where paddock is not installed and the interpreter is
3.9, so it speaks the wire format directly (documented in
``paddock/policy/protocol.py``) and depends only on numpy, httpx and NAVSIM's
own dataclasses. Keep it that way; the coupling is the wire format, which is
NeuroNCAP's and has been stable across every model node.

Three things here exist because the fork made them necessary:

* ``super().__init__()`` is called. The fork added ``move_all_modules_to_gpu``
  to the scoring script, which walks ``dir(agent)``; ``HumanAgent`` never
  initialises ``nn.Module``, so that walk raises ``AttributeError: _modules``
  and the fork's own human baseline cannot run.
* ``compute_trajectory`` takes the fork's widened four-argument signature.
  ``ConstantVelocityAgent`` still has the stock two-argument override, so the
  fork's other baseline is equally dead on arrival.
* ``PaddockConstantVelocityAgent`` exists to give the scoring path a baseline
  that actually runs, without patching vendor source.

The privileged arguments (``metric_cache``, ``past_ego_simulated_states``) are
accepted and **ignored**. Drive-JEPA's native agent uses them to re-rank its
proposals against the scorer's own comfort metric; a model behind an HTTP node
structurally cannot, and letting one model see the scorer's state while the
others do not is not a comparison. See docs/benchmarks/navsim.md.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from paddock.adapters.benchmarks.navsim.convert import (
    CAMERA_SLOTS,
    COMMAND_FROM_ONE_HOT,
    NAVSIM_INTERVAL_S,
    NAVSIM_NUM_POSES,
    canbus,
    encode_image,
    pose_matrix,
    to_navsim_poses,
)


class _PaddockAgentBase(AbstractAgent):
    """Shared plumbing. Calls super().__init__(), which the fork's agents do not."""

    requires_scene = False

    def __init__(self, trajectory_sampling: TrajectorySampling | None = None) -> None:
        if trajectory_sampling is None:
            trajectory_sampling = TrajectorySampling(
                time_horizon=NAVSIM_NUM_POSES * NAVSIM_INTERVAL_S,
                interval_length=NAVSIM_INTERVAL_S,
            )
        super().__init__(trajectory_sampling)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """Nothing to bring up: the model lives in another process."""


class PaddockConstantVelocityAgent(_PaddockAgentBase):
    """Straight ahead at the current speed. No model, no node, no GPU.

    The reference baseline for the scoring path itself: if this does not
    produce a plausible EPDMS over the configured scenarios, the problem is the
    data, the metric cache or the maps -- not any model. The fork ships an agent
    that means to do this and cannot run.
    """

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(
        self,
        agent_input: AgentInput,
        past_ego_simulated_states: Any = None,
        metric_cache: Any = None,
        observation_interval: Any = None,
    ) -> Trajectory:
        velocity = np.asarray(agent_input.ego_statuses[-1].ego_velocity, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        steps = np.arange(1, NAVSIM_NUM_POSES + 1) * NAVSIM_INTERVAL_S
        poses = np.stack([speed * steps, np.zeros_like(steps), np.zeros_like(steps)], axis=1)
        return Trajectory(poses.astype(np.float32), self._trajectory_sampling)


class PaddockProxyAgent(_PaddockAgentBase):
    """Forwards NAVSIM's AgentInput to a paddock policy node and returns its plan.

    The node is already running and health-checked by the time the scorer starts
    -- paddock owns that lifecycle -- so this only translates.

    Frames are shipped as **paths, not pixels**. The scorer and the node share a
    filesystem, and one NAVSIM history is four frames of 1920x1080 per camera;
    base64 over a socket would move ~8 GB across a 242-scenario run to redeliver
    files the node can already open. Same reasoning as the KITScenes harness.
    """

    def __init__(
        self,
        node_url: str,
        node_urls: list[str] | None = None,
        cameras: list[str] | None = None,
        sensor_blobs_path: str | None = None,
        timeout_s: float = 600.0,
        dump_path: str | None = None,
        trajectory_sampling: TrajectorySampling | None = None,
    ) -> None:
        super().__init__(trajectory_sampling)
        # One node per scorer worker, when the harness started several. Each
        # worker is its own process and picks by PID, so the assignment is
        # sticky for the life of the run and, with the consecutive PIDs a
        # process pool spawns, distinct across workers. A collision costs
        # throughput, never correctness: any node serves any request.
        pool = [u.rstrip("/") for u in (node_urls or [node_url])]
        self._node_url = pool[os.getpid() % len(pool)]
        # Where each plan is recorded. A NAVSIM score is a single composite
        # number: when it comes back low there is no way to tell a domain gap
        # from a frame-convention bug by staring at it. The plans cost a few KB
        # per run and answer that question directly.
        self._dump_path = dump_path
        self._cameras = list(cameras or ["CAM_FRONT"])
        unknown = [c for c in self._cameras if c not in CAMERA_SLOTS]
        if unknown:
            raise ValueError(f"no NAVSIM camera slot for {unknown!r}; the nuPlan rig has no equivalent view")
        # Resolved once: Camera.camera_path is relative to the blob root.
        root = sensor_blobs_path or os.environ.get("PADDOCK_NAVSIM_SENSOR_BLOBS", "")
        self._sensor_blobs_path = root
        self._timeout_s = float(timeout_s)

    def get_sensor_config(self) -> SensorConfig:
        """Load only the cameras this model declared, for every history frame.

        Loading all eight would read 8x the JPEGs per scenario for models that
        consume one.
        """
        wanted = {CAMERA_SLOTS[c] for c in self._cameras}
        fields = dict.fromkeys(
            ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"], False
        )
        for slot in wanted:
            fields[slot] = True
        return SensorConfig(lidar_pc=False, **fields)

    def _yaw_rates(self, ego_statuses) -> list[float]:
        """Yaw rate per frame, differenced from headings at NAVSIM's 0.5 s spacing.

        EgoStatus carries velocity and acceleration but no rotation rate, and
        openpilot reads yaw rate off the CAN bus every frame.
        """
        headings = np.unwrap([float(status.ego_pose[2]) for status in ego_statuses])
        if len(headings) < 2:
            return [0.0]
        rates = np.gradient(headings, NAVSIM_INTERVAL_S)
        return [float(r) for r in rates]

    def _frame_payload(self, agent_input: AgentInput, index: int, yaw_rate: float) -> dict[str, Any]:
        status = agent_input.ego_statuses[index]
        cameras = agent_input.cameras[index]

        images: dict[str, str] = {}
        image_paths: dict[str, str] = {}
        camera2image: dict[str, Any] = {}
        camera2ego: dict[str, Any] = {}

        for name in self._cameras:
            camera = getattr(cameras, CAMERA_SLOTS[name])
            if camera.intrinsics is None:
                raise RuntimeError(
                    f"NAVSIM delivered no {name} for history frame {index}; the sensor config "
                    f"and the model's declared cameras disagree"
                )
            intrinsics = np.eye(4, dtype=np.float64)
            intrinsics[:3, :3] = np.asarray(camera.intrinsics, dtype=np.float64)
            camera2image[name] = intrinsics[:3, :3].tolist()

            # lidar2ego is identity in OpenScene, so sensor2lidar is camera2ego.
            extrinsics = np.eye(4, dtype=np.float64)
            extrinsics[:3, :3] = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
            extrinsics[:3, 3] = np.asarray(camera.sensor2lidar_translation, dtype=np.float64)
            camera2ego[name] = extrinsics.tolist()

            if camera.camera_path is not None and self._sensor_blobs_path:
                image_paths[name] = os.path.join(self._sensor_blobs_path, str(camera.camera_path))
            elif camera.image is not None:
                images[name] = encode_image(camera.image)
            else:
                raise RuntimeError(f"NAVSIM delivered neither pixels nor a path for {name}")

        command = np.asarray(status.driving_command).ravel()
        hot = int(np.argmax(command)) if command.size else 3

        return {
            "images": images,
            "image_paths": image_paths,
            # The current frame is the origin: NAVSIM expresses history poses
            # relative to it, and every paddock policy uses ego2world only for
            # relative motion, so a local origin is exact rather than an
            # approximation.
            "ego2world": pose_matrix(status.ego_pose).tolist(),
            "canbus": canbus(status, yaw_rate),
            # NAVSIM carries no wall-clock timestamp in AgentInput. Frames are
            # spaced at exactly NAVSIM_INTERVAL_S, which is what the models that
            # select history by timestamp actually need.
            "timestamp": int(round((index - (len(agent_input.ego_statuses) - 1)) * NAVSIM_INTERVAL_S * 1e6)),
            "command": int(COMMAND_FROM_ONE_HOT[hot]) if hot < 4 else 3,
            "calibration": {
                "camera2image": camera2image,
                "camera2ego": camera2ego,
                # Identity, but sent explicitly: models that project through the
                # lidar frame call require_lidar2ego() and must not have to
                # assume what this benchmark's convention is.
                "lidar2ego": np.eye(4).tolist(),
            },
        }

    def compute_trajectory(
        self,
        agent_input: AgentInput,
        past_ego_simulated_states: Any = None,
        metric_cache: Any = None,
        observation_interval: Any = None,
    ) -> Trajectory:
        import httpx

        yaw_rates = self._yaw_rates(agent_input.ego_statuses)
        observations = [
            self._frame_payload(agent_input, index, yaw_rates[min(index, len(yaw_rates) - 1)])
            for index in range(len(agent_input.ego_statuses))
        ]

        response = httpx.post(
            f"{self._node_url}/infer_sequence",
            json={"observations": observations},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        body = response.json()

        poses = to_navsim_poses(body["trajectory"], float(body.get("dt", NAVSIM_INTERVAL_S)))
        self._record(agent_input, body, poses)
        return Trajectory(poses.astype(np.float32), self._trajectory_sampling)

    def _record(self, agent_input: AgentInput, body: dict, poses: np.ndarray) -> None:
        """Append one plan to the dump, alongside the ego state that produced it.

        Best-effort: a failure to write a diagnostic must never fail a run, and
        four scorer processes append to one file, so each line is written in a
        single `write` call to stay atomic on POSIX for payloads this small.
        """
        if not self._dump_path:
            return
        try:
            status = agent_input.ego_statuses[-1]
            speed = float(np.linalg.norm(np.asarray(status.ego_velocity, dtype=np.float64)))
            line = json.dumps(
                {
                    "ego_speed_ms": round(speed, 4),
                    "driving_command": np.asarray(status.driving_command).ravel().tolist(),
                    "model_dt": float(body.get("dt", NAVSIM_INTERVAL_S)),
                    "model_waypoints": np.asarray(body["trajectory"], dtype=float).round(4).tolist(),
                    "navsim_poses": poses.round(4).tolist(),
                }
            )
            with open(self._dump_path, "a") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must not break a scoring run
            pass
