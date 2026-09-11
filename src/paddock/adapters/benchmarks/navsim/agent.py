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

**This module runs inside the vendor's Python 3.9 interpreter**, with
paddock's source on ``PYTHONPATH`` (the harness puts it there) but none of
paddock's dependencies installed. So it imports only from
``paddock.policy.protocol`` and ``convert`` -- the wire contract and the
geometry -- and *those two modules must stay 3.9-importable*: numpy-only,
no ``match``, no 3.10+ syntax evaluated at import. A ``match`` statement in
``protocol.py`` would fail every NAVSIM run inside hydra with only a log tail
to show for it. ``tests/test_navsim_adapter.py`` compiles them under a 3.9
interpreter when the site has one.

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
    pose_matrix,
    to_navsim_poses,
)
from paddock.adapters.benchmarks.navsim.dense import DenseHistory, as_ego_status
from paddock.policy.protocol import Command, Observation, constant_velocity

#: Seconds between proxy retries, multiplied by the attempt number.
RETRY_BACKOFF_S = 1.0


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

    Built from the protocol's own `constant_velocity` and run through
    `to_navsim_poses`, so the one agent meant to validate the conversion path
    exercises it rather than sidestepping it.
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
        plan = constant_velocity(speed, n=NAVSIM_NUM_POSES, dt=NAVSIM_INTERVAL_S)
        poses = to_navsim_poses(plan.xy, plan.dt)
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
        retries: int = 2,
        trajectory_sampling: TrajectorySampling | None = None,
        nuplan_root: str | None = None,
        nuplan_split: str | None = None,
        dense_hz: float = 0.0,
        history_s: float = 0.0,
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
        self._retries = max(0, int(retries))
        # One connection per agent, opened on first use: the scorer instantiates
        # the agent inside each worker, so a client created here would be built
        # before the fork and a fresh one per scenario is a TCP handshake ~12k
        # times on the full split.
        self._client = None
        # A 10 Hz history from nuPlan's full-rate data, for a model that needs
        # denser input than OpenScene's four 0.5 s frames (dense.py). Set by the
        # harness per model; when unset the scorer's own AgentInput is forwarded.
        self._dense = None
        self._history_s = float(history_s or 0.0)
        if nuplan_root and float(dense_hz or 0.0) > 0:
            split = nuplan_split or (os.path.basename(os.path.normpath(root)) if root else None)
            if not split:
                raise ValueError("nuplan_split is needed with nuplan_root; sensor_blobs_path names no split")
            self._dense = DenseHistory(nuplan_root, split, self._cameras, float(dense_hz))

    def _http(self):
        import httpx

        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

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

    def _calibration(self, cameras, index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """(camera2image, camera2ego) of this model's cameras at one NAVSIM frame."""
        camera2image: dict[str, np.ndarray] = {}
        camera2ego: dict[str, np.ndarray] = {}
        for name in self._cameras:
            camera = getattr(cameras, CAMERA_SLOTS[name])
            if camera.intrinsics is None:
                raise RuntimeError(
                    f"NAVSIM delivered no {name} for history frame {index}; the sensor config "
                    f"and the model's declared cameras disagree"
                )
            camera2image[name] = np.asarray(camera.intrinsics, dtype=np.float64)[:3, :3]
            # lidar2ego is identity in OpenScene, so sensor2lidar is camera2ego.
            extrinsics = np.eye(4, dtype=np.float64)
            extrinsics[:3, :3] = np.asarray(camera.sensor2lidar_rotation, dtype=np.float64)
            extrinsics[:3, 3] = np.asarray(camera.sensor2lidar_translation, dtype=np.float64)
            camera2ego[name] = extrinsics
        return camera2image, camera2ego

    @staticmethod
    def _command_of(status) -> Command:
        command = np.asarray(status.driving_command).ravel()
        hot = int(np.argmax(command)) if command.size else 3
        return Command(int(COMMAND_FROM_ONE_HOT[hot]) if hot < 4 else 3)

    def _observation(self, agent_input: AgentInput, index: int, yaw_rate: float) -> Observation:
        """One history frame as a paddock `Observation`.

        Built as the same object the open-loop runner builds and encoded with
        its `to_wire`, so this proxy and the runner cannot disagree about the
        wire shape.
        """
        status = agent_input.ego_statuses[index]
        cameras = agent_input.cameras[index]
        camera2image, camera2ego = self._calibration(cameras, index)

        images: dict[str, np.ndarray] = {}
        image_paths: dict[str, str] = {}
        for name in self._cameras:
            camera = getattr(cameras, CAMERA_SLOTS[name])
            if camera.camera_path is not None and self._sensor_blobs_path:
                image_paths[name] = os.path.join(self._sensor_blobs_path, str(camera.camera_path))
            elif camera.image is not None:
                images[name] = np.asarray(camera.image)
            else:
                raise RuntimeError(f"NAVSIM delivered neither pixels nor a path for {name}")

        return Observation(
            images=images,
            image_paths=image_paths,
            # The current frame is the origin: NAVSIM expresses history poses
            # relative to it, and every paddock policy uses ego2world only for
            # relative motion, so a local origin is exact rather than an
            # approximation.
            ego2world=pose_matrix(status.ego_pose),
            canbus=np.asarray(canbus(status, yaw_rate), dtype=np.float64),
            # NAVSIM carries no wall-clock timestamp in AgentInput. Frames are
            # spaced at exactly NAVSIM_INTERVAL_S, which is what the models that
            # select history by timestamp actually need.
            timestamp_us=int(round((index - (len(agent_input.ego_statuses) - 1)) * NAVSIM_INTERVAL_S * 1e6)),
            command=self._command_of(status),
            camera2image=camera2image,
            camera2ego=camera2ego,
            # Identity, but sent explicitly: models that project through the
            # lidar frame call require_lidar2ego() and must not have to assume
            # what this benchmark's convention is.
            lidar2ego=np.eye(4),
        )

    def _frame_payload(self, agent_input: AgentInput, index: int, yaw_rate: float) -> dict[str, Any]:
        return self._observation(agent_input, index, yaw_rate).to_wire()

    def _dense_payloads(self, agent_input: AgentInput) -> list[dict[str, Any]]:
        """The scenario's history at nuPlan's rate, as wire observations.

        Anchored on the keyframe's own image of the model's first camera, so
        the last observation *is* NAVSIM's current frame -- same image, same
        pose origin -- and the frames before it are the ones OpenScene left
        out. Calibration comes from the keyframe: one rig, one log. The driving
        command is the nearest keyframe's; nuPlan records none at 10 Hz.
        """
        statuses = agent_input.ego_statuses
        cameras = agent_input.cameras[-1]
        anchor = getattr(cameras, CAMERA_SLOTS[self._cameras[0]])
        if anchor.camera_path is None:
            raise RuntimeError(
                f"NAVSIM delivered no path for {self._cameras[0]}; the dense history is keyed by it"
            )
        camera2image, camera2ego = self._calibration(cameras, len(statuses) - 1)
        last = len(statuses) - 1
        payloads = []
        frames = self._dense.frames(str(anchor.camera_path), self._history_s)
        # For the plan dump: how much of this history was not the frame asked for.
        self._last_history = {
            "clamped": sum(f.clamped for f in frames),
            "approximate": sum(f.approximate for f in frames),
        }
        for frame in frames:
            nearest = last + int(round(frame.time_us / 1e6 / NAVSIM_INTERVAL_S))
            status = statuses[min(max(nearest, 0), last)]
            observation = Observation(
                images={},
                image_paths=dict(frame.paths),
                ego2world=pose_matrix(frame.pose),
                canbus=np.asarray(canbus(as_ego_status(frame), frame.yaw_rate), dtype=np.float64),
                timestamp_us=int(frame.time_us),
                command=self._command_of(status),
                camera2image=camera2image,
                camera2ego=camera2ego,
                lidar2ego=np.eye(4),
            )
            payloads.append(observation.to_wire())
        return payloads

    def compute_trajectory(
        self,
        agent_input: AgentInput,
        past_ego_simulated_states: Any = None,
        metric_cache: Any = None,
        observation_interval: Any = None,
    ) -> Trajectory:
        if self._dense is not None:
            observations = self._dense_payloads(agent_input)
        else:
            yaw_rates = self._yaw_rates(agent_input.ego_statuses)
            observations = [
                self._frame_payload(agent_input, index, yaw_rates[min(index, len(yaw_rates) - 1)])
                for index in range(len(agent_input.ego_statuses))
            ]

        # A model can fail on one scenario for reasons that are not the
        # scenario's: alpamayo's VLM occasionally emits no trajectory token and
        # its policy raises. For an unseeded sampler a second request is a fresh
        # draw from the same distribution, not a different measurement; for a
        # deterministic model it fails again at the cost of one request. The
        # final failure still propagates, so the scorer records the scenario
        # as failed rather than paddock inventing a plan for it.
        # A transport error -- the node swapping, or mid-restart, which is the
        # case the runner-level retry exists for -- is retried the same way as
        # a 5xx, after a short backoff so the second draw is not the same
        # instant's; the last failure of either kind propagates.
        import time

        import httpx

        body = None
        last_error = None
        for attempt in range(1 + self._retries):
            if attempt:
                time.sleep(RETRY_BACKOFF_S * attempt)
            try:
                response = self._http().post(
                    f"{self._node_url}/infer_sequence",
                    json={"observations": observations},
                )
            except httpx.TransportError as exc:
                last_error = exc
                continue
            if response.status_code >= 500 and attempt < self._retries:
                continue
            response.raise_for_status()
            body = response.json()
            break
        if body is None:
            raise last_error if last_error is not None else RuntimeError("policy node gave no answer")

        poses = to_navsim_poses(body["trajectory"], float(body.get("dt", NAVSIM_INTERVAL_S)))
        self._record(agent_input, body, poses, metric_cache, len(observations))
        return Trajectory(poses.astype(np.float32), self._trajectory_sampling)

    def _record(
        self,
        agent_input: AgentInput,
        body: dict,
        poses: np.ndarray,
        metric_cache: Any = None,
        history_frames: int = 0,
    ) -> None:
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
            # Which scenario this plan belongs to. MetricCache has no token
            # field, but its file lives at <cache>/<log>/<type>/<token>/..., so
            # the token is the parent directory's name. Read as a label only --
            # never as an input to the model -- so that a run whose scorer dies
            # after the last forward pass can be re-scored from these plans
            # through run_pdm_score_from_submission.py instead of re-run.
            # 12,144 alpamayo plans were once lost for want of this.
            scenario = {}
            if metric_cache is not None:
                path = getattr(metric_cache, "file_path", None)
                scenario = {
                    "token": os.path.basename(os.path.dirname(str(path))) if path else None,
                    "log_name": getattr(metric_cache, "log_name", None),
                    "time_s": getattr(getattr(metric_cache, "timepoint", None), "time_s", None),
                }
            line = json.dumps(
                {
                    **scenario,
                    "ego_speed_ms": round(speed, 4),
                    "driving_command": np.asarray(status.driving_command).ravel().tolist(),
                    "history_frames": history_frames,
                    "history_source": "nuplan" if self._dense is not None else "openscene",
                    **(getattr(self, "_last_history", None) or {}),
                    "model_dt": float(body.get("dt", NAVSIM_INTERVAL_S)),
                    "model_waypoints": np.asarray(body["trajectory"], dtype=float).round(4).tolist(),
                    "navsim_poses": poses.round(4).tolist(),
                }
            )
            with open(self._dump_path, "a") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must not break a scoring run
            pass
