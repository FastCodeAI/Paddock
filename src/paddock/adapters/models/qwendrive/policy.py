"""Qwen-Drive-1.0 adapter: a Qwen3.5-4B VLM with a flow-matching planning expert.

Qwen-Drive reads three cameras at four timestamps 0.5 s apart, 1.5 s of ego
poses at 10 Hz with their velocities and accelerations, and a navigation
command, and regresses 50 waypoints at 10 Hz (vendor `docs/data.md`). The
vendor's benchmark scene files carry exactly that, built offline per dataset;
here the same scene is assembled from whatever a benchmark streams, so the
model sees one input shape on every benchmark. Three adaptations matter.

*Frames.* Benchmarks hand over one frame per observation, at 10 Hz (KITScenes,
neuroNCAP, nuPlan) or 2 Hz (OpenScene). The four camera frames are picked from
the recorded history by timestamp, nearest to t-1.5/-1.0/-0.5/0 s, so a 10 Hz
source contributes every fifth frame and a 2 Hz source every frame. Before
1.5 s of a closed-loop run has elapsed the oldest frame stands in for the
missing ones, and `diagnostics()` counts how often that happened.

*Poses.* The 16 history poses are resampled onto a 0.1 s grid by timestamp,
not taken as the last 16 requests: neuroNCAP's priming steps arrive 0.5 s
apart and only its loop steps at 0.1 s, so counting requests would hand the
model a history five times longer than its label says. Velocity and
acceleration come from the CAN bus where a source records them; KITScenes
records speed only, so acceleration is differenced from it.

*Output.* 50 waypoints at 10 Hz become 8 on the 0.5 s grid, indices 4, 9, ...
39 -- the vendor's own NAVSIM conversion (`qwen_drive.trajectory.resample_10hz_to_2hz`).
Eight rather than six because NAVSIM scores 4 s; neuroNCAP takes the first
six and KITScenes truncates at 3 s, so the extra two cost nothing there.

The model's package (torch, transformers 5.14) exists only in its own
environment and is imported inside `load()`. Everything that can be checked
without it -- frame selection, pose resampling, command mapping, the output
grid -- is a module-level function so the tests can run in paddock's env.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

from paddock.policy.protocol import Command, Observation, Policy, Trajectory

_log = logging.getLogger(__name__)

#: paddock camera -> the view tag the model was trained to read (scene.py CAMERA_VIEWS).
CAMERA_VIEWS: tuple[tuple[str, str], ...] = (
    ("CAM_FRONT", "<FRONT VIEW>"),
    ("CAM_FRONT_LEFT", "<FRONT LEFT VIEW>"),
    ("CAM_FRONT_RIGHT", "<FRONT RIGHT VIEW>"),
)
NEEDED_CAMERAS = tuple(camera for camera, _ in CAMERA_VIEWS)

#: Camera frames per view and their spacing: t-1.5, t-1.0, t-0.5, t-0 s.
NUM_CAM_FRAMES = 4
FRAME_DT = 0.5
#: Ego poses consumed and their spacing: 1.5 s at 10 Hz, last row is now.
NUM_HISTORY_STEPS = 16
HISTORY_DT = 0.1
#: A frame farther than this from its slot's time is a neighbour standing in,
#: not jitter. nuPlan's cameras fire within ~8 ms of each other.
FRAME_TOLERANCE_S = 0.06
#: Recorded observations kept. 64 is 6.4 s at 10 Hz, 3.2 s at CARLA's 20 Hz;
#: both cover the 1.5 s the model reads with room for uneven request cadence.
MAX_RECORDED = 64

#: Model output: 50 poses at 10 Hz. Index i is at (i+1)/10 s, so the 0.5 s
#: grid is 4, 9, ... 39 -- eight points, 4 s.
OUTPUT_POINTS = 50
GRID_INDICES = (4, 9, 14, 19, 24, 29, 34, 39)

DEFAULT_PLANNER = "planner-sft"
MODES = ("direct", "reasoning")

#: paddock's 4-way command -> the model's text command (scene.py NAV_COMMANDS:
#: 0 GO STRAIGHT, 1 TURN LEFT, 2 TURN RIGHT) and its one-hot ego-status
#: command [left, straight, right, unknown] (docs/data.md). The text vocabulary
#: has no "unknown", so FOLLOW_REFERENCE reads as straight in the prompt and
#: as unknown in the status, which is how NAVSIM's own `unknown` rows encode.
NAV_COMMAND: dict[Command, int] = {
    Command.STRAIGHT: 0,
    Command.LEFT: 1,
    Command.RIGHT: 2,
    Command.FOLLOW_REFERENCE: 0,
}
DRIVING_COMMAND: dict[Command, tuple[float, float, float, float]] = {
    Command.LEFT: (1.0, 0.0, 0.0, 0.0),
    Command.STRAIGHT: (0.0, 1.0, 0.0, 0.0),
    Command.RIGHT: (0.0, 0.0, 1.0, 0.0),
    Command.FOLLOW_REFERENCE: (0.0, 0.0, 0.0, 1.0),
}


def _env_or(name: str, default):
    raw = os.environ.get(name)
    return type(default)(raw) if raw not in (None, "") else default


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def pose_xy_yaw(ego2world: np.ndarray) -> np.ndarray:
    """(x, y, yaw) of a 4x4 ego->world; yaw from the rotated x axis."""
    m = np.asarray(ego2world, dtype=np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def select_frame_indices(times_s: np.ndarray) -> tuple[list[int], list[float]]:
    """Which recorded frames fill the four slots, oldest first, and how far
    each is from its slot's time.

    Nearest by timestamp to now-1.5, now-1.0, now-0.5 and now. A 10 Hz source
    lands within a few ms; a 2 Hz source lands exactly; a history shorter than
    1.5 s repeats its oldest frame, which the gaps make visible.
    """
    times = np.asarray(times_s, dtype=np.float64)
    wanted = times[-1] - np.arange(NUM_CAM_FRAMES - 1, -1, -1) * FRAME_DT
    indices = [int(np.argmin(np.abs(times - w))) for w in wanted]
    gaps = [float(abs(times[i] - w)) for i, w in zip(indices, wanted, strict=True)]
    return indices, gaps


def resample_history(
    times_s: np.ndarray, poses: np.ndarray, motion: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The model's 16-row history from recorded (x, y, yaw) and (vx, vy, ax, ay).

    Rows are 0.1 s apart ending now, in the current ego frame: x forward,
    y left, heading positive for a left turn, last row (0, 0, 0). Positions
    and unwrapped yaw are interpolated by time and clipped at the oldest
    record, so a sparse early history reads as the standing start it is.
    Acceleration is differenced from velocity when the source recorded none.
    """
    times = np.asarray(times_s, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    motion = np.asarray(motion, dtype=np.float64).reshape(-1, 4)
    now = times[-1]
    wanted = np.clip(now - np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT, times[0], now)

    x = np.interp(wanted, times, poses[:, 0])
    y = np.interp(wanted, times, poses[:, 1])
    yaw = np.interp(wanted, times, np.unwrap(poses[:, 2]))

    x0, y0, yaw0 = poses[-1]
    c, s = np.cos(yaw0), np.sin(yaw0)
    dx, dy = x - x0, y - y0
    history = np.stack([c * dx + s * dy, -s * dx + c * dy, wrap_angle(yaw - yaw0)], axis=1)
    history[-1] = 0.0

    velocity = np.stack([np.interp(wanted, times, motion[:, i]) for i in (0, 1)], axis=1)
    acceleration = np.stack([np.interp(wanted, times, motion[:, i]) for i in (2, 3)], axis=1)
    if not np.any(motion[:, 2:4]) and len(times) > 1:
        acceleration = np.gradient(velocity, HISTORY_DT, axis=0)
    return history, velocity, acceleration


def medoid_index(trajectories: np.ndarray) -> int:
    """The draw closest to all the others; not a mean, which can land between modes."""
    flat = trajectories.reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None], axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


class QwenDrivePolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = 0.5

    def __init__(
        self,
        planner: str | None = None,
        mode: str | None = None,
        num_samples: int | None = None,
        attn: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.planner = planner or _env_or("QWENDRIVE_PLANNER", DEFAULT_PLANNER)
        self.mode = mode or _env_or("QWENDRIVE_MODE", "direct")
        if self.mode not in MODES:
            raise ValueError(f"QWENDRIVE_MODE must be one of {MODES}, got {self.mode!r}")
        # planner-rl was reward-optimised only under reasoning-conditioned
        # rollouts and the vendor says to run it in reasoning mode; a direct-mode
        # number from it would be nobody's model.
        if "rl" in os.path.basename(self.planner).split("-") and self.mode != "reasoning":
            raise ValueError(f"planner {self.planner!r} must run with QWENDRIVE_MODE=reasoning")
        self.num_samples = num_samples or _env_or("QWENDRIVE_NUM_SAMPLES", 1)
        self.attn = attn or _env_or("QWENDRIVE_ATTN", "sdpa")
        raw_seed = os.environ.get("QWENDRIVE_SEED")
        self.seed = seed if seed is not None else (int(raw_seed) if raw_seed else None)
        self.model = None
        self._mode = None
        self._infer_seconds: list[float] = []
        self._frames_repeated = 0
        self._last_gaps: list[float] = []
        self._last_command: str | None = None
        self._last_reasoning: str | None = None
        self._n_infer = 0
        self.reset()

    # ---- lifecycle ------------------------------------------------------

    def load(self) -> None:
        # torch and qwen_drive exist only inside the model's own environment.
        import torch
        from qwen_drive import InferenceMode, QwenDriveForPlanning

        checkpoint = os.environ.get("QWENDRIVE_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError(
                "QWENDRIVE_CHECKPOINT is not set; model.yaml points it at {weights}/Qwen-Drive-1.0-4B"
            )
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"QWENDRIVE_CHECKPOINT does not exist: {checkpoint}")
        planner = self.planner if os.path.isabs(self.planner) else os.path.join(checkpoint, self.planner)
        if not os.path.isdir(planner):
            raise FileNotFoundError(f"planner directory does not exist: {planner}")

        _log.info("loading %s + %s (attn=%s, mode=%s)", checkpoint, planner, self.attn, self.mode)
        started = time.perf_counter()
        self.model = (
            QwenDriveForPlanning.from_pretrained(
                checkpoint, planner=planner, dtype=torch.bfloat16, attn_implementation=self.attn
            )
            .to("cuda")
            .eval()
        )
        self._mode = (
            InferenceMode.REASONING_PLANNING if self.mode == "reasoning" else InferenceMode.DIRECT_PLANNING
        )
        _log.info("loaded in %.0f s", time.perf_counter() - started)

    def reset(self) -> None:
        """Drop the pose, motion and frame history between runs."""
        self.timestamps: list[int] = []
        self.poses: list[np.ndarray] = []
        self.motion: list[np.ndarray] = []
        self.frames: list[dict[str, Any]] = []

    # ---- history --------------------------------------------------------

    def _record(self, obs: Observation) -> None:
        """Append one observation.

        A frame stamped no later than the last is the same frame again
        (KITScenes clamps its warm-up at the scene start) and is dropped: the
        history is resampled by time. Frames are kept as whatever the source
        gave -- a path on the open-loop benchmarks, decoded pixels in closed
        loop -- and only the four that are used get opened.
        """
        if self.timestamps and obs.timestamp_us <= self.timestamps[-1]:
            return
        sources: dict[str, Any] = {}
        for camera in self.cameras:
            if camera in obs.image_paths:
                sources[camera] = obs.image_paths[camera]
            elif camera in obs.images:
                sources[camera] = obs.images[camera]
            else:
                raise KeyError(
                    f"observation has no {camera} (has {sorted(obs.images) or sorted(obs.image_paths)}); "
                    f"Qwen-Drive needs all of {list(self.cameras)}"
                )
        canbus = np.asarray(obs.canbus, dtype=np.float64)
        self.timestamps.append(int(obs.timestamp_us))
        self.poses.append(pose_xy_yaw(obs.ego2world))
        self.motion.append(np.array([canbus[13], canbus[14], canbus[7], canbus[8]]))
        self.frames.append(sources)
        if len(self.timestamps) > MAX_RECORDED:
            del self.timestamps[:-MAX_RECORDED]
            del self.poses[:-MAX_RECORDED]
            del self.motion[:-MAX_RECORDED]
            del self.frames[:-MAX_RECORDED]

    def observe(self, obs: Observation) -> None:
        """Record a history frame without running the model.

        The model conditions on the recorded history alone, so an open-loop
        sequence costs one forward pass, not one per frame.
        """
        self._record(obs)

    def scene_inputs(self, command: Command) -> dict[str, Any]:
        """Everything but the images for one `DrivingScene`, plus which
        recorded frames to open. Pure numpy, so the tests can check it."""
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        indices, gaps = select_frame_indices(times)
        history, velocity, acceleration = resample_history(times, np.stack(self.poses), np.stack(self.motion))
        return {
            "frame_indices": indices,
            "frame_gaps_s": gaps,
            "history": history,
            "history_velocity": velocity,
            "history_acceleration": acceleration,
            "ego_velocity": tuple(float(v) for v in velocity[-1]),
            "ego_acceleration": tuple(float(a) for a in acceleration[-1]),
            "driving_command": list(DRIVING_COMMAND[Command(command)]),
            "nav_command": NAV_COMMAND[Command(command)],
        }

    def _scene(self, obs: Observation, inputs: dict[str, Any]):
        """A `DrivingScene` from `scene_inputs` plus the four frames per view."""
        from PIL import Image
        from qwen_drive import CameraFrame, DrivingScene

        inputs = dict(inputs)
        indices = inputs.pop("frame_indices")
        inputs.pop("frame_gaps_s")
        views: dict[str, list[Any]] = {}
        for camera, view in CAMERA_VIEWS:
            frames = []
            for index in indices:
                source = self.frames[index][camera]
                frames.append(CameraFrame(source if isinstance(source, str) else Image.fromarray(source)))
            views[view] = frames
        return DrivingScene(views=views, token=str(obs.timestamp_us), **inputs)

    # ---- planning -------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        self._record(obs)
        inputs = self.scene_inputs(obs.command)
        self._last_gaps = inputs["frame_gaps_s"]
        self._frames_repeated += NUM_CAM_FRAMES - len(set(inputs["frame_indices"]))
        self._last_command = Command(obs.command).name
        scene = self._scene(obs, inputs)

        started = time.perf_counter()
        result = self.model.generate_trajectory(
            scene, mode=self._mode, num_samples=self.num_samples, seed=self.seed
        )
        self._infer_seconds.append(time.perf_counter() - started)
        del self._infer_seconds[:-200]

        samples = np.asarray(result.trajectories, dtype=np.float64)  # (K, 50, 3)
        chosen = samples[medoid_index(samples)] if len(samples) > 1 else samples[0]
        self._last_reasoning = result.reasoning
        # The model plans x forward, y left, heading positive left: paddock's frame.
        waypoints = chosen[list(GRID_INDICES)]

        self._n_infer += 1
        if self._n_infer % 100 == 1:
            _log.info(
                "infer #%d command=%s reach %.1f m in %.2f s (frames repeated so far: %d)",
                self._n_infer,
                self._last_command,
                float(np.linalg.norm(waypoints[-1, :2])),
                self._infer_seconds[-1],
                self._frames_repeated,
            )
        return Trajectory(waypoints, dt=self.waypoint_dt)

    def diagnostics(self) -> dict[str, Any]:
        """Cadence, history depth and how often a slot had no frame of its own.

        `frames_repeated_total` is the one to watch in closed loop: it counts
        slots filled by a stand-in because less than 1.5 s had been seen. A few
        per episode is the warm-up; growing steadily means the request cadence
        is not what the benchmark claims.
        """
        if not self.timestamps:
            return {"status": "no frame seen yet"}
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        return {
            "planner": self.planner,
            "mode": self.mode,
            "num_samples": self.num_samples,
            "attn": self.attn,
            "seed": self.seed,
            "history_span_s": round(float(times[-1] - times[0]), 3),
            "request_dt_s": round(float(times[-1] - times[-2]), 3) if len(times) > 1 else None,
            "frames_repeated_total": self._frames_repeated,
            "last_frame_gaps_s": [round(g, 3) for g in self._last_gaps],
            "last_command": self._last_command,
            "infer_seconds_mean": (
                round(float(np.mean(self._infer_seconds)), 3) if self._infer_seconds else None
            ),
            "reasoning": (self._last_reasoning or "")[:400],
        }


POLICY = QwenDrivePolicy
