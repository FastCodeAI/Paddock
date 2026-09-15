"""Qwen-Drive 1.0 as a paddock Policy.

Qwen-Drive is a Qwen3.5-4B VLM with a 1 B flow-matching planning expert bolted
to the post-rotary K/V of its eight full-attention layers. In
`REASONING_PLANNING` the VLM first writes a one-sentence rationale and the
expert plans from *that* cache, so the text is upstream of the trajectory rather
than commentary on it.

Its conventions already match paddock's: x forward, y **left**, heading positive
for a left turn, metres and radians, 50 waypoints at 10 Hz. Nothing here flips a
sign -- comma2k19's device frame is y-right, but `benchmarks/comma2k19/loader.py`
converts to the ego convention before the observation is built, so a flip in the
policy would mirror every plan and still look smooth. See
`docs/datasets/comma2k19.md`.

Four adaptations matter.

*Three views from one camera.* `DrivingScene.__post_init__` **raises** on a
missing view, so `<FRONT VIEW>`, `<FRONT LEFT VIEW>` and `<FRONT RIGHT VIEW>`
must all be filled from whatever the source has. `alpamayo15/policy.py::slots_for`
argues the opposite case and is right there -- that model tolerates a missing
slot, so dropping one says "there is no +55 deg view" where repeating the centre
frame would say "here is one" and lie. Qwen-Drive does not tolerate it, so the
choice is between two lies, and it was measured rather than argued: `replicate`
(the true forward frame in all three slots) scored 1.47/2.00 ADE against
`crop`'s 1.78/2.13 on the reference's city/highway segments. Honest geometry --
each view really is a forward camera -- beat fabricated lateral context, because
the model's learned cross-view extrinsics are load-bearing. `QWENDRIVE_VIEWS=crop`
runs the ablation.

*History.* 16 poses at 10 Hz over 1.5 s plus `[vx, vy, ax, ay]`, all in the ego
frame of the scored frame. These are resampled **by timestamp** from whatever
cadence the source runs at, never by taking the last 16 requests: comma2k19
feeds 20 Hz, and differencing a 20 Hz history labelled as 10 Hz would tell the
model it is going half speed. Ego state is by far this model's strongest input
-- the reference's ablation put a stationary history at ADE 15.79 city / 70.74
highway against a 1.47/2.00 baseline -- so `diagnostics()` publishes the
pose-derived speed beside the recorded one, and a disagreement there is the
first thing to look at when a number will not line up.

*No route.* comma2k19 ships no turn-by-turn signal and every observation carries
`Command.STRAIGHT`, so the model plans every sample under "GO STRAIGHT". That is
a real cost, not a formality: feeding the command a route planner *would* have
issued (read off the recorded future, hence diagnostic-only and never scored)
improved the reference's city ADE 1.47 -> 1.28. The residual on turns is partly
a missing input rather than a planning failure, and `nav_command` in
`diagnostics()` records which it was.

*Determinism.* The flow sampler draws sample *k* from `torch.Generator` seeded
`seed + k`, and the reasoning turn is generated with `do_sample=False`. So
Qwen-Drive is reproducible given a seed, which is why `model.yaml` declares
`stochastic: false` -- an explicit `QWENDRIVE_SEED` is passed on every call
rather than relying on the checkpoint's default staying put.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # annotations only -- torch lives in the model's own env
    from PIL.Image import Image as PILImage

from paddock.policy.protocol import Command, Observation, Policy, Trajectory

_log = logging.getLogger(__name__)

CAMERA = "CAM_FRONT"

#: The three view slots the model was trained to read, in that order. Imported
#: from the vendor package at load time; restated here because the tests run in
#: paddock's own interpreter, which cannot import `qwen_drive`.
CAMERA_VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")

#: 16 poses at 10 Hz, oldest first, the last one exactly (0, 0, 0).
NUM_HISTORY_STEPS = 16
HISTORY_DT = 0.1

#: Camera timestamps, seconds before the scored frame, oldest first. The model's
#: prompt labels them "t-1.5s" .. "t-0s"; feeding a different spacing would make
#: the labels false without raising anything.
VIEW_OFFSETS_S = (1.5, 1.0, 0.5, 0.0)

#: 50 waypoints at 10 Hz = 5 s. `model.yaml` repeats both, because the report's
#: time axis reads them from the registry rather than from the array.
NUM_WAYPOINTS = 50
WAYPOINT_DT = 0.1

#: Centred box filter width, in history samples, applied before differentiating.
#: 3 at 10 Hz spans 0.3 s -- the reference used 5 at 20 Hz, i.e. 0.25 s.
SMOOTH_WIDTH = 3

#: Frames kept per episode. 1.5 s at comma2k19's 20 Hz is 30; the margin covers
#: a jittery cadence without holding a whole sequence's worth of decoded PNGs.
HISTORY_KEEP = 64

VIEW_MODES = ("replicate", "crop")

#: Overlapping thirds, as a fraction of image width, for the `crop` ablation.
CROP_SPAN = 0.55

#: `Command` -> (nav_command index, one-hot driving command). The one-hot is
#: ordered [left, straight, right, unknown]; the index selects from
#: `scene.NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")`. Both were
#: cross-checked against the released WOD scene files, where a left turn has
#: future y = +11.98 m and a right turn y = -11.0 m.
COMMANDS: dict[int, tuple[int, tuple[float, float, float, float]]] = {
    Command.STRAIGHT: (0, (0.0, 1.0, 0.0, 0.0)),
    Command.FOLLOW_REFERENCE: (0, (0.0, 1.0, 0.0, 0.0)),
    Command.LEFT: (1, (1.0, 0.0, 0.0, 0.0)),
    Command.RIGHT: (2, (0.0, 0.0, 1.0, 0.0)),
}

#: Below this the recorded speed channel is treated as absent rather than as a
#: parked car, and the pose-derived speed is used instead. A source that fills
#: neither is a source this model cannot be run on; `needs_speed` says so.
SPEED_FLOOR_MS = 0.1


def _env_or(name: str, default):
    """`default` unless the environment names a value, cast to default's type."""
    raw = os.environ.get(name)
    return type(default)(raw) if raw else default


def smooth(values: np.ndarray, width: int = SMOOTH_WIDTH) -> np.ndarray:
    """Centred box filter, edges extended by odd reflection.

    The padding is the whole point. Replicating the endpoint -- the obvious
    choice, and the one the reference used, where it only ever touched the ends
    of a segment -- flattens the filter's last samples towards the last value,
    and here the last sample is `t = 0`: the frame being planned from. A car
    braking at a steady 3 m/s^2 came out at 2 m/s^2, a third of the deceleration
    lost at exactly the moment the expert reads it. Odd reflection continues the
    local slope instead, so a constant acceleration stays constant through the
    boundary.
    """
    if width <= 1 or len(values) < 2:
        return values
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="reflect", reflect_type="odd")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def interp_angle(target: np.ndarray, source: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Interpolate an angle through its sine and cosine, so wrap-around is safe.

    Linear interpolation of the angle itself turns a crossing of +-pi into most
    of a lap, which on a 10 Hz history reads as a violent spin that never
    happened.
    """
    return np.arctan2(np.interp(target, source, np.sin(angle)), np.interp(target, source, np.cos(angle)))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Fold to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def synthesize_views(image: PILImage, mode: str) -> dict[str, PILImage]:
    """Turn one road camera into the three views the model requires.

    `replicate` is the default and the measured winner; see the module
    docstring. `crop` takes overlapping `CROP_SPAN` slices -- the centre keeps
    the vanishing point, the sides keep the shoulders -- which gives real
    lateral content at the price of extrinsics that are simply wrong, since a
    crop is not a rotated camera.
    """
    if mode == "replicate":
        return dict.fromkeys(CAMERA_VIEWS, image)
    if mode == "crop":
        width, height = image.size
        span = int(width * CROP_SPAN)
        boxes = {
            "<FRONT VIEW>": ((width - span) // 2, 0, (width + span) // 2, height),
            "<FRONT LEFT VIEW>": (0, 0, span, height),
            "<FRONT RIGHT VIEW>": (width - span, 0, width, height),
        }
        return {view: image.crop(box) for view, box in boxes.items()}
    raise ValueError(f"unknown view mode {mode!r}, expected one of {VIEW_MODES}")


class Frame:
    """One recorded frame, decoded and cut into views only if the model reads it.

    An open-loop harness hands over the whole history the model would have had:
    comma2k19 sends 160 observations and Qwen-Drive looks at four. Those frames
    are PNGs, and decoding them is the single largest cost on this benchmark --
    59 % of wall clock against 10 % in the network, measured on the openpilot
    sweep. Decoding on record would spend all of it on frames read for their
    pose alone.

    The decode is also shared across the three view slots, so `replicate` opens
    one file rather than three.
    """

    __slots__ = ("_mode", "_obs", "_views")

    def __init__(self, obs: Observation, mode: str) -> None:
        self._obs = obs
        self._mode = mode
        self._views: dict[str, PILImage] | None = None

    @property
    def timestamp_us(self) -> int:
        return self._obs.timestamp_us

    def view(self, name: str) -> PILImage:
        if self._views is None:
            image = Image.fromarray(self._obs.image(CAMERA)).convert("RGB")
            self._views = synthesize_views(image, self._mode)
        return self._views[name]

    def loader(self, name: str):
        """A zero-argument loader, which is what `CameraFrame` accepts.

        Deferring to here means the processor -- not this policy -- decides when
        a frame is worth decoding, and a scene that is built and discarded costs
        nothing.
        """

        def load(name: str = name) -> PILImage:
            return self.view(name)

        return load

    def release(self) -> None:
        """Drop the decoded pixels, keeping the pose and timestamp.

        A 1164x874 RGB frame is 3 MB and `Observation.image` caches it in place,
        so a sequence held whole would carry half a gigabyte of images nobody
        will look at again.
        """
        self._views = None
        self._obs.images.pop(CAMERA, None)


class QwenDrivePolicy(Policy):
    cameras = (CAMERA,)
    waypoint_dt = WAYPOINT_DT

    def __init__(
        self,
        weights: str | None = None,
        planner: str | None = None,
        mode: str | None = None,
        view_mode: str | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        # The checkpoint directory, and the planner head inside it. `planner-sft`
        # serves both planning modes; `planner-rl` is reasoning-only, so it is
        # not the default -- picking it would silently change which modes work.
        self.weights = weights or os.environ.get("QWENDRIVE_WEIGHTS", "")
        self.planner = planner or _env_or("QWENDRIVE_PLANNER", "planner-sft")
        self.mode = mode or _env_or("QWENDRIVE_MODE", "reasoning_planning")
        self.view_mode = view_mode or _env_or("QWENDRIVE_VIEWS", "replicate")
        # One sample. The three the reference drew differed by a standard
        # deviation of 0.09-0.14 m at t=5 s, so best-of-3 bought 0.03-0.04 ADE
        # -- and best-of-N needs the ground truth, which a deployed planner does
        # not have. Sample 0 is what it would follow.
        self.num_samples = num_samples if num_samples is not None else _env_or("QWENDRIVE_SAMPLES", 1)
        # Passed explicitly rather than left to the checkpoint's `noise_seed`,
        # so a recorded number can be re-derived from this file alone.
        self.seed = seed if seed is not None else _env_or("QWENDRIVE_SEED", 42)
        # sdpa, not flash-attn: see setup.sh for why flash-attn cannot be built
        # here. It changes speed, not results.
        self.attn = _env_or("QWENDRIVE_ATTN", "sdpa")
        if self.view_mode not in VIEW_MODES:
            raise ValueError(f"QWENDRIVE_VIEWS={self.view_mode!r}, expected one of {VIEW_MODES}")
        self.model = None
        self._last_reasoning = ""
        self._speed_source = "unknown"
        self._n_infer = 0
        self.reset()

    # ---- lifecycle -------------------------------------------------------

    def load(self) -> None:
        # qwen_drive and torch exist only inside the model's own environment.
        from pathlib import Path

        import torch
        from qwen_drive.modeling_qwen_drive import InferenceMode, QwenDriveForPlanning

        if not self.weights or not Path(self.weights).is_dir():
            raise RuntimeError(
                f"QWENDRIVE_WEIGHTS={self.weights!r} is not a directory. Point it at the "
                "Qwen-Drive-1.0-4B snapshot (the one holding config.json and planner-sft/)."
            )
        # A bare head name resolves inside the snapshot; an absolute path wins,
        # so a head can be swapped without copying the whole checkpoint.
        planner = Path(self.planner)
        if not planner.is_absolute():
            planner = Path(self.weights) / planner

        self._mode_enum = InferenceMode(self.mode)
        self.model = (
            QwenDriveForPlanning.from_pretrained(
                self.weights,
                planner=planner,
                dtype=torch.bfloat16,
                attn_implementation=self.attn,
            )
            .to("cuda")
            .eval()
        )
        _log.info(
            "qwen-drive loaded: %.3f B params, mode=%s planner=%s views=%s",
            sum(p.numel() for p in self.model.parameters()) / 1e9,
            self.mode,
            planner.name,
            self.view_mode,
        )

    def reset(self) -> None:
        """Drop the pose, speed and frame history between sequences."""
        self.timestamps: list[int] = []
        self.poses: list[np.ndarray] = []
        self.speeds: list[float] = []
        self.yaw_rates: list[float] = []
        self.frames: list[Frame] = []

    # ---- history ---------------------------------------------------------

    def _record(self, obs: Observation) -> None:
        """Append one observation.

        A frame stamped no later than the last one is the same frame again --
        an open-loop warm-up that clamps at the start of a sequence repeats
        frame 0 -- and is dropped, because the history is resampled by time and
        `np.interp` needs a strictly increasing x.
        """
        if self.timestamps and obs.timestamp_us <= self.timestamps[-1]:
            return
        pose = obs.ego2world
        self.poses.append(
            np.array([pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0])], dtype=np.float64)
        )
        self.timestamps.append(obs.timestamp_us)
        self.speeds.append(obs.speed)
        self.yaw_rates.append(obs.yaw_rate)
        self.frames.append(Frame(obs, self.view_mode))
        if len(self.timestamps) > HISTORY_KEEP:
            dropped, self.frames = self.frames[:-HISTORY_KEEP], self.frames[-HISTORY_KEEP:]
            for frame in dropped:
                frame.release()
            self.timestamps = self.timestamps[-HISTORY_KEEP:]
            self.poses = self.poses[-HISTORY_KEEP:]
            self.speeds = self.speeds[-HISTORY_KEEP:]
            self.yaw_rates = self.yaw_rates[-HISTORY_KEEP:]

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(offsets before now, wanted absolute times) for the 10 Hz history.

        Clipped to what was actually recorded rather than extrapolated: a short
        history should read as a car that was standing still before the
        sequence began, not as one that teleported in.
        """
        offsets = np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT
        times = np.asarray(self.timestamps, dtype=np.float64) / 1e6
        return offsets, np.clip(times[-1] - offsets, times[0], times[-1])

    def _poses_in_current_frame(self, wanted: np.ndarray) -> np.ndarray:
        """(N, 3) history poses as (x, y, heading) in the ego frame of now.

        The last row is exactly (0, 0, 0) by construction, which is what the
        model's prompt asserts about it.
        """
        times = np.asarray(self.timestamps, dtype=np.float64) / 1e6
        track = np.stack(self.poses)
        if len(times) == 1:
            world = np.repeat(track[:1], len(wanted), axis=0)
        else:
            world = np.stack(
                [
                    np.interp(wanted, times, track[:, 0]),
                    np.interp(wanted, times, track[:, 1]),
                    interp_angle(wanted, times, track[:, 2]),
                ],
                axis=1,
            )
        x0, y0, h0 = world[-1]
        cos0, sin0 = np.cos(h0), np.sin(h0)
        dx, dy = world[:, 0] - x0, world[:, 1] - y0
        return np.stack([cos0 * dx + sin0 * dy, -sin0 * dx + cos0 * dy, wrap_angle(world[:, 2] - h0)], axis=1)

    def _ego_series(self, local: np.ndarray, wanted: np.ndarray, t_rel: np.ndarray):
        """Body-frame (velocity, acceleration) on the history grid, each (N, 2).

        Every term comes from a first-class protocol field or from the pose
        history the model is also shown, so the two cannot disagree silently:

        ``vx``  the recorded speed, resampled. A sensor reading rather than a
                derivative, so it carries no differentiation lag at t = 0 --
                which is the one sample the expert is most sensitive to, and the
                one where a policy has no future frames to centre a difference
                on. A source that records no speed falls back to the poses, and
                `diagnostics()["speed_source"]` says which was used.
        ``vy``  sideslip, from the pose history. Near zero for a road vehicle;
                deriving it is still better than asserting a zero, because a
                hard-coded zero would hide a genuine frame error.
        ``ax``  differentiated from a smoothed ``vx``.
        ``ay``  ``vx * yaw_rate``. In planar vehicle motion the centripetal term
                dominates lateral acceleration, and the recorded yaw rate is far
                quieter than differentiating a 10 Hz heading would be -- at
                30 m/s, 0.01 rad of heading noise over 0.1 s is 3 m/s^2 of
                fictitious lateral acceleration.
        """
        times = np.asarray(self.timestamps, dtype=np.float64) / 1e6
        heading = local[:, 2]
        if len(times) > 1:
            velocity = np.stack([np.gradient(smooth(local[:, i]), t_rel) for i in range(2)], axis=1)
        else:
            velocity = np.zeros((len(local), 2))
        # local is already expressed in the frame of *now*; rotate each sample
        # further into the frame the car was pointing at the time.
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        vx_pose = cos_h * velocity[:, 0] + sin_h * velocity[:, 1]
        vy = -sin_h * velocity[:, 0] + cos_h * velocity[:, 1]

        recorded = np.asarray(self.speeds, dtype=np.float64)
        vx = np.interp(wanted, times, recorded) if len(times) > 1 else np.full(len(local), recorded[-1])
        if np.max(np.abs(vx)) < SPEED_FLOOR_MS < np.max(np.abs(vx_pose)):
            vx, self._speed_source = vx_pose, "pose"
        else:
            self._speed_source = "recorded"

        yaw_rate = (
            np.interp(wanted, times, np.asarray(self.yaw_rates, dtype=np.float64))
            if len(times) > 1
            else np.zeros(len(local))
        )
        ax = np.gradient(smooth(vx), t_rel) if len(times) > 1 else np.zeros(len(local))
        return np.stack([vx, vy], axis=1), np.stack([ax, vx * yaw_rate], axis=1)

    def _view_frames(self) -> tuple[list[Frame], list[float]]:
        """The four frames nearest `VIEW_OFFSETS_S`, oldest first, and their ages.

        Nearest-by-timestamp rather than by index, because a source's request
        cadence is not paddock's business: comma2k19's camera runs at 19.93 Hz,
        not 20, and index arithmetic drifts ~0.2 s over a minute -- comparable to
        the 0.5 s spacing the prompt claims these frames have.
        """
        times = np.asarray(self.timestamps, dtype=np.float64) / 1e6
        ages = []
        chosen = []
        for offset in VIEW_OFFSETS_S:
            index = int(np.argmin(np.abs(times - (times[-1] - offset))))
            chosen.append(self.frames[index])
            ages.append(float(times[-1] - times[index]))
        return chosen, ages

    # ---- inference -------------------------------------------------------

    def _scene(self, obs: Observation):
        from qwen_drive.scene import CameraFrame, DrivingScene

        offsets, wanted = self._grid()
        local = self._poses_in_current_frame(wanted)
        velocity, acceleration = self._ego_series(local, wanted, -offsets)

        nav_command, driving_command = COMMANDS[int(obs.command)]
        frames, self._view_ages = self._view_frames()
        # Frames carry no `target_size`, so the processor falls back to the
        # config's history/current pixel budgets. That keeps the source's own
        # aspect ratio instead of forcing the portrait crop the released
        # benchmark metadata encodes for Waymo imagery.
        views = {view: [CameraFrame(image=f.loader(view)) for f in frames] for view in CAMERA_VIEWS}
        return DrivingScene(
            views=views,
            history=local.astype(np.float32),
            history_velocity=velocity.astype(np.float32),
            history_acceleration=acceleration.astype(np.float32),
            ego_velocity=tuple(velocity[-1]),
            ego_acceleration=tuple(acceleration[-1]),
            driving_command=driving_command,
            nav_command=nav_command,
            token=str(obs.timestamp_us),
            metadata={"view_mode": self.view_mode},
        )

    def observe(self, obs: Observation) -> None:
        """Record a history frame without sampling the model.

        The default `observe` runs a full rollout per frame and keeps nothing.
        Qwen-Drive is a pure function of its recorded history, so on comma2k19
        that would be 160 VLM rollouts to use one, and a ~1.5 s sample would
        take four minutes. Recording N-1 frames and planning on the last gives
        the identical scene -- and, since the sampler is seeded, the identical
        trajectory.
        """
        self._record(obs)

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        self._record(obs)

        scene = self._scene(obs)
        output = self.model.run(self._mode_enum, scene=scene, num_samples=self.num_samples, seed=self.seed)
        # (num_samples, 50, 3) of (x, y, heading), ego frame, 10 Hz. Sample 0 is
        # the deployed one; paddock scores what would be driven.
        waypoints = np.asarray(output.trajectories, dtype=np.float64)[0, :, :2]
        self._last_reasoning = (output.reasoning or "")[:400]

        for frame in self.frames:
            frame.release()
        self._n_infer += 1
        if self._n_infer % 100 == 1:
            _log.info(
                "infer #%d v=%.1f m/s (%s) reach %.1f m: %s",
                self._n_infer,
                scene.ego_velocity[0],
                self._speed_source,
                float(np.linalg.norm(waypoints[-1])),
                self._last_reasoning[:200],
            )
        return Trajectory(waypoints, dt=self.waypoint_dt)

    # ---- diagnostics -----------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """What would make a number wrong without making it fail.

        `history_span_s` and `speed_recorded` vs `speed_pose` are the two to
        watch. A history spanning much less than 1.5 s means the source did not
        feed one and the model is planning from a near-standstill it invented; a
        speed disagreement beyond a percent or so means the pose frame, the
        timestamps or the odometer channel are not describing the same motion,
        and the reference's own conversion check was exactly this comparison.
        """
        if not self.timestamps:
            return {"status": "no frame seen yet", "view_mode": self.view_mode, "mode": self.mode}
        offsets, wanted = self._grid()
        local = self._poses_in_current_frame(wanted)
        velocity, _ = self._ego_series(local, wanted, -offsets)
        times = np.asarray(self.timestamps, dtype=np.float64) / 1e6
        pose_step = np.hypot(*(local[-1, :2] - local[-2, :2])) if len(local) > 1 else 0.0
        return {
            "mode": self.mode,
            "planner": self.planner,
            "attn": self.attn,
            "view_mode": self.view_mode,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "nav_command": "GO STRAIGHT",
            "recorded_frames": len(self.timestamps),
            "history_span_s": round(float(times[-1] - times[0]), 3),
            "request_dt_s": (round(float(times[-1] - times[-2]), 4) if len(times) > 1 else None),
            "view_ages_s": [round(a, 3) for a in getattr(self, "_view_ages", [])],
            "speed_source": self._speed_source,
            "speed_recorded": round(self.speeds[-1], 3),
            "speed_pose": round(float(pose_step / HISTORY_DT), 3),
            "reasoning": self._last_reasoning,
        }


POLICY = QwenDrivePolicy
