"""NVIDIA Alpamayo 1.5 as a paddock Policy.

Adapted from the NeuroNCAP model node. The camera crop it used to fork now
lives once in `camera_warp.crop_box`, shared with the open-loop path -- see
tests/test_camera_warp.py for why that mattered.

Alpamayo is a 10B vision-language-action model: the VLM writes a
chain-of-causation trace, then a diffusion expert samples trajectories
conditioned on its hidden states. Three adaptations matter.

*Cameras.* Alpamayo takes four slots -- three nominally 120 deg views at
+55/0/-55 deg and a 30 deg telephoto. nuScenes' front three cameras sit almost
exactly at those bearings, which is the good news. The bad news is field of
view: nuScenes gives about 65 deg where the model expects 120, so the three wide
slots are filled with the widest 16:9 crop available and objects subtend roughly
1.8x the angle the model was trained to associate with a given distance. The
KITScenes run hit the same wall at 87 deg and found `fill` still beat
`letterbox` overall, so `fill` is the default here too -- but this is the single
largest source of doubt in Alpamayo's numbers on this benchmark, and it biases
depth. The telephoto slot is exact, because 30 deg is narrower than the source.

*History.* The model wants 16 ego poses at dt=0.1 and 4 camera frames, and
derives its initial speed purely by differencing that history -- a wrong rate
produces a confident, meaningless trajectory. That is why this node insists on
a 0.1 s simulation step (`--engine.scenario.time-step 0.1`) rather than
resampling: at 10 Hz the request stream *is* the history, one frame per step.
A mismatched step is rejected loudly instead of silently mis-scaling the speed.

*Output.* Alpamayo predicts 64 waypoints at 10 Hz (6.4 s). NeuroNCAP wants six
on a 0.5 s grid, so indices 4, 9, ... 29 are taken -- the same subsampling the
KITScenes harness used. With more than one sample drawn, the medoid is
executed: the closed loop can only drive one trajectory, and the medoid is the
consensus draw rather than an oracle-selected best.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # annotations only -- torch lives in the model's own env
    import torch

from paddock.adapters.models.alpamayo15.camera_warp import (
    TARGET_H,
    TARGET_W,
    TELE_HFOV_DEG,
    crop_box,
)
from paddock.adapters.models.alpamayo15.camera_warp import (
    Pinhole as WarpPinhole,
)
from paddock.policy.protocol import Observation, Pinhole, Policy, Trajectory

_log = logging.getLogger(__name__)

# Alpamayo slot -> (nuScenes camera, how to fill it). Matched by bearing:
# nuScenes' front-left/front-right sit within a couple of degrees of the
# +55/-55 the model expects.
CAMERA_SLOTS: list[tuple[int, str, str]] = [
    (0, "CAM_FRONT_LEFT", "fill"),
    (1, "CAM_FRONT", "fill"),
    (2, "CAM_FRONT_RIGHT", "fill"),
    (6, "CAM_FRONT", "zoom"),
]
NEEDED_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

TARGET_ASPECT = TARGET_H / TARGET_W  # 9:16, the model frame's height/width


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw else None


def _env_or(name: str, default):
    """`default` unless the environment names a value, cast to default's type."""
    raw = os.environ.get(name)
    return type(default)(raw) if raw else default


def nav_text_for(maneuver: str | None, distance_m: float | None) -> str:
    """The navigation instruction in the phrasing the model was trained on.

    The reference notebook and `nav_utils` use "Turn left in 30m" /
    "Turn right in 9m" for turns and "Continue straight" otherwise; a route with
    no junction ahead is the "otherwise". Distances are whole metres, never
    negative.
    """
    if maneuver in ("TURN_LEFT", "TURN_RIGHT"):
        side = "left" if maneuver == "TURN_LEFT" else "right"
        if distance_m is None:  # a source with a turn command but no distance (NAVSIM)
            return f"Turn {side}"
        return f"Turn {side} in {max(0, int(round(distance_m)))}m"
    return "Continue straight"


def route_maneuver(obs) -> tuple[str | None, float | None]:
    """The next maneuver for *obs*: the explicit fields when the source sends
    them (Bench2Drive, KITScenes), else the 4-way command every source carries
    (NAVSIM's driving command), so no benchmark leaves the model without its
    route instruction."""
    man = getattr(obs, "nav_maneuver", None)
    if man:
        return man, getattr(obs, "nav_distance_m", None)
    cmd = int(getattr(obs, "command", 3))
    return ({1: "TURN_LEFT", 0: "TURN_RIGHT"}.get(cmd), None)


NUM_HISTORY_STEPS = 16
NUM_CAM_FRAMES = 4
HISTORY_DT = 0.1
MODEL_HZ = 10.0
# 64 waypoints at 10 Hz -> the eight 0.5 s points NAVSIM scores (0.5 .. 4.0 s).
#
# Extending this from six changes neither of the other two benchmarks, which is
# why it is safe to widen rather than fork: `Trajectory.to_wire` hands NeuroNCAP
# the first six, and KITScenes' `displacement_errors` truncates at its 3 s
# horizon, so points seven and eight are dropped by both. It also brings the
# policy into line with `benchmarks/kitscenes/metrics.GRID_IDX`, which already
# subsamples eight.
GRID_INDICES = np.array([4, 9, 14, 19, 24, 29, 34, 39])


def to_pinhole(p: Pinhole) -> WarpPinhole:
    """Protocol intrinsics in the shape the warp code expects."""
    return WarpPinhole(focal=p.fx, cu=p.cx, cv=p.cy, width=p.width, height=p.height)


def warp(image: Image.Image, src: WarpPinhole, kind: str) -> np.ndarray:
    """Crop to the slot's field of view and rescale to the model's frame."""
    box = crop_box(src, TELE_HFOV_DEG if kind == "zoom" else None)
    resized = image.crop(box).resize((TARGET_W, TARGET_H), Image.BICUBIC)
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def medoid(trajectories: np.ndarray) -> np.ndarray:
    """The draw closest to all the others: (K, T, D) -> (T, D).

    Executed rather than the best-scoring draw, because in closed loop there is
    no ground truth to score against and only one trajectory can be driven.
    """
    if len(trajectories) == 1:
        return trajectories[0]
    flat = trajectories.reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).sum(axis=1)
    return trajectories[int(np.argmin(distances))]


class AlpamayoPolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = 0.5

    def __init__(
        self,
        num_traj_samples: int | None = None,
        reserve_gb: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_generation_length: int | None = None,
        use_nav: bool | None = None,
    ) -> None:
        # Sampling defaults follow the reference inference notebook
        # (notebooks/inference.ipynb: top_p 0.98, temperature 0.6, one sample,
        # 256 generation tokens). More samples give a medoid to execute, but the
        # VLM writes one chain-of-thought per sample (num_return_sequences), so
        # 6 samples measured 3.7 s per tick against 1.0 s for one on this card.
        # Each is overridable from the environment.
        self.num_traj_samples = num_traj_samples or _env_or("ALPAMAYO_NUM_TRAJ_SAMPLES", 1)
        # Claim VRAM before the slow checkpoint read. On a shared card the
        # alternative is discovering someone else took it, 12 minutes in.
        self.reserve_gb = reserve_gb or _env_float("ALPAMAYO_RESERVE_GB")
        self.temperature = temperature or _env_or("ALPAMAYO_TEMPERATURE", 0.6)
        self.top_p = top_p or _env_or("ALPAMAYO_TOP_P", 0.98)
        self.max_generation_length = max_generation_length or _env_or("ALPAMAYO_MAX_GEN", 256)
        # Navigation conditioning: the observation's next-maneuver fields become
        # the model's route instruction ("Turn left in 40m"). ALPAMAYO_NAV=0
        # runs the no-nav baseline.
        self.use_nav = use_nav if use_nav is not None else _env_or("ALPAMAYO_NAV", 1) == 1
        self._last_nav: str | None = None
        self.model = None
        self.processor = None
        self.reset()

    def load(self) -> None:
        # alpamayo1_5 and torch exist only inside the model's own environment.
        from paddock.adapters.models.alpamayo15.model_loader import load_model

        self.model, self.processor = load_model(reserve_gb=self.reserve_gb)

    def reset(self) -> None:
        """Drop the pose and image history between runs."""
        self.poses: list[np.ndarray] = []
        self.timestamps: list[int] = []
        self.frames: list[dict[str, np.ndarray]] = []

    def _sample_poses(self, offsets_s: np.ndarray) -> np.ndarray:
        """Ego poses at the given offsets before now, interpolated over time.

        The history is resampled by timestamp rather than by taking the last N
        requests, because NeuroNCAP's request cadence is not uniform: the
        priming steps at the start of every run advance by dataset keyframes
        (0.5 s) whatever the scenario time step is, and only the closed-loop
        steps honour it. Taking the last 16 requests would therefore hand
        Alpamayo a history spanning 0.5 s intervals labelled as 0.1 s ones, and
        it derives its initial speed by differencing exactly those -- so it
        would confidently plan for a car going five times too fast.

        Translation is interpolated linearly and rotation with slerp, so a
        sparse early history still yields metrically correct spacing rather
        than repeated poses reading as a standing start.
        """
        from scipy.spatial import transform as spt

        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        stack = np.stack(self.poses)
        now = times[-1]
        wanted = np.clip(now - offsets_s, times[0], now)

        xyz = np.stack([np.interp(wanted, times, stack[:, i, 3]) for i in range(3)], axis=1)
        rotations = spt.Rotation.from_matrix(stack[:, :3, :3])
        if len(times) == 1:
            matrices = np.repeat(stack[:1, :3, :3], len(wanted), axis=0)
        else:
            matrices = spt.Slerp(times, rotations)(wanted).as_matrix()

        poses = np.repeat(np.eye(4)[None], len(wanted), axis=0)
        poses[:, :3, :3] = matrices
        poses[:, :3, 3] = xyz
        return poses

    def _history(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The last 16 ego poses at dt=0.1, expressed in the current (t0) frame.

        Full SO(3), matching load_physical_aiavdataset rather than a yaw-only
        approximation.
        """
        import torch
        from scipy.spatial import transform as spt

        offsets = np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT
        stack = self._sample_poses(offsets)

        t0_inv = spt.Rotation.from_matrix(stack[-1, :3, :3]).inv()
        xyz = t0_inv.apply(stack[:, :3, 3] - stack[-1, :3, 3])
        rot = (t0_inv * spt.Rotation.from_matrix(stack[:, :3, :3])).as_matrix()
        return (
            torch.from_numpy(xyz).float()[None, None],
            torch.from_numpy(rot).float()[None, None],
        )

    def _select_frames(self) -> list[dict[str, np.ndarray]]:
        """The NUM_CAM_FRAMES camera frames to condition on, oldest last-but-one.

        Two strategies, because the right one depends on data paddock does not
        control, and picking silently would hide the tradeoff:

        ``grid`` (default) takes the frame nearest each point of a 0.1 s grid.
        Correct whenever the source is at least 10 Hz -- neuroNCAP and KITScenes
        both are -- and it tolerates a jittery request cadence, which is why it
        was written this way. On a slower source it degenerates: at NAVSIM's
        2 Hz it returns frame indices [2, 3, 3, 3], two distinct images in four
        slots, so the model is shown a world that is not moving while its pose
        history says otherwise.

        ``distinct`` takes the most recent distinct frames instead. The model
        still believes they are 0.1 s apart, so on a slower source apparent
        motion reads high rather than absent -- a different error, not no error.

        ``auto`` (the default) uses ``grid`` unless the *source* is slow: the
        frames that would fill the slots are themselves all at least
        2 x HISTORY_DT apart. Testing the grid's output alone was not enough --
        neuroNCAP primes two steps at 0.5 s dataset keyframes before its 0.1 s
        loop, so at the first closed-loop steps the history is
        [t, t+0.5, t+1.0, t+1.1]: the grid repeats a frame there too, yet the
        newest frames are 0.1 s apart and the recorded neuroNCAP scores were
        made on that path. Deciding from the source cadence keeps that
        transition on ``grid`` (pinned by test) and still catches NAVSIM, where
        every gap is 0.5 s.

        Which error is smaller was measured rather than argued. On 60 NAVSIM
        scenarios, paired, ``distinct`` beat ``grid`` 0.710 to 0.503 EPDMS
        (better on 36, worse on 19, sign test p = 0.03), and the term that moved
        was collisions: ``no_at_fault_collisions`` 0.708 -> 0.992. Shown four
        copies of one frame the model cannot see other vehicles moving, so it
        drives into them. See docs/benchmarks/navsim.md.
        """
        mode = os.environ.get("ALPAMAYO_FRAME_SELECT", "auto")
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        wanted = times[-1] - np.arange(NUM_CAM_FRAMES - 1, -1, -1) * HISTORY_DT
        grid = [int(np.argmin(np.abs(times - w))) for w in wanted]

        gaps = np.diff(times[-NUM_CAM_FRAMES:])
        slow_source = len(self.frames) >= NUM_CAM_FRAMES and bool(gaps.min() >= 2 * HISTORY_DT)
        degenerate = len(set(grid)) < NUM_CAM_FRAMES
        if mode == "distinct" or (mode == "auto" and slow_source and degenerate):
            recent = self.frames[-NUM_CAM_FRAMES:]
            # Early in an episode there may be fewer frames than slots; repeat
            # the oldest rather than reaching past the start of the sequence.
            while len(recent) < NUM_CAM_FRAMES:
                recent = [recent[0], *recent]
            return recent
        return [self.frames[i] for i in grid]

    def _image_frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N_cam, 4, 3, H, W) uint8, cameras in ascending slot order."""
        import torch

        recent = self._select_frames()

        ordered = sorted(CAMERA_SLOTS, key=lambda entry: entry[0])
        per_camera = []
        for _, camera, kind in ordered:
            warped = np.stack([frame[f"{camera}:{kind}"] for frame in recent])
            per_camera.append(torch.from_numpy(warped).permute(0, 3, 1, 2))
        indices = torch.tensor([slot for slot, _, _ in ordered], dtype=torch.int64)
        return torch.stack(per_camera, dim=0), indices

    def _warp_frame(self, obs: Observation) -> dict[str, np.ndarray]:
        warped = {}
        for _, camera, kind in CAMERA_SLOTS:
            key = f"{camera}:{kind}"
            if key in warped:
                continue
            image = Image.fromarray(obs.image(camera))
            pinhole = to_pinhole(obs.pinhole(camera))
            warped[key] = warp(image, pinhole, kind)
        return warped

    def _record(self, obs: Observation) -> None:
        """Append one observation to the pose and frame history.

        A frame stamped no later than the last one is the same frame again (the
        KITScenes warm-up clamps its 48-frame history at the scene start, so a
        keyframe at 40 repeats frame 0 eight times) and is dropped: the history
        is resampled by time, and Slerp needs strictly increasing times.
        """
        if self.timestamps and obs.timestamp_us <= self.timestamps[-1]:
            return
        self.poses.append(obs.ego2world)
        self.timestamps.append(obs.timestamp_us)
        self.frames.append(self._warp_frame(obs))
        # A run is at most a few hundred steps, but each entry holds four warped
        # 1920x1080 frames, so old ones are dropped once they are out of reach.
        if len(self.poses) > 64:
            self.poses, self.timestamps, self.frames = (
                self.poses[-64:],
                self.timestamps[-64:],
                self.frames[-64:],
            )

    def observe(self, obs: Observation) -> None:
        """Record a history frame without sampling the model.

        The default `observe` runs a full VLM rollout per frame and keeps
        nothing: on NAVSIM that is four 10B-parameter chain-of-causation
        generations to use one, and it made alpamayo the long pole of the sweep
        at ~17 h. The model conditions on the recorded history alone, so
        recording N-1 frames and sampling on the last draws from the same
        distribution -- not the same draw, since alpamayo samples unseeded, but
        it was never the same draw run to run either.
        """
        self._record(obs)

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")

        import torch
        from alpamayo1_5 import helper

        self._record(obs)

        hist_xyz, hist_rot = self._history()
        image_frames, camera_indices = self._image_frames()

        nav_text = None
        if self.use_nav:
            nav_text = nav_text_for(*route_maneuver(obs))
        self._last_nav = nav_text
        messages = helper.create_message(
            frames=image_frames.flatten(0, 1), camera_indices=camera_indices, nav_text=nav_text
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {"tokenized_data": inputs, "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}, "cuda"
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=self.top_p,
                temperature=self.temperature,
                num_traj_samples=self.num_traj_samples,
                max_generation_length=self.max_generation_length,
                return_extra=True,
            )

        samples = pred_xyz.float().cpu().numpy()[0, 0]  # (K, 64, 3)
        chosen = medoid(samples)
        # Alpamayo is x-forward, y-left, the same convention NeuroNCAP drives.
        waypoints = chosen[GRID_INDICES][:, :2]

        # extract_text_tokens returns {"cot": [B, sets, samples] of str, ...}; the
        # executed draw's chain-of-causation is the one worth keeping.
        cot = ""
        if isinstance(extra, dict) and "cot" in extra:
            flat = np.asarray(extra["cot"]).reshape(-1)
            cot = str(flat[0]) if flat.size else ""
        self._last_reasoning = cot[:400]
        self._n_infer = getattr(self, "_n_infer", 0) + 1
        if self._n_infer % 100 == 1:
            _log.info(
                "infer #%d nav=%r reach %.2f m cot=%r",
                self._n_infer,
                nav_text,
                float(np.linalg.norm(waypoints[-1])),
                cot[:240],
            )
        return Trajectory(waypoints, dt=self.waypoint_dt)

    def diagnostics(self) -> dict[str, Any]:
        """Request cadence and history depth, plus the latest reasoning trace.

        `request_dt_s` is the one to watch. Alpamayo derives its initial speed
        by differencing a history it believes is spaced at 0.1 s; if the
        simulator is stepping at 0.5 s this reads five times too fast and the
        model plans confidently for a car that is not there.
        """
        if not self.timestamps:
            return {"status": "no frame seen yet"}
        return {
            "num_traj_samples": self.num_traj_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "nav": self.use_nav,
            "nav_text": self._last_nav,
            "history_frames": min(len(self.poses), NUM_HISTORY_STEPS),
            "expected_dt_s": HISTORY_DT,
            "request_dt_s": (
                round((self.timestamps[-1] - self.timestamps[-2]) / 1e6, 3)
                if len(self.timestamps) > 1
                else None
            ),
            "reasoning": getattr(self, "_last_reasoning", ""),
        }


POLICY = AlpamayoPolicy
