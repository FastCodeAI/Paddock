"""NVIDIA Alpamayo 2 Edge as a paddock Policy.

Alpamayo 2 Edge is a 2B vision-language-action model on the Cosmos3-Edge
backbone: the VLM writes its driving reasoning, then a flow-matching expert
decodes 64 future poses at 0.1 s. Built on alpamayo15's adapter, but its input
contract is stricter and different in three ways, each of which this file
reproduces from upstream's own loader (`load_physical_aiavdataset.py` and
`sample.py` at the sha in repos.yaml) rather than from Alpamayo 1.5's:

*Six cameras, not four.* Camera ids 0/1/2 are the 120 deg cross-left, front-wide
and cross-right views, 3/5 the 70 deg rear-left and rear-right, and 6 the 30 deg
front telephoto. Id 4, the rear telephoto that looks straight back, is not an
Edge input, so paddock's `CAM_BACK` is never read. Each slot is filled by the
camera at the matching bearing (`SLOTS`) and cropped to the slot's field of view
around its principal point (`camera_warp.crop_box`, the crop alpamayo15 uses):
a source wider than the slot is cropped to it exactly; a narrower one gives its
widest 16:9 window, since no crop can invent field of view. The telephoto is a
30 deg crop of `CAM_FRONT`, exact on every source paddock has.

Upstream refuses an input without all six (`sample._validate_camera_inputs`),
so a camera the source does not have is filled with black frames: comma2k19
has only `CAM_FRONT`, and four of its six slots are black. Never with another
camera's image, which would tell the model there is a view at that bearing.
`diagnostics()` names every black slot.

*A 22-frame, two-rate schedule with real timestamps.* Front and telephoto take
five frames at -2, -1, -0.5, -0.1 and 0 s; the other four cameras three, at -1,
-0.1 and 0 s. Each image's capture time enters the prompt as a label
("<t=-0.5s>"), so the time handed over is the recorded time of the frame
actually chosen, never the time that was asked for: on nuScenes' 2 Hz keyframes
the -0.1 s slot holds a frame labelled -0.5 s or 0 s, and the model is told so.

*20 ego poses, not 16.* Positions and full SO(3) rotations at -1.9 .. 0 s in
0.1 s steps, in the ego frame at t0, which the model turns into 57 history
tokens. Interpolated by timestamp, as alpamayo15 does, so a source whose
request cadence is not 10 Hz still yields metrically correct spacing.

Everything after that is upstream's: `helper.prepare_model_inputs` sorts,
labels, resizes and normalises, and `sample_trajectories_from_data` samples with
the CLI's defaults (seed 43, top_p 0.98, temperature 0.6, one reasoning trace,
256 tokens, 10 flow steps), so a plan here is what `python -m
alpamayo2_edge.inference` would produce from the same 22 frames.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # annotations only -- torch lives in the model's own env
    pass

from paddock.adapters.models.alpamayo15.camera_warp import TARGET_H, TARGET_W, crop_box
from paddock.adapters.models.alpamayo15.camera_warp import Pinhole as WarpPinhole
from paddock.adapters.models.alpamayo15.policy import nav_text_for, route_maneuver
from paddock.policy.protocol import Observation, Pinhole, Policy, Trajectory

_log = logging.getLogger(__name__)

#: Upstream camera id -> (paddock camera, field of view the slot was trained on, degrees).
#: Matched by bearing. Every benchmark's CAM_FRONT_LEFT/RIGHT is its forward-diagonal view
#: (nuScenes ~+-55, KITScenes ring front-left/right, NAVSIM l0/r0) and CAM_BACK_LEFT/RIGHT
#: its backward-diagonal one (nuScenes ~+-110, KITScenes +-120, NAVSIM l2/r2), which is where
#: the 70 deg rear-left/right cameras look. Rear-diagonal bearings in the training rig were
#: not available to check against; `diagnostics()` reports each source camera's yaw.
SLOTS: dict[int, tuple[str, float]] = {
    0: ("CAM_FRONT_LEFT", 120.0),
    1: ("CAM_FRONT", 120.0),
    2: ("CAM_FRONT_RIGHT", 120.0),
    3: ("CAM_BACK_LEFT", 70.0),
    5: ("CAM_BACK_RIGHT", 70.0),
    6: ("CAM_FRONT", 30.0),
}
#: Seconds before t0 of each frame a slot is shown (load_physical_aiavdataset._FRONT_OFFSETS_US
#: and _OTHER_OFFSETS_US). `load()` checks the frame counts against upstream's, so a new
#: checkpoint that changes the schedule fails at startup rather than at the first sample.
FRONT_OFFSETS_S = (2.0, 1.0, 0.5, 0.1, 0.0)
OTHER_OFFSETS_S = (1.0, 0.1, 0.0)
OFFSETS_S: dict[int, tuple[float, ...]] = {
    slot: FRONT_OFFSETS_S if slot in (1, 6) else OTHER_OFFSETS_S for slot in SLOTS
}
#: The cameras a source is asked for. CAM_BACK is not among them: see the module docstring.
NEEDED_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

NUM_HISTORY_STEPS = 20
HISTORY_DT = 0.1
#: The model's own 64 poses are 0.1 s apart; 4, 9, ... 39 are the 0.5 s grid out to 4 s,
#: the eight points NAVSIM scores. Six suffice for neuroNCAP and KITScenes truncates at 3 s.
GRID_INDICES = np.array([4, 9, 14, 19, 24, 29, 34, 39])
#: History older than this is never selected: the oldest frame asked for is 2 s back.
KEEP_S = 2.5

#: Decode-and-warp workers. A sample has at most 17 distinct (frame, camera) reads.
WARP_THREADS = 8

#: The CLI's sampling settings (alpamayo2_edge/inference.py `main`).
SEED = 43
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_GENERATION_LENGTH = 256
DIFFUSION_STEPS = 10

#: `ALPAMAYO2_EDGE_REASONING`. "model" (the default) lets the model write its own reasoning.
#: "nominal" forces exactly what it writes when it chooses `<NOMINAL>` for a routine scene, which
#: fixes decode at 7 steps; "none" forces `<|traj_future_start|>` at once. Both are speed
#: ablations, and a run with either is recorded valid: false. The ids are the model's own
#: output on KITScenes samples where it chose `<NOMINAL>`: '<N' 'OM' 'IN' 'AL' '>\n' then
#: <|traj_future_start|>. Tokenising the text "<NOMINAL>" ends in '>' (1062), not the '>\n'
#: (1561) the model writes, so the forced cache would be one the model never produces.
FORCED_REASONING_IDS: dict[str, list[int]] = {
    "nominal": [124549, 11135, 3174, 4286, 1561, 135084],
    "none": [135084],
}
REASONING_MODES = ("model", *FORCED_REASONING_IDS)

#: The model's own plan spacing: 64 poses 0.1 s apart.
PLAN_DT = 0.1
#: `ALPAMAYO2_EDGE_WARM_START_STEPS`: 0 (the released model) draws every plan from fresh noise.
#: N > 0 starts the flow from the previous plan instead, carried to now (`carry_plan`), and
#: runs N Euler steps over the same 0 -> 1 time span. The first plan of a sequence has no
#: previous one and is drawn as usual, with `ALPAMAYO2_EDGE_FLOW_STEPS`.
#: A previous plan older than this is from a dropped frame or another sequence: not carried.
WARM_START_MAX_GAP_S = 0.5
#: How far ahead `carry` compares the new plan with the carried old one: the 3 s neuroNCAP reads.
CARRY_COMPARE_POINTS = 30


def carry_plan(
    xyz: np.ndarray, rot: np.ndarray, plan_ego2world: np.ndarray, ego2world: np.ndarray, elapsed_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """A plan made ``elapsed_s`` ago, as the same path seen from now.

    ``xyz`` (64, 3) and ``rot`` (64, 3, 3) are the model's poses at 0.1 .. 6.4 s in the ego frame
    it planned in (``plan_ego2world``). Returns its poses at 0.1 .. 6.4 s after *now*, in the
    current ego frame: interpolated in time from the origin through the plan, and past its last
    pose continued at its last step's rate (one 0.1 s step, at a 10 Hz call rate). Heading is
    carried as yaw, the only part of a rotation upstream's action encoding reads.
    """
    n = len(xyz)
    times = PLAN_DT * np.arange(n + 1)
    points = np.vstack([np.zeros(3), xyz])
    yaw = np.unwrap(np.concatenate([[0.0], np.arctan2(rot[:, 1, 0], rot[:, 0, 0])]))
    wanted = elapsed_s + PLAN_DT * np.arange(1, n + 1)
    beyond = wanted > times[-1]

    def along(values: np.ndarray) -> np.ndarray:
        out = np.interp(wanted, times, values)
        out[beyond] = values[-1] + (wanted[beyond] - times[-1]) * (values[-1] - values[-2]) / PLAN_DT
        return out

    moved = np.stack([along(points[:, i]) for i in range(3)], axis=1)
    headings = along(yaw)
    rel = np.linalg.inv(np.asarray(ego2world, dtype=np.float64)) @ np.asarray(plan_ego2world, np.float64)
    c, s = np.cos(headings), np.sin(headings)
    rz = np.zeros((n, 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -s, s, c, 1.0
    return moved @ rel[:3, :3].T + rel[:3, 3], np.einsum("ij,njk->nik", rel[:3, :3], rz)


class StageTimer:
    """Per-plan GPU time of each model stage, from ``torch.cuda.synchronize()``-bracketed hooks.

    The four buckets of the compression study's ``stage_latency.py``, which this mirrors so
    the numbers compare: the vision encoder; the language model's prefill (the call with more
    than one token, vision included) and its decode steps; and the action expert, summed over
    its flow steps. Off unless ``ALPAMAYO2_EDGE_STAGE_TIMING=1``: every hook synchronizes the
    card, which costs a little time and changes no plan.
    """

    def __init__(self, torch: Any) -> None:
        self._torch = torch
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._started: dict[str, float] = {}

    def reset(self) -> None:
        self.totals, self.calls = {}, {}

    def _begin(self, key: str) -> None:
        self._torch.cuda.synchronize()
        self._started[key] = time.perf_counter()

    def _end(self, key: str, bucket: str) -> None:
        self._torch.cuda.synchronize()
        self.totals[bucket] = self.totals.get(bucket, 0.0) + time.perf_counter() - self._started[key]
        self.calls[bucket] = self.calls.get(bucket, 0) + 1

    def attach(self, module: Any, name: str) -> None:
        module.register_forward_pre_hook(lambda _m, _i: self._begin(name))
        module.register_forward_hook(lambda _m, _i, _o: self._end(name, name))

    def attach_prefill_vs_decode(self, module: Any) -> None:
        """One bucket per kind of call: "prefill" for a call over more than one token (Edge
        prefills its prompt with a direct call, before generate), "decode" for each one-token
        step generate makes after it."""
        kind = {"bucket": "prefill"}

        def pre(_m, _args, kwargs):
            ids = kwargs.get("input_ids")
            kind["bucket"] = "prefill" if ids is None or ids.shape[1] > 1 else "decode"
            self._begin("lm")

        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.register_forward_hook(lambda _m, _a, _k, _o: self._end("lm", kind["bucket"]), with_kwargs=True)


def _env_or(name: str, default):
    """`default` unless the environment names a value, cast to default's type."""
    raw = os.environ.get(name)
    return type(default)(raw) if raw else default


def to_pinhole(p: Pinhole) -> WarpPinhole:
    """Protocol intrinsics in the shape the crop code expects."""
    return WarpPinhole(focal=p.fx, cu=p.cx, cv=p.cy, width=p.width, height=p.height)


def warp(image: np.ndarray, src: Pinhole, hfov_deg: float) -> np.ndarray:
    """Crop to the slot's field of view around the principal point, resize to 1920x1080.

    1920x1080 is the frame size of every PhysicalAI-AV camera upstream decodes, so
    from here on upstream's own resize to each slot's pixel budget runs unchanged.
    """
    box = crop_box(to_pinhole(src), hfov_deg)
    resized = Image.fromarray(image).crop(box).resize((TARGET_W, TARGET_H), Image.BICUBIC)
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def nearest_index(times_s: np.ndarray, wanted_s: float) -> int:
    """The recorded frame nearest *wanted_s*; a tie goes to the earlier frame.

    Nearest rather than upstream's at-or-before: upstream decodes 30 Hz cameras, where
    at-or-before is at most 33 ms early, but on a 10 Hz source with timing jitter it can
    land a whole frame early. The label carries the chosen frame's real time either way.
    """
    return int(np.argmin(np.abs(times_s - wanted_s)))


def camera_yaw_deg(camera2ego: np.ndarray) -> float:
    """Bearing of a camera's optical axis (its +z) in the ego frame, degrees, left positive."""
    axis = np.asarray(camera2ego, dtype=np.float64)[:3, 2]
    return math.degrees(math.atan2(axis[1], axis[0]))


class Frame:
    """One recorded observation, its slots warped only when a slot selects it.

    An open-loop harness hands over its whole history (comma2k19 sends 160 frames
    for the 13 distinct ones this model reads), and a warp is a full-resolution
    crop and resize. Warping on demand leaves the unread frames free.
    """

    __slots__ = ("_obs", "_warped", "timestamp_us")

    def __init__(self, obs: Observation) -> None:
        self._obs = obs
        self.timestamp_us = obs.timestamp_us
        self._warped: dict[int, np.ndarray] = {}

    def slot_image(self, slot: int) -> np.ndarray:
        if slot not in self._warped:
            camera, hfov = SLOTS[slot]
            self._warped[slot] = warp(self._obs.image(camera), self._obs.pinhole(camera), hfov)
        return self._warped[slot]


def drop_expert_layers(expert_text: Any, drop: tuple[int, ...]) -> list[int]:
    """Delete expert layers ``drop`` from the forward; return the original indices kept.

    The layer list itself is shortened, so the deleted layers cost nothing. Each kept layer
    keeps its ``self_attn.layer_idx``, so it still reads its own language-model cache layer:
    expert layer i is trained against cache layer i only, and pointing it at another cache
    layer (measured) makes the plan worse than reading nothing.
    ``type(layers)(...)`` rebuilds the same container (torch's ModuleList in the model) without
    importing torch here.
    """
    n = len(expert_text.layers)
    bad = [i for i in drop if not 0 <= i < n]
    if bad:
        raise ValueError(f"expert layers {bad} are not in a {n}-layer expert")
    kept = [i for i in range(n) if i not in set(drop)]
    expert_text.layers = type(expert_text.layers)([expert_text.layers[i] for i in kept])
    expert_text.config.num_hidden_layers = len(kept)
    return kept


class Alpamayo2EdgePolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = 0.5

    def __init__(self) -> None:
        self.seed = _env_or("ALPAMAYO2_EDGE_SEED", SEED)
        # Speed ablations (model.yaml score_notes): flow-matching steps and forced reasoning.
        # `flow_steps` None means the module constant, read at call time.
        steps = os.environ.get("ALPAMAYO2_EDGE_FLOW_STEPS")
        self.flow_steps = int(steps) if steps else None
        if self.flow_steps is not None and self.flow_steps < 1:
            raise ValueError(f"ALPAMAYO2_EDGE_FLOW_STEPS must be a positive integer, got {steps!r}")
        self.reasoning = os.environ.get("ALPAMAYO2_EDGE_REASONING") or "model"
        if self.reasoning not in REASONING_MODES:
            raise ValueError(
                f"ALPAMAYO2_EDGE_REASONING must be one of {REASONING_MODES}, got {self.reasoning!r}"
            )
        # A size ablation: expert layers to delete at load, comma-separated ("" = none).
        drop = os.environ.get("ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS") or ""
        try:
            self.drop_expert_layers = tuple(sorted({int(x) for x in drop.split(",") if x.strip()}))
        except ValueError:
            raise ValueError(
                f"ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS must be comma-separated layer indices, got {drop!r}"
            ) from None
        # A speed ablation: flow steps from the previous plan instead of noise (0 = off).
        warm = os.environ.get("ALPAMAYO2_EDGE_WARM_START_STEPS") or "0"
        if not warm.isdigit():
            raise ValueError(f"ALPAMAYO2_EDGE_WARM_START_STEPS must be a whole number, got {warm!r}")
        self.warm_steps = int(warm)
        # Per-stage timers; tuning only (the plan is the same with them on).
        self.stage_timing = _env_or("ALPAMAYO2_EDGE_STAGE_TIMING", 0) == 1
        self._timer: StageTimer | None = None
        self._x_init = None
        self.model = None
        self.reset()

    def load(self) -> None:
        # torch and alpamayo2_edge exist only inside the model's own environment.
        import torch
        from alpamayo2_edge.checkpoint import load_model
        from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS

        expected = {slot: len(offsets) for slot, offsets in OFFSETS_S.items()}
        if dict(CAMERA_FRAME_COUNTS) != expected:
            raise RuntimeError(
                f"alpamayo2_edge expects camera frame counts {dict(CAMERA_FRAME_COUNTS)}; this "
                f"adapter builds {expected}. The checkpoint's input schedule changed: update "
                "SLOTS and OFFSETS_S from load_physical_aiavdataset.py before running."
            )
        checkpoint = os.environ.get("ALPAMAYO2_EDGE_CHECKPOINT") or os.path.join(
            os.environ.get("PADDOCK_WEIGHTS", ""), "Alpamayo2-Edge"
        )
        self.model, info = load_model(checkpoint, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
        history_steps = getattr(self.model.config, "history_num_steps", 20)
        if history_steps != NUM_HISTORY_STEPS:
            raise RuntimeError(
                f"checkpoint wants {history_steps} history poses; this adapter sends {NUM_HISTORY_STEPS}"
            )
        self._torch = torch
        if self.drop_expert_layers:
            drop_expert_layers(self.model.expert.expert, self.drop_expert_layers)
        if self.reasoning != "model":
            self._force_reasoning(FORCED_REASONING_IDS[self.reasoning])
        if self.warm_steps:
            self._install_warm_start()
        if self.stage_timing:
            self._timer = StageTimer(torch)
            self._timer.attach(self.model.vlm.model.visual, "vision")
            self._timer.attach_prefill_vs_decode(self.model.vlm.model)
            # `model.expert` is a container whose forward is never called; each flow step
            # calls its transformer, `model.expert.expert`.
            self._timer.attach(self.model.expert.expert, "expert")
        _log.info(
            "loaded Alpamayo 2 Edge from %s (flow steps %s, reasoning %s, expert layers deleted %s, "
            "warm start %s, stage timing %s)",
            checkpoint,
            self.flow_steps or DIFFUSION_STEPS,
            self.reasoning,
            list(self.drop_expert_layers) or "none",
            f"{self.warm_steps} step(s) from the previous plan" if self.warm_steps else "off",
            "on" if self.stage_timing else "off",
        )

    def _install_warm_start(self) -> None:
        """Let a plan start the flow from ``self._x_init`` rather than from noise.

        Upstream calls ``model.expert.diffusion.sample(...)`` with keywords only, and its
        ``_euler`` draws the start from ``torch.randn``. When a start has been set for this plan,
        the wrapper runs the same Euler loop (same 0 -> 1 time grid, same ``t`` shape) from it;
        otherwise it calls upstream unchanged. Guidance is off in this checkpoint; refused if not,
        since the loop below has no guided branch.
        """
        torch = self._torch
        diffusion = self.model.expert.diffusion
        if diffusion.use_classifier_free_guidance or diffusion.int_method != "euler":
            raise RuntimeError("warm start reproduces only the unguided Euler sampler this checkpoint uses")
        original = diffusion.sample
        n_dim = len(diffusion.x_dims)

        def sample(*, batch_size, step_fn, device, inference_step=None, return_all_steps=False, **kwargs):
            x, self._x_init = self._x_init, None
            if x is None or return_all_steps:
                return original(
                    batch_size=batch_size,
                    step_fn=step_fn,
                    device=device,
                    inference_step=inference_step,
                    return_all_steps=return_all_steps,
                    **kwargs,
                )
            steps = inference_step or diffusion.num_inference_steps
            x = x.to(device=device, dtype=torch.float32).expand(batch_size, *diffusion.x_dims).clone()
            grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
            for i in range(steps):
                t = grid[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
                x = x + (grid[i + 1] - grid[i]) * step_fn(x=x, t=t)
            return x

        diffusion.sample = sample

    def _carried(self, now_ego2world: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The previous plan carried to now (`carry_plan`), or None if there is none to carry."""
        prev = self._prev_plan
        if prev is None or not self.timestamps:
            return None
        elapsed = (self.timestamps[-1] - prev["timestamp_us"]) / 1e6
        if not 0.0 < elapsed <= WARM_START_MAX_GAP_S:
            return None
        return carry_plan(prev["xyz"], prev["rot"], prev["ego2world"], now_ego2world, elapsed)

    def _warm_start_action(self, inputs: dict[str, Any], carried: tuple[np.ndarray, np.ndarray]) -> Any:
        """The carried plan in the expert's own space: upstream's ``traj_to_action`` (normalised
        acceleration and curvature, 64 x 2) against this plan's ego history, the encoding its
        ``action_to_traj`` inverts.

        On the CPU: the encoding is a handful of small regularised least-squares solves, which
        on the GPU are a launch per tiny kernel -- 82 ms there on a busy H200, more than the
        flow steps the warm start saves. The action space moves its own buffers to its input's
        device, so the numbers are the same either way.
        """
        torch = self._torch
        xyz, rot = carried
        hist_xyz = inputs["ego_history_xyz"][:, -1].float()
        action = self.model.expert.action_space.traj_to_action(
            hist_xyz.cpu(),
            inputs["ego_history_rot"][:, -1].float().cpu(),
            torch.as_tensor(xyz, dtype=torch.float32)[None],
            torch.as_tensor(rot, dtype=torch.float32)[None],
        )
        return action.to(hist_xyz.device)

    @staticmethod
    def _force_reasoning(ids: list[int]) -> None:
        """Allow only ``ids[k]`` at generation step k, then generate as usual.

        Upstream builds its logits processors with the module-level name `LogitsProcessorList`
        on every call, so wrapping that name adds the filter, fresh each call, without
        touching upstream code. Generation then runs on as usual, including the one extra step
        its stop rule takes after <|traj_future_start|>.
        """
        import torch
        from alpamayo2_edge.models import alpamayo2_edge as upstream
        from transformers.generation.logits_process import LogitsProcessor

        class ForceTokens(LogitsProcessor):
            def __init__(self) -> None:
                self.start: int | None = None

            def __call__(self, input_ids, scores):
                if self.start is None:
                    self.start = input_ids.shape[1]
                step = input_ids.shape[1] - self.start
                if step < len(ids):
                    forced = torch.full_like(scores, -torch.inf)
                    forced[:, ids[step]] = 0.0
                    return forced
                return scores

        original = upstream.LogitsProcessorList
        upstream.LogitsProcessorList = lambda procs=(): original([*procs, ForceTokens()])

    def reset(self) -> None:
        """Drop the pose and image history between runs."""
        self.poses: list[np.ndarray] = []
        self.timestamps: list[int] = []
        self.frames: list[Frame] = []
        # Which slots the source can fill, decided by the first frame and held for the
        # sequence: a source that drops a camera halfway fails on the missing image
        # rather than quietly turning a real view black.
        self._present: set[str] | None = None
        self._yaws: dict[str, float] = {}
        self._last: dict[str, Any] = {}
        # The last plan made and where from, for the warm start and the `carry` comparison.
        # Cleared here, so a sequence never starts from another sequence's plan.
        self._prev_plan: dict[str, Any] | None = None

    # -- history -----------------------------------------------------------

    def _record(self, obs: Observation) -> None:
        """Append one observation. A frame stamped no later than the last is the same
        frame again (KITScenes clamps its warm-up at the scene start) and is dropped:
        history is resampled by time, and Slerp needs strictly increasing times."""
        if self.timestamps and obs.timestamp_us <= self.timestamps[-1]:
            return
        if self._present is None:
            have = set(obs.images) | set(obs.image_paths)
            self._present = {camera for camera in NEEDED_CAMERAS if camera in have}
            if "CAM_FRONT" not in self._present:
                raise ValueError(f"Alpamayo 2 Edge needs at least CAM_FRONT; the source sent {sorted(have)}")
            self._yaws = {
                camera: round(camera_yaw_deg(obs.camera2ego[camera]), 1)
                for camera in sorted(self._present)
                if camera in obs.camera2ego
            }
        self.poses.append(np.asarray(obs.ego2world, dtype=np.float64))
        self.timestamps.append(int(obs.timestamp_us))
        self.frames.append(Frame(obs))
        # Keep KEEP_S of history plus the frame just before it, so the oldest slot
        # still has a frame on each side of its time to choose between.
        cutoff = self.timestamps[-1] - int(KEEP_S * 1e6)
        first = max(0, int(np.searchsorted(self.timestamps, cutoff)) - 1)
        if first:
            self.poses, self.timestamps, self.frames = (
                self.poses[first:],
                self.timestamps[first:],
                self.frames[first:],
            )

    def observe(self, obs: Observation) -> None:
        """Record a history frame without sampling: the plan is a function of the
        recorded history alone, so one rollout per sequence draws the same plan."""
        self._record(obs)

    def _sample_poses(self, offsets_s: np.ndarray) -> np.ndarray:
        """Ego poses at the given offsets before now: translation linear, rotation slerp.

        Clamped to the recorded span, so a history shorter than 1.9 s repeats its oldest
        pose rather than extrapolating one.
        """
        from scipy.spatial import transform as spt

        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        stack = np.stack(self.poses)
        wanted = np.clip(times[-1] - offsets_s, times[0], times[-1])
        xyz = np.stack([np.interp(wanted, times, stack[:, i, 3]) for i in range(3)], axis=1)
        if len(times) == 1:
            matrices = np.repeat(stack[:1, :3, :3], len(wanted), axis=0)
        else:
            matrices = spt.Slerp(times, spt.Rotation.from_matrix(stack[:, :3, :3]))(wanted).as_matrix()
        poses = np.repeat(np.eye(4)[None], len(wanted), axis=0)
        poses[:, :3, :3] = matrices
        poses[:, :3, 3] = xyz
        return poses

    def ego_history(self) -> tuple[np.ndarray, np.ndarray]:
        """(20, 3) positions and (20, 3, 3) rotations at -1.9 .. 0 s, in the t0 ego frame,
        oldest first -- the transform load_physical_aiavdataset applies to egomotion."""
        from scipy.spatial import transform as spt

        stack = self._sample_poses(np.arange(NUM_HISTORY_STEPS - 1, -1, -1) * HISTORY_DT)
        t0_inv = spt.Rotation.from_matrix(stack[-1, :3, :3]).inv()
        xyz = t0_inv.apply(stack[:, :3, 3] - stack[-1, :3, 3])
        rot = (t0_inv * spt.Rotation.from_matrix(stack[:, :3, :3])).as_matrix()
        return xyz, rot

    def _warp_ahead(self, times: np.ndarray, present: set[str]) -> None:
        """Decode and warp every frame a slot will select, on a thread pool.

        One KITScenes sample is 17 full-resolution decodes and 22 crops, about 0.1 s and
        0.06 s each, and was 5 s of a 5.7 s sample done in turn; PIL releases the GIL for
        both. One task per (frame, camera), so CAM_FRONT is decoded once for its two slots.
        The pixels are the ones `camera_inputs` would make serially; only the order differs.
        """
        tasks: dict[tuple[int, str], list[int]] = {}
        for slot, (camera, _) in SLOTS.items():
            if camera not in present:
                continue
            for offset in OFFSETS_S[slot]:
                slots = tasks.setdefault((nearest_index(times, times[-1] - offset), camera), [])
                if slot not in slots:
                    slots.append(slot)

        def run(item: tuple[tuple[int, str], list[int]]) -> None:
            (index, _), slots = item
            for slot in slots:
                self.frames[index].slot_image(slot)

        with ThreadPoolExecutor(max_workers=min(WARP_THREADS, len(tasks) or 1)) as pool:
            list(pool.map(run, tasks.items()))

    def camera_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """The 22 frames upstream's loader would decode: (22, 3, H, W) uint8 images, camera
        ids, the chosen frames' recorded timestamps in microseconds, and the black slots.

        Grouped by camera in slot order, oldest first; upstream sorts them again anyway.
        """
        times = np.array(self.timestamps, dtype=np.float64) / 1e6
        present = self._present or set()
        self._warp_ahead(times, present)
        images, cameras, stamps, black = [], [], [], []
        for slot, (camera, _) in SLOTS.items():
            if camera not in present:
                black.append(slot)
            for offset in OFFSETS_S[slot]:
                frame = self.frames[nearest_index(times, times[-1] - offset)]
                if camera in present:
                    image = frame.slot_image(slot)
                else:
                    image = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
                images.append(image)
                cameras.append(slot)
                stamps.append(frame.timestamp_us)
        stacked = np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2))
        return stacked, np.array(cameras, dtype=np.int64), np.array(stamps, dtype=np.int64), black

    def sample(self, obs: Observation) -> dict[str, Any]:
        """The raw sample dict upstream's `prepare_model_inputs` takes, as torch tensors."""
        import torch
        from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS

        images, cameras, stamps, black = self.camera_inputs()
        xyz, rot = self.ego_history()
        nav_text = nav_text_for(*route_maneuver(obs))
        self._last.update(
            nav_text=nav_text, black_slots=black, frame_times_s=self._frame_times(cameras, stamps)
        )
        return {
            "image_frames": torch.from_numpy(images).unsqueeze(1).contiguous(),
            "camera_indices": torch.from_numpy(cameras),
            "absolute_timestamps": torch.from_numpy(stamps),
            "ego_history_xyz": torch.from_numpy(xyz).float().unsqueeze(0),
            "ego_history_rot": torch.from_numpy(rot).float().unsqueeze(0),
            "nav_text": [nav_text],
            "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        }

    def _frame_times(self, cameras: np.ndarray, stamps: np.ndarray) -> dict[int, list[float]]:
        now = self.timestamps[-1]
        out: dict[int, list[float]] = {}
        for camera, stamp in zip(cameras.tolist(), stamps.tolist(), strict=True):
            out.setdefault(camera, []).append(round((stamp - now) / 1e6, 3))
        return out

    # -- inference -----------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        from alpamayo2_edge.helper import prepare_model_inputs, to_device

        torch = self._torch
        started = time.perf_counter()
        self._record(obs)
        inputs = to_device(
            prepare_model_inputs(self.sample(obs), self.model.config, self.model.tokenizer), "cuda"
        )
        prepared = time.perf_counter()

        # The previous plan, carried to now: the warm start's starting point, and what the
        # new plan is compared with (`carry`) whatever the setting.
        now_ego2world = self.poses[-1]
        carried = self._carried(now_ego2world)
        warm = bool(self.warm_steps) and carried is not None
        if warm:
            with torch.inference_mode():
                self._x_init = self._warm_start_action(inputs, carried)
            torch.cuda.synchronize()
        steps = self.warm_steps if warm else (self.flow_steps or DIFFUSION_STEPS)
        encoded = time.perf_counter()

        if self._timer is not None:
            self._timer.reset()
        # Seeded per call, as the upstream CLI seeds each sample: the plan is a function
        # of the input, so a re-run reproduces it and two models see the same draw rule.
        torch.manual_seed(self.seed)
        generator = torch.Generator(device="cuda").manual_seed(self.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, _, extra = self.model.sample_trajectories_from_data(
                inputs,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                num_traj_sets=1,
                num_traj_samples=1,
                max_generation_length=MAX_GENERATION_LENGTH,
                diffusion_kwargs={"inference_step": steps, "generator": generator},
                return_extra=True,
            )
        plan = pred_xyz.float().cpu().numpy()[0, 0, 0]  # (64, 3), x forward, y left
        generated = time.perf_counter()
        self._x_init = None
        if not np.isfinite(plan).all():
            raise RuntimeError("Alpamayo 2 Edge returned a non-finite trajectory")
        # How far the new plan's first 3 s lie from the old plan carried to now: plan-to-plan
        # consistency, the thing a warm start changes.
        carry_m = None
        if carried is not None:
            k = CARRY_COMPARE_POINTS
            carry_m = float(np.linalg.norm(plan[:k, :2] - carried[0][:k, :2], axis=1).mean())
        self._prev_plan = {
            "timestamp_us": self.timestamps[-1],
            "ego2world": np.array(now_ego2world, dtype=np.float64),
            "xyz": plan.astype(np.float64),
            "rot": pred_rot.float().cpu().numpy()[0, 0, 0].astype(np.float64),
        }

        cot = ""
        if isinstance(extra, dict) and "cot" in extra:
            flat = np.asarray(extra["cot"]).reshape(-1)
            cot = str(flat[0]) if flat.size else ""
        self._last["reasoning"] = cot[:400]
        self._n_infer = getattr(self, "_n_infer", 0) + 1
        # `generate` is the model call alone, as before; encoding a warm start is `warm_encode`.
        _log.info(
            "infer #%d: prep %.0f ms, generate %.0f ms, nav %r, black slots %s, reach %.1f m, cot %r",
            self._n_infer,
            (prepared - started) * 1e3,
            (generated - encoded) * 1e3,
            self._last.get("nav_text"),
            self._last.get("black_slots"),
            float(np.linalg.norm(plan[GRID_INDICES[-1], :2])),
            cot[:160],
        )
        stages: dict[str, Any] = {
            "n": self._n_infer,
            "t_us": self.timestamps[-1],
            "prep_ms": round((prepared - started) * 1e3, 2),
            "warm_encode_ms": round((encoded - prepared) * 1e3, 2),
            "generate_ms": round((generated - encoded) * 1e3, 2),
            "flow_steps": steps,
            "warm": warm,
            "carry_m": None if carry_m is None else round(carry_m, 4),
        }
        if self._timer is not None:
            t, calls = self._timer.totals, self._timer.calls
            vision, prefill = t.get("vision", 0.0), t.get("prefill", 0.0)
            decode, expert = t.get("decode", 0.0), t.get("expert", 0.0)
            stages.update(
                vision_ms=round(vision * 1e3, 2),
                # Vision runs inside the prefill call, so the text part is the difference.
                prefill_ms=round((prefill - vision) * 1e3, 2),
                decode_ms=round(decode * 1e3, 2),
                decode_steps=calls.get("decode", 0),
                expert_ms=round(expert * 1e3, 2),
                expert_calls=calls.get("expert", 0),
                # Sampling, the stop rule and cache handling between the calls, and decoding
                # the action into poses: the model call less the four buckets.
                other_ms=round(((generated - encoded) - prefill - decode - expert) * 1e3, 2),
            )
        _log.info("stages %s", json.dumps(stages))
        extras = {"reasoning": cot[:400]} if _env_or("ALPAMAYO2_EDGE_SAVE_REASONING", 0) == 1 else None
        return Trajectory(plan[GRID_INDICES, :2], dt=self.waypoint_dt, extras=extras)

    def diagnostics(self) -> dict[str, Any]:
        """Which camera fills each slot, which slots are black, the frame times the model
        was told, and the bearing of each source camera -- the checks for a wrong camera."""
        if not self.timestamps:
            return {"status": "no frame seen yet"}
        present = self._present or set()
        return {
            "slots": {
                slot: (f"{camera} @ {hfov:g} deg" if camera in present else "black")
                for slot, (camera, hfov) in SLOTS.items()
            },
            "source_camera_yaw_deg": self._yaws,
            "history_frames_kept": len(self.timestamps),
            "request_dt_s": (
                round((self.timestamps[-1] - self.timestamps[-2]) / 1e6, 3)
                if len(self.timestamps) > 1
                else None
            ),
            **self._last,
        }


POLICY = Alpamayo2EdgePolicy
