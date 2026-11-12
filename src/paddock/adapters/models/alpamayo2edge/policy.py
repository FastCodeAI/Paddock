"""NVIDIA Alpamayo 2 Edge as a paddock Policy.

Edge (NVlabs/alpamayo2-edge, weights nvidia/Alpamayo2-Edge) is a 2.65B
vision-language-action model. A Cosmos3-Edge VLM reads 22 camera frames, 20 past
poses and one route sentence, and writes a one-sentence rationale, or `<NOMINAL>`
when nothing calls for one. A flow-matching expert that reads the VLM's cache then
plans 64 poses, 0.1 s apart. inputs.py builds that input from paddock's
observations; this module runs the pinned checkpoint on it the way the vendor's own
command does (`inference.predict`, `inference.main`'s settings): seed 43, top-p
0.98, temperature 0.6, at most 256 generated tokens, 10 flow steps.

Settings, read from the environment; model.yaml declares the defaults:

    ALPAMAYO2_EDGE_FRAMES          nearest   which recorded frame fills each slot
                                             (nearest | distinct, inputs.select_frames)
    ALPAMAYO2_EDGE_HISTORY         hermite   how the poses between frames are rebuilt
                                             (hermite | linear, inputs.resample_poses)
    ALPAMAYO2_EDGE_SAMPLES         1         plans drawn; the medoid is returned
    ALPAMAYO2_EDGE_SEED            43        set before every plan
    ALPAMAYO2_EDGE_NAV_DISTANCE_M  unset     a distance for turns the source gives none
    ALPAMAYO2_EDGE_SAVE_REASONING  0         1 keeps the rationale in each plan's extras
    ALPAMAYO2_EDGE_ABSENT_CAMERAS  refuse    black feeds a slot whose camera the source lacks
                                             black frames (comma2k19), and names it
    ALPAMAYO2_EDGE_THREADS         4         torch's CPU threads in the node; speed only

The seed is set before every plan, so a plan depends on its input alone -- not on
the node, the order, a retry or a resumed run. It is still one draw of the model's
sampling: another seed gives another plan.

The plan goes back whole: 64 poses at 0.1 s with the model's own heading, which
NAVSIM resamples to its eight 0.5 s poses. Nothing here is fitted to a benchmark.

*AlpaSim.* AlpaSim's scenes are recordings of Edge's own camera rig, so there the
input is built from the cameras Edge was trained on rather than adapted to it. Each
part is switched on by the input that calls for it, never by which simulator is
asking, so a source that sends none of them gets exactly what it got before:

- a camera named `CAM_FRONT_TELE` fills ID 6 itself, instead of a crop of CAM_FRONT;
- a camera sent with no pinhole intrinsics (AlpaSim's f-theta lenses) is not cropped,
  only resized to the vendor's frame size if it is not already that size;
- a route sent with the time of the frame it is in (`route_xy`, `route_timestamp_us`)
  is moved into the newest pose's frame and classified as AlpaSim's own Alpamayo driver
  classifies it, instead of the sentence being read from the command, which AlpaSim
  sends as FOLLOW_REFERENCE whatever the route does.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from paddock.adapters.models.alpamayo2edge import inputs
from paddock.policy.protocol import CANBUS_VELOCITY, Observation, Policy, Trajectory

_log = logging.getLogger(__name__)

PREFIX = "ALPAMAYO2_EDGE_"
CHECKPOINT_ENV = PREFIX + "CHECKPOINT"
#: The vendor's sampling (`inference.main` at 02f55a7), apart from what Settings sets.
SAMPLING: dict[str, Any] = {
    "top_p": 0.98,
    "top_k": None,
    "temperature": 0.6,
    "num_traj_sets": 1,
    "max_generation_length": 256,
    "diffusion_kwargs": {"inference_step": 10},
}
#: Frames kept from one sequence. Edge looks back 2 s, 21 frames at 10 Hz.
MAX_FRAMES = 64
#: Resized frames kept across sequences, by file. NAVSIM scores a log's scenarios
#: in order, 0.5 s apart, so a scenario shares three of its four frames with the
#: one before it; at 6 MB a frame this is under 1 GB.
MAX_WARPED = 128
#: The frame an absent camera's slot gets with ABSENT_CAMERAS=black, at the vendor's size.
BLACK_FRAME = np.zeros((inputs.FRAME_SIZE[1], inputs.FRAME_SIZE[0], 3), dtype=np.uint8)
BLACK_FRAME.setflags(write=False)
#: Characters of the rationale kept in extras and diagnostics.
REASONING_CHARS = 400
NOMINAL = "<NOMINAL>"
#: The vendor refuses a plan whose rationale has not reached the trajectory token
#: within `max_generation_length` ("rejecting incomplete CoTs", models/expert_utils.py).
#: At seed 43 that happened in 2 of navtest's 12,146 scenarios, every time: the seed
#: is fixed, so NAVSIM's retry asked for the same draw again. Such a plan is drawn again
#: with the next seed, the redraw the NAVSIM proxy's retry already is for an unseeded
#: model, and the scenario fails only if every draw is refused.
INCOMPLETE_RATIONALE = "No <traj_future_start> token found"
MAX_DRAWS = 4


@dataclass(frozen=True)
class Settings:
    frames: str = "nearest"
    history: str = "hermite"
    samples: int = 1
    seed: int = 43
    nav_distance_m: float | None = None
    save_reasoning: bool = False
    absent_cameras: str = "refuse"
    #: torch's CPU threads (`torch.set_num_threads`). Torch takes every core by default,
    #: and the vendor's processor then spends 1.4 s packing a plan's 22 frames on a
    #: 48-core box shared with CARLA, where 4 threads take 0.18 s and 16 take 0.07 s.
    #: The packed pixels and the plan are bit-identical at 2, 4, 8, 16 and 48 threads.
    threads: int = 4

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """The settings in `env` (the process environment by default), checked. A bad
        value stops the node at start-up rather than changing what it measures."""
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(PREFIX + name, "").strip()
            return value or None

        frames = get("FRAMES") or cls.frames
        history = get("HISTORY") or cls.history
        if frames not in inputs.FRAME_MODES:
            raise ValueError(f"{PREFIX}FRAMES={frames!r} is not one of {inputs.FRAME_MODES}")
        if history not in inputs.HISTORY_MODES:
            raise ValueError(f"{PREFIX}HISTORY={history!r} is not one of {inputs.HISTORY_MODES}")
        absent = get("ABSENT_CAMERAS") or cls.absent_cameras
        if absent not in inputs.ABSENT_MODES:
            raise ValueError(f"{PREFIX}ABSENT_CAMERAS={absent!r} is not one of {inputs.ABSENT_MODES}")
        samples = int(get("SAMPLES") or cls.samples)
        if samples < 1:
            raise ValueError(f"{PREFIX}SAMPLES must be at least 1, got {samples}")
        nav = get("NAV_DISTANCE_M")
        nav_distance_m = float(nav) if nav is not None else None
        if nav_distance_m is not None and not nav_distance_m >= 0:
            raise ValueError(f"{PREFIX}NAV_DISTANCE_M must be a distance in metres, got {nav!r}")
        threads = int(get("THREADS") or cls.threads)
        if threads < 1:
            raise ValueError(f"{PREFIX}THREADS must be at least 1, got {threads}")
        return cls(
            frames=frames,
            history=history,
            samples=samples,
            seed=int(get("SEED") or cls.seed),
            nav_distance_m=nav_distance_m,
            save_reasoning=(get("SAVE_REASONING") or "0") == "1",
            absent_cameras=absent,
            threads=threads,
        )


@dataclass
class Sample:
    """One input in numpy, before the vendor's packing."""

    #: (22, 1080, 1920, 3) uint8, slot by slot, each slot's frames oldest first.
    frames: np.ndarray
    #: (22,) int64 Edge camera IDs and capture times in microseconds.
    camera_ids: np.ndarray
    timestamps_us: np.ndarray
    #: (20, 3) and (20, 3, 3), in the newest pose's frame.
    history_xyz: np.ndarray
    history_rot: np.ndarray
    nav_text: str
    #: Camera ID -> the chosen frames' times, seconds before the newest frame.
    frame_times_s: dict[int, list[float]] = field(default_factory=dict)
    #: Seconds of the history before the oldest frame, continued rather than read.
    extrapolated_s: float = 0.0
    #: Camera IDs fed black frames because the source sent no camera for them.
    black_slots: list[int] = field(default_factory=list)


def vendor_sample(sample: Sample) -> dict[str, Any]:
    """The raw dict the vendor's PhysicalAI-AV loader returns, filled from `sample`."""
    import torch
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS

    return {
        "image_frames": torch.from_numpy(sample.frames).permute(0, 3, 1, 2).unsqueeze(1).contiguous(),
        "camera_indices": torch.from_numpy(np.asarray(sample.camera_ids, dtype=np.int64)),
        "absolute_timestamps": torch.from_numpy(np.asarray(sample.timestamps_us, dtype=np.int64)),
        "ego_history_xyz": torch.from_numpy(sample.history_xyz).float().unsqueeze(0),
        "ego_history_rot": torch.from_numpy(sample.history_rot).float().unsqueeze(0),
        "nav_text": [sample.nav_text],
        "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
        "min_pixels": 163840,
        "max_pixels": 196608,
        "input_profile": "camera_subsampled",
        "history_num_steps": inputs.HISTORY_STEPS,
    }


def prepare_inputs(data: dict[str, Any], config: Any, processor: Any) -> dict[str, Any]:
    """The vendor's `helper.prepare_model_inputs`, with a processor read once.

    The vendor's version reads the processor from the checkpoint on every call,
    0.84 s each on this site's disks -- about 2.8 h over navtest. These are its
    steps in its order (helper.py at 02f55a7); tests/test_vendor_inputs.py checks
    that both return the same tensors.
    """
    from alpamayo2_edge import helper
    from alpamayo2_edge.chat_template import build_conversation

    canonical = helper._canonical_inputs(data, config)
    messages = build_conversation(
        canonical,
        num_tokens_per_history_traj=config.tokens_per_history_traj,
        include_camera_ids=config.include_camera_ids,
        system_prompt=getattr(config, "system_prompt", None),
    )
    tokenized = helper._tokenize_messages(messages, processor, include_closed_thinking=True)
    xyz, rot = canonical["ego_history_xyz"], canonical["ego_history_rot"]
    return {
        "tokenized_data": tokenized,
        "ego_history_xyz": xyz.unsqueeze(0) if xyz.ndim == 3 else xyz,
        "ego_history_rot": rot.unsqueeze(0) if rot.ndim == 4 else rot,
    }


def draw_until_complete(attempt: Callable[[int], Any], seed: int, draws: int = MAX_DRAWS) -> tuple[Any, int]:
    """`attempt(seed)`, then `attempt(seed + 1)` and so on while the vendor refuses the
    draw for an incomplete rationale: (the first complete result, the seed it used).

    Any other error, and the last refusal, are raised as they are.
    """
    for offset in range(draws):
        try:
            return attempt(seed + offset), seed + offset
        except ValueError as exc:
            if not str(exc).startswith(INCOMPLETE_RATIONALE) or offset == draws - 1:
                raise
            _log.warning("seed %d: %s; drawing again with seed %d", seed + offset, exc, seed + offset + 1)
    raise AssertionError("unreachable")


def _has(obs: Observation, camera: str) -> bool:
    return camera in obs.images or camera in obs.image_paths


class Alpamayo2EdgePolicy(Policy):
    # The cameras a node decodes from a request. TELE_CAMERA is decoded only when a source
    # sends one (AlpaSim); model.yaml's `cameras`, what a simulator is asked to render,
    # keeps the five every source has, and AlpaSim's profile renders the telephoto itself.
    cameras = (*inputs.CAMERAS, inputs.TELE_CAMERA)
    waypoint_dt = inputs.PLAN_DT_S

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = None
        self.processor = None
        self.checkpoint: str | None = None
        self._warped: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._warp_hits = 0
        self._warp_misses = 0
        self._geometry: dict[int, dict[str, float]] = {}
        self._plans = 0
        self._nominal = 0
        self._redrawn = 0
        self._last_seed: int | None = None
        self._ms = np.zeros(3)
        self._last: dict[str, Any] = {}
        #: The slots the last plan was built from: SLOTS, or with ID 6 from a real telephoto.
        self._slots = inputs.SLOTS
        self.reset()

    def load(self) -> None:
        if self.model is not None:
            return
        # torch and the vendor package exist only in the model's own environment.
        import torch
        from alpamayo2_edge.checkpoint import load_model
        from alpamayo2_edge.helper import get_processor

        # Set on the thread that plans: the node calls load() and then serves /infer on the
        # same thread (policy/server.py), and OpenMP's thread count is per thread.
        torch.set_num_threads(self.settings.threads)

        checkpoint = os.environ.get(CHECKPOINT_ENV)
        if not checkpoint:
            raise RuntimeError(
                f"{CHECKPOINT_ENV} is not set; model.yaml sets it to {{weights}}/Alpamayo2-Edge"
            )
        self.model, _ = load_model(checkpoint, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
        self.processor = get_processor(self.model.tokenizer, self.model.config)
        self.checkpoint = str(self.model.config._name_or_path)

    def reset(self) -> None:
        """Forget the sequence. Resized frames stay cached by file for the next one."""
        self._frames: list[Observation] = []
        #: This sequence's frames that came as pixels (a closed-loop client), resized, by the
        #: time they were taken: the next sequence's clock may start again at the same time.
        self._held: dict[tuple, np.ndarray] = {}

    def _record(self, obs: Observation) -> None:
        """Keep one observation. One stamped no later than the last is the same frame
        again and is dropped: the history is rebuilt by time, which must increase."""
        if self._frames and obs.timestamp_us <= self._frames[-1].timestamp_us:
            return
        self._frames.append(obs)
        del self._frames[:-MAX_FRAMES]

    def observe(self, obs: Observation) -> None:
        """Record a frame without planning. Edge plans from the recorded sequence
        alone, so planning on every frame of a sequence would draw N plans to use one."""
        self._record(obs)

    # -- input --------------------------------------------------------------------

    def _warp(self, frame: Observation, camera: str, hfov: float | None) -> np.ndarray:
        """One camera of one frame, cropped for its slot, once.

        A file is known by its path, across sequences. Pixels sent in the body (a
        closed-loop client) are known by the time they were taken, which `_record`
        keeps strictly increasing within a sequence: at 10 Hz a plan reads 22 frames,
        16 of them read by an earlier plan, and resizing one is 30-40 ms.

        A camera sent without intrinsics is an f-theta lens (AlpaSim's): it is resized
        whole (`inputs.fit`), never cropped, so it may fill only a slot that is not a crop.
        """
        fx = cx = cy = None
        if camera in frame.camera2image:
            intrinsics = np.asarray(frame.camera2image[camera], dtype=np.float64)
            fx, cx, cy = float(intrinsics[0, 0]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
        elif hfov is not None:
            raise ValueError(
                f"{camera} came without pinhole intrinsics, so it cannot be cropped to {hfov:g} deg "
                f"for its slot; send {inputs.TELE_CAMERA} for the telephoto, or a pinhole K"
            )
        path = frame.image_paths.get(camera)
        cache = self._warped if path else self._held
        key = (path, hfov, fx, cx, cy) if path else (frame.timestamp_us, camera, hfov, fx, cx, cy)
        if key in cache:
            self._warp_hits += 1
            if path:
                self._warped.move_to_end(key)
            return cache[key]
        self._warp_misses += 1
        image = frame.image(camera)
        cache[key] = inputs.fit(image) if fx is None else inputs.warp(image, fx, cx, cy, hfov)
        while len(self._warped) > MAX_WARPED:
            self._warped.popitem(last=False)
        return cache[key]

    def _slot_geometry(self, camera_id: int, frame: Observation, camera: str, hfov: float | None) -> None:
        """Where a slot's camera points and how wide its crop is, once per slot, for diagnostics."""
        if camera_id in self._geometry or camera not in frame.camera2ego:
            return
        axis = np.asarray(frame.camera2ego[camera], dtype=np.float64)[:3, 2]
        geometry: dict[str, Any] = {"bearing_deg": round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1)}
        if camera in frame.camera2image:
            k = np.asarray(frame.camera2image[camera], dtype=np.float64)
            height, width = frame.image(camera).shape[:2]
            geometry["hfov_deg"] = round(
                inputs.crop_hfov_deg(width, height, k[0, 0], k[0, 2], k[1, 2], hfov), 1
            )
        else:
            # An f-theta lens, sent whole: its field of view is the camera's own.
            geometry["hfov_deg"] = "as rendered"
        self._geometry[camera_id] = geometry

    def build_sample(self) -> Sample:
        """The recorded sequence as Edge's input. Raises if a camera Edge needs never arrived:
        the vendor requires all six IDs, and a black frame would be a claim the source never made.
        With `ABSENT_CAMERAS=black` that slot instead gets a black frame at each time it asks
        for, taken on the frames the source did send, and is named in `black_slots`."""
        if not self._frames:
            raise RuntimeError("no frame has been recorded")
        frames = self._frames
        now_us = frames[-1].timestamp_us
        images, ids, stamps, chosen_times, black = [], [], [], {}, []
        self._slots = inputs.slots(any(_has(frame, inputs.TELE_CAMERA) for frame in frames))
        for camera_id, camera, hfov in self._slots:
            carrying = [frame for frame in frames if _has(frame, camera)]
            if not carrying:
                if self.settings.absent_cameras != "black":
                    raise ValueError(
                        f"Edge's camera {camera_id} is filled from {camera}, which this source did not "
                        f"send (set {PREFIX}ABSENT_CAMERAS=black to feed it black frames)"
                    )
                black.append(camera_id)
                times = [frame.timestamp_us / 1e6 for frame in frames]
                picks = inputs.select_frames(
                    times, inputs.offsets_for(camera_id), self.settings.frames, now_us / 1e6
                )
                for i in picks:
                    images.append(BLACK_FRAME)
                    ids.append(camera_id)
                    stamps.append(frames[i].timestamp_us)
                chosen_times[camera_id] = [round((frames[i].timestamp_us - now_us) / 1e6, 3) for i in picks]
                continue
            times = [frame.timestamp_us / 1e6 for frame in carrying]
            picks = inputs.select_frames(
                times, inputs.offsets_for(camera_id), self.settings.frames, now_us / 1e6
            )
            chosen = [carrying[i] for i in picks]
            for frame in chosen:
                images.append(self._warp(frame, camera, hfov))
                ids.append(camera_id)
                stamps.append(frame.timestamp_us)
            chosen_times[camera_id] = [round((frame.timestamp_us - now_us) / 1e6, 3) for frame in chosen]
            self._slot_geometry(camera_id, chosen[-1], camera, hfov)
        # Kept for the 2 s the slots look back, 21 frames at 10 Hz. An older frame that a slot
        # reads again for want of a newer one, as on a source slower than 0.5 Hz, is resized again.
        oldest_us = now_us - int(round((max(map(abs, inputs.FRONT_OFFSETS_S)) + inputs.JITTER_S) * 1e6))
        self._held = {key: image for key, image in self._held.items() if key[0] >= oldest_us}
        times = [frame.timestamp_us / 1e6 for frame in frames]
        poses = np.stack([np.asarray(frame.ego2world, dtype=np.float64) for frame in frames])
        # Measured velocity, ego frame -> world frame (NAVSIM's EgoStatus, the CAN bus elsewhere).
        velocities = np.stack(
            [
                pose[:3, :3] @ frame.canbus[CANBUS_VELOCITY : CANBUS_VELOCITY + 3]
                for pose, frame in zip(poses, frames, strict=True)
            ]
        )
        xyz, rot = inputs.ego_history(times, poses, velocities, self.settings.history)
        last = frames[-1]
        text = self._route_text(last, times, poses, velocities)
        if text is None:
            text = inputs.nav_text(
                last.command, last.nav_maneuver, last.nav_distance_m, self.settings.nav_distance_m
            )
        # The resized frames are what is kept; the decoded originals can go.
        for frame in frames:
            for camera in [c for c in frame.images if c in frame.image_paths]:
                del frame.images[camera]
        return Sample(
            frames=np.stack(images),
            camera_ids=np.asarray(ids, dtype=np.int64),
            timestamps_us=np.asarray(stamps, dtype=np.int64),
            history_xyz=xyz,
            history_rot=rot,
            nav_text=text,
            frame_times_s=chosen_times,
            extrapolated_s=inputs.history_before_first_frame_s(times),
            black_slots=black,
        )

    def _route_text(
        self, last: Observation, times: list[float], poses: np.ndarray, velocities: np.ndarray
    ) -> str | None:
        """The route sentence from a route polyline, when the source sends one with its time.

        None when it does not, or when the route has no usable point; the command then
        decides, as on every other source. The route is in the rig frame at its own time,
        which can trail the newest pose: it is moved into the newest pose's frame through
        the pose at that time, rebuilt as the history is, as AlpaSim's driver does.
        """
        if last.route_xy is None or last.route_timestamp_us is None:
            return None
        route_s = int(last.route_timestamp_us) / 1e6
        if not times[0] - 1e-6 <= route_s <= times[-1] + 1e-6:
            # Resampling would hold or continue the history, which places the route in the
            # wrong frame and still gives a sentence.
            raise ValueError(
                f"route_timestamp_us {last.route_timestamp_us} is outside the kept history "
                f"[{int(round(times[0] * 1e6))}, {int(round(times[-1] * 1e6))}]"
            )
        route_xyz, route_rot = inputs.resample_poses(
            times, poses, [route_s], velocities, self.settings.history
        )
        route_pose = np.eye(4)
        route_pose[:3, :3], route_pose[:3, 3] = route_rot[0], route_xyz[0]
        xy = np.asarray(last.route_xy, dtype=np.float64).reshape(-1, 2)
        # The bridge sends the route in two dimensions; it sits at the rig's height.
        route = np.column_stack([xy, np.zeros(len(xy))])
        return inputs.route_to_nav_text(inputs.route_in_frame(route, poses[-1], route_pose))

    # -- the model ----------------------------------------------------------------

    def _run_model(self, sample: Sample) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """The vendor's packing and sampling: positions (K, 64, 3), rotations (K, 64, 3, 3), rationales."""
        from alpamayo2_edge.helper import to_device
        from alpamayo2_edge.inference import predict, validate_predictions

        prepared = to_device(prepare_inputs(vendor_sample(sample), self.model.config, self.processor), "cuda")
        settings = {**SAMPLING, "num_traj_samples": self.settings.samples}

        def attempt(seed: int):
            return predict(self.model, prepared, {**settings, "seed": seed}, "cuda")

        (xyz, rot, extra), self._last_seed = draw_until_complete(attempt, self.settings.seed)
        validate_predictions(xyz, rot, settings["num_traj_sets"], settings["num_traj_samples"])
        rationales = [str(text) for text in np.asarray(extra.get("cot", [])).reshape(-1)]
        return xyz.float().cpu().numpy()[0, 0], rot.float().cpu().numpy()[0, 0], rationales

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        if obs.plan_dt is not None and abs(obs.plan_dt - inputs.PLAN_DT_S) > 1e-9:
            raise ValueError(f"plan_dt {obs.plan_dt} s asked for; Edge plans every {inputs.PLAN_DT_S} s")
        started = time.perf_counter()
        self._record(obs)
        sample = self.build_sample()
        built = time.perf_counter()
        self._last_seed = self.settings.seed
        xyz, rot, rationales = self._run_model(sample)
        ran = time.perf_counter()
        chosen = inputs.medoid_index(xyz[:, :, :2])
        plan = inputs.plan_from_poses(xyz[chosen], rot[chosen])
        reasoning = rationales[chosen] if chosen < len(rationales) else ""
        self._account(sample, plan, reasoning, (started, built, ran, time.perf_counter()))
        extras = {"reasoning": reasoning[:REASONING_CHARS]} if self.settings.save_reasoning else None
        return Trajectory(plan, dt=inputs.PLAN_DT_S, extras=extras)

    def _account(self, sample: Sample, plan: np.ndarray, reasoning: str, marks: tuple[float, ...]) -> None:
        """Counters for diagnostics and one log line per plan: where the time went, the route
        sentence, whether the model reasoned, and how far the plan reaches."""
        ms = np.diff(marks) * 1000.0
        self._plans += 1
        nominal = reasoning.strip() == NOMINAL
        self._nominal += int(nominal)
        redrawn = self._last_seed != self.settings.seed
        self._redrawn += int(redrawn)
        self._ms += ms
        reach = float(np.linalg.norm(plan[-1, :2]))
        self._last = {
            "nav_text": sample.nav_text,
            "black_slots": sample.black_slots,
            "frame_times_s": sample.frame_times_s,
            "history_extrapolated_s": sample.extrapolated_s,
            "reasoning": reasoning[:REASONING_CHARS],
            "plan_reach_m": round(reach, 2),
            "seed": self._last_seed,
        }
        _log.info(
            "plan #%d: inputs %.0f ms, model %.0f ms, rest %.0f ms; nav %r; %s; reaches %.1f m in 6.4 s%s",
            self._plans,
            *ms,
            sample.nav_text,
            NOMINAL if nominal else f"rationale {reasoning[:160]!r}",
            reach,
            f"; drawn again, seed {self._last_seed}" if redrawn else "",
        )

    def diagnostics(self) -> dict[str, Any]:
        """The settings, the slots and what fills them, and the last plan's inputs. Answers before
        `load()` and before any frame. `slot_geometry` is where each slot's camera points and how
        wide its crop is: Edge's front and cross slots were trained on 120 deg cameras."""
        return {
            "settings": asdict(self.settings),
            "checkpoint": self.checkpoint,
            "slots": [
                f"{camera_id}: {camera}" + (f", {hfov:g} deg crop" if hfov else "")
                for camera_id, camera, hfov in self._slots
            ],
            "slot_geometry": dict(self._geometry),
            "plans": self._plans,
            "nominal_rationales": self._nominal,
            # Plans whose first draw the vendor refused for an incomplete rationale.
            "drawn_again": self._redrawn,
            "mean_ms": (
                dict(
                    zip(
                        ("inputs", "model", "rest"), np.round(self._ms / self._plans, 1).tolist(), strict=True
                    )
                )
                if self._plans
                else None
            ),
            "warp_cache": {
                "entries": len(self._warped),
                # This sequence's frames that came as pixels, kept while a slot may still read them.
                "held": len(self._held),
                "hits": self._warp_hits,
                "misses": self._warp_misses,
            },
            "last": self._last or None,
        }


POLICY = Alpamayo2EdgePolicy
