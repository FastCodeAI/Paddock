"""TIER IV METEOR (v157, released ONNX) as a paddock Policy.

METEOR is an 8-camera multi-task BEV network whose seventh head is an
end-to-end planner: three trajectory hypotheses of six waypoints at 0.5 s in
the current ego frame, x forward and y left, plus mode logits and
steer/accel/brake. That is paddock's own trajectory convention, so nothing is
resampled; the adapter's work is on the input side.

Input side, and why each rule is what it is:

- The graph is static: uint8 RGB ``[1, 8, 3, 432, 768]`` in METEOR's fixed slot
  order, with per-slot intrinsics at 768x432 and ego->camera transforms. Camera
  frames are OpenCV optical (x right, y down, z forward), the convention every
  benchmark here already supplies, so ``T_cam_ego = inv(camera2ego)``.
- paddock's six nuScenes cameras fill six slots. The two telephoto NARROW slots
  have no counterpart anywhere in paddock and are fed as *absent*: a black
  image (the zero tensor METEOR's own runtime feeds for a missing camera)
  with a donor camera's calibration, the convention the network was trained
  with for 7-camera rigs (``bevlane/dataset.py`` ``cam_drop`` /
  ``_CAM_FALLBACK``), so one missing camera is in-distribution. A benchmark supplying fewer cameras
  leaves more slots absent the same way -- unless the front camera is alone.
- A front camera alone (comma2k19) is filled by its own rule,
  ``METEOR_FRONT_ONLY``. With the rule above, all seven black slots would sit on
  the front camera's pose, and the lift (``bevlane/model.py`` ``project_bev``)
  averages per BEV cell every slot that sees it, weighted by depth probability
  + 0.05 -- so every cell the real camera sees would be about 7/8 black-image
  feature. The default instead poses each black slot where METEOR's own rig has
  it (what training's per-camera dropout looks like: a zero image with that
  slot's own K and T) and gives CAM_FRONT_NARROW the centred 30 deg window of
  the same frame (upstream's KITScenes converter does the same).
- Frames of any size are cropped to 16:9 around the principal point and
  area-downscaled to 768x432; the intrinsics follow the pixels. KITScenes is
  3504x2272, nuScenes 1600x900, CARLA 800x450, comma2k19 1164x874.
- No driving command: the released ONNX has no intent input, so the model runs
  in its trained "no navigation" mode (30 % intent dropout in training).
  Mode choice is the plain argmax METEOR's own evaluator uses.
- History is baked out of the export (single-frame), so ``observe`` is a no-op
  and ``self_prime`` is off: replaying KITScenes' 48 history frames would be
  48 wasted forward passes with no effect on the plan.

The model plans 3 s. ``METEOR_HORIZON_S`` (unset by default) continues the last
0.5 s segment to a longer horizon for benchmarks that score further out
(NAVSIM, 4 s). It is an extrapolation the model never made; it is opt-in, is
recorded in ``extras`` and, set through ``[models.meteor.env]``, in run.json.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from paddock.policy.protocol import Observation, Policy, Trajectory

logger = logging.getLogger(__name__)

#: METEOR's slot order, fixed by the graph (hf/onnx_smoke_test.py CAMS).
SLOTS = (
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_NARROW",
    "CAM_BACK_NARROW",
)

#: paddock camera -> METEOR slot. The nuScenes rig has one front and one back
#: camera; both go to the WIDE slots, which is the closer field of view.
SLOT_FROM_CAMERA = {
    "CAM_FRONT": "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT": "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT",
    "CAM_BACK": "CAM_BACK_WIDE",
    "CAM_BACK_LEFT": "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT": "CAM_BACK_RIGHT",
}

#: Calibration donor for an absent slot, as upstream's dataset does it
#: (bevlane/dataset.py _CAM_FALLBACK); after that, the first present slot.
DONOR = {
    "CAM_BACK_NARROW": "CAM_BACK_WIDE",
    "CAM_FRONT_NARROW": "CAM_FRONT_WIDE",
    "CAM_BACK_WIDE": "CAM_BACK_LEFT",
    "CAM_FRONT_WIDE": "CAM_FRONT_NARROW",
}

INPUT_H, INPUT_W = 432, 768
EGO_K = 3  # trajectory hypotheses
EGO_HORIZON = 6  # waypoints per hypothesis
WAYPOINT_DT = 0.5
NATIVE_HORIZON_S = EGO_HORIZON * WAYPOINT_DT
#: ego output layout: K*12 waypoint coords, K mode logits, steer, accel, brake logit.
EGO_LEN = EGO_K * EGO_HORIZON * 2 + EGO_K + 3

#: What an absent camera is fed: a black image, which the graph's own /255
#: turns into the zero tensor METEOR's runtime feeds for a missing camera
#: (deploy/realtime_trt.py, model.zero_cams). The release notes say ImageNet
#: mean/std normalisation is in-graph, which would make the zero pixel
#: (124, 116, 104) instead; it is not -- the graph is Cast, Div 255, Conv, and
#: that Conv is the checkpoint's conv1 with only BatchNorm folded in. See
#: docs/models/meteor.md for the check.
ABSENT_FILL_RGB = (0, 0, 0)

#: How the slots are filled when CAM_FRONT is the only camera supplied:
#: rule -> (CAM_FRONT_NARROW gets the 30 deg window, where the black slots are
#: posed). "donor" is the rule every other feed uses; "refpose" poses each black
#: slot as METEOR's own rig has it (REF_RIG_K / REF_RIG_T_CAM_EGO).
FRONT_ONLY_RULES = {
    "donor": (False, "donor"),
    "narrow": (True, "donor"),
    "refpose": (False, "refpose"),
    "narrow_refpose": (True, "refpose"),
}
#: Chosen on METEOR's own demo scene (valday, 132 frames with a 3 s future) with
#: only its front camera fed, not on comma2k19: ADE 1.01 m against 1.21 for
#: refpose, 1.94 for narrow and 2.22 for donor; 0.65 with all eight cameras and
#: 1.37 for constant velocity. Measured outside paddock; docs/models/meteor.md.
DEFAULT_FRONT_ONLY = "narrow_refpose"
#: upstream tools/kitscenes/kitscenes_to_bevlane.py NARROW_HFOV_DEG.
NARROW_HFOV_DEG = 30.0

# METEOR's own rig at 768x432, in SLOTS order: (fx, fy, cx, cy), and the top
# three rows of T_cam_ego. Copied from upstream bevlane/ref_rig_drs.npz, which
# arrived in 27a5bd0, after the repos.yaml pin (sha256 0b64185cac862ae7...);
# every value is exact as float32, and test_reference_rig_matches_the_clone
# compares the two wherever a clone that has the file exists.
_REF_RIG_INTRINSICS = (
    (335.4, 292.1226, 372.976, 219.5849),
    (335.4, 292.1226, 372.976, 219.5849),
    (335.4, 292.1226, 372.976, 219.5849),
    (335.4, 292.1226, 372.976, 219.5849),
    (335.4, 292.1226, 372.976, 219.5849),
    (335.4, 292.1226, 372.976, 219.5849),
    (1414.7067, 1240.8643, 376.704, 222.66605),
    (1412.2773, 1233.6991, 403.10132, 250.29755),
)
_REF_RIG_POSE = (
    (
        (0.004410922, -0.9999781, 0.0048197866, 0.23507825),
        (0.020875853, -0.004726181, -0.9997704, 1.8506082),
        (0.99977183, 0.004509997, 0.02085509, -1.1066022),
    ),
    (
        (0.87826204, -0.47794378, 0.0150217675, -0.45931658),
        (-0.19166245, -0.38063002, -0.9046481, 2.1622138),
        (0.43808836, 0.79163915, -0.42589685, -0.106102616),
    ),
    (
        (-0.88886726, -0.458152, -0.0032623385, 0.630096),
        (-0.18359882, 0.36270875, -0.91363746, 2.0406098),
        (0.41976824, -0.811504, -0.40651578, 0.2910722),
    ),
    (
        (-0.008049742, 0.99995536, 0.0048712073, -0.36646634),
        (0.02349667, 0.00505959, -0.99971074, 1.9111043),
        (-0.9996911, -0.007933373, -0.023535943, 0.2945107),
    ),
    (
        (0.8759256, 0.48205388, 0.019456666, -0.9101501),
        (0.23416989, -0.38955182, -0.8907386, 1.8373057),
        (-0.42180437, 0.7847769, -0.45410094, 0.61060315),
    ),
    (
        (-0.8737767, 0.4854349, -0.029500682, 0.6453274),
        (0.23460798, 0.36760145, -0.89990515, 1.7286803),
        (-0.4260005, -0.7932359, -0.43508935, 0.9264041),
    ),
    (
        (0.009009867, -0.99995214, 0.003942861, 0.27554005),
        (0.020045724, -0.0037621267, -0.99979246, 1.8388923),
        (0.999759, 0.009087552, 0.02001034, -1.245205),
    ),
    (
        (0.0072997278, 0.9999596, 0.005418633, -0.30384293),
        (0.043151125, 0.005097742, -0.9990565, 1.8799614),
        (-0.9990428, 0.007527604, -0.043113068, 0.085124314),
    ),
)
REF_RIG_K = np.array(
    [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]] for fx, fy, cx, cy in _REF_RIG_INTRINSICS],
    dtype=np.float32,
)
REF_RIG_T_CAM_EGO = np.array([[*rows, (0.0, 0.0, 0.0, 1.0)] for rows in _REF_RIG_POSE], dtype=np.float32)

DEFAULT_THREADS = 4
DEFAULT_GPU_MEM_MB = 6144


def ort_threads() -> int:
    """Intra-op threads per session from METEOR_ORT_THREADS; a bad value means the default."""
    raw = os.environ.get("METEOR_ORT_THREADS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_THREADS


def horizon_s() -> float:
    """Planning horizon from METEOR_HORIZON_S; unset means the model's own 3 s."""
    raw = os.environ.get("METEOR_HORIZON_S", "").strip()
    if not raw:
        return NATIVE_HORIZON_S
    value = float(raw)
    if value < NATIVE_HORIZON_S:
        raise ValueError(f"METEOR_HORIZON_S={raw} is shorter than the model's {NATIVE_HORIZON_S} s plan")
    return value


def front_only_rule() -> str:
    """How a front camera alone fills the slots, from METEOR_FRONT_ONLY; unset means the default."""
    raw = os.environ.get("METEOR_FRONT_ONLY", "").strip() or DEFAULT_FRONT_ONLY
    if raw not in FRONT_ONLY_RULES:
        raise ValueError(f"METEOR_FRONT_ONLY={raw} is not one of {tuple(FRONT_ONLY_RULES)}")
    return raw


def has_image(obs: Observation, camera: str) -> bool:
    return camera in obs.images or camera in obs.image_paths


def crop_to_input(
    image: np.ndarray, k: np.ndarray, x0: int, y0: int, cw: int, ch: int
) -> tuple[np.ndarray, np.ndarray]:
    """The (x0, y0, cw, ch) window of a uint8 HWC RGB frame -> 768x432, and its intrinsics.

    An area filter when shrinking -- PIL's BOX, the same family as the
    INTER_AREA resize in hf/onnx_smoke_test.py. When enlarging, BOX degenerates
    to nearest neighbour, so a window smaller than 768x432 (comma2k19's 30 deg
    narrow window is 488x274) is interpolated bilinearly instead.
    """
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    sx, sy = INPUT_W / cw, INPUT_H / ch
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    if crop.shape[:2] == (INPUT_H, INPUT_W):
        out = np.ascontiguousarray(crop, dtype=np.uint8)
    else:
        shrink = cw >= INPUT_W and ch >= INPUT_H
        resample = Image.BOX if shrink else Image.BILINEAR
        out = np.asarray(Image.fromarray(crop).resize((INPUT_W, INPUT_H), resample), dtype=np.uint8)
    k2 = np.array(
        [[fx * sx, 0.0, (cx - x0) * sx], [0.0, fy * sy, (cy - y0) * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return out, k2


def fit_frame(image: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """uint8 HWC RGB of any size -> 768x432, with the intrinsics of that frame.

    Crop to 16:9 around the principal point (so cx, cy stay near the centre
    the network trained on), then resize (``crop_to_input``).
    """
    h, w = image.shape[:2]
    target = INPUT_W / INPUT_H
    if w / h > target:
        cw, ch = int(round(h * target)), h
    else:
        cw, ch = w, int(round(w / target))
    x0 = int(np.clip(round(float(k[0, 2]) - cw / 2), 0, w - cw))
    y0 = int(np.clip(round(float(k[1, 2]) - ch / 2), 0, h - ch))
    return crop_to_input(image, k, x0, y0, cw, ch)


def narrow_crop(
    image: np.ndarray, k: np.ndarray, hfov_deg: float = NARROW_HFOV_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """The centred ``hfov_deg`` window of a frame, as CAM_FRONT_NARROW sees it.

    Upstream's KITScenes converter feeds its narrow slot this way. The window
    must lie inside the frame: a camera narrower than 30 deg has no such window,
    and padding one would show the network black bars it never saw.
    """
    h, w = image.shape[:2]
    cw = int(round(2 * float(k[0, 0]) * np.tan(np.radians(hfov_deg / 2))))
    ch = int(round(cw * INPUT_H / INPUT_W))
    x0, y0 = int(round(float(k[0, 2]) - cw / 2)), int(round(float(k[1, 2]) - ch / 2))
    if x0 < 0 or y0 < 0 or x0 + cw > w or y0 + ch > h:
        raise ValueError(f"a {hfov_deg} deg window ({cw}x{ch} at {x0},{y0}) does not fit a {w}x{h} frame")
    return crop_to_input(image, k, x0, y0, cw, ch)


def donor_for(slot: str, present: set[str]) -> str:
    for candidate in (DONOR.get(slot), *SLOTS):
        if candidate in present:
            return candidate
    raise ValueError("no camera present to donate calibration from")


def build_inputs(
    obs: Observation, front_only: str = DEFAULT_FRONT_ONLY
) -> tuple[dict[str, np.ndarray], tuple[str, ...], str]:
    """The graph's feed for one observation, the slots carrying a real image, and the rule for the rest.

    ``front_only`` applies only when CAM_FRONT is the one camera supplied; any
    other set of cameras is filled by the donor rule, as before it existed.
    """
    frames: dict[str, np.ndarray] = {}
    calib: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    front: tuple[np.ndarray, np.ndarray] | None = None
    for camera, slot in SLOT_FROM_CAMERA.items():
        if not has_image(obs, camera):
            continue
        image = obs.image(camera)
        k = np.asarray(obs.camera2image[camera], dtype=np.float64)
        frame, k2 = fit_frame(image, k)
        camera2ego = np.asarray(obs.camera2ego[camera], dtype=np.float64)
        frames[slot] = frame
        calib[slot] = (k2, np.linalg.inv(camera2ego).astype(np.float32))
        if camera == "CAM_FRONT":
            front = (image, k)
    if not frames:
        raise ValueError(f"observation carries none of {tuple(SLOT_FROM_CAMERA)}")

    rule = front_only if set(frames) == {"CAM_FRONT_WIDE"} else "donor"
    narrow, absent = FRONT_ONLY_RULES[rule]
    if narrow and front is not None:
        frames["CAM_FRONT_NARROW"], k_narrow = narrow_crop(*front)
        # The same physical camera as CAM_FRONT_WIDE, so the same pose.
        calib["CAM_FRONT_NARROW"] = (k_narrow, calib["CAM_FRONT_WIDE"][1])

    present = set(frames)
    imgs = np.empty((len(SLOTS), 3, INPUT_H, INPUT_W), dtype=np.uint8)
    intrinsics = np.empty((len(SLOTS), 3, 3), dtype=np.float32)
    t_cam_ego = np.empty((len(SLOTS), 4, 4), dtype=np.float32)
    for i, slot in enumerate(SLOTS):
        if slot in present:
            imgs[i] = frames[slot].transpose(2, 0, 1)
            intrinsics[i], t_cam_ego[i] = calib[slot]
        else:
            imgs[i] = np.asarray(ABSENT_FILL_RGB, dtype=np.uint8)[:, None, None]
            if absent == "refpose":
                intrinsics[i], t_cam_ego[i] = REF_RIG_K[i], REF_RIG_T_CAM_EGO[i]
            else:
                intrinsics[i], t_cam_ego[i] = calib[donor_for(slot, present)]
    feed = {
        "imgs": imgs[None],
        "K": intrinsics[None],
        "T_cam_ego": t_cam_ego[None],
        "v0": np.array([obs.speed], dtype=np.float32),
    }
    return feed, tuple(s for s in SLOTS if s in present), rule


@dataclass(frozen=True)
class EgoDecode:
    waypoints: np.ndarray  # (6, 2), ego frame, x forward, y left
    mode: int
    logits: list[float]
    steer_rad: float
    accel: float
    brake_logit: float


def decode_ego(ego: np.ndarray) -> EgoDecode:
    """The decode METEOR's own evaluator uses: argmax mode, that mode's block."""
    v = np.asarray(ego, dtype=np.float32).reshape(-1)
    if v.shape != (EGO_LEN,):
        raise ValueError(f"ego output must have {EGO_LEN} values, got {v.shape}")
    logits = v[EGO_K * 12 : EGO_K * 12 + EGO_K]
    k = int(np.argmax(logits))
    waypoints = v[k * 12 : (k + 1) * 12].reshape(EGO_HORIZON, 2).astype(np.float64)
    tail = EGO_K * 12 + EGO_K
    return EgoDecode(
        waypoints=waypoints,
        mode=k,
        logits=[float(x) for x in logits],
        steer_rad=float(v[tail]),
        accel=float(v[tail + 1]),
        brake_logit=float(v[tail + 2]),
    )


def extend_horizon(waypoints: np.ndarray, dt: float, horizon: float) -> np.ndarray:
    """Continue the last segment until the plan reaches *horizon* seconds.

    Constant displacement per step from the last two waypoints: the model's
    velocity and heading at the end of its plan, held. Nothing is changed
    inside the model's own horizon.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    n_needed = int(round(horizon / dt))
    if len(waypoints) >= n_needed:
        return waypoints
    step = waypoints[-1] - waypoints[-2]
    extra = waypoints[-1] + step * np.arange(1, n_needed - len(waypoints) + 1)[:, None]
    return np.vstack([waypoints, extra])


class MeteorPolicy(Policy):
    cameras = tuple(SLOT_FROM_CAMERA)
    waypoint_dt = WAYPOINT_DT
    # Single-frame model: there is no warm-up to manufacture.
    self_prime = False

    def __init__(self, onnx: str | Path | None = None, providers: str | None = None) -> None:
        self.onnx = Path(onnx or os.environ.get("METEOR_ONNX", ""))
        self.provider_mode = (providers or os.environ.get("METEOR_PROVIDERS", "cuda")).strip().lower()
        self.threads = ort_threads()
        self.horizon_s = horizon_s()
        self.front_only = front_only_rule()
        self.session: Any = None
        self.active_providers: list[str] = []
        self._last: dict[str, Any] = {}

    # ---- lifecycle -------------------------------------------------------

    def load(self) -> None:
        # The path check comes before the import on purpose: a missing file
        # must say so, not surface as a stack trace from inside onnxruntime.
        if not str(self.onnx) or not self.onnx.is_file():
            raise RuntimeError(
                f"METEOR ONNX not found at {self.onnx or '<unset>'}: set METEOR_ONNX "
                "(model.yaml supplies {weights}/meteor/meteor_v157c3Z.onnx) or run setup.sh"
            )
        import onnxruntime as ort

        want_cuda = self.provider_mode.startswith("cuda")
        if want_cuda and hasattr(ort, "preload_dlls"):
            # cuDNN/cuBLAS come from NVIDIA's CUDA 13 pip wheels (nvidia-cudnn-cu13,
            # nvidia-cublas==13.*, ... -- see setup.sh); there is no system CUDA
            # on the reference box for the loader to find.
            ort.preload_dlls()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        providers: list[Any]
        if want_cuda:
            mem_mb = int(os.environ.get("METEOR_GPU_MEM_MB", str(DEFAULT_GPU_MEM_MB)))
            cuda_opts = {
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": mem_mb * 1024 * 1024,
                "cudnn_conv_algo_search": "HEURISTIC",
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx), opts, providers=providers)
        self.active_providers = list(self.session.get_providers())
        if want_cuda and "CUDAExecutionProvider" not in self.active_providers:
            message = f"CUDA requested but onnxruntime is running on {self.active_providers} (~4 s a frame)"
            if self.provider_mode == "cuda-strict":
                raise RuntimeError(message)
            logger.warning(message)
        self._check_signature()
        self._warm_up()
        logger.info(
            "meteor ready: %s, %d threads, horizon %.1f s, front camera alone: %s",
            self.active_providers,
            self.threads,
            self.horizon_s,
            self.front_only,
        )

    def _check_signature(self) -> None:
        inputs = {i.name: i for i in self.session.get_inputs()}
        expected = {"imgs", "K", "T_cam_ego", "v0"}
        if set(inputs) != expected:
            raise RuntimeError(
                f"{self.onnx.name} takes {sorted(inputs)}; this adapter feeds the camera-only graph "
                f"{sorted(expected)} (a LiDAR-input export needs a different feed)"
            )
        shape = list(inputs["imgs"].shape)
        if "uint8" not in inputs["imgs"].type or shape != [1, len(SLOTS), 3, INPUT_H, INPUT_W]:
            raise RuntimeError(
                f"imgs is {inputs['imgs'].type} {shape}; expected uint8 [1, 8, 3, {INPUT_H}, {INPUT_W}]"
            )
        if "ego" not in {o.name for o in self.session.get_outputs()}:
            raise RuntimeError(f"{self.onnx.name} has no 'ego' output")

    def _warm_up(self) -> None:
        """One forward on a blank frame, so a kernel failure surfaces at startup."""
        k = np.array([[600.0, 0.0, INPUT_W / 2], [0.0, 600.0, INPUT_H / 2], [0.0, 0.0, 1.0]], np.float32)
        feed = {
            "imgs": np.zeros((1, len(SLOTS), 3, INPUT_H, INPUT_W), np.uint8),
            "K": np.tile(k, (1, len(SLOTS), 1, 1)),
            "T_cam_ego": np.tile(np.eye(4, dtype=np.float32), (1, len(SLOTS), 1, 1)),
            "v0": np.zeros(1, np.float32),
        }
        decode_ego(self._run(feed))

    def _run(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        # Only the ego head is requested; onnxruntime prunes what nothing asked for.
        return self.session.run(["ego"], feed)[0]

    def reset(self) -> None:
        self._last = {}

    def observe(self, obs: Observation) -> None:
        """Nothing to record: the graph is a pure function of one frame."""

    # ---- inference -------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        if self.session is None:
            raise RuntimeError("load() must be called before infer()")
        feed, present, rule = build_inputs(obs, self.front_only)
        decoded = decode_ego(self._run(feed))
        if not np.isfinite(decoded.waypoints).all():
            # A crash must not become a score: never substitute a plan here.
            raise RuntimeError(
                f"METEOR returned non-finite waypoints (mode {decoded.mode}, logits {decoded.logits})"
            )
        waypoints = extend_horizon(decoded.waypoints, WAYPOINT_DT, self.horizon_s)
        extras: dict[str, Any] = {
            "meteor_mode": decoded.mode,
            "meteor_mode_logits": decoded.logits,
            "meteor_steer_rad": decoded.steer_rad,
            "meteor_accel": decoded.accel,
            "meteor_brake_logit": decoded.brake_logit,
            "meteor_native_horizon_s": NATIVE_HORIZON_S,
        }
        if len(waypoints) > EGO_HORIZON:
            extras["meteor_extrapolated_to_s"] = float(len(waypoints) * WAYPOINT_DT)
        self._last = {
            **extras,
            "cameras_present": list(present),
            "cameras_absent": [s for s in SLOTS if s not in present],
            "absent_rule": rule,
            "v0": float(obs.speed),
        }
        return Trajectory(waypoints, dt=WAYPOINT_DT, extras=extras)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "onnx": self.onnx.name,
            "providers": self.active_providers,
            "intra_op_threads": self.threads,
            "horizon_s": self.horizon_s,
            "front_only": self.front_only,
            "last": self._last or "no frame seen yet",
        }


#: Paddock looks for this name when loading an adapter.
POLICY = MeteorPolicy
