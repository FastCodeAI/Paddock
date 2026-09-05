"""Driving openpilot's supercombo ONNX directly, without the tinygrad JIT.

`selfdrive/modeld/modeld.py::ModelState` loads a `TinyJit` pickle that
SConstruct builds against a GPU backend from `tinygrad_repo`. That pickle is
what upstream runs, and where it exists this module is not needed --
`policy.py` prefers it. But it does not exist on a machine with no GPU (or with
`tinygrad_repo` unpopulated, which is the state of a plain clone), and the two
ONNX files ship in the repo regardless.

So this drives the ONNX with onnxruntime and reimplements, in numpy, the glue
that `compile_modeld.py` wraps around it: the perspective warp, the channel
packing, and the four input queues. Everything else -- constants, output
slicing, parsing, the action head -- is imported from openpilot unchanged.

The four things that are easy to get wrong, and are the reason this is a
careful port rather than a quick one:

1. **`img` is not two adjacent frames.** `frame_skip = MODEL_RUN_FREQ //
   MODEL_CONTEXT_FREQ = 20 // 5 = 4`, and `img_q` is 5 deep sampled `[::4]`, so
   the two frames are 0.2 s apart. Feeding adjacent frames shows the network a
   quarter of the true apparent motion; measured on comma2k19 that alone takes
   ego-speed error from 4 % to 74 % and ADE from 3.1 m to 41.9 m.
2. **`features_buffer` lags by three ticks.** `feat_q` is 96 deep sampled
   `[::4]`, which yields indices 0, 4, ..., 92 -- the newest inserted feature at
   95 is not sampled. Reproduced, not "fixed".
3. **`desire_pulse` is max-pooled** over each 4-frame window, not subsampled, so
   a one-frame pulse cannot fall between samples.
4. **`hidden_state` is fed back one tick late.** `ModelState.run` writes
   `prev_feat` *after* the model call and `run_policy` pushes it *before*, so
   frame t's queue receives frame t-1's hidden state.

The warp is nearest-neighbour with border replicate, matching
`warp_perspective_tinygrad` exactly -- not bilinear, which would be a different
input than the network was trained on.
"""

from __future__ import annotations

import codecs
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_UV_SCALE = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])

_ORT_DTYPES = {
    "tensor(uint8)": np.uint8,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
}


# --------------------------------------------------------------------------
# Warp
# --------------------------------------------------------------------------


def perspective_index_map(m, dst_shape, src_shape, stride_pad: int = 0):
    """Flat gather indices implementing `compile_modeld.warp_perspective_tinygrad`.

    `m` maps destination (model-frame) pixels to source (camera-frame) pixels --
    the direction `get_warp_matrix` returns. Rounding is `np.round`, matching
    tinygrad's `Tensor.round`: both are round-half-to-even.
    """
    m = np.asarray(m, dtype=np.float64)
    w_dst, h_dst = dst_shape
    h_src, w_src = src_shape
    x = np.tile(np.arange(w_dst, dtype=np.float64), h_dst)
    y = np.repeat(np.arange(h_dst, dtype=np.float64), w_dst)

    sx = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    sy = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    sw = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        sx, sy = sx / sw, sy / sw

    xr, yr = np.round(sx), np.round(sy)
    in_bounds = (xr >= 0) & (xr <= w_src - 1) & (yr >= 0) & (yr <= h_src - 1)
    xr = np.nan_to_num(xr, nan=0.0, posinf=0.0, neginf=0.0)
    yr = np.nan_to_num(yr, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.clip(yr, 0, h_src - 1).astype(np.int64) * (w_src + stride_pad) + np.clip(
        xr, 0, w_src - 1
    ).astype(np.int64)
    return idx, in_bounds


@dataclass
class WarpPlan:
    """Precomputed gather maps: one camera frame -> one 6-channel model tensor.

    The maps depend only on the warp matrix and the frame geometry, both fixed
    for a whole sequence, so each frame costs one fancy-index gather per plane.
    """

    luma_idx: np.ndarray
    chroma_idx: np.ndarray
    model_size: tuple[int, int]
    in_bounds_frac: float

    @classmethod
    def build(cls, warp_matrix, model_size, src_size) -> WarpPlan:
        model_w, model_h = model_size
        src_w, src_h = src_size
        y_idx, y_ib = perspective_index_map(warp_matrix, (model_w, model_h), (src_h, src_w))
        uv_idx, _ = perspective_index_map(
            np.asarray(warp_matrix) * _UV_SCALE,
            (model_w // 2, model_h // 2),
            (src_h // 2, src_w // 2),
        )
        grid = y_idx.reshape(model_h, model_w)
        # frames_to_tensor's channel order: even/even, odd/even, even/odd, odd/odd.
        luma = np.stack([grid[0::2, 0::2], grid[1::2, 0::2], grid[0::2, 1::2], grid[1::2, 1::2]])
        return cls(
            luma_idx=np.ascontiguousarray(luma),
            chroma_idx=np.ascontiguousarray(uv_idx.reshape(model_h // 2, model_w // 2)),
            model_size=model_size,
            in_bounds_frac=float(y_ib.mean()),
        )

    def apply(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(Y, U, V) planes -> (6, model_h//2, model_w//2) uint8."""
        model_w, model_h = self.model_size
        out = np.empty((6, model_h // 2, model_w // 2), dtype=np.uint8)
        out[:4] = y.reshape(-1)[self.luma_idx]
        out[4] = u.reshape(-1)[self.chroma_idx]
        out[5] = v.reshape(-1)[self.chroma_idx]
        return out


def rgb_to_yuv420(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB HWC -> BT.601 limited-range (Y, U, V) planes.

    The network consumes YUV, and paddock's protocol carries RGB, so exactly one
    conversion happens and it happens here. Odd dimensions are truncated to even
    before subsampling, since a 4:2:0 chroma plane is only defined on pairs.
    """
    a = np.asarray(rgb, dtype=np.float32)
    h, w = a.shape[0] & ~1, a.shape[1] & ~1
    a = a[:h, :w]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    u = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    v = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    box = lambda p: (p[0::2, 0::2] + p[0::2, 1::2] + p[1::2, 0::2] + p[1::2, 1::2]) * 0.25  # noqa: E731
    return (
        np.clip(y, 0, 255).astype(np.uint8),
        np.clip(box(u), 0, 255).astype(np.uint8),
        np.clip(box(v), 0, 255).astype(np.uint8),
    )


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    input_shapes: dict
    input_dtypes: dict
    output_name: str
    output_width: int
    output_slices: dict
    model_checkpoint: str | None

    @property
    def n_frames(self) -> int:
        return self.input_shapes["img"][1] // 6

    @property
    def feature_dim(self) -> int:
        return int(np.prod(self.input_shapes["features_buffer"][2:]))

    @property
    def feature_depth(self) -> int:
        return self.input_shapes["features_buffer"][1]

    @property
    def desire_depth(self) -> int:
        return self.input_shapes["desire_pulse"][1]


class SupercomboSession:
    """One loaded supercombo ONNX graph."""

    def __init__(self, onnx_path: Path, intra_op_threads: int = 8) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        providers = ort.get_available_providers()
        preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=preferred)
        self.providers = self.session.get_providers()

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        meta = self.session.get_modelmeta().custom_metadata_map
        raw = meta.get("output_slices")
        if raw is None:
            raise ValueError(
                f"{self.onnx_path.name} has no 'output_slices' in metadata_props; "
                "openpilot's own get_model_metadata.py asserts on this too"
            )
        self.spec = ModelSpec(
            input_shapes={i.name: tuple(i.shape) for i in inputs},
            input_dtypes={i.name: _ORT_DTYPES[i.type] for i in inputs},
            output_name=outputs[0].name,
            output_width=int(outputs[0].shape[-1]),
            output_slices=pickle.loads(codecs.decode(raw.encode(), "base64")),
            model_checkpoint=meta.get("model_checkpoint"),
        )
        self._check_graph_contract()

    def _check_graph_contract(self) -> None:
        """Refuse a supercombo graph this module was not written against.

        This matters because openpilot's model input set has changed more than
        once. The graph expected here takes
        `{img, big_img, features_buffer, desire_pulse, traffic_convention,
        action_t}` and carries `output_slices` in `metadata_props`, and that is
        what `repos.yaml`'s pin must resolve to for the ONNX backend to work.

        **This module was developed and verified against openpilot
        `675ff5698`.** `repos.yaml` currently pins `555f48c5d`, which is what
        the recorded kitscenes / neuroncap / bench2drive runs were produced
        against -- so the pin is deliberately *not* changed here, because moving
        it would silently make those numbers incomparable. If the pinned
        revision's graph differs, this raises with the mismatch named instead of
        failing later on a shape error, and the fix is to pin the verified
        revision (and re-run whatever else that pin affects) rather than to
        loosen this check.
        """
        expected = {
            "img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t",
        }
        got = set(self.spec.input_shapes)
        if got != expected:
            raise ValueError(
                f"{self.onnx_path.name} declares inputs {sorted(got)}, but this backend was "
                f"written for {sorted(expected)}.\n"
                "openpilot's supercombo input set has changed across revisions; this module was "
                "verified against openpilot 675ff5698. Pin repos.yaml to a revision whose graph "
                "matches, or use the tinygrad backend, which tracks whatever the checkout ships."
            )
        img = self.spec.input_shapes["img"]
        if len(img) != 4 or img[1] % 6 != 0:
            raise ValueError(f"{self.onnx_path.name}: unexpected img shape {img}, want (1, 6*n, h, w)")
        if "hidden_state" not in self.spec.output_slices:
            raise ValueError(
                f"{self.onnx_path.name}: no 'hidden_state' in output_slices, so the feature "
                "recurrence cannot be closed"
            )

    def run(self, feed: dict) -> np.ndarray:
        """Raw (N,) output for one frame, as float32.

        The tinygrad path casts to float32 before parsing (`run_policy`:
        `.cast('float32')`), and `parse_model_outputs.safe_exp` behaves
        differently in float16, so this is not cosmetic.
        """
        typed = {
            k: np.ascontiguousarray(feed[k], dtype=dt) for k, dt in self.spec.input_dtypes.items()
        }
        (out,) = self.session.run([self.spec.output_name], typed)
        return out[0].astype(np.float32)

    def slice_outputs(self, model_output: np.ndarray) -> dict:
        """`ModelState.slice_outputs`, verbatim in effect."""
        return {k: model_output[np.newaxis, v] for k, v in self.spec.output_slices.items()}


# --------------------------------------------------------------------------
# Queues and the run loop
# --------------------------------------------------------------------------


@dataclass
class InputQueues:
    """`compile_modeld.make_input_queues`, as numpy ring buffers."""

    img_q: np.ndarray
    big_img_q: np.ndarray
    feat_q: np.ndarray
    desire_q: np.ndarray
    frame_skip: int
    features_buffer_shape: tuple
    desire_pulse_shape: tuple
    img_shape: tuple

    @classmethod
    def build(cls, spec: ModelSpec, frame_skip: int) -> InputQueues:
        img = spec.input_shapes["img"]
        depth = frame_skip * (spec.n_frames - 1) + 1
        return cls(
            img_q=np.zeros((depth, 6, img[2], img[3]), dtype=np.uint8),
            big_img_q=np.zeros((depth, 6, img[2], img[3]), dtype=np.uint8),
            feat_q=np.zeros((frame_skip * spec.feature_depth, spec.feature_dim), dtype=np.float32),
            desire_q=np.zeros(
                (frame_skip * spec.desire_depth, spec.input_shapes["desire_pulse"][2]),
                dtype=np.float32,
            ),
            frame_skip=frame_skip,
            features_buffer_shape=spec.input_shapes["features_buffer"],
            desire_pulse_shape=spec.input_shapes["desire_pulse"],
            img_shape=img,
        )

    @staticmethod
    def _shift(buf: np.ndarray, new_val) -> None:
        buf[:-1] = buf[1:]
        buf[-1] = new_val

    def push_frames(self, img, big_img):
        self._shift(self.img_q, img)
        self._shift(self.big_img_q, big_img)
        return (
            self.img_q[:: self.frame_skip].reshape(self.img_shape),
            self.big_img_q[:: self.frame_skip].reshape(self.img_shape),
        )

    def push_feature(self, prev_feat):
        self._shift(self.feat_q, prev_feat)
        return self.feat_q[:: self.frame_skip].reshape(self.features_buffer_shape)

    def push_desire(self, desire):
        self._shift(self.desire_q, desire)
        depth = self.desire_q.shape[0] // self.frame_skip
        return self.desire_q.reshape(depth, self.frame_skip, -1).max(axis=1).reshape(
            self.desire_pulse_shape
        )


@dataclass
class SupercomboRunner:
    """Stateful per-episode inference. Construct one per sequence.

    The queues, `prev_desire` and `prev_action` are episode state; carrying them
    across sequences would leak one sample's motion into the next one's warm-up.
    """

    session: SupercomboSession
    traffic_convention: np.ndarray
    lat_delay: float = 0.2
    long_actuator_delay: float = 0.15

    queues: InputQueues = field(init=False)
    prev_desire: np.ndarray = field(init=False)
    prev_feat: np.ndarray = field(init=False)
    prev_curvature: float = field(init=False, default=0.0)
    prev_accel: float = field(init=False, default=0.0)
    frames_seen: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        from openpilot.selfdrive.modeld.constants import ModelConstants
        from openpilot.selfdrive.modeld.parse_model_outputs import Parser

        self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
        self.queues = InputQueues.build(self.session.spec, self.frame_skip)
        self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        self.prev_feat = np.zeros(self.session.spec.feature_dim, dtype=np.float32)
        self.parser = Parser()

    @property
    def warmup_frames(self) -> int:
        """Ticks until no queue still holds initialisation zeros."""
        return self.frame_skip * self.session.spec.feature_depth

    def _action_t(self) -> np.ndarray:
        from openpilot.common.realtime import DT_MDL

        # modeld.py: lat/long_action_t = delay + DT_MDL + DT_MDL/2, with
        # LAT_SMOOTH_SECONDS = 0.0 and LONG_SMOOTH_SECONDS = 0.3.
        frame_delay, action_delay = DT_MDL, DT_MDL / 2
        return np.array(
            [
                self.lat_delay + 0.0 + frame_delay + action_delay,
                self.long_actuator_delay + 0.3 + frame_delay + action_delay,
            ],
            dtype=np.float32,
        )

    def run(self, img: np.ndarray, big_img: np.ndarray, v_ego: float, desire=None) -> dict:
        """One model tick. `img`/`big_img` are (6, h/2, w/2) uint8, already warped."""
        from openpilot.selfdrive.modeld.constants import ModelConstants

        spec = self.session.spec
        desire_pulse = (
            np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
            if desire is None
            else np.asarray(desire, dtype=np.float32).copy()
        )
        desire_pulse[0] = 0
        edge = np.where(desire_pulse - self.prev_desire > 0.99, desire_pulse, 0.0)
        self.prev_desire[:] = desire_pulse

        img_in, big_in = self.queues.push_frames(img, big_img)
        feat_in = self.queues.push_feature(self.prev_feat)
        desire_in = self.queues.push_desire(edge)

        raw = self.session.run(
            {
                "img": img_in,
                "big_img": big_in,
                "features_buffer": feat_in,
                "desire_pulse": desire_in,
                "traffic_convention": self.traffic_convention.reshape(
                    spec.input_shapes["traffic_convention"]
                ),
                "action_t": self._action_t().reshape(spec.input_shapes["action_t"]),
            }
        )
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("supercombo output is not finite")

        out = self.parser.parse_outputs(self.session.slice_outputs(raw))
        self.prev_feat = raw[spec.output_slices["hidden_state"]].astype(np.float32)
        out["action_value"] = self._action(out, v_ego)
        self.frames_seen += 1
        return out

    def _action(self, out: dict, v_ego: float) -> dict:
        """`modeld.py::get_action_from_model`, reusing openpilot's helpers."""
        from openpilot.selfdrive.controls.lib.drive_helpers import (
            get_accel_from_plan,
            get_curvature_from_plan,
            should_stop,
            smooth_value,
        )
        from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

        action_t = self._action_t()
        if "action" not in out:
            plan = out["plan"][0]
            accel = get_accel_from_plan(
                plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0],
                ModelConstants.T_IDXS, action_t=float(action_t[1]),
            )
            curvature = get_curvature_from_plan(
                plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
                ModelConstants.T_IDXS, v_ego, float(action_t[0]),
            )
        else:
            accel = out["action"][0, 1]
            curvature = out["action"][0, 0] / (max(1.0, v_ego)) ** 2

        stop = should_stop(v_ego, accel)
        accel = smooth_value(accel, self.prev_accel, 0.3)
        curvature = smooth_value(curvature, self.prev_curvature, 0.0) if v_ego > 0.3 else self.prev_curvature
        self.prev_accel, self.prev_curvature = float(accel), float(curvature)
        return {
            "desiredCurvature": float(curvature),
            "desiredAcceleration": float(accel),
            "shouldStop": bool(stop),
        }
