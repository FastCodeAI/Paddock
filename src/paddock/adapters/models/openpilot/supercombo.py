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
2. **`features_buffer` can lag by three ticks.** On the combined graph `feat_q`
   is 96 deep sampled `[::4]`, which yields indices 0, 4, ..., 92 -- the newest
   inserted feature at 95 is not sampled. Reproduced, not "fixed".
3. **`desire_pulse` is max-pooled** over each 4-frame window, not subsampled, so
   a one-frame pulse cannot fall between samples.
4. **`hidden_state` can be fed back one tick late.** On the combined graph
   `ModelState.run` writes `prev_feat` *after* the model call and `run_policy`
   pushes it *before*, so frame t's queue receives frame t-1's hidden state.

The warp is nearest-neighbour with border replicate, matching
`warp_perspective_tinygrad` exactly -- not bilinear, which would be a different
input than the network was trained on.

## Two graph layouts ship under this name

Which one a checkout has is not a detail, and it is not a version anyone can
infer from a tag ordering -- both of these are real revisions of this repo:

- **Combined** (`driving_supercombo.onnx`): one graph taking `{img, big_img,
  features_buffer, desire_pulse, traffic_convention, action_t}`. This is what
  `repos.yaml` pins (`555f48c5d`, 2026-08-19) and what every recorded kitscenes
  / neuroncap / bench2drive run was produced against. `SupercomboSession`.
- **Split** (`driving_vision.onnx` + `driving_policy.onnx`): vision takes the
  two image streams, the policy takes `{features_buffer, desire_pulse,
  traffic_convention}`, and they run in series. This is openpilot v0.11.1
  (`4df40d2c1`, 2026-05-29). `SplitSession`.

The two differ in three ways that change the numbers, which is why the split
path is a separate class rather than a path resolution:

- The vision graph runs *first*, so the feature it produces is pushed into
  `feat_q` before the policy graph reads it: the split layout has **no
  recurrence lag** (item 4 above), and `feat_q` is `frame_skip * (D - 1) + 1`
  deep so that `[::frame_skip]` samples the newest entry (no item 2 either).
- The policy graph has no `action_t` input and no `action` output, so the
  action is always derived from the plan.
- Its tensors are float16 end to end; the combined graph's are float32.

Everything above the graph -- queues, warp, parsing, the action head -- is
shared, so a number from one layout is comparable with a number from the other
to exactly the extent that the two networks are.
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


class Graph:
    """One loaded ONNX graph: its declared inputs, its `output_slices`, and a run.

    Both layouts are made of these -- the combined one has a single graph, the
    split one has two -- so loading, dtype coercion and slicing are written
    once and the session classes hold only what actually differs.
    """

    def __init__(self, session, path: Path) -> None:
        self.session = session
        self.path = Path(path)
        self.name = self.path.name
        self.providers = session.get_providers()
        meta = session.get_modelmeta().custom_metadata_map
        raw = meta.get("output_slices")
        if raw is None:
            raise ValueError(
                f"{self.name} has no 'output_slices' in metadata_props; "
                "openpilot's own get_model_metadata.py asserts on this too"
            )
        outputs = session.get_outputs()
        self.input_shapes = {i.name: tuple(i.shape) for i in session.get_inputs()}
        self.input_dtypes = {i.name: _ORT_DTYPES[i.type] for i in session.get_inputs()}
        self.output_name = outputs[0].name
        self.output_width = int(outputs[0].shape[-1])
        self.output_slices = pickle.loads(codecs.decode(raw.encode(), "base64"))
        self.model_checkpoint = meta.get("model_checkpoint")

    @classmethod
    def open(cls, onnx_path: Path, intra_op_threads: int = 8) -> Graph:
        import onnxruntime as ort

        # CUDA and cuDNN arrive as pip wheels under `site-packages/nvidia/`, whose
        # lib dirs are on no loader path: onnxruntime dlopens the unversioned
        # `libcudnn.so`, which those wheels do not ship. Without this the CUDA
        # provider loads happily and then fails on the first Conv, mid-run and far
        # from anything that explains why. `preload_dlls` is onnxruntime's own
        # answer to exactly that layout, and is a no-op where the libs are system-
        # installed.
        if hasattr(ort, "preload_dlls"):  # onnxruntime >= 1.21
            ort.preload_dlls()

        onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        providers = ort.get_available_providers()
        preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        return cls(ort.InferenceSession(str(onnx_path), opts, providers=preferred), onnx_path)

    def run(self, feed: dict) -> np.ndarray:
        """Raw (N,) output for one frame, as float32.

        The tinygrad path casts to float32 before parsing (`run_policy`:
        `.cast('float32')`), and `parse_model_outputs.safe_exp` behaves
        differently in float16, so this is not cosmetic -- and the split
        layout's graphs are float16 throughout.
        """
        typed = {k: np.ascontiguousarray(feed[k], dtype=dt) for k, dt in self.input_dtypes.items()}
        (out,) = self.session.run([self.output_name], typed)
        raw = out[0].astype(np.float32)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError(f"{self.name} output is not finite")
        return raw

    def slice_outputs(self, model_output: np.ndarray) -> dict:
        """`ModelState.slice_outputs`, verbatim in effect."""
        return {k: model_output[np.newaxis, v] for k, v in self.output_slices.items()}

    def expect_inputs(self, expected: set[str]) -> None:
        got = set(self.input_shapes)
        if got != expected:
            raise ValueError(
                f"{self.name} declares inputs {sorted(got)}, but this backend was written for "
                f"{sorted(expected)}.\n"
                "openpilot's model input set has changed across revisions. Pin repos.yaml to a "
                "revision whose graph matches one of the two layouts this module implements "
                "(see its docstring), or use the tinygrad backend, which tracks whatever the "
                "checkout ships."
            )


@dataclass(frozen=True)
class ModelSpec:
    """The model-level view of a session, whatever its layout.

    `input_shapes` and `output_slices` are the union across the graphs, which is
    exactly how the rest of the adapter reads them: the queues care about `img`
    and `features_buffer`, `build_extras` cares about the head names, and
    neither has any business knowing which graph produced which.
    """

    input_shapes: dict
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
    """The combined graph: one ONNX taking images, queues and `action_t`.

    `repos.yaml`'s pin (`555f48c5d`) resolves to this, and every recorded
    openpilot run in `results/` was produced through it.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        graph.expect_inputs(
            {"img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"}
        )
        self.spec = ModelSpec(
            input_shapes=dict(graph.input_shapes),
            output_slices=dict(graph.output_slices),
            model_checkpoint=graph.model_checkpoint,
        )
        _check_shapes(self.spec, graph.name)
        if "hidden_state" not in graph.output_slices:
            raise ValueError(
                f"{graph.name}: no 'hidden_state' in output_slices, so the feature "
                "recurrence cannot be closed"
            )

    @property
    def paths(self) -> dict[str, Path]:
        return {"supercombo": self.graph.path}

    @property
    def providers(self) -> list[str]:
        return list(self.graph.providers)

    def make_queues(self, frame_skip: int) -> InputQueues:
        # frame_skip * D, so `[::frame_skip]` stops at D*frame_skip - frame_skip
        # and the newest entry is never sampled. See item 2 in the module
        # docstring: reproduced, not fixed.
        return InputQueues.build(self.spec, frame_skip, frame_skip * self.spec.feature_depth)

    def tick(self, queues: InputQueues, parser, *, img, big_img, desire_edge, traffic_convention, action_t):
        shapes = self.spec.input_shapes
        img_in, big_in = queues.push_frames(img, big_img)
        desire_in = queues.push_desire(desire_edge)
        # Last tick's feature: `ModelState.run` writes `prev_feat` after the
        # call and `run_policy` pushes it before. Item 4 in the module docstring.
        feat_in = queues.push_feature(queues.feature)
        raw = self.graph.run(
            {
                "img": img_in,
                "big_img": big_in,
                "features_buffer": feat_in,
                "desire_pulse": desire_in,
                "traffic_convention": traffic_convention.reshape(shapes["traffic_convention"]),
                "action_t": action_t.reshape(shapes["action_t"]),
            }
        )
        queues.feature = raw[self.graph.output_slices["hidden_state"]].astype(np.float32)
        return parser.parse_outputs(self.graph.slice_outputs(raw))


class SplitSession:
    """The split layout: a vision graph feeding a policy graph, run in series.

    openpilot v0.11.1's `compile_modeld.make_run_policy`, without the JIT: the
    vision graph turns two image streams into a 512-wide feature, that feature
    is pushed into `feat_q`, and the policy graph reads the buffer. Because the
    push happens between the two calls, the policy sees the *current* frame's
    feature -- unlike the combined graph, which sees the previous one.
    """

    def __init__(self, vision: Graph, policy: Graph) -> None:
        self.vision, self.policy = vision, policy
        vision.expect_inputs({"img", "big_img"})
        policy.expect_inputs({"features_buffer", "desire_pulse", "traffic_convention"})
        self.spec = ModelSpec(
            # Union, and `pad` is the one key both graphs declare -- a
            # whole-vector slice neither parser reads, so the collision is
            # harmless. Every head name is unique to its graph.
            input_shapes={**vision.input_shapes, **policy.input_shapes},
            output_slices={**vision.output_slices, **policy.output_slices},
            model_checkpoint=f"vision {vision.model_checkpoint} / policy {policy.model_checkpoint}",
        )
        _check_shapes(self.spec, vision.name)
        if "hidden_state" not in vision.output_slices:
            raise ValueError(
                f"{vision.name}: no 'hidden_state' in output_slices, so there is nothing to "
                "feed the policy graph's features_buffer with"
            )
        if "plan" not in policy.output_slices:
            raise ValueError(f"{policy.name}: no 'plan' in output_slices, so it predicts no path")

    @property
    def paths(self) -> dict[str, Path]:
        return {"vision": self.vision.path, "policy": self.policy.path}

    @property
    def providers(self) -> list[str]:
        return list(self.vision.providers)

    def make_queues(self, frame_skip: int) -> InputQueues:
        # frame_skip * (D - 1) + 1, so `[::frame_skip]` lands exactly on the
        # newest entry: this layout has no recurrence lag to reproduce.
        return InputQueues.build(self.spec, frame_skip, frame_skip * (self.spec.feature_depth - 1) + 1)

    def tick(self, queues: InputQueues, parser, *, img, big_img, desire_edge, traffic_convention, action_t):
        """`action_t` is accepted and ignored: this policy graph has no such
        input, and its action comes from the plan in `SupercomboRunner._action`.
        """
        img_in, big_in = queues.push_frames(img, big_img)
        desire_in = queues.push_desire(desire_edge)

        vision_raw = self.vision.run({"img": img_in, "big_img": big_in})
        queues.feature = vision_raw[self.vision.output_slices["hidden_state"]].astype(np.float32)
        feat_in = queues.push_feature(queues.feature)
        policy_raw = self.policy.run(
            {
                "features_buffer": feat_in,
                "desire_pulse": desire_in,
                "traffic_convention": traffic_convention.reshape(
                    self.policy.input_shapes["traffic_convention"]
                ),
            }
        )
        out = parser.parse_vision_outputs(self.vision.slice_outputs(vision_raw))
        out.update(parser.parse_policy_outputs(self.policy.slice_outputs(policy_raw)))
        return out


def _check_shapes(spec: ModelSpec, name: str) -> None:
    img = spec.input_shapes["img"]
    if len(img) != 4 or img[1] % 6 != 0:
        raise ValueError(f"{name}: unexpected img shape {img}, want (1, 6*n, h, w)")


# --------------------------------------------------------------------------
# Queues and the run loop
# --------------------------------------------------------------------------


@dataclass
class InputQueues:
    """`compile_modeld.make_input_queues`, as numpy ring buffers.

    `feature` is the recurrence state: the most recent `hidden_state`. Both
    layouts push it into `feat_q`, and the only difference between them is
    *when* -- the combined graph pushes it before the call that produces the
    next one, the split layout pushes it between its two calls. One field, two
    orderings; see each session's `tick`.
    """

    img_q: np.ndarray
    big_img_q: np.ndarray
    feat_q: np.ndarray
    desire_q: np.ndarray
    feature: np.ndarray
    frame_skip: int
    features_buffer_shape: tuple
    desire_pulse_shape: tuple
    img_shape: tuple

    @classmethod
    def build(cls, spec: ModelSpec, frame_skip: int, feat_depth: int) -> InputQueues:
        img = spec.input_shapes["img"]
        depth = frame_skip * (spec.n_frames - 1) + 1
        return cls(
            img_q=np.zeros((depth, 6, img[2], img[3]), dtype=np.uint8),
            big_img_q=np.zeros((depth, 6, img[2], img[3]), dtype=np.uint8),
            feat_q=np.zeros((feat_depth, spec.feature_dim), dtype=np.float32),
            desire_q=np.zeros(
                (frame_skip * spec.desire_depth, spec.input_shapes["desire_pulse"][2]),
                dtype=np.float32,
            ),
            feature=np.zeros(spec.feature_dim, dtype=np.float32),
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
        return self.desire_q.reshape(depth, self.frame_skip, -1).max(axis=1).reshape(self.desire_pulse_shape)


@dataclass
class SupercomboRunner:
    """Stateful per-episode inference. Construct one per sequence.

    The queues, `prev_desire` and `prev_action` are episode state; carrying them
    across sequences would leak one sample's motion into the next one's warm-up.
    """

    session: SupercomboSession | SplitSession
    traffic_convention: np.ndarray
    lat_delay: float = 0.2
    long_actuator_delay: float = 0.15

    queues: InputQueues = field(init=False)
    prev_desire: np.ndarray = field(init=False)
    prev_curvature: float = field(init=False, default=0.0)
    prev_accel: float = field(init=False, default=0.0)
    frames_seen: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        from openpilot.selfdrive.modeld.constants import ModelConstants
        from openpilot.selfdrive.modeld.parse_model_outputs import Parser

        self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
        self.queues = self.session.make_queues(self.frame_skip)
        self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
        self.parser = Parser()

    @property
    def warmup_frames(self) -> int:
        """Ticks until no queue still holds initialisation zeros.

        The feature queue is the deepest, and how deep it is depends on the
        layout, so this is read off the queue rather than recomputed.
        """
        return int(self.queues.feat_q.shape[0])

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

        desire_pulse = (
            np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
            if desire is None
            else np.asarray(desire, dtype=np.float32).copy()
        )
        desire_pulse[0] = 0
        edge = np.where(desire_pulse - self.prev_desire > 0.99, desire_pulse, 0.0)
        self.prev_desire[:] = desire_pulse

        out = self.session.tick(
            self.queues,
            self.parser,
            img=img,
            big_img=big_img,
            desire_edge=edge,
            traffic_convention=self.traffic_convention,
            action_t=self._action_t(),
        )
        out["action_value"] = self._action(out, v_ego)
        self.frames_seen += 1
        return out

    def _action(self, out: dict, v_ego: float) -> dict:
        """`modeld.py::get_action_from_model`, reusing openpilot's helpers.

        The split layout has no `action` head, so it always takes the first
        branch -- which is also what v0.11.1's own `get_action_from_model` does.
        """
        from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan, smooth_value
        from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

        from paddock.adapters.models.openpilot import upstream

        action_t = self._action_t()
        if "action" not in out:
            plan = out["plan"][0]
            accel, stop = upstream.accel_and_stop(
                plan[:, Plan.VELOCITY][:, 0],
                plan[:, Plan.ACCELERATION][:, 0],
                ModelConstants.T_IDXS,
                action_t=float(action_t[1]),
                v_ego=v_ego,
            )
            curvature = get_curvature_from_plan(
                plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                plan[:, Plan.ORIENTATION_RATE][:, 2],
                ModelConstants.T_IDXS,
                v_ego,
                float(action_t[0]),
            )
        else:
            accel = out["action"][0, 1]
            curvature = out["action"][0, 0] / (max(1.0, v_ego)) ** 2
            stop = upstream.stopped(v_ego, accel)

        accel = smooth_value(accel, self.prev_accel, 0.3)
        curvature = smooth_value(curvature, self.prev_curvature, 0.0) if v_ego > 0.3 else self.prev_curvature
        self.prev_accel, self.prev_curvature = float(accel), float(curvature)
        return {
            "desiredCurvature": float(curvature),
            "desiredAcceleration": float(accel),
            "shouldStop": bool(stop),
        }
