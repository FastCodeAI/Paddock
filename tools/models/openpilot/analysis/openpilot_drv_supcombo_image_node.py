"""A supercombo node whose image manipulation comes from the caller.

The image notebooks beside it -- the KITScenes image calibration and scene sweep, and the
image-tweak notebooks for KITScenes, comma2k19 and Bench2Drive -- launch this in openpilot's
own interpreter and environment, as `paddock run` launches the adapter's node
(`proc.model_node`), and drive it over localhost HTTP. Paddock never imports a model and
neither does a notebook: everything that needs openpilot runs in this process, and the
notebooks only compose matrices and lookup tables.

The node is the adapter, not a copy of it. `ImagePolicy` subclasses `OpenpilotPolicy` and
changes what happens between a frame's (Y, U, V) planes and the (6, 128, 256) tensor each
view hands the model -- `View` -- and nothing after it:

- the warp each view is cut with, composed here from two 3x3 matrices the caller sends,
  `calib` (on rays in openpilot's calibrated frame) and `post` (on model pixels):
  `camera_from_calib @ calib @ calib_from_model @ post`, the adapter's own product with
  `calib` where its height scale puts `diag(1, 1, s)`;
- `supersample`: each model pixel the mean of n x n nearest-neighbour samples spread over
  its own footprint, instead of one;
- `kernel`: a small filter on the full-resolution model luma (blur, sharpen);
- `lut`: one 256-entry lookup table per plane (gamma, contrast, saturation);
- `fill`: a blank view, for ablation.

The 20 Hz repetition, the desire pulse, the runner, the plan and the extras are the
adapter's own code. A request with no views is therefore the stock adapter, and `/stock`
runs an untouched `OpenpilotPolicy` without the frame cache so the notebook can check that
bit for bit rather than assume it.

Endpoints, beside `build_app`'s `/alive`, `/whoami`, `/reset` and `/diagnostics`:

- `/geometry`: decode and cache one sample's frames; return the camera as openpilot sees
  it (`camera_from_calib`, each view's `calib_from_model`, the stock warps).
- `/run`: one sample through `infer_sequence` with the given views; returns the plan, the
  extras, the warps used and, if asked, the keyframe's model input.
- `/stock`: one sample through an untouched `OpenpilotPolicy`.

Single-camera sources only (KITScenes' CAM_FRONT feeds both views), on the ONNX backend.
"""

from __future__ import annotations

import argparse
import base64
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from paddock.adapters.models.openpilot import supercombo
from paddock.adapters.models.openpilot.policy import CameraAdapter, OpenpilotPolicy
from paddock.policy import protocol
from paddock.policy.protocol import Observation
from paddock.policy.server import build_app

logger = logging.getLogger("image_node")

NAME = "openpilot-image-node"
VIEWS = ("img", "big_img")


def _openpilot_geometry():
    """(view_frame_from_device_frame, rot_from_euler, {view: calib_from_model}) from openpilot."""
    from openpilot.common.transformations.camera import view_frame_from_device_frame
    from openpilot.common.transformations.model import calib_from_medmodel, calib_from_sbigmodel
    from openpilot.common.transformations.orientation import rot_from_euler

    return (
        view_frame_from_device_frame,
        rot_from_euler,
        {"img": calib_from_medmodel, "big_img": calib_from_sbigmodel},
    )


def camera_from_calib(cam: CameraAdapter) -> np.ndarray:
    """The product `get_warp_matrix` and `_height_scaled` both start from, computed the same way.

    Same factors, same order, same float32 `rpy`: an identity `calib` then reproduces the
    stock warp exactly, not to a tolerance, which `/run` relies on and the notebook checks.
    """
    view_from_device, rot_from_euler, _ = _openpilot_geometry()
    return cam.intrinsics @ view_from_device @ rot_from_euler(cam.rpy)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------


class FrameRef(np.ndarray):
    """A frame's shape standing in for its pixels; `FrameCache` swaps it back for its planes."""

    path = ""


class FrameCache:
    """Every frame's (Y, U, V) planes, converted once per node.

    The adapter decodes and converts every frame of every sequence it is shown: 0.2-0.3 s
    for a 3504x2272 KITScenes frame, against about 20 ms for a model tick on CPU. A sweep
    shows the same 49 frames dozens of times, so the planes are kept.

    `protocol.load_frame` hands `Observation.image` a zero-stride stand-in of the frame's
    shape (`CameraAdapter` reads the shape and nothing else), and
    `supercombo.rgb_to_yuv420` swaps the stand-in for the planes it made from the decoded
    frame. On the ONNX backend `prepare` returns its argument, so the planes reach the gather
    exactly as the adapter would have made them: the cache changes the cost, not the values.
    `/stock` turns it off to show that.
    """

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self.planes: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.enabled = True
        self._load_frame = protocol.load_frame
        self._rgb_to_yuv420 = supercombo.rgb_to_yuv420

    def install(self) -> None:
        # Both are looked up at call time (`Observation.image` reads the module global;
        # `_infer_onnx` imports `rgb_to_yuv420` inside the method), so this is enough.
        protocol.load_frame = self.load_frame
        supercombo.rgb_to_yuv420 = self.rgb_to_yuv420

    def load_frame(self, path):
        if not self.enabled:
            return self._load_frame(path)
        key = str(path)
        if key in self.planes:
            self.planes.move_to_end(key)
        else:
            rgb = self._load_frame(key)
            self.planes[key] = self._rgb_to_yuv420(rgb)
            self.shapes[key] = rgb.shape
            while len(self.planes) > self.max_frames:
                self.planes.popitem(last=False)
        ref = np.broadcast_to(np.zeros((), dtype=np.uint8), self.shapes[key]).view(FrameRef)
        ref.path = key
        return ref

    def rgb_to_yuv420(self, rgb):
        if isinstance(rgb, FrameRef):
            return self.planes[rgb.path]
        return self._rgb_to_yuv420(rgb)


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


def _shift(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def full_luma(tensor: np.ndarray) -> np.ndarray:
    """(6, h, w) model tensor -> its (2h, 2w) luma, undoing `WarpPlan`'s sub-plane order."""
    h, w = tensor.shape[1:]
    y = np.empty((2 * h, 2 * w), dtype=tensor.dtype)
    y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = tensor[0], tensor[1], tensor[2], tensor[3]
    return y


def filter_luma(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The tensor with its full-resolution luma convolved with `kernel`, edges repeated."""
    y = full_luma(tensor).astype(np.float32)
    r = kernel.shape[0] // 2
    padded = np.pad(y, r, mode="edge")
    acc = np.zeros_like(y)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel[i, j] != 0.0:
                acc += kernel[i, j] * padded[i : i + y.shape[0], j : j + y.shape[1]]
    y = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
    out = tensor.copy()
    out[0], out[1], out[2], out[3] = y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2]
    return out


@dataclass
class View:
    """One model view: the warp it is cut with, and what happens to the warped tensor.

    Also the adapter's warp plan for that view. `_infer_onnx` calls `apply(y, u, v)` on
    whatever `_plans` holds, so everything here runs where the adapter's gather runs, once
    per frame, and the model sees the result on every tick that frame is repeated. In order:
    the gather (supersampled or not), the luma filter, the lookup tables, the fill.
    """

    warp: np.ndarray
    plans: list
    kernel: np.ndarray | None = None
    lut: np.ndarray | None = None
    fill: int | None = None
    last: np.ndarray | None = None

    @classmethod
    def build(cls, cam: CameraAdapter, name: str, spec: dict | None, model_size) -> View:
        spec = spec or {}
        calib, post = spec.get("calib"), spec.get("post")
        if calib is None and post is None:
            warp = cam.transforms[name]
        else:
            _, _, calib_from_model = _openpilot_geometry()
            m = np.eye(3) if calib is None else np.asarray(calib, dtype=np.float64)
            p = np.eye(3) if post is None else np.asarray(post, dtype=np.float64)
            warp = (camera_from_calib(cam) @ m @ calib_from_model[name] @ p).astype(np.float32)
        return cls.from_warp(warp, spec, model_size, cam.size)

    @classmethod
    def from_warp(cls, warp: np.ndarray, spec: dict, model_size, source_size) -> View:
        """Everything after the warp's composition, which alone needs openpilot.

        The notebook builds its views here too, from `/geometry`'s matrices, so it can draw
        what the model is handed without running the model, and checks against `/run`'s
        `inputs` that the two agree bit for bit.
        """
        n = int(spec.get("supersample") or 1)
        if n == 1:
            plans = [supercombo.WarpPlan.build(warp, model_size, source_size)]
        else:
            # Offsets in model pixels, centred on the pixel: n = 3 samples at -1/3, 0, +1/3.
            offsets = [(i + 0.5) / n - 0.5 for i in range(n)]
            plans = [
                supercombo.WarpPlan.build(warp @ _shift(dx, dy), model_size, source_size)
                for dy in offsets
                for dx in offsets
            ]
        kernel = spec.get("kernel")
        lut = spec.get("lut")
        return cls(
            warp=warp,
            plans=plans,
            kernel=None if kernel is None else np.asarray(kernel, dtype=np.float32),
            lut=None if lut is None else np.asarray(lut, dtype=np.uint8),
            fill=spec.get("fill"),
        )

    @property
    def in_bounds_frac(self) -> float:
        return float(np.mean([plan.in_bounds_frac for plan in self.plans]))

    def apply(self, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if len(self.plans) == 1:
            out = self.plans[0].apply(y, u, v)
        else:
            acc = sum(plan.apply(y, u, v).astype(np.float32) for plan in self.plans)
            out = np.clip(np.rint(acc / len(self.plans)), 0, 255).astype(np.uint8)
        if self.kernel is not None:
            out = filter_luma(out, self.kernel)
        if self.lut is not None:
            out[:4] = self.lut[0][out[:4]]
            out[4] = self.lut[1][out[4]]
            out[5] = self.lut[2][out[5]]
        if self.fill is not None:
            out[:4] = self.fill
            out[4:] = 128
        self.last = out
        return out


class ImagePolicy(OpenpilotPolicy):
    """The adapter, with each view's `View` and the tick rate set per request."""

    def __init__(self) -> None:
        super().__init__(backend="onnx")
        #: This request's {view: spec}; None is the stock warps and nothing else.
        self.spec: dict | None = None
        #: Model ticks per 0.1 s frame; None is the adapter's measured 2.
        self.tick_rate: float | None = None
        self.views: dict[str, View] = {}
        self._frames = 0

    def reset(self) -> None:
        super().reset()
        self.views = {}
        self._frames = 0

    def _bind_cameras(self, obs: Observation) -> bool:
        fresh = self.camera is None
        two = super()._bind_cameras(obs)
        if two:
            raise ValueError(f"{NAME} handles single-camera sources only; this sample has two")
        if fresh:
            cam = self.camera
            if cam.mode != "native":
                raise ValueError(f"{NAME} needs the ONNX backend's native frames, not {cam.mode!r}")
            spec = self.spec or {}
            self.views = {
                name: View.build(cam, name, spec.get(name), self._model_size(name)) for name in VIEWS
            }
            cam.transforms = {name: view.warp for name, view in self.views.items()}
            # Seen by `_infer_onnx` as plans already built for this adapter, so it uses them.
            self._plans = dict(self.views)
            self._plans_for = (cam, cam)
        return two

    def _repeats(self, obs: Observation) -> int:
        repeats = super()._repeats(obs)
        # Only the 10 Hz frames, which the adapter shows twice. The scene's first frame
        # repeated before keyframe 40 has no time step and keeps its single tick.
        if self.tick_rate is None or repeats != 2:
            return repeats
        i, rate = self._frames, self.tick_rate
        self._frames += 1
        return max(1, math.floor(rate * (i + 1)) - math.floor(rate * i))


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


def _watch_parent(parent: int) -> None:
    """Exit when the notebook's kernel does, instead of holding gigabytes of frames."""
    while True:
        time.sleep(5)
        if os.getppid() != parent:
            logger.info("parent %d gone; exiting", parent)
            os._exit(0)


def _result(traj, t0: float) -> dict:
    out = {"trajectory": traj.waypoints.tolist(), "dt": traj.dt, "extras": traj.extras or {}}
    out["ms"] = (time.monotonic() - t0) * 1000.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--nonce", default=None)
    parser.add_argument("--max-frames", type=int, default=600, help="frames' planes kept, about 12 MB each")
    parser.add_argument("--parent", type=int, default=None, help="exit when this process is gone")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    if args.parent:
        threading.Thread(target=_watch_parent, args=(args.parent,), daemon=True).start()

    cache = FrameCache(args.max_frames)
    cache.install()
    policy = ImagePolicy()
    policy.load()
    desire_default = policy.desire_mode
    stock: list[OpenpilotPolicy] = []
    logger.info("ready on port %d, providers %s", args.port, policy.session.providers)

    app = build_app(policy, model=NAME, nonce=args.nonce)

    def observations(body: dict, cameras) -> list[Observation]:
        return [Observation.from_wire(item, cameras=cameras) for item in body["observations"]]

    # Async, like build_app's routes: they run one at a time on the event loop, which is
    # what a single stateful policy needs.
    @app.post("/geometry")
    async def geometry(body: dict) -> dict:
        t0 = time.monotonic()
        obs = observations(body, policy.cameras)
        for o in obs:
            o.image("CAM_FRONT")
        cam = CameraAdapter(obs[-1].pinhole("CAM_FRONT"), obs[-1].camera2ego["CAM_FRONT"], resample=False)
        _, _, calib_from_model = _openpilot_geometry()
        return {
            "source_size": list(cam.size),
            "intrinsics": cam.intrinsics.tolist(),
            "rpy": [float(x) for x in cam.rpy],
            "camera_from_calib": camera_from_calib(cam).tolist(),
            "calib_from_model": {name: calib_from_model[name].tolist() for name in VIEWS},
            "model_size": {name: list(policy._model_size(name)) for name in VIEWS},
            "stock": {name: cam.transforms[name].astype(np.float64).tolist() for name in VIEWS},
            "providers": list(policy.session.providers),
            "frames_cached": len(cache.planes),
            "ms": (time.monotonic() - t0) * 1000.0,
        }

    @app.post("/run")
    async def run(body: dict) -> dict:
        t0 = time.monotonic()
        obs = observations(body, policy.cameras)
        policy.spec = body.get("views") or None
        policy.tick_rate = body.get("tick_rate")
        policy.desire_mode = policy._desire_mode(body.get("desire") or desire_default)
        try:
            traj = policy.infer_sequence(obs)
        finally:
            policy.spec, policy.tick_rate, policy.desire_mode = None, None, desire_default
        out = _result(traj, t0)
        out["warps"] = {name: policy.views[name].warp.astype(np.float64).tolist() for name in VIEWS}
        out["in_bounds"] = {name: policy.views[name].in_bounds_frac for name in VIEWS}
        out["ticks"] = int(policy.model.frames_seen)
        if body.get("inputs"):
            out["inputs"] = {
                name: base64.b64encode(policy.views[name].last.tobytes()).decode() for name in VIEWS
            }
        return out

    @app.post("/stock")
    async def run_stock(body: dict) -> dict:
        t0 = time.monotonic()
        if not stock:
            reference = OpenpilotPolicy(backend="onnx")
            reference.load()
            stock.append(reference)
        cache.enabled = False
        try:
            traj = stock[0].infer_sequence(observations(body, stock[0].cameras))
        finally:
            cache.enabled = True
        out = _result(traj, t0)
        out["warps"] = {name: stock[0].camera.transforms[name].astype(np.float64).tolist() for name in VIEWS}
        out["ticks"] = int(stock[0].model.frames_seen)
        return out

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
