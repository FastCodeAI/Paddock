"""openpilot supercombo as a paddock Policy.

Adapted from the NeuroNCAP model node (`servers/openpilot_server.py`). The
model-specific reasoning below is unchanged from that node; what changed is the
seams -- it now implements `Policy`, so the same class serves open-loop scoring
and closed-loop simulation instead of one implementation per loop type.

What this adapts, and why each piece is load-bearing:

*Camera.* The network never sees a raw frame -- it sees two warped 512x256
views, roughly 31.4 x 15.9 deg (`img`) and 58.7 x 31.3 deg (`big_img`), built by
`get_warp_matrix` from the camera intrinsics and the mounting rotation. Both
views must fit inside what the source camera actually saw, or the warp samples
past the edge and border pixels are smeared into the model's input.
`diagnostics()` reports the measured in-bounds fraction, not an estimate.

*Two backends.* `tinygrad` runs openpilot's own `ModelState`, which loads the
`TinyJit` pickle SConstruct builds -- upstream's exact execution path, and the
default where it exists. `onnx` drives the shipped ONNX files with onnxruntime
(`supercombo.py`), for a machine with no GPU or an unpopulated `tinygrad_repo`.
The two differ in one way that shows up in results and is worth stating: the
tinygrad path's warp is compiled for a fixed camera geometry (1928x1208), so a
source of any other size has to be resampled into it first, while the ONNX path
builds its warp at the source's native resolution and resamples nothing.

*Frame convention.* openpilot predicts in its calibrated frame, x forward and
y RIGHT, z down. Paddock wants x forward, y LEFT. y is negated on the way out.

*Rate.* supercombo consumes 20 Hz and its feature buffer is 96 frames deep
(128 for the big model). A benchmark that supplies real history through
`Policy.infer_sequence` should supply at least that; `diagnostics()` reports how
many frames were actually seen so a short sequence is visible rather than
silently scored.

## The mounting-rotation guard

`CameraAdapter` derives openpilot's `rpyCalib` from `Observation.camera2ego`.
That derivation is correct and round-trips exactly -- but it is only as good as
the `camera2ego` it is handed, and a benchmark that passes `np.eye(4)`, reading
it as "no mounting rotation", is really saying "the camera's axes are the ego's
axes". In paddock's conventions (camera [right, down, forward], ego
[forward, left, up]) that is a 90 degree error, and it comes out as
`rpy = [-90, 0, +90] deg`: a sideways warp, the network shown a rotated world,
and a plan that predicts a stationary car.

That is not hypothetical. It is what the removed comma2k19 loader did, and it
produced a published ADE of 38.8 m that was read as a hardware generalisation
gap. Nothing failed; the number just looked like a finding.

So `MAX_MOUNT_DEG` now rejects it. A real windscreen mount is a few degrees off
level; anything past 15 is a coordinate-convention error, and this class of bug
must be loud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from paddock.adapters.models.openpilot.nv12 import pack_nv12, rgb_to_nv12_batch
from paddock.policy.protocol import Observation, Pinhole, Policy, Trajectory, constant_velocity

# The camera geometry openpilot's tinygrad JIT is compiled for.
TARGET_W, TARGET_H = 1928, 1208

# nuScenes/paddock ego frame is x-forward, y-left, z-up; openpilot's device and
# calibrated frames are x-forward, y-right, z-down.
DEVICE_FROM_EGO = np.diag([1.0, -1.0, -1.0])

# openpilot sets `traffic_convention[int(is_rhd)] = 1` (modeld.py), where is_rhd
# means right-hand *drive* -- steering wheel on the right, i.e. left-hand
# traffic. Every scene evaluated here is left-hand-drive, right-hand-traffic
# (neuroNCAP boston-seaport, KITScenes Germany, comma2k19 California), so index
# 0 is set.
TRAFFIC_CONVENTION = np.array([1.0, 0.0], dtype=np.float32)

#: The 0.5 s grid the closed-loop vehicle model expects.
GRID_S = np.arange(1, 9) * 0.5

FEATURE_BUFFER_FRAMES = 130  # >= the big model's 128-deep buffer
MODEL_HZ = 20.0  # the rate supercombo believes its input arrives at

#: A camera mounting further from level than this is a coordinate-convention
#: error, not a mounting. See the module docstring.
MAX_MOUNT_DEG = 15.0

#: Sources whose geometry openpilot already describes, so no resampling is
#: needed on the ONNX backend. Keyed by (width, height) -> focal length.
#: comma2k19's EON is `("neo", "unknown").narrow_road` in openpilot's own
#: `DEVICE_CAMERAS` table: 1164x874 at fl 910, which is also the medmodel
#: frame's focal length -- the model frame is a plain crop of an EON image.
NATIVE_CAMERAS = {(1164, 874): 910.0, (1928, 1208): 2648.0}

#: Emit openpilot's own longitudinal intent (desired accel + should_stop) so the
#: closed-loop client can drive it natively instead of through the TCP tracker's
#: brake-below-0.4 m/s rule. Off by default: the standard tracker path is what
#: every published Bench2Drive baseline uses.
NATIVE_LONG = os.environ.get("PADDOCK_OPENPILOT_NATIVE_LONG", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LONG_ACTION_T = 0.5  # the action_t[1] this adapter feeds the model
LONG_SMOOTH_SECONDS = 0.3  # openpilot modeld's longitudinal smoothing
#: Which openpilot longitudinal mode to reproduce (selfdriveState.experimentalMode):
#: "chill" = lead-follow + cruise only (default openpilot; does not stop for lights),
#: "experimental" = also the e2e plan candidate (stops for lights/hazards).
OP_MODE = os.environ.get("PADDOCK_OPENPILOT_MODE", "experimental").strip().lower()
# longitudinal_planner.py / drive_helpers.py / long_mpc.py constants (0.11.2)
ACCEL_MIN, ACCEL_MAX = -3.5, 2.0
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS = [0.0, 10.0, 25.0, 40.0], [1.6, 1.2, 0.8, 0.6]
J_CRUISE = 0.5  # m/s^3 cruise jerk limit (chill)
COMFORT_BRAKE, STOP_DISTANCE, T_FOLLOW = 2.5, 6.0, 1.45
K_ROOM, K_VREL = 0.3, 0.6  # closed-form stand-in for the lead MPC on its safe-distance model
# radard.RADAR_TO_CAMERA: model lead sits at the camera; the MPC uses the radar frame.
RADAR_TO_CAMERA = 1.52
V_CRUISE_FALLBACK = 13.9  # m/s when the source sends no road limit
#: Driver-resume emulation (experimental mode). Stock openpilot in experimental mode
#: holds after a full stop until the e2e plan asks to go; a driver in stop-and-go taps
#: resume/gas. In CARLA the e2e plan at standstill stays ~0 indefinitely (static
#: frames), so after this many seconds stopped WITH NO LEAD the e2e candidate is
#: dropped -- openpilot's own chill-mode candidates (cruise, lead) take over -- until
#: the car is rolling (>= 1 m/s), then experimental resumes control. 0 disables.
RESUME_S = float(os.environ.get("PADDOCK_OPENPILOT_RESUME_S", "3.0"))
RESUME_UNTIL_V = 1.0


class ImplausibleMounting(ValueError):
    """The derived camera mounting cannot be a real one."""


@dataclass(frozen=True)
class CameraFit:
    """How a source frame maps into openpilot's fixed 1928x1208 tinygrad input.

    Only used by the `tinygrad` backend, whose warp is compiled for one camera
    geometry. The `onnx` backend warps from the source resolution directly and
    skips all of this -- which matters for a small sensor: comma2k19's 1164x874
    EON was previously being upscaled 66 % to fill 1928x1208 and then warped
    back down, two resamples where zero are needed.

    Two strategies, chosen from the source's aspect:

    *Crop* when the source is taller than the target -- trimming height costs no
    horizontal field of view. KITScenes' 3504x2272 is this case.

    *Letterbox* when the source is wider. Cropping would trim width, and
    horizontal field of view is the scarce resource: the wide model view wants
    58.7 deg and a nuScenes camera has about 64 to give.
    """

    mode: str
    crop: tuple[int, int, int, int]
    scale: float
    scaled_size: tuple[int, int]
    pad_top: int
    intrinsics: np.ndarray

    @classmethod
    def compute(cls, pinhole: Pinhole) -> CameraFit:
        target_aspect = TARGET_W / TARGET_H
        if pinhole.width / pinhole.height <= target_aspect:
            mode, crop_w = "crop", pinhole.width
            crop_h = int(round(pinhole.width / target_aspect))
        else:
            mode, crop_w, crop_h = "letterbox", pinhole.width, pinhole.height

        # Anchored on the principal point, not the image centre -- they differ
        # by a few pixels and it is free to get right.
        x0 = int(round(np.clip(pinhole.cx - crop_w / 2, 0, pinhole.width - crop_w)))
        y0 = int(round(np.clip(pinhole.cy - crop_h / 2, 0, pinhole.height - crop_h)))
        scale = TARGET_W / crop_w
        scaled_h = int(round(crop_h * scale))
        # Crop sets crop_h = w/aspect, so scaled_h == TARGET_H exactly; letterbox
        # is taken only when w/h > aspect, which makes scaled_h strictly less.
        assert scaled_h <= TARGET_H, (
            f"{mode} of {pinhole.width}x{pinhole.height} produced {scaled_h} rows, "
            f"more than {TARGET_H} -- the branch invariant no longer holds"
        )
        pad_top = (TARGET_H - scaled_h) // 2
        return cls(
            mode=mode,
            crop=(x0, y0, x0 + crop_w, y0 + crop_h),
            scale=scale,
            scaled_size=(TARGET_W, scaled_h),
            pad_top=pad_top,
            intrinsics=np.array(
                [
                    [pinhole.fx * scale, 0.0, (pinhole.cx - x0) * scale],
                    [0.0, pinhole.fy * scale, (pinhole.cy - y0) * scale + pad_top],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    @classmethod
    def identity(cls, pinhole: Pinhole) -> CameraFit:
        """No resampling: the source is used as it arrived."""
        return cls(
            mode="native",
            crop=(0, 0, pinhole.width, pinhole.height),
            scale=1.0,
            scaled_size=(pinhole.width, pinhole.height),
            pad_top=0,
            intrinsics=np.array(
                [[pinhole.fx, 0.0, pinhole.cx], [0.0, pinhole.fy, pinhole.cy], [0.0, 0.0, 1.0]]
            ),
        )


class CameraAdapter:
    """A `CameraFit` plus the openpilot warp matrices it implies."""

    def __init__(self, pinhole: Pinhole, camera2ego: np.ndarray, resample: bool = True) -> None:
        self.src = pinhole
        self.fit = CameraFit.compute(pinhole) if resample else CameraFit.identity(pinhole)
        self.mode = self.fit.mode
        self.crop = self.fit.crop
        self.scaled_size = self.fit.scaled_size
        self.pad_top = self.fit.pad_top
        self.intrinsics = self.fit.intrinsics
        self.size = (TARGET_W, TARGET_H) if resample else (pinhole.width, pinhole.height)

        from openpilot.common.transformations.camera import view_frame_from_device_frame
        from openpilot.common.transformations.model import get_warp_matrix
        from openpilot.common.transformations.orientation import rot2euler

        # Mounting rotation: the residual rotation satisfying
        #   view_frame_from_device_frame @ rot_from_euler(rpy) == R_cam_from_device
        r_dev_cam = DEVICE_FROM_EGO @ np.asarray(camera2ego)[:3, :3]
        self.rpy = np.asarray(rot2euler(view_frame_from_device_frame.T @ r_dev_cam.T), dtype=np.float32)
        self._check_mounting(camera2ego)

        self.transforms = {
            "img": get_warp_matrix(self.rpy, self.intrinsics, False).astype(np.float32),
            "big_img": get_warp_matrix(self.rpy, self.intrinsics, True).astype(np.float32),
        }

    def _check_mounting(self, camera2ego: np.ndarray) -> None:
        deg = np.degrees(self.rpy)
        if np.all(np.abs(deg) <= MAX_MOUNT_DEG):
            return
        hint = ""
        if np.allclose(np.asarray(camera2ego)[:3, :3], np.eye(3), atol=1e-6):
            hint = (
                "\nThe camera2ego supplied was the identity. That is not 'no mounting "
                "rotation' -- paddock's camera frame is [right, down, forward] and its "
                "ego frame is [forward, left, up], so an unrotated camera is\n"
                "    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]\n"
                "and the identity means a 90 degree error."
            )
        raise ImplausibleMounting(
            f"derived camera mounting rpy = {np.round(deg, 2)} deg, beyond +-{MAX_MOUNT_DEG} deg. "
            f"A real mount is a few degrees off level, so this is a coordinate-convention "
            f"error in the benchmark's camera2ego rather than a mounting.{hint}"
        )

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Source frame -> the frame `intrinsics` describes."""
        if self.mode == "native":
            return image
        x0, y0, x1, y1 = self.crop
        resized = np.asarray(Image.fromarray(image[y0:y1, x0:x1]).resize(self.scaled_size, Image.BILINEAR))
        if self.scaled_size[1] == TARGET_H:
            return resized
        out = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        out[self.pad_top : self.pad_top + self.scaled_size[1]] = resized
        return out

    #: (hfov, vfov) each model view carves out of its source, degrees.
    VIEW_FOV = {"img": (31.4, 15.9), "big_img": (58.7, 31.3)}

    def margin_deg(self, slot: str = "big_img") -> dict[str, float]:
        """How much field of view is left over after the model view is carved out.

        `slot` is the view this source feeds: "big_img" (the default, and the
        tighter one when a single frame feeds both views) or "img" for a
        dedicated narrow road camera. Negative means the warp samples outside
        what the camera saw -- black wedges at the edges of the model's input.
        """
        h, v = self.VIEW_FOV[slot]
        return {"horizontal": self.src.hfov_deg - h, "vertical": self.src.vfov_deg - v}


def plan_to_grid(plan: np.ndarray, grid_s: np.ndarray = GRID_S) -> np.ndarray:
    """(33, 15) plan -> (N, 2) waypoints on `grid_s`, y flipped to left."""
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

    pos = plan[:, Plan.POSITION]
    t = np.asarray(ModelConstants.T_IDXS)
    return np.stack([np.interp(grid_s, t, pos[:, 0]), -np.interp(grid_s, t, pos[:, 1])], axis=-1)


class OpenpilotPolicy(Policy):
    #: comma-three rig: the narrow road camera feeds `img`, the wide camera feeds
    #: `big_img` -- as on the device. A source that only has CAM_FRONT still works
    #: (both views cut from it), which is what NeuroNCAP/NAVSIM/KITScenes send.
    cameras = (
        "CAM_OP_ROAD",
        "CAM_OP_WIDE",
        "CAM_FRONT",
    )  # CAM_FRONT: single-camera fallback for open-loop sources
    waypoint_dt = 0.5

    def __init__(self, backend: str | None = None, model: str | None = None) -> None:
        #: "tinygrad" (openpilot's own JIT) or "onnx". Auto-detected in `load`.
        self.backend = backend or os.environ.get("PADDOCK_OPENPILOT_BACKEND", "auto")
        #: "small" (driving_supercombo) or "big" (big_driving_supercombo).
        self.model_name = model or os.environ.get("PADDOCK_OPENPILOT_MODEL", "small")
        self.model = None
        self.session = None
        self.camera: CameraAdapter | None = None
        self.cam_road: CameraAdapter | None = None
        self.cam_wide: CameraAdapter | None = None
        self.buf_wide: np.ndarray | None = None
        self.frames_seen = 0
        self.prev_timestamp: int | None = None
        #: Seconds between the last two observations, measured in `_repeats`.
        self.dt_s = 1.0 / MODEL_HZ
        self.prev_accel = 0.0
        self.a_cruise_prev = 0.0
        self.still_s = 0.0
        self.resuming = False
        self.nv12_info = None
        self.buf: np.ndarray | None = None
        self.bufs: dict[str, np.ndarray] = {}
        self._last_diag: dict[str, Any] = {}

    # ---- loading --------------------------------------------------------

    def _models_dir(self) -> Path:
        from openpilot.selfdrive.modeld import models as _models

        return Path(_models.__file__).parent

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            from openpilot.selfdrive.modeld.helpers import chestnut_compiled  # noqa: F401
            from openpilot.selfdrive.modeld.modeld import ModelState  # noqa: F401

            return "tinygrad"
        except Exception:
            # No tinygrad, no compiled pickle, or no GPU to compile against.
            # The ONNX files ship in the repo either way.
            return "onnx"

    def load(self) -> None:
        # Imported here, not at module scope: openpilot's package only exists
        # inside its own environment, and this module is imported by
        # `paddock list` and the test suite, which run in paddock's.
        self.backend = self._resolve_backend()
        if self.backend == "tinygrad":
            from openpilot.selfdrive.modeld.modeld import ModelState
            from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

            self.model = ModelState(TARGET_W, TARGET_H, usbgpu=False)
            # ModelState caches its device tensors by buffer address, so these
            # arrays must be allocated once and written in place.
            self.nv12_info = get_nv12_info(TARGET_W, TARGET_H)
            self.buf = np.zeros(self.nv12_info[3], dtype=np.uint8)
            self.buf_wide = np.zeros(self.nv12_info[3], dtype=np.uint8)
            # One source: both slots read the same buffer. Two sources: each
            # slot gets its own, rebound in `_build_camera`.
            self.bufs = {"img": self.buf, "big_img": self.buf}
        else:
            from paddock.adapters.models.openpilot.supercombo import SupercomboSession

            self.session = SupercomboSession(self._onnx_path(), self._threads())
            self._new_onnx_runner()

    def _onnx_path(self) -> Path:
        override = os.environ.get("PADDOCK_OPENPILOT_ONNX")
        if override:
            return Path(override)
        prefix = "big_" if self.model_name == "big" else ""
        return self._models_dir() / f"{prefix}driving_supercombo.onnx"

    @staticmethod
    def _threads() -> int:
        return int(os.environ.get("PADDOCK_OPENPILOT_THREADS", "8"))

    def _new_onnx_runner(self) -> None:
        from paddock.adapters.models.openpilot.supercombo import SupercomboRunner

        self.model = SupercomboRunner(session=self.session, traffic_convention=TRAFFIC_CONVENTION)

    # ---- episode state ---------------------------------------------------

    def reset(self) -> None:
        """Drop all temporal state.

        supercombo carries a feature buffer and a previous-desire term across
        frames; without this, run N+1 starts with run N's memory and the runs
        stop being independent samples.

        The camera adapter is episode state too, and forgetting that was a real
        bug: it is built from the first frame's calibration and was surviving
        across samples, so every KITScenes scene after the first was warped with
        the first scene's mounting rotation.
        """
        if self.backend == "tinygrad":
            from openpilot.selfdrive.modeld.modeld import ModelState

            self.model = ModelState(TARGET_W, TARGET_H, usbgpu=False)
        elif self.session is not None:
            self._new_onnx_runner()
        self.camera = None
        self.cam_road = None
        self.cam_wide = None
        if self.buf is not None:
            # Back to one source feeding both views; `_infer_tinygrad` rebinds
            # "big_img" to buf_wide again if the next sample has two cameras.
            self.bufs = {"img": self.buf, "big_img": self.buf}
        self.frames_seen = 0
        self.prev_timestamp = None
        self.dt_s = 1.0 / MODEL_HZ
        self.prev_accel = 0.0
        self.a_cruise_prev = 0.0
        self.still_s = 0.0
        self.resuming = False

    # ---- inference -------------------------------------------------------

    def _build_camera(self, obs: Observation) -> CameraAdapter:
        pinhole = obs.pinhole("CAM_FRONT")
        # The ONNX warp is built per source geometry, so nothing needs
        # resampling. The tinygrad warp is compiled for 1928x1208 and does.
        resample = self.backend == "tinygrad"
        return CameraAdapter(pinhole, obs.camera2ego["CAM_FRONT"], resample=resample)

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")
        return self._infer_tinygrad(obs) if self.backend == "tinygrad" else self._infer_onnx(obs)

    def _repeats(self, obs: Observation) -> int:
        """How many times to show one rendered frame.

        supercombo believes it is fed 20 Hz. A simulator advancing 0.5 s per
        request would otherwise tell the model that half a second of motion
        happened in 1/20 s, and it would read the ego as travelling ten times
        its actual speed. Repeating each frame `dt * 20` times keeps the average
        rate honest. It is still jerky -- the world holds still and then jumps --
        which is the argument for running the simulation at 0.1 s, where this
        lands on the 2 repeats the KITScenes run used.

        The step is measured rather than assumed, so this follows whatever time
        step the run was launched with. comma2k19 is natively 20 Hz, so there it
        comes out to exactly 1 throughout and no repetition happens.

        Also records `self.dt_s`, the measured step, which the native
        longitudinal planner integrates over.
        """
        repeats = 1
        self.dt_s = 1.0 / MODEL_HZ
        if self.prev_timestamp is not None:
            dt = (obs.timestamp_us - self.prev_timestamp) / 1e6
            if dt > 0:
                self.dt_s = dt
                repeats = max(1, round(dt * MODEL_HZ))
        if self.frames_seen == 0 and self.self_prime:
            # No history supplied, so manufacture one deep enough to fill the
            # feature buffer. `infer_sequence` clears `self_prime`, so this only
            # happens in closed loop.
            repeats = FEATURE_BUFFER_FRAMES
        self.prev_timestamp = obs.timestamp_us
        return repeats

    def _infer_tinygrad(self, obs: Observation) -> Trajectory:
        from openpilot.selfdrive.modeld.constants import ModelConstants

        stride, y_height = self.nv12_info[0], self.nv12_info[1]
        have = set(obs.images) | set(obs.image_paths)
        # A comma rig has two cameras; every dataset in this repo but Bench2Drive
        # has one, and then the single frame feeds both model views.
        two = "CAM_OP_ROAD" in have and "CAM_OP_WIDE" in have
        if two:
            if self.cam_road is None:
                self.cam_road = CameraAdapter(obs.pinhole("CAM_OP_ROAD"), obs.camera2ego["CAM_OP_ROAD"])
                self.cam_wide = CameraAdapter(obs.pinhole("CAM_OP_WIDE"), obs.camera2ego["CAM_OP_WIDE"])
                self.camera = self.cam_road
                self.bufs = {"img": self.buf, "big_img": self.buf_wide}
            packs = [
                (
                    self.buf,
                    rgb_to_nv12_batch(
                        self.cam_road.prepare(obs.image("CAM_OP_ROAD"))[None], TARGET_W, TARGET_H
                    )[0],
                ),
                (
                    self.buf_wide,
                    rgb_to_nv12_batch(
                        self.cam_wide.prepare(obs.image("CAM_OP_WIDE"))[None], TARGET_W, TARGET_H
                    )[0],
                ),
            ]
            transforms = {
                "img": self.cam_road.transforms["img"],
                "big_img": self.cam_wide.transforms["big_img"],
            }
        else:
            if self.camera is None:
                self.camera = self._build_camera(obs)
                self.bufs = {"img": self.buf, "big_img": self.buf}
            packs = [
                (
                    self.buf,
                    rgb_to_nv12_batch(self.camera.prepare(obs.image("CAM_FRONT"))[None], TARGET_W, TARGET_H)[
                        0
                    ],
                )
            ]
            transforms = self.camera.transforms

        inputs = {
            "desire_pulse": np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
            "traffic_convention": TRAFFIC_CONVENTION,
            "action_t": np.array([0.3, 0.5], dtype=np.float32),
        }
        out = None
        for _ in range(self._repeats(obs)):
            for buf, packed in packs:
                pack_nv12(buf, packed, TARGET_W, TARGET_H, stride, y_height)
            out = self.model.run(self.bufs, transforms, inputs)
        self.frames_seen += 1
        if out is None or "plan" not in out:
            return constant_velocity(obs.speed, dt=self.waypoint_dt)
        plan33 = np.asarray(out["plan"])[0]
        traj = Trajectory(plan_to_grid(plan33), dt=self.waypoint_dt)
        if NATIVE_LONG:
            traj.extras = {
                "native_long": self._native_long(out, plan33, obs.speed, self.dt_s, obs.speed_limit)
            }
        return traj

    def _infer_onnx(self, obs: Observation) -> Trajectory:
        from paddock.adapters.models.openpilot.supercombo import WarpPlan, rgb_to_yuv420

        if self.camera is None:
            self.camera = self._build_camera(obs)
        if not hasattr(self, "_plans") or self._plans_for is not self.camera:
            src = self.camera.size
            model_size = (
                self.session.spec.input_shapes["img"][3] * 2,
                self.session.spec.input_shapes["img"][2] * 2,
            )
            self._plans = {
                k: WarpPlan.build(self.camera.transforms[k], model_size, src) for k in ("img", "big_img")
            }
            self._plans_for = self.camera

        rgb = self.camera.prepare(obs.image("CAM_FRONT"))
        y, u, v = rgb_to_yuv420(rgb)
        img = self._plans["img"].apply(y, u, v)
        big = self._plans["big_img"].apply(y, u, v)

        out = None
        for _ in range(self._repeats(obs)):
            out = self.model.run(img, big, max(float(obs.speed), 0.0))
        self.frames_seen += 1
        if out is None or "plan" not in out:
            return constant_velocity(obs.speed, dt=self.waypoint_dt)
        self._last_diag = {
            "desiredCurvature": out["action_value"]["desiredCurvature"],
            "desiredAcceleration": out["action_value"]["desiredAcceleration"],
            "pose_trans": np.asarray(out["pose"][0][:3]).tolist(),
        }
        plan33 = np.asarray(out["plan"])[0]
        traj = Trajectory(plan_to_grid(plan33), dt=self.waypoint_dt)
        if NATIVE_LONG:
            traj.extras = {
                "native_long": self._native_long(out, plan33, obs.speed, self.dt_s, obs.speed_limit)
            }
        return traj

    def _native_long(self, out, plan33, v_ego: float, dt_s: float, speed_limit) -> dict:
        """openpilot 0.11 longitudinal planner, reproduced (longitudinal_planner.py:128-149):
        desired accel = MIN of (lead-follow, cruise-to-set-speed, and in experimental
        mode the e2e plan); should_stop = ANY candidate says stop. The cruise set-speed
        is the posted road limit the simulator sends. Lead-follow uses the model's own
        lead / lead_prob (current-time hypothesis) on the MPC's safe-distance model."""
        from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, should_stop
        from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

        e2e = OP_MODE == "experimental"
        v_cruise = float(speed_limit) if speed_limit else V_CRUISE_FALLBACK
        max_accel = ACCEL_MAX if e2e else float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))
        # cruise candidate (get_cruise_accel)
        a_cruise = float(np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel))
        if not e2e:
            a_cruise = float(
                np.clip(a_cruise, self.a_cruise_prev - J_CRUISE * dt_s, self.a_cruise_prev + J_CRUISE * dt_s)
            )
        self.a_cruise_prev = a_cruise
        cands = [(a_cruise, "cruise", bool(should_stop(v_ego, a_cruise)))]
        # lead candidate (radard uses the model lead above prob 0.5; MPC keeps x_lead >= safe distance)
        lead = np.asarray(out["lead"])[0, 0]
        prob = float(np.asarray(out["lead_prob"])[0, 0])
        if prob > 0.5:
            # radard.get_RadarState_from_vision (radard.py:136-143): the model's
            # lead velocity is relative to the model's OWN ego-speed estimate, so
            # recover the absolute lead speed in the measured-ego frame before
            # comparing (vLead = v_ego + lead.v[0] - model_v_ego). The lead sits
            # at the camera; the MPC wants the radar-frame distance.
            model_v_ego = float(plan33[:, Plan.VELOCITY][0, 0])
            x_lead = float(lead[0, 0]) - RADAR_TO_CAMERA
            v_lead = v_ego + (float(lead[0, 2]) - model_v_ego)
            safe = v_ego**2 / (2 * COMFORT_BRAKE) + T_FOLLOW * v_ego + STOP_DISTANCE
            room = x_lead + v_lead**2 / (2 * COMFORT_BRAKE) - safe
            a_lead = float(np.clip(K_ROOM * room + K_VREL * (v_lead - v_ego), ACCEL_MIN, max_accel))
            cands.append((a_lead, "lead", bool(should_stop(v_ego, a_lead))))
        # e2e candidate (modeld get_action_from_model, plan branch, 0.3 s smoothed)
        a_raw = float(
            get_accel_from_plan(
                plan33[:, Plan.VELOCITY][:, 0],
                plan33[:, Plan.ACCELERATION][:, 0],
                ModelConstants.T_IDXS,
                action_t=LONG_ACTION_T,
            )
        )
        alpha = dt_s / (LONG_SMOOTH_SECONDS + dt_s) if dt_s > 0 else 1.0
        a_e2e = self.prev_accel + alpha * (a_raw - self.prev_accel)
        self.prev_accel = a_e2e
        # driver-resume emulation (see RESUME_S)
        self.still_s = self.still_s + dt_s if v_ego < 0.3 else 0.0
        if v_ego >= RESUME_UNTIL_V:
            self.resuming = False
        if e2e and RESUME_S > 0 and not self.resuming and self.still_s >= RESUME_S and prob < 0.5:
            self.resuming = True
        if e2e and not self.resuming:
            cands.append((a_e2e, "e2e", bool(should_stop(v_ego, a_e2e))))
        a_target, source, _ = min(cands, key=lambda c: c[0])
        if self.resuming:
            source = source + "+resume"
        return {
            "accel": a_target,
            "v_target": max(0.0, v_ego + a_target * LONG_ACTION_T),
            "should_stop": any(c[2] for c in cands),
            "source": source,
            "mode": OP_MODE,
            "a_cruise": a_cruise,
            "a_e2e": a_e2e,
            "lead_prob": prob,
            "v_cruise": v_cruise,
        }

    # ---- diagnostics -----------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """The checks that catch silent misconfiguration.

        `warp_in_bounds` is the important one on the ONNX backend: it is the
        measured fraction of the model's input that came from real pixels. Less
        than 1.0 means border smearing, which no score will show.
        """
        if self.camera is None:
            return {"status": "no frame seen yet", "backend": self.backend}
        # Each view's margin from the camera that actually feeds it: in the
        # two-camera rig the road cam feeds "img" and the wide cam "big_img", so
        # reading both off one camera reports a false deficit for the other. A
        # single-camera source feeds both views from the one frame.
        slot_cam = (
            {"img": self.cam_road, "big_img": self.cam_wide}
            if self.cam_wide is not None
            else {"img": self.camera, "big_img": self.camera}
        )
        diag: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model_name,
            "source_size": [self.camera.src.width, self.camera.src.height],
            "resample_mode": self.camera.mode,
            "source_hfov_deg": round(self.camera.src.hfov_deg, 2),
            "source_vfov_deg": round(self.camera.src.vfov_deg, 2),
            "margin_deg": {
                slot: {k: round(v, 2) for k, v in cam.margin_deg(slot).items()}
                for slot, cam in slot_cam.items()
            },
            "rpy_deg": [round(float(v), 4) for v in np.degrees(self.camera.rpy)],
            "frames_seen": self.frames_seen,
            "feature_buffer_frames": FEATURE_BUFFER_FRAMES,
        }
        if self.session is not None:
            diag["onnx"] = str(self.session.onnx_path)
            diag["providers"] = self.session.providers
            diag["model_checkpoint"] = self.session.spec.model_checkpoint
            if getattr(self, "_plans", None):
                diag["warp_in_bounds"] = {k: round(p.in_bounds_frac, 5) for k, p in self._plans.items()}
            if self.model is not None:
                diag["warmup_frames"] = self.model.warmup_frames
                diag["warmed_up"] = self.frames_seen >= self.model.warmup_frames
        diag.update(self._last_diag)
        return diag


POLICY = OpenpilotPolicy
