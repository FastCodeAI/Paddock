"""openpilot supercombo as a paddock Policy.

Adapted from the NeuroNCAP model node (`servers/openpilot_server.py`). The
model-specific reasoning below is unchanged from that node; what changed is the
seams -- it now implements `Policy`, so the same class serves open-loop scoring
and closed-loop simulation instead of one implementation per loop type.

What this adapts, and why each piece is load-bearing:

*Camera.*  The network never sees a raw frame -- it sees two warped 512x256
views, roughly 31.4 x 15.9 deg (``img``) and 58.7 x 31.3 deg (``big_img``).
nuScenes' front camera is about 64 x 39 deg, so both fit inside it and no field
of view has to be invented.  The margin on the wide view is thin (single
degrees), so the source is letterboxed rather than cropped to openpilot's
aspect: cropping to 1928x1208's aspect would throw away ~5 deg of width and put
the wide view right at the edge of what the sensor actually saw.  `diagnostics()`
reports the real margin once the first frame has arrived.

*Rate.*  supercombo consumes 20 Hz and needs ~96 frames to fill its feature
buffer.  A simulator stepping at 0.5 s would give it 2 Hz and about 20 frames
per run -- the model would perceive a world at a tenth speed.  So the simulation
is run at 0.1 s and every rendered frame is emitted twice.  Warm-up frames come
from the scenario's open-loop priming steps, repeated to fill the buffer on the
first inference.

*Frame convention.*  openpilot predicts in its device frame, x forward and y
RIGHT.  Paddock and NeuroNCAP want y LEFT.  y is negated on the way out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from paddock.adapters.models.openpilot.nv12 import pack_nv12, rgb_to_nv12_batch
from paddock.policy.protocol import Observation, Pinhole, Policy, Trajectory, constant_velocity

# ModelState compiles its warp for a fixed set of camera geometries; 1928x1208
# is the tici one and the only sensible target here.
TARGET_W, TARGET_H = 1928, 1208

# nuScenes ego frame is x-forward, y-left, z-up; openpilot's device frame is
# x-forward, y-right, z-down.
DEVICE_FROM_EGO = np.diag([1.0, -1.0, -1.0])

# openpilot sets this as `traffic_convention[int(is_rhd)] = 1` (modeld.py), where
# is_rhd means right-hand *drive* -- steering wheel on the right, i.e. left-hand
# traffic (UK, Japan, Singapore).
#
# The scenes evaluated here are left-hand-drive, right-hand-traffic: the
# neuroNCAP sequences (0103, 0108) are boston-seaport, and KITScenes is Germany.
# So is_rhd is False and index 0 is set.
#
# This was [0, 1] in the closed-loop node -- telling the model it was driving a
# right-hand-drive car in Boston -- while the open-loop KITScenes harness used
# [1, 0]. Corrected to agree, which is why closed-loop openpilot scores recorded
# before 2026-09-01 are not comparable with later ones.
TRAFFIC_CONVENTION = np.array([1.0, 0.0], dtype=np.float32)

# The 0.5 s grid the vehicle model expects, regardless of sim step.
GRID_S = np.arange(1, 9) * 0.5

FEATURE_BUFFER_FRAMES = 100  # >= 96, the depth needed to fill features_buffer
MODEL_HZ = 20.0  # the rate supercombo believes its input arrives at


class CameraAdapter:
    """Maps a rendered nuScenes camera into the frame openpilot expects."""

    def __init__(self, pinhole: Pinhole, camera2ego: np.ndarray) -> None:
        self.src = pinhole
        # Letterbox: scale so the full width survives, then pad height. Keeping
        # the whole width is the point -- the wide model view needs nearly all
        # of the sensor's horizontal field of view.
        #
        # Checked before importing openpilot: it is pure arithmetic, and a bad
        # source should be reported as a bad source rather than surfacing only
        # on a machine that happens to have the model installed.
        scale = TARGET_W / pinhole.width
        scaled_h = int(round(pinhole.height * scale))
        if scaled_h > TARGET_H:
            raise ValueError(
                f"source {pinhole.width}x{pinhole.height} is too tall to letterbox into "
                f"{TARGET_W}x{TARGET_H} without losing vertical field of view"
            )

        from openpilot.common.transformations.camera import view_frame_from_device_frame
        from openpilot.common.transformations.model import get_warp_matrix
        from openpilot.common.transformations.orientation import rot2euler

        self.scale = scale
        self.scaled_size = (TARGET_W, scaled_h)
        self.pad_top = (TARGET_H - scaled_h) // 2
        self.intrinsics = np.array(
            [
                [pinhole.fx * scale, 0.0, pinhole.cx * scale],
                [0.0, pinhole.fy * scale, pinhole.cy * scale + self.pad_top],
                [0.0, 0.0, 1.0],
            ]
        )
        # Mounting rotation: the residual rotation satisfying
        #   view_frame_from_device_frame @ rot_from_euler(rpy) == R_cam_from_device
        r_ego_cam = camera2ego[:3, :3]
        r_dev_cam = DEVICE_FROM_EGO @ r_ego_cam
        self.rpy = np.asarray(rot2euler(view_frame_from_device_frame.T @ r_dev_cam.T), dtype=np.float32)
        self.transforms = {
            "img": get_warp_matrix(self.rpy, self.intrinsics, False).astype(np.float32),
            "big_img": get_warp_matrix(self.rpy, self.intrinsics, True).astype(np.float32),
        }

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Rendered frame -> the TARGET_W x TARGET_H frame `intrinsics` describes."""
        resized = np.asarray(Image.fromarray(image).resize(self.scaled_size, Image.BILINEAR))
        out = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        out[self.pad_top : self.pad_top + self.scaled_size[1]] = resized
        return out

    def margin_deg(self) -> dict[str, float]:
        """How much field of view is left over after each model view is carved out.

        Negative means the warp is sampling outside what the camera saw, which
        shows up as black wedges at the edges of the model's input.
        """
        return {
            "horizontal": self.src.hfov_deg - 58.7,
            "vertical": self.src.vfov_deg - 31.3,
        }


def plan_to_grid(plan: np.ndarray) -> np.ndarray:
    """(33, 15) plan -> (8, 2) waypoints on the 0.5 s grid, y flipped to left."""
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

    pos = plan[:, Plan.POSITION]
    t = np.asarray(ModelConstants.T_IDXS)
    x = np.interp(GRID_S, t, pos[:, 0])
    y = np.interp(GRID_S, t, pos[:, 1])
    return np.stack([x, -y], axis=-1)


class OpenpilotPolicy(Policy):
    cameras = ("CAM_FRONT",)
    waypoint_dt = 0.5

    def __init__(self) -> None:
        self.model = None
        self.camera: CameraAdapter | None = None
        self.frames_seen = 0
        self.prev_timestamp: int | None = None
        self.nv12_info = None
        self.buf: np.ndarray | None = None
        self.bufs: dict[str, np.ndarray] = {}

    def load(self) -> None:
        # Imported here, not at module scope: openpilot's package only exists
        # inside its own environment, and this module is imported by
        # `paddock list` and the test suite, which run in paddock's.
        from openpilot.selfdrive.modeld.modeld import ModelState
        from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

        self.model = ModelState(TARGET_W, TARGET_H, usbgpu=False)
        # ModelState caches its device tensors by buffer address, so this array
        # must be allocated once and written in place -- never reallocated.
        self.nv12_info = get_nv12_info(TARGET_W, TARGET_H)
        self.buf = np.zeros(self.nv12_info[3], dtype=np.uint8)
        self.bufs = {"img": self.buf, "big_img": self.buf}

    def reset(self) -> None:
        """Drop all temporal state.

        supercombo carries a 24-deep feature buffer and a previous-desire term
        across frames.  Without this, run N+1 starts with run N's memory and the
        runs stop being independent samples.  The pixel buffer is deliberately
        kept: it is reused in place, and reallocating it would invalidate
        ModelState's address-keyed tensor cache.
        """
        from openpilot.selfdrive.modeld.modeld import ModelState

        self.model = ModelState(TARGET_W, TARGET_H, usbgpu=False)
        self.frames_seen = 0
        self.prev_timestamp = None

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")

        from openpilot.selfdrive.modeld.constants import ModelConstants

        if self.camera is None:
            self.camera = CameraAdapter(obs.pinhole("CAM_FRONT"), obs.camera2ego["CAM_FRONT"])

        prepared = self.camera.prepare(obs.images["CAM_FRONT"])
        packed = rgb_to_nv12_batch(prepared[None], TARGET_W, TARGET_H)[0]
        stride, y_height = self.nv12_info[0], self.nv12_info[1]

        inputs = {
            "desire_pulse": np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
            "traffic_convention": TRAFFIC_CONVENTION,
            "action_t": np.array([0.3, 0.5], dtype=np.float32),
        }

        # How many times to show each rendered frame, so that the model's own
        # clock matches the simulator's.
        #
        # supercombo believes it is fed 20 Hz. The simulator advances the world
        # by its scenario time step per request -- 0.5 s by default. Emitting one
        # model frame per rendered frame would tell the model that 0.5 s of
        # motion happened in 1/20 s, and it would read the ego as travelling ten
        # times its actual speed. Repeating each frame `dt * 20` times keeps the
        # average rate honest. It is still jerky -- the world holds still and
        # then jumps -- which is the argument for running the simulation at
        # 0.1 s, where this lands on the 2 repeats the KITScenes run used.
        #
        # The step is measured rather than assumed, so this follows whatever
        # time step the run was launched with.
        repeats = 2
        if self.prev_timestamp is not None:
            dt = (obs.timestamp_us - self.prev_timestamp) / 1e6
            if dt > 0:
                repeats = max(1, round(dt * MODEL_HZ))
        if self.frames_seen == 0:
            repeats = FEATURE_BUFFER_FRAMES
        self.prev_timestamp = obs.timestamp_us

        out = None
        for _ in range(repeats):
            pack_nv12(self.buf, packed, TARGET_W, TARGET_H, stride, y_height)
            out = self.model.run(self.bufs, self.camera.transforms, inputs)
        self.frames_seen += 1

        if out is None or "plan" not in out:
            return constant_velocity(obs.speed, dt=self.waypoint_dt)
        return Trajectory(plan_to_grid(np.asarray(out["plan"])[0]), dt=self.waypoint_dt)

    def diagnostics(self) -> dict[str, Any]:
        """Field-of-view margin, once a first frame has established the camera.

        A negative margin means the warp samples outside what the camera saw --
        black wedges at the edges of the model's input, invisible in the score.
        """
        if self.camera is None:
            return {"status": "no frame seen yet"}
        return {
            "source_hfov_deg": round(self.camera.src.hfov_deg, 2),
            "source_vfov_deg": round(self.camera.src.vfov_deg, 2),
            "margin_deg": {k: round(v, 2) for k, v in self.camera.margin_deg().items()},
            "rpy": self.camera.rpy.tolist(),
            "frames_seen": self.frames_seen,
        }


POLICY = OpenpilotPolicy
