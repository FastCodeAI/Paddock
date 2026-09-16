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

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CameraFit:
    """How a source frame maps into openpilot's fixed 1928x1208 input.

    Pure geometry, deliberately separate from `CameraAdapter`: it needs no
    openpilot import, so it can be reasoned about and unit-tested in paddock's
    own environment. Getting this wrong changes what the model sees without
    anything failing.

    Two strategies, chosen from the source's aspect rather than hardcoded:

    *Crop* when the source is taller than the target. Trimming height costs no
    horizontal field of view and the model sees only real pixels. KITScenes'
    3504x2272 is this case.

    *Letterbox* when the source is wider. Cropping would trim width, and
    horizontal field of view is the scarce resource -- the wide model view wants
    58.7 deg and a nuScenes camera has about 64 to give. Pad height instead.
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

        # Anchored on the principal point, not the image centre -- they differ by
        # a few pixels and it is free to get right.
        x0 = int(round(np.clip(pinhole.cx - crop_w / 2, 0, pinhole.width - crop_w)))
        y0 = int(round(np.clip(pinhole.cy - crop_h / 2, 0, pinhole.height - crop_h)))

        scale = TARGET_W / crop_w
        scaled_h = int(round(crop_h * scale))
        # Neither branch can overflow, and it is worth knowing why rather than
        # carrying a check that looks load-bearing. Crop sets crop_h = w/aspect,
        # so scaled_h = TARGET_W/aspect = TARGET_H exactly. Letterbox is taken
        # only when w/h > aspect, which makes scaled_h strictly less than that.
        # Kept as an assertion so a future change to the branching is caught.
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


class CameraAdapter:
    """A `CameraFit` plus the openpilot warp matrices it implies."""

    #: Horizontal field of view openpilot's wide model view consumes, degrees.
    WIDE_HFOV_DEG = 58.7

    def __init__(self, pinhole: Pinhole, camera2ego: np.ndarray) -> None:
        self.src = pinhole
        self.fit = CameraFit.compute(pinhole)
        self.mode = self.fit.mode
        self.crop = self.fit.crop
        self.scaled_size = self.fit.scaled_size
        self.pad_top = self.fit.pad_top
        self.intrinsics = self.fit.intrinsics

        from openpilot.common.transformations.camera import view_frame_from_device_frame
        from openpilot.common.transformations.model import get_warp_matrix
        from openpilot.common.transformations.orientation import rot2euler

        # Mounting rotation: the residual rotation satisfying
        #   view_frame_from_device_frame @ rot_from_euler(rpy) == R_cam_from_device
        r_dev_cam = DEVICE_FROM_EGO @ camera2ego[:3, :3]
        self.rpy = np.asarray(rot2euler(view_frame_from_device_frame.T @ r_dev_cam.T), dtype=np.float32)
        self.transforms = {
            "img": get_warp_matrix(self.rpy, self.intrinsics, False).astype(np.float32),
            "big_img": get_warp_matrix(self.rpy, self.intrinsics, True).astype(np.float32),
        }

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Source frame -> the TARGET_W x TARGET_H frame `intrinsics` describes."""
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


def plan_to_grid(plan: np.ndarray) -> np.ndarray:
    """(33, 15) plan -> (8, 2) waypoints on the 0.5 s grid, y flipped to left."""
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

    pos = plan[:, Plan.POSITION]
    t = np.asarray(ModelConstants.T_IDXS)
    x = np.interp(GRID_S, t, pos[:, 0])
    y = np.interp(GRID_S, t, pos[:, 1])
    return np.stack([x, -y], axis=-1)


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

    def __init__(self) -> None:
        self.model = None
        self.camera: CameraAdapter | None = None
        self.cam_road: CameraAdapter | None = None
        self.cam_wide: CameraAdapter | None = None
        self.buf_wide: np.ndarray | None = None
        self.frames_seen = 0
        self.prev_timestamp: int | None = None
        self.prev_accel = 0.0
        self.a_cruise_prev = 0.0
        self.still_s = 0.0
        self.resuming = False
        self.nv12_info = None
        self.buf: np.ndarray | None = None
        self.bufs: dict[str, np.ndarray] = {}

    def _new_model(self):
        """Load the default artifact or an explicitly compiled CUDA artifact.

        Upstream selects its artifact and queue devices through module helpers.
        Override those only during construction; each sample still gets fresh
        temporal state. The policy node constructs models serially.
        """
        from pathlib import Path
        from unittest.mock import patch

        import openpilot.selfdrive.modeld.modeld as modeld

        artifact = os.environ.get("PADDOCK_OPENPILOT_CUDA_MODEL")
        if not artifact:
            return modeld.ModelState(TARGET_W, TARGET_H, usbgpu=False)
        path = Path(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"Openpilot CUDA artifact missing: {path}")
        with (
            patch.object(modeld, "modeld_pkl_path", return_value=path),
            patch.object(
                modeld, "get_tg_input_devices", return_value={"WARP_DEV": "CPU", "QUEUE_DEV": "CUDA"}
            ),
        ):
            return modeld.ModelState(TARGET_W, TARGET_H, usbgpu=False)

    def load(self) -> None:
        # Imported here, not at module scope: openpilot's package only exists
        # inside its own environment, and this module is imported by
        # `paddock list` and the test suite, which run in paddock's.
        from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

        self.model = self._new_model()
        import inspect

        self._run_takes_prepare_only = "prepare_only" in inspect.signature(self.model.run).parameters
        # ModelState caches its device tensors by buffer address, so this array
        # must be allocated once and written in place -- never reallocated.
        self.nv12_info = get_nv12_info(TARGET_W, TARGET_H)
        self.buf = np.zeros(self.nv12_info[3], dtype=np.uint8)
        self.buf_wide = np.zeros(self.nv12_info[3], dtype=np.uint8)
        # one source: both slots read the same buffer; two sources: each slot its own
        self.bufs = {"img": self.buf, "big_img": self.buf}

    def reset(self) -> None:
        """Drop all temporal state.

        supercombo carries a 24-deep feature buffer and a previous-desire term
        across frames.  Without this, run N+1 starts with run N's memory and the
        runs stop being independent samples.  The pixel buffer is deliberately
        kept: it is reused in place, and reallocating it would invalidate
        ModelState's address-keyed tensor cache.

        The camera adapter is episode state too, and forgetting that was a real
        bug: it is built from the first frame's calibration and was surviving
        across samples, so every KITScenes scene after the first was warped with
        the first scene's mounting rotation. KITScenes rigs are recalibrated
        between sessions -- 3 distinct front-camera rotations across 60 val
        scenes, differing by up to 1 degree -- so roughly a quarter of samples
        were being fed a subtly wrong view, with nothing failing.
        """
        self.model = self._new_model()
        import inspect

        self._run_takes_prepare_only = "prepare_only" in inspect.signature(self.model.run).parameters
        self.camera = None
        self.cam_road = None
        self.cam_wide = None
        self.bufs = {"img": self.buf, "big_img": self.buf}
        self.frames_seen = 0
        self.prev_timestamp = None
        self.prev_accel = 0.0
        self.a_cruise_prev = 0.0
        self.still_s = 0.0
        self.resuming = False

    def infer(self, obs: Observation) -> Trajectory:
        if self.model is None:
            raise RuntimeError("load() must be called before infer()")

        from openpilot.selfdrive.modeld.constants import ModelConstants

        stride, y_height = self.nv12_info[0], self.nv12_info[1]
        have = set(obs.images) | set(obs.image_paths)
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
                self.camera = CameraAdapter(obs.pinhole("CAM_FRONT"), obs.camera2ego["CAM_FRONT"])
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
        dt_s = 1.0 / MODEL_HZ
        if self.prev_timestamp is not None:
            dt = (obs.timestamp_us - self.prev_timestamp) / 1e6
            if dt > 0:
                dt_s = dt
            if dt > 0:
                repeats = max(1, round(dt * MODEL_HZ))
        if self.frames_seen == 0 and self.self_prime:
            # No history supplied, so manufacture one deep enough to fill the
            # 24-frame feature buffer. When a harness feeds the real preceding
            # frames (see Policy.infer_sequence) this is skipped.
            repeats = FEATURE_BUFFER_FRAMES
        self.prev_timestamp = obs.timestamp_us

        out = None
        for _ in range(repeats):
            for buf, packed in packs:
                pack_nv12(buf, packed, TARGET_W, TARGET_H, stride, y_height)
            # openpilot 0.11 added a `prepare_only` positional to ModelState.run.
            # Probed rather than pinned: the sha in repos.yaml predates it, and
            # hardcoding either arity breaks whichever checkout a site has.
            if self._run_takes_prepare_only:
                out = self.model.run(self.bufs, transforms, inputs, False)
            else:
                out = self.model.run(self.bufs, transforms, inputs)
        self.frames_seen += 1

        if out is None or "plan" not in out:
            return constant_velocity(obs.speed, dt=self.waypoint_dt)
        plan33 = np.asarray(out["plan"])[0]
        traj = Trajectory(plan_to_grid(plan33), dt=self.waypoint_dt)
        if NATIVE_LONG:
            traj.extras = {"native_long": self._native_long(out, plan33, obs.speed, dt_s, obs.speed_limit)}
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

    def diagnostics(self) -> dict[str, Any]:
        """Field-of-view margin, once a first frame has established the camera.

        A negative margin means the warp samples outside what the camera saw --
        black wedges at the edges of the model's input, invisible in the score.
        """
        if self.camera is None:
            return {"status": "no frame seen yet"}
        # Each view's margin from the camera that actually feeds it: in the
        # two-camera rig the road cam feeds "img" and the wide cam "big_img", so
        # reading both off one camera reports a false deficit for the other. A
        # single-camera source feeds both views from the one frame.
        slot_cam = (
            {"img": self.cam_road, "big_img": self.cam_wide}
            if self.cam_wide is not None
            else {"img": self.camera, "big_img": self.camera}
        )
        return {
            "source_hfov_deg": round(self.camera.src.hfov_deg, 2),
            "source_vfov_deg": round(self.camera.src.vfov_deg, 2),
            "margin_deg": {
                slot: {k: round(v, 2) for k, v in cam.margin_deg(slot).items()}
                for slot, cam in slot_cam.items()
            },
            "rpy": self.camera.rpy.tolist(),
            "frames_seen": self.frames_seen,
        }


POLICY = OpenpilotPolicy
