"""Drive-JEPA (perception-free, NAVSIM v1) as a paddock Policy.

A V-JEPA-2 ViT-L encoder over two front frames 0.5 s apart, plus an eight-number
ego status, feeding a three-layer transformer decoder that emits eight poses at
0.5 s -- 4 s of plan. There is no BEVFormer, no lidar, no map and no anchor
scorer, which is exactly why this variant exists in paddock: the perception-based
sibling in `adapters/models/drivejepa/` needs four cameras and where each sits on
the car, and a dashcam dataset has neither.

**Why a second registry entry rather than a mode on the first.** `runtime:` is
global to a model. This agent lives in `navsim_v1/`, not `navsim_v2/`; its
config path is resolved against the *process* working directory
(`drive_jepa_model.py` does `fname = "./vjepa2/configs/eval/vitl/in1k.yaml"`),
so `cwd` differs; and it loads a different checkpoint. Folding both into one
entry would mean a `runtime:` block that depends on which benchmark is running,
which is the shape of thing that silently scores the wrong weights. Two entries
keeps NAVSIM v2, nuScenes, KITScenes, neuroNCAP and Bench2Drive bit-identical.

Four adaptations.

*One camera into `cam_f0`.* `DriveJEPAFeatureDIBuilder(front_only=True)` reads
`cameras[-1].cam_f0` and `cameras[-2].cam_f0` and nothing else; the remaining
seven slots stay empty `Camera()` and `lidars` are placeholders. No
`require_lidar2ego()` -- the perception-free path never projects anything, so
demanding a lidar mounting would refuse comma2k19 for a transform nobody reads.

*The nuPlan crop is hard-coded in the feature builder,* which does
`image[28:-28]` unconditionally before resizing to 512x256. An image that is
already the right size silently loses 56 rows. So the frame is FOV-matched and
then **pre-padded** by 28 replicated rows top and bottom, which the builder's
own slice removes again. See `crop_box`.

*`cameras[-2]` is the OLDER frame.* `forward` builds its temporal axis as
`cat([camera_feature_2, camera_feature_1])`, so index -2 is t-0.5 s and -1 is
now. Getting this backwards does not raise and moves the endpoint by only about
1.1 m -- an error that would survive every smoke test. `_select_history` picks
by timestamp, so a source running at 20 Hz still yields a 0.5 s pair rather than
two frames 50 ms apart, which would read as a nearly static scene.

*Acceleration is derived.* `status_feature` is
`[driving_command(4), ego_velocity(2), ego_acceleration(2)]`, and comma2k19
publishes speed and yaw rate but no longitudinal acceleration channel, so
`obs.accel` is a flat zero there. Zeroing the acceleration fields cost the
reference 0.63 m of endpoint shift, so it is derived from the recorded speed
history instead, and `diagnostics()` says which source was used.

**Expect a poor headline on highway data, and read it correctly.** The plan's
4 s reach is near-linear in input speed up to ~14 m/s and then saturates flat at
~64 m for any input from 25 to 40 m/s: the model has no representation of
"faster than about 16 m/s", because NAVSIM is urban nuPlan data. On comma2k19 --
33 h of Californian freeway -- that puts raw ADE near 16.7 m while rescaling
each path so its reach matches ground truth drops it to 2.19 m. The path *shape*
transfers; the speed profile does not. `lateral_at_arclength_mae` and
`path_length_ratio` are the two columns that separate those, and this model is
the clearest demonstration in the repo of why this benchmark refuses to report
ADE alone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from paddock.policy.protocol import Observation, Policy, Trajectory

CAMERA = "CAM_FRONT"

#: `DriveJEPAFeatureDIBuilder` reads exactly two entries, 0.5 s apart -- the
#: spacing NAVSIM's `SensorConfig(include=[2, 3])` produces.
HISTORY_FRAMES = 2
HISTORY_DT = 0.5

#: `TrajectorySampling(time_horizon=4, interval_length=0.5)`: eight poses.
NUM_POSES = 8
POSE_DT = 0.5
HORIZON_S = 4.0

#: The released agent is evaluated with `num_keyval=129`, not the 257 its
#: config yaml defaults to. This is not cosmetic: it sizes the decoder's
#: key/value bank (8x16 pooled tokens plus the status token), so the wrong
#: value fails to load the checkpoint rather than scoring badly. Taken from
#: `navsim_v1/scripts/evaluation/eval_drive_jepa_perception_free.sh`, which
#: overrides the yaml on the command line.
NUM_KEYVAL = 129

#: nuPlan's CAM_F0 as the feature builder sees it: 1920x1080 at f = 1545 px,
#: cropped to 1920x1024 by the builder's own `[28:-28]`. Stored as tangents of
#: the half-angles so the crop below is a statement about field of view rather
#: than about one camera's pixel count.
NUPLAN_TAN_HALF_H = 960.0 / 1545.0  # hFOV 63.7 deg
NUPLAN_TAN_HALF_V = 512.0 / 1545.0  # vFOV 36.7 deg

#: Rows the feature builder removes, top and bottom, before resizing.
NAVSIM_CROP_MARGIN = 28

#: NAVSIM's driving command is a one-hot over [left, straight, right, unknown].
#: `Command` values are RIGHT=0, LEFT=1, STRAIGHT=2, FOLLOW_REFERENCE=3; the
#: signs were confirmed by the reference's own ablation, where forcing `left`
#: moved the plan +0.078 m in y and `right` moved it -0.124 m.
COMMAND_ONE_HOT = {
    0: (0, 0, 1, 0),  # RIGHT
    1: (1, 0, 0, 0),  # LEFT
    2: (0, 1, 0, 0),  # STRAIGHT
    3: (0, 0, 0, 1),  # FOLLOW_REFERENCE -> unknown
}

#: Frames kept per episode. comma2k19 sends 160 observations per sample and
#: this model looks at two; the rest are recorded for their timestamps and
#: ego state alone, and their pixels are never decoded.
HISTORY_KEEP = 64

#: Seconds of recorded speed to fit a slope through when the source publishes
#: no acceleration channel. Long enough to survive 20 Hz odometer quantisation,
#: short enough that a real acceleration is not averaged away.
ACCEL_WINDOW_S = 0.5


def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(x, y, heading) of *pose* expressed in *reference*'s frame."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], yaw_of(delta)], dtype=np.float64)


def crop_box(pinhole) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the window that reproduces nuPlan's FOV.

    Derived from the camera's own intrinsics rather than from comma2k19's
    numbers, so the same policy is correct on any front camera: the requested
    window is whatever this focal length makes subtend nuPlan's post-crop angles,
    centred on the principal point, clamped to the image.

    On the EON (1164x874, f = 910) that is 1131x603 -- hFOV 63.7 deg against the
    sensor's native 65.2, vFOV 36.7 against 51.2 -- and its 1.876 aspect is
    within a rounding of the 1.875 the builder's resize to 512x256 expects, so
    the squash the model was trained under is reproduced rather than added to.
    A camera that cannot supply the full angle is clamped and says so in
    `diagnostics()`; it is not padded, because padding widens a field of view
    with pixels that were never imaged.
    """
    half_w = min(pinhole.fx * NUPLAN_TAN_HALF_H, pinhole.cx, pinhole.width - pinhole.cx)
    half_h = min(pinhole.fy * NUPLAN_TAN_HALF_V, pinhole.cy, pinhole.height - pinhole.cy)
    left = max(0, int(round(pinhole.cx - half_w)))
    right = min(pinhole.width, int(round(pinhole.cx + half_w)))
    top = max(0, int(round(pinhole.cy - half_h)))
    bottom = min(pinhole.height, int(round(pinhole.cy + half_h)))
    return top, bottom, left, right


def to_model_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """FOV-match a frame and pre-pad it for the feature builder's own crop."""
    top, bottom, left, right = box
    cropped = image[top:bottom, left:right]
    return np.pad(
        cropped,
        ((NAVSIM_CROP_MARGIN, NAVSIM_CROP_MARGIN), (0, 0), (0, 0)),
        mode="edge",
    )


def cropped_intrinsics(pinhole, box: tuple[int, int, int, int]) -> np.ndarray:
    """Intrinsics of the crop, before the builder's resize to 512x256.

    Describes the crop rather than the padded array handed over, because the
    padding exists only to be removed again. Nothing on the perception-free path
    reads this -- the feature builder takes pixels and ignores calibration -- but
    a `Camera` with intrinsics that do not match its image is a trap for whoever
    reuses this next.
    """
    top, _, left, _ = box
    return np.array(
        [[pinhole.fx, 0.0, pinhole.cx - left], [0.0, pinhole.fy, pinhole.cy - top], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class Frame:
    """One recorded observation, with its pixels decoded on demand.

    comma2k19 sends image *paths* and 160 observations per sample, of which two
    are looked at. Decoding on record would spend 80x the work of the forward
    pass on PNGs that are thrown away -- and on this benchmark decode already
    dominates wall clock.
    """

    __slots__ = ("accel", "command", "ego2world", "obs", "speed", "timestamp", "yaw_rate")

    def __init__(self, obs: Observation) -> None:
        self.obs = obs
        self.timestamp = int(obs.timestamp_us)
        self.ego2world = np.asarray(obs.ego2world, dtype=np.float64)
        self.speed = float(obs.speed)
        self.accel = float(obs.accel)
        self.yaw_rate = float(obs.yaw_rate)
        self.command = int(obs.command)

    def image(self) -> np.ndarray:
        return self.obs.image(CAMERA)

    def release(self) -> None:
        """Drop decoded pixels; the ego state above is what the history needs.

        Only where a path can produce them again. `Observation.image` caches
        into `images`, and an open-loop harness sends paths, so dropping the
        cache there frees 3 MB a frame and costs a re-read nobody makes. A
        closed-loop observation arrives *as* pixels with no path behind it:
        popping that is not a cache eviction, it is deleting the only copy, and
        the next read raises `KeyError` instead of decoding. Found by
        `paddock verify`, whose second `infer_sequence` after `reset()` replays
        the very observations the first one had emptied.
        """
        if CAMERA in self.obs.image_paths:
            self.obs.images.pop(CAMERA, None)


class DriveJepaFreePolicy(Policy):
    cameras = (CAMERA,)
    waypoint_dt = POSE_DT

    def __init__(
        self,
        checkpoint: str | None = None,
        pretrain: str | None = None,
        device: str = "cuda",
    ) -> None:
        # Both paths arrive from model.yaml. The agent constructor builds the
        # encoder *from* the pretrain file and only then overwrites it with the
        # agent checkpoint, so supplying only the checkpoint fails at load.
        self.checkpoint = checkpoint or os.environ.get("DRIVEJEPA_FREE_CHECKPOINT")
        self.pretrain = pretrain or os.environ.get("DRIVEJEPA_VJEPA_WEIGHTS")
        self.device = device
        self.agent = None
        self.feature_builder = None
        self.frames: list[Frame] = []
        self._accel_source = "unknown"
        self._crop: tuple[int, int, int, int] | None = None
        self._crop_fov: tuple[float, float] | None = None
        self._intrinsics: np.ndarray | None = None

    def load(self) -> None:
        # navsim and nuplan exist only inside Drive-JEPA's own environment; this
        # module is imported by `paddock list` and by the test suite, which run
        # in paddock's.
        weights = (
            ("DRIVEJEPA_FREE_CHECKPOINT", self.checkpoint),
            ("DRIVEJEPA_VJEPA_WEIGHTS", self.pretrain),
        )
        # Checked for existence, not just for being set. `model.yaml` always
        # sets both from `{weights}`, so the emptiness test alone could never
        # fire on the `paddock run` path -- and with `startup_timeout_s: 900` a
        # missing 8.3 GB checkpoint cost up to fifteen minutes per node before
        # torch got far enough to say which file it could not find.
        for name, value in weights:
            if not value:
                raise RuntimeError(f"no weights: set {name} (model.yaml supplies it from {{weights}})")
        # A second pass, so "you did not set this" is never reported as the
        # absence of whatever the unset half would have pointed at.
        for name, value in weights:
            if not Path(str(value)).is_file():
                raise RuntimeError(
                    f"{name} points at {value}, which is not a file -- "
                    f"check [roots] weights in paddock.toml and that the checkpoint was downloaded"
                )

        import torch
        from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import DriveJEPAAgent
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        # These arguments are the released evaluation configuration, not the
        # yaml defaults: `eval_drive_jepa_perception_free.sh` overrides
        # num_keyval, freeze_encoder and double_image on the command line, and
        # the checkpoint only fits the overridden shape.
        agent = DriveJEPAAgent(
            trajectory_sampling=TrajectorySampling(time_horizon=HORIZON_S, interval_length=POSE_DT),
            pretrain_pt_path=self.pretrain,
            image_architecture="vit_large",
            tf_dropout=0.0,
            num_keyval=NUM_KEYVAL,
            front_only=True,
            freeze_encoder=False,
            double_image=True,
            checkpoint_path=self.checkpoint,
        )
        agent.initialize()
        self.agent = agent.eval().to(self.device)
        self.feature_builder = agent.get_feature_builders()[0]
        torch.set_grad_enabled(False)
        self.reset()

    def reset(self) -> None:
        """Drop the frame history between samples."""
        self.frames = []
        self._accel_source = "unknown"
        self._crop = None
        self._crop_fov = None
        self._intrinsics = None

    # ---- history --------------------------------------------------------

    def _record(self, obs: Observation) -> None:
        frame = Frame(obs)
        # A repeated or out-of-order timestamp would make the 0.5 s selection
        # below pick the same frame twice, which the ablations show costs about
        # 2 m -- and it would look like an ordinary bad sample.
        if self.frames and frame.timestamp <= self.frames[-1].timestamp:
            return
        self.frames.append(frame)
        if len(self.frames) > HISTORY_KEEP:
            for dropped in self.frames[:-HISTORY_KEEP]:
                dropped.release()
            self.frames = self.frames[-HISTORY_KEEP:]

    def _select_history(self) -> list[Frame]:
        """Two frames HISTORY_DT apart, oldest first, clamped at the run start.

        By timestamp, not by position: comma2k19 feeds 20 Hz and a simulator may
        step at 10, and taking the last two requests would hand the encoder two
        frames 50 ms apart. The temporal difference is real input -- making the
        two frames identical cost the reference nearly as much as blanking the
        camera outright -- so collapsing it is a silent 2 m error.
        """
        now = self.frames[-1].timestamp
        chosen = []
        for step in range(HISTORY_FRAMES - 1, -1, -1):
            target = now - int(round(step * HISTORY_DT * 1e6))
            chosen.append(min(self.frames, key=lambda f: abs(f.timestamp - target)))
        return chosen

    def _acceleration(self) -> np.ndarray:
        """(ax, ay) in the ego frame, m/s^2.

        `ax` prefers the recorded channel and falls back to the slope of the
        recorded speed over the last ACCEL_WINDOW_S, because comma2k19 fills
        speed and yaw rate but leaves the acceleration slot at zero, and a
        constant zero there is a different claim from "not measured".

        `ay` is the centripetal term `v * yaw_rate`. Differentiating a lateral
        position instead would need a heading history this model is not given,
        and the recorded yaw rate is a first-class protocol field.
        """
        current = self.frames[-1]
        if any(f.accel != 0.0 for f in self.frames):
            self._accel_source = "canbus"
            ax = current.accel
        else:
            self._accel_source = "speed_slope"
            window = [f for f in self.frames if current.timestamp - f.timestamp <= ACCEL_WINDOW_S * 1e6]
            if len(window) < 2:
                ax = 0.0
            else:
                t = np.array([f.timestamp for f in window], dtype=np.float64) / 1e6
                v = np.array([f.speed for f in window], dtype=np.float64)
                ax = float(np.polyfit(t - t[-1], v, 1)[0])
        return np.array([ax, current.speed * current.yaw_rate], dtype=np.float32)

    # ---- input ----------------------------------------------------------

    def _cameras(self, frame: Frame, obs: Observation):
        from navsim.common.dataclasses import Camera, Cameras

        if self._crop is None:
            pinhole = obs.pinhole(CAMERA)
            self._crop = crop_box(pinhole)
            top, bottom, left, right = self._crop
            self._crop_fov = (
                float(np.degrees(2 * np.arctan((right - left) / (2 * pinhole.fx)))),
                float(np.degrees(2 * np.arctan((bottom - top) / (2 * pinhole.fy)))),
            )
            self._intrinsics = cropped_intrinsics(pinhole, self._crop)

        empty = Camera()
        return Cameras(
            cam_f0=Camera(
                image=to_model_image(frame.image(), self._crop),
                intrinsics=self._intrinsics.astype(np.float32),
            ),
            cam_l0=empty,
            cam_l1=empty,
            cam_l2=empty,
            cam_r0=empty,
            cam_r1=empty,
            cam_r2=empty,
            cam_b0=empty,
        )

    def _agent_input(self, obs: Observation):
        from navsim.common.dataclasses import AgentInput, EgoStatus, Lidar

        frames = self._select_history()
        current = frames[-1].ego2world
        acceleration = self._acceleration()

        statuses = [
            EgoStatus(
                # Unused by the perception-free feature builder, which reads only
                # the last status's command, velocity and acceleration. Filled
                # correctly anyway, so a reader does not have to know that.
                ego_pose=relative_pose(frame.ego2world, current),
                ego_velocity=np.array([frame.speed, 0.0], dtype=np.float32),
                ego_acceleration=acceleration,
                driving_command=np.array(COMMAND_ONE_HOT[frame.command], dtype=np.int64),
            )
            for frame in frames
        ]
        cameras = [self._cameras(frame, obs) for frame in frames]
        # Placeholders: the perception-free agent never touches them, and
        # demanding a real lidar mounting would refuse a dashcam dataset for a
        # transform nobody reads.
        return AgentInput(ego_statuses=statuses, cameras=cameras, lidars=[Lidar(), Lidar()])

    # ---- protocol -------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        """Record a frame without running the model.

        The model is a pure function of its recorded history, so recording the
        first N-1 observations and inferring on the last produces the identical
        tensor -- same `_select_history`, same crop, same forward. On comma2k19
        that is one ViT-L pass per sample instead of 160.
        """
        self._record(obs)

    def infer(self, obs: Observation) -> Trajectory:
        if self.agent is None:
            raise RuntimeError("load() must be called before infer()")

        import torch

        self._record(obs)
        features = self.feature_builder.compute_features(self._agent_input(obs))
        features = {
            k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v) for k, v in features.items()
        }
        with torch.no_grad():
            output = self.agent.forward(features)

        for frame in self.frames:
            frame.release()

        # Eight poses at 0.5 s as (x, y, heading), ego frame, x forward and
        # y left -- already paddock's grid, so nothing is resampled.
        waypoints = output["trajectory"][0].float().cpu().numpy()
        return Trajectory(np.asarray(waypoints, dtype=np.float64), dt=POSE_DT)

    def diagnostics(self) -> dict[str, Any]:
        """What the model was actually shown, where it is cheap to be wrong.

        The history span is the one that matters most: it is selected by
        timestamp, and if it collapses towards zero the encoder is reading two
        near-identical frames and its plans will look plausible while being
        badly under-informed. The crop's field of view is second -- a camera
        whose angles do not reach nuPlan's is being clamped, and this is where
        that becomes visible rather than merely disappointing.
        """
        if not self.frames:
            return {"status": "no frame seen yet"}
        chosen = self._select_history()
        out: dict[str, Any] = {
            "frames_recorded": len(self.frames),
            "history_span_s": round((self.frames[-1].timestamp - chosen[0].timestamp) / 1e6, 3),
            "history_target_s": round((HISTORY_FRAMES - 1) * HISTORY_DT, 3),
            "speed_ms": round(self.frames[-1].speed, 3),
            "accel_source": self._accel_source,
        }
        if self._crop_fov is not None:
            out["crop_hfov_deg"] = round(self._crop_fov[0], 2)
            out["crop_vfov_deg"] = round(self._crop_fov[1], 2)
            out["crop_box"] = list(self._crop or ())
        return out


POLICY = DriveJepaFreePolicy
