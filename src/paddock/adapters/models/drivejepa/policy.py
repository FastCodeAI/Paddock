"""Drive-JEPA (perception-based, v2) as a paddock Policy.

Adapted from the NeuroNCAP model node. The model-specific reasoning is
unchanged; the seams now implement `Policy`, and every path arrives through
`model.yaml` rather than a hardcoded default.

Of the four models this is the closest fit to NeuroNCAP out of the box: it is a
NAVSIM agent, and NAVSIM's planning grid -- eight poses at 0.5 s, ego frame,
x forward and y left -- is exactly the grid NeuroNCAP drives on. The first six
poses are taken and nothing is resampled.

Rather than reimplement the feature pipeline, this builds the ``AgentInput``
dataclass NAVSIM already knows how to consume and calls the shipped feature
builder and agent. That keeps the BEVFormer projection maths (lidar2img from
each camera's sensor2lidar and intrinsics) in the code that was trained with it.

Five adaptations:

*Frame.* NAVSIM places every camera relative to the lidar (``sensor2lidar``),
and in every OpenScene log the lidar frame *is* the ego frame: lidar2ego is the
identity. The model relies on that without saying so -- the points it samples
the image at are the candidate plans' box corners, written in ego coordinates
(x forward, y left), and ``lidar2img`` is applied to them directly. So each
camera goes in at its pose on the car, ``camera2ego``, whatever lidar the
benchmark has. Placing it relative to a real lidar was a silent failure:
nuScenes' LIDAR_TOP is yawed -90 degrees, which turned every sample point a
quarter turn off the front camera. None landed in the image, the spatial
cross-attention masked all of them, and the plan came from ego status alone --
bit-identical given a black frame. Every neuroNCAP and nuScenes row made before
this was that camera-blind model; NAVSIM and KITScenes pass an identity
lidar2ego and were not.

*Cameras.* Drive-JEPA's perception branch reads four views -- front, front-left,
front-right and back -- which map onto nuScenes cameras of the same names. The
resolutions differ (NAVSIM is 1920x1080, nuScenes 1600x900) but the geometry
travels in the intrinsics and the pipeline rescales, so this is a genuine
adaptation rather than a fudge. The one hard-coded number that does not travel
is the ``[28:-28]`` crop the stitching path applies to reach a 2:1 aspect: on
1080 rows that yields 1024, on 900 it yields 844. Both are then resized to
512x256, so the effect is a slightly different vertical crop, not a distortion.

*Rate.* Drive-JEPA is a 2 Hz agent with a four-frame history. NeuroNCAP may be
stepping at 10 Hz to suit the other models, so the history is selected by
timestamp at ~0.5 s spacing rather than by taking the last four requests --
otherwise the model would see 0.3 s of history and read the scene as nearly
static.

*Absent views.* A source with one camera -- comma2k19's road-facing phone --
has no image for the other three slots, and they cannot be left empty. The
model keeps one view, camera index 1 (upstream ``drive_jepa_model.py:50``,
``lidar2img[:, 1:2]``), and its loader skips an empty slot
(``bev_feature_build.py:26-29``), which would move the front camera off index 1.
Projected points are also scaled by camera 0's image size, the back slot's
(``bevformer/encoder.py:141-142``). So a camera the source does not have
(`_absent`) gets a black frame the size of the front image, with the front
camera's calibration, which the loader needs to invert. Neither the black
pixels nor that calibration reach the plan. It is never the forward image:
a slot for another direction keeps its meaning (docs/benchmarks/comma2k19.md,
"What each model sees"). `diagnostics()` lists these slots as ``blank_views``.

*Acceleration.* The model's ego status carries forward acceleration. A source
that never fills the CAN bus's acceleration channel (comma2k19, KITScenes) would
hand it a constant zero, which tells the model the car holds its speed. So the
channel is used when the source has filled it at least once in the episode;
otherwise the slope of the recorded speed over the last 0.5 s stands in, as the
devkit's own KITScenes loader differentiates its poses. Sideways acceleration
stays zero, as on every source. `diagnostics()` reports which as
``accel_source``.

*Momentum-aware selection* (paper section 3.5) is on: ``model.yaml`` declares
``DRIVEJEPA_MOMENTUM=on``. ``off`` selects on the plain score, and is also what
a node gets with the variable unset, for example one launched outside paddock.
With it on, each replan is handed the plan this node chose about 0.5 s
earlier, the way upstream's own evaluation loop hands it over, so the model's
``calibrate_score`` runs before the argmax. See `momentum.py`. Open loop that
never happens -- every sample resets the node -- except in a streamed mode,
where each sample names its recorded drive (`infer_sequence_streamed`):
navtest's and navhard's per log, and the runner's for nuScenes, KITScenes and
comma2k19.

*The pick, reported.* Every answer carries ``extras["drivejepa_pick"]``: which
of the 32 proposals was driven, its score, and whether the momentum stage was
given a previous plan for it (`_pick`). A closed-loop client can then log the
model's own choice rather than infer it from the path. Clients that read only
``trajectory`` (neuroNCAP) never see it; `protocol.validate_extras` passes it
as a model-native block.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from paddock.adapters.models.drivejepa import momentum
from paddock.policy.protocol import CANBUS_ACCEL, Observation, Policy, StreamFrame, Trajectory

# Drive-JEPA camera slot -> nuScenes camera. The perception branch reads these
# four; the remaining slots stay empty and are skipped by the loader.
CAMERA_MAP = {
    "cam_f0": "CAM_FRONT",
    "cam_l0": "CAM_FRONT_LEFT",
    "cam_r0": "CAM_FRONT_RIGHT",
    "cam_b0": "CAM_BACK",
}
NEEDED_CAMERAS = tuple(CAMERA_MAP.values())
FRONT_CAMERA = CAMERA_MAP["cam_f0"]

#: Seconds of recorded speed whose slope stands in for an acceleration channel
#: the source never fills (module docstring, *Acceleration*).
ACCEL_WINDOW_S = 0.5

IMAGE_GEOMETRIES = ("native", "openscene")

#: Threads decoding the selected history frames' images; see `_decode`.
DECODE_THREADS = 4

HISTORY_FRAMES = 4  # NAVSIM agent input depth
HISTORY_DT = 0.5  # seconds between history frames, NAVSIM's rate

# NeuroNCAP's Command enum -> NAVSIM's one-hot [left, straight, right, unknown].
COMMAND_ONE_HOT = {
    0: [0, 0, 1, 0],  # RIGHT
    1: [1, 0, 0, 0],  # LEFT
    2: [0, 1, 0, 0],  # STRAIGHT
    3: [0, 0, 0, 1],  # FOLLOW_REFERENCE -> unknown
}


def yaw_of(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(x, y, heading) of *pose* expressed in *reference*'s frame."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], yaw_of(delta)], dtype=np.float64)


class DriveJepaPolicy(Policy):
    cameras = NEEDED_CAMERAS
    waypoint_dt = HISTORY_DT

    def __init__(self, checkpoint: str | None = None, device: str = "cuda") -> None:
        # Resolved from model.yaml (DRIVEJEPA_CHECKPOINT) rather than defaulted
        # to one machine's path, so a site with the checkpoint elsewhere needs
        # no code change.
        self.checkpoint = checkpoint or os.environ.get("DRIVEJEPA_CHECKPOINT")
        self.device = device
        # "openscene" crops and rescales each camera onto the OpenScene camera
        # the checkpoint trained on -- what the devkit's own KITScenes loader
        # does -- and is what model.yaml sets. "native" hands each camera over at
        # its own intrinsics; every Drive-JEPA row published before 2026-10-22 was
        # made that way. It matters: KITScenes' ring is 87 degrees against
        # OpenScene's 64, and uncropped the model's ADE there was 3.222 m against
        # 2.521 m. The fallback here stays "native" for a policy built outside the
        # runner, with no model.yaml env.
        self.image_geometry = os.environ.get("DRIVEJEPA_IMAGE_GEOMETRY", "native")
        if self.image_geometry not in IMAGE_GEOMETRIES:
            raise ValueError(
                f"DRIVEJEPA_IMAGE_GEOMETRY={self.image_geometry!r}; expected one of {IMAGE_GEOMETRIES}"
            )
        # Off is how every published drivejepa row was made: the node took the
        # raw argmax of the learned score. On adds the stage upstream's own
        # evaluation runs before that argmax (`momentum.py`). It changes which
        # of the 32 proposals is driven, so it is refused if misspelt rather
        # than read as off.
        switch = os.environ.get(momentum.SWITCH_ENV, "off")
        if switch not in momentum.SWITCHES:
            raise ValueError(f"{momentum.SWITCH_ENV}={switch!r}; expected one of {momentum.SWITCHES}")
        self.momentum = momentum.Momentum() if switch == "on" else None
        #: The recorded drive the sample being planned belongs to, while
        #: `infer_sequence_streamed` runs; None otherwise.
        self._stream: StreamFrame | None = None
        self.agent = None
        self.feature_builder = None
        self.history: list[dict[str, Any]] = []
        self.blank_views: set[str] = set()
        self.source_has_accel = False
        self.momentum_fed = False

    def load(self) -> None:
        # navsim only exists inside Drive-JEPA's own environment; this module is
        # imported by `paddock list` and the test suite, which run in paddock's.
        if not self.checkpoint:
            raise RuntimeError(
                "no checkpoint: set DRIVEJEPA_CHECKPOINT (model.yaml supplies it "
                "from {data.navsim.exp}) or pass one to DriveJepaPolicy"
            )
        from navsim.agents.drive_jepa_perception_based.drive_jepa_agent import DriveJEPAAgent
        from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

        config = DriveJEPAConfig()
        self.agent = DriveJEPAAgent(config=config, lr=1e-4, checkpoint_path=self.checkpoint)
        self.agent.initialize()
        self.agent.eval().to(self.device)
        self.feature_builder = self.agent.get_feature_builders()[0]
        if self.momentum is not None:
            # Here rather than on the first replan: the stage needs nuPlan's
            # simulator and four more upstream modules, and a node that cannot
            # import them should fail to start, not fail its first `/infer`
            # after the simulator and renderer are already up.
            self.momentum.upstream = momentum.Upstream(self.agent._trajectory_sampling)
        self.reset()

    def reset(self) -> None:
        """Drop the frame history, and the previous plan, between runs."""
        self.history: list[dict[str, Any]] = []
        #: Cameras this episode's source does not have, whose slots were filled
        #: black (`_absent`). Reported by `diagnostics()`, so a run records which
        #: of its views were real.
        self.blank_views: set[str] = set()
        #: Whether this source has filled the CAN bus's acceleration channel in
        #: this episode. Sticky, so trimming old history cannot change how
        #: acceleration is derived (`_accel`).
        self.source_has_accel = False
        if self.momentum is not None:
            self.momentum.reset()

    def _record(self, obs: Observation) -> None:
        # Images are decoded later, and only for the four frames
        # `_select_history` picks (`_decode`). An open-loop harness hands over
        # every 10 Hz frame of its history window -- 49 on KITScenes, four ring
        # cameras each -- as paths; decoding them all here cost ~10 s a sample
        # to use 16 of 196.
        self.history.append(
            {
                "obs": obs,
                "timestamp": obs.timestamp_us,
                "ego2world": obs.ego2world,
                "camera2ego": obs.camera2ego,
                "speed": obs.speed,
                "accel": obs.accel,
                "command": int(obs.command),
            }
        )
        canbus = np.asarray(obs.canbus, dtype=np.float64)
        if canbus[CANBUS_ACCEL] or canbus[CANBUS_ACCEL + 1]:
            self.source_has_accel = True
        # 12 s of 10 Hz history is far more than the 1.5 s ever used. An entry
        # holds its Observation until `_decode` picks it -- paths open-loop,
        # pixels closed-loop -- then the four decoded cameras; trimming keeps
        # either from growing without bound.
        if len(self.history) > 128:
            self.history = self.history[-128:]

    @staticmethod
    def _absent(obs: Observation, camera: str) -> bool:
        """Is `camera` one this source does not have, as opposed to one that went missing?

        Only a source that describes its front camera can say another is
        absent: comma2k19 calibrates its road camera and sends nothing at all
        for the rest. So a camera counts as absent only when the front has both
        its intrinsics and its pose on the car -- the two a filled slot copies --
        and this camera has neither an image nor any calibration. Anything else
        is a camera that went missing, and `obs.image` raises for it as before.
        """
        described = FRONT_CAMERA in obs.camera2image and FRONT_CAMERA in obs.camera2ego
        traced = camera in obs.images or camera in obs.image_paths
        calibrated = camera in obs.camera2image or camera in obs.camera2ego
        return camera != FRONT_CAMERA and described and not traced and not calibrated

    def _decode(self, frames: list[dict[str, Any]]) -> None:
        """Give each frame its `images` and `camera2image`, decoding in parallel.

        PIL releases the GIL while decoding a JPEG, so four threads overlap four
        8-megapixel reads. A frame keeps its pixels once decoded and drops the
        observation it came from.

        A camera the source does not have (`_absent`) is decided here, for the
        frames the model reads and nowhere else, so this branch runs only where
        `obs.image` would raise: a source that sends all four cameras is
        untouched. Its slot is filled after the image geometry is applied, so the
        black frame has the processed front image's size and intrinsics.
        """
        pending = list({id(f): f for f in frames if "images" not in f}.values())
        if not pending:
            return
        absent = {id(f): [c for c in self.cameras if self._absent(f["obs"], c)] for f in pending}
        jobs = [
            (frame, camera) for frame in pending for camera in self.cameras if camera not in absent[id(frame)]
        ]
        with ThreadPoolExecutor(DECODE_THREADS) as pool:
            pixels = list(pool.map(lambda job: job[0]["obs"].image(job[1]), jobs))
        decoded = {(id(frame), camera): pixels[i] for i, (frame, camera) in enumerate(jobs)}
        for frame in pending:
            obs = frame.pop("obs")
            blank = absent[id(frame)]
            images = {camera: decoded[(id(frame), camera)] for camera in self.cameras if camera not in blank}
            camera2image = dict(obs.camera2image)
            if self.image_geometry == "openscene":
                images, camera2image = self._to_openscene(images, camera2image)
            if blank:
                # Copies, never the observation's own dicts: the front keeps its
                # calibration, and the filled slots borrow it only to be invertible.
                for camera in blank:
                    images[camera] = np.zeros_like(images[FRONT_CAMERA])
                    camera2image[camera] = camera2image[FRONT_CAMERA]
                front2ego = frame["camera2ego"][FRONT_CAMERA]
                frame["camera2ego"] = {**frame["camera2ego"], **{camera: front2ego for camera in blank}}
                self.blank_views.update(blank)
            frame["images"], frame["camera2image"] = images, camera2image

    def _to_openscene(self, images: dict[str, np.ndarray], camera2image: dict[str, np.ndarray]):
        """Crop and rescale every camera onto the OpenScene pinhole.

        The window is the devkit KITScenes loader's own (`_crop_box`), imported
        rather than restated, so a paddock run and that loader feed the model
        the same pixels. Extrinsics are untouched: a crop about the principal
        point changes only the intrinsics, which become OpenScene's exactly.
        """
        from PIL import Image

        from paddock.adapters.models.drivejepa.kitscenes_adapter import (
            TARGET_HEIGHT,
            TARGET_WIDTH,
            _crop_box,
            target_intrinsics,
        )

        out_images, out_k = {}, {}
        for camera, image in images.items():
            k = camera2image[camera]
            pinhole = {"focal_length": k[0, 0], "principal_point_u": k[0, 2], "principal_point_v": k[1, 2]}
            cropped = Image.fromarray(image).crop(_crop_box(pinhole)[:4])
            cropped = cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
            out_images[camera] = np.asarray(cropped)
            out_k[camera] = target_intrinsics().astype(np.float64)
        return out_images, out_k

    def _select_history(self) -> list[dict[str, Any]]:
        """Four frames at ~0.5 s spacing, oldest first, clamped at the run start.

        Selected by timestamp rather than by position, so the same node behaves
        correctly whether NeuroNCAP is stepping at 0.5 s or 0.1 s.
        """
        now = self.history[-1]["timestamp"]
        chosen = []
        for step in range(HISTORY_FRAMES - 1, -1, -1):
            target = now - int(step * HISTORY_DT * 1e6)
            nearest = min(self.history, key=lambda f: abs(f["timestamp"] - target))
            chosen.append(nearest)
        return chosen

    def _cameras(self, frame: dict[str, Any]):
        from navsim.common.dataclasses import Camera, Cameras

        slots: dict[str, Camera] = {}
        for slot, camera in CAMERA_MAP.items():
            # NAVSIM's "lidar" frame is the ego frame (module docstring, *Frame*),
            # so sensor2lidar is the camera's pose on the car.
            sensor2lidar = frame["camera2ego"][camera]
            slots[slot] = Camera(
                image=frame["images"][camera],
                sensor2lidar_rotation=sensor2lidar[:3, :3].astype(np.float32),
                sensor2lidar_translation=sensor2lidar[:3, 3].astype(np.float32),
                intrinsics=frame["camera2image"][camera].astype(np.float32),
            )
        empty = Camera()
        return Cameras(
            cam_f0=slots["cam_f0"],
            cam_l0=slots["cam_l0"],
            cam_l1=empty,
            cam_l2=empty,
            cam_r0=slots["cam_r0"],
            cam_r1=empty,
            cam_r2=empty,
            cam_b0=slots["cam_b0"],
        )

    def _agent_input(self):
        from navsim.common.dataclasses import AgentInput, EgoStatus

        frames = self._select_history()
        self._decode(frames)
        current = frames[-1]["ego2world"]

        ego_statuses = []
        for frame in frames:
            ego_statuses.append(
                EgoStatus(
                    ego_pose=relative_pose(frame["ego2world"], current),
                    ego_velocity=np.array([frame["speed"], 0.0], dtype=np.float32),
                    ego_acceleration=np.array([self._accel(frame), 0.0], dtype=np.float32),
                    driving_command=np.array(COMMAND_ONE_HOT[frame["command"]], dtype=np.int64),
                )
            )
        cameras = [self._cameras(frame) for frame in frames]
        return AgentInput(ego_statuses=ego_statuses, cameras=cameras, lidars=[])

    def _accel(self, frame: dict[str, Any]) -> float:
        """Forward acceleration at *frame*, m/s^2 (module docstring, *Acceleration*).

        The recorded channel when this source has filled it in the episode;
        otherwise the least-squares slope of the recorded speed over the
        ACCEL_WINDOW_S ending at *frame*, and 0.0 when the window holds fewer
        than two distinct times.
        """
        if self.source_has_accel:
            return float(frame["accel"])
        end = frame["timestamp"]
        window = [f for f in self.history if 0 <= end - f["timestamp"] <= ACCEL_WINDOW_S * 1e6]
        t = np.array([f["timestamp"] for f in window], dtype=np.float64) / 1e6
        if len(window) < 2 or np.ptp(t) == 0:
            return 0.0
        v = np.array([f["speed"] for f in window], dtype=np.float64)
        return float(np.polyfit(t - t[-1], v, 1)[0])

    def observe(self, obs: Observation) -> None:
        """Record a history frame without running the model.

        The default `observe` is a full BEVFormer forward pass per frame with
        the plan thrown away -- four passes to use one on NAVSIM. The model is
        a pure function of its recorded history, so recording the first N-1
        and inferring on the last produces the identical tensor: same
        `_select_history`, same features, same forward. Bit-identical, four
        times cheaper.

        That holds with momentum on as well, for a narrower reason: a frame
        that is only recorded is never planned from, so it leaves no previous
        plan behind, and `infer_sequence` resets before it starts. A sequence
        therefore always selects on the plain score -- which is why the NAVSIM
        proxy, which scores through `infer_sequence`, is untouched by the switch.
        (The streamed modes, navtest's and navhard's, are the exception, and they
        go through the per-drive book `infer_sequence_streamed` keeps, not this.)

        Not a pattern to copy into openpilot or visionpilot: both carry state
        that each forward pass advances, so for them every pass is load-bearing.
        """
        self._record(obs)

    def forward(self, obs: Observation) -> dict[str, Any]:
        """Record *obs* and return the agent's raw output for it.

        Besides the chosen ``trajectory`` this holds all 32 ``proposals``, the
        scorer's six ``pred_logit`` per proposal and its ``pdm_score``. `infer`
        keeps only the pick; anything that studies the candidates needs the
        rest, and reads it from here so it sees exactly the input path `infer`
        does.
        """
        if self.agent is None:
            raise RuntimeError("load() must be called before infer()")

        import torch

        self._record(obs)
        agent_input = self._agent_input()
        features = self.feature_builder.compute_features(agent_input)
        features = {
            k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v) for k, v in features.items()
        }

        # Whether this pass's pick was made against a previous plan; read by
        # `_pick`. Set per call so a stale value cannot survive.
        self.momentum_fed = False
        if self.momentum is None:
            with torch.no_grad():
                return self.agent.forward(features)

        extra, ego_state, time_us = self._momentum_features(obs)
        features.update(extra)
        with torch.no_grad():
            output = self.agent.forward(features)
            # What the argmax would have been without the stage: the same
            # expression, on the same tensor, as drive_jepa_model.py:85.
            plain = torch.argmax(torch.sigmoid(output["pred_logit"])[:, :, -1], dim=1)
            changed = bool((plain != torch.argmax(output["pdm_score"], dim=1)).any())
        chosen = output["trajectory"][0].float().cpu().numpy()
        self._momentum_remember(chosen, ego_state, time_us, changed)
        return output

    def _momentum_features(self, obs: Observation) -> tuple[dict[str, Any], Any, int]:
        """The stage's three keys for *obs*, the ego state behind them, and the
        time the plan will be kept under. Sets `momentum_fed`.

        Apart from `forward` only so the bookkeeping can be tested without
        torch; `forward` calls this and `_momentum_remember` around its pass.
        """
        pose, velocity, acceleration, time_us = self._ego_now(obs)
        key = self._stream.key if self._stream is not None else None
        extra, ego_state = self.momentum.features(pose, velocity, acceleration, time_us, stream=key)
        self.momentum_fed = extra["past_ego_simulated_states"] is not None
        return extra, ego_state, time_us

    def _momentum_remember(self, chosen: np.ndarray, ego_state: Any, time_us: int, changed: bool) -> None:
        key = self._stream.key if self._stream is not None else None
        self.momentum.remember(chosen, ego_state, time_us, changed=changed, stream=key)

    def _ego_now(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Pose, velocity, acceleration and time the momentum stage is given.

        Velocity and acceleration are the newest `EgoStatus` row `_agent_input`
        builds -- the numbers the model itself is given -- which is upstream's
        pairing: one frame record feeds both the model and the stage. So the
        acceleration is `_accel` of the record *obs* made, which `forward` has
        just appended. The pose is ego2world, which plays the part of upstream's
        global frame.

        In a streamed sample (`infer_sequence_streamed`) pose and time are the
        stream's instead: an open-loop harness places every sample's scored
        frame at its own origin and time zero, which is right for the model's
        input and useless for comparing two samples' plans, so the drive's own
        frame and clock come with the stream -- upstream's `metric_cache.ego_state`
        is that frame's recorded global pose too. Velocity and acceleration are
        the observation's either way, as is everything the model itself is fed.
        """
        velocity = np.array([obs.speed, 0.0], dtype=np.float32)
        acceleration = np.array([self._accel(self.history[-1]), 0.0], dtype=np.float32)
        if self._stream is not None:
            return (
                np.asarray(self._stream.pose, dtype=np.float64),
                velocity,
                acceleration,
                int(self._stream.timestamp_us),
            )
        pose = np.array([obs.ego2world[0, 3], obs.ego2world[1, 3], yaw_of(obs.ego2world)], dtype=np.float64)
        return pose, velocity, acceleration, int(obs.timestamp_us)

    def infer_sequence_streamed(self, observations: list[Observation], stream: StreamFrame) -> Trajectory:
        """One open-loop sample that is a frame of a recorded drive (`protocol.StreamFrame`).

        Exactly `infer_sequence` -- the same reset, history and forward pass --
        except that the momentum stage keeps its book under the drive's name, on
        the drive's own clock and frame, so the next frame of the same drive is
        handed this frame's plan if it comes 0.45-0.55 s later (`momentum.py`).
        With momentum off there is no book, and the stream changes nothing.
        """
        if self.momentum is None:
            return self.infer_sequence(observations)
        self._stream = stream
        try:
            return self.infer_sequence(observations)
        finally:
            self._stream = None

    def infer(self, obs: Observation) -> Trajectory:
        output = self.forward(obs)
        # NAVSIM plans eight poses at 0.5 s as (x, y, heading), ego frame,
        # x forward and y left -- exactly the grid paddock uses, so nothing is
        # resampled. All eight are returned: closed-loop takes the first six,
        # and open-loop scoring wants the full horizon for FDE@3s.
        trajectory = output["trajectory"][0].float().cpu().numpy()
        return Trajectory(trajectory, dt=HISTORY_DT, extras={"drivejepa_pick": self._pick(output)})

    def _pick(self, output: dict[str, Any]) -> dict[str, Any]:
        """The proposal driven, as the model chose it (module docstring).

        ``chosen_index`` is the proposal (of 32) whose trajectory `infer`
        returns and ``chosen_score`` its score: the argmax and max of
        ``pdm_score``, which is the momentum-recalibrated score when a previous
        plan was fed and the learned one otherwise (drive_jepa_model.py:85-91).
        ``momentum_fed`` is whether a previous plan was fed to this pick.
        """
        scores = output["pdm_score"][0].float().cpu().numpy()
        index = int(np.argmax(scores))
        return {
            "chosen_index": index,
            "chosen_score": float(scores[index]),
            "momentum_fed": bool(self.momentum_fed),
        }

    def diagnostics(self) -> dict[str, Any]:
        """History depth actually achieved, the image geometry, and momentum.

        The reason history matters: it is selected by timestamp, so a
        simulator stepping faster than 0.5 s still yields ~1.5 s of context. If
        this collapses toward zero the model is reading a near-static scene and
        its plans will look plausible while being badly under-informed.

        The geometry is here because it changes the pixels the model sees, and
        the runner stores this dict in run.json as `provenance.policy`. An
        explicit env override also lands in `model_env`; the default does not,
        so without this a `native` run would carry no record of being one.

        Momentum is here for the same reason, and with its counts: `fed` of
        `replans` is how often the stage actually had a previous plan, and a
        run with the switch on and `fed` at zero selected on the plain score
        throughout -- an open-loop sequence, or a simulator whose calls never
        fall 0.45 to 0.55 s apart.

        `camera_frame` marks a node that places each camera on the ego frame; a
        run.json without it predates that fix and saw no image on nuScenes.
        That matters for momentum in particular: its first neuroNCAP comparison
        was made before the fix, on a model planning from ego status alone. The
        marker describes the attempt that wrote run.json, not every episode in
        it: a resumed run keeps the episodes of its earlier attempts, and the
        runner does not yet check that those came from a node reporting the same.

        `blank_views` lists the slots filled black because the source has no
        such camera, and `accel_source` says whether the model's acceleration
        was the source's channel or the slope of its speed (module docstring).
        Both describe the latest episode, which is the one the runner reads.
        """
        selection: dict[str, Any] = {"momentum": "off"}
        if self.momentum is not None:
            selection = {"momentum": "on", "momentum_counts": dict(self.momentum.counts)}
        if not self.history:
            return {
                "status": "no frame seen yet",
                "image_geometry": self.image_geometry,
                "camera_frame": "ego",
                **selection,
            }
        chosen = self._select_history()
        return {
            "image_geometry": self.image_geometry,
            "camera_frame": "ego",
            **selection,
            "frames_recorded": len(self.history),
            "history_span_s": round((self.history[-1]["timestamp"] - chosen[0]["timestamp"]) / 1e6, 2),
            "history_target_s": round((HISTORY_FRAMES - 1) * HISTORY_DT, 2),
            "blank_views": sorted(self.blank_views),
            "accel_source": "canbus" if self.source_has_accel else "speed_slope",
        }


POLICY = DriveJepaPolicy
