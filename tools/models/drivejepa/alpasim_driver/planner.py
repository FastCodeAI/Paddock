# SPDX-License-Identifier: Apache-2.0
"""DriveJEPAPlanner: AlpaSim observations in, Drive-JEPA plan out.

Implements the translation map in ``INTERFACE_DRIVEJEPA.md`` §5 against the
A4a scaffold's :class:`~scaffold.Planner` interface. Two translations, in order:

    Translation 1  AlpaSim ObservationView  ->  Drive-JEPA feature dict   (§5.1)
    [ the model ]  features                 ->  output dict               (§2)
    Translation 2  output dict              ->  PlanResult                (§5.2)

Three things are deliberately isolated, each in exactly one function, because
each is a place where a future answer or a future component has to land without
a hunt:

    _compute_sensor2lidar   the OQ1/OQ2 extrinsics convention  -> §8.1, §8.2
    _run_model              the model call itself              -> swap fake/real
    _bev_normalize          the to_rgb swap that must not be "fixed" -> §6.2

The planner never touches the local frame: it returns rig-frame waypoints and
the scaffold composes and anchors them (§5.2).

**What 0.1.4 changed, and why.** Three edits, all forced by the shared 20 Hz
referee (control_timestep_us == frame_interval_us == 50_000):

  _newest_velocity   the standstill gate is a SPEED, not a 0.5 m distance.
                     The old constant was 1.0 m/s at 500 ms ticks and silently
                     became 10.0 m/s at 50 ms. See K.HANDOVER_MIN_SPEED_M_S.
  _select_frames     off-grid context is now a HARD failure, not a silent
                     accept. The old +-50% window swallowed the 350 ms pair
                     that an 8-frame buffer produces at 20 Hz.
  _assert_push_cadence  once per session, the frame push interval must equal
                     the control timestep. Nothing in the harness checks this
                     (FLAG-2), and a mismatch silently breaks the anchor
                     contract the scaffold and controller share.

Replanning at Drive-JEPA's native 2 Hz on this 20 Hz clock is NOT done here --
it is the divider's job (divider.py), which wraps this planner and short-
circuits before build_features on the nine ticks in ten that do not replan.

**What 0.1.4b changed, and why.** One thing: the model is given the plan it
chose at the previous replan, so the vendored ``calibrate_score`` (paper §3.5,
drive_jepa_model.py:87-88,98-151) runs before the argmax, as it does in
upstream's own evaluation (run_pdm_score_sequential.py:130-138,167-168). The
planner's part is bookkeeping only -- remember, per rollout, the simulated
states of the last chosen trajectory and when it was chosen, and send them with
the next replan (``_momentum_request`` / ``_momentum_update``). Everything that
touches nuPlan happens in the worker (momentum.py). "Previous" means the last
real inference: the divider calls ``plan`` only on replan ticks, so warm-up
inferences count, and the one at -0.5 s is the predecessor of the handover
replan. With DRIVEJEPA_MOMENTUM=off (or ``momentum="off"``) ``plan`` makes the
0.1.4 call and ``build_plan_result`` runs its 0.1.4 body.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from paddock.adapters.benchmarks.alpasim.scaffold import ObservationView, Planner, PlanResult

from . import constants as K

logger = logging.getLogger(__name__)

#: The seam the real model plugs into. Takes the feature dict Translation 1
#: builds, returns the output dict Translation 2 reads.
ModelFn = Callable[[dict], dict]

class ContextSpacingError(RuntimeError):
    """The context pair is not on the model's grid, and never will be.

    Raised rather than returned as "not ready" because the two cases are
    different: a buffer that cannot yet reach 0.5 s back IS not-ready and
    resolves itself a few ticks later, while a buffer that reaches back and
    still cannot produce a 0.5 s pair is a misconfiguration that will hold for
    the whole rollout. 0.1.3 could not tell them apart -- its +-50% window
    accepted the second case silently, which is how a 20 Hz run would have fed
    350 ms of context to a checkpoint trained on 500 ms.

    Failing the rollout is the point. A scored run on corrupted input is worse
    than no run, and the scaffold turns "not ready" into a legal empty
    trajectory (INTERFACE.md §4.8) -- so degrading here would have been silent.
    """


class PushCadenceError(RuntimeError):
    """Camera frames are not arriving at the control cadence.

    Closes the hole the audit logged as FLAG-2: nothing in the runtime
    validates ``frame_interval_us == control_timestep_us``. The sensorsim
    check covers only ``force_gt_duration_us % control_timestep_us``
    (sensorsim_service.py:125-136), and ``assert_zero_decision_delay`` only
    rejects frames stamped in the FUTURE (events/policy.py:198-205), not stale
    ones. When the two clocks differ the newest camera frame no longer
    coincides with the policy step, ``model_t0_us != time_now_us``, and the
    runtime/controller frame handoff silently mis-composes -- with no warning
    anywhere in the stack. So the adapter checks it at its own boundary.
    """


#: How fast the car must be moving, by the finite difference of the last two
#: poses, before a reported velocity of exactly zero is treated as the warm-up
#: sentinel rather than a standstill.
#:
#: ``ObservationView.latest_speed_m_s`` and the ``DynamicState`` behind it use
#: ``0.0`` for both "no egomotion has arrived" and "the car is stopped"
#: (scaffold.py:150-163), and the runtime pushes zero-velocity dynamic states
#: through the force-GT warm-up -- ``DynamicState()``, all fields zero, is what
#: the controller emits while force-GT drives (controller_service.py:284-292).
#: Meanwhile the poses in the *same* observation advance. The two streams
#: disagree, and the newest ego_status row is the only motion signal the
#: network gets: its pose block is all zeros by construction (§3.3), so a
#: zeroed velocity leaves the model with no evidence it is moving at all.
#:
#: 1 m/s is comfortably above the pose noise of a genuinely stopped car and far
#: below any handover speed. A real standstill stays under it and is passed
#: through untouched, which is the point: spoofing a speed onto a stopped car
#: would be the worse failure.
#:
#: **This replaces 0.1.3's ``HANDOVER_MIN_POSE_GAP_M = 0.5``**, which was the
#: same threshold expressed as a distance and therefore only correct at the
#: tick rate it was tuned at. See K.HANDOVER_MIN_SPEED_M_S for the arithmetic.
HANDOVER_MIN_SPEED_M_S = K.HANDOVER_MIN_SPEED_M_S


# ===========================================================================
# Extrinsics -- the one function OQ1/OQ2 can change
# ===========================================================================


def _compute_sensor2lidar(
    rig_to_camera: np.ndarray,
    *,
    rig_to_camera_is_camera_pose_in_rig: bool = True,
    rig_to_lidar: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Drive-JEPA's ``sensor2lidar`` from AlpaSim's ``rig_to_camera``.

    ===================================================================
    THE ONLY PLACE THE EXTRINSICS CONVENTION IS DECIDED.
    If the upstream answer changes anything, it changes here and nowhere else.
    ===================================================================

    **OQ1 -- direction: RESOLVED** (INTERFACE_DRIVEJEPA.md §8.1, high
    confidence). ``AvailableCamera.rig_to_camera`` is the active transform
    rig -> camera, i.e. *the pose of the camera expressed in the rig frame*, so
    it is used **as-is, with no inversion**. Four independent sources agree:
    AlpaSim's renderer composes ``ego_pose @ rig_to_camera`` to place the camera
    (`sensorsim_service.py:292-301,322-325`); its own docstring says "the pose of
    the camera in the rig" (`camera_catalog.py:35`); the alternative reading puts
    the camera 1.65 m underground facing right; and the source data names the
    same quantity ``nominalSensor2Rig_FLU``.

    The flag is kept anyway as hygiene -- it makes the assumption greppable and
    reversible in one edit rather than implicit in the arithmetic.

    **OQ2 -- the lidar frame: STILL OPEN** (§8.2). Drive-JEPA expresses geometry
    lidar-relative, but AlpaSim has no lidar and Drive-JEPA's own config sets
    ``lidar_height = 0``, which suggests lidar coincides with the rig. Default is
    therefore identity. Since Drive-JEPA only ever uses ``lidar2img``, a constant
    rig->lidar offset is absorbed into the projection -- so the practical
    question is whether the residual is zero or merely small.

    Args:
        rig_to_camera: 4x4 homogeneous transform from the AlpaSim camera
            declaration.
        rig_to_camera_is_camera_pose_in_rig: True (the resolved reading) uses the
            matrix directly. False inverts it first -- kept only so the
            resolution can be reversed in one place if it is ever overturned.
        rig_to_lidar: 4x4 rig -> lidar transform, or None for "lidar is the rig".

    Returns:
        ``(rotation 3x3, translation 3)``, the camera's pose in the lidar frame,
        which is exactly what ``Camera.sensor2lidar_*`` means.
    """
    camera_pose_in_rig = (
        np.asarray(rig_to_camera, dtype=np.float64)
        if rig_to_camera_is_camera_pose_in_rig
        else np.linalg.inv(np.asarray(rig_to_camera, dtype=np.float64))
    )

    if rig_to_lidar is None:
        camera_pose_in_lidar = camera_pose_in_rig
    else:
        # Re-express the camera's rig-frame pose in the lidar frame.
        camera_pose_in_lidar = (
            np.linalg.inv(np.asarray(rig_to_lidar, dtype=np.float64))
            @ camera_pose_in_rig
        )

    return camera_pose_in_lidar[:3, :3].copy(), camera_pose_in_lidar[:3, 3].copy()


# ===========================================================================
# Small geometry helpers
# ===========================================================================


def _pose_to_matrix(pose) -> np.ndarray:
    """4x4 homogeneous matrix from a ``common.Pose`` protobuf.

    The proto's quaternion is **w-first** (INTERFACE.md §5); scipy and most
    numpy code are xyzw, so the reordering happens here, once.
    """
    q = pose.quat
    v = pose.vec
    w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        rotation = np.eye(3)
    else:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [float(v.x), float(v.y), float(v.z)]
    return matrix


def _yaw_of(matrix: np.ndarray) -> float:
    """Yaw (rotation about z) of a 4x4 or 3x3 transform, radians."""
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(dx, dy, dyaw)`` of ``pose`` expressed in ``reference``'s frame (§5.1)."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], _yaw_of(delta)], dtype=np.float64)


# ===========================================================================
# Translation 1 -- image pipelines
# ===========================================================================


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode an AlpaSim camera frame to RGB uint8 HWC.

    AlpaSim sends an *encoded* image with no format field on the message, so we
    sniff rather than assume JPEG -- PIL does that for us (INTERFACE.md §4.3).
    ``convert("RGB")`` pins the channel order and drops any alpha, so everything
    downstream can rely on RGB.
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _pipeline_a(image_rgb: np.ndarray) -> np.ndarray:
    """The V-JEPA branch: crop, resize, ToTensor (§3.1).

    Stops at [0, 1]. The model applies the ImageNet normalisation itself
    (`drive_jepa_model.py:42-47`), so pre-applying it here would double-normalise
    and put the input far out of distribution.
    """
    if image_rgb.shape[0] <= 2 * K.CROP_ROWS:
        raise ValueError(
            f"image is {image_rgb.shape[0]} rows, too short for the "
            f"[{K.CROP_ROWS}:-{K.CROP_ROWS}] crop that §3.1 requires"
        )
    cropped = image_rgb[K.CROP_ROWS : -K.CROP_ROWS]
    # cv2.resize takes (width, height) -- the opposite of numpy's (H, W).
    resized = cv2.resize(cropped, K.RESIZE_WH, interpolation=cv2.INTER_LINEAR)
    # Same mask as the retraining converter. Zero in [0, 1] space is zero in
    # uint8, so masking before ToTensor gives the identical tensor.
    resized = resized.copy()
    resized[-K.BONNET_MASK_ROWS :, :, :] = 0
    # ToTensor: HWC uint8 -> CHW float32 in [0, 1].
    return (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _bev_normalize(image_rgb: np.ndarray) -> np.ndarray:
    """The BEV branch's mmcv-equivalent ``imnormalize`` (§3.2).

    ===================================================================
    THE CHANNEL SWAP BELOW IS DELIBERATE. DO NOT "FIX" IT.
    ===================================================================

    Upstream calls ``mmcv.imnormalize(..., to_rgb=True)``
    (`bev_feature_build.py:11`), a flag whose documented job is converting
    **BGR -> RGB**. But NAVSIM loads its images with PIL
    (`navsim_v2/navsim/common/dataclasses.py:81`), which yields **RGB** already.
    So the flag swaps channels the wrong way, and the model is normalised on
    BGR data with RGB-ordered ImageNet statistics.

    **The published checkpoint was trained through exactly this pipeline**, so
    those weights expect channels in the order this code produces. Correcting
    the swap would put inputs out of distribution and silently degrade the model
    -- a regression with no error message. We replicate the training-time
    behaviour and record the deviation (§6.2) instead of repairing it.

    (Today the swap is inert: these normalised pixels are built and then never
    read by ``forward`` (§3.2). It becomes live the moment anyone re-enables the
    four-view ``camera_feature`` upstream, which is precisely why it is
    replicated now rather than left as a comment.)
    """
    image = image_rgb.astype(np.float32)
    if K.BEV_TO_RGB:
        # cv2.COLOR_BGR2RGB on a 3-channel array is a channel reversal.
        image = image[..., ::-1]
    mean = np.array(K.BEV_MEAN_255, dtype=np.float32)
    std = np.array(K.BEV_STD_255, dtype=np.float32)
    return (image - mean) / std


def _pipeline_b(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The BEV branch: normalize, scale 0.4, pad to /32, CHW (§3.2).

    Returns the CHW tensor and the padded ``img_shape`` (H, W, C) -- the latter
    is genuinely consumed, to normalise projected pixel coordinates
    (`encoder.py:141-142`).
    """
    normalized = _bev_normalize(image_rgb)

    height, width = normalized.shape[:2]
    # int() truncation, matching RandomScaleImageMultiViewImage (transform3d.py:335-336).
    new_w = int(width * K.BEV_SCALE)
    new_h = int(height * K.BEV_SCALE)
    scaled = cv2.resize(normalized, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # impad_to_multiple: pad right/bottom with zeros up to a multiple of 32.
    pad_h = int(np.ceil(new_h / K.BEV_PAD_DIVISOR)) * K.BEV_PAD_DIVISOR
    pad_w = int(np.ceil(new_w / K.BEV_PAD_DIVISOR)) * K.BEV_PAD_DIVISOR
    padded = np.zeros((pad_h, pad_w, scaled.shape[2]), dtype=np.float32)
    padded[:new_h, :new_w] = scaled

    return padded.transpose(2, 0, 1), (pad_h, pad_w, scaled.shape[2])


def _scale_matrix(scale: float) -> np.ndarray:
    """The 4x4 the BEV scale applies to ``lidar2img`` (transform3d.py:337-339)."""
    matrix = np.eye(4)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    return matrix


def _lidar2img(
    intrinsics_k: np.ndarray,
    sensor2lidar_rotation: np.ndarray,
    sensor2lidar_translation: np.ndarray,
) -> np.ndarray:
    """Replicate upstream's ``lidar2img`` construction (`bev_feature_build.py:35-44`).

    Kept arithmetically identical to upstream, transposes and all, rather than
    rewritten into a cleaner equivalent -- if upstream's version has a quirk, we
    want the same quirk, since that is what the checkpoint was trained with.
    """
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsics_k.shape[0], : intrinsics_k.shape[1]] = intrinsics_k
    return viewpad @ lidar2cam_rt.T


def _intrinsics_matrix(spec) -> np.ndarray:
    """3x3 pinhole K from an AlpaSim ``CameraSpec`` (§5.1).

    Only the pinhole model is supported. The doc says "rectify f-theta ->
    pinhole if needed"; our target preset renders pinhole directly
    (`vavam_configs.yaml`), so rather than half-implement a rectifier this
    raises with a pointer to the one AlpaSim already ships.
    """
    which = spec.WhichOneof("camera_param")
    if which == "opencv_pinhole_param":
        p = spec.opencv_pinhole_param
        return np.array(
            [
                [p.focal_length_x, 0.0, p.principal_point_x],
                [0.0, p.focal_length_y, p.principal_point_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    raise NotImplementedError(
        f"camera {spec.logical_id!r} uses lens model {which!r}; Drive-JEPA needs "
        "pinhole intrinsics (INTERFACE_DRIVEJEPA.md §5.1). Rectify first with "
        "alpasim_driver.rectification.build_ftheta_rectifier_for_resolution, or "
        "render pinhole as the VaVAM preset does."
    )


def _dummy_image(height: int, width: int) -> np.ndarray:
    """A uniform, recognisable RGB frame to hold BEV slot 0 (§6.1).

    Its pixels are never read by ``forward`` -- only its position in the list
    matters, keeping the front camera at index 1, the one index the model keeps.
    The fill is a marker the fake can detect, so a broken ordering fails a test
    instead of silently projecting through a side camera.
    """
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = K.DUMMY_FILL_RGB[0]
    image[..., 1] = K.DUMMY_FILL_RGB[1]
    image[..., 2] = K.DUMMY_FILL_RGB[2]
    return image


# ===========================================================================
# Translation 1 -- non-image inputs
# ===========================================================================


def _command_one_hot(route, *, min_lookahead_m: float, lateral_m: float) -> tuple:
    """Route waypoints (rig frame) -> ``[left, straight, right, unknown]`` (§5.1).

    Take the first waypoint at least ``min_lookahead_m`` away; if its rig-frame
    y is beyond ``lateral_m`` it is a turn. Rig +y is LEFT, so positive y is a
    left turn. Thresholds are AlpaSim's VaVAM values (INTERFACE.md §6.3(c)).
    """
    if route is None or len(route.waypoints) == 0:
        return K.COMMAND_UNKNOWN

    target = None
    for waypoint in route.waypoints:
        if float(np.hypot(waypoint.x, waypoint.y)) >= min_lookahead_m:
            target = waypoint
            break
    if target is None:
        # A route exists but stops short of the lookahead: nothing to turn for.
        return K.COMMAND_STRAIGHT

    if target.y > lateral_m:
        return K.COMMAND_LEFT
    if target.y < -lateral_m:
        return K.COMMAND_RIGHT
    return K.COMMAND_STRAIGHT


# ===========================================================================
# The planner
# ===========================================================================


def momentum_mode(requested: str = "auto") -> str:
    """Resolve the 0.1.4b switch to ``"on"`` or ``"off"``.

    ``"auto"`` reads env DRIVEJEPA_MOMENTUM (default on). An unknown value
    raises rather than falling back: a typo must not silently pick a side.
    """
    mode = requested
    if mode == "auto":
        mode = os.environ.get(K.MOMENTUM_ENV, "on")
    mode = str(mode).strip().lower()
    if mode not in K.MOMENTUM_MODES:
        raise ValueError(
            f"{K.MOMENTUM_ENV}={mode!r} is not one of {K.MOMENTUM_MODES}"
        )
    return mode


class DriveJEPAPlanner(Planner):
    """Drive-JEPA (perception-based, NAVSIM v2) behind the scaffold's interface."""

    # Derived, never a literal: a draft that symlinks constants.py cannot then
    # report a version different from the code it is running. Before 0.1.3 this
    # was hand-written here, which is exactly how a draft's real planner.py
    # could claim 0.1.2 while a symlinked one claimed 0.1.0.
    version_id = f"drivejepa-v2-driver-{K.ADAPTER_VERSION}"

    def __init__(
        self,
        model_fn: ModelFn,
        *,
        front_camera_id: str,
        frame_spacing_us: int = K.FRAME_SPACING_US,
        frame_spacing_tolerance: float = K.CONTEXT_SPACING_TOLERANCE,
        expected_frame_interval_us: int | None = None,
        min_lookahead_m: float = K.MIN_LOOKAHEAD_M,
        command_lateral_m: float = K.COMMAND_LATERAL_M,
        rig_to_camera_is_camera_pose_in_rig: bool = True,
        rig_to_lidar: np.ndarray | None = None,
        momentum: str = "auto",
    ) -> None:
        """
        Args:
            model_fn: The model, as a callable taking the feature dict and
                returning the output dict. **The only route to the model.**
            momentum: 0.1.4b. ``"on"`` feeds the previous plan to the model
                (paper §3.5), ``"off"`` is exact 0.1.4 selection, ``"auto"``
                (the default) reads env DRIVEJEPA_MOMENTUM, itself defaulting
                to on. Anything else raises.
            front_camera_id: AlpaSim logical_id of the front camera. Its frames
                feed both pipelines and it must land in BEV slot 1.
            frame_spacing_us: Target gap between the two frames (§3.1).
            frame_spacing_tolerance: Fractional slack on that gap, as a HARD
                limit. 0.1.4 defaults to K.CONTEXT_SPACING_TOLERANCE (0.10);
                0.1.3's 0.5 accepted 250-750 ms, which at 20 Hz swallowed the
                350 ms pair an 8-frame buffer produces. Off-grid now raises
                :class:`ContextSpacingError` rather than degrading.
            expected_frame_interval_us: The scenario's control timestep. When
                set, the first observation carrying two frames must show that
                push interval or :class:`PushCadenceError` is raised -- the
                FLAG-2 closure. None disables the check.
            min_lookahead_m, command_lateral_m: Route -> command thresholds.
            rig_to_camera_is_camera_pose_in_rig, rig_to_lidar: forwarded to
                :func:`_compute_sensor2lidar`. See OQ1/OQ2 there.
        """
        self._model_fn = model_fn
        self._front_camera_id = front_camera_id
        self._frame_spacing_us = frame_spacing_us
        self._frame_spacing_tolerance = frame_spacing_tolerance
        self._expected_frame_interval_us = expected_frame_interval_us
        # Per-session, so one long-lived process serving many rollouts checks
        # the cadence once per rollout rather than once per process. Guarded:
        # plan() runs outside the scaffold's session lock and two rollouts can
        # be inside it at the same time (scaffold.py:246-251).
        self._cadence_checked: set[str] = set()
        self._cadence_lock = threading.Lock()
        self._min_lookahead_m = min_lookahead_m
        self._command_lateral_m = command_lateral_m
        self._rig_to_camera_is_camera_pose_in_rig = rig_to_camera_is_camera_pose_in_rig
        self._rig_to_lidar = rig_to_lidar
        # 0.1.4b: resolved once, at construction, so a rollout cannot change
        # behaviour halfway through if the environment changes underneath it.
        self._momentum_on = momentum_mode(momentum) == "on"
        # Per-session: (model_t0_us, simulated states) of the last chosen
        # trajectory -- what upstream's loop carries from one token to the next
        # (run_pdm_score_sequential.py:167-168). Guarded for the same reason
        # as the cadence set: plan() runs outside the scaffold's session lock.
        self._momentum_prev: dict[str, tuple[int, np.ndarray]] = {}
        self._momentum_lock = threading.Lock()

    # ------------------------------------------------------------------
    # The model seam
    # ------------------------------------------------------------------

    def _run_model(self, features: dict) -> dict:
        """Call the model. **The only place in this file that touches it.**

        Swapping the fake for the real agent is a change to the injected
        ``model_fn`` and nothing else. For the real one, that callable owns the
        numpy -> torch conversion, the ``.eval()`` call and ``torch.no_grad()``;
        this planner stays numpy-only and CPU-only.
        """
        return self._model_fn(features)

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------

    def _assert_push_cadence(
        self, observations: ObservationView, frames: tuple[tuple[int, bytes], ...]
    ) -> None:
        """Once per session: frames must arrive at the control cadence.

        This is the FLAG-2 closure, and it is a *startup* check by design --
        validated on the first observation that carries two frames, then never
        again. Checking every tick would turn a single dropped frame into a
        failed rollout; checking once catches the thing that actually goes
        wrong, which is a config where ``frame_interval_us`` and
        ``control_timestep_us`` disagree for the whole run.

        Why the adapter has to do this at all: nothing upstream does. See
        :class:`PushCadenceError`.

        No-ops when ``expected_frame_interval_us`` is None, which is how the
        2 Hz launchers and the unit tests opt out.
        """
        if self._expected_frame_interval_us is None:
            return
        uuid = observations.session_uuid
        with self._cadence_lock:
            if uuid in self._cadence_checked:
                return
            self._cadence_checked.add(uuid)

        observed = int(frames[-1][0]) - int(frames[-2][0])
        expected = int(self._expected_frame_interval_us)
        slack = expected * self._frame_spacing_tolerance
        if abs(observed - expected) > slack:
            raise PushCadenceError(
                f"camera {self._front_camera_id!r} is pushing frames "
                f"{observed} us apart, but this driver was launched for a "
                f"{expected} us control timestep "
                f"(tolerance +-{slack:.0f} us).\n"
                "frame_interval_us and control_timestep_us must be equal: the "
                "scaffold anchors plans on the newest frame's timestamp, which "
                "only coincides with the policy step when the two clocks match. "
                "Nothing in the runtime validates this -- fix the driver YAML."
            )
        logger.info(
            "push cadence ok: session %s, frames %d us apart, expected %d us",
            uuid,
            observed,
            expected,
        )

    def _select_frames(
        self, observations: ObservationView
    ) -> tuple[tuple[int, bytes], tuple[int, bytes]] | None:
        """Newest frame plus the one closest to one step earlier (§3.1).

        Returns ``(older, newer)``, or None when the buffer cannot yet reach
        one step back -- the planner's "not ready" signal.

        Selected by timestamp rather than by position, so a runtime ticking
        faster than 2 Hz still yields ~0.5 s of temporal context instead of two
        nearly identical frames.

        **0.1.4 separates two cases 0.1.3 conflated.** The buffer not yet
        reaching ``newest - frame_spacing_us`` is *not ready*: it is the normal
        state for the first few ticks of a rollout and it resolves itself. The
        buffer reaching that far and still producing an off-grid pair is a
        *misconfiguration* -- too short a history, or a push cadence that
        cannot land on the grid -- and it will hold for the whole rollout. The
        first returns None; the second raises.

        0.1.3 answered None to both, with a +-50% window that in fact accepted
        the second case outright: at 20 Hz with the scaffold's default 8-frame
        buffer the widest available pair is 350 ms, and 350 ms sits inside
        [250 ms, 750 ms]. The run would have scored on 350 ms of context
        against a 500 ms checkpoint, silently.
        """
        frames = observations.image_cache.get(self._front_camera_id, ())
        if len(frames) < K.CONTEXT_FRAMES:
            return None

        self._assert_push_cadence(observations, frames)

        newest = frames[-1]
        target = newest[0] - self._frame_spacing_us

        # Not ready: the oldest frame we hold is still newer than the frame we
        # want. Nothing is wrong -- the buffer simply has not filled yet.
        if int(frames[0][0]) > target:
            logger.debug(
                "context buffer reaches back only %d us, need %d us; not ready",
                int(newest[0]) - int(frames[0][0]),
                self._frame_spacing_us,
            )
            return None

        older = min(frames[:-1], key=lambda entry: abs(entry[0] - target))

        gap = newest[0] - older[0]
        low = self._frame_spacing_us * (1.0 - self._frame_spacing_tolerance)
        high = self._frame_spacing_us * (1.0 + self._frame_spacing_tolerance)
        if not (low <= gap <= high):
            # The buffer DOES span the target (checked above) and the nearest
            # frame to it is still off-grid. That is structural, not transient.
            raise ContextSpacingError(
                f"context pair is {gap} us apart; the model wants "
                f"{self._frame_spacing_us} us +-{self._frame_spacing_tolerance:.0%} "
                f"(accepted [{low:.0f}, {high:.0f}]).\n"
                f"buffer holds {len(frames)} frames spanning "
                f"{int(newest[0]) - int(frames[0][0])} us.\n"
                "Raise --image-history so the buffer can reach one full step "
                "back, or fix frame_interval_us. Refusing to run: 0.1.3 would "
                "have accepted this and scored the rollout on short context."
            )
        return older, newest

    # ------------------------------------------------------------------
    # Translation 1
    # ------------------------------------------------------------------

    def build_features(
        self,
        observations: ObservationView,
        older: tuple[int, bytes],
        newer: tuple[int, bytes],
    ) -> dict:
        """AlpaSim observations -> the Drive-JEPA feature dict (§5.1).

        Public so tests can exercise it directly, without a model or a server.
        """
        older_rgb = _decode_rgb(older[1])
        newer_rgb = _decode_rgb(newer[1])

        # --- Pipeline A: newest is camera_feature_1, older is _2 (§3.1) ---
        camera_feature_1 = _pipeline_a(newer_rgb)[None]
        camera_feature_2 = _pipeline_a(older_rgb)[None]

        # --- Pipeline B: dummy in slot 0, front camera in slot 1 (§6.1) ---
        height, width = newer_rgb.shape[:2]
        slot_images = [_dummy_image(height, width), newer_rgb]

        camera = observations.cameras.get(self._front_camera_id)
        if camera is None:
            raise ValueError(
                f"camera {self._front_camera_id!r} was not declared in "
                "start_session; its intrinsics and rig_to_camera are needed for "
                "lidar2img (INTERFACE_DRIVEJEPA.md §5.1)"
            )
        intrinsics_k = _intrinsics_matrix(camera.intrinsics)
        rotation, translation = _compute_sensor2lidar(
            _pose_to_matrix(camera.rig_to_camera),
            rig_to_camera_is_camera_pose_in_rig=(
                self._rig_to_camera_is_camera_pose_in_rig
            ),
            rig_to_lidar=self._rig_to_lidar,
        )
        scale = _scale_matrix(K.BEV_SCALE)

        bev_tensors, img_shapes, lidar2imgs = [], [], []
        for image in slot_images:
            tensor, shape = _pipeline_b(image)
            bev_tensors.append(tensor)
            img_shapes.append(shape)
            # Every slot carries a lidar2img, including the dummy: upstream
            # computes one per present camera before the slice discards it.
            lidar2imgs.append(scale @ _lidar2img(intrinsics_k, rotation, translation))

        return {
            "camera_feature_1": camera_feature_1.astype(np.float32),
            "camera_feature_2": camera_feature_2.astype(np.float32),
            "camera_feature": np.stack(bev_tensors)[None].astype(np.float32),
            "lidar2img": np.stack(lidar2imgs)[None].astype(np.float32),
            "img_shape": np.array(img_shapes, dtype=np.float32)[None],
            "ego_status": self._model_ego_status(
                self._build_ego_status(observations)
            )[None],
        }

    @staticmethod
    def _model_ego_status(ego_status: np.ndarray) -> np.ndarray:
        """The ego_status the MODEL reads -- acceleration slots zeroed.

        The retrained weights were trained with these two slots at zero on
        every sample; feeding the reported acceleration would be an input they
        have never seen. Everything else in the row is untouched.
        """
        out = ego_status.copy()
        out[..., K.EGO_ACCEL_SLOTS[0] : K.EGO_ACCEL_SLOTS[1]] = 0.0
        return out

    def _newest_velocity(
        self, observations: ObservationView, current: np.ndarray
    ) -> tuple[float, float]:
        """Rig-frame ``(vx, vy)`` for the newest ego_status row (§3.3).

        Normally the reported velocity, verbatim. The one exception is the
        warm-up sentinel: a velocity of *exactly* zero standing beside poses
        that are visibly moving. There the reported value is not a measurement,
        it is the harness's "unknown" wearing a measurement's clothes, and the
        finite difference of the last two poses is the honest reading of the
        same observation.

        Only the newest row is ever corrected. ``forward`` reads only that row
        (``ego_status[:, -1]``, §3.3), and rewriting history rows would put
        derived numbers where the interface promises reported ones.

        Acceleration is left exactly as reported either way: this function
        knows the car was moving, not how its speed was changing.
        """
        state = observations.dynamic_states[-1][1]
        reported = (float(state.linear_velocity.x), float(state.linear_velocity.y))
        if reported != (0.0, 0.0):
            return reported
        if len(observations.poses) < 2:
            # One pose is not a displacement. Nothing to derive from.
            return reported

        dt_s = (
            int(observations.poses[-1].timestamp_us)
            - int(observations.poses[-2].timestamp_us)
        ) / 1e6
        if dt_s <= 0.0:
            return reported

        # _relative_pose puts the PREVIOUS pose in the CURRENT rig frame, so
        # negating it is the displacement the car actually covered -- already
        # in the frame the velocity is specified in (x forward, y left), which
        # is why this cannot just difference the world-frame vectors.
        previous = _relative_pose(
            _pose_to_matrix(observations.poses[-2].pose), current
        )
        dx, dy = -float(previous[0]), -float(previous[1])
        distance_m = float(np.hypot(dx, dy))
        derived = (dx / dt_s, dy / dt_s)
        derived_speed_m_s = float(np.hypot(*derived))
        if derived_speed_m_s <= HANDOVER_MIN_SPEED_M_S:
            # Genuinely stopped, as far as this observation can tell. Pass the
            # zero through untouched: spoofing a speed onto a stationary car is
            # a worse failure than reporting an honest one on a moving car.
            #
            # 0.1.4: compared as a SPEED. 0.1.3 compared `distance_m` against a
            # fixed 0.5 m, which is this same threshold only at 500 ms ticks.
            # At the referee's 50 ms ticks that constant meant 10 m/s and the
            # defence would have been off below 36 km/h. At 500 ms the two
            # forms agree bit-for-bit: dt is exactly 0.5, dividing by it is
            # exact in binary, so `d/0.5 <= 1.0` iff `d <= 0.5`.
            return reported
        logger.info(
            "handover sentinel: reported velocity is exactly zero but the last "
            "two poses are %.3f m apart over %.3f s; substituting derived speed "
            "%.4f m/s (vx=%.4f, vy=%.4f)",
            distance_m,
            dt_s,
            derived_speed_m_s,
            derived[0],
            derived[1],
        )
        return derived

    def _build_ego_status(self, observations: ObservationView) -> np.ndarray:
        """``(F, 11)`` ego status, newest last (§3.3).

        ``forward`` reads only the newest row, but the full stack is built
        because that is what the interface specifies and what a future variant
        would consume.
        """
        current = _pose_to_matrix(observations.poses[-1].pose)
        command = _command_one_hot(
            observations.route,
            min_lookahead_m=self._min_lookahead_m,
            lateral_m=self._command_lateral_m,
        )
        newest_index = len(observations.poses) - 1
        newest_velocity = self._newest_velocity(observations, current)

        rows = []
        for index, pose_at_time in enumerate(observations.poses):
            # Poses are relative to the current one, so the newest row is zeros.
            relative = _relative_pose(_pose_to_matrix(pose_at_time.pose), current)
            state = observations.dynamic_states[index][1]
            velocity = (
                newest_velocity
                if index == newest_index
                else (state.linear_velocity.x, state.linear_velocity.y)
            )
            rows.append(
                np.concatenate(
                    [
                        relative,
                        # 2-D, rig frame: x forward, y left. z is dropped.
                        velocity,
                        [state.linear_acceleration.x, state.linear_acceleration.y],
                        command,
                    ]
                )
            )
        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Translation 2
    # ------------------------------------------------------------------

    @staticmethod
    def build_plan_result(
        output: dict, model_t0_us: int, extra_debug: dict | None = None
    ) -> PlanResult:
        """Drive-JEPA output -> ``PlanResult`` (§5.2).

        Static so tests can check the numerics with no planner and no model.

        0.1.4b: ``extra_debug`` is merged into the debug payload. The pick is
        still read from the output -- with a previous plan fed, the model's own
        ``forward`` has already taken the argmax of the recalibrated score
        (drive_jepa_model.py:88-91), so ``trajectory`` and ``pdm_score`` are
        the momentum-aware ones. ``None`` is the 0.1.4 body, untouched.
        """
        trajectory = np.asarray(output["trajectory"])[0]  # (8, 3) = (x, y, heading)

        # Drive-JEPA's frame IS AlpaSim's rig frame -- ego rear-axle, x forward,
        # y left (§4.2) -- so positions pass through untouched, gaining only the
        # z column the scaffold expects.
        positions = np.zeros((trajectory.shape[0], 3), dtype=np.float64)
        positions[:, :2] = trajectory[:, :2]

        # Heading -> a rotation about z, per waypoint.
        headings = trajectory[:, 2]
        cos, sin = np.cos(headings), np.sin(headings)
        rotations = np.zeros((len(headings), 3, 3), dtype=np.float64)
        rotations[:, 0, 0] = cos
        rotations[:, 0, 1] = -sin
        rotations[:, 1, 0] = sin
        rotations[:, 1, 1] = cos
        rotations[:, 2, 2] = 1.0

        scores = np.asarray(output["pdm_score"])[0]
        debug = {
            "planner": "drivejepa-v2",
            "num_proposals": int(np.asarray(output["proposals"]).shape[1]),
            "chosen_index": int(np.argmax(scores)),
            "chosen_score": float(np.max(scores)),
        }
        if extra_debug:
            debug.update(extra_debug)
        return PlanResult(
            positions_in_rig=positions,
            step_us=K.STEP_US,
            rotations_in_rig=rotations,
            # The plan was built from the newest camera frame, which trails the
            # drive request. Reporting it lets the scaffold anchor honestly
            # rather than fall back to stamping time_now_us onto a stale pose
            # (§5.2, and LEARNINGS.md "anchor staleness").
            model_t0_us=model_t0_us,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # 0.1.4b: the previous plan, per rollout
    # ------------------------------------------------------------------

    def _momentum_request(
        self, observations: ObservationView, features: dict, model_t0_us: int
    ) -> dict:
        """What the worker needs to rebuild upstream's three keys (momentum.py).

        Plain numpy and ints only: the bridge crosses two numpy versions.

        The ego pose is the pose the plan will be anchored on -- the cached
        pose nearest ``model_t0_us``, the scaffold's own rule
        (scaffold.py ``_resolve_anchor``) -- in AlpaSim's local frame, which is
        fixed for the rollout and so stands in for upstream's global frame.
        Velocity and acceleration are the newest ``ego_status`` row, i.e. the
        numbers the model itself is about to read: upstream's ego state and its
        model input come from the same frame record in the same way.
        """
        anchor = min(
            observations.poses,
            key=lambda pose: abs(int(pose.timestamp_us) - int(model_t0_us)),
        )
        matrix = _pose_to_matrix(anchor.pose)
        newest = np.asarray(features["ego_status"])[0, -1]
        # Acceleration is zero in BOTH places -- the model's ego_status row
        # (_model_ego_status) and the ego state the §3.5 re-score simulates
        # from (momentum.py ego_state reads request["ego_acceleration"]).
        acceleration = np.zeros(2, dtype=np.float32)
        with self._momentum_lock:
            previous = self._momentum_prev.get(observations.session_uuid)
        return {
            "time_us": int(model_t0_us),
            "anchor_time_us": int(anchor.timestamp_us),
            "ego_pose": np.array(
                [matrix[0, 3], matrix[1, 3], _yaw_of(matrix)], dtype=np.float64
            ),
            "ego_velocity": np.array(newest[3:5], dtype=np.float32),
            "ego_acceleration": acceleration,
            "past_time_us": None if previous is None else int(previous[0]),
            "past_states": None if previous is None else previous[1],
        }

    def _momentum_update(self, session_uuid: str, request: dict, output: dict) -> dict:
        """Remember this replan's chosen trajectory; return the debug payload."""
        missing = [
            key
            for key in ("momentum_states", "momentum_meta", "momentum_S")
            if key not in output
        ]
        if missing:
            raise RuntimeError(
                f"momentum is on but the model returned no {missing}. The worker "
                "behind model_fn is not 0.1.7's (alpasim_driver/momentum.py); "
                "refusing to drive on plain S while claiming §3.5."
            )
        meta = output["momentum_meta"]
        states = np.array(output["momentum_states"], dtype=np.float64)
        with self._momentum_lock:
            # run_pdm_score_sequential.py:167-168.
            self._momentum_prev[session_uuid] = (int(request["time_us"]), states)

        comfort = output.get("momentum_Sc")
        debug = {
            "momentum": "on",
            "momentum_code": meta["code"],
            "prev_plan_present": request["past_states"] is not None,
            "prev_plan_age_s": meta["interval_s"],
            "overlap_start": meta["overlap_start"],
            "S": [float(v) for v in np.asarray(output["momentum_S"]).reshape(-1)],
            "S_c": (
                None if comfort is None
                else [float(v) for v in np.asarray(comfort).reshape(-1)]
            ),
            "calibrated": [float(v) for v in np.asarray(output["pdm_score"])[0]],
            "pick_with_momentum": int(meta["pick_with"]),
            "pick_without_momentum": int(meta["pick_without"]),
            "momentum_ms": float(
                meta["ms_prepare"] + meta["ms_calibrate"] + meta["ms_simulate"]
            ),
            "momentum_ms_parts": [
                float(meta["ms_prepare"]),
                float(meta["ms_calibrate"]),
                float(meta["ms_simulate"]),
            ],
            "momentum_anchor_dt_us": int(request["anchor_time_us"]) - int(request["time_us"]),
        }
        logger.info(
            "momentum: %s prev_plan_present=%s age_s=%s overlap=%s comfortable=%s/%d "
            "pick %d (plain S: %d) %.1f ms",
            debug["momentum_code"],
            debug["prev_plan_present"],
            "-" if debug["prev_plan_age_s"] is None else f"{debug['prev_plan_age_s']:.3f}",
            "-" if debug["overlap_start"] is None else debug["overlap_start"],
            "-" if debug["S_c"] is None else int(sum(debug["S_c"])),
            len(debug["S"]),
            debug["pick_with_momentum"],
            debug["pick_without_momentum"],
            debug["momentum_ms"],
        )
        return debug

    # ------------------------------------------------------------------
    # Planner interface
    # ------------------------------------------------------------------

    def on_session_start(self, observations: ObservationView) -> None:
        """0.1.4b: a new rollout starts with no previous plan."""
        with self._momentum_lock:
            self._momentum_prev.pop(observations.session_uuid, None)

    def on_session_close(self, session_uuid: str) -> None:
        """Drop this rollout's cadence flag. Safe to call for an unknown uuid."""
        with self._cadence_lock:
            self._cadence_checked.discard(session_uuid)
        with self._momentum_lock:
            self._momentum_prev.pop(session_uuid, None)

    def plan(
        self,
        observations: ObservationView,
        time_now_us: int,
        time_query_us: int,
    ) -> PlanResult | None:
        selected = self._select_frames(observations)
        if selected is None:
            # Not ready: fewer than two front-camera frames at roughly 0.5 s
            # apart. The scaffold turns None into the contract's legal empty
            # trajectory (INTERFACE.md §4.8).
            return None
        older, newer = selected

        features = self.build_features(observations, older, newer)
        if not self._momentum_on:
            # Switch off / 0.1.4 path: identical call, identical result.
            output = self._run_model(features)
            # model_t0_us is the newest frame's timestamp, not time_now_us.
            return self.build_plan_result(output, model_t0_us=newer[0])

        # 0.1.4b: the same features plus one key. build_features' own dict is
        # left as 0.1.4 built it; the key rides on a shallow copy.
        request = self._momentum_request(observations, features, int(newer[0]))
        output = self._run_model({**features, K.MOMENTUM_FEATURE_KEY: request})
        debug = self._momentum_update(observations.session_uuid, request, output)
        return self.build_plan_result(
            output, model_t0_us=newer[0], extra_debug=debug
        )
