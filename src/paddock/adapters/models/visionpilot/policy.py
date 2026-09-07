"""NeuroNCAP model node for Autoware's Vision Pilot.

Adapted from the NeuroNCAP model node. The planner reimplementation below is
unchanged; the seams now implement `Policy`, and the ONNX weights arrive via
model.yaml rather than a hardcoded default.

Vision Pilot is an L2 assist stack, not a planner, and it is the only one of the
four models here that never produces a metric trajectory.  Three ONNX models do
the perceiving:

    AutoSteer   64 ego-path points, in *image* space -- a normalised lateral
                offset at each of 64 fixed image rows, plus a per-point mask
    AutoDrive   distance to the closest in-path object (0-150 m), road
                curvature, and the probability that such an object exists
    AutoSpeed   2D boxes after NMS, each carrying a *threat class*

Everything below the networks is ported from the C++ rather than reinvented:
``LongitudinalFusion`` (fusion/src/longitudinal_fusion.cpp), the IDM in
``LongitudinalPlanner`` (planning/src/longitudinal_planning.cpp), and the speed
schedule in ``Planner::compute_plan`` (planning/src/planning.cpp).

*Threat classes are the in-path test.*  AutoSpeed's four classes are not object
categories, they are threat levels: 1 = CIPO Level 1 (same lane), 2 = Level 2
(adjacent lane, potential cut-in), 3 = Level 3, 0 = other.  ``select_cipo``
keeps only 1 and 2, takes the closest of each, and prefers the Level 2 when it
is nearer -- which is how it names a cut-in.  Do not substitute a geometric
corridor test for this; the network's own lane assignment is the stack's
opinion about what is in its path, and replacing it measures a different system.

*Image -> ground.*  ``LongitudinalFusion`` projects each box's bottom-centre
through a homography ``H_``, hard-coded from the Zenseact Open Dataset and
marked DO NOT MODIFY, with a ``set_H`` setter provided for exactly this case --
"call once when AutoSpeed runs on a non-BEV image".  Rather than fit one,
NeuroNCAP sends exact intrinsics and the camera-to-ego transform every request,
so each pixel's ray is intersected with the road plane analytically.  Measured
against logged actor poses on scene-0103 that recovers range to 0.9-2.1 m.

*The gap and the closing speed come out of the particle filter*, not out of
differencing.  The filter carries [distance, velocity] over 500 particles and
reports a weighted posterior mean of both, which is the whole reason it exists;
its header notes "no EMA, no finite difference, no deadband needed".

One genuine inconsistency in the C++ has to be resolved rather than copied.
``cipo_v`` is documented in both headers as the lead's *absolute* speed
("speed_limit when free road"), and ``Planner::compute_plan`` updates the
horizon gap as ``gap += (cipo_v - v_plan) * dt``, which is only correct for an
absolute speed.  But ``LongitudinalFusion`` emits a *relative* velocity
("negative = approaching"), and ``compute_acceleration`` assigns
``delta_v = cipo_v`` directly, which is only correct for a relative one.  Those
two cannot both hold.  This node keeps the fusion's relative velocity and uses
the physical definitions on both sides: ``delta_v = -v_rel`` (positive when
closing, as IDM requires) and ``gap += v_rel * dt``.  Copying either call site
verbatim breaks the other -- taken literally, ``compute_acceleration`` never
brakes for a stationary lead, since a parked car gives ``delta_v = 0``.

Still not reproduced: the CTE/yaw particle filter in ``LateralFusion``.  Lateral
control here is the fitted AutoSteer path, and the horizon is stretched from the
stack's own 1.0 s (N=20 at dt=0.05) to the 3.0 s NeuroNCAP asks for, so the tail
of every trajectory is an extrapolation of a controller that never plans that
far.  Say so when reporting the number.

The honest framing for the result: this system can brake but cannot swerve
around an obstacle.  On NeuroNCAP it can therefore only earn the 5 points for
avoidance by stopping short, never by going around -- which is exactly what a
real NCAP autonomous-braking test measures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from paddock.policy.protocol import Observation, Pinhole, Policy, Trajectory

NET_W, NET_H = 1024, 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

D_MAX_M = 150.0  # AutoDrive's distance normalisation, from auto_drive.hpp
CURV_SCALE = 1.0  # curvature_raw -> 1/m; auto_drive.hpp leaves this to the caller
CIPO_PROB_MIN = 0.40  # longitudinal_fusion.cpp: below this AutoDrive does not confirm a CIPO
STEER_MASK_THRESHOLD = 0.5  # h_vector gate, from lateral_fusion

AUTOSPEED_CONF = 0.6  # auto_speed.hpp default conf_thres
AUTOSPEED_IOU = 0.45  # auto_speed.hpp default iou_thres
LETTERBOX_PAD = 114  # the grey auto_speed.hpp pads with
CIPO_CLASSES = (1, 2)  # select_cipo keeps Level 1 and Level 2 only
MAX_CIPO_RANGE_M = 60.0  # beyond this a bottom-edge projection is not trustworthy

# AutoSteer points near the horizon project to enormous ranges -- one pixel of
# error is worth tens of metres out there -- and an unweighted quadratic lets a
# single one of them set the shape of the whole path.
MAX_FIT_RANGE_M = 50.0

CONTROL_DT = 0.05  # planning.cpp: `double dt = 0.05`, the 50 Hz control period
# Eight waypoints at 0.5 s: NAVSIM's grid. NeuroNCAP takes the first six, and
# those are unchanged by the extension -- `_speed_schedule` integrates the IDM
# forward at a fixed CONTROL_DT, so the first 60 steps are identical whether the
# loop runs 60 or 80 times, and the six interpolation targets fall inside them.
#
# Laterally the fourth second is thinner evidence than the first three: the ego
# can travel past `MAX_FIT_RANGE_M`, beyond which `PathFit.lateral_at` continues
# on the tangent rather than running the quadratic out. That is VisionPilot's
# own designed behaviour, not something added for NAVSIM, but it does mean a
# far waypoint rests on extrapolated geometry from a single forward camera.
PLAN_HORIZON_S = 4.0
WAYPOINT_DT = 0.5


def ort_threads() -> int:
    """Intra-op threads per ORT session from VISIONPILOT_ORT_THREADS; 0 = ORT default."""
    raw = os.environ.get("VISIONPILOT_ORT_THREADS", "").strip()
    return int(raw) if raw.isdigit() else 0


# ─── LongitudinalFusion, ported from fusion/src/longitudinal_fusion.cpp ───────


@dataclass
class Meas:
    distance_m: float = 0.0
    stddev_m: float = 15.0
    valid: bool = False


@dataclass
class CIPOFusionEstimate:
    valid: bool = False
    distance_m: float = 0.0
    velocity_ms: float = 0.0  # negative = approaching, from the particle ensemble
    distance_stddev_m: float = 0.0
    cipo_raw_found: bool = False
    cipo_raw_dist_m: float = 0.0
    cut_in_detected: bool = False
    has_cipo: bool = False


class LongitudinalFusion:
    """500-particle filter over [distance, velocity], fusing AutoDrive with AutoSpeed.

    A direct port. The filter is the reason the stack does not need to
    difference anything: velocity is a state it estimates, so the closing speed
    comes out of the posterior rather than out of two consecutive distances.

    The RNG is seeded rather than drawn from ``random_device`` as the C++ does.
    These runs are a paired comparison across four models with a per-run seed,
    and a node that answers differently on a re-run would make that pairing a
    lie.
    """

    N_PARTICLES = 500
    # Shipped value. Its own comment reads "Keep small (MRPT uses 0.03 m). Large
    # values cause particle drift -> bad velocity", and that is measurably true:
    # against a lead closing at a constant 12 m/s with a perfect distance
    # measurement, the posterior velocity reaches only -4.3 m/s after 4 s at
    # 2.0, and -12.1 m/s within 2.4 s at 0.3. The distance is tracked exactly
    # either way -- it is only the velocity state that is starved, and that is
    # what IDM's dynamic term runs on. Left at the shipped value by default so
    # the score measures VisionPilot as delivered; --process-noise-dist 0.3
    # measures it as its author evidently intended.
    PROCESS_NOISE_DIST_M = 2.0
    PROCESS_NOISE_VEL_MS = 0.50
    AUTODRIVE_NOISE_MIN_M = 1.5  # noise floor when AutoDrive is fully confident
    AUTODRIVE_NOISE_M = 8.0  # noise ceiling at minimum AutoDrive confidence
    CIPO_NOISE_M = 3.0  # AutoSpeed bbox-projected distance noise
    RESET_GATE_M = 25.0  # a jump this far past the cloud is a real cut-in/cut-out

    def __init__(self, seed: int = 0, process_noise_dist_m: float | None = None) -> None:
        self._seed = seed
        if process_noise_dist_m is not None:
            self.PROCESS_NOISE_DIST_M = process_noise_dist_m
        self.reset()

    def reset(self) -> None:
        self.distance = np.empty(0)
        self.velocity = np.empty(0)
        self.log_w = np.empty(0)
        self.initialised = False
        self.rng = np.random.default_rng(self._seed)

    def _init_from(self, dist_m: float, stddev_m: float) -> None:
        n = self.N_PARTICLES
        self.distance = np.clip(self.rng.normal(dist_m, stddev_m, n), 0.0, D_MAX_M)
        self.velocity = self.rng.normal(0.0, 2.0, n)
        self.log_w = np.zeros(n)

    def _predict(self, dt_s: float) -> None:
        n = self.N_PARTICLES
        self.distance = np.clip(
            self.distance + self.velocity * dt_s + self.rng.normal(0.0, self.PROCESS_NOISE_DIST_M, n),
            0.0,
            D_MAX_M,
        )
        self.velocity = self.velocity + self.rng.normal(0.0, self.PROCESS_NOISE_VEL_MS, n)

    def _weight_update(self, ad: Meas, cipo_raw: Meas) -> None:
        for meas in (ad, cipo_raw):
            if meas.valid:
                self.log_w += -0.5 * ((meas.distance_m - self.distance) / meas.stddev_m) ** 2

    def _linear_weights(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        total = w.sum()
        return np.full_like(w, 1.0 / len(w)) if total < 1e-12 else w / total

    def _resample(self) -> None:
        """Systematic resampling, as in the C++."""
        n = self.N_PARTICLES
        cumulative = np.cumsum(self._linear_weights())
        thresholds = self.rng.uniform(0.0, 1.0 / n) + np.arange(n) / n
        idx = np.searchsorted(cumulative, thresholds).clip(0, n - 1)
        self.distance = self.distance[idx]
        self.velocity = self.velocity[idx]
        self.log_w = np.zeros(n)

    @staticmethod
    def select_cipo(dets: np.ndarray, ranges: np.ndarray) -> tuple[Meas, bool]:
        """Closest Level 1 and Level 2 detection; Level 2 wins when nearer (cut-in).

        ``dets`` is (N, 6) with the class in column 5; ``ranges`` is the ground
        range of each box's bottom-centre. Anything that is not Level 1 or
        Level 2 is not a CIPO candidate at all -- that class filter is the
        stack's own in-path test.
        """
        best = {1: np.inf, 2: np.inf}
        for j in range(len(dets)):
            cls = int(dets[j, 5])
            if cls in CIPO_CLASSES and np.isfinite(ranges[j]) and 0.0 < ranges[j] < MAX_CIPO_RANGE_M:
                best[cls] = min(best[cls], float(ranges[j]))

        have_l1, have_l2 = np.isfinite(best[1]), np.isfinite(best[2])
        if have_l2 and have_l1 and best[2] < best[1]:
            return Meas(best[2], LongitudinalFusion.CIPO_NOISE_M, True), True
        if have_l1:
            return Meas(best[1], LongitudinalFusion.CIPO_NOISE_M, True), False
        if have_l2:
            return Meas(best[2], LongitudinalFusion.CIPO_NOISE_M, True), False
        return Meas(), False

    def update(
        self,
        autodrive_dist_norm: float,
        autodrive_flag_prob: float,
        dets: np.ndarray,
        ranges: np.ndarray,
        dt_s: float,
    ) -> CIPOFusionEstimate:
        est = CIPOFusionEstimate()

        cipo_raw, est.cut_in_detected = self.select_cipo(dets, ranges)
        if cipo_raw.valid:
            est.cipo_raw_found = True
            est.cipo_raw_dist_m = cipo_raw.distance_m

        ad_confirmed = autodrive_flag_prob >= CIPO_PROB_MIN
        if not ad_confirmed and not cipo_raw.valid:
            # Neither network confirms a target: free road, and the filter is
            # dropped so a reappearing target starts clean.
            if self.initialised:
                self.reset()
            est.valid, est.distance_m, est.has_cipo = True, D_MAX_M, False
            return est

        ad_meas = Meas()
        if ad_confirmed:
            ad_meas.distance_m = D_MAX_M * (1.0 - autodrive_dist_norm)
            p = float(np.clip(autodrive_flag_prob, 0.0, 1.0))
            ad_meas.stddev_m = self.AUTODRIVE_NOISE_MIN_M + (
                self.AUTODRIVE_NOISE_M - self.AUTODRIVE_NOISE_MIN_M
            ) * (1.0 - p)
            ad_meas.valid = True

        if not self.initialised:
            if ad_meas.valid:
                self._init_from(ad_meas.distance_m, ad_meas.stddev_m)
            else:
                self._init_from(cipo_raw.distance_m, self.CIPO_NOISE_M)
            self.initialised = True
        else:
            self._predict(dt_s)
            cloud_mean = float(self.distance.mean())
            if cipo_raw.valid and abs(cipo_raw.distance_m - cloud_mean) > self.RESET_GATE_M:
                self._init_from(cipo_raw.distance_m, self.CIPO_NOISE_M)  # target change
            elif ad_meas.valid and (ad_meas.distance_m - cloud_mean) > self.RESET_GATE_M:
                self._init_from(ad_meas.distance_m, ad_meas.stddev_m)  # cut-out

        self._weight_update(ad_meas, cipo_raw)
        w = self._linear_weights()
        if 1.0 / (np.sum(w**2) + 1e-12) < 0.5 * self.N_PARTICLES:
            self._resample()
            w = self._linear_weights()

        mean_d = float(w @ self.distance)
        est.valid = True
        est.has_cipo = True
        est.distance_m = mean_d
        est.velocity_ms = float(w @ self.velocity)
        est.distance_stddev_m = float(np.sqrt(max(0.0, w @ (self.distance - mean_d) ** 2)))
        return est


# ─── LongitudinalPlanner, ported from planning/src/longitudinal_planning.cpp ──


class IDM:
    """Vision Pilot's longitudinal planner, ported from longitudinal_planning.cpp."""

    def __init__(
        self,
        speed_limit: float = 33.3,
        a: float = 1.5,
        b: float = 3.0,
        t_headway: float = 1.5,
        s0: float = 2.0,
        delta: float = 4.0,
        mu: float = 0.2,
        g: float = 9.81,
    ) -> None:
        self.speed_limit, self.a, self.b = speed_limit, a, b
        self.T, self.s0, self.delta = t_headway, s0, delta
        self.mu, self.g = mu, g

    def acceleration(self, kappa: float, ego_v: float, has_cipo: bool, closing_v: float, gap: float) -> float:
        """``closing_v`` is positive when the gap is shrinking.

        The C++ writes ``delta_v = cipo_v`` and is fed the fusion's relative
        velocity, which is negative when approaching -- so taken literally the
        dynamic term has the wrong sign and a stationary lead produces no
        braking demand at all. IDM's Δv is ``v_ego - v_lead``, and that is what
        is used here; the caller negates the fusion's velocity to get it.
        """
        curv_v_max = np.sqrt(self.mu * self.g / abs(kappa)) if abs(kappa) > 1e-9 else np.inf
        speed_limit = min(self.speed_limit, curv_v_max)

        # max(0, ...) is load-bearing and is in the C++: without it a gap that is
        # opening shrinks s_star below s0, collapses the interaction term, and
        # IDM outputs full free-road acceleration.
        dynamic = (ego_v * closing_v) / (2.0 * np.sqrt(self.a * self.b))
        s_star = self.s0 + max(0.0, ego_v * self.T + dynamic)
        s = max(0.5, gap)

        free_road = (ego_v / speed_limit) ** self.delta if speed_limit > 0 else 1.0
        interaction = (s_star / s) ** 2 if has_cipo else 0.0
        return float(self.a * (1.0 - free_road - interaction))


# ─── geometry ─────────────────────────────────────────────────────────────────


def project_to_ground(
    uv: np.ndarray, pinhole: Pinhole, camera2ego: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Image points -> (x forward, y left) metres on the road plane.

    Each pixel defines a ray from the camera centre; the road is the z = 0 plane
    of the ego frame (NeuroNCAP zeroes the ego pose's z every step, so the ego
    frame sits on the ground).  Returns the points and a mask of which rays
    actually descend toward the plane -- those at or above the horizon do not
    meet it, and their entries are meaningless.
    """
    if len(uv) == 0:
        return np.empty((0, 2)), np.zeros(0, dtype=bool)
    dirs_cam = np.stack(
        [
            (uv[:, 0] - pinhole.cx) / pinhole.fx,
            (uv[:, 1] - pinhole.cy) / pinhole.fy,
            np.ones(len(uv)),
        ],
        axis=1,
    )
    dirs_ego = dirs_cam @ camera2ego[:3, :3].T
    origin = camera2ego[:3, 3]

    descending = dirs_ego[:, 2] < -1e-6
    scale = np.where(descending, -origin[2] / np.where(descending, dirs_ego[:, 2], -1.0), np.nan)
    points = origin[None, :] + scale[:, None] * dirs_ego
    return points[:, :2], descending


@dataclass
class PathFit:
    """The fitted ego path, and where the evidence for it stops."""

    coeffs: np.ndarray | None
    x_max: float

    def lateral_at(self, x: np.ndarray) -> np.ndarray:
        """Lateral offset at each longitudinal distance, without extrapolating the fit.

        ``lateral_fusion`` records the forward extent of its inliers precisely so
        downstream users do not sample the quadratic past where the evidence
        stops. That matters here: a quadratic run far beyond its data bends away
        hard -- a straight road came back curving 6 m off centre at 44 m. Past
        the fitted extent the path continues on the tangent instead.
        """
        arc = np.asarray(x, dtype=float)
        if self.coeffs is None:
            return np.zeros_like(arc)
        y = np.polyval(self.coeffs, np.minimum(arc, self.x_max))
        beyond = arc > self.x_max
        if beyond.any():
            slope = np.polyval(np.polyder(self.coeffs), self.x_max)
            y[beyond] = np.polyval(self.coeffs, self.x_max) + slope * (arc[beyond] - self.x_max)
        return y


# ─── AutoSpeed post-processing, ported from models/src/auto_speed.cpp ─────────


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order):
        keep.append(int(order[0]))
        if len(order) == 1:
            break
        order = order[1:][_iou(boxes[order[0]], boxes[order[1:]]) <= iou_thres]
    return keep


class VisionPilotPolicy(Policy):
    cameras = ("CAM_FRONT",)
    waypoint_dt = WAYPOINT_DT

    def __init__(
        self,
        weights: Path | str | None = None,
        providers: list[str] | None = None,
        process_noise_dist_m: float | None = None,
    ) -> None:
        # Supplied by model.yaml from {repo}; the ONNX files live inside the
        # VisionPilot checkout rather than the shared weights root.
        self.weights = Path(weights or os.environ.get("VISIONPILOT_WEIGHTS", ""))
        # CPU by default: the closed-loop runs used it, and the renderer wants
        # the GPU. Set VISIONPILOT_PROVIDERS=cuda to change that.
        self.providers = providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if os.environ.get("VISIONPILOT_PROVIDERS", "cpu").lower().startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        # The shipped 2.0 starves the filter's velocity state; 0.3 lets it
        # identify closing speed within ~2.4 s. See LongitudinalFusion.
        env_noise = os.environ.get("VISIONPILOT_PROCESS_NOISE_DIST")
        self.process_noise_dist_m = (
            process_noise_dist_m
            if process_noise_dist_m is not None
            else (float(env_noise) if env_noise else None)
        )
        self.steer = self.drive = self.speed = None
        self.idm = IDM()
        self.fusion = LongitudinalFusion(process_noise_dist_m=self.process_noise_dist_m)
        self._diagnostics: dict[str, Any] = {}
        self.reset()

    def load(self) -> None:
        # Checked before the import it guards: otherwise a missing-weights
        # configuration error only surfaces on a machine that already has
        # onnxruntime, and cannot be tested at all.
        if not self.weights or not self.weights.exists():
            raise RuntimeError(
                f"VisionPilot ONNX weights not found at {self.weights or '<unset>'}. "
                "model.yaml supplies VISIONPILOT_WEIGHTS from {repo}; run its setup.sh "
                "if the checkout has not been built."
            )

        import onnxruntime as ort

        opts = ort.SessionOptions()
        # onnxruntime's intra-op pool defaults to every core and ignores
        # OMP_NUM_THREADS. One node on an idle box, that is fine. Four nodes on a
        # shared box, each spinning 32 threads, measured at ~16 cores between
        # them for no more throughput than one node -- they thrash each other
        # and starve everything else. Unset leaves ORT's default, so recorded
        # single-node runs are unchanged; set it when running several nodes.
        threads = ort_threads()
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        # CUDA: be frugal on a shared GPU. ORT's defaults grow the arena freely and
        # run an exhaustive cuDNN conv-algorithm search that wants large
        # workspaces; with other tenants on the card that failed at the first
        # inference (bfc_arena alloc). Cap the per-session arena, grow it only as
        # requested, and pick conv algorithms heuristically. Same maths, less VRAM.
        mem_mb = int(os.environ.get("VISIONPILOT_GPU_MEM_MB", "1536"))
        cuda_opts = {
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": mem_mb * 1024 * 1024,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }
        providers = [(p, cuda_opts) if p == "CUDAExecutionProvider" else p for p in self.providers]
        self.steer = ort.InferenceSession(
            str(self.weights / "autosteer_fp32.onnx"), opts, providers=providers
        )
        self.drive = ort.InferenceSession(
            str(self.weights / "autodrive_fp32.onnx"), opts, providers=providers
        )
        self.speed = ort.InferenceSession(
            str(self.weights / "autospeed_fp32.onnx"), opts, providers=providers
        )

    def reset(self) -> None:
        """Clear all cross-frame state.

        AutoDrive is a two-frame model, so it needs the previous frame; the
        fusion filter carries a distance/velocity posterior; and the cruise
        set-speed is latched from the start of the run.  Carrying any of them
        across a run boundary would leak one run's ending into the next one's
        start.
        """
        self.prev_frame: np.ndarray | None = None
        self.prev_timestamp: int | None = None
        self.set_speed: float | None = None
        self._last_box_ground = np.empty((0, 2))
        self.fusion.reset()
        # Set by the visualiser to have /infer also return the raw intermediates
        # (boxes, lane points, path fit). Off by default: it roughly doubles the
        # response size and nothing in a scored run reads it.
        self.debug_detail = getattr(self, "debug_detail", False)

    # ─── preprocessing ────────────────────────────────────────────────────────

    def _stretch(self, image: np.ndarray) -> np.ndarray:
        """Plain resize, which is what auto_steer.hpp and auto_drive.hpp specify."""
        return np.asarray(Image.fromarray(image).resize((NET_W, NET_H), Image.BILINEAR))

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Aspect-preserving resize with grey padding, as auto_speed.hpp specifies.

        AutoSpeed is the one net of the three that does *not* take a stretch.
        Returns the scale and pad so boxes can be mapped back to source pixels.
        """
        h, w = image.shape[:2]
        scale = min(NET_W / w, NET_H / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
        canvas = np.full((NET_H, NET_W, 3), LETTERBOX_PAD, dtype=np.uint8)
        px, py = (NET_W - nw) // 2, (NET_H - nh) // 2
        canvas[py : py + nh, px : px + nw] = resized
        return canvas, scale, float(px), float(py)

    # ─── the three networks ───────────────────────────────────────────────────

    def _run_steer(self, resized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # auto_steer.hpp: RGB, [0,1], CHW, no ImageNet normalisation.
        chw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        name = self.steer.get_inputs()[0].name
        out = self.steer.run(None, {name: chw})
        return np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel()

    def _run_drive(self, resized: np.ndarray) -> tuple[float, float, float]:
        """Returns (normalised distance, curvature, CIPO probability).

        The distance is left normalised: ``LongitudinalFusion`` is what converts
        it to metres, and it needs the raw value to scale its measurement noise.
        """
        # auto_drive.hpp: RGB, ImageNet-normalised, two frames.
        norm = ((resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        prev = self.prev_frame if self.prev_frame is not None else norm
        names = [i.name for i in self.drive.get_inputs()]
        out = self.drive.run(None, {names[0]: prev, names[1]: norm})
        self.prev_frame = norm
        dist_norm, curvature_raw, flag_logit = (float(np.asarray(o).ravel()[0]) for o in out)
        return dist_norm, curvature_raw * CURV_SCALE, float(1.0 / (1.0 + np.exp(-flag_logit)))

    def _run_speed(self, image: np.ndarray) -> np.ndarray:
        """AutoSpeed boxes in source-image pixels: (N, 6) [x1, y1, x2, y2, score, cls].

        The raw tensor is (1, 4 + K, anchors) with box centre/size first and
        per-class logits after; auto_speed.cpp sigmoids the logits, keeps the
        best class per anchor, thresholds, and runs NMS. That is reproduced
        here, then the letterbox is undone.
        """
        canvas, scale, px, py = self._letterbox(image)
        chw = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        raw = np.asarray(self.speed.run(None, {self.speed.get_inputs()[0].name: chw})[0])[0]

        cx, cy, w, h = raw[0], raw[1], raw[2], raw[3]
        probs = 1.0 / (1.0 + np.exp(-raw[4:]))
        best_prob, best_cls = probs.max(axis=0), probs.argmax(axis=0)

        sel = best_prob >= AUTOSPEED_CONF
        if not sel.any():
            return np.empty((0, 6))
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)[sel]
        scores, classes = best_prob[sel], best_cls[sel]

        keep = _nms(boxes, scores, AUTOSPEED_IOU)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
        return np.column_stack([boxes, scores, classes])

    # ─── ground projection ────────────────────────────────────────────────────

    def _path_points(self, xp: np.ndarray, mask: np.ndarray, obs: Observation) -> np.ndarray:
        """AutoSteer output -> ground points, following lateral_fusion's reading.

        ``xp[i]`` is a normalised lateral position at the i-th of 64 fixed image
        rows; the rows are ``linspace(0, NET_H - 1, 64)``.  Coordinates are in
        the 1024x512 network frame and are scaled back to the rendered image
        before projection.
        """
        image = obs.image("CAM_FRONT")
        keep = mask >= STEER_MASK_THRESHOLD
        if not keep.any():
            return np.empty((0, 2))
        rows = np.linspace(0, NET_H - 1, len(xp))
        uv_net = np.stack([xp * NET_W, rows], axis=1)[keep]
        uv_src = uv_net * np.array([image.shape[1] / NET_W, image.shape[0] / NET_H])
        points, ok = project_to_ground(uv_src, obs.pinhole("CAM_FRONT"), obs.camera2ego["CAM_FRONT"])
        return points[ok]

    def _lane_points_detail(self, xp: np.ndarray, mask: np.ndarray, obs: Observation):
        """The kept AutoSteer points in image pixels and on the ground, for drawing."""
        image = obs.image("CAM_FRONT")
        keep = mask >= STEER_MASK_THRESHOLD
        if not keep.any():
            return np.empty((0, 2)), np.empty((0, 2))
        rows = np.linspace(0, NET_H - 1, len(xp))
        uv_net = np.stack([xp * NET_W, rows], axis=1)[keep]
        uv_src = uv_net * np.array([image.shape[1] / NET_W, image.shape[0] / NET_H])
        ground, ok = project_to_ground(uv_src, obs.pinhole("CAM_FRONT"), obs.camera2ego["CAM_FRONT"])
        return uv_src, ground[ok]

    def _box_ranges(self, dets: np.ndarray, obs: Observation) -> np.ndarray:
        """Ground range of each box's bottom-centre, or NaN if it does not meet the road.

        This is what ``LongitudinalFusion::project_dist`` computes through its
        homography: the norm of the projected bottom-centre. A box clipped by
        the bottom of the frame has no observed contact point and comes out
        short -- the safe direction for an obstacle, and only when it is very
        close already.
        """
        if not len(dets):
            return np.empty(0)
        bottom = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], axis=1)
        points, ok = project_to_ground(bottom, obs.pinhole("CAM_FRONT"), obs.camera2ego["CAM_FRONT"])
        self._last_box_ground = np.where(ok[:, None], points, np.nan)
        return np.where(ok, np.linalg.norm(points, axis=1), np.nan)

    @staticmethod
    def _fit_path(ground: np.ndarray) -> PathFit:
        """Fit ``y = a x^2 + b x + c`` to the near field only.

        Points beyond ``MAX_FIT_RANGE_M`` are dropped and the rest weighted
        toward the camera. Near the horizon a pixel of error is worth tens of
        metres, so an unweighted fit over the full set lets one bad far point
        set the shape of the whole path.
        """
        near = ground[ground[:, 0] <= MAX_FIT_RANGE_M] if len(ground) else ground
        if len(near) < 3:
            return PathFit(coeffs=None, x_max=0.0)
        coeffs = np.polyfit(near[:, 0], near[:, 1], 2, w=1.0 / (1.0 + near[:, 0]))
        return PathFit(coeffs=coeffs, x_max=float(near[:, 0].max()))

    # ─── speed schedule, ported from Planner::compute_plan ────────────────────

    @staticmethod
    def warning_for(acceleration: float) -> str | None:
        """VisionPilot's own FCW/AEB bands, from Planner::compute_plan."""
        if acceleration < -5.0:
            return "AEB"
        if -5.0 <= acceleration <= -3.0:
            return "FCW"
        return None

    def _speed_schedule(
        self, ego_v: float, kappa: float, has_cipo: bool, v_rel: float, gap: float
    ) -> np.ndarray:
        """Cumulative distance travelled at each NeuroNCAP waypoint time.

        ``compute_plan`` integrates ego speed forward at the 50 Hz control
        period, recomputing the IDM acceleration at each step from the *planned*
        speed and the gap as it evolves -- so a braking ego closes the gap more
        slowly, which a fixed-closing-speed integration misses entirely. The
        same loop runs here, at the same ``dt``; only its length changes, from
        the stack's N = 20 (1.0 s) to the ``PLAN_HORIZON_S`` the benchmarks want.

        ``v_rel`` is the fusion's relative velocity, negative when approaching.
        """
        steps = int(round(PLAN_HORIZON_S / CONTROL_DT))
        v_plan, gap_plan, travelled = ego_v, gap, 0.0
        distances = np.empty(steps)
        for i in range(steps):
            accel = self.idm.acceleration(kappa, v_plan, has_cipo, -v_rel, gap_plan)
            v_next = max(0.0, v_plan + accel * CONTROL_DT)
            travelled += 0.5 * (v_plan + v_next) * CONTROL_DT
            if has_cipo:
                # The gap changes at the relative velocity. That already accounts
                # for both vehicles, so the ego's own travel must not be
                # subtracted again on top of it.
                gap_plan = max(0.5, gap_plan + v_rel * CONTROL_DT)
            v_plan = v_next
            distances[i] = travelled

        times = np.arange(1, int(PLAN_HORIZON_S / WAYPOINT_DT) + 1) * WAYPOINT_DT
        return np.interp(times, (np.arange(steps) + 1) * CONTROL_DT, distances)

    # ─── entry point ──────────────────────────────────────────────────────────

    def infer(self, obs: Observation) -> Trajectory:
        if self.steer is None:
            raise RuntimeError("load() must be called before infer()")
        image = obs.image("CAM_FRONT")
        resized = self._stretch(image)

        dt_s = (
            0.1 if self.prev_timestamp is None else max(1e-3, (obs.timestamp_us - self.prev_timestamp) / 1e6)
        )
        self.prev_timestamp = obs.timestamp_us

        xp, h_vector = self._run_steer(resized)
        dist_norm, curvature, flag_prob = self._run_drive(resized)
        dets = self._run_speed(image)

        box_ranges = self._box_ranges(dets, obs)
        cipo = self.fusion.update(dist_norm, flag_prob, dets, box_ranges, dt_s)

        ground = self._path_points(xp, h_vector, obs)
        path = self._fit_path(ground)

        # Latch the cruise set-speed at the speed the run starts from. IDM's
        # configured default is 33.3 m/s (120 km/h), a motorway number; on these
        # urban scenes it would have the car flooring the throttle for reasons
        # that have nothing to do with what the perception models see. The speed
        # the logged driver was holding is the honest set-point for an ACC.
        # Bench2Drive sends the posted road limit: use it as the ACC set-point every
        # tick (curvature still caps it inside IDM). Without one -- NeuroNCAP -- the
        # latch below is wrong for a run that starts from standstill (max(0, 1.0)
        # pinned VisionPilot at 1 m/s for whole Bench2Drive routes), so it is only
        # the fallback.
        if obs.speed_limit is not None and obs.speed_limit > 0:
            self.set_speed = float(obs.speed_limit)
        elif self.set_speed is None:
            self.set_speed = max(obs.speed, 1.0)
        self.idm.speed_limit = self.set_speed

        arc = self._speed_schedule(obs.speed, curvature, cipo.has_cipo, cipo.velocity_ms, cipo.distance_m)
        trajectory = np.stack([arc, path.lateral_at(arc)], axis=1)

        # The acceleration Planner::compute_plan returns for this step: the first
        # IDM evaluation, before the horizon is integrated. This is the number
        # its FCW/AEB bands are tested against.
        accel = self.idm.acceleration(curvature, obs.speed, cipo.has_cipo, -cipo.velocity_ms, cipo.distance_m)

        detail = {}
        if self.debug_detail:
            lane_uv, lane_ground = self._lane_points_detail(xp, h_vector, obs)
            detail = {
                "detections": dets.tolist(),
                "box_ranges": np.where(np.isfinite(box_ranges), box_ranges, -1.0).tolist(),
                "box_ground": np.nan_to_num(self._last_box_ground, nan=-999.0).tolist(),
                "lane_uv": lane_uv.tolist(),
                "lane_ground": lane_ground.tolist(),
                "path_coeffs": None if path.coeffs is None else path.coeffs.tolist(),
                "path_x_max": path.x_max,
            }

        self._diagnostics = {
            "accel_cmd": round(accel, 3),
            "warning": self.warning_for(accel),
            **detail,
            "fused_distance_m": round(cipo.distance_m, 2),
            "fused_velocity_ms": round(cipo.velocity_ms, 2),
            "fused_stddev_m": round(cipo.distance_stddev_m, 2),
            "has_cipo": cipo.has_cipo,
            "cut_in": cipo.cut_in_detected,
            "autospeed_boxes": int(len(dets)),
            "autospeed_cipo_m": round(cipo.cipo_raw_dist_m, 2) if cipo.cipo_raw_found else None,
            "autodrive_m": round(D_MAX_M * (1.0 - dist_norm), 2),
            "autodrive_prob": round(flag_prob, 3),
            "curvature": round(float(curvature), 5),
            "path_points": int(len(ground)),
            "path_extent_m": round(path.x_max, 1),
            "set_speed_ms": round(float(self.set_speed or 0.0), 2),
        }
        return Trajectory(trajectory, dt=self.waypoint_dt)

    def diagnostics(self) -> dict[str, Any]:
        """The planner's internal state from the last step.

        VisionPilot is a reimplementation of a C++ planner, so its intermediates
        are where a wiring error shows up: `fused_distance_m` drifting from
        `autospeed_cipo_m`, or `path_extent_m` collapsing, both indicate a
        problem long before the score does.
        """
        return self._diagnostics or {"status": "no frame seen yet"}


POLICY = VisionPilotPolicy
