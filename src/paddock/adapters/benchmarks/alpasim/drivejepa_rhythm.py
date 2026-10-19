# SPDX-License-Identifier: Apache-2.0
"""Drive-JEPA's native AlpaSim driver logic, ported into the bridge.

Drive-JEPA's baseline AlpaSim numbers come from its native driver (adapter
0.1.4b), which talks to the simulator directly. Through `paddock run` the model
sits behind the generic policy node instead, and the node only sees what the
bridge sends it. Everything that driver did around the model -- when it plans,
which two frames it pairs, what speed and command it reports, what it answers
between plans -- is therefore the bridge's job, and this module does it the same
way, function for function. The model node gets settings only.

What is copied, and from where in that driver:

    the 2 Hz replan grid       ReplanDivider._due      keyed on elapsed sim time
    the cached-plan answer      ReplanDivider._rebuild  re-timed and re-anchored
    the frame pair              _select_frames          newest + nearest 0.5 s back,
                                                        +-10 % or ContextSpacingError
    the push-cadence check      _assert_push_cadence    once per session
    velocity at handover        _newest_velocity        pose-derived when the
                                                        reported velocity is exactly 0
    forward speed and accel     _build_ego_status       vx and ax, rig frame
    the driving command         _command_one_hot        20 m ahead, +-3 m
    the plan's headings         build_plan_result       one rotation per waypoint

What differs, because the node is not the native driver:

* **Scene start.** The node builds its frame pair from the frames it has been
  sent, and it is sent one only with a plan request. So the session's first
  replan-grid ask -- where the native driver had one frame and planned nothing --
  sends that frame and throws the plan away, after a `/reset` that clears the
  previous scene's frames. The first real plan is 0.5 s later, on the native
  driver's first replan. The node's momentum stage will have stored the
  thrown-away plan, so that first replan is compared with it where the native
  driver had no previous plan.

* **Sideways speed and acceleration.** The native driver gave the model and
  its momentum stage (vx, vy) and (ax, ay). The node's input carries forward
  speed and acceleration only (canbus), so the model gets (vx, 0) and (ax, 0).
  On a scene with real sideways motion the two can pick differently; the
  measured vy and ay are logged beside what was sent so the comparison can
  tell that gap from anything else.

The plan driven is the node's ``plan`` -- all eight (x, y, heading) poses, read
the way the bridge reads every model's (`PolicyPlanner.plan_from_answer`), so
the headings are the model's own. Every re-anchored answer between replans is
built from it. The node's ``extras`` (Drive-JEPA's pick, its score, and whether
a previous plan was fed) are logged with each replan as the node sent them.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.alpasim.bridge import (
    PolicyPlanner,
    calibration_from,
    canbus_from,
    decode_frame,
)
from paddock.adapters.benchmarks.alpasim.scaffold import PlanResult
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command, encode_image

logger = logging.getLogger("alpasim.bridge.drivejepa")

#: The replan grid and the spacing of both the frame pair and the plan's
#: waypoints: Drive-JEPA is a 2 Hz agent.
REPLAN_PERIOD_US = 500_000
FRAME_SPACING_US = 500_000
STEP_US = 500_000
#: Hard limit on the frame pair's spacing. At 50 ms pushes only the 500 ms pair
#: passes (+-50 ms).
CONTEXT_SPACING_TOLERANCE = 0.10
#: The handover fix replaces an exactly-zero reported velocity only when the
#: pose-derived speed is above this.
HANDOVER_MIN_SPEED_M_S = 1.0
#: Route -> command: the first route waypoint at least this far away decides it.
MIN_LOOKAHEAD_M = 20.0
COMMAND_LATERAL_M = 3.0
#: How far ahead the controller samples a plan; a rebuilt plan shorter than this
#: is logged.
MAX_CONTROLLER_LOOKAHEAD_US = 2_000_000
#: Frames the scaffold must keep for the 500 ms pair to exist at 50 ms pushes.
IMAGE_HISTORY = 16

#: The native driver's one-hot rows, as the node's Command enum: the node maps
#: each back to exactly that row (policy.COMMAND_ONE_HOT).
_COMMAND_LEFT = Command.LEFT
_COMMAND_STRAIGHT = Command.STRAIGHT
_COMMAND_RIGHT = Command.RIGHT
_COMMAND_UNKNOWN = Command.FOLLOW_REFERENCE


class ContextSpacingError(RuntimeError):
    """The frame pair at a replan is not 0.5 s +-10 % apart."""


class PushCadenceError(RuntimeError):
    """Frames arrive at a different interval from the control step."""


# ---------------------------------------------------------------------------
# geometry, as the native driver computes it
# ---------------------------------------------------------------------------


def pose_to_matrix(pose: Any) -> np.ndarray:
    """4x4 from a `common.Pose`; the proto's quaternion is w-first."""
    q, v = pose.quat, pose.vec
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


def yaw_of(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def relative_pose(pose: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(dx, dy, dyaw) of *pose* in *reference*'s frame."""
    delta = np.linalg.inv(reference) @ pose
    return np.array([delta[0, 3], delta[1, 3], yaw_of(delta)], dtype=np.float64)


def yaw_rotation(yaw: float) -> np.ndarray:
    cos, sin = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def nearest_pose(view: Any, at_us: int) -> Any:
    """The cached `PoseAtTime` nearest *at_us* -- the scaffold's own anchor rule."""
    return min(view.poses, key=lambda p: abs(int(p.timestamp_us) - int(at_us)))


# ---------------------------------------------------------------------------
# the inputs the native driver gave the model
# ---------------------------------------------------------------------------


def command_from_route(route: Any) -> Command:
    """Route waypoints (rig frame) -> left / straight / right / unknown.

    The first waypoint at least 20 m away decides: beyond +-3 m in y it is a
    turn, rig +y being left. No route, or none of it 20 m away, as there.
    """
    if route is None or len(route.waypoints) == 0:
        return _COMMAND_UNKNOWN
    target = None
    for waypoint in route.waypoints:
        if float(np.hypot(waypoint.x, waypoint.y)) >= MIN_LOOKAHEAD_M:
            target = waypoint
            break
    if target is None:
        return _COMMAND_STRAIGHT
    if target.y > COMMAND_LATERAL_M:
        return _COMMAND_LEFT
    if target.y < -COMMAND_LATERAL_M:
        return _COMMAND_RIGHT
    return _COMMAND_STRAIGHT


def newest_velocity(view: Any) -> tuple[float, float, bool]:
    """Rig-frame (vx, vy) for the newest observation, and whether it was derived.

    The reported velocity, verbatim -- except that a velocity of exactly zero
    beside poses that are visibly moving is the simulator's warm-up "unknown",
    and the displacement of the last two poses over their time gap is used
    instead, when that is faster than 1 m/s.
    """
    state = view.dynamic_states[-1][1]
    reported = (float(state.linear_velocity.x), float(state.linear_velocity.y))
    if reported != (0.0, 0.0):
        return reported[0], reported[1], False
    if len(view.poses) < 2:
        return reported[0], reported[1], False
    dt_s = (int(view.poses[-1].timestamp_us) - int(view.poses[-2].timestamp_us)) / 1e6
    if dt_s <= 0.0:
        return reported[0], reported[1], False
    current = pose_to_matrix(view.poses[-1].pose)
    previous = relative_pose(pose_to_matrix(view.poses[-2].pose), current)
    dx, dy = -float(previous[0]), -float(previous[1])
    derived = (dx / dt_s, dy / dt_s)
    if float(np.hypot(*derived)) <= HANDOVER_MIN_SPEED_M_S:
        return reported[0], reported[1], False
    return derived[0], derived[1], True


def reported_acceleration(view: Any) -> tuple[float, float]:
    state = view.dynamic_states[-1][1]
    return float(state.linear_acceleration.x), float(state.linear_acceleration.y)


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------


@dataclass
class _Session:
    """Per-rollout state. Never shared between sessions."""

    next_replan_us: int | None = None
    first_now_us: int | None = None
    #: The frame the node was last sent, which is the one it will pair next.
    sent_frame_us: int | None = None
    cadence_checked: bool = False
    t0_us: int | None = None
    anchor_local_to_rig: np.ndarray | None = None
    positions: np.ndarray | None = None
    rotations: np.ndarray | None = None
    step_us: int = STEP_US
    replans: int = 0
    cached: int = 0
    max_age_us: int = 0


class DriveJepaPlanner(PolicyPlanner):
    """`PolicyPlanner` with the native driver's rhythm and inputs (module docstring)."""

    def __init__(
        self,
        policy_url: str,
        *,
        front_camera: str,
        expected_frame_interval_us: int,
        force_gt_us: int,
        **translation: Any,
    ) -> None:
        super().__init__(policy_url, waypoint_dt=STEP_US / 1e6, **translation)
        self._front_camera = front_camera
        self._expected_frame_interval_us = int(expected_frame_interval_us)
        self._force_gt_us = int(force_gt_us)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def on_session_start(self, observations: Any) -> None:
        with self._lock:
            self._sessions[observations.session_uuid] = _Session()
        super().on_session_start(observations)

    def on_session_close(self, session_uuid: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_uuid, None)
        if session is not None:
            logger.info(
                "drivejepa rhythm: session %s closed after %d replans, %d cached asks, max cache age %d us",
                session_uuid,
                session.replans,
                session.cached,
                session.max_age_us,
            )

    # -- the grid -----------------------------------------------------------

    def _due(self, session: _Session, time_query_us: int) -> bool:
        """Is a replan due at this ask? Anchored on the session's first ask,
        warm-up included; skipped slots are caught up without moving the grid."""
        with self._lock:
            if session.next_replan_us is None:
                session.next_replan_us = int(time_query_us)
            due = int(time_query_us) >= session.next_replan_us
            if due:
                while session.next_replan_us <= int(time_query_us):
                    session.next_replan_us += REPLAN_PERIOD_US
            return due

    # -- frames -------------------------------------------------------------

    def _check_cadence(self, session: _Session, frames: tuple) -> None:
        if session.cadence_checked:
            return
        session.cadence_checked = True
        observed = int(frames[-1][0]) - int(frames[-2][0])
        expected = self._expected_frame_interval_us
        slack = expected * CONTEXT_SPACING_TOLERANCE
        if abs(observed - expected) > slack:
            raise PushCadenceError(
                f"camera {self._front_camera!r} is pushing frames {observed} us apart, but the "
                f"bridge was launched for a {expected} us control step (+-{slack:.0f} us). "
                "frame_interval_us and control_timestep_us must be equal."
            )

    def _select_frames(self, session: _Session, view: Any) -> tuple | None:
        """(older, newer), None when the buffer cannot yet reach 0.5 s back,
        ContextSpacingError when it can and the nearest frame is off the grid."""
        frames = view.image_cache.get(self._front_camera, ())
        if len(frames) < 2:
            return None
        self._check_cadence(session, frames)
        newest = frames[-1]
        target = newest[0] - FRAME_SPACING_US
        if int(frames[0][0]) > target:
            return None
        older = min(frames[:-1], key=lambda entry: abs(entry[0] - target))
        gap = newest[0] - older[0]
        low = FRAME_SPACING_US * (1.0 - CONTEXT_SPACING_TOLERANCE)
        high = FRAME_SPACING_US * (1.0 + CONTEXT_SPACING_TOLERANCE)
        if not (low <= gap <= high):
            raise ContextSpacingError(
                f"context pair is {gap} us apart; the model wants {FRAME_SPACING_US} us "
                f"+-{CONTEXT_SPACING_TOLERANCE:.0%} (accepted [{low:.0f}, {high:.0f}]); "
                f"buffer holds {len(frames)} frames spanning {int(newest[0]) - int(frames[0][0])} us"
            )
        return older, newest

    # -- the body -----------------------------------------------------------

    def _body(
        self,
        view: Any,
        frame: tuple,
        speed: float,
        accel: float,
        command: Command,
    ) -> dict:
        """An `/infer` body for *frame*, stamped with the frame's own time.

        The pose sent is the cached pose nearest the frame -- the pose the plan
        is anchored on, which is also the pose the native driver handed its
        momentum stage. Only the front camera is sent from AlpaSim; the
        translation adds the placeholders. canbus carries the forward speed
        and acceleration, which is all of the motion the node reads.
        """
        frame_us, payload = int(frame[0]), frame[1]
        pixels = decode_frame(payload)
        images = {self._front_camera: encode_image(pixels)}
        shapes = {self._front_camera: pixels.shape[:2]}
        calibration = calibration_from(view, pinhole_intrinsics=self._pinhole_intrinsics)
        images, calibration = self._translate_cameras(images, shapes, calibration)
        canbus = canbus_from(view)
        canbus[CANBUS_VELOCITY] = speed
        canbus[CANBUS_ACCEL] = accel
        return {
            "images": images,
            "ego2world": pose_to_matrix(nearest_pose(view, frame_us).pose).tolist(),
            "canbus": canbus.tolist(),
            "timestamp": frame_us,
            "command": int(command),
            "calibration": calibration,
        }

    # -- the planner interface -----------------------------------------------

    def plan(self, view: Any, time_now_us: int, time_query_us: int) -> Any:
        with self._lock:
            session = self._sessions.setdefault(view.session_uuid, _Session())
            if session.first_now_us is None:
                session.first_now_us = int(time_now_us)
        if not self._due(session, time_query_us):
            return self._rebuild(session, view, time_now_us)

        frames = view.image_cache.get(self._front_camera, ())
        if session.sent_frame_us is None:
            # The session's first replan-grid ask. The native driver had one
            # frame here and planned nothing; the node needs that frame for the
            # next pair, so it is sent and the plan is thrown away.
            if not frames:
                return None
            vx, vy, derived = newest_velocity(view)
            ax, ay = reported_acceleration(view)
            command = command_from_route(view.route)
            self.post_infer(self._body(view, frames[-1], vx, ax, command))
            session.sent_frame_us = int(frames[-1][0])
            self.log_row(
                {
                    "kind": "store_first_frame",
                    "sim_time_us": int(time_now_us),
                    "query_us": int(time_query_us),
                    "frame_us": session.sent_frame_us,
                }
            )
            return None

        selected = self._select_frames(session, view)
        if selected is None:
            return None
        older, newer = selected
        if int(older[0]) != session.sent_frame_us:
            # The node pairs the new frame with the last one it was sent. If
            # that is not the frame the native driver would pair, the model
            # sees a different pair; refuse rather than plan on it.
            raise ContextSpacingError(
                f"the node holds frame {session.sent_frame_us} as the older frame, but the pair "
                f"for this replan is ({int(older[0])}, {int(newer[0])})"
            )

        vx, vy, derived = newest_velocity(view)
        ax, ay = reported_acceleration(view)
        command = command_from_route(view.route)
        answer = self.post_infer(self._body(view, newer, vx, ax, command))
        session.sent_frame_us = int(newer[0])
        trajectory, step_us, headings = self.plan_from_answer(answer)
        result = self.plan_result(trajectory, step_us, int(newer[0]))

        anchor = pose_to_matrix(nearest_pose(view, result.model_t0_us).pose)
        with self._lock:
            session.replans += 1
            session.t0_us = int(result.model_t0_us)
            session.anchor_local_to_rig = anchor
            session.positions = np.array(result.positions_in_rig, dtype=np.float64)
            session.rotations = np.array(result.rotations_in_rig, dtype=np.float64)
            session.step_us = int(result.step_us)
            replans = session.replans
        in_warmup = int(time_now_us) < int(session.first_now_us) + self._force_gt_us
        local = [(anchor @ np.array([p[0], p[1], 0.0, 1.0]))[:2].tolist() for p in trajectory]
        self.log_row(
            {
                "kind": "replan",
                "replan": replans,
                "sim_time_us": int(time_now_us),
                "query_us": int(time_query_us),
                "phase": "warm-up" if in_warmup else "after handover",
                "frame_us": int(newer[0]),
                "older_frame_us": int(older[0]),
                "gap_us": int(newer[0]) - int(older[0]),
                "speed_sent": vx,
                "accel_sent": ax,
                # Measured, not sent: the node's input has no sideways terms.
                "vy_measured": vy,
                "ay_measured": ay,
                "speed_derived_at_handover": derived,
                "command": command.name,
                "headings": headings,
                "extras": answer.get("extras"),
                "trajectory_rig": trajectory.tolist(),
                "trajectory_local_xy": local,
            }
        )
        logger.info(
            "drivejepa rhythm: replan #%d at sim_time_us=%d model_t0_us=%d",
            replans,
            time_now_us,
            result.model_t0_us,
        )
        return result

    # -- between replans -------------------------------------------------------

    def _rebuild(self, session: _Session, view: Any, time_now_us: int) -> PlanResult | None:
        """The cached plan, re-timed onto now + k * 0.5 s and re-expressed in the
        rig frame at now, with model_t0_us = now."""
        if session.positions is None or session.t0_us is None:
            return None
        count_cached = len(session.positions)
        times = np.array(
            [session.t0_us + k * session.step_us for k in range(count_cached + 1)], dtype=np.int64
        )
        knot_positions = np.zeros((count_cached + 1, 3), dtype=np.float64)
        knot_positions[1:] = session.positions
        knot_yaws = np.zeros(count_cached + 1, dtype=np.float64)
        if session.rotations is not None:
            knot_yaws[1:] = [yaw_of(r) for r in session.rotations]
        knot_yaws = np.unwrap(knot_yaws)

        horizon_end_us = int(times[-1])
        age_us = int(time_now_us) - int(session.t0_us)
        if age_us < 0:
            logger.warning(
                "cached plan is from the future (t0=%d, now=%d); dropping", session.t0_us, time_now_us
            )
            return None
        count = int((horizon_end_us - int(time_now_us)) // session.step_us)
        if count <= 0:
            logger.warning(
                "cached plan expired (t0=%d, horizon_end=%d, now=%d)",
                session.t0_us,
                horizon_end_us,
                time_now_us,
            )
            return None
        targets = np.array(
            [int(time_now_us) + j * session.step_us for j in range(1, count + 1)], dtype=np.int64
        )
        covered_us = int(targets[-1]) - int(time_now_us)
        if covered_us < MAX_CONTROLLER_LOOKAHEAD_US:
            logger.warning(
                "rebuilt plan covers only %d us, controller may sample to %d us (t0=%d, now=%d)",
                covered_us,
                MAX_CONTROLLER_LOOKAHEAD_US,
                session.t0_us,
                time_now_us,
            )
        interp_x = np.interp(targets, times, knot_positions[:, 0])
        interp_y = np.interp(targets, times, knot_positions[:, 1])
        interp_z = np.interp(targets, times, knot_positions[:, 2])
        interp_yaw = np.interp(targets, times, knot_yaws)

        pose_now = pose_to_matrix(nearest_pose(view, time_now_us).pose)
        delta = np.linalg.inv(pose_now) @ session.anchor_local_to_rig
        positions = np.zeros((count, 3), dtype=np.float64)
        rotations = np.zeros((count, 3, 3), dtype=np.float64)
        for index in range(count):
            waypoint = np.eye(4, dtype=np.float64)
            waypoint[:3, :3] = yaw_rotation(float(interp_yaw[index]))
            waypoint[:3, 3] = (float(interp_x[index]), float(interp_y[index]), float(interp_z[index]))
            reanchored = delta @ waypoint
            positions[index] = reanchored[:3, 3]
            rotations[index] = reanchored[:3, :3]
        session.cached += 1
        session.max_age_us = max(session.max_age_us, age_us)
        return PlanResult(
            positions_in_rig=positions,
            step_us=session.step_us,
            rotations_in_rig=rotations,
            model_t0_us=int(time_now_us),
        )
