# SPDX-License-Identifier: Apache-2.0
"""A fixed replan rhythm for any node, with an optional padded ego history.

The plain `PolicyPlanner` asks the node for a plan at every tick. A model that
plans at a lower rate than the simulator's control step (AutoE2E: 10 Hz, against
AlpaSim's 20 Hz) would then run its forward pass twice as often as it was built
for, and a forward pass that takes seconds makes a scene take hours. This planner
asks on a grid instead and answers the ticks in between from the last plan:

    warm-up (before --force-gt-us)   every tick's frame -> /observe, no plan
    on the grid, after the warm-up   /infer, the plan cached
    between grid points              the cached plan, re-timed onto now and
                                     re-anchored to the car's current pose

The grid is anchored on the session's first ask and advances by
``replan_every_us`` in query time, the way Drive-JEPA's rhythm keeps its 2 Hz
grid (`drivejepa_rhythm.DriveJepaPlanner._due`). The re-timed, re-anchored answer
is the same computation as Drive-JEPA's `_rebuild`, built from the same geometry
helpers so the two cannot drift apart. Drive-JEPA's own planner is not involved:
its frame pairing, speed and command handling stay in `drivejepa_rhythm`.

**Observing between replans** (``observe_between_replans``). Off, a node hears
from the bridge only at replans once the warm-up is over. A node that keeps its
own history at the control step (Qwen-Drive picks frames 0.5 s apart and ego
rows 0.1 s apart from what it was sent) then finds holes where the ticks between
replans were. On, every such tick's body goes to /observe as well, before the
cached plan is answered.

Warm-up frames go to /observe because AlpaSim replays the recorded drive during
the warm-up and ignores the driver's plan (`events/controller.py`, the force-GT
branch); a forward pass there would only cost time. The node still receives
every frame, so its image and motion history are full at handover.

**Padded ego history** (``pad_ego_history``). A node that builds its motion
history from the poses it was sent (AutoE2E's 64 rows reach 6.4 s back) sees
almost no past at handover, because AlpaSim starts each scene a fraction of a
second into its clip. With padding on, right after `/reset` and before the
first real frame, the node is sent ego-only frames (no images) running straight
back from that frame's pose along its heading, at the speed measured from the
last two poses, one per control step back to `PAD_HISTORY_US`. The speed is
measured from poses because AlpaSim reports a velocity of zero for the forced
replay ticks. A node that builds its history by finite differences reads those
frames as constant speed with no acceleration or turning. The first real frame
sits at the join, where a centred difference takes one neighbour from the
padding: there AutoE2E's node reads the true speed, half the true yaw rate, and
(the padding's speed being the 50 ms before that frame) three quarters of the
true acceleration (tests/test_replan_rhythm.py). Every later row is unchanged.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.alpasim.bridge import PolicyPlanner
from paddock.adapters.benchmarks.alpasim.drivejepa_rhythm import (
    nearest_pose,
    pose_to_matrix,
    yaw_of,
    yaw_rotation,
)
from paddock.adapters.benchmarks.alpasim.scaffold import PlanResult
from paddock.policy.protocol import CANBUS_VELOCITY

logger = logging.getLogger("alpasim.bridge.rhythm")

#: How far back the padded ego history reaches. AutoE2E's oldest motion row is
#: 6.4 s back and its server keeps frames 6.6 s (one row more, for the centred
#: difference), so padding to 6.6 s fills every row.
PAD_HISTORY_US = 6_600_000
#: Rows of motion history the padding is counted against in the replan log
#: (AutoE2E: 64 rows, 0.1 s apart, oldest 6.4 s back).
EGO_ROWS = 64
EGO_ROW_US = 100_000


@dataclass
class CachedPlan:
    """The last plan the node returned, in the rig frame at its own time."""

    t0_us: int
    anchor_local_to_rig: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    step_us: int


@dataclass
class _Session:
    next_replan_us: int | None = None
    first_now_us: int | None = None
    #: Time of the first real frame the node was sent; None until then.
    first_frame_us: int | None = None
    plan: CachedPlan | None = None
    replans: int = 0
    reused: int = 0
    observed: int = 0


def rebuild_plan(plan: CachedPlan, view: Any, time_now_us: int) -> PlanResult | None:
    """*plan* re-timed onto now + k * step and re-expressed in the rig frame at now.

    The same computation as `DriveJepaPlanner._rebuild`: the cached waypoints are
    knots in time from the plan's own t0 (the car at the origin), interpolated at
    now + k * step for every k still inside the plan's horizon, then moved from
    the rig frame at t0 into the rig frame at the pose nearest now. None when the
    plan is from the future or has run out.
    """
    count_cached = len(plan.positions)
    times = np.array([plan.t0_us + k * plan.step_us for k in range(count_cached + 1)], dtype=np.int64)
    knot_positions = np.zeros((count_cached + 1, 3), dtype=np.float64)
    knot_positions[1:] = plan.positions
    knot_yaws = np.zeros(count_cached + 1, dtype=np.float64)
    knot_yaws[1:] = [yaw_of(r) for r in plan.rotations]
    knot_yaws = np.unwrap(knot_yaws)

    if int(time_now_us) < int(plan.t0_us):
        logger.warning("cached plan is from the future (t0=%d, now=%d); dropping", plan.t0_us, time_now_us)
        return None
    count = int((int(times[-1]) - int(time_now_us)) // plan.step_us)
    if count <= 0:
        logger.warning("cached plan expired (t0=%d, now=%d)", plan.t0_us, time_now_us)
        return None
    targets = np.array([int(time_now_us) + j * plan.step_us for j in range(1, count + 1)], dtype=np.int64)
    interp = [np.interp(targets, times, knot_positions[:, axis]) for axis in range(3)]
    interp_yaw = np.interp(targets, times, knot_yaws)

    pose_now = pose_to_matrix(nearest_pose(view, time_now_us).pose)
    delta = np.linalg.inv(pose_now) @ plan.anchor_local_to_rig
    positions = np.zeros((count, 3), dtype=np.float64)
    rotations = np.zeros((count, 3, 3), dtype=np.float64)
    for index in range(count):
        waypoint = np.eye(4, dtype=np.float64)
        waypoint[:3, :3] = yaw_rotation(float(interp_yaw[index]))
        waypoint[:3, 3] = (float(interp[0][index]), float(interp[1][index]), float(interp[2][index]))
        reanchored = delta @ waypoint
        positions[index] = reanchored[:3, 3]
        rotations[index] = reanchored[:3, :3]
    return PlanResult(
        positions_in_rig=positions,
        step_us=plan.step_us,
        rotations_in_rig=rotations,
        model_t0_us=int(time_now_us),
    )


def pose_speed(view: Any) -> float | None:
    """Ground speed from the last two poses (displacement over their time gap), or None."""
    if len(view.poses) < 2:
        return None
    newer, older = view.poses[-1], view.poses[-2]
    dt_s = (int(newer.timestamp_us) - int(older.timestamp_us)) / 1e6
    if dt_s <= 0.0:
        return None
    dx = float(newer.pose.vec.x) - float(older.pose.vec.x)
    dy = float(newer.pose.vec.y) - float(older.pose.vec.y)
    return math.hypot(dx, dy) / dt_s


def padding_bodies(body: dict[str, Any], speed_m_s: float, step_us: int) -> list[dict[str, Any]]:
    """Ego-only bodies straight back from *body*'s pose along its heading, oldest first.

    One every *step_us*, from one step before *body* back to `PAD_HISTORY_US`, at
    *speed_m_s*, each with *body*'s orientation and calibration and no images.
    """
    ego2world = np.asarray(body["ego2world"], dtype=np.float64)
    heading = yaw_of(ego2world)
    direction = np.array([math.cos(heading), math.sin(heading)])
    canbus = [0.0] * len(body["canbus"])
    canbus[CANBUS_VELOCITY] = float(speed_m_s)
    out = []
    for k in range(PAD_HISTORY_US // step_us, 0, -1):
        pose = ego2world.copy()
        pose[:2, 3] = ego2world[:2, 3] - direction * speed_m_s * (k * step_us / 1e6)
        out.append(
            {
                "images": {},
                "ego2world": pose.tolist(),
                "canbus": canbus,
                "timestamp": int(body["timestamp"]) - k * step_us,
                "command": body["command"],
                "calibration": body["calibration"],
            }
        )
    return out


class ReplanRhythmPlanner(PolicyPlanner):
    """`PolicyPlanner` asked on a fixed grid, warm-up frames observed (module docstring)."""

    def __init__(
        self,
        policy_url: str,
        *,
        replan_every_us: int,
        force_gt_us: int,
        frame_interval_us: int,
        pad_ego_history: bool = False,
        observe_between_replans: bool = False,
        **translation: Any,
    ) -> None:
        super().__init__(policy_url, **translation)
        if frame_interval_us <= 0 or replan_every_us <= 0 or replan_every_us % frame_interval_us:
            # Off a multiple, grid points fall between asks and the rhythm drifts
            # against the control step instead of landing on every n-th tick.
            raise ValueError(
                f"--replan-every-us {replan_every_us} must be a positive multiple of "
                f"--frame-interval-us {frame_interval_us}"
            )
        self._replan_every_us = int(replan_every_us)
        self._force_gt_us = int(force_gt_us)
        self._frame_interval_us = int(frame_interval_us)
        self._pad = bool(pad_ego_history)
        self._observe_between = bool(observe_between_replans)
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
                "replan rhythm: session %s closed after %d replans, %d reused, %d observed",
                session_uuid,
                session.replans,
                session.reused,
                session.observed,
            )

    # -- the grid -----------------------------------------------------------

    def _due(self, session: _Session, time_query_us: int) -> bool:
        """A replan due at this ask? Anchored on the first ask; skipped slots caught up."""
        with self._lock:
            if session.next_replan_us is None:
                session.next_replan_us = int(time_query_us)
            due = int(time_query_us) >= session.next_replan_us
            if due:
                while session.next_replan_us <= int(time_query_us):
                    session.next_replan_us += self._replan_every_us
            return due

    def padded_rows(self, session: _Session, frame_us: int) -> int:
        """Motion-history rows of a call at *frame_us* that fall before the first real frame."""
        if session.first_frame_us is None:
            return 0
        return sum(frame_us - k * EGO_ROW_US < session.first_frame_us for k in range(1, EGO_ROWS + 1))

    # -- the one method the scaffold calls -------------------------------------

    def plan(self, view: Any, time_now_us: int, time_query_us: int) -> Any:
        with self._lock:
            session = self._sessions.setdefault(view.session_uuid, _Session())
            if session.first_now_us is None:
                session.first_now_us = int(time_now_us)
        in_warmup = int(time_now_us) < int(session.first_now_us) + self._force_gt_us
        due = self._due(session, time_query_us)
        row = {"tick": int(view.drive_count), "sim_time_us": int(time_now_us), "query_us": int(time_query_us)}

        if not in_warmup and not due:
            observed = {}
            if self._observe_between:
                between, between_t_us = self.build_infer_body(view, time_now_us)
                observed = {"sent_as": self.observe_frame(between), "frame_us": int(between_t_us)}
                session.observed += 1
            result = rebuild_plan(session.plan, view, time_now_us) if session.plan is not None else None
            session.reused += 1
            self.log_row(
                {
                    **row,
                    "kind": "reuse",
                    "model_called": False,
                    "plan_age_us": None if session.plan is None else int(time_now_us) - session.plan.t0_us,
                    "plan_waypoints": 0 if result is None else len(result.positions_in_rig),
                    "plan_dt_s": None if result is None else result.step_us / 1e6,
                    **observed,
                }
            )
            return result

        body, source_t_us = self.build_infer_body(view, time_now_us)
        if session.first_frame_us is None:
            if self._pad and not self._send_padding(session, view, body, row):
                return None
            session.first_frame_us = int(source_t_us)

        if in_warmup:
            sent_as = self.observe_frame(body)
            session.observed += 1
            self.log_row(
                {
                    **row,
                    "kind": "observe",
                    "model_called": False,
                    "sent_as": sent_as,
                    "frame_us": int(source_t_us),
                    "padded_ego_rows": self.padded_rows(session, int(source_t_us)),
                }
            )
            return None

        answer = self.post_infer(body)
        trajectory, step_us, headings = self.plan_from_answer(answer)
        result = self.plan_result(trajectory, step_us, int(source_t_us))
        session.plan = CachedPlan(
            t0_us=int(result.model_t0_us),
            anchor_local_to_rig=pose_to_matrix(nearest_pose(view, result.model_t0_us).pose),
            positions=np.array(result.positions_in_rig, dtype=np.float64),
            rotations=np.array(result.rotations_in_rig, dtype=np.float64),
            step_us=int(result.step_us),
        )
        session.replans += 1
        extras = answer.get("extras") or {}
        model = next((v for v in extras.values() if isinstance(v, dict)), {})
        self.log_row(
            {
                **row,
                "kind": "infer",
                "model_called": True,
                "replan": session.replans,
                "frame_us": int(source_t_us),
                "padded_ego_rows": self.padded_rows(session, int(source_t_us)),
                # As the node reports them; None for a node that does not.
                "repeated_image_frames": model.get("history_repeated"),
                "zero_ego_rows": model.get("ego_rows_zero"),
                "plan_waypoints": len(trajectory),
                "plan_dt_s": step_us / 1e6,
                "headings": headings,
                "extras": extras,
                "trajectory_rig": trajectory.tolist(),
            }
        )
        return result

    def _send_padding(self, session: _Session, view: Any, body: dict[str, Any], row: dict[str, Any]) -> bool:
        """The padded frames, before the first real one. False if no speed can be measured yet."""
        speed = pose_speed(view)
        if speed is None:
            # Two poses are needed for a speed; the next ask will have them. The
            # first real frame waits too, so the padding still comes before it.
            self.log_row({**row, "kind": "pad_wait", "model_called": False, "poses": len(view.poses)})
            return False
        bodies = padding_bodies(body, speed, self._frame_interval_us)
        for padded in bodies:
            self.observe_frame(padded)
        self.log_row(
            {
                **row,
                "kind": "pad",
                "model_called": False,
                "padded_frames": len(bodies),
                "speed_m_s": speed,
                "oldest_us": int(bodies[0]["timestamp"]),
                "newest_us": int(bodies[-1]["timestamp"]),
            }
        )
        return True
