# SPDX-License-Identifier: Apache-2.0
"""ReplanDivider: run a 2 Hz planner on the shared 20 Hz referee clock.

Under the shared referee every model is scored on one clock --
``control_timestep_us == frame_interval_us == 50_000``, ``n_sim_steps == 400``.
Drive-JEPA is a 2 Hz agent (``K.FRAME_SPACING_US``, ``K.STEP_US``), so the
adapter divides: the harness asks every 50 ms, real inference happens on every
tenth ask, and the nine ticks in between answer from the last plan.

Two things make that safe rather than merely cheap, and both are load-bearing:

    the GRID        keyed on elapsed sim time, not on a counter (see below)
    the RE-ANCHOR   a cached plan is never returned verbatim (see below)

**Why elapsed time and not a counter.** ``ObservationView.drive_count`` looks
like a tick index and is not one: the scaffold increments it *after* the "no
egomotion cached yet" early return (scaffold.py:630-638), so a tick that never
reached the planner does not advance it. A modulo counter also shifts phase
permanently if the harness ever skips a query, and double-counts if it ever
repeats one. Elapsed sim time has neither failure: a skipped tick is caught up
by the ``while`` below without moving the grid, and a repeated ``time_query_us``
cannot cross the threshold twice.

**Why the grid lands on the handover tick for free.** ``first_policy_timestamp_us``
== ``render_start_us`` == the first camera frame's shutter close
(unbound_rollout.py:95,104; sensorsim_service.py:139-142), and
``closed_loop_start_us = render_start_us + force_gt_duration_us``
(unbound_rollout.py:108). With force-GT pinned at 3.0 s -- a whole number of
500 ms periods -- a grid anchored on the session's first ``drive`` hits
``render_start + 6 * 500_000`` exactly, which IS the first closed-loop tick.
So warm-up queries must be *counted*, not skipped: skipping them would put the
first real replan somewhere after handover instead of on it.

The anchor is the first query rather than absolute-time modulo because
``first_policy_timestamp_us`` is a raw clip timestamp with no useful alignment
(``5322242706587`` in the 0.1.3 reference run; ``% 500_000 == 206_587``).

**Why the cached plan is re-anchored and re-timed.** The MPC samples a
reference by absolute time and would tolerate a stale ``t0``
(nonlinear_mpc.py:315-330). The frame handoff above it would not. The runtime
expresses the plan in the rig frame at ``step_start_us``
(events/controller.py:123-126) and the controller un-expresses it using its own
pose at ``ref.timestamps_us[0]`` (system.py:342-350, the variable is named
``pose_rig_now_to_rig_at_traj_time``). Those agree only when
``model_t0_us == step_start_us``. Returning a 450 ms-stale plan verbatim leaves
a spurious ``C(t0)·R⁻¹`` in the composition -- the ego's displacement over the
staleness, applied a second time, ~11 m backwards at highway speed.

So every tick, replan or not, reports ``model_t0_us = time_now_us`` and hands
back waypoints expressed in the rig frame at that instant. The contract holds
on all ten ticks, and ``interpolate_pose`` is never asked for a timestamp
outside the controller's own trajectory (utils_rs ``__init__.pyi:439-446`` --
out of range RAISES, so a coverage gap is a mid-run crash, not a bad plan).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from paddock.adapters.benchmarks.alpasim.scaffold import ObservationView, Planner, PlanResult

from . import constants as K
from .planner import _pose_to_matrix, _yaw_of

logger = logging.getLogger(__name__)


@dataclass
class _SessionCache:
    """Per-rollout divider state. Never shared between sessions."""

    #: Next sim time at which a real inference is due. None until the first ask.
    next_replan_us: int | None = None
    #: The instant the cached plan was built from (its original model_t0_us).
    t0_us: int | None = None
    #: Pose of the rig in the local frame at ``t0_us``, 4x4. The frame the
    #: cached waypoints are expressed in.
    anchor_local_to_rig: np.ndarray | None = None
    #: (T, 3) cached waypoint positions, rig frame at ``t0_us``.
    positions: np.ndarray | None = None
    #: (T, 3, 3) cached waypoint rotations, same frame. None means "keep heading".
    rotations: np.ndarray | None = None
    #: Waypoint spacing of the cached plan.
    step_us: int = K.STEP_US
    #: Debug payload of the plan this cache came from.
    debug: dict = field(default_factory=dict)
    #: How many real inferences this rollout has run. Reported in debug.
    replans: int = 0
    #: How many asks were answered from cache, and the worst staleness seen.
    #: Reported at session close so a smoke run can gate on the divider having
    #: actually taken the cached path rather than replanning every tick.
    cached: int = 0
    max_age_us: int = 0


def _nearest_pose(observations: ObservationView, at_us: int) -> np.ndarray:
    """4x4 local->rig pose nearest ``at_us``, mirroring the scaffold's anchor.

    The scaffold resolves an anchor as "cached pose nearest ``model_t0_us``"
    (scaffold.py:722-728). The divider has to pick the same pose or its
    re-anchor would be composed against a different frame than the one the
    response is finally hung off.
    """
    nearest = min(
        observations.poses, key=lambda p: abs(int(p.timestamp_us) - int(at_us))
    )
    return _pose_to_matrix(nearest.pose)


def _yaw_rotation(yaw: float) -> np.ndarray:
    """3x3 rotation about z. Matches build_plan_result's construction (§5.2)."""
    cos, sin = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


class ReplanDivider(Planner):
    """Wrap a planner so it replans on its own grid, not the harness's.

    Delegates to ``inner`` on grid ticks and answers from cache otherwise. The
    short-circuit is here, above ``inner.plan``, so a cached tick never reaches
    ``build_features`` -- no JPEG decode, no image pipelines, and no bridge
    traffic (``bridge.forward`` is reachable only through ``_run_model``, which
    ``plan`` calls after ``build_features``).
    """

    def __init__(
        self,
        inner: Planner,
        *,
        replan_period_us: int = K.REPLAN_PERIOD_US,
        max_lookahead_us: int = K.MAX_CONTROLLER_LOOKAHEAD_US,
    ) -> None:
        """
        Args:
            inner: The planner that actually runs the model.
            replan_period_us: Grid spacing. Drive-JEPA's native period.
            max_lookahead_us: How far ahead the controller may sample a plan.
                A rebuilt plan that cannot cover it is logged loudly -- see
                :meth:`_rebuild`.
        """
        self._inner = inner
        self._replan_period_us = int(replan_period_us)
        self._max_lookahead_us = int(max_lookahead_us)
        self._sessions: dict[str, _SessionCache] = {}
        # plan() runs outside the scaffold's session lock and two rollouts may
        # be inside it at once (scaffold.py:246-251). One rollout has at most
        # one drive in flight, so this guards the dict, not the arithmetic.
        self._lock = threading.Lock()

    #: Reported by get_version. Derived from the inner planner so the divider
    #: can never claim a version the model code does not.
    @property
    def version_id(self) -> str:  # type: ignore[override]
        return getattr(self._inner, "version_id", "unnamed-planner-0.0.0")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def on_session_start(self, observations: ObservationView) -> None:
        with self._lock:
            self._sessions[observations.session_uuid] = _SessionCache()
        self._inner.on_session_start(observations)

    def on_session_close(self, session_uuid: str) -> None:
        with self._lock:
            cache = self._sessions.pop(session_uuid, None)
        if cache is not None:
            logger.info(
                "divider: session %s closed after %d real inferences, "
                "%d cached asks, max cache age %d us",
                session_uuid,
                cache.replans,
                cache.cached,
                cache.max_age_us,
            )
        self._inner.on_session_close(session_uuid)

    # ------------------------------------------------------------------
    # The grid
    # ------------------------------------------------------------------

    def _due(self, session_uuid: str, time_query_us: int) -> tuple[bool, _SessionCache]:
        """Is a real inference due at this ask? Advances the grid if so.

        Catch-up rather than reset: if the harness skipped ticks, every missed
        slot is stepped over in one go and the grid keeps its original phase.
        Resetting to ``time_query_us + period`` would let skips accumulate into
        permanent drift, which is the whole reason this is not a counter.
        """
        with self._lock:
            cache = self._sessions.setdefault(session_uuid, _SessionCache())
            if cache.next_replan_us is None:
                # Anchor on the session's first ask, warm-up included.
                cache.next_replan_us = int(time_query_us)
            due = int(time_query_us) >= cache.next_replan_us
            if due:
                while cache.next_replan_us <= int(time_query_us):
                    cache.next_replan_us += self._replan_period_us
            return due, cache

    # ------------------------------------------------------------------
    # Re-anchor + re-time
    # ------------------------------------------------------------------

    def _knots(self, cache: _SessionCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached plan as (times, positions, yaws), anchor included as knot 0.

        The anchor is a real knot: in the rig frame the plan was built in, the
        ego is at the origin with zero heading at ``t0``. Including it is what
        lets a rebuilt plan be sampled between ``t0`` and the first waypoint.
        """
        positions = np.asarray(cache.positions, dtype=np.float64)
        count = len(positions)
        times = np.array(
            [cache.t0_us + k * cache.step_us for k in range(count + 1)],
            dtype=np.int64,
        )
        knot_positions = np.zeros((count + 1, 3), dtype=np.float64)
        knot_positions[1:] = positions

        knot_yaws = np.zeros(count + 1, dtype=np.float64)
        if cache.rotations is not None:
            knot_yaws[1:] = [_yaw_of(r) for r in cache.rotations]
        # Unwrap so interpolation takes the short way round even if the plan
        # crosses +-pi. Knot 0 is the 0 rad anchor, so unwrapping from there.
        knot_yaws = np.unwrap(knot_yaws)
        return times, knot_positions, knot_yaws

    def _rebuild(
        self, cache: _SessionCache, observations: ObservationView, time_now_us: int
    ) -> PlanResult | None:
        """Re-express and re-time the cached plan for ``time_now_us``.

        Returns a plan whose ``model_t0_us`` is ``time_now_us`` and whose
        waypoints sit on ``time_now_us + j * step_us`` in the CURRENT rig
        frame -- so the scaffold, the runtime and the controller all compose
        against the same instant. See the module docstring.
        """
        if cache.positions is None or cache.t0_us is None:
            return None

        times, knot_positions, knot_yaws = self._knots(cache)
        horizon_end_us = int(times[-1])
        age_us = int(time_now_us) - int(cache.t0_us)
        if age_us < 0:
            # Sim time went backwards. Not something to paper over.
            logger.warning(
                "divider: cached plan is from the future (t0=%d, now=%d); dropping",
                cache.t0_us,
                time_now_us,
            )
            return None

        # How many whole steps of the cached plan still lie ahead of us.
        count = int((horizon_end_us - int(time_now_us)) // cache.step_us)
        if count <= 0:
            logger.warning(
                "divider: cached plan expired (t0=%d, horizon_end=%d, now=%d)",
                cache.t0_us,
                horizon_end_us,
                time_now_us,
            )
            return None

        targets = np.array(
            [int(time_now_us) + j * cache.step_us for j in range(1, count + 1)],
            dtype=np.int64,
        )
        covered_us = int(targets[-1]) - int(time_now_us)
        if covered_us < self._max_lookahead_us:
            # Not fatal here -- the MPC clips its sample times into the range it
            # is handed (nonlinear_mpc.py:325-327) -- but it means the tail of
            # the horizon is being held flat, which is worth seeing in a log.
            logger.warning(
                "divider: rebuilt plan covers only %d us, controller may sample "
                "to %d us (t0=%d, now=%d)",
                covered_us,
                self._max_lookahead_us,
                cache.t0_us,
                time_now_us,
            )

        # Linear in position, linear in unwrapped yaw. The cached knots are
        # 0.5 s apart and the model's own output grid is the same, so this only
        # ever interpolates within one step.
        interp_x = np.interp(targets, times, knot_positions[:, 0])
        interp_y = np.interp(targets, times, knot_positions[:, 1])
        interp_z = np.interp(targets, times, knot_positions[:, 2])
        interp_yaw = np.interp(targets, times, knot_yaws)

        # Re-express: rig-at-t0 -> rig-at-now. LEFT multiplication, matching
        # utils_rs Trajectory.transform (__init__.pyi:363-372, "otherwise left
        # multiplication (transform @ pose)") and the composition the runtime
        # and controller perform. Flipping this order is exactly the bug the
        # re-anchor exists to avoid.
        pose_now = _nearest_pose(observations, time_now_us)
        delta = np.linalg.inv(pose_now) @ cache.anchor_local_to_rig

        positions = np.zeros((count, 3), dtype=np.float64)
        rotations = np.zeros((count, 3, 3), dtype=np.float64)
        for index in range(count):
            waypoint = np.eye(4, dtype=np.float64)
            waypoint[:3, :3] = _yaw_rotation(float(interp_yaw[index]))
            waypoint[:3, 3] = (
                float(interp_x[index]),
                float(interp_y[index]),
                float(interp_z[index]),
            )
            reanchored = delta @ waypoint
            positions[index] = reanchored[:3, 3]
            rotations[index] = reanchored[:3, :3]

        cache.cached += 1
        cache.max_age_us = max(cache.max_age_us, age_us)
        debug = dict(cache.debug)
        debug.update(
            {
                "replan": False,
                "cache_age_us": age_us,
                "source_t0_us": int(cache.t0_us),
                "waypoints": count,
            }
        )
        return PlanResult(
            positions_in_rig=positions,
            step_us=cache.step_us,
            rotations_in_rig=rotations,
            # The contract: t0 IS this tick, so the runtime's rig frame and the
            # controller's re-expression agree and the spurious factor is I.
            model_t0_us=int(time_now_us),
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Planner interface
    # ------------------------------------------------------------------

    def plan(
        self,
        observations: ObservationView,
        time_now_us: int,
        time_query_us: int,
    ) -> PlanResult | None:
        due, cache = self._due(observations.session_uuid, time_query_us)

        if not due:
            # THE SHORT CIRCUIT. Nothing below inner.plan runs: no frame
            # selection, no JPEG decode, no model, no bridge.
            return self._rebuild(cache, observations, time_now_us)

        result = self._inner.plan(
            observations, time_now_us=time_now_us, time_query_us=time_query_us
        )
        if result is None:
            # Not ready. The grid has already advanced, so phase is kept and
            # the next replan is still on the 0.5 s lattice -- which is what
            # puts a real inference on the handover tick.
            return None

        with self._lock:
            cache.replans += 1
            cache.t0_us = int(result.model_t0_us)
            cache.anchor_local_to_rig = _nearest_pose(
                observations, int(result.model_t0_us)
            )
            cache.positions = np.array(result.positions_in_rig, dtype=np.float64)
            cache.rotations = (
                None
                if result.rotations_in_rig is None
                else np.array(result.rotations_in_rig, dtype=np.float64)
            )
            cache.step_us = int(result.step_us)
            cache.debug = dict(result.debug or {})
            replans = cache.replans

        result.debug = dict(result.debug or {})
        result.debug.update({"replan": True, "cache_age_us": 0, "replans": replans})

        # One line per REAL inference -- ~40 per rollout at the referee cadence,
        # not 400. A smoke run gates the divider on these: consecutive
        # sim_time_us values must be exactly one replan period apart.
        logger.info(
            "divider: replan #%d at sim_time_us=%d model_t0_us=%d",
            replans,
            int(time_now_us),
            int(result.model_t0_us),
        )

        # A fresh plan is already anchored at the frame it was built from, and
        # at the referee cadence that frame IS this tick -- the camera clock and
        # the policy clock share an origin and interval (types.py:38-58;
        # unbound_rollout.py:95,104), and CameraFrameEvent runs before
        # PolicyEvent at the same timestamp (events/base.py:27-39). The
        # PushCadenceError check in the planner is what keeps that true.
        return result
