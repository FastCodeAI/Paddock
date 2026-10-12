"""Drive-JEPA with the paper's momentum-aware trajectory selection turned on.

Everything up to the 32 scored candidates is `drivejepa` unchanged: same
checkpoint, same input construction, same forward pass
(`DriveJepaPolicy.forward`). Only the pick differs.

**What the step is.** The Drive-JEPA paper's last stage, "Momentum-aware
Trajectory Selection", picks the final trajectory "by accounting for cross-frame
comfort". At the pinned sha it is `DriveJEPAModel.calibrate_score` in
`navsim_v2/.../drive_jepa_model.py`. Each candidate and the plan driven on the
previous frame are both rolled out through NAVSIM's PDM simulator (LQR tracker +
kinematic bicycle, 40 steps at 0.1 s), and the candidate passes if, over the
time the two overlap, the RMS differences of acceleration, jerk, yaw rate and
yaw acceleration all stay under NAVSIM v2's two-frame Extended Comfort
thresholds (0.7, 0.5, 0.1, 0.1). The pass is a 0 or 1, and the score becomes

    (14 * pdm_score + 2 * passed) / 16

before the argmax: a flat bonus of 0.125 for agreeing with the last plan.

**Why `drivejepa` does not do it.** The vendor only runs the step when the
caller hands it `past_ego_simulated_states`, which only NAVSIM v2's sequential
scorer does. Every paddock benchmark calls the agent without it, so every
published `drivejepa` row is the plain argmax. That was deliberate; this entry
exists to measure what the step is worth.

**Why a second registry entry rather than a flag.** It is a different model
from the point of view of every result: a different pick on any tick where the
bonus reorders the top. A flag on `drivejepa` would put both under one results
directory and one palette colour.

Three adaptations.

*The ego state is built, not loaded.* The vendor reads the starting state from
NAVSIM's metric cache. Closed loop has none, so it is built from the
observation: rear axle at the origin of its own frame, measured speed and
acceleration, yaw rate, and the tire angle a bicycle with the simulator's
wheelbase needs for that yaw rate. The comfort features are acceleration
*magnitude*, jerk, yaw rate and yaw acceleration, none of which changes under a
rigid transform, so each frame's rollouts can stay in that frame's own
coordinates.

*The previous plan is the previous tick's.* NAVSIM steps at 0.5 s; neuroNCAP
replans every 0.1 s. "Cross-frame" is taken literally: the bonus compares with
the plan driven one tick ago, over the 3.9 s the two share. The vendor drops the
previous plan when frames are more than 0.55 s apart and so does this.

*The overlap is rounded, not truncated.* The vendor does
`int(observation_interval / 0.1)`, which is right only when the division lands
on or just above an integer; 0.3 / 0.1 is 2.9999999999999996 in floating point.
Here the interval comes from microsecond timestamps, so it is rounded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paddock.adapters.models.drivejepa.policy import HISTORY_DT, DriveJepaPolicy, relative_pose
from paddock.policy.protocol import Observation, Trajectory

#: The vendor's weights: `(14 * pdm_score + 2 * two_frame_comfort) / 16`.
SCORE_WEIGHT = 14.0
COMFORT_WEIGHT = 2.0

#: The simulator's step, and the vendor's cut-off past which the previous plan
#: is dropped (`run_pdm_score_sequential.py`: `observation_interval > 0.55`).
SIM_DT = 0.1
MAX_INTERVAL_S = 0.55

#: Below this the bicycle relation divides by ~0; the tire angle is left at 0.
MIN_STEER_SPEED = 0.5


def momentum_score(score: np.ndarray, passed: np.ndarray) -> np.ndarray:
    """The vendor's blend, in the score's own precision.

    The vendor applies it to a float32 tensor, so it is computed in float32
    here as well; an argmax between two close candidates can otherwise flip on
    the last bit.
    """
    score = np.asarray(score, dtype=np.float32)
    passed = np.asarray(passed, dtype=np.float32)
    return ((SCORE_WEIGHT * score + COMFORT_WEIGHT * passed) / np.float32(16.0)).astype(np.float32)


def overlap_steps(interval_s: float) -> int | None:
    """Simulator steps the previous plan is ahead by, or None if it is unusable."""
    if not 0 < interval_s <= MAX_INTERVAL_S:
        return None
    steps = int(round(interval_s / SIM_DT))
    return steps if steps >= 1 else None


class DriveJepaMomentumPolicy(DriveJepaPolicy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.past: tuple[np.ndarray, int] | None = None
        self.totals = {
            "ticks": 0,
            "with_previous": 0,
            "pick_changed": 0,
            "argmax_failed": 0,
            "passed_sum": 0.0,
        }
        self.episodes: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.past = None
        # One record per episode, in the order the harness ran them.
        if getattr(self, "episodes", None) is not None and (not self.episodes or self.episodes[-1]["ticks"]):
            self.episodes.append({"ticks": 0, "with_previous": 0, "pick_changed": 0})

    # -- the two vendor calls, behind seams the tests can replace ----------------

    def _simulate(self, proposals: np.ndarray, obs: Observation) -> np.ndarray:
        """(32, 41, 11) PDM rollouts of every candidate from the observed state.

        `fast_transform_trajectory` then `simulate_proposals`, as `calibrate_score`
        does, through the vendor's own module-level simulator.
        """
        from navsim.agents.drive_jepa_perception_based.drive_jepa_model import simulator
        from navsim.common.dataclasses import Trajectory as NavsimTrajectory
        from navsim.evaluate.pdm_score import fast_transform_trajectory

        ego = self._ego_state(obs)
        states = np.stack(
            [
                fast_transform_trajectory(
                    NavsimTrajectory(p.astype(np.float32)), simulator.proposal_sampling, ego
                )
                for p in proposals
            ]
        )
        extended = np.zeros((*states.shape[:-1], 11), dtype=states.dtype)
        extended[:, :, :3] = states
        return simulator.simulate_proposals(extended, ego)

    def _comfort(self, past: np.ndarray, simulated: np.ndarray, steps: int) -> np.ndarray:
        """Per candidate, does it pass two-frame Extended Comfort against *past*."""
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
            ego_is_two_frame_extended_comfort,
        )

        previous = past[steps:]
        n = previous.shape[0]
        time_point_s = np.arange(n) * SIM_DT
        return np.array(
            [
                ego_is_two_frame_extended_comfort(previous[None], sim[:-steps][None], time_point_s)[0]
                for sim in simulated
            ],
            dtype=np.float64,
        )

    def _ego_state(self, obs: Observation):
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

        vehicle = get_pacifica_parameters()
        speed, yaw_rate = obs.speed, self._yaw_rate(obs)
        steer = float(np.arctan(vehicle.wheel_base * yaw_rate / speed)) if speed > MIN_STEER_SPEED else 0.0
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
            rear_axle_velocity_2d=StateVector2D(speed, 0.0),
            rear_axle_acceleration_2d=StateVector2D(obs.accel, 0.0),
            tire_steering_angle=steer,
            time_point=TimePoint(int(obs.timestamp_us)),
            vehicle_parameters=vehicle,
            angular_vel=yaw_rate,
        )

    def _yaw_rate(self, obs: Observation) -> float:
        """Heading change over the last recorded step, wrapped to (-pi, pi].

        Not `obs.yaw_rate` where it can be avoided: neuroNCAP's engine differences
        raw headings without wrapping (`emulate_nuscenes_canbus_signals`), so a
        heading crossing +-180 degrees reads as ~63 rad/s, which the bicycle
        relation above turns into full lock. None of the 200 episodes of the
        September sweep crossed it, but a different pick drives a different path.
        """
        if len(self.history) >= 2:
            previous, current = self.history[-2], self.history[-1]
            dt = (current["timestamp"] - previous["timestamp"]) / 1e6
            if 0 < dt <= MAX_INTERVAL_S:
                return float(relative_pose(current["ego2world"], previous["ego2world"])[2] / dt)
        # First frame of an episode: nothing to difference against. A car cannot
        # turn at pi rad/s, so anything faster is the wrap and is dropped.
        return obs.yaw_rate if abs(obs.yaw_rate) < np.pi else 0.0

    # -----------------------------------------------------------------------------

    def infer(self, obs: Observation) -> Trajectory:
        output = self.forward(obs)
        proposals = output["proposals"][0].float().cpu().numpy()
        score = output["pdm_score"][0].float().cpu().numpy()
        simulated = self._simulate(proposals, obs)

        base_pick = int(np.argmax(score))
        pick, passed = base_pick, None
        if self.past is not None:
            steps = overlap_steps((obs.timestamp_us - self.past[1]) / 1e6)
            if steps is not None:
                passed = self._comfort(self.past[0], simulated, steps)
                pick = int(np.argmax(momentum_score(score, passed)))

        self.past = (simulated[pick], obs.timestamp_us)
        self._count(base_pick, pick, passed)
        return Trajectory(proposals[pick], dt=HISTORY_DT)

    def _count(self, base_pick: int, pick: int, passed: np.ndarray | None) -> None:
        if not self.episodes:
            self.episodes.append({"ticks": 0, "with_previous": 0, "pick_changed": 0})
        for record in (self.totals, self.episodes[-1]):
            record["ticks"] += 1
            if passed is not None:
                record["with_previous"] += 1
                record["pick_changed"] += int(pick != base_pick)
        if passed is not None:
            self.totals["argmax_failed"] += int(not passed[base_pick])
            self.totals["passed_sum"] += float(passed.mean())

    def diagnostics(self) -> dict[str, Any]:
        """The base model's history check, plus how often momentum moved the pick.

        `pick_changed` is the whole effect of this model over `drivejepa` on the
        ticks it ran: everywhere else the two drive the same candidate. Counted
        over the node's lifetime, so a resumed run reports only what it re-ran.
        """
        out = super().diagnostics()
        t = self.totals
        n = t["with_previous"]
        out["momentum"] = {
            "ticks": t["ticks"],
            "ticks_with_previous_plan": n,
            "pick_changed": t["pick_changed"],
            "pick_changed_frac": round(t["pick_changed"] / n, 4) if n else None,
            "argmax_failed_comfort_frac": round(t["argmax_failed"] / n, 4) if n else None,
            "mean_candidates_passing_frac": round(t["passed_sum"] / n, 4) if n else None,
            "per_episode": [e for e in self.episodes if e["ticks"]],
        }
        return out


POLICY = DriveJepaMomentumPolicy
