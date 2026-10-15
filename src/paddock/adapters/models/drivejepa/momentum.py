"""Momentum-aware trajectory selection: hand the model its own previous plan.

Drive-JEPA's paper (section 3.5) recalibrates the learned score of its 32
proposals against the trajectory it selected at the previous frame, so the pick
does not flip between two near-equal proposals from one frame to the next.
Upstream implements it in ``DriveJEPAModel.calibrate_score``
(``drive_jepa_model.py:98-151`` at the sha ``repos.yaml`` pins), reached only
when the feature dict carries ``past_ego_simulated_states`` (the gate, ``:87``).
Upstream's evaluation loop, ``run_pdm_score_sequential.py``, is what fills that
key. A node that calls ``agent.forward`` with the feature builder's output
alone never reaches the stage.

What the stage reads is narrower than the argument names suggest:

    features["past_ego_simulated_states"]   the previous *chosen* plan, run
                                            through the PDM simulator
    features["observation_interval"]        seconds since that plan
    features["metric_cache"].ego_state      the current ego state -- the only
                                            attribute of the cache it touches
                                            (``:101-102``)

None of it is ground truth and none of it is the scorer's state: a previous
plan, an ego state the node is sent with every observation, and a clock. So the
node can keep the book itself. This module repeats the loop's bookkeeping, one
step per line of upstream:

    upstream (run_pdm_score_sequential.py)        here
    ------------------------------------------    -------------------------
    :129  the frame's ego state                   Upstream.ego_state
          (navsim_scenario_utils.py:42-52)
    :130  observation_interval = now - past       Momentum.features
    :131-132  older than 0.55 s -> no past plan   Momentum.features
    :138  compute_trajectory(..., past, cache,    Momentum.features (the three
          interval)                               keys, abstract_agent.py:85-87)
    :140  pdm_score(...) -> ego_simulated_states  Upstream.simulate_chosen
          (pdm_score.py:177, 200-214, 287)
    :167-168  remember time and states            Momentum.remember

Every function on the right calls upstream's own code; ``calibrate_score`` --
overlap arithmetic included -- is reached through the model's own ``forward``
and nothing in the vendored tree is edited.

One thing has no upstream counterpart. Upstream's loop steps at 0.5 s, so "the
previous frame" and "the plan 0.5 s ago" are the same plan. A closed-loop
simulator may call the node at 10 or 20 Hz, where the previous call is 0.05 s
old: the stage's overlap offset is ``int(interval / 0.1)``, which is 0 there and
leaves it an empty window. So the previous plan is chosen the way
`DriveJepaPolicy._select_history` chooses history frames -- by timestamp, the
one nearest 0.5 s ago -- and at upstream's own cadence that is upstream's rule.

Only a plan 0.45 to 0.55 s old is fed, and the stage always works at offset 5.
The two trajectories are compared sample against sample on a 0.1 s grid, so an
offset that does not match the real interval compares states from different
moments: ``int(0.3 / 0.1)`` is 2, a 0.3 s-old plan read 0.1 s out of step. The
first replans of an episode at 10 Hz, which have nothing older than 0.4 s to
compare with, therefore select on the plain score rather than on a comparison
the stage was never evaluated at.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

SWITCH_ENV = "DRIVEJEPA_MOMENTUM"
SWITCHES = ("off", "on")

#: run_pdm_score_sequential.py:131 -- a previous plan older than this is dropped.
MAX_AGE_S = 0.55
#: The spacing upstream's loop steps at, and the age a previous plan is sought at.
TARGET_AGE_S = 0.5
#: The youngest plan fed: as far below the target as upstream's cutoff is above
#: it. Anything younger belongs to another overlap offset (module docstring).
MIN_AGE_S = 0.45
#: Ages are compared in whole microseconds, the unit timestamps arrive in, so
#: the window's edges do not depend on how a difference of two floats rounds.
MIN_AGE_US, MAX_AGE_US = round(MIN_AGE_S * 1e6), round(MAX_AGE_S * 1e6)


class Upstream:
    """The upstream functions the bookkeeping calls, imported once.

    nuPlan and the vendored NAVSIM exist only inside Drive-JEPA's own
    environment, so this is built on the first replan rather than at import:
    `paddock list` and the test suite import this module in paddock's.
    """

    def __init__(self, trajectory_sampling: Any) -> None:
        from navsim.common.dataclasses import EgoStatus, Trajectory
        from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
        from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        self._EgoStatus = EgoStatus
        self._Trajectory = Trajectory
        self._TimePoint = TimePoint
        self._get_trajectory_as_array = get_trajectory_as_array
        self._transform_trajectory = transform_trajectory
        self._ego_status_to_ego_state = ego_status_to_ego_state
        # The agent's own sampling (abstract_agent.py:95): eight poses at 0.5 s.
        self._trajectory_sampling = trajectory_sampling
        # navsim_scenario.py:53 -- the scenario's default vehicle.
        self._vehicle_parameters = get_pacifica_parameters()
        # cfg.simulator of upstream's evaluation: default_run_pdm_score.yaml
        # with proposal_sampling from default_common.yaml -- 4 s at 0.1 s.
        self._simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))

    def ego_state(
        self, pose: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, time_us: int
    ) -> Any:
        """The frame's ego state, built by upstream's own converter.

        `ego_status_to_ego_state` fixes the steering angle at 0.0 and takes
        pose, velocity and acceleration from the frame's ego status. *pose* is
        (x, y, heading) in any frame that stays fixed for the episode -- it
        plays the part of upstream's global frame, and the stage only ever
        compares two plans expressed in it.
        """
        status = self._EgoStatus(
            ego_pose=np.asarray(pose, dtype=np.float64),
            ego_velocity=np.asarray(velocity, dtype=np.float32),
            ego_acceleration=np.asarray(acceleration, dtype=np.float32),
            driving_command=np.zeros(4, dtype=int),  # not read by the converter
            in_global_frame=True,
        )
        return self._ego_status_to_ego_state(
            ego_status=status,
            vehicle_parameters=self._vehicle_parameters,
            time_point=self._TimePoint(int(time_us)),
        )

    def simulate_chosen(self, poses: np.ndarray, ego_state: Any) -> np.ndarray:
        """`pdm_score()`'s second return value for the chosen trajectory.

        Upstream simulates the pair [PDM-closed reference, prediction] and
        keeps the prediction's states. There is no reference trajectory in a
        node, so the prediction fills both slots and slot 1 is returned; the
        simulator treats batch entries independently.
        """
        trajectory = self._Trajectory(poses, self._trajectory_sampling)
        pred_trajectory = self._transform_trajectory(trajectory, ego_state)
        pred_states = self._get_trajectory_as_array(
            pred_trajectory, self._simulator.proposal_sampling, ego_state.time_point
        )
        trajectory_states = np.concatenate([pred_states[None, ...], pred_states[None, ...]], axis=0)
        simulated_states = self._simulator.simulate_proposals(trajectory_states, ego_state)
        return np.asarray(simulated_states[1], dtype=np.float64)


class Momentum:
    """The previous plan, kept per episode, and the three keys built from it."""

    def __init__(self, upstream: Any = None) -> None:
        # Injected by the tests, which have no nuPlan; built from the agent in
        # `DriveJepaPolicy.load` otherwise, so an environment that cannot
        # import upstream's functions fails when the node starts.
        self.upstream = upstream
        self.reset()
        self.counts = {"replans": 0, "fed": 0, "no_previous": 0, "out_of_window": 0, "picks_changed": 0}

    def reset(self) -> None:
        """Forget the previous plan. Counts survive: they describe the run."""
        self._plans: list[tuple[int, np.ndarray]] = []

    def _previous(self, time_us: int) -> tuple[np.ndarray | None, float | None]:
        """The plan nearest 0.5 s ago, or why there is none to feed.

        Age is also the guard against another episode's plan: a simulator that
        starts a new rollout without `/reset` restarts or jumps its clock, and
        a plan from before the jump is either in the future or too old.
        """
        aged = [(time_us - at_us, states) for at_us, states in self._plans]
        usable = [(age_us, states) for age_us, states in aged if MIN_AGE_US <= age_us <= MAX_AGE_US]
        if not usable:
            # `out_of_window`: plans are kept, but none is the right age -- too
            # old after a gap, or too young in an episode's first half second.
            self.counts["out_of_window" if aged else "no_previous"] += 1
            return None, None
        age_us, states = min(usable, key=lambda entry: abs(entry[0] - TARGET_AGE_S * 1e6))
        # The stage reads the interval for one thing, its overlap offset
        # `int(interval / 0.1)` (drive_jepa_model.py:120-121). A plan 0.48 s old
        # would floor to offset 4 and be read 0.08 s out of step; held at the
        # target it is read at offset 5, 0.02 s out. From 0.5 s up upstream's
        # own arithmetic already gives 5 and the interval is passed as it is.
        return states, max(age_us / 1e6, TARGET_AGE_S)

    def features(
        self, pose: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, time_us: int
    ) -> tuple[dict[str, Any], Any]:
        """The three keys `abstract_agent.py:85-87` adds, and the ego state used.

        Added to the feature dict after the tensors are built and left as they
        are -- a numpy array, an object, a float -- which is how upstream hands
        them over and what `calibrate_score` expects.
        """
        ego_state = self.upstream.ego_state(pose, velocity, acceleration, time_us)
        self.counts["replans"] += 1
        states, age = self._previous(time_us)
        if states is not None:
            self.counts["fed"] += 1
        extra = {
            "past_ego_simulated_states": states,
            # Only `.ego_state` is read; this is not a MetricCache and carries
            # nothing else one holds.
            "metric_cache": SimpleNamespace(ego_state=ego_state),
            "observation_interval": age,
        }
        return extra, ego_state

    def remember(self, poses: np.ndarray, ego_state: Any, time_us: int, changed: bool = False) -> None:
        """Simulate the chosen plan and keep it for the replans that follow."""
        self.counts["picks_changed"] += int(changed)
        self._plans.append((int(time_us), self.upstream.simulate_chosen(poses, ego_state)))
        # Nothing older than the cutoff can be fed again. A clock that went
        # backwards (a new episode without a reset) makes every kept plan
        # "newer" than now; those go too.
        self._plans = [(at_us, s) for at_us, s in self._plans if 0 <= time_us - at_us <= MAX_AGE_US]
