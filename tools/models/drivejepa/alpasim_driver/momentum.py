# SPDX-License-Identifier: Apache-2.0
"""Worker half of 0.1.4b: hand the model its previous plan, upstream's way.

Paper §3.5 ("Momentum-aware Trajectory Selection") recalibrates the learned
score against the trajectory selected at the previous frame. Upstream
implements it in ``DriveJEPAModel.calibrate_score``
(drive_jepa_model.py:98-151), reached only when the feature dict carries
``past_ego_simulated_states`` (gate, :87-88), and feeds that key from its own
evaluation loop, ``run_pdm_score_sequential.py``. 0.1.4 called
``agent.forward`` with image and ego features only, so the branch never ran.

This module repeats that loop's bookkeeping, one step per line of upstream:

    upstream (run_pdm_score_sequential.py)           here
    ---------------------------------------------    ------------------------
    :129  metric_cache.ego_state, the frame's ego    ego_state()
          state (navsim_scenario.py:206-210 ->
          navsim_scenario_utils.py:42-52)
    :130  observation_interval = time_s - past       previous()
    :131-132  > 0.55 s -> no previous plan           previous()
    :138  compute_trajectory(agent_input, past,      forward()  (the three keys
          metric_cache, interval)                    of abstract_agent.py:85-87)
    :140  pdm_score(...) -> ego_simulated_states     simulate_chosen()
          (pdm_score.py:177, 200-214, 287)
    :167-168  remember time and states               the planner, per session

Every function named on the right calls upstream's own code. Nothing in the
vendored tree is edited, and ``calibrate_score`` -- overlap arithmetic
included -- is reached through the model's own ``forward``.

The worker stays stateless: the planner keeps the previous plan's simulated
states per rollout and sends them back with the next replan, so two rollouts
sharing this worker cannot see each other's history.

**Runs only inside the Drive-JEPA conda env** (nuPlan lives there), like
real_model.py. With the switch off the planner sends no ``momentum`` key and
``forward`` is 0.1.4's call, untouched.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import constants as K
from .real_model import REQUIRED_FEATURES, RETURNED, RealDriveJEPA, RealDriveJEPAError


class _Upstream:
    """The upstream functions this module calls, imported once.

    Imported lazily because they pull in nuPlan and the vendored NAVSIM, which
    are only importable after ``RealDriveJEPA`` has put ``navsim_v2`` on the
    path.
    """

    def __init__(self) -> None:
        import torch
        from navsim.common.dataclasses import EgoStatus, Trajectory
        from navsim.evaluate.pdm_score import (
            get_trajectory_as_array,
            transform_trajectory,
        )
        from navsim.planning.scenario_builder.navsim_scenario_utils import (
            ego_status_to_ego_state,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
            PDMSimulator,
        )
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.common.actor_state.vehicle_parameters import (
            get_pacifica_parameters,
        )
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        self.torch = torch
        self.EgoStatus = EgoStatus
        self.Trajectory = Trajectory
        self.TimePoint = TimePoint
        self.get_trajectory_as_array = get_trajectory_as_array
        self.transform_trajectory = transform_trajectory
        self.ego_status_to_ego_state = ego_status_to_ego_state
        # navsim_scenario.py:53 -- the scenario's default vehicle.
        self.vehicle_parameters = get_pacifica_parameters()
        # cfg.simulator of the evaluation: default_run_pdm_score.yaml:18-21
        # with proposal_sampling from default_common.yaml:25-29.
        self.simulator = PDMSimulator(
            TrajectorySampling(num_poses=40, interval_length=0.1)
        )

    # -- :129 ---------------------------------------------------------------
    def ego_state(self, request: dict):
        """The frame's ego state, built by upstream's own converter.

        ``ego_status_to_ego_state`` fixes the steering angle at 0.0 and takes
        pose, velocity and acceleration from the frame's ego status
        (navsim_scenario_utils.py:42-52). Velocity and acceleration here are
        the newest ``ego_status`` row -- the numbers the model itself is given
        -- which is the same pairing upstream has: one frame record feeds both.
        The pose is the plan's anchor in AlpaSim's local frame, which plays
        the part of upstream's global frame: fixed for the whole rollout.
        """
        status = self.EgoStatus(
            ego_pose=np.asarray(request["ego_pose"], dtype=np.float64),
            ego_velocity=np.asarray(request["ego_velocity"], dtype=np.float32),
            ego_acceleration=np.asarray(request["ego_acceleration"], dtype=np.float32),
            driving_command=np.zeros(4, dtype=int),  # not read by the converter
            in_global_frame=True,
        )
        return self.ego_status_to_ego_state(
            ego_status=status,
            vehicle_parameters=self.vehicle_parameters,
            time_point=self.TimePoint(int(request["time_us"])),
        )

    # -- :130-132 -----------------------------------------------------------
    def previous(self, request: dict, ego_state):
        """``(past_states, observation_interval, code)`` as the loop decides them."""
        past = request.get("past_states")
        if past is None:
            return None, None, K.MOMENTUM_NO_PREV
        past_start_time = self.TimePoint(int(request["past_time_us"])).time_s  # :167
        observation_interval = ego_state.time_point.time_s - past_start_time  # :130
        if observation_interval > K.MOMENTUM_MAX_AGE_S:  # :131
            return None, observation_interval, K.MOMENTUM_DROPPED_AGE  # :132
        return np.asarray(past, dtype=np.float64), observation_interval, K.MOMENTUM_OK

    # -- :140 ---------------------------------------------------------------
    def simulate_chosen(self, poses: np.ndarray, trajectory_sampling, ego_state) -> np.ndarray:
        """``pdm_score()``'s second return value for the chosen trajectory.

        pdm_score.py:177 (``transform_trajectory``), :200-207
        (``get_trajectory_as_array``), :212-214 (``simulate_proposals``) and
        :287 (``simulated_states[pred_idx]``). Upstream simulates the pair
        [PDM-closed reference, prediction]; there is no reference trajectory
        here, so the prediction fills both slots and slot 1 is returned. The
        simulator treats batch entries independently.
        """
        trajectory = self.Trajectory(poses, trajectory_sampling)  # abstract_agent.py:95
        pred_trajectory = self.transform_trajectory(trajectory, ego_state)
        pred_states = self.get_trajectory_as_array(
            pred_trajectory, self.simulator.proposal_sampling, ego_state.time_point
        )
        trajectory_states = np.concatenate(
            [pred_states[None, ...], pred_states[None, ...]], axis=0
        )
        simulated_states = self.simulator.simulate_proposals(trajectory_states, ego_state)
        return np.asarray(simulated_states[1], dtype=np.float64)


class MomentumDriveJEPA(RealDriveJEPA):
    """``RealDriveJEPA`` that also feeds the three keys §3.5 needs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._up: _Upstream | None = None
        self._calibrate_s = 0.0
        self._time_calibrate_score()

    # ------------------------------------------------------------------
    def _upstream(self) -> _Upstream:
        if self._up is None:
            self._up = _Upstream()
        return self._up

    def _time_calibrate_score(self) -> None:
        """Time the vendored stage without touching it.

        ``forward`` looks ``calibrate_score`` up on the model instance, so an
        instance attribute that calls the class's own bound method and reads a
        clock around it is all this is. Needed for the latency gate: the stage
        runs inside ``forward`` and cannot be timed from outside it.
        """
        model = self._agent._pad_model
        vendored = model.calibrate_score

        def timed(features, proposals, pdm_score):
            start = time.perf_counter()
            result = vendored(features, proposals, pdm_score)
            self._calibrate_s = time.perf_counter() - start
            return result

        model.calibrate_score = timed

    def _forward_with(self, features: dict, extra: dict) -> dict:
        """real_model.py:140-169, with ``extra`` added after the conversion.

        Upstream adds the three keys after the feature tensors are built and
        leaves them as they are -- a numpy array, a cache object, a float
        (abstract_agent.py:83-87). ``RealDriveJEPA.forward`` would turn the
        array into a tensor, which ``calibrate_score`` does not expect, so the
        body is repeated here with the keys added in upstream's order.
        """
        torch = self._torch
        missing = [k for k in REQUIRED_FEATURES if k not in features]
        if missing:
            raise RealDriveJEPAError(f"features missing keys: {missing}")

        tensors: dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.from_numpy(np.ascontiguousarray(value)).to(
                    self._device
                )
            else:
                tensors[key] = value
        tensors.update(extra)

        with torch.no_grad():
            out = self._agent.forward(tensors)
            # The score before recalibration: the same expression, on the same
            # tensor, as drive_jepa_model.py:85.
            learned = torch.sigmoid(out["pred_logit"])[:, :, -1]
            out["momentum_S"] = learned
            out["momentum_pick_without"] = torch.argmax(learned, dim=1)
            out["momentum_pick_with"] = torch.argmax(out["pdm_score"], dim=1)  # :90

        result = {}
        for key, value in out.items():
            result[key] = (
                value.detach().cpu().numpy() if hasattr(value, "detach") else value
            )
        absent = [k for k in RETURNED if k not in result]
        if absent:
            raise RealDriveJEPAError(f"model returned no {absent}; got {list(result)}")
        return result

    # ------------------------------------------------------------------
    def forward(self, features: dict) -> dict:
        request = features.get(K.MOMENTUM_FEATURE_KEY)
        if request is None:
            # Switch off: exactly the call 0.1.4 makes.
            return super().forward(features)

        up = self._upstream()
        started = time.perf_counter()
        plain = {k: v for k, v in features.items() if k != K.MOMENTUM_FEATURE_KEY}
        ego_state = up.ego_state(request)
        past, interval, code = up.previous(request, ego_state)
        prepared = time.perf_counter()

        self._calibrate_s = 0.0
        result = self._forward_with(
            plain,
            {
                # abstract_agent.py:85-87, same keys, same order.
                "past_ego_simulated_states": past,
                "metric_cache": SimpleNamespace(ego_state=ego_state),
                "observation_interval": interval,
            },
        )
        calibrate_s = self._calibrate_s

        learned = np.asarray(result["momentum_S"], dtype=np.float32)[0]
        calibrated = np.asarray(result["pdm_score"], dtype=np.float32)[0]
        if past is None:
            comfort = None
            if not np.array_equal(calibrated, learned):
                raise RealDriveJEPAError(
                    "no previous plan was fed, yet pdm_score differs from the "
                    "learned score: the model is not selecting on plain S"
                )
        else:
            # S <- (14 S + 2 S_c) / 16 (drive_jepa_model.py:149), solved for S_c.
            comfort = np.rint((16.0 * calibrated.astype(np.float64)
                               - 14.0 * learned.astype(np.float64)) / 2.0)
            rebuilt = (14.0 * learned.astype(np.float64) + 2.0 * comfort) / 16.0
            if not (np.isin(comfort, (0.0, 1.0)).all()
                    and np.abs(rebuilt - calibrated).max() < 1e-5):
                raise RealDriveJEPAError(
                    "calibrated score is not (14 S + 2 S_c) / 16 with S_c in "
                    "{0, 1}; the vendored stage did not run as expected"
                )
            comfort = comfort.astype(np.float32)

        post_started = time.perf_counter()
        chosen = np.asarray(result["trajectory"])[0]
        states = up.simulate_chosen(chosen, self._agent._trajectory_sampling, ego_state)
        finished = time.perf_counter()

        result["momentum_Sc"] = comfort
        result["momentum_states"] = states
        result["momentum_meta"] = {
            "code": code,
            "interval_s": None if interval is None else float(interval),
            # What drive_jepa_model.py:121 computes from that interval. For the
            # log only; the stage does its own arithmetic.
            "overlap_start": (
                int(interval / 0.1) if code == K.MOMENTUM_OK else None
            ),
            "pick_with": int(np.asarray(result.pop("momentum_pick_with"))[0]),
            "pick_without": int(np.asarray(result.pop("momentum_pick_without"))[0]),
            "ms_prepare": (prepared - started) * 1e3,
            "ms_calibrate": calibrate_s * 1e3,
            "ms_simulate": (finished - post_started) * 1e3,
        }
        result["momentum_S"] = learned
        return result

    __call__ = forward


# ---------------------------------------------------------------------------
# Test seam: the momentum path with no weights
# ---------------------------------------------------------------------------


class CannedMomentumDriveJEPA(MomentumDriveJEPA):
    """``--model canned``: everything above, on outputs the caller supplies.

    Loading the checkpoint costs ~90 s and 5 GiB; the tests of the momentum
    bookkeeping need neither. The stand-in agent's ``forward`` repeats
    drive_jepa_model.py:85-94 on ``canned_proposals`` / ``canned_pred_logit``
    from the request, and calls the vendored ``calibrate_score`` at the same
    gate. Never constructed on a production path.
    """

    def __init__(self, device: str = "cpu") -> None:  # noqa: D401 - no super().__init__
        import sys

        import torch

        from .real_model import NAVSIM_V2

        if str(NAVSIM_V2) not in sys.path:
            sys.path.insert(0, str(NAVSIM_V2))
        from navsim.agents.drive_jepa_perception_based.drive_jepa_model import (
            DriveJEPAModel,
        )
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        self._torch = torch
        self._device = torch.device(device)

        class _Model:
            def calibrate_score(self, features, proposals, pdm_score):
                return DriveJEPAModel.calibrate_score(self, features, proposals, pdm_score)

        class _Agent:
            _pad_model = _Model()
            # drive_jepa_perception_based_agent.yaml:8-12
            _trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

            def forward(self, features):
                proposals = features["canned_proposals"]
                pred_logit = features["canned_pred_logit"]
                batch_size = proposals.shape[0]
                pdm_score = torch.sigmoid(pred_logit)[:, :, -1]  # :85
                if (
                    "past_ego_simulated_states" in features
                    and features["past_ego_simulated_states"] is not None
                ):  # :87
                    pdm_score = self._pad_model.calibrate_score(
                        features, proposals, pdm_score
                    )  # :88
                token = torch.argmax(pdm_score, dim=1)  # :90
                trajectory = proposals[torch.arange(batch_size), token]  # :91
                return {
                    "proposals": proposals,
                    "pred_logit": pred_logit,
                    "trajectory": trajectory,
                    "pdm_score": pdm_score,
                }

        self._agent = _Agent()
        self._up = None
        self._calibrate_s = 0.0
        self._time_calibrate_score()

    def _forward_with(self, features: dict, extra: dict) -> dict:
        # The canned request carries no camera tensors; satisfy the key check
        # the real path makes, and nothing else.
        padded = {**{k: None for k in REQUIRED_FEATURES}, **features}
        return super()._forward_with(padded, extra)
