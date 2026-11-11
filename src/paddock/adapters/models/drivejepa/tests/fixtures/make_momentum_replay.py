"""Write momentum_replay_made_up.npz: a made-up drive, and the picks upstream's own chain makes on it.

Nothing here comes from paddock's momentum.py, which is the code the replay
tests. The loop follows upstream's evaluation loop
(run_pdm_score_sequential.py:121-168) and the chosen plan is simulated the way
pdm_score.py:189-214 does it, calling upstream's functions directly. Simulator
slot 0 holds a different plan from slot 1 (upstream's would be the PDM-closed
reference), so a simulator that coupled its batch entries would make paddock's
replay disagree with this one.

Every number is invented: a ~8 m/s drive with a slow left bend, and 32 arcs per
replan, three quarters of them close to the current motion, as a trained
planner's are, and a quarter outliers. The learned score sits in a band narrow
enough for the stage's 2/16 comfort term to matter; with this seed the stage
changes the pick on 13 of the 26 replans it is fed.

Run it in Drive-JEPA's environment, from this directory:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<Drive-JEPA>/navsim_v2 \\
        <drive-jepa env>/bin/python make_momentum_replay.py
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel
from navsim.common.dataclasses import EgoStatus, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

OUT = Path(__file__).with_name("momentum_replay_made_up.npz")
N_REPLANS, N_PROPOSALS, N_POSES, DT_US = 27, 32, 8, 500_000
#: Spread of the learned score's logit, and how far a clustered proposal strays.
SPREAD, VARIETY = 0.6, 0.1

rng = np.random.default_rng(20261110)

# --- the drive -----------------------------------------------------------------
time_us = 1_000_000_000 + DT_US * np.arange(N_REPLANS, dtype=np.int64)
speed = 8.0 + 0.4 * np.sin(np.arange(N_REPLANS) / 4.0)
yaw_rate = np.where(np.arange(N_REPLANS) < 14, 0.06, 0.0)
pose = np.zeros((N_REPLANS, 3), dtype=np.float64)
for i in range(1, N_REPLANS):
    x, y, h = pose[i - 1]
    d = speed[i - 1] * 0.5
    pose[i] = (x + d * np.cos(h), y + d * np.sin(h), h + yaw_rate[i - 1] * 0.5)
velocity = np.stack([speed, np.zeros(N_REPLANS)], axis=1).astype(np.float32)
accel = np.stack([np.gradient(speed, 0.5), np.zeros(N_REPLANS)], axis=1).astype(np.float32)

# --- the menus: 32 arcs per replan, ego frame, poses at 0.5 s .. 4.0 s ----------
t = 0.5 * np.arange(1, N_POSES + 1)


def arc(v: float, curvature: float, a: float) -> np.ndarray:
    dist = v * t + 0.5 * a * t**2
    if abs(curvature) < 1e-9:
        xy = np.stack([dist, np.zeros_like(dist)], axis=1)
    else:
        xy = np.stack(
            [np.sin(curvature * dist) / curvature, (1 - np.cos(curvature * dist)) / curvature], axis=1
        )
    return np.concatenate([xy, (curvature * dist)[:, None]], axis=1)


proposals = np.zeros((N_REPLANS, N_PROPOSALS, N_POSES, 3), dtype=np.float32)
pred_logit = rng.normal(0.0, 1.0, size=(N_REPLANS, N_PROPOSALS, 6)).astype(np.float32)
for i in range(N_REPLANS):
    for k in range(N_PROPOSALS):
        wide = k % 4 == 3
        v = speed[i] * (rng.uniform(0.7, 1.2) if wide else rng.uniform(1 - VARIETY, 1 + VARIETY))
        bend = rng.uniform(-0.03, 0.03) if wide else rng.uniform(-VARIETY / 10, VARIETY / 10)
        curvature = bend + yaw_rate[i] / max(speed[i], 1.0)
        a = rng.uniform(-3.0, 2.0) if wide and rng.random() < 0.5 else 0.0
        proposals[i, k] = arc(v, curvature, a)
    pred_logit[i, :, -1] = (1.0 + SPREAD * rng.normal(size=N_PROPOSALS)).astype(np.float32)

# --- upstream's chain ------------------------------------------------------------
vehicle = get_pacifica_parameters()
agent_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
simulator = PDMSimulator(TrajectorySampling(num_poses=40, interval_length=0.1))


def ego_state_at(i: int):
    status = EgoStatus(
        ego_pose=pose[i].astype(np.float64),
        ego_velocity=velocity[i].astype(np.float32),
        ego_acceleration=accel[i].astype(np.float32),
        driving_command=np.zeros(4, dtype=int),
        in_global_frame=True,
    )
    return ego_status_to_ego_state(
        ego_status=status, vehicle_parameters=vehicle, time_point=TimePoint(int(time_us[i]))
    )


def as_states(poses: np.ndarray, state) -> np.ndarray:
    trajectory = transform_trajectory(Trajectory(poses, agent_sampling), state)
    return get_trajectory_as_array(trajectory, simulator.proposal_sampling, state.time_point)


def simulated(poses: np.ndarray, state) -> np.ndarray:
    """pdm_score.py:189-214 for the chosen plan, with another plan in slot 0."""
    other = as_states(arc(5.0, 0.0, 0.0).astype(np.float32), state)
    both = np.stack([other, as_states(poses, state)])
    return np.asarray(simulator.simulate_proposals(both, state)[1], dtype=np.float64)


pick = np.zeros(N_REPLANS, dtype=np.int64)
calibrated = np.zeros((N_REPLANS, N_PROPOSALS), dtype=np.float32)
past, past_time, changed = None, None, 0
for i in range(N_REPLANS):
    state = ego_state_at(i)
    interval = None if past_time is None else (int(time_us[i]) - past_time) / 1e6
    if interval is not None and interval > 0.55:  # run_pdm_score_sequential.py:131-132
        past = None
    score = torch.sigmoid(torch.from_numpy(pred_logit[i][None]))[:, :, -1]
    plain = int(torch.argmax(score, dim=1)[0])
    if past is not None:
        features = {
            "past_ego_simulated_states": past,
            "observation_interval": interval,
            "metric_cache": SimpleNamespace(ego_state=state),
        }
        score = DriveJEPAModel.calibrate_score(None, features, torch.from_numpy(proposals[i][None]), score)
    pick[i] = int(torch.argmax(score, dim=1)[0])
    calibrated[i] = score.numpy()[0]
    changed += int(pick[i] != plain)
    past, past_time = simulated(proposals[i, pick[i]], state), int(time_us[i])

print(f"changed the pick on {changed} of {N_REPLANS - 1} fed replans")
np.savez_compressed(
    OUT,
    time_us=time_us,
    proposals=proposals,
    pred_logit=pred_logit,
    ego_pose=pose,
    ego_velocity=velocity,
    ego_acceleration=accel,
    pick=pick,
    calibrated=calibrated,
)
