"""Record NAVSIM's own simulator and v1 comfort check on a fixed set of plans.

    <navsim python> tools/benchmarks/navsim/pdm_sim_fixture.py --navsim-v1 <Drive-JEPA>/navsim_v1 \\
        --out tests/fixtures/pdm_sim_navsim.json

`paddock.metrics.pdm_sim` ports NAVSIM's vehicle simulator, its comfort check
and scipy's `savgol_filter` to numpy; `tests/test_pdm_sim.py` holds the port to
this file. It runs in NAVSIM's interpreter, which has navsim, nuplan-devkit and
scipy: `PDMSimulator.simulate_proposals` (the same code in v1 and v2) and v1's
`ego_is_comfortable`, loaded from the v1 checkout by path.

The plans are made here, deterministically, to cover what a driving model
plans: constant speed, speeding up, braking to a stop, standing still with
centimetre jitter, a left and a right turn, a swerve and back, and a plan that
starts far off the car's own state. Each runs at 3 s (neuroNCAP's plans) and 4 s
(NAVSIM's), from three different measured states of the car.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

DT = 0.1


def plans(seconds: float) -> dict[str, np.ndarray]:
    """name -> (K, 3) rear-axle x, y, heading on the 0.1 s grid, starting at the origin."""
    t = np.round(np.arange(0.0, seconds + 1e-9, DT), 6)

    def along(x, y):
        # A step too short to have a direction keeps the last heading, as
        # `paddock.metrics.pdm.derive_headings` does. arctan2 of a zero step
        # can flip the heading by pi at a stop, which leaves the tracker's fit
        # so ill-conditioned that its answer depends on the BLAS and thread count.
        dx, dy = np.gradient(x), np.gradient(y)
        heading = np.zeros(len(x))
        for i in range(1, len(x)):
            heading[i] = math.atan2(dy[i], dx[i]) if math.hypot(dx[i], dy[i]) > 1e-6 else heading[i - 1]
        return np.stack([x, y, heading], axis=1)

    jitter = np.random.default_rng(7).normal(0.0, 0.02, (len(t), 2))
    jitter[0] = 0.0
    out = {
        "constant": along(8.0 * t, 0.0 * t),
        "speeding_up": along(5.0 * t + 0.9 * t**2, 0.0 * t),
        "braking_to_a_stop": along(np.where(t < 2.0, 10.0 * t - 2.5 * t**2, 10.0), 0.0 * t),
        "standing_jitter": np.c_[jitter, np.zeros(len(t))],
        "left_turn": along(6.0 * np.sin(t / 3.0) * 8.0, 8.0 * (1 - np.cos(t / 3.0)) * 6.0 / 3.0),
        "right_turn": along(7.0 * t, -0.35 * t**2),
        "swerve": along(9.0 * t, 1.8 * np.sin(np.pi * t / seconds)),
        "off_the_car": along(4.0 + 3.0 * t, 1.5 + 0.0 * t),
    }
    return out


#: Measured states of the car at the decision: speed, acceleration, steering angle, yaw rate.
STARTS = {"cruising": (8.0, 0.2, 0.01, 0.02), "slow": (1.5, -0.6, -0.05, -0.03), "stopped": (0.0, 0.0, 0.0, 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--navsim-v1", type=Path, required=True, help="Drive-JEPA's navsim_v1 checkout")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
    )
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from scipy.signal import savgol_filter

    path = args.navsim_v1 / "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py"
    spec = importlib.util.spec_from_file_location("navsim_v1_comfort", path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)

    cases = []
    for seconds in (3.0, 4.0):
        named = plans(seconds)
        k = len(next(iter(named.values())))
        sampling = TrajectorySampling(num_poses=k - 1, interval_length=DT)
        times = np.round(np.arange(k) * DT, 6)
        proposals = np.zeros((len(named), k, 11))
        for i, plan in enumerate(named.values()):
            proposals[i, :, :3] = plan
        for start, (speed, accel, steer, yaw_rate) in STARTS.items():
            ego = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
                rear_axle_velocity_2d=StateVector2D(speed, 0.0),
                rear_axle_acceleration_2d=StateVector2D(accel, 0.0),
                tire_steering_angle=steer,
                time_point=TimePoint(0),
                vehicle_parameters=get_pacifica_parameters(),
                angular_vel=yaw_rate,
            )
            initial = ego_state_to_state_array(ego)
            simulated = PDMSimulator(sampling).simulate_proposals(proposals.copy(), ego)
            comfortable = v1.ego_is_comfortable(simulated, times)
            cases.append(
                {
                    "seconds": seconds,
                    "start": start,
                    "plans": list(named),
                    "proposals_se2": proposals[:, :, :3].round(12).tolist(),
                    "initial": initial.round(12).tolist(),
                    "simulated": simulated.round(12).tolist(),
                    "comfortable": comfortable.astype(int).tolist(),
                }
            )

    rng = np.random.default_rng(11)
    signals = []
    for n, window, polyorder, deriv in ((31, 8, 2, 0), (31, 31, 2, 1), (31, 5, 2, 1), (31, 5, 3, 2), (41, 41, 2, 0)):
        x = rng.normal(0.0, 1.0, n).cumsum()
        y = savgol_filter(x, window_length=window, polyorder=polyorder, deriv=deriv, delta=DT)
        signals.append(
            {
                "window": window,
                "polyorder": polyorder,
                "deriv": deriv,
                "delta": DT,
                "x": x.round(12).tolist(),
                "y": y.round(12).tolist(),
            }
        )

    args.out.write_text(
        json.dumps({"source": "NAVSIM PDMSimulator + v1 ego_is_comfortable", "cases": cases, "savgol": signals})
        + "\n"
    )
    print(f"{args.out}: {len(cases)} simulator cases, {len(signals)} savgol signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
