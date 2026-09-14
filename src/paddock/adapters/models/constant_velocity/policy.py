"""The reference floor: hold the current speed, straight ahead.

Not a model, and not pretending to be. It exists because a closed-loop score is
uninterpretable on its own -- AlpaSim's `scene_score` zeroes a scene outright on
an at-fault collision or an offroad event, and a policy that never steers
commits neither, so it outscores three of the four real models on the pilot
set. Reading that number without the floor beside it would suggest those models
are worse than doing nothing, when what it actually measures is that the metric
punishes at-fault failure and constant velocity has no way to be at fault.

Deterministic by construction, which is what makes it useful as a harness test:
the same observation always produces the same plan, so any difference between
two runs of it is the harness, never the policy.
"""

from __future__ import annotations

import numpy as np

# N_WAYPOINTS and CANBUS_VELOCITY come from the protocol, not from here. This
# file already imports from it two names up; redefining them makes a second
# definition that drifts silently, and the canbus offset is exactly the constant
# whose drift nothing would raise on.
from paddock.policy.protocol import (
    CANBUS_VELOCITY,
    N_WAYPOINTS,
    Observation,
    Policy,
    Trajectory,
)


class ConstantVelocityPolicy(Policy):
    cameras: tuple[str, ...] = ("CAM_FRONT",)
    waypoint_dt: float = 0.5

    #: Nothing to prime: the plan is a function of one scalar, so a manufactured
    #: history would change nothing and only obscure that.
    self_prime: bool = False

    def infer(self, obs: Observation) -> Trajectory:
        speed = self.speed_of(obs)
        # x forward, y left; evenly spaced from t = dt, no waypoint for "now".
        distances = speed * self.waypoint_dt * np.arange(1, N_WAYPOINTS + 1, dtype=np.float64)
        waypoints = np.zeros((N_WAYPOINTS, 2), dtype=np.float64)
        waypoints[:, 0] = distances
        return Trajectory(waypoints=waypoints, dt=self.waypoint_dt)

    @staticmethod
    def speed_of(obs: Observation) -> float:
        """Ego speed in m/s, or zero when the source did not supply one.

        Zero rather than a guess: a floor that invented a speed would be driving
        on its own invention, and a stationary plan is at least honest about
        having been told nothing. Negative speeds are clamped -- the floor does
        not reverse.
        """
        canbus = np.asarray(obs.canbus, dtype=np.float64).ravel()
        if canbus.size <= CANBUS_VELOCITY:
            return 0.0
        speed = float(canbus[CANBUS_VELOCITY])
        if not np.isfinite(speed):
            return 0.0
        return max(speed, 0.0)


#: Paddock looks for this name when loading an adapter.
POLICY = ConstantVelocityPolicy
