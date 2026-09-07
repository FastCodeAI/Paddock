"""Constant-velocity baseline.

Projects the ego forward in a straight line at its current speed -- the floor a
perception-driven planner must clear. It reads only ego speed (from the CAN
bus), so it needs no images, no weights and no environment of its own; it runs
in paddock's interpreter (`runtime.kind: builtin`).

The same policy serves open-loop scoring (nuScenes) and closed-loop simulation
(Bench2Drive); in closed loop the shared tracker turns the straight plan into
throttle and steering, so the baseline simply keeps going at its entry speed.
"""

from __future__ import annotations

from paddock.policy.protocol import Observation, Policy, Trajectory, constant_velocity


class ConstantVelocityPolicy(Policy):
    cameras = ()
    waypoint_dt = 0.5

    def load(self) -> None:
        """Nothing to load."""

    def reset(self) -> None:
        """No per-episode state."""

    def infer(self, obs: Observation) -> Trajectory:
        return constant_velocity(obs.speed, n=6, dt=self.waypoint_dt)


POLICY = ConstantVelocityPolicy
