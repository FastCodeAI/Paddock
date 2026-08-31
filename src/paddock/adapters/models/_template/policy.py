"""Template model adapter.

Copy this directory, rename it, and implement the three methods below. As
shipped it is a working constant-velocity policy, so you can run the pipeline
end to end before writing any model code:

    paddock run --model <yours> --bench kitscenes --limit 5

Keep that habit: get the plumbing green first, then swap in the model. Most
integration bugs are coordinate-frame bugs, and they are far easier to see when
nothing else is moving.
"""

from __future__ import annotations

from paddock.policy.protocol import Observation, Policy, Trajectory, constant_velocity


class TemplatePolicy(Policy):
    # Declare only the cameras infer() reads -- see model.yaml.
    cameras = ("CAM_FRONT",)
    waypoint_dt = 0.5

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        """Bring weights up. Called once, before any infer().

        Import the model's own package here rather than at module scope: this
        file is imported by `paddock list` and by the test suite, which run in
        paddock's environment where the model's dependencies do not exist.
        """
        # from mymodel import Net
        # self.model = Net.from_pretrained(...).eval().cuda()

    def reset(self) -> None:
        """Clear per-episode state between runs.

        Anything stateful -- frame history, filters, trackers -- must be cleared
        here, or run N+1 starts with run N's context and the second run of a
        scenario quietly differs from the first.
        """

    def infer(self, obs: Observation) -> Trajectory:
        """Plan from one observation.

        The returned trajectory is in the EGO frame: x forward, y **left**,
        metres, evenly spaced by `waypoint_dt`, starting at t = waypoint_dt.
        There is no waypoint for the current position.

        Return at least 6 waypoints. Closed-loop benchmarks use exactly the
        first 6 and index them positionally.

        Never return zeros to mean "no plan" -- an all-zero trajectory is a
        commanded emergency stop and scores as one. Use constant_velocity().
        """
        # image = obs.images["CAM_FRONT"]        # uint8 HWC
        # pinhole = obs.pinhole("CAM_FRONT")     # fx, fy, cx, cy, width, height
        # speed = obs.speed                      # m/s forward
        return constant_velocity(obs.speed, n=6, dt=self.waypoint_dt)


#: Paddock looks for this name when loading an adapter.
POLICY = TemplatePolicy
