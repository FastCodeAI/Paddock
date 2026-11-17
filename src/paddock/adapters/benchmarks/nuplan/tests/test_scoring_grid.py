"""A plan is scored at the times its points claim, not at the index they sit at.

The metrics pair point i of a plan with point i of the recorded future, and the
future is on a 0.5 s grid. Edge plans every 0.1 s, so before this was fixed its
point 4 (0.5 s) was scored against the recorded point 4 (2.5 s).
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuplan.harness import INTERVAL_S, on_scoring_grid
from paddock.policy.protocol import Trajectory

SPEED = 10.0  # m/s, straight ahead


def _straight(dt: float, n: int) -> Trajectory:
    t = dt * np.arange(1, n + 1)
    waypoints = np.column_stack([SPEED * t, np.zeros_like(t), np.zeros_like(t)])
    return Trajectory(waypoints, dt=dt)


def test_a_plan_every_tenth_of_a_second_is_sampled_on_the_half_second_grid():
    grid = on_scoring_grid(_straight(0.1, 64), 8)

    assert grid.shape == (8, 2)
    np.testing.assert_allclose(grid[:, 0], SPEED * INTERVAL_S * np.arange(1, 9))
    np.testing.assert_allclose(grid[:, 1], 0.0)


def test_a_plan_already_on_the_grid_is_returned_unchanged():
    plan = _straight(INTERVAL_S, 8)

    np.testing.assert_allclose(on_scoring_grid(plan, 8), plan.xy)


def test_a_plan_shorter_than_the_horizon_is_trimmed_not_padded():
    grid = on_scoring_grid(_straight(INTERVAL_S, 4), 8)  # ends at 2.0 s

    assert grid.shape == (4, 2)


def test_score_puts_a_fine_plan_on_the_grid_before_comparing():
    from types import SimpleNamespace

    from paddock.adapters.benchmarks.nuplan.harness import NuPlanBenchmark

    ground_truth = _straight(INTERVAL_S, 8).xy
    metrics = NuPlanBenchmark.score(None, SimpleNamespace(ground_truth=ground_truth), _straight(0.1, 64))
    assert metrics["ade"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["fde@1s"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["fde@3s"] == pytest.approx(0.0, abs=1e-9)
