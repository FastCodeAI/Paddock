"""Contract tests for drivejepa_momentum. No navsim, no torch, no GPU, no weights.

The two vendor calls (`_simulate`, `_comfort`) are replaced by fakes; what is
tested is the bookkeeping around them -- which previous plan is compared, when
it is dropped, and how the bonus moves the pick. That the real calls reproduce
the vendor's `calibrate_score` bit for bit is checked against recorded
neuroNCAP ticks, not here (docs/models/drivejepa_momentum.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.drivejepa_momentum.policy import (
    POLICY,
    momentum_score,
    overlap_steps,
)
from paddock.policy.protocol import Policy


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, index):
        return FakeTensor(self.array[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Obs:
    def __init__(self, t_us: int):
        self.timestamp_us = t_us
        self.speed = 5.0
        self.accel = 0.0
        self.yaw_rate = 0.0


def make_policy(scores, passes):
    """A policy whose forward returns *scores* per tick and whose comfort check returns *passes*."""
    policy = POLICY(checkpoint="unused")
    ticks = iter(scores)
    verdicts = iter(passes)
    policy.compared = []

    def forward(obs):
        score = np.asarray(next(ticks), dtype=np.float32)
        proposals = np.zeros((1, len(score), 8, 3), dtype=np.float32)
        proposals[0, :, :, 0] = np.arange(len(score))[:, None]  # candidate i drives x = i
        return {"proposals": FakeTensor(proposals), "pdm_score": FakeTensor(score[None])}

    def simulate(proposals, obs):
        # One "state" row per candidate, tagged with the candidate's index.
        return np.repeat(proposals[:, :1, :1], 41, axis=1)

    def comfort(past, simulated, steps):
        policy.compared.append((int(past[0, 0]), steps))
        return np.asarray(next(verdicts), dtype=np.float64)

    policy.forward, policy._simulate, policy._comfort = forward, simulate, comfort
    policy.reset()
    return policy


def test_is_a_policy() -> None:
    assert issubclass(POLICY, Policy)
    assert POLICY.waypoint_dt == 0.5


def test_blend_is_the_vendors() -> None:
    out = momentum_score(np.array([0.8, 0.7]), np.array([0.0, 1.0]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [(14 * 0.8) / 16, (14 * 0.7 + 2) / 16], rtol=1e-6)


@pytest.mark.parametrize(
    "interval, steps",
    [
        (0.1, 1),
        (0.0999, 1),
        (0.3, 3),
        (0.5, 5),
        (0.54, 5),
        (0.56, None),
        (0.0, None),
        (-0.1, None),
        (0.04, None),
    ],
)
def test_overlap_is_rounded_and_capped(interval, steps) -> None:
    # 0.3 is the case the vendor's int() truncation gets wrong (2, not 3).
    assert overlap_steps(interval) == steps


def test_first_tick_is_the_plain_argmax() -> None:
    policy = make_policy([[0.1, 0.9, 0.5]], [])
    trajectory = policy.infer(Obs(0))
    assert trajectory.waypoints[0, 0] == 1
    assert policy.compared == []


def test_bonus_moves_the_pick_and_the_driven_plan_is_remembered() -> None:
    # Tick 2: candidate 2 (0.85) passes comfort and candidate 1 (0.9) does not:
    # 14*0.85+2 = 13.9 > 14*0.9 = 12.6, so 2 is driven -- and tick 3 compares to 2.
    policy = make_policy([[0.1, 0.9, 0.5], [0.1, 0.9, 0.85], [0.1, 0.9, 0.85]], [[0, 0, 1], [1, 1, 1]])
    picks = [int(policy.infer(Obs(t)).waypoints[0, 0]) for t in (0, 100_000, 200_000)]
    assert picks == [1, 2, 1]
    assert policy.compared == [(1, 1), (2, 1)]
    stats = policy.diagnostics()["momentum"]
    assert stats["ticks"] == 3
    assert stats["ticks_with_previous_plan"] == 2
    assert stats["pick_changed"] == 1


def test_a_stale_previous_plan_is_dropped() -> None:
    policy = make_policy([[0.1, 0.9, 0.5], [0.1, 0.9, 0.85]], [])
    policy.infer(Obs(0))
    policy.infer(Obs(600_000))  # 0.6 s later: past the vendor's 0.55 s cut-off
    assert policy.compared == []


def test_reset_forgets_the_previous_plan_and_opens_an_episode() -> None:
    policy = make_policy([[0.1, 0.9], [0.1, 0.9]], [])
    policy.infer(Obs(0))
    policy.reset()
    policy.infer(Obs(100_000))
    assert policy.compared == []
    assert [e["ticks"] for e in policy.diagnostics()["momentum"]["per_episode"]] == [1, 1]


def _pose(yaw: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    return pose


def test_yaw_rate_is_wrapped_across_180_degrees() -> None:
    # neuroNCAP's own canbus would report (-179 - 179) deg / 0.1 s = -62.5 rad/s here.
    policy = POLICY(checkpoint="unused")
    policy.history = [
        {"timestamp": 0, "ego2world": _pose(np.radians(179.0))},
        {"timestamp": 100_000, "ego2world": _pose(np.radians(-179.0))},
    ]
    obs = Obs(100_000)
    obs.yaw_rate = -62.5
    assert policy._yaw_rate(obs) == pytest.approx(np.radians(2.0) / 0.1)


def test_first_frame_drops_an_impossible_yaw_rate() -> None:
    policy = POLICY(checkpoint="unused")
    obs = Obs(0)
    obs.yaw_rate = 0.2
    assert policy._yaw_rate(obs) == 0.2
    obs.yaw_rate = -62.5
    assert policy._yaw_rate(obs) == 0.0
