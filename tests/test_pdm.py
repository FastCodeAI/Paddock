"""Tests for `paddock.metrics.pdm` and the open-loop benchmarks that fill its terms.

The claims a wrong answer would hide: the formula is NAVSIM v1's with its
weights, a missing term never passes (the composite is NaN), progress is capped
at the human's distance, and every open-loop scorer that declares `pdm_ep` and
`pdm_comfort` actually emits them.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.benchmark import Sample
from paddock.metrics import pdm
from paddock.policy.protocol import Trajectory, constant_velocity


def _straight(speed: float, n: int = 6, dt: float = 0.5) -> np.ndarray:
    return np.stack([speed * dt * np.arange(1, n + 1), np.zeros(n)], axis=1)


def test_the_composite_is_navsim_v1s_formula() -> None:
    assert sum(pdm.WEIGHTS.values()) == 12
    assert pdm.composite(1, 1, 1, 1, 1) == pytest.approx(1.0)
    assert pdm.composite(1, 1, 0, 0.5, 1) == pytest.approx((5 * 0 + 5 * 0.5 + 2 * 1) / 12)
    assert pdm.composite(0, 1, 1, 1, 1) == 0.0, "NC multiplies: an at-fault collision zeroes it"
    assert pdm.composite(1, 0, 1, 1, 1) == 0.0, "so does leaving the drivable area"


@pytest.mark.parametrize("missing", range(5))
def test_a_missing_term_is_never_a_pass(missing: int) -> None:
    terms = [1.0] * 5
    terms[missing] = math.nan
    assert math.isnan(pdm.composite(*terms))


def test_progress_is_capped_at_the_humans_distance() -> None:
    human = _straight(10.0)
    assert pdm.ego_progress(_straight(5.0), human) == pytest.approx(0.5)
    assert pdm.ego_progress(_straight(20.0), human) == 1.0, "overshooting earns nothing"
    assert pdm.ego_progress(_straight(5.0), _straight(0.0)) == 1.0, "a parked human: nothing falls short"


def test_plan_terms_score_progress_on_the_pair_and_comfort_on_the_whole_plan() -> None:
    plan = _straight(10.0, n=8)
    out = pdm.plan_terms(plan[:6], _straight(10.0), plan, 0.5)
    assert out == {"pdm_ep": pytest.approx(1.0), "pdm_comfort": 1.0}
    jerky = plan.copy()
    jerky[7, 1] = 6.0  # past the scored 3 s, still the model's plan
    assert pdm.plan_terms(jerky[:6], _straight(10.0), jerky, 0.5)["pdm_comfort"] == 0.0
    short = pdm.plan_terms(plan[:2], _straight(10.0)[:2], plan[:2], 0.5)
    assert math.isnan(short["pdm_comfort"]), "too short to have a jerk is not comfortable"


def test_kitscenes_scores_its_pdm_terms_over_the_horizon_ade_covers() -> None:
    from paddock.adapters.benchmarks.kitscenes.harness import KitScenesBenchmark

    sample = Sample(sample_id="s", observations=[], ground_truth=_straight(10.0, n=8))
    out = KitScenesBenchmark.score(None, sample, constant_velocity(5.0, n=8))
    assert out["pdm_ep"] == pytest.approx(0.5) and out["pdm_comfort"] == 1.0


def test_nuscenes_scores_its_pdm_terms_on_the_grid_it_scores_l2_on() -> None:
    from paddock.adapters.benchmarks.nuscenes.harness import NuScenesBenchmark

    sample = Sample(sample_id="s", observations=[], ground_truth=_straight(10.0))
    half = NuScenesBenchmark.score(SimpleNamespace(_boxes={}), sample, constant_velocity(5.0, n=6))
    assert half["pdm_ep"] == pytest.approx(0.5)
    assert half["pdm_comfort"] == 1.0
    fine = Trajectory(_straight(10.0, n=30, dt=0.1), dt=0.1)
    assert NuScenesBenchmark.score(SimpleNamespace(_boxes={}), sample, fine)["pdm_ep"] == pytest.approx(1.0)
