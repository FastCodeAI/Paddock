"""Tests for `comma2k19.score` -- the miss rate and the PDM-style composite.

The property that matters most is **NaN discipline**, because every failure
mode of a borrowed scoring formula looks the same from outside: a term that
could not be measured quietly becoming a pass, and the composite reading high
for a model that simply did not answer. So the tests here are mostly about
what *does not* get credit -- a plan too short to have a jerk, a plan that
leaves the corridor, a plan that stops short of where the human got to.

The second property is that the composite is a pure function of columns that
are published separately. Every input below is set by hand in the `scored` dict
rather than being computed, which is what a consumer reweighting these from
`samples.jsonl` would be doing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from paddock.adapters.benchmarks.comma2k19 import report as R
from paddock.adapters.benchmarks.comma2k19 import score as SC

DT = 0.5
HORIZON = 3.0


def _straight(speed: float = 20.0, n: int = 8, offset: float = 0.0) -> np.ndarray:
    """A constant-speed straight line, optionally displaced `offset` m to the left."""
    return np.stack([speed * DT * np.arange(1, n + 1), np.full(n, offset)], axis=1)


def _score(scored=None, pred=None, gt=None) -> dict[str, float]:
    pred = _straight() if pred is None else np.asarray(pred, dtype=np.float64)
    gt = pred.copy() if gt is None else np.asarray(gt, dtype=np.float64)
    base = {"comfort_within_bounds": 1.0}
    base.update(scored or {})
    return SC.score(base, pred, gt, interval_s=DT, eval_horizon_s=HORIZON)


def test_a_perfect_plan_on_an_empty_road_scores_one():
    """The ceiling is reachable, so a number below it means something."""
    out = _score()
    assert out["pdm_style_score"] == pytest.approx(1.0)
    assert out["pdm_nc_proxy"] == 1.0  # no radar lead at all
    assert out["pdm_ttc"] == 1.0
    assert out["pdm_dac_proxy"] == 1.0
    assert out["pdm_ep"] == pytest.approx(1.0)


def test_every_declared_key_comes_back_whatever_the_model_published():
    """One metric list serves every model, so the keys cannot depend on inputs.

    A model with no extras has no `lead_gap_min_m` and no `ttc_below_2s`; it
    must still produce the whole column set, or its rows and a richer model's
    rows would not line up in the report.
    """
    assert set(_score()) == set(SC.keys())
    assert set(_score({"fde@3s": 4.0, "lead_gap_min_m": 2.0})) == set(SC.keys())


class TestMissRate:
    def test_a_miss_is_further_than_the_threshold_at_that_horizon(self):
        out = _score({"fde@2s": 1.9, "fde@3s": 2.1, "fde@4s": 100.0})
        assert out["miss_rate@2s"] == 0.0
        assert out["miss_rate@3s"] == 1.0
        assert out["miss_rate@4s"] == 1.0

    def test_exactly_at_the_threshold_is_not_a_miss(self):
        """`>` not `>=`: 2 m is the tolerance, so 2 m is inside it."""
        assert _score({"fde@3s": SC.MISS_THRESHOLD_M})["miss_rate@3s"] == 0.0

    def test_a_horizon_the_plan_does_not_reach_is_nan_not_a_miss(self):
        """A short plan must not be counted as having missed.

        It would be defensible to call an unreached horizon a miss -- the car
        did not get there either -- but `fde@Ns` is already NaN in that case,
        and a rate averaged over a mix of "missed" and "did not answer" is two
        different measurements in one column.
        """
        out = _score({"fde@3s": float("nan")})
        assert np.isnan(out["miss_rate@3s"])
        assert np.isnan(out["miss_rate@4s"])  # absent from `scored` entirely


class TestCollisionAndTtc:
    def test_an_empty_road_passes_rather_than_going_unmeasured(self):
        """No near-range lead means nothing to hit, which is a genuine 1.0."""
        out = _score()
        assert out["pdm_nc_proxy"] == 1.0
        assert out["pdm_ttc"] == 1.0

    def test_closing_the_gap_to_zero_zeroes_the_whole_composite(self):
        """NC multiplies, so a collision is not something the other terms offset."""
        out = _score({"lead_gap_min_m": 0.0})
        assert out["pdm_nc_proxy"] == 0.0
        assert out["pdm_style_score"] == 0.0

    def test_a_gap_that_stays_open_is_not_a_collision(self):
        assert _score({"lead_gap_min_m": 0.4})["pdm_nc_proxy"] == 1.0

    def test_ttc_is_the_complement_of_the_time_spent_under_two_seconds(self):
        assert _score({"ttc_below_2s": 1.0})["pdm_ttc"] == 0.0
        assert _score({"ttc_below_2s": 0.0})["pdm_ttc"] == 1.0

    def test_ttc_alone_cannot_zero_the_composite(self):
        """It is one of three weighted terms, unlike NC and DAC which multiply.

        This is NAVSIM's shape, kept: a near miss is heavily penalised, an
        actual collision is disqualifying, and the difference between them is
        the point of the formula having two halves.
        """
        out = _score({"ttc_below_2s": 1.0})
        assert out["pdm_style_score"] == pytest.approx(
            (SC.WEIGHTS["ep"] + SC.WEIGHTS["comfort"]) / sum(SC.WEIGHTS.values())
        )


class TestCorridor:
    def test_half_a_lane_off_is_still_inside(self):
        pred = _straight(offset=SC.CORRIDOR_HALF_WIDTH_M - 0.05)
        assert _score(pred=pred, gt=_straight())["pdm_dac_proxy"] == 1.0

    def test_a_lane_away_is_outside_and_zeroes_the_composite(self):
        out = _score(pred=_straight(offset=SC.CORRIDOR_HALF_WIDTH_M + 0.05), gt=_straight())
        assert out["pdm_dac_proxy"] == 0.0
        assert out["pdm_style_score"] == 0.0

    def test_one_waypoint_outside_is_enough(self):
        """The corridor is a bound on the worst waypoint, not on the average.

        A plan that leaves the lane for a second and comes back has left the
        lane; averaging would let a large excursion hide behind the waypoints
        either side of it.
        """
        pred = _straight()
        pred[2, 1] = 5.0
        assert _score(pred=pred, gt=_straight())["pdm_dac_proxy"] == 0.0

    def test_the_corridor_is_measured_only_to_the_scored_horizon(self):
        """Past 3 s the pair is not compared, so a departure there is not scored.

        Otherwise `pdm_style_score` and `ade` would disagree about which
        waypoints the sample is being judged on, and a 5 s planner would be
        penalised for waypoints a 3 s planner is never asked about.
        """
        pred = _straight()
        pred[-1, 1] = 50.0  # t = 4.0 s, outside the 3 s horizon
        assert _score(pred=pred, gt=_straight())["pdm_dac_proxy"] == 1.0


class TestProgress:
    def test_stopping_short_of_the_human_loses_progress(self):
        out = _score(pred=_straight(speed=10.0), gt=_straight(speed=20.0))
        assert out["pdm_ep"] == pytest.approx(0.5)

    def test_going_nowhere_scores_zero_progress_not_a_nan(self):
        """The plan-that-does-nothing case, which is the whole reason EP exists.

        An all-zero trajectory is a commanded emergency stop in paddock, and
        against a human doing 20 m/s it must cost the model its progress term
        rather than quietly dropping the sample from the average.
        """
        out = _score(pred=np.zeros((8, 2)), gt=_straight())
        assert out["pdm_ep"] == 0.0
        assert out["pdm_style_score"] == pytest.approx(
            (SC.WEIGHTS["ttc"] + SC.WEIGHTS["comfort"]) / sum(SC.WEIGHTS.values())
        )

    def test_overshooting_the_human_is_capped_rather_than_rewarded(self):
        """NAVSIM's `min(1, ego/expert)`. Driving too far is not extra credit."""
        assert _score(pred=_straight(speed=40.0), gt=_straight(speed=20.0))["pdm_ep"] == 1.0

    def test_a_stationary_recording_gives_full_progress(self):
        """Against a human who went nowhere, nothing falls short.

        This is where `pdm_ep` and the reported `path_length_ratio` deliberately
        differ -- the ratio is NaN at a standstill because its denominator is
        ~0. Whether sitting still was the right call is NC and TTC's business.
        """
        stopped = np.zeros((8, 2))
        assert _score(pred=stopped, gt=stopped)["pdm_ep"] == 1.0
        assert _score(pred=_straight(), gt=stopped)["pdm_ep"] == 1.0


class TestNanDiscipline:
    def test_an_unmeasurable_comfort_term_is_not_a_free_pass(self):
        """A plan too short to have a jerk scores NaN, and NaN propagates.

        This is the single most important line in the module. Comfort is the
        one term a model can fail to answer without failing to drive, so a
        0-for-NaN slip here would let a two-waypoint planner outscore a
        complete one.
        """
        out = _score({"comfort_within_bounds": float("nan")})
        assert np.isnan(out["pdm_comfort"])
        assert np.isnan(out["pdm_style_score"])

    def test_an_uncomfortable_plan_is_scored_not_skipped(self):
        out = _score({"comfort_within_bounds": 0.0})
        assert out["pdm_comfort"] == 0.0
        assert out["pdm_style_score"] == pytest.approx(
            (SC.WEIGHTS["ttc"] + SC.WEIGHTS["ep"]) / sum(SC.WEIGHTS.values())
        )

    def test_a_pair_with_no_overlap_at_all_is_nan_throughout(self):
        """`T.align` returns an empty pair when the two time ranges do not meet.

        That is "not scoreable" rather than an error -- `decompose` already
        treats it that way -- and it must not arrive here as a corridor the
        plan trivially stayed inside of.
        """
        out = _score(pred=np.empty((0, 2)), gt=np.empty((0, 2)))
        assert np.isnan(out["pdm_dac_proxy"])
        assert np.isnan(out["pdm_style_score"])

    def test_a_single_waypoint_is_still_scoreable(self):
        """It has no tangent, but `path_tangent_split` assumes straight ahead.

        Asserted rather than left implicit: the fallback is what makes "cross"
        mean "lateral" for a one-waypoint plan, and a future change that made
        it NaN instead would silently drop those samples from the composite.
        """
        out = _score(pred=_straight(n=1, offset=3.0), gt=_straight(n=1))
        assert out["pdm_dac_proxy"] == 0.0

    @pytest.mark.parametrize("junk", [None, "", "n/a", float("inf")])
    def test_a_non_numeric_input_reads_as_unmeasured(self, junk):
        """Robustness at the seam: these arrive from a model's own extras.

        `auxiliary.score` is careful, but `score` reads a dict it did not build,
        and the failure mode of a bare `float(...)` here is a crashed sweep
        several hours in.
        """
        assert np.isnan(_score({"comfort_within_bounds": junk})["pdm_comfort"])


def test_the_declared_keys_and_the_emitted_keys_agree():
    """`benchmark.yaml` is hand-written; this is what keeps it true.

    `tests/test_report.py` already requires every *declared* key to sit in
    exactly one `report.GROUPS` entry. Neither that test nor this one alone is
    enough: together they close the loop from what `score` emits, through what
    the benchmark advertises, to what the report prints.
    """
    declared = yaml.safe_load((Path(__file__).resolve().parents[1] / "benchmark.yaml").read_text())["metrics"]
    assert set(SC.keys()) <= set(declared)
    assert [k for k in declared if k in set(SC.keys())] == list(SC.keys())


def test_the_report_group_carries_exactly_this_module_s_keys():
    groups = {title: keys for title, _, keys in R.GROUPS}
    assert groups["Driving score (proxy)"] == SC.keys()
