"""Tests for `comma2k19.auxiliary`.

Two properties carry most of the weight. First, **a model that publishes no
extras must get NaN everywhere** -- the declared metric list is one list for
every model, so a missing head has to read as "not answered" and never as a
zero that averages into someone else's score. Second, the sign and frame
conventions: every reference here was recorded in the car's own frame and every
prediction arrives in paddock's, and a mirrored `y` is exactly the kind of
defect that leaves plausible-looking numbers behind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from paddock.adapters.benchmarks.comma2k19 import auxiliary as AUX
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.benchmark import Sample
from paddock.policy.protocol import Trajectory

DT = 0.5


def _sample(**reference) -> Sample:
    return Sample(sample_id="t", observations=[], native_id={"reference": reference})


def _straight(speed: float = 20.0, n: int = 8) -> np.ndarray:
    return np.stack([speed * DT * np.arange(1, n + 1), np.zeros(n)], axis=1)


def _score(extras=None, reference=None, pred=None, gt=None) -> dict[str, float]:
    pred = _straight() if pred is None else np.asarray(pred, dtype=np.float64)
    gt = pred.copy() if gt is None else np.asarray(gt, dtype=np.float64)
    prediction = Trajectory(pred, dt=DT, extras=extras)
    return AUX.score(_sample(**(reference or {})), prediction, pred, gt)


# --------------------------------------------------------------------------
# The contract that lets one benchmark.yaml describe every model
# --------------------------------------------------------------------------


def test_a_model_with_no_extras_scores_nan_for_every_auxiliary_key():
    m = _score(extras=None, reference={"speed": 20.0, "yaw_rate": 0.0})
    assert set(m) == set(AUX.ALL_KEYS)
    assert all(np.isnan(m[k]) for k in AUX.AUX_KEYS), [k for k in AUX.AUX_KEYS if not np.isnan(m[k])]


def test_comfort_is_scored_without_extras_because_it_reads_the_plan_alone():
    """The one group that needs no cooperation from the model."""
    m = _score(extras=None)
    assert m["comfort_within_bounds"] == 1.0


def test_benchmark_yaml_declares_every_key_this_module_produces():
    """The declared list is the conformance contract; drift makes it a lie.

    Both directions matter and only this one is checked here: an undeclared but
    produced key is invisible in a report, while a declared but unproduced one
    is what `paddock.conformance` catches at run time (and what `inference_seconds`
    was doing in this file before the auxiliary metrics arrived).
    """
    spec = yaml.safe_load((Path(AUX.__file__).parent / "benchmark.yaml").read_text())
    declared = set(spec["metrics"])
    assert set(AUX.ALL_KEYS) <= declared
    # The horizons are the other half of the same contract, and the easy half to
    # forget: widening REPORTED_HORIZONS_S produces keys nothing declares.
    horizons = {f"{kind}@{h:g}s" for h in MC.REPORTED_HORIZONS_S for kind in ("ade", "fde")}
    assert horizons <= declared


def test_every_call_returns_exactly_the_declared_keys():
    rich = {
        "pose": {"velocity": [20.0, 0.0, 0.0], "rotation_rate": [0.0, 0.0, 0.1]},
        "lead": [{"d": 30.0, "y": 0.2, "v_rel": -2.0, "a_rel": 0.0, "prob": 0.9}],
        "sigma": [[0.5, 0.3]] * 8,
        "lane": {
            "lines": [[[0.0, 5.4]], [[0.0, 1.85]], [[0.0, -1.85]], [[0.0, -5.4]]],
            "line_probs": [0.2, 0.95, 0.93, 0.2],
            "edges": [[[0.0, 7.0]], [[0.0, -7.0]]],
        },
        "road": {"camera_height": 1.22},
    }
    m = _score(extras=rich, reference={"speed": 20.0, "yaw_rate": 0.1, "lead_d": 30.0})
    assert set(m) == set(AUX.ALL_KEYS)


# --------------------------------------------------------------------------
# Pose
# --------------------------------------------------------------------------


def test_speed_error_is_against_the_measured_speed_and_relative_to_it():
    m = _score(
        extras={"pose": {"velocity": [21.0, 0.0, 0.0], "rotation_rate": [0, 0, 0]}},
        reference={"speed": 20.0},
    )
    assert m["speed_abs_err"] == pytest.approx(1.0)
    assert m["speed_rel_err"] == pytest.approx(0.05)


def test_the_relative_speed_denominator_is_floored_at_walking_pace():
    """At 0.2 m/s a 0.3 m/s error is not a 150 % failure of anything."""
    m = _score(
        extras={"pose": {"velocity": [0.5, 0.0, 0.0], "rotation_rate": [0, 0, 0]}},
        reference={"speed": 0.2},
    )
    assert m["speed_rel_err"] == pytest.approx(0.3)


def test_yaw_rate_keeps_its_sign_as_a_bias_and_drops_it_as_an_error():
    """A systematically-mirrored yaw rate shows as a bias; an unbiased jitter
    averages to zero there and is only visible in the absolute error."""
    m = _score(
        extras={"pose": {"velocity": [20.0, 0, 0], "rotation_rate": [0.0, 0.0, -0.1]}},
        reference={"speed": 20.0, "yaw_rate": 0.1},
    )
    assert m["yaw_rate_abs_err"] == pytest.approx(0.2)
    assert m["yaw_rate_bias"] == pytest.approx(-0.2)


# --------------------------------------------------------------------------
# Lead
# --------------------------------------------------------------------------


def _lead_extras(prob: float = 0.9, d: float = 30.0, y: float = 0.0, v_rel: float = 0.0):
    return {"lead": [{"d": d, "y": y, "v_rel": v_rel, "a_rel": 0.0, "prob": prob}]}


def test_a_confident_lead_matching_the_radar_scores_range_lateral_and_closing():
    m = _score(
        extras=_lead_extras(d=31.0, y=0.5, v_rel=-3.0),
        reference={"speed": 20.0, "lead_d": 30.0, "lead_y": 0.2, "lead_v_rel": -2.0},
    )
    assert m["lead_range_err"] == pytest.approx(1.0)
    assert m["lead_range_abs_err"] == pytest.approx(1.0)
    assert m["lead_lateral_abs_err"] == pytest.approx(0.3)
    assert m["lead_rel_speed_abs_err"] == pytest.approx(1.0)
    assert m["lead_precision"] == 1.0
    assert m["lead_recall"] == 1.0


def test_a_missed_lead_is_a_recall_failure_with_no_precision_number():
    """Precision is undefined when the model claimed nothing, and NaN keeps it
    out of the mean rather than counting as a miss twice."""
    m = _score(extras=_lead_extras(prob=0.1), reference={"lead_d": 30.0})
    assert m["lead_recall"] == 0.0
    assert np.isnan(m["lead_precision"])


def test_a_lead_on_an_empty_road_is_a_precision_failure_with_no_recall_number():
    m = _score(extras=_lead_extras(), reference={"speed": 20.0})
    assert m["lead_precision"] == 0.0
    assert np.isnan(m["lead_recall"])


def test_a_far_radar_lead_is_not_used_as_a_reference_at_all():
    """Past LEAD_NEAR_M the two sensors routinely hold different vehicles, so a
    disagreement measures the sensors and not the model."""
    m = _score(
        extras=_lead_extras(d=80.0),
        reference={"speed": 20.0, "lead_d": AUX.LEAD_NEAR_M + 10.0, "lead_v_rel": 0.0},
    )
    assert np.isnan(m["lead_range_abs_err"])
    assert np.isnan(m["lead_recall"])
    assert np.isnan(m["lead_gap_min_m"])


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def test_sigma_is_scored_against_the_realised_error_on_the_aligned_pair():
    pred = _straight()
    gt = pred + np.array([0.0, 0.4])  # 0.4 m of cross-track error throughout
    m = _score(extras={"sigma": [[1.0, 0.5]] * len(pred)}, pred=pred, gt=gt)
    assert m["within_1sigma_x"] == 1.0  # no longitudinal error at all
    assert m["within_1sigma_y"] == 1.0  # 0.4 < 0.5
    m = _score(extras={"sigma": [[1.0, 0.2]] * len(pred)}, pred=pred, gt=gt)
    assert m["within_1sigma_y"] == 0.0  # 0.4 > 0.2
    assert m["within_2sigma_y"] == 0.0  # and 0.4 > 0.4 is false at the boundary


def test_a_sigma_longer_than_the_aligned_pair_uses_its_matching_prefix():
    """`align` trims the prediction's tail, so the survivors are a prefix."""
    pred = _straight(n=4)
    m = _score(extras={"sigma": [[1.0, 1.0]] * 12}, pred=pred, gt=pred.copy())
    assert m["within_1sigma_x"] == 1.0


# --------------------------------------------------------------------------
# Geometry and gates
# --------------------------------------------------------------------------


def _lane(left: float = 1.85, right: float = -1.85):
    return {
        "lane": {
            "lines": [[[0.0, left + 3.7]], [[0.0, left]], [[0.0, right]], [[0.0, right - 3.7]]],
            "line_probs": [0.2, 0.9, 0.9, 0.2],
            "edges": [[[0.0, 7.0]], [[0.0, -7.0]]],
        },
        "road": {"camera_height": 1.22},
    }


def test_lane_width_and_offset_read_the_ego_lane_in_the_y_left_convention():
    m = _score(extras=_lane(left=2.2, right=-1.5), reference={"speed": 25.0})
    assert m["lane_width_m"] == pytest.approx(3.7)
    # The centre sits 0.35 m to the left, so the car is right of centre.
    assert m["lane_offset_m"] == pytest.approx(0.35)
    assert m["road_width_m"] == pytest.approx(14.0)
    assert m["lane_prob"] == pytest.approx(0.9)


def test_an_implausible_lane_width_fails_the_gate_and_the_overall_pass():
    m = _score(extras=_lane(left=1.0, right=-1.0), reference={"speed": 25.0})
    assert m["gate_lane_width"] == 0.0
    assert m["gate_camera_height"] == 1.0
    assert m["sanity_pass"] == 0.0


def test_geometry_gates_are_not_applicable_below_the_calibration_speed():
    """A crawl has no observable road frame, so a stop-and-go segment must not
    be able to vouch for a warp that is in fact broken."""
    m = _score(extras=_lane(left=1.0, right=-1.0), reference={"speed": 2.0})
    assert np.isnan(m["gate_lane_width"])
    assert np.isnan(m["gate_camera_height"])
    assert m["lane_width_m"] == pytest.approx(2.0)  # still reported, just not gated


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


def test_a_plan_that_drives_into_a_stopped_lead_reports_the_time_it_happens():
    # Lead stopped 20 m ahead (v_rel = -v_ego); the plan holds 20 m/s.
    m = _score(reference={"speed": 20.0, "lead_d": 20.0, "lead_v_rel": -20.0})
    assert m["lead_ttc_min_s"] == pytest.approx(1.0)
    assert m["ttc_below_2s"] == 1.0
    assert m["lead_gap_min_m"] < 0.0


def test_a_lead_pulling_away_has_no_collision_time_rather_than_an_infinite_one():
    m = _score(reference={"speed": 20.0, "lead_d": 30.0, "lead_v_rel": 5.0})
    assert np.isnan(m["lead_ttc_min_s"])
    assert m["ttc_below_2s"] == 0.0
    # The gap is minimised at the first planned waypoint and opens from there:
    # 30 m + 25 m/s * 0.5 s of lead motion - 10 m of ego travel. The present
    # moment is not in the minimum because the gap now is not the plan's doing.
    assert m["lead_gap_min_m"] == pytest.approx(32.5)


def test_an_empty_road_has_no_safety_numbers_to_be_right_about():
    """Otherwise a run on deserted roads would out-score a careful one in traffic."""
    m = _score(reference={"speed": 20.0})
    assert np.isnan(m["lead_gap_min_m"])
    assert np.isnan(m["ttc_below_2s"])
