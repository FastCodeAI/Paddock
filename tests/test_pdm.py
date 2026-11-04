"""Tests for `paddock.metrics.pdm` and the open-loop benchmarks that fill its terms.

The claims a wrong answer would hide: the formula is NAVSIM v1's with its
weights, a missing term never passes (the composite is NaN), progress is capped
at the human's distance, and every open-loop scorer that declares `pdm_ep` and
`pdm_comfort` actually emits them.

Then `score_plan`, NAVSIM's terms where the other road users' motion is known,
on synthetic scenes: the ego starts at the origin facing +x (identity pose),
plans six waypoints at 0.5 s, and meets other road users placed by hand. Each
test pins one of NAVSIM's rules as the port has it.
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


# ---- score_plan: every term from the other road users' motion ----------

EYE = np.eye(4)
DT = 0.5


def straight(speed: float, n: int = 6, y: float = 0.0) -> np.ndarray:
    """A plan at constant speed along +x."""
    t = np.arange(1, n + 1) * DT
    return np.stack([speed * t, np.full(n, y)], axis=1)


def track(
    x0: float, y0: float, vx: float, vy: float = 0.0, *, length=4.5, width=1.9, agent=True
) -> pdm.Track:
    """A road user moving at constant velocity, logged at 2 Hz for 6 s."""
    t = np.arange(0.0, 6.01, 0.5)
    xy = np.stack([x0 + vx * t, y0 + vy * t], axis=1)
    yaw = np.full(len(t), math.atan2(vy, vx) if (vx or vy) else 0.0)
    return pdm.Track(t * 1e6, xy, yaw, length, width, agent)


def road(y_min: float = -6.0, y_max: float = 6.0) -> pdm.DrivableArea:
    """A straight road from x = -20 to 200 between y_min and y_max, at 0.25 m."""
    cell = 0.25
    origin = np.array([-20.0, -20.0])
    xs = origin[0] + (np.arange(int(220 / cell)) + 0.5) * cell
    ys = origin[1] + (np.arange(int(40 / cell)) + 0.5) * cell
    mask = (ys[:, None] >= y_min) & (ys[:, None] <= y_max) & (xs[None, :] >= -20)
    return pdm.DrivableArea(mask=mask, origin=origin, cell=cell)


def score(plan, tracks, **kw) -> dict[str, float]:
    return pdm.score_plan(plan, DT, EYE, 0.0, tracks, **kw)


def test_hitting_a_stopped_car_ahead_is_at_fault_and_a_ttc_violation() -> None:
    line = np.array([[-10.0, 0.0], [200.0, 0.0]])
    out = score(straight(10.0), [track(20.0, 0.0, 0.0)], area=road(), centreline=line)
    assert out["pdm_nc"] == 0.0 and out["pdm_ttc"] == 0.0
    assert out["pdm_style_score"] == 0.0


def test_following_a_car_that_pulls_away_is_clear() -> None:
    out = score(straight(10.0), [track(25.0, 0.0, 12.0)], area=road())
    assert out["pdm_nc"] == 1.0 and out["pdm_ttc"] == 1.0


def test_closing_on_a_slower_car_breaks_ttc_before_it_collides() -> None:
    """At 10 m/s on a car 25 m ahead doing 4 m/s, bumper to bumper 20.2 m, the gap
    closes at 6 m/s: still 2.2 m at 3 s, so no collision, but 0.9 s of carrying on
    at 10 m/s covers it."""
    out = score(straight(10.0), [track(25.0, 0.0, 4.0)], area=road())
    assert out["pdm_nc"] == 1.0
    assert out["pdm_ttc"] == 0.0


def test_being_rear_ended_is_not_the_egos_fault() -> None:
    """A car from behind at more than twice the ego's speed runs into it: nuPlan calls
    that an active rear collision, not at fault. (NAVSIM's TTC still fires on it once
    the car's projected position passes the ego's current rear axle; the port keeps
    that, so only NC is pinned here.)"""
    out = score(straight(5.0), [track(-12.0, 0.0, 12.0)], area=road())
    assert out["pdm_nc"] == 1.0


def test_a_stopped_ego_that_is_hit_is_not_at_fault() -> None:
    plan = np.zeros((6, 2))
    out = score(plan, [track(15.0, 0.0, -8.0)], area=road())
    assert out["pdm_nc"] == 1.0 and out["pdm_ttc"] == 1.0


def test_a_static_object_costs_half() -> None:
    out = score(straight(10.0), [track(20.0, 0.0, 0.0, length=0.5, width=0.5, agent=False)], area=road())
    assert out["pdm_nc"] == 0.5


def test_a_lateral_collision_is_at_fault_only_off_the_drivable_area() -> None:
    """A car alongside drifts into the ego's flank: not the front bumper, not behind.
    On the road it is the other's fault; with the ego's corners off the road, NAVSIM
    holds the ego responsible."""
    side = [track(0.5, 3.0, 10.0, -1.6, length=3.0)]  # alongside, clear of the front bumper
    assert score(straight(10.0), side, area=road())["pdm_nc"] == 1.0
    narrow = road(-0.5, 6.0)  # the ego's right corners are off it from the start
    out = score(straight(10.0), side, area=narrow)
    assert out["pdm_dac"] == 0.0 and out["pdm_nc"] == 0.0


def test_a_parked_car_that_pulls_into_the_egos_side_is_the_egos_fault() -> None:
    """Review of #102 (4): NAVSIM calls the other "stopped" by its speed when first
    seen, not at contact. Parked beside the lane for 1 s, then sliding into the
    ego's flank, behind its front bumper and on the road: at fault in NAVSIM."""
    t = np.arange(0.0, 6.01, 0.5)
    y = np.where(t <= 1.0, 2.6, 2.6 - 1.5 * (t - 1.0))
    pulls_out = pdm.Track(t * 1e6, np.stack([np.full(len(t), 4.0), y], axis=1), np.zeros(len(t)), 4.5, 1.9)
    assert score(straight(4.0), [pulls_out], area=road())["pdm_nc"] == 0.0
    moving = track(4.0, 4.1, 0.0, -1.5)  # the same slide, but already moving when first seen
    assert score(straight(4.0), [moving], area=road())["pdm_nc"] == 1.0


def test_leaving_the_road_fails_dac_and_no_map_is_nan_not_a_pass() -> None:
    swerve = np.array([[5, 1], [10, 3], [15, 6], [20, 9], [25, 12], [30, 15]], dtype=float)
    assert score(swerve, [], area=road())["pdm_dac"] == 0.0
    assert score(straight(10.0), [], area=road())["pdm_dac"] == 1.0
    out = score(straight(10.0), [])
    assert math.isnan(out["pdm_dac"]) and math.isnan(out["pdm_style_score"])


def test_progress_is_normalised_by_the_best_safe_path() -> None:
    line = np.array([[-10.0, 0.0], [200.0, 0.0]])
    faster = pdm.world_states(np.array([0.0, 3.0]), np.array([[0.0, 0.0], [30.0, 0.0]]), 3.0)
    out = score(straight(5.0), [], area=road(), centreline=line, references=[faster])
    assert out["pdm_ep"] == pytest.approx(0.5, abs=1e-6)
    # A reference that would hit something does not count: the plan is the best safe path.
    wall = [track(25.0, 0.0, 0.0)]
    out = score(straight(5.0), wall, area=road(), centreline=line, references=[faster])
    assert out["pdm_nc"] == 1.0 and out["pdm_ep"] == pytest.approx(1.0)
    # Under 5 m of best progress, EP is 1 (NAVSIM's threshold).
    out = score(straight(1.0), [], area=road(), centreline=line)
    assert out["pdm_ep"] == 1.0
    assert math.isnan(score(straight(5.0), [], area=road())["pdm_ep"]), "no centreline, no EP"


def test_comfort_and_the_composite() -> None:
    line = np.array([[-10.0, 0.0], [200.0, 0.0]])
    calm = score(straight(10.0), [], area=road(), centreline=line)
    assert calm["pdm_comfort"] == 1.0 and calm["pdm_style_score"] == pytest.approx(1.0)
    jerky = np.array([[1, 0], [9, 0], [10, 0], [22, 0], [23, 0], [35, 0]], dtype=float)
    out = score(jerky, [], area=road(), centreline=line)
    assert out["pdm_comfort"] == 0.0
    expected = out["pdm_nc"] * out["pdm_dac"] * (5 * out["pdm_ep"] + 5 * out["pdm_ttc"] + 2 * 0.0) / 12
    assert out["pdm_style_score"] == pytest.approx(expected)


def test_every_key_is_returned_and_every_term_is_higher_is_better() -> None:
    from paddock.results.report import has_direction, lower_is_better

    out = score(straight(10.0), [])
    assert tuple(out) == pdm.PDM_KEYS
    for key in pdm.PDM_KEYS:
        assert has_direction(key) and not lower_is_better(key), key


def test_headings_follow_the_path_and_standing_still_keeps_the_last() -> None:
    assert pdm.derive_headings(straight(5.0)) == pytest.approx(np.zeros(6))
    left = np.array([[1.0, 1.0], [1.0, 1.01], [1.0, 2.0]])
    h = pdm.derive_headings(left)
    assert h[0] == pytest.approx(math.pi / 4)
    assert h[1] == pytest.approx(math.pi / 4), "a 1 cm step keeps the previous heading"
    assert h[2] == pytest.approx(math.pi / 2)


def test_overlap_counts_touching_and_tests_a_segment_against_a_box() -> None:
    a = pdm.box_corners(np.array([0.0, 0.0]), np.array(0.0), 2.0, 2.0)
    b = pdm.box_corners(np.array([2.0, 0.0]), np.array(0.0), 2.0, 2.0)  # shares an edge
    c = pdm.box_corners(np.array([2.01, 0.0]), np.array(0.0), 2.0, 2.0)
    assert pdm.convex_overlap(a, b) and not pdm.convex_overlap(a, c)
    tilted = pdm.box_corners(np.array([0.0, 0.0]), np.array(math.pi / 4), 2.0, 2.0)
    far_segment = np.array([[1.5, 1.5], [3.0, 3.0]])  # along the diagonal, beyond the corner
    near_segment = np.array([[0.5, 0.5], [3.0, 3.0]])
    assert not pdm.convex_overlap(far_segment, tilted) and pdm.convex_overlap(near_segment, tilted)


def test_a_plan_in_another_frame_lands_where_the_pose_puts_it() -> None:
    pose = np.eye(4)
    yaw = math.pi / 2
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:2, 3] = [100.0, 50.0]
    states = pdm.plan_states(straight(10.0), DT, pose)
    assert states.xy[0] == pytest.approx([100.0, 50.0])
    assert states.xy[-1] == pytest.approx([100.0, 80.0])
    assert states.yaw[-1] == pytest.approx(yaw)
    assert states.speed[1:] == pytest.approx(np.full(len(states.taus) - 1, 10.0))


def test_a_plan_that_stands_still_is_comfortable_as_in_navsim() -> None:
    """`metrics.comfort` cannot give a stopped plan a yaw rate; PDM's C passes it, as
    NAVSIM passes a stopped car, so a run that waits is not made unmeasurable."""
    assert pdm.plan_comfort(np.zeros((6, 2)), 0.5) == 1.0
    assert math.isnan(pdm.plan_comfort(np.zeros((2, 2)), 0.5)), "too short is still not a pass"
    lurch = np.array([[1, 0], [9, 0], [10, 0], [22, 0], [23, 0], [35, 0]], dtype=float)
    assert pdm.plan_comfort(lurch, 0.5) == 0.0
