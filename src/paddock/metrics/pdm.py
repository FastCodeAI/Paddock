"""NAVSIM v1's PDM score shape, shared by every benchmark that can fill a term of it.

    pdm_style_score = NC x DAC x (5*TTC + 5*EP + 2*C) / 12

NAVSIM's weights, unchanged; paddock evaluates someone else's published formula
rather than inventing one (the argument is in
`adapters/benchmarks/comma2k19/score.py`, which first computed it). Each
benchmark's `benchmark.yaml` says, under `pdm:`, which of its metrics fills
which term (`SLOTS`), so the regression gate can lay every benchmark out in the
same five rows. What a term is *made of* differs: NAVSIM's NC re-simulates the
agents, nuScenes' is an overlap with recorded boxes, AlpaSim's is a simulator's
at-fault call. The slot says which question a metric answers, not that two
benchmarks answer it alike.

Two terms any open-loop benchmark can compute from a plan and its recorded
future, and `plan_terms` does: EP (progress against the human, capped at 1) and
C (`metrics.comfort`'s five bounds). NC, DAC and TTC need agents, a map or a
simulator, and a benchmark without them leaves the slot out. A missing term is
never a pass: `composite` is NaN unless all five are finite.
"""

from __future__ import annotations

import math

import numpy as np

from paddock.metrics.comfort import comfort
from paddock.metrics.openloop import MIN_GT_PATH_LENGTH_M, path_length_ratio

__all__ = ["LABELS", "SCORE", "SLOTS", "WEIGHTS", "composite", "ego_progress", "plan_terms"]

#: The five terms, in the order NAVSIM writes the formula.
SLOTS = ("nc", "dac", "ttc", "ep", "comfort")
#: The `pdm:` key naming a benchmark's own composite of the five, when it has one.
SCORE = "score"
#: How the gate labels each slot's row.
LABELS = {"nc": "NC", "dac": "DAC", "ttc": "TTC", "ep": "EP", "comfort": "C", SCORE: "PDM"}

#: NAVSIM v1's weights for the additive half of PDMS. Restated rather than
#: imported -- nuplan-devkit is not a paddock dependency -- so they are cited at
#: the strength of published constants, the same standing as
#: `metrics.comfort.BOUNDS`.
WEIGHTS = {"ttc": 5.0, "ep": 5.0, "comfort": 2.0}


def composite(nc: float, dac: float, ttc: float, ep: float, comfort: float) -> float:
    """NC x DAC x the weighted mean of TTC, EP and C; NaN if any term is not finite."""
    terms = (nc, dac, ttc, ep, comfort)
    if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in terms):
        return float("nan")
    weighted = (WEIGHTS["ttc"] * ttc + WEIGHTS["ep"] * ep + WEIGHTS["comfort"] * comfort) / sum(
        WEIGHTS.values()
    )
    return float(nc * dac * weighted)


def ego_progress(pred_xy, gt_xy) -> float:
    """Distance covered against the distance the human covered, capped at 1.

    The cap is NAVSIM's -- EP is `min(1, ego / expert)` -- so overshooting the
    human's distance earns nothing extra. Driving too far is caught by the
    collision terms; rewarding it here would pay a model twice for the same
    speed prior.

    A car that barely moved scores 1.0 rather than NaN, which is where this and
    the reported `path_length_ratio` deliberately differ. The ratio is undefined
    at a standstill because its denominator is ~0 and it is published as a
    ratio; the question here is only "did the plan fall short of the human", and
    against a human who went nowhere nothing falls short. Whether sitting still
    was *right* is the collision terms' business, not progress's.
    """
    pred_len, gt_len, _ = path_length_ratio(pred_xy, gt_xy)
    if gt_len < MIN_GT_PATH_LENGTH_M:
        return 1.0
    return float(np.clip(pred_len / gt_len, 0.0, 1.0))


def plan_terms(pred_xy, gt_xy, plan_xy, plan_dt: float) -> dict[str, float]:
    """`pdm_ep` and `pdm_comfort` for one open-loop sample.

    `pred_xy` / `gt_xy` are the plan and the recorded future as the benchmark
    already pairs them for its displacement errors, truncated to its horizon, so
    progress covers exactly what `ade` covers. Comfort reads the model's whole
    plan on its own spacing (`plan_xy`, `plan_dt`): a plan that jerks at 5 s is
    uncomfortable whether or not the benchmark scores 5 s. A plan too short to
    have a jerk is NaN, never a pass.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred), len(gt))
    ep = ego_progress(pred[:n], gt[:n]) if n else float("nan")
    return {
        "pdm_ep": ep,
        "pdm_comfort": comfort(plan_xy, plan_dt)["comfort_within_bounds"],
    }
