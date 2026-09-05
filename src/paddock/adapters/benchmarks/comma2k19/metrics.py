"""comma2k19's metric definition.

## Why this is defined at all

comma2k19 publishes no model-evaluation protocol. Its own benchmark code
(`utils/benchmarks.py::get_altitude_errors`, `notebooks/position_benchmark.ipynb`)
scores *GNSS positioning* -- it bins the road into 5 m squares and measures
altitude deviation within each bin, which evaluates the dataset's own ground
truth, not a driving model. What comma2k19 contributes to model evaluation is
the ground truth itself.

## What is here, and what is shared

Almost nothing is here. The metric *set* is
`paddock.metrics.openloop.decompose`, which was extracted from this adapter so
the next open-loop benchmark gets it for free; `ade` and `fde@Ns` inside it come
from `paddock.metrics.trajectory` unchanged, so these numbers stay comparable
with every other open-loop benchmark in the repo.

This module only fixes the horizons: 3 s for the headline (matching paddock's
other open-loop benchmarks) and 5 s in addition, because supercombo plans to
10 s and a 3 s window does not exercise it.

Read `paddock.metrics.openloop` for *why* the decomposition exists. The short
version: a plan is a path plus a speed profile, they fail independently, and
comma2k19's first published result (ADE 38.8 m, `lateral_mae` 0.96 m) was read
as "tracks the lane, misjudges speed" when the model was in fact predicting a
stationary car.
"""

from __future__ import annotations

from paddock.metrics.openloop import (
    MIN_GT_PATH_LENGTH_M,
    MIN_PRED_PATH_LENGTH_M,
    decompose,
    lateral_at_arclength,
    path_length_ratio,
    path_tangent_split,
)

__all__ = [
    "MIN_GT_PATH_LENGTH_M",
    "MIN_PRED_PATH_LENGTH_M",
    "lateral_at_arclength",
    "path_length_ratio",
    "path_tangent_split",
    "score_trajectory",
]

#: Kept as an alias for the adapter's own tests, which predate the extraction.
MIN_PATH_LENGTH_M = MIN_PRED_PATH_LENGTH_M


def score_trajectory(
    pred_xy,
    gt_xy,
    interval_s: float = 0.5,
    eval_horizon_s: float = 3.0,
    long_horizon_s: float = 5.0,
) -> dict[str, float]:
    """Metrics for one prediction. Keys are the canonical paddock names."""
    return decompose(
        pred_xy,
        gt_xy,
        interval_s=interval_s,
        eval_horizon_s=eval_horizon_s,
        extra_horizons_s=(long_horizon_s,),
    )
