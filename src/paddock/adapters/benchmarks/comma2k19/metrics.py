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
other open-loop benchmarks), plus 1, 2, 5 and 10 s reported alongside. 10 s is
not an arbitrary ceiling -- it is `ModelConstants.T_IDXS[-1]`, the horizon
supercombo plans over, and comma2k19 is comma's dataset for comma's model. The
segments are 60 s, so the recorded future is nowhere near the constraint; the
*prediction* is, and a horizon the model does not reach comes back NaN rather
than as a shorter horizon wearing a longer label.

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


#: Reported in addition to the 3 s headline. `ade@1s`/`ade@2s` are inside the
#: headline horizon and still worth having: they separate a model that is wrong
#: from the first waypoint from one that drifts only late.
REPORTED_HORIZONS_S = (1.0, 2.0, 5.0, 10.0)


def score_trajectory(
    pred_xy,
    gt_xy,
    interval_s: float = 0.5,
    eval_horizon_s: float = 3.0,
    horizons_s: tuple[float, ...] = REPORTED_HORIZONS_S,
) -> dict[str, float]:
    """Metrics for one prediction. Keys are the canonical paddock names.

    `pred_xy` and `gt_xy` must already be on a common time grid of spacing
    `interval_s` -- `harness.score` puts them there with
    `paddock.metrics.trajectory.align`, which is the only thing that makes
    `interval_s` true for both at once.
    """
    return decompose(
        pred_xy,
        gt_xy,
        interval_s=interval_s,
        eval_horizon_s=eval_horizon_s,
        extra_horizons_s=horizons_s,
    )
