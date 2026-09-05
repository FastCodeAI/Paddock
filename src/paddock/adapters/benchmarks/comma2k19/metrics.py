"""comma2k19's metric definition.

## Why this is defined here rather than taken from the dataset

comma2k19 publishes no model-evaluation protocol. Its own benchmark code
(`utils/benchmarks.py::get_altitude_errors`, `notebooks/position_benchmark.ipynb`)
scores *GNSS positioning* -- it bins the road into 5 m squares and measures
altitude deviation within each bin, which evaluates the dataset's own ground
truth, not a driving model. What comma2k19 contributes to model evaluation is
the ground truth itself.

So the metric set is derived from what the dataset can actually ground, and
`paddock.metrics.trajectory` supplies ADE/FDE so the headline numbers are
computed by the same code as every other paddock open-loop benchmark.

## The one addition, and why it is not optional

`paddock.metrics.trajectory.longitudinal_lateral` splits error into the
**x and y components of the ego frame**. That is the right split when the car is
going straight and the wrong one the moment it is not: on a curve, part of a
pure speed error lands in y and reads as a lane-keeping failure.

`score_trajectory` therefore also reports a split against the **ground-truth
path tangent** (`along_track` / `cross_track`), and a third quantity,
`lateral_at_arclength`, which resamples the predicted path to the arc length the
car actually travelled before measuring lateral offset. That last one removes
speed error from the lateral number entirely, and it is what openpilot's lateral
controller actually consumes -- it follows a path, not a schedule.

This matters concretely. The previous comma2k19 run reported ADE 38.8 m with
`lateral_mae` 0.96 m and read the low lateral figure as "lane tracking works
fine, the model just misjudges speed". It did not: the model had been shown a
sideways image and was predicting a *stationary* car, so the prediction sat at
the origin and `lateral_mae` was just the ground truth's own small lateral
excursion. A metric that cannot tell "tracks the lane" from "predicts nothing"
is not a safe metric to draw a conclusion from, and `predicted_path_length` is
reported alongside for exactly that reason: a plan that goes nowhere is visible
in one number.
"""

from __future__ import annotations

import numpy as np

from paddock.metrics import trajectory as T

__all__ = ["score_trajectory", "path_tangent_split", "lateral_at_arclength"]


def path_tangent_split(pred_xy: np.ndarray, gt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-waypoint (along-track, cross-track) error against the GT path tangent.

    The tangent comes from the ground truth, not the prediction, so the
    decomposition does not move when the prediction does.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred_xy), len(gt_xy))
    if n == 0:
        return np.empty(0), np.empty(0)
    pred_xy, gt_xy = pred_xy[:n], gt_xy[:n]

    d = np.gradient(gt_xy, axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    # A stationary car has no tangent; straight ahead is the sensible limit and
    # makes "along" mean forward, "cross" mean lateral.
    tangent = np.where(norm > 1e-6, d / np.maximum(norm, 1e-12), np.array([1.0, 0.0]))
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    err = pred_xy - gt_xy
    return np.einsum("nk,nk->n", err, tangent), np.einsum("nk,nk->n", err, normal)


#: Below this predicted path length the shape metric is undefined rather than
#: zero -- see `lateral_at_arclength`.
MIN_PATH_LENGTH_M = 0.5

#: Below this *recorded* path length over the scored horizon, `path_length_ratio`
#: is undefined: the car was stopped and there is no distance to be right about.
#:
#: Not a nicety. On the reference implementation's dense run, 5.6 % of frames
#: have under 1 m of recorded travel over a 10 s horizon -- stopped in traffic --
#: and the ratio there reached 30,000, which dragged the *mean* from a correct
#: 1.02 to a meaningless 3.2. `paddock.results.store` drops NaN and counts it as
#: `n_missing`, so gating here removes those samples from the average and says
#: how many were removed. Prefer the median for this metric in any case: a ratio
#: is dominated by whichever sample had the smallest denominator.
MIN_GT_PATH_LENGTH_M = 2.0


def lateral_at_arclength(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    """Lateral error with speed error removed, per waypoint.

    Both paths are reparametrised by arc length **measured from the ego
    origin**, then the prediction's lateral coordinate is read at the arc
    lengths the ground truth reached. A model that drives the right line at the
    wrong speed scores zero here, which is the intent -- that is a longitudinal
    failure and it is reported as one.

    Two details that a first version of this function got wrong, and that the
    tests in `tests/test_harness.py` caught:

    *Arc length starts at the car, not at the first waypoint.* Paddock's
    `Trajectory` has no waypoint for the current position, so waypoint 0 is
    already `speed * dt` metres out -- and by a *different* distance for the
    prediction than for the ground truth whenever the speed estimate differs.
    Measuring from the first waypoint therefore compares two different points on
    the same curve, which reintroduces exactly the speed error this metric
    exists to remove. Prepending the origin fixes it (0.15 m -> 0.01 m of
    spurious error on a 200 m-radius curve).

    *A plan that goes nowhere has no shape.* A stationary prediction has zero
    arc length, so the only ground-truth arc length inside its range is 0, and a
    naive implementation reports a single perfect sample: mean lateral error
    0.0, for a plan that predicted nothing. That is the precise failure this
    metric was added to expose, so a prediction shorter than
    `MIN_PATH_LENGTH_M` returns all-NaN and the caller reports "not scoreable"
    rather than "perfect".

    Ground-truth waypoints beyond the end of the predicted path also get NaN, so
    "the plan did not reach this far" stays distinguishable from "the plan was
    right there".
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred_xy), len(gt_xy))
    if n < 1:
        return np.empty(0)
    pred_xy, gt_xy = pred_xy[:n], gt_xy[:n]

    origin = np.zeros((1, 2))

    def arclen(xy: np.ndarray) -> np.ndarray:
        path = np.concatenate([origin, xy])
        return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])

    pred_s_full, gt_s_full = arclen(pred_xy), arclen(gt_xy)
    if pred_s_full[-1] < MIN_PATH_LENGTH_M:
        return np.full(n, np.nan)

    pred_y_full = np.concatenate([[0.0], pred_xy[:, 1]])
    gt_s = gt_s_full[1:]  # drop the prepended origin; it is trivially equal
    out = np.interp(gt_s, pred_s_full, pred_y_full) - gt_xy[:, 1]
    return np.where(gt_s <= pred_s_full[-1] + 1e-9, out, np.nan)


def score_trajectory(
    pred_xy,
    gt_xy,
    interval_s: float = 0.5,
    eval_horizon_s: float = 3.0,
    long_horizon_s: float = 5.0,
) -> dict[str, float]:
    """Metrics for one prediction. Keys are the canonical paddock names.

    `ade` and `fde@Ns` come from `paddock.metrics.trajectory` unchanged, so they
    are comparable with every other open-loop benchmark in the repo.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]

    metrics = T.displacement_errors(pred_xy, gt_xy, interval_s, eval_horizon_s)
    horizon = min(len(pred_xy), len(gt_xy), int(round(eval_horizon_s / interval_s)))
    long_h = min(len(pred_xy), len(gt_xy), int(round(long_horizon_s / interval_s)))

    if long_h > horizon:
        long_err = np.linalg.norm(pred_xy[:long_h] - gt_xy[:long_h], axis=1)
        metrics[f"ade@{long_horizon_s:g}s"] = float(long_err.mean())
        metrics[f"fde@{long_horizon_s:g}s"] = float(long_err[-1])

    # Axis-aligned split, from paddock's shared implementation. Kept because it
    # is what the other benchmarks report; superseded on curves by the tangent
    # split below.
    longitudinal, lateral = T.longitudinal_lateral(pred_xy[:horizon], gt_xy[:horizon])
    metrics["longitudinal_mae"] = longitudinal
    metrics["lateral_mae"] = lateral

    along, cross = path_tangent_split(pred_xy[:horizon], gt_xy[:horizon])
    metrics["along_track_mae"] = float(np.mean(np.abs(along))) if len(along) else float("nan")
    metrics["cross_track_mae"] = float(np.mean(np.abs(cross))) if len(cross) else float("nan")
    metrics["along_track_bias"] = float(np.mean(along)) if len(along) else float("nan")

    lat_arc = lateral_at_arclength(pred_xy[:horizon], gt_xy[:horizon])
    finite = lat_arc[np.isfinite(lat_arc)]
    metrics["lateral_at_arclength_mae"] = float(np.mean(np.abs(finite))) if len(finite) else float("nan")

    # Two length figures, so "the plan went nowhere" is a number rather than an
    # inference from a large ADE. A ratio near 1 means the model got the
    # distance right whatever it did with the shape.
    pred_len = float(np.sum(np.linalg.norm(np.diff(pred_xy[:horizon], axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_xy[:horizon], axis=0), axis=1)))
    metrics["predicted_path_length"] = pred_len
    metrics["recorded_path_length"] = gt_len
    metrics["path_length_ratio"] = (
        pred_len / gt_len if gt_len > MIN_GT_PATH_LENGTH_M else float("nan")
    )
    return metrics
