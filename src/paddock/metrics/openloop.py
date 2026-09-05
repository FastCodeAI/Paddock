"""Open-loop trajectory metrics that decompose *how* a prediction is wrong.

`paddock.metrics.trajectory` answers "by how much". This module answers "in
which way", and the distinction is not academic -- it is the difference between
a usable finding and a misleading one.

A driving model's plan is a **path plus a speed profile**, and the two fail
independently:

* right path, 10 % slow -> large ADE, almost entirely along-track;
* right speed, drifting a lane -> small ADE.

Only the second is a lateral-control failure. A single ADE cannot tell them
apart, and neither can the axis-aligned split in
`trajectory.longitudinal_lateral` once the road curves: on a curve part of a
pure speed error lands in ego-y and reads as a lane-keeping failure.

This was not hypothetical. comma2k19's first published result was ADE 38.8 m
with `lateral_mae` 0.96 m, read as "tracks the lane, misjudges speed". The
model had in fact been shown a rotated image and was predicting a **stationary
car**: the prediction sat at the origin, so the lateral figure was just the
ground truth's own small excursion. `path_length_ratio` below is ~0 in that
situation and 1.0 when the distance is right, which turns the whole question
into one number.

Extracted from the comma2k19 adapter, which now delegates here, so the next
open-loop benchmark gets these without reimplementing them. Coordinates are the
ego frame throughout: x forward, y **left**, metres.
"""

from __future__ import annotations

import numpy as np

from paddock.metrics import trajectory as T

__all__ = [
    "MIN_GT_PATH_LENGTH_M",
    "MIN_PRED_PATH_LENGTH_M",
    "decompose",
    "lateral_at_arclength",
    "path_length_ratio",
    "path_tangent_split",
]

#: Below this predicted path length the *shape* metrics are undefined rather
#: than zero. A plan that goes nowhere has no shape to be right about, and a
#: naive implementation reports a perfect score for it -- see
#: `lateral_at_arclength`.
MIN_PRED_PATH_LENGTH_M = 0.5

#: Below this *recorded* path length, `path_length_ratio` is undefined: the car
#: was stopped and there is no distance to be right about.
#:
#: Not a nicety. On comma2k19's dense run 5.6 % of frames have under 1 m of
#: recorded travel over a 10 s horizon (stopped in traffic) and the ratio there
#: reached 30,000, dragging the *mean* from a correct 1.02 to a meaningless 3.2.
#: Prefer the median for any ratio regardless: it is dominated by whichever
#: sample had the smallest denominator.
MIN_GT_PATH_LENGTH_M = 2.0


def path_tangent_split(pred_xy, gt_xy) -> tuple[np.ndarray, np.ndarray]:
    """Per-waypoint (along-track, cross-track) error against the GT path tangent.

    The tangent is taken from the **ground truth**, not the prediction, so the
    decomposition does not move when the prediction does. Sign convention:
    along-track is positive when the prediction is ahead of the recording;
    cross-track is positive to the left of the recorded path.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred_xy), len(gt_xy))
    if n == 0:
        return np.empty(0), np.empty(0)
    pred_xy, gt_xy = pred_xy[:n], gt_xy[:n]

    d = np.gradient(gt_xy, axis=0) if n > 1 else np.array([[1.0, 0.0]])
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    # A stationary car has no tangent; straight ahead is the sensible limit and
    # makes "along" mean forward, "cross" mean lateral.
    tangent = np.where(norm > 1e-6, d / np.maximum(norm, 1e-12), np.array([1.0, 0.0]))
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    err = pred_xy - gt_xy
    return np.einsum("nk,nk->n", err, tangent), np.einsum("nk,nk->n", err, normal)


def lateral_at_arclength(pred_xy, gt_xy) -> np.ndarray:
    """Lateral error with speed error removed, per waypoint.

    Both paths are reparametrised by arc length **measured from the ego
    origin**, then the prediction's lateral coordinate is read at the arc
    lengths the ground truth reached. A model driving the right line at the
    wrong speed scores zero here, which is the intent: that is a longitudinal
    failure and is reported as one.

    Two details a first version of this got wrong, both caught by tests:

    *Arc length starts at the car, not at the first waypoint.* A `Trajectory`
    has no waypoint for the current position, so waypoint 0 is already
    `speed * dt` metres out -- and by a *different* distance for the prediction
    than the ground truth whenever the speed estimate differs. Measuring from
    the first waypoint compares two different points on the same curve and
    reintroduces exactly the speed error this metric removes (0.15 m -> 0.01 m
    of spurious error on a 200 m-radius curve).

    *A plan that goes nowhere has no shape.* A stationary prediction has zero
    arc length, the only ground-truth arc length inside its range is 0, and a
    naive implementation reports one perfect sample -- mean lateral error 0.0,
    for a plan that predicted nothing. So a prediction shorter than
    `MIN_PRED_PATH_LENGTH_M` returns all-NaN and the caller reports "not
    scoreable" rather than "perfect".

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

    pred_s, gt_s_full = arclen(pred_xy), arclen(gt_xy)
    if pred_s[-1] < MIN_PRED_PATH_LENGTH_M:
        return np.full(n, np.nan)

    pred_y = np.concatenate([[0.0], pred_xy[:, 1]])
    gt_s = gt_s_full[1:]  # drop the prepended origin; it is trivially equal
    out = np.interp(gt_s, pred_s, pred_y) - gt_xy[:, 1]
    return np.where(gt_s <= pred_s[-1] + 1e-9, out, np.nan)


def path_length_ratio(pred_xy, gt_xy) -> tuple[float, float, float]:
    """(predicted length, recorded length, ratio) in metres.

    The ratio is NaN when the car barely moved. **1.0 is right; ~0 means the
    model predicted a stationary world**, which is a broken input pipeline
    rather than a modelling limitation, and is invisible in ADE alone.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred_xy), len(gt_xy))
    if n < 2:
        return 0.0, 0.0, float("nan")
    pred_len = float(np.sum(np.linalg.norm(np.diff(pred_xy[:n], axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_xy[:n], axis=0), axis=1)))
    ratio = pred_len / gt_len if gt_len > MIN_GT_PATH_LENGTH_M else float("nan")
    return pred_len, gt_len, ratio


def decompose(
    pred_xy,
    gt_xy,
    interval_s: float = 0.5,
    eval_horizon_s: float = 3.0,
    extra_horizons_s: tuple[float, ...] = (),
) -> dict[str, float]:
    """The full open-loop metric set for one prediction, keyed by canonical name.

    `ade` and `fde@Ns` come from `paddock.metrics.trajectory` unchanged, so a
    benchmark using this stays comparable with every other one in the repo; the
    decomposition is added on top.

    `extra_horizons_s` adds `ade@Ns` / `fde@Ns` beyond `eval_horizon_s`, for a
    model whose plan is much longer than the benchmark's headline horizon
    (supercombo plans to 10 s; a 3 s window does not exercise it).
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt_xy = np.asarray(gt_xy, dtype=np.float64)[:, :2]

    metrics = T.displacement_errors(pred_xy, gt_xy, interval_s, eval_horizon_s)
    horizon = min(len(pred_xy), len(gt_xy), int(round(eval_horizon_s / interval_s)))

    for h in extra_horizons_s:
        end = min(len(pred_xy), len(gt_xy), int(round(h / interval_s)))
        if end > horizon:
            err = np.linalg.norm(pred_xy[:end] - gt_xy[:end], axis=1)
            metrics[f"ade@{h:g}s"] = float(err.mean())
            metrics[f"fde@{h:g}s"] = float(err[-1])

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

    pred_len, gt_len, ratio = path_length_ratio(pred_xy[:horizon], gt_xy[:horizon])
    metrics["predicted_path_length"] = pred_len
    metrics["recorded_path_length"] = gt_len
    metrics["path_length_ratio"] = ratio
    return metrics
