"""Open-loop trajectory metrics: ADE, FDE, minADE.

Ported from causaldrivebench's `common/trajectory_eval.py`, keeping its
alignment rules, which are the part everyone gets wrong:

1. **Trim the prediction to the ground-truth time range.** Never extrapolate
   recorded future past where the recording ends. A model that plans 6 s ahead
   against a 3 s ground truth is scored on 3 s, not on an invented tail.
2. **Interpolate ground truth onto the prediction's timestamps**, not the other
   way round. Models plan on their own grids; resampling the prediction would
   score a smoothed version of what the model actually said.
3. Pair what remains.

Where prediction and ground truth already share a grid -- the common case, and
what every existing KITScenes run does -- this reduces to elementwise distance,
and the simple entry points below take that path without interpolating.

Coordinates are ego frame, x forward and y left, metres. Waypoint k is at
t = (k+1)*dt: there is no waypoint for the current position.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "displacement_errors",
    "ade",
    "fde",
    "displacements",
    "min_ade",
    "medoid_index",
    "longitudinal_lateral",
    "align",
]

_EPS = 1e-6


def _xy(array) -> np.ndarray:
    out = np.asarray(array, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] < 2:
        raise ValueError(f"expected (N, 2+) waypoints, got shape {out.shape}")
    return out[:, :2]


def displacements(pred, gt) -> np.ndarray:
    """Per-timestep L2 distance, over the overlapping prefix."""
    p, g = _xy(pred), _xy(gt)
    n = min(len(p), len(g))
    if n == 0:
        return np.empty(0)
    return np.linalg.norm(p[:n] - g[:n], axis=1)


def ade(pred, gt) -> float:
    """Average displacement error, metres.

    Both arguments must already be on the same time grid. Use `align` first if
    they are not.
    """
    d = displacements(pred, gt)
    return float(d.mean()) if len(d) else float("nan")


def fde(pred, gt, t: float, dt: float = 0.5) -> float:
    """Final displacement error at horizon `t` seconds.

    Waypoint k sits at (k+1)*dt, so t = 3.0 s with dt = 0.5 is index 5. Returns
    NaN if either trajectory is too short to reach `t` -- silently scoring a
    shorter horizon would make an unfair comparison look like a fair one.
    """
    index = int(round(t / dt)) - 1
    p, g = _xy(pred), _xy(gt)
    if index < 0 or index >= len(p) or index >= len(g):
        return float("nan")
    return float(np.linalg.norm(p[index] - g[index]))


def min_ade(preds, gt) -> float:
    """Best ADE over K sampled trajectories.

    `preds` is (K, N, 2+). Reported as minADE@K. It measures whether the model
    considered the right future at all, which is a different question from
    whether its chosen future was right.
    """
    scores = [ade(p, gt) for p in preds]
    return float(np.nanmin(scores)) if scores else float("nan")


def medoid_index(preds) -> int:
    """Index of the sample closest to all the others.

    The representative pick when a model emits K samples but the benchmark wants
    one. Preferred over the mean, which can land between modes on a trajectory
    that the model never proposed and would never drive.
    """
    stack = np.asarray([_xy(p) for p in preds], dtype=np.float64)
    if len(stack) == 0:
        raise ValueError("no trajectories given")
    diff = stack[:, None, :, :] - stack[None, :, :, :]
    total = np.linalg.norm(diff, axis=3).mean(axis=2).sum(axis=1)
    return int(np.argmin(total))


def longitudinal_lateral(pred, gt) -> tuple[float, float]:
    """Mean absolute error split into along-track and cross-track, metres.

    Worth reporting separately: every model evaluated so far errs far more
    longitudinally than laterally -- it holds the lane and misjudges the speed --
    and a single ADE hides which failure you are looking at.
    """
    p, g = _xy(pred), _xy(gt)
    n = min(len(p), len(g))
    if n == 0:
        return float("nan"), float("nan")
    delta = p[:n] - g[:n]
    return float(np.abs(delta[:, 0]).mean()), float(np.abs(delta[:, 1]).mean())


def align(pred, pred_t, gt, gt_t) -> tuple[np.ndarray, np.ndarray, int]:
    """Put a prediction and a ground truth on a common time grid.

    Applies the three rules in the module docstring. Returns
    `(pred_aligned, gt_aligned, n)`; `n` is 0 when the two do not overlap in
    time, which callers should treat as "not scoreable", not as a zero error.
    """
    p, g = _xy(pred), _xy(gt)
    pt = np.asarray(pred_t, dtype=np.float64)
    gt_times = np.asarray(gt_t, dtype=np.float64)

    if len(p) == 0 or len(g) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0
    if len(pt) != len(p) or len(gt_times) != len(g):
        raise ValueError("timestamp arrays must match their trajectories in length")

    order = np.argsort(gt_times)
    gt_times, g = gt_times[order], g[order]

    # 1. trim the prediction to the recorded range
    keep = (pt >= gt_times[0] - _EPS) & (pt <= gt_times[-1] + _EPS)
    if not keep.any():
        return np.empty((0, 2)), np.empty((0, 2)), 0
    p, pt = p[keep], pt[keep]

    # 2. interpolate ground truth onto the prediction's timestamps
    aligned_gt = np.stack([np.interp(pt, gt_times, g[:, 0]), np.interp(pt, gt_times, g[:, 1])], axis=1)
    return p, aligned_gt, len(p)


def displacement_errors(
    predicted, ground_truth, interval_s: float = 0.5, eval_horizon_s: float = 3.0
) -> dict[str, float]:
    """ADE up to `eval_horizon_s`, plus FDE at each whole second within it.

    The KITScenes metric definition, unchanged from the harness that produced
    the published n=223 tables. Kept as one function rather than assembled from
    `ade` and `fde` at each call site, so every caller truncates to the horizon
    the same way -- the difference between scoring 3 s and scoring whatever the
    model happened to emit is exactly the kind of drift that makes two numbers
    look comparable when they are not.
    """
    predicted = _xy(predicted)
    ground_truth = _xy(ground_truth)
    available = min(len(predicted), len(ground_truth))
    horizon = min(available, int(round(eval_horizon_s / interval_s)))
    error = np.linalg.norm(predicted[:horizon, :2] - ground_truth[:horizon, :2], axis=-1)
    metrics: dict[str, float] = {"ade": float(error.mean())}
    for seconds in range(1, int(eval_horizon_s) + 1):
        index = int(round(seconds / interval_s)) - 1
        if 0 <= index < horizon:
            metrics[f"fde@{seconds}s"] = float(error[index])
    return metrics
