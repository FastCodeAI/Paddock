"""Displacement metrics, on exactly the grid the Drive-JEPA run used.

Alpamayo emits 64 waypoints at 10 Hz (6.4 s); Drive-JEPA emitted 8 poses at 0.5 s
(4.0 s), scored over 0-3 s.  To compare them the Alpamayo output is *subsampled* onto
the same 0.5 s grid -- no interpolation, since 10 Hz divides 0.5 s exactly.

Two things are deliberately not "improved":

* ``displacement_errors`` is a byte-for-byte port of the reference implementation.
  It is the definition the baseline numbers were produced under, so any tidying here
  would silently change what "ADE 2.364" means.
* The headline ADE averages the **6 points of the 0.5 s grid**, not all 30 native
  10 Hz points inside 0-3 s.  The dense version includes the 0.1-0.4 s region where
  error is near zero, which would lower ADE by a systematic amount and flatter
  Alpamayo against the baseline.  It is reported separately, labelled non-comparable.

Alpamayo samples trajectories, whereas Drive-JEPA is deterministic, so a single draw
is not the whole story.  ``k_metrics`` reports the full spread; the honest
single-trajectory number against a deterministic planner is the **medoid** -- minADE is
oracle-selected and flatters the sampler, mean-over-K penalises multimodality that the
metric cannot reward.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

INTERVAL_S = 0.5
DT = 0.1
# 10 Hz index of each 0.5 s grid point: 0.5 s -> 4, 1.0 s -> 9, ... 4.0 s -> 39.
GRID_IDX = np.arange(4, 44, 5)
EVAL_HORIZON_S = 3.0


def displacement_errors(
    predicted: npt.NDArray, ground_truth: npt.NDArray, interval_s: float, eval_horizon_s: float
) -> dict[str, float]:
    """ADE up to *eval_horizon_s* plus FDE at each whole second within it.

    Ported unchanged from the original KITScenes inference script's
    ``displacement_errors`` -- this is the metric definition the comparison
    baseline was computed with.
    """
    available = min(len(predicted), len(ground_truth))
    horizon = min(available, int(round(eval_horizon_s / interval_s)))
    error = np.linalg.norm(predicted[:horizon, :2] - ground_truth[:horizon, :2], axis=-1)
    metrics: dict[str, float] = {"ade": float(error.mean())}
    for seconds in range(1, int(eval_horizon_s) + 1):
        index = int(round(seconds / interval_s)) - 1
        if 0 <= index < horizon:
            metrics[f"fde@{seconds}s"] = float(error[index])
    return metrics


def to_half_second_grid(pred_xyz: npt.NDArray, n: int = 8) -> npt.NDArray:
    """(K, 64, 3) 10 Hz predictions -> (K, n, 3) on the 0.5 s grid, by subsampling."""
    idx = GRID_IDX[:n]
    assert idx[-1] < pred_xyz.shape[-2], f"need {idx[-1] + 1} waypoints, have {pred_xyz.shape[-2]}"
    return pred_xyz[..., idx, :]


def yaw_frame_correction(t0_rot: npt.NDArray, t0_yaw: float) -> npt.NDArray:
    """Rotate from the full-SO(3) t0 frame into the yaw-only t0 frame the GT uses.

    Predictions live in the frame implied by the ego history we handed the model,
    ``R_t0^-1 (p - p_t0)``; the ground truth uses Drive-JEPA's planar convention,
    ``R_yaw(t0)^-1 (p - p_t0)``.  The two differ by the pod's roll and pitch, which is
    under 2 cm over 4 s here, but the correction is exact and free.
    """
    cos_y, sin_y = np.cos(t0_yaw), np.sin(t0_yaw)
    r_yaw = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    return r_yaw.T @ t0_rot


def apply_correction(pred_xyz: npt.NDArray, correction: npt.NDArray) -> npt.NDArray:
    """Apply a 3x3 rotation to every waypoint of a (..., T, 3) prediction."""
    return pred_xyz @ correction.T


def medoid_index(trajectories: npt.NDArray) -> int:
    """The sample closest to all the others -- the sampler's consensus trajectory."""
    if len(trajectories) == 1:
        return 0
    flat = trajectories[..., :2].reshape(len(trajectories), -1)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(distances.sum(axis=1).argmin())


def _min_over_subsets(ades: npt.NDArray, size: int, rng: np.random.Generator) -> float:
    """Expected minADE over a random *size*-subset of the available draws.

    With K=16 draws in hand, minADE@6 is estimated by averaging the minimum over
    6-subsets: exact by enumeration when that is cheap, sampled otherwise.  This is
    what lets one K=16 run report the K=6 number NVIDIA publishes.
    """
    k = len(ades)
    if size >= k:
        return float(ades.min())
    combinations = list(itertools.combinations(range(k), size))
    if len(combinations) > 4000:
        picks = rng.choice(k, size=(4000, size), replace=True)
        return float(ades[picks].min(axis=1).mean())
    return float(np.mean([ades[list(c)].min() for c in combinations]))


def k_metrics(
    pred_grid: npt.NDArray,
    gt_grid: npt.NDArray,
    interval_s: float = INTERVAL_S,
    eval_horizon_s: float = EVAL_HORIZON_S,
    subset_sizes: tuple[int, ...] = (6,),
    seed: int = 0,
) -> dict[str, Any]:
    """Every displacement statistic for a (K, n, 3) set of draws against one GT.

    ``ade``/``fde@Ns`` are the *first draw* -- the strict single-sample number that is
    apples-to-apples with a deterministic planner.  ``medoid_*`` is the fair consensus
    number, ``min_*`` the oracle-selected best-of-K.
    """
    per_draw = [displacement_errors(p, gt_grid, interval_s, eval_horizon_s) for p in pred_grid]
    ades = np.array([m["ade"] for m in per_draw])
    medoid = medoid_index(pred_grid)
    rng = np.random.default_rng(seed)

    out: dict[str, Any] = dict(per_draw[0])  # ade, fde@1s, fde@2s, fde@3s
    out["num_samples"] = int(len(pred_grid))
    out["mean_ade"] = float(ades.mean())
    out["min_ade"] = float(ades.min())
    out["std_ade"] = float(ades.std())
    out["medoid_index"] = medoid
    for key, value in per_draw[medoid].items():
        out[f"medoid_{key}"] = value

    horizon_idx = int(round(eval_horizon_s / interval_s)) - 1
    final = np.linalg.norm(pred_grid[:, horizon_idx, :2] - gt_grid[horizon_idx, :2], axis=-1)
    out[f"min_fde@{int(eval_horizon_s)}s"] = float(final.min())

    for size in subset_sizes:
        out[f"min_ade@{size}"] = _min_over_subsets(ades, size, rng)
    return out


def native_metrics(pred_xyz: npt.NDArray, gt_xyz: npt.NDArray) -> dict[str, float]:
    """Dense 10 Hz metrics: 0-3 s and the full 6.4 s horizon Alpamayo natively predicts.

    Reported alongside the headline numbers but never mixed with them -- the sampling
    grid differs, so these are not comparable to the 0.5 s-grid baseline.
    """
    out: dict[str, float] = {}
    available = min(pred_xyz.shape[-2], gt_xyz.shape[0])
    error = np.linalg.norm(pred_xyz[..., :available, :2] - gt_xyz[None, :available, :2], axis=-1)

    if available >= 30:
        out["ade_10hz_3s"] = float(error[:, :30].mean(axis=-1).min())
    if available >= 64:
        out["ade_10hz_6p4s"] = float(error[:, :64].mean(axis=-1).min())
        out["fde_10hz_6p4s"] = float(error[:, 63].min())
    out["num_future_valid"] = int(available)
    return out


def constant_velocity(speed: float, n: int = 8, interval_s: float = INTERVAL_S) -> npt.NDArray:
    """Straight-ahead constant-speed baseline, matching ``visualize_kitscenes``."""
    forward = speed * interval_s * np.arange(1, n + 1)
    return np.stack([forward, np.zeros(n), np.zeros(n)], axis=-1)
