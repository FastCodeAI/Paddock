"""Scoring everything the model predicts *besides* the path.

A supercombo forward pass produces a plan, its own speed and rotation rate, a
lead vehicle, a per-waypoint uncertainty, four lane lines, two road edges and
where the road surface sits relative to the device. Until now this benchmark
scored the first of those and discarded the rest -- which is a shame, because
comma2k19 is one of the few datasets that can *check* the rest: it ships the
car's own CAN speed, its radar returns, and an INS/GNSS/vision-fused pose.

So this module takes `Trajectory.extras` (`policy.protocol.EXTRAS_KEYS`) and the
measured references stashed in `Sample.native_id["reference"]`, and scores one
against the other. Nothing here is openpilot-specific: a model that fills no
extras gets NaN for every key and the benchmark's declared metric list does not
change, which is the property that lets one `benchmark.yaml` describe runs from
models with very different heads.

## What the references are, and their known limits

* **speed** -- `processed_log/CAN/speed`, the car's own odometer, falling back to
  the GNSS/INS pose speed on a segment with no CAN. The two disagree by about
  1 % and neither is privileged; which one was used is recorded per sample.
* **yaw rate** -- differenced from the recorded orientation (`loader.SegmentPose.yaw_rate`).
* **lead** -- the closest in-lane radar track (`loader.lead_track`), in the
  camera frame with no offset applied; `overlay.radar_to_device` documents the
  measurement behind that.
* **lane geometry** -- there is none. comma2k19 has no lane annotations, so the
  lane and road-height numbers here are *plausibility gates*, not errors: a US
  lane is about 3.7 m and the camera is about 1.2 m up, and a model reporting
  2 m or 8 m has a broken warp rather than a modelling limitation. Thresholds
  are the reference implementation's
  (`external_code/openpilot/.tmp/c2k19_eval/metrics_ad.py::sanity_gate`).

## One deliberate omission

The reference reports `yaw_rate_corr`, a per-segment correlation between the
model's yaw rate and the recorded one. It is a good diagnostic and it cannot be
computed here: this benchmark scores a handful of independent keyframes per
segment, not a time series, and a correlation over 5 points is noise. The
per-sample error and its signed bias are reported instead, and a correlation --
if it is wanted -- belongs in a report that pools samples by segment, not in a
per-sample score.
"""

from __future__ import annotations

import numpy as np

from paddock.metrics.comfort import COMFORT_KEYS, comfort

__all__ = ["ALL_KEYS", "AUX_KEYS", "LEAD_NEAR_M", "LEAD_PROB_THRESHOLD", "score"]

#: Beyond this the model and the radar routinely lock onto different vehicles on
#: a multi-lane road, so a range disagreement measures the sensor difference
#: rather than the model. The reference draws the line in the same place and for
#: the same reason.
LEAD_NEAR_M = 60.0

#: `lead_prob > 0.5` is "there is a lead", matching openpilot's own consumer.
LEAD_PROB_THRESHOLD = 0.5

#: Sanity thresholds, from the reference's `sanity_gate`. These are gates, not
#: errors: failing one means the warp or the calibration is wrong and every
#: trajectory number for that sample is meaningless.
GATE_SPEED_REL_ERR = 0.05
GATE_LANE_WIDTH_M = (2.5, 4.5)
GATE_CAMERA_HEIGHT_M = (1.0, 1.5)

#: Geometry is only meaningful while the car is moving -- a stationary car has no
#: road frame to speak of, and openpilot's own calibrator ignores those frames
#: for the same reason. 15 mph, as `calibration.MIN_SPEED_FILTER`.
MIN_GATE_SPEED = 15 * 0.44704

#: A gap closing inside this is the number a safety reviewer asks for first.
TTC_ALERT_S = 2.0

AUX_KEYS: tuple[str, ...] = (
    # pose head vs CAN / INS
    "speed_abs_err",
    "speed_rel_err",
    "yaw_rate_abs_err",
    "yaw_rate_bias",
    # lead head vs radar
    "lead_prob",
    "lead_precision",
    "lead_recall",
    "lead_range_err",
    "lead_range_abs_err",
    "lead_lateral_abs_err",
    "lead_rel_speed_abs_err",
    # plan uncertainty vs realised error
    "within_1sigma_x",
    "within_1sigma_y",
    "within_2sigma_x",
    "within_2sigma_y",
    # geometry, as plausibility rather than error
    "lane_width_m",
    "lane_offset_m",
    "lane_prob",
    "road_width_m",
    "camera_height_m",
    # gates
    "gate_speed",
    "gate_lane_width",
    "gate_camera_height",
    "sanity_pass",
    # safety, against the radar lead
    "lead_gap_min_m",
    "lead_ttc_min_s",
    "ttc_below_2s",
)

#: What `score` returns, and what `benchmark.yaml` must declare. The comfort
#: half is computed by the shared `paddock.metrics.comfort` rather than here, so
#: a comfort number on this benchmark and one on kitscenes mean the same thing.
ALL_KEYS: tuple[str, ...] = AUX_KEYS + COMFORT_KEYS

_NAN = float("nan")


def _finite(value) -> float:
    """A float, or NaN for None / non-numeric / non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return _NAN
    return out if np.isfinite(out) else _NAN


def _pose(pose, reference) -> dict[str, float]:
    if not isinstance(pose, dict):
        return {}
    out: dict[str, float] = {}
    v_model = _finite((pose.get("velocity") or [None])[0])
    v_ref = _finite(reference.get("speed"))
    if np.isfinite(v_model) and np.isfinite(v_ref):
        out["speed_abs_err"] = abs(v_model - v_ref)
        # Relative to at least 1 m/s: at walking pace the denominator dominates
        # and every crawling frame would look like a catastrophic speed error.
        out["speed_rel_err"] = abs(v_model - v_ref) / max(v_ref, 1.0)

    yaw_model = _finite((pose.get("rotation_rate") or [None, None, None])[2])
    yaw_ref = _finite(reference.get("yaw_rate"))
    if np.isfinite(yaw_model) and np.isfinite(yaw_ref):
        out["yaw_rate_abs_err"] = abs(yaw_model - yaw_ref)
        out["yaw_rate_bias"] = yaw_model - yaw_ref
    return out


def _lead(leads, reference) -> dict[str, float]:
    """Detection agreement and, where both agree, range/lateral/closing error.

    Precision and recall are emitted as 0/1 with NaN where the question does not
    arise -- NaN when the radar saw nothing (recall is undefined) and NaN when
    the model reported nothing (precision is undefined). Averaged over a run
    those two columns *are* precision and recall, without the harness needing a
    confusion matrix it has nowhere to put.
    """
    if leads is None:
        return {}
    model = next((entry for entry in leads if _finite(entry.get("prob")) >= LEAD_PROB_THRESHOLD), None)
    out: dict[str, float] = {}
    if leads:
        out["lead_prob"] = _finite(leads[0].get("prob"))

    radar_d = _finite(reference.get("lead_d"))
    # Only near-range radar leads are a fair reference; see LEAD_NEAR_M.
    radar_near = np.isfinite(radar_d) and radar_d < LEAD_NEAR_M
    if radar_near:
        out["lead_recall"] = float(model is not None)
    if model is not None:
        out["lead_precision"] = float(radar_near)

    if model is None or not radar_near:
        return out

    model_d = _finite(model.get("d"))
    if np.isfinite(model_d):
        out["lead_range_err"] = model_d - radar_d
        out["lead_range_abs_err"] = abs(model_d - radar_d)
    model_y, radar_y = _finite(model.get("y")), _finite(reference.get("lead_y"))
    if np.isfinite(model_y) and np.isfinite(radar_y):
        out["lead_lateral_abs_err"] = abs(model_y - radar_y)
    model_v, radar_v = _finite(model.get("v_rel")), _finite(reference.get("lead_v_rel"))
    if np.isfinite(model_v) and np.isfinite(radar_v):
        out["lead_rel_speed_abs_err"] = abs(model_v - radar_v)
    return out


def _sigma(sigma, pred_xy, gt_xy) -> dict[str, float]:
    """Is the model's own uncertainty honest?

    A calibrated 1-sigma puts ~68 % of the truth inside it and 2-sigma ~95 %.
    `pred_xy` and `gt_xy` arrive already aligned onto the prediction's grid, and
    `align` trims only the prediction's *tail* (its times are increasing and the
    ground truth starts at t = 0), so the surviving waypoints are a prefix and
    `sigma[:n]` is the matching uncertainty.
    """
    if sigma is None:
        return {}
    s = np.asarray(sigma, dtype=np.float64)
    n = min(len(pred_xy), len(gt_xy), len(s))
    if n == 0:
        return {}
    err = np.abs(np.asarray(pred_xy)[:n] - np.asarray(gt_xy)[:n])
    z = err / np.maximum(s[:n], 1e-6)
    ok = np.isfinite(z).all(axis=1)
    if not ok.any():
        return {}
    z = z[ok]
    return {
        "within_1sigma_x": float((z[:, 0] < 1.0).mean()),
        "within_1sigma_y": float((z[:, 1] < 1.0).mean()),
        "within_2sigma_x": float((z[:, 0] < 2.0).mean()),
        "within_2sigma_y": float((z[:, 1] < 2.0).mean()),
    }


def _abeam(polyline) -> float:
    """Lateral offset of a predicted polyline at x = 0, i.e. abeam the car.

    openpilot's own `fill_lane_line_meta` reads `laneLines[i].y[0]`, the first
    `X_IDXS` entry, which is 0 m. Doing the same here keeps the lane width
    comparable with the number openpilot's UI shows.
    """
    points = np.asarray(polyline, dtype=np.float64)
    return float(points[0, 1]) if points.ndim == 2 and len(points) else _NAN


def _geometry(lane, road) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(lane, dict):
        lines = lane.get("lines") or []
        if len(lines) >= 4:
            # Lines run left to right, so indices 1 and 2 bound the ego lane.
            # y is positive *left*, so the left line's y is the larger one.
            left, right = _abeam(lines[1]), _abeam(lines[2])
            out["lane_width_m"] = left - right
            # Positive means the car sits right of the lane centre: the centre
            # is at positive y, i.e. to the left of the car.
            out["lane_offset_m"] = (left + right) / 2.0
        probs = lane.get("line_probs") or []
        if len(probs) >= 4:
            out["lane_prob"] = float(np.mean([_finite(probs[1]), _finite(probs[2])]))
        edges = lane.get("edges") or []
        if len(edges) >= 2:
            out["road_width_m"] = _abeam(edges[0]) - _abeam(edges[1])
    if isinstance(road, dict):
        out["camera_height_m"] = _finite(road.get("camera_height"))
    return out


def _gates(scored, reference) -> dict[str, float]:
    """The hard correctness gates. NaN means "not applicable", not "passed"."""
    out: dict[str, float] = {}
    rel = scored.get("speed_rel_err", _NAN)
    if np.isfinite(rel):
        out["gate_speed"] = float(rel < GATE_SPEED_REL_ERR)

    # Below 15 mph the road frame is not observable; reporting these as passed
    # would let a segment of stop-and-go traffic vouch for a broken warp.
    if _finite(reference.get("speed")) >= MIN_GATE_SPEED:
        for key, gate, (lo, hi) in (
            ("lane_width_m", "gate_lane_width", GATE_LANE_WIDTH_M),
            ("camera_height_m", "gate_camera_height", GATE_CAMERA_HEIGHT_M),
        ):
            value = scored.get(key, _NAN)
            if np.isfinite(value):
                out[gate] = float(lo <= value <= hi)

    applicable = [v for v in out.values() if np.isfinite(v)]
    if applicable:
        out["sanity_pass"] = float(all(applicable))
    return out


def _safety(pred_xy, dt: float, reference) -> dict[str, float]:
    """Would the plan close the gap on the radar lead?

    The lead is propagated at constant speed from its measured range and closing
    rate, and the ego along the planned path's arclength. That is a proxy and it
    is stated as one: the human was driving, so this asks "would the model have
    driven into the car in front", not "did anyone nearly crash".

    NaN when there is no near-range radar lead -- an empty road has no gap to be
    right about, and scoring it as a large safe gap would reward driving on
    empty roads.
    """
    d0 = _finite(reference.get("lead_d"))
    if not np.isfinite(d0) or d0 >= LEAD_NEAR_M:
        return {}
    v_rel = _finite(reference.get("lead_v_rel"))
    v_ego = _finite(reference.get("speed"))
    if not np.isfinite(v_rel) or not np.isfinite(v_ego):
        return {}

    xy = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    if len(xy) == 0 or dt <= 0:
        return {}
    path = np.vstack([np.zeros((1, 2)), xy])
    travelled = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    t = dt * np.arange(1, len(travelled) + 1)
    # The lead's own speed is v_ego + v_rel; a negative gap is a collision.
    gap = d0 + (v_ego + v_rel) * t - travelled

    out = {"lead_gap_min_m": float(gap.min())}
    hit = np.flatnonzero(gap <= 0.0)
    # No collision within the plan is NaN, not "infinite TTC": averaging the
    # latter with a real 1.2 s would be meaningless, and `ttc_below_2s` already
    # carries the "did it happen" half as a rate.
    out["lead_ttc_min_s"] = float(t[hit[0]]) if len(hit) else _NAN
    out["ttc_below_2s"] = float(len(hit) > 0 and t[hit[0]] < TTC_ALERT_S)
    return out


def score(sample, prediction, pred_xy, gt_xy) -> dict[str, float]:
    """Every auxiliary metric for one sample. Always the same keys.

    `pred_xy` / `gt_xy` are the time-aligned pair `harness.score` already built;
    they are passed in rather than rebuilt so the uncertainty check is scored on
    exactly the pairing the trajectory metrics were.
    """
    out: dict[str, float] = dict.fromkeys(ALL_KEYS, _NAN)
    reference = ((sample.native_id or {}).get("reference")) or {}
    extras = getattr(prediction, "extras", None) or {}

    out.update(_pose(extras.get("pose"), reference))
    out.update(_lead(extras.get("lead"), reference))
    out.update(_sigma(extras.get("sigma"), pred_xy, gt_xy))
    out.update(_geometry(extras.get("lane"), extras.get("road")))
    out.update(_gates(out, reference))
    out.update(_safety(prediction.xy, prediction.dt, reference))
    out.update(comfort(prediction.xy, prediction.dt))
    return out
