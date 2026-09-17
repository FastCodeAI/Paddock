"""Is the *planned* trajectory one a passenger would sit through?

Every open-loop benchmark in this repo scores a plan against a recorded human.
That answers "did it predict where the car went" and nothing else. A plan can
have a respectable ADE and still command 8 m/s^2 of lateral acceleration on the
way there -- the displacement at each half-second matches, the motion between
them does not. That failure is invisible in ADE by construction, because ADE
looks only at the sampled points and never at the curve through them.

So this module reads the kinematics *off the plan alone*. No ground truth is
involved, which is what makes it usable on any open-loop benchmark and on any
model: a plan is comfortable or it is not, regardless of what the human did.

## The bounds

The five bounds are nuPlan's `ego_is_comfortable` thresholds, which NAVSIM's PDM
score uses unchanged as its `history_comfort` term:

    longitudinal acceleration    <=  2.40 m/s^2
    longitudinal deceleration    >= -4.05 m/s^2
    |lateral acceleration|       <=  4.89 m/s^2
    |longitudinal jerk|          <=  4.13 m/s^3
    |yaw rate|                   <=  0.95 rad/s

They are restated here rather than imported: nuplan-devkit is not a paddock
dependency, and the one adapter that does reach NAVSIM's scorer
(`adapters/benchmarks/navsim/`) gets these numbers from the external scorer
rather than computing them. **Not verified against a checkout** -- no site in
this repo clones nuplan-devkit -- so they are cited at the strength of published
constants, not of code read. What they buy is a common scale: a `comfort_*`
number on comma2k19 and one on kitscenes are the same statement.

## What is measured, and on which grid

paddock's `Trajectory` convention puts waypoint `i` at `t = (i+1) * dt` with the
ego origin at `t = 0`. The origin is therefore a real, known point of the plan
and is prepended before differencing -- without it the first commanded
acceleration is unmeasurable, and a plan that jumps away from the car (the
classic "model has not warmed up" artefact) would score as perfectly smooth.

Differencing three times is a coarse instrument, and it is worse than coarse if
each model is differenced on its own waypoint spacing. Position quantisation of
size `e` becomes an acceleration of `e/dt^2` and a jerk of `e/dt^3`, so halving
the spacing multiplies the noise floor eightfold while changing nothing about
the path. Measured on comma2k19: Qwen-Drive plans at `dt = 0.1` and carries
about 4 cm of output quantisation on a plan that is otherwise smooth enough to
score the best ADE of any real model on that benchmark -- and scored a peak jerk
of 1932 m/s^3 against openpilot's 5.0 on the identical road at the identical
speed. Every one of its 10,165 plans came out "uncomfortable". None of that was
a property of the plan; all of it was the grid.

So the kinematics are read on a **common grid** -- `REFERENCE_DT`, coarsened to
it when a model plans more finely, left alone when it does not. That restores
the comparison this module exists to make. The cost is that content faster than
2 Hz is invisible, which is the right trade: these are bounds on what a plan
commands, and a plan is handed to a controller that smooths it anyway.

Read these as "did the plan command something no car should do", not as
ride-quality telemetry.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BOUNDS", "COMFORT_KEYS", "MIN_WAYPOINTS", "REFERENCE_DT", "comfort"]

#: The spacing every model is measured on, whatever it plans at. 0.5 s because
#: that is what every model in this repo emits and what the open-loop harnesses
#: score at, so for all but one of them this changes precisely nothing -- the
#: resampled points are the original points. It is deliberately *not* nuPlan's
#: own 0.1 s: nuPlan differences a simulated, dynamically feasible rollout, where
#: a 4 cm step between samples cannot occur, while this module differences raw
#: network output, where it routinely does.
REFERENCE_DT = 0.5

#: nuPlan's `ego_is_comfortable` bounds. Stored as positive magnitudes --
#: `lon_decel` is a limit on braking, compared against `-min(a_lon)`.
BOUNDS = {
    "lon_accel": 2.40,
    "lon_decel": 4.05,
    "lat_accel": 4.89,
    "lon_jerk": 4.13,
    "yaw_rate": 0.95,
}

#: Jerk is a third difference, so it needs three intervals, so it needs three
#: waypoints past the origin. Below this the whole set is NaN rather than a
#: subset being quietly emitted: a model whose plan is too short to have a jerk
#: should not appear to have a *comfortable* one.
MIN_WAYPOINTS = 3


#: Every key `comfort` returns, so a benchmark can declare them in its
#: `benchmark.yaml` and a harness can NaN-fill them without calling it.
COMFORT_KEYS: tuple[str, ...] = (
    *(f"comfort_{name}_max" for name in BOUNDS),
    *(f"comfort_{name}_within_bounds" for name in BOUNDS),
    "comfort_within_bounds",
)


def _on_reference_grid(xy, dt: float):
    """Resample a finer-than-reference plan onto `REFERENCE_DT` spacing.

    Only ever coarsens. Interpolating a coarse plan *up* to a finer grid would
    manufacture the smoothness being measured -- linear interpolation has zero
    jerk between its knots -- so a model planning at or above `REFERENCE_DT` is
    left on its own grid and its numbers are unchanged by this function existing.

    Returns None when the plan is too short to place `MIN_WAYPOINTS` on the
    common grid, which is a refusal to measure and not a comfortable verdict.
    """
    n = int(len(xy) * dt / REFERENCE_DT + 1e-9)
    if n < MIN_WAYPOINTS:
        return None
    # t = 0 is the ego origin, so it anchors the interpolation as a real sample
    # rather than being extrapolated to.
    src_t = np.arange(len(xy) + 1) * dt
    src = np.vstack([np.zeros((1, 2)), np.asarray(xy, dtype=np.float64)])
    dst_t = np.arange(1, n + 1) * REFERENCE_DT
    return np.column_stack([np.interp(dst_t, src_t, src[:, i]) for i in range(2)])


def comfort(xy, dt: float) -> dict[str, float]:
    """Kinematic extremes of one planned path, plus whether each is in bounds.

    `xy` is (N, 2) in paddock's ego frame, waypoint `i` at `t = (i+1) * dt`;
    `dt` is the plan's own spacing in seconds. It is what the plan is *given*
    on, not what it is measured on -- a plan finer than `REFERENCE_DT` is
    coarsened onto it first, so two models are always compared on one grid.
    Returns every key in `COMFORT_KEYS`, NaN throughout for a plan too short or
    not finite, so a caller aggregating across models never sees a key appear
    and disappear.

    The `*_within_bounds` flags are 0.0/1.0 rather than bools so that averaging
    them across samples gives the fraction of plans that stayed in bounds, which
    is the number a report wants.

    **That average does not have the same denominator for every model.** A
    sample whose plan was too short, not finite, or too coarse to resample
    contributes NaN, and an aggregate over NaN drops the sample rather than
    counting it as a failure -- deliberately, because "not measurable" is not
    "uncomfortable". On the recorded comma2k19 runs that is `n = 9148,
    n_missing = 1017` for constant_velocity against `n = 10165, n_missing = 0`
    for alpamayo15, openpilot, qwendrive and drivejepa_free, and `10161` for
    visionpilot. So the column reads "of the plans this model made that could
    be judged, this fraction were comfortable" -- fine to compare as a rate,
    not as a count, and never to subtract.

    `metrics.json` carries `n` and `n_missing` for every metric, so the
    denominator is on the record; it is the *rendered* leaderboard column that
    shows the mean alone. Surfacing the coverage next to the number is a
    reporting change that touches every benchmark's table, which is why it is a
    ROADMAP item rather than a quiet edit here.
    """
    nan = dict.fromkeys(COMFORT_KEYS, float("nan"))
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < MIN_WAYPOINTS or dt <= 0:
        return nan
    xy = xy[:, :2]
    if not np.isfinite(xy).all():
        return nan

    # Everything below differences `xy` on `dt`, so the grid has to be common
    # before any of it runs -- see the module docstring on why a model's own
    # spacing is not a usable one.
    if dt < REFERENCE_DT:
        xy = _on_reference_grid(xy, dt)
        if xy is None:
            return nan
        dt = REFERENCE_DT

    # t = 0 is the ego origin by construction of the frame; see the docstring.
    path = np.vstack([np.zeros((1, 2)), xy])
    step = np.diff(path, axis=0)  # per-interval displacement
    length = np.linalg.norm(step, axis=1)
    speed = length / dt  # at interval midpoints

    lon_accel = np.diff(speed) / dt
    lon_jerk = np.diff(lon_accel) / dt

    # Heading is undefined where the car did not move, and `arctan2(0, 0)` is 0
    # rather than NaN -- so a stopped plan would otherwise report a yaw rate
    # swinging between whatever headings its noise happened to produce.
    moving = length > 1e-6
    heading = np.where(moving, np.arctan2(step[:, 1], step[:, 0]), np.nan)
    # `np.unwrap` is a cumulative sum over differences, so a single NaN voids
    # every heading after it: a plan that waited at a light and then turned off
    # reported `comfort_yaw_rate_max` and `comfort_lat_accel_max` as NaN for the
    # whole rest of the plan, which `summarise()` reads as "not measurable" and
    # drops -- so the turn never reached an aggregate at all. Unwrap the moving
    # headings among themselves instead; a constant 2*pi offset picked up across
    # a gap cancels in the differences taken below.
    unwrapped = np.full(heading.shape, np.nan)
    if moving.any():
        unwrapped[moving] = np.unwrap(heading[moving])
    # Putting the NaNs back where the car was stopped is also what keeps a yaw
    # rate from being reported *across* a stop: nothing in a path says whether
    # the car rotated while stationary, and a pair straddling one has a NaN at
    # one end, so the difference stays NaN without needing a second mask.
    yaw_rate = np.diff(unwrapped) / dt
    # Lateral acceleration is v * yaw_rate, at the point where the yaw rate is
    # defined: between two interval midpoints, so the mean of their speeds.
    lat_accel = yaw_rate * 0.5 * (speed[:-1] + speed[1:])

    def peak(values, signed: bool = False) -> float:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(finite.max() if signed else np.abs(finite).max())

    out = {
        "comfort_lon_accel_max": peak(lon_accel, signed=True),
        # Braking as a positive magnitude, so it compares against a positive
        # bound like the other four and reads the same way in a table.
        "comfort_lon_decel_max": -float(np.min(lon_accel)) if np.isfinite(lon_accel).any() else float("nan"),
        "comfort_lat_accel_max": peak(lat_accel),
        "comfort_lon_jerk_max": peak(lon_jerk),
        "comfort_yaw_rate_max": peak(yaw_rate),
    }
    for name, limit in BOUNDS.items():
        value = out[f"comfort_{name}_max"]
        # A NaN extreme is "not measurable", not "in bounds": NaN propagates so
        # the aggregate mean excludes the sample instead of counting it as a pass.
        out[f"comfort_{name}_within_bounds"] = float("nan") if np.isnan(value) else float(value <= limit)
    flags = [out[f"comfort_{n}_within_bounds"] for n in BOUNDS]
    out["comfort_within_bounds"] = float("nan") if np.isnan(flags).any() else float(all(flags))
    return out
