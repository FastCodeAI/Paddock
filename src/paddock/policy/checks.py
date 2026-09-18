"""The invariants an observed step can assert about itself.

Each entry names a defect that has actually happened, in this repo or in the
work it was rescued from, and says whose defect it is -- `adapter`, `harness` or
`model` (see `trace`'s module docstring). That attribution is the output the
whole exercise exists to produce: a number without it is a leaderboard row, and
with it, it is a thing to go and fix.

These are the checks that need nothing but the protocol: a plan, an observation,
a frame. They are registered against the protocol's own steps, so **every model
gets them without annotating anything**. Model-specific checks (openpilot's warp
margin, qwendrive's history cadence) are registered beside the adapters that
know what they mean.

Kept apart from `trace` so that file stays mechanism only, and imported from its
foot so registration cannot depend on anyone remembering to import this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from paddock.policy.trace import Step, check, mounting_off_level_deg

#: A camera further from level than this is a coordinate-convention error, not a
#: mounting. Same limit as `conformance.MAX_MOUNT_DEG` and openpilot's
#: `MAX_MOUNT_DEG`, and for the same reason: a real bracket is a few degrees off
#: level and a frame slip is 90.
MAX_MOUNT_DEG = 15.0

#: Below this the car is not really turning and a plan's bend is noise, so the
#: turn-direction check abstains rather than guessing. Applied to both sides of
#: that comparison. rad/s.
MIN_YAW_RATE_RPS = 0.02

#: How far into a plan the measured yaw rate is allowed to speak for. An
#: instantaneous rate constrains the next second or two and nothing beyond it;
#: past that the plan is following the road, not the current steering angle.
TURN_HORIZON_S = 1.5

#: Below this the car is not really moving, so "does the plan advance" and "is
#: the spacing right" have no answer. m/s.
MIN_SPEED_MPS = 1.0

#: How far the first waypoint may sit from `speed * dt` before the plan's grid
#: and its declared `dt` are saying different things. Generous on purpose: a
#: model braking hard or accelerating out of a stop legitimately lands well off
#: a constant-velocity estimate. A factor-of-five error is a grid mismatch.
SPACING_LO, SPACING_HI = 0.35, 2.5


def _plan(step: Step) -> Any:
    """The `Trajectory`-shaped thing this step produced, or None."""
    out = step.output
    return out if hasattr(out, "waypoints") and hasattr(out, "dt") else None


def _observations(step: Step) -> list[Any]:
    """Every `Observation`-shaped value this step was handed, in order."""
    found: list[Any] = []
    for value in step.inputs.values():
        items = value if isinstance(value, (list, tuple)) else [value]
        found += [v for v in items if hasattr(v, "canbus") and hasattr(v, "camera2ego")]
    return found


def _speed(step: Step) -> float | None:
    obs = _observations(step)
    return float(obs[-1].speed) if obs else None


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@check(
    "trajectory.not_a_commanded_stop",
    "An all-zero trajectory is how a vehicle model is told to brake to a stop, not how a "
    "model says 'no plan'. A model that has not warmed up must return constant_velocity(); "
    "zeros score as a deliberate brake, which looks like a good result for the wrong reason.",
    attribution="adapter",
)
def _not_a_commanded_stop(step: Step):
    plan = _plan(step)
    if plan is None:
        return None
    xy = np.asarray(plan.waypoints, dtype=np.float64)[:, :2]
    biggest = float(np.abs(xy).max()) if xy.size else 0.0
    return not np.allclose(xy, 0.0), f"largest waypoint magnitude is {biggest:.2f} m"


@check(
    "trajectory.moves_forward",
    "At a nonzero speed the plan must advance in +x. A plan that does not is either a frame "
    "error or a model shown a static world -- comma2k19's first result was the latter and "
    "reported ADE 38.8 m as a modelling finding.",
    attribution="adapter",
)
def _moves_forward(step: Step):
    plan, speed = _plan(step), _speed(step)
    if plan is None or speed is None or speed < MIN_SPEED_MPS:
        return None
    xy = np.asarray(plan.waypoints, dtype=np.float64)[:, :2]
    if len(xy) < 2:
        return None
    ok = bool(np.all(np.diff(xy[:, 0]) >= -0.05)) and xy[-1, 0] > 0.5
    return ok, f"plan reaches x = {xy[-1, 0]:.2f} m at {speed:.1f} m/s"


@check(
    "trajectory.spacing_matches_speed",
    "The first waypoint sits at t = dt, so it is roughly `speed * dt` metres out. A plan "
    "whose spacing disagrees with its declared dt by a large factor is a model emitting its "
    "native grid under someone else's label -- the same class of error as fde@5s that was "
    "FDE at 4 s, and invisible in ADE because both sides of the comparison move together.",
    attribution="adapter",
)
def _spacing_matches_speed(step: Step):
    plan, speed = _plan(step), _speed(step)
    if plan is None or speed is None or speed < MIN_SPEED_MPS:
        return None
    xy = np.asarray(plan.waypoints, dtype=np.float64)[:, :2]
    if not len(xy):
        return None
    expected = speed * float(plan.dt)
    reached = float(np.linalg.norm(xy[0]))
    ratio = reached / expected if expected > 0 else float("nan")
    detail = (
        f"first waypoint {reached:.2f} m out; {speed:.1f} m/s x dt {plan.dt:g} s "
        f"expects ~{expected:.2f} m (ratio {ratio:.2f})"
    )
    return SPACING_LO <= ratio <= SPACING_HI, detail


@check(
    "trajectory.turn_direction_matches_yaw_rate",
    "The car's own yaw rate says which way it is turning, and a plan that immediately bends "
    "the other way is a y-sign error: paddock is x-forward y-LEFT, openpilot's device frame "
    "is y-right, and the flip between them is one negation in one place. No metric shows "
    "this -- ADE is a magnitude -- and it is the failure a side-by-side viewer was built to "
    "look for by eye. Compared as rate against rate over the plan's first seconds: the far "
    "end of a 10 s plan is wherever the road goes, and on an S-bend it legitimately sits on "
    "the opposite side from the turn the car is finishing.",
    attribution="adapter",
)
def _turn_matches_yaw_rate(step: Step):
    plan = _plan(step)
    obs = _observations(step)
    if plan is None or not obs:
        return None
    yaw_rate, speed = float(obs[-1].yaw_rate), float(obs[-1].speed)
    if abs(yaw_rate) < MIN_YAW_RATE_RPS or speed < MIN_SPEED_MPS:
        return None
    planned = _early_yaw_rate(plan)
    # A plan that is straightening out has no side to be on, and the sign of its
    # residual bend is noise. Abstain rather than accuse -- the same gate the
    # measured rate gets, for the same reason.
    if planned is None or abs(planned) < MIN_YAW_RATE_RPS:
        return None
    turning = "left" if yaw_rate > 0 else "right"
    planning = "left" if planned > 0 else "right"
    return (
        (yaw_rate > 0) == (planned > 0),
        f"measured yaw rate {yaw_rate:+.3f} rad/s ({turning}); plan turns {planned:+.3f} rad/s "
        f"({planning}) over its first {TURN_HORIZON_S:g} s",
    )


def _early_yaw_rate(plan: Any) -> float | None:
    """The plan's own yaw rate over its first `TURN_HORIZON_S`, rad/s.

    Compared against the measured rate this is like for like, which the plan's
    *endpoint* is not: a 10 s plan ends wherever the road goes, and on an S-bend
    the recorded expert legitimately finishes a left turn with its future ending
    30 m to the right. Calibrated on viewer10's 850 keyframes: comparing
    endpoints fails the human driver on 16 of the 240 it fires on, comparing
    rates fails the human on **none** of 204 while still catching a negated y on
    all 204. A check that accuses a correct adapter is worse than no check.
    """
    xy = np.asarray(plan.waypoints, dtype=np.float64)[:, :2]
    steps = max(1, int(round(TURN_HORIZON_S / float(plan.dt))))
    if len(xy) <= steps:
        return None
    # Heading per leg, the first leg running from the ego origin: waypoint 0 is
    # at t = dt, so the segment before it is the one the car is on now.
    heading = np.arctan2(np.diff(xy[:, 1], prepend=0.0), np.diff(xy[:, 0], prepend=0.0))
    return float(heading[steps] - heading[0]) / (steps * float(plan.dt))


# --------------------------------------------------------------------------
# The input
# --------------------------------------------------------------------------


@check(
    "observation.mounting_is_level",
    "camera2ego is the single easiest thing to get wrong and the hardest to notice. The "
    "identity matrix reads as 'no mounting rotation' and is a 90 degree error in paddock's "
    "frames; the model is then shown a sideways world and predicts a stationary car.",
    attribution="harness",
)
def _mounting_is_level(step: Step):
    obs = _observations(step)
    if not obs:
        return None
    # Taken as a max rather than accumulated past a running best, so a rig where
    # every camera is exactly level still *passes* instead of abstaining -- a
    # check that goes quiet on the correct case is the failure mode this whole
    # file exists to avoid.
    angles = {cam: mounting_off_level_deg(m) for cam, m in obs[-1].camera2ego.items()}
    if not angles:
        return None
    where = max(angles, key=lambda cam: angles[cam])
    worst = angles[where]
    detail = f"{where} is {worst:.2f} deg off level (limit {MAX_MOUNT_DEG:g})"
    if worst > MAX_MOUNT_DEG and np.allclose(np.asarray(obs[-1].camera2ego[where])[:3, :3], np.eye(3)):
        detail += "; the matrix is the identity, so this is the convention error, not a mounting"
    return worst <= MAX_MOUNT_DEG, detail


@check(
    "observation.history_is_ordered",
    "Models that infer ego motion from frame-to-frame change read a repeated or out-of-order "
    "timestamp as a parked car. `min_input_hz` refuses the pairing up front; this catches a "
    "window that is mis-ordered within an accepted pairing.",
    attribution="harness",
)
def _history_is_ordered(step: Step):
    obs = _observations(step)
    if len(obs) < 2:
        return None
    times = np.array([o.timestamp_us for o in obs], dtype=np.float64) / 1e6
    gaps = np.diff(times)
    span = float(times[-1] - times[0])
    hz = (len(obs) - 1) / span if span > 0 else float("nan")
    detail = f"{len(obs)} frames over {span:.2f} s at {hz:.1f} Hz, gaps {gaps.min():.3f}-{gaps.max():.3f} s"
    return bool(np.all(gaps > 0)), detail


@check(
    "image.is_not_blank",
    "A frame that is uniform is a decode that returned nothing, or a warp that sampled "
    "entirely outside the source. The model still plans, the run still completes, and the "
    "score reads as a modelling result.",
    attribution="adapter",
    severity="warning",
)
def _image_is_not_blank(step: Step):
    for value in (*step.inputs.values(), step.output):
        arr = np.asarray(value) if isinstance(value, np.ndarray) else None
        if arr is None or arr.ndim != 3 or arr.dtype != np.uint8:
            continue
        spread = float(np.std(arr[..., :3]))
        return spread > 1.0, f"pixel std {spread:.2f} over {arr.shape[1]}x{arr.shape[0]}"
    return None


# --------------------------------------------------------------------------
# The grid the model was trained on
# --------------------------------------------------------------------------


@check(
    "resample.is_interpolation_not_extrapolation",
    "Reading a model past its native horizon invents a tail it never learned to produce, and "
    "the benchmark then reports that invention as the model's 3 s accuracy. This is the check "
    "that separates 'the model is bad at this horizon' from 'the model was never trained to "
    "emit this horizon' -- and the second is the harness asking the wrong question.",
    attribution="harness",
)
def _is_interpolation(step: Step):
    native, requested = step.notes.get("native_t"), step.notes.get("requested_t")
    if native is None or requested is None:
        return None
    native = np.asarray(_values(native), dtype=np.float64)
    requested = np.asarray(_values(requested), dtype=np.float64)
    if not native.size or not requested.size:
        return None
    beyond = requested[requested > native.max() + 1e-9]
    detail = (
        f"model plans to {native.max():.2f} s, asked for up to {requested.max():.2f} s; "
        f"{len(beyond)} of {len(requested)} points extrapolate"
    )
    return len(beyond) == 0, detail


def _values(summary: Any) -> Any:
    """`note()` summarises before storing, so an array arrives as its record."""
    if isinstance(summary, dict):
        return summary.get("values", [])
    return summary
