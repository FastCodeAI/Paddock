"""NAVSIM's vehicle simulator and v1 comfort check, ported to numpy.

NAVSIM does not score a plan as given. It first drives it: an LQR tracker
steers and accelerates a kinematic bicycle model after the plan, every 0.1 s,
from the car's measured state, and every PDM term is scored on the path that
comes out (`navsim/evaluate/pdm_score.py`, `PDMSimulator.simulate_proposals`).
The driven path is smoother than the plan, which matters most for comfort: on
235 stored Drive-JEPA neuroNCAP decisions the raw plan failed comfort on 51
that NAVSIM's simulated path passed (review of #102).

Ported line by line from NAVSIM v1 (Drive-JEPA's `navsim_v1` checkout;
v2's simulator is the same code): `simulation/batch_lqr.py`,
`batch_lqr_utils.py`, `batch_kinematic_bicycle.py`, `pdm_simulator.py`, and
`scoring/pdm_comfort_metrics.py`. scipy's `savgol_filter` is restated, because
paddock's environment has no scipy. `tests/test_pdm_sim.py` holds every part
to NAVSIM's own outputs, recorded by `tools/benchmarks/navsim/pdm_sim_fixture.py`.

States are arrays `(..., 11)` in NAVSIM's `StateIndex` order, at the rear axle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# NAVSIM's StateIndex.
X, Y, HEADING, VX, VY, AX, AY, STEER, STEER_RATE, YAW_RATE, YAW_ACCEL = range(11)
N_STATE = 11


@dataclass(frozen=True)
class Vehicle:
    wheel_base: float


#: nuPlan's `get_pacifica_parameters()`, the car NAVSIM simulates.
PACIFICA = Vehicle(wheel_base=3.089)

#: `BatchKinematicBicycleModel` defaults.
MAX_STEERING_ANGLE = math.pi / 3
ACCEL_TIME_CONSTANT = 0.2
STEERING_TIME_CONSTANT = 0.05

#: `BatchLQRTracker` defaults.
Q_LONGITUDINAL = 10.0
R_LONGITUDINAL = 1.0
Q_LATERAL = np.diag([1.0, 10.0, 0.0])
R_LATERAL = 1.0
TRACKING_HORIZON = 10
JERK_PENALTY = 1e-4
CURVATURE_RATE_PENALTY = 1e-2
STOPPING_GAIN = 0.5
STOPPING_VELOCITY = 0.2
INITIAL_CURVATURE_PENALTY = 1e-10


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def principal_value(angle, min_: float = -np.pi):
    return (angle - min_) % (2 * np.pi) + min_


# ---------------------------------------------------------------------------
# Kinematic bicycle model (batch_kinematic_bicycle.py)
# ---------------------------------------------------------------------------


def _state_dot(states: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    dots = np.zeros(states.shape, dtype=np.float64)
    v = states[:, VX]
    dots[:, X] = v * np.cos(states[:, HEADING])
    dots[:, Y] = v * np.sin(states[:, HEADING])
    dots[:, HEADING] = v * np.tan(states[:, STEER]) / vehicle.wheel_base
    dots[:, VX : VY + 1] = states[:, AX : AY + 1]
    dots[:, AX : AY + 1] = 0.0
    dots[:, STEER] = states[:, STEER_RATE]
    return dots


def propagate(
    states: np.ndarray, accel: np.ndarray, steer_rate: np.ndarray, dt: float, vehicle: Vehicle
) -> np.ndarray:
    """One 0.1 s step: the commands through a first-order lag, then Euler integration."""
    prop = states.copy()
    old_accel, old_steer = states[:, AX], states[:, STEER]
    ideal_steer = dt * steer_rate + old_steer
    new_accel = dt / (dt + ACCEL_TIME_CONSTANT) * (accel - old_accel) + old_accel
    new_steer = dt / (dt + STEERING_TIME_CONSTANT) * (ideal_steer - old_steer) + old_steer
    prop[:, AX] = new_accel
    prop[:, AY] = 0.0
    prop[:, STEER_RATE] = (new_steer - old_steer) / dt

    out = states.copy()
    dots = _state_dot(prop, vehicle)
    out[:, X] = states[:, X] + dots[:, X] * dt
    out[:, Y] = states[:, Y] + dots[:, Y] * dt
    out[:, HEADING] = principal_value(states[:, HEADING] + dots[:, HEADING] * dt)
    out[:, VX] = states[:, VX] + dots[:, VX] * dt
    out[:, VY] = 0.0
    out[:, STEER] = np.clip(prop[:, STEER] + dots[:, STEER] * dt, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    out[:, YAW_RATE] = out[:, VX] * np.tan(out[:, STEER]) / vehicle.wheel_base
    out[:, AX : AY + 1] = dots[:, VX : VY + 1]
    out[:, YAW_ACCEL] = (out[:, YAW_RATE] - states[:, YAW_RATE]) / dt
    out[:, STEER_RATE] = dots[:, STEER]
    return out


# ---------------------------------------------------------------------------
# Profiles the tracker follows (batch_lqr_utils.py)
# ---------------------------------------------------------------------------


def _bmm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bij, bjk -> bik", a, b)


def _profile(initial: np.ndarray, derivatives: np.ndarray, dt: float) -> np.ndarray:
    cumsum = np.cumsum(derivatives * dt, axis=-1)
    return initial[..., None] + np.pad(cumsum, [(0, 0), (1, 0)], mode="constant")


def _fit_velocity(
    xy_displacements: np.ndarray, headings: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    batch, m = xy_displacements.shape[:2]
    y = xy_displacements.reshape(batch, -1)
    column = np.zeros(y.shape, dtype=np.float64)
    column[:, 0::2] = np.cos(headings)
    column[:, 1::2] = np.sin(headings)
    a = np.repeat(column[..., None] * dt**2, m, axis=2)
    a[..., 0] = column * dt
    upper = np.repeat(np.triu(np.ones((m, m), dtype=bool), k=1), 2, axis=0)
    a[:, upper] = 0.0
    # nuPlan's `_make_banded_difference_matrix`, assignment for assignment: the
    # second write covers the first, so only the last row keeps its +1. Not a
    # difference matrix, but what every NAVSIM score was computed with.
    banded = np.zeros((m - 2, m - 1), dtype=np.float64)
    banded[:, 1:] = np.eye(m - 2)
    banded[:, :-1] = -np.eye(m - 2)
    r = np.repeat(np.block([np.zeros((m - 2, 1)), banded])[None], batch, axis=0)
    a_t, r_t = a.transpose(0, 2, 1), r.transpose(0, 2, 1)
    solution = _bmm(np.linalg.pinv(_bmm(a_t, a) + JERK_PENALTY * _bmm(r_t, r)), a_t)
    x = np.einsum("bij, bj -> bi", solution, y)
    return x[:, 0], x[:, 1:]


def _fit_curvature(
    heading_displacements: np.ndarray, velocity: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    batch, dim = heading_displacements.shape
    a = np.repeat(np.tri(dim, dtype=np.float64)[None], batch, axis=0)
    a[:, :, 0] = velocity * dt
    scaled = velocity * dt**2
    a[:, 1:, 1:] *= scaled[:, None, 1:].transpose(0, 2, 1)
    q = CURVATURE_RATE_PENALTY * np.eye(dim)
    q[0, 0] = INITIAL_CURVATURE_PENALTY
    a_t = a.transpose(0, 2, 1)
    x = np.einsum("bij,bj->bi", _bmm(np.linalg.pinv(_bmm(a_t, a) + q), a_t), heading_displacements)
    return x[:, 0], x[:, 1:]


def velocity_curvature_profiles(poses: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """(B, K, 3) poses -> velocity and curvature profiles, each (B, K - 1)."""
    differences = np.diff(poses, axis=1)
    v0, accel = _fit_velocity(differences[..., :2], poses[:, :-1, 2], dt)
    velocity = _profile(v0, accel, dt)
    k0, curvature_rate = _fit_curvature(normalize_angle(differences[..., 2]), velocity, dt)
    return velocity, _profile(k0, curvature_rate, dt)


# ---------------------------------------------------------------------------
# LQR tracker (batch_lqr.py)
# ---------------------------------------------------------------------------


class Tracker:
    """`BatchLQRTracker` with its defaults, following one batch of proposals."""

    def __init__(self, proposals: np.ndarray, dt: float, vehicle: Vehicle) -> None:
        self.proposals, self.dt, self.wheel_base = proposals, dt, vehicle.wheel_base
        self.velocity, self.curvature = velocity_curvature_profiles(proposals[..., X : HEADING + 1], dt)

    def track(self, index: int, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = self.proposals[:, index]
        dx, dy = states[:, X] - reference[:, X], states[:, Y] - reference[:, Y]
        h = reference[:, HEADING]
        lateral = np.stack(
            [-dx * np.sin(h) + dy * np.cos(h), normalize_angle(states[:, HEADING] - h), states[:, STEER]],
            axis=-1,
        )
        v0 = states[:, VX]
        batch, n = self.velocity.shape
        ref_idx = min(index + TRACKING_HORIZON, n - 1)
        v_ref = self.velocity[:, ref_idx]
        curvature = np.zeros((batch, TRACKING_HORIZON), dtype=np.float64)
        length = ref_idx - index
        curvature[:, :length] = self.curvature[:, index:ref_idx]
        if length < TRACKING_HORIZON:
            curvature[:, length:] = self.curvature[:, ref_idx, None]

        accel = np.zeros(batch, dtype=np.float64)
        steer_rate = np.zeros(batch, dtype=np.float64)
        stop = (v_ref <= STOPPING_VELOCITY) & (v0 <= STOPPING_VELOCITY)
        accel[stop] = -STOPPING_GAIN * (v0[stop] - v_ref[stop])
        go = ~stop
        b = TRACKING_HORIZON * self.dt
        accel[go] = -1 / (b * Q_LONGITUDINAL * b + R_LONGITUDINAL) * b * Q_LONGITUDINAL * (v0[go] - v_ref[go])
        velocity = _profile(v0[go], np.repeat(accel[go, None], TRACKING_HORIZON, axis=-1), self.dt)[
            :, :TRACKING_HORIZON
        ]
        steer_rate[go] = self._lateral(lateral[go], velocity, curvature[go])
        return accel, steer_rate

    def _lateral(self, initial: np.ndarray, velocity: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        batch = velocity.shape[0]
        eye = np.eye(3, dtype=np.float64)
        b_in = np.zeros((3, 1), dtype=np.float64)
        b_in[2] = self.dt
        step = np.tile(eye[None, None], [TRACKING_HORIZON, batch, 1, 1])
        step[:, :, 0, 1] = velocity.T * self.dt
        step[:, :, 1, 2] = velocity.T * self.dt / self.wheel_base
        affine = np.zeros((TRACKING_HORIZON, batch, 3), dtype=np.float64)
        affine[:, :, 1] = -velocity.T * curvature.T * self.dt
        a = np.tile(eye[None], [batch, 1, 1])
        b = np.zeros((batch, 3, 1), dtype=np.float64)
        g = np.zeros((batch, 3), dtype=np.float64)
        for s, f in zip(step, affine, strict=True):
            a = np.einsum("bij, bjk -> bik", s, a)
            b = np.einsum("bij, bjk -> bik", s, b) + b_in
            g = np.einsum("bij, bj  -> bi", s, g) + f
        error = np.einsum("bij, bj -> bi", a, initial) + g
        error[..., [1, 2]] = np.arctan2(np.sin(error[..., [1, 2]]), np.cos(error[..., [1, 2]]))
        b_t_q = np.einsum("bij, jk -> bik", b.transpose(0, 2, 1), Q_LATERAL)
        inverse = -1 / (np.einsum("bij, bji -> bi", b_t_q, b) + R_LATERAL)
        return np.squeeze(inverse * np.einsum("bij, bj -> bi", b_t_q, error), axis=-1)


def simulate(
    proposals: np.ndarray, initial: np.ndarray, dt: float = 0.1, vehicle: Vehicle = PACIFICA
) -> np.ndarray:
    """`PDMSimulator.simulate_proposals`: (B, K, 11) proposals -> (B, K, 11) driven states.

    Only the proposals' x, y and heading are read. `initial` (11,) is the car's
    measured state at the rear axle, which every proposal starts from.
    """
    proposals = np.asarray(proposals, dtype=np.float64)
    tracker = Tracker(proposals, dt, vehicle)
    states = np.zeros(proposals.shape, dtype=np.float64)
    states[:, 0] = np.asarray(initial, dtype=np.float64)
    for t in range(1, proposals.shape[1]):
        accel, steer_rate = tracker.track(t - 1, states[:, t - 1])
        states[:, t] = propagate(states[:, t - 1], accel, steer_rate, dt, vehicle)
    return states


# ---------------------------------------------------------------------------
# scipy.signal.savgol_filter, mode="interp" (scipy 1.13)
# ---------------------------------------------------------------------------


def savgol_coeffs(window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    halflen, rem = divmod(window_length, 2)
    pos = halflen if rem else halflen - 0.5
    x = np.arange(-pos, window_length - pos, dtype=float)[::-1]
    a = x ** np.arange(polyorder + 1).reshape(-1, 1)
    y = np.zeros(polyorder + 1)
    y[deriv] = math.factorial(deriv) / delta**deriv
    return np.linalg.lstsq(a, y, rcond=None)[0]


def savgol_filter(
    x: np.ndarray, window_length: int, polyorder: int, deriv: int = 0, delta: float = 1.0
) -> np.ndarray:
    """Along the last axis, as scipy's with its default mode, "interp"."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if window_length > n:
        raise ValueError("window_length must be at most the length of the input")
    coeffs = savgol_coeffs(window_length, polyorder, deriv, delta)
    # scipy convolves (ndimage.convolve1d, mode="constant"); an even kernel sits
    # one sample left of centre there. Every output point whose window would
    # leave the data is replaced below, so the padding never reaches a result.
    weights = coeffs[::-1]
    shift = window_length // 2 - (0 if window_length % 2 else 1)
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(shift, window_length - 1 - shift)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_length, axis=-1)
    y = windows @ weights
    halflen = window_length // 2
    t = np.arange(window_length)
    for start, lo, hi in ((0, 0, halflen), (n - window_length, n - halflen, n)):
        edge = x[..., start : start + window_length].reshape(-1, window_length)
        fit = np.polynomial.polynomial.polyfit(t, edge.T, polyorder)  # (polyorder + 1, rows)
        if deriv:
            fit = np.polynomial.polynomial.polyder(fit, deriv)
        values = np.polynomial.polynomial.polyval(np.arange(lo - start, hi - start), fit) / delta**deriv
        y[..., lo:hi] = values.reshape(x.shape[:-1] + (hi - lo,))
    return y


# ---------------------------------------------------------------------------
# Comfort (NAVSIM v1 scoring/pdm_comfort_metrics.py)
# ---------------------------------------------------------------------------

MAX_ABS_MAG_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95
#: The six checks, in `ego_is_comfortable`'s order.
COMFORT_CHECKS = ("lon_accel", "lat_accel", "jerk", "lon_jerk", "yaw_accel", "yaw_rate")


def _acceleration(states: np.ndarray, coordinate: str, window_length: int = 8) -> np.ndarray:
    n_time = states.shape[-2]
    if coordinate == "x":
        a = states[..., AX]
    elif coordinate == "y":
        a = states[..., AY]
    else:
        a = np.hypot(states[..., AX], states[..., AY])
    return np.round(savgol_filter(a, min(window_length, n_time), 2), 8)


def _derivative(y: np.ndarray, times: np.ndarray, window_length: int = 5, polyorder: int = 2, deriv: int = 1):
    window_length = min(window_length, len(times))
    if not polyorder < window_length:
        raise ValueError(f"{polyorder} < {window_length} does not hold!")
    dx = np.diff(times)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")
    return savgol_filter(y, window_length, polyorder, deriv, float(dx.mean()))


def _jerk(states: np.ndarray, coordinate: str, times: np.ndarray, window_length: int) -> np.ndarray:
    return np.round(
        _derivative(_acceleration(states, coordinate), times, min(window_length, states.shape[-2])), 8
    )


def _phase_unwrap(headings: np.ndarray) -> np.ndarray:
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / (2 * np.pi)), axis=-1)
    return headings - 2 * np.pi * adjustments


def _yaw(states: np.ndarray, times: np.ndarray, deriv: int = 1, polyorder: int = 2) -> np.ndarray:
    # v1 passes its window_length nowhere: the derivative keeps its default 5.
    return np.round(
        _derivative(_phase_unwrap(states[..., HEADING]), times, polyorder=polyorder, deriv=deriv), 8
    )


def _within(metric: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    lo = lo if lo else -np.inf
    hi = hi if hi else np.inf
    return np.all((metric > lo) & (metric < hi), axis=-1)


def ego_is_comfortable(states: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(B, K, 11) driven states, (K,) seconds -> (B, 6) bool, `COMFORT_CHECKS` order."""
    n_time = states.shape[1]
    return np.stack(
        [
            _within(_acceleration(states, "x", n_time), MIN_LON_ACCEL, MAX_LON_ACCEL),
            _within(_acceleration(states, "y", n_time), -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
            _within(_jerk(states, "magnitude", times, n_time), -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
            _within(_jerk(states, "x", times, n_time), -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
            _within(_yaw(states, times, deriv=2, polyorder=3), -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
            _within(_yaw(states, times), -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
        ],
        axis=-1,
    )
