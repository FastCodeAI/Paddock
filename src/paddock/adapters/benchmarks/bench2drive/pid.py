"""Waypoints -> vehicle control: the dual-PID tracker Bench2Drive's baselines use, reimplemented.

The Bench2Drive checkout ships no controller. The tracker its baselines drive
with (TCP's, reused unchanged by the UniAD / VAD agents in Bench2DriveZoo's
``team_code/pid_controller.py``) lives in a *second* repository that a site set
up from the Bench2Drive README does not have, so an import of it passes on the
author's machine and fails on every other one. Paddock reimplements it here:
the adapter runs from a plain checkout, and the control law under the score is
pinned where a change to it shows up in a diff.

Same law and constants as TCP (Wu et al., 2022):

* lateral: a PID on the heading error to an aim point, the waypoint whose
  segment midpoint lies nearest ``AIM_DIST`` ahead;
* longitudinal: a PID on the speed error, where the desired speed is the mean
  waypoint spacing divided by the waypoint interval, over the first
  ``CONTROL_HORIZON_S`` of plan (see there for why the window is fixed);
* brake when the desired speed is below ``BRAKE_SPEED`` or the car is more
  than ``BRAKE_RATIO`` times faster than desired.

TCP also blends in the route's next target point for steering; paddock's wire
format carries the route as a discrete ``Command``, not a point, so that branch
is left out (``use_target_to_aim = False`` in TCP's terms).

The tracker reads every model's plan the same way: its own points at 0.5, 1.0,
... 3.0 s (``on_grid``), six points 0.5 s apart. That is the whole of what the
wire's ``trajectory`` carries for a plan 0.5 s apart, and every fifth point of a
plan 0.1 s apart (alpamayo2edge).

Pure numpy and no CARLA, so it is unit-tested with the rest of the adapter. It
also runs inside the CARLA client's own interpreter, which is Python 3.8 at the
one site that has one -- keep the syntax to what 3.8 accepts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from paddock.policy.protocol import grid_stride

#: TCP's gains, unchanged in Bench2DriveZoo.
TURN_KP, TURN_KI, TURN_KD, TURN_N = 1.25, 0.75, 0.3, 40
SPEED_KP, SPEED_KI, SPEED_KD, SPEED_N = 5.0, 0.5, 1.0, 40
#: Metres ahead the steering aims at.
AIM_DIST = 4.0
MAX_THROTTLE = 0.75
#: m/s. A plan slower than this is a stop: brake rather than crawl.
BRAKE_SPEED = 0.4
#: Brake when actual speed exceeds the desired speed by this factor.
BRAKE_RATIO = 1.1
#: m/s. The speed error the throttle PID sees is capped here, so a plan far
#: faster than the car does not produce a lurch.
CLIP_DELTA = 0.25

#: Seconds of plan the control law reads. TCP's longitudinal term is the *mean*
#: waypoint spacing, so without this the desired speed is a function of how far
#: the model happened to plan: a model emitting 4 s of plan and one emitting
#: 10 s of the same drive get different throttle from the same intent, and
#: extending any model's horizon would silently move its recorded driving score.
#: Fixing the window makes the law a property of the controller, which is what
#: TCP's constants were tuned for. 4.0 s is what every model here reached when
#: this was introduced, so no recorded number moves: alpamayo15 and openpilot
#: plan exactly 8 waypoints at 0.5 s, and visionpilot's 3 s plan is shorter than
#: the window and therefore untouched.
CONTROL_HORIZON_S = 4.0


def on_grid(waypoints, dt: float, grid_dt: float, n: int) -> np.ndarray:
    """The plan's own points at ``grid_dt``, ``2 grid_dt``, ... ``n grid_dt`` s: (n, 2).

    Nothing is interpolated. ``waypoints`` is the whole plan, (N, 2) or (N, 3) with yaw, evenly spaced by
    ``dt`` starting at t = ``dt``. For a plan already ``grid_dt`` apart these are its
    first ``n`` points. Raises ValueError for a spacing with no points on the grid
    (``grid_stride``) and for a plan that ends before ``n grid_dt`` s: the controller
    would otherwise read a shorter horizon than every other model gives it.
    """
    stride = grid_stride(dt, grid_dt)
    if stride is None:
        raise ValueError(
            f"a plan {dt:g} s apart has no points on the {grid_dt:g} s grid the controller reads"
        )
    wps = np.asarray(waypoints, dtype=np.float64)
    if wps.ndim != 2 or wps.shape[1] < 2:
        raise ValueError(f"need (N, 2) or (N, 3) waypoints, got shape {wps.shape}")
    picked = wps[stride - 1 :: stride][:n, :2]
    if len(picked) < n:
        raise ValueError(
            f"a plan of {len(wps)} points {dt:g} s apart ends at {len(wps) * dt:g} s; the "
            f"controller reads its points to {n * grid_dt:g} s"
        )
    return picked


class PID:
    """TCP's windowed PID: the integral term is the mean of the last ``n`` errors."""

    def __init__(self, k_p: float, k_i: float, k_d: float, n: int) -> None:
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self._window: deque[float] = deque([0.0] * n, maxlen=n)

    def step(self, error: float) -> float:
        self._window.append(float(error))
        integral = float(np.mean(self._window))
        derivative = self._window[-1] - self._window[-2]
        return self.k_p * error + self.k_i * integral + self.k_d * derivative


class WaypointTracker:
    """Turn a paddock trajectory into (steer, throttle, brake), one tick at a time.

    Stateful: both PIDs keep an error window, so use one tracker per route.
    """

    def __init__(self, waypoint_dt: float = 0.5) -> None:
        if waypoint_dt <= 0:
            raise ValueError(f"waypoint_dt must be positive, got {waypoint_dt}")
        self.waypoint_dt = waypoint_dt
        self.turn = PID(TURN_KP, TURN_KI, TURN_KD, TURN_N)
        self.speed = PID(SPEED_KP, SPEED_KI, SPEED_KD, SPEED_N)

    def _as_xy(self, waypoints) -> np.ndarray:
        """Parse to (N, 2) and clip to `CONTROL_HORIZON_S`. A shorter plan passes through."""
        wps = np.asarray(waypoints, dtype=np.float64)
        if wps.ndim != 2 or wps.shape[0] < 2 or wps.shape[1] < 2:
            raise ValueError(f"need at least two (x, y) waypoints, got shape {wps.shape}")
        return wps[: max(2, int(round(CONTROL_HORIZON_S / self.waypoint_dt))), :2]

    def _steer(self, wps: np.ndarray) -> float:
        """Lateral control from parsed (N, 2) waypoints, advancing the turn PID.

        Aim at the waypoint opening the segment whose midpoint is nearest
        AIM_DIST from the car. Heading error is measured from the forward axis
        toward the right, in units of 90 degrees, so a point to the right gives
        a positive error and a positive (right) steer.
        """
        midpoints = np.linalg.norm((wps[1:] + wps[:-1]) / 2.0, axis=1)
        aim = wps[int(np.argmin(np.abs(midpoints - AIM_DIST)))]
        angle = float(np.degrees(np.arctan2(-aim[1], aim[0])) / 90.0)
        return float(np.clip(self.turn.step(angle), -1.0, 1.0))

    def steer_only(self, waypoints) -> float:
        """Steering alone, advancing ONLY the turn PID -- for model-native
        longitudinal modes that drive the speed PID themselves. Calling
        ``control`` there would step the speed PID a second time in the same
        tick with a different target, mixing two error histories into its window.
        """
        return self._steer(self._as_xy(waypoints))

    def control(self, waypoints, speed: float) -> tuple[float, float, float, float]:
        """``(steer, throttle, brake, desired_speed)`` for one tick.

        ``waypoints``: paddock's plan -- ego frame, x forward, y LEFT, metres,
        evenly spaced by ``waypoint_dt`` starting at t = ``waypoint_dt``; at
        least two. ``speed``: current forward speed, m/s.

        Steer follows CARLA's sign (+1 is full right); throttle and brake are
        0..1, brake being all-or-nothing as in TCP.
        """
        wps = self._as_xy(waypoints)

        steps = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        desired_speed = float(steps.mean() / self.waypoint_dt)

        steer = self._steer(wps)

        brake = desired_speed < BRAKE_SPEED or speed > BRAKE_RATIO * desired_speed
        delta = float(np.clip(desired_speed - speed, 0.0, CLIP_DELTA))
        throttle = float(np.clip(self.speed.step(delta), 0.0, MAX_THROTTLE))
        if brake:
            throttle = 0.0
        return steer, throttle, (1.0 if brake else 0.0), desired_speed
