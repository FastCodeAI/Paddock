"""Waypoints -> vehicle control: Bench2Drive's dual-PID tracker, vendored.

The Bench2Drive checkout ships no controller. The tracker its baselines drive
with (TCP's, reused unchanged by the UniAD / VAD agents in Bench2DriveZoo's
``team_code/pid_controller.py``) lives in a *second* repository that a site set
up from the Bench2Drive README does not have, so an import of it passes on the
author's machine and fails on every other one. Paddock owns its own copy: the
adapter runs from a plain checkout, and the control law under the score is
pinned here where a change to it shows up in a diff.

Same law and constants as TCP (Wu et al., 2022):

* lateral: a PID on the heading error to an aim point, the waypoint whose
  segment midpoint lies nearest ``AIM_DIST`` ahead;
* longitudinal: a PID on the speed error, where the desired speed is the mean
  waypoint spacing divided by the waypoint interval;
* brake when the desired speed is below ``BRAKE_SPEED`` or the car is more
  than ``BRAKE_RATIO`` times faster than desired.

TCP also blends in the route's next target point for steering; paddock's wire
format carries the route as a discrete ``Command``, not a point, so that branch
is left out (``use_target_to_aim = False`` in TCP's terms).

Pure numpy and no CARLA, so it is unit-tested with the rest of the adapter. It
also runs inside the CARLA client's own interpreter, which is Python 3.8 at the
one site that has one -- keep the syntax to what 3.8 accepts.
"""

from __future__ import annotations

from collections import deque

import numpy as np

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

    def control(self, waypoints, speed: float) -> tuple[float, float, float, float]:
        """``(steer, throttle, brake, desired_speed)`` for one tick.

        ``waypoints``: paddock's plan -- ego frame, x forward, y LEFT, metres,
        evenly spaced by ``waypoint_dt`` starting at t = ``waypoint_dt``; at
        least two. ``speed``: current forward speed, m/s.

        Steer follows CARLA's sign (+1 is full right); throttle and brake are
        0..1, brake being all-or-nothing as in TCP.
        """
        wps = np.asarray(waypoints, dtype=np.float64)
        if wps.ndim != 2 or wps.shape[0] < 2 or wps.shape[1] < 2:
            raise ValueError(f"need at least two (x, y) waypoints, got shape {wps.shape}")
        wps = wps[:, :2]

        steps = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        desired_speed = float(steps.mean() / self.waypoint_dt)

        # Aim at the waypoint opening the segment whose midpoint is nearest
        # AIM_DIST from the car. Heading error is measured from the forward
        # axis toward the right, in units of 90 degrees, so a point to the
        # right gives a positive error and a positive (right) steer.
        midpoints = np.linalg.norm((wps[1:] + wps[:-1]) / 2.0, axis=1)
        aim = wps[int(np.argmin(np.abs(midpoints - AIM_DIST)))]
        angle = float(np.degrees(np.arctan2(-aim[1], aim[0])) / 90.0)
        steer = float(np.clip(self.turn.step(angle), -1.0, 1.0))

        brake = desired_speed < BRAKE_SPEED or speed > BRAKE_RATIO * desired_speed
        delta = float(np.clip(desired_speed - speed, 0.0, CLIP_DELTA))
        throttle = float(np.clip(self.speed.step(delta), 0.0, MAX_THROTTLE))
        if brake:
            throttle = 0.0
        return steer, throttle, (1.0 if brake else 0.0), desired_speed
