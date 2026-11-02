"""Route lookups the agent sends beside the command: the next lane change ahead.

The leaderboard hands every agent its route with a CARLA ``RoadOption`` per
point, lane changes included (``CHANGELANELEFT`` / ``CHANGELANERIGHT``; a
SequentialLaneChange route carries two). The 4-way ``Command`` has no slot for
them, so the agent sends them as STRAIGHT and the side is lost -- and the
junction maneuver ``nav_maneuver`` names is read as navigation text by other
policies, so lane changes cannot ride inside it either. They go on the wire as
their own pair of fields (``Observation.nav_lane_change`` / ``_m``).

Read off the dense route (the leaderboard's own, at 1 m spacing; see
``agent.set_global_plan``), not the base agent's 50 m downsample: a lane change
is two route points, which the downsample can skip entirely, and its distance is
what a policy times a lane change by.

Pure and CARLA-free, so it is unit-tested; it runs inside the CARLA client's
interpreter (Python 3.8 at the one site that has one), so keep the syntax to
what 3.8 accepts.
"""

from __future__ import annotations

#: RoadOption names of a lane change, and the side each one goes to.
LANE_CHANGE_SIDE = {"CHANGELANELEFT": "LEFT", "CHANGELANERIGHT": "RIGHT"}

#: Metres ahead a lane change is reported from. Further than any policy here
#: acts on (openpilot asks a few seconds ahead), and a lane change beyond it is
#: not yet the next thing the car has to do.
LANE_CHANGE_HORIZON_M = 100.0


def lane_change_ahead(options, s, index, horizon_m=LANE_CHANGE_HORIZON_M):
    """``(side, metres)`` of the first lane change at or ahead of route ``index``, else ``(None, None)``.

    ``options`` are the route's RoadOption names, ``s`` its along-route metres
    (both one entry per route point), ``index`` the route point nearest the ego.
    A lane change the ego is on reads 0 m until the ego's nearest point moves
    past it; then the next one, if any, is reported.
    """
    if not len(options) or index is None:
        return None, None
    start = max(0, min(int(index), len(options) - 1))
    for j in range(start, len(options)):
        ahead = float(s[j]) - float(s[start])
        if ahead > horizon_m:
            break
        side = LANE_CHANGE_SIDE.get(options[j])
        if side is not None:
            return side, round(ahead, 1)
    return None, None
