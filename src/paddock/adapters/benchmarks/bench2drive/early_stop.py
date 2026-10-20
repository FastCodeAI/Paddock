"""When to end a Bench2Drive route before the leaderboard would, and how.

The leaderboard ends a route on completion, a deviation, its blocked test (60 s
under 0.1 m/s, which the scenarios that make the ego wait for a gap -- such as
AccidentTwoWays -- switch off) or the 4000-tick cap (200 s of game time). A car
that stands behind an obstacle with the blocked test off, or that a driver-resume
nudge keeps rolling a metre at a time, therefore drives the whole cap: on the
openpilot sweep, three such routes took 27-42 minutes of wall clock each, and
every minute after their first half-minute standing still added nothing.

Three rules, each off unless the site sets it (``[benchmarks.bench2drive]
stop_stuck_s`` / ``stop_expert_factor`` / ``stop_on_collision``, see ``harness.py``):

* **stuck** -- the ego has not got :data:`PROGRESS_M` further along the route in
  ``stop_stuck_s`` game seconds. Progress along the route, not speed, so a car
  that creeps without getting anywhere counts too. Standing at a red light is
  waiting, not being stuck: the clock is held while the ego is at one, so a
  short window does not end every route that meets a red signal.
* **over_time** -- the game clock has reached ``stop_expert_factor`` times the
  expert's (Think2Drive's) time on the route.
* **collision** -- the leaderboard's own ``CollisionTest`` has counted a
  collision: the one the score's penalty is for (:func:`scenario_collisions`).

And one that is not a rule: **manual** -- someone asked for it, by creating the
route's stop-request file (the dashboard's "Stop route" button does). A person
choosing when a route ends is not a measurement, so a run with one is not ranked.

Any of them ends the route through the leaderboard's own tick cap
(:func:`end_route`), the one exit that is scored as it stands: the route
completion and infractions of that moment, status ``Failed - TickRuntime``.
Raising from the agent instead would score as "Agent crashed" -- not a
measurement, so a sweep would drive the route again. What ended it, and when,
the agent writes to :data:`RECORD` beside ``results.json``.

A route either rule ends is a failed drive. A car in a driving scene is expected
to move, and one that gets no further along its route for that long, or needs
that many times the expert's time, has failed the scene -- so the route is
scored as it stood when it ended, and a run with the rules on ranks. Its notes
say which rules it ran under, and why its scores stand (:func:`remark`). Routes
stopped under *different* rules are not one experiment, so a pool of them does
not rank (:func:`mixed_reason`).

Kept 3.8-compatible: ``agent.py`` runs :class:`EarlyStop` every tick inside the
CARLA client, and :func:`replay` runs the same rule over a recorded
``ticks.jsonl`` to say whether a route driven without it would have been ended.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

#: Bench2Drive's per-route tick cap: ``ScenarioManager._tick_scenario`` raises
#: TickRuntimeError once ``tick_count > 4000`` (scenario_manager.py:182 at 7ec25d1).
TICK_CAP = 4000

#: Metres along the route that count as progress. The blocked test counts a car
#: under 0.1 m/s as standing, which over a 30 s window is 3 m.
PROGRESS_M = 3.0

#: The reasons, as ``early_stop.json`` and the per-route metrics name them.
STUCK = "stuck"
OVER_TIME = "over_time"
COLLISION = "collision"
MANUAL = "manual"
REASONS = (STUCK, OVER_TIME, COLLISION, MANUAL)

#: What the agent writes beside ``results.json`` when it ends a route early.
RECORD = "early_stop.json"


class EarlyStop:
    """Fed each tick's game time and along-route position; says when to stop, and why.

    ``stuck_s`` or ``after_s`` of None, or ``collision`` False, switches that rule off.
    """

    def __init__(
        self, stuck_s: float | None = None, after_s: float | None = None, collision: bool = False
    ) -> None:
        self.stuck_s = stuck_s
        self.after_s = after_s
        self.collision = collision
        # (metres along, game time) when the car last got PROGRESS_M further on
        self._mark: tuple[float, float] | None = None

    def update(self, t: float, s: float, collisions: int = 0, waiting: bool = False) -> str | None:
        """The reason to stop at game time ``t``, the ego ``s`` m along the route, or None.

        ``collisions`` is the leaderboard's count so far; ``waiting`` says the ego
        stands at a red light, which holds the stuck clock rather than running it.
        """
        if self._mark is None or s >= self._mark[0] + PROGRESS_M or waiting:
            self._mark = (s, t)
        if self.collision and collisions > 0:
            return COLLISION
        if self.after_s is not None and t >= self.after_s:
            return OVER_TIME
        if self.stuck_s is not None and t - self._mark[1] >= self.stuck_s:
            return STUCK
        return None


def replay(
    ticks: Iterable[tuple],
    route: Any,
    stuck_s: float | None = None,
    after_s: float | None = None,
) -> tuple[float, str] | None:
    """``(game time, reason)`` at which :class:`EarlyStop` would have ended a drive, or None.

    ``ticks`` is the ego's (game time, x, y) per tick, as ``ticks.jsonl`` holds
    them, optionally with a fourth item, whether it stood at a red light then,
    and ``route`` a ``recorder.Route`` of the leaderboard's plan -- the agent
    feeds the rule the same way. A rule changes nothing until it fires, so a
    drive this returns None for is one the rule would have let run to its end.
    The collision rule needs the leaderboard's count, which a recording lacks.
    """
    rule = EarlyStop(stuck_s, after_s)
    for tick in ticks:
        t, x, y = tick[:3]
        index, _ = route.locate(x, y)
        reason = rule.update(t, float(route.s[index]), waiting=bool(tick[3]) if len(tick) > 3 else False)
        if reason is not None:
            return t, reason
    return None


def _scenario_manager() -> Any:
    """The leaderboard's ScenarioManager, found up the stack from inside ``run_step``, or None.

    ``ScenarioManager._tick_scenario`` calls the agent wrapper, which calls
    ``run_step``; the leaderboard holds no reference to the manager anywhere an
    agent can reach, so it is found by that method's name and the counter it keeps.
    """
    frame = sys._getframe(1)
    while frame is not None:
        manager = frame.f_locals.get("self")
        if frame.f_code.co_name == "_tick_scenario" and hasattr(manager, "tick_count"):
            return manager
        frame = frame.f_back
    return None


def scenario_collisions() -> int | None:
    """Collisions the leaderboard has counted on this route so far, or None if it cannot be read.

    The route's ``CollisionTest`` criterion (``route_scenario._create_test_criteria``
    at 7ec25d1) counts each collision once (the same actor within 5 s is one) and
    is what the infraction penalty is computed from, so stopping on it stops on
    exactly the collisions the score holds. Read through the manager's
    ``scenario.get_criteria()``; it is updated after the agent's tick, so a
    collision is seen one tick after it happens.
    """
    manager = _scenario_manager()
    scenario = getattr(manager, "scenario", None)
    if scenario is None or not hasattr(scenario, "get_criteria"):
        return None
    try:
        return sum(
            int(getattr(c, "actual_value", 0) or 0)
            for c in scenario.get_criteria()
            if type(c).__name__ == "CollisionTest"
        )
    except Exception:  # a criterion torn down mid-read: no answer this tick
        return None


def end_route() -> bool:
    """Push the leaderboard's tick count to :data:`TICK_CAP`, so it ends the route next tick.

    ``ScenarioManager._tick_scenario`` calls the agent wrapper, which calls
    ``run_step``; the manager is found up the stack by that method's name and the
    counter it keeps, because the leaderboard holds no reference to it anywhere an
    agent can reach. False when there is none -- a leaderboard other than the one
    pinned -- so the caller can say so instead of believing the route ended.
    """
    manager = _scenario_manager()
    if manager is None:
        return False
    manager.tick_count = max(int(manager.tick_count), TICK_CAP)
    return True


def describe(rules: dict[str, Any]) -> str:
    """The rules of a run config's ``early_stop``, in words."""
    parts = []
    if rules.get("stuck_s"):
        progress_m = rules.get("progress_m", PROGRESS_M)
        parts.append(f"{rules['stuck_s']:g} s without {progress_m:g} m of route progress")
    if rules.get("expert_factor"):
        parts.append(f"{rules['expert_factor']:g}x the expert's time on the route")
    if rules.get("collision"):
        parts.append("a collision the leaderboard counts")
    return " or ".join(parts)


def remark(rules: dict[str, Any]) -> str:
    """The note a run with these rules carries: which rules, and why its scores stand."""
    return (
        f"early stop ({describe(rules)}): a car in a driving scene is expected to move, to finish in "
        "reasonable time and not to collide, so a route ended this way is a failed drive, scored as it "
        "stood when it ended (a collision's penalty included)"
    )


def manual_reason(n: int) -> str:
    """Why a run in which someone ended routes by hand is not ranked."""
    return (
        f"{n} route(s) ended by hand (a stop request): scored as they stood, at a moment a person chose, "
        "which no rule reproduces"
    )


def mixed_reason(rules: list[Any]) -> str:
    """Why a pool whose source runs stopped routes under different rules does not rank."""
    said = "; ".join(describe(r) if isinstance(r, dict) else str(r) for r in rules)
    return f"early stop settings differ between source runs ({said}): one mean describes none of them"
