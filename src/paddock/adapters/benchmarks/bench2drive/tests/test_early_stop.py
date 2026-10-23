"""The early-stop rules, and the exit through the leaderboard's tick cap.

Pure: the rule is fed (game time, metres along the route) exactly as the agent
feeds it, and the stack walk is run under a stand-in ``_tick_scenario`` of the
shape the 7ec25d1 ScenarioManager has. No CARLA.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from paddock.adapters.benchmarks.bench2drive import early_stop
from paddock.adapters.benchmarks.bench2drive.early_stop import (
    COLLISION,
    LIGHT_QUEUE_M,
    OVER_TIME,
    PROGRESS_M,
    STUCK,
    TICK_CAP,
    EarlyStop,
    describe,
    end_route,
    replay,
    scenario_collisions,
    waiting_at_light,
)
from paddock.adapters.benchmarks.bench2drive.recorder import Route


def _first(rule: EarlyStop, drive) -> tuple[float, str] | None:
    for t, s in drive:
        reason = rule.update(t, s)
        if reason is not None:
            return t, reason
    return None


def _ticks(until_s: float, speed) -> list[tuple[float, float]]:
    """(game time, metres along) at 20 Hz, for a speed profile ``speed(t)``."""
    out, s, t = [], 0.0, 0.05
    while t <= until_s + 1e-9:
        out.append((round(t, 2), s))
        s += speed(t) * 0.05
        t += 0.05
    return out


def test_a_car_that_never_pulls_away_is_stuck_after_stuck_s() -> None:
    t, reason = _first(EarlyStop(stuck_s=30.0), _ticks(100.0, lambda t: 0.0))
    assert reason == STUCK
    assert t == pytest.approx(30.05, abs=0.06), "counted from the first tick"


def test_the_window_starts_at_the_last_progress_not_the_route_start() -> None:
    """Drives 40 s, then stands: stuck 30 s after it last got PROGRESS_M further."""
    t, reason = _first(EarlyStop(stuck_s=30.0), _ticks(200.0, lambda t: 5.0 if t < 40.0 else 0.0))
    assert reason == STUCK
    assert 69.0 < t < 71.0


def test_creeping_without_getting_anywhere_is_stuck_too() -> None:
    """The driver-resume nudge rolls a car a little at a time; below the blocked test's
    0.1 m/s it never makes PROGRESS_M in the window, though its speed is never zero."""
    assert _first(EarlyStop(stuck_s=30.0), _ticks(100.0, lambda t: 0.09))[1] == STUCK


def test_a_slow_car_that_keeps_getting_further_is_not_stuck() -> None:
    assert _first(EarlyStop(stuck_s=30.0), _ticks(200.0, lambda t: 0.15)) is None


def test_a_stop_shorter_than_stuck_s_is_not_stuck() -> None:
    """A red light: stands 25 s, then drives on."""
    drive = _ticks(150.0, lambda t: 0.0 if 20.0 <= t < 45.0 else 6.0)
    assert _first(EarlyStop(stuck_s=30.0), drive) is None


def test_going_backwards_is_not_progress() -> None:
    """Reversing off an obstacle and back is no further along than before."""
    drive = [(t / 20.0, 50.0 + 2.0 * (1 if (t // 100) % 2 else -1)) for t in range(1, 2000)]
    assert _first(EarlyStop(stuck_s=30.0), drive)[1] == STUCK


def test_over_time_fires_at_the_budget_whatever_the_progress() -> None:
    t, reason = _first(EarlyStop(after_s=88.0), _ticks(200.0, lambda t: 8.0))
    assert reason == OVER_TIME
    assert t == pytest.approx(88.0, abs=0.06)


def test_with_both_rules_the_first_to_fire_wins() -> None:
    stands_early = _ticks(200.0, lambda t: 0.0)
    assert _first(EarlyStop(stuck_s=30.0, after_s=88.0), stands_early)[1] == STUCK
    keeps_driving = _ticks(200.0, lambda t: 8.0)
    assert _first(EarlyStop(stuck_s=30.0, after_s=88.0), keeps_driving)[1] == OVER_TIME


def test_both_rules_off_never_fire() -> None:
    assert _first(EarlyStop(), _ticks(200.0, lambda t: 0.0)) is None


def test_replay_reads_a_recorded_drive_along_the_route() -> None:
    """A straight 100 m route at 1 m spacing; the car drives 40 m, then stands."""
    route = Route([[0.0, float(y)] for y in range(101)], ["LANEFOLLOW"] * 101)
    ticks = [(t, 0.0, s) for t, s in _ticks(120.0, lambda t: 4.0 if t < 10.0 else 0.0)]
    t, reason = replay(ticks, route, stuck_s=30.0)
    assert reason == STUCK and 39.0 < t < 41.0
    assert replay(ticks, Route([[0.0, float(y)] for y in range(101)], ["LANEFOLLOW"] * 101)) is None
    assert PROGRESS_M == 3.0, "the window the docs and the rule's numbers above assume"


class _Manager:
    """The shape of leaderboard 7ec25d1's ScenarioManager that ``end_route`` relies on."""

    def __init__(self) -> None:
        self.tick_count = 812

    def _tick_scenario(self, agent) -> None:
        agent()


def test_end_route_pushes_the_managers_tick_count_to_the_cap() -> None:
    manager = _Manager()
    ended = []
    manager._tick_scenario(lambda: ended.append(end_route()))
    assert ended == [True]
    assert manager.tick_count == TICK_CAP
    # The manager's own check, on its next tick: `tick_count += 1; if tick_count > 4000: raise`.
    manager.tick_count += 1
    assert manager.tick_count > TICK_CAP


def test_outside_a_leaderboard_tick_end_route_says_it_ended_nothing() -> None:
    assert end_route() is False


def test_the_remark_says_which_rules_and_why_the_scores_stand() -> None:
    remark = early_stop.remark({"stuck_s": 30.0, "expert_factor": 4.0, "progress_m": 3.0})
    assert "30 s without 3 m of route progress" in remark
    assert "4x the expert's time" in remark
    assert "expected to move" in remark and "failed drive" in remark
    mixed = early_stop.mixed_reason([{"stuck_s": 30.0}, {"stuck_s": 60.0}])
    assert "differ between source runs" in mixed and "30 s" in mixed and "60 s" in mixed


def test_a_collision_stops_the_route_only_when_that_rule_is_on() -> None:
    assert EarlyStop(collision=True).update(1.0, 0.0, collisions=1) == COLLISION
    assert EarlyStop(collision=True).update(1.0, 0.0, collisions=0) is None
    assert EarlyStop(stuck_s=30.0).update(1.0, 0.0, collisions=3) is None


def test_waiting_at_a_red_light_holds_the_stuck_clock() -> None:
    rule = EarlyStop(stuck_s=5.0)
    assert rule.update(0.0, 10.0) is None
    for t in range(1, 40):  # 39 s at a red light, not one metre on: not stuck
        assert rule.update(float(t), 10.0, waiting=True) is None
    assert rule.update(43.0, 10.0) is None, "the clock runs on when the light turns"
    assert rule.update(44.5, 10.0) == STUCK


def test_the_wait_pauses_the_stuck_clock_it_does_not_restart_it() -> None:
    """A car that stood 20 s on a green has 10 s left after the red: a light that keeps
    cycling must not keep a stalled car from ever being stuck."""
    rule = EarlyStop(stuck_s=30.0)
    ticks = [(t / 20, t / 20 >= 20.0 and t / 20 < 60.0) for t in range(0, 20 * 80)]
    fired = next(t for t, waiting in ticks if rule.update(t, 5.0, waiting=waiting) is not None)
    assert fired == pytest.approx(70.0, abs=0.06)


def test_a_car_queued_short_of_a_red_light_is_waiting() -> None:
    """CARLA's `is_at_traffic_light` is true only inside the light's trigger box at the
    stop line; route 25975's drive stood 9.8 m short of a red one for 45 s."""
    red_ahead = {"id": 7, "d": 9.8, "state": "Red"}
    assert waiting_at_light(False, "Green", red_ahead)
    assert waiting_at_light(False, "Green", {**red_ahead, "state": "Yellow"})
    assert waiting_at_light(True, "Red", None), "inside the box, as before"
    assert not waiting_at_light(False, "Green", {**red_ahead, "state": "Green"})
    assert not waiting_at_light(False, "Green", {**red_ahead, "d": LIGHT_QUEUE_M + 1.0})
    assert not waiting_at_light(False, "Green", None)
    assert not waiting_at_light(True, "Green", {**red_ahead, "state": "Green"})


class _Light:
    def __init__(self, state: str) -> None:
        self.state = state

    def get_state(self) -> str:
        return self.state


class _Hero:
    """The ego standing at ``y`` on a straight route, outside every light's trigger box."""

    def __init__(self, y: float) -> None:
        self.y = y

    def get_location(self) -> SimpleNamespace:
        return SimpleNamespace(x=0.0, y=self.y)

    def is_at_traffic_light(self) -> bool:
        return False

    def get_traffic_light_state(self) -> str:
        return "Green"


def _stand_short_of_a_light(load_agent, state: str) -> list[tuple[float, str]]:
    """45 s of the agent's own early-stop glue, the car 10 m short of a light in ``state``."""
    client = object.__new__(load_agent().PaddockB2DAgent)
    client._nav_route = Route([[0.0, float(y)] for y in range(101)], ["LANEFOLLOW"] * 101)
    client._stop_lights = [{"id": 7, "i": 50, "x": 0.0, "y": 50.0}]
    client._lights_by_id = {7: _Light(state)}
    client._hero = _Hero(40.0)
    client._early_stop = EarlyStop(stuck_s=30.0)
    client._stop_record = {"reason": None}
    client._stop_requested, client._t = False, 0
    ended: list[tuple[float, str]] = []
    client._end_early = lambda reason, t, along: ended.append((t, reason))
    for k in range(1, 20 * 45):
        client._check_early_stop(k / 20)
        if ended:
            break
    return ended


def test_the_agent_holds_the_clock_for_a_car_waiting_short_of_a_red_light(load_agent) -> None:
    assert _stand_short_of_a_light(load_agent, "Red") == []
    ((t, reason),) = _stand_short_of_a_light(load_agent, "Green")
    assert reason == STUCK and t == pytest.approx(30.05, abs=0.06)


def test_describe_names_the_collision_rule() -> None:
    assert describe({"stuck_s": 5.0, "collision": True}) == (
        "5 s without 3 m of route progress or a collision the leaderboard records"
    )


def test_the_collision_count_is_the_leaderboards_own_read_up_the_stack() -> None:
    """``scenario_collisions`` finds the manager by its `_tick_scenario` frame, as `end_route` does."""

    class CollisionTest:  # the leaderboard criterion's class name is what is matched
        def __init__(self, n):
            self.events = [object()] * n  # what it recorded, and the penalty is computed from
            # A sidewalk brush raises actual_value without an event: not a collision the score holds.
            self.actual_value = n + 3

    class RunningRedLightTest:
        actual_value = 7

    class Scenario:
        def get_criteria(self):
            return [CollisionTest(2), RunningRedLightTest()]

    class Manager:
        tick_count = 10
        scenario = Scenario()

        def _tick_scenario(self):
            return scenario_collisions()

    assert Manager()._tick_scenario() == 2
    assert scenario_collisions() is None, "outside the leaderboard's tick there is nothing to read"
