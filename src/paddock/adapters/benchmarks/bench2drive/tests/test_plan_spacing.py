"""A plan whose points are not 0.5 s apart is driven on its own points at every 0.5 s.

The controller reads six points 0.5 s apart. The node's reply carries them twice
over for a plan 0.5 s apart: ``trajectory`` is the plan's first six points and
``plan`` the whole plan with its spacing. For alpamayo2edge, 0.1 s apart, the first
six points are 0.6 s of plan; read as 3 s, the car would follow it at a fifth of
the planned speed. So the client reads ``plan`` at its own points every 0.5 s
(``pid.on_grid``), the harness refuses a spacing with none before anything
launches, and the record keeps what the car followed. The client imports carla and
the leaderboard; conftest.py stands in for them (``agent_module``).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive import recorder
from paddock.adapters.benchmarks.bench2drive.harness import BENCHMARK, WAYPOINT_DT
from paddock.adapters.benchmarks.bench2drive.pid import WaypointTracker
from paddock.policy.protocol import Trajectory
from paddock.registry import Registry


def _agent(agent_module):
    """The client with nothing but what `decide` reads: a fresh tracker on the 0.5 s grid."""
    agent_module.carla.VehicleControl = types.SimpleNamespace
    agent = object.__new__(agent_module.PaddockB2DAgent)
    agent._tracker = WaypointTracker(agent_module.WAYPOINT_DT)
    return agent


def _fine_reply(speed: float = 6.0) -> dict:
    """What alpamayo2edge's node sends: 64 poses 0.1 s apart, with heading, bending left."""
    t = 0.1 * np.arange(1, 65)
    waypoints = np.stack([speed * t, 0.05 * t**2, np.arctan2(0.1 * t, speed)], axis=1)
    return Trajectory(waypoints, dt=0.1).to_wire()


def test_the_client_drives_a_fine_plan_on_its_own_points_every_half_second(agent_module) -> None:
    agent = _agent(agent_module)
    reply = _fine_reply(speed=6.0)
    whole = np.asarray(reply["plan"]["waypoints"])
    control, desired, plan, native = agent.decide(reply, speed=6.0)

    assert plan.dt == WAYPOINT_DT == 0.5
    assert np.array_equal(plan.xy, whole[4:30:5, :2]), "the points at 0.5, 1.0, ... 3.0 s, none made up"
    assert desired == pytest.approx(6.0, rel=1e-3), "the planned speed, not a fifth of it"
    assert native is None
    assert control.brake == 0.0
    # What the reply's `trajectory` holds for this plan, and what the client used to drive:
    assert np.array_equal(np.asarray(reply["trajectory"]), whole[:6, :2])
    assert WaypointTracker(0.5).control(reply["trajectory"], 6.0)[3] == pytest.approx(6.0 / 5, rel=1e-3)


def test_a_plan_half_a_second_apart_is_driven_exactly_as_before(agent_module) -> None:
    """Every model before alpamayo2edge: the six points are the reply's `trajectory`, bit for bit,
    and the controls are the ones `trajectory` gave."""
    t = 0.5 * np.arange(1, 9)
    reply = Trajectory(np.stack([4.0 * t, 0.3 * t**2], axis=1), dt=0.5).to_wire()
    control, desired, plan, _ = _agent(agent_module).decide(reply, speed=3.0)
    assert plan.xy.tolist() == reply["trajectory"]
    assert (control.steer, control.throttle, control.brake, desired) == WaypointTracker(0.5).control(
        reply["trajectory"], 3.0
    )


def test_a_reply_whose_plan_has_no_points_on_the_grid_is_refused(agent_module) -> None:
    """A NaN-free plan the controller cannot read is an error on the route, not a slow drive."""
    reply = Trajectory(np.stack([np.arange(1.0, 21.0), np.zeros(20)], axis=1), dt=0.2).to_wire()
    with pytest.raises(ValueError, match="no points on the 0.5 s grid"):
        _agent(agent_module).decide(reply, speed=0.0)


def test_the_harness_refuses_a_spacing_off_the_grid_before_anything_launches() -> None:
    harness = BENCHMARK(Registry.load().benchmarks["bench2drive"], None)
    models = Registry.load().models
    spec = models["constant_velocity"]
    for dt in (0.5, 0.25, 0.1):
        assert harness.input_problems(types.SimpleNamespace(name=spec.name, waypoint_dt=dt)) == []
    (problem,) = harness.input_problems(types.SimpleNamespace(name="m", waypoint_dt=0.2))
    assert "plans 0.2 s apart" in problem and "must divide 0.5 s" in problem
    for name, model in models.items():
        if "bench2drive" in model.benchmarks:
            assert harness.input_problems(model) == [], name


def test_the_record_keeps_the_points_the_car_followed() -> None:
    """`plan` in ticks.jsonl is what the controller read, `dt` apart; without one given, the
    reply's `trajectory`, which is the same six points for a plan 0.5 s apart."""
    reply = _fine_reply()
    followed = np.asarray(reply["plan"]["waypoints"])[4:30:5, :2].tolist()
    ego, ctrl = {"x": 0.0, "y": 0.0, "yaw": 0.0}, {"steer": 0.0, "throttle": 0.3, "brake": 0.0, "v_des": 6.0}
    record = recorder.tick_record(1, 0.05, 900.0, ego, {"command": 3}, reply, ctrl, plan=followed, dt=0.5)
    assert record["plan"] == recorder.rounded(followed, 3) and record["dt"] == 0.5
    record = recorder.tick_record(1, 0.05, 900.0, ego, {"command": 3}, reply, ctrl)
    assert record["plan"] == recorder.rounded(reply["trajectory"], 3)
