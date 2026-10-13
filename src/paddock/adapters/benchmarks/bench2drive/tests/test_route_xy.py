"""The route polyline the CARLA client sends (`route_xy`), checked for the sign
CARLA's left-handed world gets wrong most easily: a route point on the ego's
left must arrive with y > 0. The client imports carla and the leaderboard, so
they are stood in for; only the geometry under test is real."""

from __future__ import annotations

import enum
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

AGENT = Path(__file__).resolve().parents[1] / "agent.py"


class _RoadOption(enum.IntEnum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class _Location:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y

    def distance(self, other: _Location) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class _Hero:
    """A CARLA actor at (x, y) with yaw in degrees, CARLA's own convention."""

    def __init__(self, x: float, y: float, yaw_deg: float) -> None:
        self.location = _Location(x, y)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        # carla.Transform.get_matrix() with roll = pitch = 0.
        self.matrix = [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    def get_location(self) -> _Location:
        return self.location

    def get_transform(self):
        return types.SimpleNamespace(get_matrix=lambda: self.matrix)


@pytest.fixture
def agent_module(monkeypatch):
    def module(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        return mod

    provider = types.SimpleNamespace(get_hero_actor=lambda: None, get_world=lambda: None)
    stubs = {
        "carla": module("carla"),
        "agents": module("agents"),
        "agents.navigation": module("agents.navigation"),
        "agents.navigation.local_planner": module("agents.navigation.local_planner", RoadOption=_RoadOption),
        "leaderboard": module("leaderboard"),
        "leaderboard.autoagents": module("leaderboard.autoagents"),
        "leaderboard.autoagents.autonomous_agent": module(
            "leaderboard.autoagents.autonomous_agent",
            AutonomousAgent=object,
            Track=types.SimpleNamespace(SENSORS="SENSORS"),
        ),
        "srunner": module("srunner"),
        "srunner.scenariomanager": module("srunner.scenariomanager"),
        "srunner.scenariomanager.carla_data_provider": module(
            "srunner.scenariomanager.carla_data_provider", CarlaDataProvider=provider
        ),
    }
    for name in ("cv2", "requests"):
        if importlib.util.find_spec(name) is None:
            stubs[name] = module(name)
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("_paddock_b2d_agent_under_test", AGENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _agent(agent_module, hero: _Hero, plan_xy: list[tuple[float, float]]):
    agent = object.__new__(agent_module.PaddockB2DAgent)
    agent._hero = hero
    agent._global_plan_world_coord = [
        (types.SimpleNamespace(location=_Location(x, y)), _RoadOption.LANEFOLLOW) for x, y in plan_xy
    ]
    return agent


def test_a_route_on_the_egos_left_arrives_with_positive_y(agent_module) -> None:
    """CARLA's y points right: a lane 3 m to the left is CARLA y = -3, paddock y = +3.

    The plan is spaced as the agent receives it. The leaderboard hands `set_global_plan`
    a 1 m route and the base class keeps `downsample_route(route, 50)` of it (Bench2Drive
    7ec25d1, autonomous_agent.py): a point at each road-option change, otherwise every
    ~50 m. `route_xy` adds no points between them."""
    plan = [(float(x), -3.0) for x in range(-50, 401, 50)]
    route = np.array(_agent(agent_module, _Hero(0.0, 0.0, 0.0), plan)._route_xy())
    assert np.allclose(route[:, 1], 3.0)
    assert route[0, 0] == pytest.approx(-50.0), "the first plan point at least 10 m behind the nearest"
    assert route[-1, 0] == pytest.approx(200.0), "the first at least 200 m ahead of it"
    assert np.allclose(np.diff(route[:, 0]), 50.0), "no denser than the leaderboard's plan"


def test_the_polyline_turns_with_the_ego(agent_module) -> None:
    """Facing CARLA +y (yaw 90), straight ahead is +x and CARLA +x is the ego's left."""
    plan = [(3.0, float(y)) for y in range(0, 100)]
    route = np.array(_agent(agent_module, _Hero(0.0, 0.0, 90.0), plan)._route_xy())
    assert route[-1] == pytest.approx([99.0, 3.0], abs=1e-6)
    assert np.allclose(route[:, 1], 3.0)


def test_a_route_that_ends_ahead_stops_where_it_ends(agent_module) -> None:
    """A polyline shorter than 200 m is the route's end -- AutoE2E draws its destination there."""
    plan = [(float(x), 0.0) for x in range(0, 60)]
    route = np.array(_agent(agent_module, _Hero(20.0, 0.0, 0.0), plan)._route_xy())
    assert route[-1, 0] == pytest.approx(39.0)


def test_no_plan_sends_no_route(agent_module) -> None:
    assert _agent(agent_module, _Hero(0.0, 0.0, 0.0), [])._route_xy() is None
