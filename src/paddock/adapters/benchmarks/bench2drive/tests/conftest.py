"""The CARLA client (`agent.py`) under test. It imports carla and the leaderboard,
so they are stood in for; only the code under test is real."""

from __future__ import annotations

import enum
import importlib.util
import sys
import types
from pathlib import Path

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


@pytest.fixture
def load_agent(monkeypatch):
    """``load(**env)``: agent.py loaded afresh with ``env`` set, as the harness sets it for a route.

    The agent reads its environment once, at import, so every load is a new module.
    """

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

    def load(**env: str) -> types.ModuleType:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        spec = importlib.util.spec_from_file_location("_paddock_b2d_agent_under_test", AGENT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load


@pytest.fixture
def agent_module(load_agent):
    return load_agent()
