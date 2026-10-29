"""How often the CARLA client asks the node for a plan (`PADDOCK_INFER_HZ`).

At 2 Hz on Bench2Drive's 20 Hz simulation the client asks once per 10 frames and
follows the last plan, unchanged, in between: the rule the native-rate campaign
drove Drive-JEPA under (57.60 against 31.71 at every tick). The client imports
carla and the leaderboard; they are stood in for as in test_route_xy.py.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive.tests.test_route_xy import (
    agent_module,  # noqa: F401 -- the fixture
)


class _Node:
    """A policy node that plans 1 m further ahead on every call, so a held plan shows."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, url, json, timeout):
        self.calls += 1
        reach = float(self.calls)
        plan = [[reach + i, 0.0] for i in range(6)]
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"trajectory": plan})


def _client(agent_module, hz: float):  # noqa: F811 -- the fixture's name
    agent_module.INFER_HZ = hz
    agent_module.carla.VehicleControl = types.SimpleNamespace
    agent = object.__new__(agent_module.PaddockB2DAgent)
    agent.__dict__.update(
        _prev_speed=0.0,
        _prev_us=None,
        _plan=None,
        _native=None,
        _plan_us=None,
        _http=_Node(),
        _metric_path=None,
        _log=None,
        _t=0,
        _cam2img={},
        _cam2ego={},
    )
    steered: list[float] = []

    def control(wps, speed):
        steered.append(float(wps[0][0]))
        return 0.0, 0.3, 0.0, 5.0

    agent._tracker = types.SimpleNamespace(control=control)
    agent._ego2world = lambda: np.eye(4)
    agent._command = lambda: 2
    agent._nav = lambda: (None, None)
    agent._route_xy = lambda: None
    agent._hero_actor = lambda: None
    return agent, steered


@pytest.mark.parametrize("hz, calls", [(2.0, 3), (0.0, 21)])
def test_the_client_asks_at_the_models_rate_and_holds_the_plan_between(agent_module, hz, calls) -> None:  # noqa: F811
    agent, steered = _client(agent_module, hz)
    for frame in range(21):  # 1 s of a 20 Hz simulation, frames 0..20
        agent.run_step({"SPEED": (frame, {"speed": 5.0})}, frame * 0.05)
    assert agent._http.calls == calls
    if hz:
        # Asked at frames 0, 10 and 20; every frame between follows the plan last returned.
        assert steered == [1.0] * 10 + [2.0] * 10 + [3.0]
