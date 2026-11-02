"""How often the CARLA client asks the node for a plan (`PADDOCK_INFER_HZ`).

At 2 Hz on Bench2Drive's 20 Hz simulation the client asks once per 10 frames and
follows the last plan, unchanged, in between: the rule the native-rate campaign
drove Drive-JEPA under (57.60 against 31.71 at every tick). The client imports
carla and the leaderboard; conftest.py stands in for them (`agent_module`).
"""

from __future__ import annotations

import json
import math
import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive import recorder


class _Node:
    """A policy node that plans 1 m further ahead on every call, so a held plan shows."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, url, json, timeout):
        self.calls += 1
        reach = float(self.calls)
        plan = [[reach + i, 0.0] for i in range(6)]
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"trajectory": plan})


def _client(agent_module, hz: float):
    agent_module.INFER_HZ = hz
    agent_module.carla.VehicleControl = types.SimpleNamespace
    agent = object.__new__(agent_module.PaddockB2DAgent)
    agent.__dict__.update(
        _prev_speed=0.0,
        _prev_us=None,
        _plan=None,
        _native=None,
        _payload=None,
        _plan_us=None,
        _http=_Node(),
        _metric_path=None,
        _log=None,
        _t=0,
        _cam2img={},
        _cam2ego={},
        # No analysis record and no early stop on this route.
        _ticks=None,
        _record_failed=False,
        _early_stop=None,
        _stop_requested=False,
    )
    steered: list[float] = []

    def control(wps, speed):
        steered.append(float(wps[0][0]))
        return 0.0, 0.3, 0.0, 5.0

    agent._tracker = types.SimpleNamespace(control=control)
    agent._ego2world = lambda: np.eye(4)
    agent._command = lambda: 2
    agent._nav = lambda: (None, None)
    agent._lane_change = lambda: (None, None)
    agent._route_xy = lambda: None
    agent._hero_actor = lambda: None
    return agent, steered


@pytest.mark.parametrize("hz, calls", [(2.0, 3), (0.0, 21)])
def test_the_client_asks_at_the_models_rate_and_holds_the_plan_between(agent_module, hz, calls) -> None:
    agent, steered = _client(agent_module, hz)
    for frame in range(21):  # 1 s of a 20 Hz simulation, frames 0..20
        agent.run_step({"SPEED": (frame, {"speed": 5.0})}, frame * 0.05)
    assert agent._http.calls == calls
    if hz:
        # Asked at frames 0, 10 and 20; every frame between follows the plan last returned.
        assert steered == [1.0] * 10 + [2.0] * 10 + [3.0]


class _V3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _V3()
        self.rotation = rotation or types.SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0)

    def get_matrix(self):
        m = np.eye(4)
        m[:3, 3] = self.location.x, self.location.y, self.location.z
        return m.tolist()


class _Hero:
    """The ego, rolling along +x at 5 m/s; no lights, no other actors, no map lane."""

    id, type_id, attributes = 1, "vehicle.hero", {"role_name": "hero"}

    def __init__(self) -> None:
        self.x = 0.0

    def get_transform(self):
        return _Transform(_V3(self.x))

    def get_location(self):
        return _V3(self.x)

    def get_velocity(self):
        return _V3(5.0)

    def get_acceleration(self):
        return _V3()

    def get_angular_velocity(self):
        return _V3()

    def is_at_traffic_light(self):
        return False

    def get_traffic_light_state(self):
        return "Green"

    def get_speed_limit(self):
        return 50.0


class _Actors(list):
    def filter(self, pattern):
        return _Actors()


class _RecordingNode(_Node):
    """`_Node`, that answers `/reset` too and counts only the plans asked for. Never a real socket:
    other runs' nodes may be listening on this machine."""

    def post(self, url, json, timeout):
        if not url.endswith("/infer"):
            return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
        return super().post(url, json, timeout)

    def close(self) -> None:
        pass


def test_a_tick_that_does_not_ask_sends_nothing_and_its_record_says_so(
    load_agent, tmp_path, monkeypatch
) -> None:
    """At a declared 10 Hz with the analysis record and the inputs on, the client asks on every second
    tick, packs frames and saves inputs only then, and records every tick: the ones between say they
    did not ask and carry the plan they follow."""
    route = tmp_path / "route"
    agent_module = load_agent(
        PADDOCK_NODE_URL="http://node.invalid",
        PADDOCK_CAMERAS="CAM_FRONT",
        PADDOCK_INFER_HZ="10",
        PADDOCK_RECORD_DIR=str(route),
        PADDOCK_RECORD_INPUTS="1",
        PADDOCK_METRICS_DIR=str(tmp_path / "metrics"),
    )
    hero, node, packed = _Hero(), _RecordingNode(), []
    world = types.SimpleNamespace(get_actors=lambda: _Actors())
    agent_module.carla.__dict__.update(
        Transform=_Transform,
        Location=_V3,
        Rotation=types.SimpleNamespace,
        VehicleControl=types.SimpleNamespace,
    )
    agent_module.CarlaDataProvider.__dict__.update(
        get_hero_actor=lambda: hero,
        get_world=lambda: world,
        get_map=lambda: types.SimpleNamespace(get_waypoint=lambda location: None),
    )
    monkeypatch.setattr(agent_module, "requests", types.SimpleNamespace(Session=lambda: node))
    # OpenCV is the [analysis] extra; the thumbnails and frames it writes are not what is tested.
    cv2 = types.SimpleNamespace(
        resize=lambda img, size, interpolation=None: img,
        imwrite=lambda *a: True,
        INTER_AREA=3,
        IMWRITE_JPEG_QUALITY=1,
    )
    monkeypatch.setattr(agent_module, "cv2", cv2)
    encode = agent_module.encode_image
    monkeypatch.setattr(agent_module, "encode_image", lambda rgb, **kw: packed.append(1) or encode(rgb, **kw))

    agent = object.__new__(agent_module.PaddockB2DAgent)
    agent._dense_plan = [
        (_Transform(_V3(float(i))), agent_module.RoadOption.LANEFOLLOW) for i in range(300)
    ]  # a straight 1 m route, as set_global_plan leaves it
    agent.setup("route1_rep0")
    for frame in range(7):
        hero.x = 0.25 * frame
        rgba = np.full((450, 800, 4), frame, np.uint8)
        agent.run_step({"SPEED": (frame, {"speed": 5.0}), "CAM_FRONT": (frame, rgba)}, 0.05 * frame)
    agent.destroy()

    ticks = [json.loads(line) for line in (route / "ticks.jsonl").read_text().splitlines()]
    assert not agent._record_failed and [r["t"] for r in ticks] == list(range(7))
    assert [r["asked"] for r in ticks] == [True, False] * 3 + [True]
    assert node.calls == 4 and len(packed) == 4, "frames are packed and sent only on the ticks that ask"
    saved = sorted(p.name for p in (route / recorder.INPUTS_DIR).glob("*.npz"))
    assert saved == ["00000.npz", "00002.npz", "00004.npz", "00006.npz"]
    for held in ticks[1::2]:
        assert held["infer_ms"] == 0.0 and held["plan"] == ticks[held["t"] - 1]["plan"]
        assert not recorder.tick_asked(held)
