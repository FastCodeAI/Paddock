"""The camera rigs (`rig.py`), and the agent rendering through the one it is given."""

from __future__ import annotations

import json
import math

import pytest

from paddock.adapters.benchmarks.bench2drive import rig

SIX = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def test_the_default_rig_is_the_one_the_agent_always_rendered() -> None:
    """Moving the rig out of agent.py moved no camera: a run on paddock's rig drives as before."""
    cams = rig.cameras(rig.DEFAULT)
    assert cams["CAM_FRONT"] == {"pose": [1.5, 0.0, 1.5, 0.0, 0.0, 0.0], "fov": 70.0, "size": [800, 450]}
    assert cams["CAM_FRONT_LEFT"]["pose"] == [1.0, -0.5, 1.5, 0.0, 0.0, -55.0]
    assert cams["CAM_BACK"] == {"pose": [-1.5, 0.0, 1.5, 0.0, 0.0, 180.0], "fov": 110.0, "size": [800, 450]}
    assert cams["CAM_BACK_RIGHT"]["pose"] == [0.0, 0.6, 1.5, 0.0, 0.0, 110.0]
    assert cams["CAM_OP_ROAD"] == {"pose": [1.0, 0.0, 1.3, 0.0, 0.0, 0.0], "fov": 40.0, "size": [1928, 1208]}
    assert cams["CAM_OP_WIDE"]["fov"] == 119.0
    assert {tuple(cams[c]["size"]) for c in SIX} == {(800, 450)}, "one size: feature builders np.stack them"


@pytest.mark.parametrize("name", sorted(rig.RIGS))
def test_every_rig_places_every_camera_within_the_leaderboards_radius(name: str) -> None:
    cams = rig.cameras(name)
    assert set(cams) == set(rig.cameras(rig.DEFAULT)), "a rig moves cameras; it never drops one"
    for cid, cam in cams.items():
        x, y, z = cam["pose"][:3]
        assert math.sqrt(x * x + y * y + z * z) <= rig.MAX_RADIUS_M, cid
        assert 0.0 < cam["fov"] < 180.0, cid
        width, height = cam["size"]
        assert width > 0 and height > 0, cid


def test_bench2drive_is_the_rig_its_dataset_was_recorded_with() -> None:
    """tools/data_collect.py at 7ec25d1: nuScenes' six cameras at nuScenes' 1600x900."""
    cams = rig.cameras("bench2drive")
    assert {tuple(cams[c]["size"]) for c in SIX} == {(1600, 900)}
    assert cams["CAM_FRONT"]["pose"] == [0.80, 0.0, 1.60, 0.0, 0.0, 0.0]
    assert cams["CAM_BACK"]["pose"] == [-2.0, 0.0, 1.60, 0.0, 0.0, 180.0] and cams["CAM_BACK"]["fov"] == 110.0
    paddock = rig.cameras(rig.DEFAULT)
    assert cams["CAM_OP_ROAD"] == paddock["CAM_OP_ROAD"], "what a rig does not list is paddock's"


def test_an_unknown_rig_names_the_rigs_there_are() -> None:
    with pytest.raises(KeyError, match="bench2drive, kitscenes, paddock"):
        rig.cameras("bench2driv")


def test_the_agent_renders_through_paddocks_rig_unless_given_another(load_agent, monkeypatch) -> None:
    monkeypatch.delenv("PADDOCK_CAMERA_RIG", raising=False)
    agent = load_agent(PADDOCK_CAMERAS="CAM_FRONT")
    (front,) = (s for s in object.__new__(agent.PaddockB2DAgent).sensors() if s["id"] == "CAM_FRONT")
    assert agent.CAMERA_RIG == rig.DEFAULT
    mount = (front["width"], front["height"], front["fov"], front["x"], front["z"])
    assert mount == (800, 450, 70.0, 1.5, 1.5)


def test_the_agent_renders_and_records_the_rig_it_is_given(load_agent, tmp_path) -> None:
    agent = load_agent(
        PADDOCK_CAMERAS=",".join(SIX), PADDOCK_CAMERA_RIG="bench2drive", PADDOCK_RECORD_DIR=str(tmp_path)
    )
    client = object.__new__(agent.PaddockB2DAgent)
    sensors = {s["id"]: s for s in client.sensors() if s["type"] == "sensor.camera.rgb"}
    assert set(sensors) == set(SIX)
    for cid, cam in rig.cameras("bench2drive").items():
        if cid in sensors:
            s = sensors[cid]
            assert [s["x"], s["y"], s["z"], s["roll"], s["pitch"], s["yaw"]] == cam["pose"], cid
            assert (s["fov"], [s["width"], s["height"]]) == (cam["fov"], cam["size"]), cid

    client._write_rig_record()
    record = json.loads((tmp_path / rig.RECORD).read_text())
    assert record == {"name": "bench2drive", "cameras": {c: rig.cameras("bench2drive")[c] for c in SIX}}


def test_an_unknown_rig_fails_the_agent_before_it_drives(load_agent) -> None:
    with pytest.raises(KeyError, match="no camera rig 'nope'"):
        load_agent(PADDOCK_CAMERA_RIG="nope")
