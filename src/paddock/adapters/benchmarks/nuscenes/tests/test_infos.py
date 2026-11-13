"""The datasets' own inputs (`infos`): what UniAD's and VAD's converters and datasets give a keyframe.

Pinned here, on synthetic data: each of the converters' computations (the CAN-bus message, each camera at its
own ego pose, the stored quaternion, each repo's command), that the tables are read into them as the
converters read them, that only a model that asks gets them and its run says so, and that every other
model's observations are byte for byte what they were before. On real data the same is held to the repos'
own code in `adapters/models/uniad/tests/test_scene_golden.py`.
"""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes import harness as NH
from paddock.adapters.benchmarks.nuscenes import infos as I
from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.config import ConfigError
from paddock.policy.protocol import Command, Observation

#: nuScenes LIDAR_TOP's mount (x right, y forward, ~1 m ahead of the rear axle, 1.84 m up).
LIDAR_CALIBRATION = ((0.985793, 0.0, 1.84019), (0.706749, -0.015301, 0.017397, -0.707085))
LIDAR2EGO = L.pose_matrix(*LIDAR_CALIBRATION)


def _message(t: int, x: float) -> dict:
    """A CAN-bus pose message in the expansion's own key order."""
    return {
        "accel": [x, 0.1, 9.8],
        "orientation": [1.0, 0.0, 0.0, 0.0],
        "pos": [x, 2 * x, 0.0],
        "rotation_rate": [0.0, 0.0, x / 100],
        "utime": t,
        "vel": [x + 5, 0.0, 0.0],
    }


def _yaw_quaternion(yaw: float, sign: float = 1.0) -> list[float]:
    return [sign * math.cos(yaw / 2), 0.0, 0.0, sign * math.sin(yaw / 2)]


def _scene(name: str, n: int, yaw_rate: float = 0.0, speed: float = 5.0) -> tuple[list[L.Keyframe], dict]:
    """n keyframes 0.5 s apart, turning at `yaw_rate` rad/keyframe, with cameras and their own ego poses."""
    frames, infos, x, y, yaw = [], {}, 100.0, 200.0, 0.3
    for i in range(n):
        sign = -1.0 if i % 3 == 0 else 1.0  # nuScenes stores either sign
        q = _yaw_quaternion(yaw, sign)
        pose = L.pose_matrix([x, y, 0.5], q)
        cameras, camera_poses, calibrations = {}, {}, {}
        for k, cam in enumerate(L.CAMERAS):
            cameras[cam] = {
                "path": f"/data/samples/{cam}/{name}-{i}.jpg",
                "intrinsic": np.array([[1266.0, 0.0, 816.0], [0.0, 1266.0, 491.0], [0.0, 0.0, 1.0]]),
                "cam2ego": L.pose_matrix([1.0 + 0.1 * k, 0.2 * k, 1.5], _yaw_quaternion(k * 1.0 - 1.57)),
            }
            calibrations[cam] = ((1.0 + 0.1 * k, 0.2 * k, 1.5), tuple(_yaw_quaternion(k * 1.0 - 1.57)))
            # each camera fired a little before or after the lidar: its own ego pose is ahead or behind
            dt = (k - 2.5) * 0.008
            camera_poses[cam] = (
                (x + speed * dt * math.cos(yaw), y + speed * dt * math.sin(yaw), 0.5),
                tuple(_yaw_quaternion(yaw + yaw_rate * dt / 0.5)),
            )
        token = f"{name}-{i}"
        frames.append(
            L.Keyframe(
                token=token,
                scene=name,
                index_in_scene=i,
                timestamp_us=1_000_000 + i * 500_000,
                ego2world=pose,
                cameras=cameras,
                lidar2ego=LIDAR2EGO,
                canbus=np.arange(18, dtype=np.float64),
            )
        )
        infos[token] = I.KeyframeInfo(
            ego_translation=(x, y, 0.5),
            ego_rotation=tuple(q),
            can_bus=tuple(float(v) for v in 100 + np.arange(18)),
            camera_poses=camera_poses,
            lidar_calibration=LIDAR_CALIBRATION,
            camera_calibrations=calibrations,
        )
        x, y, yaw = x + speed * 0.5 * math.cos(yaw), y + speed * 0.5 * math.sin(yaw), yaw + yaw_rate
    return frames, infos


def _harness(monkeypatch, scenes, infos=None, **settings):
    harness = object.__new__(NH.NuScenesBenchmark)
    monkeypatch.setattr(harness, "setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(harness, "_index", lambda: scenes)
    if infos is None:

        def never():
            raise AssertionError("a model that did not ask must not load the datasets' inputs")

        monkeypatch.setattr(harness, "_infos", never)
    else:
        monkeypatch.setattr(harness, "_infos", lambda: infos)
    harness._boxes = {}
    harness._camera_history_cache = None
    return harness


def _model(name: str, inputs: str | None = None, history_s: float = 0.0):
    return SimpleNamespace(
        name=name, inputs={"nuscenes": inputs} if inputs else {}, history_s=history_s, min_input_hz=0.0
    )


# ---------------------------------------------------------------------------
# the converters' computations
# ---------------------------------------------------------------------------


def test_the_can_bus_takes_its_pose_from_the_last_message_and_its_motion_from_the_next():
    messages = [_message(100, 1.0), _message(200, 2.0), _message(300, 3.0)]
    at = I.converter_can_bus(messages, 200)  # a message exactly at the keyframe is "at or before"
    assert at[0:3].tolist() == [2.0, 4.0, 0.0] and at[3:7].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert at[7:10].tolist() == [3.0, 0.1, 9.8], "accel from the first message after the keyframe"
    assert at[10:13].tolist() == [0.0, 0.0, 0.03] and at[13:16].tolist() == [8.0, 0.0, 0.0]
    assert at[16:18].tolist() == [0.0, 0.0]
    between = I.converter_can_bus(messages, 250)
    assert between[0] == 2.0 and between[7] == 3.0
    before = I.converter_can_bus(messages, 50)  # the scene's CAN bus starts after its first keyframe
    assert before[0] == 1.0 and before[7] == 1.0
    after = I.converter_can_bus(messages, 400)  # none after it: the loop ends on the last message
    assert after[0] == 3.0 and after[7] == 3.0


def test_the_can_bus_follows_the_messages_key_order_and_refuses_a_message_it_cannot_lay_out():
    order = ("vel", "utime", "pos", "orientation", "accel", "rotation_rate")
    reordered = [{k: m[k] for k in order} for m in (_message(100, 1.0), _message(200, 2.0))]
    out = I.converter_can_bus(reordered, 150)
    assert out[7:10].tolist() == [7.0, 0.0, 0.0], "the converter extends in the message's own key order"
    with pytest.raises(ValueError, match="18"):
        I.converter_can_bus([dict(_message(100, 1.0), extra=[1.0])], 150)
    with pytest.raises(ValueError):
        I.converter_can_bus([], 150)


def test_each_camera_is_placed_where_it_was_when_it_fired():
    frames, infos = _scene("scene-0001", 3, yaw_rate=0.2, speed=12.0)
    frame, info = frames[1], infos[frames[1].token]
    for cam in L.CAMERAS:
        translation, rotation = info.camera_poses[cam]
        # camera -> its own ego -> world -> the keyframe's ego
        want = (
            np.linalg.inv(frame.ego2world)
            @ L.pose_matrix(translation, rotation)
            @ frame.cameras[cam]["cam2ego"]
        )
        got = I.camera_in_keyframe(frame, cam, info)
        np.testing.assert_allclose(got, want, atol=1e-9)
        assert not np.allclose(got[:3, 3], frame.cameras[cam]["cam2ego"][:3, 3], atol=1e-3), "it moved"
    # fired with the lidar: exactly the calibration
    same = dataclasses.replace(
        info, camera_poses={cam: (info.ego_translation, info.ego_rotation) for cam in L.CAMERAS}
    )
    for cam in L.CAMERAS:
        np.testing.assert_allclose(
            I.camera_in_keyframe(frame, cam, same), frame.cameras[cam]["cam2ego"], atol=1e-9
        )


def test_rotations_are_built_as_pyquaternion_builds_them():
    """The converters' rotations: the same matrix as the closed form to rounding, from q or -q, normalised."""
    q = [0.570994, -0.003045, 0.011553, 0.820877]  # a nuScenes ego pose, stored with |q| a hair off 1
    np.testing.assert_allclose(I.rotation_matrix(q), L.quat_to_rot(q), atol=1e-15)
    assert I.rotation_matrix(q).tolist() == I.rotation_matrix([-v for v in q]).tolist()
    np.testing.assert_allclose(I.rotation_matrix([2 * v for v in q]), I.rotation_matrix(q), atol=1e-15)
    assert I.rotation_matrix([1.0, 0.0, 0.0, 0.0]).tolist() == np.eye(3).tolist()


def test_the_can_bus_carries_the_quaternion_as_stored_and_the_heading_wrapped():
    from paddock.adapters.models.uniad import bev_planners as bp

    frames, infos = _scene("scene-0001", 3)
    frame, info = frames[0], infos[frames[0].token]
    assert info.ego_rotation[0] < 0, "this keyframe's quaternion is stored with w < 0"
    out = I.dataset_can_bus(frame, info)
    assert out[3:7].tolist() == list(info.ego_rotation)
    assert out[0:3].tolist() == frame.ego2world[:3, 3].tolist()
    assert out[7:16].tolist() == list(info.can_bus[7:16])
    assert 0.0 <= out[17] < 360.0 and out[16] == pytest.approx(math.radians(out[17]), abs=1e-12)
    # what the adapter makes of it: the stored sign reaches the model (w four times at 3:7)
    q = bp.source_quaternion(out, frame.ego2world)
    assert q is not None and q[0] < 0
    can_bus, gravity = bp.nuscenes_can_bus(frame.ego2world, out, q)
    assert can_bus[3:7].tolist() == [info.ego_rotation[0]] * 4 and not gravity


def _drift_right(lateral: float, heading: float) -> list[L.Keyframe]:
    """Seven keyframes whose ego ends `lateral` m to the left (negative: right) and turned by `heading`."""
    frames = []
    for i in range(7):
        s = i / 6
        pose = L.pose_matrix([10.0 * i, lateral * s * s, 0.0], _yaw_quaternion(heading * s))
        frames.append(L.Keyframe(f"k{i}", "scene-0001", i, i * 500_000, pose, lidar2ego=LIDAR2EGO))
    return frames


def test_the_command_is_each_repos_rule_on_the_lidars_pose_in_the_lidar_frame():
    # The ego ends 1.9 m right of where it started, the car turned 30 degrees right: the lidar, a metre ahead
    # of the rear axle, ends ~2.4 m right. The ego frame says straight; both datasets say right.
    frames = _drift_right(-1.9, -math.radians(30))
    ego = L.to_ego_xy(frames[0].ego2world, np.array([f.ego2world[:3, 3] for f in frames[1:]]))
    assert L.command_from_future(ego)[0] == int(Command.STRAIGHT)
    assert I.vad_offset(frames, 0) > 2.0 and I.uniad_offset(frames, 0) > 2.0
    assert I.dataset_command("vad", frames, 0) == I.dataset_command("uniad", frames, 0) == 0
    assert I.command("uniad", frames, 0) == (int(Command.RIGHT), "TURN_RIGHT", None)
    left = _drift_right(3.0, math.radians(20))
    assert I.command("vad", left, 0) == (int(Command.LEFT), "TURN_LEFT", None)
    # the scene's end: VAD holds the last pose, UniAD reads the last valid one; nothing ahead is straight
    assert I.vad_offset(left, 6) == pytest.approx(0.0, abs=1e-9) and I.uniad_offset(left, 6) is None
    assert I.dataset_command("uniad", left, 6) == I.dataset_command("vad", left, 6) == 2
    with pytest.raises(ValueError, match="convention"):
        I.dataset_command("drivejepa", left, 0)


def test_uniads_label_sits_under_the_lidar_and_is_cast_to_float32_as_its_boxes_are():
    frames = _drift_right(-1.0, -0.2)
    f32 = I.uniad_offset(frames, 0)
    exact = I.vad_offset(frames, 0)  # the lidar itself, in float64
    assert f32 is not None and abs(f32 - exact) < 0.05, "the 0.78 m drop barely moves x on flat ground"
    assert np.float32(f32) == f32


# ---------------------------------------------------------------------------
# from the tables
# ---------------------------------------------------------------------------


def _tables(tmp_path, with_can: bool = True):
    """A one-scene nuScenes root: two samples, the lidar and six cameras each at its own pose, a radar."""
    root = tmp_path / "nuscenes"
    (root / "v1.0-trainval").mkdir(parents=True)
    channels = ["LIDAR_TOP", *L.CAMERAS, "RADAR_FRONT"]
    sensor = [{"token": f"s-{c}", "channel": c, "modality": "x"} for c in channels]
    identity = {"translation": [0.0, 0.0, 1.0], "rotation": [1.0, 0, 0, 0], "camera_intrinsic": []}
    calibrated = [{"token": f"c-{c}", "sensor_token": f"s-{c}", **identity} for c in channels]
    samples, sample_data, poses = [], [], []
    for i in range(2):
        links = {"prev": f"t{i - 1}" if i else "", "next": "t1" if i == 0 else ""}
        samples.append({"token": f"t{i}", "timestamp": 1000 + 500 * i, "scene_token": "sc", **links})
        for k, c in enumerate(channels):
            rotation = [-1.0, 0.0, 0.0, 0.0] if k == 0 else [1.0, 0.0, 0.0, 0.0]
            poses.append({"token": f"p{i}{k}", "translation": [i + k / 10, 0.0, 0.0], "rotation": rotation})
            sample_data.append(
                {
                    "token": f"d{i}{k}",
                    "sample_token": f"t{i}",
                    "ego_pose_token": f"p{i}{k}",
                    "calibrated_sensor_token": f"c-{c}",
                    "is_key_frame": True,
                    "filename": f"samples/{c}/{i}.jpg",
                }
            )
        sweep = {"ego_pose_token": "missing", "calibrated_sensor_token": "c-CAM_FRONT", "is_key_frame": False}
        sample_data.append(
            {"token": f"sweep{i}", "sample_token": f"t{i}", "filename": "sweeps/x.jpg", **sweep}
        )
    scene = [{"token": "sc", "name": "scene-0001", "first_sample_token": "t0", "last_sample_token": "t1"}]
    tables = {
        "scene": scene,
        "sample": samples,
        "sensor": sensor,
        "calibrated_sensor": calibrated,
        "sample_data": sample_data,
        "ego_pose": poses,
    }
    for name, rows in tables.items():
        (root / "v1.0-trainval" / f"{name}.json").write_text(json.dumps(rows))
    (root / "can_bus").mkdir()
    if with_can:
        messages = [_message(900, 1.0), _message(1200, 2.0), _message(1400, 3.0), _message(1600, 4.0)]
        (root / "can_bus" / "scene-0001_pose.json").write_text(json.dumps(messages))
    return root


def test_the_tables_are_read_as_the_converters_read_them(tmp_path):
    root = _tables(tmp_path)
    infos = I.build(root, "v1.0-trainval", ["scene-0001"], log=lambda m: None)
    assert sorted(infos) == ["t0", "t1"]
    first = infos["t0"]
    assert first.ego_rotation == (-1.0, 0.0, 0.0, 0.0), "the stored sign"
    assert first.ego_translation == (0.0, 0.0, 0.0)
    assert sorted(first.camera_poses) == sorted(L.CAMERAS), "the six cameras, no radar"
    assert first.camera_poses["CAM_FRONT"][0] == (0.1, 0.0, 0.0), "each camera's own keyframe ego pose"
    assert (
        first.lidar_calibration == first.camera_calibrations["CAM_FRONT"] == ((0.0, 0.0, 1.0), (1.0, 0, 0, 0))
    )
    assert first.can_bus[0] == 1.0 and first.can_bus[7] == 2.0, "pos at or before 1000, motion after it"
    assert infos["t1"].can_bus[0] == 3.0 and infos["t1"].can_bus[7] == 4.0
    with pytest.raises(FileNotFoundError, match="CAN-bus"):
        I.build(
            _tables(tmp_path / "nocan", with_can=False), "v1.0-trainval", ["scene-0001"], log=lambda m: None
        )
    with pytest.raises(FileNotFoundError, match="absent"):
        I.build(root, "v1.0-trainval", ["scene-9999"], log=lambda m: None)


def test_the_infos_are_cached_under_their_own_key(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        I, "build", lambda root, version, scenes, log=print: calls.append(scenes) or {"n": len(calls)}
    )
    cache = tmp_path / "infos.pkl"
    assert I.load_or_build(tmp_path, "v1.0-trainval", ["b", "a"], cache) == {"n": 1}
    assert I.load_or_build(tmp_path, "v1.0-trainval", ["a", "b"], cache) == {"n": 1}, "served from the cache"
    assert I.load_or_build(tmp_path / "elsewhere", "v1.0-trainval", ["a", "b"], cache) == {"n": 2}
    assert calls[0] == ["a", "b"], "built in one order whatever order it was asked in"
    # the index's key is not the infos' key: one file can never be read as the other
    assert I._cache_key(tmp_path, "v", ["a"]) != L._cache_key(tmp_path, "v", ["a"])


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inputs", I.CONVENTIONS)
def test_a_model_that_asks_is_sent_its_datasets_inputs_and_the_run_says_so(monkeypatch, inputs):
    frames, infos = _scene("scene-0001", 12, yaw_rate=-0.15, speed=8.0)
    window = _harness(monkeypatch, [frames], infos)
    assert window.model_run_config(_model("uniad", inputs)) == {"inputs": inputs}
    sample = next(iter(window.samples()))
    scene = _harness(monkeypatch, [frames], infos, sequence="scene")
    assert scene.model_run_config(_model("uniad", inputs)) == {"inputs": inputs}
    sequence = next(iter(scene.sequences()))
    for j, obs in enumerate(sequence.observations):
        frame, info = frames[j], infos[frames[j].token]
        assert obs.canbus.tolist() == I.dataset_can_bus(frame, info).tolist()
        assert obs.canbus[3:7].tolist() == list(info.ego_rotation)
        for cam in L.CAMERAS:
            assert obs.camera2ego[cam].tolist() == I.camera_in_keyframe(frame, cam, info).tolist()
        cmd, maneuver, distance = I.command(inputs, frames, j)
        assert (int(obs.command), obs.nav_maneuver, obs.nav_distance_m) == (cmd, maneuver, distance)
        assert obs.ego2world is frame.ego2world and obs.lidar2ego is frame.lidar2ego
    for i, scored in sequence.scored:
        assert scored.native_id["command"] == int(sequence.observations[i].command)
    assert sample.native_id["command"] == int(sample.observations[-1].command)
    assert {int(o.command) for o in sequence.observations} != {int(Command.STRAIGHT)}, "the scene turns"


def _legacy_observation(frame, cmd, maneuver, distance) -> Observation:
    """`NuScenesBenchmark._observation` as it was before the datasets' inputs (5730097), verbatim."""
    return Observation(
        images={},
        image_paths={cam: rec["path"] for cam, rec in frame.cameras.items()},
        ego2world=frame.ego2world,
        canbus=frame.canbus,
        timestamp_us=frame.timestamp_us,
        command=Command(cmd),
        camera2image={cam: rec["intrinsic"] for cam, rec in frame.cameras.items()},
        camera2ego={cam: rec["cam2ego"] for cam, rec in frame.cameras.items()},
        lidar2ego=frame.lidar2ego,
        nav_maneuver=maneuver,
        nav_distance_m=distance,
    )


@pytest.mark.parametrize(
    "name, history_s", [("drivejepa", 0.0), ("autoe2e", 6.5), ("constant_velocity", 0.0)]
)
def test_every_other_model_is_sent_byte_for_byte_what_it_was_sent_before(monkeypatch, name, history_s):
    frames, _ = _scene("scene-0001", 16, yaw_rate=0.2, speed=9.0)
    harness = _harness(monkeypatch, [frames])  # its _infos raises: never loaded for this model
    config = harness.model_run_config(_model(name, history_s=history_s))
    assert config == ({} if history_s <= 2.0 else {"history_seconds": 6.5}), "the run id is what it was"
    wanted = harness._history_keyframes
    for sample in harness.samples():
        i = sample.native_id["index_in_scene"]
        future = np.array([f.ego2world[:3, 3] for f in frames[i + 1 : i + 7]])
        legacy = L.command_from_future(L.to_ego_xy(frames[i].ego2world, future))
        assert sample.native_id["command"] == legacy[0]
        window = frames[max(0, i - wanted) : i + 1]
        assert [json.dumps(o.to_wire(), sort_keys=True) for o in sample.observations] == [
            json.dumps(_legacy_observation(f, *legacy).to_wire(), sort_keys=True) for f in window
        ]


def test_an_unknown_convention_and_sweeps_are_refused_before_the_model_loads(monkeypatch):
    harness = _harness(monkeypatch, [])
    assert harness.input_problems(_model("uniad", "uniad")) == []
    assert any("conventions are uniad, vad" in p for p in harness.input_problems(_model("x", "navsim")))
    sweeps = _harness(monkeypatch, [], camera_history="CAM_FRONT")
    assert any("camera_history" in p for p in sweeps.input_problems(_model("uniad", "vad")))
    assert not any("info file" in p for p in sweeps.input_problems(_model("drivejepa")))


def test_uniad_and_vad_ask_for_their_own_datasets_and_no_other_model_asks_for_anything():
    from paddock.registry import Registry

    declared = {name: spec.inputs for name, spec in Registry.load().models.items() if spec.inputs}
    assert declared == {
        "uniad": {"nuscenes": "uniad"},
        "vad": {"nuscenes": "vad"},
    }


def test_a_malformed_inputs_block_is_refused_naming_the_file(tmp_path):
    from paddock.registry import load_models

    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "model.yaml").write_text("name: m\ninputs: [nuscenes]\n")
    with pytest.raises(ConfigError, match="inputs must map"):
        load_models(tmp_path)
