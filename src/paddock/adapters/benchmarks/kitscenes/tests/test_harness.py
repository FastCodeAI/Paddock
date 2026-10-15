"""The KITScenes harness: the camera ring, where it is found and how it is checked, and the route hint.

No dataset. A synthetic KITScenes with the real file layout -- `calib.json`,
`poses.txt`, one directory of frame files per ring camera -- is written under
tmp_path, across two tiers. The frame files are empty: the harness names a
frame, it never opens one, so nothing here needs a pixel.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.kitscenes import harness as H
from paddock.adapters.benchmarks.kitscenes.harness import BENCHMARK, POD_HEIGHT_M, RING
from paddock.config import SiteConfig
from paddock.policy.protocol import CAMERAS, Command, Observation
from paddock.registry import Registry
from paddock.results.ingest.kitscenes import STRAIGHT_COMMAND

SCENE = "0a1b2c3d-0000-4000-8000-000000000001"
N_FRAMES = 100
KEYFRAME = 50
WIDTH, HEIGHT = 3504, 2272
#: Where each camera sits in the rig reference, in the dataset's calibration:
#: the roof pod's origin is 0.183 m above the ring.
POD_Z = -0.183
#: Heading of each ring camera, degrees left of forward.
HEADING_DEG = {
    "camera_ring_front": 0.0,
    "camera_ring_front_left": 60.0,
    "camera_ring_front_right": -60.0,
    "camera_ring_rear": 180.0,
    "camera_ring_rear_left": 120.0,
    "camera_ring_rear_right": -120.0,
}
REAR = tuple(sensor for sensor in RING if sensor.startswith("camera_ring_rear"))
#: The rig's other forward camera, and where the real calibration puts it in
#: the rig reference: 0.2 m further forward than the ring's front, and 0.2 m lower.
BASE = "camera_base_front_center"
BASE_XYZ = (0.421, 0.011, -0.384)


def _t_to_reference(heading_deg: float) -> np.ndarray:
    """Camera (OpenCV optical: x right, y down, z forward) -> rig (x forward, y left, z up).

    The convention the dataset's `T_to_reference` follows; the doc records the
    projection check that established it on the real calibration.
    """
    theta = np.radians(heading_deg)
    forward = np.array([np.cos(theta), np.sin(theta), 0.0])
    left = np.array([-np.sin(theta), np.cos(theta), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2] = -left, -up, forward
    t[:3, 3] = [0.2 * forward[0], 0.2 * forward[1], POD_Z]
    return t


def _calib() -> dict:
    """Seven distinct pinholes, the ring's and the base camera's.

    A shared or guessed value must be caught, not averaged in.
    """
    calib: dict = {"reference": "base_frame"}
    for i, sensor in enumerate(RING):
        calib[sensor] = {
            "T_to_reference": _t_to_reference(HEADING_DEG[sensor]).tolist(),
            "camera_model": "pinhole",
            "resolution": {"width": WIDTH, "height": HEIGHT},
            "intrinsics": {
                "focal_length": 1841.0 + i,
                "principal_point_u": 1740.0 + i,
                "principal_point_v": 1139.0,
            },
        }
    base = _t_to_reference(0.0)
    base[:3, 3] = BASE_XYZ
    calib[BASE] = {
        "T_to_reference": base.tolist(),
        "camera_model": "pinhole",
        "resolution": {"width": 5856, "height": 3104},
        "intrinsics": {"focal_length": 3022.0, "principal_point_u": 2924.0, "principal_point_v": 1545.0},
    }
    return calib


def _write_scene(root: Path, sensors, split: str = "val") -> Path:
    scene = root / split / SCENE
    (scene / "calibration").mkdir(parents=True)
    (scene / "calibration" / "calib.json").write_text(json.dumps(_calib()))
    # TUM poses at 10 Hz, straight ahead at 5 m/s.
    (scene / "poses.txt").write_text(
        "".join(f"{i / 10:.3f} {i * 0.5:.3f} 0.0 0.0 0.0 0.0 0.0 1.0\n" for i in range(N_FRAMES))
    )
    for sensor in sensors:
        (scene / sensor).mkdir()
        for i in range(N_FRAMES):
            (scene / sensor / f"{i:010d}.jpg").touch()
    return scene


def _site(
    tmp_path: Path, monkeypatch, tiers: list[Path], settings: str = "", row: dict | None = None
) -> SiteConfig:
    """A throwaway site config, written and loaded the way a real one is.

    *settings* are extra `[benchmarks.kitscenes]` lines; *row* extra fields of
    the one sample-list row.
    """
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir(exist_ok=True)
    sample_list = tmp_path / "sample_list.pkl"
    sample_list.write_bytes(
        pickle.dumps([{"scene_id": SCENE, "keyframe_index": KEYFRAME, "split": "val", **(row or {})}])
    )
    config = tmp_path / "site.toml"
    config.write_text(
        'site = "test-site"\n'
        "[roots]\n"
        f'repos = "{tmp_path / "repos"}"\n'
        f'runs = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n'
        "[datasets.kitscenes]\n"
        f"tiers = [{', '.join(f'{str(t)!r}' for t in tiers)}]\n"
        "[benchmarks.kitscenes]\n"
        f'sample_list = "{sample_list}"\n' + settings
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


def _benchmark(tmp_path: Path, monkeypatch, tiers: list[Path], **site):
    return BENCHMARK(Registry.load().benchmarks["kitscenes"], _site(tmp_path, monkeypatch, tiers, **site))


def _scored_observation(bench) -> Observation:
    (sample,) = list(bench.samples())
    return sample.observations[-1]


# ---------------------------------------------------------------------------
# What every observation carries
# ---------------------------------------------------------------------------


def test_the_ring_names_the_protocols_cameras_in_its_order() -> None:
    """Adapters select cameras by the nuScenes names; the ring must speak them."""
    assert tuple(RING.values()) == CAMERAS


def test_every_ring_camera_is_supplied_with_its_own_calibration(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    scored = _scored_observation(_benchmark(tmp_path, monkeypatch, [archive]))

    assert set(scored.image_paths) == set(CAMERAS)
    assert set(scored.camera2image) == set(CAMERAS) == set(scored.camera2ego)
    assert scored.images == {}, "paths, not pixels: the node opens what its policy reads"
    for i, (sensor, camera) in enumerate(RING.items()):
        assert scored.image_paths[camera] == str(archive / "val" / SCENE / sensor / f"{KEYFRAME:010d}.jpg")
        assert scored.camera2image[camera][0, 0] == 1841.0 + i
        assert scored.camera2image[camera][0, 2] == 1740.0 + i
        expected = _t_to_reference(HEADING_DEG[sensor])
        assert np.allclose(scored.camera2ego[camera][:3, :3], expected[:3, :3])
        assert np.allclose(scored.camera2ego[camera][:2, 3], expected[:2, 3])


def test_the_ground_offset_lifts_every_camera_alike(tmp_path, monkeypatch) -> None:
    """The ring shares the roof pod, so the road-level correction is the rig's."""
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    scored = _scored_observation(_benchmark(tmp_path, monkeypatch, [archive]))
    for camera in CAMERAS:
        assert scored.camera_height(camera) == pytest.approx(POD_Z + POD_HEIGHT_M)


def test_the_lidar_frame_is_the_ego_frame(tmp_path, monkeypatch) -> None:
    """KITScenes calibrates against lidar_top itself, so lidar2ego is the identity.

    Sent explicitly so a model that projects through the lidar frame need not
    assume this benchmark's convention.
    """
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    scored = _scored_observation(_benchmark(tmp_path, monkeypatch, [archive]))
    np.testing.assert_array_equal(scored.require_lidar2ego(), np.eye(4))
    # And it survives the trip to a node.
    np.testing.assert_array_equal(Observation.from_wire(scored.to_wire()).require_lidar2ego(), np.eye(4))


def test_the_history_carries_the_ring_at_every_frame(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    (sample,) = list(_benchmark(tmp_path, monkeypatch, [archive]).samples())

    assert len(sample.observations) == H.HISTORY_FRAMES + 1
    for offset, obs in enumerate(sample.observations):
        index = KEYFRAME - H.HISTORY_FRAMES + offset
        assert [Path(p).name for p in obs.image_paths.values()] == [f"{index:010d}.jpg"] * len(CAMERAS)
        assert obs.timestamp_us == int(index / H.SRC_HZ * 1e6)


def test_the_calibration_frame_puts_the_road_ahead_in_the_front_image(tmp_path, monkeypatch) -> None:
    """The convention the harness passes through, made concrete.

    A ground point 10 m ahead projects into the lower middle of the front
    image and sits behind the rear camera; 10 m behind does the reverse; a
    side camera, looking 60 deg out, sees its side's point 30 deg off-axis, in
    its inner half. Mirrors the check the doc records on the real calibration,
    so a wrong axis reading would fail here before it could be argued about.
    """
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    scored = _scored_observation(_benchmark(tmp_path, monkeypatch, [archive]))

    def project(camera: str, point: tuple[float, float, float]) -> tuple[float, float] | None:
        in_camera = np.linalg.inv(scored.camera2ego[camera]) @ np.array([*point, 1.0])
        if in_camera[2] <= 0:
            return None
        u, v, w = scored.camera2image[camera] @ in_camera[:3]
        return u / w / WIDTH, v / w / HEIGHT

    ahead, behind, left, right = (10.0, 0.0, 0.0), (-10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, -10.0, 0.0)
    for camera, point in (("CAM_FRONT", ahead), ("CAM_BACK", behind)):
        u, v = project(camera, point)
        assert 0.4 < u < 0.6, (camera, u)
        assert 0.6 < v < 0.75, (camera, v)  # 2.0 m up, 10 m out: 11 deg below the axis
    for camera, point, half in (("CAM_FRONT_LEFT", left, "left"), ("CAM_BACK_RIGHT", right, "left")):
        u, v = project(camera, point)
        assert (0.0 < u < 0.5) if half == "left" else (0.5 < u < 1.0), (camera, u)
        assert 0.6 < v < 0.75, (camera, v)
    assert project("CAM_BACK", ahead) is None
    assert project("CAM_FRONT", behind) is None
    assert project("CAM_FRONT_LEFT", right) is None


def test_the_run_record_says_which_ring_cameras_the_model_was_given(tmp_path, monkeypatch) -> None:
    """A ring run must not share a run id with a front-only one of the same model.

    `config.cameras` is what the model declares and did not change; this is
    what the harness handed it, so the two records differ in the hash and say
    why. The benchmark's own settings stay the ground offset alone.
    """
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive])

    assert bench.run_config() == {"ground_offset_m": POD_HEIGHT_M}
    comma_three = SimpleNamespace(cameras=("CAM_OP_ROAD", "CAM_OP_WIDE", "CAM_FRONT"))
    assert bench.model_run_config(comma_three) == {"cameras_supplied": ["CAM_FRONT"]}
    six = SimpleNamespace(cameras=tuple(reversed(CAMERAS)))
    assert bench.model_run_config(six) == {"cameras_supplied": list(CAMERAS)}, (
        "the protocol's order, not the model's"
    )


# ---------------------------------------------------------------------------
# The route hint
# ---------------------------------------------------------------------------


def test_without_a_route_hint_every_frame_is_straight(tmp_path, monkeypatch) -> None:
    """The default ignores the list's command, so every earlier run stands as it was."""
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive], row={"driving_command": [1, 0, 0, 0]})

    (sample,) = list(bench.samples())
    assert {obs.command for obs in sample.observations} == {Command.STRAIGHT}
    assert bench.run_config() == {"ground_offset_m": POD_HEIGHT_M}, "no new key, so no new run id"


@pytest.mark.parametrize(
    ("one_hot", "expected"),
    [
        ([1, 0, 0, 0], Command.LEFT),
        (STRAIGHT_COMMAND, Command.STRAIGHT),
        ([0, 0, 1, 0], Command.RIGHT),
        ([0, 0, 0, 1], Command.STRAIGHT),  # unknown: no hint
        ([0, 0, 0, 0], Command.STRAIGHT),
        ([1, 0, 1, 0], Command.STRAIGHT),
    ],
)
def test_a_route_hint_reaches_the_keyframe_alone(tmp_path, monkeypatch, one_hot, expected) -> None:
    """The list's one-hot is read in ingest's order, and only the scored frame carries it."""
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(
        tmp_path,
        monkeypatch,
        [archive],
        settings='command_source = "gt"\n',
        row={"driving_command": np.asarray(one_hot)},
    )

    (sample,) = list(bench.samples())
    *history, scored = sample.observations
    assert scored.command == expected
    assert {obs.command for obs in history} == {Command.STRAIGHT}
    # And it survives the trip to a node.
    assert Observation.from_wire(scored.to_wire()).command == expected


def test_a_route_hint_is_a_different_measurement(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive], settings='command_source = "gt"\n')

    assert bench.run_config() == {"ground_offset_m": POD_HEIGHT_M, "command_source": "gt"}


def test_a_route_hint_is_recorded_and_ranked(tmp_path, monkeypatch) -> None:
    """`gt` is the route signal ranked rows here and every nuScenes model get: recorded, not invalid."""
    archive = tmp_path / "archive"
    _write_scene(archive, [*RING, BASE])
    hinted = _benchmark(tmp_path, monkeypatch, [archive], settings='command_source = "gt"\n')
    assert hinted.invalid_reasons() == [], "a navigation command is an input, not a look at the scorer"
    assert hinted.run_config()["command_source"] == "gt", "and the record says the model was given it"

    base = _benchmark(tmp_path, monkeypatch, [archive], settings='front_camera = "base"\n')
    assert base.invalid_reasons() == [], "another camera is another input, not a look at the answer"


def test_preflight_names_an_unknown_command_source(tmp_path, monkeypatch) -> None:
    """A typo must not quietly run as `straight` and be recorded as something else."""
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive], settings='command_source = "sample_list"\n')

    (problem,) = bench.preflight()
    assert "command_source = 'sample_list'" in problem
    assert "straight, gt" in problem


# ---------------------------------------------------------------------------
# The front camera
# ---------------------------------------------------------------------------


def test_the_base_camera_stands_in_for_the_ring_front(tmp_path, monkeypatch) -> None:
    """Its frames, its pinhole and its mounting, at every frame; the other five stay the ring's."""
    archive = tmp_path / "archive"
    _write_scene(archive, [*RING, BASE])
    bench = _benchmark(tmp_path, monkeypatch, [archive], settings='front_camera = "base"\n')

    assert bench.preflight() == []
    (sample,) = list(bench.samples())
    assert {Path(obs.image_paths["CAM_FRONT"]).parent.name for obs in sample.observations} == {BASE}
    scored = sample.observations[-1]
    assert scored.camera2image["CAM_FRONT"][0, 0] == scored.camera2image["CAM_FRONT"][1, 1] == 3022.0
    assert scored.camera2image["CAM_FRONT"][:2, 2].tolist() == [2924.0, 1545.0]
    assert np.allclose(scored.camera2ego["CAM_FRONT"][:2, 3], BASE_XYZ[:2])
    assert scored.camera_height("CAM_FRONT") == pytest.approx(BASE_XYZ[2] + POD_HEIGHT_M), (
        "the rig's offset too"
    )
    for i, (sensor, camera) in enumerate(RING.items()):
        if camera != "CAM_FRONT":
            assert Path(scored.image_paths[camera]).parent.name == sensor
            assert scored.camera2image[camera][0, 0] == 1841.0 + i


def test_another_front_camera_is_a_different_measurement(tmp_path, monkeypatch) -> None:
    """Hashed into the run id when given; `ring`, the default, adds nothing even said aloud."""
    archive = tmp_path / "archive"
    _write_scene(archive, [*RING, BASE])
    base = _benchmark(tmp_path, monkeypatch, [archive], settings='front_camera = "base"\n')

    assert base.run_config() == {"ground_offset_m": POD_HEIGHT_M, "front_camera": "base"}
    comma_three = SimpleNamespace(cameras=("CAM_OP_ROAD", "CAM_OP_WIDE", "CAM_FRONT"))
    assert base.model_run_config(comma_three) == {"cameras_supplied": ["CAM_FRONT"]}, "the protocol's name"
    ring = _benchmark(tmp_path, monkeypatch, [archive], settings='front_camera = "ring"\n')
    assert ring.run_config() == {"ground_offset_m": POD_HEIGHT_M}, "so every earlier run keeps its id"


def test_preflight_names_an_unknown_front_camera(tmp_path, monkeypatch) -> None:
    """A typo must not quietly run as `ring` and be recorded as something else."""
    archive = tmp_path / "archive"
    _write_scene(archive, [*RING, BASE])
    bench = _benchmark(tmp_path, monkeypatch, [archive], settings='front_camera = "wide"\n')

    (problem,) = bench.preflight()
    assert "front_camera = 'wide'" in problem
    assert "ring, base" in problem


def test_preflight_checks_the_front_camera_that_will_be_read(tmp_path, monkeypatch) -> None:
    """The base camera is checked in the front's place, and the ring's front is then not needed."""
    ring_only, base_only = tmp_path / "ring_only", tmp_path / "base_only"
    _write_scene(ring_only, RING)
    _write_scene(base_only, [BASE, *(sensor for sensor in RING if sensor != "camera_ring_front")])

    assert _benchmark(tmp_path, monkeypatch, [ring_only]).preflight() == []
    (problem,) = _benchmark(
        tmp_path, monkeypatch, [ring_only], settings='front_camera = "base"\n'
    ).preflight()
    assert "1 of 1 listed scenes lack a ring camera on every configured tier" in problem
    assert f"{SCENE}: {BASE}" in problem
    assert (
        _benchmark(tmp_path, monkeypatch, [base_only], settings='front_camera = "base"\n').preflight() == []
    )


# ---------------------------------------------------------------------------
# What a single-camera node sees of it
# ---------------------------------------------------------------------------


def test_the_node_keeps_only_the_cameras_its_policy_declares(tmp_path, monkeypatch) -> None:
    """Why six on the wire changes nothing for openpilot or VisionPilot.

    `Observation.from_wire` is the protocol's, not this harness's; it is
    exercised here because the claim that single-camera models are numerically
    unchanged rests on exactly this filtering.
    """
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    scored = _scored_observation(_benchmark(tmp_path, monkeypatch, [archive]))
    wire = scored.to_wire()

    front_only = Observation.from_wire(wire, cameras=("CAM_FRONT",))
    assert set(front_only.image_paths) == {"CAM_FRONT"}
    assert front_only.image_paths["CAM_FRONT"] == scored.image_paths["CAM_FRONT"]
    assert np.array_equal(front_only.camera2image["CAM_FRONT"], scored.camera2image["CAM_FRONT"])
    assert np.array_equal(front_only.camera2ego["CAM_FRONT"], scored.camera2ego["CAM_FRONT"])

    everything = Observation.from_wire(wire, cameras=CAMERAS)
    assert set(everything.image_paths) == set(CAMERAS)


# ---------------------------------------------------------------------------
# Where the frames come from, and when the harness refuses
# ---------------------------------------------------------------------------


def test_a_camera_the_fast_tier_lacks_is_read_from_the_next(tmp_path, monkeypatch) -> None:
    """A working set pruned to the front camera fronts an archive holding the ring."""
    fast, archive = tmp_path / "fast", tmp_path / "archive"
    _write_scene(fast, ["camera_ring_front", "camera_ring_front_left"])
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [fast, archive])

    assert bench.preflight() == []
    scored = _scored_observation(bench)
    paths = {camera: Path(path) for camera, path in scored.image_paths.items()}
    assert paths["CAM_FRONT"].is_relative_to(fast)
    assert paths["CAM_FRONT_LEFT"].is_relative_to(fast)
    for camera in ("CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"):
        assert paths[camera].is_relative_to(archive), camera
    assert all(path.exists() for path in paths.values())


def test_preflight_names_a_scene_whose_ring_no_tier_completes(tmp_path, monkeypatch) -> None:
    """Refuse rather than score a six-camera model on three; the harness cannot
    see which model is paired, so it cannot know the gap is harmless."""
    fast = tmp_path / "fast"
    _write_scene(fast, [sensor for sensor in RING if sensor not in REAR])
    bench = _benchmark(tmp_path, monkeypatch, [fast])

    (problem,) = bench.preflight()
    assert "1 of 1 listed scenes lack a ring camera on every configured tier" in problem
    assert f"{SCENE}: {', '.join(REAR)}" in problem
    with pytest.raises(FileNotFoundError, match="camera_ring_rear"):
        list(bench.samples())


def test_preflight_names_a_camera_with_a_frame_for_every_pose_but_one(tmp_path, monkeypatch) -> None:
    """One frame per pose is the synchronisation the harness relies on."""
    archive = tmp_path / "archive"
    scene = _write_scene(archive, RING)
    (scene / "camera_ring_rear_left" / f"{N_FRAMES - 1:010d}.jpg").unlink()
    bench = _benchmark(tmp_path, monkeypatch, [archive])

    (problem,) = bench.preflight()
    assert "one frame per pose" in problem
    assert f"{SCENE}: camera_ring_rear_left has {N_FRAMES - 1} frames for {N_FRAMES} poses" in problem


def test_preflight_still_reports_the_missing_sample_list_first(tmp_path, monkeypatch) -> None:
    """No list, no scenes to walk: the ring check must not mask the older message."""
    archive = tmp_path / "archive"
    _write_scene(archive, RING)
    bench = _benchmark(tmp_path, monkeypatch, [archive])
    (tmp_path / "sample_list.pkl").unlink()

    (problem,) = bench.preflight()
    assert "sample_list does not exist" in problem
