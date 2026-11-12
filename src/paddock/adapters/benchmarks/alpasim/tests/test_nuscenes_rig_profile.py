"""UniAD's and VAD's AlpaSim profile: AlpaSim's six cameras re-rendered as nuScenes' rig.

Both models are driven on `configs/driver/paddock_bridge_nuscenes_rig.yaml`. These tests hold
it to three things: the bridge flags agree with it (control step, warm-up, and the 500 ms
camera interval every replan lands on); its six cameras are the six the flags rename, each an
undistorted 1600 x 900 pinhole with a 1 us global shutter; and every intrinsic and pose is
nuScenes' own calibration of scene-0103, as the adapters' golden fixture recorded it from the
nuScenes tables.

No grpc, no AlpaSim, no model: runs in paddock's environment. What the bridge then sends the
node under these flags is checked in test_bridge_switches.py, in AlpaSim's.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import yaml

from paddock.adapters.benchmarks.alpasim import harness, launcher
from paddock.adapters.models.uniad.bev_planners import LOAD_SIZE, NUSC_CAMERAS, NUSC_LIDAR2EGO
from paddock.registry import Registry

MODELS = ("uniad", "vad")
DRIVER = "paddock_bridge_nuscenes_rig"

#: AlpaSim id -> the nuScenes camera it is re-rendered as (and the node's name for it).
RIG = {
    "camera_front_wide_120fov": "CAM_FRONT",
    "camera_cross_left_120fov": "CAM_FRONT_LEFT",
    "camera_cross_right_120fov": "CAM_FRONT_RIGHT",
    "camera_rear_left_70fov": "CAM_BACK_LEFT",
    "camera_front_tele_30fov": "CAM_BACK",
    "camera_rear_right_70fov": "CAM_BACK_RIGHT",
}

#: The adapters' golden fixture: scene-0103's calibration as nuScenes' tables hold it,
#: camera_intrinsic as stored and each calibrated_sensor pose as a 4x4.
GOLDEN = harness.MODELS_DIR / "uniad" / "tests" / "fixtures" / "golden_uniad.json"


def _config() -> dict:
    return yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{DRIVER}.yaml").read_text())


def _args(model: str) -> list[str]:
    return list(harness.profile_for(model).bridge_args)


def _flag(args: list[str], name: str) -> int:
    return int(args[args.index(name) + 1])


def _rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """AlpaSim's (x, y, z, w) quaternion -> 3x3, normalised first as AlpaSim's Pose does."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _extra() -> dict[str, dict]:
    return {c["logical_id"]: c for c in _config()["runtime"]["extra_cameras"]}


@pytest.mark.parametrize("model", MODELS)
def test_the_model_lists_alpasim_and_can_be_driven_there(model: str) -> None:
    """Its profile loads, its driver config and scene list exist, every camera it reads arrives."""
    spec = Registry.load().models[model]
    assert "alpasim" in spec.benchmarks
    assert harness.profile_problems(spec) == []
    profile = harness.profile_for(model)
    assert profile.driver == DRIVER
    assert profile.scenes_csv == harness.profile_for("drivejepa").scenes_csv
    assert profile.replan_log is True
    # Shared by both models, so in this benchmark's folder and in neither model's.
    in_benchmark, in_model = harness.driver_config_paths(model, DRIVER)
    assert in_benchmark.is_file() and not in_model.exists()
    assert harness.driver_searchpath(model, profile) is None


def test_uniad_and_vad_are_driven_alike() -> None:
    assert harness.profile_for("uniad") == harness.profile_for("vad")


@pytest.mark.parametrize("model", MODELS)
def test_the_six_renamed_cameras_are_the_six_the_model_reads(model: str) -> None:
    args = _args(model)
    renames = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--camera-name")
    assert renames == RIG
    assert sorted(renames.values()) == sorted(Registry.load().models[model].cameras) == sorted(NUSC_CAMERAS)
    assert "--placeholder" not in args and "--planner" not in args


@pytest.mark.parametrize("model", MODELS)
def test_the_bridge_flags_match_the_driver_config(model: str) -> None:
    """Control step and warm-up are the driver config's; every camera renders on the replan grid.

    Each camera's first frame closes at the first policy step (assert_zero_decision_delay)
    and the replan grid is anchored there, so with the camera interval equal to the replan
    interval and the warm-up a whole number of it, every replan's newest frame closed at the
    replan's own time (the driver config's header has the chain of reasons).
    """
    args = _args(model)
    simulation = _config()["runtime"]["simulation_config"]
    control = simulation["control_timestep_us"]
    assert _flag(args, "--frame-interval-us") == control == 50_000
    assert _flag(args, "--force-gt-us") == simulation["force_gt_duration_us"] == 3_000_000
    assert simulation["skip_driver_during_force_gt"] is False
    replan = _flag(args, "--replan-every-us")
    assert replan == 500_000 == round(1e6 / Registry.load().models[model].planning_hz)
    assert replan % control == 0
    assert simulation["force_gt_duration_us"] % replan == 0
    assert simulation["assert_zero_decision_delay"] is True
    for camera in simulation["cameras"]:
        assert camera["frame_interval_us"] == replan, camera["logical_id"]
        # Drawn at shutter open, labelled with the pose at shutter close: 1 us apart.
        assert camera["shutter_duration_us"] == 1, camera["logical_id"]
    # The route with its frame's time: the node derives the command from it (bev_planners.route_command).
    for flag in (
        "--pinhole-intrinsics",
        "--canbus-ego-frame",
        "--canbus-acceleration",
        "--send-route",
        "--send-route-time",
    ):
        assert flag in args
    # Padded frames carry no images; frames between replans are never stepped on; 0.5 s
    # is the models' spacing.
    for flag in ("--pad-ego-history", "--observe-between-replans", "--waypoint-dt", "--send-plan-dt"):
        assert flag not in args


def test_six_undistorted_global_pinholes_at_the_models_load_size() -> None:
    config = _config()
    extra = _extra()
    rendered = {c["logical_id"]: c for c in config["runtime"]["simulation_config"]["cameras"]}
    assert sorted(extra) == sorted(rendered) == sorted(RIG)
    for camera_id, camera in extra.items():
        assert camera["intrinsics"]["model"] == "opencv_pinhole", camera_id
        pinhole = camera["intrinsics"]["opencv_pinhole"]
        for key, count in (("radial", 6), ("tangential", 2), ("thin_prism", 4)):
            assert pinhole[key] == [0.0] * count, (camera_id, key)
        assert camera["shutter_type"] == "GLOBAL", camera_id
        height, width = camera["resolution_hw"]
        assert (rendered[camera_id]["height"], rendered[camera_id]["width"]) == (height, width)
        # The size every config loads at, so the adapter resizes nothing (bev_planners.to_load_size).
        assert (width, height) == LOAD_SIZE
        # Every pose is nuScenes', the front and cross cameras' included.
        assert "rig_to_camera" in camera, camera_id


def test_every_intrinsic_and_pose_is_nuscenes_calibration_of_scene_0103() -> None:
    """K exactly, and rig_to_camera to 1e-9: translation as stored, the (w, x, y, z)
    quaternion reordered to AlpaSim's (x, y, z, w)."""
    observations = json.loads(GOLDEN.read_text())["observations"]
    first = observations[0]
    # One calibration for the whole scene, and its lidar is the one the grid is laid out in.
    for later in observations[1:]:
        assert later["camera2image"] == first["camera2image"]
        assert later["camera2ego"] == first["camera2ego"]
    np.testing.assert_allclose(first["lidar2ego"], NUSC_LIDAR2EGO, atol=1e-12)
    extra = _extra()
    for camera_id, name in RIG.items():
        k = first["camera2image"][name]
        pinhole = extra[camera_id]["intrinsics"]["opencv_pinhole"]
        assert pinhole["focal_length"] == [k[0][0], k[1][1]], name
        assert pinhole["principal_point"] == [k[0][2], k[1][2]], name
        assert k[0][1] == 0.0 and k[2] == [0.0, 0.0, 1.0], name
        camera2ego = np.asarray(first["camera2ego"][name])
        pose = extra[camera_id]["rig_to_camera"]
        np.testing.assert_allclose(pose["translation_m"], camera2ego[:3, 3], rtol=0, atol=1e-9)
        assert math.sqrt(sum(v * v for v in pose["rotation_xyzw"])) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(_rotation(*pose["rotation_xyzw"]), camera2ego[:3, :3], rtol=0, atol=1e-9)


#: Where each nuScenes camera's optical axis points in the rig, degrees from +x towards +y.
YAW = {
    "camera_front_wide_120fov": 0.582,
    "camera_cross_left_120fov": 55.337,
    "camera_cross_right_120fov": -57.579,
    "camera_rear_left_70fov": 108.577,
    "camera_front_tele_30fov": 179.476,
    "camera_rear_right_70fov": -112.523,
}


def test_each_camera_looks_where_its_nuscenes_camera_looks() -> None:
    """OpenCV axes in AlpaSim's rig (x forward, y left, z up): the optical axis (z) at the
    nuScenes camera's yaw, near level; the image's down (y) near -z, so the image is upright."""
    extra = _extra()
    for camera_id, yaw in YAW.items():
        rotation = _rotation(*extra[camera_id]["rig_to_camera"]["rotation_xyzw"])
        optical, down = rotation[:, 2], rotation[:, 1]
        assert math.degrees(math.atan2(optical[1], optical[0])) == pytest.approx(yaw, abs=1e-3), camera_id
        assert abs(math.degrees(math.asin(optical[2]))) < 1.2, camera_id
        assert down[2] < -math.cos(math.radians(1.5)), camera_id


def test_cam_back_sees_no_ground_under_the_car() -> None:
    """CAM_BACK keeps nuScenes' pose over the rear axle: its bottom image row meets the ground
    behind the car (AlpaSim's default vehicle ends 1.3 m behind the axle), so it never looks
    onto ground the car stands on, which no recorded camera saw (the driver config's header)."""
    camera = _extra()["camera_front_tele_30fov"]
    pinhole = camera["intrinsics"]["opencv_pinhole"]
    (fx, fy), (cx, cy) = pinhole["focal_length"], pinhole["principal_point"]
    rotation = _rotation(*camera["rig_to_camera"]["rotation_xyzw"])
    position = np.asarray(camera["rig_to_camera"]["translation_m"])
    height, width = camera["resolution_hw"]
    for u in np.linspace(0, width - 1, 9):
        ray = rotation @ np.array([(u - cx) / fx, (height - 1 - cy) / fy, 1.0])
        assert ray[2] < 0
        ground = position + ray * (-position[2] / ray[2])
        assert ground[0] < -2.5, (u, ground)
