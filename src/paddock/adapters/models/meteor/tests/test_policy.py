"""Contract tests for the METEOR adapter. No onnxruntime, no weights, no GPU.

They cannot say whether the model is good. They check that the adapter builds
the graph's feed the way METEOR's own smoke test does, decodes the ego head
the way METEOR's own evaluator does, and obeys paddock's trajectory contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from paddock.adapters.models.meteor.policy import (
    ABSENT_FILL_RGB,
    EGO_LEN,
    INPUT_H,
    INPUT_W,
    NATIVE_HORIZON_S,
    POLICY,
    SLOT_FROM_CAMERA,
    SLOTS,
    WAYPOINT_DT,
    build_inputs,
    decode_ego,
    extend_horizon,
    fit_frame,
)
from paddock.policy.protocol import CANBUS_VELOCITY, N_WAYPOINTS, Command, Observation, Policy
from paddock.registry import Registry

NUSCENES_K = np.array([[1266.0, 0.0, 816.0], [0.0, 1266.0, 491.0], [0.0, 0.0, 1.0]])


def cam2ego(x: float, y: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    """OpenCV-optical camera -> ego (x fwd, y left, z up), yawed about z."""
    optical_to_forward = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.eye(4)
    t[:3, :3] = yaw @ optical_to_forward
    t[:3, 3] = [x, y, z]
    return t


def make_observation(cameras: tuple[str, ...] = ("CAM_FRONT",), speed: float = 10.0) -> Observation:
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    return Observation(
        images={c: np.full((900, 1600, 3), 90, dtype=np.uint8) for c in cameras},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={c: NUSCENES_K.copy() for c in cameras},
        camera2ego={c: cam2ego(1.5, 0.0, 1.5, yaw_deg=30.0 * i) for i, c in enumerate(cameras)},
    )


def synthetic_ego(best_mode: int = 1, speed: float = 10.0) -> np.ndarray:
    """A 42-vector whose mode blocks are distinguishable and whose best mode drives ahead."""
    v = np.zeros(EGO_LEN, dtype=np.float32)
    for k in range(3):
        t = WAYPOINT_DT * np.arange(1, 7)
        v[k * 12 : (k + 1) * 12] = np.stack([speed * t, (k - 1) * 0.5 * t], axis=1).reshape(-1)
    v[36:39] = [-1.0, -1.0, -1.0]
    v[36 + best_mode] = 2.0
    v[39:42] = [0.05, -0.3, 1.5]
    return v


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: checks the feed, returns a canned ego."""

    def __init__(self, ego: np.ndarray) -> None:
        self.ego = ego
        self.calls = 0
        self.feeds: list[dict] = []

    def run(self, names, feed):
        assert names == ["ego"]
        assert feed["imgs"].dtype == np.uint8 and feed["imgs"].shape == (1, 8, 3, INPUT_H, INPUT_W)
        assert feed["K"].dtype == np.float32 and feed["K"].shape == (1, 8, 3, 3)
        assert feed["T_cam_ego"].dtype == np.float32 and feed["T_cam_ego"].shape == (1, 8, 4, 4)
        assert feed["v0"].dtype == np.float32 and feed["v0"].shape == (1,)
        self.calls += 1
        self.feeds.append(feed)
        return [self.ego[None].copy()]


def loaded_policy(ego: np.ndarray | None = None):
    policy = POLICY()
    policy.session = FakeSession(synthetic_ego() if ego is None else ego)
    return policy


# ---- registration -----------------------------------------------------------


def test_policy_imports_without_onnxruntime() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.cameras == tuple(SLOT_FROM_CAMERA)
    assert policy.self_prime is False


def test_model_yaml_agrees_with_the_policy() -> None:
    """The runner renders model.yaml's cameras; the node decodes POLICY.cameras."""
    spec = Registry.load().models["meteor"]
    assert spec.cameras == POLICY.cameras
    assert spec.waypoint_dt == WAYPOINT_DT
    assert spec.min_input_hz == 0.0


def test_load_without_onnx_names_the_setting() -> None:
    with pytest.raises(RuntimeError, match="METEOR_ONNX"):
        POLICY(onnx="/nonexistent/meteor.onnx").load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(make_observation())


def test_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("METEOR_ORT_THREADS", raising=False)
    monkeypatch.delenv("METEOR_HORIZON_S", raising=False)
    assert POLICY().threads == 4
    assert POLICY().horizon_s == NATIVE_HORIZON_S
    monkeypatch.setenv("METEOR_ORT_THREADS", "2")
    monkeypatch.setenv("METEOR_HORIZON_S", "4.0")
    monkeypatch.setenv("METEOR_PROVIDERS", "cpu")
    policy = POLICY()
    assert policy.threads == 2
    assert policy.horizon_s == 4.0
    assert policy.provider_mode == "cpu"
    monkeypatch.setenv("METEOR_ORT_THREADS", "all")
    assert POLICY().threads == 4
    monkeypatch.setenv("METEOR_HORIZON_S", "2.0")
    with pytest.raises(ValueError, match="shorter"):
        POLICY()


# ---- input side -------------------------------------------------------------


def test_fit_frame_crops_kitscenes_to_16_9_around_the_principal_point() -> None:
    h, w = 2272, 3504
    k = np.array([[1841.0, 0.0, 1740.0], [0.0, 1841.0, 1139.0], [0.0, 0.0, 1.0]])
    image = np.zeros((h, w, 3), dtype=np.uint8)
    u, v = 2400, 1500  # a source pixel inside the crop
    image[v - 8 : v + 8, u - 8 : u + 8] = 255
    out, k2 = fit_frame(image, k)
    assert out.shape == (INPUT_H, INPUT_W, 3) and out.dtype == np.uint8
    ch = round(w / (INPUT_W / INPUT_H))  # 1971 rows survive
    y0 = round(1139 - ch / 2)
    s = INPUT_W / w
    np.testing.assert_allclose([k2[0, 0], k2[1, 1]], [1841.0 * s, 1841.0 * INPUT_H / ch], rtol=1e-3)
    np.testing.assert_allclose([k2[0, 2], k2[1, 2]], [1740.0 * s, (1139.0 - y0) * INPUT_H / ch], rtol=1e-3)
    bright = np.argwhere(out[:, :, 0] > 128)
    np.testing.assert_allclose(bright.mean(axis=0), [(v - y0) * INPUT_H / ch, u * s], atol=1.5)


def test_fit_frame_nuscenes_is_a_pure_scale() -> None:
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    out, k2 = fit_frame(image, NUSCENES_K)
    assert out.shape == (INPUT_H, INPUT_W, 3)
    np.testing.assert_allclose(k2, np.diag([0.48, 0.48, 1.0]) @ NUSCENES_K, rtol=1e-6)


def test_fit_frame_passes_a_native_frame_through() -> None:
    image = np.arange(INPUT_H * INPUT_W * 3, dtype=np.uint32).reshape(INPUT_H, INPUT_W, 3) % 251
    out, _ = fit_frame(image.astype(np.uint8), np.array([[500.0, 0, 384.0], [0, 500.0, 216.0], [0, 0, 1.0]]))
    np.testing.assert_array_equal(out, image.astype(np.uint8))


def test_build_inputs_single_camera_fills_the_rest_from_the_front() -> None:
    """KITScenes: a front image path, single-key calibration dicts, nothing else."""
    obs = make_observation(("CAM_FRONT",), speed=7.5)
    feed, present = build_inputs(obs)
    assert present == ("CAM_FRONT_WIDE",)
    imgs, k, t = feed["imgs"][0], feed["K"][0], feed["T_cam_ego"][0]
    assert imgs[0].mean() == pytest.approx(90.0)
    for i in range(1, 8):
        np.testing.assert_array_equal(imgs[i][:, 0, 0], ABSENT_FILL_RGB)
        assert (imgs[i] == imgs[i][:, :1, :1]).all()
        np.testing.assert_array_equal(k[i], k[0])
        np.testing.assert_array_equal(t[i], t[0])
    np.testing.assert_allclose(t[0] @ obs.camera2ego["CAM_FRONT"], np.eye(4), atol=1e-5)
    assert feed["v0"][0] == pytest.approx(7.5)


def test_build_inputs_six_cameras_use_upstream_donors() -> None:
    obs = make_observation(tuple(SLOT_FROM_CAMERA))
    feed, present = build_inputs(obs)
    assert present == SLOTS[:6]
    k = feed["K"][0]
    t = feed["T_cam_ego"][0]
    idx = {s: i for i, s in enumerate(SLOTS)}
    np.testing.assert_array_equal(t[idx["CAM_FRONT_NARROW"]], t[idx["CAM_FRONT_WIDE"]])
    np.testing.assert_array_equal(t[idx["CAM_BACK_NARROW"]], t[idx["CAM_BACK_WIDE"]])
    assert not np.array_equal(t[idx["CAM_BACK_WIDE"]], t[idx["CAM_FRONT_WIDE"]])
    assert np.array_equal(k[0], k[1])  # same intrinsics everywhere in this fixture


def test_build_inputs_reads_paths_without_touching_absent_cameras(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "front.jpg"
    Image.fromarray(np.full((900, 1600, 3), 60, dtype=np.uint8)).save(path)
    obs = make_observation(())
    obs.image_paths = {"CAM_FRONT": str(path)}
    obs.camera2image = {"CAM_FRONT": NUSCENES_K}
    obs.camera2ego = {"CAM_FRONT": cam2ego(1.5, 0.0, 1.5)}
    feed, present = build_inputs(obs)
    assert present == ("CAM_FRONT_WIDE",)
    assert 55 <= feed["imgs"][0, 0].mean() <= 65


def test_build_inputs_with_no_camera_is_an_error() -> None:
    with pytest.raises(ValueError, match="carries none"):
        build_inputs(make_observation(()))


# ---- output side ------------------------------------------------------------


def test_decode_ego_picks_the_argmax_mode_block() -> None:
    decoded = decode_ego(synthetic_ego(best_mode=2))
    assert decoded.mode == 2
    assert decoded.waypoints.shape == (6, 2)
    np.testing.assert_allclose(decoded.waypoints[:, 1], 0.5 * WAYPOINT_DT * np.arange(1, 7))
    assert decoded.logits == [-1.0, -1.0, 2.0]
    assert (decoded.steer_rad, decoded.accel, decoded.brake_logit) == (
        pytest.approx(0.05),
        pytest.approx(-0.3),
        1.5,
    )
    with pytest.raises(ValueError, match=str(EGO_LEN)):
        decode_ego(np.zeros(41))


def test_extend_horizon_holds_the_last_segment() -> None:
    wp = np.array([[1.0, 0.0], [2.0, 0.1], [3.0, 0.3]])
    out = extend_horizon(wp, 0.5, 2.5)
    assert out.shape == (5, 2)
    np.testing.assert_allclose(out[3], [4.0, 0.5])
    np.testing.assert_allclose(out[4], [5.0, 0.7])
    np.testing.assert_array_equal(extend_horizon(wp, 0.5, 1.5), wp)


def test_infer_obeys_the_contract() -> None:
    policy = loaded_policy()
    policy.reset()
    traj = policy.infer(make_observation(speed=10.0))
    assert traj.waypoints.shape == (6, 2)
    assert len(traj.waypoints) >= N_WAYPOINTS
    assert np.isfinite(traj.waypoints).all()
    assert traj.xy[-1, 0] > traj.xy[0, 0] > 0, "x is forward; a moving ego plans ahead of itself"
    assert not np.allclose(traj.xy, 0.0), "all-zero means emergency stop"
    assert traj.dt == WAYPOINT_DT
    assert traj.extras["meteor_mode"] == 1
    assert "meteor_extrapolated_to_s" not in traj.extras
    json.dumps(traj.to_wire())  # numpy scalars would not survive the wire
    diag = policy.diagnostics()
    assert diag["last"]["cameras_present"] == ["CAM_FRONT_WIDE"]
    assert len(diag["last"]["cameras_absent"]) == 7


def test_infer_extends_to_the_requested_horizon(monkeypatch) -> None:
    monkeypatch.setenv("METEOR_HORIZON_S", "4.0")
    traj = loaded_policy().infer(make_observation())
    assert traj.waypoints.shape == (8, 2)
    assert traj.extras["meteor_extrapolated_to_s"] == 4.0
    np.testing.assert_allclose(np.diff(traj.xy[4:], axis=0), np.diff(traj.xy[4:6], axis=0)[[0, 0, 0]])


def test_infer_sequence_runs_one_forward() -> None:
    """The model is single-frame: history frames are observed for free."""
    policy = loaded_policy()
    traj = policy.infer_sequence([make_observation() for _ in range(5)])
    assert policy.session.calls == 1
    assert traj.waypoints.shape == (6, 2)


def test_non_finite_output_raises_rather_than_planning() -> None:
    ego = synthetic_ego()
    ego[12:24] = np.nan  # the best mode's block
    with pytest.raises(RuntimeError, match="non-finite"):
        loaded_policy(ego).infer(make_observation())


def test_reset_is_idempotent() -> None:
    policy = loaded_policy()
    obs = make_observation()
    policy.reset()
    first = policy.infer(obs)
    policy.reset()
    policy.reset()
    second = policy.infer(obs)
    np.testing.assert_allclose(first.waypoints, second.waypoints)
