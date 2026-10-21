"""Contract tests for the Drive-JEPA adapter. No navsim, no torch, no GPU."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from paddock.adapters.models.drivejepa.policy import (
    CAMERA_MAP,
    COMMAND_ONE_HOT,
    HISTORY_DT,
    HISTORY_FRAMES,
    POLICY,
    relative_pose,
)
from paddock.policy.protocol import Command, Observation, Policy


def test_policy_imports_without_navsim_installed() -> None:
    policy = POLICY(checkpoint="/nonexistent.ckpt")
    assert isinstance(policy, Policy)
    assert set(policy.cameras) == set(CAMERA_MAP.values())


def test_load_without_a_checkpoint_says_so() -> None:
    """The failure must name the missing setting, not surface as a file error."""
    policy = POLICY(checkpoint=None)
    policy.checkpoint = None
    with pytest.raises(RuntimeError, match="DRIVEJEPA_CHECKPOINT"):
        policy.load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY(checkpoint="/x.ckpt").infer(object())  # type: ignore[arg-type]


def test_every_command_maps_to_a_one_hot() -> None:
    """A missing command would KeyError mid-run, several minutes into a scenario."""
    for command in Command:
        one_hot = COMMAND_ONE_HOT[int(command)]
        assert sum(one_hot) == 1, f"{command} is not one-hot: {one_hot}"


def test_relative_pose_is_identity_against_itself() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 5.0, 0.0]
    np.testing.assert_allclose(relative_pose(pose, pose), [0.0, 0.0, 0.0], atol=1e-12)


def test_relative_pose_recovers_a_translation() -> None:
    ref = np.eye(4)
    pose = np.eye(4)
    pose[:3, 3] = [3.0, -2.0, 0.0]
    np.testing.assert_allclose(relative_pose(pose, ref), [3.0, -2.0, 0.0], atol=1e-12)


def test_history_target_matches_navsim_rate() -> None:
    """Drive-JEPA is a 2 Hz agent with four frames of history: 1.5 s of context."""
    assert HISTORY_FRAMES == 4
    assert HISTORY_DT == 0.5
    assert round((HISTORY_FRAMES - 1) * HISTORY_DT, 2) == 1.5


def test_infer_sequence_records_the_history_and_runs_the_model_once() -> None:
    """Four frames in, one forward pass out.

    The base class feeds every frame through `infer`; this policy overrides that
    to record the first N-1 and infer on the last, because the model is a pure
    function of its recorded history. Pinned here so a refactor that drops the
    override quietly quadruples the cost of every open-loop run.
    """
    import numpy as np

    from paddock.policy.protocol import Trajectory

    class _Stub(POLICY):
        def __init__(self):
            self.recorded, self.inferred, self.resets = [], [], 0
            self.self_prime = True

        def reset(self):
            self.resets += 1
            self.recorded = []

        def _record(self, obs):
            self.recorded.append(obs)

        def infer(self, obs):
            self.inferred.append(obs)
            return Trajectory(np.zeros((8, 3)))

    stub = _Stub()
    frames = ["t-1.5", "t-1.0", "t-0.5", "t0"]
    stub.infer_sequence(frames)

    assert stub.resets == 1
    assert stub.recorded == frames[:-1], "history must be recorded without inference"
    assert stub.inferred == [frames[-1]], "exactly one forward pass, on the scored frame"
    assert stub.self_prime is True, "self_prime must be restored after the sequence"


def test_an_unknown_image_geometry_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("DRIVEJEPA_IMAGE_GEOMETRY", "fisheye")
    with pytest.raises(ValueError, match="DRIVEJEPA_IMAGE_GEOMETRY"):
        POLICY(checkpoint="/x.ckpt")


def test_the_default_geometry_is_the_one_published_rows_used(monkeypatch) -> None:
    monkeypatch.delenv("DRIVEJEPA_IMAGE_GEOMETRY", raising=False)
    assert POLICY(checkpoint="/x.ckpt").image_geometry == "native"


def _counting_observation(t_s: float, opened: list) -> Observation:
    """An observation whose images are counted as they are opened."""

    class Counted(Observation):
        def image(self, camera: str = "CAM_FRONT") -> np.ndarray:
            opened.append((t_s, camera))
            return np.full((4, 4, 3), int(t_s * 10), dtype=np.uint8)

    k = np.eye(3)
    return Counted(
        images={},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=int(t_s * 1e6),
        command=Command.STRAIGHT,
        camera2image={c: k for c in CAMERA_MAP.values()},
        camera2ego={c: np.eye(4) for c in CAMERA_MAP.values()},
        lidar2ego=np.eye(4),
    )


def test_only_the_selected_history_frames_are_decoded() -> None:
    """A 10 Hz harness sends 49 frames; the model reads four. Only those are opened."""
    policy = POLICY(checkpoint="/x.ckpt")
    opened: list = []
    for tick in range(49):
        policy.observe(_counting_observation(tick / 10, opened))
    assert opened == [], "recording must not decode"

    frames = policy._select_history()
    policy._decode(frames)
    assert sorted({t for t, _ in opened}) == [3.3, 3.8, 4.3, 4.8]
    assert len(opened) == 4 * len(CAMERA_MAP), "each selected frame, each camera, once"
    assert all(f["images"]["CAM_FRONT"][0, 0, 0] == int(round(f["timestamp"] / 1e5)) for f in frames)

    policy._decode(frames)
    assert len(opened) == 4 * len(CAMERA_MAP), "a decoded frame is not decoded again"


@pytest.mark.parametrize("geometry", ["native", "openscene"])
def test_the_image_geometry_reaches_the_run_record(monkeypatch, geometry: str) -> None:
    """The geometry changes the pixels the model sees, so every run records it.

    An explicit env override also lands in `model_env`, but the default does not;
    `diagnostics()` is what the runner stores as `provenance.policy` either way.
    """
    monkeypatch.setenv("DRIVEJEPA_IMAGE_GEOMETRY", geometry)
    policy = POLICY(checkpoint="/x.ckpt")
    assert policy.diagnostics()["image_geometry"] == geometry, "before any frame"
    policy.observe(_counting_observation(0.0, []))
    assert policy.diagnostics()["image_geometry"] == geometry, "after a frame"


#: nuScenes' LIDAR_TOP on the ego: yawed -90 degrees (its x axis points to the
#: car's right), 0.94 m forward and 1.84 m up.
NUSCENES_LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, 0.94], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]]
)
#: A nuScenes-like front camera: optical axis along ego x, 1.7 m forward, 1.5 m up.
FRONT2EGO = np.array(
    [[0.0, 0.0, 1.0, 1.7], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
)
FRONT_K = np.array([[1266.0, 0.0, 800.0], [0.0, 1266.0, 450.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def navsim_dataclasses(monkeypatch):
    """Stand-ins for the navsim dataclasses the adapter builds, keeping every keyword."""
    stub = types.ModuleType("navsim.common.dataclasses")
    for name in ("AgentInput", "Camera", "Cameras", "EgoStatus"):
        setattr(stub, name, type(name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}))
    for name, module in (
        ("navsim", types.ModuleType("navsim")),
        ("navsim.common", types.ModuleType("navsim.common")),
        ("navsim.common.dataclasses", stub),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return stub


def _nuscenes_observation(t_s: float) -> Observation:
    class Pixels(Observation):
        def image(self, camera: str = "CAM_FRONT") -> np.ndarray:
            return np.zeros((900, 1600, 3), dtype=np.uint8)

    ego2world = np.eye(4)
    ego2world[0, 3] = 5.0 * t_s
    return Pixels(
        images={},
        ego2world=ego2world,
        canbus=np.zeros(18),
        timestamp_us=int(t_s * 1e6),
        command=Command.STRAIGHT,
        camera2image={c: FRONT_K for c in CAMERA_MAP.values()},
        camera2ego={c: FRONT2EGO for c in CAMERA_MAP.values()},
        lidar2ego=NUSCENES_LIDAR2EGO,
    )


def test_a_point_ahead_of_the_car_lands_in_the_front_image(navsim_dataclasses) -> None:
    """The model reads the image at points written in ego coordinates.

    It applies each camera's `sensor2lidar` to them as if they were lidar
    coordinates, which holds in NAVSIM (lidar2ego is the identity) and nowhere
    else. Composing nuScenes' real LIDAR_TOP into the extrinsics put a point
    10 m ahead 10 m to the side, off the front camera, so the model planned
    without the image. The projection is the vendor's `lidar2img`: the inverse
    of sensor2lidar, then the intrinsics.
    """
    policy = POLICY(checkpoint="/x.ckpt")
    for t_s in (0.0, 0.5, 1.0, 1.5):
        policy.observe(_nuscenes_observation(t_s))
    front = policy._agent_input().cameras[-1].cam_f0

    rotation, translation = front.sensor2lidar_rotation, front.sensor2lidar_translation
    in_camera = rotation.T @ (np.array([10.0, 0.0, 0.5]) - translation)
    assert in_camera[2] > 0, "a point 10 m ahead is behind the front camera"
    u, v = (front.intrinsics @ in_camera)[:2] / in_camera[2]
    assert 0 < u < 1600 and 0 < v < 900, f"a point 10 m ahead projects to ({u:.0f}, {v:.0f}), off the image"
    np.testing.assert_allclose(rotation, FRONT2EGO[:3, :3], atol=1e-6)
    np.testing.assert_allclose(translation, FRONT2EGO[:3, 3], atol=1e-6)


@pytest.fixture
def devkit_crop(monkeypatch):
    """The devkit loader's crop, importable without navsim.

    `kitscenes_adapter` imports five navsim dataclasses at module level and uses
    none of them in the crop, so stand-ins are enough to reach `_crop_box`.
    """
    stub = types.ModuleType("navsim.common.dataclasses")
    for name in ("AgentInput", "Camera", "Cameras", "EgoStatus", "Lidar"):
        setattr(stub, name, type(name, (), {}))
    for name, module in (
        ("navsim", types.ModuleType("navsim")),
        ("navsim.common", types.ModuleType("navsim.common")),
        ("navsim.common.dataclasses", stub),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    adapter = "paddock.adapters.models.drivejepa.kitscenes_adapter"
    monkeypatch.delitem(sys.modules, adapter, raising=False)
    import importlib

    yield importlib.import_module(adapter)
    sys.modules.pop(adapter, None)


def test_openscene_crops_every_camera_onto_the_openscene_pinhole(devkit_crop, monkeypatch) -> None:
    """A KITScenes-like 87-degree camera comes out 1920x1080 at OpenScene's K.

    The principal point must land at OpenScene's (960, 560): a marker placed
    there in the source is found there in the output.
    """
    monkeypatch.setenv("DRIVEJEPA_IMAGE_GEOMETRY", "openscene")
    policy = POLICY(checkpoint="/x.ckpt")
    k = np.array([[1841.0, 0.0, 1752.0], [0.0, 1841.0, 1136.0], [0.0, 0.0, 1.0]])
    image = np.zeros((2272, 3504, 3), dtype=np.uint8)
    image[1116:1156, 1732:1772] = 255  # 40 px square on the principal point

    images, intrinsics = policy._to_openscene({"CAM_FRONT": image}, {"CAM_FRONT": k})

    out = images["CAM_FRONT"]
    assert out.shape == (devkit_crop.TARGET_HEIGHT, devkit_crop.TARGET_WIDTH, 3)
    np.testing.assert_array_equal(intrinsics["CAM_FRONT"], devkit_crop.target_intrinsics())
    assert out[560, 960].min() == 255, "the principal point moved"
    assert out[560, 960 + 60].max() == 0 and out[560 - 60, 960].max() == 0, "the marker grew"


class _Tensor:
    """Just enough of a torch tensor for `infer`: index, .float(), .cpu(), .numpy()."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array.astype(np.float32)


def _answering(policy, fed: bool):
    """Stub `forward` with a canned output whose pick is proposal 5."""
    rng = np.random.default_rng(1)
    proposals = rng.normal(size=(1, 32, 8, 3))
    scores = np.full((1, 32), 0.25)
    scores[0, 5] = 0.75

    def forward(obs):
        policy.momentum_fed = fed
        return {"trajectory": _Tensor(proposals[:, 5]), "pdm_score": _Tensor(scores)}

    policy.forward = forward
    return proposals[0, 5].astype(np.float32).astype(np.float64)


@pytest.mark.parametrize("fed", [True, False])
def test_the_answer_carries_all_eight_poses_and_the_pick(fed: bool) -> None:
    """`plan`: the eight (x, y, heading) rows at 0.5 s; `trajectory`: the six (x, y)
    it always was; extras: which proposal was driven, its score, and whether the
    momentum stage had a previous plan for it."""
    policy = POLICY(checkpoint="/x.ckpt")
    chosen = _answering(policy, fed=fed)
    wire = policy.infer(_nuscenes_observation(0.0)).to_wire()
    np.testing.assert_array_equal(np.asarray(wire["plan"]["waypoints"]), chosen)
    assert wire["plan"]["dt"] == HISTORY_DT
    np.testing.assert_array_equal(np.asarray(wire["trajectory"]), chosen[:6, :2])
    assert wire["extras"] == {
        "drivejepa_pick": {"chosen_index": 5, "chosen_score": 0.75, "momentum_fed": fed}
    }
