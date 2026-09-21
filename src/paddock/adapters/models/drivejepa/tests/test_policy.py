"""Contract tests for the Drive-JEPA adapter. No navsim, no torch, no GPU."""

from __future__ import annotations

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


def test_missing_lidar2ego_is_reported_as_calibration() -> None:
    """Without this guard the failure lands deep inside a feature builder.

    Drive-JEPA projects through the lidar frame, so an Observation from a source
    that omits lidar2ego must be rejected where the cause is still legible.
    """
    obs = Observation(
        images={},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
        lidar2ego=None,
    )
    with pytest.raises(ValueError, match="lidar2ego"):
        obs.require_lidar2ego()


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
