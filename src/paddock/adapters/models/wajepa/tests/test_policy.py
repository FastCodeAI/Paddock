"""Contract tests for the WA-JEPA adapter. No torch, no cv2, no GPU, no weights.

What can go wrong here without an error is the bookkeeping around the model --
which camera fills which slot, which command index means "left", how far apart
the history frames are -- so that is what these pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.wajepa.policy import (
    CAMERA_MAP,
    COMMAND_TO_NAVSIM,
    HISTORY_DT,
    HISTORY_FRAMES,
    POLICY,
    ego_status,
    pose_se2,
)
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command, Observation, Policy


def make_observation(
    command: Command = Command.STRAIGHT, speed: float = 8.0, accel: float = 0.5
) -> Observation:
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = speed
    canbus[CANBUS_ACCEL] = accel
    return Observation(
        images={},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=command,
        camera2image={},
        camera2ego={},
    )


def test_policy_imports_without_torch() -> None:
    policy = POLICY(checkpoint="/nonexistent.pt", config="/nonexistent.yaml")
    assert isinstance(policy, Policy)


def test_cameras_are_declared_in_the_models_slot_order() -> None:
    """The model indexes a camera embedding by list position; order is the contract."""
    assert list(CAMERA_MAP) == ["cam_l0", "cam_f0", "cam_r0", "cam_b0"]
    assert POLICY.cameras == ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK")


def test_load_without_paths_names_the_settings() -> None:
    policy = POLICY()
    policy.checkpoint = policy.config = None
    with pytest.raises(RuntimeError, match="WAJEPA_CHECKPOINT"):
        policy.load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY(checkpoint="/x.pt", config="/x.yaml").infer(make_observation())


def test_commands_map_onto_navsims_order() -> None:
    """NAVSIM: 0 left, 1 straight, 2 right, 3 unknown. An inverted map steers the wrong way silently."""
    assert COMMAND_TO_NAVSIM[Command.LEFT] == 0
    assert COMMAND_TO_NAVSIM[Command.STRAIGHT] == 1
    assert COMMAND_TO_NAVSIM[Command.RIGHT] == 2
    assert COMMAND_TO_NAVSIM[Command.FOLLOW_REFERENCE] == 3
    assert set(COMMAND_TO_NAVSIM) == set(Command)


def test_ego_status_layout() -> None:
    status = ego_status(make_observation(Command.LEFT, speed=8.0, accel=0.5))
    np.testing.assert_allclose(status, [1, 0, 0, 0, 8.0, 0.0, 0.5, 0.0])
    assert status.dtype == np.float32


def test_pose_se2_reads_position_and_yaw() -> None:
    yaw = 0.3
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:3, 3] = [4.0, -2.0, 0.5]
    np.testing.assert_allclose(pose_se2(pose), [4.0, -2.0, yaw], atol=1e-12)


def test_history_is_half_a_second_apart_at_ten_hertz() -> None:
    """NeuroNCAP steps at 0.1 s; the last four requests would be 0.3 s of history."""
    policy = POLICY(checkpoint="/x.pt", config="/x.yaml")
    policy.history = [{"timestamp": i * 100_000} for i in range(30)]
    chosen = [f["timestamp"] for f in policy._select_history()]
    assert len(chosen) == HISTORY_FRAMES
    assert np.allclose(np.diff(chosen), HISTORY_DT * 1e6)
    assert chosen[-1] == 29 * 100_000


def test_history_repeats_the_first_frame_while_warming_up() -> None:
    policy = POLICY(checkpoint="/x.pt", config="/x.yaml")
    policy.history = [{"timestamp": i * 100_000} for i in range(3)]
    assert [f["timestamp"] for f in policy._select_history()] == [0, 0, 0, 200_000]
