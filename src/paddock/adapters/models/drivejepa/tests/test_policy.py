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
