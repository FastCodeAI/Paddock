"""Contract tests for the Alpamayo 1.5 adapter. No torch, no weights, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.alpamayo15.policy import (
    CAMERA_SLOTS,
    GRID_INDICES,
    HISTORY_DT,
    NUM_CAM_FRAMES,
    NUM_HISTORY_STEPS,
    POLICY,
    medoid,
)
from paddock.policy.protocol import Policy


def test_policy_imports_without_the_model_environment() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.waypoint_dt == 0.5


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(object())  # type: ignore[arg-type]


def test_grid_indices_land_on_the_half_second_grid() -> None:
    """Alpamayo predicts 64 waypoints at 10 Hz; paddock drives 0.5 s spacing.

    Index i is at (i+1)/10 s, so 0.5 s is index 4, 1.0 s is index 9, and so on.
    An off-by-one here shifts every waypoint 100 ms and is invisible in the
    output -- it just makes the model look slightly mistimed.
    """
    expected = [int(round(t / 0.1)) - 1 for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    assert list(GRID_INDICES) == expected


def test_medoid_picks_the_consensus_draw() -> None:
    """Not the mean: a mean can land between modes, on a path nothing proposed."""
    a = np.zeros((6, 2))
    b = np.ones((6, 2)) * 0.1
    outlier = np.ones((6, 2)) * 100.0
    chosen = medoid(np.stack([a, b, outlier]))
    assert not np.allclose(chosen, outlier)


def test_medoid_of_one_is_that_one() -> None:
    only = np.arange(12, dtype=float).reshape(6, 2)
    np.testing.assert_array_equal(medoid(np.stack([only])), only)


def test_camera_slots_cover_the_declared_cameras() -> None:
    """Every camera a slot needs must be one the policy asks the renderer for."""
    declared = set(POLICY().cameras)
    used = {camera for _, camera, _ in CAMERA_SLOTS}
    assert used <= declared, f"slots use cameras not declared: {used - declared}"


def test_history_depth_matches_the_model_contract() -> None:
    assert (NUM_HISTORY_STEPS, NUM_CAM_FRAMES, HISTORY_DT) == (16, 4, 0.1)
