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

    Eight points, not six: NAVSIM scores a 4 s horizon. NeuroNCAP still takes
    the first six and KITScenes truncates at 3 s, so the extra two are dropped
    by both -- see the note on GRID_INDICES.
    """
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert list(GRID_INDICES) == [int(round(t / 0.1)) - 1 for t in grid]
    assert GRID_INDICES[-1] < 64, "alpamayo predicts 64 waypoints; the grid must fit"


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
