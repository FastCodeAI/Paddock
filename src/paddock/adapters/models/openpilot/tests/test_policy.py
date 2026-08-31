"""Contract tests for the openpilot adapter.

Runs in paddock's own environment: no openpilot, no GPU, no dataset. It cannot
check that the model is good -- it checks that the adapter is wired correctly,
which is what a reviewer can actually verify on a PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.openpilot.policy import POLICY, TARGET_H, TARGET_W, CameraAdapter, plan_to_grid
from paddock.policy.protocol import Policy


def test_policy_imports_without_openpilot_installed() -> None:
    """Module scope must stay free of openpilot imports.

    This module is imported by `paddock list`, by the registry, and by this test
    suite -- all of which run in paddock's environment, where openpilot's
    package does not exist. Every openpilot import belongs inside a method.
    """
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.cameras == ("CAM_FRONT",)
    assert policy.waypoint_dt == 0.5


def test_diagnostics_before_any_frame() -> None:
    """Diagnostics must answer before the camera is established, not raise."""
    assert POLICY().diagnostics() == {"status": "no frame seen yet"}


def test_infer_without_load_is_a_clear_error() -> None:
    policy = POLICY()
    with pytest.raises(RuntimeError, match="load"):
        policy.infer(object())  # type: ignore[arg-type]


def test_letterbox_rejects_a_source_it_would_have_to_crop() -> None:
    """A too-tall source must fail loudly rather than silently losing field of view.

    Losing vertical field of view here would not raise anywhere downstream; it
    would just quietly change what the model sees.
    """
    from paddock.policy.protocol import Pinhole

    tall = Pinhole(fx=500, fy=500, cx=100, cy=100, width=200, height=400)
    with pytest.raises(ValueError, match="too tall to letterbox"):
        CameraAdapter(tall, np.eye(4))


def test_plan_is_converted_to_y_left() -> None:
    """openpilot predicts y RIGHT; paddock and NeuroNCAP want y LEFT.

    Getting this wrong mirrors every trajectory and is invisible in a
    straight-line scenario -- it only shows up as a model that steers the wrong
    way around obstacles.
    """
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

    plan = np.zeros((len(ModelConstants.T_IDXS), 15))
    plan[:, Plan.POSITION][:, 0] = np.asarray(ModelConstants.T_IDXS) * 10.0
    plan[:, Plan.POSITION][:, 1] = 1.0  # 1 m to openpilot's right

    grid = plan_to_grid(plan)
    assert grid.shape == (8, 2)
    assert np.all(grid[:, 1] < 0), "y must be negated into the left-positive convention"


def test_target_resolution_matches_the_compiled_warp() -> None:
    """ModelState compiles its warp for a fixed geometry; 1928x1208 is the tici one."""
    assert (TARGET_W, TARGET_H) == (1928, 1208)
