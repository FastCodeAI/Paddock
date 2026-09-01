"""Contract tests for the openpilot adapter.

Runs in paddock's own environment: no openpilot, no GPU, no dataset. It cannot
check that the model is good -- it checks that the adapter is wired correctly,
which is what a reviewer can actually verify on a PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.openpilot.policy import (
    POLICY,
    TARGET_H,
    TARGET_W,
    CameraFit,
    plan_to_grid,
)
from paddock.policy.protocol import Pinhole, Policy


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


@pytest.mark.parametrize(
    ("name", "pinhole", "mode"),
    [
        # Wider than openpilot's 1.596 target: cropping would trim width, and
        # horizontal field of view is what the wide model view is short of.
        ("nuscenes render", Pinhole(1266, 1266, 800, 450, 1600, 900), "letterbox"),
        # Taller: trimming height costs no horizontal field of view at all.
        ("kitscenes ring", Pinhole(1841, 1841, 1740, 1139, 3504, 2272), "crop"),
    ],
)
def test_fit_strategy_follows_the_source_aspect(name, pinhole, mode) -> None:
    """Both loop types feed this one class, with sources of opposite aspect.

    Before paddock these were two classes, each hardcoding one strategy, which
    is why neither loop could run the other's data.
    """
    fit = CameraFit.compute(pinhole)
    assert fit.mode == mode, name
    assert fit.scaled_size[0] == TARGET_W
    assert fit.scaled_size[1] <= TARGET_H


def test_cropping_preserves_full_width() -> None:
    """The point of cropping rather than letterboxing a tall source."""
    fit = CameraFit.compute(Pinhole(1841, 1841, 1740, 1139, 3504, 2272))
    left, _, right, _ = fit.crop
    assert right - left == 3504, "cropping a tall source must not lose any width"


def test_letterboxing_preserves_full_width_too() -> None:
    fit = CameraFit.compute(Pinhole(1266, 1266, 800, 450, 1600, 900))
    left, _, right, _ = fit.crop
    assert right - left == 1600


@pytest.mark.parametrize(
    "pinhole",
    [
        Pinhole(1266, 1266, 800, 450, 1600, 900),
        Pinhole(1841, 1841, 1740, 1139, 3504, 2272),
    ],
)
def test_intrinsics_describe_the_actual_transform(pinhole) -> None:
    """The output intrinsics must match what `prepare` really does to the pixels.

    Not that the principal point lands at the output centre -- it does not, and
    should not. When the full width is kept (the crop branch) the principal
    point stays wherever the sensor put it, 6.6 px off centre for KITScenes.
    Forcing it to the middle would misdescribe the image and bend every
    projection that uses these intrinsics.
    """
    fit = CameraFit.compute(pinhole)
    x0, y0, _, _ = fit.crop
    assert fit.intrinsics[0, 2] == pytest.approx((pinhole.cx - x0) * fit.scale)
    assert fit.intrinsics[1, 2] == pytest.approx((pinhole.cy - y0) * fit.scale + fit.pad_top)
    assert fit.intrinsics[0, 0] == pytest.approx(pinhole.fx * fit.scale)


@pytest.mark.parametrize("aspect", [1.0, 1.33, 1.542, 1.596, 1.6, 1.778, 2.35])
def test_any_aspect_fits_the_compiled_input_size(aspect: float) -> None:
    """The branch invariant: neither strategy can overflow the target height.

    Crop sets crop_h = w/target_aspect, so the scaled height is exactly
    TARGET_H; letterbox is only chosen when the source is wider, which makes it
    strictly less. Checked across a range so a change to the branching shows up
    here rather than as a silently truncated image.
    """
    width = 2000
    height = int(width / aspect)
    fit = CameraFit.compute(Pinhole(1500, 1500, width / 2, height / 2, width, height))
    assert fit.scaled_size[1] <= TARGET_H
    assert fit.scaled_size[0] == TARGET_W


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
