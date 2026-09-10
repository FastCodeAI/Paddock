"""Signed clearance between oriented rectangles, checked against hand geometry.

The boolean `collides` it supplements is why a per-scene collision column came
out more than half exactly zero: a clean miss and a four-centimetre miss were
the same number. These cases pin the sign convention and the two branches, and
run with no dataset and no GPU.
"""

from __future__ import annotations

import math

import pytest

from paddock.adapters.benchmarks.nuscenes import loader as L

# [x, y, yaw, length, width]
UNIT = [0.0, 0.0, 0.0, 2.0, 2.0]  # 2x2 square at the origin


def test_axis_aligned_gap_is_the_edge_to_edge_distance() -> None:
    """Two squares 3 m apart on x, each 2 m long: 1 m of clear road between."""
    assert L.obb_signed_margin(UNIT, [3.0, 0.0, 0.0, 2.0, 2.0]) == pytest.approx(1.0)


def test_touching_boxes_are_zero() -> None:
    assert L.obb_signed_margin(UNIT, [2.0, 0.0, 0.0, 2.0, 2.0]) == pytest.approx(0.0, abs=1e-9)


def test_overlap_is_negative_and_is_the_penetration_depth() -> None:
    """Centres 1.5 m apart, half-widths 1 m each: 0.5 m of overlap."""
    assert L.obb_signed_margin(UNIT, [1.5, 0.0, 0.0, 2.0, 2.0]) == pytest.approx(-0.5)


def test_corner_to_corner_uses_the_true_distance_not_the_face_normal_gap() -> None:
    """The case face-normal SAT gets wrong.

    Diagonally offset squares touch corner to corner. The true gap is the
    diagonal between the two nearest corners; the largest face-normal gap is
    only its x (or y) component, which is smaller.
    """
    other = [4.0, 4.0, 0.0, 2.0, 2.0]
    corner_gap = math.hypot(2.0, 2.0)  # corner (1, 1) to corner (3, 3)
    assert L.obb_signed_margin(UNIT, other) == pytest.approx(corner_gap)
    # Face normals see only the x (or y) component: [-1, 1] against [3, 5] is a
    # gap of 2.0, short of the real 2.83 m.
    assert L.obb_signed_margin(UNIT, other) > 2.0


def test_rotation_is_respected() -> None:
    """A 45-degree square reaches further along x by its half-diagonal."""
    rotated = [3.0, 0.0, math.pi / 4, 2.0, 2.0]
    assert L.obb_signed_margin(UNIT, rotated) == pytest.approx(3.0 - 1.0 - math.sqrt(2.0))


def test_sign_agrees_with_the_boolean_it_supplements() -> None:
    for dx in (0.5, 1.5, 1.999, 2.0, 2.001, 3.0, 10.0):
        other = [dx, 0.0, 0.0, 2.0, 2.0]
        hit = L.obb_intersects(UNIT, other)
        assert hit == (L.obb_signed_margin(UNIT, other) <= 0.0), dx


def test_margin_to_takes_the_nearest_box() -> None:
    boxes = [[10.0, 0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 0.0, 2.0, 2.0], [20.0, 0.0, 0.0, 2.0, 2.0]]
    assert L.margin_to(UNIT, boxes) == pytest.approx(2.0)


def test_margin_to_an_empty_scene_is_infinite_not_a_number() -> None:
    """A fabricated large distance would enter the metric as if it were measured."""
    assert L.margin_to(UNIT, []) == float("inf")
