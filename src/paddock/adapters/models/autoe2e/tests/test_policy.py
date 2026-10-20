"""Contract tests for the AutoE2E adapter: the input packing and the control
rollout, which are the only parts not imported from upstream. No torch, no
checkpoint, no GPU. Conventions are upstream's at 40a75cf8:
navigation/geometry.py (raster rows front to rear, columns left to right),
training/losses/control_rollout.py (speed clamped first, heading from the new
speed), kit_scenes/egomotion.py (gradients on the real timestamps)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock.adapters.models.autoe2e.policy import (
    EGO_STEPS,
    HISTORY_OFFSETS_S,
    NAV_H,
    NAV_MPP,
    NAV_W,
    NAV_X_MAX,
    NAV_Y_MAX,
    OUT_STEPS,
    POLICY,
    VIEW_ORDER,
    _Frame,
    egomotion_rows,
    finite_controls,
    integrate_controls,
    route_raster,
)
from paddock.policy.protocol import Policy

#: Camera optical (x right, y down, z forward) -> ego FLU, for a camera facing forward.
OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _pixel(x: float, y: float) -> tuple[int, int]:
    """Upstream's ego_to_pixel, rounded to the pixel whose centre is nearest."""
    return round((NAV_X_MAX - x) / NAV_MPP - 0.5), round((NAV_Y_MAX - y) / NAV_MPP - 0.5)


def test_policy_imports_without_torch_and_asks_for_six_views() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert tuple(policy.cameras) == VIEW_ORDER and len(VIEW_ORDER) == 6
    assert policy.waypoint_dt == 0.5


def test_load_names_the_missing_setting(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AUTOE2E_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AUTOE2E_CHECKPOINT"):
        POLICY().load()
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("AUTOE2E_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("AUTOE2E_REPO", str(tmp_path))
    with pytest.raises(RuntimeError, match="AUTOE2E_REPO"):
        POLICY().load()


def test_diagnostics_name_the_release_checkpoint_not_where_this_site_keeps_it() -> None:
    """`/diagnostics` is copied verbatim into the tracked run.json: no weights root may reach it."""
    policy = POLICY()
    policy.checkpoint = "/path/to/weights/auto_e2e/models/kitscenes-epoch-5/checkpoint.pt"
    assert policy.diagnostics()["checkpoint"] == "models/kitscenes-epoch-5/checkpoint.pt"
    policy.checkpoint = "/path/to/finetunes/run7.pt"
    assert policy.diagnostics()["checkpoint"] == "run7.pt"


# -- ego history -------------------------------------------------------------


def test_egomotion_on_a_circle_reads_speed_yaw_rate_and_curvature() -> None:
    v, w = 8.0, 0.2  # m/s, rad/s: a 40 m radius
    t = np.arange(0.0, 8.0 + 1e-9, 0.1)
    yaw = w * t
    xy = np.column_stack([np.sin(yaw), 1.0 - np.cos(yaw)]) * (v / w)
    rows = egomotion_rows(t, xy, yaw, t_now=8.0)
    assert rows.shape == (EGO_STEPS, 4)
    assert rows[:, 0] == pytest.approx(v, rel=1e-3)
    assert rows[:, 1] == pytest.approx(0.0, abs=1e-2)
    assert rows[:, 2] == pytest.approx(w, rel=1e-3)
    assert rows[:, 3] == pytest.approx(w / v, rel=1e-3)


def test_rows_older_than_the_first_pose_stay_zero() -> None:
    """Upstream pads left with zeros; a route start must not invent a past."""
    t = np.arange(0.0, 2.0 + 1e-9, 0.1)
    xy = np.column_stack([5.0 * t, np.zeros_like(t)])
    rows = egomotion_rows(t, xy, np.zeros_like(t), t_now=2.0)
    held = (np.abs(rows).sum(1) > 0).sum()
    assert held == 20, "t-2.0 .. t-0.1 s are inside the 2 s held"
    assert np.all(rows[:-20] == 0) and rows[-20:, 0] == pytest.approx(5.0)


def test_curvature_divides_by_the_speed_floor_when_stopped() -> None:
    t = np.arange(0.0, 7.0, 0.1)
    rows = egomotion_rows(t, np.zeros((len(t), 2)), 0.05 * t, t_now=7.0)
    assert rows[:, 3] == pytest.approx(0.05 / 0.1)


# -- control rollout -----------------------------------------------------------


def test_zero_controls_hold_speed_in_a_straight_line() -> None:
    out = integrate_controls(np.zeros((OUT_STEPS, 2)), v0=10.0)
    assert out.shape == (OUT_STEPS, 3)
    assert out[:, 0] == pytest.approx(10.0 * 0.1 * np.arange(1, OUT_STEPS + 1))
    assert np.all(out[:, 1:] == 0)


def test_braking_stops_and_never_reverses() -> None:
    controls = np.zeros((OUT_STEPS, 2))
    controls[:, 0] = -5.0
    out = integrate_controls(controls, v0=4.0)
    assert out[7, 0] == pytest.approx(out[-1, 0]), "stopped after 0.8 s"
    assert np.all(np.diff(out[:, 0]) >= 0)


def test_non_finite_controls_fail_the_call_instead_of_becoming_a_plan() -> None:
    """A crash must not become a score: NaN or inf from the model raises, so the scenario is
    counted as failed, never scored on a substitute plan under AutoE2E's name."""
    controls = np.zeros((OUT_STEPS, 2))
    assert finite_controls(controls) is controls
    for bad in (np.nan, np.inf):
        broken = controls.copy()
        broken[10, 1] = bad
        with pytest.raises(RuntimeError, match="non-finite controls at 1 of 64 steps"):
            finite_controls(broken)


def test_positive_curvature_turns_left() -> None:
    controls = np.zeros((OUT_STEPS, 2))
    controls[:, 1] = 0.05
    out = integrate_controls(controls, v0=10.0)
    assert out[-1, 2] == pytest.approx(10.0 * 0.05 * 6.4)
    assert out[-1, 1] > 0, "paddock y is left"


# -- navigation raster -----------------------------------------------------------


def test_the_route_corridor_lands_where_upstreams_geometry_puts_it() -> None:
    route = np.column_stack([np.arange(-50.0, 101.0), np.full(151, 3.0)])  # a lane 3 m to the left
    raster, corridor, destination = route_raster(route)
    assert raster.shape == (2, NAV_H, NAV_W) and corridor
    assert raster[0][_pixel(50.0, 3.0)] == 1
    assert raster[0][_pixel(50.0, -3.0)] == 0, "the mirror image must be empty"
    assert raster[0][_pixel(50.0, 3.0 + 2.5)] == 0, "1.75 m either side, no wider"
    assert raster[0][_pixel(-5.0, 3.0)] == 1 and raster[0][_pixel(-20.0, 3.0)] == 0, "rear-clipped at -10 m"
    assert destination, "a route that stops 100 m ahead ends there"
    assert raster[1][_pixel(100.0, 3.0)] == 1


def test_a_route_running_past_the_look_ahead_has_no_destination() -> None:
    route = np.column_stack([np.arange(-10.0, 201.0), np.zeros(211)])
    raster, corridor, destination = route_raster(route)
    assert corridor and not destination and raster[1].sum() == 0


def test_no_route_is_upstreams_camera_only_track() -> None:
    raster, corridor, destination = route_raster(None)
    assert not corridor and not destination and raster.sum() == 0


# -- camera history and projection ------------------------------------------------


def _pose(x: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 3] = x
    return m


def _frame(t_s: float, x: float, views=VIEW_ORDER, size=(8, 8)) -> _Frame:
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = OPTICAL
    cam2ego[:3, 3] = (1.5, 0.0, 1.5)
    k = np.array([[400.0, 0.0, 400.0], [0.0, 400.0, 225.0], [0.0, 0.0, 1.0]])
    return _Frame(
        t_s=t_s,
        ego2world=_pose(x),
        camera2image={c: k for c in views},
        camera2ego={c: cam2ego for c in views},
        images={c: np.zeros((size[1], size[0], 3), dtype=np.uint8) for c in views},
    )


def test_history_is_seven_six_view_frames_half_a_second_apart() -> None:
    """NAVSIM's 10 Hz frames between keyframes carry three views; only six-view frames count."""
    policy = POLICY()
    for k in range(51):  # 5 s at 10 Hz, six views at every 0.5 s only
        policy.frames.append(_frame(k / 10, 0.0, VIEW_ORDER if k % 5 == 0 else VIEW_ORDER[:3]))
    picked, gaps = policy._history(policy.frames[-1])
    assert [f.t_s for f in picked] == pytest.approx([5.0 + off for off in HISTORY_OFFSETS_S])
    assert gaps == [0.0] * 7


def test_a_short_past_repeats_the_oldest_frame_and_says_so() -> None:
    policy = POLICY()
    for k in range(21):  # 2 s held
        policy.frames.append(_frame(k / 10, 0.0))
    picked, gaps = policy._history(policy.frames[-1])
    assert gaps[:3] == [None, None, None] and gaps[3:] == [0.0] * 4
    assert all(f is policy.frames[0] for f in picked[:3])


def test_projection_is_from_the_raised_reference_frame_and_folds_in_ego_motion() -> None:
    """A world point on the camera's axis 10 m ahead lands at the tile centre at depth 10;
    in a frame taken 5 m further back, the same point is at depth 15."""
    policy = POLICY()
    policy.ref_height = 2.1
    now, past = _frame(1.0, 0.0, size=(800, 450)), _frame(0.5, -5.0, size=(800, 450))
    point_ref = np.array([11.5, 0.0, 1.5 - 2.1, 1.0])  # ego (11.5, 0, 1.5) in the reference frame
    p = policy._projections(now, None, 512)
    assert p.shape == (6, 3, 4)
    u, v, w = p[0] @ point_ref
    assert (u / w, v / w, w) == pytest.approx((256.0, 256.0, 10.0))
    ref_now2world = now.ego2world @ policy._ref2ego()
    u, v, w = policy._projections(past, ref_now2world, 512)[0] @ point_ref
    assert (u / w, v / w, w) == pytest.approx((256.0, 256.0, 15.0))
    assert math.isclose(policy._ref2ego()[2, 3], 2.1)
