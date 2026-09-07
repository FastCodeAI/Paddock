"""nuScenes open-loop protocol: geometry, the command rule, and the scorer."""

import math

import numpy as np

from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.adapters.benchmarks.nuscenes.harness import plan_headings, plan_on_grid, score_plan
from paddock.policy.protocol import Trajectory


def test_quaternion_identity_and_yaw():
    assert np.allclose(L.quat_to_rot([1, 0, 0, 0]), np.eye(3))
    q90 = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]  # 90 deg about z
    assert abs(L.yaw_of(q90) - math.pi / 2) < 1e-9


def test_world_points_land_in_the_ego_frame():
    ego = L.pose_matrix(
        [10.0, 5.0, 0.0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    )  # heading +90 deg
    ahead = np.array([[10.0, 15.0, 0.0]])  # 10 m north of the ego, i.e. straight ahead
    assert np.allclose(L.to_ego_xy(ego, ahead), [[10.0, 0.0]], atol=1e-9)
    boxes = np.array([[10.0, 15.0, math.pi / 2, 4.0, 2.0]])
    local = L.boxes_to_ego(ego, boxes)
    assert np.allclose(local[0, :3], [10.0, 0.0, 0.0], atol=1e-9)


def test_oriented_box_intersection():
    ego = [0.0, 0.0, 0.0, 4.084, 1.85]
    assert L.obb_intersects(ego, [3.0, 0.0, 0.0, 4.0, 2.0])  # overlapping ahead
    assert not L.obb_intersects(ego, [6.0, 0.0, 0.0, 4.0, 2.0])  # clear ahead
    assert not L.obb_intersects(ego, [0.0, 3.0, 0.0, 4.0, 2.0])  # clear to the left
    assert L.obb_intersects(ego, [2.5, 1.3, math.pi / 4, 4.0, 2.0])  # rotated, clipping the corner
    assert L.collides(ego, np.array([[6.0, 0.0, 0.0, 4.0, 2.0], [3.0, 0.0, 0.0, 4.0, 2.0]]))
    assert not L.collides(ego, np.zeros((0, 5)))


def test_command_follows_the_3s_lateral_offset():
    straight = np.column_stack([np.linspace(2, 12, 6), np.zeros(6)])
    assert L.command_from_future(straight) == (2, None, None)
    left = np.column_stack([np.linspace(2, 12, 6), np.linspace(0, 3, 6)])
    cmd, man, dist = L.command_from_future(left)
    # Distance is to where the turn BEGINS (first point past the 2 m threshold),
    # the along-track arc length there, not the straight-line end displacement.
    crossing = np.flatnonzero(np.abs(left[:, 1]) >= L.TURN_THRESHOLD_M)[0]
    seg = np.linalg.norm(np.diff(left, axis=0), axis=1)
    expected = round(float(np.concatenate([[0.0], np.cumsum(seg)])[crossing]), 1)
    assert (cmd, man) == (1, "TURN_LEFT") and dist == expected
    right = np.column_stack([np.linspace(2, 12, 6), -np.linspace(0, 3, 6)])
    assert L.command_from_future(right)[:2] == (0, "TURN_RIGHT")


def test_command_distance_is_zero_when_already_mid_turn():
    # Lateral offset already past the threshold at the first future point.
    turning = np.column_stack([np.linspace(2, 12, 6), np.linspace(2.5, 8.0, 6)])
    cmd, man, dist = L.command_from_future(turning)
    assert (cmd, man) == (1, "TURN_LEFT") and dist == 0.0


def test_plan_resampling_and_headings():
    traj = Trajectory(np.column_stack([np.arange(1, 9) * 2.0, np.zeros(8)]), dt=0.5)  # 4 m/s straight
    grid = plan_on_grid(traj)
    assert grid.shape == (6, 2) and np.allclose(grid[:, 0], np.arange(1, 7) * 2.0)
    fine = Trajectory(np.column_stack([np.arange(1, 31) * 0.4, np.zeros(30)]), dt=0.1)  # same plan at 10 Hz
    assert np.allclose(plan_on_grid(fine)[:, 0], np.arange(1, 7) * 2.0)
    # A coarse plan (1 s spacing, first point at 10 m) must interpolate its 0.5 s
    # grid from the ego origin, not flat-hold the first waypoint: x(0.5s) = 5, not 10.
    coarse = Trajectory(np.column_stack([np.arange(1, 4) * 10.0, np.zeros(3)]), dt=1.0)
    assert abs(plan_on_grid(coarse)[0, 0] - 5.0) < 1e-9
    yaws = plan_headings(np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]))
    assert np.allclose(yaws, [math.pi / 4, math.pi / 4, math.pi / 4])  # zero step keeps heading


def test_scorer_l2_and_collision_conventions():
    future = np.column_stack([np.arange(1, 7) * 2.0, np.zeros(6)])
    exact = future.copy()
    empty = [np.zeros((0, 5)) for _ in range(6)]
    m = score_plan(exact, future, empty)
    assert m["l2_1s"] == m["l2_2s"] == m["l2_3s"] == 0.0 and m["col_avg"] == 0.0
    # 1 m lateral error everywhere: point values and cumulative means agree
    off = exact + np.array([0.0, 1.0])
    m = score_plan(off, future, empty)
    assert (
        abs(m["l2_avg"] - 1.0) < 1e-9
        and abs(m["l2_uniad_avg"] - 1.0) < 1e-9
        and abs(m["ade_3s"] - 1.0) < 1e-9
    )
    # a parked car sitting where the plan is at 2 s (step 4) and after
    boxes = [np.zeros((0, 5)) for _ in range(3)] + [
        np.array([[8.0 + 2.0 * k, 0.0, 0.0, 4.0, 2.0]]) for k in range(3)
    ]
    m = score_plan(exact, future, boxes)
    assert m["col_1s"] == 0.0 and m["col_2s"] == 1.0 and m["col_3s"] == 1.0
    assert abs(m["col_uniad_2s"] - 0.25) < 1e-9 and abs(m["col_uniad_3s"] - 0.5) < 1e-9
    assert abs(m["col_avg"] - 2.0 / 3.0) < 1e-9


def test_absent_boxes_omit_collision_metrics_rather_than_zeroing_them():
    # None boxes = "not scored" (a run scored outside the process that built the
    # index). Collision keys must be absent, never a false 0 "collision-free".
    future = np.column_stack([np.arange(1, 7) * 2.0, np.zeros(6)])
    m = score_plan(future.copy(), future, None)
    assert m["l2_avg"] == 0.0 and "ade_3s" in m
    assert not any(k.startswith("col_") for k in m)


def test_index_cache_is_rebuilt_when_the_dataset_key_changes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_build(root, version, scene_names, log=print):
        calls["n"] += 1
        return [[f"{version}:{','.join(scene_names)}"]]

    monkeypatch.setattr(L, "build_index", fake_build)
    cache = tmp_path / "idx.pkl"
    root = tmp_path / "data"

    first = L.load_or_build_index(root, "v1.0-trainval", ["a", "b"], cache)
    assert calls["n"] == 1 and cache.exists()
    # Same args: served from cache, no rebuild.
    again = L.load_or_build_index(root, "v1.0-trainval", ["b", "a"], cache)
    assert calls["n"] == 1 and again == first
    # Different scene set: key mismatch -> rebuild.
    L.load_or_build_index(root, "v1.0-trainval", ["a", "b", "c"], cache)
    assert calls["n"] == 2
    # Different version: rebuild again.
    L.load_or_build_index(root, "v1.0-mini", ["a", "b"], cache)
    assert calls["n"] == 3
