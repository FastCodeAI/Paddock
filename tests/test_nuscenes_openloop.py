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
    assert (cmd, man) == (1, "TURN_LEFT") and dist == round(math.hypot(12, 3), 1)
    right = np.column_stack([np.linspace(2, 12, 6), -np.linspace(0, 3, 6)])
    assert L.command_from_future(right)[:2] == (0, "TURN_RIGHT")


def test_plan_resampling_and_headings():
    traj = Trajectory(np.column_stack([np.arange(1, 9) * 2.0, np.zeros(8)]), dt=0.5)  # 4 m/s straight
    grid = plan_on_grid(traj)
    assert grid.shape == (6, 2) and np.allclose(grid[:, 0], np.arange(1, 7) * 2.0)
    fine = Trajectory(np.column_stack([np.arange(1, 31) * 0.4, np.zeros(30)]), dt=0.1)  # same plan at 10 Hz
    assert np.allclose(plan_on_grid(fine)[:, 0], np.arange(1, 7) * 2.0)
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
