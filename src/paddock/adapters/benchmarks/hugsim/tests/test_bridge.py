"""The HUGSIM bridge: camera matrices, the held plan and what HUGSIM is sent. No HUGSIM, no GPU.

``fixtures/bridge_golden.json`` holds what the bridge computed on ``fixtures/camera_rig.yaml`` (a made-up rig
in HUGSIM's layout) as it was when it drove the October campaign whose results are in ``results/hugsim/``:
that version built its rotations with scipy. Matching it to 1e-12 is what says the module here, which has no
scipy, is the same code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.hugsim import bridge
from paddock.policy.protocol import Command

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "bridge_golden.json").read_text())


@pytest.fixture(scope="module")
def rig() -> bridge.Rig:
    return bridge.Rig(str(FIXTURES / "camera_rig.yaml"))


def test_euler_matches_scipy_intrinsic_xyz() -> None:
    spatial = pytest.importorskip("scipy.spatial.transform")
    for rot in ([89.5, -0.7, 89.4], [0.0, 0.0, 0.0], [-12.0, 33.0, 170.0], [92.3, 0.5, -179.5]):
        want = spatial.Rotation.from_euler("XYZ", rot, degrees=True).as_matrix()
        np.testing.assert_allclose(bridge.euler_xyz_deg(rot), want, atol=1e-14)


def test_rig_reproduces_the_campaign_bridge(rig: bridge.Rig) -> None:
    assert rig.cam_yaw == pytest.approx(GOLDEN["cam_yaw"], abs=1e-12)
    np.testing.assert_allclose(rig.cam_xy, GOLDEN["cam_xy"], atol=1e-12)
    assert rig.cam_z == pytest.approx(GOLDEN["cam_z"], abs=1e-12)
    np.testing.assert_allclose(rig.camera2ego["CAM_FRONT"], GOLDEN["camera2ego_front"], atol=1e-12)
    np.testing.assert_allclose(rig.camera2ego["CAM_BACK"], GOLDEN["camera2ego_back"], atol=1e-12)
    np.testing.assert_allclose(rig.lidar2ego, GOLDEN["lidar2ego"], atol=1e-12)
    np.testing.assert_allclose(rig.camera2image["CAM_FRONT"], GOLDEN["camera2image_front"], atol=1e-9)


def test_the_vehicle_frame_points_where_hugsim_drives(rig: bridge.Rig) -> None:
    """After the yaw the front camera looks along x, so "straight" is HUGSIM's direction of travel."""
    front = rig.camera2ego["CAM_FRONT"]
    optical_axis = front[:3, 2]
    assert math.atan2(optical_axis[1], optical_axis[0]) == pytest.approx(0.0, abs=1e-12)


def test_lidar_is_sent_as_the_ego_frame(rig: bridge.Rig) -> None:
    assert np.allclose(rig.calibration()["lidar2ego"], np.eye(4))


def test_vehicle_pose_and_ground(rig: bridge.Rig) -> None:
    T, ground = bridge.vehicle_pose({"ego_box": GOLDEN["ego_box"]}, rig)
    np.testing.assert_allclose(T, GOLDEN["vehicle_pose"], atol=1e-12)
    assert ground == pytest.approx(GOLDEN["ground_z"], abs=1e-12)


def test_plan_sent_to_hugsim_matches_the_campaign_bridge(rig: bridge.Rig) -> None:
    T = np.asarray(GOLDEN["vehicle_pose"])
    held = bridge.HeldPlan(10.0, T, GOLDEN["traj"], 0.5)
    np.testing.assert_allclose(bridge.to_hugsim(held, 10.0, T, rig), GOLDEN["plan_now"], atol=1e-12)
    # a held plan, re-sampled a step later from where the car actually is
    later = bridge.to_hugsim(held, 10.25, np.asarray(GOLDEN["T_later"]), rig)
    np.testing.assert_allclose(later, GOLDEN["plan_later"], atol=1e-12)


def test_a_stop_plan_keeps_its_heading(rig: bridge.Rig) -> None:
    """Steps under MIN_STEP_M carry no heading: a near-stationary plan must not swing the camera point."""
    T = np.asarray(GOLDEN["vehicle_pose"])
    stop = bridge.HeldPlan(10.0, T, [[0.01, -0.03], [0.012, -0.029], [0.03, -0.02], [0.05, -0.02]], 0.5)
    assert np.allclose(stop.yaw, stop.yaw[0])
    np.testing.assert_allclose(bridge.to_hugsim(stop, 10.0, T, rig), GOLDEN["plan_stop"], atol=1e-12)


def test_hugsim_is_sent_the_models_own_horizon(rig: bridge.Rig) -> None:
    T = np.eye(3)
    six = bridge.HeldPlan(0.0, T, [[k, 0.0] for k in range(1, 7)], 0.5)
    ten = bridge.HeldPlan(0.0, T, [[k, 0.0] for k in range(1, 11)], 0.5)
    assert six.n == 6 and len(bridge.to_hugsim(six, 0.0, T, rig)) == 6  # never padded
    assert ten.n == bridge.N_MAX  # cut at 4 s


def test_the_horizon_is_counted_in_time_not_rows(rig: bridge.Rig) -> None:
    """A plan 0.1 s apart holds a 0.5 s waypoint every fifth row: 3 s of it is six, never eight."""
    T = np.eye(3)
    three_s = bridge.HeldPlan(0.0, T, [[0.2 * k, 0.0] for k in range(1, 31)], 0.1)
    four_s = bridge.HeldPlan(0.0, T, [[0.2 * k, 0.0] for k in range(1, 41)], 0.1)
    assert three_s.n == 6 and four_s.n == 8


def test_the_whole_plan_is_read_not_the_six_the_wire_keeps_for_neuroncap() -> None:
    """Drive-JEPA plans 4 s: `trajectory` holds its first six waypoints, `plan` all eight. HUGSIM scores the
    plan it is sent, so it must be sent all eight (22 Oct campaign: every model was sent six)."""
    eight = [[float(k), 0.0, 0.0] for k in range(1, 9)]
    reply = {"trajectory": [w[:2] for w in eight[:6]], "plan": {"waypoints": eight, "dt": 0.5}}
    traj, dt = bridge.plan_of(reply)
    assert traj.shape == (8, 2) and dt == 0.5
    traj, dt = bridge.plan_of({"trajectory": [w[:2] for w in eight[:6]], "dt": 0.5})
    assert traj.shape == (6, 2), "a node without `plan` still drives, on what it sent"
    dense = {"waypoints": [[0.1 * k, 0.0] for k in range(1, 41)], "dt": 0.1}
    traj, dt = bridge.plan_of({"trajectory": [[1.0, 0.0]] * 6, "plan": dense})
    assert traj.shape == (40, 2) and dt == 0.1


def test_a_straight_plan_goes_straight_ahead_in_hugsim_axes(rig: bridge.Rig) -> None:
    """HUGSIM's plan axes are (x right, y forward): a straight plan is x = 0, y increasing."""
    T = np.eye(3)
    held = bridge.HeldPlan(0.0, T, [[2.0 * k, 0.0] for k in range(1, 7)], 0.5)
    plan = bridge.to_hugsim(held, 0.0, T, rig)
    np.testing.assert_allclose(plan[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(plan[:, 1], [2.0 * k for k in range(1, 7)], atol=1e-12)


def test_hugsim_commands_are_paddock_commands() -> None:
    """The bridge forwards info['command'] unchanged: HUGSIM 0 right, 1 left, 2 straight."""
    assert (Command(0), Command(1), Command(2)) == (Command.RIGHT, Command.LEFT, Command.STRAIGHT)


@pytest.mark.parametrize(("hz", "every"), [(None, 1), (0.0, 1), (4.0, 1), (2.0, 2), (1.0, 4), (10.0, 1)])
def test_call_interval_follows_the_models_rate(hz: float | None, every: int) -> None:
    assert bridge.infer_every(hz) == every
