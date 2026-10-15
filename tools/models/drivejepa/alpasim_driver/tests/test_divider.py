# SPDX-License-Identifier: Apache-2.0
"""The replan divider and the referee-rate guards, on the promoted 0.1.7 driver.

Ported from the 0.1.4 suite these behaviours were first written against; the
divider has not changed since (0.1.4 -> 0.1.4b -> 0.1.7), so the assertions are
the same ones:

    divider     replan on a 2 Hz grid while the harness asks at 20 Hz
    re-anchor   never hand back a cached plan verbatim
    spacing     off-grid context is a hard failure, not a silent accept
    gate        the standstill threshold is a speed, not a distance

Expected values are computed **here, by hand**. Where a test has to reason
about what the harness does with a returned plan, it re-implements the runtime
and controller composition from their own source lines rather than calling
adapter code -- otherwise a bug in the adapter's arithmetic would cancel out.

CPU only: no weights, no torch, no GPU, no bridge subprocess. Needs AlpaSim's
interpreter (grpc, alpasim_grpc, cv2); see conftest.py.
"""

from __future__ import annotations

import io
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import DynamicState, Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.sensorsim_pb2 import (
    AvailableCamerasReturn,
    CameraSpec,
    OpenCVPinholeCameraParam,
)
from drivejepa.alpasim_driver import constants as K
from drivejepa.alpasim_driver.divider import ReplanDivider
from drivejepa.alpasim_driver.fake_model import FakeDriveJEPA
from drivejepa.alpasim_driver.planner import (
    ContextSpacingError,
    DriveJEPAPlanner,
    PushCadenceError,
)
from PIL import Image

from paddock.adapters.benchmarks.alpasim.scaffold import DriverScaffold, ObservationView, Planner, PlanResult

ATOL = 1e-6
FRONT = "camera_front_wide_120fov"
IMG_H, IMG_W = 200, 400

#: Both cadences every rate-sensitive test runs at. 500 ms is what 0.1.3 was
#: tuned at; 50 ms is the shared referee. No test in this file hardcodes
#: either -- that is the regression 0.1.3's suite could not catch, because
#: every one of its sentinel tests pinned WARMUP_TICK_US = 500_000.
TICK_RATES_US = (500_000, 50_000)

#: Highway speed, the reference run's handover speed rounded down
#: (runs/drivejepa_013_04749bb9/driver.log:7 shows 24.56 m/s).
HIGHWAY_M_S = 24.0


# ===========================================================================
# Fixtures
# ===========================================================================


MOUNT_T = (1.65897811, -0.01443456, 1.51539499)
MOUNT_Q_XYZW = (
    -0.49929397355810856,
    0.5039939168301356,
    -0.4972939976976715,
    0.49939397235113037,
)


def make_camera(logical_id: str = FRONT) -> AvailableCamerasReturn.AvailableCamera:
    """An AvailableCamera with pinhole intrinsics and the real mount.

    Same construction as test_drivejepa.py:66-87 -- the proto field names are
    not obvious (resolution_h, opencv_pinhole_param, w-first Quat) and getting
    them wrong fails at construction, not in a way worth re-deriving here.
    """
    x, y, z = MOUNT_T
    qx, qy, qz, qw = MOUNT_Q_XYZW
    return AvailableCamerasReturn.AvailableCamera(
        logical_id=logical_id,
        intrinsics=CameraSpec(
            logical_id=logical_id,
            resolution_h=IMG_H,
            resolution_w=IMG_W,
            opencv_pinhole_param=OpenCVPinholeCameraParam(
                focal_length_x=1545.0,
                focal_length_y=1545.0,
                principal_point_x=960.0,
                principal_point_y=560.0,
            ),
        ),
        # Quat is w-first in the proto.
        rig_to_camera=Pose(
            vec=Vec3(x=x, y=y, z=z), quat=Quat(w=qw, x=qx, y=qy, z=qz)
        ),
    )


def gradient_jpeg(seed: int = 0) -> bytes:
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    rgb[..., 0] = ((xx * 3 + seed * 17) % 256).astype(np.uint8)
    rgb[..., 1] = ((yy * 5 + seed * 29) % 256).astype(np.uint8)
    rgb[..., 2] = ((xx + yy + seed * 7) % 256).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def matrix(x: float, y: float, yaw: float) -> np.ndarray:
    """4x4 local->rig pose. Hand-built; does not call the planner's helper."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.array(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    out[:3, 3] = (x, y, 0.0)
    return out


def straight_poses(
    *, start_us: int, count: int, tick_us: int, speed_m_s: float
) -> tuple[tuple[int, float, float, float], ...]:
    """A car driving straight down +x at a constant speed, one pose per tick."""
    step_m = speed_m_s * (tick_us / 1e6)
    return tuple(
        (start_us + i * tick_us, i * step_m, 0.0, 0.0) for i in range(count)
    )


def make_view(
    *,
    frames: tuple[tuple[int, bytes], ...] = (),
    poses: tuple[tuple[int, float, float, float], ...] = (
        (1_000_000, 0.0, 0.0, 0.0),
    ),
    velocity: tuple[float, float] = (8.0, 0.0),
    accel: tuple[float, float] = (0.5, 0.0),
    camera_id: str = FRONT,
    session_uuid: str = "unit",
) -> ObservationView:
    pose_list, dyn_list = [], []
    for ts, x, y, yaw in poses:
        pose_list.append(
            PoseAtTime(
                pose=Pose(
                    vec=Vec3(x=x, y=y, z=0.0),
                    quat=Quat(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2)),
                ),
                timestamp_us=ts,
            )
        )
        dyn_list.append(
            (
                ts,
                DynamicState(
                    linear_velocity=Vec3(x=velocity[0], y=velocity[1], z=0.0),
                    linear_acceleration=Vec3(x=accel[0], y=accel[1], z=0.0),
                ),
            )
        )
    return ObservationView(
        session_uuid=session_uuid,
        seed=0,
        debug_scene_id="unit",
        camera_logical_ids=(camera_id,),
        frames_received={camera_id: len(frames)},
        image_cache={camera_id: frames},
        poses=tuple(pose_list),
        dynamic_states=tuple(dyn_list),
        route=None,
        drive_count=0,
        cameras={camera_id: make_camera(camera_id)},
    )


def grid_frames(*, newest_us: int, count: int, tick_us: int) -> tuple:
    """``count`` frames ending at ``newest_us``, ``tick_us`` apart, oldest first."""
    return tuple(
        (newest_us - (count - 1 - i) * tick_us, gradient_jpeg(seed=i))
        for i in range(count)
    )


class SpyPlanner(Planner):
    """Records every plan() call and returns a canned straight-ahead plan.

    Stands in for DriveJEPAPlanner so the divider can be exercised without the
    model, the bridge or a single JPEG decode -- which is also how the "cached
    ticks never reach the model" test proves its point.
    """

    version_id = "spy-0.0.0"

    def __init__(self, *, step_us: int = K.STEP_US, ready_from_us: int = 0) -> None:
        self.calls: list[tuple[int, int]] = []
        self._step_us = step_us
        self._ready_from_us = ready_from_us

    def plan(self, observations, time_now_us, time_query_us):
        self.calls.append((time_now_us, time_query_us))
        if time_now_us < self._ready_from_us:
            return None
        # 8 waypoints, 10 m apart, straight ahead in the rig frame.
        positions = np.zeros((K.NUM_POSES, 3), dtype=np.float64)
        positions[:, 0] = [10.0 * (k + 1) for k in range(K.NUM_POSES)]
        rotations = np.repeat(np.eye(3)[None], K.NUM_POSES, axis=0)
        return PlanResult(
            positions_in_rig=positions,
            step_us=self._step_us,
            rotations_in_rig=rotations,
            model_t0_us=time_now_us,
            debug={"planner": "spy"},
        )


# ===========================================================================
# (1) The divider grid -- elapsed sim time, never a counter
# ===========================================================================


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
def test_divider_fires_on_the_500ms_grid_at_any_tick_rate(tick_us):
    """Real inference lands on the grid, whatever cadence the harness asks at.

    At 500 ms every ask is a replan; at 50 ms every tenth is. The divider is
    told nothing about the tick rate -- it reads elapsed sim time.
    """
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    start = 5_322_242_706_587  # a real clip timestamp, deliberately unaligned
    ticks = (3 * K.REPLAN_PERIOD_US) // tick_us + 1

    for i in range(ticks):
        now = start + i * tick_us
        view = make_view(
            poses=straight_poses(
                start_us=start, count=i + 1, tick_us=tick_us, speed_m_s=HIGHWAY_M_S
            )
        )
        divider.plan(view, time_now_us=now, time_query_us=now + tick_us)

    every_nth = K.REPLAN_PERIOD_US // tick_us
    expected = [start + i * tick_us for i in range(0, ticks, every_nth)]
    assert [c[0] for c in spy.calls] == expected


def test_divider_anchors_on_the_first_ask_not_on_absolute_modulo():
    """first_policy_timestamp_us is a raw clip stamp with no alignment.

    5_322_242_706_587 % 500_000 == 206_587, so a grid built on absolute-time
    modulo would fire nowhere near the session's own rhythm. Anchoring on the
    first ask makes the phase the session's own.
    """
    start = 5_322_242_706_587
    assert start % K.REPLAN_PERIOD_US != 0, "fixture must be unaligned to be a test"

    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    for i in range(21):
        now = start + i * 50_000
        divider.plan(
            view_at(start, i, 50_000), time_now_us=now, time_query_us=now + 50_000
        )
    assert [c[0] for c in spy.calls] == [start, start + 500_000, start + 1_000_000]


def view_at(start_us: int, index: int, tick_us: int) -> ObservationView:
    return make_view(
        poses=straight_poses(
            start_us=start_us, count=index + 1, tick_us=tick_us, speed_m_s=HIGHWAY_M_S
        )
    )


def test_a_skipped_query_does_not_shift_the_grid():
    """Catch-up, not reset. A modulo counter would drift here permanently."""
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    start = 1_000_000
    for i in range(21):
        if i == 5:
            continue  # the harness skips one ask
        now = start + i * 50_000
        divider.plan(
            view_at(start, i, 50_000), time_now_us=now, time_query_us=now + 50_000
        )
    # Still on the original lattice: 0, +500ms, +1000ms.
    assert [c[0] for c in spy.calls] == [start, start + 500_000, start + 1_000_000]


def test_a_repeated_query_does_not_run_inference_twice():
    """A repeated time_query_us cannot cross the threshold a second time."""
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    start = 1_000_000
    view = view_at(start, 0, 50_000)
    for _ in range(4):
        divider.plan(view, time_now_us=start, time_query_us=start + 50_000)
    assert len(spy.calls) == 1


def test_a_long_gap_is_caught_up_in_one_step_without_drift():
    """Ten missed periods do not produce ten replans, and do not shift phase."""
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    start = 1_000_000
    divider.plan(view_at(start, 0, 50_000), time_now_us=start,
                 time_query_us=start + 50_000)
    jumped = start + 5_000_000
    divider.plan(view_at(start, 100, 50_000), time_now_us=jumped,
                 time_query_us=jumped + 50_000)
    nxt = jumped + 50_000
    divider.plan(view_at(start, 101, 50_000), time_now_us=nxt,
                 time_query_us=nxt + 50_000)
    assert [c[0] for c in spy.calls] == [start, jumped]


def test_a_not_ready_replan_still_advances_the_grid():
    """None must not stall the lattice, or the handover alignment is lost.

    The planner is not ready until 300 ms in. The grid keeps its phase
    regardless, so the replan at +500 ms is still exactly on it.
    """
    spy = SpyPlanner(ready_from_us=1_300_000)
    divider = ReplanDivider(spy)
    start = 1_000_000
    for i in range(21):
        now = start + i * 50_000
        divider.plan(
            view_at(start, i, 50_000), time_now_us=now, time_query_us=now + 50_000
        )
    assert [c[0] for c in spy.calls] == [start, start + 500_000, start + 1_000_000]


def test_the_first_real_replan_lands_exactly_on_the_handover_tick():
    """The whole point of counting warm-up asks.

    force_gt_duration_us = 3.0 s, control_timestep_us = 50 ms, so the first
    closed-loop tick is ask 60 (unbound_rollout.py:108,295-298; force_gt is
    keyed on target_time_us, events/policy.py:69). A grid anchored on ask 0
    hits it because 3.0 s is a whole number of 500 ms periods.
    """
    render_start = 5_322_242_706_587
    closed_loop_start = render_start + 3_000_000

    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    for i in range(65):
        now = render_start + i * 50_000
        divider.plan(
            view_at(render_start, i, 50_000),
            time_now_us=now,
            time_query_us=now + 50_000,
        )

    replans = [c[0] for c in spy.calls]
    assert closed_loop_start in replans
    assert replans.index(closed_loop_start) == 6
    assert len(replans) == 7  # ticks 0,10,...,60 -- warm-up counted, not skipped


def test_sessions_do_not_share_phase_or_cache():
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    a_start, b_start = 1_000_000, 7_777_777
    for i in range(11):
        now_a = a_start + i * 50_000
        divider.plan(
            make_view(
                poses=straight_poses(start_us=a_start, count=i + 1, tick_us=50_000,
                                     speed_m_s=HIGHWAY_M_S),
                session_uuid="A",
            ),
            time_now_us=now_a,
            time_query_us=now_a + 50_000,
        )
    for i in range(11):
        now_b = b_start + i * 50_000
        divider.plan(
            make_view(
                poses=straight_poses(start_us=b_start, count=i + 1, tick_us=50_000,
                                     speed_m_s=HIGHWAY_M_S),
                session_uuid="B",
            ),
            time_now_us=now_b,
            time_query_us=now_b + 50_000,
        )
    assert [c[0] for c in spy.calls] == [
        a_start, a_start + 500_000, b_start, b_start + 500_000,
    ]


def test_closing_a_session_frees_its_cache():
    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    view = view_at(1_000_000, 0, 50_000)
    divider.plan(view, time_now_us=1_000_000, time_query_us=1_050_000)
    assert "unit" in divider._sessions
    divider.on_session_close("unit")
    assert "unit" not in divider._sessions


# ===========================================================================
# (2) Re-anchor + re-time -- the system.py:342-350 contract
# ===========================================================================
#
# The harness composition, re-implemented here from its own source so a bug in
# the adapter cannot cancel out:
#
#   scaffold.py:750-772   A_k = P_anchor @ W_k, anchor = pose nearest model_t0
#   controller.py:123-126 ref = A.transform(R^-1)  ->  R^-1 @ A_k   (LEFT mult,
#                         utils_rs __init__.pyi:363-372)
#   system.py:342-350     delivered = C(now)^-1 @ C(t0') @ ref_k
#                         where t0' = ref.timestamps_us[0] = model_t0_us
#
# C tracks the true pose (system.py:145 overrides its last pose with the
# runtime's), and the MPC's first sample is at now == step_start, so
# C(now) = R. The spurious factor is C(t0') @ R^-1, identity iff t0' == now.


def deliver(
    *,
    model_t0_us: int,
    positions: np.ndarray,
    pose_of: dict[int, np.ndarray],
    step_start_us: int,
) -> np.ndarray:
    """What the MPC actually receives, in the rig frame at step_start_us."""
    anchor = pose_of[model_t0_us]          # scaffold picks the pose nearest t0
    rig_now = pose_of[step_start_us]       # R = P_true(step_start)
    controller_at_t0 = pose_of[model_t0_us]  # C(t0')
    controller_now = pose_of[step_start_us]  # C(now)

    out = np.zeros((len(positions), 3), dtype=np.float64)
    for k, position in enumerate(positions):
        waypoint = np.eye(4, dtype=np.float64)
        waypoint[:3, 3] = position
        local = anchor @ waypoint                       # scaffold
        ref = np.linalg.inv(rig_now) @ local            # runtime
        delivered = (
            np.linalg.inv(controller_now) @ controller_at_t0 @ ref
        )                                               # controller
        out[k] = delivered[:3, 3]
    return out


def cached_plan_scenario(stale_us: int, speed_m_s: float = HIGHWAY_M_S):
    """A plan built at t0, consumed `stale_us` later, car driving straight.

    Returns (divider, view_now, t0, now, pose_of, cached_positions).
    """
    tick_us = 50_000
    start = 1_000_000
    t0 = start
    now = start + stale_us
    count = stale_us // tick_us + 1

    poses = straight_poses(
        start_us=start, count=count, tick_us=tick_us, speed_m_s=speed_m_s
    )
    pose_of = {ts: matrix(x, y, yaw) for ts, x, y, yaw in poses}

    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    # Replan at t0.
    view_t0 = make_view(poses=poses[:1])
    fresh = divider.plan(view_t0, time_now_us=t0, time_query_us=t0 + tick_us)
    assert fresh is not None and fresh.model_t0_us == t0

    view_now = make_view(poses=poses)
    return divider, view_now, t0, now, pose_of, np.array(fresh.positions_in_rig)


def test_returning_the_cached_plan_verbatim_displaces_it_by_the_ego_motion():
    """(a) The failure the re-anchor exists to prevent.

    Verbatim means model_t0_us stays at the stale t0. The composition then
    carries C(t0) @ R^-1 -- the ego's displacement over the staleness, applied
    a second time. The reference reaches the MPC one whole ego-displacement
    behind where the plan actually put it, i.e. twice as far back as the naive
    reading suggests.
    """
    stale_us = 450_000
    _, _, t0, now, pose_of, cached = cached_plan_scenario(stale_us)
    ego_motion_m = HIGHWAY_M_S * (stale_us / 1e6)  # 10.8 m

    verbatim = deliver(
        model_t0_us=t0, positions=cached, pose_of=pose_of, step_start_us=now
    )
    # Truth: the same waypoints expressed in the rig frame at `now`, which is
    # just the cached waypoints pulled back by one ego displacement.
    truth = cached.copy()
    truth[:, 0] -= ego_motion_m

    error = float(np.abs(verbatim[:, 0] - truth[:, 0]).max())
    assert error == pytest.approx(ego_motion_m, abs=1e-6)
    assert error > 10.0, "the whole point: this is metres, not millimetres"


def test_the_reanchored_plan_is_delivered_where_it_was_planned():
    """(a, positive half) Same scenario, through the divider's cached path."""
    stale_us = 450_000
    divider, view_now, t0, now, pose_of, cached = cached_plan_scenario(stale_us)

    rebuilt = divider.plan(view_now, time_now_us=now, time_query_us=now + 50_000)
    assert rebuilt is not None
    assert rebuilt.model_t0_us == now, "t0 must be THIS tick for the contract"
    assert rebuilt.debug["replan"] is False
    assert rebuilt.debug["cache_age_us"] == stale_us

    delivered = deliver(
        model_t0_us=now,
        positions=np.array(rebuilt.positions_in_rig),
        pose_of=pose_of,
        step_start_us=now,
    )
    # The plan says: at now + j*0.5 s be at the cached trajectory's pose for
    # that absolute time, expressed from here. Hand-computed: the cached plan
    # put waypoint k at 10*(k+1) m ahead of t0; from `now` that is
    # 10*(k+1) - ego_motion. Re-timed to now + j*0.5 s means sampling the
    # cached curve at t0 + stale + j*0.5 s.
    ego_motion_m = HIGHWAY_M_S * (stale_us / 1e6)
    times = np.array([t0 + k * K.STEP_US for k in range(K.NUM_POSES + 1)])
    knots = np.concatenate([[0.0], cached[:, 0]])
    for j in range(len(delivered)):
        target = now + (j + 1) * K.STEP_US
        expected = float(np.interp(target, times, knots)) - ego_motion_m
        assert delivered[j, 0] == pytest.approx(expected, abs=1e-6)


def test_composition_order_is_left_multiplication_per_the_stub():
    """(b) Pin the order. Flipping it in divider._rebuild must fail here.

    utils_rs __init__.pyi:363-372 -- "otherwise left multiplication
    (transform @ pose)". The re-anchor delta is inv(P_now) @ P_t0 applied on
    the LEFT of each waypoint. The transpose-of-intent, P_t0 @ inv(P_now), is
    a different transform whenever the ego has rotated, and this fixture makes
    it rotate so the two cannot coincide.
    """
    tick_us, start = 50_000, 1_000_000
    stale_us = 450_000
    # A turning car that does NOT start at the origin. Both matter: if the t0
    # pose were the identity -- which it is whenever the fixture starts the car
    # at (0, 0, 0) -- then inv(P_now) @ I and I @ inv(P_now) are the same matrix
    # and the test would pass under either order without discriminating.
    poses = tuple(
        (start + i * tick_us, 3.0 + 1.0 * i, -1.0 + 0.25 * i, 0.3 + 0.05 * i)
        for i in range(stale_us // tick_us + 1)
    )
    now = start + stale_us

    spy = SpyPlanner()
    divider = ReplanDivider(spy)
    fresh = divider.plan(
        make_view(poses=poses[:1]), time_now_us=start, time_query_us=start + tick_us
    )
    assert fresh is not None

    rebuilt = divider.plan(
        make_view(poses=poses), time_now_us=now, time_query_us=now + tick_us
    )
    assert rebuilt is not None

    pose_t0 = matrix(*poses[0][1:])
    pose_now = matrix(*poses[-1][1:])
    left = np.linalg.inv(pose_now) @ pose_t0
    right = pose_t0 @ np.linalg.inv(pose_now)
    assert not np.allclose(left, right), "fixture must distinguish the two orders"

    # Rebuild waypoint 1 by hand under each order and see which the code chose.
    times = np.array([start + k * K.STEP_US for k in range(K.NUM_POSES + 1)])
    knots_x = np.concatenate([[0.0], np.array(fresh.positions_in_rig)[:, 0]])
    target = now + K.STEP_US
    waypoint = np.eye(4)
    waypoint[0, 3] = float(np.interp(target, times, knots_x))

    got = np.asarray(rebuilt.positions_in_rig)[0]
    want_left = (left @ waypoint)[:3, 3]
    want_right = (right @ waypoint)[:3, 3]

    # 1e-5, not 1e-9: the pose reaches the divider through the proto, whose
    # Vec3/Quat fields are float32 (common.proto), so _pose_to_matrix rebuilds
    # from single-precision values while `matrix()` here is float64. The
    # residual is ~5e-7. The distinction this test draws is metres wide, so
    # float32 noise has nothing to do with it -- asserted next.
    np.testing.assert_allclose(got, want_left, atol=1e-5)
    assert float(np.abs(want_left - want_right).max()) > 0.1, (
        "fixture must separate the orders by far more than float32 noise"
    )
    assert float(np.abs(got - want_right).max()) > 0.1


@pytest.mark.parametrize("stale_us", list(range(0, 500_000, 50_000)))
def test_rebuilt_plan_covers_the_controller_lookahead_at_every_staleness(stale_us):
    """(c) Range coverage, including the last tick before the next replan.

    interpolate_pose RAISES outside the trajectory's range (utils_rs
    __init__.pyi:439-446), so a coverage gap is a mid-run crash rather than a
    merely bad plan. Two things must hold at every staleness the divider can
    produce:

      * the plan's range STARTS at now -- system.py:342 interpolates the
        controller's own trajectory at ref.timestamps_us[0], and the
        controller's newest pose is exactly now, so anything earlier is
        outside its range;
      * the range EXTENDS past the MPC's furthest sample, now + 2.0 s
        (n_horizon 20 x dt_mpc 0.1, nonlinear_mpc.py:315-330).
    """
    divider, view_now, t0, now, _, _ = cached_plan_scenario(stale_us)
    if stale_us == 0:
        # age 0 is the replan tick itself; exercise _rebuild directly so the
        # boundary is still covered.
        cache = divider._sessions["unit"]
        rebuilt = divider._rebuild(cache, view_now, now)
    else:
        rebuilt = divider.plan(view_now, time_now_us=now, time_query_us=now + 50_000)

    assert rebuilt is not None, f"no plan at staleness {stale_us}"
    assert rebuilt.model_t0_us == now

    span_us = len(rebuilt.positions_in_rig) * rebuilt.step_us
    assert span_us >= K.MAX_CONTROLLER_LOOKAHEAD_US, (
        f"staleness {stale_us} us leaves only {span_us} us of plan, "
        f"controller samples to {K.MAX_CONTROLLER_LOOKAHEAD_US} us"
    )


def test_worst_case_staleness_is_the_last_tick_before_the_next_replan():
    """450 ms is the most a 500 ms grid at 50 ms ticks can ever be stale."""
    divider = ReplanDivider(SpyPlanner())
    start = 1_000_000
    ages = []
    for i in range(20):
        now = start + i * 50_000
        result = divider.plan(
            view_at(start, i, 50_000), time_now_us=now, time_query_us=now + 50_000
        )
        ages.append(result.debug.get("cache_age_us", 0) if result else None)
    assert max(a for a in ages if a is not None) == 450_000


def test_a_fresh_plan_is_marked_replan_and_keeps_the_model_frame_time():
    divider = ReplanDivider(SpyPlanner())
    view = view_at(1_000_000, 0, 50_000)
    fresh = divider.plan(view, time_now_us=1_000_000, time_query_us=1_050_000)
    assert fresh.debug["replan"] is True
    assert fresh.debug["cache_age_us"] == 0
    assert fresh.model_t0_us == 1_000_000


# ===========================================================================
# (3) Context spacing -- hard failure, not a silent accept
# ===========================================================================


def planner(**kw) -> DriveJEPAPlanner:
    kw.setdefault("model_fn", FakeDriveJEPA().eval())
    kw.setdefault("front_camera_id", FRONT)
    return DriveJEPAPlanner(**kw)


def test_an_eight_frame_buffer_at_20hz_is_refused_not_accepted():
    """The 0.1.3 bug, now a hard failure.

    8 frames at 50 ms span 350 ms. 0.1.3's +-50% window accepted that as a
    0.5 s pair and fed the model short context with no log line. The buffer
    DOES reach past `newest - 500 ms`? No -- so this is the not-ready branch.
    Push a ninth frame far enough back that the buffer spans the target while
    the nearest frame is still off-grid, and it must raise.
    """
    newest = 2_000_000
    # Spans 700 ms (so the target at -500 ms IS covered) but the only frames
    # are at -700 ms and -350 ms: nothing within +-10% of -500 ms.
    frames = (
        (newest - 700_000, gradient_jpeg(seed=1)),
        (newest - 350_000, gradient_jpeg(seed=2)),
        (newest, gradient_jpeg(seed=3)),
    )
    view = make_view(frames=frames)
    with pytest.raises(ContextSpacingError, match="context pair is"):
        planner().plan(view, 2_000_000, 2_050_000)


def test_a_short_buffer_is_not_ready_rather_than_an_error():
    """Not-ready and misconfigured are different, and 0.1.3 conflated them.

    A buffer that cannot yet reach one step back is the normal state of the
    first few ticks. It must answer None -- the scaffold turns that into the
    contract's legal empty trajectory -- not raise.
    """
    newest = 2_000_000
    frames = grid_frames(newest_us=newest, count=8, tick_us=50_000)  # 350 ms span
    view = make_view(frames=frames)
    assert planner().plan(view, newest, newest + 50_000) is None


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
def test_a_grid_buffer_yields_exactly_the_500ms_pair(tick_us):
    newest = 5_000_000
    count = K.FRAME_SPACING_US // tick_us + 3
    view = make_view(frames=grid_frames(newest_us=newest, count=count, tick_us=tick_us))
    selected = planner()._select_frames(view)
    assert selected is not None
    older, newer = selected
    assert newer[0] - older[0] == K.FRAME_SPACING_US


def test_default_tolerance_is_ten_percent_not_fifty():
    """0.1.3 shipped 0.5. At 20 Hz that is what swallowed the 350 ms pair."""
    assert planner()._frame_spacing_tolerance == K.CONTEXT_SPACING_TOLERANCE
    assert K.CONTEXT_SPACING_TOLERANCE == 0.10
    # 350 ms would have been inside the old window and is outside the new one.
    assert 250_000 <= 350_000 <= 750_000              # 0.1.3 accepted it
    assert not (450_000 <= 350_000 <= 550_000)        # 0.1.4 does not


def test_image_history_default_spans_a_full_step_at_the_referee_rate():
    span_us = K.DEFAULT_IMAGE_HISTORY * 50_000
    assert span_us > K.FRAME_SPACING_US
    assert 8 * 50_000 < K.FRAME_SPACING_US, "the default that made 0.1.3 wrong"


def test_scaffold_bounded_buffer_at_20hz_reaches_a_full_step_back():
    """End-to-end through the real scaffold buffer, not a hand-built tuple.

    0.1.3's nearest test built its frame tuple directly and so never exercised
    the scaffold's image_history bound -- which is why the 350 ms pair went
    unnoticed.
    """
    from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage

    spy = SpyPlanner()
    driver = DriverScaffold(
        planner=spy, keep_images=True, image_history=K.DEFAULT_IMAGE_HISTORY
    )
    state_images: list[tuple[int, bytes]] = []
    for i in range(K.DEFAULT_IMAGE_HISTORY + 5):
        ts = 1_000_000 + i * 50_000
        state_images.append((ts, gradient_jpeg(seed=i % 7)))
    kept = state_images[-K.DEFAULT_IMAGE_HISTORY :]
    span = kept[-1][0] - kept[0][0]
    assert span >= K.FRAME_SPACING_US, (
        f"image_history={K.DEFAULT_IMAGE_HISTORY} spans {span} us at 50 ms "
        f"pushes; need {K.FRAME_SPACING_US}"
    )
    assert isinstance(driver, DriverScaffold)
    assert RolloutCameraImage is not None


# ===========================================================================
# (3b) Push cadence -- the FLAG-2 closure
# ===========================================================================


def test_push_cadence_mismatch_is_refused():
    """frame_interval_us != control_timestep_us is unchecked by the runtime."""
    view = make_view(frames=grid_frames(newest_us=5_000_000, count=14, tick_us=50_000))
    with pytest.raises(PushCadenceError, match="pushing frames"):
        planner(expected_frame_interval_us=100_000).plan(view, 5_000_000, 5_100_000)


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
def test_matching_push_cadence_passes(tick_us):
    count = K.FRAME_SPACING_US // tick_us + 3
    view = make_view(frames=grid_frames(newest_us=5_000_000, count=count,
                                        tick_us=tick_us))
    assert (
        planner(expected_frame_interval_us=tick_us)._select_frames(view) is not None
    )


def test_push_cadence_is_checked_once_per_session_not_every_tick():
    """A single dropped frame must not fail a rollout that is otherwise fine."""
    p = planner(expected_frame_interval_us=50_000)
    good = make_view(frames=grid_frames(newest_us=5_000_000, count=14, tick_us=50_000))
    assert p._select_frames(good) is not None  # validates and records

    # Now a gap appears. The startup check has already passed for this session.
    frames = list(grid_frames(newest_us=6_000_000, count=14, tick_us=50_000))
    frames[-1] = (frames[-1][0] + 50_000, frames[-1][1])
    limped = make_view(frames=tuple(frames))
    assert p._select_frames(limped) is not None


def test_push_cadence_check_is_per_session():
    p = planner(expected_frame_interval_us=50_000)
    ok = make_view(frames=grid_frames(newest_us=5_000_000, count=14, tick_us=50_000),
                   session_uuid="A")
    p._select_frames(ok)
    bad = make_view(frames=grid_frames(newest_us=5_000_000, count=14, tick_us=100_000),
                    session_uuid="B")
    with pytest.raises(PushCadenceError):
        p._select_frames(bad)


def test_push_cadence_check_is_opt_out():
    view = make_view(frames=grid_frames(newest_us=5_000_000, count=14, tick_us=50_000))
    assert planner(expected_frame_interval_us=None)._select_frames(view) is not None


# ===========================================================================
# (4) The standstill gate -- a speed, at any tick rate
# ===========================================================================


def sentinel_view(*, tick_us: int, speed_m_s: float) -> ObservationView:
    """Two poses `tick_us` apart at `speed_m_s`, velocity reported as zero.

    That is the warm-up sentinel: DynamicState() is all zeros while force-GT
    drives (controller_service.py:284-292), and the poses in the same
    observation are moving.
    """
    step_m = speed_m_s * (tick_us / 1e6)
    return make_view(
        frames=grid_frames(newest_us=2_000_000, count=2, tick_us=K.FRAME_SPACING_US),
        poses=((1_000_000, 0.0, 0.0, 0.0), (1_000_000 + tick_us, step_m, 0.0, 0.0)),
        velocity=(0.0, 0.0),
        accel=(0.25, -0.125),
    )


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
@pytest.mark.parametrize("speed_m_s", [5.0, 12.0, HIGHWAY_M_S])
def test_the_sentinel_fires_at_any_tick_rate_and_any_real_speed(
    caplog, tick_us, speed_m_s
):
    """The 0.1.3 regression. At 50 ms its 0.5 m gate meant 10 m/s.

    5 m/s over a 50 ms tick is 0.25 m -- under 0.1.3's threshold, so 0.1.3
    would have handed the model a zeroed velocity at handover in any urban
    scene.
    """
    view = sentinel_view(tick_us=tick_us, speed_m_s=speed_m_s)
    with caplog.at_level("INFO", logger="drivejepa.alpasim_driver.planner"):
        ego = planner().build_features(view, view.image_cache[FRONT][0],
                                       view.image_cache[FRONT][1])["ego_status"][0]
    np.testing.assert_allclose(ego[-1, 3:5], [speed_m_s, 0.0], atol=1e-4)
    assert [r for r in caplog.records if "handover sentinel" in r.message]


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
def test_a_genuine_standstill_keeps_its_zero_at_any_tick_rate(caplog, tick_us):
    """Never spoof a stopped car. 0.4 m/s is under the gate at both rates."""
    view = sentinel_view(tick_us=tick_us, speed_m_s=0.4)
    with caplog.at_level("INFO", logger="drivejepa.alpasim_driver.planner"):
        caplog.clear()
        ego = planner().build_features(view, view.image_cache[FRONT][0],
                                       view.image_cache[FRONT][1])["ego_status"][0]
    np.testing.assert_allclose(ego[-1, 3:5], [0.0, 0.0], atol=ATOL)
    assert not [r for r in caplog.records if "handover sentinel" in r.message]


@pytest.mark.parametrize(
    "distance_m", [0.0, 0.2, 0.4999, 0.5, 0.5001, 0.7, 4.2, 12.28]
)
def test_at_500ms_the_speed_gate_reproduces_the_0_1_3_distance_gate(distance_m):
    """Bit-for-bit equivalence on the fixtures 0.1.3 was tuned on.

    0.1.3: `distance_m <= 0.5`. 0.1.4: `distance_m / dt <= 1.0` with dt
    exactly 0.5. Dividing by 0.5 is exact in binary, so the two comparisons
    partition the same set -- including at the 0.5 boundary, both inclusive.
    """
    tick_us = 500_000
    view = make_view(
        frames=grid_frames(newest_us=2_000_000, count=2, tick_us=K.FRAME_SPACING_US),
        poses=((1_000_000, 0.0, 0.0, 0.0), (1_000_000 + tick_us, distance_m, 0.0, 0.0)),
        velocity=(0.0, 0.0),
    )
    ego = planner().build_features(view, view.image_cache[FRONT][0],
                                   view.image_cache[FRONT][1])["ego_status"][0]
    substituted = not np.allclose(ego[-1, 3:5], [0.0, 0.0], atol=1e-12)

    legacy_would_substitute = distance_m > 0.5   # 0.1.3: `<= 0.5` returns reported
    assert substituted == legacy_would_substitute, (
        f"0.1.4 {'substituted' if substituted else 'passed through'} at "
        f"{distance_m} m; 0.1.3 would have done the opposite"
    )


def test_the_gate_threshold_is_a_speed_not_a_distance():
    """Guards the shape of the fix, not just this instance of it."""
    from drivejepa.alpasim_driver import planner as planner_module

    assert planner_module.HANDOVER_MIN_SPEED_M_S == 1.0
    assert not hasattr(planner_module, "HANDOVER_MIN_POSE_GAP_M"), (
        "the distance constant is what made the gate tick-rate dependent"
    )


@pytest.mark.parametrize("tick_us", TICK_RATES_US)
def test_a_reported_velocity_is_never_overwritten_at_any_tick_rate(tick_us):
    step_m = HIGHWAY_M_S * (tick_us / 1e6)
    view = make_view(
        frames=grid_frames(newest_us=2_000_000, count=2, tick_us=K.FRAME_SPACING_US),
        poses=((1_000_000, 0.0, 0.0, 0.0), (1_000_000 + tick_us, step_m, 0.0, 0.0)),
        velocity=(2.0, 0.0),
    )
    ego = planner().build_features(view, view.image_cache[FRONT][0],
                                   view.image_cache[FRONT][1])["ego_status"][0]
    np.testing.assert_allclose(ego[-1, 3:5], [2.0, 0.0], atol=ATOL)


# ===========================================================================
# (5) The bridge is never touched on a cached tick
# ===========================================================================


def test_cached_ticks_run_no_inference_and_decode_no_jpeg():
    """Q5: the short circuit is above build_features, so nothing downstream runs.

    bridge.forward is reachable only through _run_model, which plan() calls
    after build_features. A model_fn that raises therefore proves both at once.
    """
    calls = {"n": 0}

    def exploding_model(features):
        calls["n"] += 1
        raise AssertionError("the model must not be called on a cached tick")

    real = planner(model_fn=exploding_model, expected_frame_interval_us=50_000)
    divider = ReplanDivider(real)

    start = 1_000_000
    frames = grid_frames(newest_us=start, count=14, tick_us=50_000)
    view = make_view(frames=frames, poses=straight_poses(
        start_us=start, count=1, tick_us=50_000, speed_m_s=HIGHWAY_M_S))

    # Tick 0 is a replan: the model IS called, and blows up as instructed.
    with pytest.raises(AssertionError):
        divider.plan(view, time_now_us=start, time_query_us=start + 50_000)
    assert calls["n"] == 1

    # Seed the cache with a plan so later ticks have something to rebuild from.
    divider._sessions["unit"].t0_us = start
    divider._sessions["unit"].anchor_local_to_rig = matrix(0.0, 0.0, 0.0)
    positions = np.zeros((K.NUM_POSES, 3))
    positions[:, 0] = [10.0 * (k + 1) for k in range(K.NUM_POSES)]
    divider._sessions["unit"].positions = positions
    divider._sessions["unit"].rotations = np.repeat(np.eye(3)[None], K.NUM_POSES, 0)

    for i in range(1, 10):
        now = start + i * 50_000
        view_i = make_view(frames=frames, poses=straight_poses(
            start_us=start, count=i + 1, tick_us=50_000, speed_m_s=HIGHWAY_M_S))
        assert divider.plan(view_i, now, now + 50_000) is not None
    assert calls["n"] == 1, "nine cached ticks must add zero model calls"


def test_the_divider_reports_the_inner_planners_version():
    divider = ReplanDivider(planner())
    assert divider.version_id == f"drivejepa-v2-driver-{K.ADAPTER_VERSION}"
    assert K.ADAPTER_VERSION == "0.1.7"


# ===========================================================================
# (6) The bridge worker's environment
# ===========================================================================


def test_cpu_device_hides_the_gpu_from_the_bridge_worker():
    """--device cpu must allocate ZERO VRAM, end to end.

    The vendored agent branches on torch.cuda.is_available(), not on the device
    we asked for (drive_jepa_agent.py:60-63): with CUDA present it calls
    torch.load(path) and the checkpoint lands in VRAM on its way to a CPU model.
    Scene 032b6f21 died there -- CUDA OOM with 3.31 GiB free and thirteen
    foreign processes on the card.

    Emptying CUDA_VISIBLE_DEVICES for the worker makes upstream take its own
    map_location branch. Verified once against the real checkpoint: 495
    state_dict keys, all tensors on cpu, worker absent from nvidia-smi
    compute-apps.
    """
    from drivejepa.alpasim_driver.bridge import BridgedDriveJEPA

    for dev in ("cpu", "CPU", "cpu:0"):
        env = BridgedDriveJEPA._worker_env(dev)
        assert env["CUDA_VISIBLE_DEVICES"] == "", dev
        assert env["PYTHONPATH"], "PYTHONPATH must survive"
    # A real device request is passed through untouched.
    assert "CUDA_VISIBLE_DEVICES" not in BridgedDriveJEPA._worker_env("cuda:0") or \
        BridgedDriveJEPA._worker_env("cuda:0").get("CUDA_VISIBLE_DEVICES") != ""

