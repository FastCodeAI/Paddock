"""The Drive-JEPA bridge profile: replan grid, frame pair, ego status, full plan, re-anchoring.

Duck-typed observation views stand in for the scaffold's; the node is a stub on
the planner. Runs in AlpaSim's environment, which supplies grpc for the scaffold.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("grpc", reason="AlpaSim's environment supplies grpc")

from paddock.adapters.benchmarks.alpasim import drivejepa_rhythm as rhythm  # noqa: E402
from paddock.policy.protocol import Command  # noqa: E402

CAMERA = "camera_front_wide_120fov"


def _pose(timestamp_us: int, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    quat = SimpleNamespace(w=np.cos(yaw / 2), x=0.0, y=0.0, z=np.sin(yaw / 2))
    return SimpleNamespace(
        timestamp_us=timestamp_us, pose=SimpleNamespace(quat=quat, vec=SimpleNamespace(x=x, y=y, z=0.0))
    )


def _state(vx=0.0, vy=0.0, ax=0.0, ay=0.0):
    return SimpleNamespace(
        linear_velocity=SimpleNamespace(x=vx, y=vy), linear_acceleration=SimpleNamespace(x=ax, y=ay)
    )


def _view(now_us: int, speed: float = 5.0, vy: float = 0.25, ax: float = 0.5, ay: float = -0.75, route=None):
    """A car driving along x at *speed*, a pose and a frame every 50 ms up to *now_us*."""
    times = list(range(0, now_us + 1, 50_000))[-40:]
    return SimpleNamespace(
        session_uuid="s-1",
        poses=[_pose(t, x=speed * t / 1e6) for t in times],
        dynamic_states=[(t, _state(speed, vy, ax, ay)) for t in times],
        image_cache={CAMERA: [(t, b"jpeg") for t in times[-rhythm.IMAGE_HISTORY :]]},
        route=route,
    )


def _route(*points):
    return SimpleNamespace(waypoints=[SimpleNamespace(x=x, y=y) for x, y in points])


def _planner(tmp_path, answer_index: int = 7):
    planner = rhythm.DriveJepaPlanner(
        "http://127.0.0.1:1",
        front_camera=CAMERA,
        expected_frame_interval_us=50_000,
        force_gt_us=3_000_000,
        replan_log=tmp_path / "replans.jsonl",
    )
    planner.sent = []
    poses = np.stack([[0.5 * (i + 1), 0.1 * (i + 1), 0.05 * (i + 1)] for i in range(8)])

    def body(view, frame, speed, accel, command):
        return {"frame": int(frame[0]), "speed": speed, "accel": accel, "command": command}

    def post_infer(sent):
        planner.sent.append(sent)
        return {
            "trajectory": poses[:6, :2].tolist(),
            "aux_outputs": {},
            "plan": {"waypoints": poses.tolist(), "dt": 0.5},
            "extras": {
                "drivejepa_pick": {
                    "chosen_index": answer_index,
                    "chosen_score": 0.875,
                    "momentum_fed": len(planner.sent) > 2,
                }
            },
        }

    planner._body, planner.post_infer, planner._post = body, post_infer, lambda path, b: True
    return planner, poses


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "replans.jsonl").read_text().splitlines()]


# -- the inputs ----------------------------------------------------------------


@pytest.mark.parametrize(
    "route, command",
    [
        (None, Command.FOLLOW_REFERENCE),
        (_route(), Command.FOLLOW_REFERENCE),
        (_route((5.0, 0.0), (10.0, 0.0)), Command.STRAIGHT),
        (_route((10.0, 1.0), (25.0, 4.0)), Command.LEFT),
        (_route((10.0, 1.0), (25.0, -4.0)), Command.RIGHT),
        (_route((25.0, 2.9)), Command.STRAIGHT),
    ],
)
def test_the_command_is_read_off_the_first_route_point_twenty_metres_out(route, command) -> None:
    assert rhythm.command_from_route(route) is command


def test_a_reported_velocity_is_sent_as_it_is_both_components() -> None:
    assert rhythm.newest_velocity(_view(1_000_000, speed=5.0, vy=0.25)) == (5.0, 0.25, False)
    assert rhythm.reported_acceleration(_view(1_000_000, ax=0.5, ay=-0.75)) == (0.5, -0.75)


def test_a_zero_velocity_beside_moving_poses_is_derived_from_them() -> None:
    """The simulator's warm-up "unknown" at handover: 6 m/s of poses, 0 reported."""
    view = _view(1_000_000, speed=6.0)
    view.dynamic_states[-1] = (1_000_000, _state())
    vx, vy, derived = rhythm.newest_velocity(view)
    assert derived and vx == pytest.approx(6.0) and vy == pytest.approx(0.0, abs=1e-9)


def test_a_zero_velocity_beside_a_crawl_stays_zero() -> None:
    view = _view(1_000_000, speed=0.5)
    view.dynamic_states[-1] = (1_000_000, _state())
    assert rhythm.newest_velocity(view) == (0.0, 0.0, False)


# -- the grid and the frame pair -------------------------------------------------


def test_replans_fall_every_half_second_from_the_first_ask_and_catch_up() -> None:
    planner = rhythm.DriveJepaPlanner(
        "http://x", front_camera=CAMERA, expected_frame_interval_us=50_000, force_gt_us=0
    )
    session = rhythm._Session()
    due = [planner._due(session, t) for t in (100_000, 150_000, 550_000, 600_000, 1_700_000, 2_100_000)]
    assert due == [True, False, False, True, True, True]
    assert session.next_replan_us == 2_600_000, "the grid stays on the first ask's phase"


def test_the_pair_is_the_newest_frame_and_the_one_half_a_second_back() -> None:
    planner = rhythm.DriveJepaPlanner(
        "http://x", front_camera=CAMERA, expected_frame_interval_us=50_000, force_gt_us=0
    )
    older, newer = planner._select_frames(rhythm._Session(), _view(2_000_000))
    assert (older[0], newer[0]) == (1_500_000, 2_000_000)
    assert planner._select_frames(rhythm._Session(), _view(300_000)) is None, "nothing 0.5 s back yet"


def test_an_off_grid_pair_is_refused() -> None:
    planner = rhythm.DriveJepaPlanner(
        "http://x", front_camera=CAMERA, expected_frame_interval_us=50_000, force_gt_us=0
    )
    view = _view(2_000_000)
    view.image_cache[CAMERA] = [
        entry for entry in view.image_cache[CAMERA] if not 1_400_000 <= entry[0] <= 1_600_000
    ]
    with pytest.raises(rhythm.ContextSpacingError):
        planner._select_frames(rhythm._Session(), view)


def test_frames_pushed_at_another_rate_than_the_bridge_was_launched_for_are_refused() -> None:
    planner = rhythm.DriveJepaPlanner(
        "http://x", front_camera=CAMERA, expected_frame_interval_us=100_000, force_gt_us=0
    )
    with pytest.raises(rhythm.PushCadenceError):
        planner._select_frames(rhythm._Session(), _view(2_000_000))


# -- a session -------------------------------------------------------------------------


def test_the_first_ask_only_stores_a_frame_and_the_next_replan_drives_the_full_plan(tmp_path) -> None:
    planner, poses = _planner(tmp_path)
    assert planner.plan(_view(1_000_000), 1_000_000, 1_000_000) is None, "first ask: frame stored, no plan"
    assert planner.plan(_view(1_050_000), 1_050_000, 1_050_000) is None, "no plan cached yet"
    result = planner.plan(_view(1_500_000), 1_500_000, 1_500_000)

    assert [sent["frame"] for sent in planner.sent] == [1_000_000, 1_500_000]
    assert (planner.sent[1]["speed"], planner.sent[1]["accel"]) == (5.0, 0.5)
    assert result.model_t0_us == 1_500_000 and result.step_us == 500_000
    np.testing.assert_array_equal(result.positions_in_rig[:, :2], poses[:, :2])
    assert len(result.positions_in_rig) == 8, "all eight poses, not the wire's six"
    yaws = [rhythm.yaw_of(np.pad(r, ((0, 1), (0, 1)))) for r in result.rotations_in_rig]
    np.testing.assert_allclose(yaws, poses[:, 2])

    first, replan = _log(tmp_path)
    assert first["kind"] == "store_first_frame" and replan["kind"] == "replan"
    assert replan["extras"] == {
        "drivejepa_pick": {"chosen_index": 7, "chosen_score": 0.875, "momentum_fed": False}
    }, "the node's extras, as it sent them"
    assert replan["headings"] == "model"
    assert (replan["vy_measured"], replan["ay_measured"]) == (0.25, -0.75), "logged, not sent"
    assert np.asarray(replan["trajectory_rig"]).shape == (8, 3)


def test_between_replans_the_plan_is_re_anchored_and_keeps_its_headings(tmp_path) -> None:
    planner, poses = _planner(tmp_path)
    planner.plan(_view(1_000_000), 1_000_000, 1_000_000)
    planner.plan(_view(1_500_000), 1_500_000, 1_500_000)
    # 0.5 s on, the car has moved 2.5 m along x and the cached plan's first
    # waypoint is due now: what is left starts at the second, shifted back.
    rebuilt = planner.plan(_view(2_000_000), 2_000_000, 1_950_000)
    assert rebuilt.model_t0_us == 2_000_000 and len(rebuilt.positions_in_rig) == 7
    np.testing.assert_allclose(rebuilt.positions_in_rig[:, 0], poses[1:, 0] - 2.5, atol=1e-9)
    np.testing.assert_allclose(rebuilt.positions_in_rig[:, 1], poses[1:, 1], atol=1e-9)
    yaws = [float(np.arctan2(r[1, 0], r[0, 0])) for r in rebuilt.rotations_in_rig]
    np.testing.assert_allclose(yaws, poses[1:, 2], atol=1e-9)


def test_a_session_start_resets_the_node(tmp_path) -> None:
    planner, _ = _planner(tmp_path)
    posted = []
    planner._post = lambda path, body: posted.append(path)
    planner.on_session_start(SimpleNamespace(session_uuid="s-1"))
    assert posted == ["/reset"]


def test_the_body_carries_forward_speed_and_acceleration_in_canbus_only(monkeypatch) -> None:
    monkeypatch.setattr(rhythm, "decode_frame", lambda payload: np.zeros((4, 6, 3), dtype=np.uint8))
    monkeypatch.setattr(rhythm, "calibration_from", lambda view, pinhole_intrinsics=None: {})
    planner = rhythm.DriveJepaPlanner(
        "http://x", front_camera=CAMERA, expected_frame_interval_us=50_000, force_gt_us=0
    )
    planner._translate_cameras = lambda images, shapes, calibration: (images, calibration)
    view = _view(1_000_000)
    view.latest_dynamic_state = view.dynamic_states[-1][1]
    body = planner._body(view, (1_000_000, b"jpeg"), 5.0, 0.5, Command.LEFT)
    assert "ego_velocity_xy" not in body and "ego_acceleration_xy" not in body
    assert body["canbus"][13] == 5.0 and body["canbus"][7] == 0.5
    assert body["timestamp"] == 1_000_000 and body["command"] == int(Command.LEFT)
