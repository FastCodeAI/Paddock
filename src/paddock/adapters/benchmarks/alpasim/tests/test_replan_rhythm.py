"""The fixed replan rhythm: grid, warm-up frames to /observe, re-timed plans, padded history.

Duck-typed observation views stand in for the scaffold's; the node is a stub on
the planner. The padding tests then hand every body the bridge sent to the
AutoE2E node's own frame store and motion-history code, so what is checked is
what that server computes. Runs in AlpaSim's environment, which supplies grpc
for the scaffold.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("grpc", reason="AlpaSim's environment supplies grpc")

from paddock.adapters.benchmarks.alpasim import (
    bridge,  # noqa: E402
    drivejepa_rhythm,  # noqa: E402
)
from paddock.adapters.benchmarks.alpasim import replan_rhythm as rhythm  # noqa: E402
from paddock.adapters.benchmarks.alpasim.scaffold import PlanResult  # noqa: E402

STEP_US = 50_000
FORCE_GT_US = 3_000_000
#: The car: 8 m/s, speeding up at 1 m/s^2 and turning left at 0.1 rad/s.
V0, ACCEL, YAW_RATE = 8.0, 1.0, 0.1


def _state(t_s: float) -> tuple[float, float, float]:
    """(x, y, yaw) of the car at *t_s*, exactly integrated."""
    # Arc length s(t) = V0 t + a t^2 / 2, heading w t; x, y by fine quadrature.
    ts = np.linspace(min(t_s, 0.0), max(t_s, 0.0), 2001)
    speed = V0 + ACCEL * ts
    yaw = YAW_RATE * ts
    sign = 1.0 if t_s >= 0 else -1.0
    x = sign * np.trapezoid(speed * np.cos(yaw), ts)
    y = sign * np.trapezoid(speed * np.sin(yaw), ts)
    return float(x), float(y), YAW_RATE * t_s


def _pose(timestamp_us: int):
    x, y, yaw = _state(timestamp_us / 1e6)
    quat = SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))
    return SimpleNamespace(
        timestamp_us=timestamp_us, pose=SimpleNamespace(quat=quat, vec=SimpleNamespace(x=x, y=y, z=0.0))
    )


def _view(now_us: int, first_pose_us: int = -STEP_US):
    """The session at *now_us*: a pose every 50 ms from *first_pose_us*, reported velocity 0
    (as AlpaSim reports it during the forced replay)."""
    times = list(range(first_pose_us, now_us + 1, STEP_US))
    zero = SimpleNamespace(linear_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0))
    return SimpleNamespace(
        session_uuid="s-1",
        poses=[_pose(t) for t in times],
        dynamic_states=[(t, zero) for t in times],
        image_cache={},
        route=None,
        drive_count=now_us // STEP_US + 1,
    )


def _matrix(pose) -> list:
    return drivejepa_rhythm.pose_to_matrix(pose.pose).tolist()


def _planner(tmp_path, pad: bool = True):
    planner = rhythm.ReplanRhythmPlanner(
        "http://127.0.0.1:1",
        replan_every_us=100_000,
        force_gt_us=FORCE_GT_US,
        frame_interval_us=STEP_US,
        pad_ego_history=pad,
        replan_log=tmp_path / "replans.jsonl",
    )
    planner.sent = []  # (endpoint, body), in order

    def body(view, time_now_us):
        pose, state = view.poses[-1], view.dynamic_states[-1][1]
        return (
            {
                "images": {},
                "ego2world": _matrix(pose),
                "canbus": bridge.canbus_from(SimpleNamespace(latest_dynamic_state=state)).tolist(),
                "timestamp": int(pose.timestamp_us),
                "command": int(bridge.Command.FOLLOW_REFERENCE),
                "calibration": {"camera2image": {}, "camera2ego": {}},
            },
            int(pose.timestamp_us),
        )

    plan = np.array([[2.0 * (k + 1), 0.1 * (k + 1), 0.02 * (k + 1)] for k in range(12)])

    def post_infer(sent):
        planner.sent.append(("infer", sent))
        return {
            "trajectory": plan[:6, :2].tolist(),
            "plan": {"waypoints": plan.tolist(), "dt": 0.5},
            "extras": {"autoe2e": {"history_repeated": 1, "ego_rows_zero": 0}},
        }

    def observe_frame(sent):
        planner.sent.append(("observe", sent))
        return "observe"

    planner.build_infer_body = body
    planner.post_infer = post_infer
    planner.observe_frame = observe_frame
    planner._post = lambda path, payload: {}
    return planner


def _drive(planner, until_us: int, first_pose_us: int = -STEP_US) -> list:
    """Ask at every 50 ms tick from 0 to *until_us*, query = now + one step, as AlpaSim asks."""
    planner.on_session_start(SimpleNamespace(session_uuid="s-1"))
    return [
        planner.plan(_view(t, first_pose_us), time_now_us=t, time_query_us=t + STEP_US)
        for t in range(0, until_us + 1, STEP_US)
    ]


def _log(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "replans.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


def test_warmup_frames_are_observed_then_the_model_is_asked_every_second_tick(tmp_path) -> None:
    planner = _planner(tmp_path, pad=False)
    results = _drive(planner, 4_000_000)
    kinds = [row["kind"] for row in _log(tmp_path)]
    # 0 .. 2.95 s: sixty warm-up ticks, every frame to /observe, no plan.
    assert kinds[:60] == ["observe"] * 60
    assert results[:60] == [None] * 60
    # From the handover at 3.0 s: infer on the 100 ms grid, the cached plan in between.
    assert kinds[60:] == ["infer", "reuse"] * 10 + ["infer"]
    assert [e for e, _ in planner.sent] == ["observe"] * 60 + ["infer"] * 11
    assert all(isinstance(r, PlanResult) for r in results[60:])
    infers = [row for row in _log(tmp_path) if row["kind"] == "infer"]
    assert [row["sim_time_us"] for row in infers] == list(range(3_000_000, 4_000_001, 100_000))
    assert all(row["model_called"] for row in infers)
    assert {(row["plan_waypoints"], row["plan_dt_s"], row["repeated_image_frames"]) for row in infers} == {
        (12, 0.5, 1)
    }


def test_the_grid_is_anchored_on_the_first_ask_not_on_round_times(tmp_path) -> None:
    planner = _planner(tmp_path, pad=False)
    planner.on_session_start(SimpleNamespace(session_uuid="s-1"))
    start = 1_234_000
    for t in range(start, start + FORCE_GT_US + 500_001, STEP_US):
        planner.plan(_view(t), time_now_us=t, time_query_us=t + STEP_US)
    infers = [row["sim_time_us"] for row in _log(tmp_path) if row["kind"] == "infer"]
    assert infers == [start + FORCE_GT_US + k * 100_000 for k in range(6)]


def test_the_period_must_be_a_whole_number_of_control_steps() -> None:
    with pytest.raises(ValueError, match="multiple"):
        rhythm.ReplanRhythmPlanner(
            "http://127.0.0.1:1", replan_every_us=75_000, force_gt_us=FORCE_GT_US, frame_interval_us=STEP_US
        )


def test_the_held_plan_is_drivejepas_rebuild_on_the_same_cache() -> None:
    """Same knots, same now, same pose: the two computations return the same plan."""
    positions = np.array([[2.0 * (k + 1), 0.3 * (k + 1) ** 1.5, 0.0] for k in range(12)])
    rotations = bridge.yaw_rotations(np.array([0.04 * (k + 1) for k in range(12)]))
    anchor = drivejepa_rhythm.pose_to_matrix(_pose(3_000_000).pose)
    view = _view(3_050_000)
    cached = rhythm.CachedPlan(3_000_000, anchor, positions, rotations, 500_000)
    ours = rhythm.rebuild_plan(cached, view, 3_050_000)
    session = drivejepa_rhythm._Session(
        t0_us=3_000_000, anchor_local_to_rig=anchor, positions=positions, rotations=rotations, step_us=500_000
    )
    planner = drivejepa_rhythm.DriveJepaPlanner(
        "http://127.0.0.1:1", front_camera="c", expected_frame_interval_us=STEP_US, force_gt_us=FORCE_GT_US
    )
    theirs = planner._rebuild(session, view, 3_050_000)
    assert ours.model_t0_us == theirs.model_t0_us == 3_050_000
    assert ours.step_us == theirs.step_us
    np.testing.assert_array_equal(ours.positions_in_rig, theirs.positions_in_rig)
    np.testing.assert_array_equal(ours.rotations_in_rig, theirs.rotations_in_rig)


def test_the_held_plan_is_retimed_onto_now_and_reanchored_to_the_car(tmp_path) -> None:
    """50 ms after the plan, its first waypoint is the plan's own 0.55 s point, seen from the car now."""
    planner = _planner(tmp_path, pad=False)
    results = _drive(planner, 3_050_000)
    held = results[-1]
    assert held.model_t0_us == 3_050_000 and held.step_us == 500_000
    assert len(held.positions_in_rig) == 11  # 0.5 .. 5.5 s still inside the 6 s plan
    plan = results[-2]
    knots_t = np.arange(13) * 0.5
    knots = np.vstack([[0, 0, 0], plan.positions_in_rig])
    at = 0.55
    in_plan_frame = np.array([np.interp(at, knots_t, knots[:, i]) for i in range(2)])
    plan_pose = drivejepa_rhythm.pose_to_matrix(_pose(3_000_000).pose)
    now_pose = drivejepa_rhythm.pose_to_matrix(_pose(3_050_000).pose)
    expected = (np.linalg.inv(now_pose) @ plan_pose @ np.array([*in_plan_frame, 0.0, 1.0]))[:2]
    assert held.positions_in_rig[0, :2] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# padding, through the AutoE2E server's own history code
# ---------------------------------------------------------------------------


def _server_rows(sent: list, upto: int) -> tuple[np.ndarray, list]:
    """The 64 motion rows the AutoE2E node computes at the *upto*-th body it was sent."""
    from paddock.adapters.models.autoe2e.policy import AutoE2EPolicy, egomotion_rows, yaw_of
    from paddock.policy.protocol import Observation

    node = AutoE2EPolicy()
    for _, body in sent[: upto + 1]:
        node.observe(Observation.from_wire(body, cameras=()))
    # As infer() builds them (policy.py): every frame held, the newest included.
    frames = node.frames
    rows = egomotion_rows(
        np.array([f.t_s for f in frames]),
        np.array([f.ego2world[:2, 3] for f in frames]),
        np.array([yaw_of(f.ego2world) for f in frames]),
        frames[-1].t_s,
    )
    return rows, frames


def test_padding_gives_34_constant_speed_rows_and_halves_the_join(tmp_path) -> None:
    planner = _planner(tmp_path, pad=True)
    _drive(planner, FORCE_GT_US)
    sent = planner.sent
    first_infer = next(i for i, (e, _) in enumerate(sent) if e == "infer")
    pad_rows = [row for row in _log(tmp_path) if row["kind"] == "pad"]
    (pad,) = pad_rows
    # 132 frames, 50 ms apart, back to 6.6 s before the first real frame, all sent before it.
    assert pad["padded_frames"] == 132
    assert [e for e, _ in sent[:133]] == ["observe"] * 133
    assert [b["images"] for _, b in sent[:132]] == [{}] * 132
    assert sent[0][1]["timestamp"] == -6_600_000 and sent[131][1]["timestamp"] == -50_000

    # The speed is the poses' (reported velocity is 0 here, as in the forced replay).
    assert all(b["canbus"][bridge.CANBUS_VELOCITY] == 0.0 for _, b in sent[132:])
    view = _view(0)
    newer, older = view.poses[-1].pose.vec, view.poses[-2].pose.vec
    expected_v = math.hypot(newer.x - older.x, newer.y - older.y) / (STEP_US / 1e6)
    assert pad["speed_m_s"] == pytest.approx(expected_v, rel=1e-12)

    rows, _ = _server_rows(sent, first_infer)
    infer_row = next(row for row in _log(tmp_path) if row["kind"] == "infer")
    assert infer_row["padded_ego_rows"] == 34
    # Rows 0..33 (6.4 .. 3.1 s back) are padding: the first speed, nothing else.
    np.testing.assert_allclose(rows[:34], np.tile([expected_v, 0.0, 0.0, 0.0], (34, 1)), atol=1e-9)

    # Without padding, the same real frames give today's rows; padding changes only the join.
    unpadded = [(e, b) for e, b in sent if not (e == "observe" and not b["images"] and b["timestamp"] < 0)]
    today, _ = _server_rows(unpadded, len(unpadded) - 1)
    np.testing.assert_allclose(rows[35:], today[35:], rtol=0, atol=1e-12)
    # The join (the first real frame, t = 0) takes one neighbour from the padding:
    #   speed: centred, the true 8.0 (today one-sided, 8.025);
    #   yaw rate: half the true value (the padding does not turn);
    #   acceleration: 3/4 of the true value. The padding's speed is the last
    #   50 ms before the first frame (true speed at -25 ms), so the centred
    #   difference spans 75 ms of the true 100 ms of speed-up. Today's one-sided
    #   value is 1/2.
    assert rows[34, 0] == pytest.approx(V0, abs=1e-3)
    assert today[34, 0] == pytest.approx(V0 + ACCEL * 0.025, abs=1e-3)
    assert rows[34, 2] == pytest.approx(YAW_RATE / 2, abs=1e-6)
    assert today[34, 2] == pytest.approx(YAW_RATE, abs=1e-6)
    assert rows[34, 1] == pytest.approx(0.75 * ACCEL, abs=1e-3)
    assert today[34, 1] == pytest.approx(0.5 * ACCEL, abs=1e-3)


def test_padding_waits_for_a_second_pose_and_still_comes_first(tmp_path) -> None:
    planner = _planner(tmp_path, pad=True)
    _drive(planner, 200_000, first_pose_us=0)
    kinds = [row["kind"] for row in _log(tmp_path)]
    assert kinds[:3] == ["pad_wait", "pad", "observe"]
    timestamps = [b["timestamp"] for _, b in planner.sent]
    assert timestamps == sorted(timestamps)
    assert planner.sent[131][1]["timestamp"] == 0  # padding ends one step before the first real frame


def test_without_the_flag_nothing_is_padded(tmp_path) -> None:
    planner = _planner(tmp_path, pad=False)
    _drive(planner, 100_000)
    assert all(b["images"] == {} and b["timestamp"] >= 0 for _, b in planner.sent)
    assert "pad" not in [row["kind"] for row in _log(tmp_path)]


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def test_the_autoe2e_profile_flags_select_this_planner(monkeypatch) -> None:
    from paddock.adapters.benchmarks.alpasim import harness

    seen = {}
    monkeypatch.setattr(bridge, "serve", lambda *a, **k: seen.update(k))
    bridge.main(["--policy-url", "http://x", "--port", "1", *harness.profile_for("autoe2e").bridge_args])
    assert seen["rhythm"] is None
    assert seen["replan_rhythm"] == {
        "replan_every_us": 100_000,
        "frame_interval_us": 50_000,
        "force_gt_us": 3_000_000,
        "pad_ego_history": True,
        "observe_between_replans": False,
    }
    assert seen["pinhole_intrinsics"] is True and seen["placeholders"] == ()


@pytest.mark.parametrize(
    "args",
    [
        ["--pad-ego-history"],
        ["--replan-every-us", "100000", "--force-gt-us", "3000000"],
        [
            "--replan-every-us",
            "100000",
            "--frame-interval-us",
            "50000",
            "--force-gt-us",
            "1",
            "--planner",
            "drivejepa-0.1.4b",
            "--front-camera",
            "c",
        ],
    ],
)
def test_inconsistent_rhythm_flags_are_refused(args, monkeypatch) -> None:
    monkeypatch.setattr(bridge, "serve", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        bridge.main(["--policy-url", "http://x", "--port", "1", *args])


def test_drivejepas_flags_still_select_drivejepas_planner(monkeypatch) -> None:
    from paddock.adapters.benchmarks.alpasim import harness

    seen = {}
    monkeypatch.setattr(bridge, "serve", lambda *a, **k: seen.update(k))
    bridge.main(["--policy-url", "http://x", "--port", "1", *harness.profile_for("drivejepa").bridge_args])
    assert seen["replan_rhythm"] is None
    assert seen["rhythm"]["front_camera"] == "camera_front_wide_120fov"
