"""Contract tests for the Alpamayo 2 Edge adapter. No torch, no weights, no GPU.

What they pin is the input upstream's loader builds -- which camera fills which
slot, the 22-frame schedule and the times it is labelled with, the 20 poses --
because that is where an adapter goes wrong silently: the model still returns a
confident plan from a sideways camera or a mislabelled frame.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock import conformance
from paddock.adapters.models.alpamayo2edge.policy import (
    GRID_INDICES,
    HISTORY_DT,
    NEEDED_CAMERAS,
    NUM_HISTORY_STEPS,
    OFFSETS_S,
    PLAN_DT,
    POLICY,
    SLOTS,
    camera_yaw_deg,
    carry_plan,
    nearest_index,
)
from paddock.adapters.models.alpamayo15.camera_warp import TARGET_H, TARGET_W
from paddock.policy.protocol import Command, Observation, Policy

#: Bearings, degrees left of forward, of a nuScenes-like ring.
YAWS = {"CAM_FRONT": 0.0, "CAM_FRONT_LEFT": 55.0, "CAM_FRONT_RIGHT": -55.0, "CAM_BACK_LEFT": 110.0,
        "CAM_BACK_RIGHT": -110.0, "CAM_BACK": 180.0}  # fmt: skip
W, H = 1600, 900
K = np.array([[1260.0, 0.0, 800.0], [0.0, 1260.0, 450.0], [0.0, 0.0, 1.0]])


def camera2ego(yaw_deg: float) -> np.ndarray:
    """A camera mounted at *yaw_deg*, in paddock's [right, down, forward] camera frame."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = yaw @ conformance.UNROTATED_CAMERA2EGO
    out[2, 3] = 1.5
    return out


def observation(
    t_s: float, cameras=tuple(YAWS), speed: float = 10.0, heading_deg: float = 30.0
) -> Observation:
    """Ego driving straight at *speed* along *heading_deg* in the world; each camera's
    image is a flat colour encoding its name and the frame time, so a test can tell
    which camera and which frame ended up in a slot."""
    h = math.radians(heading_deg)
    ego2world = np.eye(4)
    ego2world[:3, :3] = [[math.cos(h), -math.sin(h), 0.0], [math.sin(h), math.cos(h), 0.0], [0.0, 0.0, 1.0]]
    ego2world[:3, 3] = [speed * t_s * math.cos(h), speed * t_s * math.sin(h), 0.0]
    images = {}
    for i, camera in enumerate(cameras):
        image = np.zeros((H, W, 3), dtype=np.uint8)
        image[..., 0] = 10 + 40 * i
        image[..., 1] = int(round(t_s * 10)) % 256
        images[camera] = image
    return Observation(
        images=images,
        ego2world=ego2world,
        canbus=np.zeros(18),
        timestamp_us=int(round(t_s * 1e6)),
        command=Command.STRAIGHT,
        camera2image={c: K for c in cameras},
        camera2ego={c: camera2ego(YAWS[c]) for c in cameras},
    )


def recorded(times_s, **kwargs) -> POLICY:
    policy = POLICY()
    for t in times_s:
        policy.observe(observation(t, **kwargs))
    return policy


def test_policy_imports_without_the_model_environment() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.waypoint_dt == 0.5
    assert "CAM_BACK" not in policy.cameras, "the straight-back camera is not an Edge input"


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(object())  # type: ignore[arg-type]


def test_slots_are_upstreams_six_cameras_at_their_fields_of_view() -> None:
    """Ids 0/1/2/3/5/6 with 3/5/3/3/3/5 frames: upstream refuses any other set
    (sample.CAMERA_FRAME_COUNTS). Id 4, the rear telephoto, is not among them."""
    assert sorted(SLOTS) == [0, 1, 2, 3, 5, 6]
    assert {s: len(o) for s, o in OFFSETS_S.items()} == {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
    assert sum(len(o) for o in OFFSETS_S.values()) == 22
    assert SLOTS[1] == ("CAM_FRONT", 120.0) and SLOTS[6] == ("CAM_FRONT", 30.0)
    assert SLOTS[3] == ("CAM_BACK_LEFT", 70.0) and SLOTS[5] == ("CAM_BACK_RIGHT", 70.0)
    assert {camera for camera, _ in SLOTS.values()} == set(NEEDED_CAMERAS)


def test_each_slot_is_filled_from_the_camera_at_its_bearing() -> None:
    """Left slots from cameras that look left, right from right, rear from rear."""
    for slot, (camera, _) in SLOTS.items():
        yaw = YAWS[camera]
        if slot in (1, 6):
            assert yaw == 0.0
        elif slot in (0, 3):
            assert yaw > 0, f"slot {slot} is a left camera"
        else:
            assert yaw < 0, f"slot {slot} is a right camera"
        if slot in (3, 5):
            assert abs(yaw) > 90, f"slot {slot} looks backward"


def test_camera_yaw_reads_the_optical_axis() -> None:
    assert camera_yaw_deg(camera2ego(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert camera_yaw_deg(camera2ego(55.0)) == pytest.approx(55.0)
    assert camera_yaw_deg(camera2ego(-110.0)) == pytest.approx(-110.0)


def test_grid_indices_land_on_the_half_second_grid() -> None:
    """Index i is at (i+1) * 0.1 s; an off-by-one shifts every waypoint 100 ms."""
    grid = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    assert list(GRID_INDICES) == [int(round(t / 0.1)) - 1 for t in grid]


def test_ten_hertz_history_gives_upstreams_schedule_exactly() -> None:
    """At 10 Hz every asked-for time exists, so the labels are exactly upstream's offsets."""
    policy = recorded(np.arange(0.0, 3.01, 0.1))
    images, cameras, stamps, black = policy.camera_inputs()
    assert images.shape == (22, 3, TARGET_H, TARGET_W) and images.dtype == np.uint8
    assert black == []
    now = policy.timestamps[-1]
    for slot, offsets in OFFSETS_S.items():
        got = (stamps[cameras == slot] - now) / 1e6
        np.testing.assert_allclose(got, [-o for o in offsets], atol=1e-6)


def test_slow_source_is_labelled_with_the_real_frame_times() -> None:
    """nuScenes keyframes are 0.5 s apart: the -0.1 s slot holds the 0 s frame, and its
    label says 0 s rather than the -0.1 s that was asked for."""
    policy = recorded(np.arange(0.0, 3.01, 0.5))
    _, cameras, stamps, _ = policy.camera_inputs()
    now = policy.timestamps[-1]
    front = (stamps[cameras == 1] - now) / 1e6
    np.testing.assert_allclose(front, [-2.0, -1.0, -0.5, 0.0, 0.0], atol=1e-6)


def test_frames_come_from_the_right_camera_and_time() -> None:
    """The colour encodes camera and frame time; a mixed-up slot or frame shows here."""
    policy = recorded(np.arange(0.0, 3.01, 0.1))
    images, cameras, stamps, _ = policy.camera_inputs()
    order = list(YAWS)
    for image, slot, stamp in zip(images, cameras, stamps, strict=True):
        camera = SLOTS[int(slot)][0]
        centre = image[:, TARGET_H // 2, TARGET_W // 2]
        assert centre[0] == 10 + 40 * order.index(camera)
        assert centre[1] == int(round(stamp / 1e5)) % 256


def test_forward_only_source_gets_black_slots_never_the_front_image() -> None:
    """comma2k19 has one camera. Upstream refuses fewer than six, so four slots are
    black -- copying the front view into them would claim a view the car lacks."""
    policy = recorded(np.arange(0.0, 3.01, 0.05), cameras=("CAM_FRONT",))
    images, cameras, _, black = policy.camera_inputs()
    assert black == [0, 2, 3, 5]
    for slot in (0, 2, 3, 5):
        assert not images[cameras == slot].any()
    for slot in (1, 6):
        assert images[cameras == slot].any()
    assert policy.diagnostics()["slots"][3] == "black"


def test_source_without_a_front_camera_is_refused() -> None:
    with pytest.raises(ValueError, match="CAM_FRONT"):
        recorded([0.0], cameras=("CAM_FRONT_LEFT",))


def test_telephoto_is_a_thirty_degree_crop_of_the_front() -> None:
    """Exact, because 30 deg is narrower than any source: the crop's width in source
    pixels is 2 f tan(15 deg)."""
    from paddock.adapters.models.alpamayo2edge.policy import to_pinhole
    from paddock.adapters.models.alpamayo15.camera_warp import crop_box
    from paddock.policy.protocol import Pinhole

    pinhole = Pinhole.from_calibration(K, W, H)
    left, _, right, _ = crop_box(to_pinhole(pinhole), SLOTS[6][1])
    assert right - left == pytest.approx(2 * K[0, 0] * math.tan(math.radians(15)), abs=1.0)
    # The 120 deg slot cannot be honoured by a 65 deg sensor: it takes the widest window.
    left, _, right, _ = crop_box(to_pinhole(pinhole), SLOTS[1][1])
    assert right - left >= W - 4  # symmetric about the principal point, so a pixel or two short


def test_nearest_frame_ties_go_to_the_earlier_one() -> None:
    times = np.array([0.0, 0.5, 1.0])
    assert nearest_index(times, 0.75) == 1
    assert nearest_index(times, 0.9) == 2


def test_history_is_twenty_poses_in_the_t0_frame() -> None:
    """10 m/s straight along a 30 deg world heading: in the car's frame the poses lie on
    -x at 1 m per 0.1 s, with no lateral offset and no rotation."""
    pytest.importorskip("scipy")
    policy = recorded(np.arange(0.0, 3.01, 0.1))
    xyz, rot = policy.ego_history()
    assert xyz.shape == (NUM_HISTORY_STEPS, 3) and rot.shape == (NUM_HISTORY_STEPS, 3, 3)
    expected_x = -10.0 * HISTORY_DT * np.arange(NUM_HISTORY_STEPS - 1, -1, -1)
    np.testing.assert_allclose(xyz[:, 0], expected_x, atol=1e-6)
    np.testing.assert_allclose(xyz[:, 1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(rot, np.repeat(np.eye(3)[None], NUM_HISTORY_STEPS, axis=0), atol=1e-6)


def test_history_from_a_slow_source_keeps_its_metric_spacing() -> None:
    """At 2 Hz the 0.1 s poses are interpolated, not repeated: repeats would read as a
    car that stopped, and the model takes its speed from exactly this spacing."""
    pytest.importorskip("scipy")
    xyz, _ = recorded(np.arange(0.0, 3.01, 0.5)).ego_history()
    np.testing.assert_allclose(np.diff(xyz[:, 0]), 1.0, atol=1e-6)


def test_old_history_is_dropped_but_the_window_survives() -> None:
    policy = recorded(np.arange(0.0, 10.01, 0.1))
    span = (policy.timestamps[-1] - policy.timestamps[0]) / 1e6
    assert 2.0 < span <= 2.7


def test_speed_ablation_settings_default_to_the_released_model(monkeypatch) -> None:
    """Unset, the policy plans as released: upstream's 10 steps and its own reasoning."""
    monkeypatch.delenv("ALPAMAYO2_EDGE_FLOW_STEPS", raising=False)
    monkeypatch.delenv("ALPAMAYO2_EDGE_REASONING", raising=False)
    policy = POLICY()
    assert policy.flow_steps is None and policy.reasoning == "model"


def test_speed_ablation_settings_are_read_and_checked(monkeypatch) -> None:
    monkeypatch.setenv("ALPAMAYO2_EDGE_FLOW_STEPS", "2")
    monkeypatch.setenv("ALPAMAYO2_EDGE_REASONING", "nominal")
    policy = POLICY()
    assert policy.flow_steps == 2 and policy.reasoning == "nominal"
    monkeypatch.setenv("ALPAMAYO2_EDGE_REASONING", "short")
    with pytest.raises(ValueError, match="ALPAMAYO2_EDGE_REASONING"):
        POLICY()
    monkeypatch.setenv("ALPAMAYO2_EDGE_REASONING", "model")
    monkeypatch.setenv("ALPAMAYO2_EDGE_FLOW_STEPS", "0")
    with pytest.raises(ValueError, match="ALPAMAYO2_EDGE_FLOW_STEPS"):
        POLICY()


def test_expert_layer_deletion_is_off_by_default_and_checked(monkeypatch) -> None:
    monkeypatch.delenv("ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS", raising=False)
    assert POLICY().drop_expert_layers == ()
    monkeypatch.setenv("ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS", "")
    assert POLICY().drop_expert_layers == ()
    monkeypatch.setenv("ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS", "19, 1,5,1")
    assert POLICY().drop_expert_layers == (1, 5, 19)
    monkeypatch.setenv("ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS", "1;2")
    with pytest.raises(ValueError, match="ALPAMAYO2_EDGE_DROP_EXPERT_LAYERS"):
        POLICY()


def test_deleted_expert_layers_leave_the_rest_reading_their_own_cache_layer() -> None:
    """A kept layer keeps its layer_idx: expert layer i only knows cache layer i."""
    from types import SimpleNamespace

    from paddock.adapters.models.alpamayo2edge.policy import drop_expert_layers

    layers = [SimpleNamespace(self_attn=SimpleNamespace(layer_idx=i)) for i in range(28)]
    text = SimpleNamespace(layers=layers, config=SimpleNamespace(num_hidden_layers=28))
    kept = drop_expert_layers(text, (1, 2, 3, 4, 5, 12, 15, 19))
    assert len(text.layers) == text.config.num_hidden_layers == 20 == len(kept)
    assert [layer.self_attn.layer_idx for layer in text.layers] == kept
    assert kept[:3] == [0, 6, 7]
    with pytest.raises(ValueError, match="not in a 20-layer expert"):
        drop_expert_layers(text, (25,))


def test_forced_nominal_is_the_models_own_token_sequence() -> None:
    """'<N' 'OM' 'IN' 'AL' '>\\n' then <|traj_future_start|>, as the model writes it. The last
    text piece is '>\\n' (1561), not the '>' (1062) that tokenising "<NOMINAL>" gives."""
    from paddock.adapters.models.alpamayo2edge.policy import FORCED_REASONING_IDS

    assert FORCED_REASONING_IDS["nominal"] == [124549, 11135, 3174, 4286, 1561, 135084]
    assert FORCED_REASONING_IDS["none"] == [135084]
    assert all(ids[-1] == 135084 for ids in FORCED_REASONING_IDS.values())


def test_warm_start_is_off_by_default_and_checked(monkeypatch) -> None:
    monkeypatch.delenv("ALPAMAYO2_EDGE_WARM_START_STEPS", raising=False)
    assert POLICY().warm_steps == 0
    monkeypatch.setenv("ALPAMAYO2_EDGE_WARM_START_STEPS", "1")
    assert POLICY().warm_steps == 1
    monkeypatch.setenv("ALPAMAYO2_EDGE_WARM_START_STEPS", "-1")
    with pytest.raises(ValueError, match="ALPAMAYO2_EDGE_WARM_START_STEPS"):
        POLICY()


def pose(x: float, y: float, yaw: float) -> np.ndarray:
    out = np.eye(4)
    out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    out[:2, 3] = x, y
    return out


def plan_along(speed: float, yaw_rate: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """64 poses at 0.1 s of a car at constant speed and yaw rate, starting at the origin."""
    t = PLAN_DT * np.arange(1, 65)
    yaw = yaw_rate * t
    if yaw_rate:
        xyz = np.stack([speed / yaw_rate * np.sin(yaw), speed / yaw_rate * (1 - np.cos(yaw)), 0 * t], 1)
    else:
        xyz = np.stack([speed * t, 0 * t, 0 * t], 1)
    return xyz, np.stack([pose(0, 0, a)[:3, :3] for a in yaw])


def test_a_plan_carried_where_the_car_followed_it_is_the_same_plan() -> None:
    """Straight at 10 m/s in a world frame turned 30 deg: 0.1 s later, having driven 1 m along
    it, the plan seen from the new pose is the plan made there."""
    xyz, rot = plan_along(10.0)
    start = pose(5.0, -2.0, math.radians(30))
    later = start @ pose(1.0, 0.0, 0.0)
    moved, turned = carry_plan(xyz, rot, start, later, 0.1)
    np.testing.assert_allclose(moved, xyz, atol=1e-9)
    np.testing.assert_allclose(turned, rot, atol=1e-9)


def test_a_turning_plan_is_carried_in_heading_too() -> None:
    """On a constant-rate turn, the carried poses are the turn continued from the new pose,
    the two past the old plan's end extrapolated along its last step."""
    xyz, rot = plan_along(8.0, yaw_rate=0.2)
    later = pose(*xyz[1, :2], 0.2 * 0.2)  # 0.2 s on, where the old plan put the car
    moved, turned = carry_plan(xyz, rot, np.eye(4), later, 0.2)
    np.testing.assert_allclose(moved[:-2], xyz[:-2], atol=1e-9)
    np.testing.assert_allclose(turned, rot, atol=1e-9)
    # Straight on along the last step, so off the arc by centimetres at 6.4 s.
    np.testing.assert_allclose(moved[-2:], xyz[-2:], atol=0.1)


def test_a_plan_carried_no_time_from_the_same_pose_is_unchanged() -> None:
    xyz, rot = plan_along(6.0, yaw_rate=-0.1)
    moved, turned = carry_plan(xyz, rot, pose(3, 4, 1.0), pose(3, 4, 1.0), 0.0)
    np.testing.assert_allclose(moved, xyz, atol=1e-12)
    np.testing.assert_allclose(turned, rot, atol=1e-12)


def test_reset_forgets_the_previous_plan() -> None:
    """A warm start never reaches across sequences: reset() drops the plan it would start from."""
    policy = POLICY()
    policy._prev_plan = {"timestamp_us": 0}
    policy.reset()
    assert policy._prev_plan is None
