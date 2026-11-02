"""The AlpaSim inputs: a real telephoto, f-theta frames as rendered, the route's
navigation text and the whole 0.1 s plan -- each switched on by its own input.

The navigation-text and slot tests are numpy only and run in `make check`; the
ones that call `infer` need torch and run in `make check-alpamayo`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock.adapters.models.alpamayo15.policy import (
    TELE_CAMERA,
    Frame,
    nav_text_for,
    slots_for,
)
from paddock.adapters.models.alpamayo15.route_nav import route_in_frame, route_to_nav_text
from paddock.policy.protocol import Command, Observation

# -- navigation text from a route --------------------------------------------------


def _bend(turn_at_m: float, side: float, length_m: float = 30.0) -> np.ndarray:
    """Straight ahead, then a 90 degree bend to *side* (+1 left, -1 right) at *turn_at_m*."""
    straight = np.array([[x, 0.0] for x in np.arange(-2.0, turn_at_m + 0.01, 1.0)])
    after = np.array([[turn_at_m, side * y] for y in np.arange(1.0, length_m, 1.0)])
    return np.vstack([straight, after])


def test_a_left_bend_ahead_is_a_left_turn_at_its_distance() -> None:
    assert route_to_nav_text(_bend(20.0, +1.0)) == "Turn left in 20m"


def test_a_right_bend_ahead_is_a_right_turn_at_its_distance() -> None:
    assert route_to_nav_text(_bend(12.0, -1.0)) == "Turn right in 12m"


def test_a_straight_route_is_continue_straight() -> None:
    assert route_to_nav_text(np.array([[x, 0.02 * x] for x in range(-5, 60)], dtype=float)) == (
        "Continue straight"
    )


def test_a_bend_beyond_40_m_is_not_announced_yet() -> None:
    assert route_to_nav_text(_bend(45.0, +1.0)) == "Continue straight"


def test_no_usable_waypoint_is_no_instruction() -> None:
    """None, so the model runs without a route instruction, as AlpaSim's driver does."""
    assert route_to_nav_text(np.zeros((0, 2))) is None
    assert route_to_nav_text(np.full((4, 2), np.nan)) is None


def test_two_points_cannot_turn() -> None:
    assert route_to_nav_text(np.array([[0.0, 0.0], [10.0, 5.0]])) == "Continue straight"


def test_a_turn_close_by_is_never_called_less_than_5_m_and_never_0_m() -> None:
    """Turns nearer than 5 m are ignored (the ego is already in them), so a distance
    rounding to 0 cannot occur on this path; the command path can still say "in 0m"."""
    assert route_to_nav_text(_bend(3.0, +1.0)) == "Turn left in 5m"
    assert nav_text_for("TURN_LEFT", 0.3) == "Turn left in 0m"


def test_a_route_in_an_older_frame_is_moved_into_the_newest() -> None:
    """A route stamped one step back, from a car that has since moved 1 m forward and turned
    0.1 rad left, reads as 1 m nearer and turned 0.1 rad right."""
    now = np.eye(4)
    yaw = 0.1
    now[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    now[:3, 3] = (1.0, 0.0, 0.0)
    moved = route_in_frame(np.array([[11.0, 0.0, 0.0]]), now, np.eye(4))
    expected = np.array([math.cos(yaw) * 10.0, -math.sin(yaw) * 10.0, 0.0])
    np.testing.assert_allclose(moved[0], expected, atol=1e-12)
    np.testing.assert_allclose(route_in_frame(np.array([[3.0, 4.0, 0.0]]), now, now), [[3.0, 4.0, 0.0]])


# -- cameras ------------------------------------------------------------------------


def _observation(cameras: dict[str, bool], t_us: int = 0, **fields) -> Observation:
    """*cameras*: name -> whether it is sent with a pinhole K."""
    k = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 90.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[0, 3] = 10.0 * t_us / 1e6
    return Observation(
        images={name: np.full((180, 320, 3), 7 + i, dtype=np.uint8) for i, name in enumerate(cameras)},
        ego2world=pose,
        canbus=np.zeros(18),
        timestamp_us=t_us,
        command=Command.FOLLOW_REFERENCE,
        camera2image={name: k for name, pinhole in cameras.items() if pinhole},
        camera2ego={name: np.eye(4) for name in cameras},
        **fields,
    )


PINHOLE_THREE = {"CAM_FRONT": True, "CAM_FRONT_LEFT": True, "CAM_FRONT_RIGHT": True}
FTHETA_FOUR = {"CAM_FRONT": False, "CAM_FRONT_LEFT": False, "CAM_FRONT_RIGHT": False, TELE_CAMERA: False}


def test_three_pinhole_cameras_fill_the_slots_as_before() -> None:
    assert slots_for(_observation(PINHOLE_THREE)) == [
        (0, "CAM_FRONT_LEFT", "fill"),
        (1, "CAM_FRONT", "fill"),
        (2, "CAM_FRONT_RIGHT", "fill"),
        (6, "CAM_FRONT", "zoom"),
    ]


def test_a_real_telephoto_fills_slot_6_instead_of_the_crop() -> None:
    slots = slots_for(_observation({**PINHOLE_THREE, TELE_CAMERA: True}))
    assert slots[-1] == (6, TELE_CAMERA, "zoom")
    assert [camera for _, camera, _ in slots].count("CAM_FRONT") == 1


def test_f_theta_cameras_are_passed_as_rendered() -> None:
    assert slots_for(_observation(FTHETA_FOUR)) == [
        (0, "CAM_FRONT_LEFT", "raw"),
        (1, "CAM_FRONT", "raw"),
        (2, "CAM_FRONT_RIGHT", "raw"),
        (6, TELE_CAMERA, "raw"),
    ]


def test_a_raw_frame_is_the_rendered_pixels_untouched() -> None:
    """No crop, no resize, uint8: what AlpaSim's driver hands the processor."""
    obs = _observation(FTHETA_FOUR)
    warped = Frame(obs, slots_for(obs)).warped()
    for camera in FTHETA_FOUR:
        frame = warped[f"{camera}:raw"]
        assert frame.dtype == np.uint8 and frame.shape == (180, 320, 3)
        np.testing.assert_array_equal(frame, obs.images[camera])


def test_the_policy_decodes_the_telephoto_when_it_is_sent() -> None:
    from paddock.adapters.models.alpamayo15.policy import POLICY

    assert TELE_CAMERA in POLICY.cameras


# -- the wire ------------------------------------------------------------------------


def test_plan_dt_and_route_time_travel_only_when_set() -> None:
    plain = _observation(PINHOLE_THREE)
    wire = plain.to_wire()
    assert "plan_dt" not in wire and "route_timestamp_us" not in wire
    back = Observation.from_wire(wire, cameras=tuple(PINHOLE_THREE))
    assert back.plan_dt is None and back.route_timestamp_us is None
    asked = _observation(PINHOLE_THREE, route_xy=np.zeros((3, 2)), route_timestamp_us=5, plan_dt=0.1)
    back = Observation.from_wire(asked.to_wire(), cameras=tuple(PINHOLE_THREE))
    assert (back.plan_dt, back.route_timestamp_us) == (0.1, 5)


# -- infer (the model's environment) -------------------------------------------------


def _drive(observations, samples: int = 1):
    from paddock.adapters.models.alpamayo15.tests.test_unchanged_inputs import run_case

    return run_case(observations, samples)


@pytest.fixture
def model_env():
    pytest.importorskip("torch", reason="the model's environment supplies torch")
    pytest.importorskip("scipy", reason="the model's environment supplies scipy")
    pytest.importorskip("alpamayo1_5", reason="the model's environment supplies alpamayo1_5")


def _alpasim_episode(**fields) -> list[Observation]:
    """20 Hz frames over 1.6 s, four f-theta cameras, as the bridge sends them on AlpaSim."""
    return [_observation(FTHETA_FOUR, t_us=50_000 * i, **fields) for i in range(33)]


def test_plan_dt_returns_all_64_points_at_0_1_s_with_the_models_headings(model_env) -> None:
    result = _drive(_alpasim_episode(plan_dt=0.1))
    waypoints = np.array(result["waypoints"])
    assert waypoints.shape == (64, 3) and result["dt"] == 0.1
    t = np.arange(1, 65) * 0.1
    np.testing.assert_allclose(waypoints[:, 0], 9.0 * t, atol=1e-5)
    np.testing.assert_allclose(waypoints[:, 2], 0.02 * t, atol=1e-5)  # the stub's yaw, not a derived one
    assert "extras" in result["wire_keys"]


def test_another_plan_dt_is_refused(model_env) -> None:
    with pytest.raises(ValueError, match="plan_dt"):
        _drive(_alpasim_episode(plan_dt=0.5))


def test_without_plan_dt_the_plan_is_the_usual_eight_points(model_env) -> None:
    result = _drive(_alpasim_episode())
    assert len(result["waypoints"]) == 8 and result["dt"] == 0.5
    assert "extras" not in result["wire_keys"]


def test_alpasim_frames_reach_the_prompt_as_rendered_in_slot_order(model_env) -> None:
    result = _drive(_alpasim_episode(plan_dt=0.1))
    images = [m for m in result["messages"] if m[0] == "image"]
    assert len(images) == 16 and all(m[1] == [3, 180, 320] for m in images)
    names = [m[1] for m in result["messages"] if m[0] == "user" and m[1].endswith(": ")]
    assert names == [
        "Front left camera: ",
        "Front camera: ",
        "Front right camera: ",
        "Front telephoto camera: ",
    ]
    assert result["camera_slots"] == [
        "0:CAM_FRONT_LEFT:raw",
        "1:CAM_FRONT:raw",
        "2:CAM_FRONT_RIGHT:raw",
        "6:CAM_FRONT_TELE:raw",
    ]


def test_a_route_with_its_time_gives_the_native_navigation_text(model_env) -> None:
    route = _bend(20.0, +1.0)
    episode = [
        _observation(FTHETA_FOUR, t_us=50_000 * i, route_xy=route, route_timestamp_us=50_000 * i)
        for i in range(33)
    ]
    assert _drive(episode)["nav_text"] == "Turn left in 20m"


def test_a_route_without_its_time_keeps_the_command_path(model_env) -> None:
    """Bench2Drive sends an ego-frame route with no timestamp: the old path, unchanged."""
    episode = [_observation(FTHETA_FOUR, t_us=50_000 * i, route_xy=_bend(20.0, +1.0)) for i in range(33)]
    assert _drive(episode)["nav_text"] == "Continue straight"


def test_a_route_stamped_earlier_is_moved_to_the_newest_pose(model_env) -> None:
    """The car drives 10 m/s; a route stamped 0.5 s ago has its bend 5 m closer now."""
    route = _bend(20.0, +1.0)
    episode = [
        _observation(FTHETA_FOUR, t_us=50_000 * i, route_xy=route, route_timestamp_us=50_000 * (i - 10))
        for i in range(10, 43)
    ]
    assert _drive(episode)["nav_text"] == "Turn left in 15m"


def test_a_route_stamped_outside_the_kept_history_is_refused(model_env) -> None:
    episode = [
        _observation(FTHETA_FOUR, t_us=50_000 * i, route_xy=_bend(20.0, +1.0), route_timestamp_us=-1_000_000)
        for i in range(33)
    ]
    with pytest.raises(ValueError, match="route_timestamp_us"):
        _drive(episode)


def test_no_route_on_alpasim_is_no_navigation_text(model_env) -> None:
    """AlpaSim's own driver gives the model no route sentence when it has no route."""
    assert _drive(_alpasim_episode(plan_dt=0.1))["nav_text"] is None
