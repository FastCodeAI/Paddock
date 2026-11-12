"""What Edge is shown on AlpaSim: its own six cameras through their own lenses, and the route.

AlpaSim's bridge sends pixels with no pinhole intrinsics (f-theta lenses), a real
telephoto under CAM_FRONT_TELE when the profile renames it so, FOLLOW_REFERENCE as
the command, and the route polyline with the time of the frame it is in. Every part
is switched on by that input alone; tests/test_policy.py pins that a source sending
none of it gets what it got before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock.adapters.models.alpamayo2edge import inputs
from paddock.adapters.models.alpamayo2edge.policy import Alpamayo2EdgePolicy
from paddock.adapters.models.alpamayo2edge.tests.test_policy import _camera2ego, _loaded
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation

#: AlpaSim's cameras renamed by alpasim.yaml, and where each points (degrees left of forward).
ALPASIM_BEARINGS = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 60,
    "CAM_FRONT_RIGHT": -60,
    "CAM_BACK_LEFT": 140,
    "CAM_BACK_RIGHT": -140,
    "CAM_FRONT_TELE": 0,
}
#: A tenth of AlpaSim's 1920x1080, so the tests stay fast; `fit` resizes it to the vendor's size.
SMALL = (108, 192, 3)
STEP_S = 0.05
SPEED = 8.0


def _frame(camera: str, t: float) -> np.ndarray:
    """A frame whose pixels say which camera and which time it is."""
    image = np.zeros(SMALL, dtype=np.uint8)
    image[..., 0] = list(ALPASIM_BEARINGS).index(camera) * 40
    image[..., 1] = int(round((t + 5.0) * 20)) % 256
    return image


def _straight(t: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = SPEED * t
    return pose


def _alpasim(
    times_s=None,
    cameras=tuple(ALPASIM_BEARINGS),
    route_xy: np.ndarray | None = None,
    route_time_s: float | None = None,
) -> list[Observation]:
    """AlpaSim at 20 Hz through the bridge: pixels, no intrinsics, the route on the newest frame."""
    times_s = np.round(np.arange(-3.0, 1e-9, STEP_S), 3) if times_s is None else times_s
    sequence = []
    for k, t in enumerate(times_s):
        canbus = np.zeros(18)
        canbus[CANBUS_VELOCITY] = SPEED
        newest = k == len(times_s) - 1
        sequence.append(
            Observation(
                images={camera: _frame(camera, t) for camera in cameras},
                ego2world=_straight(t),
                canbus=canbus,
                timestamp_us=int(round(t * 1e6)),
                command=Command.FOLLOW_REFERENCE,
                # f-theta lenses: the bridge sends no K for them.
                camera2image={},
                camera2ego={camera: _camera2ego(ALPASIM_BEARINGS[camera]) for camera in cameras},
                route_xy=route_xy if newest else None,
                route_timestamp_us=(
                    int(round(route_time_s * 1e6)) if newest and route_time_s is not None else None
                ),
                plan_dt=0.1,
            )
        )
    return sequence


def _plan(policy: Alpamayo2EdgePolicy, sequence: list[Observation]):
    for obs in sequence[:-1]:
        policy.observe(obs)
    return policy.infer(sequence[-1])


def _left_turn(at_m: float, n: int = 60) -> np.ndarray:
    """Straight ahead to `at_m`, then a quarter turn to the left: (N, 2), x forward, y left."""
    ahead = np.column_stack([np.linspace(0.0, at_m, n // 2), np.zeros(n // 2)])
    side = np.column_stack([np.full(n // 2, at_m), np.linspace(1.0, 30.0, n // 2)])
    return np.vstack([ahead, side])


# -- cameras --------------------------------------------------------------------------


def test_the_node_decodes_the_telephoto_alpasim_sends() -> None:
    assert inputs.TELE_CAMERA in Alpamayo2EdgePolicy.cameras


def test_a_real_telephoto_fills_camera_6_whole_and_the_others_are_their_own_lens() -> None:
    policy = _loaded()
    trajectory = _plan(policy, _alpasim())
    sample = policy.runs[-1]
    assert sample.black_slots == []
    assert sample.frames.shape == (22, inputs.FRAME_SIZE[1], inputs.FRAME_SIZE[0], 3)
    # Every slot is its own camera at the times the vendor's loader asks for, 0.1 s grid included.
    assert sample.frame_times_s[6] == sample.frame_times_s[1] == [-2.0, -1.0, -0.5, -0.1, 0.0]
    assert sample.frame_times_s[3] == [-1.0, -0.1, 0.0]
    # Camera 6 is the telephoto frame resized whole, not a 30 deg crop of the front camera.
    tele = sample.frames[sample.camera_ids == 6][-1]
    assert np.array_equal(tele, inputs.fit(_frame("CAM_FRONT_TELE", 0.0)))
    front = sample.frames[sample.camera_ids == 1][-1]
    assert np.array_equal(front, inputs.fit(_frame("CAM_FRONT", 0.0)))
    assert policy.diagnostics()["slots"][-1] == "6: CAM_FRONT_TELE"
    assert policy.diagnostics()["slot_geometry"][6]["hfov_deg"] == "as rendered"
    # All 64 poses at the spacing the bridge said it drives.
    assert trajectory.waypoints.shape == (inputs.PLAN_STEPS, 3) and trajectory.dt == pytest.approx(0.1)


def test_a_frame_already_the_vendors_size_is_passed_as_it_came() -> None:
    image = np.full((inputs.FRAME_SIZE[1], inputs.FRAME_SIZE[0], 3), 7, dtype=np.uint8)
    assert inputs.fit(image) is image or np.shares_memory(inputs.fit(image), image)


def test_an_ftheta_front_camera_cannot_be_cropped_into_the_telephoto_slot() -> None:
    """Without CAM_FRONT_TELE camera 6 is a 30 deg crop, and a crop needs a K the f-theta
    lens does not have: refused by name rather than approximated."""
    policy = _loaded()
    cameras = tuple(c for c in ALPASIM_BEARINGS if c != inputs.TELE_CAMERA)
    with pytest.raises(ValueError, match="CAM_FRONT came without pinhole intrinsics.*CAM_FRONT_TELE"):
        _plan(policy, _alpasim(cameras=cameras))


# -- route ----------------------------------------------------------------------------


def test_the_route_decides_the_sentence_not_the_follow_reference_command() -> None:
    policy = _loaded()
    _plan(policy, _alpasim(route_xy=_left_turn(20.0), route_time_s=0.0))
    assert policy.runs[-1].nav_text == "Turn left in 20m"
    assert policy.runs[-1].nav_from == "route" == policy.diagnostics()["last"]["nav_from"]


def test_a_route_from_an_earlier_frame_is_moved_to_the_newest_one_first() -> None:
    """The route is stamped 0.5 s back; the car has driven 4 m since, so the turn is 4 m nearer."""
    policy = _loaded()
    _plan(policy, _alpasim(route_xy=_left_turn(24.0), route_time_s=-0.5))
    assert policy.runs[-1].nav_text == "Turn left in 20m"


def test_a_route_with_no_usable_point_falls_back_to_the_command() -> None:
    policy = _loaded()
    _plan(policy, _alpasim(route_xy=np.full((5, 2), np.nan), route_time_s=0.0))
    assert policy.runs[-1].nav_text == "Continue straight"
    assert policy.runs[-1].nav_from == "command"


def test_a_route_without_its_time_is_not_read() -> None:
    """Without the time, the frame the route is in is unknown: the command decides, as before."""
    policy = _loaded()
    _plan(policy, _alpasim(route_xy=_left_turn(20.0), route_time_s=None))
    assert policy.runs[-1].nav_text == "Continue straight"


def test_a_route_stamped_outside_the_history_is_refused() -> None:
    policy = _loaded()
    with pytest.raises(ValueError, match="outside the kept history"):
        _plan(policy, _alpasim(route_xy=_left_turn(20.0), route_time_s=-10.0))


@pytest.mark.parametrize(
    "route, expected",
    [
        (_left_turn(20.0), "Turn left in 20m"),
        (_left_turn(20.0) * [1.0, -1.0], "Turn right in 20m"),
        (_left_turn(60.0), "Continue straight"),  # beyond the 40 m lookahead
        # Turning at 3 m: the bend nearer than 5 m is ignored, the heading it leads to is not.
        (_left_turn(3.0), "Turn left in 5m"),
        (np.column_stack([np.linspace(0, 50, 30), 0.05 * np.linspace(0, 50, 30)]), "Continue straight"),
        (np.array([[0.0, 0.0], [10.0, 0.0]]), "Continue straight"),
        (np.empty((0, 2)), None),
    ],
)
def test_the_route_is_classified_as_alpasims_driver_classifies_it(route, expected) -> None:
    assert inputs.route_to_nav_text(route) == expected


def test_the_port_agrees_with_alpamayo15s_port_of_the_same_function() -> None:
    """Both adapters port AlpaSim's nav_text.py number for number; they must not drift apart."""
    from paddock.adapters.models.alpamayo15 import route_nav

    rng = np.random.default_rng(3)
    for _ in range(200):
        headings = np.cumsum(rng.normal(0.0, 0.25, 40))
        steps = rng.uniform(0.5, 3.0, 40)[:, None] * np.column_stack([np.cos(headings), np.sin(headings)])
        route = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
        assert inputs.route_to_nav_text(route) == route_nav.route_to_nav_text(route)
    frame, other = np.eye(4), np.eye(4)
    other[:3, 3] = [3.0, 1.0, 0.0]
    yaw = math.radians(20)
    other[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    xyz = rng.normal(size=(10, 3))
    assert np.allclose(inputs.route_in_frame(xyz, frame, other), route_nav.route_in_frame(xyz, frame, other))
