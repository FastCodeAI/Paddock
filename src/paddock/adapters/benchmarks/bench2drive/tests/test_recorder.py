"""The analysis record's geometry, without CARLA.

The agent hands ``recorder`` plain CARLA-world numbers; what comes out is in
paddock's ego frame. A sign error here would put every recorded actor on the
wrong side of the car and nothing downstream would notice, so the frame
conventions are pinned with points whose answer is obvious.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive import recorder


def test_to_ego_at_yaw_zero_keeps_forward_and_puts_carla_right_on_negative_y() -> None:
    # CARLA is left-handed: at yaw 0 the car faces +x and +y is to its RIGHT.
    ego = recorder.to_ego(np.array([[11.0, 5.0], [10.0, 6.0]]), 10.0, 5.0, 0.0)
    assert ego[0] == pytest.approx([1.0, 0.0])
    assert ego[1] == pytest.approx([0.0, -1.0])


def test_to_ego_follows_the_heading() -> None:
    # Facing +y (yaw 90): a point further along +y is straight ahead, and +x is
    # then on the car's left.
    ego = recorder.to_ego(np.array([[0.0, 3.0], [2.0, 0.0]]), 0.0, 0.0, 90.0)
    assert ego[0] == pytest.approx([3.0, 0.0], abs=1e-9)
    assert ego[1] == pytest.approx([0.0, 2.0], abs=1e-9)


def test_model_view_dir_keeps_openpilots_names_and_names_every_other_camera() -> None:
    # model_road / model_wide are what openpilot's recorded routes already hold.
    assert recorder.model_view_dir("CAM_OP_ROAD") == "model_road"
    assert recorder.model_view_dir("CAM_OP_WIDE") == "model_wide"
    # A six-view model (AutoE2E) gets one directory per view, none colliding.
    six = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    dirs = [recorder.model_view_dir(c) for c in six]
    assert dirs[1] == "model_front_left"
    assert len(set(dirs)) == len(six)
    # front/ and bev/ are the recording cameras' own directories; no model view lands there.
    assert not {"front", "bev"} & set(dirs)


def test_wrap_deg() -> None:
    assert recorder.wrap_deg(190.0) == pytest.approx(-170.0)
    assert recorder.wrap_deg(-190.0) == pytest.approx(170.0)
    assert recorder.wrap_deg(-180.0) == 180.0


def _straight(n: int = 200) -> recorder.Route:
    return recorder.Route([[float(i), 0.0] for i in range(n)], ["LANEFOLLOW"] * n)


def test_route_length_and_locate() -> None:
    route = _straight()
    assert route.length == pytest.approx(199.0)
    assert route.locate(10.2, 1.5) == (10, pytest.approx(math.hypot(0.2, 1.5)))


def test_route_locate_falls_back_to_a_full_search_when_the_window_has_lost_the_ego() -> None:
    route = _straight()
    route.locate(0.0, 0.0)
    index, off = route.locate(150.0, 0.5)
    assert index == 150 and off == pytest.approx(0.5)


def test_route_locate_prefers_the_pass_it_is_on_where_the_route_doubles_back() -> None:
    # Out along y = 0 and back along y = 3: near x = 5 both passes are close,
    # and only the window keeps the ego on the pass it is driving.
    out = [[float(i), 0.0] for i in range(20)]
    back = [[float(i), 3.0] for i in range(19, -1, -1)]
    route = recorder.Route(out + back, ["LANEFOLLOW"] * 40)
    for x in range(6):
        index, _ = route.locate(float(x), 1.4)
    assert index == 5


def test_next_on_route_reports_along_route_distance() -> None:
    route = _straight()
    marks = [{"id": 7, "i": 50}, {"id": 8, "i": 120}]
    assert recorder.next_on_route(marks, 40, route) == {"id": 7, "d": 10.0}
    assert recorder.next_on_route(marks, 51, route) == {"id": 8, "d": 69.0}
    assert recorder.next_on_route(marks, 121, route) is None


def test_actor_row_is_ego_frame_and_drops_far_actors() -> None:
    ego = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    # 10 m ahead, 2 m to the car's right (CARLA +y), heading 90 deg clockwise of
    # the ego -- i.e. pointing to its right, which is negative in a y-left frame.
    row = recorder.actor_row(42, 10.0, 2.0, 90.0, 3.0, 4.0, ego)
    assert row == [42, 10.0, -2.0, -90.0, 5.0]
    assert recorder.actor_row(1, recorder.ACTOR_RADIUS_M + 1.0, 0.0, 0.0, 0.0, 0.0, ego) is None


def test_rounded_is_json_safe() -> None:
    value = {"a": [1.234567, float("nan")], "b": np.float32(2.5), "c": "x", "d": None}
    out = recorder.rounded(value, 2)
    assert out == {"a": [1.23, None], "b": 2.5, "c": "x", "d": None}
    json.dumps(out, allow_nan=False)


def test_tick_record_keeps_the_whole_reply() -> None:
    reply = {
        "trajectory": [[0.5 * (i + 1), 0.0] for i in range(20)],
        "dt": 0.5,
        "extras": {"lead": [{"d": 12.3456, "prob": 0.9}], "plan_v0": 4.0},
    }
    record = recorder.tick_record(
        3,
        0.15,
        12.34,
        {"x": 1.0, "y": 2.0, "yaw": 0.0},
        {"command": 3},
        reply,
        {"steer": 0.0, "throttle": 0.5, "brake": 0.0, "v_des": 4.0},
        route={"i": 0, "s": 0.0, "dev": 0.2, "frac": 0.0},
        actors=[[42, 10.0, -2.0, -90.0, 5.0]],
    )
    assert len(record["plan"]) == 20
    assert record["extras"]["lead"][0]["d"] == 12.346
    line = recorder.dumps(record)
    assert "\n" not in line and json.loads(line)["t"] == 3
