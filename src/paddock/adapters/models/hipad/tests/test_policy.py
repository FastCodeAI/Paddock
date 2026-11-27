"""What HiP-AD is fed, checked in paddock's own env: no model, no GPU, no weights.

The model itself, against HiP-AD's own CARLA agent on the same frames, is ``agent_reference.py``
(``make check-hipad``): every input equal, the first plan within 1e-3 m.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from paddock import conformance
from paddock.adapters.models.hipad import inputs as hi
from paddock.adapters.models.hipad.policy import AUTHORS_PROJECT_DIR, POLICY, repoint
from paddock.adapters.models.uniad import bev_planners as bp
from paddock.policy.protocol import Command
from paddock.registry import Registry


def _pose(x: float, y: float, yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:2, 3] = [x, y]
    return out


# -- the command --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (Command.LEFT, hi.LEFT),
        (Command.RIGHT, hi.RIGHT),
        (Command.STRAIGHT, hi.STRAIGHT),
        (Command.FOLLOW_REFERENCE, hi.LANEFOLLOW),
    ],
)
def test_paddock_commands_map_to_bench2drives_road_options(command: Command, expected: int) -> None:
    assert hi.command_index(command, None, None) == expected


@pytest.mark.parametrize(("side", "expected"), [("LEFT", hi.CHANGELANELEFT), ("RIGHT", hi.CHANGELANERIGHT)])
def test_a_lane_change_within_reach_is_the_command(side: str, expected: int) -> None:
    """Paddock's Bench2Drive client sends a lane change as STRAIGHT; within 4 m (the planner's) it is one."""
    assert hi.command_index(Command.STRAIGHT, side, 0.0) == expected
    assert hi.command_index(Command.STRAIGHT, side, hi.LANE_CHANGE_NEAR_M) == expected
    assert hi.command_index(Command.STRAIGHT, side, hi.LANE_CHANGE_NEAR_M + 0.1) == hi.STRAIGHT
    # A turn at a junction stays a turn whatever lane change lies ahead.
    assert hi.command_index(Command.LEFT, side, 0.0) == hi.LEFT


def test_the_command_is_one_hot_over_six() -> None:
    np.testing.assert_array_equal(hi.one_hot(hi.LANEFOLLOW), [0, 0, 0, 1, 0, 0])


# -- the target point ---------------------------------------------------------------------------------


def test_ego_axes_become_the_lidars() -> None:
    """Forward is the lidar's +y, left its -x (x right)."""
    np.testing.assert_allclose(hi.ego_to_lidar_axes([10.0, 0.0]), [0.0, 10.0])
    np.testing.assert_allclose(hi.ego_to_lidar_axes([0.0, 2.0]), [-2.0, 0.0])


def test_the_tracker_passes_points_within_four_metres_as_the_agent_does() -> None:
    """The agent's RoutePlanner(4.0, 50.0): a point within 4 m is passed, the target is the next."""
    tracker = hi.RouteTracker()
    route = np.array([[-10.0, 0.0], [0.0, 0.0], [30.0, 0.0], [80.0, 0.0]])
    np.testing.assert_allclose(tracker.target(route, np.eye(4)), [30.0, 0.0])
    # 3 m short of the 30 m point the car is within 4 m of it: it is passed, and 80 m is next.
    pose = _pose(27.0, 0.0, 0.0)
    np.testing.assert_allclose(tracker.target(route - [27.0, 0.0], pose), [53.0, 0.0])


def test_a_passed_point_stays_passed_in_the_next_window() -> None:
    """The client sends a window of the route each tick; a point already passed is not taken again."""
    tracker = hi.RouteTracker()
    route_world = np.array([[-10.0, 0.0], [0.0, 0.0], [30.0, 0.0], [80.0, 0.0], [130.0, 0.0]])
    tracker.target(route_world - [27.0, 0.0], _pose(27.0, 0.0, 0.0))  # passes the 30 m point
    # 10 m on, the 30 m point is 7 m behind -- outside 4 m -- but the agent does not go back to it.
    np.testing.assert_allclose(tracker.target(route_world - [37.0, 0.0], _pose(37.0, 0.0, 0.0)), [43.0, 0.0])


def test_the_target_follows_the_cars_heading() -> None:
    tracker = hi.RouteTracker()
    pose = _pose(5.0, 0.0, 90.0)  # facing north
    route_ego = np.array([[-5.0, 0.0], [25.0, 0.0], [55.0, 0.0]])  # straight ahead, in the ego frame
    np.testing.assert_allclose(tracker.target(route_ego, pose), [25.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(tracker.front, [5.0, -5.0], atol=1e-9)  # held in the world: 5 m behind, south


def test_the_route_decides_the_target_where_it_is_in_the_observations_frame() -> None:
    route = np.array([[-5.0, 0.0], [0.0, 0.0], [40.0, 0.0]])
    record = hi.target_record(route, None, 0, np.eye(4), hi.LANEFOLLOW, hi.RouteTracker())
    assert record["source"] == "route"
    np.testing.assert_allclose(record["target_point"], [0.0, 40.0])
    # A route stamped at another time is in another frame: the command decides instead.
    other = hi.target_record(route, 1, 0, np.eye(4), hi.LANEFOLLOW, hi.RouteTracker())
    assert other["source"] == "command"


@pytest.mark.parametrize(
    ("index", "ego"),
    [
        (hi.LANEFOLLOW, (hi.COMMAND_TARGET_M, 0.0)),
        (hi.STRAIGHT, (hi.COMMAND_TARGET_M, 0.0)),
        (hi.LEFT, (hi.COMMAND_TARGET_M / math.sqrt(2), hi.COMMAND_TARGET_M / math.sqrt(2))),
        (hi.RIGHT, (hi.COMMAND_TARGET_M / math.sqrt(2), -hi.COMMAND_TARGET_M / math.sqrt(2))),
        (hi.CHANGELANELEFT, (hi.COMMAND_TARGET_M, hi.LANE_WIDTH_M)),
        (hi.CHANGELANERIGHT, (hi.COMMAND_TARGET_M, -hi.LANE_WIDTH_M)),
    ],
)
def test_without_a_route_the_command_places_the_target(index: int, ego: tuple[float, float]) -> None:
    record = hi.target_record(None, None, 0, np.eye(4), index, hi.RouteTracker())
    assert record["source"] == "command"
    np.testing.assert_allclose(record["target_point"], hi.ego_to_lidar_axes(np.asarray(ego)), atol=1e-5)


# -- the frame and the images -------------------------------------------------------------------------


def test_the_lidar_is_bench2drives_mount_in_lidar_top_axes() -> None:
    assert bp.has_lidar_top_axes(hi.LIDAR2EGO)
    np.testing.assert_allclose(hi.LIDAR2EGO[:3, 3], [-0.39, 0.0, 1.84])


def test_the_image_pipeline_is_the_agents() -> None:
    """``get_augmentation`` at 1600x900 -> 640x352: x0.4 to 640x360, cut 8 rows from the top."""
    aug = hi.aug_config()
    assert aug["resize"] == pytest.approx(0.4)
    assert aug["resize_dims"] == (640, 360)
    assert aug["crop"] == (0, 8, 640, 360)
    assert not aug["flip"] and aug["rotate"] == 0


def test_the_authors_path_is_moved_to_the_checkout() -> None:
    cfg = {"a": f"{AUTHORS_PROJECT_DIR}/data/kmeans/x.npy", "b": [f"{AUTHORS_PROJECT_DIR}/y", "keep"], "c": 3}
    out = repoint(cfg, AUTHORS_PROJECT_DIR, "/repo")
    assert out == {"a": "/repo/data/kmeans/x.npy", "b": ["/repo/y", "keep"], "c": 3}


# -- the policy contract and its declaration ----------------------------------------------------------


class StubHiPAD(POLICY):
    """The adapter with the model replaced: plans 4 m every 0.5 s ahead in the lidar frame."""

    def __init__(self) -> None:
        super().__init__()
        self.config = self.checkpoint = __file__

    def _build(self) -> None:
        self.model = object()

    def _results(self, obs, lidar2ego):
        return {"t": obs.timestamp_us}

    def _forward(self, results):
        return np.c_[np.zeros(6), 4.0 * np.arange(1, 7)]

    def _prepare(self, results):
        return results

    def _run(self, prepared, results):
        return self._forward(prepared)

    def _reset_model(self) -> None:
        pass


def test_the_adapter_obeys_the_policy_contract() -> None:
    findings = conformance.policy_findings(StubHiPAD())
    failures = [f for f in findings if f.failed]
    assert not failures, "\n" + "\n".join(f.render() for f in failures)


def test_the_declaration_matches_the_policy() -> None:
    """model.yaml's cameras, plan grid, rate and settings are the ones the policy reads."""
    spec = Registry.load().models["hipad"]
    assert tuple(spec.cameras) == hi.CAMERAS == POLICY.cameras
    assert spec.waypoint_dt == bp.PLAN_DT and spec.horizon_s == bp.PLAN_DT * bp.PLAN_STEPS
    assert spec.planning_hz == 1 / bp.PLAN_DT, "its temporal memory steps 0.5 s at a time"
    assert {POLICY.config_env, POLICY.checkpoint_env, "HIPAD_REPO"} <= set(spec.runtime.env)
