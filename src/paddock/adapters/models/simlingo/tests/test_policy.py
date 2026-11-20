"""SimLingo's adapter, without its model: what runs in paddock's own environment.

The model itself (InternVL2-1B and its checkpoint) needs the GPU and SimLingo's environment; it is
checked by `paddock verify --model simlingo`. Here the vendor forward pass is stubbed, and what is
paddock's own is tested: the route the agent's RoutePlanner would hold, the frame flips, the bonnet
cut, and the plan padded to the closed-loop horizon.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock import conformance
from paddock.adapters.benchmarks.bench2drive import rig
from paddock.adapters.models.simlingo import inputs
from paddock.adapters.models.simlingo.policy import CAMERA, POLICY, horizon_s
from paddock.policy.protocol import N_WAYPOINTS, Command, Trajectory


def _pose(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """An ego2world for an ego at (x, y) facing world +x."""
    m = np.eye(4)
    m[:2, 3] = [x, y]
    return m


#: A straight leaderboard-style plan along world +x: a point every 50 m, the first 10 m behind.
STRAIGHT = np.array([[-10.0, 0.0], [40.0, 0.0], [90.0, 0.0], [140.0, 0.0], [190.0, 0.0]])


def _ego(world: np.ndarray, ego_x: float) -> np.ndarray:
    return world - [ego_x, 0.0]


class _Stub(POLICY):
    """The policy with the vendor forward pass replaced: straight ahead, 1 m per 0.25 s."""

    def load(self) -> None:
        self.model = object()

    def drive(self, rgb, speed, route_text, targets_sim):
        self.seen = {"rgb": rgb, "speed": speed, "route": route_text, "targets": targets_sim}
        self.calls = getattr(self, "calls", 0) + 1
        t = np.arange(1, inputs.NATIVE_POINTS + 1)
        return np.c_[t * 1.0, np.zeros(len(t))]


# -- the contract ---------------------------------------------------------------------


def test_the_policy_meets_the_contract_with_its_model_stubbed() -> None:
    findings = conformance.policy_findings(_Stub())
    failed = [f"{f.check}: {f.detail}" for f in findings if f.failed]
    assert not failed, failed


def test_planning_before_load_is_refused() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        POLICY().infer(conformance.synthetic_observation(cameras=(CAMERA,)))


def test_it_reads_its_own_camera_or_the_front_one_at_its_own_spacing() -> None:
    assert POLICY.cameras == ("CAM_SIMLINGO", "CAM_FRONT")
    assert POLICY.waypoint_dt == 0.25


# -- the camera -------------------------------------------------------------------------


def test_the_rig_renders_simlingos_camera_where_its_agent_mounts_it() -> None:
    cam = rig.cameras(rig.DEFAULT)["CAM_SIMLINGO"]
    # team_code/config_simlingo.py: pos [-1.5, 0.0, 2.0], rot [0, 0, 0], 1024x512, fov 110.
    assert cam == {"pose": [-1.5, 0.0, 2.0, 0.0, 0.0, 0.0], "fov": 110.0, "size": [1024, 512]}


def test_the_bonnet_cut_keeps_the_agents_rows() -> None:
    assert inputs.cropped_rows(512) == 359  # int(512 - (512 * 4.8) // 16)


def _pinhole_frame(width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, float]:
    """A white frame from a centred pinhole camera of the given width and horizontal fov."""
    return np.full((height, width, 3), 255, dtype=np.uint8), width / (2 * np.tan(np.radians(hfov_deg) / 2))


def test_a_narrower_camera_sits_at_its_true_angular_scale_in_the_110_deg_frame() -> None:
    rgb, f = _pinhole_frame(1600, 900, 70.0)
    canvas = inputs.to_canvas(rgb, f, f, 800.0, 450.0)
    assert canvas.shape == (512, 1024, 3)
    lit = np.flatnonzero(canvas[256, :, 0])
    # 70 deg of a 110 deg frame: tan(35) / tan(55) of the half width, either side of the centre.
    half = 512 * np.tan(np.radians(35)) / np.tan(np.radians(55))
    assert abs((lit[-1] - lit[0] + 1) / 2 - half) <= 1.5
    assert abs((lit[0] + lit[-1]) / 2 - 511.5) <= 1.0
    assert canvas[256, 0].sum() == 0 and canvas[256, -1].sum() == 0  # unseen: black


def test_the_frame_keeps_the_principal_point_on_the_centre() -> None:
    rgb, f = _pinhole_frame(1000, 600, 60.0)
    rgb[:, 600:] = 0  # light only the left part, up to x = 600 > cx = 400
    canvas = inputs.to_canvas(rgb, f, f, 400.0, 300.0)
    lit = np.flatnonzero(canvas[256, :, 0])
    # The column at cx lands on the canvas centre; light spans (600 - 400) * scale right of it.
    assert abs(lit[-1] - (512 + 200 * inputs.CANVAS_F / f)) <= 1.5


def test_its_own_camera_is_used_untouched_where_the_source_renders_it() -> None:
    policy = _Stub()
    policy.load()
    obs = conformance.synthetic_observation(cameras=(CAMERA,), width=1024, height=512)
    policy.infer(obs)
    assert policy.seen["rgb"] is obs.images[CAMERA]
    assert policy.diagnostics()["counts"]["own_camera"] == 1


def test_a_dataset_front_camera_is_placed_in_the_110_deg_frame() -> None:
    policy = _Stub()
    policy.load()
    policy.infer(conformance.synthetic_observation(cameras=("CAM_FRONT",)))
    assert policy.seen["rgb"].shape == (512, 1024, 3)
    assert policy.diagnostics()["counts"]["front_in_110_frame"] == 1


# -- the route --------------------------------------------------------------------------


def test_at_the_start_the_targets_are_the_next_two_plan_points() -> None:
    tracker = inputs.RouteTracker()
    targets = tracker.targets(_ego(STRAIGHT, 0.0), _pose(0.0))
    np.testing.assert_allclose(targets, [[40.0, 0.0], [90.0, 0.0]])


def test_a_point_within_7_5_m_is_passed_and_stays_passed() -> None:
    tracker = inputs.RouteTracker()
    tracker.targets(_ego(STRAIGHT, 0.0), _pose(0.0))
    # 5 m short of the 40 m point: within 7.5 m, so the agent pops up to it.
    targets = tracker.targets(_ego(STRAIGHT, 35.0), _pose(35.0))
    np.testing.assert_allclose(targets, [[55.0, 0.0], [105.0, 0.0]])
    # Past it and 9 m on: it is no longer within 7.5 m, but stays popped.
    targets = tracker.targets(_ego(STRAIGHT, 49.0), _pose(49.0))
    np.testing.assert_allclose(targets, [[41.0, 0.0], [91.0, 0.0]])


def test_reset_starts_the_route_again() -> None:
    tracker = inputs.RouteTracker()
    tracker.targets(_ego(STRAIGHT, 35.0), _pose(35.0))
    tracker.reset()
    np.testing.assert_allclose(tracker.targets(_ego(STRAIGHT, 0.0), _pose(0.0))[0], [40.0, 0.0])


def test_a_route_to_the_left_reaches_simlingo_with_y_to_the_right_negative() -> None:
    policy = _Stub()
    policy.load()
    obs = conformance.synthetic_observation(cameras=(CAMERA,))
    obs.route_xy = np.array([[-10.0, 0.0], [0.0, 0.0], [20.0, 15.0], [40.0, 40.0]])
    text, targets_sim = policy.route(obs)
    assert text == "Target waypoint: <TARGET_POINT><TARGET_POINT>."
    np.testing.assert_allclose(targets_sim, [[20.0, -15.0], [40.0, -40.0]])


@pytest.mark.parametrize(
    ("command", "text"),
    [
        (Command.LEFT, "Command: go left at the next intersection."),
        (Command.RIGHT, "Command: go right at the next intersection."),
        (Command.STRAIGHT, "Command: follow the road."),
        (Command.FOLLOW_REFERENCE, "Command: follow the road."),
    ],
)
def test_without_a_route_the_command_is_the_agents_sentence(command, text) -> None:
    policy = _Stub()
    policy.load()
    obs = conformance.synthetic_observation(cameras=("CAM_FRONT",))
    obs.command = command
    assert policy.route(obs) == (text, None)
    assert policy.diagnostics()["counts"]["command"] == 1


def test_a_known_distance_to_the_turn_is_said_as_the_agent_says_it() -> None:
    assert (
        inputs.command_sentence(Command.LEFT, 25.7)
        == "Command: go left at the next intersection in 25 meter."
    )
    assert inputs.command_sentence(Command.STRAIGHT, 25.7) == "Command: follow the road."


def test_history_frames_cost_no_forward_pass() -> None:
    policy = _Stub()
    policy.load()
    frames = [
        conformance.synthetic_observation(cameras=("CAM_FRONT",), timestamp_us=i * 500_000) for i in range(4)
    ]
    policy.infer_sequence(frames)
    assert policy.calls == 1


# -- the plan ---------------------------------------------------------------------------


def _native(lateral: float = 0.0) -> np.ndarray:
    return np.c_[np.arange(1, 11) * 2.0, np.full(10, lateral)]


def test_the_models_ten_points_are_kept_and_y_is_flipped() -> None:
    wps = _native(lateral=1.0)  # 1 m to SimLingo's right
    out = inputs.plan(wps, inputs.NATIVE_HORIZON_S)
    assert out.shape == (10, 2)
    np.testing.assert_allclose(out[:, 0], wps[:, 0])
    np.testing.assert_allclose(out[:, 1], -1.0)  # paddock: right is negative y


def test_the_plan_is_padded_with_its_last_position() -> None:
    wps = _native(lateral=0.5)
    out = inputs.plan(wps, 3.0)
    assert out.shape == (12, 2)
    np.testing.assert_allclose(out[:10], inputs.flip_y(wps))
    np.testing.assert_allclose(out[10:], [[20.0, -0.5], [20.0, -0.5]])


def test_a_horizon_shorter_than_the_models_or_off_its_grid_is_refused() -> None:
    wps = _native()
    with pytest.raises(ValueError):
        inputs.plan(wps, 2.0)
    with pytest.raises(ValueError):
        inputs.plan(wps, 3.1)


def test_the_default_plan_fills_the_closed_loop_wire() -> None:
    assert horizon_s() == 3.0
    wire = Trajectory(inputs.plan(_native(), horizon_s()), dt=inputs.PLAN_DT).to_wire()
    assert len(wire["trajectory"]) == N_WAYPOINTS
    np.testing.assert_allclose(wire["trajectory"][0], [4.0, 0.0])  # the model's own 0.5 s point
