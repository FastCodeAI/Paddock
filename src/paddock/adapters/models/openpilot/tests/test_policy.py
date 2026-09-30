"""Contract tests for the openpilot adapter.

Runs in paddock's own environment: no openpilot, no GPU, no dataset. It cannot
check that the model is good -- it checks that the adapter is wired correctly,
which is what a reviewer can actually verify on a PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.openpilot.policy import (
    DESIRE_LANE_CHANGE_LEFT,
    DESIRE_LANE_CHANGE_RIGHT,
    DESIRE_NONE,
    DESIRE_TURN_LEFT,
    DESIRE_TURN_RIGHT,
    DEVICE_FROM_EGO,
    GRID_S,
    POLICY,
    TARGET_H,
    TARGET_W,
    CameraAdapter,
    CameraFit,
    desire_index,
    desire_pulse,
    plan_to_grid,
)
from paddock.policy.protocol import Command, Pinhole, Policy


def test_policy_imports_without_openpilot_installed() -> None:
    """Module scope must stay free of openpilot imports.

    This module is imported by `paddock list`, by the registry, and by this test
    suite -- all of which run in paddock's environment, where openpilot's
    package does not exist. Every openpilot import belongs inside a method.
    """
    policy = POLICY()
    assert isinstance(policy, Policy)
    # comma-three rig for closed loop (road cam -> img, wide cam -> big_img) plus
    # CAM_FRONT, the single-camera fallback open-loop sources still rely on
    assert policy.cameras == ("CAM_OP_ROAD", "CAM_OP_WIDE", "CAM_FRONT")
    assert policy.waypoint_dt == 0.5


def test_diagnostics_before_any_frame() -> None:
    """Diagnostics must answer before the camera is established, not raise.

    Everything derived from a frame is necessarily absent, but the chosen
    backend is known from construction and is worth reporting -- a node that
    silently fell back from the tinygrad JIT to onnxruntime produces different
    numbers, and waiting for the first frame to say so is too late.
    """
    diag = POLICY().diagnostics()
    assert diag["status"] == "no frame seen yet"
    assert diag["backend"] in {"auto", "tinygrad", "onnx"}
    for frame_derived in ("rpy_deg", "margin_deg", "warp_in_bounds", "source_size"):
        assert frame_derived not in diag


def test_infer_without_load_is_a_clear_error() -> None:
    policy = POLICY()
    with pytest.raises(RuntimeError, match="load"):
        policy.infer(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "pinhole", "mode"),
    [
        # Wider than openpilot's 1.596 target: cropping would trim width, and
        # horizontal field of view is what the wide model view is short of.
        ("nuscenes render", Pinhole(1266, 1266, 800, 450, 1600, 900), "letterbox"),
        # Taller: trimming height costs no horizontal field of view at all.
        ("kitscenes ring", Pinhole(1841, 1841, 1740, 1139, 3504, 2272), "crop"),
    ],
)
def test_fit_strategy_follows_the_source_aspect(name, pinhole, mode) -> None:
    """Both loop types feed this one class, with sources of opposite aspect.

    Before paddock these were two classes, each hardcoding one strategy, which
    is why neither loop could run the other's data.
    """
    fit = CameraFit.compute(pinhole)
    assert fit.mode == mode, name
    assert fit.scaled_size[0] == TARGET_W
    assert fit.scaled_size[1] <= TARGET_H


def test_cropping_preserves_full_width() -> None:
    """The point of cropping rather than letterboxing a tall source."""
    fit = CameraFit.compute(Pinhole(1841, 1841, 1740, 1139, 3504, 2272))
    left, _, right, _ = fit.crop
    assert right - left == 3504, "cropping a tall source must not lose any width"


def test_letterboxing_preserves_full_width_too() -> None:
    fit = CameraFit.compute(Pinhole(1266, 1266, 800, 450, 1600, 900))
    left, _, right, _ = fit.crop
    assert right - left == 1600


@pytest.mark.parametrize(
    "pinhole",
    [
        Pinhole(1266, 1266, 800, 450, 1600, 900),
        Pinhole(1841, 1841, 1740, 1139, 3504, 2272),
    ],
)
def test_intrinsics_describe_the_actual_transform(pinhole) -> None:
    """The output intrinsics must match what `prepare` really does to the pixels.

    Not that the principal point lands at the output centre -- it does not, and
    should not. When the full width is kept (the crop branch) the principal
    point stays wherever the sensor put it, 6.6 px off centre for KITScenes.
    Forcing it to the middle would misdescribe the image and bend every
    projection that uses these intrinsics.
    """
    fit = CameraFit.compute(pinhole)
    x0, y0, _, _ = fit.crop
    assert fit.intrinsics[0, 2] == pytest.approx((pinhole.cx - x0) * fit.scale)
    assert fit.intrinsics[1, 2] == pytest.approx((pinhole.cy - y0) * fit.scale + fit.pad_top)
    assert fit.intrinsics[0, 0] == pytest.approx(pinhole.fx * fit.scale)


@pytest.mark.parametrize("aspect", [1.0, 1.33, 1.542, 1.596, 1.6, 1.778, 2.35])
def test_any_aspect_fits_the_compiled_input_size(aspect: float) -> None:
    """The branch invariant: neither strategy can overflow the target height.

    Crop sets crop_h = w/target_aspect, so the scaled height is exactly
    TARGET_H; letterbox is only chosen when the source is wider, which makes it
    strictly less. Checked across a range so a change to the branching shows up
    here rather than as a silently truncated image.
    """
    width = 2000
    height = int(width / aspect)
    fit = CameraFit.compute(Pinhole(1500, 1500, width / 2, height / 2, width, height))
    assert fit.scaled_size[1] <= TARGET_H
    assert fit.scaled_size[0] == TARGET_W


def test_plan_is_converted_to_y_left() -> None:
    """openpilot predicts y RIGHT; paddock and NeuroNCAP want y LEFT.

    Getting this wrong mirrors every trajectory and is invisible in a
    straight-line scenario -- it only shows up as a model that steers the wrong
    way around obstacles.
    """
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

    plan = np.zeros((len(ModelConstants.T_IDXS), 15))
    plan[:, Plan.POSITION][:, 0] = np.asarray(ModelConstants.T_IDXS) * 10.0
    plan[:, Plan.POSITION][:, 1] = 1.0  # 1 m to openpilot's right

    grid = plan_to_grid(plan)
    assert grid.shape == (len(GRID_S), 2)
    assert np.all(grid[:, 1] < 0), "y must be negated into the left-positive convention"


def test_the_grid_reaches_supercombos_own_horizon_without_extrapolating() -> None:
    """`plan_to_grid` interpolates T_IDXS; a grid past its end would invent plan."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.selfdrive.modeld.constants import ModelConstants

    assert GRID_S[-1] <= ModelConstants.T_IDXS[-1] + 1e-9, "the grid must not outrun the model"
    assert GRID_S[-1] == pytest.approx(ModelConstants.T_IDXS[-1]), "and should not stop short of it"


def test_extending_the_grid_left_the_shorter_benchmarks_alone() -> None:
    """Widening for comma2k19's 10 s must not move a recorded neuroNCAP,
    KITScenes or Bench2Drive score.

    The guarantee is a prefix one: the first eight entries are untouched, and
    every consumer reads a fixed prefix, truncates to its own ground truth, or
    resamples. Bench2Drive is the one that had no prefix -- its tracker averaged
    the whole plan -- so `pid.CONTROL_HORIZON_S` was added alongside this, and
    the second assertion is what keeps the pair honest.
    """
    from paddock.adapters.benchmarks.bench2drive.pid import CONTROL_HORIZON_S
    from paddock.policy.protocol import N_WAYPOINTS

    assert list(GRID_S[:8]) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert GRID_S[N_WAYPOINTS - 1] == 3.0, "neuroNCAP reads the first six"
    assert CONTROL_HORIZON_S <= GRID_S[7], "Bench2Drive must still see the plan it was tuned on"


def test_target_resolution_matches_the_compiled_warp() -> None:
    """ModelState compiles its warp for a fixed geometry; 1928x1208 is the tici one."""
    assert (TARGET_W, TARGET_H) == (1928, 1208)


def test_reset_clears_the_camera_adapter() -> None:
    """The camera is episode state, not model state.

    It is derived from the first frame's calibration. Surviving a reset means
    every subsequent scene is warped with the first scene's mounting rotation --
    which is exactly what happened, and cost a 40-minute run to find. KITScenes
    rigs are recalibrated between sessions (3 distinct front-camera rotations
    across 60 val scenes, up to 1 degree apart), so a stale adapter silently
    feeds a quarter of samples a wrong view.
    """
    policy = POLICY()
    policy.camera = object()  # stand-in for an adapter built from another scene
    policy.frames_seen = 12
    policy.model = object()  # so reset() does not need openpilot to be installed

    try:
        policy.reset()
    except ModuleNotFoundError:
        pytest.skip("reset() constructs a ModelState; needs openpilot's environment")

    assert policy.camera is None, "a stale camera adapter would warp the next scene wrongly"
    assert policy.frames_seen == 0


def test_calibration_changes_between_scenes_are_material() -> None:
    """Why the above matters: a degree of mounting rotation moves the warp.

    Not a large error, and nothing raises -- which is the point. It shifts what
    the model sees just enough to change its plan.
    """
    import numpy as np

    from paddock.adapters.models.openpilot.policy import DEVICE_FROM_EGO

    upright = np.eye(4)
    tilted = np.eye(4)
    angle = np.radians(1.0)
    tilted[:3, :3] = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    a = DEVICE_FROM_EGO @ upright[:3, :3]
    b = DEVICE_FROM_EGO @ tilted[:3, :3]
    assert not np.allclose(a, b), "a 1 degree change must not be silently equivalent"


def test_auxiliary_heads_are_converted_out_of_the_device_frame() -> None:
    """The sign conventions in `build_extras`, checked without openpilot.

    supercombo speaks [forward, right, down] and predicts its lane lines as (y,
    z) sampled at fixed distances rather than as (x, y) points. Both are easy to
    carry through unchanged, and neither shows up as an exception -- a mirrored
    lane line scores as a model that cannot see lanes, and a flipped yaw rate
    scores as a model that turns the wrong way. This is the cheapest place to
    catch either.
    """
    from paddock.adapters.models.openpilot.policy import _polyline, _to_ego

    # 1 m/s forward, 2 m/s to the device's right, 3 m/s down.
    assert _to_ego([1.0, 2.0, 3.0]) == [1.0, -2.0, -3.0]
    # Its own inverse, which is what lets one matrix serve both directions.
    assert _to_ego(_to_ego([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]

    # A line 1.85 m to the device's right is 1.85 m to ego's left of nothing --
    # it is the RIGHT lane line, at negative ego y.
    line = _polyline([[1.85, 0.0], [1.9, 0.0]], x_idxs=[0.0, 4.0])
    assert line == [[0.0, -1.85], [4.0, -1.9]]


def test_build_extras_matches_supercombos_own_output_layout() -> None:
    """Against the real `ModelConstants`, so a head's width cannot drift unseen."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.selfdrive.modeld.constants import ModelConstants as MC
    from openpilot.selfdrive.modeld.constants import Plan

    from paddock.adapters.models.openpilot.policy import build_extras
    from paddock.policy.protocol import validate_extras

    n = MC.IDX_N
    plan = np.zeros((n, MC.PLAN_WIDTH))
    plan[:, 0] = np.asarray(MC.T_IDXS) * 10.0
    plan[0, Plan.VELOCITY.start] = 9.5  # the plan's own forward speed at t = 0
    lead = np.zeros((1, MC.LEAD_MHP_SELECTION, MC.LEAD_TRAJ_LEN, MC.LEAD_WIDTH))
    lead[0, 0, 0] = [42.0, 1.5, 25.0, -0.2]  # x, y(right), absolute speed, accel
    out = {
        "plan": plan[None],
        "plan_stds": np.full((1, n, MC.PLAN_WIDTH), 0.25),
        "pose": np.array([[30.0, 0.5, 0.1, 0.0, 0.0, 0.02]]),
        "road_transform": np.array([[0.0, 0.0, 1.22, 0.0, 0.03, -0.004]]),
        "lead": lead,
        "lead_prob": np.array([[0.97, 0.5, 0.2]]),
        "lane_lines": np.zeros((1, MC.NUM_LANE_LINES, n, MC.LANE_LINES_WIDTH)),
        "lane_lines_prob": np.tile([0.1, 0.9], (1, MC.NUM_LANE_LINES)),
        "road_edges": np.zeros((1, MC.NUM_ROAD_EDGES, n, MC.LANE_LINES_WIDTH)),
        "desire_state": np.eye(MC.DESIRE_PRED_WIDTH)[None, DESIRE_TURN_LEFT],
    }

    extras = build_extras(out, plan, desire=DESIRE_TURN_LEFT)
    validate_extras(extras, n_waypoints=len(GRID_S))

    assert extras["pose"]["velocity"] == [30.0, -0.5, -0.1]
    assert extras["pose"]["rotation_rate"][2] == pytest.approx(-0.02), "yaw rate must be left-positive"
    # The camera is above the road, not below it.
    assert extras["road"]["camera_height"] == pytest.approx(1.22)
    assert len(extras["sigma"]) == len(GRID_S)

    (lead_out,) = extras["lead"]
    assert lead_out["d"] == pytest.approx(42.0), "camera-origin, as radard reads it"
    assert lead_out["y"] == pytest.approx(-1.5), "openpilot's right is paddock's negative y"
    assert lead_out["v_rel"] == pytest.approx(-5.0), "25 m/s lead, 30 m/s ego: closing"
    assert lead_out["prob"] == pytest.approx(0.97)

    assert extras["lane"]["line_probs"] == [pytest.approx(0.9)] * MC.NUM_LANE_LINES
    assert len(extras["lane"]["lines"]) == MC.NUM_LANE_LINES
    assert len(extras["lane"]["edges"]) == MC.NUM_ROAD_EDGES
    assert len(extras["lane"]["lines"][0]) == n

    assert extras["plan_v0"] == pytest.approx(9.5), "velocity x, not position, at the first T_IDX"
    assert extras["desire"]["input"] == DESIRE_TURN_LEFT
    assert len(extras["desire"]["state"]) == MC.DESIRE_PRED_WIDTH
    assert extras["desire"]["state"][DESIRE_TURN_LEFT] == 1.0


@pytest.mark.parametrize("fail", [False, True])
def test_cuda_constructor_restores_upstream_helpers(tmp_path, monkeypatch, fail):
    import sys
    from types import ModuleType

    names = [
        "openpilot",
        "openpilot.selfdrive",
        "openpilot.selfdrive.modeld",
        "openpilot.selfdrive.modeld.modeld",
    ]
    for name in names:
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    modeld = sys.modules[names[-1]]
    artifact = tmp_path / "driving_cuda.pkl"
    artifact.touch()
    monkeypatch.setenv("PADDOCK_OPENPILOT_CUDA_MODEL", str(artifact))

    def original_path():
        return None

    def original_devices():
        return {}

    modeld.modeld_pkl_path = original_path
    modeld.get_tg_input_devices = original_devices
    sentinel = object()

    def construct(*args, **kwargs):
        assert modeld.modeld_pkl_path() == artifact
        assert modeld.get_tg_input_devices() == {"WARP_DEV": "CPU", "QUEUE_DEV": "CUDA"}
        if fail:
            raise RuntimeError("construction failed")
        return sentinel

    modeld.ModelState = construct
    if fail:
        with pytest.raises(RuntimeError, match="construction failed"):
            POLICY()._new_model()
    else:
        assert POLICY()._new_model() is sentinel
    assert modeld.modeld_pkl_path is original_path
    assert modeld.get_tg_input_devices is original_devices


# --------------------------------------------------------------------------
# Two cameras reach two views
# --------------------------------------------------------------------------


class _FakeCameraAdapter:
    """`CameraAdapter` without openpilot, keeping only what inference reads.

    The real one imports `openpilot.common.transformations` in its constructor,
    which paddock's environment does not have. Everything downstream of it here
    -- `WarpPlan`, `rgb_to_yuv420` -- is numpy and runs for real, so this stands
    in for the one part that cannot and leaves the wiring under test.
    """

    def __init__(self, pinhole, camera2ego, resample: bool = True, height_scale: float = 1.0) -> None:
        self.src = pinhole
        self.resample = resample
        self.height_scale = height_scale
        self.size = (pinhole.width, pinhole.height)
        self.transforms = {"img": np.eye(3, dtype=np.float32), "big_img": np.eye(3, dtype=np.float32)}

    def prepare(self, image):
        return image


def _two_camera_observation():
    """A comma rig: distinguishable road and wide frames, and nothing else."""
    from paddock.policy.protocol import Command, Observation

    def frame(fill):
        return np.full((TARGET_H, TARGET_W, 3), fill, dtype=np.uint8)

    k = np.array([[900.0, 0.0, TARGET_W / 2], [0.0, 900.0, TARGET_H / 2], [0.0, 0.0, 1.0]])
    mount = np.eye(4)
    mount[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    return Observation(
        images={"CAM_OP_ROAD": frame(40), "CAM_OP_WIDE": frame(200)},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={"CAM_OP_ROAD": k, "CAM_OP_WIDE": k},
        camera2ego={"CAM_OP_ROAD": mount, "CAM_OP_WIDE": mount},
    )


def _onnx_policy(monkeypatch):
    """A policy on the ONNX backend, wired to record what the graph is handed."""
    import paddock.adapters.models.openpilot.policy as P

    monkeypatch.setattr(P, "CameraAdapter", _FakeCameraAdapter)
    policy = POLICY()
    policy.backend = "onnx"
    policy.self_prime = False
    shapes = {"img": (1, 12, 128, 256), "desire_pulse": (1, 25, 8)}
    policy.session = type("S", (), {"spec": type("Spec", (), {"input_shapes": shapes})})()
    seen = {}

    def run(img, big, speed, desire=None):
        seen["img"], seen["big_img"], seen["desire"] = img, big, desire
        return {}

    policy.model = type("M", (), {"run": staticmethod(run)})()
    return policy, seen


def test_the_onnx_path_feeds_the_wide_camera_to_the_wide_view(monkeypatch) -> None:
    """The defect that made a backend choice a scoring difference.

    `_resolve_backend` picks ONNX by *trying an import*, and `setup.sh` makes a
    failed `scons` non-fatal, so a site that never built tinygrad got the ONNX
    path silently. That path had no two-camera branch: it passed the road frame
    as both `img` and `big_img`, so on comma2k19 and Bench2Drive openpilot was
    scored with the narrow camera warped into the wide field, and the wide
    camera it declares was never read at all.
    """
    policy, seen = _onnx_policy(monkeypatch)
    policy.infer(_two_camera_observation())

    assert policy.cam_wide is not None, "the rig has two cameras and only one was bound"
    assert not np.array_equal(seen["img"], seen["big_img"])
    # Not just different: `big_img` has to be the *wide* frame. The two source
    # fills differ, and the warp is the identity here, so the fill survives.
    assert seen["big_img"].max() > seen["img"].max()


def _one_camera_observation():
    """Every other rig: one front camera."""
    obs = _two_camera_observation()
    obs.images = {"CAM_FRONT": obs.images["CAM_OP_ROAD"]}
    obs.camera2image = {"CAM_FRONT": obs.camera2image["CAM_OP_ROAD"]}
    obs.camera2ego = {"CAM_FRONT": obs.camera2ego["CAM_OP_ROAD"]}
    return obs


def test_one_camera_still_feeds_both_views(monkeypatch) -> None:
    """Every dataset here but the comma rigs has one camera; it fills both."""
    policy, seen = _onnx_policy(monkeypatch)
    obs = _one_camera_observation()

    policy.infer(obs)

    assert policy.cam_wide is None
    assert np.array_equal(seen["img"], seen["big_img"])


def test_the_two_backends_bind_the_same_cameras(monkeypatch) -> None:
    """They disagreed for a year, and nothing on the record showed it.

    `_bind_cameras` is shared so that "which image reaches big_img" is one
    answer. Only `resample` may differ, because only the warp does: the ONNX
    warp is built per source geometry and the tinygrad one is compiled for a
    fixed size.
    """
    import paddock.adapters.models.openpilot.policy as P

    monkeypatch.setattr(P, "CameraAdapter", _FakeCameraAdapter)
    obs = _two_camera_observation()
    bound = {}
    for backend in ("onnx", "tinygrad"):
        policy = POLICY()
        policy.backend = backend
        assert policy._bind_cameras(obs) is True
        road, wide, road_key, wide_key = policy._views(True)
        bound[backend] = (road_key, wide_key, road.resample, wide.resample)

    assert bound["onnx"][:2] == bound["tinygrad"][:2] == ("CAM_OP_ROAD", "CAM_OP_WIDE")
    assert bound["onnx"][2:] == (False, False)
    assert bound["tinygrad"][2:] == (True, True)


def _counting_policy(monkeypatch):
    """An ONNX policy that records one entry per model tick."""
    policy, _ = _onnx_policy(monkeypatch)
    ticks: list[float] = []
    policy.model = type(
        "M", (), {"run": staticmethod(lambda img, big, speed, desire=None: ticks.append(speed) or {})}
    )()
    # `reset` would rebuild the runner from the stub session, which has no
    # graph behind it. Everything else `reset` clears is what is under test.
    monkeypatch.setattr(policy, "_new_onnx_runner", lambda: None)
    return policy, ticks


def _sequence(hz: float, n: int):
    out = []
    for i in range(n):
        obs = _two_camera_observation()
        obs.timestamp_us = int(round(i * 1e6 / hz))
        out.append(obs)
    return out


def test_the_first_frame_of_a_sequence_is_repeated_like_every_other(monkeypatch) -> None:
    """supercombo runs at 20 Hz; a 10 Hz benchmark must show each frame twice.

    `_repeats` measures against the previous observation, so the first frame of
    a sequence had nothing to measure and fell back to one tick -- the history
    was a tick short of what the sequence paid for, and the motion across the
    opening pair read as half of what it was. Every recorded 10 Hz run here
    (KITScenes, nuScenes, NAVSIM) went through this path.
    """
    policy, ticks = _counting_policy(monkeypatch)
    policy.infer_sequence(_sequence(hz=10.0, n=4))
    assert len(ticks) == 8, "two ticks per frame at 10 Hz, the first frame included"


def test_a_native_rate_sequence_is_not_repeated_at_all(monkeypatch) -> None:
    """comma2k19 is recorded at the model's own rate: one tick per frame."""
    policy, ticks = _counting_policy(monkeypatch)
    policy.infer_sequence(_sequence(hz=20.0, n=5))
    assert len(ticks) == 5


def test_one_dropped_frame_does_not_redefine_the_rate(monkeypatch) -> None:
    """The seed is the median step, so a gap in the middle stays a local gap."""
    policy, ticks = _counting_policy(monkeypatch)
    frames = _sequence(hz=20.0, n=6)
    for obs in frames[3:]:
        obs.timestamp_us += 50_000  # one frame missing partway through
    policy.infer_sequence(frames)
    # Five frames at the native rate, plus the doubled step across the gap.
    assert len(ticks) == 7


def test_closed_loop_still_primes_its_own_history(monkeypatch) -> None:
    """There is no sequence to read a rate from, so the first frame self-primes.

    The seed is scoped to `infer_sequence`; a simulator handing over one frame
    at a time must keep getting the manufactured warm-up instead.
    """
    from paddock.adapters.models.openpilot.policy import FEATURE_BUFFER_FRAMES

    policy, ticks = _counting_policy(monkeypatch)
    policy.self_prime = True
    policy.infer(_sequence(hz=10.0, n=1)[0])
    assert len(ticks) == FEATURE_BUFFER_FRAMES
    assert policy._seed_dt_s is None


# --------------------------------------------------------------------------
# A route command reaches the desire input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "command", "slot"),
    [
        ("off", Command.LEFT, DESIRE_NONE),
        ("off", Command.RIGHT, DESIRE_NONE),
        ("turn", Command.LEFT, DESIRE_TURN_LEFT),
        ("turn", Command.RIGHT, DESIRE_TURN_RIGHT),
        ("turn", Command.STRAIGHT, DESIRE_NONE),
        ("turn", Command.FOLLOW_REFERENCE, DESIRE_NONE),
        ("lane_change", Command.LEFT, DESIRE_LANE_CHANGE_LEFT),
        ("lane_change", Command.RIGHT, DESIRE_LANE_CHANGE_RIGHT),
        ("lane_change", Command.STRAIGHT, DESIRE_NONE),
    ],
)
def test_a_route_command_is_sent_as_the_desire_its_mode_names(mode, command, slot) -> None:
    """An explicit table, because the two enums do not line up.

    `Command.RIGHT` is 0 and desire 0 is `none`, so passing a command through
    as a slot would turn every right turn into no desire and every left into a
    left turn -- a plausible-looking asymmetry nobody would think to question.
    """
    assert desire_index(command, mode) == slot
    # What arrives off the wire is the enum's value; the table must take it.
    assert desire_index(int(command), mode) == slot

    pulse = desire_pulse(command, mode, width=8)
    assert pulse.dtype == np.float32
    assert np.flatnonzero(pulse).tolist() == ([] if slot == DESIRE_NONE else [slot])


@pytest.mark.parametrize(("mode", "sent"), [("off", []), ("turn", [DESIRE_TURN_LEFT])])
def test_the_desire_reaches_the_graph(monkeypatch, mode, sent) -> None:
    """Held at the frame's level; `SupercomboRunner.run` makes the rising edge.

    "off" must hand over the all-zero vector the runner substitutes for
    `desire=None`, which is what every recorded run was fed.
    """
    policy, seen = _onnx_policy(monkeypatch)
    policy.desire_mode = mode
    obs = _two_camera_observation()
    obs.command = Command.LEFT

    policy.infer(obs)

    assert seen["desire"].shape == (8,), "as wide as the graph's desire_pulse input"
    assert np.flatnonzero(seen["desire"]).tolist() == sent


def test_the_desire_mode_is_read_from_the_env(monkeypatch) -> None:
    monkeypatch.delenv("PADDOCK_OPENPILOT_DESIRE", raising=False)
    assert POLICY().desire_mode == "off"
    monkeypatch.setenv("PADDOCK_OPENPILOT_DESIRE", " Lane_Change ")
    assert POLICY().desire_mode == "lane_change"
    assert POLICY().diagnostics()["desire_mode"] == "lane_change", "and it is on the run record"


def test_an_unknown_desire_mode_stops_the_node(monkeypatch) -> None:
    """At construction: unchecked, the same typo fails every sample's first frame."""
    monkeypatch.setenv("PADDOCK_OPENPILOT_DESIRE", "turns")
    with pytest.raises(ValueError, match="'turns': expected one of off, turn, lane_change"):
        POLICY()


def test_desire_slots_are_openpilots_own() -> None:
    """Restated at module scope, so pinned here to cereal's enum itself."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.cereal import log
    from openpilot.selfdrive.modeld.constants import ModelConstants

    assert DESIRE_NONE == log.Desire.none
    assert (DESIRE_TURN_LEFT, DESIRE_TURN_RIGHT) == (log.Desire.turnLeft, log.Desire.turnRight)
    assert (DESIRE_LANE_CHANGE_LEFT, DESIRE_LANE_CHANGE_RIGHT) == (
        log.Desire.laneChangeLeft,
        log.Desire.laneChangeRight,
    )
    assert ModelConstants.DESIRE_LEN == 8, "the width the ONNX stub above declares"


# --------------------------------------------------------------------------
# A camera's height, shown to the model as openpilot's
# --------------------------------------------------------------------------

#: KITScenes' ring_front and a nuScenes camera: 63 and 39 degrees of vertical field.
KITSCENES_RING = Pinhole(1841, 1841, 1740, 1139, 3504, 2272)
NUSCENES_FRONT = Pinhole(1266, 1266, 800, 450, 1600, 900)


def _mount(pitch_deg: float = 0.0) -> np.ndarray:
    """camera2ego for a camera facing forward, pitched down by `pitch_deg`."""
    p = np.radians(pitch_deg)
    nose_down = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    mount = np.eye(4)
    mount[:3, :3] = nose_down @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    return mount


def _pixel(m, ray) -> np.ndarray:
    p = np.asarray(m, dtype=np.float64) @ np.asarray(ray, dtype=np.float64)
    return p[:2] / p[2]


def test_the_height_scale_is_read_from_the_env(monkeypatch) -> None:
    monkeypatch.delenv("PADDOCK_OPENPILOT_HEIGHT_SCALE", raising=False)
    assert POLICY().height_scale == 1.0
    monkeypatch.setenv("PADDOCK_OPENPILOT_HEIGHT_SCALE", " 1.53 ")
    assert POLICY().height_scale == 1.53
    assert POLICY().diagnostics()["height_scale"] == 1.53, "and it is on the run record"
    assert POLICY(height_scale=1.3).height_scale == 1.3, "an argument wins over the env"


@pytest.mark.parametrize("raw", ["0", "-1.53", "nan", "inf", "1.87 m"])
def test_a_height_scale_that_is_not_a_positive_number_stops_the_node(monkeypatch, raw) -> None:
    """Zero puts every row on the horizon, and a negative scale puts sky on the road."""
    monkeypatch.setenv("PADDOCK_OPENPILOT_HEIGHT_SCALE", raw)
    with pytest.raises(ValueError, match="expected a positive number"):
        POLICY()


@pytest.mark.parametrize("two", [True, False], ids=["two cameras", "one camera"])
def test_every_camera_is_warped_at_the_height_scale(monkeypatch, two) -> None:
    import paddock.adapters.models.openpilot.policy as P

    monkeypatch.setattr(P, "CameraAdapter", _FakeCameraAdapter)
    policy = POLICY(height_scale=1.3)
    assert policy._bind_cameras(_two_camera_observation() if two else _one_camera_observation()) is two
    road, wide, _, _ = policy._views(two)
    assert road.height_scale == wide.height_scale == 1.3


@pytest.mark.parametrize("resample", [False, True], ids=["onnx", "tinygrad"])
def test_a_unit_height_scale_is_openpilots_own_warp(resample) -> None:
    """Bit for bit: it is the warp every recorded run was made with."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.common.transformations.model import get_warp_matrix

    for scale in ({}, {"height_scale": 1.0}):
        cam = CameraAdapter(KITSCENES_RING, _mount(3.0), resample=resample, **scale)
        for slot, big in (("img", False), ("big_img", True)):
            own = get_warp_matrix(cam.rpy, cam.intrinsics, big).astype(np.float32)
            assert np.array_equal(cam.transforms[slot], own)
        assert cam.view_fov == CameraAdapter.VIEW_FOV


def test_a_height_scale_shows_the_road_as_openpilots_camera_would() -> None:
    """A road point lands on the model pixel openpilot's 1.22 m camera saw it on.

    Where this camera sees the point is projected straight from camera2ego, not
    through the adapter's rpy, so the test does not share the code it checks;
    and the camera is pitched, so a scale on the wrong side of the rotation
    fails it.
    """
    pytest.importorskip("openpilot", reason="needs openpilot's environment")
    from openpilot.common.transformations.model import calib_from_medmodel, calib_from_sbigmodel

    height, mount = 1.87, _mount(3.0)
    scaled = CameraAdapter(KITSCENES_RING, mount, resample=False, height_scale=height / 1.22)
    plain = CameraAdapter(KITSCENES_RING, mount, resample=False)
    camera_from_ego = scaled.intrinsics @ mount[:3, :3].T
    for slot, calib_from_model in (("img", calib_from_medmodel), ("big_img", calib_from_sbigmodel)):
        model_from_calib = np.linalg.inv(calib_from_model)
        for x in (6.0, 10.0, 20.0, 40.0, 80.0):
            for y in (-3.5, 0.0, 3.5):  # the calibrated frame is y right, z down
                seen = _pixel(camera_from_ego, DEVICE_FROM_EGO @ [x, y, height])
                shown = np.append(_pixel(model_from_calib, [x, y, 1.22]), 1.0)
                assert np.allclose(_pixel(scaled.transforms[slot], shown), seen, atol=0.05)
                off = np.abs(_pixel(plain.transforms[slot], shown) - seen).max()
                assert off > 10.0, "the plain warp shows openpilot a nearer road"
            # The horizon row does not move.
            horizon = np.append(_pixel(model_from_calib, [x, x / 10.0, 0.0]), 1.0)
            assert np.allclose(
                _pixel(scaled.transforms[slot], horizon), _pixel(plain.transforms[slot], horizon), atol=0.05
            )


@pytest.mark.parametrize("resample", [False, True], ids=["onnx", "tinygrad letterbox"])
def test_a_height_scale_that_takes_a_view_off_the_sensor_is_refused(resample) -> None:
    """A nuScenes camera has 39 degrees of vertical field; big_img at s = 1.53 wants 46."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")

    plain = CameraAdapter(NUSCENES_FRONT, _mount(), resample=resample)
    assert plain._real_pixels(plain.transforms["big_img"], (512, 256)) == 1.0, "on the sensor unscaled"
    if resample:
        # Letterboxed: a view laid over the frame's top rows is partly padding.
        on_top_rows = plain._real_pixels(np.eye(3), (512, 256))
        assert plain.pad_top > 0 and on_top_rows == pytest.approx(1 - plain.pad_top / 256)
    with pytest.raises(ValueError, match="takes the big_img view off this 1600x900 camera"):
        CameraAdapter(NUSCENES_FRONT, _mount(), resample=resample, height_scale=1.53)
    CameraAdapter(KITSCENES_RING, _mount(3.0), resample=resample, height_scale=1.53)  # 63 degrees has room


def test_the_margin_is_the_scaled_views() -> None:
    """Checked against the rows the warp itself puts the view's top and bottom on."""
    pytest.importorskip("openpilot", reason="needs openpilot's environment")

    for s in (0.8, 1.0, 1.53):
        cam = CameraAdapter(KITSCENES_RING, _mount(), resample=False, height_scale=s)
        for slot in ("img", "big_img"):
            rows = [_pixel(cam.transforms[slot], [256.0, v, 1.0])[1] for v in (0.0, 256.0)]
            top, bottom = (np.degrees(np.arctan((r - KITSCENES_RING.cy) / KITSCENES_RING.fy)) for r in rows)
            vfov = KITSCENES_RING.vfov_deg - cam.margin_deg(slot)["vertical"]
            # VIEW_FOV, the unscaled figure, is rounded to 0.1 degree.
            assert vfov == pytest.approx(bottom - top, abs=0.06 if s == 1.0 else 1e-3)
