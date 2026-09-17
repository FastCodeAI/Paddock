"""Contract tests for the openpilot adapter.

Runs in paddock's own environment: no openpilot, no GPU, no dataset. It cannot
check that the model is good -- it checks that the adapter is wired correctly,
which is what a reviewer can actually verify on a PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.openpilot.policy import (
    GRID_S,
    POLICY,
    TARGET_H,
    TARGET_W,
    CameraFit,
    plan_to_grid,
)
from paddock.policy.protocol import Pinhole, Policy


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

    from paddock.adapters.models.openpilot.policy import build_extras
    from paddock.policy.protocol import validate_extras

    n = MC.IDX_N
    plan = np.zeros((n, MC.PLAN_WIDTH))
    plan[:, 0] = np.asarray(MC.T_IDXS) * 10.0
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
    }

    extras = build_extras(out, plan)
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

    def __init__(self, pinhole, camera2ego, resample: bool = True) -> None:
        self.src = pinhole
        self.resample = resample
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
    policy.session = type("S", (), {"spec": type("Spec", (), {"input_shapes": {"img": (1, 12, 128, 256)}})})()
    seen = {}

    def run(img, big, speed):
        seen["img"], seen["big_img"] = img, big
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


def test_one_camera_still_feeds_both_views(monkeypatch) -> None:
    """Every dataset here but the comma rigs has one camera; it fills both."""
    policy, seen = _onnx_policy(monkeypatch)
    obs = _two_camera_observation()
    obs.images = {"CAM_FRONT": obs.images["CAM_OP_ROAD"]}
    obs.camera2image = {"CAM_FRONT": obs.camera2image["CAM_OP_ROAD"]}
    obs.camera2ego = {"CAM_FRONT": obs.camera2ego["CAM_OP_ROAD"]}

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
