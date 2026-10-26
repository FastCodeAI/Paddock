"""Contract tests for the AutoE2E adapter: the input packing and the control
rollout, which are the only parts not imported from upstream. No torch, no
checkpoint, no GPU. Conventions are upstream's at 40a75cf8:
navigation/geometry.py (raster rows front to rear, columns left to right),
training/losses/control_rollout.py (speed clamped first, heading from the new
speed), kit_scenes/egomotion.py (gradients on the real timestamps)."""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.models.autoe2e.policy import (
    EGO_STEPS,
    HISTORY_OFFSETS_S,
    MATCH_TOLERANCE_S,
    NAV_H,
    NAV_MPP,
    NAV_W,
    NAV_X_MAX,
    NAV_Y_MAX,
    OUT_STEPS,
    POLICY,
    VIEW_ORDER,
    _Frame,
    _Navigation,
    blank_views,
    egomotion_rows,
    finite_controls,
    integrate_controls,
    match_tolerance,
    navigation_modules,
    pack_tile,
    route_raster,
    steady_before_start,
)
from paddock.policy.protocol import Command, Observation, Policy

#: Camera optical (x right, y down, z forward) -> ego FLU, for a camera facing forward.
OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _pixel(x: float, y: float) -> tuple[int, int]:
    """Upstream's ego_to_pixel, rounded to the pixel whose centre is nearest."""
    return round((NAV_X_MAX - x) / NAV_MPP - 0.5), round((NAV_Y_MAX - y) / NAV_MPP - 0.5)


def test_policy_imports_without_torch_and_asks_for_six_views() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert tuple(policy.cameras) == VIEW_ORDER and len(VIEW_ORDER) == 6
    assert policy.waypoint_dt == 0.5


def test_load_names_the_missing_setting(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AUTOE2E_CHECKPOINT", raising=False)
    with pytest.raises(RuntimeError, match="AUTOE2E_CHECKPOINT"):
        POLICY().load()
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("AUTOE2E_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("AUTOE2E_REPO", str(tmp_path))
    with pytest.raises(RuntimeError, match="AUTOE2E_REPO"):
        POLICY().load()


def test_diagnostics_name_the_release_checkpoint_not_where_this_site_keeps_it() -> None:
    """`/diagnostics` is copied verbatim into the tracked run.json: no weights root may reach it."""
    policy = POLICY()
    policy.checkpoint = "/path/to/weights/auto_e2e/models/kitscenes-epoch-5/checkpoint.pt"
    assert policy.diagnostics()["checkpoint"] == "models/kitscenes-epoch-5/checkpoint.pt"
    policy.checkpoint = "/path/to/finetunes/run7.pt"
    assert policy.diagnostics()["checkpoint"] == "run7.pt"


# -- ego history -------------------------------------------------------------


def test_egomotion_on_a_circle_reads_speed_yaw_rate_and_curvature() -> None:
    v, w = 8.0, 0.2  # m/s, rad/s: a 40 m radius
    t = np.arange(0.0, 8.0 + 1e-9, 0.1)
    yaw = w * t
    xy = np.column_stack([np.sin(yaw), 1.0 - np.cos(yaw)]) * (v / w)
    rows = egomotion_rows(t, xy, yaw, t_now=8.0)
    assert rows.shape == (EGO_STEPS, 4)
    assert rows[:, 0] == pytest.approx(v, rel=1e-3)
    assert rows[:, 1] == pytest.approx(0.0, abs=1e-2)
    assert rows[:, 2] == pytest.approx(w, rel=1e-3)
    assert rows[:, 3] == pytest.approx(w / v, rel=1e-3)


def test_rows_older_than_the_first_pose_stay_zero() -> None:
    """Upstream pads left with zeros; a route start must not invent a past."""
    t = np.arange(0.0, 2.0 + 1e-9, 0.1)
    xy = np.column_stack([5.0 * t, np.zeros_like(t)])
    rows = egomotion_rows(t, xy, np.zeros_like(t), t_now=2.0)
    held = (np.abs(rows).sum(1) > 0).sum()
    assert held == 20, "t-2.0 .. t-0.1 s are inside the 2 s held"
    assert np.all(rows[:-20] == 0) and rows[-20:, 0] == pytest.approx(5.0)


def test_closed_loop_fills_the_past_with_steady_driving() -> None:
    """2 s into an episode at 5 m/s: the 44 older rows read 5 m/s, no acceleration, no turn."""
    t = np.arange(0.0, 2.0 + 1e-9, 0.1)
    xy = np.column_stack([5.0 * t, np.zeros_like(t)])
    rows = egomotion_rows(t, xy, np.zeros_like(t), t_now=2.0)
    filled, n = steady_before_start(rows, t, t_now=2.0, speed=9.9)
    assert n == 44
    assert filled[:44] == pytest.approx(np.tile([5.0, 0.0, 0.0, 0.0], (44, 1)))
    assert np.array_equal(filled[44:], rows[44:]), "the measured rows are left as they are"


def test_the_first_call_takes_the_simulators_speed() -> None:
    """One pose measures nothing; a zero speed would plan a stop for a car doing 8 m/s."""
    filled, n = steady_before_start(np.zeros((EGO_STEPS, 4)), np.array([0.0]), t_now=0.0, speed=8.3)
    assert n == EGO_STEPS
    assert filled == pytest.approx(np.tile([8.3, 0.0, 0.0, 0.0], (EGO_STEPS, 1)))


def test_a_full_past_is_not_touched() -> None:
    t = np.arange(0.0, 7.0, 0.1)
    rows = egomotion_rows(t, np.column_stack([3.0 * t, np.zeros_like(t)]), np.zeros_like(t), t_now=6.9)
    filled, n = steady_before_start(rows, t, t_now=6.9, speed=0.0)
    assert n == 0 and np.array_equal(filled, rows)


def test_only_closed_loop_fills_the_past(monkeypatch) -> None:
    """The fill is gated on `self_prime`, which `infer_sequence` (every open-loop benchmark) clears."""
    seen = []
    monkeypatch.setattr(POLICY, "observe", lambda self, obs: None)
    monkeypatch.setattr(POLICY, "infer", lambda self, obs: seen.append(self.self_prime))
    policy = POLICY()
    policy.infer(object())
    policy.infer_sequence([object(), object()])
    assert seen == [True, False]
    assert policy.self_prime is True


def test_the_adapter_fills_the_past_only_in_closed_loop() -> None:
    """The history `infer()` builds: filled with `self_prime` on, zero-padded with it off."""
    frames = []
    for t in np.arange(0.0, 2.0 + 1e-9, 0.1):
        m = np.eye(4)
        m[0, 3] = 5.0 * t
        frames.append(_Frame(t_s=float(t), ego2world=m, camera2image={}, camera2ego={}))
    policy = POLICY()
    policy.frames = frames
    closed, n_closed = policy._ego_history(frames[-1], speed=9.9)
    policy.self_prime = False
    opened, n_open = policy._ego_history(frames[-1], speed=9.9)
    assert n_closed == 44 and closed[:44] == pytest.approx(np.tile([5.0, 0.0, 0.0, 0.0], (44, 1)))
    assert n_open == 0 and np.all(opened[:44] == 0)
    assert np.array_equal(opened[44:], closed[44:]), "the measured rows are the same either way"


def test_curvature_divides_by_the_speed_floor_when_stopped() -> None:
    t = np.arange(0.0, 7.0, 0.1)
    rows = egomotion_rows(t, np.zeros((len(t), 2)), 0.05 * t, t_now=7.0)
    assert rows[:, 3] == pytest.approx(0.05 / 0.1)


# -- control rollout -----------------------------------------------------------


def test_zero_controls_hold_speed_in_a_straight_line() -> None:
    out = integrate_controls(np.zeros((OUT_STEPS, 2)), v0=10.0)
    assert out.shape == (OUT_STEPS, 3)
    assert out[:, 0] == pytest.approx(10.0 * 0.1 * np.arange(1, OUT_STEPS + 1))
    assert np.all(out[:, 1:] == 0)


def test_braking_stops_and_never_reverses() -> None:
    controls = np.zeros((OUT_STEPS, 2))
    controls[:, 0] = -5.0
    out = integrate_controls(controls, v0=4.0)
    assert out[7, 0] == pytest.approx(out[-1, 0]), "stopped after 0.8 s"
    assert np.all(np.diff(out[:, 0]) >= 0)


def test_non_finite_controls_fail_the_call_instead_of_becoming_a_plan() -> None:
    """A crash must not become a score: NaN or inf from the model raises, so the scenario is
    counted as failed, never scored on a substitute plan under AutoE2E's name."""
    controls = np.zeros((OUT_STEPS, 2))
    assert finite_controls(controls) is controls
    for bad in (np.nan, np.inf):
        broken = controls.copy()
        broken[10, 1] = bad
        with pytest.raises(RuntimeError, match="non-finite controls at 1 of 64 steps"):
            finite_controls(broken)


def test_positive_curvature_turns_left() -> None:
    controls = np.zeros((OUT_STEPS, 2))
    controls[:, 1] = 0.05
    out = integrate_controls(controls, v0=10.0)
    assert out[-1, 2] == pytest.approx(10.0 * 0.05 * 6.4)
    assert out[-1, 1] > 0, "paddock y is left"


# -- navigation raster -----------------------------------------------------------


def test_the_route_corridor_lands_where_upstreams_geometry_puts_it() -> None:
    route = np.column_stack([np.arange(-50.0, 101.0), np.full(151, 3.0)])  # a lane 3 m to the left
    raster, corridor, destination = route_raster(route)
    assert raster.shape == (2, NAV_H, NAV_W) and corridor
    assert raster[0][_pixel(50.0, 3.0)] == 1
    assert raster[0][_pixel(50.0, -3.0)] == 0, "the mirror image must be empty"
    assert raster[0][_pixel(50.0, 3.0 + 2.5)] == 0, "1.75 m either side, no wider"
    assert raster[0][_pixel(-5.0, 3.0)] == 1 and raster[0][_pixel(-20.0, 3.0)] == 0, "rear-clipped at -10 m"
    assert destination, "a route that stops 100 m ahead ends there"
    assert raster[1][_pixel(100.0, 3.0)] == 1


def test_a_route_running_past_the_look_ahead_has_no_destination() -> None:
    route = np.column_stack([np.arange(-10.0, 201.0), np.zeros(211)])
    raster, corridor, destination = route_raster(route)
    assert corridor and not destination and raster[1].sum() == 0


def test_no_route_is_upstreams_camera_only_track() -> None:
    raster, corridor, destination = route_raster(None)
    assert not corridor and not destination and raster.sum() == 0


# -- camera history and projection ------------------------------------------------


def _pose(x: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 3] = x
    return m


def _frame(t_s: float, x: float, views=VIEW_ORDER, size=(8, 8)) -> _Frame:
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = OPTICAL
    cam2ego[:3, 3] = (1.5, 0.0, 1.5)
    k = np.array([[400.0, 0.0, 400.0], [0.0, 400.0, 225.0], [0.0, 0.0, 1.0]])
    return _Frame(
        t_s=t_s,
        ego2world=_pose(x),
        camera2image={c: k for c in views},
        camera2ego={c: cam2ego for c in views},
        images={c: np.zeros((size[1], size[0], 3), dtype=np.uint8) for c in views},
    )


def test_history_is_seven_six_view_frames_half_a_second_apart() -> None:
    """NAVSIM's 10 Hz frames between keyframes carry three views; only six-view frames count."""
    policy = POLICY()
    for k in range(51):  # 5 s at 10 Hz, six views at every 0.5 s only
        policy.frames.append(_frame(k / 10, 0.0, VIEW_ORDER if k % 5 == 0 else VIEW_ORDER[:3]))
    picked, gaps = policy._history(policy.frames[-1])
    assert [f.t_s for f in picked] == pytest.approx([5.0 + off for off in HISTORY_OFFSETS_S])
    assert gaps == [0.0] * 7


def test_a_short_past_repeats_the_oldest_frame_and_says_so() -> None:
    policy = POLICY()
    for k in range(21):  # 2 s held
        policy.frames.append(_frame(k / 10, 0.0))
    picked, gaps = policy._history(policy.frames[-1])
    assert gaps[:3] == [None, None, None] and gaps[3:] == [0.0] * 4
    assert all(f is policy.frames[0] for f in picked[:3])


def test_jittered_keyframes_are_taken_for_the_slots_they_fill() -> None:
    """nuScenes keyframes drift up to ~0.1 s off the 0.5 s grid; each is still the frame meant."""
    policy = POLICY()
    times = [0.0, 0.55, 1.08, 1.62, 2.10, 2.58, 3.07, 3.58]  # gaps 0.47-0.55 s, like nuScenes
    for t in times:
        policy.frames.append(_frame(t, 0.0))
    picked, gaps = policy._history(policy.frames[-1])
    assert [f.t_s for f in picked] == times[:-1], "every real keyframe, none repeated"
    assert all(g is not None for g in gaps)
    assert max(abs(g) for g in gaps) > MATCH_TOLERANCE_S, "the old 0.06 s match refused some of these"


def test_a_10_hz_stream_keeps_the_old_match_even_with_a_dropped_frame() -> None:
    """What KITScenes and neuroNCAP were fed is unchanged: at 10 Hz the tolerance stays 0.06 s,
    so a slot whose frame is missing still repeats the oldest frame, as it did."""
    policy = POLICY()
    for k in range(41):
        if k == 25:  # t = 2.5 s, the -1.5 s slot of the frame at 4.0 s, is missing
            continue
        policy.frames.append(_frame(k / 10, 0.0))
    assert match_tolerance(np.array([f.t_s for f in policy.frames[:-1]])) == MATCH_TOLERANCE_S
    picked, gaps = policy._history(policy.frames[-1])
    assert gaps[4] is None and picked[4] is policy.frames[0]


def test_match_tolerance_is_half_the_spacing_with_the_old_floor() -> None:
    assert match_tolerance(np.array([0.0])) == MATCH_TOLERANCE_S
    assert match_tolerance(np.arange(0.0, 3.0, 0.1)) == MATCH_TOLERANCE_S
    assert match_tolerance(np.arange(0.0, 3.0, 0.5)) == pytest.approx(0.25)


def test_projection_is_from_the_raised_reference_frame_and_folds_in_ego_motion() -> None:
    """A world point on the camera's axis 10 m ahead lands at the tile centre at depth 10;
    in a frame taken 5 m further back, the same point is at depth 15."""
    policy = POLICY()
    policy.ref_height = 2.1
    now, past = _frame(1.0, 0.0, size=(800, 450)), _frame(0.5, -5.0, size=(800, 450))
    point_ref = np.array([11.5, 0.0, 1.5 - 2.1, 1.0])  # ego (11.5, 0, 1.5) in the reference frame
    p = policy._projections(now, None, 512)
    assert p.shape == (6, 3, 4)
    u, v, w = p[0] @ point_ref
    assert (u / w, v / w, w) == pytest.approx((256.0, 256.0, 10.0))
    ref_now2world = now.ego2world @ policy._ref2ego()
    u, v, w = policy._projections(past, ref_now2world, 512)[0] @ point_ref
    assert (u / w, v / w, w) == pytest.approx((256.0, 256.0, 15.0))
    assert math.isclose(policy._ref2ego()[2, 3], 2.1)


# -- a benchmark with fewer cameras ------------------------------------------------


def _obs(images=(), paths=(), calibrated=()) -> Observation:
    return Observation(
        images={c: np.zeros((4, 4, 3), dtype=np.uint8) for c in images},
        image_paths={c: f"/tmp/{c}.npy" for c in paths},
        ego2world=np.eye(4),
        canbus=np.zeros(18),
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={c: np.eye(3) for c in calibrated},
        camera2ego={c: np.eye(4) for c in calibrated},
    )


def test_only_a_view_the_benchmark_says_nothing_about_is_left_out() -> None:
    """comma2k19 is the road camera alone; a view that lost its pixels, or the front, still refuses."""
    assert blank_views(_obs(paths=("CAM_FRONT",), calibrated=("CAM_FRONT",))) == frozenset(VIEW_ORDER[1:])
    assert blank_views(_obs(images=VIEW_ORDER, calibrated=VIEW_ORDER)) == frozenset()
    # NAVSIM's dense frames: three images, six calibrations -- missing, not absent.
    with pytest.raises(RuntimeError, match="CAM_BACK"):
        blank_views(_obs(images=VIEW_ORDER[:3], calibrated=VIEW_ORDER))
    with pytest.raises(RuntimeError, match="CAM_FRONT"):
        blank_views(_obs(paths=VIEW_ORDER[1:], calibrated=VIEW_ORDER[1:]))
    with pytest.raises(RuntimeError, match="intrinsics"):
        blank_views(_obs(images=VIEW_ORDER, calibrated=VIEW_ORDER[:5]))


def test_a_left_out_view_is_a_zero_tile_that_sees_nothing() -> None:
    """The road camera is packed and projected as in a six-view frame; the five others put every
    point behind the camera, so upstream's depth test masks them from every BEV cell."""
    policy = POLICY()
    blank = frozenset(VIEW_ORDER[1:])
    front = np.random.default_rng(0).integers(0, 256, (45, 80, 3), dtype=np.uint8)
    mono = _frame(1.0, 0.0, views=("CAM_FRONT",), size=(80, 45))
    mono.images["CAM_FRONT"] = front
    tiles = mono.tiles(blank)
    assert np.array_equal(tiles[0], pack_tile(front, 512))
    assert tiles.shape == (6, 512, 512, 3) and not tiles[1:].any()
    p = policy._projections(mono, None, 512, blank=blank)
    assert np.array_equal(p[0], policy._projections(_frame(1.0, 0.0, size=(80, 45)), None, 512)[0])
    for point in ([20.0, 3.0, -1.0, 1.0], [-30.0, -8.0, 0.5, 1.0], [0.0, 0.0, 0.0, 1.0]):
        assert all((q @ np.array(point))[2] < 0 for q in p[1:])


def test_history_on_a_one_camera_source_is_its_own_frames_half_a_second_apart() -> None:
    """comma2k19 sends the road camera alone at 20 Hz: every frame counts, none is repeated.
    Counting six-view frames only, as on NAVSIM, finds none and repeats the current one."""
    policy = POLICY()
    for k in range(101):  # 5 s at 20 Hz
        policy.frames.append(_frame(k / 20, 0.0, ("CAM_FRONT",)))
    picked, gaps = policy._history(policy.frames[-1], frozenset(VIEW_ORDER[1:]))
    assert [f.t_s for f in picked] == pytest.approx([5.0 + off for off in HISTORY_OFFSETS_S])
    assert gaps == [0.0] * 7
    _, six_view_gaps = policy._history(policy.frames[-1])
    assert six_view_gaps == [None] * 7


def test_a_staged_npy_frame_reports_its_size_without_pil(tmp_path) -> None:
    """comma2k19 stages raw .npy frames; PIL cannot read their header."""
    path = tmp_path / "frame.npy"
    np.save(path, np.zeros((874, 1164, 3), dtype=np.uint8))
    frame = _Frame(
        t_s=0.0, ego2world=np.eye(4), camera2image={}, camera2ego={}, paths={"CAM_FRONT": str(path)}
    )
    assert frame.size("CAM_FRONT") == (1164, 874)


# -- the map and the route: upstream's navigation stack, stood in for ---------------


class _Upstream:
    """What `_Navigation` asks of upstream's map loader, matcher and rasterizer, recorded."""

    def __init__(self) -> None:
        self.loaded: list[Path] = []
        self.parsed: list[str] = []
        self.matched: list[dict] = []
        self.rendered: list[tuple] = []

    def modules(self) -> SimpleNamespace:
        calls = self

        class MapAdapter:
            def __init__(self, scene_map, **kwargs) -> None:
                self.scene_map, self.kwargs = scene_map, kwargs

            def extract(self):
                # The SDK's SceneMap parses map.osm on first use, and extract() is that use.
                calls.parsed.append((self.scene_map / "maps" / "map.osm").read_text())
                return SimpleNamespace(map_version=self.kwargs["map_version"])

        class Matcher:
            def __init__(self, scene_map, navigation_map, **kwargs) -> None:
                pass

            def match(self, **kwargs):
                calls.matched.append(kwargs)
                return SimpleNamespace(valid=True)

        class Rasterizer:
            def render(self, navigation_map, route, pose):
                calls.rendered.append((route, pose))
                return SimpleNamespace(
                    map_context=np.ones((14, NAV_H, NAV_W)),
                    route_mask=np.full((2, NAV_H, NAV_W), route is not None, dtype=np.uint8),
                    map_valid=True,
                    route_valid=route is not None,
                )

        return SimpleNamespace(
            scene_map=lambda path: calls.loaded.append(Path(path)) or Path(path),
            revision="data-revision",
            map_adapter=MapAdapter,
            file_sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            matcher=Matcher,
            pose=lambda **kwargs: SimpleNamespace(**kwargs),
            rasterizer=Rasterizer(),
        )


def _map_files(tmp_path: Path) -> dict[str, str]:
    maps = tmp_path / "dataset" / "scene-1" / "maps"
    maps.mkdir(parents=True)
    (maps / "map.osm").write_text("<osm/>")
    (maps / "origin.json").write_text('{"latitude": 49.0, "longitude": 8.4}')
    return {"format": "lanelet2_osm", "map": str(maps / "map.osm"), "origin": str(maps / "origin.json")}


def _ego2world(x: float, y: float, yaw: float) -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    m[:2, 3] = (x, y)
    return m


def test_the_map_is_read_from_a_scratch_copy_and_drawn_at_the_ego_pose(tmp_path) -> None:
    """Upstream writes a sanitised map.osm beside the one it reads; that must never be the dataset's."""
    upstream, files = _Upstream(), _map_files(tmp_path)
    nav = _Navigation(upstream.modules())
    semantic, route, map_valid, route_valid = nav.raster(files, None, _ego2world(3.0, 4.0, 0.3), 5_000_000)

    (read,) = upstream.loaded
    assert read.name == "scene-1" and not read.is_relative_to(tmp_path / "dataset")
    assert upstream.parsed == ["<osm/>"]
    assert sorted(p.name for p in Path(files["map"]).parent.iterdir()) == ["map.osm", "origin.json"]
    ((drawn_route, pose),) = upstream.rendered
    assert (pose.x_enu_m, pose.y_enu_m, pose.timestamp_ns) == (3.0, 4.0, 5_000_000_000)
    assert pose.yaw_rad == pytest.approx(0.3)
    assert drawn_route is None and map_valid and not route_valid and not route.any()
    assert semantic.dtype == route.dtype == np.float32 and semantic.shape == (14, NAV_H, NAV_W)


def test_nothing_of_the_scratch_copy_outlives_the_parse(tmp_path) -> None:
    """The runner stops a node with terminate() and then kill(), so no finalizer is sure to run:
    a copy kept for the node's life was left on the host by every run, 93 MB each."""
    upstream, files = _Upstream(), _map_files(tmp_path)
    nav = _Navigation(upstream.modules())
    trace = np.array([[0.0, 0.0, 0.0, 0.0], [0.1, 1.0, 0.0, 0.01]])
    for _ in range(2):
        _, _, map_valid, route_valid = nav.raster(files, trace, _ego2world(1.0, 0.0, 0.01), 100_000)
        assert map_valid and route_valid
    (read,) = upstream.loaded
    assert upstream.parsed == ["<osm/>"], "parsed once, from the copy"
    assert not read.parent.exists(), "and the copy is gone once it is"


def test_the_route_is_matched_to_the_logged_drive_once_per_scene_and_drive(tmp_path) -> None:
    upstream, files = _Upstream(), _map_files(tmp_path)
    nav = _Navigation(upstream.modules())
    trace = np.array([[0.0, 0.0, 0.0, 0.0], [0.1, 1.0, 0.0, 0.01], [0.2, 2.0, 0.01, 0.02]])
    for _ in range(2):
        _, route, _, route_valid = nav.raster(files, trace, _ego2world(1.0, 0.0, 0.01), 100_000)
        assert route_valid and route.any()
    (matched,) = upstream.matched
    assert matched["scene_id"] == "scene-1"
    assert np.array_equal(matched["positions_enu_m"], trace[:, 1:3])
    assert np.array_equal(matched["yaws_rad"], trace[:, 3])
    assert matched["timestamps_ns"].tolist() == [0, 100_000_000, 200_000_000]
    nav.raster(files, trace[:2], _ego2world(1.0, 0.0, 0.01), 100_000)
    assert len(upstream.matched) == 2, "another drive is another route"
    assert len(upstream.loaded) == 1, "the map is read once"


def test_a_map_autoe2e_cannot_draw_is_refused_not_dropped(tmp_path, monkeypatch) -> None:
    """Dropping it would score the camera-only track under the map's name."""
    files = _map_files(tmp_path)
    with pytest.raises(RuntimeError, match="Lanelet2"):
        _Navigation(_Upstream().modules()).raster({**files, "format": "opendrive"}, None, np.eye(4), 0)
    monkeypatch.setitem(sys.modules, "navigation", None)
    monkeypatch.setitem(sys.modules, "data_parsing", None)
    with pytest.raises(RuntimeError, match="setup.sh"):
        navigation_modules()
