"""nuScenes open-loop protocol: geometry, the command rule, and the scorer."""

import math

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes import loader as L
from paddock.adapters.benchmarks.nuscenes.harness import plan_headings, plan_on_grid, score_plan
from paddock.policy.protocol import Trajectory


def test_quaternion_identity_and_yaw():
    assert np.allclose(L.quat_to_rot([1, 0, 0, 0]), np.eye(3))
    q90 = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]  # 90 deg about z
    assert abs(L.yaw_of(q90) - math.pi / 2) < 1e-9


def test_world_points_land_in_the_ego_frame():
    ego = L.pose_matrix(
        [10.0, 5.0, 0.0], [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    )  # heading +90 deg
    ahead = np.array([[10.0, 15.0, 0.0]])  # 10 m north of the ego, i.e. straight ahead
    assert np.allclose(L.to_ego_xy(ego, ahead), [[10.0, 0.0]], atol=1e-9)
    boxes = np.array([[10.0, 15.0, math.pi / 2, 4.0, 2.0]])
    local = L.boxes_to_ego(ego, boxes)
    assert np.allclose(local[0, :3], [10.0, 0.0, 0.0], atol=1e-9)


def test_oriented_box_intersection():
    ego = [0.0, 0.0, 0.0, 4.084, 1.85]
    assert L.obb_intersects(ego, [3.0, 0.0, 0.0, 4.0, 2.0])  # overlapping ahead
    assert not L.obb_intersects(ego, [6.0, 0.0, 0.0, 4.0, 2.0])  # clear ahead
    assert not L.obb_intersects(ego, [0.0, 3.0, 0.0, 4.0, 2.0])  # clear to the left
    assert L.obb_intersects(ego, [2.5, 1.3, math.pi / 4, 4.0, 2.0])  # rotated, clipping the corner
    assert L.collides(ego, np.array([[6.0, 0.0, 0.0, 4.0, 2.0], [3.0, 0.0, 0.0, 4.0, 2.0]]))
    assert not L.collides(ego, np.zeros((0, 5)))


def test_command_follows_the_3s_lateral_offset():
    straight = np.column_stack([np.linspace(2, 12, 6), np.zeros(6)])
    assert L.command_from_future(straight) == (2, None, None)
    left = np.column_stack([np.linspace(2, 12, 6), np.linspace(0, 3, 6)])
    cmd, man, dist = L.command_from_future(left)
    # Distance is to where the turn begins (first point past the 2 m threshold),
    # measured along the path FROM THE EGO ORIGIN (so the leg to the first future
    # point counts), not the straight-line end displacement.
    assert (cmd, man) == (1, "TURN_LEFT")
    assert dist == 10.4  # origin->(10,2.4) along the ramp
    right = np.column_stack([np.linspace(2, 12, 6), -np.linspace(0, 3, 6)])
    assert L.command_from_future(right)[:2] == (0, "TURN_RIGHT")


def test_command_distance_from_the_ego_includes_the_first_leg():
    # Already turning at the first future point: distance is the leg to it (from
    # the ego origin), never zero, since the ego is at lateral 0 in its own frame.
    turning = np.column_stack([np.linspace(2, 12, 6), np.linspace(2.5, 8.0, 6)])
    cmd, man, dist = L.command_from_future(turning)
    assert (cmd, man) == (1, "TURN_LEFT")
    assert abs(dist - round(math.hypot(2.0, 2.5), 1)) < 1e-9  # origin -> (2, 2.5)


def test_command_distance_ignores_a_jog_in_the_wrong_direction():
    # A brief right jog past the threshold, then a committed left turn: the
    # distance must be to the LEFT crossing, not the earlier right one.
    y = np.array([-2.5, -2.1, 0.0, 2.1, 2.6, 3.0])
    path = np.column_stack([np.linspace(2, 12, 6), y])
    cmd, man, dist = L.command_from_future(path)
    assert (cmd, man) == (1, "TURN_LEFT")
    full = np.vstack([[0.0, 0.0], path[:, :2]])
    cumulative = np.cumsum(np.linalg.norm(np.diff(full, axis=0), axis=1))
    first_left = np.flatnonzero(path[:, 1] >= L.TURN_THRESHOLD_M)[0]
    assert dist == round(float(cumulative[first_left]), 1)


def test_plan_resampling_and_headings():
    traj = Trajectory(np.column_stack([np.arange(1, 9) * 2.0, np.zeros(8)]), dt=0.5)  # 4 m/s straight
    grid = plan_on_grid(traj)
    assert grid.shape == (6, 2) and np.allclose(grid[:, 0], np.arange(1, 7) * 2.0)
    fine = Trajectory(np.column_stack([np.arange(1, 31) * 0.4, np.zeros(30)]), dt=0.1)  # same plan at 10 Hz
    assert np.allclose(plan_on_grid(fine)[:, 0], np.arange(1, 7) * 2.0)
    # A coarse plan (1 s spacing, first point at 10 m) must interpolate its 0.5 s
    # grid from the ego origin, not flat-hold the first waypoint: x(0.5s) = 5, not 10.
    coarse = Trajectory(np.column_stack([np.arange(1, 4) * 10.0, np.zeros(3)]), dt=1.0)
    assert abs(plan_on_grid(coarse)[0, 0] - 5.0) < 1e-9
    yaws = plan_headings(np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]))
    assert np.allclose(yaws, [math.pi / 4, math.pi / 4, math.pi / 4])  # zero step keeps heading


def test_scorer_l2_and_collision_conventions():
    future = np.column_stack([np.arange(1, 7) * 2.0, np.zeros(6)])
    exact = future.copy()
    empty = [np.zeros((0, 5)) for _ in range(6)]
    m = score_plan(exact, future, empty)
    assert m["l2_1s"] == m["l2_2s"] == m["l2_3s"] == 0.0 and m["col_avg"] == 0.0
    # 1 m lateral error everywhere: point values and cumulative means agree
    off = exact + np.array([0.0, 1.0])
    m = score_plan(off, future, empty)
    assert (
        abs(m["l2_avg"] - 1.0) < 1e-9
        and abs(m["l2_uniad_avg"] - 1.0) < 1e-9
        and abs(m["ade_3s"] - 1.0) < 1e-9
    )
    # a parked car sitting where the plan is at 2 s (step 4) and after
    boxes = [np.zeros((0, 5)) for _ in range(3)] + [
        np.array([[8.0 + 2.0 * k, 0.0, 0.0, 4.0, 2.0]]) for k in range(3)
    ]
    m = score_plan(exact, future, boxes)
    assert m["col_1s"] == 0.0 and m["col_2s"] == 1.0 and m["col_3s"] == 1.0
    assert abs(m["col_uniad_2s"] - 0.25) < 1e-9 and abs(m["col_uniad_3s"] - 0.5) < 1e-9
    assert abs(m["col_avg"] - 2.0 / 3.0) < 1e-9


def test_absent_boxes_omit_collision_metrics_rather_than_zeroing_them():
    # None boxes = "not scored" (a run scored outside the process that built the
    # index). Collision keys must be absent, never a false 0 "collision-free".
    future = np.column_stack([np.arange(1, 7) * 2.0, np.zeros(6)])
    m = score_plan(future.copy(), future, None)
    assert m["l2_avg"] == 0.0 and "ade_3s" in m
    assert not any(k.startswith("col_") for k in m)


def test_a_shared_cache_directory_keeps_one_file_per_dataset_key(tmp_path):
    """Two roots in one directory must not evict each other; the name carries the key."""
    shared = tmp_path / "cache"
    nas = L.cache_file(shared, tmp_path / "nas", "v1.0-trainval", ["a", "b"])
    staged = L.cache_file(shared, tmp_path / "staged", "v1.0-trainval", ["b", "a"])
    assert nas.parent == staged.parent == shared
    assert nas != staged
    same = L.cache_file(shared, tmp_path / "nas", "v1.0-trainval", ["b", "a"])
    assert nas == same, "scene order is not part of the key"


def test_a_failed_cache_write_leaves_the_old_cache_whole(tmp_path, monkeypatch):
    """Parallel lanes read one cache; a write that dies half-way must not be what they read."""
    monkeypatch.setattr(L, "build_index", lambda root, version, scene_names, log=print: [["fresh"]])
    cache = tmp_path / "idx.pkl"
    root = tmp_path / "data"
    first = L.load_or_build_index(root, "v1.0-trainval", ["a"], cache)
    good = cache.read_bytes()

    def dies_half_way(obj, handle):
        handle.write(b"\x80\x04 half a pickle")
        raise OSError("disk full")

    monkeypatch.setattr(L.pickle, "dump", dies_half_way)
    with pytest.raises(OSError, match="disk full"):
        L.load_or_build_index(root, "v1.0-trainval", ["a", "b"], cache)
    assert cache.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".part")] == []
    assert first == [["fresh"]]


def test_index_cache_is_rebuilt_when_the_dataset_key_changes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_build(root, version, scene_names, log=print):
        calls["n"] += 1
        return [[f"{version}:{','.join(scene_names)}"]]

    monkeypatch.setattr(L, "build_index", fake_build)
    cache = tmp_path / "idx.pkl"
    root = tmp_path / "data"

    first = L.load_or_build_index(root, "v1.0-trainval", ["a", "b"], cache)
    assert calls["n"] == 1 and cache.exists()
    # Same args: served from cache, no rebuild.
    again = L.load_or_build_index(root, "v1.0-trainval", ["b", "a"], cache)
    assert calls["n"] == 1 and again == first
    # Different scene set: key mismatch -> rebuild.
    L.load_or_build_index(root, "v1.0-trainval", ["a", "b", "c"], cache)
    assert calls["n"] == 2
    # Different version: rebuild again.
    L.load_or_build_index(root, "v1.0-mini", ["a", "b"], cache)
    assert calls["n"] == 3


def test_a_sample_list_scores_only_its_samples(monkeypatch):
    """The gate's nuScenes subset is spread over every val scene; --limit alone takes a prefix."""
    from types import SimpleNamespace

    from paddock.adapters.benchmarks.nuscenes import harness as NH

    frames = [
        [SimpleNamespace(scene=scene, index_in_scene=i) for i in range(10)]
        for scene in ("scene-0003", "scene-0012")
    ]
    fake = SimpleNamespace(
        _index=lambda: frames,
        _sample=lambda fr, i: SimpleNamespace(sample_id=f"nuscenes/val/{fr[i].scene}/{fr[i].index_in_scene}"),
        _sample_id=NH.NuScenesBenchmark._sample_id,
    )
    fake._scorable = lambda split: NH.NuScenesBenchmark._scorable(fake, split)
    monkeypatch.setenv(NH.ENV_SAMPLES, "nuscenes/val/scene-0012/1,nuscenes/val/scene-0003/2")
    got = [s.sample_id for s in NH.NuScenesBenchmark.samples(fake)]
    assert got == ["nuscenes/val/scene-0003/2", "nuscenes/val/scene-0012/1"]
    monkeypatch.delenv(NH.ENV_SAMPLES)
    assert len(list(NH.NuScenesBenchmark.samples(fake))) == 2 * (10 - NH.N_STEPS)


def test_a_run_on_a_sample_list_is_never_stored_as_a_full_score(tmp_path, monkeypatch):
    """With the list set and no --limit, the run was stored valid: true."""
    from paddock.adapters.benchmarks.nuscenes import harness as NH
    from paddock.config import SiteConfig
    from paddock.registry import Registry

    for sub in ("repos", "runs", "weights", "nuscenes"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n[datasets.nuscenes]\npath = "{tmp_path / "nuscenes"}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    harness = NH.BENCHMARK(Registry.load().benchmarks["nuscenes"], SiteConfig.load())
    assert harness.invalid_reasons() == []
    monkeypatch.setenv(NH.ENV_SAMPLES, "nuscenes/val/scene-0003/0,nuscenes/val/scene-0003/1")
    (reason,) = harness.invalid_reasons()
    assert reason.startswith("2 samples named by PADDOCK_NUSCENES_SAMPLES")
