"""neuroNCAP's PDM terms from a run's stored files, on a synthetic run.

The ego drives straight along +x at 10 m/s for 3 s, planning six waypoints at
0.5 s that carry on at 10 m/s, toward a car stopped at x = 40. Its plans reach
the car from t = 0.6 s on, but the run ends before the ego does: an at-fault
collision planned on most decisions and never driven into. The files have the
shapes the engine writes (`trajectories.json`, `ego_poses.json`, `actors.json`,
`reference_trajectory.json`, `metrics.json`). No GPU, no dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.neuroncap import pdm_terms
from paddock.benchmark import Episode
from paddock.metrics import pdm

T0 = 1_533_151_600_000_000


def _pose(x: float, y: float = 0.0, yaw: float = 0.0) -> list[list[float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def write_run(run_dir: Path, *, car_x: float = 40.0, ego_speed: float = 10.0, collided: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = [T0 + k * 100_000 for k in range(31)]
    plan = [[ego_speed * 0.5 * (i + 1), 0.0] for i in range(6)]
    (run_dir / "trajectories.json").write_text(json.dumps({str(t): plan for t in steps}))
    (run_dir / "ego_poses.json").write_text(
        json.dumps({str(t): _pose(ego_speed * (t - T0) / 1e6) for t in steps})
    )
    keyframes = [T0 + k * 500_000 for k in range(-2, 14)]
    actors = [
        {
            "timestamps": keyframes,
            "poses": [_pose(car_x) for _ in keyframes],
            "dims": [1.9, 4.5, 1.6],
            "uuid": "car-token",
            "is_target_actor": True,
        },
        {  # a pedestrian far away, which neuroNCAP's own check never considers
            "timestamps": keyframes,
            "poses": [_pose(5.0, 30.0) for _ in keyframes],
            "dims": [0.6, 0.6, 1.8],
            "uuid": "ped-token",
            "is_target_actor": False,
        },
    ]
    (run_dir / "actors.json").write_text(json.dumps(actors))
    human = {str(t): [10.0 * (t - T0) / 1e6, 0.0, 0.0] for t in keyframes}
    (run_dir / "reference_trajectory.json").write_text(json.dumps(human))
    (run_dir / "metrics.json").write_text(json.dumps({"any_collide@0.0s": collided, "ncap_score": 5.0}))


CATEGORIES = {"car-token": "vehicle.car", "ped-token": "human.pedestrian.adult"}


def road() -> pdm.DrivableArea:
    cell, origin = 0.25, np.array([-20.0, -20.0])
    ys = origin[1] + (np.arange(160) + 0.5) * cell
    mask = np.repeat(((ys >= -6) & (ys <= 6))[:, None], 1000, axis=1)
    return pdm.DrivableArea(mask=mask, origin=origin, cell=cell)


def test_a_run_planning_into_a_stopped_car_loses_nc_on_the_decisions_that_reach_it(tmp_path: Path) -> None:
    run = tmp_path / "run_0"
    write_run(run)
    terms = pdm_terms.run_terms(run, road(), CATEGORIES)
    assert set(terms) == set(pdm.PDM_KEYS)
    # Decisions at t = 0.0 .. 0.5 s plan short of the car; the other 25 of 31 reach it.
    assert terms["pdm_nc"] == pytest.approx(6 / 31)
    assert terms["pdm_ttc"] <= terms["pdm_nc"], "every plan that hits it also breaks TTC first"
    assert terms["pdm_dac"] == 1.0
    assert 0.0 < terms["pdm_style_score"] < terms["pdm_nc"]


def test_without_a_map_there_are_no_terms_at_all(tmp_path: Path) -> None:
    """Review of #102 (3): with no map, NC, TTC and EP were still recorded, under looser rules."""
    run = tmp_path / "run_0"
    write_run(run, car_x=200.0)
    assert pdm_terms.run_terms(run, None, CATEGORIES) == {}
    assert pdm_terms.run_terms(run, road(), CATEGORIES)["pdm_ep"] == pytest.approx(1.0)


def test_a_run_without_its_files_has_no_terms(tmp_path: Path) -> None:
    (tmp_path / "run_0").mkdir()
    assert pdm_terms.run_terms(tmp_path / "run_0", road(), CATEGORIES) == {}


def test_the_restated_engine_check_sees_a_driven_collision_and_ignores_pedestrians(tmp_path: Path) -> None:
    write_run(tmp_path / "short", car_x=40.0)
    assert pdm_terms.collides_like_neuroncap(tmp_path / "short", CATEGORIES) is False
    write_run(tmp_path / "into", car_x=28.0)
    assert pdm_terms.collides_like_neuroncap(tmp_path / "into", CATEGORIES) is True
    # The same car, filed as a pedestrian, is not a class the engine checks.
    assert (
        pdm_terms.collides_like_neuroncap(tmp_path / "into", {"car-token": "human.pedestrian.adult"}) is False
    )


def test_categories_follow_the_engines_rule_and_static_objects_cost_half() -> None:
    assert pdm_terms._checked("vehicle.bus.rigid") and pdm_terms._checked("vehicle.car")
    assert not pdm_terms._checked("vehicle.construction"), "the engine reads it as 'construction', unlisted"
    assert not pdm_terms._checked("human.pedestrian.adult")
    actor = {"timestamps": [0, 1], "poses": [_pose(0), _pose(1)], "dims": [1, 1, 1], "uuid": "u"}
    (cone,) = pdm_terms.tracks_from_actors([actor], {"u": "movable_object.trafficcone"})
    (unknown,) = pdm_terms.tracks_from_actors([actor], {})
    assert cone.agent is False and unknown.agent is True


def _harness(tmp_path: Path, monkeypatch, *, raster: bool = True):
    """A neuroNCAP harness on a stand-in site: nuScenes' category tables, and scene 0103's raster."""
    from paddock.adapters.benchmarks.neuroncap.harness import BENCHMARK
    from paddock.config import SiteConfig
    from paddock.registry import Registry

    tables = tmp_path / "nuscenes" / "v1.0-trainval"
    tables.mkdir(parents=True)
    (tables / "category.json").write_text(json.dumps([{"token": "c1", "name": "vehicle.car"}]))
    (tables / "instance.json").write_text(json.dumps([{"token": "car-token", "category_token": "c1"}]))
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    if raster:
        area = road()
        np.savez_compressed(geometry / "scene-0103.npz", mask=area.mask, origin=area.origin, cell=area.cell)
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n[datasets.nuscenes]\npath = "{tmp_path / "nuscenes"}"\n'
        f'[benchmarks.neuroncap]\npdm_geometry = "{geometry}"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    return BENCHMARK(Registry.load().benchmarks["neuroncap"], SiteConfig.load())


EPISODE = Episode(episode_id="frontal-0103", scenario="frontal", scene="0103", config={})


def test_the_harness_adds_the_terms_to_each_runs_sample(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(tmp_path, monkeypatch)
    out_dir = tmp_path / "frontal-0103"
    write_run(out_dir / "run_0")
    write_run(out_dir / "run_1", car_x=200.0)
    samples = harness.episode_samples(EPISODE, {}, out_dir)
    by_run = {s["native_id"]["run"]: s["metrics"] for s in samples}
    assert by_run[0]["ncap_score"] == 5.0 and by_run[0]["pdm_nc"] == pytest.approx(6 / 31)
    assert by_run[1]["pdm_nc"] == 1.0 and by_run[1]["pdm_dac"] == 1.0
    assert set(pdm.PDM_KEYS) <= set(harness.spec.metrics)
    assert dict(harness.spec.pdm)["score"] == "pdm_style_score"


def test_one_unreadable_run_loses_only_its_own_terms(tmp_path: Path, monkeypatch) -> None:
    """Review of #102 (6): one bad file dropped the PDM terms of every run in the episode."""
    harness = _harness(tmp_path, monkeypatch)
    out_dir = tmp_path / "frontal-0103"
    write_run(out_dir / "run_0")
    write_run(out_dir / "run_1")
    (out_dir / "run_1" / "trajectories.json").write_text("{")
    by_run = {s["native_id"]["run"]: s["metrics"] for s in harness.episode_samples(EPISODE, {}, out_dir)}
    assert by_run[0]["pdm_nc"] == pytest.approx(6 / 31)
    assert by_run[1]["ncap_score"] == 5.0 and not set(pdm.PDM_KEYS) & set(by_run[1])


def test_a_scene_without_its_map_records_no_terms_and_doctor_says_so(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(tmp_path, monkeypatch, raster=False)
    out_dir = tmp_path / "frontal-0103"
    write_run(out_dir / "run_0")
    (sample,) = harness.episode_samples(EPISODE, {}, out_dir)
    assert sample["metrics"]["ncap_score"] == 5.0 and not set(pdm.PDM_KEYS) & set(sample["metrics"])
    ((label, detail),) = harness.status()
    assert label == "PDM maps" and detail.startswith("14 of 14 scenes have none")
    assert "0103" in detail and "map expansion" in detail


def test_an_error_in_the_pdm_code_leaves_the_ncap_score(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(tmp_path, monkeypatch)
    out_dir = tmp_path / "frontal-0103"
    write_run(out_dir / "run_0")

    def broken(*_a, **_k):
        raise RuntimeError("a bug in the PDM code")

    monkeypatch.setattr(pdm_terms, "run_terms", broken)
    (sample,) = harness.episode_samples(EPISODE, {}, out_dir)
    assert sample["metrics"]["ncap_score"] == 5.0 and not set(pdm.PDM_KEYS) & set(sample["metrics"])


def test_the_geometry_builder_covers_every_neuroncap_scene() -> None:
    import importlib.util

    from paddock.adapters.benchmarks.neuroncap.harness import OFFICIAL_SET
    from paddock.config import REPO_ROOT

    path = REPO_ROOT / "tools" / "benchmarks" / "nuscenes" / "build_pdm_geometry.py"
    spec = importlib.util.spec_from_file_location("build_pdm_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.NEURONCAP_SCENES) == {s for scenes in OFFICIAL_SET.values() for s in scenes}
