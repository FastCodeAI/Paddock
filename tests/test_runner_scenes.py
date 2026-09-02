"""`paddock run` plumbing that needs no simulator: scene selection, pre-run
checks, the run record, and `paddock doctor`'s benchmarks section."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from paddock import doctor
from paddock.adapters.benchmarks.bench2drive.harness import Bench2DriveBenchmark
from paddock.config import SiteConfig
from paddock.runner import run_benchmark

FIXTURES = Path("src/paddock/adapters/benchmarks/bench2drive/tests/fixtures")


@pytest.fixture
def b2d_site(tmp_path, monkeypatch) -> SiteConfig:
    """A site with the files Bench2Drive's preflight looks for, and nothing else."""
    carla = tmp_path / "carla"
    (carla / "PythonAPI" / "carla" / "agents" / "navigation").mkdir(parents=True)
    (carla / "CarlaUE4.sh").write_text("")
    (carla / "PythonAPI" / "carla" / "agents" / "navigation" / "local_planner.py").write_text("")
    repo = tmp_path / "Bench2Drive"
    for relative in (
        "leaderboard/leaderboard/leaderboard_evaluator.py",
        "leaderboard/leaderboard/autoagents/autonomous_agent.py",
        "scenario_runner/srunner/scenariomanager/carla_data_provider.py",
    ):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("")
    (repo / "leaderboard" / "data").mkdir()
    shutil.copy(FIXTURES / "routes_sample.xml", repo / "leaderboard" / "data" / "routes_sample.xml")
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site.toml").write_text(
        f'site = "test-site"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n[benchmarks.bench2drive]\npython = "{sys.executable}"\n'
        f'carla_root = "{carla}"\nrepo = "{repo}"\nroutes_file = "routes_sample.xml"\n'
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    # This interpreter has no CARLA client library; that check has its own test.
    monkeypatch.setattr(Bench2DriveBenchmark, "_import_problems", lambda self: [])
    return SiteConfig.load()


def _dry_run(cfg, **kwargs) -> tuple[int, str]:
    lines: list[str] = []
    code = run_benchmark(
        cfg, bench="bench2drive", model="drivejepa", dry_run=True, echo=lines.append, **kwargs
    )
    return code, "\n".join(lines)


def test_no_scene_means_the_benchmarks_default_not_another_benchmarks_id(b2d_site) -> None:
    code, out = _dry_run(b2d_site)
    assert code == 0, out
    assert "1711-Town12-ParkingCutIn, 1773-Town12-ParkedObstacle" in out
    assert "0103" not in out


def test_scene_is_comma_separated_like_scenarios(b2d_site) -> None:
    code, out = _dry_run(b2d_site, scene="1711,1773")
    assert code == 0, out
    assert "1711-Town12-ParkingCutIn, 1773-Town12-ParkedObstacle" in out
    code, out = _dry_run(b2d_site, scene="1773")
    assert "1711" not in out and "1773-Town12-ParkedObstacle" in out


def test_an_unknown_scene_or_scenario_stops_the_run_before_any_model_loads(b2d_site) -> None:
    code, out = _dry_run(b2d_site, scene="9999")
    assert code == 1 and "9999" in out and "Town12" in out
    code, out = _dry_run(b2d_site, scenarios=["ParkedObstacl"])
    assert code == 1 and "ParkedObstacl" in out and "ParkedObstacle" in out


def test_neuroncap_reports_an_unknown_scenario_set_too(b2d_site) -> None:
    from paddock.adapters.benchmarks.neuroncap.harness import BENCHMARK
    from paddock.registry import Registry

    harness = BENCHMARK(Registry.load().benchmarks["neuroncap"], b2d_site)
    assert harness.scenario_problems(["frontal"]) == []
    assert "stationary" in harness.scenario_problems(["frontl"])[0]


def test_doctor_runs_every_benchmarks_preflight(b2d_site) -> None:
    """The doc promises a green doctor means preflight passed; make it true."""
    report = doctor.run()
    by_name = {c.name: c for c in report.checks if c.section == "benchmarks"}
    assert by_name["bench2drive"].status is doctor.Status.OK, by_name["bench2drive"]
    assert by_name["neuroncap"].status is doctor.Status.WARN
    assert "[benchmarks.neuroncap]" in by_name["neuroncap"].fix
