"""`paddock run` plumbing that needs no simulator: scene selection, pre-run
checks, the run record, and `paddock doctor`'s benchmarks section."""

from __future__ import annotations

import json
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


def test_base_port_is_validated_and_named(monkeypatch) -> None:
    from paddock.runner import DEFAULT_BASE_PORT, base_port

    monkeypatch.setenv("PADDOCK_BASE_PORT", "")
    assert base_port() == DEFAULT_BASE_PORT
    monkeypatch.setenv("PADDOCK_BASE_PORT", "9400")
    assert base_port() == 9400
    for bad in ("abc", "9000-9100", "70000"):
        monkeypatch.setenv("PADDOCK_BASE_PORT", bad)
        with pytest.raises(ValueError, match="PADDOCK_BASE_PORT"):
            base_port()


def test_validity_is_decided_in_one_place() -> None:
    """A run is valid only when nothing failed and nothing about how it was
    made keeps it from being ranked; the reasons land in notes."""
    from paddock.runner import _validity

    assert _validity(None, []) == ("complete", True, "")
    assert _validity(None, ["--limit 5: a subset"]) == ("complete", False, "--limit 5: a subset")
    assert _validity("2 sample(s) failed", ["a subset"]) == ("partial", False, "a subset; 2 sample(s) failed")


# ---- closed-loop resume -------------------------------------------------


def _stub_model_node(url="http://127.0.0.1:9999"):
    import contextlib

    @contextlib.contextmanager
    def _node(*_a, **_k):
        yield url

    return _node


def test_resume_drives_only_the_episodes_that_did_not_finish(tmp_path, monkeypatch) -> None:
    """Through `_run_one_model`, because the bug this guards is in the wiring.

    A resume that ignores what is on disk costs the hours it was meant to save.
    """
    from paddock import runner
    from paddock.benchmark import Episode
    from paddock.runtime import proc

    driven: list[str] = []

    class _Harness:
        spec = type(
            "S", (), {"name": "fake", "dataset": "d", "mode": "closed_loop", "primary_metric": "score"}
        )()

        def episodes(self, scenes=None, runs=1, scenarios=None):
            for i in ("a", "b", "c"):
                yield Episode(episode_id=i, scenario="frontal", scene=i)

        def completed_metrics(self, episode, out_dir, runs=1):
            return {"score": 1.0} if episode.episode_id in ("a", "b") else None

        def run_episode(self, episode, policy_url, out_dir, echo=print):
            driven.append(episode.episode_id)
            return {"score": 2.0}

        def episode_samples(self, episode, metrics, out_dir):
            return [
                {
                    "sample_id": f"fake/{episode.episode_id}",
                    "native_id": {},
                    "split": episode.scenario,
                    "metrics": metrics,
                }
            ]

        def samples_per_episode(self, runs=1):
            return 1

        def run_config(self):
            return {}

        def invalid_reasons(self):
            return []

    monkeypatch.setattr(proc, "model_node", _stub_model_node())
    _isolate_results(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path)
    artifacts = cfg.runs / "fake" / "visionpilot" / "resume-me"
    artifacts.mkdir(parents=True)

    said: list[str] = []
    runner._run_one_model(
        cfg,
        _Harness(),
        _Harness.spec,
        _model_spec(),
        None,
        None,
        10,
        9999,
        said.append,
        resume="resume-me",
    )

    assert driven == ["c"], "only the unfinished episode should be driven again"
    assert sum("reusing" in s for s in said) == 2


def test_resume_refuses_a_run_id_that_does_not_exist(tmp_path, monkeypatch) -> None:
    """A typo must not silently start a fresh run under that name."""
    from paddock import runner
    from paddock.benchmark import Episode

    class _Harness:
        spec = type(
            "S", (), {"name": "fake", "dataset": "d", "mode": "closed_loop", "primary_metric": "score"}
        )()

        def episodes(self, scenes=None, runs=1, scenarios=None):
            yield Episode(episode_id="a", scenario="frontal", scene="a")

        def run_config(self):
            return {}

    said: list[str] = []
    rc = runner._run_one_model(
        _cfg(tmp_path),
        _Harness(),
        _Harness.spec,
        _model_spec(),
        None,
        None,
        1,
        9999,
        said.append,
        resume="20260101T000000-nosuch",
    )
    assert rc == 2
    assert any("nothing to resume" in s for s in said)


def _model_spec():
    from paddock.registry import Registry

    return Registry.load().models["visionpilot"]


def _isolate_results(tmp_path, monkeypatch):
    """Keep a test's run records out of the repo's tracked `results/`.

    `store.write_run` defaults `root` to RESULTS_ROOT, bound at definition, so
    patching the module attribute does not reach it -- the function itself has
    to be wrapped. Without this a runner test writes real run.json and
    metrics.json into results/, where they are indistinguishable from a
    measurement and would be committed by the next `git add`.
    """
    from paddock.results import store

    real = store.write_run

    def _write(run, metrics, samples=None, root=None, artifacts_root=None):
        return real(run, metrics, samples, root=tmp_path / "results", artifacts_root=artifacts_root)

    monkeypatch.setattr(store, "write_run", _write)


def _cfg(tmp_path):
    from paddock.config import SiteConfig

    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "site.toml").write_text(
        f'site = "t"\n[roots]\nrepos = "{tmp_path}"\nruns = "{tmp_path / "runs"}"\nweights = "{tmp_path}"\n'
    )
    return SiteConfig.load(tmp_path / "site.toml")


def test_a_fully_complete_resume_never_starts_a_node(tmp_path, monkeypatch) -> None:
    """Re-issuing the command to see where a sweep got to must not cost a model load.

    Alpamayo is a 21 GB checkpoint read plus a VRAM wait on a shared card. Paying
    that to reuse everything and write the record is an hour for nothing, and
    with a 12-hour sweep people will re-issue the command.
    """
    from paddock import runner
    from paddock.benchmark import Episode
    from paddock.runtime import proc

    started = []

    def _never(*_a, **_k):
        started.append(1)
        raise AssertionError("no node should be started when nothing is outstanding")

    class _Harness:
        spec = type(
            "S", (), {"name": "fake", "dataset": "d", "mode": "closed_loop", "primary_metric": "score"}
        )()

        def episodes(self, scenes=None, runs=1, scenarios=None):
            yield Episode(episode_id="a", scenario="frontal", scene="a")

        def completed_metrics(self, episode, out_dir, runs=1):
            return {"score": 4.0}

        def episode_samples(self, episode, metrics, out_dir):
            return [{"sample_id": "fake/a", "native_id": {}, "split": "frontal", "metrics": metrics}]

        def samples_per_episode(self, runs=1):
            return 1

        def run_config(self):
            return {}

        def invalid_reasons(self):
            return []

    monkeypatch.setattr(proc, "model_node", _never)
    _isolate_results(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path)
    (cfg.runs / "fake" / "visionpilot" / "done").mkdir(parents=True)

    said = []
    rc = runner._run_one_model(
        cfg,
        _Harness(),
        _Harness.spec,
        _model_spec(),
        None,
        None,
        10,
        9999,
        said.append,
        resume="done",
    )

    assert started == []
    assert rc == 0
    assert any("already scored" in s for s in said)
    assert any("without starting a node" in s for s in said)


def test_a_resumed_run_is_still_rankable(tmp_path, monkeypatch) -> None:
    """Resuming says something about how a row was produced, not about its worth.

    Everything in `invalid_reasons` sets `valid: false`, so folding the resume
    note in there excluded a perfectly good run from the leaderboard. That is
    what `remarks` is for, and this asserts the distinction end to end rather
    than trusting the call site to keep using the right channel.
    """
    from paddock import runner
    from paddock.benchmark import Episode
    from paddock.runtime import proc

    class _Harness:
        spec = type(
            "S", (), {"name": "fake", "dataset": "d", "mode": "closed_loop", "primary_metric": "score"}
        )()

        def episodes(self, scenes=None, runs=1, scenarios=None):
            yield Episode(episode_id="a", scenario="frontal", scene="a")

        def completed_metrics(self, episode, out_dir, runs=1):
            return {"score": 4.0}

        def episode_samples(self, episode, metrics, out_dir):
            return [{"sample_id": "fake/a", "native_id": {}, "split": "frontal", "metrics": metrics}]

        def samples_per_episode(self, runs=1):
            return 1

        def run_config(self):
            return {}

        def invalid_reasons(self):
            return []

    monkeypatch.setattr(proc, "model_node", _stub_model_node())
    _isolate_results(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path)
    (cfg.runs / "fake" / "visionpilot" / "r").mkdir(parents=True)

    runner._run_one_model(
        cfg,
        _Harness(),
        _Harness.spec,
        _model_spec(),
        None,
        None,
        1,
        9999,
        lambda _s: None,
        resume="r",
    )

    written = json.loads((tmp_path / "results" / "fake" / "visionpilot" / "r" / "run.json").read_text())
    assert written["valid"] is True, "a resumed run must stay rankable"
    assert "resumed" in written["notes"] and "this attempt only" in written["notes"]
    assert written["dataset"]["n_reused"] == 1


def test_a_real_reason_still_invalidates_alongside_a_remark() -> None:
    """`remarks` must not become a way to launder an invalidating reason."""
    from paddock.runner import _validity

    status, valid, notes = _validity(None, ["2 of the 20 official pairs"], ["resumed: 3 reused"])
    assert valid is False
    assert "2 of the 20 official pairs" in notes and "resumed" in notes
