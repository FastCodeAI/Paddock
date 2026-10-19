"""`paddock run` plumbing that needs no simulator: scene selection, pre-run
checks, the run record, the open-loop dispatch (parallel nodes and `--resume`),
and `paddock doctor`'s benchmarks section."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paddock import doctor
from paddock.adapters.benchmarks.bench2drive.harness import Bench2DriveBenchmark
from paddock.benchmark import OpenLoopBenchmark, Sample
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


def test_doctor_prints_status_rows_as_facts_rather_than_verdicts() -> None:
    """`Benchmark.status` rows must not read as a fourth status.

    They are descriptions -- "2035 segments" is neither pass nor fail -- so they
    carry no `[ ok ]`/`[warn]` glyph and no `->` fix arrow, which are the two
    marks a reader uses to find what needs doing.
    """
    report = doctor.Report()
    report.add(
        "benchmarks",
        "comma2k19",
        doctor.Status.OK,
        "preflight passed",
        info=[("segments", "2035 in 178 routes"), ("rate", "20 Hz")],
    )
    body = [line for line in report.render().splitlines() if "2035" in line or "20 Hz" in line]
    assert len(body) == 2
    assert not any(sym in line for line in body for sym in ("[ ok ]", "[warn]", "[FAIL]", "->"))
    # Labels padded to a common width, so the values line up as a column.
    assert [line.index("2035") for line in body[:1]] == [line.index("20 Hz") for line in body[1:]]


def test_doctor_reads_the_headroom_of_the_card_the_run_will_use(b2d_site, monkeypatch) -> None:
    """Same rule as `runtime.gpu`, because it is now the same code.

    doctor carried its own nvidia-smi reader that always took row 0, so on a
    two-card box pinned to card 1 it showed the *other* card's headroom -- and
    then the launch gate, which does honour CUDA_VISIBLE_DEVICES, waited on a
    number doctor had never printed. A command whose whole job is "what can this
    machine run" must not be the one place that answers for a different machine.
    """
    from paddock.runtime import gpu

    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(args=[], returncode=0, stdout="24576\n92160\n"),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    check = {c.name: c for c in doctor.run().checks if c.section == "hardware"}["gpu"]
    assert "device 1: 90.0 GB free" in check.detail


def test_doctor_reports_a_status_that_raises_instead_of_dying(b2d_site, monkeypatch) -> None:
    """A broken description must not cost the site its diagnosis."""
    from paddock.benchmark import Benchmark

    def boom(self):
        raise RuntimeError("the NAS went away")

    monkeypatch.setattr(Benchmark, "status", boom)
    report = doctor.run()
    check = {c.name: c for c in report.checks if c.section == "benchmarks"}["bench2drive"]
    assert check.status is doctor.Status.OK
    assert check.info == (("status", "unavailable: RuntimeError: the NAS went away"),)


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

        def model_run_config(self, model):

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

        def model_run_config(self, model):

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

        def model_run_config(self, model):

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

        def model_run_config(self, model):

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


def _site_with_env(tmp_path, model: str, env: dict[str, str]):
    from paddock.config import SiteConfig

    cfg = _cfg(tmp_path)
    body = "".join(f'{k} = "{v}"\n' for k, v in env.items())
    cfg.source.write_text(cfg.source.read_text() + f"[models.{model}.env]\n{body}")
    return SiteConfig.load(cfg.source)


@pytest.mark.parametrize(
    ("env", "valid"),
    [
        ({"VISIONPILOT_ORT_THREADS": "2"}, True),  # declared tuning: what shakaal sets
        ({"OMP_NUM_THREADS": "1"}, True),  # thread counts are tuning for every model
        ({"LD_LIBRARY_PATH": "/opt/cuda/lib", "HF_HOME": "/cache/hf"}, True),  # where things live
        ({"PADDOCK_FFMPEG": "/usr/bin/ffmpeg"}, False),  # another ffmpeg can change the input
        ({"VISIONPILOT_PROVIDERS": "cpu"}, True),  # restating the declared default
        ({"VISIONPILOT_SETPOINT": "run_start"}, False),  # changes the plan
        ({"VISIONPILOT_WEIGHTS": "/opt/other"}, False),  # other weights
        ({"VISIONPILOT_ORT_THREADS": "2", "VISIONPILOT_SETPOINT": "run_start"}, False),
    ],
)
def test_a_site_env_override_that_can_change_the_plan_makes_the_run_unrankable(
    tmp_path, monkeypatch, env, valid
) -> None:
    """End to end through `_run_one_model`, since the bug was in which path applied the rule.

    A `[models.<name>.env]` override was recorded as `model_env` but never read
    by `_validity`, so a finetuned `DRIVEJEPA_CHECKPOINT` run was stored
    `valid: true` and would have been ranked as the released model.
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

        def model_run_config(self, model):
            return {}

        def invalid_reasons(self):
            return []

    monkeypatch.setattr(proc, "model_node", _stub_model_node())
    _isolate_results(tmp_path, monkeypatch)
    cfg = _site_with_env(tmp_path, "visionpilot", env)
    (cfg.runs / "fake" / "visionpilot" / "r").mkdir(parents=True)

    runner._run_one_model(
        cfg, _Harness(), _Harness.spec, _model_spec(), None, None, 1, 9999, lambda _s: None, resume="r"
    )

    written = json.loads((tmp_path / "results" / "fake" / "visionpilot" / "r" / "run.json").read_text())
    assert written["valid"] is valid, written["notes"]
    assert set(written["config"]["model_env"]) == set(env), "the override stays on the record either way"
    if not valid:
        assert "site env override" in written["notes"]
        assert "VISIONPILOT_ORT_THREADS" not in written["notes"], "tuning keys are not named as reasons"


def test_the_released_drivejepa_checkpoint_swapped_by_a_site_is_named(tmp_path) -> None:
    """The case that found the gap: a finetune swapped in through the site config."""
    from paddock.registry import Registry
    from paddock.runner import _env_override_reasons

    spec = Registry.load().models["drivejepa"]
    cfg = _site_with_env(tmp_path, "drivejepa", {"DRIVEJEPA_CHECKPOINT": "/w/finetune/agent.ckpt"})
    (reason,) = _env_override_reasons(cfg, spec)
    assert "DRIVEJEPA_CHECKPOINT" in reason
    assert _env_override_reasons(_cfg(tmp_path), spec) == [], "no env block, nothing to say"


def test_autoe2e_logging_where_it_records_is_not_a_change_of_plan(tmp_path) -> None:
    """AutoE2E's NAVSIM row (`20261009T055926-3ceb18`) set only `AUTOE2E_RECORD_DIR`,
    a log written after the plan is made. It must stay rankable; its checkpoint must not."""
    from paddock.registry import Registry
    from paddock.runner import _env_override_reasons

    spec = Registry.load().models["autoe2e"]
    cfg = _site_with_env(tmp_path, "autoe2e", {"AUTOE2E_RECORD_DIR": "/scratch/autoe2e_calls"})
    assert _env_override_reasons(cfg, spec) == []
    cfg = _site_with_env(tmp_path, "autoe2e", {"AUTOE2E_CHECKPOINT": "/w/other/checkpoint.pt"})
    (reason,) = _env_override_reasons(cfg, spec)
    assert "AUTOE2E_CHECKPOINT" in reason


def test_restating_alpamayos_nav_default_is_not_an_override(tmp_path) -> None:
    """`ALPAMAYO_NAV = "1"` is what its Bench2Drive pooled run set: the default, so
    still rankable. `"0"` is the no-nav baseline, a different plan."""
    from paddock.registry import Registry
    from paddock.runner import _env_override_reasons

    spec = Registry.load().models["alpamayo15"]
    assert _env_override_reasons(_site_with_env(tmp_path, "alpamayo15", {"ALPAMAYO_NAV": "1"}), spec) == []
    (reason,) = _env_override_reasons(_site_with_env(tmp_path, "alpamayo15", {"ALPAMAYO_NAV": "0"}), spec)
    assert "ALPAMAYO_NAV" in reason


def test_every_declared_tuning_key_is_one_the_model_reads() -> None:
    """A misspelt `tuning_env` entry would silently let the real key invalidate,
    or worse, name a key nothing reads. Each must appear in the adapter's source
    or its declared env."""
    from paddock.registry import Registry

    models_dir = Path("src/paddock/adapters/models")
    for name, spec in Registry.load().models.items():
        source = "\n".join(p.read_text() for p in (models_dir / name).glob("*.py"))
        for key in spec.runtime.tuning_env:
            assert key in source or key in spec.runtime.env, f"{name}: tuning_env {key} is read nowhere"


def test_a_real_reason_still_invalidates_alongside_a_remark() -> None:
    """`remarks` must not become a way to launder an invalidating reason."""
    from paddock.runner import _validity

    status, valid, notes = _validity(None, ["2 of the 20 official pairs"], ["resumed: 3 reused"])
    assert valid is False
    assert "2 of the 20 official pairs" in notes and "resumed" in notes


def test_site_paths_are_withheld_from_a_tracked_record() -> None:
    """`run.json` is tracked, so no `[models.*] env` value may name a machine."""
    from paddock.runner import _portable_value

    assert "/opt/weights" not in _portable_value("/opt/weights")
    # A flag keeps its name -- that says what the override does, not where it lives.
    assert _portable_value("--ckpt=/opt/w.pt").startswith("--ckpt=<site path:")
    # Home-relative is machine-specific for the same reason absolute is.
    assert "weights" not in _portable_value("~/weights")
    # Relative values are portable and are the interesting part; they pass through.
    assert _portable_value("cuda") == "cuda"
    assert _portable_value("2") == "2"


def test_two_different_site_paths_do_not_redact_to_the_same_value() -> None:
    """The redaction must not collapse the run id.

    `_measurement_config` feeds `store.make_run_id` and the `--resume` config
    check, so if two checkpoint directories redacted identically, `--resume`
    would continue a run made against one into a run made against the other --
    silently, and only for sites that set an override at all.
    """
    from paddock.runner import _portable_value

    assert _portable_value("/opt/a") != _portable_value("/opt/b")
    assert _portable_value("--ckpt=/w/a.pt") != _portable_value("--ckpt=/w/b.pt")
    # Same path, same digest: an unchanged override must not move the run id.
    assert _portable_value("/opt/a") == _portable_value("/opt/a")


def test_every_segment_of_a_path_list_is_withheld() -> None:
    """`LD_LIBRARY_PATH`-shaped values are the ones that leaked in the first place."""
    from paddock.runner import _portable_value

    out = _portable_value("/opt/x/lib:relative/lib:/srv/y/lib")
    assert "/opt/x" not in out and "/srv/y" not in out
    assert "relative/lib" in out, "a relative segment stays readable"
    assert out.count("<site path:") == 2


# ---- open-loop dispatch: batching, parallel nodes, resume ---------------


class _OpenLoop(OpenLoopBenchmark):
    """`n` cheap samples, for the runner's open-loop path without a dataset.

    Records what it was asked to skip and what it actually yielded, which is
    what the resume tests assert on: a resume that re-reads the whole dataset
    costs the hours it was meant to save, so "did not yield it" is the claim,
    not "did not score it".
    """

    spec = type("S", (), {"name": "fake", "dataset": "d", "mode": "open_loop", "primary_metric": "ade"})()

    def __init__(self, cfg, n=6, nodes=1, fail_at=None, honour_skip=True) -> None:
        super().__init__(self.spec, cfg)
        self.n, self.nodes, self.fail_at, self.honour_skip = n, nodes, fail_at, honour_skip
        self.skipped: list[set[str]] = []
        self.yielded: list[str] = []

    def preflight(self) -> list[str]:
        return []

    def node_count(self, model) -> int:
        return self.nodes

    def samples(self, limit=None, split=None, skip=()):
        from paddock import conformance

        self.skipped.append(set(skip))
        produced = 0
        for i in range(self.n):
            sample_id = f"fake/{i}"
            if self.honour_skip and sample_id in skip:
                continue
            if i == self.fail_at:
                raise RuntimeError("the dataset went away")
            self.yielded.append(sample_id)
            yield Sample(
                sample_id=sample_id,
                observations=[conformance.synthetic_observation(width=8, height=6, timestamp_us=i)],
                ground_truth=np.zeros((8, 2)),
                split="test",
            )
            produced += 1
            if limit is not None and produced >= limit:
                return

    def score(self, sample, prediction) -> dict[str, float]:
        return {"ade": 1.0}


def _stub_open_loop_nodes(monkeypatch, waypoints: int = 8, extras: dict | None = None):
    """Start no subprocess and answer `/infer_sequence` from memory.

    Returns `(urls, requests)`, both filled as the run goes: which nodes were
    started, and which sample each one was asked about. `extras` is what the
    node puts beside the trajectory -- the auxiliary heads a real policy sends
    and the runner has to carry through to the harness and the record.
    """
    import contextlib

    import httpx

    from paddock.runtime import ports, proc

    urls: list[str] = []
    requests: list[tuple[str, int]] = []

    @contextlib.contextmanager
    def _node(_model, _cfg, port, *_a, **_k):
        urls.append(f"http://127.0.0.1:{port}")
        yield urls[-1]

    class _Response:
        def __init__(self, body: dict) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._body

    def _post(url, json=None, timeout=None, **_kwargs):
        requests.append((url.rsplit("/", 1)[0], int(json["observations"][0]["timestamp"])))
        body: dict = {"trajectory": [[float(i), 0.0] for i in range(1, waypoints + 1)]}
        if extras is not None:
            body["extras"] = extras
        return _Response(body)

    def _get(url, timeout=None, **_kwargs):
        # `/diagnostics`, fetched once while the node is still up. Answered
        # here rather than left to fail so the tests below can assert what
        # reaches run.json, which is the whole point of fetching it.
        return _Response({"backend": "onnx", "note": "what the policy says it is"})

    monkeypatch.setattr(proc, "model_node", _node)
    monkeypatch.setattr(ports, "find_free", lambda port: port)
    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(httpx, "get", _get)
    return urls, requests


def _artifacts(cfg):
    """The one run directory under `{runs}`, whatever id it was given."""
    (directory,) = list((cfg.runs / "fake" / "visionpilot").iterdir())
    return directory


def test_batching_keeps_the_tail_whether_or_not_it_prefetches() -> None:
    """A dropped final partial batch is samples silently missing from a sweep."""
    from paddock.runner import _batches

    for prefetch in (False, True):
        batches = list(_batches(iter(range(7)), 3, prefetch=prefetch))
        assert batches == [[0, 1, 2], [3, 4, 5], [6]], f"prefetch={prefetch}"
        assert list(_batches(iter([]), 3, prefetch=prefetch)) == []


def test_a_prefetching_stream_runs_no_more_than_two_batches_ahead() -> None:
    """How far ahead the puller may get is a correctness bound, not a tuning knob.

    A harness releases a sample's inputs once the stream has moved past them --
    comma2k19 deletes a segment's staged frames -- so it has to know how many
    it must still hold. One batch in the consumer's hands, one in the queue and
    one being built is the whole of it.
    """
    import time

    from paddock.runner import _batches

    produced: list[int] = []

    def source():
        for i in range(12):
            produced.append(i)
            yield i

    ahead = 0
    for batch in _batches(source(), 2, prefetch=True):
        time.sleep(0.02)  # a slow consumer: give the puller every chance to run on
        ahead = max(ahead, len(produced) - (batch[-1] + 1))
    assert ahead <= 4, "two batches of two is the bound a harness is entitled to assume"


def test_a_source_that_raises_reaches_the_consumer_rather_than_ending_the_stream() -> None:
    """On a background thread an exception is an early, silent end of stream --
    which is a short sweep reported as a complete one."""
    from paddock.runner import _batches

    def source():
        yield 1
        raise RuntimeError("the dataset went away")

    with pytest.raises(RuntimeError, match="went away"):
        list(_batches(source(), 1, prefetch=True))


def test_a_journal_truncated_mid_write_loses_one_sample_not_the_run(tmp_path) -> None:
    """The process is killed while a line is half written; the rest still counts."""
    from paddock import runner

    (tmp_path / runner.PARTIAL_SAMPLES).write_text('{"sample_id": "a"}\n{"sample_id": "b"}\n{"sample_i')
    assert [r["sample_id"] for r in runner._scored_samples(tmp_path)] == ["a", "b"]


def test_resuming_a_finished_run_reads_its_real_samples_file(tmp_path) -> None:
    """So re-issuing the command after a sweep completed is a no-op, not a repeat."""
    from paddock import runner

    assert runner._scored_samples(tmp_path) == []
    (tmp_path / "samples.jsonl").write_text('{"sample_id": "a"}\n')
    assert [r["sample_id"] for r in runner._scored_samples(tmp_path)] == ["a"]


def test_an_interrupted_open_loop_run_resumes_from_its_journal(tmp_path, monkeypatch) -> None:
    """The one that matters: a sweep dies at sample 4 of 6 and the second attempt
    yields only 4 and 5, keeping the run id and the samples already paid for."""
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    said: list[str] = []
    died = _OpenLoop(cfg, n=6, fail_at=4)
    assert runner._run_open_loop(cfg, died, died.spec, _model_spec(), 9000, None, said.append) == 1

    artifacts = _artifacts(cfg)
    journal = artifacts / runner.PARTIAL_SAMPLES
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [r["sample_id"] for r in lines] == [f"fake/{i}" for i in range(4)]
    assert not (artifacts / "samples.jsonl").exists(), "a partial run is not a result"

    resumed = _OpenLoop(cfg, n=6)
    rc = runner._run_open_loop(
        cfg, resumed, resumed.spec, _model_spec(), 9000, None, said.append, resume=artifacts.name
    )

    assert rc == 0
    assert resumed.skipped == [{f"fake/{i}" for i in range(4)}]
    assert resumed.yielded == ["fake/4", "fake/5"], "the first four must never be built again"
    assert not journal.exists(), "the journal outliving the run is a double count waiting to happen"
    stored = tmp_path / "results" / "fake" / "visionpilot" / artifacts.name / "run.json"
    written = json.loads(stored.read_text())
    assert written["dataset"]["n_samples"] == 6
    assert written["valid"] is True and "resumed" in written["notes"]
    assert "this attempt only" in written["notes"], "the start and duration are the second attempt's"


def test_resume_is_refused_when_the_switches_changed(tmp_path, monkeypatch) -> None:
    """Two measurements under one run id is the failure mode; `--limit` is enough
    to make them two."""
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    died = _OpenLoop(cfg, n=6, fail_at=2)
    runner._run_open_loop(cfg, died, died.spec, _model_spec(), 9000, None, lambda _s: None)
    artifacts = _artifacts(cfg)

    said: list[str] = []
    again = _OpenLoop(cfg, n=6)
    rc = runner._run_open_loop(
        cfg, again, again.spec, _model_spec(), 9000, 3, said.append, resume=artifacts.name
    )
    assert rc == 2
    assert any("differs in limit" in line for line in said)
    assert again.yielded == []


def test_a_harness_that_ignores_skip_still_does_not_score_a_sample_twice(tmp_path, monkeypatch) -> None:
    """`skip` saves work; it is not what keeps the record honest. One id twice in
    samples.jsonl double-counts it in every aggregate."""
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    died = _OpenLoop(cfg, n=6, fail_at=3)
    runner._run_open_loop(cfg, died, died.spec, _model_spec(), 9000, None, lambda _s: None)
    artifacts = _artifacts(cfg)

    stubborn = _OpenLoop(cfg, n=6, honour_skip=False)
    runner._run_open_loop(
        cfg, stubborn, stubborn.spec, _model_spec(), 9000, None, lambda _s: None, resume=artifacts.name
    )

    lines = (artifacts / "samples.jsonl").read_text().splitlines()
    scored = [json.loads(line)["sample_id"] for line in lines]
    assert sorted(scored) == [f"fake/{i}" for i in range(6)]
    assert len(set(scored)) == len(scored)


def test_every_node_a_benchmark_asks_for_is_started_and_used(tmp_path, monkeypatch) -> None:
    """`node_count` is the whole of the open-loop speedup: N nodes score N
    samples in the time one node scores one, and the aggregate is order-free."""
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    urls, requests = _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    harness = _OpenLoop(cfg, n=7, nodes=3)
    rc = runner._run_open_loop(cfg, harness, harness.spec, _model_spec(), 9000, None, lambda _s: None)

    assert rc == 0
    assert len(urls) == 3 and len(set(urls)) == 3, "three nodes on three ports"
    assert len(set(url for url, _ in requests)) == 3, "and every one of them was asked something"
    assert sorted(stamp for _, stamp in requests) == list(range(7))


def test_the_auxiliary_heads_a_node_sends_reach_both_the_harness_and_the_record(
    tmp_path, monkeypatch
) -> None:
    """`extras` is the only part of a prediction the runner does not compute.

    It arrives on the wire and has to survive two hops: into the `Trajectory`
    the harness scores (a metric like `lead_mae` is computed from nothing else),
    and into `samples.jsonl`, because re-running a 10,165-sample sweep to get a
    lead distance back costs hours. Neither hop had a test; the stub node
    returned no `extras` at all, so the plumbing was exercised only in the
    branch where it does nothing.
    """
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    aux = {"pose": {"v": [1.0, 0.0, 0.0]}, "lead": [{"x": 30.0, "v": 12.0, "a": 0.0, "prob": 0.9}]}
    _stub_open_loop_nodes(monkeypatch, extras=aux)
    cfg = _cfg(tmp_path)

    seen: list[dict | None] = []

    class _Scoring(_OpenLoop):
        def score(self, sample, prediction) -> dict[str, float]:
            seen.append(prediction.extras)
            return super().score(sample, prediction)

    harness = _Scoring(cfg, n=3)
    rc = runner._run_open_loop(cfg, harness, harness.spec, _model_spec(), 9000, None, lambda _s: None)

    assert rc == 0
    assert seen == [aux] * 3, "the harness scores what the node sent, not a stripped copy"
    records = [json.loads(line) for line in (_artifacts(cfg) / "samples.jsonl").read_text().splitlines()]
    assert [r["extras"] for r in records] == [aux] * 3


def test_a_model_with_no_auxiliary_heads_writes_no_extras_key(tmp_path, monkeypatch) -> None:
    """`"extras": null` on every sample of every model that has none is noise in
    a file a dashboard reads by key. Absent is the honest encoding, and the
    runner's `if prediction.extras` is what makes it so."""
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    harness = _OpenLoop(cfg, n=2)
    runner._run_open_loop(cfg, harness, harness.spec, _model_spec(), 9000, None, lambda _s: None)

    records = [json.loads(line) for line in (_artifacts(cfg) / "samples.jsonl").read_text().splitlines()]
    assert records and all("extras" not in r for r in records)


# --------------------------------------------------------------------------
# What a run records about itself, and when it gives up
# --------------------------------------------------------------------------


def test_a_finished_open_loop_resume_never_starts_a_node(tmp_path, monkeypatch) -> None:
    """The open-loop half of `test_a_fully_complete_resume_never_starts_a_node`.

    The closed-loop path decides this above its ExitStack and has had a test
    pinning it since it was written; the open-loop path started every node
    first and only then looked at `outstanding`. comma2k19 runs four, so
    re-issuing a finished command paid four checkpoint reads and four VRAM
    waits to rewrite a record it already had.
    """
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    urls, _ = _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    first = _OpenLoop(cfg, n=3, nodes=4)
    assert runner._run_open_loop(cfg, first, first.spec, _model_spec(), 9000, 3, lambda _s: None) == 0
    artifacts = _artifacts(cfg)
    started_by_the_first = len(urls)

    said: list[str] = []
    again = _OpenLoop(cfg, n=3, nodes=4)
    rc = runner._run_open_loop(
        cfg, again, again.spec, _model_spec(), 9000, 3, said.append, resume=artifacts.name
    )

    assert rc == 0
    assert len(urls) == started_by_the_first, "a node was started for a run with nothing outstanding"
    assert any("no node started" in line for line in said)


def test_a_finished_full_sweep_resume_never_starts_a_node_either(tmp_path, monkeypatch) -> None:
    """The case the guard above was written for and could not reach.

    `outstanding` is None whenever there is no `--limit`, so a guard on it
    alone only ever fired for `--limit` runs -- while comma2k19's sweep, the
    run that motivated it, has no limit. Re-issuing a finished 10,165-sample
    run still paid four checkpoint reads and four VRAM waits to rewrite the
    record it already had. Without a limit the count to compare against is the
    harness's own `coverage()`.
    """
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    urls, _ = _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    class _Counted(_OpenLoop):
        def coverage(self):
            return [{"sample_key": str(i)} for i in range(self.n)], []

    first = _Counted(cfg, n=3, nodes=4)
    assert runner._run_open_loop(cfg, first, first.spec, _model_spec(), 9000, None, lambda _s: None) == 0
    artifacts = _artifacts(cfg)
    started_by_the_first = len(urls)

    said: list[str] = []
    again = _Counted(cfg, n=3, nodes=4)
    rc = runner._run_open_loop(
        cfg, again, again.spec, _model_spec(), 9000, None, said.append, resume=artifacts.name
    )

    assert rc == 0
    assert len(urls) == started_by_the_first, "four nodes started for a sweep with nothing left to do"
    assert any("no node started" in line for line in said)


def test_a_wedged_node_stops_the_sweep_instead_of_grinding(tmp_path, monkeypatch) -> None:
    """A node that fails everything costs two weeks and produces an invalid run.

    `score_one` swallows a failure per sample on purpose -- one truncated video
    must not end a 10,165-sample sweep -- and `infer_sequence` waits 600 s for
    each. Nothing put those two together, so a node that wedged early kept four
    pinned nodes busy for a fortnight to write `valid: false`.
    """
    import httpx

    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    attempts = []

    def _wedged(url, json=None, timeout=None, **_kwargs):
        attempts.append(url)
        raise httpx.ReadTimeout("node stopped answering")

    monkeypatch.setattr(httpx, "post", _wedged)

    said: list[str] = []
    bench = _OpenLoop(cfg, n=10_000)
    rc = runner._run_open_loop(cfg, bench, bench.spec, _model_spec(), 9000, None, said.append)

    assert rc == 1, "nothing scored, so nothing is recorded"
    assert len(attempts) <= runner.MAX_CONSECUTIVE_FAILURES + 1, "kept handing samples to a dead node"
    assert any("consecutive sample failures" in line for line in said)


def test_one_wedged_node_among_four_stops_the_sweep_too(tmp_path, monkeypatch) -> None:
    """The count has to be per node, or four nodes make it unreachable.

    A batch of four with one wedged node comes back fail, ok, ok, ok. A single
    counter is reset by the three that worked, so the guard above never fires
    and the sweep runs its full length paying a 600 s timeout a batch -- about
    a fortnight over 10,165 samples, which is the thing it exists to end.
    """
    import httpx

    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)
    answering = httpx.post
    wedged_url = "http://127.0.0.1:9000"
    attempts: list[str] = []

    def _one_node_wedged(url, json=None, timeout=None, **kwargs):
        attempts.append(url)
        if url.startswith(wedged_url):
            raise httpx.ReadTimeout("node stopped answering")
        return answering(url, json=json, timeout=timeout, **kwargs)

    monkeypatch.setattr(httpx, "post", _one_node_wedged)

    said: list[str] = []
    bench = _OpenLoop(cfg, n=4_000, nodes=4)
    runner._run_open_loop(cfg, bench, bench.spec, _model_spec(), 9000, None, said.append)

    dead = [url for url in attempts if url.startswith(wedged_url)]
    assert len(dead) <= runner.MAX_CONSECUTIVE_FAILURES + 1, "kept handing samples to the dead node"
    assert len(attempts) < 4_000, "the other three nodes carried the sweep to the end regardless"
    assert any("consecutive sample failures" in line for line in said)
    assert any(wedged_url in line for line in said), "the abort has to name which node wedged"


def test_a_run_records_the_revision_and_the_policy_that_produced_it(tmp_path, monkeypatch) -> None:
    """The blocker this exists for: a headline number nobody could attribute.

    `RunRecord.repo_sha` has been a field since the store was written and
    nothing ever set it, and nothing in paddock fetched `/diagnostics` -- so
    `openpilot/model.yaml`'s claim that "the difference is recorded in every
    run's diagnostics" was simply false. The published comma2k19 openpilot row
    came from a checkout two shas and one graph layout away from the pin, and
    the only way to find that out was to run `git -C` in the clone.
    """
    import subprocess

    from paddock import runner
    from paddock.runtime import provenance

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    clone = cfg.repo(_model_spec().repo)
    clone.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)
    (clone / "f").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one"],
        cwd=clone,
        check=True,
    )
    head = provenance.git_sha(clone)

    bench = _OpenLoop(cfg, n=2)
    assert runner._run_open_loop(cfg, bench, bench.spec, _model_spec(), 9000, None, lambda _s: None) == 0

    written = json.loads(
        (tmp_path / "results" / "fake" / "visionpilot" / _artifacts(cfg).name / "run.json").read_text()
    )
    assert written["repo_sha"] == head
    assert written["provenance"]["repo"]["at_pin"] is False, "the pin is not this throwaway commit"
    assert written["provenance"]["policy"]["backend"] == "onnx"
    # Not in `config`: config is hashed into the run id and compared by
    # `--resume`, so a `git pull` between attempts must not refuse the resume.
    assert "repo" not in written["config"] and "backend" not in written["config"]


def test_a_run_that_scored_fewer_samples_than_it_should_have_says_so(tmp_path, monkeypatch) -> None:
    """`n: 10105`, `valid: true`, `notes: ""` -- with 60 samples silently gone.

    A route whose calibration failed or a window running off a truncated video
    drops samples from the stream with nothing recorded. `coverage()` already
    knew the expected count; the record did not carry it, so a partial dataset
    ranked beside complete ones.
    """
    from paddock import runner

    _isolate_results(tmp_path, monkeypatch)
    _stub_open_loop_nodes(monkeypatch)
    cfg = _cfg(tmp_path)

    class _Short(_OpenLoop):
        def coverage(self):
            return [{"sample_key": str(i)} for i in range(6)], []

    bench = _Short(cfg, n=4)
    assert runner._run_open_loop(cfg, bench, bench.spec, _model_spec(), 9000, None, lambda _s: None) == 0

    written = json.loads(
        (tmp_path / "results" / "fake" / "visionpilot" / _artifacts(cfg).name / "run.json").read_text()
    )
    assert written["dataset"]["n_expected"] == 6
    assert written["dataset"]["n_samples"] == 4
    assert written["valid"] is False
    assert "scored 4 of 6" in written["notes"]


def test_resuming_reads_both_a_finished_file_and_a_later_journal(tmp_path) -> None:
    """Three attempts, and the second one's work must not vanish.

    `_store_run` unlinks the journal when it writes `samples.jsonl`, so holding
    both means a resume completed and was then interrupted again. Returning the
    journal alone -- the first of the two, as this did -- silently discards
    everything the completed attempt scored, and the run stores short with
    nothing in `notes` to say why.
    """
    from paddock import runner

    (tmp_path / "samples.jsonl").write_text('{"sample_id": "a"}\n{"sample_id": "b"}\n')
    (tmp_path / runner.PARTIAL_SAMPLES).write_text('{"sample_id": "b"}\n{"sample_id": "c"}\n')

    assert sorted(r["sample_id"] for r in runner._scored_samples(tmp_path)) == ["a", "b", "c"]
