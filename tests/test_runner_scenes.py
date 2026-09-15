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


def test_a_real_reason_still_invalidates_alongside_a_remark() -> None:
    """`remarks` must not become a way to launder an invalidating reason."""
    from paddock.runner import _validity

    status, valid, notes = _validity(None, ["2 of the 20 official pairs"], ["resumed: 3 reused"])
    assert valid is False
    assert "2 of the 20 official pairs" in notes and "resumed" in notes


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


def _stub_open_loop_nodes(monkeypatch, waypoints: int = 8):
    """Start no subprocess and answer `/infer_sequence` from memory.

    Returns `(urls, requests)`, both filled as the run goes: which nodes were
    started, and which sample each one was asked about.
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
        return _Response({"trajectory": [[float(i), 0.0] for i in range(1, waypoints + 1)]})

    monkeypatch.setattr(proc, "model_node", _node)
    monkeypatch.setattr(ports, "find_free", lambda port: port)
    monkeypatch.setattr(httpx, "post", _post)
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
