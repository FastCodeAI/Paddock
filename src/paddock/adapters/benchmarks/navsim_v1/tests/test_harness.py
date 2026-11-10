"""NAVSIM v1 PDMS: the subclass, its vocabulary, the checks that keep the two
devkit generations apart, and the node-less runs (baseline, human, replay).

The stand-in site comes from navsim/tests/test_harness.py; nothing here needs
a devkit, a cache or a model, except the one site-gated test at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.navsim import convert, nodeless
from paddock.adapters.benchmarks.navsim import harness as v2
from paddock.adapters.benchmarks.navsim.harness import (
    CONSTANT_VELOCITY_AGENT,
    ENV_AGENT,
    ENV_ALLOW_PARTIAL,
    ENV_ALLOW_SLOW,
    HUMAN_AGENT,
    PROXY_AGENT,
    REPLAY_AGENT,
    NavsimBenchmark,
    limit_problems,
    merge_chunk_csvs,
)
from paddock.adapters.benchmarks.navsim.tests.test_harness import _harness, _navsim_site
from paddock.adapters.benchmarks.navsim_v1.harness import CONF_DIR as V1_CONF_DIR
from paddock.adapters.benchmarks.navsim_v1.harness import SCORER as V1_SCORER
from paddock.adapters.benchmarks.navsim_v1.harness import NavsimV1Benchmark
from paddock.config import REPO_ROOT, ConfigError, SiteConfig
from paddock.registry import Registry
from paddock.results import store
from paddock.results.ingest.navsim import EPDMS, PDMS, parse_scores
from paddock.runner import load_benchmark


def _v1_site(tmp_path: Path, monkeypatch, **kwargs):
    kwargs.setdefault("bench", "navsim_v1")
    kwargs.setdefault("scorer", V1_SCORER)
    return _navsim_site(tmp_path, monkeypatch, **kwargs)


#: A stand-in for the v1 devkit's scorer: the v1 columns behind an unnamed index
#: column, three scored rows per log, and the one `average` row whose `valid` is
#: the AND of the others -- as run_pdm_score.py writes them.
V1_STUB_SCORER = """\
import os, sys
args = sys.argv[1:]
def opt(prefix):
    return next((a[len(prefix):] for a in args if a.startswith(prefix)), "")
out = opt("output_dir=")
logs = [x.strip("'") for x in opt("train_test_split.scene_filter.log_names=").strip("[]").split(",") if x]
if os.environ.get("STUB_FAIL_LOG") and os.environ["STUB_FAIL_LOG"] in logs:
    print("stub scorer: boom", flush=True)
    sys.exit(1)
from paddock.results.ingest.navsim import PDMS
os.makedirs(out, exist_ok=True)
ones = ["1.0"] * len(PDMS.sub_metrics)
n = 0
with open(os.path.join(out, "2026.01.01.00.00.00.csv"), "w") as f:
    f.write(",".join(["", "token", "valid", *PDMS.sub_metrics, "score"]) + "\\n")
    for log in logs or ["single"]:
        for i in range(3):
            f.write(",".join([str(n), f"{log}-t{i}", "True", *ones, "0.5"]) + "\\n")
            n += 1
    f.write(",".join([str(n), "average", "True", *ones, "0.5"]) + "\\n")
print("stub scorer done", flush=True)
"""


def _stub_v1_site(tmp_path: Path, monkeypatch, **settings):
    cfg = _v1_site(tmp_path, monkeypatch, tokens_per_log=3, **settings)
    (tmp_path / "devkit" / V1_SCORER).write_text(V1_STUB_SCORER)
    monkeypatch.delenv("STUB_FAIL_LOG", raising=False)
    return cfg


# -- registry and the subclass ----------------------------------------------------


def test_registered_as_its_own_benchmark(tmp_path: Path, monkeypatch) -> None:
    spec = Registry.load().benchmarks["navsim_v1"]
    assert spec.dataset == "navsim" and spec.primary_metric == "pdms" and spec.mode == "open_loop"
    assert tuple(spec.metrics) == ("pdms", *PDMS.sub_metrics)
    assert spec.directory.name == "navsim_v1"
    harness = _harness(_v1_site(tmp_path, monkeypatch), "navsim_v1")
    assert isinstance(harness, NavsimV1Benchmark) and isinstance(harness, NavsimBenchmark)
    assert harness.VOCABULARY is PDMS and harness.MIN_SCENARIOS == 1 and harness.DEVKIT_MAJOR == "1"
    assert NavsimBenchmark.VOCABULARY is EPDMS and NavsimBenchmark.DEVKIT_MAJOR == "2"


def test_the_command_names_the_v1_scorer_and_conf_dir(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_v1_site(tmp_path, monkeypatch), "navsim_v1")
    config = {"cameras": ["CAM_FRONT"], "model": "m"}
    command = harness._command("http://127.0.0.1:1", tmp_path / "out", config, None)
    assert command[2].endswith(V1_SCORER) and f"file://{V1_CONF_DIR}" in command[3]
    assert f"agent={PROXY_AGENT}" in command
    assert any(arg.startswith("agent.node_url=") for arg in command)

    monkeypatch.setenv(ENV_AGENT, CONSTANT_VELOCITY_AGENT)
    config = {"cameras": [], "model": "constant_velocity"}
    command = harness._command("", tmp_path / "out", config, None)
    assert not any(arg.startswith("agent.") for arg in command), "no node, so no node arguments"

    monkeypatch.setenv(ENV_AGENT, REPLAY_AGENT)
    config = {"cameras": [], "model": "alpamayo15", "plans_dir": "/runs/navsim/alpamayo15/x"}
    command = harness._command("", tmp_path / "out", config, None)
    assert "agent.plans_dir=/runs/navsim/alpamayo15/x" in command


def test_the_devkit_goes_first_on_pythonpath_for_both_generations(tmp_path: Path, monkeypatch) -> None:
    """The scorer interpreter's `navsim` is an editable install of one
    generation; PYTHONPATH decides which tree the scorer -- and hydra's pkg://
    search path -- imports, so the devkit being scored must lead."""
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    for bench, site in (("navsim", _navsim_site), ("navsim_v1", _v1_site)):
        root = tmp_path / bench
        harness = _harness(site(root, monkeypatch), bench)
        parts = harness._environment(root / "data", root / "exp")["PYTHONPATH"].split(":")
        assert parts[0] == str(root / "devkit"), bench
        assert parts[1] == str(REPO_ROOT / "src") and parts[2] == "/elsewhere"


# -- the checks that keep the generations apart -------------------------------------


def test_the_other_generations_devkit_is_refused(tmp_path: Path, monkeypatch) -> None:
    """Both trees ship a run_pdm_score.py, so the scorer path proves little; a
    v2 tree scored under a PDMS label would look exactly like a result."""
    cfg = _v1_site(tmp_path / "a", monkeypatch, devkit_version="2.0.0")
    problems = _harness(cfg, "navsim_v1").preflight()
    assert len(problems) == 1 and "declares navsim 2.x" in problems[0]
    assert "[benchmarks.navsim_v1] needs a v1 devkit" in problems[0]

    cfg = _navsim_site(tmp_path / "b", monkeypatch, devkit_version="1.1.0")
    problems = _harness(cfg).preflight()
    assert len(problems) == 1 and "declares navsim 1.x" in problems[0] and "needs a v2 devkit" in problems[0]

    cfg = _v1_site(tmp_path / "c", monkeypatch)
    (tmp_path / "c" / "devkit" / "setup.py").unlink()
    assert _harness(cfg, "navsim_v1").preflight() == [], "no setup.py, no opinion"


def test_an_unpatched_v1_devkit_is_refused_before_a_run(tmp_path: Path, monkeypatch) -> None:
    """Without Camera.camera_path every scenario fails inside the scorer, hours
    in; preflight says so in a second and names the patch."""
    cfg = _v1_site(tmp_path, monkeypatch, camera_path=False)
    problems = _harness(cfg, "navsim_v1").preflight()
    assert len(problems) == 1 and "camera_path" in problems[0] and "patches/README.md" in problems[0]


def test_a_cache_without_its_metadata_csv_is_refused(tmp_path: Path, monkeypatch) -> None:
    """The scorer reads the cache through metadata/*.csv, written last by the
    build; an interrupted build has the files and no csv."""
    cfg = _v1_site(tmp_path, monkeypatch, tokens_per_log=3, metadata_csv=False)
    problems = _harness(cfg, "navsim_v1").preflight()
    assert len(problems) == 1 and "no metadata/*.csv" in problems[0] and "re-run" in problems[0].lower()


def test_a_cache_listing_fewer_scenarios_than_declared_is_a_subset(tmp_path: Path, monkeypatch) -> None:
    """A csv from an earlier, smaller build silently scores a subset under the
    split's name -- refused like partial sensor coverage, and never valid."""
    cfg = _v1_site(tmp_path, monkeypatch, tokens_per_log=3, declare_tokens=True)
    csv_path = tmp_path / "cache" / "metadata" / "metric_cache_metadata_node_0.csv"
    lines = csv_path.read_text().splitlines()
    csv_path.write_text("\n".join(lines[:3]) + "\n")  # header + 2 of the 6 declared
    harness = _harness(cfg, "navsim_v1")
    problems = harness.preflight()
    assert len(problems) == 1 and "lists 2 of the 6 scenarios navtest declares" in problems[0]
    assert harness.invalid_reasons() == []

    monkeypatch.setenv(ENV_ALLOW_PARTIAL, "1")
    assert harness.preflight() == []
    reasons = harness.invalid_reasons()
    assert len(reasons) == 1 and "lists 2 of the 6" in reasons[0]
    assert harness.switch_reasons() == reasons and harness.agent_reasons() == []


def test_messages_name_the_v1_section_and_doc(tmp_path: Path, monkeypatch) -> None:
    cfg = _navsim_site(tmp_path, monkeypatch)  # only [benchmarks.navsim] is configured
    harness = load_benchmark(Registry.load().benchmarks["navsim_v1"], cfg)
    problems = harness.preflight()
    assert len(problems) == 1
    assert "[benchmarks.navsim_v1]" in problems[0] and "docs/benchmarks/navsim_v1.md" in problems[0]
    assert "[benchmarks.navsim]" not in problems[0] and "docs/benchmarks/navsim.md" not in problems[0]


def test_conf_yamls_name_the_intended_classes() -> None:
    """hydra resolves agents by dotted path; a rename must break a test, not a run."""
    proxy = (V1_CONF_DIR / "agent" / f"{PROXY_AGENT}.yaml").read_text()
    assert "paddock.adapters.benchmarks.navsim_v1.agent.PaddockV1ProxyAgent" in proxy
    assert "node_url: ???" in proxy
    shared = {
        CONSTANT_VELOCITY_AGENT: "PaddockConstantVelocityAgent",
        HUMAN_AGENT: "PaddockHumanAgent",
        REPLAY_AGENT: "PaddockReplayAgent",
    }
    for conf_dir in (V1_CONF_DIR, v2.CONF_DIR):
        for name, cls in shared.items():
            text = (conf_dir / "agent" / f"{name}.yaml").read_text()
            assert f"paddock.adapters.benchmarks.navsim.agent.{cls}" in text, (conf_dir, name)
        assert "plans_dir: ???" in (conf_dir / "agent" / f"{REPLAY_AGENT}.yaml").read_text()


# -- the vocabulary -------------------------------------------------------------


def _v1_csv(path: Path) -> Path:
    terms = ",".join(PDMS.sub_metrics)
    path.write_text(
        f",token,valid,{terms},score\n"
        "0,aaa,True,1.0,1.0,0.8,1.0,1.0,1.0,0.9166667\n"
        "1,bbb,True,0.5,1.0,0.6,0.0,1.0,0.5,0.2916667\n"
        "2,ccc,False,,,,,,,\n"
        "3,average,False,0.75,1.0,0.7,0.5,1.0,0.75,0.6041667\n"
    )
    return path


def test_the_pdms_vocabulary_parses_a_v1_csv(tmp_path: Path) -> None:
    """Golden: v1 columns, the unnamed index column, a failed row, and the
    `average` row -- whose `valid` reads False after one failure and must be
    context, never a failed scenario."""
    path = _v1_csv(tmp_path / "s.csv")
    samples, aggregates, failed = parse_scores(path, vocabulary=PDMS, benchmark="navsim_v1")
    assert [s["sample_id"] for s in samples] == ["navsim_v1/navtest/aaa", "navsim_v1/navtest/bbb"]
    assert samples[0]["metrics"]["pdms"] == pytest.approx(0.9166667)
    assert samples[0]["metrics"]["comfort"] == 1.0 and samples[1]["metrics"]["comfort"] == 1.0
    assert samples[1]["metrics"]["time_to_collision_within_bound"] == 0.0
    assert "epdms" not in samples[0]["metrics"] and "traffic_light_compliance" not in samples[0]["metrics"]
    assert failed == ["ccc"]
    assert set(aggregates) == {"average"} and aggregates["average"]["score"] == pytest.approx(0.6041667)

    # The default is the v2 vocabulary, unchanged: `score` is epdms and the ids are navsim's.
    default, *_ = parse_scores(_v1_csv(tmp_path / "d.csv"))
    assert "epdms" in default[0]["metrics"] and "comfort" not in default[0]["metrics"]
    assert default[0]["sample_id"] == "navsim/navtest/aaa"


def test_merging_v1_chunks_drops_each_chunks_average_row(tmp_path: Path) -> None:
    for name in ("000", "001"):
        _v1_csv(tmp_path / f"{name}.csv")
    chunks = [tmp_path / "000.csv", tmp_path / "001.csv"]
    merged = merge_chunk_csvs(chunks, tmp_path / "m.csv", PDMS.aggregate_tokens)
    samples, aggregates, failed = parse_scores(merged, vocabulary=PDMS, benchmark="navsim_v1")
    assert len(samples) == 2 and aggregates == {} and failed == ["ccc"], "one token set, merged by token"
    assert not any(line.split(",")[1] == "average" for line in merged.read_text().splitlines()[1:])


def test_the_floor_is_one_scenario(tmp_path: Path, monkeypatch) -> None:
    assert limit_problems(1, floor=1) == [] and limit_problems(1) != []
    assert "two-frame" not in " ".join(limit_problems(0, floor=1))
    model = Registry.load().models["drivejepa"]
    assert _harness(_v1_site(tmp_path / "v1", monkeypatch), "navsim_v1").model_problems(model, 1) == []
    assert _harness(_navsim_site(tmp_path / "v2", monkeypatch)).model_problems(model, 1) != []


def test_agent_and_switch_reasons_partition_invalid_reasons(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_v1_site(tmp_path, monkeypatch), "navsim_v1")
    assert harness.invalid_reasons() == [] and harness.agent_reasons() == []
    assert harness.switch_reasons() == []
    monkeypatch.setenv(ENV_AGENT, "drivejepa")
    monkeypatch.setenv(ENV_ALLOW_SLOW, "1")
    assert harness.invalid_reasons() == harness.agent_reasons() + harness.switch_reasons()
    assert len(harness.agent_reasons()) == 1 and "native agent 'drivejepa'" in harness.agent_reasons()[0]
    assert len(harness.switch_reasons()) == 1 and ENV_ALLOW_SLOW in harness.switch_reasons()[0]


# -- scoring through the stand-in v1 scorer -----------------------------------------


def test_a_chunked_run_and_the_node_alive_gate(tmp_path: Path, monkeypatch) -> None:
    """Chunking, n_expected and the PDMS parse end to end; then a failed chunk
    with no node behind the agent continues to the next chunk, where the same
    failure behind the proxy hands the run back for a node restart."""
    harness = _harness(_stub_v1_site(tmp_path, monkeypatch, chunk_logs="1"), "navsim_v1")
    monkeypatch.setattr(v2, "node_alive", lambda url, timeout_s=5.0: True)
    config = {"cameras": ["CAM_FRONT"], "model": "m", "attempt": 1}
    evaluation = harness.evaluate("http://127.0.0.1:1", tmp_path / "out", config)
    assert len(evaluation.samples) == 6 and evaluation.n_expected == 6
    assert evaluation.failed == [] and evaluation.failed_chunks == []
    assert all({"pdms", "comfort"} <= set(s["metrics"]) for s in evaluation.samples)
    assert evaluation.samples[0]["sample_id"].startswith("navsim_v1/navtest/")
    assert "chunk 000: ok" in (tmp_path / "out" / "chunks.log").read_text()

    monkeypatch.setattr(v2, "node_alive", lambda url, timeout_s=5.0: False)
    monkeypatch.setenv("STUB_FAIL_LOG", "log_a")
    monkeypatch.setenv(ENV_AGENT, CONSTANT_VELOCITY_AGENT)
    nodeless_config = {"cameras": [], "model": "constant_velocity", "attempt": 1}
    evaluation = harness.evaluate("", tmp_path / "nodeless", nodeless_config)
    assert len(evaluation.samples) == 3 and [c["logs"] for c in evaluation.failed_chunks] == [["log_a"]]

    monkeypatch.delenv(ENV_AGENT)
    with pytest.raises(RuntimeError, match="not answering"):
        harness.evaluate("http://127.0.0.1:1", tmp_path / "proxy", config)


def test_the_retry_pass_finds_logs_through_the_cache_metadata(tmp_path: Path, monkeypatch) -> None:
    """v1's CSV has no log_name column; the cache's metadata csv says which
    log a token is in, so the retry loads one log's metadata instead of 136."""
    harness = _harness(_v1_site(tmp_path, monkeypatch, tokens_per_log=3), "navsim_v1")
    without = tmp_path / "scores.csv"
    without.write_text(",token,valid,score\n0,log_b-t1,False,\n1,log_a-t0,True,0.5\n")
    assert harness._logs_of(without, ["log_b-t1"]) == ["log_b"]
    assert harness._logs_of(without, ["log_b-t1", "log_a-t0"]) == ["log_a", "log_b"]
    assert harness._logs_of(without, ["unknown"]) == []
    with_column = tmp_path / "v2.csv"
    with_column.write_text("token,valid,score,log_name\nx,True,0.5,zzz\n")
    assert harness._logs_of(with_column, ["x"]) == ["zzz"], "a log_name column is still preferred"


def test_load_plans_indexes_by_token_and_keeps_the_later_line(tmp_path: Path) -> None:
    first = tmp_path / "chunks" / "000" / "trajectories.jsonl"
    first.parent.mkdir(parents=True)
    unlabelled = json.dumps({"navsim_poses": [[9, 9, 9]]})
    first.write_text(json.dumps({"token": "t1", "navsim_poses": [[1, 0, 0]]}) + "\n" + unlabelled + "\n")
    retry = tmp_path / "chunks" / "retry" / "trajectories.jsonl"
    retry.parent.mkdir(parents=True)
    later = json.dumps({"token": "t1", "navsim_poses": [[2, 0, 0]]})
    retry.write_text(later + "\n\n" + json.dumps({"token": "t2", "navsim_poses": [[3]]}) + "\n")
    plans = convert.load_plans(sorted(tmp_path.rglob("trajectories.jsonl")))
    assert plans == {"t1": [[2, 0, 0]], "t2": [[3]]}
    assert convert.load_plans([]) == {}


# -- node-less runs: baseline, human, replay ---------------------------------------


def _results_root(tmp_path: Path, monkeypatch) -> Path:
    """Store runs under tmp: `store.write_run` binds RESULTS_ROOT at definition."""
    root = tmp_path / "results"
    real = store.write_run

    def _write(run, metrics, samples=None, root_=None, artifacts_root=None):
        return real(run, metrics, samples, root=root, artifacts_root=artifacts_root)

    monkeypatch.setattr(store, "write_run", _write)
    monkeypatch.setattr(store, "RESULTS_ROOT", root)
    return root


def _only_run(root: Path, bench: str, model: str) -> tuple[dict, dict]:
    runs = sorted((root / bench / model).glob("*/run.json"))
    assert len(runs) == 1, runs
    return json.loads(runs[0].read_text()), json.loads((runs[0].parent / "metrics.json").read_text())


def test_the_constant_velocity_baseline_is_stored_valid(tmp_path: Path, monkeypatch) -> None:
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    lines: list[str] = []
    assert nodeless.run(cfg, "navsim_v1", agent="constant_velocity", echo=lines.append) == 0
    run, metrics = _only_run(root, "navsim_v1", "constant_velocity")
    assert run["valid"] is True and run["status"] == "complete" and run["model"] == "constant_velocity"
    assert run["config"]["agent"] == CONSTANT_VELOCITY_AGENT and run["config"]["node"] is None
    assert "reference baseline" in run["notes"] and run["dataset"]["n_samples"] == 6
    assert run["dataset"]["n_expected"] == 6 and run["dataset"]["split"] == "navtest"
    assert metrics["primary_metric"] == "pdms" and metrics["metrics"]["pdms"]["n"] == 6
    assert os.environ.get(ENV_AGENT) is None, "the switch is restored afterwards"
    assert any("pdms mean=" in line for line in lines)


def test_the_human_anchor_is_never_valid(tmp_path: Path, monkeypatch) -> None:
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    assert nodeless.run(cfg, "navsim_v1", agent="human", echo=lambda _: None) == 0
    run, _ = _only_run(root, "navsim_v1", "human")
    assert run["valid"] is False and nodeless.PRIVILEGED_HUMAN in run["notes"]
    assert run["config"]["agent"] == HUMAN_AGENT


def _source_run(
    cfg,
    root: Path,
    bench: str,
    model: str,
    run_id: str,
    tokens: list[str] | None,
    valid: bool = True,
    notes: str = "",
) -> None:
    record = root / bench / model / run_id
    record.mkdir(parents=True)
    run = {"run_id": run_id, "benchmark": bench, "model": model, "valid": valid, "notes": notes}
    (record / "run.json").write_text(json.dumps(run))
    metrics = {"primary_metric": "epdms", "metrics": {"epdms": {"mean": 0.6}}}
    (record / "metrics.json").write_text(json.dumps(metrics))
    dump = cfg.runs / bench / model / run_id / "chunks" / "000" / "trajectories.jsonl"
    dump.parent.mkdir(parents=True)
    poses = [[1.0, 0.0, 0.0]] * 8
    lines = [json.dumps({**({"token": t} if t else {}), "navsim_poses": poses}) for t in (tokens or [None])]
    dump.write_text("\n".join(lines) + "\n")


def test_a_replay_is_stored_under_the_source_model_with_its_provenance(tmp_path: Path, monkeypatch) -> None:
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    _source_run(cfg, root, "navsim", "alpamayo15", "src1", ["log_a-t0", "log_a-t1"])
    lines: list[str] = []
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/alpamayo15/src1", echo=lines.append) == 0
    run, metrics = _only_run(root, "navsim_v1", "alpamayo15")
    assert run["valid"] is True and run["benchmark"] == "navsim_v1"
    assert run["config"]["agent"] == REPLAY_AGENT
    assert run["config"]["replayed_from"] == "navsim/alpamayo15/src1" and run["config"]["plans"]["n"] == 2
    assert len(run["config"]["plans"]["sha256"]) == 16
    assert "replayed from navsim/alpamayo15/src1" in run["notes"] and "epdms 0.6000" in run["notes"]
    assert str(cfg.runs) not in json.dumps(run), "no machine path in a tracked record"
    assert metrics["primary_metric"] == "pdms"
    scorer_log = next((cfg.runs / "navsim_v1" / "alpamayo15").rglob("navsim_scorer.log")).read_text()
    assert f"agent={REPLAY_AGENT}" in scorer_log and "agent.plans_dir=" in scorer_log


def test_a_replay_follows_a_merged_run_back_to_its_earlier_plans(tmp_path: Path, monkeypatch) -> None:
    """A run that resumed from an earlier run's chunks keeps those plans in the
    earlier directory; the replay reads both, older first, and says so."""
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    _source_run(cfg, root, "navsim", "alpamayo15", "old1", ["log_a-t0", "log_b-t0"])
    (root / "navsim" / "alpamayo15" / "old1").rename(tmp_path / "old-record-gone")  # only artifacts remain
    _source_run(cfg, root, "navsim", "alpamayo15", "new1", ["log_b-t0"])
    record = root / "navsim" / "alpamayo15" / "new1" / "run.json"
    run = json.loads(record.read_text())
    run["config"] = {"merged_from": "old1"}
    record.write_text(json.dumps(run))
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/alpamayo15/new1", echo=lambda _: None) == 0
    stored, _ = _only_run(root, "navsim_v1", "alpamayo15")
    assert stored["config"]["plans"] == {
        "n": 2,
        "sha256": stored["config"]["plans"]["sha256"],
        "from": ["old1", "new1"],
    }
    scorer_log = next((cfg.runs / "navsim_v1" / "alpamayo15").rglob("navsim_scorer.log")).read_text()
    old_dir, new_dir = (
        cfg.runs / "navsim" / "alpamayo15" / "old1",
        cfg.runs / "navsim" / "alpamayo15" / "new1",
    )
    assert f"agent.plans_dir={old_dir}:{new_dir}" in scorer_log, "older first, so the scored plan wins"
    located = nodeless.locate_replay(cfg, "navsim/alpamayo15/new1")
    assert isinstance(located, dict) and located["sources"] == ["old1", "new1"]


def test_a_replay_inherits_the_source_runs_validity(tmp_path: Path, monkeypatch) -> None:
    """A source stored valid=false -- a subset, a rate override -- gives a replay
    that is valid=false for the same reason; the plans are that measurement's."""
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    why = "PADDOCK_NAVSIM_ALLOW_SLOW: a model below NAVSIM's input rate was run anyway"
    _source_run(cfg, root, "navsim", "openpilot", "slow1", ["log_a-t0"], valid=False, notes=why)
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/openpilot/slow1", echo=lambda _: None) == 0
    run, _ = _only_run(root, "navsim_v1", "openpilot")
    assert run["valid"] is False
    assert (
        "replayed from navsim/openpilot/slow1, which is stored valid=false" in run["notes"]
        and why in run["notes"]
    )


@pytest.mark.parametrize(
    ("model", "env", "replay"),
    [
        ("visionpilot", {"VISIONPILOT_ORT_THREADS": "2"}, True),  # what a CPU-limited site sets
        ("visionpilot", {"VISIONPILOT_SETPOINT": "run_start"}, True),  # changes a plan, but none is made
        ("constant_velocity", {"PADDOCK_CV_SPEED_M_S": "6"}, False),  # Bench2Drive's speed bands
    ],
)
def test_a_site_model_env_does_not_judge_a_run_no_model_process_made(
    tmp_path: Path, monkeypatch, model, env, replay
) -> None:
    """No node starts on this path, so a site's `[models.<name>.env]` reaches nothing.

    The runner's env rule would otherwise judge it against a stand-in spec with
    no tuning keys, and store a replay of a valid run, or paddock's own
    baseline, as valid=false. A replay's validity is its source's.
    """
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    body = "".join(f'{k} = "{v}"\n' for k, v in env.items())
    cfg.source.write_text(cfg.source.read_text() + f"\n[models.{model}.env]\n{body}")
    cfg = SiteConfig.load(cfg.source)
    root = _results_root(tmp_path, monkeypatch)
    if replay:
        _source_run(cfg, root, "navsim", model, "src1", ["log_a-t0"])
        assert nodeless.run(cfg, "navsim_v1", replay=f"navsim/{model}/src1", echo=lambda _: None) == 0
    else:
        assert nodeless.run(cfg, "navsim_v1", agent=model, echo=lambda _: None) == 0
    run, _ = _only_run(root, "navsim_v1", model)
    assert run["valid"] is True, run["notes"]
    assert "site env override" not in run["notes"]


def test_a_replay_refuses_what_it_cannot_vouch_for(tmp_path: Path, monkeypatch) -> None:
    cfg = _stub_v1_site(tmp_path, monkeypatch, chunk_logs="1")
    root = _results_root(tmp_path, monkeypatch)
    lines: list[str] = []
    echo = lines.append
    assert nodeless.run(cfg, "navsim_v1", replay="not/a/run", echo=echo) == 2
    assert "no recorded run" in lines[-1]
    assert nodeless.run(cfg, "navsim_v1", replay="bad", echo=echo) == 2
    _source_run(cfg, root, "navsim", "alpamayo15", "src1", ["log_a-t0"])
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/drivejepa/src1", echo=echo) == 2
    (root / "navsim" / "drivejepa" / "src1").mkdir(parents=True)
    wrong = {"benchmark": "navsim", "model": "alpamayo15", "valid": True}
    (root / "navsim" / "drivejepa" / "src1" / "run.json").write_text(json.dumps(wrong))
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/drivejepa/src1", echo=echo) == 2
    assert "names navsim/alpamayo15, not navsim/drivejepa" in lines[-1]
    _source_run(cfg, root, "navsim", "visionpilot", "old", None)  # a dump from before tokens were recorded
    assert nodeless.run(cfg, "navsim_v1", replay="navsim/visionpilot/old", echo=echo) == 2
    assert "no scenario tokens" in lines[-1]
    assert nodeless.run(cfg, "navsim_v1", agent="human", replay="navsim/alpamayo15/src1", echo=echo) == 2
    assert nodeless.run(cfg, "navsim_v1", echo=echo) == 2
    assert nodeless.run(cfg, "kitscenes", agent="human", echo=echo) == 2
    assert "not a NAVSIM benchmark" in lines[-1]
    assert not list(root.glob("navsim_v1/**/run.json")), "nothing was stored"


def test_a_limited_nodeless_run_is_a_pipeline_test(tmp_path: Path, monkeypatch) -> None:
    cfg = _stub_v1_site(tmp_path, monkeypatch)
    root = _results_root(tmp_path, monkeypatch)
    assert nodeless.run(cfg, "navsim_v1", agent="constant_velocity", limit=1, echo=lambda _: None) == 0
    run, _ = _only_run(root, "navsim_v1", "constant_velocity")
    assert run["valid"] is False and "--limit 1" in run["notes"] and run["config"]["limit"] == 1


# -- the real interpreter, when the site has one --------------------------------------


def test_the_agents_initialise_under_the_v1_devkit() -> None:
    """The base adapts to v1's AbstractAgent constructor: requires_scene comes
    from the class, _trajectory_sampling is set, and the v1 proxy asks for the
    four history frames by index. Run in the vendor interpreter with the v1
    devkit first on PYTHONPATH, exactly as the harness launches the scorer."""
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    settings = cfg.benchmarks.get("navsim_v1")
    if settings is None or not settings.get("python") or not settings.get("devkit"):
        pytest.skip("navsim_v1 not configured at this site")
    devkit, python = settings.get("devkit"), settings.get("python")
    if not Path(python).exists() or not (Path(devkit) / V1_SCORER).exists():
        pytest.skip("v1 devkit or interpreter not present")
    code = f"""
import navsim
assert navsim.__file__.startswith({devkit!r}), navsim.__file__
from paddock.adapters.benchmarks.navsim.agent import (
    PaddockConstantVelocityAgent, PaddockHumanAgent, PaddockProxyAgent, PaddockReplayAgent)
from paddock.adapters.benchmarks.navsim_v1.agent import PaddockV1ProxyAgent
proxy = PaddockProxyAgent(node_url="http://127.0.0.1:1")
assert proxy.requires_scene is False and proxy._trajectory_sampling.num_poses == 8
v1 = PaddockV1ProxyAgent(node_url="http://127.0.0.1:1", cameras=["CAM_FRONT", "CAM_BACK"])
assert v1.requires_scene is True and v1._trajectory_sampling.num_poses == 8
sensors = v1.get_sensor_config()
assert sensors.cam_f0 == [0, 1, 2, 3] and sensors.cam_b0 == [0, 1, 2, 3] and sensors.cam_l0 is False
assert PaddockHumanAgent().requires_scene is True and PaddockConstantVelocityAgent().requires_scene is False
print("ok")
"""
    env = {**os.environ, "PYTHONPATH": f"{devkit}:{REPO_ROOT / 'src'}", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
