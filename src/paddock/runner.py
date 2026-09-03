"""`paddock run` -- drive a model through a benchmark and store the result.

Paddock owns the model process; the benchmark owns its simulator. The ordering
here is the one the pre-paddock ladder driver arrived at, and each step exists
because skipping it cost a run:

1. **Preflight before anything loads.** A missing asset should be reported in
   seconds, not an hour into a sweep.
2. **One model resident at a time** (when `gpu.serialize`). The renderer alone
   wants ~8 GB and Alpamayo ~26; holding two model nodes at once is how a long
   job becomes a failed one.
3. **Every model sees identical episodes and seeds.** The comparison is paired,
   not merely averaged, which is worth far more and costs nothing here.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paddock.benchmark import ClosedLoopBenchmark, ExternallyScoredBenchmark, OpenLoopBenchmark
from paddock.config import SiteConfig
from paddock.policy.protocol import Trajectory
from paddock.registry import BenchmarkSpec, ModelSpec, Registry
from paddock.results import store
from paddock.runtime import ports, proc

#: Where port selection starts. Not assumed free -- on a shared machine 9000 is
#: somebody else's service as often as not, and `ports.find_free` scans past it.
#: Ports are still assigned one per model so a stale node from a previous attempt
#: is obvious rather than silently answering for the wrong model.
#:
#: Overridable per run because `find_free` only knows a port was free when it
#: looked. Two runs launched together scan the same range, and the one whose
#: node loads slowest loses: its port is taken by the time it binds. Give each
#: concurrent run its own base and they never meet.
DEFAULT_BASE_PORT = 9000


def base_port() -> int:
    """`PADDOCK_BASE_PORT`, validated: one port number, or the default.

    Read at run time rather than import time, and checked, so an empty or
    ranged value is reported by name before preflight instead of as a
    traceback from the top of the module.
    """
    raw = os.environ.get("PADDOCK_BASE_PORT", "").strip()
    if not raw:
        return DEFAULT_BASE_PORT
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        raise ValueError(f"PADDOCK_BASE_PORT must be a single port number in 1-65535, got {raw!r}")
    return int(raw)


def _measurement_config(cfg: SiteConfig, harness, model_spec: ModelSpec, **given) -> dict:
    """Everything that changes what was measured, for the run id and `run.json`.

    What the caller knows (`scene`, `runs`, `limit`), the model's cameras, the
    benchmark's own measurement-changing settings (`run_config()`), and any
    site `[models.<name>] env` block -- an environment override can change a
    model's input (alpamayo's frame selection, say), so it is on the record.
    """
    config = {**given, "cameras": list(model_spec.cameras), **harness.run_config()}
    override = cfg.models.get(model_spec.name)
    if override is not None and override.env:
        config["model_env"] = dict(override.env)
    return config


def _validity(failure_note: str | None, invalid_reasons: list[str]) -> tuple[str, bool, str]:
    """`(status, valid, notes)` for a run record. The one place that decides `valid`.

    A run is valid only when nothing failed *and* nothing about how it was made
    keeps it from being ranked: no subset, no override that changes what the
    number means. The reasons go into `notes`, so a `valid: false` row explains
    itself and nobody has to hand-edit a results file to say so.
    """
    notes = [*invalid_reasons]
    if failure_note:
        notes.append(failure_note)
    status = "partial" if failure_note else "complete"
    return status, not failure_note and not invalid_reasons, "; ".join(notes)


def _store_run(
    cfg: SiteConfig,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    run_id: str,
    started: datetime,
    began: float,
    samples: list[dict],
    config: dict,
    failure_note: str | None,
    invalid_reasons: list[str],
    by_group: dict | None = None,
    hardware: dict | None = None,
    split: str | None = None,
    extra_dataset: dict | None = None,
) -> tuple[Path, store.MetricsRecord]:
    """Write one run to the results store. Every path stores through here.

    `split` is already in every sample id and therefore in the sequence hash;
    recorded in `dataset` too so run.json answers "which split" without
    opening samples.jsonl.
    """
    status, valid, notes = _validity(failure_note, invalid_reasons)
    dataset: dict = {"name": spec.dataset}
    if split is not None:
        dataset["split"] = split
    dataset.update(
        {
            "n_samples": len(samples),
            "sequences_sha256": store.sequence_hash([s["sample_id"] for s in samples]),
        }
    )
    if extra_dataset:
        dataset.update(extra_dataset)
    run = store.RunRecord(
        run_id=run_id,
        site=cfg.site,
        benchmark=spec.name,
        model=model_spec.name,
        mode=spec.mode,
        started_utc=started.isoformat(timespec="seconds"),
        duration_s=round(time.time() - began, 1),
        dataset=dataset,
        config=config,
        hardware=hardware or {},
        status=status,
        valid=valid,
        notes=notes,
    )
    metrics_record = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    if by_group:
        metrics_record.by_group = dict(by_group)
    written = store.write_run(run, metrics_record, samples, artifacts_root=cfg.runs)
    return written, metrics_record


def _external_failure_note(evaluation) -> str | None:
    """What an external run failed to score, in the units it was lost in.

    A scenario the model failed on is one scenario; a chunk the scorer never
    completed is hundreds. "4 scenario(s) failed" for a thousand unscored ones
    misdescribed a run by 250x once, so the two are counted apart.
    """
    parts = []
    if evaluation.failed:
        parts.append(f"{len(evaluation.failed)} scenario(s) failed in the scorer and are excluded")
    if evaluation.failed_chunks:
        logs = sum(len(chunk.get("logs") or []) for chunk in evaluation.failed_chunks)
        parts.append(
            f"{len(evaluation.failed_chunks)} chunk(s) never scored: {evaluation.unscored} scenario(s) "
            f"across {logs} log(s)"
        )
    return "; ".join(parts) or None


def _config_marker_problems(artifacts: Path, config: dict) -> list[str]:
    """Refuse to resume a run under a different measurement configuration.

    `run.json` is written only at the end, so the configuration a run started
    under is recorded in its artifacts directory at the start; `--resume`
    compares against it, and a changed agent, switch or chunk size is a
    refusal rather than two measurements merged under one id. A run from
    before the marker existed has nothing to compare against; it is written
    now and said so.
    """
    marker = artifacts / "run_config.json"
    if marker.exists():
        recorded = json.loads(marker.read_text())
        differing = sorted(k for k in set(recorded) | set(config) if recorded.get(k) != config.get(k))
        if differing:
            return [
                f"--resume refused: {marker} was written under a different configuration; "
                f"differs in {', '.join(differing)}. Start a new run, or restore the switches "
                f"it was made under."
            ]
        return []
    artifacts.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return []


def _limit_reason(limit: int | None) -> list[str]:
    return [f"--limit {limit}: a subset of the benchmark, run to test the pipeline"] if limit else []


def load_benchmark(spec: BenchmarkSpec, cfg: SiteConfig):
    module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.harness")
    return module.BENCHMARK(spec, cfg)


def run_benchmark(
    cfg: SiteConfig,
    bench: str,
    model: str,
    scene: str | None = None,
    scenarios: list[str] | None = None,
    runs: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    resume: str | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    registry = Registry.load()

    spec = registry.benchmarks.get(bench)
    if spec is None:
        echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
        return 2

    if resume and model == "all":
        echo("--resume names one run, so it needs one --model")
        return 2
    if model == "all":
        models = [m for m in registry.models.values() if bench in m.benchmarks]
        if not models:
            echo(f"no models declare benchmark {bench!r}")
            return 2
    else:
        chosen = registry.models.get(model)
        if chosen is None:
            echo(f"unknown model {model!r}; known: {', '.join(sorted(registry.models)) or 'none'}")
            return 2
        models = [chosen]

    harness = load_benchmark(spec, cfg)

    try:
        base = base_port()
    except ValueError as exc:
        echo(str(exc))
        return 2

    # `--scene` is comma-separated like `--scenarios`; None means the benchmark's
    # own default (neuroNCAP: 0103; Bench2Drive: every route). No benchmark's
    # default id is passed to another benchmark.
    scenes = [s.strip() for s in scene.split(",") if s.strip()] if scene else None

    problems = harness.preflight()
    if scenes and hasattr(harness, "scene_problems"):
        for one in scenes:
            problems += harness.scene_problems(one)
    if scenarios and hasattr(harness, "scenario_problems"):
        problems += harness.scenario_problems(scenarios)
    if problems:
        echo(f"{bench} cannot run here:")
        for problem in problems:
            echo("  - " + problem.replace("\n", "\n    "))
        return 1

    # Scene and run count are closed-loop concepts; printing them for an
    # open-loop benchmark reads as though a sweep were scoped to one scene.
    if isinstance(harness, ClosedLoopBenchmark):
        echo(
            f"{bench}: {len(models)} model(s), scene {scene or '(benchmark default)'}, "
            f"{runs} run(s) per scenario"
        )
    else:
        echo(f"{bench}: {len(models)} model(s)" + (f", limit {limit}" if limit else ""))
    if dry_run:
        if isinstance(harness, ExternallyScoredBenchmark):
            # The scorer decides what it can score, from the intersection of the
            # logs on disk and the metric cache; preflight has already refused a
            # partial split. What can be said cheaply is whether each model can
            # be scored at all, and whether the result could be ranked.
            for m in models:
                problems = harness.model_problems(m, limit)
                echo(f"  {m.name}: " + ("; ".join(problems) if problems else "can be scored"))
            reasons = [*harness.invalid_reasons(), *_limit_reason(limit)]
            if reasons:
                echo("  would be stored valid=false: " + "; ".join(reasons))
            echo("  the scorer reports its scenario count at run time")
        elif isinstance(harness, OpenLoopBenchmark):
            found, missing = harness.coverage() if hasattr(harness, "coverage") else ([], [])
            echo(f"  {len(found)} sample(s) available" + (f", {len(missing)} missing" if missing else ""))
            if missing:
                echo(f"    missing scenes: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
        else:
            episodes = list(harness.episodes(scenes, runs=runs, scenarios=scenarios))
            for m in models:
                echo(f"  {m.name}: {', '.join(e.episode_id for e in episodes)}")
        echo("dry run: nothing started")
        return 0

    if isinstance(harness, ExternallyScoredBenchmark):
        failures = 0
        for index, model_spec in enumerate(models):
            port = ports.find_free(base + index)
            failures += _run_external(cfg, harness, spec, model_spec, port, limit, echo, resume=resume)
        return 1 if failures else 0

    if isinstance(harness, OpenLoopBenchmark):
        failures = 0
        for model_spec in models:
            failures += _run_open_loop(cfg, harness, spec, model_spec, base, limit, echo)
        return 1 if failures else 0

    failures = 0
    for index, model_spec in enumerate(models):
        port = ports.find_free(base + index)
        failures += _run_one_model(cfg, harness, spec, model_spec, scenes, scenarios, runs, port, echo)
    return 1 if failures else 0


def _run_one_model(
    cfg: SiteConfig,
    harness: ClosedLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    scenes: list[str] | None,
    scenarios: list[str] | None,
    runs: int,
    port: int,
    echo: Callable[[str], None],
) -> int:
    started = datetime.now(UTC)
    began = time.time()
    # Everything that changes what was measured. Scenarios were missing once,
    # so two runs over different scenario sets hashed identically. A benchmark
    # adds its own measurement-changing site settings through `run_config()`
    # (a routes subset, say), so two sites' numbers stay distinguishable.
    config = _measurement_config(
        cfg, harness, model_spec, scene=",".join(scenes) if scenes else None, runs=runs
    )
    run_id = store.make_run_id(
        started,
        {
            **config,
            "model": model_spec.name,
            "scenarios": sorted(scenarios) if scenarios else "all",
        },
    )
    artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===")
    samples: list[dict] = []
    failures = 0

    try:
        with proc.model_node(model_spec, cfg, port, artifacts, echo=echo, nonce=run_id) as policy_url:
            # Materialised so run.json can say how many episodes were *meant* to
            # score: n_samples alone cannot tell 220 scored from 6 scored and
            # 214 failed, and a mean over the six reads like a full sweep.
            episodes = list(harness.episodes(scenes, runs=runs, scenarios=scenarios))
            for episode in episodes:
                episode.config["cameras"] = list(model_spec.cameras)
                out_dir = artifacts / episode.episode_id
                try:
                    metrics = harness.run_episode(episode, policy_url, out_dir)
                except RuntimeError as exc:
                    echo(f"  {episode.episode_id}: FAILED - {exc}")
                    failures += 1
                    continue
                echo(
                    f"  {episode.episode_id}: "
                    + ", ".join(f"{k} {v:.4g}" for k, v in sorted(metrics.items()))
                )
                samples.append(
                    {
                        "sample_id": f"{spec.name}/{episode.scene}/{episode.scenario}",
                        "native_id": {"scenario": episode.scenario, "scene": episode.scene},
                        "split": episode.scenario,
                        "metrics": {k: float(v) for k, v in metrics.items() if k not in ("n_runs",)},
                    }
                )
    except (proc.NodeStartupError, RuntimeError) as exc:
        echo(f"  node failed: {exc}")
        return 1

    if not samples:
        echo("  no episodes completed; nothing recorded")
        return 1

    written, _ = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note=f"{failures} episode(s) failed" if failures else None,
        invalid_reasons=harness.invalid_reasons(),
        hardware={"gpu_free_gb_at_start": None},
        extra_dataset={"n_expected": len(episodes), "n_failed": failures},
    )
    echo(f"  -> {written.relative_to(Path.cwd()) if written.is_relative_to(Path.cwd()) else written}")
    return failures


def _run_external(
    cfg: SiteConfig,
    harness: ExternallyScoredBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    port: int,
    limit: int | None,
    echo: Callable[[str], None],
    resume: str | None = None,
) -> int:
    """Score one model with a benchmark that brings its own scorer.

    Same lifecycle as everywhere else -- paddock starts the node, the benchmark
    does its work against it, paddock stores the result -- except that the
    per-sample records come back from the benchmark in one batch rather than
    being accumulated a sample at a time, because the scorer owned the loop.

    A run this long must survive what happens during it. If the scorer or a
    node dies, the nodes are torn down and restarted and `evaluate` is
    re-invoked on the same artifacts, up to `harness.run_retries` times; a
    benchmark that scores in resumable chunks then continues from the last
    one that completed. `resume` names an earlier run id to continue in place
    after paddock itself died -- same directory, same run id, nothing scored
    twice.
    """
    started = datetime.now(UTC)
    began = time.time()
    config = _measurement_config(cfg, harness, model_spec, limit=limit)
    if resume:
        run_id = resume
        artifacts = cfg.runs / spec.name / model_spec.name / run_id
        if not artifacts.is_dir():
            echo(f"  nothing to resume: {artifacts} does not exist")
            return 2
    else:
        run_id = store.make_run_id(started, {**config, "bench": spec.name, "model": model_spec.name})
        artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===" + (f" (resuming {run_id})" if resume else ""))
    for problem in _config_marker_problems(artifacts, config):
        echo("  " + problem)
        return 2
    # Pure in the model spec and the limit, so asked before any VRAM is waited
    # for or checkpoint read: `--model openpilot` used to load supercombo in
    # full to be told NAVSIM's input rate is too low for it.
    problems = harness.model_problems(model_spec, limit)
    if problems:
        echo(f"  {model_spec.name} cannot be scored here:")
        for problem in problems:
            echo("  - " + problem.replace("\n", "\n    "))
        return 1
    evaluation = None
    attempts = 1 + max(0, int(getattr(harness, "run_retries", 0)))
    for attempt in range(1, attempts + 1):
        try:
            with contextlib.ExitStack() as stack:
                # Started one after another, each waiting for /alive before the
                # next port is chosen, so `find_free` never hands out a port a
                # sibling is about to bind. A retry re-enters here with fresh
                # nodes: a node that died is replaced, not mourned.
                urls: list[str] = []
                next_port = port
                for _ in range(max(1, harness.node_count(model_spec))):
                    next_port = ports.find_free(next_port)
                    urls.append(
                        stack.enter_context(
                            proc.model_node(model_spec, cfg, next_port, artifacts, echo=echo, nonce=run_id)
                        )
                    )
                    next_port += 1
                evaluation = harness.evaluate(
                    urls[0],
                    artifacts,
                    {
                        "cameras": list(model_spec.cameras),
                        "model": model_spec.name,
                        "min_input_hz": model_spec.min_input_hz,
                        "policy_urls": urls,
                        "attempt": attempt,
                    },
                    limit=limit,
                    echo=echo,
                )
        except (proc.NodeStartupError, RuntimeError) as exc:
            if attempt < attempts:
                why = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                echo(f"  attempt {attempt}/{attempts} failed: {why}; restarting nodes and resuming")
                continue
            echo(f"  failed after {attempts} attempt(s): {exc}")
            return 1
        # Whole chunks the scorer could not run are worth another attempt with
        # fresh nodes; a scenario the model itself failed on twice is not.
        if evaluation.failed_chunks and attempt < attempts:
            echo(
                f"  attempt {attempt}/{attempts}: {len(evaluation.failed_chunks)} chunk(s) unfinished "
                f"({evaluation.unscored} scenario(s)); restarting"
            )
            continue
        break

    samples, failures = evaluation.samples, evaluation.failed
    if not samples:
        echo("  no scenarios scored; nothing recorded")
        return 1

    # Same rule as the closed-loop path: a scenario the scorer could not run
    # is excluded from the metrics and the run is marked, never averaged. A
    # zero from a timed-out request is indistinguishable from a model that
    # drove into a wall. And a subset or an override that changes what the
    # number means is stored, but never valid.
    invalid_reasons = [*harness.invalid_reasons(), *_limit_reason(limit)]
    failure_note = _external_failure_note(evaluation)
    extra_dataset = {"n_failed": len(failures), "n_unscored": evaluation.unscored}
    if evaluation.n_expected is not None:
        extra_dataset["n_expected"] = evaluation.n_expected
    _, metrics_record = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note=failure_note,
        invalid_reasons=invalid_reasons,
        # The scorer's own headline and aggregate rows, kept in the tracked
        # metrics.json exactly where the ingest path puts them.
        by_group=evaluation.by_group,
        split=harness.split,
        extra_dataset=extra_dataset,
    )
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}")
    if failure_note:
        echo(f"  {failure_note}; run recorded as partial, valid=false")
    for reason in invalid_reasons:
        echo(f"  recorded valid=false: {reason}")
    return 1 if failure_note else 0


def _run_open_loop(
    cfg: SiteConfig,
    harness: OpenLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    base: int,
    limit: int | None,
    echo: Callable[[str], None],
) -> int:
    """Score one model over a benchmark's samples, in its own interpreter.

    Open loop needs no simulator, but it still needs the model's environment --
    so the work runs as a subprocess through the same node machinery, driven
    over HTTP. Keeping one execution path for both loop types is the point: a
    second in-process path would be where the two quietly diverge.
    """
    import httpx

    started = datetime.now(UTC)
    began = time.time()
    config = _measurement_config(cfg, harness, model_spec, limit=limit)
    run_id = store.make_run_id(started, {**config, "bench": spec.name, "model": model_spec.name})
    artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===")
    samples: list[dict] = []
    failures = 0

    try:
        with proc.model_node(
            model_spec, cfg, ports.find_free(base), artifacts, echo=echo, nonce=run_id
        ) as policy_url:
            for index, sample in enumerate(harness.samples(limit=limit)):
                try:
                    payload = {
                        "observations": [o.to_wire() for o in sample.observations],
                    }
                    response = httpx.post(f"{policy_url}/infer_sequence", json=payload, timeout=600.0)
                    response.raise_for_status()
                    prediction = Trajectory(response.json()["trajectory"], dt=model_spec.waypoint_dt)
                    metrics = harness.score(sample, prediction)
                except Exception as exc:  # noqa: BLE001 - one bad sample must not end the run
                    echo(f"  {sample.sample_id}: FAILED - {exc}")
                    failures += 1
                    continue

                samples.append(
                    {
                        "sample_id": sample.sample_id,
                        "native_id": sample.native_id,
                        "split": sample.split,
                        "metrics": metrics,
                        # Kept so a result can be looked at, not just ranked.
                        # These live outside the repo with the rest of the run
                        # artifacts, so the cost is disk rather than git.
                        "prediction": prediction.xy.tolist(),
                        "ground_truth": np.asarray(sample.ground_truth)[:, :2].tolist()
                        if sample.ground_truth is not None
                        else None,
                    }
                )
                if (index + 1) % 25 == 0:
                    running = sum(s["metrics"].get("ade", 0.0) for s in samples) / len(samples)
                    echo(f"  {index + 1} samples, running ADE {running:.3f}")
    except (proc.NodeStartupError, RuntimeError) as exc:
        echo(f"  node failed: {exc}")
        return 1

    if not samples:
        echo("  no samples scored; nothing recorded")
        return 1

    _, metrics_record = _store_run(
        cfg,
        spec,
        model_spec,
        run_id,
        started,
        began,
        samples,
        config,
        failure_note=f"{failures} sample(s) failed" if failures else None,
        invalid_reasons=[*harness.invalid_reasons(), *_limit_reason(limit)],
    )
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(
        f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}"
        f" median={headline.get('median')}"
    )
    return failures
