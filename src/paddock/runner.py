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

import importlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from paddock.benchmark import ClosedLoopBenchmark
from paddock.config import SiteConfig
from paddock.registry import BenchmarkSpec, ModelSpec, Registry
from paddock.results import store
from paddock.runtime import proc

#: Ports are assigned per model so a stale node from a previous attempt is
#: obvious rather than silently answering for the wrong model.
BASE_PORT = 9000


def load_benchmark(spec: BenchmarkSpec, cfg: SiteConfig):
    module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.harness")
    return module.BENCHMARK(spec, cfg)


def run_benchmark(
    cfg: SiteConfig,
    bench: str,
    model: str,
    scene: str = "0103",
    scenarios: list[str] | None = None,
    runs: int = 1,
    limit: int | None = None,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
) -> int:
    registry = Registry.load()

    spec = registry.benchmarks.get(bench)
    if spec is None:
        echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
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

    problems = harness.preflight()
    if hasattr(harness, "scene_problems"):
        problems += harness.scene_problems(scene)
    if problems:
        echo(f"{bench} cannot run here:")
        for problem in problems:
            echo("  - " + problem.replace("\n", "\n    "))
        return 1

    echo(f"{bench}: {len(models)} model(s), scene {scene}, {runs} run(s) per scenario")
    if dry_run:
        episodes = list(harness.episodes([scene], runs=runs, scenarios=scenarios))
        for m in models:
            echo(f"  {m.name}: {', '.join(e.episode_id for e in episodes)}")
        echo("dry run: nothing started")
        return 0

    if not isinstance(harness, ClosedLoopBenchmark):
        echo(f"open-loop benchmarks are not wired into `paddock run` yet ({bench} is {spec.mode})")
        return 2

    failures = 0
    for index, model_spec in enumerate(models):
        failures += _run_one_model(
            cfg, harness, spec, model_spec, scene, scenarios, runs, BASE_PORT + index, echo
        )
    return 1 if failures else 0


def _run_one_model(
    cfg: SiteConfig,
    harness: ClosedLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    scene: str,
    scenarios: list[str] | None,
    runs: int,
    port: int,
    echo: Callable[[str], None],
) -> int:
    started = datetime.now(UTC)
    began = time.time()
    run_id = store.make_run_id(
        started,
        # Everything that changes what was measured. Scenarios were missing,
        # so two runs over different scenario sets hashed identically.
        {
            "scene": scene,
            "runs": runs,
            "model": model_spec.name,
            "scenarios": sorted(scenarios) if scenarios else "all",
            "cameras": list(model_spec.cameras),
        },
    )
    artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===")
    samples: list[dict] = []
    failures = 0

    try:
        with proc.model_node(model_spec, cfg, port, artifacts, echo=echo) as policy_url:
            for episode in harness.episodes([scene], runs=runs, scenarios=scenarios):
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

    run = store.RunRecord(
        run_id=run_id,
        site=cfg.site,
        benchmark=spec.name,
        model=model_spec.name,
        mode=spec.mode,
        started_utc=started.isoformat(timespec="seconds"),
        duration_s=round(time.time() - began, 1),
        dataset={
            "name": spec.dataset,
            "n_samples": len(samples),
            "sequences_sha256": store.sequence_hash([s["sample_id"] for s in samples]),
        },
        config={"scene": scene, "runs": runs, "cameras": list(model_spec.cameras)},
        hardware={"gpu_free_gb_at_start": None},
        status="complete" if not failures else "partial",
        valid=failures == 0,
        notes="" if not failures else f"{failures} episode(s) failed",
    )
    metrics_record = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    written = store.write_run(run, metrics_record, samples, artifacts_root=cfg.runs)
    echo(f"  -> {written.relative_to(Path.cwd()) if written.is_relative_to(Path.cwd()) else written}")
    return failures
