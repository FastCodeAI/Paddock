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
BASE_PORT = int(os.environ.get("PADDOCK_BASE_PORT", "9000"))


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

    # Scene and run count are closed-loop concepts; printing them for an
    # open-loop benchmark reads as though a sweep were scoped to one scene.
    if isinstance(harness, ClosedLoopBenchmark):
        echo(f"{bench}: {len(models)} model(s), scene {scene}, {runs} run(s) per scenario")
    else:
        echo(f"{bench}: {len(models)} model(s)" + (f", limit {limit}" if limit else ""))
    if dry_run:
        if isinstance(harness, ExternallyScoredBenchmark):
            # The scorer decides what it can score, from the intersection of the
            # logs on disk and the metric cache. Preflight has already refused a
            # partial split, so there is nothing left to enumerate cheaply here.
            echo("  preflight clean; the scorer reports its scenario count at run time")
        elif isinstance(harness, OpenLoopBenchmark):
            found, missing = harness.coverage() if hasattr(harness, "coverage") else ([], [])
            echo(f"  {len(found)} sample(s) available" + (f", {len(missing)} missing" if missing else ""))
            if missing:
                echo(f"    missing scenes: {', '.join(missing[:5])}" + (" ..." if len(missing) > 5 else ""))
        else:
            episodes = list(harness.episodes([scene], runs=runs, scenarios=scenarios))
            for m in models:
                echo(f"  {m.name}: {', '.join(e.episode_id for e in episodes)}")
        echo("dry run: nothing started")
        return 0

    if isinstance(harness, ExternallyScoredBenchmark):
        failures = 0
        for index, model_spec in enumerate(models):
            port = ports.find_free(BASE_PORT + index)
            failures += _run_external(cfg, harness, spec, model_spec, port, limit, echo)
        return 1 if failures else 0

    if isinstance(harness, OpenLoopBenchmark):
        failures = 0
        for model_spec in models:
            failures += _run_open_loop(cfg, harness, spec, model_spec, limit, echo)
        return 1 if failures else 0

    failures = 0
    for index, model_spec in enumerate(models):
        port = ports.find_free(BASE_PORT + index)
        failures += _run_one_model(cfg, harness, spec, model_spec, scene, scenarios, runs, port, echo)
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


def _run_external(
    cfg: SiteConfig,
    harness: ExternallyScoredBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
    port: int,
    limit: int | None,
    echo: Callable[[str], None],
) -> int:
    """Score one model with a benchmark that brings its own scorer.

    Same lifecycle as everywhere else -- paddock starts the node, the benchmark
    does its work against it, paddock stores the result -- except that the
    per-sample records come back from the benchmark in one batch rather than
    being accumulated a sample at a time, because the scorer owned the loop.
    """
    started = datetime.now(UTC)
    began = time.time()
    run_id = store.make_run_id(
        started,
        {"bench": spec.name, "model": model_spec.name, "limit": limit, "cameras": list(model_spec.cameras)},
    )
    artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===")
    try:
        with contextlib.ExitStack() as stack:
            # Started one after another, each waiting for /alive before the
            # next port is chosen, so `find_free` never hands out a port a
            # sibling is about to bind.
            urls: list[str] = []
            next_port = port
            for _ in range(max(1, harness.node_count(model_spec))):
                next_port = ports.find_free(next_port)
                urls.append(
                    stack.enter_context(proc.model_node(model_spec, cfg, next_port, artifacts, echo=echo))
                )
                next_port += 1
            samples, failures = harness.evaluate(
                urls[0],
                artifacts,
                {
                    "cameras": list(model_spec.cameras),
                    "model": model_spec.name,
                    "min_input_hz": model_spec.min_input_hz,
                    "policy_urls": urls,
                },
                limit=limit,
                echo=echo,
            )
    except (proc.NodeStartupError, RuntimeError) as exc:
        echo(f"  failed: {exc}")
        return 1

    if not samples:
        echo("  no scenarios scored; nothing recorded")
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
        config={"limit": limit, "cameras": list(model_spec.cameras)},
        # Same rule as the closed-loop path: a scenario the scorer could not
        # run is excluded from the metrics and the run is marked, never
        # averaged. A zero from a timed-out request is indistinguishable from
        # a model that drove into a wall.
        status="complete" if not failures else "partial",
        valid=not failures,
        notes="" if not failures else f"{len(failures)} scenario(s) failed in the scorer and are excluded",
    )
    metrics_record = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    store.write_run(run, metrics_record, samples, artifacts_root=cfg.runs)
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}")
    if failures:
        echo(f"  {len(failures)} scenario(s) FAILED in the scorer; run recorded as partial, valid=false")
    return 1 if failures else 0


def _run_open_loop(
    cfg: SiteConfig,
    harness: OpenLoopBenchmark,
    spec: BenchmarkSpec,
    model_spec: ModelSpec,
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
    run_id = store.make_run_id(started, {"bench": spec.name, "model": model_spec.name, "limit": limit})
    artifacts = cfg.runs / spec.name / model_spec.name / run_id

    echo(f"\n=== {model_spec.name} ===")
    samples: list[dict] = []
    failures = 0

    try:
        with proc.model_node(model_spec, cfg, ports.find_free(BASE_PORT), artifacts, echo=echo) as policy_url:
            for index, sample in enumerate(harness.samples(limit=limit)):
                try:
                    payload = {
                        "observations": [_encode(o) for o in sample.observations],
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
        config={"limit": limit},
        status="complete" if not failures else "partial",
        valid=failures == 0,
        notes="" if not failures else f"{failures} sample(s) failed",
    )
    metrics_record = store.MetricsRecord.from_samples(run_id, samples, primary_metric=spec.primary_metric)
    store.write_run(run, metrics_record, samples, artifacts_root=cfg.runs)
    headline = metrics_record.metrics.get(spec.primary_metric, {})
    echo(
        f"  n={metrics_record.n}  {spec.primary_metric} mean={headline.get('mean')}"
        f" median={headline.get('median')}"
    )
    return failures


def _encode(obs) -> dict:
    """Observation -> the wire shape the node decodes."""
    from paddock.policy.protocol import encode_image

    # Paths where the frame is on a shared filesystem, pixels only where it is
    # not. A 223-sample KITScenes run would otherwise push 348 GB of base64.
    return {
        "images": {k: encode_image(v) for k, v in obs.images.items()},
        "image_paths": dict(obs.image_paths),
        "ego2world": obs.ego2world.tolist(),
        "canbus": obs.canbus.tolist(),
        "timestamp": obs.timestamp_us,
        "command": int(obs.command),
        "calibration": {
            "camera2image": {k: v.tolist() for k, v in obs.camera2image.items()},
            "camera2ego": {k: v.tolist() for k, v in obs.camera2ego.items()},
        },
    }
