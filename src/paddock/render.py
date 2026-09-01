"""`paddock render` -- compare several models on one benchmark.

The debugging move this exists for: take the runs you have, from whichever
models, and look at them together. So rendering is scoped to the **benchmark**,
not the model. A benchmark knows what its ground truth is and what a comparison
means; a model does not, and cannot without being told about its rivals.

Each benchmark adapter may supply a `render.py` exposing:

    def render(runs: list[LoadedRun], out_dir: Path, cfg) -> list[Path]

returning what it wrote. Adapters without one are simply not renderable, which
is a fine state to be in.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paddock.config import SiteConfig
from paddock.registry import Registry
from paddock.results.store import RESULTS_ROOT


def _started_at(run_dir: Path) -> str:
    """When a run started, for ordering. Ingested runs have no timestamp."""
    try:
        return json.loads((run_dir / "run.json").read_text()).get("started_utc") or ""
    except (OSError, ValueError):
        return ""


@dataclass
class LoadedRun:
    """One run's records, ready to draw.

    Metrics come from the tracked store; per-sample records from the run
    artifacts outside the repo, which may be absent if they were cleaned up.
    """

    model: str
    benchmark: str
    run_id: str
    run: dict[str, Any]
    metrics: dict[str, Any]
    samples: list[dict[str, Any]]

    @property
    def label(self) -> str:
        return self.model


def load_runs(
    cfg: SiteConfig,
    benchmark: str,
    models: list[str] | None = None,
    run_ids: list[str] | None = None,
) -> list[LoadedRun]:
    """The most recent valid run per model, or specific runs by id."""
    loaded: list[LoadedRun] = []
    bench_root = RESULTS_ROOT / benchmark
    if not bench_root.is_dir():
        return loaded

    for model_dir in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        if models and model_dir.name not in models:
            continue
        # Newest by when it actually ran. Sorting run-id strings puts "legacy-*"
        # above "2026...", which quietly selected ingested historical runs over
        # ones from this afternoon.
        candidates = sorted(
            (p for p in model_dir.iterdir() if p.is_dir()),
            key=_started_at,
            reverse=True,
        )
        for run_dir in candidates:
            if run_ids and run_dir.name not in run_ids:
                continue
            run = json.loads((run_dir / "run.json").read_text())
            if not run.get("valid", True) and not run_ids:
                continue  # never quietly draw a run marked invalid
            metrics = json.loads((run_dir / "metrics.json").read_text())
            samples_path = cfg.runs / benchmark / model_dir.name / run_dir.name / "samples.jsonl"
            samples = []
            if samples_path.exists():
                samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
            loaded.append(
                LoadedRun(
                    model=model_dir.name,
                    benchmark=benchmark,
                    run_id=run_dir.name,
                    run=run,
                    metrics=metrics,
                    samples=samples,
                )
            )
            if not run_ids:
                break  # newest valid run only
    return loaded


def render(
    cfg: SiteConfig,
    benchmark: str,
    models: list[str] | None = None,
    run_ids: list[str] | None = None,
    out_dir: Path | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    registry = Registry.load()
    spec = registry.benchmarks.get(benchmark)
    if spec is None:
        echo(f"unknown benchmark {benchmark!r}; known: {', '.join(sorted(registry.benchmarks))}")
        return 2

    try:
        module = importlib.import_module(f"paddock.adapters.benchmarks.{spec.directory.name}.render")
    except ModuleNotFoundError:
        echo(
            f"{benchmark} has no render.py. Adding one is optional -- see "
            f"docs/adding-a-benchmark.md for the contract."
        )
        return 2

    runs = load_runs(cfg, benchmark, models, run_ids)
    if not runs:
        echo(f"no valid runs found for {benchmark}. Run one first, or pass --run-id.")
        return 1

    echo(f"{benchmark}: {len(runs)} run(s) - " + ", ".join(f"{r.model}/{r.run_id}" for r in runs))

    # Runs over different sample sets are not comparable, however similar the
    # counts look. Say so rather than drawing a figure that implies they are.
    fingerprints = {r.run.get("dataset", {}).get("sequences_sha256") for r in runs}
    if len(fingerprints) > 1:
        echo("")
        echo("  WARNING: these runs cover different sample sets, so the comparison")
        echo("  is not like for like:")
        for r in runs:
            data = r.run.get("dataset", {})
            echo(f"    {r.model:14} n={data.get('n_samples')}  {data.get('sequences_sha256')}")
        echo("  Re-run them over the same samples, or pass --run-id to pick matching ones.")
        echo("")
    destination = out_dir or (cfg.runs / benchmark / "_compare")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        written = module.render(runs, destination, cfg)
    except ImportError as exc:
        echo(
            f"rendering needs the 'viz' extra, which is not installed here: {exc}\n"
            f"  uv pip install -e '.[viz]'    (or re-run 'make setup')"
        )
        return 2
    for path in written:
        echo(f"  wrote {path}")
    return 0
