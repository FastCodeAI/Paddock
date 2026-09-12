"""`paddock` -- the command line.

Commands that are not implemented yet say so and name the milestone, rather than
succeeding quietly with nothing done.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from paddock import doctor as doctor_mod
from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate driving world models across open- and closed-loop benchmarks.",
)


@app.command()
def doctor(
    model: str = typer.Option(None, "--model", "-m", help="Check only this model's environment."),
) -> None:
    """Report what this machine can run, and what to fix.

    Safe to run at any time; changes nothing.
    """
    report = doctor_mod.run(model_filter=model)
    typer.echo(report.render())
    raise typer.Exit(code=1 if report.failed else 0)


@app.command("list")
def list_(
    what: str = typer.Argument("all", help="all | models | benchmarks | datasets | repos"),
) -> None:
    """List what is registered."""
    registry = Registry.load()

    def show(title: str, rows: list[tuple[str, str]]) -> None:
        typer.echo(f"\n{title}")
        if not rows:
            typer.echo("  (none registered)")
            return
        width = max(len(name) for name, _ in rows)
        for name, detail in rows:
            typer.echo(f"  {name:<{width}}  {detail}")

    if what in ("all", "models"):
        show(
            "models",
            [
                (
                    m.name,
                    f"{m.runtime.kind}, {', '.join(m.cameras)}, {m.vram_gb:.0f} GB"
                    + (f" -> {', '.join(m.benchmarks)}" if m.benchmarks else ""),
                )
                for m in registry.models.values()
            ],
        )
    if what in ("all", "benchmarks"):
        show(
            "benchmarks",
            [
                (b.name, f"{b.mode}, dataset={b.dataset}, primary={b.primary_metric or '-'}")
                for b in registry.benchmarks.values()
            ],
        )
    if what in ("all", "datasets"):
        show("datasets", [(d.name, d.description) for d in registry.datasets.values()])
    if what in ("all", "repos"):
        show("repos", [(r.name, f"{r.url} @ {r.sha}") for r in registry.repos.values()])


@app.command("new")
def new_(
    kind: str = typer.Argument(..., help="model | benchmark"),
    name: str = typer.Argument(..., help="lowercase identifier, e.g. 'mymodel'"),
    dataset: str = typer.Option(
        None, "--dataset", help="Benchmarks only: the dataset key. Default: the adapter's name."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing adapter directory."),
) -> None:
    """Scaffold a new model or benchmark adapter from the template.

    Creating the directory *is* the registration -- discovery is a directory
    walk, so there is no registry file to edit and none to forget.
    """
    from paddock.scaffold import ScaffoldError, create

    try:
        result = create(kind, name, dataset=dataset, force=force)
    except ScaffoldError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.render())


@app.command()
def verify(
    model: str = typer.Option(None, "--model", "-m", help="Check this model's Policy contract."),
    bench: str = typer.Option(None, "--bench", "-b", help="Check this benchmark's contract."),
    explain: bool = typer.Option(
        False, "--explain", help="Print what every contract check is for, and exit."
    ),
) -> None:
    """Check an adapter against the contract before running it for real.

    Every serious defect found in this repo so far has been a silent contract
    violation rather than a crash -- a camera2ego of `np.eye(4)` (a 90 degree
    error that reads as "no rotation"), episode state surviving `reset()`, a
    metric set that cannot tell "predicted nothing" from "tracked the lane".
    These checks are chosen from failures that actually happened.

    Run it on a new adapter before its first real sweep, and on an existing one
    after touching anything to do with frames or calibration.
    """
    from paddock import conformance as conformance_mod
    from paddock.verify import verify as do_verify

    if explain:
        typer.echo(conformance_mod.explain())
        raise typer.Exit(code=0)

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    code = do_verify(cfg, model=model, bench=bench, echo=typer.echo)
    raise typer.Exit(code=code)


@app.command()
def config() -> None:
    """Show the resolved site configuration."""
    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"source   {cfg.source}")
    typer.echo(f"site     {cfg.site}")
    typer.echo(f"repos    {cfg.repos}")
    typer.echo(f"runs     {cfg.runs}")
    typer.echo(f"weights  {cfg.weights}")
    typer.echo(f"gpu      {cfg.gpu.budget_gb:.0f} GB budget, serialize={cfg.gpu.serialize}")
    typer.echo("datasets")
    for name, dataset in sorted(cfg.datasets.items()):
        mark = "" if dataset.resolve().exists() else "  (missing)"
        roots = " | ".join(str(r) for r in dataset.roots)
        typer.echo(f"  {name:<12} {roots}{mark}")


def _not_yet(command: str, milestone: str) -> None:
    typer.echo(
        f"`paddock {command}` is not implemented yet (planned for {milestone}).\nNothing was changed.",
        err=True,
    )
    sys.exit(2)


@app.command()
def setup(repo: str = typer.Option(None, "--repo", help="Only this repository.")) -> None:
    """Clone or adopt vendor repos, create the symlink farm, apply patches."""
    _not_yet("setup", "M3")


@app.command()
def renderer(
    scene: str = typer.Option(..., "--scene", help="nuScenes sequence, zero-padded (e.g. 0103)."),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the NeuroNCAP renderer in the foreground.

    Model-independent and slow to start, so one instance serves a whole sweep.
    Leave it running; `paddock run` connects to it.
    """
    from paddock.adapters.benchmarks.neuroncap import launcher

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=launcher.run_foreground(cfg, scene=scene, port=port))


@app.command()
def run(
    bench: str = typer.Option(..., "--bench", "-b"),
    model: str = typer.Option(..., "--model", "-m", help="Model name, or 'all'."),
    scene: str = typer.Option(
        None,
        "--scene",
        help="Comma-separated scene ids (neuroNCAP: nuScenes sequence; Bench2Drive: route id or town). "
        "Default: the benchmark's own default -- 0103 for neuroNCAP, every route for Bench2Drive.",
    ),
    scenarios: str = typer.Option(None, "--scenarios", help="Comma-separated scenario names; default all."),
    runs: int = typer.Option(1, "--runs"),
    limit: int = typer.Option(None, "--limit", help="Open-loop only: cap the sample count."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preflight only; start nothing."),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Run id of an earlier run of this model to continue in place: its completed "
        "samples, episodes or chunks are kept, only the rest are scored, and it keeps its id. "
        "Refused if the switches differ from the ones the run was started under.",
    ),
) -> None:
    """Run a model against a benchmark."""
    from paddock.runner import run_benchmark

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    code = run_benchmark(
        cfg,
        bench=bench,
        model=model,
        scene=scene,
        scenarios=scenarios.split(",") if scenarios else None,
        runs=runs,
        limit=limit,
        resume=resume,
        dry_run=dry_run,
        echo=typer.echo,
    )
    raise typer.Exit(code=code)


@app.command("render")
def render_(
    bench: str = typer.Option(..., "--bench", "-b"),
    model: str = typer.Option(None, "--model", "-m", help="Comma-separated; default every model."),
    run_id: str = typer.Option(None, "--run-id", help="Comma-separated specific runs."),
    sample: str = typer.Option(None, "--sample", help="Draw one sample by id instead of the run figures."),
    out: str = typer.Option(None, "--out", help="Where to write; default the run store."),
) -> None:
    """Draw a comparison of several models on one benchmark.

    Rendering is scoped to the benchmark: it knows what its ground truth is and
    what a comparison means. Uses the newest valid run per model unless --run-id
    names specific ones.

    With --sample, draws that one sample instead -- the aggregate says how often
    a model is wrong and only a sample says how. Sample ids are in the run's
    samples.jsonl; not every benchmark implements it.
    """
    from paddock import render as rendering

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    common = {
        "benchmark": bench,
        "models": model.split(",") if model else None,
        "run_ids": run_id.split(",") if run_id else None,
        "out_dir": Path(out) if out else None,
        "echo": typer.echo,
    }
    if sample:
        code = rendering.render_sample(cfg, sample_id=sample, **common)
    else:
        code = rendering.render(cfg, **common)
    raise typer.Exit(code=code)


@app.command()
def ingest(
    all_: bool = typer.Option(False, "--all", help="Ingest every source in [ingest] of paddock.toml."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report what would be written, write nothing."),
) -> None:
    """Import results produced before paddock into the canonical store."""
    from paddock.results import legacy

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if not all_:
        typer.echo("Nothing selected. Use --all to ingest every source in [ingest].", err=True)
        raise typer.Exit(code=2)

    written, skipped = legacy.ingest_all(cfg, dry_run=dry_run)
    for line in skipped:
        typer.echo(f"  skipped: {line}")
    verb = "would write" if dry_run else "wrote"
    typer.echo(f"{verb} {written} run(s) to results/")


@app.command()
def report(
    include_invalid: bool = typer.Option(False, "--include-invalid"),
    out: str = typer.Option(None, "--out", help="Write to a file instead of stdout."),
) -> None:
    """Render the leaderboard from the results store."""
    from paddock.results.report import markdown

    text = markdown(include_invalid=include_invalid)
    if out:
        Path(out).write_text(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


if __name__ == "__main__":
    app()
