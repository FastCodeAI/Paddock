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
from paddock.registry import ModelSpec, Registry

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


def model_card(m: ModelSpec) -> str:
    """One line of what a model IS, for `paddock list models`.

    Only what is recorded: a model declaring nothing gets no card line, which
    reads as "nobody has written this down" rather than as a row of zeros.
    """
    parts: list[str] = []
    if m.architecture:
        parts.append(m.architecture)
    if m.params_m is not None:
        # Billions past 1000 M, because "10000 M params" reads as a typo.
        # Billions past 1000 M, because "10000 M params" reads as a typo -- but
        # to one decimal, because :.0f turns 1500 M into "2 B".
        big = m.params_m >= 1000
        billions = round(m.params_m / 1000, 1)
        parts.append(f"{billions:g} B params" if big else f"{m.params_m:g} M params")
    if m.size_gb is not None:
        parts.append(f"{m.size_gb:g} GB{' ' + m.precision if m.precision else ''}")
    if m.macs_g is not None:
        parts.append(f"{m.macs_g:g} GMACs")
    if m.input_size is not None:
        parts.append(f"{m.input_size[0]}x{m.input_size[1]}")
    if m.history_frames is not None:
        parts.append(f"{m.history_frames} frame{'s' if m.history_frames != 1 else ''}")
    out = m.output_kind
    if m.horizon_s is not None:
        out += f" @{m.horizon_s:g}s"
    parts.append(out)
    if m.stochastic:
        parts.append("stochastic")
    if not m.needs_speed:
        parts.append("no speed input")
    if m.trained_on:
        parts.append(
            "trained on "
            + ", ".join(f"{t.dataset}/{t.split}" if t.split else t.dataset for t in m.trained_on)
        )
    return " · ".join(parts)


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
        models = list(registry.models.values())
        typer.echo("\nmodels")
        if not models:
            typer.echo("  (none registered)")
        width = max((len(m.name) for m in models), default=0)
        for m in models:
            runtime = f"{m.runtime.kind}, {', '.join(m.cameras)}, {m.vram_gb:.0f} GB" + (
                f" -> {', '.join(m.benchmarks)}" if m.benchmarks else ""
            )
            typer.echo(f"  {m.name:<{width}}  {runtime}")
            card = model_card(m)
            if card:
                # Indented under the runtime line rather than appended to it:
                # what a model IS and how to launch it are different questions,
                # and joining them makes a line nobody reads to the end of.
                typer.echo(f"  {'':<{width}}  {card}")
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
        "chunks are kept, only the rest are scored, and it keeps its id.",
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
    out: str = typer.Option(None, "--out", help="Where to write; default the run store."),
) -> None:
    """Draw a comparison of several models on one benchmark.

    Rendering is scoped to the benchmark: it knows what its ground truth is and
    what a comparison means. Uses the newest valid run per model unless --run-id
    names specific ones.
    """
    from paddock.render import render as do_render

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    code = do_render(
        cfg,
        benchmark=bench,
        models=model.split(",") if model else None,
        run_ids=run_id.split(",") if run_id else None,
        out_dir=Path(out) if out else None,
        echo=typer.echo,
    )
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
