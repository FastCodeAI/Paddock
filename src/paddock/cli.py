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
def run(
    model: str = typer.Option(..., "--model", "-m"),
    bench: str = typer.Option(..., "--bench", "-b"),
) -> None:
    """Run one model against one benchmark."""
    _not_yet("run", "M5 (closed-loop) / M6 (open-loop)")


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
