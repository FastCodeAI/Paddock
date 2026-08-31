"""`paddock` -- the command line.

Commands that are not implemented yet say so and name the milestone, rather than
succeeding quietly with nothing done.
"""

from __future__ import annotations

import sys

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
        show("models", [
            (m.name, f"{m.runtime.kind}, {', '.join(m.cameras)}, {m.vram_gb:.0f} GB"
                     + (f" -> {', '.join(m.benchmarks)}" if m.benchmarks else ""))
            for m in registry.models.values()
        ])
    if what in ("all", "benchmarks"):
        show("benchmarks", [
            (b.name, f"{b.mode}, dataset={b.dataset}, primary={b.primary_metric or '-'}")
            for b in registry.benchmarks.values()
        ])
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
        f"`paddock {command}` is not implemented yet (planned for {milestone}).\n"
        f"Nothing was changed.",
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
def ingest(all_: bool = typer.Option(False, "--all")) -> None:
    """Import existing results into the canonical store."""
    _not_yet("ingest", "M4")


@app.command()
def report() -> None:
    """Render the leaderboard from the results store."""
    _not_yet("report", "M4")


if __name__ == "__main__":
    app()
