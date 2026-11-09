"""`paddock` -- the command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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


@app.command()
def setup(
    repo: str = typer.Option(None, "--repo", help="Only this repos.yaml entry."),
    model: str = typer.Option(
        None, "--model", "-m", help="This model's repo (if it has one), then its setup.sh."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Say what would be done; change nothing."),
) -> None:
    """Clone or adopt the vendor repos at their pinned shas, and apply their patches.

    A clone already in place is adopted, not re-cloned. One whose HEAD is not
    the pinned sha is left exactly as it is: setup prints both shas and the
    checkout that would fix it, and exits 1. Patches already applied are
    skipped, so running it again is safe.

    With --model, the model's own setup.sh then builds its environment, with
    PADDOCK_REPO, PADDOCK_VENV and PADDOCK_WEIGHTS set from paddock.toml.
    """
    from paddock import vendor

    try:
        cfg = SiteConfig.load()
        code = vendor.setup(cfg, repo=repo, model=model, dry_run=dry_run, echo=typer.echo)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    raise typer.Exit(code=code)


data_app = typer.Typer(no_args_is_help=True, help="Download datasets from their official sources.")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def data_fetch(
    name: str = typer.Argument(..., help="A datasets.yaml entry with a fetch recipe, e.g. comma2k19."),
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Only this part of the dataset; repeatable.")
    ] = None,
    match: Annotated[
        list[str] | None,
        typer.Option(
            "--match", help="Only files whose name or path matches this glob, e.g. Chunk_1.zip; repeatable."
        ),
    ] = None,
    dest: Annotated[
        Path | None, typer.Option("--dest", help="Where it goes. Default: its path in paddock.toml.")
    ] = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="List the files and sizes; download nothing."),
    accept_terms: bool = typer.Option(
        False, "--accept-terms", help="You have read the dataset's terms of use and agree to them."
    ),
) -> None:
    """Download a dataset's evaluation files from where its owners publish them.

    Every file is checked against the size and sha256 its pinned source gives,
    then unpacked into the layout paddock reads. Interrupted runs continue where
    they stopped. When paddock.toml has no entry for the dataset yet, one is
    added pointing at --dest.
    """
    from paddock import fetch

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    code = fetch.fetch(
        cfg,
        name,
        only=only or None,
        match=match or None,
        dest=dest,
        dry_run=dry_run,
        accept_terms=accept_terms,
        echo=typer.echo,
    )
    raise typer.Exit(code=code)


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
def supersede(
    bench: str = typer.Option(..., "--bench", "-b"),
    model: str = typer.Option(..., "--model", "-m"),
    run_id: str = typer.Option(..., "--run-id", help="The older run, the one to stop ranking."),
    by: str = typer.Option(..., "--by", help="Run id of the later run that replaces it."),
    reason: str = typer.Option(..., "--reason", help="Why, in one sentence, for the record."),
    by_model: str = typer.Option(
        "",
        "--by-model",
        help="Model of the replacing run, when it is not --model's own (default: the same model).",
    ),
) -> None:
    """Mark an older run as replaced by a later one.

    Two runs of one model on one benchmark are both valid as their runner left
    them, and the leaderboard has no way to pick. This is how the choice is
    recorded -- the alternative people reach for is editing `valid` in the file,
    which is the thing `results/` most needs nobody to do.

    The later run is normally the same model's. `--by-model` names another
    model's run on the same benchmark, for a row that moves to a different
    registry entry of the model.

    Changes `valid`, `superseded_by` and `notes`. The measured numbers and the
    run's own identity are left exactly as produced, and `metrics.json` is not
    touched: the older run still happened and still measured what it measured.
    """
    from paddock.config import REPO_ROOT
    from paddock.results.store import SupersedeError
    from paddock.results.store import supersede as do_supersede

    try:
        path = do_supersede(bench, model, run_id, by=by, reason=reason, by_model=by_model)
    except SupersedeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    later = by if by_model in ("", model) else f"{by_model}/{by}"
    typer.echo(f"{run_id} superseded by {later}; rewrote {path.relative_to(REPO_ROOT)}")
    typer.echo("Commit the change: results/ is tracked.")


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
