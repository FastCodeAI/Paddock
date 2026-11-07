"""Drive the legacy ingest readers from the site's `[ingest]` configuration.

Where the old results live is site-specific, so it belongs in `paddock.toml`
alongside everything else machine-shaped:

    [ingest]
    neuroncap_outputs = "/path/to/neuroncap/outputs"
    alpasim_runs      = "/path/to/alpasim-runs"
    navsim_csv        = "/path/to/exp/<experiment>/<uid>/<timestamp>.csv"

    [ingest.kitscenes_runs]
    openpilot  = "/path/to/openpilot/eval_kitscenes/runs"
    alpamayo15 = "/path/to/alpamayo1.5/eval_kitscenes/runs"

    # One AlpaSim sweep that left one directory per scene, merged into one run.
    [ingest.alpasim_campaigns.legacy-my_campaign]
    dirs  = ["/path/to/runs/my_campaign_scene01", "/path/to/runs/my_campaign_scene02"]
    split = "public_2601"   # optional; for sweeps driven by explicit scene ids
    notes = ""      # optional; required by the store when valid = false
    valid = true

    [ingest.kitscenes_pkl]
    drivejepa  = "/path/to/exp/inference/kitscenes_e2e_final.pkl"
    [[ingest.bench2drive]]           # an array of tables: one block per run
    model  = "drivejepa"
    runs   = "/path/to/paddock-runs/bench2drive/drivejepa"
    label  = "pooled50"
    routes = ["1711", "1852"]        # optional, or "all" (the routes file); default: every route found
    [[ingest.hugsim]]                # likewise one block per run
    model  = "drivejepa"
    runs   = "/path/to/hugsim-eval/dj_released"
    label  = "campaign"

Absent sources are reported and skipped, not treated as errors: no site is
expected to have all of them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from paddock.config import ConfigError, SiteConfig
from paddock.results import store
from paddock.results.ingest import alpasim, bench2drive, hugsim, kitscenes, navsim, neuroncap


def _ingest_section(cfg: SiteConfig) -> dict:
    with cfg.source.open("rb") as fh:
        return tomllib.load(fh).get("ingest", {})


def _bench2drive_routes(cfg: SiteConfig, entry: dict) -> list[str] | None:
    """An `[[ingest.bench2drive]]` entry's `routes`: a list of ids, "all", or absent.

    "all" is every route of the site's routes file. Left out, the expected routes are the
    ones the trees hold, so a route a sweep never drove leaves no trace; a full-set pool
    says "all" so that such a route is reported, and one set aside is named as set aside.
    """
    routes = entry.get("routes")
    if not routes:
        return None
    if isinstance(routes, list):
        return [str(r) for r in routes]
    if routes != "all":
        # A bare string used to be iterated: routes = "1711" asked for routes 1, 7, 1 and 1.
        raise ConfigError(f'[[ingest.bench2drive]] routes is a list of route ids or "all", got {routes!r}')
    from paddock.adapters.benchmarks.bench2drive.harness import parse_routes, routes_path

    settings = cfg.benchmarks.get("bench2drive")
    repo = settings.get("repo") if settings is not None else None
    if not repo:
        raise ConfigError('[[ingest.bench2drive]] routes = "all" needs [benchmarks.bench2drive] repo')
    path = routes_path(Path(repo), settings.get("routes_file"))
    if not path.exists():
        raise ConfigError(f'[[ingest.bench2drive]] routes = "all": routes file not found: {path}')
    return [rid for rid, _, _ in parse_routes(path)]


def ingest_all(cfg: SiteConfig, dry_run: bool = False) -> tuple[int, list[str]]:
    """Returns (runs written, human-readable reasons things were skipped)."""
    section = _ingest_section(cfg)
    skipped: list[str] = []
    collected = []

    if not section:
        skipped.append("no [ingest] section in paddock.toml - nothing to import")
        return 0, skipped

    outputs = section.get("neuroncap_outputs")
    if outputs and Path(outputs).exists():
        collected += neuroncap.ingest(Path(outputs), site=cfg.site)
    elif outputs:
        skipped.append(f"neuroncap_outputs does not exist: {outputs}")

    for model, runs_root in (section.get("kitscenes_runs") or {}).items():
        if Path(runs_root).exists():
            collected += kitscenes.ingest(Path(runs_root), model=model, site=cfg.site)
        else:
            skipped.append(f"kitscenes_runs.{model} does not exist: {runs_root}")

    alpasim_runs = section.get("alpasim_runs")
    if alpasim_runs and Path(alpasim_runs).exists():
        collected += alpasim.ingest(
            Path(alpasim_runs),
            site=cfg.site,
            invalid=section.get("alpasim_invalid") or {},
        )
    elif alpasim_runs:
        skipped.append(f"alpasim_runs does not exist: {alpasim_runs}")

    # AlpaSim runs are not all under one root: the wizard writes into a bucket
    # named for the model, so a site has several. Naming the directories is the
    # difference between ingesting three known rows and walking four roots
    # hoping everything under them is a run worth publishing.
    for run_dir in section.get("alpasim_run_dirs") or []:
        path = Path(run_dir)
        scanned = alpasim.ingest_one(path, site=cfg.site, invalid=section.get("alpasim_invalid") or {})
        if scanned is None:
            skipped.append(f"alpasim_run_dirs: not an ingestible AlpaSim run: {run_dir}")
        else:
            collected.append(scanned)

    # A campaign that drove one scene per wizard invocation left one directory
    # per scene. Which directories those are is site data -- they are paths on
    # one box -- but what they add up to is not: the tracked run.json carries
    # `sequences_sha256` over the scene ids, which is what a reviewer on
    # another machine compares against, and n_samples says how many scenes it was.
    for run_id, campaign in (section.get("alpasim_campaigns") or {}).items():
        dirs = [Path(d) for d in campaign.get("dirs", [])]
        missing = [d for d in dirs if not (d / "aggregate" / "results-summary.json").exists()]
        if not dirs or missing:
            skipped.append(
                f"alpasim_campaigns.{run_id}: {len(missing) or 'all'} of {len(dirs)} run dirs "
                f"have no aggregate/results-summary.json"
            )
            continue
        try:
            collected.append(
                alpasim.ingest_campaign(
                    dirs,
                    run_id=run_id,
                    site=cfg.site,
                    note=campaign.get("notes", ""),
                    valid=campaign.get("valid", True),
                    split=campaign.get("split"),
                )
            )
        except alpasim.Incomparable as exc:
            # Refusing is the point: a merged run whose parts disagree about the
            # driver or the control rate describes no run that ever happened.
            skipped.append(str(exc))

    # A file per run, not a directory of them: see kitscenes.ingest_pkl.
    for model, pkl_path in (section.get("kitscenes_pkl") or {}).items():
        if Path(pkl_path).exists():
            collected += kitscenes.ingest_pkl(Path(pkl_path), model=model, site=cfg.site)
        else:
            skipped.append(f"kitscenes_pkl.{model} does not exist: {pkl_path}")

    # One pooled closed-loop run per entry: see results/ingest/bench2drive.py.
    entries = section.get("bench2drive") or []
    if not isinstance(entries, list):
        # `[ingest.bench2drive]` instead of `[[ingest.bench2drive]]`: iterating it
        # walks the dict's keys, and `entry["runs"]` on a string raised a bare
        # `TypeError: string indices must be integers` an hour from the cause.
        raise ConfigError(
            "[ingest.bench2drive] must be an array of tables -- write [[ingest.bench2drive]] "
            "(double brackets), one block per model or speed band"
        )
    for entry in entries:
        for key in ("model", "runs"):
            if key not in entry:
                raise ConfigError(f"[[ingest.bench2drive]] entry is missing {key!r}: {entry}")
        runs = entry["runs"]
        roots = [Path(r) for r in (runs if isinstance(runs, list) else [runs])]
        if all(r.exists() for r in roots):
            collected += bench2drive.ingest(
                roots,
                model=entry["model"],
                site=cfg.site,
                routes=_bench2drive_routes(cfg, entry),
                run_label=entry.get("label", "pooled"),
                notes=entry.get("notes", ""),
                model_env=entry.get("model_env"),
                ranked=bool(entry.get("ranked", False)),
                setups={
                    label: [str(r) for r in group] for label, group in (entry.get("setups") or {}).items()
                },
            )
        else:
            skipped.append(f"bench2drive.{entry.get('model', '?')} runs do not exist: {runs}")

    # One run per recorded campaign: see results/ingest/hugsim.py.
    entries = section.get("hugsim") or []
    if not isinstance(entries, list):
        raise ConfigError(
            "[ingest.hugsim] must be an array of tables -- write [[ingest.hugsim]], one block per run"
        )
    for entry in entries:
        for key in ("model", "runs"):
            if key not in entry:
                raise ConfigError(f"[[ingest.hugsim]] entry is missing {key!r}: {entry}")
        runs = entry["runs"]
        roots = [Path(r) for r in (runs if isinstance(runs, list) else [runs])]
        if all((r / "results.jsonl").exists() for r in roots):
            collected += hugsim.ingest(
                roots,
                model=entry["model"],
                site=cfg.site,
                run_label=entry.get("label", "campaign"),
                notes=entry.get("notes", ""),
                model_env=entry.get("model_env"),
            )
        else:
            skipped.append(f"hugsim.{entry.get('model', '?')}: no results.jsonl under {runs}")

    csv_path = section.get("navsim_csv")
    if csv_path and Path(csv_path).exists():
        model = section.get("navsim_model", "drivejepa")
        collected += navsim.ingest(Path(csv_path), model=model, site=cfg.site)
    elif csv_path:
        skipped.append(f"navsim_csv does not exist: {csv_path}")

    if dry_run:
        return len(collected), skipped

    for run, metrics, samples in collected:
        store.write_run(run, metrics, samples, artifacts_root=cfg.runs)
    return len(collected), skipped
