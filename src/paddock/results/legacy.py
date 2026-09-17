"""Drive the legacy ingest readers from the site's `[ingest]` configuration.

Where the old results live is site-specific, so it belongs in `paddock.toml`
alongside everything else machine-shaped:

    [ingest]
    neuroncap_outputs = "/path/to/neuroncap/outputs"
    navsim_csv        = "/path/to/exp/<experiment>/<uid>/<timestamp>.csv"

    [ingest.kitscenes_runs]
    openpilot  = "/path/to/openpilot/eval_kitscenes/runs"
    alpamayo15 = "/path/to/alpamayo1.5/eval_kitscenes/runs"

    [ingest.kitscenes_pkl]
    drivejepa  = "/path/to/exp/inference/kitscenes_e2e_final.pkl"
    [[ingest.bench2drive]]           # an array of tables: one block per run
    model  = "drivejepa"
    runs   = "/path/to/paddock-runs/bench2drive/drivejepa"
    label  = "pooled50"
    routes = ["1711", "1852"]        # optional; default: every route found

Absent sources are reported and skipped, not treated as errors: no site is
expected to have all of them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from paddock.config import ConfigError, SiteConfig
from paddock.results import store
from paddock.results.ingest import bench2drive, kitscenes, navsim, neuroncap


def _ingest_section(cfg: SiteConfig) -> dict:
    with cfg.source.open("rb") as fh:
        return tomllib.load(fh).get("ingest", {})


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
                routes=[str(r) for r in entry["routes"]] if entry.get("routes") else None,
                run_label=entry.get("label", "pooled"),
                notes=entry.get("notes", ""),
                model_env=entry.get("model_env"),
                setups={
                    label: [str(r) for r in group] for label, group in (entry.get("setups") or {}).items()
                },
            )
        else:
            skipped.append(f"bench2drive.{entry.get('model', '?')} runs do not exist: {runs}")

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
