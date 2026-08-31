"""Drive the legacy ingest readers from the site's `[ingest]` configuration.

Where the old results live is site-specific, so it belongs in `paddock.toml`
alongside everything else machine-shaped:

    [ingest]
    neuroncap_outputs = "/path/to/neuroncap/outputs"
    navsim_csv        = "/path/to/exp/<experiment>/<uid>/<timestamp>.csv"

    [ingest.kitscenes_runs]
    openpilot  = "/path/to/openpilot/eval_kitscenes/runs"
    alpamayo15 = "/path/to/alpamayo1.5/eval_kitscenes/runs"

Absent sources are reported and skipped, not treated as errors: no site is
expected to have all of them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from paddock.config import SiteConfig
from paddock.results import store
from paddock.results.ingest import kitscenes, navsim, neuroncap


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
