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

Absent sources are reported and skipped, not treated as errors: no site is
expected to have all of them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from paddock.config import SiteConfig
from paddock.results import store
from paddock.results.ingest import alpasim, kitscenes, navsim, neuroncap


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
    # `sequences_sha256` over the scene ids, which is what a reviewer at the
    # other site compares against, and n_samples says how many scenes it was.
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
