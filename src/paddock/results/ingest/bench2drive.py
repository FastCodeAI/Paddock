"""Ingest recorded Bench2Drive route results into one pooled paddock run per model.

Layout on disk, paddock's own recording layout, consolidated across the runs that
together evaluated a route set (any nesting above the run directory is fine)::

    <runs_root>/.../<run_id>/<route>-<Town>-<Scenario>/results.json

Each ``results.json`` is the leaderboard checkpoint for one route, one record per
repetition. A route may sit under several run ids: a re-run after a crash, a
route evaluated at a second site, a campaign completed in two halves. Per route
the ingest keeps the **latest run id whose results parse as a measurement**;
:func:`paddock.adapters.benchmarks.bench2drive.harness.parse_results` drops
crashed records and refuses a route that never scored, so a crashed first
attempt is superseded by its rerun and a route with no valid attempt is reported
missing rather than averaged in as zero.

The pooled run is an ordinary paddock run record: the aggregate is recomputed
from per-route metrics by the store, and every source run id and results.json
digest is kept in the samples, so each number can be traced to the CARLA file it
came from. Nothing here is copied from a report.

Site configuration (``paddock.toml``)::

    [[ingest.bench2drive]]
    model  = "drivejepa"
    runs   = "/path/to/paddock-runs/bench2drive/drivejepa"   # or a list of roots
    label  = "pooled50"                  # run id becomes <label>-<sequence hash>
    routes = ["1711", "1852", ...]       # optional; default: every route found
    notes  = "24 routes from the 03-04 Sep campaign, 26 from the 09 Sep expansion"
    [ingest.bench2drive.model_env]       # optional; recorded in config
    PADDOCK_CV_SPEED_M_S = "6"
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.bench2drive.harness import NotScored, parse_results
from paddock.results.store import MetricsRecord, RunRecord, sequence_hash

_ROUTE_DIR = re.compile(r"^(?P<route>\d+)-(?P<town>Town\w+?)-(?P<scenario>[A-Za-z]+)$")
_RUN_ID = re.compile(r"^(?P<stamp>\d{8}T\d{6})-[0-9a-f]{6}$")


def route_parts(name: str) -> tuple[str, str, str] | None:
    """``2802-Town12-InvadingTurn`` -> ("2802", "Town12", "InvadingTurn"); None otherwise."""
    match = _ROUTE_DIR.match(name)
    return (match["route"], match["town"], match["scenario"]) if match else None


def _started(run_id: str) -> datetime | None:
    match = _RUN_ID.match(run_id)
    return datetime.strptime(match["stamp"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC) if match else None


def discover(runs_roots: Sequence[Path], routes: list[str] | None = None) -> dict[str, tuple[str, Path]]:
    """route folder -> (run id, results.json) of the latest run that scored the route.

    Run ids sort chronologically by construction (``<UTC stamp>-<hash>``), so
    "latest" is the lexicographically greatest run id. Attempts that did not
    score are skipped here so that a crashed attempt never shadows a rerun.
    """
    chosen: dict[str, tuple[str, Path]] = {}
    for root in runs_roots:
        for results in sorted(root.rglob("results.json")):
            route_dir, run_dir = results.parent, results.parent.parent
            parts = route_parts(route_dir.name)
            if parts is None or (routes is not None and parts[0] not in routes):
                continue
            try:
                parse_results(results)
            except (NotScored, ValueError, json.JSONDecodeError):
                continue
            previous = chosen.get(route_dir.name)
            if previous is None or run_dir.name > previous[0]:
                chosen[route_dir.name] = (run_dir.name, results)
    return chosen


def ingest(
    runs_root: Path | Sequence[Path],
    model: str,
    site: str,
    routes: list[str] | None = None,
    run_label: str = "pooled",
    notes: str = "",
    model_env: dict[str, str] | None = None,
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One pooled paddock run for ``model`` over the routes under ``runs_root``."""
    roots = [runs_root] if isinstance(runs_root, Path) else list(runs_root)
    found = discover(roots, routes)
    samples: list[dict[str, Any]] = []
    for folder, (run_id, results) in sorted(found.items()):
        route, town, scenario = route_parts(folder)  # type: ignore[misc]
        metrics = parse_results(results)
        checkpoint = json.loads(results.read_text()).get("_checkpoint", {})
        durations = [
            float(r["meta"]["duration_system"])
            for r in checkpoint.get("records", [])
            if isinstance(r.get("meta"), dict) and r["meta"].get("duration_system") is not None
        ]
        if durations:
            metrics["duration_s"] = round(sum(durations) / len(durations), 1)
        samples.append(
            {
                "sample_id": f"bench2drive/{folder}",
                "native_id": {
                    "route_id": route,
                    "town": town,
                    "scenario": scenario,
                    "run_id": run_id,
                    "source": f"{run_id}/{folder}/results.json",
                    "sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
                },
                "split": scenario,
                "metrics": metrics,
            }
        )
    wanted = sorted(routes) if routes is not None else sorted(s["native_id"]["route_id"] for s in samples)
    have = {s["native_id"]["route_id"] for s in samples}
    missing = [r for r in wanted if r not in have]
    source_runs = sorted({s["native_id"]["run_id"] for s in samples})
    starts = [d for d in (_started(r) for r in source_runs) if d is not None]
    sample_ids = [s["sample_id"] for s in samples]
    run_id = f"{run_label}-{sequence_hash(sample_ids)}"
    note = notes.strip()
    if missing:
        note = (note + " " if note else "") + f"routes requested but never scored: {', '.join(missing)}."
    run = RunRecord(
        run_id=run_id,
        site=site,
        benchmark="bench2drive",
        model=model,
        mode="closed_loop",
        started_utc=min(starts).isoformat() if starts else "",
        dataset={
            "name": "bench2drive220",
            "n_samples": len(samples),
            "sequences_sha256": sequence_hash(sample_ids),
            "n_expected": len(wanted),
            "n_failed": len(missing),
        },
        config={
            "routes_file": "bench2drive220.xml",
            "routes_subset": wanted,
            "source_runs": source_runs,
            "waypoint_dt": 0.5,
            "model_env": dict(model_env or {}),
        },
        status="complete" if not missing else "partial",
        valid=not missing,
        notes=note,
        # A relative tail only: results/ is tracked and must not name a machine's disk.
        ingested_from="; ".join("/".join(r.parts[-3:]) for r in roots),
    )
    metrics_record = MetricsRecord.from_samples(run_id, samples, primary_metric="driving_score")
    return [(run, metrics_record, samples)]
