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

**A route that was attempted and never scored is expected, not absent.** With
``routes`` unset the expected set is every route seen under the roots -- the ones
that scored *and* the ones that did not -- so such a route lands in
``routes_subset``, counts in ``n_failed``, makes the run ``partial`` and is named
in ``notes`` with the reason its last attempt gave. The alternative, deriving the
expected set from the samples that survived, cannot report a loss at all: it is
the case where a model crashed out of a route that the record most needs to show,
and it would show a smaller route set, a higher mean and ``n_failed: 0``.

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
from typing import Any, NamedTuple

from paddock.adapters.benchmarks.bench2drive.harness import NotScored, crash_reason, parse_results
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


class Attempt(NamedTuple):
    """The run that scored a route, and what it scored."""

    run_id: str
    results: Path
    metrics: dict[str, float]


class Survey(NamedTuple):
    """What the recording trees hold: what scored, and what was seen and did not.

    ``rejected`` is the half a plain "latest scoring attempt wins" walk throws
    away, and it is the half that matters for the record: a route every attempt
    of which crashed is *evidence*, not absence. Keeping it here is what lets
    :func:`ingest` report it as attempted-and-lost instead of quietly shrinking
    the route set (and with it the denominator of every mean).
    """

    #: route folder -> the attempt whose numbers are used.
    scored: dict[str, Attempt]
    #: route folder -> why the last attempt seen there is not a measurement.
    rejected: dict[str, str]


def discover(runs_roots: Sequence[Path], routes: list[str] | None = None) -> Survey:
    """Walk the recording trees once: the scored routes, and the rejected ones.

    Run ids sort chronologically by construction (``<UTC stamp>-<hash>``), so
    "latest" is the lexicographically greatest run id. Attempts that did not
    score never shadow a rerun that did -- but they are remembered, so a route
    with no scoring attempt anywhere still leaves a trace.

    Unreadable files are rejections too, not silence: a truncated ``results.json``
    on a network mount used to abort the whole walk, and a malformed one used to
    be indistinguishable from a route that crashed.
    """
    scored: dict[str, Attempt] = {}
    rejected: dict[str, str] = {}
    for root in runs_roots:
        for results in sorted(root.rglob("results.json")):
            route_dir, run_dir = results.parent, results.parent.parent
            parts = route_parts(route_dir.name)
            if parts is None or (routes is not None and parts[0] not in routes):
                continue
            try:
                metrics = parse_results(results)
            except NotScored as exc:
                # The message names the file; keep the run id and drop the path,
                # because this reason is written into a tracked run.json.
                rejected[route_dir.name] = f"{run_dir.name}: {str(exc).split('.json: ', 1)[-1]}"
                continue
            # json.JSONDecodeError and UnicodeDecodeError are ValueError; OSError
            # is the NFS read that used to take the whole ingest down with it.
            except (ValueError, OSError) as exc:
                rejected[route_dir.name] = f"{run_dir.name}: unreadable ({type(exc).__name__})"
                continue
            previous = scored.get(route_dir.name)
            if previous is None or run_dir.name > previous.run_id:
                scored[route_dir.name] = Attempt(run_dir.name, results, metrics)
    return Survey(scored, {k: v for k, v in rejected.items() if k not in scored})


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
    survey = discover(roots, routes)
    samples: list[dict[str, Any]] = []
    for folder, attempt in sorted(survey.scored.items()):
        route, town, scenario = route_parts(folder)  # type: ignore[misc]
        raw = attempt.results.read_bytes()
        metrics = dict(attempt.metrics)
        checkpoint = json.loads(raw).get("_checkpoint", {})
        # The repetitions that scored, and only those: parse_results drops the
        # crashed ones, so averaging their `duration_system` here would put a
        # 1.0-second stack trace next to the `n_runs` that excludes it.
        durations = [
            float(r["meta"]["duration_system"])
            for r in checkpoint.get("records", [])
            if isinstance(r.get("meta"), dict)
            and r["meta"].get("duration_system") is not None
            and not crash_reason(str(r.get("status", "")))
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
                    "run_id": attempt.run_id,
                    "source": f"{attempt.run_id}/{folder}/results.json",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "split": scenario,
                "metrics": metrics,
            }
        )
    have = {s["native_id"]["route_id"] for s in samples}
    # A route that was attempted and never scored is expected too. Defaulting
    # `wanted` to the samples that survived would make `missing` empty by
    # construction -- n_failed: 0, status complete, valid true -- on exactly the
    # case the record exists to show: a model that crashed out of a route.
    attempted = {route_parts(f)[0]: why for f, why in survey.rejected.items() if route_parts(f)}  # type: ignore[index]
    wanted = sorted(routes) if routes is not None else sorted(have | set(attempted))
    missing = [r for r in wanted if r not in have]
    source_runs = sorted({s["native_id"]["run_id"] for s in samples})
    starts = [d for d in (_started(r) for r in source_runs) if d is not None]
    sample_ids = [s["sample_id"] for s in samples]
    run_id = f"{run_label}-{sequence_hash(sample_ids)}"
    note = notes.strip()
    if missing:
        lost = ", ".join(f"{r} ({attempted.get(r, 'no attempt found')})" for r in missing)
        note = (note + " " if note else "") + f"routes expected but never scored: {lost}."
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
