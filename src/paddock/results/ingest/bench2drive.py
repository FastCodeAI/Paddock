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

**The configuration in the record is read from the runs, not typed here.**
``routes_file`` is a site setting and ``waypoint_dt`` a model capability, so
neither is knowable from a results tree; both come from the harness's own
``run.json`` (or the older ``run_config.json``) beside the route folders, as does
``model_env`` and the site that produced them. Source runs that disagree are
recorded disagreeing rather than collapsed to the first value, the operator's
``model_env`` is layered on top as an override, and a route whose source run
names a *different* model is refused -- pointing ``runs`` one component too high
otherwise pools two models into one row. ``source_config_read`` says how many of
the source runs kept such a record at all.

The run id is ``<label>-<route-set hash>-<config hash>``. The route set alone
cannot distinguish two settings of the same routes -- five constant-velocity
speed bands, say -- and ``store.write_run`` overwrites a directory that already
exists, so with the label as the only difference a copied config block silently
leaves one directory holding whichever entry ran last.

Site configuration (``paddock.toml``)::

    [[ingest.bench2drive]]
    model  = "drivejepa"
    runs   = "/path/to/paddock-runs/bench2drive/drivejepa"   # or a list of roots
    label  = "pooled50"                  # run id: <label>-<route hash>-<config hash>
    routes = ["1711", "1852", ...]       # optional; default: every route found
    notes  = "24 routes from the 03-04 Sep campaign, 26 from the 09 Sep expansion"
    [ingest.bench2drive.model_env]       # optional; overrides what the runs recorded
    PADDOCK_CV_SPEED_M_S = "6"
    [ingest.bench2drive.setups]          # optional; recording setup -> route ids
    assisted = ["1711", "2802"]

``setups`` names the configuration each route was actually driven under, and lands
on the samples as ``group``, so the tracked ``metrics.json`` carries a per-setup
breakdown in ``by_group``. Routes driven under **more than one** setup are not one
experiment: a mean over them describes no configuration anybody ran, so such a run
is written ``valid: false`` with the setups in ``notes``. It is kept, and the
leaderboard lists it as excluded rather than ranking it -- which is the whole point
of recording the setup instead of a sentence in prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from paddock.adapters.benchmarks.bench2drive.harness import (
    DEFAULT_ROUTES_FILE,
    NotScored,
    crash_reason,
    parse_results,
)
from paddock.registry import Registry
from paddock.results.store import MetricsRecord, RunRecord, config_hash, sequence_hash

#: Bench2Drive's published protocol calls the model once per simulator tick --
#: 20 Hz (docs/benchmarks/bench2drive.md). A run driven at another rate ranks only
#: when that rate is the one the model's model.yaml declares (`_registered_native_hz`);
#: at any other rate it measures a different protocol, and is kept and not ranked.
PROTOCOL_HZ = 20.0

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


def source_record(run_dir: Path) -> dict[str, Any]:
    """What the harness recorded about the run that produced these routes.

    ``run.json`` is the harness's own run record -- model, site and the config it
    ran with. Campaigns that predate it left ``run_config.json``, the same config
    without the identity. Either is better than a constant typed into an ingest:
    ``routes_file`` is a site setting and ``waypoint_dt`` a model capability, so
    neither is knowable from the results tree alone. Missing or unreadable is not
    an error -- the oldest recordings have neither file -- it just means nothing
    was read, and the caller records how much it had to assume.
    """
    record: dict[str, Any] = {}
    payload = _read_json(run_dir / "run.json")
    if isinstance(payload, dict):
        record = {key: payload[key] for key in ("model", "site") if payload.get(key)}
        if isinstance(payload.get("config"), dict):
            record["config"] = payload["config"]
    if "config" not in record:
        fallback = _read_json(run_dir / "run_config.json")
        if isinstance(fallback, dict):
            record["config"] = fallback
    return record


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (ValueError, OSError):
        return None


def _driven_off_protocol(model_env: dict[str, Any]) -> float | None:
    """The rate this run was driven at, when that is not the benchmark's protocol.

    ``None`` for every run driven at 20 Hz, which is every run that does not say
    otherwise -- the rate is only in the record when an operator put it there.
    """
    raw = (model_env or {}).get("PADDOCK_INFER_HZ")
    if raw in (None, ""):
        return None
    try:
        hz = float(raw)
    except (TypeError, ValueError):
        return None
    return None if hz <= 0 or hz == PROTOCOL_HZ else hz


def _agreed(values: Sequence[Any]) -> Any:
    """One value if the source runs agree, the sorted list if they do not.

    Collapsing a disagreement to "the first one" is how a record comes to assert
    a configuration that only some of its routes ran under.
    """
    distinct = sorted({json.dumps(v, sort_keys=True) for v in values if v is not None})
    if not distinct:
        return None
    return json.loads(distinct[0]) if len(distinct) == 1 else [json.loads(d) for d in distinct]


def _registered_waypoint_dt(model: str) -> float | None:
    try:
        spec = Registry.load().models.get(model)
    except Exception:  # noqa: BLE001 -- a broken registry must not fail an ingest
        return None
    return getattr(spec, "waypoint_dt", None) if spec else None


def _registered_native_hz(model: str) -> tuple[float, str] | None:
    """The rate ``model``'s model.yaml declares it runs at, and the key that says so.

    ``planning_hz`` first: how often the model replans, and the rate ``paddock run``
    drives it at on Bench2Drive. A model that declares no planning rate but declares
    the observation rate it was built for (``native_input_hz``) is called once per
    observation at that rate, so that rate stands in. None when it declares neither.
    """
    try:
        spec = Registry.load().models.get(model)
    except Exception:  # noqa: BLE001 -- a broken registry must not fail an ingest
        return None
    for key in ("planning_hz", "native_input_hz"):
        hz = float(getattr(spec, key, 0.0) or 0.0) if spec else 0.0
        if hz > 0:
            return hz, key
    return None


def discover(runs_roots: Sequence[Path], routes: list[str] | None = None) -> Survey:
    """Walk the recording trees once: the scored routes, and the rejected ones.

    Run ids sort chronologically by construction (``<UTC stamp>-<hash>``), so
    "latest" is the lexicographically greatest run id. Attempts that did not
    score never shadow a rerun that did -- but they are remembered, so a route
    with no scoring attempt anywhere still leaves a trace.

    Unreadable files are rejections too, not silence: a truncated ``results.json``
    on a network mount used to abort the whole walk, and a malformed one used to
    be indistinguishable from a route that crashed.

    ``routes`` selects what is *scored*, not what is *seen*: a route excluded by
    the subset and never scored anywhere is still reported, because "excluded"
    and "attempted and lost" are different facts about a benchmark run and only
    one of them is a choice.
    """
    scored: dict[str, Attempt] = {}
    rejected: dict[str, str] = {}
    for root in runs_roots:
        for results in sorted(root.rglob("results.json")):
            route_dir, run_dir = results.parent, results.parent.parent
            parts = route_parts(route_dir.name)
            if parts is None:
                continue
            wanted_here = routes is None or parts[0] in routes
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
            if not wanted_here:
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
    setups: dict[str, list[str]] | None = None,
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One pooled paddock run for ``model`` over the routes under ``runs_root``."""
    roots = [runs_root] if isinstance(runs_root, Path) else list(runs_root)
    setup_of = {route: label for label, group in (setups or {}).items() for route in group}
    survey = discover(roots, routes)
    rejected = dict(survey.rejected)
    # Keyed by directory, not by run id: two roots can hold the same run id, and
    # that collision is exactly the mix this is here to catch.
    sources: dict[Path, dict[str, Any]] = {}
    contributing: list[Path] = []
    samples: list[dict[str, Any]] = []
    for folder, attempt in sorted(survey.scored.items()):
        route, town, scenario = route_parts(folder)  # type: ignore[misc]
        run_dir = attempt.results.parent.parent
        source = sources.setdefault(run_dir, source_record(run_dir))
        recorded_model = source.get("model")
        if recorded_model and recorded_model != model:
            # One path component too high (`.../bench2drive` for
            # `.../bench2drive/drivejepa`) otherwise pools another model's routes
            # into this model's row, and `native_id.source` does not name the
            # model directory, so the mix is not recoverable afterwards.
            rejected[folder] = f"{attempt.run_id}: recorded for model {recorded_model!r}, not {model!r}"
            continue
        contributing.append(run_dir)
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
                # The recording setup, aggregated into metrics.json's `by_group`.
                "group": setup_of.get(route, ""),
                "metrics": metrics,
            }
        )
    have = {s["native_id"]["route_id"] for s in samples}
    # A route that was attempted and never scored is expected too. Defaulting
    # `wanted` to the samples that survived would make `missing` empty by
    # construction -- n_failed: 0, status complete, valid true -- on exactly the
    # case the record exists to show: a model that crashed out of a route.
    attempted = {route_parts(f)[0]: why for f, why in rejected.items() if route_parts(f)}  # type: ignore[index]
    wanted = sorted(routes) if routes is not None else sorted(have | set(attempted))
    missing = [r for r in wanted if r not in have]
    source_runs = sorted({s["native_id"]["run_id"] for s in samples})
    starts = [d for d in (_started(r) for r in source_runs) if d is not None]
    sample_ids = [s["sample_id"] for s in samples]

    # Configuration comes from the runs being ingested, not from this function.
    used = [sources[d] for d in dict.fromkeys(contributing)]
    configs = [rec.get("config") or {} for rec in used]
    recorded_env: dict[str, Any] = {}
    for key in sorted({k for c in configs for k in (c.get("model_env") or {})}):
        recorded_env[key] = _agreed([(c.get("model_env") or {}).get(key) for c in configs])
    config = {
        "routes_file": _agreed([c.get("routes_file") for c in configs]) or DEFAULT_ROUTES_FILE,
        "routes_subset": wanted,
        "source_runs": source_runs,
        # `site` names the operation that pooled these runs; the machines that
        # produced them are here, as far as the recordings state them.
        "source_sites": sorted({rec["site"] for rec in used if rec.get("site")}),
        "waypoint_dt": _agreed([c.get("waypoint_dt") for c in configs]) or _registered_waypoint_dt(model),
        # The operator's entry is an override of what ran, not a substitute for it.
        "model_env": {**recorded_env, **dict(model_env or {})},
        "source_config_read": f"{sum(1 for c in configs if c)}/{len(used)} source runs",
    }
    # The model.yaml env each source run was made under (#99). Two values are
    # two settings of the model, not one experiment. `_agreed` skips a missing
    # value, so a run that recorded none (every run before #99) is counted
    # rather than read as agreeing with the others.
    declared = _agreed([c.get("declared_env") for c in configs])
    undeclared = sum(1 for c in configs if "declared_env" not in c) if declared is not None else 0
    if declared is not None:
        config["declared_env"] = declared
    # Every route that was attempted and never scored, whether or not the
    # configured subset asked for it. Route 11715 is the case: excluded for every
    # model because CARLA crashes on it, so it is not in `routes` and not in
    # `n_failed` -- and without this line `results/` would not say it happened.
    if attempted:
        config["routes_never_scored"] = dict(sorted(attempted.items()))
    pooled_setups = sorted({s["group"] for s in samples if s["group"]})
    if setups:
        config["recording_setups"] = {label: sorted(group) for label, group in sorted(setups.items())}
    run_id = f"{run_label}-{sequence_hash(sample_ids)}-{config_hash(config)}"
    note = notes.strip()
    outside = sorted(r for r in attempted if r not in set(wanted))
    if outside:
        lost = ", ".join(f"{r} ({attempted[r]})" for r in outside)
        note = (note + " " if note else "") + (
            f"attempted outside the configured route subset and never scored: {lost}."
        )
    if len(pooled_setups) > 1:
        note = (note + " " if note else "") + (
            f"pooled across {len(pooled_setups)} recording setups ({', '.join(pooled_setups)}); "
            "one mean describes none of them, so this run does not rank -- the per-setup "
            "figures are in metrics.json by_group."
        )
    if isinstance(declared, list):
        note = (note + " " if note else "") + (
            f"source runs were made under {len(declared)} different model.yaml envs "
            "(config.declared_env); one mean describes none of them, so this run does not rank."
        )
    if undeclared:
        note = (note + " " if note else "") + (
            f"{undeclared} of {len(used)} source runs recorded no declared_env (made before #99), "
            "so which model.yaml env their routes ran is not known and this run does not rank."
        )
    # The override is layered on top of what the runs recorded, so a key that is
    # only in the override is an operator's assertion -- the note has to say so,
    # or `results/` claims the source runs recorded a value they never held.
    overridden = sorted(k for k in dict(model_env or {}) if k not in recorded_env)
    unverified = sum(1 for rec in used if not rec.get("model"))
    if unverified:
        note = (note + " " if note else "") + (
            f"{unverified} of {len(used)} source runs kept no run.json, so their routes are "
            f"attributed to {model} on the tree they were found in; config.model_env and the "
            f"rest of config are what {config['source_config_read']} recorded."
        )
    if overridden:
        keys = ", ".join(overridden)
        was = "was" if len(overridden) == 1 else "were"
        note = (note + " " if note else "") + (
            f"{keys} {was} supplied by the ingest config and recorded by no source run, "
            "so it is the operator's assertion about the campaign, not a reading from it."
        )
    # A model driven at the rate its own model.yaml declares is on protocol: that is
    # the rate `paddock run` drives it at, and that run ranks. Any other rate is not.
    driven = _driven_off_protocol(config["model_env"])
    native = _registered_native_hz(model) if driven is not None else None
    off_protocol = None if native and math.isclose(driven, native[0]) else driven
    if driven is not None and off_protocol is None:
        note = (note + " " if note else "") + (
            f"driven at {driven:g} Hz, the rate {model}/model.yaml declares as {native[1]}, not the "
            f"benchmark's {PROTOCOL_HZ:g} Hz: a model driven at its own declared rate ranks, as "
            "`paddock run` drives it -- the paired 20 Hz comparison is "
            "docs/benchmarks/bench2drive-native-rate.md."
        )
    elif off_protocol is not None:
        declared = f"the {native[0]:g} Hz {model}/model.yaml declares ({native[1]})" if native else (
            f"a rate {model}/model.yaml declares (it declares none)"
        )
        note = (note + " " if note else "") + (
            f"driven at {off_protocol:g} Hz, neither the {PROTOCOL_HZ:g} Hz the benchmark's published "
            f"protocol calls the model at nor {declared}, so this run is kept and does not rank -- "
            "the paired comparison is docs/benchmarks/bench2drive-native-rate.md."
        )
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
        config=config,
        status="complete" if not missing else "partial",
        # More than one recording setup is not one experiment: its mean is not a
        # measurement of anything, so it is kept and not ranked rather than
        # published as a number nobody drove. A run driven at a rate that is
        # neither 20 Hz nor the model's own declared rate is a different protocol.
        # Source runs under different, or unrecorded, model.yaml envs are the
        # first case again: two settings of the model behind one mean.
        valid=not missing
        and len(pooled_setups) <= 1
        and off_protocol is None
        and not isinstance(declared, list)
        and not undeclared,
        notes=note,
        # A relative tail only: results/ is tracked and must not name a machine's disk.
        ingested_from="; ".join("/".join(r.parts[-3:]) for r in roots),
    )
    metrics_record = MetricsRecord.from_samples(run_id, samples, primary_metric="driving_score")
    return [(run, metrics_record, samples)]
