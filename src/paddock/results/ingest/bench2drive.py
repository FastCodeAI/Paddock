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

An early stop (``stop_*`` in ``[benchmarks.bench2drive]``, see
``adapters/benchmarks/bench2drive/early_stop.py``) is read the same way: the
pooled ``config.early_stop`` is what the source runs recorded, every route says
in ``ended_stuck`` / ``ended_over_time`` / ``ended_collision`` / ``ended_manual``
whether one ended it, and the notes say why the run still ranks: a route the
early stop ended is a failed drive, scored as it stood. Source runs that stopped
routes under different rules -- or some with the rules and some without -- do
not pool into a ranked run, and neither do source runs that would not rank on
their own: driven through a camera rig other than paddock's (``camera_rig``,
then in the pooled config), or with a setting the model's ``model.yaml``
declares set to another value.

**A route set aside for every model is neither expected nor lost.** A route in
``harness.SET_ASIDE`` failed in the simulator on every attempt (the list is empty since patch
0003 made 11715, 3800, 23695 and 24071 run), so it measures no model: such a route leaves the
pool before anything is read from it, its id goes in ``config.routes_set_aside`` and its reason
in the notes -- the list as it stands when the pool is made, so every model's pool of the same
routes says the same. A full-set pool
asks for ``routes = "all"``: with ``routes`` left out the expected set is what the trees
hold, and a route a sweep never drove would leave no trace.

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
    routes = ["1711", "1852", ...]       # optional, or "all": every route of the routes file;
                                         # default: every route found
    notes  = "24 routes from the 03-04 Sep campaign, 26 from the 09 Sep expansion"
    ranked = true                        # optional; ranks it as recorded despite its rate or a lost route
    tracked = ["bench2drive/drivejepa/full220-..."]   # optional; records in results/ whose
                                         # upstream_scores.json rows supply routes not under `runs`
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

**A route driven at another site can come from that site's tracked record.** A record
in ``results/bench2drive/`` carries, beside its ``metrics.json``, the per-route rows of
``upstream_scores.json`` (``tools/benchmarks/bench2drive/upstream_scores.py``): each
route's scores and infraction counts from its ``results.json``, and how the early stop
ended it. ``tracked = ["bench2drive/<model>/<run_id>"]`` names such records, and a
route not scored under ``runs`` is taken from them -- so routes driven here can complete
a record made where the raw results are not. A raw route under ``runs`` takes precedence,
route by route. Each route is measured as :func:`parse_results` and :func:`ended_early`
would measure its ``results.json``, with one exception: ``duration_s``, the evaluator's
wall-clock seconds, is not in the rows, so a route taken from a record carries none and
the store counts it missing, as it does for a route ``--resume`` reused.

That is exact only for a record the rows fully describe, so :func:`read_tracked` refuses
one that is not: no ``upstream_scores.json``; rows that are not the record's routes; a
``metrics.json`` that does not pin one scored repetition per route (``n_runs`` mean and
max 1, ``n_crashed`` max 0) -- the rows hold the first record only; ``ended_*`` blocks
that are neither absent nor present for every route -- a row's ``ended_early: null``
cannot say whether its route had an ``early_stop.json``; or rows that, aggregated as
here, do not reproduce the record's own ``metrics.json``. The record's ``config`` joins
the source runs' in every agreement check above, a record that does not rank keeps the
pool from ranking, and every pooled route names its origin: the raw run and the sha256 of
its ``results.json``, or the record and the sha256 of its ``upstream_scores.json``.
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

from paddock.adapters.benchmarks.bench2drive import early_stop, rig
from paddock.adapters.benchmarks.bench2drive.harness import (
    DEFAULT_ROUTES_FILE,
    SET_ASIDE,
    NotScored,
    crash_reason,
    ended_early,
    every_tick,
    failure_said,
    parse_checkpoint,
    parse_results,
    read_failure,
    set_aside_remark,
    stop_shares,
)
from paddock.registry import Registry
from paddock.results.store import MetricsRecord, RunRecord, config_hash, sequence_hash

#: The per-route rows a tracked record carries beside its metrics.json
#: (tools/benchmarks/bench2drive/upstream_scores.py).
UPSTREAM_SCORES = "upstream_scores.json"
#: What upstream_scores.json does not hold: the evaluator's wall-clock seconds
#: (results.json ``meta.duration_system``).
NOT_IN_ROWS = "duration_s"

#: Bench2Drive's published protocol calls the model once per simulator tick --
#: 20 Hz (docs/benchmarks/bench2drive.md). A model may declare a slower rate of its
#: own (`planning_hz`), and a pool asked at it ranks. A rate an operator set in its
#: place -- the host's PADDOCK_INFER_HZ -- is a different protocol: kept, not ranked.
#: Before runs recorded their rate, it is PADDOCK_INFER_HZ in `model_env`, and ranks
#: only when it is the rate the model's model.yaml declares (`_registered_native_hz`).
PROTOCOL_HZ = 20.0

# \w, not letters: MergerIntoSlowTrafficV2 and T_Junction are Bench2Drive-220 scenarios, and a folder this
# misses is neither scored nor rejected -- it silently leaves the route set.
_ROUTE_DIR = re.compile(r"^(?P<route>\d+)-(?P<town>Town\w+?)-(?P<scenario>\w+)$")
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


def _observed_in_env(model_env: dict[str, Any]) -> float | None:
    """The between-plan frame rate a run's ``model_env`` sets (PADDOCK_OBSERVE_HZ), for runs made before
    runs recorded ``observe_hz`` -- the host set it, and the record kept it, if at all, only there."""
    raw = (model_env or {}).get("PADDOCK_OBSERVE_HZ")
    try:
        hz = float(raw) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        return None
    return hz if hz > 0 else None


def _driven_off_protocol(model_env: dict[str, Any]) -> float | None:
    """The rate this run was driven at, when that is not the benchmark's protocol.

    ``None`` for every run driven at 20 Hz, which is every run that does not say
    otherwise -- before 23 Oct 2026 the rate was only in the record when an operator put
    it there. Read only for source runs that recorded no ``infer_hz``.
    """
    raw = (model_env or {}).get("PADDOCK_INFER_HZ")
    if raw in (None, ""):
        return None
    try:
        hz = float(raw)
    except (TypeError, ValueError):
        return None
    return None if hz <= 0 or hz == PROTOCOL_HZ else hz


#: The pattern `store.scrub()` used before #175 (#173). It also matched the `/` after a template
#: variable, so it cut the directories out of a path model.yaml names at every site:
#: `{weights}/meteor/x.onnx` was stored as `{weights}x.onnx`. A record pooled then keeps that
#: `declared_env`: it is regenerated only from its raw results, which may never leave its site.
_ABSOLUTE_BEFORE_175 = re.compile(r"(?<![\w:/])/(?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.+-]*")


def _cut_before_175(value: Any) -> Any:
    """``value`` as `store.scrub()` wrote it before #175."""
    if isinstance(value, str):
        return _ABSOLUTE_BEFORE_175.sub(lambda m: Path(m.group(0)).name, value)
    if isinstance(value, dict):
        return {k: _cut_before_175(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut_before_175(v) for v in value]
    return value


def _declared_through_the_cut(runs: Sequence[Any], records: Sequence[Any]) -> tuple[list[Any], list[int]]:
    """The records' `declared_env`, each read through the pre-#175 cut where that is all that differs.

    A tracked record whose `declared_env` is not the runs' but is exactly what scrub() made of the
    runs' before #175 recorded the same model.yaml env, cut: it stands for the runs' value. Only
    when the runs agree among themselves, and only for an exact match after the cut; any other
    difference is two settings, as before. Returns the values to compare and the indices of the
    records read through the cut.
    """
    agreed = _agreed(list(runs))
    out, cut = list(records), []
    if isinstance(agreed, dict):
        for i, env in enumerate(records):
            if isinstance(env, dict) and env != agreed and env == _cut_before_175(agreed):
                out[i] = agreed
                cut.append(i)
    return out, cut


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


def _registered_ranked_with_note(model: str) -> tuple[str, ...]:
    """The settings ``model``'s model.yaml keeps ranked, as a ``paddock run`` would."""
    try:
        spec = Registry.load().models.get(model)
    except Exception:  # noqa: BLE001 -- a broken registry must not fail an ingest
        return ()
    return getattr(spec, "ranked_with_note", ()) if spec else ()


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


def _rate(value: Any) -> float | None:
    """A recorded ``infer_hz`` as Hz, 0.0 for every tick however it is written; None if not a number."""
    try:
        hz = float(value)
    except (TypeError, ValueError):
        return None
    return 0.0 if every_tick(hz) else hz


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
    # Folders whose reason came from a failure record: a later attempt cut off by a restart
    # leaves none and says nothing about the route, so it does not replace the record's reason.
    recorded: set[str] = set()
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
                # Why the attempt failed, when the harness wrote it down (failure.json): a
                # CARLA crash and a cut-off attempt both leave an empty checkpoint. Otherwise
                # the parser's words, unchanged, so a stored record ingested again keeps its
                # id. The message names the file; keep the run id and drop the path, because
                # this reason is written into a tracked run.json.
                failure = read_failure(route_dir)
                if failure is None and route_dir.name in recorded:
                    continue
                said = failure_said(failure) if failure else str(exc).split(".json: ", 1)[-1]
                rejected[route_dir.name] = f"{run_dir.name}: {said}"
                if failure is not None:
                    recorded.add(route_dir.name)
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


class TrackedRefused(ValueError):
    """A tracked record whose rows cannot stand in for its routes' ``results.json``, and why."""


class Tracked(NamedTuple):
    """A tracked record whose ``upstream_scores.json`` rows stand in for its routes' ``results.json``."""

    #: ``bench2drive/<model>/<run_id>``: how ``tracked`` names it, and where ``results/`` keeps it.
    name: str
    #: Its run.json's ``site``: where it was pooled. ``source_sites``: where its runs were, as they recorded.
    site: str
    source_sites: list[str]
    started: datetime | None
    #: Its run.json's ``config``, checked against the source runs' as theirs are against each other.
    config: dict[str, Any]
    valid: bool
    #: Of its upstream_scores.json, the file every route taken from it was measured from.
    sha256: str
    #: Route folder -> the route's metrics, measured as from its results.json.
    routes: dict[str, dict[str, Any]]
    #: Its run.json's ``superseded_by``: the run that replaced it, when one did.
    superseded_by: str = ""
    #: True when that run is a pool of this record (its ``config.source_records`` names it): the record
    #: stopped ranking because a pool of its rows replaced it, not because of anything in them.
    replaced_by_its_pool: bool = False


def _differing(mine: dict[str, Any], stored: dict[str, Any]) -> list[str]:
    """The keys whose blocks differ, but for what the rows do not hold."""
    keys = sorted((set(mine) | set(stored)) - {NOT_IN_ROWS})
    return [k for k in keys if mine.get(k) != stored.get(k)]


def read_tracked(record_dir: Path, model: str) -> Tracked:
    """A tracked record's routes, measured from its ``upstream_scores.json`` -- or :class:`TrackedRefused`.

    Each row goes through :func:`parse_checkpoint` as the one record of its route and through
    :func:`stop_shares` as its ``early_stop.json``, so a route is measured by the code that measures
    a ``results.json``. The rows hold what that needs but for three things, and the record's own
    ``metrics.json`` must pin those, or it is refused rather than guessed at (module docstring):
    how many repetitions a route had (the rows hold the first), whether a route with no early stop
    had an ``early_stop.json`` at all, and ``duration_s``, which is left out (:data:`NOT_IN_ROWS`).
    Last, the rows aggregated as the store aggregates them must give the record's ``metrics.json``,
    every block but that one: a record they do not reproduce is not one they describe.
    """
    record_dir = Path(record_dir)
    run, stored = (_read_json(record_dir / name) for name in ("run.json", "metrics.json"))
    if not isinstance(run, dict) or not isinstance(stored, dict):
        raise TrackedRefused(f"{record_dir.name}: no readable run.json and metrics.json there")
    name = f"{run.get('benchmark')}/{run.get('model')}/{run.get('run_id')}"
    if run.get("benchmark") != "bench2drive" or run.get("model") != model:
        raise TrackedRefused(f"{name} is not a Bench2Drive record of {model!r}")
    config = run.get("config")
    if not isinstance(config, dict) or not config:
        raise TrackedRefused(f"{name}: its run.json holds no config to check against the source runs'")
    try:
        raw = (record_dir / UPSTREAM_SCORES).read_bytes()
        upstream = json.loads(raw)
    except OSError as exc:
        raise TrackedRefused(
            f"{name}: no {UPSTREAM_SCORES} beside it, so nothing holds its routes' rows; "
            "`python -m tools.benchmarks.bench2drive.upstream_scores` writes it where its raw results are"
        ) from exc
    except ValueError as exc:
        raise TrackedRefused(f"{name}: its {UPSTREAM_SCORES} is not JSON ({type(exc).__name__})") from exc
    if upstream.get("record") != run.get("run_id") or upstream.get("model") != model:
        raise TrackedRefused(
            f"{name}: its {UPSTREAM_SCORES} is for {upstream.get('model')}/{upstream.get('record')}"
        )
    rows = upstream.get("routes") or []
    metrics = stored.get("metrics") or {}
    n = len(rows)
    dataset = run.get("dataset") or {}
    if n != dataset.get("n_samples"):
        raise TrackedRefused(f"{name}: the {n} routes of its {UPSTREAM_SCORES} are not the record's own")

    def pinned(key: str, **values: float) -> bool:
        block = metrics.get(key) or {}
        return (
            block.get("n") == n
            and block.get("n_missing") == 0
            and all(block.get(k) == v for k, v in values.items())
        )

    # The rows are each route's first record (upstream_scores_client.py): one scored repetition a route.
    if not pinned("n_runs", mean=1, max=1) or not pinned("n_crashed", max=0):
        raise TrackedRefused(
            f"{name}: its metrics.json does not pin one scored repetition per route (n_runs mean and max 1, "
            f"n_crashed max 0, over all {n} routes), and a row holds a route's first record only"
        )
    # A row's `ended_early: null` is a route that went to Bench2Drive's own end, or one with no
    # early_stop.json at all.
    ended = [f"ended_{reason}" for reason in early_stop.REASONS]
    stops = bool(config.get("early_stop")) or any(key in metrics for key in ended)
    if stops and not all(pinned(key) for key in ended):
        raise TrackedRefused(
            f"{name}: its metrics.json does not hold {', '.join(ended)} for every one of its {n} routes "
            f"(the early stop {'on' if config.get('early_stop') else 'off'}), and a row's ended_early null "
            "cannot say which routes had an early_stop.json"
        )
    routes: dict[str, dict[str, Any]] = {}
    split: dict[str, str] = {}
    for row in rows:
        try:
            rid, town, scenario = str(row["route_id"]), str(row["town"]), str(row["scenario"])
            record = {
                "route_id": f"RouteScenario_{rid}",
                "status": row["status"],
                "scores": {
                    "score_composed": row["driving_score"],
                    "score_route": row["route_completion"],
                    "score_penalty": row["infraction_penalty"],
                },
                "infractions": dict(row["infractions"]),
            }
            reason = row["ended_early"]
        except (KeyError, TypeError) as exc:
            raise TrackedRefused(f"{name}: a row of its {UPSTREAM_SCORES} lacks {exc}") from exc
        folder = f"{rid}-{town}-{scenario}"
        if route_parts(folder) is None or folder in routes:
            raise TrackedRefused(f"{name}: {folder} is not one route folder of its {UPSTREAM_SCORES}")
        if reason not in (None, *early_stop.REASONS) or (reason and not stops):
            raise TrackedRefused(
                f"{name}: route {rid} ended_early {reason!r}, which its metrics.json does not hold"
            )
        try:
            measured = parse_checkpoint({"records": [record]}, f"{name}/{UPSTREAM_SCORES} route {rid}")
        except NotScored as exc:
            raise TrackedRefused(str(exc)) from exc
        measured[NOT_IN_ROWS] = None
        if stops:
            measured.update(stop_shares({"reason": reason}))
        routes[folder], split[folder] = measured, scenario
    ids = [f"bench2drive/{folder}" for folder in routes]
    if sequence_hash(ids) != dataset.get("sequences_sha256"):
        raise TrackedRefused(f"{name}: the {n} routes of its {UPSTREAM_SCORES} are not the record's own")
    samples = [
        {"sample_id": f"bench2drive/{f}", "split": split[f], "metrics": routes[f]} for f in sorted(routes)
    ]
    mine = MetricsRecord.from_samples(str(run.get("run_id")), samples, primary_metric="driving_score")
    differ = _differing(mine.metrics, metrics) + [
        f"by_split {s}"
        for s in sorted(set(mine.by_split) | set(stored.get("by_split") or {}))
        if _differing(mine.by_split.get(s) or {}, (stored.get("by_split") or {}).get(s) or {})
    ]
    if differ:
        raise TrackedRefused(
            f"{name}: its {UPSTREAM_SCORES} rows, measured as a results.json is, do not give its "
            f"metrics.json ({', '.join(differ[:5])}{', ...' if len(differ) > 5 else ''})"
        )
    started = run.get("started_utc")
    try:
        when = datetime.fromisoformat(started) if started else None
    except ValueError:
        when = None
    superseded_by = str(run.get("superseded_by") or "")
    return Tracked(
        name=name,
        site=str(run.get("site") or ""),
        source_sites=[str(s) for s in config.get("source_sites") or []],
        started=when,
        config=config,
        valid=bool(run.get("valid", True)),
        sha256=hashlib.sha256(raw).hexdigest(),
        routes=routes,
        superseded_by=superseded_by,
        replaced_by_its_pool=_a_pool_of(record_dir, superseded_by, name),
    )


def _a_pool_of(record_dir: Path, superseded_by: str, name: str) -> bool:
    """Whether the run ``superseded_by`` names, beside ``record_dir`` in the store, pooled ``name``."""
    if not superseded_by:
        return False
    other, _, run_id = superseded_by.rpartition("/")
    path = record_dir.parent.parent / (other or record_dir.parent.name) / run_id / "run.json"
    try:
        sources = (json.loads(path.read_text()).get("config") or {}).get("source_records") or {}
    except (OSError, ValueError):
        return False
    return name in sources


def ingest(
    runs_root: Path | Sequence[Path],
    model: str,
    site: str,
    routes: list[str] | None = None,
    run_label: str = "pooled",
    notes: str = "",
    model_env: dict[str, str] | None = None,
    setups: dict[str, list[str]] | None = None,
    ranked: bool = False,
    tracked: Sequence[Path] = (),
) -> list[tuple[RunRecord, MetricsRecord, list[dict[str, Any]]]]:
    """One pooled paddock run for ``model`` over the routes under ``runs_root``.

    ``ranked`` ranks the run as recorded although its rate or a route that never
    scored would keep it off the board; ``notes`` says so. Every other reason a
    pool does not rank still holds.

    ``tracked`` names record directories whose ``upstream_scores.json`` rows supply
    the routes not scored under ``runs_root`` (:func:`read_tracked`, module docstring).
    """
    roots = [runs_root] if isinstance(runs_root, Path) else list(runs_root)
    records = [read_tracked(Path(d), model) for d in tracked]
    if len({rec.name for rec in records}) < len(records):
        raise TrackedRefused(f"a tracked record is named twice: {', '.join(rec.name for rec in records)}")
    setup_of = {route: label for label, group in (setups or {}).items() for route in group}
    survey = discover(roots, routes)
    rejected = dict(survey.rejected)
    # Routes set aside for every model (harness.SET_ASIDE) leave the pool before anything is read
    # from them: neither their samples nor their source runs' configs, which would otherwise join
    # the agreement checks below. Out of the routes asked for, or of every route the trees hold.
    seen = {parts[0] for folder in (*survey.scored, *survey.rejected) if (parts := route_parts(folder))}
    seen |= {parts[0] for rec in records for folder in rec.routes if (parts := route_parts(folder))}
    aside = sorted(
        (r for r in (routes if routes is not None else seen) if r in SET_ASIDE), key=lambda r: (len(r), r)
    )
    scored_aside = sorted(
        {parts[0] for folder in survey.scored if (parts := route_parts(folder)) and parts[0] in aside},
        key=lambda r: (len(r), r),
    )
    # Keyed by directory, not by run id: two roots can hold the same run id, and
    # that collision is exactly the mix this is here to catch.
    sources: dict[Path, dict[str, Any]] = {}
    contributing: list[Path] = []
    samples: list[dict[str, Any]] = []
    for folder, attempt in sorted(survey.scored.items()):
        route, town, scenario = route_parts(folder)  # type: ignore[misc]
        if route in aside:
            continue
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
        metrics.update(ended_early(attempt.results.parent))
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
    # A route no run under `runs` scored, from the tracked record that holds it: a raw route takes
    # precedence route by route, and a route two records hold is the operator's to settle, not ours.
    raw_routes = {s["native_id"]["route_id"] for s in samples}
    taken: dict[str, Tracked] = {}
    for rec in records:
        for folder, measured in sorted(rec.routes.items()):
            route, town, scenario = route_parts(folder)  # type: ignore[misc]
            if route in raw_routes or route in aside or (routes is not None and route not in routes):
                continue
            if route in taken:
                raise TrackedRefused(
                    f"route {route} is in both {taken[route].name} and {rec.name}, and scored under no run "
                    "here: name the one it comes from"
                )
            taken[route] = rec
            samples.append(
                {
                    "sample_id": f"bench2drive/{folder}",
                    "native_id": {
                        "route_id": route,
                        "town": town,
                        "scenario": scenario,
                        "record": rec.name,
                        "source": f"results/{rec.name}/{UPSTREAM_SCORES}",
                        "sha256": rec.sha256,
                    },
                    "split": scenario,
                    "group": setup_of.get(route, ""),
                    "metrics": dict(measured),
                }
            )
    # In sample-id order, as a pool of raw routes alone is: the store's float sums follow it.
    samples.sort(key=lambda s: s["sample_id"])
    used_records = [rec for rec in records if any(t is rec for t in taken.values())]
    have = {s["native_id"]["route_id"] for s in samples}
    # A route that was attempted and never scored is expected too. Defaulting
    # `wanted` to the samples that survived would make `missing` empty by
    # construction -- n_failed: 0, status complete, valid true -- on exactly the
    # case the record exists to show: a model that crashed out of a route.
    attempted = {
        parts[0]: why for f, why in rejected.items() if (parts := route_parts(f)) and parts[0] not in taken
    }
    # What a record that contributed says of a route it never scored, unless a run here says more.
    for rec in used_records:
        for route, why in (rec.config.get("routes_never_scored") or {}).items():
            if route not in have and route not in attempted:
                attempted[route] = f"{rec.name}: {why}"
    # A route set aside is neither expected nor lost: it measures no model.
    wanted = [
        r for r in (sorted(routes) if routes is not None else sorted(have | set(attempted))) if r not in aside
    ]
    missing = [r for r in wanted if r not in have]
    source_runs = sorted({s["native_id"]["run_id"] for s in samples if "run_id" in s["native_id"]})
    starts = [d for d in (_started(r) for r in source_runs) if d is not None]
    starts += [rec.started for rec in used_records if rec.started is not None]
    sample_ids = [s["sample_id"] for s in samples]

    # Configuration comes from the runs being ingested, not from this function. A tracked record's
    # config is one more source here, checked as the runs' are against each other.
    runs_used = [sources[d] for d in dict.fromkeys(contributing)]
    used = runs_used + [{"model": model, "config": rec.config, "record": rec.name} for rec in used_records]
    sources_said = "source runs and tracked records" if used_records else "source runs"
    raw_sites = sorted({rec["site"] for rec in runs_used if rec.get("site")})
    configs = [rec.get("config") or {} for rec in used]
    recorded_env: dict[str, Any] = {}
    for key in sorted({k for c in configs for k in (c.get("model_env") or {})}):
        recorded_env[key] = _agreed([(c.get("model_env") or {}).get(key) for c in configs])
    config = {
        "routes_file": _agreed([c.get("routes_file") for c in configs]) or DEFAULT_ROUTES_FILE,
        "routes_subset": wanted,
        "source_runs": source_runs,
        # Each tracked record a route came from: where it was pooled, and the file its routes were read from.
        **(
            {
                "source_records": {
                    rec.name: {"site": rec.site, "upstream_scores_sha256": rec.sha256} for rec in used_records
                }
            }
            if used_records
            else {}
        ),
        # `site` names the operation that pooled these runs; the machines that
        # produced them are here, as far as the recordings state them.
        "source_sites": sorted(set(raw_sites) | {s for rec in used_records for s in rec.source_sites}),
        "waypoint_dt": _agreed([c.get("waypoint_dt") for c in configs]) or _registered_waypoint_dt(model),
        # The operator's entry is an override of what ran, not a substitute for it.
        "model_env": {**recorded_env, **dict(model_env or {})},
        "source_config_read": f"{sum(1 for r in runs_used if r.get('config'))}/{len(runs_used)} source runs",
    }
    # The model.yaml env each source run was made under. Two values are
    # two settings of the model, not one experiment. `_agreed` skips a missing
    # value, so a run that recorded none (every run before 21 Oct 2026) is counted
    # rather than read as agreeing with the others. A tracked record pooled before #175 holds its
    # declared_env cut (`_declared_through_the_cut`); the runs' uncut value is what is recorded.
    runs_declared = [c.get("declared_env") for c in configs[: len(runs_used)]]
    records_declared, read_cut = _declared_through_the_cut(
        runs_declared, [c.get("declared_env") for c in configs[len(runs_used) :]]
    )
    declared = _agreed(runs_declared + records_declared)
    undeclared = sum(1 for c in configs if "declared_env" not in c) if declared is not None else 0
    if declared is not None:
        config["declared_env"] = declared
    # Every route that was attempted and never scored, whether or not the
    # configured subset asked for it. Route 11715 is the case: excluded for every
    # model because CARLA crashes on it, so it is not in `routes` and not in
    # `n_failed` -- and without this line `results/` would not say it happened.
    if attempted:
        config["routes_never_scored"] = dict(sorted(attempted.items()))
    # The routes set aside for every model that this pool would otherwise expect, by id: the list
    # as it stands when the pool is made, never what its source runs recorded, so every model's
    # pool says the same. Only when there are some, so a pool without any keeps its run id.
    if aside:
        config["routes_set_aside"] = aside
    # The early stop as the source runs recorded it; absent from the config of a
    # pool none of whose runs had it, so such a pool keeps its run id.
    stopped = [c.get("early_stop") for c in configs]
    stop_rules = _agreed(stopped)
    if stop_rules is not None:
        config["early_stop"] = stop_rules
    # The camera rig, in a run's config only when it is not paddock's, so a pool of
    # default-rig runs keeps its run id; a run that predates the setting had paddock's.
    rigs = _agreed([c.get("camera_rig") or rig.DEFAULT for c in configs])
    other_rig = rigs is not None and rigs != rig.DEFAULT
    if other_rig:
        config["camera_rig"] = rigs
    # How often the node was asked for a plan (`infer_hz`), judged from the runs' own record
    # and never from model.yaml as it is when the ingest runs. A run that recorded no rate was asked
    # every tick (or, before 23 Oct 2026, at a rate in `model_env`: `_driven_off_protocol`); one that kept
    # no config at all is not known. The rate the model declared ranks; the host's PADDOCK_INFER_HZ
    # in its place is recorded with the declared one beside it (`planning_hz`) and does not, nor do
    # two rates in one pool. In the config only when some run was asked slower than every tick, or
    # at the host's rate, so a pool asked every tick keeps its run id and one at 10 Hz never takes it.
    kept = [c for c in configs if c]
    unread = len(configs) - len(kept)
    read = [(c, _rate(c.get("infer_hz", 0.0))) for c in kept]
    not_a_rate = sorted({repr(c.get("infer_hz")) for c, hz in read if hz is None})
    rates = _agreed([hz for _, hz in read if hz is not None])
    host_set = [c for c, hz in read if c.get("planning_hz") is not None and hz != _rate(c["planning_hz"])]
    if rates is not None and (rates != 0.0 or host_set):
        config["infer_hz"] = rates
    if host_set:
        config["planning_hz"] = _agreed([c["planning_hz"] for c in host_set])
    mixed_rates = isinstance(rates, list)
    unknown_rate = unread if rates not in (None, 0.0) else 0
    # Frames handed to the node between plans (`observe_hz`, harness.observe_hz), as the source runs
    # recorded them. In the config only when some run had them, so a pool without keeps its run id.
    # Runs with and without them, or at two rates, read two different histories: one mean, no rank.
    # A run made before that was recorded holds the rate, if anywhere, in `model_env` (the host's
    # PADDOCK_OBSERVE_HZ, or the ingest config's statement of it): read there, so it pools with runs that
    # recorded the same rate and its record says what was fed.
    observed = _agreed(
        [
            float(c.get("observe_hz") or 0.0)
            if "observe_hz" in c
            else (_observed_in_env({**(c.get("model_env") or {}), **dict(model_env or {})}) or 0.0)
            for c in kept
        ]
    )
    if observed not in (None, 0.0):
        config["observe_hz"] = observed
    mixed_observe = isinstance(observed, list)
    # A setting the model's model.yaml declares, set to another value: what keeps a
    # `paddock run` from ranking (runner._env_override_reasons) keeps the pool from ranking.
    # Judged against the declared_env the source runs recorded, which leaves tuning keys
    # out; a key the model does not declare, and a pool that recorded none, are not judged.
    ranked_anyway = _registered_ranked_with_note(model)
    overrides = (
        sorted(
            k
            for k, v in config["model_env"].items()
            if k in declared and declared[k] != v and k not in ranked_anyway
        )
        if isinstance(declared, dict)
        else []
    )
    pooled_setups = sorted({s["group"] for s in samples if s["group"]})
    if setups:
        config["recording_setups"] = {label: sorted(group) for label, group in sorted(setups.items())}
    run_id = f"{run_label}-{sequence_hash(sample_ids)}-{config_hash(config)}"
    note = notes.strip()
    if used_records:
        origins = []
        if len(samples) > len(taken):
            origins.append(
                f"{len(samples) - len(taken)} route(s) from {len(source_runs)} run(s) under `runs`, read "
                f"from their results.json (sites recorded: {', '.join(raw_sites) or 'none'})"
            )
        for rec in used_records:
            their_sites = ", ".join(rec.source_sites) or "none"
            origins.append(
                f"{sum(1 for t in taken.values() if t is rec)} route(s) from tracked record {rec.name}, "
                f"read from its {UPSTREAM_SCORES} (sha256 {rec.sha256[:12]}; pooled at site "
                f"{rec.site or 'not recorded'}, its runs' sites recorded: {their_sites})"
            )
        note = (note + " " if note else "") + (
            f"of the {len(samples)} routes pooled: {'; '.join(origins)}. A route scored under `runs` takes "
            f"precedence over a record's row. {NOT_IN_ROWS}, the evaluator's wall-clock seconds, is not in "
            f"{UPSTREAM_SCORES}, "
            f"so the {len(taken)} route(s) from tracked records carry none and metrics.json counts them "
            "missing, as for a route --resume reused."
        )
    # A record superseded by a pool of its own rows -- this one, pooled again, or an earlier one -- stopped
    # ranking because that pool replaced it, not because of anything in its rows: pooling it again must give
    # the same run, notes and all, and a pool under another config (a route driven again, a new label) must
    # still rank. Superseded by anything else, it keeps the pool from ranking.
    unranked = [
        rec.name
        for rec in used_records
        if not rec.valid and rec.superseded_by != run_id and not rec.replaced_by_its_pool
    ]
    if unranked:
        note = (note + " " if note else "") + (
            f"tracked record(s) {', '.join(unranked)} do not rank (valid: false; their notes say why), "
            "so neither does this run."
        )
    outside = sorted(r for r in attempted if r not in set(wanted) and r not in aside)
    if outside:
        lost = ", ".join(f"{r} ({attempted[r]})" for r in outside)
        note = (note + " " if note else "") + (
            f"attempted outside the configured route subset and never scored: {lost}."
        )
    if aside:
        said = set_aside_remark(aside)
        if scored_aside:
            did = "it scored here" if len(scored_aside) == 1 else "they scored here"
            said += f"; {', '.join(scored_aside)} left out although {did}, as for every model"
        note = (note + " " if note else "") + said + "."
    if len(pooled_setups) > 1:
        note = (note + " " if note else "") + (
            f"pooled across {len(pooled_setups)} recording setups ({', '.join(pooled_setups)}); "
            "one mean describes none of them, so this run does not rank -- the per-setup "
            "figures are in metrics.json by_group."
        )
    if read_cut:
        names = ", ".join(used_records[i].name for i in read_cut)
        note = (note + " " if note else "") + (
            f"the declared_env of {names} was written by scrub() before #175, which cut the directories "
            "after a template variable ({weights}/meteor/x.onnx became {weights}x.onnx, #173); it is exactly "
            "the runs' declared_env through that cut, so it counts as the same model.yaml env and the runs' "
            "value is recorded."
        )
    if isinstance(declared, list):
        note = (note + " " if note else "") + (
            f"source runs were made under {len(declared)} different model.yaml envs "
            "(config.declared_env); one mean describes none of them, so this run does not rank."
        )
    if undeclared:
        note = (note + " " if note else "") + (
            f"{undeclared} of {len(used)} {sources_said} recorded no declared_env (made before 21 Oct 2026), "
            "so which model.yaml env their routes ran is not known and this run does not rank."
        )
    # The override is layered on top of what the runs recorded, so a key that is
    # only in the override is an operator's assertion -- the note has to say so,
    # or `results/` claims the source runs recorded a value they never held.
    overridden = sorted(k for k in dict(model_env or {}) if k not in recorded_env)
    unverified = sum(1 for rec in runs_used if not rec.get("model"))
    if unverified:
        note = (note + " " if note else "") + (
            f"{unverified} of {len(runs_used)} source runs kept no run.json, so their routes are "
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
    # Runs that recorded no rate (before 23 Oct 2026) were driven at PADDOCK_INFER_HZ in `model_env`;
    # at the rate the model's own model.yaml declares, that is on protocol, and ranks.
    driven = None if any("infer_hz" in c for c in kept) else _driven_off_protocol(config["model_env"])
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
        declared_rate = (
            f"the {native[0]:g} Hz {model}/model.yaml declares ({native[1]})"
            if native
            else (f"a rate {model}/model.yaml declares (it declares none)")
        )
        note = (note + " " if note else "") + (
            f"driven at {off_protocol:g} Hz, neither the {PROTOCOL_HZ:g} Hz the benchmark's published "
            f"protocol calls the model at nor {declared_rate}, so this run is kept and does not rank -- "
            "the paired comparison is docs/benchmarks/bench2drive-native-rate.md."
        )
    if overrides:
        note = (note + " " if note else "") + (
            f"source runs set {', '.join(overrides)}, which the model's model.yaml declares, to other "
            "values (config.model_env), so this run does not rank, as a paddock run of them would not."
        )
    if not_a_rate:
        note = (note + " " if note else "") + (
            f"source runs recorded infer_hz {', '.join(not_a_rate)}, which is not a rate, so this run "
            "does not rank."
        )
    if host_set:
        note = (note + " " if note else "") + (
            f"{len(host_set)} of {len(used)} {sources_said} were asked for a plan at the rate the host's "
            "PADDOCK_INFER_HZ set (config.infer_hz, 0 for every tick), where the model's model.yaml "
            "declared config.planning_hz: an operator's rate, so this run does not rank."
        )
    if mixed_rates:
        note = (note + " " if note else "") + (
            f"source runs were asked for a plan at different rates (config.infer_hz {rates}, 0 for every "
            "tick); one mean describes none of them, so this run does not rank."
        )
    if mixed_observe:
        note = (note + " " if note else "") + (
            f"source runs were handed frames between plans at different rates (config.observe_hz {observed}, "
            "0 for none); one mean describes none of them, so this run does not rank."
        )
    if unknown_rate:
        note = (note + " " if note else "") + (
            f"{unread} of {len(used)} {sources_said} kept no config, so the rate their routes were asked at "
            "is not known, and this run does not rank."
        )
    if rates not in (None, 0.0) and not (mixed_rates or host_set or not_a_rate or unknown_rate):
        note = (note + " " if note else "") + (
            f"asked for a plan at {rates:g} Hz, the planning_hz the model's model.yaml declared for these "
            "runs, the last plan followed in between."
        )
    if other_rig:
        said = (
            f"camera rigs differ between source runs ({', '.join(rigs)})"
            if isinstance(rigs, list)
            else f"driven through the {rigs!r} camera rig"
        )
        note = (note + " " if note else "") + (
            f"{said}, not the {rig.DEFAULT!r} rig every model here is compared through, "
            "so this run does not rank."
        )
    # A route the early stop ended is a failed drive, and the pool ranks with it;
    # routes stopped under different rules, or some with the rules and some without,
    # are as much one experiment as two setups.
    without = sum(1 for rules in stopped if not rules) if stop_rules is not None else 0
    mixed_stops = isinstance(stop_rules, list) or bool(without)
    # Ending a route at its first collision is not Bench2Drive's score: the run would not rank.
    at_collision = isinstance(stop_rules, dict) and bool(stop_rules.get("collision"))
    if at_collision:
        note = (note + " " if note else "") + early_stop.collision_reason() + "."
    # A route someone ended by hand was scored at a moment a person chose: kept, named, not ranked.
    by_hand = sum(1 for sample in samples if sample["metrics"].get(f"ended_{early_stop.MANUAL}"))
    if by_hand:
        note = (note + " " if note else "") + early_stop.manual_reason(by_hand) + "."
    if stop_rules is not None:
        mixed = isinstance(stop_rules, list)
        # Counted over the routes pooled here: the runs recorded the site's routes with no expert drive.
        said = early_stop.mixed_reason(stop_rules, wanted) if mixed else early_stop.remark(stop_rules, wanted)
        note = (note + " " if note else "") + said + "."
        if without:
            note += (
                f" {without} of {len(stopped)} {sources_said} ran without it; their routes went to "
                "Bench2Drive's own end, so this run does not rank."
            )
    if missing:
        lost = ", ".join(f"{r} ({attempted.get(r, 'no attempt found')})" for r in missing)
        note = (note + " " if note else "") + f"routes expected but never scored: {lost}."
    if ranked and (missing or off_protocol is not None):
        note = (note + " " if note else "") + (
            f"ranked = true in the ingest config; the mean is over the {len(samples)} routes that scored."
        )
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
        # published as a number nobody drove. A run driven at a rate its model does
        # not declare is a different protocol, and ranking it beside runs at the
        # model's own rate says they measure the same thing -- which is the claim
        # the native-rate campaign exists to deny.
        # Source runs under different, or unrecorded, model.yaml envs are the
        # first case again: two settings of the model behind one mean.
        # An early stop ranks; a pool of routes stopped under different rules does not.
        # A declared setting run at another value, or another camera rig, is what keeps a
        # single run from ranking, and so the pool of them. A pool asked at the rate its model
        # declared ranks; at the host's rate, at two rates, or at a rate not known or not a
        # number, it does not. `ranked` lifts only a lost route and an off-protocol rate.
        valid=(bool(ranked) or (not missing and off_protocol is None))
        and len(pooled_setups) <= 1
        and not isinstance(declared, list)
        and not undeclared
        and not overrides
        and not other_rig
        and not mixed_rates
        and not mixed_observe
        and not host_set
        and not unknown_rate
        and not not_a_rate
        and not mixed_stops
        and not at_collision
        and not by_hand
        and not unranked,
        notes=note,
        # A relative tail only: results/ is tracked and must not name a machine's disk.
        ingested_from="; ".join(
            ["/".join(r.parts[-3:]) for r in roots] + [f"results/{rec.name}" for rec in records]
        ),
    )
    metrics_record = MetricsRecord.from_samples(run_id, samples, primary_metric="driving_score")
    return [(run, metrics_record, samples)]
