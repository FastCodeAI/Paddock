"""Bench2Drive closed-loop benchmark (CARLA 0.9.15 / Thinklab-SJTU leaderboard).

Bench2Drive drives the model through CARLA on the official leaderboard's 220
short routes (Towns 12/13/15, 44 scenario types x 5). The model runs behind
paddock's HTTP policy node; this harness spawns the leaderboard evaluator once
per route, points its AutonomousAgent (``agent.py``, model-agnostic) at that
node, and reads the driving score CARLA writes to ``results.json``.

Closed-loop, like neuroNCAP: the model decides what happens next and errors
compound. The shape mirrors ``neuroncap/harness.py`` deliberately --

* ``preflight`` checks the CARLA install, the checkout, the routes file and
  that the CARLA-client interpreter can import everything the agent needs, so
  a missing asset is reported in seconds rather than an hour into a sweep;
* ``episodes`` enumerates routes from the leaderboard routes XML;
* ``run_episode`` shells out to ``leaderboard_evaluator.py`` with the node URL
  and parses the ``results.json`` it produces.

Three things the evaluator does that shape the code below (all read from the
0.0.4 checkout, ``leaderboard_evaluator.py`` / ``utils/statistics_manager.py``):

* it **self-launches CARLA** from ``CARLA_ROOT`` in its own session, on the
  first free port at or above ``--port``, and reaps it with an ``atexit`` hook
  -- which never runs if the evaluator itself is killed;
* it **writes the checkpoint at startup**, so ``results.json`` existing proves
  nothing, and it **still fills in scores for a crashed route**, marking the
  crash only in the record's ``status``;
* it exits -1 (status 255) when the scenario cannot be loaded, the agent or
  its sensors cannot be set up, or the simulation crashes; an agent crash
  mid-route exits 0.

CARLA and the Bench2Drive checkout are vendored outside the package; the
``[benchmarks.bench2drive]`` section of ``paddock.toml`` says where they live
(see ``docs/benchmarks/bench2drive.md``).
"""

from __future__ import annotations

import functools
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Collection, Iterable, Iterator
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.bench2drive import early_stop, rig
from paddock.benchmark import ClosedLoopBenchmark, Episode
from paddock.config import REPO_ROOT
from paddock.policy.protocol import grid_stride
from paddock.runtime import ports
from paddock.runtime.env import PASSTHROUGH
from paddock.runtime.proc import tail as _tail

#: The model-agnostic CARLA client shipped with this adapter. The evaluator loads
#: it via ``--agent``; it reads its settings from PADDOCK_* variables set by
#: ``run_episode`` below.
AGENT = Path(__file__).parent / "agent.py"

#: Default per-route wall-clock budget. The evaluator boots CARLA afresh for
#: every route (about ten minutes of map load on the site measured, a little
#: more at default render quality), and a long route then rides CARLA's
#: 4000-frame cap: 200 s of game time, which a model at 0.05x real-time
#: (Alpamayo) turns into ~4000 s of wall clock. 1800 s timed out every such
#: route and recorded nothing; the budget bounds hangs, so err generous.
#: Override per site with ``per_route_timeout_s``.
DEFAULT_ROUTE_TIMEOUT_S = 5400


def infer_hz(planning_hz: float | None) -> float:
    """How often the client asks for a plan, Hz; 0 means every tick.

    The model's `planning_hz`, unless the host sets PADDOCK_INFER_HZ, which
    drives every model at that rate (the native-rate campaign set it by hand).
    """
    raw = os.environ.get("PADDOCK_INFER_HZ")
    if not raw:
        return float(planning_hz or 0.0)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"PADDOCK_INFER_HZ={raw!r}: expected a rate in Hz") from None


#: Bench2Drive steps its simulation at 20 Hz, so a client asked at that rate or faster, or at
#: none (0), asks on every tick: one protocol however the rate is written.
SIM_HZ = 20.0


def every_tick(hz: float) -> bool:
    """Whether a client asked at ``hz`` asks the node on every simulation tick."""
    return hz <= 0 or hz >= SIM_HZ


def observe_hz(native_input_hz: float | None, asked_hz: float) -> float:
    """How often the client hands the node a frame between plans (POST /observe), Hz; 0 never.

    A model asked for a plan slower than the rate its history is built on (``native_input_hz``) is
    handed a frame at that rate in between, so its history holds real frames, not the planning
    rate's. Qwen-Drive plans at 2 Hz and reads sixteen ego poses on a 0.1 s grid: asked at 2 Hz
    with nothing in between, its pose history came 5x sparser than it was trained on. A model asked
    every tick (``asked_hz`` 0), or at its own input rate, never observes.
    """
    native = float(native_input_hz or 0.0)
    return native if asked_hz > 0 and native > asked_hz and not _same_rate(native, asked_hz) else 0.0


def _same_rate(a: float, b: float) -> bool:
    """Two rates that ask the node on the same ticks."""
    return (every_tick(a) and every_tick(b)) or a == b


def _rate_said(hz: float) -> str:
    return "on every tick" if every_tick(hz) else f"at {hz:g} Hz"


def _cameras_override() -> list[str]:
    """The cameras the host's PADDOCK_CAMERAS_OVERRIDE renders and ships in place of the model's
    own (a sim2real diagnostic); empty entries are dropped, so a trailing comma is not a camera id."""
    return [c for c in os.environ.get("PADDOCK_CAMERAS_OVERRIDE", "").split(",") if c.strip()]


#: The CARLA launch flag that changes what the cameras see. Bench2Drive 7ec25d1
#: launches CARLA at its default (Epic) quality; ``-quality-level=Low`` renders
#: the sky black and the wet-road decal layer as white speckle, so frames scored
#: that way are not the benchmark's frames. It is a one-line local edit to the
#: evaluator that leaves no trace in run.json, hence a preflight refusal.
QUALITY_FLAG = "-quality-level"

#: How the evaluator reads with patches/Bench2Drive/ applied. Unpatched, a crashed
#: route kills every CARLA its user runs on the GPU (the `kill -9` line), and a
#: server slower than the evaluator's 30 s sleep costs a 600 s wait before the
#: connection retry (`--connect-timeout` is the fix).
CONNECT_TIMEOUT_FLAG = "--connect-timeout"
KILL_ALL_LINE = "xargs -r kill -9"
#: The scenario_runner file patch 0003 changes, and what it adds. Unpatched, a flow car that misses
#: its sink keeps its collision sensor until a large map puts it to sleep 650 m from the hero, and
#: CARLA 0.9.15 segfaults (route 11715 for every model).
ATOMIC_BEHAVIORS = Path("scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py")
FAR_CAR_GUARD = "def _remove_far_cars("

#: Seconds each connection attempt to a just-launched CARLA may take before the
#: evaluator asks again (patch 0002). It bounds one attempt, not the boot: the
#: evaluator makes up to 20. Override per site with ``connect_timeout_s``. Not in
#: ``run_config``: it decides only how soon a server that is not up yet is asked
#: again, never what is measured.
DEFAULT_CONNECT_TIMEOUT_S = 60

#: The grid the controller reads a plan on: the plan's own points at 0.5, 1.0, ...
#: 3.0 s (``pid.on_grid``), whatever the model's spacing. Every model but
#: alpamayo2edge plans 0.5 s apart, so for them these are the plan's first six
#: points, which is all the agent ever read; Edge's are every fifth of its 0.1 s
#: points. The agent turns their spacing into a target speed, so it is a
#: measurement parameter, not a knob. ``input_problems`` refuses a model whose
#: spacing has no points on this grid.
WAYPOINT_DT = 0.5

DEFAULT_ROUTES_FILE = "bench2drive220.xml"

#: The file that, appearing in a route's directory while it drives, ends the route
#: by hand (``early_stop.MANUAL``). The dashboard's "Stop route" button writes it.
STOP_REQUEST = "stop_request"

#: Why an attempt that scored nothing failed, beside its ``results.json``
#: (:func:`classify_failure`). A CARLA crash and an attempt cut off by a restart both
#: leave an empty checkpoint, so without this only the logs could tell them apart -- and
#: only a crash is evidence that a route cannot be run. An interrupted attempt writes none.
FAILURE = "failure.json"

#: What the campaign guard writes into a route's directory just before it kills the
#: route's evaluator, so the failure record can say why the evaluator died.
GUARD_KILL = "guard.json"

#: The leaderboard's statuses for a failure on the agent's side: the paddock client or the
#: model's node (``CRASH_STATUSES``), never the route's.
MODEL_SIDE = ("Agent crashed", "Agent couldn't be set up", "Agent's sensors were invalid")

#: CARLA's crash handler's line, and the agent's per-tick line, in ``evaluator.log``.
_CRASH_SIGNAL = re.compile(r"CommonUnixCrashHandler: Signal=(\d+)")
_GAME_TIME = re.compile(r"Game time = ([0-9.]+)")

#: One expert drive's summary, ``<expert>/<route>/expert_traj.json`` as
#: ``tools.benchmarks.bench2drive.expert summarize`` writes it: where a route's time budget comes from.
EXPERT_TRAJ = "expert_traj.json"

#: CARLA's client library as the 0.9.15 release ships it. Put on the client's
#: path when present, as the checkout's own ``run_evaluation.sh`` does, so a
#: site need not also ``pip install carla`` -- unless the interpreter has a
#: ``carla`` of its own (``_has_own_carla``).
CARLA_EGG = Path("PythonAPI") / "carla" / "dist" / "carla-0.9.15-py3.7-linux-x86_64.egg"

#: ``statistics_manager.FAILURE_MESSAGES`` in Bench2Drive 0.0.4 (7ec25d1). A
#: record whose status ends in one of these describes a route that did not
#: run, whatever its ``scores`` say. Everything else the manager writes --
#: ``Perfect``, ``Completed``, ``Failed - Agent got blocked``, ``Failed - Agent
#: deviated from the route``, ``Failed - Agent timed out``, ``Failed -
#: TickRuntime`` (the 4000-frame cap) -- is the benchmark working, and scores.
CRASH_STATUSES = frozenset(
    {
        "Simulation crashed",
        "Agent's sensors were invalid",
        "Agent couldn't be set up",
        "Agent crashed",
    }
)

#: Routes left out of every model's numbers, with the reason. Each failed in the simulator
#: on every attempt by whichever model drives (11715), or by Alpamayo 1.5 at its declared
#: 10 Hz (3800, 23695, 24071), so it measures no model, and a mean with it for one
#: model and without it for another would not be over the same routes. Poor
#: driving is never a reason: a route the car is blocked on, deviates from, times out on or
#: has ended early scores. Nor is a model's own node failing (an HTTP 500, out of memory):
#: that is the model's to fix. A route joins after three attempts that failed in the
#: simulator (``failure.json``, ``campaign --tries``), with its evidence in
#: docs/benchmarks/bench2drive.md, "Routes set aside". ``--scene <id>`` still drives one, to
#: see whether it still fails; a run that does is not a score (``invalid_reasons``).
SET_ASIDE: dict[str, str] = {
    # 11715 (Town12 EnterActorFlow) was the first, from #159: CARLA's Signal=11 about 54 s in, on
    # every attempt. Under gdb that was a flow car with a collision sensor put to sleep 650 m from
    # the hero (ALargeMapManager::ConvertActiveToDormantActors -> ASensor::EndPlay). Patch 0003
    # removes such a car first, as its sink would; with it 11715 scored for the three models of the
    # 220-route sweep (Drive-JEPA, Qwen-Drive, VisionPilot; 8 Nov 2026).
    # Three Town13 routes then crashed CARLA the same way at another site (3800, 23695, 24071: every
    # attempt by Alpamayo 1.5 at its declared 10 Hz, 9 Nov 2026), and the four 216-route records made
    # there set all four aside. Under patch 0003 the four routes scored for Alpamayo 1.5, AutoE2E,
    # METEOR and openpilot here (12 Nov 2026), and for UniAD and VAD in their 220-route sweep.
}

#: Everything ``agent.py`` imports, checked in the client interpreter by
#: ``preflight`` so a missing package fails before CARLA boots.
AGENT_IMPORTS = (
    "carla",
    "cv2",
    "numpy",
    "requests",
    "agents.navigation.local_planner",
    "leaderboard.autoagents.autonomous_agent",
    "srunner.scenariomanager.carla_data_provider",
    "paddock.policy.protocol",
    "paddock.adapters.benchmarks.bench2drive.pid",
)

_RPC_PORT = re.compile(r"-carla-rpc-port=(\d+)")


class NotScored(ValueError):
    """``results.json`` holds no score for the route: empty, or the route crashed."""


# ---------------------------------------------------------------------------
# Pure helpers, golden-tested without CARLA
# ---------------------------------------------------------------------------


def routes_path(repo: Path, routes_file: str | None = None) -> Path:
    """Where a Bench2Drive checkout keeps a routes file: ``leaderboard/data/<name>``."""
    return Path(repo) / "leaderboard" / "data" / (routes_file or DEFAULT_ROUTES_FILE)


def parse_routes(routes_xml: Path) -> list[tuple[str, str, str]]:
    """``(route_id, town, scenario_type)`` for every route in a leaderboard XML.

    Pure, so the golden test can enumerate a fixture with no CARLA involved.
    """
    tree = ET.parse(routes_xml)
    routes: list[tuple[str, str, str]] = []
    for route in tree.getroot().findall("route"):
        rid = route.get("id") or ""
        town = route.get("town") or ""
        scenario = route.find("scenarios/scenario")
        stype = scenario.get("type") if scenario is not None else "route"
        routes.append((rid, town, stype))
    return routes


def select_routes(
    routes: list[tuple[str, str, str]],
    scenes: list[str] | None = None,
    scenarios: list[str] | None = None,
    subset: list[str] | None = None,
    set_aside: Collection[str] = (),
) -> Iterator[tuple[str, str, str]]:
    """Filter routes by ``--scene`` (route id or town), ``--scenarios`` (type) and
    an optional site ``subset`` (route ids, from ``[benchmarks.bench2drive]
    routes_subset``) that restricts the run to a chosen set -- e.g. one route per
    scenario category -- without editing the routes file.

    ``scenes=None`` (or empty) means every route: the runner passes ``None``
    when ``--scene`` was not given, the same contract every closed-loop
    benchmark gets.

    A route in ``set_aside`` (:data:`SET_ASIDE`) is driven only when ``--scene``
    names its id: naming its town, or nothing, means the routes that are scored.

    Pure and separate from :meth:`Bench2DriveBenchmark.episodes` so the selection
    logic is unit-testable without a site config.
    """
    subset_ids = set(subset) if subset else None
    scene_filter = set(scenes) if scenes else None
    scenario_filter = set(scenarios) if scenarios else None
    for rid, town, stype in routes:
        if subset_ids is not None and rid not in subset_ids:
            continue
        if rid in set_aside and (scene_filter is None or rid not in scene_filter):
            continue
        if scene_filter is not None and rid not in scene_filter and town not in scene_filter:
            continue
        if scenario_filter is not None and stype not in scenario_filter:
            continue
        yield rid, town, stype


def patch_problems(evaluator_source: str, evaluator: Path) -> list[str]:
    """Why ``evaluator_source`` is not the evaluator with patches/Bench2Drive/ applied.

    Both patches matter beyond one run: without 0001 a crashed route kills every
    CARLA its user runs on the GPU, a parallel sweep worker's or another gate
    lane's included; without 0002 the harness passes a flag the evaluator rejects.
    """
    problems = []
    if KILL_ALL_LINE in evaluator_source:
        problems.append(
            f"{evaluator} still kills every CARLA on its GPU when a route crashes; apply "
            f"patches/Bench2Drive/0001-kill-only-own-carla-on-crash.patch (patches/README.md)"
        )
    if CONNECT_TIMEOUT_FLAG not in evaluator_source:
        problems.append(
            f"{evaluator} has no {CONNECT_TIMEOUT_FLAG}; apply "
            f"patches/Bench2Drive/0002-connect-timeout-for-a-booting-server.patch (patches/README.md)"
        )
    return problems


def scenario_patch_problems(behaviors_source: str, behaviors: Path) -> list[str]:
    """Why ``behaviors_source`` is not scenario_runner's atomic_behaviors.py with patch 0003 applied.

    Without it CARLA crashes on route 11715, which is no longer set aside, and on any flow route
    where a car that missed its sink gets 650 m from the hero (patches/README.md).
    """
    if FAR_CAR_GUARD in behaviors_source:
        return []
    return [
        f"{behaviors} has no far-car guard in ActorFlow, so CARLA crashes when a large map puts a flow "
        f"car with a collision sensor to sleep (route 11715 every time); apply "
        f"patches/Bench2Drive/0003-actorflow-remove-far-flow-cars-before-dormancy.patch (patches/README.md)"
    ]


def crash_reason(status: str) -> str | None:
    """The failure message if ``status`` marks a route that did not run, else None."""
    for reason in CRASH_STATUSES:
        if status.endswith(reason):
            return reason
    return None


def set_aside_remark(routes: Iterable[str]) -> str:
    """The note a run, or a pool, carries for the routes :data:`SET_ASIDE` left out of it; "" for none.

    One wording for both, as ``early_stop.remark`` is: the ids are in the config, and why
    they are out is said here, so rewording a reason moves no run id.
    """
    ids = sorted(set(routes), key=lambda r: (len(r), r))
    if not ids:
        return ""
    said = "; ".join(f"{r}: {SET_ASIDE[r]}" for r in ids)
    return (
        f"{len(ids)} route(s) set aside for every model ({said}): a route the simulator fails on "
        "every time measures no model, so it is in no mean and not counted as lost"
    )


def parse_results(results_json: Path) -> dict[str, float]:
    """A Bench2Drive ``results.json`` -> paddock metrics for one route.

    One record per repetition (``--repetitions N`` gives N); metrics are the
    mean over them and ``n_runs`` says how many, as neuroNCAP's collector does.
    The aggregate ``global_record`` is ignored.

    A record whose ``status`` is one of :data:`CRASH_STATUSES` is not a
    measurement: the leaderboard fills in ``score_composed`` for a crashed route
    too, and a zero that came from a stack trace must not average into the
    leaderboard as if the car had driven into a wall. Such records are
    **dropped**: with ``--runs 3`` one crashed repetition must not discard two
    that drove. ``n_runs`` counts the repetitions that scored and ``n_crashed``
    the ones that did not, so a thin average is visible as thin. Raises
    :class:`NotScored` when there is no record, or when every record crashed.
    Pure, so the golden test asserts it reproduces the score CARLA recorded.
    """
    checkpoint = json.loads(Path(results_json).read_text()).get("_checkpoint", {})
    return parse_checkpoint(checkpoint, str(results_json))


def parse_checkpoint(checkpoint: dict[str, Any], where: str) -> dict[str, float]:
    """:func:`parse_results` on a ``_checkpoint`` already read; ``where`` names it in a refusal.

    The ingest measures a route taken from a tracked record's ``upstream_scores.json`` row
    through this too, so the two are one derivation, not two kept in step.
    """
    all_records = checkpoint.get("records") or []
    if not all_records:
        raise NotScored(f"{where}: no records in _checkpoint (the route did not score)")

    records, crashed = [], []
    for record in all_records:
        status = str(record.get("status", ""))
        if crash_reason(status):
            crashed.append(f"{record.get('route_id', 'route')} status {status!r}")
        else:
            records.append(record)
    if not records:
        raise NotScored(
            f"{where}: {'; '.join(crashed)} -- the route did not run, so its scores are not a measurement"
        )

    def count(infractions: dict, key: str) -> int:
        value = infractions.get(key, [])
        return len(value) if isinstance(value, list) else int(value or 0)

    driving, completion, penalty, collided = [], [], [], []
    for record in records:
        scores = record.get("scores", {})
        infractions = record.get("infractions", {})
        driving.append(float(scores.get("score_composed", 0.0)))
        completion.append(float(scores.get("score_route", 0.0)))
        penalty.append(float(scores.get("score_penalty", 0.0)))
        collision_keys = ("collisions_vehicle", "collisions_pedestrian", "collisions_layout")
        collisions = sum(count(infractions, k) for k in collision_keys)
        collided.append(1.0 if collisions > 0 else 0.0)

    n = len(records)
    return {
        "driving_score": round(sum(driving) / n, 4),
        "route_completion": round(sum(completion) / n, 4),
        "infraction_penalty": round(sum(penalty) / n, 4),
        # Per-route 0/1 (a fraction over repetitions); paddock aggregates it to a
        # rate across routes, as neuroNCAP does.
        "collision_rate": round(sum(collided) / n, 4),
        "n_runs": n,
        "n_crashed": len(crashed),
    }


def ended_early(route_dir: Path) -> dict[str, float]:
    """``ended_<reason>`` for one route (stuck, over_time, collision, manual): share of repetitions so ended.

    Read from the ``early_stop.json`` the agent writes beside ``results.json``
    whenever the early stop is on -- with ``"reason": null`` for a route that went
    to Bench2Drive's own end -- so this reads only the route's directory, as
    ``--resume`` and the ingest need. Empty without one: the stop was off, and a
    0 would claim a rule was there to fire. Each repetition's verdict is in its
    ``repetitions`` (a record from before those has only the one); with one
    repetition it is 0/1 per route, so a run's aggregate is the share of its
    routes each rule ended.
    """
    try:
        record = json.loads((Path(route_dir) / early_stop.RECORD).read_text())
    except (OSError, ValueError):
        return {}
    return stop_shares(record)


def stop_shares(record: Any) -> dict[str, float]:
    """:func:`ended_early` on an ``early_stop.json`` already read; empty for anything but a record."""
    if not isinstance(record, dict):
        return {}
    reps = [r for r in (record.get("repetitions") or [record]) if isinstance(r, dict)] or [record]
    return {
        f"ended_{name}": sum(1.0 for r in reps if r.get("reason") == name) / len(reps)
        for name in early_stop.REASONS
    }


def classify_failure(
    log_text: str, *, timed_out: bool, returncode: int | None, results_json: Path
) -> dict[str, Any]:
    """What ended an attempt that scored nothing, from its ``evaluator.log`` and ``results.json``.

    ``kind`` is the first that applies: ``carla_crash`` (CARLA's own crash handler is in
    the log, with its signal), ``timeout``, ``evaluator_exit`` (any other non-zero exit),
    ``no_results`` or ``not_scored`` (the leaderboard's statuses, in ``status``). The crash
    handler is read first, however the attempt then ended: after a crash the evaluator
    deadlocks in its own shutdown, so the process ends by the guard's kill (exit -15) or
    the timeout, and either alone would hide the crash. ``game_time_s`` is the last tick
    the agent logged, when it logged one.
    """
    record: dict[str, Any] = {"timed_out": timed_out, "returncode": returncode}
    times = _GAME_TIME.findall(log_text)
    if times:
        record["game_time_s"] = float(times[-1])
    # The leaderboard's own verdict, whenever it wrote one: it exits -1 for a scenario that
    # would not load, an agent or sensors it could not set up and a simulation crash alike,
    # so the exit code alone cannot say whose failure it was.
    try:
        records = (json.loads(results_json.read_text()).get("_checkpoint") or {}).get("records") or []
    except (OSError, ValueError, AttributeError):
        records = []
    statuses = sorted({str(r.get("status", "")) for r in records if isinstance(r, dict)})
    if statuses:
        record["status"] = statuses
    crash = _CRASH_SIGNAL.search(log_text)
    if crash:
        record["kind"], record["signal"] = "carla_crash", int(crash.group(1))
    elif timed_out:
        record["kind"] = "timeout"
    elif returncode not in (0, None):
        record["kind"] = "evaluator_exit"
    elif not results_json.exists():
        record["kind"] = "no_results"
    else:
        record["kind"] = "not_scored"
    return record


def node_alive(policy_url: str, timeout_s: float = 3.0) -> bool:
    """Whether the model's node still answers ``/alive``.

    Asked after an attempt fails: with the node gone, the next routes of the same run fail too, as
    "Agent crashed", and they failed the node, not the route -- so they use up none of its tries.
    """
    try:
        with urllib.request.urlopen(f"{policy_url.rstrip('/')}/alive", timeout=timeout_s) as answer:
            return answer.status == 200
    except (OSError, ValueError):
        return False


def failure_said(record: dict[str, Any]) -> str:
    """One line for a failure record, naming no path: it reaches tracked notes."""
    kind = record.get("kind")
    at = f" at {record['game_time_s']:g} s of game time" if record.get("game_time_s") is not None else ""
    statuses = [s for s in record.get("status") or [] if s]
    status = f" ({', '.join(statuses)})" if statuses else ""
    if kind == "carla_crash":
        said = f"CARLA crashed (signal {record.get('signal')}){at}{status}"
    elif kind == "timeout":
        said = f"timed out (per_route_timeout_s){at}{status}"
    elif kind == "evaluator_exit":
        said = f"evaluator exited {record.get('returncode')}{status}{at}"
    elif kind == "no_results":
        said = "evaluator exited 0 but wrote no results.json"
    else:
        said = f"not scored{status}" if statuses else "no records in _checkpoint"
    if record.get("node_alive") is False:
        said += "; the model's node was not answering"
    guard = record.get("guard")
    return f"{said}; the guard killed it: {guard}" if guard else said


def simulator_side(record: dict[str, Any]) -> bool | None:
    """Whether a failure is the simulator's (True), the model's or its agent's (False), or unclear (None).

    Only the simulator's count toward setting a route aside: CARLA's crash handler on the
    route, the leaderboard's ``Simulation crashed``, or the guard finding the route's CARLA
    gone. CARLA's crash handler is read first, because after a crash the agent's next call
    can surface as ``Agent crashed``. A crash before the first tick says more about the
    machine than the route, so it is unclear. ``Agent crashed``, ``couldn't be set up`` and
    ``sensors were invalid`` are the client or the model's node -- the model's to fix, never
    a reason to take the route away from every model.
    """
    statuses = record.get("status") or []
    if record.get("kind") == "carla_crash":
        return True if record.get("game_time_s") is not None else None
    if any(s.endswith(MODEL_SIDE) for s in statuses):
        return False
    if any(s.endswith("Simulation crashed") for s in statuses) or record.get("guard_carla_gone"):
        return True
    return None


def read_failure(route_dir: Path) -> dict[str, Any] | None:
    """The failure record beside a route's ``results.json``; None if it scored, was cut off or predates it."""
    try:
        record = json.loads((Path(route_dir) / FAILURE).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def write_failure(route_dir: Path, record: dict[str, Any]) -> None:
    """Write ``record`` as the route's failure record, whole or not at all.

    Atomic, because the campaign counts these to decide when a route is given up on, and
    a half-written file read at the wrong moment would count as no failure.
    """
    target = Path(route_dir) / FAILURE
    partial = target.with_name(target.name + ".tmp")
    partial.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(partial, target)


def _positive(value: Any) -> float | None:
    """A site setting as a number above zero, or None for anything else."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


# ---------------------------------------------------------------------------
# Process hygiene: we kill only what we started
# ---------------------------------------------------------------------------


def carla_port_from_log(log_path: Path, fallback: int) -> int:
    """The port CARLA was actually started on.

    The evaluator moves up from ``--port`` to the first free one and prints the
    launch command; that line is the only record of the port it chose.
    """
    if log_path.exists():
        match = _RPC_PORT.search(log_path.read_text(errors="replace"))
        if match:
            return int(match.group(1))
    return fallback


def carla_pids(port: int) -> list[int]:
    """PIDs whose command line carries exactly ``-carla-rpc-port=<port>``.

    That token appears only on a CARLA server (and the shell that launched it),
    and only one server can hold a port -- so this names the server *this*
    evaluator launched and nothing else on the host. Never by process name:
    ``CarlaUE4-Linux-Shipping`` is every CARLA on the machine.
    """
    token = re.compile(rf"(?:^|\s)-carla-rpc-port={port}(?:\s|$)")
    me = os.getpid()
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:  # it exited between listing and reading
            continue
        if token.search(cmdline):
            found.append(int(entry.name))
    return found


def kill_group(proc: subprocess.Popen) -> None:
    """Take the evaluator and everything in its session down, TERM then KILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def reap_carla(port: int, wait_s: float = 30.0) -> list[int]:
    """Kill the CARLA server on ``port`` if it outlived its evaluator.

    On a normal exit the evaluator's own ``atexit`` has already done this; after
    a timeout kill nothing has, and CARLA sits in its own session where the
    evaluator's process group cannot reach it. Polls the port free rather than
    sleeping a fixed time, so the next route neither races the old server nor
    waits longer than it must.
    """
    pids = carla_pids(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.time() + wait_s
    while pids and not ports.is_free(port) and time.time() < deadline:
        time.sleep(1.0)
    return pids


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


#: The three settings every run needs, and the command that makes each.
_MAKES_SETTING = {
    "python": "`paddock setup --bench bench2drive` builds a CARLA-client interpreter",
    "carla_root": "`paddock data fetch carla` downloads CARLA 0.9.15 and its maps",
    "repo": "`paddock setup --bench bench2drive` clones Bench2Drive at its pin",
}


class Bench2DriveBenchmark(ClosedLoopBenchmark):
    # ---- site settings: `self.setting(key)` / `self.settings` from the base ----

    @property
    def repo(self) -> Path | None:
        value = self.setting("repo")
        return Path(value) if value else None

    @property
    def carla_root(self) -> Path | None:
        value = self.setting("carla_root")
        return Path(value) if value else None

    @property
    def routes_file(self) -> Path | None:
        repo = self.repo
        if repo is None:
            return None
        return routes_path(repo, self.setting("routes_file"))

    @property
    def evaluator(self) -> Path | None:
        repo = self.repo
        return None if repo is None else repo / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"

    @property
    def routes_subset(self) -> list[str] | None:
        value = self.setting("routes_subset")
        ids = [s.strip() for s in value.split(",") if s.strip()] if value else []
        return ids or None

    @property
    def record(self) -> bool:
        """The per-tick record (front/BEV JPEGs, model-camera thumbnails, log.csv, ticks.jsonl,
        actors.jsonl, route_plan.json, rig.json): off unless asked for.

        Measured ~100 KB a tick on paddock's rig (59 routes, 8,077 ticks), so ~22 GB
        for 220 routes of 1,000 ticks and ~90 GB if every route rode the 4000-tick
        cap; ~280 KB a tick through the ``bench2drive`` rig and ~390 KB through
        ``kitscenes`` (one route each). Plus an extra render on CARLA's side every tick.
        """
        return (self.setting("record") or "false").strip().lower() in ("1", "true", "yes", "on")

    @property
    def early_stop(self) -> dict[str, Any] | None:
        """The early-stop rules the site set (``early_stop.py``), or None when all are off.

        ``stop_stuck_s``: game seconds without :data:`early_stop.PROGRESS_M` of
        progress along the route. ``stop_expert_factor``: a multiple of the
        expert's time on the route, read from ``expert``. ``stop_on_collision``:
        the first collision the leaderboard records. A malformed value reads as
        off here; ``preflight`` is what reports it.
        """
        rules: dict[str, Any] = {}
        for key, name in (("stop_stuck_s", "stuck_s"), ("stop_expert_factor", "expert_factor")):
            value = _positive(self.setting(key))
            if value is not None:
                rules[name] = value
        if (self.setting("stop_on_collision") or "").strip().lower() in ("1", "true", "yes", "on"):
            rules["collision"] = True
        if not rules:
            return None
        rules["progress_m"] = early_stop.PROGRESS_M
        return rules

    def expert_budget_s(self, route_id: str) -> float | None:
        """``stop_expert_factor`` x the expert's game time on ``route_id``, or None.

        The expert's time is ``duration_s`` of ``<expert>/<route>/expert_traj.json``
        as ``tools.benchmarks.bench2drive.expert summarize`` writes it: Think2Drive's
        drive of the same route, start to end. Nine of the 220 routes have no
        expert clip; those get the stuck rule alone.
        """
        rules, expert = self.early_stop, self.setting("expert")
        if not rules or "expert_factor" not in rules or not expert:
            return None
        try:
            traj = json.loads((Path(expert) / route_id / EXPERT_TRAJ).read_text())
            duration = float(traj["duration_s"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return round(rules["expert_factor"] * duration, 2) if duration > 0 else None

    @property
    def record_inputs(self) -> bool:
        """Each tick's whole request under ``inputs/`` (``recorder.write_inputs``): off unless asked for.

        ~2.5 MB a tick for six 800x450 views, a replay's worth of a few routes and not
        a sweep's; it needs ``record`` on, and changes nothing the model is shown.
        """
        return (self.setting("record_inputs") or "false").strip().lower() in ("1", "true", "yes", "on")

    @property
    def camera_rig(self) -> str:
        """The rig the agent renders the model's cameras with (``rig.py``); paddock's unless named.

        Another rig changes what the model is shown, so it is on the run's record and
        the run does not rank (``invalid_reasons``). An unknown name is ``preflight``'s to report.
        """
        return (self.setting("camera_rig") or rig.DEFAULT).strip()

    def _pythonpath(self) -> list[str]:
        """What the client interpreter needs, in the checkout's own order.

        ``REPO_ROOT/src`` first, as for every process paddock launches: the
        agent imports the wire contract from ``paddock.policy.protocol`` and the
        tracker from this package rather than restating either.
        """
        carla_root, repo = self.carla_root, self.repo
        entries = [str(REPO_ROOT / "src")]
        if carla_root is not None:
            if (carla_root / CARLA_EGG).exists() and not self._has_own_carla:
                entries.append(str(carla_root / CARLA_EGG))
            entries.append(str(carla_root / "PythonAPI" / "carla"))
        if repo is not None:
            entries += [str(repo / "leaderboard"), str(repo / "scenario_runner")]
        return entries

    @functools.cached_property
    def _has_own_carla(self) -> bool:
        """True when the client interpreter has a ``carla`` package installed.

        The release's egg needs the system's libtiff.so.5, which newer systems
        replace with libtiff.so.6, and a conda env supplies it itself. PyPI's
        carla 0.9.15 wheel carries its own copy, so it is what
        ``paddock setup --bench bench2drive`` installs. An egg on PYTHONPATH
        would be imported ahead of that wheel, so it goes on only for an
        interpreter without one. ``find_spec`` finds carla without running it;
        whether it imports is preflight's check.
        """
        python = self.setting("python")
        if not python:
            return False
        probe = "import importlib.util, sys; sys.exit(importlib.util.find_spec('carla') is None)"
        try:
            found = subprocess.run([str(python), "-I", "-c", probe], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
        return found.returncode == 0

    def _environment(self, extra: dict[str, str]) -> dict[str, str]:
        """The evaluator's environment: the host passthrough set, plus ours.

        Built like ``runtime.env.build``, not inherited wholesale -- a stray
        PYTHONPATH from paddock's own env would make the client half-import
        paddock's dependencies. ``CUDA_VISIBLE_DEVICES`` passes through with the
        rest of the host's set; the CARLA the evaluator launches picks its card by
        ``gpu_rank`` (``-graphicsadapter``, a Vulkan index), which ignores it.
        """
        env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
        env.update(
            {
                "CARLA_ROOT": str(self.carla_root),
                "SCENARIO_RUNNER_ROOT": str(self.repo / "scenario_runner"),
                "LEADERBOARD_ROOT": str(self.repo / "leaderboard"),
                "PYTHONPATH": os.pathsep.join(self._pythonpath()),
            }
        )
        env.update(extra)
        return env

    # ---- preflight -----------------------------------------------------

    def preflight(self) -> list[str]:
        """Everything that must hold before a model is loaded, checked cheaply."""
        problems: list[str] = []
        if self.settings is None:
            return [
                "no [benchmarks.bench2drive] section in paddock.toml; it needs `python` and "
                "`repo` (`paddock setup --bench bench2drive` makes both) and `carla_root` "
                "(`paddock data fetch carla`); see docs/benchmarks/bench2drive.md"
            ]

        for key, how in _MAKES_SETTING.items():
            if not self.setting(key):
                problems.append(f"[benchmarks.bench2drive] {key} is not set in paddock.toml; {how}")

        python = self.setting("python")
        if python and not Path(python).exists():
            problems.append(f"CARLA-client interpreter not found: {python}")

        carla_root = self.carla_root
        if carla_root is not None:
            if not (carla_root / "CarlaUE4.sh").exists():
                problems.append(f"CARLA 0.9.15 not found at carla_root (no CarlaUE4.sh): {carla_root}")
            planner = carla_root / "PythonAPI" / "carla" / "agents" / "navigation" / "local_planner.py"
            if not planner.exists():
                problems.append(
                    f"CARLA PythonAPI is incomplete (agent.py imports RoadOption from it): {planner}"
                )

        repo = self.repo
        if repo is not None:
            for relative, why in (
                (Path("leaderboard/leaderboard/leaderboard_evaluator.py"), "the evaluator"),
                (Path("leaderboard/leaderboard/autoagents/autonomous_agent.py"), "the AutonomousAgent base"),
                (Path("scenario_runner/srunner/scenariomanager/carla_data_provider.py"), "scenario_runner"),
                (ATOMIC_BEHAVIORS, "scenario_runner's atomic behaviours"),
            ):
                if not (repo / relative).exists():
                    problems.append(f"Bench2Drive checkout is missing {why}: {repo / relative}")
            evaluator = self.evaluator
            if (
                evaluator is not None
                and evaluator.exists()
                and QUALITY_FLAG in evaluator.read_text(errors="replace")
            ):
                problems.append(
                    f"{evaluator} passes {QUALITY_FLAG} to CARLA; the pinned evaluator launches "
                    f"CARLA at default (Epic) quality and frames rendered at Low are not the "
                    f"benchmark's (black sky, speckled road). Revert the edit, or make it a patch "
                    f"under patches/Bench2Drive/ so it is on the record"
                )
            if evaluator is not None and evaluator.exists():
                problems.extend(patch_problems(evaluator.read_text(errors="replace"), evaluator))
            behaviors = repo / ATOMIC_BEHAVIORS
            if behaviors.exists():
                problems.extend(scenario_patch_problems(behaviors.read_text(errors="replace"), behaviors))

        routes = self.routes_file
        if routes is not None:
            if not routes.exists():
                problems.append(f"routes file not found: {routes}")
            elif self.routes_subset:
                known = {rid for rid, _, _ in parse_routes(routes)}
                unknown = [rid for rid in self.routes_subset if rid not in known]
                if unknown:
                    problems.append(
                        f"[benchmarks.bench2drive] routes_subset names route id(s) not in "
                        f"{routes.name}: {', '.join(unknown)}"
                    )

        for key in ("per_route_timeout_s", "connect_timeout_s"):
            value = self.setting(key)
            if value and not value.strip().isdigit():
                problems.append(f"[benchmarks.bench2drive] {key} must be an integer, got {value!r}")

        if self.camera_rig not in rig.RIGS:
            problems.append(
                f"[benchmarks.bench2drive] camera_rig {self.camera_rig!r} is not a rig in rig.py; "
                f"the rigs are {', '.join(sorted(rig.RIGS))}"
            )

        problems += self._early_stop_problems()
        if not problems:
            problems += self._import_problems()
        return problems

    def _early_stop_problems(self) -> list[str]:
        """The stop_* settings parse, the expert's drives exist, and the cap the agent ends routes by does."""
        problems = []
        collision = (self.setting("stop_on_collision") or "").strip().lower()
        if collision not in ("", "1", "true", "yes", "on", "0", "false", "no", "off"):
            problems.append(
                f"[benchmarks.bench2drive] stop_on_collision must be true or false, got {collision!r}"
            )
        if self.record_inputs and not self.record:
            problems.append(
                '[benchmarks.bench2drive] record_inputs = "true" needs record = "true": each tick\'s '
                "request is written beside the recorded route, and with record off there is none"
            )
        for key in ("stop_stuck_s", "stop_expert_factor"):
            value = self.setting(key)
            if value not in (None, "") and _positive(value) is None:
                problems.append(f"[benchmarks.bench2drive] {key} must be a number above zero, got {value!r}")
        if self.setting("stop_expert_factor") not in (None, ""):
            expert = self.setting("expert")
            if not expert:
                problems.append(
                    "[benchmarks.bench2drive] stop_expert_factor needs `expert`, the directory "
                    "`python -m tools.benchmarks.bench2drive.expert summarize` wrote"
                )
            elif not Path(expert).is_dir():
                problems.append(f"expert drives not found: {expert}")
            elif not any(Path(expert).glob(f"*/{EXPERT_TRAJ}")):
                problems.append(
                    f"no {EXPERT_TRAJ} under {expert}, so no route would have a time budget: "
                    "run `python -m tools.benchmarks.bench2drive.expert summarize` on the drives first"
                )
        if self.early_stop and self.repo is not None:
            manager = self.repo / "leaderboard" / "leaderboard" / "scenarios" / "scenario_manager.py"
            text = manager.read_text(errors="replace") if manager.exists() else ""
            if "def _tick_scenario" not in text or f"tick_count > {early_stop.TICK_CAP}" not in text:
                problems.append(
                    f"{manager} has no `tick_count > {early_stop.TICK_CAP}` cap in _tick_scenario; "
                    f"the early stop ends a route through it, so with this leaderboard it would end none"
                )
        return problems

    def _import_problems(self) -> list[str]:
        """Import everything the agent will, in the client interpreter.

        The agent runs in a foreign environment where a missing package shows
        up as a ModuleNotFoundError only after CARLA has booted, on every route.
        A one-second subprocess here reports it before the model is loaded.
        """
        command = [
            str(self.setting("python")),
            "-c",
            "import importlib, sys\n"
            f"for name in {list(AGENT_IMPORTS)!r}:\n"
            "    importlib.import_module(name)\n",
        ]
        try:
            result = subprocess.run(
                command, env=self._environment({}), capture_output=True, text=True, timeout=120
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"could not run the CARLA-client interpreter to check the agent's imports: {exc}"]
        if result.returncode == 0:
            return []
        last = (result.stderr.strip().splitlines() or ["(no output)"])[-1]
        return [
            f"the CARLA-client interpreter cannot import what agent.py needs: {last}\n"
            f"it must import {', '.join(AGENT_IMPORTS)} with PYTHONPATH="
            f"{os.pathsep.join(self._pythonpath())} (see docs/benchmarks/bench2drive.md)"
        ]

    def input_problems(self, model) -> list[str]:
        """Why ``model`` cannot be driven here, from its spec alone; the runner asks before any launch.

        The controller follows a plan's own points at every ``WAYPOINT_DT`` s and does
        not interpolate one (``pid.on_grid``). A spacing that does not divide 0.5 s
        has no such points, and the agent would refuse the first plan -- after CARLA
        has loaded the town and the model its checkpoint.
        """
        if grid_stride(model.waypoint_dt, WAYPOINT_DT) is None:
            return [
                f"{model.name} plans {model.waypoint_dt:g} s apart; Bench2Drive's controller follows a "
                f"plan's own points at every {WAYPOINT_DT:g} s and does not interpolate one, so the "
                f"spacing must divide {WAYPOINT_DT:g} s"
            ]
        return []

    def scene_problems(self, scene: str) -> list[str]:
        """``--scene`` names a route id or a town; say so before the model loads."""
        routes = self.routes_file
        if routes is None or not routes.exists():
            return []  # preflight has already reported it
        parsed = parse_routes(routes)
        ids = {rid for rid, _, _ in parsed}
        towns = sorted({town for _, town, _ in parsed})
        if scene in ids or scene in towns:
            return []
        return [
            f"--scene {scene} matches no route id or town in {routes.name} "
            f"({len(ids)} route ids; towns: {', '.join(towns)})"
        ]

    def scenario_problems(self, scenarios: list[str]) -> list[str]:
        """``--scenarios`` names scenario types; a typo would otherwise yield zero routes."""
        routes = self.routes_file
        if routes is None or not routes.exists():
            return []
        known = sorted({stype for _, _, stype in parse_routes(routes)})
        unknown = [s for s in scenarios if s not in known]
        if not unknown:
            return []
        return [
            f"--scenarios {', '.join(unknown)}: not a scenario type in {routes.name}; "
            f"known: {', '.join(known)}"
        ]

    def model_run_config(self, model) -> dict[str, Any]:
        """The rate the client asks this model for a plan, when not every tick.

        Recorded and hashed: a model asked at 2 Hz and at 20 Hz drives
        differently (Drive-JEPA 57.60 against 31.71 on the 50 routes), so the
        two must never share a run id. Absent means every tick, which is what
        a record without it was driven at.

        When the host's PADDOCK_INFER_HZ replaces the rate the model declares,
        the declared one is recorded beside it (``planning_hz``; ``infer_hz`` 0
        for every tick), so the record says the rate was an operator's: the run
        does not rank (``invalid_reasons``), and neither does a pool of such runs.
        A host that restates the model's own rate changes nothing.

        A model asked slower than its ``native_input_hz`` is handed frames between plans at that
        rate, recorded as ``observe_hz`` (:func:`observe_hz`).
        """
        declared = float(getattr(model, "planning_hz", 0.0) or 0.0)
        hz = infer_hz(declared)
        # Frames between plans (``observe_hz``), recorded and hashed like the rate: Qwen-Drive with
        # and without them reads different histories. Absent when none are handed, so every other
        # model keeps its run id.
        observed = observe_hz(getattr(model, "native_input_hz", 0.0), hz)
        extra = {"observe_hz": observed} if observed else {}
        if _same_rate(hz, declared):
            self._host_rate = None
            return extra if every_tick(declared) else {"infer_hz": declared, **extra}
        self._host_rate = (hz, declared)
        return {"infer_hz": 0.0 if every_tick(hz) else hz, "planning_hz": declared, **extra}

    def run_config(self) -> dict[str, Any]:
        """Site settings that change *what is measured*, for ``run.json``.

        A routes subset makes two sites' numbers incomparable unless it is on
        the record; recorded here and hashed into the run id by the runner.
        """
        routes = self.routes_file
        config: dict[str, Any] = {
            "routes_file": routes.name if routes is not None else None,
            "routes_subset": sorted(self.routes_subset) if self.routes_subset else None,
            "waypoint_dt": WAYPOINT_DT,
        }
        # Only when on, so the config hash -- and the run id -- of a run without it is unchanged.
        if self.early_stop:
            rules = dict(self.early_stop)
            # The site's routes the over-time rule cannot end, having no expert drive: which they
            # are depends on the expert drives a site has, so it is on the record and in the id.
            unbudgeted = self._routes_without_expert()
            if unbudgeted:
                rules["no_expert"] = unbudgeted
            config["early_stop"] = rules
        # The routes the site's set loses to SET_ASIDE, by id; why is in the notes (`remarks`), so
        # a reworded reason moves no run id. Only when there are some, so a set without any keeps
        # its id. Every worker of a sweep computes the same list, whatever its --scene.
        aside = self._routes_set_aside()
        if aside:
            config["routes_set_aside"] = aside
        # Likewise only when it is not paddock's.
        if self.camera_rig != rig.DEFAULT:
            config["camera_rig"] = self.camera_rig
        # And the host's sim2real diagnostics, only when set: under the
        # override the model's declared `cameras` are not the ones it was shown.
        override = _cameras_override()
        if override:
            config["cameras_override"] = override
        if os.environ.get("PADDOCK_SEGMENT"):
            # The spec's name: its path is this machine's.
            config["segment"] = Path(os.environ["PADDOCK_SEGMENT"]).name
        return config

    def _routes_without_expert(self) -> list[str]:
        """The site's routes (its ``routes_subset``, or the whole file) with no expert time budget.

        Counted over the set aside routes too: were they left out here, adding a route to
        :data:`SET_ASIDE` would change ``no_expert`` between two runs of one sweep, and a pool
        of them would read as stopped under two sets of rules.
        """
        rules, routes_file = self.early_stop, self.routes_file
        if not rules or "expert_factor" not in rules or routes_file is None or not routes_file.exists():
            return []
        routes = select_routes(parse_routes(routes_file), None, None, self.routes_subset)
        return sorted(
            (rid for rid, _, _ in routes if self.expert_budget_s(rid) is None), key=lambda r: (len(r), r)
        )

    def _routes_set_aside(self) -> list[str]:
        """The site's routes (its ``routes_subset``, or the whole file) that :data:`SET_ASIDE` leaves out."""
        routes_file = self.routes_file
        if routes_file is None or not routes_file.exists():
            return []
        routes = select_routes(parse_routes(routes_file), None, None, self.routes_subset)
        return sorted((rid for rid, _, _ in routes if rid in SET_ASIDE), key=lambda r: (len(r), r))

    def remarks(self) -> list[str]:
        """The early stop, when on: which rules, and why a route they end is a failed drive.
        And the routes set aside, when the site's set has any: which, and why."""
        rules = self.early_stop
        said = [early_stop.remark(rules)] if rules else []
        driven = set(getattr(self, "_set_aside_driven", []))
        aside = set_aside_remark(r for r in self._routes_set_aside() if r not in driven)
        return [*said, aside] if aside else said

    # ---- episodes ------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        # One run's routes ended by hand. The runner calls this once per model and drives
        # every model of `--model all` through this one harness, so the next run starts empty.
        self._ended_by_hand = set()
        routes_file = self.routes_file
        if routes_file is None or not routes_file.exists():
            return
        routes = parse_routes(routes_file)
        selected = list(select_routes(routes, scenes, scenarios, self.routes_subset, SET_ASIDE))
        # Recorded for `invalid_reasons`: the site's whole route set, less the routes set
        # aside, against what ran; and any set aside route --scene named, which no record counts.
        whole = select_routes(routes, None, None, self.routes_subset, SET_ASIDE)
        self._routes_scope = (len(selected), len(list(whole)))
        self._set_aside_driven = sorted(
            (rid for rid, _, _ in selected if rid in SET_ASIDE), key=lambda r: (len(r), r)
        )
        for rid, town, stype in selected:
            yield Episode(
                episode_id=f"{rid}-{town}-{stype}",
                scenario=stype,
                scene=rid,
                run_index=0,
                config={"route_id": rid, "town": town, "runs": runs},
            )

    def invalid_reasons(self) -> list[str]:
        """A run of fewer routes than the site runs is a subset, not a Bench2Drive score.

        `--scene` or `--scenarios` narrowing the site's set (its `routes_subset`,
        or the whole routes file) would otherwise be stored `valid: true` and
        ranked beside the full runs; the regression gate's 7-route smoke was.
        The site's own `routes_subset` is the set, not a subset: it is on the
        record (`run_config`) and every full run on the site uses it. The stuck
        and over-time rules are not a reason: a route they end is a failed drive
        (`remarks`). The collision rule is one: it scores a route at its first
        collision, where Bench2Drive charges a penalty and drives on. A camera rig
        other than paddock's is one: the model was shown other cameras. So is the
        host's PADDOCK_INFER_HZ replacing the rate the model declares: the model
        was asked for plans at an operator's rate (``model_run_config``). So is
        either sim2real diagnostic set on the host, whatever routes it drove:
        PADDOCK_CAMERAS_OVERRIDE shows the model cameras it does not declare, and
        PADDOCK_SEGMENT replays a recording and may convert the frames.
        """
        reasons = []
        manual = getattr(self, "_ended_by_hand", set())
        if manual:
            reasons.append(early_stop.manual_reason(len(manual)))
        ran, total = getattr(self, "_routes_scope", (None, None))
        if ran is not None and ran < total:
            reasons.append(
                f"{ran} of the {total} routes this site runs (--scene/--scenarios); a subset, not a score"
            )
        driven = getattr(self, "_set_aside_driven", [])
        if driven:
            reasons.append(
                f"drove route(s) set aside for every model ({', '.join(driven)}): a check whether the "
                "simulator still fails on them, not a score"
            )
        if (self.early_stop or {}).get("collision"):
            reasons.append(early_stop.collision_reason())
        if self.camera_rig != rig.DEFAULT:
            reasons.append(
                f"camera_rig {self.camera_rig!r}: the model saw other cameras than the "
                f"{rig.DEFAULT!r} rig every model here is compared through"
            )
        override = _cameras_override()
        if override:
            reasons.append(
                f"PADDOCK_CAMERAS_OVERRIDE on the host: the model was shown {', '.join(override)}, "
                "not the cameras it declares; a sim2real diagnostic, not a score"
            )
        if os.environ.get("PADDOCK_SEGMENT"):
            reasons.append(
                "PADDOCK_SEGMENT on the host: a sim2real diagnostic that replays a recorded drive "
                "and may convert its frames, not a score"
            )
        host = getattr(self, "_host_rate", None)  # set by model_run_config for the model being run
        if host is not None:
            hz, declared = host
            own = (
                "no planning_hz (a plan every tick)" if every_tick(declared) else f"planning_hz {declared:g}"
            )
            reasons.append(
                f"PADDOCK_INFER_HZ on the host asked the model for a plan {_rate_said(hz)}, where its "
                f"model.yaml declares {own}: an operator's rate, not the model's own"
            )
        return reasons

    def _note_stop(self, episode: Episode, metrics: dict[str, float]) -> None:
        """Remember a route someone ended by hand, for ``invalid_reasons``."""
        if metrics.get(f"ended_{early_stop.MANUAL}"):
            self.__dict__.setdefault("_ended_by_hand", set()).add(episode.episode_id)

    def completed_metrics(self, episode: Episode, out_dir: Path, runs: int = 1) -> dict[str, float] | None:
        """The route's metrics if its ``results.json`` already holds every repetition, else None.

        This is what ``--resume`` reuses; without it the base class answers None
        and a resumed sweep re-drives every route it had already scored. The
        evaluator writes ``results.json`` at startup and rewrites it as each
        repetition ends, so the file existing proves nothing: a route killed
        mid-drive -- a reboot, a per-route timeout -- holds no record, or fewer
        than the repetitions asked for, and is driven again. One record per
        repetition is a finished route, and it reads exactly as it did when it
        was driven: through :func:`parse_results`, which drops crashed
        repetitions and refuses a route where every one crashed, so a crash is
        driven again too.

        ``duration_s`` is the one metric not on disk -- it is the harness's own
        clock around the evaluator, CARLA's boot included -- so a reused route
        leaves it out rather than invent it, and the store counts it missing.
        """
        results = Path(out_dir) / "results.json"
        try:
            records = (json.loads(results.read_text()).get("_checkpoint") or {}).get("records") or []
        except (OSError, ValueError):
            return None
        if len(records) < max(1, runs):
            return None
        try:
            metrics = parse_results(results)
        except NotScored:
            return None
        metrics.update(ended_early(Path(out_dir)))
        self._note_stop(episode, metrics)
        return metrics

    # ---- run one episode ----------------------------------------------

    def run_episode(
        self,
        episode: Episode,
        policy_url: str,
        out_dir: Path,
        echo: Callable[[str], None] = print,
    ) -> dict[str, float]:
        """Drive one route through CARLA against a running policy node.

        ``policy_url`` is a node paddock has already started and health-checked.
        The evaluator self-launches CARLA (reading CARLA_ROOT), so nothing else
        needs starting here; the agent talks to the node over HTTP.

        Failure order matters: a timeout is reported as a timeout, a non-zero
        exit as an exit, and only then is ``results.json`` read -- which after
        either of those still exists, holding either no record or a crashed one.
        """
        repo = self.repo
        python = self.setting("python")
        cameras = episode.config.get("cameras") or ["CAM_FRONT"]
        # Diagnostic: render and ship only these cameras. The sim2real capture uses it to
        # record another model's rig behind a cheap node; it also proves a model ignores the
        # cameras it declares. Host-set, and the adapter must tolerate the gap; a run with it
        # is on the record and never ranks (`run_config`, `invalid_reasons`).
        override = _cameras_override()
        if override:
            cameras = override
        runs = int(episode.config.get("runs", 1))
        timeout_s = int(self.setting("per_route_timeout_s") or DEFAULT_ROUTE_TIMEOUT_S)
        requested_port = int(self.setting("carla_port") or 2000)

        out_dir.mkdir(parents=True, exist_ok=True)
        results_json = out_dir / "results.json"
        log_path = out_dir / "evaluator.log"

        extra = {
            "PADDOCK_NODE_URL": policy_url,
            "PADDOCK_CAMERAS": ",".join(cameras),
            "PADDOCK_CAMERA_RIG": self.camera_rig,
            "PADDOCK_WAYPOINT_DT": str(WAYPOINT_DT),
            # Per-tick ego kinematics for Bench2Drive's efficiency/comfort tool,
            # which joins ONE directory with each record's `save_name`. One dir
            # per run, not per episode, so a whole sweep scores with one `-m`.
            "PADDOCK_METRICS_DIR": str(out_dir.parent / "metric_info"),
        }
        # The agent reads PADDOCK_INFER_TIMEOUT_S for its per-tick /infer HTTP
        # timeout (default 120 s). It is set on the host, not in PASSTHROUGH, so
        # forward it explicitly or the documented override never reaches the
        # foreign-interpreter agent that honours it.
        if os.environ.get("PADDOCK_INFER_TIMEOUT_S"):
            extra["PADDOCK_INFER_TIMEOUT_S"] = os.environ["PADDOCK_INFER_TIMEOUT_S"]
        if hz := infer_hz(episode.config.get("planning_hz")):
            extra["PADDOCK_INFER_HZ"] = f"{hz:g}"
            if observed := observe_hz(episode.config.get("native_input_hz"), hz):
                extra["PADDOCK_OBSERVE_HZ"] = f"{observed:g}"
        # Segment mode (sim2real diagnostics: capture a recording, or replay it to T0 and then
        # swap the frames); see bench2drive/segment.py. Host-set like the timeout above; a run
        # with it is on the record and never ranks (`run_config`, `invalid_reasons`).
        if os.environ.get("PADDOCK_SEGMENT"):
            extra["PADDOCK_SEGMENT"] = os.environ["PADDOCK_SEGMENT"]
        if self.record:
            extra["PADDOCK_RECORD_DIR"] = str(out_dir)
            if self.record_inputs:
                extra["PADDOCK_RECORD_INPUTS"] = "1"
            # agent_wrapper reads SAVE_PATH as its "Bench2Drive mode" flag, which
            # lifts the 3 m sensor-radius limit the BEV camera at z=30 needs.
            extra["SAVE_PATH"] = str(out_dir)
        # A route driven again in the same run directory must not inherit the
        # last attempt's verdict.
        stop_record = out_dir / early_stop.RECORD
        stop_record.unlink(missing_ok=True)
        # Nor the last attempt's failure, or the guard's word on it.
        (out_dir / FAILURE).unlink(missing_ok=True)
        (out_dir / GUARD_KILL).unlink(missing_ok=True)
        # The file that ends this route by hand when it appears (the dashboard's
        # "Stop route" writes it), cleared so an old request cannot end a new drive.
        stop_request = out_dir / STOP_REQUEST
        stop_request.unlink(missing_ok=True)
        extra["PADDOCK_STOP_REQUEST"] = str(stop_request)
        extra["PADDOCK_STOP_RECORD"] = str(stop_record)
        rules = self.early_stop
        if rules:
            if rules.get("collision"):
                extra["PADDOCK_STOP_ON_COLLISION"] = "1"
            if "stuck_s" in rules:
                extra["PADDOCK_STOP_STUCK_S"] = str(rules["stuck_s"])
            budget = self.expert_budget_s(episode.config["route_id"])
            if budget is not None:
                extra["PADDOCK_STOP_AFTER_S"] = str(budget)
            elif "expert_factor" in rules:
                echo(f"  {episode.episode_id}: no expert drive, so no time budget; the stuck rule alone")

        command = [
            str(python),
            "-u",
            str(self.evaluator),
            "--routes",
            str(self.routes_file),
            "--routes-subset",
            episode.config["route_id"],
            "--repetitions",
            str(runs),
            "--track",
            "SENSORS",
            "--checkpoint",
            str(results_json),
            "--agent",
            str(AGENT),
            "--port",
            str(requested_port),
            "--traffic-manager-port",
            str(self.setting("tm_port") or 8000),
            "--gpu-rank",
            str(self.setting("gpu_rank") or 0),
            CONNECT_TIMEOUT_FLAG,
            str(int(self.setting("connect_timeout_s") or DEFAULT_CONNECT_TIMEOUT_S)),
        ]

        started = time.time()
        timed_out = False
        with log_path.open("w") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            # Its own session, so a timeout or an interrupted sweep takes the
            # evaluator and its children down together; the log streams to a
            # file as it runs, so a hung route is diagnosable while it hangs.
            # The evaluator reads `leaderboard/data/weather.xml` relative to
            # its working directory, which must therefore be the checkout root.
            evaluator = subprocess.Popen(
                command,
                cwd=repo,
                env=self._environment(extra),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = evaluator.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(evaluator)
                returncode = evaluator.returncode
            except BaseException:
                kill_group(evaluator)
                reap_carla(carla_port_from_log(log_path, requested_port))
                raise
        elapsed = time.time() - started

        # On a normal exit the evaluator's atexit hook has already reaped its
        # CARLA; after a kill it has not. Either way, only ours, by its port.
        reap_carla(carla_port_from_log(log_path, requested_port))

        def failed(message: str) -> RuntimeError:
            """Say why beside results.json (`failure.json`), then the error the runner logs.

            The record is best-effort: the runner catches only RuntimeError, so a full disk here
            must not take the worker's whole run, and the routes it already scored, with it.
            """
            try:
                record = classify_failure(
                    log_path.read_text(errors="replace") if log_path.exists() else "",
                    timed_out=timed_out,
                    returncode=returncode,
                    results_json=results_json,
                )
                record["elapsed_s"] = round(elapsed, 1)
                record["node_alive"] = node_alive(policy_url)
                try:
                    guard = json.loads((out_dir / GUARD_KILL).read_text())
                except (OSError, ValueError):
                    guard = None
                if isinstance(guard, dict) and guard.get("reason"):
                    record["guard"] = str(guard["reason"])
                    record["guard_carla_gone"] = bool(guard.get("carla_gone"))
                write_failure(out_dir, record)
            except OSError as exc:
                echo(f"  {episode.episode_id}: could not write {FAILURE} ({exc})")
            return RuntimeError(message)

        if timed_out:
            raise failed(
                f"{episode.episode_id}: timed out after {timeout_s / 60:.0f} min "
                f"(per_route_timeout_s); evaluator killed\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        if returncode != 0:
            raise failed(
                f"{episode.episode_id}: evaluator exited {returncode} after {elapsed / 60:.1f} min"
                f"\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        if not results_json.exists():
            raise failed(
                f"{episode.episode_id}: evaluator exited 0 but wrote no results.json"
                f"\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )

        try:
            metrics = parse_results(results_json)
        except NotScored as exc:
            raise failed(
                f"{episode.episode_id}: {exc}\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            ) from exc
        metrics["duration_s"] = round(elapsed, 1)
        metrics.update(ended_early(out_dir))
        self._note_stop(episode, metrics)
        return metrics


#: Paddock looks for this name when loading an adapter.
BENCHMARK = Bench2DriveBenchmark
