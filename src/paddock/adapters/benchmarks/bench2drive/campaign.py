"""Drive a Bench2Drive sweep with N parallel workers, and keep it moving through reboots and hangs.

    C="python -m paddock.adapters.benchmarks.bench2drive.campaign"
    $C status --model openpilot
    $C launch --model openpilot --workers 5 [--held 23695] [--only ids] [--exclude ids] [--dry-run]
    $C launch --model openpilot --workers 5 --rebalance
    $C launch --model openpilot --workers 5 --cards 0:0,1:2 --batch 4
    $C keep   --model openpilot --workers 5 --cards 0:0,1:2 --batch 4
    $C stop   --model openpilot
    $C guard  --model openpilot

A full sweep is 220 routes at up to ~40 minutes each, so it runs as several
``paddock run`` processes ("workers") side by side, each with its own CARLA. This
module is the bookkeeping for that, read entirely off what paddock already
writes -- no state of its own decides anything:

* **what is left** -- every route in the routes file, minus the ones with a
  scored attempt anywhere under ``{runs}/bench2drive/<model>/`` (any run id:
  earlier workers, re-drives, other launches), minus the ones an alive worker
  still has queued. "Scored" is :func:`harness.parse_results` succeeding, so a
  crashed or killed attempt counts as left. ``paddock ingest`` pools the same
  way (latest scored attempt per route), so launches can be repeated freely.
* **launch** -- the routes left, sorted, are dealt to the free worker slots
  longest-expected-first (each route's expected cost is the mean wall time of
  its scenario's finished routes in this sweep), so a sweep resumed after a
  reboot, or with three of five workers already done, is spread over every
  slot rather than left in whichever worker happened to own it. Each worker is
  a fresh ``paddock run --scene <its routes>`` with its own copy of the site
  config: CARLA and traffic-manager ports 150 apart per slot (CARLA holds N,
  N+1, N+2 and the evaluator walks up from N to the first free port; two
  workers starting together would otherwise take the same one) and its own
  node port base. ``--rebalance`` first stops the alive workers (their current
  routes are lost and driven again) so that everything left is dealt anew.
* **guard** -- kills an evaluator that can no longer finish its route, so its
  worker records the route FAILED and moves on instead of waiting out
  ``per_route_timeout_s``. Two signs, both seen in practice: its log has been
  silent for 15 minutes (a route that drives writes a line every ~0.3 s), or
  its CARLA server is gone -- when CARLA dies mid-route the evaluator logs a
  "failed to destroy actor" line every 600 s for hours, so silence alone never
  fires. It writes its reason into the route's folder (``guard.json``) first,
  so the attempt's failure record says why.
* **what is given up on** -- a route whose last ``--tries`` attempts (3 by
  default) each left a failure record (``failure.json``) and none scored is not
  dealt again; ``status`` lists it with why each attempt failed, and whether
  that was the simulator (a candidate for ``harness.SET_ASIDE``) or the model
  (the model's to fix). An attempt cut off by a restart leaves no record and
  does not count. A route given up on, or already in ``SET_ASIDE``, is dealt
  again only when ``--only`` names it: to drive it after a fix, or to see
  whether it still fails.
* **batches and cards** -- ``--batch N`` deals at most N routes to each new
  worker, so the last hours are not one long route queue in one slot while the
  others stand idle; ``keep`` refills the slots as they free. ``--cards
  CUDA:VULKAN,...`` spreads the workers over GPUs, ``--workers`` of them on
  each: each worker's config copy gets the card's ``gpu_rank`` (CARLA's
  ``-graphicsadapter`` is a Vulkan index, which on some machines is not the CUDA
  one: a software renderer can sit between two GPUs) and its node the card's
  ``CUDA_VISIBLE_DEVICES``.
* **keep** -- ``launch`` again every few minutes until nothing is left to deal
  and no worker is alive; with no worker alive it first stops what a worker
  killed hard can leave behind (an evaluator and its CARLA, a model node on the
  GPU), which nothing else would. It stops, exit 1, after three launches in a
  row whose workers all exited without attempting a route.

All four go only after this sweep's own processes: workers are found by their
command line *and* by the runs root of their ``PADDOCK_CONFIG``, and ``stop``
signals only those whose ``PADDOCK_CONFIG`` is one of the campaign's own
``worker<slot>.toml`` (a run started by hand on the same runs root keeps its
routes from being dealt twice, and is left running); evaluators by their
``--checkpoint`` lying under it; CARLA by the port in that evaluator's log. A
process whose environment cannot be read -- another user's -- is never ours.
A route suspected of crashing CARLA can be passed as ``--held``: driven last,
by one worker, once nothing else is left. (Before patch 0001, which preflight
now requires, the evaluator's crash handler also killed every CARLA on its GPU
index, the other workers' with it.)

Run it in paddock's own interpreter; ``PADDOCK_CONFIG`` picks the site config
the workers copy, exactly as for ``paddock run``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from paddock.adapters.benchmarks.bench2drive.harness import (
    GUARD_KILL,
    SET_ASIDE,
    NotScored,
    carla_pids,
    carla_port_from_log,
    failure_said,
    parse_results,
    parse_routes,
    read_failure,
    routes_path,
    simulator_side,
)
from paddock.config import REPO_ROOT, SiteConfig
from paddock.runtime import ports

#: Port spacing between worker slots, as Bench2Drive's own multi-GPU scripts space them.
SLOT_STRIDE = 150
#: Node port spacing between worker slots (the runner scans up from PADDOCK_BASE_PORT).
NODE_STRIDE = 10
#: Seconds between worker starts, so the CARLAs do not all load a town at once.
STAGGER_S = 20.0
#: A route that drives writes a log line every ~0.3 s; CARLA's boot and map load take about a minute.
SILENT_S = 900.0
#: How long an evaluator may run without its CARLA before the guard calls it dead.
CARLA_GONE_S = 120.0
#: Failed attempts after which a route is not dealt again (``--tries``): enough to tell a
#: repeatable crash from a one-off. The 220-route protocol's (docs/benchmarks/bench2drive.md).
TRIES = 3
#: Seconds between ``keep``'s launches.
KEEP_EVERY_S = 300.0
#: How long a launch waits for another launch of the same sweep -- a keeper's pass, a restart run
#: beside it -- to finish dealing, before it gives up for this pass.
LOCK_WAIT_S = 300.0
#: The sweep's own files beside the worker configs: the lock two launches deal under, and the
#: running keeper's pid, which `stop` stops first.
LAUNCH_LOCK = "launch.lock"
KEEPER_PID = "keep.pid"
#: ``keep`` passes in a row whose workers all exited without a route attempt before it stops:
#: a model node that cannot start fails every worker the same way, and would be relaunched forever.
KEEP_STRIKES = 3

_ROUTE_DIR = re.compile(r"^(?P<rid>\d+)-(?P<town>Town\w+?)-(?P<scenario>\w+)$")


# ---------------------------------------------------------------------------
# What is left: pure, tested without processes
# ---------------------------------------------------------------------------


@dataclass
class RouteState:
    route_id: str
    scored: bool = False
    attempts: int = 0
    #: Evaluator wall seconds of each scored attempt (results.json meta.duration_system).
    durations: list[float] = field(default_factory=list)
    #: The failure record of each attempt that left one (harness ``failure.json``), oldest first.
    #: An attempt cut off by a restart leaves none, and is not a failure of the route.
    failures: list[dict] = field(default_factory=list)


def route_states(runs_root: Path) -> dict[str, RouteState]:
    """Every attempt under ``runs_root`` (``<run_id>/<route>-<Town>-<Scenario>/``), folded per route."""
    states: dict[str, RouteState] = {}
    if not runs_root.is_dir():
        return states
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for route_dir in sorted(run.iterdir()):
            m = _ROUTE_DIR.match(route_dir.name)
            if not m or not route_dir.is_dir():
                continue
            st = states.setdefault(m["rid"], RouteState(m["rid"]))
            st.attempts += 1
            results = route_dir / "results.json"
            try:
                parse_results(results)
            except (NotScored, OSError, ValueError):
                failure = read_failure(route_dir)
                if failure is not None:
                    st.failures.append(failure)
                continue
            st.scored = True
            try:
                records = json.loads(results.read_text())["_checkpoint"]["records"]
                st.durations += [float((r.get("meta") or {}).get("duration_system") or 0.0) for r in records]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return states


def counted(failures: list[dict]) -> list[dict]:
    """The failures that use up a try: not one whose model node was gone, which failed the node."""
    return [f for f in failures if f.get("node_alive") is not False]


def given_up(states: dict[str, RouteState], tries: int) -> set[str]:
    """Routes with no score and ``tries`` counted failures or more: not dealt again. 0 never gives up."""
    if tries <= 0:
        return set()
    return {rid for rid, st in states.items() if not st.scored and len(counted(st.failures)) >= tries}


def whose(failures: list[dict]) -> str:
    """Who has to act on a route given up on, from its attempts' failure records."""
    sides = {simulator_side(f) for f in failures}
    if sides == {True}:
        return "the simulator's every time: a candidate for harness.SET_ASIDE"
    if False in sides:
        return "the model's or its client's at least once: fix that, then drive it again"
    return "not clear from the records: read the attempts' evaluator.log"


def select_pending(
    route_ids: Iterable[str],
    scored: set[str],
    claimed: set[str],
    only: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    """Routes still to drive, sorted by id: in the selection, not scored, not queued in an alive worker."""
    out = [
        rid
        for rid in route_ids
        if (only is None or rid in only)
        and rid not in (exclude or set())
        and rid not in scored
        and rid not in claimed
    ]
    return sorted(set(out), key=int)


def expected_costs(
    states: dict[str, RouteState], scenario_of: dict[str, str], boot_s: float = 90.0
) -> dict[str, float]:
    """Seconds each route is expected to take: its scenario's mean wall time in this sweep, plus CARLA's boot.

    Scenarios with nothing finished yet take the mean over everything finished; a sweep with nothing
    finished costs every route the same, which deals them round-robin.
    """
    per_scenario: dict[str, list[float]] = defaultdict(list)
    for rid, st in states.items():
        if st.scored and st.durations and rid in scenario_of:
            per_scenario[scenario_of[rid]] += st.durations
    every = [d for ds in per_scenario.values() for d in ds]
    overall = sum(every) / len(every) if every else 300.0
    return {
        rid: boot_s + (sum(per_scenario[s]) / len(per_scenario[s]) if per_scenario.get(s) else overall)
        for rid, s in scenario_of.items()
    }


def balance(
    routes: list[str], cost: Callable[[str], float], n: int, cap: int | None = None
) -> list[list[str]]:
    """Deal ``routes`` to ``n`` workers longest-expected-first (LPT); each worker's list sorted by id.

    Ties break on the route id, so the same inputs always give the same split. ``cap`` is the most
    routes one worker gets (``--batch``); the caller deals no more than ``n * cap``.
    """
    n = max(1, min(n, len(routes)))
    loads = [0.0] * n
    groups: list[list[str]] = [[] for _ in range(n)]
    for rid in sorted(routes, key=lambda r: (-cost(r), int(r))):
        room = [i for i in range(n) if cap is None or len(groups[i]) < cap]
        k = min(room, key=lambda i: (loads[i], i))
        groups[k].append(rid)
        loads[k] += cost(rid)
    return [sorted(g, key=int) for g in groups if g]


def with_ports(toml_text: str, carla_port: int, tm_port: int, gpu_rank: str | None = None) -> str:
    """The site config with ``[benchmarks.bench2drive]`` carla_port / tm_port set; everything else verbatim.

    Edited as text, not re-serialised, so the copy keeps every comment and key of the original and a
    reader can diff the two. Keys that are absent are added under the table header. ``gpu_rank``, when
    given, is set the same way: the card a worker's CARLA renders on.
    """
    lines = toml_text.splitlines(keepends=True)
    header = next(
        (i for i, ln in enumerate(lines) if re.match(r"^\[benchmarks\.bench2drive\]\s*(#.*)?$", ln)), None
    )
    if header is None:
        raise ValueError("the site config has no [benchmarks.bench2drive] table")
    end = next((i for i in range(header + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
    wanted = {"carla_port": str(carla_port), "tm_port": str(tm_port)}
    if gpu_rank is not None:
        wanted["gpu_rank"] = str(gpu_rank)
    for i in range(header + 1, end):
        for key, value in wanted.items():
            if re.match(rf"^\s*{key}\s*=", lines[i]):
                lines[i] = f'{key} = "{value}"\n'
                wanted[key] = ""
    missing = [f'{k} = "{v}"\n' for k, v in wanted.items() if v]
    lines[header + 1 : header + 1] = missing
    text = "".join(lines)
    table = tomllib.loads(text)["benchmarks"]["bench2drive"]
    assert (int(table["carla_port"]), int(table["tm_port"])) == (carla_port, tm_port)
    assert gpu_rank is None or str(table["gpu_rank"]) == str(gpu_rank)
    return text


def slot_ports(slot: int, base_carla: int, base_tm: int, node_base: int) -> tuple[int, int, int]:
    """(CARLA rpc port, traffic-manager port, node base port) of a worker slot."""
    return base_carla + SLOT_STRIDE * slot, base_tm + SLOT_STRIDE * slot, node_base + NODE_STRIDE * slot


def free_slots(
    n: int, base_carla: int, base_tm: int, node_base: int, taken: set[int], is_free: Callable[[int], bool]
) -> list[int]:
    """The first ``n`` slots none of whose ports is taken by an alive worker or bound by anything."""
    out, slot = [], 0
    while len(out) < n and slot < 64:
        carla, tm, _node = slot_ports(slot, base_carla, base_tm, node_base)
        needed = (carla, carla + 1, carla + 2, tm)
        if not any(p in taken for p in needed) and all(is_free(p) for p in needed):
            out.append(slot)
        slot += 1
    return out


# ---------------------------------------------------------------------------
# Processes: this sweep's workers, evaluators, CARLAs
# ---------------------------------------------------------------------------


@dataclass
class Worker:
    pid: int
    routes: list[str]
    config: Path
    carla_port: int | None
    #: The card its CARLA renders on: its config's ``gpu_rank``.
    gpu_rank: str | None = None


def _argv(pid: int) -> list[str]:
    try:
        return [
            a.decode(errors="replace") for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a
        ]
    except OSError:
        return []


def _environ(pid: int) -> dict[str, str] | None:
    """The process's environment, or None when it cannot be read: another user's, or gone."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    return dict(item.decode(errors="replace").split("=", 1) for item in raw if b"=" in item)


def _pids() -> Iterable[int]:
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            yield int(entry.name)


def _process_age_s(pid: int) -> float:
    """Seconds since ``pid`` started, from /proc (field 22 of stat is the start time in clock ticks)."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - started
    except (OSError, ValueError, IndexError):
        return 0.0


def _runs_root(cfg: SiteConfig, model: str) -> Path:
    return cfg.runs / "bench2drive" / model


def _state_dir(cfg: SiteConfig, model: str) -> Path:
    """The campaign's own files: each worker's config (``worker<slot>.toml``) and the logs."""
    return cfg.runs / "bench2drive" / "_campaign" / model


def workers(model: str, runs_root: Path) -> list[Worker]:
    """Alive ``paddock run --bench bench2drive --model <model>`` processes writing under ``runs_root``."""
    found = []
    for pid in _pids():
        argv = _argv(pid)
        if (
            not argv
            or "run" not in argv
            or not any(a.endswith("paddock") or a == "paddock.cli" for a in argv)
        ):
            continue
        try:
            bench, mdl = argv[argv.index("--bench") + 1], argv[argv.index("--model") + 1]
        except (ValueError, IndexError):
            continue
        if bench != "bench2drive" or mdl != model:
            continue
        env = _environ(pid)
        if env is None:  # nothing says where it writes: not ours to count, and never to stop
            continue
        cfg_path = Path(env.get("PADDOCK_CONFIG") or REPO_ROOT / "paddock.toml")
        try:
            cfg = SiteConfig.load(cfg_path)
        except Exception:  # an unreadable config is not one of ours
            continue
        if _runs_root(cfg, model).resolve() != runs_root.resolve():
            continue
        scene = argv[argv.index("--scene") + 1] if "--scene" in argv else ""
        settings = cfg.benchmarks.get("bench2drive")
        port = settings.get("carla_port") if settings is not None else None
        rank = settings.get("gpu_rank") if settings is not None else None
        routes = [r for r in scene.split(",") if r]
        found.append(Worker(pid, routes, cfg_path, int(port) if port else None, rank or "0"))
    return found


def evaluators(runs_root: Path) -> list[tuple[int, Path]]:
    """(pid, route dir) of every leaderboard evaluator whose checkpoint lies under ``runs_root``."""
    found = []
    root = runs_root.resolve()
    for pid in _pids():
        argv = _argv(pid)
        if not any(a.endswith("leaderboard_evaluator.py") for a in argv) or "--checkpoint" not in argv:
            continue
        checkpoint = Path(argv[argv.index("--checkpoint") + 1]).resolve()
        if root in checkpoint.parents:
            found.append((pid, checkpoint.parent))
    return found


def _ppid(pid: int) -> int | None:
    """``pid``'s parent, from /proc (field 4 of stat), or None when it cannot be read."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _a_run(pid: int) -> bool:
    """Whether ``pid`` is a ``paddock run``, however its flags are spelled (``-b``, ``--model all``)."""
    argv = _argv(pid)
    return "run" in argv and any(a.endswith("paddock") or a == "paddock.cli" for a in argv)


def orphaned(pid: int) -> bool:
    """Whether the ``paddock run`` that started ``pid`` is gone.

    An evaluator and a model node are children of their worker's run, so one whose parent is no
    longer a run is an orphan, and ours to stop; one whose run is alive is never, whoever started
    that run and however -- `workers()` knows only the campaign's own spelling of a command line.
    """
    parent = _ppid(pid)
    return parent is None or not _a_run(parent)


def orphaned_nodes(runs_root: Path) -> list[int]:
    """Orphaned model nodes of this sweep's runs: launch nonce (the runner's run id) a run under the root.

    A worker leads its own session and its node runs in it, so stopping a worker stops its node; one
    killed some other way (the kernel's OOM killer, say) leaves the node holding the GPU, and the next
    launch would wait for the memory it holds.
    """
    found = []
    for pid in _pids():
        argv = _argv(pid)
        if "paddock.policy.server" not in argv or "--nonce" not in argv:
            continue
        nonce = argv[argv.index("--nonce") + 1] if argv.index("--nonce") + 1 < len(argv) else ""
        if nonce and "/" not in nonce and (runs_root / nonce).is_dir() and orphaned(pid):
            found.append(pid)
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _signal_and_wait(pid: int, sig: int, wait_s: float, group: bool = False) -> bool:
    """Send ``sig``; True once ``pid`` has exited within ``wait_s``."""
    try:
        (os.killpg if group else os.kill)(pid, sig)
    except ProcessLookupError:
        return True
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(1.0)
    return not _alive(pid)


def _default_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def kill_evaluator(pid: int, route_dir: Path) -> None:
    """The evaluator's whole session (it is started as a session leader), then its CARLA by port."""
    if not _signal_and_wait(pid, signal.SIGTERM, 30.0, group=True):
        _signal_and_wait(pid, signal.SIGKILL, 10.0, group=True)
    port = carla_port_from_log(route_dir / "evaluator.log", 0)
    for carla in carla_pids(port) if port else []:
        try:
            os.kill(carla, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# The four commands
# ---------------------------------------------------------------------------


def _routes_meta(cfg: SiteConfig) -> list[tuple[str, str, str]]:
    settings = cfg.benchmarks.get("bench2drive")
    repo = settings.get("repo") if settings else None
    if not repo:
        raise SystemExit("[benchmarks.bench2drive] repo is not set in the site config")
    return parse_routes(routes_path(Path(repo), settings.get("routes_file")))


def _ids(text: str | None) -> set[str] | None:
    return {r.strip() for r in text.split(",") if r.strip()} if text else None


def _cards(text: str | None) -> list[tuple[str, str]] | None:
    """``--cards 0:0,1:2`` -> [("0", "0"), ("1", "2")]: (CUDA index, Vulkan index) per GPU."""
    if not text:
        return None
    pairs = [tuple(p.strip().split(":")) for p in text.split(",") if p.strip()]
    if not pairs or any(len(p) != 2 or not all(x.isdigit() for x in p) for p in pairs):
        raise SystemExit(f"--cards takes CUDA:VULKAN index pairs, e.g. 0:0,1:2; got {text!r}")
    return [(cuda, rank) for cuda, rank in pairs]


def status(cfg: SiteConfig, model: str, tries: int = TRIES, echo=print) -> dict:
    root = _runs_root(cfg, model)
    every = _routes_meta(cfg)
    aside = [rid for rid, _, _ in every if rid in SET_ASIDE]
    routes = [r for r in every if r[0] not in SET_ASIDE]
    ids = {rid for rid, _, _ in routes}
    states = route_states(root)
    alive = workers(model, root)
    driving = {_ROUTE_DIR.match(d.name)["rid"] for _, d in evaluators(root) if _ROUTE_DIR.match(d.name)}
    scored = {r for r, s in states.items() if s.scored and r in ids}
    gave_up = sorted(given_up(states, tries) & ids, key=int)
    claimed = {r for w in alive for r in w.routes if r not in scored}
    left = [rid for rid, _, _ in routes if rid not in scored and rid not in gave_up]
    echo(f"{model} on bench2drive: {len(scored)} / {len(routes)} routes scored, {len(left)} left ({root})")
    if aside:
        echo(f"  set aside for every model, so not dealt: {','.join(aside)}")
    for rid in gave_up:
        failures = states[rid].failures
        tried = counted(failures)
        echo(f"  given up on {rid} after {len(tried)} failed attempt(s), {whose(tried)}:")
        for failure in failures:
            echo(f"    {failure_said(failure)}")
    for w in alive:
        mine = [r for r in w.routes if r not in scored]
        card = f", gpu_rank {w.gpu_rank}" if w.gpu_rank is not None else ""
        echo(f"  worker pid {w.pid} (CARLA port {w.carla_port}{card}): {len(mine)} of {len(w.routes)} left")
    echo(f"  driving now: {', '.join(sorted(driving, key=int)) or 'nothing'}")
    unclaimed = [r for r in left if r not in claimed]
    if unclaimed:
        echo(f"  left and not queued in any worker ({len(unclaimed)}): {','.join(unclaimed)}")
    return {
        "scored": len(scored),
        "total": len(routes),
        "left": left,
        "workers": len(alive),
        "unclaimed": unclaimed,
        "given_up": gave_up,
    }


def keeper(cfg: SiteConfig, model: str) -> int | None:
    """The pid of this sweep's running ``keep``, from its pid file, or None."""
    try:
        pid = int((_state_dir(cfg, model) / KEEPER_PID).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    argv = _argv(pid)
    is_keep = "keep" in argv and any(a.endswith("campaign") for a in argv)
    return pid if is_keep and "--model" in argv and model in argv else None


def stop(cfg: SiteConfig, model: str, echo=print, dry_run: bool = False, keeper_too: bool = True) -> int:
    """Stop this sweep's keeper, then its workers: SIGINT first, which the harness turns into
    killing its evaluator + CARLA.

    The keeper first, or its next pass would start the workers again. Only the workers ``launch``
    started, known by their ``PADDOCK_CONFIG``: one of the campaign's ``worker<slot>.toml``. A
    ``paddock run`` of the same model started by hand counts for ``launch`` (its routes are not dealt
    twice) but is not ours to stop. ``dry_run`` says which it would stop, and signals nothing;
    ``keeper_too`` false (``--rebalance``) leaves the keeper running.
    """
    root = _runs_root(cfg, model)
    state_dir = _state_dir(cfg, model).resolve()
    alive = [
        w
        for w in workers(model, root)
        if w.config.resolve().parent == state_dir and w.config.name.startswith("worker")
    ]
    kept = keeper(cfg, model) if keeper_too else None
    if dry_run:
        if kept is not None:
            echo(f"would stop the keeper pid {kept}")
        for w in alive:
            echo(f"would stop worker pid {w.pid} ({len(w.routes)} routes queued)")
        return len(alive)
    if kept is not None:
        echo(f"stopping the keeper pid {kept}")
        if not _signal_and_wait(kept, signal.SIGTERM, 30.0):
            _signal_and_wait(kept, signal.SIGKILL, 10.0)
    for w in alive:
        echo(f"stopping worker pid {w.pid} ({len(w.routes)} routes queued)")
        if not _signal_and_wait(w.pid, signal.SIGINT, 180.0):
            # Past SIGINT, the worker's whole process group: its model node runs in it, and a
            # worker killed alone leaves the node on the GPU (workers lead their own session).
            echo(f"  pid {w.pid} ignored SIGINT for 3 min; SIGTERM to its process group")
            if not _signal_and_wait(w.pid, signal.SIGTERM, 30.0, group=True):
                _signal_and_wait(w.pid, signal.SIGKILL, 10.0, group=True)
    # A worker killed past its SIGINT handler leaves its evaluator (own session) and CARLA behind;
    # a run started by hand on this root keeps its own.
    for pid, route_dir in evaluators(root):
        if orphaned(pid):
            echo(f"cleaning up orphaned evaluator pid {pid} ({route_dir.name})")
            kill_evaluator(pid, route_dir)
    return len(alive)


def _reap_orphans(root: Path, echo=print) -> None:
    """What workers killed hard leave behind: evaluators (own session) with their CARLA, and model nodes.

    Only those whose run is gone (:func:`orphaned`): a run started by hand on this runs root, which
    the campaign may not recognise as a worker, keeps its evaluator and node. One that is not ours to
    signal (another user's) is reported and left.
    """
    for pid, route_dir in evaluators(root):
        if not orphaned(pid):
            continue
        echo(f"cleaning up orphaned evaluator pid {pid} ({route_dir.name})")
        try:
            kill_evaluator(pid, route_dir)
        except PermissionError:
            echo(f"  pid {pid} is not ours to signal; left running")
    for pid in orphaned_nodes(root):
        echo(f"cleaning up orphaned model node pid {pid}")
        try:
            if not _signal_and_wait(pid, signal.SIGTERM, 30.0):
                _signal_and_wait(pid, signal.SIGKILL, 10.0)
        except PermissionError:
            echo(f"  pid {pid} is not ours to signal; left running")


@contextlib.contextmanager
def _launching(state_dir: Path, wait_s: float):
    """Deal under the sweep's launch lock.

    Two launches at once -- a keeper's pass, and a `restart_campaign.sh` run beside it -- would each
    pick the same free slots before the 20 s pauses let the other's CARLAs bind them, and deal the
    same routes to both. The second waits here, then sees the first's workers and deals what is left.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / LAUNCH_LOCK).open("a") as handle:
        deadline = time.time() + wait_s
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise SystemExit("another launch of this sweep is still dealing; try again") from None
                time.sleep(1.0)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def launch(
    cfg: SiteConfig,
    model: str,
    n_workers: int,
    only: set[str] | None = None,
    exclude: set[str] | None = None,
    rebalance: bool = False,
    node_base: int = 9100,
    dry_run: bool = False,
    held: set[str] | None = None,
    tries: int = TRIES,
    batch: int = 0,
    cards: list[tuple[str, str]] | None = None,
    echo=print,
) -> list[dict]:
    """Deal what is left to free worker slots; see the module docstring.

    ``held`` routes are driven last and alone: one worker, once every other route has scored and no
    worker is alive, because a route that crashes CARLA does so every time (see the module docstring).
    A route set aside for every model, or one ``tries`` failed attempts have given up on, is dealt
    only when ``only`` names it. ``batch`` caps the routes a new worker is dealt. With ``cards``
    (CUDA index, Vulkan index), ``n_workers`` is per card, and each new worker goes to the card with
    the fewest. Everything but a dry run deals under the sweep's launch lock (:func:`_launching`).
    """
    if dry_run:
        return _launch(cfg, model, n_workers, only, exclude, rebalance, node_base, True, held, tries,
                       batch, cards, echo)  # fmt: skip
    with _launching(_state_dir(cfg, model), LOCK_WAIT_S):
        return _launch(cfg, model, n_workers, only, exclude, rebalance, node_base, False, held, tries,
                       batch, cards, echo)  # fmt: skip


def _launch(
    cfg: SiteConfig,
    model: str,
    n_workers: int,
    only: set[str] | None,
    exclude: set[str] | None,
    rebalance: bool,
    node_base: int,
    dry_run: bool,
    held: set[str] | None,
    tries: int,
    batch: int,
    cards: list[tuple[str, str]] | None,
    echo,
) -> list[dict]:
    root = _runs_root(cfg, model)
    state_dir = _state_dir(cfg, model)
    alive = workers(model, root)
    if rebalance and alive:
        if dry_run:
            echo(f"would stop {len(alive)} alive worker(s) to deal their routes anew")
            alive = []
        else:
            stop(cfg, model, echo, keeper_too=False)
            alive = workers(model, root)
    routes = _routes_meta(cfg)
    scenario_of = {rid: stype for rid, _, stype in routes}
    states = route_states(root)
    scored = {r for r, s in states.items() if s.scored}
    claimed = {r for w in alive for r in w.routes if r not in scored}
    ids = [rid for rid, _, _ in routes]
    # Set aside, or given up on: dealt again only when --only names it, on purpose.
    out = (exclude or set()) | ((set(SET_ASIDE) | given_up(states, tries)) - (only or set()))
    pending = select_pending(ids, scored, claimed, only, out | (held or set()))
    target = n_workers * (len(cards) if cards else 1)
    if not pending and held:
        if alive:
            echo(f"only held routes left; they wait for the {len(alive)} alive worker(s) to finish")
            return []
        pending = select_pending(ids, scored, claimed, held if only is None else held & only, out)
        target = 1
        if pending:
            echo(f"every other route has scored: driving the held route(s) {','.join(pending)} alone")
    free = max(0, target - len(alive))
    workers_note = f"{len(alive)} worker(s) alive, {free} slot(s) to fill"
    counted = [rid for rid in ids if rid not in SET_ASIDE]
    echo(f"{len(scored & set(counted))} / {len(counted)} scored; {len(pending)} to deal; {workers_note}")
    if not pending or not free:
        return []
    cost = expected_costs(states, scenario_of)
    if batch > 0:
        # The longest expected first, so the slow routes start early and the last hours are short ones;
        # the rest wait for the next launch, which deals them to whichever slots are free by then.
        pending = sorted(pending, key=lambda r: (-cost.get(r, 300.0), int(r)))[: free * batch]
    groups = balance(pending, lambda r: cost.get(r, 300.0), free, cap=batch or None)
    settings = cfg.benchmarks.get("bench2drive")
    base_carla = int(settings.get("carla_port") or 2000)
    base_tm = int(settings.get("tm_port") or 8000)
    taken = set()
    for w in alive:
        if w.carla_port:
            taken |= {w.carla_port, w.carla_port + 1, w.carla_port + 2}
    slots = free_slots(len(groups), base_carla, base_tm, node_base, taken, ports.is_free)
    if len(slots) < len(groups):
        raise SystemExit(f"only {len(slots)} free worker slot(s) for {len(groups)} worker(s)")
    # Workers per card so far, so each new one goes where there are fewest (the first card on a tie).
    on_card = {rank: sum(1 for w in alive if w.gpu_rank == rank) for _, rank in cards or []}
    base_text = cfg.source.read_text()
    launched = []
    for slot, group in zip(slots, groups, strict=False):
        carla, tm, node = slot_ports(slot, base_carla, base_tm, node_base)
        hours = sum(cost.get(r, 300.0) for r in group) / 3600
        card = min(cards, key=lambda c: (on_card[c[1]], cards.index(c))) if cards else None
        if card is not None:
            on_card[card[1]] += 1
        where = f"CARLA {carla}, TM {tm}" + (f", card {card[0]} (gpu_rank {card[1]})" if card else "")
        echo(f"worker slot {slot}: {len(group)} route(s), ~{hours:.1f} h, {where}: {','.join(group)}")
        if dry_run:
            continue
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "logs").mkdir(exist_ok=True)
        config = state_dir / f"worker{slot}.toml"
        config.write_text(with_ports(base_text, carla, tm, gpu_rank=card[1] if card else None))
        log = state_dir / "logs" / f"worker{slot}.log"
        env = {**os.environ, "PADDOCK_CONFIG": str(config), "PADDOCK_BASE_PORT": str(node)}
        if card is not None:
            # The node's card; CARLA's is gpu_rank above. Vulkan ignores CUDA_VISIBLE_DEVICES.
            env["CUDA_VISIBLE_DEVICES"] = card[0]
        command = [sys.executable, "-m", "paddock.cli", "run", "--bench", "bench2drive", "--model", model]
        command += ["--scene", ",".join(group)]
        with log.open("a") as handle:
            handle.write(f"\n=== {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC: {' '.join(command)}\n")
            handle.flush()
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # outlives the shell that launched it
                # A launch from a background shell inherits SIGINT ignored, and Python then installs no
                # KeyboardInterrupt -- `stop` would have nothing graceful to send. The harness turns
                # KeyboardInterrupt into killing its evaluator and CARLA, so give every worker one.
                preexec_fn=_default_sigint,
            )
        launched.append(
            {
                "slot": slot,
                "pid": proc.pid,
                "routes": group,
                "config": str(config),
                "log": str(log),
                **({"card": list(card)} if card else {}),
            }
        )
        if len(launched) < len(groups):
            time.sleep(STAGGER_S)
    if launched:
        manifest = state_dir / "workers.json"
        history = json.loads(manifest.read_text()) if manifest.exists() else []
        history.append({"launched_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", "workers": launched})
        manifest.write_text(json.dumps(history, indent=1))
    return launched


def keep(
    cfg: SiteConfig,
    model: str,
    n_workers: int,
    every_s: float = KEEP_EVERY_S,
    sleep: Callable[[float], None] = time.sleep,
    echo=print,
    **deal,
) -> int:
    """``launch`` every ``every_s`` until nothing is left to deal and no worker is alive.

    A slot a worker left -- its batch done, or the worker dead -- is filled at the next pass. With no
    worker alive, what a worker killed hard left behind is stopped first (``_reap_orphans``), or the
    next launch would wait for the memory an orphaned node holds. ``deal`` is ``launch``'s. Returns 0
    once settled, 1 when :data:`KEEP_STRIKES` launches in a row drove nothing.
    """
    root = _runs_root(cfg, model)
    pid_file = _state_dir(cfg, model) / KEEPER_PID
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")
    try:
        return _keep(cfg, model, n_workers, every_s, sleep, echo, root, deal)
    finally:
        with contextlib.suppress(OSError):
            if pid_file.read_text().split()[0] == str(os.getpid()):
                pid_file.unlink()


def _keep(cfg, model, n_workers, every_s, sleep, echo, root, deal) -> int:
    strikes, launched, attempts_then = 0, [], None
    while True:
        alive = workers(model, root)
        attempts = sum(st.attempts for st in route_states(root).values())
        if launched and not alive and attempts == attempts_then:
            strikes += 1
            if strikes >= KEEP_STRIKES:
                echo(
                    f"{datetime.now(UTC):%F %T} the last {strikes} launches' workers all exited without "
                    "attempting a route (a model node that cannot start?): stopping; read the worker logs"
                )
                return 1
        elif attempts != attempts_then:
            strikes = 0
        if not alive:
            _reap_orphans(root, echo)
        try:
            launched = launch(cfg, model, n_workers, echo=echo, **deal)
        except SystemExit as exc:  # no free slot this pass: CARLAs still releasing their ports
            launched = []
            echo(f"{datetime.now(UTC):%F %T} launch: {exc}; trying again in {every_s:.0f} s")
        attempts_then = attempts
        if not workers(model, root):
            echo(f"{datetime.now(UTC):%F %T} nothing left to deal and no worker alive: the sweep has settled")
            return 0
        sleep(every_s)


def _guard_one(pid: int, route_dir: Path, now: float, carla_missing_since: dict, once, dry_run, echo) -> None:
    """The guard's two tests for one evaluator, and the kill when one fires."""
    log = route_dir / "evaluator.log"
    try:
        silent = now - log.stat().st_mtime
    except OSError:
        return
    port = carla_port_from_log(log, 0)
    reason, carla_gone = None, False
    if silent > SILENT_S:
        reason = f"evaluator.log silent {silent:.0f} s"
    elif port and _process_age_s(pid) > CARLA_GONE_S and not carla_pids(port):
        first = carla_missing_since.setdefault(pid, now)
        if once or now - first >= CARLA_GONE_S:
            reason, carla_gone = f"its CARLA (port {port}) is gone", True
    else:
        carla_missing_since.pop(pid, None)
    if reason:
        echo(f"{datetime.now(UTC):%F %T} {route_dir.parent.name}/{route_dir.name}: {reason}; "
             f"{'would kill' if dry_run else 'killing'} evaluator pid {pid}")  # fmt: skip
        if not dry_run:
            # Before the kill: the worker writes the attempt's failure record when its evaluator
            # exits, and the reason is the difference between a crash and a hang in it.
            try:
                (route_dir / GUARD_KILL).write_text(
                    json.dumps({"reason": reason, "carla_gone": carla_gone}) + "\n"
                )
            except OSError:
                pass
            kill_evaluator(pid, route_dir)
        carla_missing_since.pop(pid, None)


def guard(cfg: SiteConfig, model: str, once: bool = False, dry_run: bool = False, echo=print) -> None:
    """Kill evaluators that cannot finish; see the module docstring for the two signs."""
    root = _runs_root(cfg, model)
    carla_missing_since: dict[int, float] = {}
    rule = f"log silent {SILENT_S:.0f} s, or CARLA gone {CARLA_GONE_S:.0f} s"
    echo(f"{datetime.now(UTC):%F %T} guard up over {root}: {rule}")
    while True:
        now = time.time()
        seen = set()
        for pid, route_dir in evaluators(root):
            seen.add(pid)
            # One evaluator's trouble -- its log gone mid-read, a process it may not
            # signal -- is that evaluator's; the guard goes on watching the others.
            try:
                _guard_one(pid, route_dir, now, carla_missing_since, once, dry_run, echo)
            except OSError as exc:
                echo(f"{datetime.now(UTC):%F %T} {route_dir.parent.name}/{route_dir.name}: {exc!r}; skipped")
        for pid in set(carla_missing_since) - seen:
            carla_missing_since.pop(pid)
        if once:
            return
        time.sleep(60.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("status", "launch", "keep", "stop", "guard"))
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=5,
        help="launch/keep: workers wanted in total, alive ones included; on each card with --cards",
    )
    ap.add_argument(
        "--tries",
        type=int,
        default=TRIES,
        help="launch/keep/status: failed attempts after which a route is not dealt again; 0: never",
    )
    ap.add_argument(
        "--batch", type=int, default=0, help="launch/keep: at most this many routes per new worker"
    )
    ap.add_argument(
        "--cards", help="launch/keep: CUDA:VULKAN index pairs to spread the workers over, e.g. 0:0,1:2"
    )
    ap.add_argument("--every", type=float, default=KEEP_EVERY_S, help="keep: seconds between launches")
    ap.add_argument("--only", help="launch: comma-separated route ids to consider (default: the routes file)")
    ap.add_argument("--exclude", help="launch: comma-separated route ids to leave out entirely")
    ap.add_argument("--held", help="launch: route ids to drive last, alone (routes that crash CARLA)")
    ap.add_argument(
        "--rebalance", action="store_true", help="launch: stop alive workers and deal everything anew"
    )
    ap.add_argument(
        "--node-base", type=int, default=9100, help="launch: first node port; slot k gets base + 10k"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="guard: one pass, then exit")
    args = ap.parse_args(argv)
    cfg = SiteConfig.load()
    deal = {
        "only": _ids(args.only),
        "exclude": _ids(args.exclude),
        "node_base": args.node_base,
        "held": _ids(args.held),
        "tries": args.tries,
        "batch": args.batch,
        "cards": _cards(args.cards),
    }
    if args.command == "status":
        status(cfg, args.model, tries=args.tries)
    elif args.command == "stop":
        stop(cfg, args.model, dry_run=args.dry_run)
    elif args.command == "guard":
        guard(cfg, args.model, once=args.once, dry_run=args.dry_run, echo=lambda s: print(s, flush=True))
    elif args.command == "keep" and args.dry_run:
        # One pass of what keep would deal: a dry run starts nothing, and keep would start workers.
        launch(cfg, args.model, args.workers, dry_run=True, **deal)
    elif args.command == "keep":
        return keep(
            cfg, args.model, args.workers, every_s=args.every, echo=lambda s: print(s, flush=True), **deal
        )
    else:
        launch(cfg, args.model, args.workers, rebalance=args.rebalance, dry_run=args.dry_run, **deal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
