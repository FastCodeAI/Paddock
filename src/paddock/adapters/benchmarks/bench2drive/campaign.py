"""Drive a Bench2Drive sweep with N parallel workers, and keep it moving through reboots and hangs.

    C="python -m paddock.adapters.benchmarks.bench2drive.campaign"
    $C status --model openpilot
    $C launch --model openpilot --workers 5 [--held 11715] [--only ids] [--exclude ids] [--dry-run]
    $C launch --model openpilot --workers 5 --rebalance
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
  its CARLA server is gone -- when CARLA dies inside ``spawn_parked_vehicles``
  the evaluator logs a "failed to destroy actor" line every 600 s for hours,
  so silence alone never fires.

All four go only after this sweep's own processes: workers are found by their
command line *and* by the runs root of their ``PADDOCK_CONFIG``, and ``stop``
signals only those whose ``PADDOCK_CONFIG`` is one of the campaign's own
``worker<slot>.toml`` (a run started by hand on the same runs root keeps its
routes from being dealt twice, and is left running); evaluators by their
``--checkpoint`` lying under it; CARLA by the port in that evaluator's log. A
process whose environment cannot be read -- another user's -- is never ours.
The evaluator's own crash handler kills every CARLA on its GPU index
(``harness`` docstring), so routes known to crash CARLA are passed as
``--held``: they are driven last, by one worker, once nothing else is left.

Run it in paddock's own interpreter; ``PADDOCK_CONFIG`` picks the site config
the workers copy, exactly as for ``paddock run``.
"""

from __future__ import annotations

import argparse
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
    DEFAULT_ROUTES_FILE,
    NotScored,
    carla_pids,
    carla_port_from_log,
    parse_results,
    parse_routes,
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
                continue
            st.scored = True
            try:
                records = json.loads(results.read_text())["_checkpoint"]["records"]
                st.durations += [float((r.get("meta") or {}).get("duration_system") or 0.0) for r in records]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return states


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


def balance(routes: list[str], cost: Callable[[str], float], n: int) -> list[list[str]]:
    """Deal ``routes`` to ``n`` workers longest-expected-first (LPT); each worker's list sorted by id.

    Ties break on the route id, so the same inputs always give the same split.
    """
    n = max(1, min(n, len(routes)))
    loads = [0.0] * n
    groups: list[list[str]] = [[] for _ in range(n)]
    for rid in sorted(routes, key=lambda r: (-cost(r), int(r))):
        k = min(range(n), key=lambda i: (loads[i], i))
        groups[k].append(rid)
        loads[k] += cost(rid)
    return [sorted(g, key=int) for g in groups if g]


def with_ports(toml_text: str, carla_port: int, tm_port: int) -> str:
    """The site config with ``[benchmarks.bench2drive]`` carla_port / tm_port set; everything else verbatim.

    Edited as text, not re-serialised, so the copy keeps every comment and key of the original and a
    reader can diff the two. Keys that are absent are added under the table header.
    """
    lines = toml_text.splitlines(keepends=True)
    header = next(
        (i for i, ln in enumerate(lines) if re.match(r"^\[benchmarks\.bench2drive\]\s*(#.*)?$", ln)), None
    )
    if header is None:
        raise ValueError("the site config has no [benchmarks.bench2drive] table")
    end = next((i for i in range(header + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
    wanted = {"carla_port": str(carla_port), "tm_port": str(tm_port)}
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
        found.append(Worker(pid, [r for r in scene.split(",") if r], cfg_path, int(port) if port else None))
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
    routes_file = Path(repo) / "leaderboard" / "data" / (settings.get("routes_file") or DEFAULT_ROUTES_FILE)
    return parse_routes(routes_file)


def _ids(text: str | None) -> set[str] | None:
    return {r.strip() for r in text.split(",") if r.strip()} if text else None


def status(cfg: SiteConfig, model: str, echo=print) -> dict:
    root = _runs_root(cfg, model)
    routes = _routes_meta(cfg)
    states = route_states(root)
    alive = workers(model, root)
    driving = {_ROUTE_DIR.match(d.name)["rid"] for _, d in evaluators(root) if _ROUTE_DIR.match(d.name)}
    scored = {r for r, s in states.items() if s.scored}
    claimed = {r for w in alive for r in w.routes if r not in scored}
    left = [rid for rid, _, _ in routes if rid not in scored]
    echo(f"{model} on bench2drive: {len(scored)} / {len(routes)} routes scored, {len(left)} left ({root})")
    for w in alive:
        mine = [r for r in w.routes if r not in scored]
        echo(f"  worker pid {w.pid} (CARLA port {w.carla_port}): {len(mine)} of {len(w.routes)} routes left")
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
    }


def stop(cfg: SiteConfig, model: str, echo=print) -> int:
    """Stop this sweep's workers: SIGINT first, which the harness turns into killing its evaluator + CARLA.

    Only the workers ``launch`` started, known by their ``PADDOCK_CONFIG``: one of the
    campaign's ``worker<slot>.toml``. A ``paddock run`` of the same model started by
    hand counts for ``launch`` (its routes are not dealt twice) but is not ours to stop.
    """
    root = _runs_root(cfg, model)
    state_dir = _state_dir(cfg, model).resolve()
    alive = [
        w
        for w in workers(model, root)
        if w.config.resolve().parent == state_dir and w.config.name.startswith("worker")
    ]
    for w in alive:
        echo(f"stopping worker pid {w.pid} ({len(w.routes)} routes queued)")
        if not _signal_and_wait(w.pid, signal.SIGINT, 180.0):
            echo(f"  pid {w.pid} ignored SIGINT for 3 min; SIGTERM")
            if not _signal_and_wait(w.pid, signal.SIGTERM, 30.0):
                _signal_and_wait(w.pid, signal.SIGKILL, 10.0)
    # A worker killed past its SIGINT handler leaves its evaluator (own session) and CARLA behind.
    for pid, route_dir in evaluators(root):
        echo(f"cleaning up orphaned evaluator pid {pid} ({route_dir.name})")
        kill_evaluator(pid, route_dir)
    return len(alive)


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
    echo=print,
) -> list[dict]:
    """Deal what is left to free worker slots; see the module docstring.

    ``held`` routes are driven last and alone: one worker, once every other route has scored and no
    worker is alive, because a route that crashes CARLA can take every other worker's CARLA with it
    (the evaluator's crash handler kills all CARLA on its GPU index).
    """
    root = _runs_root(cfg, model)
    state_dir = _state_dir(cfg, model)
    alive = workers(model, root)
    if rebalance and alive:
        if dry_run:
            echo(f"would stop {len(alive)} alive worker(s) to deal their routes anew")
            alive = []
        else:
            stop(cfg, model, echo)
            alive = workers(model, root)
    routes = _routes_meta(cfg)
    scenario_of = {rid: stype for rid, _, stype in routes}
    states = route_states(root)
    scored = {r for r, s in states.items() if s.scored}
    claimed = {r for w in alive for r in w.routes if r not in scored}
    ids = [rid for rid, _, _ in routes]
    pending = select_pending(ids, scored, claimed, only, (exclude or set()) | (held or set()))
    if not pending and held:
        if alive:
            echo(f"only held routes left; they wait for the {len(alive)} alive worker(s) to finish")
            return []
        pending = select_pending(ids, scored, claimed, held if only is None else held & only, exclude)
        n_workers = 1
        if pending:
            echo(f"every other route has scored: driving the held route(s) {','.join(pending)} alone")
    free = max(0, n_workers - len(alive))
    workers_note = f"{len(alive)} worker(s) alive, {free} slot(s) to fill"
    echo(f"{len(scored)} / {len(routes)} scored; {len(pending)} to deal; {workers_note}")
    if not pending or not free:
        return []
    cost = expected_costs(states, scenario_of)
    groups = balance(pending, lambda r: cost.get(r, 300.0), free)
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
    base_text = cfg.source.read_text()
    launched = []
    for slot, group in zip(slots, groups, strict=False):
        carla, tm, node = slot_ports(slot, base_carla, base_tm, node_base)
        hours = sum(cost.get(r, 300.0) for r in group) / 3600
        where = f"CARLA {carla}, TM {tm}"
        echo(f"worker slot {slot}: {len(group)} route(s), ~{hours:.1f} h, {where}: {','.join(group)}")
        if dry_run:
            continue
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "logs").mkdir(exist_ok=True)
        config = state_dir / f"worker{slot}.toml"
        config.write_text(with_ports(base_text, carla, tm))
        log = state_dir / "logs" / f"worker{slot}.log"
        env = {**os.environ, "PADDOCK_CONFIG": str(config), "PADDOCK_BASE_PORT": str(node)}
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
            {"slot": slot, "pid": proc.pid, "routes": group, "config": str(config), "log": str(log)}
        )
        if len(launched) < len(groups):
            time.sleep(STAGGER_S)
    if launched:
        manifest = state_dir / "workers.json"
        history = json.loads(manifest.read_text()) if manifest.exists() else []
        history.append({"launched_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}", "workers": launched})
        manifest.write_text(json.dumps(history, indent=1))
    return launched


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
            log = route_dir / "evaluator.log"
            try:
                silent = now - log.stat().st_mtime
            except OSError:
                continue
            port = carla_port_from_log(log, 0)
            reason = None
            if silent > SILENT_S:
                reason = f"evaluator.log silent {silent:.0f} s"
            elif port and _process_age_s(pid) > CARLA_GONE_S and not carla_pids(port):
                first = carla_missing_since.setdefault(pid, now)
                if once or now - first >= CARLA_GONE_S:
                    reason = f"its CARLA (port {port}) is gone"
            else:
                carla_missing_since.pop(pid, None)
            if reason:
                echo(f"{datetime.now(UTC):%F %T} {route_dir.parent.name}/{route_dir.name}: {reason}; "
                     f"{'would kill' if dry_run else 'killing'} evaluator pid {pid}")  # fmt: skip
                if not dry_run:
                    kill_evaluator(pid, route_dir)
                carla_missing_since.pop(pid, None)
        for pid in set(carla_missing_since) - seen:
            carla_missing_since.pop(pid)
        if once:
            return
        time.sleep(60.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("status", "launch", "stop", "guard"))
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--workers", type=int, default=5, help="launch: workers wanted in total, alive ones included"
    )
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
    if args.command == "status":
        status(cfg, args.model)
    elif args.command == "stop":
        stop(cfg, args.model)
    elif args.command == "guard":
        guard(cfg, args.model, once=args.once, dry_run=args.dry_run, echo=lambda s: print(s, flush=True))
    else:
        launch(
            cfg,
            args.model,
            args.workers,
            only=_ids(args.only),
            exclude=_ids(args.exclude),
            rebalance=args.rebalance,
            node_base=args.node_base,
            dry_run=args.dry_run,
            held=_ids(args.held),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
