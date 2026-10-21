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
* it exits non-zero only for ``Simulation crashed``; an agent crash exits 0.

CARLA and the Bench2Drive checkout are vendored outside the package; the
``[benchmarks.bench2drive]`` section of ``paddock.toml`` says where they live
(see ``docs/benchmarks/bench2drive.md``).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.bench2drive import early_stop
from paddock.benchmark import ClosedLoopBenchmark, Episode
from paddock.config import REPO_ROOT
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

#: The CARLA launch flag that changes what the cameras see. Bench2Drive 7ec25d1
#: launches CARLA at its default (Epic) quality; ``-quality-level=Low`` renders
#: the sky black and the wet-road decal layer as white speckle, so frames scored
#: that way are not the benchmark's frames. It is a one-line local edit to the
#: evaluator that leaves no trace in run.json, hence a preflight refusal.
QUALITY_FLAG = "-quality-level"

#: Spacing of the waypoints the node returns. Every closed-loop model plans on a
#: 0.5 s grid (see ``neuroncap/harness.py``); the agent turns waypoint spacing
#: into a target speed with it, so it is a measurement parameter, not a knob.
WAYPOINT_DT = 0.5

DEFAULT_ROUTES_FILE = "bench2drive220.xml"

#: The file that, appearing in a route's directory while it drives, ends the route
#: by hand (``early_stop.MANUAL``). The dashboard's "Stop route" button writes it.
STOP_REQUEST = "stop_request"

#: CARLA's client library as the 0.9.15 release ships it. Put on the client's
#: path when present, as the checkout's own ``run_evaluation.sh`` does, so a
#: site need not also ``pip install carla``.
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
) -> Iterator[tuple[str, str, str]]:
    """Filter routes by ``--scene`` (route id or town), ``--scenarios`` (type) and
    an optional site ``subset`` (route ids, from ``[benchmarks.bench2drive]
    routes_subset``) that restricts the run to a chosen set -- e.g. one route per
    scenario category -- without editing the routes file.

    ``scenes=None`` (or empty) means every route: the runner passes ``None``
    when ``--scene`` was not given, the same contract every closed-loop
    benchmark gets.

    Pure and separate from :meth:`Bench2DriveBenchmark.episodes` so the selection
    logic is unit-testable without a site config.
    """
    subset_ids = set(subset) if subset else None
    scene_filter = set(scenes) if scenes else None
    scenario_filter = set(scenarios) if scenarios else None
    for rid, town, stype in routes:
        if subset_ids is not None and rid not in subset_ids:
            continue
        if scene_filter is not None and rid not in scene_filter and town not in scene_filter:
            continue
        if scenario_filter is not None and stype not in scenario_filter:
            continue
        yield rid, town, stype


def crash_reason(status: str) -> str | None:
    """The failure message if ``status`` marks a route that did not run, else None."""
    for reason in CRASH_STATUSES:
        if status.endswith(reason):
            return reason
    return None


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
    all_records = checkpoint.get("records") or []
    if not all_records:
        raise NotScored(f"{results_json}: no records in _checkpoint (the route did not score)")

    records, crashed = [], []
    for record in all_records:
        status = str(record.get("status", ""))
        if crash_reason(status):
            crashed.append(f"{record.get('route_id', 'route')} status {status!r}")
        else:
            records.append(record)
    if not records:
        raise NotScored(
            f"{results_json}: {'; '.join(crashed)} -- "
            f"the route did not run, so its scores are not a measurement"
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
    """``ended_<reason>`` for one route (stuck, over_time, collision, manual): 1 for what ended it, else 0.

    Read from the ``early_stop.json`` the agent writes beside ``results.json``
    whenever the early stop is on -- with ``"reason": null`` for a route that went
    to Bench2Drive's own end -- so this reads only the route's directory, as
    ``--resume`` and the ingest need. Empty without one: the stop was off, and a
    0 would claim a rule was there to fire. Per route 0/1, so a run's aggregate is
    the share of its routes each rule ended.
    """
    try:
        record = json.loads((Path(route_dir) / early_stop.RECORD).read_text())
    except (OSError, ValueError):
        return {}
    reason = record.get("reason") if isinstance(record, dict) else None
    return {f"ended_{name}": 1.0 if reason == name else 0.0 for name in early_stop.REASONS}


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
        return repo / "leaderboard" / "data" / (self.setting("routes_file") or DEFAULT_ROUTES_FILE)

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
        """Per-tick front/BEV JPEGs and a log.csv: off unless asked for.

        Roughly 50 KB x 4000 frames x 220 routes, about 40 GB a sweep, plus an
        extra render on CARLA's side every tick.
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
            traj = json.loads((Path(expert) / route_id / "expert_traj.json").read_text())
            duration = float(traj["duration_s"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return round(rules["expert_factor"] * duration, 2) if duration > 0 else None

    @property
    def record_inputs(self) -> bool:
        """Each tick's whole request under ``inputs/`` (``recorder.write_inputs``): off unless asked for.

        ~3 MB a tick for six 800x450 views, a replay's worth of a few routes and not
        a sweep's; it needs ``record`` on, and changes nothing the model is shown.
        """
        return (self.setting("record_inputs") or "false").strip().lower() in ("1", "true", "yes", "on")

    def _pythonpath(self) -> list[str]:
        """What the client interpreter needs, in the checkout's own order.

        ``REPO_ROOT/src`` first, as for every process paddock launches: the
        agent imports the wire contract from ``paddock.policy.protocol`` and the
        tracker from this package rather than restating either.
        """
        carla_root, repo = self.carla_root, self.repo
        entries = [str(REPO_ROOT / "src")]
        if carla_root is not None:
            if (carla_root / CARLA_EGG).exists():
                entries.append(str(carla_root / CARLA_EGG))
            entries.append(str(carla_root / "PythonAPI" / "carla"))
        if repo is not None:
            entries += [str(repo / "leaderboard"), str(repo / "scenario_runner")]
        return entries

    def _environment(self, extra: dict[str, str]) -> dict[str, str]:
        """The evaluator's environment: the host passthrough set, plus ours.

        Built like ``runtime.env.build``, not inherited wholesale -- a stray
        PYTHONPATH from paddock's own env would make the client half-import
        paddock's dependencies. ``CUDA_VISIBLE_DEVICES`` passes through so the
        CARLA the evaluator launches honours how the sweep was pinned.
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
                "no [benchmarks.bench2drive] section in paddock.toml; it needs `python`, "
                "`carla_root` and `repo` (see docs/benchmarks/bench2drive.md)"
            ]

        for key in ("python", "carla_root", "repo"):
            if not self.setting(key):
                problems.append(f"[benchmarks.bench2drive] {key} is not set in paddock.toml")

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

        timeout = self.setting("per_route_timeout_s")
        if timeout and not timeout.strip().isdigit():
            problems.append(
                f"[benchmarks.bench2drive] per_route_timeout_s must be an integer, got {timeout!r}"
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
            config["early_stop"] = self.early_stop
        return config

    def remarks(self) -> list[str]:
        """The early stop, when on: which rules, and why a route they end is a failed drive."""
        rules = self.early_stop
        return [early_stop.remark(rules)] if rules else []

    # ---- episodes ------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        routes_file = self.routes_file
        if routes_file is None or not routes_file.exists():
            return
        routes = parse_routes(routes_file)
        selected = list(select_routes(routes, scenes, scenarios, self.routes_subset))
        # Recorded for `invalid_reasons`: the site's whole route set, against what ran.
        self._routes_scope = (len(selected), len(list(select_routes(routes, None, None, self.routes_subset))))
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
        record (`run_config`) and every full run on the site uses it. An early
        stop is not a reason: a route it ends is a failed drive (`remarks`).
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
        runs = int(episode.config.get("runs", 1))
        timeout_s = int(self.setting("per_route_timeout_s") or DEFAULT_ROUTE_TIMEOUT_S)
        requested_port = int(self.setting("carla_port") or 2000)

        out_dir.mkdir(parents=True, exist_ok=True)
        results_json = out_dir / "results.json"
        log_path = out_dir / "evaluator.log"

        extra = {
            "PADDOCK_NODE_URL": policy_url,
            "PADDOCK_CAMERAS": ",".join(cameras),
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

        if timed_out:
            raise RuntimeError(
                f"{episode.episode_id}: timed out after {timeout_s / 60:.0f} min "
                f"(per_route_timeout_s); evaluator killed\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        if returncode != 0:
            raise RuntimeError(
                f"{episode.episode_id}: evaluator exited {returncode} after {elapsed / 60:.1f} min"
                f"\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        if not results_json.exists():
            raise RuntimeError(
                f"{episode.episode_id}: evaluator exited 0 but wrote no results.json"
                f"\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )

        try:
            metrics = parse_results(results_json)
        except NotScored as exc:
            raise RuntimeError(
                f"{episode.episode_id}: {exc}\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            ) from exc
        metrics["duration_s"] = round(elapsed, 1)
        metrics.update(ended_early(out_dir))
        self._note_stop(episode, metrics)
        return metrics


#: Paddock looks for this name when loading an adapter.
BENCHMARK = Bench2DriveBenchmark
