"""Reading recorded Bench2Drive routes: one place for the renderer, the dashboard and the analysis.

A sweep is usually several paddock runs (shards, re-drives after a crash), each
``<runs>/bench2drive/<model>/<run_id>/<route>-<Town>-<Scenario>/``. Everything
here works from those directories and the Bench2Drive checkout alone:

* :func:`route_table` -- town, scenario, weather and length of every route in the
  routes file, plus the abilities Bench2Drive scores it under;
* :func:`attempts` / :func:`latest` -- every attempt at each route across any number
  of run directories, and the one that counts: the latest that *scored*, the rule
  ``paddock ingest`` applies, so a crashed attempt never shadows its re-drive;
* :func:`load_result`, :func:`load_ticks`, :func:`load_json` -- the per-route files,
  tolerant of a route still being written.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bench2Drive's multi-ability grouping, as ``tools/ability_benchmark.py`` defines it
#: at 7ec25d1 (``Ability``). A scenario can count towards more than one ability.
ABILITIES: dict[str, tuple[str, ...]] = {
    "Overtaking": (
        "Accident", "AccidentTwoWays", "ConstructionObstacle", "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays", "HazardAtSideLane", "ParkedObstacleTwoWays", "ParkedObstacle",
        "VehicleOpensDoorTwoWays",
    ),
    "Merging": (
        "CrossingBicycleFlow", "EnterActorFlow", "HighwayExit", "InterurbanActorFlow", "HighwayCutIn",
        "InterurbanAdvancedActorFlow", "MergerIntoSlowTrafficV2", "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn", "NonSignalizedJunctionLeftTurnEnterFlow",
        "ParkingExit", "SequentialLaneChange", "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
    ),
    "Emergency_Brake": (
        "BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute", "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight", "ParkingCutIn", "PedestrianCrossing", "ParkingCrossingPedestrian",
        "StaticCutIn", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "ControlLoss",
    ),
    "Give_Way": ("InvadingTurn", "YieldToEmergencyVehicle"),
    "Traffic_Signs": (
        "BlockedIntersection", "OppositeVehicleTakingPriority", "OppositeVehicleRunningRedLight",
        "PedestrianCrossing", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "EnterActorFlow",
        "CrossingBicycleFlow", "NonSignalizedJunctionLeftTurn", "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow", "SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow", "T_Junction", "VanillaNonSignalizedTurn",
        "VanillaSignalizedTurnEncounterGreenLight", "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurnEncounterStopsign",
    ),
}  # fmt: skip

#: CARLA leaderboard infraction multipliers (``statistics_manager.py`` at 7ec25d1), by results.json key.
#: The same table as ``paddock.adapters.benchmarks.bench2drive.tally.INFRACTION_PENALTY``.
PENALTY = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "scenario_timeouts": 0.70,
    "yield_emergency_vehicle_infractions": 0.70,
    "stop_infraction": 0.80,
}

#: Statuses that mean the route did not run (``harness.CRASH_STATUSES``).
CRASH_SUFFIXES = (
    "Simulation crashed",
    "Agent's sensors were invalid",
    "Agent couldn't be set up",
    "Agent crashed",
)

_ROUTE_DIR = re.compile(r"^(?P<rid>\d+)-(?P<town>Town\w+?)-(?P<scenario>\w+)$")
_LOCATION = re.compile(r"x=(-?[\d.]+), y=(-?[\d.]+)")


def abilities(scenario: str) -> list[str]:
    return [name for name, members in ABILITIES.items() if scenario in members]


def route_table(routes_xml: Path) -> dict[str, dict[str, Any]]:
    """``{route_id: {town, scenario, weather, length_m, waypoints, abilities}}`` from a routes file."""
    table: dict[str, dict[str, Any]] = {}
    for route in ET.parse(routes_xml).getroot().iter("route"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in route.find("waypoints")]
        scenario_el = route.find("scenarios/scenario")
        scenario = scenario_el.get("type") if scenario_el is not None else "route"
        weather = route.find("weathers/weather")
        table[route.get("id")] = {
            "route_id": route.get("id"),
            "town": route.get("town"),
            "scenario": scenario,
            "weather": dict(weather.attrib) if weather is not None else {},
            "length_m": round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False)), 1),
            "waypoints": pts,
            "abilities": abilities(scenario),
        }
    return table


@dataclass
class Attempt:
    """One drive of one route: a ``<run>/<route>-<Town>-<Scenario>/`` directory."""

    route_id: str
    town: str
    scenario: str
    run_id: str
    path: Path
    result: dict | None = field(default=None, repr=False)

    @property
    def status(self) -> str:
        return (self.result or {}).get("status", "running" if self.running else "no result")

    @property
    def running(self) -> bool:
        """No record yet, and the evaluator log grew in the last two minutes."""
        if self.result is not None:
            return False
        log = self.path / "evaluator.log"
        import time

        return log.exists() and time.time() - log.stat().st_mtime < 120

    @property
    def scored(self) -> bool:
        return self.result is not None and not any(self.status.endswith(s) for s in CRASH_SUFFIXES)


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_result(route_dir: Path) -> dict | None:
    """The route's record from ``results.json`` (first repetition), or None while there is none."""
    data = load_json(Path(route_dir) / "results.json")
    records = ((data or {}).get("_checkpoint") or {}).get("records") or []
    return records[0] if records else None


def load_ticks(route_dir: Path, every: int = 1) -> list[dict]:
    """``ticks.jsonl`` as dicts; a half-written last line (a route still driving) is skipped."""
    path = Path(route_dir) / "ticks.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as handle:
        for n, line in enumerate(handle):
            if n % every:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                break
    return out


def infraction_points(result: dict) -> list[dict]:
    """Every infraction in a record with its kind, text and (x, y) where the leaderboard logged one."""
    points = []
    for kind, events in (result.get("infractions") or {}).items():
        if kind == "min_speed_infractions":
            continue
        for text in events if isinstance(events, list) else []:
            m = _LOCATION.search(text)
            points.append(
                {
                    "kind": kind,
                    "text": text,
                    "x": float(m.group(1)) if m else None,
                    "y": float(m.group(2)) if m else None,
                    "penalty": PENALTY.get(kind),
                }
            )
    return points


def attempts(run_dirs: list[Path]) -> list[Attempt]:
    """Every route directory under the given run directories (or directories of runs)."""
    found: list[Attempt] = []
    for root in run_dirs:
        root = Path(root)
        runs = (
            [root]
            if any(_ROUTE_DIR.match(p.name) for p in root.iterdir() if p.is_dir())
            else [p for p in sorted(root.iterdir()) if p.is_dir()]
        )
        for run in runs:
            for route_dir in sorted(run.iterdir()):
                m = _ROUTE_DIR.match(route_dir.name)
                if m and route_dir.is_dir():
                    found.append(
                        Attempt(
                            m["rid"], m["town"], m["scenario"], run.name, route_dir, load_result(route_dir)
                        )
                    )
    return found


def latest(all_attempts: list[Attempt]) -> dict[str, Attempt]:
    """Per route, the latest attempt that scored; else the latest attempt of any kind.

    Run ids start with a UTC timestamp, so they sort in time order.
    """
    best: dict[str, Attempt] = {}
    for att in sorted(all_attempts, key=lambda a: a.run_id):
        current = best.get(att.route_id)
        if current is None or att.scored or not current.scored:
            best[att.route_id] = att
    return best
