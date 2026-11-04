"""Draw the regression gate's smoke subsets: varied groups, random within each.

    python -m tools.gate_subsets neuroncap
    python -m tools.gate_subsets nuscenes    --samples <full nuScenes run>/samples.jsonl --nuscenes <nuScenes root>
    python -m tools.gate_subsets navsim      --samples <full navtest run>/samples.jsonl --logs <navsim_logs/test>
    python -m tools.gate_subsets bench2drive --routes <Bench2Drive>/leaderboard/data/bench2drive220.xml
    python -m tools.gate_subsets alpasim     --suites <alpasim>/data/scenes/sim_suites.csv --scene-cache <scene_cache>
    python -m tools.gate_subsets kitscenes   --samples <full KITScenes run>/samples.jsonl
    python -m tools.gate_subsets quick_closed --routes <Bench2Drive>/leaderboard/data/bench2drive220.xml
    python -m tools.gate_subsets alpasim_replace --suites <alpasim>/data/scenes/sim_suites.csv --scene-cache <scene_cache>

Each writes `tools/gate_smoke/<name>.toml` (AlpaSim also its scene set), with the
rule in its header. The rule, everywhere (agreed 2026-10-22):

1. Split the benchmark into groups that change how a model has to drive: the
   maneuver (left, right, straight), the speed band, the city or town, the
   crash type, the Bench2Drive scenario family -- whatever the benchmark records.
2. Give every group the same share (`gate.balanced_pick`), so a rare group, a
   right turn at speed, is as present in the gate as straight driving. The
   gate's mean then weighs rare driving more than the full benchmark does; it is
   a test for what broke, not an estimate of the full score.
3. Draw at random within each group, seeded with `SEED`, so the pick can be
   audited and repeated, and no one chose a sample.

A subset is drawn once. Re-drawing it to make a result look better is how a gate
gets steered; a replacement goes in its own commit that says why, and every
baseline paired with it is run again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.gate import SMOKE_DIR, balanced_pick, load_subset  # noqa: E402

#: The seed of every draw made on 2026-10-22.
SEED = 20261022
#: Speed bands, m/s: slow under 3 (11 km/h), fast from 8 (29 km/h), medium between.
SPEED_BANDS_MPS = (3.0, 8.0)

HEADER_TAIL = """#
# Drawn once, on {date}, by `python -m tools.gate_subsets {name}` (seed {seed}): every
# group the same share, random within it (tools/gate_subsets.py has the rule). It must
# never be re-picked, re-drawn or edited to make a result look better: a gate whose
# subset moves can be steered. If it turns out to be wrong (a sample that cannot run,
# say), replace it in its own commit that says why, and re-run every baseline paired
# with it.
"""
DATE = "2026-10-22"


def speed_band(speed_mps: float) -> str:
    slow, fast = SPEED_BANDS_MPS
    return "slow" if speed_mps < slow else "fast" if speed_mps >= fast else "medium"


def path_speed(points: Iterable[Iterable[float]], seconds: float) -> float:
    """Mean speed along a path that starts at the origin: its length over `seconds`."""
    xy = [(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in points]
    return sum(math.dist(a, b) for a, b in zip(xy, xy[1:])) / seconds


def _toml_value(v) -> str:
    return json.dumps(v)


def write_subset(
    name: str,
    header: str,
    meta: dict,
    strata: dict[str, list[str]],
    directory: Path = SMOKE_DIR,
    date: str = DATE,
    seed: int = SEED,
    command: str | None = None,
) -> Path:
    """`tools/gate_smoke/<name>.toml`: the header comment, the meta keys, then the groups."""
    lines = [f"# {line}" if line else "#" for line in header.strip("\n").splitlines()]
    command = command or name.split("_")[0]
    lines.append(HEADER_TAIL.format(date=date, name=command, seed=seed).rstrip("\n"))
    lines.append("")
    lines += [f"{k} = {_toml_value(v)}" for k, v in meta.items()]
    lines += ["", "[strata]"]
    for group in sorted(strata):
        ids = ", ".join(json.dumps(i) for i in strata[group])
        lines.append(f"{json.dumps(group)} = [{ids}]")
    path = directory / f"{name}.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _counts(groups: dict[str, list[str]]) -> str:
    return ", ".join(f"{g} {len(v)}" for g, v in sorted(groups.items()))


# ---------------------------------------------------------------------------
# neuroNCAP: crash type; pairs from the official set, one run each.
# ---------------------------------------------------------------------------


def neuroncap_groups() -> dict[str, list[str]]:
    from paddock.adapters.benchmarks.neuroncap.harness import OFFICIAL_SET

    return {shape: [f"{shape}-{scene}" for scene in scenes] for shape, scenes in OFFICIAL_SET.items()}


def draw_neuroncap(directory: Path = SMOKE_DIR) -> list[Path]:
    groups = neuroncap_groups()
    closed = balanced_pick(groups, 15, SEED)
    quick = balanced_pick(groups, 3, SEED + 1)
    common = (
        "The unit is the scenario-scene pair. Runs are seeded 0..R-1, so a one-run subset is\n"
        "run 0 of every full sweep and pairs with one directly. Run as `--scene <its scenes>\n"
        "--runs 1` with PADDOCK_NCAP_PAIRS naming the pairs (`--scene` alone would run every\n"
        "shape each scene has) and PADDOCK_NCAP_MANAGE_RENDERER=true. Stored valid=false."
    )
    return [
        write_subset(
            "neuroncap",
            "neuroNCAP closed-loop tier: 15 of the 20 official scenario-scene pairs, 1 run each.\n"
            f"Groups: the crash type ({_counts(groups)} pairs); 5 drawn from each, so frontal\n"
            "and side are all in and stationary is 5 of its 10.\n" + common,
            {"bench": "neuroncap", "unit": "scenario-scene pair", "seed": SEED, "runs": 1, "count": 15},
            closed,
            directory,
        ),
        write_subset(
            "neuroncap_quick",
            "neuroNCAP quick tier: 3 of the 20 official scenario-scene pairs, one per crash type,\n"
            "1 run each -- 3 episodes, about 20 min. With 3 units the bar is |t| >= 4.303: it\n"
            "catches a large break, not a small regression. It can fail the gate but not ship\n"
            "it (can_ship = false); the closed tier's 15 decide.\n" + common,
            {
                "bench": "neuroncap",
                "can_ship": False,
                "unit": "scenario-scene pair",
                "seed": SEED + 1,
                "runs": 1,
                "count": 3,
            },
            quick,
            directory,
        ),
    ]


# ---------------------------------------------------------------------------
# nuScenes: city x maneuver x speed, from a full run's stored samples.
# ---------------------------------------------------------------------------

#: `Command` codes as the nuScenes harness records them (`paddock.policy.protocol`).
NUSCENES_MANEUVER = {0: "right", 1: "left", 2: "straight"}


def nuscenes_locations(root: Path, version: str = "v1.0-trainval") -> dict[str, str]:
    """Scene name -> city area, from nuScenes' own scene and log tables."""
    tables = root / version
    logs = {r["token"]: r["location"] for r in json.loads((tables / "log.json").read_text())}
    return {s["name"]: logs[s["log_token"]] for s in json.loads((tables / "scene.json").read_text())}


def nuscenes_groups(samples: Iterable[dict], locations: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for s in samples:
        nid = s["native_id"]
        speed = path_speed(s["ground_truth"], 3.0)
        key = f"{locations[nid['scene']]}/{NUSCENES_MANEUVER[int(nid['command'])]}/{speed_band(speed)}"
        groups.setdefault(key, []).append(s["sample_id"])
    return groups


def draw_nuscenes(samples_path: Path, root: Path, n: int = 1000, directory: Path = SMOKE_DIR) -> Path:
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    groups = nuscenes_groups(samples, nuscenes_locations(root))
    picked = balanced_pick(groups, n, SEED)
    scenes = {i.split("/")[2] for ids in picked.values() for i in ids}
    return write_subset(
        "nuscenes",
        f"nuScenes open-loop smoke tier: {n} of the {len(samples)} scorable val keyframes, from\n"
        f"{len(scenes)} of the 150 val scenes. Groups: city area x maneuver x speed --\n"
        "the maneuver is the harness's own command, read from the recorded 3 s future\n"
        "(left, right, straight); the speed is that future's length over 3 s, in bands\n"
        f"slow < {SPEED_BANDS_MPS[0]:g} m/s <= medium < {SPEED_BANDS_MPS[1]:g} m/s <= fast. "
        f"{len(groups)} groups.\n"
        f"Read from the full run {samples_path.parent.name} (every sample, with its ground\n"
        "truth) and nuScenes' scene and log tables. Run as `--limit <count>` with\n"
        "PADDOCK_NUSCENES_SAMPLES naming the ids. Stored valid=false (--limit). The unit is\n"
        "the scene; samples pair by id with any run of the same config, a full one included.",
        {"bench": "nuscenes", "unit": "scene", "seed": SEED, "count": n},
        picked,
        directory,
    )


# ---------------------------------------------------------------------------
# NAVSIM: city x driving command x speed, from navtest's own logs.
# ---------------------------------------------------------------------------

#: NAVSIM's one-hot `driving_command` order (docs/agents.md): left, straight, right, unknown.
NAVSIM_COMMANDS = ("left", "straight", "right", "unknown")
#: The scorer dies below two scenarios in a log (navsim harness MIN_SCENARIOS).
NAVSIM_LOG_FLOOR = 2


def navsim_frames(logs_dir: Path, tokens: set[str]) -> dict[str, dict]:
    """token -> {log, city, command, speed} for every navtest token, from the log pickles."""
    out: dict[str, dict] = {}
    for path in sorted(logs_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            frames = pickle.load(handle)
        for f in frames:
            if f["token"] in tokens:
                vx, vy = f["ego_dynamic_state"][:2]
                command = int(max(range(4), key=lambda i: f["driving_command"][i]))
                out[f["token"]] = {
                    "log": f["log_name"],
                    "city": f["map_location"],
                    "command": NAVSIM_COMMANDS[command],
                    "speed": math.hypot(vx, vy),
                }
    return out


def navsim_group(frame: dict) -> str:
    return f"{frame['city']}/{frame['command']}/{speed_band(frame['speed'])}"


def navsim_groups(frames: dict[str, dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for token, f in frames.items():
        groups.setdefault(navsim_group(f), []).append(token)
    return groups


def navsim_log_floor(
    picked: dict[str, list[str]], frames: dict[str, dict], seed: int
) -> dict[str, list[str]]:
    """Top every drawn log up to `NAVSIM_LOG_FLOOR` tokens, from its own group where it can.

    A log scored with one scenario kills NAVSIM's scorer, so a log the draw touched
    once gets one more of its tokens, drawn the same seeded way.
    """
    import random

    chosen = {t for ids in picked.values() for t in ids}
    by_log: dict[str, list[str]] = {}
    for t in sorted(frames):
        by_log.setdefault(frames[t]["log"], []).append(t)
    out = {g: list(ids) for g, ids in picked.items()}
    group_of = {t: g for g, ids in picked.items() for t in ids}
    for log, tokens in sorted(by_log.items()):
        mine = [t for t in tokens if t in chosen]
        if not mine or len(mine) >= NAVSIM_LOG_FLOOR:
            continue
        rest = [t for t in tokens if t not in chosen]
        group = group_of[mine[0]]
        same = [t for t in rest if navsim_group(frames[t]) == group]
        pool = same or rest
        extra = random.Random(f"{seed}:floor:{log}").sample(
            pool, min(len(pool), NAVSIM_LOG_FLOOR - len(mine))
        )
        for t in extra:
            out.setdefault(navsim_group(frames[t]), []).append(t)
            chosen.add(t)
    return {g: sorted(ids) for g, ids in out.items()}


def draw_navsim(samples_path: Path, logs_dir: Path, n: int = 1000, directory: Path = SMOKE_DIR) -> Path:
    tokens = {
        json.loads(line)["native_id"]["token"] for line in samples_path.read_text().splitlines() if line
    }
    frames = navsim_frames(logs_dir, tokens)
    if missing := tokens - set(frames):
        raise SystemExit(f"{len(missing)} navtest tokens are in no log under {logs_dir}")
    groups = navsim_groups(frames)
    picked = navsim_log_floor(balanced_pick(groups, n, SEED), frames, SEED)
    count = sum(len(v) for v in picked.values())
    logs = {frames[t]["log"] for ids in picked.values() for t in ids}
    return write_subset(
        "navsim",
        f"NAVSIM v2 smoke tier: {count} of navtest's {len(tokens)} scenarios, from {len(logs)} of its\n"
        f"{len({f['log'] for f in frames.values()})} logs. Groups: city x driving command x speed, read from\n"
        "the frame each token names in navtest's own log pickles (`map_location`; the one-hot\n"
        "`driving_command`, left / straight / right / unknown; the speed from\n"
        f"`ego_dynamic_state`, slow < {SPEED_BANDS_MPS[0]:g} m/s <= medium < {SPEED_BANDS_MPS[1]:g} m/s <= fast). "
        f"{len(groups)} groups; {n} drawn,\n"
        f"then every log the draw touched once topped up to {NAVSIM_LOG_FLOOR} (the scorer dies on one).\n"
        "Run as `--limit <count>` plus PADDOCK_NAVSIM_OVERRIDES naming the tokens\n"
        "(python -m tools.gate subset navsim --env PADDOCK_NAVSIM_OVERRIDES). Stored valid=false.\n"
        "Pair only with a smoke run of this same list: two-frame extended comfort compares\n"
        "each scenario with the next scored one in its log, so a token can score differently\n"
        "here and in a full run.",
        {"bench": "navsim", "unit": "log", "seed": SEED, "count": count},
        picked,
        directory,
    )


# ---------------------------------------------------------------------------
# Bench2Drive: one route per scenario family, from all 220.
# ---------------------------------------------------------------------------

#: What a route that goes straight is about: Bench2Drive's 44 scenario types in 5
#: hazard families. A route that turns is grouped by its turn instead (`b2d_group`).
B2D_HAZARDS = {
    "junction": (
        "SignalizedJunctionLeftTurn",
        "NonSignalizedJunctionLeftTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "SignalizedJunctionRightTurn",
        "NonSignalizedJunctionRightTurn",
        "VanillaSignalizedTurnEncounterGreenLight",
        "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn",
        "VanillaNonSignalizedTurnEncounterStopsign",
        "VehicleTurningRoute",
        "OppositeVehicleRunningRedLight",
        "OppositeVehicleTakingPriority",
        "BlockedIntersection",
        "T_Junction",
    ),
    "highway or high speed": (
        "HighwayCutIn",
        "HighwayExit",
        "MergerIntoSlowTraffic",
        "MergerIntoSlowTrafficV2",
        "EnterActorFlow",
        "InterurbanActorFlow",
        "InterurbanAdvancedActorFlow",
        "SequentialLaneChange",
        "YieldToEmergencyVehicle",
    ),
    "obstacle in the lane": (
        "Accident",
        "AccidentTwoWays",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "ParkedObstacle",
        "ParkedObstacleTwoWays",
        "HazardAtSideLane",
        "HazardAtSideLaneTwoWays",
        "VehicleOpensDoorTwoWays",
        "InvadingTurn",
    ),
    "pedestrian or cyclist": (
        "PedestrianCrossing",
        "DynamicObjectCrossing",
        "ParkingCrossingPedestrian",
        "CrossingBicycleFlow",
        "VehicleTurningRoutePedestrian",
    ),
    "cut-in or hard brake": (
        "ParkingCutIn",
        "StaticCutIn",
        "ParkingExit",
        "HardBreakRoute",
        "ControlLoss",
    ),
}
#: A route whose heading changes by at least this much, start to end, is a turn.
B2D_TURN_DEG = 45.0


def b2d_hazard(scenario_type: str) -> str:
    for family, types in B2D_HAZARDS.items():
        if scenario_type in types:
            return family
    raise KeyError(f"Bench2Drive scenario type {scenario_type!r} is in no hazard family")


def b2d_turn_deg(points: list[tuple[float, float]]) -> float:
    """The route's heading change, degrees, left positive.

    CARLA's world is left-handed (y to the right of x), so a heading that grows
    is a right turn: every SignalizedJunctionLeftTurn route comes out at -90 raw.
    """
    headings = [
        math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(points, points[1:]) if math.dist(a, b) > 1.0
    ]
    raw = sum(((b - a + math.pi) % (2 * math.pi)) - math.pi for a, b in zip(headings, headings[1:]))
    return -math.degrees(raw)


def b2d_group(scenario_type: str, turn_deg: float) -> str:
    if turn_deg >= B2D_TURN_DEG:
        return "left turn"
    if turn_deg <= -B2D_TURN_DEG:
        return "right turn"
    return f"straight: {b2d_hazard(scenario_type)}"


def b2d_routes(xml_path: Path) -> dict[str, tuple[str, str, float]]:
    """route id -> (scenario type, town, turn in degrees), from the leaderboard's route file."""
    out = {}
    for route in ET.parse(xml_path).getroot().findall("route"):
        types = sorted({s.get("type") for s in route.iter("scenario")})
        if len(types) != 1:
            raise SystemExit(f"route {route.get('id')} has scenario types {types}; expected one")
        points = [(float(p.get("x")), float(p.get("y"))) for p in route.find("waypoints").findall("position")]
        out[route.get("id")] = (types[0], route.get("town"), b2d_turn_deg(points))
    return out


def draw_bench2drive(xml_path: Path, n: int = 7, directory: Path = SMOKE_DIR) -> Path:
    routes = b2d_routes(xml_path)
    groups: dict[str, list[str]] = {}
    for rid, (stype, _town, turn) in routes.items():
        groups.setdefault(b2d_group(stype, turn), []).append(rid)
    picked = balanced_pick(groups, n, SEED)
    shown = "\n".join(
        f"  {group}: route {rid} ({routes[rid][0]}, {routes[rid][1]}, turns {routes[rid][2]:+.0f} deg)"
        for group, ids in sorted(picked.items())
        for rid in ids
    )
    return write_subset(
        "bench2drive",
        f"Bench2Drive smoke tier (closed-loop tier): {n} of the {len(routes)} routes of {xml_path.name},\n"
        "one per group. Groups: a route that turns by 45 degrees or more, read from its own\n"
        "waypoints, is a left or a right turn; one that goes straight is grouped by what it is\n"
        f"about, its scenario type's hazard family (tools/gate_subsets.py, B2D_HAZARDS):\n"
        f"{_counts(groups)}.\n"
        f"{shown}\n"
        "Run as `--scene <the ids>`. Stored valid=false (a route subset).\n"
        "can_ship = false: seven routes detect only a large change (17-38 driving-score\n"
        "points, from paired route differences of 13.6-30.4 sd). A REGRESSION that large still\n"
        "fails the gate; a GAIN from so few routes is too likely to be chance to ship on.",
        {"bench": "bench2drive", "can_ship": False, "unit": "route", "seed": SEED, "count": n},
        picked,
        directory,
    )


# ---------------------------------------------------------------------------
# AlpaSim: 10 of the public_2601 suite, at random.
# ---------------------------------------------------------------------------

ALPASIM_SUITE = "public_2601"
ALPASIM_SCENE_SET = "public_2601_gate10"


#: Redrawn on 2026-10-27 from the scenes on disk (see `draw_alpasim`), with a seed of its own.
ALPASIM_DATE = "2026-10-27"
ALPASIM_SEED = 20261027
#: A clip is a turn when its heading changes by this much, degrees: Bench2Drive's rule.
ALPASIM_TURN_DEG = B2D_TURN_DEG


def alpasim_drive(usdz: Path) -> tuple[float, float]:
    """A scene's recorded drive: its net heading change, degrees, left positive, and its
    mean speed, m/s. Read from `rig_trajectories.json` in the scene's USDZ (a zip): the
    car's 4x4 poses in the world over the clip, x forward, z up."""
    rig = json.loads(zipfile.ZipFile(usdz).read("rig_trajectories.json"))["rig_trajectories"][0]
    poses, stamps = rig["T_rig_worlds"], rig["T_rig_world_timestamps_us"]
    xy = [(m[0][3], m[1][3]) for m in poses]
    yaw = [math.atan2(m[1][0], m[0][0]) for m in poses]
    # Summed step by step, so a heading that crosses +-180 degrees is not read as a U-turn.
    turn = sum((b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(yaw, yaw[1:], strict=False))
    seconds = (stamps[-1] - stamps[0]) / 1e6
    return math.degrees(turn), sum(math.dist(a, b) for a, b in zip(xy, xy[1:], strict=False)) / seconds


def alpasim_group(turn_deg: float, speed_mps: float) -> str:
    """`<maneuver>/<speed band>`, the naming nuScenes' and NAVSIM's subsets use."""
    if turn_deg >= ALPASIM_TURN_DEG:
        maneuver = "left"
    elif turn_deg <= -ALPASIM_TURN_DEG:
        maneuver = "right"
    else:
        maneuver = "straight"
    return f"{maneuver}/{speed_band(speed_mps)}"


#: The scene list the AlpaSim driver profiles hand the wizard (`harness.DRIVER_PROFILES`,
#: `scenes_csv`): the wizard looks a scene up there only, and one outside it fails with
#: "Failed to find scenes" (2 of the gate's 10 on 2026-10-27, though both were on disk).
ALPASIM_CATALOG = "public_2601_30scenes_26p01.csv"

#: Scenes the wizard runs that measure nothing, with why: the run ends before any model
#: has driven, so every model scores the same. `replace_unusable_alpasim` swaps them out.
ALPASIM_UNUSABLE = {
    "clipgt-04749bb9-9b37-495b-bed0-77f0e33ac7da": "constant velocity and Drive-JEPA both leave the road within 0.5 s",
    "clipgt-791d68ce-fe38-460a-b11d-c19a68f4c8b1": "constant velocity and Drive-JEPA both leave the road at their first step",
}


def alpasim_catalog() -> set[str]:
    from paddock.adapters.benchmarks.alpasim.harness import SCENES_DIR

    with (SCENES_DIR / ALPASIM_CATALOG).open() as handle:
        return {r["scene_id"] for r in csv.DictReader(handle)}


def _alpasim_groups(
    suites_csv: Path, scene_cache: Path, only: set[str] | None = None
) -> dict[str, list[str]]:
    """The suite's scenes on disk (and in `only`, when given), grouped by `alpasim_group`."""
    with suites_csv.open() as handle:
        uuids = {
            r["scene_id"]: r["uuid"] for r in csv.DictReader(handle) if r["test_suite_id"] == ALPASIM_SUITE
        }
    groups: dict[str, list[str]] = {}
    for scene, uuid in sorted(uuids.items()):
        usdz = scene_cache / "all-usdzs" / f"{uuid}.usdz"
        if (only is None or scene in only) and usdz.is_file():
            groups.setdefault(alpasim_group(*alpasim_drive(usdz)), []).append(scene)
    return groups


def _write_alpasim(
    picked: dict[str, list[str]], header: str, pool_note: str, directory: Path, date: str, seed: int
) -> list[Path]:
    from paddock.adapters.benchmarks.alpasim.harness import SCENES_DIR

    n = sum(len(v) for v in picked.values())
    # Group by group, as the subset file lists them: the gate's tests compare the two in order.
    ids = [i for group in sorted(picked) for i in picked[group]]
    scene_set = SCENES_DIR / f"{ALPASIM_SCENE_SET}.yaml"
    scene_set.write_text(
        "# The regression gate's AlpaSim scenes. "
        + header.replace("\n", "\n# ")
        + f"\n# tools/gate_smoke/alpasim.toml lists the same {n}; tests/test_gate.py holds them equal.\n"
        f"suite: {ALPASIM_SUITE}\n"
        f"selection: {pool_note}\n"
        "scenes:\n" + "".join(f"- {s}\n" for s in ids)
    )
    return [
        scene_set,
        write_subset(
            "alpasim",
            f"AlpaSim smoke tier: {header}\n"
            f"Run as the scene set {ALPASIM_SCENE_SET} (adapters/benchmarks/alpasim/scenes/),\n"
            "which the smoke's derived config pins as [benchmarks.alpasim] scene_set, so a site\n"
            "whose own scene_set is another still runs these.",
            {"bench": "alpasim", "unit": "scene", "seed": seed, "count": n, "scene_set": ALPASIM_SCENE_SET},
            picked,
            directory,
            date=date,
            seed=seed,
        ),
    ]


def replace_unusable_alpasim(suites_csv: Path, scene_cache: Path, directory: Path = SMOKE_DIR) -> list[Path]:
    """Swap each gate scene the wizard cannot find, or that is in `ALPASIM_UNUSABLE`, and keep the rest.

    Only those are swapped, so the scenes a baseline already holds stay paired. A
    replacement is a seeded pick, among scenes on disk and in `ALPASIM_CATALOG` that the
    subset does not hold and never held, from the swapped scene's own group, else from its
    own turn. Earlier swaps are read back from the subset's header, so it lists every one.
    """
    catalog = alpasim_catalog()
    off_list = f"not in {ALPASIM_CATALOG}, so the wizard cannot find it"
    runnable = _alpasim_groups(suites_csv, scene_cache, only=catalog)
    group_of = {scene: g for g, ids in runnable.items() for scene in ids}
    subset = load_subset("alpasim", directory)
    picked = {g: list(ids) for g, ids in subset.strata.items()}
    # A swap line of an earlier run, `  <old> (<group>) -> <new> (<group>)[: <why>]`; the
    # first run (2026-10-27) wrote no why, and its scenes were the ones off the list.
    earlier = re.findall(
        r"^#   ((\S+) \(\S+\) -> \S+ \(\S+\))(?:: (.+))?$", (directory / "alpasim.toml").read_text(), re.M
    )
    swaps = [
        f"{line}: {why or (off_list if old not in catalog else 'no reason recorded')}"
        for line, old, why in earlier
    ]
    held = set(subset.ids) | {old for _, old, _ in earlier} | set(ALPASIM_UNUSABLE)
    for group in sorted(subset.strata):
        for scene in subset.strata[group]:
            if scene in catalog and scene not in ALPASIM_UNUSABLE:
                continue
            same = [c for c in runnable.get(group, []) if c not in held]
            turn = group.split("/")[0]
            near = [c for g, ids in runnable.items() if g.split("/")[0] == turn for c in ids if c not in held]
            if not (same or near):
                raise SystemExit(f"no runnable {turn} scene left to replace {scene} ({group})")
            new = random.Random(f"{ALPASIM_SEED}:replace:{scene}").choice(sorted(same or near))
            picked[group].remove(scene)
            picked.setdefault(group_of[new], []).append(new)
            held.add(new)
            swaps.append(
                f"{scene} ({group}) -> {new} ({group_of[new]}): {ALPASIM_UNUSABLE.get(scene, off_list)}"
            )
    picked = {g: sorted(ids) for g, ids in picked.items() if ids}
    header = (
        f"the 10 drawn on {ALPASIM_DATE} (seed {ALPASIM_SEED}) from the 50 public_2601 scenes\n"
        "downloaded on shakaal, by turn and speed, with each scene that could not be used\n"
        "swapped for a seeded pick from its own group, else its own turn, among the scenes on\n"
        f"disk and in {ALPASIM_CATALOG} (the scene list the driver profiles hand the wizard):\n"
        + "".join(f"  {line}\n" for line in swaps)
        + "The pool is what had been downloaded, not a random sample of the suite; the list's\n"
        "30 scenes are the ones behind the published 30-scene AlpaSim results."
    )
    return _write_alpasim(
        picked,
        header,
        f"10 by turn and speed, seed {ALPASIM_SEED}; {len(swaps)} swapped for usable scenes",
        directory,
        ALPASIM_DATE,
        ALPASIM_SEED,
    )


def draw_alpasim(suites_csv: Path, scene_cache: Path, n: int = 10, directory: Path = SMOKE_DIR) -> list[Path]:
    """The gate's AlpaSim scenes, drawn from the suite's scenes already in `scene_cache`.

    The first draw (2026-10-22) took 10 of all 913 scenes at random, and none of
    them was downloaded: AlpaSim fetches a scene from a gated Hugging Face dataset
    at run time, which the gate queue's account has no token for, onto a disk that
    was 99% full. The scenes on disk are also the only ones whose recorded drive can
    be read, so this draw is grouped by turn and speed like every other subset.
    """
    from paddock.adapters.benchmarks.alpasim.harness import SCENES_DIR

    with suites_csv.open() as handle:
        uuids = {
            r["scene_id"]: r["uuid"] for r in csv.DictReader(handle) if r["test_suite_id"] == ALPASIM_SUITE
        }
    # Only scenes the wizard can run (`ALPASIM_CATALOG`) as well as on disk.
    groups = _alpasim_groups(suites_csv, scene_cache, only=alpasim_catalog())
    pool = sum(len(v) for v in groups.values())
    picked = balanced_pick(groups, n, ALPASIM_SEED)
    # Group by group, as the subset file lists them: the gate's tests compare the two in order.
    ids = [i for group in sorted(picked) for i in picked[group]]
    where = (
        f"{n} of the {pool} scenes of {ALPASIM_SUITE} downloaded on shakaal (not of all {len(uuids)}),\n"
        f"grouped by turn (left or right from {ALPASIM_TURN_DEG:g} degrees of heading change over the clip)\n"
        f"and speed band ({_counts(groups)}), every group the same share."
    )
    caveat = (
        "The pool is what had been downloaded, not a random sample of the suite: the first\n"
        "ten of its list, twenty drawn for public_2601_30scenes, and twenty more. Thirty of\n"
        "them are the scenes behind the published 30-scene AlpaSim results."
    )
    scene_set = SCENES_DIR / f"{ALPASIM_SCENE_SET}.yaml"
    scene_set.write_text(
        f"# The regression gate's AlpaSim scenes, drawn by `python -m tools.gate_subsets alpasim`\n"
        f"# on {ALPASIM_DATE} (seed {ALPASIM_SEED}): "
        + where.replace("\n", "\n# ")
        + "\n# "
        + caveat.replace("\n", "\n# ")
        + f"\n# tools/gate_smoke/alpasim.toml lists the same {n}; tests/test_gate.py holds them equal.\n"
        f"suite: {ALPASIM_SUITE}\n"
        f"selection: {n} of the {pool} downloaded, by turn and speed, seed {ALPASIM_SEED}, by scene_id\n"
        "scenes:\n" + "".join(f"- {s}\n" for s in ids)
    )
    return [
        scene_set,
        write_subset(
            "alpasim",
            f"AlpaSim smoke tier: {where}\n{caveat}\n"
            f"Run as the scene set {ALPASIM_SCENE_SET} (adapters/benchmarks/alpasim/scenes/),\n"
            "which the smoke's derived config pins as [benchmarks.alpasim] scene_set, so a site\n"
            "whose own scene_set is another still runs these.\n"
            "Replaces the draw of 2026-10-22 (10 of all 913 at random, none of them downloaded).",
            {
                "bench": "alpasim",
                "unit": "scene",
                "seed": ALPASIM_SEED,
                "count": n,
                "scene_set": ALPASIM_SCENE_SET,
            },
            picked,
            directory,
            date=ALPASIM_DATE,
            seed=ALPASIM_SEED,
        ),
    ]


# ---------------------------------------------------------------------------
# KITScenes: all 223, deciding on the 186 val keyframes.
# ---------------------------------------------------------------------------


def draw_kitscenes(samples_path: Path, directory: Path = SMOKE_DIR) -> Path:
    groups: dict[str, list[str]] = {}
    for line in samples_path.read_text().splitlines():
        if line:
            s = json.loads(line)
            groups.setdefault(s["split"], []).append(s["sample_id"])
    groups = {g: sorted(ids) for g, ids in groups.items()}
    total = sum(len(v) for v in groups.values())
    return write_subset(
        "kitscenes",
        f"KITScenes smoke tier: the whole benchmark, all {total} keyframes of the shared sample\n"
        f"list, run as the default set. Grouped by KITScenes' own split ({_counts(groups)}):\n"
        "the verdict reads only `val`. The overlap_train_val keyframes are places that are\n"
        "also in KITScenes' training data, so a checkpoint trained on it may have seen them;\n"
        "they still run, and `check` reports them alone in a note.\n"
        f"Read from the full run {samples_path.parent.name}. Nothing is drawn: there is nothing to\n"
        "leave out.",
        {
            "bench": "kitscenes",
            "unit": "scene",
            "count": total,
            "decide_on": ["val"],
            "left_out_reason": "overlap_train_val: places in the training data",
        },
        groups,
        directory,
    )


# ---------------------------------------------------------------------------
# Quick-tier crash checks: one Bench2Drive route and one AlpaSim scene.
# ---------------------------------------------------------------------------

#: Picked on 2026-10-27 (the user's idea that day), from the closed tier's own picks.
QUICK_CLOSED_DATE = "2026-10-27"
QUICK_CLOSED_SEED = 20261027
CRASH_CHECK = (
    "One is too few for a verdict: the gate says TOO FEW UNITS and shows the score. What it\n"
    "catches is a checkpoint that cannot drive there at all: a run that fails is INCOMPLETE,\n"
    "which stops the gate before the closed tier's hours. can_ship = false."
)


def b2d_route_length_m(xml_path: Path) -> dict[str, float]:
    """route id -> length of its waypoint path, metres."""
    out = {}
    for route in ET.parse(xml_path).getroot().findall("route"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in route.find("waypoints").findall("position")]
        out[route.get("id")] = sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))
    return out


def draw_quick_closed(xml_path: Path, directory: Path = SMOKE_DIR) -> list[Path]:
    """`bench2drive_quick` and `alpasim_quick`, each one sample of its closed-tier subset.

    Bench2Drive's is the shortest route of `bench2drive.toml` by path length, so the
    quick tier stays quick; the choice reads the route file, not any model's result.
    AlpaSim's clips are all 20 s, so its scene is drawn at random from `alpasim.toml`.
    Both come from the closed tier's subsets, so the closed tier's baselines hold them.
    """
    from tools.gate import load_subset

    b2d = load_subset("bench2drive", directory)
    lengths = b2d_route_length_m(xml_path)
    route = min(b2d.ids, key=lambda rid: (lengths[rid], rid))
    route_group = next(g for g, ids in b2d.strata.items() if route in ids)
    alpasim = load_subset("alpasim", directory)
    scene = random.Random(f"{QUICK_CLOSED_SEED}:alpasim_quick").choice(sorted(alpasim.ids))
    scene_group = next(g for g, ids in alpasim.strata.items() if scene in ids)
    meta = {"can_ship": False, "crash_check": True, "seed": QUICK_CLOSED_SEED, "count": 1}
    return [
        write_subset(
            "bench2drive_quick",
            "Bench2Drive quick tier: one route, a crash check. The shortest by path length\n"
            f"({lengths[route]:.0f} m) of the closed tier's {len(b2d.ids)} routes (bench2drive.toml), so the\n"
            "quick tier stays short and the closed tier's baseline holds it.\n" + CRASH_CHECK,
            {"bench": "bench2drive", "unit": "route", **meta},
            {route_group: [route]},
            directory,
            date=QUICK_CLOSED_DATE,
            seed=QUICK_CLOSED_SEED,
            command="quick_closed",
        ),
        write_subset(
            "alpasim_quick",
            "AlpaSim quick tier: one scene, a crash check. Drawn at random from the closed tier's\n"
            f"{len(alpasim.ids)} scenes (alpasim.toml), so the closed tier's baseline holds it. Run as\n"
            f"`--scene <it>` within the scene set {alpasim.meta['scene_set']}.\n" + CRASH_CHECK,
            {"bench": "alpasim", "unit": "scene", **meta, "scene_set": alpasim.meta["scene_set"]},
            {scene_group: [scene]},
            directory,
            date=QUICK_CLOSED_DATE,
            seed=QUICK_CLOSED_SEED,
            command="quick_closed",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="bench", required=True)
    sub.add_parser("neuroncap")
    p = sub.add_parser("nuscenes")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--nuscenes", type=Path, required=True)
    p = sub.add_parser("navsim")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p = sub.add_parser("bench2drive")
    p.add_argument("--routes", type=Path, required=True)
    p = sub.add_parser("alpasim")
    p.add_argument("--suites", type=Path, required=True)
    p.add_argument("--scene-cache", type=Path, required=True, help="[benchmarks.alpasim] scene_cache")
    p = sub.add_parser("kitscenes")
    p.add_argument("--samples", type=Path, required=True)
    p = sub.add_parser("alpasim_replace")
    p.add_argument("--suites", type=Path, required=True)
    p.add_argument("--scene-cache", type=Path, required=True, help="[benchmarks.alpasim] scene_cache")
    p = sub.add_parser("quick_closed")
    p.add_argument("--routes", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.bench == "neuroncap":
        written = draw_neuroncap()
    elif args.bench == "nuscenes":
        written = [draw_nuscenes(args.samples, args.nuscenes)]
    elif args.bench == "navsim":
        written = [draw_navsim(args.samples, args.logs)]
    elif args.bench == "bench2drive":
        written = [draw_bench2drive(args.routes)]
    elif args.bench == "alpasim":
        written = draw_alpasim(args.suites, args.scene_cache)
    elif args.bench == "alpasim_replace":
        written = replace_unusable_alpasim(args.suites, args.scene_cache)
    elif args.bench == "quick_closed":
        written = draw_quick_closed(args.routes)
    else:
        written = [draw_kitscenes(args.samples)]
    for path in written:
        print(path.relative_to(_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
