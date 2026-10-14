"""Shared nuScenes scene-description keyword tagging.

Consolidates the keyword-tagging logic that used to be independently
duplicated in worst_cases.py, turn_overlap.py, and slice_by_scenario.py, so
"bad category" analysis stays consistent across tools instead of drifting.

The word lists below are checked against the real 850 nuScenes scene
descriptions (v1.0-trainval scene.json), not guessed -- see the frequency
counts in each comment. That process caught two real bugs worth noting for
anyone adding a new keyword here:
  - bare "ped" as a substring matches inside "stopped" (37 hits) and
    "flipped" (3 hits), not just "peds"/"ped" -- needs a whole-word check.
  - bare "bus" as a substring matches inside "busy" (17 hits) and "bushes"
    (2 hits) -- same problem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_scene_descriptions(nuscenes_root: Path, version: str) -> dict[str, str]:
    scene_json = nuscenes_root / version / "scene.json"
    if not scene_json.exists():
        raise SystemExit(f"scene.json not found at {scene_json}")
    scenes = json.loads(scene_json.read_text())
    return {s["name"]: s.get("description", "") for s in scenes}


def cmd_int(value) -> int:
    """Coerce a command field (int, numeric string, float) to 0-3, default STRAIGHT=2."""
    try:
        return int(float(value))
    except Exception:
        return 2


def is_turn(cmd: int) -> bool:
    return cmd in (0, 1)


# Command enum order from policy/protocol.py: RIGHT=0, LEFT=1, STRAIGHT=2, FOLLOW_REFERENCE=3.
COMMAND_NAMES = {0: "RIGHT", 1: "LEFT", 2: "STRAIGHT", 3: "FOLLOW_REFERENCE"}


def command_name(value) -> str:
    """The raw nav instruction given to the model, by name -- lets you tell
    "it turned left because the route said LEFT" apart from "it turned left
    on its own with a STRAIGHT/FOLLOW_REFERENCE input"."""
    return COMMAND_NAMES.get(cmd_int(value), "STRAIGHT")


def tag_maneuver(desc: str, cmd: int) -> str:
    d = desc.lower()
    if any(w in d for w in ("switch lane", "change lane", "lane change", "merge left", "merge right")):
        return "lane_change"
    if any(w in d for w in ("standstill", "stationary")) or (
        cmd in (2, 3) and any(w in d for w in ("wait", "red light"))
    ):
        return "stop"
    if any(w in d for w in ("pulling out", "start from", "begin moving", "accelerat")):
        return "start"
    if cmd == 0:
        return "turn_right"
    if cmd == 1:
        return "turn_left"
    return "straight"


def tag_command_maneuver(cmd) -> str:
    """Coarse maneuver from the nav command alone (no scene text).

    Accepts either the numeric Command value or its enum name
    ("RIGHT"/"LEFT"), since callers have historically passed either.
    """
    c = str(cmd).strip()
    if c in ("0", "RIGHT"):
        return "turn_right"
    if c in ("1", "LEFT"):
        return "turn_left"
    return "straight"


def mentions(text: str, words: tuple) -> bool:
    d = text.lower()
    return any(w in d for w in words)


def _word(text: str, word: str) -> bool:
    """Whole-word match, for short/ambiguous tokens where a plain substring
    check would false-positive (see module docstring: "ped" inside
    "stopped", "bus" inside "busy")."""
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None


INTERSECTION_WORDS = ("intersection", "junction", "crossroads")  # 110+82+51+11+10 hits across phrasings

# "pedestrian"/"pedestrians" (33+11 hits) as substrings; "ped"/"peds" (164+424
# hits -- the dominant real form) need a whole-word check, see PEDESTRIAN_WHOLE_WORDS.
PEDESTRIAN_WORDS = ("pedestrian", "jaywalker", "crosswalk", "walkers", "people crossing", "crossing")
PEDESTRIAN_WHOLE_WORDS = ("ped", "peds")

# "truck"(172)/"scooter"(70)/"motorcycle"(65) are safe substrings; "bus"(196)
# needs a whole-word check (collides with "busy"/"bushes").
VEHICLE_WORDS = (
    "following",
    "lead vehicle",
    "cyclist",
    "bicycle",
    "moving vehicle",
    "following traffic",
    "truck",
    "scooter",
    "motorcycle",
)
VEHICLE_WHOLE_WORDS = ("bus", "buses")

# 148 "construction" + 91 "cone(s)" + 47 "barrier(s)" hits -- a bigger cluster
# than several of the categories below, worth its own tag.
CONSTRUCTION_WORDS = ("construction", "cone", "barrier")

# "stop sign": 0 hits in this dataset's descriptions -- not worth a tag here.
# "traffic light": 8, "red light": 21 -- rare but real; "green light"/"stop
# light": 0, dropped.
TRAFFIC_LIGHT_WORDS = ("traffic light", "red light")


def tag_intersection(desc: str) -> str:
    return "intersection" if mentions(desc, INTERSECTION_WORDS) else "non-intersection"


def tag_traffic_density(desc: str) -> str:
    d = desc.lower()
    if any(
        w in d
        for w in (
            "busy",
            "heavy traffic",
            "congested",
            "congestion",
            "crowded",
            "dense traffic",
            "many vehicles",
        )
    ):
        return "high"
    if any(
        w in d
        for w in (
            "empty",
            "light traffic",
            "low traffic",
            "sparse",
            "quiet",
            "few vehicles",
            "no traffic",
            "no other vehicle",
        )
    ):
        return "low"
    return "medium"


def tag_road_type(desc: str) -> str:
    d = desc.lower()
    if any(w in d for w in ("highway", "motorway", "freeway", "expressway")):
        return "highway"
    if any(w in d for w in ("parking", "car park", "garage")):
        return "parking"
    if any(w in d for w in ("residential", "suburb", "neighborhood")):
        return "residential"
    return "urban"


def tag_pedestrians(desc: str) -> str:
    if mentions(desc, PEDESTRIAN_WORDS) or any(_word(desc, w) for w in PEDESTRIAN_WHOLE_WORDS):
        return "pedestrians"
    return "no_pedestrians"


def tag_moving_vehicles(desc: str) -> str:
    if mentions(desc, VEHICLE_WORDS) or any(_word(desc, w) for w in VEHICLE_WHOLE_WORDS):
        return "moving_vehicles"
    return "no_moving_vehicles"


def tag_construction(desc: str) -> str:
    return "construction_zone" if mentions(desc, CONSTRUCTION_WORDS) else "no_construction_zone"


def tag_traffic_light(desc: str) -> str:
    return "traffic_light" if mentions(desc, TRAFFIC_LIGHT_WORDS) else "no_traffic_light"


def tag_weather(desc: str) -> str:
    d = desc.lower()
    return "rain" if any(w in d for w in ("rain", "wet", "drizzl")) else "clear"


def tag_lighting(desc: str) -> str:
    d = desc.lower()
    return "night" if any(w in d for w in ("night", "dark", "dusk", "dawn")) else "day"


def tag_scenario_bucket(desc: str) -> str:
    """Fine-grained single-label scenario bucket. Priority order matters — more
    specific scenarios are checked first."""
    d = desc.lower()
    if any(w in d for w in ("red light", "traffic light", "wait at intersection", "wait for")):
        return "traffic_light_wait"
    if "roundabout" in d:
        return "roundabout"
    if any(w in d for w in ("construction", "worker", "traffic cone", "cone", "barrier", "barriers")):
        return "construction"
    if any(w in d for w in ("u-turn", "u turn")):
        return "u_turn"
    if any(w in d for w in ("turn left", "merge left", "switch lane")):
        return "lane_change_or_turn_left"
    if "turn right" in d:
        return "turn_right"
    if any(w in d for w in ("cross intersection", "cross busy", "cross bridge", "cross round")):
        return "intersection_crossing"
    if mentions(d, PEDESTRIAN_WORDS) or any(_word(d, w) for w in PEDESTRIAN_WHOLE_WORDS):
        return "pedestrians"
    if any(w in d for w in ("cyclist", "bicycle", "bicyclist", "bike")):
        return "cyclists"
    if any(_word(d, w) for w in VEHICLE_WHOLE_WORDS):
        return "bus_interaction"
    if any(w in d for w in ("parking", "parked", "park")):
        return "parking_zone"
    if any(w in d for w in ("highway", "motorway", "freeway", "expressway")):
        return "highway"
    return "urban_follow"


def tag_scene(desc: str, cmd: int) -> dict:
    """Core scene tags: maneuver, intersection, traffic_density, road_type."""
    return dict(
        maneuver=tag_maneuver(desc, cmd),
        intersection=tag_intersection(desc),
        traffic_density=tag_traffic_density(desc),
        road_type=tag_road_type(desc),
    )


def tag_categories(desc: str, cmd: int) -> dict:
    """Core scene tags plus dynamic-agent, construction, and traffic-light tags."""
    tags = tag_scene(desc, cmd)
    tags["dynamic_agents_ped"] = tag_pedestrians(desc)
    tags["dynamic_agents_veh"] = tag_moving_vehicles(desc)
    tags["construction"] = tag_construction(desc)
    tags["traffic_light"] = tag_traffic_light(desc)
    return tags
