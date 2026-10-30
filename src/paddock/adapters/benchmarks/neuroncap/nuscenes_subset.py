"""A nuScenes dataroot cut down to neuroNCAP's own scenes, for the engine and the renderer.

Both build a full `NuScenes('v1.0-trainval')` to read one scene: the engine in
`neuro_ncap/components/nuscenes_api.py`, the renderer in NeuRAD's nuScenes
dataparser. That loads 2.5 GB of JSON tables (sample_data alone is 1.35 GB) from
wherever the dataset lives, at every episode's engine start and every renderer
start. Engine starts measured 4.0-6.6 min on 2026-10-23, with the tables on the
NAS, against 32-53 s on a quiet day.

The cut-down copy holds the same records for the scenes it keeps, and nothing
else: every scene's samples, all their sample_data (keyframes and sweeps), the
ego poses and calibrations those use, their annotations and instances, their
logs, the map records naming those logs, and the small tables (category,
attribute, visibility, sensor) whole. Everything else at the dataroot (samples/,
sweeps/, maps/, can_bus/) is a link to the original, so file paths resolve to the
same files. `dangling` checks that no kept record points at one left out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Small tables every record may name; copied whole.
WHOLE = ("category", "attribute", "visibility", "sensor")
#: The file `build` writes beside the tables, saying what it holds and where from.
MANIFEST = "PADDOCK_SUBSET.json"

Tables = dict[str, list[dict[str, Any]]]


def subset_tables(tables: Tables, scene_names: set[str]) -> Tables:
    """The records of `tables` that `scene_names` use, table by table."""
    scenes = [s for s in tables["scene"] if s["name"] in scene_names]
    if missing := scene_names - {s["name"] for s in scenes}:
        raise ValueError(f"not in this nuScenes: {', '.join(sorted(missing))}")
    scene_tokens = {s["token"] for s in scenes}
    log_tokens = {s["log_token"] for s in scenes}
    samples = [r for r in tables["sample"] if r["scene_token"] in scene_tokens]
    sample_tokens = {r["token"] for r in samples}
    sample_data = [r for r in tables["sample_data"] if r["sample_token"] in sample_tokens]
    ego = {r["ego_pose_token"] for r in sample_data}
    calib = {r["calibrated_sensor_token"] for r in sample_data}
    anns = [r for r in tables["sample_annotation"] if r["sample_token"] in sample_tokens]
    instances = {r["instance_token"] for r in anns}
    maps = []
    for m in tables["map"]:
        kept = [t for t in m["log_tokens"] if t in log_tokens]
        if kept:
            maps.append({**m, "log_tokens": kept})
    out: Tables = {
        "scene": scenes,
        "log": [r for r in tables["log"] if r["token"] in log_tokens],
        "map": maps,
        "sample": samples,
        "sample_data": sample_data,
        "ego_pose": [r for r in tables["ego_pose"] if r["token"] in ego],
        "calibrated_sensor": [r for r in tables["calibrated_sensor"] if r["token"] in calib],
        "sample_annotation": anns,
        "instance": [r for r in tables["instance"] if r["token"] in instances],
    }
    out.update({name: tables[name] for name in WHOLE})
    return out


def dangling(tables: Tables) -> list[str]:
    """Every reference in `tables` to a record that is not there; [] for a whole dataroot."""
    tokens = {name: {r["token"] for r in rows if "token" in r} for name, rows in tables.items()}
    links = {
        "scene": [("log_token", "log"), ("first_sample_token", "sample"), ("last_sample_token", "sample")],
        "sample": [("scene_token", "scene"), ("prev", "sample"), ("next", "sample")],
        "sample_data": [
            ("sample_token", "sample"),
            ("ego_pose_token", "ego_pose"),
            ("calibrated_sensor_token", "calibrated_sensor"),
            ("prev", "sample_data"),
            ("next", "sample_data"),
        ],
        "calibrated_sensor": [("sensor_token", "sensor")],
        "sample_annotation": [
            ("sample_token", "sample"),
            ("instance_token", "instance"),
            ("visibility_token", "visibility"),
            ("prev", "sample_annotation"),
            ("next", "sample_annotation"),
        ],
        "instance": [
            ("category_token", "category"),
            ("first_annotation_token", "sample_annotation"),
            ("last_annotation_token", "sample_annotation"),
        ],
    }
    out = []
    for table, fields in links.items():
        for row in tables.get(table, []):
            for field, target in fields:
                token = row.get(field)
                if token and token not in tokens[target]:
                    out.append(f"{table} {row['token']}: {field} {token} not in {target}")
        if table == "sample_annotation":
            for row in tables.get(table, []):
                out += [
                    f"sample_annotation {row['token']}: attribute {a} not in attribute"
                    for a in row.get("attribute_tokens", [])
                    if a not in tokens["attribute"]
                ]
    for m in tables.get("map", []):
        out += [f"map {m['token']}: log {t} not in log" for t in m["log_tokens"] if t not in tokens["log"]]
    return out


def build(source: Path, out: Path, version: str, scene_names: set[str]) -> dict[str, Any]:
    """Write the cut-down dataroot at `out` and return its manifest."""
    meta = source / version
    names = [p.stem for p in sorted(meta.glob("*.json"))]
    tables = {name: json.loads((meta / f"{name}.json").read_text()) for name in names}
    kept = subset_tables(tables, scene_names)
    if problems := dangling(kept):
        raise ValueError(f"{len(problems)} dangling references, e.g. {problems[0]}")
    (out / version).mkdir(parents=True, exist_ok=True)
    for name, rows in kept.items():
        part = out / version / f".{name}.json.part"
        part.write_text(json.dumps(rows))
        os.replace(part, out / version / f"{name}.json")
    for entry in sorted(source.iterdir()):
        link = out / entry.name
        if entry.name == version or entry.name.startswith(".") or link.exists() or link.is_symlink():
            continue
        link.symlink_to(entry)
    manifest = {
        "source": str(source),
        "version": version,
        "scenes": sorted(scene_names),
        "rows": {name: len(rows) for name, rows in sorted(kept.items())},
        "source_tables": {
            name: {
                "bytes": (meta / f"{name}.json").stat().st_size,
                "mtime": (meta / f"{name}.json").stat().st_mtime,
            }
            for name in names
        },
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def scenes_in(root: Path, version: str) -> set[str] | None:
    """The scene names a cut-down dataroot holds, or None if `root` has no tables."""
    path = root / version / "scene.json"
    if not path.is_file():
        return None
    return {s["name"] for s in json.loads(path.read_text())}
