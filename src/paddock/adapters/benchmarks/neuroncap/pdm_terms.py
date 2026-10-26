"""NAVSIM's PDM terms for one neuroNCAP run, from the files the engine writes beside it.

Every run directory holds everything `paddock.metrics.pdm.score_plan` needs but
the map:

* `trajectories.json`: the plan the model made at every 10 Hz step, six (x, y)
  waypoints at 0.5 s in the ego frame (x forward, y left, from the rear axle);
* `ego_poses.json`: the ego's world pose at the same steps, at the rear axle;
* `actors.json`: every actor's logged poses (box centre, about 2 Hz) and size
  (nuScenes `[width, length, height]`), the inserted one included, each named by
  its nuScenes instance token;
* `reference_trajectory.json`: the human's logged path.

The map comes from `tools/benchmarks/nuscenes/build_pdm_geometry.py`, which
rasterises nuScenes' drivable area once per scene, and each actor's category from
nuScenes' own `instance.json` and `category.json`.

**Per decision**, the plan is scored against the actors' real motion over its
3 s, on the drivable raster, with the human's logged path as the centreline and
the human's own motion over the same 3 s as EP's reference (counted only where
it does not crash or leave the road: the human never saw the inserted actor).
**A run's term is the mean over its decisions**, and its `pdm_style_score` the
mean of the decisions' composites, so a run that plans into the actor for its
last second scores lower than one that plans into it for three.

`collides_like_neuroncap` is the engine's own collision check, restated on the
realised path. It is how the geometry here is held to the benchmark's: on stored
runs it must reproduce the engine's `any_collide` verdict
(`tools/benchmarks/neuroncap/check_pdm_geometry.py`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from paddock.metrics import pdm

#: The plan's waypoint spacing (`harness.WAYPOINT_DT`, restated to keep this
#: importable without the harness).
WAYPOINT_DT = 0.5
#: nuScenes categories that are not agents: NAVSIM scores a collision with one 0.5.
STATIC_PREFIXES = ("movable_object.", "static_object.")
#: The classes neuroNCAP's collision check considers (`COLLISION_CLASSES_TO_CHECK`
#: in `neuro_ncap/evaluation/collision.py`), matched as it matches them: the
#: second dotted part of the nuScenes category name.
NEURONCAP_COLLISION_CLASSES = frozenset(
    {"car", "bus", "construction_vehicle", "bicycle", "motorcycle", "truck", "trailer"}
)


def load_categories(nuscenes_root: Path, version: str = "v1.0-trainval") -> dict[str, str]:
    """Every nuScenes instance token's category name, from the dataset's own tables."""
    tables = Path(nuscenes_root) / version
    names = {c["token"]: c["name"] for c in json.loads((tables / "category.json").read_text())}
    return {
        i["token"]: names[i["category_token"]] for i in json.loads((tables / "instance.json").read_text())
    }


def _yaw(pose: np.ndarray) -> float:
    return math.atan2(pose[1, 0], pose[0, 0])


def _measured(stamps: np.ndarray, poses: np.ndarray, i: int) -> tuple[float | None, float, float]:
    """The car's speed, acceleration and yaw rate at step `i`, from its logged poses.

    Each over the step that ended at it (the first decision: the step that starts
    there). The state NAVSIM's simulator starts a plan from; a benchmark that logs
    only poses gives it by differences. Acceleration needs two steps, else 0.
    """

    def step(a: int, b: int) -> tuple[float, float, float] | None:
        gap = (stamps[b] - stamps[a]) / 1e6
        if gap <= 0:
            return None
        speed = float(np.linalg.norm(poses[b][:2, 3] - poses[a][:2, 3]) / gap)
        turn = math.atan2(
            math.sin(_yaw(poses[b]) - _yaw(poses[a])), math.cos(_yaw(poses[b]) - _yaw(poses[a]))
        )
        return speed, turn / gap, gap

    last = step(i - 1, i) if i else step(0, min(1, len(stamps) - 1))
    if last is None:
        return None, 0.0, 0.0
    before = step(i - 2, i - 1) if i >= 2 else None
    accel = (last[0] - before[0]) / last[2] if before is not None else 0.0
    return last[0], accel, last[1]


def tracks_from_actors(actors: list[dict], categories: dict[str, str]) -> list[pdm.Track]:
    """`actors.json` as `pdm.Track`s. An actor whose category is unknown counts as an agent."""
    tracks = []
    for actor in actors:
        poses = np.asarray(actor["poses"], dtype=float)
        category = categories.get(str(actor.get("uuid")), "")
        tracks.append(
            pdm.Track(
                times_us=np.asarray(actor["timestamps"], dtype=float),
                xy=poses[:, :2, 3],
                yaw=np.array([_yaw(p) for p in poses]),
                length=float(actor["dims"][1]),
                width=float(actor["dims"][0]),
                agent=not category.startswith(STATIC_PREFIXES),
                name=category or str(actor.get("uuid")),
            )
        )
    return tracks


def _poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text())
    stamps = np.array(sorted(int(k) for k in raw), dtype=np.int64)
    return stamps, np.asarray([raw[str(s)] for s in stamps], dtype=float)


def run_terms(run_dir: Path, area: pdm.DrivableArea | None, categories: dict[str, str]) -> dict[str, float]:
    """The mean of each PDM term over a run's decisions, or {} without its files or a map.

    No map means no terms at all, not only no DAC: without the drivable area a
    sideways collision is never at fault, and TTC and EP's best safe path change
    with it, so the other terms would be measured under looser rules.
    """
    files = {
        n: run_dir / f"{n}.json" for n in ("trajectories", "ego_poses", "actors", "reference_trajectory")
    }
    if area is None or not all(p.is_file() for p in files.values()):
        return {}
    plans = json.loads(files["trajectories"].read_text())
    stamps, poses = _poses(files["ego_poses"])
    tracks = tracks_from_actors(json.loads(files["actors"].read_text()), categories)
    ref_raw = json.loads(files["reference_trajectory"].read_text())
    ref_t = np.array(sorted(int(k) for k in ref_raw), dtype=float)
    ref_xy = np.asarray([ref_raw[str(int(t))][:2] for t in ref_t], dtype=float)
    horizon = None
    per_decision: dict[str, list[float]] = {k: [] for k in pdm.PDM_KEYS}
    for i, stamp in enumerate(stamps):
        plan = plans.get(str(int(stamp)))
        if plan is None or len(plan) < 2:
            continue
        plan = np.asarray(plan, dtype=float)
        horizon = len(plan) * WAYPOINT_DT
        speed0, accel0, yaw_rate0 = _measured(stamps, poses, i)
        human = pdm.world_states((ref_t - stamp) / 1e6, ref_xy, horizon)
        terms = pdm.score_plan(
            plan,
            WAYPOINT_DT,
            poses[i],
            float(stamp),
            tracks,
            area=area,
            centreline=ref_xy,
            references=[human],
            speed0=speed0,
            accel0=accel0,
            yaw_rate0=yaw_rate0,
        )
        for key, value in terms.items():
            per_decision[key].append(value)
    if horizon is None:
        return {}
    out = {}
    for key, values in per_decision.items():
        finite = [v for v in values if math.isfinite(v)]
        out[key] = float(np.mean(finite)) if finite and len(finite) == len(values) else float("nan")
    return out


def _checked(category: str) -> bool:
    """Whether the engine's collision check considers this category: `split(".")[1]`, as it does.

    That rule makes `vehicle.bus.rigid` a "bus" and leaves `vehicle.construction`
    as "construction", which its list does not name, so the engine never checks it.
    """
    parts = category.split(".")
    return len(parts) > 1 and parts[1] in NEURONCAP_COLLISION_CLASSES


def collides_like_neuroncap(run_dir: Path, categories: dict[str, str]) -> bool | None:
    """The engine's own collision check, restated on the realised path; None without the files.

    Ego box 4.018 x 1.85 m, its centre 0.5 m ahead of the rear axle; at every step
    of `ego_poses.json`, against each actor of a checked class active then,
    interpolated to that time. Touching counts, as `intersects` does there.
    """
    ego_file, actor_file = run_dir / "ego_poses.json", run_dir / "actors.json"
    if not (ego_file.is_file() and actor_file.is_file()):
        return None
    stamps, poses = _poses(ego_file)
    actors = [
        a for a in json.loads(actor_file.read_text()) if _checked(categories.get(str(a.get("uuid")), ""))
    ]
    tracks = tracks_from_actors(actors, categories)
    if not tracks:
        return False
    yaw = np.array([_yaw(p) for p in poses])
    states = pdm.States(
        taus=(stamps - stamps[0]) / 1e6, xy=poses[:, :2, 3], yaw=yaw, speed=np.zeros(len(stamps))
    )
    boxes = pdm.ego_boxes(states, pdm.NUSCENES_EGO)
    for track in tracks:
        xy, tyaw, _, active = track.at(stamps.astype(float))
        tboxes = pdm.box_corners(xy, tyaw, track.length, track.width)
        if (pdm.convex_overlap(boxes, tboxes) & active).any():
            return True
    return False
