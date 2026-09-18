"""Score trajectory candidates against a KITScenes scene's Lanelet2 HD map.

NAVSIM's PDM score has map terms (drivable area, driving direction, lane keeping)
and agent terms (collisions, time to collision). KITScenes ships the map half --
a Lanelet2 `maps/map.osm` per scene -- and no 3D boxes, so only the map half can
be rebuilt here. This is a diagnostic, not a scorer: it asks whether a model's
scorer, trained on NAVSIM, ranks candidates in a way the road itself explains.

    python tools/benchmarks/kitscenes/map_terms.py \\
        --candidates <candidates.jsonl> --output <map_terms.jsonl>

Runs in an environment with the `kitscenes` package and its Lanelet2 wheel
(`kitscenes[map]`) plus shapely. `--candidates` rows need scene_id, split,
keyframe_index, tier, proposals and human_poses, which
`tools/models/drivejepa/candidates.py` writes.

Per candidate, over all of its poses (proxies for NAVSIM's terms, not the terms):

- `outside_m`: how far the farthest corner of a 4.9 x 2.0 m box, at any pose,
  lies outside every lanelet and area a vehicle may use (German traffic rules, as
  the dataset API builds them); 0 when the box always fits. NAVSIM's drivable-area
  compliance is the pass/fail version. It is kept continuous here because lanelet
  polygons do not cover the outside of a tight turn, where the recorded drive
  itself pokes out: pick the tolerance from the human's distribution, not a guess.
- `in_lane`: fraction of pose centres inside a vehicle lanelet.
- `wrong_way`: some pose sits only in one-way lanelets whose direction is more
  than 90 deg from the candidate's heading there.
- `lane_offset_max`: largest distance from a pose centre to the centreline of a
  lanelet containing it (NaN when no pose is inside one). NAVSIM's lane keeping
  fails above 0.5 m.

Poses in `poses.txt` are the roof pod in the map's local UTM frame
(`kitscenes_adapter.load_poses`), and candidates are in the pod frame at the
keyframe, so the box is centred on the pod. `--sanity` checks that alignment on
each scene's own driven path before any candidate is trusted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from kitscenes.map_api import load_scene_map
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

BOX_LENGTH_M = 4.9
BOX_WIDTH_M = 2.0
WRONG_WAY_RAD = math.pi / 2


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Same convention as `kitscenes_adapter._quaternion_to_yaw` (TUM ordering)."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_poses(scene: Path) -> np.ndarray:
    raw = np.loadtxt(scene / "poses.txt", ndmin=2)
    yaw = [quaternion_to_yaw(*row) for row in raw[:, 4:8]]
    return np.stack([raw[:, 1], raw[:, 2], yaw], axis=-1)


class SceneRoads:
    """The vehicle-usable part of one scene's map, indexed for point queries."""

    def __init__(self, scene: Path) -> None:
        scene_map = load_scene_map(scene)
        if scene_map is None:
            raise FileNotFoundError(f"no Lanelet2 map (or lanelet2 not installed) for {scene}")
        rules = scene_map.traffic_rules
        self.lanelets: list[tuple[Polygon, LineString, bool]] = []
        for lanelet in scene_map.lanelet_map.laneletLayer:
            if not rules.canPass(lanelet):
                continue
            polygon = Polygon([(p.x, p.y) for p in lanelet.polygon2d()])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            centreline = LineString([(p.x, p.y) for p in lanelet.centerline])
            one_way = not rules.canPass(lanelet.invert())
            self.lanelets.append((polygon, centreline, one_way))
        areas = [Polygon(a) for a in scene_map.get_drivable_area() if len(a) >= 3]
        self.tree = STRtree([polygon for polygon, _, _ in self.lanelets])
        self.drivable = unary_union([p for p, _, _ in self.lanelets] + [a.buffer(0) for a in areas])

    def containing(self, point: Point) -> list[tuple[Polygon, LineString, bool]]:
        return [self.lanelets[i] for i in self.tree.query(point, predicate="within")]

    def terms(self, poses_world: np.ndarray) -> dict:
        outside = 0.0
        inside = 0
        wrong_way = False
        offsets: list[float] = []
        for x, y, heading in poses_world:
            centre = Point(x, y)
            cos_h, sin_h = math.cos(heading), math.sin(heading)
            for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                cx = x + cos_h * dx * BOX_LENGTH_M / 2 - sin_h * dy * BOX_WIDTH_M / 2
                cy = y + sin_h * dx * BOX_LENGTH_M / 2 + cos_h * dy * BOX_WIDTH_M / 2
                outside = max(outside, self.drivable.distance(Point(cx, cy)))
            lanes = self.containing(centre)
            if not lanes:
                continue
            inside += 1
            offsets.append(min(centreline.distance(centre) for _, centreline, _ in lanes))
            against = []
            for _, centreline, one_way in lanes:
                s = centreline.project(centre)
                ahead = centreline.interpolate(min(s + 0.5, centreline.length))
                behind = centreline.interpolate(max(s - 0.5, 0.0))
                direction = math.atan2(ahead.y - behind.y, ahead.x - behind.x)
                difference = abs(math.remainder(heading - direction, 2 * math.pi))
                against.append(one_way and difference > WRONG_WAY_RAD)
            if all(against):
                wrong_way = True
        return {
            "outside_m": round(outside, 3),
            "in_lane": inside / len(poses_world),
            "wrong_way": wrong_way,
            "lane_offset_max": max(offsets) if offsets else float("nan"),
        }


def to_world(origin: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """(x, y, heading) in the frame of *origin* -> the map's local frame."""
    cos_o, sin_o = math.cos(origin[2]), math.sin(origin[2])
    x = origin[0] + cos_o * relative[:, 0] - sin_o * relative[:, 1]
    y = origin[1] + sin_o * relative[:, 0] + cos_o * relative[:, 1]
    return np.stack([x, y, origin[2] + relative[:, 2]], axis=-1)


def driven_path_sanity(roads: SceneRoads, poses: np.ndarray, stride: int = 10) -> dict:
    """The recorded drive, scored like a candidate. Near-perfect or the alignment is wrong."""
    path = poses[::stride]
    step = np.diff(path[:, :2], axis=0)
    moving = np.linalg.norm(step, axis=1) > 0.5
    headings = np.where(moving, np.arctan2(step[:, 1], step[:, 0]), path[:-1, 2])
    return roads.terms(np.stack([path[:-1, 0], path[:-1, 1], headings], axis=-1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanity", type=Path, help="write per-scene driven-path checks here")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (r["split"], r["scene_id"], r["keyframe_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sanity: dict[str, dict] = {}
    roads: SceneRoads | None = None
    current = None
    with open(args.output, "w") as out:
        for n, row in enumerate(rows, 1):
            scene = Path(row["tier"]) / "data" / row["split"] / row["scene_id"]
            if scene != current:
                roads, current = SceneRoads(scene), scene
                poses = load_poses(scene)
                sanity[row["scene_id"]] = driven_path_sanity(roads, poses)
            origin = poses[row["keyframe_index"]]
            human = to_world(origin, np.asarray(row["human_poses"]))
            candidates = [roads.terms(to_world(origin, np.asarray(p))) for p in row["proposals"]]
            out.write(
                json.dumps(
                    {
                        "scene_id": row["scene_id"],
                        "keyframe_index": row["keyframe_index"],
                        "human": roads.terms(human),
                        "candidates": candidates,
                    }
                )
                + "\n"
            )
            if n % 25 == 0 or n == len(rows):
                print(f"  {n}/{len(rows)}", flush=True)

    if args.sanity:
        args.sanity.write_text(json.dumps(sanity, indent=1))
    worst = max((s["outside_m"] for s in sanity.values()), default=float("nan"))
    wrong = sum(s["wrong_way"] for s in sanity.values())
    in_lane = min((s["in_lane"] for s in sanity.values()), default=float("nan"))
    print(
        f"driven paths over {len(sanity)} scenes: wrong-way {wrong}, lowest in-lane {in_lane:.2f}, worst outside {worst:.2f} m"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
