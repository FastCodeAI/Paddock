"""Rasterise nuScenes' drivable area around scenes, for the PDM-style DAC term.

    <ncap python> tools/benchmarks/nuscenes/build_pdm_geometry.py \\
        --nuscenes <nuScenes root> --out <dir> --neuroncap        # its 14 scenes
    <ncap python> tools/benchmarks/nuscenes/build_pdm_geometry.py \\
        --nuscenes <nuScenes root> --out <dir> --scene scene-0103 --scene scene-0110

`paddock.metrics.pdm` asks "is this point on the drivable surface?" a few hundred
thousand times per run, in paddock's own interpreter, which has no map library.
So the map is asked once, here, in an interpreter that has nuscenes-devkit and
shapely (neuroNCAP's), and the answer is kept as a boolean raster: 0.25 m cells,
80 m around the scene's logged ego path. A plan stays within about 4 m of that
path laterally and reaches about 40 m ahead, so the margin is generous; a point
outside the raster counts as not drivable.

The format is gallops' (`expert/drivable.py`: `mask`, `origin`, `cell`, `scene`,
`location`), so a raster built by either repo serves both. The layer is nuScenes'
`drivable_area`: the surface a vehicle may use, oncoming carriageway included. It
is not a lane test.

The harness reads `<dir>/<scene>.npz`; point `[benchmarks.neuroncap] pdm_geometry`
at `<dir>`, or build into `{roots.runs}/geometry/nuscenes-drivable`, its default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

#: Metres per cell: the ego is 1.85 m wide and the margins that matter are half a metre.
CELL_M = 0.25
#: How far around the logged ego path to rasterise.
PAD_M = 80.0
#: neuroNCAP's scenes (`adapters/benchmarks/neuroncap/harness.py`, OFFICIAL_SET; a test holds them equal).
NEURONCAP_SCENES = (
    "0099",
    "0101",
    "0103",
    "0106",
    "0108",
    "0110",
    "0278",
    "0331",
    "0346",
    "0783",
    "0796",
    "0921",
    "0923",
    "0966",
)


def ego_path(nusc, scene_name: str) -> tuple[np.ndarray, str]:
    """The scene's keyframe ego positions (n, 2), and its map location."""
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    location = nusc.get("log", scene["log_token"])["location"]
    token, path = scene["first_sample_token"], []
    while token:
        sample = nusc.get("sample", token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        path.append(nusc.get("ego_pose", lidar["ego_pose_token"])["translation"][:2])
        token = sample["next"]
    return np.asarray(path, dtype=float), location


def rasterise(maps: dict, root: Path, location: str, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """(rows, cols) True where a cell's centre is on the drivable area."""
    from nuscenes.map_expansion.map_api import NuScenesMap
    from shapely.ops import unary_union

    if location not in maps:
        maps[location] = NuScenesMap(dataroot=str(root), map_name=location)
    nusc_map = maps[location]
    records = nusc_map.get_records_in_patch(
        (low[0], low[1], high[0], high[1]), ["drivable_area"], mode="intersect"
    )
    polygons = [
        nusc_map.extract_polygon(token)
        for record in records["drivable_area"]
        for token in nusc_map.get("drivable_area", record)["polygon_tokens"]
    ]
    cols = int(np.ceil((high[0] - low[0]) / CELL_M))
    rows = int(np.ceil((high[1] - low[1]) / CELL_M))
    xs = low[0] + (np.arange(cols) + 0.5) * CELL_M
    ys = low[1] + (np.arange(rows) + 0.5) * CELL_M
    if not polygons:
        return np.zeros((rows, cols), dtype=bool)
    union = unary_union(polygons)
    grid_x, grid_y = np.meshgrid(xs, ys)
    try:  # shapely 2
        from shapely import contains_xy

        return contains_xy(union, grid_x, grid_y)
    except ImportError:  # shapely 1.8, neuroNCAP's env
        from shapely import vectorized

        return vectorized.contains(union, grid_x, grid_y)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nuscenes", type=Path, required=True, help="nuScenes root (v1.0-trainval/, maps/)")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", action="append", default=[], help="scene-XXXX; repeatable")
    ap.add_argument("--neuroncap", action="store_true", help="every scene neuroNCAP drives")
    ap.add_argument("--force", action="store_true", help="rebuild rasters that exist")
    args = ap.parse_args(argv)
    scenes = list(args.scene) + ([f"scene-{s}" for s in NEURONCAP_SCENES] if args.neuroncap else [])
    if not scenes:
        ap.error("name scenes with --scene, or --neuroncap")

    from nuscenes.nuscenes import NuScenes

    started = time.time()
    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes), verbose=False)
    print(f"nuScenes tables loaded in {time.time() - started:.0f} s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    maps: dict = {}
    for name in scenes:
        out = args.out / f"{name}.npz"
        if out.exists() and not args.force:
            print(f"  {name}: exists, kept", flush=True)
            continue
        path, location = ego_path(nusc, name)
        low, high = path.min(0) - PAD_M, path.max(0) + PAD_M
        mask = rasterise(maps, args.nuscenes, location, low, high)
        part = out.with_name(out.name + ".part.npz")
        np.savez_compressed(part, mask=mask, origin=low, cell=CELL_M, scene=name, location=location)
        part.replace(out)
        print(
            f"  {name} ({location}): {mask.shape[0]}x{mask.shape[1]} cells, {mask.mean():.1%} drivable",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
