"""AutoE2E v1.0 (Autoware Foundation) Reactive model as a paddock Policy.

Checkpoint: Hugging Face ``AutowareFoundation/auto_e2e`` v1.0, ``models/kitscenes-epoch-5``
(nuPlan epoch 5 fine-tuned on KITScenes), loaded with upstream's own recipe at the
training revision 40a75cf8 (``experimental/reactive-bev-learning``). The model is imported
from that checkout; nothing of it is re-implemented here except the input packing, which
follows upstream's KITScenes evaluation path (pre_extracted.py + kit_scenes/*.py):

- Six views in slot order front, front-left, front-right, rear-left, rear, rear-right
  (paddock CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT).
  Each frame is squashed to 512x512 with PIL bilinear and JPEG q90 round-tripped, the
  current front also to 1024x1024; slot 0 of the current views is the antialiased
  bilinear downsample of the normalised 1024 tile. ImageNet normalisation; the model
  converts to its caffe BGR input itself. A view the benchmark does not have at all
  (no image, path or calibration: comma2k19 is one road camera) is left out -- a zero
  tile with a projection under which nothing is visible, so it reaches no BEV cell.
  CAM_FRONT is always required.
- BEVFormer V2 T8: the current frame plus seven frames at -3.5 .. -0.5 s, oldest first,
  each with a projection that maps the *current* reference frame into that frame's
  cameras (ego motion folded in), taken from the frames that carry every view the
  current one does. Where the past does not reach back far enough (a closed-loop route
  start) the oldest such frame held is repeated with its own pose.
- Reference frame: paddock's ego frame (x forward, y left, on the ground) raised by
  ``AUTOE2E_REF_HEIGHT_M`` (default 2.1 m). The KITScenes fine-tune saw projections from
  its top-LiDAR frame, whose ground plane sits at z = -2.1 m (kit_scenes/dataset.py
  ``_TRAJECTORY_GROUND_Z_M``); set 0 to use the ground-level frame nuPlan used.
- Ego history: 64 rows x [speed, accel, yaw rate, curvature] at 10 Hz for t-6.4 .. t-0.1 s,
  derived from the pose stream as kit_scenes/egomotion.py does (centred gradients on the
  real timestamps, curvature floor 0.1 m/s), the newest acceleration zeroed (the KITScenes
  stage's causal mask). Rows older than the first observation: left at zero (upstream's
  ``left_zero`` padding) when the benchmark sends the real past; in closed loop, steady
  driving at the first known speed (see ``steady_before_start``).
- Navigation: where the benchmark names a Lanelet2 map (``obs.map_files``: KITScenes), the
  14-channel semantic map and the route matched to ``obs.route_trace`` are drawn by
  upstream's own code (`_Navigation`), as upstream's KITScenes packing draws them.
  Elsewhere no semantic map (zeros, ``map_valid`` False), and a route raster (corridor
  1.75 m either side of the route polyline, rear-clipped at -10 m, plus a 2 m destination
  disk) drawn from ``obs.route_xy`` when the benchmark sends one. ``AUTOE2E_ROUTE`` "0"
  draws no route either way; with none, zeros and ``route_valid`` False, upstream's
  camera-only track.
- Output: 64 x (acceleration, curvature) at 10 Hz, integrated with upstream's rollout
  (dt 0.1, speed first, from the t-0.1 s speed of the history) and sampled every 0.5 s:
  12 waypoints to 6.0 s with heading.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from paddock.policy.protocol import Observation, Policy, Trajectory

logger = logging.getLogger(__name__)

VIEW_ORDER = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
TILE = 512
FRONT_TILE = 1024
JPEG_QUALITY = 90
HISTORY_OFFSETS_S = (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5)
#: A history frame farther than this from its target time is not that frame -- the floor of
#: `match_tolerance`, which widens it to half the frame spacing on a sparser stream.
MATCH_TOLERANCE_S = 0.06
EGO_STEPS = 64
EGO_DT = 0.1
SPEED_FLOOR = 0.1
OUT_STEPS = 64
#: Navigation raster (navigation/geometry.py AUTOE2E_NAVIGATION_GEOMETRY).
NAV_H, NAV_W, NAV_MPP = 450, 300, 0.4
NAV_X_MAX, NAV_Y_MAX = 120.0, 60.0
NAV_X_MIN, NAV_Y_MIN = -60.0, -60.0
CORRIDOR_HALF_M = 1.75
REAR_CLIP_M = 10.0
DEST_RADIUS_M = 2.0
#: route_xy stops 200 m ahead of the ego (bench2drive agent); a shorter polyline ends at the route's end.
ROUTE_AHEAD_M = 200.0
#: Packed six-view frames kept by path (~4.7 MB each).
TILE_CACHE_FRAMES = 48
_TILE_CACHE: OrderedDict[tuple, tuple[dict, np.ndarray]] = OrderedDict()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: A left-out view's projection: every point lands at depth -1, behind the camera, so upstream's
#: depth test masks it (it clamps the depth before dividing, so no NaN) and the visible-only patch
#: skips the view outright.
INVISIBLE_PROJECTION = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]], dtype=np.float32)


def _jpeg_roundtrip(image: Image.Image) -> np.ndarray:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def pack_tile(rgb: np.ndarray, size: int) -> np.ndarray:
    """uint8 HxWx3 -> uint8 size x size x 3: PIL bilinear squash, JPEG q90.

    As kit_scenes/camera.py and parallel_pack.py pack them.
    """
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    return _jpeg_roundtrip(image)


def scaled_intrinsic(k: np.ndarray, width: int, height: int, size: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)[:3, :3].copy()
    k[0, :] *= size / width
    k[1, :] *= size / height
    return k


def yaw_of(m: np.ndarray) -> float:
    return math.atan2(m[1, 0], m[0, 0])


def match_tolerance(times: np.ndarray) -> float:
    """How far a history frame may sit from its target time and still be taken for it.

    Half the median spacing of the frames on offer, never below `MATCH_TOLERANCE_S`. A 10 Hz
    stream (KITScenes, neuroNCAP) keeps 0.06 s exactly, a dropped frame included. nuScenes
    keyframes are 0.45-0.55 s apart: at 0.06 s, 11% of the real past keyframes within 2 s were
    refused and replaced by a copy of the oldest frame. At half the spacing the nearest keyframe
    is the frame meant, since no other one is closer.

    The pick changes only where the nearest frame is between 0.06 s and half the spacing from
    its target, mid-history. NAVSIM's six-view frames are its 2 Hz keyframes, within
    milliseconds of 0.5 s steps; on its 136 test logs no history slot falls in that band (a
    missing keyframe leaves a 0.5 s hole, which still falls back), so its input is unchanged.
    Before the first frame the nearest one is the oldest, which is what the fallback takes.
    """
    if len(times) < 2:
        return MATCH_TOLERANCE_S
    return max(MATCH_TOLERANCE_S, 0.5 * float(np.median(np.diff(np.sort(times)))))


def egomotion_rows(times_s: np.ndarray, xy: np.ndarray, yaw: np.ndarray, t_now: float) -> np.ndarray:
    """[64, 4] = speed, accel, yaw rate, curvature at t_now-6.4 .. t_now-0.1 (oldest first).

    kit_scenes/egomotion.py on the pose stream: speed = |d xy / dt|, accel = d speed / dt,
    yaw rate = d unwrap(yaw) / dt (np.gradient on the real timestamps, centred inside),
    curvature = yaw rate / max(speed, 0.1). Rows before the first pose stay zero.
    """
    rows = np.zeros((EGO_STEPS, 4), dtype=np.float64)
    if len(times_s) < 2:
        return rows
    order = np.argsort(times_s)
    t, xy, yaw = times_s[order], xy[order], np.unwrap(yaw[order])
    speed = np.linalg.norm(np.gradient(xy, t, axis=0), axis=1)
    accel = np.gradient(speed, t)
    yaw_rate = np.gradient(yaw, t)
    curvature = yaw_rate / np.where(speed < SPEED_FLOOR, SPEED_FLOOR, speed)
    targets = t_now - EGO_DT * np.arange(EGO_STEPS, 0, -1)
    inside = targets >= t[0] - 1e-6
    for col, signal in enumerate((speed, accel, yaw_rate, curvature)):
        rows[inside, col] = np.interp(targets[inside], t, signal)
    return rows


def steady_before_start(
    rows: np.ndarray, times_s: np.ndarray, t_now: float, speed: float
) -> tuple[np.ndarray, int]:
    """`rows` with every row older than the first pose set to steady driving, and how many.

    Steady driving is the first measured speed with no acceleration and no turn; with fewer
    than two poses nothing is measured yet and every row takes `speed`, the simulator's.

    Closed loop only. The KITScenes fine-tune always had all 64 rows real (samples start at
    ``frame_idx >= history_steps``, kit_scenes/egomotion.py) and upstream's own benchmark
    zero-pads at most 24 of them. A simulator episode starts with no past at all, so zeros
    fill 32-64 rows for its whole length and read as a car that stood still until a moment
    ago. On neuroNCAP 0103 that turned the car right within 2 s, into a parked car; with the
    past filled it drove straight.
    """
    out = np.array(rows, dtype=np.float64, copy=True)
    if len(times_s) < 2:
        lead = np.ones(EGO_STEPS, dtype=bool)
    else:
        targets = t_now - EGO_DT * np.arange(EGO_STEPS, 0, -1)
        lead = targets < float(np.min(times_s)) - 1e-6  # the rows egomotion_rows left at zero
    if not lead.any():
        return out, 0
    first = float(out[~lead][0, 0]) if (~lead).any() else float(speed)
    out[lead] = (first, 0.0, 0.0, 0.0)
    return out, int(lead.sum())


#: ``AUTOE2E_EGO_HISTORY``: what the network is told about the car's past.
EGO_HISTORY_MODES = ("real", "steady", "no_turn", "steady_speed")


def ego_history_as(rows: np.ndarray, mode: str) -> np.ndarray:
    """The 64 ego rows as ``mode`` feeds them; ``"real"`` returns them unchanged.

    ``"steady"`` replaces the past with steady straight driving at the newest
    measured speed -- an extrapolated past, no launch and no turn; ``"no_turn"``
    keeps the measured speed and acceleration and zeroes yaw rate and curvature;
    ``"steady_speed"`` is its other half: speed and acceleration steady at the
    newest measured speed, yaw rate and curvature as measured. None of them is
    upstream's input: they exist to test, in closed loop, whether the measured
    past is what turns AutoE2E off its lane on Bench2Drive (the deep-dive notebook
    under tools/models/autoe2e/analysis shows it moves the plan offline).
    """
    if mode not in EGO_HISTORY_MODES:
        raise ValueError(f"AUTOE2E_EGO_HISTORY must be one of {EGO_HISTORY_MODES}, got {mode!r}")
    out = np.array(rows, dtype=np.float64, copy=True)
    if mode == "steady":
        out[:] = (float(rows[-1, 0]), 0.0, 0.0, 0.0)
    elif mode == "no_turn":
        out[:, 2:] = 0.0
    elif mode == "steady_speed":
        out[:, :2] = (float(rows[-1, 0]), 0.0)
    return out


def route_raster(route_xy: np.ndarray | None) -> tuple[np.ndarray, bool, bool]:
    """(2, 450, 300) float32 corridor + destination in the ego frame.

    Returns (raster, corridor drawn, destination drawn).
    """
    out = np.zeros((2, NAV_H, NAV_W), dtype=np.float32)
    if route_xy is None or len(route_xy) < 2:
        return out, False, False
    pts = np.asarray(route_xy, dtype=np.float64)[:, :2]
    rows = np.arange(NAV_H, dtype=np.float64)
    cols = np.arange(NAV_W, dtype=np.float64)
    cx = NAV_X_MAX - (rows + 0.5) * NAV_MPP  # pixel-centre x per row
    cy = NAV_Y_MAX - (cols + 0.5) * NAV_MPP  # pixel-centre y per column
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    corridor = np.zeros((NAV_H, NAV_W), dtype=bool)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        lo = np.minimum(a, b) - CORRIDOR_HALF_M
        hi = np.maximum(a, b) + CORRIDOR_HALF_M
        if hi[0] < NAV_X_MIN or lo[0] > NAV_X_MAX or hi[1] < NAV_Y_MIN or lo[1] > NAV_Y_MAX:
            continue
        r0 = max(0, int(math.floor((NAV_X_MAX - hi[0]) / NAV_MPP - 0.5)))
        r1 = min(NAV_H - 1, int(math.ceil((NAV_X_MAX - lo[0]) / NAV_MPP - 0.5)))
        c0 = max(0, int(math.floor((NAV_Y_MAX - hi[1]) / NAV_MPP - 0.5)))
        c1 = min(NAV_W - 1, int(math.ceil((NAV_Y_MAX - lo[1]) / NAV_MPP - 0.5)))
        if r0 > r1 or c0 > c1:
            continue
        px, py = X[r0 : r1 + 1, c0 : c1 + 1], Y[r0 : r1 + 1, c0 : c1 + 1]
        d = b - a
        L2 = float(d @ d)
        u = (
            np.zeros_like(px)
            if L2 <= 0
            else np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0.0, 1.0)
        )
        dist = np.hypot(px - (a[0] + u * d[0]), py - (a[1] + u * d[1]))
        corridor[r0 : r1 + 1, c0 : c1 + 1] |= dist <= CORRIDOR_HALF_M
    corridor &= X >= -REAR_CLIP_M
    out[0] = corridor
    # Destination: the polyline's end when it stops short of the 200 m look-ahead (the route ends there).
    nearest = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))
    ahead = (
        float(np.linalg.norm(np.diff(pts[nearest:], axis=0), axis=1).sum()) if nearest < len(pts) - 1 else 0.0
    )
    dest_drawn = False
    if ahead < ROUTE_AHEAD_M - 1.0:
        dx, dy = pts[-1]
        if NAV_X_MIN <= dx <= NAV_X_MAX and NAV_Y_MIN <= dy <= NAV_Y_MAX:
            disk = np.hypot(X - dx, Y - dy) <= DEST_RADIUS_M
            out[1] = disk
            dest_drawn = bool(disk.any())
    return out, bool(corridor.any()), dest_drawn


def finite_controls(controls: np.ndarray) -> np.ndarray:
    """The model's [64, 2] controls, or RuntimeError if any is not finite.

    A crash must not become a score: a plan substituted here would be scored under
    AutoE2E's name with nothing in the record to tell it apart. Raising fails the
    scenario visibly instead (METEOR's adapter does the same).
    """
    bad = ~np.isfinite(controls)
    if bad.any():
        raise RuntimeError(
            f"AutoE2E returned non-finite controls at {int(bad.any(axis=1).sum())} of {len(controls)} steps"
        )
    return controls


def blank_views(obs: Observation) -> frozenset:
    """The views to leave out of a plan: the ones this benchmark does not have at all.

    A view is absent only if the observation carries no image, no path *and* no calibration
    for it -- comma2k19 is one road camera and says nothing about the other five. Anything
    short of that raises: a view with calibration but no pixels is a frame that lost its
    image (NAVSIM's dense frames keep all six calibrations), pixels without calibration
    cannot be projected, and CAM_FRONT is never left out. A benchmark that should have sent
    a view must not be scored as if it had none.
    """
    blank = frozenset(
        c
        for c in VIEW_ORDER[1:]
        if c not in obs.images
        and c not in obs.image_paths
        and c not in obs.camera2image
        and c not in obs.camera2ego
    )
    missing = [c for c in VIEW_ORDER if c not in blank and c not in obs.images and c not in obs.image_paths]
    if missing:
        raise RuntimeError(
            f"AutoE2E plans from every view the benchmark has; the current observation has none for {missing}"
        )
    uncalibrated = [
        c for c in VIEW_ORDER if c not in blank and (c not in obs.camera2image or c not in obs.camera2ego)
    ]
    if uncalibrated:
        raise RuntimeError(f"AutoE2E needs intrinsics and extrinsics for {uncalibrated}")
    return blank


def portable_checkpoint(path: str) -> str:
    """The checkpoint as the release names it, never where this machine keeps it.

    `diagnostics()` lands verbatim in the tracked run.json (runner.py copies `/diagnostics`
    into its provenance), so an absolute path there commits a site's weights root and fails
    the portability gate. The release layout `models/<variant>/checkpoint.pt` keeps the
    variant, the part a reader needs; `checkpoint_sha256` pins the exact file.
    """
    parts = Path(path).parts
    if len(parts) >= 3 and parts[-3] == "models":
        return "/".join(parts[-3:])
    return Path(path).name


def integrate_controls(controls: np.ndarray, v0: float) -> np.ndarray:
    """[64, 2] (accel, curvature) -> [64, 3] (x, y, heading); training/losses/control_rollout.py."""
    accel, kappa = controls[:, 0].astype(np.float64), controls[:, 1].astype(np.float64)
    v = np.empty(OUT_STEPS)
    speed = float(v0)
    for k in range(OUT_STEPS):
        speed = max(speed + accel[k] * EGO_DT, 0.0)
        v[k] = speed
    heading = np.cumsum(v * kappa * EGO_DT)
    x = np.cumsum(v * np.cos(heading) * EGO_DT)
    y = np.cumsum(v * np.sin(heading) * EGO_DT)
    return np.stack([x, y, heading], axis=1)


def navigation_modules() -> SimpleNamespace:
    """Upstream's map and route code: what kit_scenes/navigation.py calls to pack a sample.

    Imported only when a benchmark sends a map, and refused loudly if it cannot be: a map
    quietly dropped would score the camera-only track under the map's name.
    """
    try:
        from data_parsing.kit_scenes.map import _cached_scene_map
        from data_parsing.kit_scenes.source import KITSCENES_DATA_REVISION
        from navigation.geometry import AUTOE2E_NAVIGATION_GEOMETRY
        from navigation.lanelet2_adapter import Lanelet2MapAdapter, file_sha256
        from navigation.lanelet2_matcher import Lanelet2TraceMatcher
        from navigation.rasterizer import EgoPose, NativeNavigationRasterizer

        rasterizer = NativeNavigationRasterizer(AUTOE2E_NAVIGATION_GEOMETRY)
    except Exception as exc:  # ImportError, or the rasterizer library not built
        raise RuntimeError(
            f"the benchmark sent a map and AutoE2E cannot draw it ({type(exc).__name__}: {exc}); "
            "setup.sh installs the KITScenes SDK (Lanelet2) and builds upstream's rasterizer"
        ) from exc
    return SimpleNamespace(
        scene_map=_cached_scene_map,
        revision=KITSCENES_DATA_REVISION,
        map_adapter=Lanelet2MapAdapter,
        file_sha256=file_sha256,
        matcher=Lanelet2TraceMatcher,
        pose=EgoPose,
        rasterizer=rasterizer,
    )


class _Navigation:
    """A scene's semantic map, and the route matched to its logged drive, drawn by upstream.

    The calls kit_scenes/navigation.py makes when it packs a KITScenes sample: the map
    extracted by navigation/lanelet2_adapter.py, the route matched by
    navigation/lanelet2_matcher.py (whose quality gates decide `route_valid`, as upstream's
    do), both rendered by navigation/rasterizer.py in AUTOE2E_NAVIGATION_GEOMETRY. One
    difference, kept on purpose: upstream renders at the latest pose on a 0.5 s grid and
    warps that raster to the sample's pose, while this renders at the sample's pose. On
    paddock's KITScenes keyframes the two are the same pose, except where upstream's
    nanosecond truncation of the log's clock puts its grid pose 0.5 s back (docs/models/autoe2e.md).

    Maps are read from a scratch copy: for a scene with degenerate lanelets, upstream writes a
    sanitised map.osm beside the one it was given (kit_scenes/map.py), and the dataset is not
    this process's to write to. The copy lasts only for the load (the SDK parses map.osm on
    first use, which `extract()` is; the matcher works from memory): a node is stopped with
    terminate() and then kill(), so a copy held for its life was left on the host by every run.
    """

    def __init__(self, modules: SimpleNamespace) -> None:
        self.m = modules
        self._maps: dict[str, tuple] = {}
        self._routes: dict[tuple[str, str], Any] = {}

    def _scene(self, map_files: dict[str, str]) -> tuple:
        """(scene id, the SDK's scene map, upstream's NavigationMap, map sha256), once per map."""
        if map_files.get("format") != "lanelet2_osm":
            raise RuntimeError(f"AutoE2E draws Lanelet2 maps; the benchmark sent {map_files.get('format')!r}")
        key = map_files["map"]
        if key not in self._maps:
            scene_id = Path(map_files["map"]).parent.parent.name
            with tempfile.TemporaryDirectory(prefix="autoe2e-maps-") as scratch:
                shadow = Path(scratch) / scene_id
                (shadow / "maps").mkdir(parents=True)
                shutil.copyfile(map_files["map"], shadow / "maps" / "map.osm")
                shutil.copyfile(map_files["origin"], shadow / "maps" / "origin.json")
                sha = self.m.file_sha256(shadow / "maps" / "map.osm")
                scene_map = self.m.scene_map(shadow)
                navigation_map = self.m.map_adapter(
                    scene_map,
                    map_version=f"kitscenes:{scene_id}:{sha[:16]}",
                    map_sha256=sha,
                    frame_id=f"kitscenes:{scene_id}:local_enu",
                    source_revision=self.m.revision,
                ).extract()
            self._maps[key] = (scene_id, scene_map, navigation_map, sha)
        return self._maps[key]

    def _route(self, scene: tuple, trace: np.ndarray):
        """The route upstream's matcher fits to the logged drive, once per scene and drive."""
        scene_id, scene_map, navigation_map, sha = scene
        trace = np.asarray(trace, dtype=np.float64)
        key = (sha, hashlib.sha256(trace.tobytes()).hexdigest())
        if key not in self._routes:
            self._routes[key] = self.m.matcher(
                scene_map, navigation_map, map_sha256=sha, source_revision=self.m.revision
            ).match(
                scene_id=scene_id,
                positions_enu_m=trace[:, 1:3],
                yaws_rad=trace[:, 3],
                timestamps_ns=np.round(trace[:, 0] * 1e9).astype(np.int64),
            )
        return self._routes[key]

    def raster(
        self, map_files: dict[str, str], trace: np.ndarray | None, ego2world: np.ndarray, t_us: int
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        """(map [14, 450, 300], route [2, 450, 300], map_valid, route_valid), float32, at the ego pose.

        `trace` None draws the map alone (route zeros, `route_valid` False)."""
        scene = self._scene(map_files)
        route = self._route(scene, trace) if trace is not None else None
        pose = self.m.pose(
            x_enu_m=float(ego2world[0, 3]),
            y_enu_m=float(ego2world[1, 3]),
            yaw_rad=yaw_of(ego2world),
            timestamp_ns=int(t_us) * 1000,
        )
        drawn = self.m.rasterizer.render(scene[2], route, pose)
        return (
            np.ascontiguousarray(drawn.map_context, dtype=np.float32),
            np.ascontiguousarray(drawn.route_mask, dtype=np.float32),
            bool(drawn.map_valid),
            bool(drawn.route_valid),
        )


class _ConstTensorTorch:
    """`torch` for bevformer_v2_t1, with `torch.tensor([w, h], device=cuda)` built once per value.

    MultiScaleSpatialCrossAttention.forward builds its [width, height] normaliser inside the
    view x query-chunk x level loop: 17,280 pageable host->device copies a forward, each one a
    stream synchronisation (6.3 of 9.1 s CPU in a profile, against 2.1 s of GPU work, and far
    worse on a shared GPU). The cached tensor holds the same float32 values, so outputs are
    bit-identical; every other attribute is torch's own.
    """

    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self._cache: dict = {}

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def tensor(self, data, *args, device=None, dtype=None, **kwargs):
        if (
            not args
            and not kwargs
            and device is not None
            and str(device) != "cpu"
            and isinstance(data, (list, tuple))
            and all(isinstance(v, (int, float)) for v in data)
        ):
            key = (tuple(data), str(device), dtype)
            cached = self._cache.get(key)
            if cached is None:
                cached = self._cache[key] = self._torch.tensor(list(data), device=device, dtype=dtype)
            return cached
        return self._torch.tensor(data, *args, device=device, dtype=dtype, **kwargs)


def visible_only_cross_attention(original_forward):
    """MultiScaleSpatialCrossAttention.forward computing each view only for the BEV queries it sees.

    Upstream samples all 60,000 queries in every one of the six views and multiplies the
    queries a view does not see by zero (``camera_outputs + view_output * view_visible``).
    Every op on the way is per query (row-wise linear layers, per-row softmax, per-point
    grid_sample), so restricting a view to its visible queries changes nothing for those
    queries and skips the ones that are zeroed anyway: same sums, in the same view order.
    The 96%-of-runtime BEV encoder then does ~1/4 of the cross-attention work. Batch 1 only;
    anything else runs upstream's code.
    """
    import torch

    def forward(
        self,
        query,
        features,
        reference_points_2d,
        reference_mask,
        *,
        num_views,
        level_embeddings,
        camera_embeddings,
        content_delta_only=False,
    ):
        batch_size, query_count, channels = query.shape
        if batch_size != 1 or reference_points_2d.shape[0] != 1:
            return original_forward(
                self,
                query,
                features,
                reference_points_2d,
                reference_mask,
                num_views=num_views,
                level_embeddings=level_embeddings,
                camera_embeddings=camera_embeddings,
                content_delta_only=content_delta_only,
            )
        if channels != self.embed_dim or len(features) != self.num_levels:
            raise ValueError("spatial cross-attention input differs from contract")
        values = self._project_values(
            features,
            batch_size=1,
            num_views=num_views,
            level_embeddings=level_embeddings,
            camera_embeddings=camera_embeddings,
            content_delta_only=content_delta_only,
        )
        pillar_points = reference_points_2d.shape[3]
        anchor_indices = torch.arange(self.num_points, device=query.device) % pillar_points
        camera_outputs = query.new_zeros(1, query_count, self.embed_dim)
        camera_counts = query.new_zeros(1, query_count, 1)
        normalizers = [
            torch.tensor([v.shape[-1], v.shape[-2]], device=query.device, dtype=torch.float32) for v in values
        ]
        for view_index in range(num_views):
            index = reference_mask[0, view_index].any(dim=-1).nonzero().squeeze(1)
            count = int(index.numel())
            if count == 0:
                continue
            view_query = query[:, index]
            view_reference = reference_points_2d[0, view_index][index][:, anchor_indices].float().unsqueeze(0)
            output_chunks = []
            for start in range(0, count, self.query_chunk_size):
                stop = min(start + self.query_chunk_size, count)
                chunk_size = stop - start
                chunk_query = view_query[:, start:stop]
                offsets = self.sampling_offsets(chunk_query).reshape(
                    1, chunk_size, self.num_heads, self.num_levels, self.num_points, 2
                )
                weights = self.attention_weights(chunk_query).reshape(
                    1, chunk_size, self.num_heads, self.num_levels, self.num_points
                )
                weights = weights.flatten(-2).softmax(dim=-1).reshape_as(weights)
                attended = chunk_query.new_zeros(1, chunk_size, self.num_heads, self.head_dim)
                reference = view_reference[:, start:stop]
                for level, value in enumerate(values):
                    height, width = value.shape[-2:]
                    locations = reference.unsqueeze(2) + offsets[:, :, :, level].float() / normalizers[level]
                    sample_grid = (
                        (locations * 2.0 - 1.0)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(self.num_heads, chunk_size, self.num_points, 2)
                    )
                    value_per_head = (
                        value[:, view_index]
                        .reshape(1, self.num_heads, self.head_dim, height, width)
                        .reshape(self.num_heads, self.head_dim, height, width)
                    )
                    attended = attended + self._sample_level(
                        value_per_head, sample_grid, weights[:, :, :, level]
                    )
                output_chunks.append(attended.reshape(1, chunk_size, self.embed_dim))
            view_output = torch.cat(output_chunks, dim=1)
            ones = torch.ones(1, count, 1, dtype=query.dtype, device=query.device)
            camera_outputs[:, index] = camera_outputs[:, index] + view_output * ones
            camera_counts[:, index] = camera_counts[:, index] + ones
        attended = camera_outputs / camera_counts.clamp_min(1.0)
        if content_delta_only:
            import torch.nn.functional as F

            return self.dropout(F.linear(attended, self.output_proj.weight, bias=None))
        return query + self.dropout(self.output_proj(attended))

    return forward


@dataclass
class _Frame:
    t_s: float
    ego2world: np.ndarray
    camera2image: dict[str, np.ndarray]
    camera2ego: dict[str, np.ndarray]
    images: dict[str, np.ndarray] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    _sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    _tiles: np.ndarray | None = None
    #: Views the observation arrived with. Fixed at construction: `images` is emptied once the
    #: frame is packed, so testing it later would call a packed six-view frame incomplete.
    views: frozenset = frozenset()

    def __post_init__(self) -> None:
        self.views = frozenset(c for c in VIEW_ORDER if c in self.images or c in self.paths)

    @property
    def full(self) -> bool:
        """All six views present (NAVSIM's 10 Hz frames between keyframes carry the ego state only)."""
        return len(self.views) == len(VIEW_ORDER)

    def rgb(self, cam: str) -> np.ndarray:
        if cam in self.images:
            return self.images[cam]
        from paddock.policy.protocol import load_frame

        return load_frame(self.paths[cam])

    def size(self, cam: str) -> tuple[int, int]:
        """(width, height) of the delivered frame, read from the header for a path."""
        if cam not in self._sizes:
            if cam in self.images:
                self._sizes[cam] = (self.images[cam].shape[1], self.images[cam].shape[0])
            elif str(self.paths[cam]).endswith(".npy"):
                # A raw staged frame (comma2k19): PIL cannot open it, and the header holds the shape.
                shape = np.load(self.paths[cam], mmap_mode="r").shape
                self._sizes[cam] = (int(shape[1]), int(shape[0]))
            else:
                with Image.open(self.paths[cam]) as im:
                    self._sizes[cam] = im.size
        return self._sizes[cam]

    def tiles(self, blank: frozenset = frozenset()) -> np.ndarray:
        """uint8 [6, 512, 512, 3], computed once (also the history copy of the front).

        A view in `blank` (`blank_views`) is a zero tile, never read; `infer` zeroes it again
        after normalisation. Frames read from disk are cached by their six paths across
        episodes: consecutive NAVSIM scenes of a log share seven of their eight camera
        keyframes, and the pixels are the same.
        """
        if self._tiles is None:
            key = tuple(self.paths.get(c) for c in VIEW_ORDER) if not self.images else None
            hit = _TILE_CACHE.get(key) if key is not None else None
            if hit is not None:
                _TILE_CACHE.move_to_end(key)
                self._sizes, self._tiles = dict(hit[0]), hit[1]
                return self._tiles
            for c in VIEW_ORDER:
                if c not in blank:
                    self.size(c)
            self._tiles = np.stack(
                [
                    np.zeros((TILE, TILE, 3), dtype=np.uint8) if c in blank else pack_tile(self.rgb(c), TILE)
                    for c in VIEW_ORDER
                ]
            )
            self.images = {}  # pixels no longer needed once packed
            if key is not None:
                _TILE_CACHE[key] = (dict(self._sizes), self._tiles)
                while len(_TILE_CACHE) > TILE_CACHE_FRAMES:
                    _TILE_CACHE.popitem(last=False)
        return self._tiles


class AutoE2EPolicy(Policy):
    cameras = VIEW_ORDER
    waypoint_dt = 0.5
    #: Closed loop (a simulator calling /infer each tick): the ego history before the first
    #: pose is filled with steady driving. `infer_sequence` clears this, so open-loop
    #: benchmarks, which send the real past, keep upstream's zero padding. Camera history is
    #: never manufactured: a short past repeats the oldest frame, as upstream does.
    self_prime = True

    def __init__(self) -> None:
        self.checkpoint = os.environ.get("AUTOE2E_CHECKPOINT", "")
        self.repo = os.environ.get("AUTOE2E_REPO", "")
        self.ref_height = float(os.environ.get("AUTOE2E_REF_HEIGHT_M", "2.1"))
        self.use_route = os.environ.get("AUTOE2E_ROUTE", "1") != "0"
        self.bf16 = os.environ.get("AUTOE2E_BF16", "1") != "0"
        self.record_dir = os.environ.get("AUTOE2E_RECORD_DIR") or None
        #: Seconds between forward passes. 0 = every call. Closed loop calls at 10 Hz so the ego
        #: history is real 10 Hz; the plan is refreshed every `replan_s` and held in between,
        #: re-expressed in the current ego frame. (The Bench2Drive client, asking at a declared
        #: `planning_hz`, holds its plan unchanged in the frame it was made in.)
        self.replan_s = float(os.environ.get("AUTOE2E_REPLAN_S", "0") or 0)
        self.ego_history_mode = os.environ.get("AUTOE2E_EGO_HISTORY", "real") or "real"
        ego_history_as(np.zeros((EGO_STEPS, 4)), self.ego_history_mode)  # refuse a typo before any load
        self._plan: tuple[float, np.ndarray, np.ndarray] | None = None
        self.model = None
        self.device = None
        self.frames: list[_Frame] = []
        self.last: dict[str, Any] = {}
        self.calls = 0
        #: Plans made with a semantic map, and with a route raster: what run.json says was drawn.
        self.map_calls = 0
        self.route_calls = 0
        self._nav: _Navigation | None = None
        self._ckpt_sha = None
        self._record = None

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        # Both checks before torch: a missing setting is named, not hidden behind an ImportError.
        if not self.checkpoint or not os.path.isfile(self.checkpoint):
            raise RuntimeError(f"AUTOE2E_CHECKPOINT {self.checkpoint!r} is not a file")
        model_dir = os.path.join(self.repo, "Model")
        if not os.path.isdir(os.path.join(model_dir, "model_components")):
            raise RuntimeError(
                f"AUTOE2E_REPO {self.repo!r} has no Model/model_components (upstream checkout)"
            )
        import torch

        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from model_components.auto_e2e import AutoE2E
        from model_components.view_fusion import bevformer_v2_t1
        from model_components.view_fusion.projection import PinholeProjection
        from training.reactive_multitask import ReactiveTrainingStage, reactive_model_kwargs

        if os.environ.get("AUTOE2E_CACHE_CONSTS", "1") != "0":
            bevformer_v2_t1.torch = _ConstTensorTorch(torch)
        cross = bevformer_v2_t1.MultiScaleSpatialCrossAttention
        if not hasattr(cross, "_upstream_forward"):
            cross._upstream_forward = cross.forward
        cross.forward = (
            visible_only_cross_attention(cross._upstream_forward)
            if os.environ.get("AUTOE2E_VISIBLE_ONLY", "1") != "0"
            else cross._upstream_forward
        )
        self._Pinhole = PinholeProjection
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        config = payload["config"]
        self.config = {
            k: config[k]
            for k in ("training_stage", "backbone", "embed_dim", "num_views", "model_architecture_version")
        }
        if int(config["num_views"]) != len(VIEW_ORDER):
            raise RuntimeError(
                f"checkpoint has {config['num_views']} views; this adapter packs {len(VIEW_ORDER)}"
            )
        model = AutoE2E(
            backbone=str(config["backbone"]),
            embed_dim=int(config["embed_dim"]),
            is_pretrained=False,
            **reactive_model_kwargs(
                ReactiveTrainingStage(config["training_stage"]), num_views=int(config["num_views"])
            ),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        h = hashlib.sha256()
        with open(self.checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        self._ckpt_sha = h.hexdigest()
        if self.record_dir:
            os.makedirs(self.record_dir, exist_ok=True)
        logger.info("AutoE2E loaded %s (%s) on %s", self.checkpoint, self.config, self.device)

    def reset(self) -> None:
        self.frames = []
        self.last = {}
        self._plan = None
        if self._record is not None:
            self._record.close()
            self._record = None

    # ------------------------------------------------------------------ history
    def _push(self, obs: Observation) -> _Frame:
        frame = _Frame(
            t_s=obs.timestamp_us / 1e6,
            ego2world=np.asarray(obs.ego2world, dtype=np.float64),
            camera2image={
                c: np.asarray(obs.camera2image[c], dtype=np.float64)
                for c in VIEW_ORDER
                if c in obs.camera2image
            },
            camera2ego={
                c: np.asarray(obs.camera2ego[c], dtype=np.float64) for c in VIEW_ORDER if c in obs.camera2ego
            },
            images={c: obs.images[c] for c in VIEW_ORDER if c in obs.images},
            paths={c: obs.image_paths[c] for c in VIEW_ORDER if c in obs.image_paths and c not in obs.images},
        )
        if self.frames and frame.t_s == self.frames[-1].t_s:
            # The same instant again (NAVSIM's dense history repeats a log's first frame
            # for the seconds before the log starts): one frame, the newest copy.
            self.frames[-1] = frame
            return frame
        if self.frames and frame.t_s < self.frames[-1].t_s:
            # Time going backwards is a new episode the caller did not reset.
            self.frames = []
        self.frames.append(frame)
        # Keep 6.6 s: the ego history reaches back 6.4 s plus one row for the centred gradient.
        while len(self.frames) > 2 and frame.t_s - self.frames[1].t_s >= 6.6:
            self.frames.pop(0)
        return frame

    def observe(self, obs: Observation) -> None:
        self._push(obs)

    def _history(self, now: _Frame, blank: frozenset = frozenset()) -> tuple[list[_Frame], list[float]]:
        """Seven frames at -3.5 .. -0.5 s, oldest first, from the frames that carry every view not
        in `blank`: all six on NAVSIM, whose 10 Hz frames between keyframes carry three and never
        count; the road camera alone on comma2k19."""
        needed = frozenset(VIEW_ORDER) - blank
        full = [f for f in self.frames if needed <= f.views and f is not now] or [now]
        times = np.array([f.t_s for f in full])
        tolerance = match_tolerance(times)
        picked, gaps = [], []
        for off in HISTORY_OFFSETS_S:
            target = now.t_s + off
            i = int(np.argmin(np.abs(times - target)))
            if abs(times[i] - target) <= tolerance:
                picked.append(full[i])
                gaps.append(round(float(times[i] - target), 3))
            else:
                # Not that far back: the oldest such frame held, with its own pose (route / log start).
                picked.append(full[0])
                gaps.append(None)
        return picked, gaps

    # ------------------------------------------------------------------ geometry
    def _ref2ego(self) -> np.ndarray:
        m = np.eye(4)
        m[2, 3] = self.ref_height
        return m

    def _projections(
        self,
        frame: _Frame,
        ref_now2world: np.ndarray | None,
        size: int,
        views=VIEW_ORDER,
        blank: frozenset = frozenset(),
    ) -> np.ndarray:
        """[V, 3, 4]: current reference frame -> pixels of `frame`'s cameras at `size` x `size`.

        A view in `blank` gets INVISIBLE_PROJECTION."""
        ref2ego = self._ref2ego()
        ref_frame2world = frame.ego2world @ ref2ego
        hist_from_now = np.eye(4) if ref_now2world is None else np.linalg.inv(ref_frame2world) @ ref_now2world
        out = []
        for c in views:
            if c in blank:
                out.append(INVISIBLE_PROJECTION)
                continue
            w, h = frame.size(c)
            k = scaled_intrinsic(frame.camera2image[c], w, h, size)
            cam2ref = np.linalg.inv(ref2ego) @ frame.camera2ego[c]
            out.append(k @ (np.linalg.inv(cam2ref) @ hist_from_now)[:3, :])
        return np.stack(out).astype(np.float32)

    def _ego_history(self, now: _Frame, speed: float) -> tuple[np.ndarray, int]:
        """[64, 4] ego history at `now` from every frame held, and how many rows were filled.

        All frames held, the current one included, so the newest rows get a centred gradient.
        Closed loop fills the rows before the first pose (`steady_before_start`); open loop,
        where `infer_sequence` has cleared `self_prime`, keeps upstream's zero padding.
        """
        t = np.array([f.t_s for f in self.frames])
        e2w = [f.ego2world for f in self.frames]
        ego = egomotion_rows(
            t, np.array([m[:2, 3] for m in e2w]), np.array([yaw_of(m) for m in e2w]), now.t_s
        )
        if not self.self_prime:
            return ego, 0
        return steady_before_start(ego, t, now.t_s, speed)

    # ------------------------------------------------------------------ infer
    def infer(self, obs: Observation) -> Trajectory:
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        now = self._push(obs)
        blank = blank_views(obs)
        if (
            self.replan_s > 0
            and self._plan is not None
            and now.t_s - self._plan[0] < self.replan_s - 0.02
            and now.t_s > self._plan[0]
        ):
            t_plan, e2w_plan, waypoints = self._plan
            rel = np.linalg.inv(now.ego2world) @ e2w_plan
            xy = waypoints[:, :2] @ rel[:2, :2].T + rel[:2, 3]
            held = np.column_stack([xy, waypoints[:, 2] + math.atan2(rel[1, 0], rel[0, 0])])
            return Trajectory(
                held, dt=self.waypoint_dt, extras={"autoe2e": {"held_s": round(now.t_s - t_plan, 3)}}
            )
        history, gaps = self._history(now, blank)
        dev = self.device

        def normalise(arr: np.ndarray) -> torch.Tensor:  # uint8 [..., H, W, 3] -> float [..., 3, H, W]
            x = torch.from_numpy(arr).to(dev, non_blocking=True).float().div_(255.0)
            x = (x - torch.from_numpy(IMAGENET_MEAN).to(dev)) / torch.from_numpy(IMAGENET_STD).to(dev)
            return x.movedim(-1, -3).contiguous()

        front_1024 = pack_tile(now.rgb("CAM_FRONT"), FRONT_TILE)
        current = now.tiles(blank)
        front_t = normalise(front_1024)[None]  # [1, 3, 1024, 1024]
        tiles_t = normalise(current)[None]  # [1, 6, 3, 512, 512]
        tiles_t[:, 0] = F.interpolate(
            front_t, size=(TILE, TILE), mode="bilinear", align_corners=False, antialias=True
        )
        hist_t = normalise(np.stack([f.tiles(blank) for f in history]))[None]  # [1, 7, 6, 3, 512, 512]
        if blank:
            # A left-out view is no image at all, not a black one: zero after normalisation.
            slots = [VIEW_ORDER.index(c) for c in sorted(blank)]
            tiles_t[:, slots] = 0.0
            hist_t[:, :, slots] = 0.0

        ref_now2world = now.ego2world @ self._ref2ego()
        P = torch.from_numpy(self._projections(now, None, TILE, blank=blank))[None].to(dev)
        PF = torch.from_numpy(self._projections(now, None, FRONT_TILE, views=("CAM_FRONT",)))[None].to(dev)
        PH = [
            torch.from_numpy(self._projections(f, ref_now2world, TILE, blank=blank))[None].to(dev)
            for f in history
        ]

        ego, filled = self._ego_history(now, float(obs.speed))
        ego = ego_history_as(ego, self.ego_history_mode)
        v0 = float(ego[-1, 0])
        ego_in = ego.copy()
        ego_in[-1, 1] = 0.0  # KITScenes stage: newest acceleration masked (dataset_policy.py)
        ego_t = torch.from_numpy(ego_in.reshape(1, -1).astype(np.float32)).to(dev)

        semantic = None
        if obs.map_files:
            if self._nav is None:
                self._nav = _Navigation(navigation_modules())
            semantic, route, map_valid, route_valid = self._nav.raster(
                obs.map_files, obs.route_trace if self.use_route else None, now.ego2world, obs.timestamp_us
            )
            dest = route_valid and bool(route[1].any())
        else:
            route, corridor, dest = route_raster(obs.route_xy if self.use_route else None)
            map_valid, route_valid = False, bool(corridor)
        map_t = (
            torch.from_numpy(semantic)[None].to(dev)
            if map_valid
            else torch.zeros(1, 14, NAV_H, NAV_W, device=dev)
        )
        route_t = (
            torch.from_numpy(route)[None].to(dev)
            if route_valid
            else torch.zeros(1, 2, NAV_H, NAV_W, device=dev)
        )
        self.map_calls += map_valid
        self.route_calls += route_valid
        Pin = self._Pinhole
        kwargs = dict(
            route_mask=route_t,
            map_valid=torch.tensor([map_valid], dtype=torch.bool, device=dev),
            route_valid=torch.tensor([route_valid], dtype=torch.bool, device=dev),
            projection=Pin(P, geometry_type="pinhole"),
            geometry_type="pinhole",
            camera_history_tiles=hist_t,
            history_projections=tuple(Pin(p, geometry_type="pinhole") for p in PH),
            front_camera_tile=front_t,
            front_projection=Pin(PF, geometry_type="pinhole"),
            mode="infer",
        )
        t1 = time.time()
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and dev.type == "cuda"),
        ):
            out = self.model(tiles_t, map_t, torch.zeros(1, 896, device=dev), ego_t, **kwargs)
        controls = (out[0] if isinstance(out, tuple) else out).float().reshape(OUT_STEPS, 2).cpu().numpy()
        t2 = time.time()
        rollout = integrate_controls(finite_controls(controls), v0)
        waypoints = rollout[4::5][:12]  # t = 0.5, 1.0, ..., 6.0 s
        self._plan = (now.t_s, now.ego2world.copy(), waypoints.copy())
        self.calls += 1
        self.last = {
            "t_us": int(obs.timestamp_us),
            "v0_history": round(v0, 3),
            "speed_canbus": round(float(obs.speed), 3),
            "views": len(VIEW_ORDER) - len(blank),
            "history_gaps_s": gaps,
            "history_repeated": sum(g is None for g in gaps),
            "frames_held": len(self.frames),
            "ego_rows_zero": int((np.abs(ego).sum(1) == 0).sum()),
            "ego_rows_filled": filled,
            "map_valid": map_valid,
            "route_valid": route_valid,
            "destination": dest,
            "pack_s": round(t1 - t0, 3),
            "model_s": round(t2 - t1, 3),
            "accel_0": round(float(controls[0, 0]), 3),
            "kappa_0": round(float(controls[0, 1]), 4),
            "end_6s": [round(float(v), 2) for v in waypoints[-1, :2]],
        }
        if self.record_dir:
            self._write_record(obs, controls, rollout, ego)
        return Trajectory(
            waypoints,
            dt=self.waypoint_dt,
            extras={"autoe2e": {k: v for k, v in self.last.items() if k != "history_gaps_s"}},
        )

    def _write_record(self, obs, controls, rollout, ego) -> None:
        import json

        if self._record is None:
            self._record = open(os.path.join(self.record_dir, f"calls_{os.getpid()}.jsonl"), "a")
        self._record.write(
            json.dumps(
                {
                    **self.last,
                    "controls": np.round(controls, 4).tolist(),
                    "xy_10hz": np.round(rollout[:, :2], 3).tolist(),
                    "ego": np.round(ego, 4).tolist(),
                }
            )
            + "\n"
        )
        self._record.flush()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "checkpoint": portable_checkpoint(self.checkpoint),
            "checkpoint_sha256": self._ckpt_sha,
            "config": getattr(self, "config", None),
            "ref_height_m": self.ref_height,
            "route": self.use_route,
            "bf16": self.bf16,
            "replan_s": self.replan_s,
            "ego_history": self.ego_history_mode,
            "calls": self.calls,
            "map_calls": self.map_calls,
            "route_calls": self.route_calls,
            "last_inference": self.last,
        }


POLICY = AutoE2EPolicy
