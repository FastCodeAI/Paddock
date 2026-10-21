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
  converts to its caffe BGR input itself.
- BEVFormer V2 T8: the current frame plus seven frames at -3.5 .. -0.5 s, oldest first,
  each with a projection that maps the *current* reference frame into that frame's
  cameras (ego motion folded in). Where the past does not reach back far enough (a
  closed-loop route start) the oldest frame held is repeated with its own pose.
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
- Navigation: no semantic map (zeros, ``map_valid`` False). The route raster (corridor
  1.75 m either side of the route polyline, rear-clipped at -10 m, plus a 2 m destination
  disk) is drawn from ``obs.route_xy`` when the benchmark sends one and
  ``AUTOE2E_ROUTE`` is not "0"; otherwise zeros with ``route_valid`` False, upstream's
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
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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
#: A history frame farther than this from its target time is not that frame.
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

    Steady driving is the first measured row -- speed, acceleration, yaw rate and curvature
    all held at whatever the earliest real measurement reads, not speed alone with the other
    three assumed zero: a scenario that starts mid-turn or mid-brake should read as having
    been mid-turn or mid-brake for the whole unseen past, the same way it reads as having
    been at the first measured speed rather than standing still. With fewer than two poses
    nothing is measured yet and every row takes `(speed, 0, 0, 0)`, the simulator's speed with
    no acceleration or turn assumed, since no derivative exists to hold instead.

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
    first_row = out[~lead][0].copy() if (~lead).any() else np.array([speed, 0.0, 0.0, 0.0])
    out[lead] = first_row
    return out, int(lead.sum())


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


#: Stage name -> submodule attribute on `AutoE2E.Reactive_E2E` whose forward
#: output is worth a snapshot for a from-scratch walkthrough. `TemporalFusedBEV`
#: is not here: `_fuse_temporal_camera_bevs` is a plain method, not a submodule
#: call, so it never reaches `forward` and is monkeypatched instead, below.
_DEBUG_STAGES = (
    "Backbone",
    "FeatureFusion",
    "NavigationEncoder",
    "MapBEVFusion",
    "FusedFeaturePooling",
    "TemporalMemory",
    "TrajectoryPlanner",
)


def _install_debug_hooks(re2e, state: dict) -> None:
    """Make every forward call of `re2e`'s key stages land in `state["stages"]`.

    One list per stage name, appended to on every call within the current
    `infer()` -- the caller clears `state["stages"]` before each model() call
    so a dump reflects exactly that call's stages, not a running history.
    `FeatureFusion` is called eight times per call (seven history frames, oldest
    first, then the current frame last -- `_fuse_temporal_camera_bevs`), so its
    list is length 8; every other stage here is called once.
    """
    import torch

    state["stages"] = {}

    def stash(name):
        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                value: Any = output.detach().to("cpu")
            elif isinstance(output, (tuple, list)):
                value = tuple(o.detach().to("cpu") if torch.is_tensor(o) else o for o in output)
            else:
                value = output
            state["stages"].setdefault(name, []).append(value)

        return hook

    for name in _DEBUG_STAGES:
        module = getattr(re2e, name, None)
        if module is not None:
            module.register_forward_hook(stash(name))

    original_fuse = re2e._fuse_temporal_camera_bevs

    def fuse_and_stash(current_image_bev, history_bevs):
        fused = original_fuse(current_image_bev, history_bevs)
        state["stages"].setdefault("TemporalFusedBEV", []).append(fused.detach().to("cpu"))
        return fused

    re2e._fuse_temporal_camera_bevs = fuse_and_stash


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
            else:
                with Image.open(self.paths[cam]) as im:
                    self._sizes[cam] = im.size
        return self._sizes[cam]

    def tiles(self) -> np.ndarray:
        """uint8 [6, 512, 512, 3], computed once (also the history copy of the front).

        Frames read from disk are cached by their six paths across episodes: consecutive NAVSIM
        scenes of a log share seven of their eight camera keyframes, and the pixels are the same.
        """
        if self._tiles is None:
            key = tuple(self.paths.get(c) for c in VIEW_ORDER) if not self.images else None
            hit = _TILE_CACHE.get(key) if key is not None else None
            if hit is not None:
                _TILE_CACHE.move_to_end(key)
                self._sizes, self._tiles = dict(hit[0]), hit[1]
                return self._tiles
            for c in VIEW_ORDER:
                self.size(c)
            self._tiles = np.stack([pack_tile(self.rgb(c), TILE) for c in VIEW_ORDER])
            self.images = {}  # pixels no longer needed once packed
            if key is not None:
                _TILE_CACHE[key] = (dict(self._sizes), self._tiles)
                while len(_TILE_CACHE) > TILE_CACHE_FRAMES:
                    _TILE_CACHE.popitem(last=False)
        return self._tiles


def _nuscenes_lead_in_frames(
    root: str, version: str, now_t_us: int, camera2image: dict, camera2ego: dict, max_frames: int = 7
) -> list[_Frame]:
    """Real nuScenes frames from before `now_t_us` in the same scene, oldest first.

    EXPERIMENTAL, opt-in via `AUTOE2E_NUSCENES_ROOT`: closes the cold-start gap `_history()`
    otherwise fills by repeating the single available frame, by reading the real frames the
    scenario's own `start-frame` left unused -- most official scenes have several seconds of
    real lead-in on disk (``start-frame: 8`` on nuScenes' 2 Hz keyframes is 4 s), never sent
    over the wire because the vendor engine only ever starts a scenario at `start-frame`.

    Walks `sample.json`'s own `prev` chain backward from the sample matching `now_t_us`, up
    to `max_frames` steps or the scene's own start, whichever comes first -- a scenario with
    less lead-in than the model's 3.5 s history window gets fewer real frames, and the rest
    still fall back to `_history()`'s held-oldest frame rather than fabricated ones.

    Calibration is reused from `now`, not re-read per frame: nuScenes calibration is constant
    for a whole scene (the vendor engine's own `get_camera_calibration` always reads the
    scene's first sample regardless of frame index), so there is nothing left to look up.

    Loads `sample_data.json` and `ego_pose.json` whole (reduced to the handful of tokens this
    needs) once per episode -- acceptable here as a one-time cost while this is being
    evaluated, not yet optimised for a full sweep's worth of repeated episodes.
    """
    import json
    from pathlib import Path

    from paddock.adapters.benchmarks.nuscenes.loader import pose_matrix

    meta = Path(root) / version
    samples = {s["token"]: s for s in json.loads((meta / "sample.json").read_text())}
    match = min(samples.values(), key=lambda s: abs(s["timestamp"] - now_t_us), default=None)
    if match is None or abs(match["timestamp"] - now_t_us) > 1000:  # 1 ms
        return []

    chain = []
    token = match["prev"]
    while token and len(chain) < max_frames:
        chain.append(token)
        token = samples[token]["prev"]
    if not chain:
        return []
    chain.reverse()  # oldest first

    wanted = frozenset(chain)
    channels = (*VIEW_ORDER, "LIDAR_TOP")
    by_sample: dict[str, dict[str, dict]] = {}
    for d in json.loads((meta / "sample_data.json").read_text()):
        if d.get("sample_token") not in wanted or not d.get("is_key_frame"):
            continue
        filename = d.get("filename", "")
        channel = next((c for c in channels if filename.startswith(f"samples/{c}/")), None)
        if channel is not None:
            by_sample.setdefault(d["sample_token"], {})[channel] = d

    pose_tokens = frozenset(
        rec["LIDAR_TOP"]["ego_pose_token"] for rec in by_sample.values() if "LIDAR_TOP" in rec
    )
    ego_poses = {
        p["token"]: pose_matrix(p["translation"], p["rotation"])
        for p in json.loads((meta / "ego_pose.json").read_text())
        if p["token"] in pose_tokens
    }

    frames = []
    for token in chain:
        rec = by_sample.get(token)
        if rec is None or "LIDAR_TOP" not in rec or any(c not in rec for c in VIEW_ORDER):
            continue  # this sample is missing a view or lidar; skip rather than fabricate
        ego2world = ego_poses.get(rec["LIDAR_TOP"]["ego_pose_token"])
        if ego2world is None:
            continue
        images = {
            c: np.asarray(Image.open(Path(root) / rec[c]["filename"]).convert("RGB"), dtype=np.uint8)
            for c in VIEW_ORDER
        }
        frames.append(
            _Frame(
                t_s=samples[token]["timestamp"] / 1e6,
                ego2world=ego2world,
                camera2image=camera2image,
                camera2ego=camera2ego,
                images=images,
            )
        )
    return frames


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
        #: Per-call dump of every forward stage (backbone, BEV fusion, temporal
        #: fusion, navigation, planner) plus the raw packed inputs, one file per
        #: call -- for a from-scratch, frame-by-frame walkthrough of what the
        #: model saw and did. Off by default and zero-cost when unset.
        self.debug_dump_dir = os.environ.get("AUTOE2E_DEBUG_DUMP_DIR") or None
        self._debug_state: dict[str, Any] = {}
        #: EXPERIMENTAL: real nuScenes frames from before the scenario's own `start-frame`,
        #: primed into camera history once per episode instead of repeating the first frame.
        #: Off by default; see `_nuscenes_lead_in_frames`.
        self.nuscenes_root = os.environ.get("AUTOE2E_NUSCENES_ROOT") or None
        self.nuscenes_version = os.environ.get("AUTOE2E_NUSCENES_VERSION", "v1.0-trainval")
        self._primed_lead_in = False
        #: Seconds between forward passes. 0 = every call. Closed loop calls at 10 Hz so the ego
        #: history is real 10 Hz; the plan is refreshed every `replan_s` and held in between,
        #: re-expressed in the current ego frame (what the Bench2Drive client does with a held plan).
        self.replan_s = float(os.environ.get("AUTOE2E_REPLAN_S", "0") or 0)
        self._plan: tuple[float, np.ndarray, np.ndarray] | None = None
        self.model = None
        self.device = None
        self.frames: list[_Frame] = []
        self.last: dict[str, Any] = {}
        self.calls = 0
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
        if self.debug_dump_dir:
            os.makedirs(self.debug_dump_dir, exist_ok=True)
            _install_debug_hooks(self.model.Reactive_E2E, self._debug_state)
        logger.info("AutoE2E loaded %s (%s) on %s", self.checkpoint, self.config, self.device)

    def reset(self) -> None:
        self.frames = []
        self.last = {}
        self._plan = None
        self._primed_lead_in = False
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

    def _history(self, now: _Frame) -> tuple[list[_Frame], list[float]]:
        """Seven frames at -3.5 .. -0.5 s, oldest first, from the frames that carry all six views."""
        full = [f for f in self.frames if f.full and f is not now] or [now]
        times = np.array([f.t_s for f in full])
        picked, gaps = [], []
        for off in HISTORY_OFFSETS_S:
            target = now.t_s + off
            i = int(np.argmin(np.abs(times - target)))
            if abs(times[i] - target) <= MATCH_TOLERANCE_S:
                picked.append(full[i])
                gaps.append(round(float(times[i] - target), 3))
            else:
                # Not that far back: the oldest six-view frame held, with its own pose (route / log start).
                picked.append(full[0])
                gaps.append(None)
        return picked, gaps

    # ------------------------------------------------------------------ geometry
    def _ref2ego(self) -> np.ndarray:
        m = np.eye(4)
        m[2, 3] = self.ref_height
        return m

    def _projections(
        self, frame: _Frame, ref_now2world: np.ndarray | None, size: int, views=VIEW_ORDER
    ) -> np.ndarray:
        """[V, 3, 4]: current reference frame -> pixels of `frame`'s cameras at `size` x `size`."""
        ref2ego = self._ref2ego()
        ref_frame2world = frame.ego2world @ ref2ego
        hist_from_now = np.eye(4) if ref_now2world is None else np.linalg.inv(ref_frame2world) @ ref_now2world
        out = []
        for c in views:
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
        if not now.full:
            missing = [c for c in VIEW_ORDER if c not in now.views]
            raise RuntimeError(
                f"AutoE2E plans from all six views; the current observation has none for {missing}"
            )
        if self.nuscenes_root and not self._primed_lead_in and len(self.frames) == 1:
            lead_in = _nuscenes_lead_in_frames(
                self.nuscenes_root,
                self.nuscenes_version,
                int(obs.timestamp_us),
                now.camera2image,
                now.camera2ego,
            )
            self.frames = [*lead_in, *self.frames]
            self._primed_lead_in = True
            logger.info("AutoE2E: primed %d real nuScenes frame(s) before episode start", len(lead_in))
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
        history, gaps = self._history(now)
        dev = self.device

        def normalise(arr: np.ndarray) -> torch.Tensor:  # uint8 [..., H, W, 3] -> float [..., 3, H, W]
            x = torch.from_numpy(arr).to(dev, non_blocking=True).float().div_(255.0)
            x = (x - torch.from_numpy(IMAGENET_MEAN).to(dev)) / torch.from_numpy(IMAGENET_STD).to(dev)
            return x.movedim(-1, -3).contiguous()

        front_1024 = pack_tile(now.rgb("CAM_FRONT"), FRONT_TILE)
        current = now.tiles()
        front_t = normalise(front_1024)[None]  # [1, 3, 1024, 1024]
        tiles_t = normalise(current)[None]  # [1, 6, 3, 512, 512]
        tiles_t[:, 0] = F.interpolate(
            front_t, size=(TILE, TILE), mode="bilinear", align_corners=False, antialias=True
        )
        hist_t = normalise(np.stack([f.tiles() for f in history]))[None]  # [1, 7, 6, 3, 512, 512]

        ref_now2world = now.ego2world @ self._ref2ego()
        P = torch.from_numpy(self._projections(now, None, TILE))[None].to(dev)
        PF = torch.from_numpy(self._projections(now, None, FRONT_TILE, views=("CAM_FRONT",)))[None].to(dev)
        PH = [torch.from_numpy(self._projections(f, ref_now2world, TILE))[None].to(dev) for f in history]

        ego, filled = self._ego_history(now, float(obs.speed))
        v0 = float(ego[-1, 0])
        ego_in = ego.copy()
        ego_in[-1, 1] = 0.0  # KITScenes stage: newest acceleration masked (dataset_policy.py)
        ego_t = torch.from_numpy(ego_in.reshape(1, -1).astype(np.float32)).to(dev)

        route, corridor, dest = route_raster(obs.route_xy if self.use_route else None)
        route_valid = bool(corridor)
        route_t = (
            torch.from_numpy(route)[None].to(dev)
            if route_valid
            else torch.zeros(1, 2, NAV_H, NAV_W, device=dev)
        )
        Pin = self._Pinhole
        kwargs = dict(
            route_mask=route_t,
            map_valid=torch.zeros(1, dtype=torch.bool, device=dev),
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
        if self.debug_dump_dir:
            self._debug_state["stages"] = {}
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and dev.type == "cuda"),
        ):
            out = self.model(
                tiles_t,
                torch.zeros(1, 14, NAV_H, NAV_W, device=dev),
                torch.zeros(1, 896, device=dev),
                ego_t,
                **kwargs,
            )
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
            "history_gaps_s": gaps,
            "history_repeated": sum(g is None for g in gaps),
            "frames_held": len(self.frames),
            "ego_rows_zero": int((np.abs(ego).sum(1) == 0).sum()),
            "ego_rows_filled": filled,
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
        if self.debug_dump_dir:
            self._dump_debug_step(
                front_1024=front_1024,
                current=current,
                history=history,
                gaps=gaps,
                ego=ego,
                route=route,
                route_valid=route_valid,
                controls=controls,
                rollout=rollout,
                waypoints=waypoints,
                v0=v0,
            )
        return Trajectory(
            waypoints,
            dt=self.waypoint_dt,
            extras={"autoe2e": {k: v for k, v in self.last.items() if k != "history_gaps_s"}},
        )

    def _dump_debug_step(
        self, *, front_1024, current, history, gaps, ego, route, route_valid, controls, rollout, waypoints, v0
    ) -> None:
        """One `step_<call>.pt` per `infer()` call: raw packed inputs + every stage's output.

        `self.calls` is already incremented by the time this runs, so `self.calls - 1`
        is this call's 0-based index -- `step_0000.pt` is the episode's first call.
        """
        import torch

        payload = {
            "call": self.calls - 1,
            "t_us": int(self.last["t_us"]),
            "inputs": {
                "front_1024": front_1024,  # uint8 [1024,1024,3]
                "current_tiles": current,  # uint8 [6,512,512,3], VIEW_ORDER
                "history_tiles": np.stack([f.tiles() for f in history]),  # uint8 [7,6,512,512,3]
                "history_gaps_s": gaps,  # None where a frame is held-oldest, not real
                "ego_rows": ego,  # float64 [64,4]: speed, accel, yaw_rate, curvature
                "v0": v0,
                "route_raster": route,  # float32 [2,450,300]: corridor, destination
                "route_valid": route_valid,
            },
            "stages": dict(self._debug_state.get("stages", {})),
            "output": {
                "controls": controls,  # float32 [64,2]: accel, curvature @ 10 Hz
                "rollout_xy_heading": rollout,  # float64 [64,3] @ 10 Hz
                "waypoints": waypoints,  # the <=12 @ 0.5 Hz; first 6 go on the wire
            },
        }
        torch.save(payload, os.path.join(self.debug_dump_dir, f"step_{self.calls - 1:04d}.pt"))

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
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self._ckpt_sha,
            "config": getattr(self, "config", None),
            "ref_height_m": self.ref_height,
            "route": self.use_route,
            "bf16": self.bf16,
            "replan_s": self.replan_s,
            "calls": self.calls,
            "last_inference": self.last,
        }


POLICY = AutoE2EPolicy
