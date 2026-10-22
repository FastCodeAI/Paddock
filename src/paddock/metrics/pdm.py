"""NAVSIM v1's PDM score, its terms on every benchmark that can measure them, and a scorer that measures them.

    pdm_style_score = NC x DAC x (5*TTC + 5*EP + 2*C) / 12

NAVSIM's weights, unchanged; paddock evaluates someone else's published formula
rather than inventing one (the argument is in
`adapters/benchmarks/comma2k19/score.py`, which first computed it). Each
benchmark's `benchmark.yaml` says, under `pdm:`, which of its metrics fills
which term (`SLOTS`), so the regression gate can lay every benchmark out in the
same five rows. What a term is *made of* differs between benchmarks; the slot
says which question a metric answers, not that two benchmarks answer it alike.
A missing term is never a pass: `composite` is NaN unless all five are finite.

Two layers live here.

**Open loop, from a plan and its recorded future** (`plan_terms`): EP and C.
**Wherever the other road users' motion is known** (`score_plan`): all five,
following NAVSIM's scorer (`pdm_scorer.py`, `pdm_scorer_utils.py`, nuPlan's
`collision_utils` / `idm/utils`) term by term. What is scored is the *plan, at
each decision*, against what the other road users actually did over the plan's
horizon: the log on an open-loop benchmark, the scripted or replayed actors on a
closed-loop one. That is NAVSIM's own reading (it never drives the plan either),
and it means the same thing open and closed loop. What the car then did in
closed loop is the benchmark's own score's business (NCAP score, driving score).

* **NC, no at-fault collision**: nuPlan's collision types, ported. A collision
  while the ego is stopped, or with an agent behind it, is not the ego's fault and
  that agent is ignored from then on; one with a stopped agent or by the ego's
  front bumper is. A lateral collision is at fault only when the ego is off the
  drivable area -- NAVSIM also counts it when the ego straddles two lanes, and no
  benchmark here has lane polygons. 0 for an agent, 0.5 for a static object.
* **DAC**: every footprint corner on the drivable area at every step.
* **TTC**: the footprint carried on at its current velocity for 0, 0.3, 0.6 and
  0.9 s; a violation when it meets an agent ahead (within 30 degrees), or one not
  behind while the ego is off the drivable area (NAVSIM: also in an intersection),
  and the ego is moving.
* **EP, NAVSIM's rule on every benchmark**: progress along a centreline, divided
  by the best progress among the plan and the reference paths, each counted only
  where its own NC x DAC holds (NAVSIM's references are PDM-Closed's proposals;
  here, the human's path); 1 when that best covers under 5 m. On an open-loop
  log the human is the reference and is safe by construction, so EP is the
  plan's progress along the human's path over the human's, at most 1
  (`ego_progress`). In closed loop the human's logged path can drive through an
  inserted actor, and then it does not count.
* **C**: nuPlan's comfort bounds on the plan alone (`paddock.metrics.comfort`),
  a plan that stands still passing as a stopped car does in NAVSIM (`plan_comfort`).
  NAVSIM v1 applies them to the tracked plan; v2 also compares consecutive plans.
* **Horizon**: the plan's own (neuroNCAP's six waypoints are 3 s), on a 0.1 s grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paddock.metrics.comfort import BOUNDS, comfort

__all__ = [
    "LABELS",
    "PDM_KEYS",
    "SCORE",
    "SLOTS",
    "WEIGHTS",
    "DrivableArea",
    "Track",
    "composite",
    "ego_progress",
    "plan_terms",
    "score_plan",
]

#: The five terms, in the order NAVSIM writes the formula.
SLOTS = ("nc", "dac", "ttc", "ep", "comfort")
#: The `pdm:` key naming a benchmark's own composite of the five, when it has one.
SCORE = "score"
#: How the gate labels each slot's row.
LABELS = {"nc": "NC", "dac": "DAC", "ttc": "TTC", "ep": "EP", "comfort": "C", SCORE: "PDM"}

#: NAVSIM v1's weights for the additive half of PDMS. Restated rather than
#: imported -- nuplan-devkit is not a paddock dependency -- so they are cited at
#: the strength of published constants, the same standing as
#: `metrics.comfort.BOUNDS`.
WEIGHTS = {"ttc": 5.0, "ep": 5.0, "comfort": 2.0}

#: Every key `score_plan` returns.
PDM_KEYS: tuple[str, ...] = ("pdm_nc", "pdm_dac", "pdm_ttc", "pdm_ep", "pdm_comfort", "pdm_style_score")
#: The grid every plan is checked on, in seconds.
STEP_S = 0.1
#: NAVSIM's TTC projections: offsets 0, 3, 6, 9 steps of 0.1 s.
TTC_OFFSETS_S = (0.0, 0.3, 0.6, 0.9)
#: `PDMScorerConfig.stopped_speed_threshold`: below it the ego has no TTC.
TTC_STOPPED_MPS = 5e-3
#: `get_collision_type`'s and `is_track_stopped`'s threshold.
COLLISION_STOPPED_MPS = 5e-2
#: `PDMScorerConfig.progress_distance_threshold`.
PROGRESS_THRESHOLD_M = 5.0
#: `is_agent_ahead` / `is_agent_behind` angle tolerances, degrees.
AHEAD_DEG, BEHIND_DEG = 30.0, 150.0
#: A path segment shorter than this keeps the previous heading, so a plan that
#: stands still does not spin (the AlpaSim bridge's rule, `derive_headings`).
HEADING_MIN_STEP_M = 0.05


@dataclass(frozen=True)
class Ego:
    """The ego's footprint: length and width in metres, its centre ahead of the rear axle."""

    length: float
    width: float
    rear_axle_to_centre: float


#: The nuScenes car (Renault Zoe) as neuroNCAP's collision check has it
#: (`neuro_ncap/evaluation/collision.py`): 4.018 x 1.85 m, centre 0.5 m ahead.
NUSCENES_EGO = Ego(4.018, 1.85, 0.5)


@dataclass
class Track:
    """One other road user's logged motion: its centre and heading at known times.

    `agent` is False for a static object (barrier, cone, debris): NAVSIM scores a
    collision with one 0.5, not 0, and it is always "stopped".
    """

    times_us: np.ndarray  # (T,)
    xy: np.ndarray  # (T, 2) world, box centre
    yaw: np.ndarray  # (T,)
    length: float
    width: float
    agent: bool = True
    name: str = ""

    def at(self, times_us: np.ndarray):
        """Centre (n, 2), yaw (n,), speed (n,) and whether it exists (n,) at `times_us`.

        Linear between logged poses, shortest way round for the heading, as
        neuroNCAP's `interpolate_actor_trajectory` and gallops' grader do. Outside
        its logged span a track does not exist; a single pose exists only then.
        """
        times_us = np.asarray(times_us, dtype=float)
        t = np.asarray(self.times_us, dtype=float)
        n = len(times_us)
        active = (t[0] <= times_us) & (times_us <= t[-1])
        if len(t) < 2:
            xy = np.repeat(np.asarray(self.xy, dtype=float)[:1], n, axis=0)
            return xy, np.full(n, float(self.yaw[0])), np.zeros(n), active
        i = np.clip(np.searchsorted(t, times_us) - 1, 0, len(t) - 2)
        w = np.clip((times_us - t[i]) / (t[i + 1] - t[i]), 0.0, 1.0)
        xy = self.xy[i] * (1 - w)[:, None] + self.xy[i + 1] * w[:, None]
        dyaw = np.angle(np.exp(1j * (self.yaw[i + 1] - self.yaw[i])))
        yaw = self.yaw[i] + dyaw * w
        speed = np.linalg.norm(self.xy[i + 1] - self.xy[i], axis=1) / ((t[i + 1] - t[i]) / 1e6)
        return xy, yaw, speed, active


@dataclass
class DrivableArea:
    """A boolean raster of the drivable surface, in world coordinates.

    The format gallops' `expert/drivable.py` writes (`mask`, `origin`, `cell`), so
    one build serves both repos. A point outside the raster is not drivable: the
    raster covers everything a plan from the logged path can reach.
    """

    mask: np.ndarray  # (rows, cols), True where drivable
    origin: np.ndarray  # world (x, y) of cell [0, 0]
    cell: float
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DrivableArea:
        raw = np.load(path)
        meta = {k: str(raw[k]) for k in raw.files if k not in ("mask", "origin", "cell")}
        return cls(mask=raw["mask"], origin=raw["origin"], cell=float(raw["cell"]), meta=meta)

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        index = np.floor((flat - self.origin) / self.cell).astype(int)
        rows, cols = self.mask.shape
        inside = (index[:, 1] >= 0) & (index[:, 1] < rows) & (index[:, 0] >= 0) & (index[:, 0] < cols)
        out = np.zeros(len(flat), dtype=bool)
        out[inside] = self.mask[index[inside, 1], index[inside, 0]]
        return out.reshape(shape)


@dataclass
class States:
    """A path in the world on the `STEP_S` grid: rear-axle position, heading, speed, time."""

    taus: np.ndarray  # (K,) seconds from the decision
    xy: np.ndarray  # (K, 2)
    yaw: np.ndarray  # (K,)
    speed: np.ndarray  # (K,)


def derive_headings(xy: np.ndarray) -> np.ndarray:
    """Each waypoint faces along the segment that reaches it, from the origin.

    A segment under `HEADING_MIN_STEP_M` keeps the previous heading; the first
    falls back to 0, the ego's own heading in its frame.
    """
    pts = np.vstack([np.zeros((1, 2)), np.asarray(xy, dtype=float)[:, :2]])
    out = np.zeros(len(pts) - 1)
    previous = 0.0
    for k in range(1, len(pts)):
        step = pts[k] - pts[k - 1]
        if np.hypot(*step) >= HEADING_MIN_STEP_M:
            previous = math.atan2(step[1], step[0])
        out[k - 1] = previous
    return out


def _pose(ego2world: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    m = np.asarray(ego2world, dtype=float)
    rotation, origin = m[:2, :2], m[:2, -1]
    return rotation, origin, math.atan2(rotation[1, 0], rotation[0, 0])


def plan_states(plan: np.ndarray, dt: float, ego2world: np.ndarray, speed0: float | None = None) -> States:
    """An ego-frame plan (N, 2|3), waypoint i at (i+1) dt, as world `States` on the 0.1 s grid.

    x forward, y left, from the rear axle at t = 0 (paddock's `Trajectory`). The
    heading is the plan's third column when it has one, else derived from the path.
    The speed at t = 0 is `speed0` (the measured one) when given, else the plan's
    first step's; after that, the distance covered in each 0.1 s step.
    """
    plan = np.asarray(plan, dtype=float)
    heading = plan[:, 2] if plan.shape[1] >= 3 else derive_headings(plan)
    stamps = np.arange(len(plan) + 1) * dt
    x = np.r_[0.0, plan[:, 0]]
    y = np.r_[0.0, plan[:, 1]]
    h = np.unwrap(np.r_[0.0, heading])
    taus = np.round(np.arange(0.0, stamps[-1] + 1e-9, STEP_S), 6)
    local = np.stack([np.interp(taus, stamps, x), np.interp(taus, stamps, y)], axis=1)
    rotation, origin, yaw0 = _pose(ego2world)
    xy = origin + local @ rotation.T
    yaw = yaw0 + np.interp(taus, stamps, h)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / STEP_S
    first = float(speed0) if speed0 is not None else (float(step[0]) if len(step) else 0.0)
    return States(taus, xy, yaw, np.r_[first, step])


def world_states(
    times_s: np.ndarray, xy: np.ndarray, horizon_s: float, yaw: np.ndarray | None = None
) -> States:
    """A world path given at its own times (seconds from the decision), on the 0.1 s grid.

    For a reference path such as the human's log. Headings are interpolated when
    given; else each given point faces along the segment that reaches it, as
    `derive_headings` does for a plan, and the first faces the second.
    """
    times_s = np.asarray(times_s, dtype=float)
    xy = np.asarray(xy, dtype=float)[:, :2]
    if yaw is None:
        yaw = np.zeros(len(xy))
        previous = 0.0
        for k in range(len(xy)):
            j = max(k, 1)
            if j < len(xy):
                step = xy[j] - xy[j - 1]
                if np.hypot(*step) >= HEADING_MIN_STEP_M:
                    previous = math.atan2(step[1], step[0])
            yaw[k] = previous
    taus = np.round(np.arange(0.0, horizon_s + 1e-9, STEP_S), 6)
    grid = np.stack([np.interp(taus, times_s, xy[:, 0]), np.interp(taus, times_s, xy[:, 1])], axis=1)
    heading = np.interp(taus, times_s, np.unwrap(np.asarray(yaw, dtype=float)))
    step = np.linalg.norm(np.diff(grid, axis=0), axis=1) / STEP_S
    return States(taus, grid, heading, np.r_[step[0] if len(step) else 0.0, step])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def box_corners(centre: np.ndarray, yaw: np.ndarray, length, width) -> np.ndarray:
    """(..., 2) centres and (...,) yaws -> (..., 4, 2): front-left, front-right, rear-right, rear-left."""
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    left = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1)
    half_l = np.asarray(length, dtype=float)[..., None, None] / 2
    half_w = np.asarray(width, dtype=float)[..., None, None] / 2
    signs = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)], dtype=float)
    return (
        centre[..., None, :]
        + signs[:, :1] * fwd[..., None, :] * half_l
        + signs[:, 1:] * left[..., None, :] * half_w
    )


def ego_boxes(states: States, ego: Ego) -> np.ndarray:
    """(K, 4, 2) footprints of a path of rear-axle states."""
    fwd = np.stack([np.cos(states.yaw), np.sin(states.yaw)], axis=-1)
    centre = states.xy + ego.rear_axle_to_centre * fwd
    return box_corners(centre, states.yaw, ego.length, ego.width)


def convex_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Separating-axis test between broadcastable convex polygons (..., n, 2) and (..., m, 2).

    Touching counts as overlapping, as in shapely's `intersects`. A two-point
    "polygon" is a segment, which is how the front bumper is tested.
    """
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    separated = np.zeros(shape, dtype=bool)
    for polygon in (a, b):
        k = polygon.shape[-2]
        for i in range(k if k > 2 else 1):
            edge = polygon[..., (i + 1) % k, :] - polygon[..., i, :]
            normal = np.stack([-edge[..., 1], edge[..., 0]], axis=-1)[..., None, :]
            pa = (a * normal).sum(-1)
            pb = (b * normal).sum(-1)
            separated |= (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    if a.shape[-2] == 2 or b.shape[-2] == 2:
        # A segment and a box can be separated along the segment's own direction
        # alone, which no edge normal above tests.
        seg = a if a.shape[-2] == 2 else b
        axis = (seg[..., 1, :] - seg[..., 0, :])[..., None, :]
        pa = (a * axis).sum(-1)
        pb = (b * axis).sum(-1)
        separated |= (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~separated


def relative_angle(xy: np.ndarray, yaw: np.ndarray, other_xy: np.ndarray) -> np.ndarray:
    """Angle between the ego's heading and the direction to another position, radians (nuPlan's)."""
    v = other_xy - xy
    norm = np.maximum(np.linalg.norm(v, axis=-1), 1e-9)
    cos = (np.cos(yaw) * v[..., 0] + np.sin(yaw) * v[..., 1]) / norm
    return np.arccos(np.clip(cos, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


@dataclass
class TrackGrid:
    """The tracks near a path, at every grid step plus each TTC offset, computed once.

    Arrays are (A, K, O, ...) for A tracks, K steps and O = `TTC_OFFSETS_S`;
    offset 0 is the step itself, which is what the collision check reads.
    """

    tracks: list[Track]
    boxes: np.ndarray  # (A, K, O, 4, 2)
    centres: np.ndarray  # (A, K, O, 2)
    speed: np.ndarray  # (A, K, O)
    active: np.ndarray  # (A, K, O)


#: How close a track must come to the path, at some step, to be considered at all:
#: half the ego's and a long vehicle's diagonals, 0.9 s at 40 m/s, and a margin.
NEAR_M = 3.0 + 7.0 + 36.0 + 5.0


def track_grid(tracks: Sequence[Track], states: States, t0_us: float) -> TrackGrid:
    k, o = len(states.taus), len(TTC_OFFSETS_S)
    times = t0_us + (states.taus[:, None] + np.asarray(TTC_OFFSETS_S)[None, :]).reshape(-1) * 1e6
    kept, boxes, centres, speeds, active = [], [], [], [], []
    for track in tracks:
        xy, yaw, speed, on = track.at(times)
        if not on.any():
            continue
        near = np.linalg.norm(xy.reshape(k, o, 2)[:, 0] - states.xy, axis=-1)
        if near.min() > NEAR_M:
            continue
        kept.append(track)
        boxes.append(box_corners(xy, yaw, track.length, track.width).reshape(k, o, 4, 2))
        centres.append(xy.reshape(k, o, 2))
        speeds.append(speed.reshape(k, o))
        active.append(on.reshape(k, o))
    if not kept:
        z = np.zeros((0, k, o))
        return TrackGrid([], np.zeros((0, k, o, 4, 2)), np.zeros((0, k, o, 2)), z, z.astype(bool))
    return TrackGrid(kept, np.stack(boxes), np.stack(centres), np.stack(speeds), np.stack(active))


def off_road(boxes: np.ndarray, area: DrivableArea | None) -> np.ndarray | None:
    """(K,) whether any footprint corner is off the drivable area, or None without a map."""
    if area is None:
        return None
    return ~area.contains(boxes).all(-1)


def no_at_fault_collision(
    states: States, boxes: np.ndarray, grid: TrackGrid, offroad: np.ndarray | None
) -> tuple[float, float | None]:
    """NC for one path, and the time of its first at-fault collision (s) or None.

    nuPlan's collision types, in order: ego stopped -> not at fault; the other
    stopped (or a static object) -> at fault; the other behind (> 150 deg) -> not
    at fault; the ego's front bumper on it -> at fault; else lateral, at fault only
    off the drivable area. A not-at-fault collision drops that track for the rest
    of the path, as NAVSIM's `collided_track_ids` does.
    """
    if not grid.tracks:
        return 1.0, None
    tboxes = grid.boxes[:, :, 0]  # (A, K, 4, 2)
    hit = convex_overlap(boxes[None], tboxes) & grid.active[:, :, 0]  # (A, K)
    if not hit.any():
        return 1.0, None
    ignored: set[int] = set()
    score, first = 1.0, None
    for step in range(len(states.taus)):
        for a in np.flatnonzero(hit[:, step]):
            if a in ignored:
                continue
            track = grid.tracks[a]
            if states.speed[step] <= COLLISION_STOPPED_MPS:
                fault = False
            elif not track.agent or grid.speed[a, step, 0] <= COLLISION_STOPPED_MPS:
                fault = True
            elif relative_angle(states.xy[step], states.yaw[step], grid.centres[a, step, 0]) > math.radians(
                BEHIND_DEG
            ):
                fault = False
            elif convex_overlap(boxes[step, :2], tboxes[a, step]):
                fault = True
            else:
                fault = bool(offroad is not None and offroad[step])
            if fault:
                score = min(score, 0.0 if track.agent else 0.5)
                first = float(states.taus[step]) if first is None else first
            else:
                ignored.add(a)
    return score, first


def time_to_collision(
    states: States, boxes: np.ndarray, grid: TrackGrid, offroad: np.ndarray | None
) -> float:
    """TTC for one path: 0 if the footprint, carried on at its velocity, meets an agent ahead."""
    if not grid.tracks:
        return 1.0
    fwd = np.stack([np.cos(states.yaw), np.sin(states.yaw)], axis=-1)
    shift = (fwd * states.speed[:, None])[:, None, :] * np.asarray(TTC_OFFSETS_S)[None, :, None]  # (K, O, 2)
    moved = boxes[:, None] + shift[:, :, None, :]  # (K, O, 4, 2)
    hit = convex_overlap(moved[None], grid.boxes) & grid.active  # (A, K, O)
    hit &= (states.speed >= TTC_STOPPED_MPS)[None, :, None]
    if not hit.any():
        return 1.0
    ignored: set[int] = set()
    for step in range(len(states.taus)):
        for offset in range(len(TTC_OFFSETS_S)):
            for a in np.flatnonzero(hit[:, step, offset]):
                if a in ignored:
                    continue
                angle = relative_angle(states.xy[step], states.yaw[step], grid.centres[a, step, offset])
                ahead = angle < math.radians(AHEAD_DEG)
                behind = angle > math.radians(BEHIND_DEG)
                if ahead or (offroad is not None and offroad[step] and not behind):
                    return 0.0
                ignored.add(a)
    return 1.0


def arc_length(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Where each point projects onto the polyline, as distance along it from its start."""
    polyline = np.asarray(polyline, dtype=float)[:, :2]
    points = np.atleast_2d(np.asarray(points, dtype=float))
    seg = np.diff(polyline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    start = np.r_[0.0, np.cumsum(seg_len)][:-1]
    along = np.clip(
        ((points[:, None] - polyline[None, :-1]) * seg[None]).sum(-1) / np.maximum(seg_len**2, 1e-9), 0.0, 1.0
    )
    feet = polyline[None, :-1] + along[..., None] * seg[None]
    nearest = np.linalg.norm(feet - points[:, None], axis=-1).argmin(1)
    rows = np.arange(len(points))
    return start[nearest] + along[rows, nearest] * seg_len[nearest]


def progress(states: States, centreline: np.ndarray) -> float:
    """Metres gained along the centreline from the path's first point to its last, at least 0."""
    s = arc_length(states.xy[[0, -1]], centreline)
    return max(0.0, float(s[1] - s[0]))


def composite(nc: float, dac: float, ttc: float, ep: float, comfort: float) -> float:
    """NC x DAC x the weighted mean of TTC, EP and C; NaN if any term is not finite."""
    terms = (nc, dac, ttc, ep, comfort)
    if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in terms):
        return float("nan")
    weighted = (WEIGHTS["ttc"] * ttc + WEIGHTS["ep"] * ep + WEIGHTS["comfort"] * comfort) / sum(
        WEIGHTS.values()
    )
    return float(nc * dac * weighted)


def normalised_progress(own: float, own_mult: float, references: Sequence[float]) -> float:
    """NAVSIM's EP: own progress x own NC x DAC over the best such value, 1 under 5 m.

    `references` are the other paths' progress, each already multiplied by its own
    NC x DAC, so a reference that crashes or leaves the road counts for nothing.
    """
    raw = own * own_mult
    best = max([raw, *references])
    if best > PROGRESS_THRESHOLD_M:
        return float(raw / best)
    return 1.0 if own_mult > 0 else 0.0


def ego_progress(pred_xy, gt_xy) -> float:
    """EP for an open-loop plan against the human's recorded future, by NAVSIM's rule.

    The human's path, from the ego's origin, is the centreline and the one
    reference; a logged human is safe by construction, and the plan's own NC and
    DAC are the benchmark's other terms. So EP is the plan's progress along the
    human's path over the human's, at most 1 (overshooting earns nothing, since
    the projection stops at the path's end), and 1 when neither covers 5 m: a
    human who barely moved leaves nothing to fall short of. Whether sitting
    still was *right* is the collision terms' business, not progress's.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    if not len(pred) or not len(gt) or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        return float("nan")
    path = np.vstack([np.zeros((1, 2)), gt])
    human = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if human <= 0.0:
        return 1.0
    plan = max(0.0, float(arc_length(pred[-1:], path)[0]))
    return normalised_progress(plan, 1.0, [human])


def plan_comfort(plan_xy, dt: float) -> float:
    """C for one plan: every nuPlan bound `metrics.comfort` can measure is met.

    A yaw or lateral channel that is unmeasurable only because the plan stood
    still passes, as in NAVSIM, where a stopped car is comfortable: a plan that
    commands no motion commands no turn. A plan too short to have a jerk, or not
    finite, is NaN, never a pass.
    """
    c = comfort(plan_xy, dt)
    flags = []
    for name in BOUNDS:
        flag = c[f"comfort_{name}_within_bounds"]
        if (
            math.isnan(flag)
            and name in ("yaw_rate", "lat_accel")
            and math.isfinite(c["comfort_lon_accel_max"])
        ):
            flag = 1.0
        flags.append(flag)
    if any(math.isnan(f) for f in flags):
        return float("nan")
    return float(all(f == 1.0 for f in flags))


def plan_terms(pred_xy, gt_xy, plan_xy, plan_dt: float) -> dict[str, float]:
    """`pdm_ep` and `pdm_comfort` for one open-loop sample.

    `pred_xy` / `gt_xy` are the plan and the recorded future as the benchmark
    already pairs them for its displacement errors, truncated to its horizon, so
    progress covers exactly what `ade` covers. Comfort reads the model's whole
    plan on its own spacing (`plan_xy`, `plan_dt`): a plan that jerks at 5 s is
    uncomfortable whether or not the benchmark scores 5 s. A plan too short to
    have a jerk is NaN, never a pass.
    """
    pred = np.asarray(pred_xy, dtype=np.float64)[:, :2]
    gt = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    n = min(len(pred), len(gt))
    ep = ego_progress(pred[:n], gt[:n]) if n else float("nan")
    return {
        "pdm_ep": ep,
        "pdm_comfort": plan_comfort(plan_xy, plan_dt),
    }


def score_plan(
    plan: np.ndarray,
    dt: float,
    ego2world: np.ndarray,
    t0_us: float,
    tracks: Sequence[Track],
    *,
    area: DrivableArea | None = None,
    centreline: np.ndarray | None = None,
    references: Sequence[States] = (),
    ego: Ego = NUSCENES_EGO,
    speed0: float | None = None,
) -> dict[str, float]:
    """The PDM-style terms of one ego-frame plan made at `t0_us`, against `tracks`.

    `area` and `centreline` are optional: without a map DAC is NaN; without a
    centreline EP is. `references` are the other paths EP is normalised against
    (each scored for NC x DAC the same way, and counted only where that holds).
    """
    states = plan_states(plan, dt, ego2world, speed0)
    boxes = ego_boxes(states, ego)
    offroad = off_road(boxes, area)
    grid = track_grid(tracks, states, t0_us)
    nc, _ = no_at_fault_collision(states, boxes, grid, offroad)
    dac = float("nan") if offroad is None else float(not offroad.any())
    ttc = time_to_collision(states, boxes, grid, offroad)
    plan_xy = np.asarray(plan, dtype=float)[:, :2]
    c = plan_comfort(plan_xy, dt)
    ep = float("nan")
    if centreline is not None and len(centreline) >= 2:
        refs = []
        for ref in references:
            rboxes = ego_boxes(ref, ego)
            roff = off_road(rboxes, area)
            rnc, _ = no_at_fault_collision(ref, rboxes, track_grid(tracks, ref, t0_us), roff)
            rdac = 1.0 if roff is None else float(not roff.any())
            refs.append(progress(ref, centreline) * rnc * rdac)
        ep = normalised_progress(progress(states, centreline), nc * (1.0 if math.isnan(dac) else dac), refs)
    terms = {"pdm_nc": nc, "pdm_dac": dac, "pdm_ttc": ttc, "pdm_ep": ep, "pdm_comfort": float(c)}
    terms["pdm_style_score"] = composite(nc, dac, ttc, ep, float(c))
    return terms
