"""Choose the comma2k19 segments a viewer working set should hold, and pin them.

comma2k19 is 2035 segments of mostly straight highway. A working set drawn
uniformly from it is ten minutes of a car going forwards, which tells you
nothing about six planners you are trying to tell apart -- they only disagree
where the road does something. So the draw is random *within* behaviour buckets
measured off the recorded pose and radar:

    sustained curve   >= CURVE_S cumulative with |yaw rate| over CURVE_YAW at speed
    speed change      max - min speed over SPEED_RANGE
    close lead        an in-lane radar lead inside LEAD_RANGE_M for >= LEAD_S
    straight highway  fast and flat throughout -- the control

The gates are the ones docs/benchmarks/comma2k19.md already used to pick its
three hand-chosen scenes, so "a curve" keeps meaning a measured one rather than
one that looked like a curve in a thumbnail.

    python -m tools.benchmarks.comma2k19.pick_scenes --seed 20260917

Writes `src/paddock/adapters/benchmarks/comma2k19/scenes/<name>.yaml`, which the
harness then reads through `PADDOCK_COMMA2K19_SCENE_SET=<name>` -- that file is
tracked, and is the reason a re-run of these ten scenes at another site scores
the same ten. The features it measured go in beside the list, so a reader can
see what they got without re-running this.

Measuring every segment is ~2035 pose reads off the dataset mount and takes a
few minutes, so the feature table is cached under
`{runs}/comma2k19/_viewer/segment_features.json` and reused. `--refresh` pays
for it again.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from tools import paths

#: Where the pinned sets live -- the harness's own directory, not a tools one.
#: The file is an input to a benchmark run, so it belongs with the benchmark.
SCENES_DIR = H.SCENES_DIR

#: A prediction every 0.5 s. On a standard 1200-frame segment the scoreable span
#: runs frame 159 -> 999, and 85 keyframes land exactly 10 frames apart across
#: it -- so this number and `SRC_HZ / 10` are the same statement.
KEYFRAMES_PER_SEGMENT = 85

#: Segments shorter than this cannot carry a full 42 s scoreable span, so 85
#: keyframes would crowd together and the set would not mean what it says.
MIN_FRAMES = 1160

# Bucket gates. Named rather than inlined because these are the definitions of
# "interesting" this set is pinned against, and a reader comparing two sets
# needs to be able to diff them.
CURVE_YAW = 0.06  # rad/s -- about a 450 m radius at 27 m/s
CURVE_SPEED = 20.0  # m/s; a yaw rate in a car park is not the curve wanted here
CURVE_S = 3.0
SPEED_RANGE = 5.0  # m/s between the slowest and fastest moment of the span
LEAD_RANGE_M = 25.0
LEAD_S = 3.0
#: A lead is only interesting if the ego car is doing something about it. 72 of
#: the 123 segments passing the range and duration gates are a stationary car
#: behind a stationary car, where every planner correctly predicts nothing.
LEAD_MIN_SPEED = 5.0
STRAIGHT_SPEED = 29.0  # m/s throughout

#: Straightness is `curve_s == 0`, i.e. not one frame of the span turning hard
#: enough to count, rather than a ceiling on `yaw_abs_max`. Differencing the
#: recorded pose leaves a noise floor around 0.024 rad/s: across the 257
#: fast-and-flat segments here, the *smallest* peak |yaw rate| is 0.024 and the
#: median 0.047, so any threshold fine enough to mean "straight" selects nothing
#: and any threshold that selects something is measuring noise. The curve gate
#: already survives that floor by requiring CURVE_S seconds above it.
STRAIGHT_CURVE_S = 0.0

#: How many of each to draw, and the order buckets are tried in. First match
#: wins, so a curve that also changes speed is a curve: the rarer behaviour is
#: the one worth guaranteeing.
QUOTA = (("curve", 3), ("speed_change", 3), ("close_lead", 2), ("straight", 2))

#: Radar rows are ~15 Hz and a lead does not appear and vanish between frames,
#: so the lead gate samples the span at this rate rather than per frame. Four
#: times a second over a 42 s span is 168 lookups instead of 840.
LEAD_SAMPLE_HZ = 4.0


def scoreable_span(n_frames: int, grid: H.EvalGrid) -> tuple[int, int]:
    """(first, last) frame that could be a keyframe, inclusive.

    The same arithmetic as `Comma2k19Benchmark.keyframe_indices`, and the reason
    features are measured over it rather than over the whole segment: a segment
    that brakes hard in its first two seconds is not a braking *sample*, because
    no keyframe can ever land there.
    """
    return grid.history_span_frames - 1, n_frames - grid.future_frames - 1


def lead_seconds(seg_path: Path, pose: L.SegmentPose, first: int, last: int) -> float:
    """Seconds of the span with an in-lane lead inside `LEAD_RANGE_M`."""
    radar = L.load_radar(seg_path)
    if radar is None:
        return 0.0
    radar_t, radar_v = radar
    step = max(1, int(round(L.SRC_HZ / LEAD_SAMPLE_HZ)))
    indices = range(first, last + 1, step)
    close = 0
    for i in indices:
        lead = L.lead_track(L.radar_tracks(radar_t, radar_v, float(pose.times[i])))
        if lead is not None and float(lead[0]) < LEAD_RANGE_M:
            close += 1
    return close * step / L.SRC_HZ


def measure(segment: L.SegmentRef, grid: H.EvalGrid) -> dict | None:
    """What one segment does over the frames that could be scored, or None.

    None means the segment cannot carry the working set -- too short, or its
    pose stream will not load -- which is a fact about the data rather than an
    error, and is why this returns rather than raises.
    """
    try:
        pose = L.load_pose(segment.path)
    except (OSError, ValueError):
        return None
    n_frames = len(pose)
    if n_frames < MIN_FRAMES:
        return None
    first, last = scoreable_span(n_frames, grid)
    if last <= first:
        return None

    speed = pose.speed[first : last + 1]
    yaw = pose.yaw_rate[first : last + 1]
    turning = (np.abs(yaw) > CURVE_YAW) & (speed > CURVE_SPEED)
    return {
        "vehicle": segment.vehicle,
        "n_frames": n_frames,
        "span_s": round((last - first + 1) / L.SRC_HZ, 2),
        "speed_mean": round(float(speed.mean()), 2),
        "speed_min": round(float(speed.min()), 2),
        "speed_max": round(float(speed.max()), 2),
        "yaw_abs_max": round(float(np.abs(yaw).max()), 4),
        "curve_s": round(float(turning.sum()) / L.SRC_HZ, 2),
        "lead_close_s": round(lead_seconds(segment.path, pose, first, last), 2),
    }


def bucket_of(f: dict) -> str:
    """Which behaviour this segment is an example of. First match wins."""
    if f["curve_s"] >= CURVE_S:
        return "curve"
    if f["speed_max"] - f["speed_min"] > SPEED_RANGE:
        return "speed_change"
    if f["lead_close_s"] >= LEAD_S and f["speed_max"] > LEAD_MIN_SPEED:
        return "close_lead"
    if f["speed_min"] > STRAIGHT_SPEED and f["curve_s"] <= STRAIGHT_CURVE_S:
        return "straight"
    return "other"


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------


def feature_table(segments: list[L.SegmentRef], cache: Path, refresh: bool) -> dict[str, dict]:
    """Measure every segment, reusing whatever a previous run already measured.

    Cached outside the repo because it is a fact about this site's copy of the
    dataset -- 2035 rows naming segments another site may not have -- while the
    ten scenes drawn from it are a fact about the selection, and tracked.
    """
    known: dict[str, dict] = {}
    if cache.exists() and not refresh:
        try:
            known = json.loads(cache.read_text())
        except (OSError, ValueError):
            known = {}
    grid = H.EvalGrid.build(L.SRC_HZ, H.DEFAULT_HISTORY_FRAMES)

    todo = [s for s in segments if s.sample_key not in known]
    began = time.time()
    for n, segment in enumerate(todo, 1):
        measured = measure(segment, grid)
        # A `None` is cached too: re-reading an unusable segment on every run is
        # the cost this table exists to avoid, and "unusable" is also a result.
        known[segment.sample_key] = measured if measured is not None else {}
        if n % 100 == 0 or n == len(todo):
            rate = n / max(time.time() - began, 1e-6)
            print(f"  measured {n}/{len(todo)} ({rate:.0f}/s)", file=sys.stderr)

    if todo:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(known, sort_keys=True, indent=1) + "\n")
        os.replace(tmp, cache)
        print(f"  feature table: {cache}", file=sys.stderr)
    return {k: v for k, v in known.items() if v}


# --------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------


def draw(features: dict[str, dict], seed: int) -> tuple[list[str], list[str]]:
    """(chosen keys, notes about how the quota was met).

    At most one segment per route. Two segments of one drive are sixty seconds
    apart on the same road with the same car and the same calibration, so they
    are close to one sample for the purpose of telling planners apart -- and the
    quota is small enough that spending two of ten on it would be felt.
    """
    rng = random.Random(seed)
    notes: list[str] = []
    buckets: dict[str, list[str]] = {}
    for key, f in sorted(features.items()):
        buckets.setdefault(bucket_of(f), []).append(key)

    chosen: list[str] = []
    used_routes: set[str] = set()

    def take(pool: list[str], k: int) -> list[str]:
        got = []
        for key in rng.sample(pool, len(pool)):
            route = key.rsplit("/", 1)[0]
            if route in used_routes:
                continue
            used_routes.add(route)
            got.append(key)
            if len(got) == k:
                break
        return got

    for name, quota in QUOTA:
        pool = [k for k in buckets.get(name, []) if k not in chosen]
        got = take(pool, quota)
        chosen.extend(got)
        if len(got) < quota:
            notes.append(f"{name}: wanted {quota}, {len(got)} available")

    # Back-fill from whatever else passed a gate, rather than silently returning
    # a short set: the viewer's layout and the run cost are both stated in tens.
    # `other` stays excluded -- a short set of interesting scenes is a better
    # working set than a full one padded with empty motorway.
    wanted = sum(k for _, k in QUOTA)
    if len(chosen) < wanted:
        rest = sorted(
            (k for f, keys in buckets.items() if f != "other" for k in keys if k not in chosen),
            key=lambda k: bucket_of(features[k]),
        )
        filled = take(rest, wanted - len(chosen))
        chosen.extend(filled)
        notes.append(f"back-filled {len(filled)} to reach {wanted}")
    return chosen, notes


def gate_of(bucket: str) -> str:
    """What `bucket_of` actually tested, in words, for the file's group headers.

    Built from the gate constants rather than restated, so loosening a gate
    cannot leave a pinned file describing the gate it was drawn under.
    """
    return {
        "curve": f">= {CURVE_S:g} s cumulative over {CURVE_YAW:g} rad/s at {CURVE_SPEED:g} m/s or more",
        "speed_change": f"fastest - slowest moment > {SPEED_RANGE:g} m/s",
        "close_lead": (
            f"in-lane lead within {LEAD_RANGE_M:g} m for >= {LEAD_S:g} s, ego over {LEAD_MIN_SPEED:g} m/s"
        ),
        "straight": f"no frame turning, and over {STRAIGHT_SPEED:g} m/s throughout",
    }.get(bucket, "did not match a gate")


def to_yaml(chosen: list[str], features: dict[str, dict], seed: int, notes: list[str]) -> str:
    """The scene-set file, hand-written rather than dumped.

    `yaml.safe_dump` would drop the comment block, and the comment block is the
    provenance: without it the reader has ten opaque segment ids and no way to
    know what they were chosen to show. It would also drop the per-bucket group
    headers below, which are what let a reader see the shape of the draw -- the
    grouping is comments and ordering only, so `scenes:` is still the flat list
    of ids the harness reads.
    """
    by_bucket = {k: bucket_of(features[k]) for k in chosen}
    order = [b for b, _ in QUOTA]
    rank = {b: i for i, b in enumerate(order)}
    drawn = sorted(set(by_bucket.values()), key=lambda b: (rank.get(b, len(order)), b))
    # Counts in draw order, not alphabetical: that is the order the buckets were
    # filled in and the order the two blocks below are grouped in.
    counts = ", ".join(f"{b}:{sum(1 for v in by_bucket.values() if v == b)}" for b in drawn)
    lines = [
        f"# The pinned {len(chosen)}-scene working set for the side-by-side viewer.",
        "#",
        "# Drawn by tools/benchmarks/comma2k19/pick_scenes.py: random within behaviour",
        "# buckets measured off each segment's recorded pose and radar, over the frames",
        "# that could actually be scored. Pinned as an explicit list for the same reason",
        "# alpasim pins its suite -- a set re-derived at run time is a set that can change",
        "# under two sites comparing numbers.",
        "#",
        f"# Buckets drawn, first match wins: {counts}. At most one segment per route.",
        "# `features` is recorded, not read at run time: it is how a reader sees what",
        "# these ten actually do without re-measuring 2035 segments. Both blocks below",
        "# are grouped by bucket in draw order, under the gate each bucket was drawn on.",
        "selection: seeded random draw within behaviour buckets; see tools/benchmarks/comma2k19/pick_scenes.py",
        f"seed: {seed}",
        f"keyframes_per_segment: {KEYFRAMES_PER_SEGMENT}",
    ]
    if notes:
        lines.append("quota_notes:")
        lines.extend(f"- {n}" for n in notes)
    lines.append("scenes:")
    order = [b for b, _ in QUOTA]
    ranked = sorted(chosen, key=lambda k: (order.index(by_bucket[k]) if by_bucket[k] in order else 99, k))
    lines.extend(f"- {k}" for k in ranked)
    lines.append("features:")
    for key in ranked:
        row = {"bucket": by_bucket[key], **features[key]}
        body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in row.items())
        lines.append(f"  {key}: {{{body}}}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260917, help="pinned into the file; changing it redraws")
    ap.add_argument("--name", default="viewer10", help="scene set name, i.e. scenes/<name>.yaml")
    ap.add_argument("--refresh", action="store_true", help="re-measure every segment, ignoring the cache")
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N segments (smoke test)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = paths.config()
    dataset = cfg.datasets.get("comma2k19")
    if dataset is None:
        raise SystemExit("dataset 'comma2k19' is not configured in paddock.toml; see docs/data-setup.md")
    segments = L.list_segments(dataset)
    if not segments:
        raise SystemExit(f"no comma2k19 segments under {', '.join(str(r) for r in dataset.roots)}")
    if args.limit:
        segments = segments[: args.limit]
    print(f"{len(segments)} segments to consider", file=sys.stderr)

    cache = cfg.runs / "comma2k19" / "_viewer" / "segment_features.json"
    features = feature_table(segments, cache, args.refresh)
    here = {s.sample_key for s in segments}
    features = {k: v for k, v in features.items() if k in here}

    tally: dict[str, int] = {}
    for f in features.values():
        tally[bucket_of(f)] = tally.get(bucket_of(f), 0) + 1
    print("  usable by bucket: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())), file=sys.stderr)

    chosen, notes = draw(features, args.seed)
    if not chosen:
        raise SystemExit("no segment passed any bucket gate; nothing to pin")

    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    out = SCENES_DIR / f"{args.name}.yaml"
    out.write_text(to_yaml(chosen, features, args.seed, notes))
    for key in chosen:
        f = features[key]
        print(
            f"  {bucket_of(f):>12}  {key}  speed {f['speed_min']:.0f}-{f['speed_max']:.0f} m/s"
            f"  curve {f['curve_s']:.1f} s  lead {f['lead_close_s']:.1f} s",
            file=sys.stderr,
        )
    for note in notes:
        print(f"  note: {note}", file=sys.stderr)
    print(f"wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
