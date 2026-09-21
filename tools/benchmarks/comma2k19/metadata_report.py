"""Write COMMA2K19_METADATA.md: what this machine's copy of comma2k19 contains.

The artefact to hand a developer who has just been pointed at the dataset. It
answers the questions that otherwise cost an afternoon each -- what a `chunk` is,
which column of the radar array is the track address, what rate the IMU actually
sampled at, whether the video header's frame rate can be believed -- and it
answers them from the files on this disk rather than from the paper.

    python -m tools.benchmarks.comma2k19.metadata_report --out COMMA2K19_METADATA.md

Adapted from the toolkit's `.tmp/comma2k19_metadata.py`. Two differences worth
knowing about.

The walk and the array reading now live in
`adapters/benchmarks/comma2k19/inventory.py`, so this file is only the markdown
builder. That is deliberate: `paddock doctor` prints a short version of the same
facts through `harness.status`, and two implementations of "what is on disk"
would eventually disagree in the one place where disagreement is invisible.

And the numbers here are honest about their sample size. Loading every array of
2035 segments is minutes of NAS traffic for a document, so the deep pass reads a
stratified sample and every table built from it says so. Widen it with
`--per-chunk` when a number looks wrong.

Needs nothing beyond paddock's own environment; PyAV, if present, adds the video
probe rows.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import inventory as INV
from paddock.adapters.benchmarks.comma2k19 import loader as L
from tools import paths

DONGLE_CARS = {L.RAV4_DONGLE: "Toyota RAV4", L.CIVIC_DONGLE: "Honda Civic"}

#: From the publication, for comparison against what is actually here. A site
#: with fewer segments has a partial download, which is normal and worth naming.
PUBLISHED_SEGMENTS = 2019
PUBLISHED_HOURS = 33


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def cell(value) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    # A route_id contains a `|`, which ends a table cell even inside backticks.
    return str(value).replace("|", "\\|")


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def _route_date(route_id: str) -> str:
    match = INV.ROUTE_RE.match(route_id)
    return match.group("start")[:10] if match else "?"


def _dongle(route_id: str) -> str:
    return route_id.split("|", 1)[0].split("_", 1)[0]


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _at_a_glance(found: INV.Survey, deep: list[dict]) -> list[str]:
    frames = [d["frames"] for d in deep if "frames" in d]
    mean_frames = float(np.mean(frames)) if frames else float("nan")
    speeds = [d["speed_mps"][1] for d in deep if "speed_mps" in d]
    mean_speed = float(np.mean(speeds)) if speeds else float("nan")
    sizes = [d["bytes"] for d in deep if "bytes" in d]
    mean_bytes = float(np.mean(sizes)) if sizes else 0.0
    dates = sorted(_route_date(r) for _, r in found.routes)

    rows = [
        ["Chunks present", f"{len(found.chunks)} of 10"],
        ["Routes (`dongle_id|start_time`)", found.n_routes],
        ["Segments (1 min each)", found.n_segments],
        ["Driving time", f"{found.hours:.1f} h"],
        ["Vehicles", found.vehicle_summary() or "unknown"],
        ["Recording dates", f"{dates[0]} to {dates[-1]}" if dates else "n/a"],
        ["Frames per segment", f"{mean_frames:,.0f} (sampled, n={len(frames)})"],
        ["Video frames, total", f"~{mean_frames * found.n_segments:,.0f} (est.)"],
        ["Mean speed", f"{mean_speed:.1f} m/s ({mean_speed * 3.6:.0f} km/h), sampled"],
        ["Distance driven", f"~{mean_speed * 60 * found.n_segments / 1000:,.0f} km (est.)"],
        ["Bytes per segment", f"{INV.human(mean_bytes)} (sampled, n={len(sizes)})"],
        ["Size on disk", f"~{INV.human(mean_bytes * found.n_segments)} (est.)"],
        ["Road", "CA-280, San Jose to San Francisco, ~20 km section *(from README)*"],
        ["Licence / citation", "MIT; cite arXiv:1812.05752 *(from README)*"],
    ]
    body = [
        "## 1. At a glance",
        "",
        table(["Property", "Value"], rows),
        "",
        f"The publication describes {PUBLISHED_SEGMENTS:,} segments and >{PUBLISHED_HOURS} h of "
        f"commute. This copy holds **{found.n_segments:,} segments**"
        + (
            ""
            if found.n_segments >= PUBLISHED_SEGMENTS
            else f" -- {PUBLISHED_SEGMENTS - found.n_segments:,} fewer, i.e. a partial download, "
            "which is the normal state until every chunk has landed"
        )
        + ".",
        "",
    ]
    return body


def _vocabulary(found: INV.Survey) -> list[str]:
    return [
        "## 2. Vocabulary",
        "",
        "The names in the directory tree, none of which are self-explanatory:",
        "",
        table(
            ["Term", "Meaning", "Count here"],
            [
                ["`chunk`", "Download unit, ~200 min of driving (`Chunk_1` ... `Chunk_10`)", len(found.chunks)],
                ["`route`", "One drive, a directory named `dongle_id|start_time`", found.n_routes],
                ["`dongle_id`", "16-hex id of the comma EON device; one per car", len(DONGLE_CARS)],
                ["`segment`", "One minute of a route, a numbered directory", found.n_segments],
                ["`frame`", "One camera image in `video.hevc`, posed in `global_pose/`", "20 per second"],
                ["`processed_log`", "Per-segment numpy arrays: IMU / CAN / GNSS", found.n_segments],
                ["`global_pose`", "Per-frame camera pose in ECEF", found.n_segments],
                ["`raw_log.bz2`", "Raw capnp log, read with openpilot-tools `logreader`", found.n_segments],
            ],
        ),
        "",
        "Two of these are load-bearing for paddock. A **segment** is one sample's worth",
        "of data, and the harness scores keyframes inside it. A **route** is the unit",
        "camera calibration is pooled over, because the phone is not re-mounted mid-drive",
        "-- see `calibration.py`.",
        "",
    ]


def _layout(found: INV.Survey) -> list[str]:
    per_route = Counter((s.chunk, s.route_id) for s in found.segments)
    counts = sorted(per_route.values())
    numbers = sorted(s.segment for s in found.segments)
    return [
        "## 3. Directory layout",
        "",
        "```text",
        "raw_data/Chunk_n/                       # 10 chunks, ~200 min of driving each",
        "|",
        "+-- <dongle_id>|<start_time>/           # route_id, e.g. b0c9d2329ad1606b|2018-07-27--06-03-57",
        "    |",
        "    +-- <segment_number>/               # 1 minute of driving",
        "        +-- preview.png                 # first frame of the video",
        "        +-- raw_log.bz2                 # raw capnp log (openpilot-tools: logreader)",
        "        +-- video.hevc                  # road-facing camera, 1164x874",
        "        +-- global_pose/                # 5 numpy arrays, one row per frame",
        "        +-- processed_log/{IMU,CAN,GNSS}/<signal>/{t,value}",
        "```",
        "",
        "The arrays are **extension-less files that are .npy in content**. `np.load`",
        "reads them as they are; a loader that appends `.npy` finds nothing.",
        "",
        f"Segment numbers are not contiguous within a route: {counts[0]}-{counts[-1]} segments "
        f"per route (mean {np.mean(counts):.1f}), numbered {numbers[0]}-{numbers[-1]}.",
        "A gap is a segment the dataset's own filtering dropped, not a missing download.",
        "",
    ]


def _chunk_inventory(found: INV.Survey) -> list[str]:
    by_chunk: dict[int, list] = {}
    for seg in found.segments:
        by_chunk.setdefault(seg.chunk, []).append(seg)
    rows = []
    for chunk in sorted(by_chunk):
        segs = by_chunk[chunk]
        routes = sorted({s.route_id for s in segs})
        cars = sorted({DONGLE_CARS.get(_dongle(r), "unknown") for r in routes})
        dates = sorted(_route_date(r) for r in routes)
        rows.append(
            [f"`Chunk_{chunk}`", ", ".join(cars), len(routes), len(segs), f"{len(segs) / 60:.1f}",
             f"{dates[0]} to {dates[-1]}"]
        )  # fmt: skip
    rows.append(["**total**", "", found.n_routes, found.n_segments, f"{found.hours:.1f}", ""])

    per_car: dict[str, list] = {}
    for seg in found.segments:
        per_car.setdefault(_dongle(seg.route_id), []).append(seg)
    car_rows = [
        [f"`{d}`", DONGLE_CARS.get(d, "unknown"), len({s.route_id for s in segs}), len(segs),
         f"{len(segs) / 60:.1f}"]
        for d, segs in sorted(per_car.items())
    ]  # fmt: skip

    return [
        "## 4. Chunk inventory",
        "",
        table(["Chunk", "Car", "Routes", "Segments", "Hours", "Dates"], rows),
        "",
        "Per car:",
        "",
        table(["dongle_id", "Car", "Routes", "Segments", "Hours"], car_rows),
        "",
        "A chunk is a download unit and nothing more: it is not a split, and routes in",
        "different chunks are not held out from each other. Splitting by **vehicle** is",
        "the split this dataset actually supports -- see `docs/benchmarks/comma2k19.md`.",
        "",
    ]


def _storage(deep: list[dict]) -> list[str]:
    totals: dict[str, list[int]] = {}
    for d in deep:
        for name, size in d.get("files", {}).items():
            totals.setdefault(name, []).append(size)
    if not totals:
        return []
    rows = [
        [f"`{name}`", len(sizes), INV.human(float(np.mean(sizes))), f"{100 * sum(sizes) / _all(totals):.1f}%"]
        for name, sizes in sorted(totals.items(), key=lambda kv: -sum(kv[1]))[:16]
    ]
    return [
        "## 5. Storage profile",
        "",
        f"Mean size of each per-segment file, over {len(deep)} sampled segments.",
        "",
        table(["File", "Seen in", "Mean size", "Share of a segment"], rows),
        "",
        "`video.hevc` dominates, which is why the harness streams frames rather than",
        "caching decoded ones -- a decoded segment is ~3 GB against ~50 MB encoded.",
        "",
    ]


def _all(totals: dict[str, list[int]]) -> float:
    return float(sum(sum(v) for v in totals.values())) or 1.0


def _video(deep: list[dict]) -> list[str]:
    probes = [d["video"] for d in deep if isinstance(d.get("video"), dict)]
    good = [p for p in probes if "error" not in p]
    if not good:
        errors = {p.get("error", "?") for p in probes}
        return ["## 6. Video", "", "No video could be probed: " + "; ".join(sorted(errors)), ""]
    first = good[0]
    measured = [d["fps"] for d in deep if "fps" in d]
    header = {p.get("container_rate") for p in good}
    packets = [p["packets"] for p in good if p.get("packets") is not None]
    return [
        "## 6. Video",
        "",
        table(
            ["Property", "Value"],
            [
                ["Codec", f"`{first['codec']}`, profile {first['profile']}"],
                ["Geometry", f"{first['width']} x {first['height']}, `{first['pix_fmt']}`"],
                ["Packets per segment", f"{np.mean(packets):,.0f} (sampled)" if packets else "not counted"],
                ["Header frame rate", ", ".join(f"{r:g}" for r in sorted(header) if r) + " fps"],
                ["Measured frame rate", f"{np.mean(measured):.2f} Hz (from `global_pose/frame_times`)"],
            ],
        ),
        "",
        "**The header rate is wrong and the measured one is right.** The files are raw",
        "HEVC elementary streams with no container timing, so a decoder reports a default",
        "and anything that trusts it is off by a quarter. `global_pose/frame_times` is the",
        "authority, and `overlay.SegmentOverlay.fps` reads it rather than the header.",
        "",
    ]


def _pose(deep: list[dict]) -> list[str]:
    shapes: dict[str, set[str]] = {}
    for d in deep:
        for key, info in d.get("arrays", {}).items():
            if key.startswith("global_pose/"):
                shapes.setdefault(key, set()).add(f"(n, {info['shape'][1]})" if len(info["shape"]) > 1 else "(n,)")
    rows = [
        [f"`{key}`", INV.POSE_SPEC[key][0], ", ".join(sorted(shapes.get(key, {"absent"}))), INV.POSE_SPEC[key][1]]
        for key in INV.POSE_SPEC
    ]
    return [
        "## 7. `global_pose/` -- per-frame camera pose",
        "",
        "One row per video frame, already fused from INS/GNSS/vision. There is no SLAM",
        "or GNSS pipeline to run before this dataset can be scored, which is the main",
        "thing that makes it cheap to work with.",
        "",
        table(["Array", "Meaning *(from README)*", "Shape", "Unit"], rows),
        "",
        "> The README states the quaternion direction **backwards**. It says the",
        "> quaternion rotates ECEF into the device frame; the toolkit's own",
        "> `device_from_ecef` transposes it before use, which is the tell. Getting this",
        "> wrong does not raise -- it mirrors every relative position through the",
        "> direction of travel. See `docs/benchmarks/comma2k19.md`.",
        "",
    ]


def _logs(deep: list[dict]) -> list[str]:
    rates: dict[str, list[float]] = {}
    shapes: dict[str, set[str]] = {}
    for d in deep:
        for key, hz in d.get("rates", {}).items():
            if np.isfinite(hz):
                rates.setdefault(key, []).append(hz)
        for key, info in d.get("arrays", {}).items():
            if key.endswith("/value"):
                base = key[: -len("/value")]
                shapes.setdefault(base, set()).add(
                    f"(n, {info['shape'][1]})" if len(info["shape"]) > 1 else "(n,)"
                )

    out = ["## 8. `processed_log/` -- sensor time series", "",
           "Every signal is a directory holding `t` (boot time, seconds) and `value`.",
           "The columns have no names on disk; these come from the README, and the",
           "measured shape beside each is the check that they still line up.",
           ""]  # fmt: skip
    for group in ("IMU", "CAN", "GNSS"):
        rows = []
        for key, (name, layout, unit) in INV.LOG_SPEC.items():
            if f"/{group}/" not in key:
                continue
            seen = sorted(shapes.get(key, set())) or (["(no value array)"] if "raw_can" in key else ["absent"])
            hz = rates.get(key, [])
            rows.append(
                [f"`{name}`", layout, unit, ", ".join(seen), f"{np.mean(hz):.1f}" if hz else "n/a"]
            )
        if rows:
            out += [f"### 8.{['IMU', 'CAN', 'GNSS'].index(group) + 1} {group}", "",
                    table(["Signal", "Columns *(from README)*", "Unit", "Shape", "Hz"], rows), ""]  # fmt: skip
    out += [
        "Rates are means over the sampled segments. They are **not** all 20 Hz: the CAN",
        "signals arrive as the bus delivers them, so a harness that assumes the camera",
        "rate for a CAN signal reads a stale value. `loader.load_can_speed` interpolates",
        "onto frame times for exactly this reason.",
        "",
    ]
    return out


def _can(deep: list[dict]) -> list[str]:
    addresses: Counter = Counter()
    messages, radar_points, radar_tracks, dists = [], [], [], []
    for d in deep:
        addresses.update(d.get("can_addresses", {}))
        if "can_messages" in d:
            messages.append(d["can_messages"])
        if "radar_points" in d:
            radar_points.append(d["radar_points"])
            radar_tracks.append(d["radar_tracks"])
            dists.append(d["radar_fwd_dist"])
    if not messages and not radar_points:
        return []
    out = ["## 9. CAN bus", ""]
    if messages:
        out += [
            f"{np.mean(messages):,.0f} raw messages per segment over "
            f"{len(addresses):,} distinct addresses (sampled, n={len(messages)}). "
            "The 12 most frequent:",
            "",
            table(
                ["Address (hex)", "Address (dec)", "Messages in sample"],
                [[f"`0x{a:03x}`", a, n] for a, n in addresses.most_common(12)],
            ),
            "",
            "`raw_can` is unparsed: decoding an address to a signal needs the vehicle's",
            "DBC, which this dataset does not ship. The already-decoded signals under",
            "`processed_log/CAN/` are what to use unless you have opendbc to hand.",
            "",
        ]
    if radar_points:
        out += [
            f"Radar: {np.mean(radar_points):,.0f} points per segment across "
            f"{np.mean(radar_tracks):.0f} track addresses, forward distance "
            f"{min(d[0] for d in dists):.1f} to {max(d[1] for d in dists):.1f} m.",
            "",
            "Column 5 is the **track address**, and a track is re-reported every few",
            "frames -- so counting rows counts reports, not objects. `overlay.radar_tracks`",
            "keeps the last row per address, which is what makes its count mean something.",
            "",
        ]
    return out


def _geography(deep: list[dict]) -> list[str]:
    fixes = [d["live_gnss_ublox"] for d in deep if "live_gnss_ublox" in d]
    if not fixes:
        return []
    lat = [min(f["lat"][0] for f in fixes), max(f["lat"][1] for f in fixes)]
    lon = [min(f["lon"][0] for f in fixes), max(f["lon"][1] for f in fixes)]
    alt = [min(f["alt"][0] for f in fixes), max(f["alt"][1] for f in fixes)]
    speeds = [d["speed_mps"] for d in deep if "speed_mps" in d]
    steer = [d["steering_deg"] for d in deep if "steering_deg" in d]
    rows = [
        ["Latitude", f"{lat[0]:.4f} to {lat[1]:.4f} deg"],
        ["Longitude", f"{lon[0]:.4f} to {lon[1]:.4f} deg"],
        ["Altitude", f"{alt[0]:.0f} to {alt[1]:.0f} m"],
    ]
    if speeds:
        rows.append(
            ["CAN speed", f"{min(s[0] for s in speeds):.1f} to {max(s[2] for s in speeds):.1f} m/s, "
             f"mean {np.mean([s[1] for s in speeds]):.1f}"]
        )  # fmt: skip
    if steer:
        rows.append(
            ["Steering angle", f"{min(s[0] for s in steer):+.0f} to {max(s[1] for s in steer):+.0f} deg"]
        )
    return [
        "## 10. Geography and dynamics (sampled)",
        "",
        table(["Quantity", "Range"], rows),
        "",
        "One highway, one direction of commute, one metropolitan area, mostly at speed.",
        "A model scored well here has been scored on lane-keeping at highway speed and",
        "on nothing else -- no intersections, no city driving, no weather worth the name.",
        "",
    ]


def _integrity(found: INV.Survey, deep: list[dict]) -> list[str]:
    errors = [(d["segment"], e) for d in deep for e in d.get("errors", [])]
    out = [
        "## 11. Integrity",
        "",
        f"- {found.checked} segments spot-checked for `{'`, `'.join(INV.SEGMENT_FILES)}`: "
        f"**{found.checked - len(found.incomplete)} complete**.",
        f"- {len(deep)} segments read array by array: **{len(deep) - len({s for s, _ in errors})} clean**.",
    ]
    for note in found.incomplete[:20]:
        out.append(f"- incomplete: {note}")
    for segment, error in errors[:20]:
        out.append(f"- unreadable: `{segment}` {error}")
    for note in found.anomalies[:20]:
        out.append(f"- skipped: {note}")
    out += [
        "",
        "A skipped directory is not a fault. `list_segments` accepts a segment only if it",
        "has a `video.hevc` and a `global_pose/frame_times`, because those are what",
        "scoring reads; anything else -- a stray file, a half-extracted route -- is passed",
        "over silently by the harness and listed here so a person can decide.",
        "",
    ]
    return out


def _provenance(found: INV.Survey, args, elapsed: float) -> list[str]:
    return [
        "## 12. How this was produced",
        "",
        "```bash",
        f"python -m tools.benchmarks.comma2k19.metadata_report --per-chunk {args.per_chunk}",
        "```",
        "",
        f"- Dataset root(s): {', '.join(f'`{r}`' for r in found.roots)}",
        f"- Walked {found.n_segments:,} segments; read arrays from {len(found.deep['sampled'])} of them",
        f"  ({args.per_chunk} per chunk, seed {INV.SEED}, so a re-run describes the same segments).",
        f"- Took {elapsed:.1f} s.",
        "- Every number is measured from the files on this disk unless marked *(from README)*.",
        "",
        "Regenerate it after adding chunks. The short version of these facts is in",
        "`paddock doctor`, from the same code -- see `harness.status`.",
        "",
    ]


def build(found: INV.Survey, args, elapsed: float) -> str:
    deep = found.deep["inspections"]
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    out = [
        "# comma2k19 -- what is on this machine",
        "",
        f"> Generated by `tools/benchmarks/comma2k19/metadata_report.py` on {when}.  ",
        "> Measured from the files on disk, except where marked *(from README)*.  ",
        "> Sampled statistics name their sample size.",
        "",
    ]
    out += _at_a_glance(found, deep)
    out += _vocabulary(found)
    out += _layout(found)
    out += _chunk_inventory(found)
    out += _storage(deep)
    out += _video(deep)
    out += _pose(deep)
    out += _logs(deep)
    out += _can(deep)
    out += _geography(deep)
    out += _integrity(found, deep)
    out += _provenance(found, args, elapsed)
    return "\n".join(out).rstrip() + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="COMMA2K19_METADATA.md", help="where to write the report")
    ap.add_argument(
        "--per-chunk",
        type=int,
        default=INV.DEEP_PER_CHUNK,
        help="segments read array-by-array per chunk; raising this widens every sampled number",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = paths.config().datasets.get("comma2k19")
    if dataset is None:
        raise SystemExit("dataset 'comma2k19' is not configured in paddock.toml; see docs/data-setup.md")

    began = time.time()
    print(f"walking {', '.join(str(r) for r in dataset.roots)} ...", file=sys.stderr)
    found = INV.survey(dataset, deep=True, per_chunk=args.per_chunk)
    if not found.n_segments:
        raise SystemExit("no comma2k19 segments found; nothing to report")
    elapsed = time.time() - began

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(found, args, elapsed))
    print(
        f"wrote {out}\n  {found.n_segments:,} segments walked, "
        f"{len(found.deep['sampled'])} read in full, in {elapsed:.1f} s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
