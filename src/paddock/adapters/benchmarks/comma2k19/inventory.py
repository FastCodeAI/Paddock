"""What is actually on this machine's disk, under `[datasets] comma2k19`.

comma2k19 arrives as ten ~100 GB chunks from separate downloads, so a partial
copy is the normal state at a new site and not an error: someone fetches
Chunk_1 to get something running and the rest lands over the following week.
The question "what do I have" therefore comes up constantly, and answering it
by eye means counting directories three levels down.

Two depths, because there are two callers with incompatible budgets.

`survey(dataset)` walks the tree and reads nothing. It is what `harness.status`
puts in `paddock doctor`, which runs on every benchmark and must stay
interactive -- so it stats no files and loads no arrays, and its integrity check
is a *sample* rather than a sweep. On 2035 segments it is a few hundred
milliseconds.

`survey(dataset, deep=True)` additionally loads every array of a stratified
sample of segments: shapes, dtypes, measured rates, the GNSS bounding box, the
CAN address histogram, a video probe. That is minutes, not milliseconds, so
nothing on the doctor path may call it -- it exists for
`tools/benchmarks/comma2k19/metadata_report.py`, which writes the document you
hand a developer setting the dataset up.

Adapted from the toolkit's `.tmp/comma2k19_metadata.py`. The walk itself is
**not** adapted: it delegates to `loader.list_segments`, because a second
implementation of "what counts as a segment" is how a report comes to describe
a dataset the scoring path would not accept.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.config import Dataset

__all__ = ["Survey", "human", "rate_hz", "survey"]

#: Files every complete segment has. `loader._is_segment` requires only
#: `video.hevc` and the pose, because those are what scoring reads; a segment
#: missing `raw_log.bz2` still scores. Reported, therefore, rather than refused.
SEGMENT_FILES = ("video.hevc", "preview.png", "raw_log.bz2", "global_pose/frame_positions")

#: `dongle_id|start_time`, as the dataset writes it. Underscore as well as pipe:
#: a copy that has been through a filesystem which rejects `|` renames it, and
#: such a copy is still perfectly usable.
ROUTE_RE = re.compile(r"^(?P<dongle>[0-9a-f]{16})[|_](?P<start>\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})$")

#: How many segments `survey` spot-checks for completeness, spread over chunks.
#: Enough that a chunk which failed to extract shows up; small enough to stat in
#: well under a second.
SAMPLE_SEGMENTS = 64

#: Deep-pass sample, per chunk. Each one is ~20 array loads and a video demux.
DEEP_PER_CHUNK = 2

#: Fixed, so two runs of the report describe the same segments and a diff
#: between them is a change in the data rather than a change in the dice.
SEED = 20191219


def human(nbytes: float) -> str:
    """Bytes as something a person can compare at a glance."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:,.0f} B" if unit == "B" else f"{nbytes:,.1f} {unit}"
        nbytes /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def rate_hz(t) -> float:
    """Mean sample rate of a boot-time timestamp array, or NaN.

    Mean rather than median: the question is how many samples per second the
    log actually delivered over the segment, and a handful of dropped frames
    should move that number, not be smoothed away.
    """
    if t is None or len(t) < 2:
        return float("nan")
    span = float(t[-1]) - float(t[0])
    return (len(t) - 1) / span if span > 0 else float("nan")


def png_header(path: Path) -> dict[str, int] | None:
    """Width/height of a PNG from its first 33 bytes, without decoding it."""
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", head[16:24])
    return {"width": width, "height": height, "bit_depth": head[24], "color_type": head[25]}


@dataclass
class Survey:
    """What the walk found. Cheap fields always; `deep` only with `deep=True`."""

    roots: list[Path]
    segments: list[L.SegmentRef]
    #: chunk number -> segments present in it.
    chunks: dict[int, int] = field(default_factory=dict)
    #: vehicle ("rav4"/"civic"/"unknown") -> segments.
    vehicles: dict[str, int] = field(default_factory=dict)
    #: (chunk, route_id) for every route holding at least one usable segment.
    routes: list[tuple[int, str]] = field(default_factory=list)
    #: Route directories whose names do not parse, and segment directories that
    #: `loader._is_segment` rejected. Facts, not failures: a partial download
    #: produces both, and the harness simply skips them.
    anomalies: list[str] = field(default_factory=list)
    #: Segments spot-checked for the files scoring does not require.
    checked: int = 0
    incomplete: list[str] = field(default_factory=list)
    #: Only with `deep=True`; see `_deep`.
    deep: dict[str, Any] = field(default_factory=dict)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_routes(self) -> int:
        return len(self.routes)

    @property
    def hours(self) -> float:
        """Recording time, at one minute per segment -- the dataset's own unit."""
        return self.n_segments / 60.0

    def vehicle_summary(self) -> str:
        known = [f"{name} {n}" for name, n in sorted(self.vehicles.items()) if name != "unknown"]
        if self.vehicles.get("unknown"):
            known.append(f"unknown dongle {self.vehicles['unknown']}")
        return ", ".join(known)


def _anomalies(dataset: Dataset, found: set[str]) -> tuple[list[str], list[tuple[int, str]]]:
    """Directory names the walk passed over, and the routes it accepted.

    Separate from `list_segments` on purpose: the harness is right to skip these
    silently -- a run must not fail because a download left a `.part` file --
    but a person setting the dataset up wants to be told.
    """
    notes: list[str] = []
    routes: list[tuple[int, str]] = []
    for root in dataset.roots:
        if not root.exists():
            notes.append(f"{root} does not exist")
            continue
        for chunk_dir in sorted(root.glob("Chunk_*")):
            suffix = chunk_dir.name.removeprefix("Chunk_")
            if not suffix.isdigit():
                notes.append(f"{chunk_dir.name}: not Chunk_<n>, skipped")
                continue
            chunk = int(suffix)
            for route_dir in sorted(chunk_dir.iterdir()):
                if not route_dir.is_dir():
                    notes.append(f"Chunk_{chunk}/{route_dir.name}: file where a route directory goes")
                    continue
                if not ROUTE_RE.match(route_dir.name):
                    notes.append(f"Chunk_{chunk}/{route_dir.name}: not a dongle_id|start_time, skipped")
                    continue
                numbered = [s for s in route_dir.iterdir() if s.is_dir() and s.name.isdigit()]
                usable = [s for s in numbered if f"{chunk}/{route_dir.name}/{int(s.name)}" in found]
                unusable = len(numbered) - len(usable)
                if unusable > 0:
                    notes.append(
                        f"Chunk_{chunk}/{route_dir.name}: {unusable} segment(s) without a video.hevc "
                        "and a global_pose/frame_times, skipped"
                    )
                if usable:
                    routes.append((chunk, route_dir.name))
    return notes, routes


def _spot_check(segments: list[L.SegmentRef], n: int, rng: random.Random) -> tuple[int, list[str]]:
    """Stat `SEGMENT_FILES` on `n` segments spread across the chunks.

    A sample, not a sweep: `os.lstat` on 2035 segments x 4 files is ~8000 cold
    stats on a NAS, which is seconds, and `paddock doctor` calls this on every
    machine. A chunk that half-extracted shows up in a sample of 64; a single
    missing preview.png is not worth making everyone wait for.
    """
    if not segments:
        return 0, []
    step = max(1, len(segments) // max(n, 1))
    picked = segments[::step][:n]
    if len(picked) < min(n, len(segments)):
        picked = rng.sample(segments, min(n, len(segments)))
    missing: list[str] = []
    for seg in picked:
        absent = [f for f in SEGMENT_FILES if not (seg.path / f).exists()]
        if absent:
            missing.append(f"{seg.sample_key}: missing {', '.join(absent)}")
    return len(picked), missing


def survey(
    dataset: Dataset,
    deep: bool = False,
    segments: list[L.SegmentRef] | None = None,
    per_chunk: int = DEEP_PER_CHUNK,
) -> Survey:
    """What is on disk. See the module docstring for what each depth costs.

    Pass `segments` when the caller has already walked the tree -- `harness`
    holds the same list on a `cached_property` that `preflight` populates, and
    on a cold NAS that walk is seconds while everything after it is
    milliseconds. Omitted, the walk happens here.
    """
    segments = L.list_segments(dataset) if segments is None else segments
    found = {s.sample_key for s in segments}
    notes, routes = _anomalies(dataset, found)

    out = Survey(roots=list(dataset.roots), segments=segments, routes=routes, anomalies=notes)
    out.chunks = dict(sorted(Counter(s.chunk for s in segments).items()))
    out.vehicles = dict(Counter(s.vehicle for s in segments))
    rng = random.Random(SEED)
    out.checked, out.incomplete = _spot_check(segments, SAMPLE_SEGMENTS, rng)
    if deep:
        out.deep = _deep(segments, rng, per_chunk)
    return out


# --------------------------------------------------------------------------
# The expensive half
# --------------------------------------------------------------------------

#: on-disk log name -> (README name, value layout, unit). From the dataset's own
#: README and paper, not measured: the arrays are bare float matrices with no
#: column names anywhere on disk, so without this table a reader has no way to
#: know that radar column 5 is a track address. The deep pass checks the shapes
#: against it, which is how a layout change would be noticed.
LOG_SPEC: dict[str, tuple[str, str, str]] = {
    "processed_log/IMU/accelerometer": ("acceleration", "[forward, right, down]", "m/s^2"),
    "processed_log/IMU/gyro": ("gyro", "[forward, right, down]", "rad/s"),
    "processed_log/IMU/gyro_uncalibrated": ("gyro_uncalibrated", "[forward, right, down]", "rad/s"),
    "processed_log/IMU/gyro_bias": ("gyro_bias", "[forward, right, down]", "rad/s"),
    "processed_log/IMU/magnetometer": ("magnetic", "[forward, right, down]", "T"),
    "processed_log/IMU/magnetometer_uncalibrated": ("magnetic_uncalibrated", "[forward, right, down]", "T"),
    "processed_log/CAN/speed": ("car_speed", "[car_speed]", "m/s"),
    "processed_log/CAN/steering_angle": ("steering_angle", "scalar", "deg"),
    "processed_log/CAN/wheel_speed": (
        "wheel_speeds",
        "[front_left, front_right, rear_left, rear_right]",
        "m/s",
    ),
    "processed_log/CAN/radar": (
        "radar",
        "[fwd_dist, left_dist, rel_speed, nan, nan, address, new_track]",
        "m, m, m/s, -, -, -, bool",
    ),
    "processed_log/CAN/raw_can": ("raw CAN", "[src, address, data] (no value array)", "-"),
    "processed_log/GNSS/live_gnss_qcom": (
        "live_gnss_qcom",
        "[lat, lon, speed, utc_timestamp, altitude, bearing]",
        "deg, deg, m/s, s, m, deg",
    ),
    "processed_log/GNSS/live_gnss_ublox": (
        "live_gnss_ublox",
        "[lat, lon, speed, utc_timestamp, altitude, bearing]",
        "deg, deg, m/s, s, m, deg",
    ),
    "processed_log/GNSS/raw_gnss_qcom": (
        "raw_gnss_qcom",
        "[prn, gps_week, tow, nan, pseudorange, pseudorange_std, pseudorange_rate, pseudorange_rate_std]",
        "-, weeks, s, -, m, m, m/s, m/s",
    ),
    "processed_log/GNSS/raw_gnss_ublox": (
        "raw_gnss_ublox",
        "[prn, gps_week, tow, glonass_chan, pseudorange, pseudorange_std, pseudorange_rate, "
        "pseudorange_rate_std]",
        "-, weeks, s, -, m, m, m/s, m/s",
    ),
}

POSE_SPEC: dict[str, tuple[str, str]] = {
    "global_pose/frame_times": ("timestamps of video frames in boot time", "s"),
    "global_pose/frame_gps_times": ("[gps_week, time-of-week] of every frame", "weeks, s"),
    "global_pose/frame_positions": ("global camera position in ECEF", "m"),
    "global_pose/frame_velocities": ("global camera velocity in ECEF", "m/s"),
    "global_pose/frame_orientations": (
        "ECEF -> local camera frame [forward, right, down], hamilton quaternion",
        "-",
    ),
}


def _load(base: Path, key: str):
    path = base / key
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:  # raw_can/data is an object array; nothing here needs it
        return None


def _ffprobe() -> str | None:
    """ffmpeg's probe tool, looked for where `loader._ffmpeg` looks for ffmpeg.

    A site that set `PADDOCK_FFMPEG` to a binary inside some other project's
    venv has ffprobe beside it and almost certainly not on `PATH`, so the
    sibling is tried before `PATH` rather than after.
    """
    candidates = [os.environ.get("PADDOCK_FFPROBE")]
    ffmpeg = os.environ.get("PADDOCK_FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg:
        candidates.append(str(Path(ffmpeg).with_name("ffprobe")))
    candidates.append(shutil.which("ffprobe"))
    return next((c for c in candidates if c and Path(c).exists()), None)


def _probe_with_ffprobe(path: Path) -> dict[str, Any]:
    """The same fields as the PyAV branch, from `ffprobe -of json`.

    `-count_packets` demuxes the whole file, which is the same work PyAV's
    branch does and the reason the packet count is trustworthy: a truncated
    video reports fewer packets than its header's duration implies.
    """
    exe = _ffprobe()
    if exe is None:
        return {
            "error": "no video probe available; install PyAV (`pip install av`) "
            "or set PADDOCK_FFMPEG/PADDOCK_FFPROBE to an ffmpeg build"
        }
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=codec_name,profile,width,height,pix_fmt,nb_read_packets,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        stream = (json.loads(out.stdout).get("streams") or [{}])[0]
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not stream:
        return {"error": "ffprobe found no video stream"}

    num, _, den = (stream.get("avg_frame_rate") or "0/0").partition("/")
    rate = float(num) / float(den) if den and float(den) else None
    packets = stream.get("nb_read_packets")
    return {
        "codec": stream.get("codec_name"),
        "profile": str(stream.get("profile")),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pix_fmt": stream.get("pix_fmt"),
        "packets": int(packets) if packets is not None else None,
        "container_rate": rate,
        "probe": Path(exe).name,
    }


def probe_video(path: Path) -> dict[str, Any]:
    """Codec, geometry and packet count, via PyAV if it is here, else ffprobe.

    Same backend split as `loader.decode_frames`, and for the same reason:
    paddock's own environment bundles no codec stack, so a site that can decode
    at all can decode through one of these two. Reporting "PyAV not installed"
    at a site whose scoring path is happily using ffmpeg would describe the
    environment wrongly.

    Never raises. A corrupt or truncated video is exactly what this report is
    for, so it is a row in the output rather than a traceback over it.
    """
    try:
        import av
    except ImportError:
        return _probe_with_ffprobe(path)
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            codec = stream.codec_context
            packets = sum(1 for _ in container.demux(stream))
            return {
                "codec": codec.name,
                "profile": str(stream.profile),
                "width": codec.width,
                "height": codec.height,
                "pix_fmt": codec.pix_fmt,
                "packets": packets,
                # The header says 25 fps and the segment is 20 Hz. Reported so
                # the discrepancy is on the record rather than rediscovered.
                "container_rate": float(stream.average_rate) if stream.average_rate else None,
            }
    except Exception as exc:  # noqa: BLE001 - a bad video is a finding, not a crash
        return {"error": f"{type(exc).__name__}: {exc}"}


def inspect_segment(seg: L.SegmentRef) -> dict[str, Any]:
    """Load every array of one segment and summarise it. Seconds, not ms."""
    base = seg.path
    out: dict[str, Any] = {"segment": seg.sample_key, "arrays": {}, "rates": {}, "errors": []}

    keys = list(POSE_SPEC)
    keys += [f"{k}/{sub}" for k in LOG_SPEC for sub in ("t", "value")]
    keys += [f"processed_log/CAN/raw_can/{sub}" for sub in ("t", "src", "address")]
    for key in keys:
        array = _load(base, key)
        if array is None:
            continue
        out["arrays"][key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if key.endswith("/t"):
            out["rates"][key.rsplit("/", 1)[0]] = rate_hz(array)
        elif key == "global_pose/frame_times":
            out["rates"][key] = rate_hz(array)

    times = _load(base, "global_pose/frame_times")
    if times is not None and len(times) > 1:
        out["frames"] = int(len(times))
        out["duration_s"] = float(times[-1] - times[0])
        out["fps"] = rate_hz(times)

    gps = _load(base, "global_pose/frame_gps_times")
    if gps is not None and len(gps):
        out["gps_week"] = float(gps[0][0])

    for src in ("ublox", "qcom"):
        value = _load(base, f"processed_log/GNSS/live_gnss_{src}/value")
        if value is not None and len(value):
            out[f"live_gnss_{src}"] = {
                "lat": [float(np.nanmin(value[:, 0])), float(np.nanmax(value[:, 0]))],
                "lon": [float(np.nanmin(value[:, 1])), float(np.nanmax(value[:, 1]))],
                "alt": [float(np.nanmin(value[:, 4])), float(np.nanmax(value[:, 4]))],
                "speed": [float(np.nanmin(value[:, 2])), float(np.nanmax(value[:, 2]))],
            }

    raw = _load(base, "processed_log/GNSS/raw_gnss_ublox/value")
    if raw is not None and len(raw):
        out["sats_ublox"] = int(len(np.unique(raw[:, 0])))
        out["epochs_ublox"] = int(len(np.unique(raw[:, 2])))

    speed = _load(base, "processed_log/CAN/speed/value")
    if speed is not None and len(speed):
        flat = np.asarray(speed).reshape(len(speed), -1)[:, 0]
        out["speed_mps"] = [float(np.nanmin(flat)), float(np.nanmean(flat)), float(np.nanmax(flat))]

    steer = _load(base, "processed_log/CAN/steering_angle/value")
    if steer is not None and len(steer):
        flat = np.asarray(steer).reshape(len(steer), -1)[:, 0]
        out["steering_deg"] = [float(np.nanmin(flat)), float(np.nanmax(flat))]

    address = _load(base, "processed_log/CAN/raw_can/address")
    if address is not None and len(address):
        out["can_messages"] = int(len(address))
        out["can_addresses"] = Counter(int(a) for a in address)
        src = _load(base, "processed_log/CAN/raw_can/src")
        out["can_src"] = Counter(int(s) for s in src) if src is not None else Counter()

    radar = _load(base, "processed_log/CAN/radar/value")
    if radar is not None and len(radar):
        out["radar_points"] = int(len(radar))
        out["radar_tracks"] = int(len(np.unique(radar[:, 5])))
        out["radar_fwd_dist"] = [float(np.nanmin(radar[:, 0])), float(np.nanmax(radar[:, 0]))]

    if (base / "preview.png").exists():
        out["preview"] = png_header(base / "preview.png")
    if seg.video.exists():
        out["video"] = probe_video(seg.video)
    out["bytes"], out["files"] = _segment_bytes(base)
    return out


def _segment_bytes(base: Path) -> tuple[int, dict[str, int]]:
    total, files = 0, {}
    for dirpath, _, filenames in os.walk(base):
        rel_dir = os.path.relpath(dirpath, base)
        for name in filenames:
            try:
                size = os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            files[name if rel_dir == "." else f"{rel_dir}/{name}"] = size
            total += size
    return total, files


def _deep(segments: list[L.SegmentRef], rng: random.Random, per_chunk: int) -> dict[str, Any]:
    """Stratified by chunk, so a chunk whose extraction failed cannot hide."""
    by_chunk: dict[int, list[L.SegmentRef]] = {}
    for seg in segments:
        by_chunk.setdefault(seg.chunk, []).append(seg)
    picked: list[L.SegmentRef] = []
    for chunk in sorted(by_chunk):
        pool = by_chunk[chunk]
        picked += rng.sample(pool, min(per_chunk, len(pool)))
    inspections = [inspect_segment(seg) for seg in picked]
    return {
        "sampled": [s.sample_key for s in picked],
        "per_chunk": per_chunk,
        "seed": SEED,
        "inspections": inspections,
    }
