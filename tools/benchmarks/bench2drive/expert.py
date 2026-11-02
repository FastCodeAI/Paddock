"""Think2Drive's own drives of the Bench2Drive-220 routes, from the released dataset.

Bench2Drive's expert is Think2Drive, an RL policy that is not released as code.
Its drives are: the full dataset (``rethinklab/Bench2Drive-Full`` and
``-Full-Sup``, 13,638 clips listed in the checkout's
``docs/bench2drive_full+sup_13638.json``) holds a clip for 211 of the 220
evaluation route ids, with the same town and scenario. Two scenarios are named
differently there (``LaneChange`` for ``SequentialLaneChange``, ``TJunction``
for ``T_Junction``); nine routes have no clip at all.

So "the expert trajectory" of a route is that clip's ``anno/`` frames: the
expert's world pose, speed and controls at 10 Hz. A clip is a ~300-450 MB
tar.gz whose camera frames come before its annotations, so the archive is
streamed through and only ``anno/`` is kept (~1-5 MB a route); the stream's
SHA-256 is checked against the dataset's list as it passes.

    python -m tools.benchmarks.bench2drive.expert fetch --out <dir> [--routes 3144,1711] [--jobs 6]
    python -m tools.benchmarks.bench2drive.expert summarize --out <dir>

``fetch`` is resumable: a route whose ``source.json`` exists is skipped, and a
route is written to a temporary directory and renamed only once complete.
``summarize`` writes ``<dir>/<route_id>/expert_traj.json``, the compact
per-frame table the renderer and the analysis read.

The pose in ``expert_traj.json`` is CARLA's ground truth: the ``ego_vehicle``
entry of each frame's ``bounding_boxes`` (actor location, yaw in degrees, speed),
the same actor location a paddock run records. The annotation's top-level
``x``/``y`` are the GNSS sensor's -- 1.35 m off the ground truth on a sampled
frame, which drew the expert's path as a ±1 m zigzag -- and ``theta`` is the
compass; both are kept as ``gnss_x``/``gnss_y``/``theta`` for reference only.
All positions are CARLA world metres; aligning them with a paddock run is the
reader's job, because only it knows which frame it wants.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

HF_REPOS = ("rethinklab/Bench2Drive-Full", "rethinklab/Bench2Drive-Full-Sup")
CLIP_LIST = Path("docs") / "bench2drive_full+sup_13638.json"
ROUTES_XML = Path("leaderboard") / "data" / "bench2drive220.xml"

#: Scenario names the dataset spells differently from the routes file.
SCENARIO_ALIASES = {"LaneChange": "SequentialLaneChange", "TJunction": "T_Junction"}

#: The dataset's own capture rate (``tools/data_collect.py``: ``frame_rate = 10.0``).
EXPERT_HZ = 10.0

#: Per-frame annotation fields kept in ``expert_traj.json`` beside the ground-truth
#: pose. Everything else in a frame (bounding boxes of every actor, sensor
#: calibrations) stays in ``anno/``.
FRAME_FIELDS = (
    "theta",
    "speed",
    "throttle",
    "steer",
    "brake",
    "reverse",
    "command_near",
    "command_far",
    "next_command",
    "x_command_near",
    "y_command_near",
    "x_target",
    "y_target",
    "should_brake",
    "only_ap_brake",
)

_CLIP = re.compile(
    r"^(?P<scenario>\w+?)_(?P<town>Town\w+?)_Route(?P<route>\d+)_Weather(?P<weather>\d+)\.tar\.gz$"
)


def eval_routes(b2d_repo: Path) -> dict[str, dict[str, str]]:
    """``{route_id: {town, scenario}}`` for the 220 evaluation routes."""
    routes: dict[str, dict[str, str]] = {}
    for route in ET.parse(b2d_repo / ROUTES_XML).getroot().iter("route"):
        scenario = route.find("scenarios/scenario")
        routes[route.get("id") or ""] = {
            "town": route.get("town") or "",
            "scenario": scenario.get("type") if scenario is not None else "",
        }
    return routes


def match_clips(b2d_repo: Path) -> dict[str, dict[str, Any]]:
    """The dataset clip that drove each evaluation route, by route id.

    Matched on id *and* town and scenario, so a route id that happens to recur
    in another town cannot pull in the wrong drive.
    """
    clips = json.loads((b2d_repo / CLIP_LIST).read_text())
    routes = eval_routes(b2d_repo)
    found: dict[str, dict[str, Any]] = {}
    for name, meta in clips.items():
        m = _CLIP.match(name)
        if not m or m["route"] not in routes:
            continue
        want = routes[m["route"]]
        scenario = SCENARIO_ALIASES.get(m["scenario"], m["scenario"])
        if scenario == want["scenario"] and m["town"] == want["town"]:
            found[m["route"]] = {"clip": name, "weather": int(m["weather"]), **meta}
    return found


def hf_listing(cache: Path, token: str | None) -> dict[str, str]:
    """``{clip name: HF repo}`` for both dataset repos, cached in ``cache``."""
    if cache.exists():
        return json.loads(cache.read_text())
    where: dict[str, str] = {}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for repo in HF_REPOS:
        response = httpx.get(f"https://huggingface.co/api/datasets/{repo}", headers=headers, timeout=60)
        response.raise_for_status()
        for sibling in response.json().get("siblings", []):
            where.setdefault(sibling["rfilename"], repo)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(where))
    return where


class _Hashing:
    """A read-only stream that hashes and counts what passes through it."""

    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self._buffer = b""
        self.sha256 = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        while size < 0 or len(self._buffer) < size:
            chunk = next(self._chunks, None)
            if chunk is None:
                break
            self.sha256.update(chunk)
            self.bytes += len(chunk)
            self._buffer += chunk
        if size < 0:
            out, self._buffer = self._buffer, b""
        else:
            out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out

    def drain(self) -> None:
        """Read to the end, so the digest covers the whole archive."""
        while self.read(1 << 20):
            pass


def fetch_one(route_id: str, clip: dict[str, Any], repo: str, out: Path, token: str | None) -> dict:
    """Stream one clip and keep its ``anno/``; returns what was written to ``source.json``."""
    final = out / route_id
    if (final / "source.json").exists():
        return json.loads((final / "source.json").read_text())
    partial = out / f".{route_id}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "anno").mkdir(parents=True)

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{clip['clip']}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    started = time.time()
    frames = 0
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        stream = _Hashing(response.iter_bytes(1 << 20))
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                if member.isfile() and len(parts) >= 2 and parts[-2] == "anno":
                    handle = tar.extractfile(member)
                    if handle is not None:
                        (partial / "anno" / parts[-1]).write_bytes(handle.read())
                        frames += 1
        stream.drain()

    digest = stream.sha256.hexdigest()
    if clip.get("sha256") and digest != clip["sha256"]:
        shutil.rmtree(partial, ignore_errors=True)
        raise RuntimeError(f"{route_id}: {clip['clip']} sha256 {digest} != listed {clip['sha256']}")
    source = {
        "route_id": route_id,
        "repo": repo,
        "clip": clip["clip"],
        "weather": clip.get("weather"),
        "sha256": digest,
        "sha256_verified": bool(clip.get("sha256")),
        "archive_bytes": stream.bytes,
        "anno_frames": frames,
        "download_s": round(time.time() - started, 1),
    }
    (partial / "source.json").write_text(json.dumps(source, indent=1))
    shutil.rmtree(final, ignore_errors=True)
    partial.rename(final)
    return source


def summarize(route_dir: Path) -> dict | None:
    """``anno/*.json.gz`` -> the compact ``expert_traj.json`` beside it."""
    anno = sorted((route_dir / "anno").glob("*.json.gz"))
    if not anno:
        return None
    frames = []
    for path in anno:
        with gzip.open(path, "rt") as handle:
            raw = json.load(handle)
        frame: dict[str, Any] = {"frame": int(path.name.split(".")[0])}
        ego = next((b for b in raw.get("bounding_boxes") or [] if b.get("class") == "ego_vehicle"), None)
        if ego is not None:
            frame["x"], frame["y"] = round(ego["location"][0], 3), round(ego["location"][1], 3)
            frame["yaw"] = round(ego["rotation"][2], 3)
            frame["gt_speed"] = round(ego.get("speed", float("nan")), 3)
        else:  # no ground truth in this frame: fall back to GNSS, and say so
            frame["x"], frame["y"], frame["yaw"], frame["pose_source"] = (
                raw.get("x"),
                raw.get("y"),
                None,
                "gnss",
            )
        frame["gnss_x"], frame["gnss_y"] = raw.get("x"), raw.get("y")
        for key in FRAME_FIELDS:
            value = raw.get(key)
            if isinstance(value, float):
                value = round(value, 4)
            frame[key] = value
        frames.append(frame)
    source = json.loads((route_dir / "source.json").read_text())
    summary = {
        "route_id": source["route_id"],
        "clip": source["clip"],
        "hz": EXPERT_HZ,
        "n_frames": len(frames),
        "duration_s": round(len(frames) / EXPERT_HZ, 1),
        "frames": frames,
    }
    (route_dir / "expert_traj.json").write_text(json.dumps(summary))
    return summary


def load_expert(expert_root: Path, route_id: str) -> dict | None:
    """The route's ``expert_traj.json``, or None where the dataset has no drive of it."""
    path = Path(expert_root) / str(route_id) / "expert_traj.json"
    return json.loads(path.read_text()) if path.exists() else None


def _b2d_repo(given: str | None) -> Path:
    if given:
        return Path(given)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools import paths

    repo = paths.config().benchmarks["bench2drive"].get("repo")
    if not repo:
        raise SystemExit("no [benchmarks.bench2drive] repo in paddock.toml; pass --b2d-repo")
    return Path(repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("fetch", "summarize", "status"))
    parser.add_argument("--out", required=True, type=Path, help="one directory per route id is written here")
    parser.add_argument("--b2d-repo", help="Bench2Drive checkout (default: [benchmarks.bench2drive] repo)")
    parser.add_argument("--routes", help="comma-separated route ids (default: every evaluation route)")
    parser.add_argument("--jobs", type=int, default=6, help="clips streamed at once")
    args = parser.parse_args(argv)

    b2d = _b2d_repo(args.b2d_repo)
    clips = match_clips(b2d)
    wanted = [r.strip() for r in args.routes.split(",")] if args.routes else sorted(eval_routes(b2d), key=int)
    missing = [r for r in wanted if r not in clips]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "status":
        done = [r for r in wanted if (args.out / r / "source.json").exists()]
        print(
            f"{len(done)}/{len(wanted) - len(missing)} fetched; no expert clip for {len(missing)}: {missing}"
        )
        return 0

    if args.command == "summarize":
        n = 0
        for rid in wanted:
            if (args.out / rid / "source.json").exists() and summarize(args.out / rid):
                n += 1
        print(f"summarized {n} route(s)")
        return 0

    token = os.environ.get("HF_TOKEN")
    where = hf_listing(args.out / "_hf_listing.json", token)
    todo = [r for r in wanted if r in clips and not (args.out / r / "source.json").exists()]
    print(
        f"{len(todo)} to fetch, {len(wanted) - len(missing) - len(todo)} already here, no clip for {missing}"
    )
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for rid in todo:
            repo = where.get(clips[rid]["clip"])
            if repo is None:
                print(f"  {rid}: {clips[rid]['clip']} is in neither {HF_REPOS}", flush=True)
                failures += 1
                continue
            futures[pool.submit(fetch_one, rid, clips[rid], repo, args.out, token)] = rid
        for future in as_completed(futures):
            rid = futures[future]
            try:
                source = future.result()
                summarize(args.out / rid)
                print(
                    f"  {rid}: {source['anno_frames']} frames, {source['archive_bytes'] / 1e6:.0f} MB "
                    f"in {source['download_s']:.0f} s",
                    flush=True,
                )
            except Exception as exc:  # one bad clip must not stop the other 210
                failures += 1
                print(f"  {rid}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
