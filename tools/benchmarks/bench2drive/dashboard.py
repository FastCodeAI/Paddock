"""A live dashboard for a Bench2Drive campaign: results as they land, a video per route.

    python -m tools.benchmarks.bench2drive.dashboard serve --root <experiment_dir> --runs <runs_dir> \
        --expert <dir> --routes-xml <bench2drive220.xml> [--port 8765] [--every 60] [--jobs 3] \
        [--model NAME] [--title TEXT] [--routes 1711,1852]
    python -m tools.benchmarks.bench2drive.dashboard build ...   # one pass, no server

``serve`` does two things at once:

* every ``--every`` seconds, a builder pass: finds every attempt below ``--runs``
  (any number of paddock runs: shards, re-drives), takes the latest scored one per
  route (``runs.latest``), renders ``video.mp4`` / ``summary.png`` / ``route.json``
  for each finished route that has none yet (``route_video``, ``--jobs`` at a time,
  on the CPU), reads the routes still driving from their live files, and writes
  ``<root>/dashboard/data.json``;
* serves ``<root>`` over HTTP on 127.0.0.1 -- the dashboard is ``/dashboard/``,
  and the live frames of a route still driving are read straight out of its run
  directory, which is why the root is the experiment directory and not the
  dashboard's own. Byte ranges are honoured so a video can be scrubbed.

The page (``dashboard_assets/``) polls ``data.json`` and redraws in place, so a
browser left open follows the campaign; nothing needs restarting while it runs.
Bind to 127.0.0.1 and forward the port (VS Code's Ports view does it in one
click); the server has no authentication.
"""

from __future__ import annotations

import argparse
import calendar
import json
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from paddock.adapters.benchmarks.bench2drive.harness import STOP_REQUEST  # noqa: E402
from tools.benchmarks.bench2drive import route_video, runs  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "dashboard_assets"
EXPECTED_ROUTES = 220
#: Ticks per second of game time (CARLA's fixed step in the leaderboard).
TICK_HZ = 20.0

_GAME = re.compile(r"Game time = ([\d.]+) -- Ratio = ([\d.]+)x")


def outcome(att: runs.Attempt) -> str:
    """One word for how a route ended -- the dashboard's grouping, not the leaderboard's."""
    if att.result is None:
        return "Running" if att.running else "Incomplete"
    status = att.status
    if any(status.endswith(s) for s in runs.CRASH_SUFFIXES):
        return "Crashed"
    if status == "Perfect":
        return "Perfect"
    if status == "Completed":
        return "Completed"
    if att.stopped:  # ended through the tick cap, but by a rule (or a person), not by running out
        return f"Stopped: {att.stopped}"
    for key, name in (
        ("blocked", "Blocked"),
        ("deviated", "Deviated"),
        ("timed out", "Timed out"),
        ("TickRuntime", "Tick limit"),
    ):
        if key in status:
            return name
    return "Failed"


def success(result: dict | None) -> bool:
    """Bench2Drive's success: the route completed with no infraction but min-speed ones.

    ``tools/statistic_route_json.is_success`` / ``ability_benchmark.get_infraction_status`` at 7ec25d1.
    """
    if not result or result.get("status") not in ("Perfect", "Completed"):
        return False
    return not any(v for k, v in (result.get("infractions") or {}).items() if k != "min_speed_infractions")


def live_state(att: runs.Attempt, root: Path) -> dict[str, Any]:
    """What a route still driving looks like right now, from the files it is writing."""
    state: dict[str, Any] = {
        "route_id": att.route_id,
        "scenario": att.scenario,
        "town": att.town,
        "run_id": att.run_id,
        "stop_requested": (att.path / STOP_REQUEST).exists(),
    }
    log = att.path / "evaluator.log"
    if log.exists():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 4096))
            tail = handle.read().decode(errors="replace")
        hits = _GAME.findall(tail)
        if hits:
            state["game_time"], state["ratio"] = float(hits[-1][0]), float(hits[-1][1])
    ticks = att.path / "ticks.jsonl"
    if ticks.exists() and ticks.stat().st_size:
        with ticks.open("rb") as handle:
            handle.seek(max(0, ticks.stat().st_size - 65536))
            lines = handle.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            state["tick"] = rec["t"]
            state["speed"] = rec["ego"]["speed"]
            state["frac"] = (rec.get("route") or {}).get("frac")
            frame = f"{rec['t']:05d}.jpg"
            for view in ("front", "bev"):
                if (att.path / view / frame).exists():
                    state[view] = "/" + str((att.path / view / frame).relative_to(root))
            break
    return state


def route_row(att: runs.Attempt, meta: dict, out: Path, root: Path) -> dict[str, Any]:
    """One row of the routes table: the leaderboard's numbers and, once rendered, the video's."""
    result = att.result or {}
    scores = result.get("scores") or {}
    infractions: dict[str, int] = {}
    for kind, events in (result.get("infractions") or {}).items():
        if events and kind != "min_speed_infractions":
            infractions[kind] = len(events)
    row: dict[str, Any] = {
        "route_id": att.route_id,
        "town": att.town,
        "scenario": att.scenario,
        "abilities": meta.get("abilities") or runs.abilities(att.scenario),
        "length_m": meta.get("length_m"),
        "status": att.status,
        "outcome": outcome(att),
        "success": success(att.result),
        "ds": scores.get("score_composed"),
        "rc": scores.get("score_route"),
        "penalty": scores.get("score_penalty"),
        "infractions": infractions,
        "min_speed_entries": len((result.get("infractions") or {}).get("min_speed_infractions") or []),
        "game_s": (result.get("meta") or {}).get("duration_game"),
        "wall_s": (result.get("meta") or {}).get("duration_system"),
        "run_id": att.run_id,
        "dir": "/" + str(att.path.relative_to(root)),
        "early_stop": runs.load_json(att.path / "early_stop.json"),
    }
    summary = runs.load_json(out / att.route_id / "route.json")
    if summary and summary.get("source") == str(att.path):
        for key in (
            "mean_speed", "max_speed", "stopped_frac", "max_route_dev", "mean_infer_ms",
            "expert_duration_s", "expert_mean_speed", "expert_progress_pct", "has_expert", "ticks",
        ):  # fmt: skip
            row[key] = summary.get(key)
        rel = "/" + str((out / att.route_id).relative_to(root))
        row["video"] = rel + "/video.mp4"
        row["summary"] = rel + "/summary.png"
    return row


class Builder:
    """The periodic pass: render what finished, then rewrite data.json."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.root.resolve()
        self.out = self.root / "dashboard"
        self.routes_out = self.out / "routes"
        self.runs = [Path(p).resolve() for p in args.runs]
        self.expert = args.expert
        # The driver's name on the page and the videos. A sweep's runs live in
        # <runs>/bench2drive/<model>/, so --runs usually names it already.
        self.model = args.model or (self.runs[0].name if self.runs else "model")
        self.title = args.title or ""
        self.table = runs.route_table(args.routes_xml)
        if args.routes:
            # A campaign over a subset lists that subset, not 220 rows mostly "Queued".
            wanted = {r.strip() for r in args.routes.split(",") if r.strip()}
            unknown = sorted(wanted - set(self.table))
            if unknown:
                raise SystemExit(
                    f"--routes names route ids not in {args.routes_xml.name}: {', '.join(unknown)}"
                )
            self.table = {rid: meta for rid, meta in self.table.items() if rid in wanted}
        self.features = args.features
        # Spawned, not forked: a forked worker inherits the server's listening socket and holds
        # the port after the server is gone, so a restart fails with "address already in use".
        self.pool = ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn"))
        self.pending: dict[str, Future] = {}
        self.errors: dict[str, str] = {}
        self.lock = threading.Lock()

    def install_assets(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        for asset in ASSETS.iterdir():
            shutil.copy2(asset, self.out / asset.name)

    def _render_done(self, route_id: str, source: str, fut: Future) -> None:
        with self.lock:
            self.pending.pop(route_id, None)
            try:
                fut.result()
                self.errors.pop(route_id, None)
            except Exception as exc:  # shown on the page; retried after the route's files change
                self.errors[route_id] = f"{type(exc).__name__}: {exc}"
                marker = self.routes_out / route_id / ".failed"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(self.errors[route_id])
                print(f"[dashboard] render {route_id} failed: {exc}", flush=True)

    def needs_render(self, att: runs.Attempt) -> bool:
        """Every attempt that is over gets a video -- scored, stopped early, crashed or cut off --
        and gets it again if its outcome changed after it was drawn."""
        ticks = att.path / "ticks.jsonl"
        # An attempt cut off before its first tick was written has nothing to draw.
        if att.running or att.route_id in self.pending or not ticks.exists() or not ticks.stat().st_size:
            return False
        done = runs.load_json(self.routes_out / att.route_id / "route.json")
        status = (att.result or {}).get("status", "running")
        if not done or done.get("source") != str(att.path) or done.get("status") != status:
            return att.route_id not in self.errors or self._changed_since_error(att)
        return False

    def _changed_since_error(self, att: runs.Attempt) -> bool:
        marker = self.routes_out / att.route_id / ".failed"
        results = att.path / "results.json"
        return not marker.exists() or (results.exists() and results.stat().st_mtime > marker.stat().st_mtime)

    def request_stop(self, route_id: str) -> list[str]:
        """Ask every attempt of ``route_id`` that is driving now to end, scored as it stands.

        Writes the route's stop-request file (``harness.STOP_REQUEST``), which the
        agent checks every tick (``early_stop.MANUAL``). Returns the run ids asked.
        """
        asked = []
        for att in runs.attempts(self.runs):
            if att.route_id == route_id and att.running:
                (att.path / STOP_REQUEST).write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
                asked.append(att.run_id)
        return asked

    def load_features(self) -> dict[str, dict[str, str]]:
        """``--features`` by route id. Generic on purpose: this tool knows the benchmark, not the model."""
        if not self.features or not Path(self.features).exists():
            return {}
        import csv

        with Path(self.features).open() as handle:
            return {row["route_id"]: row for row in csv.DictReader(handle) if row.get("route_id")}

    def build(self) -> dict:
        started = time.time()
        everything = [a for a in (runs.attempts(self.runs) if self.runs else []) if a.route_id in self.table]
        chosen = runs.latest(everything)
        for att in chosen.values():
            if self.needs_render(att):
                fut = self.pool.submit(
                    route_video.render_route,
                    att.path,
                    self.expert,
                    self.routes_out,
                    self.table.get(att.route_id),
                    model=self.model,
                )
                with self.lock:
                    self.pending[att.route_id] = fut
                fut.add_done_callback(partial(self._render_done, att.route_id, str(att.path)))
        rows = [
            route_row(a, self.table.get(a.route_id, {}), self.routes_out, self.root) for a in chosen.values()
        ]
        running = [live_state(a, self.root) for a in everything if a.running]
        with self.lock:
            rendering, errors = sorted(self.pending), dict(self.errors)
        extra = self.load_features()
        for row in rows:
            if row["route_id"] in extra:
                row["features"] = extra[row["route_id"]]
                row["cause"] = extra[row["route_id"]].get("cause")
            row["rendering"] = row["route_id"] in rendering
            if row["route_id"] in errors:
                row["render_error"] = errors[row["route_id"]]
        # Routes not attempted yet, so the table always has the whole benchmark.
        for rid, meta in self.table.items():
            if rid not in chosen:
                rows.append(
                    {
                        "route_id": rid,
                        "town": meta["town"],
                        "scenario": meta["scenario"],
                        "abilities": meta["abilities"],
                        "length_m": meta["length_m"],
                        "outcome": "Queued",
                        "status": "not started",
                    }
                )
        finished = [a for a in chosen.values() if a.result is not None]
        data = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model": self.model,
            "title": self.title,
            "build_s": round(time.time() - started, 1),
            "expected_routes": len(self.table) or EXPECTED_ROUTES,
            "attempts": len(everything),
            "runs": sorted({a.run_id for a in everything}),
            "routes": sorted(rows, key=lambda r: int(r["route_id"])),
            "running": running,
            "rendering": rendering,
            "throughput": self.throughput(finished),
            # A written analysis a model's own tool may leave beside the dashboard (openpilot's
            # report writes analysis/REPORT.html); the page links it only when it is there.
            "report": (self.root / "analysis" / "REPORT.html").is_file(),
        }
        tmp = self.out / "data.json.tmp"
        tmp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False, default=str))
        os.replace(tmp, self.out / "data.json")
        return data

    @staticmethod
    def throughput(finished: list[runs.Attempt]) -> dict:
        """Routes finished per hour since the campaign started.

        Measured from the earliest run's start (its id is a UTC timestamp), not
        from the first finish: an hour in, the first finishes are minutes apart
        and a rate over just those overstates the sweep's pace several times.
        """
        if not finished:
            return {}
        starts = []
        for att in finished:
            try:
                starts.append(calendar.timegm(time.strptime(att.run_id[:15], "%Y%m%dT%H%M%S")))
            except ValueError:
                continue
        if not starts:
            return {"finished": len(finished)}
        span = max(time.time() - min(starts), 60.0)
        return {"finished": len(finished), "per_hour": round(3600 * len(finished) / span, 2)}


class Handler(SimpleHTTPRequestHandler):
    """Static files with byte ranges (video scrubbing) and no caching of data.json.

    One action besides reading: ``POST /api/stop?route=<id>`` asks the route
    driving now to end (:meth:`Builder.request_stop`). The server binds to
    127.0.0.1 only and has no authentication, so whoever can reach the port can
    end a route; that is the forwarded port's user.
    """

    builder: Builder | None = None

    def do_POST(self) -> None:  # noqa: N802 (the stdlib's name)
        url = urlparse(self.path)
        route = (parse_qs(url.query).get("route") or [""])[0]
        if url.path != "/api/stop" or not route.isdigit() or self.builder is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "POST /api/stop?route=<route id>")
            return
        asked = self.builder.request_stop(route)
        body = json.dumps({"route": route, "runs": asked, "ok": bool(asked)}).encode()
        self.send_response(HTTPStatus.OK if asked else HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"[dashboard] stop requested for route {route}: {asked or 'not driving'}", flush=True)

    def end_headers(self) -> None:
        if self.path.split("?")[0].endswith(".json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
        if self.path in ("/", ""):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().do_GET()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        size = path.stat().st_size
        if not m or (not m.group(1) and not m.group(2)):
            return super().do_GET()
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        else:  # suffix range: the last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        if start > end or start >= size:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = handle.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def log_message(self, fmt: str, *args) -> None:  # quiet: one line per request floods the log
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("serve", "build"))
    ap.add_argument(
        "--root", type=Path, required=True, help="served directory; dashboard/ is written inside it"
    )
    ap.add_argument(
        "--runs", type=Path, nargs="+", required=True, help="run directories, or directories of runs"
    )
    ap.add_argument("--expert", type=Path, help="expert.py's output directory")
    ap.add_argument("--routes-xml", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--every", type=float, default=60.0, help="seconds between builder passes")
    ap.add_argument("--jobs", type=int, default=3, help="routes rendered at once (CPU; the GPU is CARLA's)")
    ap.add_argument("--model", help="the driver's name on the page (default: the --runs directory's name)")
    ap.add_argument("--title", help="a line under the name, e.g. the checkpoint and how it was driven")
    ap.add_argument(
        "--routes", help="comma-separated route ids: the campaign's subset (default: every route)"
    )
    ap.add_argument(
        "--features",
        type=Path,
        help="a per-route CSV keyed by route_id (e.g. a model's analysis); its columns are shown beside each route",
    )
    args = ap.parse_args(argv)

    builder = Builder(args)
    builder.install_assets()
    if args.command == "build":
        data = builder.build()
        builder.pool.shutdown(wait=True)
        builder.build()
        print(f"{len(data['routes'])} routes in data.json")
        return 0

    def loop() -> None:
        while True:
            try:
                data = builder.build()
                done = sum(1 for r in data["routes"] if r.get("ds") is not None)
                print(
                    f"[dashboard] {data['generated_at']}: {done} scored, {len(data['running'])} running, "
                    f"{len(data['rendering'])} rendering",
                    flush=True,
                )
            except Exception:
                traceback.print_exc()
            time.sleep(args.every)

    threading.Thread(target=loop, daemon=True).start()
    Handler.builder = builder
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(builder.root)))
    print(f"[dashboard] http://{args.host}:{args.port}/dashboard/  (root {builder.root})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
