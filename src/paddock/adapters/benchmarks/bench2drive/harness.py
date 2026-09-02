"""Bench2Drive closed-loop benchmark (CARLA 0.9.15 / Thinklab-SJTU leaderboard).

Bench2Drive drives the model through CARLA on the official leaderboard's 220 short
routes (Towns 12/13/15, 44 scenario types x 5). The model runs behind paddock's
HTTP policy node; this harness spawns the leaderboard evaluator once per route,
points its AutonomousAgent (``agent.py``, model-agnostic) at that node, and reads
the driving score CARLA writes to ``results.json``.

Closed-loop, like neuroNCAP: the model decides what happens next and errors
compound. The shape mirrors ``neuroncap/harness.py`` deliberately --

* ``preflight`` checks the CARLA install, the checkout and the routes file, so a
  missing asset is reported in seconds rather than an hour into a sweep;
* ``episodes`` enumerates routes from the leaderboard routes XML;
* ``run_episode`` shells out to ``leaderboard_evaluator.py`` with the node URL and
  parses the ``results.json`` it produces.

CARLA and the Bench2Drive checkout are vendored outside the package; the
``[benchmarks.bench2drive]`` section of ``paddock.toml`` says where they live here
(see ``docs/benchmarks/bench2drive.md``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from paddock.benchmark import ClosedLoopBenchmark, Episode

#: The model-agnostic CARLA client shipped with this adapter. The evaluator loads
#: it via ``--agent``; it reads the node URL from PADDOCK_NODE_URL and the cameras
#: to render from PADDOCK_CAMERAS, both set by ``run_episode`` below.
AGENT = Path(__file__).parent / "agent.py"

#: Default per-route wall-clock budget. Long routes ride CARLA's 4000-frame cap,
#: and a slow model (Alpamayo runs ~0.05x real-time) needs a generous ceiling.
DEFAULT_ROUTE_TIMEOUT_S = 1800

#: The CLI's default ``--scene`` ("0103") is a neuroNCAP scene id, meaningless
#: here; treat it -- and the explicit "all" -- as "no route filter".
_NO_SCENE_FILTER = {"0103", "all"}


def parse_routes(routes_xml: Path) -> list[tuple[str, str, str]]:
    """``(route_id, town, scenario_type)`` for every route in a leaderboard XML.

    Pure, so the golden test can enumerate a fixture with no CARLA involved.
    """
    tree = ET.parse(routes_xml)
    routes: list[tuple[str, str, str]] = []
    for route in tree.getroot().findall("route"):
        rid = route.get("id") or ""
        town = route.get("town") or ""
        scenario = route.find("scenarios/scenario")
        stype = scenario.get("type") if scenario is not None else "route"
        routes.append((rid, town, stype))
    return routes


def select_routes(
    routes: list[tuple[str, str, str]],
    scenes: list[str] | None = None,
    scenarios: list[str] | None = None,
    subset: list[str] | None = None,
) -> Iterator[tuple[str, str, str]]:
    """Filter routes by ``--scene`` (route id or town), ``--scenarios`` (type) and
    an optional site ``subset`` (route ids, from ``[benchmarks.bench2drive]
    routes_subset``) that restricts the run to a chosen set -- e.g. one route per
    scenario category -- without editing the routes file.

    Pure and separate from :meth:`Bench2DriveBenchmark.episodes` so the selection
    logic is unit-testable without a site config.
    """
    subset_ids = set(subset) if subset else None
    scene_filter = None if not scenes or set(scenes) <= _NO_SCENE_FILTER else set(scenes)
    scenario_filter = set(scenarios) if scenarios else None
    for rid, town, stype in routes:
        if subset_ids is not None and rid not in subset_ids:
            continue
        if scene_filter is not None and rid not in scene_filter and town not in scene_filter:
            continue
        if scenario_filter is not None and stype not in scenario_filter:
            continue
        yield rid, town, stype


def parse_results(results_json: Path) -> dict[str, float]:
    """A Bench2Drive ``results.json`` -> paddock metrics for one route.

    Reads the per-route record (``records[0]``), not the aggregate
    ``global_record``. Pure, so the golden test asserts it reproduces the score
    CARLA recorded.
    """
    checkpoint = json.loads(Path(results_json).read_text()).get("_checkpoint", {})
    records = checkpoint.get("records") or []
    if not records:
        raise ValueError(f"{results_json}: no records in _checkpoint (the route did not score)")
    record = records[0]
    scores = record.get("scores", {})
    infractions = record.get("infractions", {})

    def count(key: str) -> int:
        value = infractions.get(key, [])
        return len(value) if isinstance(value, list) else int(value or 0)

    collisions = count("collisions_vehicle") + count("collisions_pedestrian") + count("collisions_layout")
    return {
        "driving_score": round(float(scores.get("score_composed", 0.0)), 4),
        "route_completion": round(float(scores.get("score_route", 0.0)), 4),
        "infraction_penalty": round(float(scores.get("score_penalty", 0.0)), 4),
        # Per-route 0/1; paddock aggregates it to a rate across routes, as neuroNCAP does.
        "collision_rate": 1.0 if collisions > 0 else 0.0,
    }


class Bench2DriveBenchmark(ClosedLoopBenchmark):
    # ---- configuration -------------------------------------------------

    @property
    def _settings(self):
        return self.cfg.benchmarks.get("bench2drive")

    def _setting(self, key: str, default: str | None = None) -> str | None:
        return self._settings.get(key, default) if self._settings else default

    def _repo(self) -> Path | None:
        repo = self._setting("repo")
        return Path(repo) if repo else None

    def _routes_file(self) -> Path | None:
        repo = self._repo()
        if repo is None:
            return None
        return repo / "leaderboard" / "data" / (self._setting("routes_file", "bench2drive220.xml") or "")

    def _evaluator(self) -> Path | None:
        repo = self._repo()
        if repo is None:
            return None
        return repo / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"

    # ---- preflight -----------------------------------------------------

    def preflight(self) -> list[str]:
        """Everything that must hold before a model is loaded, checked cheaply."""
        problems: list[str] = []
        if self._settings is None:
            return [
                "no [benchmarks.bench2drive] section in paddock.toml; it needs `python`, "
                "`carla_root` and `repo` (see docs/benchmarks/bench2drive.md)"
            ]

        for key in ("python", "carla_root", "repo"):
            if not self._setting(key):
                problems.append(f"[benchmarks.bench2drive] {key} is not set in paddock.toml")

        python = self._setting("python")
        if python and not Path(python).exists():
            problems.append(f"CARLA-client interpreter not found: {python}")

        carla_root = self._setting("carla_root")
        if carla_root and not (Path(carla_root) / "CarlaUE4.sh").exists():
            problems.append(f"CARLA 0.9.15 not found at carla_root (no CarlaUE4.sh): {carla_root}")

        evaluator = self._evaluator()
        if evaluator and not evaluator.exists():
            problems.append(f"Bench2Drive checkout has no leaderboard_evaluator.py: {evaluator}")

        routes = self._routes_file()
        if routes and not routes.exists():
            problems.append(f"routes file not found: {routes}")
        return problems

    # ---- episodes ------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        routes_file = self._routes_file()
        if routes_file is None or not routes_file.exists():
            return
        subset_setting = self._setting("routes_subset")
        subset = [s.strip() for s in subset_setting.split(",") if s.strip()] if subset_setting else None
        for rid, town, stype in select_routes(parse_routes(routes_file), scenes, scenarios, subset):
            yield Episode(
                episode_id=f"{rid}-{town}-{stype}",
                scenario=stype,
                scene=rid,
                run_index=0,
                config={"route_id": rid, "town": town, "runs": runs},
            )

    # ---- run one episode ----------------------------------------------

    def run_episode(self, episode: Episode, policy_url: str, out_dir: Path) -> dict[str, float]:
        """Drive one route through CARLA against a running policy node.

        ``policy_url`` is a node paddock has already started and health-checked.
        The evaluator self-launches CARLA (reading CARLA_ROOT), so nothing else
        needs starting here; the agent talks to the node over HTTP.
        """
        repo = self._repo()
        carla_root = self._setting("carla_root")
        python = self._setting("python")
        leaderboard = repo / "leaderboard"
        scenario_runner = repo / "scenario_runner"
        cameras = episode.config.get("cameras") or ["CAM_FRONT"]
        timeout_s = int(
            self._setting("per_route_timeout_s", str(DEFAULT_ROUTE_TIMEOUT_S)) or DEFAULT_ROUTE_TIMEOUT_S
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        results_json = out_dir / "results.json"

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": os.environ.get("HOME", str(Path.home())),
            "CARLA_ROOT": str(carla_root),
            "SCENARIO_RUNNER_ROOT": str(scenario_runner),
            "LEADERBOARD_ROOT": str(leaderboard),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(carla_root) / "PythonAPI" / "carla"),
                    str(scenario_runner),
                    str(leaderboard),
                    str(leaderboard / "team_code"),
                ]
            ),
            # The model-agnostic agent reads these two.
            "PADDOCK_NODE_URL": policy_url,
            "PADDOCK_CAMERAS": ",".join(cameras),
            # Record front/BEV frames + a per-tick log for later visualisation.
            "REC_OUT": str(out_dir),
            "SAVE_PATH": str(out_dir),  # relaxes the 3m sensor-radius limit for the BEV cam
        }
        if os.environ.get("DISPLAY"):
            env["DISPLAY"] = os.environ["DISPLAY"]

        command = [
            str(python),
            str(self._evaluator()),
            "--routes",
            str(self._routes_file()),
            "--routes-subset",
            episode.config["route_id"],
            "--repetitions",
            "1",
            "--track",
            "SENSORS",
            "--checkpoint",
            str(results_json),
            "--agent",
            str(AGENT),
            "--port",
            str(self._setting("carla_port", "2000")),
            "--traffic-manager-port",
            str(self._setting("tm_port", "8000")),
            "--gpu-rank",
            str(self._setting("gpu_rank", "0")),
        ]

        started = time.time()
        timed_out = False
        try:
            result = subprocess.run(
                command, cwd=leaderboard, env=env, capture_output=True, text=True, timeout=timeout_s
            )
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = (
                exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            )
            stderr = (
                exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
            )
            returncode = -1
        elapsed = time.time() - started

        (out_dir / "stdout.log").write_text(stdout or "")
        (out_dir / "stderr.log").write_text(stderr or "")
        self._cleanup_carla()

        if not results_json.exists():
            tail = "\n".join((stderr or stdout or "").strip().splitlines()[-15:])
            reason = "timed out" if timed_out else f"exit {returncode}"
            raise RuntimeError(
                f"{episode.episode_id}: no results.json after {elapsed / 60:.1f} min ({reason})\n{tail}"
            )

        # A route that ran but produced no scored record (crash, disconnect, empty
        # checkpoint) must fail only this episode -- raise RuntimeError so the runner
        # records it as failed and moves on, rather than letting a ValueError abort
        # the whole sweep.
        try:
            metrics = parse_results(results_json)
        except ValueError as exc:
            raise RuntimeError(f"{episode.episode_id}: {exc}") from exc
        metrics["duration_s"] = round(elapsed, 1)
        return metrics

    @staticmethod
    def _cleanup_carla() -> None:
        """CARLA is self-launched by the evaluator and can outlive a killed run.

        Kill by exact process name; a ``pkill -f`` on a pattern that appears in
        this process's own command line would take the harness down with it.
        """
        subprocess.run(["pkill", "-x", "CarlaUE4-Linux-"], stderr=subprocess.DEVNULL)
        time.sleep(3)


#: Paddock looks for this name when loading an adapter.
BENCHMARK = Bench2DriveBenchmark
