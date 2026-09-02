"""NeuroNCAP closed-loop benchmark.

Adapted from the pre-paddock ladder driver. A NeuRAD render of a real nuScenes
sequence is driven by the model under test while a staged actor is manoeuvred
into a collision course; the score is how much of the impact the model avoided.

Two constants that look like free choices and are not:

* **The simulation steps at 0.1 s.** openpilot needs 20 Hz input and reads a
  0.5 s step as a parked car. Every model is driven at the same rate so that
  control authority is equal; the scenario files under `scenarios_10hz/` encode
  it.
* **The waypoint grid stays at 0.5 s.** The collision lookahead and the
  reference-speed calculation both assume it. `--engine.waypoint-dt` exists to
  keep these two rates separate; see `patches/neuro-ncap/0001`.

The renderer is model-independent and expensive to start, so it is expected to
be running already and is shared across every model in a sweep. `preflight`
checks it and says how to start it.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from paddock.benchmark import ClosedLoopBenchmark, Episode

SCENARIOS_DIR = Path(__file__).parent / "scenarios_10hz"
WAYPOINT_DT = 0.5
DEFAULT_SCENARIOS = ("stationary", "frontal", "side")

_SCORE_LINE = re.compile(r"ncap_score:\s*([-\d.eE+]+)")


class NeuroNcapBenchmark(ClosedLoopBenchmark):
    # ---- configuration -------------------------------------------------

    @property
    def _settings(self):
        return self.cfg.benchmarks.get("neuroncap")

    def _setting(self, key: str) -> str | None:
        return self._settings.get(key) if self._settings else None

    def available_scenes(self) -> list[str]:
        """Scenes with both a scenario file and a renderer checkpoint."""
        checkpoints = self._setting("checkpoints")
        have_ckpt = (
            {p.name for p in Path(checkpoints).iterdir() if p.is_dir()}
            if checkpoints and Path(checkpoints).exists()
            else set()
        )
        scenes = set()
        for scenario_dir in SCENARIOS_DIR.iterdir():
            if scenario_dir.is_dir():
                scenes |= {p.stem for p in scenario_dir.glob("*.yaml")}
        return sorted(scenes & have_ckpt) if have_ckpt else sorted(scenes)

    # ---- preflight -----------------------------------------------------

    def preflight(self) -> list[str]:
        """Everything that must hold before a model is loaded.

        Checked here rather than discovered an hour into a sweep -- a missing
        renderer checkpoint for one scene is otherwise a late, confusing failure.
        """
        from paddock.runtime import proc

        problems: list[str] = []

        if self._settings is None:
            return [
                "no [benchmarks.neuroncap] section in paddock.toml; it needs `python`, "
                "`repo`, `renderer_url` and `checkpoints` (see docs/benchmarks/neuroncap.md)"
            ]

        for key in ("python", "repo", "renderer_url"):
            if not self._setting(key):
                problems.append(f"[benchmarks.neuroncap] {key} is not set in paddock.toml")

        python = self._setting("python")
        if python and not Path(python).exists():
            problems.append(f"neuroNCAP interpreter not found: {python}")

        repo = self._setting("repo")
        if repo and not (Path(repo) / "main.py").exists():
            problems.append(f"neuro-ncap checkout has no main.py: {repo}")

        dataset = self.dataset
        if dataset is None:
            problems.append("dataset 'nuscenes' is not configured in paddock.toml")
        elif not dataset.resolve().exists():
            problems.append(f"nuScenes root does not exist: {dataset.resolve()}")

        url = self._setting("renderer_url")
        if url and not proc.is_alive(url):
            problems.append(
                f"renderer is not answering at {url}. Start it first -- it is "
                f"model-independent and slow to start, so one instance serves a whole sweep:\n"
                f"    paddock renderer --scene <scene>"
            )
        return problems

    def scene_problems(self, scene: str) -> list[str]:
        """Per-scene checks: a scenario with no renderer checkpoint cannot run."""
        problems = []
        checkpoints = self._setting("checkpoints")
        if checkpoints and not (Path(checkpoints) / scene).exists():
            problems.append(
                f"scene {scene} has no renderer checkpoint under {checkpoints}; "
                f"available: {', '.join(self.available_scenes()) or 'none'}"
            )
        return problems

    def scenario_problems(self, scenarios: list[str]) -> list[str]:
        """A scenario name with no tracked directory would otherwise yield nothing, late."""
        known = sorted(p.name for p in SCENARIOS_DIR.iterdir() if p.is_dir())
        unknown = [s for s in scenarios if s not in known]
        if not unknown:
            return []
        return [f"--scenarios {', '.join(unknown)}: no such scenario set; known: {', '.join(known)}"]

    # ---- episodes ------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        for scene in scenes or ["0103"]:
            for scenario in scenarios or list(DEFAULT_SCENARIOS):
                path = SCENARIOS_DIR / scenario / f"{scene}.yaml"
                if not path.exists():
                    continue
                yield Episode(
                    episode_id=f"{scenario}-{scene}",
                    scenario=scenario,
                    scene=scene,
                    run_index=0,
                    config={"scenario_path": str(path), "runs": runs},
                )

    def run_episode(self, episode: Episode, policy_url: str, out_dir: Path) -> dict[str, float]:
        """Drive one scenario against a running policy node.

        `policy_url` is a node paddock has already started and health-checked.
        """
        repo = Path(self._setting("repo"))
        port = policy_url.rsplit(":", 1)[-1]
        renderer_port = self._setting("renderer_url").rsplit(":", 1)[-1]
        runs = int(episode.config.get("runs", 1))
        cameras = episode.config.get("cameras") or ["CAM_FRONT"]

        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self._setting("python"),
            "-u",
            "main.py",
            "--engine.renderer.port",
            renderer_port,
            "--engine.model.port",
            port,
            "--engine.dataset.data_root",
            str(self.dataset.resolve()),
            "--engine.dataset.version",
            self.dataset.version or "v1.0-trainval",
            "--engine.dataset.sequence",
            str(int(episode.scene)),
            "--engine.scenario.path",
            episode.config["scenario_path"],
            "--engine.waypoint-dt",
            str(WAYPOINT_DT),
            "--engine.cameras",
            *cameras,
            "--engine.logger.log-dir",
            str(out_dir),
            "--runs",
            str(runs),
        ]

        started = time.time()
        result = subprocess.run(
            command,
            cwd=repo,
            env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin"},
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - started
        (out_dir / "stdout.log").write_text(result.stdout)
        (out_dir / "stderr.log").write_text(result.stderr)

        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
            raise RuntimeError(
                f"{episode.episode_id} failed after {elapsed / 60:.1f} min "
                f"(exit {result.returncode}):\n{tail}"
            )

        metrics = self._collect(out_dir)
        metrics["duration_s"] = round(elapsed, 1)
        scores = [float(m.group(1)) for m in _SCORE_LINE.finditer(result.stdout)]
        if scores and "ncap_score" not in metrics:
            metrics["ncap_score"] = sum(scores) / len(scores)
        return metrics

    @staticmethod
    def _collect(out_dir: Path) -> dict[str, float]:
        """Read the metrics.json the engine wrote, for one run or several."""
        files = sorted(out_dir.rglob("metrics.json"))
        if not files:
            return {}
        scores, collisions = [], []
        for path in files:
            raw = json.loads(path.read_text())
            if raw.get("ncap_score") is not None:
                scores.append(float(raw["ncap_score"]))
            if raw.get("any_collide@0.0s") is not None:
                collisions.append(1.0 if raw["any_collide@0.0s"] else 0.0)
        out: dict[str, Any] = {"n_runs": len(files)}
        if scores:
            out["ncap_score"] = sum(scores) / len(scores)
        if collisions:
            out["collision_rate"] = sum(collisions) / len(collisions)
        return out


BENCHMARK = NeuroNcapBenchmark
