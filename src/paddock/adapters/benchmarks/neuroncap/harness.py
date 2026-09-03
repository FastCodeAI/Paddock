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
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from paddock.benchmark import ClosedLoopBenchmark, Episode

SCENARIOS_DIR = Path(__file__).parent / "scenarios_10hz"
WAYPOINT_DT = 0.5
DEFAULT_SCENARIOS = ("stationary", "frontal", "side")

# The official evaluation set, transcribed from the vendor's own job arrays in
# `scripts/arrays/ncap_slurm_array_*.txt`: 20 scenario-scene pairs over 14
# distinct scenes. Sequence 0061 is in the stationary array carrying the comment
# "mini set (do not use for eval)" and is deliberately absent here.
#
# Tracked as data, not inferred from the directory listing, so that a file going
# missing is a test failure rather than a quietly shorter sweep.
OFFICIAL_SET: dict[str, tuple[str, ...]] = {
    "stationary": ("0099", "0101", "0103", "0106", "0108", "0278", "0331", "0783", "0796", "0966"),
    "frontal": ("0103", "0106", "0110", "0346", "0923"),
    "side": ("0103", "0108", "0110", "0278", "0921"),
}

_SCORE_LINE = re.compile(r"ncap_score:\s*([-\d.eE+]+)")
_RUN_DIR = re.compile(r"^run_(\d+)$")

# The engine already writes a frame per step per run; these are the two derived
# streams worth watching. FC_TRAJ is the front camera with the planned
# trajectory drawn on it -- what the model saw and what it decided -- and
# COMBINED_OUTPUTS is the multi-panel figure. The raw CAM_* jpegs are the
# renderer's own output and are left alone: they are small, and together with
# trajectories.json / ego_poses.json / actors.json they are enough to rebuild
# either derived stream offline, which is why pruning them is safe.
VIDEO_STREAMS = ("FC_TRAJ", "COMBINED_OUTPUTS")
#: The simulation steps at 0.1 s, so 10 fps plays back in real time. Worth
#: keeping: the question these clips answer is whether a model reacted in time.
VIDEO_FPS = 10
#: Hold the last frame ~1.5 s so the outcome is readable rather than a flicker.
VIDEO_HOLD_FRAMES = 15
VIDEO_MODES = ("encode", "keep", "off")


def _official_set_summary() -> str:
    return ", ".join(f"{k} {len(v)} scenes" for k, v in OFFICIAL_SET.items())


class NeuroNcapBenchmark(ClosedLoopBenchmark):
    # ---- configuration: `self.setting(key)` / `self.settings` from the base ----

    def available_scenes(self) -> list[str]:
        """Scenes with both a scenario file and a renderer checkpoint."""
        checkpoints = self.setting("checkpoints")
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

        if self.settings is None:
            return [
                "no [benchmarks.neuroncap] section in paddock.toml; it needs `python`, "
                "`repo`, `renderer_url` and `checkpoints` (see docs/benchmarks/neuroncap.md)"
            ]

        for key in ("python", "repo", "renderer_url"):
            if not self.setting(key):
                problems.append(f"[benchmarks.neuroncap] {key} is not set in paddock.toml")

        python = self.setting("python")
        if python and not Path(python).exists():
            problems.append(f"neuroNCAP interpreter not found: {python}")

        repo = self.setting("repo")
        if repo and not (Path(repo) / "main.py").exists():
            problems.append(f"neuro-ncap checkout has no main.py: {repo}")

        dataset = self.dataset
        if dataset is None:
            problems.append("dataset 'nuscenes' is not configured in paddock.toml")
        elif not dataset.resolve().exists():
            problems.append(f"nuScenes root does not exist: {dataset.resolve()}")

        missing = [
            f"{scenario}/{scene}.yaml"
            for scenario, scene_ids in OFFICIAL_SET.items()
            for scene in scene_ids
            if not (SCENARIOS_DIR / scenario / f"{scene}.yaml").exists()
        ]
        if missing:
            problems.append(
                f"{len(missing)} of the 20 official scenario files are missing from "
                f"{SCENARIOS_DIR.name}/: {', '.join(missing)}. A sweep would score the "
                f"rest and record it as a sweep."
            )

        url = self.setting("renderer_url")
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
        checkpoints = self.setting("checkpoints")
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
        # An explicit --scenarios means the caller named the pairs it wants, so a
        # missing file is an error. Defaulted, it means "whatever this scene is
        # in", and a gap is the benchmark's own shape: the official set puts 10
        # scenes in stationary and 5 each in frontal and side, so scene 0099 has
        # a stationary file and no other. Skipping that is right; skipping a
        # pair the caller asked for by name is how a sweep silently shrinks.
        explicit = scenarios is not None
        for scene in scenes or ["0103"]:
            for scenario in scenarios or list(DEFAULT_SCENARIOS):
                path = SCENARIOS_DIR / scenario / f"{scene}.yaml"
                if not path.exists():
                    if explicit:
                        raise FileNotFoundError(
                            f"--scenarios {scenario} --scene {scene}: no scenario file at "
                            f"{path.relative_to(SCENARIOS_DIR.parent)}. The official set is "
                            f"{_official_set_summary()}."
                        )
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
        repo = Path(self.setting("repo"))
        port = policy_url.rsplit(":", 1)[-1]
        renderer_port = self.setting("renderer_url").rsplit(":", 1)[-1]
        runs = int(episode.config.get("runs", 1))
        cameras = episode.config.get("cameras") or ["CAM_FRONT"]

        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.setting("python"),
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
        if self.video_mode() == "off":
            # No frames at all: skips the per-step jpeg and matplotlib writes,
            # which is most of what the logger costs.
            command += ["--engine.logger.enabled", "False"]

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
        self.render_videos(out_dir)
        return metrics

    # ---- debugging video -----------------------------------------------

    def video_mode(self) -> str:
        """`encode` (default), `keep`, or `off` -- from `[benchmarks.neuroncap] video`.

        Deliberately *not* in `run_config()`: what is kept for debugging does not
        change what was measured, and putting it there would give the same sweep
        two different run ids depending on whether videos were wanted.
        """
        mode = (self.setting("video") or "encode").strip().lower()
        if mode not in VIDEO_MODES:
            raise ValueError(
                f"[benchmarks.neuroncap] video = {mode!r}: expected one of {', '.join(VIDEO_MODES)}"
            )
        return mode

    def _ffmpeg(self) -> str | None:
        """`[benchmarks.neuroncap] ffmpeg`, then `PADDOCK_FFMPEG`, then PATH."""
        for candidate in (self.setting("ffmpeg"), os.environ.get("PADDOCK_FFMPEG")):
            if candidate and Path(candidate).exists():
                return candidate
        return shutil.which("ffmpeg")

    def render_videos(self, out_dir: Path, echo: Callable[[str], None] = print) -> list[Path]:
        """One mp4 per stream per run, written beside that run's frames.

        Runs are the unit because runs are what differ: each is a different draw
        of the scenario jitter, so `run_3/FC_TRAJ.mp4` is the clip of the draw
        that `run_3` scored, and the two sit in one directory.

        Never fatal. Video is a debugging aid, not a result -- a missing ffmpeg
        or a failed encode leaves the frames in place and says so, rather than
        losing an episode that already cost minutes of simulator time.
        """
        mode = self.video_mode()
        if mode == "off":
            return []

        ffmpeg = self._ffmpeg()
        if ffmpeg is None:
            echo(
                "    no ffmpeg found; frames kept, no video. Set [benchmarks.neuroncap] "
                "ffmpeg or PADDOCK_FFMPEG to a binary, or video = 'keep' to silence this."
            )
            return []

        # `run_<i>/` when the engine drove several runs, the episode dir itself
        # when it drove one -- the same shape `_per_run` reads.
        run_dirs = sorted(
            (p for p in out_dir.iterdir() if p.is_dir() and _RUN_DIR.match(p.name)),
            key=lambda p: int(_RUN_DIR.match(p.name).group(1)),
        ) or [out_dir]

        written: list[Path] = []
        for run_dir in run_dirs:
            for stream in VIDEO_STREAMS:
                frames_dir = run_dir / stream
                if not frames_dir.is_dir():
                    continue
                video = self._encode(ffmpeg, frames_dir, run_dir / f"{stream}.mp4", echo)
                if video is None:
                    continue
                written.append(video)
                if mode == "encode":
                    # Only the stream that actually encoded, so a partial
                    # failure never deletes frames that have no video.
                    shutil.rmtree(frames_dir, ignore_errors=True)
        return written

    @staticmethod
    def _encode(ffmpeg: str, frames_dir: Path, dest: Path, echo: Callable[[str], None]) -> Path | None:
        """Frames to one mp4. Returns the path, or None if there was nothing to do."""
        # Filenames are microsecond timestamps, so lexical order is chronological.
        frames = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
        if not frames:
            return None

        # concat demuxer rather than a glob pattern: the timestamps are not a
        # contiguous numeric sequence, which -pattern_type would require.
        listing = frames_dir.parent / f".{frames_dir.name}.concat"
        lines = [f"file '{f}'" for f in frames]
        lines += [f"file '{frames[-1]}'"] * VIDEO_HOLD_FRAMES
        listing.write_text("\n".join(lines) + "\n")

        result = subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-r",
                str(VIDEO_FPS),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                # Bounded: shakaal fails at >=90% box CPU, and libx264 would
                # otherwise take every core it can see.
                "-threads",
                "2",
                # H.264 needs even dimensions; the matplotlib figures are not.
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
        listing.unlink(missing_ok=True)

        if result.returncode != 0 or not dest.exists():
            tail = (result.stderr or "").strip().splitlines()[-3:]
            echo(f"    video {dest.name} failed; frames kept: {' '.join(tail)}")
            return None
        return dest

    def samples_per_episode(self, runs: int = 1) -> int:
        """One sample per run -- see `episode_samples`."""
        return max(1, runs)

    def episode_samples(
        self, episode: Episode, metrics: dict[str, float], out_dir: Path
    ) -> list[dict[str, Any]]:
        """One sample per *run*, not per episode.

        A neuroNCAP run is not a repeat: the engine seeds the scenario jitter
        with the run index (`engine.run(i)` in the vendor's `main.py`), so each
        run draws a different actor start frame, position, size and identity.
        The spread across runs is therefore a property of the model under the
        scenario distribution, and it is the thing the mean has to be read
        against -- alpamayo's sd on frontal-0103 is 1.82 on a 0-5 scale.

        Recording only the episode mean would put `n` at 20 for a full sweep
        where 200 episodes were driven, and would leave no way to recover the
        spread from `results/`, which is the only part of a run that is tracked.
        """
        per_run = self._per_run(out_dir)
        if not per_run:
            # Nothing parseable on disk: fall back to whatever run_episode
            # scraped from stdout rather than dropping the episode entirely.
            return super().episode_samples(episode, metrics, out_dir)
        return [
            {
                # `run_<i>`, matching results/ingest/neuroncap.py exactly: the
                # legacy August runs are in the store under that shape, and a
                # live run has to be comparable with them sample for sample.
                "sample_id": f"{self.spec.name}/{episode.scene}/{episode.scenario}/run_{index}",
                "native_id": {
                    "scenario": episode.scenario,
                    "scene": episode.scene,
                    "run": index,
                },
                "split": episode.scenario,
                "metrics": run_metrics,
            }
            for index, run_metrics in per_run
        ]

    @staticmethod
    def _per_run(out_dir: Path) -> list[tuple[int, dict[str, float]]]:
        """Every run's metrics.json under `out_dir`, in run order.

        The engine writes `run_<i>/metrics.json` when driven with `--runs > 1`
        and a bare `metrics.json` when driven with one. Ordering is by the parsed
        integer, not the path: sorted lexically, `run_10` falls between `run_1`
        and `run_2`, which would misattribute every run index above nine.
        """
        out: list[tuple[int, dict[str, float]]] = []
        for path in out_dir.rglob("metrics.json"):
            raw = json.loads(path.read_text())
            match = _RUN_DIR.match(path.parent.name)
            index = int(match.group(1)) if match else 0
            record: dict[str, float] = {}
            for key in ("ncap_score", "impact_speed", "reference_speed"):
                if raw.get(key) is not None:
                    record[key] = float(raw[key])
            if raw.get("any_collide@0.0s") is not None:
                record["collision_rate"] = 1.0 if raw["any_collide@0.0s"] else 0.0
            if record:
                out.append((index, record))
        return sorted(out, key=lambda item: item[0])

    @classmethod
    def _collect(cls, out_dir: Path) -> dict[str, float]:
        """The episode's headline: the mean over its runs, for the console line.

        The per-run records are what reach `results/` -- see `episode_samples`.
        """
        per_run = cls._per_run(out_dir)
        if not per_run:
            return {}
        out: dict[str, Any] = {"n_runs": len(per_run)}
        for key in ("ncap_score", "collision_rate"):
            values = [m[key] for _, m in per_run if key in m]
            if values:
                out[key] = sum(values) / len(values)
        return out


BENCHMARK = NeuroNcapBenchmark
