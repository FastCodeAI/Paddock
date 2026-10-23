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

import atexit
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.neuroncap import pdm_terms
from paddock.benchmark import ClosedLoopBenchmark, Episode
from paddock.metrics import pdm

SCENARIOS_DIR = Path(__file__).parent / "scenarios_10hz"
WAYPOINT_DT = 0.5
DEFAULT_SCENARIOS = ("stationary", "frontal", "side")
#: What `episodes()` uses when --scene is not given; preflight checks it too.
DEFAULT_SCENE = "0103"

#: The cameras NeuRAD (nuScenes) can render. A model may declare cameras a
#: closed-loop simulator synthesises but this dataset lacks -- openpilot's
#: comma-three rig (CAM_OP_ROAD / CAM_OP_WIDE) is real in Bench2Drive's CARLA
#: client but has no calibration here -- so declared cameras are intersected
#: with this set, falling back to CAM_FRONT, before they reach --engine.cameras.
NUSC_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def renderable_cameras(declared: list[str] | None) -> list[str]:
    """The declared cameras this dataset can render, falling back to CAM_FRONT.

    A model may declare cameras a closed-loop simulator synthesises but nuScenes
    lacks (openpilot's comma rig); those are dropped rather than passed to
    ``--engine.cameras``, where an unknown name fails NeuRAD calibration.
    """
    return [c for c in (declared or []) if c in NUSC_CAMERAS] or ["CAM_FRONT"]


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

# The engine prints all three but writes only ncap_score to metrics.json:
#     ncap_score: 3.14488,  impact_speed: 1.77743,  reference_speed: 8.31438
# The speeds are the score's own inputs -- it is 5*max(0, 1 - v_impact/v_ref) --
# so dropping them leaves a score nothing can be checked against. The legacy
# ingest recovers them from the ladder logs; the live path reads the stdout it
# already captures. The speed group is optional so a score-only line still matches.
_SCORE_LINE = re.compile(
    r"ncap_score:\s*([-\d.eE+]+)"
    r"(?:,\s*impact_speed:\s*([-\d.eE+]+),\s*reference_speed:\s*([-\d.eE+]+))?"
)
_RUN_DIR = re.compile(r"^run_(\d+)$")

#: How a site gets the drivable rasters the PDM terms need, said wherever one is missing.
PDM_MAP_HOWTO = (
    "build them with tools/benchmarks/nuscenes/build_pdm_geometry.py, which needs nuScenes' "
    "map expansion pack (maps/expansion/*.json, a separate download from nuscenes.org)"
)

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

#: A scene's hash grids take ~80 s to load; allow for a cold page cache.
RENDERER_STARTUP_S = 900


def _lidar_sample_data(path: Path) -> dict[str, str]:
    """sample_token -> LIDAR_TOP keyframe filename, read without loading the file.

    `sample_data.json` is 1.25 GB here and `json.loads` on it peaks around
    4.6 GB of RSS. That is an absurd cost for preflight -- the cheap check that
    runs before anything else, often on a box already busy with someone else's
    job -- so this reads it a line at a time instead and keeps only the two
    fields it needs: peak RSS 64 MB for the whole dataset.

    It is not faster. Reading the file whole takes about 4 s on an idle box
    against 6 s for this scan, and 20-30 s when the machine is loaded. Memory is
    what the trade buys, and it is the right way round: preflight failing for
    want of RAM would defeat the point of checking early.

    nuScenes writes these files pretty-printed, one field per line, which is
    what makes a scan this simple viable. The parse is deliberately narrow: a
    line that does not look like one of the three fields is skipped, and a
    record missing any of them is dropped rather than guessed at.

    A compact, single-line file yields nothing from that scan, and "nothing"
    here would mean "no scene has any lidar" -- preflight would then refuse
    every scene, confidently and wrongly. So an empty result on a non-empty file
    falls back to loading it whole: slow and memory-hungry, but right. Being
    wrong in the blocking direction is worse than being expensive.
    """
    out: dict[str, str] = {}
    token = filename = None
    keyframe = False
    with path.open() as handle:
        for line in handle:
            line = line.strip().rstrip(",")
            if line in ("{", "},", "}"):
                if keyframe and token and filename and filename.startswith("samples/LIDAR_TOP"):
                    out[token] = filename
                token = filename = None
                keyframe = False
            elif line.startswith('"sample_token"'):
                token = line.split('": "', 1)[-1].rstrip('"')
            elif line.startswith('"filename"'):
                filename = line.split('": "', 1)[-1].rstrip('"')
            elif line.startswith('"is_key_frame"'):
                keyframe = line.endswith("true")

    if not out and path.stat().st_size:
        return {
            d["sample_token"]: d["filename"]
            for d in json.loads(path.read_text())
            if d.get("is_key_frame") and str(d.get("filename", "")).startswith("samples/LIDAR_TOP")
        }
    return out


@lru_cache(maxsize=4)
def _lidar_keyframes_by_scene(meta: Path) -> dict[str, list[str]]:
    """scene name -> its LIDAR_TOP keyframe filenames, for one nuScenes metadata dir.

    Cached because a 14-scene sweep would otherwise redo the scan fourteen times.
    """
    scenes = {s["name"]: s for s in json.loads((meta / "scene.json").read_text())}
    samples = {s["token"]: s for s in json.loads((meta / "sample.json").read_text())}
    lidar = _lidar_sample_data(meta / "sample_data.json")
    out: dict[str, list[str]] = {}
    for name, scene in scenes.items():
        names, token = [], scene["first_sample_token"]
        while token:
            filename = lidar.get(token)
            if filename:
                names.append(filename)
            token = samples[token]["next"]
        out[name] = names
    return out


def _official_set_summary() -> str:
    return ", ".join(f"{k} {len(v)} scenes" for k, v in OFFICIAL_SET.items())


class NeuroNcapBenchmark(ClosedLoopBenchmark):
    # ---- configuration: `self.setting(key)` / `self.settings` from the base ----

    def __init__(self, spec, cfg) -> None:
        super().__init__(spec, cfg)
        self._renderer_proc: subprocess.Popen | None = None
        self._renderer_scene: str | None = None
        self._renderer_log_dir: Path | None = None
        self._pdm_areas: dict[str, pdm.DrivableArea | None] = {}
        self._pdm_categories: dict[str, str] | None = None
        # A sweep holds ~8 GB of VRAM on a card six other people share; leaving
        # it behind on a crash or a Ctrl-C is not acceptable.
        atexit.register(self._stop_renderer, lambda _msg: None)

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

        try:
            self.video_mode()
        except ValueError as exc:
            # Otherwise a typo raises from the first episode's run_episode, after
            # the node is up, and as a traceback -- the runner catches
            # RuntimeError, not ValueError.
            problems.append(str(exc))

        # The scenes a run will actually touch. `scene_problems` covers whatever
        # --scene names, but a run without it falls back to the default and
        # would never be checked -- the same asymmetry as verifying the 20
        # scenario files exist while not verifying their scenes can render.
        problems += self._lidar_problems(DEFAULT_SCENE)

        url = self.setting("renderer_url")
        if url and self.manages_renderer():
            # We start one per scene; a live one here would be someone else's,
            # and our port must be free rather than occupied.
            if proc.is_alive(url):
                problems.append(
                    f"manage_renderer is on but something is already answering at {url}. "
                    f"Stop it, or point renderer_url at a free port -- this harness will "
                    f"not take over or kill a renderer it did not start."
                )
            try:
                from paddock.adapters.benchmarks.neuroncap import launcher

                for scene in sorted({s for v in OFFICIAL_SET.values() for s in v}):
                    launcher.renderer_command(self.cfg, scene, int(url.rsplit(":", 1)[-1]))
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                problems.append(f"cannot start a renderer: {exc}")
        elif url and not proc.is_alive(url):
            problems.append(
                f"renderer is not answering at {url}. Start it first -- it is "
                f"model-independent and slow to start, so one instance serves a whole sweep:\n"
                f"    paddock renderer --scene <scene>"
            )
        return problems

    def invalid_reasons(self) -> list[str]:
        """Why a run under the current scope cannot be ranked against the others.

        A neuroNCAP figure means "the official 20 scenario-scene pairs". Anything
        narrower -- one scene for a smoke, one scenario while debugging -- is a
        different measurement wearing the same name, and stored `valid: true` it
        would average into the leaderboard as a full sweep. NAVSIM has refused a
        partial split this way from the start; this is the same guard.

        The run *count* is deliberately not checked. Runs are seeded 0..R-1, so
        10 of upstream's 50 is an unbiased subsample of the same protocol rather
        than a different scope -- see docs/benchmarks/neuroncap.md.
        """
        total = sum(len(v) for v in OFFICIAL_SET.values())
        ran = getattr(self, "_pairs_run", None)
        if ran is None or ran >= total:
            return []
        return [
            f"{ran} of the {total} official scenario-scene pairs; a subset, "
            f"not a neuroNCAP score ({_official_set_summary()})"
        ]

    def scene_problems(self, scene: str) -> list[str]:
        """Per-scene checks: what stops this scene rendering, before anything loads."""
        problems = []
        checkpoints = self.setting("checkpoints")
        if checkpoints and not (Path(checkpoints) / scene).exists():
            problems.append(
                f"scene {scene} has no renderer checkpoint under {checkpoints}; "
                f"available: {', '.join(self.available_scenes()) or 'none'}"
            )
        problems += self._lidar_problems(scene)
        return problems

    def _lidar_problems(self, scene: str) -> list[str]:
        """A scene with no lidar keyframe on disk cannot be rendered.

        The renderer never uses the points -- the scene is in the trained hash
        grids -- but NeuRAD's dataparser reads every lidar file eagerly, and
        `renderer.py` needs one real cloud to copy a tensor width from when it
        substitutes empty ones for the rest. With none it raises, and with the
        raise removed the dataparser falls over later instead: `len(lidars)` is
        0, so `_get_world_to_mean_transform` builds the transform from the
        cameras and `ad_dataparser.py:320` multiplies a Float against a Double.

        This is not hypothetical. The 2026-09-04 sweep spent about 13 GPU-hours
        finding it: this nuScenes copy was extracted with a camera-only filter,
        so 177 of 850 scenes have no lidar, three of them in the official 20
        pairs. Every attempt failed those three and retried the other 17.
        """
        dataset = self.dataset
        if dataset is None:
            return []
        root = Path(dataset.resolve())
        meta = root / (dataset.version or "v1.0-trainval")
        if not meta.exists():
            return []  # the dataset checks in preflight() cover a missing root

        try:
            names = self._lidar_keyframes(meta, scene)
        except (OSError, ValueError, KeyError):
            # Reading 850 scenes of metadata is not worth failing preflight over;
            # the run will say so itself if the data really is unusable.
            return []
        if names is None:
            return []
        if not any((root / n).exists() for n in names):
            return [
                f"scene {scene} has none of its {len(names)} LIDAR_TOP keyframes under {root}. "
                f"The renderer does not use the points, but NeuRAD's dataparser reads them "
                f"eagerly and needs at least one. To fetch just this scene's lidar without "
                f"re-downloading nuScenes, see the 'cameras but no lidar for some scenes' "
                f"recipe in docs/data-setup.md; or leave the scene out of --scene."
            ]
        return []

    @staticmethod
    def _lidar_keyframes(meta: Path, scene: str) -> list[str] | None:
        """The lidar keyframe filenames nuScenes lists for `scene`, or None if unknown."""
        return _lidar_keyframes_by_scene(meta).get(f"scene-{scene}")

    def scenario_problems(self, scenarios: list[str], scenes: list[str] | None = None) -> list[str]:
        """A named scenario, or a named (scenario, scene) pair, that cannot be run.

        Both checks live here rather than in `episodes()` because this runs
        before the policy node loads. `episodes()` still raises for a named pair
        with no file -- it is a generator and cannot be trusted to have been
        preflighted -- but by then a model is resident and the raise surfaces as
        a traceback, since the runner catches `RuntimeError`, not this.
        """
        known = sorted(p.name for p in SCENARIOS_DIR.iterdir() if p.is_dir())
        unknown = [s for s in scenarios if s not in known]
        if unknown:
            return [f"--scenarios {', '.join(unknown)}: no such scenario set; known: {', '.join(known)}"]

        missing = [
            f"{scenario}/{scene}"
            for scene in scenes or []
            for scenario in scenarios
            if not (SCENARIOS_DIR / scenario / f"{scene}.yaml").exists()
        ]
        if missing:
            return [
                f"no scenario file for {', '.join(missing)}. Not every scene is in every "
                f"category -- the official set is {_official_set_summary()}, so e.g. 0099 is "
                f"stationary-only. Drop --scenarios to run whichever categories a scene is in."
            ]
        return []

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
        self._pairs_run = 0
        for scene in scenes or [DEFAULT_SCENE]:
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
                self._pairs_run += 1
                yield Episode(
                    episode_id=f"{scenario}-{scene}",
                    scenario=scenario,
                    scene=scene,
                    run_index=0,
                    config={"scenario_path": str(path), "runs": runs},
                )

    def run_episode(
        self,
        episode: Episode,
        policy_url: str,
        out_dir: Path,
        echo: Callable[[str], None] = print,
    ) -> dict[str, float]:
        """Drive one scenario against a running policy node.

        `policy_url` is a node paddock has already started and health-checked.
        """
        # The renderer serves one scene; a multi-scene sweep restarts it here so
        # the whole official set is one run rather than one run per scene.
        self._renderer_log_dir = out_dir.parent
        self._ensure_renderer(episode.scene, echo)

        repo = Path(self.setting("repo"))
        port = policy_url.rsplit(":", 1)[-1]
        renderer_port = self.setting("renderer_url").rsplit(":", 1)[-1]
        runs = int(episode.config.get("runs", 1))
        # Keep only cameras this dataset can render; a model that also declares
        # simulator-only cameras (openpilot's comma rig) falls back to CAM_FRONT
        # rather than failing NeuRAD calibration on a name nuScenes never had.
        cameras = renderable_cameras(episode.config.get("cameras"))

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
            # The vendor logger's `allowed_cams` is `tuple[str]`, fixed arity
            # one -- tyro's CLI can take exactly one STR or the literal
            # `None`, never a multi-camera list. `None` logs every camera the
            # engine has images for (i.e. whatever `--engine.cameras` above
            # already restricted it to); the field's own default of
            # `("CAM_FRONT",)` would otherwise silently drop every other
            # camera a multi-camera model used. See the PR description for
            # the full investigation.
            "--engine.logger.allowed-cams",
            "None",
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
        self.render_videos(out_dir, echo)
        return metrics

    # ---- renderer ------------------------------------------------------

    def manages_renderer(self) -> bool:
        """Whether this harness starts and stops renderers itself.

        Off by default: the usual single-scene workflow is `paddock renderer
        --scene X` in one shell and `paddock run` in another, and the renderer
        outliving the run is the point -- it costs ~80 s to load and is shared
        across models.

        A sweep over the full official set cannot work that way. The renderer
        loads one scene's hash grids at startup and cannot switch, so 14 scenes
        means 14 renderers. Driving them from outside would mean one `paddock
        run` per scene, and then every run covers 1-3 of the 20 pairs and is
        correctly recorded `valid: false` -- 28 partial runs and no rankable
        score. With this on, one invocation covers the whole set.
        """
        raw = os.environ.get("PADDOCK_NCAP_MANAGE_RENDERER") or self.setting("manage_renderer", "false")
        return str(raw).strip().lower() in ("true", "1", "yes")

    def _ensure_renderer(self, scene: str, echo: Callable[[str], None] = print) -> None:
        """Make the renderer on our port serve `scene`, starting one if needed.

        Only ever stops a renderer this harness started. One left running by
        `paddock renderer` belongs to whoever started it -- on a shared box that
        may well be another person's sweep.
        """
        if not self.manages_renderer() or getattr(self, "_renderer_scene", None) == scene:
            return

        from paddock.adapters.benchmarks.neuroncap import launcher
        from paddock.runtime import proc

        self._stop_renderer(echo)

        url = self.setting("renderer_url")
        port = int(url.rsplit(":", 1)[-1])
        command, cwd, env = launcher.renderer_command(self.cfg, scene, port)
        echo(f"  renderer: loading scene {scene}")
        # Its own stdout is large and mostly nerfstudio noise; keep it beside the
        # run rather than interleaved with the episode lines.
        log = self._renderer_log_dir / f"renderer_{scene}.log" if self._renderer_log_dir else None
        handle = open(log, "w") if log else subprocess.DEVNULL  # noqa: SIM115
        self._renderer_proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle, stderr=handle)
        self._renderer_scene = scene

        deadline = time.time() + RENDERER_STARTUP_S
        while time.time() < deadline:
            if self._renderer_proc.poll() is not None:
                self._renderer_scene = None
                raise RuntimeError(
                    f"renderer for scene {scene} exited with {self._renderer_proc.returncode}"
                    + (f"; see {log}" if log else "")
                )
            if proc.is_alive(url):
                echo(f"  renderer: scene {scene} ready")
                return
            time.sleep(5)
        self._stop_renderer(echo)
        raise RuntimeError(f"renderer for scene {scene} did not answer within {RENDERER_STARTUP_S}s")

    def _stop_renderer(self, echo: Callable[[str], None] = print) -> None:
        """Stop only a renderer we started; leave anyone else's alone."""
        running = getattr(self, "_renderer_proc", None)
        if running is None:
            return
        running.terminate()
        try:
            running.wait(timeout=30)
        except subprocess.TimeoutExpired:
            running.kill()
            running.wait(timeout=10)
        self._renderer_proc = None
        self._renderer_scene = None

    def close(self) -> None:
        """Stop the renderer this harness started, once the runner is done with it.

        The renderer otherwise lives until exit (the `atexit` above). That was
        enough while one harness served a whole sweep, but a sweep split by a
        pairing table builds one harness per model, and the next one's
        preflight then finds this renderer still answering and refuses.
        """
        self._stop_renderer(lambda _msg: None)

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

    def completed_metrics(self, episode: Episode, out_dir: Path, runs: int = 1) -> dict[str, float] | None:
        """The episode's metrics if all `runs` of it are on disk, else None.

        The engine writes each run's `metrics.json` only after that run has been
        scored, so a file present is a run finished. Counting them is therefore
        the whole check: fewer than asked for means the episode was interrupted
        part-way and has to be driven again, since a mean over the runs that
        happened to complete is a different measurement.
        """
        found = self._per_run(out_dir)
        if len(found) < max(1, runs):
            return None
        return self._collect(out_dir)

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
        terms = self.pdm_terms(out_dir, episode.scene)
        for index, run_metrics in per_run:
            run_metrics.update(terms.get(index, {}))
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

    # ---- PDM terms ---------------------------------------------------------

    def pdm_geometry(self) -> Path:
        """Where the drivable rasters are: `[benchmarks.neuroncap] pdm_geometry`,
        else `{roots.runs}/geometry/nuscenes-drivable`, which
        `tools/benchmarks/nuscenes/build_pdm_geometry.py` builds."""
        raw = self.setting("pdm_geometry")
        return Path(raw).expanduser() if raw else self.cfg.runs / "geometry" / "nuscenes-drivable"

    def _pdm_area(self, scene: str) -> pdm.DrivableArea | None:
        if scene not in self._pdm_areas:
            path = self.pdm_geometry() / f"scene-{scene}.npz"
            self._pdm_areas[scene] = pdm.DrivableArea.load(path) if path.is_file() else None
            if self._pdm_areas[scene] is None:
                print(
                    f"  no drivable raster {path}: no PDM terms for scene {scene}; {PDM_MAP_HOWTO}",
                    flush=True,
                )
        return self._pdm_areas[scene]

    def pdm_maps_missing(self) -> list[str]:
        """The official scenes with no drivable raster, which therefore record no PDM terms."""
        scenes = sorted({s for v in OFFICIAL_SET.values() for s in v})
        return [s for s in scenes if not (self.pdm_geometry() / f"scene-{s}.npz").is_file()]

    def status(self) -> list[tuple[str, str]]:
        """For `paddock doctor`: whether every official scene has the map its PDM terms need."""
        if self.settings is None:
            return []
        missing, where = self.pdm_maps_missing(), self.pdm_geometry()
        total = len({s for v in OFFICIAL_SET.values() for s in v})
        if not missing:
            return [("PDM maps", f"all {total} scenes in {where}")]
        return [
            (
                "PDM maps",
                f"{len(missing)} of {total} scenes have none in {where} ({', '.join(missing)}): "
                f"they record no PDM terms; {PDM_MAP_HOWTO}",
            )
        ]

    def _categories(self) -> dict[str, str]:
        """nuScenes instance token -> category, read once; {} (every actor an agent) if unreadable."""
        if self._pdm_categories is None:
            try:
                dataset = self.dataset
                self._pdm_categories = pdm_terms.load_categories(
                    dataset.resolve(), dataset.version or "v1.0-trainval"
                )
            except (AttributeError, OSError, KeyError, ValueError) as exc:
                print(f"  nuScenes categories unreadable ({exc}); every actor counts as an agent", flush=True)
                self._pdm_categories = {}
        return self._pdm_categories

    def pdm_terms(self, out_dir: Path, scene: str) -> dict[int, dict[str, float]]:
        """NAVSIM's PDM terms per run of an episode, from its stored files (`pdm_terms.py`).

        Never fatal: a run whose terms cannot be computed keeps its NCAP metrics
        and records no PDM terms, which the gate reads as not recorded. They are
        report-only, and an episode costs minutes of simulator time.
        """
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir() and _RUN_DIR.match(p.name)] or [out_dir]
        out: dict[int, dict[str, float]] = {}
        try:
            area, categories = self._pdm_area(scene), self._categories()
        except Exception as exc:  # noqa: BLE001 -- report-only terms must never cost an episode
            print(f"  PDM terms not computed for {out_dir.name}: {exc!r}", flush=True)
            return {}
        for run_dir in run_dirs:
            match = _RUN_DIR.match(run_dir.name)
            # Per run: one unreadable file costs that run its terms, not the episode.
            try:
                terms = pdm_terms.run_terms(run_dir, area, categories)
            except Exception as exc:  # noqa: BLE001 -- report-only terms must never cost an episode
                print(f"  PDM terms not computed for {out_dir.name}/{run_dir.name}: {exc!r}", flush=True)
                continue
            if terms:
                out[int(match.group(1)) if match else 0] = terms
        return out

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
            try:
                raw = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                # A run killed mid-write leaves a half-written metrics.json, and
                # that is precisely the state `--resume` exists to recover from.
                # Skipping it makes the run count as missing, so the episode is
                # driven again -- which is both the safe answer and the correct
                # one. Raising here would kill the resume on the first episode it
                # inspected, and would fail a *fresh* run into the same directory
                # too, since `_collect` and `episode_samples` read through here.
                continue
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
        out.sort(key=lambda item: item[0])
        return NeuroNcapBenchmark._with_speeds(out, out_dir)

    @staticmethod
    def _with_speeds(
        runs: list[tuple[int, dict[str, float]]], out_dir: Path
    ) -> list[tuple[int, dict[str, float]]]:
        """Fill in impact_speed / reference_speed from the captured stdout.

        The engine prints them per run and writes neither to metrics.json, so
        without this a stored row has a score and no way to see what produced it.

        Matched by position: the engine drives runs sequentially and prints one
        line each, so line i belongs to run i. If the counts disagree -- a run
        crashed, or the log was truncated -- the join is abandoned rather than
        guessed, because a speed attributed to the wrong run is worse than none.
        """
        log = out_dir / "stdout.log"
        if not runs or not log.exists():
            return runs
        parsed = [m.groups() for m in _SCORE_LINE.finditer(log.read_text())]
        if len(parsed) != len(runs):
            return runs
        for (_, record), (score, impact, reference) in zip(runs, parsed, strict=True):
            if impact is None or reference is None:
                continue
            # The score is in both places at full precision; disagreement means
            # the positional assumption is wrong, so leave that run alone.
            if "ncap_score" in record and abs(record["ncap_score"] - float(score)) > 1e-9:
                continue
            record.setdefault("impact_speed", float(impact))
            record.setdefault("reference_speed", float(reference))
        return runs

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
