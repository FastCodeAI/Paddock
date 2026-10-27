"""HUGSIM closed-loop benchmark (3D Gaussian splatting scenes, released scenarios).

HUGSIM (hyzhou404/HUGSIM) renders six cameras from a Gaussian-splatting reconstruction of a real drive,
inserts scripted vehicles, and moves the ego with its own iLQR controller and a kinematic bicycle, 0.25 s a
step. The model plans; the simulator decides what happens next and scores the episode. Errors compound, as on
Bench2Drive.

Paddock drives it the way the Bench2Drive harness drives CARLA, one process per scenario:

* ``preflight`` checks the HUGSIM checkout and its patches, the interpreter that runs it, and the data;
* ``episodes`` enumerates the released scenario files, minus the sixteen the release itself cannot run
  (:data:`EXCLUDED`);
* ``run_episode`` starts ``launcher.py`` in HUGSIM's interpreter. It runs upstream's simulation loop, with
  paddock's ``bridge.py`` as the AD client, which turns each simulator step into a ``/infer`` on the model
  node and the plan back into HUGSIM's waypoints. Then it reads the ``eval.json`` HUGSIM writes.

What the harness fixes about the protocol, and records in ``run.json``:

* **Command only.** The bridge sends HUGSIM's route command (right, left, straight) and no route geometry.
* **The model's own rate.** HUGSIM steps at 4 Hz; a model that declares ``planning_hz`` is called at that rate
  and its last plan is held in between (Drive-JEPA: every second step). One that declares none is called every
  step.
* **The controller fix.** Upstream's ``traj2control`` gives a straight plan a reference heading of about +90
  deg, so the iLQR steers it right (4.3 m in 14 m on nuScenes scene-0013). ``patches/HUGSIM/0001`` is upstream
  PR #57, open and unmerged, which corrects it. Every run here uses it.
* **Only the model's cameras are rendered** (``patches/HUGSIM/0002``); the others are sent black. Scoring
  never reads the images, so the scores are unchanged.

HUGSIM, its interpreter and its data live outside the package; ``[benchmarks.hugsim]`` and
``[datasets.hugsim]`` in ``paddock.toml`` say where (see ``docs/benchmarks/hugsim.md``).
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.hugsim.bridge import STEP_DT, infer_every
from paddock.benchmark import ClosedLoopBenchmark, Episode
from paddock.config import REPO_ROOT
from paddock.runtime import gpu
from paddock.runtime.env import PASSTHROUGH
from paddock.runtime.proc import tail as _tail

LAUNCHER = Path(__file__).with_name("launcher.py")

#: The four source datasets of the release, each a directory under ``scenarios/`` and ``scenes/``.
DATASETS = ("kitti360", "nuscenes", "pandaset", "waymo")

#: Upstream ends an episode after this many steps (``closed_loop.create_gym_env``: ``cnt > 400``), 100 s.
STEP_CAP = 400

#: Wall clock for one scenario. A 400-step episode of a model at 2 Hz took up to ~4 min on the site measured;
#: the budget bounds a hang, so it is generous. Override per site with ``per_episode_timeout_s``.
DEFAULT_EPISODE_TIMEOUT_S = 3600

#: Free VRAM to wait for before starting a HUGSIM process (which peaks near 4.4 GB on PandaSet), so a scenario
#: does not start on a card another job is about to fill. Override per site with ``min_free_gb``.
DEFAULT_MIN_FREE_GB = 12.0

#: Scenarios the release cannot run for any model, so they are not part of the benchmark here. Fifteen
#: PandaSet scenarios give a scripted car ``ConstantPlanner`` a ``max_t`` argument the released code does not
#: take (``TypeError`` while the environment is built); one Waymo scenario has ``start_velo: 1R`` where its
#: siblings have ``1`` (``TypeError: new(): invalid data type 'str'``). Both fail before the first step, with
#: any client.
EXCLUDED: dict[str, str] = {
    **{
        f"scene-{n}-medium-02": "ConstantPlanner(max_t=25): the released ConstantPlanner takes no max_t"
        for n in (
            "021",
            "027",
            "028",
            "032",
            "033",
            "034",
            "038",
            "039",
            "040",
            "044",
            "053",
            "090",
            "095",
            "113",
            "124",
        )
    },
    "scene-140045460035-hard-00": "start_velo: 1R (a typo for 1): the ego state is built from a string",
}

#: What each patch under patches/HUGSIM/ leaves in the checkout, checked by `preflight`: (file, text, patch).
PATCH_MARKERS = (
    (
        "sim/utils/sim_utils.py",
        "for i, (b, a) in enumerate(plan_traj):",
        "0001-traj2control-heading-pr57.patch",
    ),
    ("sim/hugsim_env/envs/hug_sim.py", "HUGSIM_RENDER_CAMS", "0002-render-only-requested-cameras.patch"),
)

#: What the bridge and the launcher import, checked in HUGSIM's interpreter by `preflight` so a missing
#: package fails before the model loads. ``closed_loop`` brings the simulator itself (torch, gsplat, the gym
#: env).
CLIENT_IMPORTS = (
    "numpy",
    "yaml",
    "requests",
    "omegaconf",
    "closed_loop",
    "paddock.policy.protocol",
    "paddock.adapters.benchmarks.hugsim.bridge",
)

#: How ``HUGSimEnv.step`` says an episode ended. It prints exactly one of these when it terminates; an episode
#: that printed none ran into the step cap or stopped for another reason. Checked in this order, as the report
#: that first tabulated them did.
ENDINGS = (
    ("Complete", "completed"),
    ("Collision with background", "hit scenery"),
    ("Collision with foreground", "hit a vehicle"),
    ("Far from preset trajectory", "off route"),
)
COLLISION_ENDINGS = frozenset({"hit scenery", "hit a vehicle"})

_SCENE_NAME = re.compile(r"^scene_name:\s*'?([^'\s]+)'?\s*$", re.M)
_MODE = re.compile(r"^mode:\s*'?([^'\s]+)'?\s*$", re.M)
_HD_MAP = re.compile(r"^load_HD_map:\s*true\s*$", re.M | re.I)


class NotScored(ValueError):
    """HUGSIM wrote no score for the episode."""


# --------------------------------------------------------------------------- Pure helpers, tested without
# HUGSIM ---------------------------------------------------------------------------


def scenario_meta(text: str) -> tuple[str, str]:
    """``(scene_name, mode)`` of a scenario file.

    The file name is not enough: ``kitti360/scene-1126_1326-medium-00.yaml`` holds an easy scenario (``mode:
    easy_00``), the release's one mismatch, and HUGSIM names its outputs after ``mode``.
    """
    scene, mode = _SCENE_NAME.search(text), _MODE.search(text)
    if scene is None or mode is None:
        raise ValueError("not a HUGSIM scenario: no scene_name or mode")
    return scene.group(1), mode.group(1)


def difficulty(mode: str) -> str:
    """``medium_01`` -> ``medium``: the release's four difficulty levels."""
    return mode.split("_")[0]


def parse_eval(eval_json: Path) -> dict[str, float]:
    """HUGSIM's ``eval.json`` -> paddock metrics for one episode.

    HUGSIM scores every plan the client sent (``sim/utils/score_calculator.py``): no collision, drivable area,
    time to collision and comfort, and per plan PDMS = NC x DAC x (5 TTC + 2 C) / 7 -- NAVSIM's composite
    without its progress term. ``pdms`` is the mean over the episode's plans, ``rc`` the furthest route
    completion reached (at most 1) and ``hdscore`` = ``pdms`` x ``rc``. ``route_completion`` is stored in
    percent, paddock's unit. Pure, so the golden test asserts it against a recorded file.
    """
    try:
        e = json.loads(Path(eval_json).read_text())
    except FileNotFoundError as exc:
        raise NotScored(f"{eval_json}: no eval.json (HUGSIM did not finish the episode)") from exc
    missing = [k for k in ("nc", "dac", "ttc", "c", "pdms", "rc", "hdscore") if k not in e]
    if missing:
        raise NotScored(f"{eval_json}: no {', '.join(missing)}")
    values = {k: float(e[k]) for k in ("nc", "dac", "ttc", "c", "pdms", "rc", "hdscore")}
    if not all(math.isfinite(v) for v in values.values()):
        raise NotScored(f"{eval_json}: a score is not finite: {values}")
    return {
        "hd_score": values["hdscore"],
        "hugsim_pdms": values["pdms"],
        "route_completion": 100.0 * values["rc"],
        "hugsim_nc": values["nc"],
        "hugsim_dac": values["dac"],
        "hugsim_ttc": values["ttc"],
        "hugsim_comfort": values["c"],
    }


def ending(log_lines: set[str], steps: int | None) -> str:
    """How the episode ended, from the simulator's own termination message in its log (:data:`ENDINGS`).

    Exact lines only: the scorer also prints lower-case "collision with ..." for any *planned* trajectory that
    would collide, which says nothing about how the episode ended.
    """
    for message, name in ENDINGS:
        if message in log_lines:
            return name
    if (steps or 0) >= STEP_CAP:
        return "step cap"
    return "other"


def step_counts(agent_log: Path) -> dict[str, int]:
    """Steps the bridge saw, model calls it made, and steps it failed, from its ``agent_log.jsonl``."""
    if not agent_log.exists():
        return {}
    rows = [json.loads(line) for line in agent_log.read_text().splitlines() if line.strip()]
    return {
        "steps": len(rows),
        "calls": sum(bool(r.get("called")) for r in rows),
        "bridge_errors": sum("error" in r for r in rows),
    }


def episode_record(
    eval_json: Path, sim_log: Path, agent_log: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """``(metrics, extra)`` for one finished episode: the scores, and how it ended.

    ``collision_rate`` is 1 for an episode that ended in a collision, so its mean over a run is the fraction
    of episodes that did. Shared by the harness and the ingest reader, so a run driven here and a recorded
    campaign read the same way.
    """
    metrics = parse_eval(eval_json)
    counts = step_counts(agent_log)
    lines = (
        {x.strip() for x in sim_log.read_text(errors="ignore").splitlines()} if sim_log.exists() else set()
    )
    how = ending(lines, counts.get("steps"))
    metrics["collision_rate"] = 1.0 if how in COLLISION_ENDINGS else 0.0
    return metrics, {"ending": how, **counts}


def patch_problems(repo: Path) -> list[str]:
    """Why the checkout at ``repo`` is not HUGSIM with patches/HUGSIM/ applied."""
    problems = []
    for relative, marker, patch in PATCH_MARKERS:
        path = repo / relative
        if not path.exists():
            problems.append(f"HUGSIM checkout is missing {path}")
        elif marker not in path.read_text(errors="replace"):
            problems.append(f"{path} lacks patches/HUGSIM/{patch}; apply it (patches/README.md)")
    return problems


def select(
    scenarios: list[tuple[str, str, str]],
    scenes: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Filter ``(dataset, stem, scene_name)`` by ``--scene``: a scenario file stem, a dataset or a scene.

    ``scenes=None`` means every scenario. Excluded scenarios are dropped before this sees them.
    """
    if not scenes:
        return list(scenarios)
    wanted = set(scenes)
    return [s for s in scenarios if s[0] in wanted or s[1] in wanted or s[2] in wanted]


def kill_group(proc: subprocess.Popen) -> None:
    """Take the launcher and everything in its session (simulator, bridge) down, TERM then KILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


# --------------------------------------------------------------------------- The benchmark
# ---------------------------------------------------------------------------


class HugsimBenchmark(ClosedLoopBenchmark):
    # ---- site settings -------------------------------------------------

    @property
    def repo(self) -> Path | None:
        value = self.setting("repo")
        return Path(value) if value else None

    @property
    def data_root(self) -> Path | None:
        dataset = self.dataset
        return dataset.resolve() if dataset is not None else None

    def _scenarios(self) -> list[tuple[str, str, str]]:
        """``(dataset, stem, scene_name)`` for every released scenario not in :data:`EXCLUDED`, sorted."""
        root = self.data_root
        if root is None:
            return []
        found = []
        for ds in DATASETS:
            for path in sorted((root / "scenarios" / ds).glob("*.yaml")):
                if path.stem in EXCLUDED:
                    continue
                scene, _ = scenario_meta(path.read_text())
                found.append((ds, path.stem, scene))
        return found

    def _environment(self, extra: dict[str, str]) -> dict[str, str]:
        """HUGSIM's environment: the host passthrough set, plus ours.

        Built like ``runtime.env.build``, not inherited: ``REPO_ROOT/src`` first, so the bridge imports the
        wire contract, then the checkout, where the simulator's own packages are.
        """
        env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(self.repo)])
        # HUGSIM's renderer runs fine on four threads; unset, numpy and torch take every core of a shared box.
        env["OMP_NUM_THREADS"] = "4"
        env.update(extra)
        return env

    # ---- preflight -----------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []
        if self.settings is None:
            return [
                "no [benchmarks.hugsim] section in paddock.toml; it needs `python` and `repo` "
                "(see docs/benchmarks/hugsim.md)"
            ]
        for key in ("python", "repo"):
            if not self.setting(key):
                problems.append(f"[benchmarks.hugsim] {key} is not set in paddock.toml")
        python = self.setting("python")
        if python and not Path(python).exists():
            problems.append(f"HUGSIM interpreter not found: {python}")

        repo = self.repo
        if repo is not None:
            for relative in ("closed_loop.py", "configs/sim/kinematic.yaml"):
                if not (repo / relative).exists():
                    problems.append(f"HUGSIM checkout is missing {repo / relative}")
            if not problems:
                problems += patch_problems(repo)

        root = self.data_root
        if root is None:
            problems.append(
                f"dataset {self.spec.dataset!r} is not configured; "
                f"add [datasets.{self.spec.dataset}] to paddock.toml"
            )
        else:
            for relative in ("scenarios", "scenes", "3DRealCar"):
                if not (root / relative).is_dir():
                    problems.append(f"HUGSIM data has no {relative}/ under {root}")
            hd_map = [
                p.stem for p in sorted((root / "scenarios").glob("*/*.yaml")) if _HD_MAP.search(p.read_text())
            ]
            if hd_map and not Path("~/.unified_data_cache").expanduser().exists():
                problems.append(
                    f"{len(hd_map)} scenarios need the nuScenes HD map, which HUGSIM reads from "
                    f"~/.unified_data_cache (hard-coded upstream); link it to the release's map cache: "
                    f"ln -s {root / 'nusc_map_cache'} ~/.unified_data_cache"
                )

        for key in ("per_episode_timeout_s",):
            value = self.setting(key)
            if value and not value.strip().isdigit():
                problems.append(f"[benchmarks.hugsim] {key} must be an integer, got {value!r}")
        value = self.setting("min_free_gb")
        if value:
            try:
                float(value)
            except ValueError:
                problems.append(f"[benchmarks.hugsim] min_free_gb must be a number, got {value!r}")

        if not problems:
            problems += self._import_problems()
        return problems

    def _import_problems(self) -> list[str]:
        """Import what the launcher and bridge will, in HUGSIM's interpreter, from the checkout."""
        command = [
            str(self.setting("python")),
            "-c",
            f"import importlib\nfor name in {list(CLIENT_IMPORTS)!r}:\n    importlib.import_module(name)\n",
        ]
        try:
            result = subprocess.run(
                command, cwd=self.repo, env=self._environment({}), capture_output=True, text=True, timeout=300
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"could not run the HUGSIM interpreter to check the client's imports: {exc}"]
        if result.returncode == 0:
            return []
        last = (result.stderr.strip().splitlines() or ["(no output)"])[-1]
        return [
            f"the HUGSIM interpreter cannot import what the launcher and bridge need: {last}\n"
            f"it must import {', '.join(CLIENT_IMPORTS)} from the checkout (see docs/benchmarks/hugsim.md)"
        ]

    def scene_problems(self, scene: str) -> list[str]:
        """``--scene`` names a scenario, a dataset or a scene; say so before the model loads."""
        scenarios = self._scenarios()
        if not scenarios or select(scenarios, [scene]):
            return []
        if scene in EXCLUDED:
            return [f"--scene {scene} is excluded for every model: {EXCLUDED[scene]}"]
        return [
            f"--scene {scene} matches no scenario, dataset ({', '.join(DATASETS)}) or scene name "
            f"in the {len(scenarios)} scenarios this site has"
        ]

    # ---- what is measured ----------------------------------------------

    def run_config(self) -> dict[str, Any]:
        """The protocol, for ``run.json``: the scenarios left out, the controller, the cap, and no route."""
        return {
            "excluded": sorted(EXCLUDED),
            "controller": "traj2control with upstream PR #57 (patches/HUGSIM/0001)",
            "step_cap": STEP_CAP,
            "route": "command only",
        }

    def model_run_config(self, model) -> dict[str, Any]:
        """How often this model is called, and which cameras are rendered for it.

        The call rate changes what is driven -- a plan held for 0.5 s is not a plan refreshed every 0.25 s --
        so two rates never share a run id.
        """
        n = infer_every(getattr(model, "planning_hz", None))
        return {
            "infer_every": n,
            "infer_hz": round(1.0 / (STEP_DT * n), 3),
            "render_cams": sorted(model.cameras),
        }

    # ---- episodes ------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        root = self.data_root
        every = self._scenarios()
        chosen = select(every, scenes)
        self._scope = (len(chosen), len(every))
        for ds, stem, _ in chosen:
            path = root / "scenarios" / ds / f"{stem}.yaml"
            scene, mode = scenario_meta(path.read_text())
            yield Episode(
                episode_id=f"{ds}-{stem}",
                scenario=stem,
                scene=scene,
                run_index=0,
                config={"dataset": ds, "path": str(path), "mode": mode, "difficulty": difficulty(mode)},
            )

    def invalid_reasons(self) -> list[str]:
        """A run of fewer scenarios than the site has is a subset, not a HUGSIM score."""
        ran, total = getattr(self, "_scope", (None, None))
        if ran is None or ran >= total:
            return []
        return [f"{ran} of the {total} scenarios (--scene); a subset, not a score"]

    def episode_samples(
        self, episode: Episode, metrics: dict[str, float], out_dir: Path
    ) -> list[dict[str, Any]]:
        """One sample per scenario, split by source dataset and grouped by difficulty.

        So ``metrics.json`` carries the per-dataset table in ``by_split`` and the per-difficulty one in
        ``by_group``, the two breakdowns HUGSIM's own paper reports. How the episode ended goes in ``extra``.
        """
        extra = {}
        record = out_dir / "episode.json"
        if record.exists():
            extra = json.loads(record.read_text()).get("extra", {})
        return [
            {
                "sample_id": f"hugsim/{episode.config['dataset']}/{episode.scenario}",
                "native_id": {
                    "scenario": episode.scenario,
                    "dataset": episode.config["dataset"],
                    "scene": episode.scene,
                    "mode": episode.config["mode"],
                },
                "split": episode.config["dataset"],
                "group": episode.config["difficulty"],
                "metrics": {k: float(v) for k, v in metrics.items()},
                "extra": extra,
            }
        ]

    def completed_metrics(self, episode: Episode, out_dir: Path, runs: int = 1) -> dict[str, float] | None:
        """The episode's metrics if its ``eval.json`` is on disk; HUGSIM writes it after the last step."""
        try:
            metrics, _ = episode_record(
                out_dir / "eval.json", out_dir / "hugsim.log", out_dir / "agent_log.jsonl"
            )
        except NotScored:
            return None
        return metrics

    # ---- run one episode -----------------------------------------------

    def run_episode(
        self,
        episode: Episode,
        policy_url: str,
        out_dir: Path,
        echo: Callable[[str], None] = print,
    ) -> dict[str, float]:
        """Drive one scenario through HUGSIM against a running policy node."""
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "hugsim.log"
        timeout_s = int(self.setting("per_episode_timeout_s") or DEFAULT_EPISODE_TIMEOUT_S)
        cameras = episode.config.get("cameras") or []
        env = self._environment({"HUGSIM_RENDER_CAMS": ",".join(cameras)} if cameras else {})
        command = [
            str(self.setting("python")),
            "-u",
            str(LAUNCHER),
            "--hugsim",
            str(self.repo),
            "--data_root",
            str(self.data_root),
            "--scenario",
            episode.config["path"],
            "--output",
            str(out_dir),
            "--server",
            policy_url,
            "--infer_every",
            str(infer_every(episode.config.get("planning_hz"))),
        ]

        gpu.wait_for(
            float(self.setting("min_free_gb") or DEFAULT_MIN_FREE_GB),
            env=env,
            on_wait=lambda need, free: echo(
                f"  waiting for {need:.0f} GB free before HUGSIM starts ({free:.1f} GB)"
            ),
        )
        started = time.time()
        timed_out = False
        with log_path.open("w") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            # Its own session, so a timeout or an interrupted sweep takes the simulator and the bridge down
            # together.
            launcher = subprocess.Popen(
                command,
                cwd=self.repo,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = launcher.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(launcher)
                returncode = launcher.returncode
            except BaseException:
                kill_group(launcher)
                raise
        elapsed = time.time() - started

        if timed_out:
            raise RuntimeError(
                f"{episode.episode_id}: timed out after {timeout_s / 60:.0f} min (per_episode_timeout_s); "
                f"HUGSIM killed\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        if returncode != 0:
            raise RuntimeError(
                f"{episode.episode_id}: HUGSIM exited {returncode} after {elapsed:.0f} s"
                f"\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            )
        try:
            metrics, extra = episode_record(out_dir / "eval.json", log_path, out_dir / "agent_log.jsonl")
        except NotScored as exc:
            raise RuntimeError(
                f"{episode.episode_id}: {exc}\n--- tail of {log_path.name} ---\n{_tail(log_path)}"
            ) from exc
        extra["wall_s"] = round(elapsed, 1)
        (out_dir / "episode.json").write_text(json.dumps({"metrics": metrics, "extra": extra}, indent=1))
        # HUGSIM re-derives these point clouds from the scene on every run (about 38 MB each); the scores are
        # in eval.json, so keeping them costs ~30 GB a sweep for nothing.
        if not self.flag("keep_ply"):
            for name in ("scene.ply", "ground.ply"):
                (out_dir / name).unlink(missing_ok=True)
        echo(f"  {episode.episode_id}: {extra['ending']} after {extra.get('steps', '?')} steps")
        return metrics


#: Paddock looks for this name when loading an adapter.
BENCHMARK = HugsimBenchmark
