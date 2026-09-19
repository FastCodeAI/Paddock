"""AlpaSim closed-loop benchmark (NVlabs/alpasim, pinned at affc2ea).

AlpaSim replays a recorded clip as a NuRec neural reconstruction, hands the
policy camera frames at a fixed control rate, and steps the world on what the
policy commands. Errors compound, which is the point.

Closed loop like neuroNCAP and Bench2Drive, and the shape mirrors them: a
`preflight` that names every missing piece before a model loads, `episodes` that
enumerate the pinned scene list, and a `run_episode` that shells out to the
simulator and parses what it wrote. What differs is one thing, and it is the
reason `bridge.py` exists:

**AlpaSim's driver contract is gRPC, not HTTP.** Every other simulator paddock
drives can be pointed at the policy node directly. AlpaSim calls an
EgodriverService, so `launcher.py` runs a bridge that terminates that contract
and translates to `/infer`, and the wizard is told to use it through
`wizard.external_services.driver` rather than starting a driver of its own.

Three properties of AlpaSim that shape the code below:

* **The scored window is not the whole rollout.** Aggregation trims warm-up,
  truncates after the first collision or offroad event -- including a
  *not-at-fault* rear-end -- and truncates again at a divergence cap. A policy
  that fails early is scored over less of the scene than one that drives it
  out, so its fidelity numbers look better. Only `scene_score` and the outcome
  rates are safe to compare across runs; `docs/benchmarks/alpasim.md` says so
  at more length.
* **`scene_score` is per-rollout and is not in `metrics_results`.** It lives at
  `rollouts[].score`. The aggregate block looks complete and does not contain
  the primary metric at all.
* **A rollout writes `_complete` when it finishes.** That is what makes
  `completed_metrics` -- and therefore `--resume` -- honest here rather than a
  guess about whether a directory holds a finished episode.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from paddock.adapters.benchmarks.alpasim import launcher
from paddock.benchmark import ClosedLoopBenchmark, Episode

#: Pinned scene lists that ship with this adapter.
SCENES_DIR = Path(__file__).parent / "scenes"
DEFAULT_SCENE_SET = "public_2601_first10"

#: Per-scene wall-clock budget. Measured on the site: about 5 min per scene with
#: a 10B driver at 4 cameras / 10 Hz, and about 50 s for the constant-velocity
#: floor, whose cost is rendering rather than inference. Generous, because the
#: timeout is a backstop against a wedged container, not a performance target.
DEFAULT_SCENE_TIMEOUT_S = 1800

#: Preferred bridge port, chosen to collide with nothing the run also starts.
#: The wizard enumerates its containers' ports from `defines.baseport: 6000`
#: (`src/wizard/configs/base_config.yaml:76`), and this site's own driver
#: configs have historically sat in the 9100s. 9200 is clear of both. A bridge
#: that landed on a service port would be found by the runtime and mistaken for
#: that service. (An earlier comment here said AlpaSim's services occupy
#: 9100-9107; they do not, at this pin -- the choice is still right, the stated
#: reason was not.)
BRIDGE_PORT_BASE = 9200

#: The reconstruction the scene is replayed through, when the site does not say.
#: NuRec is what the pinned sha wires: `runtime.scene_provider.kind: usdz` with
#: `runtime.renderer.kind: sensorsim` and no `video_model_config`.
DEFAULT_RENDERER = "nurec"

#: What this adapter can actually start. OmniDreams is a real AlpaSim backend
#: and a column in the README, but selecting it needs a `video_model_config`
#: this adapter does not set, so it is refused in preflight rather than
#: recorded as if it had run.
SUPPORTED_RENDERERS = ("nurec",)


def scorer_sha() -> str:
    """The AlpaSim pin from repos.yaml, which is also the scorer's version.

    Read rather than copied. The scorer ships inside the pinned repo, so the
    next person to bump the pin -- in its own PR, which is the workflow
    repos.yaml exists to support -- moves every number this benchmark produces.
    A literal here would keep saying `affc2ea` while something else did the
    scoring, and the provenance field would hide the change it exists to explain.
    """
    from paddock.registry import Registry

    return Registry.load().repos["alpasim"].sha


#: Metrics we lift out of a rollout record, mapped to canonical names.
_METRIC_KEYS = {
    "progress": "progress",
    "collision_any": "collision_rate",
    "collision_at_fault": "collision_at_fault",
    "offroad": "offroad",
    "duration_frac_20s": "duration_frac_20s",
    "dist_to_gt_trajectory": "dist_to_gt_trajectory",
    "open_loop_collision": "open_loop_collision",
}


#: The only reasons the scorer zeroes a scene it actually scored. Anything else
#: in `failure_reason` is the runtime reporting that the rollout did not run.
_HARD_FAILURE_REASONS = frozenset({"collision_at_fault", "offroad"})


class NotScored(ValueError):
    """The simulator produced a summary that holds no usable score."""


def _crashed(rollout: dict[str, Any]) -> bool:
    """Did this rollout fail to run, as opposed to running and scoring zero?

    `eval.allow_aggregation_with_failed_rollouts=true` -- which launcher.py sets,
    so that one dead scene does not discard nine good ones -- makes AlpaSim emit
    a summary row for a rollout that crashed. Upstream writes that row with
    **`score: 0.0`**, not `None` (`eval/aggregation/failed_rollouts.py`), so the
    score alone cannot tell a crash from a model that drove into a wall. Both
    then average into the leaderboard as the same number.

    The shapes do differ. A scored rollout carries a full numeric block and a
    `failure_reason` drawn from `_HARD_FAILURE_REASONS`; a crashed one carries
    `{run_name, run_uuid, clipgt_id, rollout_id, error}` and no metric at all.
    Both tests are applied, because either alone would be a guess about which
    field upstream keeps stable.
    """
    values = rollout.get("metrics") or {}
    scored_anything = any(values.get(key) is not None for key in _METRIC_KEYS)
    if scored_anything:
        return False
    reason = rollout.get("failure_reason")
    return "error" in values or rollout.get("status") == "fail" or reason not in _HARD_FAILURE_REASONS


def load_scene_set(name: str = DEFAULT_SCENE_SET) -> list[str]:
    """The pinned scene ids for one suite selection."""
    path = SCENES_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCENES_DIR.glob("*.yaml"))) or "none"
        raise FileNotFoundError(f"no scene set {name!r} in {SCENES_DIR} (have: {available})")
    return list(yaml.safe_load(path.read_text())["scenes"])


def parse_summary(summary_json: Path, scene_id: str | None = None) -> dict[str, float]:
    """`results-summary.json` -> metrics for one scene.

    Reads `rollouts[]`, not `metrics_results`: the aggregate block does not
    carry `scene_score` at all, so a reader that trusts it silently drops the
    primary metric. See the module docstring.
    """
    body = json.loads(summary_json.read_text())
    rollouts = body.get("rollouts") or []
    if scene_id is not None:
        rollouts = [r for r in rollouts if r.get("clipgt_id") == scene_id]
    if not rollouts:
        raise NotScored(f"{summary_json} holds no rollout for {scene_id or 'any scene'}")
    if len(rollouts) > 1:
        raise NotScored(
            f"{summary_json} holds {len(rollouts)} rollouts for {scene_id}; "
            "one episode is one scene, so this is an unexpected shape"
        )

    rollout = rollouts[0]
    if rollout.get("score") is None:
        raise NotScored(
            f"{summary_json}: rollout for {scene_id} has no score (status {rollout.get('status')!r})"
        )
    if _crashed(rollout):
        raise NotScored(
            f"{summary_json}: rollout for {scene_id} did not run -- "
            f"status {rollout.get('status')!r}, failure_reason "
            f"{rollout.get('failure_reason')!r}, error "
            f"{(rollout.get('metrics') or {}).get('error')!r}. "
            f"Its score of {rollout.get('score')!r} is the aggregator filling in a "
            f"row for a crash, not a measurement of the policy"
        )

    values = rollout.get("metrics") or {}
    metrics = {"scene_score": float(rollout["score"])}
    for source, canonical in _METRIC_KEYS.items():
        if values.get(source) is not None:
            metrics[canonical] = float(values[source])
    return metrics


class AlpaSimBenchmark(ClosedLoopBenchmark):
    # -- site settings -----------------------------------------------------

    @property
    def repo(self) -> Path | None:
        value = self.setting("repo")
        return Path(value) if value else None

    @property
    def scene_set(self) -> str:
        return self.setting("scene_set") or DEFAULT_SCENE_SET

    @property
    def scene_timeout_s(self) -> int:
        return int(self.setting("per_scene_timeout_s") or DEFAULT_SCENE_TIMEOUT_S)

    @property
    def startup_timeout_s(self) -> int:
        return int(self.setting("startup_timeout_s") or launcher.DEFAULT_STARTUP_TIMEOUT_S)

    @property
    def renderer(self) -> str:
        """Which reconstruction drives the scene: `nurec` or `omnidreams`.

        The README carries them as separate columns because they are different
        reconstructions of the same clip -- a score from one is not a statement
        about the other. It is in `run_config` so the two hash to different run
        ids and can never be averaged together by accident.
        """
        return str(self.setting("renderer") or DEFAULT_RENDERER)

    def run_config(self) -> dict[str, Any]:
        """What changes the number, hashed into the run id and stored."""
        return {
            "scene_set": self.scene_set,
            "startup_timeout_s": self.startup_timeout_s,
            # Read from repos.yaml, never copied: the scorer ships in that repo,
            # so bumping the pin moves every published number. A literal here
            # would keep reporting the old sha while something else did the
            # scoring -- the field that exists to explain a change in the
            # numbers would be the field that concealed it.
            "scorer_repo_sha": scorer_sha(),
            "renderer": self.renderer,
            "topology": self.setting("topology") or launcher.DEFAULT_TOPOLOGY,
        }

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> list[str]:
        """Everything that must hold before a model loads, checked cheaply."""
        problems: list[str] = []
        if self.renderer not in SUPPORTED_RENDERERS:
            problems.append(
                f"[benchmarks.alpasim] renderer = {self.renderer!r}: this adapter wires "
                f"{', '.join(SUPPORTED_RENDERERS)} only. OmniDreams needs a "
                f"runtime.renderer.video_model_config the launcher does not set, and a run "
                f"labelled omnidreams that rendered NuRec would be worse than no run"
            )
        if self.settings is None:
            return [
                "no [benchmarks.alpasim] section in paddock.toml; it needs `repo` and "
                "`python` (see docs/benchmarks/alpasim.md)"
            ]
        for key in ("repo", "python"):
            if not self.setting(key):
                problems.append(f"[benchmarks.alpasim] {key} is not set in paddock.toml")

        python = self.setting("python")
        if python and not Path(python).exists():
            problems.append(f"AlpaSim interpreter not found: {python}")

        repo = self.repo
        if repo is not None:
            for relative, why in (
                (Path("src/wizard/alpasim_wizard/__main__.py"), "the wizard entry point"),
                (Path("src/eval/src/eval/aggregation/processing.py"), "the scorer"),
                (Path("src/grpc/alpasim_grpc/v0/egodriver.proto"), "the driver contract"),
            ):
                if not (repo / relative).exists():
                    problems.append(f"AlpaSim checkout is missing {why}: {repo / relative}")

        try:
            scenes = load_scene_set(self.scene_set)
        except FileNotFoundError as exc:
            problems.append(str(exc))
        else:
            if not scenes:
                problems.append(f"scene set {self.scene_set!r} is empty")

        problems += self._docker_problems()
        return problems

    def _docker_problems(self) -> list[str]:
        """Docker must be usable, and the images must already be built.

        Checked here because building them is a long job that would otherwise
        start inside the first episode and look like a hung simulator.
        """
        import shutil
        import subprocess

        if shutil.which("docker") is None:
            return ["docker is not on PATH; AlpaSim runs its services through docker compose"]
        try:
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"could not talk to the docker daemon: {exc}"]
        if probe.returncode != 0:
            return [f"docker daemon is not answering: {(probe.stderr or '').strip()[:200]}"]
        return []

    def scene_problems(self, scenes: list[str] | None) -> list[str]:
        """Scene ids that are not in the pinned set, named before a model loads."""
        if not scenes:
            return []
        known = set(load_scene_set(self.scene_set))
        unknown = [s for s in scenes if s not in known]
        if not unknown:
            return []
        return [f"scene id(s) not in scene set {self.scene_set!r}: {', '.join(sorted(unknown))}"]

    # -- episodes ----------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        """One episode per scene, in the pinned order.

        Deterministic by construction: the order comes from the pinned YAML, not
        from the filesystem, so `--limit` selects the same scenes at both sites.

        `scenarios` is the scene *set* here -- AlpaSim has no scenario axis of
        its own, and `Episode.scenario` carries the set name so a run's episodes
        say which pinned list they came from. The parameter is accepted because
        the runner always passes it, even though `ClosedLoopBenchmark.episodes`
        does not declare it; a harness that omits it dies with a TypeError
        before the first scene.
        """
        selected = set(scenes) if scenes else None
        wanted = set(scenarios) if scenarios else None
        if wanted is not None and self.scene_set not in wanted:
            return
        for scene in load_scene_set(self.scene_set):
            if selected is not None and scene not in selected:
                continue
            yield Episode(
                episode_id=scene,
                scenario=self.scene_set,
                scene=scene,
                run_index=0,
                config={"scene_set": self.scene_set, "runs": runs},
            )

    def scenario_problems(self, scenarios: list[str] | None) -> list[str]:
        """A named scenario that is not this run's scene set, reported early.

        Without this the filter silently yields nothing and the run records zero
        episodes, which reads as a benchmark with no work rather than a typo.
        """
        if not scenarios:
            return []
        available = ", ".join(sorted(p.stem for p in SCENES_DIR.glob("*.yaml")))
        unknown = [s for s in scenarios if s != self.scene_set]
        if not unknown:
            return []
        return [
            f"scenario(s) {', '.join(sorted(unknown))} are not this run's scene set "
            f"({self.scene_set!r}); available scene sets: {available}"
        ]

    # -- resume ------------------------------------------------------------

    def completed_metrics(self, episode: Episode, out_dir: Path, runs: int = 1) -> dict[str, float] | None:
        """Metrics already on disk for this scene, or None to drive it again.

        AlpaSim writes a `_complete` marker inside a rollout directory when that
        rollout finished, so a finished episode is distinguishable from an
        interrupted one and `--resume` can be honest. Without both the marker
        and a parseable summary this returns None and the scene is re-driven --
        half an episode counted as whole is a wrong number, not a slow one.
        """
        summary = out_dir / "aggregate" / "results-summary.json"
        if not summary.exists():
            return None
        markers = list((out_dir / "rollouts").glob(f"{episode.scene}/*/_complete"))
        if not markers:
            return None
        try:
            return parse_summary(summary, episode.scene)
        except (NotScored, json.JSONDecodeError, OSError):
            return None

    # -- run one episode ---------------------------------------------------

    def run_episode(
        self,
        episode: Episode,
        policy_url: str,
        out_dir: Path,
        echo: Callable[[str], None] = print,
    ) -> dict[str, float]:
        """Drive one scene against a running policy node.

        The bridge goes up first and comes down last: the wizard cannot start
        without a driver answering, and a bridge left listening would collide
        with the next episode's port.

        Failure order matters -- a timeout is reported as a timeout, a non-zero
        exit as an exit, and only then is the summary read, because AlpaSim
        writes a partial summary either way.
        """
        from paddock.runtime import ports

        repo = self.repo
        python = self.setting("python")
        if repo is None or python is None:
            raise RuntimeError(
                "[benchmarks.alpasim] needs `repo` and `python`; preflight should have caught this"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        bridge_port = ports.find_free(BRIDGE_PORT_BASE)
        bridge_log = out_dir / "bridge.log"
        wizard_log = out_dir / "wizard.log"

        echo(f"{episode.scene}: bridge on :{bridge_port} -> {policy_url}")
        with launcher.BridgeProcess(
            python=python,
            policy_url=policy_url,
            port=bridge_port,
            version_id=f"paddock-bridge/{episode.scenario}",
            log_path=bridge_log,
        ):
            command = launcher.wizard_command(
                repo=repo,
                python=python,
                scene_id=episode.scene,
                out_dir=out_dir,
                bridge_port=bridge_port,
                searchpath=None,
                startup_timeout_s=self.startup_timeout_s,
                deploy=self.setting("deploy") or launcher.DEFAULT_DEPLOY,
                topology=self.setting("topology") or launcher.DEFAULT_TOPOLOGY,
                scene_cache=self.setting("scene_cache"),
                video=self.flag("video"),
                video_every_nth_frame=int(self.setting("video_every_nth_frame") or 1),
            )
            returncode, timed_out, elapsed = launcher.run_wizard(
                command,
                cwd=repo,
                log_path=wizard_log,
                timeout_s=self.scene_timeout_s,
                echo=echo,
            )

        if timed_out:
            raise RuntimeError(
                f"{episode.scene}: timed out after {self.scene_timeout_s / 60:.0f} min "
                f"(per_scene_timeout_s); process group killed\n"
                f"--- tail of {wizard_log.name} ---\n{launcher._tail(wizard_log)}"
            )
        if returncode != 0:
            hint = "" if launcher.node_alive(policy_url) else " (the policy node had stopped answering)"
            raise RuntimeError(
                f"{episode.scene}: wizard exited {returncode} after {elapsed / 60:.1f} min{hint}\n"
                f"--- tail of {wizard_log.name} ---\n{launcher._tail(wizard_log)}\n"
                f"--- tail of {bridge_log.name} ---\n{launcher._tail(bridge_log)}"
            )

        summary = out_dir / "aggregate" / "results-summary.json"
        if not summary.exists():
            raise RuntimeError(
                f"{episode.scene}: wizard exited 0 but wrote no results-summary.json\n"
                f"--- tail of {wizard_log.name} ---\n{launcher._tail(wizard_log)}"
            )
        try:
            metrics = parse_summary(summary, episode.scene)
        except (NotScored, json.JSONDecodeError, OSError) as exc:
            # Not NotScored alone: a summary truncated by an OOM kill or a reboot
            # raises JSONDecodeError, which is not a RuntimeError, so it escapes
            # the runner's per-episode handler and the sweep dies with a
            # traceback -- discarding every episode already scored. The right
            # tuple is the one `completed_metrics` above already uses.
            raise RuntimeError(
                f"{episode.scene}: {exc}\n--- tail of {wizard_log.name} ---\n{launcher._tail(wizard_log)}"
            ) from exc

        metrics["duration_s"] = round(elapsed, 1)
        echo(f"{episode.scene}: scene_score {metrics['scene_score']:.4f} in {elapsed / 60:.1f} min")
        return metrics


#: Paddock looks for this name when loading an adapter.
BENCHMARK = AlpaSimBenchmark
