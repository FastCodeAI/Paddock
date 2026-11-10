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
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from paddock.adapters.benchmarks.alpasim import launcher
from paddock.benchmark import ClosedLoopBenchmark, Episode
from paddock.registry import MODELS_DIR

#: Pinned scene lists that ship with this adapter.
SCENES_DIR = Path(__file__).parent / "scenes"
DEFAULT_SCENE_SET = "public_2601_first10"

#: The two halves of the 30-scene set, in lookup order: "pilot10" is the pinned
#: ten (`public_2601_first10`), "ext20" the twenty drawn after them. Every
#: sample records which half its scene is in, whichever set the run drove, so a
#: 30-scene run can be read back as its ten and its twenty -- the names the
#: stored `..._pilot10` and `..._30scenes` rows already use.
SCENE_PARTS = (("pilot10", "public_2601_first10"), ("ext20", "public_2601_30scenes"))

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

#: Within a run's own port range (`PADDOCK_BASE_PORT`, set per lane by the gate),
#: where the bridge and the wizard's services start. The policy node takes the
#: first ports of the range; the wizard needs about a dozen. Two AlpaSim runs with
#: different bases then never pick the same host ports, which with fixed bases
#: (9200 here, the wizard's 6000) they could whenever both started at once.
BRIDGE_PORT_OFFSET = 50
WIZARD_PORT_OFFSET = 60


def port_bases(environ: dict[str, str] | None = None) -> tuple[int, int | None]:
    """(bridge port to start from, wizard baseport or None for its own default).

    Only a run that asked for a range of its own moves: without
    `PADDOCK_BASE_PORT` the ports are what they always were.
    """
    raw = (os.environ if environ is None else environ).get("PADDOCK_BASE_PORT", "").strip()
    if not raw.isdigit():
        return BRIDGE_PORT_BASE, None
    base = int(raw)
    return base + BRIDGE_PORT_OFFSET, base + WIZARD_PORT_OFFSET


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


@dataclass(frozen=True)
class DriverProfile:
    """The simulation settings and bridge translation one model is driven with.

    Every model gets the default unless its own folder has an `alpasim.yaml`
    (PROFILE_FILE below): the `paddock_bridge` driver config, no translation
    flags, scenes named by id.
    Video is the site's choice for every model, profile or not: AlpaSim renders
    it after its scorer has written the metrics, so it changes no score. (What
    the bridge does for every model, profile or not -- read the plan, reset the
    node per scene, record where headings came from -- is in bridge.py.)
    """

    #: Driver config, passed as `driver=`: `configs/driver/<driver>.yaml` in this
    #: benchmark, or `driver/<driver>.yaml` in the model's own folder.
    driver: str = "paddock_bridge"
    #: Extra `bridge.py` flags: camera renames, placeholders, intrinsics.
    bridge_args: tuple[str, ...] = ()
    #: A file under scenes/ pinning each scene to one artifact uuid, or None to
    #: name scenes by id alone.
    scenes_csv: str | None = None
    #: Write the bridge's per-replan log next to the run's samples.jsonl, as
    #: `<scene>.replans.jsonl`. Only a bridge running a replan rhythm writes one.
    replan_log: bool = False
    #: The cameras the model reads on AlpaSim, when that is fewer than the
    #: `cameras` its model.yaml declares for every benchmark; None means all of
    #: them. Used only to check, before a run, that each one reaches the node
    #: (`input_problems`); it changes nothing the bridge sends. Constant velocity
    #: says `[]`: it declares CAM_FRONT for the benchmarks that need a camera to
    #: render, and reads none.
    cameras: tuple[str, ...] | None = None

    def drives_like_default(self) -> bool:
        """True when this profile drives the model exactly as DEFAULT_PROFILE does."""
        return (self.driver, self.bridge_args, self.scenes_csv, self.replan_log) == (
            DEFAULT_PROFILE.driver,
            DEFAULT_PROFILE.bridge_args,
            DEFAULT_PROFILE.scenes_csv,
            DEFAULT_PROFILE.replan_log,
        )


DEFAULT_PROFILE = DriverProfile()

#: The file, in a model's own folder (`adapters/models/<model>/`), that gives
#: that model settings other than the default. A model without one gets
#: DEFAULT_PROFILE. Each file says why its model needs what it sets.
PROFILE_FILE = "alpasim.yaml"

#: Every key a profile file must set. All four are required so that a typo or a
#: forgotten key fails at load time instead of quietly driving the model on a
#: default it was never meant to have.
_PROFILE_KEYS = ("driver", "bridge_args", "scenes_csv", "replan_log")

#: Keys a profile file may set. Any other key fails at load time.
_OPTIONAL_PROFILE_KEYS = ("cameras",)


class DriverProfileError(ValueError):
    """A model's alpasim.yaml exists but does not describe a valid profile."""


def profile_path(model: str) -> Path:
    """Where `model`'s AlpaSim profile would be, whether or not it exists."""
    return MODELS_DIR / model / PROFILE_FILE


def load_profile(path: Path) -> DriverProfile:
    """Read and check one profile file. Any problem names the file."""
    try:
        with path.open() as fh:
            body = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise DriverProfileError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(body, dict):
        raise DriverProfileError(f"{path}: must be a mapping of {', '.join(_PROFILE_KEYS)}")
    unknown = sorted(set(body) - set(_PROFILE_KEYS) - set(_OPTIONAL_PROFILE_KEYS))
    missing = [key for key in _PROFILE_KEYS if key not in body]
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown key(s) {', '.join(map(str, unknown))}")
        if missing:
            parts.append(f"missing key(s) {', '.join(missing)}")
        raise DriverProfileError(
            f"{path}: {'; '.join(parts)} (required: {', '.join(_PROFILE_KEYS)}; "
            f"optional: {', '.join(_OPTIONAL_PROFILE_KEYS)})"
        )
    driver, bridge_args = body["driver"], body["bridge_args"]
    scenes_csv, replan_log = body["scenes_csv"], body["replan_log"]
    if not isinstance(driver, str) or not driver:
        raise DriverProfileError(f"{path}: driver must be a non-empty string, got {driver!r}")
    if not isinstance(bridge_args, list) or not all(isinstance(arg, str) for arg in bridge_args):
        raise DriverProfileError(f"{path}: bridge_args must be a list of strings, got {bridge_args!r}")
    if scenes_csv is not None and (not isinstance(scenes_csv, str) or not scenes_csv):
        raise DriverProfileError(
            f"{path}: scenes_csv must be a file name under scenes/ or null, got {scenes_csv!r}"
        )
    if not isinstance(replan_log, bool):
        raise DriverProfileError(f"{path}: replan_log must be true or false, got {replan_log!r}")
    cameras = body.get("cameras")
    if "cameras" in body and (
        not isinstance(cameras, list) or not all(isinstance(c, str) and c for c in cameras)
    ):
        raise DriverProfileError(f"{path}: cameras must be a list of camera names, got {cameras!r}")
    return DriverProfile(
        driver=driver,
        bridge_args=tuple(bridge_args),
        scenes_csv=scenes_csv,
        replan_log=replan_log,
        cameras=None if cameras is None else tuple(cameras),
    )


def has_profile(model: str) -> bool:
    return profile_path(model).is_file()


def profile_for(model: str) -> DriverProfile:
    """`model`'s own profile from its folder, or DEFAULT_PROFILE when it has none.

    Only a missing file means the default. A file that is there but wrong raises
    DriverProfileError naming it.
    """
    path = profile_path(model)
    if not path.is_file():
        return DEFAULT_PROFILE
    return load_profile(path)


def profiled_models() -> tuple[str, ...]:
    """Every model folder that has a profile file, sorted."""
    return tuple(sorted(path.parent.name for path in MODELS_DIR.glob(f"*/{PROFILE_FILE}")))


def driver_config_paths(model: str, driver: str) -> tuple[Path, Path]:
    """Where `driver` may live: this benchmark's configs, then the model's folder."""
    return (
        launcher.CONFIGS_DIR / "driver" / f"{driver}.yaml",
        MODELS_DIR / model / "driver" / f"{driver}.yaml",
    )


def driver_searchpath(model: str, profile: DriverProfile) -> Path | None:
    """The model's folder, when its driver config lives there; otherwise None.

    None leaves the wizard command exactly as it was before a model could keep
    its own driver config, so every model whose config is in this benchmark is
    launched as before. AlpaSim's config loader searches this benchmark's
    configs first, then this folder (launcher.wizard_command).
    """
    _, own = driver_config_paths(model, profile.driver)
    return MODELS_DIR / model if own.is_file() else None


def _is_global_package(path: Path) -> bool:
    """Does this config's leading comment block carry Hydra's `# @package _global_`?

    Hydra reads the package directive only from the comments at the top of the
    file, before the first line of YAML.
    """
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return False
        if stripped.lstrip("#").split() == ["@package", "_global_"]:
            return True
    return False


def cameras_reaching_node(bridge_args: tuple[str, ...]) -> set[str]:
    """The paddock camera names the bridge sends under these flags.

    The bridge sends each AlpaSim camera under AlpaSim's own id (such as
    `camera_front_wide_120fov`) unless `--camera-name ID=NAME` renames it, and
    `--placeholder NAME` adds a frame under NAME. A node keeps only the cameras
    its model declares, by name, so a declared camera reaches it only through
    one of those two flags.
    """
    names: set[str] = set()
    args = list(bridge_args)
    for i, arg in enumerate(args):
        for flag in ("--camera-name", "--placeholder"):
            if arg == flag and i + 1 < len(args):
                value = args[i + 1]
            elif arg.startswith(flag + "="):
                value = arg[len(flag) + 1 :]
            else:
                continue
            names.add(value.split("=", 1)[1] if flag == "--camera-name" and "=" in value else value)
    return names


def profile_problems(model) -> list[str]:
    """Why `model` cannot be driven on AlpaSim as its folder describes, or [].

    Checked before a dry run or any model starts: a profile that does not load,
    a driver config or scene list that is not there, or a camera the model reads
    that would never reach its node. The last is the silent failure: the model
    would plan from poses alone and the run would still be scored.
    """
    name = model.name
    try:
        profile = profile_for(name)
    except DriverProfileError as exc:
        return [str(exc)]
    problems: list[str] = []
    in_benchmark, in_model = driver_config_paths(name, profile.driver)
    if in_benchmark.is_file() and in_model.is_file():
        problems.append(
            f"{name}'s driver config {profile.driver!r} is both in {in_benchmark.parent} and in "
            f"{in_model.parent}; AlpaSim would load the first. Rename the one in the model's folder"
        )
    elif not in_benchmark.is_file() and not in_model.is_file():
        problems.append(f"{name}'s driver config {profile.driver!r} is neither {in_model} nor {in_benchmark}")
    elif in_model.is_file() and not _is_global_package(in_model):
        # Without the header Hydra nests the file under `driver.`, where no
        # AlpaSim setting is read, so the run keeps the shared values silently.
        problems.append(
            f"{in_model} does not start with `# @package _global_`; without it AlpaSim would load it "
            f"but ignore its settings (docs/adding-a-model.md, AlpaSim settings)"
        )
    if profile.scenes_csv is not None and not (SCENES_DIR / profile.scenes_csv).is_file():
        problems.append(
            f"{name}'s profile pins scenes in {SCENES_DIR / profile.scenes_csv}, which does not exist"
        )
    declared = tuple(model.cameras)
    reads = declared if profile.cameras is None else profile.cameras
    if extra := sorted(set(reads) - set(declared)):
        problems.append(
            f"{profile_path(name)}: cameras lists {', '.join(extra)}, which {name}'s model.yaml does not "
            f"declare ({', '.join(declared) or 'none'})"
        )
    if missing := [c for c in reads if c not in cameras_reaching_node(profile.bridge_args)]:
        why = (
            f"{profile_path(name)} neither renames an AlpaSim camera to it (--camera-name) nor fills it "
            f"(--placeholder)"
            if has_profile(name)
            else f"there is no {profile_path(name)} to rename AlpaSim's cameras to these names"
        )
        problems.append(
            f"{name} reads {', '.join(missing)}, which would never reach its node on AlpaSim: {why}. "
            f"Rename them in the model's alpasim.yaml, or set `cameras: []` there if it reads no image "
            f"on AlpaSim (docs/adding-a-model.md, AlpaSim settings)"
        )
    return problems


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


#: The bridge's per-scene count of where each plan's headings came from, in the
#: episode's directory (bridge.py `--plan-record`).
PLAN_RECORD = "bridge_plans.json"


def _headings_source(record: dict[str, Any]) -> str | None:
    counts = record.get("headings") or {}
    used = [source for source in ("model", "derived") if counts.get(source)]
    return {0: None, 1: used[0] if used else None, 2: "mixed"}[len(used)]


def scene_part(scene: str) -> str | None:
    """Which half of the 30-scene set *scene* is in (see `SCENE_PARTS`), or None."""
    for part, scene_set in SCENE_PARTS:
        if scene in load_scene_set(scene_set):
            return part
    return None


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

    def model_run_config(self, model) -> dict[str, Any]:
        """A model's profile, when it drives the model differently, so the run id and run.json say so.

        Empty for a model with no profile, so it records exactly what it recorded
        before profiles existed. Empty too for a profile that only says which
        cameras the model reads (constant velocity's): that is a check made
        before the run, and it changes nothing the bridge sends.
        """
        profile = profile_for(model.name)
        if profile.drives_like_default():
            return {}
        return {
            "driver_profile": model.name,
            "driver_config": profile.driver,
            "bridge_args": list(profile.bridge_args),
            "scenes_csv": profile.scenes_csv,
            "replan_log": profile.replan_log,
        }

    # -- preflight ---------------------------------------------------------

    def input_problems(self, model) -> list[str]:
        """What stops `model` being driven on AlpaSim; the runner asks before any dry run or launch."""
        return profile_problems(model)

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

    def invalid_reasons(self) -> list[str]:
        """Fewer clips than the pinned scene set is a subset, not an AlpaSim score.

        `--scene` selects clips of the set; stored `valid: true`, a few clips would
        be ranked beside full runs of the set.
        """
        ran, total = getattr(self, "_scenes_scope", (None, None))
        if ran is None or ran >= total:
            return []
        return [f"{ran} of the {total} clips of {self.scene_set} (--scene); a subset, not an AlpaSim score"]

    def scene_problems(self, scene: str) -> list[str]:
        """``--scene`` names a clip id outside the pinned set; say so before a model loads.

        One id per call, like every other harness: `run_benchmark` splits
        ``--scene a,b`` and calls this once per id. Taking a list here made it
        iterate the id's characters and reject every scene.
        """
        if scene in set(load_scene_set(self.scene_set)):
            return []
        return [f"scene id not in scene set {self.scene_set!r}: {scene}"]

    # -- episodes ----------------------------------------------------------

    def episodes(
        self,
        scenes: list[str] | None = None,
        runs: int = 1,
        scenarios: list[str] | None = None,
    ) -> Iterator[Episode]:
        """One episode per scene, in the pinned order.

        Deterministic by construction: the order comes from the pinned YAML, not
        from the filesystem, so `--limit` selects the same scenes at every site.

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
        pinned = load_scene_set(self.scene_set)
        # Recorded for `invalid_reasons`: the pinned set, against what runs.
        self._scenes_scope = (sum(1 for s in pinned if selected is None or s in selected), len(pinned))
        for scene in pinned:
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

    def episode_samples(
        self, episode: Episode, metrics: dict[str, float], out_dir: Path
    ) -> list[dict[str, Any]]:
        """One record per scene, identified by the clip alone.

        The default id appends the scene set (`alpasim/<clip>/<set>`), so one
        clip driven in the 10-scene set and again in the 30-scene set was two
        different samples. A paired comparison joins two runs on `sample_id`,
        and a 30-scene run then shared no sample with a 10-scene baseline. The
        clip is the unit AlpaSim scores, and `alpasim/<clip>` is the id the
        ingested rows already use (`results/ingest/alpasim.py`), so a run over
        the 30 also fingerprints (`sequences_sha256`) like the stored 30-scene
        rows.

        Where the scene came from is kept as data, not in the id: `split` is
        its half of the 30 (`pilot10` or `ext20`; the set's own name for a
        scene in neither), and `native_id.scene_set` the set the run drove.

        `headings` says how the car was steered: ``model`` when every plan
        carried the model's own yaw, ``derived`` when the bridge derived them
        from the path, ``mixed`` for both, None when the bridge left no record.
        `bridge_plans` holds the bridge's counts behind it. `history_frames`
        says how frames the bridge needed no plan from were sent: ``observe``,
        ``infer`` (a node without `/observe`, plan dropped), or None when it sent
        none (a model planned at every tick).
        """
        record = self._plan_record(out_dir)
        return [
            {
                "sample_id": f"{self.spec.name}/{episode.scene}",
                "native_id": {"scene": episode.scene, "scene_set": episode.scenario},
                "split": scene_part(episode.scene) or episode.scenario,
                "metrics": {k: float(v) for k, v in metrics.items() if k not in ("n_runs",)},
                "headings": record and _headings_source(record),
                "bridge_plans": record,
                "history_frames": (record or {}).get("history_frames"),
            }
        ]

    @staticmethod
    def _plan_record(out_dir: Path) -> dict[str, Any] | None:
        try:
            return json.loads((out_dir / PLAN_RECORD).read_text())
        except (OSError, json.JSONDecodeError):
            return None

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

        model = launcher.serving_model(policy_url)
        profile = profile_for(model)
        scenes_csv = SCENES_DIR / profile.scenes_csv if profile.scenes_csv else None
        if scenes_csv is not None and not scenes_csv.is_file():
            raise RuntimeError(f"{model}'s driver profile pins scenes in {scenes_csv}, which does not exist")

        out_dir.mkdir(parents=True, exist_ok=True)
        bridge_base, wizard_base = port_bases()
        bridge_port = ports.find_free(bridge_base)
        bridge_log = out_dir / "bridge.log"
        wizard_log = out_dir / "wizard.log"

        echo(f"{episode.scene}: bridge on :{bridge_port} -> {policy_url}")
        with launcher.BridgeProcess(
            python=python,
            policy_url=policy_url,
            port=bridge_port,
            version_id=f"paddock-bridge/{episode.scenario}",
            log_path=bridge_log,
            extra_args=profile.bridge_args
            + ("--plan-record", str(out_dir / PLAN_RECORD))
            + (
                ("--replan-log", str(out_dir.parent / f"{episode.episode_id}.replans.jsonl"))
                if profile.replan_log
                else ()
            ),
        ):
            command = launcher.wizard_command(
                repo=repo,
                python=python,
                scene_id=episode.scene,
                out_dir=out_dir,
                bridge_port=bridge_port,
                searchpath=driver_searchpath(model, profile),
                driver=profile.driver,
                startup_timeout_s=self.startup_timeout_s,
                deploy=self.setting("deploy") or launcher.DEFAULT_DEPLOY,
                topology=self.setting("topology") or launcher.DEFAULT_TOPOLOGY,
                scene_cache=self.setting("scene_cache"),
                baseport=wizard_base,
                video=self.flag("video"),
                video_every_nth_frame=int(self.setting("video_every_nth_frame") or 1),
                scenes_csv=scenes_csv,
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
