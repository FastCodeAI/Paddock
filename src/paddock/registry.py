"""Loading the tracked registries.

Four registries, all tracked, none containing a real path:

    repos.yaml                          upstream URL + pinned sha + patches
    datasets.yaml                       what each dataset IS: layout, splits, checks
    adapters/models/<n>/model.yaml      how to run one model
    adapters/benchmarks/<n>/benchmark.yaml   what one benchmark measures

Models and benchmarks are discovered by walking the adapter directories, so
adding one means adding a directory -- never editing this file, and never
editing anything else under src/paddock/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paddock.config import REPO_ROOT, ConfigError, SiteConfig

MODELS_DIR = REPO_ROOT / "src" / "paddock" / "adapters" / "models"
BENCHMARKS_DIR = REPO_ROOT / "src" / "paddock" / "adapters" / "benchmarks"


@dataclass(frozen=True)
class RepoSpec:
    """An upstream repository paddock drives but does not vendor."""

    name: str
    url: str
    sha: str
    patches: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset is, independent of where any site keeps it."""

    name: str
    description: str = ""
    #: Subdirectories that must exist under the resolved root for the dataset
    #: to be usable. `paddock doctor` checks these.
    requires: tuple[str, ...] = ()
    #: Named splits, for benchmarks that select by split.
    splits: tuple[str, ...] = ()
    #: Free-form notes surfaced by `paddock doctor` when the dataset is partial.
    notes: str = ""


@dataclass(frozen=True)
class Runtime:
    """How to launch a model's interpreter.

    Five Python versions across three managers is not a mess awaiting cleanup --
    it is what evaluating four independent research codebases costs. The registry
    makes it declarative instead of tribal.
    """

    kind: str  # uv | conda | shell | builtin (paddock's own interpreter)
    venv: str | None = None
    python: str | None = None
    pythonpath: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    #: Env keys a site may override without changing what is measured: thread
    #: counts, memory caps. Any other key in a site's `[models.<name>.env]`,
    #: or an `env_passthrough` name set in the launching shell, makes the run
    #: `valid: false` (`runner._env_override_reasons`), because a
    #: checkpoint or a horizon swapped in by a site would otherwise be ranked
    #: under the model's name.
    tuning_env: tuple[str, ...] = ()
    #: Names of environment variables whose *value* should come from whatever
    #: shell launched `paddock run`, not a template resolved here -- ablation
    #: knobs like ALPAMAYO_NUM_TRAJ_SAMPLES that a debugging script sets ad hoc.
    #: `env` above is for the opposite case: a fixed value this model always
    #: needs. Without this, `FOO=1 paddock run ...` silently does nothing,
    #: because runtime/env.py builds the subprocess's environment from a
    #: strict allowlist rather than inheriting the caller's wholesale.
    env_passthrough: tuple[str, ...] = ()

    def interpreter(self, cfg: SiteConfig, **ctx: str) -> Path:
        """The interpreter to launch, with any site override applied.

        The model name comes from `ModelSpec.context`, which already supplies it
        as the {model} template variable -- taking it from there keeps one
        source of truth rather than two that can disagree.
        """
        model = ctx.get("model", "")
        override = cfg.models.get(model, None)
        # An explicit venv override always wins (a site may redirect any model).
        if override is not None and override.venv is not None:
            return override.venv / "bin" / "python"
        # Otherwise the builtin kind runs on paddock's own interpreter -- even
        # when a site sets an env-only override (which leaves venv None); that
        # must not fall through to the "no venv" error below.
        if self.kind == "builtin":
            import sys

            return Path(sys.executable)
        if not self.venv:
            raise ConfigError(
                f"runtime kind {self.kind!r} declares no 'venv', and paddock.toml has no "
                f"[models.{model}] venv override to supply one"
            )
        return Path(cfg.resolve(self.venv, **ctx)) / "bin" / "python"


@dataclass(frozen=True)
class TrainedOn:
    """One dataset a model saw during training.

    `dataset` should name a key in `datasets.yaml` where it is one -- that is
    what makes an overlap with a benchmark's data detectable. A corpus paddock
    does not know about (comma's fleet, NVIDIA internal) is still worth
    recording: the useful claim is usually "none of ours".

    This is a **self-declaration, not a measurement** -- the same class of claim
    as a pinned sha in repos.yaml. Nothing verifies it.
    """

    dataset: str
    split: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    runtime: Runtime
    cameras: tuple[str, ...] = ("CAM_FRONT",)
    vram_gb: float = 0.0
    startup_timeout_s: int = 600
    waypoint_dt: float = 0.5
    #: Slowest observation rate, Hz, at which this model still perceives motion.
    #: 0 means it does not care. Models that infer ego motion from frame-to-frame
    #: change do care, sharply: openpilot reads a 0.5 s step as a parked car, and
    #: a benchmark whose data is slower than this measures the gap rather than
    #: the model. Benchmarks check it before scoring.
    min_input_hz: float = 0.0
    #: Observation rate the model was built for, Hz, when it also tolerates
    #: slower input. 0 means unstated. alpamayo takes 0.1 s frames and works at
    #: 0.5 s ones, differently; a benchmark holding both rates should use this.
    native_input_hz: float = 0.0
    #: Seconds of observation history the model consumes before a plan. 0 means
    #: unstated, and a benchmark with a history of its own feeds that. openpilot
    #: fills a 96-frame feature buffer at 20 Hz, so it asks for 4.8 s.
    history_s: float = 0.0
    #: How often the model replans in closed loop, Hz; 0 means every simulator
    #: tick. A harness that honours it (Bench2Drive) asks the node for a plan
    #: this often and follows the last one, unchanged, in between. Drive-JEPA
    #: plans every 0.5 s: driven at Bench2Drive's 20 Hz it scored 31.71 on the 50
    #: routes, at 2 Hz 57.60 (docs/benchmarks/bench2drive-native-rate.md).
    planning_hz: float = 0.0

    # -- the model card ---------------------------------------------------
    # What the model IS, as opposed to how to launch it. Each field is here
    # because it answers a question a reader of the support matrix asks and
    # paddock could not previously answer. Absent is honest: `None` means
    # nobody has recorded it, and reads as "-" rather than as zero.
    # See docs/models/comparison.md for the values and where they came from.

    #: One line. Enough to tell two rows of the matrix apart.
    architecture: str = ""
    #: Millions of parameters.
    params_m: float | None = None
    #: Multiply-accumulates per forward pass, billions. Nobody has computed one
    #: for any model yet; the slot exists so the deployment column has somewhere
    #: to land.
    macs_g: float | None = None
    #: On-disk size of the weights actually loaded, GB.
    size_gb: float | None = None
    #: fp32 | bf16 | fp16 | int8 | mixed
    precision: str = ""
    #: Per-camera input, (width, height) in pixels.
    input_size: tuple[int, int] | None = None
    #: Frames of camera history one inference consumes. 1 means the current
    #: frame only.
    history_frames: int | None = None
    #: Rate of those camera frames, Hz: 2 means 0.5 s apart. None where it is not
    #: recorded, or the model reads one frame.
    camera_hz: float | None = None
    #: True where the adapter feeds the last `history_frames` requests as they
    #: arrive, so their spacing is the benchmark's step and `camera_hz` is only the
    #: rate the model was trained at. visionpilot pairs the previous request: on a
    #: 2 Hz benchmark its two frames are 0.5 s apart, not 0.1 s.
    camera_frames_by_request: bool = False
    #: The views the model actually reads, where paddock makes some of them out of
    #: `cameras`: alpamayo's telephoto slot is a crop of CAM_FRONT. Empty means
    #: the model reads `cameras` as declared.
    camera_views: tuple[str, ...] = ()
    #: Past ego states one inference reads (poses, or rows of speed and
    #: acceleration), and their rate in Hz. Camera and ego history often run at
    #: different rates: qwendrive reads frames 0.5 s apart and poses 0.1 s apart.
    #: None where the model reads none, or only the current speed.
    ego_history_steps: int | None = None
    ego_history_hz: float | None = None
    #: waypoints | controls | perception. Explains an `n/a` in an ADE column
    #: without sending the reader to find a footnote.
    output_kind: str = "waypoints"
    #: Planning horizon, seconds. None where the output is not time-parameterised.
    horizon_s: float | None = None
    #: Rate of the plan points the model emits, Hz, before any adapter resamples
    #: them to `waypoint_dt`. None where they are not evenly spaced (openpilot's
    #: 33 sit on a quadratic grid) or there is no plan. How far apart one plan's
    #: points are -- not `planning_hz`, how often the model makes a new plan.
    plan_points_hz: float | None = None
    #: True if repeated inference on identical input can differ. A single run of
    #: a stochastic model is a draw, not a measurement: alpamayo's frontal
    #: neuroNCAP score has sd 1.817 across seeds on one scenario.
    stochastic: bool = False
    #: Whether ego speed reaches the model at all. openpilot's complete
    #: non-image input set is {desire_pulse, traffic_convention, action_t} --
    #: which is why it reads a 2 Hz dataset as a parked car and why its
    #: min_input_hz is load-bearing rather than cautious.
    needs_speed: bool = True
    #: Datasets seen in training. Empty means "not recorded", which is not the
    #: same claim as "trained on nothing" -- say which you mean in the adapter.
    trained_on: tuple[TrainedOn, ...] = ()

    benchmarks: tuple[str, ...] = ()
    #: Env keys that change what a score means, each with the line the
    #: leaderboard prints under a ranked run that set it to anything but the
    #: declared default. `{value}` in the line is replaced by what the run set.
    #: Only the model knows that a horizon was stretched or a camera repeated:
    #: the report sees a name and a number, and the row reads as comparable.
    score_notes: dict[str, str] = field(default_factory=dict)
    description: str = ""
    directory: Path = REPO_ROOT

    def context(self, cfg: SiteConfig) -> dict[str, str]:
        """Template variables available to this model's config strings."""
        return {"repo": str(cfg.repo(self.repo)), "model": self.name}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    mode: str  # open_loop | closed_loop
    dataset: str
    metrics: tuple[str, ...] = ()
    primary_metric: str = ""
    #: (slot, metric) pairs: which declared metric fills which term of NAVSIM
    #: v1's PDM score (`paddock.metrics.pdm.SLOTS`, plus `score`), in slot
    #: order. The regression gate lays every benchmark out in these rows, so a
    #: candidate that trades collisions for progress reads the same everywhere.
    pdm: tuple[tuple[str, str], ...] = ()
    description: str = ""
    directory: Path = REPO_ROOT


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_repos(root: Path = REPO_ROOT) -> dict[str, RepoSpec]:
    raw = _load_yaml(root / "repos.yaml")
    return {
        name: RepoSpec(
            name=name,
            url=body["url"],
            sha=body["sha"],
            patches=tuple(body.get("patches", ())),
            description=body.get("description", ""),
        )
        for name, body in (raw.get("repos") or {}).items()
    }


def load_datasets(root: Path = REPO_ROOT) -> dict[str, DatasetSpec]:
    raw = _load_yaml(root / "datasets.yaml")
    return {
        name: DatasetSpec(
            name=name,
            description=body.get("description", ""),
            requires=tuple(body.get("requires", ())),
            splits=tuple(body.get("splits", ())),
            notes=body.get("notes", ""),
        )
        for name, body in (raw.get("datasets") or {}).items()
    }


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _input_size(value: object, path: Path) -> tuple[int, int] | None:
    """Per-camera input as (width, height).

    Refuses anything but a two-item sequence rather than accepting it and
    surfacing much later as a silently transposed image size.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{path}: capabilities.input_size must be [width, height], got {value!r}")
    return (int(value[0]), int(value[1]))


def _trained_on(value: object, path: Path) -> tuple[TrainedOn, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"{path}: capabilities.trained_on must be a list, got {value!r}")
    out = []
    for entry in value:
        if not isinstance(entry, dict) or "dataset" not in entry:
            raise ConfigError(
                f"{path}: each capabilities.trained_on entry needs a 'dataset' key, got {entry!r}"
            )
        out.append(TrainedOn(dataset=str(entry["dataset"]), split=str(entry.get("split", ""))))
    return tuple(out)


def load_models(directory: Path = MODELS_DIR) -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {}
    for path in sorted(directory.glob("*/model.yaml")):
        if path.parent.name.startswith("_"):
            continue  # _template is a starting point, not a model
        body = _load_yaml(path)
        rt = body.get("runtime", {})
        caps = body.get("capabilities", {})
        name = body.get("name", path.parent.name)
        models[name] = ModelSpec(
            name=name,
            repo=body.get("repo", name),
            runtime=Runtime(
                kind=rt.get("kind", "shell"),
                venv=rt.get("venv"),
                python=rt.get("python"),
                pythonpath=tuple(rt.get("pythonpath", ())),
                cwd=rt.get("cwd"),
                env=dict(rt.get("env", {})),
                tuning_env=tuple(rt.get("tuning_env", ())),
                env_passthrough=tuple(rt.get("env_passthrough", ())),
            ),
            cameras=tuple(caps.get("cameras", ("CAM_FRONT",))),
            vram_gb=float(caps.get("vram_gb", 0.0)),
            startup_timeout_s=int(caps.get("startup_timeout_s", 600)),
            waypoint_dt=float(caps.get("waypoint_dt", 0.5)),
            min_input_hz=float(caps.get("min_input_hz", 0.0)),
            native_input_hz=float(caps.get("native_input_hz", 0.0)),
            history_s=float(caps.get("history_s", 0.0)),
            planning_hz=float(caps.get("planning_hz", 0.0)),
            architecture=caps.get("architecture", ""),
            params_m=_optional_float(caps.get("params_m")),
            macs_g=_optional_float(caps.get("macs_g")),
            size_gb=_optional_float(caps.get("size_gb")),
            precision=caps.get("precision", ""),
            input_size=_input_size(caps.get("input_size"), path),
            history_frames=_optional_int(caps.get("history_frames")),
            camera_hz=_optional_float(caps.get("camera_hz")),
            camera_frames_by_request=bool(caps.get("camera_frames_by_request", False)),
            camera_views=tuple(caps.get("camera_views", ())),
            ego_history_steps=_optional_int(caps.get("ego_history_steps")),
            ego_history_hz=_optional_float(caps.get("ego_history_hz")),
            output_kind=caps.get("output_kind", "waypoints"),
            horizon_s=_optional_float(caps.get("horizon_s")),
            plan_points_hz=_optional_float(caps.get("plan_points_hz")),
            stochastic=bool(caps.get("stochastic", False)),
            needs_speed=bool(caps.get("needs_speed", True)),
            trained_on=_trained_on(caps.get("trained_on"), path),
            benchmarks=tuple(body.get("benchmarks", ())),
            score_notes=dict(body.get("score_notes") or {}),
            description=body.get("description", ""),
            directory=path.parent,
        )
    return models


def _pdm(body: object, metrics: tuple[str, ...], path: Path) -> tuple[tuple[str, str], ...]:
    """`pdm:` as (slot, metric) pairs in slot order, refused when the gate could not read it.

    A metric the harness never declares pairs as NO RECORDS on every run, and
    one with no better direction cannot be signed: both would look like a term
    that is being checked when it is not.
    """
    from paddock.metrics.pdm import SCORE, SLOTS
    from paddock.results.report import has_direction

    if body is None:
        return ()
    if not isinstance(body, dict):
        raise ConfigError(f"{path}: pdm must map a term to a metric, got {body!r}")
    order = (*SLOTS, SCORE)
    problems = [f"{k!r} is not a PDM term (one of {', '.join(order)})" for k in body if k not in order]
    problems += [f"{v!r} is not in metrics" for v in body.values() if v not in metrics]
    problems += [f"{v!r} has no better direction" for v in body.values() if not has_direction(str(v))]
    if len(set(body.values())) != len(body):
        problems.append("one metric fills two terms")
    if problems:
        raise ConfigError(f"{path}: pdm: {'; '.join(problems)}")
    return tuple((k, str(body[k])) for k in order if k in body)


def load_benchmarks(directory: Path = BENCHMARKS_DIR) -> dict[str, BenchmarkSpec]:
    benchmarks: dict[str, BenchmarkSpec] = {}
    for path in sorted(directory.glob("*/benchmark.yaml")):
        if path.parent.name.startswith("_"):
            continue
        body = _load_yaml(path)
        name = body.get("name", path.parent.name)
        mode = body.get("mode", "open_loop")
        if mode not in ("open_loop", "closed_loop"):
            raise ConfigError(f"{path}: mode must be open_loop or closed_loop, got {mode!r}")
        metrics = tuple(body.get("metrics", ()))
        benchmarks[name] = BenchmarkSpec(
            name=name,
            mode=mode,
            dataset=body.get("dataset", ""),
            metrics=metrics,
            primary_metric=body.get("primary_metric", ""),
            pdm=_pdm(body.get("pdm"), metrics, path),
            description=body.get("description", ""),
            directory=path.parent,
        )
    return benchmarks


@dataclass(frozen=True)
class Registry:
    repos: dict[str, RepoSpec]
    datasets: dict[str, DatasetSpec]
    models: dict[str, ModelSpec]
    benchmarks: dict[str, BenchmarkSpec]

    @classmethod
    def load(cls, root: Path = REPO_ROOT) -> Registry:
        return cls(
            repos=load_repos(root),
            datasets=load_datasets(root),
            models=load_models(root / "src" / "paddock" / "adapters" / "models"),
            benchmarks=load_benchmarks(root / "src" / "paddock" / "adapters" / "benchmarks"),
        )
