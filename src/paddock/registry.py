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

    kind: str  # uv | conda | shell
    venv: str | None = None
    python: str | None = None
    pythonpath: tuple[str, ...] = ()
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def interpreter(self, cfg: SiteConfig, **ctx: str) -> Path:
        """The interpreter to launch, with any site override applied.

        The model name comes from `ModelSpec.context`, which already supplies it
        as the {model} template variable -- taking it from there keeps one
        source of truth rather than two that can disagree.
        """
        model = ctx.get("model", "")
        override = cfg.models.get(model, None)
        if override is not None and override.venv is not None:
            return override.venv / "bin" / "python"
        if not self.venv:
            raise ConfigError(
                f"runtime kind {self.kind!r} declares no 'venv', and paddock.toml has no "
                f"[models.{model}] venv override to supply one"
            )
        return Path(cfg.resolve(self.venv, **ctx)) / "bin" / "python"


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
    benchmarks: tuple[str, ...] = ()
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
            ),
            cameras=tuple(caps.get("cameras", ("CAM_FRONT",))),
            vram_gb=float(caps.get("vram_gb", 0.0)),
            startup_timeout_s=int(caps.get("startup_timeout_s", 600)),
            waypoint_dt=float(caps.get("waypoint_dt", 0.5)),
            min_input_hz=float(caps.get("min_input_hz", 0.0)),
            native_input_hz=float(caps.get("native_input_hz", 0.0)),
            history_s=float(caps.get("history_s", 0.0)),
            benchmarks=tuple(body.get("benchmarks", ())),
            description=body.get("description", ""),
            directory=path.parent,
        )
    return models


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
        benchmarks[name] = BenchmarkSpec(
            name=name,
            mode=mode,
            dataset=body.get("dataset", ""),
            metrics=tuple(body.get("metrics", ())),
            primary_metric=body.get("primary_metric", ""),
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
