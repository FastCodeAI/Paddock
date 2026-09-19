"""Site configuration: the only place that knows where anything lives.

Paddock is shared across sites with different disks, mounts and clone
locations. The split that makes that work:

    tracked in git          what a thing IS      datasets.yaml, models/*.yaml
    site-local (gitignored) where it LIVES      paddock.toml

Tracked files never contain a real path. They refer to locations through
template variables which this module expands:

    {repos}            roots.repos
    {repo}             roots.repos/<the repo this model uses>   (context-supplied)
    {runs}             roots.runs
    {weights}          roots.weights
    {site}             the site label
    {data.NAME}        the dataset's path, or its first existing tier
    {data.NAME.tiers}  every tier, colon-joined (for search-path env vars
                       such as openpilot's KITSCENES_ROOTS)

`tests/test_portability.py` enforces the split by failing if a tracked file
contains an absolute path. That test is why this repo will still work at the
other site six months from now.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Repo root: src/paddock/config.py -> src/paddock -> src -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]

_TEMPLATE = re.compile(r"\{([a-zA-Z_][\w.]*)\}")


class ConfigError(Exception):
    """Site configuration is missing or wrong. Always actionable by the reader."""


@dataclass(frozen=True)
class Dataset:
    """Where one dataset lives here.

    Either a single `path`, or `tiers` — a search path, fastest first. Tiering
    is not a workaround: an NVMe working set can front an HDD or NAS archive so
    the hot subset is fast without duplicating hundreds of GB. openpilot's
    loader already works this way via KITSCENES_ROOTS.
    """

    name: str
    path: Path | None = None
    tiers: tuple[Path, ...] = ()
    version: str | None = None
    extra: dict[str, Path] = field(default_factory=dict)

    @property
    def roots(self) -> tuple[Path, ...]:
        """Every root to search, in priority order."""
        return self.tiers if self.tiers else ((self.path,) if self.path else ())

    def resolve(self) -> Path:
        """The first root that exists on disk.

        Falls back to the first declared root so error messages name something
        concrete rather than reporting an empty path.
        """
        for root in self.roots:
            if root.exists():
                return root
        if not self.roots:
            raise ConfigError(f"dataset {self.name!r} has neither 'path' nor 'tiers' in paddock.toml")
        return self.roots[0]

    def find(self, relative: str) -> Path | None:
        """Locate one entry across the tiers, fastest first."""
        for root in self.roots:
            candidate = root / relative
            if candidate.exists():
                return candidate
        return None


@dataclass(frozen=True)
class ModelOverride:
    """Site-local corrections to a model's declared runtime.

    Conda environments in particular live wherever a site happens to keep them,
    which is often nowhere near the repo. Rather than force every site into one
    layout, `model.yaml` declares a sensible default and paddock.toml overrides
    it where reality differs:

        [models.drivejepa]
        venv = "/wherever/this/site/keeps/conda/envs/drive-jepa"
    """

    venv: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkOverride:
    """Site-local settings a benchmark needs.

    Simulators bring their own environments and assets, which live wherever a
    site keeps them:

        [benchmarks.neuroncap]
        python = "/path/to/envs/ncap/bin/python"
        renderer_url = "http://127.0.0.1:8000"

    Values arrive as strings -- `_parse` stringifies every setting so that a
    harness reading one never has to care whether the site wrote `4` or `"4"`.
    That makes `bool(get(...))` a trap: TOML `false` becomes the string
    `"False"`, which is true. Use `flag()` for anything a site writes as a
    boolean.
    """

    settings: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.settings.get(key, default)

    def flag(self, key: str, default: bool = False) -> bool:
        """A boolean setting, read the way a site would expect to write it.

        Every falsehood TOML or a shell-shaped habit can produce -- `false`,
        `0`, `no`, `off`, an empty value -- means off. This exists because
        `bool(self.setting("keep_frames", False))` returned **True** for a site
        that had written `keep_frames = false`, so the one setting whose job is
        to stop comma2k19 retaining ~5 TB of staged frames turned retention on
        for anybody who set it explicitly. Only leaving it *out* worked.
        """
        raw = self.settings.get(key)
        if raw is None:
            return default
        return str(raw).strip().lower() not in {"false", "0", "no", "off", ""}


@dataclass(frozen=True)
class Gpu:
    budget_gb: float = 24.0
    serialize: bool = True


@dataclass(frozen=True)
class SiteConfig:
    site: str
    repos: Path
    runs: Path
    weights: Path
    datasets: dict[str, Dataset]
    models: dict[str, ModelOverride]
    benchmarks: dict[str, BenchmarkOverride]
    gpu: Gpu
    source: Path

    # -- loading ----------------------------------------------------------

    @classmethod
    def path(cls) -> Path:
        """Where paddock.toml is expected: $PADDOCK_CONFIG, else the repo root."""
        override = os.environ.get("PADDOCK_CONFIG")
        return Path(override).expanduser() if override else REPO_ROOT / "paddock.toml"

    @classmethod
    def load(cls, path: Path | None = None) -> SiteConfig:
        cfg_path = path or cls.path()
        if not cfg_path.exists():
            raise ConfigError(
                f"no site configuration at {cfg_path}\n"
                f"  cp {REPO_ROOT / 'paddock.toml.example'} {cfg_path}\n"
                f"  then edit the paths for this machine (see docs/site-setup.md)"
            )
        raw = _parse(cfg_path)

        roots = raw.get("roots", {})
        missing = [k for k in ("repos", "runs", "weights") if k not in roots]
        if missing:
            raise ConfigError(f"{cfg_path}: [roots] is missing {', '.join(missing)}")

        datasets = {name: cls._dataset(name, body) for name, body in raw.get("datasets", {}).items()}
        models = {
            name: ModelOverride(
                venv=_abs(body["venv"]) if "venv" in body else None,
                env={k: str(v) for k, v in (body.get("env") or {}).items()},
            )
            for name, body in raw.get("models", {}).items()
        }
        benchmarks = {
            name: BenchmarkOverride(settings={k: str(v) for k, v in (body or {}).items()})
            for name, body in raw.get("benchmarks", {}).items()
        }
        gpu_raw = raw.get("gpu", {})

        return cls(
            site=raw.get("site", "unnamed-site"),
            repos=_abs(roots["repos"]),
            runs=_abs(roots["runs"]),
            weights=_abs(roots["weights"]),
            datasets=datasets,
            models=models,
            benchmarks=benchmarks,
            gpu=Gpu(
                budget_gb=float(gpu_raw.get("budget_gb", 24.0)),
                serialize=bool(gpu_raw.get("serialize", True)),
            ),
            source=cfg_path,
        )

    @staticmethod
    def _dataset(name: str, body: dict) -> Dataset:
        reserved = {"path", "tiers", "version"}
        return Dataset(
            name=name,
            path=_abs(body["path"]) if "path" in body else None,
            tiers=tuple(_abs(t) for t in body.get("tiers", [])),
            version=body.get("version"),
            extra={k: _abs(v) for k, v in body.items() if k not in reserved and isinstance(v, str)},
        )

    # -- template expansion -----------------------------------------------

    def resolve(self, template: str, **context: str | Path) -> str:
        """Expand template variables in a string from a tracked config file.

        Unknown variables raise rather than silently surviving into a command
        line, where they would surface much later as a confusing file-not-found.
        """

        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in context:
                return str(context[key])
            if key in ("repos", "runs", "weights", "site"):
                return str(getattr(self, key))
            if key.startswith("data."):
                return self._resolve_data(key)
            raise ConfigError(
                f"unknown template variable {{{key}}} in {template!r}; "
                f"known: repos, runs, weights, site, data.<name>, data.<name>.tiers, "
                f"data.<name>.parent, "
                f"plus context {sorted(context)}"
            )

        return _TEMPLATE.sub(substitute, template)

    def _resolve_data(self, key: str) -> str:
        parts = key.split(".")
        name = parts[1]
        dataset = self.datasets.get(name)
        if dataset is None:
            known = ", ".join(sorted(self.datasets)) or "none configured"
            raise ConfigError(
                f"dataset {name!r} is not configured in {self.source}\n"
                f"  add a [datasets.{name}] section (see docs/data-setup.md); known: {known}"
            )
        if len(parts) == 3 and parts[2] == "tiers":
            return ":".join(str(r) for r in dataset.roots)
        if len(parts) == 3 and parts[2] == "parent":
            # The same dataset is addressed two ways by different repos: some
            # want the directory holding data/, others want data/ itself. Rather
            # than store it twice and let the two drift, store one root and
            # project it. See docs/data-setup.md.
            return str(dataset.resolve().parent)
        if len(parts) == 3:
            extra = dataset.extra.get(parts[2])
            if extra is None:
                raise ConfigError(
                    f"dataset {name!r} has no key {parts[2]!r} in {self.source} "
                    f"(has: {', '.join(sorted(dataset.extra)) or 'none'})"
                )
            return str(extra)
        return str(dataset.resolve())

    def repo(self, name: str) -> Path:
        """Where a vendor clone lives here."""
        return self.repos / name


def _abs(value: str | Path) -> Path:
    return Path(value).expanduser()


#: Parsed `paddock.toml`, keyed by identity *and* contents.
_PARSED: dict[tuple[str, int, int], dict] = {}


def _parse(cfg_path: Path) -> dict:
    """The TOML, parsed once per (path, mtime, size).

    `SiteConfig.load()` is called from places that turn out to be per-frame --
    `comma2k19.toolkit.load()` resolves the clone through it on every call, and
    the loader asks the toolkit for a rotation matrix per pose. Profiling a
    comma2k19 sweep put roughly half the runner's CPU in this one `open` and
    `tomllib.load`, most of it waiting on a network filesystem.

    Keyed on mtime and size rather than path alone because tests rewrite a
    config in place and expect the next `load()` to see it; an identity-only
    cache would make that silently stale, which is a far worse bug than the
    parse was slow.
    """
    stat = cfg_path.stat()
    key = (str(cfg_path), stat.st_mtime_ns, stat.st_size)
    cached = _PARSED.get(key)
    if cached is None:
        with cfg_path.open("rb") as fh:
            cached = tomllib.load(fh)
        _PARSED.clear()  # one live config per process; this is a cache, not a registry
        _PARSED[key] = cached
    return cached
