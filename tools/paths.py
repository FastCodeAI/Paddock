"""Path resolution for the diagnostic scripts in `tools/`.

These are one-off analysis and plotting scripts kept alongside the adapters.
They are not on the evaluation path, but they still must not hardcode one
machine's layout, or they stop working the moment anyone else runs them.

    from tools import paths
    scene = paths.nuscenes()
    weights = paths.repo("vision_pilot") / "VisionPilot/modules/models/weights"

Everything resolves from the same `paddock.toml` the framework uses, so a script
run here and at another site reads the same configuration.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.config import SiteConfig  # noqa: E402  (needs the path above first)


@lru_cache(maxsize=1)
def config() -> SiteConfig:
    return SiteConfig.load()


def repos() -> Path:
    """Root holding the vendor clones."""
    return config().repos


def repo(name: str) -> Path:
    """One vendor clone, e.g. `repo("openpilot")`."""
    return config().repo(name)


def runs() -> Path:
    """Where run artifacts are written."""
    return config().runs


def weights() -> Path:
    return config().weights


def dataset(name: str) -> Path:
    """A dataset's resolved root, e.g. `dataset("nuscenes")`."""
    entry = config().datasets.get(name)
    if entry is None:
        raise RuntimeError(
            f"dataset {name!r} is not configured in paddock.toml; add a "
            f"[datasets.{name}] section (see docs/data-setup.md)"
        )
    return entry.resolve()


def nuscenes() -> Path:
    return dataset("nuscenes")


def kitscenes() -> Path:
    return dataset("kitscenes")


def _benchmark_setting(benchmark: str, key: str) -> Path:
    settings = config().benchmarks.get(benchmark)
    value = settings.get(key) if settings else None
    if not value:
        raise RuntimeError(
            f"[benchmarks.{benchmark}] {key} is not set in paddock.toml; "
            f"these tools need it to find the simulator's working tree"
        )
    return Path(value)


def ncap() -> Path:
    """The neuroNCAP working tree: outputs, checkpoints, scenarios, envs."""
    return _benchmark_setting("neuroncap", "renderer_cwd")


def ncap_outputs() -> Path:
    return ncap() / "outputs"
