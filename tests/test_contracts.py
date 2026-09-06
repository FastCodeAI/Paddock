"""The mechanical half of the review checklist.

docs/reviewing.md and CONTRIBUTING.md ask a reviewer to check these by hand
on every PR. Each one is a statement about tracked files that a machine can
verify, and each has slipped before: a config key read by a harness and named
nowhere else, a patch on disk that repos.yaml did not know about, a metric
name in benchmark.yaml that no doc explains.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from paddock.config import REPO_ROOT
from paddock.registry import Registry

ADAPTERS = REPO_ROOT / "src" / "paddock" / "adapters"
DOCS = REPO_ROOT / "docs"
REGISTRY = Registry.load()

#: Models that predate the "one doc per adapter" rule and still have none.
#: Remove an entry when its doc is written; never add one for a new model.
KNOWN_MISSING_MODEL_DOCS = {"drivejepa"}


def _normalised(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# -- every adapter has a doc a stranger can follow ------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY.benchmarks))
def test_every_benchmark_has_a_doc(name: str) -> None:
    assert (DOCS / "benchmarks" / f"{name}.md").exists(), f"docs/benchmarks/{name}.md is missing"


@pytest.mark.parametrize("name", sorted(REGISTRY.models))
def test_every_model_has_a_doc(name: str) -> None:
    if name in KNOWN_MISSING_MODEL_DOCS:
        pytest.xfail(f"{name} has no doc yet; listed in KNOWN_MISSING_MODEL_DOCS")
    docs = [p.name for p in (DOCS / "models").glob("*.md") if _normalised(name) in _normalised(p.stem)]
    assert docs, f"no docs/models/*.md names {name}"


# -- every site setting a harness reads is documented in the example -----------


def _benchmark_setting_keys(harness: Path) -> set[str]:
    return set(re.findall(r'self\.setting\(\s*"([^"]+)"', harness.read_text()))


@pytest.mark.parametrize(
    "harness",
    sorted(p for p in ADAPTERS.glob("benchmarks/*/harness.py") if not p.parent.name.startswith("_")),
    ids=lambda p: p.parent.name,
)
def test_every_config_key_a_harness_reads_is_in_the_example_toml(harness: Path) -> None:
    """A key the code reads but the example never names is one that silently
    does nothing at the next site (docs/reviewing.md §7)."""
    example = (REPO_ROOT / "paddock.toml.example").read_text()
    missing = sorted(
        key
        for key in _benchmark_setting_keys(harness)
        if not re.search(rf"^#?\s*{re.escape(key)}\s*=", example, re.MULTILINE)
    )
    assert not missing, (
        f"[benchmarks.{harness.parent.name}] reads {missing}; paddock.toml.example does not show them"
    )


# -- pins and patches ----------------------------------------------------------


def test_every_repo_pins_a_sha_and_every_patch_is_listed_both_ways() -> None:
    repos = yaml.safe_load((REPO_ROOT / "repos.yaml").read_text())["repos"]
    for name, entry in repos.items():
        sha = str(entry.get("sha", ""))
        assert re.fullmatch(r"[0-9a-f]{7,40}", sha), f"{name}: sha {sha!r} is not a pin"
        listed = list(entry.get("patches") or [])
        on_disk = sorted(p.name for p in (REPO_ROOT / "patches" / name).glob("*.patch"))
        assert sorted(listed) == on_disk, (
            f"{name}: repos.yaml lists {listed}, patches/{name}/ holds {on_disk}; "
            f"`paddock setup` applies only what is listed"
        )
    orphans = sorted(p.name for p in (REPO_ROOT / "patches").iterdir() if p.is_dir() and p.name not in repos)
    assert not orphans, f"patches/ has directories for repos not in repos.yaml: {orphans}"


# -- every metric a benchmark declares is documented with its direction ----------


def _documented_metric_names() -> set[str]:
    """Names in docs/results-schema.md's tables plus each benchmark doc's own table."""
    names: set[str] = set()
    for doc in [DOCS / "results-schema.md", *(DOCS / "benchmarks").glob("*.md")]:
        for line in doc.read_text().splitlines():
            if line.startswith("|"):
                names.update(re.findall(r"`([A-Za-z0-9_@.]+)`", line.split("|")[1]))
    return names


@pytest.mark.parametrize("name", sorted(REGISTRY.benchmarks))
def test_every_declared_metric_is_documented(name: str) -> None:
    """A new metric name is either canonical or explained with a direction in a
    table a reader will find (docs/reviewing.md §6)."""
    spec = REGISTRY.benchmarks[name]
    documented = _documented_metric_names()
    undocumented = [m for m in spec.metrics if m not in documented]
    assert not undocumented, f"{name} declares {undocumented} in benchmark.yaml; no doc table names them"
