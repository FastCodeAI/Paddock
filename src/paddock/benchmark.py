"""What a benchmark must implement.

Two shapes, because driving benchmarks come in two genuinely different kinds and
pretending otherwise helps nobody:

**Open loop** -- the dataset decides what happens next. Paddock iterates samples,
calls `Policy.infer` on each, and scores the predictions against recorded ground
truth. Errors do not compound; the ego never deviates from what actually
happened. KITScenes, nuScenes and NAVSIM's PDMS proxy work this way.

**Closed loop** -- the model decides what happens next. A simulator drives the
policy over HTTP, feeds it back its own consequences, and scores the outcome.
Errors compound, which is the point. neuroNCAP and Bench2Drive work this way.

The same `Policy` serves both. That is the whole reason this file has two
classes instead of two frameworks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paddock.config import SiteConfig
from paddock.policy.protocol import Observation, Trajectory
from paddock.registry import BenchmarkSpec


@dataclass
class Sample:
    """One scoreable unit of an open-loop benchmark."""

    sample_id: str
    observation: Observation
    #: Recorded future, ego frame, same convention as Trajectory: x fwd, y left.
    ground_truth: Any = None
    split: str = ""
    #: Identity as the source dataset expressed it, preserved for traceability.
    native_id: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """One scoreable unit of a closed-loop benchmark."""

    episode_id: str
    scenario: str
    scene: str
    run_index: int = 0
    config: dict[str, Any] = field(default_factory=dict)


class Benchmark(ABC):
    """Common surface. Subclass OpenLoopBenchmark or ClosedLoopBenchmark."""

    def __init__(self, spec: BenchmarkSpec, cfg: SiteConfig) -> None:
        self.spec = spec
        self.cfg = cfg

    @property
    def dataset(self):
        return self.cfg.datasets.get(self.spec.dataset)

    @abstractmethod
    def preflight(self) -> list[str]:
        """Return human-readable reasons this benchmark cannot run here.

        Empty list means ready. Called by `paddock doctor` and before every run,
        so a missing scene or absent checkpoint is reported in seconds rather
        than an hour into a sweep.
        """


class OpenLoopBenchmark(Benchmark):
    @abstractmethod
    def samples(self, limit: int | None = None, split: str | None = None) -> Iterator[Sample]:
        """Yield samples to evaluate, in a deterministic order."""

    @abstractmethod
    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """Metrics for one prediction, keyed by canonical metric name.

        Per-sample, not aggregated: paddock aggregates, so that every benchmark
        aggregates the same way and per-sample records stay inspectable.
        """


class ClosedLoopBenchmark(Benchmark):
    @abstractmethod
    def episodes(self, scenes: list[str] | None = None, runs: int = 1) -> Iterator[Episode]:
        """Yield episodes to simulate."""

    @abstractmethod
    def run_episode(self, episode: Episode, policy_url: str, out_dir: Path) -> dict[str, float]:
        """Drive one episode against a running policy node; return its metrics.

        `policy_url` is a node paddock has already started and health-checked.
        The benchmark owns the simulator; paddock owns the model process.
        """
