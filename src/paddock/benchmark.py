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
    """One scoreable unit of an open-loop benchmark.

    Carries a *sequence*, not a single frame. Several models need real history
    to produce their scored plan -- openpilot fills a 24-deep feature buffer,
    Alpamayo differences 16 ego poses to derive its speed -- and handing them
    one frame would either fail or quietly measure a model working from a
    history it invented.
    """

    sample_id: str
    #: The frames leading to the scored moment, oldest first; the last is the
    #: one scored. A model needing no history gets a one-element list.
    observations: list[Observation]
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


class ExternallyScoredBenchmark(Benchmark):
    """Dataset-driven like open loop, but the scorer owns the iteration.

    A third shape, added because NAVSIM genuinely is one. It is open loop in the
    sense that matters -- the dataset decides what happens next and errors do not
    compound -- so `mode` in benchmark.yaml stays `open_loop` and the leaderboard
    reads correctly. What differs is who drives: NAVSIM's PDM scorer loads its
    own scenes, re-simulates each predicted trajectory through a kinematic model
    against a cached ground truth, and computes metrics that are not per-sample
    functions at all (two-frame comfort needs the preceding frame). Paddock
    cannot iterate that from the outside without reimplementing it, and
    reimplementing a scorer is how two sites end up reporting different numbers
    for the same model.

    So paddock does what it does everywhere else -- own the model process, hand
    the benchmark a running node, and store what comes back.
    """

    def node_count(self, model) -> int:
        """How many policy nodes to start for *model*. One unless the benchmark
        knows better.

        A scorer that runs N worker processes against one node serialises on
        it: a CPU-bound model that answers one request at a time leaves N-1
        workers waiting. Starting N nodes and spreading workers across them is
        an N-fold speedup with no change to what is computed, since every
        request is a self-contained `infer_sequence`.
        """
        return 1

    @abstractmethod
    def evaluate(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None = None,
        echo=print,
    ) -> list[dict[str, Any]]:
        """Drive the external scorer against a running policy node.

        `config` carries what the model contributes to the run -- its declared
        cameras, above all, since those decide what the scorer loads.

        Returns per-sample records in the results store's shape
        (`sample_id`, `native_id`, `split`, `metrics`), one per scored scenario,
        so an externally scored run is as inspectable as any other.
        """
