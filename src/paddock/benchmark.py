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

    @property
    def settings(self):
        """This benchmark's `[benchmarks.<name>]` block, or None when the site has none."""
        return self.cfg.benchmarks.get(self.spec.name)

    def setting(self, key: str, default: str | None = None) -> str | None:
        """One key from this benchmark's `[benchmarks.<name>]` block, or *default*.

        Tolerates the block being absent, which `BenchmarkOverride.get` alone
        does not. Defined here once, and used by every adapter, so none carries
        its own copy -- docs/reviewing.md names that duplication as a finding.
        """
        settings = self.settings
        return settings.get(key, default) if settings is not None else default

    def run_config(self) -> dict[str, Any]:
        """Site settings and switches that change *what is measured*.

        The runner hashes these into the run id and writes them to `run.json`,
        so two runs that measured different things never share a config hash
        and a stored row says how it was produced: a routes subset, a
        non-default scorer agent, a switch that lets a partial split run.
        Anything read from the environment or the site config that alters the
        number belongs here. Default: nothing.
        """
        return {}

    def invalid_reasons(self) -> list[str]:
        """Why a run made under the current settings must be stored `valid: false`.

        A subset of the benchmark, a vendor agent that sees the scorer, a
        rate-gap override: each is worth keeping and none may enter an average.
        The runner joins these into `notes`, so the rule is a mechanism rather
        than a discipline. Default: none.
        """
        return []

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


@dataclass
class Evaluation:
    """What an external scorer produced for one model.

    `samples`: per-sample records in the results store's shape (`sample_id`,
    `native_id`, `split`, `metrics`), one per scenario the scorer actually
    scored. `failed`: the ids of scenarios it could not score, kept apart so
    they mark the run rather than average into it. `by_group`: run-level
    context the scorer reported (its own headline, its aggregate rows), stored
    in the tracked `metrics.json` where the ingest path puts the same thing.
    """

    samples: list[dict[str, Any]]
    failed: list[str] = field(default_factory=list)
    by_group: dict[str, Any] = field(default_factory=dict)


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

    @property
    def run_retries(self) -> int:
        """How many times the runner may restart the nodes and re-invoke
        `evaluate` on the same artifacts after a failure. Zero unless the
        benchmark can resume -- retrying a scorer that starts from scratch
        would only repeat the loss."""
        return 0

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

    def model_problems(self, model, limit: int | None = None) -> list[str]:
        """Why *model* cannot be scored here at all, decided before it is loaded.

        Pure in the model spec and the requested limit -- an input-rate gap, a
        sample count the scorer cannot aggregate -- so the runner asks before
        waiting for VRAM and reading a checkpoint, and `--dry-run` reports it.
        `evaluate` must refuse the same things, for callers that skip the runner.
        """
        return []

    @abstractmethod
    def evaluate(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None = None,
        echo=print,
    ) -> Evaluation:
        """Drive the external scorer against a running policy node.

        `config` carries what the model contributes to the run -- its declared
        cameras, above all, since those decide what the scorer loads.

        Returns an `Evaluation`; see its docstring for the three parts.
        """
