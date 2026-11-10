"""Template benchmark adapter.

Copy this directory, rename it, and implement one of the two shapes. Delete the
one you are not using.

Get `preflight()` right before anything else. It is what turns "the sweep died
after fifty minutes" into "scene 0061 has no renderer checkpoint, here is why".
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterator
from pathlib import Path

import numpy as np

from paddock.benchmark import ClosedLoopBenchmark, Episode, OpenLoopBenchmark, Sample
from paddock.policy.protocol import Trajectory


class TemplateBenchmark(OpenLoopBenchmark):
    def preflight(self) -> list[str]:
        """Everything that must be true, checked cheaply, before any model loads."""
        problems: list[str] = []
        dataset = self.dataset
        if dataset is None:
            problems.append(
                f"dataset {self.spec.dataset!r} is not configured; "
                f"add [datasets.{self.spec.dataset}] to paddock.toml"
            )
            return problems
        root = dataset.resolve()
        if not root.exists():
            problems.append(f"{self.spec.dataset} root does not exist: {root}")
        return problems

    def samples(
        self,
        limit: int | None = None,
        split: str | None = None,
        skip: Container[str] = (),
    ) -> Iterator[Sample]:
        """Yield samples in a deterministic order.

        Determinism matters more than it looks: `--limit 5` must select the same
        five samples at every site, or two people comparing smoke tests will
        reach different conclusions. Sort explicitly; do not rely on the order
        the filesystem hands back.

        `skip` holds sample ids `--resume` already has on disk. Drop them as
        early as you can -- before reading or decoding anything a skipped
        sample would need -- or resuming a long run costs what running it cost.
        """
        raise NotImplementedError("implement samples() for your benchmark")

    def score(self, sample: Sample, prediction: Trajectory) -> dict[str, float]:
        """Metrics for one prediction.

        Use paddock.metrics rather than writing your own ADE. The shared
        implementation handles the alignment rule that catches people out:
        interpolate ground truth onto the prediction's timestamps, and never
        extrapolate past the end of the recorded future -- trim instead.
        """
        from paddock.metrics import trajectory as M

        gt = np.asarray(sample.ground_truth, dtype=np.float64)
        return {
            "ade": M.ade(prediction.xy, gt[:, :2]),
            "fde@3s": M.fde(prediction.xy, gt[:, :2], t=3.0, dt=prediction.dt),
        }


class TemplateClosedLoopBenchmark(ClosedLoopBenchmark):
    """Delete this class if your benchmark is open-loop."""

    def preflight(self) -> list[str]:
        return []

    def episodes(self, scenes: list[str] | None = None, runs: int = 1) -> Iterator[Episode]:
        raise NotImplementedError

    def run_episode(
        self,
        episode: Episode,
        policy_url: str,
        out_dir: Path,
        echo: Callable[[str], None] = print,
    ) -> dict[str, float]:
        #: `echo` is the run's own output stream -- say anything about how the
        #: episode went through it, so a detached sweep's log holds the whole
        #: story. The runner calls this positionally, so keep the parameter.
        raise NotImplementedError


#: Paddock looks for this name when loading an adapter.
BENCHMARK = TemplateBenchmark
