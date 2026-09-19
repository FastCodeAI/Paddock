"""Golden test for the template benchmark adapter.

Every benchmark adapter ships one. Two jobs:

1. **Contract.** `paddock.conformance.benchmark_findings` is the same set of
   checks `paddock verify --bench <name>` runs, including the two that matter
   most -- scoring the recorded future against itself must give ~0, and an
   all-zero plan must score unmistakably badly. Both need the dataset, so this
   part skips when it is absent rather than failing on a reviewer's laptop.
2. **Golden numbers.** Pin a handful of samples and their expected metric
   values, so a refactor that changes the numbers fails loudly instead of
   silently rewriting history.
"""

from __future__ import annotations

import numpy as np
import pytest


def _benchmark():
    """The adapter, or a skip if this machine cannot serve its dataset."""
    from paddock.config import ConfigError, SiteConfig
    from paddock.registry import Registry

    try:
        cfg = SiteConfig.load()
    except ConfigError as exc:
        pytest.skip(f"no site config: {exc}")
    registry = Registry.load()
    spec = registry.benchmarks.get("_template")
    if spec is None:
        pytest.skip("_template is not a registered benchmark")

    import importlib

    module = importlib.import_module("paddock.adapters.benchmarks._template.harness")
    bench = module.BENCHMARK(spec, cfg)
    problems = bench.preflight()
    if problems:
        pytest.skip(f"benchmark not runnable here: {problems[0]}")
    return bench, spec


def test_conformance() -> None:
    """The shared Benchmark contract. Equivalent to `paddock verify --bench <name>`."""
    from paddock import conformance

    bench, spec = _benchmark()
    findings = conformance.benchmark_findings(bench, spec)
    failures = [f for f in findings if f.failed]
    assert not failures, "\n" + "\n".join(f.render() for f in failures)


def test_metrics_are_stable() -> None:
    """A straight-line prediction against a straight-line future scores zero."""
    from paddock.metrics import trajectory as M

    pred = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    gt = pred.copy()

    assert M.ade(pred, gt) == 0.0
    assert M.fde(pred, gt, t=3.0, dt=0.5) == 0.0


def test_lateral_offset_is_measured() -> None:
    from paddock.metrics import trajectory as M

    pred = np.stack([np.arange(1, 7) * 5.0, np.full(6, 2.0)], axis=1)
    gt = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)

    assert M.ade(pred, gt) == 2.0


def test_a_stationary_plan_is_distinguishable_from_a_good_one() -> None:
    """The failure mode a lateral-only metric set cannot see.

    A model predicting a stationary car has a *small* axis-aligned lateral error
    -- its prediction sits at the origin and a straight recorded path has little
    lateral excursion -- so `lateral_mae` alone reads as "tracks the lane".
    `paddock.metrics.openloop.decompose` reports `path_length_ratio`, which is
    ~0 for such a plan and ~1 when the distance is right. If your benchmark
    defines its own metrics, keep something with this property.
    """
    from paddock.metrics import openloop

    gt = np.stack([np.arange(1, 11) * 10.0, np.linspace(0, 1.0, 10)], axis=1)
    stationary = openloop.decompose(np.zeros((10, 2)), gt, eval_horizon_s=5.0)

    assert stationary["lateral_mae"] < 1.0  # looks fine, and is not
    assert stationary["along_track_mae"] > 25.0  # the honest signal
    assert stationary["path_length_ratio"] < 0.01  # unmistakable
