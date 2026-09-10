"""The tracked results store is reviewed by diff; this is the part a machine can check.

Every `results/<benchmark>/<model>/<run_id>/` is a claim someone will compare
numbers against. The claims that are mechanical -- the record parses, it is
filed where it says it belongs, its aggregate agrees with its sample count, a
run marked bad says why -- are pinned here so a PR that adds a row cannot
break them quietly. What the numbers mean is still a human's job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddock.registry import Registry
from paddock.results.store import RESULTS_ROOT, MetricsRecord, RunRecord

RUNS = sorted(RESULTS_ROOT.rglob("run.json"))


def test_the_store_has_runs() -> None:
    assert RUNS, "results/ holds no run.json; the store tests would pass vacuously"


def test_every_top_level_directory_is_a_registered_benchmark() -> None:
    """`results/fake/` from a local experiment must not be committed by accident."""
    known = set(Registry.load().benchmarks)
    found = {p.name for p in RESULTS_ROOT.iterdir() if p.is_dir()}
    assert found <= known, f"results/ has directories for unregistered benchmarks: {sorted(found - known)}"


@pytest.mark.parametrize("run_json", RUNS, ids=lambda p: str(p.parent.relative_to(RESULTS_ROOT)))
def test_each_run_is_filed_consistently_and_explains_itself(run_json: Path) -> None:
    rel = run_json.parent.relative_to(RESULTS_ROOT)
    benchmark, model, run_id = rel.parts
    run = RunRecord(**json.loads(run_json.read_text()))
    metrics_json = run_json.parent / "metrics.json"
    assert metrics_json.exists(), f"{rel}: no metrics.json beside run.json"
    metrics = MetricsRecord(**json.loads(metrics_json.read_text()))

    assert run.run_id == run_id, f"{rel}: run_id {run.run_id!r} does not match its directory"
    assert metrics.run_id == run_id, f"{rel}: metrics.json belongs to {metrics.run_id!r}"
    assert (run.benchmark, run.model) == (benchmark, model), (
        f"{rel}: record says {run.benchmark}/{run.model}, filed under {benchmark}/{model}"
    )
    assert run.status in {"complete", "partial"}, f"{rel}: status {run.status!r}"
    assert metrics.n == run.dataset.get("n_samples"), (
        f"{rel}: metrics.json n={metrics.n} but run.json n_samples={run.dataset.get('n_samples')}"
    )
    assert metrics.primary_metric in metrics.metrics or not metrics.metrics, (
        f"{rel}: primary metric {metrics.primary_metric!r} is not among the metrics recorded"
    )
    if not run.valid:
        assert run.notes.strip(), f"{rel}: valid=false with no note saying why"
    if run.status == "partial":
        assert run.notes.strip(), f"{rel}: partial with no note saying what was lost"
    assert run.site, f"{rel}: no site recorded"


# --- aggregation: `group` partitions a run the same way `split` does ---------
#
# nuScenes sets `group` to the scene so per-scene L2 lands in the tracked
# metrics.json. Before that, answering "how did this model do on scene 0103"
# meant re-reading the per-sample records outside the repo -- which for the
# September nuScenes runs no longer exist.


def _sample(metric: float, split: str = "val", group: str = "") -> dict:
    return {"sample_id": f"s{metric}", "split": split, "group": group, "metrics": {"l2": metric}}


def test_by_group_aggregates_each_group_separately() -> None:
    record = MetricsRecord.from_samples(
        "r",
        [
            _sample(1.0, group="scene-0103"),
            _sample(3.0, group="scene-0103"),
            _sample(10.0, group="scene-0110"),
        ],
        primary_metric="l2",
    )
    assert set(record.by_group) == {"scene-0103", "scene-0110"}
    assert record.by_group["scene-0103"]["n"] == 2
    assert record.by_group["scene-0103"]["l2"]["mean"] == pytest.approx(2.0)
    assert record.by_group["scene-0110"]["n"] == 1
    # the overall figure still spans every sample
    assert record.n == 3
    assert record.metrics["l2"]["mean"] == pytest.approx(14.0 / 3)


def test_samples_without_a_group_leave_by_group_empty() -> None:
    """Every benchmark that predates `group` keeps its current record shape."""
    record = MetricsRecord.from_samples("r", [_sample(1.0), _sample(2.0)], primary_metric="l2")
    assert record.by_group == {}
    assert record.by_split["val"]["n"] == 2


def test_group_and_split_partition_independently() -> None:
    """A scene spans splits and a split spans scenes; neither may absorb the other."""
    record = MetricsRecord.from_samples(
        "r",
        [
            _sample(1.0, split="val", group="scene-0103"),
            _sample(5.0, split="test", group="scene-0103"),
        ],
        primary_metric="l2",
    )
    assert record.by_group["scene-0103"]["n"] == 2
    assert record.by_split["val"]["n"] == 1
    assert record.by_split["test"]["n"] == 1
