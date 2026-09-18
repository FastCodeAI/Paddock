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
    if run.superseded_by:
        # The replacement has to exist, under the same model, and be the one a
        # reader would actually land on. A chain ending nowhere leaves the model
        # with no ranked row, which is worse than the two rows it was fixing.
        later = RESULTS_ROOT / benchmark / model / run.superseded_by / "run.json"
        assert later.exists(), f"{rel}: superseded by {run.superseded_by}, which is not in results/"
        replacement = RunRecord(**json.loads(later.read_text()))
        assert replacement.valid, f"{rel}: superseded by {run.superseded_by}, which is itself invalid"
        assert not run.valid, f"{rel}: superseded but still ranked"
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


REPLAYS = [p for p in RUNS if (json.loads(p.read_text()).get("config") or {}).get("replayed_from")]


@pytest.mark.parametrize("run_json", REPLAYS, ids=lambda p: str(p.parent.relative_to(RESULTS_ROOT)))
def test_a_replay_names_a_source_run_that_is_in_the_store(run_json: Path) -> None:
    """A replay scored plans another recorded run produced. Its `replayed_from`
    is the provenance a reader follows to find what made them, so the source
    has to be in the store, under the model the replay is filed under -- not in
    another branch, and not only in an artifacts directory somewhere."""
    rel = run_json.parent.relative_to(RESULTS_ROOT)
    run = json.loads(run_json.read_text())
    ref = run["config"]["replayed_from"]
    source = RESULTS_ROOT / ref / "run.json"
    assert source.exists(), f"{rel}: replayed from {ref}, which is not in results/"
    src = json.loads(source.read_text())
    bench, model, _ = ref.split("/")
    assert (src["benchmark"], src["model"]) == (bench, model), (
        f"{rel}: {ref} names {src['benchmark']}/{src['model']}"
    )
    assert src["model"] == run["model"], (
        f"{rel}: filed under {run['model']} but replays {src['model']}'s plans"
    )


# --- supersede: the verb that exists so nobody edits `valid` by hand ---------


def _store(tmp_path: Path, run_id: str, **fields) -> Path:
    from paddock.results.store import RunRecord as R

    directory = tmp_path / "comma2k19" / "openpilot" / run_id
    directory.mkdir(parents=True)
    record = R(run_id=run_id, site="test", benchmark="comma2k19", model="openpilot", **fields)
    (directory / "run.json").write_text(json.dumps(record.to_json(), indent=2) + "\n")
    return directory / "run.json"


def test_supersede_retires_the_older_row_and_leaves_its_measurements_alone(tmp_path: Path) -> None:
    """The point of the verb: the numbers are still true, the ranking is not."""
    from paddock.results.store import supersede

    old = _store(tmp_path, "20260912T053225-2a91ca", duration_s=1234.0, notes="ran clean.")
    _store(tmp_path, "20260916T051616-2a91ca")

    supersede(
        "comma2k19",
        "openpilot",
        "20260912T053225-2a91ca",
        by="20260916T051616-2a91ca",
        reason="the later run carries the proxy metrics",
        root=tmp_path,
    )

    after = RunRecord(**json.loads(old.read_text()))
    assert after.valid is False
    assert after.superseded_by == "20260916T051616-2a91ca"
    assert after.notes.startswith("ran clean."), "what the run said about itself is still true"
    assert "the later run carries the proxy metrics" in after.notes
    # Identity and measurement untouched: a superseded run still happened.
    assert (after.run_id, after.duration_s, after.status) == ("20260912T053225-2a91ca", 1234.0, "complete")


def test_supersede_refuses_the_edits_that_leave_no_ranked_row(tmp_path: Path) -> None:
    """Each of these is a state a hand-edit produces silently."""
    from paddock.results.store import SupersedeError, supersede

    _store(tmp_path, "a-run")
    _store(tmp_path, "b-run")
    args = ("comma2k19", "openpilot")

    with pytest.raises(SupersedeError, match="itself"):
        supersede(*args, "a-run", by="a-run", reason="x", root=tmp_path)
    with pytest.raises(SupersedeError, match="no such run"):
        supersede(*args, "missing", by="b-run", reason="x", root=tmp_path)
    with pytest.raises(SupersedeError, match="no such run to supersede it"):
        supersede(*args, "a-run", by="missing", reason="x", root=tmp_path)
    with pytest.raises(SupersedeError, match="reason"):
        supersede(*args, "a-run", by="b-run", reason="   ", root=tmp_path)

    supersede(*args, "b-run", by="a-run", reason="a is current", root=tmp_path)
    with pytest.raises(SupersedeError, match="point at that one instead"):
        # b is retired, so pointing a at it would retire every row there is.
        supersede(*args, "a-run", by="b-run", reason="circular", root=tmp_path)


def test_supersede_output_is_what_the_store_would_have_written(tmp_path: Path) -> None:
    """It goes through `RunRecord`, so it cannot invent a field or skip `scrub`."""
    from paddock.results.store import supersede

    old = _store(tmp_path, "a-run", ingested_from="/" + "/".join(["home", "someone", "old-runs"]))
    _store(tmp_path, "b-run")
    supersede("comma2k19", "openpilot", "a-run", by="b-run", reason="newer", root=tmp_path)

    raw = json.loads(old.read_text())
    assert set(raw) == set(RunRecord(run_id="", site="", benchmark="", model="").to_json())
    assert raw["ingested_from"] == "old-runs", "the rewrite scrubs like every other write"


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
