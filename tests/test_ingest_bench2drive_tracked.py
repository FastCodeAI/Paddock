"""A Bench2Drive pool takes routes it has no results.json for from a tracked record's rows.

The files are real throughout (fixtures/bench2drive_tracked/README.md). There are four 216-route
records made at another site, whose raw results never reached this repo. There are also five
routes openpilot drove, whose raw results are in the repo, with the upstream_scores.json the
repo's tool wrote for them.

What must hold:
- A route taken from a record measures what its results.json would, except for duration_s,
  which the rows do not hold and the store counts as missing.
- A record the rows do not fully describe is refused.
- A raw route under `runs` takes precedence, route by route, and every route names its origin.
- The record's config is checked against the runs' configs the same way the runs' configs are
  checked against each other.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from paddock.config import ConfigError, SiteConfig
from paddock.results import legacy, store
from paddock.results.ingest import bench2drive

TRACKED = Path(__file__).parent / "fixtures" / "bench2drive_tracked"
FIVE = Path(__file__).parent / "fixtures" / "bench2drive_upstream"
A100 = sorted((TRACKED / "bench2drive").glob("*/*"))
OPENPILOT = TRACKED / "bench2drive" / "openpilot" / "full220-836dda734659a86c-2803bd"
#: The run that drove each of the five routes in fixtures/bench2drive_upstream.
FIVE_RUNS = {
    "2050": "20261031T003248-fd22d5",
    "25857": "20261031T003248-fd22d5",
    "4183": "20261031T003328-3fb93b",
    "3575": "20261031T003408-52e5e7",
    "24224": "20261031T003448-bed441",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _but_duration(metrics: dict) -> dict:
    """metrics.json without duration_s, the one block the rows cannot give, and without its run id."""
    out = copy.deepcopy(metrics)
    out.pop("run_id")
    out["metrics"].pop("duration_s", None)
    for table in ("by_split", "by_group"):
        for block in out[table].values():
            block.pop("duration_s", None)
    return out


def _five_raw(tmp_path: Path) -> Path:
    """The five routes as their sweep left them."""
    root = tmp_path / "runs"
    shutil.copytree(FIVE, root, ignore=shutil.ignore_patterns("README.md"))
    return root


def _five_record(tmp_path: Path) -> tuple[Path, Path]:
    """The five routes pooled from their results.json, the tool's upstream_scores.json beside the record."""
    root = _five_raw(tmp_path)
    ((run, metrics, samples),) = bench2drive.ingest(root, model="openpilot", site="s")
    record = store.write_run(run, metrics, samples, root=tmp_path / "results")
    upstream = TRACKED / "five_routes_upstream_scores.json"
    assert _read(upstream)["record"] == run.run_id, (
        "the fixture was made from this pool; regenerate it (fixtures/bench2drive_tracked/README.md)"
    )
    shutil.copy(upstream, record / bench2drive.UPSTREAM_SCORES)
    return root, record


def _copy(record: Path, tmp_path: Path) -> Path:
    """A record to change, in the layout results/ keeps it."""
    out = tmp_path / "bench2drive" / record.parent.name / record.name
    shutil.copytree(record, out)
    return out


def _edit(path: Path, change: Callable[[dict], object]) -> None:
    data = _read(path)
    change(data)
    path.write_text(json.dumps(data))


# --- a route from a record measures what its results.json does ---------------


@pytest.mark.parametrize("record", A100, ids=lambda p: p.parent.name)
def test_a_record_pooled_from_its_own_rows_gives_its_own_metrics(record: Path) -> None:
    """Each of the four a100 records, pooled from its upstream_scores.json alone, gives its stored
    metrics.json in every field and key order except duration_s: 216 routes, 44 scenarios."""
    run, stored = _read(record / "run.json"), _read(record / "metrics.json")
    ((pooled, metrics, samples),) = bench2drive.ingest(
        [], model=run["model"], site="s", routes=run["config"]["routes_subset"], tracked=[record]
    )
    mine = metrics.to_json()
    assert _but_duration(mine) == _but_duration(stored)
    assert list(mine["metrics"]) == list(stored["metrics"]), "the same blocks, in the same order"
    assert mine["metrics"]["duration_s"] == {"n": 0, "n_missing": 216}
    assert stored["metrics"]["duration_s"]["n"] == 216, "the raw path had it for every route"
    assert pooled.dataset == run["dataset"]
    assert pooled.status == "complete" and pooled.valid, pooled.notes
    assert pooled.started_utc == run["started_utc"]
    name = f"bench2drive/{run['model']}/{run['run_id']}"
    sha = hashlib.sha256((record / bench2drive.UPSTREAM_SCORES).read_bytes()).hexdigest()
    assert pooled.config["source_records"] == {
        name: {"site": "a100-gpu-vm-002", "upstream_scores_sha256": sha}
    }
    assert pooled.config["source_runs"] == []
    assert {(s["native_id"]["record"], s["native_id"]["sha256"]) for s in samples} == {(name, sha)}
    assert f"216 route(s) from tracked record {name}" in pooled.notes
    assert "pooled at site a100-gpu-vm-002" in pooled.notes
    assert "the 216 route(s) from tracked records carry none" in pooled.notes


def test_routes_from_a_record_measure_what_their_results_json_does(tmp_path: Path) -> None:
    """The same five drives, pooled once from their results.json and once from the rows the repo's
    tool wrote from them, agree route by route on every metric but duration_s."""
    root, record = _five_record(tmp_path)
    ((_, raw_metrics, raw_samples),) = bench2drive.ingest(root, model="openpilot", site="s")
    ((run, metrics, samples),) = bench2drive.ingest([], model="openpilot", site="s", tracked=[record])

    assert _but_duration(metrics.to_json()) == _but_duration(raw_metrics.to_json())
    assert [s["sample_id"] for s in samples] == [s["sample_id"] for s in raw_samples]
    for row, raw in zip(samples, raw_samples, strict=True):
        assert {k: v for k, v in row["metrics"].items() if k != "duration_s"} == {
            k: v for k, v in raw["metrics"].items() if k != "duration_s"
        }
        assert row["metrics"]["duration_s"] is None and raw["metrics"]["duration_s"] > 0
    assert metrics.metrics["duration_s"] == {"n": 0, "n_missing": 5}
    assert {s["metrics"]["ended_stuck"] for s in samples} == {0.0, 1.0}, "the early stop carried, both ways"


# --- raw routes first, and each route says where it came from ----------------


def test_a_raw_route_takes_precedence_and_every_route_names_its_origin(tmp_path: Path) -> None:
    """Five routes driven here and 211 from the a100 openpilot record: the five are the raw drives,
    with their run and results.json digest; the others name the record and its file's digest."""
    root = _five_raw(tmp_path)
    routes = _read(OPENPILOT / "run.json")["config"]["routes_subset"]
    ((run, metrics, samples),) = bench2drive.ingest(
        root, model="openpilot", site="s", routes=routes, tracked=[OPENPILOT]
    )
    ((_, _, raw_samples),) = bench2drive.ingest(root, model="openpilot", site="s")
    raw = {s["sample_id"]: s for s in raw_samples}

    assert len(samples) == 216 and run.dataset["sequences_sha256"] == "836dda734659a86c"
    here = [s for s in samples if s["sample_id"] in raw]
    assert {s["native_id"]["route_id"]: s["native_id"]["run_id"] for s in here} == FIVE_RUNS
    for sample in here:
        assert sample["native_id"] == raw[sample["sample_id"]]["native_id"], "the raw route's own origin"
        assert sample["metrics"] == raw[sample["sample_id"]]["metrics"], "the raw drive, not the record's row"
    sha = hashlib.sha256((OPENPILOT / bench2drive.UPSTREAM_SCORES).read_bytes()).hexdigest()
    elsewhere = [s for s in samples if s["sample_id"] not in raw]
    assert {(s["native_id"]["record"], s["native_id"]["sha256"]) for s in elsewhere} == {
        ("bench2drive/openpilot/full220-836dda734659a86c-2803bd", sha)
    }
    assert metrics.metrics["duration_s"]["n"] == 5 and metrics.metrics["duration_s"]["n_missing"] == 211
    assert run.config["source_runs"] == sorted(set(FIVE_RUNS.values()))
    assert "5 route(s) from 4 run(s) under `runs`" in run.notes
    assert (
        "211 route(s) from tracked record bench2drive/openpilot/full220-836dda734659a86c-2803bd" in run.notes
    )
    assert "the 211 route(s) from tracked records carry none" in run.notes


def test_a_route_neither_record_nor_run_scored_is_lost_with_the_records_reason(tmp_path: Path) -> None:
    """Alpamayo's record never scored 3800 (CARLA crashed); asked for it, the pool says so and why."""
    record = TRACKED / "bench2drive" / "alpamayo15" / "full220-836dda734659a86c-f81033"
    routes = [*_read(record / "run.json")["config"]["routes_subset"], "3800"]
    ((run, _, _),) = bench2drive.ingest([], model="alpamayo15", site="s", routes=routes, tracked=[record])
    assert run.status == "partial" and not run.valid
    assert run.dataset["n_failed"] == 1
    assert (
        "3800 (bench2drive/alpamayo15/full220-836dda734659a86c-f81033: 20261109T113106-b2cdb0: CARLA crashed"
        in (run.notes)
    )


# --- the record's config is one more source run's -----------------------------


def _five_as_the_record_ran(tmp_path: Path, change: Callable[[dict], object] = lambda c: None) -> Path:
    """The five routes, each run given a run.json with the a100 openpilot record's config, then `change`d."""
    root = _five_raw(tmp_path)
    recorded = _read(OPENPILOT / "run.json")["config"]
    for run_id in sorted(set(FIVE_RUNS.values())):
        config = {
            k: copy.deepcopy(recorded[k]) for k in ("routes_file", "waypoint_dt", "model_env", "declared_env")
        }
        config["early_stop"] = copy.deepcopy(recorded["early_stop"])
        change(config)
        record = {"run_id": run_id, "site": "this-site", "model": "openpilot", "config": config}
        (root / run_id / "run.json").write_text(json.dumps(record))
    return root


def test_runs_that_agree_with_the_record_pool_into_a_ranked_run(tmp_path: Path) -> None:
    root = _five_as_the_record_ran(tmp_path)
    routes = _read(OPENPILOT / "run.json")["config"]["routes_subset"]
    ((run, _, _),) = bench2drive.ingest(root, model="openpilot", site="s", routes=routes, tracked=[OPENPILOT])
    assert run.valid and run.status == "complete", run.notes
    assert run.config["source_sites"] == ["this-site"], "the record's runs recorded none"
    assert run.config["source_config_read"] == "4/4 source runs"
    assert run.config["declared_env"] == _read(OPENPILOT / "run.json")["config"]["declared_env"]


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda c: c.pop("early_stop"), "4 of 5 source runs and tracked records ran without it"),
        (lambda c: c["early_stop"].update(stuck_s=60.0), "early stop"),
        (lambda c: c["declared_env"].update(PADDOCK_OPENPILOT_MODEL="big"), "different model.yaml envs"),
        (lambda c: c.pop("declared_env"), "4 of 5 source runs and tracked records recorded no declared_env"),
        (lambda c: c.update(camera_rig="bench2drive"), "camera rig"),
        (lambda c: c.update(infer_hz=10.0), "different rates"),
    ],
    ids=[
        "no-early-stop",
        "other-early-stop",
        "other-declared-env",
        "no-declared-env",
        "other-rig",
        "other-rate",
    ],
)
def test_runs_that_disagree_with_the_record_do_not_rank(tmp_path: Path, change, said: str) -> None:
    """What keeps two source runs from pooling into a ranked run keeps a run and a record from it."""
    root = _five_as_the_record_ran(tmp_path, change)
    routes = _read(OPENPILOT / "run.json")["config"]["routes_subset"]
    ((run, _, _),) = bench2drive.ingest(root, model="openpilot", site="s", routes=routes, tracked=[OPENPILOT])
    assert not run.valid
    assert said in run.notes


def test_a_record_that_does_not_rank_keeps_the_pool_from_ranking(tmp_path: Path) -> None:
    record = _copy(OPENPILOT, tmp_path)
    _edit(record / "run.json", lambda r: r.update(valid=False))
    ((run, _, _),) = bench2drive.ingest(
        [],
        model="openpilot",
        site="s",
        routes=_read(record / "run.json")["config"]["routes_subset"],
        tracked=[record],
    )
    assert not run.valid
    assert "full220-836dda734659a86c-2803bd do not rank (valid: false" in run.notes


# --- a record the rows do not fully describe is refused, by name --------------


def _pin(key: str, **values: float) -> Callable[[dict], object]:
    return lambda m: m["metrics"][key].update(values)


REFUSALS = {
    "no-upstream-scores": ("upstream_scores.json", None, "no upstream_scores.json beside it"),
    "two-repetitions": ("metrics.json", _pin("n_runs", max=2), "one scored repetition per route"),
    "a-crashed-repetition": ("metrics.json", _pin("n_crashed", max=1.0), "one scored repetition per route"),
    "early-stop-not-everywhere": ("metrics.json", _pin("ended_stuck", n=215, n_missing=1), "ended_stuck"),
    "early-stop-on-none-recorded": (
        "metrics.json",
        lambda m: [m["metrics"].pop(f"ended_{r}") for r in ("stuck", "over_time", "collision", "manual")],
        "cannot say which routes had an early_stop.json",
    ),
    "a-row-changed": (
        "upstream_scores.json",
        lambda u: u["routes"][0].update(driving_score=u["routes"][0]["driving_score"] + 1),
        "do not give its metrics.json (driving_score",
    ),
    "a-row-that-crashed": (
        "upstream_scores.json",
        lambda u: u["routes"][0].update(status="Failed - Agent crashed"),
        "not a measurement",
    ),
    "a-row-missing": ("upstream_scores.json", lambda u: u["routes"].pop(), "are not the record's own"),
    "another-records-rows": (
        "upstream_scores.json",
        lambda u: u.update(record="full220-0000000000000000-000000"),
        "is for openpilot/full220-0000000000000000-000000",
    ),
}


@pytest.mark.parametrize("case", REFUSALS, ids=list(REFUSALS))
def test_a_record_the_rows_do_not_fully_describe_is_refused(tmp_path: Path, case: str) -> None:
    """Refused by name, never pooled with a guess: the rows hold one repetition, an early stop's
    record only where metrics.json says every route had one, and nothing a stored figure disowns."""
    file, change, said = REFUSALS[case]
    record = _copy(OPENPILOT, tmp_path)
    if change is None:
        (record / file).unlink()
    else:
        _edit(record / file, change)
    with pytest.raises(bench2drive.TrackedRefused, match=said.replace("(", r"\(")):
        bench2drive.ingest([], model="openpilot", site="s", tracked=[record])


def test_a_record_of_another_model_is_refused() -> None:
    with pytest.raises(bench2drive.TrackedRefused, match="not a Bench2Drive record of 'alpamayo15'"):
        bench2drive.ingest([], model="alpamayo15", site="s", tracked=[OPENPILOT])


def test_a_route_two_records_hold_and_no_run_scored_is_refused(tmp_path: Path) -> None:
    """Which of two drives a route's number is is the operator's call, named in `tracked`, not ours."""
    _, five = _five_record(tmp_path)
    with pytest.raises(bench2drive.TrackedRefused, match="route 2050 is in both"):
        bench2drive.ingest([], model="openpilot", site="s", tracked=[OPENPILOT, five])


# --- the site config names records as results/ keeps them ---------------------


def _site(tmp_path: Path, toml: str) -> SiteConfig:
    source = tmp_path / "paddock.toml"
    source.write_text(toml)
    return SiteConfig(
        site="s",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={},
        models={},
        benchmarks={},
        gpu=None,  # type: ignore[arg-type]
        source=source,
    )


def test_the_site_config_names_a_record_as_results_keeps_it(tmp_path: Path, monkeypatch) -> None:
    """`tracked = ["bench2drive/<model>/<run_id>"]` is read from the results store; `runs` may be left
    out when every route is a record's."""
    monkeypatch.setattr(store, "RESULTS_ROOT", TRACKED)
    cfg = _site(
        tmp_path,
        '[[ingest.bench2drive]]\nmodel = "openpilot"\nlabel = "again"\n'
        'tracked = ["bench2drive/openpilot/full220-836dda734659a86c-2803bd"]\n',
    )
    assert legacy.ingest_all(cfg, dry_run=True) == (1, [])


@pytest.mark.parametrize(
    ("tracked", "said"),
    [
        ('"bench2drive/openpilot/full220-836dda734659a86c-2803bd"', "tracked is a list"),
        ('["openpilot/full220-836dda734659a86c-2803bd"]', 'names a record "bench2drive/<model>/<run_id>"'),
        ('["bench2drive/openpilot/no-such-record"]', "no readable run.json and metrics.json"),
    ],
)
def test_a_tracked_entry_that_names_no_record_fails_by_name(
    tmp_path: Path, monkeypatch, tracked, said
) -> None:
    monkeypatch.setattr(store, "RESULTS_ROOT", TRACKED)
    cfg = _site(tmp_path, f'[[ingest.bench2drive]]\nmodel = "openpilot"\ntracked = {tracked}\n')
    with pytest.raises(ConfigError, match=said):
        legacy.ingest_all(cfg, dry_run=True)
