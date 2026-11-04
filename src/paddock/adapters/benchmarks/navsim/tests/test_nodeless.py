"""A replay's record: how its plans were made comes from the run it replays."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.navsim import nodeless
from paddock.adapters.benchmarks.navsim.harness import PLAN_SETTINGS, NavsimBenchmark
from paddock.adapters.benchmarks.navsim.tests.test_harness import _harness, _navsim_site
from paddock.benchmark import Evaluation
from paddock.registry import ModelSpec, Runtime
from paddock.results import store

#: A stored NAVSIM v2 run at settings no NAVSIM v1 harness default would give.
SOURCE_SETTINGS = {
    "input_hz": 10.0,
    "history_source": "nuplan",
    "history_s": 6.5,
    "keyframe_cameras": True,
}


def _replay(tmp_path: Path, monkeypatch, source_config: dict) -> dict:
    """Replay a stored run into NAVSIM v1 with the scorer stubbed; the new run.json."""
    cfg = _navsim_site(tmp_path, monkeypatch, bench="navsim_v1")
    root = tmp_path / "results"
    monkeypatch.setattr(store, "RESULTS_ROOT", root)
    # `write_run` bound its default root at import: without this the test would
    # write into the repo's own results/.
    monkeypatch.setattr(store, "write_run", functools.partial(store.write_run, root=root))

    source = root / "navsim" / "toy" / "r1"
    source.mkdir(parents=True)
    (source / "run.json").write_text(
        json.dumps({"benchmark": "navsim", "model": "toy", "valid": True, "config": source_config})
    )
    plans = cfg.runs / "navsim" / "toy" / "r1" / "chunk0"
    plans.mkdir(parents=True)
    pose = [[1.0, 0.0, 0.0]] * 8
    (plans / "trajectories.jsonl").write_text(json.dumps({"token": "tok", "navsim_poses": pose}) + "\n")

    sample = {"sample_id": "navtest/tok", "native_id": "tok", "split": "navtest", "metrics": {"pdms": 0.5}}
    monkeypatch.setattr(NavsimBenchmark, "preflight", lambda self: [])
    monkeypatch.setattr(NavsimBenchmark, "model_problems", lambda self, model, limit=None: [])
    monkeypatch.setattr(NavsimBenchmark, "evaluate", lambda self, *a, **k: Evaluation(samples=[sample]))

    assert nodeless.run(cfg, "navsim_v1", replay="navsim/toy/r1", echo=lambda _: None) == 0
    (written,) = (root / "navsim_v1" / "toy").glob("*/run.json")
    return json.loads(written.read_text())


def test_a_replay_records_the_settings_of_the_run_it_replays(tmp_path: Path, monkeypatch) -> None:
    config = _replay(tmp_path, monkeypatch, dict(SOURCE_SETTINGS))["config"]

    assert {key: config.get(key) for key in PLAN_SETTINGS} == SOURCE_SETTINGS
    assert config["replayed_from"] == "navsim/toy/r1"


def test_a_source_that_recorded_no_settings_leaves_them_out(tmp_path: Path, monkeypatch) -> None:
    """Older runs predate these fields. Nothing here knows them, so nothing says them."""
    config = _replay(tmp_path, monkeypatch, {})["config"]

    assert not set(PLAN_SETTINGS) & set(config)


@pytest.mark.parametrize("hz", [2.0, 10.0])
def test_plan_settings_covers_everything_the_harness_records(tmp_path: Path, monkeypatch, hz: float) -> None:
    """A field added to `model_run_config` and not here would be the benchmark's again."""
    harness = _harness(
        _navsim_site(tmp_path, monkeypatch, bench="navsim_v1", keyframe_cameras="true"), "navsim_v1"
    )
    model = ModelSpec(name="toy", repo="", runtime=Runtime(kind="shell"), min_input_hz=hz)

    assert set(harness.model_run_config(model)) <= set(PLAN_SETTINGS)
