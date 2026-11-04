"""Tests for the constant-velocity baseline in `tools/benchmarks/comma2k19/`.

The tool lives under `tools/`, which pytest does not collect (`norecursedirs`
in `pyproject.toml`, because older modules there import torch and openpilot).
This one imports cleanly in paddock's own environment, and it writes into the
tracked results store, so it is tested from here rather than not at all.

What matters about a baseline is that it is honest: that it predicts exactly
what it claims to, that skipping the video changes nothing else about how the
benchmark is measured, and that a subset run cannot end up ranked.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from tools.benchmarks.comma2k19 import constant_velocity as CV

# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def test_the_plan_covers_the_whole_recorded_future():
    """Defined at every horizon the report prints, not just the scored one.

    On the benchmark's own grid rather than any model's: `score` pairs by time,
    so this costs the comparison nothing and buys a baseline row at 10 s.
    """
    traj = CV.plan(20.0)
    assert traj.dt == H.INTERVAL_S
    assert len(traj.xy) == H.N_GT_POSES
    assert traj.xy[-1][0] == pytest.approx(20.0 * H.N_GT_POSES * H.INTERVAL_S)
    assert traj.xy[:, 1] == pytest.approx(np.zeros(H.N_GT_POSES))


def test_a_stopped_car_predicts_standing_still():
    """Which is an all-zero trajectory -- paddock's commanded emergency stop.

    Deliberate, and the one case where those two meanings coincide: the
    constant-velocity continuation of a stationary car *is* to stay put, and
    the recorded future it is scored against will say the same.
    """
    assert CV.plan(0.0).xy == pytest.approx(np.zeros((H.N_GT_POSES, 2)))


def test_the_plan_is_straight_at_every_speed():
    for speed in (1.0, 13.4, 31.0):
        xy = CV.plan(speed).xy
        steps = np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0)
        assert steps[:, 0] == pytest.approx(np.full(len(xy), speed * H.INTERVAL_S))


# --------------------------------------------------------------------------
# Skipping the video, and nothing else
# --------------------------------------------------------------------------


class _Bench(CV.NoDecodeBenchmark):
    """The subclass with the site config it never gets to use."""

    def __init__(self):  # noqa: D107 - the parent needs a spec and a config
        self.spec = types.SimpleNamespace(name="comma2k19", dataset="comma2k19", mode="open_loop")
        self.cfg = types.SimpleNamespace(site="test", benchmarks={}, datasets={})


def test_materialise_names_every_frame_and_decodes_none(tmp_path, monkeypatch):
    """A path nobody opens costs nothing to be wrong; a decode costs a day."""
    monkeypatch.setattr(L, "decode_frames", _never)
    bench = _Bench()
    monkeypatch.setattr(type(bench), "_segment_frame_dir", lambda self, seg: tmp_path)
    frames = bench._materialise(object(), {3, 1, 2})
    assert sorted(frames) == [1, 2, 3]
    assert frames[2].name == "000002.png"
    assert not list(tmp_path.iterdir())  # nothing written, either


def _never(*args, **kwargs):
    raise AssertionError("the baseline must not decode video")


def test_only_the_decoder_check_is_waived(monkeypatch):
    """Every other preflight problem still stops the run.

    Matched against the exception the parent turns into a problem rather than
    by looking for a word in it, so rewording that message cannot silently
    start waiving something else.
    """
    boom = RuntimeError("no decoder here")
    monkeypatch.setattr(L, "probe_decoder", _raise(boom))
    monkeypatch.setattr(H.Comma2k19Benchmark, "preflight", lambda self: [str(boom), "no segments found"])
    assert _Bench().preflight() == ["no segments found"]


def test_a_machine_with_a_decoder_keeps_every_problem(monkeypatch):
    monkeypatch.setattr(L, "probe_decoder", lambda: "av")
    monkeypatch.setattr(H.Comma2k19Benchmark, "preflight", lambda self: ["no segments found"])
    assert _Bench().preflight() == ["no segments found"]


def _raise(exc):
    def go(*args, **kwargs):
        raise exc

    return go


def test_the_run_config_says_the_video_was_skipped(monkeypatch):
    """Not decorative: it is hashed into the run id, so the baseline's id cannot
    collide with a run of the same benchmark that did read pixels."""
    monkeypatch.setattr(H.Comma2k19Benchmark, "run_config", lambda self: {"history_frames": 160})
    config = _Bench().run_config()
    assert config["frames_decoded"] is False
    assert config["policy"] == "constant_velocity"
    assert config["history_frames"] == 160

    from paddock.results import store

    assert store.make_run_id(config={"history_frames": 160}) != store.make_run_id(config=config)


# --------------------------------------------------------------------------
# The record it writes
# --------------------------------------------------------------------------


def _sample(sample_id="comma2k19/1/r|x/3/159", speed=20.0):
    obs = types.SimpleNamespace(speed=speed)
    return types.SimpleNamespace(
        sample_id=sample_id,
        native_id={"keyframe_index": 159},
        split="rav4",
        observations=[obs, obs],
        ground_truth=np.column_stack([10.0 * np.arange(1, 21), np.zeros(20)]),
    )


class _StubBench:
    """Enough harness to drive `sweep` and `build_run` without a dataset."""

    spec = types.SimpleNamespace(name="comma2k19", dataset="comma2k19", mode="open_loop")

    def __init__(self, samples, reasons=()):
        self._samples = samples
        self._reasons = list(reasons)

    def samples(self, limit=None, split=None):
        return iter(self._samples[:limit] if limit else self._samples)

    def score(self, sample, prediction):
        return {"ade": 1.5}

    def invalid_reasons(self):
        return list(self._reasons)

    def remarks(self):
        return []

    def run_config(self):
        return {"frames_decoded": False}


def test_a_record_has_exactly_the_keys_the_runner_writes():
    """`paddock render` and `comma2k19.report` read these files without knowing
    which process produced them, so an extra or missing key is a broken artefact
    rather than a cosmetic difference. The list is `runner._run_open_loop`'s."""
    records = CV.sweep(_StubBench([_sample()]), None, None, lambda line: None)
    assert set(records[0]) == {"sample_id", "native_id", "split", "metrics", "prediction", "ground_truth"}
    assert records[0]["metrics"] == {"ade": 1.5}
    # JSON-serialisable, because these go straight to samples.jsonl.
    assert isinstance(records[0]["prediction"], list)
    assert isinstance(records[0]["prediction"][0], list)


def test_a_sample_with_no_recorded_future_stores_none_rather_than_failing():
    sample = _sample()
    sample.ground_truth = None
    assert CV.sweep(_StubBench([sample]), None, None, lambda line: None)[0]["ground_truth"] is None


def _built(samples, limit=None, reasons=()):
    from datetime import UTC, datetime

    bench = _StubBench(samples, reasons)
    records = CV.sweep(bench, limit, None, lambda line: None)
    cfg = types.SimpleNamespace(site="test")
    run = CV.build_run(bench, cfg, "id", datetime.now(UTC), 0.0, records, limit, None)
    return run, records


def test_a_subset_run_is_not_rankable_and_says_why():
    """The rule is `runner._validity`'s, imported rather than restated -- a
    second copy of it is how a `--limit 20` baseline reaches a leaderboard."""
    run, _ = _built([_sample(), _sample("comma2k19/1/r|x/3/300")], limit=1)
    assert run.valid is False
    assert "--limit 1" in run.notes


def test_a_full_run_is_rankable_and_still_explains_what_it_is():
    run, records = _built([_sample()])
    assert run.valid is True
    assert "Reference baseline" in run.notes
    assert run.model == CV.MODEL
    assert run.dataset["n_samples"] == len(records) == 1
    assert run.dataset["sequences_sha256"]


def test_a_reason_from_the_harness_invalidates_the_baseline_too():
    run, _ = _built([_sample()], reasons=["the rate was overridden"])
    assert run.valid is False
    assert "the rate was overridden" in run.notes
