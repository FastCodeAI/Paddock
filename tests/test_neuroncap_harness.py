"""The closed-loop harness, checked without a simulator."""

from __future__ import annotations

import json

import pytest

from paddock.adapters.benchmarks.neuroncap.harness import (
    DEFAULT_SCENARIOS,
    OFFICIAL_SET,
    SCENARIOS_DIR,
    WAYPOINT_DT,
)
from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry


def _harness():
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    spec = Registry.load().benchmarks.get("neuroncap")
    if spec is None:
        pytest.skip("neuroncap benchmark not registered")
    from paddock.adapters.benchmarks.neuroncap.harness import BENCHMARK

    return BENCHMARK(spec, cfg)


def test_scenario_files_are_tracked_in_the_repo() -> None:
    """The 10 Hz scenarios are part of paddock, not a scratch directory."""
    assert SCENARIOS_DIR.is_dir()
    for scenario in DEFAULT_SCENARIOS:
        assert (SCENARIOS_DIR / scenario).is_dir(), f"missing scenario set: {scenario}"


def test_scenarios_step_at_ten_hertz() -> None:
    """0.1 s is forced, not chosen: openpilot reads a 0.5 s step as a parked car.

    If a scenario file drifts back to 0.5 s, every model is silently compared at
    a rate one of them cannot use.
    """
    for path in SCENARIOS_DIR.rglob("*.yaml"):
        text = path.read_text()
        assert "time-step: 0.1" in text, f"{path.name} does not step at 0.1 s"


def test_waypoint_grid_is_independent_of_the_sim_step() -> None:
    """Scoring and collision lookahead both assume 0.5 s waypoints."""
    assert WAYPOINT_DT == 0.5


def test_episode_ids_are_stable_and_paired() -> None:
    """Every model must see identical episodes, in a deterministic order."""
    harness = _harness()
    first = [e.episode_id for e in harness.episodes(["0103"], runs=1)]
    second = [e.episode_id for e in harness.episodes(["0103"], runs=1)]
    assert first == second
    assert first == ["stationary-0103", "frontal-0103", "side-0103"]


def test_a_scene_without_a_checkpoint_is_reported_before_running() -> None:
    """This is what preflight is for: catching it in seconds, not an hour in."""
    harness = _harness()
    if not harness.setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    problems = harness.scene_problems("9999")
    assert problems and "no renderer checkpoint" in problems[0]


def test_available_scenes_intersects_scenarios_with_checkpoints() -> None:
    """A scenario yaml without a renderer checkpoint cannot be rendered."""
    harness = _harness()
    if not harness.setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    scenes = harness.available_scenes()
    assert scenes, "expected at least one runnable scene"
    assert all(harness.scene_problems(s) == [] for s in scenes)


def test_the_official_evaluation_set_is_complete() -> None:
    """20 pairs over 14 scenes, exactly as the vendor's job arrays define them.

    Scored on fewer, every headline number quietly means something else. The
    files are cheap to lose (one is a one-line edit of an upstream file) and
    `episodes()` skips a scene that is legitimately not in a category, so a gap
    here would not raise -- it would just shrink the sweep.
    """
    assert sum(len(v) for v in OFFICIAL_SET.values()) == 20
    for scenario, scenes in OFFICIAL_SET.items():
        for scene in scenes:
            path = SCENARIOS_DIR / scenario / f"{scene}.yaml"
            assert path.exists(), f"missing official scenario file: {scenario}/{scene}.yaml"


def test_no_scenario_files_beyond_the_official_set() -> None:
    """0061 is in the upstream stationary array marked 'do not use for eval'.

    An extra file would be scored by a default sweep and inflate `n` against
    numbers nobody else can reproduce.
    """
    tracked = {(p.parent.name, p.stem) for p in SCENARIOS_DIR.rglob("*.yaml")}
    official = {(s, scene) for s, scenes in OFFICIAL_SET.items() for scene in scenes}
    assert tracked == official, f"unexpected: {tracked - official}, missing: {official - tracked}"


def test_a_named_scenario_scene_pair_that_does_not_exist_is_an_error() -> None:
    """Explicitly asked for, explicitly refused -- not silently dropped."""
    harness = _harness()
    with pytest.raises(FileNotFoundError, match="no scenario file"):
        list(harness.episodes(["0099"], runs=1, scenarios=["frontal"]))


def test_a_scene_outside_a_category_is_skipped_when_scenarios_default() -> None:
    """0099 is a stationary-only scene; that is the benchmark's shape, not a gap."""
    harness = _harness()
    assert [e.episode_id for e in harness.episodes(["0099"], runs=1)] == ["stationary-0099"]


def test_runs_are_ordered_numerically_not_lexically(tmp_path) -> None:
    """`run_10` sorts between `run_1` and `run_2` as a string.

    At the 50 runs the official protocol asks for, a lexical sort misattributes
    every run index above nine.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    for i in (0, 1, 2, 10, 11):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"ncap_score": float(i)}))

    per_run = NeuroNcapBenchmark._per_run(tmp_path)
    assert [i for i, _ in per_run] == [0, 1, 2, 10, 11]
    assert [m["ncap_score"] for _, m in per_run] == [0.0, 1.0, 2.0, 10.0, 11.0]


def test_every_run_reaches_the_results_store_separately(tmp_path) -> None:
    """The spread is the point: a mean of 10 runs with no n and no sd is unreadable."""
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    for i, score in enumerate([0.0, 2.084, 5.0]):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"ncap_score": score, "any_collide@0.0s": score < 5.0}))

    harness = _harness()
    episode = next(iter(harness.episodes(["0103"], runs=3, scenarios=["frontal"])))
    samples = harness.episode_samples(episode, {}, tmp_path)

    assert len(samples) == 3
    assert [s["metrics"]["ncap_score"] for s in samples] == [0.0, 2.084, 5.0]
    assert [s["native_id"]["run"] for s in samples] == [0, 1, 2]
    assert len({s["sample_id"] for s in samples}) == 3, "sample ids must not collide"
    # Same shape the legacy ingest writes, so August runs and live runs compare.
    assert samples[0]["sample_id"] == "neuroncap/0103/frontal/run_0"
    assert all(s["split"] == "frontal" for s in samples)

    # The episode headline stays the mean, for the console line.
    assert NeuroNcapBenchmark._collect(tmp_path)["ncap_score"] == pytest.approx(2.361333, abs=1e-5)


def test_expected_sample_count_matches_what_a_sweep_records() -> None:
    """`n_samples / n_expected` is the schema's completion ratio, so both count samples.

    neuroNCAP is the one benchmark where an episode is not one sample, so it is
    the one that breaks the ratio if `samples_per_episode` and `episode_samples`
    ever disagree about the factor.
    """
    harness = _harness()
    runs = 10
    episodes = list(harness.episodes(harness.available_scenes(), runs=runs))
    assert harness.samples_per_episode(runs) == runs
    assert len(episodes) * harness.samples_per_episode(runs) == 200


def test_the_default_closed_loop_benchmark_is_one_sample_per_episode() -> None:
    """Bench2Drive and anything added later keep the old meaning for free."""
    from paddock.benchmark import ClosedLoopBenchmark

    assert ClosedLoopBenchmark.samples_per_episode(object(), runs=10) == 1
