"""The closed-loop harness, checked without a simulator."""

from __future__ import annotations

import pytest

from paddock.adapters.benchmarks.neuroncap.harness import (
    DEFAULT_SCENARIOS,
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
    if not harness._setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    problems = harness.scene_problems("9999")
    assert problems and "no renderer checkpoint" in problems[0]


def test_available_scenes_intersects_scenarios_with_checkpoints() -> None:
    """A scenario yaml without a renderer checkpoint cannot be rendered."""
    harness = _harness()
    if not harness._setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    scenes = harness.available_scenes()
    assert scenes, "expected at least one runnable scene"
    assert all(harness.scene_problems(s) == [] for s in scenes)
