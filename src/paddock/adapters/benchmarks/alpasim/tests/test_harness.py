"""Golden and launch-path tests for the AlpaSim adapter.

Pins the pure functions the harness leans on -- the `results-summary.json` ->
metrics parse and scene selection -- against a real recorded run, so a refactor
that changes a score fails loudly instead of quietly rewriting one. Then drives
`run_episode` against a stand-in wizard, so the failure paths (timeout, exit
code, missing summary, unscored rollout) are tested rather than argued.

No Docker, no GPU, no model, no AlpaSim install.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.alpasim import launcher
from paddock.adapters.benchmarks.alpasim.harness import (
    BENCHMARK,
    DEFAULT_SCENE_SET,
    NotScored,
    load_scene_set,
    parse_summary,
)
from paddock.benchmark import Episode
from paddock.config import SiteConfig
from paddock.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"
SUMMARY = FIXTURES / "results_summary_two_scenes.json"
CLEAN = "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"
OFFROAD = "clipgt-02e075b9-fd24-426b-971b-7cfcb2074cc9"


def _site(tmp_path: Path, monkeypatch, **settings: str) -> SiteConfig:
    """A throwaway site config, written and loaded the way a real one is."""
    for sub in ("repos", "runs", "weights"):
        (tmp_path / sub).mkdir(exist_ok=True)
    lines = [
        'site = "test-site"',
        "[roots]",
        f'repos = "{tmp_path / "repos"}"',
        f'runs = "{tmp_path / "runs"}"',
        f'weights = "{tmp_path / "weights"}"',
    ]
    if settings:
        lines.append("[benchmarks.alpasim]")
        lines += [f'{key} = "{value}"' for key, value in settings.items()]
    config = tmp_path / "site.toml"
    config.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("PADDOCK_CONFIG", str(config))
    return SiteConfig.load()


def _benchmark(tmp_path: Path, monkeypatch, **settings):
    return BENCHMARK(Registry.load().benchmarks["alpasim"], _site(tmp_path, monkeypatch, **settings))


# ---------------------------------------------------------------------------
# parse_summary
# ---------------------------------------------------------------------------


def test_parse_summary_reproduces_the_recorded_score() -> None:
    metrics = parse_summary(SUMMARY, CLEAN)
    assert metrics["scene_score"] == 1.0
    assert metrics["progress"] == pytest.approx(1.0)


def test_a_zeroed_scene_keeps_the_progress_that_earned_nothing() -> None:
    """The hard-failure rule zeroes the scene, not the progress underneath it.

    Losing that distinction would make a policy that drove 99% of a scene and
    then left the road indistinguishable from one that never moved.
    """
    metrics = parse_summary(SUMMARY, OFFROAD)
    assert metrics["scene_score"] == 0.0
    assert metrics["progress"] == pytest.approx(0.9878, abs=1e-4)
    assert metrics["offroad"] == 1.0


def test_the_score_is_read_per_rollout_not_from_metrics_results() -> None:
    """`metrics_results` does not carry scene_score; the fixture proves it.

    The fixture is trimmed to exclude that block deliberately: a reader that
    depended on it would fail here rather than silently drop the primary metric.
    """
    assert "metrics_results" not in json.loads(SUMMARY.read_text())
    assert parse_summary(SUMMARY, CLEAN)["scene_score"] == 1.0


def test_an_unknown_scene_is_not_scored(tmp_path: Path) -> None:
    with pytest.raises(NotScored, match="no rollout"):
        parse_summary(SUMMARY, "clipgt-does-not-exist")


def test_a_rollout_without_a_score_is_not_scored(tmp_path: Path) -> None:
    body = json.loads(SUMMARY.read_text())
    body["rollouts"] = [dict(body["rollouts"][0], score=None, status="failed")]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(body))
    with pytest.raises(NotScored, match="no score"):
        parse_summary(path, CLEAN)


# ---------------------------------------------------------------------------
# scene selection
# ---------------------------------------------------------------------------


def test_the_pinned_scene_set_is_ten_scenes() -> None:
    scenes = load_scene_set(DEFAULT_SCENE_SET)
    assert len(scenes) == 10
    assert CLEAN in scenes and OFFROAD in scenes


def test_scene_order_is_the_pinned_order_not_the_filesystems(tmp_path: Path, monkeypatch) -> None:
    """`--limit 5` must select the same five at both sites."""
    bench = _benchmark(tmp_path, monkeypatch)
    once = [e.scene for e in bench.episodes()]
    twice = [e.scene for e in bench.episodes()]
    assert once == twice == load_scene_set(DEFAULT_SCENE_SET)


def test_a_scene_filter_selects_only_what_was_asked_for(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch)
    assert [e.scene for e in bench.episodes(scenes=[CLEAN])] == [CLEAN]


def test_an_unknown_scene_is_named_before_a_model_loads(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch)
    problems = bench.scene_problems(["clipgt-nope"])
    assert problems and "clipgt-nope" in problems[0]


def test_an_unknown_scene_set_says_what_is_available() -> None:
    with pytest.raises(FileNotFoundError, match=DEFAULT_SCENE_SET):
        load_scene_set("no_such_suite")


# ---------------------------------------------------------------------------
# the wizard invocation
# ---------------------------------------------------------------------------


def test_the_wizard_is_invoked_as_the_package_not_a_main_module(tmp_path: Path) -> None:
    """`-m alpasim_wizard`, never `-m alpasim_wizard.main`.

    The package's entry point is `__main__.py`, exposed as the console script
    `alpasim_wizard = "alpasim_wizard.__main__:main"`. There is no `main`
    module; naming one fails at the first scene with an import error that says
    nothing about the wizard. Pinned because it is invisible until a live run.
    """
    command = launcher.wizard_command(
        repo=tmp_path,
        python="python",
        scene_id="clipgt-x",
        out_dir=tmp_path / "out",
        bridge_port=9200,
        searchpath=None,
    )
    assert command[1:3] == ["-m", "alpasim_wizard"]
    assert "alpasim_wizard.main" not in " ".join(command)


def test_the_wizard_is_told_to_seed_and_force_determinism(tmp_path: Path) -> None:
    """Alpamayo samples at temperature 0.6 and seeds only under this flag.

    Without both, a run is one unseeded draw that neither paddock nor AlpaSim
    itself can reproduce -- which is what the recorded pre-adapter run was.
    """
    joined = " ".join(
        launcher.wizard_command(
            repo=tmp_path,
            python="python",
            scene_id="clipgt-x",
            out_dir=tmp_path / "out",
            bridge_port=9200,
            searchpath=None,
        )
    )
    assert "+driver.model.force_determinism=true" in joined
    assert f"+scenes.random_seed={launcher.DEFAULT_RANDOM_SEED}" in joined


def test_the_bridge_is_the_driver_and_the_wizard_starts_no_driver_of_its_own(tmp_path: Path) -> None:
    joined = " ".join(
        launcher.wizard_command(
            repo=tmp_path,
            python="python",
            scene_id="s",
            out_dir=tmp_path,
            bridge_port=9271,
            searchpath=None,
        )
    )
    assert "wizard.external_services.driver=[localhost:9271]" in joined
    assert '"driver"' not in joined.split("run_sim_services=")[1].split()[0]


def test_episodes_accepts_exactly_what_the_runner_passes(tmp_path: Path, monkeypatch) -> None:
    """The runner calls `episodes(scenes, runs=..., scenarios=...)`.

    `ClosedLoopBenchmark.episodes` declares only `(scenes, runs)`, so a harness
    written to the documented signature raises TypeError on the runner's third
    keyword -- before the first scene, after the node has been started. That is
    exactly what happened on the first live attempt, so it is pinned here
    against the runner's real call rather than against the base class.
    """
    import inspect

    from paddock.runner import _run_one_model

    source = inspect.getsource(_run_one_model)
    assert "harness.episodes(scenes, runs=runs, scenarios=scenarios)" in source, (
        "the runner's call changed; update this test and the harness together"
    )
    bench = _benchmark(tmp_path, monkeypatch)
    episodes = list(bench.episodes(None, runs=1, scenarios=None))
    assert len(episodes) == 10


def test_a_scenario_filter_selects_the_scene_set(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch)
    assert list(bench.episodes(scenarios=[DEFAULT_SCENE_SET]))
    assert list(bench.episodes(scenarios=["something-else"])) == []


def test_an_unknown_scenario_is_named_rather_than_silently_empty(tmp_path: Path, monkeypatch) -> None:
    """A typo must not read as a benchmark with no work to do."""
    problems = _benchmark(tmp_path, monkeypatch).scenario_problems(["typo"])
    assert problems and "typo" in problems[0] and DEFAULT_SCENE_SET in problems[0]


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_without_a_site_section_says_which_keys_are_missing(tmp_path: Path, monkeypatch) -> None:
    problems = _benchmark(tmp_path, monkeypatch).preflight()
    assert problems and "[benchmarks.alpasim]" in problems[0]


def test_preflight_names_every_missing_piece_of_the_checkout(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch, repo=str(tmp_path / "alpasim"), python=sys.executable)
    problems = "\n".join(bench.preflight())
    assert "the wizard entry point" in problems
    assert "the scorer" in problems
    assert "the driver contract" in problems


def test_preflight_reports_an_interpreter_that_does_not_exist(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch, repo=str(tmp_path), python=str(tmp_path / "nope" / "python"))
    assert any("interpreter not found" in p for p in bench.preflight())


def test_run_config_records_what_changes_the_number(tmp_path: Path, monkeypatch) -> None:
    """Two runs that measured different things must not share a config hash."""
    config = _benchmark(tmp_path, monkeypatch).run_config()
    assert config["scene_set"] == DEFAULT_SCENE_SET
    assert config["scorer_repo_sha"] == "affc2ea"
    assert config["startup_timeout_s"] == 900


# ---------------------------------------------------------------------------
# the launch path, against the stand-in wizard
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch) -> Path:
    """A checkout whose wizard is the stand-in, and a bridge that is a no-op."""
    repo = tmp_path / "alpasim"
    (repo / "src" / "wizard" / "alpasim_wizard").mkdir(parents=True)
    (repo / "src" / "wizard" / "alpasim_wizard" / "main.py").write_text("")
    (repo / "src" / "eval" / "src" / "eval" / "aggregation").mkdir(parents=True)
    (repo / "src" / "eval" / "src" / "eval" / "aggregation" / "processing.py").write_text("")
    (repo / "src" / "grpc" / "alpasim_grpc" / "v0").mkdir(parents=True)
    (repo / "src" / "grpc" / "alpasim_grpc" / "v0" / "egodriver.proto").write_text("")

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    # The bridge needs AlpaSim's interpreter; these tests are about the wizard
    # side of run_episode, so stand it down rather than half-starting it.
    class NoBridge:
        def __init__(self, **kwargs):
            self.port = 9999

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(launcher, "BridgeProcess", NoBridge)
    monkeypatch.setattr(harness.launcher, "BridgeProcess", NoBridge)

    def fake_command(**kwargs):
        return [
            sys.executable,
            str(FIXTURES / "fake_wizard.py"),
            f"hydra.run.dir={kwargs['out_dir']}",
            f"scenes.scene_ids=[{kwargs['scene_id']}]",
        ]

    monkeypatch.setattr(harness.launcher, "wizard_command", fake_command)
    return repo


def _episode() -> Episode:
    return Episode(episode_id=CLEAN, scenario=DEFAULT_SCENE_SET, scene=CLEAN, config={})


def test_a_completed_scene_is_scored(tmp_path: Path, fake_repo: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_WIZARD_MODE", "ok")
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    metrics = bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)
    assert metrics["scene_score"] == 0.75
    assert "duration_s" in metrics


def test_a_nonzero_exit_fails_before_any_score_is_read(tmp_path: Path, fake_repo: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_WIZARD_MODE", "exit3")
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    with pytest.raises(RuntimeError, match="exited 3"):
        bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)


def test_exiting_zero_with_no_summary_is_an_error(tmp_path: Path, fake_repo: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_WIZARD_MODE", "no_summary")
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    with pytest.raises(RuntimeError, match="no results-summary.json"):
        bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)


def test_an_unscored_rollout_is_an_error_not_a_zero(tmp_path: Path, fake_repo: Path, monkeypatch) -> None:
    """A missing score must never be recorded as 0.0 -- that is a real result."""
    monkeypatch.setenv("FAKE_WIZARD_MODE", "unscored")
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    with pytest.raises(RuntimeError, match="no score"):
        bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)


def test_a_timeout_is_reported_as_one_and_the_group_is_killed(
    tmp_path: Path, fake_repo: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_WIZARD_MODE", "hang")
    bench = _benchmark(
        tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable, per_scene_timeout_s="2"
    )
    with pytest.raises(RuntimeError, match="timed out"):
        bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


def test_resume_returns_nothing_without_a_completion_marker(tmp_path: Path, monkeypatch) -> None:
    """Half an episode counted as whole is a wrong number, not a slow one."""
    out = tmp_path / "out"
    (out / "aggregate").mkdir(parents=True)
    (out / "aggregate" / "results-summary.json").write_text(SUMMARY.read_text())
    assert _benchmark(tmp_path, monkeypatch).completed_metrics(_episode(), out) is None


def test_resume_returns_metrics_when_the_rollout_completed(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    (out / "aggregate").mkdir(parents=True)
    (out / "aggregate" / "results-summary.json").write_text(SUMMARY.read_text())
    marker = out / "rollouts" / CLEAN / "session-0"
    marker.mkdir(parents=True)
    (marker / "_complete").write_text("")
    metrics = _benchmark(tmp_path, monkeypatch).completed_metrics(_episode(), out)
    assert metrics is not None and metrics["scene_score"] == 1.0
