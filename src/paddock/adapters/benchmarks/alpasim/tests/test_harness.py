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
import os
import re
import sys
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.alpasim import launcher
from paddock.adapters.benchmarks.alpasim.harness import (
    BENCHMARK,
    DEFAULT_RENDERER,
    DEFAULT_SCENE_SET,
    NotScored,
    load_scene_set,
    parse_summary,
    port_bases,
    scorer_sha,
)
from paddock.benchmark import Episode
from paddock.config import REPO_ROOT, SiteConfig
from paddock.registry import Registry
from paddock.results.store import sequence_hash

FIXTURES = Path(__file__).parent / "fixtures"
SUMMARY = FIXTURES / "results_summary_two_scenes.json"
#: The fixture's made-up outcomes, one per scene: which scene passes is invented too.
CLEAN = "clipgt-02e075b9-fd24-426b-971b-7cfcb2074cc9"
OFFROAD = "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"


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

    Losing that distinction would make a policy that drove 96% of a scene and
    then left the road indistinguishable from one that never moved.
    """
    metrics = parse_summary(SUMMARY, OFFROAD)
    assert metrics["scene_score"] == 0.0
    assert metrics["progress"] == pytest.approx(0.96, abs=1e-4)
    assert metrics["offroad"] == 1.0


# -- a rollout that crashed is not a rollout that scored zero -----------------
#
# `eval.allow_aggregation_with_failed_rollouts=true` -- which launcher.py sets so
# that one dead scene does not discard nine good ones -- makes AlpaSim emit a
# summary row for a rollout that never ran. Upstream's
# `eval/aggregation/failed_rollouts.py` writes that row with `score: 0.0`, NOT
# `None`, so the score alone cannot tell it from a model that drove into a wall.
# The two shapes are what differ, and this is the shape upstream produces.


def _crashed_row(clipgt_id: str) -> dict:
    """A failed-rollout row as `failed_rollout_summary_rows` writes it.

    Only the identity keys and an `error`: none of the metric keys a scored
    rollout carries, and a `failure_reason` holding the exception text rather
    than one of the scorer's hard-failure reasons.
    """
    return {
        "clipgt_id": clipgt_id,
        "rollout_id": "rollout-0",
        "run_name": "LR-test",
        "run_uuid": "0" * 64,
        "status": "fail",
        "passed": False,
        "failure_reason": "RpcError: driver call failed after 3 retries",
        "score": 0.0,
        "metrics": {
            "clipgt_id": clipgt_id,
            "rollout_id": "rollout-0",
            "run_name": "LR-test",
            "run_uuid": "0" * 64,
            "error": "RpcError: driver call failed after 3 retries",
        },
    }


def test_a_crashed_rollout_is_refused_not_recorded_as_zero(tmp_path: Path) -> None:
    """The bug this guards: an HTTP 500 landing on the leaderboard as 0.0.

    The bridge's RPC raising -- a model OOMs, the node stops answering, the
    renderer dies -- comes back through aggregation as `scene_score 0.0`,
    indistinguishable from a policy that drove off the road, and averages in.
    """
    body = json.loads(SUMMARY.read_text())
    body["rollouts"] = [_crashed_row(CLEAN)]
    path = tmp_path / "crashed.json"
    path.write_text(json.dumps(body))

    with pytest.raises(NotScored, match="did not run"):
        parse_summary(path, CLEAN)


def test_a_scene_the_scorer_zeroed_is_still_a_measurement() -> None:
    """The other half: over-rejecting would throw away real zeros.

    An offroad scene scores 0.0 legitimately -- it ran, it was scored, and the
    zero is the scorer's verdict on it. It carries a full numeric block and a
    hard-failure reason, which is exactly what the crashed row does not.
    """
    assert parse_summary(SUMMARY, OFFROAD)["scene_score"] == 0.0


def test_a_truncated_summary_does_not_take_the_sweep_with_it(tmp_path: Path) -> None:
    """A half-written results-summary.json raises JSONDecodeError, not NotScored.

    Uncaught it escapes the runner's per-episode handler and the whole sweep
    dies, discarding every episode already scored -- #7's lesson, one file over.
    """
    path = tmp_path / "truncated.json"
    path.write_text('{"rollouts": [{"clipgt_id": "x", "sco')
    with pytest.raises(json.JSONDecodeError):
        parse_summary(path, CLEAN)


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
    clean = next(r for r in body["rollouts"] if r["clipgt_id"] == CLEAN)
    body["rollouts"] = [dict(clean, score=None, status="failed")]
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


def test_the_thirty_scene_set_is_the_one_the_stored_rows_cover() -> None:
    """The 30-scene rows were ingested, not driven from this list, so the list
    is only worth tracking if it names the scenes those rows actually cover."""
    thirty = load_scene_set("public_2601_30scenes")
    assert len(thirty) == len(set(thirty)) == 30
    assert thirty[:10] == load_scene_set(DEFAULT_SCENE_SET)
    rows = sorted((REPO_ROOT / "results" / "alpasim").glob("*/legacy-*_30scenes/run.json"))
    assert rows, "no 30-scene row to check the list against"
    want = sequence_hash([f"alpasim/{scene}" for scene in thirty])
    for row in rows:
        assert json.loads(row.read_text())["dataset"]["sequences_sha256"] == want, row


def _samples(bench, tmp_path: Path) -> list[dict]:
    return [
        record
        for episode in bench.episodes()
        for record in bench.episode_samples(episode, {"scene_score": 1.0, "n_runs": 1.0}, tmp_path)
    ]


def test_a_sample_says_which_half_of_the_thirty_its_scene_is_in(tmp_path: Path, monkeypatch) -> None:
    """Not the run's set name for every scene: the ten are pilot10 and the twenty ext20."""
    samples = _samples(_benchmark(tmp_path, monkeypatch, scene_set="public_2601_30scenes"), tmp_path)
    assert [s["split"] for s in samples] == ["pilot10"] * 10 + ["ext20"] * 20
    assert {s["native_id"]["scene_set"] for s in samples} == {"public_2601_30scenes"}
    assert samples[0]["metrics"] == {"scene_score": 1.0}


def test_a_sample_says_how_its_car_was_steered(tmp_path: Path, monkeypatch) -> None:
    """The bridge's count of model vs derived headings lands in the record."""
    bench = _benchmark(tmp_path, monkeypatch)
    episode = next(bench.episodes())
    (record,) = bench.episode_samples(episode, {"scene_score": 1.0}, tmp_path)
    assert record["headings"] is None and record["bridge_plans"] is None, "no record, no claim"
    for counts, want in (({"model": 0, "derived": 9}, "derived"), ({"model": 3, "derived": 1}, "mixed")):
        (tmp_path / "bridge_plans.json").write_text(json.dumps({"headings": counts}))
        (record,) = bench.episode_samples(episode, {"scene_score": 1.0}, tmp_path)
        assert record["headings"] == want and record["bridge_plans"]["headings"] == counts
    assert record["history_frames"] is None
    (tmp_path / "bridge_plans.json").write_text(
        json.dumps({"headings": {"model": 1}, "history_frames": "observe"})
    )
    (record,) = bench.episode_samples(episode, {"scene_score": 1.0}, tmp_path)
    assert record["history_frames"] == "observe", "how the frame the bridge needed no plan from was sent"


def test_a_clip_is_the_same_sample_whichever_set_drove_it(tmp_path: Path, monkeypatch) -> None:
    """A paired comparison joins on sample_id: a 30-scene run must share its first ten
    with a 10-scene run, and fingerprint like the stored 30-scene rows."""
    ten = _samples(_benchmark(tmp_path, monkeypatch), tmp_path)
    thirty = _samples(_benchmark(tmp_path, monkeypatch, scene_set="public_2601_30scenes"), tmp_path)
    assert [s["split"] for s in ten] == ["pilot10"] * 10
    assert {s["sample_id"] for s in ten} < {s["sample_id"] for s in thirty}
    assert ten[0]["sample_id"] == f"alpasim/{load_scene_set(DEFAULT_SCENE_SET)[0]}"
    want = sequence_hash([f"alpasim/{scene}" for scene in load_scene_set("public_2601_30scenes")])
    assert sequence_hash([s["sample_id"] for s in thirty]) == want


def test_scene_order_is_the_pinned_order_not_the_filesystems(tmp_path: Path, monkeypatch) -> None:
    """`--limit 5` must select the same five at every site."""
    bench = _benchmark(tmp_path, monkeypatch)
    once = [e.scene for e in bench.episodes()]
    twice = [e.scene for e in bench.episodes()]
    assert once == twice == load_scene_set(DEFAULT_SCENE_SET)


def test_a_scene_filter_selects_only_what_was_asked_for(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch)
    assert [e.scene for e in bench.episodes(scenes=[CLEAN])] == [CLEAN]


def test_an_unknown_scene_is_named_before_a_model_loads(tmp_path: Path, monkeypatch) -> None:
    bench = _benchmark(tmp_path, monkeypatch)
    problems = bench.scene_problems("clipgt-nope")
    assert problems and "clipgt-nope" in problems[0]


def test_every_pinned_scene_passes_the_check_the_runner_makes(tmp_path: Path, monkeypatch) -> None:
    """`--scene` with a real clip id is accepted, called the way `run_benchmark` calls it.

    The runner splits ``--scene a,b`` and passes each id as a string; when this
    harness took a list it iterated the characters and refused every scene.
    """
    bench = _benchmark(tmp_path, monkeypatch)
    scene_arg = f"{CLEAN},{load_scene_set(DEFAULT_SCENE_SET)[1]}"
    scenes = [s.strip() for s in scene_arg.split(",") if s.strip()]  # as runner.run_benchmark does
    assert [p for one in scenes for p in bench.scene_problems(one)] == []


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


def test_a_run_with_its_own_port_range_moves_the_bridge_and_the_wizard_into_it(tmp_path: Path) -> None:
    """Two AlpaSim runs at once (two gate lanes) both found the same free ports from
    the fixed bases before either bound them. A run given PADDOCK_BASE_PORT keeps
    to its range; one without keeps the ports it always had."""
    assert port_bases({}) == (9200, None)
    assert port_bases({"PADDOCK_BASE_PORT": "9700"}) == (9750, 9760)
    assert port_bases({"PADDOCK_BASE_PORT": "9800"}) == (9850, 9860)

    def command(**kw) -> list[str]:
        return launcher.wizard_command(
            repo=tmp_path,
            python="python",
            scene_id="clipgt-x",
            out_dir=tmp_path / "out",
            bridge_port=9750,
            searchpath=None,
            **kw,
        )

    assert "wizard.baseport=9760" in command(baseport=9760)
    assert not any(arg.startswith("wizard.baseport=") for arg in command())


def test_the_wizard_is_not_told_to_force_determinism(tmp_path: Path) -> None:
    """AlpaSim's determinism flag seeds a model AlpaSim runs -- and it runs none.

    Under this adapter paddock owns the model process and the bridge only
    relays, so `driver.model.force_determinism` would be a no-op that reads as a
    guarantee. Sampling determinism belongs to the model's paddock policy.
    Asserted rather than merely omitted, because adding it back looks harmless.
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
    assert "force_determinism" not in joined
    assert "random_seed" not in joined


def test_the_wizard_gets_the_required_groups_and_the_log_dir(tmp_path: Path) -> None:
    """deploy/topology are required to compose; artifacts follow wizard.log_dir."""
    out = tmp_path / "out"
    joined = " ".join(
        launcher.wizard_command(
            repo=tmp_path,
            python="python",
            scene_id="clipgt-x",
            out_dir=out,
            bridge_port=9200,
            searchpath=None,
        )
    )
    assert "deploy=local" in joined and "topology=1gpu" in joined
    assert f"wizard.log_dir={out}" in joined
    # The adapter's own configs must be on the search path or `driver=` resolves
    # to nothing and the run dies during composition.
    assert f"file://{launcher.CONFIGS_DIR}" in joined
    assert "driver=paddock_bridge" in joined


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
    assert config["startup_timeout_s"] == 900


def test_the_scorer_sha_is_read_from_repos_yaml_not_copied() -> None:
    """The pin and the recorded sha cannot be allowed to drift apart.

    The scorer ships inside the pinned AlpaSim repo, so bumping that pin -- in
    its own PR, which is the workflow repos.yaml exists to support -- moves
    every number this benchmark produces. A literal in the harness would keep
    reporting the old sha while something else did the scoring: the field that
    exists to explain a change in the numbers would be the field that hid it.
    """
    assert scorer_sha() == Registry.load().repos["alpasim"].sha


def test_the_renderer_is_in_the_run_config(tmp_path: Path, monkeypatch) -> None:
    """nuRec and OmniDreams are separate columns in the README for a reason.

    They are different reconstructions of the same clip, so a score from one is
    not a statement about the other. Without the renderer in run_config the two
    produce the same config hash and the same run.json, and results/alpasim/
    cannot tell them apart -- the first OmniDreams campaign would average
    silently against nuRec rows.
    """
    config = _benchmark(tmp_path, monkeypatch).run_config()
    assert config["renderer"] == DEFAULT_RENDERER


def test_an_unwired_renderer_is_refused_in_preflight(tmp_path: Path, monkeypatch) -> None:
    """Recording a run as omnidreams that rendered nuRec is worse than no run."""
    bench = _benchmark(tmp_path, monkeypatch, renderer="omnidreams")
    assert any("renderer" in p and "omnidreams" in p for p in bench.preflight())


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
    # No node answers here; the stand-in serves a model with the default profile.
    monkeypatch.setattr(harness.launcher, "serving_model", lambda url: "constant_velocity")
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


# -- a bridge that starts but never binds must not outlive the failure ---------


def test_a_bridge_that_never_binds_is_killed_not_leaked(tmp_path: Path) -> None:
    """`__enter__` raising means the `with` never entered, so `__exit__` never runs.

    The child is started with `start_new_session=True`, so without an explicit
    teardown on this path it survives in its own process group holding the
    bridge port, under AlpaSim's interpreter, for whoever uses the box next.
    Here the "bridge" is a sleep that never listens, so `wait_ready` times out.
    """
    import sys

    bridge = launcher.BridgeProcess(
        python=sys.executable,
        policy_url="http://127.0.0.1:1",
        port=1,  # privileged and unbound: nothing will ever accept here
        version_id="test",
        log_path=tmp_path / "bridge.log",
    )
    # A stand-in for the real bridge command: alive, silent, never binding.
    bridge._command_override = [sys.executable, "-c", "import time; time.sleep(300)"]
    bridge.ready_timeout_s = 2.0

    with pytest.raises(RuntimeError):
        with bridge:
            pass

    assert bridge.proc is not None
    assert bridge.proc.poll() is not None, "the bridge survived a failed __enter__"


def test_a_scene_subset_is_not_a_score(tmp_path: Path, monkeypatch) -> None:
    """Fewer clips than the pinned set would be ranked beside full runs."""
    bench = _benchmark(tmp_path, monkeypatch)
    list(bench.episodes(None))
    assert bench.invalid_reasons() == []
    list(bench.episodes([CLEAN]))
    (reason,) = bench.invalid_reasons()
    assert f"1 of the {len(load_scene_set(DEFAULT_SCENE_SET))} clips" in reason


def test_a_model_without_a_profile_is_driven_exactly_as_before(tmp_path: Path, monkeypatch) -> None:
    """Any model with no alpasim.yaml: default driver, no bridge flags, nothing recorded.

    A made-up model, so a real one can gain a profile without editing this test.
    """
    from types import SimpleNamespace

    from paddock.adapters.benchmarks.alpasim import harness

    bench = _benchmark(tmp_path, monkeypatch)
    models_dir = tmp_path / "models"
    (models_dir / "m").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(harness, "MODELS_DIR", models_dir)
    assert not harness.has_profile("m")
    profile = harness.profile_for("m")
    assert profile == harness.DriverProfile()
    assert (profile.driver, profile.bridge_args, profile.scenes_csv) == ("paddock_bridge", (), None)
    assert bench.model_run_config(SimpleNamespace(name="m")) == {}


def test_the_drivejepa_profile_pins_every_scene_it_can_be_asked_for() -> None:
    """Its driver config exists, and its catalog pins each 30-scene id to one 26.01 artifact."""
    import csv

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("drivejepa")
    assert (launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").is_file()
    rows = list(csv.DictReader((harness.SCENES_DIR / profile.scenes_csv).open()))
    ids = [row["scene_id"] for row in rows]
    assert sorted(ids) == sorted(load_scene_set("public_2601_30scenes"))
    assert set(load_scene_set("public_2601_first10")) <= set(ids)
    assert {row["nre_version_string"] for row in rows} == {"26.1.112-59673f88"}
    assert len({row["uuid"] for row in rows}) == len(rows)


def test_the_node_gets_a_minute_to_say_which_model_it_serves(monkeypatch) -> None:
    # 10 s failed three scenes of a 30-scene run while the disk was saturated;
    # the node was healthy and answered the scene after. Startup only: this is
    # asked once per scene, before the bridge starts.
    waited: list[float] = []

    class _Answer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return json.dumps({"model": "drivejepa"}).encode()

    def urlopen(url, timeout):
        assert url.endswith("/whoami")
        waited.append(timeout)
        return _Answer()

    monkeypatch.setattr(launcher.urllib.request, "urlopen", urlopen)
    assert launcher.serving_model("http://127.0.0.1:9800") == "drivejepa"
    assert waited == [60.0]


def test_drivejepa_records_video_only_when_the_site_asks(
    tmp_path: Path, fake_repo: Path, monkeypatch
) -> None:
    """Its profile pins driver, cameras and scenes, but not video: render_video only
    renders after AlpaSim's scorer has written its metrics, so it changes no score,
    and every-frame video is about 150 MB a scene."""
    from paddock.adapters.benchmarks.alpasim import harness

    monkeypatch.setenv("FAKE_WIZARD_MODE", "ok")
    monkeypatch.setattr(harness.launcher, "serving_model", lambda url: "drivejepa")
    seen: list[dict] = []
    fake = harness.launcher.wizard_command

    def capture(**kwargs):
        seen.append(kwargs)
        return fake(**kwargs)

    monkeypatch.setattr(harness.launcher, "wizard_command", capture)
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)
    (call,) = seen
    assert call["driver"] == "paddock_bridge_drivejepa"
    assert (call["video"], call["video_every_nth_frame"]) == (False, 1)


def test_the_drivejepa_bridge_flags_match_its_driver_config() -> None:
    """The bridge's control step and warm-up length are the driver config's.

    A different control step would fail the first scene loudly (PushCadenceError);
    a different warm-up would only mislabel warm-up and after-handover replans.
    """
    import yaml

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("drivejepa")
    args = list(profile.bridge_args)
    flag = {name: int(args[args.index(name) + 1]) for name in ("--frame-interval-us", "--force-gt-us")}
    config = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").read_text())
    simulation = config["runtime"]["simulation_config"]
    assert flag["--frame-interval-us"] == simulation["control_timestep_us"]
    assert flag["--force-gt-us"] == simulation["force_gt_duration_us"]
    (front,) = [c for c in simulation["cameras"] if c["logical_id"] == args[args.index("--front-camera") + 1]]
    assert front["frame_interval_us"] == simulation["control_timestep_us"]


def test_autoe2e_is_driven_on_its_own_driver_config_and_bridge_flags(
    tmp_path: Path, fake_repo: Path, monkeypatch
) -> None:
    """The node's model name selects the autoe2e profile: its driver config, its six
    camera names, its rhythm and padding flags, and a per-replan log beside the samples."""
    from paddock.adapters.benchmarks.alpasim import harness

    monkeypatch.setenv("FAKE_WIZARD_MODE", "ok")
    monkeypatch.setattr(harness.launcher, "serving_model", lambda url: "autoe2e")
    seen: list[dict] = []
    bridges: list[dict] = []
    fake = harness.launcher.wizard_command
    no_bridge = harness.launcher.BridgeProcess

    def capture(**kwargs):
        seen.append(kwargs)
        return fake(**kwargs)

    def bridge(**kwargs):
        bridges.append(kwargs)
        return no_bridge(**kwargs)

    monkeypatch.setattr(harness.launcher, "wizard_command", capture)
    monkeypatch.setattr(harness.launcher, "BridgeProcess", bridge)
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)
    (call,) = seen
    (started,) = bridges
    assert call["driver"] == "paddock_bridge_autoe2e"
    args = list(started["extra_args"])
    names = [args[i + 1] for i, a in enumerate(args) if a == "--camera-name"]
    assert names == [
        "camera_front_wide_120fov=CAM_FRONT",
        "camera_cross_left_120fov=CAM_FRONT_LEFT",
        "camera_cross_right_120fov=CAM_FRONT_RIGHT",
        "camera_rear_left_70fov=CAM_BACK_LEFT",
        "camera_front_tele_30fov=CAM_BACK",
        "camera_rear_right_70fov=CAM_BACK_RIGHT",
    ]
    assert "--placeholder" not in args and "--planner" not in args
    for flag in ("--pinhole-intrinsics", "--pad-ego-history"):
        assert flag in args
    assert int(args[args.index("--replan-every-us") + 1]) == 100_000
    assert args[args.index("--replan-log") + 1].endswith(".replans.jsonl")
    # The six node names are the six cameras the model declares, in its slot order.
    model = Registry.load().models["autoe2e"]
    assert [n.split("=")[1] for n in names] == list(model.cameras)
    assert "alpasim" in model.benchmarks


def test_the_autoe2e_profile_pins_every_scene_it_can_be_asked_for() -> None:
    """Its driver config exists, and it uses the same 26.01 uuid catalog as drivejepa."""
    import csv

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("autoe2e")
    assert (launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").is_file()
    assert profile.scenes_csv == harness.profile_for("drivejepa").scenes_csv
    rows = list(csv.DictReader((harness.SCENES_DIR / profile.scenes_csv).open()))
    assert sorted(row["scene_id"] for row in rows) == sorted(load_scene_set("public_2601_30scenes"))
    assert {row["nre_version_string"] for row in rows} == {"26.1.112-59673f88"}
    assert len({row["uuid"] for row in rows}) == len(rows)


def test_the_autoe2e_bridge_flags_match_its_driver_config() -> None:
    """Control step and warm-up are the driver config's; the six renamed cameras are
    exactly the six it renders and re-declares, each an undistorted pinhole with a 1 us
    global shutter, whose size, focal length and principal point give the KITScenes
    field of view."""
    import math

    import yaml

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("autoe2e")
    args = list(profile.bridge_args)
    flag = {name: int(args[args.index(name) + 1]) for name in ("--frame-interval-us", "--force-gt-us")}
    config = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").read_text())
    simulation = config["runtime"]["simulation_config"]
    assert flag["--frame-interval-us"] == simulation["control_timestep_us"]
    assert flag["--force-gt-us"] == simulation["force_gt_duration_us"]
    assert simulation["skip_driver_during_force_gt"] is False
    assert int(args[args.index("--replan-every-us") + 1]) % simulation["control_timestep_us"] == 0

    sources = [args[i + 1].split("=")[0] for i, a in enumerate(args) if a == "--camera-name"]
    extra = {c["logical_id"]: c for c in config["runtime"]["extra_cameras"]}
    rendered = {c["logical_id"]: c for c in simulation["cameras"]}
    assert sorted(sources) == sorted(extra) == sorted(rendered)
    fov = {"camera_front_wide_120fov": (88.2, 54.4)}
    for camera_id, camera in extra.items():
        pinhole = camera["intrinsics"]["opencv_pinhole"]
        assert camera["intrinsics"]["model"] == "opencv_pinhole"
        assert camera["shutter_type"] == "GLOBAL", camera_id
        for key, count in (("radial", 6), ("tangential", 2), ("thin_prism", 4)):
            assert pinhole[key] == [0.0] * count, (camera_id, key)
        height, width = camera["resolution_hw"]
        assert (rendered[camera_id]["height"], rendered[camera_id]["width"]) == (height, width)
        assert rendered[camera_id]["frame_interval_us"] == simulation["control_timestep_us"]
        # The image is drawn at shutter open and labelled with the pose at shutter close.
        assert rendered[camera_id]["shutter_duration_us"] == 1, camera_id
        fx, fy = pinhole["focal_length"]
        assert fx == fy
        assert pinhole["principal_point"] == [width / 2, height / 2]
        h_fov = 2 * math.degrees(math.atan(width / 2 / fx))
        v_fov = 2 * math.degrees(math.atan(height / 2 / fy))
        assert (h_fov, v_fov) == pytest.approx(fov.get(camera_id, (87.1, 63.3)), abs=0.06)
    # Each scene's own pose for the front and cross cameras; one pose for the three new directions.
    assert {c for c, v in extra.items() if "rig_to_camera" in v} == {
        "camera_rear_left_70fov",
        "camera_front_tele_30fov",
        "camera_rear_right_70fov",
    }


def test_the_autoe2e_fixed_poses_look_where_the_kitscenes_cameras_look() -> None:
    """rig_to_camera is the camera's pose in the rig with OpenCV axes (z forward, y down):
    the optical axis of each fixed view points at its KITScenes yaw, level, image upright."""
    import math

    import numpy as np
    import yaml

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    def rotation(x, y, z, w):
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    profile = harness.profile_for("autoe2e")
    config = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").read_text())
    extra = {c["logical_id"]: c for c in config["runtime"]["extra_cameras"]}
    expected = {
        "camera_rear_left_70fov": (120.0, [2.10, 1.04, 1.02]),
        "camera_front_tele_30fov": (180.0, [1.0, 0.0, 1.93]),
        "camera_rear_right_70fov": (-120.0, [2.06, -1.01, 1.02]),
    }
    for camera_id, (yaw, position) in expected.items():
        pose = extra[camera_id]["rig_to_camera"]
        assert pose["translation_m"] == position
        matrix = rotation(*pose["rotation_xyzw"])
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        optical, down = matrix[:, 2], matrix[:, 1]
        heading = [math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0]
        assert optical == pytest.approx(heading, abs=1e-9)
        assert down == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


# -- every run's containers are logged and taken down, whatever the outcome ---


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch) -> Path:
    """A `docker` on PATH that records each call and answers `compose logs`.

    `FAKE_DOCKER_DOWN=fail` makes `compose down` exit non-zero, standing in for
    a daemon that refuses the teardown. The compose project each call would act
    on (`COMPOSE_PROJECT_NAME`) goes to `docker_projects.txt`, one per call.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "docker_calls.txt"
    projects = tmp_path / "docker_projects.txt"
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        f"open({str(projects)!r}, 'a').write(os.environ.get('COMPOSE_PROJECT_NAME', '') + '\\n')\n"
        "if 'logs' in sys.argv:\n"
        "    print('2026-10-27T00:00:00Z runtime-0-1  | hello from the container')\n"
        "if 'down' in sys.argv and os.environ.get('FAKE_DOCKER_DOWN') == 'fail':\n"
        "    sys.exit(1)\n"
        "if sys.argv[1:2] == ['ps']:\n"
        "    print(os.environ.get('FAKE_DOCKER_PS', ''))\n"
        "if sys.argv[1:2] == ['update'] and os.environ.get('FAKE_DOCKER_UPDATE') == 'fail':\n"
        "    print('Error response from daemon: no such container', file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


def _run_dir(tmp_path: Path) -> Path:
    out = tmp_path / "clipgt-run"
    out.mkdir()
    # The wizard writes this before `compose up`; teardown keys off it.
    (out / "docker-compose.yaml").write_text("services: {}\n")
    return out


def _calls(calls: Path, verb: str) -> int:
    lines = calls.read_text().splitlines() if calls.exists() else []
    return sum(1 for line in lines if f" {verb}" in line)


def test_a_successful_run_saves_logs_and_is_taken_down_once(tmp_path: Path, fake_docker: Path) -> None:
    out = _run_dir(tmp_path)
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "pass"], tmp_path, out / "wizard.log", 30, echo=lambda _: None
    )
    assert (returncode, timed_out) == (0, False)
    assert "hello from the container" in (out / "container_logs.txt").read_text()
    assert _calls(fake_docker, "logs") == 1
    assert _calls(fake_docker, "down") == 1
    # Logs before down: down removes the containers the logs live in.
    verbs = [line.split()[3] for line in fake_docker.read_text().splitlines()]
    assert verbs == ["logs", "down"]


def test_a_failing_run_is_taken_down_and_keeps_its_exit_code(tmp_path: Path, fake_docker: Path) -> None:
    out = _run_dir(tmp_path)
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "raise SystemExit(3)"], tmp_path, out / "wizard.log", 30, echo=lambda _: None
    )
    assert (returncode, timed_out) == (3, False)
    assert (out / "container_logs.txt").exists()
    assert _calls(fake_docker, "down") == 1


def test_a_timed_out_run_is_taken_down_once_not_twice(tmp_path: Path, fake_docker: Path) -> None:
    out = _run_dir(tmp_path)
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        tmp_path,
        out / "wizard.log",
        1,
        echo=lambda _: None,
    )
    assert timed_out and returncode != 0
    assert (out / "container_logs.txt").exists()
    assert _calls(fake_docker, "down") == 1


def test_an_interrupted_run_is_taken_down_and_the_interrupt_still_raises(
    tmp_path: Path, fake_docker: Path
) -> None:
    """Ctrl-C reaches paddock but not the wizard, which runs in its own session.

    The stand-in wizard sends the parent a SIGINT -- what a Ctrl-C at the
    terminal does -- then keeps running, as the real one would.
    """
    out = _run_dir(tmp_path)
    pid_file = tmp_path / "wizard.pid"
    wizard = (
        "import os, signal, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "os.kill(os.getppid(), signal.SIGINT)\n"
        "time.sleep(300)\n"
    )
    with pytest.raises(KeyboardInterrupt):
        launcher.run_wizard(
            [sys.executable, "-c", wizard], tmp_path, out / "wizard.log", 60, echo=lambda _: None
        )
    assert (out / "container_logs.txt").exists()
    assert _calls(fake_docker, "down") == 1
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_a_failing_teardown_does_not_change_the_runs_result(
    tmp_path: Path, fake_docker: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_DOWN", "fail")
    out = _run_dir(tmp_path)
    said: list[str] = []
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "raise SystemExit(3)"], tmp_path, out / "wizard.log", 30, echo=said.append
    )
    assert (returncode, timed_out) == (3, False)
    assert any("warning: compose down exited 1" in line for line in said)


def test_no_docker_at_all_is_a_warning_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    out = _run_dir(tmp_path)
    said: list[str] = []
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "pass"], tmp_path, out / "wizard.log", 30, echo=said.append
    )
    assert (returncode, timed_out) == (0, False)
    assert any("could not save container logs" in line for line in said)
    assert any("compose down failed" in line for line in said)


def test_two_runs_of_one_scene_do_not_share_containers(tmp_path: Path, fake_docker: Path) -> None:
    """Overlapping runs of one scene must not start or remove each other's containers.

    Left to itself, compose names a project after the folder that holds
    `docker-compose.yaml` -- here `clipgt-scene` for both runs -- so the first
    run to finish would remove the second's containers mid-scene.
    """
    names = []
    for run_id in ("20261101T000000-aaaaaa", "20261101T010000-bbbbbb"):
        out = tmp_path / run_id / "clipgt-scene"
        out.mkdir(parents=True)
        (out / "docker-compose.yaml").write_text("services: {}\n")
        seen = out / "wizard_saw.txt"
        wizard = f"import os; open({str(seen)!r}, 'w').write(os.environ.get('COMPOSE_PROJECT_NAME', ''))"
        launcher.run_wizard(
            [sys.executable, "-c", wizard], tmp_path, out / "wizard.log", 30, echo=lambda _: None
        )
        names.append(seen.read_text())

    assert names[0] and names[1] and names[0] != names[1]
    # Docker refuses any other character, upper case included.
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) for name in names)
    # The logs and the teardown act on the project the wizard started, per run.
    projects = (tmp_path / "docker_projects.txt").read_text().split()
    assert projects == [names[0], names[0], names[1], names[1]]


def test_a_run_held_to_some_cores_holds_its_containers_to_them(
    tmp_path: Path, fake_docker: Path, monkeypatch
) -> None:
    """Docker starts AlpaSim's containers, so the lane's taskset never reached them and the gate
    ran AlpaSim beside no other simulator. Each container is pinned once, as it appears."""
    monkeypatch.setattr(launcher, "run_cores", lambda: "12-15")
    monkeypatch.setattr(launcher, "PIN_POLL_S", 0.05)
    monkeypatch.setenv("FAKE_DOCKER_PS", "c1\nc2")
    out = _run_dir(tmp_path)
    said: list[str] = []
    launcher.run_wizard(
        [sys.executable, "-c", "import time; time.sleep(0.6)"],
        tmp_path,
        out / "wizard.log",
        30,
        echo=said.append,
    )
    updates = [line for line in fake_docker.read_text().splitlines() if line.startswith("update")]
    assert updates == ["update --cpuset-cpus 12-15 c1 c2"], "pinned once, both containers"
    assert any(
        "label=com.docker.compose.project=" + launcher.compose_project_name(out) in line
        for line in fake_docker.read_text().splitlines()
        if line.startswith("ps")
    )
    assert "pinned 2 container(s) to cores 12-15" in said


def test_a_run_on_every_core_pins_nothing(tmp_path: Path, fake_docker: Path, monkeypatch) -> None:
    monkeypatch.setattr(launcher, "run_cores", lambda: None)
    out = _run_dir(tmp_path)
    launcher.run_wizard(
        [sys.executable, "-c", "pass"], tmp_path, out / "wizard.log", 30, echo=lambda _m: None
    )
    assert not any(line.startswith(("ps", "update")) for line in fake_docker.read_text().splitlines())


def test_a_pin_that_fails_warns_once_and_leaves_the_run_alone(
    tmp_path: Path, fake_docker: Path, monkeypatch
) -> None:
    monkeypatch.setattr(launcher, "run_cores", lambda: "12-15")
    monkeypatch.setattr(launcher, "PIN_POLL_S", 0.05)
    monkeypatch.setenv("FAKE_DOCKER_PS", "c1")
    monkeypatch.setenv("FAKE_DOCKER_UPDATE", "fail")
    out = _run_dir(tmp_path)
    said: list[str] = []
    returncode, timed_out, _ = launcher.run_wizard(
        [sys.executable, "-c", "import time; time.sleep(0.4)"],
        tmp_path,
        out / "wizard.log",
        30,
        echo=said.append,
    )
    assert (returncode, timed_out) == (0, False)
    warnings = [m for m in said if m.startswith("warning: could not pin")]
    assert len(warnings) == 1 and "no such container" in warnings[0]


def test_the_run_cores_read_as_dockers_cpuset(monkeypatch) -> None:
    monkeypatch.setattr(launcher.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: {12, 13, 14, 15, 20})
    assert launcher.run_cores() == "12-15,20"
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: set(range(32)))
    assert launcher.run_cores() is None, "no taskset: nothing to hold the containers to"


def test_qwendrive_is_driven_on_its_own_driver_config_and_bridge_flags(
    tmp_path: Path, fake_repo: Path, monkeypatch
) -> None:
    """The node's model name selects the qwendrive profile: its driver config, its three
    camera names and nothing else, the 1.5 s rhythm with every step observed, its canbus,
    route and scene-id switches, 0.1 s waypoints, and a per-replan log beside the samples."""
    from paddock.adapters.benchmarks.alpasim import harness

    monkeypatch.setenv("FAKE_WIZARD_MODE", "ok")
    monkeypatch.setattr(harness.launcher, "serving_model", lambda url: "qwendrive")
    seen: list[dict] = []
    bridges: list[dict] = []
    fake = harness.launcher.wizard_command
    no_bridge = harness.launcher.BridgeProcess

    def capture(**kwargs):
        seen.append(kwargs)
        return fake(**kwargs)

    def bridge(**kwargs):
        bridges.append(kwargs)
        return no_bridge(**kwargs)

    monkeypatch.setattr(harness.launcher, "wizard_command", capture)
    monkeypatch.setattr(harness.launcher, "BridgeProcess", bridge)
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)
    (call,) = seen
    (started,) = bridges
    assert call["driver"] == "paddock_bridge_qwendrive"
    args = list(started["extra_args"])
    names = [args[i + 1] for i, a in enumerate(args) if a == "--camera-name"]
    assert names == [
        "camera_front_wide_120fov=CAM_FRONT",
        "camera_cross_left_120fov=CAM_FRONT_LEFT",
        "camera_cross_right_120fov=CAM_FRONT_RIGHT",
    ]
    for flag in (
        "--named-cameras-only",
        "--observe-between-replans",
        "--canbus-ego-frame",
        "--send-route",
        "--send-scene-id",
    ):
        assert flag in args
    for flag in (
        "--placeholder",
        "--planner",
        "--pinhole-intrinsics",
        "--pad-ego-history",
        "--canbus-acceleration",
    ):
        assert flag not in args
    assert int(args[args.index("--replan-every-us") + 1]) == 1_500_000
    assert float(args[args.index("--waypoint-dt") + 1]) == 0.1
    assert args[args.index("--replan-log") + 1].endswith(".replans.jsonl")
    # The three node names are the three cameras the model declares, in its slot order.
    model = Registry.load().models["qwendrive"]
    assert [n.split("=")[1] for n in names] == list(model.cameras)
    assert "alpasim" in model.benchmarks


def test_the_qwendrive_profile_is_the_native_lane() -> None:
    """Control step, warm-up and 30-scene catalog as drivejepa; four f-theta cameras at 20 Hz
    (no extra_cameras), 30 ms shutter, the driver awake in the warm-up, the hyperion_8_1 mask,
    and a replan period that lands on the handover tick."""
    import yaml

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("qwendrive")
    args = list(profile.bridge_args)
    flag = {name: int(args[args.index(name) + 1]) for name in ("--frame-interval-us", "--force-gt-us")}
    config = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").read_text())
    simulation = config["runtime"]["simulation_config"]
    assert flag["--frame-interval-us"] == simulation["control_timestep_us"] == 50_000
    assert flag["--force-gt-us"] == simulation["force_gt_duration_us"] == 3_000_000
    assert simulation["skip_driver_during_force_gt"] is False
    assert (simulation["n_sim_steps"], simulation["ego_mask_rig_config_id"]) == (400, "hyperion_8_1")
    replan = int(args[args.index("--replan-every-us") + 1])
    assert (
        replan % simulation["control_timestep_us"] == 0 and simulation["force_gt_duration_us"] % replan == 0
    )
    assert "extra_cameras" not in config.get("runtime", {})
    rendered = {c["logical_id"]: c for c in simulation["cameras"]}
    named = {args[i + 1].split("=")[0] for i, a in enumerate(args) if a == "--camera-name"}
    assert named < set(rendered) and set(rendered) - named == {"camera_front_tele_30fov"}
    for camera in rendered.values():
        assert camera["frame_interval_us"] == simulation["control_timestep_us"]
        assert camera["shutter_duration_us"] == 30_000
        assert (camera["height"], camera["width"]) == (1080, 1900)
    assert profile.scenes_csv == harness.profile_for("drivejepa").scenes_csv


def test_alpamayo_is_driven_on_its_own_driver_config_and_bridge_flags(
    tmp_path: Path, fake_repo: Path, monkeypatch
) -> None:
    """The node's model name selects the alpamayo15 profile: its driver config, its four
    camera names and nothing else, the 100 ms rhythm with every step observed, the route,
    its time and the plan spacing, 0.1 s waypoints, and a per-replan log beside the samples."""
    from paddock.adapters.benchmarks.alpasim import harness

    monkeypatch.setenv("FAKE_WIZARD_MODE", "ok")
    monkeypatch.setattr(harness.launcher, "serving_model", lambda url: "alpamayo15")
    seen: list[dict] = []
    bridges: list[dict] = []
    fake = harness.launcher.wizard_command
    no_bridge = harness.launcher.BridgeProcess

    def capture(**kwargs):
        seen.append(kwargs)
        return fake(**kwargs)

    def bridge(**kwargs):
        bridges.append(kwargs)
        return no_bridge(**kwargs)

    monkeypatch.setattr(harness.launcher, "wizard_command", capture)
    monkeypatch.setattr(harness.launcher, "BridgeProcess", bridge)
    bench = _benchmark(tmp_path, monkeypatch, repo=str(fake_repo), python=sys.executable)
    bench.run_episode(_episode(), "http://127.0.0.1:1", tmp_path / "out", echo=lambda _: None)
    (call,) = seen
    (started,) = bridges
    assert call["driver"] == "paddock_bridge_alpamayo15"
    assert call["scenes_csv"].name == "public_2601_30scenes_26p01.csv"
    args = list(started["extra_args"])
    names = [args[i + 1] for i, a in enumerate(args) if a == "--camera-name"]
    assert names == [
        "camera_front_wide_120fov=CAM_FRONT",
        "camera_cross_left_120fov=CAM_FRONT_LEFT",
        "camera_cross_right_120fov=CAM_FRONT_RIGHT",
        "camera_front_tele_30fov=CAM_FRONT_TELE",
    ]
    for flag in (
        "--named-cameras-only",
        "--observe-between-replans",
        "--send-route",
        "--send-route-time",
        "--send-plan-dt",
    ):
        assert flag in args
    for flag in (
        "--placeholder",
        "--planner",
        "--pinhole-intrinsics",
        "--pad-ego-history",
        "--canbus-ego-frame",
        "--canbus-acceleration",
        "--send-scene-id",
    ):
        assert flag not in args
    assert int(args[args.index("--replan-every-us") + 1]) == 100_000
    assert float(args[args.index("--waypoint-dt") + 1]) == 0.1
    assert args[args.index("--replan-log") + 1].endswith(".replans.jsonl")
    # The node names are the policy's own: the three it declares and the telephoto.
    from paddock.adapters.models.alpamayo15.policy import AlpamayoPolicy

    assert sorted(n.split("=")[1] for n in names) == sorted(AlpamayoPolicy.cameras)
    model = Registry.load().models["alpamayo15"]
    assert "alpasim" in model.benchmarks


def test_the_alpamayo_profile_is_the_stored_rows_simulation() -> None:
    """Control step, warm-up and 30-scene catalog as drivejepa; the four f-theta cameras AlpaSim's
    Alpamayo driver reads, at 20 Hz (no extra_cameras), 30 ms shutter, 1080 x 1900; the driver awake
    in the warm-up; the hyperion_8_1 mask; a replan period of the model's 100 ms that lands on the
    handover tick; and the same simulation values as the qwendrive profile."""
    import yaml

    from paddock.adapters.benchmarks.alpasim import harness, launcher

    profile = harness.profile_for("alpamayo15")
    args = list(profile.bridge_args)
    flag = {name: int(args[args.index(name) + 1]) for name in ("--frame-interval-us", "--force-gt-us")}
    config = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / f"{profile.driver}.yaml").read_text())
    simulation = config["runtime"]["simulation_config"]
    assert flag["--frame-interval-us"] == simulation["control_timestep_us"] == 50_000
    assert flag["--force-gt-us"] == simulation["force_gt_duration_us"] == 3_000_000
    assert simulation["skip_driver_during_force_gt"] is False
    assert (simulation["n_sim_steps"], simulation["ego_mask_rig_config_id"]) == (400, "hyperion_8_1")
    replan = int(args[args.index("--replan-every-us") + 1])
    assert replan == 100_000
    assert (
        replan % simulation["control_timestep_us"] == 0 and simulation["force_gt_duration_us"] % replan == 0
    )
    assert "extra_cameras" not in config.get("runtime", {})
    rendered = {c["logical_id"]: c for c in simulation["cameras"]}
    named = {args[i + 1].split("=")[0] for i, a in enumerate(args) if a == "--camera-name"}
    assert (
        named
        == set(rendered)
        == {
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        }
    )
    for camera in rendered.values():
        assert camera["frame_interval_us"] == simulation["control_timestep_us"]
        assert camera["shutter_duration_us"] == 30_000
        assert (camera["height"], camera["width"]) == (1080, 1900)
    assert profile.scenes_csv == harness.profile_for("drivejepa").scenes_csv
    qwen = yaml.safe_load((launcher.CONFIGS_DIR / "driver" / "paddock_bridge_qwendrive.yaml").read_text())
    assert simulation == qwen["runtime"]["simulation_config"]
