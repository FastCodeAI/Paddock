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


# ---- debugging video ---------------------------------------------------


def _frames(run_dir, stream, n=3):
    """A few real-ish PNG frames named the way the engine names them."""
    from PIL import Image

    d = run_dir / stream
    d.mkdir(parents=True)
    for i in range(n):
        # Microsecond timestamps, as the engine writes them; even dimensions are
        # NOT used here on purpose -- H.264 needs even, and the scale filter in
        # _encode is what makes an odd matplotlib figure encodable.
        Image.new("RGB", (65, 33), (i * 40, 90, 140)).save(d / f"153315160{i:04d}7766.png")
    return d


def _harness_with_video(mode):
    harness = _harness()
    real = harness.setting

    def setting(key, default=None):
        return mode if key == "video" else real(key, default)

    harness.setting = setting
    return harness


def test_video_mode_defaults_to_encode_and_rejects_nonsense() -> None:
    assert _harness().video_mode() == "encode"
    with pytest.raises(ValueError, match="expected one of"):
        _harness_with_video("mp4").video_mode()


def test_encode_writes_one_video_per_run_and_prunes_the_frames(tmp_path) -> None:
    """The frames are reconstructible from CAM_* plus the json, so pruning is safe.

    83 MB per run is 66 GB across a 20-pair sweep at 10 runs for four models;
    the same content as video is about 5 GB.
    """
    harness = _harness_with_video("encode")
    if harness._ffmpeg() is None:
        pytest.skip("no ffmpeg")

    for i in (0, 1):
        run = tmp_path / f"run_{i}"
        _frames(run, "FC_TRAJ")
        (run / "CAM_FRONT").mkdir()
        (run / "trajectories.json").write_text("{}")

    written = harness.render_videos(tmp_path, echo=lambda _: None)

    assert [p.relative_to(tmp_path).as_posix() for p in written] == [
        "run_0/FC_TRAJ.mp4",
        "run_1/FC_TRAJ.mp4",
    ]
    for i in (0, 1):
        run = tmp_path / f"run_{i}"
        assert (run / "FC_TRAJ.mp4").stat().st_size > 0
        assert not (run / "FC_TRAJ").exists(), "encoded frames should be pruned"
        # What makes pruning safe: the raw camera and the trajectories survive.
        assert (run / "CAM_FRONT").is_dir()
        assert (run / "trajectories.json").exists()
    assert not list(tmp_path.rglob("*.concat")), "the concat list is a temp file"


def test_keep_encodes_without_deleting_anything(tmp_path) -> None:
    harness = _harness_with_video("keep")
    if harness._ffmpeg() is None:
        pytest.skip("no ffmpeg")
    _frames(tmp_path / "run_0", "FC_TRAJ")

    harness.render_videos(tmp_path, echo=lambda _: None)

    assert (tmp_path / "run_0" / "FC_TRAJ.mp4").exists()
    assert (tmp_path / "run_0" / "FC_TRAJ").is_dir()


def test_off_writes_nothing(tmp_path) -> None:
    _frames(tmp_path / "run_0", "FC_TRAJ")
    assert _harness_with_video("off").render_videos(tmp_path, echo=lambda _: None) == []
    assert not list(tmp_path.rglob("*.mp4"))


def test_a_single_run_episode_has_no_run_dir_and_still_renders(tmp_path) -> None:
    """`--runs 1` writes straight into the episode dir; `_per_run` reads it the same way."""
    harness = _harness_with_video("encode")
    if harness._ffmpeg() is None:
        pytest.skip("no ffmpeg")
    _frames(tmp_path, "FC_TRAJ")

    written = harness.render_videos(tmp_path, echo=lambda _: None)

    assert [p.name for p in written] == ["FC_TRAJ.mp4"]


def test_a_missing_ffmpeg_keeps_the_frames_and_never_raises(tmp_path, monkeypatch) -> None:
    """An episode costs minutes of simulator time; a debugging aid must not lose it."""
    harness = _harness_with_video("encode")
    monkeypatch.setattr(type(harness), "_ffmpeg", lambda self: None)
    _frames(tmp_path / "run_0", "FC_TRAJ")

    said = []
    assert harness.render_videos(tmp_path, echo=said.append) == []

    assert (tmp_path / "run_0" / "FC_TRAJ").is_dir(), "frames must survive"
    assert said and "ffmpeg" in said[0]


# ---- impact / reference speed ------------------------------------------

# Exactly as the engine prints it, two spaces after each comma.
_STDOUT = """\
starting run 0
ncap_score: 1.7853428451189424,  impact_speed: 7.352228805237549,  reference_speed: 13.27921802981268
starting run 1
ncap_score: 2.458287437510784,  impact_speed: 5.118184314148877,  reference_speed: 13.27921802981268
"""


def _episode_with_stdout(tmp_path, scores, stdout=_STDOUT):
    for i, s in enumerate(scores):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"ncap_score": s, "any_collide@0.0s": True}))
    (tmp_path / "stdout.log").write_text(stdout)
    return tmp_path


def test_speeds_are_recovered_from_stdout(tmp_path) -> None:
    """The engine prints impact/reference speed and writes neither to metrics.json.

    They are the score's own inputs -- 5*max(0, 1 - v_impact/v_ref) -- so a row
    without them records a number nothing can be checked against. The legacy
    ingest recovers them from its logs; the live path has the same text in the
    stdout it already captures.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    per_run = NeuroNcapBenchmark._per_run(
        _episode_with_stdout(tmp_path, [1.7853428451189424, 2.458287437510784])
    )

    assert [m["impact_speed"] for _, m in per_run] == [7.352228805237549, 5.118184314148877]
    assert [m["reference_speed"] for _, m in per_run] == [13.27921802981268, 13.27921802981268]


def test_speeds_are_left_out_rather_than_guessed_when_the_join_is_unsafe(tmp_path) -> None:
    """A speed on the wrong run is worse than no speed.

    Positional matching holds only while the engine prints one line per run. If
    a run crashed or the log was truncated the counts disagree, and then there
    is no way to tell which line belongs to which run.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    one_line = "\n".join(_STDOUT.splitlines()[:2]) + "\n"
    per_run = NeuroNcapBenchmark._per_run(
        _episode_with_stdout(tmp_path, [1.7853428451189424, 2.458287437510784], stdout=one_line)
    )

    assert len(per_run) == 2
    assert all("impact_speed" not in m for _, m in per_run)
    assert [m["ncap_score"] for _, m in per_run] == [1.7853428451189424, 2.458287437510784]


def test_a_score_that_disagrees_leaves_that_run_alone(tmp_path) -> None:
    """The score appears at full precision in both places, so it validates the join."""
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    per_run = NeuroNcapBenchmark._per_run(_episode_with_stdout(tmp_path, [9.9, 2.458287437510784]))

    by_index = dict(per_run)
    assert "impact_speed" not in by_index[0], "mismatched score must not take the line's speeds"
    assert by_index[1]["impact_speed"] == 5.118184314148877


def test_a_score_only_line_still_matches(tmp_path) -> None:
    """Older logs print the score alone; the speed group is optional."""
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    per_run = NeuroNcapBenchmark._per_run(_episode_with_stdout(tmp_path, [5.0], stdout="ncap_score: 5.0\n"))

    assert [m["ncap_score"] for _, m in per_run] == [5.0]
    assert "impact_speed" not in per_run[0][1]


def test_a_partial_scope_is_stored_invalid(tmp_path) -> None:
    """A smoke on one scene must not average into the leaderboard as a sweep.

    This is the failure the whole adapter guards against, one level up: not a
    sweep that silently scores fewer pairs, but a deliberately narrow run that
    gets recorded wearing the full benchmark's name.
    """
    harness = _harness()
    list(harness.episodes(["0103"], runs=2, scenarios=["stationary", "frontal"]))
    reasons = harness.invalid_reasons()
    assert reasons and "2 of the 20 official" in reasons[0]


def test_the_full_official_set_is_valid() -> None:
    harness = _harness()
    list(harness.episodes(harness.available_scenes(), runs=10))
    assert harness.invalid_reasons() == []


def test_run_count_alone_does_not_invalidate() -> None:
    """10 runs of upstream's 50 is a seeded prefix -- a subsample, not a subset."""
    harness = _harness()
    list(harness.episodes(harness.available_scenes(), runs=1))
    assert harness.invalid_reasons() == []
