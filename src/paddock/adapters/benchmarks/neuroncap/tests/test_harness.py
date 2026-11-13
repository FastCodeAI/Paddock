"""The closed-loop harness, checked without a simulator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from paddock.adapters.benchmarks.neuroncap.harness import (
    DEFAULT_SCENARIOS,
    OFFICIAL_SET,
    SCENARIOS_DIR,
    WAYPOINT_DT,
    NeuroNcapBenchmark,
)
from paddock.config import ConfigError, SiteConfig
from paddock.registry import Registry


def _stub_ffmpeg(tmp_path: Path, fail: bool = False) -> Path:
    """A stand-in encoder, so the video tests do not need a real ffmpeg.

    What these tests check is the harness's part -- which frames it lists, that
    it writes and cleans up the concat file, that it prunes only a stream that
    encoded, that a failure keeps the frames. None of that is ffmpeg's
    behaviour, and requiring the binary meant the whole video pipeline had no
    coverage on CI or at another site. A real encode is verified separately.
    """
    stub = tmp_path / ("ffmpeg_fail" if fail else "ffmpeg_stub")
    stub.write_text(
        "#!/bin/sh\n" + ("exit 1\n" if fail else 'for a; do out="$a"; done\nprintf fake > "$out"\n')
    )
    stub.chmod(0o755)
    return stub


def _nuscenes_meta(tmp_path: Path, scenes: dict[str, int], present: set[str]) -> None:
    """A minimal nuScenes metadata tree plus whichever lidar files should exist.

    `scenes` maps scene name to its keyframe count; `present` names the scenes
    whose .pcd.bin files are actually written to disk.
    """
    meta = tmp_path / "nuscenes" / "v1.0-trainval"
    meta.mkdir(parents=True, exist_ok=True)
    lidar_dir = tmp_path / "nuscenes" / "samples" / "LIDAR_TOP"
    lidar_dir.mkdir(parents=True, exist_ok=True)

    scene_rows, sample_rows, sd_rows = [], [], []
    for name, count in scenes.items():
        tokens = [f"{name}-s{i}" for i in range(count)]
        scene_rows.append({"name": name, "first_sample_token": tokens[0]})
        for i, tok in enumerate(tokens):
            sample_rows.append({"token": tok, "next": tokens[i + 1] if i + 1 < count else ""})
            filename = f"samples/LIDAR_TOP/{name}__LIDAR_TOP__{i}.pcd.bin"
            sd_rows.append({"sample_token": tok, "is_key_frame": True, "filename": filename})
            if name in present:
                (tmp_path / "nuscenes" / filename).write_bytes(b"")
    (meta / "scene.json").write_text(json.dumps(scene_rows))
    (meta / "sample.json").write_text(json.dumps(sample_rows))
    (meta / "sample_data.json").write_text(json.dumps(sd_rows))


def _ncap_site(tmp_path: Path, monkeypatch, **settings) -> SiteConfig:
    """A neuroNCAP site built from a temp layout, so no test needs this machine.

    Most of this harness needs nothing from the site: `episodes` reads tracked
    scenario files, `episode_samples` takes an `out_dir`, `video_mode` reads one
    setting and `_encode` needs ffmpeg and a directory. Loading the real
    `paddock.toml` made all of that skip on CI and at another site, which left
    the video pipeline and the per-run sampling with no coverage anywhere but
    here.
    """
    for sub in ("repos", "runs", "weights", "nuscenes", "ncap-repo"):
        (tmp_path / sub).mkdir(exist_ok=True)
    (tmp_path / "ncap-repo" / "main.py").touch()
    # One checkpoint dir per official scene, so available_scenes() is the full set.
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    for scene in {s for v in OFFICIAL_SET.values() for s in v}:
        (ckpt / scene).mkdir(exist_ok=True)

    for scene in {s for v in OFFICIAL_SET.values() for s in v}:
        (ckpt / scene / "config.yml").write_text("")
    block = {
        "python": str(tmp_path / "python"),
        "repo": str(tmp_path / "ncap-repo"),
        "renderer_url": "http://127.0.0.1:8765",
        "checkpoints": str(ckpt),
        "renderer_python": str(tmp_path / "python"),
        "renderer_cwd": str(tmp_path),
        "neurad_studio": str(tmp_path / "studio"),
        **settings,
    }
    (tmp_path / "studio").mkdir(exist_ok=True)
    if "ffmpeg" not in block:
        block["ffmpeg"] = str(_stub_ffmpeg(tmp_path))
    (tmp_path / "python").touch()
    lines = "\n".join(f'{k} = "{v}"' for k, v in block.items())
    (tmp_path / "site.toml").write_text(
        f'site = "test-site"\n[roots]\nrepos = "{tmp_path / "repos"}"\n'
        f'runs = "{tmp_path / "runs"}"\nweights = "{tmp_path / "weights"}"\n'
        f'[datasets.nuscenes]\npath = "{tmp_path / "nuscenes"}"\nversion = "v1.0-trainval"\n'
        f"[benchmarks.neuroncap]\n{lines}\n"
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    return SiteConfig.load()


def _harness_for(cfg: SiteConfig):
    from paddock.adapters.benchmarks.neuroncap.harness import BENCHMARK

    return BENCHMARK(Registry.load().benchmarks["neuroncap"], cfg)


def _harness():
    """The real site, for the few checks that are genuinely about this machine."""
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


@pytest.mark.parametrize(
    ("dt", "refused"), [(0.5, False), (0.1, False), (0.25, False), (0.2, True), (1.0, True)]
)
def test_a_plan_spacing_off_the_half_second_grid_is_refused_before_loading(dt, refused) -> None:
    """0.5 s plans and Alpamayo 2 Edge's 0.1 s ones have points at every 0.5 s; 0.2 s has none."""
    model = SimpleNamespace(name="m", waypoint_dt=dt)
    problems = NeuroNcapBenchmark.input_problems(object.__new__(NeuroNcapBenchmark), model)
    assert bool(problems) is refused
    if refused:
        assert "must divide 0.5 s" in problems[0]


def test_the_runner_refuses_an_off_grid_plan_spacing_before_anything_starts(tmp_path, monkeypatch) -> None:
    """Asked by `paddock run` itself, dry run included, not only by the test above.

    The check once lived in `model_problems`, which the closed-loop runner never
    calls: a 0.2 s model would have started the renderer and loaded its
    checkpoint, then failed on its first plan.
    """
    from paddock import runner

    cfg = _ncap_site(tmp_path, monkeypatch)
    harness = _harness_for(cfg)
    model = SimpleNamespace(name="m", waypoint_dt=0.2, min_input_hz=None, cameras=[])
    lines: list[str] = []
    code = runner._run_harness(
        cfg, harness.spec, harness, [model], "neuroncap", "0103", None, 1, None, True, None, lines.append
    )
    assert code == 1
    assert any("must divide 0.5 s" in line for line in lines), lines


def test_episode_ids_are_stable_and_paired(tmp_path, monkeypatch) -> None:
    """Every model must see identical episodes, in a deterministic order."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    first = [e.episode_id for e in harness.episodes(["0103"], runs=1)]
    second = [e.episode_id for e in harness.episodes(["0103"], runs=1)]
    assert first == second
    assert first == ["stationary-0103", "frontal-0103", "side-0103"]


def test_a_scene_without_a_checkpoint_is_reported_before_running(tmp_path, monkeypatch) -> None:
    """This is what preflight is for: catching it in seconds, not an hour in."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))

    if not harness.setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    problems = harness.scene_problems("9999")
    assert problems and "no renderer checkpoint" in problems[0]


def test_available_scenes_intersects_scenarios_with_checkpoints(tmp_path, monkeypatch) -> None:
    """A scenario yaml without a renderer checkpoint cannot be rendered."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
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


def test_a_named_scenario_scene_pair_that_does_not_exist_is_an_error(tmp_path, monkeypatch) -> None:
    """Explicitly asked for, explicitly refused -- not silently dropped."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    with pytest.raises(FileNotFoundError, match="no scenario file"):
        list(harness.episodes(["0099"], runs=1, scenarios=["frontal"]))


def test_a_scene_outside_a_category_is_skipped_when_scenarios_default(tmp_path, monkeypatch) -> None:
    """0099 is a stationary-only scene; that is the benchmark's shape, not a gap."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
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


def test_every_run_reaches_the_results_store_separately(tmp_path, monkeypatch) -> None:
    """The spread is the point: a mean of 10 runs with no n and no sd is unreadable."""
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    for i, score in enumerate([0.0, 2.084, 5.0]):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"ncap_score": score, "any_collide@0.0s": score < 5.0}))

    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
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


def test_expected_sample_count_matches_what_a_sweep_records(tmp_path, monkeypatch) -> None:
    """`n_samples / n_expected` is the schema's completion ratio, so both count samples.

    neuroNCAP is the one benchmark where an episode is not one sample, so it is
    the one that breaks the ratio if `samples_per_episode` and `episode_samples`
    ever disagree about the factor.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
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


def _harness_with_video(mode, tmp_path, monkeypatch):
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch, video=mode))
    real = harness.setting

    def setting(key, default=None):
        return mode if key == "video" else real(key, default)

    harness.setting = setting
    return harness


def test_video_mode_defaults_to_encode_and_rejects_nonsense(tmp_path, monkeypatch) -> None:
    # A site that never set `video` gets encode.
    assert _harness_for(_ncap_site(tmp_path, monkeypatch)).video_mode() == "encode"
    with pytest.raises(ValueError, match="expected one of"):
        _harness_with_video("mp4", tmp_path, monkeypatch).video_mode()


def test_encode_writes_one_video_per_run_and_prunes_the_frames(tmp_path, monkeypatch) -> None:
    """The frames are reconstructible from CAM_* plus the json, so pruning is safe.

    83 MB per run is 66 GB across a 20-pair sweep at 10 runs for four models;
    the same content as video is about 5 GB.
    """
    harness = _harness_with_video("encode", tmp_path, monkeypatch)
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


def test_keep_encodes_without_deleting_anything(tmp_path, monkeypatch) -> None:
    harness = _harness_with_video("keep", tmp_path, monkeypatch)
    if harness._ffmpeg() is None:
        pytest.skip("no ffmpeg")
    _frames(tmp_path / "run_0", "FC_TRAJ")

    harness.render_videos(tmp_path, echo=lambda _: None)

    assert (tmp_path / "run_0" / "FC_TRAJ.mp4").exists()
    assert (tmp_path / "run_0" / "FC_TRAJ").is_dir()


def test_off_writes_nothing(tmp_path, monkeypatch) -> None:
    _frames(tmp_path / "run_0", "FC_TRAJ")
    assert (
        _harness_with_video("off", tmp_path, monkeypatch).render_videos(tmp_path, echo=lambda _: None) == []
    )
    assert not list(tmp_path.rglob("*.mp4"))


def test_a_single_run_episode_has_no_run_dir_and_still_renders(tmp_path, monkeypatch) -> None:
    """`--runs 1` writes straight into the episode dir; `_per_run` reads it the same way."""
    harness = _harness_with_video("encode", tmp_path, monkeypatch)
    if harness._ffmpeg() is None:
        pytest.skip("no ffmpeg")
    _frames(tmp_path, "FC_TRAJ")

    written = harness.render_videos(tmp_path, echo=lambda _: None)

    assert [p.name for p in written] == ["FC_TRAJ.mp4"]


def test_a_failing_encoder_keeps_the_frames(tmp_path, monkeypatch) -> None:
    """A non-zero ffmpeg must not prune: frames with no video are the worst case."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch, ffmpeg=str(_stub_ffmpeg(tmp_path, fail=True))))
    _frames(tmp_path / "run_0", "FC_TRAJ")

    said = []
    assert harness.render_videos(tmp_path, echo=said.append) == []

    assert (tmp_path / "run_0" / "FC_TRAJ").is_dir(), "frames must survive a failed encode"
    assert not (tmp_path / "run_0" / "FC_TRAJ.mp4").exists()
    assert said and "failed" in said[0]
    assert not list(tmp_path.rglob("*.concat")), "the concat list must be cleaned up"


@pytest.mark.parametrize("interpreter", ["unreachable", "missing"])
def test_an_ffmpeg_that_cannot_start_keeps_the_frames_and_never_raises(
    tmp_path, monkeypatch, interpreter
) -> None:
    """Run as the gate queue's account, the site's ffmpeg was a script whose Python that
    account could not reach: PermissionError at start, and the finished episode was lost."""
    python = tmp_path / "python"
    if interpreter == "unreachable":
        python.write_text("")
        python.chmod(0o644)  # not executable, even by root
    wrapper = tmp_path / "ffmpeg"
    wrapper.write_text(f"#!{python}\n")
    wrapper.chmod(0o755)
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch, ffmpeg=str(wrapper)))
    _frames(tmp_path / "run_0", "FC_TRAJ")

    said = []
    assert harness.render_videos(tmp_path, echo=said.append) == []

    assert (tmp_path / "run_0" / "FC_TRAJ").is_dir(), "frames must survive"
    assert said and "cannot run" in said[0]
    assert not list(tmp_path.rglob("*.concat")), "the concat list must be cleaned up"


def test_a_missing_ffmpeg_keeps_the_frames_and_never_raises(tmp_path, monkeypatch) -> None:
    """An episode costs minutes of simulator time; a debugging aid must not lose it."""
    harness = _harness_with_video("encode", tmp_path, monkeypatch)
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


def test_speeds_are_recovered_from_stdout(tmp_path, monkeypatch) -> None:
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


def test_a_partial_scope_is_stored_invalid(tmp_path, monkeypatch) -> None:
    """A smoke on one scene must not average into the leaderboard as a sweep.

    This is the failure the whole adapter guards against, one level up: not a
    sweep that silently scores fewer pairs, but a deliberately narrow run that
    gets recorded wearing the full benchmark's name.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    list(harness.episodes(["0103"], runs=2, scenarios=["stationary", "frontal"]))
    reasons = harness.invalid_reasons()
    assert reasons and "2 of the 20 official" in reasons[0]


def test_the_full_official_set_is_valid(tmp_path, monkeypatch) -> None:
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    list(harness.episodes(harness.available_scenes(), runs=10))
    assert harness.invalid_reasons() == []


def test_run_count_alone_does_not_invalidate(tmp_path, monkeypatch) -> None:
    """10 runs of upstream's 50 is a seeded prefix -- a subsample, not a subset."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    list(harness.episodes(harness.available_scenes(), runs=1))
    assert harness.invalid_reasons() == []


# ---- the built command line ---------------------------------------------


def test_run_episode_passes_allowed_cams_none(tmp_path, monkeypatch) -> None:
    """`--engine.logger.allowed-cams None` must reach the subprocess, not just the comment above it.

    The vendor logger's own default keeps only CAM_FRONT; without this flag a
    multi-camera model's closed-loop run would silently lose every other
    camera's frames. Nothing previously inspected the built command line, so a
    regression here had no test to break.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    scene = OFFICIAL_SET["stationary"][0]
    episode = next(iter(harness.episodes([scene], runs=1, scenarios=["stationary"])))
    episode.config["cameras"] = ["CAM_FRONT"]

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    harness.run_episode(episode, "http://127.0.0.1:9000", tmp_path / "out")

    command = captured["command"]
    assert "--engine.logger.allowed-cams" in command
    idx = command.index("--engine.logger.allowed-cams")
    assert command[idx + 1] == "None"


# ---- renderer management -----------------------------------------------


def test_renderer_management_is_off_by_default(tmp_path, monkeypatch) -> None:
    """The single-scene workflow keeps a renderer alive across runs on purpose."""
    assert _harness_for(_ncap_site(tmp_path, monkeypatch)).manages_renderer() is False


def test_managing_renderers_makes_the_whole_official_set_one_valid_run(tmp_path, monkeypatch) -> None:
    """This is the reason the switch exists.

    The renderer cannot change scene, so driving 14 scenes from outside means one
    `paddock run` per scene -- and then every run covers 1-3 of the 20 pairs and
    is correctly recorded `valid: false`. 28 partial runs, no rankable score.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    list(harness.episodes(harness.available_scenes(), runs=10))
    assert harness.invalid_reasons() == []

    per_scene = _harness_for(_ncap_site(tmp_path, monkeypatch))
    list(per_scene.episodes(["0103"], runs=10))
    assert per_scene.invalid_reasons(), "a single-scene run must not rank as a sweep"


def test_close_stops_the_renderer_this_harness_started(tmp_path, monkeypatch) -> None:
    """A sweep split by a pairing builds a harness per model; each must let its renderer go."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    calls: list[str] = []

    class _Renderer:
        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout=None) -> int:
            calls.append("wait")
            return 0

    harness._renderer_proc, harness._renderer_scene = _Renderer(), "0103"
    harness.close()
    assert calls == ["terminate", "wait"]
    assert harness._renderer_proc is None and harness._renderer_scene is None
    harness.close()  # nothing left: a second close touches nothing
    assert calls == ["terminate", "wait"]


def test_a_renderer_we_did_not_start_is_never_stopped(tmp_path, monkeypatch) -> None:
    """On a shared box that renderer may be another person's sweep."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    harness._renderer_proc = None
    harness._stop_renderer(lambda _: None)  # must not raise, must not touch anything
    assert harness._renderer_proc is None


def test_ensure_renderer_is_a_no_op_when_unmanaged(monkeypatch, tmp_path) -> None:
    """Without the switch, run_episode must not try to start anything."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    called = []
    monkeypatch.setattr(type(harness), "_stop_renderer", lambda self, echo=print: called.append(1))
    harness._ensure_renderer("0106", echo=lambda _: None)
    assert called == []


def test_this_site_can_render_every_official_scene() -> None:
    """The real site, deliberately: a sweep must not die on scene 9 of 14.

    Tautological against a temp layout, which is why this one still loads the
    machine's own config and skips where there is none.
    """
    from paddock.adapters.benchmarks.neuroncap import launcher

    harness = _harness()
    if not harness.setting("checkpoints"):
        pytest.skip("no checkpoints configured")
    for scene in sorted({s for v in OFFICIAL_SET.values() for s in v}):
        command, cwd, env = launcher.renderer_command(harness.cfg, scene, 8000)
        assert command and cwd and "NEURAD_STUDIO" in env


# ---- usage errors reach the user, before the model loads ---------------


def test_a_named_pair_that_does_not_exist_is_refused_by_preflight(tmp_path, monkeypatch) -> None:
    """`episodes()` raising is too late and the wrong type.

    It is called inside the `model_node` block, so the model is already resident;
    and the runner catches `RuntimeError`, which `FileNotFoundError` is not, so
    the carefully written message arrives as a traceback. preflight runs first.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    problems = harness.scenario_problems(["frontal"], ["0099"])
    assert problems and "no scenario file for frontal/0099" in problems[0]
    assert "stationary-only" in problems[0]

    assert harness.scenario_problems(["stationary"], ["0099"]) == []
    assert harness.scenario_problems(["frontal"], ["0103"]) == []


def test_an_unknown_scenario_name_is_still_refused(tmp_path, monkeypatch) -> None:
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    problems = harness.scenario_problems(["sideways"], ["0103"])
    assert problems and "no such scenario set" in problems[0]


def test_a_bad_video_setting_is_refused_by_preflight(tmp_path, monkeypatch) -> None:
    """Otherwise a typo raises ValueError from the first episode, after the node is up.

    The runner catches `RuntimeError`, not `ValueError`, so that is a traceback
    minutes into a sweep instead of a refusal in seconds.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch, video="encoded"))
    assert any("expected one of" in p for p in harness.preflight())


def test_episode_output_goes_to_the_runs_own_stream(tmp_path, monkeypatch) -> None:
    """A detached sweep's log should hold the whole story, not just stdout."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    monkeypatch.setattr(type(harness), "_ffmpeg", lambda self: None)
    _frames(tmp_path / "run_0", "FC_TRAJ")

    said = []
    harness.render_videos(tmp_path, echo=said.append)
    assert said and "ffmpeg" in said[0], "must reach the caller's echo, not print"


# ---- resume -------------------------------------------------------------


def _episode_dir(tmp_path, n_runs, scenario="frontal", scene="0103"):
    d = tmp_path / f"{scenario}-{scene}"
    for i in range(n_runs):
        run = d / f"run_{i}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(json.dumps({"ncap_score": 2.0 + i, "any_collide@0.0s": True}))
    return d


def test_a_fully_scored_episode_is_reusable(tmp_path, monkeypatch) -> None:
    """The point of resume: simulator time already spent is not spent again."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    episode = next(iter(harness.episodes(["0103"], runs=3, scenarios=["frontal"])))
    out = _episode_dir(tmp_path, 3)

    metrics = harness.completed_metrics(episode, out, runs=3)
    assert metrics is not None
    assert metrics["n_runs"] == 3
    assert metrics["ncap_score"] == pytest.approx(3.0)


def test_a_half_scored_episode_is_not_reused(tmp_path, monkeypatch) -> None:
    """The failure that matters. An interrupted episode has some runs on disk.

    Counting those as the whole episode would make a mean over however many
    happened to finish -- a wrong number, not a slow one. It has to be re-driven.
    """
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    episode = next(iter(harness.episodes(["0103"], runs=10, scenarios=["frontal"])))
    out = _episode_dir(tmp_path, 4)

    assert harness.completed_metrics(episode, out, runs=10) is None


def test_an_episode_never_started_is_not_reused(tmp_path, monkeypatch) -> None:
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    episode = next(iter(harness.episodes(["0103"], runs=3, scenarios=["frontal"])))
    assert harness.completed_metrics(episode, tmp_path / "nothing-here", runs=3) is None


def test_more_runs_on_disk_than_asked_for_is_still_reusable(tmp_path, monkeypatch) -> None:
    """Resuming a 10-run sweep as a 3-run one: the first three are there."""
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    episode = next(iter(harness.episodes(["0103"], runs=3, scenarios=["frontal"])))
    assert harness.completed_metrics(episode, _episode_dir(tmp_path, 10), runs=3) is not None


def test_benchmarks_that_cannot_tell_opt_out_by_default() -> None:
    """Default None: resume re-runs everything until a benchmark opts in.

    Half an episode counted as whole is worse than an episode driven twice.
    """
    from paddock.benchmark import ClosedLoopBenchmark
    from paddock.benchmark import Episode as Ep

    ep = Ep(episode_id="x", scenario="frontal", scene="0103")
    assert ClosedLoopBenchmark.completed_metrics(object(), ep, Path("/nowhere"), 5) is None


def test_a_metrics_file_killed_mid_write_is_skipped_not_fatal(tmp_path, monkeypatch) -> None:
    """The exact state a SIGKILL, an OOM kill or a reboot leaves behind.

    An unguarded `json.loads` made `--resume` die on the first episode it
    inspected -- and, because `_per_run` also backs `_collect` and
    `episode_samples`, made a *fresh* run into a directory holding one stale
    truncated file fail the same way. Skipping the file counts the run as
    missing, so the episode is driven again, which is both the safe answer and
    the correct one.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    (tmp_path / "run_0").mkdir()
    (tmp_path / "run_0" / "metrics.json").write_text(json.dumps({"ncap_score": 5.0}))
    (tmp_path / "run_1").mkdir()
    (tmp_path / "run_1" / "metrics.json").write_text('{"ncap_score": 4.')

    per_run = NeuroNcapBenchmark._per_run(tmp_path)
    assert [i for i, _ in per_run] == [0], "the truncated run must be skipped, not raise"

    episode = next(iter(harness.episodes(["0103"], runs=2, scenarios=["frontal"])))
    assert harness.completed_metrics(episode, tmp_path, runs=2) is None, "so the episode re-runs"

    # A fresh run reading the same directory must not die either.
    assert NeuroNcapBenchmark._collect(tmp_path)["n_runs"] == 1
    assert len(harness.episode_samples(episode, {}, tmp_path)) == 1


def test_an_unreadable_metrics_file_is_also_survivable(tmp_path, monkeypatch) -> None:
    """A directory where a file should be: same class of damage, same answer."""
    from paddock.adapters.benchmarks.neuroncap.harness import NeuroNcapBenchmark

    (tmp_path / "run_0").mkdir()
    (tmp_path / "run_0" / "metrics.json").mkdir()
    assert NeuroNcapBenchmark._per_run(tmp_path) == []


# ---- a scene with no lidar is refused before anything loads --------------


def test_a_scene_with_no_lidar_keyframes_is_refused(tmp_path, monkeypatch) -> None:
    """The check that would have saved the 2026-09-04 sweep thirteen GPU-hours.

    The renderer never uses the points, but NeuRAD's dataparser reads them
    eagerly and `renderer.py` needs one real cloud to copy a tensor width from.
    With none it raises; with that raise removed the dataparser falls over later
    instead, in `_get_world_to_mean_transform`. Either way the scene cannot run,
    and finding that out mid-sweep costs hours.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_keyframes_by_scene

    _lidar_keyframes_by_scene.cache_clear()
    _nuscenes_meta(tmp_path, {"scene-0796": 40, "scene-0103": 40}, present={"scene-0103"})
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))

    problems = harness.scene_problems("0796")
    assert problems, "a scene with no lidar must not reach the renderer"
    assert "none of its 40 LIDAR_TOP keyframes" in problems[0]

    assert harness.scene_problems("0103") == [], "a scene with its lidar must pass"


def test_a_scene_missing_only_some_lidar_still_runs(tmp_path, monkeypatch) -> None:
    """Partial lidar is the normal case here, not an error.

    This nuScenes copy has no `sweeps/` at all, so every scene is already
    rendering with thousands of empty stand-ins. One real cloud is enough.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_keyframes_by_scene

    _lidar_keyframes_by_scene.cache_clear()
    _nuscenes_meta(tmp_path, {"scene-0103": 40}, present={"scene-0103"})
    # Delete all but one, leaving the single cloud the width is copied from.
    kept = sorted((tmp_path / "nuscenes" / "samples" / "LIDAR_TOP").glob("*.pcd.bin"))
    for f in kept[1:]:
        f.unlink()

    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    assert harness.scene_problems("0103") == []


def test_an_unknown_scene_is_not_refused_for_lidar(tmp_path, monkeypatch) -> None:
    """A scene nuScenes has never heard of fails the checkpoint check, not this one.

    Two messages for one problem sends the reader looking in the wrong place.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_keyframes_by_scene

    _lidar_keyframes_by_scene.cache_clear()
    _nuscenes_meta(tmp_path, {"scene-0103": 40}, present={"scene-0103"})
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))

    problems = harness.scene_problems("9999")
    assert problems and "no renderer checkpoint" in problems[0]
    assert not any("LIDAR" in p for p in problems)


def test_unreadable_metadata_does_not_fail_preflight(tmp_path, monkeypatch) -> None:
    """Preflight refuses what it knows is broken; it does not invent reasons.

    A truncated sample_data.json says nothing about whether a scene has lidar,
    and blocking a sweep on an unparseable file the run may not even need is
    worse than letting the run report the real problem itself.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_keyframes_by_scene

    _lidar_keyframes_by_scene.cache_clear()
    _nuscenes_meta(tmp_path, {"scene-0103": 40}, present={"scene-0103"})
    (tmp_path / "nuscenes" / "v1.0-trainval" / "sample_data.json").write_text('[{"broken"')

    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    assert harness.scene_problems("0103") == []


def test_the_streaming_parse_matches_a_whole_file_parse(tmp_path, monkeypatch) -> None:
    """The scan replaced `json.loads` on a 1.25 GB file; it must give the same answer.

    Reading the file whole peaked at 4.6 GB of RSS, which is an absurd cost for
    the check that runs before everything else. The scan relies on nuScenes
    writing these files pretty-printed, one field per line, so this pins that
    the two agree on a file shaped exactly like the real one.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_sample_data

    rows = [
        {
            "token": "a",
            "sample_token": "s1",
            "is_key_frame": True,
            "filename": "samples/LIDAR_TOP/x__LIDAR_TOP__1.pcd.bin",
        },
        # a sweep, not a keyframe -- must be skipped
        {
            "token": "b",
            "sample_token": "s2",
            "is_key_frame": False,
            "filename": "sweeps/LIDAR_TOP/x__LIDAR_TOP__2.pcd.bin",
        },
        # a camera keyframe -- must be skipped
        {
            "token": "c",
            "sample_token": "s3",
            "is_key_frame": True,
            "filename": "samples/CAM_FRONT/x__CAM_FRONT__3.jpg",
        },
        # a keyframe lidar file for a second sample
        {
            "token": "d",
            "sample_token": "s4",
            "is_key_frame": True,
            "filename": "samples/LIDAR_TOP/x__LIDAR_TOP__4.pcd.bin",
        },
    ]
    path = tmp_path / "sample_data.json"
    path.write_text(json.dumps(rows, indent=1))

    expected = {
        d["sample_token"]: d["filename"]
        for d in json.loads(path.read_text())
        if d["is_key_frame"] and d["filename"].startswith("samples/LIDAR_TOP")
    }
    assert _lidar_sample_data(path) == expected
    assert _lidar_sample_data(path) == {
        "s1": "samples/LIDAR_TOP/x__LIDAR_TOP__1.pcd.bin",
        "s4": "samples/LIDAR_TOP/x__LIDAR_TOP__4.pcd.bin",
    }


def test_preflight_checks_the_default_scene_when_no_scene_is_named(tmp_path, monkeypatch) -> None:
    """A run without --scene falls back to the default, which was never checked.

    `scene_problems` only sees what --scene names, so the one scene a bare
    `paddock run` will actually drive was the one scene preflight skipped.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import (
        DEFAULT_SCENE,
        _lidar_keyframes_by_scene,
    )

    _lidar_keyframes_by_scene.cache_clear()
    _nuscenes_meta(tmp_path, {f"scene-{DEFAULT_SCENE}": 40}, present=set())
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))

    assert any("LIDAR_TOP keyframes" in p for p in harness.preflight())


def test_a_compact_metadata_file_falls_back_rather_than_reporting_nothing(tmp_path) -> None:
    """A single-line file yields nothing from the scan, and nothing means "no lidar".

    Left there, preflight would refuse every scene on a copy whose metadata
    happens to be compact -- confidently and wrongly. Being wrong in the
    blocking direction is worse than being slow, so it falls back to a whole
    read. Caught by the fixture in this file writing compact JSON.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_sample_data

    rows = [
        {
            "token": "a",
            "sample_token": "s1",
            "is_key_frame": True,
            "filename": "samples/LIDAR_TOP/x__LIDAR_TOP__1.pcd.bin",
        },
        {
            "token": "b",
            "sample_token": "s2",
            "is_key_frame": True,
            "filename": "samples/CAM_FRONT/x__CAM_FRONT__2.jpg",
        },
    ]
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(rows))  # no indent: one line
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(rows, indent=1))

    want = {"s1": "samples/LIDAR_TOP/x__LIDAR_TOP__1.pcd.bin"}
    assert _lidar_sample_data(pretty) == want, "the fast path"
    assert _lidar_sample_data(compact) == want, "the fallback"


def test_an_empty_metadata_file_is_not_a_fallback_loop(tmp_path) -> None:
    """An empty file legitimately has no lidar; it must not try to json.loads it."""
    from paddock.adapters.benchmarks.neuroncap.harness import _lidar_sample_data

    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert _lidar_sample_data(empty) == {}


def test_a_pair_list_keeps_only_its_pairs_of_the_named_scenes(tmp_path, monkeypatch) -> None:
    """The gate's subsets are pairs spread over scenes; `--scene` alone runs every shape a scene has."""
    from paddock.adapters.benchmarks.neuroncap.harness import ENV_PAIRS

    harness = _harness_for(_ncap_site(tmp_path, monkeypatch))
    monkeypatch.setenv(ENV_PAIRS, "side-0103,stationary-0099")
    assert [e.episode_id for e in harness.episodes(["0099", "0103"], runs=1)] == [
        "stationary-0099",
        "side-0103",
    ]
    assert harness.invalid_reasons(), "a pair subset is not a neuroNCAP score"
    assert harness.run_config() == {"pairs": ["side-0103", "stationary-0099"]}
    monkeypatch.setenv(ENV_PAIRS, "frontal-0099")
    with pytest.raises(ValueError, match="not official scenario-scene pairs: frontal-0099"):
        harness.run_config()
    monkeypatch.delenv(ENV_PAIRS)
    assert harness.run_config() == {}
    assert [e.episode_id for e in harness.episodes(["0103"], runs=1)] == [
        "stationary-0103",
        "frontal-0103",
        "side-0103",
    ]


# ---- the cut-down nuScenes and the optional debug views (gate speed, 2026-10-27) ----


def _engine_command(harness, tmp_path, monkeypatch) -> list[str]:
    scene = OFFICIAL_SET["stationary"][0]
    episode = next(iter(harness.episodes([scene], runs=1, scenarios=["stationary"])))
    episode.config["cameras"] = ["CAM_FRONT"]
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    harness.run_episode(episode, "http://127.0.0.1:9000", tmp_path / "out")
    return captured["command"]


def _tiny_nuscenes(root: Path, scenes: list[str]) -> None:
    """Two-sample scenes with a sweep each, an annotation and an instance: enough for every link."""
    t: dict[str, list[dict]] = {k: [] for k in (
        "scene", "log", "map", "sample", "sample_data", "ego_pose", "calibrated_sensor",
        "sample_annotation", "instance", "category", "attribute", "visibility", "sensor",
    )}  # fmt: skip
    t["sensor"] = [{"token": "sen", "channel": "LIDAR_TOP", "modality": "lidar"}]
    t["category"] = [{"token": "cat", "name": "vehicle.car"}]
    t["attribute"] = [{"token": "att", "name": "vehicle.moving"}]
    t["visibility"] = [{"token": "vis", "level": "v80-100"}]
    for name in scenes:
        n = name[-4:]
        t["log"].append({"token": f"log{n}", "location": "boston-seaport"})
        t["map"].append({"token": f"map{n}", "log_tokens": [f"log{n}"], "filename": f"maps/{n}.png"})
        s0, s1 = f"s{n}a", f"s{n}b"
        t["scene"].append({"token": f"sc{n}", "name": name, "log_token": f"log{n}",
                           "first_sample_token": s0, "last_sample_token": s1})  # fmt: skip
        t["sample"] += [
            {"token": s0, "scene_token": f"sc{n}", "prev": "", "next": s1},
            {"token": s1, "scene_token": f"sc{n}", "prev": s0, "next": ""},
        ]
        t["calibrated_sensor"].append({"token": f"cs{n}", "sensor_token": "sen"})
        for i, (sample, sd) in enumerate([(s0, f"sd{n}a"), (s0, f"sd{n}sweep"), (s1, f"sd{n}b")]):
            prev = ["", f"sd{n}a", f"sd{n}sweep"][i]
            nxt = [f"sd{n}sweep", f"sd{n}b", ""][i]
            t["ego_pose"].append({"token": f"ep{sd}"})
            t["sample_data"].append(
                {
                    "token": sd,
                    "sample_token": sample,
                    "ego_pose_token": f"ep{sd}",
                    "calibrated_sensor_token": f"cs{n}",
                    "prev": prev,
                    "next": nxt,
                    "filename": f"samples/LIDAR_TOP/{sd}.pcd.bin",
                }
            )
        first_last = {"first_annotation_token": f"an{n}", "last_annotation_token": f"an{n}"}
        t["instance"].append({"token": f"in{n}", "category_token": "cat", **first_last})
        t["sample_annotation"].append({"token": f"an{n}", "sample_token": s0, "instance_token": f"in{n}",
                                       "visibility_token": "vis", "attribute_tokens": ["att"],
                                       "prev": "", "next": ""})  # fmt: skip
    (root / "v1.0-trainval").mkdir(parents=True, exist_ok=True)
    for name, rows in t.items():
        (root / "v1.0-trainval" / f"{name}.json").write_text(json.dumps(rows))
    for sub in ("samples", "sweeps", "maps", "can_bus"):
        (root / sub).mkdir(exist_ok=True)


def test_the_cut_down_nuscenes_keeps_every_record_its_scenes_use_and_nothing_else(tmp_path) -> None:
    """The engine and the renderer loaded all 2.5 GB of v1.0-trainval to read one scene."""
    from paddock.adapters.benchmarks.neuroncap import nuscenes_subset

    full, out = tmp_path / "full", tmp_path / "cut"
    _tiny_nuscenes(full, ["scene-0103", "scene-0200"])
    manifest = nuscenes_subset.build(full, out, "v1.0-trainval", {"scene-0103"})
    tables = {p.stem: json.loads(p.read_text()) for p in (out / "v1.0-trainval").glob("*.json")}
    assert nuscenes_subset.dangling(tables) == []
    assert [s["name"] for s in tables["scene"]] == ["scene-0103"]
    assert {r["token"] for r in tables["sample_data"]} == {"sd0103a", "sd0103sweep", "sd0103b"}, "sweeps too"
    assert [m["log_tokens"] for m in tables["map"]] == [["log0103"]]
    assert manifest["rows"]["sample"] == 2 and nuscenes_subset.scenes_in(out, "v1.0-trainval") == {
        "scene-0103"
    }
    for sub in ("samples", "sweeps", "maps", "can_bus"):
        assert (out / sub).is_symlink() and (out / sub).resolve() == (full / sub).resolve()
    del tables["ego_pose"][0]
    assert any("ego_pose" in p for p in nuscenes_subset.dangling(tables)), "a missing record is caught"
    with pytest.raises(ValueError, match="scene-9999"):
        nuscenes_subset.build(full, tmp_path / "x", "v1.0-trainval", {"scene-9999"})


def test_the_engine_and_the_renderer_read_the_cut_down_nuscenes_when_set(tmp_path, monkeypatch) -> None:
    from paddock.adapters.benchmarks.neuroncap import launcher, nuscenes_subset

    cut = tmp_path / "cut"
    _tiny_nuscenes(tmp_path / "full", ["scene-0099", "scene-0103"])
    nuscenes_subset.build(tmp_path / "full", cut, "v1.0-trainval", {"scene-0099"})
    plain = _harness_for(_ncap_site(tmp_path, monkeypatch))
    command = _engine_command(plain, tmp_path, monkeypatch)
    assert command[command.index("--engine.dataset.data_root") + 1] == str(tmp_path / "nuscenes")
    config_before = plain.run_config()

    cfg = _ncap_site(tmp_path, monkeypatch, nuscenes_subset=str(cut))
    harness = _harness_for(cfg)
    command = _engine_command(harness, tmp_path, monkeypatch)
    assert command[command.index("--engine.dataset.data_root") + 1] == str(cut)
    assert harness.run_config() == config_before, "not in the run config: runs still pair"
    _, _, env = launcher.renderer_command(cfg, "0099")
    assert env[launcher.NUSCENES_SUBSET_ENV] == str(cut)
    assert not any("nuscenes_subset" in p for p in harness.scene_problems("0099"))
    assert "is not in [benchmarks.neuroncap] nuscenes_subset" in " ".join(harness.scene_problems("0103"))


def test_video_off_turns_the_logger_off_in_the_form_the_engine_accepts(tmp_path, monkeypatch) -> None:
    """`--engine.logger.enabled False` is refused by neuro-ncap's command
    line ("Unrecognized arguments: False", exit 2), so `video = "off"` failed every episode."""
    off = _harness_for(_ncap_site(tmp_path, monkeypatch, video="off"))
    command = _engine_command(off, tmp_path, monkeypatch)
    assert "--engine.logger.no-enabled" in command
    assert "--engine.logger.enabled" not in command and "False" not in command


def test_the_debug_views_stay_on_unless_the_site_turns_them_off(tmp_path, monkeypatch) -> None:
    """FC_TRAJ and COMBINED_OUTPUTS took about 1 s of each 2.7 s step; the gate turns them off."""
    default = _harness_for(_ncap_site(tmp_path, monkeypatch))
    assert "--engine.logger.no-debug-views" not in _engine_command(default, tmp_path, monkeypatch)
    config_before = default.run_config()
    off = _harness_for(_ncap_site(tmp_path, monkeypatch, debug_views="false"))
    command = _engine_command(off, tmp_path, monkeypatch)
    # tyro's switch form; "--engine.logger.debug-views False" is refused (exit 2).
    assert "--engine.logger.no-debug-views" in command and "False" not in command
    assert "--engine.logger.enabled" not in command, "the records are still written"
    assert off.run_config() == config_before


def test_debug_views_off_needs_the_patched_neuro_ncap(tmp_path, monkeypatch) -> None:
    logger = tmp_path / "ncap-repo" / "neuro_ncap" / "components" / "logger.py"
    logger.parent.mkdir(parents=True)
    logger.write_text("class LoggerConfig:\n    enabled: bool = True\n")
    harness = _harness_for(_ncap_site(tmp_path, monkeypatch, debug_views="false"))
    assert any("0003-optional-debug-views" in p for p in harness.preflight())
    logger.write_text("class LoggerConfig:\n    debug_views: bool = True\n")
    assert not any("0003-optional-debug-views" in p for p in harness.preflight())
