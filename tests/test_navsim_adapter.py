"""The NAVSIM adapter, checked without NAVSIM.

Everything here runs in paddock's own environment. The conversion between
NAVSIM's conventions and paddock's lives in `convert.py` precisely so it can be
pinned here rather than only being exercised by a 40-minute scoring run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.navsim import convert
from paddock.adapters.benchmarks.navsim.harness import (
    CONF_DIR,
    ENV_AGENT,
    ENV_ALLOW_PARTIAL,
    ENV_ALLOW_SLOW,
    ENV_OVERRIDES,
    PROXY_AGENT,
    SCORER,
    SPLITS_DIR,
    input_rate_problems,
    limit_problems,
)
from paddock.config import ConfigError, SiteConfig
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, CANBUS_YAW_RATE, Command
from paddock.registry import Registry

GRID = convert.NAVSIM_INTERVAL_S * np.arange(1, convert.NAVSIM_NUM_POSES + 1)


def _straight(n: int, dt: float, speed: float = 10.0) -> np.ndarray:
    """A straight-ahead plan at constant speed, n waypoints spaced dt apart."""
    t = dt * np.arange(1, n + 1)
    return np.stack([speed * t, np.zeros(n)], axis=1)


# -- the planning grid -----------------------------------------------------


def test_output_is_always_navsims_eight_poses_at_half_a_second() -> None:
    """The scorer indexes the trajectory positionally; 8x0.5 s is not negotiable."""
    for n, dt in ((8, 0.5), (40, 0.1), (64, 0.1), (16, 0.25)):
        poses = convert.to_navsim_poses(_straight(n, dt), dt)
        assert poses.shape == (8, 3), f"{n} waypoints at {dt}s"


def test_a_constant_speed_plan_resamples_to_the_same_positions() -> None:
    """Resampling must not move a waypoint the model actually planned.

    Alpamayo emits 64 poses at 0.1 s and openpilot 8 at 0.5 s; both describe the
    same motion and must land on the same grid.
    """
    dense = convert.to_navsim_poses(_straight(64, 0.1), 0.1)
    sparse = convert.to_navsim_poses(_straight(8, 0.5), 0.5)
    assert np.allclose(dense[:, :2], sparse[:, :2])
    assert np.allclose(dense[:, 0], 10.0 * GRID)


def test_a_short_horizon_is_refused_rather_than_extrapolated() -> None:
    """VisionPilot plans 3 s. Padding it would score a plan it never made.

    This is the one that must fail loudly: a silently extrapolated fourth second
    would look like a real prediction and be ranked against models that made one.
    """
    with pytest.raises(ValueError, match="extend the model's horizon"):
        convert.to_navsim_poses(_straight(6, 0.5), 0.5)


def test_a_model_planning_past_four_seconds_is_truncated_not_refused() -> None:
    """openpilot's 10 s output is fine; only a short horizon is a problem."""
    poses = convert.to_navsim_poses(_straight(20, 0.5), 0.5)
    assert poses.shape == (8, 3)
    assert poses[-1, 0] == pytest.approx(40.0)


def test_the_origin_anchors_interpolation() -> None:
    """t=0 is exact -- the ego is at its own origin -- so odd spacings still work.

    Without the anchor, np.interp would clamp the first target to the model's
    first waypoint whenever dt > 0.5, quietly reporting a 0.5 s position the
    model placed at 0.8 s.
    """
    poses = convert.to_navsim_poses(_straight(5, 0.8), 0.8)
    assert poses[0, 0] == pytest.approx(5.0)  # 10 m/s at 0.5 s


# -- heading ---------------------------------------------------------------


def test_a_native_heading_survives_untouched() -> None:
    """Drive-JEPA predicts yaw. Nothing should be recomputed for it."""
    xy = _straight(8, 0.5)
    headings = np.linspace(0.0, 0.4, 8)
    poses = convert.to_navsim_poses(np.column_stack([xy, headings]), 0.5)
    assert np.allclose(poses[:, 2], headings)


def test_heading_is_derived_from_the_chord_when_a_model_has_none() -> None:
    """PDM orients the ego footprint by heading when it tests collisions."""
    turning = np.stack([np.arange(1, 9) * 5.0, np.arange(1, 9) * 5.0], axis=1)
    poses = convert.to_navsim_poses(turning, 0.5)
    assert np.allclose(poses[:, 2], np.pi / 4)


def test_a_stopped_ego_keeps_a_stable_heading() -> None:
    """Below a millimetre the chord is noise, and a random yaw fails checks it should pass."""
    poses = convert.to_navsim_poses(np.zeros((8, 2)), 0.5)
    assert np.allclose(poses[:, 2], 0.0)


def test_heading_interpolation_does_not_cross_the_pi_seam() -> None:
    """Unwrapped before interpolating, or a heading near pi averages to ~0."""
    xy = _straight(8, 0.5)
    headings = np.linspace(np.pi - 0.2, np.pi + 0.2, 8)  # wraps
    wrapped = np.arctan2(np.sin(headings), np.cos(headings))
    poses = convert.to_navsim_poses(np.column_stack([xy, wrapped]), 0.5)
    steps = np.abs(np.diff(poses[:, 2]))
    assert steps.max() < 0.5, "heading jumped the seam"


def test_a_non_finite_plan_is_rejected() -> None:
    bad = _straight(8, 0.5)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        convert.to_navsim_poses(bad, 0.5)


# -- ego state -------------------------------------------------------------


def test_pose_matrix_round_trips() -> None:
    matrix = convert.pose_matrix([3.0, -2.0, 0.7])
    assert matrix[0, 3] == pytest.approx(3.0)
    assert matrix[1, 3] == pytest.approx(-2.0)
    assert np.arctan2(matrix[1, 0], matrix[0, 0]) == pytest.approx(0.7)


def test_canbus_fills_the_offsets_the_protocol_documents() -> None:
    """Every model reads speed, accel and yaw rate at fixed offsets.

    NAVSIM has no CAN bus, so these are reconstructed. Putting speed in the
    wrong slot would hand every model a plausible, wrong number.
    """

    class _Status:
        ego_pose = np.array([1.0, 2.0, 0.3])
        ego_velocity = np.array([7.5, 0.25])
        ego_acceleration = np.array([1.5, -0.5])

    bus = convert.canbus(_Status(), yaw_rate=0.125)
    assert len(bus) == 18
    assert bus[CANBUS_VELOCITY] == pytest.approx(7.5)
    assert bus[CANBUS_ACCEL] == pytest.approx(1.5)
    assert bus[CANBUS_YAW_RATE] == pytest.approx(0.125)


def test_the_command_mapping_inverts_the_drivejepa_adapter() -> None:
    """One mapping, two directions. They must agree or a turn becomes a straight.

    drivejepa's policy converts paddock -> NAVSIM; this converts NAVSIM ->
    paddock. KITScenes showed the nav signal moves ADE by ~9%, so a silent
    disagreement here is a real error, not a cosmetic one.
    """
    from paddock.adapters.models.drivejepa.policy import COMMAND_ONE_HOT

    for command, one_hot in COMMAND_ONE_HOT.items():
        index = int(np.argmax(one_hot))
        assert convert.COMMAND_FROM_ONE_HOT[index] == command, f"{Command(command).name} does not invert"


# -- cameras ---------------------------------------------------------------


def test_every_camera_a_model_declares_has_a_navsim_slot() -> None:
    """The nuPlan rig is not nuScenes'. A model asking for a view it cannot get
    should fail at preflight, not partway into a scoring run."""
    for model in Registry.load().models.values():
        if not any(name.startswith("navsim") for name in model.benchmarks):
            continue
        for camera in model.cameras:
            assert camera in convert.CAMERA_SLOTS, f"{model.name} wants {camera}"


def test_camera_slots_are_distinct() -> None:
    """Two paddock cameras mapping to one NAVSIM slot would silently duplicate a view."""
    slots = list(convert.CAMERA_SLOTS.values())
    assert len(slots) == len(set(slots))


# -- wiring ----------------------------------------------------------------


def test_the_proxy_agent_config_is_tracked_and_names_the_module() -> None:
    """hydra resolves the agent by dotted path; a rename must break a test, not a run."""
    config = (CONF_DIR / "agent" / f"{PROXY_AGENT}.yaml").read_text()
    assert "paddock.adapters.benchmarks.navsim.agent.PaddockProxyAgent" in config
    assert "node_url: ???" in config, "node_url must be supplied by the harness, never defaulted"


#: What the stand-in devkit's setup.py declares per benchmark: the harness
#: refuses the other generation's tree (both ship a run_pdm_score.py).
DEVKIT_VERSIONS = {"navsim": "2.0.0", "navsim_v1": "1.1.0"}


def _navsim_site(
    tmp_path: Path,
    monkeypatch,
    split: str = "navtest",
    data_split: str = "test",
    declared: tuple[str, ...] | None = ("log_a", "log_b"),
    blobs: tuple[str, ...] = ("log_a", "log_b"),
    cached: tuple[str, ...] = ("log_a", "log_b", "metadata"),
    tokens_per_log: int = 0,
    bench: str = "navsim",
    scorer: str = SCORER,
    devkit_version: str | None = None,
    camera_path: bool = True,
    declare_tokens: bool = False,
    metadata_csv: bool = True,
    **settings: str,
) -> SiteConfig:
    """A site with a stand-in devkit, dataset and metric cache laid out as NAVSIM's are.

    `bench` picks the `[benchmarks.<bench>]` block and, with `scorer`, the
    devkit generation being stood in for; the stand-in carries the setup.py
    version and the `camera_path` field preflight looks for, and a metric cache
    metadata csv listing the stand-in scenarios, as the devkit's caching writes.
    """
    import sys

    devkit = tmp_path / "devkit"
    (devkit / scorer).parent.mkdir(parents=True)
    (devkit / scorer).write_text("")
    version = devkit_version or DEVKIT_VERSIONS[bench]
    (devkit / "setup.py").write_text(f'setup(name="navsim", version="{version}")\n')
    (devkit / "navsim" / "common").mkdir(parents=True, exist_ok=True)
    (devkit / "navsim" / "common" / "dataclasses.py").write_text(
        "class Camera:\n    image = None\n" + ("    camera_path = None\n" if camera_path else "")
    )
    splits = devkit / SPLITS_DIR
    (splits / "scene_filter").mkdir(parents=True)
    (splits / f"{split}.yaml").write_text(f"data_split: {data_split}\n")
    tokens = [f"{name}-t{i}" for name in cached if name != "metadata" for i in range(tokens_per_log)]
    if declared is not None:
        (splits / "scene_filter" / f"{split}.yaml").write_text(
            "log_names:\n"
            + "".join(f"  - {name}\n" for name in declared)
            + ("tokens:\n" + "".join(f"  - {token}\n" for token in tokens) if declare_tokens else "")
        )
    data = tmp_path / "data"
    for name in blobs:
        (data / "sensor_blobs" / data_split / name).mkdir(parents=True)
    (data / "navsim_logs" / data_split).mkdir(parents=True)
    (data / "maps").mkdir()
    cache = tmp_path / "cache"
    rows: list[str] = []
    for name in cached:
        (cache / name).mkdir(parents=True)
        # The cache is <cache>/<log>/<scenario type>/<token>/, one directory
        # per scenario; a stand-in with a few tokens per log.
        for i in range(tokens_per_log if name != "metadata" else 0):
            (cache / name / "unknown" / f"{name}-t{i}").mkdir(parents=True)
            rows.append(str(cache / name / "unknown" / f"{name}-t{i}" / "metric_cache.pkl"))
    if metadata_csv:
        (cache / "metadata").mkdir(exist_ok=True)
        (cache / "metadata" / "metric_cache_metadata_node_0.csv").write_text(
            "file_name\n" + "".join(r + "\n" for r in rows)
        )
    for sub in ("repos", "runs", "weights", "exp"):
        (tmp_path / sub).mkdir()
    (tmp_path / "site.toml").write_text(
        f'site = "test-site"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
        f'weights = "{tmp_path / "weights"}"\n'
        f'[datasets.navsim]\npath = "{data}"\nexp = "{tmp_path / "exp"}"\n'
        f'[benchmarks.{bench}]\npython = "{sys.executable}"\ndevkit = "{devkit}"\n'
        f'metric_cache = "{cache}"\nsplit = "{split}"\n'
        + "".join(f'{key} = "{value}"\n' for key, value in settings.items())
    )
    monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
    for name in (ENV_AGENT, ENV_ALLOW_PARTIAL, ENV_ALLOW_SLOW, ENV_OVERRIDES):
        monkeypatch.delenv(name, raising=False)
    return SiteConfig.load()


def _harness(cfg: SiteConfig, bench: str = "navsim"):
    from paddock.runner import load_benchmark

    return load_benchmark(Registry.load().benchmarks[bench], cfg)


def test_data_split_is_read_from_the_devkits_own_yaml(tmp_path: Path, monkeypatch) -> None:
    """No table of splits in paddock: the devkit's train_test_split yaml says
    which OpenScene split the scorer reads, and preflight reads the same file."""
    harness = _harness(_navsim_site(tmp_path, monkeypatch, split="navmini", data_split="mini"))
    assert harness.data_split == "mini"
    assert harness.preflight() == []

    (tmp_path / "devkit" / SPLITS_DIR / "navmini.yaml").unlink()
    problems = harness.preflight()
    assert len(problems) == 1 and "navmini" in problems[0] and "data_split" in problems[0]


def test_a_complete_download_is_not_refused_for_the_metadata_directory(tmp_path: Path, monkeypatch) -> None:
    """The cache holds `metadata/` beside the per-log directories; 2 logs with
    blobs against a 3-directory cache must read as 2 of 2, not 2 of 3."""
    assert _harness(_navsim_site(tmp_path, monkeypatch)).preflight() == []


def test_an_unreadable_log_list_is_reported_not_guessed_around(tmp_path: Path, monkeypatch) -> None:
    """When the split's scene filter cannot be read there is no fallback to the
    raw directory count, which is known to be wrong by one."""
    problems = _harness(_navsim_site(tmp_path, monkeypatch, declared=None)).preflight()
    assert len(problems) == 1 and "cannot read the log list" in problems[0]


def test_partial_coverage_is_refused_and_when_allowed_is_never_valid(tmp_path: Path, monkeypatch) -> None:
    cfg = _navsim_site(tmp_path, monkeypatch, blobs=("log_a",))
    harness = _harness(cfg)
    problems = harness.preflight()
    assert len(problems) == 1 and "1 of 2 navtest logs" in problems[0]
    assert harness.run_config()["allow_partial"] is False

    monkeypatch.setenv(ENV_ALLOW_PARTIAL, "1")
    assert harness.preflight() == []
    assert harness.run_config()["allow_partial"] is True
    reasons = harness.invalid_reasons()
    assert len(reasons) == 1 and "1 of 2 navtest logs" in reasons[0]


def test_measurement_switches_reach_run_config_and_validity(tmp_path: Path, monkeypatch) -> None:
    """Every switch that changes what is measured is on the record, and the
    ones that make a run unrankable say so -- by mechanism, not discipline."""
    harness = _harness(_navsim_site(tmp_path, monkeypatch))
    assert harness.run_config() == {
        "split": "navtest",
        "agent": PROXY_AGENT,
        "allow_partial": False,
        "allow_slow": False,
        "overrides": [],
    }
    assert harness.invalid_reasons() == []

    monkeypatch.setenv(ENV_AGENT, "drivejepa")
    monkeypatch.setenv(ENV_ALLOW_SLOW, "1")
    monkeypatch.setenv(ENV_OVERRIDES, "worker.max_workers=2 train_test_split.scene_filter.max_scenes=9")
    config = harness.run_config()
    assert config["agent"] == "drivejepa" and config["allow_slow"] is True
    assert config["overrides"] == ["worker.max_workers=2", "train_test_split.scene_filter.max_scenes=9"]
    reasons = harness.invalid_reasons()
    assert any("native agent 'drivejepa'" in r for r in reasons)
    assert any(ENV_ALLOW_SLOW in r for r in reasons)


def test_model_problems_are_pure_so_they_run_before_any_node_starts() -> None:
    assert "10 Hz observations" in input_rate_problems(10.0, "openpilot")[0]
    assert input_rate_problems(0.0, "alpamayo15") == []
    assert "below NAVSIM's floor" in limit_problems(1)[0]
    assert limit_problems(2) == [] and limit_problems(None) == []


def test_dry_run_reports_the_rate_gap_and_the_validity_without_loading_a_model(tmp_path, monkeypatch) -> None:
    from paddock.runner import run_benchmark

    cfg = _navsim_site(tmp_path, monkeypatch)
    lines: list[str] = []
    code = run_benchmark(cfg, bench="navsim", model="openpilot", dry_run=True, limit=3, echo=lines.append)
    assert code == 0
    out = "\n".join(lines)
    assert "openpilot needs 10 Hz" in out
    assert "valid=false: --limit 3" in out


def test_wire_modules_compile_under_python_39() -> None:
    """The proxy agent and what it imports run inside the vendor's 3.9
    interpreter. A 3.10+ construct in protocol.py or convert.py would fail
    every NAVSIM run inside hydra with only a log tail to show for it."""
    import shutil
    import subprocess

    from paddock.config import REPO_ROOT

    candidates = [shutil.which("python3.9")]
    try:
        settings = SiteConfig.load().benchmarks.get("navsim")
        candidates.append(settings.get("python") if settings else None)
    except ConfigError:
        pass
    # A review worktree has no paddock.toml of its own; the checkout it was
    # made from usually does, and its [benchmarks.navsim] python is the
    # interpreter these modules actually run in.
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    site_toml = (REPO_ROOT / common).resolve().parent / "paddock.toml" if common else None
    if site_toml is not None and site_toml.exists():
        import tomllib

        parent_site = tomllib.loads(site_toml.read_text())
        candidates.append(parent_site.get("benchmarks", {}).get("navsim", {}).get("python"))

    def is_39(path: str | None) -> bool:
        if not path:
            return False
        version = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
        return version.startswith("Python 3.9")

    interpreter = next((c for c in candidates if is_39(c)), None)
    if interpreter is None:
        pytest.skip(
            "no Python 3.9 interpreter found: not python3.9 on PATH, not [benchmarks.navsim] python "
            "in PADDOCK_CONFIG or this worktree's parent checkout"
        )
    base = REPO_ROOT / "src" / "paddock"
    files = [
        base / "policy" / "protocol.py",
        base / "adapters" / "benchmarks" / "navsim" / "convert.py",
        base / "adapters" / "benchmarks" / "navsim" / "agent.py",
        base / "adapters" / "benchmarks" / "navsim_v1" / "agent.py",
    ]
    command = [interpreter, "-m", "py_compile", *map(str, files)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_scorer_path_exists_in_the_devkit() -> None:
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    settings = cfg.benchmarks.get("navsim")
    if settings is None or not settings.get("devkit"):
        pytest.skip("navsim devkit not configured at this site")

    from pathlib import Path

    assert (Path(settings.get("devkit")) / SCORER).exists()


# -- model grids -----------------------------------------------------------


def test_every_navsim_model_plans_far_enough_to_be_scored() -> None:
    """Each model's waypoint grid must reach NAVSIM's 4 s horizon.

    `to_navsim_poses` refuses a short plan rather than extrapolating, so a model
    whose grid is narrowed back below 4 s would fail every scenario at
    conversion. Catching it here costs a millisecond instead of a scoring run.

    The grids are read from each policy's own constants rather than restated, so
    this tracks the models rather than a copy of them.
    """
    horizon = convert.NAVSIM_NUM_POSES * convert.NAVSIM_INTERVAL_S

    from paddock.adapters.models.alpamayo15.policy import GRID_INDICES, HISTORY_DT
    from paddock.adapters.models.openpilot.policy import GRID_S
    from paddock.adapters.models.visionpilot.policy import PLAN_HORIZON_S, WAYPOINT_DT

    reach = {
        "alpamayo15": (float(GRID_INDICES[-1] + 1) * HISTORY_DT, len(GRID_INDICES)),
        "openpilot": (float(GRID_S[-1]), len(GRID_S)),
        "visionpilot": (PLAN_HORIZON_S, int(round(PLAN_HORIZON_S / WAYPOINT_DT))),
    }
    for model, (last, count) in reach.items():
        assert last >= horizon, f"{model} plans {last} s, NAVSIM scores {horizon} s"
        assert count >= convert.NAVSIM_NUM_POSES, f"{model} emits {count} poses"


def test_extending_alpamayos_grid_left_the_other_benchmarks_alone() -> None:
    """The first six indices are what NeuroNCAP and KITScenes consume.

    Widening the grid for NAVSIM must not move a recorded neuroNCAP or KITScenes
    score, and the guarantee is that the first six entries are untouched.
    """
    from paddock.adapters.models.alpamayo15.policy import GRID_INDICES
    from paddock.policy.protocol import N_WAYPOINTS

    assert list(GRID_INDICES[:N_WAYPOINTS]) == [4, 9, 14, 19, 24, 29]


def test_alpamayos_policy_grid_matches_the_kitscenes_subsampler() -> None:
    """Two places subsample the same 10 Hz output; they must agree."""
    from paddock.adapters.benchmarks.kitscenes.metrics import GRID_IDX
    from paddock.adapters.models.alpamayo15.policy import GRID_INDICES

    assert list(GRID_INDICES) == list(GRID_IDX[: len(GRID_INDICES)])


def test_a_limit_below_navsims_floor_is_refused_with_a_reason() -> None:
    """One scenario cannot be aggregated, and the scorer's own failure is a
    pandas KeyError that names neither the cause nor the fix."""
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    spec = Registry.load().benchmarks.get("navsim")
    if spec is None:
        pytest.skip("navsim not registered")
    from paddock.adapters.benchmarks.navsim.harness import BENCHMARK, MIN_SCENARIOS

    harness = BENCHMARK(spec, cfg)
    with pytest.raises(RuntimeError, match="below NAVSIM's floor"):
        harness.evaluate("http://127.0.0.1:0", Path("/tmp"), {"cameras": ["CAM_FRONT"]}, limit=1)
    assert MIN_SCENARIOS == 2


def test_a_model_needing_denser_input_than_navsim_has_is_refused() -> None:
    """NAVSIM is a 2 Hz dataset. A model that reads motion from frame-to-frame
    change scores near the constant-velocity baseline on it while measuring the
    rate gap, which is worse than not running: it looks like a result."""
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    spec = Registry.load().benchmarks.get("navsim")
    if spec is None:
        pytest.skip("navsim not registered")
    from paddock.adapters.benchmarks.navsim.harness import BENCHMARK, INPUT_HZ

    assert INPUT_HZ == pytest.approx(2.0)
    harness = BENCHMARK(spec, cfg)
    with pytest.raises(RuntimeError, match="Hz observations"):
        harness.evaluate(
            "http://127.0.0.1:0",
            Path("/tmp"),
            {"cameras": ["CAM_FRONT"], "model": "openpilot", "min_input_hz": 10.0},
            limit=4,
        )


def test_openpilot_declares_the_rate_neuroncap_already_established() -> None:
    """neuroNCAP rebuilt its scenarios at 10 Hz because openpilot reads a 0.5 s
    step as a parked car. That fact now lives in the registry, where a benchmark
    can act on it, instead of only in a doc."""
    openpilot = Registry.load().models["openpilot"]
    assert openpilot.min_input_hz >= 10.0
    assert "navsim" not in openpilot.benchmarks and "navsim_v1" not in openpilot.benchmarks, (
        "openpilot cannot be measured on a 2 Hz dataset; declaring it would make "
        "`--bench navsim --model all` (or navsim_v1) fail on every sweep"
    )


def test_alpamayo_falls_back_to_distinct_frames_only_when_the_grid_degenerates() -> None:
    """The fix for alpamayo's camera collapse, and its blast radius.

    The fallback is decided from the *source cadence* (the frames that would
    fill the slots are all >= 0.2 s apart), not from the grid's output alone:
    neuroNCAP's priming transition also makes the grid repeat a frame, and
    changing that path would move a recorded score.

    `_image_frames` picks four frames on a fixed 0.1 s grid. At NAVSIM's 2 Hz
    that returned the same frame for three of four slots, so the model could not
    see other vehicles moving and drove into them: measured over 60 paired
    scenarios, `no_at_fault_collisions` was 0.708 with the grid and 0.992 with
    distinct frames, worth +0.207 EPDMS overall.

    `auto` must therefore fall back on sparse input -- and must NOT change
    anything at 10 Hz, uniform or jittered, or it would move recorded neuroNCAP
    and KITScenes scores.
    """
    import os

    import numpy as np

    from paddock.adapters.models.alpamayo15.policy import NUM_CAM_FRAMES, AlpamayoPolicy

    class _Stub(AlpamayoPolicy):
        def __init__(self, timestamps):
            self.timestamps = list(timestamps)
            self.frames = [{"id": i} for i in range(len(timestamps))]

    previous = os.environ.pop("ALPAMAYO_FRAME_SELECT", None)
    try:
        uniform = [int(-t * 1e5) for t in range(9, -1, -1)]
        jittered = [-900000, -805000, -698000, -604000, -499000, -401000, -297000, -203000, -98000, 0]
        for label, times in (("uniform", uniform), ("jittered", jittered)):
            chosen = [f["id"] for f in _Stub(times)._select_frames()]
            assert chosen == [6, 7, 8, 9], f"10 Hz {label} must keep the grid path, got {chosen}"

        sparse = [-1500000, -1000000, -500000, 0]
        chosen = [f["id"] for f in _Stub(sparse)._select_frames()]
        assert len(set(chosen)) == NUM_CAM_FRAMES, f"2 Hz must fall back, got {chosen}"

        # neuroNCAP's real cadence: two priming steps at 0.5 s dataset keyframes,
        # then the 0.1 s loop. The grid repeats a frame here too, but the source
        # is 10 Hz by now and the recorded neuroNCAP scores were made on the
        # grid path -- so the grid path it stays, at both transition steps.
        priming = [0, 500000, 1000000, 1100000]
        chosen = [f["id"] for f in _Stub(priming)._select_frames()]
        assert chosen == [2, 2, 2, 3], f"priming transition must keep the grid path, got {chosen}"
        chosen = [f["id"] for f in _Stub([*priming, 1200000])._select_frames()]
        assert chosen == [2, 2, 3, 4], f"first closed-loop step must keep the grid path, got {chosen}"

        # and the grid really would have collapsed, or the fallback guards nothing
        times = np.array(sparse, dtype=np.float64) / 1e6
        wanted = times[-1] - np.arange(NUM_CAM_FRAMES - 1, -1, -1) * 0.1
        grid = [int(np.argmin(np.abs(times - w))) for w in wanted]
        assert len(set(grid)) < NUM_CAM_FRAMES
    finally:
        if previous is not None:
            os.environ["ALPAMAYO_FRAME_SELECT"] = previous


def test_alpamayo_is_evaluable_on_navsim() -> None:
    """It was gated on an n=4 read that a 60-scenario run overturned: 0.226
    became 0.503 with the grid selection and 0.710 with the fallback, against a
    0.257 constant-velocity baseline. Sample size, not capability."""
    alpamayo = Registry.load().models["alpamayo15"]
    assert alpamayo.min_input_hz == 0.0
    assert "navsim" in alpamayo.benchmarks


def test_coverage_counts_logs_not_directories() -> None:
    """The metric cache holds a `metadata/` directory alongside the per-log ones.

    Counting directories makes a *complete* 136-log download read as 136 of 137
    and refuse -- a false negative that would surface only after a ~106 GiB
    fetch, which is the worst possible moment to find it.
    """
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    spec = Registry.load().benchmarks.get("navsim")
    settings = cfg.benchmarks.get("navsim")
    if spec is None or settings is None or not settings.get("devkit"):
        pytest.skip("navsim not configured at this site")
    from paddock.adapters.benchmarks.navsim.harness import BENCHMARK

    declared = BENCHMARK(spec, cfg)._declared_logs()
    if not declared:
        pytest.skip("devkit scene filter not present")
    assert len(declared) == 136, f"navtest declares 136 logs, got {len(declared)}"
    assert "metadata" not in declared


# -- a crash must not become a score (docs/reviewing.md §4) ----------------


def test_a_failed_scenario_is_excluded_and_reported_not_averaged(tmp_path: Path) -> None:
    """The scorer writes `get_empty_results()` -- every metric NaN, stored as
    empty cells -- with valid=False for a scenario whose agent raised. Before
    this, such a row became a sample with no metrics: counted in n, invisible
    as a failure, the run reading as complete. Values here are arbitrary; the
    row must be excluded whatever it holds."""
    from paddock.results.ingest.navsim import SUB_METRICS, parse_scores

    header = ["token", "valid", *SUB_METRICS, "score"]
    good = ["aaaa", "True", *["1.0"] * len(SUB_METRICS), "0.9"]
    crashed = ["bbbb", "False", *["0.0"] * len(SUB_METRICS), "0.0"]
    aggregate = ["average_all_frames", "True", *["0.5"] * len(SUB_METRICS), "0.45"]
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("\n".join(",".join(r) for r in (header, good, crashed, aggregate)) + "\n")

    samples, aggregates, failed = parse_scores(csv_path)
    assert [s["native_id"]["token"] for s in samples] == ["aaaa"], "only the scored scenario is a sample"
    assert failed == ["bbbb"], "the crashed one is named, not silently dropped"
    assert "average_all_frames" in aggregates
    assert all(s["metrics"]["epdms"] > 0 for s in samples), "no zero from a crash reached the metrics"


def test_ingest_marks_a_run_with_failed_scenarios_partial_and_invalid(tmp_path: Path) -> None:
    from paddock.results.ingest.navsim import SUB_METRICS, ingest

    header = ["token", "valid", *SUB_METRICS, "score"]
    rows = [
        header,
        ["aaaa", "True", *["1.0"] * len(SUB_METRICS), "0.9"],
        ["bbbb", "False", *["0.0"] * len(SUB_METRICS), "0.0"],
    ]
    run_dir = tmp_path / "2026.01.01.00.00.00"
    run_dir.mkdir()
    (run_dir / "scores.csv").write_text("\n".join(",".join(r) for r in rows) + "\n")

    run, metrics, samples = ingest(run_dir / "scores.csv", "somemodel", site="test")[0]
    assert run.valid is False and run.status == "partial"
    assert "1 scenario(s) failed" in run.notes
    assert metrics.n == 1


# -- contracts are imported, not restated (docs/reviewing.md §2) -----------


def test_command_map_is_built_from_the_enum_not_literals() -> None:
    """A renumbering of Command must flow through, not leave a stale tuple."""
    assert convert.COMMAND_FROM_ONE_HOT == (
        int(Command.LEFT),
        int(Command.STRAIGHT),
        int(Command.RIGHT),
        int(Command.FOLLOW_REFERENCE),
    )


def test_the_image_encoder_is_the_protocols_and_round_trips() -> None:
    """One encoder, beside the decoder that sniffs it; the runner and the
    vendor-side agent must not each carry a copy."""
    from paddock.policy import protocol

    assert convert.encode_image is protocol.encode_image
    image = (np.arange(2 * 3 * 3) % 256).astype(np.uint8).reshape(2, 3, 3)
    assert np.array_equal(protocol.decode_image(protocol.encode_image(image)), image)


# -- a lost run is a chunk, not a split -------------------------------------


def test_chunking_is_deterministic_and_covers_every_log() -> None:
    from paddock.adapters.benchmarks.navsim.harness import chunk_logs

    logs = [f"log{i:03d}" for i in range(10)]
    chunks = chunk_logs(logs, 4)
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert [x for c in chunks for x in c] == logs, "order preserved, nothing dropped or repeated"
    assert chunk_logs(logs, 0) == [logs], "0 means one invocation, as the vendor does"


def test_the_hydra_list_literal_quotes_names_with_dots() -> None:
    """Log names look like 2021.05.25.14.16.10_veh-35_02482_02649; unquoted,
    hydra reads the dots as structure."""
    from paddock.adapters.benchmarks.navsim.harness import hydra_list

    assert hydra_list(["2021.05.25.14.16.10_veh-35_02482_02649", "b"]) == (
        "['2021.05.25.14.16.10_veh-35_02482_02649','b']"
    )


def test_merging_chunk_csvs_drops_each_chunks_aggregate_rows(tmp_path: Path) -> None:
    """Every chunk CSV carries average_all_frames and friends; across chunks
    they mean nothing, and left in they would be parsed as scenarios."""
    from paddock.adapters.benchmarks.navsim.harness import merge_chunk_csvs
    from paddock.results.ingest.navsim import SUB_METRICS, parse_scores

    header = ["token", "valid", *SUB_METRICS, "score"]

    def chunk(path: Path, tokens: list[str]) -> Path:
        rows = [header] + [[t, "True", *["1.0"] * len(SUB_METRICS), "0.9"] for t in tokens]
        rows.append(["average_all_frames", "True", *["1.0"] * len(SUB_METRICS), "0.9"])
        path.write_text("\n".join(",".join(r) for r in rows) + "\n")
        return path

    merged = merge_chunk_csvs(
        [chunk(tmp_path / "a.csv", ["t1", "t2"]), chunk(tmp_path / "b.csv", ["t3"])], tmp_path / "scores.csv"
    )
    samples, aggregates, failed = parse_scores(merged)
    assert [s["native_id"]["token"] for s in samples] == ["t1", "t2", "t3"]
    assert aggregates == {} and failed == []


def test_a_completed_chunk_is_detected_by_its_csv(tmp_path: Path) -> None:
    """Resume keys on the CSV the scorer writes last; a chunk without one did
    not finish, whatever else its directory holds."""
    from paddock.adapters.benchmarks.navsim.harness import completed_chunk_csv

    d = tmp_path / "000"
    d.mkdir()
    (d / "navsim_scorer.log").write_text("half way")
    assert completed_chunk_csv(d) is None
    (d / "2026.09.03.11.00.00.csv").write_text("token\n")
    assert completed_chunk_csv(d).name.endswith(".csv")
    assert completed_chunk_csv(tmp_path / "missing") is None


def test_merging_prefers_a_retried_score_over_the_failed_row_it_replaces(tmp_path: Path) -> None:
    """The retry pass re-scores failed tokens into their own chunk; on merge the
    scored row wins, and a failed row never overwrites a scored one."""
    from paddock.adapters.benchmarks.navsim.harness import merge_chunk_csvs
    from paddock.results.ingest.navsim import SUB_METRICS, parse_scores

    header = ["token", "valid", *SUB_METRICS, "score"]
    first = [
        header,
        ["t1", "True", *["1.0"] * len(SUB_METRICS), "0.9"],
        ["t2", "False", *[""] * len(SUB_METRICS), ""],
    ]
    retry = [
        header,
        ["t2", "True", *["1.0"] * len(SUB_METRICS), "0.8"],
        ["t1", "False", *[""] * len(SUB_METRICS), ""],
    ]
    (tmp_path / "000.csv").write_text("\n".join(",".join(r) for r in first) + "\n")
    (tmp_path / "retry.csv").write_text("\n".join(",".join(r) for r in retry) + "\n")

    samples, _, failed = parse_scores(
        merge_chunk_csvs([tmp_path / "000.csv", tmp_path / "retry.csv"], tmp_path / "m.csv")
    )
    assert sorted(s["native_id"]["token"] for s in samples) == ["t1", "t2"]
    assert failed == [], "t2 was rescued by the retry; t1's later failed row did not unseat its score"


# -- a dead node or scorer is retried with fresh nodes, on the same artifacts --


def test_the_runner_restarts_nodes_and_resumes_after_a_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first attempt dies (a scorer crash, a node that fell over); the
    runner tears the nodes down, starts fresh ones, and re-invokes `evaluate`
    on the same artifacts directory, where a resumable benchmark continues."""
    import contextlib

    from paddock import runner
    from paddock.benchmark import Evaluation, ExternallyScoredBenchmark
    from paddock.registry import BenchmarkSpec, ModelSpec, Runtime
    from paddock.runtime import proc

    started_nodes: list[int] = []

    @contextlib.contextmanager
    def fake_node(spec, cfg, port, log_dir, echo=print, nonce=None):
        started_nodes.append(port)
        yield f"http://127.0.0.1:{port}"

    calls: list[dict] = []

    class Flaky(ExternallyScoredBenchmark):
        run_retries = 2
        split = "navtest"

        def preflight(self):
            return []

        def model_problems(self, model, limit):
            return []

        def invalid_reasons(self):
            return []

        def evaluate(self, policy_url, out_dir, config, limit=None, echo=print):
            calls.append(dict(config))
            if len(calls) == 1:
                raise RuntimeError("NAVSIM scorer exited 1")  # the crash
            return Evaluation(
                samples=[
                    {
                        "sample_id": "navsim/navtest/x",
                        "native_id": {},
                        "split": "navtest",
                        "metrics": {"epdms": 0.5},
                    }
                ]
            )

    stored = {}
    monkeypatch.setattr(proc, "model_node", fake_node)
    monkeypatch.setattr(
        runner,
        "_store_run",
        lambda *a, **k: (
            stored.setdefault("called", True),
            type("M", (), {"n": 1, "metrics": {"epdms": {"mean": 0.5}}})(),
        ),
    )
    monkeypatch.setattr(runner.ports, "find_free", lambda p: p)

    # Nothing here depends on the site: nodes, storage and ports are all
    # stand-ins, so the config is a constructed one and the test runs on a
    # machine with no paddock.toml (CI, a fresh clone, the other site).
    cfg = _navsim_site(tmp_path, monkeypatch)
    spec = BenchmarkSpec(name="navsim", mode="open_loop", dataset="navsim", primary_metric="epdms")
    model = ModelSpec(name="m", repo="r", runtime=Runtime(kind="shell"), vram_gb=0)
    code = runner._run_external(cfg, Flaky(spec, cfg), spec, model, 9900, None, lambda m: None)

    assert code == 0
    assert len(calls) == 2, "one crash, one successful retry"
    assert calls[1]["attempt"] == 2
    assert len(started_nodes) == 2, "fresh nodes for the retry, not the ones that were up when it died"
    assert stored.get("called")


def test_node_alive_is_false_for_a_port_nobody_listens_on() -> None:
    from paddock.adapters.benchmarks.navsim.harness import node_alive

    assert node_alive("http://127.0.0.1:1", timeout_s=0.5) is False


# -- chunk accounting, the chunk log, and resuming under one configuration ----

STUB_SCORER = """\
import os, sys
args = sys.argv[1:]
def opt(prefix):
    return next((a[len(prefix):] for a in args if a.startswith(prefix)), "")
out = opt("output_dir=")
logs = [x.strip("'") for x in opt("train_test_split.scene_filter.log_names=").strip("[]").split(",") if x]
if os.environ.get("STUB_FAIL_LOG") and os.environ["STUB_FAIL_LOG"] in logs:
    print("stub scorer: boom", flush=True)
    sys.exit(1)
from paddock.results.ingest.navsim import SUB_METRICS
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "scores.csv"), "w") as f:
    f.write(",".join(["token", "valid", *SUB_METRICS, "score", "log_name"]) + "\\n")
    for log in logs or ["single"]:
        for i in range(3):
            f.write(",".join([f"{log}-t{i}", "True", *["1.0"] * len(SUB_METRICS), "0.5", log]) + "\\n")
    f.write(",".join(["average_all_frames", "True", *["1.0"] * len(SUB_METRICS), "0.5", ""]) + "\\n")
print("stub scorer done", flush=True)
"""


def _stub_scored_site(tmp_path: Path, monkeypatch, **settings: str) -> SiteConfig:
    """A site whose devkit 'scorer' is a stand-in that writes three scored
    rows per log it is given, and fails on the log named in STUB_FAIL_LOG."""
    cfg = _navsim_site(tmp_path, monkeypatch, tokens_per_log=3, **settings)
    (tmp_path / "devkit" / SCORER).write_text(STUB_SCORER)
    monkeypatch.delenv("STUB_FAIL_LOG", raising=False)
    return cfg


def test_chunk_log_is_its_own_file_appended_across_attempts_and_the_scorer_log_keeps_its_command(
    tmp_path: Path, monkeypatch
) -> None:
    """`chunks.log` is the run's story; `navsim_scorer.log` is the scorer's and
    opens with the command that was run -- also for an unchunked (--limit) run,
    where both live at the same level and used to share one path."""
    import sys

    from paddock.adapters.benchmarks.navsim import harness as harness_module

    harness = _harness(_stub_scored_site(tmp_path, monkeypatch, chunk_logs="1"))
    monkeypatch.setattr(harness_module, "node_alive", lambda url, timeout_s=5.0: True)
    out = tmp_path / "out"
    config = {"cameras": ["CAM_FRONT"], "model": "m", "attempt": 1}

    first = harness.evaluate("http://127.0.0.1:1", out, config)
    assert len(first.samples) == 6 and first.failed == [] and first.failed_chunks == []
    assert first.n_expected == 6, "two logs, three cached scenarios each"
    story = (out / "chunks.log").read_text()
    assert "attempt 1: 2 chunk(s)" in story and "chunk 000: ok (1 log(s))" in story
    scorer_log = (out / "chunks" / "000" / "navsim_scorer.log").read_text()
    assert scorer_log.startswith(sys.executable), "the scorer log opens with the command line"

    second = harness.evaluate("http://127.0.0.1:1", out, {**config, "attempt": 2})
    assert len(second.samples) == 6
    story = (out / "chunks.log").read_text()
    assert "attempt 1:" in story and "attempt 2:" in story, "appended, not truncated"
    assert story.count("reused") == 2, "both chunks reused, no scorer invocation"

    smoke = tmp_path / "smoke"
    harness.evaluate("http://127.0.0.1:1", smoke, config, limit=2)
    assert (smoke / "navsim_scorer.log").read_text().startswith(sys.executable)
    assert "chunk 000: ok" in (smoke / "chunks.log").read_text()


def test_a_failed_chunk_is_counted_in_scenarios_not_logs(tmp_path: Path, monkeypatch) -> None:
    """A lost chunk of four logs is about a thousand scenarios; the run record
    must say that, not "4 scenario(s) failed"."""
    from paddock.adapters.benchmarks.navsim import harness as harness_module
    from paddock.runner import _external_failure_note

    harness = _harness(_stub_scored_site(tmp_path, monkeypatch, chunk_logs="1"))
    monkeypatch.setattr(harness_module, "node_alive", lambda url, timeout_s=5.0: True)
    monkeypatch.setenv("STUB_FAIL_LOG", "log_b")

    evaluation = harness.evaluate(
        "http://127.0.0.1:1", tmp_path / "out", {"cameras": ["CAM_FRONT"], "model": "m", "attempt": 1}
    )
    assert len(evaluation.samples) == 3 and evaluation.failed == []
    assert evaluation.failed_chunks == [{"index": 1, "logs": ["log_b"], "scenarios": 3}]
    assert evaluation.unscored == 3 and evaluation.n_expected == 6
    assert _external_failure_note(evaluation) == "1 chunk(s) never scored: 3 scenario(s) across 1 log(s)"
    assert "FAILED" in (tmp_path / "out" / "chunks.log").read_text()

    evaluation.failed = ["tok1", "tok2"]
    note = _external_failure_note(evaluation)
    assert note.startswith("2 scenario(s) failed in the scorer") and "1 chunk(s) never scored" in note


def test_an_unreadable_log_list_refuses_evaluate_the_way_preflight_does(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(_stub_scored_site(tmp_path, monkeypatch, declared=None, chunk_logs="1"))
    with pytest.raises(RuntimeError, match="cannot read the log list"):
        harness.evaluate("http://127.0.0.1:1", tmp_path / "out", {"cameras": ["CAM_FRONT"], "model": "m"})


def test_resume_refuses_a_different_measurement_configuration(tmp_path: Path) -> None:
    """run.json is written at the end; the configuration a run started under
    is written at the start, and a resume under another one is refused."""
    from paddock.runner import _config_marker_problems

    run_dir = tmp_path / "run"
    config = {"agent": PROXY_AGENT, "allow_partial": False, "limit": None, "cameras": ["CAM_FRONT"]}
    assert _config_marker_problems(run_dir, config) == []
    assert (run_dir / "run_config.json").exists()
    assert _config_marker_problems(run_dir, dict(config)) == [], "the same configuration resumes"

    problems = _config_marker_problems(run_dir, {**config, "agent": "drivejepa", "allow_partial": True})
    assert len(problems) == 1
    assert "agent, allow_partial" in problems[0] and "refused" in problems[0]
