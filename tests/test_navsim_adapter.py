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
from paddock.adapters.benchmarks.navsim.harness import CONF_DIR, PROXY_AGENT, SCORER, SPLIT_DATA
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
        if "navsim" not in model.benchmarks:
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


def test_the_split_table_matches_the_devkit() -> None:
    """SPLIT_DATA is duplicated from the devkit's config so preflight can run
    before hydra does. If the devkit is present, the two must agree."""
    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    settings = cfg.benchmarks.get("navsim")
    if settings is None or not settings.get("devkit"):
        pytest.skip("navsim devkit not configured at this site")

    from pathlib import Path

    splits = Path(settings.get("devkit")) / "navsim/planning/script/config/common/train_test_split"
    if not splits.is_dir():
        pytest.skip("devkit not present")
    for split, expected in SPLIT_DATA.items():
        declared = splits / f"{split}.yaml"
        if not declared.exists():
            continue
        for line in declared.read_text().splitlines():
            if line.startswith("data_split:"):
                assert line.split(":", 1)[1].strip() == expected, split


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
    assert "navsim" not in openpilot.benchmarks, (
        "openpilot cannot be measured on a 2 Hz dataset; declaring it would make "
        "`--bench navsim --model all` fail on every sweep"
    )


def test_alpamayo_falls_back_to_distinct_frames_only_when_the_grid_degenerates() -> None:
    """The fix for alpamayo's camera collapse, and its blast radius.

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
