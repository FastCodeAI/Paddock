"""Golden tests for the nuPlan adapter.

Builds a nuPlan-shaped log from scratch -- a car driving due east at a constant
10 m/s -- so the whole path from sqlite to a scored metric runs in paddock's own
environment with no dataset, no devkit and no GPU. A synthetic log is worth more
than a trimmed real one here: the expected ground truth is arithmetic rather than
a recorded number nobody can re-derive, so a change in the pose maths fails with
an obvious diff instead of a mysterious one.

The calibration test is the load-bearing one. nuPlan pickles its camera
intrinsics as devkit objects, and paddock's interpreter has no devkit; if that
ever stops working, every sample loses its intrinsics and openpilot silently
warps to the wrong focal length.
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuplan import loader as L
from paddock.adapters.benchmarks.nuplan.harness import (
    BENCHMARK,
    HISTORY_FRAMES,
    MIN_SEPARATION_S,
    N_GT_POSES,
)
from paddock.benchmark import Sample
from paddock.config import Dataset, Gpu, SiteConfig
from paddock.metrics import trajectory as M
from paddock.policy.protocol import (
    CANBUS_ACCEL,
    CANBUS_VELOCITY,
    CANBUS_YAW_RATE,
    constant_velocity,
)
from paddock.registry import Registry

LOG_NAME = "2021.05.12.22.00.38_veh-99_00000_00100"
SPEED_MS = 10.0
#: Distinct non-zero values, and distinct from each other: with both at
#: zero, swapping CANBUS_ACCEL and CANBUS_YAW_RATE passed every test while
#: loader.py argues those channels are load-bearing for drivejepa.
ACCEL_MS2 = 1.75
YAW_RATE = 0.166
DURATION_S = 100.0
T0 = 1_600_000_000_000_000
DEVKIT_MODULE = "nuplan.database.common.data_types"

#: (scenario type, seconds into the log). Each sits past the 4.8 s of history
#: the harness needs and more than 4 s before the log ends, and same-type tags
#: are more than MIN_SEPARATION_S apart, so all six survive selection.
TAGS = [
    ("near_pedestrian_on_crosswalk", 6.0),
    ("near_pedestrian_on_crosswalk", 30.0),
    ("near_pedestrian_on_crosswalk", 55.0),
    ("traversing_intersection", 12.0),
    ("traversing_intersection", 40.0),
    ("traversing_intersection", 70.0),
]

#: Same type, well inside MIN_SEPARATION_S of a tag above: the duplicate-moment
#: case thinning exists to remove.
CLOSE_TAGS = [("near_pedestrian_on_crosswalk", 6.05), ("near_pedestrian_on_crosswalk", 6.10)]

#: A second, rarer type on a keyframe that already carries one -- nuPlan's tags
#: overlap, and only the rarer label should survive the moment.
OVERLAPPING_TAG = ("traversing_narrow_lane", 12.0)


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------


def _devkit_blob(name: str, value: list) -> bytes:
    """A pickle that looks exactly like nuPlan's, written without the devkit.

    Registers a throwaway module under the devkit's name so `pickle` emits the
    same STACK_GLOBAL as a real database does, then removes it -- the decoder
    under test must work on a machine where that import would fail, which is
    every machine paddock runs on. The whole `nuplan.database.common` chain has
    to be registered, not just the leaf: pickle re-imports the module to confirm
    the class it is about to name really lives there.
    """
    parts = DEVKIT_MODULE.split(".")
    names = [".".join(parts[: i + 1]) for i in range(len(parts))]
    cls = type(name, (list,), {"__module__": DEVKIT_MODULE})
    created = [n for n in names if n not in sys.modules]
    for n in names:
        sys.modules.setdefault(n, types.ModuleType(n))
    setattr(sys.modules[DEVKIT_MODULE], name, cls)
    try:
        return pickle.dumps(cls(value), protocol=5)
    finally:
        for n in created:
            del sys.modules[n]


class _Reduces:
    """Pickles as a call to whatever global it is handed.

    The shape a tampered calibration column would take: `__reduce__` naming a
    function rather than a container, which is what `find_class` exists to
    refuse.
    """

    def __init__(self, target, args):
        self.target, self.args = target, args

    def __reduce__(self):
        return self.target, self.args


def _token(kind: str, index: int) -> bytes:
    return f"{kind}{index:08d}".encode()[:16]


def _build_log(path: Path) -> None:
    """One log: 100 Hz poses, 10 Hz front camera, 20 Hz tagged keyframes."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE log (token BLOB, vehicle_name TEXT, date TEXT, timestamp INTEGER,
                          logfile TEXT, location TEXT, map_version TEXT);
        CREATE TABLE camera (token BLOB, log_token BLOB, channel TEXT, model TEXT,
                             translation BLOB, rotation BLOB, intrinsic BLOB,
                             distortion BLOB, width INTEGER, height INTEGER);
        CREATE TABLE image (token BLOB, next_token BLOB, prev_token BLOB, ego_pose_token BLOB,
                            camera_token BLOB, filename_jpg TEXT, timestamp INTEGER);
        CREATE TABLE ego_pose (token BLOB, timestamp INTEGER, x FLOAT, y FLOAT, z FLOAT,
                               qw FLOAT, qx FLOAT, qy FLOAT, qz FLOAT,
                               vx FLOAT, vy FLOAT, vz FLOAT,
                               acceleration_x FLOAT, acceleration_y FLOAT, acceleration_z FLOAT,
                               angular_rate_x FLOAT, angular_rate_y FLOAT, angular_rate_z FLOAT);
        CREATE TABLE lidar_pc (token BLOB, next_token BLOB, prev_token BLOB, ego_pose_token BLOB,
                               lidar_token BLOB, scene_token BLOB, filename TEXT, timestamp INTEGER);
        CREATE TABLE scenario_tag (token BLOB, lidar_pc_token BLOB, type TEXT, agent_track_token BLOB);
        CREATE TABLE lidar (token BLOB, log_token BLOB, channel TEXT, model TEXT,
                            translation BLOB, rotation BLOB);
        """
    )
    logfile = path.stem
    conn.execute(
        "INSERT INTO log VALUES (?,?,?,?,?,?,?)",
        (_token("log", 0), "veh-99", "2021-05-12", T0, logfile, "las_vegas", "us-nv-las-vegas-strip"),
    )

    camera_token = _token("cam", 0)
    conn.execute(
        "INSERT INTO camera VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            camera_token,
            _token("log", 0),
            L.FRONT_CAMERA,
            "DesignCore D3CM-IMX390",
            _devkit_blob("Translation", [1.5, 0.0, 1.4]),
            _devkit_blob("Rotation", [1.0, 0.0, 0.0, 0.0]),
            _devkit_blob("CameraIntrinsic", [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]]),
            _devkit_blob("Rotation", []),
            1920,
            1080,
        ),
    )

    # nuPlan's MergedPointCloud is already expressed in the ego frame, so its
    # translation really is zero and its rotation really is identity -- that is
    # the dataset, not a decode failure, and this fixture keeps it that way so
    # the identity in `lidar2ego` is pinned rather than merely observed.
    conn.execute(
        "INSERT INTO lidar VALUES (?,?,?,?,?,?)",
        (
            _token("lid", 0),
            _token("log", 0),
            "MergedPointCloud",
            "Hesai Pandar",
            _devkit_blob("Translation", [0.0, 0.0, 0.0]),
            _devkit_blob("Rotation", [1.0, 0.0, 0.0, 0.0]),
        ),
    )

    # Straight and due east at a constant speed: yaw 0, so the ego frame is the
    # world frame translated, and the expected ground truth is just distance.
    poses = []
    for i in range(int(DURATION_S * L.EGO_HZ)):
        t = i / L.EGO_HZ
        poses.append(
            # token, timestamp, x, y, z, qw..qz, vx, vy, vz, accel xyz, angular xyz.
            # Constant velocity in a straight line: accel and yaw rate are 0,
            # which here is the truth rather than an unfilled channel.
            (
                _token("ego", i),
                int(T0 + t * 1e6),
                SPEED_MS * t,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                SPEED_MS,
                0.0,
                0.0,
                ACCEL_MS2,
                0.0,
                0.0,
                0.0,
                0.0,
                YAW_RATE,
            )
        )
    conn.executemany("INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", poses)

    images = []
    for i in range(int(DURATION_S * L.CAMERA_HZ)):
        t = i / L.CAMERA_HZ
        images.append(
            (
                _token("img", i),
                None,
                None,
                _token("ego", int(t * L.EGO_HZ)),
                camera_token,
                f"{logfile}/{L.FRONT_CAMERA}/{i:06d}.jpg",
                int(T0 + t * 1e6),
            )
        )
    conn.executemany("INSERT INTO image VALUES (?,?,?,?,?,?,?)", images)

    lidar, tags = [], []
    for i in range(int(DURATION_S * 20.0)):
        t = i / 20.0
        lidar.append(
            # token, next, prev, ego_pose_token, lidar_token, scene_token, filename, timestamp
            (
                _token("lpc", i),
                None,
                None,
                _token("ego", int(t * L.EGO_HZ)),
                None,
                None,
                "",
                int(T0 + t * 1e6),
            )
        )
    conn.executemany("INSERT INTO lidar_pc VALUES (?,?,?,?,?,?,?,?)", lidar)

    for n, (scenario_type, seconds) in enumerate([*TAGS, *CLOSE_TAGS, OVERLAPPING_TAG]):
        tags.append((_token("tag", n), _token("lpc", int(seconds * 20.0)), scenario_type, None))
    conn.executemany("INSERT INTO scenario_tag VALUES (?,?,?,?)", tags)

    conn.commit()
    conn.close()


@pytest.fixture
def site(tmp_path: Path) -> SiteConfig:
    """A site whose nuplan root holds one synthetic log and a non-empty blob tree."""
    root = tmp_path / "nuplan"
    split = root / "nuplan-v1.1" / "splits" / "mini"
    split.mkdir(parents=True)
    _build_log(split / f"{LOG_NAME}.db")
    # Sharded layout, which is what the nuPlan camera zips actually unpack to.
    (L.sensor_root(root) / "nuplan-v1.1_mini_camera_8" / LOG_NAME / L.FRONT_CAMERA).mkdir(parents=True)
    (L.sensor_root(root) / "nuplan-v1.1_mini_camera_8" / "LICENSE").write_text("")
    return SiteConfig(
        site="test",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={"nuplan": Dataset(name="nuplan", path=root)},
        models={},
        benchmarks={},
        gpu=Gpu(),
        source=tmp_path / "paddock.toml",
    )


def _harness(cfg: SiteConfig):
    return BENCHMARK(Registry.load().benchmarks["nuplan"], cfg)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibration_decodes_without_the_devkit() -> None:
    """The intrinsics come back as numbers on a machine with no nuplan package."""
    assert DEVKIT_MODULE not in sys.modules
    blob = _devkit_blob("CameraIntrinsic", [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])
    intrinsics = L.decode_calibration(blob).reshape(3, 3)
    assert intrinsics[0, 0] == 1545.0
    assert intrinsics[0, 2] == 960.0
    assert intrinsics[2, 2] == 1.0


def test_calibration_refuses_a_numpy_callable() -> None:
    """The escape a reviewer found: `numpy.testing.runstring` is a function whose
    body is `exec`, and a prefix match on "numpy" admitted it -- so a tampered
    blob ran arbitrary code through the class that promised it could not.

    Pinned with the real global rather than a stand-in, because the bug was
    precisely that the allowlist was a prefix and not a set of names.
    """
    import numpy.testing

    blob = pickle.dumps(_Reduces(numpy.testing.runstring, ("pass", {})), protocol=5)
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        L.decode_calibration(blob)


def test_calibration_admits_the_numpy_value_reconstructors() -> None:
    """The allowlist has to stay wide enough to decode a real blob: nuPlan's
    Translation and Rotation carry float64 scalars, not plain floats."""
    import numpy as np_

    blob = _devkit_blob("Translation", [np_.float64(1.67), np_.float64(-0.03), np_.float64(1.52)])
    assert L.decode_calibration(blob) == pytest.approx([1.67, -0.03, 1.52])


def test_calibration_refuses_an_unexpected_class() -> None:
    """A blob naming anything outside the calibration containers is rejected
    rather than executed -- these come out of a downloaded file."""
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        L.decode_calibration(pickle.dumps(__import__("os").system, protocol=5))


# ---------------------------------------------------------------------------
# sample selection
# ---------------------------------------------------------------------------


def test_manifest_is_deterministic(site: SiteConfig) -> None:
    """--limit N must pick the same N at every site, so the order is fixed."""
    harness = _harness(site)
    assert harness.manifest() == harness.manifest()


def test_well_separated_anchors_all_become_samples(site: SiteConfig) -> None:
    harness = _harness(site)
    samples = list(harness.samples())
    assert len(samples) == len(TAGS)
    assert [s.split for s in samples] == [
        # Sorted by (log, timestamp): the types interleave in time. At t=12 the
        # keyframe carries two tags and the rarer one is the label kept.
        "near_pedestrian_on_crosswalk",
        "traversing_narrow_lane",
        "near_pedestrian_on_crosswalk",
        "traversing_intersection",
        "near_pedestrian_on_crosswalk",
        "traversing_intersection",
    ]


def test_one_moment_is_scored_once_under_its_rarest_label(site: SiteConfig) -> None:
    """nuPlan tags overlap: t=12 is both `traversing_intersection` (three
    anchors) and `traversing_narrow_lane` (one). Scoring it twice would put the
    same moment in the average twice and bias the headline toward whichever
    situation happens to carry the most tags."""
    manifest = _harness(site).manifest()
    moments = [(db.name, ts) for db, _, ts in manifest]
    assert len(moments) == len(set(moments)), "each moment must appear once"
    at_twelve = [t for _, t, ts in manifest if ts == T0 + int(12.0 * 1e6)]
    assert at_twelve == ["traversing_narrow_lane"]


def test_anchors_of_one_type_are_thinned_to_distinct_moments(site: SiteConfig) -> None:
    """CLOSE_TAGS sits 50 and 100 ms after a kept anchor of the same type.

    Scoring those would be one situation counted three times -- the count would
    treble while the information did not, which is exactly the way a benchmark
    silently overstates its own weight.
    """
    kept = [ts for _, scenario_type, ts in _harness(site).manifest() if scenario_type == TAGS[0][0]]
    assert len(kept) == 3, "the two near-duplicate tags should have been dropped"
    gaps = [(b - a) / 1e6 for a, b in zip(kept, kept[1:], strict=False)]
    assert min(gaps) >= MIN_SEPARATION_S


def test_limit_is_a_prefix_of_the_full_set(site: SiteConfig) -> None:
    harness = _harness(site)
    everything = [s.sample_id for s in harness.samples()]
    assert [s.sample_id for s in harness.samples(limit=2)] == everything[:2]


def test_skip_drops_the_anchors_a_resumed_run_already_scored(site: SiteConfig) -> None:
    """The runner calls `samples(limit=..., skip=...)` on every resume.

    A harness that never grew `skip` raises TypeError on the first pull of a
    resumed sweep, and one that ignores it rescores what is already in the run.
    """
    harness = _harness(site)
    everything = [s.sample_id for s in harness.samples()]
    done = set(everything[:2])
    assert [s.sample_id for s in harness.samples(skip=done)] == everything[2:]
    # `limit` counts what is left, not the whole stream: the runner passes the
    # outstanding count, so a resumed sweep would otherwise stop short.
    assert [s.sample_id for s in harness.samples(limit=2, skip=done)] == everything[2:4]


def test_a_log_with_nothing_left_to_do_is_never_opened(site: SiteConfig, monkeypatch) -> None:
    """Filtering the manifest, not the built samples, is the point of `skip`."""
    harness = _harness(site)
    done = {s.sample_id for s in harness.samples()}
    opened: list[Path] = []
    real = L.open_log
    monkeypatch.setattr(L, "open_log", lambda db: (opened.append(db), real(db))[1])
    assert list(harness.samples(skip=done)) == []
    assert opened == [], "a database was read for a log whose anchors were all scored"


def test_each_sample_carries_a_full_history(site: SiteConfig) -> None:
    """A short history would measure a model working from one it invented."""
    sample = next(iter(_harness(site).samples()))
    assert len(sample.observations) == HISTORY_FRAMES + 1
    timestamps = [o.timestamp_us for o in sample.observations]
    assert timestamps == sorted(timestamps), "observations must be oldest-first"
    # 10 Hz, so the 48-frame history spans 4.8 s.
    assert timestamps[-1] - timestamps[0] == pytest.approx(4.8e6)


def test_observations_carry_real_calibration_and_speed(site: SiteConfig) -> None:
    scored = next(iter(_harness(site).samples())).observations[-1]
    assert scored.camera2image["CAM_FRONT"][0, 0] == 1545.0
    # Through the protocol's constants, not slot numbers: a literal here is the
    # restated contract CLAUDE.md names, and it would not catch the channels
    # being swapped.
    assert scored.canbus[CANBUS_VELOCITY] == pytest.approx(SPEED_MS)
    assert scored.canbus[CANBUS_ACCEL] == pytest.approx(ACCEL_MS2)
    assert scored.canbus[CANBUS_YAW_RATE] == pytest.approx(YAW_RATE)
    assert scored.speed == pytest.approx(SPEED_MS)
    assert scored.accel == pytest.approx(ACCEL_MS2)
    assert scored.yaw_rate == pytest.approx(YAW_RATE)
    assert scored.image_paths["CAM_FRONT"].endswith(".jpg")


def test_image_paths_resolve_through_the_camera_shard(site: SiteConfig) -> None:
    """nuPlan's camera zips each unpack into their own directory, but the
    database only knows `<log>/CAM_F0/<token>.jpg`. Resolving that against
    `sensor_blobs` directly would miss every image at a sharded site."""
    path = next(iter(_harness(site).samples())).observations[-1].image_paths["CAM_FRONT"]
    assert f"/nuplan-v1.1_mini_camera_8/{LOG_NAME}/{L.FRONT_CAMERA}/" in path


def test_a_partial_camera_copy_is_scored_but_marked_invalid(site: SiteConfig, tmp_path: Path) -> None:
    """One shard of nine is a normal state, so the run happens -- but a subset
    of the scored set must not be ranked against a full run."""
    harness = _harness(site)
    assert harness.invalid_reasons() == []

    import shutil

    shutil.rmtree(L.sensor_root(tmp_path / "nuplan") / "nuplan-v1.1_mini_camera_8" / LOG_NAME)
    partial = _harness(site)
    served, missing = partial.coverage()
    assert served == [] and missing == [LOG_NAME]
    assert "not comparable" in partial.invalid_reasons()[0]
    assert list(partial.samples()) == [], "anchors with no images are skipped, not failed"


# ---------------------------------------------------------------------------
# ground truth and scoring
# ---------------------------------------------------------------------------


def test_ground_truth_is_the_distance_actually_travelled(site: SiteConfig) -> None:
    """Driving straight at 10 m/s, the future at 0.5 s steps is 5 m apart."""
    sample = next(iter(_harness(site).samples()))
    gt = np.asarray(sample.ground_truth)
    assert gt.shape == (N_GT_POSES, 2)
    assert gt[:, 0] == pytest.approx(np.arange(1, N_GT_POSES + 1) * SPEED_MS * 0.5, abs=1e-6)
    assert gt[:, 1] == pytest.approx(np.zeros(N_GT_POSES), abs=1e-9)


def test_a_perfect_prediction_scores_zero(site: SiteConfig) -> None:
    """The golden number. A constant-velocity plan at the true speed is exactly
    right on a constant-velocity log, so every displacement metric is 0."""
    harness = _harness(site)
    sample = next(iter(harness.samples()))
    metrics = harness.score(sample, constant_velocity(SPEED_MS, n=N_GT_POSES, dt=0.5))
    assert metrics["ade"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["fde@3s"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["lateral_mae"] == pytest.approx(0.0, abs=1e-9)


def test_the_pdm_plan_terms_are_scored(site: SiteConfig) -> None:
    """`pdm_ep` and `pdm_comfort` are declared, so the gate reads them as EP and C."""
    harness = _harness(site)
    sample = next(iter(harness.samples()))
    metrics = harness.score(sample, constant_velocity(SPEED_MS, n=N_GT_POSES, dt=0.5))
    assert metrics["pdm_ep"] == pytest.approx(1.0)
    assert metrics["pdm_comfort"] == 1.0
    slow = harness.score(sample, constant_velocity(SPEED_MS / 2, n=N_GT_POSES, dt=0.5))
    assert slow["pdm_ep"] == pytest.approx(0.5)


def test_a_lateral_offset_is_measured(site: SiteConfig) -> None:
    harness = _harness(site)
    sample = next(iter(harness.samples()))
    perfect = constant_velocity(SPEED_MS, n=N_GT_POSES, dt=0.5).waypoints
    drifted = perfect + np.array([0.0, 2.0])
    metrics = harness.score(sample, type(constant_velocity(0.0))(drifted, dt=0.5))
    assert metrics["ade"] == pytest.approx(2.0, abs=1e-6)
    assert metrics["lateral_mae"] == pytest.approx(2.0, abs=1e-6)


def test_ground_truth_is_dropped_at_the_end_of_a_log() -> None:
    """Padding a truncated future would read as the car stopping, and score a
    correct prediction as though it were wrong."""
    poses = np.zeros((100, 9))
    poses[:, 0] = np.arange(100) * 1e4  # 100 Hz, so 1 s of log
    poses[:, 3] = 1.0
    assert L.ground_truth(poses, origin=0, interval_s=0.5, n=N_GT_POSES) is None


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def test_preflight_is_clean_when_everything_is_present(site: SiteConfig) -> None:
    assert _harness(site).preflight() == []


def test_preflight_reports_missing_cameras(site: SiteConfig, tmp_path: Path) -> None:
    """The 450 GB of camera blobs is a separate download from the logs, and
    having one without the other is the normal half-way state."""
    import shutil

    shutil.rmtree(L.sensor_root(tmp_path / "nuplan") / "nuplan-v1.1_mini_camera_8")
    problems = _harness(site).preflight()
    assert len(problems) == 1
    assert "holds no camera data" in problems[0]


def test_preflight_reports_an_unconfigured_dataset(tmp_path: Path) -> None:
    cfg = SiteConfig(
        site="test",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={},
        models={},
        benchmarks={},
        gpu=Gpu(),
        source=tmp_path / "paddock.toml",
    )
    assert "not configured" in _harness(cfg).preflight()[0]


def test_preflight_reports_an_empty_split(tmp_path: Path) -> None:
    root = tmp_path / "nuplan"
    (root / "nuplan-v1.1" / "sensor_blobs").mkdir(parents=True)
    cfg = SiteConfig(
        site="test",
        repos=tmp_path,
        runs=tmp_path,
        weights=tmp_path,
        datasets={"nuplan": Dataset(name="nuplan", path=root)},
        models={},
        benchmarks={},
        gpu=Gpu(),
        source=tmp_path / "paddock.toml",
    )
    assert "no mini log databases found" in _harness(cfg).preflight()[0]


def test_sample_is_the_shape_the_runner_expects(site: SiteConfig) -> None:
    sample = next(iter(_harness(site).samples()))
    assert isinstance(sample, Sample)
    assert sample.native_id["log"].startswith("2021.05.12")
    assert sample.sample_id.startswith("nuplan/mini/")
    # Scored against the metric module the rest of the repo uses, not a local copy.
    assert M.ade(np.zeros((N_GT_POSES, 2)), np.asarray(sample.ground_truth)) > 0
