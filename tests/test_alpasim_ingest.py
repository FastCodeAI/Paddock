"""AlpaSim ingest must not quietly change what a published number means.

Runs against a committed two-scene fixture, so it needs no GPU, no dataset and
no AlpaSim install. The two scenes are chosen to exercise the rule that catches
people out rather than the happy path: one clean pass, and one that made real
progress and was then zeroed by an offroad event.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.alpasim.harness import scorer_sha
from paddock.results.ingest import alpasim

FIXTURE = Path(__file__).parent / "fixtures" / "alpasim_run"
RUN = FIXTURE / "alpamayo_two_scenes"


def test_driver_identity_comes_from_the_console_not_the_directory_name() -> None:
    """Nothing under aggregate/ records which driver ran.

    A model inferred from the directory name would survive a rename and be
    wrong; the version_id the runtime probed at connect is the only
    machine-readable statement of what was actually scored.
    """
    assert alpasim.driver_version_id(RUN) == "alpamayo1_5-driver-0.91.0"
    assert alpasim.model_name("alpamayo1_5-driver-0.91.0") == "alpamayo15"
    assert alpasim.model_name("constant_velocity-driver-0.2.0") == "constant_velocity"
    assert alpasim.model_name("meteor-0.1.1") == "meteor"
    assert alpasim.model_name(None) is None
    assert alpasim.model_name("something-we-have-never-seen-1.0") is None


def test_tick_config_is_read_from_yaml_structure() -> None:
    """The camera list is `- logical_id: camera_...`, not `camera_...:`.

    A regex keyed on the latter reads zero cameras and silently reports the
    group as unknown, which would let two incomparable runs be averaged.
    """
    tick = alpasim.tick_config(RUN)
    assert tick["cameras"] == 4
    assert tick["control_timestep_us"] == 100_000
    assert tick["control_hz"] == 10.0
    assert alpasim.group_key(tick) == "4cam-10Hz"


def test_group_key_refuses_to_guess() -> None:
    assert alpasim.group_key({}) == "unknown"
    assert alpasim.group_key({"cameras": 4, "control_hz": None}) == "unknown"


def test_scene_score_is_taken_per_rollout_not_from_metrics_results() -> None:
    """`metrics_results` looks complete and does not contain the score at all.

    Reading it alone drops the primary metric silently, which is why this
    asserts the values rather than merely that a key exists.
    """
    runs = alpasim.ingest(FIXTURE, site="test-site")
    assert len(runs) == 1
    run, metrics, samples = runs[0]

    assert metrics.primary_metric == "scene_score"
    assert metrics.n == 2
    scores = sorted(s["metrics"]["scene_score"] for s in samples)
    assert scores == [0.0, 1.0]
    assert metrics.metrics["scene_score"]["mean"] == 0.5


def test_a_zeroed_scene_keeps_the_progress_that_earned_nothing() -> None:
    """The hard-failure rule zeroes the scene, not the progress underneath it.

    Losing that would hide the difference between a policy that drove most of a
    scene and then went offroad, and one that never moved.
    """
    _, _, samples = alpasim.ingest(FIXTURE, site="test-site")[0]
    zeroed = [s for s in samples if s["metrics"]["scene_score"] == 0.0][0]
    assert zeroed["extra"]["failure_reason"] == "offroad"
    assert zeroed["metrics"]["progress"] > 0.9


def test_provenance_records_what_makes_runs_comparable() -> None:
    run, metrics, samples = alpasim.ingest(FIXTURE, site="test-site")[0]
    assert run.benchmark == "alpasim"
    assert run.model == "alpamayo15"
    assert run.mode == "closed_loop"
    assert run.config["group"] == "4cam-10Hz"
    # nuRec and OmniDreams are separate columns in the support matrix: a score
    # from one is not a statement about the other, and only the console says which.
    assert run.config["renderer"] == "nurec"
    # The divergence cap truncates the scored window, so it changes what every
    # fidelity number means and belongs in provenance.
    assert run.config["max_dist_to_gt_trajectory"] == 4.0
    # Read from the repos.yaml pin, not copied: a literal would keep naming the
    # old scorer after the next bump.
    assert run.config["scorer_repo_sha"] == scorer_sha()
    assert run.dataset["sequences_sha256"]
    assert metrics.by_group == {"tick_config": {"4cam-10Hz": {"n": 2}}}


@pytest.fixture
def local_zone(monkeypatch: pytest.MonkeyPatch):
    """Pin the ingesting machine's zone for the duration of a test.

    `monkeypatch` restores the environment variable, but the C library caches
    the zone until `tzset()` is called again, so the teardown has to be here.
    """

    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield _set
    time.tzset()


#: The fixture's `run_time`, as AlpaSim wrote it: bare wall clock, no zone.
_FIXTURE_RUN_TIME_UTC = {
    "Asia/Kolkata": "2026-09-07T02:43:51Z",  # +05:30
    "America/Los_Angeles": "2026-09-07T15:13:51Z",  # -07:00, September
    # In UTC the conversion is legitimately the identity. Asserting merely that
    # the result *differs* from the naive string passes at any site with an
    # offset and fails in CI, which runs in UTC -- which is exactly what this
    # test did before it was pinned to a zone.
    "UTC": "2026-09-07T08:13:51Z",
}


@pytest.mark.parametrize(("zone", "expected"), sorted(_FIXTURE_RUN_TIME_UTC.items()))
def test_local_wall_clock_is_converted_not_relabelled_as_utc(local_zone, zone: str, expected: str) -> None:
    """AlpaSim writes run_time with no zone; copying it verbatim mislabels it.

    Relabelling local time as UTC misorders a shared history by whatever the
    producing site's offset is -- five and a half hours, between these two.
    """
    local_zone(zone)
    run, _, _ = alpasim.ingest(FIXTURE, site="test-site")[0]
    assert run.started_utc == expected


def test_known_bad_runs_are_marked_not_dropped() -> None:
    """Deleting a bad run loses the record that it happened; marking it keeps it."""
    runs = alpasim.ingest(FIXTURE, site="test-site", invalid={"alpamayo_two_scenes": "actor never rendered"})
    run, _, _ = runs[0]
    assert run.valid is False
    assert run.notes == "actor never rendered"


# --- the console log has three names ----------------------------------------
#
# Driver identity decides whether a run is ingested at all, and a run whose
# driver cannot be read is skipped rather than failing. So a console filename
# the glob misses does not announce itself: the row is simply absent from the
# store. Four of the five 20 Hz rows were lost exactly this way -- the wizard
# writes `wizard_console.log` when it was given no label, and a run launched
# into a container has no wizard console at all, only `container_console.log`.


def _renamed_console(tmp_path: Path, console: str) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RUN, run)
    (run / "wizard_console_alpamayo_two_scenes.log").rename(run / console)
    return run


@pytest.mark.parametrize(
    "console",
    ["wizard_console.log", "wizard_console_labelled.log", "container_console.log"],
)
def test_driver_identity_is_found_whatever_the_console_is_called(tmp_path: Path, console: str) -> None:
    assert alpasim.driver_version_id(_renamed_console(tmp_path, console)) == "alpamayo1_5-driver-0.91.0"


def test_a_run_with_no_console_at_all_is_skipped_not_guessed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(RUN, run)
    (run / "wizard_console_alpamayo_two_scenes.log").unlink()
    assert alpasim.driver_version_id(run) is None
    assert alpasim.ingest(tmp_path, site="test-site") == []


# --- a sweep that left one directory per scene ------------------------------


#: The two scene ids the fixture's rollouts carry.
_FIXTURE_SCENES = ("02e075b9", "01d503d4")


def _scene_variant(tmp_path: Path, name: str, *scenes: str) -> Path:
    """A copy of the fixture standing in for a different pair of scenes.

    Both ids are rewritten, not one: leaving either in place makes the copy
    overlap the original, and an overlapping campaign is a different test.
    """
    run = tmp_path / name
    shutil.copytree(RUN, run)
    summary = run / "aggregate" / "results-summary.json"
    text = summary.read_text()
    for old, new in zip(_FIXTURE_SCENES, scenes or _FIXTURE_SCENES, strict=True):
        text = text.replace(old, new)
    summary.write_text(text)
    return run


def test_a_campaign_of_per_scene_directories_is_one_run(tmp_path: Path) -> None:
    """Ingested individually these are runs of n=1 and the sweep's mean is nowhere.

    `paddock report` would then rank a model on whichever single scene sorted
    first, which is not the measurement anyone made.
    """
    a = _scene_variant(tmp_path, "sweep_a")
    b = _scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "0a0a0a0a")
    run, metrics, samples = alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site")

    assert run.run_id == "legacy-sweep"
    assert run.model == "alpamayo15"
    assert metrics.n == 4
    assert run.dataset["n_samples"] == 4
    assert run.config["n_run_dirs"] == 2
    assert len({s["sample_id"] for s in samples}) == 4


def test_a_campaign_refuses_directories_that_ran_different_drivers(tmp_path: Path) -> None:
    a = _scene_variant(tmp_path, "sweep_a")
    b = _scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "0a0a0a0a")
    log = b / "wizard_console_alpamayo_two_scenes.log"
    log.write_text(log.read_text().replace("alpamayo1_5-driver-0.91.0", "constant_velocity-driver-0.2.3"))

    with pytest.raises(alpasim.Incomparable, match="different drivers"):
        alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site")


def test_a_driver_that_serves_a_trained_checkpoint_is_not_ingested(tmp_path: Path) -> None:
    """Filed by its prefix, such a run would be ranked as the released model.

    That is how the retrained Drive-JEPA's two AlpaSim runs reached the
    leaderboard as Drive-JEPA's own. Paddock keeps only the released models'
    results; a trained checkpoint's runs belong to gallops.
    """
    assert alpasim.model_name("drivejepa-v2-driver-0.1.4b") == "drivejepa"
    assert alpasim.model_name("drivejepa-v2-driver-0.1.7") is None

    run = _scene_variant(tmp_path, "retrained")
    log = run / "wizard_console_alpamayo_two_scenes.log"
    log.write_text(log.read_text().replace("alpamayo1_5-driver-0.91.0", "drivejepa-v2-driver-0.1.7"))

    assert alpasim.ingest(tmp_path, site="test-site") == []
    with pytest.raises(alpasim.Incomparable, match="not the released checkpoint"):
        alpasim.ingest_campaign([run], run_id="legacy-retrained", site="test-site")


def test_a_campaign_refuses_directories_that_ran_different_tick_configs(tmp_path: Path) -> None:
    """10 Hz control gives a policy five times the chances to correct that 2 Hz does.

    A mean across that describes no run that ever happened.
    """
    a = _scene_variant(tmp_path, "sweep_a")
    b = _scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "0a0a0a0a")
    cfg = b / "generated-user-config-0.yaml"
    cfg.write_text(cfg.read_text().replace("control_timestep_us: 100000", "control_timestep_us: 500000"))

    with pytest.raises(alpasim.Incomparable, match="different tick configs"):
        alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site")


def test_a_campaign_refuses_to_score_the_same_scene_twice(tmp_path: Path) -> None:
    """Two directories holding the same scene is a listing mistake, not a sweep.

    Silently merged it inflates n and shifts the mean toward whichever scene was
    listed twice -- a ten-scene row that is really nine scenes and a duplicate.
    """
    a = _scene_variant(tmp_path, "sweep_a")
    b = _scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "01d503d4")

    with pytest.raises(alpasim.Incomparable, match="same scene"):
        alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site")


def _unsuited(run: Path) -> Path:
    """What a sweep driven by explicit scene ids leaves: no test_suite_id."""
    meta = run / "run_metadata.yaml"
    meta.write_text(meta.read_text().replace("test_suite_id: public_2601", "test_suite_id: null"))
    return run


def test_a_campaign_driven_by_scene_ids_takes_its_split_from_the_config(tmp_path: Path) -> None:
    """`split: null` beside `n_samples: 10` reads as an unknown subset."""
    a = _unsuited(_scene_variant(tmp_path, "sweep_a"))
    b = _unsuited(_scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "0a0a0a0a"))

    run, _, _ = alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site")
    assert run.dataset["split"] == ""

    run, _, _ = alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site", split="public_2601")
    assert run.dataset["split"] == "public_2601"


def test_a_campaign_refuses_a_split_its_directories_contradict(tmp_path: Path) -> None:
    a = _scene_variant(tmp_path, "sweep_a")
    b = _scene_variant(tmp_path, "sweep_b", "0b0b0b0b", "0a0a0a0a")

    with pytest.raises(alpasim.Incomparable, match="split"):
        alpasim.ingest_campaign([a, b], run_id="legacy-sweep", site="test-site", split="public_2602")
