"""Ingest must reproduce the numbers the original harnesses published.

Two layers:

* **Store logic** — pure, runs anywhere.
* **Golden values** — needs this site's legacy result directories, so it skips
  where they are absent. Where they are present it is the strongest check in the
  suite: paddock recomputes every aggregate from per-sample records, so agreement
  means the ported aggregation matches three independent implementations that
  produced the published tables.

The reference values come from the reports the original runs wrote, not from a
previous paddock run. Regenerating them from paddock would make this test agree
with itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from paddock.config import ConfigError, SiteConfig
from paddock.results.report import has_direction, lower_is_better
from paddock.results.store import scrub, sequence_hash, summarise

# ---------------------------------------------------------------------------
# Store logic
# ---------------------------------------------------------------------------


def test_scrub_removes_machine_paths_but_keeps_the_name() -> None:
    """results/ is tracked, so nothing written may name one machine's disk.

    The tail is kept: a run-group directory name is useful provenance.

    The offending path is assembled from parts rather than written out, because
    a literal one here would (correctly) fail `test_portability.py` -- this file
    is tracked like any other.
    """
    machine_path = "/" + "/".join(["shakaal_hdd_1", "workspace", "someone", "outputs", "runs1"])
    assert scrub({"from": machine_path, "n": 3}) == {"from": "runs1", "n": 3}

    for root in ("mnt", "home", "proj", "scratch", "datadrive"):
        assert scrub("/" + root + "/x/y/runs5") == "runs5"


def test_sequence_hash_is_order_independent_and_content_sensitive() -> None:
    """Two runs with different hashes evaluated different data, whatever their counts say."""
    assert sequence_hash(["b", "a"]) == sequence_hash(["a", "b"])
    assert sequence_hash(["a", "b"]) != sequence_hash(["a", "c"])


def test_summarise_reports_missing_rather_than_averaging_them_away() -> None:
    """A NaN usually means "the horizon was shorter than the metric asked for".

    Counting it as zero, or dropping it silently, both change what the number means.
    """
    out = summarise([1.0, 3.0, float("nan")])
    assert out["n"] == 2 and out["n_missing"] == 1
    assert out["mean"] == pytest.approx(2.0)


def test_summarise_of_nothing_is_not_zero() -> None:
    out = summarise([float("nan")])
    assert out["n"] == 0 and "mean" not in out


@pytest.mark.parametrize(
    ("metric", "lower_better"),
    [
        ("ade", True),
        ("fde@3s", True),
        ("medoid_fde@3s", True),
        ("min_ade@6", True),
        ("collision_rate", True),
        ("impact_speed", True),
        ("inference_seconds", True),
        ("ncap_score", False),
        ("epdms", False),
        ("reference_speed", False),
        ("ego_progress", False),
        ("lane_keeping", False),
        # AlpaSim: outcome rates and a fidelity distance whose names read the
        # wrong way round. An up arrow on these tells a reader the model that
        # crashed more often did better.
        ("collision_at_fault", True),
        ("offroad", True),
        ("open_loop_collision", True),
        ("dist_to_gt_trajectory", True),
        ("scene_score", False),
        ("progress", False),
        ("duration_frac_20s", False),
        # comma2k19. A miss rate and a time-below-threshold fraction both count
        # samples that went wrong; the comfort maxima are the worst value the
        # plan reaches. All nine rendered with an up arrow before they were
        # named, because no substring marker catches them and "_max" is too
        # broad to be one.
        ("miss_rate@2s", True),
        ("miss_rate@3s", True),
        ("miss_rate@4s", True),
        ("ttc_below_2s", True),
        ("comfort_lon_accel_max", True),
        ("comfort_lon_decel_max", True),
        ("comfort_lat_accel_max", True),
        ("comfort_lon_jerk_max", True),
        ("comfort_yaw_rate_max", True),
        ("pdm_style_score", False),
        ("comfort_within_bounds", False),
    ],
)
def test_metric_direction(metric: str, lower_better: bool) -> None:
    """A mis-signed metric silently inverts the leaderboard for that column."""
    assert has_direction(metric), f"{metric} is listed with a direction; it should have one"
    assert lower_is_better(metric) is lower_better


@pytest.mark.parametrize(
    "metric",
    [
        # Signed: best at zero, worse in both directions. A down arrow ranks a
        # model that thinks the lead is 40 m nearer than it is above one that
        # is exactly right.
        "lead_range_err",
        "along_track_bias",
        "yaw_rate_bias",
        # Best at a value that is not an extreme: 1.0 for a ratio of lengths,
        # and the nominal 0.68 / 0.95 for coverage. A model above a coverage
        # target is not better calibrated, it is under-confident.
        "path_length_ratio",
        "within_1sigma_lateral",
        "within_2sigma_lateral",
        # How far the model plans, not how well it plans. comma2k19 renders it
        # as a column, and an up arrow there says a 10 s planner beat a 3 s one
        # on a metric that is not a score.
        "scored_horizon_s",
    ],
)
def test_a_metric_with_no_better_direction_gets_no_arrow(metric: str) -> None:
    """An arrow is a claim. On these it is a false one, so none is drawn.

    `lower_is_better` still answers for them -- it answers for every string --
    which is why the check is a separate predicate rather than a third return
    value: the renderer asks whether to draw an arrow before it asks which way.
    """
    assert not has_direction(metric)


# ---------------------------------------------------------------------------
# Golden values, where the source data is present
# ---------------------------------------------------------------------------


def _ingest_config() -> dict:
    import tomllib

    try:
        cfg = SiteConfig.load()
    except ConfigError:
        pytest.skip("no site configuration")
    with cfg.source.open("rb") as fh:
        section = tomllib.load(fh).get("ingest", {})
    if not section:
        pytest.skip("no [ingest] sources configured at this site")
    return section


def _kitscenes_runs(model: str) -> Path:
    root = _ingest_config().get("kitscenes_runs", {}).get(model)
    if not root or not Path(root).exists():
        pytest.skip(f"kitscenes runs for {model} not present at this site")
    return Path(root)


@pytest.mark.parametrize(
    ("model", "run", "arm", "ade", "median", "fde3"),
    [
        # openpilot/eval_kitscenes/runs/full223/report.md
        ("openpilot", "full223", None, 1.599, 1.323, 2.894),
        # alpamayo1.5/eval_kitscenes/runs/full/report.md
        ("alpamayo15", "full", "nonav", 0.921, 0.696, 2.003),
        ("alpamayo15", "full", "nav", 1.002, 0.661, 2.158),
        ("alpamayo15", "full", "letterbox", 1.176, 0.774, 2.535),
    ],
)
def test_kitscenes_matches_the_published_report(
    model: str, run: str, arm: str | None, ade: float, median: float, fde3: float
) -> None:
    from paddock.results.ingest import kitscenes

    runs = kitscenes.ingest_run_dir(_kitscenes_runs(model) / run, model, site="test")
    wanted = f"legacy-{run}" + (f"-{arm}" if arm else "")
    match = next((m for r, m, _ in runs if r.run_id == wanted), None)
    assert match is not None, f"{wanted} not produced by ingest"

    assert match.n == 223
    assert match.metrics["ade"]["mean"] == pytest.approx(ade, abs=0.001)
    assert match.metrics["ade"]["median"] == pytest.approx(median, abs=0.001)
    assert match.metrics["fde@3s"]["mean"] == pytest.approx(fde3, abs=0.001)


def _kitscenes_pkl(model: str) -> Path:
    path = _ingest_config().get("kitscenes_pkl", {}).get(model)
    if not path or not Path(path).exists():
        pytest.skip(f"kitscenes pkl for {model} not present at this site")
    return Path(path)


def test_kitscenes_pkl_matches_the_published_report() -> None:
    """Drive-JEPA's row in alpamayo1.5/eval_kitscenes/runs/full/report.md.

    An independent reference in the strict sense: that table was written by the
    Alpamayo harness, which read this same pickle and carried its per-sample ADE
    through as `reference_ade`. Paddock re-aggregates from the per-sample
    records and has to land on the same four numbers.
    """
    from paddock.results.ingest import kitscenes

    runs = kitscenes.ingest_pkl(_kitscenes_pkl("drivejepa"), "drivejepa", site="test")
    assert len(runs) == 1, "one pickle is one run"
    _, metrics, _ = runs[0]

    assert metrics.n == 223
    assert metrics.metrics["ade"]["mean"] == pytest.approx(2.364, abs=0.001)
    assert metrics.metrics["fde@1s"]["mean"] == pytest.approx(1.306, abs=0.001)
    assert metrics.metrics["fde@2s"]["mean"] == pytest.approx(2.668, abs=0.001)
    assert metrics.metrics["fde@3s"]["mean"] == pytest.approx(4.199, abs=0.001)


def test_kitscenes_pkl_is_scored_on_the_same_keyframes_as_the_jsonl_runs() -> None:
    """The whole point of ingesting it: one shared list, so the rows compare.

    A matching sequence hash is the claim; equal counts are not. Checked against
    openpilot's full223, whose harness never shared code with Drive-JEPA's.
    """
    from paddock.results.ingest import kitscenes

    reference = kitscenes.ingest_run_dir(_kitscenes_runs("openpilot") / "full223", "openpilot", site="test")
    run, _, _ = kitscenes.ingest_pkl(_kitscenes_pkl("drivejepa"), "drivejepa", site="test")[0]

    assert run.dataset["sequences_sha256"] == reference[0][0].dataset["sequences_sha256"]


def test_kitscenes_pkl_records_the_route_hint() -> None:
    """Drive-JEPA got a ground-truth-derived turn command; the others did not.

    Recorded because it is a difference in what the model was told, which a
    later reader comparing these numbers needs and cannot recover from the
    metrics. 39 of 223 is the figure the report's caveats section cites.
    """
    from paddock.results.ingest import kitscenes

    run, _, _ = kitscenes.ingest_pkl(_kitscenes_pkl("drivejepa"), "drivejepa", site="test")[0]

    assert run.config["command_source"] == "gt"
    assert run.config["turn_commands"] == 39


def test_navsim_drops_all_three_aggregate_rows() -> None:
    """The CSV holds three label rows, not two. Counting one as a scenario gives n=243."""
    from paddock.results.ingest import navsim

    csv_path = _ingest_config().get("navsim_csv")
    if not csv_path or not Path(csv_path).exists():
        pytest.skip("navsim csv not present at this site")

    run, metrics, samples = navsim.ingest(Path(csv_path), "drivejepa", site="test")[0]
    assert metrics.n == 242, "expected 242 scenario tokens"
    assert not any(s["native_id"]["token"].startswith("average") for s in samples)
    assert not any(s["native_id"]["token"] == "pseudo_closed_loop" for s in samples)


def test_navsim_keeps_the_source_reported_figure() -> None:
    """The log and the CSV disagree; both are kept rather than one chosen silently."""
    from paddock.results.ingest import navsim

    csv_path = _ingest_config().get("navsim_csv")
    if not csv_path or not Path(csv_path).exists():
        pytest.skip("navsim csv not present at this site")

    _, metrics, _ = navsim.ingest(Path(csv_path), "drivejepa", site="test")[0]
    assert metrics.metrics["epdms"]["mean"] == pytest.approx(0.90673, abs=0.0001)
    assert metrics.by_group["source_reported"]["final_average_valid"] == pytest.approx(0.90488, abs=0.0001)
    # 41 of 242 rows have no two-frame comfort: a log's first frame cannot have one.
    assert metrics.metrics["two_frame_extended_comfort"]["n_missing"] == 41


def test_neuroncap_recovers_speeds_only_the_logs_recorded() -> None:
    """impact_speed and reference_speed exist nowhere in metrics.json."""
    from paddock.results.ingest import neuroncap

    outputs = _ingest_config().get("neuroncap_outputs")
    if not outputs or not Path(outputs).exists():
        pytest.skip("neuroncap outputs not present at this site")

    runs = neuroncap.ingest(Path(outputs), site="test")
    target = next((m for r, m, _ in runs if r.run_id == "legacy-runs1" and r.model == "openpilot"), None)
    assert target is not None
    # ladder_runs1.log: stationary 5.0, frontal 1.8621510453829888, side 5.0
    assert target.metrics["ncap_score"]["mean"] == pytest.approx((5.0 + 1.86215 + 5.0) / 3, abs=1e-4)
    assert not math.isnan(target.metrics["impact_speed"]["mean"])


def test_invalid_runs_are_marked_not_dropped() -> None:
    """Deleting a bad run loses the record that it happened."""
    from paddock.results.ingest import neuroncap

    outputs = _ingest_config().get("neuroncap_outputs")
    if not outputs or not Path(outputs).exists():
        pytest.skip("neuroncap outputs not present at this site")

    runs = neuroncap.ingest(Path(outputs), site="test")
    invalid = [r for r, _, _ in runs if not r.valid]
    assert invalid, "the known-bad run group should be present and flagged"
    assert all("actor was not rendered" in r.notes for r in invalid)


def test_alias_run_groups_are_not_counted_twice() -> None:
    """runs1_current is a symlink farm; following it double-counts runs1."""
    from paddock.results.ingest import neuroncap

    outputs = _ingest_config().get("neuroncap_outputs")
    if not outputs or not Path(outputs).exists():
        pytest.skip("neuroncap outputs not present at this site")

    ids = [(r.run_id, r.model) for r, _, _ in neuroncap.ingest(Path(outputs), site="test")]
    assert ("legacy-runs1_current", "openpilot") not in ids
