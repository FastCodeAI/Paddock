"""The published Bench2Drive tables must still be what the records say.

Two things are guarded here, both of which a reviewer had to recompute by hand
on PR #21:

* **Every cell reproduces from the committed records.** The tables in
  `docs/benchmarks/bench2drive.md` (moved there from the README) are
  transcribed by a human from `bench2drive-speed-normalization.json`; this
  recomputes them and fails on any drift, in either direction.
* **OpenPilot's two recording sets are never pooled into one mean.** Its 24
  historical routes were driven with human resume assistance on 16 of them, and
  its 26 expansion routes use a different upstream longitudinal setup. A pooled
  driving score of 48.11 would top the table as though one setup had driven it;
  the support matrix publishes a measured record of one setup instead.

No GPU, no dataset, no site config: this reads tracked files only.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import pytest

from paddock.adapters.benchmarks.bench2drive.harness import every_tick
from paddock.results.ingest.bench2drive import _driven_off_protocol
from tools.readme_figures import load_matrix, matrix_cell, rank

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS = REPO_ROOT / "docs/benchmarks/bench2drive-speed-normalization.json"
UNASSISTED_CSV = REPO_ROOT / "docs/benchmarks/openpilot-24-infractions.csv"
#: The page that holds the Bench2Drive tables; the README keeps only the matrix.
PAGE = REPO_ROOT / "docs/benchmarks/bench2drive.md"

# README label -> key in the records file.
PUBLISHED = {
    "DriveJEPA": "drivejepa",
    "VisionPilot": "visionpilot",
    "Alpamayo 1.5 (nav)": "alpamayo15",
    "OpenPilot †": "openpilot",
    "QwenDrive ‡": "qwendrive",
    "CV (fixed 3 m/s)": "constant_velocity_3mps",
    "CV (fixed 6 m/s)": "constant_velocity_6mps",
    "CV (fixed 9 m/s)": "constant_velocity_9mps",
    "CV (fixed 12 m/s)": "constant_velocity_12mps",
    "CV (fixed 15 m/s)": "constant_velocity_15mps",
}
BLANK = "—"


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(RECORDS.read_text())


def _matrix_cell(model: str) -> str:
    """The support matrix's Bench2Drive number for one model, as docs/support-matrix.toml gives it."""
    return matrix_cell(model, "bench2drive").shown


def _readme_table(after: str) -> list[list[str]]:
    """The first markdown table following a heading or marker line on the Bench2Drive page."""
    lines = PAGE.read_text().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == after)
    rows = []
    for ln in lines[start:]:
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif rows:
            break
    return rows


# ---------------------------------------------------------------------------
# OpenPilot is never pooled
# ---------------------------------------------------------------------------


def test_pooled_openpilot_means_are_withheld(data: dict) -> None:
    pooled = data["summaries"]["all50"]["openpilot"]
    for key in ("driving_score", "route_completion", "below_100_percent"):
        assert pooled[key] is None, (
            f"summaries.all50.openpilot.{key} pools 16 assisted recordings with a "
            "different upstream longitudinal setup; it must stay withheld"
        )
    assert "pooled_means_withheld" in pooled
    # The cohorts it was withheld in favour of are still published.
    assert data["summaries"]["original24"]["openpilot"]["driving_score"] is not None
    assert data["summaries"]["expansion26"]["openpilot"]["driving_score"] is not None


def test_readme_publishes_no_pooled_openpilot_score() -> None:
    rows = _readme_table("### Bench2Drive — 50 routes")
    row = next(r for r in rows if r[0] == "OpenPilot †")
    assert row[2] == BLANK and row[3] == BLANK, (
        "the 50-route table must not rank OpenPilot: its DS and RC cells are blank "
        "and the scores live in the by-recording-set block"
    )


def test_every_record_carries_an_assistance_label(data: dict) -> None:
    assert len(data["records"]) == 50 * 10, "50 routes for each published system"
    for rec in data["records"]:
        block = rec["assistance"]
        assert isinstance(block["assisted"], bool)
        assert block["recorded_measurement"] in {
            "observed_no_intervention",
            "observed_with_assistance",
        }
    assisted = [r for r in data["records"] if r["assistance"]["assisted"]]
    assert len(assisted) == 16
    assert {(r["model"], r["cohort"]) for r in assisted} == {("openpilot", "original24")}


def test_unassisted_subset_is_recomputable_from_the_linked_file(data: dict) -> None:
    """The 30.09 unassisted estimate must fall out of the records themselves."""
    est = [
        r["assistance"]["unassisted_prefix_estimate"]["driving_score"]
        if r["assistance"]["assisted"]
        else r["scores"]["score_composed"]
        for r in data["records"]
        if r["model"] == "openpilot" and r["cohort"] == "original24"
    ]
    assert len(est) == 24
    assert round(sum(est) / len(est), 2) == 30.09

    csv_rows = list(csv.DictReader(UNASSISTED_CSV.open()))
    from_csv = sum(float(r["driving_score"]) for r in csv_rows) / len(csv_rows)
    assert round(from_csv, 2) == 30.09
    summary = data["summaries"]["openpilot_unassisted24_estimate"]
    assert round(summary["driving_score"], 2) == 30.09


# ---------------------------------------------------------------------------
# The published tables reproduce from the records
# ---------------------------------------------------------------------------


def test_fifty_route_table_reproduces(data: dict) -> None:
    counts = [
        "collisions_pedestrian",
        "collisions_vehicle",
        "collisions_layout",
        "red_light",
        "stop_infraction",
        "outside_route_lanes",
        "route_dev",
        "vehicle_blocked",
        "yield_emergency_vehicle_infractions",
        "min_speed_infractions",
        "route_timeout",
        "scenario_timeouts",
    ]
    rows = {r[0]: r for r in _readme_table("### Bench2Drive — 50 routes")[1:]}
    assert set(rows) == set(PUBLISHED)
    for label, key in PUBLISHED.items():
        s = data["summaries"]["all50"][key]
        row = rows[label]
        assert row[1] == f"{s['n']}/50"
        if s["driving_score"] is not None:
            assert row[2] == f"{s['driving_score']:.2f}"
            assert row[3] == f"{s['route_completion']:.2f}"
        # The trailing columns are the twelve infraction counts, then the two
        # collision totals — stable however many descriptive columns precede them.
        for cell, name in zip(row[-14:-2], counts, strict=True):
            assert cell == str(s["counts"][name]), f"{label} {name}"
        assert row[-2] == str(s["collision_entries"]), label
        assert row[-1] == f"{s['collision_routes']}/50", label


def test_progress_km_column_reproduces(data: dict) -> None:
    """The denominator has to be visible, and right."""
    rows = {r[0]: r for r in _readme_table("### Bench2Drive — 50 routes")[1:]}
    for label, key in PUBLISHED.items():
        s = data["summaries"]["all50"][key]
        assert rows[label][4] == f"{s['route_progress_km']:.2f}", label


def test_collisions_per_km_table_reproduces(data: dict) -> None:
    rows = {r[0]: r for r in _readme_table("#### Collisions per progress km")[1:]}
    assert "OpenPilot †" not in rows, "OpenPilot's two recording sets are not pooled into a rate"
    for label, row in rows.items():
        s = data["summaries"]["all50"][PUBLISHED[label]]
        km = s["route_progress_km"]
        assert row[1] == f"{km:.2f}", label
        for cell, name in zip(
            row[2:5],
            ("collisions_pedestrian", "collisions_vehicle", "collisions_layout"),
            strict=True,
        ):
            assert cell == f"{s['counts'][name] / km:.2f}", f"{label} {name}"
        assert row[5] == f"{s['collision_entries'] / km:.2f}", label


def test_entries_per_km_is_not_published(data: dict) -> None:
    """It reduces to the checkpoint grid: 20 / completion-weighted mean route length."""
    spread = []
    for key in PUBLISHED.values():
        recs = [r for r in data["records"] if r["model"] == key]
        weight = sum(r["scores"]["score_route"] / 100 for r in recs)
        weighted_len_km = sum(r["route_length_m"] * r["scores"]["score_route"] / 100 for r in recs)
        spread.append(20 / (weighted_len_km / weight / 1000))
    # The claim is comparative, so assert it that way rather than against a bare
    # threshold: over the same rows the derived quantity varies by a few percent
    # while `below_100_percent`, the column that does carry the speed signal,
    # varies several-fold. (Ten published rows now, so the flat 2% of PR #21 --
    # five rows, 1.1% -- no longer states anything about the set being published.)
    signal = [
        s["below_100_percent"]
        for s in (data["summaries"]["all50"][k] for k in PUBLISHED.values())
        if s["below_100_percent"] is not None
    ]
    derived_ratio = max(spread) / min(spread)
    signal_ratio = max(signal) / min(signal)
    assert derived_ratio - 1 < 0.05, spread
    assert signal_ratio > 10 * (derived_ratio - 1), (derived_ratio, signal_ratio)
    table = _readme_table("#### Minimum-speed normalization")
    assert not any("Entries / progress km" in c for c in table[0])


def test_minimum_speed_table_reproduces(data: dict) -> None:
    rows = {r[0]: r for r in _readme_table("#### Minimum-speed normalization")[1:]}
    assert "OpenPilot †" not in rows, "OpenPilot's entries belong in its own block"
    for label, row in rows.items():
        s = data["summaries"]["all50"][PUBLISHED[label]]
        assert row[1] == str(s["counts"]["min_speed_infractions"]), label
        assert row[-1] == f"{s['below_100_percent']:.2f}", label


def test_openpilot_block_reproduces(data: dict) -> None:
    rows = _readme_table("#### OpenPilot, by recording set")[1:]
    assert len(rows) == 3
    for row, cohort in zip(rows[:2], ("original24", "expansion26"), strict=True):
        s = data["summaries"][cohort]["openpilot"]
        assert row[1] == str(s["n"])
        assert row[2] == f"{s['driving_score']:.2f}"
        assert row[3] == f"{s['route_completion']:.2f}"
        assert row[4] == str(s["counts"]["min_speed_infractions"])
        assert row[5] == f"{s['below_100_percent']:.2f}"
    est = data["summaries"]["openpilot_unassisted24_estimate"]
    assert rows[2][2] == f"{est['driving_score']:.2f}"
    assert rows[2][3] == f"{est['route_completion']:.2f}"


def test_support_matrix_openpilot_cell_is_one_declared_setup() -> None:
    """The cell is a measured record of one setup, never a mean pooled over setups.

    The recordings in the records file pool three setups, which is why their mean
    stays withheld above, and the unassisted estimate rebuilt from them is a
    reconstruction. The cell names one valid run: openpilot's own planner and its
    resume nudge, declared in model.yaml, on every route. So it carries P (the
    nudge stands in for a driver pressing resume). Its routes can end early, and it
    ranks all the same (the last test below).
    """
    columns, rows = load_matrix()
    j = next(j for j, c in enumerate(columns) if c.key == "bench2drive")
    row = next(r for r in rows if r.key == "openpilot")
    cell = row.cells[j]
    assert cell.run, f"the cell shows {cell.shown!r} with no run: a reconstruction, not a measurement"
    run = json.loads((REPO_ROOT / "results" / cell.run / "run.json").read_text())
    assert run["valid"], f"{cell.run} does not rank: {run['notes']}"
    assert "recording_setups" not in run["config"], f"{cell.run} pools setups"
    assert run["config"]["declared_env"]["PADDOCK_OPENPILOT_NATIVE_LONG"] == "1"
    assert "P" in {*row.marks, *cell.marks}, f"{cell.run} needs P"


# ---------------------------------------------------------------------------
# Run artifacts stay outside the repo
# ---------------------------------------------------------------------------

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def test_no_video_is_tracked() -> None:
    """`.gitignore`: run artifacts live outside the repo, and there is no LFS here."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    offenders = [p for p in tracked if p and Path(p).suffix.lower() in VIDEO_SUFFIXES]
    assert not offenders, (
        f"{offenders} are run artifacts; move them under [roots.runs] and keep the "
        "manifest, as docs/evidence/driving-videos/README.md does"
    )


def test_video_manifest_points_outside_the_repo() -> None:
    doc = (REPO_ROOT / "docs/evidence/driving-videos/README.md").read_text()
    manifest = json.loads((REPO_ROOT / "docs/evidence/driving-videos/manifest.json").read_text())
    for name in manifest["video_sha256"]:
        assert f"{{runs}}/bench2drive/evidence/driving-videos/{name}" in doc, name
        assert not (REPO_ROOT / "docs/evidence/driving-videos" / name).exists()


# ---------------------------------------------------------------------------
# The constant-velocity baseline is a sweep, and the README says which row is which
# ---------------------------------------------------------------------------


def test_speed_sweep_reproduces(data: dict) -> None:
    rows = _readme_table("#### Constant velocity — the speed sweep")[1:]
    assert [r[0] for r in rows] == ["3", "6", "9", "12", "15"]
    for row in rows:
        s = data["summaries"]["all50"][f"constant_velocity_{row[0]}mps"]
        assert row[1] == f"{s['n']}/50"
        assert row[2] == f"{s['driving_score']:.2f}", row[0]
        assert row[3] == f"{s['route_completion']:.2f}", row[0]


def test_support_matrix_baseline_cell_is_the_all50_three_mps(data: dict) -> None:
    """♣ 48.31 and the table's 40.39 are different experiments; both must be stated."""
    s = data["summaries"]["all50"]["constant_velocity_3mps"]
    assert f"**{s['driving_score']:.2f} ♣**" in PAGE.read_text()
    cell = _matrix_cell("constant_velocity")
    assert cell == f"{s['driving_score']:.2f}", cell


def test_bench2drive_page_still_links_a_tracked_run() -> None:
    """The section must not be the one that drops the last results/ link."""
    links = re.findall(r"\((\.\./\.\./results/bench2drive/[^)]+)\)", PAGE.read_text())
    assert links, "no docs link a results/bench2drive run any more"
    for link in links:
        assert (PAGE.parent / link).resolve().exists(), link


# ---------------------------------------------------------------------------
# A cell asked at its own rate, or whose routes end early, ranks and says so
# ---------------------------------------------------------------------------


def test_a_cell_asked_at_its_own_rate_or_stopped_early_ranks_and_says_so() -> None:
    """A valid record asked less often than every tick, or whose routes an early stop could end, ranks.

    Its rate is the one its model.yaml declares, and a route the stuck or over-time rule
    ends is a failed drive (early_stop.py), so neither leaves a valid run unranked. The
    rest of this column is asked every tick and drives every route to Bench2Drive's own
    end, and the README says a cell without a mark was measured the same way as the rest
    of its column unless a note says how it differs. So such a cell says how it was
    measured: its rate under the number, and its model named in the README's note on
    routes that end early. The rate is read as the ingest reads it: `infer_hz`, or for
    runs that predate it, PADDOCK_INFER_HZ in `model_env`.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    start = readme.index("**Ranked, though some Bench2Drive routes end early.**")
    early = readme[start : readme.index("\n- ", start)]
    columns, rows = load_matrix()
    rank(columns, rows)
    j = next(j for j, c in enumerate(columns) if c.key == "bench2drive")
    own_rated, stopped = set(), set()
    for row in rows:
        cell = row.cells[j]
        if not cell.run:
            continue
        run = json.loads((REPO_ROOT / "results" / cell.run / "run.json").read_text())
        config = run["config"]
        hz = config.get("infer_hz")
        if hz is None:
            hz = _driven_off_protocol(config.get("model_env") or {})
        own_rate = not every_tick(hz or 0.0)
        stopped_early = bool(config.get("early_stop"))
        # A run that is not valid (the collision rule, a stop by hand) must not rank;
        # tools/readme_figures.py refuses to rank it.
        if not run["valid"] or not (own_rate or stopped_early):
            continue
        assert "R" not in row.marks + cell.marks, f"{row.key}: {cell.run} is marked R"
        assert cell.rank is not None, f"{row.key}: {cell.run} is not ranked"
        if own_rate:
            own_rated.add(row.key)
            assert cell.qualifier == f"{hz:g} Hz", f"{row.key}: its rate is not named"
        if stopped_early:
            stopped.add(row.key)
            assert row.name in early, f"{row.key}: the README does not say its routes end early"
    # The cells this rule is about; a loop that matched none of them would pass anything.
    own_rate_cells = {"drivejepa", "alpamayo15", "qwendrive", "meteor", "autoe2e"}
    assert own_rate_cells <= own_rated, f"own-rate cells checked: {sorted(own_rated)}"
    assert {"openpilot", "meteor", "autoe2e"} <= stopped, f"early-stopped cells checked: {sorted(stopped)}"
