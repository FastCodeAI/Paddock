"""The published Bench2Drive tables must still be what the records say.

Two things are guarded here, both of which a reviewer had to recompute by hand
on PR #21:

* **Every cell reproduces from the committed records.** The README tables are
  transcribed by a human from `bench2drive-speed-normalization.json`; this
  recomputes them and fails on any drift, in either direction.
* **OpenPilot's two recording sets are never pooled into one mean.** Its 24
  historical routes were driven with human resume assistance on 16 of them, and
  its 26 expansion routes use a different upstream longitudinal setup. A pooled
  driving score of 48.11 would top the table while the support matrix publishes
  30.09 for the same model on the same benchmark.

No GPU, no dataset, no site config: this reads tracked files only.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDS = REPO_ROOT / "docs/benchmarks/bench2drive-speed-normalization.json"
UNASSISTED_CSV = REPO_ROOT / "docs/benchmarks/openpilot-24-infractions.csv"
README = REPO_ROOT / "README.md"

# README label -> key in the records file.
PUBLISHED = {
    "DriveJEPA": "drivejepa",
    "VisionPilot": "visionpilot",
    "Alpamayo 1.5 (nav)": "alpamayo15",
    "OpenPilot †": "openpilot",
    "CV (fixed 6 m/s)": "constant_velocity_6mps",
}
BLANK = "—"


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(RECORDS.read_text())


def _readme_table(after: str) -> list[list[str]]:
    """The first markdown table following a heading or marker line."""
    lines = README.read_text().split("\n")
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
    assert len(data["records"]) == 450
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
    """The 30.09 in the support matrix must fall out of the records themselves."""
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


def test_support_matrix_openpilot_cell_is_the_unassisted_estimate(data: dict) -> None:
    est = data["summaries"]["openpilot_unassisted24_estimate"]["driving_score"]
    line = next(ln for ln in README.read_text().split("\n") if ln.startswith("| OpenPilot |"))
    cell = [c.strip() for c in line.strip().strip("|").split("|")][5]
    assert re.match(rf"^{est:.2f}\b", cell), (
        f"the support matrix publishes {cell!r} for Bench2Drive; it must stay the "
        "unassisted estimate, not a pooled mean"
    )
