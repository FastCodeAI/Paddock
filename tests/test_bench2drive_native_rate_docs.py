"""The native-rate tables must still be what the records say.

Sibling of `test_bench2drive_docs.py`, same contract, one campaign later: every
published cell is recomputed here from tracked files and fails on drift in
either direction. Three things are guarded:

* **The per-route JSON agrees with the ingested run.** `bench2drive-native-rate.json`
  is the per-route detail the ingest does not keep (status, infraction
  breakdown). If it and `results/bench2drive/<model>/native50-*/metrics.json`
  ever disagreed, the doc would be built on a file nothing checks. They come
  from the same `results.json` attempts, so their means must match.
* **Every cell of the doc and the README block reproduces.** The 20 Hz column is
  recomputed from the *published* records in
  `bench2drive-speed-normalization.json`, restricted to the routes scored at
  both rates -- which is the only comparison the 49-route runs support.
* **Route 3800 is recorded as attempted and never scored, never as a zero.** Its
  two runs must stay `partial` with `n_failed: 1`, and the route must be in
  `routes_subset` (requested) and absent from the records (not measured).

No GPU, no dataset, no site config: this reads tracked files only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE = REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.json"
TWENTY = REPO_ROOT / "docs/benchmarks/bench2drive-speed-normalization.json"
DOC = REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.md"
README = REPO_ROOT / "README.md"

#: model -> (native rate in Hz, tracked native run id, README row label).
CAMPAIGN = {
    "drivejepa": (2, "native50-2hz-40c29d24c37d542a-447e23", "DriveJEPA"),
    "alpamayo15": (10, "native50-10hz-084418258d833a9f-c4d076", "Alpamayo 1.5 (nav)"),
    "qwendrive": (2, "native50-2hz-084418258d833a9f-9b2bc9", "QwenDrive ‡"),
}
#: The models whose route 3800 never scored, and what their record must say.
UNSCORABLE_3800 = ("alpamayo15", "qwendrive")

#: `min_speed_infractions` is a per-checkpoint speed-ratio log, not a violation
#: (see bench2drive-speed-normalization.md), so it is not in any total.
TOTALLED = (
    "collisions_pedestrian",
    "collisions_vehicle",
    "collisions_layout",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "yield_emergency_vehicle_infractions",
    "route_timeout",
    "scenario_timeouts",
)
#: Doc row label -> infraction key, for the breakdown rows of the per-model tables.
BREAKDOWN = {
    "Pedestrian collision": "collisions_pedestrian",
    "Vehicle collision": "collisions_vehicle",
    "Layout collision": "collisions_layout",
    "Red light": "red_light",
    "Stop sign": "stop_infraction",
    "Outside lanes": "outside_route_lanes",
    "Route deviation": "route_dev",
    "Blocked / stalled": "vehicle_blocked",
    "Emergency yield": "yield_emergency_vehicle_infractions",
}


@pytest.fixture(scope="module")
def native() -> dict:
    return json.loads(NATIVE.read_text())


@pytest.fixture(scope="module")
def twenty() -> dict:
    return json.loads(TWENTY.read_text())


def _run_dir(model: str) -> Path:
    return REPO_ROOT / "results/bench2drive" / model / CAMPAIGN[model][1]


def _summarise(records: list[dict]) -> dict:
    """The published quantities, from per-route records alone."""
    n = len(records)
    counts = {k: sum(r["infractions"][k] for r in records) for k in BREAKDOWN.values()}
    return {
        "n": n,
        "driving_score": sum(r["scores"]["score_composed"] for r in records) / n,
        "route_completion": sum(r["scores"]["score_route"] for r in records) / n,
        "infraction_penalty": sum(r["scores"]["score_penalty"] for r in records) / n,
        "routes_completed": sum(1 for r in records if str(r["status"]).startswith("Completed")),
        "counts": counts,
        "total": sum(sum(r["infractions"][k] for r in records) for k in TOTALLED),
        "min_speed": sum(r["infractions"]["min_speed_infractions"] for r in records),
    }


def _pair(native: dict, twenty: dict, model: str) -> tuple[dict, dict]:
    """(20 Hz, native) summaries over the routes scored at BOTH rates.

    The restriction is the whole point for the 49-route runs: comparing a
    49-route native mean against a published all-50 mean would attribute one
    route's absence to the rate change.
    """
    nat = [r for r in native["records"] if r["model"] == model]
    scored = {r["route_id"] for r in nat}
    old = [r for r in twenty["records"] if r["model"] == model and r["route_id"] in scored]
    assert len(old) == len(nat), f"{model}: {len(old)} 20 Hz records for {len(nat)} native routes"
    return _summarise(old), _summarise(nat)


def _tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in a document, as lists of stripped cells."""
    tables, current = [], []
    for line in text.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _doc_table(heading: str) -> dict[str, list[str]]:
    """The first table under a doc heading, keyed by its first column."""
    lines = DOC.read_text().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == heading)
    rows: dict[str, list[str]] = {}
    for line in lines[start:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows[cells[0]] = cells
        elif rows:
            break
    return rows


def _readme_rows() -> dict[str, list[str]]:
    lines = README.read_text().split("\n")
    start = next(
        i for i, ln in enumerate(lines) if ln.strip() == "#### Native inference rate — three models re-driven"
    )
    rows: dict[str, list[str]] = {}
    for line in lines[start:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows[cells[0]] = cells
        elif rows:
            break
    return rows


def _number(cell: str) -> float:
    """A table cell's number, ignoring bold markers and the unicode minus."""
    return float(cell.replace("*", "").replace("−", "-").strip())


# ---------------------------------------------------------------------------
# The per-route file and the ingested run describe the same runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_per_route_records_reproduce_the_ingested_metrics(native: dict, model: str) -> None:
    """`bench2drive-native-rate.json` must not drift from the run it documents."""
    metrics = json.loads((_run_dir(model) / "metrics.json").read_text())
    records = [r for r in native["records"] if r["model"] == model]
    got = _summarise(records)
    assert got["n"] == metrics["n"]
    for name, key in (
        ("driving_score", "driving_score"),
        ("route_completion", "route_completion"),
        ("infraction_penalty", "infraction_penalty"),
    ):
        assert round(got[key], 3) == round(metrics["metrics"][name]["mean"], 3), name


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_every_record_names_the_run_and_the_file_it_came_from(native: dict, model: str) -> None:
    hz, run_id, _ = CAMPAIGN[model]
    records = [r for r in native["records"] if r["model"] == model]
    assert records
    for rec in records:
        assert rec["rate_hz"] == hz
        assert rec["run_id"] == run_id
        assert re.fullmatch(r"[0-9a-f]{64}", rec["source_sha256"]), rec["route_id"]
        assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{6}", rec["source_run_id"]), rec["route_id"]


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_the_rate_is_in_the_ingested_config(model: str) -> None:
    """Without this the native run and the 20 Hz run hash to the same config."""
    hz, run_id, _ = CAMPAIGN[model]
    run = json.loads((_run_dir(model) / "run.json").read_text())
    assert run["config"]["model_env"]["PADDOCK_INFER_HZ"] == str(hz)
    assert run["run_id"] == run_id
    # Same routes and same waypoint_dt as the 20 Hz run, or the comparison is
    # measuring two things at once.
    pooled = next((REPO_ROOT / "results/bench2drive" / model).glob("pooled50-*/run.json"))
    old = json.loads(pooled.read_text())
    assert run["config"]["routes_subset"] == old["config"]["routes_subset"]
    assert run["config"]["routes_file"] == old["config"]["routes_file"]
    assert run["config"]["waypoint_dt"] == old["config"]["waypoint_dt"]
    assert "PADDOCK_INFER_HZ" not in old["config"]["model_env"]
    assert run["run_id"] != old["run_id"]


# ---------------------------------------------------------------------------
# Route 3800: attempted, never scored, never a zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", UNSCORABLE_3800)
def test_route_3800_is_requested_and_reported_lost(native: dict, model: str) -> None:
    run = json.loads((_run_dir(model) / "run.json").read_text())
    assert "3800" in run["config"]["routes_subset"], "the route was requested"
    assert run["dataset"]["n_expected"] == 50
    assert run["dataset"]["n_samples"] == 49
    assert run["dataset"]["n_failed"] == 1
    assert run["status"] == "partial"
    assert "3800" in run["notes"]
    scored = {r["route_id"] for r in native["records"] if r["model"] == model}
    assert "3800" not in scored, "an unscorable route must not carry a score, not even 0"
    assert "3800" in native["unscorable"][model]


def test_route_3800_scored_for_drivejepa(native: dict) -> None:
    """What makes 3800 a model/route pairing rather than a broken route."""
    rec = next(r for r in native["records"] if r["model"] == "drivejepa" and r["route_id"] == "3800")
    assert rec["scores"]["score_composed"] == 100.0
    assert "scores **100**" in DOC.read_text()


def test_the_doc_states_the_denominator_it_used() -> None:
    text = DOC.read_text()
    assert "scored at both rates" in text
    assert "49/50" in text and "50/50" in text


# ---------------------------------------------------------------------------
# The published tables reproduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_doc_per_model_table_reproduces(native: dict, twenty: dict, model: str) -> None:
    hz, _, _ = CAMPAIGN[model]
    old, new = _pair(native, twenty, model)
    label = {"drivejepa": "DriveJEPA", "alpamayo15": "Alpamayo 1.5 (nav)", "qwendrive": "QwenDrive"}[model]
    rows = _doc_table(f"### {label} — 20 Hz → {hz} Hz, {new['n']}/50 routes")

    for row_label, key, places in (
        ("Driving score ↑", "driving_score", 2),
        ("Route completion ↑ (%)", "route_completion", 2),
        ("Infraction penalty ↑", "infraction_penalty", 3),
    ):
        row = rows[row_label]
        assert _number(row[1]) == round(old[key], places), (model, row_label, "20 Hz")
        assert _number(row[2]) == round(new[key], places), (model, row_label, "native")
        assert _number(row[3]) == round(round(new[key], places) - round(old[key], places), places), (
            model,
            row_label,
            "delta",
        )

    for row_label, key in (("Routes completed ↑", "routes_completed"), ("Total infractions ↓", "total")):
        row = rows[row_label]
        assert int(_number(row[1])) == old[key], (model, row_label, "20 Hz")
        assert int(_number(row[2])) == new[key], (model, row_label, "native")
        assert int(_number(row[3])) == new[key] - old[key], (model, row_label, "delta")

    published = {lbl: key for lbl, key in BREAKDOWN.items() if f"— {lbl} ↓" in rows}
    for lbl, key in published.items():
        row = rows[f"— {lbl} ↓"]
        assert int(_number(row[1])) == old["counts"][key], (model, lbl, "20 Hz")
        assert int(_number(row[2])) == new["counts"][key], (model, lbl, "native")
    # Every non-zero category is published: a breakdown that silently drops one
    # is how a total stops adding up.
    for lbl, key in BREAKDOWN.items():
        if old["counts"][key] or new["counts"][key]:
            assert lbl in published, f"{model}: {lbl} is non-zero and not in the table"

    ms = rows["Min-speed entries (not a violation)"]
    assert int(_number(ms[1])) == old["min_speed"] and int(_number(ms[2])) == new["min_speed"]


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_readme_block_reproduces(native: dict, twenty: dict, model: str) -> None:
    hz, _, label = CAMPAIGN[model]
    old, new = _pair(native, twenty, model)
    row = _readme_rows()[label]
    assert row[1] == f"{hz} Hz"
    assert row[2] == f"{new['n']}/50"
    assert _number(row[3]) == round(old["driving_score"], 2)
    assert _number(row[4]) == round(new["driving_score"], 2)
    assert _number(row[5]) == round(old["route_completion"], 2)
    assert _number(row[6]) == round(new["route_completion"], 2)
    assert row[7] == f"{old['routes_completed']} → {new['routes_completed']}"


def test_readme_block_lists_exactly_the_models_that_were_re_run() -> None:
    """OpenPilot and VisionPilot were not re-run; they must not appear as rows."""
    rows = _readme_rows()
    assert set(rows) - {"Model"} == {label for _, _, label in CAMPAIGN.values()}


def test_readme_does_not_pool_openpilot_into_this_block() -> None:
    """The by-recording-set block is upstream of here and stays untouched."""
    readme = README.read_text()
    start = readme.index("#### Native inference rate")
    end = readme.index("[Calculation](docs/benchmarks/bench2drive-speed-normalization.md)", start)
    block = readme[start:end]
    assert "OpenPilot's native rate is\n20 Hz" in block
    # No DS/RC number is published for openpilot anywhere in the new block.
    assert not re.search(r"\|\s*OpenPilot", block)


# ---------------------------------------------------------------------------
# The exposure caveat is stated where the numbers are
# ---------------------------------------------------------------------------


def test_the_exposure_caveat_is_published_in_both_places() -> None:
    """A score that rose because the car drove further must say so."""
    assert "partly from more exposure" in DOC.read_text()
    assert "partly more exposure" in README.read_text()


def test_doc_keeps_verified_and_argued_apart() -> None:
    """docs/reviewing.md: the separation is the point of the page."""
    text = DOC.read_text()
    assert "## Verified, and argued" in text
    strengths = _tables(text)[-1]
    assert strengths[0][-1] == "Strength"
    body = strengths[1:]
    assert any("**verified**" in r[-1] for r in body)
    assert any("**argued**" in r[-1] for r in body)
    assert any("**unverified**" in r[-1] or "not claimed" in r[-1] for r in body)


def test_recordings_are_named_outside_the_repo() -> None:
    """No video or frame from this campaign is tracked; the doc says where they are."""
    text = DOC.read_text()
    assert "bench2drive-diagnostics/native-50" in text
    assert "Bench2drive/Simulation/native50" in text
    assert not (REPO_ROOT / "docs/benchmarks/native-50").exists()
