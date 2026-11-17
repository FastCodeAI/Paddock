"""The native-rate tables must still be what the records say.

Sibling of `test_bench2drive_docs.py`, same contract, one campaign later: every
published cell is recomputed here from tracked files and fails on drift in
either direction. Three things are guarded:

* **The per-route JSON agrees with the ingested run.** `bench2drive-native-rate.json`
  is the per-route detail the ingest does not keep (status, infraction
  breakdown). If it and `results/bench2drive/<model>/native50-*/metrics.json`
  ever disagreed, the doc would be built on a file nothing checks. They come
  from the same `results.json` attempts, so their means must match.
* **Every cell of the doc and of the block on the Bench2Drive page reproduces.** The 20 Hz column is
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

from tools.readme_figures import load_matrix, matrix_cell, rank

REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE = REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.json"
TWENTY = REPO_ROOT / "docs/benchmarks/bench2drive-speed-normalization.json"
DOC = REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.md"
README = REPO_ROOT / "README.md"
#: The page that holds the per-benchmark tables moved out of the README.
PAGE = REPO_ROOT / "docs/benchmarks/bench2drive.md"

#: model -> (native rate in Hz, tracked native run id, README row label).
CAMPAIGN = {
    "drivejepa": (2, "native50-2hz-40c29d24c37d542a-447e23", "DriveJEPA"),
    "alpamayo15": (10, "native50-10hz-084418258d833a9f-c4d076", "Alpamayo 1.5 (nav)"),
    "qwendrive": (2, "native50-2hz-084418258d833a9f-9b2bc9", "QwenDrive ‡"),
}
#: The models whose route 3800 never scored, and what their record must say.
UNSCORABLE_3800 = ("alpamayo15", "qwendrive")

#: Every `model_env` key that differs between each model's native-rate run and its
#: 20 Hz run, the rate itself excluded. The page argues each one immaterial; this
#: is the set it argues about, so a new difference fails rather than slipping under
#: an argument that was made about something else.
EXPECTED_ENV_DRIFT = {
    "drivejepa": {"MKL_NUM_THREADS", "OMP_NUM_THREADS"},
    "qwendrive": {"MKL_NUM_THREADS", "OMP_NUM_THREADS", "QWEN_DRIVE_TRACE"},
    "alpamayo15": {
        "ALPAMAYO_CHECKPOINT",
        "ALPAMAYO_COSMOS",
        "ALPAMAYO_NAV",
        "ALPAMAYO_RESERVE_GB",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    },
}

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
        # In the published total and in no published row: zero everywhere today.
        "unrowed": {
            k: sum(r["infractions"][k] for r in records) for k in set(TOTALLED) - set(BREAKDOWN.values())
        },
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
    lines = PAGE.read_text().split("\n")
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
    """The rate is recorded per run; nothing else records it."""
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


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_the_environment_difference_is_the_one_the_page_argues(model: str) -> None:
    """The page argues each non-rate difference immaterial. Pin the set it argues about.

    The claim that only the rate changed was published once and was not true. If a
    later campaign introduces another difference, this fails rather than letting the
    page keep saying the same thing about a different set.
    """
    run = json.loads((_run_dir(model) / "run.json").read_text())
    pooled = next((REPO_ROOT / "results/bench2drive" / model).glob("pooled50-*/run.json"))
    old = json.loads(pooled.read_text())
    new_env, old_env = run["config"]["model_env"], old["config"]["model_env"]
    differing = {
        k for k in set(new_env) | set(old_env) if new_env.get(k) != old_env.get(k) and k != "PADDOCK_INFER_HZ"
    }
    assert differing == EXPECTED_ENV_DRIFT[model], (
        f"{model}: model_env differs from the 20 Hz run in {sorted(differing)}, and "
        f"bench2drive-native-rate.md argues about {sorted(EXPECTED_ENV_DRIFT[model])}"
    )
    # Whatever differs, the page must name it rather than claim nothing did.
    page = (REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.md").read_text()
    for key in differing:
        named = key in page or (key.endswith("_NUM_THREADS") and "MKL/OMP_NUM_THREADS" in page)
        assert named, f"{model}: {key} differs from the 20 Hz run and the page does not name it"


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_a_native_rate_run_ranks_with_its_rate_named(model: str) -> None:
    """The native rate is how these models are ranked, and the ingest writes it so.

    Drive-JEPA by the rate its model.yaml declares; Alpamayo and Qwen-Drive, which
    lost route 3800, by `ranked = true` in their ingest entries. A re-ingest
    reproduces the record instead of undoing an edit.
    """
    hz, _, _ = CAMPAIGN[model]
    run = json.loads((_run_dir(model) / "run.json").read_text())
    assert run["valid"] is True
    assert f"driven at {hz} Hz" in run["notes"]
    why = "declares as planning_hz" if model == "drivejepa" else "ranked = true in the ingest config"
    assert why in run["notes"], f"{model}: the note does not say how it ranks"
    board = (REPO_ROOT / "docs/leaderboard.md").read_text()
    table = board.split("## bench2drive", 1)[1].split("\n## ", 1)[0]
    assert run["run_id"] in table, f"{run['run_id']} is not ranked in the bench2drive table"


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
    # Two keys go into the published total and into no published row. They are
    # zero everywhere today; if one ever is not, the total stops adding up and
    # this is where that is caught rather than in a reader's arithmetic.
    for key, count in new["unrowed"].items():
        assert not count and not old["unrowed"][key], (
            f"{model}: {key} is counted in the total and has no row in the breakdown, "
            "so the total no longer equals the rows above it"
        )

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


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_support_matrix_publishes_the_native_score(model: str) -> None:
    """The matrix's Bench2Drive cell for a re-driven model is asked at its native rate, rate named.
    The column is the 220-route one now; the native-rate 50-route run this page is about stays a
    valid record."""
    hz, run_id, _ = CAMPAIGN[model]
    cell = matrix_cell(model, "bench2drive")
    record = json.loads((REPO_ROOT / "results" / cell.run / "run.json").read_text())
    assert record["config"].get("infer_hz") == hz, f"{model}: the support matrix reads {cell.run!r}"
    assert (cell.qualifier, "R" in cell.marks) == (f"{hz} Hz", False), f"{model}: {cell}"
    columns, rows = load_matrix()
    rank(columns, rows)
    j = next(j for j, c in enumerate(columns) if c.key == "bench2drive")
    assert next(r for r in rows if r.key == model).cells[j].rank is not None, f"{model}: not ranked"
    native = json.loads((REPO_ROOT / "results" / "bench2drive" / model / run_id / "run.json").read_text())
    assert native["valid"], f"{model}: {run_id}"


def test_readme_block_lists_exactly_the_models_that_were_re_run() -> None:
    """OpenPilot and VisionPilot were not re-run; they must not appear as rows."""
    rows = _readme_rows()
    assert set(rows) - {"Model"} == {label for _, _, label in CAMPAIGN.values()}


def test_readme_does_not_pool_openpilot_into_this_block() -> None:
    """The by-recording-set block is upstream of here and stays untouched."""
    readme = PAGE.read_text()
    start = readme.index("#### Native inference rate")
    end = readme.index("[Protocol, full infraction breakdown", start)
    block = readme[start:end]
    assert "OpenPilot's native rate is\n20 Hz" in block
    # No DS/RC number is published for openpilot anywhere in the new block.
    assert not re.search(r"\|\s*OpenPilot", block)


#: OpenPilot's 50-route run the matrix ranked when this page was written, which it sets aside by id.
OPENPILOT_50 = "pooled50-40c29d24c37d542a-c7d2b2"


def test_the_page_names_the_openpilot_run_the_matrix_ranked() -> None:
    """The page sets aside OpenPilot's ranked 50-route run by its id: a date alone named an unranked one."""
    assert OPENPILOT_50 in DOC.read_text(), f"the page does not name {OPENPILOT_50}"
    record = json.loads(
        (REPO_ROOT / "results" / "bench2drive" / "openpilot" / OPENPILOT_50 / "run.json").read_text()
    )
    assert record["valid"]


# ---------------------------------------------------------------------------
# The exposure caveat is stated where the numbers are
# ---------------------------------------------------------------------------


#: Collisions per progress-km, as the page publishes them. Recomputed below.
PER_KM = {
    "drivejepa": (2.14, 3.93, 8, 12, 3.74, 3.05),
    "alpamayo15": (3.91, 4.15, 46, 27, 11.76, 6.50),
    "qwendrive": (2.63, 3.36, 109, 105, 41.48, 31.25),
}
COLLISION_KEYS = ("collisions_vehicle", "collisions_pedestrian", "collisions_layout")


@pytest.mark.parametrize("model", sorted(CAMPAIGN))
def test_the_exposure_normalization_reproduces(native: dict, twenty: dict, model: str) -> None:
    """Progress km and collisions per km, recomputed from the two tracked JSONs.

    Route length is a property of the route, so it joins from the
    speed-normalization records; the page's whole argument about exposure rests on
    this table, so it is recomputed rather than transcribed.
    """
    length = {r["route_id"]: r["route_length_m"] for r in twenty["records"]}
    old_by = {(r["model"], r["route_id"]): r for r in twenty["records"]}
    km_old = km_new = coll_old = coll_new = 0.0
    for rec in (r for r in native["records"] if r["model"] == model):
        route = rec["route_id"]
        prev = old_by[(model, route)]
        km_new += length[route] * rec["scores"]["score_route"] / 100 / 1000
        km_old += length[route] * prev["scores"]["score_route"] / 100 / 1000
        coll_new += sum(rec["infractions"][k] for k in COLLISION_KEYS)
        coll_old += sum(prev["infractions"][k] for k in COLLISION_KEYS)
    want = PER_KM[model]
    assert (round(km_old, 2), round(km_new, 2)) == want[:2], (model, "progress km")
    assert (int(coll_old), int(coll_new)) == want[2:4], (model, "collisions")
    assert (round(coll_old / km_old, 2), round(coll_new / km_new, 2)) == want[4:], (model, "per km")
    page = (REPO_ROOT / "docs/benchmarks/bench2drive-native-rate.md").read_text()
    for value in want[4:]:
        assert f"{value:.2f}" in page, (model, value, "not published")


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
