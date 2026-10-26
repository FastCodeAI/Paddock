"""The README's pictures are drawn from their sources, and must say what the sources say."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from paddock.registry import TrainedOn, load_datasets, load_models
from tools import readme_figures

HEADER = """
[[benchmark]]
key = "open"
name = "Open"
loop = "open"
header = "ADE ↓ (m)"
metric = "score"
decimals = 2

[[benchmark]]
key = "closed"
name = "Closed"
loop = "closed"
header = "score ↑ (0–5)"
metric = "score"
decimals = 2

[[benchmark]]
key = "side"
name = "Side"
loop = "open"
header = "ADE ↓ (m), not ranked"
metric = "score"
decimals = 2
ranked = false
"""

ROWS = """
[[model]]
key = "good"
name = "Good"

[[model]]
key = "tied"
name = "Tied"

[[model]]
key = "flagged"
name = "Flagged"
marks = "R"

[[model]]
key = "bad"
name = "Bad"

[[model]]
key = "invalid"
name = "Invalid"

[[model]]
key = "floor"
name = "Baseline (Constant Velocity)"
floor = true

[cells.good]
open = { run = "open/good/1" }
closed = { run = "closed/good/1", marks = "A" }
side = { run = "side/good/1" }

[cells.tied]
open = { run = "open/tied/1" }
closed = { run = "closed/tied/1", marks = "R" }
side = "not run"

[cells.flagged]
open = { run = "open/flagged/1" }
closed = { run = "closed/flagged/1" }

[cells.bad]
open = { run = "open/bad/1" }
closed = { run = "closed/bad/1" }

[cells.invalid]
open = { run = "open/invalid/1", marks = "V" }
closed = { run = "closed/invalid/1" }

[cells.floor]
open = { value = 0.7, source = "page.md" }
closed = { value = 0, source = "page.md", note = "by construction" }
"""

RUNS = {
    "open/good/1": 0.5, "open/tied/1": 0.5, "open/flagged/1": 0.1, "open/bad/1": 0.9,
    "closed/good/1": 4.0, "closed/tied/1": 9.9, "closed/flagged/1": 5.0, "closed/bad/1": 1.0,
    "closed/invalid/1": 3.0, "side/good/1": 1.0,
}  # fmt: skip


def _load(tmp_path: Path, text: str, runs: dict[str, float], invalid: tuple[str, ...] = ()):
    for run, value in runs.items():
        folder = tmp_path / "results" / run
        folder.mkdir(parents=True)
        (folder / "metrics.json").write_text(json.dumps({"metrics": {"score": {"mean": value}}}))
        (folder / "run.json").write_text(json.dumps({"valid": run not in invalid}))
    (tmp_path / "page.md").write_text("constant velocity scores 0.70 here, and 0 on the closed loop")
    (tmp_path / "matrix.toml").write_text(text)
    columns, rows = readme_figures.load_matrix(tmp_path / "matrix.toml", tmp_path / "results", root=tmp_path)
    readme_figures.rank(columns, rows)
    return columns, {r.key: r for r in rows}


@pytest.mark.parametrize("path", [readme_figures.MATRIX_SVG, readme_figures.MODELS_SVG], ids=lambda p: p.name)
def test_each_committed_picture_is_drawn_from_its_current_source(path) -> None:
    assert path.read_text(encoding="utf-8") == readme_figures.figures()[path], (
        f"{path.relative_to(readme_figures.ROOT)} no longer matches its source: "
        "run python -m tools.readme_figures and commit the result"
    )


def test_ranks_follow_the_readme_notes(tmp_path: Path) -> None:
    runs = {**RUNS, "open/invalid/1": 0.2}
    columns, by = _load(tmp_path, HEADER + ROWS, runs, invalid=("open/invalid/1",))

    assert [c.lower_is_better for c in columns] == [True, False, True]
    # Lower is better, ties share a rank, an R on the model unranks its whole row, and
    # a V unranks a run stored valid: false.
    ranks = [by[n].cells[0].rank for n in ("good", "tied", "flagged", "bad", "invalid")]
    assert ranks == [1, 1, None, 3, None]
    # Higher is better, and an R on one cell unranks only that cell.
    assert [by[n].cells[1].rank for n in ("good", "tied", "bad", "invalid")] == [1, None, 3, 2]
    # A column declared ranked = false ranks nobody; the floor is never ranked.
    assert all(r.cells[2].rank is None for r in by.values())
    assert by["floor"].cells[0].rank is None

    # Only a ranked cell strictly worse than the floor is flagged.
    assert by["bad"].cells[0].below_floor and not by["good"].cells[0].below_floor
    assert not by["bad"].cells[1].below_floor  # 1.0 beats a floor of 0

    # Numbers are read from the run and printed to the column's decimals; a cell left
    # out reads as a dash.
    assert (by["good"].cells[0].shown, by["good"].cells[0].run) == ("0.50", "open/good/1")
    assert (by["floor"].cells[1].shown, by["floor"].cells[1].qualifier) == ("0", "by construction")
    assert (by["tied"].cells[2].text, by["bad"].cells[2].text) == ("not run", "—")


@pytest.mark.parametrize(
    "change, error",
    [
        ('open = { run = "open/good/1" }', "no run at"),  # its folder is never written
        ('open = { run = "closed/good/1" }', "is not a open run"),
        ('open = { value = 0.55, source = "page.md" }', "does not show 0.55"),
        ('open = { run = "open/good/1", mark = "A" }', "a cell is"),
    ],
)
def test_a_cell_that_cannot_be_traced_fails_loudly(tmp_path: Path, change: str, error: str) -> None:
    rows = ROWS.replace('open = { run = "open/good/1" }', change)
    runs = {k: v for k, v in RUNS.items() if not (k == "open/good/1" and "no run" in error)}
    with pytest.raises(ValueError, match=error):
        _load(tmp_path, HEADER + rows, {**runs, "open/invalid/1": 0.2})


def test_a_run_stored_invalid_cannot_be_ranked(tmp_path: Path) -> None:
    rows = ROWS.replace('open = { run = "open/invalid/1", marks = "V" }', 'open = { run = "open/invalid/1" }')
    with pytest.raises(ValueError, match="valid: false"):
        _load(tmp_path, HEADER + rows, {**RUNS, "open/invalid/1": 0.2}, invalid=("open/invalid/1",))


def test_cameras_are_counted_by_view() -> None:
    # openpilot declares CAM_FRONT as a fallback for one-camera sources; it is the same
    # view as its road camera, so the picture shows two cameras, not three.
    openpilot = load_models()["openpilot"]
    assert len(openpilot.cameras) == 3
    assert [v[0] for v in readme_figures.views(openpilot)] == ["front", "front wide"]

    with pytest.raises(ValueError, match="CAM_ROOF"):
        readme_figures.views(replace(openpilot, cameras=("CAM_ROOF",)))

    # alpamayo is rendered three cameras and sees four views: its telephoto slot is a
    # crop of CAM_FRONT, and the picture shows what the model reads.
    alpamayo = load_models()["alpamayo15"]
    assert len(alpamayo.cameras) == 3
    assert [v[0] for v in readme_figures.views(alpamayo)] == [
        "front left",
        "front",
        "front right",
        "front tele",
    ]


def test_history_shows_camera_and_ego_at_their_own_rates() -> None:
    models = load_models()
    # Camera frames and ego poses run at different rates: 0.5 s apart and 0.1 s apart.
    frames, ego, caption = readme_figures.history(models["qwendrive"])
    assert (frames, len(ego), caption) == ([0.0, 0.5, 1.0, 1.5], 16, "cam 4 @ 2 Hz · ego 16 @ 10 Hz")
    assert ego[-1] == pytest.approx(1.5)
    # openpilot reads no speed at all; METEOR reads the current speed and nothing older.
    assert readme_figures.history(models["openpilot"])[1:] == ([], "cam 2 @ 20 Hz · no ego input")
    assert readme_figures.history(models["meteor"])[1:] == ([0.0], "cam 1 · speed now")


def test_only_paddocks_own_datasets_are_marked_as_benchmark_data() -> None:
    model = replace(
        load_models()["qwendrive"],
        trained_on=(TrainedOn("navsim", "navtrain"), TrainedOn("waymo-e2e", "train")),
    )
    assert readme_figures.chips(model, {"navsim", "kitscenes"}) == [
        ("NAVSIM navtrain", True),
        ("Waymo E2E", False),
    ]


def test_drivejepa_shows_every_training_stage() -> None:
    """V-JEPA 2's video, the driving-video pretraining, then navtrain: all three stages."""
    own = set(load_datasets())
    assert readme_figures.chips(load_models()["drivejepa"], own) == [
        ("VideoMix22M", False),
        ("CoVLA", False),
        ("DrivingDojo", False),
        ("OpenScene trainval", True),
        ("NAVSIM navtrain", True),
    ]
