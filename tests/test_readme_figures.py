"""The README's pictures are drawn from their sources, and must say what the sources say."""

from __future__ import annotations

from dataclasses import replace

import pytest

from paddock.registry import TrainedOn, load_models
from tools import readme_figures

TABLE = """\
## Support matrix

| | [🟢 Open](a.md) ADE ↓ (m) | [🔴 Closed](b.md) score ↑ (0–5) | [🟢 Side](c.md) ADE ↓ (m), not ranked |
|---|---|---|---|
| Good | 0.5 | 4.0 <sup>A</sup> | 1.0 |
| Tied | 0.5 | 9.9 <sup>R</sup> | 2.0 |
| Flagged <sup>R</sup> | 0.1 | 5.0 | 3.0 |
| Bad | 0.9 | 1.0 | — |
| Invalid | 0.2 <sup>V</sup> | 3.0 | 2.5 |
| Baseline (Constant Velocity) | 0.7 | 0, by construction | 1.5 |

| | not | this | table |
|---|---|---|---|
"""


@pytest.mark.parametrize("path", [readme_figures.MATRIX_SVG, readme_figures.MODELS_SVG], ids=lambda p: p.name)
def test_each_committed_picture_is_drawn_from_its_current_source(path) -> None:
    assert path.read_text(encoding="utf-8") == readme_figures.figures()[path], (
        f"{path.relative_to(readme_figures.ROOT)} no longer matches its source: "
        "run python -m tools.readme_figures and commit the result"
    )


def test_ranks_follow_the_readme_notes() -> None:
    columns, rows = readme_figures.parse(TABLE)
    readme_figures.rank(columns, rows)
    by = {r.name: r for r in rows}

    assert [c.lower_is_better for c in columns] == [True, False, True]
    assert len(rows) == 6, "the second table is not part of the matrix"

    # Lower is better, ties share a rank, an R on the name unranks the whole row, and a
    # V (stored valid: false) unranks its cell.
    ranks = [by[n].cells[0].rank for n in ("Good", "Tied", "Flagged", "Bad", "Invalid")]
    assert ranks == [1, 1, None, 3, None]
    # Higher is better, and an R on one cell unranks only that cell.
    assert [by[n].cells[1].rank for n in ("Good", "Tied", "Bad", "Invalid")] == [1, None, 3, 2]
    # A column whose header says "not ranked" ranks nobody; the floor is never ranked.
    assert all(r.cells[2].rank is None for r in rows)
    assert by["Baseline (Constant Velocity)"].cells[0].rank is None

    # Only a ranked cell strictly worse than the floor is flagged.
    assert by["Bad"].cells[0].below_floor and not by["Good"].cells[0].below_floor
    assert not by["Bad"].cells[1].below_floor  # 1.0 beats a floor of 0

    floor = by["Baseline (Constant Velocity)"].cells[1]
    assert (floor.value, floor.qualifier) == (0.0, "by construction")
    assert by["Bad"].cells[2].value is None


def test_cameras_are_counted_by_view() -> None:
    # openpilot declares CAM_FRONT as a fallback for one-camera sources; it is the same
    # view as its road camera, so the picture shows two cameras, not three.
    openpilot = load_models()["openpilot"]
    assert len(openpilot.cameras) == 3
    assert [v[0] for v in readme_figures.views(openpilot)] == ["front", "front wide"]

    with pytest.raises(ValueError, match="CAM_ROOF"):
        readme_figures.views(replace(openpilot, cameras=("CAM_ROOF",)))


def test_only_paddocks_own_datasets_are_marked_as_benchmark_data() -> None:
    model = replace(
        load_models()["qwendrive"],
        trained_on=(TrainedOn("navsim", "navtrain"), TrainedOn("waymo-e2e", "train")),
    )
    assert readme_figures.chips(model, {"navsim", "kitscenes"}) == [
        ("NAVSIM navtrain", True),
        ("Waymo E2E", False),
    ]
