"""The README heatmap is drawn from the README table, and must say what the table says."""

from __future__ import annotations

from tools import readme_matrix

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


def test_the_committed_heatmap_is_drawn_from_the_current_table() -> None:
    readme = readme_matrix.README.read_text(encoding="utf-8")
    svg = readme_matrix.SVG.read_text(encoding="utf-8")
    assert svg == readme_matrix.build(readme), (
        "docs/assets/support-matrix.svg no longer matches the README table: "
        "run python -m tools.readme_matrix and commit the result"
    )


def test_ranks_follow_the_readme_notes() -> None:
    columns, rows = readme_matrix.parse(TABLE)
    readme_matrix.rank(columns, rows)
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
