"""Draw the README support matrix as a heatmap shaded by rank.

    python -m tools.readme_matrix            # rewrite docs/assets/support-matrix.svg
    python -m tools.readme_matrix --check    # exit 1 if the picture is stale

The README's markdown table is the one source of the numbers; the SVG is drawn from it
and never edited by hand. tests/test_readme_matrix.py fails the build when the two
disagree, so a cell changed in the table without re-running this cannot merge.

What the picture claims is what the README's own notes claim, read off the table:
- A column ranks lower-is-better when its header has a down arrow, else higher.
- A cell marked R ("run differently from the rest of its column") or V ("stored
  valid: false"), and every cell of a column whose header says "not ranked", are
  hatched and carry no rank.
- A mark on a model's name applies to every cell in its row, as the README says.
- The baseline row is the floor: never ranked, and a ranked cell strictly worse than it
  gets a red triangle.
Ties share a rank. Shading is by rank, not by value, so one outlier (Drive-JEPA's 16 m
on comma2k19) cannot wash the rest of its column to one shade.

GitHub shows a README image through <img>, so the SVG cannot load fonts, scripts or
links. Dark mode is a prefers-color-scheme block inside it; a viewer whose browser
ignores that still gets the light card, which is opaque and readable on either theme.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SVG = ROOT / "docs" / "assets" / "support-matrix.svg"

LOOP = {"🟢": "open loop", "🟡": "pseudo-closed loop", "🔴": "closed loop"}
MISSING = ("—", "not run")
UNRANKED_MARKS = {"R", "V"}
FLOOR_PREFIX = "Baseline"

# Rank 1..6 (6 also covers anything lower). Light mode runs dark-to-light; dark mode
# flips the anchor so rank 1 is the step with the most contrast against a dark card.
# Text colour per step is whichever of white or ink clears 4.5:1 on it.
RAMP_LIGHT = ["#104281", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
TEXT_LIGHT = ["#ffffff", "#ffffff", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#0b0b0b"]
RAMP_DARK = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
TEXT_DARK = ["#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]

PAD, LABEL_W, COL_W, CELL_H, GAP = 20, 196, 84, 46, 3
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


@dataclass
class Column:
    name: str
    loop: str
    metric: str
    lower_is_better: bool
    ranked: bool


@dataclass
class Cell:
    text: str
    value: float | None = None
    shown: str = ""
    qualifier: str = ""
    marks: list[str] = field(default_factory=list)
    rank: int | None = None
    below_floor: bool = False


@dataclass
class Row:
    name: str
    marks: list[str]
    cells: list[Cell]

    @property
    def is_floor(self) -> bool:
        return self.name.startswith(FLOOR_PREFIX)


def _split_marks(text: str) -> tuple[str, list[str]]:
    marks = [m for sup in re.findall(r"<sup>(.*?)</sup>", text) for m in sup.split()]
    return re.sub(r"\s*<sup>.*?</sup>", "", text).strip(), marks


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(readme: str) -> tuple[list[Column], list[Row]]:
    """The first table under '## Support matrix'."""
    lines = readme.split("## Support matrix", 1)[1].splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("|"))
    header = lines[start]
    # The table ends at its first non-table line; a later '|' block is another table.
    body = []
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        body.append(ln)

    columns = []
    for head in _cells(header)[1:]:
        m = re.match(r"\[(\S+)\s+(.+?)\]\(.+?\)\s*(.*)", head)
        if not m or m.group(1) not in LOOP:
            raise ValueError(f"support matrix header not understood: {head!r}")
        emoji, name, metric = m.groups()
        if ("↓" in metric) == ("↑" in metric):
            raise ValueError(f"header needs exactly one of ↑ or ↓: {head!r}")
        columns.append(Column(name, LOOP[emoji], metric, "↓" in metric, "not ranked" not in metric))

    rows = []
    for line in body:
        name_cell, *raw = _cells(line)
        if len(raw) != len(columns):
            raise ValueError(f"row has {len(raw)} cells, header has {len(columns)}: {line!r}")
        name, row_marks = _split_marks(name_cell)
        cells = []
        for text in raw:
            shown, marks = _split_marks(text)
            if shown in MISSING:
                cells.append(Cell(shown, marks=marks))
                continue
            m = re.match(r"(-?\d+(?:\.\d+)?)(.*)", shown)
            if not m:
                raise ValueError(f"cell is neither a number nor one of {MISSING}: {text!r}")
            qualifier = m.group(2).strip(" ,").removeprefix("(").removesuffix(")")
            cells.append(Cell(shown, float(m.group(1)), m.group(1), qualifier, marks))
        rows.append(Row(name, row_marks, cells))
    return columns, rows


def rank(columns: list[Column], rows: list[Row]) -> None:
    floor = next((r for r in rows if r.is_floor), None)
    for j, col in enumerate(columns):
        if not col.ranked:
            continue
        pool = [
            r.cells[j]
            for r in rows
            if not r.is_floor
            and r.cells[j].value is not None
            and not UNRANKED_MARKS & {*r.cells[j].marks, *r.marks}
        ]
        sign = 1 if col.lower_is_better else -1
        for cell in pool:
            cell.rank = 1 + sum(sign * o.value < sign * cell.value for o in pool)
            base = floor.cells[j].value if floor else None
            cell.below_floor = base is not None and sign * cell.value > sign * base


def _wrap(text: str, width: int) -> list[str]:
    out = []
    for segment in text.split(", "):
        line = ""
        for word in segment.split():
            if line and len(line) + 1 + len(word) > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out


def _text(x, y, s, cls, size, anchor="start", family=SANS, weight=None, extra="") -> str:
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" font-family="{family}"'
        f' text-anchor="{anchor}"{w}{extra}>{s}</text>'
    )


def render(columns: list[Column], rows: list[Row]) -> str:
    width = 2 * PAD + LABEL_W + COL_W * len(columns)
    metric_lines = [_wrap(c.metric, 14) for c in columns]
    legend_h, band_h = 34, 24
    head_h = band_h + 18 + 13 * max(len(m) for m in metric_lines) + 8
    floor_gap = 10
    grid_top = PAD + legend_h + head_h
    height = grid_top + CELL_H * len(rows) + floor_gap * any(r.is_floor for r in rows) + PAD
    x0 = PAD + LABEL_W

    css = [".card{fill:#fcfcfb;stroke:#dddcd7}", ".ink{fill:#0b0b0b}", ".sec{fill:#52514e}"]
    css += [".miss{fill:#f0efec}", ".floor{fill:#fcfcfb;stroke:#a9a8a3;stroke-dasharray:3 3}"]
    css += [".hbg{fill:#f0efec}", ".hln{stroke:#c9c8c3}", ".worse{fill:#d03b3b;stroke:#fcfcfb}"]
    css += [".open{fill:#15803d;stroke:#15803d}", ".pseudo{fill:#b45309;stroke:#b45309}"]
    css += [".closed{fill:#b91c1c;stroke:#b91c1c}"]
    css += [f".r{i + 1}{{fill:{c}}}" for i, c in enumerate(RAMP_LIGHT)]
    css += [f".t{i + 1}{{fill:{c}}}" for i, c in enumerate(TEXT_LIGHT)]
    dark = [".card{fill:#1a1a19;stroke:#3a3a37}", ".ink{fill:#f0efec}", ".sec{fill:#c3c2b7}"]
    dark += [".miss{fill:#2a2a28}", ".floor{fill:#1a1a19;stroke:#6b6a66}"]
    dark += [".hbg{fill:#2a2a28}", ".hln{stroke:#45453f}", ".worse{fill:#e66767;stroke:#1a1a19}"]
    dark += [".open{fill:#4ade80;stroke:#4ade80}", ".pseudo{fill:#fbbf24;stroke:#fbbf24}"]
    dark += [".closed{fill:#f87171;stroke:#f87171}"]
    dark += [f".r{i + 1}{{fill:{c}}}" for i, c in enumerate(RAMP_DARK)]
    dark += [f".t{i + 1}{{fill:{c}}}" for i, c in enumerate(TEXT_DARK)]

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}"'
        f' height="{height}" role="img" aria-labelledby="title">',
        '<title id="title">Paddock support matrix: each model\'s headline score on each '
        "benchmark, shaded by rank within the benchmark</title>",
        "<!-- Generated by tools/readme_matrix.py from the README table. Do not edit. -->",
        "<style>",
        *css,
        "@media (prefers-color-scheme: dark) {",
        *dark,
        "}",
        "</style>",
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse"'
        ' patternTransform="rotate(45)"><rect class="hbg" width="6" height="6"/>'
        '<line class="hln" x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern></defs>',
        f'<rect class="card" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8"/>',
    ]

    # Legend.
    y = PAD + 14
    out.append(_text(PAD, y, "rank 1", "sec", 12))
    sx = PAD + 42
    for i in range(len(RAMP_LIGHT)):
        out.append(f'<rect class="r{i + 1}" x="{sx + 26 * i}" y="{y - 11}" width="24" height="14" rx="3"/>')
    sx += 26 * len(RAMP_LIGHT) + 4
    out.append(_text(sx, y, "rank 6+", "sec", 12))
    sx += 70
    out.append(f'<rect fill="url(#hatch)" x="{sx}" y="{y - 11}" width="24" height="14" rx="3"/>')
    out.append(_text(sx + 30, y, "not ranked", "sec", 12))
    sx += 120
    out.append(f'<path class="worse" d="M{sx} {y - 10}h11l-5.5 9z" stroke-width="1.5"/>')
    out.append(_text(sx + 16, y, "worse than constant velocity", "sec", 12))
    sx += 200
    out.append(_text(sx, y, "corner letters: see the notes under the table", "sec", 12))

    # Loop-type bands over runs of adjacent columns.
    by = PAD + legend_h + 10
    j = 0
    while j < len(columns):
        k = j
        while k + 1 < len(columns) and columns[k + 1].loop == columns[j].loop:
            k += 1
        cls = columns[j].loop.split()[0].split("-")[0]
        bx, bw = x0 + COL_W * j + GAP, COL_W * (k - j + 1) - GAP
        out.append(f'<rect class="{cls}" x="{bx}" y="{by - 9}" width="{bw}" height="3" rx="1.5"/>')
        out.append(_text(bx, by + 8, columns[j].loop.upper(), "sec", 9.5, extra=' letter-spacing="0.8"'))
        j = k + 1

    # Column headers.
    for j, col in enumerate(columns):
        cx = x0 + COL_W * j + (COL_W + GAP) / 2
        hy = PAD + legend_h + band_h + 14
        out.append(_text(cx, hy, escape(col.name), "ink", 12, "middle", weight="600"))
        for i, line in enumerate(metric_lines[j]):
            out.append(_text(cx, hy + 15 + 13 * i, escape(line), "sec", 10.5, "middle"))

    # Rows.
    y = grid_top
    for row in rows:
        if row.is_floor:
            y += floor_gap
        # "Drive-JEPA, KITScenes re-ranker" and "Baseline (Constant Velocity)" are too
        # long for one line: the part after the comma or bracket goes on a second.
        main, sub = (re.split(r", | \(", row.name, maxsplit=1) + [""])[:2]
        sub = sub.removesuffix(")")
        sup = (
            f'<tspan dx="3" dy="-6" font-size="9" class="sec">{escape(" ".join(row.marks))}</tspan>'
            if row.marks
            else ""
        )
        name_cls = "sec" if row.is_floor else "ink"
        ty = y + CELL_H / 2 + (0 if sub else 5)
        out.append(_text(PAD, ty - (2 if sub else 0), escape(main) + sup, name_cls, 14, weight="600"))
        if sub:
            out.append(_text(PAD, ty + 13, escape(sub), "sec", 11.5))

        for j, cell in enumerate(row.cells):
            cx, cw = x0 + COL_W * j + GAP, COL_W - GAP
            cy, ch = y + GAP / 2, CELL_H - GAP
            mid = cx + cw / 2
            if cell.value is None:
                out.append(f'<rect class="miss" x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4"/>')
                out.append(_text(mid, cy + ch / 2 + 4, escape(cell.text), "sec", 11.5, "middle"))
                continue
            if row.is_floor:
                fill, txt, sub_txt = 'class="floor"', "sec", cell.qualifier or "floor"
            elif cell.rank is None:
                fill, txt = 'fill="url(#hatch)"', "ink"
                sub_txt = cell.qualifier or ("not ranked" if columns[j].ranked else "")
            else:
                step = min(cell.rank, len(RAMP_LIGHT))
                fill, txt = f'class="r{step}"', f"t{step}"
                sub_txt = f"#{cell.rank}" + (f" · {cell.qualifier}" if cell.qualifier else "")
            out.append(f'<rect {fill} x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4"/>')
            vy = cy + (21 if sub_txt else ch / 2 + 5)
            out.append(_text(mid, vy, cell.shown, txt, 14, "middle", MONO, weight="600"))
            if sub_txt:
                size = 10.5 if len(sub_txt) <= 12 else 9.5  # "by construction" fits at 9.5
                out.append(_text(mid, cy + 36, escape(sub_txt), txt, size, "middle", extra=' opacity="0.85"'))
            if cell.marks:
                out.append(
                    _text(
                        cx + cw - 4,
                        cy + 10,
                        " ".join(cell.marks),
                        txt,
                        9,
                        "end",
                        weight="600",
                        extra=' opacity="0.85"',
                    )
                )
            if cell.below_floor:
                out.append(f'<path class="worse" d="M{cx + 3} {cy + 3}h10l-5 8.5z" stroke-width="1.5"/>')
        y += CELL_H

    out.append("</svg>")
    return "\n".join(out) + "\n"


def build(readme: str) -> str:
    columns, rows = parse(readme)
    rank(columns, rows)
    return render(columns, rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the SVG is stale")
    args = ap.parse_args(argv)
    svg = build(README.read_text(encoding="utf-8"))
    if args.check:
        stale = not SVG.exists() or SVG.read_text(encoding="utf-8") != svg
        if stale:
            print(f"{SVG.relative_to(ROOT)} is stale: run python -m tools.readme_matrix", file=sys.stderr)
        return int(stale)
    SVG.write_text(svg, encoding="utf-8")
    print(f"wrote {SVG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
