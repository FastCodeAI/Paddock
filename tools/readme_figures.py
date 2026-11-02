"""Draw the README's two pictures: the support matrix and what the models are.

    python -m tools.readme_figures            # rewrite both SVGs in docs/assets/
    python -m tools.readme_figures --check    # exit 1 if either is stale

Neither picture holds a number of its own. Each support-matrix cell in
docs/support-matrix.toml names the run its number is read from; the models picture
comes from each adapter's model.yaml. tests/test_readme_figures.py fails the build
when a picture and its source disagree, a named run is missing, or a run stored
valid: false would be ranked.

The support matrix claims what the README's notes claim:
- A column ranks lower-is-better when its header has a down arrow, else higher.
- A cell marked R ("measured on different scenes or by a different method") or V
  ("stored valid: false"), and every cell of a column declared `ranked = false`, are hatched
  and carry no rank.
- A mark on a model applies to every cell in its row, as the README says.
- The floor row (constant velocity) is ranked like a model, and any other ranked
  cell strictly worse than it gets a red triangle.
Ties share a rank. Shading is by rank, not by value, so one outlier (Drive-JEPA's 16 m
on comma2k19) cannot wash the rest of its column to one shade.

The models picture colours each model as viz/palette.py does in every other paddock
figure. A training set that is one of paddock's own datasets (a key in datasets.yaml)
is drawn inverted: that overlap is what the matrix's A mark records.

GitHub shows a README image through <img>, so the SVG cannot load fonts, scripts or
links. Dark mode is a prefers-color-scheme block inside it; a viewer whose browser
ignores that still gets the light card, which is opaque and readable on either theme.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paddock.registry import ModelSpec, load_benchmarks, load_datasets, load_models  # noqa: E402
from paddock.results import report  # noqa: E402
from paddock.viz.palette import color  # noqa: E402

MATRIX = ROOT / "docs" / "support-matrix.toml"
RESULTS = ROOT / "results"
MATRIX_SVG = ROOT / "docs" / "assets" / "support-matrix.svg"
MODELS_SVG = ROOT / "docs" / "assets" / "models.svg"

LOOP = {"open": "open loop", "pseudo": "pseudo-closed loop", "closed": "closed loop"}
MISSING = ("—", "not run")
UNRANKED_MARKS = {"R", "V"}
CELL_KEYS = {"run", "value", "source", "marks", "note", "model"}

# Rank 1..6 (6 also covers anything lower). Light mode runs dark-to-light; dark mode
# flips the anchor so rank 1 is the step with the most contrast against a dark card.
# Text colour per step is whichever of white or ink clears 4.5:1 on it.
RAMP_LIGHT = ["#104281", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
TEXT_LIGHT = ["#ffffff", "#ffffff", "#0b0b0b", "#0b0b0b", "#0b0b0b", "#0b0b0b"]
RAMP_DARK = ["#b7d3f6", "#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"]
TEXT_DARK = ["#0b0b0b", "#0b0b0b", "#0b0b0b", "#ffffff", "#ffffff", "#ffffff"]

#: The README's models, in its row order, under the names the support matrix uses.
MODELS = {
    "openpilot": "OpenPilot",
    "alpamayo15": "Alpamayo 1.5",
    "drivejepa": "Drive-JEPA",
    "visionpilot": "VisionPilot",
    "meteor": "METEOR",
    "qwendrive": "Qwen-Drive 1.0",
    "autoe2e": "AutoE2E",
}

#: Where each declared camera or view looks: (view, yaw in degrees left of forward, field of
#: view). Cameras that share a view are one camera: openpilot's CAM_FRONT is its
#: fallback for one-camera sources (its model.yaml), the same view as its road camera.
CAMERA_VIEW = {
    "CAM_FRONT": ("front", 0, 60),
    "CAM_OP_ROAD": ("front", 0, 60),
    "CAM_OP_WIDE": ("front wide", 0, 120),
    "CAM_FRONT_LEFT": ("front left", 55, 60),
    "CAM_FRONT_RIGHT": ("front right", -55, 60),
    "CAM_BACK": ("back", 180, 100),
    "CAM_BACK_LEFT": ("back left", 110, 60),
    "CAM_BACK_RIGHT": ("back right", -110, 60),
    # Not a camera anyone renders: alpamayo crops its 30 deg slot out of CAM_FRONT.
    "CAM_FRONT_TELE": ("front tele", 0, 30),
}

#: Short names for `trained_on` datasets. Anything missing prints its key.
DATASET_LABEL = {
    "comma-fleet": "comma fleet",
    "nvidia-internal": "NVIDIA internal",
    "navsim": "NAVSIM",
    "openlane": "OpenLane (Waymo)",
    "tusimple": "TuSimple",
    "curvelanes": "CurveLanes",
    "tier4_comlops_drs": "TIER IV DRS",
    "waymo-e2e": "Waymo E2E",
    "nvidia-physical-ai": "NVIDIA Physical AI",
    "nuplan": "nuPlan",
    "kitscenes": "KITScenes",
    "videomix22m": "VideoMix22M",
    "covla": "CoVLA",
    "drivingdojo": "DrivingDojo",
    "qwen3-vl": "Qwen3-VL corpus",
    "cosmos-reason2": "Cosmos-Reason2 data",
    "public-driving-qa": "public driving QA sets",
    "qwen3.5": "Qwen3.5 corpus",
    "in-house-china": "in-house (China)",
    "zod": "ZOD",
    "imagenet": "ImageNet",
    "coco": "COCO",
    "nuscenes": "nuScenes",
    # OpenScene is what datasets.yaml calls navsim; its trainval split is the raw
    # video, of which NAVSIM's navtrain is a filter.
    "navsim/trainval": "OpenScene trainval",
}

PAD, LABEL_W, COL_W, CELL_H, GAP = 20, 196, 84, 46, 3
WIDTH = 2 * PAD + LABEL_W + COL_W * 9
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

BASE_LIGHT = [".card{fill:#fcfcfb;stroke:#dddcd7}", ".ink{fill:#0b0b0b}", ".sec{fill:#52514e}"]
BASE_DARK = [".card{fill:#1a1a19;stroke:#3a3a37}", ".ink{fill:#f0efec}", ".sec{fill:#c3c2b7}"]


def _text(x, y, s, cls, size, anchor="start", family=SANS, weight=None, extra="") -> str:
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:g}" y="{y:g}" class="{cls}" font-size="{size}" font-family="{family}"'
        f' text-anchor="{anchor}"{w}{extra}>{s}</text>'
    )


def _document(
    height: int, title: str, source: str, light: list[str], dark: list[str], defs: str
) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}"'
        f' height="{height}" role="img" aria-labelledby="title">',
        f'<title id="title">{escape(title)}</title>',
        f"<!-- Generated by tools/readme_figures.py from {source}. Do not edit. -->",
        "<style>",
        *BASE_LIGHT,
        *light,
        "@media (prefers-color-scheme: dark) {",
        *BASE_DARK,
        *dark,
        "}",
        "</style>",
        f"<defs>{defs}</defs>",
        f'<rect class="card" x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="8"/>',
    ]


# -- the support matrix ------------------------------------------------------


@dataclass
class Column:
    key: str
    name: str
    loop: str
    header: str
    metric: str
    decimals: int
    lower_is_better: bool
    ranked: bool


@dataclass
class Cell:
    text: str
    value: float | None = None
    shown: str = ""
    qualifier: str = ""
    marks: list[str] = field(default_factory=list)
    run: str = ""
    source: str = ""
    valid: bool = True
    rank: int | None = None
    below_floor: bool = False


@dataclass
class Row:
    key: str
    name: str
    marks: list[str]
    cells: list[Cell]
    is_floor: bool = False


def _cell(spec: object, col: Column, model: str, where: str, results: Path, root: Path) -> Cell:
    if spec is None:
        return Cell("—")
    if spec in MISSING:
        return Cell(str(spec))
    if not isinstance(spec, dict) or not set(spec) <= CELL_KEYS or ("run" in spec) == ("value" in spec):
        raise ValueError(
            f"{where}: a cell is {{ run = ... }} or {{ value = ..., source = ... }}, got {spec!r}"
        )
    marks, note = spec.get("marks", "").split(), spec.get("note", "")
    if "run" in spec:
        run = spec["run"]
        if not run.startswith(f"{col.key}/"):
            raise ValueError(f"{where}: {run!r} is not a {col.key} run")
        # A line copied from another row and not edited would show that model's number
        # under this one. A cell that borrows a run on purpose says whose it is.
        owner = run.split("/")[1]
        if owner != spec.get("model", model):
            raise ValueError(
                f"{where}: {run!r} is a {owner} run; add model = {owner!r} if this row borrows it"
            )
        folder = results / run
        if not (folder / "metrics.json").exists():
            raise ValueError(f"{where}: no run at results/{run}")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        if col.metric not in metrics:
            raise ValueError(f"{where}: results/{run} has no {col.metric!r}")
        value = metrics[col.metric]
        value = value["mean"] if isinstance(value, dict) else value
        valid = json.loads((folder / "run.json").read_text(encoding="utf-8")).get("valid", True)
        shown = f"{value:.{col.decimals}f}"
        return Cell(shown, value, shown, note, marks, run=run, valid=valid)
    if "source" not in spec:
        raise ValueError(f"{where}: a typed value needs a source page")
    value, source = spec["value"], spec["source"]
    # An int is a figure that is exact by construction (neuroNCAP's 0), not a measurement.
    shown = str(value) if isinstance(value, int) else f"{value:.{col.decimals}f}"
    page = root / source
    if not page.exists() or shown not in page.read_text(encoding="utf-8"):
        raise ValueError(f"{where}: {source} does not show {shown}")
    return Cell(shown, float(value), shown, note, marks, source=source)


def load_matrix(
    path: Path | None = None, results: Path | None = None, root: Path = ROOT
) -> tuple[list[Column], list[Row]]:
    """docs/support-matrix.toml, with every run's number read out of results/."""
    path, results = path or MATRIX, results or RESULTS
    body = tomllib.loads(path.read_text(encoding="utf-8"))
    primary = {}
    columns = []
    for b in body["benchmark"]:
        header = b["header"]
        if ("↓" in header) == ("↑" in header):
            raise ValueError(f"{b['key']}: the header needs exactly one of ↑ or ↓: {header!r}")
        if "metric" not in b and not primary:
            primary = {k: spec.primary_metric for k, spec in load_benchmarks().items()}
        metric = b.get("metric") or primary.get(b["key"], "")
        loop = LOOP[b["loop"]]
        lower = "↓" in header
        # The arrow is typed by hand; paddock's own report already knows the direction.
        if not report.has_direction(metric) or report.lower_is_better(metric) != lower:
            said = "lower" if lower else "higher"
            raise ValueError(
                f"{b['key']}: the header says {said} is better, which paddock's report does not say of {metric!r}"
            )
        columns.append(
            Column(b["key"], b["name"], loop, header, metric, b["decimals"], lower, b.get("ranked", True))
        )

    keys = {c.key for c in columns}
    cells = body.get("cells", {})
    unknown = set(cells) - {m["key"] for m in body["model"]}
    if unknown:
        raise ValueError(f"[cells.*] names models that are not listed: {sorted(unknown)}")
    rows = []
    for m in body["model"]:
        given = cells.get(m["key"], {})
        if set(given) - keys:
            raise ValueError(
                f"[cells.{m['key']}] names benchmarks that are not listed: {sorted(set(given) - keys)}"
            )
        row = [
            _cell(given.get(c.key), c, m["key"], f"cells.{m['key']}.{c.key}", results, root) for c in columns
        ]
        rows.append(Row(m["key"], m["name"], m.get("marks", "").split(), row, m.get("floor", False)))
    return columns, rows


def rank(columns: list[Column], rows: list[Row]) -> None:
    floor = next((r for r in rows if r.is_floor), None)
    for j, col in enumerate(columns):
        if not col.ranked:
            continue
        pool = [
            r.cells[j]
            for r in rows
            if r.cells[j].value is not None and not UNRANKED_MARKS & {*r.cells[j].marks, *r.marks}
        ]
        floor_cell = floor.cells[j] if floor else None
        sign = 1 if col.lower_is_better else -1
        for cell in pool:
            if not cell.valid:
                # paddock report leaves a valid: false run out; the matrix may not rank it either.
                raise ValueError(f"results/{cell.run} is stored valid: false, so its cell needs V or R")
            cell.rank = 1 + sum(sign * o.value < sign * cell.value for o in pool)
            base = floor_cell.value if floor_cell else None
            cell.below_floor = cell is not floor_cell and base is not None and sign * cell.value > sign * base


def matrix_cell(model: str, bench: str) -> Cell:
    """One cell of docs/support-matrix.toml, by model and benchmark key."""
    columns, rows = load_matrix()
    j = next(j for j, c in enumerate(columns) if c.key == bench)
    return next(r for r in rows if r.key == model).cells[j]


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


def render_matrix(columns: list[Column], rows: list[Row]) -> str:
    # Both pictures share one width so they scale alike on the page; more benchmarks
    # means narrower columns, not a wider picture.
    col_w = (WIDTH - 2 * PAD - LABEL_W) // len(columns)
    metric_lines = [_wrap(c.header, 14) for c in columns]
    legend_h, band_h = 34, 24
    head_h = band_h + 18 + 13 * max(len(m) for m in metric_lines) + 8
    floor_gap = 10
    grid_top = PAD + legend_h + head_h
    height = grid_top + CELL_H * len(rows) + floor_gap * any(r.is_floor for r in rows) + PAD
    x0 = PAD + LABEL_W

    light = [".miss{fill:#f0efec}", ".floor{fill:#fcfcfb;stroke:#a9a8a3;stroke-dasharray:3 3}"]
    light += [".floorline{fill:none;stroke:#52514e;stroke-width:1.5;stroke-dasharray:4 3}"]
    light += [".hbg{fill:#f0efec}", ".hln{stroke:#c9c8c3}", ".worse{fill:#d03b3b;stroke:#fcfcfb}"]
    light += [".open{fill:#15803d;stroke:#15803d}", ".pseudo{fill:#b45309;stroke:#b45309}"]
    light += [".closed{fill:#b91c1c;stroke:#b91c1c}"]
    light += [f".r{i + 1}{{fill:{c}}}" for i, c in enumerate(RAMP_LIGHT)]
    light += [f".t{i + 1}{{fill:{c}}}" for i, c in enumerate(TEXT_LIGHT)]
    dark = [".miss{fill:#2a2a28}", ".floor{fill:#1a1a19;stroke:#6b6a66}", ".floorline{stroke:#c3c2b7}"]
    dark += [".hbg{fill:#2a2a28}", ".hln{stroke:#45453f}", ".worse{fill:#e66767;stroke:#1a1a19}"]
    dark += [".open{fill:#4ade80;stroke:#4ade80}", ".pseudo{fill:#fbbf24;stroke:#fbbf24}"]
    dark += [".closed{fill:#f87171;stroke:#f87171}"]
    dark += [f".r{i + 1}{{fill:{c}}}" for i, c in enumerate(RAMP_DARK)]
    dark += [f".t{i + 1}{{fill:{c}}}" for i, c in enumerate(TEXT_DARK)]
    defs = (
        '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse"'
        ' patternTransform="rotate(45)"><rect class="hbg" width="6" height="6"/>'
        '<line class="hln" x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern>'
    )
    title = (
        "Paddock support matrix: each model's headline score on each benchmark, "
        "shaded by rank within the benchmark"
    )
    out = _document(height, title, "docs/support-matrix.toml", light, dark, defs)

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
    out.append(_text(sx, y, "corner letters: see the notes below", "sec", 12))

    # Loop-type bands over runs of adjacent columns.
    by = PAD + legend_h + 10
    j = 0
    while j < len(columns):
        k = j
        while k + 1 < len(columns) and columns[k + 1].loop == columns[j].loop:
            k += 1
        cls = columns[j].loop.split()[0].split("-")[0]
        bx, bw = x0 + col_w * j + GAP, col_w * (k - j + 1) - GAP
        out.append(f'<rect class="{cls}" x="{bx}" y="{by - 9}" width="{bw}" height="3" rx="1.5"/>')
        out.append(_text(bx, by + 8, columns[j].loop.upper(), "sec", 9.5, extra=' letter-spacing="0.8"'))
        j = k + 1

    # Column headers.
    for j, col in enumerate(columns):
        cx = x0 + col_w * j + (col_w + GAP) / 2
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
            cx, cw = x0 + col_w * j + GAP, col_w - GAP
            cy, ch = y + GAP / 2, CELL_H - GAP
            mid = cx + cw / 2
            if cell.value is None:
                out.append(f'<rect class="miss" x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4"/>')
                out.append(_text(mid, cy + ch / 2 + 4, escape(cell.text), "sec", 11.5, "middle"))
                continue
            if row.is_floor and cell.rank is None:
                fill, txt, sub_txt = 'class="floor"', "sec", cell.qualifier or "floor"
            elif cell.rank is None:
                fill, txt = 'fill="url(#hatch)"', "ink"
                sub_txt = cell.qualifier or ("not ranked" if columns[j].ranked else "")
            else:
                step = min(cell.rank, len(RAMP_LIGHT))
                fill, txt = f'class="r{step}"', f"t{step}"
                # The floor is ranked like a model, but keeps its dashed edge and its name.
                label = "floor" if row.is_floor else cell.qualifier
                sub_txt = f"#{cell.rank}" + (f" · {label}" if label else "")
            out.append(f'<rect {fill} x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4"/>')
            if row.is_floor and cell.rank is not None:
                out.append(f'<rect class="floorline" x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="4"/>')
            vy = cy + (21 if sub_txt else ch / 2 + 5)
            out.append(_text(mid, vy, cell.shown, txt, 14, "middle", MONO, weight="600"))
            if sub_txt:
                size = 10.5 if len(sub_txt) <= 12 else 9.5  # "by construction" fits at 9.5
                out.append(_text(mid, cy + 36, escape(sub_txt), txt, size, "middle", extra=' opacity="0.85"'))
            if cell.marks:
                marks = " ".join(cell.marks)
                extra = ' opacity="0.85"'
                out.append(_text(cx + cw - 4, cy + 10, marks, txt, 9, "end", weight="600", extra=extra))
            if cell.below_floor:
                out.append(f'<path class="worse" d="M{cx + 3} {cy + 3}h10l-5 8.5z" stroke-width="1.5"/>')
        y += CELL_H

    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_matrix() -> str:
    columns, rows = load_matrix()
    rank(columns, rows)
    return render_matrix(columns, rows)


# -- what the models are -----------------------------------------------------

ROW_H = 64
X_PARAMS, X_CAMS, X_HIST, X_PLAN, X_TRAIN = 182, 370, 496, 650, 818
PARAMS_TRACK = 90  # 30 px per decade of parameters, 10 M to 10 B
HIST_W, PLAN_W = 120, 110  # past and future time axes


def views(model: ModelSpec) -> list[tuple[str, int, int]]:
    """The distinct views the model reads, in declaration order."""
    declared = model.camera_views or model.cameras
    unknown = [c for c in declared if c not in CAMERA_VIEW]
    if unknown:
        raise ValueError(f"{model.name}: no view recorded for camera(s) {unknown}; add them to CAMERA_VIEW")
    out: dict[str, tuple[str, int, int]] = {}
    for cam in declared:
        view = CAMERA_VIEW[cam]
        out.setdefault(view[0], view)
    return list(out.values())


def history(model: ModelSpec) -> tuple[list[float], list[float], str]:
    """(camera frame ages, ego state ages, in seconds before now) and their caption."""
    n = model.history_frames or 1
    hz = model.camera_hz
    frames = [k / hz for k in range(n)] if hz else [0.0]
    cam = f"cam {n} @ {hz:g} Hz" if hz and n > 1 else "cam 1" if n == 1 else f"cam {n}, back to back"
    if hz and n > 1 and model.camera_frames_by_request:
        cam += " in training"
    if model.ego_history_steps and model.ego_history_hz:
        ego = [k / model.ego_history_hz for k in range(model.ego_history_steps)]
        said = f"ego {model.ego_history_steps} @ {model.ego_history_hz:g} Hz"
    elif model.needs_speed:
        ego, said = [0.0], "speed now"
    else:
        ego, said = [], "no ego input"
    return frames, ego, f"{cam} · {said}"


def history_note(model: ModelSpec) -> str:
    """A second caption line where the frames paddock feeds are not at `camera_hz`."""
    if model.camera_frames_by_request and (model.history_frames or 1) > 1:
        return f"paddock feeds the last {model.history_frames} steps, at the benchmark's rate"
    return ""


def chips(model: ModelSpec, own: set[str]) -> list[tuple[str, bool]]:
    """(label, is one of paddock's own datasets) per training set."""
    out = []
    for t in model.trained_on:
        label = DATASET_LABEL.get(t.dataset, t.dataset)
        # The split says which part of one of our benchmarks it saw (navtrain, train).
        label = f"{label} {t.split}" if t.dataset in own and t.split else label
        out.append((DATASET_LABEL.get(f"{t.dataset}/{t.split}", label), t.dataset in own))
    return out


def chip_lines(model: ModelSpec, own: set[str]) -> list[list[tuple[str, bool]]]:
    """The training-set chips, flowed onto as many lines as the column needs."""
    lines: list[list[tuple[str, bool]]] = [[]]
    used = 0.0
    for chip in chips(model, own):
        w = _chip_width(chip[0])
        if lines[-1] and used + 4 + w > WIDTH - PAD - X_TRAIN:
            lines.append([])
            used = 0.0
        used += (4 if lines[-1] else 0) + w
        lines[-1].append(chip)
    return lines


def _params(m: float) -> str:
    # As `paddock list models` prints it: billions past 1000 M, to one decimal.
    return f"{round(m / 1000, 1):g} B" if m >= 1000 else f"{m:g} M"


def _wedge(gx: float, gy: float, yaw: int, fov: int, r: float, fill: str) -> str:
    def pt(deg: float) -> str:
        a = math.radians(deg)
        return f"{gx - r * math.sin(a):.1f} {gy - r * math.cos(a):.1f}"

    return (
        f'<path d="M{gx:g} {gy:g} L{pt(yaw + fov / 2)} A{r:g} {r:g} 0 0 1 {pt(yaw - fov / 2)} Z"'
        f' fill="{fill}" opacity="0.55"/>'
    )


def _chip_width(label: str) -> float:
    return 6.0 * len(label) + 12


def _tick(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x - w / 2:.1f}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}"/>'


def render_models(models: list[tuple[str, ModelSpec]], own: set[str]) -> str:
    plan_scale = 5 * math.ceil(max(m.horizon_s or 0 for _, m in models) / 5)
    ages = [a for _, m in models for a in (*history(m)[0], *history(m)[1])]
    hist_scale = 2 * math.ceil(max(ages) / 2)
    past, future = HIST_W / hist_scale, PLAN_W / plan_scale  # px per second
    top = PAD + 72
    # A row grows to fit its training sets: Drive-JEPA's five take four lines.
    flowed = [chip_lines(m, own) for _, m in models]
    heights = [max(ROW_H, 21 * len(lines) + 8) for lines in flowed]
    height = top + sum(heights) + PAD - 6

    light = [".track{fill:#ecebe7}", ".none{fill:none;stroke:#a9a8a3;stroke-dasharray:3 3}"]
    light += [".car{fill:#8a8984}", ".stripe{stroke:#fcfcfb}", ".now{fill:#a9a8a3}"]
    light += [".chip{fill:#f0efec}", ".chipt{fill:#52514e}", ".own{fill:#2b2b29}", ".ownt{fill:#fcfcfb}"]
    dark = [".track{fill:#2f2f2c}", ".none{stroke:#6b6a66}", ".stripe{stroke:#1a1a19}", ".now{fill:#6b6a66}"]
    dark += [".chip{fill:#2a2a28}", ".chipt{fill:#c3c2b7}", ".own{fill:#f0efec}", ".ownt{fill:#1a1a19}"]
    defs = (
        '<pattern id="stripes" width="5" height="5" patternUnits="userSpaceOnUse"'
        ' patternTransform="rotate(45)"><line class="stripe" x1="0" y1="0" x2="0" y2="5"'
        ' stroke-width="2"/></pattern>'
    )
    title = (
        "What the models are: parameters, cameras, the camera and ego history each reads and "
        "at what rate, the plan each emits and at what rate, and training data, from each model.yaml"
    )
    out = _document(height, title, "each model.yaml", light, dark, defs)

    # Legend: what a tick is, then the two fills that need saying.
    y = PAD + 14
    sx = PAD
    for w, h, text in [(3, 10, "camera frame"), (1.5, 7, "ego state"), (1, 12, "plan point")]:
        out.append(f'<rect class="car" x="{sx}" y="{y - 1 - h:g}" width="{w:g}" height="{h:g}"/>')
        out.append(_text(sx + 8, y, text, "sec", 12))
        sx += 8 + 6.6 * len(text) + 16
    sx += 10
    out.append(f'<rect class="car" x="{sx:g}" y="{y - 10}" width="24" height="12" rx="6"/>')
    out.append(f'<rect fill="url(#stripes)" x="{sx:g}" y="{y - 10}" width="24" height="12" rx="6"/>')
    out.append(_text(sx + 30, y, "outputs controls, not waypoints", "sec", 12))
    sx += 30 + 6.6 * 31 + 26
    out.append(f'<rect class="own" x="{sx:g}" y="{y - 11}" width="24" height="14" rx="7"/>')
    out.append(_text(sx + 30, y, "trained on a paddock dataset (the A in the matrix)", "sec", 12))

    # Column headers, with the scale of each axis under its header.
    hy = PAD + 46
    for x, label, sub in [
        (PAD, "Model", ""),
        (X_PARAMS, "Parameters", ""),
        (X_CAMS, "Cameras", "as the model sees them"),
        (X_HIST, "History", ""),
        (X_PLAN, "Plan", ""),
        (X_TRAIN, "Trained on", ""),
    ]:
        out.append(_text(x, hy, label, "ink", 12, weight="600"))
        if sub:
            out.append(_text(x, hy + 15, sub, "sec", 10))
    out.append(_text(X_PARAMS, hy + 15, "10 M", "sec", 10))
    out.append(_text(X_PARAMS + PARAMS_TRACK, hy + 15, "10 B", "sec", 10, "end"))
    for k, tick in enumerate([f"−{hist_scale} s", f"−{hist_scale // 2}", "now"]):
        anchor = "start" if k == 0 else "end" if k == 2 else "middle"
        out.append(_text(X_HIST + HIST_W / 2 * k, hy + 15, tick, "sec", 10, anchor))
    for k, tick in enumerate(["now", f"{plan_scale // 2:g}", f"{plan_scale} s"]):
        anchor = "start" if k == 0 else "end" if k == 2 else "middle"
        out.append(_text(X_PLAN + PLAN_W / 2 * k, hy + 15, tick, "sec", 10, anchor))

    for i, (label, m) in enumerate(models):
        cy = top + sum(heights[:i]) + heights[i] / 2
        hue = color(m.name)

        # Name, with the first part of its architecture under it.
        out.append(f'<circle cx="{PAD + 5}" cy="{cy - 6}" r="5" fill="{hue}"/>')
        out.append(_text(PAD + 16, cy - 1, escape(label), "ink", 14, weight="600"))
        arch = re.split(r" \+ | -- |: |, | \(", m.architecture)[0]
        out.append(_text(PAD + 16, cy + 15, escape(arch), "sec", 10.5))

        # Parameters, on a log scale.
        bar_y = cy - 12
        label_x = X_PARAMS + PARAMS_TRACK + 6
        if m.params_m is None:
            out.append(
                f'<rect class="none" x="{X_PARAMS}" y="{bar_y}" width="{PARAMS_TRACK}" height="12" rx="6"/>'
            )
            out.append(_text(label_x, cy - 2, "not published", "sec", 10.5))
        else:
            w = max(6.0, (math.log10(m.params_m) - 1) / 3 * PARAMS_TRACK)
            out.append(
                f'<rect class="track" x="{X_PARAMS}" y="{bar_y}" width="{PARAMS_TRACK}" height="12" rx="6"/>'
            )
            out.append(f'<rect x="{X_PARAMS}" y="{bar_y}" width="{w:.1f}" height="12" rx="6" fill="{hue}"/>')
            out.append(_text(label_x, cy - 2, _params(m.params_m), "ink", 12.5, weight="600"))
        if m.size_gb is not None:
            disk = f"{m.size_gb:g} GB" + (f" {m.precision}" if m.precision else "")
            out.append(_text(label_x, cy + 13, disk, "sec", 10.5))

        # Cameras: a car seen from above, one wedge per view the model reads.
        gx, gy = X_CAMS + 24, cy
        seen = views(m)
        out += [_wedge(gx, gy, yaw, fov, 24, hue) for _, yaw, fov in seen]
        out.append(f'<rect class="car" x="{gx - 5}" y="{gy - 9}" width="10" height="18" rx="3"/>')
        out.append(
            _text(
                X_CAMS + 54,
                cy - 2,
                f"{len(seen)} view{'s' if len(seen) != 1 else ''}",
                "ink",
                12.5,
                weight="600",
            )
        )
        if m.input_size:
            out.append(_text(X_CAMS + 54, cy + 13, f"{m.input_size[0]}×{m.input_size[1]}", "sec", 10.5))

        # History: what the model reads of the past, one tick per frame or state, on a
        # shared time axis ending now. Dense ticks are a high rate.
        frames, ego, caption = history(m)
        now = X_HIST + HIST_W
        out.append(f'<rect class="track" x="{X_HIST}" y="{cy - 15.5}" width="{HIST_W}" height="1"/>')
        out.append(f'<rect class="track" x="{X_HIST}" y="{cy - 3.5}" width="{HIST_W}" height="1"/>')
        out.append(f'<rect class="now" x="{now}" y="{cy - 23}" width="1" height="24"/>')
        out += [_tick(now - a * past, cy - 20, 3, 10, hue) for a in frames]
        out += [_tick(now - a * past, cy - 6.5, 1.5, 7, hue) for a in ego]
        out.append(_text(X_HIST, cy + 15, escape(caption), "sec", 10))
        if note := history_note(m):
            out.append(_text(X_HIST, cy + 27, escape(note), "sec", 10))

        # Plan: one tick per point the model emits, from now out to its horizon.
        out.append(f'<rect class="now" x="{X_PLAN}" y="{cy - 23}" width="1" height="24"/>')
        if m.horizon_s is None:
            out.append(
                f'<rect class="none" x="{X_PLAN + 2}" y="{cy - 14}" width="{PLAN_W - 2}" height="10" rx="5"/>'
            )
            head, rate = "none", ""
        else:
            w = m.horizon_s * future
            out.append(
                f'<rect class="track" x="{X_PLAN + 1}" y="{cy - 14}" width="{PLAN_W - 1}" height="10"/>'
            )
            out.append(
                f'<rect x="{X_PLAN + 1}" y="{cy - 14}" width="{w:.1f}" height="10" fill="{hue}"'
                f' opacity="{0.35 if m.plan_points_hz else 0.85}"/>'
            )
            if m.output_kind == "controls":
                out.append(
                    f'<rect x="{X_PLAN + 1}" y="{cy - 14}" width="{w:.1f}" height="10" fill="url(#stripes)"/>'
                )
            if m.plan_points_hz:
                points = round(m.horizon_s * m.plan_points_hz)
                out += [
                    _tick(X_PLAN + 1 + k / m.plan_points_hz * future, cy - 15, 1, 12, hue)
                    for k in range(1, points + 1)
                ]
            head, rate = f"{m.horizon_s:g} s", f"{m.plan_points_hz:g} Hz" if m.plan_points_hz else "uneven"
        out.append(_text(X_PLAN + PLAN_W + 6, cy - 4, head, "ink", 12.5, weight="600"))
        if rate:
            out.append(_text(X_PLAN + PLAN_W + 6, cy + 11, rate, "sec", 10.5))

        # Trained on: chips, flowed onto as many lines as they need.
        lines = flowed[i]
        if not lines[0]:
            out.append(_text(X_TRAIN, cy + 4, "not recorded", "sec", 10.5))
        y0 = cy - (21 * len(lines) - 4) / 2
        for li, line in enumerate(lines):
            x = X_TRAIN
            for text, is_own in line:
                w = _chip_width(text)
                box, ink = ("own", "ownt") if is_own else ("chip", "chipt")
                cy0 = y0 + 21 * li
                out.append(f'<rect class="{box}" x="{x:g}" y="{cy0:g}" width="{w:g}" height="17" rx="8.5"/>')
                out.append(_text(x + w / 2, cy0 + 12.5, escape(text), ink, 10.5, "middle"))
                x += w + 4

    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_models() -> str:
    registry = load_models()
    missing = [name for name in MODELS if name not in registry]
    if missing:
        raise ValueError(f"no model.yaml for {missing}")
    own = set(load_datasets())
    return render_models([(label, registry[name]) for name, label in MODELS.items()], own)


# -----------------------------------------------------------------------------


def figures() -> dict[Path, str]:
    return {MATRIX_SVG: build_matrix(), MODELS_SVG: build_models()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if either SVG is stale")
    args = ap.parse_args(argv)
    stale = []
    for path, svg in figures().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                stale.append(path)
        else:
            path.write_text(svg, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    for path in stale:
        print(f"{path.relative_to(ROOT)} is stale: run python -m tools.readme_figures", file=sys.stderr)
    return int(bool(stale))


if __name__ == "__main__":
    sys.exit(main())
