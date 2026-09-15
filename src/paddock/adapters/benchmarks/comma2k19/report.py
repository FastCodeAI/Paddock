"""`report.md` for a set of comma2k19 runs: every declared metric, in groups.

Written beside the figures by `render`, and readable on its own -- a figure
shows a shape and a table gives the number someone is going to quote.

## Why a `_table` helper exists

The previous report built rows with f-strings and produced markdown that did not
render: a cell containing `|` closes the column wherever it appears, so a header
like `fde@3s (m) | lower is better` silently became two columns and threw the
whole table's alignment out from that row down. Every cell here goes through
`_cell`, which escapes the pipe and flattens newlines, and `_table` pads the
columns so the source is readable too. That is the entire defect: markdown
tables are a line-oriented format with exactly one metacharacter, and the fix is
to escape it in one place rather than remember it at forty call sites.

## What is reported, and what is not

Every metric the benchmark declares, grouped the way `docs/benchmarks/comma2k19.md`
groups them, with the mean and the median side by side. Both, because a mean is
what gets quoted and a median is what is true: the ratios in particular are
dominated by whichever sample had the smallest denominator.

`n` is per metric, not per run, and it is load-bearing. A model that publishes no
lead head scores NaN for every lead metric, which aggregates to `n = 0` -- the
report says "not answered" rather than printing a zero that looks like a result.

There is no composite driving score **of paddock's own**. See the benchmark's
docs for why: the components are measured against different references with
different coverage, and a weighted sum of them would hide exactly the thing each
one exists to show.

The "Driving score (proxy)" group is not a counterexample to that, and the
distinction is worth stating precisely because it is easy to erode. What is
refused is paddock *choosing* weights and calling the result a driving score.
What that group does instead is evaluate NAVSIM v1's published formula, with
NAVSIM's weights, over log-replay proxies -- reported beside all five of its
sub-scores and never as the headline. `score.py` carries the full argument and
the list of ways the proxies are weaker than the terms they stand in for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from paddock.adapters.benchmarks.comma2k19 import auxiliary as AUX
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.adapters.benchmarks.comma2k19 import score as SC

#: Worst samples listed by the primary metric, with the command to draw each.
WORST_N = 10

#: Metric groups, in report order: `(title, blurb, keys)`.
#:
#: The grouping is presentation, so it lives here rather than in `auxiliary`;
#: `tests/test_report.py` holds it to covering every declared key exactly once,
#: which is what stops a new head from being scored and never shown.
GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Trajectory",
        "The planned path against the recorded one, decomposed against the "
        "ground-truth tangent. `ade` is the headline, at the 3 s horizon.",
        (
            "ade",
            "fde@3s",
            "longitudinal_mae",
            "lateral_mae",
            "along_track_mae",
            "cross_track_mae",
            "lateral_at_arclength_mae",
            "along_track_bias",
            "path_length_ratio",
            "predicted_path_length",
            "recorded_path_length",
            "scored_horizon_s",
        ),
    ),
    (
        "Pose head",
        "The model's own ego motion against the car's CAN speed and the INS yaw "
        "rate. Independent of the trajectory: a model can plan well and still "
        "misread how fast it is going.",
        ("speed_abs_err", "speed_rel_err", "yaw_rate_abs_err", "yaw_rate_bias"),
    ),
    (
        "Lead head",
        "Against the radar's closest in-lane track. Only leads inside "
        f"{AUX.LEAD_NEAR_M:.0f} m are used as a reference -- past that the two "
        "sensors routinely hold different vehicles.",
        (
            "lead_prob",
            "lead_precision",
            "lead_recall",
            "lead_range_err",
            "lead_range_abs_err",
            "lead_lateral_abs_err",
            "lead_rel_speed_abs_err",
        ),
    ),
    (
        "Plan uncertainty",
        "How often the realised error fell inside the model's own predicted "
        "sigma. Near 0.68 and 0.95 is calibrated; near 1.0 is a model that is "
        "unsure of everything.",
        ("within_1sigma_x", "within_1sigma_y", "within_2sigma_x", "within_2sigma_y"),
    ),
    (
        "Road geometry",
        "comma2k19 records no lane labels, so these are plausibility gates and "
        "not errors. A failed gate means the calibration or the warp is wrong "
        "and that sample's other numbers are not worth reading.",
        (
            "lane_width_m",
            "lane_offset_m",
            "lane_prob",
            "road_width_m",
            "camera_height_m",
            "gate_speed",
            "gate_lane_width",
            "gate_camera_height",
            "sanity_pass",
        ),
    ),
    (
        "Safety",
        "The plan integrated forward against the radar lead. Undefined on an "
        "empty road, so a run on deserted roads cannot out-score a careful one "
        "in traffic.",
        ("lead_gap_min_m", "lead_ttc_min_s", "ttc_below_2s"),
    ),
    (
        "Comfort",
        "Read off the planned trajectory alone, so every model gets these -- "
        "including one that publishes no auxiliary heads at all. Bounds are "
        "`paddock.metrics.comfort.BOUNDS`.",
        tuple(k for k in AUX.ALL_KEYS if k.startswith("comfort_")),
    ),
    (
        "Driving score (proxy)",
        "NAVSIM v1's PDMS formula and weights evaluated over log-replay "
        f"proxies: `NC x DAC x (5*TTC + 5*EP + 2*C) / {sum(SC.WEIGHTS.values()):.0f}`. "
        "Read the sub-scores, not the composite -- NC sees only the radar lead "
        "and DAC uses the human's own path as the corridor, so this is not a "
        "PDMS and may not be compared with one. `ade` is still the headline. "
        "The miss rate is the fraction of samples whose FDE at that horizon "
        f"exceeded {SC.MISS_THRESHOLD_M:.0f} m. See `score.py`.",
        SC.keys(),
    ),
)


def _cell(value) -> str:
    r"""One markdown cell: `|` escaped, newlines flattened, None as an em dash.

    The pipe is the only metacharacter a table row has, and an unescaped one in
    any cell silently re-columns that row. `\|` is the GitHub-flavoured escape
    and renders as a literal pipe.
    """
    if value is None:
        return "--"
    return str(value).replace("|", r"\|").replace("\n", " ").strip()


def _table(headers: Sequence, rows: Sequence[Sequence]) -> str:
    """A markdown table, padded, with every cell escaped. Empty rows give "".

    Ragged rows are padded rather than rejected: a report that loses a section
    because one model was missing one metric is worse than a report with a gap
    in it, and the gap is visible.
    """
    if not len(rows):
        return ""
    head = [_cell(h) for h in headers]
    body = [[_cell(c) for c in row] + [""] * (len(head) - len(row)) for row in rows]
    widths = [max(len(head[i]), *(len(r[i]) for r in body)) for i in range(len(head))]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(head, widths, strict=True)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) + " |" for r in body]
    return "\n".join(lines)


def _stat(run, key: str, stat: str) -> float | None:
    """One aggregate from a run's metrics record, or None if it has no finite samples."""
    entry = (run.metrics.get("metrics") or {}).get(key) or {}
    value = entry.get(stat)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _num(value: float | None, places: int = 3) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def _count(run, key: str) -> int:
    return int(((run.metrics.get("metrics") or {}).get(key) or {}).get("n", 0))


def _group_table(runs, keys: Sequence[str]) -> str:
    """Metric down the rows, model across the columns, mean and median in each.

    Metric-major rather than model-major because a report is read one metric at
    a time -- "who is better at cross-track" -- and because the row count is
    fixed by the benchmark while the column count is whatever ran.
    """
    headers = ["metric"] + [f"{r.model} mean / median (n)" for r in runs]
    rows = []
    for key in keys:
        if not any(_count(r, key) for r in runs):
            # Declared, produced by nobody here: shown, so "not answered" is
            # visible, but with nothing dressed up as a measurement.
            rows.append([f"`{key}`"] + ["not answered"] * len(runs))
            continue
        cells = []
        for run in runs:
            n = _count(run, key)
            if not n:
                cells.append("not answered")
            else:
                cells.append(f"{_num(_stat(run, key, 'mean'))} / {_num(_stat(run, key, 'median'))} ({n})")
        rows.append([f"`{key}`"] + cells)
    return _table(headers, rows)


def _horizons(runs) -> str:
    """ADE and FDE at every reported horizon, which is the shape of the failure.

    A model that is wrong from the first waypoint and one that drifts only late
    can share an `ade`; they cannot share this table.
    """
    headers = ["horizon"] + [f"{r.model} ADE / FDE (n)" for r in runs]
    rows = []
    for horizon in MC.REPORTED_HORIZONS_S:
        ade, fde = f"ade@{horizon:g}s", f"fde@{horizon:g}s"
        cells = []
        for run in runs:
            n = _count(run, ade)
            cells.append(
                f"{_num(_stat(run, ade, 'mean'), 2)} / {_num(_stat(run, fde, 'mean'), 2)} ({n})"
                if n
                else "beyond the plan"
            )
        rows.append([f"{horizon:g} s"] + cells)
    return _table(headers, rows)


def _headline(runs, primary: str) -> str:
    headers = ["model", "run", "n", f"{primary} mean", f"{primary} median", "cross-track MAE", "valid"]
    rows = [
        [
            r.model,
            f"`{r.run_id}`",
            (r.metrics.get("n") or len(r.samples)),
            _num(_stat(r, primary, "mean")),
            _num(_stat(r, primary, "median")),
            _num(_stat(r, "cross_track_mae", "mean")),
            "yes" if r.run.get("valid", True) else f"no -- {r.run.get('notes', '')}",
        ]
        for r in runs
    ]
    return _table(headers, rows)


def _splits(runs) -> str:
    """Per-split means. The split is the vehicle, which is also the camera mount."""
    names = sorted({s for r in runs for s in (r.metrics.get("by_split") or {})})
    if not names:
        return ""
    headers = ["split"] + [f"{r.model} mean (n)" for r in runs]
    rows = []
    for name in names:
        cells = []
        for run in runs:
            split = (run.metrics.get("by_split") or {}).get(name) or {}
            entry = split.get("ade") or {}
            mean = entry.get("mean")
            cells.append("--" if mean is None else f"{float(mean):.3f} ({split.get('n', 0)})")
        rows.append([name] + cells)
    return _table(headers, rows)


def _worst(run, primary: str, benchmark: str = "comma2k19") -> str:
    """The samples this run did worst on, with the command that draws each.

    The command rather than a still image: a still is one frame and the thing
    worth looking at is the eight seconds leading into it, which is what
    `render.render_sample` produces.
    """
    scored = [
        (s, float(s["metrics"][primary]))
        for s in run.samples
        if isinstance((s.get("metrics") or {}).get(primary), (int, float))
        and math.isfinite(s["metrics"][primary])
    ]
    if not scored:
        return ""
    scored.sort(key=lambda pair: -pair[1])
    rows = [
        [
            f"`{sample['sample_id']}`",
            f"{value:.2f}",
            _num(_finite_metric(sample, "path_length_ratio"), 2),
            f"`paddock render --bench {benchmark} --sample {sample['sample_id']}`",
        ]
        for sample, value in scored[:WORST_N]
    ]
    return _table(["sample", primary, "path length ratio", "draw it"], rows)


def _finite_metric(sample, key: str) -> float | None:
    value = (sample.get("metrics") or {}).get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _provenance(runs) -> str:
    """What would have to match for two of these numbers to be comparable."""
    headers = ["model", "run", "started (UTC)", "samples", "sample set", "history", "input Hz", "keyframes"]
    rows = []
    for run in runs:
        data = run.run.get("dataset") or {}
        config = run.run.get("config") or {}
        rows.append(
            [
                run.model,
                f"`{run.run_id}`",
                run.run.get("started_utc") or "--",
                data.get("n_samples", "--"),
                f"`{data.get('sequences_sha256', '--')}`",
                config.get("history_frames", "--"),
                config.get("input_hz", "--"),
                config.get("keyframes_per_segment", "--"),
            ]
        )
    return _table(headers, rows)


def write(runs, out_dir: Path, figures: Sequence[Path] = (), primary: str = "ade") -> Path:
    """Write `report.md` into `out_dir` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "# comma2k19",
        "",
        "Open-loop trajectory scoring against comma2k19's own recorded future pose, "
        "plus everything else the model publishes. Generated by "
        "`paddock render --bench comma2k19`; see `docs/benchmarks/comma2k19.md` for "
        "what each metric means and why it is here.",
        "",
        "## Headline",
        "",
        _headline(runs, primary),
        "",
        "## Error against horizon",
        "",
        _horizons(runs),
        "",
    ]
    for title, blurb, keys in GROUPS:
        table = _group_table(runs, keys)
        if not table:
            continue
        parts += [f"## {title}", "", blurb, "", table, ""]

    splits = _splits(runs)
    if splits:
        parts += [
            "## By split",
            "",
            "The split is the vehicle, and therefore also the camera mount and the "
            "radar. A gap here is a calibration result, not a driving one.",
            "",
            splits,
            "",
        ]

    for run in runs:
        worst = _worst(run, primary)
        if worst:
            parts += [f"## Worst {WORST_N}: {run.model}", "", worst, ""]

    if figures:
        parts += ["## Figures", ""]
        parts += [f"![{Path(f).stem}]({Path(f).name})" for f in figures]
        parts += [""]

    parts += ["## Provenance", "", _provenance(runs), ""]

    path = out_dir / "report.md"
    path.write_text("\n".join(parts))
    return path
