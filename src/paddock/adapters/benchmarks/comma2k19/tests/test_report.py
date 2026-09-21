"""Tests for `comma2k19.report`.

Two things are worth testing about a report. First that it **renders**: the
previous one did not, because a header cell contained a `|` and silently
re-columned the table from that row down, and nothing caught it because the
generator ran perfectly and produced a file. So the tests here parse the output
back as a table rather than checking that the string contains what was put into
it.

Second, that it **covers what the benchmark declares**. A metric that is scored
and never shown is worse than one that was never scored: the cost was paid and
the answer thrown away.
"""

from __future__ import annotations

import math
import types

import pytest
import yaml

from paddock.adapters.benchmarks.comma2k19 import auxiliary as AUX
from paddock.adapters.benchmarks.comma2k19 import metrics as MC
from paddock.adapters.benchmarks.comma2k19 import report as RP


def _rows(table: str) -> list[list[str]]:
    """A rendered table back into cells, the way a markdown reader sees it.

    Splits on unescaped pipes only -- which is precisely the distinction the
    module exists to get right.
    """
    out = []
    for line in table.splitlines():
        stripped = line.strip()
        assert stripped.startswith("|") and stripped.endswith("|"), line
        body = stripped[1:-1]
        cells, current, escaped = [], "", False
        for char in body:
            if escaped:
                current += char
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "|":
                cells.append(current.strip())
                current = ""
            else:
                current += char
        cells.append(current.strip())
        out.append(cells)
    return out


# --------------------------------------------------------------------------
# The escaping, which is the whole reason this module exists
# --------------------------------------------------------------------------


def test_a_pipe_in_a_header_does_not_open_a_new_column():
    """The exact defect in the previous report, as a test."""
    table = RP._table(["fde@3s (m) | lower is better", "value"], [["a", "b"]])
    rows = _rows(table)
    assert len(rows[0]) == 2, rows[0]
    assert rows[0][0] == "fde@3s (m) | lower is better"
    assert all(len(row) == 2 for row in rows), rows


def test_a_pipe_in_a_body_cell_does_not_open_a_new_column():
    rows = _rows(RP._table(["a", "b"], [["x|y", "z"]]))
    assert rows[-1] == ["x|y", "z"]


def test_every_row_has_the_same_column_count_including_the_separator():
    table = RP._table(["one", "two", "three"], [["a|b", "c", "d"], ["e", "f", "g"]])
    rows = _rows(table)
    assert {len(r) for r in rows} == {3}
    assert len(rows) == 4  # header, separator, two body rows


def test_a_ragged_row_is_padded_rather_than_dropped():
    """A gap is visible; a missing section is not."""
    rows = _rows(RP._table(["a", "b", "c"], [["only one"]]))
    assert rows[-1] == ["only one", "", ""]


def test_a_newline_in_a_cell_is_flattened_because_a_row_is_one_line():
    rows = _rows(RP._table(["a"], [["two\nlines"]]))
    assert rows[-1] == ["two lines"]


def test_no_rows_gives_no_table_rather_than_a_header_with_nothing_under_it():
    assert RP._table(["a", "b"], []) == ""


# --------------------------------------------------------------------------
# Coverage of the declared metric set
# --------------------------------------------------------------------------


def test_every_auxiliary_metric_appears_in_exactly_one_group():
    """`GROUPS` is presentation, so it can drift from `ALL_KEYS`; this is the
    thing that notices when it does."""
    grouped = [key for _, _, keys in RP.GROUPS for key in keys]
    assert len(grouped) == len(set(grouped)), "a key is in two groups"
    missing = [key for key in AUX.ALL_KEYS if key not in set(grouped)]
    assert not missing, missing


def test_every_grouped_metric_is_one_the_benchmark_declares():
    """The other direction: a group naming a metric nothing produces would print
    "not answered" forever and read as a model failing rather than a typo."""
    from pathlib import Path

    spec = yaml.safe_load((Path(RP.__file__).parent / "benchmark.yaml").read_text())
    declared = set(spec["metrics"])
    grouped = {key for _, _, keys in RP.GROUPS for key in keys}
    assert grouped <= declared, sorted(grouped - declared)


def test_the_horizon_table_covers_every_reported_horizon():
    run = _run("openpilot", {})
    rows = _rows(RP._horizons([run]))
    labels = [row[0] for row in rows[2:]]
    assert labels == [f"{h:g} s" for h in MC.REPORTED_HORIZONS_S]


# --------------------------------------------------------------------------
# What the numbers mean when they are missing
# --------------------------------------------------------------------------


def _run(model: str, metrics: dict, samples=None, **run_fields):
    return types.SimpleNamespace(
        model=model,
        run_id="20260101T000000-abc123",
        run={"valid": True, "dataset": {"n_samples": 2}, "config": {}, **run_fields},
        metrics={"n": 2, "metrics": metrics, "by_split": {}},
        samples=samples or [],
    )


def test_a_head_nobody_published_reads_as_not_answered_and_never_as_zero():
    """The NaN contract, at the point where a reader sees it. `n = 0` is the
    aggregate of an all-NaN metric, and printing 0.000 there would be a lie
    about a model that simply has no such head."""
    run = _run("cv", {"lead_range_abs_err": {"n": 0, "n_missing": 2}})
    rows = _rows(RP._group_table([run], ("lead_range_abs_err",)))
    assert rows[-1] == ["`lead_range_abs_err`", "not answered"]
    assert "0.000" not in RP._group_table([run], ("lead_range_abs_err",))


def test_a_published_head_reports_mean_median_and_the_sample_count():
    run = _run("openpilot", {"speed_abs_err": {"n": 7, "mean": 0.5, "median": 0.25}})
    rows = _rows(RP._group_table([run], ("speed_abs_err",)))
    assert rows[-1] == ["`speed_abs_err`", "0.500 / 0.250 (7)"]


def test_a_horizon_past_the_plan_says_so_rather_than_showing_a_shorter_one():
    run = _run("short", {"ade@1s": {"n": 4, "mean": 0.2}, "fde@1s": {"n": 4, "mean": 0.3}})
    rows = {row[0]: row[1] for row in _rows(RP._horizons([run]))[2:]}
    assert rows["1 s"].startswith("0.20 / 0.30")
    assert rows["10 s"] == "beyond the plan"


def test_a_nan_aggregate_is_treated_as_absent_rather_than_printed():
    run = _run("odd", {"ade": {"n": 3, "mean": float("nan"), "median": 1.0}})
    assert RP._stat(run, "ade", "mean") is None
    assert math.isclose(RP._stat(run, "ade", "median"), 1.0)


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------


def _sample(sample_id: str, ade: float, ratio: float = 1.0):
    return {"sample_id": sample_id, "metrics": {"ade": ade, "path_length_ratio": ratio}}


def test_write_produces_a_document_whose_every_table_parses():
    samples = [_sample(f"comma2k19/1/route|x/{i}/159", float(i)) for i in range(3)]
    run = _run("openpilot", {"ade": {"n": 3, "mean": 1.0, "median": 1.0}}, samples=samples)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = RP.write([run], Path(tmp), figures=[Path(tmp) / "decomposition.png"])
        text = path.read_text()
    # Every contiguous run of table lines must parse, and a sample id containing
    # a `|` -- which every comma2k19 route id does -- must not break one.
    block: list[str] = []
    tables = 0
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            rows = _rows("\n".join(block))
            assert len({len(r) for r in rows}) == 1, rows
            tables += 1
            block = []
    assert tables >= 4
    assert "![decomposition](decomposition.png)" in text


def test_the_worst_table_is_ordered_worst_first_and_names_the_draw_command():
    samples = [_sample("comma2k19/1/r|x/3/159", 1.0), _sample("comma2k19/1/r|x/4/159", 9.0)]
    run = _run("openpilot", {}, samples=samples)
    rows = _rows(RP._worst(run, "ade"))[2:]
    assert [row[1] for row in rows] == ["9.00", "1.00"]
    assert "paddock render --bench comma2k19 --sample" in rows[0][3]


@pytest.mark.parametrize("value, expected", [(None, "--"), ("a|b", r"a\|b"), (1.5, "1.5")])
def test_cell_escapes_only_what_it_must(value, expected):
    assert RP._cell(value) == expected
