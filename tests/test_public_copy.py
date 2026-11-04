"""`tools.public_copy`: the one place run records are rewritten, so it must touch little and stop early.

Everything runs on a small copy made in a temporary directory. No rule here is real:
the real ones are private, because a name written in a tracked file reaches the copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paddock.results import store
from tools import public_copy

RULES = """
[site]
"old-machine" = "site-01"
"old-pool" = "site-02"

[[note]]
run = "bench/model/run-a"
old = " (see PR 12)"
new = ""

[forbid]
words = ["secretword"]
"""


def _run(copy: Path, run_id: str, site: str, notes: str = "", config: dict | None = None) -> Path:
    run = store.RunRecord(
        run_id=run_id, site=site, benchmark="bench", model="model", notes=notes, config=config or {}
    )
    metrics = store.MetricsRecord(run_id=run_id, primary_metric="ade", n=3, metrics={"ade": {"mean": 1.5}})
    return store.write_run(run, metrics, root=copy / "results")


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "leaderboard.md").write_text("old leaderboard\n")
    _run(root, "run-a", "old-machine", notes="Kept as a check (see PR 12).", config={"seed": 1})
    _run(root, "run-b", "old-machine", config={"source_sites": ["old-pool", "old-machine"]})
    return root


def _clean(copy: Path, tmp_path: Path, rules: str = RULES) -> int:
    path = tmp_path / "rules.toml"
    path.write_text(rules)
    return public_copy.main([str(copy), "--rules", str(path)])


def _record(copy: Path, run_id: str) -> dict:
    return json.loads((copy / "results" / "bench" / "model" / run_id / "run.json").read_text())


def _files(copy: Path) -> dict[str, bytes]:
    return {p.relative_to(copy).as_posix(): p.read_bytes() for p in copy.rglob("*") if p.is_file()}


def test_labels_and_notes_change_and_nothing_else(copy: Path, tmp_path: Path) -> None:
    before = _files(copy)
    a_before, b_before = _record(copy, "run-a"), _record(copy, "run-b")

    assert _clean(copy, tmp_path) == 0

    a, b = _record(copy, "run-a"), _record(copy, "run-b")
    assert a["site"] == b["site"] == "site-01"
    assert a["notes"] == "Kept as a check."
    assert b["config"]["source_sites"] == ["site-02", "site-01"]
    # Every other field, the run's id above all: docs and the matrix name runs by it.
    assert {k: v for k, v in a.items() if k not in ("site", "notes")} == {
        k: v for k, v in a_before.items() if k not in ("site", "notes")
    }
    assert b["run_id"] == b_before["run_id"]

    after = _files(copy)
    assert set(after) == set(before), "a file appeared or vanished"
    scores = [name for name in before if name.endswith("metrics.json")]
    assert len(scores) == 2 and all(after[name] == before[name] for name in scores)


def test_the_format_of_a_record_is_what_the_store_writes(copy: Path, tmp_path: Path) -> None:
    """A reformatted record would turn one changed value into a whole-file diff."""
    assert _clean(copy, tmp_path) == 0
    path = copy / "results" / "bench" / "model" / "run-b" / "run.json"
    assert path.read_text() == json.dumps(json.loads(path.read_text()), indent=2) + "\n"


def test_the_leaderboard_is_rebuilt_from_the_changed_records(copy: Path, tmp_path: Path) -> None:
    (copy / "results" / "bench" / "model" / "run-a" / "run.json").write_text(
        json.dumps({**_record(copy, "run-a"), "valid": False}, indent=2) + "\n"
    )
    assert _clean(copy, tmp_path) == 0
    board = (copy / "docs" / "leaderboard.md").read_text()
    assert "Kept as a check." in board and "PR 12" not in board


@pytest.mark.parametrize(
    ("rules", "said"),
    [
        (RULES.replace('"old-machine" = "site-01"\n', ""), "no rule for the label 'old-machine'"),
        (RULES.replace('"old-pool" = "site-02"\n', ""), "no rule for the label 'old-pool'"),
        (RULES.replace("see PR 12", "see PR 13"), "found it 0 times"),
        (RULES.replace("bench/model/run-a", "bench/model/gone"), "not in the copy"),
    ],
)
def test_a_rule_that_does_not_fit_changes_nothing(
    copy: Path, tmp_path: Path, capsys, rules: str, said: str
) -> None:
    """A label with no rule would go public as it is, so one stops the whole run, half-done never."""
    before = _files(copy)
    assert _clean(copy, tmp_path, rules) == 1
    assert said in capsys.readouterr().err
    assert _files(copy) == before


@pytest.mark.parametrize("make", [Path.mkdir, Path.touch], ids=["clone", "worktree"])
def test_a_checkout_is_refused(copy: Path, tmp_path: Path, capsys, make) -> None:
    """Run records in a checkout are the real ones. A worktree's .git is a file."""
    make(copy / ".git")
    before = _files(copy)
    assert _clean(copy, tmp_path) == 2
    assert "checkout" in capsys.readouterr().err
    assert _files(copy) == before


def test_a_forbidden_word_left_behind_fails_and_says_where(copy: Path, tmp_path: Path, capsys) -> None:
    (copy / "docs" / "page.md").write_text("fine\nThis names SecretWord.\n")
    assert _clean(copy, tmp_path) == 1
    assert "docs/page.md:2: secretword" in capsys.readouterr().err


def test_a_site_label_is_forbidden_without_being_listed(copy: Path, tmp_path: Path, capsys) -> None:
    """A doc quoting a label is the leak the records' rewrite cannot reach."""
    (copy / "docs" / "page.md").write_text("Made on old-pool.\n")
    assert _clean(copy, tmp_path) == 1
    assert "docs/page.md:1: old-pool" in capsys.readouterr().err


def test_an_allowed_file_may_hold_a_forbidden_word(copy: Path, tmp_path: Path) -> None:
    (copy / "docs" / "page.md").write_text("This names secretword.\n")
    assert _clean(copy, tmp_path, RULES + 'allow = ["docs/page.md"]\n') == 0
