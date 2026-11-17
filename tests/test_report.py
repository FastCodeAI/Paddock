"""What `paddock report` ranks together, and what it says about a row."""

from __future__ import annotations

import re
from pathlib import Path

from paddock.registry import load_models
from paddock.results import report, store


def _run(
    root: Path, bench: str, model: str, run_id: str, scenes: str, score: float, n: int, env=None, config=None
):
    config = {**({"model_env": env} if env else {}), **(config or {})}
    run = store.RunRecord(
        run_id=run_id,
        site="test",
        benchmark=bench,
        model=model,
        dataset={"sequences_sha256": scenes, "n_samples": n},
        config=config,
    )
    samples = [{"sample_id": f"{scenes}-{i}", "metrics": {"score": score}} for i in range(n)]
    metrics = store.MetricsRecord.from_samples(run_id, samples, primary_metric="score")
    store.write_run(run, metrics, samples, root=root)


def _tables(text: str) -> list[list[str]]:
    """Each table's data rows, in order, split on the header row."""
    tables: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("| model |"):
            tables.append([])
        elif tables and line.startswith("| ") and not line.startswith("|---"):
            tables[-1].append(line)
    return tables


def test_two_scene_sets_never_share_a_table(tmp_path: Path) -> None:
    """The easy ten-scene row must not outrank the thirty-scene rows."""
    _run(tmp_path, "b", "easy", "r1", "aaaa1111", 0.9, 10)
    _run(tmp_path, "b", "hard", "r2", "bbbb2222", 0.5, 30)
    _run(tmp_path, "b", "harder", "r3", "bbbb2222", 0.4, 30)

    text = report.markdown(tmp_path)
    large, small = _tables(text)

    assert [row.split("|")[1].strip() for row in large] == ["hard", "harder"]
    assert [row.split("|")[1].strip() for row in small] == ["easy"]
    assert "### n = 30, scene set bbbb2222" in text
    assert "### n = 10, scene set aaaa1111" in text


def test_the_largest_scene_set_comes_first_whatever_it_scored(tmp_path: Path) -> None:
    _run(tmp_path, "b", "m", "small", "aaaa1111", 1.0, 3)
    _run(tmp_path, "b", "m", "large", "bbbb2222", 0.1, 200)

    assert report.markdown(tmp_path).index("n = 200") < report.markdown(tmp_path).index("n = 3")


def test_a_benchmark_with_one_scene_set_keeps_its_plain_table(tmp_path: Path) -> None:
    _run(tmp_path, "b", "m", "r1", "aaaa1111", 0.9, 10)
    _run(tmp_path, "b", "n", "r2", "aaaa1111", 0.5, 10)

    text = report.markdown(tmp_path)

    assert "###" not in text
    assert len(_tables(text)) == 1


def test_a_setting_that_changes_the_score_is_said_under_the_row(tmp_path: Path) -> None:
    _run(tmp_path, "navsim", "meteor", "r1", "aaaa1111", 0.3, 10, env={"METEOR_HORIZON_S": "4.0"})

    text = report.markdown(tmp_path)

    assert "| meteor [1] | r1 |" in text
    assert "- [1] meteor: plan continued to 4.0 s by paddock" in text


def test_restating_the_declared_default_is_not_a_note(tmp_path: Path) -> None:
    """`ring` is the model as trained, so a run that spells it out needs no warning."""
    _run(tmp_path, "b", "qwendrive", "r1", "aaaa1111", 0.3, 10, env={"QWENDRIVE_VIEWS": "ring"})

    assert "| qwendrive | r1 |" in report.markdown(tmp_path)


def test_a_setting_that_moves_no_score_is_not_a_note(tmp_path: Path) -> None:
    """Most ranked runs set a thread count: marking those would mark nearly every row."""
    _run(tmp_path, "b", "visionpilot", "r1", "aaaa1111", 0.3, 10, env={"VISIONPILOT_ORT_THREADS": "2"})

    text = report.markdown(tmp_path)

    assert "| visionpilot | r1 |" in text
    assert "[1]" not in text


def test_a_streamed_run_says_so_under_its_row(tmp_path: Path) -> None:
    """Drive-JEPA's momentum-aware rows feed each log, scene or warm-up in order to one node; the standard
    protocol scores each sample on its own. Ranked, but a reader must see which it is."""
    _run(tmp_path, "navsim", "drivejepa", "r1", "aaaa1111", 0.8, 10, config={"streamed": "per_log"})
    _run(tmp_path, "navsim", "constant_velocity", "r2", "aaaa1111", 0.2, 10)

    text = report.markdown(tmp_path)

    assert "| drivejepa [1] | r1 |" in text and "| constant_velocity | r2 |" in text
    assert "- [1] drivejepa: streamed (per_log)" in text


def test_a_run_of_a_model_no_longer_registered_still_renders(tmp_path: Path) -> None:
    _run(tmp_path, "b", "alpamayo", "r1", "aaaa1111", 0.3, 10, env={"ANYTHING": "1"})

    assert "| alpamayo | r1 |" in report.markdown(tmp_path)


def test_notes_are_numbered_per_table_and_shared_by_the_same_setting(tmp_path: Path) -> None:
    env = {"METEOR_HORIZON_S": "4.0"}
    _run(tmp_path, "b", "meteor", "r1", "aaaa1111", 0.3, 10, env=env)
    _run(tmp_path, "b", "meteor", "r2", "aaaa1111", 0.2, 10, env=env)
    _run(tmp_path, "b", "meteor", "r3", "bbbb2222", 0.1, 5, env=env)

    text = report.markdown(tmp_path)

    assert text.count("| meteor [1] |") == 3
    assert text.count("- [1] meteor:") == 2
    assert "[2]" not in text


def test_the_stored_stretched_meteor_run_carries_its_note() -> None:
    """The row the issue was filed about, in the real store."""
    navsim = report.markdown().split("## navsim\n", 1)[1].split("\n## ", 1)[0]

    # Its mark's number depends on the rows above it in the table (Drive-JEPA's streamed row has one too).
    mark = re.search(r"\| meteor \[(\d+)\] \| 20260914T234307-e56cfc \|", navsim)
    assert mark, "the stored stretched METEOR run carries no mark"
    assert f"- [{mark.group(1)}] meteor: plan continued to 4.0 s by paddock" in navsim


def test_every_note_names_a_setting_its_model_or_record_uses() -> None:
    """A misspelt key would never match a run, and the note would silently never print."""
    recorded = {key for run, _ in store.load_runs() for key in (run.config.get("model_env") or {})}
    for spec in load_models().values():
        for key, text in spec.score_notes.items():
            assert key in recorded, f"{spec.name}: no stored run sets {key}"
            assert text.strip(), f"{spec.name}: {key} has no line"
