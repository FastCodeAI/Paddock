"""The run/segment alignment logic in annotate_reasoning_video.py, without ffmpeg."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmarks" / "neuroncap"))

from annotate_reasoning_video import (  # noqa: E402
    _episode_dirs,
    _run_dirs,
    _segment_reasoning,
    _select_runs_and_segments,
)


def test_segment_reasoning_splits_on_status_marker() -> None:
    entries = [
        {"status": "no frame seen yet"},
        {"wall_time": 1.0, "reasoning": "a"},
        {"wall_time": 2.0, "reasoning": "b"},
        {"status": "no frame seen yet"},
        {"wall_time": 3.0, "reasoning": "c"},
    ]
    assert _segment_reasoning(entries) == [
        [{"wall_time": 1.0, "reasoning": "a"}, {"wall_time": 2.0, "reasoning": "b"}],
        [{"wall_time": 3.0, "reasoning": "c"}],
    ]


def test_segment_reasoning_no_trailing_marker_keeps_last_segment() -> None:
    entries = [
        {"wall_time": 1.0, "reasoning": "a"},
        {"status": "no frame seen yet"},
        {"wall_time": 2.0, "reasoning": "b"},
    ]
    assert _segment_reasoning(entries) == [
        [{"wall_time": 1.0, "reasoning": "a"}],
        [{"wall_time": 2.0, "reasoning": "b"}],
    ]


def test_segment_reasoning_empty_input() -> None:
    assert _segment_reasoning([]) == []


def test_episode_dirs_ordered_by_stdout_log_mtime(tmp_path: Path) -> None:
    older, newer = tmp_path / "side-0103", tmp_path / "frontal-0103"
    for d in (newer, older):
        d.mkdir()
    (newer / "stdout.log").write_text("")
    time.sleep(0.01)
    (older / "stdout.log").write_text("")
    (tmp_path / "no_log_here").mkdir()

    dirs = _episode_dirs(tmp_path)

    assert [d.name for d in dirs] == ["frontal-0103", "side-0103"]


def test_run_dirs_sorted_numerically_not_lexically(tmp_path: Path) -> None:
    episode = tmp_path / "side-0103"
    episode.mkdir()
    for name in ("run_10", "run_2", "run_1"):
        (episode / name).mkdir()

    dirs = _run_dirs(episode)

    assert [d.name for d in dirs] == ["run_1", "run_2", "run_10"]


def test_run_dirs_falls_back_to_episode_dir_when_no_run_subdirs(tmp_path: Path) -> None:
    episode = tmp_path / "side-0103"
    episode.mkdir()

    assert _run_dirs(episode) == [episode]


def test_segment_count_matches_run_count_for_a_real_sweep_shape(tmp_path: Path) -> None:
    """Regression guard for the strict=True zip in main(): a well-formed sweep's
    run-dir count and reasoning-segment count must agree, one segment per run."""
    sweep = tmp_path / "sweep"
    episode = sweep / "side-0103"
    for i in range(3):
        (episode / f"run_{i}").mkdir(parents=True)
    (episode / "stdout.log").write_text("")

    entries: list[dict] = []
    for i in range(3):
        entries.append({"status": "no frame seen yet"})
        entries.append({"wall_time": float(i), "reasoning": f"run {i}"})
    reasoning_path = sweep / "reasoning.jsonl"
    reasoning_path.write_text("\n".join(json.dumps(e) for e in entries))

    segments = _segment_reasoning([json.loads(line) for line in reasoning_path.read_text().splitlines()])
    all_runs = [(ep, run) for ep in _episode_dirs(sweep) for run in _run_dirs(ep)]

    assert len(all_runs) == len(segments) == 3


def test_mismatched_run_and_segment_counts_raise_instead_of_silently_misaligning(tmp_path: Path) -> None:
    """Regression guard for the `strict=True` zip in `main()`: unlike
    `test_segment_count_matches_run_count_for_a_real_sweep_shape` above (which
    only covers the well-formed case), this builds a sweep with one more run
    dir than reasoning segment and asserts the zip actually raises, rather
    than silently dropping the extra run -- the bug `strict=True` was added to
    fix."""
    sweep = tmp_path / "sweep"
    episode = sweep / "side-0103"
    for i in range(3):
        (episode / f"run_{i}").mkdir(parents=True)
    (episode / "stdout.log").write_text("")

    entries: list[dict] = []
    for i in range(2):
        entries.append({"status": "no frame seen yet"})
        entries.append({"wall_time": float(i), "reasoning": f"run {i}"})
    reasoning_path = sweep / "reasoning.jsonl"
    reasoning_path.write_text("\n".join(json.dumps(e) for e in entries))

    segments = _segment_reasoning([json.loads(line) for line in reasoning_path.read_text().splitlines()])
    all_runs, segments = _select_runs_and_segments(sweep, segments, None, None)

    assert len(all_runs) == 3
    assert len(segments) == 2
    with pytest.raises(ValueError):
        list(zip(all_runs, segments, strict=True))


def _multi_episode_sweep(tmp_path: Path, episode_run_counts: dict[str, int]) -> tuple[Path, list[list[dict]]]:
    """A sweep with several episodes, created in dict order -- so the first
    key is oldest by `stdout.log` mtime and sorts first in episode order.
    Each run's segment carries its own episode name and run index, so a test
    can tell "episode 0's segment" from "episode 1's segment" apart, not just
    count them."""
    sweep = tmp_path / "sweep"
    entries: list[dict] = []
    for name, n_runs in episode_run_counts.items():
        episode = sweep / name
        for i in range(n_runs):
            (episode / f"run_{i}").mkdir(parents=True)
        (episode / "stdout.log").write_text("")
        time.sleep(0.01)
        for i in range(n_runs):
            entries.append({"status": "no frame seen yet"})
            entries.append({"wall_time": float(i), "reasoning": f"{name} run {i}"})
    reasoning_path = sweep / "reasoning.jsonl"
    reasoning_path.write_text("\n".join(json.dumps(e) for e in entries))
    segments = _segment_reasoning([json.loads(line) for line in reasoning_path.read_text().splitlines()])
    return sweep, segments


def test_episode_only_narrowing_slices_segments_to_that_episodes_own_span(tmp_path: Path) -> None:
    """Regression for the bug where `--episode` alone left `segments` as the
    whole sweep's, so a non-first episode got captioned with episode 0's
    reasoning. "side-0103" here is the second (newer) episode by mtime, so its
    segments start at offset 2, not 0."""
    sweep, segments = _multi_episode_sweep(tmp_path, {"frontal-0103": 2, "side-0103": 3})

    all_runs, narrowed = _select_runs_and_segments(sweep, segments, "side-0103", None)

    assert [rd.name for _, rd in all_runs] == ["run_0", "run_1", "run_2"]
    assert [seg[0]["reasoning"] for seg in narrowed] == [
        "side-0103 run 0",
        "side-0103 run 1",
        "side-0103 run 2",
    ]


def test_episode_and_run_narrows_to_exactly_one_pair(tmp_path: Path) -> None:
    sweep, segments = _multi_episode_sweep(tmp_path, {"frontal-0103": 2, "side-0103": 3})

    all_runs, narrowed = _select_runs_and_segments(sweep, segments, "side-0103", 1)

    assert [rd.name for _, rd in all_runs] == ["run_1"]
    assert [seg[0]["reasoning"] for seg in narrowed] == ["side-0103 run 1"]


def test_run_without_episode_is_rejected(tmp_path: Path) -> None:
    sweep, segments = _multi_episode_sweep(tmp_path, {"frontal-0103": 2})

    with pytest.raises(SystemExit):
        _select_runs_and_segments(sweep, segments, None, 0)
