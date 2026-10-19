"""The run/segment alignment logic in annotate_reasoning_video.py, without ffmpeg."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "benchmarks" / "neuroncap"))

from annotate_reasoning_video import _episode_dirs, _run_dirs, _segment_reasoning  # noqa: E402


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
