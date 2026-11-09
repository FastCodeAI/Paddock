"""Scene order (`[benchmarks.nuscenes] sequence = "scene"`): the same keyframes, each scene fed in order.

UniAD's and VAD's own test loops run every val scene once, keyframe by keyframe, carrying the previous BEV
grid (and UniAD its tracks) through the scene. Scene order feeds paddock's nodes the same way: one episode
per scene, a plan from every keyframe, each sample scored from the plan made at its own keyframe. What is
pinned here: the scored set and its ground truth are the window's, keyframe for keyframe; every scene is fed
from its first keyframe whatever is skipped, limited or filtered out; each keyframe carries its own command;
the run says which order it was made in; and a model that was not built for it is refused before it loads.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.benchmarks.nuscenes import harness as NH
from paddock.adapters.benchmarks.nuscenes.loader import Keyframe


def _scene(name: str, n: int, turn_at: int | None = None) -> list[Keyframe]:
    """n keyframes 0.5 s apart at 5 m/s, turning left 90 degrees over two seconds from `turn_at` on."""
    frames, x, y, yaw = [], 0.0, 0.0, 0.0
    for i in range(n):
        pose = np.eye(4)
        pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        pose[0, 3], pose[1, 3] = x, y
        frames.append(Keyframe(f"{name}-{i}", name, i, 1_000_000 * len(name) + i * 500_000, pose))
        if turn_at is not None and turn_at <= i < turn_at + 4:
            yaw += math.pi / 8
        x, y = x + 2.5 * math.cos(yaw), y + 2.5 * math.sin(yaw)
    return frames


@pytest.fixture
def scenes():
    return [_scene("scene-0003", 14, turn_at=9), _scene("scene-0012", 9)]


def _harness(monkeypatch, scenes, **settings):
    harness = object.__new__(NH.NuScenesBenchmark)
    monkeypatch.setattr(harness, "setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setattr(harness, "_index", lambda: scenes)
    harness._boxes = {}
    harness._camera_history_cache = None
    return harness


def test_scene_order_scores_the_windows_keyframes_from_one_pass_per_scene(monkeypatch, scenes):
    window = _harness(monkeypatch, scenes)
    by_window = list(window.samples())
    harness = _harness(monkeypatch, scenes, sequence="scene")
    assert harness.in_sequence() and not window.in_sequence()
    sequences = list(harness.sequences())

    assert [s.sequence_id for s in sequences] == ["nuscenes/val/scene-0003", "nuscenes/val/scene-0012"]
    scored = [sample for sequence in sequences for _, sample in sequence.scored]
    assert [s.sample_id for s in scored] == [s.sample_id for s in by_window], "the same keyframes, in order"
    for mine, theirs in zip(scored, by_window, strict=True):
        np.testing.assert_array_equal(mine.ground_truth, theirs.ground_truth)
        assert mine.native_id == theirs.native_id and mine.group == theirs.group
        # The scored frame is the window's last frame, command and all.
        assert mine.observations[-1].timestamp_us == theirs.observations[-1].timestamp_us
        assert mine.observations[-1].command == theirs.observations[-1].command

    for sequence, frames in zip(sequences, scenes, strict=True):
        last = sequence.scored[-1][0]
        assert [o.timestamp_us for o in sequence.observations] == [
            f.timestamp_us for f in frames[: last + 1]
        ], "every keyframe from the scene's first to the last one scored, and nothing after"
        for index, sample in sequence.scored:
            assert sample.observations[-1] is sequence.observations[index], (
                "scored from the plan at its frame"
            )
            assert sample.observations == sequence.observations[: index + 1], (
                "its history is the scene so far"
            )
        assert last == len(frames) - 1 - NH.N_STEPS, "the last six keyframes have no full 3 s future"


def test_each_keyframe_carries_its_own_command_not_the_scored_ones(monkeypatch, scenes):
    """In a window, the history frames carry the scored keyframe's command; in a scene each keyframe is
    the scored one at its own turn, so each carries its own, as UniAD's dataset gives every keyframe."""
    harness = _harness(monkeypatch, scenes, sequence="scene")
    sequence = next(iter(harness.sequences()))
    commands = [int(o.command) for o in sequence.observations]
    assert len(set(commands)) > 1, "the turn has to show up in the commands for this to test anything"
    for index, sample in sequence.scored:
        assert commands[index] == sample.native_id["command"]


def test_a_resumed_or_limited_scene_is_still_fed_from_its_first_keyframe(monkeypatch, scenes):
    harness = _harness(monkeypatch, scenes, sequence="scene")
    every = [s.sample_id for seq in harness.sequences() for _, s in seq.scored]

    # Resume: the first scene's first five and last one already scored.
    done = set(every[:5]) | {every[7]}
    first, second = list(harness.sequences(skip=done))
    assert [s.sample_id for _, s in first.scored] == [every[5], every[6]]
    assert first.observations[0].timestamp_us == scenes[0][0].timestamp_us, (
        "the state is rebuilt from the start"
    )
    assert len(first.observations) == 7, "and fed only as far as the last keyframe still to score"
    assert [s.sample_id for _, s in second.scored] == every[8:]

    # A finished scene is not fed at all.
    assert [seq.sequence_id for seq in harness.sequences(skip=set(every[:8]))] == ["nuscenes/val/scene-0012"]

    # --limit counts samples across scenes and stops mid-scene, as `samples()` does.
    limited = list(harness.sequences(limit=10))
    assert [len(seq.scored) for seq in limited] == [8, 2]
    assert len(limited[1].observations) == 2
    assert [s.sample_id for seq in limited for _, s in seq.scored] == every[:10]
    # On a resume `limit` caps what is still to do, not the run's total.
    assert [
        s.sample_id for seq in harness.sequences(limit=3, skip=set(every[:6])) for _, s in seq.scored
    ] == (every[6:9])


def test_a_sample_list_in_scene_order_still_feeds_each_scene_from_its_start(monkeypatch, scenes):
    monkeypatch.setenv(NH.ENV_SAMPLES, "nuscenes/val/scene-0003/4,nuscenes/val/scene-0012/1")
    harness = _harness(monkeypatch, scenes, sequence="scene")
    first, second = list(harness.sequences())
    assert [(i, s.sample_id) for i, s in first.scored] == [(4, "nuscenes/val/scene-0003/4")]
    assert len(first.observations) == 5 and len(second.observations) == 2


def test_the_run_says_it_was_made_in_scene_order_and_a_window_run_hashes_as_before(monkeypatch, scenes):
    window = _harness(monkeypatch, scenes)
    # Exactly the config every window run so far was stored under, so their ids and resumes still match.
    assert window.run_config() == {"camera_history": "keyframes", "history_seconds": 2.0, "scoring_hz": 2.0}
    harness = _harness(monkeypatch, scenes, sequence="scene")
    assert harness.run_config() == {"camera_history": "keyframes", "sequence": "scene", "scoring_hz": 2.0}
    # No window, so no window length per model either (AutoE2E would otherwise record 6.5 s).
    assert harness.model_run_config(SimpleNamespace(history_s=6.5)) == {}


@pytest.mark.parametrize("name", ["uniad", "vad", "uniad_b2d", "vad_b2d"])
def test_scene_order_is_offered_to_the_models_whose_own_evaluation_ran_that_way(monkeypatch, scenes, name):
    harness = _harness(monkeypatch, scenes, sequence="scene")
    assert harness.input_problems(SimpleNamespace(name=name, min_input_hz=2.0)) == []


@pytest.mark.parametrize("name", ["qwendrive", "drivejepa", "alpamayo15", "visionpilot", "constant_velocity"])
def test_scene_order_is_refused_for_a_model_it_would_feed_a_history_it_was_not_built_for(
    monkeypatch, scenes, name
):
    """QwenDrive among them: one model call per sample anyway, and its dense pose history is window-based."""
    harness = _harness(monkeypatch, scenes, sequence="scene")
    (problem,) = harness.input_problems(SimpleNamespace(name=name, min_input_hz=2.0))
    assert "scene" in problem and name in problem
    window = _harness(monkeypatch, scenes)
    assert window.input_problems(SimpleNamespace(name=name, min_input_hz=2.0)) == []


def test_a_site_can_name_the_models_scene_order_is_for(monkeypatch, scenes):
    harness = _harness(monkeypatch, scenes, sequence="scene", scene_models="uniad, drivejepa")
    assert harness.input_problems(SimpleNamespace(name="drivejepa", min_input_hz=2.0)) == []
    assert harness.input_problems(SimpleNamespace(name="vad", min_input_hz=2.0))


def test_preflight_names_a_sequence_it_does_not_know_and_one_it_cannot_combine(tmp_path, monkeypatch):
    from paddock.config import SiteConfig
    from paddock.registry import Registry

    for sub in ("repos", "runs", "weights", "nuscenes"):
        (tmp_path / sub).mkdir()

    def preflight(block: str) -> list[str]:
        (tmp_path / "site.toml").write_text(
            f'site = "t"\n[roots]\nrepos = "{tmp_path / "repos"}"\nruns = "{tmp_path / "runs"}"\n'
            f'weights = "{tmp_path / "weights"}"\n[datasets.nuscenes]\npath = "{tmp_path / "nuscenes"}"\n'
            f"[benchmarks.nuscenes]\n{block}\n"
        )
        monkeypatch.setenv("PADDOCK_CONFIG", str(tmp_path / "site.toml"))
        return NH.BENCHMARK(Registry.load().benchmarks["nuscenes"], SiteConfig.load()).preflight()

    assert not [p for p in preflight('sequence = "scene"') if "sequence" in p]
    assert [p for p in preflight('sequence = "scenes"') if "sequence = 'scenes'" in p]
    assert [
        p for p in preflight('sequence = "scene"\ncamera_history = "CAM_FRONT"') if "cannot be combined" in p
    ]
