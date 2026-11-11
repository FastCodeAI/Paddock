"""navhard's parser: per-frame products, pair dropping, and the check against the devkit's row."""

import json

import pytest

from paddock.adapters.benchmarks.navhard.harness import FRAMES, parse_two_stage


def _frame(token, frame, pair, s1, s2, follow_ups=3):
    combined = None if s1 is None or s2 is None else s1 * s2
    return {
        "token": token, "frame": frame, "pair": pair, "follow_ups": follow_ups, "follow_ups_scored": follow_ups,
        "stage_one": {"score": s1, "ego_progress": 0.5}, "stage_two": {"score": s2}, "combined": {"score": combined},
    }


def _write(tmp_path, frames, combined_row, failed=()):
    rows = ["token,valid,score,ego_progress_stage_one"]
    rows += [f"{t},False,," for t in failed]
    rows += [f"extended_pdm_score_combined,True,{combined_row},"]
    csv_path = tmp_path / "2026.11.10.05.00.00.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    (tmp_path / FRAMES).write_text("\n".join(json.dumps(f) for f in frames) + "\n")
    return csv_path


def test_mean_of_frames_is_the_devkits_combined(tmp_path):
    frames = [_frame("a", "now", ["a", "b"], 0.9, 0.5), _frame("b", "previous", ["a", "b"], 0.8, 1.0),
              _frame("c", "now", ["c", "d"], 1.0, 0.6), _frame("d", "previous", ["c", "d"], 0.5, 0.4)]
    official = ((0.45 + 0.8) / 2 + (0.6 + 0.2) / 2) / 2
    samples, aggregates, failed = parse_two_stage(_write(tmp_path, frames, official), tmp_path / FRAMES)
    assert [s["metrics"]["epdms"] for s in samples] == pytest.approx([0.45, 0.8, 0.6, 0.2])
    assert aggregates["paddock_check"]["mean_epdms"] == pytest.approx(official)
    assert samples[0]["metrics"]["ego_progress_stage_one"] == 0.5 and failed == []


def test_a_pair_with_an_unscored_frame_drops_out_whole(tmp_path):
    frames = [_frame("a", "now", ["a", "b"], 0.9, 0.5), _frame("b", "previous", ["a", "b"], 0.8, None),
              _frame("c", "now", ["c", "d"], 1.0, 0.6), _frame("d", "previous", ["c", "d"], 0.5, 0.4)]
    samples, aggregates, failed = parse_two_stage(_write(tmp_path, frames, 0.4, failed=["x1"]), tmp_path / FRAMES)
    assert ["epdms" in s["metrics"] for s in samples] == [False, False, True, True]
    assert samples[0]["metrics"]["epdms_stage_one"] == 0.9  # stage scores stay
    assert failed == ["x1"]


def test_a_mismatch_with_the_devkit_is_refused(tmp_path):
    frames = [_frame("a", "now", ["a", "b"], 1.0, 1.0), _frame("b", "previous", ["a", "b"], 1.0, 1.0)]
    with pytest.raises(RuntimeError, match="no longer matches"):
        parse_two_stage(_write(tmp_path, frames, 0.5), tmp_path / FRAMES)
