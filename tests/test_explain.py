"""`paddock explain`: the report, the verdict, and where it refuses to guess.

The parts that need a GPU node are not tested here -- `_infer_with_trace` is one
HTTP call and `_explain_model` is `runtime.proc` -- but everything between the
document and the printed verdict is, because that is where the command either
earns its three-way split or quietly turns an adapter bug into a model result.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from paddock import conformance as C
from paddock import explain as E
from paddock.benchmark import OpenLoopBenchmark, Sample
from paddock.policy import trace
from paddock.policy.protocol import Trajectory


def _raw(check: str, ok: bool, attribution: str, detail: str = "d", step: str = "infer", **kw):
    return {
        "check": check,
        "ok": ok,
        "severity": kw.get("severity", "error"),
        "detail": detail,
        "rationale": kw.get("rationale", "because"),
        "attribution": attribution,
        "step": step,
    }


def _document(**over) -> dict:
    document = {
        "version": 1,
        "bench": "toybench",
        "model": "toymodel",
        "sample_id": "toybench/scene/7",
        "metrics": {"ade": 1.25, "fde": 2.5},
        "inference": {"steps": [{"name": "infer", "seconds": 0.01}], "findings": [], "assumptions": []},
        "scoring": {"steps": [{"name": "score", "seconds": 0.001}], "findings": []},
        "extra_findings": [],
    }
    document.update(over)
    return document


# --------------------------------------------------------------------------
# Where a dissection lands
# --------------------------------------------------------------------------


def test_a_dissection_is_written_outside_the_repo_under_a_legible_name(tmp_path) -> None:
    """Beside `_viewer/` and `_compare/`: a document with thumbnails in it is
    hundreds of kilobytes, and `results/` is the part that stays diff-able."""

    class _Cfg:
        runs = tmp_path / "runs"

    path = E.explanation_path(_Cfg(), "comma2k19", "comma2k19/3/99c9|2018/16/519", "openpilot")
    assert path.parent.parent == tmp_path / "runs" / "comma2k19" / "_explain"
    assert path.name == "openpilot.json"
    assert "/" not in path.parent.name and "|" not in path.parent.name
    assert "519" in path.parent.name, "the slug should still be greppable for the id it came from"
    assert E._slug("///") == "sample"  # never an empty path component


# --------------------------------------------------------------------------
# Findings, from two interpreters into one vocabulary
# --------------------------------------------------------------------------


def test_both_halves_arrive_as_findings_with_the_step_that_produced_them() -> None:
    """The join that makes the chain reconnectable: the model's interpreter and
    paddock's, under one sample id, in one list."""
    document = _document(
        inference={"findings": [_raw("trajectory.moves_forward", False, "adapter", step="infer")]},
        scoring={"findings": [_raw("align.trimmed_nothing", True, "harness", step="score/align")]},
        extra_findings=[
            {
                "check": "sample.score_of_ground_truth",
                "ok": False,
                "severity": "error",
                "detail": "gives ade = 0.4000",
                "rationale": "r",
                "attribution": "harness",
            }
        ],
    )
    findings = E._findings(document)
    assert [f.check for f in findings] == [
        "trajectory.moves_forward [infer]",
        "sample.score_of_ground_truth",
        "align.trimmed_nothing [score/align]",
    ]
    assert [f.ok for f in findings] == [False, False, True]  # failures first
    assert findings[0].attribution == "adapter"


def test_a_check_that_fired_on_every_history_frame_collapses_to_one_line() -> None:
    """A 160-frame window runs the observation checks 160 times. A wall of
    identical failures buries the one that fired once, which is the interesting
    one -- so the repeats collapse and keep their count."""
    repeated = [
        _raw("observation.mounting_is_level", False, "harness", "120.00 deg off level", f"observe[{i}]")
        for i in range(160)
    ]
    once = _raw("trajectory.moves_forward", False, "adapter", "plan reaches x = 0.01 m", "infer")
    findings = E._deduplicate(E._findings(_document(inference={"findings": [*repeated, once]})))

    checks = {f.check.split(" [")[0]: f for f in findings}
    assert set(checks) == {"observation.mounting_is_level", "trajectory.moves_forward"}
    assert "[x160, e.g. observe[0]]" in checks["observation.mounting_is_level"].check
    assert "[x" not in checks["trajectory.moves_forward"].check


def test_the_same_check_failing_differently_is_not_collapsed() -> None:
    """On every frame and on one frame are different defects, and so are two
    different details reported under one check name."""
    findings = E._deduplicate(
        E._findings(
            _document(
                inference={
                    "findings": [
                        _raw("observation.mounting_is_level", False, "harness", "120 deg", "observe[0]"),
                        _raw("observation.mounting_is_level", False, "harness", "3 deg", "observe[1]"),
                    ]
                }
            )
        )
    )
    assert len(findings) == 2


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_the_verdict_sorts_failures_into_the_three_classes() -> None:
    findings = E._findings(
        _document(
            inference={
                "findings": [
                    _raw("trajectory.moves_forward", False, "adapter", "plan reaches x = 0.01 m"),
                    _raw("resample.is_interpolation", False, "harness", "8 of 20 extrapolate"),
                    _raw("observation.history_is_ordered", True, "harness"),
                    _raw("image.is_not_blank", False, "adapter", "std 0.0", severity="warning"),
                ]
            }
        )
    )
    text = "\n".join(E._verdict(findings))
    assert "adapter  1 failing check(s)" in text
    assert "- trajectory.moves_forward: plan reaches x = 0.01 m" in text
    assert "harness  1 failing check(s)" in text
    assert "model    clean" in text
    assert "1 warning(s), which do not decide the verdict" in text
    assert "no other explanation was found" not in text


def test_a_clean_trace_says_what_it_does_not_prove() -> None:
    """The weakest of the three claims, stated as weak. A trace shows what the
    adapter did; it can only say of the model that nothing else explained the
    number. `docs/reviewing.md`: separate what you verified from what you argue."""
    text = "\n".join(E._verdict(E._findings(_document(inference={"findings": [_raw("x", True, "adapter")]}))))
    assert text.count("clean") == 3
    assert "it says no other explanation was found, not that the model is at fault" in text


def test_every_attribution_a_finding_can_carry_gets_a_line() -> None:
    """The three classes are `trace.ATTRIBUTIONS` and nothing else, so a new one
    added there without a line here would silently never be reported."""
    text = "\n".join(E._verdict([]))
    for attribution in trace.ATTRIBUTIONS:
        assert f"  {attribution:<8} clean" in text


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def test_the_report_leads_with_the_number_and_ends_with_the_verdict() -> None:
    document = _document(
        inference={
            "steps": [{"name": "infer", "steps": [{"name": "plan_to_grid"}]}],
            "findings": [_raw("trajectory.moves_forward", False, "adapter", "plan reaches x = 0.01 m")],
            "assumptions": [
                {
                    "what": "the model emits 33 points on its own T_IDXS out to 10 s",
                    "source": "supercombo docs",
                    "step": "plan_to_grid",
                    "attribution": "adapter",
                }
            ],
        },
        scoring={"steps": [{"name": "score"}], "findings": []},
    )
    text = E._render([document])
    assert "toymodel x toybench/scene/7" in text
    assert "ade=1.250" in text
    assert "assumes [plan_to_grid] the model emits 33 points" in text
    assert "source: supercombo docs" in text
    assert "FAIL" in text and "why:" in text.lower()
    assert "2 inference step(s), 1 scoring step(s)" in text
    assert text.rstrip().endswith("not that the model is at fault") is False
    assert "verdict" in text


def test_what_held_is_counted_rather_than_printed_line_by_line() -> None:
    """A 160-frame comma2k19 window asserts the plan invariants 160 times. Every
    green line printed in full buries the one red line under three hundred --
    so passes are evidence in the form of a count, and only failures get their
    detail and their `why`."""
    document = _document(
        inference={
            "findings": [
                *(
                    _raw("observation.mounting_is_level", True, "harness", f"{i} deg", f"observe[{i}]")
                    for i in range(160)
                ),
                _raw("trajectory.moves_forward", True, "adapter", "x = 88 m"),
                _raw("trajectory.not_a_commanded_stop", False, "adapter", "largest is 0.00 m"),
            ]
        }
    )
    text = E._render([document])
    assert "161 passing assertion(s):" in text
    assert "observation.mounting_is_level x160" in text
    assert "trajectory.moves_forward" in text and "trajectory.moves_forward x" not in text
    assert text.count("[ok  ]") == 1, "the passes are one line between them"
    assert "FAIL" in text and "largest is 0.00 m" in text
    # The tally is not a verdict: a passing check never argues for the model.
    assert "adapter  1 failing check(s)" in text


def test_a_thin_trace_says_it_is_thin() -> None:
    """An un-annotated adapter produces the protocol skeleton and nothing more.
    A trace that looked complete would be worse than one that says it is not."""
    text = E._render([_document(inference={"steps": [], "findings": [], "assumptions": []})])
    assert "(no invariant applied to any recorded step)" in text
    assert "0 inference step(s), 1 scoring step(s)" in text

    over = _document(inference={"steps": [], "findings": [], "truncated": True})
    assert "truncated: over budget" in E._render([over])


def test_a_finding_prints_whose_bug_it_is() -> None:
    """The shared vocabulary: `paddock verify` and `paddock explain` render the
    same `Finding`, and the attribution is the field that turns a number into a
    thing to go and fix."""
    rendered = C.Finding(
        check="trajectory.moves_forward",
        ok=False,
        detail="plan reaches x = 0.01 m",
        rationale="a frame error or a static world",
        attribution="adapter",
    ).render()
    assert "trajectory.moves_forward (adapter)" in rendered
    # `verify`'s own contract findings carry no attribution and must not grow a
    # stray "()" from this field being empty.
    assert "()" not in C.Finding(check="contract.x", ok=True, detail="").render()


# --------------------------------------------------------------------------
# The scoring half
# --------------------------------------------------------------------------


class _ToyBench(OpenLoopBenchmark):
    """A benchmark whose right answer is known by construction."""

    def __init__(self, gt_dt: float = 0.5):
        self.gt_dt = gt_dt

    def samples(self):
        yield from (Sample(sample_id=f"toy/{i}", observations=[], ground_truth=_gt()) for i in range(3))

    def score(self, sample, prediction):
        gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
        xy = np.asarray(prediction.xy, dtype=np.float64)[: len(gt)]
        # Deliberately naive: index-against-index, ignoring dt. That is the
        # defect `sample.score_of_ground_truth` is pointed at.
        return {"ade": float(np.mean(np.linalg.norm(xy - gt[: len(xy)], axis=1)))}

    def preflight(self, cfg):
        return []

    def run_config(self):
        return {}

    def invalid_reasons(self, cfg):
        return []


def _gt(n: int = 6, dt: float = 0.5, speed: float = 20.0) -> np.ndarray:
    t = np.arange(1, n + 1) * dt
    return np.stack([speed * t, np.zeros(n), np.zeros(n)], axis=1)


def test_the_scoring_half_is_traced_in_paddocks_own_interpreter() -> None:
    """The other half of "are the formulae implemented correctly": the metrics
    ride on the same document as the steps that produced them."""
    sample = Sample(sample_id="toy/0", observations=[], ground_truth=_gt())
    metrics, section, _ = E._score_with_trace(_ToyBench(), None, sample, Trajectory(_gt(), dt=0.5))
    assert metrics["ade"] == pytest.approx(0.0)
    assert section["label"] == "score:toy/0"
    assert section["metrics"]["ade"] == pytest.approx(0.0)
    assert "findings" in section


def test_the_ground_truth_is_scored_against_itself_on_the_sample_being_examined() -> None:
    """`conformance` runs this once per benchmark. Running it here, on this
    sample, separates "the metric is broken" from "the metric is fine and the
    model is wrong" at the moment the question is asked."""
    sample = Sample(sample_id="toy/0", observations=[], ground_truth=_gt())
    (finding,) = E._ground_truth_findings(_ToyBench(), None, sample)
    assert finding["ok"] is True
    assert finding["attribution"] == "harness"
    assert "ade = 0.0000, expected ~0" in finding["detail"]


def test_a_metric_that_cannot_score_its_own_ground_truth_is_a_harness_failure() -> None:
    """The grid disagreement that makes every model on a benchmark look bad, and
    that no leaderboard row can distinguish from six bad models."""

    class _OffByOne(_ToyBench):
        def score(self, sample, prediction):
            gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
            xy = np.asarray(prediction.xy, dtype=np.float64)
            # One waypoint of skew: the fde@5s-that-was-4s class of defect.
            return {"ade": float(np.mean(np.linalg.norm(xy[:-1] - gt[1:], axis=1)))}

    sample = Sample(sample_id="toy/0", observations=[], ground_truth=_gt())
    (finding,) = E._ground_truth_findings(_OffByOne(), None, sample)
    assert finding["ok"] is False and "expected ~0" in finding["detail"]


def test_a_metric_that_raises_becomes_the_finding_rather_than_a_traceback() -> None:
    class _Broken(_ToyBench):
        def score(self, sample, prediction):
            raise ZeroDivisionError("no horizon")

    sample = Sample(sample_id="toy/0", observations=[], ground_truth=_gt())
    (finding,) = E._ground_truth_findings(_Broken(), None, sample)
    assert finding["ok"] is False and "ZeroDivisionError: no horizon" in finding["detail"]


def test_a_sample_with_no_recorded_future_is_not_a_failure() -> None:
    """Some benchmarks are externally scored and carry no ground truth. The
    check abstains rather than inventing a verdict about them."""
    sample = Sample(sample_id="toy/0", observations=[], ground_truth=None)
    assert E._ground_truth_findings(_ToyBench(), None, sample) == []


def test_nan_survives_into_json_as_a_word() -> None:
    """A document is written with `allow_nan=False`, so a bare NaN would fail
    the write. "not measured" has to stay visible, not become null."""
    assert E._finite(float("nan")) == "nan"
    assert E._finite(2.5) == 2.5
    assert E._finite("ade") == "ade"
    json.dumps({"ade": E._finite(float("inf"))}, allow_nan=False)


# --------------------------------------------------------------------------
# Reaching a sample
# --------------------------------------------------------------------------


def test_the_default_lookup_is_correct_everywhere_and_fast_nowhere() -> None:
    """`OpenLoopBenchmark.samples_by_id` walks the stream, which is right for
    any benchmark and cheap for none. A benchmark that can seek overrides it."""
    bench = _ToyBench()
    assert [s.sample_id for s in bench.samples_by_id(["toy/2", "toy/0"])] == ["toy/0", "toy/2"]
    assert list(bench.samples_by_id([])) == []
    assert bench.scene_sample_ids("anything") == [], "a benchmark with no scenes names none"


def test_an_unreachable_sample_is_named_rather_than_silently_dropped(capsys) -> None:
    echoed: list[str] = []
    found = E._resolve_samples(_ToyBench(), "toy", ["toy/1", "toy/99"], [], echoed.append)
    assert [s.sample_id for s in found] == ["toy/1"]
    assert any("no such sample: toy/99" in line for line in echoed)


def test_an_empty_scene_says_what_a_scene_key_looks_like() -> None:
    echoed: list[str] = []
    assert E._resolve_samples(_ToyBench(), "toy", [], ["nope"], echoed.append) == []
    assert any("names no samples in scene 'nope'" in line for line in echoed)
    assert any("<chunk>/<route_id>/<segment>" in line for line in echoed)


def test_explain_refuses_an_unknown_benchmark_or_model_before_starting_anything() -> None:
    """Two seconds of registry lookup beats eight minutes of checkpoint read
    followed by a typo."""
    echoed: list[str] = []
    code = E.explain(None, model="all", bench="nosuchbench", sample_ids=["x"], echo=echoed.append)
    assert code == 2 and any("unknown benchmark 'nosuchbench'" in line for line in echoed)

    echoed.clear()
    code = E.explain(None, model="nosuchmodel", bench="neuroncap", sample_ids=["x"], echo=echoed.append)
    assert code == 2 and any("unknown model 'nosuchmodel'" in line for line in echoed)
