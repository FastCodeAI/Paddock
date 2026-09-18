"""The observability layer: what it records, what it costs, and what it refuses.

Two claims here are load-bearing and everything else supports them.

**Off, it is free.** `test_an_untraced_call_costs_a_context_lookup` measures the
per-call overhead of a decorated function with no recorder active. If that ever
stops being nanoseconds, annotating a hot adapter seam stops being safe and the
whole decorator premise goes with it.

**On, it catches the defects that were published as modelling results.** The
`camera2ego = eye(4)` reproduction is here at unit scale: the identity mounting
is refused *and* named as a convention error, which is the one screen that would
have replaced the 38.8 m ADE table.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import numpy as np
import pytest

from paddock import conformance as C
from paddock.policy import trace
from paddock.policy.protocol import (
    CANBUS_YAW_RATE,
    INFER_CHECKS,
    SEQUENCE_CHECKS,
    Command,
    Policy,
    Trajectory,
    constant_velocity,
)


def _observation(speed: float = 20.0, yaw_rate: float = 0.0, t_us: int = 0, level: bool = True):
    """A small observation: 16x12 frames, so a test is not 3 MB of mid-grey."""
    obs = C.synthetic_observation(speed=speed, width=16, height=12, timestamp_us=t_us)
    obs.canbus[CANBUS_YAW_RATE] = yaw_rate
    if not level:
        # The defect itself: "no rotation" written as the identity, which in
        # paddock's frames is 90 degrees of error.
        obs.camera2ego["CAM_FRONT"] = np.eye(4)
    return obs


def _plan(n: int = 6, dt: float = 0.5, speed: float = 20.0, lateral: float = 0.0) -> Trajectory:
    t = np.arange(1, n + 1) * dt
    xy = np.stack([speed * t, lateral * t / t[-1], np.zeros(n)], axis=1)
    return Trajectory(waypoints=xy, dt=dt)


def _arc(omega: float, n: int = 6, dt: float = 0.5, speed: float = 20.0) -> Trajectory:
    """A plan driven at a constant yaw rate. Positive omega bends left, y > 0."""
    t = np.arange(1, n + 1) * dt
    r = speed / omega
    xy = np.stack([r * np.sin(omega * t), r * (1 - np.cos(omega * t)), np.zeros(n)], axis=1)
    return Trajectory(waypoints=xy, dt=dt)


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def test_an_untraced_call_costs_a_context_lookup() -> None:
    """The premise of the decorator form, measured rather than asserted in prose.

    A decorated call with no recorder active is one `ContextVar.get()` and a
    direct call -- no signature binding, no summarising, no check. The bound
    below is loose enough to survive a loaded CI box and tight enough that
    anything doing real work per call (binding the signature at import, building
    a Step unconditionally) blows straight through it.
    """

    def bare(a, b):
        return a + b

    decorated = trace.observed("add")(bare)
    assert decorated(2, 3) == 5

    def elapsed(fn, n=200_000):
        best = float("inf")
        for _ in range(3):  # best-of-three: a scheduler blip inflates a mean
            start = time.perf_counter()
            for _ in range(n):
                fn(2, 3)
            best = min(best, time.perf_counter() - start)
        return best / n

    overhead = elapsed(decorated) - elapsed(bare)
    print(f"\n  untraced overhead: {overhead * 1e9:.0f} ns/call")
    assert overhead < 5e-6, f"{overhead * 1e6:.2f} us/call is not a context lookup"


def test_nothing_is_retained_when_no_recorder_is_listening() -> None:
    """Not just fast: the wrapper must not even look at what it was handed.

    A summary built and thrown away would keep a frame alive for the length of
    the call, which on a 24 MB KITScenes image is the difference between a sweep
    that fits in memory and one that does not.
    """
    seen: list[str] = []

    class Loud:
        def __array__(self, *a, **k):
            seen.append("array")
            raise AssertionError("summarised while not recording")

        def __repr__(self):
            seen.append("repr")
            return "<Loud>"

    @trace.observed("touch")
    def touch(value):
        return 1

    assert touch(Loud()) == 1
    assert seen == []


# --------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------


def test_recording_nests_steps_and_times_them() -> None:
    @trace.observed("inner", why="the leaf")
    def inner(x):
        return x * 2

    @trace.observed("outer", why="the root")
    def outer(x):
        trace.note(branch="fast")
        return inner(x) + inner(x)

    with trace.recording("t") as rec:
        assert outer(3) == 12

    doc = rec.document()
    assert [s["name"] for s in doc["steps"]] == ["outer"]
    outer_step = doc["steps"][0]
    assert trace.why_of(doc, outer_step) == "the root"
    assert outer_step["notes"] == {"branch": "fast"}
    assert [s["name"] for s in outer_step["steps"]] == ["inner", "inner"]
    assert outer_step["inputs"] == {"x": 3}
    assert outer_step["output"] == 12
    assert outer_step["seconds"] >= 0.0


def test_a_raising_step_records_the_reason_and_re_raises() -> None:
    """A trace observes; it never changes what happens. The exception still
    propagates, and the step that failed says why instead of vanishing."""

    @trace.observed("boom")
    def boom():
        raise RuntimeError("no weights")

    with trace.recording("t") as rec, pytest.raises(RuntimeError, match="no weights"):
        boom()

    step = rec.document()["steps"][0]
    assert step["error"] == "RuntimeError: no weights"
    assert "output" not in step


def test_recursion_stops_being_recorded_but_keeps_running() -> None:
    """`MAX_DEPTH` is a guard against an accidentally-annotated recursive
    helper filling a trace with itself. The call still returns correctly."""

    @trace.observed("down")
    def down(n):
        return 0 if n == 0 else 1 + down(n - 1)

    with trace.recording("t") as rec:
        assert down(trace.MAX_DEPTH + 5) == trace.MAX_DEPTH + 5

    depth, step = 0, rec.document()["steps"][0]
    while step.get("steps"):
        depth, step = depth + 1, step["steps"][0]
    assert depth < trace.MAX_DEPTH + 5


def test_note_outside_a_recording_is_a_no_op() -> None:
    trace.note(anything="at all")  # must not raise


# --------------------------------------------------------------------------
# Summaries and the budget
# --------------------------------------------------------------------------


def test_a_big_array_becomes_statistics_and_a_small_one_survives() -> None:
    big = np.arange(trace.MAX_ELEMENTS * 4, dtype=np.float32).reshape(-1, 4)
    small = np.array([[1.0, 2.0], [3.0, 4.0]])

    summary = trace.summarise(big)
    assert summary["shape"] == list(big.shape) and summary["dtype"] == "float32"
    assert "values" not in summary
    assert summary["min"] == 0.0 and summary["max"] == float(big.max())

    assert trace.summarise(small)["values"] == [[1.0, 2.0], [3.0, 4.0]]


def test_capture_full_keeps_the_numbers_themselves() -> None:
    """`--capture full` exists for a suspected sub-pixel defect, where the array
    *is* the evidence. It raises the budget rather than writing a side file, so
    the numbers still ride back on the one response."""
    big = np.zeros((trace.MAX_ELEMENTS * 2,), dtype=np.float64)

    @trace.observed("warp")
    def warp(m):
        return m

    with trace.recording("t", capture="full") as rec:
        warp(big)
    assert len(rec.document()["steps"][0]["inputs"]["m"]["values"]) == big.size

    with trace.recording("t") as rec:
        warp(big)
    assert "values" not in rec.document()["steps"][0]["inputs"]["m"]


def test_a_non_finite_value_is_counted_not_hidden() -> None:
    values = np.array([1.0, np.nan, np.inf, 4.0])
    summary = trace.summarise(values)
    assert summary["non_finite"] == 2
    assert summary["min"] == 1.0 and summary["max"] == 4.0  # statistics over the finite part
    assert trace.summarise(float("nan")) == "nan"  # JSON has no NaN; the string says so


def test_an_image_becomes_a_thumbnail_and_a_plan_becomes_its_shape() -> None:
    pytest.importorskip("PIL")
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    summary = trace.summarise(image, trace.Recorder("t"))
    assert summary["kind"] == "image" and summary["png_b64"]
    assert "values" not in summary  # a frame is never inlined as numbers

    plan = trace.summarise(_plan(n=4, dt=0.5, speed=10.0))
    assert plan["kind"] == "trajectory" and plan["n"] == 4 and plan["dt"] == 0.5
    assert plan["first"] == [5.0, 0.0] and plan["last"] == [20.0, 0.0]
    assert plan["path_length_m"] == pytest.approx(20.0)


def test_an_observation_summary_names_the_mounting_angle() -> None:
    """The one number that would have caught the 38.8 m result, on every
    observation in every trace, without anybody asking for it."""
    level = trace.summarise(_observation())
    assert level["cameras"] == ["CAM_FRONT"] and level["speed_mps"] == 20.0
    assert level["mounting_off_level_deg"]["CAM_FRONT"] == pytest.approx(0.0, abs=1e-6)

    slipped = trace.summarise(_observation(level=False))
    assert slipped["mounting_off_level_deg"]["CAM_FRONT"] == pytest.approx(120.0, abs=1e-6)


def test_the_budget_truncates_captures_and_keeps_the_report() -> None:
    """Past the budget the arrays stop; names, timings and findings do not.
    A truncated trace still says which step failed what -- which is the part a
    report is built from."""

    @trace.observed("step", checks=("trajectory.not_a_commanded_stop",))
    def step(m):
        return Trajectory(waypoints=np.zeros((4, 3)), dt=0.5)

    with trace.recording("t", budget_bytes=64) as rec:
        for _ in range(3):
            step(np.arange(100, dtype=np.float64))

    doc = rec.document()
    assert doc["truncated"] is True
    assert len(doc["steps"]) == 3
    assert all("values" not in s["inputs"]["m"] for s in doc["steps"][1:])
    assert [f["check"] for f in doc["findings"]] == ["trajectory.not_a_commanded_stop"] * 3


def test_the_budget_counts_the_bytes_the_document_actually_costs() -> None:
    """A budget that undercounts is not a budget.

    The estimate was once a flat 12 bytes per number, which put a 160-frame
    comma2k19 trace at 77 KB against a real megabyte -- so `BUDGET_BYTES` never
    engaged and the cap existed on paper only. `tolist()` hands back Python
    floats that print to full round-trip precision, and float32 costs the same as
    float64 because widening to double is what prints. So the estimate is
    per-dtype and pinned here against `json.dumps` of the real thing.
    """

    @trace.observed("step", capture=("m", "return"))
    def step(m):
        return m * 2.0

    rng = np.random.default_rng(0)
    with trace.recording("t", budget_bytes=10_000_000) as rec:
        for _ in range(8):
            step(np.cumsum(rng.normal(12.0, 3.0, (20, 2)), axis=0))

    doc = rec.document()
    actual = len(json.dumps(doc))
    # Within a factor of two of the truth, and never *under* it: an estimate that
    # runs low is the failure mode, because it is the one that lets a trace
    # through the cap.
    assert doc["approx_bytes"] >= actual / 2, (doc["approx_bytes"], actual)
    assert doc["approx_bytes"] <= actual * 2, (doc["approx_bytes"], actual)


def test_an_integer_array_is_not_charged_as_though_it_were_float() -> None:
    """Indices and shapes are cheap; charging them 20 bytes each would evict the
    float arrays that carry the evidence."""

    @trace.observed("step", capture=("m",))
    def step(m):
        return None

    with trace.recording("t") as ints:
        step(np.arange(400, dtype=np.int32))
    with trace.recording("t") as floats:
        step(np.arange(400, dtype=np.float64))

    assert ints.spent < floats.spent / 2
    assert "values" in ints.document()["steps"][0]["inputs"]["m"]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _fire(name: str, inputs: dict, output=None):
    """Run one registered check against a hand-built step. `None` means it abstained."""
    step = trace.Step(name="t", inputs=inputs, output=output)
    return trace.CHECKS[name].test(step)


def test_every_registered_check_declares_an_attribution_and_a_reason() -> None:
    """The attribution is the output the exercise exists to produce, so a check
    without one is a number without a thing to go and fix."""
    assert trace.CHECKS, "checks.py is imported from trace's foot; an empty registry means it is not"
    for name, spec in trace.CHECKS.items():
        assert spec.attribution in trace.ATTRIBUTIONS, name
        assert len(spec.why) > 40, f"{name}: `why` should name the defect, not restate the rule"
        assert spec.severity in ("error", "warning"), name


def test_the_identity_mounting_is_refused_and_named() -> None:
    """Verification item 6, at unit scale: the `camera2ego = eye(4)` defect that
    published ADE 38.8 m as a modelling result."""
    ok, detail = _fire("observation.mounting_is_level", {"obs": _observation(level=False)})
    assert not ok
    assert "120.00 deg off level" in detail
    assert "the matrix is the identity, so this is the convention error" in detail

    ok, _ = _fire("observation.mounting_is_level", {"obs": _observation()})
    assert ok


def test_an_all_zero_plan_reads_as_a_commanded_stop() -> None:
    zeros = Trajectory(waypoints=np.zeros((6, 3)), dt=0.5)
    ok, detail = _fire("trajectory.not_a_commanded_stop", {}, zeros)
    assert not ok and "0.00 m" in detail
    assert _fire("trajectory.not_a_commanded_stop", {}, _plan())[0]


def test_spacing_catches_a_plan_emitted_on_the_wrong_grid() -> None:
    """The fde@5s-that-was-4s class: both sides of a comparison move together,
    so ADE stays plausible and only the grid gives it away."""
    obs = _observation(speed=20.0)
    ok, _ = _fire("trajectory.spacing_matches_speed", {"obs": obs}, _plan(dt=0.5, speed=20.0))
    assert ok

    # The model's own 0.1 s grid, labelled as paddock's 0.5 s one.
    native = Trajectory(waypoints=_plan(dt=0.1, speed=20.0).waypoints, dt=0.5)
    ok, detail = _fire("trajectory.spacing_matches_speed", {"obs": obs}, native)
    assert not ok and "ratio 0.20" in detail


def test_the_turn_check_reads_the_sign_and_abstains_when_it_cannot() -> None:
    """A y-sign flip is invisible to ADE, which is a magnitude. This is the
    check for it -- and it says nothing at all when the car is going straight,
    rather than reporting green."""
    name = "trajectory.turn_direction_matches_yaw_rate"
    left = _observation(yaw_rate=0.1)
    assert _fire(name, {"o": left}, _arc(0.1))[0]
    ok, detail = _fire(name, {"o": left}, _arc(-0.1))
    assert not ok and "plan turns -0.100 rad/s (right)" in detail

    straight = _observation(yaw_rate=0.0)
    assert _fire(name, {"o": straight}, _arc(0.1)) is None
    # And the mirror: the car is turning but the plan is not, so the sign of its
    # residual bend is noise. Abstaining beats accusing a correct adapter.
    assert _fire(name, {"o": left}, _plan(lateral=5.0)) is None


def test_the_turn_check_does_not_accuse_a_correct_plan_through_an_s_bend() -> None:
    """Regression, found by running this check against the recorded expert.

    Comparing the measured yaw rate to the plan's *endpoint* failed the human
    driver on 16 of the 240 viewer10 keyframes it fired on -- every one of them a
    curvature reversal, where a car finishing a left bend correctly plans a
    future that ends far to the right. That produced a confident `adapter`
    verdict against openpilot on a segment where openpilot was right. The
    comparison is rate against rate over the plan's first seconds for exactly
    this reason.
    """
    name = "trajectory.turn_direction_matches_yaw_rate"
    n, dt, speed = 20, 0.5, 20.0
    t = np.arange(1, n + 1) * dt
    # Turning left now, into a road that swings right: heading starts positive,
    # crosses zero, and the plan ends 20 m to the right of where it started.
    heading = 0.12 * t - 0.03 * t**2
    xy = np.stack(
        [
            np.cumsum(speed * dt * np.cos(heading)),
            np.cumsum(speed * dt * np.sin(heading)),
            np.zeros(n),
        ],
        axis=1,
    )
    plan = Trajectory(waypoints=xy, dt=dt)
    assert xy[-1, 1] < -15.0, "the fixture must actually end on the right"

    ok, detail = _fire(name, {"o": _observation(yaw_rate=0.08)}, plan)
    assert ok, detail
    assert "(left)" in detail


def test_a_check_abstains_rather_than_passing_when_it_cannot_see_its_subject() -> None:
    """Abstaining is not a pass. A check reporting green on a step that does not
    carry its subject is how a check quietly stops checking."""
    for name in ("trajectory.not_a_commanded_stop", "trajectory.moves_forward"):
        assert _fire(name, {"x": 1}, output="not a plan") is None
    assert _fire("observation.mounting_is_level", {"x": 1}) is None
    assert _fire("resample.is_interpolation_not_extrapolation", {}) is None


def test_extrapolation_past_the_native_grid_is_the_harness_asking_wrongly() -> None:
    """The check that separates 'bad at 3 s' from 'never trained to emit 3 s'."""
    step = trace.Step(name="grid", notes={"native_t": {"values": [0.5, 1.0, 1.5]}})
    step.notes["requested_t"] = {"values": [0.5, 1.0, 1.5]}
    assert trace.CHECKS["resample.is_interpolation_not_extrapolation"].test(step)[0]

    step.notes["requested_t"] = {"values": [0.5, 1.0, 1.5, 2.0, 3.0]}
    ok, detail = trace.CHECKS["resample.is_interpolation_not_extrapolation"].test(step)
    assert not ok and "2 of 5 points extrapolate" in detail
    assert trace.CHECKS["resample.is_interpolation_not_extrapolation"].attribution == "harness"


def test_a_check_that_raises_becomes_a_finding_not_a_crash() -> None:
    """A trace must never be the thing that fails the run it is observing."""
    trace.check("test.explodes", "A deliberately broken predicate, for this test only.", "adapter")(
        lambda step: 1 / 0
    )
    try:

        @trace.observed("s", checks=("test.explodes", "test.unregistered"))
        def s():
            return 1

        with trace.recording("t") as rec:
            assert s() == 1
        findings = {f["check"]: f for f in rec.document()["findings"]}
        assert "ZeroDivisionError" in findings["test.explodes"]["detail"]
        assert "not registered" in findings["test.unregistered"]["detail"]
        assert not any(f["ok"] for f in findings.values())
    finally:
        trace.CHECKS.pop("test.explodes", None)


def test_findings_are_flattened_with_their_path_and_failures_come_first() -> None:
    @trace.observed("leaf", checks=("trajectory.not_a_commanded_stop",))
    def leaf(zeros):
        return Trajectory(waypoints=np.zeros((4, 3)) if zeros else _plan().waypoints, dt=0.5)

    @trace.observed("root")
    def root():
        return leaf(False), leaf(True)

    with trace.recording("t") as rec:
        root()

    document = rec.document()
    findings = document["findings"]
    assert [f["ok"] for f in findings] == [False, True]
    assert all(f["step"] == "root/leaf" for f in findings)
    assert findings[0]["attribution"] == "adapter"
    # The `why` rides in the document once, not on each of the two findings.
    assert "rationale" not in findings[0]
    name = "trajectory.not_a_commanded_stop"
    assert document["rationales"] == {name: trace.CHECKS[name].why}


def test_a_finding_names_the_one_step_that_produced_it() -> None:
    """The two `leaf` calls share a path, so only `uid` tells them apart.

    That matters because the tree no longer repeats the findings inside it --
    storing each one twice was 115 KB of a comma2k19 document -- so `uid` is the
    whole join between a finding and the step it came from.
    """

    @trace.observed("leaf", checks=("trajectory.not_a_commanded_stop",))
    def leaf(zeros):
        return Trajectory(waypoints=np.zeros((4, 3)) if zeros else _plan().waypoints, dt=0.5)

    with trace.recording("t") as rec:
        leaf(False)
        leaf(True)

    document = rec.document()
    steps = {s["uid"]: s for s in document["steps"]}
    assert len(steps) == 2, "two calls, two distinct uids"
    assert all("findings" not in s for s in steps.values()), "findings live in one place only"

    by_uid = {f["uid"]: f for f in document["findings"]}
    assert set(by_uid) == set(steps)
    failed = next(f for f in document["findings"] if not f["ok"])
    assert steps[failed["uid"]]["output"]["path_length_m"] == 0.0


# --------------------------------------------------------------------------
# Protocol steps, free for every model
# --------------------------------------------------------------------------


class _Toy(Policy):
    """A model with zero annotations: everything it traces comes from the protocol."""

    cameras = ("CAM_FRONT",)

    def infer(self, obs):
        return constant_velocity(obs.speed, dt=self.waypoint_dt)


def test_an_unannotated_policy_still_produces_a_useful_trace() -> None:
    """Phase 1's whole claim. No adapter changes: the skeleton, the timings, the
    observation summaries and the protocol invariants arrive anyway."""
    obs = [_observation(t_us=i * 500_000) for i in range(3)]
    with trace.recording("toy.infer_sequence") as rec:
        plan = _Toy().infer_sequence(obs)
    assert isinstance(plan, Trajectory)

    doc = rec.document()
    assert [s["name"] for s in doc["steps"]] == ["infer_sequence"]
    sequence = doc["steps"][0]
    # reset, then a full forward pass per history frame (the default `observe`),
    # then the real infer -- the cost model an adapter overrides `observe` to escape.
    assert [s["name"] for s in sequence["steps"]] == ["reset", "observe", "observe", "infer"]
    assert sequence["steps"][1]["steps"][0]["name"] == "infer"
    assert trace.why_of(doc, sequence), "the protocol step carries the assumption it encodes"

    # Every check that can see its subject here, and only those: the plan checks
    # on `infer`, the history check on `infer_sequence` (the one step that sees
    # the whole window), and no turn-direction finding at all, because the car is
    # going straight and a plan with no lateral excursion carries no sign to check.
    fired = {f["check"] for f in doc["findings"]}
    expected = set(INFER_CHECKS) | set(SEQUENCE_CHECKS)
    assert fired == expected - {"trajectory.turn_direction_matches_yaw_rate"}
    assert all(f["ok"] for f in doc["findings"])


def test_the_identity_mounting_surfaces_through_a_real_policy() -> None:
    """The same defect as the unit check above, reached the way it was actually
    reached: a policy inferring on an observation nobody looked at twice."""
    with trace.recording("toy") as rec:
        _Toy().infer(_observation(level=False))
    failed = [f for f in rec.document()["findings"] if not f["ok"]]
    assert [f["check"] for f in failed] == ["observation.mounting_is_level"]
    assert failed[0]["attribution"] == "harness"


def test_a_hand_annotation_wins_over_the_automatic_wrapping() -> None:
    """An adapter that has something to say about its own `infer` -- a check the
    protocol cannot know about -- keeps it, and is not wrapped twice."""

    class _Annotated(_Toy):
        @trace.observed("infer", why="mine", checks=(*INFER_CHECKS, "image.is_not_blank"))
        def infer(self, obs):
            return constant_velocity(obs.speed, dt=self.waypoint_dt)

    with trace.recording("t") as rec:
        _Annotated().infer(_observation())
    doc = rec.document()
    step = doc["steps"][0]
    assert step["name"] == "infer" and trace.why_of(doc, step) == "mine"
    assert not step.get("steps"), "wrapped twice: the hand annotation should have been left alone"


def test_an_abstract_policy_method_is_left_abstract() -> None:
    """`Policy.infer` is abstract; wrapping it would make it concrete and a
    model that forgot to implement it would return None instead of failing."""
    with pytest.raises(TypeError, match="abstract"):
        Policy()  # type: ignore[abstract]
    assert getattr(Policy.infer, "__isabstractmethod__", False)


def test_the_unrotated_frame_is_the_one_conformance_uses() -> None:
    """`trace` imports nothing from paddock, so it restates this matrix rather
    than importing it -- and a restated constant is one that can drift. This is
    the pin that stops it: one convention, checked in two places, never two."""
    assert np.array_equal(trace.UNROTATED_CAMERA2EGO, C.UNROTATED_CAMERA2EGO)

    rng = np.random.default_rng(7)
    for _ in range(8):
        m = np.eye(4)
        m[:3, :3] = C.UNROTATED_CAMERA2EGO @ _small_rotation(rng)
        mine = trace.mounting_off_level_deg(m)
        theirs = C.check_camera_mounting(m)
        assert (mine <= C.MAX_MOUNT_DEG) == theirs.ok
        assert f"{mine:.2f} deg off level" in theirs.detail


def _small_rotation(rng) -> np.ndarray:
    """A rotation of a few degrees about a random axis -- a real bracket."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(0.0, 25.0))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def test_the_protocol_wrapping_does_not_change_what_a_policy_returns() -> None:
    """The safety claim for the whole change: every decorated function returns
    exactly what it returned before, traced or not."""
    obs = [_observation(speed=13.0, t_us=i * 500_000) for i in range(3)]
    untraced = _Toy().infer_sequence(obs)
    with trace.recording("t"):
        traced = _Toy().infer_sequence(obs)
    assert np.array_equal(untraced.waypoints, traced.waypoints)
    assert untraced.dt == traced.dt


def test_the_command_enum_survives_a_summary() -> None:
    """Steps get handed things `summarise` has no branch for. They must arrive
    as a readable repr rather than as a traceback."""
    assert trace.summarise(Command.STRAIGHT) == "Command.STRAIGHT"  # not the IntEnum's 2
    assert trace.summarise({"a": 1, "b": [1, 2]}) == {"a": 1, "b": [1, 2]}
    assert trace.summarise(list(range(100)))["n"] == 100
    assert len(trace.summarise(object())) <= 200
    assert trace.summarise(np.int64(3)) == 3
    assert trace.summarise(None) is None


def test_a_document_is_json_able_without_nan() -> None:
    """`paddock explain` writes with `allow_nan=False`, so a NaN anywhere in a
    trace would fail the write rather than produce a file nothing can parse."""
    import json

    @trace.observed("s")
    def s(x):
        return {"ratio": float("nan"), "arr": np.array([np.inf, 1.0])}

    with trace.recording("t") as rec:
        s(np.float64("nan"))
    json.dumps(rec.document(), allow_nan=False)  # must not raise


def test_trace_imports_nothing_from_paddock_above_its_foot() -> None:
    """The cycle rule, enforced rather than commented.

    `protocol` imports `trace` so that every adapter's protocol methods are
    wrapped; `trace` importing `protocol` back would break each model's
    interpreter at load, which is why `Trajectory` and `Observation` are
    recognised structurally here. The single import at the foot is the check
    registry, which imports only this module.
    """
    tree = ast.parse(Path(trace.__file__).read_text(encoding="utf-8"))
    imported = [
        (node.lineno, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("paddock")
    ] + [
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("paddock")
    ]
    assert imported == [(imported[0][0], "paddock.policy")], imported
    assert imported[0][0] > max(
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ), "the checks import must stay at the foot, or an empty registry becomes possible"


class _Stateful(Policy):
    """The KITScenes defect in miniature: state that survives a reset."""

    cameras = ("CAM_FRONT",)

    def __init__(self):
        self.seen = 0

    def reset(self):
        pass  # deliberately does not clear `seen`

    def infer(self, obs):
        self.seen += 1
        return constant_velocity(obs.speed, dt=self.waypoint_dt)


def test_reset_is_a_step_so_state_surviving_it_is_visible() -> None:
    """`reset` does no arithmetic and takes microseconds, which is exactly why
    it was invisible when a camera adapter leaked across scenes. In a trace it
    is a row, with the assumption it encodes written next to it."""
    policy = _Stateful()
    with trace.recording("t") as rec:
        policy.infer_sequence([_observation(t_us=i * 500_000) for i in range(2)])
    doc = rec.document()
    reset = doc["steps"][0]["steps"][0]
    assert reset["name"] == "reset"
    assert "reset" in trace.why_of(doc, reset) and policy.seen == 2
