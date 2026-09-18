"""Observability for inference: what each step did, and whether it was allowed to.

A benchmark score says *which* model is wrong. It never says why. A plan that
sits at the origin, a plan built by subsampling a 50-point output onto a 0.5 s
grid, and a plan warped from a camera whose mounting was derived wrongly all
produce the same shape of row in a leaderboard -- and the four worst defects
this repo has found were all of that kind (see `paddock.conformance`'s module
docstring: each ran to completion and published a plausible number).

This module is the instrument. It has three parts:

`observed` decorates functions that **already exist**. The adapters' private
methods -- `_bind_cameras`, `_repeats`, `plan_to_grid`, `build_extras`,
`_letterbox`, `_select_frames`, `resample_history` -- are already the right step
boundaries; annotating them adds nothing inline and restructures nothing. That
is the whole reason for the decorator form: an explainability layer that made
each adapter longer would be paid for on every read of a file that is already
1000 lines.

`recording` turns it on for one call. **When no recorder is active the wrapper
is one `ContextVar.get()` and a direct call** -- nothing is bound, summarised,
checked or retained. An untraced sweep pays a nanosecond-scale lookup per
decorated call, which is what keeps this usable on the 10,165-sample runs.

`CHECKS` are the invariants a step asserts about itself, each with the reason it
exists and an `attribution` saying whose bug a failure is: `adapter` (paddock
got it wrong and paddock can fix it), `harness` (the benchmark is asking
something the pairing cannot support) or `model` (the input was faithful, the
output was read in the model's own frame and units, and the plan is still
wrong). Only the last licenses a statement about model quality, and separating
the three is the point of the exercise.

## What a step records

Summaries, never payloads. An array over `MAX_ELEMENTS` becomes shape, dtype,
range, mean and a non-finite count; an image becomes a thumbnail. A whole trace
is kilobytes, so it rides back on the `/infer_sequence` response rather than
needing a second channel out of the model's interpreter.

## Why this module imports nothing from paddock

`protocol.py` imports it -- `Policy.__init_subclass__` wraps every adapter's
`load`/`reset`/`observe`/`infer`, so a model with no annotations still traces --
and importing `protocol` back would be a cycle. So `Trajectory`, `Observation`
and `Pinhole` are recognised structurally, by the attributes they carry, rather
than by `isinstance`. That also means this file can be imported inside a model's
own interpreter, which is as old as the model needs (Drive-JEPA's is 3.9) and
where the rest of paddock may not resolve.
"""

from __future__ import annotations

import base64
import functools
import inspect
import io
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "ATTRIBUTIONS",
    "BUDGET_BYTES",
    "CHECKS",
    "FULL_BUDGET_BYTES",
    "Check",
    "Recorder",
    "Step",
    "active",
    "budget_for",
    "check",
    "note",
    "observed",
    "recording",
    "summarise",
    "why_of",
]

#: Arrays with at most this many elements are kept verbatim; larger ones are
#: reduced to statistics. 512 covers a (33, 15) supercombo plan, a warp matrix,
#: a 20x3 trajectory and a camera's intrinsics -- the things a reader actually
#: recomputes by hand -- and excludes anything tensor-shaped.
MAX_ELEMENTS = 512

#: Longest edge of an image thumbnail, pixels. Big enough to see that the road
#: is the right way up and the horizon is where it should be, which is what a
#: preprocessing bug shows as.
THUMBNAIL_PX = 256

#: Soft cap on one trace. Past it, arrays and thumbnails stop being captured and
#: `truncated` is set; step names, timings, shapes and findings keep accruing,
#: because those are what the report is built from.
#:
#: Sized against the worst real case rather than a guess: a comma2k19 keyframe
#: replays a 160-frame history window, which is 321 steps and 672 findings --
#: 300 KB of scaffolding before a single array -- and lands at 570 KB with every
#: plan kept. A megabyte fits that with headroom and still puts a six-model
#: keyframe in single-digit megabytes. Every other benchmark here is an order of
#: magnitude under it.
#:
#: Eviction is first-come-first-served, so on an over-budget trace it is the
#: *late* steps that lose their arrays. That is survivable because the scored
#: plan is not in the step tree: `explain` writes `prediction` and
#: `ground_truth` at the document's top level, outside the budget entirely.
BUDGET_BYTES = 1_000_000

#: Cap for `capture="full"`, where every array is kept verbatim regardless of
#: `MAX_ELEMENTS`. Sixteen times the default because `full` exists for one sample
#: under suspicion of a sub-pixel defect, where the numbers themselves are the
#: evidence -- never for a sweep. Still a cap: a trace that exhausts a machine's
#: memory is a worse outcome than one that says it was truncated.
FULL_BUDGET_BYTES = 16_000_000

#: What a step and a finding cost in the document beyond any array they carry --
#: the name, uid, timing, the summary dicts around each input, the check name and
#: its detail sentence. Measured on a comma2k19 openpilot trace: 497 and 220
#: bytes respectively. Charged so the budget tracks the document rather than only
#: the arrays in it.
STEP_BYTES = 500
FINDING_BYTES = 220

#: Steps nested deeper than this are executed without being recorded. A guard
#: against a recursive or per-pixel helper being annotated by accident; the
#: adapters' real nesting is four or five.
MAX_DEPTH = 12

#: The three places a defect can live. See the module docstring.
ATTRIBUTIONS = ("adapter", "harness", "model")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One invariant a step can be asked to assert about itself.

    `test` is handed the finished `Step` -- its raw inputs and output, before
    summarising -- and returns `(ok, detail)`, or `None` to abstain when the
    step does not carry what the check needs. Abstaining is deliberate and not a
    pass: a check that cannot see its subject must say nothing rather than
    report green, which is how a check quietly stops checking.
    """

    name: str
    why: str
    attribution: str
    test: Callable[[Step], tuple[bool, str] | None]
    severity: str = "error"


CHECKS: dict[str, Check] = {}


def check(name: str, why: str, attribution: str, severity: str = "error"):
    """Register a check. Decorates the predicate.

    `why` is printed under a failure by `paddock explain`, the same way
    `conformance.CHECKS` entries are, so it should name the defect the check
    would have caught rather than restate the rule.
    """
    if attribution not in ATTRIBUTIONS:
        raise ValueError(f"attribution must be one of {ATTRIBUTIONS}, got {attribution!r}")

    def register(test):
        if name in CHECKS:
            raise ValueError(f"check {name!r} is registered twice")
        CHECKS[name] = Check(name=name, why=why, attribution=attribution, test=test, severity=severity)
        return test

    return register


# --------------------------------------------------------------------------
# Steps and the recorder
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One observed call.

    `inputs` and `output` hold the **raw** values for as long as the checks need
    them and are replaced by summaries immediately afterwards, so a trace never
    keeps a frame alive.
    """

    name: str
    why: str = ""
    seconds: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    #: Free-form facts a step chose to record about itself, via `note`.
    notes: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    children: list[Step] = field(default_factory=list)
    error: str | None = None
    #: Position in the recording, assigned on creation. The document's flattened
    #: `findings` carry it, which is how a finding is matched back to the one step
    #: that produced it -- a path cannot do that when 160 sibling `infer` steps
    #: all share `infer_sequence/observe/infer`.
    uid: int = -1

    def input(self, *names: str) -> Any:
        """The first of `names` this step was called with, or None.

        Steps are annotated on the signatures adapters already have, which do
        not agree on what the frame or the plan is called; a check that wants
        "the image, whatever this one calls it" asks for the alternatives.
        """
        for n in names:
            if n in self.inputs:
                return self.inputs[n]
        return None

    def record(self, whys: dict[str, int]) -> dict[str, Any]:
        """This step as JSON. `whys` interns the rationales; see `Recorder.document`."""
        out: dict[str, Any] = {"uid": self.uid, "name": self.name, "seconds": round(self.seconds, 6)}
        if self.why:
            out["why_id"] = whys.setdefault(self.why, len(whys))
        if self.inputs:
            out["inputs"] = self.inputs
        if self.output is not None:
            out["output"] = self.output
        if self.notes:
            out["notes"] = self.notes
        if self.error:
            out["error"] = self.error
        if self.children:
            out["steps"] = [c.record(whys) for c in self.children]
        return out


_ACTIVE: ContextVar[Recorder | None] = ContextVar("paddock_trace_recorder", default=None)


def active() -> Recorder | None:
    """The recorder listening right now, or None. The hot-path check."""
    return _ACTIVE.get()


class Recorder:
    """Collects the tree of steps for one traced call.

    Not thread-safe by design: a `ContextVar` follows an async task and a plain
    call chain, which is what `/infer_sequence` and `paddock explain` are. A
    policy that farms work out to a thread pool loses the steps inside those
    threads -- they simply do not appear, rather than appearing wrong.
    """

    def __init__(self, label: str, *, capture: str = "summary", budget_bytes: int = BUDGET_BYTES) -> None:
        self.label = label
        #: "summary" keeps arrays over `MAX_ELEMENTS` as statistics; "full"
        #: keeps everything the budget allows, for a suspected sub-pixel defect
        #: where the numbers themselves are the evidence.
        self.capture = capture
        self.budget_bytes = budget_bytes
        self.spent = 0
        self.truncated = False
        #: check name -> why it exists, for every check that actually produced a
        #: finding. Sent once instead of once per finding; see `_finding`.
        self.rationales: dict[str, str] = {}
        self._next_uid = 0
        self.root = Step(name=label)
        self._stack: list[Step] = [self.root]
        self._started = time.perf_counter()

    # -- budget --------------------------------------------------------

    def afford(self, n_bytes: int) -> bool:
        """Charge `n_bytes` against the budget; False once it is gone.

        For the optional part of a document -- an array's values, a thumbnail.
        Refusing to spend is what keeps the trace inside the cap.
        """
        if self.spent + n_bytes > self.budget_bytes:
            self.truncated = True
            return False
        self.spent += n_bytes
        return True

    def charge(self, n_bytes: int) -> None:
        """Charge bytes that are going to be written regardless.

        A step's name, timing and findings are what a report is built from, so
        they accrue past the cap rather than being dropped. They still have to be
        *counted*: on a comma2k19 window 321 steps and 672 findings are 300 KB of
        scaffolding, and a budget that ignores them is a budget that never
        engages while the document quietly reaches a megabyte. Going over stops
        the next array instead.
        """
        self.spent += n_bytes
        if self.spent > self.budget_bytes:
            self.truncated = True

    # -- recording -----------------------------------------------------

    def run(
        self,
        name: str,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        *,
        why: str,
        checks: Sequence[str],
        capture: Sequence[str],
        bind: Callable[[tuple, dict], dict],
    ) -> Any:
        if len(self._stack) > MAX_DEPTH:
            return fn(*args, **kwargs)
        self._next_uid += 1
        step = Step(name=name, why=why, uid=self._next_uid)
        self._stack[-1].children.append(step)
        self._stack.append(step)
        raw_out = _MISSING
        started = time.perf_counter()
        try:
            raw_out = fn(*args, **kwargs)
            return raw_out
        except Exception as exc:  # noqa: BLE001 -- re-raised; the trace just wants the reason
            step.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step.seconds = time.perf_counter() - started
            self._stack.pop()
            try:
                raw_in = bind(args, kwargs)
            except Exception:  # noqa: BLE001 -- a trace must never be the thing that fails
                raw_in = {}
            self._finish(step, raw_in, raw_out, checks, capture)

    def _finish(self, step: Step, raw_in: dict, raw_out: Any, checks, capture) -> None:
        """Run the checks against raw values, then replace them with summaries."""
        self.charge(STEP_BYTES)
        step.inputs = raw_in
        step.output = None if raw_out is _MISSING else raw_out
        for name in checks:
            spec = CHECKS.get(name)
            if spec is None:
                missing = f"check {name!r} is not registered"
                self.charge(FINDING_BYTES)
                step.findings.append(_finding(name, False, missing, "adapter"))
                continue
            try:
                verdict = spec.test(step)
            except Exception as exc:  # noqa: BLE001
                verdict = (False, f"the check itself raised {type(exc).__name__}: {exc}")
            if verdict is None:
                continue  # abstained: this step does not carry what the check needs
            ok, detail = verdict
            self.rationales[spec.name] = spec.why
            self.charge(FINDING_BYTES)
            step.findings.append(_finding(spec.name, ok, detail, spec.attribution, severity=spec.severity))

        full = set(capture)
        step.inputs = {
            k: summarise(v, self, full=self.capture == "full" or k in full) for k, v in raw_in.items()
        }
        step.output = (
            None
            if raw_out is _MISSING
            else summarise(raw_out, self, full=self.capture == "full" or "return" in full)
        )

    def note(self, **facts: Any) -> None:
        """Record facts about the step being executed right now."""
        self._stack[-1].notes.update({k: summarise(v, self) for k, v in facts.items()})

    # -- output --------------------------------------------------------

    def findings(self) -> list[dict[str, Any]]:
        """Every finding in the tree, flattened, each carrying its step path.

        Flattened **only**: `Step.record` no longer repeats them inside the tree,
        because a finding stored in both places is a finding stored twice, and on
        a comma2k19 window that was 115 KB of the document saying the same thing
        as another 140 KB. `uid` is what puts each one back on its own step.
        """
        out: list[dict[str, Any]] = []

        def walk(step: Step, path: str) -> None:
            here = f"{path}/{step.name}" if path else step.name
            for f in step.findings:
                out.append({**f, "step": here, "uid": step.uid})
            for child in step.children:
                walk(child, here)

        for child in self.root.children:
            walk(child, "")
        # Failures first, then warnings, then the passes -- a report is read
        # from the top and the top should be what is wrong.
        rank = {"error": 0, "warning": 1}
        return sorted(out, key=lambda f: (f["ok"], rank.get(f["severity"], 2)))

    def document(self) -> dict[str, Any]:
        """The JSON-able trace. Safe to call once the recording has closed.

        Two string tables, for the same reason: a decorator's `why` and a check's
        rationale are each a paragraph attached to something that happens once per
        history frame. Written inline they were 40% of a comma2k19 document.
        `why_id` indexes `whys`; a finding's `check` keys `rationales`.
        """
        whys: dict[str, int] = {}
        steps = [c.record(whys) for c in self.root.children]
        return {
            "version": 1,
            "label": self.label,
            "seconds": round(time.perf_counter() - self._started, 6),
            "capture": self.capture,
            "truncated": self.truncated,
            "approx_bytes": self.spent,
            "steps": steps,
            "whys": list(whys),
            "findings": self.findings(),
            "rationales": dict(sorted(self.rationales.items())),
        }


class _Missing:
    def __repr__(self) -> str:
        return "<no value: the call raised>"


_MISSING = _Missing()


def _finding(name, ok, detail, attribution, severity="error") -> dict[str, Any]:
    """Shaped exactly like `conformance.Finding`, plus `attribution`.

    A dict rather than the dataclass because `conformance` imports `protocol`,
    which imports this module; and because it has to cross JSON out of the
    model's interpreter anyway. `paddock.explain` rebuilds `Finding`s from these
    so both commands print with one vocabulary.

    The `why` is deliberately **not** here. A check's rationale is a paragraph and
    the same check fires once per history frame -- 160 times on a comma2k19
    window -- so carrying it per finding spent a third of the document repeating
    one sentence. It travels once, in the document's `rationales` map, keyed by
    check name; it still has to travel at all because a model-specific check is
    registered in the adapter's interpreter, which paddock never imports.
    """
    return {
        "check": name,
        "ok": bool(ok),
        "severity": severity,
        "detail": detail,
        "attribution": attribution,
    }


def budget_for(capture: str) -> int:
    """The byte budget a capture mode gets. One definition, two callers."""
    return FULL_BUDGET_BYTES if capture == "full" else BUDGET_BYTES


def why_of(document: dict[str, Any], step: dict[str, Any]) -> str:
    """A recorded step's rationale, resolved through the document's `whys` table.

    One place doing the indirection, so a reader -- the report, the viewer, a
    notebook -- never has to know the strings are interned.
    """
    whys = document.get("whys") or []
    index = step.get("why_id")
    if index is None or not isinstance(index, int) or not 0 <= index < len(whys):
        return str(step.get("why", ""))  # a document written before the table
    return str(whys[index])


@contextmanager
def recording(label: str, *, capture: str = "summary", budget_bytes: int | None = None) -> Iterator[Recorder]:
    """Listen to every `observed` call made inside this block."""
    recorder = Recorder(label, capture=capture, budget_bytes=budget_bytes or budget_for(capture))
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


def note(**facts: Any) -> None:
    """Record facts about the step currently executing. A no-op when not tracing.

    For the things a step knows and its signature does not show -- how many
    waypoints an alignment trimmed, which of two branches was taken, how many
    history slots were filled by a stand-in.
    """
    recorder = _ACTIVE.get()
    if recorder is not None:
        recorder.note(**facts)


def observed(
    name: str | None = None,
    *,
    why: str = "",
    checks: Sequence[str] = (),
    capture: Sequence[str] = (),
    skip: Sequence[str] = (),
):
    """Record this call when a trace is listening, and do nothing otherwise.

    `why` is the assumption this step encodes, in one sentence -- the thing a
    reader needs in order to judge whether the step is right, which is usually
    a frame convention, a unit, or what the model's native grid is.

    `checks` names entries in `CHECKS`. `capture` names parameters (or
    `"return"`) to keep verbatim past `MAX_ELEMENTS`. `skip` drops parameters
    from the record entirely -- for a weights object or a session handle whose
    `repr` is noise.

    Apply **under** `@staticmethod`/`@classmethod`, never over them:
    a staticmethod object is not callable on 3.9, which is a live version here.
    """

    def decorate(fn: Callable) -> Callable:
        step_name = name or fn.__name__
        skipped = set(skip) | {"self", "cls"}
        # Built on the first *recorded* call, not at import: `inspect.signature`
        # is not free and most processes never trace anything.
        cache: dict[str, Any] = {}

        def bind(args: tuple, kwargs: dict) -> dict:
            sig = cache.get("sig")
            if sig is None:
                sig = cache["sig"] = inspect.signature(fn)
            bound = sig.bind_partial(*args, **kwargs)
            return {k: v for k, v in bound.arguments.items() if k not in skipped}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            recorder = _ACTIVE.get()
            if recorder is None:
                return fn(*args, **kwargs)
            return recorder.run(
                step_name, fn, args, kwargs, why=why, checks=checks, capture=capture, bind=bind
            )

        wrapper.__paddock_step__ = step_name  # type: ignore[attr-defined]
        return wrapper

    return decorate


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _is_image(value: np.ndarray) -> bool:
    return value.ndim == 3 and value.shape[2] in (3, 4) and value.dtype == np.uint8


def _thumbnail(image: np.ndarray, recorder: Recorder | None) -> str | None:
    """A base64 PNG of `image`, long edge `THUMBNAIL_PX`, or None.

    Returns None rather than raising when Pillow is absent: this runs inside the
    model's interpreter, and a model that does its own decoding may not have it.
    The array statistics beside it still say whether the frame was black.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    h, w = image.shape[:2]
    if max(h, w) < 1:
        return None
    scale = THUMBNAIL_PX / max(h, w)
    size = (max(1, int(w * scale)), max(1, int(h * scale))) if scale < 1 else (w, h)
    buffer = io.BytesIO()
    Image.fromarray(image[:, :, :3]).resize(size).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    if recorder is not None and not recorder.afford(len(encoded)):
        return None
    return encoded


def _plain(value: Any) -> Any:
    """`tolist()` hands back `inf` and `nan`, which JSON has no spelling for.

    `paddock explain` writes with `allow_nan=False` on purpose -- a file only
    Python can read is not a record. So a non-finite number travels as its name,
    which is also how `summarise` sends a bare float.
    """
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


#: JSON bytes one array element costs, by dtype kind, including its separator.
#: Measured, not guessed: `json.dumps` of a float array of realistic waypoint
#: magnitudes runs 20.0-20.5 bytes per number, because `tolist()` hands back
#: Python floats that print to full round-trip precision -- and float32 costs
#: the same, since a float32 widened to double prints as `30.123456954956055`.
#: An earlier estimate of 12 undercounted a comma2k19 trace by more than ten
#: times, which is how the budget stopped being a budget.
_JSON_BYTES = {"f": 20, "i": 5, "u": 5, "b": 7}
_JSON_BYTES_DEFAULT = 20


def _json_bytes_per(dtype: np.dtype) -> int:
    """What one element of `dtype` costs in the document, rounded up."""
    return _JSON_BYTES.get(dtype.kind, _JSON_BYTES_DEFAULT)


def _array(value: np.ndarray, recorder: Recorder | None, full: bool) -> dict[str, Any]:
    finite = np.isfinite(value) if value.dtype.kind == "f" else None
    n_bad = 0
    out: dict[str, Any] = {
        "kind": "array",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if finite is not None:
        n_bad = int(value.size - finite.sum())
        if n_bad:
            # Counted as well as shown: a single NaN in a plan is the defect
            # `Trajectory.__post_init__` refuses, and in an intermediate it is
            # the thing worth seeing before the refusal.
            out["non_finite"] = n_bad
    usable = value[finite] if finite is not None and finite.any() else (value if finite is None else None)
    if usable is not None and usable.size:
        out["min"] = float(np.min(usable))
        out["max"] = float(np.max(usable))
        out["mean"] = float(np.mean(usable))
    if _is_image(value):
        out["kind"] = "image"
        thumb = _thumbnail(value, recorder)
        if thumb:
            out["png_b64"] = thumb
        return out
    keep = full or value.size <= MAX_ELEMENTS
    if keep and (recorder is None or recorder.afford(value.size * _json_bytes_per(value.dtype))):
        listed = np.asarray(value).tolist()
        out["values"] = _plain(listed) if n_bad else listed
    return out


def _trajectory(value: Any, recorder: Recorder | None, full: bool) -> dict[str, Any]:
    xy = np.asarray(value.waypoints, dtype=np.float64)[:, :2]
    origin = np.zeros((1, 2))
    length = float(np.sum(np.linalg.norm(np.diff(np.concatenate([origin, xy]), axis=0), axis=1)))
    out = {
        "kind": "trajectory",
        "n": int(len(xy)),
        "dt": float(value.dt),
        # The convention worth restating at every sight of a plan: waypoint 0 is
        # already `speed * dt` out, so a "first waypoint at the origin" is a
        # commanded stop and not a plan that starts where the car is.
        "first": [float(x) for x in xy[0]] if len(xy) else None,
        "last": [float(x) for x in xy[-1]] if len(xy) else None,
        "path_length_m": length,
        "extras": sorted(value.extras) if getattr(value, "extras", None) else [],
    }
    out["waypoints"] = _array(xy, recorder, full)
    return out


def _observation(value: Any, recorder: Recorder | None, full: bool) -> dict[str, Any]:
    cameras = sorted(set(value.images) | set(value.image_paths))
    out: dict[str, Any] = {
        "kind": "observation",
        "cameras": cameras,
        "timestamp_us": int(value.timestamp_us),
        "speed_mps": float(value.speed),
        "yaw_rate_rps": float(value.yaw_rate),
        "carries_pixels": bool(value.images),
    }
    if getattr(value, "speed_limit", None) is not None:
        out["speed_limit_mps"] = float(value.speed_limit)
    for cam in cameras:
        m = value.camera2ego.get(cam)
        if m is not None:
            out.setdefault("mounting_off_level_deg", {})[cam] = round(mounting_off_level_deg(m), 3)
    return out


def summarise(value: Any, recorder: Recorder | None = None, full: bool = False) -> Any:
    """One value, small enough to put in a trace.

    Dispatch is structural rather than by `isinstance`, so this module stays
    free of a `protocol` import -- see the module docstring.
    """
    if isinstance(value, Enum):
        # Before the int branch: `Command` is an IntEnum, and a trace that says
        # `command: 2` has thrown away the only part a reader needs. Spelled out
        # rather than `str()`, which since 3.11 gives an IntEnum back its number.
        return f"{type(value).__name__}.{value.name}"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return summarise(value.item(), recorder, full)
    if isinstance(value, np.ndarray):
        return _array(value, recorder, full)
    if hasattr(value, "waypoints") and hasattr(value, "dt"):
        return _trajectory(value, recorder, full)
    if hasattr(value, "canbus") and hasattr(value, "camera2ego"):
        return _observation(value, recorder, full)
    if isinstance(value, dict):
        return {str(k): summarise(v, recorder, full) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        summarised = [summarise(v, recorder, full) for v in items[:32]]
        return summarised if len(items) <= 32 else {"kind": "sequence", "n": len(items), "head": summarised}
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


# --------------------------------------------------------------------------
# Geometry shared with the checks
# --------------------------------------------------------------------------

#: paddock ego [forward, left, up] <-> a camera frame [right, down, forward].
#: An *unrotated* camera is this matrix, not the identity -- the distinction
#: that cost a published ADE of 38.8 m. Restated here rather than imported from
#: `conformance` only because of the import cycle; `tests/test_trace.py` pins
#: the two against each other so they cannot drift.
UNROTATED_CAMERA2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def mounting_off_level_deg(camera2ego) -> float:
    """How far this mounting is from level, degrees, with yaw divided out.

    Which way a camera faces is a rig decision; whether it is level is not.
    Same derivation as `conformance.check_camera_mounting`, which explains why
    yaw has to come out first (nuScenes' ring sits at 55, 70, 110 and 180
    degrees and every one of those is correctly mounted).
    """
    r = np.asarray(camera2ego, dtype=np.float64)[:3, :3]
    forward = r[:, 2]
    yaw = float(np.arctan2(forward[1], forward[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    about_up = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    residual = r @ (about_up @ UNROTATED_CAMERA2EGO).T
    return float(np.degrees(np.arccos(np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0))))


# Imported from the foot rather than by each caller: `CHECKS` is a registry, and
# a registry that is only populated if somebody remembered to import the module
# that fills it is a check that silently stops checking. `checks` imports this
# module and nothing else from paddock, so there is no cycle.
from paddock.policy import checks as _checks  # noqa: E402,F401  (registration side effect)
