"""`paddock explain` -- dissect one scored moment, step by step, model by model.

A leaderboard row says which model is wrong. A viewer says where. Neither says
why, and the difference matters because the three explanations for a bad number
need three different fixes:

* **adapter** -- paddock got it wrong and paddock can fix it. A frame flip, a
  mounting derived from the wrong matrix, a warp sampling outside the image, a
  native plan resampled onto the wrong grid. Every historical defect in
  `conformance`'s module docstring is this class.
* **harness** -- the benchmark is asking something the pairing cannot support.
  A horizon past the model's native support, a rate below `min_input_hz`, a
  keyframe whose recorded future is too short for the metric to mean anything.
  Fixable by changing the question, not the model.
* **model** -- the input was faithful, the output was read in the model's own
  frame, grid and units, and the plan is still wrong. Only this licenses a
  statement about model quality.

So this command runs the **same** path `paddock run` uses -- the model's own
interpreter, started by `runtime.proc`, answering `/infer_sequence` -- with
`trace: true` set, and adds the scoring half from paddock's own process. Two
sections, one sample id: pixels to warp to waypoints in one, alignment to ADE in
the other. Then it sorts every finding into the three classes above and says
which.

    paddock explain --model openpilot --bench comma2k19 --sample comma2k19/.../519
    paddock explain --model all --bench comma2k19 --scene 3/99c94dc769b5d96e/16
    paddock explain ... --format json --capture full

It is deliberately **not** a flag on `paddock run`. Published numbers stay
untouched and this is pointed at the handful of samples something else raised a
question about. Nothing here writes into `results/`.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from paddock import conformance as C
from paddock.config import SiteConfig
from paddock.policy import trace
from paddock.policy.protocol import Trajectory
from paddock.registry import Registry
from paddock.runtime import ports, proc

__all__ = ["explain", "explanation_path"]

#: Waypoint spacing assumed for a benchmark's recorded future when scoring it
#: against itself. The same 0.5 s `conformance.benchmark_findings` assumes, and
#: for the same reason: every open-loop benchmark here grids ground truth at
#: half a second, and one that does not shows up as a non-zero self-score --
#: which is precisely the signal this check exists to raise.
GT_DT = 0.5

#: A self-score above this is not rounding. `conformance` uses the same number
#: for the benchmark-wide version of this check.
GT_SELF_SCORE_TOL = 0.05


def explanation_path(cfg: SiteConfig, bench: str, sample_id: str, model: str) -> Path:
    """Where one dissection is written.

    Beside `_viewer/` and `_compare/` under `{roots.runs}`, i.e. outside the
    repo: these are hundreds of kilobytes with thumbnails in them, and results/
    is the part that has to stay reviewable in a diff.
    """
    return cfg.runs / bench / "_explain" / _slug(sample_id) / f"{model}.json"


def _slug(sample_id: str) -> str:
    """A sample id as one path component. Reversible enough to grep for."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_") or "sample"


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


def _findings(document: dict[str, Any]) -> list[C.Finding]:
    """The whole document's findings as `conformance.Finding`s.

    One vocabulary for `paddock verify` and `paddock explain`, so a check name
    means the same thing in both and a collision between them is a real conflict
    rather than two registries drifting apart.
    """
    out: list[C.Finding] = []
    for section in ("inference", "scoring"):
        part = document.get(section, {})
        # A check's `why` travels once per document rather than once per finding;
        # `C.CHECKS` is the fallback for a section written before that, and for
        # the scoring half, whose checks paddock registers in its own process.
        rationales = part.get("rationales", {})
        for raw in part.get("findings", []):
            name = raw["check"]
            out.append(
                C.Finding(
                    check=f"{name} [{raw.get('step', section)}]",
                    ok=bool(raw["ok"]),
                    severity=str(raw.get("severity", "error")),
                    detail=str(raw.get("detail", "")),
                    rationale=str(raw.get("rationale") or rationales.get(name) or C.CHECKS.get(name, "")),
                    attribution=str(raw.get("attribution", "")),
                )
            )
    out += [C.Finding(**f) for f in document.get("extra_findings", [])]
    # Failures first: a report is read from the top, and the top should be what
    # is wrong. Stable within a rank, so two runs of the same sample diff cleanly.
    rank = {"error": 0, "warning": 1}
    return sorted(out, key=lambda f: (f.ok, rank.get(f.severity, 2)))


def _deduplicate(findings: list[C.Finding]) -> list[C.Finding]:
    """Collapse a check that fired identically on every frame of a history.

    A 160-frame window runs the observation checks 160 times and a wall of
    identical failures buries the one that only fired once. The count is kept,
    because "on every frame" and "on one frame" are different defects.
    """
    seen: dict[tuple[str, bool, str], list[C.Finding]] = {}
    for f in findings:
        key = (f.check.split(" [")[0], f.ok, f.detail)
        seen.setdefault(key, []).append(f)
    out: list[C.Finding] = []
    for group in seen.values():
        first = group[0]
        if len(group) == 1:
            out.append(first)
            continue
        # One bracket, not every trailing one: a step path may end in its own,
        # as `observe[0]` does, and eating it would misname where this fired.
        where = first.check.split(" [", 1)[-1].removesuffix("]")
        out.append(
            C.Finding(
                check=f"{first.check.split(' [')[0]} [x{len(group)}, e.g. {where}]",
                ok=first.ok,
                severity=first.severity,
                detail=first.detail,
                rationale=first.rationale,
                attribution=first.attribution,
            )
        )
    return out


def _verdict(findings: list[C.Finding]) -> list[str]:
    """The lines this command exists to print: adapter, harness, or model?

    A statement about the model is only licensed once the other two classes are
    clean, so the absence of adapter and harness failures is reported as
    explicitly as their presence. It is still a claim rather than a proof --
    a trace shows what the adapter did, and can only say of the model that no
    other explanation was found. `docs/reviewing.md`'s standard applies:
    separate what was verified from what is argued.
    """
    failed = [f for f in findings if not f.ok and f.severity == "error"]
    warned = [f for f in findings if not f.ok and f.severity == "warning"]
    lines = ["", "verdict"]
    for attribution in trace.ATTRIBUTIONS:
        here = [f for f in failed if f.attribution == attribution]
        if here:
            lines.append(f"  {attribution:<8} {len(here)} failing check(s):")
            lines += [f"             - {f.check.split(' [')[0]}: {f.detail}" for f in here[:6]]
        else:
            lines.append(f"  {attribution:<8} clean")
    if warned:
        lines.append(f"  ({len(warned)} warning(s), which do not decide the verdict)")
    if not failed:
        lines.append(
            "  Every invariant this trace could assert holds, so nothing here explains the "
            "number on the adapter or harness side. That is the weakest of the three claims: "
            "it says no other explanation was found, not that the model is at fault."
        )
    return lines


# --------------------------------------------------------------------------
# One (model, sample) dissection
# --------------------------------------------------------------------------


def _infer_with_trace(url: str, sample, waypoint_dt: float, capture: str) -> tuple[Trajectory, dict]:
    """POST the sample with tracing on; return the plan and the model's half.

    The same request `runner.score_one` makes, plus two keys. The plan that
    comes back is the plan that would have been scored -- which is the whole
    reason the trace is a flag on the scoring endpoint rather than a second one.
    """
    import httpx

    payload = {
        "observations": [o.to_wire() for o in sample.observations],
        "trace": True,
        "capture": capture,
    }
    response = httpx.post(f"{url}/infer_sequence", json=payload, timeout=900.0)
    response.raise_for_status()
    body = response.json()
    prediction = Trajectory(body["trajectory"], dt=waypoint_dt, extras=body.get("extras"))
    section = body.get("trace") or {}
    section["assumptions"] = body.get("assumptions") or []
    with contextlib.suppress(Exception):  # a node without /diagnostics is not a failure
        section["diagnostics"] = httpx.get(f"{url}/diagnostics", timeout=60.0).json()
    return prediction, section


def _score_with_trace(harness, spec, sample, prediction: Trajectory) -> tuple[dict, dict, list[dict]]:
    """(metrics, the scoring half, extra findings) -- paddock's own arithmetic.

    Runs in paddock's interpreter, so the second half of "are the formulae
    implemented correctly" is visible on the same document as the first.
    """
    with trace.recording(f"score:{sample.sample_id}") as recorder:
        metrics = harness.score(sample, prediction)
    section = recorder.document()
    section["metrics"] = {k: _finite(v) for k, v in metrics.items()}
    return metrics, section, _ground_truth_findings(harness, spec, sample)


def _ground_truth_findings(harness, spec, sample) -> list[dict]:
    """Score this sample's recorded future against itself; it must come out ~0.

    `conformance` runs this once per benchmark. Running it again *here*, on the
    sample under examination, is what separates "this metric is broken" from
    "this metric is fine and this model is wrong" at the moment the question is
    being asked, rather than only in aggregate.
    """
    primary = (spec.primary_metric if spec else None) or "ade"
    if sample.ground_truth is None:
        return []
    gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
    try:
        value = float(harness.score(sample, Trajectory(gt, dt=GT_DT)).get(primary, float("nan")))
    except Exception as exc:  # noqa: BLE001 -- a broken metric is the finding
        value, detail = float("nan"), f"{type(exc).__name__}: {exc}"
    else:
        detail = f"the recorded future scored against itself gives {primary} = {value:.4f}, expected ~0"
    return [
        {
            "check": "sample.score_of_ground_truth",
            "ok": bool(np.isfinite(value) and abs(value) < GT_SELF_SCORE_TOL),
            "severity": "error",
            "detail": detail,
            "rationale": C.CHECKS["benchmark.score_of_ground_truth"],
            "attribution": "harness",
        }
    ]


def _finite(value: Any) -> Any:
    """NaN is not JSON. Kept as its own string so "not measured" stays visible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if np.isfinite(number) else str(number)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def _resolve_samples(harness, bench: str, sample_ids: list[str], scenes: list[str], echo) -> list:
    """Build the named samples, plus every sample in the named scenes."""
    wanted = list(sample_ids)
    for scene in scenes:
        found = harness.scene_sample_ids(scene)
        if not found:
            echo(
                f"  {bench} names no samples in scene {scene!r}. A scene key is the benchmark's "
                f"own -- for comma2k19 it is <chunk>/<route_id>/<segment>, the same key "
                f"`scenes/*.yaml` lists."
            )
        wanted += found
    if not wanted:
        return []
    echo(f"  building {len(wanted)} sample(s) (this reads the dataset)")
    built = list(harness.samples_by_id(wanted))
    missing = sorted(set(wanted) - {s.sample_id for s in built})
    for sample_id in missing:
        echo(f"  no such sample: {sample_id}")
    return built


def explain(
    cfg: SiteConfig,
    *,
    model: str,
    bench: str,
    sample_ids: list[str],
    scenes: list[str] | None = None,
    capture: str = "summary",
    fmt: str = "text",
    echo=print,
) -> int:
    """Dissect the named samples for the named model(s). Returns an exit code."""
    from paddock.runner import load_benchmark

    registry = Registry.load()
    spec = registry.benchmarks.get(bench)
    if spec is None:
        echo(f"unknown benchmark {bench!r}; known: {', '.join(sorted(registry.benchmarks)) or 'none'}")
        return 2
    if model == "all":
        models = [m for m in registry.models.values() if bench in m.benchmarks]
        if not models:
            echo(f"no models declare benchmark {bench!r}")
            return 2
    else:
        chosen = registry.models.get(model)
        if chosen is None:
            echo(f"unknown model {model!r}; known: {', '.join(sorted(registry.models)) or 'none'}")
            return 2
        models = [chosen]

    harness = load_benchmark(spec, cfg)
    try:
        samples = _resolve_samples(harness, bench, sample_ids, scenes or [], echo)
    except ValueError as exc:  # a harness refusing a reachable-looking id says why
        echo(f"  {exc}")
        return 2
    if not samples:
        echo("nothing to explain: pass --sample <id> or --scene <key>")
        return 2

    documents: list[dict[str, Any]] = []
    for model_spec in models:
        echo(f"\n=== {model_spec.name} ===")
        try:
            documents += _explain_model(cfg, spec, harness, model_spec, samples, capture, echo)
        except Exception as exc:  # noqa: BLE001 -- one model's node dying is not the others'
            echo(f"  {model_spec.name}: FAILED - {type(exc).__name__}: {exc}")

    if not documents:
        echo("\nnothing was traced")
        return 1
    if fmt == "json":
        echo(json.dumps(documents, indent=2, allow_nan=False))
        return 0
    echo(_render(documents))
    return 0


def _explain_model(cfg, spec, harness, model_spec, samples, capture: str, echo) -> list[dict[str, Any]]:
    """Every sample through one model, on one node.

    A node is a checkpoint read and a VRAM wait, so it is started once for the
    whole set rather than once per sample -- but `infer_sequence` resets the
    policy on every call, so the samples stay independent.
    """
    from paddock.runner import base_port

    port = ports.find_free(base_port())
    log_dir = cfg.runs / spec.name / "_explain" / "nodes"
    out: list[dict[str, Any]] = []
    with proc.model_node(model_spec, cfg, port, log_dir, echo=echo) as url:
        for sample in samples:
            echo(f"  {sample.sample_id}")
            prediction, inference = _infer_with_trace(url, sample, model_spec.waypoint_dt, capture)
            metrics, scoring, extra = _score_with_trace(harness, spec, sample, prediction)
            document = {
                "version": 1,
                "bench": spec.name,
                "model": model_spec.name,
                "sample_id": sample.sample_id,
                "native_id": sample.native_id,
                "capture": capture,
                "waypoint_dt": model_spec.waypoint_dt,
                "prediction": prediction.xy.tolist(),
                "ground_truth": np.asarray(sample.ground_truth)[:, :2].tolist()
                if sample.ground_truth is not None
                else None,
                "metrics": {k: _finite(v) for k, v in metrics.items()},
                "inference": inference,
                "scoring": scoring,
                "extra_findings": extra,
            }
            path = explanation_path(cfg, spec.name, sample.sample_id, model_spec.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, indent=2, allow_nan=False))
            echo(f"    -> {path}")
            out.append(document)
    return out


def _render(documents: list[dict[str, Any]]) -> str:
    """The terminal report: failures in full, passes as a tally, then a verdict.

    The default `observe` runs a whole forward pass per history frame, so a
    160-frame comma2k19 window asserts the plan invariants 160 times. Printing
    every green line would bury the one red one under three hundred, which is
    the opposite of what a report is for -- so a passing check is counted, and
    only what failed gets its detail and its `why`.
    """
    lines: list[str] = []
    everything: list[C.Finding] = []
    for document in documents:
        raw = _findings(document)
        failures = _deduplicate([f for f in raw if not f.ok])
        everything += failures
        header = f"{document['model']} x {document['sample_id']}"
        lines.append(f"\n{header}")
        lines.append(f"  {_metric_line(document)}")
        for assumption in document.get("inference", {}).get("assumptions", []):
            step = f" [{assumption['step']}]" if assumption.get("step") else ""
            lines.append(f"  assumes{step} {assumption['what']}")
            lines.append(f"           source: {assumption['source']}")
        if not raw:
            lines.append("  (no invariant applied to any recorded step)")
        lines += [f.render() for f in failures]
        lines += _passes_line([f for f in raw if f.ok])
        lines.append(f"  {_step_line(document)}")
    lines += _verdict(everything)
    return "\n".join(lines)


def _passes_line(passed: list[C.Finding]) -> list[str]:
    """What held, as counts. Evidence that the checks ran, not a wall of green.

    The count matters: a check that asserted itself once and one that asserted
    itself on all 160 history frames are different amounts of evidence, and the
    difference is invisible if both print as "ok".
    """
    if not passed:
        return []
    counted = Counter(f.check.split(" [")[0] for f in passed)
    shown = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in counted.most_common())
    return [f"  [ok  ] {len(passed)} passing assertion(s): {shown}"]


def _metric_line(document: dict[str, Any]) -> str:
    metrics = document.get("metrics") or {}
    shown = ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(metrics.items())[:6]
    )
    return shown or "(no metrics)"


def _step_line(document: dict[str, Any]) -> str:
    """How much of the chain this trace actually covers, stated rather than implied.

    An un-annotated adapter produces the protocol skeleton and nothing else, and
    a thin trace that looks complete is worse than one that says it is thin.
    """
    inference, scoring = document.get("inference") or {}, document.get("scoring") or {}
    n_in, n_score = _count_steps(inference.get("steps", [])), _count_steps(scoring.get("steps", []))
    note = " (truncated: over budget)" if inference.get("truncated") or scoring.get("truncated") else ""
    return f"{n_in} inference step(s), {n_score} scoring step(s){note}"


def _count_steps(steps: list[dict[str, Any]]) -> int:
    return sum(1 + _count_steps(s.get("steps", [])) for s in steps)
