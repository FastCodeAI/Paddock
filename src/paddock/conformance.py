"""The adapter contract, as checks that can fail.

`docs/adding-a-model.md` and `docs/adding-a-benchmark.md` describe what an
adapter must do. This module *executes* that description, so a new adapter is
verified rather than reviewed:

    paddock verify --model openpilot --bench comma2k19

Use it three ways -- from the CLI, from an adapter's own test module
(`from paddock.conformance import policy_findings`; one test per finding), and
from `paddock run`, which calls the cheap subset in preflight.

## Why this exists rather than a checklist

Every serious defect found in this repo so far has been a *silent* contract
violation, not a crash:

* comma2k19 passed `camera2ego = np.eye(4)`, which reads like "no mounting
  rotation" and is a 90 degree error in paddock's frame conventions. The model
  was shown a sideways world, predicted a stationary car, and the run completed
  and reported ADE 38.8 m as though it were a modelling result.
* KITScenes reused the first scene's camera adapter for every later scene, so a
  quarter of samples were warped with the wrong mounting rotation.
* The closed-loop openpilot node sent `traffic_convention = [0, 1]` while the
  open-loop harness sent `[1, 0]`, so the same model was told it was driving on
  the other side of the road depending on the loop type.

None of those would be caught by "does it run". Each is caught below. The
checks are therefore chosen from failures that actually happened, not from a
notion of completeness -- `CHECKS` documents the provenance of each.

## The two checks worth understanding

`benchmark.score_of_ground_truth` scores the recorded future **against itself**.
A correct metric returns ~0. Anything else means the metric and the ground truth
disagree about a frame, a sign, or a time grid -- and that error then sits
underneath every number the benchmark ever reports, applying equally to every
model so that no comparison reveals it.

`benchmark.score_of_a_stationary_plan` scores an all-zero trajectory. It must
come out *unmistakably* bad. If a benchmark's metric set cannot distinguish
"predicted nothing" from "tracked the lane well", it cannot support the
conclusion anyone will draw from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from paddock.policy.protocol import N_WAYPOINTS, Command, Observation, Policy, Trajectory

__all__ = [
    "CHECKS",
    "Finding",
    "benchmark_findings",
    "check_camera_mounting",
    "pair_findings",
    "policy_findings",
    "render_findings",
    "synthetic_observation",
]

#: A camera further from level than this is a coordinate-convention error, not a
#: mounting. Wider than openpilot's own tici-specific limits, because the job
#: here is only to catch a frame slip -- and a 90 degree one is two orders of
#: magnitude outside it.
MAX_MOUNT_DEG = 15.0

#: paddock ego [forward, left, up] <-> a camera-frame [right, down, forward].
#: An *unrotated* camera is this matrix, not the identity.
UNROTATED_CAMERA2EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass
class Finding:
    """One contract check's result."""

    check: str
    ok: bool
    #: "error" fails the run; "warning" is reported and does not.
    severity: str = "error"
    detail: str = ""
    #: Why this check exists -- usually a defect it would have caught.
    rationale: str = ""

    @property
    def failed(self) -> bool:
        return not self.ok and self.severity == "error"

    def render(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.severity == "error" else "warn")
        line = f"  [{mark}] {self.check}"
        if self.detail:
            line += f"\n         {self.detail}"
        if not self.ok and self.rationale:
            line += f"\n         why: {self.rationale}"
        return line


#: What each check is for, and where it came from. Kept as data so
#: `paddock verify --explain` can print it and so the provenance does not rot
#: into a comment nobody reads.
CHECKS: dict[str, str] = {
    "policy.load_is_idempotent": "load() twice must not raise; runners may retry.",
    "policy.returns_trajectory": "infer() must return a Trajectory, not an array or a dict.",
    "policy.enough_waypoints": (
        f"Closed-loop engines index the first {N_WAYPOINTS} waypoints positionally and "
        "raise IndexError on fewer."
    ),
    "policy.finite": "A NaN waypoint propagates into the score as a silent NaN, not an error.",
    "policy.not_a_commanded_stop": (
        "An all-zero trajectory is how the vehicle model is told to brake to a stop. A model "
        "that has not warmed up must return constant_velocity(), never zeros."
    ),
    "policy.moves_forward": (
        "At a nonzero speed the plan must advance in +x. A plan that does not is either a "
        "frame error or a model shown a static world -- comma2k19's first result was the "
        "latter and reported ADE 38.8 m as a modelling finding."
    ),
    "policy.reset_clears_state": (
        "reset() must make inference reproducible. Without it run N+1 starts with run N's "
        "feature buffer and the runs stop being independent samples."
    ),
    "policy.diagnostics_before_load": (
        "diagnostics() is served over HTTP and must answer before a frame arrives, not raise."
    ),
    "policy.declares_cameras_it_reads": (
        "Rendering only the declared cameras dominates closed-loop wall-clock, so a camera "
        "read but not declared will be absent at run time."
    ),
    "benchmark.preflight_returns_reasons": (
        "preflight() must return a list of human-readable strings, so a sweep fails in "
        "seconds with a reason rather than an hour in with a traceback."
    ),
    "benchmark.samples_are_deterministic": (
        "--limit N must select the same N samples at every site, or two people comparing "
        "smoke tests reach different conclusions."
    ),
    "benchmark.sample_shape": "Each sample needs an id, at least one observation, and ground truth.",
    "benchmark.ground_truth_finite": "A NaN in ground truth silently poisons every metric.",
    "benchmark.score_of_ground_truth": (
        "Scoring the recorded future against itself must give ~0. Anything else means the "
        "metric and the ground truth disagree about a frame, a sign or a time grid -- an "
        "error that applies to every model equally, so no comparison reveals it."
    ),
    "benchmark.score_of_a_stationary_plan": (
        "An all-zero plan must score unmistakably badly. If the metric set cannot tell "
        "'predicted nothing' from 'tracked the lane', it cannot support a conclusion. "
        "comma2k19's ADE 38.8 m / lateral_mae 0.96 m was read as the latter and was the former."
    ),
    "benchmark.declared_metrics_are_produced": (
        "benchmark.yaml's `metrics` list is what the leaderboard reads. A key in the yaml "
        "that score() never returns shows as a blank column forever."
    ),
    "benchmark.primary_metric_is_produced": "`primary_metric` is what runs are ranked by.",
    "benchmark.camera_mounting_is_plausible": (
        "camera2ego is the single easiest thing to get wrong and the hardest to notice. "
        "The identity matrix reads as 'no rotation' and is a 90 degree error."
    ),
    "benchmark.observations_carry_calibration": (
        "A model needs intrinsics and extrinsics per camera; a missing entry surfaces deep "
        "inside a feature builder as a shape error that says nothing about calibration."
    ),
    "pair.end_to_end": "One real sample through the real policy, scored -- the smoke test.",
    "pair.declared_cameras_available": (
        "The benchmark must supply every camera the model declares, or infer() fails on the "
        "first frame of a long sweep."
    ),
}


def _finding(check: str, ok: bool, detail: str = "", severity: str = "error") -> Finding:
    """One finding, with `ok` forced to a real `bool`.

    The coercion is load-bearing. Almost every check here is a numpy comparison,
    so `ok` arrives as a `np.bool_` -- and `paddock verify --model` runs these
    inside the *model's* interpreter and ships the results back as JSON, which
    refuses a `np.bool_`. numpy 2 named that scalar `bool`, so the whole run
    dies on "Object of type bool is not JSON serializable": a message that reads
    as a paddock bug about a type json handles natively, several frames away
    from the check that produced it. One coercion here rather than a `bool(...)`
    every caller has to remember.
    """
    return Finding(
        check=check, ok=bool(ok), severity=severity, detail=detail, rationale=CHECKS.get(check, "")
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def check_camera_mounting(camera2ego, name: str = "CAM_FRONT") -> Finding:
    """Is this 4x4 a plausible camera mounting, or a frame slip?

    Derives the mounting rotation the way an adapter does and rejects anything
    beyond `MAX_MOUNT_DEG`. Names the identity case explicitly, because that is
    the one people actually write.
    """
    m = np.asarray(camera2ego, dtype=np.float64)
    if m.shape != (4, 4):
        return _finding(
            "benchmark.camera_mounting_is_plausible", False, f"{name}: camera2ego is {m.shape}, want (4, 4)"
        )
    r = m[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4):
        return _finding(
            "benchmark.camera_mounting_is_plausible",
            False,
            f"{name}: camera2ego rotation is not orthonormal",
        )
    if np.allclose(r, np.eye(3), atol=1e-6):
        return _finding(
            "benchmark.camera_mounting_is_plausible",
            False,
            f"{name}: camera2ego rotation is the identity. That is not 'no mounting rotation' -- "
            "paddock's camera frame is [right, down, forward] and its ego frame is "
            "[forward, left, up], so an unrotated camera is\n"
            "             [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]\n"
            "         and the identity is a 90 degree error.",
        )
    # Residual rotation relative to an unrotated camera, as an angle.
    residual = r @ UNROTATED_CAMERA2EGO.T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0))))
    ok = angle <= MAX_MOUNT_DEG
    return _finding(
        "benchmark.camera_mounting_is_plausible",
        ok,
        f"{name}: mounting is {angle:.2f} deg off level (limit {MAX_MOUNT_DEG:g} deg)",
    )


def synthetic_observation(
    speed: float = 20.0,
    cameras: tuple[str, ...] = ("CAM_FRONT",),
    width: int = 1164,
    height: int = 874,
    focal: float = 910.0,
    timestamp_us: int = 0,
) -> Observation:
    """A structurally valid `Observation` with a plausible camera and mid-grey frames.

    For contract checks that need *an* observation but not real pixels. A model
    is free to produce nonsense from it -- the checks that use it look at shape,
    finiteness and statefulness, not accuracy.
    """
    intrinsics = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    camera2ego = np.eye(4)
    camera2ego[:3, :3] = UNROTATED_CAMERA2EGO
    camera2ego[2, 3] = 1.22
    canbus = np.zeros(18)
    canbus[13] = speed
    return Observation(
        images={c: np.full((height, width, 3), 128, dtype=np.uint8) for c in cameras},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=timestamp_us,
        command=Command.STRAIGHT,
        camera2image={c: intrinsics for c in cameras},
        camera2ego={c: camera2ego for c in cameras},
    )


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def policy_findings(
    policy: Policy, observations: list[Observation] | None = None, load: bool = True
) -> list[Finding]:
    """Check a Policy against the contract.

    `observations` lets a caller supply real frames -- a model needing warm-up
    produces a more meaningful plan from them. Without them a short synthetic
    sequence is used, and the accuracy-dependent checks downgrade to warnings
    because a model handed grey rectangles is entitled to say anything.
    """
    out: list[Finding] = []
    synthetic = observations is None
    if synthetic:
        observations = [
            synthetic_observation(cameras=tuple(policy.cameras), timestamp_us=int(i * 50_000))
            for i in range(4)
        ]

    if load:
        try:
            policy.load()
            policy.load()
            out.append(_finding("policy.load_is_idempotent", True))
        except Exception as exc:  # noqa: BLE001 -- reporting is the point
            out.append(_finding("policy.load_is_idempotent", False, f"{type(exc).__name__}: {exc}"))
            return out

    read: set[str] = set()
    last = observations[-1]
    original_image = type(last).image

    def spy(self, camera: str = "CAM_FRONT"):
        read.add(camera)
        return original_image(self, camera)

    try:
        type(last).image = spy
        try:
            traj = policy.infer_sequence(observations) if len(observations) > 1 else policy.infer(last)
        except Exception as exc:  # noqa: BLE001
            out.append(_finding("policy.returns_trajectory", False, f"{type(exc).__name__}: {exc}"))
            return out
    finally:
        type(last).image = original_image

    if not isinstance(traj, Trajectory):
        out.append(_finding("policy.returns_trajectory", False, f"returned {type(traj).__name__}"))
        return out
    out.append(_finding("policy.returns_trajectory", True))

    xy = np.asarray(traj.xy, dtype=np.float64)
    out.append(
        _finding(
            "policy.enough_waypoints",
            len(xy) >= N_WAYPOINTS,
            f"{len(xy)} waypoints, need >= {N_WAYPOINTS}",
        )
    )
    out.append(_finding("policy.finite", bool(np.isfinite(xy).all())))

    all_zero = bool(np.allclose(xy, 0.0))
    out.append(
        _finding(
            "policy.not_a_commanded_stop",
            not all_zero,
            "every waypoint is (0, 0), which a simulator reads as brake-to-stop",
        )
    )

    forward = len(xy) > 1 and bool(np.all(np.diff(xy[:, 0]) >= -0.05)) and xy[-1, 0] > 0.5
    out.append(
        _finding(
            "policy.moves_forward",
            forward,
            f"plan reaches x = {xy[-1, 0]:.2f} m at {observations[-1].speed:.1f} m/s",
            # Grey rectangles are not a road; a model may legitimately refuse.
            severity="warning" if synthetic else "error",
        )
    )

    try:
        policy.reset()
        again = policy.infer_sequence(observations) if len(observations) > 1 else policy.infer(last)
        same = np.allclose(np.asarray(again.xy, dtype=np.float64), xy, atol=1e-4)
        out.append(
            _finding(
                "policy.reset_clears_state",
                same,
                "the same input after reset() produced a different plan, so per-episode state survives reset",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("policy.reset_clears_state", False, f"{type(exc).__name__}: {exc}"))

    fresh = type(policy)()
    try:
        diag = fresh.diagnostics()
        out.append(
            _finding(
                "policy.diagnostics_before_load", isinstance(diag, dict), f"returned {type(diag).__name__}"
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("policy.diagnostics_before_load", False, f"{type(exc).__name__}: {exc}"))

    undeclared = read - set(policy.cameras)
    out.append(
        _finding(
            "policy.declares_cameras_it_reads",
            not undeclared,
            f"read {sorted(undeclared)} but declares {list(policy.cameras)}"
            if undeclared
            else f"read {sorted(read) or 'none'}",
        )
    )
    return out


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def benchmark_findings(bench, spec=None, sample_limit: int = 2) -> list[Finding]:
    """Check an open-loop Benchmark against the contract.

    Builds real samples, so it needs the dataset present. `spec` is the
    `BenchmarkSpec` whose `metrics` / `primary_metric` are cross-checked against
    what `score()` actually returns; omit it to skip that pair of checks.
    """
    out: list[Finding] = []

    try:
        problems = bench.preflight()
        ok = isinstance(problems, list) and all(isinstance(p, str) for p in problems)
        out.append(
            _finding(
                "benchmark.preflight_returns_reasons",
                ok,
                f"returned {type(problems).__name__}"
                if not ok
                else (f"{len(problems)} problem(s): {problems[0][:120]}" if problems else "ready"),
            )
        )
        if problems:
            out.append(
                Finding(
                    "benchmark.preflight_is_clean",
                    False,
                    "warning",
                    "preflight reported problems, so the sample checks below are skipped",
                    "A benchmark that is not ready cannot be conformance-checked here.",
                )
            )
            return out
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("benchmark.preflight_returns_reasons", False, f"{type(exc).__name__}: {exc}"))
        return out

    try:
        first = list(bench.samples(limit=sample_limit))
        second = list(bench.samples(limit=sample_limit))
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("benchmark.sample_shape", False, f"samples() raised {type(exc).__name__}: {exc}"))
        return out

    if not first:
        out.append(_finding("benchmark.sample_shape", False, "samples(limit=...) yielded nothing"))
        return out

    ids_match = [s.sample_id for s in first] == [s.sample_id for s in second]
    out.append(
        _finding(
            "benchmark.samples_are_deterministic",
            ids_match,
            f"first call {[s.sample_id for s in first][:2]}, second {[s.sample_id for s in second][:2]}"
            if not ids_match
            else "",
        )
    )

    sample = first[0]
    shape_ok = bool(sample.sample_id) and bool(sample.observations) and sample.ground_truth is not None
    out.append(
        _finding(
            "benchmark.sample_shape",
            shape_ok,
            f"id={sample.sample_id!r} observations={len(sample.observations)} "
            f"ground_truth={'set' if sample.ground_truth is not None else 'None'}",
        )
    )
    if not shape_ok:
        return out

    gt = np.asarray(sample.ground_truth, dtype=np.float64)
    out.append(
        _finding(
            "benchmark.ground_truth_finite", bool(np.isfinite(gt).all()), f"ground truth shape {gt.shape}"
        )
    )

    obs = sample.observations[-1]
    have_calib = bool(obs.camera2image) and bool(obs.camera2ego)
    out.append(
        _finding(
            "benchmark.observations_carry_calibration",
            have_calib,
            f"camera2image={sorted(obs.camera2image)} camera2ego={sorted(obs.camera2ego)}",
        )
    )
    for cam, m in obs.camera2ego.items():
        out.append(check_camera_mounting(m, cam))

    # --- the two that matter -------------------------------------------
    gt_xy = gt[:, :2]
    try:
        perfect = bench.score(sample, Trajectory(gt_xy, dt=0.5))
        primary = (spec.primary_metric if spec else None) or "ade"
        value = perfect.get(primary, float("nan"))
        ok = np.isfinite(value) and abs(value) < 0.05
        out.append(
            _finding(
                "benchmark.score_of_ground_truth",
                ok,
                f"scoring the recorded future against itself gives {primary} = {value:.4f}, expected ~0",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("benchmark.score_of_ground_truth", False, f"{type(exc).__name__}: {exc}"))
        perfect = {}

    try:
        stationary = bench.score(sample, Trajectory(np.zeros_like(gt_xy), dt=0.5))
        primary = (spec.primary_metric if spec else None) or "ade"
        travelled = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())
        # An all-zero plan's error is the distance travelled. Either the primary
        # metric reflects that, or some metric must flag it outright.
        flagged = (
            np.isfinite(stationary.get(primary, np.nan)) and stationary[primary] > 0.25 * max(travelled, 1.0)
        ) or (
            np.isfinite(stationary.get("path_length_ratio", np.nan)) and stationary["path_length_ratio"] < 0.1
        )
        out.append(
            _finding(
                "benchmark.score_of_a_stationary_plan",
                bool(flagged),
                f"a zero plan scores {primary} = {stationary.get(primary, float('nan')):.2f} "
                f"against {travelled:.1f} m travelled"
                + (
                    f", path_length_ratio = {stationary['path_length_ratio']:.3f}"
                    if "path_length_ratio" in stationary
                    else ""
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("benchmark.score_of_a_stationary_plan", False, f"{type(exc).__name__}: {exc}"))
        stationary = {}

    if spec is not None and perfect:
        declared = set(spec.metrics) - {"inference_seconds"}
        produced = set(perfect)
        missing = sorted(declared - produced)
        out.append(
            _finding(
                "benchmark.declared_metrics_are_produced",
                not missing,
                f"benchmark.yaml declares {missing} which score() does not return"
                if missing
                else f"{len(declared)} declared metrics all produced",
            )
        )
        has_primary = spec.primary_metric in produced
        out.append(
            _finding(
                "benchmark.primary_metric_is_produced",
                has_primary,
                "" if has_primary else f"primary_metric {spec.primary_metric!r} not in {sorted(produced)}",
            )
        )
    return out


# --------------------------------------------------------------------------
# Pair
# --------------------------------------------------------------------------


def pair_findings(bench, policy: Policy, spec=None) -> list[Finding]:
    """One real sample from `bench` through `policy`, scored. The smoke test."""
    out: list[Finding] = []
    try:
        sample = next(iter(bench.samples(limit=1)))
    except StopIteration:
        return [_finding("pair.end_to_end", False, "the benchmark yielded no samples")]
    except Exception as exc:  # noqa: BLE001
        return [_finding("pair.end_to_end", False, f"samples() raised {type(exc).__name__}: {exc}")]

    obs = sample.observations[-1]
    available = set(obs.images) | set(obs.image_paths)
    missing = sorted(set(policy.cameras) - available)
    out.append(
        _finding(
            "pair.declared_cameras_available",
            not missing,
            f"model declares {list(policy.cameras)}, sample has {sorted(available)}",
        )
    )
    if missing:
        return out

    try:
        traj = policy.infer_sequence(sample.observations)
        metrics = bench.score(sample, traj)
        primary = (spec.primary_metric if spec else None) or "ade"
        detail = ", ".join(
            f"{k}={v:.3f}" for k, v in list(metrics.items())[:6] if isinstance(v, (int, float))
        )
        out.append(
            _finding(
                "pair.end_to_end", np.isfinite(metrics.get(primary, np.nan)), f"{sample.sample_id}: {detail}"
            )
        )
        out += policy_findings(policy, observations=sample.observations, load=False)
    except Exception as exc:  # noqa: BLE001
        out.append(_finding("pair.end_to_end", False, f"{type(exc).__name__}: {exc}"))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass
class Report:
    sections: dict[str, list[Finding]] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(f.failed for fs in self.sections.values() for f in fs)

    def render(self) -> str:
        lines: list[str] = []
        n_ok = n_warn = n_fail = 0
        for title, findings in self.sections.items():
            lines.append(f"\n{title}")
            if not findings:
                lines.append("  (nothing checked)")
            for f in findings:
                lines.append(f.render())
                if f.ok:
                    n_ok += 1
                elif f.severity == "error":
                    n_fail += 1
                else:
                    n_warn += 1
        lines.append("")
        lines.append(f"{n_ok} ok, {n_warn} warning(s), {n_fail} failure(s)")
        if n_fail:
            lines.append("Adapter does NOT satisfy the contract. See `why:` on each failure.")
        return "\n".join(lines)


def render_findings(sections: dict[str, list[Finding]]) -> tuple[str, bool]:
    """(text, failed) for a set of named finding lists."""
    report = Report(sections=sections)
    return report.render(), report.failed


def explain(check: str | None = None) -> str:
    """What a check is for. With no argument, every check."""
    if check:
        return CHECKS.get(check, f"unknown check {check!r}")
    width = max(len(k) for k in CHECKS)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in CHECKS.items())
