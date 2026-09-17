"""Does the conformance harness actually catch the bugs it claims to?

A contract checker that passes everything is worse than none: it manufactures
confidence. So each test below builds an adapter with **one** deliberate defect
-- each taken from a defect that really happened in this repo -- and asserts
that the corresponding check fails and the others do not.

The reference implementations at the top are deliberately correct and boring, so
a failure in these tests points at the checker rather than at them.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from paddock import conformance as C
from paddock.metrics import openloop
from paddock.policy.protocol import Observation, Policy, Trajectory, constant_velocity

# --------------------------------------------------------------------------
# Correct reference adapters
# --------------------------------------------------------------------------


class GoodPolicy(Policy):
    cameras = ("CAM_FRONT",)
    waypoint_dt = 0.5

    def __init__(self) -> None:
        self.loaded = False
        self.seen = 0

    def load(self) -> None:
        self.loaded = True

    def reset(self) -> None:
        self.seen = 0

    def infer(self, obs: Observation) -> Trajectory:
        obs.image("CAM_FRONT")  # declared, so reading it is fine
        self.seen += 1
        return constant_velocity(obs.speed, n=8, dt=self.waypoint_dt)

    def diagnostics(self) -> dict:
        return {"frames_seen": self.seen}


def _straight_gt(n: int = 10, speed: float = 20.0, dt: float = 0.5) -> np.ndarray:
    """A straight-ahead recorded future, x forward and y left."""
    x = speed * dt * np.arange(1, n + 1)
    return np.stack([x, np.zeros(n)], axis=1)


class _Spec:
    name = "fake"
    dataset = "fake"
    metrics = ["ade", "fde@3s", "cross_track_mae", "path_length_ratio", "inference_seconds"]
    primary_metric = "ade"


class GoodBenchmark:
    """The minimum an open-loop benchmark has to be, without paddock's base class.

    Written against the duck type `conformance` uses, so these tests do not need
    a SiteConfig or a dataset on disk.
    """

    def __init__(self, n_samples: int = 3) -> None:
        self.n_samples = n_samples

    def preflight(self) -> list[str]:
        return []

    def _sample(self, i: int):
        from paddock.benchmark import Sample

        return Sample(
            sample_id=f"fake/{i}",
            observations=[C.synthetic_observation(timestamp_us=int(k * 50_000)) for k in range(3)],
            ground_truth=_straight_gt(),
            split="test",
        )

    def samples(self, limit: int | None = None, split: str | None = None):
        for i in range(self.n_samples if limit is None else min(limit, self.n_samples)):
            yield self._sample(i)

    def score(self, sample, prediction: Trajectory) -> dict[str, float]:
        gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
        return openloop.decompose(prediction.xy, gt, interval_s=0.5, eval_horizon_s=3.0)


def _by_check(findings: list[C.Finding]) -> dict[str, C.Finding]:
    return {f.check: f for f in findings}


def _drift_of(finding: C.Finding) -> float:
    """The metres out of a `reset_clears_state` row, read back from its detail.

    Read rather than recomputed on purpose: the number a reader sees is the one
    under test, and a detail that stopped carrying it would still pass an
    assertion made against a separately computed value.
    """
    match = re.search(r"([\d.]+e[-+]\d+) m", finding.detail)
    assert match, f"no magnitude in {finding.detail!r}"
    return float(match.group(1))


# --------------------------------------------------------------------------
# The correct adapters must pass
# --------------------------------------------------------------------------


def test_a_correct_policy_passes():
    findings = C.policy_findings(GoodPolicy())
    failures = [f.check for f in findings if f.failed]
    assert not failures, failures


def test_a_correct_benchmark_passes():
    findings = C.benchmark_findings(GoodBenchmark(), _Spec())
    failures = [f.check for f in findings if f.failed]
    assert not failures, failures


def test_a_correct_pair_passes():
    policy = GoodPolicy()
    policy.load()
    findings = C.pair_findings(GoodBenchmark(), policy, _Spec())
    failures = [f.check for f in findings if f.failed]
    assert not failures, failures


# --------------------------------------------------------------------------
# Policy defects
# --------------------------------------------------------------------------


def test_it_catches_a_plan_of_zeros():
    """The commanded-emergency-stop trap. A real closed-loop scoring hazard."""

    class Zeros(GoodPolicy):
        def infer(self, obs):
            return Trajectory(np.zeros((8, 2)), dt=0.5)

    got = _by_check(C.policy_findings(Zeros()))
    assert got["policy.not_a_commanded_stop"].failed
    assert not got["policy.finite"].failed


def test_it_catches_too_few_waypoints():
    class Short(GoodPolicy):
        def infer(self, obs):
            return Trajectory(np.stack([np.arange(1, 4) * 5.0, np.zeros(3)], axis=1), dt=0.5)

    got = _by_check(C.policy_findings(Short()))
    assert got["policy.enough_waypoints"].failed


def test_it_catches_a_nan_waypoint():
    class Nan(GoodPolicy):
        def infer(self, obs):
            xy = constant_velocity(obs.speed, n=8, dt=0.5).waypoints
            # Trajectory's own __post_init__ rejects NaN, so bypass it the way a
            # buggy adapter that builds the object itself would not -- what is
            # being tested is the checker, not the dataclass.
            traj = Trajectory(xy, dt=0.5)
            traj.waypoints = np.array(xy, dtype=np.float64)
            traj.waypoints[3, 1] = np.nan
            return traj

    got = _by_check(C.policy_findings(Nan()))
    assert got["policy.finite"].failed


def test_it_catches_state_surviving_reset():
    """KITScenes reused the first scene's camera adapter for every later scene."""

    class Leaky(GoodPolicy):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        def reset(self) -> None:
            pass  # forgets to clear

        def infer(self, obs):
            self._calls += 1
            return constant_velocity(obs.speed + 0.5 * self._calls, n=8, dt=0.5)

    got = _by_check(C.policy_findings(Leaky()))
    assert got["policy.reset_clears_state"].failed
    # The magnitude belongs in the row: it is the only thing that separates a
    # leaked buffer from float noise, and a reader of a warning row has nothing
    # else to judge on. A leak is metres.
    assert _drift_of(got["policy.reset_clears_state"]) > 0.1


def test_a_declared_stochastic_model_is_not_failed_for_drifting_but_is_still_measured():
    """`stochastic: true` is a claim about the model, so the check answers accordingly.

    qwendrive is the case: seeded sampler, greedy decode that emits the same
    sentence 5/5 times, and still one rollout in five landing 0.64 m away at the
    end of a 5 s plan -- one bf16 step at 90 m, from a reduction that split
    differently. No bf16 GPU model can meet `RESET_DRIFT_M`, and loosening the
    tolerance to let it through would have blinded the check for every
    deterministic model instead. So the declaration moves the severity and
    leaves the number, which is what a reviewer actually reads.
    """

    class Drifty(GoodPolicy):
        """Reproduces its plan except for one bf16-sized step at the far end."""

        def __init__(self) -> None:
            super().__init__()
            self._rollouts = 0

        def infer_sequence(self, observations):
            self._rollouts += 1
            return super().infer_sequence(observations)

        def infer(self, obs):
            traj = constant_velocity(obs.speed, n=8, dt=0.5)
            traj.xy[-1, 0] += 0.64 * (self._rollouts == 2)  # only the second rollout
            return traj

    got = _by_check(C.policy_findings(Drifty(), stochastic=True))["policy.reset_clears_state"]
    assert not got.ok, "a downgrade is not a pass; the drift still happened"
    assert not got.failed, "a model that declared itself a draw must not fail for being one"
    assert got.severity == "warning"
    assert _drift_of(got) == pytest.approx(0.64)

    # Same policy, no declaration: the drift is a defect again.
    assert _by_check(C.policy_findings(Drifty()))["policy.reset_clears_state"].failed


def test_a_passing_row_does_not_print_a_sentence_about_failing():
    """`render()` prints `detail` under `[ok]` as well, so it cannot be the verdict.

    Both of these read as bug reports when they were phrased as one -- `[ok]
    policy.reset_clears_state / the same input after reset() produced a
    different plan` was printed, in those words, by a model that had just
    reproduced its plan exactly. The verdict is the mark; `detail` is the
    measurement that earned it.
    """
    rendered = "\n".join(f.render() for f in C.policy_findings(GoodPolicy()) if f.ok)
    for phrase in ("produced a different plan", "which a simulator reads as brake-to-stop"):
        assert phrase not in rendered


def test_it_catches_an_undeclared_camera_read():
    class Sneaky(GoodPolicy):
        cameras = ("CAM_FRONT",)

        def infer(self, obs):
            obs.image("CAM_FRONT")
            obs.images.setdefault("CAM_BACK", np.zeros((4, 4, 3), np.uint8))
            obs.image("CAM_BACK")  # not declared
            return constant_velocity(obs.speed, n=8, dt=0.5)

    got = _by_check(C.policy_findings(Sneaky()))
    assert got["policy.declares_cameras_it_reads"].failed


def test_it_catches_diagnostics_that_raise():
    class Raises(GoodPolicy):
        def diagnostics(self) -> dict:
            raise RuntimeError("no camera yet")

    got = _by_check(C.policy_findings(Raises()))
    assert got["policy.diagnostics_before_load"].failed


# --------------------------------------------------------------------------
# Benchmark defects
# --------------------------------------------------------------------------


def test_it_catches_a_metric_that_disagrees_with_its_own_ground_truth():
    """The check that would have caught a frame or sign error in the metric.

    Here `score` mirrors y before comparing, which is what a left/right mix-up
    looks like. Scoring the ground truth against itself then does not give zero.
    """

    class Mirrored(GoodBenchmark):
        def score(self, sample, prediction):
            gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2].copy()
            gt[:, 1] *= -1
            gt[:, 1] += 3.0  # make the mirror visible on a straight path
            return openloop.decompose(prediction.xy, gt, 0.5, 3.0)

    got = _by_check(C.benchmark_findings(Mirrored(), _Spec()))
    assert got["benchmark.score_of_ground_truth"].failed


def test_it_catches_a_metric_set_blind_to_a_stationary_plan():
    """comma2k19's original failure, as a contract violation.

    A benchmark reporting only a lateral error cannot tell "predicted nothing"
    from "tracked the lane", because a zero plan sits at the origin and a
    straight recorded path has almost no lateral excursion.
    """

    class LateralOnly(GoodBenchmark):
        def score(self, sample, prediction):
            gt = np.asarray(sample.ground_truth, dtype=np.float64)[:, :2]
            n = min(len(prediction.xy), len(gt))
            return {"ade": float(np.abs(prediction.xy[:n, 1] - gt[:n, 1]).mean())}

    spec = _Spec()
    spec.metrics = ["ade"]
    got = _by_check(C.benchmark_findings(LateralOnly(), spec))
    assert got["benchmark.score_of_a_stationary_plan"].failed
    # ...and the self-consistency check still passes, so the two are independent.
    assert not got["benchmark.score_of_ground_truth"].failed


def test_it_catches_nondeterministic_sample_selection():
    class Shuffled(GoodBenchmark):
        def __init__(self) -> None:
            super().__init__()
            self._n = 0

        def samples(self, limit=None, split=None):
            self._n += 1
            order = range(3) if self._n % 2 else reversed(range(3))
            for i in order:
                yield self._sample(i)

    got = _by_check(C.benchmark_findings(Shuffled(), _Spec()))
    assert got["benchmark.samples_are_deterministic"].failed


def test_it_catches_a_declared_metric_that_is_never_produced():
    spec = _Spec()
    spec.metrics = ["ade", "a_metric_nobody_computes"]
    got = _by_check(C.benchmark_findings(GoodBenchmark(), spec))
    assert got["benchmark.declared_metrics_are_produced"].failed
    assert "a_metric_nobody_computes" in got["benchmark.declared_metrics_are_produced"].detail


def test_it_catches_preflight_that_returns_the_wrong_type():
    class Bad(GoodBenchmark):
        def preflight(self):
            return "not ready"

    got = _by_check(C.benchmark_findings(Bad(), _Spec()))
    assert got["benchmark.preflight_returns_reasons"].failed


# --------------------------------------------------------------------------
# Camera mounting -- the comma2k19 defect
# --------------------------------------------------------------------------


def test_it_catches_the_identity_camera2ego():
    """`CAMERA2EGO = np.eye(4)`: the one line that produced ADE 38.8 m."""
    finding = C.check_camera_mounting(np.eye(4))
    assert finding.failed
    assert "identity" in finding.detail
    assert "[[0, 0, 1], [-1, 0, 0], [0, -1, 0]]" in finding.detail


def test_it_accepts_a_real_mounting_and_rejects_a_ninety_degree_one():
    good = np.eye(4)
    good[:3, :3] = C.UNROTATED_CAMERA2EGO
    assert not C.check_camera_mounting(good).failed

    # 3.5 deg of pitch, as measured on comma2k19's RAV4 routes.
    pitch = np.radians(3.5)
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    mounted = np.eye(4)
    mounted[:3, :3] = ry @ C.UNROTATED_CAMERA2EGO
    assert not C.check_camera_mounting(mounted).failed

    # A roll of 90 deg about the ego forward axis is not a mounting: the horizon
    # runs down the image.
    rx = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    slipped = np.eye(4)
    slipped[:3, :3] = rx @ C.UNROTATED_CAMERA2EGO
    assert C.check_camera_mounting(slipped).failed


def _facing(degrees: float) -> np.ndarray:
    """A correctly mounted, level camera pointing `degrees` off straight ahead."""
    t = np.radians(degrees)
    rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = rz @ C.UNROTATED_CAMERA2EGO
    return m


@pytest.mark.parametrize("bearing", [30.0, 55.0, 70.0, 90.0, 110.0, 120.0, 180.0, -70.0])
def test_a_camera_that_faces_sideways_is_a_rig_choice_and_not_a_slip(bearing):
    """Every one of these failed while the residual still counted yaw.

    nuScenes hands out a six-camera ring -- roughly 55, 70, 110 and 180 degrees
    off straight ahead -- and `policy_findings` checks the mounting of every
    camera the observation carries. A check that calls five of six correct
    mountings an error is one people learn to ignore, which costs the identity
    case it exists to catch. What no real bracket does is tilt or roll, so that
    is what the angle now measures.
    """
    finding = C.check_camera_mounting(_facing(bearing), "CAM_BACK_LEFT")
    assert not finding.failed
    assert "0.00 deg off level" in finding.detail
    # The bearing is still on the record, so a camera named CAM_FRONT facing
    # sideways is visible to whoever reads the report.
    assert f"{bearing:+.0f} deg" in finding.detail


def test_a_sideways_camera_that_is_also_tilted_still_fails():
    """Dividing yaw out must not divide the pitch out with it."""
    yaw, pitch = np.radians(70.0), np.radians(40.0)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    tilted = np.eye(4)
    tilted[:3, :3] = rz @ ry @ C.UNROTATED_CAMERA2EGO
    finding = C.check_camera_mounting(tilted, "CAM_FRONT_LEFT")
    assert finding.failed
    assert "40.00 deg off level" in finding.detail


def test_it_catches_a_non_rotation_matrix():
    scaled = np.eye(4)
    scaled[:3, :3] = 2.0 * C.UNROTATED_CAMERA2EGO
    assert C.check_camera_mounting(scaled).failed


def test_a_benchmark_handing_out_the_identity_mounting_fails():
    """End to end: the defect reaches the report through the real code path."""

    class IdentityMount(GoodBenchmark):
        def _sample(self, i):
            sample = super()._sample(i)
            for obs in sample.observations:
                obs.camera2ego["CAM_FRONT"] = np.eye(4)
            return sample

    got = _by_check(C.benchmark_findings(IdentityMount(), _Spec()))
    assert got["benchmark.camera_mounting_is_plausible"].failed


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_report_renders_and_reports_failure():
    findings = C.policy_findings(GoodPolicy())
    text, failed = C.render_findings({"policy good": findings})
    assert not failed
    assert "0 failure(s)" in text

    class Zeros(GoodPolicy):
        def infer(self, obs):
            return Trajectory(np.zeros((8, 2)), dt=0.5)

    text, failed = C.render_findings({"policy zeros": C.policy_findings(Zeros())})
    assert failed
    assert "FAIL" in text
    assert "why:" in text  # every failure explains itself


def test_every_check_has_a_rationale():
    """A failure a reader cannot act on is not much of a failure."""
    emitted = {f.check for f in C.policy_findings(GoodPolicy())}
    emitted |= {f.check for f in C.benchmark_findings(GoodBenchmark(), _Spec())}
    missing = [c for c in emitted if not C.CHECKS.get(c)]
    assert not missing, f"checks with no entry in CHECKS: {missing}"


def test_explain_covers_every_check():
    text = C.explain()
    for name in C.CHECKS:
        assert name in text
    assert C.explain("policy.moves_forward").startswith("At a nonzero speed")


@pytest.mark.parametrize("bad", [np.eye(3), np.zeros((4, 4))])
def test_malformed_camera2ego_is_reported_not_raised(bad):
    assert C.check_camera_mounting(bad).failed
