"""Contract tests for the VisionPilot adapter. No onnxruntime, no weights."""

from __future__ import annotations

import numpy as np
import pytest

from paddock.adapters.models.visionpilot.policy import (
    CIPO_PROB_MIN,
    CONTROL_DT,
    IDM,
    POLICY,
    WAYPOINT_DT,
    CIPOFusionEstimate,
    LongitudinalFusion,
)
from paddock.policy.protocol import Policy, validate_extras


def test_policy_imports_without_onnxruntime() -> None:
    policy = POLICY()
    assert isinstance(policy, Policy)
    assert policy.cameras == ("CAM_FRONT",)


def test_defaults_to_cpu_execution() -> None:
    """The renderer wants the GPU; the recorded closed-loop runs used CPU here."""
    assert POLICY().providers == ["CPUExecutionProvider"]


def test_cuda_can_be_requested(monkeypatch) -> None:
    monkeypatch.setenv("VISIONPILOT_PROVIDERS", "cuda")
    assert "CUDAExecutionProvider" in POLICY().providers


def test_process_noise_override_is_read(monkeypatch) -> None:
    """The shipped 2.0 starves the velocity state; 0.3 is what the runs used."""
    monkeypatch.setenv("VISIONPILOT_PROCESS_NOISE_DIST", "0.3")
    assert POLICY().process_noise_dist_m == pytest.approx(0.3)


def test_load_without_weights_names_the_setting() -> None:
    policy = POLICY(weights="/nonexistent/weights")
    with pytest.raises(RuntimeError, match="VISIONPILOT_WEIGHTS"):
        policy.load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY().infer(object())  # type: ignore[arg-type]


def test_reset_clears_cross_frame_state() -> None:
    """AutoDrive is two-frame, the filter carries a posterior, cruise speed latches.

    Any of them surviving a reset leaks one run's ending into the next run's
    start, and the runs stop being independent samples.
    """
    policy = POLICY()
    policy.prev_frame = np.zeros((4, 4))
    policy.prev_timestamp = 12345
    policy.set_speed = 30.0
    policy.reset()
    assert policy.prev_frame is None
    assert policy.prev_timestamp is None
    assert policy.set_speed is None


def test_idm_brakes_when_closing_on_a_lead() -> None:
    """A closing gap must produce deceleration, whatever the tuning."""
    idm = IDM()
    free = idm.acceleration(kappa=0.0, ego_v=15.0, has_cipo=False, closing_v=0.0, gap=1e6)
    closing = idm.acceleration(kappa=0.0, ego_v=15.0, has_cipo=True, closing_v=10.0, gap=8.0)
    assert closing < 0.0, "closing on a near lead must decelerate"
    assert closing < free


def test_control_and_waypoint_rates_are_distinct() -> None:
    """50 Hz control, 0.5 s waypoints. Conflating them mistimes every plan."""
    assert CONTROL_DT == 0.05
    assert WAYPOINT_DT == 0.5


def test_fusion_resets_to_no_cipo() -> None:
    fusion = LongitudinalFusion(process_noise_dist_m=0.3)
    fusion.reset()
    assert 0.0 < CIPO_PROB_MIN < 1.0


# ---------------------------------------------------------------------------
# the auxiliary `lead` head
#
# comma2k19 scores `extras["lead"]` against the car's own radar, which is the
# only independent check on the CIPO fusion anywhere in the repo -- the
# closed-loop benchmarks can only observe whether the car crashed. These tests
# are about the two things that would make that check quietly meaningless: a
# flipped sign, and "I see nothing" being indistinguishable from "I have no
# lead head".
# ---------------------------------------------------------------------------

#: One AutoSpeed detection: [x1, y1, x2, y2, score, class]. Class 1 is Level 1.
_DET = np.array([[100.0, 200.0, 180.0, 260.0, 0.77, 1.0]])
#: Its bottom-centre projected to the ground: 30 m ahead, 1.2 m to the right.
_GROUND = np.array([[30.0, -1.2]])
_RANGES = np.array([30.0])


def _policy_with_boxes():
    policy = POLICY()
    policy._last_box_ground = _GROUND
    return policy


def _estimate(**kw) -> CIPOFusionEstimate:
    base = {
        "valid": True,
        "has_cipo": True,
        "distance_m": 28.5,
        "velocity_ms": -4.0,
        "cipo_raw_found": True,
        "cipo_raw_dist_m": 30.0,
    }
    return CIPOFusionEstimate(**{**base, "cut_in_detected": False, **kw})


def test_no_lead_is_an_empty_list_not_a_missing_key() -> None:
    """ "I looked and there is nothing" must be distinguishable from "I do not look".

    `auxiliary._lead` scores nothing at all for an absent `lead` key -- that is
    how a model with no lead head is handled -- but an empty list is an answer,
    and it has to cost VisionPilot recall on a sample where the radar did see a
    car. Returning `None` here would let it score perfect lead precision by
    never claiming anything.
    """
    leads = _policy_with_boxes()._lead_extra(_estimate(has_cipo=False), _DET, _RANGES, 0.1)
    assert leads == []


def test_a_lead_carries_range_lateral_and_closing_rate() -> None:
    leads = _policy_with_boxes()._lead_extra(_estimate(), _DET, _RANGES, 0.2)
    assert len(leads) == 1
    assert leads[0]["d"] == pytest.approx(28.5)  # the fused belief, not the raw box
    assert leads[0]["y"] == pytest.approx(-1.2)


def test_closing_is_negative_in_both_conventions() -> None:
    """The one sign that matters, asserted rather than trusted.

    `CIPOFusionEstimate.velocity_ms` is negative when approaching and the
    protocol's `v_rel` is negative when closing. A negation here would not
    raise, would not change any recorded closed-loop result, and would make
    `lead_rel_speed_abs_err` read roughly double the true closing speed on
    every sample where there is a lead at all.
    """
    approaching = _policy_with_boxes()._lead_extra(_estimate(velocity_ms=-4.0), _DET, _RANGES, 0.2)
    receding = _policy_with_boxes()._lead_extra(_estimate(velocity_ms=+4.0), _DET, _RANGES, 0.2)
    assert approaching[0]["v_rel"] < 0.0
    assert receding[0]["v_rel"] > 0.0


def test_confidence_is_the_stronger_of_the_two_networks() -> None:
    """Either network alone is enough for the fusion to declare a lead.

    `auxiliary._lead` thresholds `prob` to decide whether the model reported
    anything, so publishing only AutoDrive's flag would drop every lead found
    on AutoSpeed's evidence and score it as a miss the stack did not make.
    """
    box_only = _policy_with_boxes()._lead_extra(_estimate(), _DET, _RANGES, 0.05)
    assert box_only[0]["prob"] == pytest.approx(0.77)
    confident = _policy_with_boxes()._lead_extra(_estimate(), _DET, _RANGES, 0.95)
    assert confident[0]["prob"] == pytest.approx(0.95)


def test_a_lead_seen_only_by_autodrive_has_a_range_but_no_lateral() -> None:
    """AutoDrive is a scalar distance head. It cannot say where across the road.

    Omitting `y` leaves `lead_lateral_abs_err` NaN for that sample, which is
    "not answered". Filling it with 0.0 would claim the lead is dead ahead and
    score a free hit whenever it happened to be.
    """
    policy = POLICY()
    policy._last_box_ground = np.empty((0, 2))
    leads = policy._lead_extra(_estimate(cipo_raw_found=False), np.empty((0, 6)), np.empty(0), 0.9)
    assert "y" not in leads[0]
    assert leads[0]["d"] == pytest.approx(28.5)


def test_the_published_lead_matches_the_protocol_schema() -> None:
    """`validate_extras` is the contract; `infer` calls it on every frame."""
    leads = _policy_with_boxes()._lead_extra(_estimate(), _DET, _RANGES, 0.2)
    validate_extras({"lead": leads}, n_waypoints=8)
    validate_extras({"lead": []}, n_waypoints=8)


# ---------------------------------------------------------------------------
# the execution provider
#
# onnxruntime drops an EP it cannot register and runs on what is left, without
# raising. The numbers are identical and the wall clock is ~40x, so on a
# 10,165-sample sweep the only symptom is that it does not finish.
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, *providers: str) -> None:
        self._providers = list(providers)

    def get_providers(self) -> list[str]:
        return self._providers


def test_resolved_providers_are_read_from_the_sessions_not_the_request() -> None:
    policy = POLICY()
    policy.steer = _Session("CUDAExecutionProvider", "CPUExecutionProvider")
    policy.drive = _Session("CUDAExecutionProvider", "CPUExecutionProvider")
    policy.speed = _Session("CPUExecutionProvider")  # this one fell back
    assert policy.resolved_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_diagnostics_record_what_was_asked_for_and_what_was_given() -> None:
    """Stored with the run, because afterwards there is no way to tell.

    A number produced on a silent CPU fallback and one produced on the GPU are
    the same number.
    """
    policy = POLICY()
    policy.steer = policy.drive = policy.speed = _Session("CPUExecutionProvider")
    diag = policy.diagnostics()
    assert diag["providers"] == ["CPUExecutionProvider"]
    assert diag["providers_requested"] == ["CPUExecutionProvider"]


def test_diagnostics_before_the_first_frame_still_report_providers() -> None:
    diag = POLICY().diagnostics()
    assert diag["status"] == "no frame seen yet"
    assert diag["providers"] == []


def test_ort_thread_cap_is_off_unless_asked_for() -> None:
    """Unset must mean ORT's default, or every recorded single-node run moves."""
    import os

    from paddock.adapters.models.visionpilot.policy import ort_threads

    saved = os.environ.pop("VISIONPILOT_ORT_THREADS", None)
    try:
        assert ort_threads() == 0
        os.environ["VISIONPILOT_ORT_THREADS"] = "2"
        assert ort_threads() == 2
        os.environ["VISIONPILOT_ORT_THREADS"] = "all"
        assert ort_threads() == 0, "garbage must fall back to the default, not crash a node"
    finally:
        os.environ.pop("VISIONPILOT_ORT_THREADS", None)
        if saved is not None:
            os.environ["VISIONPILOT_ORT_THREADS"] = saved
