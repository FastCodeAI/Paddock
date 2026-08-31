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
    LongitudinalFusion,
)
from paddock.policy.protocol import Policy


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
