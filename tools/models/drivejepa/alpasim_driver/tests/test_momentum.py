# SPDX-License-Identifier: Apache-2.0
"""Paper §3.5 momentum-aware selection, on the promoted 0.1.7 driver.

Ported from the 0.1.4b suite; momentum.py has not changed since.

    switch      DRIVEJEPA_MOMENTUM is read once; a typo refuses
    bookkeeping first replan -> plain S; next replan -> previous plan fed;
                older than 0.55 s -> dropped; sessions do not share history
    stage       the recalibrated score is (14 S + 2 S_c) / 16 with S_c in {0, 1}
    provenance  the launcher prints the §3.5 line and names this directory
    debug       the per-replan payload survives the scaffold's pickle round trip

The momentum path needs nuPlan, which lives only in the Drive-JEPA environment,
so those tests talk to a ``--model canned`` bridge worker: the real worker code
(momentum.py, the upstream calibrate_score, upstream's simulator) on model
outputs the test supplies, with no checkpoint. They skip unless
DRIVEJEPA_WORKER_PYTHON and NAVSIM_DEVKIT_ROOT are set. CPU only.
"""

from __future__ import annotations

import logging
import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import pytest
from drivejepa.alpasim_driver import constants as K
from drivejepa.alpasim_driver.bridge import WORKER_PYTHON_ENV, BridgedDriveJEPA
from drivejepa.alpasim_driver.planner import DriveJEPAPlanner, momentum_mode

from .test_divider import FRONT, grid_frames, make_view, straight_poses

LIVE = pathlib.Path(__file__).resolve().parents[1]
SERVE = LIVE / "serve.py"

_WORKER_PYTHON = os.environ.get(WORKER_PYTHON_ENV, "")
_HAVE_WORKER = (
    bool(_WORKER_PYTHON) and pathlib.Path(_WORKER_PYTHON).is_file() and bool(os.environ.get("NAVSIM_DEVKIT_ROOT"))
)
needs_worker = pytest.mark.skipif(
    not _HAVE_WORKER,
    reason=f"set {WORKER_PYTHON_ENV} and NAVSIM_DEVKIT_ROOT to run the momentum stage",
)


# ---------------------------------------------------------------- synthetic menus
def menu(end_speeds, lateral: float = 0.0) -> np.ndarray:
    """32 straight proposals at constant chord speed end_speeds[i]."""
    P = np.zeros((32, 8, 3), dtype=np.float32)
    for i, v in enumerate(end_speeds):
        P[i, :, 0] = np.cumsum(np.full(8, v * 0.5))
        P[i, :, 1] = lateral * (i - 16) / 16 * np.linspace(0, 1, 8)
    return P


def canned(P: np.ndarray, pdm: np.ndarray) -> dict:
    """What 0.1.4's model returns for this menu (drive_jepa_model.py:85-94)."""
    logit = np.tile(np.log(pdm / (1 - pdm))[:, None], (1, 6))[None].astype(np.float32)
    score = (1.0 / (1.0 + np.exp(-logit[0, :, -1].astype(np.float64)))).astype(np.float32)
    return {
        "proposals": P[None],
        "pred_logit": logit,
        "pdm_score": score[None],
        "trajectory": P[None][:, int(np.argmax(score))],
    }


def view_at(now_us: int, *, speed: float = 8.0, session: str = "unit"):
    """A car at ``speed`` down +x, frames and poses on the 50 ms grid up to now."""
    frames = grid_frames(newest_us=now_us, count=16, tick_us=50_000)
    poses = straight_poses(start_us=now_us - 50_000, count=2, tick_us=50_000, speed_m_s=speed)
    # x measured from t = 0 so consecutive views describe one continuous drive.
    shift = speed * (now_us - 50_000) / 1e6
    poses = tuple((ts, x + shift, y, yaw) for ts, x, y, yaw in poses)
    return make_view(frames=frames, poses=poses, velocity=(speed, 0.0), accel=(0.0, 0.0), session_uuid=session)


@pytest.fixture(scope="module")
def worker():
    if not _HAVE_WORKER:
        pytest.skip("no Drive-JEPA worker interpreter")
    w = BridgedDriveJEPA(device="cpu", worker_args=("--model", "canned"))
    yield w
    w.close()


def through(worker, output: dict):
    """A ``model_fn``: the canned worker on ``output``'s proposals and logits."""

    def fn(features: dict) -> dict:
        fn.requests.append(features.get(K.MOMENTUM_FEATURE_KEY))
        return worker(
            {
                "canned_proposals": output["proposals"],
                "canned_pred_logit": output["pred_logit"],
                K.MOMENTUM_FEATURE_KEY: features[K.MOMENTUM_FEATURE_KEY],
            }
        )

    fn.requests = []
    return fn



def test_the_switch_defaults_to_on_and_refuses_a_typo(monkeypatch):
    monkeypatch.delenv(K.MOMENTUM_ENV, raising=False)
    assert momentum_mode("auto") == "on"
    monkeypatch.setenv(K.MOMENTUM_ENV, "OFF")
    assert momentum_mode("auto") == "off"
    assert momentum_mode("on") == "on"  # an explicit argument beats the env
    monkeypatch.setenv(K.MOMENTUM_ENV, "maybe")
    with pytest.raises(ValueError, match="DRIVEJEPA_MOMENTUM"):
        momentum_mode("auto")
    with pytest.raises(ValueError):
        DriveJEPAPlanner(model_fn=lambda f: None, front_camera_id=FRONT)


def test_the_switch_is_read_once_at_construction(monkeypatch):
    monkeypatch.setenv(K.MOMENTUM_ENV, "off")
    p = DriveJEPAPlanner(model_fn=lambda f: None, front_camera_id=FRONT)
    monkeypatch.setenv(K.MOMENTUM_ENV, "on")
    assert p._momentum_on is False



def test_momentum_on_refuses_a_model_without_the_momentum_outputs():
    out = canned(menu(np.linspace(4.0, 10.0, 32)), np.linspace(0.1, 0.9, 32))
    p = DriveJEPAPlanner(model_fn=lambda f: out, front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    with pytest.raises(RuntimeError, match="not 0.1.7"):
        p.plan(view_at(5_000_000), time_now_us=5_000_000, time_query_us=5_000_000)




# ---------------------------------------------------------------- bookkeeping
@needs_worker
def test_first_replan_is_plain_s_then_the_previous_plan_is_fed(worker, caplog):
    speed = 8.0
    P = menu(np.linspace(6.0, 10.0, 32))
    # The scorer prefers the fastest proposal by a hair; the menu is a fan of
    # constant speeds around the ego's own.
    pdm = np.linspace(0.60, 0.70, 32)
    out = canned(P, pdm)
    fn = through(worker, out)
    p = DriveJEPAPlanner(model_fn=fn, front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    caplog.set_level(logging.INFO, logger="drivejepa.alpasim_driver.planner")

    first = p.plan(view_at(5_000_000, speed=speed), time_now_us=5_000_000, time_query_us=5_000_000)
    d = first.debug
    assert d["momentum_code"] == K.MOMENTUM_NO_PREV and d["prev_plan_present"] is False
    assert d["prev_plan_age_s"] is None and d["S_c"] is None
    assert d["calibrated"] == d["S"], "no previous plan -> the model selects on plain S"
    assert d["pick_with_momentum"] == d["pick_without_momentum"] == d["chosen_index"] == 31
    assert first.positions_in_rig.tobytes() == DriveJEPAPlanner.build_plan_result(out, 5_000_000).positions_in_rig.tobytes()
    assert "momentum: NO_PREV prev_plan_present=False" in caplog.text
    assert fn.requests[0]["past_states"] is None and fn.requests[0]["past_time_us"] is None

    second = p.plan(view_at(5_500_000, speed=speed), time_now_us=5_500_000, time_query_us=5_500_000)
    d = second.debug
    assert d["momentum_code"] == K.MOMENTUM_OK and d["prev_plan_present"] is True
    assert d["prev_plan_age_s"] == pytest.approx(0.5) and d["overlap_start"] == 5
    assert fn.requests[1]["past_time_us"] == 5_000_000 and fn.requests[1]["past_states"].shape == (41, 11)
    S, Sc, cal = (np.array(d[k]) for k in ("S", "S_c", "calibrated"))
    assert set(np.unique(Sc)) <= {0.0, 1.0}
    assert np.allclose(cal, (14.0 * S + 2.0 * Sc) / 16.0, atol=1e-6)
    assert d["pick_with_momentum"] == int(np.argmax(cal)) == d["chosen_index"]
    assert d["pick_without_momentum"] == int(np.argmax(S)) == 31
    assert np.allclose(second.positions_in_rig[:, :2], P[d["chosen_index"], :, :2])
    assert d["chosen_score"] == pytest.approx(float(cal.max()))
    assert "momentum: OK prev_plan_present=True age_s=0.500 overlap=5" in caplog.text


@needs_worker
def test_a_previous_plan_older_than_0_55_s_is_dropped_and_logged(worker, caplog):
    out = canned(menu(np.linspace(6.0, 10.0, 32)), np.linspace(0.60, 0.70, 32))
    fn = through(worker, out)
    p = DriveJEPAPlanner(model_fn=fn, front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    caplog.set_level(logging.INFO, logger="drivejepa.alpasim_driver.planner")
    p.plan(view_at(5_000_000), time_now_us=5_000_000, time_query_us=5_000_000)

    late = p.plan(view_at(5_600_000), time_now_us=5_600_000, time_query_us=5_600_000)  # 0.60 s later
    d = late.debug
    assert d["momentum_code"] == K.MOMENTUM_DROPPED_AGE
    assert d["prev_plan_present"] is True, "a previous plan existed; it was dropped for its age"
    assert d["prev_plan_age_s"] == pytest.approx(0.6) and d["S_c"] is None
    assert d["calibrated"] == d["S"] and d["pick_with_momentum"] == d["pick_without_momentum"]
    assert "momentum: DROPPED_AGE prev_plan_present=True age_s=0.600" in caplog.text

    # The late replan becomes the new predecessor (run_pdm_score_sequential.py:167-168).
    edge = p.plan(view_at(6_150_000), time_now_us=6_150_000, time_query_us=6_150_000)  # 0.55 s: kept (> 0.55 drops)
    assert edge.debug["momentum_code"] == K.MOMENTUM_OK
    assert edge.debug["prev_plan_age_s"] == pytest.approx(0.55) and edge.debug["overlap_start"] == 5


@needs_worker
def test_sessions_do_not_share_a_previous_plan(worker):
    out = canned(menu(np.linspace(6.0, 10.0, 32)), np.linspace(0.60, 0.70, 32))
    p = DriveJEPAPlanner(model_fn=through(worker, out), front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    p.plan(view_at(5_000_000, session="a"), time_now_us=5_000_000, time_query_us=5_000_000)
    other = p.plan(view_at(5_500_000, session="b"), time_now_us=5_500_000, time_query_us=5_500_000)
    assert other.debug["momentum_code"] == K.MOMENTUM_NO_PREV
    same = p.plan(view_at(5_500_000, session="a"), time_now_us=5_500_000, time_query_us=5_500_000)
    assert same.debug["momentum_code"] == K.MOMENTUM_OK
    p.on_session_close("a")
    assert "a" not in p._momentum_prev and "b" in p._momentum_prev
    again = p.plan(view_at(6_000_000, session="a"), time_now_us=6_000_000, time_query_us=6_000_000)
    assert again.debug["momentum_code"] == K.MOMENTUM_NO_PREV, "a closed session starts over"
    p.on_session_start(view_at(6_000_000, session="b"))
    assert "b" not in p._momentum_prev


@needs_worker
def test_the_term_can_move_the_pick_to_the_proposal_that_matches_the_previous_plan(worker):
    """A steady 8 m/s car whose scorer now marginally prefers a hard brake.

    The previous plan was a steady 8 m/s; the scorer's new favourite (index 0,
    2 m/s) is inconsistent with it, the steady proposal is consistent and lies
    within 1/7 of the top score, so §3.5 moves the pick. That is the mechanism,
    shown end to end through the planner.
    """
    speed = 8.0
    steady = menu(np.full(32, speed))
    p = DriveJEPAPlanner(model_fn=None, front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    p._model_fn = through(worker, canned(steady, np.linspace(0.60, 0.70, 32)))
    p.plan(view_at(5_000_000, speed=speed), time_now_us=5_000_000, time_query_us=5_000_000)

    ends = np.full(32, speed)
    ends[0] = 2.0
    P = menu(ends)
    pdm = np.full(32, 0.70)
    pdm[0] = 0.75
    p._model_fn = through(worker, canned(P, pdm))
    r = p.plan(view_at(5_500_000, speed=speed), time_now_us=5_500_000, time_query_us=5_500_000)
    d = r.debug
    assert d["pick_without_momentum"] == 0
    assert d["S_c"][0] == 0.0 and sum(d["S_c"]) >= 1
    assert d["pick_with_momentum"] != 0 and d["S_c"][d["pick_with_momentum"]] == 1.0
    assert np.allclose(r.positions_in_rig[:, :2], P[d["pick_with_momentum"], :, :2])


@needs_worker
def test_the_debug_payload_carries_the_brief_fields_and_survives_the_round_trip(worker):
    egodriver = pytest.importorskip("alpasim_grpc.v0.egodriver_pb2")
    out = canned(menu(np.linspace(6.0, 10.0, 32)), np.linspace(0.60, 0.70, 32))
    p = DriveJEPAPlanner(model_fn=through(worker, out), front_camera_id=FRONT, expected_frame_interval_us=50_000, momentum="on")
    p.plan(view_at(5_000_000), time_now_us=5_000_000, time_query_us=5_000_000)
    d = p.plan(view_at(5_500_000), time_now_us=5_500_000, time_query_us=5_500_000).debug
    for key in ("prev_plan_present", "prev_plan_age_s", "S", "S_c", "calibrated", "pick_with_momentum", "pick_without_momentum",
                "momentum_code", "overlap_start", "momentum_ms", "chosen_index", "chosen_score"):
        assert key in d, key
    assert len(d["S"]) == len(d["S_c"]) == len(d["calibrated"]) == 32
    msg = egodriver.DriveResponse(debug_info=egodriver.DriveResponse.DebugInfo(unstructured_debug_info=pickle.dumps(d)))
    assert pickle.loads(egodriver.DriveResponse.FromString(msg.SerializeToString()).debug_info.unstructured_debug_info) == d


# ---------------------------------------------------------------- layout + provenance


# ---------------------------------------------------------------- provenance


def _launcher(code: str, **env) -> str:
    """Run a snippet against the launcher in a fresh interpreter; return its stdout.

    A subprocess because the proof is about import order: the launcher puts its
    own directory's parent first on sys.path, and an in-process load would
    resolve whatever this pytest process already imported.
    """
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('serve_b', {str(SERVE)!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n" + code
    )
    full_env = {k: v for k, v in os.environ.items() if k != K.MOMENTUM_ENV}
    full_env.update(env)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=full_env)
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def test_launcher_provenance_names_the_paper_section_and_the_switch():
    text = _launcher("print(mod.log_provenance())")
    assert "adapter_version=0.1.7" in text and str(LIVE / "planner.py") in text
    assert "version_id=drivejepa-v2-driver-0.1.7" in text and "replan_period_us=500000" in text
    assert "paper §3.5 momentum-aware selection (upstream calibrate_score)" in text
    assert "switch=on" in text and str(LIVE / "momentum.py") in text and "drive_jepa_model.py sha256=" in text
    off = _launcher("print(mod.log_provenance())", **{K.MOMENTUM_ENV: "off"})
    assert "switch=OFF (exact 0.1.4 selection)" in off


def test_launcher_expects_this_package_and_this_version():
    text = _launcher("print('LAUNCHER', mod.EXPECTED_VERSION, mod._LIVE_PACKAGE, mod.build_parser().parse_args([]).port)")
    line = next(row for row in text.splitlines() if row.startswith("LAUNCHER"))
    assert line.split()[1:] == ["0.1.7", str(LIVE), "9149"]


def test_launcher_refuses_an_unknown_switch_value():
    with pytest.raises(AssertionError, match="DRIVEJEPA_MOMENTUM"):
        _launcher("mod.log_provenance()", **{K.MOMENTUM_ENV: "maybe"})


def test_launcher_refuses_target_frame_spacing_instead_of_dropping_it():
    code = (
        "args = mod.build_parser().parse_args(['--target-frame-spacing-us', '500000'])\n"
        "try:\n    mod.build_driver(args)\nexcept SystemExit as exc:\n    print('REFUSED', exc)\n"
    )
    assert "REFUSED --target-frame-spacing-us is not accepted" in _launcher(code)


def test_launcher_refuses_an_image_history_too_short_for_the_context_pair():
    code = (
        "args = mod.build_parser().parse_args(['--image-history', '8'])\n"
        "try:\n    mod.build_driver(args)\nexcept SystemExit as exc:\n    print('REFUSED', exc)\n"
    )
    assert "REFUSED --image-history 8" in _launcher(code)
