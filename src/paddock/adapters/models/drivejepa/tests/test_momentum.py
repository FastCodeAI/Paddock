"""The previous-plan bookkeeping behind DRIVEJEPA_MOMENTUM. No GPU, no dataset.

All but one of these pin the book the node keeps -- which plan is handed back,
when none is -- against a stand-in for upstream's functions, and run in
paddock's own environment.

The exception is `test_the_recorded_drive_is_reproduced_pick_for_pick`, which
runs the real thing: `momentum.Upstream` and the vendored `calibrate_score`
over one drive's recorded menus. Those need nuPlan and the vendored NAVSIM, so it skips
under `make check` and has to be run with Drive-JEPA's interpreter:

    <drive-jepa env>/bin/python -m pytest \
        src/paddock/adapters/models/drivejepa/tests/test_momentum.py -rs

with ``<Drive-JEPA>/navsim_v2`` on ``PYTHONPATH`` and ``CUDA_VISIBLE_DEVICES=""``
(it needs no GPU). The recording is described in ``fixtures/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paddock.adapters.models.drivejepa import momentum
from paddock.adapters.models.drivejepa.policy import POLICY
from paddock.policy.protocol import CANBUS_ACCEL, CANBUS_VELOCITY, Command, Observation

RECORDED_DRIVE = Path(__file__).parent / "fixtures" / "momentum_replay_2e1ce314.npz"


class _FakeUpstream:
    """Stands in for `momentum.Upstream`: an ego state that carries its time,
    and "simulated states" that carry which plan they came from."""

    def ego_state(self, pose, velocity, acceleration, time_us):
        return SimpleNamespace(pose=pose, velocity=velocity, acceleration=acceleration, time_us=time_us)

    def simulate_chosen(self, poses, ego_state):
        return np.full((41, 11), float(poses[0, 0]))


def _replan(book: momentum.Momentum, t_s: float, plan_id: float = 0.0) -> dict:
    """One replan at *t_s*: build the keys, then remember a plan tagged *plan_id*."""
    time_us = int(round(t_s * 1e6))
    extra, ego_state = book.features(np.zeros(3), np.zeros(2), np.zeros(2), time_us)
    book.remember(np.full((8, 3), plan_id), ego_state, time_us)
    return extra


def _fed(extra: dict) -> float | None:
    states = extra["past_ego_simulated_states"]
    return None if states is None else float(states[0, 0])


def test_the_three_keys_are_upstreams_and_the_first_replan_has_no_past() -> None:
    """The gate at drive_jepa_model.py:87 is the key's presence *and* non-None value."""
    extra = _replan(momentum.Momentum(_FakeUpstream()), 0.0)
    assert list(extra) == ["past_ego_simulated_states", "metric_cache", "observation_interval"]
    assert extra["past_ego_simulated_states"] is None
    assert extra["observation_interval"] is None
    assert extra["metric_cache"].ego_state.time_us == 0


def test_a_replan_half_a_second_later_is_handed_the_previous_pick() -> None:
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, 10.0, plan_id=7.0)
    extra = _replan(book, 10.5, plan_id=8.0)
    assert _fed(extra) == 7.0
    assert extra["observation_interval"] == pytest.approx(0.5)
    assert _fed(_replan(book, 11.0)) == 8.0, "each replan's predecessor, not the first plan"


def test_a_previous_plan_older_than_the_cutoff_is_dropped() -> None:
    """run_pdm_score_sequential.py:131 -- beyond 0.55 s the model selects on the plain score."""
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, 0.0)
    assert _fed(_replan(book, 0.55)) is not None, "0.55 s is still fed (upstream tests '>')"
    assert _fed(_replan(book, 1.2)) is None
    assert book.counts["out_of_window"] == 1


def test_at_ten_hertz_the_plan_half_a_second_ago_is_the_one_fed() -> None:
    """A simulator stepping faster than upstream's loop must not shrink the interval.

    The stage's overlap offset is int(interval / 0.1); fed the previous *call*
    at 20 Hz it would be 0 and the comparison window empty.
    """
    book = momentum.Momentum(_FakeUpstream())
    for tick in range(10):
        extra = _replan(book, tick / 10, plan_id=float(tick))
    assert _fed(extra) == 4.0, "tick 9 is compared with the plan of tick 4"
    assert extra["observation_interval"] == pytest.approx(0.5)


def test_a_plan_younger_than_the_window_is_not_fed() -> None:
    """The first replans of an episode at 10 Hz have only young plans to hand.

    The stage floors the interval to an overlap offset: int(0.3 / 0.1) is 2, so
    a 0.3 s-old plan would be compared 0.1 s out of step. Those replans select
    on the plain score instead.
    """
    book = momentum.Momentum(_FakeUpstream())
    fed = [_fed(_replan(book, tick / 10, plan_id=float(tick))) for tick in range(7)]
    assert fed == [None, None, None, None, None, 0.0, 1.0]
    assert book.counts["no_previous"] == 1 and book.counts["out_of_window"] == 4


@pytest.mark.parametrize("age_us", [450_000, 480_000, 499_999, 500_000, 520_000, 550_000])
def test_every_plan_that_is_fed_is_read_at_offset_five(age_us: int) -> None:
    """A clock that jitters around 0.5 s must not move the stage's overlap offset.

    `calibrate_score` derives it as int(observation_interval / 0.1), written out
    here because the stage does that arithmetic itself, on the float it is given.
    """
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, 0.0, plan_id=3.0)
    extra, _ = book.features(np.zeros(3), np.zeros(2), np.zeros(2), age_us)
    assert _fed(extra) == 3.0
    assert int(extra["observation_interval"] / 0.1) == 5


def test_a_plan_just_outside_the_window_is_not_fed() -> None:
    for age_us in (449_999, 550_001):
        book = momentum.Momentum(_FakeUpstream())
        _replan(book, 0.0)
        extra, _ = book.features(np.zeros(3), np.zeros(2), np.zeros(2), age_us)
        assert _fed(extra) is None, age_us


def test_reset_forgets_the_previous_plan_and_keeps_the_counts() -> None:
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, 0.0)
    book.reset()
    assert _fed(_replan(book, 0.5)) is None
    assert book.counts["replans"] == 2 and book.counts["fed"] == 0 and book.counts["no_previous"] == 2


def test_a_clock_that_restarts_does_not_feed_another_episodes_plan() -> None:
    """A simulator that begins a new rollout without /reset restarts its clock."""
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, 100.0, plan_id=1.0)
    assert _fed(_replan(book, 0.0, plan_id=2.0)) is None
    assert _fed(_replan(book, 0.5)) == 2.0


def _observation(t_s: float, speed: float = 3.0, accel: float = -1.0) -> Observation:
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY], canbus[CANBUS_ACCEL] = speed, accel
    ego2world = np.eye(4)
    yaw = np.pi / 2
    ego2world[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    ego2world[:3, 3] = [12.0, -4.0, 0.0]
    return Observation(
        images={},
        ego2world=ego2world,
        canbus=canbus,
        timestamp_us=int(t_s * 1e6),
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
        lidar2ego=np.eye(4),
    )


def test_the_stage_is_given_the_ego_status_the_model_is_given() -> None:
    """Pose from ego2world; speed and acceleration as `_agent_input` writes them."""
    pose, velocity, acceleration, time_us = POLICY._ego_now(_observation(2.5))
    np.testing.assert_allclose(pose, [12.0, -4.0, np.pi / 2])
    np.testing.assert_allclose(velocity, [3.0, 0.0])
    np.testing.assert_allclose(acceleration, [-1.0, 0.0])
    assert time_us == 2_500_000


def test_momentum_is_off_unless_asked_for(monkeypatch) -> None:
    """Every published drivejepa row took the raw argmax; the default must stay that."""
    monkeypatch.delenv(momentum.SWITCH_ENV, raising=False)
    policy = POLICY(checkpoint="/x.ckpt")
    assert policy.momentum is None
    assert policy.diagnostics()["momentum"] == "off"


def test_an_unknown_momentum_switch_is_refused(monkeypatch) -> None:
    monkeypatch.setenv(momentum.SWITCH_ENV, "yes")
    with pytest.raises(ValueError, match=momentum.SWITCH_ENV):
        POLICY(checkpoint="/x.ckpt")


def test_the_switch_and_its_counts_reach_the_run_record(monkeypatch) -> None:
    """It changes which proposal is driven, so `provenance.policy` records it --
    with how often the stage really had a previous plan to compare against."""
    monkeypatch.setenv(momentum.SWITCH_ENV, "on")
    policy = POLICY(checkpoint="/x.ckpt")
    policy.momentum.upstream = _FakeUpstream()
    assert policy.diagnostics()["momentum"] == "on", "before any frame"
    _replan(policy.momentum, 0.0)
    _replan(policy.momentum, 0.5)
    policy.observe(_observation(0.5))
    report = policy.diagnostics()
    assert report["momentum"] == "on"
    assert report["momentum_counts"]["replans"] == 2 and report["momentum_counts"]["fed"] == 1


def test_policy_reset_forgets_the_previous_plan(monkeypatch) -> None:
    """`infer_sequence` resets first, so an open-loop sequence never sees a past plan."""
    monkeypatch.setenv(momentum.SWITCH_ENV, "on")
    policy = POLICY(checkpoint="/x.ckpt")
    policy.momentum.upstream = _FakeUpstream()
    _replan(policy.momentum, 0.0)
    policy.reset()
    assert _fed(_replan(policy.momentum, 0.5)) is None


def test_the_recorded_drive_is_one_the_bookkeeping_can_replay() -> None:
    """The recording's shape, checked where nuPlan is absent.

    Twenty-seven replans 0.5 s apart, 32 proposals each. The first had no
    previous plan, so the driver's pick there is the raw argmax of the learned
    score -- sigmoid of the last logit, drive_jepa_model.py:85 -- and its
    recorded score is that score untouched.
    """
    drive = np.load(RECORDED_DRIVE)
    assert drive["proposals"].shape == (27, 32, 8, 3) and drive["pred_logit"].shape == (27, 32, 6)
    assert set(np.diff(drive["time_us"]).tolist()) == {500_000}
    learned = 1.0 / (1.0 + np.exp(-drive["pred_logit"][0, :, -1].astype(np.float64)))
    assert int(drive["pick"][0]) == int(np.argmax(learned))
    np.testing.assert_allclose(drive["calibrated"][0], learned, atol=1e-6)


def test_the_recorded_drive_is_reproduced_pick_for_pick() -> None:
    """Upstream's own stage, fed by this bookkeeping, picks what the recorded driver picked.

    The recording is 27 consecutive menus replayed through the AlpaSim driver
    that first carried the stage; its picks were produced by the vendored
    `calibrate_score` and reproduced by a standalone script that shares no code
    with either adapter (fixtures/README.md). The agent is canned -- drive_jepa_model.py:85-91
    on the recorded proposals and logits -- so this covers `Momentum`,
    `Upstream` and the stage, and not the forward pass that produced the menus.
    """
    pytest.importorskip("nuplan", reason="needs Drive-JEPA's environment (nuPlan)")
    model = pytest.importorskip(
        "navsim.agents.drive_jepa_perception_based.drive_jepa_model",
        reason="needs Drive-JEPA's environment (vendored NAVSIM on PYTHONPATH)",
    )
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    drive = np.load(RECORDED_DRIVE)
    # drive_jepa_perception_based_agent.yaml: eight poses at 0.5 s.
    book = momentum.Momentum(momentum.Upstream(TrajectorySampling(time_horizon=4, interval_length=0.5)))
    changed = 0
    for i, time_us in enumerate(drive["time_us"].tolist()):
        extra, ego_state = book.features(
            drive["ego_pose"][i], drive["ego_velocity"][i], drive["ego_acceleration"][i], time_us
        )
        proposals = torch.from_numpy(drive["proposals"][i][None])
        score = torch.sigmoid(torch.from_numpy(drive["pred_logit"][i][None]))[:, :, -1]  # :85
        plain = int(torch.argmax(score, dim=1)[0])
        if extra["past_ego_simulated_states"] is not None:  # :87
            score = model.DriveJEPAModel.calibrate_score(None, extra, proposals, score)  # :88
        pick = int(torch.argmax(score, dim=1)[0])  # :90
        assert pick == int(drive["pick"][i]), f"replan {i}"
        np.testing.assert_array_equal(score.numpy()[0], drive["calibrated"][i], err_msg=f"replan {i}")
        changed += int(pick != plain)
        book.remember(proposals[0, pick].numpy(), ego_state, time_us, changed=pick != plain)

    assert book.counts["fed"] == 26, "every replan after the first had a 0.5 s-old plan"
    assert changed == 19, "the stage must be doing something on this drive, not agreeing by default"
