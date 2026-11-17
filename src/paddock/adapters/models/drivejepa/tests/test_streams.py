"""Momentum per recorded drive: the books navhard's streamed mode relies on. No GPU, no nuPlan.

An open-loop sample resets the node, so the episode's book never holds a plan
there. In streamed mode each real frame names its log (`protocol.StreamFrame`)
and `Momentum` keeps one book per log across those resets. These pin that the
books are keyed, that a frame sent without a stream never reaches one, that the
0.45-0.55 s rule holds per book, and that the policy hands the stage the
stream's own pose and clock.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from paddock.adapters.models.drivejepa import momentum
from paddock.adapters.models.drivejepa.policy import POLICY, DriveJepaPolicy
from paddock.adapters.models.drivejepa.tests.test_momentum import _FakeUpstream, _observation
from paddock.policy.protocol import StreamFrame, Trajectory

#: Log clocks are absolute microseconds; two of one day overlap.
T0 = 1_624_895_254_000_000


def _replan(book: momentum.Momentum, time_us: int, stream: str | None, plan_id: float) -> float | None:
    """One replan in *stream*: what it was fed (the plan id it was handed), then remember plan *plan_id*."""
    extra, ego_state = book.features(np.zeros(3), np.zeros(2), np.zeros(2), time_us, stream=stream)
    book.remember(np.full((8, 3), plan_id), ego_state, time_us, stream=stream)
    states = extra["past_ego_simulated_states"]
    return None if states is None else float(states[0, 0])


def test_interleaved_logs_on_one_node_never_cross() -> None:
    """Two scorer workers on one node interleave two logs whose clocks overlap.

    log_b's frame at T0 is exactly 0.5 s older than log_a's at T0 + 0.5 s; a
    shared book would hand it over.
    """
    book = momentum.Momentum(_FakeUpstream())
    assert _replan(book, T0, "log_b", 1.0) is None
    assert _replan(book, T0 + 500_000, "log_a", 2.0) is None, "log_b's plan is 0.5 s older, and not log_a's"
    assert _replan(book, T0 + 500_000, "log_b", 3.0) == 1.0
    assert _replan(book, T0 + 1_000_000, "log_a", 4.0) == 2.0
    assert _replan(book, T0 + 999_500, "log_b", 5.0) == 3.0, "0.4995 s, the gap navhard's pairs actually have"
    assert book.counts["fed"] == 3 and book.counts["no_previous"] == 2


def test_stream_books_survive_the_per_sample_reset_and_the_episodes_does_not() -> None:
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, T0, "log_a", 7.0)
    _replan(book, T0, None, 8.0)
    book.reset()  # what every /infer_sequence does first
    assert _replan(book, T0 + 500_000, None, 9.0) is None
    assert _replan(book, T0 + 500_000, "log_a", 10.0) == 7.0


def test_a_frame_sent_without_a_stream_never_enters_a_streams_book() -> None:
    """A synthetic stage-two scene goes as a standard sample: the episode's book,
    emptied by its own reset. It shares the log's clock, so had it been kept
    under the log it would read as 0.5 s before the next real frame."""
    book = momentum.Momentum(_FakeUpstream())
    assert _replan(book, T0, "log_a", 1.0) is None
    book.reset()
    assert _replan(book, T0 + 4_000_000, None, 99.0) is None  # synthetic, 4 s on
    book.reset()
    assert _replan(book, T0 + 4_500_000, "log_a", 2.0) is None, "the synthetic plan is not log_a's"
    book.reset()
    assert _replan(book, T0 + 5_000_000, "log_a", 3.0) == 2.0


@pytest.mark.parametrize(
    ("age_us", "fed"),
    [(449_999, False), (450_000, True), (499_500, True), (500_000, True), (550_000, True), (550_001, False)],
)
def test_the_window_is_the_same_in_a_stream(age_us: int, fed: bool) -> None:
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, T0, "log_a", 3.0)
    extra, _ = book.features(np.zeros(3), np.zeros(2), np.zeros(2), T0 + age_us, stream="log_a")
    assert (extra["past_ego_simulated_states"] is not None) is fed
    if fed:
        assert int(extra["observation_interval"] / 0.1) == 5


def test_a_frame_planned_twice_hands_on_its_second_plan() -> None:
    """The proxy retries a request whose answer it lost; the scorer scored the retry's plan."""
    book = momentum.Momentum(_FakeUpstream())
    _replan(book, T0, "log_a", 1.0)
    _replan(book, T0, "log_a", 2.0)
    assert _replan(book, T0 + 500_000, "log_a", 3.0) == 2.0


# -- the policy ------------------------------------------------------------------


def _sample(t0_s: float = 0.0) -> list:
    """Four history frames as the NAVSIM proxy sends them: relative to the scored frame, 0.5 s apart."""
    return [_observation(t0_s + offset) for offset in (-1.5, -1.0, -0.5, 0.0)]


def _policy(monkeypatch, switch: str = "on") -> DriveJepaPolicy:
    """A momentum policy whose torch pass is replaced by the bookkeeping around it.

    `forward` runs the pass between `_momentum_features` and
    `_momentum_remember`; this stand-in runs those two with a plan tagged by
    the time it was made for, and `infer` returns it with the pick block
    `_pick` builds. `test_forward_still_brackets_its_pass_with_the_bookkeeping`
    pins that the real `forward` calls the same two.
    """
    monkeypatch.setenv(momentum.SWITCH_ENV, switch)
    policy = POLICY(checkpoint="/x.ckpt")
    if policy.momentum is not None:
        policy.momentum.upstream = _FakeUpstream()
    policy.seen = []

    def infer(obs):
        policy._record(obs)
        policy.momentum_fed = False
        tag = 0.0
        if policy.momentum is not None:
            extra, ego_state, time_us = policy._momentum_features(obs)
            policy.seen.append(extra)
            tag = float(time_us)
            policy._momentum_remember(np.full((8, 3), tag), ego_state, time_us, False)
        return Trajectory(
            np.full((8, 2), tag), dt=0.5, extras={"drivejepa_pick": {"momentum_fed": policy.momentum_fed}}
        )

    policy.infer = infer
    return policy


def _fed(policy: DriveJepaPolicy) -> float | None:
    states = policy.seen[-1]["past_ego_simulated_states"]
    return None if states is None else float(states[0, 0])


def test_forward_still_brackets_its_pass_with_the_bookkeeping() -> None:
    source = inspect.getsource(DriveJepaPolicy.forward)
    assert "self._momentum_features(obs)" in source and "self._momentum_remember(" in source


def test_a_streamed_sample_keeps_its_plan_under_the_log_on_the_logs_own_clock_and_frame(monkeypatch) -> None:
    policy = _policy(monkeypatch)
    first = StreamFrame(key="log_a", timestamp_us=T0, pose=(664_000.5, 3_997_000.25, 1.2))
    answer = policy.infer_sequence_streamed(_sample(), first)
    assert answer.extras["drivejepa_pick"]["momentum_fed"] is False
    ego = policy.seen[-1]["metric_cache"].ego_state
    np.testing.assert_allclose(ego.pose, first.pose)
    assert ego.time_us == T0, "the stream's clock, not the sample's relative zero"
    np.testing.assert_allclose(ego.velocity, [3.0, 0.0])  # velocity is still the observation's

    later = StreamFrame(key="log_a", timestamp_us=T0 + 499_600, pose=(664_004.0, 3_997_001.0, 1.21))
    answer = policy.infer_sequence_streamed(_sample(), later)
    assert answer.extras["drivejepa_pick"]["momentum_fed"] is True
    assert _fed(policy) == float(T0), "handed the plan made for log_a's frame 0.5 s earlier"
    assert policy._stream is None, "the stream does not outlive its sample"


def test_a_plain_sample_between_two_streamed_ones_changes_nothing(monkeypatch) -> None:
    """A synthetic scene planned between two real frames of the same log -- as two
    workers sharing a node make happen -- neither feeds nor is fed."""
    policy = _policy(monkeypatch)
    policy.infer_sequence_streamed(_sample(), StreamFrame("log_a", T0, (0.0, 0.0, 0.0)))
    plain = policy.infer_sequence(_sample())
    assert plain.extras["drivejepa_pick"]["momentum_fed"] is False
    policy.infer_sequence_streamed(_sample(), StreamFrame("log_a", T0 + 500_000, (1.0, 0.0, 0.0)))
    assert _fed(policy) == float(T0)


def test_the_default_sequence_is_still_never_fed(monkeypatch) -> None:
    """Without a stream, a sample after another one 0.5 s earlier selects on the plain score, as before."""
    policy = _policy(monkeypatch)
    policy.infer_sequence(_sample(0.0))
    policy.infer_sequence(_sample(0.5))
    assert _fed(policy) is None
    assert policy.momentum.counts == {**policy.momentum.counts, "fed": 0, "no_previous": 2}


def test_with_momentum_off_the_stream_changes_nothing(monkeypatch) -> None:
    policy = _policy(monkeypatch, switch="off")
    streamed = policy.infer_sequence_streamed(_sample(), StreamFrame("log_a", T0, (0.0, 0.0, 0.0)))
    plain = policy.infer_sequence(_sample())
    np.testing.assert_array_equal(streamed.waypoints, plain.waypoints)
    assert streamed.extras == plain.extras


def test_the_recorded_drive_through_a_stream_book_with_a_reset_per_replan_picks_the_same() -> None:
    """Upstream's stage, fed from a stream's book with the episode reset before
    every replan -- as each streamed `/infer_sequence` resets -- picks what the
    recorded driver picked. Needs Drive-JEPA's environment, as
    `test_momentum.test_the_recorded_drive_is_reproduced_pick_for_pick` does."""
    pytest.importorskip("nuplan", reason="needs Drive-JEPA's environment (nuPlan)")
    model = pytest.importorskip(
        "navsim.agents.drive_jepa_perception_based.drive_jepa_model",
        reason="needs Drive-JEPA's environment (vendored NAVSIM on PYTHONPATH)",
    )
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from paddock.adapters.models.drivejepa.tests.test_momentum import RECORDED_DRIVE

    drive = np.load(RECORDED_DRIVE)
    book = momentum.Momentum(momentum.Upstream(TrajectorySampling(time_horizon=4, interval_length=0.5)))
    for i, time_us in enumerate(drive["time_us"].tolist()):
        book.reset()
        extra, ego_state = book.features(
            drive["ego_pose"][i],
            drive["ego_velocity"][i],
            drive["ego_acceleration"][i],
            time_us,
            stream="drive",
        )
        proposals = torch.from_numpy(drive["proposals"][i][None])
        score = torch.sigmoid(torch.from_numpy(drive["pred_logit"][i][None]))[:, :, -1]
        if extra["past_ego_simulated_states"] is not None:
            score = model.DriveJEPAModel.calibrate_score(None, extra, proposals, score)
        pick = int(torch.argmax(score, dim=1)[0])
        assert pick == int(drive["pick"][i]), f"replan {i}"
        book.remember(proposals[0, pick].numpy(), ego_state, time_us, stream="drive")
    assert book.counts["fed"] == 26
