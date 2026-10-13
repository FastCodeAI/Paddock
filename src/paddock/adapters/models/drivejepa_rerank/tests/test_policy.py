"""Contract tests for drivejepa_rerank. No navsim, no torch, no GPU, no weights."""

from __future__ import annotations

import json

import numpy as np
import pytest

from paddock.adapters.models.drivejepa_rerank import ranker
from paddock.adapters.models.drivejepa_rerank.policy import POLICY
from paddock.policy.protocol import CANBUS_VELOCITY, Command, Observation, Policy


def straight(n: int, metres_per_step: float) -> np.ndarray:
    """(n, 8, 3) candidates driving straight at different speeds."""
    steps = np.arange(1, 9) * metres_per_step
    out = np.zeros((n, 8, 3))
    out[:, :, 0] = steps[None, :] * (1 + np.arange(n))[:, None] / n
    return out


def write_weights(directory, *, fmt=ranker.FORMAT, names=ranker.FEATURE_NAMES, tamper=0.0, hidden=4):
    """A `reranker.npz` shaped as gallops writes one, with a self-consistent probe."""
    rng = np.random.default_rng(0)
    f = len(ranker.FEATURE_NAMES)
    arrays = {
        "mean": np.zeros(f),
        "std": np.ones(f),
        "w2": rng.normal(size=(hidden or f, 1)),
        "b2": np.zeros(1),
    }
    if hidden:
        arrays |= {"w1": rng.normal(size=(f, hidden)), "b1": np.zeros(hidden)}
    probe_x = rng.normal(size=(2, 32, f))
    arrays["probe_x"] = probe_x
    arrays["probe_scores"] = ranker.Reranker(arrays).score(probe_x) + tamper
    arrays["format"] = np.array(fmt)
    arrays["feature_names"] = np.array(names)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "reranker.npz", **arrays)
    (directory / "checkpoint.json").write_text(
        json.dumps({"train_run": "t", "step": 5, "weights_sha256": "ab" * 32})
    )
    return directory


def test_features_have_one_column_per_name() -> None:
    features = ranker.candidate_features(straight(32, 2.0), np.zeros((32, 6)), speed=4.0)
    assert features.shape == (32, len(ranker.FEATURE_NAMES))
    assert np.isfinite(features).all()
    # ego_speed is broadcast; travel_3s grows with the candidate's speed.
    assert (features[:, ranker.FEATURE_NAMES.index("ego_speed")] == 4.0).all()
    assert np.all(np.diff(features[:, ranker.FEATURE_NAMES.index("travel_3s")]) > 0)


def test_features_match_gallops() -> None:
    """Gallops computes the training features with its own copy of `candidate_features`.

    Neither repo imports the other, and the load-time probe replays stored
    features, so it cannot see the two definitions part. This input and these
    values are also in gallops' tests/test_rerank_data.py (`GOLDEN`); if this
    fails, the definition here moved, and gallops' copy and every trained
    checkpoint move with it.
    """
    t = 0.5 * np.arange(1, 9)
    proposals = np.stack(
        [
            np.column_stack([4.0 * t, np.zeros(8), np.zeros(8)]),
            np.column_stack([3.0 * t, 0.2 * t**2, np.arctan(0.4 * t / 3.0)]),
            np.column_stack([2.0 * t - 0.1 * t**2, -0.3 * t, -0.05 * t]),
        ]
    )
    logits = np.array(
        [
            [0.5, -1.0, 2.0, 0.0, 1.5, -0.5],
            [1.0, 0.25, -2.0, 3.0, 0.0, 0.75],
            [-0.5, 0.5, 1.0, -1.5, 2.5, 0.125],
        ]
    )
    # travel_1s .. total_turning
    golden = [
        [4.0, 8.0, 12.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
        [
            3.0066592756745814,
            6.053098380168622,
            9.178235124467012,
            1.8,
            0.38050637711236496,
            1.8,
            1.5388329608336029,
            1.5008331019803633,
            1.5976545308670458,
            0.0340556209568158,
            0.31393821333654115,
        ],
        [
            1.9235384061671343,
            3.6496575181789317,
            5.1788029504896205,
            -0.8999999999999999,
            -0.15000000000000002,
            0.8999999999999999,
            0.8632635125463421,
            0.7403546447480419,
            0.9864709828474428,
            0.08407005752833539,
            0.12500000000000003,
        ],
    ]
    features = ranker.candidate_features(proposals, logits, 4.2)
    np.testing.assert_array_equal(features[:, :6], logits)
    np.testing.assert_allclose(features[:, 6:17], golden, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(features[:, 17], 4.2)


@pytest.mark.parametrize("hidden", [4, 0])
def test_consistent_weights_load(tmp_path, hidden) -> None:
    reranker = ranker.load(write_weights(tmp_path / "ok", hidden=hidden))
    assert 0 <= reranker.choose(straight(32, 2.0), np.zeros((32, 6)), 4.0) < 32


def test_wrong_format_is_refused(tmp_path) -> None:
    with pytest.raises(ranker.RerankerError, match="format"):
        ranker.load(write_weights(tmp_path / "w", fmt="something-else/9"))


def test_features_trained_on_another_definition_are_refused(tmp_path) -> None:
    names = tuple(reversed(ranker.FEATURE_NAMES))
    with pytest.raises(ranker.RerankerError, match="trained on features"):
        ranker.load(write_weights(tmp_path / "w", names=names))


def test_a_drifted_forward_pass_is_refused(tmp_path) -> None:
    """The probe is gallops's forward pass; ours must reproduce it."""
    with pytest.raises(ranker.RerankerError, match="drifted"):
        ranker.load(write_weights(tmp_path / "w", tamper=1e-3))


def test_missing_weights_name_the_path(tmp_path) -> None:
    with pytest.raises(ranker.RerankerError, match="no re-ranker weights"):
        ranker.load(tmp_path / "absent")


def test_policy_imports_without_navsim_installed() -> None:
    policy = POLICY(weights="/nonexistent", checkpoint="/x.ckpt")
    assert isinstance(policy, Policy)


def test_the_momentum_switch_is_refused(monkeypatch) -> None:
    """With it on, the stage would be fed the model's pick, not the plan that was driven."""
    monkeypatch.setenv("DRIVEJEPA_MOMENTUM", "on")
    with pytest.raises(ValueError, match="not supported by drivejepa_rerank"):
        POLICY(weights="/w", checkpoint="/x.ckpt")


def test_momentum_off_is_accepted_and_recorded(monkeypatch) -> None:
    monkeypatch.setenv("DRIVEJEPA_MOMENTUM", "off")
    assert POLICY(weights="/w", checkpoint="/x.ckpt").diagnostics()["momentum"] == "off"


def test_load_without_weights_says_which_setting(monkeypatch) -> None:
    monkeypatch.delenv("DRIVEJEPA_RERANK_WEIGHTS", raising=False)
    with pytest.raises(RuntimeError, match="DRIVEJEPA_RERANK_WEIGHTS"):
        POLICY(checkpoint="/x.ckpt").load()


def test_infer_without_load_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        POLICY(weights="/w", checkpoint="/x.ckpt").infer(object())  # type: ignore[arg-type]


class _Tensor:
    """Just enough of a torch tensor for `infer`: [0], .float(), .cpu(), .numpy(), .argmax()."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def argmax(self):
        return int(self.array.argmax())


def test_infer_drives_the_rerankers_choice_not_the_models(tmp_path) -> None:
    policy = POLICY(weights=str(write_weights(tmp_path)), checkpoint="/x.ckpt")
    policy._load_reranker()  # the half of load() that needs no navsim
    proposals = straight(32, 2.0)
    logits = np.random.default_rng(1).normal(size=(32, 6))
    expected = policy.reranker.choose(proposals, logits, 4.0)
    model_pick = (expected + 7) % 32
    pdm = np.zeros(32)
    pdm[model_pick] = 1.0
    policy.forward = lambda obs: {  # type: ignore[method-assign]
        "proposals": _Tensor(proposals[None]),
        "pred_logit": _Tensor(logits[None]),
        "pdm_score": _Tensor(pdm[None]),
    }
    canbus = np.zeros(18)
    canbus[CANBUS_VELOCITY] = 4.0
    obs = Observation(
        images={},
        ego2world=np.eye(4),
        canbus=canbus,
        timestamp_us=0,
        command=Command.STRAIGHT,
        camera2image={},
        camera2ego={},
    )

    plan = policy.infer(obs)
    np.testing.assert_allclose(plan.waypoints, proposals[expected])
    assert policy.last == {"chosen": expected, "model_pick": model_pick}
    assert policy.diagnostics()["reranker"]["train_run"] == "t"
    assert policy.diagnostics()["reranker"]["weights_sha256"] == "ab" * 32


def test_weights_without_their_sidecar_are_refused(tmp_path) -> None:
    """The sidecar says what the weights were trained on; the train/test guard
    in `rerank.py report` and the run record both depend on it. Copying only
    reranker.npz must fail loudly, not run with an empty provenance."""
    directory = write_weights(tmp_path / "w")
    (directory / "checkpoint.json").unlink()
    with pytest.raises(ranker.RerankerError, match="checkpoint.json"):
        ranker.load(directory)


def test_the_loaded_sidecar_travels_with_the_weights(tmp_path) -> None:
    reranker = ranker.load(write_weights(tmp_path / "w"))
    assert reranker.sidecar["train_run"] == "t"
