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
    (directory / "checkpoint.json").write_text(json.dumps({"train_run": "t", "step": 5}))
    return directory


def test_features_have_one_column_per_name() -> None:
    features = ranker.candidate_features(straight(32, 2.0), np.zeros((32, 6)), speed=4.0)
    assert features.shape == (32, len(ranker.FEATURE_NAMES))
    assert np.isfinite(features).all()
    # ego_speed is broadcast; travel_3s grows with the candidate's speed.
    assert (features[:, ranker.FEATURE_NAMES.index("ego_speed")] == 4.0).all()
    assert np.all(np.diff(features[:, ranker.FEATURE_NAMES.index("travel_3s")]) > 0)


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
