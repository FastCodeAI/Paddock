"""Contract tests for alpamayo15_overfit. No torch, no GPU, no weights.

`super().load()` -- the 20 GB Alpamayo + 8B Cosmos VLM read -- is never
reached in these tests: every case here is refused before it, by design (see
policy.py's own comment on why the checkpoint check comes first).
"""

from __future__ import annotations

import pytest

from paddock.adapters.models.alpamayo15_overfit import policy as overfit_policy


def test_load_raises_when_overlay_checkpoint_is_unset(monkeypatch) -> None:
    monkeypatch.setattr(overfit_policy, "OVERLAY_CHECKPOINT", None)
    p = overfit_policy.OverfitAlpamayoPolicy()

    with pytest.raises(RuntimeError, match="ALPAMAYO_OVERLAY_CHECKPOINT"):
        p.load()


def test_load_raises_when_overlay_checkpoint_does_not_exist(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setattr(overfit_policy, "OVERLAY_CHECKPOINT", str(missing))
    p = overfit_policy.OverfitAlpamayoPolicy()

    with pytest.raises(RuntimeError, match="does not exist"):
        p.load()


def test_load_checks_the_overlay_before_the_expensive_base_load(monkeypatch, tmp_path) -> None:
    """Regression guard: the overlay check must come before `super().load()`'s
    20 GB+ checkpoint read, so a bad/missing overlay fails in seconds."""
    missing = tmp_path / "does-not-exist.pt"
    monkeypatch.setattr(overfit_policy, "OVERLAY_CHECKPOINT", str(missing))

    def _fail_if_called(self):
        raise AssertionError("super().load() must not run before the overlay check")

    monkeypatch.setattr(overfit_policy.AlpamayoPolicy, "load", _fail_if_called)
    p = overfit_policy.OverfitAlpamayoPolicy()

    with pytest.raises(RuntimeError, match="does not exist"):
        p.load()
