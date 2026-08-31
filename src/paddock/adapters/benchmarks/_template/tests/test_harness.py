"""Golden test for the template benchmark adapter.

Every benchmark adapter ships one. Pin a handful of samples and their expected
metric values, so a refactor that changes the numbers fails loudly instead of
silently rewriting history.
"""

from __future__ import annotations

import numpy as np


def test_metrics_are_stable() -> None:
    """A straight-line prediction against a straight-line future scores zero."""
    from paddock.metrics import trajectory as M

    pred = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    gt = pred.copy()

    assert M.ade(pred, gt) == 0.0
    assert M.fde(pred, gt, t=3.0, dt=0.5) == 0.0


def test_lateral_offset_is_measured() -> None:
    from paddock.metrics import trajectory as M

    pred = np.stack([np.arange(1, 7) * 5.0, np.full(6, 2.0)], axis=1)
    gt = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)

    assert M.ade(pred, gt) == 2.0
