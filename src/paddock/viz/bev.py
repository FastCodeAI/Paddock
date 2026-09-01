"""Bird's-eye trajectory drawing, shared by every benchmark.

Knows nothing about any model or benchmark: it takes named trajectories and
draws them consistently. What a comparison *means* -- which runs to load, what
the ground truth is, how to label it -- belongs to the benchmark's `render.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from paddock.viz import palette


def draw(
    ax,
    predictions: dict[str, Sequence],
    ground_truth=None,
    labels: dict[str, str] | None = None,
    ego_marker: bool = True,
) -> None:
    """Draw one BEV panel: ground truth underneath, predictions over it.

    Axes are the vehicle convention rotated for reading: x forward is plotted
    upward, y left is plotted leftward, so the figure matches the view out of
    the windscreen rather than a maths textbook.
    """
    if ground_truth is not None:
        gt = np.asarray(ground_truth, dtype=float)
        band, line = palette.ground_truth_style()
        ax.plot(gt[:, 1], gt[:, 0], **band)
        ax.plot(gt[:, 1], gt[:, 0], **line)

    # Keys are model names, so the colour is the model's. The legend text is
    # passed separately: folding a metric into the key would make the colour a
    # function of the number, and a model would change colour between panels --
    # which is the exact confusion the palette exists to prevent.
    for model, trajectory in predictions.items():
        xy = np.asarray(trajectory, dtype=float)
        ax.plot(
            xy[:, 1],
            xy[:, 0],
            **palette.style(model),
            label=(labels or {}).get(model, model),
        )

    if ego_marker:
        ax.plot(0, 0, "ks", ms=7, zorder=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3, lw=0.6)
    ax.set_xlabel("lateral (m, left positive)")
    ax.set_ylabel("forward (m)")
    ax.invert_xaxis()  # y-left positive reads naturally on the left


def metric_labels(metrics: dict[str, float], unit: str = "m") -> dict[str, str]:
    """Legend text carrying each model's headline number, keyed by model.

    A comparison figure is read by looking at the lines and then wanting the
    number; putting it in the legend saves a lookup and makes the figure
    self-contained when pasted into a message. Returned as a separate mapping so
    the colour stays keyed on the model -- see `draw`.
    """
    return {model: f"{model}  {value:.2f} {unit}" for model, value in metrics.items()}
