"""The drawing layer.

Mostly one property: a model's colour must not depend on anything except its
name. Comparing models means producing several figures at a time, and a colour
that shifts between them costs the reader either a legend re-read or, worse, a
wrong conclusion.

The regression here is real. The first renderer folded each model's metric into
the dict key ("openpilot  2.84 m") before drawing, which made the colour a
function of the number -- openpilot came out grey, brown, grey and yellow-green
across four panels of a single figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from paddock.viz import bev, palette


def test_colour_is_a_function_of_the_model_name_only() -> None:
    assert palette.color("openpilot") == palette.color("openpilot")
    assert palette.color("a-model-nobody-registered") == palette.color("a-model-nobody-registered")


def test_known_models_have_distinct_colours() -> None:
    names = ["openpilot", "alpamayo15", "drivejepa", "visionpilot"]
    assert len({palette.color(n) for n in names}) == len(names)


def test_no_model_takes_the_ground_truth_colour() -> None:
    """Ground truth must stay visually distinct from every prediction."""
    assert palette.GROUND_TRUTH not in palette.ASSIGNED.values()
    for name in ("openpilot", "alpamayo15", "drivejepa", "visionpilot", "unregistered"):
        assert palette.color(name) != palette.GROUND_TRUTH


def test_metric_labels_do_not_change_the_key() -> None:
    """The regression: legend text must be separate from the colour key.

    If labelling returned `{"openpilot  2.84 m": traj}` the colour would follow
    the metric, and a model would be a different colour in every panel.
    """
    labels = bev.metric_labels({"openpilot": 2.84, "alpamayo15": 0.51})
    assert set(labels) == {"openpilot", "alpamayo15"}
    assert labels["openpilot"] == "openpilot  2.84 m"


def test_colour_survives_relabelling() -> None:
    """Same model, different metric, same colour -- across panels."""
    first = bev.metric_labels({"openpilot": 2.84})
    second = bev.metric_labels({"openpilot": 0.62})
    assert set(first) == set(second)
    assert palette.color(next(iter(first))) == palette.color(next(iter(second)))


def test_draw_uses_the_model_key_for_colour_and_the_label_for_text() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    straight = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    fig, ax = plt.subplots()
    bev.draw(
        ax,
        {"openpilot": straight, "drivejepa": straight + 1.0},
        ground_truth=straight,
        labels=bev.metric_labels({"openpilot": 2.84, "drivejepa": 0.51}),
    )

    drawn = {line.get_label(): line.get_color() for line in ax.get_lines()}
    assert drawn["openpilot  2.84 m"] == palette.color("openpilot")
    assert drawn["drivejepa  0.51 m"] == palette.color("drivejepa")
    plt.close(fig)


def test_draw_orients_the_axes_for_reading() -> None:
    """x forward plots upward and y left plots leftward, matching the windscreen."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ahead_and_left = np.array([[10.0, 4.0], [20.0, 8.0]])
    bev.draw(ax, {"m": ahead_and_left})
    line = ax.get_lines()[0]
    assert line.get_ydata()[-1] > line.get_ydata()[0], "forward must increase upward"
    left, right = ax.get_xlim()
    assert left > right, "y-left positive must read on the left (inverted x axis)"
    plt.close(fig)
