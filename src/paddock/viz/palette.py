"""Stable colours for models and ground truth.

The point is stability across figures. Debugging means putting several models
side by side, often in several plots at once -- a BEV here, an error curve
there, a contact sheet after. If a model's colour is assigned by its position in
whatever list a script happened to build, it changes between figures and the
reader has to re-read the legend every time, or worse, does not notice.

So colour is a function of the model's name, fixed once. A model added later
gets a colour without disturbing anyone else's.
"""

from __future__ import annotations

import hashlib

#: Ground truth is always this blue, and no model may take it.
GROUND_TRUTH = "#1f77b4"

#: Assigned colours, chosen to stay distinguishable in both light and dark and
#: for the most common forms of colour blindness. Extend rather than reorder:
#: reordering silently recolours every figure ever produced.
ASSIGNED: dict[str, str] = {
    "openpilot": "#d95f02",
    "alpamayo15": "#7570b3",
    "alpamayo": "#7570b3",
    "drivejepa": "#1b9e77",
    "visionpilot": "#e7298a",
    "sunnypilot": "#a6761d",
    "alpamayo2": "#66a61e",
    # Added when METEOR's hashed fallback (#9467bd) turned out to be
    # indistinguishable from alpamayo15's #7570b3 in the NAVSIM comparison.
    "meteor": "#e6ab02",
}

#: Drawn from for models with no assigned colour.
_FALLBACK = ("#666666", "#8c564b", "#17becf", "#bcbd22", "#9467bd", "#e377c2")


def color(model: str) -> str:
    """A stable colour for one model.

    Deterministic for unknown names too, so a model someone adds this afternoon
    keeps the same colour tomorrow without needing an entry here first.
    """
    if model in ASSIGNED:
        return ASSIGNED[model]
    digest = hashlib.sha256(model.encode()).digest()
    return _FALLBACK[digest[0] % len(_FALLBACK)]


def style(model: str, **overrides) -> dict:
    """Line style for one model's trajectory."""
    base = {"color": color(model), "lw": 2.0, "zorder": 4, "solid_capstyle": "round"}
    return base | overrides


def ground_truth_style() -> tuple[dict, dict]:
    """A wide translucent band plus a thin centre line.

    The band makes the recorded path readable underneath several predictions
    without hiding any of them; the line keeps its exact position legible.
    """
    band = {
        "color": GROUND_TRUTH,
        "lw": 7.0,
        "alpha": 0.28,
        "zorder": 2,
        "solid_capstyle": "round",
    }
    line = {"color": GROUND_TRUTH, "lw": 1.4, "ls": "-", "zorder": 3, "label": "human (GT)"}
    return band, line
