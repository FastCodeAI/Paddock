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
    "alpamayo2": "#66a61e",
    # Added when METEOR's hashed fallback (#9467bd) turned out to be
    # indistinguishable from alpamayo15's #7570b3 in the NAVSIM comparison.
    "meteor": "#e6ab02",
    # Red because it is the one hue nothing else here occupies. The obvious
    # choices were taken: the comma2k19 branch originally gave Qwen-Drive
    # #e6ab02, which METEOR had already claimed above, and Qwen-Drive now
    # appears on every board METEOR does. Against openpilot's #d95f02 -- the
    # nearest neighbour -- it separates on lightness as well as hue, which is
    # what has to hold once red and orange both desaturate under deuteranopia.
    "qwendrive": "#cb181d",
    # Deliberately a sibling of drivejepa's green: same V-JEPA backbone, a
    # different agent head. Where a reader meets both, the kinship is the useful
    # signal and the hue difference still separates them.
    "drivejepa_free": "#0f766e",
    # The floor gets grey, and gets an entry at all because the hash handed it
    # #e377c2 -- a pink two steps from visionpilot's #e7298a, on a board where
    # both now appear. Grey also says what it is: the line every other line is
    # being read against, not a competitor.
    "constant_velocity": "#666666",
    # The hash gave autoe2e that same #666666, so it drew as the floor. Cyan is
    # the remaining hue: every other model and ground truth's blue stay at least
    # 11.8 apart from it under colour blindness (OKLab x100).
    "autoe2e": "#17becf",
    # The hash gave alpamayo2edge the floor's #666666 too. A deep purple, kin to
    # alpamayo15's #7570b3 because it is the same family, and the most separated
    # of the candidates tried: under colour blindness it stays 20.0 from
    # visionpilot's #e7298a, its nearest, and 25.1 from alpamayo15 (OKLab x100).
    "alpamayo2edge": "#3f007d",
    # The hash gave uniad #9467bd, the purple METEOR left because it could not be
    # told from alpamayo15's; and vad #e377c2, two steps from visionpilot's pink.
    # By the same measure as above (OKLab x100, nearest model under colour
    # blindness, then ground truth's blue), each at least 1.8:1 against both chart
    # surfaces: uniad 10.1 from wajepa, 23.4 from ground truth; vad 10.0 from
    # autoe2e, 28.3. The palette is crowded enough that no colour clears 20 from
    # ground truth and 8 from every model at once.
    "uniad": "#900000",
    "vad": "#70d0a0",
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
