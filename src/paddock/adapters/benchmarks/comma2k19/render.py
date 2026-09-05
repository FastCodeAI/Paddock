"""Comparison figures for comma2k19 runs.

Takes runs from any models and draws them together, scoped to the benchmark
rather than to any model -- the same contract as `kitscenes/render.py`:

    def render(runs, out_dir, cfg) -> list[Path]

Three figures, and the choice of them is the whole point of this module:

1. **Error distribution**, because a mean hides the shape. Two models with the
   same ADE can fail on entirely different segments.
2. **The along/cross decomposition**, because on this benchmark a single ADE is
   genuinely ambiguous. A supercombo-style plan is a path plus a speed profile
   and the two fail independently; only the cross-track half is a lane-keeping
   failure. Reading the total as lateral is exactly the mistake the previous
   comma2k19 report made.
3. **Path-length ratio**, because it separates the one failure mode that looks
   like a modelling result and is not: a plan that goes nowhere. A ratio near 0
   means the model predicted a stationary car, which produces a large ADE that
   is entirely longitudinal and a *small* axis-aligned lateral error -- a
   combination that reads as "tracks the lane, misjudges speed" and is nothing
   of the kind.

Colour comes from `paddock.viz.palette`, so a model keeps its colour across
every figure in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddock.viz import palette

_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#8f8e88"
_SURFACE = "#fcfcfb"
_GRID = "#e5e4e0"


def render(runs, out_dir: Path, cfg) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _error_distribution(runs, out_dir, plt),
        _decomposition(runs, out_dir, plt),
        _path_length(runs, out_dir, plt),
    ]
    return [p for p in written if p is not None]


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.xaxis.label.set_color(_INK2)
    ax.yaxis.label.set_color(_INK2)
    ax.title.set_color(_INK)


def _values(run, key: str) -> np.ndarray:
    return np.array(
        [
            s["metrics"][key]
            for s in run.samples
            if isinstance(s.get("metrics", {}).get(key), (int, float)) and np.isfinite(s["metrics"][key])
        ],
        dtype=np.float64,
    )


def _legend(ax) -> None:
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(_INK2)


def _error_distribution(runs, out_dir: Path, plt) -> Path | None:
    """Histogram and CDF of per-sample ADE."""
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=_SURFACE)
    _style(ax_hist)
    _style(ax_cdf)
    drew = False
    for run in runs:
        ades = _values(run, "ade")
        if not len(ades):
            continue
        drew = True
        colour = palette.color(run.model)
        ax_hist.hist(
            ades,
            bins=30,
            histtype="step",
            lw=1.8,
            color=colour,
            label=f"{run.model}  mean {ades.mean():.2f}  median {np.median(ades):.2f}  n={len(ades)}",
        )
        ordered = np.sort(ades)
        ax_cdf.plot(ordered, np.linspace(0, 1, len(ordered)), lw=1.8, color=colour, label=run.model)
    if not drew:
        plt.close(fig)
        return None
    ax_hist.set_xlabel("ADE (m)")
    ax_hist.set_ylabel("segments")
    ax_hist.set_title("Per-sample ADE", fontsize=11, loc="left")
    ax_cdf.set_xlabel("ADE (m)")
    ax_cdf.set_ylabel("fraction of samples at or below")
    ax_cdf.set_title("Cumulative", fontsize=11, loc="left")
    _legend(ax_hist)
    _legend(ax_cdf)
    fig.tight_layout()
    path = out_dir / "error_distribution.png"
    fig.savefig(path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    return path


def _decomposition(runs, out_dir: Path, plt) -> Path | None:
    """Along-track vs cross-track, per model, as grouped bars.

    Drawn side by side rather than stacked: the question is which of the two
    dominates, and a stacked bar makes the smaller one hard to read.
    """
    keys = [
        ("along_track_mae", "along-track (speed)"),
        ("cross_track_mae", "cross-track (path)"),
        ("lateral_at_arclength_mae", "lateral @ arc length"),
    ]
    have = [r for r in runs if len(_values(r, "along_track_mae"))]
    if not have:
        return None

    fig, ax = plt.subplots(figsize=(1.9 * len(have) + 5.0, 4.4), facecolor=_SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    width = 0.8 / len(keys)
    x = np.arange(len(have))
    # Hatch as well as position carries the metric identity, so the three bars
    # are distinguishable without relying on the model's single colour.
    hatches = ["", "///", "..."]
    for j, ((key, label), hatch) in enumerate(zip(keys, hatches, strict=True)):
        vals = [float(np.mean(_values(r, key))) if len(_values(r, key)) else np.nan for r in have]
        ax.bar(
            x + j * width - 0.4 + width / 2,
            vals,
            width * 0.92,
            color=[palette.color(r.model) for r in have],
            alpha=1.0 - 0.22 * j,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for xi, v in zip(x + j * width - 0.4 + width / 2, vals, strict=True):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.2f}",
                    (xi, v),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    color=_INK2,
                    fontsize=8,
                )
    ax.set_xticks(x, [r.model for r in have], fontsize=9)
    ax.set_ylabel("mean absolute error (m)")
    ax.set_title(
        "Error decomposed against the ground-truth path tangent (lower is better)",
        fontsize=11,
        loc="left",
    )
    _legend(ax)
    fig.tight_layout()
    path = out_dir / "decomposition.png"
    fig.savefig(path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    return path


def _path_length(runs, out_dir: Path, plt) -> Path | None:
    """Predicted path length / recorded path length, per sample.

    1.0 is right. Near 0 means the plan went nowhere, which is the signature of
    a broken camera pipeline rather than a modelling limitation -- see the
    module docstring.
    """
    have = [r for r in runs if len(_values(r, "path_length_ratio"))]
    if not have:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor=_SURFACE)
    _style(ax)
    for run in have:
        vals = _values(run, "path_length_ratio")
        ordered = np.sort(vals)
        ax.plot(
            ordered,
            np.linspace(0, 1, len(ordered)),
            lw=1.8,
            color=palette.color(run.model),
            label=f"{run.model}  median {np.median(vals):.3f}",
        )
    ax.axvline(1.0, color=_MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.annotate(
        "correct distance", (1.0, 0.06), textcoords="offset points", xytext=(6, 0), color=_INK2, fontsize=9
    )
    ax.set_xlim(0, max(1.6, ax.get_xlim()[1]))
    ax.set_xlabel("predicted path length / recorded path length")
    ax.set_ylabel("fraction of samples at or below")
    ax.set_title("Did the plan cover the right distance?", fontsize=11, loc="left")
    _legend(ax)
    fig.tight_layout()
    path = out_dir / "path_length_ratio.png"
    fig.savefig(path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    return path
