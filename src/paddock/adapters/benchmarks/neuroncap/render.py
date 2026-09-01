"""Comparison figures for neuroNCAP runs.

Closed-loop comparison has a different shape from open-loop. There is no
per-sample trajectory error to plot a distribution of: there are a handful of
scenarios, a score each, and — for stochastic models — a spread across seeds
that matters more than any single value.

So this draws scores per scenario with their spread, and says n. Alpamayo's
frontal score ranges over the whole 0–5 scale across seeds while openpilot's
moves by 0.75, which a bar chart of single runs would hide entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddock.viz import palette


def render(runs, out_dir: Path, cfg) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = [_scores_by_scenario(runs, out_dir, plt)]
    return [p for p in written if p is not None]


def _scores_by_scenario(runs, out_dir: Path, plt) -> Path | None:
    """One panel per scenario: each model's score, with its spread across runs."""
    by_scenario: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for sample in run.samples:
            score = sample.get("metrics", {}).get("ncap_score")
            if score is None:
                continue
            scenario = sample.get("split") or "unknown"
            by_scenario.setdefault(scenario, {}).setdefault(run.model, []).append(float(score))

    if not by_scenario:
        return None

    scenarios = sorted(by_scenario)
    models = sorted({m for per_model in by_scenario.values() for m in per_model})
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(3.6 * len(scenarios), 4.2), squeeze=False, sharey=True
    )

    for ax, scenario in zip(axes[0], scenarios, strict=True):
        for index, model in enumerate(models):
            values = by_scenario[scenario].get(model)
            if not values:
                continue
            colour = palette.color(model)
            mean = float(np.mean(values))
            ax.bar(index, mean, color=colour, alpha=0.55, width=0.62)
            # Every run as a point, so a mean over a wide spread cannot be
            # mistaken for a tight one.
            ax.scatter(
                np.full(len(values), index, dtype=float),
                values,
                color=colour,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                s=34,
            )
            if len(values) > 1:
                ax.plot([index, index], [min(values), max(values)], color=colour, lw=1.4, zorder=2)
            ax.text(index, 5.12, f"n={len(values)}", ha="center", fontsize=7, color="#555")

        ax.set_title(scenario, fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 5.4)
        ax.grid(axis="y", alpha=0.3, lw=0.6)

    axes[0][0].set_ylabel("NCAP score (5 = avoided, 0 = full-speed impact)")
    fig.suptitle("neuroNCAP scores by scenario — bar is the mean, points are runs", fontsize=11)
    fig.tight_layout()
    path = out_dir / "scores_by_scenario.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
