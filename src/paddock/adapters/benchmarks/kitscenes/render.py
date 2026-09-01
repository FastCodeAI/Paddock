"""Comparison figures for KITScenes runs.

Takes runs from any models and draws them together, because that is what
debugging an open-loop result actually looks like: which model is wrong, where,
and by how much.

Scoped to the benchmark rather than to any model. Before paddock this existed
three times -- a three-way plot under openpilot, a trajectory plot under
Drive-JEPA, a per-sample sheet under Drive-JEPA again -- each able to draw only
the model it lived beside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddock.viz import bev, palette


def render(runs, out_dir: Path, cfg) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    written.append(_error_distribution(runs, out_dir, plt))
    worst = _worst_samples(runs, limit=6)
    if worst:
        written.append(_worst_grid(runs, worst, out_dir, plt))
    return [p for p in written if p is not None]


def _error_distribution(runs, out_dir: Path, plt) -> Path:
    """Where each model's error sits, not just its mean.

    A mean hides the shape: two models with the same ADE can fail on entirely
    different samples, and the tail is usually what matters.
    """
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.2))

    for run in runs:
        ades = [s["metrics"]["ade"] for s in run.samples if "ade" in s.get("metrics", {})]
        if not ades:
            continue
        colour = palette.color(run.model)
        mean = float(np.mean(ades))
        ax_hist.hist(
            ades,
            bins=30,
            histtype="step",
            lw=1.8,
            color=colour,
            label=f"{run.model}  mean {mean:.3f}  n={len(ades)}",
        )
        ordered = np.sort(ades)
        ax_cdf.plot(ordered, np.linspace(0, 1, len(ordered)), lw=1.8, color=colour, label=run.model)

    ax_hist.set_xlabel("ADE (m)")
    ax_hist.set_ylabel("samples")
    ax_hist.legend(fontsize=7.5)
    ax_hist.grid(alpha=0.3, lw=0.6)

    ax_cdf.set_xlabel("ADE (m)")
    ax_cdf.set_ylabel("fraction of samples below")
    ax_cdf.grid(alpha=0.3, lw=0.6)
    ax_cdf.axhline(0.5, color="#999", lw=0.8, ls="--")

    fig.suptitle("KITScenes error distribution", fontsize=11)
    fig.tight_layout()
    path = out_dir / "error_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _worst_samples(runs, limit: int) -> list[str]:
    """Samples where the models disagree most, which is where to look first."""
    by_sample: dict[str, list[float]] = {}
    for run in runs:
        for sample in run.samples:
            ade = sample.get("metrics", {}).get("ade")
            if ade is not None:
                by_sample.setdefault(sample["sample_id"], []).append(ade)
    spread = {k: max(v) - min(v) for k, v in by_sample.items() if len(v) > 1}
    if not spread:  # a single run: fall back to its own worst
        spread = {k: v[0] for k, v in by_sample.items()}
    return [k for k, _ in sorted(spread.items(), key=lambda kv: -kv[1])[:limit]]


def _worst_grid(runs, sample_ids: list[str], out_dir: Path, plt) -> Path | None:
    """BEV panels for the samples the models most disagree on."""
    indexed = {run.model: {s["sample_id"]: s for s in run.samples} for run in runs}
    if not any(indexed.values()):
        return None

    columns = min(3, len(sample_ids))
    rows = (len(sample_ids) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), squeeze=False)

    for ax, sample_id in zip(axes.flat, sample_ids, strict=False):
        predictions, metrics, ground_truth = {}, {}, None
        for model, samples in indexed.items():
            sample = samples.get(sample_id)
            if sample is None:
                continue
            if "prediction" in sample:
                predictions[model] = sample["prediction"]
                metrics[model] = sample["metrics"].get("ade", float("nan"))
            ground_truth = sample.get("ground_truth", ground_truth)
        if predictions:
            bev.draw(ax, predictions, ground_truth, labels=bev.metric_labels(metrics))
            ax.legend(fontsize=6.5, loc="best", framealpha=0.88)
        ax.set_title(sample_id.rsplit("/", 2)[-2][:8] + "/" + sample_id.rsplit("/", 1)[-1], fontsize=8)

    for ax in list(axes.flat)[len(sample_ids) :]:
        ax.axis("off")
    fig.suptitle("Samples the models most disagree on", fontsize=11)
    fig.tight_layout()
    path = out_dir / "worst_disagreement.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
