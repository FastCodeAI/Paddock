"""Comparison figures for nuPlan runs.

Three of the four figures here exist because of what nuPlan has and the other
open-loop benchmarks do not: **a scenario-type label on every sample**. An ADE
of 3.2 says a model is wrong; `stopping_at_stop_sign` at 0.2 m/s scoring 3.4
while `accelerating_at_stop_sign` scores 1.6 says *how*, and that is the
difference between a number and a finding.

Everything is computed from the per-sample records alone -- predictions, ground
truth and the type label are all in the store -- so `paddock render` needs no
dataset, no camera images and no GPU. The constant-velocity baseline is derived
from the recorded future rather than read from a run, because it is the floor
every model must clear and it would be absurd to require a saved run to draw it.
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from paddock.metrics import trajectory as M
from paddock.viz import bev, palette

#: Ground truth is sampled every 0.5 s (see harness.INTERVAL_S), so the first
#: waypoint's distance over that interval is the ego speed entering the scored
#: moment. Derived rather than stored: the records carry no speed column, and
#: the whole point is to avoid needing the dataset to draw a figure.
INTERVAL_S = 0.5
EVAL_HORIZON_S = 3.0


def render(runs, out_dir: Path, cfg) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = [
        _error_distribution(runs, out_dir, plt),
        _ade_against_speed(runs, out_dir, plt),
        _by_scenario_type(runs, out_dir, plt),
        _worst_grid(runs, out_dir, plt),
    ]
    return [p for p in written if p is not None]


# ---- helpers -------------------------------------------------------------


def _speed_of(sample) -> float | None:
    """Ego speed entering the scored moment, from the recorded future."""
    gt = sample.get("ground_truth")
    if gt is None or len(gt) == 0:
        return None
    return float(np.hypot(*np.asarray(gt, dtype=np.float64)[0, :2])) / INTERVAL_S


def _constant_velocity_ade(sample) -> float | None:
    """What a straight line at the current speed would have scored.

    The honest floor for an open-loop score, and on nuPlan a surprisingly hard
    one: a third of the dataset's tagged moments are a stationary car, where
    predicting no motion is nearly exact.
    """
    gt = sample.get("ground_truth")
    speed = _speed_of(sample)
    if gt is None or speed is None:
        return None
    truth = np.asarray(gt, dtype=np.float64)[:, :2]
    steps = np.arange(1, len(truth) + 1)
    straight = np.stack([speed * INTERVAL_S * steps, np.zeros(len(truth))], axis=1)
    return float(M.displacement_errors(straight, truth, INTERVAL_S, EVAL_HORIZON_S)["ade"])


def _scored(runs):
    """(run, samples with an ade) pairs, skipping runs that recorded none."""
    for run in runs:
        samples = [s for s in run.samples if "ade" in s.get("metrics", {})]
        if samples:
            yield run, samples


def _baseline(runs) -> tuple[list[float], float | None]:
    """Constant-velocity ADEs over the first run's samples, and their mean.

    One run's sample set stands in for all of them: `paddock render` already
    warns when `sequences_sha256` differs, and a baseline drawn over a
    different set than the models were scored on would be the very comparison
    that warning exists to prevent.
    """
    for _, samples in _scored(runs):
        values = [v for v in (_constant_velocity_ade(s) for s in samples) if v is not None]
        return values, (float(np.mean(values)) if values else None)
    return [], None


# ---- figures -------------------------------------------------------------


def _error_distribution(runs, out_dir: Path, plt) -> Path | None:
    """Where each model's error sits, not just its mean -- the tail is the story."""
    pairs = list(_scored(runs))
    if not pairs:
        return None
    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.2))

    for run, samples in pairs:
        ades = [s["metrics"]["ade"] for s in samples]
        colour = palette.color(run.model)
        ax_hist.hist(
            ades,
            bins=20,
            histtype="step",
            lw=1.8,
            color=colour,
            label=f"{run.model}  mean {np.mean(ades):.3f}  n={len(ades)}",
        )
        ordered = np.sort(ades)
        ax_cdf.plot(ordered, np.linspace(0, 1, len(ordered)), lw=1.8, color=colour, label=run.model)

    cv, cv_mean = _baseline(runs)
    if cv:
        for ax in (ax_hist, ax_cdf):
            ax.axvline(cv_mean, color=palette.GROUND_TRUTH, lw=1.4, ls="--")
        ax_hist.plot(
            [],
            [],
            color=palette.GROUND_TRUTH,
            lw=1.4,
            ls="--",
            label=f"constant velocity  mean {cv_mean:.3f}",
        )
        ordered = np.sort(cv)
        ax_cdf.plot(
            ordered,
            np.linspace(0, 1, len(ordered)),
            lw=1.4,
            ls="--",
            color=palette.GROUND_TRUTH,
            label="constant velocity",
        )

    ax_hist.set_xlabel("ADE (m)")
    ax_hist.set_ylabel("samples")
    ax_hist.legend(fontsize=7.5)
    ax_hist.grid(alpha=0.3, lw=0.6)
    ax_cdf.set_xlabel("ADE (m)")
    ax_cdf.set_ylabel("fraction of samples below")
    ax_cdf.legend(fontsize=7.5)
    ax_cdf.grid(alpha=0.3, lw=0.6)
    ax_cdf.axhline(0.5, color="#999", lw=0.8, ls="--")

    fig.suptitle("nuPlan error distribution", fontsize=11)
    fig.tight_layout()
    path = out_dir / "error_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _ade_against_speed(runs, out_dir: Path, plt) -> Path | None:
    """ADE against ego speed -- the figure that explains a nuPlan score.

    A third of nuPlan's tagged moments are a stationary or near-stationary car,
    and a policy latched to a non-zero cruise speed drives off from a red light.
    That shows here as a cloud of high error at low speed which the
    constant-velocity baseline does not have, and it is invisible in a mean.
    """
    pairs = list(_scored(runs))
    if not pairs:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    drawn = False
    for run, samples in pairs:
        xs = [_speed_of(s) for s in samples]
        ys = [s["metrics"]["ade"] for s in samples]
        keep = [(x, y) for x, y in zip(xs, ys, strict=False) if x is not None]
        if not keep:
            continue
        ax.scatter(
            *zip(*keep, strict=False),
            s=34,
            alpha=0.85,
            color=palette.color(run.model),
            label=run.model,
            zorder=3,
        )
        drawn = True

    _, samples = pairs[0]
    cv = [(_speed_of(s), _constant_velocity_ade(s)) for s in samples]
    cv = [(x, y) for x, y in cv if x is not None and y is not None]
    if cv:
        ax.scatter(
            *zip(*cv, strict=False),
            s=30,
            alpha=0.75,
            marker="x",
            color=palette.GROUND_TRUTH,
            label="constant velocity",
            zorder=2,
        )

    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("ego speed entering the scored moment (m/s)")
    ax.set_ylabel("ADE (m)")
    ax.set_title("nuPlan: ADE against ego speed", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.6)
    fig.tight_layout()
    path = out_dir / "ade_vs_speed.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _by_scenario_type(runs, out_dir: Path, plt) -> Path | None:
    """Per-scenario-type ADE. nuPlan's labels are what make this possible."""
    pairs = list(_scored(runs))
    if not pairs:
        return None

    def type_of(sample) -> str:
        return sample.get("native_id", {}).get("scenario_type") or sample.get("split", "?")

    per_model: dict[str, dict[str, float]] = {}
    for run, samples in pairs:
        grouped = collections.defaultdict(list)
        for s in samples:
            grouped[type_of(s)].append(s["metrics"]["ade"])
        per_model[run.model] = {k: float(np.mean(v)) for k, v in grouped.items()}

    _, samples = pairs[0]
    cv_grouped = collections.defaultdict(list)
    counts: collections.Counter = collections.Counter()
    for s in samples:
        value = _constant_velocity_ade(s)
        counts[type_of(s)] += 1
        if value is not None:
            cv_grouped[type_of(s)].append(value)
    cv_mean = {k: float(np.mean(v)) for k, v in cv_grouped.items()}

    # Every type any model scored, not just the first model's -- with partial
    # camera coverage the models can cover different types, and taking one
    # model's keys silently dropped the rest.
    types = {t for values in per_model.values() for t in values}
    first = next(iter(per_model.values()))
    # Ordered by how much the first model loses to the baseline: the top of the
    # chart is then the list of scenario types to go and look at.
    types = sorted(types, key=lambda t: -(first.get(t, 0.0) - cv_mean.get(t, 0.0)))
    if not types:
        return None

    y = np.arange(len(types))
    width = 0.8 / (len(per_model) + 1)
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.42 * len(types) + 1.4)))

    for index, (model, values) in enumerate(per_model.items()):
        # A type this model never scored is left out rather than plotted at
        # zero, which on an error chart reads as a perfect score rather than as
        # no data. Absence shows as a gap in the group.
        present = [t for t in types if t in values]
        ax.barh(
            [y[types.index(t)] + index * width for t in present],
            [values[t] for t in present],
            height=width,
            color=palette.color(model),
            label=model,
        )
    cv_present = [t for t in types if t in cv_mean]
    ax.barh(
        [y[types.index(t)] + len(per_model) * width for t in cv_present],
        [cv_mean[t] for t in cv_present],
        height=width,
        color=palette.GROUND_TRUTH,
        alpha=0.75,
        label="constant velocity",
    )

    # No junction/non-junction marker: nuPlan's tagged set is so
    # junction-dominated (24 of 25 at one-shard coverage) that the split
    # separates nothing, and a legend that groups without discriminating
    # invites a conclusion the data does not support.
    labels = [f"{t}  (n={counts[t]})" for t in types]
    ax.set_yticks(y + 0.4 - width / 2)
    ax.set_yticklabels(labels, fontsize=7.5, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("mean ADE (m)")
    ax.set_title("nuPlan ADE by scenario type", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.6, axis="x")
    fig.tight_layout()
    path = out_dir / "ade_by_scenario_type.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _worst_grid(runs, out_dir: Path, plt, limit: int = 6) -> Path | None:
    """BEV panels for the worst samples, with the type and speed named.

    Worth drawing rather than tabulating: a stationary ground truth beside a
    prediction that drives 15 m forward is one glance, and no column of numbers
    makes the same point as fast.
    """
    pairs = list(_scored(runs))
    if not pairs:
        return None
    run, samples = pairs[0]
    ranked = sorted(samples, key=lambda s: -s["metrics"]["ade"])[:limit]
    if not ranked:
        return None

    columns = min(3, len(ranked))
    rows = (len(ranked) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), squeeze=False)

    for ax, sample in zip(axes.flat, ranked, strict=False):
        prediction = sample.get("prediction")
        if prediction is None:
            continue
        bev.draw(
            ax,
            {run.model: prediction},
            sample.get("ground_truth"),
            labels=bev.metric_labels({run.model: sample["metrics"]["ade"]}),
        )
        ax.legend(fontsize=6.5, loc="best", framealpha=0.88)
        speed = _speed_of(sample)
        scenario = sample.get("native_id", {}).get("scenario_type", "?")
        ax.set_title(f"{scenario}\nv={speed:.1f} m/s" if speed is not None else scenario, fontsize=7.5)

    for ax in list(axes.flat)[len(ranked) :]:
        ax.axis("off")
    fig.suptitle(f"nuPlan: worst {len(ranked)} samples for {run.model}", fontsize=11)
    fig.tight_layout()
    path = out_dir / "worst_samples.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
