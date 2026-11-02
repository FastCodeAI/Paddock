"""Comparison figures for comma2k19 runs, and one sample as video.

Takes runs from any models and draws them together, scoped to the benchmark
rather than to any model -- the same contract as `kitscenes/render.py`:

    def render(runs, out_dir, cfg) -> list[Path]

and the optional per-sample hook `paddock render --sample` dispatches to:

    def render_sample(runs, sample_id, out_dir, cfg) -> list[Path]

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

Alongside them, `report.py` writes `report.md`: every declared metric in groups,
which is the artefact people actually quote from. The figures are linked from
it, so the directory reads as one document rather than a pile of PNGs.

The video is a different question from the figures: a distribution says *how
often* a model is wrong, and only a picture says *how*. It runs the history
window the model was actually shown and stops on the scored keyframe, where the
recorded future and every run's stored plan are drawn together and held. The
numbers ride alongside it in `dashboard.py` rather than on the road.

Colour comes from `paddock.viz.palette`, so a model keeps its colour across
every figure in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paddock.adapters.benchmarks.comma2k19 import dashboard as DB
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import overlay as OV
from paddock.adapters.benchmarks.comma2k19 import report as RP
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
    figures = [p for p in written if p is not None]
    # The report last, so it can link the figures that were actually produced
    # rather than the ones this module hoped to produce.
    return [*figures, RP.write(runs, out_dir, figures)]


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


# --------------------------------------------------------------------------
# One sample, as video
# --------------------------------------------------------------------------

#: Seconds the keyframe is held at the end of the clip. The plans are only drawn
#: there -- a plan is a statement made at one moment, and painting it over the
#: history frames would show the model predicting things it had not yet seen --
#: so without a hold the interesting frame flashes past in one frame time.
HOLD_S = 2.5


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """One scored sample as a dashboard .mp4: the road, and the instruments.

    The clip runs the frames the model was actually shown, at the rate it was
    shown them, and stops on the scored keyframe with the recorded future and
    every run's stored plan drawn together. So the thing being compared is the
    thing that was scored, not a nearby frame that looks like it.

    The image carries only what is spatial and `dashboard` carries everything
    numeric beside it, on a shared time axis. See that module for why the HUD
    left the road.

    Raises `LookupError` if no loaded run holds `sample_id`, or if the segment
    behind it is not present at this site -- `paddock.render` turns that into a
    message and a non-zero exit.
    """
    found = [(run, rec) for run, rec in ((r, _record(r, sample_id)) for r in runs) if rec is not None]
    if not found:
        raise LookupError(_no_such_sample(runs, sample_id))

    # Runs at different input rates saw different windows of the same keyframe.
    # Play the densest one: it is the most road per second of clip, and the
    # keyframe -- the only frame the plans are drawn on -- is the same either way.
    native = min(found, key=lambda pair: int(pair[1]["native_id"].get("stride", 1)))[1]["native_id"]
    dataset = cfg.datasets.get("comma2k19")
    if dataset is None:
        raise LookupError("dataset 'comma2k19' is not configured here; see docs/data-setup.md")
    segment = L.find_segment(dataset, int(native["chunk"]), str(native["route_id"]), int(native["segment"]))
    if segment is None:
        raise LookupError(
            f"segment {native['chunk']}/{native['route_id']}/{native['segment']} is not under any "
            "configured comma2k19 tier, so the frames behind this sample cannot be decoded here"
        )

    keyframe = int(native["keyframe_index"])
    stride = int(native.get("stride", 1))
    first = int(native.get("first_frame_index", keyframe))
    fps = float(native.get("input_hz") or L.SRC_HZ)
    plans = [
        OV.Plan.for_model(
            run.model, rec["prediction"], label=_plan_label(run, rec), rpy=_calibration(rec["native_id"])
        )
        for run, rec in found
        if rec.get("prediction")
    ]
    scored = [_scored(run, rec) for run, rec in found if rec.get("prediction")]

    # The HUD, the legend and the minimap are the dashboard's job now; leaving
    # them on would print every number twice, once over moving road.
    overlay = OV.SegmentOverlay(
        segment.path, layers=OV.Layers(hud=False, minimap=False), rpy=_calibration(native)
    )
    dash = DB.Dashboard(
        overlay,
        scored,
        keyframe=keyframe,
        window_s=(keyframe - first) / L.SRC_HZ,
        note=f"history at {fps:g} Hz, stride {stride}; plans drawn on the scored frame only",
        legend=overlay.legend_entries(plans),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (sample_id.removeprefix("comma2k19/").replace("|", "_").replace("/", "_") + ".mp4")
    window = range(first, keyframe + 1, stride)
    try:
        drawn = _overlaid(overlay, segment, window, keyframe, plans, fps)
        L.encode_frames(DB.frames(dash, drawn), path, fps=fps)
    finally:
        dash.close()
    return [path]


def _scored(run, record) -> DB.Scored:
    """One run's stored answer, in the shape the panels read.

    `waypoint_dt` comes from the registry because the per-sample record does not
    carry it, and falls back to paddock's own 0.5 s for a model that is no
    longer registered -- a run outlives the roster, and a missing model should
    cost a correct time axis, not the whole video.
    """
    from paddock.registry import load_models

    spec = load_models().get(run.model)
    return DB.Scored(
        model=run.model,
        plan=np.asarray(record["prediction"], dtype=np.float64),
        ground_truth=record.get("ground_truth"),
        metrics=record.get("metrics") or {},
        dt=float(getattr(spec, "waypoint_dt", 0.5) or 0.5),
    )


def _overlaid(overlay, segment, window, keyframe: int, plans, fps: float):
    """The window's frames, drawn, as `(index, rgb, show_plans)`.

    Decoding starts at frame 0 however late the keyframe is: the streams carry
    no container index, so there is nothing to seek with -- the same constraint
    `loader.decode_frames` documents for scoring.

    The keyframe is repeated for `HOLD_S`; the dashboard is composed per
    repetition rather than once, because it is cheap next to the decode and
    composing once would freeze the playhead a frame early.
    """
    wanted = set(window)
    for index, rgb in enumerate(L.decode_frames(segment.video, limit=keyframe + 1)):
        if index not in wanted:
            continue
        if index != keyframe:
            yield index, overlay.draw(rgb, index), False
            continue
        drawn = overlay.draw(rgb, index, plans=plans)
        for _ in range(max(1, int(round(HOLD_S * fps)))):
            yield index, drawn, True


def _record(run, sample_id: str) -> dict | None:
    """The sample's stored record in one run, by full id or by its tail.

    The tail is accepted because the full id starts with the benchmark name that
    `--bench` already carried, and nobody types it twice.
    """
    for sample in run.samples:
        stored = str(sample.get("sample_id", ""))
        if stored == sample_id or stored.removeprefix("comma2k19/") == sample_id.removeprefix("comma2k19/"):
            return sample
    return None


def _plan_label(run, record) -> str:
    """Model name plus what it scored, so the picture carries its own number."""
    metrics = record.get("metrics") or {}
    scored = [
        f"{key} {metrics[key]:.2f}"
        for key in ("ade", "fde@3s")
        if isinstance(metrics.get(key), (int, float)) and np.isfinite(metrics[key])
    ]
    return f"{run.model} plan" + ("   " + "  ".join(scored) if scored else "")


def _calibration(native: dict) -> np.ndarray | None:
    """The mount rotation the harness handed the model for this sample, radians.

    It defines the frame the plan is in, so it is the rotation to draw the plan
    back with -- per run, since runs are drawn together and each was handed its
    own -- and reusing it spares `SegmentOverlay` a pooled estimate over the
    whole route. None for a record from before the harness stored it; the
    overlay then estimates, as it always did.
    """
    rpy = native.get("rpy_calib_deg")
    return None if rpy is None else np.radians(np.asarray(rpy, dtype=np.float64))


def _no_such_sample(runs, sample_id: str) -> str:
    examples = [str(s.get("sample_id")) for run in runs for s in run.samples[:2]][:3]
    total = sum(len(run.samples) for run in runs)
    if not total:
        return (
            f"no per-sample records loaded, so {sample_id!r} cannot be drawn. They live under "
            "[roots] runs rather than in the repo, and may have been cleaned up -- re-run the "
            "benchmark, or point --run-id at a run whose artifacts are still there."
        )
    return f"no loaded run holds sample {sample_id!r} ({total} records loaded, e.g. {', '.join(examples)})"
