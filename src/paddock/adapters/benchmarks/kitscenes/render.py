"""Comparison figures for KITScenes runs.

Takes runs from any models and draws them together, because that is what
debugging an open-loop result actually looks like: which model is wrong, where,
and by how much.

Scoped to the benchmark rather than to any model. Before paddock this existed
three times -- a three-way plot under openpilot, a trajectory plot under
Drive-JEPA, a per-sample sheet under Drive-JEPA again -- each able to draw only
the model it lived beside.

The website's videos are drawn here too (`viz.moment`): `showcase` picks the
moments and `render_sample` draws the scene a moment is in
(`paddock render --bench kitscenes --sample kitscenes/val/<scene>/40`). The sample
list scores one to six keyframes a scene, 9 s apart, too few to play as a drive.
So each scored keyframe is shown with every model's plan, and then the recorded
drive plays on for the 4 s the plans cover, 10 frames a second, with each plan
and the recorded path held where they lie on the road: the car drives the
recorded path past every model's plan.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from paddock.viz import bev, moment, palette, video

#: The results table's KITScenes column.
SCORE = "ade"
#: How long the drive plays on after a scored keyframe: the recorded path's 8 poses, 0.5 s apart.
HORIZON_S = 4.0
#: The scored keyframe itself stays this long, so its plans can be read before the car moves.
PAUSE_S = 1.0
NOUNS = {"left": "left turn", "right": "right turn", "straight": "moment driving straight on"}
_SAMPLE = re.compile(r"kitscenes/(?P<split>[^/]+)/(?P<scene>[^/]+)/(?P<index>\d+)")


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


# ---------------------------------------------------------------------------
# videos for the website
# ---------------------------------------------------------------------------


def showcase(runs, n: int = 4, cfg=None) -> list[dict]:
    """The hardest left turn, right turn and moment driving straight on, then the
    moment the models disagree on most; one per scene. See `viz.moment.choose`.

    The kind is read off the recorded path by VAD's rule (`moment.kind_from_path`):
    the models are given no turn command here by default, so it says what the
    driver did, not what a model was told. The table's floor, constant velocity,
    is drawn but not counted: it fails every turn, so it would pick only turns.
    Returns [{"sample", "title", "caption"}].
    """
    scores: dict[str, dict[str, float]] = {}
    kinds: dict[str, str] = {}
    places: dict[str, str] = {}
    for run in runs:
        if getattr(run, "floor", False):
            continue
        for record in run.samples:
            value = (record.get("metrics") or {}).get(SCORE)
            if value is None or not record.get("prediction") or not record.get("ground_truth"):
                continue
            sample = record["sample_id"]
            scores.setdefault(sample, {})[run.model] = float(value)
            kinds[sample] = moment.kind_from_path(record["ground_truth"])
            places[sample] = (record.get("native_id") or {}).get("scene_id", sample)
    picked = moment.choose(scores, kinds, places, n, higher_is_better=False)
    if not picked:
        raise LookupError(
            "no KITScenes sample was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )
    return [
        {
            "sample": p["sample"],
            "title": f"{moment.KIND_NAMES.get(p['kind'], 'A moment')}, scene {places[p['sample']][:8]}",
            "caption": _caption(p, int(p["sample"].rsplit("/", 1)[1]) / 10.0),
        }
        for p in picked
    ]


def _caption(pick: dict, at_s: float) -> str:
    """What the clip was picked for, and when (the clip's clock is the scene's, 10 frames a second)."""
    if pick["why"] == "hardest":
        return (
            f"The hardest {NOUNS.get(pick['kind'], 'moment')}, at {at_s:.1f} s: on average the models' "
            f"plans are {pick['mean']:.1f} m from the path the car took."
        )
    return (
        f"Where the models disagree most, at {at_s:.1f} s: their errors run from "
        f"{pick['low']:.1f} m to {pick['high']:.1f} m."
    )


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """The scene `sample_id` is in, as a video. For each keyframe of it a run scored,
    in order: that keyframe with every run's plan, held `PAUSE_S`, then the next
    `HORIZON_S` of the recorded drive with those plans held on the road. The poster
    is `sample_id`'s keyframe.

    Writes `<scene>-<keyframe>.mp4` and a `.jpg` poster.
    """
    from paddock.adapters.benchmarks.kitscenes import loader as L
    from paddock.adapters.benchmarks.kitscenes.harness import SRC_HZ, KitScenesBenchmark, _read_calibration
    from paddock.registry import Registry

    m = _SAMPLE.fullmatch(sample_id)
    if not m:
        raise LookupError(f"{sample_id!r} is not a KITScenes sample; they look like kitscenes/val/<scene>/40")
    by_run = [(run, {r["sample_id"]: r for r in run.samples if r.get("prediction")}) for run in runs]
    if not any(sample_id in records for _, records in by_run):
        raise LookupError(
            f"no run given holds a plan for {sample_id}; looked in "
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )
    harness = KitScenesBenchmark(Registry.load().benchmarks["kitscenes"], cfg)
    row = next((r for r in harness.coverage()[0] if harness._sample_id(r) == sample_id), None)
    if row is None:
        raise LookupError(f"{sample_id} is not in this site's KITScenes sample list")
    # The camera the models were shown as CAM_FRONT: the ring's front unless the site chose the base one.
    sensor = next(s for s, camera in harness._sensors().items() if camera == "CAM_FRONT")
    scene = harness._sensor_scene(row["scene_id"], row["split_dir"], sensor)
    if scene is None:
        raise FileNotFoundError(
            f"scene {row['scene_id']} has no {sensor} under any configured kitscenes tier"
        )
    scene_path = harness._scene_path(row["scene_id"], row["split_dir"])
    camera2image, camera2ego = _read_calibration(
        scene_path, {sensor: "CAM_FRONT"}, ground_offset=harness._ground_offset()
    )
    # `_read_calibration` puts the ego origin on the road, so the road is at z = 0.
    camera = moment.Camera(camera2image["CAM_FRONT"], camera2ego["CAM_FRONT"])
    poses = L.load_poses(scene_path)
    picked, prefix = int(m["index"]), sample_id.rsplit("/", 1)[0] + "/"
    keyframes = sorted(
        {int(k.rsplit("/", 1)[1]) for _, records in by_run for k in records if k.startswith(prefix)}
    )
    steps = []
    for k in keyframes:
        found = [(run, records[f"{prefix}{k}"]) for run, records in by_run if f"{prefix}{k}" in records]
        truth = found[0][1].get("ground_truth")
        made_here = "the frame this clip was picked for" if k == picked else "plans made at this frame"
        scored = moment.Step(
            L.frame_path(scene, k, sensor),
            camera,
            [
                moment.Plan(run.label, palette.color(run.model), rec["prediction"], _score(rec))
                for run, rec in found
            ],
            truth,
            k / SRC_HZ,
            made_here,
            poster=k == picked,
        )
        steps += [scored] * round(PAUSE_S * SRC_HZ)
        for f in range(k + 1, min(k + round(HORIZON_S * SRC_HZ), len(poses) - 1) + 1):
            picture = L.frame_path(scene, f, sensor)
            if not picture.exists():
                break
            steps.append(
                moment.Step(
                    picture,
                    camera,
                    [
                        moment.Plan(
                            run.label,
                            palette.color(run.model),
                            _held(rec["prediction"], poses[k], poses[f]),
                            _score(rec),
                        )
                        for run, rec in found
                    ],
                    None if truth is None else _held(truth, poses[k], poses[f]),
                    f / SRC_HZ,
                    f"the plans made at {k / SRC_HZ:.1f} s, held on the road",
                    from_car=False,
                )
            )
    ffmpeg = video.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("drawing a video needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
    return moment.write(
        Path(out_dir) / f"{row['scene_id'][:8]}-{picked}",
        steps,
        step_s=1 / SRC_HZ,
        title=f"KITScenes · scene {row['scene_id'][:8]}",
        score_name="error, m (lower is better)",
        ffmpeg=ffmpeg,
    )


def _held(xy, made_at: np.ndarray, now: np.ndarray) -> np.ndarray:
    """A path in the ego frame of pose `made_at` (x, y, yaw), moved into that of pose `now`,
    starting from where the car was at `made_at`: where the path lies on the road as seen now."""
    from paddock.metrics.pose import to_relative_poses

    pts = np.vstack([np.zeros((1, 2)), np.asarray(xy, dtype=np.float64)[:, :2]])
    c, s = np.cos(made_at[2]), np.sin(made_at[2])
    world = made_at[:2] + pts @ np.array([[c, s], [-s, c]])
    return to_relative_poses(
        np.asarray(now, dtype=np.float64), np.column_stack([world, np.zeros(len(world))])
    )[:, :2]


def _score(record: dict) -> str:
    value = (record.get("metrics") or {}).get(SCORE)
    return "" if value is None else f"{value:.2f} m"


def _record(run, sample_id: str) -> dict | None:
    return next((r for r in run.samples if r.get("sample_id") == sample_id), None)
