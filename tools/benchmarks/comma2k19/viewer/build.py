"""Build the side-by-side viewer bundle for one comma2k19 scene set.

Takes whatever runs of that scene set exist on disk -- none, one, or all six --
and writes a directory the viewer serves:

    {runs}/comma2k19/_viewer/<scene_set>/
      index.json                  generation, models present, colours, scene list
      <slug>/scene.json           per keyframe: every model's plan, the recorded
                                  future, the metrics, the reference state
      <slug>/frames/<k>.jpg       the road frame with every plan drawn on it
      <slug>/plans/<k>.jpg        the same frame with the radar boxes off
      <slug>/bev/<k>.png          those plans against the recorded future, BEV,
                                  over the scored horizon
      <slug>/overlay.mp4          the whole scene at 20 Hz
      _cache/<slug>/<k>.npy       raw decoded keyframe pixels -- see below

Run it as often as you like: a rebuild is incremental and atomic.

    python -m tools.benchmarks.comma2k19.viewer.build --scene-set viewer10
    python -m tools.benchmarks.comma2k19.viewer.build --watch

## Why a raw-pixel cache

Every composited output depends on *which models are present*, so a sixth model
landing invalidates all of them. The expensive input is the decode: comma2k19's
HEVC streams carry no container index, so reaching keyframe 999 means decoding
from frame 0 (`loader.decode_frames` documents this), and a naive rebuild would
pay a full decode pass per model per scene.

So model-independent work is split from model-dependent compositing. Pass A
decodes each segment once and keeps the keyframes' *raw* RGB; pass B reads that
back and composites. Raw rather than pre-drawn, deliberately: plans have to go
through the one `SegmentOverlay.draw` that also lays down the ribbon, the radar,
the HUD and the minimap in the right order. Caching a drawn base and painting
plans on afterwards would be a second copy of that layer order.

`--drop-cache` reclaims the space once the bundle is finished with.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from paddock import render as R
from paddock.adapters.benchmarks.comma2k19 import harness as H
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import overlay as OV
from paddock.registry import Registry
from paddock.results.store import RESULTS_ROOT
from paddock.runner import load_benchmark
from paddock.viz import bev, palette
from tools import paths

#: The order models are run in and reported in, everywhere in the viewer. Fixed
#: rather than dict-insertion or alphabetical so a model finishing does not
#: reshuffle the rows under someone reading them. `pipeline.py` imports this,
#: so the order the runs happen in and the order they are displayed in cannot
#: drift apart.
MODEL_ORDER = (
    "openpilot",
    "visionpilot",
    "qwendrive",
    "alpamayo15",
    "constant_velocity",
    "drivejepa_free",
)

#: The columns the viewer's per-keyframe table shows, out of the ~90 keys
#: `benchmark.yaml` declares. Four, because this table is read at 2 Hz while
#: something plays: `ade` is the headline, `fde@3s` is where it ended up,
#: `cross_track_mae` is the part of the error that is about the path rather than
#: the speed, and `path_length_ratio` says whether the model kept up at all.
TABLE_METRICS = ("ade", "fde@3s", "cross_track_mae", "path_length_ratio")

#: The BEV panel draws each plan only as far as the metrics beside it were
#: computed over -- `H.EVAL_HORIZON_S`, 3 s. openpilot plans to 10 s and
#: `constant_velocity` to 3; drawing every waypoint each model happens to emit
#: would put a 10 s line beside an `ade` that stops at 3 s, and make a model look
#: wrong (or right) over a stretch no number in the table describes. The full
#: plan is still drawn on the road frame, where reach is the thing being shown,
#: and `scene.json` still carries every waypoint.
BEV_HORIZON_S = H.EVAL_HORIZON_S

#: The road frame is composited once per entry here, into a directory of that
#: name, and the viewer switches between them with a checkbox. Two, because the
#: radar boxes answer "what did the car see" and get in the way of "where is
#: each model going" -- and which question is being asked changes minute to
#: minute, while a rebuild takes two. `plans` keeps the recorded-path ribbon and
#: the HUD: the ribbon is what the plans are being compared against by eye, and
#: dropping it would leave nothing to judge them by.
LAYER_SETS = {
    "frames": OV.Layers(),
    "plans": OV.Layers(radar=False),
}

#: Metres of air around the drawn lines in the BEV panel.
BEV_MARGIN_M = 2.0
#: Half the narrowest lateral window the BEV panel will show, metres. A straight
#: scene has every line inside a metre of the origin, and a window scaled to that
#: alone would magnify pose noise into an apparent disagreement.
BEV_MIN_LATERAL_M = 4.0
#: The axes, in inches. Both fixed, rather than the width following the scene's
#: extent: three seconds of driving is ~100 m long and ~20 m wide, so an
#: equal-aspect panel is a strip an inch across on which two plans a metre apart
#: are two pixels apart -- which is the one comparison this panel exists to make.
#: The lateral axis is therefore stretched, and `draw_bev` prints the factor it
#: stretched by, so nobody reads a shape off a plot that is not to scale.
BEV_AX_WIDTH_IN = 4.4
BEV_AX_HEIGHT_IN = 6.0
#: Inches reserved under the axes for the legend, which sits there in two columns
#: rather than in a column beside the panel: seven entries stacked to the right
#: cost more width than the plot itself had.
BEV_LEGEND_IN = 1.05
#: Inches between the axes and the top of that legend -- the tick labels and the
#: x axis label live there.
BEV_LABEL_IN = 0.5
BEV_DPI = 100

#: JPEG quality for the road frames. 85 is visually clean and about 180 KB a
#: frame; 850 frames per rebuild makes the difference between 85 and 95 matter.
JPEG_QUALITY = 85

#: How often `--watch` looks for a new run.
WATCH_POLL_S = 20.0


def slug(sample_key: str) -> str:
    """A path-safe name for one segment. `|` and `/` are both illegal here."""
    return sample_key.replace("|", "_").replace("/", "_")


def bundle_root(cfg, scene_set: str) -> Path:
    """Beside `_compare/` and `_samples/`, outside the repo -- it is gigabytes."""
    return cfg.runs / "comma2k19" / "_viewer" / scene_set


# --------------------------------------------------------------------------
# What has run
# --------------------------------------------------------------------------


def resolve_runs(cfg, scene_set: str) -> list[R.LoadedRun]:
    """The newest run per model that scored *this* scene set, in pipeline order.

    Found by reading the tracked run records rather than by being told run ids,
    so a model that has not finished is simply absent and the caller needs to
    type nothing. These runs are `valid: false` by design -- ten segments is not
    comma2k19 -- so they are fetched by id, which is the one way
    `render.load_runs` will return an invalid run.

    "Newest" means newest *complete*: a run that died half way through is
    recorded `status: partial`, and taking it because it started later would
    replace a model's finished plans with the fraction of them that survived --
    the viewer would appear to lose a model that had, in fact, already run.
    """
    newest: dict[str, tuple[str, str]] = {}
    bench_root = RESULTS_ROOT / "comma2k19"
    if not bench_root.is_dir():
        return []
    for run_json in sorted(bench_root.glob("*/*/run.json")):
        try:
            run = json.loads(run_json.read_text())
        except (OSError, ValueError):
            continue
        if run.get("config", {}).get("scene_set") != scene_set:
            continue
        if run.get("status") != "complete":
            continue
        model = run_json.parent.parent.name
        started = run.get("started_utc") or ""
        if model not in newest or started > newest[model][0]:
            newest[model] = (started, run_json.parent.name)

    loaded = R.load_runs(cfg, "comma2k19", run_ids=[rid for _, rid in newest.values()])
    # Guard against a run id shared by two models: keep only the one this model
    # was actually recorded under.
    loaded = [r for r in loaded if newest.get(r.model, ("", ""))[1] == r.run_id]
    return sorted(loaded, key=lambda r: MODEL_ORDER.index(r.model) if r.model in MODEL_ORDER else 99)


def records_by_sample(run: R.LoadedRun) -> dict[str, dict]:
    """That run's per-sample records, keyed by `sample_id`."""
    return {rec["sample_id"]: rec for rec in run.samples if rec.get("sample_id")}


def sample_id(scene_key: str, keyframe: int) -> str:
    """How the harness names one scored moment. Written once, read three times."""
    return f"comma2k19/{scene_key}/{keyframe}"


def waypoint_dt(model: str) -> float:
    """That model's declared waypoint spacing, seconds.

    Read from the registry rather than assumed, because clipping a plan to a
    horizon means counting waypoints, and counting them at the wrong spacing is
    a silent error: every model in this roster declares 0.5 s today, and a model
    that declared 0.25 s would be cut off at 1.5 s with nothing to show for it.
    """
    spec = Registry.load().models.get(model)
    return float(getattr(spec, "waypoint_dt", H.INTERVAL_S)) if spec else H.INTERVAL_S


def to_horizon(xy: np.ndarray, dt: float, horizon_s: float = BEV_HORIZON_S) -> np.ndarray:
    """The first `horizon_s` of a trajectory laid out every `dt` seconds.

    Waypoint `i` is at `t = (i + 1) * dt` -- a paddock trajectory starts at
    `t = dt`, not at the origin -- so `horizon_s` is the first `horizon_s / dt`
    of them.
    """
    return xy[: max(1, int(round(horizon_s / dt)))]


# --------------------------------------------------------------------------
# What a scene is
# --------------------------------------------------------------------------


@dataclass
class Scene:
    """One segment of the working set, resolved against this site's copy."""

    key: str
    bucket: str
    segment: L.SegmentRef
    keyframes: list[int]
    pose: L.SegmentPose
    ground_truth: dict[int, np.ndarray]

    @property
    def slug(self) -> str:
        return slug(self.key)


def load_scenes(cfg, scene_set: str) -> list[Scene]:
    """Resolve every scene in the set, and compute the recorded future per keyframe.

    The keyframes come from the *harness*, not from any run's records: the
    bundle has to be buildable with no model finished, and a keyframe list
    re-derived here from a different rule would silently stop lining up with
    what was scored. The ground truth likewise comes from
    `loader.future_in_ego_frame`, which is the same call `samples()` makes.
    """
    registry = Registry.load()
    spec = registry.benchmarks.get("comma2k19")
    if spec is None:
        raise SystemExit("benchmark 'comma2k19' is not registered")
    # Asked for through the environment the same way a run asks, so this reads
    # the scene set's own `keyframes_per_segment` rather than a second copy of
    # the precedence rule.
    os.environ[H.ENV_SCENE_SET] = scene_set
    harness = load_benchmark(spec, cfg)
    scene_set_spec = harness.scene_set
    if scene_set_spec is None or scene_set_spec.error:
        raise SystemExit(scene_set_spec.error if scene_set_spec else f"no scene set {scene_set!r}")
    dataset = cfg.datasets.get("comma2k19")
    if dataset is None:
        raise SystemExit("dataset 'comma2k19' is not configured in paddock.toml; see docs/data-setup.md")
    features = _features(scene_set)
    grid = harness.grid

    scenes, missing = [], []
    for key in scene_set_spec.keys:
        chunk, route_id, number = key.split("/")
        segment = L.find_segment(dataset, int(chunk), route_id, int(number))
        if segment is None:
            missing.append(key)
            continue
        pose = L.load_pose(segment.path)
        keyframes = harness.keyframe_indices(len(pose), grid)
        scenes.append(
            Scene(
                key=segment.sample_key,
                bucket=str(features.get(segment.sample_key, {}).get("bucket", "")),
                segment=segment,
                keyframes=keyframes,
                pose=pose,
                # `[:, :2]` for the same reason `samples()` does it: a paddock
                # trajectory is (forward, left), and carrying the pose's third
                # column here would put a z in `scene.json` that the stored
                # predictions beside it do not have.
                ground_truth={
                    k: L.future_in_ego_frame(pose, k, grid.gt_horizons_s)[:, :2] for k in keyframes
                },
            )
        )
    if missing:
        # A warning rather than a refusal: the bundle is a thing to look at, and
        # nine scenes to look at beats none. `preflight` is where a *run* stops
        # over the same fact, because there a short set changes a number.
        print(f"  warning: {len(missing)} scene(s) not present here: {', '.join(missing)}", file=sys.stderr)
    return scenes


def _features(scene_set: str) -> dict:
    """The `features:` block the scene file records, for the bucket labels."""
    import yaml

    path = H.SCENES_DIR / f"{scene_set}.yaml"
    body = yaml.safe_load(path.read_text()) if path.exists() else {}
    return (body or {}).get("features") or {}


# --------------------------------------------------------------------------
# Writing, atomically
# --------------------------------------------------------------------------


def write_bytes(path: Path, data: bytes) -> None:
    """Write through a temp name, so a reader sees before or after, never half.

    The server re-reads the bundle on every request and a rebuild may be running
    while someone is playing a scene; a torn JSON read would break the page with
    no way for the reader to tell why.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, body: dict) -> None:
    write_bytes(path, (json.dumps(body, separators=(",", ":")) + "\n").encode())


# --------------------------------------------------------------------------
# Pass A: decode once, keep the keyframes
# --------------------------------------------------------------------------


def cached_keyframes(cache_dir: Path, keyframes: list[int]) -> bool:
    return all((cache_dir / f"{k}.npy").exists() for k in keyframes)


def decode_pass(scene: Scene, overlay: OV.SegmentOverlay, cache_dir: Path, mp4: Path | None, plans_at) -> int:
    """Decode the segment once: cache the keyframes, optionally write the mp4.

    One pass for both, because the decode is the expensive part and it cannot be
    seeked into. With no mp4 wanted it stops at the last keyframe rather than
    running to the end of the segment.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Stated rather than assumed: `composite` leaves this overlay set to
    # whichever layer set it drew last, and the mp4 is the standalone artefact,
    # so it gets everything.
    overlay.layers = LAYER_SETS["frames"]
    want = set(scene.keyframes)
    limit = len(scene.pose) if mp4 is not None else max(want) + 1

    def frames():
        for index, rgb in enumerate(L.decode_frames(scene.segment.video, limit=limit)):
            if index in want:
                np.save(cache_dir / f"{index}.npy", rgb)
            if mp4 is not None:
                # Plans come from the nearest keyframe at or *before* this frame:
                # drawing a plan on a frame the model had not been shown yet
                # would show it predicting the future it was scored against.
                yield overlay.draw(rgb, index, plans=plans_at(index), note=_video_note(index, scene))

    if mp4 is None:
        for _ in frames():
            pass
        return limit
    # The temp name keeps the `.mp4` suffix: PyAV infers the container format
    # from the extension, and `overlay.mp4.tmp1234` is a `ValueError: Could not
    # determine output format` after the whole decode pass has been paid for.
    tmp = mp4.with_name(f".{mp4.stem}.tmp{os.getpid()}{mp4.suffix}")
    L.encode_frames(frames(), tmp, fps=overlay.fps)
    os.replace(tmp, mp4)
    return limit


def _video_note(index: int, scene: Scene) -> str:
    t = float(scene.pose.times[index] - scene.pose.times[0])
    if index < scene.keyframes[0]:
        return f"t={t:5.1f} s  before the first scored keyframe"
    return f"t={t:5.1f} s  frame {index}"


# --------------------------------------------------------------------------
# Pass B: composite from the cache
# --------------------------------------------------------------------------


def bev_limits(scene: Scene, runs: list[R.LoadedRun]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed (xlim, ylim) for this scene's BEV panel.

    Fixed *within* a scene because `bev.draw` asks for `adjustable="datalim"`,
    which rescales per frame and makes playback jitter -- the lines would move
    because the axis moved. One window for all 85 keyframes, so stepping through
    them compares like with like.

    Over the recorded future **and every plan drawn beside it**: the panel is
    there to show where the models disagree, and a window sized to the recording
    alone pushes the disagreement -- which is the signal -- into the margin or
    off the edge. The cost is that the window moves when a model lands, so the
    same keyframe is not pixel-comparable across two generations of the bundle.
    `built_from` in `scene.json` says which models a panel was drawn with, and
    the staleness rule redraws the scene when that changes.

    Over the scored horizon, like the panel itself: sizing the axes to a 10 s
    future and drawing 3 s of it would leave the plans in the bottom third.
    """
    drawn = [to_horizon(xy, H.INTERVAL_S) for xy in scene.ground_truth.values()]
    for run in runs:
        dt = waypoint_dt(run.model)
        records = records_by_sample(run)
        # This scene's keyframes only: a run holds all ten scenes, and one wide
        # plan in another segment would stretch this panel's axis for no reason.
        for k in scene.keyframes:
            rec = records.get(sample_id(scene.key, k))
            if rec and rec.get("prediction"):
                drawn.append(to_horizon(np.asarray(rec["prediction"], dtype=float), dt))
    xy = np.concatenate(drawn)
    lat, fwd = xy[:, 1], xy[:, 0]
    # The lateral window covers the drawn range rather than being symmetric about
    # the ego. A right-hand curve reaching 13 m right never goes left at all, and
    # a symmetric panel would spend half its width on empty road -- halving the
    # scale at which two plans a metre apart can be told apart.
    lo = min(0.0, float(lat.min())) - BEV_MARGIN_M
    hi = max(0.0, float(lat.max())) + BEV_MARGIN_M
    short = 2 * BEV_MIN_LATERAL_M - (hi - lo)
    if short > 0:
        lo, hi = lo - short / 2, hi + short / 2
    return (lo, hi), (min(0.0, float(fwd.min())) - BEV_MARGIN_M, float(fwd.max()) + BEV_MARGIN_M)


def make_bev_figure():
    """The figure every keyframe of a scene is drawn into, and its title artist.

    One fixed size, laid out with explicit axes rectangles rather than
    `tight_layout`, so every frame of a scene is exactly the same pixels in
    exactly the same places: a legend that grows by a row must not shift the
    road sideways between two frames of a playback. Room is reserved under the
    axes for a legend of every model plus the recording, whether or not they
    have all landed yet.

    The title is a figure text rather than `ax.set_title`, because the axes are
    narrower than the text and a title is clipped at the axes' right edge. It is
    updated per keyframe rather than recreated -- a figure keeps every text ever
    added to it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left, bottom, top, right = 0.75, 0.55, 0.55, 0.25
    fig_w = left + BEV_AX_WIDTH_IN + right
    fig_h = bottom + BEV_LEGEND_IN + BEV_AX_HEIGHT_IN + top
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=BEV_DPI)
    ax = fig.add_axes(
        [
            left / fig_w,
            (bottom + BEV_LEGEND_IN) / fig_h,
            BEV_AX_WIDTH_IN / fig_w,
            BEV_AX_HEIGHT_IN / fig_h,
        ]
    )
    title = fig.text(
        left / fig_w,
        (bottom + BEV_LEGEND_IN + BEV_AX_HEIGHT_IN + 0.08) / fig_h,
        "",
        fontsize=8,
        ha="left",
        va="bottom",
    )
    return fig, ax, title


def draw_bev(fig, ax, title, predictions: dict, ground_truth, labels: dict, xlim, ylim, text: str) -> bytes:
    """One BEV panel as PNG bytes: `viz.bev.draw` plus this scene's fixed axes."""
    import io

    ax.clear()
    bev.draw(ax, predictions, ground_truth=ground_truth, labels=labels)
    # After `bev.draw` the x axis is inverted so left-positive reads leftward.
    # Limits therefore go in descending order, or setting them would quietly
    # flip the panel and mirror every plan in it.
    ax.set_xlim(xlim[1], xlim[0])
    ax.set_ylim(*ylim)
    # Not equal aspect -- see `BEV_AX_WIDTH_IN`. The factor is printed beside the
    # title so the reader knows a 5 m lateral gap is drawn as wide as a 25 m
    # longitudinal one, and does not read a shape off it.
    stretch = ((ylim[1] - ylim[0]) / BEV_AX_HEIGHT_IN) / ((xlim[1] - xlim[0]) / BEV_AX_WIDTH_IN)
    ax.set_aspect("auto", adjustable="box")
    title.set_text(f"{text}  ·  lateral scale {stretch:.0f}× longitudinal")
    if predictions:
        # Under the axes in two columns: seven entries down the right-hand side
        # took more width than the plot they annotate.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -BEV_LABEL_IN / BEV_AX_HEIGHT_IN),
            ncol=2,
            fontsize=8,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.6,
        )
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def plan_label(model: str, metrics: dict) -> str:
    ade = metrics.get("ade")
    return f"{model}  {ade:.2f} m" if isinstance(ade, (int, float)) and ade == ade else model


def composite(scene: Scene, overlay: OV.SegmentOverlay, runs: list[R.LoadedRun], out: Path, limits) -> dict:
    """Redraw this scene's frames and BEV panels for the models now present.

    Returns the `scene.json` body. Reads the raw keyframes back from the cache,
    so this costs no decode and can be run again every time a model lands.
    """
    from PIL import Image

    by_model = {run.model: records_by_sample(run) for run in runs}
    spacing = {run.model: waypoint_dt(run.model) for run in runs}
    xlim, ylim = limits
    fig, ax, heading = make_bev_figure()
    bev_dir = out / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    for layer_set in LAYER_SETS:
        (out / layer_set).mkdir(parents=True, exist_ok=True)
    cache_dir = out.parent / "_cache" / scene.slug

    rows = []
    try:
        for k in scene.keyframes:
            sid = sample_id(scene.key, k)
            present = {m: recs[sid] for m, recs in by_model.items() if sid in recs}
            predictions = {
                m: np.asarray(r["prediction"], dtype=float) for m, r in present.items() if r.get("prediction")
            }
            gt = scene.ground_truth[k]

            plans = [
                OV.Plan.for_model(m, xy, label=plan_label(m, present[m].get("metrics") or {}))
                for m, xy in predictions.items()
            ]
            rgb = np.load(cache_dir / f"{k}.npy")
            note = f"keyframe {k}  {len(plans)} model(s)"
            # Once per layer set, from the same cached pixels: the radar boxes
            # are what a reader turns off to see the plans, and a toggle in the
            # page beats a rebuild with a different flag.
            for name, layers in LAYER_SETS.items():
                overlay.layers = layers
                image = Image.fromarray(overlay.draw(rgb, k, plans=plans, note=note))
                write_bytes(out / name / f"{k}.jpg", _jpeg(image))

            labels = {m: plan_label(m, present[m].get("metrics") or {}) for m in predictions}
            text = f"{scene.key}\nkeyframe {k}  ·  first {BEV_HORIZON_S:g} s, the scored horizon"
            drawn = {m: to_horizon(xy, spacing[m]) for m, xy in predictions.items()}
            panel = draw_bev(fig, ax, heading, drawn, to_horizon(gt, H.INTERVAL_S), labels, xlim, ylim, text)
            write_bytes(bev_dir / f"{k}.png", panel)

            rows.append(
                {
                    "keyframe": k,
                    "sample_id": sid,
                    "t": round(float(scene.pose.times[k] - scene.pose.times[0]), 2),
                    "ground_truth": np.asarray(gt, dtype=float).round(3).tolist(),
                    "reference": _reference(present),
                    "models": {
                        m: {
                            "prediction": predictions[m].round(3).tolist(),
                            "metrics": {n: _num(present[m].get("metrics", {}).get(n)) for n in TABLE_METRICS},
                        }
                        for m in predictions
                    },
                }
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return {
        "scene": scene.key,
        "slug": scene.slug,
        "bucket": scene.bucket,
        "fps": round(overlay.fps, 4),
        "n_frames": len(scene.pose),
        "keyframes": scene.keyframes,
        "models": [run.model for run in runs],
        "built_from": {run.model: run.run_id for run in runs},
        "bev_xlim": list(xlim),
        "bev_ylim": list(ylim),
        "bev_horizon_s": BEV_HORIZON_S,
        "waypoint_dt": spacing,
        "layer_sets": list(LAYER_SETS),
        "frames": rows,
    }


def _jpeg(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _num(value):
    """JSON has no NaN. A metric that did not apply becomes null, not a string."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else round(value, 4)


def _reference(present: dict) -> dict:
    """The recorded ego state at this keyframe, as the runs recorded it.

    Taken from whichever run is present rather than recomputed: it is part of
    the sample the models were scored on, and a second derivation here could
    disagree with the one the metrics used.
    """
    for record in present.values():
        reference = (record.get("native_id") or {}).get("reference")
        if reference:
            return {k: _num(v) if isinstance(v, (int, float)) else v for k, v in reference.items()}
    return {}


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def plans_for_video(scene: Scene, runs: list[R.LoadedRun]):
    """`index -> plans`, using the nearest keyframe at or before `index`.

    Before the first keyframe there are no plans: the model had not been shown
    enough history to have one, and inventing a plan there would misrepresent
    both the model and the protocol.
    """
    by_model = {run.model: records_by_sample(run) for run in runs}
    cache: dict[int, list] = {}
    for k in scene.keyframes:
        sid = sample_id(scene.key, k)
        cache[k] = [
            OV.Plan.for_model(m, recs[sid]["prediction"], label=plan_label(m, recs[sid].get("metrics") or {}))
            for m, recs in by_model.items()
            if sid in recs and recs[sid].get("prediction")
        ]
    keyframes = scene.keyframes

    def at(index: int):
        if index < keyframes[0]:
            return ()
        pos = int(np.searchsorted(keyframes, index, side="right")) - 1
        return cache[keyframes[pos]]

    return at


def build(cfg, scene_set: str, video: str = "final", verbose: bool = True, force: bool = False) -> dict:
    """Build or refresh the whole bundle. Returns the `index.json` body.

    `force` recomposites every scene whether or not the staleness rule below
    fires. That rule watches the *inputs* -- which models, which keyframes,
    which axes -- and cannot see a change to the drawing code itself, so
    editing how a panel looks otherwise leaves the old PNGs on disk.
    """
    out_root = bundle_root(cfg, scene_set)
    out_root.mkdir(parents=True, exist_ok=True)
    runs = resolve_runs(cfg, scene_set)
    scenes = load_scenes(cfg, scene_set)
    generation = _next_generation(out_root)
    current = {run.model: run.run_id for run in runs}
    complete = len(runs) >= len(MODEL_ORDER)

    if verbose:
        have = ", ".join(f"{r.model} ({len(r.samples)} samples)" for r in runs) or "none yet"
        print(f"generation {generation}: {len(scenes)} scene(s), models: {have}", file=sys.stderr)

    entries = []
    for scene in scenes:
        began = time.time()
        out = out_root / scene.slug
        cache_dir = out_root / "_cache" / scene.slug
        overlay = OV.SegmentOverlay(scene.segment.path)
        mp4 = out / "overlay.mp4"

        previous = _previous(out / "scene.json")
        # Which models the mp4 on disk was drawn with, rather than whether one
        # exists: an mp4 built from five models is stale once the sixth lands,
        # and `mp4.exists()` cannot tell the difference.
        video_from = previous.get("video_built_from") if mp4.exists() else None
        first_build = not cached_keyframes(cache_dir, scene.keyframes)
        want_video = video == "always" or (video == "final" and complete and video_from != current)
        # The mp4 comes free on the first pass whatever the policy, because that
        # pass is already decoding every frame of the segment.
        target = mp4 if (want_video or (first_build and video != "never")) else None
        if first_build or target is not None:
            decode_pass(scene, overlay, cache_dir, target, plans_for_video(scene, runs))
            if target is not None:
                video_from = dict(current)

        # The panel geometry is part of what a composited output is, not just the
        # models on it: change the limits rule and the PNGs on disk are stale
        # even though the same models drew them.
        limits = bev_limits(scene, runs)
        stale = (
            force
            or previous.get("built_from") != current
            or previous.get("keyframes") != scene.keyframes
            or previous.get("bev_xlim") != list(limits[0])
            or previous.get("layer_sets") != list(LAYER_SETS)
        )
        body = composite(scene, overlay, runs, out, limits) if stale else previous
        body["video_built_from"] = video_from
        write_json(out / "scene.json", body)
        entries.append(
            {
                "slug": scene.slug,
                "key": scene.key,
                "bucket": scene.bucket,
                "n_keyframes": len(scene.keyframes),
                "keyframes": scene.keyframes,
                "video": f"{scene.slug}/overlay.mp4" if mp4.exists() else None,
                "models": body.get("models", []),
            }
        )
        if verbose:
            print(
                f"  {scene.slug}  {len(scene.keyframes)} keyframes in {time.time() - began:.1f} s",
                file=sys.stderr,
            )

    index = {
        "scene_set": scene_set,
        "generation": generation,
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pipeline_order": list(MODEL_ORDER),
        "table_metrics": list(TABLE_METRICS),
        "layer_sets": list(LAYER_SETS),
        "bev_horizon_s": BEV_HORIZON_S,
        "models": [
            {
                "name": run.model,
                "color": palette.color(run.model),
                "run_id": run.run_id,
                "n_samples": len(run.samples),
            }
            for run in runs
        ],
        "waiting_on": [m for m in MODEL_ORDER if m not in current],
        "ground_truth_color": palette.GROUND_TRUTH,
        "scenes": entries,
    }
    write_json(out_root / "index.json", index)
    if verbose:
        print(f"wrote {out_root / 'index.json'}", file=sys.stderr)
    return index


def read_index(cfg, scene_set: str) -> dict:
    """The bundle's `index.json` as it stands, or `{}` if nothing is built yet."""
    return _previous(bundle_root(cfg, scene_set) / "index.json")


def _previous(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _next_generation(out_root: Path) -> int:
    return int(_previous(out_root / "index.json").get("generation", 0)) + 1


def drop_cache(cfg, scene_set: str) -> None:
    cache = bundle_root(cfg, scene_set) / "_cache"
    if cache.exists():
        shutil.rmtree(cache)
        print(f"removed {cache}", file=sys.stderr)


# --------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-set", default="viewer10")
    ap.add_argument(
        "--video",
        choices=("never", "final", "always"),
        default="final",
        help="rewrite the mp4s: never, once every model has landed (default), or every build",
    )
    ap.add_argument(
        "--watch", action="store_true", help="rebuild whenever a new run of this scene set appears"
    )
    ap.add_argument(
        "--force", action="store_true", help="recomposite every scene even if its inputs are unchanged"
    )
    ap.add_argument("--drop-cache", action="store_true", help="delete the raw keyframe cache and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = paths.config()
    if args.drop_cache:
        drop_cache(cfg, args.scene_set)
        return 0
    if not args.watch:
        build(cfg, args.scene_set, video=args.video, force=args.force)
        return 0

    seen: dict[str, str] = {}
    while True:
        current = {r.model: r.run_id for r in resolve_runs(cfg, args.scene_set)}
        if current != seen:
            seen = current
            build(cfg, args.scene_set, video=args.video, force=args.force)
        time.sleep(WATCH_POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
