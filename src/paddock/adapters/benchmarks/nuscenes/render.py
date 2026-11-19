"""nuScenes scenes as videos: which ones the website shows, and how one is drawn.

Every model is shown the same front camera, so a video is one view (`viz.moment`):
every scored keyframe of a scene in order, 2 a second as recorded, each with every
model's plan for it and the recorded path drawn on the road. `showcase` picks the
moments; `render_sample` draws the scene a moment is in
(`paddock render --bench nuscenes --sample nuscenes/val/scene-0103/12`); `render`
draws the ones `showcase` picks. All read what a run keeps: the plan and the
score in each record of samples.jsonl.
"""

from __future__ import annotations

import re
from pathlib import Path

from paddock.adapters.benchmarks.nuscenes.harness import HISTORY_KEYFRAMES, STEP_S
from paddock.policy.protocol import Command
from paddock.viz import moment, palette, video

#: The results table's nuScenes column: UniAD's L2, averaged over 1, 2 and 3 s.
SCORE = "l2_uniad_avg"
KINDS = {Command.RIGHT: "right", Command.LEFT: "left", Command.STRAIGHT: "straight"}
NOUNS = {"left": "left turn", "right": "right turn", "straight": "moment driving straight on"}
_SAMPLE = re.compile(r"nuscenes/val/(?P<scene>scene-\d+)/(?P<index>\d+)")


def render(runs, out_dir: Path, cfg) -> list[Path]:
    """The moments `showcase` picks, as videos."""
    written: list[Path] = []
    for pick in showcase(runs, cfg=cfg):
        written += render_sample(runs, pick["sample"], out_dir, cfg)
    return written


def showcase(runs, n: int = 4, cfg=None) -> list[dict]:
    """The hardest left turn, right turn and moment driving straight on, then the
    moment the models disagree on most; one per scene. See `viz.moment.choose`.

    The kind is the command the model was given, which the harness reads off the
    recorded future (VAD's rule). The table's floor, constant velocity, is drawn
    but not counted: it fails every turn, so it would pick only turns.

    A keyframe in a scene's first 2 s is not picked: there the models have less
    than their usual past. AutoE2E, which reads 6.4 s of past, averages 5.9 m of
    error on a scene's first keyframe and 0.7 m from the fifth on, so without the
    skip all four picks were first keyframes. Returns [{"sample", "title", "caption"}].
    """
    scores: dict[str, dict[str, float]] = {}
    kinds: dict[str, str] = {}
    places: dict[str, str] = {}
    for run in runs:
        if getattr(run, "floor", False):
            continue
        for record in run.samples:
            value = (record.get("metrics") or {}).get(SCORE)
            native = record.get("native_id") or {}
            if value is None or not record.get("prediction"):
                continue
            if native.get("index_in_scene", HISTORY_KEYFRAMES) < HISTORY_KEYFRAMES:
                continue
            sample = record["sample_id"]
            scores.setdefault(sample, {})[run.model] = float(value)
            kinds[sample] = KINDS.get(native.get("command"), "")
            places[sample] = native.get("scene", sample)
    picked = moment.choose(scores, kinds, places, n, higher_is_better=False)
    if not picked:
        raise LookupError(
            "no nuScenes keyframe was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )
    return [
        {
            "sample": p["sample"],
            "title": f"{moment.KIND_NAMES.get(p['kind'], 'A moment')}, {places[p['sample']]}",
            "caption": _caption(p, int(p["sample"].rsplit("/", 1)[1]) * STEP_S),
        }
        for p in picked
    ]


def _caption(pick: dict, at_s: float) -> str:
    """What the clip was picked for, and where in it (the clip starts at the scene's first keyframe)."""
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
    """The scene `sample_id` is in, as a video: every scored keyframe in order, each
    with every run's plan for it. The poster is `sample_id`'s keyframe.

    Writes `<scene>-<keyframe>.mp4` and a `.jpg` poster.
    """
    m = _SAMPLE.fullmatch(sample_id)
    if not m:
        raise LookupError(
            f"{sample_id!r} is not a nuScenes sample; they look like nuscenes/val/scene-0103/12"
        )
    by_run = [(run, {r["sample_id"]: r for r in run.samples if r.get("prediction")}) for run in runs]
    if not any(sample_id in records for _, records in by_run):
        raise LookupError(
            f"no run given holds a plan for {sample_id}; looked in "
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )
    scene, picked = m["scene"], int(m["index"])
    steps = []
    for frame in _scene(_harness(cfg), scene):
        key = f"nuscenes/val/{scene}/{frame.index_in_scene}"
        found = [(run, records[key]) for run, records in by_run if key in records]
        if not found:  # the last keyframes have no 3 s of future to score against
            continue
        front = frame.cameras["CAM_FRONT"]
        steps.append(
            moment.Step(
                Path(front["path"]),
                moment.Camera(front["intrinsic"], front["cam2ego"]),
                [
                    moment.Plan(run.label, palette.color(run.model), rec["prediction"], _score(rec))
                    for run, rec in found
                ],
                found[0][1].get("ground_truth"),
                frame.index_in_scene * STEP_S,
                "the keyframe this clip was picked for" if frame.index_in_scene == picked else "",
            )
        )
    ffmpeg = video.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("drawing a video needs ffmpeg: set PADDOCK_FFMPEG or put one on PATH")
    return moment.write(
        Path(out_dir) / f"{scene}-{picked}",
        steps,
        step_s=STEP_S,
        title=f"nuScenes · {scene}",
        score_name="error, m (lower is better)",
        ffmpeg=ffmpeg,
    )


def _score(record: dict) -> str:
    value = (record.get("metrics") or {}).get(SCORE)
    return "" if value is None else f"{value:.2f} m"


def _harness(cfg):
    from paddock.adapters.benchmarks.nuscenes.harness import NuScenesBenchmark
    from paddock.registry import Registry

    return NuScenesBenchmark(Registry.load().benchmarks["nuscenes"], cfg)


def _scene(harness, scene: str) -> list:
    """The scene's keyframes in order, with camera paths and calibration, from the harness's own index."""
    for frames in harness._index():
        if frames and frames[0].scene == scene:
            return sorted(frames, key=lambda f: f.index_in_scene)
    raise LookupError(f"{scene} is not in the nuScenes val index")
