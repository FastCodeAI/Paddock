"""Comparison figures for AlpaSim runs, one scene drawn for every model, and the website's scenes.

`render` draws every model's score on every scene in one grid (`paddock render --bench
alpasim`): a closed-loop score is one number per scene, and the mean hides which scenes a
model loses.

AlpaSim already draws each drive: with `[benchmarks.alpasim] video = "true"` every scene
folder holds one video of the wide front camera with the model's plan (pink or orange) and the route
(green) drawn on, under a map and a metrics table. `render_sample` cuts the camera picture
out of each model's video and lays the models side by side on one clock
(`paddock render --bench alpasim --sample alpasim/clipgt-<scene id>`). `showcase` picks the
scenes the website shows.

Both read only what a run already keeps: that video, the scene's record in samples.jsonl,
and the controller's log of the car's speed. Nothing here needs AlpaSim's own environment.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.viz import palette

CAMERA = "camera_front_wide_120fov"  # every model is sent it, and it is the one AlpaSim's video shows
WIDTH = 1920  # the whole video; tiles are 640 wide in a 3-column grid, 480 in a 4-column one
BAR = 64  # each tile's title bar: the name and score, then how the drive ended
STRIP = 46  # the clock along the bottom
FPS = 20  # the table's runs step at 0.05 s, and AlpaSim's video has one frame a step
HOLD_S = 1.5  # the last frame stays this long, so every outcome can be read
FRAME = (900, 1000)  # AlpaSim's video: map and table on top, the camera picture below

HANDOVER_S = 3.0  # the simulator drives the recorded path itself this long, then the model takes over
SCORED_S = 20.0  # duration_frac_20s is the scored time as a fraction of this
ROUTE_LIMIT_M = 4.0  # scoring stops this far from the route (max_dist_to_gt_trajectory)

_SAMPLE = re.compile(r"alpasim/(?P<scene>clipgt-[0-9a-f-]+)")


def render(runs, out_dir: Path, cfg) -> list[Path]:
    """One grid: a row per scene, hardest first, a column per run, each cell that scene's score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores: dict[str, dict[int, float]] = {}
    for k, run in enumerate(runs):
        for record in run.samples:
            m = _SAMPLE.fullmatch(record.get("sample_id", ""))
            score = record.get("metrics", {}).get("scene_score")
            if m and score is not None:
                scores.setdefault(m["scene"], {})[k] = float(score)
    if not scores:
        return []

    scenes = sorted(scores, key=lambda s: (statistics.fmean(scores[s].values()), s))
    grid = np.array([[scores[s].get(k, np.nan) for k in range(len(runs))] for s in scenes])
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(runs), 1.4 + 0.28 * len(scenes)))
    image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    for (row, col), value in np.ndenumerate(grid):
        if not np.isnan(value):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([run.label for run in runs], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels([_short(s) for s in scenes], fontsize=7)
    ax.set_title("AlpaSim scene score (0 to 1) by scene, hardest first; blank = not scored", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = out_dir / "scores_by_scene.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [path]


def _short(scene: str) -> str:
    return scene.removeprefix("clipgt-")[:8]


def _scored_s(metrics: dict) -> float:
    """How long the model was scored for, in seconds after it took over."""
    return SCORED_S * float(metrics.get("duration_frac_20s") or 0.0)


def showcase(runs, n: int = 4) -> list[dict]:
    """The scenes to show people.

    The two hardest scenes, then the two where the models split most. Hardest means the
    lowest scene score averaged over every model. Split means the largest spread (standard
    deviation) of the models' scores. A scene is skipped when no model is scored for a full
    second, or fewer than three models are scored for 5 seconds or more. No scene is shown
    twice. Ties go to the harder scene, then the lower scene id. Only scenes every run
    scored are considered.

    The skips are for scenes with nothing to watch: on one of the 30, every model is counted
    off the road at the moment it takes over. Three was chosen after seeing what two and three
    pick: with two, the hardest scene was one where only two models drive long enough to
    watch. The widest gap between the best and the worst model, which neuroNCAP uses, says
    nothing here: it is 1.00 on 21 of the 30 scenes.

    With another `n`, the first half (rounded up) are the hardest. Each model drives a scene
    once, so there is no most typical run to choose. Returns [{"sample", "title", "caption"}].
    """
    scores: dict[str, dict[str, dict]] = {}
    for run in runs:
        for record in run.samples:
            m = _SAMPLE.fullmatch(record.get("sample_id", ""))
            metrics = record.get("metrics", {})
            if m and metrics.get("scene_score") is not None:
                scores.setdefault(m["scene"], {})[run.model] = metrics
    full = {scene: by for scene, by in scores.items() if len(by) == len(runs)}
    if not full:
        raise LookupError(
            "no AlpaSim scene was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )

    def watchable(scene: str) -> bool:
        seconds = [_scored_s(metrics) for metrics in full[scene].values()]
        return max(seconds) >= 1.0 and sum(s >= 5.0 for s in seconds) >= 3

    def values(scene: str) -> list[float]:
        return [float(metrics["scene_score"]) for metrics in full[scene].values()]

    def hardness(scene: str) -> float:
        return statistics.fmean(values(scene))

    def spread(scene: str) -> float:
        return statistics.pstdev(values(scene))

    left = [scene for scene in full if watchable(scene)]
    hardest = sorted(left, key=lambda s: (hardness(s), s))[: n - n // 2]
    left = [scene for scene in left if scene not in hardest]
    split = sorted(left, key=lambda s: (-spread(s), hardness(s), s))[: n // 2]

    rank = ["", "second ", "third ", "fourth ", "fifth "]
    picks = []
    for k, scene in enumerate(hardest):
        nth = rank[k] if k < len(rank) else f"number {k + 1} "
        picks.append((scene, f"The {nth}hardest scene: the models average {hardness(scene):.2f} of 1."))
    for k, scene in enumerate(split):
        nth = "most" if k == 0 else "next most" if k == 1 else f"most, number {k + 1}"
        high = sum(v > 0.9 for v in values(scene))
        low = sum(v < 0.1 for v in values(scene))
        picks.append(
            (
                scene,
                f"Where the models split {nth}: {high} of {len(runs)} score above 0.9 "
                f"and {low} score below 0.1.",
            )
        )
    return [
        {
            "sample": f"alpasim/{scene}",
            "title": f"Scene {_short(scene)}",
            "caption": f"{why} One drive per model.",
        }
        for scene, why in picks
    ]


@dataclass
class _Tile:
    label: str
    colour: str
    score: str
    outcome: str
    frames: list[np.ndarray]
    stop: int | None  # the first frame that is not scored; None when the drive is scored to its end


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """Every model's front camera on one scene, side by side on one clock.

    AlpaSim stops scoring at the first collision, when the car leaves the road, or 4 m from
    the route, and lets the drive run on to 20 s. So every tile advances together, and from
    the moment a model's scoring stops its tile freezes, dimmed: what follows was not scored,
    and a car far from the recorded path sees a smeared picture. Each tile's bar gives the
    scene score and how the scored part ended. Writes `scene-<id>.mp4` and a poster `.jpg` of
    the moment before the first model's scoring stops.
    """
    from PIL import Image

    from paddock.adapters.benchmarks.neuroncap.render import _decode, _encode

    m = _SAMPLE.fullmatch(sample_id)
    if not m:
        raise LookupError(
            f"{sample_id!r} is not an AlpaSim sample; they look like "
            "alpasim/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"
        )
    scene = m["scene"]

    found = []
    for run in runs:
        record = next((r for r in run.samples if r.get("sample_id") == sample_id), {})
        video = _video(cfg.runs / run.benchmark / run.model / run.run_id / scene, scene, record)
        if video is not None:
            found.append((run, video, record.get("metrics", {})))
    if not found:
        raise LookupError(
            f"no run here has a video of {sample_id} "
            f"({scene}/rollouts/{scene}/<id>/*_{CAMERA}_default.mp4; AlpaSim writes it when "
            '[benchmarks.alpasim] video = "true"); looked under '
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )

    ffmpeg = _ffmpeg()
    cols = 3 if len(found) <= 6 else 4
    rows = math.ceil(len(found) / cols)
    tile_w = WIDTH // cols
    tile_h = tile_w * 9 // 16
    tiles = []
    for run, video, metrics in found:
        # One model at a time: a whole 900x1000 video is about 1 GB decoded, a tile's worth 150 MB.
        whole = _decode(video, *FRAME, ffmpeg)
        if not whole:
            raise RuntimeError(f"{video}: ffmpeg decoded no frames")
        box = _camera_box(whole[0], tile_w / tile_h)
        frames = [
            np.asarray(Image.fromarray(f).crop(box).resize((tile_w, tile_h), Image.LANCZOS)) for f in whole
        ]
        del whole
        stop = _stop_frame(metrics)
        if stop is not None:
            stop = min(stop, len(frames) - 1)
        score, outcome = _outcome(metrics, stop, _speed_kmh(video, stop))
        tiles.append(_Tile(run.label, palette.color(run.model), score, outcome, frames, stop))

    height = rows * (BAR + tile_h) + STRIP
    # The clip ends when the last model's scored part does; an unscored tail shows nothing.
    last = max(len(t.frames) - 1 if t.stop is None else t.stop for t in tiles)
    first_stop = min(len(t.frames) if t.stop is None else t.stop for t in tiles)
    title = f"AlpaSim · scene {_short(scene)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene-{_short(scene)}"
    video_path, poster = out_dir / f"{stem}.mp4", out_dir / f"{stem}.jpg"
    count = last + 1 + round(HOLD_S * FPS)
    frames = (_compose(tiles, min(i, last), cols, tile_w, tile_h, height, title) for i in range(count))
    _encode(frames, video_path, WIDTH, height, ffmpeg, fps=FPS)
    still = _compose(tiles, max(first_stop - 1, 0), cols, tile_w, tile_h, height, title)
    Image.fromarray(still).save(poster, quality=85)
    return [video_path, poster]


def _video(scene_dir: Path, scene: str, record: dict) -> Path | None:
    """AlpaSim's own video of this scene: `<scene>/rollouts/<scene>/<id>/..._default.mp4`.

    A scene that was retried has one folder per attempt. The record names the attempt that
    was scored when it knows; otherwise the last finished one is it.
    """
    videos = sorted((scene_dir / "rollouts" / scene).glob(f"*/*_{CAMERA}_default.mp4"))
    wanted = record.get("native_id", {}).get("rollout_id")
    if wanted:
        videos = [v for v in videos if v.parent.name == wanted]
    finished = [v for v in videos if (v.parent / "_complete").exists()]
    return (finished or videos or [None])[-1]


def _camera_box(frame: np.ndarray, aspect: float) -> tuple[int, int, int, int]:
    """Where the camera picture sits in one frame of AlpaSim's video: (left, top, right, bottom).

    Found from the frame, not from fixed numbers: the picture's place depends on the camera's
    shape (AutoE2E's front camera is 1920x1018, the others' 1920x1080). AlpaSim draws a thin
    black frame around it, and that frame is the only dark line in the video that runs most
    of the way across: the map's frame is half as wide, the table's rules a third. The box
    is then trimmed about its centre to `aspect`, so no tile is stretched.
    """
    dark = (frame < 110).all(axis=2)
    rows = np.flatnonzero(dark.mean(axis=1) > 0.6)
    if len(rows) < 2 or rows[-1] - rows[0] < 20:
        raise RuntimeError("no camera picture found in this frame of AlpaSim's video: no dark frame line")
    top, bottom = int(rows[0]), int(rows[-1])
    cols = np.flatnonzero(dark[top : bottom + 1].mean(axis=0) > 0.6)
    if len(cols) < 2 or cols[-1] - cols[0] < 20:
        raise RuntimeError("no camera picture found in this frame of AlpaSim's video: no dark frame sides")
    # Step inside the frame line, which the video's compression smears over a few pixels.
    left, right, top, bottom = int(cols[0]) + 3, int(cols[-1]) - 2, top + 3, bottom - 2
    w, h = right - left, bottom - top
    if w / h > aspect:
        trim = w - round(h * aspect)
        left, right = left + trim // 2, right - (trim - trim // 2)
    else:
        trim = h - round(w / aspect)
        top, bottom = top + trim // 2, bottom - (trim - trim // 2)
    return left, top, right, bottom


def _ended(metrics: dict) -> str:
    """Why scoring stopped: "collision", "road", "route", or "" when it ran to the end."""
    if metrics.get("collision_rate"):
        return "collision"
    if metrics.get("offroad"):
        return "road"
    if (metrics.get("dist_to_gt_trajectory") or 0.0) >= ROUTE_LIMIT_M:
        return "route"
    return ""


def _stop_frame(metrics: dict) -> int | None:
    """The first frame of the video that is not scored, or None when the drive is scored to its end.

    The scored time is `duration_frac_20s`, counted from the hand-over, and the event that
    ends it is one step later. Checked step for step against metrics.parquet for eight models
    on five scenes: this is the event's own step on all 40 drives.
    """
    if not _ended(metrics) or metrics.get("duration_frac_20s") is None:
        return None
    return round((HANDOVER_S + _scored_s(metrics)) * FPS) + 1


def _speed_kmh(video: Path, stop: int | None) -> float | None:
    """The car's own speed at frame `stop`, from the controller's log beside the rollouts.

    AlpaSim records no impact speed. The log's first row is two steps after the video's first
    frame. None when there is no log, so the bar then leaves the speed out.
    """
    if stop is None:
        return None
    logs = sorted(video.parents[3].glob(f"controller/*{video.parent.name}.csv"))
    if not logs:
        return None
    with logs[0].open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[min(max(stop - 2, 0), len(rows) - 1)]
    try:
        return 3.6 * math.hypot(float(row["vx"]), float(row["vy"]))
    except (KeyError, ValueError):
        return None


def _outcome(metrics: dict, stop: int | None, speed_kmh: float | None) -> tuple[str, str]:
    """("score 0.00", "collided at 11 s, driving 32 km/h"): the score, and how the scored part ended.

    Every speed is the car's own. Times are on the video's clock.
    """
    score = metrics.get("scene_score")
    head = "" if score is None else f"score {float(score):.2f}"
    why = _ended(metrics)
    if not why or stop is None:
        return head, "scored to the end" if score is not None else ""
    at = f"at {stop / FPS:.0f} s"
    if why == "collision":
        # "collided", not "hit a car": the record does not say what was hit, and it can be a truck.
        what = "collided" if metrics.get("collision_at_fault") else "was hit"
        if speed_kmh is None:
            return head, f"{what} {at}"
        if speed_kmh < 1.0:
            return head, f"{what} {at}, standing still"
        return head, f"{what} {at}, driving {speed_kmh:.0f} km/h"
    if why == "road":
        return head, f"left the road {at}"
    if (metrics.get("progress") or 0.0) >= 0.995:
        return head, f"passed the end of the route {at}"
    return head, f"{ROUTE_LIMIT_M:.0f} m off the route {at}"


def _compose(tiles, i, cols, tile_w, tile_h, height, title) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    # Sized for the page, where the 1920-wide video plays about 1100 wide.
    scale = tile_w / 640
    font = ImageFont.load_default(size=round(28 * scale))
    small = ImageFont.load_default(size=round(23 * scale))
    canvas = Image.new("RGB", (WIDTH, height), (14, 15, 18))
    draw = ImageDraw.Draw(canvas)
    for k, tile in enumerate(tiles):
        x, y = (k % cols) * tile_w, (k // cols) * (BAR + tile_h)
        stopped = tile.stop is not None and i >= tile.stop
        shown = tile.stop if stopped else i
        frame = Image.fromarray(tile.frames[min(shown, len(tile.frames) - 1)])
        if stopped:
            frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.55)
        canvas.paste(frame, (x, y + BAR))
        draw.rectangle([x, y, x + tile_w - 1, y + BAR - 1], fill=(26, 27, 32))
        draw.rectangle([x, y, x + 7, y + BAR - 1], fill=tile.colour)
        draw.text((x + 18, y + BAR // 4 + 2), tile.label, font=font, fill=(242, 242, 245), anchor="lm")
        draw.text(
            (x + tile_w - 14, y + BAR // 4 + 2), tile.score, font=font, fill=(242, 242, 245), anchor="rm"
        )
        draw.text((x + 18, y + 3 * BAR // 4 - 1), tile.outcome, font=small, fill=(205, 207, 214), anchor="lm")
    mid = height - STRIP // 2
    draw.rectangle([0, height - STRIP, WIDTH, height], fill=(0, 0, 0))
    draw.text((18, mid), f"t = {i / FPS:4.1f} s", font=font, fill=(255, 255, 255), anchor="lm")
    draw.text((200, mid), title, font=small, fill=(190, 192, 200), anchor="lm")
    note = "green: the route · pink or orange: the model's plan · a dimmed tile: its scoring has stopped"
    draw.text((WIDTH - 18, mid), note, font=small, fill=(150, 152, 160), anchor="rm")
    return np.asarray(canvas)


def _ffmpeg() -> str:
    """`PADDOCK_FFMPEG`, then PATH: neuroNCAP's lookup, with no AlpaSim setting of its own."""
    from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg

    found = resolve_ffmpeg(None)
    if not found:
        raise RuntimeError(
            "drawing an AlpaSim scene needs ffmpeg and none was found. Set PADDOCK_FFMPEG "
            "(AlpaSim's own environment has one: python -c 'import imageio_ffmpeg; "
            "print(imageio_ffmpeg.get_ffmpeg_exe())')."
        )
    return found
