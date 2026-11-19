"""Comparison figures for neuroNCAP runs.

Closed-loop comparison has a different shape from open-loop. There is no
per-sample trajectory error to plot a distribution of: there are a handful of
scenarios, a score each, and — for stochastic models — a spread across seeds
that matters more than any single value.

So this draws scores per scenario with their spread, and says n. Alpamayo's
frontal score ranges over the whole 0–5 scale across seeds while openpilot's
moves by 0.75, which a bar chart of single runs would hide entirely.

One episode can be drawn too: `render_sample` lays every model's front-camera
video of the same episode side by side on one clock (`paddock render --bench
neuroncap --sample neuroncap/0103/frontal/run_0`). `showcase` picks the episodes
the website shows. Both read only what a run already keeps: each episode's
`FC_TRAJ.mp4` (the front camera with the plan drawn on) and its record in
samples.jsonl.
"""

from __future__ import annotations

import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paddock.viz import palette, video


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


# -- one episode, every model -------------------------------------------------------------

KINDS = {
    "frontal": "Oncoming car",
    "side": "Car crossing from the side",
    "stationary": "Stopped car ahead",
}
WIDTH = 1920  # the whole video; tiles are 640 wide in a 3-column grid, 480 in a 4-column one
BAR = 48  # each tile's title bar
STRIP = 46  # the clock along the bottom
FPS = 10  # neuroNCAP steps at 0.1 s and writes one frame a step
HOLD_S = 1.5  # the last frame stays this long, so every outcome can be read
_SAMPLE = re.compile(r"neuroncap/(?P<scene>\d+)/(?P<scenario>[a-z]+)/run_(?P<run>\d+)")


def showcase(runs, n: int = 4) -> list[dict]:
    """The episodes to show people: hard ones, and of different kinds.

    First the hardest scenario of each kind -- frontal, side, stationary -- where hardest
    means the lowest NCAP score averaged over every model and run. Then, until there are
    `n`, the scenario where the models disagree most: the widest gap between the best and
    the worst model's average. No scene is shown twice, so the picks differ in place as
    well as in kind. Ties go to the harder scenario, then the lower scene number.

    Of a scenario's runs (each one jitters the other car's start), the one shown is the
    most typical: the run whose scores sit closest to each model's own average. The worst
    run would show every model at its unluckiest.

    Only scenarios that every run scored are considered, so no tile is ever empty.
    Returns [{"sample", "title", "caption"}], in the order to show them.
    """
    scores: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    for run in runs:
        for record in run.samples:
            nid = record.get("native_id", {})
            score = record.get("metrics", {}).get("ncap_score")
            if score is None or "scenario" not in nid:
                continue
            key = (nid["scenario"], nid["scene"])
            scores.setdefault(key, {}).setdefault(run.model, {})[int(nid["run"])] = float(score)
    full = {k: v for k, v in scores.items() if len(v) == len(runs)}
    if not full:
        raise LookupError(
            "no neuroNCAP scenario was scored by every run given: "
            + ", ".join(f"{r.model}/{r.run_id} ({len(r.samples)} records)" for r in runs)
        )

    def means(key):
        return {m: statistics.fmean(by.values()) for m, by in full[key].items()}

    def hardness(key):
        return statistics.fmean(means(key).values())

    def gap(key):
        return max(means(key).values()) - min(means(key).values())

    picks: list[tuple[tuple[str, str], str]] = []
    shown: set[str] = set()
    for kind in KINDS:
        left = sorted(
            (k for k in full if k[0] == kind and k[1] not in shown), key=lambda k: (hardness(k), k[1])
        )
        if left:
            key = left[0]
            picks.append((key, f"The hardest {kind} scenario: the models average {hardness(key):.1f} of 5."))
            shown.add(key[1])
    while len(picks) < n:
        left = sorted((k for k in full if k[1] not in shown), key=lambda k: (-gap(k), hardness(k), k[1]))
        if not left:
            break
        key = left[0]
        low, high = min(means(key).values()), max(means(key).values())
        picks.append(
            (key, f"Where the models disagree most: their averages run from {low:.1f} to {high:.1f} of 5.")
        )
        shown.add(key[1])

    out = []
    for (kind, scene), why in picks[:n]:
        by, avg = full[(kind, scene)], means((kind, scene))
        common = sorted(set.intersection(*(set(r) for r in by.values())))
        pick = min(common, key=lambda i: sum(abs(by[m][i] - avg[m]) for m in by))
        out.append(
            {
                "sample": f"neuroncap/{scene}/{kind}/run_{pick}",
                "title": f"{KINDS.get(kind, kind)}, scene {scene}",
                "caption": f"{why} Run {pick + 1} of {len(common)}, the most typical one.",
            }
        )
    return out


@dataclass
class _Tile:
    label: str
    colour: str
    metrics: dict
    frames: list[np.ndarray]


def render_sample(runs, sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """Every model's front camera on one episode, side by side on one clock.

    An episode stops when the car hits something, so a short video means an early
    crash. On one clock that reads at a glance: every tile advances together, and a
    model whose episode is over freezes, dimmed, on its last frame. Each tile's bar
    says how the episode ended. Writes `<scenario>-<scene>-run<i>.mp4` and a poster
    `.jpg` of the moment the first episode ends.
    """
    from PIL import Image

    m = _SAMPLE.fullmatch(sample_id)
    if not m:
        raise LookupError(
            f"{sample_id!r} is not a neuroNCAP sample; they look like neuroncap/0103/frontal/run_0"
        )
    scene, kind, index = m["scene"], m["scenario"], int(m["run"])
    episode = Path(f"{kind}-{scene}") / f"run_{index}" / "FC_TRAJ.mp4"

    found = []
    for run in runs:
        video = cfg.runs / run.benchmark / run.model / run.run_id / episode
        if video.exists():
            record = next((r for r in run.samples if r.get("sample_id") == sample_id), {})
            found.append((run, video, record.get("metrics", {})))
    if not found:
        raise LookupError(
            f"no run here has a video of {sample_id} ({episode}); looked under "
            + ", ".join(f"{r.model}/{r.run_id}" for r in runs)
        )

    ffmpeg = _ffmpeg(cfg)
    cols = 3 if len(found) <= 6 else 4
    rows = math.ceil(len(found) / cols)
    tile_w = WIDTH // cols
    tile_h = tile_w * 9 // 16  # the episode videos are 1598x900
    tiles = [
        _Tile(run.label, palette.color(run.model), metrics, _decode(video, tile_w, tile_h, ffmpeg))
        for run, video, metrics in found
    ]
    height = rows * (BAR + tile_h) + STRIP
    longest = max(len(t.frames) for t in tiles)
    first_end = min(len(t.frames) for t in tiles) - 1
    title = f"neuroNCAP · scene {scene} · {KINDS.get(kind, kind).lower()} · run {index + 1}"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}-{scene}-run{index}"
    video_path, poster = out_dir / f"{stem}.mp4", out_dir / f"{stem}.jpg"
    count = longest + round(HOLD_S * FPS)
    frames = (_compose(tiles, min(i, longest - 1), cols, tile_w, tile_h, height, title) for i in range(count))
    _encode(frames, video_path, WIDTH, height, ffmpeg)
    Image.fromarray(_compose(tiles, first_end, cols, tile_w, tile_h, height, title)).save(poster, quality=85)
    return [video_path, poster]


def _outcome(metrics: dict) -> str:
    score = metrics.get("ncap_score")
    if score is None:
        return ""
    if metrics.get("collision_rate"):
        # neuroNCAP's impact_speed is how fast the two cars move toward each other (the norm of
        # their velocity difference, evaluator.py), so for an oncoming car it is both speeds
        # added. "hit at 76 km/h" read as the model's own speed when it was driving at 41.
        return f"NCAP {score:.1f} · the cars met at {3.6 * metrics.get('impact_speed', 0.0):.0f} km/h"
    return f"NCAP {score:.1f} · no collision"


def _compose(tiles, i, cols, tile_w, tile_h, height, title) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    # Sized for the page, where the 1920-wide video plays about 1100 wide: the score is
    # the thing to read, so it gets nearly the name's size. A 4-column grid scales both down.
    scale = tile_w / 640
    font = ImageFont.load_default(size=round(28 * scale))
    small = ImageFont.load_default(size=round(23 * scale))
    canvas = Image.new("RGB", (WIDTH, height), (14, 15, 18))
    draw = ImageDraw.Draw(canvas)
    for k, tile in enumerate(tiles):
        x, y = (k % cols) * tile_w, (k // cols) * (BAR + tile_h)
        ended = i >= len(tile.frames)
        frame = Image.fromarray(tile.frames[min(i, len(tile.frames) - 1)])
        if ended:
            frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.55)
        canvas.paste(frame, (x, y + BAR))
        draw.rectangle([x, y, x + tile_w - 1, y + BAR - 1], fill=(26, 27, 32))
        draw.rectangle([x, y, x + 7, y + BAR - 1], fill=tile.colour)
        draw.text((x + 18, y + BAR // 2), tile.label, font=font, fill=(242, 242, 245), anchor="lm")
        note = _outcome(tile.metrics) + (" · over" if ended else "")
        draw.text((x + tile_w - 14, y + BAR // 2), note, font=small, fill=(205, 207, 214), anchor="rm")
    mid = height - STRIP // 2
    draw.rectangle([0, height - STRIP, WIDTH, height], fill=(0, 0, 0))
    draw.text((18, mid), f"t = {i / FPS:4.1f} s", font=font, fill=(255, 255, 255), anchor="lm")
    draw.text((200, mid), title, font=small, fill=(190, 192, 200), anchor="lm")
    note = "a dimmed tile: that episode is over"
    draw.text((WIDTH - 18, mid), note, font=small, fill=(150, 152, 160), anchor="rm")
    return np.asarray(canvas)


def _ffmpeg(cfg=None) -> str:
    """The harness's ffmpeg: `[benchmarks.neuroncap] ffmpeg`, then `PADDOCK_FFMPEG`, then PATH."""
    from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg

    override = getattr(cfg, "benchmarks", {}).get("neuroncap")
    found = resolve_ffmpeg(override.get("ffmpeg") if override else None)
    if not found:
        raise RuntimeError(
            "drawing a neuroNCAP episode needs ffmpeg and none was found. Set [benchmarks.neuroncap] "
            "ffmpeg or PADDOCK_FFMPEG (openpilot's env has one at <openpilot>/.venv/bin/ffmpeg)."
        )
    return found


def _decode(video: Path, width: int, height: int, ffmpeg: str | None = None) -> list[np.ndarray]:
    """Every frame of `video`, scaled to width x height, as RGB arrays.

    Scaled by ffmpeg on the way out, so the raw frame size is known without probing
    the file first. A whole episode is about 60 frames, so holding them is cheap.
    """
    cmd = [ffmpeg or _ffmpeg(), "-v", "error", "-i", str(video), "-vf", f"scale={width}:{height}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    done = subprocess.run(cmd, capture_output=True, check=False)  # communicate() drains both pipes
    if done.returncode:
        raise RuntimeError(f"{video}: ffmpeg failed: {done.stderr.decode(errors='replace').strip()}")
    size = width * height * 3
    raw = done.stdout
    return [
        np.frombuffer(raw, np.uint8, size, k).reshape(height, width, 3)
        for k in range(0, len(raw) - size + 1, size)
    ]


def _encode(frames, path: Path, width: int, height: int, ffmpeg: str | None = None, fps: int = FPS) -> None:
    """`viz.video.encode` at neuroNCAP's rate. `fps` is for a benchmark that borrows this
    and steps at another rate (AlpaSim: 20 a second)."""
    video.encode(frames, path, width, height, fps, ffmpeg or _ffmpeg())
