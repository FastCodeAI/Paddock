"""Burn a run's reasoning trace onto its own video, for a sweep already on disk.

Built for exactly the situation in
docs/models/alpamayo15-neuroncap-closed-loop-findings.md: a sweep was run with
`poll_reasoning.py` polling on the side, so `reasoning_full_sweep.jsonl` (or
whatever `--out` was given) sits next to the sweep's own artifacts -- but it
was captured independently of the engine's loop, one file spanning every
episode and every run, with no per-run boundaries recorded on its own.

This script re-derives those boundaries and burns the result onto
`COMBINED_OUTPUTS.mp4`, one new file per run, so the reasoning that produced
each drive is visible on the drive itself rather than read separately out of
a jsonl.

How the mapping works:

1. Episode order: every subdirectory of the sweep directory with its own
   `stdout.log` is an episode (`<scenario>-<scene>`, matching the harness's own
   naming). Ordered by that file's mtime, which is written once, when the
   whole episode's subprocess exits (`harness.py`), so it is a reliable
   ordering signal even though it is not a per-run one.
2. Run boundaries within the reasoning log: `/diagnostics` returns
   `{"status": "no frame seen yet"}` exactly once per `reset()` call (see
   `alpamayo15/policy.py`), which happens between every run. Splitting the
   poll log on that marker yields exactly as many segments as runs, in
   execution order -- confirmed against `stdout.log` mtimes in the case this
   was built against (30 segments, 3 episodes x 10 runs, mtimes all landing in
   the expected windows).
3. Per-run alignment is by wall-clock interpolation, not exact timestamps:
   frame i of a run's video is assumed to fall at
   `segment[0].wall_time + i / (frames - 1) * (segment[-1].wall_time -
   segment[0].wall_time)`, then matched to the nearest poll by that estimate.
   The poller runs concurrently with, not synchronized to, the engine's own
   clock, so this is an approximation -- good enough to show what the model
   was saying *around* a given moment, not frame-exact.

Usage:

    python tools/benchmarks/neuroncap/annotate_reasoning_video.py \\
        {runs}/neuroncap/alpamayo15/<run_id>

Writes `<run_dir>/COMBINED_OUTPUTS_reasoning.mp4` beside the existing video
for every run found. `--episode`/`--run` narrow it to one run instead of the
whole sweep.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_RUN_DIR = re.compile(r"^run_(\d+)$")
STREAM = "COMBINED_OUTPUTS"
FONT_SIZE = 22
LINE_HEIGHT = 28
PAD = 12


def _ffmpeg(cfg) -> str:
    """`[benchmarks.neuroncap] ffmpeg`, then `PADDOCK_FFMPEG`, then PATH.

    This script has no harness instance to call `NeuroNcapBenchmark._ffmpeg`
    on, so it goes straight to the one place that precedence is decided
    (`harness.resolve_ffmpeg`) rather than keeping a second, hand-synchronized
    copy -- which already drifted out of sync with the harness's once.
    """
    from paddock.adapters.benchmarks.neuroncap.harness import resolve_ffmpeg

    settings = cfg.benchmarks.get("neuroncap") or {}
    found = resolve_ffmpeg(settings.get("ffmpeg"))
    if not found:
        raise SystemExit("no ffmpeg found -- set [benchmarks.neuroncap] ffmpeg, $PADDOCK_FFMPEG, or PATH")
    return found


def _episode_dirs(sweep_dir: Path) -> list[Path]:
    """Every subdirectory with its own `stdout.log`, oldest first."""
    episodes = [p for p in sweep_dir.iterdir() if p.is_dir() and (p / "stdout.log").exists()]
    return sorted(episodes, key=lambda p: (p / "stdout.log").stat().st_mtime)


def _run_dirs(episode_dir: Path) -> list[Path]:
    """`run_<i>/` in order, or the episode dir itself for a single-run episode."""
    runs = sorted(
        (p for p in episode_dir.iterdir() if p.is_dir() and _RUN_DIR.match(p.name)),
        key=lambda p: int(_RUN_DIR.match(p.name).group(1)),
    )
    return runs or [episode_dir]


def _segment_reasoning(entries: list[dict]) -> list[list[dict]]:
    """Split on the `{"status": ...}` reset marker `/diagnostics` returns between runs."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for entry in entries:
        if "status" in entry:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(entry)
    if current:
        segments.append(current)
    return segments


def _font() -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_frame(path: Path, text: str, font: ImageFont.ImageFont) -> None:
    img = Image.open(path).convert("RGB")
    wrapped = textwrap.wrap(text, width=70) or [""]
    bar_height = PAD * 2 + LINE_HEIGHT * len(wrapped)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, img.height - bar_height, img.width, img.height], fill=(0, 0, 0, 160))
    for i, line in enumerate(wrapped):
        draw.text(
            (PAD, img.height - bar_height + PAD + i * LINE_HEIGHT),
            line,
            font=font,
            fill=(255, 255, 0, 255),
        )
    Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(path)


def _annotate_run(run_dir: Path, segment: list[dict], ffmpeg: str, font: ImageFont.ImageFont) -> Path | None:
    src = run_dir / f"{STREAM}.mp4"
    if not src.exists():
        print(f"  {run_dir.name}: no {STREAM}.mp4, skipping")
        return None
    if not segment:
        print(f"  {run_dir.name}: no reasoning entries mapped, skipping")
        return None

    dest = run_dir / f"{STREAM}_reasoning.mp4"
    with tempfile.TemporaryDirectory(prefix="ncap_annotate_") as tmp:
        tmp_dir = Path(tmp)
        extracted = subprocess.run(
            [ffmpeg, "-loglevel", "error", "-y", "-i", str(src), str(tmp_dir / "frame_%05d.png")],
            capture_output=True,
            text=True,
        )
        if extracted.returncode != 0:
            print(f"  {run_dir.name}: frame extraction failed: {extracted.stderr.strip()[-300:]}")
            return None

        frames = sorted(tmp_dir.glob("frame_*.png"))
        if not frames:
            print(f"  {run_dir.name}: no frames extracted from {src.name}")
            return None

        times = [e["wall_time"] for e in segment]
        t0, t1 = times[0], times[-1]
        span = max(t1 - t0, 1e-6)
        for i, frame in enumerate(frames):
            fraction = i / max(len(frames) - 1, 1)
            est_time = t0 + fraction * span
            idx = bisect.bisect_left(times, est_time)
            idx = min(idx, len(segment) - 1)
            reasoning = segment[idx].get("reasoning") or "(no reasoning reported)"
            _annotate_frame(frame, reasoning, font)

        encoded = subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "10",
                "-i",
                str(tmp_dir / "frame_%05d.png"),
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
        if encoded.returncode != 0 or not dest.exists():
            print(f"  {run_dir.name}: encode failed: {encoded.stderr.strip()[-300:]}")
            return None

    print(f"  {run_dir.name}: wrote {dest}")
    return dest


def _select_runs_and_segments(
    sweep_dir: Path,
    segments: list[list[dict]],
    episode_name: str | None,
    run_index: int | None,
) -> tuple[list[tuple[Path, Path]], list[list[dict]]]:
    """(run, segment) pairs for the whole sweep, or narrowed to one episode/run.

    `segments` is always the whole sweep's, in episode order, one per run --
    narrowing to one episode must re-slice it to that episode's own span, found
    by its offset in the *unfiltered* episode list, or every run in it gets
    paired with another episode's segment (and, once the shorter iterable is
    exhausted, a crash with no indication the videos already written are
    wrong -- the exact failure mode this tool exists to prevent).
    """
    if run_index is not None and episode_name is None:
        raise SystemExit("--run narrows a single episode's runs; it requires --episode")

    full_episodes = _episode_dirs(sweep_dir)
    episodes = full_episodes
    if episode_name is not None:
        episodes = [e for e in full_episodes if e.name == episode_name]
        if not episodes:
            raise SystemExit(f"no episode dir named {episode_name!r} under {sweep_dir}")

    all_runs = [(ep, run) for ep in episodes for run in _run_dirs(ep)]
    if len(all_runs) != len(segments) and episode_name is None:
        print(
            f"warning: {len(all_runs)} run dirs but {len(segments)} reasoning segments -- "
            "mapping may be misaligned; consider narrowing with --episode/--run to check"
        )

    if episode_name is not None:
        idx = full_episodes.index(episodes[0])
        offset = sum(len(_run_dirs(full_episodes[i])) for i in range(idx))
        episode_runs = _run_dirs(episodes[0])
        if run_index is not None:
            all_runs = [(episodes[0], episode_runs[run_index])]
            segments = segments[offset + run_index : offset + run_index + 1]
        else:
            all_runs = [(episodes[0], run) for run in episode_runs]
            segments = segments[offset : offset + len(episode_runs)]

    return all_runs, segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="the sweep's top-level artifact directory under [roots.runs]")
    ap.add_argument(
        "--reasoning", default=None, help="reasoning jsonl; default: the only reasoning*.jsonl in sweep_dir"
    )
    ap.add_argument("--episode", default=None, help="only this episode dir name, e.g. side-0103")
    ap.add_argument("--run", type=int, default=None, help="only this run index within --episode")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        raise SystemExit(f"not a directory: {sweep_dir}")

    reasoning_path = Path(args.reasoning) if args.reasoning else None
    if reasoning_path is None:
        candidates = sorted(sweep_dir.glob("reasoning*.jsonl"))
        if len(candidates) != 1:
            raise SystemExit(
                f"expected exactly one reasoning*.jsonl in {sweep_dir}, found {len(candidates)} "
                "-- pass --reasoning explicitly"
            )
        reasoning_path = candidates[0]

    entries = [json.loads(line) for line in reasoning_path.read_text().splitlines() if line.strip()]
    segments = _segment_reasoning(entries)

    all_runs, segments = _select_runs_and_segments(sweep_dir, segments, args.episode, args.run)

    # `python <full path>` puts this file's own directory on sys.path, not the
    # repo root -- see the same fix in poll_reasoning.py.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools import paths

    cfg = paths.config()
    ffmpeg = _ffmpeg(cfg)
    font = _font()

    print(f"{len(all_runs)} run(s), {len(segments)} reasoning segment(s), stream={STREAM}")
    for (episode, run_dir), segment in zip(all_runs, segments, strict=True):
        _annotate_run(run_dir, segment, ffmpeg, font)


if __name__ == "__main__":
    main()
