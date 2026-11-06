"""Render one comma2k19 segment to an .mp4 with its own labels overlaid.

No model, no run, no scoring: this is for looking at the *dataset*. The direct
descendant of the toolkit's `visualize_segment.py`, with its argparse surface
kept so anyone who has used that script can use this one, and with the drawing
delegated to `adapters/benchmarks/comma2k19/overlay.py` so there is one
implementation rather than two that drift.

    python -m tools.benchmarks.comma2k19.overlay_video \
        --segment '<root>/Chunk_1/<dongle>|<start>/40' --out /tmp/overlay.mp4 \
        --max-frames 300 --lanes

Paths resolve through `paddock.toml`, so `--segment` is optional: with nothing
given it takes the first segment the configured tiers can serve and says which.
Needs a video decoder -- PyAV in this environment, or `PADDOCK_FFMPEG` -- and
Pillow, i.e. the `viz` extra.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19 import overlay as OV
from tools import paths


def first_segment() -> Path:
    """Some segment this site can actually serve, for a default.

    Walks the tiers rather than guessing a path, so the answer is a segment that
    passed the same check the benchmark applies -- and is the same one at every
    site, because `list_segments` sorts.
    """
    dataset = paths.config().datasets.get("comma2k19")
    if dataset is None:
        raise SystemExit("dataset 'comma2k19' is not configured in paddock.toml; see docs/data-setup.md")
    segments = L.list_segments(dataset)
    if not segments:
        raise SystemExit(f"no comma2k19 segments under {', '.join(str(r) for r in dataset.roots)}")
    return segments[0].path


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--segment", default=None, help="segment directory; default the first configured one")
    ap.add_argument("--out", default=None, help="output .mp4; default <runs>/comma2k19/_segments/<route>_<seg>.mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = the whole segment")
    ap.add_argument("--path-horizon-s", type=float, default=OV.PATH_HORIZON_S, help="how far ahead to draw")
    ap.add_argument("--path-skip-s", type=float, default=OV.PATH_SKIP_S, help="skip the near field, hidden by the hood")
    ap.add_argument("--radar-window-s", type=float, default=OV.RADAR_WINDOW_S, help="radar rows kept per frame")
    ap.add_argument("--no-path", action="store_true")
    ap.add_argument("--no-radar", action="store_true")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-minimap", action="store_true")
    ap.add_argument("--lanes", action="store_true", help="also draw the derived +-1.85 m lane rails")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality; lower is bigger and better")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    segment = Path(args.segment).resolve() if args.segment else first_segment()
    if not (segment / "video.hevc").exists():
        raise SystemExit(f"{segment} has no video.hevc")
    if args.segment is None:
        print(f"no --segment given; using {segment}", file=sys.stderr)

    out = Path(args.out) if args.out else None
    if out is None:
        name = f"{segment.parent.name.replace('|', '_')}_{segment.name}.mp4"
        out = paths.runs() / "comma2k19" / "_segments" / name

    overlay = OV.SegmentOverlay(
        segment,
        layers=OV.Layers(
            path=not args.no_path,
            rails=args.lanes,
            radar=not args.no_radar,
            hud=not args.no_hud,
            minimap=not args.no_minimap,
        ),
        path_horizon_s=args.path_horizon_s,
        path_skip_s=args.path_skip_s,
        radar_window_s=args.radar_window_s,
    )
    limit = args.max_frames or len(overlay.pose)
    limit = min(limit, len(overlay.pose))  # never draw past the pose stream

    began = time.time()
    drawn = 0

    def frames():
        nonlocal drawn
        for index, rgb in enumerate(L.decode_frames(segment / "video.hevc", limit=limit)):
            yield overlay.draw(rgb, index)
            drawn += 1
            if drawn % 100 == 0:
                print(f"  {drawn}/{limit} frames", file=sys.stderr)

    L.encode_frames(frames(), out, fps=overlay.fps, crf=args.crf)
    print(
        f"wrote {out}\n  {drawn} frames @ {overlay.fps:.2f} Hz ({drawn / overlay.fps:.1f} s), "
        f"{out.stat().st_size / 1e6:.1f} MB, in {time.time() - began:.1f} s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
