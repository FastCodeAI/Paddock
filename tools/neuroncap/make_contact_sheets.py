"""Contact sheets of a run's frames, for eyeballing what a model did and when.

NeuroNCAP writes three streams per run:

    CAM_FRONT/        the raw NeuRAD render -- exactly the pixels the model got
    FC_TRAJ/          that render with the model's planned trajectory drawn on
    COMBINED_OUTPUTS/ the full panel: BEV with actors, three cameras, overlays

FC_TRAJ is the one worth scanning frame by frame, because it puts the input and
the decision in the same picture: if a model drove into something, you can see
whether it never saw it or saw it and planned through it anyway.

Sheets sample evenly across the run and label each tile with its time offset,
so "it started braking at t=1.8 s" is readable straight off the sheet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__import__("os").environ.get("NCAP_RUNS", "/mnt/p4/naren/neuroncap/outputs/runs1"))
OUT = Path("/mnt/p4/naren/neuroncap/frames")
SCENARIOS = ("stationary", "frontal", "side")


def sheet(frames: list[Path], columns: int, tile_width: int) -> Image.Image:
    with Image.open(frames[0]) as probe:
        aspect = probe.height / probe.width
    tile_height = int(tile_width * aspect)
    rows = (len(frames) + columns - 1) // columns

    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (16, 16, 18))
    t0 = int(frames[0].stem)
    for i, path in enumerate(frames):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((tile_width, tile_height), Image.BILINEAR)
        draw = ImageDraw.Draw(tile)
        label = f"t={(int(path.stem) - t0) / 1e6:5.1f}s"
        draw.rectangle([0, 0, 96, 20], fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(0, 255, 128))
        canvas.paste(tile, ((i % columns) * tile_width, (i // columns) * tile_height))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--stream", default="FC_TRAJ", choices=["FC_TRAJ", "CAM_FRONT", "COMBINED_OUTPUTS"])
    ap.add_argument("--max-tiles", type=int, default=24)
    ap.add_argument("--columns", type=int, default=6)
    ap.add_argument("--tile-width", type=int, default=420)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for scenario in SCENARIOS:
            source = ROOT / model / f"{scenario}-0103" / args.stream
            if not source.is_dir():
                print(f"skip {model}/{scenario}: no {args.stream}")
                continue
            frames = sorted(source.iterdir(), key=lambda p: int(p.stem))
            if not frames:
                continue
            # Always keep the last frame: it carries the outcome.
            if len(frames) > args.max_tiles:
                step = len(frames) / args.max_tiles
                picked = [frames[min(int(i * step), len(frames) - 1)] for i in range(args.max_tiles - 1)]
                picked.append(frames[-1])
                frames = picked

            path = OUT / f"{model}_{scenario}_{args.stream}.jpg"
            sheet(frames, args.columns, args.tile_width).save(path, quality=88)
            print(f"{path}  ({len(frames)} tiles)")


if __name__ == "__main__":
    main()
