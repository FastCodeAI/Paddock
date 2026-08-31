"""One video per scenario, all four models side by side on a shared clock.

The per-model clips are hard to compare because each ends at a different time --
a run stops on collision, so a short clip means an early crash and a full-length
one means the model survived. Laying them out on one timeline makes that
legible: every tile advances together, and a model that has already crashed
freezes on its last frame with its outcome shown rather than disappearing.

FC_TRAJ is the stream used (front camera with the planned trajectory drawn on),
since the interesting comparison is what each model saw and what it decided to
do about it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path("/mnt/p4/naren/neuroncap")
FFMPEG = "/shakaal_hdd_1/workspace/naren/renesas/openpilot/.venv/bin/ffmpeg"
MODELS = ("drivejepa", "openpilot", "alpamayo", "visionpilot")  # best-first, reading order
LABELS = {"drivejepa": "Drive-JEPA", "openpilot": "OpenPilot",
          "alpamayo": "Alpamayo 1.5", "visionpilot": "VisionPilot"}
TILE_W, TILE_H = 800, 450
BAR = 34
CLOCK = 30
STEP_S = 0.1


def scores(log: Path) -> dict[tuple[str, str], float]:
    """Parse ncap_score per (model, scenario) out of the ladder log."""
    out, model = {}, None
    for line in log.read_text().splitlines():
        if line.startswith("==="):
            model = line.split()[1]
        elif m := re.match(r"\s+(\w+): \d+ run", line):
            scenario = m.group(1)
        elif "ncap_score" in line and model:
            out[(model, scenario)] = float(line.split("ncap_score:")[1].split(",")[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/runs1")
    ap.add_argument("--log", default="ladder_runs1.log")
    ap.add_argument("--out", default="videos_comparison")
    args = ap.parse_args()

    root = BASE / args.runs
    score = scores(BASE / args.log)
    out_dir = BASE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in ("stationary", "frontal", "side"):
        streams = {}
        for model in MODELS:
            d = root / model / f"{scenario}-0103" / "FC_TRAJ"
            if d.is_dir():
                streams[model] = sorted(d.iterdir(), key=lambda p: int(p.stem))
        if not streams:
            print(f"skip {scenario}: no frames")
            continue

        n = max(len(v) for v in streams.values())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for i in range(n):
                # A clock strip along the bottom, so it cannot land on top of
                # the top-left tile's title bar.
                canvas = Image.new("RGB", (2 * TILE_W, 2 * (TILE_H + BAR) + CLOCK), (12, 12, 14))
                draw = ImageDraw.Draw(canvas)
                for k, model in enumerate(MODELS):
                    x, y = (k % 2) * TILE_W, (k // 2) * (TILE_H + BAR)
                    frames = streams.get(model)
                    if not frames:
                        draw.text((x + 10, y + 10), f"{LABELS[model]}: no data", fill=(255, 120, 120))
                        continue
                    # Freeze on the last frame once this model's run has ended.
                    ended = i >= len(frames)
                    with Image.open(frames[min(i, len(frames) - 1)]) as im:
                        tile = im.convert("RGB").resize((TILE_W, TILE_H), Image.BILINEAR)
                    if ended:
                        tile = Image.blend(tile, Image.new("RGB", tile.size, (0, 0, 0)), 0.55)
                    canvas.paste(tile, (x, y + BAR))

                    s = score.get((model, scenario))
                    tag = f"{LABELS[model]}"
                    if s is not None:
                        tag += f"   score {s:.2f}"
                    if ended:
                        tag += "   — run ended"
                    draw.rectangle([x, y, x + TILE_W, y + BAR], fill=(24, 24, 28))
                    draw.text((x + 10, y + 10), tag,
                              fill=(150, 150, 155) if ended else (0, 255, 150))
                base = 2 * (TILE_H + BAR)
                draw.rectangle([0, base, 2 * TILE_W, base + CLOCK], fill=(0, 0, 0))
                draw.text((12, base + 8), f"t = {i * STEP_S:4.1f} s", fill=(255, 255, 255))
                draw.text((160, base + 8),
                          f"scene-0103  {scenario}   |   dimmed tile = that model's run has already ended",
                          fill=(140, 140, 145))
                canvas.save(tmp / f"{i:05d}.png")

            listing = tmp / "list.txt"
            files = sorted(tmp.glob("*.png"))
            lines = [f"file '{f}'" for f in files] + [f"file '{files[-1]}'"] * 15
            listing.write_text("\n".join(lines))

            path = out_dir / f"compare_{scenario}_0103.mp4"
            subprocess.run(
                [FFMPEG, "-loglevel", "error", "-y", "-r", "10", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
                check=True,
            )
            print(f"{path}  ({n} frames, {n * STEP_S:.1f}s + hold)")


if __name__ == "__main__":
    main()
