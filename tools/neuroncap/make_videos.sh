#!/usr/bin/env bash
# Stitch NeuroNCAP's per-step debug frames into videos, one per model/scenario.
#
# The simulation steps at 10 Hz, so 10 fps plays back in real time -- worth
# keeping, because the whole question these clips answer is whether a model
# reacted early enough. The last frame is held for 1.5 s so the outcome (the
# CRASHED overlay, or its absence) is actually readable.
set -euo pipefail

FFMPEG=/shakaal_hdd_1/workspace/naren/renesas/openpilot/.venv/bin/ffmpeg
ROOT="${NCAP_RUNS:-/mnt/p4/naren/neuroncap/outputs/runs1}"
OUT="${NCAP_VIDEO_OUT:-/mnt/p4/naren/neuroncap/videos}"
mkdir -p "$OUT"

for model in "$@"; do
  for scenario in stationary frontal side; do
    frames="$ROOT/$model/$scenario-0103/COMBINED_OUTPUTS"
    [ -d "$frames" ] || { echo "skip $model/$scenario (no frames)"; continue; }
    n=$(ls "$frames"/*.png 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "skip $model/$scenario (empty)"; continue; }

    # Frame filenames are microsecond timestamps, so lexical order is chronological.
    # The final frame is repeated rather than held with tpad, which this ffmpeg
    # build does not carry.
    list=$(mktemp)
    for f in $(ls "$frames"/*.png | sort); do echo "file '$f'"; done > "$list"
    last=$(ls "$frames"/*.png | sort | tail -1)
    for _ in $(seq 15); do echo "file '$last'"; done >> "$list"

    "$FFMPEG" -loglevel error -y -r 10 -f concat -safe 0 -i "$list" \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
      -c:v libx264 -pix_fmt yuv420p -crf 20 \
      "$OUT/${model}_${scenario}_0103.mp4"
    rm -f "$list"
    echo "$OUT/${model}_${scenario}_0103.mp4  ($n frames, $(echo "scale=1; $n/10" | bc)s + hold)"
  done
done
