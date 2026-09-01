#!/bin/bash
# Run Drive-JEPA inference on a few scenes and render BEV trajectory plots.
#
#   bash scripts/inference/run_inference_and_visualize.sh [NUM_SCENES]
#
# Assumes exports.sh has been sourced and the drive-jepa conda env is active.
set -euo pipefail

export NAVSIM_DEVKIT_ROOT=$(pwd)
NUM_SCENES="${1:-8}"

python -u "$NAVSIM_DEVKIT_ROOT/scripts/inference/run_inference.py" --limit "$NUM_SCENES"
python -u "$NAVSIM_DEVKIT_ROOT/scripts/inference/visualize_bev.py" --grid
