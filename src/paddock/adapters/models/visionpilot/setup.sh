#!/usr/bin/env bash
# VisionPilot: the Python policy needs only onnxruntime and the shipped ONNX
# weights, which are in the checkout. Building the C++ binary is a separate,
# heavier job and is not required to run this adapter.
#
#   bash src/paddock/adapters/models/visionpilot/setup.sh

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model visionpilot', which sets PADDOCK_REPO}"
# Created at the path paddock will look in (model.yaml's venv, or the site's
# [models.visionpilot] venv), not by name.
: "${PADDOCK_VENV:?run via 'paddock setup --model visionpilot', which sets PADDOCK_VENV}"

# `conda create -y -p` deletes whatever environment is already at the path, and
# a site's venv can be one other benchmarks run from, so a person has to choose.
if [ -e "${PADDOCK_VENV}" ]; then
  echo "${PADDOCK_VENV} already exists, so nothing was changed." >&2
  echo "To rebuild it, delete that folder and run this again." >&2
  exit 1
fi

conda create -y -p "${PADDOCK_VENV}" python=3.10
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)" && conda activate "${PADDOCK_VENV}"
pip install onnxruntime numpy pillow fastapi uvicorn httpx

W="${PADDOCK_REPO}/VisionPilot/modules/models/weights"
for f in autosteer_fp32.onnx autodrive_fp32.onnx autospeed_fp32.onnx; do
  [ -f "${W}/${f}" ] || echo "WARNING: missing ${W}/${f}"
done

# Reminder for anyone who goes on to build the C++ binary: keep
# OMP_NUM_THREADS=1. Under conda-forge's LLVM-backed libgomp, MUMPS (inside
# IPOPT, called by the MPC planner every frame) deadlocks without it.

echo
echo "done. Verify with: paddock doctor --model visionpilot"
