#!/usr/bin/env bash
# Build Drive-JEPA's environment.
#
#   bash src/paddock/adapters/models/drivejepa/setup.sh
#
# Conda rather than uv: upstream pins torch 2.1/cu121 with an mmcv built
# against it, and the CUDA toolkit is easiest to pin through conda.
#
# If your site keeps conda envs somewhere other than the default below, point
# paddock at it instead of moving anything:
#
#   [models.drivejepa]
#   venv = "/wherever/you/keep/envs/drive-jepa"
#
# The env is created at that path (PADDOCK_VENV), not by name, so it lands
# where paddock will look for it.

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model drivejepa', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:?run via 'paddock setup --model drivejepa', which sets PADDOCK_VENV}"

# `conda create -y -p` deletes whatever environment is already at the path, and
# a site's venv can be one other benchmarks run from, so a person has to choose.
if [ -e "${PADDOCK_VENV}" ]; then
  echo "${PADDOCK_VENV} already exists, so nothing was changed." >&2
  echo "To rebuild it, delete that folder and run this again." >&2
  exit 1
fi

conda create -y -p "${PADDOCK_VENV}" python=3.9
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)" && conda activate "${PADDOCK_VENV}"

# A conda env, unlike a venv, also reads the user's ~/.local site-packages, and
# pip skips any requirement it finds there. Built that way, the env runs only on
# a machine whose ~/.local happens to hold the rest. Off for the build, so every
# package lands in the env; model.yaml turns it off for the node too.
export PYTHONNOUSERSITE=1

conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

cd "${PADDOCK_REPO}"
# The constraints hold torch where upstream pins it. Its requirements leave
# torchvision unpinned (timm pulls it in), and pip takes the newest one for
# Python 3.9, which requires torch 2.8: on 6 Nov 2026 that replaced torch
# 2.1.0+cu121, and mmcv_full 1.7.2, built against 2.1, then failed to load its
# compiled ops ("undefined symbol"), so the agent could not be imported.
pip install -r requirements.txt -c <(printf 'torch==2.1.0\ntorchvision==0.16.0\n')
pip install -e navsim_v2

# The policy node -- `python -m paddock.policy.server` -- runs inside this env,
# not paddock's. Without these the checkpoint loads fine and the node dies at
# import with ModuleNotFoundError: uvicorn. Pinned below the versions that
# dropped Python 3.9.
pip install "fastapi<0.116" "uvicorn<0.34"

# NOTE: mmcv-full's compiled _ext ships SASS for sm_50..sm_86 and embeds no PTX,
# so on newer cards the deformable-attention kernel cannot launch. The patch in
# patches/Drive-JEPA/ falls back to the equivalent pure-PyTorch path; keep
# DRIVE_JEPA_MSDA=auto unless you know your card is sm_86 or older.

echo
echo "done. Verify with: paddock doctor --model drivejepa"
