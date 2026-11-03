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

conda create -y -p "${PADDOCK_VENV}" python=3.9
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)" && conda activate "${PADDOCK_VENV}"

conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

cd "${PADDOCK_REPO}"
pip install -r requirements.txt
pip install -e navsim_v2

# NOTE: mmcv-full's compiled _ext ships SASS for sm_50..sm_86 and embeds no PTX,
# so on newer cards the deformable-attention kernel cannot launch. The patch in
# patches/Drive-JEPA/ falls back to the equivalent pure-PyTorch path; keep
# DRIVE_JEPA_MSDA=auto unless you know your card is sm_86 or older.

echo
echo "done. Verify with: paddock doctor --model drivejepa"
