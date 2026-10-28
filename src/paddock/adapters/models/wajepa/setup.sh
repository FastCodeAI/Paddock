#!/usr/bin/env bash
# Build WA-JEPA's environment and fetch the released checkpoint.
#
#   PADDOCK_REPO=<the WA-JEPA clone> PADDOCK_WEIGHTS=<weights root> \
#     bash src/paddock/adapters/models/wajepa/setup.sh
#
# (`paddock setup --model wajepa` will run this once setup is implemented.)
#
# Upstream's recipe (docs/installation.md) is a conda python 3.10 env with NAVSIM
# and the nuPlan devkit installed on top, and NAVSIM pins torch 2.0.1. None of
# that is needed here: the adapter drives the model the way upstream's own
# closed-loop client does (close_loop/hugsim_planner.py), which imports only the
# model package -- models/, third_party/vjepa2/, training/checkpoint.py,
# utils/utils.py -- and those need torch, timm, einops and omegaconf. Leaving
# NAVSIM out is also what lets torch be new enough for the Blackwell card on the
# reference box: torch 2.0.1 has no sm_120 kernels.

set -euo pipefail
: "${PADDOCK_REPO:?set PADDOCK_REPO to the WA-JEPA clone ('paddock setup' is not implemented yet)}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
# Never stage into ~/.cache on the root filesystem.
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
PY="${PADDOCK_VENV}/bin/python"

echo "building environment for wajepa in ${PADDOCK_VENV}"
uv venv "${PADDOCK_VENV}" --python 3.10 --allow-existing  # upstream's python; re-runnable

# cu128 is the first CUDA build with Blackwell (sm_120) kernels. Upstream leaves
# torch unpinned because "CUDA wheels are environment-specific"; this is the one
# the reference box verified. torchvision is never imported by WA-JEPA but timm
# requires it, and left to the next step it resolves from PyPI to a newer torch
# built for CUDA 13 -- replacing this torch and unpacking cu13 NVIDIA wheels over
# the cu12 ones. Pinning the matching torchvision here stops that.
uv pip install --python "${PY}" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0

# requirements.txt minus the training-only packages (accelerate, torchmetrics,
# wandb, deepspeed) and the plotting ones. numpy stays below 2 because upstream
# tests its paths on 1.23-1.26. opencv-python-headless rather than opencv-python:
# the node has no display, and the GUI build drags in X11 libraries.
# fastapi/uvicorn/pillow are the paddock node's own needs, not WA-JEPA's.
uv pip install --python "${PY}" \
  "numpy>=1.23.4,<2" "timm>=0.9.0,<1.0" "einops>=0.7.0" "omegaconf>=2.3.0" \
  "opencv-python-headless>=4.5.0" pyyaml huggingface_hub \
  fastapi uvicorn pillow

# The released stage-2 checkpoint (README "Checkpoints"). model_state_dict.pt
# is the whole model; state.pt is trainer state and not needed for inference,
# so it is not fetched. 1.58 GB.
DEST="${PADDOCK_WEIGHTS}/wa-jepa"
mkdir -p "${DEST}"
"${PY}" - "${DEST}" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download("AFARI-Research/WA-JEPA", "model_state_dict.pt", local_dir=sys.argv[1])
PYEOF

"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
echo "done. Verify with: paddock doctor --model wajepa"
