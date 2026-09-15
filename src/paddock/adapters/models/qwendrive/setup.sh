#!/usr/bin/env bash
# Build Qwen-Drive-1.0's environment.
#
#   PADDOCK_REPO=<the Qwen-Drive-1.0 clone> bash src/paddock/adapters/models/qwendrive/setup.sh
#
# (`paddock setup --model qwendrive` will run this once setup is implemented.)

set -euo pipefail
: "${PADDOCK_REPO:?set PADDOCK_REPO to the Qwen-Drive-1.0 clone}"

cd "${PADDOCK_REPO}"
VENV="${PADDOCK_VENV:-${PADDOCK_REPO}/.venv}"
uv venv "${VENV}" --python 3.12 --allow-existing  # re-runnable: an existing venv is updated
PY="${VENV}/bin/python"

# The versions the released weights were verified with (requirements.txt:
# Python 3.12, CUDA 12.8, torch 2.8.0, transformers 5.14.1).
uv pip install --python "${PY}" --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0

# flash-attn MUST NOT be installed: the local CUDA toolkit cannot emit sm_120
# for this Blackwell card and the build fails (alpamayo15/setup.sh hit the same
# wall). The model runs with attn_implementation="sdpa" -- QWENDRIVE_ATTN in
# model.yaml -- which both the VLM and the planning expert support.
uv pip install --python "${PY}" \
    transformers==5.14.1 accelerate==1.12.0 safetensors==0.8.0 numpy==2.2.6 \
    pillow==12.0.0 flash-linear-attention==0.5.1 opencv-python-headless \
    matplotlib==3.10.7 tqdm==4.67.1 \
    fastapi uvicorn  # paddock.policy.server runs inside this interpreter

# causal-conv1d is a CUDA extension the Qwen3.5 hybrid-attention layers use.
# Without it transformers falls back to a torch implementation: slower, same
# numbers. Try it, record the outcome, never require it.
if uv pip install --python "${PY}" --no-build-isolation causal-conv1d==1.6.2.post1; then
    echo "causal-conv1d: built"
else
    echo "WARNING: causal-conv1d did not build; the torch fallback will be used (slower)."
fi

# No install of the package itself: model.yaml puts {repo}/src on PYTHONPATH,
# and so does this check.
PYTHONPATH="${PADDOCK_REPO}/src" "${PY}" - <<'PYEOF'
import torch, transformers
import qwen_drive
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("qwen_drive", qwen_drive.__version__, "sdpa ok:", qwen_drive.QwenDriveForPlanning._supports_sdpa)
try:
    import causal_conv1d  # noqa: F401
    print("causal-conv1d: importable")
except ImportError:
    print("causal-conv1d: absent (torch fallback)")
PYEOF

echo
echo "Weights are not downloaded here. Place Qwen-Drive-1.0-4B/ (VLM + planner-sft/ + planner-rl/)"
echo "under [roots.weights]. Verify with: paddock doctor --model qwendrive"
