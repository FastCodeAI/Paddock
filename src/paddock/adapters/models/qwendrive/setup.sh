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

# flash-attn MUST NOT be installed, and two independent things stop it:
#
#   1. the local CUDA toolkit cannot emit sm_120 for this Blackwell card and the
#      build fails (alpamayo15/setup.sh hit the same wall);
#   2. on a CUDA 13 / torch 2.14 site it fails for another reason entirely --
#      torch 2.14's ATen headers need C++20 while flash-attn's setup.py appends
#      -std=c++17 to extra_compile_args, i.e. *after* any CXXFLAGS from the
#      environment, so the usual override is silently discarded.
#
# It is a speed optimisation for the 8 full-attention layers and changes no
# result. The model runs with attn_implementation="sdpa" -- QWENDRIVE_ATTN in
# model.yaml -- which both the VLM and the planning expert support.
#
# flash-linear-attention is installed WITH its dependencies. As of 0.3.2 the
# project split in two: `fla.layers` and `fla.models` stayed behind while
# `fla.modules` and `fla.ops` moved to a separate `fla-core` distribution. A
# --no-deps install therefore yields a package that imports and then fails at
# `from fla.modules import FusedRMSNormGated`, surfacing as "Could not import
# module 'Qwen3_5ForConditionalGeneration'" -- which reads exactly like a
# truncated checkpoint download and is not one.
uv pip install --python "${PY}" \
    transformers==5.14.1 accelerate==1.12.0 safetensors==0.8.0 numpy==2.2.6 \
    pillow==12.0.0 flash-linear-attention==0.5.1 opencv-python-headless \
    matplotlib==3.10.7 tqdm==4.67.1 \
    fastapi uvicorn  # paddock.policy.server runs inside this interpreter

# causal-conv1d is a CUDA extension the Qwen3.5 hybrid-attention layers use --
# `layer_types` is 3x linear_attention then 1x full_attention, repeated 8x, and
# the 24 Gated DeltaNet layers take a slow reference path without it. Check with
# `transformers.models.qwen3_5.modeling_qwen3_5.is_fast_path_available`.
# Without it transformers falls back to a torch implementation: slower, same
# numbers. Try it, record the outcome, never require it. On a torch 2.14 site
# the build needs CXXFLAGS="-std=c++20" (ATen headers); this extension, unlike
# flash-attn, does accept it from the environment.
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
echo "under [roots.weights], or point QWENDRIVE_CHECKPOINT at the snapshot directory"
echo "holding config.json, model.safetensors and planner-sft/ if the weights are an HF"
echo "cache. Set HF_HOME to a data disk first -- the default stages into the root"
echo "filesystem. Verify with: paddock doctor --model qwendrive"
