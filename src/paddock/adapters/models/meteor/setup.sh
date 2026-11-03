#!/usr/bin/env bash
# Build METEOR's environment and fetch the released ONNX.
#
#   PADDOCK_REPO=<repos>/METEOR PADDOCK_WEIGHTS=<weights> \
#     bash src/paddock/adapters/models/meteor/setup.sh
#
# METEOR ships no pyproject or requirements file; its documented recipe is a
# python3.10 venv with torch 2.1.1/cu121 for training and TensorRT. The adapter
# only runs the plain ONNX, so the env is onnxruntime plus what the paddock node
# needs. There is no /usr/local/cuda on the reference box: the CUDA runtime,
# cuDNN and cuBLAS come from NVIDIA's pip wheels, which
# onnxruntime.preload_dlls() picks up before the session is created.

set -euo pipefail

: "${PADDOCK_REPO:?run via 'paddock setup --model meteor', which sets PADDOCK_REPO}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
# Never stage into ~/.cache on the root filesystem.
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
PY="${PADDOCK_VENV}/bin/python"

echo "building environment for meteor in ${PADDOCK_VENV}"
uv venv "${PADDOCK_VENV}" --python 3.11
# onnxruntime-gpu 1.30 is a CUDA 13 build (it dlopens libcudart.so.13 and
# friends). NVIDIA's CUDA 13 wheels dropped the -cu12 suffix: nvidia-cuda-runtime,
# nvidia-cublas, ... ; only cuDNN keeps a -cu13 name. The -cu12 wheels do not
# satisfy it, and the preload check at the end is what catches a mismatch.
# Do not mix generations in one env: the -cu12 and CUDA 13 wheels unpack into
# the same nvidia/<lib>/ directories, so uninstalling one deletes files the
# other still records (seen as 'dlopen failed for libcudnn.so' at run time).
uv pip install --python "${PY}" \
  "onnxruntime-gpu==1.30.*" "numpy<3" pillow fastapi uvicorn httpx huggingface_hub onnx \
  "nvidia-cuda-runtime==13.*" "nvidia-cublas==13.*" "nvidia-cuda-nvrtc==13.*" \
  nvidia-cufft nvidia-curand nvidia-cudnn-cu13

DEST="${PADDOCK_WEIGHTS}/meteor"
mkdir -p "${DEST}"
"${PY}" - "${DEST}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    "AutowareFoundation/meteor",
    revision="v1.0",
    local_dir=sys.argv[1],
    allow_patterns=["meteor_v157c3Z.onnx", "meteor_v157.param.yaml", "SHA256SUMS"],
)
PYEOF
# SHA256SUMS lines are "<hex>  <relative path>"; check only what we fetched.
(cd "${DEST}" && grep -E '  (meteor_v157c3Z\.onnx|meteor_v157\.param\.yaml)$' SHA256SUMS | sha256sum -c -)

"${PY}" -c "import onnxruntime as ort; ort.preload_dlls(); print('onnxruntime', ort.__version__, ort.get_available_providers())"
echo "done. Verify with: paddock doctor --model meteor"
