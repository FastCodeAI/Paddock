#!/usr/bin/env bash
# Build HiP-AD's environment and fetch its checkpoint.
#
#   paddock setup --model hipad
#
# HiP-AD's requirements (mmdet 2.28.2, mmcv-full 1.7.x) are the OpenMMLab 1.x stack UniAD and VAD run on,
# so the base is the env uniad/setup.sh builds -- torch 2.7.1+cu128 and mmcv-full 1.7.2 compiled for this
# GPU (~45 minutes) -- built at this model's own path. HiP-AD then needs two things of its own:
#
# * flash-attn: the config's attention layers are MultiheadFlashAttention. The prebuilt 2.8.3 wheel for
#   torch 2.7 / CUDA 12 / Python 3.9 / the C++11 ABI, pinned by its sha256; it runs on Blackwell (sm_120).
# * its deformable aggregation op (projects/mmdet3d_plugin/ops), compiled in place for TORCH_CUDA_ARCH_LIST
#   (default 12.0, Blackwell; set it for your card). About a minute.
#
# The README's own pins (torch 1.13 + CUDA 11.7) have no kernels for recent GPUs. Every pin below is the
# one the agent comparison passed with (docs/models/hipad.md, "Verification").

set -euo pipefail

: "${PADDOCK_REPO:?run via 'paddock setup --model hipad', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:?run via 'paddock setup --model hipad', which sets PADDOCK_VENV}"
: "${PADDOCK_WEIGHTS:?run via 'paddock setup --model hipad', which sets PADDOCK_WEIGHTS}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "${PADDOCK_VENV}/bin/python" ]; then
  echo "building the UniAD/VAD stack at ${PADDOCK_VENV} (uniad/setup.sh)"
  bash "${HERE}/../uniad/setup.sh"
fi
PY="${PADDOCK_VENV}/bin/python"
export PYTHONNOUSERSITE=1
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fetch() {  # url, file, sha256
  curl -L --fail -o "$2" "$1"
  echo "$3  $2" | sha256sum -c -
}

WHEEL="flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp39-cp39-linux_x86_64.whl"
fetch "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL}" "${TMP}/${WHEEL}" \
  680375ddd2fa2a9c95d55aee8a64ed64be3f94b389520a1a0fa33880c4759d10
"${PY}" -m pip install --no-deps "${TMP}/${WHEEL}"

# The env's own nvcc (CUDA 12.8): the system's may be too old to target the card.
export PATH="${PADDOCK_VENV}/bin:${PATH}" CUDA_HOME="${PADDOCK_VENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS="${MAX_JOBS:-4}"
(cd "${PADDOCK_REPO}/projects/mmdet3d_plugin/ops" && "${PY}" setup.py build_ext --inplace)

mkdir -p "${PADDOCK_WEIGHTS}/HiP-AD"
CKPT="${PADDOCK_WEIGHTS}/HiP-AD/hipad_stage2.pth"
if [ ! -f "${CKPT}" ]; then
  fetch https://github.com/nullmax-vision/HiP-AD/releases/download/weighs/hipad_stage2.pth "${CKPT}" \
    7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956
fi

(cd "${PADDOCK_REPO}" && "${PY}" -c "import flash_attn; from projects.mmdet3d_plugin.ops import deformable_aggregation_ext")
echo "done. Check against HiP-AD's own agent with: make check-hipad (docs/models/hipad.md)"
