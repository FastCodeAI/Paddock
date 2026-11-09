#!/usr/bin/env bash
# Build Bench2DriveZoo's own `mmcv` (mmcv + mmdet + mmdet3d 0.17.1 vendored as one package) in place, in the
# env UniAD's setup.sh builds. Shared by uniad_b2d and vad_b2d.
#
#   paddock setup --model uniad && paddock setup --model uniad_b2d
#
# In place, never pip-installed: installed, it would replace the env's mmcv-full that uniad and vad import.
# The node puts the checkout first on PYTHONPATH instead (model.yaml). patches/Bench2DriveZoo/ must be
# applied first (paddock setup does): 0002 is what lets this compile against torch 2.x.
set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model uniad_b2d', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:?run via 'paddock setup --model uniad_b2d', which sets PADDOCK_VENV}"
if [ ! -x "${PADDOCK_VENV}/bin/python" ]; then
  echo "no env at ${PADDOCK_VENV}: run 'paddock setup --model uniad' first" >&2
  exit 1
fi
SP="${PADDOCK_VENV}/lib/python3.9/site-packages"
export PATH="${PADDOCK_VENV}/bin:${PATH}" PYTHONNOUSERSITE=1 CUDA_HOME="${PADDOCK_VENV}"
export CPATH="${PADDOCK_VENV}/targets/x86_64-linux/include:${SP}/nvidia/cublas/include:${SP}/nvidia/cusparse/include:${SP}/nvidia/cusolver/include:${SP}/nvidia/curand/include:${SP}/nvidia/cufft/include:${SP}/nvidia/nvtx/include:${SP}/nvidia/cuda_runtime/include"
export LIBRARY_PATH="${PADDOCK_VENV}/targets/x86_64-linux/lib:${PADDOCK_VENV}/lib"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS="${MAX_JOBS:-4}"
cd "${PADDOCK_REPO}"
python setup.py build_ext --inplace
python -c "import sys; sys.path.insert(0, '.'); import mmcv; assert mmcv.__file__.startswith('$(pwd)'), mmcv.__file__; from mmcv.models import build_model; print('Bench2DriveZoo mmcv ok')"
