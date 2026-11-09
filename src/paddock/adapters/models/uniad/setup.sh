#!/usr/bin/env bash
# Build the environment UniAD, VAD and their Bench2Drive checkpoints share.
#
#   paddock setup --model uniad        # then: paddock setup --model uniad_b2d
#
# One conda env, Python 3.9, torch 2.7.1+cu128 and the OpenMMLab 1.x stack compiled for this machine's GPU:
# mmcv-full 1.7.2, mmdet 2.28.2, mmsegmentation 0.29.1, mmdet3d 0.17.1. The older pins in the repos'
# install guides (torch 1.9-1.12, mmcv-full 1.4) have no kernels for recent GPUs; 1.7.2 is the last mmcv-full
# and builds against torch 2.7. Every pin below is the one the golden tests passed with (docs/models/uniad.md).
# Untested end to end as a script: it is the recorded build that produced the env those tests ran in.
#
# The compiled ops are built for one architecture, TORCH_CUDA_ARCH_LIST (default 12.0, Blackwell). Set it
# for your card before running; a build for another card will not load on this one. The mmcv-full build
# compiles ~130 CUDA units and peaks near 6 GB of RAM with MAX_JOBS=4: ~45 minutes.
#
# If your site keeps conda envs elsewhere, set [models.uniad] venv (and the same for vad, uniad_b2d and
# vad_b2d) in paddock.toml; the env is created at PADDOCK_VENV, not by name.

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model uniad', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:?run via 'paddock setup --model uniad', which sets PADDOCK_VENV}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# `conda create -p` replaces whatever is at the path, and four models share this env.
if [ -e "${PADDOCK_VENV}" ]; then
  echo "${PADDOCK_VENV} already exists, so nothing was changed." >&2
  echo "To rebuild it, delete that folder and run this again." >&2
  exit 1
fi

conda create -y -p "${PADDOCK_VENV}" python=3.9
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)" && conda activate "${PADDOCK_VENV}"
# A conda env also reads ~/.local site-packages, and pip skips what it finds there; off, so all of it lands
# in the env. model.yaml turns it off for the node too.
export PYTHONNOUSERSITE=1

# nvcc that can target the card: the system's may be too old (CUDA 12.0 cannot build sm_120).
conda install -y -p "${PADDOCK_VENV}" -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128

CONSTRAINTS="$(mktemp)"
trap 'rm -f "${CONSTRAINTS}"' EXIT
cat > "${CONSTRAINTS}" <<'PINS'
numpy==1.26.4
torch==2.7.1+cu128
torchvision==0.22.1+cu128
opencv-python-headless==4.10.0.84
yapf==0.40.1
matplotlib==3.5.3
mmcv-full==1.7.2
PINS
# torch pulls numpy 2; mmcv/mmdet3d need numpy 1. yapf 0.40.1: newer ones break mmcv's Config dump.
pip install -c "${CONSTRAINTS}" numpy==1.26.4 ninja psutil packaging addict yapf==0.40.1 \
  opencv-python-headless==4.10.0.84

# The CUDA build environment: the env's nvcc, and cuBLAS/cuSPARSE/cuSOLVER headers from torch's own wheels.
SP="${PADDOCK_VENV}/lib/python3.9/site-packages"
export CUDA_HOME="${PADDOCK_VENV}"
export CPATH="${PADDOCK_VENV}/targets/x86_64-linux/include:${SP}/nvidia/cublas/include:${SP}/nvidia/cusparse/include:${SP}/nvidia/cusolver/include:${SP}/nvidia/curand/include:${SP}/nvidia/cufft/include:${SP}/nvidia/nvtx/include:${SP}/nvidia/cuda_runtime/include"
export LIBRARY_PATH="${PADDOCK_VENV}/targets/x86_64-linux/lib:${PADDOCK_VENV}/lib"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

SRC="${PADDOCK_VENV}/src"
mkdir -p "${SRC}"
sdist() {  # the PyPI source tarball of $1==$2; `pip download --no-binary` cannot, its metadata step needs torch
  python - "$1" "$2" <<'PY'
import json, sys, urllib.request
release = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{sys.argv[1]}/{sys.argv[2]}/json"))
print(next(u["url"] for u in release["urls"] if u["packagetype"] == "sdist"))
PY
}

# mmcv-full with its CUDA ops, from source.
curl -L -o "${SRC}/mmcv-full-1.7.2.tar.gz" "$(sdist mmcv-full 1.7.2)"
tar -xzf "${SRC}/mmcv-full-1.7.2.tar.gz" -C "${SRC}"
(cd "${SRC}/mmcv-full-1.7.2" && MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v --no-build-isolation --no-deps .)

pip install -c "${CONSTRAINTS}" mmdet==2.28.2 mmsegmentation==0.29.1 scipy==1.13.1 scikit-learn==1.5.2 \
  scikit-image==0.24.0 matplotlib==3.5.3 shapely==2.0.6 pyquaternion numba==0.60.0 plyfile trimesh==2.35.39 \
  casadi==3.6.7 einops==0.4.1 motmetrics==1.1.3 pandas similaritymeasures descartes fire cachetools \
  pycocotools tqdm prettytable terminaltables seaborn==0.12.2
# Without their dependencies, which would pull opencv-python and an old PyYAML over the pins above.
pip install --no-deps nuscenes-devkit==1.1.11 lyft_dataset_sdk==0.0.8 pytorch-lightning==1.2.5 torchmetrics==0.6.2

# mmdet3d 0.17.1 from its sdist, patched (env/mmdet3d-0.17.1.patch says why, op by op): the sdist ships no
# CUDA sources for its point-cloud ops, none of which a camera-only model calls; they import a stub that
# raises if ever used. roiaware_pool3d's two missing kernels are Bench2DriveZoo's copies of the same 0.17.1
# files. Editable, so the source stays where the patch put it.
curl -L -o "${SRC}/mmdet3d-0.17.1.tar.gz" "$(sdist mmdet3d 0.17.1)"
tar -xzf "${SRC}/mmdet3d-0.17.1.tar.gz" -C "${SRC}"
(cd "${SRC}/mmdet3d-0.17.1" && patch -p1 < "${HERE}/env/mmdet3d-0.17.1.patch")
ZOO="$(dirname "${PADDOCK_REPO}")/Bench2DriveZoo"
if [ ! -d "${ZOO}/mmcv/ops/roiaware_pool3d/src" ]; then
  echo "need Bench2DriveZoo's checkout beside this repo for two roiaware_pool3d kernels:" >&2
  echo "  paddock setup --repo Bench2DriveZoo, then run this again" >&2
  exit 1
fi
cp "${ZOO}"/mmcv/ops/roiaware_pool3d/src/{roiaware_pool3d_kernel.cu,points_in_boxes_cuda.cu} \
  "${SRC}/mmdet3d-0.17.1/mmdet3d/ops/roiaware_pool3d/src/"
(cd "${SRC}/mmdet3d-0.17.1" && FORCE_CUDA=1 pip install -v --no-build-isolation --no-deps -e .)

# The policy node -- `python -m paddock.policy.server` -- runs inside this env, not paddock's.
pip install "fastapi<0.129" "uvicorn<0.40"

echo
echo "done. Weights go under <weights>/uniad, <weights>/vad and <weights>/bench2drivezoo (docs/models/uniad.md)."
echo "Then: paddock setup --model uniad_b2d (Bench2DriveZoo's own ops), and paddock doctor --model uniad"
