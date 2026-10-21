#!/usr/bin/env bash
# Build AutoE2E's environment and fetch the v1.0 kitscenes-epoch-5 checkpoint.
#
#   PADDOCK_REPO=<repos>/auto_e2e PADDOCK_WEIGHTS=<weights> bash src/paddock/adapters/models/autoe2e/setup.sh
#
# The checkout must be at the release revision (Hugging Face config.json source.revision):
#   git fetch --depth 1 origin 40a75cf8a26b60f6b094125e3924fe0bdb427b02 && git checkout FETCH_HEAD
# torch comes from the cu128 index in its own step, and timm etc. after it with torch already
# satisfied: installing them together re-resolves torch==2.7.1 from PyPI, the cu126 build, which
# has no sm_120 (Blackwell) kernels.
#
# The KITScenes map and the route matched to it are drawn by upstream's own code, which reads
# the map through the KITScenes SDK (Lanelet2) and renders it with a small C++ library built
# from the checkout. The SDK is pinned where upstream pins it (KITSCENES_SDK_REVISION in
# Model/data_parsing/kit_scenes/source.py) and installed the way upstream's data-prep image
# does (Platform/docker/data-prep/Dockerfile): its Lanelet2 wheel is vendored in the SDK under
# git LFS, and it requires NumPy < 2 -- hence numpy 1.26.4, which the model runs on unchanged.
set -euo pipefail
: "${PADDOCK_REPO:?set PADDOCK_REPO to the auto_e2e checkout}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
PY="${PADDOCK_VENV}/bin/python"
KITSCENES_SDK_REVISION=7765cdec5490894266070ab46e23724b58b3da42
[ "$(git -C "${PADDOCK_REPO}" rev-parse HEAD)" = 40a75cf8a26b60f6b094125e3924fe0bdb427b02 ] || { echo "auto_e2e is not at 40a75cf8"; exit 1; }
command -v git-lfs >/dev/null || { echo "git-lfs is required: the KITScenes SDK vendors its Lanelet2 wheel under git LFS"; exit 1; }
uv venv "${PADDOCK_VENV}" --python 3.12
uv pip install --python "${PY}" --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
uv pip install --python "${PY}" "timm==1.0.27" "numpy==1.26.4" "pyproj==3.7.2" "PyYAML==6.0.3" "opencv-python-headless==4.9.0.80" pillow scipy fastapi uvicorn httpx huggingface_hub
SDK="$(mktemp -d)"
trap 'rm -rf "${SDK}"' EXIT
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet https://github.com/KIT-MRT/kitscenes.git "${SDK}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${SDK}" checkout --quiet "${KITSCENES_SDK_REVISION}"
git -C "${SDK}" lfs pull
# pip, from inside the clone, exactly as upstream installs it: the SDK's metadata names its
# Lanelet2 wheels by paths relative to the clone, which uv refuses to parse ("relative path
# without a working directory") even when the [map] extra is not asked for. The SDK imports
# pyarrow at package import, and pyarrow 26 requires NumPy 2.
uv pip install --python "${PY}" pip
(cd "${SDK}" && "${PY}" -m pip install --quiet ".[map]" "numpy==1.26.4" "pyarrow==25.0.1")
"${PY}" "${PADDOCK_REPO}/Model/navigation/native/build.py"
"${PY}" -c "import torch, lanelet2, kitscenes.map_api; assert torch.version.cuda.startswith('12.8'), torch.__version__"
DEST="${PADDOCK_WEIGHTS}/auto_e2e"
mkdir -p "${DEST}"
"${PY}" - "${DEST}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("AutowareFoundation/auto_e2e", revision="v1.0", local_dir=sys.argv[1],
                  allow_patterns=["config.json", "SHA256SUMS", "release_manifest.json", "models/kitscenes-epoch-5/*"])
PYEOF
(cd "${DEST}" && grep -E "  (config\.json|models/kitscenes-epoch-5/(checkpoint\.pt|metadata\.json))$" SHA256SUMS | sha256sum -c -)
echo "done. Verify with: paddock doctor --model autoe2e"
