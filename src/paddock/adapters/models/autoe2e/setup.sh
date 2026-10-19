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
set -euo pipefail
: "${PADDOCK_REPO:?set PADDOCK_REPO to the auto_e2e checkout}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
PY="${PADDOCK_VENV}/bin/python"
[ "$(git -C "${PADDOCK_REPO}" rev-parse HEAD)" = 40a75cf8a26b60f6b094125e3924fe0bdb427b02 ] || { echo "auto_e2e is not at 40a75cf8"; exit 1; }
uv venv "${PADDOCK_VENV}" --python 3.12
uv pip install --python "${PY}" --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
uv pip install --python "${PY}" "timm==1.0.27" "numpy==2.2.6" "pyproj==3.7.2" "PyYAML==6.0.3" pillow scipy fastapi uvicorn httpx huggingface_hub
"${PY}" -c "import torch; assert torch.version.cuda.startswith('12.8'), torch.__version__"
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
