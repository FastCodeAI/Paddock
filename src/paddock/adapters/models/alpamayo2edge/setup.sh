#!/usr/bin/env bash
# Build Alpamayo 2 Edge's environment and link its pinned checkpoint.
#
#   HF_HOME=<the site's Hugging Face cache> paddock setup --model alpamayo2edge
#
# The environment is the vendor's own lock (uv.lock at 02f55a7: Python 3.12,
# torch 2.8.0 with CUDA 12.8, transformers 5.17.0), with SDPA attention; the
# flash-attn extra is not installed and the adapter does not use it. The
# checkpoint is nvidia/Alpamayo2-Edge at revision f8951d0, 5.0 GB, not gated. It
# goes into the Hugging Face cache at HF_HOME, and {weights}/Alpamayo2-Edge links
# to its pinned snapshot, so downloading another revision later cannot change
# what a run is scored against.

set -euo pipefail

: "${PADDOCK_REPO:?run via 'paddock setup --model alpamayo2edge', which sets PADDOCK_REPO}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
# Never stage into ~/.cache on the root filesystem.
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
REPO_ID="nvidia/Alpamayo2-Edge"
REVISION="f8951d095de3589b3face1dec3f4394c8ecedb38"
LINK="${PADDOCK_WEIGHTS}/Alpamayo2-Edge"
PY="${PADDOCK_VENV}/bin/python"

echo "building environment for alpamayo2edge in ${PADDOCK_VENV}"
cd "${PADDOCK_REPO}"
UV_PROJECT_ENVIRONMENT="${PADDOCK_VENV}" uv sync --locked --python 3.12
# The node, `python -m paddock.policy.server`, runs in this environment and needs
# fastapi and uvicorn, which the vendor's lock has no reason to carry. Installed
# after the sync, because a sync removes what the lock does not list.
uv pip install --python "${PY}" fastapi uvicorn

echo "fetching ${REPO_ID} at ${REVISION} into ${HF_HOME}"
SNAPSHOT=$(HF_HUB_DISABLE_PROGRESS_BARS=1 "${PY}" - "${REPO_ID}" "${REVISION}" <<'PYEOF'
import sys

from huggingface_hub import snapshot_download

repo, revision = sys.argv[1:3]
# What the vendor's resolve_checkpoint fetches. A cached snapshot is used without the network.
patterns = ["*.json", "*.jinja", "*.safetensors", "*.txt", "*.model"]
try:
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=True)
except Exception:  # noqa: BLE001 - not cached yet
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns)
print(path)
PYEOF
)

# Every file the model reads, as fetched at the pin on 2026-11-10.
(cd "${SNAPSHOT}" && sha256sum -c --quiet - <<'SUMS'
d13c1a427218baf7b7d365098c0f5496c0e5e6a6bf662836b8eebbe7eb7cfe7f  00000.safetensors
cfe424a7320ce729cd927b5ceaab4323a62443b50f461a2f5c0dd0b10634e0b5  00001.safetensors
0e922b870e530b52a3b92c7b8d3abd3e7eb14714c310b32069bf1ab8e59c1390  model.safetensors.index.json
784fc20f67294d7a4805e860033a88339cd81b7c0f9bd99d1dc95564fe159dc7  config.json
09348f5c6c3c065e0a2a3f12c8292ddeae64bd45671cc464e2015f01517a50c4  generation_config.json
a948fca6731cde35e0e0a011a2450745fb7084bbfbbf90da4152fff76a90bdc7  processor_config.json
7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8  chat_template.jinja
06a854e66144d6675e604d22ccb5a39c35645d5caf651dee9ec78f5aa28ff615  tokenizer.json
d58bf4bcb16c6e7815bfb8a0d2055f669d2142f64f38cf2935a3e3a3b334b308  tokenizer_config.json
SUMS
)
echo "sha256 ok: the checkpoint is the pinned one"

if [ -e "${LINK}" ] && [ ! -L "${LINK}" ]; then
  echo "${LINK} exists and is not a link; move it aside and run again" >&2
  exit 1
fi
# Relative, so the weights folder can be moved or mounted elsewhere whole.
ln -sfn "$(realpath -m --relative-to="${PADDOCK_WEIGHTS}" "${SNAPSHOT}")" "${LINK}"
echo "linked ${LINK} -> $(readlink "${LINK}")"

"${PY}" -c "import torch, transformers, fastapi, uvicorn, alpamayo2_edge; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "done. Verify with: paddock doctor --model alpamayo2edge, then paddock verify --model alpamayo2edge"
