#!/usr/bin/env bash
# Build Alpamayo 2 Edge's environment.
#
#   paddock setup --model alpamayo2edge
#
# Paddock passes the resolved paths in, so this script names none of its own:
#   PADDOCK_REPO      the alpamayo2-edge clone
#   PADDOCK_WEIGHTS   the shared weights root
#   PADDOCK_VENV      where the interpreter is expected (from model.yaml)

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model alpamayo2edge', which sets PADDOCK_REPO}"

cd "${PADDOCK_REPO}"
export UV_PROJECT_ENVIRONMENT="${PADDOCK_VENV:-${PADDOCK_REPO}/.venv}"
# Re-runnable: an interrupted install (a full disk, a dropped connection) leaves
# the venv behind, and `uv sync` below repairs it in place.
if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  uv venv "${UV_PROJECT_ENVIRONMENT}" --python 3.12
fi

# The locked set, as the upstream README installs it: torch 2.8.0, torchvision
# 0.23.0 and transformers 5.17.0, which is the first release with the
# Cosmos3-Edge backbone. No extras: `flash` builds flash-attn from source and
# the upstream default attention is SDPA, which is what this adapter runs; the
# `dataset` extra is the PhysicalAI-AV SDK, and paddock supplies the frames.
uv sync --locked

# The policy node -- `python -m paddock.policy.server`, which paddock talks to
# over localhost -- runs inside this venv, not paddock's own, so it needs
# fastapi and uvicorn here. The model's pyproject has no reason to know about
# paddock, so `uv sync` cannot supply them: without this the checkpoint loads
# and the node dies at import with ModuleNotFoundError: uvicorn.
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" fastapi uvicorn

echo
echo "Weights are not downloaded here. Place nvidia/Alpamayo2-Edge (not gated,"
echo "5.3 GB, revision f8951d095de3589b3face1dec3f4394c8ecedb38) at"
echo "  ${PADDOCK_WEIGHTS:-[roots.weights]}/Alpamayo2-Edge"
echo "It is self-contained: tokenizer, processor and backbone are all in it."
echo
echo "done. Verify with: paddock doctor --model alpamayo2edge"
