#!/usr/bin/env bash
# Build Alpamayo 1.5's environment.
#
#   bash src/paddock/adapters/models/alpamayo15/setup.sh

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model alpamayo15', which sets PADDOCK_REPO}"

cd "${PADDOCK_REPO}"
export UV_PROJECT_ENVIRONMENT="${PADDOCK_VENV:-${PADDOCK_REPO}/a1_5_venv}"
uv venv "${UV_PROJECT_ENVIRONMENT}" --python 3.12

# flash-attn MUST NOT be installed. It is in the project's dependencies, but the
# local CUDA toolkit cannot emit sm_120 for Blackwell cards and the build fails.
# The model runs with attn="sdpa" instead -- see model_loader.build_config.
# If you "fix" the missing dependency, you break the install.
uv sync --no-install-package flash-attn

# The policy node -- `python -m paddock.policy.server`, which paddock talks to
# over localhost -- runs inside this venv, not paddock's own, so it needs
# fastapi and uvicorn here. Alpamayo's own pyproject.toml has no reason to know
# about paddock, so `uv sync` cannot supply them: without this the checkpoint
# loads fine and the node dies at import with ModuleNotFoundError: uvicorn.
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" fastapi uvicorn

echo
echo "Weights are not downloaded here. Place them under [roots.weights]:"
echo "  Alpamayo-1.5-10B    (gated; needs a Hugging Face token)"
echo "  Cosmos-Reason2-8B"
echo "Set HF_HOME to a data disk first -- the default stages into the root filesystem."
echo
echo "done. Verify with: paddock doctor --model alpamayo15"
