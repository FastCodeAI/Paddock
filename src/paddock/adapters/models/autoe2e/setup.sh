#!/usr/bin/env bash
# Build AutoE2E's environment from nothing.
#
#   bash src/paddock/adapters/models/autoe2e/setup.sh
#
# Paddock passes the resolved paths in, so this script contains no absolute
# paths of its own:
#
#   PADDOCK_REPO      this model's clone
#   PADDOCK_WEIGHTS   the shared weights root
#   PADDOCK_VENV      where the interpreter is expected (from model.yaml)
#
# Non-obvious, and repeated in docs/models/autoe2e.md:
#
#   * Python 3.12 is a hard ceiling, not a preference. Upstream's pinned torch
#     build publishes no wheels for 3.13+, and pip resolves to a much older
#     torch instead of failing, which then breaks on the model's own ops.
#   * lanelet2 is optional and usually the thing that fails. Without it the
#     policy sends a zero map tile with the validity gate off, which is
#     upstream's own no-map path -- not a crash, but a different measurement.
#   * There is no public AutoE2E checkpoint. This script does not fetch one.

set -euo pipefail

: "${PADDOCK_REPO:?run this through 'paddock setup --model autoe2e', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
: "${PADDOCK_WEIGHTS:=}"

# cu121 works too; CPU wheels are the upstream default and are enough for the
# fixture test but far too slow for a 223-sample sweep.
TORCH_CHANNEL="${TORCH_CHANNEL:-cu118}"

echo "building environment for autoe2e in ${PADDOCK_VENV}"

cd "${PADDOCK_REPO}"

export UV_PROJECT_ENVIRONMENT="${PADDOCK_VENV}"
uv venv "${PADDOCK_VENV}" --python 3.12

# Upstream ships requirements.txt rather than a lock, and drives it through its
# own Makefile with the same --extra-index-url. Installed with uv here so the
# interpreter is the one model.yaml declares.
uv pip install --python "${PADDOCK_VENV}/bin/python" \
    -r requirements.txt \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"

# The KITScenes SDK, which the dataset parser imports. --no-deps is upstream's
# instruction: the SDK pins versions that conflict with the model's torch.
KITSCENES_SDK="${PADDOCK_REPO}/Model/data_parsing/kit_scenes/kitscenes"
if [ ! -d "${KITSCENES_SDK}" ]; then
    git clone https://github.com/KIT-MRT/kitscenes.git "${KITSCENES_SDK}"
fi
uv pip install --python "${PADDOCK_VENV}/bin/python" -e "${KITSCENES_SDK}" --no-deps

# Optional: map tiles. pip has no lanelet2 wheel on many platforms, and the
# documented fallback is conda's robostack channel:
#
#   conda install -c robostack-staging ros-humble-lanelet2-python
#
# Left non-fatal deliberately -- a site without lanelet2 can still run every
# other part of this model.
if uv pip install --python "${PADDOCK_VENV}/bin/python" lanelet2 2>/dev/null; then
    echo "lanelet2 installed; set AUTOE2E_ENABLE_MAP=1 to render map tiles"
else
    echo "lanelet2 unavailable -- AutoE2E will run with a zero map tile."
    echo "  See docs/models/autoe2e.md for what that costs and how to fix it."
fi

if [ -n "${PADDOCK_WEIGHTS}" ] && [ ! -f "${PADDOCK_WEIGHTS}/autoe2e/autoe2e.pt" ]; then
    echo
    echo "NOTE: no checkpoint at \${weights}/autoe2e/autoe2e.pt."
    echo "  AutoE2E's planner head is randomly initialised without one and the"
    echo "  policy refuses to load rather than scoring it. See docs/models/autoe2e.md."
fi

echo "done. Verify with: paddock doctor --model autoe2e"
