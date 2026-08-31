#!/usr/bin/env bash
# Build this model's environment from nothing, on a machine that has never run
# it. This script is what makes the model runnable at the other site, so prefer
# being explicit over being short.
#
#   bash src/paddock/adapters/models/<name>/setup.sh
#
# Paddock passes the resolved paths in as environment variables so this script
# contains no absolute paths of its own:
#
#   PADDOCK_REPO      this model's clone
#   PADDOCK_WEIGHTS   the shared weights root
#   PADDOCK_VENV      where the interpreter is expected (from model.yaml)

set -euo pipefail

: "${PADDOCK_REPO:?run this through 'paddock setup --model <name>', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"

echo "building environment for template in ${PADDOCK_VENV}"

cd "${PADDOCK_REPO}"

# Example, uv-managed:
# export UV_PROJECT_ENVIRONMENT="${PADDOCK_VENV}"
# uv venv "${PADDOCK_VENV}" --python 3.12
# uv sync

# Record anything non-obvious as a comment here AND in docs/models/<name>.md:
# a pinned CUDA version, a package that must NOT be installed, a build that
# needs a compiler the OS does not ship. The next person will not guess it.

echo "done. Verify with: paddock doctor --model template"
