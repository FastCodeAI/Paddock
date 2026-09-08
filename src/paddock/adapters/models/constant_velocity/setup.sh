#!/usr/bin/env bash
# Build the constant-velocity floor's environment.
#
#   bash src/paddock/adapters/models/constant_velocity/setup.sh
#
# Seconds, not minutes: the policy imports numpy and nothing else. There is no
# repo to clone and no weights to fetch, so unlike every other model here this
# needs only PADDOCK_VENV.
#
# A site that would rather not keep a venv for four lines of arithmetic can skip
# this entirely and point paddock at its own interpreter, which already has
# numpy:
#
#   [models.constant_velocity]
#   venv = "/path/to/paddock/.venv"

set -euo pipefail
: "${PADDOCK_VENV:?run via 'paddock setup --model constant_velocity', which sets PADDOCK_VENV}"

echo "building the constant-velocity floor's environment in ${PADDOCK_VENV}"
uv venv "${PADDOCK_VENV}" --python 3.11
uv pip install --python "${PADDOCK_VENV}/bin/python" numpy

echo "done. Verify with: paddock doctor"
