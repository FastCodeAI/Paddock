#!/usr/bin/env bash
# Build the constant-velocity floor's environment.
#
#   bash src/paddock/adapters/models/constant_velocity/setup.sh
#
# Seconds, not minutes: the policy itself imports numpy and nothing else, and
# there is no repo to clone and no weights to fetch -- so unlike every other
# model here this needs only PADDOCK_VENV.
#
# fastapi and uvicorn are not for the policy, they are for the node. Paddock is
# never installed into a model environment (only its source goes on PYTHONPATH,
# so a model's pinned dependency set is not disturbed), which means every model
# venv has to supply the server dependencies itself. Omit them and the node
# exits with ModuleNotFoundError: uvicorn, before answering /alive.
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
uv pip install --python "${PADDOCK_VENV}/bin/python" numpy fastapi uvicorn

echo "done. Verify with: paddock doctor"
