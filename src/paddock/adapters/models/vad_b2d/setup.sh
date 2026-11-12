#!/usr/bin/env bash
# vad_b2d shares uniad_b2d's build: Bench2DriveZoo's vendored ops, built in place in the shared env at
# PADDOCK_VENV (paddock setup --model uniad builds that env first).
set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model vad_b2d', which sets PADDOCK_REPO}"
: "${PADDOCK_VENV:?run via 'paddock setup --model vad_b2d', which sets PADDOCK_VENV}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../uniad_b2d/setup.sh"
