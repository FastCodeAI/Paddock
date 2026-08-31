#!/usr/bin/env bash
# Build openpilot's environment.
#
#   bash src/paddock/adapters/models/openpilot/setup.sh
#
# Needs PADDOCK_REPO (paddock setup supplies it). ~10 minutes, most of it in
# scons compiling the ONNX model into tinygrad JIT pickles.

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model openpilot', which sets PADDOCK_REPO}"

cd "${PADDOCK_REPO}"

# openpilot vendors several repos as submodules (tinygrad, opendbc, panda, msgq).
git submodule update --init --recursive --jobs 6

uv sync --frozen --all-extras

# The large driving model is fetched by LFS. The 'big' variant is excluded: it
# is not what supercombo uses and costs a multi-GB download.
git lfs install --local
git lfs pull --exclude="openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"

# Compiles the ONNX graph into the tinygrad pickles modeld loads at run time.
# Needs clang, which openpilot's own tools/setup_dependencies.sh does NOT
# install -- install it from your distro first if this fails.
scons -j"$(nproc)"

echo
echo "done. Verify with: paddock doctor --model openpilot"
echo "Note: tinygrad's PC backend is CPU:LLVM -- supercombo does not use the GPU."
