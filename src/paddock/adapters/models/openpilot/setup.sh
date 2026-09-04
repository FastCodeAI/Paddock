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

# paddock.policy.server (the HTTP node open- and closed-loop benchmarks both
# talk to) runs inside this venv, not paddock's own -- so it needs fastapi and
# uvicorn here too. openpilot's own pyproject.toml has no reason to know about
# either, so this can't come from `uv sync`.
uv pip install --python .venv/bin/python fastapi uvicorn

# The large driving model is fetched by LFS. The 'big' variant is excluded: it
# is not what supercombo uses and costs a multi-GB download.
git lfs install --local
git lfs pull --exclude="openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"

# Compiles the ONNX graph into the tinygrad pickles modeld loads at run time.
# Needs clang, which openpilot's own tools/setup_dependencies.sh does NOT
# install -- install it from your distro first if this fails. On a machine
# with no clang at all (only gcc/g++) and no way to install one, tinygrad's
# CPU JIT backend still hardcodes `getenv("CC", 'clang')` -- and SConstruct's
# own subprocess environment strips CC/CXX even if you export them, so that
# env var will not reach it. The only working fix found in practice was a
# `clang` shim on PATH ahead of everything else that rewrites the call to
# gcc, strips the clang-only `--target=x86_64-none-unknown-elf` triple gcc
# rejects, and writes to a temp file instead of a pipe (gcc's `as` cannot
# assemble straight to a pipe the way clang's integrated assembler can).
scons -j"$(nproc)"

echo
echo "done. Verify with: paddock doctor --model openpilot"
echo "Note: tinygrad's PC backend is CPU:LLVM -- supercombo does not use the GPU."
