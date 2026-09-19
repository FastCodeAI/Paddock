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

# onnxruntime, for the ONNX backend (see policy.py). Needed whenever the
# tinygrad JIT pickle cannot be built -- no GPU, no clang, or an unpopulated
# tinygrad_repo -- and harmless otherwise.
uv pip install --python .venv/bin/python onnxruntime

# The driving models are fetched by LFS. The 'big' variant costs ~766 MB and is
# only used when PADDOCK_OPENPILOT_MODEL=big, so it is opt-in.
git lfs install --local
if [ "${PADDOCK_OPENPILOT_BIG:-0}" = "1" ]; then
  git lfs pull
else
  git lfs pull --exclude="openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
fi

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
#
# Not fatal any more. If the JIT cannot be built, `policy.py`'s "auto" backend
# falls back to onnxruntime on the ONNX files, which ship in the repo regardless
# -- so a machine that cannot run scons can still evaluate the model. The run's
# diagnostics record which backend was used, so the two never get conflated.
if scons -j"$(nproc)"; then
  echo "tinygrad JIT built -- the 'auto' backend will prefer it."
else
  echo
  echo "WARNING: scons failed, so openpilot's tinygrad JIT pickle was not built."
  echo "The 'auto' backend will use onnxruntime on the shipped ONNX files instead."
  echo "That is a supported configuration, not a degraded one, but it is not"
  echo "byte-identical to upstream's execution path -- see the notes on"
  echo "PADDOCK_OPENPILOT_BACKEND in model.yaml. Every run records which backend"
  echo "it used in its diagnostics."
fi

echo
echo "done. Verify with: paddock doctor --model openpilot"
echo "Note: tinygrad's PC backend is CPU:LLVM -- supercombo does not use the GPU."
