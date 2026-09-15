#!/usr/bin/env bash
# Build Qwen-Drive 1.0's environment.
#
#   bash src/paddock/adapters/models/qwendrive/setup.sh
#
# Qwen-Drive ships a requirements.txt rather than a lockfile, and two of its
# dependencies have to be compiled here, so this is a `shell` runtime rather
# than `uv sync`.

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model qwendrive', which sets PADDOCK_REPO}"

VENV="${PADDOCK_VENV:-${PADDOCK_REPO}/.venv}"
python3 -m venv "${VENV}"
PY="${VENV}/bin/python"

"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install -r "${PADDOCK_REPO}/requirements.txt"

# Qwen3.5 attention is hybrid: `layer_types` is 3x linear_attention then 1x
# full_attention, repeated 8x. The 24 linear layers are Gated DeltaNet and take
# a slow reference path unless these resolve. Check afterwards with:
#
#   from transformers.models.qwen3_5 import modeling_qwen3_5 as m
#   m.is_fast_path_available   # must be True
#
# torch 2.14's ATen headers require C++20 and the system gcc defaults to C++17,
# so the standard has to be forced. This one accepts it from the environment.
MAX_JOBS="${MAX_JOBS:-16}" CAUSAL_CONV1D_FORCE_BUILD=TRUE \
  CXXFLAGS="-std=c++20" CFLAGS="-std=c++20" \
  "${PY}" -m pip install causal-conv1d==1.6.2.post1 --no-build-isolation

# BOTH of these, and both under --no-deps. As of 0.3.2 the project split in two:
# `fla.layers` and `fla.models` stayed in flash-linear-attention while
# `fla.modules` and `fla.ops` moved to a separate `fla-core` distribution. The
# repo's requirements.txt names only the first, so a --no-deps install yields a
# package that imports but fails at `from fla.modules import FusedRMSNormGated`,
# surfacing as `ModuleNotFoundError: Could not import module
# 'Qwen3_5ForConditionalGeneration'` -- which looks exactly like a truncated
# download and is not one. The wheel sizes give it away: 0.3.1 is 656 KB,
# 0.3.2 is 268 KB.
"${PY}" -m pip install --no-deps flash-linear-attention==0.5.1 fla-core==0.5.1

# flash-attn is deliberately NOT installed, for two independent reasons, and
# the policy asks for `sdpa` instead (QWENDRIVE_ATTN in model.yaml):
#
#   1. The C++20 trick above does not work here. flash-attn's setup.py puts
#      -std=c++17 in extra_compile_args, which distutils emits *after* the
#      CXXFLAGS from the environment, so the env var is silently overridden and
#      the ATen C++20 errors come straight back.
#   2. flash-attn 2.8.3 does not compile against CUDA 13 regardless: its vector
#      types hit the double4/double4_16a deprecations.
#
# It is a speed optimisation for the 8 full-attention layers. It changes no
# result, so "fixing" the missing dependency buys nothing and breaks the install.

# The policy node -- `python -m paddock.policy.server`, which paddock talks to
# over localhost -- runs inside this venv, not paddock's own. Qwen-Drive's
# requirements have no reason to know about paddock, so without this the
# checkpoint loads fine and the node dies at import with
# ModuleNotFoundError: uvicorn.
"${PY}" -m pip install fastapi uvicorn

echo
echo "Weights are not downloaded here. Place the Qwen-Drive-1.0-4B snapshot"
echo "under [roots.weights] and point QWENDRIVE_WEIGHTS at the directory that"
echo "holds config.json, model.safetensors and planner-sft/."
echo "Set HF_HOME to a data disk first -- the default stages into the root filesystem."
echo
echo "done. Verify with: paddock doctor --model qwendrive"
