#!/usr/bin/env bash
# Build the Drive-JEPA perception-free environment.
#
#   bash src/paddock/adapters/models/drivejepa_free/setup.sh
#
# A plain venv, where the perception-based sibling needs conda. The reason is
# the whole point of this entry: the perception-free agent has no BEVFormer, so
# it needs neither mmcv-full nor mmdet, and those two -- pinned against
# torch 2.1/cu121 with a compiled deformable-attention kernel -- are what made
# a conda toolchain necessary over there. Everything here installs from wheels.
#
# If your site keeps environments elsewhere, point paddock at it rather than
# moving anything:
#
#   [models.drivejepa_free]
#   venv = "/wherever/you/keep/envs/drive-jepa"

set -euo pipefail
: "${PADDOCK_REPO:?run via 'paddock setup --model drivejepa_free', which sets PADDOCK_REPO}"

VENV="${PADDOCK_VENV:-${PADDOCK_REPO}/.venv-perception-free}"
python3.9 -m venv "${VENV}"
PY="${VENV}/bin/python"

"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# The repo's requirements minus mmcv_full/mmdet, which only the perception-based
# agent imports.
grep -viE '^(mmcv[-_]full|mmdet)' "${PADDOCK_REPO}/requirements.txt" > /tmp/drivejepa-free-requirements.txt
"${PY}" -m pip install -r /tmp/drivejepa-free-requirements.txt

# av 11 has no cp39 wheel on the default index and falls back to building
# against a system ffmpeg that may not exist -- `pkg-config` then fails on
# libavformat and takes the whole requirements install down with it. 12.3 ships
# prebuilt ffmpeg binaries, so no system ffmpeg is needed for decode or encode.
"${PY}" -m pip install "av==12.3.0"

"${PY}" -m pip install -e "${PADDOCK_REPO}/navsim_v1"

# The policy node -- `python -m paddock.policy.server` -- runs inside this venv,
# not paddock's. Without these the checkpoint loads fine and the node dies at
# import with ModuleNotFoundError: uvicorn. Pinned below the versions that
# dropped Python 3.9.
"${PY}" -m pip install "fastapi<0.116" "uvicorn<0.34"

echo
echo "Weights are not downloaded here. Both files are required and they are not"
echo "interchangeable -- the agent builds its encoder from the pretrain file and"
echo "only then overwrites it with the agent checkpoint:"
echo
echo "  hf download linhanwang/Drive-JEPA drive_jepa_perception_free_agent_vitl.ckpt"
echo "  hf download linhanwang/Drive-JEPA vitl_merge_3dataset_e50.pt"
echo
echo "One invocation per file: a second --include pattern is parsed as a"
echo "positional argument and silently ignored. Use 'hf download', not the"
echo "deprecated 'huggingface-cli download', which exits 0 having fetched"
echo "nothing. Place both under [roots.weights]/drive-jepa/."
echo
echo "done. Verify with: paddock doctor --model drivejepa_free"
