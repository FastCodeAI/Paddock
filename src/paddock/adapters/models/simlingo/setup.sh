#!/usr/bin/env bash
# Build SimLingo's environment and link its two pinned checkpoints.
#
#   HF_HOME=<the site's Hugging Face cache> paddock setup --model simlingo
#
# SimLingo ships a conda environment.yaml (Python 3.8, torch 2.2.0), not a uv
# lock, so this builds a uv environment from the versions that file pins, with
# two changes:
#   - torch 2.7.1 / CUDA 12.8 instead of 2.2.0 / CUDA 12.1. 2.2.0 has no kernels
#     for Blackwell GPUs (sm_120); 2.7.1 is the first line that does, and is what
#     this site's HUGSIM environment runs.
#   - Python 3.10 instead of 3.8, so paddock's own node code imports; and
#     opencv-python-headless 4.8 instead of opencv-python 4.2, which has no
#     Python 3.10 build. OpenCV is used only for the JPEG round trip SimLingo's
#     agent puts every frame through.
# Training-only packages (deepspeed, wandb, the dataset tooling) are left out.
#
# Weights, both into the Hugging Face cache at HF_HOME and linked from
# {weights}, so a later download of another revision cannot change what a run
# is scored against:
#   RenzKa/simlingo at 26c7c89: the released driving checkpoint (epoch 13) and
#     the training config beside it; the DeepSpeed optimiser shards are skipped.
#   OpenGVLab/InternVL2-1B at 0d75ccd: the base vision-language model SimLingo
#     was fine-tuned from, which its code builds before loading the checkpoint.

set -euo pipefail

: "${PADDOCK_REPO:?run via 'paddock setup --model simlingo', which sets PADDOCK_REPO}"
: "${PADDOCK_WEIGHTS:?set PADDOCK_WEIGHTS to [roots.weights]}"
: "${PADDOCK_VENV:=${PADDOCK_REPO}/.venv}"
# Never stage into ~/.cache on the root filesystem.
export HF_HOME="${HF_HOME:-${PADDOCK_WEIGHTS}/hf_cache}"
PY="${PADDOCK_VENV}/bin/python"

echo "building environment for simlingo in ${PADDOCK_VENV}"
uv venv --allow-existing --python 3.10 "${PADDOCK_VENV}"
uv pip install --python "${PY}" \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch==2.7.1 torchvision==0.22.1
uv pip install --python "${PY}" \
  transformers==4.46.3 tokenizers==0.20.3 peft==0.13.2 accelerate==1.0.1 safetensors==0.5.2 \
  huggingface-hub==0.27.0 pytorch-lightning==2.4.0 hydra-core==1.3.2 omegaconf==2.3.0 \
  timm==0.9.16 einops==0.7.0 numpy==1.23.0 pillow==10.2.0 opencv-python-headless==4.8.1.78 \
  fastapi uvicorn

fetch() {  # fetch <repo> <revision> <pattern>... ; prints the snapshot path
  HF_HUB_DISABLE_PROGRESS_BARS=1 "${PY}" - "$@" <<'PYEOF'
import sys

from huggingface_hub import snapshot_download

repo, revision, *patterns = sys.argv[1:]
try:
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns, local_files_only=True)
except Exception:  # noqa: BLE001 - not cached yet
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns)
print(path)
PYEOF
}

link() {  # link <snapshot> <name>: {weights}/<name> -> the snapshot, relative
  local target="${PADDOCK_WEIGHTS}/$2"
  if [ -e "${target}" ] && [ ! -L "${target}" ]; then
    echo "${target} exists and is not a link; move it aside and run again" >&2
    exit 1
  fi
  ln -sfn "$(realpath -m --relative-to="${PADDOCK_WEIGHTS}" "$1")" "${target}"
  echo "linked ${target} -> $(readlink "${target}")"
}

echo "fetching RenzKa/simlingo and OpenGVLab/InternVL2-1B into ${HF_HOME}"
SIMLINGO=$(fetch RenzKa/simlingo 26c7c89e797d4e25bbf640013317af8da26a5454 \
  "simlingo/.hydra/*" "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt")
INTERNVL=$(fetch OpenGVLab/InternVL2-1B 0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187 \
  "*.json" "*.py" "*.safetensors" "*.txt")

# Every file the model reads, as fetched at the pins on 2026-11-18.
(cd "${SIMLINGO}" && sha256sum -c --quiet - <<'SUMS'
ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28  simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt
d56a7c1ebf3b6fd7ff6edff87071f1b2fe7269563e3ea626379994bef7807417  simlingo/.hydra/config.yaml
SUMS
)
(cd "${INTERNVL}" && sha256sum -c --quiet - <<'SUMS'
baea1c1e8f24ce785f49112072664b8be778e9242a1a21cf6f7c6cb92add6d10  added_tokens.json
2e85a81acfc8cbea3dc3cee831e6c4b171d456f30938aef14cf37d5eb2ac00ed  config.json
22dd687a5f142985407675fc6b1dc688c2ded93e54192543ecf28ef495a6efa8  configuration_intern_vit.py
6524e6deae6c318543d1022a599f011120f5a970eaa25f1118e7f5d0e346710c  configuration_internvl_chat.py
b1b9f280e9cb9211e519e2b8071002b57cc742aadb2f49b05fa674ee3c7d83ee  conversation.py
2decd6209b7280fbf8ddba60a24047ce26cbd6ac539bca8ec145d21aeaaf4498  generation_config.json
455e0caaa06abffc663e9282dfe71dde07fd1991eaf24146bf08793c4dba4497  merges.txt
1978e330ea5533dda4d1379096aded271fc741347fc48cb4924846b56a2a6a4b  modeling_intern_vit.py
5d8f26ccfeb8434c0bcc295050725add4b2f2b0ffb4d1ab550c877a648386597  modeling_internvl_chat.py
9420916a7fab7d2009f7907cdffa341c9cb6be7c5e0cf4ee193de16fde647dea  model.safetensors
0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4  preprocessor_config.json
af3c7a7e8e3396571d8640b822cd611de2b51b15cf65a5dc1054ffea31483427  special_tokens_map.json
2ed7120320ff66f65e1dccc704f318eeccab8941292424dfdf8290dd1b689634  tokenizer_config.json
87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d  vocab.json
SUMS
)
echo "sha256 ok: both checkpoints are the pinned ones"

link "${SIMLINGO}" SimLingo
link "${INTERNVL}" InternVL2-1B

"${PY}" -c "import torch, transformers, peft, pytorch_lightning, hydra, cv2, fastapi, uvicorn; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "done. Verify with: paddock doctor --model simlingo, then paddock verify --model simlingo"
