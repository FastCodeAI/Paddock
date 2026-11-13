#!/usr/bin/env bash
# VAD runs in the env UniAD's setup.sh builds (one OpenMMLab 1.x stack serves both); VAD's repo needs no build.
#
#   paddock setup --model uniad     # builds it; then point [models.vad] venv at the same env if it is not
#                                   # at the default path
set -euo pipefail
: "${PADDOCK_VENV:?run via 'paddock setup --model vad', which sets PADDOCK_VENV}"
if [ ! -x "${PADDOCK_VENV}/bin/python" ]; then
  echo "no env at ${PADDOCK_VENV}: run 'paddock setup --model uniad' first (it builds the env vad shares)" >&2
  exit 1
fi
"${PADDOCK_VENV}/bin/python" -c "import mmcv, mmdet, mmdet3d; print('vad env ok:', mmcv.__version__, mmdet3d.__version__)"
