#!/bin/bash
# Pod-side: convert every camera of every capture. Waits for each capture's upload (ticks.jsonl complete),
# renders wsm (fast, one GPU), then converts cameras two at a time (one per GPU).
#
# Every location is an environment variable: this runs on a rented box whose layout is its own, and a
# path baked in here would work for exactly one machine.
#   POD_ROOT        scratch root holding captures/, converted/, logs/      (required)
#   COSMOS_PY       python of the cosmos-framework venv                    (required)
#   FLASHDREAMS_PY  python of the flashdreams venv (the Ludus renderer)    (required)
#   SPEC            transfer sample spec                                   (default: ../specs/cosmos_live_spec.json)
#   HF_HOME         Hugging Face cache                                     (default: $POD_ROOT/hf)
#   HF_TOKEN_FILE   file holding the HF token -- never a token in a script (default: ~/.cache/huggingface/token)
#   STEPS CHUNK NCOND VIS ORDER WSM_GPU                                    (defaults below = the 17-Sep run)
# Usage: POD_ROOT=/pod/scratch COSMOS_PY=... FLASHDREAMS_PY=... run_convert.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); ROOT_DIR=$(dirname "$HERE")
: "${POD_ROOT:?set POD_ROOT to the pod scratch root holding captures/}"
: "${COSMOS_PY:?set COSMOS_PY to the cosmos-framework venv python}"
: "${FLASHDREAMS_PY:?set FLASHDREAMS_PY to the flashdreams venv python}"
STEPS=${STEPS:-12}; CHUNK=${CHUNK:-61}; NCOND=${NCOND:-8}; SPEC=${SPEC:-$ROOT_DIR/specs/cosmos_live_spec.json}
# segment only: from the start of each recording to 15 s after the cut-in trigger (T0: vp 50, dj 66, al 70, op 71, qw 90);
# vp's recording ends 13.5 s after it, at 320
declare -A EXPECT=( [vp]=320 [dj]=366 [op]=371 [al]=370 [qw]=390 )
declare -A RANGE=( [vp]="1 320" [dj]="1 366" [op]="1 371" [al]="1 370" [qw]="1 390" )
VIS=${VIS:-50}
declare -A CAMS=( [vp]="CAM_FRONT" [dj]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK" [op]="CAM_OP_ROAD CAM_OP_WIDE" [al]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT" [qw]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT" )
ORDER=${ORDER:-"vp dj op al qw"}
# The spec's prompt and control paths are relative, so run from the directory that holds prompts/ and controls/.
cd "$ROOT_DIR"
mkdir -p "$POD_ROOT/logs" "$POD_ROOT/converted"
# ninja must be on PATH for the Ludus JIT; the flashdreams venv's bin holds it.
export PATH=$(dirname "$FLASHDREAMS_PY"):$PATH HF_HOME=${HF_HOME:-$POD_ROOT/hf}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}
[ -r "$HF_TOKEN_FILE" ] && export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
convert() {  # model cam gpu
  local m=$1 c=$2 g=$3; local out=$POD_ROOT/converted/$m/$c
  [ -f "$out/conversion.json" ] && { echo "== $m/$c already converted"; return; }
  echo "== $m/$c start $(date -u +%T) on GPU $g"
  COSMOS_VIS_M=$VIS COSMOS_STEPS=$STEPS COSMOS_CHUNK=$CHUNK COSMOS_NCOND=$NCOND CUDA_VISIBLE_DEVICES=$g \
    "$COSMOS_PY" "$ROOT_DIR/batch/cosmos_batch.py" "$POD_ROOT/captures/$m" "$c" "$out" "$SPEC" ${RANGE[$m]} > "$POD_ROOT/logs/convert-$m-$c.log" 2>&1
  echo "== $m/$c exit $? $(date -u +%T): $(grep COSMOS_BATCH_DONE "$POD_ROOT/logs/convert-$m-$c.log" | tail -n 1)"
}
for m in $ORDER; do
  until [ -s "$POD_ROOT/captures/$m/ticks.jsonl" ] && [ "$(wc -l < "$POD_ROOT/captures/$m/ticks.jsonl")" -ge "${EXPECT[$m]}" ] && [ -f "$POD_ROOT/captures/$m/.upload_done" ]; do sleep 30; done
  echo "== $m capture ready $(date -u +%T)"
  CUDA_VISIBLE_DEVICES=${WSM_GPU:-1} "$FLASHDREAMS_PY" "$ROOT_DIR/batch/wsm_batch.py" "$POD_ROOT/captures/$m" > "$POD_ROOT/logs/wsm-$m.log" 2>&1; echo "== $m wsm: $(grep WSM_BATCH_DONE "$POD_ROOT/logs/wsm-$m.log")"
  cams=(${CAMS[$m]}); i=0
  while [ $i -lt ${#cams[@]} ]; do
    convert "$m" "${cams[$i]}" 0 &
    p0=$!
    if [ $((i+1)) -lt ${#cams[@]} ]; then convert "$m" "${cams[$((i+1))]}" 1 & p1=$!; wait $p1; fi
    wait $p0; i=$((i+2))
  done
done
echo CONVERT_ALL_DONE $(date -u +%T)
