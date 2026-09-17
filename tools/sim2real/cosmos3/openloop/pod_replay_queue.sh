#!/bin/bash
# Pod: replay alpamayo15 / qwendrive on their Cosmos frames once converted; the qwendrive baseline as soon
# as its capture is up. The models too big for the local GPU run their replays here.
#   POD_ROOT     pod scratch root (required): captures/, converted/, plans/, logs/, paddock/
#   SCRIPTS      directory holding openloop_replay.py (default: this script's directory)
#   PADDOCK_CONFIG  the pod's site config (default: $SCRIPTS/paddock.toml)
# Usage: POD_ROOT=/pod/scratch pod_replay_queue.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${POD_ROOT:?set POD_ROOT to the pod scratch root}"
SCRIPTS=${SCRIPTS:-$HERE}
cd "$POD_ROOT/paddock"; export PADDOCK_CONFIG=${PADDOCK_CONFIG:-$SCRIPTS/paddock.toml} HF_HOME=${HF_HOME:-$POD_ROOT/hf}
mkdir -p "$POD_ROOT/plans" "$POD_ROOT/logs"
declare -A NAME=( [al]=alpamayo15 [qw]=qwendrive ); declare -A END=( [al]=370 [qw]=390 )
declare -A CAMS=( [al]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT" [qw]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT" )
replay() { # model tag gpu port [frames args...]
  local m=$1 tag=$2 g=$3 port=$4; shift 4
  echo "== $m/$tag start $(date -u +%T) gpu $g"
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python "$SCRIPTS/openloop_replay.py" --model ${NAME[$m]} --capture "$POD_ROOT/captures/$m" --out "$POD_ROOT/plans/${m}_$tag.jsonl" --start 1 --end ${END[$m]} --port $port "$@" > "$POD_ROOT/logs/replay_${m}_$tag.log" 2>&1
  echo "== $m/$tag exit $? $(date -u +%T): $(tail -n 1 "$POD_ROOT/logs/replay_${m}_$tag.log" | cut -c1-100)"
}
converted() { for c in ${CAMS[$1]}; do [ -f "$POD_ROOT/converted/$1/$c/conversion.json" ] || return 1; done; }
frames_args() { for c in ${CAMS[$1]}; do printf -- "--frames %s=%s/converted/%s/%s " "$c" "$POD_ROOT" "$1" "$c"; done; }
# qw baseline as soon as its capture is uploaded (GPU 0, alongside a conversion)
( until [ -f "$POD_ROOT/captures/qw/.upload_done" ]; do sleep 30; done; replay qw carla 0 9811 ) &
# al cosmos replay once its three cameras are converted (GPU 1 with the al baseline / conversion)
( until converted al; do sleep 30; done; replay al cosmos 1 9801 $(frames_args al) ) &
# qw cosmos replay once converted
( until converted qw; do sleep 30; done; replay qw cosmos 0 9812 $(frames_args qw) ) &
wait; echo POD_REPLAYS_DONE $(date -u +%T)
