#!/bin/bash
# Local: after the CARLA baselines, replay the VP baseline (once re-captured) and dj/op/vp on the Cosmos
# frames pulled from the pod. One node at a time -- a second node on the same port scores the first one's model.
#   OPENLOOP_DIR  local working dir: captures/, plans/, converted/            (required)
#   PADDOCK_DIR   the paddock checkout to run from                            (default: $PWD)
#   PADDOCK_CONFIG  site config for these runs                                (default: $OPENLOOP_DIR/paddock-openloop.toml)
#   POD_HOST      user@host of the conversion pod                             (required)
#   POD_ROOT_REMOTE  the pod's scratch root, as seen on the pod                  (required)
#   POD_SSH       ssh command with port and key, e.g. "ssh -p 2222 -i ~/.ssh/pod"  (default: "ssh")
#   PORT          node port for the replays                                   (default: 9803)
#   WAIT_S        seconds to wait for each marker or conversion               (default: 21600)
# Usage: OPENLOOP_DIR=... POD_HOST=root@pod POD_SSH="ssh -p 2222 -i ~/.ssh/pod" local_replay_queue.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${OPENLOOP_DIR:?set OPENLOOP_DIR to the local open-loop working directory}"
: "${POD_HOST:?set POD_HOST to user@host of the conversion pod}"
O=$OPENLOOP_DIR; cd "${PADDOCK_DIR:-$PWD}"
SSH=${POD_SSH:-ssh}; H=$POD_HOST; PORT=${PORT:-9803}
: "${POD_ROOT_REMOTE:?set POD_ROOT_REMOTE to the scratch root ON THE POD (used only over ssh)}"
export PADDOCK_CONFIG=${PADDOCK_CONFIG:-$O/paddock-openloop.toml}
declare -A NAME=( [dj]=drivejepa [op]=openpilot [vp]=visionpilot ); declare -A END=( [dj]=366 [op]=371 [vp]=320 )
declare -A CAMS=( [dj]="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK" [op]="CAM_OP_ROAD CAM_OP_WIDE" [vp]="CAM_FRONT" )
# A wait gives up after WAIT_S seconds (default 6 h) and says what it was waiting for: the markers are
# written by hand (README, "Markers"), and a forgotten one must not sleep forever.
wait_for() {  # <what> <test command...>
  local what=$1 waited=0; shift
  until "$@"; do
    [ "$waited" -lt "${WAIT_S:-21600}" ] || { echo "gave up after ${waited} s waiting for $what" >&2; exit 1; }
    sleep 30; waited=$((waited + 30))
  done
}
replay() { local m=$1 tag=$2; shift 2; echo "== $m/$tag start $(date -u +%T)"
  .venv/bin/python "$HERE/openloop_replay.py" --model ${NAME[$m]} --capture "$O/captures/$m" --out "$O/plans/${m}_$tag.jsonl" --start 1 --end ${END[$m]} --port $PORT "$@" > "$O/plans/${m}_$tag.log" 2>&1
  echo "== $m/$tag exit $? $(date -u +%T): $(tail -n 1 "$O/plans/${m}_$tag.log" | cut -c1-100)"; }
converted() { for c in ${CAMS[$1]}; do $SSH "$H" "test -f $POD_ROOT_REMOTE/converted/$1/$c/conversion.json" || return 1; done; }
pull() { for c in ${CAMS[$1]}; do mkdir -p "$O/converted/$1/$c"; rsync -a -e "$SSH" "$H:$POD_ROOT_REMOTE/converted/$1/$c/" "$O/converted/$1/$c/" || return 1; done; }
wait_for "LOCAL_BASELINES_DONE in plans/local_baselines.log" grep -qs LOCAL_BASELINES_DONE "$O/plans/local_baselines.log"
[ -s "$O/plans/vp_carla.jsonl" ] || { wait_for "UPLOAD_DONE vp in run_capture_vp.log" grep -qs "UPLOAD_DONE vp" "$O/run_capture_vp.log"; replay vp carla; }
for m in op dj vp; do
  wait_for "$m converted on the pod" converted $m; echo "== $m converted on pod $(date -u +%T)"; pull $m && echo "== $m pulled $(date -u +%T)"
  replay $m cosmos $(for c in ${CAMS[$m]}; do printf -- "--frames %s=%s/converted/%s/%s " "$c" "$O" "$m" "$c"; done)
done; echo LOCAL_REPLAYS_DONE $(date -u +%T)
