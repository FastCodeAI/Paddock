#!/bin/bash
# Pod: build the paddock, alpamayo15 and qwendrive envs once the repos are synced (marker $POD_ROOT/repos/.synced).
#   POD_ROOT   pod scratch root (required): holds repos/, envs/, logs/, runs/ and the paddock checkout
#   UV         the uv binary (default: uv on PATH -- needs >= 0.11; a distro uv is usually too old)
#   QWEN_FREEZE  the qwendrive env's package list (default: qwen_freeze.txt beside this script)
#   WAIT_S     seconds to wait for repos/.synced (default: 21600)
# Usage: POD_ROOT=/pod/scratch build_model_envs.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${POD_ROOT:?set POD_ROOT to the pod scratch root}"
UV=${UV:-uv}; QWEN_FREEZE=${QWEN_FREEZE:-$HERE/qwen_freeze.txt}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$POD_ROOT/uv-cache}
# A wait gives up after WAIT_S seconds (default 6 h) and says what it was waiting for: the markers are
# written by hand (README, "Markers"), and a forgotten one must not sleep forever.
wait_for() {  # <what> <test command...>
  local what=$1 waited=0; shift
  until "$@"; do
    [ "$waited" -lt "${WAIT_S:-21600}" ] || { echo "gave up after ${waited} s waiting for $what" >&2; exit 1; }
    sleep 30; waited=$((waited + 30))
  done
}
wait_for "repos/.synced" test -f "$POD_ROOT/repos/.synced"; echo "repos synced $(date -u +%T)"
mkdir -p "$POD_ROOT"/{runs,kitscenes/data,navsim/dataset,navsim/exp,envs,logs}
cd "$POD_ROOT/paddock" && $UV venv .venv --python 3.12 -q && $UV pip install -q --python .venv/bin/python -e . && echo PADDOCK_VENV_OK || echo PADDOCK_VENV_FAIL
( cd "$POD_ROOT/repos/alpamayo1.5" && export UV_PROJECT_ENVIRONMENT="$POD_ROOT/repos/alpamayo1.5/a1_5_venv" && $UV venv a1_5_venv --python 3.12 -q && $UV sync --no-install-package flash-attn -q && $UV pip install -q --python a1_5_venv/bin/python uvicorn fastapi pillow requests httpx && echo ALPAMAYO_ENV_OK || echo ALPAMAYO_ENV_FAIL ) > "$POD_ROOT/logs/env_alpamayo.log" 2>&1 &
( $UV venv "$POD_ROOT/envs/qwen-drive" --python 3.12 -q && $UV pip install --python "$POD_ROOT/envs/qwen-drive/bin/python" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r "$QWEN_FREEZE" && $UV pip install -q --python "$POD_ROOT/envs/qwen-drive/bin/python" uvicorn fastapi pillow requests httpx && echo QWEN_ENV_OK || echo QWEN_ENV_FAIL ) > "$POD_ROOT/logs/env_qwen.log" 2>&1 &
wait; echo "ENV_BUILDS_DONE $(date -u +%T): alpamayo=$(tail -n1 "$POD_ROOT/logs/env_alpamayo.log") qwen=$(tail -n1 "$POD_ROOT/logs/env_qwen.log")"
