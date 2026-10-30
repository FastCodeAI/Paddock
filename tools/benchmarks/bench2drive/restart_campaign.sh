#!/bin/bash
# Start, restart or rebalance a Bench2Drive campaign in one idempotent call -- after a reboot, after a
# shutdown, or when some workers have finished and the rest of the routes should be spread out again.
#
#   tools/benchmarks/bench2drive/restart_campaign.sh --model openpilot [--workers 5] [--held 11715]
#       [--rebalance] [--status]
#       [--dashboard-root DIR --expert DIR [--features CSV] [--port 8765]]
#
# It (re)starts, each detached in its own session so it outlives this shell, only what is not running:
#   guard      `campaign guard`: kills an evaluator whose log went silent or whose CARLA died, so its
#              worker records the route FAILED and moves on
#   workers    `campaign launch`: every route without a scored attempt, sorted and dealt to the free
#              worker slots (own CARLA / TM / node ports each); --held routes last and alone;
#              --rebalance stops the alive workers first so everything left is dealt anew
#   dashboard  only with --dashboard-root: tools/benchmarks/bench2drive/dashboard.py on 127.0.0.1:<port>
# then prints `campaign status`. See docs/benchmarks/bench2drive.md, "Running a sweep in parallel".
#
# Environment: PADDOCK_CONFIG picks the site config the workers copy (default ./paddock.toml);
# PADDOCK_PYTHON is paddock's own interpreter (default: python3 on PATH); TMPDIR is passed through.
set -u
cd "$(dirname "$0")/../../.." || exit 1   # the Paddock checkout
PY=${PADDOCK_PYTHON:-python3}
MODEL="" WORKERS=5 HELD="" REBALANCE="" STATUS_ONLY="" ROOT="" EXPERT="" FEATURES="" PORT=8765
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --workers) WORKERS=$2; shift 2 ;;
    --held) HELD=$2; shift 2 ;;
    --rebalance) REBALANCE=--rebalance; shift ;;
    --status) STATUS_ONLY=1; shift ;;
    --dashboard-root) ROOT=$2; shift 2 ;;
    --expert) EXPERT=$2; shift 2 ;;
    --features) FEATURES=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }
C="$PY -m paddock.adapters.benchmarks.bench2drive.campaign"
LOGS=$($PY -c "from paddock.config import SiteConfig; print(SiteConfig.load().runs / 'bench2drive' / '_campaign' / '$MODEL' / 'logs')")
mkdir -p "$LOGS"

if [ -n "$STATUS_ONLY" ]; then $C status --model "$MODEL"; exit 0; fi
if ! nvidia-smi > /dev/null 2>&1; then
  echo "nvidia-smi fails: the GPU driver is not up yet. Wait a minute after boot and run this again." >&2
  exit 1
fi

if ! pgrep -f "[c]ampaign guard --model $MODEL" > /dev/null; then
  setsid nohup $C guard --model "$MODEL" < /dev/null >> "$LOGS/guard.log" 2>&1 &
  echo "guard started ($LOGS/guard.log)"
fi

$C launch --model "$MODEL" --workers "$WORKERS" ${HELD:+--held "$HELD"} $REBALANCE

if [ -n "$ROOT" ] && ! ss -ltn | grep -q ":$PORT "; then
  RUNS=$($PY -c "from paddock.config import SiteConfig; print(SiteConfig.load().runs / 'bench2drive' / '$MODEL')")
  XML=$($PY -c "
from paddock.config import SiteConfig
s = SiteConfig.load().benchmarks['bench2drive']
print(f\"{s.get('repo')}/leaderboard/data/{s.get('routes_file') or 'bench2drive220.xml'}\")")
  setsid nohup $PY -m tools.benchmarks.bench2drive.dashboard serve --root "$ROOT" --runs "$RUNS" \
    --routes-xml "$XML" ${EXPERT:+--expert "$EXPERT"} ${FEATURES:+--features "$FEATURES"} --port "$PORT" \
    < /dev/null >> "$LOGS/dashboard.log" 2>&1 &
  echo "dashboard started: http://127.0.0.1:$PORT/dashboard/ ($LOGS/dashboard.log)"
fi

sleep 3
$C status --model "$MODEL"
