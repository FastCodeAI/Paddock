#!/usr/bin/env bash
# Re-run VisionPilot on scene 0103 with one LongitudinalFusion constant changed.
#
# The shipped process_noise_dist_m = 2.0 starves the particle filter's velocity
# state, and its own source comment says so ("Keep small (MRPT uses 0.03 m).
# Large values cause particle drift -> bad velocity"). This measures what that
# one constant is worth on the benchmark. Everything else is identical to the
# runs1_fixed group, including the seed, so the pairing holds.
set -euo pipefail

BASE=/mnt/p4/naren/neuroncap
NCAP=/shakaal_hdd_1/workspace/naren/renesas/neuro-ncap
NOISE="${1:-0.3}"
OUT="${2:-runs1_noise${NOISE}}"
PORT="${3:-9011}"

PYTHONPATH="$BASE/servers" "$BASE/envs/ncap/bin/python" "$BASE/servers/visionpilot_server.py" \
  --port "$PORT" --cpu --process-noise-dist "$NOISE" > "$BASE/vp_variant_node.log" 2>&1 &
NODE=$!
trap 'kill $NODE 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf -m 5 "http://localhost:$PORT/alive" >/dev/null && break
  sleep 5
done
curl -sf -m 5 "http://localhost:$PORT/alive" >/dev/null || { echo "node never came up"; exit 1; }

for scenario in stationary frontal side; do
  echo "=== $scenario (process_noise_dist_m=$NOISE) ==="
  PYTHONPATH="$NCAP" "$BASE/envs/ncap/bin/python" -u "$NCAP/main.py" \
    --engine.renderer.port 8000 \
    --engine.model.port "$PORT" \
    --engine.dataset.data_root /shakaal_hdd_1/workspace/NeurIPS/Dataset/raw_data/nuscenes \
    --engine.dataset.version v1.0-trainval \
    --engine.dataset.sequence 103 \
    --engine.scenario.path "$BASE/scenarios_10hz/$scenario/0103.yaml" \
    --engine.waypoint-dt 0.5 \
    --engine.cameras CAM_FRONT \
    --engine.logger.log-dir "$BASE/outputs/$OUT/visionpilot/$scenario-0103" \
    --runs 1 2>&1 | grep -E "^ncap_score" || echo "  (no score line)"
done
