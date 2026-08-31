# Paddock leaderboard

32 runs across 3 benchmarks.
4 run(s) excluded as invalid — shown at the end, never averaged in.

## kitscenes

| model | run | n | site | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | min_ade@6 ↓ | medoid_ade ↓ | medoid_fde@3s ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 | streamtest2-nonav | 8 | fastcode-shakaal | 0.574 | 0.270 | 0.664 | 1.067 | 0.280 | 0.372 | 0.581 |
| alpamayo15 | streamtest-nonav | 5 | fastcode-shakaal | 0.619 | 0.317 | 0.740 | 1.067 | 0.329 | 0.405 | 0.587 |
| alpamayo15 | phase3-nav | 7 | fastcode-shakaal | 0.819 | 0.411 | 0.932 | 1.533 | 0.561 | 0.937 | 1.817 |
| alpamayo15 | phase3-nonav | 7 | fastcode-shakaal | 0.845 | 0.414 | 0.960 | 1.590 | 0.708 | 1.156 | 2.282 |
| alpamayo15 | full-nonav | 223 | fastcode-shakaal | 0.921 | 0.332 | 0.990 | 2.003 | 0.552 | 0.892 | 1.902 |
| alpamayo15 | full-nav | 223 | fastcode-shakaal | 1.002 | 0.362 | 1.088 | 2.158 | 0.543 | 0.875 | 1.862 |
| alpamayo15 | full-letterbox | 223 | fastcode-shakaal | 1.176 | 0.418 | 1.284 | 2.535 | 0.619 | 1.009 | 2.085 |
| alpamayo15 | phase3-letterbox | 7 | fastcode-shakaal | 1.188 | 0.506 | 1.357 | 2.338 | 0.750 | 1.182 | 2.288 |
| openpilot | local20 | 20 | fastcode-shakaal | 1.291 | 0.726 | 1.460 | 2.274 | - | - | - |
| openpilot | ablate_0 | 15 | fastcode-shakaal | 1.412 | 0.805 | 1.582 | 2.462 | - | - | - |
| openpilot | full223 | 223 | fastcode-shakaal | 1.599 | 0.836 | 1.808 | 2.894 | - | - | - |
| openpilot | smoke | 1 | fastcode-shakaal | 1.697 | 0.752 | 1.895 | 3.298 | - | - | - |
| openpilot | rate_r2 | 3 | fastcode-shakaal | 1.905 | 0.998 | 2.162 | 3.425 | - | - | - |
| openpilot | rate_r3 | 3 | fastcode-shakaal | 10.462 | 5.998 | 12.072 | 17.701 | - | - | - |
| openpilot | rate_r1 | 3 | fastcode-shakaal | 12.583 | 7.392 | 14.394 | 21.207 | - | - | - |
| openpilot | ablate_1 | 15 | fastcode-shakaal | 14.302 | 8.240 | 16.372 | 24.358 | - | - | - |

## navsim

| model | run | n | site | epdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | driving_direction_compliance ↑ | traffic_light_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | 2026.08.08.02.26.44 | 242 | fastcode-shakaal | 0.907 | 0.994 | 0.992 | 0.998 | 1.000 | 0.923 | 0.979 |

## neuroncap

| model | run | n | site | ncap_score ↑ | impact_speed ↓ | reference_speed ↑ | collision_rate ↓ |
|---|---|---:|---|---:|---:|---:|---:|
| drivejepa | runs5 | 15 | fastcode-shakaal | 4.042 | 2.073 | 13.279 | 0.333 |
| drivejepa | runs1 | 3 | fastcode-shakaal | 4.024 | 2.133 | 13.279 | 0.333 |
| openpilot | runs1 | 3 | fastcode-shakaal | 3.954 | 2.366 | 13.279 | 0.333 |
| openpilot | runs5 | 15 | fastcode-shakaal | 3.798 | 2.262 | 12.590 | 0.467 |
| openpilot | scene0108_runs5 | 10 | fastcode-shakaal | 3.500 | 4.036 | 13.279 | 0.300 |
| visionpilot | runs10 | 5 | fastcode-shakaal | 2.792 | 2.510 | 8.314 | 1.000 |
| alpamayo | runs5 | 15 | fastcode-shakaal | 2.458 | 4.731 | 11.226 | 0.667 |
| alpamayo | runs1 | 3 | fastcode-shakaal | 2.059 | 6.393 | 11.557 | 0.667 |
| visionpilot | runs1 | 3 | fastcode-shakaal | 1.968 | 5.133 | 9.902 | 1.000 |
| drivejepa | scene0108_runs5 | 10 | fastcode-shakaal | 1.810 | 7.321 | 11.250 | 0.900 |
| visionpilot | fixcheck | 1 | fastcode-shakaal | 1.684 | 7.687 | 13.279 | 1.000 |
| visionpilot | runs5 | 15 | fastcode-shakaal | 1.671 | 6.099 | 10.246 | 1.000 |
| visionpilot | runs1_fixed | 3 | fastcode-shakaal | 1.475 | 6.411 | 9.902 | 1.000 |
| visionpilot | runs1_noise0.3 | 3 | fastcode-shakaal | 1.371 | 13.452 | 13.279 | 1.000 |
| visionpilot | scene0108_runs5 | 10 | fastcode-shakaal | 1.352 | 8.055 | 10.743 | 1.000 |

## Excluded runs

- **neuroncap/alpamayo/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/drivejepa/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/openpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/visionpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.

---

Arrows show direction of better. Aggregates are recomputed by paddock from per-sample records, never taken from the producer.
