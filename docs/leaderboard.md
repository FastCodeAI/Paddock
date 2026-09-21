# Paddock leaderboard

91 runs across 8 benchmarks.
49 run(s) excluded as invalid — shown at the end, never averaged in.

## alpasim

| model | run | n | site | scene_score ↑ | progress ↑ | collision_rate ↓ | collision_at_fault ↓ | offroad ↓ | duration_frac_20s ↑ | dist_to_gt_trajectory ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| openpilot | openpilot_020_pilot10 | 10 | fastcode-shakaal | 0.957 | 0.901 | 0.100 | 0.000 | 0.000 | 0.777 | 1.454 |
| alpamayo15 | alpamayo_proxy_referee_20hz | 10 | fastcode-shakaal | 0.900 | 0.960 | 0.100 | 0.100 | 0.000 | 0.740 | 3.170 |
| constant_velocity | cv_023_20hz_pilot10 | 10 | fastcode-shakaal | 0.885 | 0.829 | 0.200 | 0.000 | 0.000 | 0.653 | 2.507 |
| visionpilot | visionpilot_v2_smoke_pilot10_handover_blend | 10 | fastcode-shakaal | 0.849 | 0.820 | 0.300 | 0.100 | 0.000 | 0.699 | 1.467 |
| drivejepa | drivejepa_014_20hz | 10 | fastcode-shakaal | 0.503 | 0.565 | 0.400 | 0.200 | 0.100 | 0.462 | 2.443 |

## bench2drive

| model | run | n | site | driving_score ↑ | route_completion ↑ | infraction_penalty ↑ | collision_rate ↓ | n_runs ↑ | n_crashed ↑ | duration_s ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| constant_velocity | pooled50-3mps-40c29d24c37d542a-02b6ac | 50 | fastcode-navvrat-b2d-pooled | 48.308 | 74.651 | 0.615 | 0.540 | 1.000 | 0.000 | 194.900 |
| constant_velocity | 20260908T084921-804913 | 24 | fastcode-navvrat-cv-b2d | 47.943 | 74.682 | 0.635 | 0.583 | - | 0.000 | 451.080 |
| visionpilot | pooled50-40c29d24c37d542a-d94009 | 50 | fastcode-navvrat-b2d-pooled | 47.821 | 71.191 | 0.622 | 0.580 | 1.000 | 0.000 | 257.252 |
| constant_velocity | pooled50-9mps-40c29d24c37d542a-13c440 | 50 | fastcode-navvrat-b2d-pooled | 43.574 | 72.699 | 0.577 | 0.580 | 1.000 | 0.000 | 123.802 |
| alpamayo15 | pooled50-40c29d24c37d542a-36d2d0 | 50 | fastcode-navvrat-b2d-pooled | 43.043 | 72.317 | 0.655 | 0.540 | 1.000 | 0.000 | 3555.930 |
| constant_velocity | pooled50-12mps-40c29d24c37d542a-0c11b2 | 50 | fastcode-navvrat-b2d-pooled | 42.193 | 72.125 | 0.558 | 0.560 | 1.000 | 0.000 | 108.372 |
| constant_velocity | pooled50-15mps-40c29d24c37d542a-06f021 | 50 | fastcode-navvrat-b2d-pooled | 41.448 | 73.684 | 0.546 | 0.560 | 1.000 | 0.000 | 107.988 |
| constant_velocity | pooled50-6mps-40c29d24c37d542a-35a5cb | 50 | fastcode-navvrat-b2d-pooled | 40.388 | 73.075 | 0.539 | 0.620 | 1.000 | 0.000 | 157.368 |
| drivejepa | pooled50-40c29d24c37d542a-1de367 | 50 | fastcode-navvrat-b2d-pooled | 31.706 | 39.101 | 0.862 | 0.160 | 1.000 | 0.000 | 1068.316 |
| qwendrive | pooled50-40c29d24c37d542a-f250e0 | 50 | fastcode-navvrat-b2d-pooled | 12.921 | 50.191 | 0.300 | 0.900 | 1.000 | 0.000 | 3457.266 |

## comma2k19

| model | run | n | site | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | scored_horizon_s | ade@1s ↓ | ade@2s ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 | 20260913T072437-b058c2 | 10165 | hpcai-skodate | 0.900 | 0.499 | 0.973 | 1.654 | 4.000 | 0.386 | 0.616 |
| qwendrive | 20260916T220846-971379 | 10165 | hpcai-skodate | 1.159 | 0.632 | 1.296 | 2.094 | 5.000 | 0.475 | 0.801 |
| openpilot | 20260916T051616-2a91ca | 10165 | hpcai-skodate | 1.507 | 0.804 | 1.703 | 2.712 | 10.000 | 0.598 | 1.035 |
| constant_velocity | 20260916T162837-b89038 | 10165 | hpcai-skodate | 1.639 | 0.734 | 1.786 | 3.286 | 3.000 | 0.537 | 1.017 |
| visionpilot | 20260916T012719-03350c | 10165 | hpcai-skodate | 4.045 | 1.652 | 4.553 | 8.146 | 4.000 | 1.149 | 2.462 |
| drivejepa_free | 20260916T003407-c504e5 | 10165 | hpcai-skodate | 16.048 | 10.887 | 18.016 | 26.248 | 4.000 | 7.800 | 12.077 |

## kitscenes

| model | run | n | site | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | min_ade@6 ↓ | medoid_ade ↓ | medoid_fde@3s ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 | streamtest2-nonav | 8 | fastcode-shakaal | 0.574 | 0.270 | 0.664 | 1.067 | 0.280 | 0.372 | 0.581 |
| alpamayo15 | streamtest-nonav | 5 | fastcode-shakaal | 0.619 | 0.317 | 0.740 | 1.067 | 0.329 | 0.405 | 0.587 |
| qwendrive | 20260914T220841-5a91fd | 223 | fastcode-shakaal | 0.806 | 0.289 | 0.852 | 1.777 | - | - | - |
| alpamayo15 | phase3-nav | 7 | fastcode-shakaal | 0.819 | 0.411 | 0.932 | 1.533 | 0.561 | 0.937 | 1.817 |
| alpamayo15 | phase3-nonav | 7 | fastcode-shakaal | 0.845 | 0.414 | 0.960 | 1.590 | 0.708 | 1.156 | 2.282 |
| alpamayo15 | full-nonav | 223 | fastcode-shakaal | 0.921 | 0.332 | 0.990 | 2.003 | 0.552 | 0.892 | 1.902 |
| alpamayo15 | full-nav | 223 | fastcode-shakaal | 1.002 | 0.362 | 1.088 | 2.158 | 0.543 | 0.875 | 1.862 |
| alpamayo15 | full-letterbox | 223 | fastcode-shakaal | 1.176 | 0.418 | 1.284 | 2.535 | 0.619 | 1.009 | 2.085 |
| alpamayo15 | phase3-letterbox | 7 | fastcode-shakaal | 1.188 | 0.506 | 1.357 | 2.338 | 0.750 | 1.182 | 2.288 |
| openpilot | local20 | 20 | fastcode-shakaal | 1.291 | 0.726 | 1.460 | 2.274 | - | - | - |
| openpilot | ablate_0 | 15 | fastcode-shakaal | 1.412 | 0.805 | 1.582 | 2.462 | - | - | - |
| visionpilot | 20260909T202724-153b04 | 223 | fastcode-shakaal | 1.534 | 0.700 | 1.567 | 3.089 | - | - | - |
| openpilot | 20260901T025750-fd824c | 3 | fastcode-shakaal | 1.592 | 0.581 | 1.785 | 3.340 | - | - | - |
| openpilot | 20260901T021743-fd824c | 3 | fastcode-shakaal | 1.592 | 0.581 | 1.785 | 3.340 | - | - | - |
| openpilot | full223 | 223 | fastcode-shakaal | 1.599 | 0.836 | 1.808 | 2.894 | - | - | - |
| openpilot | 20260901T175755-cc27fb | 223 | fastcode-shakaal | 1.599 | 0.836 | 1.808 | 2.894 | - | - | - |
| openpilot | smoke | 1 | fastcode-shakaal | 1.697 | 0.752 | 1.895 | 3.298 | - | - | - |
| meteor | 20260914T215457-ab0586 | 223 | fastcode-shakaal | 1.879 | 0.695 | 2.040 | 4.034 | - | - | - |
| openpilot | rate_r2 | 3 | fastcode-shakaal | 1.905 | 0.998 | 2.162 | 3.425 | - | - | - |
| drivejepa | kitscenes_e2e_final | 223 | fastcode-shakaal | 2.364 | 1.306 | 2.668 | 4.199 | - | - | - |
| meteor | 20260914T185351-5e5223 | 223 | fastcode-shakaal | 4.784 | 2.334 | 5.542 | 9.011 | - | - | - |
| openpilot | rate_r3 | 3 | fastcode-shakaal | 10.462 | 5.998 | 12.072 | 17.701 | - | - | - |
| openpilot | rate_r1 | 3 | fastcode-shakaal | 12.583 | 7.392 | 14.394 | 21.207 | - | - | - |
| openpilot | ablate_1 | 15 | fastcode-shakaal | 14.302 | 8.240 | 16.372 | 24.358 | - | - | - |

## navsim

| model | run | n | site | epdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | driving_direction_compliance ↑ | traffic_light_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | 2026.08.08.02.26.44 | 242 | fastcode-shakaal | 0.907 | 0.994 | 0.992 | 0.998 | 1.000 | 0.923 | 0.979 |
| drivejepa | 20260902T173156-7e208f | 12146 | fastcode-shakaal | 0.843 | 0.984 | 0.985 | 0.991 | 0.998 | 0.889 | 0.979 |
| qwendrive | 20260914T221802-80457c | 12146 | fastcode-shakaal | 0.840 | 0.982 | 0.964 | 0.993 | 0.998 | 0.820 | 0.977 |
| openpilot | 20260911T161513-3a4c90 | 12146 | fastcode-shakaal | 0.747 | 0.970 | 0.913 | 0.954 | 0.998 | 0.712 | 0.965 |
| alpamayo15 | 20260912T040211-a9562c | 12146 | fastcode-shakaal | 0.691 | 0.957 | 0.856 | 0.933 | 0.997 | 0.681 | 0.951 |
| alpamayo15 | 20260903T183818-5879c1 | 12146 | fastcode-shakaal | 0.608 | 0.941 | 0.788 | 0.890 | 0.998 | 0.596 | 0.939 |
| visionpilot | 20260902T175642-df5291 | 12146 | fastcode-shakaal | 0.431 | 0.727 | 0.819 | 0.926 | 0.975 | 0.404 | 0.700 |
| meteor | 20260914T234307-e56cfc | 12146 | fastcode-shakaal | 0.303 | 0.694 | 0.687 | 0.805 | 0.979 | 0.317 | 0.683 |
| constant_velocity | 20260902T175923-feba73 | 12146 | fastcode-shakaal | 0.239 | 0.681 | 0.578 | 0.805 | 0.981 | 0.209 | 0.669 |

## navsim_v1

| model | run | n | site | pdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ | comfort ↑ | driving_direction_compliance ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | 20260911T082329-575df6 | 12146 | fastcode-shakaal | 0.892 | 0.984 | 0.985 | 0.887 | 0.881 | 0.999 | 0.988 |
| qwendrive | 20260915T021715-02aab4 | 12146 | fastcode-shakaal | 0.876 | 0.982 | 0.964 | 0.817 | 0.941 | 1.000 | 0.987 |
| openpilot | 20260914T181009-dbcef4 | 12146 | fastcode-shakaal | 0.807 | 0.969 | 0.913 | 0.719 | 0.936 | 1.000 | 0.948 |
| alpamayo15 | 20260914T175447-89a9ca | 12146 | fastcode-shakaal | 0.753 | 0.957 | 0.856 | 0.690 | 0.896 | 0.991 | 0.927 |
| alpamayo15 | 20260911T044805-0aea35 | 12146 | fastcode-shakaal | 0.676 | 0.941 | 0.788 | 0.604 | 0.878 | 0.979 | 0.885 |
| visionpilot | 20260911T083323-c07583 | 12146 | fastcode-shakaal | 0.488 | 0.771 | 0.818 | 0.451 | 0.617 | 0.996 | 0.920 |
| meteor | 20260915T005223-071c0b | 12146 | fastcode-shakaal | 0.366 | 0.693 | 0.687 | 0.340 | 0.582 | 0.997 | 0.800 |
| constant_velocity | 20260911T034553-8f28ac | 12146 | fastcode-shakaal | 0.207 | 0.680 | 0.578 | 0.194 | 0.500 | 1.000 | 0.799 |

## neuroncap

| model | run | n | site | ncap_score ↑ | impact_speed ↓ | reference_speed ↑ | collision_rate ↓ | duration_s ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| alpamayo15 | 20260901T002333-6dae70 | 2 | fastcode-shakaal | 5.000 | - | - | 0.000 | 267.700 |
| drivejepa | runs5 | 15 | fastcode-shakaal | 4.042 | 2.073 | 13.279 | 0.333 | - |
| drivejepa | runs1 | 3 | fastcode-shakaal | 4.024 | 2.133 | 13.279 | 0.333 | - |
| alpamayo15 | 20260906T034647-bc409f | 200 | fastcode-shakaal | 3.727 | 2.346 | 8.604 | 0.325 | - |
| openpilot | 20260905T055713-11ac59 | 200 | fastcode-shakaal | 3.604 | 2.698 | 8.604 | 0.380 | - |
| drivejepa | 20260905T220225-f11ac0 | 200 | fastcode-shakaal | 3.568 | 2.229 | 8.604 | 0.455 | - |
| drivejepa | 20260901T001634-44b6bb | 2 | fastcode-shakaal | 3.536 | - | - | 0.500 | 140.000 |
| visionpilot | 20260901T000037-c950bd | 1 | fastcode-shakaal | 2.827 | - | - | 1.000 | 89.100 |
| visionpilot | runs10 | 5 | fastcode-shakaal | 2.792 | 2.510 | 8.314 | 1.000 | - |
| meteor | 20260915T193036-d34559 | 200 | fastcode-shakaal | 2.644 | 5.051 | 8.604 | 0.480 | - |
| qwendrive | 20260915T023604-12d5eb | 200 | fastcode-shakaal | 2.487 | 4.514 | 8.604 | 0.690 | - |
| alpamayo | runs5 | 15 | fastcode-shakaal | 2.458 | 4.731 | 11.226 | 0.667 | - |
| alpamayo | runs1 | 3 | fastcode-shakaal | 2.059 | 6.393 | 11.557 | 0.667 | - |
| visionpilot | runs1 | 3 | fastcode-shakaal | 1.968 | 5.133 | 9.902 | 1.000 | - |
| drivejepa | scene0108_runs5 | 10 | fastcode-shakaal | 1.810 | 7.321 | 11.250 | 0.900 | - |
| visionpilot | fixcheck | 1 | fastcode-shakaal | 1.684 | 7.687 | 13.279 | 1.000 | - |
| visionpilot | runs5 | 15 | fastcode-shakaal | 1.671 | 6.099 | 10.246 | 1.000 | - |
| visionpilot | 20260905T121810-155d36 | 200 | fastcode-shakaal | 1.576 | 6.218 | 8.604 | 0.855 | - |
| visionpilot | runs1_fixed | 3 | fastcode-shakaal | 1.475 | 6.411 | 9.902 | 1.000 | - |
| visionpilot | runs1_noise0.3 | 3 | fastcode-shakaal | 1.371 | 13.452 | 13.279 | 1.000 | - |
| visionpilot | scene0108_runs5 | 10 | fastcode-shakaal | 1.352 | 8.055 | 10.743 | 1.000 | - |

## nuscenes

| model | run | n | site | l2_avg ↓ | l2_1s ↓ | l2_uniad_1s ↓ | col_1s ↓ | col_uniad_1s ↓ | l2_2s ↓ | l2_uniad_2s ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| qwendrive | 20260915T190350-47479e | 5119 | fastcode-shakaal | 0.800 | 0.318 | 0.245 | 0.007 | 0.007 | 0.722 | 0.427 |
| constant_velocity | 20260914T225156-e5170c | 5119 | fastcode-shakaal | 1.288 | 0.372 | 0.260 | 0.008 | 0.008 | 1.151 | 0.594 |
| constant_velocity | 20260907T093007-9697fe | 5119 | fastcode-navvrat-nusc-dj | 1.288 | 0.372 | 0.259 | 0.008 | 0.008 | 1.151 | 0.594 |
| alpamayo15 | 20260905T093547-3dd7cd | 5119 | fastcode-navvrat-nusc-al | 1.487 | 0.509 | 0.369 | 0.008 | 0.008 | 1.351 | 0.742 |
| drivejepa | 20260905T093335-4513d4 | 5119 | fastcode-navvrat-nusc-dj | 1.882 | 0.540 | 0.376 | 0.006 | 0.008 | 1.587 | 0.824 |
| meteor | 20260914T202122-4320bb | 5119 | fastcode-shakaal | 1.885 | 0.616 | 0.447 | 0.030 | 0.023 | 1.683 | 0.914 |
| visionpilot | 20260905T093333-319ef5 | 5119 | fastcode-navvrat-nusc-vp | 2.360 | 1.031 | 0.883 | 0.025 | 0.033 | 2.213 | 1.382 |
| openpilot | 20260908T110028-b71f4f | 5119 | fastcode-navvrat-nusc-vp | 4.057 | 1.925 | 1.424 | 0.013 | 0.011 | 4.054 | 2.470 |

## Excluded runs

- **bench2drive/openpilot/pooled50-40c29d24c37d542a-32d680** — Pooled recordings, not one configuration: driver-resume assistance on 16 of the original 24 routes; upstream longitudinal controller on the 26 expansion routes. Not comparable to an unassisted estimate. attempted outside the configured route subset and never scored: 11715 (20260903T171306-b348e3: no records in _checkpoint (the route did not score)). pooled across 3 recording setups (assisted, native-long, unassisted); one mean describes none of them, so this run does not rank -- the per-setup figures are in metrics.json by_group. 9 of 9 source runs kept no run.json, so their routes are attributed to openpilot on the tree they were found in; config.model_env and the rest of config are what 4/9 source runs recorded.
- **neuroncap/openpilot/legacy-scene0108_runs5** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/openpilot/legacy-runs5** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/openpilot/legacy-runs1** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/alpamayo/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/drivejepa/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/openpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/visionpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **nuplan/openpilot/20260917T220353-9e4f44** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/visionpilot/20260917T220044-97873a** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/constant_velocity/20260917T214417-f827db** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/meteor/20260917T195335-1b447c** — camera images cover 26 of 375 anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/drivejepa/20260916T204900-6a17de** — camera images cover 26 of 375 anchors (53 logs have no images); not comparable with a full-coverage run
- **comma2k19/qwendrive/20260915T214115-405f64** — Superseded by 20260916T220846-971379: the adapter that produced it no longer exists, so no paddock run can reproduce 1.016; it emitted all 50 poses at waypoint_dt 0.1 and score() aligns the ground truth onto the prediction's grid, which is also why comfort read 1932 m/s**3 of quantiser jerk
- **neuroncap/meteor/20260915T192353-6ed840** — 1 of the 20 official scenario-scene pairs; a subset, not a neuroNCAP score (stationary 10 scenes, frontal 5 scenes, side 5 scenes)
- **nuscenes/qwendrive/20260915T190154-9212ce** — --limit 50: a subset of the benchmark, run to test the pipeline
- **nuscenes/qwendrive/20260915T185751-94386b** — --limit 5: a subset of the benchmark, run to test the pipeline
- **neuroncap/qwendrive/20260915T022633-f5592e** — 3 of the 20 official scenario-scene pairs; a subset, not a neuroNCAP score (stationary 10 scenes, frontal 5 scenes, side 5 scenes)
- **navsim/meteor/20260914T234143-046cae** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/qwendrive/20260914T221555-9aa1c1** — --limit 12: a subset of the benchmark, run to test the pipeline
- **kitscenes/qwendrive/20260914T220507-b60641** — --limit 5: a subset of the benchmark, run to test the pipeline
- **navsim/meteor/20260914T205802-4cc7fb** — --limit 12: a subset of the benchmark, run to test the pipeline
- **nuscenes/meteor/20260914T202008-98df65** — --limit 5: a subset of the benchmark, run to test the pipeline
- **kitscenes/meteor/20260914T185332-24b8ff** — --limit 5: a subset of the benchmark, run to test the pipeline
- **comma2k19/openpilot/20260912T053225-2a91ca** — Superseded by 20260916T051616-2a91ca: that is the row the docs cite. The two are bit-identical on all 65 shared metrics -- same 10,165 samples, same sequences_sha256 -- and the later one additionally carries the nine proxy and miss-rate metrics added after this ran. Kept rather than deleted so the record of it having happened survives.
- **comma2k19/openpilot/20260912T051526-e9e08e** — --limit 20: a subset of the benchmark, run to test the pipeline
- **comma2k19/constant_velocity/20260911T231819-736a8a** — Reference baseline: straight ahead at the recorded speed, no model. Produced by tools/benchmarks/comma2k19/constant_velocity.py against the benchmark's own harness and scoring; video decoding skipped, since the policy reads no pixels. Stored under the pseudo-model 'constant_velocity' because it is not a registered model. Superseded by 20260916T162837-b89038: that is the row the docs cite. This one predates measuring comfort on a common 0.5 s grid, so its comfort and far-horizon columns are not comparable with the rows beside it. Kept rather than deleted so the record of it having happened survives.
- **navsim_v1/human/20260911T041230-67c4f3** — privileged: the human agent returns the recorded future the scorer measures against; an anchor for the scoring path, not a model; no model process ran: the recorded human future through the devkit scorer
- **navsim/alpamayo15/20260911T024038-d518fc** — 1 scenario(s) failed in the scorer and are excluded
- **navsim/openpilot/20260911T004312-98f3dc** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/alpamayo15/20260911T003908-ed7b86** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/openpilot/20260911T002247-98f3dc** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/alpamayo15/20260903T183459-753b33** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/drivejepa/20260902T172532-9c2de7** — SMOKE, not a result: 8 scenarios, isolated on PADDOCK_BASE_PORT=9100 to prove the launcher identity check and the infer_sequence override against a genuine drivejepa node (plans have three columns). Superseded by the full-split run.
- **navsim/drivejepa/20260902T172248-9c2de7** — INVALID -- cross-talk. This run's health check passed against a concurrently launched visionpilot node on the same port while drivejepa's own node was still loading; every request was then served by visionpilot. The recorded plans have two columns (visionpilot's shape; drivejepa emits x, y, heading) and the score is identical to visionpilot's on the same scenarios. Fixed: nodes answer /whoami, the launcher verifies identity, PADDOCK_BASE_PORT isolates concurrent runs.
- **navsim/visionpilot/20260902T172247-37d005** — SMOKE, not a result: 8 scenarios to exercise the four-node launch path. Superseded by the full-split run.
- **navsim/alpamayo15/20260901T213227-d6117e** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. alpamayo15; this is the current headline for alpamayo15 until its full-navtest run lands, and it is a subset figure.
- **navsim/alpamayo15/20260901T212250-b5daf0** — EXPERIMENT, not a result: the 'distinct' arm of the same paired A/B. 0.709 EPDMS over the same 60 scenarios, +0.207 over 'grid' (better on 36, worse on 19, sign test p=0.03), almost all of it collisions avoided (no_at_fault_collisions 0.708 -> 0.992). This selection is now the default via `auto`; the full 242-scenario run supersedes this.
- **navsim/visionpilot/20260901T211745-df5291** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. visionpilot; the comparable figure is the full-navtest row 20260902T175642-df5291 (0.43122).
- **navsim/alpamayo15/20260901T211713-b5daf0** — EXPERIMENT, not a result: the 'grid' arm of a paired A/B on camera frame selection, run with PADDOCK_NAVSIM_ALLOW_SLOW while alpamayo was still gated. 0.503 EPDMS over 60 scenarios with the legacy 0.1 s-grid selection, which collapses to two distinct frames on 2 Hz data. Paired against the 'distinct' arm: see docs/benchmarks/navsim.md.
- **navsim/alpamayo15/20260901T205152-ff0af4** — SUPERSEDED. n=4 smoke read as a result it could not support: 0.226 here against a 0.257 baseline, which prompted gating alpamayo off NAVSIM entirely. A 60-scenario run put the same configuration at 0.503. The gate was wrong; see docs/benchmarks/navsim.md.
- **navsim/visionpilot/20260901T205133-be57d6** — SMOKE, not a result: a 4-scenario run proves the pipeline (node, conversion, scorer, CSV) and nothing about the model. Twice now the n=4 figure has been contradicted by a larger run -- visionpilot 0.640 -> 0.447 at n=242, alpamayo 0.226 -> 0.503 at n=60. NAVSIM's floor for a meaningful score is far above 4; use the full-split run.
- **navsim/drivejepa/20260901T204614-b3c6ad** — SMOKE, not a result: a 4-scenario run proves the pipeline (node, conversion, scorer, CSV) and nothing about the model. Twice now the n=4 figure has been contradicted by a larger run -- visionpilot 0.640 -> 0.447 at n=242, alpamayo 0.226 -> 0.503 at n=60. NAVSIM's floor for a meaningful score is far above 4; use the full-split run.
- **navsim/openpilot/20260901T204522-8f4682** — openpilot infers ego motion from frame-to-frame change and NAVSIM supplies 2 Hz observations, so it was shown a near-stationary world: it planned 0.30 m of travel over 4 s while doing 4.8 m/s. The score measures the rate gap, not the model. model.yaml now declares min_input_hz: 10.0 and the harness refuses this pairing. See docs/benchmarks/navsim.md.
- **navsim/visionpilot/20260901T204347-be57d6** — SMOKE, not a result: a 4-scenario run proves the pipeline (node, conversion, scorer, CSV) and nothing about the model. Twice now the n=4 figure has been contradicted by a larger run -- visionpilot 0.640 -> 0.447 at n=242, alpamayo 0.226 -> 0.503 at n=60. NAVSIM's floor for a meaningful score is far above 4; use the full-split run.
- **navsim/openpilot/20260901T204239-8f4682** — openpilot infers ego motion from frame-to-frame change and NAVSIM supplies 2 Hz observations, so it was shown a near-stationary world: it planned 0.30 m of travel over 4 s while doing 4.8 m/s. The score measures the rate gap, not the model. model.yaml now declares min_input_hz: 10.0 and the harness refuses this pairing. See docs/benchmarks/navsim.md.
- **navsim/drivejepa/20260901T202524-7e208f** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. drivejepa; the comparable figure for that model on the subset is the full-navtest row 20260902T173156-7e208f (0.84272).
- **neuroncap/openpilot/20260901T004005-88a4e0** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/openpilot/20260831T235716-88a4e0** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.

---

Arrows show direction of better; a column with no arrow has none -- a signed bias, a ratio best at one, or a description of the model rather than a score. Aggregates are recomputed by paddock from per-sample records, never taken from the producer.
