# Paddock leaderboard

63 runs across 6 benchmarks.
31 run(s) excluded as invalid — shown at the end, never averaged in.

## bench2drive

| model | run | n | site | driving_score ↑ | route_completion ↑ | infraction_penalty ↑ | collision_rate ↓ | n_crashed ↑ | duration_s ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| constant_velocity | 20260908T084921-804913 | 24 | fastcode-navvrat-cv-b2d | 47.943 | 74.682 | 0.635 | 0.583 | 0.000 | 451.080 |

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
| openpilot | 20260911T161513-3a4c90 | 12146 | fastcode-shakaal | 0.747 | 0.970 | 0.913 | 0.954 | 0.998 | 0.712 | 0.965 |
| alpamayo15 | 20260912T040211-a9562c | 12146 | fastcode-shakaal | 0.691 | 0.957 | 0.856 | 0.933 | 0.997 | 0.681 | 0.951 |
| alpamayo15 | 20260903T183818-5879c1 | 12146 | fastcode-shakaal | 0.608 | 0.941 | 0.788 | 0.890 | 0.998 | 0.596 | 0.939 |
| visionpilot | 20260902T175642-df5291 | 12146 | fastcode-shakaal | 0.431 | 0.727 | 0.819 | 0.926 | 0.975 | 0.404 | 0.700 |
| constant_velocity | 20260902T175923-feba73 | 12146 | fastcode-shakaal | 0.239 | 0.681 | 0.578 | 0.805 | 0.981 | 0.209 | 0.669 |

## navsim_v1

| model | run | n | site | pdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ | comfort ↑ | driving_direction_compliance ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | 20260911T082329-575df6 | 12146 | fastcode-shakaal | 0.892 | 0.984 | 0.985 | 0.887 | 0.881 | 0.999 | 0.988 |
| openpilot | 20260914T181009-dbcef4 | 12146 | fastcode-shakaal | 0.807 | 0.969 | 0.913 | 0.719 | 0.936 | 1.000 | 0.948 |
| alpamayo15 | 20260914T175447-89a9ca | 12146 | fastcode-shakaal | 0.753 | 0.957 | 0.856 | 0.690 | 0.896 | 0.991 | 0.927 |
| alpamayo15 | 20260911T044805-0aea35 | 12146 | fastcode-shakaal | 0.676 | 0.941 | 0.788 | 0.604 | 0.878 | 0.979 | 0.885 |
| visionpilot | 20260911T083323-c07583 | 12146 | fastcode-shakaal | 0.488 | 0.771 | 0.818 | 0.451 | 0.617 | 0.996 | 0.920 |
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
| constant_velocity | 20260914T225156-e5170c | 5119 | fastcode-shakaal | 1.288 | 0.372 | 0.260 | 0.008 | 0.008 | 1.151 | 0.594 |
| constant_velocity | 20260907T093007-9697fe | 5119 | fastcode-navvrat-nusc-dj | 1.288 | 0.372 | 0.259 | 0.008 | 0.008 | 1.151 | 0.594 |
| alpamayo15 | 20260905T093547-3dd7cd | 5119 | fastcode-navvrat-nusc-al | 1.487 | 0.509 | 0.369 | 0.008 | 0.008 | 1.351 | 0.742 |
| drivejepa | 20260905T093335-4513d4 | 5119 | fastcode-navvrat-nusc-dj | 1.882 | 0.540 | 0.376 | 0.006 | 0.008 | 1.587 | 0.824 |
| meteor | 20260914T202122-4320bb | 5119 | fastcode-shakaal | 1.885 | 0.616 | 0.447 | 0.030 | 0.023 | 1.683 | 0.914 |
| visionpilot | 20260905T093333-319ef5 | 5119 | fastcode-navvrat-nusc-vp | 2.360 | 1.031 | 0.883 | 0.025 | 0.033 | 2.213 | 1.382 |
| openpilot | 20260908T110028-b71f4f | 5119 | fastcode-navvrat-nusc-vp | 4.057 | 1.925 | 1.424 | 0.013 | 0.011 | 4.054 | 2.470 |

## Excluded runs

- **neuroncap/openpilot/legacy-scene0108_runs5** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/openpilot/legacy-runs5** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/openpilot/legacy-runs1** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/alpamayo/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/drivejepa/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/openpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **neuroncap/visionpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **nuscenes/meteor/20260914T202008-98df65** — --limit 5: a subset of the benchmark, run to test the pipeline
- **kitscenes/meteor/20260914T185332-24b8ff** — --limit 5: a subset of the benchmark, run to test the pipeline
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

Arrows show direction of better. Aggregates are recomputed by paddock from per-sample records, never taken from the producer.
