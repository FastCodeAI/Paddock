# Paddock leaderboard

156 runs across 11 benchmarks.
100 run(s) excluded as invalid — shown at the end, never averaged in.

## alpasim

### n = 30, scene set f110e944

| model | run | n | scene_score ↑ | progress ↑ | collision_rate ↓ | collision_at_fault ↓ | offroad ↓ | duration_frac_20s ↑ | dist_to_gt_trajectory ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 | 20261029T200113-9d900c | 30 | 0.795 | 0.843 | 0.100 | 0.033 | 0.133 | 0.657 | 2.679 |
| openpilot | openpilot_020_30scenes | 30 | 0.773 | 0.724 | 0.200 | 0.000 | 0.100 | 0.671 | 1.787 |
| visionpilot | visionpilot_v2_30scenes | 30 | 0.629 | 0.636 | 0.367 | 0.133 | 0.033 | 0.570 | 1.740 |
| qwendrive | 20261110T062329-602ee2 | 30 | 0.535 | 0.602 | 0.433 | 0.167 | 0.100 | 0.564 | 1.662 |
| constant_velocity | cv_023_20hz_30scenes | 30 | 0.488 | 0.541 | 0.267 | 0.067 | 0.167 | 0.447 | 2.189 |
| alpamayo2edge | 20261112T220852-f6d995 | 30 | 0.437 | 0.628 | 0.233 | 0.233 | 0.200 | 0.370 | 2.942 |
| meteor | meteor_011_published_30scenes | 30 | 0.423 | 0.430 | 0.333 | 0.033 | 0.167 | 0.469 | 1.694 |
| drivejepa | drivejepa_014_20hz_30scenes | 30 | 0.421 | 0.509 | 0.400 | 0.233 | 0.167 | 0.465 | 2.056 |
| vad | 20261121T020047-8124e8 | 30 | 0.346 | 0.397 | 0.500 | 0.167 | 0.233 | 0.331 | 1.404 |
| uniad | 20261110T120651-8ea842 | 30 | 0.284 | 0.415 | 0.267 | 0.100 | 0.433 | 0.337 | 2.351 |
| autoe2e | 20261022T045847-86b955 | 30 | 0.169 | 0.287 | 0.333 | 0.300 | 0.233 | 0.121 | 2.855 |

### n = 10, scene set f4241f1e

| model | run | n | scene_score ↑ | progress ↑ | collision_rate ↓ | collision_at_fault ↓ | offroad ↓ | duration_frac_20s ↑ | dist_to_gt_trajectory ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| openpilot | openpilot_020_pilot10 | 10 | 0.957 | 0.901 | 0.100 | 0.000 | 0.000 | 0.777 | 1.454 |
| alpamayo15 | alpamayo_proxy_referee_20hz | 10 | 0.900 | 0.960 | 0.100 | 0.100 | 0.000 | 0.740 | 3.170 |
| constant_velocity | cv_023_20hz_pilot10 | 10 | 0.885 | 0.829 | 0.200 | 0.000 | 0.000 | 0.653 | 2.507 |
| visionpilot | visionpilot_v2_smoke_pilot10_handover_blend | 10 | 0.849 | 0.820 | 0.300 | 0.100 | 0.000 | 0.699 | 1.467 |
| meteor | meteor_011_published_first10 | 10 | 0.687 | 0.607 | 0.300 | 0.000 | 0.000 | 0.589 | 2.045 |
| drivejepa | drivejepa_014_20hz | 10 | 0.503 | 0.565 | 0.400 | 0.200 | 0.100 | 0.462 | 2.443 |

## bench2drive

### n = 220, scene set 86aa45e7

| model | run | n | driving_score ↑ | route_completion ↑ | infraction_penalty ↑ | collision_rate ↓ | n_runs ↑ | n_crashed ↑ | duration_s ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| visionpilot | pooled220-86aa45e768a29a64-9f3c17 | 220 | 46.097 | 66.039 | 0.650 | 0.523 | 1.000 | 0.000 | 301.989 |
| openpilot | full220-86aa45e768a29a64-102db6 | 220 | 43.871 | 55.961 | 0.811 | 0.232 | 1.000 | 0.000 | 176.775 |
| constant_velocity [1] | full220-3mps-86aa45e768a29a64-c8f3e4 | 220 | 41.427 | 62.992 | 0.642 | 0.527 | 1.000 | 0.000 | 110.686 |
| drivejepa | pooled220-86aa45e768a29a64-1507a1 | 220 | 40.742 | 52.674 | 0.827 | 0.241 | 1.000 | 0.000 | 168.731 |
| alpamayo15 | full220-86aa45e768a29a64-600411 | 220 | 38.221 | 56.775 | 0.744 | 0.400 | 1.000 | 0.000 | 1007.200 |
| qwendrive | pooled220-86aa45e768a29a64-a23127 | 220 | 27.409 | 61.695 | 0.467 | 0.786 | 1.000 | 0.000 | 227.102 |
| uniad | full220-86aa45e768a29a64-fb1f0a | 220 | 16.226 | 41.612 | 0.422 | 0.845 | 1.000 | 0.000 | 1381.757 |
| vad | full220-caffenorm-86aa45e768a29a64-17d468 | 220 | 15.679 | 33.641 | 0.653 | 0.532 | 1.000 | 0.000 | 851.401 |
| autoe2e | full220-86aa45e768a29a64-740c68 | 220 | 14.290 | 27.717 | 0.547 | 0.800 | 1.000 | 0.000 | 594.700 |
| alpamayo2edge | full220-86aa45e768a29a64-ffe60a | 220 | 13.871 | 38.781 | 0.483 | 0.655 | 1.000 | 0.000 | 825.825 |
| meteor | full220-86aa45e768a29a64-2144df | 220 | 1.887 | 5.299 | 0.926 | 0.118 | 1.000 | 0.000 | 112.850 |

- [1] constant_velocity: driven at a fixed 3 m/s, not the measured ego speed

### n = 50, scene set 40c29d24

| model | run | n | driving_score ↑ | route_completion ↑ | infraction_penalty ↑ | collision_rate ↓ | n_runs ↑ | n_crashed ↑ | duration_s ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | native50-2hz-40c29d24c37d542a-447e23 | 50 | 57.604 | 70.965 | 0.848 | 0.200 | 1.000 | 0.000 | 354.308 |
| constant_velocity [1] | pooled50-3mps-40c29d24c37d542a-02b6ac | 50 | 48.308 | 74.651 | 0.615 | 0.540 | 1.000 | 0.000 | 194.900 |
| visionpilot [2] | pooled50-40c29d24c37d542a-d94009 | 50 | 47.821 | 71.191 | 0.622 | 0.580 | 1.000 | 0.000 | 257.252 |
| openpilot | pooled50-40c29d24c37d542a-c7d2b2 | 50 | 47.520 | 62.441 | 0.783 | 0.260 | 1.000 | 0.000 | 374.050 |
| constant_velocity [3] | pooled50-9mps-40c29d24c37d542a-13c440 | 50 | 43.574 | 72.699 | 0.577 | 0.580 | 1.000 | 0.000 | 123.802 |
| alpamayo15 | pooled50-40c29d24c37d542a-36d2d0 | 50 | 43.043 | 72.317 | 0.655 | 0.540 | 1.000 | 0.000 | 3555.930 |
| constant_velocity [4] | pooled50-12mps-40c29d24c37d542a-0c11b2 | 50 | 42.193 | 72.125 | 0.558 | 0.560 | 1.000 | 0.000 | 108.372 |
| constant_velocity [5] | pooled50-15mps-40c29d24c37d542a-06f021 | 50 | 41.448 | 73.684 | 0.546 | 0.560 | 1.000 | 0.000 | 107.988 |
| constant_velocity [6] | pooled50-6mps-40c29d24c37d542a-35a5cb | 50 | 40.388 | 73.075 | 0.539 | 0.620 | 1.000 | 0.000 | 157.368 |
| drivejepa | pooled50-40c29d24c37d542a-1de367 | 50 | 31.706 | 39.101 | 0.862 | 0.160 | 1.000 | 0.000 | 1068.316 |
| autoe2e | pooled50-40c29d24c37d542a-cc7a9c | 50 | 15.297 | 26.371 | 0.596 | 0.720 | 1.000 | 0.000 | 2441.672 |
| qwendrive | pooled50-40c29d24c37d542a-f250e0 | 50 | 12.921 | 50.191 | 0.300 | 0.900 | 1.000 | 0.000 | 3457.266 |
| meteor | pooled50-40c29d24c37d542a-867d57 | 50 | 1.051 | 6.295 | 0.899 | 0.140 | 1.000 | 0.000 | 237.998 |

- [1] constant_velocity: driven at a fixed 3 m/s, not the measured ego speed
- [2] visionpilot: onnxruntime's cuda provider, not the declared CPU one; on KITScenes the two gave ADE 1.4712 and 1.4746 m, and 10 of 223 plans differ by over 0.5 m somewhere
- [3] constant_velocity: driven at a fixed 9 m/s, not the measured ego speed
- [4] constant_velocity: driven at a fixed 12 m/s, not the measured ego speed
- [5] constant_velocity: driven at a fixed 15 m/s, not the measured ego speed
- [6] constant_velocity: driven at a fixed 6 m/s, not the measured ego speed

### n = 49, scene set 08441825

| model | run | n | driving_score ↑ | route_completion ↑ | infraction_penalty ↑ | collision_rate ↓ | n_runs ↑ | n_crashed ↑ | duration_s ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 | native50-10hz-084418258d833a9f-c4d076 | 49 | 55.821 | 76.516 | 0.725 | 0.408 | 1.000 | 0.000 | 1986.029 |
| qwendrive | native50-2hz-084418258d833a9f-9b2bc9 | 49 | 17.075 | 64.299 | 0.291 | 0.939 | 1.000 | 0.000 | 812.441 |

## comma2k19

| model | run | n | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | scored_horizon_s | ade@1s ↓ | ade@2s ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo2edge [1] | 20261112T012152-9b1e09 | 10165 | 0.438 | 0.130 | 0.462 | 1.013 | 6.000 | 0.086 | 0.225 |
| qwendrive | 20261106T030530-7d9fd3 | 10165 | 0.589 | 0.212 | 0.615 | 1.316 | 5.000 | 0.154 | 0.326 |
| alpamayo15 | 20261106T005600-6c52a2 | 10165 | 0.819 | 0.238 | 0.888 | 1.864 | 4.000 | 0.152 | 0.427 |
| openpilot | 20261106T003719-5cde54 | 10165 | 0.843 | 0.407 | 0.938 | 1.611 | 10.000 | 0.299 | 0.548 |
| constant_velocity | 20261106T045541-d0cdb5 | 10165 | 1.135 | 0.370 | 1.211 | 2.546 | 3.000 | 0.252 | 0.611 |
| autoe2e | 20261106T073744-1b4fe1 | 10165 | 1.350 | 0.826 | 1.371 | 2.396 | 6.000 | 0.734 | 0.977 |
| visionpilot | 20261106T003749-c10a7f | 10165 | 3.592 | 1.372 | 4.010 | 7.415 | 4.000 | 0.964 | 2.126 |
| vad | 20261121T043612-597a2d | 10165 | 13.850 | 7.815 | 15.808 | 23.932 | 3.000 | 5.852 | 9.826 |
| uniad | 20261112T213250-a862cc | 10165 | 14.917 | 8.369 | 16.995 | 25.928 | 3.000 | 6.242 | 10.538 |
| drivejepa [2] | 20261111T074533-0f26ee | 10165 | 18.740 | 10.322 | 21.631 | 32.869 | 4.000 | 7.469 | 13.070 |
| meteor | 20261106T064120-226c00 | 10165 | 30.167 | 16.702 | 35.804 | 51.507 | 3.000 | 11.841 | 21.621 |

- [1] alpamayo2edge: slots with no source camera fed black frames; the front and telephoto real
- [2] drivejepa: streamed (warmup_0.5s): samples fed in order to one node that keeps its state, not one at a time

## hugsim

| model | run | n | hd_score ↑ | hugsim_pdms ↑ | route_completion ↑ | hugsim_nc ↑ | hugsim_dac ↑ | hugsim_ttc ↑ | hugsim_comfort ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | fullplan-f7d8f9d4b9736ddd-b89341 | 420 | 0.371 | 0.621 | 47.182 | 0.703 | 0.960 | 0.650 | 0.873 |
| qwendrive | fullplan-f7d8f9d4b9736ddd-1ddf85 | 420 | 0.348 | 0.523 | 46.659 | 0.617 | 0.927 | 0.552 | 0.928 |
| alpamayo2edge | 20261114T172855-b502b3 | 420 | 0.312 | 0.531 | 45.347 | 0.635 | 0.887 | 0.573 | 0.921 |
| uniad | fullplan-f7d8f9d4b9736ddd-9d698c | 420 | 0.261 | 0.350 | 45.720 | 0.543 | 0.910 | 0.379 | 0.475 |
| constant_velocity | fullplan-f7d8f9d4b9736ddd-a8f5b7 | 420 | 0.242 | 0.656 | 28.578 | 0.729 | 0.930 | 0.686 | 1.000 |
| alpamayo15 | fullplan-f7d8f9d4b9736ddd-7ea973 | 420 | 0.238 | 0.546 | 35.709 | 0.708 | 0.895 | 0.652 | 0.635 |
| vad | fullplan-caffenorm-f7d8f9d4b9736ddd-9d698c | 420 | 0.175 | 0.333 | 33.380 | 0.442 | 0.896 | 0.355 | 0.966 |
| visionpilot | fullplan-f7d8f9d4b9736ddd-92c0b7 | 420 | 0.163 | 0.352 | 34.545 | 0.676 | 0.943 | 0.489 | 0.078 |
| meteor | fullplan-f7d8f9d4b9736ddd-241d7f | 420 | 0.162 | 0.307 | 32.460 | 0.483 | 0.814 | 0.357 | 0.532 |
| openpilot | fullplan-f7d8f9d4b9736ddd-cac3b5 | 420 | 0.061 | 0.735 | 6.996 | 0.841 | 0.883 | 0.826 | 1.000 |
| autoe2e | fullplan-f7d8f9d4b9736ddd-4d9640 | 420 | 0.043 | 0.275 | 13.653 | 0.448 | 0.489 | 0.370 | 0.649 |

## kitscenes

| model | run | n | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | inference_seconds ↓ | baseline_ade_drivejepa ↓ | min_ade@6 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo2edge | 20261111T230906-22b8a7 | 223 | 0.752 | 0.400 | 0.847 | 1.393 | - | - | - |
| alpamayo2edge | 20261111T223048-778679 | 223 | 0.762 | 0.400 | 0.855 | 1.423 | - | - | - |
| qwendrive | 20260914T220841-5a91fd | 223 | 0.806 | 0.289 | 0.852 | 1.777 | - | - | - |
| autoe2e | 20261020T220642-d63624 | 223 | 0.837 | 0.274 | 0.885 | 1.886 | - | - | - |
| autoe2e | 20261020T220509-dee848 | 223 | 0.847 | 0.280 | 0.897 | 1.903 | - | - | - |
| autoe2e | 20261020T215800-4bd074 | 223 | 0.847 | 0.280 | 0.897 | 1.903 | - | - | - |
| autoe2e | 20261020T220816-0a24e7 | 223 | 0.921 | 0.288 | 0.984 | 2.083 | - | - | - |
| alpamayo15 | full-nonav | 223 | 0.921 | 0.332 | 0.990 | 2.003 | 3.415 | 2.364 | 0.552 |
| alpamayo15 | full-nav | 223 | 1.002 | 0.362 | 1.088 | 2.158 | 3.486 | 2.364 | 0.543 |
| alpamayo15 | full-letterbox | 223 | 1.176 | 0.418 | 1.284 | 2.535 | 3.384 | 2.364 | 0.619 |
| openpilot [1] [2] [3] [4] | 20261008T004723-a369d1 | 223 | 1.248 | 0.618 | 1.398 | 2.355 | - | - | - |
| constant_velocity | 20261112T200903-f1ac77 | 223 | 1.381 | 0.447 | 1.486 | 3.091 | - | - | - |
| visionpilot | 20261105T191932-c71101 | 223 | 1.475 | 0.679 | 1.497 | 2.968 | - | - | - |
| openpilot | 20260930T154018-242869 | 223 | 1.595 | 0.828 | 1.801 | 2.904 | - | - | - |
| openpilot | full223 | 223 | 1.599 | 0.836 | 1.808 | 2.894 | 18.731 | 2.364 | - |
| openpilot | 20260901T175755-cc27fb | 223 | 1.599 | 0.836 | 1.808 | 2.894 | - | - | - |
| vad | 20261120T124311-06a1d5 | 223 | 1.619 | 0.624 | 1.757 | 3.431 | - | - | - |
| openpilot [1] | 20260930T014333-d329a6 | 223 | 1.649 | 0.826 | 1.848 | 3.070 | - | - | - |
| meteor | 20260914T215457-ab0586 | 223 | 1.879 | 0.695 | 2.040 | 4.034 | - | - | - |
| uniad | 20261109T082536-b881a6 | 223 | 2.073 | 1.122 | 2.253 | 3.845 | - | - | - |
| drivejepa [5] | 20261111T054224-9f8371 | 223 | 2.188 | 1.242 | 2.453 | 3.867 | - | - | - |
| meteor | 20260914T185351-5e5223 | 223 | 4.784 | 2.334 | 5.542 | 9.011 | - | - | - |

- [1] openpilot: desire input mode set to turn; the default, off, sends none
- [2] openpilot: camera height taken as 1.22x openpilot's to fit this dataset's rig
- [3] openpilot: view tilted -1.7 degrees to fit this dataset's rig
- [4] openpilot: frame cut to a 55 degree field of view to fit this dataset's rig
- [5] drivejepa: streamed (warmup_0.5s): samples fed in order to one node that keeps its state, not one at a time

## navhard

| model | run | n | epdms ↑ | epdms_stage_one ↑ | no_at_fault_collisions_stage_one ↑ | drivable_area_compliance_stage_one ↑ | driving_direction_compliance_stage_one ↑ | traffic_light_compliance_stage_one ↑ | ego_progress_stage_one ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa [1] | 20261110T165434-167fd4 | 450 | 0.428 | 0.818 | 0.956 | 0.960 | 0.997 | 0.998 | 0.843 |
| qwendrive | 20261111T060112-5231af | 450 | 0.312 | 0.763 | 0.969 | 0.880 | 0.996 | 0.998 | 0.826 |
| openpilot [2] [3] [4] | 20261113T112649-2fe1d7 | 450 | 0.233 | 0.463 | 0.971 | 0.784 | 0.980 | 0.998 | 0.350 |
| alpamayo15 | 20261111T090014-04904f | 450 | 0.193 | 0.507 | 0.924 | 0.664 | 0.861 | 0.998 | 0.796 |
| visionpilot | 20261110T165413-30c0d8 | 450 | 0.192 | 0.453 | 0.921 | 0.618 | 0.871 | 0.993 | 0.786 |
| alpamayo2edge | 20261113T111924-f36c51 | 450 | 0.180 | 0.501 | 0.912 | 0.636 | 0.834 | 0.996 | 0.865 |
| uniad [5] | 20261110T092734-19b732 | 450 | 0.123 | 0.386 | 0.890 | 0.527 | 0.896 | 0.996 | 0.847 |
| constant_velocity | 20261110T085953-e122fd | 450 | 0.115 | 0.290 | 0.889 | 0.429 | 0.707 | 0.993 | 0.775 |
| vad [6] | 20261120T184739-5e470a | 450 | 0.101 | 0.320 | 0.863 | 0.484 | 0.831 | 0.996 | 0.858 |
| meteor [7] | 20261110T181854-9b4014 | 450 | 0.088 | 0.315 | 0.868 | 0.507 | 0.752 | 0.996 | 0.802 |
| autoe2e | 20261110T192825-24d0b3 | 450 | 0.082 | 0.330 | 0.858 | 0.520 | 0.649 | 0.993 | 0.829 |

- [1] drivejepa: streamed (per_log): samples fed in order to one node that keeps its state, not one at a time
- [2] openpilot: driving model run as a site-compiled CUDA build of the same graph, not on the CPU
- [3] openpilot: tinygrad device CUDA for the driving model
- [4] openpilot: tinygrad device CPU for the image warp
- [5] uniad: plan continued to 4.0 s by paddock; the model itself plans 3 s
- [6] vad: plan continued to 4.0 s by paddock; the model itself plans 3 s
- [7] meteor: plan continued to 4.0 s by paddock; the model itself plans 3 s

## navsim

| model | run | n | epdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | driving_direction_compliance ↑ | traffic_light_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa [1] | 20261111T052813-8f96dd | 12146 | 0.870 | 0.984 | 0.984 | 0.991 | 0.998 | 0.882 | 0.978 |
| qwendrive | 20260914T221802-80457c | 12146 | 0.840 | 0.982 | 0.964 | 0.993 | 0.998 | 0.820 | 0.977 |
| openpilot | 20260911T161513-3a4c90 | 12146 | 0.747 | 0.970 | 0.913 | 0.954 | 0.998 | 0.712 | 0.965 |
| alpamayo15 | 20260912T040211-a9562c | 12146 | 0.691 | 0.957 | 0.856 | 0.933 | 0.997 | 0.681 | 0.951 |
| alpamayo2edge | 20261111T052918-f9f90a | 12146 | 0.626 | 0.924 | 0.801 | 0.879 | 0.996 | 0.654 | 0.908 |
| alpamayo15 | 20260903T183818-5879c1 | 12146 | 0.608 | 0.941 | 0.788 | 0.890 | 0.998 | 0.596 | 0.939 |
| autoe2e | 20261009T055926-3ceb18 | 12146 | 0.500 | 0.824 | 0.777 | 0.865 | 0.990 | 0.491 | 0.814 |
| uniad [2] | 20261109T082522-c24426 | 12146 | 0.478 | 0.826 | 0.729 | 0.919 | 0.984 | 0.517 | 0.792 |
| visionpilot | 20261111T173726-ff827e | 12146 | 0.454 | 0.757 | 0.802 | 0.907 | 0.981 | 0.441 | 0.729 |
| vad [3] | 20261120T124116-88559f | 12146 | 0.452 | 0.806 | 0.741 | 0.897 | 0.980 | 0.492 | 0.780 |
| meteor [4] | 20260914T234307-e56cfc | 12146 | 0.303 | 0.694 | 0.687 | 0.805 | 0.979 | 0.317 | 0.683 |
| constant_velocity | 20260902T175923-feba73 | 12146 | 0.239 | 0.681 | 0.578 | 0.805 | 0.981 | 0.209 | 0.669 |

- [1] drivejepa: streamed (per_log): samples fed in order to one node that keeps its state, not one at a time
- [2] uniad: plan continued to 4.0 s by paddock; the model itself plans 3 s
- [3] vad: plan continued to 4.0 s by paddock; the model itself plans 3 s
- [4] meteor: plan continued to 4.0 s by paddock; the model itself plans 3 s

## navsim_v1

| model | run | n | pdms ↑ | no_at_fault_collisions ↑ | drivable_area_compliance ↑ | ego_progress ↑ | time_to_collision_within_bound ↑ | comfort ↑ | driving_direction_compliance ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa [1] | 20261111T090508-ad88e6 | 12146 | 0.894 | 0.984 | 0.984 | 0.879 | 0.893 | 0.999 | 0.988 |
| qwendrive | 20261104T205054-df25d5 | 12146 | 0.876 | 0.982 | 0.964 | 0.817 | 0.941 | 1.000 | 0.987 |
| openpilot | 20260914T181009-dbcef4 | 12146 | 0.807 | 0.969 | 0.913 | 0.719 | 0.936 | 1.000 | 0.948 |
| alpamayo15 | 20260914T175447-89a9ca | 12146 | 0.753 | 0.957 | 0.856 | 0.690 | 0.896 | 0.991 | 0.927 |
| alpamayo2edge | 20261111T070439-f19e8f | 12146 | 0.699 | 0.924 | 0.801 | 0.665 | 0.847 | 0.991 | 0.873 |
| alpamayo15 | 20260911T044805-0aea35 | 12146 | 0.676 | 0.941 | 0.788 | 0.604 | 0.878 | 0.979 | 0.885 |
| autoe2e | 20261104T210206-3022fb | 12146 | 0.538 | 0.824 | 0.777 | 0.514 | 0.700 | 1.000 | 0.859 |
| uniad | 20261112T130241-e02e88 | 12146 | 0.532 | 0.826 | 0.729 | 0.525 | 0.692 | 0.963 | 0.914 |
| vad | 20261120T183501-625133 | 12146 | 0.494 | 0.805 | 0.741 | 0.505 | 0.663 | 0.938 | 0.893 |
| visionpilot | 20261111T194001-cc5073 | 12146 | 0.472 | 0.756 | 0.802 | 0.442 | 0.605 | 0.997 | 0.902 |
| meteor [2] | 20260915T005223-071c0b | 12146 | 0.366 | 0.693 | 0.687 | 0.340 | 0.582 | 0.997 | 0.800 |
| constant_velocity | 20260911T034553-8f28ac | 12146 | 0.207 | 0.680 | 0.578 | 0.194 | 0.500 | 1.000 | 0.799 |

- [1] drivejepa: streamed (per_log): samples fed in order to one node that keeps its state, not one at a time
- [2] meteor: plan continued to 4.0 s by paddock; the model itself plans 3 s

## neuroncap

### n = 200, scene set ef927e96

| model | run | n | ncap_score ↑ | impact_speed ↓ | reference_speed ↑ | collision_rate ↓ | pdm_nc ↑ | pdm_dac ↑ | pdm_ttc ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drivejepa | 20261110T204020-036fb3 | 200 | 4.064 | 2.222 | 8.604 | 0.270 | 0.771 | 0.993 | 0.702 |
| alpamayo15 | 20260906T034647-bc409f | 200 | 3.727 | 2.346 | 8.604 | 0.325 | - | - | - |
| openpilot | 20260905T055713-11ac59 | 200 | 3.604 | 2.698 | 8.604 | 0.380 | - | - | - |
| autoe2e | 20261026T233951-743060 | 200 | 3.162 | 3.721 | 8.604 | 0.420 | - | - | - |
| alpamayo2edge | 20261111T231934-33adca | 200 | 2.918 | 3.634 | 8.604 | 0.540 | - | - | - |
| meteor | 20260915T193036-d34559 | 200 | 2.644 | 5.051 | 8.604 | 0.480 | - | - | - |
| qwendrive | 20260915T023604-12d5eb | 200 | 2.487 | 4.514 | 8.604 | 0.690 | - | - | - |
| uniad | 20261109T092826-5919f1 | 200 | 2.213 | 6.548 | 8.604 | 0.590 | 0.483 | 0.908 | 0.392 |
| visionpilot [1] | 20261111T204212-854d8b | 200 | 1.487 | 6.407 | 8.604 | 0.860 | 0.391 | 0.914 | 0.283 |
| vad | 20261122T002427-4bdf83 | 200 | 0.818 | 9.365 | 8.604 | 0.870 | 0.297 | 0.934 | 0.208 |
| constant_velocity | 20261112T200906-cce2d4 | 200 | 0.599 | 8.619 | 8.604 | 0.970 | 0.296 | 0.934 | 0.184 |

- [1] visionpilot: onnxruntime's cuda provider, not the declared CPU one; on KITScenes the two gave ADE 1.4712 and 1.4746 m, and 10 of 223 plans differ by over 0.5 m somewhere

### n = 15, scene set 5312d788

| model | run | n | ncap_score ↑ | impact_speed ↓ | reference_speed ↑ | collision_rate ↓ | pdm_nc ↑ | pdm_dac ↑ | pdm_ttc ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo | runs5 | 15 | 2.458 | 4.731 | 11.226 | 0.667 | - | - | - |

## nuplan

| model | run | n | ade ↓ | fde@1s ↓ | fde@2s ↓ | fde@3s ↓ | longitudinal_mae ↓ | lateral_mae ↓ | pdm_ep ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwendrive | 20261113T201149-e56771 | 351 | 0.293 | 0.100 | 0.317 | 0.639 | 0.247 | 0.092 | 0.976 |
| alpamayo15 | 20261113T224735-d4cf3b | 351 | 0.684 | 0.197 | 0.728 | 1.579 | 0.604 | 0.170 | 0.970 |
| alpamayo2edge | 20261113T235607-4feaff | 351 | 0.689 | 0.312 | 0.756 | 1.390 | 0.623 | 0.161 | 0.973 |
| autoe2e | 20261113T175630-abc9d8 | 351 | 0.826 | 0.270 | 0.885 | 1.855 | 0.636 | 0.355 | 0.960 |
| openpilot | 20261113T215414-4e1d08 | 351 | 0.922 | 0.457 | 1.021 | 1.744 | 0.853 | 0.198 | 0.937 |
| drivejepa | 20261113T190058-0128b9 | 351 | 1.249 | 0.543 | 1.345 | 2.543 | 1.201 | 0.167 | 0.967 |
| constant_velocity | 20261113T190322-f827db | 351 | 1.255 | 0.390 | 1.353 | 2.833 | 1.086 | 0.317 | 0.873 |
| visionpilot | 20261113T190336-8048ec | 351 | 1.707 | 0.815 | 1.779 | 3.298 | 1.336 | 0.676 | 0.874 |
| meteor | 20261113T194349-e44306 | 351 | 1.966 | 0.747 | 2.127 | 4.211 | 1.718 | 0.636 | 0.880 |

## nuscenes

| model | run | n | l2_avg ↓ | l2_1s ↓ | l2_uniad_1s ↓ | col_1s ↓ | col_uniad_1s ↓ | l2_2s ↓ | l2_uniad_2s ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwendrive | 20260915T190350-47479e | 5119 | 0.800 | 0.318 | 0.245 | 0.007 | 0.007 | 0.722 | 0.427 |
| alpamayo2edge | 20261111T231928-e4e2d6 | 5119 | 1.128 | 0.455 | 0.322 | 0.007 | 0.008 | 1.058 | 0.611 |
| uniad | 20261109T183938-41eed0 | 5119 | 1.166 | 0.533 | 0.433 | 0.018 | 0.015 | 1.075 | 0.678 |
| vad | 20261120T113536-4a641f | 5119 | 1.201 | 0.526 | 0.400 | 0.008 | 0.008 | 1.129 | 0.684 |
| constant_velocity | 20260914T225156-e5170c | 5119 | 1.288 | 0.372 | 0.260 | 0.008 | 0.008 | 1.151 | 0.594 |
| constant_velocity | 20260907T093007-9697fe | 5119 | 1.288 | 0.372 | 0.259 | 0.008 | 0.008 | 1.151 | 0.594 |
| alpamayo15 | 20260905T093547-3dd7cd | 5119 | 1.487 | 0.509 | 0.369 | 0.008 | 0.008 | 1.351 | 0.742 |
| autoe2e | 20261023T004711-eedbe2 | 5119 | 1.565 | 0.553 | 0.398 | 0.008 | 0.008 | 1.444 | 0.798 |
| visionpilot | 20261111T173725-c793ce | 5119 | 1.612 | 0.748 | 0.696 | 0.024 | 0.031 | 1.463 | 0.973 |
| meteor | 20260914T202122-4320bb | 5119 | 1.885 | 0.616 | 0.447 | 0.030 | 0.023 | 1.683 | 0.914 |
| drivejepa [1] | 20261111T061849-382fec | 5119 | 1.995 | 0.819 | 0.613 | 0.008 | 0.010 | 1.806 | 1.077 |
| openpilot | 20260929T174538-b479ee | 5119 | 4.029 | 1.897 | 1.403 | 0.012 | 0.011 | 4.018 | 2.442 |
| openpilot [2] [3] [4] | 20260908T110028-b71f4f | 5119 | 4.057 | 1.925 | 1.424 | 0.013 | 0.011 | 4.054 | 2.470 |

- [1] drivejepa: streamed (per_scene): samples fed in order to one node that keeps its state, not one at a time
- [2] openpilot: driving model run as a site-compiled CUDA build of the same graph, not on the CPU
- [3] openpilot: tinygrad device CUDA for the driving model
- [4] openpilot: tinygrad device CPU for the image warp

## Excluded runs

- **bench2drive/autoe2e/pooled50-steady-noroute-40c29d24c37d542a-8d26e9** — kitscenes-epoch-5 on Paddock 7244bc5, the 50 routes of the other pooled50 rows. AUTOE2E_EGO_HISTORY=steady (a steady straight past at the current speed) and AUTOE2E_ROUTE=0 (no route raster, no map), a forward every 0.1 s, paddock's camera rig; early stop at 30 s without 3 m of progress, the first recorded collision, or 4x Think2Drive's time, where the other pooled50 rows drove every route to Bench2Drive's own end; 3 workers on one A100 80GB. attempted outside the configured route subset and never scored: 3457 (20261028T050704-4d27a7: no records in _checkpoint (the route did not score)). 14 of 14 source runs kept no run.json, so their routes are attributed to autoe2e on the tree they were found in; config.model_env and the rest of config are what 14/14 source runs recorded. source runs set AUTOE2E_EGO_HISTORY, AUTOE2E_REPLAN_S, AUTOE2E_ROUTE, which the model's model.yaml declares, to other values (config.model_env), so this run does not rank, as a paddock run of them would not. stop_on_collision: a route ends at its first collision the leaderboard records, where Bench2Drive's own protocol charges the collision as a penalty (x0.50-0.65) and lets the car drive on, so its driving score is another measure. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) with no expert drive) or a collision the leaderboard records): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included).
- **bench2drive/openpilot/native50-20hz-40c29d24c37d542a-f68000** — OpenPilot on the 50 routes, 29 Oct 2026, one RTX PRO 6000 (shared), CARLA 0.9.15, six lanes. Driven at 20 Hz, which is openpilot's own rate and the benchmark protocol. One setup on every route: the tinygrad backend (PADDOCK_OPENPILOT_BACKEND=tinygrad), openpilot's own longitudinal plan (PADDOCK_OPENPILOT_NATIVE_LONG=1) and no driver-resume assistance (PADDOCK_OPENPILOT_RESUME_S=0). Driven from 9671b49 plus the TINYGRAD_ACTION_T fallback this commit adds to policy.py: the tinygrad ModelState has no long_action_t, so the native longitudinal planner raised AttributeError without it; the value is the 0.5 s the tinygrad path already feeds the model. Seven attempts failed on infrastructure and were re-run; their directories were moved out of this root, to _failed_infra beside it: node or CARLA start-up on 24795 (20261029T080029-0ffa0a), 3800 (20261029T080140-f99ef2), 2509 (20261029T080951-e076b4) and 26990 (20261029T095341-ac924f); the 2 h wall cap on the first attempts of 2084 (20261029T103048-d4e0c1), 2086 (20261029T104135-c8ba52) and 2050 (20261029T104914-1d3202), which ran at 3-4 s per tick under full CPU. It was driven to replace the 30.09 typed in from docs/benchmarks/bench2drive.md (24 routes, unassisted estimate) as OpenPilot's Bench2Drive cell; that cell is now #79's 50-route run of the setup openpilot/model.yaml declares (pooled50-40c29d24c37d542a-c7d2b2). 50 of 50 source runs kept no run.json, so their routes are attributed to openpilot on the tree they were found in; config.model_env and the rest of config are what 50/50 source runs recorded. source runs set DEV, PADDOCK_OPENPILOT_BACKEND, which the model's model.yaml declares, to other values (config.model_env), so this run does not rank, as a paddock run of them would not.
- **neuroncap/drivejepa/legacy-runs5** — Superseded by 20261012T224707-f11ac0: a small early run of 15 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261110T204020-036fb3: The row it pointed at, 20261012T224707-f11ac0, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **neuroncap/openpilot/legacy-runs5** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/visionpilot/legacy-runs5** — Superseded by 20260905T121810-155d36: a small early run of 15 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261111T204212-854d8b: 20260905T121810-155d36, which replaced it, is itself replaced by the post-CURV_SCALE re-run.
- **neuroncap/visionpilot/legacy-runs1_fixed** — Superseded by 20260905T121810-155d36: a small early run of 3 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261111T204212-854d8b: 20260905T121810-155d36, which replaced it, is itself replaced by the post-CURV_SCALE re-run.
- **neuroncap/drivejepa/legacy-runs1** — Superseded by 20261012T224707-f11ac0: a small early run of 3 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261110T204020-036fb3: The row it pointed at, 20261012T224707-f11ac0, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **neuroncap/visionpilot/legacy-runs1** — Superseded by 20260905T121810-155d36: a small early run of 3 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261111T204212-854d8b: 20260905T121810-155d36, which replaced it, is itself replaced by the post-CURV_SCALE re-run.
- **kitscenes/drivejepa/legacy-kitscenes_e2e_final** — Superseded by 20261111T054224-9f8371: Drive-JEPA's rows are its momentum-aware model now: 20261111T054224-9f8371 plans each keyframe after a discarded warm-up plan for the frame 0.5 s earlier (a Paddock choice, not upstream protocol; 223 of 223 fed). This pre-paddock run recorded no momentum setting, and its keyframes are about 9 s apart, so momentum, which hands over a plan made 0.45-0.55 s earlier, had none to hand.
- **neuroncap/visionpilot/legacy-INVALID_runs1_actor_not_rendered** — marked invalid at the source: the actor was not rendered, so the models were never shown the vehicle they were scored against. See patches/neuro-ncap/0002.
- **navsim/drivejepa/legacy-2026.08.08.02.26.44** — aggregate rows kept as context, not scored: average_all_frames=0.90673, pseudo_closed_loop=0.45337, average_expert_frames=0.90673. Source log reported 0.90488 over the same scenarios; paddock recomputes from per-sample scores. See the ingest module docstring. Superseded by 20260902T173156-7e208f: Covers 242 of the 12,146 scenarios, and came from the vendor's own agent, which reads the scorer's simulator while planning; the full run through paddock's proxy agent is the comparable score. Superseded by 20261111T052813-8f96dd: The row it pointed at, 20260902T173156-7e208f, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **hugsim/vad/fullplan-f7d8f9d4b9736ddd-9d698c** — HUGSIM release, the 420 scenarios left of 436 after the 16 the release cannot run (harness.EXCLUDED), driven 9-10 Nov 2026 on one RTX PRO 6000 (shared with other users), two or three lanes per model sharing one scenario queue, each lane one paddock node (proc.model_node) served from worktree eval-uv: that merge (main + PR #167 + PR #168, ffd85c3) plus the UniAD/VAD adapters of models/uniad-vad, the code PR #203 merged. HUGSIM 62c690d run by hugsim_run.py with --fix_heading, --render_cams and --fix_idm_lead, which make at run time the changes patches/HUGSIM/0001, 0002 and 0003 make in the checkout (only the model's own cameras rendered, the rest black; scores never read images). The bridge was paddock's own (hugsim/bridge.py, PR #168): it sends each plan's whole horizon, up to 8 waypoints 0.5 s apart (config.waypoints_sent), where the 22 Oct campaign's bridge sent the six the wire's trajectory keeps. Command only. Each model was asked at the planning rate its model.yaml declares (bridge.infer_every); HUGSIM steps at 4 Hz, so a model that declares 4 Hz or more, or no rate, is asked every step. A scenario whose simulator failed before its first step (exit 1, 0 steps: CUDA out of memory loading the scene while the shared GPU was full) was driven again from scratch; only its scored attempt counts. VAD base (VAD_base.pth, the released nuScenes checkpoint; VAD 1688c4b), zero-shot, asked at its declared 2 Hz (every second HUGSIM step) and holding its plan between; its plan is 3 s (6 waypoints 0.5 s apart), all sent. 418 scenarios exited 0 (88 at the 400-step cap); the other 2 (PandaSet 095 medium_00 and medium_01) ran their 400 steps and HUGSIM wrote their eval.json before the campaign runner logged its wall-clock timeout, so their scores are complete episodes. Superseded by fullplan-caffenorm-f7d8f9d4b9736ddd-9d698c: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **bench2drive/vad/full220-86aa45e768a29a64-6cb75d** — VAD base (VAD_base.pth, trained on nuScenes, zero-shot here) on all 220 Bench2Drive routes, 9-12 Nov 2026, at fastcode-shakaal (one RTX PRO 6000 Blackwell, shared with other users' jobs), six lanes (sites b2duv-lane0..5-fastcode-shakaal). Paddock eval/uv: PR #167's Bench2Drive code (patch 0003, /observe) with the uniad/vad adapters of models/uniad-vad. Bench2Drive 7ec25d1 + patches 0001-0003. 2 Hz (planning_hz), waypoint_dt 0.5, paddock's six-camera rig, no early stop (Bench2Drive's own end conditions, as the Drive-JEPA, Qwen-Drive and VisionPilot 220-route records), up to three attempts a route: an attempt the leaderboard ended as a simulation or agent crash was set aside and the route driven again. Under GPU contention the node ran at 0.009-0.05x real time; per_route_timeout_s was raised from 5400 to 10800 s (wall clock only), and route 23687 finished past it while only the harness process was paused (the evaluator, CARLA and node kept driving). 220 of 220 source runs kept no run.json, so their routes are attributed to vad on the tree they were found in; config.model_env and the rest of config are what 220/220 source runs recorded. asked for a plan at 2 Hz, the planning_hz the model's model.yaml declared for these runs, the last plan followed in between. Superseded by full220-caffenorm-86aa45e768a29a64-17d468: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **bench2drive/alpamayo15/full220-836dda734659a86c-f81033** — Alpamayo 1.5 10B (the Alpamayo-1.5-10B checkpoint, alpamayo1.5 pinned at 7a8f1c7), driven on Paddock 2eaedb5 (main after #159 and #160) over every route of bench2drive220.xml at Bench2Drive 7ec25d1, the routes set aside for every model left out (issue #158, the 220-route protocol). Its three front cameras from paddock's rig, and the route as navigation text (ALPAMAYO_NAV 1). Asked at its declared planning_hz of 10 (one /infer every 0.1 s, the last plan followed in between), the agent waiting up to 600 s for a plan (PADDOCK_INFER_TIMEOUT_S; the default is 120 s); early stop at 30 s without 3 m of progress or 4x Think2Drive's time; up to three attempts a route; no site value for anything model.yaml declares; 2 workers on each of two A100 80GB, 8 Nov 10:33 to 9 Nov 14:41 UTC. 4 route(s) set aside for every model (3800: CARLA crashes (signal 11) 63-69 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 11715: CARLA crashes (signal 11) about 54 s into the route on every attempt, whichever model drives; 23695: CARLA crashes (signal 11) about 40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 24071: CARLA crashes (signal 11) 37-40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz): a route the simulator fails on every time measures no model, so it is in no mean and not counted as lost. 57 of 57 source runs kept no run.json, so their routes are attributed to alpamayo15 on the tree they were found in; config.model_env and the rest of config are what 57/57 source runs recorded. asked for a plan at 10 Hz, the planning_hz the model's model.yaml declared for these runs, the last plan followed in between. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) here with no expert drive)): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included). Superseded by full220-86aa45e768a29a64-600411: Its four set-aside routes (3800, 11715, 23695, 24071) were driven under patch 0003 and pooled with its 216 routes into the 220-route record that replaces it.
- **bench2drive/meteor/full220-836dda734659a86c-dda00e** — METEOR v157, the released ONNX as shipped (meteor_v157c3Z.onnx, METEOR pinned at dc193a81; onnxruntime's CUDA provider), driven on Paddock 2eaedb5 (main after #159 and #160) over every route of bench2drive220.xml at Bench2Drive 7ec25d1, the routes set aside for every model left out (issue #158, the 220-route protocol). Asked at its declared planning_hz of 5 (one /infer every 0.2 s, the last plan followed in between); paddock's camera rig, six cameras in six of its eight slots, the two telephoto slots absent; no driving command (the graph takes none), the argmax of its plans; early stop at 30 s without 3 m of progress or 4x Think2Drive's time; up to three attempts a route; no site value for anything model.yaml declares; 3 workers on each of two A100 80GB, 7 Nov 02:36 to 06:01 UTC. 4 route(s) set aside for every model (3800: CARLA crashes (signal 11) 63-69 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 11715: CARLA crashes (signal 11) about 54 s into the route on every attempt, whichever model drives; 23695: CARLA crashes (signal 11) about 40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 24071: CARLA crashes (signal 11) 37-40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz): a route the simulator fails on every time measures no model, so it is in no mean and not counted as lost; 3800, 23695, 24071 left out although they scored here, as for every model. 37 of 37 source runs kept no run.json, so their routes are attributed to meteor on the tree they were found in; config.model_env and the rest of config are what 37/37 source runs recorded. asked for a plan at 5 Hz, the planning_hz the model's model.yaml declared for these runs, the last plan followed in between. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) here with no expert drive)): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included). Superseded by full220-86aa45e768a29a64-2144df: Its four set-aside routes (3800, 11715, 23695, 24071) were driven under patch 0003 and pooled with its 216 routes into the 220-route record that replaces it.
- **bench2drive/autoe2e/full220-836dda734659a86c-63902e** — AutoE2E kitscenes-epoch-5 (the kitscenes_finetune checkpoint of AutowareFoundation/auto_e2e v1.0, auto_e2e pinned at 40a75cf8), driven on Paddock 2eaedb5 (main after #159 and #160) over every route of bench2drive220.xml at Bench2Drive 7ec25d1, the routes set aside for every model left out (issue #158, the 220-route protocol). With upstream's inputs: the measured ego history and the route raster; paddock's camera rig, six cameras. Asked at its declared planning_hz of 10 (one /infer every 0.1 s, the last plan followed in between); early stop at 30 s without 3 m of progress or 4x Think2Drive's time; up to three attempts a route; no site value for anything model.yaml declares; 3 workers on each of two A100 80GB, 7 Nov 06:19 to 8 Nov 09:56 UTC. 4 route(s) set aside for every model (3800: CARLA crashes (signal 11) 63-69 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 11715: CARLA crashes (signal 11) about 54 s into the route on every attempt, whichever model drives; 23695: CARLA crashes (signal 11) about 40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 24071: CARLA crashes (signal 11) 37-40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz): a route the simulator fails on every time measures no model, so it is in no mean and not counted as lost; 3800, 23695, 24071 left out although they scored here, as for every model. 74 of 74 source runs kept no run.json, so their routes are attributed to autoe2e on the tree they were found in; config.model_env and the rest of config are what 74/74 source runs recorded. asked for a plan at 10 Hz, the planning_hz the model's model.yaml declared for these runs, the last plan followed in between. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) here with no expert drive)): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included). Superseded by full220-86aa45e768a29a64-740c68: Its four set-aside routes (3800, 11715, 23695, 24071) were driven under patch 0003 and pooled with its 216 routes into the 220-route record that replaces it.
- **bench2drive/alpamayo2edge/full220-836dda734659a86c-56db50** — Alpamayo 2 Edge 2.65B (the nvidia/Alpamayo2-Edge checkpoint at revision f8951d0, alpamayo2-edge pinned at 02f55a7), driven on Paddock f873bc2 (branch alpamayo2edge/bench2drive, PR #198) over every route of bench2drive220.xml at Bench2Drive 7ec25d1, the routes set aside for every model left out (issue #158, the 220-route protocol). Five of paddock's cameras and a 30 deg crop of the front one, and the route as navigation text. Asked at its declared planning_hz of 10 (one /infer every 0.1 s, the last plan followed in between), the controller following the plan's own points every 0.5 s; torch on 4 CPU threads (model.yaml's ALPAMAYO2_EDGE_THREADS); the agent waiting up to 600 s for a plan (PADDOCK_INFER_TIMEOUT_S; the default is 120 s); early stop at 30 s without 3 m of progress or 4x Think2Drive's time; up to three attempts a route; no site value for anything model.yaml declares; 4 workers on each of two A100 80GB, 12 Nov 02:48 to 14:27 UTC. 4 route(s) set aside for every model (3800: CARLA crashes (signal 11) 63-69 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 11715: CARLA crashes (signal 11) about 54 s into the route on every attempt, whichever model drives; 23695: CARLA crashes (signal 11) about 40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 24071: CARLA crashes (signal 11) 37-40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz): a route the simulator fails on every time measures no model, so it is in no mean and not counted as lost. 55 of 55 source runs kept no run.json, so their routes are attributed to alpamayo2edge on the tree they were found in; config.model_env and the rest of config are what 55/55 source runs recorded. asked for a plan at 10 Hz, the planning_hz the model's model.yaml declared for these runs, the last plan followed in between. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) here with no expert drive)): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included). Superseded by full220-86aa45e768a29a64-ffe60a: Its four set-aside routes (3800, 11715, 23695, 24071) were driven under patch 0003 and pooled with its 216 routes into the 220-route record that replaces it.
- **bench2drive/openpilot/full220-836dda734659a86c-2803bd** — openpilot driving_supercombo (small), the ONNX graph at the pinned openpilot 555f48c5d on onnxruntime's CUDA provider, driven on Paddock 0f0b9ef (the head of PR #160; its driving code is main's after #159 and #160 merged) over every route of bench2drive220.xml at Bench2Drive 7ec25d1, the routes set aside for every model left out (issue #158, the 220-route protocol). Asked every tick at Bench2Drive's 20 Hz, supercombo's own rate; its comma-three camera pair from paddock's rig; openpilot's own longitudinal planner with its driver-resume nudge (PADDOCK_OPENPILOT_NATIVE_LONG 1, declared in model.yaml; RESUME_S 3 s): the nudge stands in for a driver pressing resume; no desire input; early stop at 30 s without 3 m of progress or 4x Think2Drive's time; up to three attempts a route; no site value for anything model.yaml declares; 5 workers on each of two A100 80GB, 6 Nov 23:39 to 7 Nov 02:31 UTC. 4 route(s) set aside for every model (3800: CARLA crashes (signal 11) 63-69 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 11715: CARLA crashes (signal 11) about 54 s into the route on every attempt, whichever model drives; 23695: CARLA crashes (signal 11) about 40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz; 24071: CARLA crashes (signal 11) 37-40 s into the route on every attempt by Alpamayo 1.5 at its declared 10 Hz): a route the simulator fails on every time measures no model, so it is in no mean and not counted as lost; 3800, 23695, 24071 left out although they scored here, as for every model. 37 of 37 source runs kept no run.json, so their routes are attributed to openpilot on the tree they were found in; config.model_env and the rest of config are what 37/37 source runs recorded. early stop (30 s without 3 m of route progress or 4x the expert's time on the route (none on the 9 route(s) here with no expert drive)): a car in a driving scene is expected to move, to finish in reasonable time and not to collide, so a route ended this way is a failed drive, scored as it stood when it ended (a collision's penalty included). Superseded by full220-86aa45e768a29a64-102db6: Its four set-aside routes (3800, 11715, 23695, 24071) were driven under patch 0003 and pooled with its 216 routes into the 220-route record that replaces it.
- **hugsim/autoe2e/campaign-f7d8f9d4b9736ddd-d03155** — AutoE2E v1.0 (KITScenes epoch 5), driven 22 Oct 2026 in up to four lanes sharing one scenario queue, from paddock b507811 unmodified. The adapter has changed since (#92, #97, #98, #107, #108), so a re-run on main may differ. Called every step (it declares no planning rate), all six cameras rendered. It takes no command, so it gets upstream's no-route input. Superseded by fullplan-f7d8f9d4b9736ddd-4d9640: the 22 Oct campaign's bridge sent HUGSIM only the six waypoints (3 s) of each plan that the wire's trajectory keeps, so the scorer judged a cut plan, and it ran without the IDM guard; the 8-9 Nov campaign drove the same 420 scenarios through paddock's own bridge with the whole plan (up to 4 s) and the guard.
- **hugsim/drivejepa/campaign-87d4f5097a85db3c-ce3603** — Released Drive-JEPA (NAVSIM v2 ViT-L checkpoint), driven 22 Oct 2026 in two lanes sharing one scenario queue. The node was the drivejepa adapter as of paddock ac7788f plus a per-call record of the 32 candidates and their scores, which does not change the plan; its camera placement goes through lidar2ego, which the bridge sends as the identity, so the cameras land where ac7788f puts them. At that commit the adapter fed cameras uncropped and picked on the plain score; main now crops to the OpenScene camera (#106) and turns momentum selection on (#95), so a re-run on main would differ. Only the front camera was rendered: the released model's forward reads only the front camera. DRIVEJEPA_IMAGE_GEOMETRY, DRIVEJEPA_MOMENTUM were supplied by the ingest config: the campaign recorded no model env, so this is the operator's statement of how the node was configured, not a reading from it. scenarios attempted and never scored: scene-0166-medium-01 (exit 1 after 60 steps: IndexError: index is out of bounds for dimension with size 0). Superseded by fullplan-f7d8f9d4b9736ddd-b89341: the 22 Oct campaign's bridge sent HUGSIM only the six waypoints (3 s) of each plan that the wire's trajectory keeps, so the scorer judged a cut plan, and it ran without the IDM guard; the 8-9 Nov campaign drove the same 420 scenarios through paddock's own bridge with the whole plan (up to 4 s) and the guard. Momentum selection and the OpenScene crop, both off here, were on there, as model.yaml declares.
- **nuplan/alpamayo2edge/20261113T233310-4feaff** — Superseded by 20261113T235607-4feaff: Re-run after the nuPlan plan-spacing fix; the earlier run scored a 0.1 s plan at the wrong times.
- **nuplan/openpilot/20261113T212314-4e1d08** — Superseded by 20261113T215414-4e1d08: Re-run at the pinned commit 555f48c5d; the earlier run used 4df40d2.
- **comma2k19/vad/20261112T213250-61c395** — resumed: 7946 sample(s) reused from an earlier attempt, so started_utc and duration_s describe this attempt only Superseded by 20261121T043612-597a2d: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **navsim_v1/vad/20261112T132424-25e0e5** — no model process ran: 12146 plans replayed from navsim/vad/20261110T095855-0ebf09 (that run: epdms 0.4277); the same plans, this benchmark's scorer Superseded by 20261120T183501-625133: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **nuplan/drivejepa/20261112T012946-7361a6** — site env override may change what was measured: PYTHONNOUSERSITE (values in config.model_env) Superseded by 20261113T190058-0128b9: Made from a checkout with uncommitted changes; replaced by a clean run on 13 Nov.
- **nuplan/meteor/20261112T012837-214b68** — Superseded by 20261113T194349-e44306: Re-run from a clean checkout; same settings.
- **nuplan/constant_velocity/20261112T010902-f827db** — Superseded by 20261113T190322-f827db: Re-run from a clean checkout; same settings.
- **nuplan/openpilot/20261112T004253-422776** — site env override may change what was measured: CACHEDB, PADDOCK_FFMPEG, PYTHONNOUSERSITE, XDG_CACHE_HOME (values in config.model_env) Superseded by 20261113T215414-4e1d08: Made from a checkout with uncommitted changes; replaced by a clean run on 13 Nov.
- **nuplan/visionpilot/20261112T004207-8048ec** — Superseded by 20261113T190336-8048ec: Re-run from a clean checkout; same settings.
- **navsim/alpamayo2edge/20261111T031938-f9f90a** — 2 scenario(s) failed in the scorer and are excluded
- **navhard/vad/20261110T185531-0cecc0** — Superseded by 20261120T184739-5e470a: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **navsim_v1/vad/20261110T160343-25e0e5** — replayed from navsim/vad/20261110T095855-0ebf09, which is stored valid=false: site env override may change what was measured: VAD_HORIZON_S (values in config.model_env); no model process ran: 12146 plans replayed from navsim/vad/20261110T095855-0ebf09 (that run: epdms 0.4277, stored valid=false); the same plans, this benchmark's scorer Superseded by 20261112T132424-25e0e5: the same plans replayed again from the NAVSIM run after its record was re-written valid (byte-identical checkpoint override no longer counted as one) Superseded by 20261120T183501-625133: Its replacement 20261112T132424-25e0e5 is itself superseded by this re-run with the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **navsim_v1/uniad/20261110T154404-e02e88** — replayed from navsim/uniad/20261109T082522-c24426, which is stored valid=false: site env override may change what was measured: UNIAD_HORIZON_S (values in config.model_env); no model process ran: 12146 plans replayed from navsim/uniad/20261109T082522-c24426 (that run: epdms 0.4781, stored valid=false); the same plans, this benchmark's scorer Superseded by 20261112T130241-e02e88: the same plans replayed again from the NAVSIM run after its record was re-written valid (byte-identical checkpoint override no longer counted as one)
- **alpasim/vad/20261110T153825-72448c** — Superseded by 20261121T020047-8124e8: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **navsim/vad/20261110T095855-0ebf09** — Superseded by 20261120T124116-88559f: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **neuroncap/vad/20261110T033544-7f435b** — resumed: 8 episode(s) reused from an earlier attempt, so started_utc and duration_s describe this attempt only Superseded by 20261122T002427-4bdf83: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **nuplan/visionpilot/20261110T012321-8048ec** — Superseded by 20261112T004207-8048ec: Same settings and numbers as the 12 Nov run; kept for the record. Superseded by 20261113T190336-8048ec: Replaced by a clean re-run on 13 Nov; same settings.
- **nuplan/meteor/20261110T010424-214b68** — Superseded by 20261112T012837-214b68: Same settings and numbers as the 12 Nov run; kept for the record. Superseded by 20261113T194349-e44306: Replaced by a clean re-run on 13 Nov; same settings.
- **nuplan/constant_velocity/20261110T004701-f827db** — Superseded by 20261112T010902-f827db: Same settings and numbers as the 12 Nov run; kept for the record. Superseded by 20261113T190322-f827db: Replaced by a clean re-run on 13 Nov; same settings.
- **nuscenes/vad/20261110T000020-53cda0** — Superseded by 20261120T113536-4a641f: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **kitscenes/vad/20261109T082536-a393ec** — Superseded by 20261120T124311-06a1d5: Fed the config's ImageNet normalisation; the later run feeds the caffe normalisation VAD's released weights were trained with (VAD_IMG_NORM released, PR #228).
- **comma2k19/drivejepa/20261106T192049-391e2d** — Superseded by 20261111T074533-0f26ee: Drive-JEPA's rows are its momentum-aware model now: 20261111T074533-0f26ee plans each keyframe after a discarded warm-up plan for the frame 0.5 s earlier (a Paddock choice, not upstream protocol; 10,163 of 10,165 fed). Momentum was on in this run but could not act: each open-loop sample reset the node (the node whose counts the record keeps fed 0 of 1,271 replans).
- **comma2k19/drivejepa_free/20261106T053407-587a3e** — Superseded by drivejepa/20261106T192049-391e2d: Drive-JEPA's comma2k19 row is now the perception-based model that every other benchmark runs. This perception-free agent was its comma2k19 row only because, when comma2k19 was added in September 2026, the perception-based adapter needed a lidar mounting that comma2k19 does not have; that need went in October 2026. Superseded by drivejepa/20261111T074533-0f26ee: The row it pointed at, drivejepa/20261106T192049-391e2d, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **nuscenes/drivejepa/20261105T201953-308dc5** — Superseded by 20261111T061849-382fec: Drive-JEPA's rows are its momentum-aware model now: 20261111T061849-382fec plans each scene's keyframes in order on one node (streamed per scene, 4,485 of 5,119 keyframes fed). Momentum was on in this run but never acted: 0 of 5,119 plans were given a previous plan, because each open-loop sample reset the node.
- **comma2k19/meteor/20261030T001935-0b8d0b** — Superseded by 20261106T064120-226c00: Re-run on 6 Nov 2026 with the clone at its pinned dc193a81 and scored against the ground truth in the car's frame; this row came from a clone at 8567eb0 and was scored against a ground truth in the camera's frame. The plans are the same in both runs.
- **nuplan/qwendrive/20261028T044026-02ca5a** — site env override may change what was measured: QWENDRIVE_CHECKPOINT (values in config.model_env) Superseded by 20261113T201149-e56771: Made from a checkout with uncommitted changes; replaced by a clean run on 13 Nov.
- **nuplan/alpamayo15/20261028T043107-9049e6** — site env override may change what was measured: ALPAMAYO_COSMOS (values in config.model_env) Superseded by 20261113T224735-d4cf3b: Made from a checkout with uncommitted changes; replaced by a clean run on 13 Nov.
- **alpasim/qwendrive/20261027T215528-d231d8** — Superseded by 20261110T062329-602ee2: Qwen-Drive asked every 0.5 s, its declared planning_hz 2.0, as the node is built to be called; this run asked every 1.5 s, the native driver's setting
- **navsim_v1/autoe2e/20261027T185517-5c7949** — no model process ran: 12146 plans replayed from navsim/autoe2e/20261009T055926-3ceb18 (that run: epdms 0.5002); the same plans, this benchmark's scorer Superseded by 20261104T210206-3022fb: this replay recorded 2 Hz OpenScene history and 1.5 s, but its plans were made at 10 Hz from nuPlan with 6.5 s; re-made with the right record, every score identical
- **comma2k19/autoe2e/20261020T213543-417344** — Superseded by 20261106T073744-1b4fe1: Re-run on 6 Nov 2026 with its past path in the car's frame (#143) and scored against the ground truth in the car's frame (issue #141); this row was shown its past path and scored in the camera's frame.
- **neuroncap/wajepa/20261015T234134-916e11** — 3 of the 20 official scenario-scene pairs; a subset, not a neuroNCAP score (stationary 10 scenes, frontal 5 scenes, side 5 scenes)
- **neuroncap/drivejepa/20261012T224707-f11ac0** — resumed: 16 episode(s) reused from an earlier attempt, so started_utc and duration_s describe this attempt only Superseded by 20261110T204020-036fb3: Drive-JEPA's rows are its momentum-aware model now: 20261110T204020-036fb3 is the same 200-run default protocol with momentum-aware selection on, as model.yaml has declared it since 26 Oct 2026. Momentum never acted in this run: it was made on 12-13 Oct 2026, when the selection stage was off by default, and its record carries no momentum setting.
- **kitscenes/openpilot/20261005T205400-a369d1** — resumed: 223 sample(s) reused from an earlier attempt, so started_utc and duration_s describe this attempt only Superseded by 20261008T004723-a369d1: the same configuration (hash a369d1) run again from #60's head, identical on all 223 keyframes; the newer run is the record
- **kitscenes/openpilot/20260929T043619-242869** — Superseded by 20260930T154018-242869: The warp-cache bug fixed in f8dce3b warped 14 of its 223 samples with another rig's calibration; the re-run on the fixed code equals it bit for bit on the other 209, and equals a fresh-warp replay on each of the 51 samples checked (every sample that follows a change of rig, and the 3 that inherited a hit within their rig).
- **nuplan/openpilot/20260917T220353-9e4f44** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/visionpilot/20260917T220044-97873a** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/constant_velocity/20260917T214417-f827db** — camera images cover 25 of 351 scorable anchors (53 logs have no images); not comparable with a full-coverage run
- **nuplan/meteor/20260917T195335-1b447c** — camera images cover 26 of 375 anchors (53 logs have no images); not comparable with a full-coverage run
- **comma2k19/qwendrive/20260916T220846-971379** — Superseded by 20261106T030530-7d9fd3: Re-run on 6 Nov 2026 with black side views (#146) and its past path in the car's frame (#143), and scored against the ground truth in the car's frame (issue #141); this row copied the road camera into the side views (QWENDRIVE_VIEWS=replicate) and was scored against a ground truth in the camera's frame.
- **nuplan/drivejepa/20260916T204900-6a17de** — camera images cover 26 of 375 anchors (53 logs have no images); not comparable with a full-coverage run
- **comma2k19/constant_velocity/20260916T162837-b89038** — Superseded by 20261106T045541-d0cdb5: Re-run on 6 Nov 2026 and scored against the ground truth in the car's frame (issue #141); this row was scored against a ground truth in the camera's frame. The plans did not change: constant velocity plans from the recorded speed alone.
- **comma2k19/openpilot/20260916T051616-2a91ca** — Superseded by 20261106T003719-5cde54: Re-run on 6 Nov 2026 at the pinned 555f48c5d and scored with the ground truth in the car's frame (issue #141); this row came from a clone at 4df40d2c1 and was scored against a ground truth in the camera's frame.
- **comma2k19/visionpilot/20260916T012719-03350c** — Superseded by 20261106T003749-c10a7f: Re-run on 6 Nov 2026 at the pinned e93eceaa with CURV_SCALE 0.21 on the CPU provider, and scored against the ground truth in the car's frame (issue #141); this row ran on the CUDA provider, did not record its commit, and was scored against a ground truth in the camera's frame.
- **comma2k19/drivejepa_free/20260916T003407-c504e5** — Superseded by 20261106T053407-587a3e: Re-run on 6 Nov 2026 with the clone at its pin and its past path in the car's frame (#143), and scored against the ground truth in the car's frame (issue #141); this row was shown its past path and scored in the camera's frame, and did not record its commit. Superseded by drivejepa/20261106T192049-391e2d: The row it pointed at, 20261106T053407-587a3e, is retired too: Drive-JEPA's comma2k19 row is now the perception-based model, so this one points at that run. Superseded by drivejepa/20261111T074533-0f26ee: The row it pointed at, drivejepa/20261106T192049-391e2d, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **comma2k19/qwendrive/20260915T214115-405f64** — Superseded by 20260916T220846-971379: the adapter that produced it no longer exists, so no paddock run can reproduce 1.016; it emitted all 50 poses at waypoint_dt 0.1 and score() aligns the ground truth onto the prediction's grid, which is also why comfort read 1932 m/s**3 of quantiser jerk Superseded by 20261106T030530-7d9fd3: the row it pointed at, 20260916T220846-971379, is itself superseded by the re-run of 6 Nov 2026 (issue #141), so this one points at the ranked row.
- **neuroncap/meteor/20260915T192353-6ed840** — 1 of the 20 official scenario-scene pairs; a subset, not a neuroNCAP score (stationary 10 scenes, frontal 5 scenes, side 5 scenes)
- **navsim_v1/qwendrive/20260915T021715-02aab4** — no model process ran: 12146 plans replayed from navsim/qwendrive/20260914T221802-80457c (that run: epdms 0.8402); the same plans, this benchmark's scorer Superseded by 20261104T205054-df25d5: this replay recorded 2 Hz OpenScene history, but its plans were made at 10 Hz from nuPlan; re-made with the right record, every score identical
- **comma2k19/alpamayo15/20260913T072437-b058c2** — Superseded by 20261106T005600-6c52a2: Re-run on 6 Nov 2026 with the clone at its pinned 7a8f1c7 and its past path in the car's frame (#143), and scored against the ground truth in the car's frame (issue #141); this row was shown its past path and scored in the camera's frame. The model samples its plan, so the re-run is also a new draw.
- **comma2k19/openpilot/20260912T053225-2a91ca** — Superseded by 20260916T051616-2a91ca: that is the row the docs cite. The two are bit-identical on all 65 shared metrics -- same 10,165 samples, same sequences_sha256 -- and the later one additionally carries the nine proxy and miss-rate metrics added after this ran. Kept rather than deleted so the record of it having happened survives. Superseded by 20261106T003719-5cde54: the row it pointed at, 20260916T051616-2a91ca, is itself superseded by the re-run at the pinned 555f48c5d (issue #141), so this one points at the ranked row.
- **comma2k19/constant_velocity/20260911T231819-736a8a** — Reference baseline: straight ahead at the recorded speed, no model. Produced by tools/benchmarks/comma2k19/constant_velocity.py against the benchmark's own harness and scoring; video decoding skipped, since the policy reads no pixels. Stored under the pseudo-model 'constant_velocity' because it is not a registered model. Superseded by 20260916T162837-b89038: that is the row the docs cite. This one predates measuring comfort on a common 0.5 s grid, so its comfort and far-horizon columns are not comparable with the rows beside it. Kept rather than deleted so the record of it having happened survives. Superseded by 20261106T045541-d0cdb5: the row it pointed at, 20260916T162837-b89038, is itself superseded by the re-run of 6 Nov 2026 (issue #141), so this one points at the ranked row.
- **navsim_v1/visionpilot/20260911T083323-c07583** — Plan dump has 12,589 lines for 12,146 scenarios: 342 scenarios were planned twice, which looks like the devkit's worker pool re-running a log; the CSV and n are clean, the model is deterministic, and a replay keys plans by token, so the duplicates resolve to one plan each. Noted after the run (2026-09-15, review of PR #32); not investigated. Superseded by 20261111T194001-cc5073: Re-run on 11 Nov 2026 at the pinned e93eceaa with CURV_SCALE 0.21 (the September 2026 fix) on the declared CPU provider; this run used 1.0.
- **navsim_v1/drivejepa/20260911T082329-575df6** — Superseded by 20261111T090508-ad88e6: Drive-JEPA's rows are its momentum-aware model now: 20261111T090508-ad88e6 scores the plans of the streamed navtest run 20261111T052813-8f96dd with the v1.1 scorer. Momentum never acted in this run: it predates the selection stage, and each open-loop sample reset the node.
- **navsim_v1/human/20260911T041230-67c4f3** — privileged: the human agent returns the recorded future the scorer measures against; an anchor for the scoring path, not a model; no model process ran: the recorded human future through the devkit scorer
- **navsim/alpamayo15/20260911T024038-d518fc** — 1 scenario(s) failed in the scorer and are excluded
- **kitscenes/visionpilot/20260909T202724-153b04** — Superseded by 20261105T191932-c71101: Re-run on 5 Nov 2026 with CURV_SCALE 0.21; this run predates the September 2026 fix, when the planner was handed a curvature 4.8x too large.
- **bench2drive/constant_velocity/20260908T084921-804913** — Four independent CARLA workers; 4 completed route checkpoints reused. Duration covers this attempt. Superseded by pooled50-3mps-40c29d24c37d542a-02b6ac: Its 24 routes are already inside the 50-route run at the same 3 m/s target, so the same drives ranked twice.
- **neuroncap/drivejepa/20260905T220225-f11ac0** — Superseded by 20261110T204020-036fb3: Drive-JEPA's rows are its momentum-aware model now: 20261110T204020-036fb3 is the same 200-run protocol, camera-correct and with momentum-aware selection on. This run was camera-blind (before the camera-frame fix of 13 Oct 2026) and predates the selection stage, so momentum never acted in it.
- **neuroncap/visionpilot/20260905T121810-155d36** — resumed: 17 episode(s) reused from an earlier attempt, so started_utc and duration_s describe this attempt only Superseded by 20261111T204212-854d8b: Re-run on 11-12 Nov 2026 at the pinned e93eceaa with CURV_SCALE 0.21 (the September 2026 fix), default protocol, 14 scenes x 10 runs; this run used 1.0.
- **nuscenes/drivejepa/20260905T093335-4513d4** — Superseded by 20261105T201953-308dc5: Re-run on 5 Nov 2026 with the camera-frame fix of 13 Oct 2026; this run was camera-blind, so the model planned from ego status alone. Superseded by 20261111T061849-382fec: The row it pointed at, 20261105T201953-308dc5, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **nuscenes/visionpilot/20260905T093333-319ef5** — Superseded by 20261111T173725-c793ce: Re-run on 11 Nov 2026 at the pinned e93eceaa with CURV_SCALE 0.21 (the September 2026 fix) on the declared CPU provider; this run used 1.0.
- **navsim/alpamayo15/20260903T183459-753b33** — --limit 12: a subset of the benchmark, run to test the pipeline
- **navsim/visionpilot/20260902T175642-df5291** — Re-derived from the scorer CSV with the reviewing.md §4 parser (the run predates it; n was 12146). Superseded by 20261111T173726-ff827e: Re-run on 11 Nov 2026 at the pinned e93eceaa with CURV_SCALE 0.21 (the September 2026 fix) on the declared CPU provider; this run used 1.0.
- **navsim/drivejepa/20260902T173156-7e208f** — Re-derived from the scorer CSV with the reviewing.md §4 parser (the run predates it; n was 12146). Superseded by 20261111T052813-8f96dd: Drive-JEPA's rows are its momentum-aware model now: 20261111T052813-8f96dd hands each navtest frame the plan the node made for its log's frame 0.5 s earlier (streamed per log, 10,040 of 12,146 frames fed). Momentum never acted in this run: it predates the selection stage, and each open-loop sample reset the node.
- **navsim/alpamayo15/20260901T213227-d6117e** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. alpamayo15; this is the current headline for alpamayo15 until its full-navtest run lands, and it is a subset figure.
- **navsim/alpamayo15/20260901T212250-b5daf0** — EXPERIMENT, not a result: the 'distinct' arm of the same paired A/B. 0.709 EPDMS over the same 60 scenarios, +0.207 over 'grid' (better on 36, worse on 19, sign test p=0.03), almost all of it collisions avoided (no_at_fault_collisions 0.708 -> 0.992). This selection is now the default via `auto`; the full 242-scenario run supersedes this.
- **navsim/visionpilot/20260901T211745-df5291** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. visionpilot; the comparable figure is the full-navtest row 20260902T175642-df5291 (0.43122).
- **navsim/alpamayo15/20260901T211713-b5daf0** — EXPERIMENT, not a result: the 'grid' arm of a paired A/B on camera frame selection, run with PADDOCK_NAVSIM_ALLOW_SLOW while alpamayo was still gated. 0.503 EPDMS over 60 scenarios with the legacy 0.1 s-grid selection, which collapses to two distinct frames on 2 Hz data. Paired against the 'distinct' arm: see docs/benchmarks/navsim.md.
- **navsim/alpamayo15/20260901T205152-ff0af4** — SUPERSEDED. n=4 smoke read as a result it could not support: 0.226 here against a 0.257 baseline, which prompted gating alpamayo off NAVSIM entirely. A 60-scenario run put the same configuration at 0.503. The gate was wrong; see docs/benchmarks/navsim.md.
- **navsim/visionpilot/20260901T205133-be57d6** — SMOKE, not a result: a 4-scenario run proves the pipeline (node, conversion, scorer, CSV) and nothing about the model. Twice now the n=4 figure has been contradicted by a larger run -- visionpilot 0.640 -> 0.447 at n=242, alpamayo 0.226 -> 0.503 at n=60. NAVSIM's floor for a meaningful score is far above 4; use the full-split run.
- **navsim/openpilot/20260901T204522-8f4682** — openpilot infers ego motion from frame-to-frame change and NAVSIM supplies 2 Hz observations, so it was shown a near-stationary world: it planned 0.30 m of travel over 4 s while doing 4.8 m/s. The score measures the rate gap, not the model. model.yaml now declares min_input_hz: 10.0 and the harness refuses this pairing. See docs/benchmarks/navsim.md.
- **navsim/visionpilot/20260901T204347-be57d6** — SMOKE, not a result: a 4-scenario run proves the pipeline (node, conversion, scorer, CSV) and nothing about the model. Twice now the n=4 figure has been contradicted by a larger run -- visionpilot 0.640 -> 0.447 at n=242, alpamayo 0.226 -> 0.503 at n=60. NAVSIM's floor for a meaningful score is far above 4; use the full-split run.
- **navsim/openpilot/20260901T204239-8f4682** — openpilot infers ego motion from frame-to-frame change and NAVSIM supplies 2 Hz observations, so it was shown a near-stationary world: it planned 0.30 m of travel over 4 s while doing 4.8 m/s. The score measures the rate gap, not the model. model.yaml now declares min_input_hz: 10.0 and the harness refuses this pairing. See docs/benchmarks/navsim.md.
- **navsim/drivejepa/20260901T202524-7e208f** — 242-scenario subset (5 of navtest's 136 logs had sensor blobs), run under PADDOCK_NAVSIM_ALLOW_PARTIAL=1 before the runner recorded that switch; marked valid=false and config.allow_partial added retroactively on 2026-09-03 under the rule that nothing below the full split is ranked against a full run. drivejepa; the comparable figure for that model on the subset is the full-navtest row 20260902T173156-7e208f (0.84272).
- **kitscenes/openpilot/20260901T025750-fd824c** — Superseded by 20261008T004723-a369d1: a 3-sample check; the full 223-sample run replaces it, and docs/benchmarks/kitscenes.md still quotes this one.
- **kitscenes/openpilot/20260901T021743-fd824c** — Superseded by 20261008T004723-a369d1: a 3-sample check; the full 223-sample run replaces it, and docs/benchmarks/kitscenes.md still quotes this one.
- **neuroncap/openpilot/20260901T004005-88a4e0** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.
- **neuroncap/alpamayo15/20260901T002333-6dae70** — Superseded by 20260906T034647-bc409f: a small early run of 2 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one.
- **neuroncap/drivejepa/20260901T001634-44b6bb** — Superseded by 20261012T224707-f11ac0: a small early run of 2 episodes; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md still quotes this one. Superseded by 20261110T204020-036fb3: The row it pointed at, 20261012T224707-f11ac0, is retired too: Drive-JEPA's rows are its momentum-aware model now, so this one points at that run.
- **neuroncap/visionpilot/20260901T000037-c950bd** — Superseded by 20260905T121810-155d36: a single-episode check; the full 200-episode run replaces it, and docs/benchmarks/neuroncap.md and docs/ROADMAP.md still quote this one. Superseded by 20261111T204212-854d8b: 20260905T121810-155d36, which replaced it, is itself replaced by the post-CURV_SCALE re-run.
- **neuroncap/openpilot/20260831T235716-88a4e0** — openpilot closed-loop scores recorded before 12f3250 fixed the traffic_convention one-hot: the node passed [0, 1], telling supercombo it was a right-hand-drive car in a left-hand-drive city. Superseded; re-measure rather than compare. See docs/benchmarks/neuroncap.md.

---

Arrows show direction of better; a column with no arrow has none -- a signed bias, a ratio best at one, or a description of the model rather than a score. Runs are ranked only against runs of the same scenes; a benchmark with more than one scene set has one table for each. A number in brackets after a model marks a setting that changes what that run's score means. Aggregates are recomputed by paddock from per-sample records, never taken from the producer.
