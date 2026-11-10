# UniAD

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD) at `532fc33` |
| Adapter | [`adapters/models/uniad/`](../../src/paddock/adapters/models/uniad/) |
| Environment | conda, Python 3.9 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [hugsim](../benchmarks/hugsim.md), [nuscenes](../benchmarks/nuscenes.md), [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuplan](../benchmarks/nuplan.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [alpasim](../benchmarks/alpasim.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is `[roots.repos]` from your `paddock.toml`,
> `<weights>` is `[roots.weights]`.

## What it is

[UniAD](https://github.com/OpenDriveLab/UniAD) (OpenDriveLab, CVPR 2023 best paper) is a planning-oriented
end-to-end stack: six cameras are lifted into one BEVFormer grid (200 × 200 cells of 0.512 m, ResNet-101 with
deformable convolutions), and tracking, online mapping, motion forecasting, occupancy prediction and planning
are stacked on it, each head reading the queries of the ones before. The planner reads the ego track query,
the ego motion queries and an embedding of the command (right, left, forward), plans six waypoints 0.5 s apart
(3 s), and a small nonlinear optimiser pushes them off predicted occupancy.

Paddock runs the released UniAD-Base stage-2 checkpoint, trained on nuScenes `train`, on the upstream v1
code. Its Bench2Drive-trained sibling is a separate model, [`uniad_b2d`](uniad_b2d.md), and VAD is
[`vad`](vad.md); all four share one adapter core, `adapters/models/uniad/bev_planners.py`, and one environment.

## Code, weights and environment

**Code.** `OpenDriveLab/UniAD` at `532fc33`, the `main` branch: UniAD v1, the code the checkpoint was trained
with. The repo's default branch is now `v2.0`, a rewrite on newer OpenMMLab; do not clone that one. One patch,
`patches/UniAD/0001-mmdet-dice-registry.patch`: mmdet 2.28 registers its own `DiceCost` and `DiceLoss`, so
the plugin's must be registered with `force=True` (UniAD 2.0 did the same).

**Weights**, under `<weights>/uniad/`:

| file | source | sha256 |
|---|---|---|
| `uniad_base_e2e.pth` (996,345,538 bytes) | `github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth` | `0ad0c2f5dc9788a41c313305779ea49346aeb742d1f6bb5ad25c46f9beffc990` |
| `motion_anchor_infos_mode6.pkl` | the UniAD v1.0 release assets (the repo does not track it) | `45761de4a965c590318987f43af7e7f50b8e043f9a2832a91246f4fd39bb57bb` |

The config names the anchors relative to the working directory; `UNIAD_MOTION_ANCHORS` in model.yaml points
the motion head at the weights folder instead.

**Environment.** One conda env (Python 3.9) serves `uniad`, `vad`, `uniad_b2d` and `vad_b2d`:
torch 2.7.1+cu128, and the OpenMMLab 1.x stack compiled from source for the machine's GPU: mmcv-full 1.7.2,
mmdet 2.28.2, mmsegmentation 0.29.1, mmdet3d 0.17.1. `adapters/models/uniad/setup.sh` builds it at
`PADDOCK_VENV` (`paddock setup --model uniad`), then `paddock setup --model uniad_b2d` builds
Bench2DriveZoo's own ops. Points to know:

- The compiled ops are built for one GPU architecture, `TORCH_CUDA_ARCH_LIST` (default `12.0`, Blackwell).
  The repos' own pins (torch 1.9-1.12, mmcv-full 1.4) have no kernels for recent GPUs.
- mmdet3d 0.17.1's PyPI tarball ships no CUDA sources for its point-cloud ops. `env/mmdet3d-0.17.1.patch`
  builds without them and makes each import a stub that raises if called; no camera-only model calls them.
- The node server's own dependencies (fastapi, uvicorn) are installed into the env, as for every model.
- A site that keeps the env elsewhere sets `[models.uniad] venv` (and the same for the other three).

The script records the build that produced the environment the golden tests passed in; it has not itself been
run end to end on a clean machine.

## What the model is fed

Every input is built as UniAD's own data path builds it (`NuScenesE2EDataset.get_data_info`, the config's
`test_pipeline`, `tools/test.py`), from what a paddock Observation carries:

| input | from | as upstream |
|---|---|---|
| images | the six cameras, in the info files' order (front, front-right, front-left, back, back-left, back-right) | resized to 1600×900 if they arrive at another size (intrinsics scaled with them); fed **BGR** float32, as `mmcv.imread` reads them; the config's own `NormalizeMultiviewImage` (caffe mean 103.530, 116.280, 123.675; std 1; no channel swap) and padding to 1600×928 |
| `lidar2img` | each camera's `camera2image` and `camera2ego`, and the model's lidar | `K @ inv(camera2ego) @ lidar2ego`, the dataset's `viewpad @ lidar2cam` |
| the lidar | the source's `lidar2ego` if it is a LIDAR_TOP (x right, y forward, z up), else nuScenes' LIDAR_TOP as calibrated on the log of scene-0103 | the grid and the plan live in this frame |
| `can_bus` (18) | translation and heading from `ego2world`; acceleration, rotation rate and velocity from paddock's canbus | 0:3 ego translation; 3:7 **w of the ego quaternion, four times** (below); 7:16 accel, rotation rate, velocity; 16:18 heading in [0, 2π), radians then degrees |
| gravity | | nuScenes' CAN-bus accel reads z ≈ +9.6-9.8 (specific force); a source that sends z ≈ 0 gets 9.81 added |
| `l2g_r_mat`, `l2g_t`, `timestamp` | `ego2world` and the lidar; the observation's time | the transpose of the lidar→global rotation, float32, as the dataset stores it; seconds |
| command | paddock's | 0 right, 1 left, 2 forward (UniAD's own order, from `get_sdc_planning_label`); FOLLOW_REFERENCE is forward |
| `use_can_bus` | | `True`, as configured |

**Three upstream behaviours are reproduced, not fixed**, because the checkpoint trained with them:

- `can_bus[3:7] = Quaternion(...)` in the dataset assigns a pyquaternion object into a numpy slice. It has
  `__float__` (its w) and no `__len__`, so numpy broadcasts w into all four slots. The sign of w is the stored
  quaternion's, which nuScenes keeps either way (57 % of its ego poses have w < 0). A source that puts the ego
  pose's own quaternion at canbus 3:7 keeps its sign; otherwise the adapter takes w ≥ 0 on an episode's first
  frame and holds the sign continuous. On scene-0103 the two signs move the 3 s waypoint by up to 0.50 m.
- On the first frame after a reset, UniAD's `forward_test` leaves the absolute position in `can_bus[:3]`
  (its first-frame test runs after the scene token is already set), so the first plan reads it.
- The evaluation-only ground truth `forward_test` dereferences (lane masks for map IoU, occupancy labels) is
  given as zero stubs. It reaches only the metrics, which are discarded, and the shape of the all-zero
  occupancy forecast when nothing is tracked, as real labels would.

**Temporal state.** The previous BEV grid, the track queries and the ego pose the CAN-bus delta is taken from
are kept across calls and returned to a freshly built model's values on `/reset`. The model trained on 0.5 s
steps, so it is asked at 2 Hz (`planning_hz: 2`); an open-loop sequence is fed on the same grid, one frame per
0.5 s back from the scored one (`history_on_grid`), so a 10 Hz history does not show it a world moving at a
fifth of its speed. A client that asks sooner than 0.5 s after the last step (neuroNCAP, every 0.1 s) gets the
last plan held fixed in the world and re-sampled at its time and in its ego frame (`held_plan`), the way the
HUGSIM and Bench2Drive clients hold a plan between replans; the model steps again once 0.5 s have passed.

**Horizon.** The plan is 3 s. `UNIAD_HORIZON_S` (unset by default) continues it by holding the last 0.5 s step,
for NAVSIM, which scores 4 s. It is an extrapolation the model never made, so a run that sets it is stored
`valid: false` until model.yaml declares the score note its comment spells out, which goes in with the first
stored NAVSIM run.

## The plan

UniAD's planner is trained on the future position of the **lidar's origin**, in the current lidar frame
(`get_sdc_planning_label`, `projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py:225-281`: a box at
the lidar origin of each future sample, carried lidar → ego → global → current ego → current lidar). Paddock's
trajectory is the ego origin's. So each waypoint is placed in the ego frame and the lidar's lever arm
(0.94-0.99 m forward on nuScenes) is taken off, turned by the heading the car will have then, which comes from
the path. On a straight road that is just the axis change; through a turn the arm swings with the car.
UniAD's other ego label, `get_sdc_traj_label`, uses the rear axle, but it trains the motion head, not the
planner.

## HUGSIM's client is not used

HUGSIM ships its own UniAD client (`hyzhou404/UniAD_SIM`). Paddock does not use it, because it changes both the
inputs and the model:

- it normalises HUGSIM's RGB renders with the caffe mean meant for BGR, so red and blue are swapped;
- it sets `use_can_bus=False`, and builds the CAN bus with zero acceleration;
- its `forward_test` decides "first frame" from `prev_frame_info['prev_bev']`, which UniAD never sets, so the
  ego-motion delta is zero on every frame and the previous grid is fused unaligned;
- its motion head keeps every class's queries instead of the vehicles' (`vehicle_mask = ones`), and it skips
  the collision optimiser when nothing is tracked.

An A/B on 9 Nov 2026 against that client, driven by HUGSIM over the same three scenarios, gave the same
outcomes as this adapter: nuScenes scene-0013 easy, HD-Score 1.00 for the client and 0.99 for the adapter;
scene-0013 hard, 0.007 and 0.0; KITTI-360 10320 medium, 0.03 and 0.0. HUGSIM renders the nuScenes rig on
every dataset, black behind the car on KITTI-360 and Waymo, and the KITTI-360 renders are out of distribution
for a nuScenes-trained model; UniAD's raw plan there zigzags sideways by up to 2 m between waypoints.

## Benchmarks

The model needs all six surround cameras (one learned embedding per camera slot) and plans 3 s.

| benchmark | runs | why |
|---|---|---|
| [nuscenes](../benchmarks/nuscenes.md) | yes | its own data; see comparison.md, *UniAD × nuscenes* |
| [hugsim](../benchmarks/hugsim.md) | yes | asked every 2nd 0.25 s step; KITTI-360 and Waymo scenes have the three back cameras black |
| [bench2drive](../benchmarks/bench2drive.md) | yes | 800×450 renders, upsampled; lidar2ego is the identity, so the nominal LIDAR_TOP is used |
| [kitscenes](../benchmarks/kitscenes.md) | yes | the ring maps onto the six names; the 10 Hz history is fed every 0.5 s; canbus carries speed only |
| [nuplan](../benchmarks/nuplan.md) | yes | six of its eight cameras (F0, L0, R0, B0, L2, R2; the side views L1 and R1 have no nuScenes slot); command always STRAIGHT; 10 Hz history fed every 0.5 s |
| [neuroncap](../benchmarks/neuroncap.md) | yes | NeuRAD renders the six cameras and sends the real LIDAR_TOP. It asks every 0.1 s whatever `planning_hz` says, so the model steps every 0.5 s and the plan is held in between (below) |
| [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md) | with `UNIAD_HORIZON_S = "4.0"` | NAVSIM scores 4 s and refuses a 3 s plan; the setting continues the last 0.5 s step (METEOR's rule), set in a `[models.uniad.benchmarks.navsim.env]` pairing. Its cameras map as nuPlan's; history is four frames at 2 Hz. Until model.yaml declares the setting's score note (its comment says how), such a run is stored `valid: false` |
| comma2k19 | no | one camera; the other five slots would have to be invented |
| [alpasim](../benchmarks/alpasim.md) | yes (not yet run) | its six cameras re-rendered as nuScenes' rig: 1600×900 undistorted pinholes at scene-0103's calibrated intrinsics and poses, CAM_BACK on the telephoto's id (`paddock_bridge_nuscenes_rig.yaml`), rendered every 0.5 s on the replan grid. Asked every 0.5 s, the 3.0 s warm-up sent to `/observe`; canbus carries ego-frame speed and the simulator's acceleration. No route: the command is always FORWARD |

## Verification

- **Golden test** (`make check-uniad-vad UNIAD_VAD_DATA=<dir>`; `adapters/models/uniad/tests/test_golden.py`).
  `golden_reference.py` runs UniAD's own data path on four consecutive nuScenes `val` keyframes of
  scene-0103 (the converter's `obtain_sensor2top` and `_get_can_bus_info`, the dataset's `get_data_info`, the
  config's pipeline with its own image loader, `prepare_test_data`, collate, `forward_test`) and stores what the
  model was fed and planned in `tests/fixtures/golden_uniad.json`. The adapter, launched through model.yaml and
  the site config, then plans from paddock Observations of the same frames. On 9 Nov 2026 (RTX PRO 6000,
  torch 2.7.1): **0.000 m maximum difference** on all four frames, for the stored quaternion and the
  canonical one, and a reset reproduced the sequence exactly. The fixture also records, untested, what the
  original path plans with each camera placed at its own ego pose instead of the lidar's (one ego pose per
  frame is all an Observation carries): up to 0.23 m.
- **Paddock-env tests** (`tests/test_bev_planners.py`, in `make check`): the inputs built from the fixture's
  Observations equal those the original path fed (lidar2img, can_bus, l2g, timestamp, command); the
  config's normalisation and `use_can_bus` where the checkout is present; the frame of the plan; the history
  grid; the policy contract.
- **Node**: started with paddock's launcher and asked over HTTP with HUGSIM-shaped requests (800×450 pixels,
  identity lidar2ego, speed / acceleration / yaw rate only): 0.47-0.49 s per `/infer` on an RTX PRO 6000,
  0.89 s for the first; about 4.1 GB of GPU memory for the process.

## Cost of a step, and the threads that prepare frames

One model step on nuScenes (six 1600×900 JPEGs from local NVMe, scene-0103, 9 Nov 2026, the card shared with
other jobs), stage by stage, before preparing went onto threads; medians over ten keyframes, ms:

| read + decode (Pillow) | BGR + float32 | the rest of `_results` | test pipeline (normalise, pad, format) | collate | to GPU | forward | plan out |
|---|---|---|---|---|---|---|---|
| 25 | 31 | 54 | 162 | 31 | 17 | 1,404 | 0.2 |

The forward is 0.29 s on an idle card. During it the node's main thread is at 100% CPU, and that is not
Python work: in a torch profile of four steps, 4.40 s of the thread's 5.07 s of CPU time is inside
`cudaStreamSynchronize`, about 3,900 calls a step. UniAD's tracker (`RuntimeTrackerBase.update`) loops over
its 900-odd track queries in Python and tests each one's tensors, every test a host-device sync, and the CUDA
runtime spin-waits on each, longer when the card is shared. That is the model's own code and is left as it is.

What paddock does around the forward is CPU work, and `UNIAD_VAD_THREADS` (model.yaml, default 5; declared
`tuning_env`) spreads it out: one thread prepares the next frame -- its images, the config's test pipeline,
collate -- while the model runs on this one, the others decode and convert the six cameras side by side, and
OpenCV's own pool is held to as many (it otherwise takes all 32 cores of this box for VAD's resize). Frames
are prepared strictly in order, one ahead, so each gets exactly the inputs it would get one at a time: the
four golden tests stay at 0.000 m, and an A/B against the previous code on twelve keyframes gave identical
plans (0.0 m) for single calls, windows and in-order passes. Per call, old → new, two rounds on the shared
card: `/infer` (reset + one keyframe) 1,544 / 1,863 → 1,167 / 1,031 ms; a five-keyframe window (nuScenes'
default) 8.19 / 8.64 → 5.06 / 5.11 s; a keyframe of an in-order pass 1.76 / 1.63 → 0.96 / 0.96 s.
`UNIAD_VAD_THREADS=1` prepares on the model's thread, one camera after another, as before.

Not verified: plan quality on any benchmark (no sweep has been scored here); `setup.sh` on a clean machine;
any GPU but sm_120.
