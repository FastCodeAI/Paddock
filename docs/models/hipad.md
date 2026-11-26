# HiP-AD

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [nullmax-vision/HiP-AD](https://github.com/nullmax-vision/HiP-AD) at `9610ed8` |
| Adapter | [`adapters/models/hipad/`](../../src/paddock/adapters/models/hipad/) |
| Environment | conda, Python 3.9 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [bench2drive](../benchmarks/bench2drive.md), [hugsim](../benchmarks/hugsim.md), [neuroncap](../benchmarks/neuroncap.md), [alpasim](../benchmarks/alpasim.md), [nuscenes](../benchmarks/nuscenes.md), [kitscenes](../benchmarks/kitscenes.md), [comma2k19](../benchmarks/comma2k19.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [navhard](../benchmarks/navhard.md), [nuplan](../benchmarks/nuplan.md) |

<!-- /summary -->

## What it is

HiP-AD ("Hierarchical and Multi-Granularity Planning with Deformable Attention for Autonomous Driving in
a Single Decoder", Tang et al., Nullmax, ICCV 2025; [arXiv 2503.08612](https://arxiv.org/abs/2503.08612))
is an end-to-end planner in the Sparse4D family. Six cameras (ResNet-50 and an FPN, at 640x352) feed one
decoder that updates agent, map and planning queries together, and the planning queries read image
features around their own waypoints (planning deformable attention) instead of attending over every
view. It plans several kinds of trajectory at once, six points each: by time at 5 Hz and at 2 Hz, by
distance every 2 m and every 5 m, and by time again with the mode matched to the car's speed.

The one released checkpoint is trained on Bench2Drive's Base set, in CARLA. Every other benchmark here is
real camera footage it has never seen.

## Code and weights

- **Code:** [nullmax-vision/HiP-AD](https://github.com/nullmax-vision/HiP-AD) (Apache-2.0), pinned in
  `repos.yaml`, unpatched.
- **Weights:** `hipad_stage2.pth` from the repo's GitHub release `weighs` (the README's Bench2Drive table,
  config `hipad_b2d_stage2.py`), into `{weights}/HiP-AD/`:

  ```bash
  curl -L -o hipad_stage2.pth \
    https://github.com/nullmax-vision/HiP-AD/releases/download/weighs/hipad_stage2.pth
  # sha256 7711b693293533463732d8a3efa8d5148d203344aad727a4661cb84263613956 (1,173,535,966 bytes)
  ```

- **Environment:** UniAD and VAD's (torch 2.7.1+cu128, mmcv-full 1.7.2, mmdet 2.28.2, built for the GPU;
  `uniad/setup.sh`), plus two things of HiP-AD's own, which `setup.sh` adds:
  - `flash-attn` 2.8.3, the prebuilt wheel for torch 2.7 / CUDA 12 / Python 3.9. The config's
    attention layers are `MultiheadFlashAttention`, so it is not optional. It runs on Blackwell (sm_120).
  - HiP-AD's deformable aggregation op (`projects/mmdet3d_plugin/ops`), compiled in place for the GPU
    (`TORCH_CUDA_ARCH_LIST`; 12.0 here). About a minute.

  The README's own pins (torch 1.13 + CUDA 11.7, mmcv-full 1.7.1) have no kernels for recent GPUs.
  `setup.sh` has not run end to end: the env the checks below ran in was a copy of an existing UniAD/VAD
  env, with the script's own steps run by hand in its order.

## How paddock drives it

HiP-AD's own CARLA agent (`bench2drive/leaderboard/team_code/hipad_b2d_agent.py`) is the reference for
every input. `policy.py` builds the model as it does (the config's `inference_only_pipeline` without its
file loader, fp16, the plugin imported from `plugin_dir`) and feeds it what it feeds:

- **Cameras:** the six in the agent's order, each JPEG-encoded at quality 20 and decoded again (as the
  agent does, to match the compressed training frames), BGR, at 1600x900. A source at another size is
  resized to it first, intrinsics with it.
- **The lidar frame:** Bench2Drive's LIDAR_TOP (x right, y forward; 0.39 m behind the ego origin, 1.84 m
  up), whatever lidar a source sends: the model trained in that mount only. The plan comes back in it and
  is moved to the ego origin (`bev_planners.lidar_plan_to_ego`).
- **The command:** one of Bench2Drive's six road options. Paddock's Bench2Drive client sends a lane change
  as STRAIGHT; a lane change within 4 m along the route (`nav_lane_change`) is read as the change.
- **The target point:** the next point of the route, chosen as the agent's route planner chooses it
  (`RoutePlanner(4.0, 50.0)`, replayed by `inputs.RouteTracker` on the route paddock sends each tick).
- **Time and pose:** seconds from the episode's start and the lidar's pose, for the temporal memory.

### What is paddock's, not HiP-AD's

- **The rate.** The agent runs the model every CARLA tick (20 Hz) with ten temporal memories, each
  stepped every 0.5 s (`with_close_loop`). Paddock runs HiP-AD's open-loop mode, one memory stepped every
  0.5 s (`planning_hz: 2`, the plan held between steps), as for UniAD and VAD. Each memory sees the same
  0.5 s steps either way; the plan is replaced every 0.5 s instead of every 0.05 s.
- **The plan.** `plan_speed_2hz`, six points 0.5 s apart (3 s), the 2 Hz twin of the `plan_speed_5hz`
  (1.2 s) that the agent's PID controller follows. Each closed-loop benchmark drives it with its own
  controller, not the agent's.
- **The target point without a route.** Only Bench2Drive and AlpaSim send a route. Everywhere else the
  target point comes from the command: 20 m ahead, turned 45 degrees towards a LEFT or RIGHT, or 3.5 m to
  the side for a lane change (`inputs.command_target_point`). HiP-AD never trained with such a point.
- **The camera rig on Bench2Drive.** Ranked runs use paddock's rig (`bench2drive/rig.py`), the same for
  every model: 800x450 and mounted where HiP-AD's training rig is not. The projections follow the real
  mounts. `camera_rig = "bench2drive"` drives HiP-AD's own rig, as a `valid: false` check.

## Benchmarks

`model.yaml` lists every benchmark that sends the six surround cameras, as for VAD. NAVSIM v1 and v2 and
navhard score 4 s, so they need the plan continued past its 3 s (`HIPAD_HORIZON_S = "4.0"` in a
`[models.hipad.benchmarks.<bench>.env]` pairing); its score note goes in with the first NAVSIM results.
comma2k19 sends one camera; the other five are black. AlpaSim re-renders its cameras as nuScenes' rig
(`alpasim.yaml`, VAD's profile) and sends its route.

## Results

### Bench2Drive

Not in this pull request.

### HUGSIM

Not in this pull request.

### neuroNCAP

Not in this pull request.

### AlpaSim

Not in this pull request.

### nuScenes

All 5,119 keyframes (`nuscenes/hipad/20261125T174612-255a76`), `valid: true`: **cumulative L2 3.108 m**,
L2 2.219 / 5.426 / 8.484 m at 1 / 2 / 3 s, collisions 5.46%. Last of the eleven models. Mostly speed: at 3 s,
7.22 m of error along the path and 1.34 m across it, and it plans 14.0 m in 3 s where the car moved under 5 m
(correlation 0.50 with the distance driven). Two keyframes failed once and were scored on `--resume`. The
target point comes from each keyframe's command, nuScenes sending no route.

### KITScenes

Not in this pull request.

### comma2k19

Not in this pull request.

### NAVSIM v1

Not in this pull request.

### NAVSIM v2

Not in this pull request.

### navhard

Not in this pull request.

### nuPlan

Not in this pull request.

## Verification

- **Against HiP-AD's own agent** (`tests/agent_reference.py`, `make check-hipad`): the agent's code at the
  pinned sha, with CARLA stubbed and switched to one temporal memory (HiP-AD's open-loop mode, which the
  adapter runs), and the adapter, fed the same six frames, 0.5 s apart, of six real 1600x900 cameras on
  Bench2Drive's rig, the adapter through the calibration paddock's Bench2Drive client computes. On every
  frame, every input equal: the camera projections, the image tensor, the command and the timestamp
  exactly, the target point and the pose to 1e-14. On every frame the plan equal to within 2.5 mm, so the
  temporal memory is fed what HiP-AD's own code feeds it. Plans are not bit-equal: fp16 and flash-attention
  are not bit-reproducible, and the memory carries the rounding on (one run differed by up to 4 cm; two
  runs of one model on one KITScenes set by up to 0.17 m). Run 26 Nov 2026. **verified**
- **Paddock's own env** (`tests/test_policy.py`, in `make check`): the command and lane-change mapping,
  the route planner's choice of target point, the stand-in target points, the axes, the image pipeline's
  numbers, and the policy contract on a stub model.
- **Each benchmark:** `paddock run --dry-run` (the pre-run checks) passes for every benchmark in
  `model.yaml` except NAVSIM v2, whose metric cache this site's account cannot read, and nuPlan, whose
  `mini` split is not on this site. nuScenes and KITScenes were driven end to end on 5 samples each.
