# UniAD-B2D

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) at `498c1f7` |
| Adapter | [`adapters/models/uniad_b2d/`](../../src/paddock/adapters/models/uniad_b2d/) |
| Environment | conda, Python 3.9 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [hugsim](../benchmarks/hugsim.md), [nuscenes](../benchmarks/nuscenes.md), [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuplan](../benchmarks/nuplan.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is `[roots.repos]` from your `paddock.toml`,
> `<weights>` is `[roots.weights]`.

## What it is

UniAD-Base trained by the Bench2Drive authors on their CARLA dataset, released in
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo): the same architecture as [`uniad`](uniad.md),
with Bench2Drive's nine classes and six commands. Bench2Drive base is 1,000 clips driven by the Think2Drive
reinforcement-learning expert; the model trains on all but a validation split.

A separate model name rather than a variant of `uniad`: it is another checkpoint, trained on another dataset,
on another code base (Bench2DriveZoo vendors mmcv, mmdet and mmdet3d 0.17.1 as one `mmcv` package), with
another camera order and command set. A variant chosen by setting would have to be a site override of the
checkpoint, which paddock records as `valid: false`, and its results would sit under `uniad`'s name.

## Code and weights

`Thinklab-SJTU/Bench2DriveZoo` at `498c1f7` (branch `uniad/vad`). Two patches, both for building against
torch 2.x: `0001` imports `NumbaPerformanceWarning` from `numba.core.errors`, `0002` dispatches the rotated-NMS
CPU op on `scalar_type()`. **Licence: CC BY-NC-ND 4.0** (non-commercial, no derivatives), unlike UniAD's
Apache-2.0.

| file | source | sha256 |
|---|---|---|
| `<weights>/bench2drivezoo/uniad_base_b2d.pth` (996,840,308 bytes) | `huggingface.co/rethinklab/Bench2DriveZoo` | `8c6fb898033ef62e0f20a45ff68d554c131c0b58ca405d57bb5bfa4af1f8dcb0` |

The config is `adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py`. The README's results table links UniAD-Base to
`tiny_e2e_b2d.py` and UniAD-Tiny to `base_e2e_b2d.py`; the base checkpoint loads strictly into the base config
only. The motion anchors (`data/others/b2d_motion_anchor_infos_mode6.pkl`) are tracked in the repo.

**Environment**: UniAD's env, plus Bench2DriveZoo's ops built in place (`paddock setup --model uniad`, then
`paddock setup --model uniad_b2d`). The vendored `mmcv` is never installed: the node puts the checkout first on
`PYTHONPATH` and the policy refuses to load if another `mmcv` was imported.

## What it is fed

Bench2DriveZoo's agent (`team_code/uniad_b2d_agent.py`) is the reference: the same model build, the config's
`inference_only_pipeline` without its file loader, the same `results` keys. What differs from `uniad`:

| input | as Bench2DriveZoo |
|---|---|
| cameras | front, front-left, front-right, back, back-left, back-right (its order); resized to 1600×900; a JPEG quality-20 round trip in BGR, as the dataset was written (`tools/data_collect.py`: `cv2.imwrite` of CARLA's BGR frame) and read back |
| the lidar | always Bench2Drive's virtual one: 0.39 m behind the CARLA vehicle origin, 1.84 m up, x right, y forward. Paddock's Bench2Drive client uses the vehicle origin as its ego, so this is exact there |
| `can_bus` | its own layout: 0:3 translation (z = 0), 3:7 the full quaternion of the heading (Bench2DriveZoo wraps it in `list()`), 7:10 (speed, 0, 0), 10:13 acceleration with gravity, 13:16 rotation rate as -gyroscope (roll, -pitch, yaw rates in right-handed terms), 16:18 heading in [0, 2π) |
| `l2g_r_mat` | the lidar→global rotation itself, not its transpose, built from the heading alone |
| command | six: 0 left, 1 right, 2 straight, 3 lane follow, 4 change lane left, 5 change lane right (CARLA's road option minus one) |

**Command mapping.** Paddock's Bench2Drive client sends the road option of the route point nearest the ego,
with lane changes sent as STRAIGHT, and the next lane change separately. The adapter restores a lane change as
CHANGELANELEFT/RIGHT when it is within 4 m (the agent's `RoutePlanner` lookahead), unless the ego is turning.
On a source without a route (HUGSIM, nuScenes, KITScenes, nuPlan), STRAIGHT means "not turning", and it is sent
as LANEFOLLOW: Bench2Drive's STRAIGHT is the command for going straight through a junction, and following the
road is LANEFOLLOW in its training data. FOLLOW_REFERENCE is LANEFOLLOW everywhere.

**Where Bench2DriveZoo's agent and its training data disagree, the training data wins:**

- **Colour.** The agent converts CARLA's frames to RGB before the JPEG round trip and hands RGB to a model
  whose dataset was read as BGR, so red and blue are swapped. The adapter feeds BGR. On the golden frames the
  agent's RGB moves the 3 s waypoint by up to 2.3 m (`agent_rgb` in `golden_uniad_b2d.json`). This is an upstream
  agent bug, and Bench2DriveZoo's published closed-loop scores were driven with it.
- **Heading at canbus 16.** The agent writes the unwrapped heading (in (-3π/2, π/2]); the dataset wraps it to
  [0, 2π). For three quarters of all headings the two differ by 2π, and the CAN-bus MLP reads the number.

## Rate, rig, and the published scores

- **Rate.** It trained on Bench2Drive's 10 Hz logs sampled every 5th frame, so its temporal BEV fuses a grid
  0.5 s old: `planning_hz: 2`, as every model in paddock runs at its native rate. Bench2DriveZoo's agent asks it
  every 20 Hz tick, and its published Driving Score (UniAD-Base 45.81) was driven that way, R/B-swapped.
  Paddock's numbers are not that protocol and are not expected to match it.
- **Rig.** It trained on Bench2Drive's own camera rig (front camera 0.80 m ahead of the vehicle origin, all
  1600×900). Paddock's Bench2Drive runs use paddock's default rig (800×450, other mounts) for every model, so
  the records stay comparable; the adapter upsamples and uses the real calibration. `camera_rig =
  "bench2drive"` in `[benchmarks.bench2drive]` reproduces the training rig, and such a run is `valid: false`.

## Benchmarks

bench2drive, hugsim, nuscenes, kitscenes, nuplan and neuroncap run (real photographs are out of distribution
for a CARLA-trained model; on neuroncap the plan is held between 0.5 s steps); navsim and navsim_v1 need
`UNIAD_B2D_HORIZON_S = "4.0"`; comma2k19 (one camera) and alpasim (no rectified six-camera rig) do not. See
[uniad.md](uniad.md#benchmarks).

## Verification

Golden test (`tests/fixtures/golden_uniad_b2d.json`): Bench2DriveZoo's own path -- its dataset's
`get_data_info` maths for lidar2img, the CAN bus and l2g, its agent's pipeline call -- fed four nuScenes
keyframes as Bench2Drive-style records (CARLA's IMU simulated from the CAN bus). That proves the input path,
not that the model plans well on nuScenes. **0.000 m maximum difference**, exact after reset (9 Nov 2026). A
paddock-env test also checks that on paddock's `bench2drive` rig the adapter's `lidar2img` equals the matrices
Bench2DriveZoo's agents hard-code. Node: 0.44-0.45 s per `/infer` (0.74 s first).
