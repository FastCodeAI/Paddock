# AutoE2E v1.0 in paddock

AutoE2E is the Autoware Foundation's open end-to-end planner. Release v1.0
(Hugging Face `AutowareFoundation/auto_e2e`, published 8 Oct 2026) ships two
checkpoints of one architecture, `bevformer_v2_t8_split_navigation_v5`:
`nuplan-epoch-5` and `kitscenes-epoch-5`, the second being the first fine-tuned
on KITScenes with the camera BEV frozen. Paddock runs **`kitscenes-epoch-5`**,
imported from upstream's own code at the release revision. Nothing of the model
is re-implemented; the adapter packs inputs and integrates the output.

Vendor repo: `auto_e2e` in `repos.yaml`, pinned to the release revision
`40a75cf8` (branch `experimental/reactive-bev-learning`, the `source.revision`
in the release's `config.json`). Adapter: `src/paddock/adapters/models/autoe2e/`.

```bash
PADDOCK_REPO=<repos>/auto_e2e PADDOCK_WEIGHTS=<weights> bash src/paddock/adapters/models/autoe2e/setup.sh
paddock doctor --model autoe2e
paddock run --model autoe2e --bench navsim          # needs [benchmarks.navsim] nuplan + keyframe_cameras
paddock run --model autoe2e --bench kitscenes       # the map and the oracle route are on by default
paddock run --model autoe2e --bench comma2k19       # the road camera alone; the other five views left out
paddock run --model autoe2e --bench nuscenes        # 6.5 s of past keyframes, the history it declares
PADDOCK_INFER_HZ=10 paddock run --model autoe2e --bench bench2drive --scene <route>
```

PR #19 is a different adapter: KITScenes only, written before any weights were
public, for a seven-view SwinV2 architecture this release does not have.

## What the model consumes and emits

From the release `config.json` and upstream's KITScenes evaluation path
(`Model/data_parsing/pre_extracted.py`, `kit_scenes/*.py`):

| | |
|---|---|
| cameras | six, in slot order front, front-left, front-right, rear-left, rear, rear-right (nuPlan F0, L0, R0, L2, B0, R2; KITScenes `base_front_center` + ring) |
| camera history | eight frames 0.5 s apart, t−3.5 … t, oldest first; each with a projection from the *current* reference frame into that frame's cameras |
| image size | 512 × 512 per view (squashed, PIL bilinear, JPEG q90 round trip); the current front also at 1024 × 1024 |
| ego history | 64 rows × (speed, acceleration, yaw rate, curvature) at 10 Hz, t−6.4 … t−0.1 s |
| navigation | optional 14-channel semantic map and 2-channel route raster, 450 × 300 at 0.4 m, rows front to rear, columns left to right |
| output | 64 steps × (acceleration, curvature) at 10 Hz, 6.4 s |
| size | 79.9 M parameters (`parameter_count`), 0.33 GB checkpoint |

## How paddock fills it

- **Views and frames.** Every view the benchmark has is required, and
  `CAM_FRONT` always is; `infer` raises if the current observation lacks one.
  A view the benchmark does not have at all (no image, path or calibration,
  as on comma2k19's single road camera) is left out: a zero tile with a
  projection that puts every point behind the camera, so upstream's depth
  test keeps it out of every BEV cell. Each plan's `extras` say how many
  views it used (`views`). The seven history frames are the held frames
  nearest t−3.5 … t−0.5 s within half the frame spacing, never less than
  0.06 s (0.25 s on nuScenes' 2 Hz keyframes, whose timing jitters 0.45–0.55 s;
  0.06 s on every 10 Hz or faster stream), that carry every view the current
  one does. Where the past does not reach that far (a route or log start)
  the oldest such frame is repeated with its own pose; each plan's `extras`
  and `diagnostics()`'s last call say how many were (`history_repeated`).
- **Reference frame.** Projections start from paddock's ego frame raised by
  `AUTOE2E_REF_HEIGHT_M` (default 2.1 m): KITScenes' top-LiDAR frame, whose
  ground sits at z = −2.1 m (`kit_scenes/dataset.py`), is what the fine-tune
  saw. Set 0 for the ground-level frame nuPlan used.
- **Ego history** is differenced from the pose stream as `kit_scenes/egomotion.py`
  does (centred `np.gradient` on the real timestamps, curvature floor 0.1 m/s),
  newest acceleration zeroed (the KITScenes stage's causal mask). It is not read
  from `canbus`, which most harnesses fill only partly. Rows before the first
  pose depend on who sends the past:
  - **Open loop** (NAVSIM, KITScenes: the real past, via `infer_sequence`): left
    at zero, upstream's left padding. Upstream's own KITScenes benchmark pads up
    to 24 of the 64 rows this way.
  - **Closed loop** (neuroNCAP, Bench2Drive: one frame per call, no past):
    steady driving, the first measured speed with no acceleration and no turn
    (the simulator's speed on the first call, when nothing is measured yet).
    Training never saw a padded row: KITScenes samples start 64 rows into a
    scene. Left at zero, an episode carries 32–64 zero rows throughout, which
    read as a car that stood still a moment ago. On neuroNCAP 0103 stationary
    that turned the car right within 2 s, into a parked car (score 0.14). With
    the past filled it drives straight for the first 2 s, then turns left, and
    the turn grows each call until the car spins: all three 0103 scenarios,
    seed 0, scored 5.00 / 0.59 / 5.00, and both 5.00s are misses, not
    avoidance. Two other fills, braking ramped in over the last filled second
    and the recorded nuScenes past from before the episode, loop the same way,
    so the loop is not this input. AutoE2E's neuroNCAP scores are not
    meaningful until it is understood.
- **Navigation, where the benchmark has a map (KITScenes).** The observation
  names the scene's Lanelet2 map (`obs.map_files`) and, under KITScenes'
  default `route = "oracle"`, carries the logged drive (`obs.route_trace`).
  The adapter draws both with upstream's own code, the calls
  `kit_scenes/navigation.py` makes when it packs a sample: the map extracted by
  `navigation/lanelet2_adapter.py`, the route matched to the drive by
  `navigation/lanelet2_matcher.py` (whose quality gates decide `route_valid`),
  both rendered by the native
  `navigation/rasterizer.py`. That is upstream's KITScenes validation input,
  with which its published numbers were made. The map is read from a scratch
  copy, because upstream writes a sanitised `map.osm` beside the one it reads
  for scenes with degenerate lanelets. One difference, kept on purpose:
  upstream renders at the latest pose on a 0.5 s grid and warps that raster to
  the sample, while the adapter renders at the sample's own pose. On the
  KITScenes keyframes these are the same pose, except for 31 of the 223 where
  upstream's truncation of the log clock to whole nanoseconds puts its grid
  pose 0.5 s back. KITScenes' `map = false` or `route = "none"` withholds
  either; `AUTOE2E_ROUTE=0` refuses any route.
- **Navigation elsewhere.** No semantic map: zeros with `map_valid` false,
  upstream's own no-map track. The route raster is drawn from `obs.route_xy`
  when the benchmark sends one: a corridor 1.75 m either side, clipped 10 m
  behind, and a 2 m destination disk where the polyline stops short of 200 m.
  Bench2Drive sends the leaderboard's route; NAVSIM has none, so its scenes run
  camera + ego only. That track is upstream's official KITScenes *test* input
  (`camera_only_missing_map_route`); training and validation always had map and
  route (`distributed_training/reactive_data.py` at `40a75cf8`), so NAVSIM shows
  this checkpoint an input it was evaluated on but never trained on.
- **Output.** The 64 controls are integrated with upstream's rollout
  (`training/losses/control_rollout.py`: dt 0.1, speed clamped at zero first,
  heading from the new speed), starting from the history's t−0.1 s speed, and
  sampled every 0.5 s: 12 waypoints to 6.0 s with heading. Non-finite controls
  raise, so the scenario fails visibly instead of being scored on a substitute plan.

| env | default | meaning |
|---|---|---|
| `AUTOE2E_CHECKPOINT` | `{weights}/auto_e2e/models/kitscenes-epoch-5/checkpoint.pt` | the checkpoint; `load()` refuses a missing file |
| `AUTOE2E_REPO` | `{repo}` | the upstream checkout (`Model/` is imported) |
| `AUTOE2E_REF_HEIGHT_M` | `2.1` | projection reference height, above |
| `AUTOE2E_ROUTE` | `1` | `0`: never draw the route raster |
| `AUTOE2E_REPLAN_S` | `0` | seconds between forward passes; the plan is held and re-expressed in between. `0` = every call |
| `AUTOE2E_BF16` | `1` | bf16 autocast on CUDA |
| `AUTOE2E_RECORD_DIR` | unset | append one JSON line per forward pass (controls, rollout, ego rows) |
| `AUTOE2E_CACHE_CONSTS`, `AUTOE2E_VISIBLE_ONLY` | `1` | the two speed patches below; `0` runs upstream's code |

## Speed: two patches, both bit-identical

Upstream's `bevformer_v2_t1.MultiScaleSpatialCrossAttention.forward` builds a
`torch.tensor([w, h], device=cuda)` inside its view × query-chunk × level loop:
17,280 blocking host-to-device copies a forward, 22–33 s per forward on a
shared GPU. The adapter gives that module a `torch` whose `tensor()` returns a
cached device constant for the same values, and computes each view's cross
attention only for the BEV queries the view sees (upstream computes all 60,000
and multiplies the unseen ones by zero; every op on the way is per query).
Both were checked with `torch.equal` against upstream's output on 9 Oct;
~2.5 s per forward afterwards, 96 % of it the eight BEV encodings.

## Environment

uv, Python 3.12, `torch 2.7.1+cu128` from the PyTorch index in its own step:
installing `timm` alongside re-resolves torch from PyPI, whose cu126 build has
no sm_120 (Blackwell) kernels. `setup.sh` refuses a checkout not at `40a75cf8`
and checks the downloaded files against the release's `SHA256SUMS`. ~3 GB of
VRAM per node.

The map needs three more things, installed as upstream's own data-prep image
installs them (`Platform/docker/data-prep/Dockerfile`):

- **The KITScenes SDK** (`github.com/KIT-MRT/kitscenes`), at the revision
  upstream pins (`KITSCENES_SDK_REVISION` in
  `Model/data_parsing/kit_scenes/source.py`, `7765cdec`). Its Lanelet2 1.2.2
  wheel is vendored in the SDK under git LFS, so `git-lfs` must be installed.
  The SDK is installed with pip from inside its clone, because its metadata
  names that wheel by a path relative to the clone, which uv refuses to parse.
- **NumPy 1.26.4.** The SDK requires NumPy < 2, and pyarrow is pinned to
  25.0.1, the last release that accepts it. The model runs on it unchanged,
  and the KITScenes runs below were made on it. The NAVSIM row and the paused
  Bench2Drive routes were not: they were made on NumPy 2.2.6, `setup.sh`'s pin
  before the map, and have not been re-run on 1.26.4.
- **Upstream's C++ rasterizer**, built by `Model/navigation/native/build.py`
  beside its source, where upstream's loader looks for it. It needs a C++17
  compiler.

## Training exposure

Declared in `model.yaml` as nuPlan train and KITScenes train. The nuPlan stage
(`nuplan_full`, from the official BEVFormer V2 R50 T8 initialisation, itself
nuScenes-trained) names no split in the release; that it held out nuPlan test
is inferred from the release's own nuPlan-test figures, not stated. NAVSIM's
`navtest` is drawn from nuPlan's test logs, so the NAVSIM number below is on
the rig and cities AutoE2E was pre-trained on. See `comparison.md`,
*AutoE2E × navsim*.

## On AlpaSim, through the bridge

**Built, not yet run.** The AlpaSim harness gives `autoe2e` its own driver
profile (`harness.DRIVER_PROFILES`, `configs/driver/paddock_bridge_autoe2e.yaml`);
the model's adapter is unchanged apart from listing `alpasim` in `model.yaml`.
It is camera-only, as on NAVSIM: the bridge sends no route and the node sends
no map.

**Cameras.** AlpaSim's 26.01 scenes record six f-theta cameras, none looking
straight back. The profile re-renders them as pinholes, every distortion
coefficient zero, at the KITScenes calibration's field of view (the release's
`datasets/kitscenes-val/v3.5/rig/*.json` with the image sizes KITScenes ships,
5856 × 3104 front and 3504 × 2272 ring):

| Slot | Node name | AlpaSim id rendered | Pose | Size, f | Field of view |
|---|---|---|---|---|---|
| front | CAM_FRONT | camera_front_wide_120fov | each scene's own (yaw ≈ 0°) | 1920 × 1018, 990.8 px | 88.2° × 54.4° |
| front-left | CAM_FRONT_LEFT | camera_cross_left_120fov | each scene's own (≈ 67°) | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| front-right | CAM_FRONT_RIGHT | camera_cross_right_120fov | each scene's own (≈ −67°) | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear-left | CAM_BACK_LEFT | camera_rear_left_70fov | fixed (2.10, 1.04, 1.02) m, 120° | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear | CAM_BACK | camera_front_tele_30fov | fixed (2.09, 0, 1.02) m, 180° | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear-right | CAM_BACK_RIGHT | camera_rear_right_70fov | fixed (2.06, −1.01, 1.02) m, −120° | 1024 × 664, 538.9 px | 87.1° × 63.3° |

AlpaSim can change a scene's camera but not add one, so the rear view takes
the tele camera's id. One override applies to every scene, so a camera keeps
its own vehicle's mounting only when the override leaves its pose out; the
three new directions take one pose, at the recorded rear cameras' median
mounting point (0.01–0.41 m from any one scene's).

**Rhythm.** The bridge asks the node every 100 ms, which is the model's 10 Hz,
and answers the 20 Hz ticks between with the last plan, re-timed onto now and
re-anchored to the car (`replan_rhythm.py`). Warm-up frames go to `/observe`,
since AlpaSim drives the recorded path during the warm-up. The node's own
`AUTOE2E_REPLAN_S` is not used: setting it for one benchmark needs a site env
override, which marks the run `valid: false`, and it re-anchors the held plan
without re-timing it.

**Motion history.** AlpaSim starts each scene 0.13–0.23 s into its clip and
hands over 3.0 s later, so the node holds about 3 s of poses where the model
reads 6.4 s. With `--pad-ego-history` the bridge first sends ego-only frames
straight back from the first frame's pose at the speed measured from the last
two poses. The node then reads 34 of the 64 rows at the first plan as that
speed with no acceleration, yaw rate or curvature. At the join (the first real
frame) it reads the true speed, half the true yaw rate and three quarters of
the true acceleration; every later row is as before
(`alpasim/tests/test_replan_rhythm.py`, through this node's own history code).
Image history is unchanged: the −3.5 s frame does not exist at handover, and the
oldest real frame is repeated.

**Known gaps**, none of them measured yet:

| Gap | Why it may matter |
|---|---|
| Plan offset 0 m | KITScenes trained the motion of the roof LiDAR point, nuPlan that of the rear axle. AlpaSim drives the plan as the rear axle's. The LiDAR's position on the KITScenes car is not published (the KITScenes dataset's own calibration is gated), so no offset is applied. In a turn of Δθ the two paths differ by 2·d·sin(Δθ/2) for an offset d along the car |
| Rear views untested | The 120°, 180° and −120° renders point 28–32° away from every camera the scene was reconstructed from |
| Cameras lower than training | AlpaSim's cameras sit 0.8–1.6 m above the ground; KITScenes' about 1.7–1.9 m, assuming its LiDAR at the 2.1 m this adapter uses |
| Front-left/right at ±67° | KITScenes' are at ±60°; the recorded pose is kept rather than one fixed pose for every vehicle |
| No correctness check | the release publishes aggregate evaluation reports only, no per-sample plans to compare against |

Cost, measured on a shared GPU on 21 Oct 2026: about 3.1 s per forward pass
with full history, so about 9 minutes of model time per scene after handover
at 10 Hz.

## Results

**NAVSIM `navtest`, 12,146 scenarios, valid** (`results/navsim/autoe2e/20261009T055926-3ceb18`).
10 Hz dense path from nuPlan with `keyframe_cameras` on: ego history from the
logs at 10 Hz, rear views from OpenScene's keyframes, which are exactly the
0.5 s frames the camera history reads. No route.

| EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.5002** | 0.824 | 0.777 | 0.865 | 0.990 | 0.491 | 0.814 | 0.830 | 0.984 | 0.812 |

Between the constant-velocity floor (0.2385) and Alpamayo 1.5 at 2 Hz
(0.6081). Where the score goes: plans lean left (on straight roads the plan
ends a median 0.41 m left of the human's, growing with speed), right turns are
under-turned, and starts from a standstill are timid (2.3 m planned against
8.4 m driven). Comfort (0.984 history, 0.812 extended) is its strongest column.

**KITScenes, the shared 223 keyframes, valid** (`results/kitscenes/autoe2e/`):
**0.847** ADE@3s (`20261020T220509-dee848`). Adapted, and given an oracle
route: see *AutoE2E × kitscenes* in
[comparison.md](comparison.md). Four runs over the same keyframes:

| run | rig | map | route | ADE@3s | FDE@3s | lateral MAE | val (186) | overlap (37) |
|---|---|:-:|:-:|---:|---:|---:|---:|---:|
| `20261020T215800-4bd074` | stock: ring front, 2.2 m | ✓ | oracle | 0.847 | 1.903 | 0.417 | 0.848 | 0.842 |
| `20261020T220509-dee848` | its own: base front, 2.1 m | ✓ | oracle | **0.847** | 1.903 | 0.416 | 0.848 | 0.842 |
| `20261020T220642-d63624` | its own | ✓ | — | 0.837 | 1.886 | 0.381 | 0.849 | 0.775 |
| `20261020T220816-0a24e7` | its own | — | — | 0.921 | 2.083 | 0.475 | 0.935 | 0.853 |

The row is the second run: this site's pairing table for AutoE2E's own rig
(`front_camera = "base"`, `ground_offset_m = 2.1`) was adopted because it beat
the stock rig's bar, set before it ran, by 0.1 mm -- which says the front
camera and the 0.1 m make no practical difference. The map is worth 0.08 m. The
oracle route is worth nothing on the val keyframes and costs on the overlap
ones and on lateral error; upstream's own route-zeroed counterfactual found the
same on its full val set (ADE@5s 1.955 without the route, 1.971 with it). 173
of the 223 keyframes have a matched route: on the other 50 upstream's quality
gates refuse the match, and the raster stays empty.

**Checked against upstream's own packed samples.** An out-of-tree reproduction
of the release's KITScenes val figures (1.948 / 5.585 m ADE / FDE@5s against the
model card's 1.941 / 5.564) packed every val sample with upstream's code at
`40a75cf8`. 186 of the 223 keyframes are among them, same scene and frame, with
the ground truth agreeing to 6 mm, so every input the adapter builds was
compared with upstream's, sample by sample:

| input | agreement on the 186 |
|---|---|
| camera tiles, current frame (the six 512 px views and the 1024 px front) | byte-identical |
| current and native-front projections, on its own rig | exact |
| `map_valid` / `route_valid` | 186 of 186; a matched route on 145 in both |
| route raster | median 0 cells differ, max 11 |
| semantic map | median 0.04 % of cells, max 0.16 %, at polygon edges |
| ego rows, on the 117 keyframe-40 samples | speed 2e-5, accel 1.8e-4, yaw rate 1.4e-3, curvature 5.8e-3 m/s / s⁻¹ at most; `v0` 1.2e-5 m/s |
| history projections | 1.0 % relative (median), 7.5 % at most |

Each difference has a cause. KITScenes stores its quaternions to 4 decimals,
and the harness reads yaw without normalising them (≤ 1.2e-4 rad from
upstream's), which moves the edge cells and the yaw-rate rows. The harness's
poses are planar where upstream's history projections use 6-DoF. And on 26 of
the 186 (31 of the 223) upstream's raster comes from a pose 0.5 s back (above),
where 8 % of the map's cells differ; those rows are left out of the two raster
figures.

The plans were compared the same way, each run against the reference made with
its own inputs: the official run's predictions (map and route), its
route-zeroed counterfactual (map only), and an input control with neither.
ADE@3s on the 117 keyframe-40 samples, whose 40-row ego history is the
protocol's own:

| inputs | paddock | upstream | per-sample Δ at 5 s, median / p95 |
|---|---:|---:|---:|
| map and route (`dee848`) | 0.838 | 0.835 | 0.090 / 0.529 m |
| map only (`d63624`) | 0.831 | 0.830 | 0.119 / 0.516 m |
| neither (`0a24e7`, the 23 of these the control covers) | 0.984 | 0.980 | 0.067 / 0.255 m |

The bar set before the runs, upstream's own batch-size noise (median ≤ 0.08 m,
p95 ≤ 0.23 m at 5 s), is met at the median and just missed in the tail where
the inputs are the same (no map or route: 0.067 / 0.255 m, at most 0.265), and
missed further in the tail once the rasters are on. Two of upstream's own valid
runs (FP32 without its stateful cache, BF16 with it) differ by 0.070 / 0.247 m
on the same measure. That the extra tail comes from the raster differences
above is argued, not verified. On the other 69 keyframes the harness gives 48 real ego rows where
upstream's protocol gives 40, and AutoE2E scores better for it (0.865 against
0.914 on those rows).

**comma2k19, 10,165 samples, valid** (`results/comma2k19/autoe2e/20261020T213543-417344`):
**2.016** ADE@3s, from the road camera alone, the five views it does not have left
out. With the ground truth rotated into the calibrated frame the plans are in
(ROADMAP M8), 1.358. Either way it trails constant velocity (1.639 / 1.137), as
an out-of-tree study of this checkpoint on comma2k19 also found.

| horizon | ADE@1s | ADE@2s | ADE@3s | ADE@4s | ADE@5s | FDE@3s | median ADE@3s | miss@3s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 s | 1.160 | 1.593 | **2.016** | 2.507 | 3.185 | 3.084 | 2.068 | 0.775 |

**Checked against that study.** It fed the same checkpoint the same way: the
front camera alone, the other five slots blank with a projection that sees
nothing. It also used its own calibration, pose velocities for the ego rows,
and an anchor every 0.5 s of each drive. On the 1,359 paddock samples that fall
on the same video frame as one of its anchors:

| | agreement |
|---|---|
| `v0`, the speed the rollout starts from | 0.012 m/s apart (median), 0.20 at most |
| controls, the model's raw output | 0.005 m/s² and 1e-4 1/m apart (mean) |
| plans | 0.11 m apart at 3 s, 0.21 m at 5 s (median) |
| ADE@3s, ground truth in the calibrated frame | 1.370 here, 1.352 there |

The plans differ by more than the controls suggest because the inputs differ
in known ways. Paddock differences 20 Hz positions for the ego rows where the
study used pose velocities, uses the route-pooled `rpyCalib` where it used its
own calibration, and builds the history projections from device-frame poses
(the same M8 frame issue).

**nuScenes val, 5,119 keyframes, valid** (`results/nuscenes/autoe2e/20261023T004711-eedbe2`):
**0.839** cumulative L2, 1.565 point L2, 1.319 ADE@3s, 1.86 % predicted collisions.
Six cameras, no map or route. The harness sends the 6.5 s of past keyframes the model
declares instead of 2 s, so the 64 ego rows are real wherever the log reaches back (the full
6.4 s on 61.5 % of samples); before that, 44 of the 64 rows were zero on every sample.

| L2 1 s | L2 2 s | L2 3 s | point avg | cumulative avg | ADE@3s | collision % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.553 | 1.444 | 2.697 | 1.565 | **0.839** | 1.319 | 1.86 |

Between Alpamayo 1.5 (0.787) and Drive-JEPA (0.912); constant velocity scores 0.641.
On the 28 keyframes of scene-0003, run before and after the two input changes, the point
L2 fell from 0.957 to 0.899 (one scene, a check of the inputs, not a score).

**Bench2Drive: not recorded here.** The 50-route sweep (`PADDOCK_INFER_HZ=10`,
a forward every call, route raster on) was paused part-way. Every route run so
far scores DS 3–27 the same way: the car reaches ~12 m/s, the plan curves
further left on each call, and it leaves the lane at 4–5 s. Checked before
calling it the model's behaviour: CARLA actors projected through the model's
front matrix land on the cars, with left cars at y > 0; the route raster maps
pixels exactly as upstream's `navigation/geometry.py`; and on 80 NAVSIM scenes
the deployed convention beats a y-mirrored BEV and a 180° rotation (4 s ADE
1.81 m against 1.88 and 1.92). The same left lean is visible open loop above.
Not ruled out: the route raster is drawn from the leaderboard's downsampled
plan, a point every ~50 m within one road option, so on a bend the corridor
runs as a chord up to ~3 m off the lane at a 100 m radius; and a route with no
map (`route_valid` true, `map_valid` false) is an input upstream never produces
(`navigation/rasterizer.py` sets `route_valid` only with a map). And these runs
predate the closed-loop history fill above, so each route spent its first 6.4 s
with zero rows in the ego history. On neuroNCAP the fill removed the early turn
but not the later loop, so it is unlikely to explain a departure at 4–5 s, but
Bench2Drive has not been re-run with it.
