# AutoE2E

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [autowarefoundation/auto_e2e](https://github.com/autowarefoundation/auto_e2e) at `40a75cf8` |
| Adapter | [`adapters/models/autoe2e/`](../../src/paddock/adapters/models/autoe2e/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK`, `CAM_BACK_RIGHT` |
| Plan | controls, 0.5 s apart, 6.4 s ahead |
| Paddock can run it on | [alpasim](../benchmarks/alpasim.md), [bench2drive](../benchmarks/bench2drive.md), [comma2k19](../benchmarks/comma2k19.md), [hugsim](../benchmarks/hugsim.md), [kitscenes](../benchmarks/kitscenes.md), [navsim](../benchmarks/navsim.md), [neuroncap](../benchmarks/neuroncap.md), [nuscenes](../benchmarks/nuscenes.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

AutoE2E is the Autoware Foundation's open end-to-end planner. Release v1.0
(Hugging Face `AutowareFoundation/auto_e2e`, published 8 Oct 2026) ships two
checkpoints of one architecture, `bevformer_v2_t8_split_navigation_v5`:
`nuplan-epoch-5` and `kitscenes-epoch-5`, the second being the first fine-tuned
on KITScenes. The camera BEV is frozen in both, so it is the nuScenes-trained
BEVFormer V2 checkpoint unchanged. Paddock runs **`kitscenes-epoch-5`**,
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
paddock run --model autoe2e --bench bench2drive --scene <route>   # at its declared 10 Hz ("On Bench2Drive")
```

An earlier, unmerged adapter targeted KITScenes only, before any weights were
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
| `AUTOE2E_EGO_HISTORY` | `real` | the ego history fed: `real` (as measured), `steady` (all 64 rows straight driving at the current speed: an extrapolated past), `no_turn` (yaw rate and curvature zeroed) or `steady_speed` (speed and acceleration steady at the current speed, the turning as measured). The last three are not upstream's input; they test in closed loop what the deep-dive notebook shows offline, that the measured past moves the plan |
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

## On Bench2Drive

AutoE2E's own rate is 10 Hz: it reads 64 ego rows at 10 Hz and plans 64
control steps at 10 Hz (the release `config.json`; `native_input_hz: 10.0`),
with its camera history sampled every 0.5 s. Bench2Drive ticks at 20 Hz, and a
forward on every tick runs the model twice as often as it was built for, at
twice the cost. So `model.yaml` declares `planning_hz: 10.0`: the client asks the
node once every 0.1 s of game time and follows the last plan, unchanged, on the
tick between, the rule every model that declares a rate is driven under
([`docs/benchmarks/bench2drive.md`](../benchmarks/bench2drive.md)). The rate is
recorded as `config.infer_hz` and needs no site setting, so such a run can rank.

Until 28 Oct 2026 the rate came from a site pairing, `AUTOE2E_REPLAN_S = "0.1"`
in `[models.autoe2e.benchmarks.bench2drive.env]`. The client then asked on every
tick, and the node ran a forward every 0.1 s and answered the tick between with the
last plan moved into the current ego frame (`extras.autoe2e.held_s` is its age).
It sets a key `model.yaml` declares, so every run made with it is `valid: false`,
the deep dive's and the steady-history sweep's included.

Cost on one A100 80 GB (20 Oct 2026, idle card): 2.0 s a forward (1.93 s model,
0.07 s packing), 2.3 GB allocated. With six 800 × 450 cameras rendered and
sent every tick, one route drives at ~0.04× real time, ~1.3 s of wall clock a
tick. Forwards do not overlap on one card: three nodes forwarding back to back
took 6.1 s each. Three routes driven side by side ran at 0.015–0.018× each,
~1.25× one route's pace, filling the GPU's idle time between forwards and no
more. At that pace a route on the 4000-tick cap takes ~3.5 h, so
`per_route_timeout_s` wants 14400; the early stop (`docs/benchmarks/bench2drive.md`,
*Ending routes early*) keeps most routes far from the cap.

## Training exposure

Declared in `model.yaml` as nuPlan train and KITScenes train. The nuPlan stage
(`nuplan_full`, from the official BEVFormer V2 R50 T8 initialisation, itself
trained on nuScenes train) is "nuPlan v1.1 train sensor logs" per the model
card, 1,085 logs pinned in the code, and none of NAVSIM's 136 `navtest` logs is
among them (checked against Motional's train inventory). NAVSIM's
`navtest` is drawn from nuPlan's test logs, so the NAVSIM number below is on
the rig and cities AutoE2E was pre-trained on. See `comparison.md`,
*AutoE2E × navsim*.

## On AlpaSim, through the bridge

**Run on the 30 scenes: 0.1694** (results below). The AlpaSim harness gives `autoe2e` its own driver
profile (`adapters/models/autoe2e/alpasim.yaml`, which names
`configs/driver/paddock_bridge_autoe2e.yaml`); the model's `policy.py` is
unchanged.
It is camera-only, as on NAVSIM: the bridge sends no route and the node sends
no map.

**Cameras.** AlpaSim's 26.01 scenes record six f-theta cameras, none looking
straight back. The profile re-renders them as pinholes, every distortion
coefficient zero and with a global shutter open for 1 µs (see "Camera setup on
AlpaSim" below), at the KITScenes calibration's field of view (the release's
`datasets/kitscenes-val/v3.5/rig/*.json` with the image sizes KITScenes ships,
5856 × 3104 front and 3504 × 2272 ring):

| Slot | Node name | AlpaSim id rendered | Pose | Size, f | Field of view |
|---|---|---|---|---|---|
| front | CAM_FRONT | camera_front_wide_120fov | each scene's own (yaw ≈ 0°) | 1920 × 1018, 990.8 px | 88.2° × 54.4° |
| front-left | CAM_FRONT_LEFT | camera_cross_left_120fov | each scene's own (≈ 67°) | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| front-right | CAM_FRONT_RIGHT | camera_cross_right_120fov | each scene's own (≈ −67°) | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear-left | CAM_BACK_LEFT | camera_rear_left_70fov | fixed (2.10, 1.04, 1.02) m, 120° | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear | CAM_BACK | camera_front_tele_30fov | fixed (1.0, 0, 1.93) m, 180°, on the roof | 1024 × 664, 538.9 px | 87.1° × 63.3° |
| rear-right | CAM_BACK_RIGHT | camera_rear_right_70fov | fixed (2.06, −1.01, 1.02) m, −120° | 1024 × 664, 538.9 px | 87.1° × 63.3° |

AlpaSim can change a scene's camera but not add one, so the rear view takes
the tele camera's id. One override applies to every scene, so a camera keeps
its own vehicle's mounting only when the override leaves its pose out; the
three new directions take one pose each. Rear-left and rear-right sit at the
recorded rear cameras' median mounting point (0.01–0.41 m from any one
scene's); the rear view sits on the roof (see "Camera setup on AlpaSim" below).

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
| Rear views far from any recorded camera | The 120°, 180° and −120° renders point 28–32° away from every camera the scene was reconstructed from. Checked by eye on two scenes only: fast cars behind can come out torn or ghosted |
| Cameras lower than training | Five of AlpaSim's views sit 0.8–1.6 m above the ground (the rear view 1.93 m); KITScenes' about 1.7–1.9 m, assuming its LiDAR at the 2.1 m this adapter uses |
| Front-left/right at ±67° | KITScenes' are at ±60°; the recorded pose is kept rather than one fixed pose for every vehicle |
| No correctness check | the release publishes aggregate evaluation reports only, no per-sample plans to compare against |

Cost, measured in the 30-scene run: a median 0.80–0.86 s per call with the
GPU to itself (168–169 calls per scene after handover, about 2.3 minutes of
model time), and a scene in 5.6–7.5 minutes; 1.2–3.7 s per call and 18–38
minutes per scene with the GPU shared. AlpaSim waits for each plan, so this
changes wall time, not the drive.

## Camera setup on AlpaSim

AutoE2E was trained on six cameras around the car: one ahead, one each at
front-left and front-right, rear-left and rear-right, and one looking straight
back. AlpaSim's scenes are rebuilt from six recorded cameras that point
elsewhere, and none looks back. AlpaSim can draw a scene from any viewpoint,
but views far from the recorded cameras come out worse.

- **Front, front-left, front-right** are drawn from the scene's own recorded
  front and side cameras, with the lens changed to KITScenes'.
- **Rear-left and rear-right** have no recorded camera at their angle. They are
  drawn from the recorded rear cameras' mounting point, turned to 120° and −120°.
- **Rear** has no recorded camera at all. It is drawn from the roof, 1.0 m
  ahead of the rear axle and 1.93 m up, looking straight back.

Why the roof: every recorded camera is near the front of the car. A
rear-facing camera mounted there looks back *through our own car*, which the
renderer never draws, so the road beneath the car was never seen by any camera
and comes out as a smudge in the bottom rows. Three spots were tried on two
scenes:
- the recorded rear cameras' mounting point, 0.3–0.6 m from a recorded camera:
  smudge;
- just behind the bumper, 3.2 m from any recorded camera: no smudge, but nearby
  cars behind are smeared and ghosted;
- the roof, 1.0 m from a recorded camera: no smudge, the cleanest image.

1.93 m is the tallest car in the 30 scenes (1.823 m) plus 0.10 m, so the camera
is above every roof. KITScenes' rear camera is about 1.9 m up. AlpaSim applies
one camera pose to every scene, so it cannot follow each car's height.

Every view uses a global shutter open for 1 µs. The recorded cameras read
their rows top to bottom over 30 ms, so a fast car or the car's own turn bends
across one image, while AutoE2E treats each image as a single instant. With a
global shutter AlpaSim draws the whole image at the moment the shutter opens,
but the bridge tells the model the car's pose at the moment it closes. A 1 µs
opening makes those the same moment. (0 cannot be used: AlpaSim reads it as
"use the recorded 30 ms".)

Remaining known gaps (table above): the plan is driven from the rear axle with
no offset; the three rear views point 28–32° away from any recorded camera; and
five of the six views sit lower than KITScenes' cameras.

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

**NAVSIM v1, the same 12,146 plans, valid** (`results/navsim_v1/autoe2e/20261104T210206-3022fb`):
**0.538** PDMS. No model ran: the v2 run's recorded plans were replayed through
the v1.1 scorer. NC 0.824 and DAC 0.777 match the v2 run's to three decimals,
as they should for the same plans.

| PDMS | NC | DAC | EP | TTC | comfort | DDC (weight 0) |
|---:|---:|---:|---:|---:|---:|---:|
| **0.538** | 0.824 | 0.777 | 0.514 | 0.700 | 1.000 | 0.859 |

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

**comma2k19, 10,165 samples, valid** (`results/comma2k19/autoe2e/20261106T073744-1b4fe1`):
**1.350** ADE@3s, from the road camera alone, the five views it does not have left
out, with its past path and the ground truth in the car's frame. The earlier run
(`20261020T213543-417344`) was shown its past path and scored in the camera's
frame: 2.016, and 1.382 with the ground truth rotated into the car's frame.
Either way it trails constant velocity (1.135 in the car's frame), as an
out-of-tree study of this checkpoint on comma2k19 also found.

| horizon | ADE@1s | ADE@2s | ADE@3s | ADE@4s | ADE@5s | FDE@3s | median ADE@3s | miss@3s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 s | 0.734 | 0.977 | **1.350** | 1.949 | 2.846 | 2.396 | 1.339 | 0.640 |

**Checked against that study, with the earlier run.** It fed the same checkpoint
the same way: the front camera alone, the other five slots blank with a
projection that sees nothing. It also used its own calibration, pose velocities
for the ego rows, and an anchor every 0.5 s of each drive. On the 1,359 paddock
samples that fall on the same video frame as one of its anchors:

| | agreement |
|---|---|
| `v0`, the speed the rollout starts from | 0.012 m/s apart (median), 0.20 at most |
| controls, the model's raw output | 0.005 m/s² and 1e-4 1/m apart (mean) |
| plans | 0.11 m apart at 3 s, 0.21 m at 5 s (median) |
| ADE@3s, ground truth in the calibrated frame | 1.370 here, 1.352 there |

The plans differ by more than the controls suggest because the inputs differ
in known ways. Paddock differences 20 Hz positions for the ego rows where the
study used pose velocities, uses the route-pooled `rpyCalib` where it used its
own calibration, and, in that run, built the history projections from
device-frame poses (the same frame issue, fixed since).

**nuScenes val, 5,119 keyframes, valid** (`results/nuscenes/autoe2e/20261023T004711-eedbe2`):
**0.839** cumulative L2, 1.565 point L2, 1.319 ADE@3s, 1.86 % predicted collisions.
Six cameras, no map or route. The harness sends the 6.5 s of past keyframes the model
declares instead of 2 s, so the 64 ego rows are real wherever the log reaches back (the full
6.4 s on 61.5 % of samples); before that, 44 of the 64 rows were zero on every sample.

| L2 1 s | L2 2 s | L2 3 s | point avg | cumulative avg | ADE@3s | collision % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.553 | 1.444 | 2.697 | 1.565 | **0.839** | 1.319 | 1.86 |

Between Alpamayo 1.5 (0.787) and METEOR (0.977); constant velocity scores 0.641.
On the 28 keyframes of scene-0003, run before and after the two input changes, the point
L2 fell from 0.957 to 0.899 (one scene, a check of the inputs, not a score).

**AlpaSim, the 30 scenes of `public_2601_30scenes`, valid**
(`results/alpasim/autoe2e/20261022T045847-86b955`): **0.1694** `scene_score`
through paddock's bridge, six pinhole cameras at 20 Hz, one rollout per scene.
Ranked with the other 30-scene rows (seventh of seven; camera count and the
bridge are not reasons to separate rows). On the same scenes, for context: OpenPilot 0.7726, Drive-JEPA
0.1.4b 0.5433 natively and 0.5302 through the bridge, the constant-velocity
floor 0.4876.

| scenes | `scene_score` | sd | progress | `duration_frac_20s` | at-fault, scored | off-road, scored |
|---|---:|---:|---:|---:|---:|---:|
| 30 | **0.1694** | 0.2034 | 0.2867 | 0.121 | 9 / 30 | 7 / 30 |
| first 10 | 0.3175 | 0.1938 | 0.2924 | 0.120 | 2 / 10 | 0 / 10 |
| other 20 | 0.0954 | 0.1638 | 0.2838 | 0.122 | 7 / 20 | 7 / 20 |

The car leaves the human's path to the left, as on neuroNCAP and Bench2Drive:

- **It crosses a lane line to the left in 28 of 30 scenes.** The two to the
  right are the stopped-car pull-outs (`b44a4a44`, `d2715023`). Above 6 m/s at
  handover (16 scenes) the crossing comes a median 0.93 s after handover; on
  the 14 slower scenes, 3.4 s. It is off the road at some point in 28 of 30.
- **The first plan is close; the drift feeds itself.** Each plan's 2 s point,
  placed on the human's path, is 0.1–3.1 m to the left at the first call on
  the fast scenes, and each later plan bends further left as the car drifts:
  on `04394343` (14.9 m/s) 1.4 → 1.6 → 2.5 → 4.0 → 6.2 → 9.4 m over the first
  0.5 s. The car follows; 2 s after each early plan it is 0.8–8.0 m further left
  than that plan's own 2 s point. On the fast scenes the plans also brake
  (3–4.5 m/s² in the first 2 s on 14 of 16), and the car holds about two thirds
  of the human's speed.
- **The score mostly measures how far the car got before the cut.** Scoring
  stops at the first of: 4 m from the human's path, an at-fault collision, or
  off road. That came a median 1.55 s after handover, 12% of the clip. 15 of the
  30 scenes have an at-fault collision somewhere in the clip; 6 of those
  collisions come after scoring stopped and so do not count. The best score,
  0.621 on `16b83346`, is a scene where the car crossed the lane line at 0.9 s
  and was off the road at 1.3 s.

The inputs were as the profile intends on every scene: 168–169 model calls
after handover, 34 padded motion-history rows at the first call, no repeated
image frame from 0.5 s on. The run was made before this adapter gained the
closed-loop history fill, the map and route rasters, left-out views and the
history-frame tolerance. None of them reaches
AlpaSim, which sends no map or route, all six views, frames 50 ms apart (so the
tolerance stays 0.06 s), and a history the bridge has already padded. Checked on one scene
(`01d503d4`) driven again with the current code: all 396 per-tick log rows and
every plan identical, the eight scene metrics equal, and the new log fields at
`views` 6, `ego_rows_filled` 0, `map_valid` false. The run's id hash (`86b955`)
predates 21 Oct 2026, when `model.yaml`'s env joined the id, so a rerun gets a new
id.

**Bench2Drive, the 50 routes of the other rows, valid**
(`results/bench2drive/autoe2e/pooled50-40c29d24c37d542a-cc7a9c`): **15.30** mean
driving score with upstream's inputs, the measured ego history and the route
raster, and nothing `model.yaml` declares set at the site, asked for a plan at its
declared 10 Hz (`planning_hz`); every route driven on Paddock `4521edc`, 28-29 Oct
2026. A route ends after 30 s without 3 m of progress or at 4x the time of
Think2Drive, the expert driver whose recorded drives Bench2Drive ships,
and is scored as a failed drive; the rows without the early stop drove every
route to Bench2Drive's own end. The record is valid and ranks, in the support
matrix as on the leaderboard; the README says how it was driven.

| inputs | record | DS | DS median | RC (%) | infraction penalty | routes with a collision | ended: stuck / over time / at a collision | valid |
|---|---|---:|---:|---:|---:|---:|---:|---|
| upstream's, at 10 Hz | `pooled50-40c29d24c37d542a-cc7a9c` | **15.30** | 12.89 | 26.37 | 0.60 | 36 / 50 | 42 / 1 / -- | yes |
| steady past, no route | `pooled50-steady-noroute-40c29d24c37d542a-8d26e9` | 37.74 | 28.17 | 57.59 | 0.63 | 33 / 50 | 0 / 0 / 33 | no |

The second row is the deep dive's fix (below), driven on `7244bc5`: a steady
straight past at the current speed and no route raster
(`AUTOE2E_EGO_HISTORY=steady`, `AUTOE2E_ROUTE=0`), 10 Hz from the
`AUTOE2E_REPLAN_S` pairing, and a route ended at its first collision (33 of the
50 did). It sets three keys `model.yaml` declares and stops where Bench2Drive's
protocol does not, so it does not rank. It scores higher on the same routes, but
the two runs also differ in the collision stop, in how the plan is held between
forwards (below) and in the code revision, so the gap is not the inputs' alone.

The two rows also hold the plan between forwards differently: the client follows
it unchanged since 23 Oct 2026; before that the pairing had the node move it into the current
ego frame. On 1711 and 2286, driven from the same start both ways (the deep
dive's recorded drives, upstream's inputs), the first plans agree within 4 cm and
2 mm, the cars are 1.1-1.5 m apart after 4 s, and the client's hold launches
about 0.5 m/s faster; both drives then crash where the other did.

Earlier runs, each `valid: false`:

- The first 50-route sweep (`PADDOCK_INFER_HZ=10`, route raster on) ran a
  forward on every 20 Hz call -- the Bench2Drive agent did not read that
  variable until 23 Oct 2026 -- and predates the closed-loop history fill. Every route scored DS 3–27 the
  same way: the car reaches ~12 m/s, the plan curves further left on each call,
  and it leaves the lane at 4–5 s. Checked before calling it the model's
  behaviour: CARLA actors projected through the model's front matrix land on
  the cars, with left cars at y > 0; the route raster maps pixels exactly as
  upstream's `navigation/geometry.py`; and on 80 NAVSIM scenes the deployed
  convention beats a y-mirrored BEV and a 180° rotation (4 s ADE 1.81 m
  against 1.88 and 1.92).
- At the model's own 10 Hz (*On Bench2Drive*), with the history fill and the
  route raster on: 59 routes scored, mean DS 13.7, before that sweep was
  stopped on 21 Oct 2026 for the deep dive below.
- The full 220-route sweep with the steady settings, from 23 Oct 2026: stopped
  on 28 Oct at 204 routes for the valid run above. Its 50 routes that the other
  rows pool are the steady row's record.

**Bench2Drive, all 220 routes, valid**
(`results/bench2drive/autoe2e/full220-86aa45e768a29a64-740c68`, the 220-route protocol: the
same inputs, rate and early stop as the 50-route row, up to three attempts a route):
**14.29** mean driving score, RC 27.72%. It pools the 216 routes driven at
`a100-gpu-vm-002` on 7-8 Nov 2026 (`full220-836dda734659a86c-63902e`, superseded by it)
with routes 3800, 11715, 23695 and 24071, set aside there and driven here on 12 Nov under
patch 0003. Over those 216 routes:
**14.36** mean driving score (median 13.05), RC 27.80%, infraction
penalty 0.55, a collision on 172 routes; the 30 s rule ended 177 and the over-time
rule 12. It
moves off on every route and completes none. The tables, and Bench2Drive's own
scores for the record, are in
[Bench2Drive — 220 routes](../benchmarks/bench2drive.md#bench2drive--220-routes).

Why it swerves, from three routes (1711, 2286, 2373) whose every request was
recorded and replayed into the model to within 0.5 mm of the plan CARLA's node
sent (an offline analysis, 21 Oct 2026). The 64 ego-history rows
set the plan's turn and acceleration; the cameras, the route raster and the
projection height move it by centimetres. The network continues the launch
every route starts with (+2 to +3.4 m/s², to 12 m/s on 8.3 m/s streets), and
once the car yaws, its own turning, fed back through the yaw-rate and
curvature rows, grows the plan's curvature into a spin. Driven closed loop
with each `AUTOE2E_EGO_HISTORY` (three routes): mean DS 51.3 with `steady` (a
straight past at the current speed), 21.4 `no_turn`, 16.2 `real`, 14.0
`steady_speed`; all four run the red light on 2373. With `steady`, the route
raster off kept 1711 in its lane (DS 91.7, ended by a 2x-expert clock with no
infraction, against 30.8 with it on; one route) -- a route with no map is an
input upstream never produces. Bench2Drive's and KITScenes' camera rigs
(`camera_rig`) scored 88.7 and 88.0 on that route, so no better; those rigs
also send JPEG where paddock's sends raw pixels. Not established: whether the
model sees the cars it hits -- its BEV segmentation shows no structure on
CARLA frames.
