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

- **Views and frames.** All six views are required; `infer` raises if the
  current observation lacks one. The seven history frames are the held frames
  nearest t−3.5 … t−0.5 s within 0.06 s that carry all six views. Where the
  past does not reach that far (a route or log start) the oldest six-view frame
  is repeated with its own pose; each plan's `extras` and `diagnostics()`'s
  last call say how many were (`history_repeated`).
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
    that turned the car right within 2 s, into a parked car (score 0.14); with
    the past filled it drove straight.
- **Navigation.** No semantic map: zeros with `map_valid` false, upstream's
  own no-map track. The route raster is drawn from `obs.route_xy` when the
  benchmark sends one: a corridor 1.75 m either side, clipped 10 m behind, and
  a 2 m destination disk where the polyline stops short of 200 m. Bench2Drive
  sends the leaderboard's route; NAVSIM has none, so its scenes run camera +
  ego only. `AUTOE2E_ROUTE=0` forces the camera-only track. That track is
  upstream's official KITScenes *test* input (`camera_only_missing_map_route`);
  training and validation always had map and route
  (`distributed_training/reactive_data.py` at `40a75cf8`), so NAVSIM shows this
  checkpoint an input it was evaluated on but never trained on.
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

## Training exposure

Declared in `model.yaml` as nuPlan train and KITScenes train. The nuPlan stage
(`nuplan_full`, from the official BEVFormer V2 R50 T8 initialisation, itself
nuScenes-trained) names no split in the release; that it held out nuPlan test
is inferred from the release's own nuPlan-test figures, not stated. NAVSIM's
`navtest` is drawn from nuPlan's test logs, so the NAVSIM number below is on
the rig and cities AutoE2E was pre-trained on. See `comparison.md`,
*AutoE2E × navsim*.

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
(`navigation/rasterizer.py` sets `route_valid` only with a map).
