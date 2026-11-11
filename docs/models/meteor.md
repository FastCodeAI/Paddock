# METEOR

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [tier4/METEOR](https://github.com/tier4/METEOR) at `dc193a81` |
| Adapter | [`adapters/models/meteor/`](../../src/paddock/adapters/models/meteor/) |
| Environment | uv, Python 3.11 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [bench2drive](../benchmarks/bench2drive.md), [nuplan](../benchmarks/nuplan.md), [comma2k19](../benchmarks/comma2k19.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

## What it is

[TIER IV METEOR](https://github.com/tier4/METEOR) ("Multi-task Estimation of
Traffic Elements, Objects & Roads") is a 54M-parameter surround-view network:
ResNet-34 + FPN per camera, a depth-gated IPM lift into one 800x500 BEV grid at
0.2 m, and twelve task heads. Head seven is an end-to-end planner: three
trajectory hypotheses of six waypoints at 0.5 s (3 s), mode logits, and
steer/accel/brake. It was trained on TIER IV's Co-MLOps DRS recordings (Japan,
plus a US subset) with auto-generated labels only; the authors publish no
accuracy figures, and list "evaluation on a public benchmark" as open.

Pinned in `repos.yaml` at `dc193a81` (branch `main`, 2026-09-09). Nothing from
the checkout is imported: the adapter runs the released ONNX. The one thing it
takes from upstream source, METEOR's own rig calibration (below), is copied in
from a later commit (`27a5bd0`, `bevlane/ref_rig_drs.npz`); a test compares the
copy with the clone's file wherever a clone has it. A run still records the
clone's sha and whether it is at the pin, but this adapter reads nothing from
the clone, so that sha does not change a plan.

## Weights and environment

Release `v1.0` on Hugging Face, `AutowareFoundation/meteor`:

| file | what | used by paddock |
|---|---|---|
| `meteor_v157c3Z.onnx` | plain ONNX, opset 17, no custom ops, camera-only, history baked out, uint8 input | **yes** |
| `meteor_v157.pt` | PyTorch checkpoint (`--model v52`), has the driving-command input | no |
| `meteor_v157.param.yaml` | camera order, resolution, BEV grid, output tensors | reference |
| `SHA256SUMS` | checksums | `setup.sh` verifies against it |

`setup.sh` builds `<repos>/METEOR/.venv` with `uv` (Python 3.11,
`onnxruntime-gpu==1.30.*`, which is a CUDA 13 build, and NVIDIA's CUDA 13
wheels for the runtime, cuBLAS, cuFFT, cuRAND, NVRTC and cuDNN --
`nvidia-cuda-runtime==13.*`, `nvidia-cublas==13.*`, `nvidia-cuda-nvrtc==13.*`,
`nvidia-cufft`, `nvidia-curand`, `nvidia-cudnn-cu13`; the CUDA 12 generation
dropped the `-cu12` suffix and does not satisfy this onnxruntime -- plus what
the paddock node needs) and downloads the three files to
`<weights>/meteor/`. It deliberately does not follow the upstream recipe
(python3.10, torch 2.1.1/cu121, TensorRT 8.6): that torch has no kernels for a
Blackwell card, and the ONNX path needs none of it.

Settings, all in `model.yaml` `runtime.env`:

| variable | default | meaning |
|---|---|---|
| `METEOR_ONNX` | `{weights}/meteor/meteor_v157c3Z.onnx` | the graph |
| `METEOR_PROVIDERS` | `cuda` | `cuda` falls back to CPU with a warning (~4 s a frame); `cuda-strict` refuses; `cpu` |
| `METEOR_ORT_THREADS` | `4` | onnxruntime intra-op threads. The reference box fails above ~90% CPU |
| `METEOR_GPU_MEM_MB` | `6144` | CUDA arena cap |
| `METEOR_HORIZON_S` | unset (3 s) | see "Horizon" below |
| `METEOR_FRONT_ONLY` | `narrow_refpose` | how the slots are filled when the front camera is the only one; see "A front camera alone" below |

`load()` runs the graph once on a blank frame, so a missing kernel for the GPU
architecture fails inside `startup_timeout_s` with the error in the node log,
not on the first scored sample.

## Input contract and how paddock fills it

The graph takes `imgs` uint8 `[1, 8, 3, 432, 768]` RGB, `K` `[1, 8, 3, 3]` at
768x432, `T_cam_ego` `[1, 8, 4, 4]` (ego to camera, OpenCV optical camera
frame), and `v0` `[1]` ego speed. Reference: `hf/onnx_smoke_test.py` upstream.

| METEOR slot | paddock camera | when absent, calibration from |
|---|---|---|
| `CAM_FRONT_WIDE` | `CAM_FRONT` | `CAM_FRONT_NARROW`, then first present |
| `CAM_FRONT_LEFT` | `CAM_FRONT_LEFT` | first present |
| `CAM_FRONT_RIGHT` | `CAM_FRONT_RIGHT` | first present |
| `CAM_BACK_WIDE` | `CAM_BACK` | `CAM_BACK_LEFT`, then first present |
| `CAM_BACK_LEFT` | `CAM_BACK_LEFT` | first present |
| `CAM_BACK_RIGHT` | `CAM_BACK_RIGHT` | first present |
| `CAM_FRONT_NARROW` | none | `CAM_FRONT_WIDE` |
| `CAM_BACK_NARROW` | none | `CAM_BACK_WIDE` |

**Absent cameras.** METEOR was trained with camera dropout for 7-camera rigs:
an absent camera is a zero image tensor with a donor camera's calibration
(`bevlane/dataset.py`, `_CAM_FALLBACK` and `cam_drop`), and METEOR's runtime
feeds a missing eighth camera as a zero buffer. The adapter feeds black, which
the graph's own `/255` turns into that zero tensor. The two NARROW slots are
absent in paddock unless the front camera is alone (below). Every benchmark
but comma2k19 supplies the nuScenes six; KITScenes did not until 14 Sep 2026; its harness handed over only `camera_ring_front`
although the dataset records a six-camera ring with calibration for each. The
run made under that harness is kept as an ablation (below): seven absent slots
is a heavier ablation than anything the network saw in training, and it cost
2.9 m of ADE.

**A front camera alone.** comma2k19 has one camera, a phone on the windscreen.
Under the rule above, all seven black slots would take that camera's
calibration. METEOR's lift (`bevlane/model.py`, `project_bev`) averages, per BEV
cell, the features of every slot that sees the cell, weighted by that slot's
depth probability plus 0.05, so each cell the real camera sees would be about
7/8 black-image feature. So when `CAM_FRONT` is the only camera supplied, the
adapter follows `METEOR_FRONT_ONLY` instead:

| rule | `CAM_FRONT_NARROW` gets | the black slots' calibration |
|---|---|---|
| `donor` | a black image | the front camera's (the rule above) |
| `narrow` | the centred 30° window of the front frame | the front camera's |
| `refpose` | a black image | METEOR's own rig |
| **`narrow_refpose`** (default) | the centred 30° window of the front frame | METEOR's own rig |

"METEOR's own rig" is each slot's `K` and `T_cam_ego` on the recording rig,
copied from upstream's `bevlane/ref_rig_drs.npz`: a black image with its own
slot's calibration is what training's per-camera dropout looks like. The 30°
window is what upstream's KITScenes converter feeds the narrow slot
(`tools/kitscenes/kitscenes_to_bevlane.py`). On comma's 1164x874 frame it is
488x274 at (338, 300), enlarged 1.57x with a bilinear filter (an area filter
degenerates when enlarging), f = 1432 x 1435 against METEOR's narrow camera's
1415 x 1241. The front slot gets the usual full-width 16:9 window, rows 110-765,
f = 600; METEOR's own wide cameras have f = 335 x 292, so the comma view looks
1.8x more zoomed in.

Any other set of cameras uses the donor rule whatever the setting says, so no
other benchmark's number moves; a test feeds the six-camera ring, a
three-camera front and a back camera alone, and checks the feed is byte for
byte what `donor` gives. The KITScenes front-only ablation below was made with
`donor`.

**The default was chosen on METEOR's own data, not on comma2k19.** An
out-of-tree study ran METEOR's demo scene (`valday` from
`AutowareFoundation/meteor-demo-scenes`, the 132 frames with a recorded 3 s
path, 1-11 m/s) with all eight cameras, then with the front camera alone under
each rule, and picked the best:

| input | ADE (m) | FDE@3s (m) |
|---|---:|---:|
| all eight cameras | 0.65 | 1.49 |
| front only, `narrow_refpose` | **1.01** | 2.11 |
| front only, `refpose` | 1.21 | 2.46 |
| front only, `narrow` | 1.94 | 4.41 |
| front only, `donor` | 2.22 | 4.92 |
| constant velocity | 1.37 | 3.24 |

The study ran the checkpoint in PyTorch. Re-run through this adapter's
`build_inputs` and the released ONNX on the CUDA provider (29 Oct 2026, a
script kept outside the repo), every figure in the table comes out the same to
two decimals. What it says: on METEOR's own data, the rule, not the camera
count, decides whether one camera beats constant velocity.

**Normalisation, as shipped.** `meteor_v157.param.yaml` and `deploy/README.md`
say the uint8 graph applies `/255` and ImageNet mean/std normalisation
in-graph. It does not. Checked on the downloaded release: the graph is
`Cast -> Div(255) -> Reshape -> Conv`, and that first Conv's weights and bias
equal the checkpoint's `stem.0` conv with only its BatchNorm folded in
(max difference 0.0; a mean/std fold would change the weights by up to 1.26).
METEOR's training loader does normalise with mean/std
(`bevlane/dataset.py`), and so does its `evaluate_ego`; its deployment path
(`deploy/eval_deploy.py`, `deploy/realtime_trt.py`, the Orin runtime) feeds
`/255` only, like the graph. paddock evaluates the released graph as shipped,
so its numbers grade the deployed model, which sees inputs scaled differently
from the ones it trained on. A re-export with the normalisation in place is
possible from `meteor_v157.pt` and would be a separate, clearly labelled
adapter variant. If the pixel that lands on the *training* zero were wanted,
it would be `round(255 * mean) = (124, 116, 104)`; with the graph as it is,
black is the deployed convention.

**Frames.** Any size is cropped to 16:9 around the principal point and
area-downscaled to 768x432; the intrinsics follow the pixels. KITScenes
(3504x2272, f=1841) becomes f=403 with the principal point near the centre;
nuScenes (1600x900) is a pure 0.48 scale; Bench2Drive's 800x450 renders are a
pure 0.96 scale (f=548 for the 70 deg views, 269 for the 110 deg back view).

**Ego frame.** The BEV lift assumes the ego origin is on the road (`z = 0`).
nuScenes and NAVSIM satisfy that; KITScenes' harness shifts its roof-pod
reference down by the pod height for the same reason. Bench2Drive sends CARLA's
vehicle origin, which is on the road but at the middle of the car, about 1.4 m
(half the wheelbase) ahead of the rear axle Autoware's `base_link` sits on.
Nothing corrects that; what it does to the plan is argued, not measured.

**No driving command.** The ONNX has no intent input, so the model runs in its
trained "no navigation" mode (30% intent dropout in training). At junctions it
picks whichever of its three modes scores highest; nothing steers that choice.
The `.pt` checkpoint does take an intent one-hot, which a later torch-based
adapter could feed from paddock's `Command`.

## Output and horizon

`ego` is `[1, 42]`: three blocks of six `(x, y)` waypoints at 0.5..3.0 s in the
current ego frame (x forward, y left, metres), then three mode logits, then
steer (rad), accel (m/s^2) and a brake logit. The adapter takes the argmax mode,
the same decode as METEOR's own `evaluate_ego` and `closed_loop_eval`. The
deployed demo adds mode hysteresis and a "prefer straight" rule; paddock does
not, so the number grades the network, not the demo's post-processing.

The chosen mode, logits and the three control scalars ride along in
`Trajectory.extras` as `meteor_*` keys. No benchmark reads them.

**Horizon.** The plan is 3 s. NAVSIM scores 4 s and its harness refuses a
shorter plan rather than extrapolate. Setting `METEOR_HORIZON_S = "4.0"` under
`[models.meteor.env]` in `paddock.toml` makes the adapter continue the last
0.5 s segment (the model's end velocity and heading, held) to 4 s, and records
`meteor_extrapolated_to_s` in `extras` and `model_env` in run.json. This is a
plan the model never made; it is opt-in so that KITScenes and the closed-loop
benchmarks always score the native output.

**Rate.** `model.yaml` declares `planning_hz: 5.0`: METEOR's recordings are
stored and labelled at 5 Hz (upstream `docs/REPRODUCE.md`), and the AlpaSim
driver below runs its graph at 5 Hz. Bench2Drive's client asks for a plan every
0.2 s and follows the last one in between, and records the rate as `infer_hz`
in the run id; HUGSIM's 4 Hz steps are slower, so every step is planned there.
The graph is single-frame, so the rate changes how often the car gets a fresh
plan, not what the model is shown.

## Running it

```bash
PADDOCK_REPO=<repos>/METEOR PADDOCK_WEIGHTS=<weights> \
  bash src/paddock/adapters/models/meteor/setup.sh
paddock doctor --model meteor
paddock run --model meteor --bench kitscenes --limit 5 --dry-run
paddock run --model meteor --bench kitscenes --limit 5
paddock run --model meteor --bench kitscenes
paddock run --model meteor --bench bench2drive --scene 2802   # one route, closed loop, a plan every 0.2 s
```

Runs on the reference box are launched under `taskset` (see
`docs/troubleshooting.md`); the node inherits the affinity.

## Validation against METEOR's own reference

On a frame of the authors' `valday` demo scene (`AutowareFoundation/meteor-demo-scenes`),
the adapter's feed built from paddock `Observation` fields equals the feed
`hf/onnx_smoke_test.py` builds: image bytes identical for all six shared
slots, `K` identical, `T_cam_ego` within 6e-8. Feeding those six slots with the
two NARROW slots absent moves the 3 s waypoint by 0.27 m against the full
8-camera feed.

On comma2k19 the reference is that out-of-tree study's standalone one-camera
adapter, which feeds the same rule. On the five keyframes paddock scores in its
sample segment (`Chunk_8`, `99c94dc769b5d96e|2018-08-17--09-18-11`, segment 50,
frames 159-999), one forward pass each, checked once on 29 Oct 2026:

| | paddock against the study |
|---|---|
| mount calibration (route-pooled) | pitch 2.0215° / yaw 2.0741° against 2.0215° / 2.0742°; `camera2ego` within 1e-6 |
| decoded frames | identical (the study's adapter fed paddock's decode reproduces its own images byte for byte) |
| `K`, `T_cam_ego`, the six black slots | identical |
| the two real images | 0.4 (front) and 0.1 (narrow) grey levels apart on average: PIL's area and bilinear filters against OpenCV's |
| the study's feed through paddock's ONNX session (CUDA) | its PyTorch plans to within 0.05 m |
| `v0` | CAN speed here, pose speed there (1.2 % apart); moves the plan by under 0.03 m |
| paddock's own plans | the study's mode on all five; the 3 s point 0.08-1.15 m away, where the study's feed through the same session is within 0.05 m, so the gap is the resize filters |

So paddock feeds the study's inputs except for the resize filters, and the
model is sensitive to them on this camera: under half a grey level on average
moves the plan by up to a metre, on plans that end about 50 m short of where the car went
(below). Repeated forward passes on the same feed, in one session or two, give
bit-identical outputs, so that sensitivity is to the pixels, not run-to-run
noise on the GPU.
`fit_frame` is shared by every benchmark, so matching OpenCV here would move
the six-camera numbers too; it is left as it is.

Timing on the reference box (RTX PRO 6000, shared): load plus warm-up 1-2 s,
39-90 ms per forward on the CUDA provider, ~4.4 s on CPU with 4 threads.

## Reference numbers

KITScenes, `results/kitscenes/meteor/20260914T215457-ab0586` (n=223, the
shared `kitscenes_e2e_final` list, the six-camera ring, native 3 s plan,
99 s wall clock):

| ADE | FDE@1s | FDE@2s | FDE@3s | longitudinal MAE | lateral MAE |
|---|---|---|---|---|---|
| **1.879** (median 1.419) | 0.695 | 2.040 | 4.034 | 1.563 | 0.688 |

Constant velocity scores 1.381 on the same list; Alpamayo 1.002, VisionPilot
1.475, OpenPilot 1.248 (its configured run; 1.599 stock), Drive-JEPA 2.364. A
second run reproduced all 223 predictions exactly.

**Front camera only, kept as an ablation:**
`results/kitscenes/meteor/20260914T185351-5e5223`, made before 14 Sep 2026 when the
harness handed over one camera: ADE **4.784** (median 3.411), FDE@3s 9.011,
longitudinal MAE 4.427, lateral MAE 1.064. The ring improves 177 of the 223
samples. Where the difference lives, by ground-truth speed band (3 s point,
predicted vs true, and ADE, one camera -> six):

| speed | n | true x@3s | one camera | six cameras | ADE one -> six |
|---|---|---|---|---|---|
| 0-3 m/s | 35 | 6.0 m | 5.2 m | 7.3 m | 1.69 -> 1.68 |
| 3-8 m/s | 103 | 16.8 m | 13.1 m | 16.2 m | 2.66 -> 1.66 |
| 8-13 m/s | 70 | 30.6 m | 17.2 m | 27.8 m | 7.45 -> 1.85 |
| 13-30 m/s | 14 | 42.8 m | 15.8 m | 35.3 m | 15.02 -> 4.11 |

So the longitudinal collapse at speed in the single-camera run was mostly the
missing cameras, not a speed prior: with the ring the plan tracks the truth to
13 m/s. Above that it still stops short (35 m at 3 s for a car that went 43),
which matches the `v0` sweep on an in-domain frame with six cameras, where
x@3s plateaus at 34-35 m from 16 m/s up. The lateral error with six cameras
is 0.69 m; the front-only demo probe (y@3s +2.97 m with one camera against
-0.04 with six) explains the rest of the single-camera number.

Both numbers are properties of the released model under paddock's inputs, not
of the adapter: the feed matches the authors' loader byte for byte (above).

nuScenes val, `results/nuscenes/meteor/20260914T202122-4320bb` (n=5,119
anchors in 150 scenes, all six cameras, 2 Hz keyframes, native 3 s plan,
14 min wall clock):

| L2@1s | L2@2s | L2@3s | point L2 avg | cumulative L2 avg | ADE@3s | point collision avg |
|---|---|---|---|---|---|---|
| 0.616 | 1.683 | 3.356 | **1.885** (median 1.581) | 0.977 | 1.570 | 6.15 % |

Drive-JEPA scores 2.344 / 1.328 / 1.63 % on the same anchors and constant
velocity 1.288 / 0.641 / 1.66 %. With its cameras filled in, METEOR's
displacement error is below Drive-JEPA's; its collision rate is above both. Where
each comes from, measured on the per-sample records:

- **Speed is in range here.** nuScenes is urban: 96 % of anchors are below
  12 m/s, and from 6 m/s up the 3 s point is within 2 m of the truth
  (22.2 vs 24.2 m at 6-9 m/s, 30.0 vs 30.4 at 9-12). The plateau above
  13 m/s seen on KITScenes is out of range here. The error moves to the *low* end instead: at
  1-3 m/s the model plans 11.0 m where the car went 6.5 (L2 avg 3.24, the
  worst band), and at standstill it creeps.
- **Collisions are concentrated at standstill, where the planner creeps.** Of
  the 893 anchors where the ego is stopped, 22.6 % collide: the plan creeps 0.95 m forward on average into
  the vehicle the car is queued behind. Anchors above 6 m/s collide 1.1 %
  of the time, in line with the other models. Constant velocity plans zero
  motion when stopped and collides on 1.7 % overall; METEOR's brake logit is
  strongly negative when v0 is zero (the demo probe gives -5.1), so this is
  the planner head, not a decode error.
- **A left bias.** On straight, moving anchors the plan ends 0.83 m to the
  left of the truth at 3 s. Split by recording location: +1.26 m in Boston
  (right-hand traffic, n=1,525) against +0.60 m in Singapore (left-hand,
  n=1,570). METEOR trained on Japanese roads, which drive on the left, so a
  prior for sitting left of the centreline is the plausible reading;
  consistent with the split, not proven by it.
- **Turns without a command.** With no intent input, left turns are still
  taken (mean y@3s +4.28 m vs +4.15 true, n=257) and right turns are cut
  short (-2.30 vs -4.34, n=361): the scene geometry drives the mode choice,
  and the left bias helps on one side and hurts on the other.

The record carries per-scene aggregates (`by_group`, all 150 scenes), because
the harness sets `group` to the scene. The four sibling nuScenes records predate
the per-scene aggregates and carry none, so a per-scene comparison against them is not possible from
tracked files until they are re-run under the current runner.

comma2k19, `results/comma2k19/meteor/20261106T064120-226c00` (the full
dataset, n=10,165, the same samples as every other comma2k19 row; one phone
camera, filled by `narrow_refpose`; native 3 s plan; scored against the ground
truth in the car's frame; `valid: true`; 56 min on one H200 with eight nodes):

| ADE@1s | ADE@2s | ADE@3s | FDE@3s | median ADE@3s | miss@3s |
|---|---|---|---|---|---|
| 11.841 | 21.621 | **30.167** | 51.507 | 37.395 | 0.892 |

The earlier run, `20261030T001935-0b8d0b`, was scored against a ground truth in
the camera's frame and gave 30.075. Its plans are the same as this run's to
within 0.01 mm, so the change is the scoring alone. Constant velocity scores
1.135 on the same samples, and every other model scores 18.754 or less. METEOR
is better than constant velocity on 3.5 % of the samples, all of them below
15 m/s. Measured on the per-sample records:

- **The plan stops about 17 m ahead, whatever the speed.** By the car's speed
  at the scored frame (3 s point, plan against the car's own path):

  | speed | n | plan x@3s | true x@3s | plan y@3s | ADE@3s |
  |---|---:|---:|---:|---:|---:|
  | below 15 m/s | 2,564 | 8.0 m | 14.2 m | −1.74 m | 3.84 |
  | 15-22 m/s | 1,190 | 17.4 m | 54.6 m | −3.35 m | 21.77 |
  | 22-27 m/s | 1,181 | 17.4 m | 74.5 m | −3.45 m | 33.53 |
  | 27 m/s and up | 5,230 | 16.8 m | 91.7 m | −3.69 m | 44.22 |

  Over the whole set the plan covers 32 % of the distance the car drove
  (`path_length_ratio` 0.321) and nearly all of the error is along the road
  (longitudinal MAE 30.1 m, lateral 1.83 m). The model's own acceleration
  output is negative on 95 % of samples (median −0.36 m/s²). Stopped, it plans
  to stay put (0.08 m at 3 s below 1 m/s), so the nuScenes creep does not
  appear here.
- **And it bends right.** The 3 s point is 3.1 m right of straight ahead on
  average (median 3.5 m), where the car's own path ends 0.04 m right. Mode 2 of
  3 is chosen on 89 % of samples. The out-of-tree study saw the plan bend
  right on its segment under every rule it tried (y@3s −1.4 to −3.7 m), so the
  rule is not what bends it.
- **Not the adapter, and not one camera as such.** The feed matches the
  study's (above), and on METEOR's own scene the same one-camera feed beats
  constant velocity (1.01 against 1.37). The out-of-tree study looked at why it
  does not on comma2k19, on one urban segment: on the road pixels the model
  itself labels, its predicted depth is 3.4x the flat-road depth on the comma
  camera against 0.98-1.06x on its own cameras, its bird's-eye view holds road
  only about 20 m ahead, and its plan ends where that road ends. That is
  measured there, not here. Read the row as the released model's perception on
  a camera it never saw (a 65° phone at 1.22 m behind a windscreen, against
  its 98° and 30° lenses at about 1.9 m), not as a score of its planner.

```bash
paddock run --model meteor --bench comma2k19            # 10,165 samples
```

NAVSIM navtest, `results/navsim/meteor/20260914T234307-e56cfc` (v2 EPDMS, 12,146
scenarios, zero failures, 69 min) and `results/navsim_v1/meteor/20260915T005223-071c0b`
(v1 PDMS, a node-less replay of exactly those plans, 12 min, no GPU):

| | EPDMS / PDMS | NC | DAC | EP | TTC | lane keeping | comfort | extended comfort |
|---|---|---|---|---|---|---|---|---|
| v2 | **0.303** | 0.694 | 0.687 | 0.317 | 0.683 | 0.795 | 0.973 | 0.451 |
| v1 | **0.366** | 0.693 | 0.687 | 0.340 | 0.582 | 0.997 | — | — |

Above the constant-velocity floor (0.239 v2, 0.207 v1) and below every learned
model (VisionPilot 0.431 / 0.488, Drive-JEPA 0.843 / 0.892). That is a different
standing from nuScenes, where its L2 sits with Drive-JEPA's, and the difference
is worth reading carefully, because **a quarter of what NAVSIM scored here is
not the model's output.**

**The horizon.** METEOR plans 3 s. NAVSIM scores 4 s and `to_navsim_poses`
refuses a shorter plan rather than extrapolating behind the reader's back, so
this run set `METEOR_HORIZON_S=4.0`, which continues the model's final 0.5 s
segment to 4 s; the setting is in `run.json` as `model_env`. Measured on the
run's own plan dumps:

- **The extension is where the plans are unstable.** Over the 10,040 pairs of
  consecutive scored frames, the mean lateral disagreement between what one
  frame planned and what the next planned for the same instant is 0.52 m inside
  the model's own 3 s and **1.62 m in the extrapolated second**. Extended
  comfort, which is exactly that comparison, is 0.451 — the lowest in the
  matrix, below the constant-velocity floor's 0.567, though the model itself is
  deterministic and single-frame.
- **And it grows with the turn.** Bucketed by the lateral offset of the model's
  own 3 s point, the extension adds 0.5 m of further lateral for a straight plan
  and 4.5 m for a hard turn; drivable-area compliance falls 0.760 → 0.561 across
  those buckets and lane keeping 0.908 → 0.716. Lane keeping over the whole
  split, 0.795, is below the floor's 0.827: a straight line stays in its lane
  better than a curving plan continued off at a tangent.

**The picture.** `paddock render --bench navsim` draws it, into
`{roots.runs}/navsim/_compare/` unless `--out` says otherwise:

```bash
paddock render --bench navsim --model meteor,alpamayo15,openpilot
```

Eight panels, the navtest scenarios where these models' 4 s plans diverge most,
one per log, each over the map the scorer re-simulated against. METEOR's plan
tracks the others through its own three seconds, then kinks where the extension
begins and leaves on a tangent -- in the turning panels, across the lane centres
and off the drivable area. Drive-JEPA and VisionPilot cannot be drawn: their
full-split plan dumps predate scenario-token labelling, so no plan can be tied
to a scenario (`docs/benchmarks/navsim.md`).

**What is the model's own.** Two findings from nuScenes reappear here, on the
first 3 s that METEOR actually planned:

- **It stops short at speed.** Planned travel at 3 s against constant speed:
  12.0 m vs 10.9 at 2–5 m/s, 19.2 vs 20.6 at 5–9, 25.9 vs 31.8 at 9–14, 34.0 vs
  43.7 above 14. The plateau near 35 m is the same one KITScenes shows. Ego
  progress, 0.317, is the second-largest loss after collisions.
- **It creeps when stopped.** At ego speeds below 0.5 m/s it plans 1.6 m of
  travel in 3 s where constant speed gives 0.4 m — the nuScenes standstill creep,
  in a second dataset.

Collisions are the largest single loss: no-at-fault-collisions 0.694 against
0.94 and above for every other learned model, and barely over the floor's 0.681.
They are worst in the 2–5 m/s band (0.605), not at standstill (0.792), where
NAVSIM's at-fault test excuses being rear-ended.

None of this is reachable by the adapter. A 4 s score for this network needs a
4 s network: the checkpoint's own horizon is 3 s (`EGO_HORIZON = 6` at 0.5 s),
so the honest options are to report the extension as an extension, as here, or
to leave NAVSIM out.

neuroNCAP, `results/neuroncap/meteor/20260915T193036-d34559` (the full official
set: 20 scenario-scene pairs over 14 scenes, 10 runs each, n=200, zero failures,
8 h wall clock). No extrapolation here: neuroNCAP takes exactly the six
waypoints at 0.5 s that METEOR plans, so this is the model's own output end to
end.

| stationary | frontal | side | overall | collision rate |
|---|---|---|---|---|
| 2.44 ±2.42 | 3.20 ±2.39 | 2.50 ±2.50 | **2.64** | 48.0% |

Paired over the same seeded episodes, at two levels. The episode level treats
the 200 episodes as independent; they are 10 runs of each of 20 scenario-scene
pairs, and `docs/benchmarks/neuroncap.md` says the between-scene term dominates
this benchmark's error, so the pair-level column (each pair's ten runs averaged
first, n = 20) is the one the conclusions rest on.

| METEOR minus | mean | SE, 200 episodes | t | SE, 20 pairs | t |
|---|---|---|---|---|---|
| Alpamayo | −1.08 | 0.20 | −5.31 | 0.44 | −2.47 |
| OpenPilot | −0.96 | 0.21 | −4.62 | 0.51 | −1.89 |
| Drive-JEPA | −0.92 | 0.22 | −4.25 | 0.58 | −1.60 |
| Qwen-Drive | +0.16 | 0.19 | +0.83 | 0.41 | +0.39 |
| VisionPilot | +1.07 | 0.20 | +5.35 | 0.48 | +2.21 |

So: level with Qwen-Drive, marginally below Alpamayo and above VisionPilot, and
not clearly separated from Drive-JEPA or OpenPilot. The episode-level t-values
alone would have put it below all of the top three.

**It steers, and it does not brake.** Measured on every episode's recorded ego
poses and plans:

| episodes | collisions | ego speed, last minus first | mean peak lateral at the plan's 3 s waypoint |
|---|---|---|---|
| frontal (50) | 36% | +2.37 m/s | 5.17 m |
| side (50) | 50% | +1.82 m/s | 6.29 m |
| stationary (100) | 53% | +1.40 m/s | 4.99 m |

It gains speed in every scenario type. Head-on, swerving is the escape, and its
frontal lateral offset is second only to Alpamayo's 10.44 m, so frontal is its
best column. Against an obstacle already standing in the lane, swerving while
accelerating is not enough, and stationary is its worst. That matches NAVSIM,
where its no-at-fault collision rate sits at the constant-velocity floor, and
nuScenes, where it creeps into stopped traffic. The peak lateral offset, read at
the 3 s waypoint, and the collision rate reproduce the method in
`docs/models/qwendrive.md` exactly for all five models there (1.22 m and 86%
for Qwen-Drive, 10.44 m and 32% for Alpamayo). Read across all six waypoints
instead, VisionPilot's frontal figure would be 3.65 m against the documented
1.91 m; METEOR's values are the same either way. The speed column there does not reproduce under last-minus-first,
so the speed figures here are this measure throughout, not that one.

**The spread is the headline as much as the mean.** Four pairs score a full
5.0 (frontal-0346, side-0108, side-0921, stationary-0108), three score exactly
0 (side-0103, side-0278, stationary-0966), and two more stay below 0.15
(stationary-0796 0.12, stationary-0099 0.14). A single run scores exactly 5.0 or at most
4.0, so a per-pair mean near 2.5 is a mix of clean escapes and full-speed
impacts.

Bench2Drive, `results/bench2drive/meteor/pooled50-40c29d24c37d542a-867d57`
(closed loop in CARLA 0.9.15: the 50 routes every other row pools, one run each,
paddock's six cameras, a plan every 0.2 s at its declared 5 Hz, and AutoE2E's
early-stop rules: a route ends after 30 s without 3 m of progress or at 4x the
expert's time and is scored as it stood; `valid: true`; 46 min wall clock on six
workers over two A100s, 29 Oct 2026, no node on the CPU):

| routes | DS | DS median | RC (%) | infraction penalty | routes with a collision | never left the start | completed |
|---|---|---|---|---|---|---|---|
| 50 | **1.05** | 0.00 | 6.30 | 0.90 | 7 | 43 | 2 |

Constant velocity at 3 m/s scores 48.31 on the same routes, VisionPilot 47.82,
AutoE2E 15.30 (its own 10 Hz, the same early stop).

**It does not start.** Every Bench2Drive route begins at a standstill, and on 43
of the 50 the car never got going (on 42 it never passed 0.1 m/s). Standing
still, METEOR plans a 3 s path that ends 0.04-0.43 m ahead (median over routes
0.17 m; on nuScenes' stopped anchors its plan creeps 0.95 m on average), and its own
acceleration output is negative (-0.45 to -0.12 m/s^2, median -0.27). The shared
controller brakes when a plan asks for less than 0.4 m/s, and it braked on
97-100 % of those routes' ticks. The stuck rule ended 30 of them at 30 s and the
over-time rule 1; the other 12 start 27-31 m short of a light that was red for
40-50 of their first 60 s, which pauses the stuck clock, so Bench2Drive's own
blocked test ended them at 60 s. The six cameras the model was shown on route
1711 are the scene as CARLA rendered it, in colour, the road ahead clear, so the
hold is the network's plan, not a blank input.

**When it moves, it collides.** It left the start on 7 routes and collided on
every one: 3364 is driven to its end with 16 collisions with static objects, so
its penalty multiplier is 0.0001 and it scores 0.01; 24252 is completed with one vehicle
collision (6.41); the best route is 24333 at 14.77.

**What differs from where it drove before, argued and not separated.**
neuroNCAP's scenarios start with the car already moving, and there METEOR drove
(and did not brake); every Bench2Drive route starts from a standstill. Its
cameras were TIER IV's (98 deg lenses, side cameras pitched about 25 deg down,
about 1.9 m up, in left-hand traffic), where paddock's rig is 70 deg (back
110 deg), level, 1.5 m up, in right-hand traffic. An open-loop replay of one
Bench2Drive clip outside this repository found METEOR's 2D detection carries over
to these cameras and the depth behind its BEV lift does not. And it gets no
driving command. Nothing here measures which of these holds it still.

Over all 220 routes, `results/bench2drive/meteor/full220-836dda734659a86c-dda00e`
(7 Nov 2026, the 220-route protocol: the same rate, cameras and early stop, up to
three attempts a route, the four routes set aside for every model left out): DS
**1.92** over 216 routes, RC 5.40%, infraction penalty 0.92, a collision on 26 routes.
189 routes end under 1% route completion, and one, 26394, is completed, with 12 collisions with
static objects. The tables, and Bench2Drive's own scores for the record, are in
[Bench2Drive — 220 routes](../benchmarks/bench2drive.md#bench2drive--220-routes).

AlpaSim, `results/alpasim/meteor/legacy-meteor_011_published_30scenes` (30
scenes, one rollout each, six cameras at 20 Hz) and `…_first10` (the first ten
of those 30 runs, not a second sweep):

| scenes | `scene_score` | progress | at-fault collisions | any collision | off-road |
|---|---|---|---|---|---|
| first 10 | 0.6869 | 0.6073 | 0 / 10 | 3 / 10 | 0 / 10 |
| 30 | **0.4232** | 0.4301 | 1 / 30 | 10 / 30 | 5 / 30 |

**This row did not go through the paddock node.** It was made by a native
AlpaSim driver, `meteor-0.1.1`, written outside this repository and not tracked in
it, and ingested from its recorded runs. `meteor` does not declare
`alpasim` in `model.yaml`. What that driver does, so the number can be read:

- **The graph and the pixels are the ones above.** The released
  `meteor_v157c3Z.onnx` (sha256 `50397b4d…`, logged by each of the 30 runs),
  raw uint8 into the graph's own `/255`, no mean/std normalisation, on the CUDA
  provider.
- **Six cameras, rendered as pinholes.** Front wide, cross left and right
  (in METEOR's front-left and front-right slots), rear left and right, and
  the front telephoto fill six slots; the rear wide
  and rear telephoto slots are absent and fed as zeros with a neighbour's
  calibration. This is a different six from the nuScenes six the node gets,
  which has a rear wide camera and no telephoto. Each camera is rendered at
  the narrower of METEOR's own field of view and the recorded one (97.70° for
  the three wide cameras, about 120° as recorded), with no distortion and the
  bonnet mask off. See [docs/benchmarks/alpasim.md](../benchmarks/alpasim.md).
- **The mode is not the argmax.** The driver uses the selector of METEOR's
  deployed runtime: the previously chosen mode gets +0.35 on its logit, and a
  non-straight mode is taken only if it leads the straight one by 1.0. The
  node, and every other METEOR cell in the matrix, takes the plain argmax.
- **A fresh plan every fourth step.** The simulator steps at 20 Hz and the
  graph is run at 5 Hz; the three steps between reuse the last plan, re-timed.
  The six waypoints are passed on as the model emitted them, with headings
  derived from them. The steer, accel and brake outputs are logged and not
  driven. No plan is extended: the simulator's controller looks 2 s ahead and
  the plan is 3 s.
- **No driving command**, as everywhere else.

The simulator drives the recorded path itself for the first 3 s of each scene
and then hands over to the model.

What the 30 runs show, from the per-scene scorer output:

- **The first ten scenes and the twenty extra are far apart**: 0.6869 against
  0.2913. All five off-road events and the one at-fault collision are on the
  twenty extra scenes, and each zeroes its scene. Four of the ten score 0.98
  or more; one of the twenty does.
- **Low progress, not only failures.** Of the 24 scenes with no off-road event
  and no at-fault collision, 12 score below 0.5 because the car covered less
  than 40 % of the recorded distance inside the scored window. Seven of those
  had a collision that was not the model's fault, which ends the window. Three
  more windows ended early without one (`04749bb9` and `2e1ce314` after less
  than a second of scored driving). On the last two, `bae463d3` and `d2715023`, the
  window ran its full length and the car covered under 15 % of the distance.
- **Not explained yet.** Which of the low-progress scenes are the model
  stopping short at speed (the plateau seen on KITScenes and NAVSIM) and which
  are something else has not been separated. An earlier ten-scene run of a
  variant of this driver showed the simulator's controller swerving when
  handed a plan much slower than the car, at 15 m/s and 24 m/s; whether
  that is what ends the fast scenes here has not been checked.
- **One rollout per scene.** The standard deviation across the 30 scenes is
  0.37, so the mean is loose.

The number is ranked with the other 30-scene rows and read against the same
constant-velocity floor (0.4876): camera count is not a reason to separate rows.

**One episode misled.** The pipeline smoke,
`results/neuroncap/meteor/20260915T192353-6ed840`, drove stationary-0103 once
and scored 0.71 at high impact speed; that pair averages 4.07 over its ten
runs. Scene 0103 misled this repo in the same way in September.

