# METEOR — Adapter Notes

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
the checkout is imported: the adapter runs the released ONNX.

## Weights and environment

Release `v1.0` on Hugging Face, `AutowareFoundation/meteor`:

| file | what | used by paddock |
|---|---|---|
| `meteor_v157c3Z.onnx` | plain ONNX, opset 17, no custom ops, camera-only, history baked out, uint8 input | **yes** |
| `meteor_v157.pt` | PyTorch checkpoint (`--model v52`), has the driving-command input | no |
| `meteor_v157.param.yaml` | camera order, resolution, BEV grid, output tensors | reference |
| `SHA256SUMS` | checksums | `setup.sh` verifies against it |

`setup.sh` builds `<repos>/METEOR/.venv` with `uv` (Python 3.11,
`onnxruntime-gpu`, the `nvidia-*-cu12` wheels for the CUDA runtime, cuDNN and
cuBLAS, plus what the paddock node needs) and downloads the three files to
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
always absent in paddock. Every other benchmark supplies the nuScenes six;
KITScenes did not until #40, whose harness handed over only `camera_ring_front`
although the dataset records a six-camera ring with calibration for each. The
run made under that harness is kept as an ablation (below): seven absent slots
is a heavier ablation than anything the network saw in training, and it cost
2.9 m of ADE.

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
nuScenes (1600x900) is a pure 0.48 scale.

**Ego frame.** The BEV lift assumes the ego origin is on the road (`z = 0`).
nuScenes and NAVSIM satisfy that; KITScenes' harness shifts its roof-pod
reference down by the pod height for the same reason.

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

## Running it

```bash
PADDOCK_REPO=<repos>/METEOR PADDOCK_WEIGHTS=<weights> \
  bash src/paddock/adapters/models/meteor/setup.sh
paddock doctor --model meteor
paddock run --model meteor --bench kitscenes --limit 5 --dry-run
paddock run --model meteor --bench kitscenes --limit 5
paddock run --model meteor --bench kitscenes
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
1.534, OpenPilot 1.599, Drive-JEPA 2.364. A second run reproduced all 223
predictions exactly.

**Front camera only, kept as an ablation:**
`results/kitscenes/meteor/20260914T185351-5e5223`, made before #40 when the
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
