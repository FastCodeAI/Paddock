# WA-JEPA

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [AFARI-Research/WA-JEPA](https://github.com/AFARI-Research/WA-JEPA) at `bec2966` |
| Adapter | [`adapters/models/wajepa/`](../../src/paddock/adapters/models/wajepa/) |
| Environment | uv, Python 3.10 |
| Cameras | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK` |
| Plan | waypoints, 0.5 s apart, 4 s ahead |
| Runs on | [neuroncap](../benchmarks/neuroncap.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

## What it is

[WA-JEPA](https://github.com/AFARI-Research/WA-JEPA) (AFARI Intelligent Drive,
arXiv 2608.20974) is a V-JEPA-native world-action model. A V-JEPA 2.1 ViT-L/16
encoder reads four cameras over four history frames at 2 Hz, and a 12-layer
joint flow predictor denoises future scene tokens and the ego trajectory in one
latent space. Output: eight (x, y, heading) waypoints at 0.5 s, ego frame,
x forward and y left — paddock's own grid, so nothing is resampled.

Training is two stages: future scene prediction on nuPlan video (trajectory
head off), then NAVSIM `navtrain` with the trajectory head on. The released
weights are stage 2. Upstream reports 91.7 EPDMS on NAVSIM v2.2 navtest (mean of
ten inference seeds) and 0.4462 HD-Score zero-shot on HUGSIM's 436 closed-loop
scenarios.

Pinned in `repos.yaml` at `bec2966` (`main`, 2026-09-04). No patches.

## Weights and environment

`model_state_dict.pt` from Hugging Face `AFARI-Research/WA-JEPA`, 1,575,763,741
bytes, to `<weights>/wa-jepa/`. It holds the whole model (788M parameters with
the EMA encoder), so the separate V-JEPA 2.1 pretrained encoder upstream's
NAVSIM presets load first is not needed; `state.pt` is trainer state and is not
fetched. The checkpoint loads **strictly** — a partly restored trajectory head
would still emit plausible waypoints, so a mismatch must fail.

`setup.sh` builds `<repos>/WA-JEPA/.venv` with `uv`: Python 3.10 (upstream's),
`torch==2.8.0` + `torchvision==0.23.0` from the cu128 index, then timm, einops,
omegaconf, headless OpenCV and the paddock node's fastapi/uvicorn/pillow. Two
deliberate departures from upstream's conda recipe:

- **No NAVSIM or nuPlan devkit.** The adapter imports only the model package
  (`models/`, `third_party/vjepa2/`, `training/checkpoint.py`) and one helper
  from upstream's closed-loop client. NAVSIM pins torch 2.0.1, which has no
  kernels for a Blackwell (sm_120) card.
- **torchvision is pinned next to torch.** WA-JEPA never imports it, but timm
  requires it, and installed in a later step it resolves from PyPI to a newer
  torch built for CUDA 13, replacing the cu128 torch and unpacking cu13 NVIDIA
  wheels over the cu12 ones. That happened on the first build.

Settings, all in `model.yaml` `runtime.env`:

| variable | default | meaning |
|---|---|---|
| `WAJEPA_CHECKPOINT` | `{weights}/wa-jepa/model_state_dict.pt` | the released stage-2 weights |
| `WAJEPA_CONFIG` | `{repo}/configs/wa_jepa_hugsim.yaml` | upstream's closed-loop preset: the NAVSIM architecture with the pretrained-encoder load off |
| `OMP_NUM_THREADS` | `2` | torch's CPU pool; the node is GPU-bound |

Measured on the reference box (RTX PRO 6000 Blackwell): load 8 s, 3.2 GB
resident, 4.8 GB peak, ~320 ms per plan including the CPU resize of four
1600x900 frames.

## Input contract and how paddock fills it

The adapter follows upstream's HUGSIM client (`close_loop/hugsim_planner.py`),
which already drives the model from rendered nuScenes-style cameras without
NAVSIM's dataclasses. What it does, and where paddock's wire differs:

| input | value | source |
|---|---|---|
| `history_images` `[1, 4, 4, 3, 256, 512]` | slots cam_l0, cam_f0, cam_r0, cam_b0 = CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK; RGB, `INTER_AREA` resize, no crop, scaled to [-1, 1] | the training loader's transform; the slot mapping is upstream's HUGSIM one |
| `history_trajectory` `[1, 4, 3]` | (x, y, yaw) of the four history poses relative to the newest, from `ego2world` | upstream's `convert_absolute_to_relative_se2_array`, imported, not restated |
| `ego_status` `[1, 8]` | NAVSIM command one-hot (left, straight, right, unknown), vx, 0, ax, 0 | vy and ay are zero, as upstream's HUGSIM client sends them |

- **Camera order is the contract.** The model identifies a camera by an
  embedding indexed by list position, so a swapped pair raises nothing.
  `load()` refuses a config whose `camera_names` differ from the adapter's slots.
- **No crop.** NAVSIM's 1920x1080 and nuScenes' 1600x900 are both 16:9, and the
  front fields of view match to a degree (63.7 vs 64.5), so the plain resize
  training used carries over.
- **Commands.** paddock's 0=RIGHT 1=LEFT 2=STRAIGHT is HUGSIM's order, so the
  map is upstream's `command_map` [2, 0, 1]; FOLLOW_REFERENCE goes to "unknown".
- **Rate.** neuroNCAP steps at 0.1 s; history is picked by timestamp at 0.5 s
  spacing, as in the drivejepa adapter, so it spans 1.5 s at any step rate.
  Until 1.5 s have passed the oldest frame is repeated (upstream's warm-up too);
  `/diagnostics` reports `plans_on_padded_history`.
- **Precision.** fp32 weights under bf16 autocast, as trained.
- **Determinism.** The flow sampler's noise generator is re-seeded
  (`flow_inference_seed: 1`) on every call: identical input, identical plan
  (verified, max difference 0.0). Upstream's 91.7 is a ten-seed mean, so a
  single seed is one draw from it.

`paddock verify --model wajepa` passes all nine policy checks inside the model's
own interpreter.

## Results

Only the neuroNCAP smoke below has been run. NAVSIM, KITScenes and nuScenes are
not run yet, and `benchmarks:` in `model.yaml` lists only neuroncap until they are.

### neuroNCAP smoke, scene 0103, one run per scenario

Run `20261015T234134-916e11`, `valid: false` because it is 3 of the 20 official
pairs. n = 1 per scenario, so this shows the adapter drives end to end. It is
not a score.

| scenario | NCAP score | collision | impact speed (m/s) |
|---|---:|---|---:|
| stationary-0103 | 5.00 | no | — |
| frontal-0103 | 0.00 | yes | 13.4 |
| side-0103 | 1.68 | yes | 4.7 |
| mean | 2.23 | 2 of 3 | |

The node's own report at the end of the run: history span 1.5 s (target 1.5 s),
8 of 44 plans on a padded warm-up history.

**Frontal: the model sees the car and still holds speed.** The ego drives at
7.9–8.5 m/s for the whole 3 s, and every plan asks for 7.5–8.5 m/s until impact.
The impact speed is the closing speed with the oncoming car. Because that pattern
also fits a model that cannot see (Drive-JEPA's camera-blind bug, see
[drivejepa.md](drivejepa.md)), the recorded CAM_FRONT frames of this episode were
replayed through the adapter. The other three cameras were blank, and plans were
made at t = 1.5, 2.2 and 2.9 s:

| t | real frame, 3 s endpoint (x, y) | black frame | mirrored frame |
|---|---|---|---|
| 1.5 s | (24.2, -0.59) | (20.5, 0.26) | (23.6, 0.63) |
| 2.2 s | (24.6, 0.20) | (21.5, 0.28) | (24.2, -0.12) |
| 2.9 s | (25.1, 0.47) | (21.8, 0.30) | (24.4, -0.62) |

The image moves the plan by 3.1–3.8 m, and mirroring it flips the lateral sign at
two of three steps, so pixels reach the planner. With the real frame it plans
*faster* than with a black one. That replay ran on CPU, so its numbers are close
to the GPU node's but not bit-identical.

Waypoints come back in steps of 0.125 m at 16–32 m. That is bf16 resolution
from the autocast forward pass, the same as upstream's HUGSIM client, which
casts the output to fp32 only afterwards.
