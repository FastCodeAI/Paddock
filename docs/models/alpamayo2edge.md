# Alpamayo 2 Edge

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge) at `02f55a7` |
| Adapter | [`adapters/models/alpamayo2edge/`](../../src/paddock/adapters/models/alpamayo2edge/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `CAM_FRONT_TELE` |
| Plan | waypoints, 0.5 s apart, 6.4 s ahead |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [navsim](../benchmarks/navsim.md), [neuroncap](../benchmarks/neuroncap.md), [bench2drive](../benchmarks/bench2drive.md), [comma2k19](../benchmarks/comma2k19.md) |

<!-- /summary -->

NVIDIA Alpamayo 2 Edge (November 2026; code Apache-2.0, weights OpenMDW-1.1) is a
2.65 B-parameter reasoning vision-language-action model. A Cosmos3-Edge (Nemotron)
VLM writes a short driving rationale, then a flow-matching expert reads the VLM's
cache and decodes 64 future poses, 0.1 s apart. Vendor repo: `alpamayo2-edge` in
`repos.yaml`; weights `Alpamayo2-Edge/` under `[roots.weights]`
([nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge), not gated,
5.3 GB, revision `f8951d095de3589b3face1dec3f4394c8ecedb38`). The checkpoint is
self-contained: tokenizer, processor and backbone are all in that one directory, so
unlike Alpamayo 1.5 there is no separate VLM to place.

```bash
paddock setup --model alpamayo2edge                          # uv sync --locked + fastapi/uvicorn
paddock doctor --model alpamayo2edge
paddock run --model alpamayo2edge --bench kitscenes --limit 5      # smoke
paddock run --model alpamayo2edge --bench kitscenes
paddock run --model alpamayo2edge --bench comma2k19
```

## What the model consumes

From upstream's own loader (`load_physical_aiavdataset.py`, `sample.py`), which the
adapter reproduces rather than Alpamayo 1.5's input:

| | |
|---|---|
| cameras | six, by upstream id: 0 cross-left 120°, 1 front-wide 120°, 2 cross-right 120°, 3 rear-left 70°, 5 rear-right 70°, 6 front tele 30°. Id 4 (rear tele, straight back) is not an input. Upstream refuses any other set. |
| frames | 22: ids 1 and 6 at −2, −1, −0.5, −0.1, 0 s; the other four at −1, −0.1, 0 s. Each image's capture time is written into the prompt (`<t=-0.5s>`). |
| ego history | 20 poses at 0.1 s, −1.9 .. 0 s, positions and full rotations in the ego frame at t0 (57 history tokens). Alpamayo 1.5 takes 16. |
| route | one navigation sentence, required ("Turn left in 30m", "Continue straight") |
| output | 64 poses at 0.1 s to 6.4 s; paddock takes every fifth (0.5 .. 4.0 s) |
| sampling | upstream CLI defaults: seed 43 per call, top_p 0.98, temperature 0.6, one reasoning trace, 256 tokens, 10 flow steps. Seeded, so a re-run reproduces each plan exactly (checked: two KITScenes runs, max difference 0.0 m). |

## How each source fills the six slots

| Slot | Source camera | Crop |
|---|---|---|
| 0 cross-left 120° | `CAM_FRONT_LEFT` | widest 16:9 window, or 120° if the sensor is wider |
| 1 front-wide 120° | `CAM_FRONT` | same |
| 2 cross-right 120° | `CAM_FRONT_RIGHT` | same |
| 3 rear-left 70° | `CAM_BACK_LEFT` | exactly 70° where the sensor is wider (KITScenes' ~87°), else widest |
| 5 rear-right 70° | `CAM_BACK_RIGHT` | same |
| 6 front tele 30° | `CAM_FRONT` | exactly 30°: narrower than every source, so this slot is geometrically exact |

Matched by bearing: every benchmark's `CAM_FRONT_LEFT/RIGHT` is its forward-diagonal
view and `CAM_BACK_LEFT/RIGHT` its backward-diagonal one (nuScenes about ±110°,
KITScenes ±120°, NAVSIM `l2`/`r2`), which is where the 70° rear cameras look.
`CAM_BACK` is never rendered or read. Every crop is centred on the principal point
(Alpamayo 1.5's `camera_warp.crop_box`) and resized to 1920×1080, the PhysicalAI-AV
frame size, after which upstream's own resize to each slot's pixel budget (576×320
front and tele, 416×224 the rest) runs unchanged.

**Not verified:** the exact mounting yaw of the 70° rear cameras in NVIDIA's rig.
The PhysicalAI-AV calibration is gated and was not readable from here.
`diagnostics()` reports each source camera's yaw from its `camera2ego`, for checking
against it once it is.

**Frame times are the real ones.** Each slot takes the recorded frame nearest the
time it asks for, and the prompt is given that frame's actual time. On nuScenes'
2 Hz keyframes the −0.1 s slot holds the 0 s frame, labelled 0 s, not a frame
claimed to be 0.1 s old.

**A missing camera is black, never a copy.** Upstream refuses fewer than six
cameras, so comma2k19 (one forward camera) runs with slots 0, 2, 3 and 5 black and
only the front and telephoto real, as Qwen-Drive does for its side views (#146).
Copying the front view into a side slot would tell the model there is a view at
that bearing. `diagnostics()` names the black slots. A comma2k19 score is therefore
a two-of-six-camera measurement.

## Speed settings (ablations, never ranked)

Two settings, both off by default, for measuring what the model's own speed-ups cost.
Setting either from the shell (they are in `env_passthrough`) records the run
`valid: false`, and its `run.json` notes name the setting:

| Setting | Default | Ablation |
|---|---|---|
| `ALPAMAYO2_EDGE_FLOW_STEPS` | `10` (upstream's) | fewer flow-matching steps for the action expert; the expert's time is about 0.013 s per step |
| `ALPAMAYO2_EDGE_REASONING` | `model` (it writes its own) | `nominal`: force the exact tokens it writes when it chooses `<NOMINAL>`, so decode is always 7 steps; `none`: no reasoning at all |

```bash
ALPAMAYO2_EDGE_REASONING=nominal ALPAMAYO2_EDGE_FLOW_STEPS=2 \
  paddock run --model alpamayo2edge --bench kitscenes
```

The forced `<NOMINAL>` is the model's own token sequence, `>\n` included, not the
tokenised text (`policy.FORCED_REASONING_IDS`). Forcing works by wrapping upstream's
`LogitsProcessorList` at load, so upstream code is unchanged. `none` costs ADE
because it removes cache entries the action expert reads; `nominal` does not.

## Speed

On an H200, with an 8-thread decode-and-crop pool, a `paddock run` on KITScenes spends
about 0.6 s per sample in the model and 3 s on CPU preparation (node log timings). Most of the preparation is decoding
seventeen 3504×2272 JPEGs and upstream's own resize of 22 images. Peak GPU
memory is 6.2 GB.

## Reference numbers

Single runs, 2026-11-10, H200. Seeded, so a re-run gives the same plans.

| Benchmark | Run | n | ADE mean | ADE median |
|---|---|---|---|---|
| KITScenes (`kitscenes_e2e_final` list) | `20261110T215043-e7b757` | 223 | 0.756 m | 0.563 m |
| comma2k19 (80-keyframe dev list; front and telephoto real, four slots black) | `20261110T220410-d98c07` | 80 | 0.575 m | 0.448 m |

On the same 223 KITScenes keyframes, Alpamayo 1.5's two runs scored 1.014 and 0.887 m.
Paired by keyframe, Edge is 0.257 and 0.130 m lower (bootstrap 95% intervals −0.375 to
−0.148 and −0.227 to −0.033). There is no like-for-like comma2k19 comparison yet: the
Alpamayo 1.5 runs on these 80 keyframes predate #142, which moved comma2k19's ground truth
into the car's frame, so the two models were scored against ground truths a median 0.59 m
apart. A new checkpoint with
undisclosed training data, not a controlled change, so the difference is not attributable
to any one part of the design.

Model compute per sample on KITScenes, idle H200, Alpamayo 1.5 timed straight after on the
same card and samples, the first (warm-up) sample left out of both: 0.35 s against 0.70 s
(vision 0.05 / 0.08, prefill 0.04 / 0.12, reasoning 0.12 / 0.30, expert 0.14 / 0.21),
6.2 GB against 23.2 GB peak. Averaged with the warm-up sample in, Edge is 0.42 s.
