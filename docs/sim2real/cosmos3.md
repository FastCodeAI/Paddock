# Sim2real with Cosmos 3 Nano

Bench2Drive renders a game. Every model in paddock was trained on camera video of the real
world, so every closed-loop number carries an unknown appearance penalty, and nobody could say
how large it was per model. This is the measurement: convert CARLA frames into real-looking ones
while keeping the geometry, ask each model for its plan on both versions of the *same* state,
and read off the difference.

Run on 16–17 September 2026, route **2802-Town12-InvadingTurn** (rain, fog, wet road; an oncoming
car invades the ego lane about 3 s and 6 s after the scenario trigger). Scripts:
[`tools/sim2real/cosmos3/`](../../tools/sim2real/cosmos3/README.md).

**Everything here is diagnostic.** A model driving on converted pixels is not a Bench2Drive
score, nothing on this page is in `results/`, and none of it belongs in the leaderboard tables.

---

## The pipeline

```
paddock run --model constant_velocity --bench bench2drive --scene 2802
  PADDOCK_SEGMENT=<spec with "mode":"capture">        capture, in CARLA, on the workstation
  PADDOCK_CAMERAS_OVERRIDE=<the target model's rig>
        |
        v  captures/<model>/
        |    frames/<cam>/carla/%05d.jpg      the CARLA frame
        |    frames/<cam>/depth/%05d.png      inverse depth, same mount and FOV
        |    ticks.jsonl                      ego2world, canbus, command, nav, speed_limit,
        |                                     camera_to_world per camera, actor boxes
        |    map.json                          static map layers + the camera rig
        v
  wsm_batch.py      Ludus (ludus_renderer, flashdreams) -> frames/<cam>/wsm/%05d.jpg     pod, ~7 ms/frame
  cosmos_batch.py   Cosmos 3 Nano video2video, edge + depth + wsm -> converted/<m>/<cam>/ pod, ~1.5 s/frame
        |
        v
  openloop_replay.py --model M --capture captures/M                  -> plans/M_carla.jsonl
  openloop_replay.py --model M --capture captures/M --frames CAM=... -> plans/M_cosmos.jsonl
  openloop_replay.py --model M --capture captures/M (again)          -> plans/M_carla2.jsonl  (the floor)
        |
        v
  openloop_compare.py     plan vs plan over identical states
```

The capture replays a model's own recorded 20 Hz poses by teleport, so the ego is on rails and
the world is reproducible. The replay posts each captured tick to a paddock model node with
**the same wire body the closed-loop agent posts** — same calibration, same canbus, same route
command — and swaps only the pixels. Nothing drives: the two plan sequences are compared
directly, which is what makes the comparison clean. The closed-loop version of the same idea
(below) could not separate the converter's effect from the controller's.

The third replay is the point of the table. Asked twice on *identical* CARLA frames, a
deterministic model returns identical plans and its floor is 0; a model that samples returns
different plans and its floor is whatever its own sampling does. A Cosmos gap near the floor
says nothing about pixels.

### Why three controls

`edge` is derived from the CARLA frame itself and holds the layout. `depth` is CARLA's depth
camera on the same mount, exported as inverse depth. `wsm` (world scenario map) is the Ludus
rasteriser's map render — lane lines with dash patterns, magenta road boundaries, filled
coloured cubes for every actor — the same raster Cosmos' own tooling consumes. Without the map
control the diffusion prior moves cars and lane structure around; with it, geometry follows
CARLA per pixel.

## The settings that were used

`tools/sim2real/cosmos3/specs/cosmos_live_spec.json`, unchanged between the closed-loop runs and
the open-loop conversion:

| Spec field | Value |
|---|---|
| `model_mode` | `video2video` |
| checkpoint | `nvidia/Cosmos3-Nano` (16B, single GPU) |
| `edge` | derived from the vision frames, preset threshold `medium`, weight **0.5** |
| `depth` | supplied control (CARLA inverse depth), weight **0.5** |
| `wsm` | supplied control (Ludus map render), weight **1.0** |
| `guidance` | **3.0** |
| `control_guidance` | **1.5** |
| `seed` | **2026** (`cosmos_batch` offsets it per window: `seed + first tick`) |
| prompt | `prompts/prompt_rain.json` — 2802 is rain and fog, so the prompt matches the scene |
| negative prompt | `prompts/negative_prompt.json`, the framework's generic negative |
| resolution | `480` at aspect `16,9`, i.e. 832×480 |

Batch conversion (`cosmos_batch.py`, the open-loop study):

| Knob | Value | Why |
|---|---|---|
| `COSMOS_CHUNK` | **61** frames per window | one denoising pass over the whole window, so consistency inside it is native |
| `COSMOS_NCOND` | **8** frames of overlap | the previous window's output conditions the next as clean context |
| `COSMOS_STEPS` | **12** | 20 steps cost 1.65× for no visible gain on these frames |
| `COSMOS_VIS_M` | **50** m | see the fog artefact below |

Throughput, from the 13 `conversion.json` records the run wrote: **4,806 frames in 1.92
GPU-hours, mean 1.43 s per frame at 480p** on an RTX PRO 6000 Blackwell (96 GB), per-camera
0.45–1.96 s/frame. **13 camera sequences — 5 models, their whole rigs — in about 95 minutes of
wall clock** on two GPUs, which is the ~57 min of GPU time plus the map renders and the wait for
each capture to finish uploading. `cosmos_batch` resumes from disk and keeps every output,
because the last window slides back to end on the final tick and re-saves a few earlier frames
so they stay consistent with what follows.

| Rig | cameras converted | frames each |
|---|---|---:|
| VisionPilot | `CAM_FRONT` | 320 |
| DriveJEPA | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK` | 366 |
| Alpamayo 1.5 | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` | 370 |
| openpilot | `CAM_OP_ROAD`, `CAM_OP_WIDE` | 371 |
| QwenDrive | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` | 390 |

The per-tick service (`cosmos_service.py`, the closed-loop runs) uses the same spec with a short
clip instead: `COSMOS_STEPS=12 COSMOS_CHUNK=5 COSMOS_NCOND=4 COSMOS_REANCHOR=40`, ~9 s per tick
alone and ~18 s with two converters sharing a GPU.

## Results: what each model plans on real-looking pixels

Route 2802, from the start of each model's recorded run to 15 s after the cut-in trigger. Metric
is the gap between the two plans' 3 s waypoint, over the ticks after the trigger. **Floor** is
the same metric with the model asked twice on identical CARLA frames.

| Model | ticks after trigger | plan gap at 3 s, mean / median (m) | floor, mean / median (m) | ticks over 1 m | planned speed, CARLA → Cosmos (m/s) | reading |
|---|---:|---:|---:|---:|---:|---|
| VisionPilot | 271 | **0.67** / 0.24 | 0 / 0 | 17 % | 2.13 → 2.05 | small, real |
| QwenDrive | 301 | **1.36** / 1.19 | 0 / 0 | 54 % | 0.97 → 0.80 | small, real; plans slower |
| DriveJEPA | 301 | **2.09** / 2.02 | 0 / 0 | 64 % | 2.30 → 2.05 | real; plans slower |
| Alpamayo 1.5 | 301 | **3.23** / 2.55 | **2.55 / 1.79** | 77 % (floor 71 %) | 3.80 → 3.62 | mostly its own sampling |
| openpilot | 301 | **6.11** / 5.28 | 0 / 0 | 75 % | 4.84 → **5.90** | largest, and systematic |

The other horizons, same runs and the same after-trigger ticks. `t0` is the trigger tick and
`ticks` the whole captured segment; every metric column is after the trigger except the
inference times, which are the whole-segment means:

| Model | t0 | segment ticks | L2 at 1 s | L2 at 2 s | L2 at 3 s mean / median / p90 | ADE | lateral abs diff at 3 s | inference CARLA / Cosmos (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisionPilot | 50 | 320 | 0.31 | 0.35 | 0.67 / 0.24 / 2.00 | 0.41 | 0.10 | 115 / 89 |
| QwenDrive | 90 | 390 | 0.12 | 0.47 | 1.36 / 1.19 / 2.85 | 0.52 | 0.19 | 1470 / 1025 |
| DriveJEPA | 66 | 366 | 0.82 | 1.51 | 2.09 / 2.02 / 4.11 | 1.30 | 0.20 | 1004 / 1422 |
| Alpamayo 1.5 | 70 | 370 | 0.41 | 1.55 | 3.23 / 2.55 / 7.10 | 1.42 | 0.94 | 1537 / 2473 |
| openpilot | 71 | 371 | 1.66 | 3.71 | 6.11 / 5.28 / 12.99 | 3.29 | 0.06 | 805 / 593 |

Every figure in both tables is read from the `after_trigger` (or, for inference time, the
top-level) block of `all_summaries.json` and rounded to two decimals. Distances are metres.

**Pixel sensitivity, most to least: openpilot > DriveJEPA > QwenDrive > VisionPilot > Alpamayo
(within its own noise).**

### openpilot is the outlier, and it is not noise

openpilot's floor is exactly 0 — it is deterministic on identical frames — so all 6.11 m is the
pixels. Two things happen on the converted frames: it plans **5.9 m/s where the CARLA-frame plan
plans 4.8**, and it keeps rolling where the CARLA-frame plan stops. That is the same mechanism
twice measured elsewhere: openpilot has no speed input and infers ego motion from
frame-to-frame change, so any change in how much the image moves between frames moves its whole
plan. The 16 Sep closed-loop run on the same route showed the same sign (it accelerated to
5.8 m/s on converted frames where it crawled at ~1 m/s on CARLA frames), and a July-style
cosmos-transfer 2.5 conversion on route 3813 pushed it to 36 m/s against a true 8.3 m/s.

### Alpamayo's gap is mostly its own sampling

Alpamayo samples one trajectory and is not seeded, so asked twice on the *same* CARLA frames it
already differs by 2.55 m mean / 1.79 m median. Against that, 3.23 / 2.55 on converted frames is
a small excess. The honest statement is that this experiment cannot resolve Alpamayo's pixel
sensitivity at n = 301 ticks; it would need repeats, not a better converter.

## Known artefacts of the converter

These are properties of Cosmos 3 Nano on this input, not of the pipeline, and they were accepted
rather than fixed — the study needed a *consistent* conversion, not a perfect one.

**A car bonnet and windscreen along the bottom edge of every front view.** Present in every
frame of every front camera. Tried and rejected: a "no bonnet" prompt, a roof-camera prompt with
no "dashcam" wording, seed 7 instead of 2026, depth weight 0.25, edge weight 0.8, control
guidance 2.0 (`specs/cosmos_nohood_spec.json`, `cosmos_roof_spec.json`, `cosmos_d25_spec.json`,
`cosmos_cg2_spec.json`). None of the three controls contains a bonnet, so it is a model prior.
The corresponding artefact on the side views is a **fender across the bottom third of the frame**.

The untested hypothesis is the depth encoding: the control is inverse depth normalised per frame
by its own 99.5th percentile, which makes the nearest road surface at the bottom edge maximally
bright, and a very near, very bright band across the bottom of a driving video is what a bonnet
looks like. Nobody re-ran the conversion with an absolute depth scale to check. **Untested.**

**CARLA's depth camera sees through fog.** On 2802 the RGB frame hides the far street but the
depth and map controls do not, so Cosmos painted vehicles and buildings the model was never
shown. `COSMOS_VIS_M=50` blanks both controls beyond 50 m, which leaves the far street hazy the
way CARLA has it. The cut uses the capture's own inverse-depth normalisation
(`v = 255 (d_ref/d)^0.6`, `d_ref = mount height / tan(vertical half FOV)`).

**Cosmetic drift.** Cars are recoloured (a red sedan comes back silver; a liveried lead car loses
its livery), shopfront detail is invented, and in the 15 s per-tick closed-loop run the overall
look drifted (brick and rain towards glass and sun). Geometry, lane markings and vehicle
positions follow the controls throughout.

**The map renderer faults above 1280×720.** Ludus raises "CUDA illegal memory access" at random
once the framebuffer exceeds about 1280×720 at MSAA 4 — 1600×900, 1920×1200 and 1928×1208 fail on
some frames and pass on others, 1280×720 never failed across 300+ frames and 33 probe frames
between 44° and 124°. Bisected one case per subprocess: not the FOV, not the FTheta polynomial
degree, not 16-pixel alignment, not the GPU. `wsm_service.py` renders large cameras on a
1280×720 canvas with the focal length shrunk to fit, then crops and resizes; the converter
downsizes to 832×480 anyway. Its `FThetaCamera.fw_poly` is also a fixed six-coefficient tensor,
so the fit is always degree 5 — a degree-9 fit for openpilot's 119° camera overran the kernel.

## The closed-loop runs, for context

Before the open-loop study, the same converter ran **inside** the driving loop on 16 Sep:
replay the recording to T0, then `carla_frame_t (+ depth_t + wsm_t) -> cosmos_frame_t -> model
-> action -> CARLA`, one converted frame and one plan per tick, nothing predicted and nothing
held. `segment.py` modes `control` (CARLA pixels through the same path) and `live` (converted
pixels) make the comparison.

| Model | recorded 20 Hz | control (CARLA pixels, same path) | live (Cosmos pixels) |
|---|---|---|---|
| VisionPilot | DS 100 / RC 100 | DS 100 / RC 100 / 0 infractions | **DS 100 / RC 100 / 0 infractions**, 289 ticks, 83 min |
| DriveJEPA | DS 60, 1 collision | DS 22.6 / RC 100 | DS 34.1 / RC 100 |
| openpilot | stops for the invader | crawls ~1 m/s | accelerates to 5.8 m/s; cut at T+18.5 s by a 4 h route timeout |

VisionPilot's mean lateral offset was 0.29 m on converted frames against 0.09 m on CARLA frames.
DriveJEPA's live 34.1 against control 22.6 is inside its own scatter (recorded 60), so it is not
a Cosmos effect. **This is why the study moved to open loop:** a driving score folds the
converter's effect together with the controller's, the harness's and the route's, and three
numbers per model cannot separate them. Open loop compares plans over identical states.

## Pod rebuild

About 25 minutes on a fresh two-GPU RTX PRO 6000 Blackwell box (driver R580+/CUDA 13,
`/path/to/scratch` = `POD_ROOT` throughout):

1. A current `uv` from the Astral installer — a distro `uv` 0.9 is too old for these projects.
2. `git clone https://github.com/NVIDIA/flashdreams` at the pinned sha, then
   `uv sync --package flashdreams-omnidreams --inexact --python 3.12`. This builds
   `ludus_renderer`; `ninja` must be on `PATH` for its JIT.
3. `git clone https://github.com/NVIDIA/cosmos-framework` at the pinned sha, then
   `uv sync --all-extras --group=cu130`. `torchcodec` needs `LD_LIBRARY_PATH` over every
   `site-packages/nvidia/*/lib` directory or it fails to `dlopen libnppicc.so.13`.
4. `hf download nvidia/Cosmos3-Nano` (33 GB) with `HF_HOME=$POD_ROOT/hf`. Model weights:
   `nvidia/Alpamayo-1.5-10B`, `nvidia/Cosmos-Reason2-8B`, `Qwen/Qwen-Drive-1.0-4B` via
   `hf download --local-dir $POD_ROOT/weights/<name>`.
5. `rsync` the paddock checkout, then `uv venv && uv pip install -e .`; the model envs come from
   `openloop/build_model_envs.sh`. Alpamayo needs `uv sync --no-install-package flash-attn` plus
   `uvicorn fastapi pillow requests`, and a placeholder `[datasets.navsim]` section in the site
   config; QwenDrive installs from a `uv pip freeze` of a working env with the `-e file://` line
   stripped.
6. The pod's site config: `openloop/paddock.toml.example`.

Both HF downloads and the repo clones run at pod bandwidth (minutes). **Never copy large files
up from the workstation** — that uplink is about 1.5 MB/s, which is 40 minutes for one 3.7 GB
checkpoint.

Alpamayo's node needs ~26 GB of VRAM and QwenDrive's ~24 GB, which the shared workstation GPU
does not have free, so those two models' replays ran on the pod and the other three locally.

## Verified, and argued

Following [docs/reviewing.md](../reviewing.md): what was run, and what is inference from it.

| Claim | How | Strength |
|---|---|---|
| The two replays see identical states and differ only in pixels | one `ticks.jsonl` per capture drives both; `openloop_replay.py` builds the same wire body either way and only `--frames` changes | **strong** |
| The numbers in the tables are what the runs produced | every cell is read from `openloop-2802/compare/all_summaries.json`, written by `openloop_compare.py` | **strong** |
| The offline replay harness reproduces what a live run did | on route 3813 the replay of openpilot on already-converted frames reproduced that run's saved plans to max abs delta 0.000 m | **proof**, but for the 3813 harness, not this one |
| openpilot's gap is systematic, not sampling | its floor is 0 over 301 ticks, the planned-speed difference has one sign throughout, and the closed-loop run and the 3813 run show the same direction | **strong** |
| Alpamayo's gap is mostly its own sampling | its floor (2.55 / 1.79) is most of its gap (3.23 / 2.55) on the same ticks | **moderate** — one pair of runs, n = 301; it was never repeated |
| Cosmos 3 preserves CARLA's geometry | frame-by-frame review of the 4 s Town13 fixture and the 2802 clips: lanes, kerbs, lead car, buildings in place; no invented actors, unlike cosmos-transfer 2.5 on 3813 | **moderate** — visual, not a numeric layout metric |
| DriveJEPA plans slower on converted frames | 2.05 against 2.30 m/s over 301 ticks, floor 0 | **strong** |
| The bonnet is a model prior | absent from all three controls; unchanged by prompt, seed, depth weight, edge weight and control guidance | **strong** |
| The bonnet comes from the depth normalisation | argument only — a per-frame inverse-depth scale makes the near road maximally bright at the bottom edge | **unverified**, never re-run with an absolute scale |
| Ludus faults above ~1280×720 at MSAA 4 | bisected one case per subprocess on both GPUs with real maps; 1280×720 clean over 300+ frames | **strong** |
| The upstream shas | `NVIDIA/cosmos-framework` efba207 = `efba2076c8` (2026-09-16) and `NVIDIA/flashdreams` 6e2d3ad = `6e2d3ad3b6` (2026-09-10), both confirmed against the GitHub API; `build_transfer_batch`, `apply_transfer_control_augmentor`, `build_control_cfg_postprocess` and `TransferHintKey{edge,blur,depth,seg,wsm}` read in that source | **verified** |
| These plan gaps predict closed-loop scores | not claimed. The one model with both (VisionPilot) scored DS 100 either way; DriveJEPA's two closed-loop runs sit inside its own scatter | **not measured** |
| QwenDrive's figures | its conversion and replay completed, but its capture was stopped at 390 ticks to free CARLA, so its segment is the shortest | **strong**, narrower window |

## Upstream

- [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) — `docs/inference.md`;
  `cosmos_framework/inference/transfer.py` (`build_transfer_batch`,
  `apply_transfer_control_augmentor`, `build_control_cfg_postprocess`);
  `cosmos_framework/inference/args.py` (`TransferHintKey`, the sample spec). Pinned in
  `repos.yaml`. Transfer is a *mode* of the Cosmos 3 generator, not a separate model: there is
  no "Cosmos-Transfer3".
- [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) — the checkpoint, 16B,
  single GPU.
- [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams) — the OmniDreams world model and
  the `ludus_renderer` package that draws the map control. Pinned in `repos.yaml`.
