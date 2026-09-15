# What these models are

The support matrix says how well each model drives. This says **what each one
is** — how large, what it consumes, what it emits, and what it was trained on.

Two reasons it matters beyond curiosity. First, the destination for our own work
is an automotive SoC, and a score with no size or latency beside it cannot
answer the only question that matters there. Second — and this is the part the
score table cannot show — **these models did not all meet these benchmarks on
equal terms.** See [what this changes](#what-this-changes-about-reading-the-support-matrix).

The machine-readable subset of this lives in each `model.yaml` under
`capabilities:` and is printed by `paddock list models`. This page is the
narrative half: provenance, licences, and the things that do not fit a schema.

---

## Inputs

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | Qwen-Drive 1.0 | AutoE2E ᵘ |
|---|---|---|---|---|---|---|---|
| **Camera views @ freq** | 1 front, 2 frames (t, t−1) | 4 (front_left, front_wide, front_right, front_tele); rig is 7-cam. AR1 paper used 2 | 2 (road + wide), 2 frames, 20 Hz | 1 mono front, 52–55° HFOV, 1–2 MP, 10+ Hz on Orin | 8 surround (front/back wide, front/back narrow, 4 corners), single frame; paddock fills 6 of the 8 | 3 view slots (front/left/right) x 4 timesteps = 12 images, 10 Hz. comma2k19 has one camera, so paddock replicates the front frame into both side slots -- `DrivingScene` raises on a missing view | 7 surround, 10 Hz |
| **Per-camera input** | 512×256 | 448×280 (→160 tokens/img); triplane tokenizer fixes 288 tokens/timestep regardless of camera count | 512×256 (medmodel); 1024×512 (bigmodel) | 1024×512 (2:1) all three nets; AutoDrive adds a 50° centre crop | 768×432 uint8 RGB | native source resolution, unchanged: frames carry no `target_size`, so the processor snaps them to its own patch budget and keeps the aspect ratio | 256×256 |
| **Other inputs** | ego state (NAVSIM protocol) | text prompt; egomotion history (3D translation + 9D rotation, multi-timestep); optional route | desire/nav command, 512-d feature history. **No speed channel at all** | vehicle speed | ego speed; optional LiDAR raster / SD map / driving command in the checkpoint, none in the released ONNX | 16-pose ego history @ 10 Hz (1.5 s) with body-frame velocity and acceleration; a nav-command string; in `reasoning_planning`, its own one-sentence rationale | rendered BEV nav-map tile; 256-d egomotion history (6.4 s @ 10 Hz); 896-d visual memory (64×14); camera projection operator |

## Outputs and architecture

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | Qwen-Drive 1.0 | AutoE2E ᵘ |
|---|---|---|---|---|---|---|---|
| **Output @ freq** | trajectory from an 8192-entry anchor vocabulary | trajectory @ 6.4 s + chain-of-causation trace; 99 ms end-to-end | 33 waypoints over 10 s @ 20 Hz + lane lines/lead | AutoSteer: ego-path waypoints in camera coords, **no time parameterisation**; AutoSpeed: CIPO boxes (3 classes); AutoDrive: CIPO distance + curvature + presence prob | K=3 × 6 waypoints @ 0.5 s (3 s) + mode logits + steer/accel/brake; plus 11 perception heads | 50 waypoints (x, y, heading) over 5 s @ 10 Hz, from a flow-matching sampler; plus the rationale and, unused here, 3D detection / occupancy / BEV-map heads | 128-d = 64 steps × (accel, curvature), 6.4 s @ 10 Hz |
| **Architecture** | V-JEPA-2-init ViT-L encoder + proposal-centric planner; MTD + momentum-aware selection | Cosmos-Reason2 + diffusion trajectory decoder (AR1: Cosmos-Reason VLM + traj decoder) | supercombo, a single ONNX graph | 3 separate conv nets, all YOLO-family 'n' variant | ResNet-34 + FPN per camera, depth-gated IPM lift to one 800×500 BEV @ 0.2 m, 12 task heads + refiners (`DepthSegIPMNetV52`) | Qwen3.5-4B VLM shared by three attached modules: a BEV perception head, a flow-matching planning expert, and the unmodified LLM decoder for VQA | SwinV2-T or ConvNextV2-T + fusion (concat / cross-attn / BEV) + planner GRU |
| **Parameters** | 307 M (encoder + planner) | **10 B** (34 B for Super) | not published | **28.1 M** — AutoSteer 8.22 + AutoSpeed 9.33 + AutoDrive 10.58 ᵐ | 54 M (refiner incl.) ᵖ | **5.58 B** (loaded VLM + expert, counted from the checkpoint) | 35.3 M (concat/cross-attn); 69.7 M (BEV, default) |
| **Size** | 1.23 GB fp32 / 0.61 fp16 ᶜ | 20 GB bf16 ᶜ | **0.061 GB** (`driving_supercombo.onnx`); 1.76 GB big variant; 0.0075 GB dmonitoring ᵐ | 0.112 GB fp32 / 0.028 INT8 ᵐ | 0.35 GB fp32 ONNX ᵐ; 2:4 sparse trunk | 13 GB bf16 ᶜ | 0.141 / 0.279 GB fp32 ᶜ |

Note the four-order-of-magnitude spread in the size row, and then read the
neuroNCAP column of the support matrix, where the top three are statistically
indistinguishable.

## Availability

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | Qwen-Drive 1.0 | AutoE2E ᵘ |
|---|---|---|---|---|---|---|---|
| **Training code** | yes | yes (Apache-2.0) | **no** — comma trains offline and ships weights | yes | yes (Apache-2.0), + autolabel factory | **no** -- inference and evaluation scripts only | yes |
| **Inference code** | yes | yes, + AlpaSim integration | yes (`modeld`, MIT) | yes (Python + C++, ONNX Runtime/TensorRT, ROS2/Zenoh) | yes (Python + C++ TensorRT runtimes; plain ONNX runs in onnxruntime) | yes (Apache-2.0) | yes |
| **Checkpoints** | yes (HF) | yes — `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo-1.5-10B`. **OpenMDW-1.1, non-commercial**; commercial on request | yes — git-LFS in repo, MIT | yes — `.pth` + ONNX fp32 + ONNX INT8, **Apache-2.0, commercial OK** | yes — `AutowareFoundation/meteor` (`.pt` + ONNX), **Apache-2.0** | yes -- `Qwen/Qwen-Drive-1.0-4B` on HF and ModelScope, **Apache-2.0** | **none** |

The licence row is the one to read before planning a product. Alpamayo's weights
are non-commercial by default; VisionPilot's and OpenPilot's are not.

## Training data — the important table

| | Pre-training | Post-training |
|---|---|---|
| **Drive-JEPA** | CoVLA + DrivingDojo + **OpenScene**, 208 h, 8-frame clips @512×256, 2 Hz; encoder init from V-JEPA 2 | **NAVSIM `navtrain`** (+ **CARLA** for Bench2Drive) |
| **Alpamayo 1.5** | NVIDIA internal: 80,000 h, >1 B images, 2,500+ cities / 25 countries | chain-of-causation traces: 700 K (AR1) / 3 M (1.5); RL via Cosmos-RL / AlpaGym |
| **OpenPilot** | comma internal fleet: 400 k × 1-min segments ≈ 6,670 h @5 Hz, not released | on-policy rollouts in reprojective sim + world model; fleet deployment, 500 users, 87 k trips |
| **VisionPilot** | OpenLane (AutoSpeed CIPO); TuSimple + OpenLane + CurveLanes (AutoSteer); AutoDrive warm-starts from the AutoSpeed backbone | none — single-stage supervised |
| **METEOR** | TIER IV Co-MLOps DRS recordings (Japan + a US subset), 953 GB / 12,595 scenes @ 5 Hz, 100 % auto-labelled; adverse conditions as Cosmos Transfer re-renders | rule-based trajectory rewards + GRPO mode loss (r48) |
| **Qwen-Drive 1.0** | **not stated.** The released card names no public driving corpus and the checkout carries no training manifest — only "multiple public driving datasets", standardised into one waypoint representation, alongside general-purpose vision-language data | staged: perception + language + planning objectives; `planner-sft` is the imitation-trained expert, `planner-rl` the same expert after reward training |
| **AutoE2E** ᵘ | KITScenes (unconfirmed) | not documented |

---

## What this changes about reading the support matrix

One model has *declared* training exposure to a benchmark's own dataset, and one
declares nothing at all. The score table cannot show either, and both change how
a column should be read.

**Every such pair gets its own section here**, named for the model and the
dataset. `test_training_on_a_benchmarks_own_dataset_is_documented` requires one
for any model that declares training on a dataset it is also scored on, and
checks there is a paragraph under it — so a future adapter cannot inherit the
situation and pass on the strength of having a row in the tables above.

### Drive-JEPA × navsim

**Drive-JEPA is the only one of the four with training exposure to a paddock
benchmark's dataset, and it leads that benchmark by 0.235 EPDMS.** It was
post-trained on NAVSIM `navtrain` and pre-trained partly on OpenScene — which
*is* the NAVSIM dataset. The other three meet `navtest` zero-shot.

To be precise about what this is and is not: training on `navtrain` and testing
on `navtest` is **the protocol NAVSIM intends**. Drive-JEPA is not cheating; it
is playing the benchmark the way the benchmark is designed to be played. What
makes the column hard to read is that the other three are not playing it that
way at all. The 0.235 EPDMS gap therefore conflates *"better planner"* with
*"in-distribution versus zero-shot"*, and nothing on the page previously said so.

**Sourced at the pinned sha, not taken on trust.** In `Drive-JEPA @ e21f474`,
the script `navsim_v2/scripts/training/train_drive_jepa_perception_based.sh`
reads:

```bash
TRAIN_TEST_SPLIT=navtrain                          # line 4
...
        agent=drive_jepa_perception_based_agent \  # line 7
        train_test_split=$TRAIN_TEST_SPLIT \
```

That agent is the one whose checkpoint paddock loads —
`drive_jepa_perception_based_agent_vitl_v2.ckpt`, named in
`adapters/models/drivejepa/model.yaml` as `DRIVEJEPA_CHECKPOINT`. So the
training split is not inferred from the paper; it is in the trainer for this
exact checkpoint. (The script lives under `navsim_v2/`, not at the top level —
the README's Training section points at `scripts/training/`, which is easy to
look for in the wrong place.)

This compounds a caveat already recorded in
[navsim.md](../benchmarks/navsim.md): Drive-JEPA's native agent is
**scorer-aware**, reading the PDM simulator at inference time, which is why
paddock reports its proxy figure instead. Two independent advantages, both
pointing the same way.

**What is *not* sourced: CARLA.** Drive-JEPA is reported to have been
post-trained on CARLA for Bench2Drive, which would matter — that benchmark is
framed in [bench2drive.md](../benchmarks/bench2drive.md) as a real→sim transfer
test where *"models trained on real data (nuScenes / NAVSIM) are out of
distribution on CARLA frames"*, and a CARLA-trained model would not be. **It
could not be confirmed from the checkout.** The only CARLA reference at
`e21f474` is a URL in a code comment citing `carla_garage` for an augmentation
recipe; there is no CARLA training script, config or doc. It is therefore
*not* declared in `trained_on`, and the Bench2Drive number — whenever that
adapter is run — should not be labelled on the strength of it. Worth resolving
before that sweep, not after.

### AutoE2E × kitscenes — unconfirmed, and not yet a paddock model

AutoE2E's pre-training row reads KITScenes, marked unconfirmed. If it is right,
an AutoE2E KITScenes ADE would be in-distribution while all four current models
are zero-shot there — and `alpamayo15-eval-notes.md` states plainly that neither
Alpamayo nor Drive-JEPA was trained on KITScenes, which is what makes that
column a fair comparison today.

AutoE2E has no merged adapter, so nothing enforces this yet. Confirm the
pre-training corpus before scoring it, and give it a section here if it holds.

None of this makes any recorded number wrong. It makes one of them mean
something different from what a reader assumes, which is the same class of
problem `sequences_sha256` and `valid: false` exist to prevent.

### METEOR × kitscenes — six cameras of eight, and the graph as shipped

METEOR trained on none of paddock's datasets, so it has no exposure section;
the unequal terms run the other way. It is an 8-camera network; KITScenes hands
it six (#40), the two telephoto slots stay absent (the convention it was
trained with for one missing camera), it has no driving-command input in the
released ONNX, and that ONNX applies no ImageNet normalisation although its
training loader did (verified against the checkpoint's first conv; see
`docs/models/meteor.md`). Its 1.879 ADE is longitudinal above 13 m/s, where its
plans plateau near 35 m at 3 s, a prior from urban Japanese driving. The
front-only run made before #40 scored 4.784 and is kept as an ablation: 2.9 m
of ADE was the missing cameras. Read the KITScenes cell as "the released model,
on a rig and a domain it never saw", which is the same caveat the other rows
carry in different proportions.

### Qwen-Drive × everything — the corpus is not stated, so nothing can be ruled out

Qwen-Drive's `trained_on` is `not-stated`, and that is a finding rather than a
gap in this page. The released card names no public driving corpus — only
"multiple public driving datasets" standardised into one waypoint
representation — and the checkout carries no training manifest, no training
code, and no data config. The useful claim would be "none of paddock's
benchmarks", and nothing in the release supports it.

So the comma2k19 row cannot be called zero-shot. comma2k19 is an unusual case in
its favour — it is a pose-estimation dataset whose video is one Californian
highway, not a standard planning corpus, and it appears in no public
E2E-planning training recipe we can find — but "unlikely" is not "declared", and
the only honest label is *unknown*. Two things follow. `trained_on: not-stated`
stays as it is, rather than being filled in with a plausible list; and the
demo scenes bundled with the checkout being Waymo Open Dataset format is
evidence about the input format the model expects, not about what it saw.

The paper's own table reports a Waymo Open Dataset E2E **test** result, which is
an evaluation, not a declaration of training data. Do not read it as one.

### Drive-JEPA (perception-free) × comma2k19 — zero-shot on the data, not on the domain

The `drivejepa_free` row is genuinely zero-shot with respect to comma2k19: the
perception-free agent is a NAVSIM v1 model, trained on nuPlan-derived data, and
comma2k19 is neither. What the row should not be read as is *domain*
transfer-free. It was trained on urban and suburban driving at nuPlan speeds,
and its 4 s reach saturates around 64 m — roughly 16 m/s — while I-280 runs at
about 30 m/s.

That is a training-domain speed prior, and it dominates its raw ADE. The
benchmark reports `lateral_at_arclength_mae` and `path_length_ratio` beside ADE
precisely so this is legible rather than mysterious: the path is close to right
and the distance along it is not. Read the comma2k19 cell with that in mind, and
see the README's footnote on the row.

## Three discrepancies against what the adapters declare

Recorded rather than resolved, because `capabilities.cameras` decides what a
simulator renders and changing it changes every closed-loop measurement.

1. **Drive-JEPA: 4 cameras declared, 1 in the architecture table.**
   `model.yaml` declares `[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
   CAM_BACK]`; the table says one front camera with two frames. The adapter runs
   the NAVSIM *perception-based* agent, whose BEVFormer stage is multi-camera by
   construction, so the two may describe different configurations. If the
   adapter is over-declaring, closed-loop runs are rendering four views to feed
   one — `adding-a-model.md` calls that "the single largest closed-loop cost".
   **Worth measuring before the next sweep.**
2. **Alpamayo: 3 slots fed of 4.** `model.yaml` declares `CAM_FRONT`,
   `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`; the model has a fourth, `front_tele`.
   Consistent with the eval notes, which match three slots by yaw — recorded so
   nobody re-derives it.
3. **AutoSpeed parameter count: 9.33 M or 9.31 M.** The measurement in
   [VISION_PILOT_GUIDE.md](VISION_PILOT_GUIDE.md) says 9.33 M; a later count
   says 9.31 M. The total is 28.1 M either way. Trivial, and noted only because
   this repo's habit is to say which number it used: `model.yaml` carries 28.1.

## Provenance of these figures

ᵐ measured · ᶜ computed from parameter count · ᵖ published by the authors, not re-counted · ᵘ **AutoE2E is not a paddock
model** — its adapter is unmerged and no checkpoint exists publicly, so it has
no row in the support matrix. It is included here because it is under
consideration and its data belongs beside the others.

Everything not marked ᵐ comes from published papers, model cards and repository
inspection rather than from paddock. Paddock has measured none of it, which is
the gap [docs/design/model-card-and-latency.md](../design/model-card-and-latency.md)
proposes to close: declared numbers rot, and the fix is a fixture test asserting
the declaration against the artifact.
