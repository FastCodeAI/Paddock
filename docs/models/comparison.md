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

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E ᵘ | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Camera views @ freq** | 1 front, 2 frames (t, t−1) | 4 (front_left, front_wide, front_right, front_tele); rig is 7-cam. AR1 paper used 2 | 2 (road + wide), 2 frames, 20 Hz | 1 mono front, 52–55° HFOV, 1–2 MP, 10+ Hz on Orin | 8 surround (front/back wide, front/back narrow, 4 corners), single frame; paddock fills 6 of the 8 | 7 surround, 10 Hz | 3 (front, front-left, front-right), 4 frames each 0.5 s apart. `DrivingScene` refuses a missing view, so a monocular source (comma2k19) replicates its road camera into both side slots -- `QWENDRIVE_VIEWS`, and the policy re-declares `cameras` to match |
| **Per-camera input** | 512×256 | 448×280 (→160 tokens/img); triplane tokenizer fixes 288 tokens/timestep regardless of camera count | 512×256 (medmodel); 1024×512 (bigmodel) | 1024×512 (2:1) all three nets; AutoDrive adds a 50° centre crop | 768×432 uint8 RGB | 256×256 | pixel budget on a 32 px grid: current ~720p (921,600 px), history ~320p (174,080 px) |
| **Other inputs** | ego state (NAVSIM protocol) | text prompt; egomotion history (3D translation + 9D rotation, multi-timestep); optional route | desire/nav command, 512-d feature history. **No speed channel at all** | vehicle speed | ego speed; optional LiDAR raster / SD map / driving command in the checkpoint, none in the released ONNX | rendered BEV nav-map tile; 256-d egomotion history (6.4 s @ 10 Hz); 896-d visual memory (64×14); camera projection operator | 16 ego poses @10 Hz (1.5 s) + velocity + acceleration; ego status [v, a, one-hot command]; nav command in the prompt |

## Outputs and architecture

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E ᵘ | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Output @ freq** | trajectory from an 8192-entry anchor vocabulary | trajectory @ 6.4 s + chain-of-causation trace; 99 ms end-to-end | 33 waypoints over 10 s @ 20 Hz + lane lines/lead | AutoSteer: ego-path waypoints in camera coords, **no time parameterisation**; AutoSpeed: CIPO boxes (3 classes); AutoDrive: CIPO distance + curvature + presence prob | K=3 × 6 waypoints @ 0.5 s (3 s) + mode logits + steer/accel/brake; plus 11 perception heads | 128-d = 64 steps × (accel, curvature), 6.4 s @ 10 Hz | 50 poses (x, y, heading) @10 Hz, 5 s; optional one-sentence rationale |
| **Architecture** | V-JEPA-2-init ViT-L encoder + proposal-centric planner; MTD + momentum-aware selection | Cosmos-Reason2 + diffusion trajectory decoder (AR1: Cosmos-Reason VLM + traj decoder) | supercombo, a single ONNX graph | 3 separate conv nets, all YOLO-family 'n' variant | ResNet-34 + FPN per camera, depth-gated IPM lift to one 800×500 BEV @ 0.2 m, 12 task heads + refiners (`DepthSegIPMNetV52`) | SwinV2-T or ConvNextV2-T + fusion (concat / cross-attn / BEV) + planner GRU | Qwen3.5-4B VLM (unchanged) + 1 B flow-matching Planning Expert reading its KV caches; separate BEV perception head |
| **Parameters** | 307 M (encoder + planner) | **10 B** (34 B for Super) | not published | **28.1 M** — AutoSteer 8.22 + AutoSpeed 9.33 + AutoDrive 10.58 ᵐ | 54 M (refiner incl.) ᵖ | 35.3 M (concat/cross-attn); 69.7 M (BEV, default) | **5.58 B** (4.54 B VLM + 1.04 B expert, counted from the checkpoint; the 125 M perception head is not loaded for planning) |
| **Size** | 1.23 GB fp32 / 0.61 fp16 ᶜ | 20 GB bf16 ᶜ | **0.061 GB** (`driving_supercombo.onnx`); 1.76 GB big variant; 0.0075 GB dmonitoring ᵐ | 0.112 GB fp32 / 0.028 INT8 ᵐ | 0.35 GB fp32 ONNX ᵐ; 2:4 sparse trunk | 0.141 / 0.279 GB fp32 ᶜ | 9.1 GB VLM + 2.1 GB per planner, bf16 ᶜ |

Note the four-order-of-magnitude spread in the size row, and then read the
neuroNCAP column of the support matrix, where the top three are statistically
indistinguishable.

## Availability

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E ᵘ | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Training code** | yes | yes (Apache-2.0) | **no** — comma trains offline and ships weights | yes | yes (Apache-2.0), + autolabel factory | yes | **no** — inference code only |
| **Inference code** | yes | yes, + AlpaSim integration | yes (`modeld`, MIT) | yes (Python + C++, ONNX Runtime/TensorRT, ROS2/Zenoh) | yes (Python + C++ TensorRT runtimes; plain ONNX runs in onnxruntime) | yes | yes (Apache-2.0) |
| **Checkpoints** | yes (HF) | yes — `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo-1.5-10B`. **OpenMDW-1.1, non-commercial**; commercial on request | yes — git-LFS in repo, MIT | yes — `.pth` + ONNX fp32 + ONNX INT8, **Apache-2.0, commercial OK** | yes — `AutowareFoundation/meteor` (`.pt` + ONNX), **Apache-2.0** | **none** | yes — `Qwen/Qwen-Drive-1.0-4B` (VLM + planner-sft + planner-rl + perception), **Apache-2.0, commercial OK** |

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
| **AutoE2E** ᵘ | KITScenes (unconfirmed) | not documented |
| **Qwen-Drive 1.0** | Qwen3.5-4B's own corpus; driving VQA, perception and planning data from public datasets (report §data); trajectories "standardized from multiple public driving datasets" — **NAVSIM `navtrain`** (declared), Waymo E2E, NVIDIA PhysicalAI | planner-rl: reward optimisation on the benchmark objectives; paddock runs planner-sft |

---

## What this changes about reading the support matrix

One of the models has training exposure to a benchmark's own dataset, and a
second may have it without anyone being able to check. The score table cannot
show either, and both change how a column should be read.

**Every such pair gets its own section here**, named for the model and the
dataset. `test_training_on_a_benchmarks_own_dataset_is_documented` requires one
for any model that declares training on a dataset it is also scored on, and
checks there is a paragraph under it — so a future adapter cannot inherit the
situation and pass on the strength of having a row in the tables above.

### Drive-JEPA × navsim

**Drive-JEPA is the only model here with *declared* training exposure to a
paddock benchmark's dataset, and it leads that benchmark by 0.235 EPDMS.** It
was post-trained on NAVSIM `navtrain` and pre-trained partly on OpenScene —
which *is* the NAVSIM dataset. Every other model meets `navtest` zero-shot.
(OpenPilot × comma2k19 below is the undeclared case: same fleet, no way to
check.)

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

### Qwen-Drive 1.0 × navsim

**Qwen-Drive's NAVSIM number is in-distribution, and it is the number the
vendor chose as its headline.** The technical report standardises trajectories
from several public driving datasets into one waypoint representation and
reports NAVSIM v1.1 navtest PDMS (87.8 sft direct, 88.2 sft reasoning, 90.7 rl)
alongside Waymo E2E and NVIDIA PhysicalAI; a model reporting navtest as its
first planning result trained on `navtrain`, and `trained_on` declares that.
The NAVSIM frames, cameras (`cam_f0/l0/r0`) and 10 Hz ego history paddock feeds
it are the ones its own scene files are built from, so on this column it plays
the benchmark as designed, like Drive-JEPA, while Alpamayo, OpenPilot and
VisionPilot meet navtest zero-shot.

**What is not sourced.** The exact split list is not in the inference repo:
`docs/data.md` describes building navtest scene files, and the report's data
section names the datasets rather than the splits. `navtrain` is the honest
assumption, not a citation; if the report turns out to state otherwise, the
declaration in `adapters/models/qwendrive/model.yaml` is the one line to change.

KITScenes, neuroNCAP (nuScenes, rendered) and comma2k19 are zero-shot for this
model, and its KITScenes cameras sit at wider bearings and field of view than
the nuPlan rig it was trained on (see [qwendrive.md](qwendrive.md)). comma2k19
is the unequal column in a different way: nothing in the corpus is a
windscreen-mounted mono camera on a Californian highway, and paddock has to
replicate that one frame into three view slots for the model to read it at all.
Second place on that board is therefore a claim about a model driven outside
both its rig and its domain.

### OpenPilot × comma2k19 — same fleet, unverifiable overlap

comma2k19 is **comma's own released fleet data**: 2035 segments of I-280,
recorded on a RAV4 and a Civic with an EON, and released by comma in 2018.
OpenPilot's driving model is trained on comma's internal fleet — 400 k
one-minute segments, not released, not enumerated. Whether these particular
2035 segments are inside that corpus is **not answerable from outside comma**,
and nothing in this repository can settle it.

What can be said: this is not the zero-shot situation the other comma2k19 rows
are in. Alpamayo, Qwen-Drive, Drive-JEPA and VisionPilot meet this dataset
having never seen a comma device; OpenPilot meets a route recorded by the
hardware it ships on, in the domain its fleet drives, with the camera geometry
it was trained against. Even under the most favourable reading — none of these
2035 segments in training — it is the only model here being scored **in**
distribution, and the 1.507 should be read as such rather than as a like-for-like
win over models driven outside their rig and domain.

`model.yaml` declares one dataset (`comma-fleet`) rather than two, so the
contract test that requires this section does not fire for OpenPilot; the
section exists because the situation is real, not because a test asked for it.
Declaring `comma2k19` there would assert an overlap nobody has evidence for,
which is the same failure as the sentence it replaced denying one.

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

### METEOR × navsim — a 3 s planner on a 4 s benchmark

The other unequal term, and it runs the same way as the camera one. NAVSIM
scores four seconds; METEOR's checkpoint plans three. Paddock will not quietly
extrapolate, so the run declared `METEOR_HORIZON_S=4.0` and the last second of
every plan is its final 0.5 s segment continued — one quarter of the scored
horizon, recorded in `run.json`. The columns that read the far end of the plan
are where it falls to the constant-velocity floor: lane keeping 0.795 against
the floor's 0.827, extended comfort 0.451 against 0.567. Its 0.303 EPDMS sits
between the floor and VisionPilot, while on nuScenes, which scores three
seconds and needs no extension, its L2 sits with Drive-JEPA's. Both numbers are
real; they are not measuring the same object. `docs/models/meteor.md` has the
measurements.

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
