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

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Camera views @ freq** | 1 front, 2 frames (t, t−1) | 4 (front_left, front_wide, front_right, front_tele); rig is 7-cam. AR1 paper used 2 | 2 (road + wide), 2 frames, 20 Hz | 1 mono front, 52–55° HFOV, 1–2 MP, 10+ Hz on Orin | 8 surround (front/back wide, front/back narrow, 4 corners), single frame; paddock fills 6 of the 8 | 6 surround (front, front-left, front-right, rear-left, rear, rear-right), 8 frames 0.5 s apart | 3 (front, front-left, front-right), 4 frames each 0.5 s apart. `DrivingScene` refuses a missing view, so a monocular source (comma2k19) replicates its road camera into both side slots -- `QWENDRIVE_VIEWS`, and the policy re-declares `cameras` to match |
| **Per-camera input** | 512×256 | 448×280 (→160 tokens/img); triplane tokenizer fixes 288 tokens/timestep regardless of camera count | 512×256 (medmodel); 1024×512 (bigmodel) | 1024×512 (2:1) all three nets; AutoDrive adds a 50° centre crop | 768×432 uint8 RGB | 512×512 (squashed); the current front also 1024×1024 | pixel budget on a 32 px grid: current ~720p (921,600 px), history ~320p (174,080 px) |
| **Other inputs** | ego state (NAVSIM protocol) | text prompt; egomotion history (3D translation + 9D rotation, multi-timestep); optional route | desire/nav command, 512-d feature history. **No speed channel at all** | vehicle speed | ego speed; optional LiDAR raster / SD map / driving command in the checkpoint, none in the released ONNX | 256-d egomotion history (6.4 s @ 10 Hz); optional route raster + 14-channel semantic map (450×300 @ 0.4 m; paddock sends the route where the benchmark has one, never the map); per-frame camera projections | 16 ego poses @10 Hz (1.5 s) + velocity + acceleration; ego status [v, a, one-hot command]; nav command in the prompt |

## Outputs and architecture

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Output @ freq** | trajectory from an 8192-entry anchor vocabulary | trajectory @ 6.4 s + chain-of-causation trace; 99 ms end-to-end | 33 waypoints over 10 s @ 20 Hz + lane lines/lead | AutoSteer: ego-path waypoints in camera coords, **no time parameterisation**; AutoSpeed: CIPO boxes (3 classes); AutoDrive: CIPO distance + curvature + presence prob | K=3 × 6 waypoints @ 0.5 s (3 s) + mode logits + steer/accel/brake; plus 11 perception heads | 128-d = 64 steps × (accel, curvature), 6.4 s @ 10 Hz; paddock integrates them with upstream's rollout | 50 poses (x, y, heading) @10 Hz, 5 s; optional one-sentence rationale |
| **Architecture** | V-JEPA-2-init ViT-L encoder + proposal-centric planner; MTD + momentum-aware selection | Cosmos-Reason2 + diffusion trajectory decoder (AR1: Cosmos-Reason VLM + traj decoder) | supercombo, a single ONNX graph | 3 separate conv nets, all YOLO-family 'n' variant | ResNet-34 + FPN per camera, depth-gated IPM lift to one 800×500 BEV @ 0.2 m, 12 task heads + refiners (`DepthSegIPMNetV52`) | ResNet-50 + BEVFormer V2 T8 (300×200 BEV, frozen for the KITScenes fine-tune), deformable map/route fusion, GRU planner | Qwen3.5-4B VLM (unchanged) + 1 B flow-matching Planning Expert reading its KV caches; separate BEV perception head |
| **Parameters** | 307 M (encoder + planner) | **10 B** (34 B for Super) | not published | **28.1 M** — AutoSteer 8.22 + AutoSpeed 9.33 + AutoDrive 10.58 ᵐ | 54 M (refiner incl.) ᵖ | 79.9 M ᵖ | **5.58 B** (4.54 B VLM + 1.04 B expert, counted from the checkpoint; the 125 M perception head is not loaded for planning) |
| **Size** | 1.23 GB fp32 / 0.61 fp16 ᶜ | 20 GB bf16 ᶜ | **0.061 GB** (`driving_supercombo.onnx`); 1.76 GB big variant; 0.0075 GB dmonitoring ᵐ | 0.112 GB fp32 / 0.028 INT8 ᵐ | 0.35 GB fp32 ONNX ᵐ; 2:4 sparse trunk | 0.32 GB fp32 ᶜ | 9.1 GB VLM + 2.1 GB per planner, bf16 ᶜ |

Note the four-order-of-magnitude spread in the size row, and then read the
neuroNCAP column of the support matrix, where the top three are statistically
indistinguishable.

## Availability

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 |
|---|---|---|---|---|---|---|---|
| **Training code** | yes | yes (Apache-2.0) | **no** — comma trains offline and ships weights | yes | yes (Apache-2.0), + autolabel factory | yes | **no** — inference code only |
| **Inference code** | yes | yes, + AlpaSim integration | yes (`modeld`, MIT) | yes (Python + C++, ONNX Runtime/TensorRT, ROS2/Zenoh) | yes (Python + C++ TensorRT runtimes; plain ONNX runs in onnxruntime) | yes | yes (Apache-2.0) |
| **Checkpoints** | yes (HF) | yes — `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo-1.5-10B`. **OpenMDW-1.1, non-commercial**; commercial on request | yes — git-LFS in repo, MIT | yes — `.pth` + ONNX fp32 + ONNX INT8, **Apache-2.0, commercial OK** | yes — `AutowareFoundation/meteor` (`.pt` + ONNX), **Apache-2.0** | yes — `AutowareFoundation/auto_e2e` v1.0 (`nuplan-epoch-5`, `kitscenes-epoch-5`); code Apache-2.0, checkpoint licence "other" (model card), BEVFormer initialisation NOASSERTION | yes — `Qwen/Qwen-Drive-1.0-4B` (VLM + planner-sft + planner-rl + perception), **Apache-2.0, commercial OK** |

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
| **AutoE2E** | nuPlan (`nuplan_full` stage; logs not listed) from a nuScenes-trained BEVFormer V2 initialisation | **KITScenes** fine-tune, camera BEV frozen (`kitscenes-epoch-5`, the checkpoint paddock runs) |
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

**Every Drive-JEPA row on this page is the released checkpoint.** A second
checkpoint exists for the same architecture and is not in any table above:

| | Drive-JEPA (released) | Drive-JEPA, retrained (AlpaSim driver 0.1.7) |
|---|---|---|
| Architecture, parameters | V-JEPA-2 ViT-L encoder + proposal-centric planner, 307 M | the same; encoder untouched (294 tensors bit-identical) |
| Trained parts | — | generator (2.38 M) and scorer (4.70 M) in a first run; generator again, alone, in a second |
| Trajectory vocabulary | 8,192 anchors | 10,240 (the 8,192 plus 2,048 faster shapes) |
| Inputs | four cameras; acceleration as reported | one front camera, bottom 37 rows masked; acceleration slots zeroed |
| `trained_on` | `navsim` / `navtrain` | `navsim` / `navtrain` (inherited) **+ `physicalai_av` / `train`** |
| Runs in paddock | every Drive-JEPA row | none — `tools/models/drivejepa/alpasim_driver/` only |

`physicalai_av` is NVIDIA's Physical AI AV dataset. It is **not a key in
`datasets.yaml`**, so no test here can cross-check that declaration against a
benchmark's scene list: some AlpaSim scenes are reconstructions of clips in
that dataset, and their exclusion from training (126,981 samples, 4,015 clips,
`train` split, evaluation scenes asserted absent in every script) is a property
of the training code in the gallops repository, taken on that code's word. Two
rows in `results/` were made with these weights, both on AlpaSim: 30 scenes,
0.7697 (`legacy-drivejepa_017_20hz_30scenes`), and the benchmark's ten scenes
alone, 0.9956 (`legacy-drivejepa_017_20hz`). The 30-scene figure has its own
row in the support matrix, marked as adapted, and the section below
says how to read it. Detail in [drivejepa.md](drivejepa.md).

**Every Drive-JEPA row on this page is the perception-*based* v2 agent.** The
comma2k19 board carries a second, separate registry entry —
[`drivejepa_free`](drivejepa_free.md), the perception-free NAVSIM v1 agent —
because comma2k19 supplies neither the four cameras nor the lidar extrinsics v2
requires. Different checkpoint (`drive_jepa_perception_free_agent_vitl.ckpt`),
different repo subtree, different inputs; the two are not comparable to each
other and are deliberately not merged into one column here.

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

`model.yaml` declares one dataset (`comma-fleet`) rather than two, because
declaring `comma2k19` there would assert an overlap nobody has evidence for —
the same failure as the sentence it replaced denying one. The cost is that
`test_training_on_a_benchmarks_own_dataset_is_documented`, which fires off
`trained_on`, cannot fire for the pairing where the question is sharpest. So a
second test reads this file from the other end:
`test_an_overlap_section_nobody_declared_is_still_held_to_the_same_standard`
requires every `model × dataset` heading here to carry real prose about a
pairing that can actually happen. This section is held to the same standard as a
declared one; it is no longer surviving on goodwill.

### Drive-JEPA rerank × kitscenes

**`drivejepa_rerank` is adapted to KITScenes, not zero-shot on it.** It is
Drive-JEPA unchanged up to its 32 candidate trajectories. A selector then
chooses among them in place of the NAVSIM-trained scorer, and that selector was
fitted to KITScenes' **train** split: 5,711 keyframes from 373 scenes. Its
epoch was chosen on a held-out slice of those same train scenes.

The benchmark's 223 keyframes come from the 140 scenes of the val and
overlap_train_val splits. None of those scenes is among the 373, which was
checked by scene id. So the row has seen the dataset's cameras, cities and
driving, but not its test scenes. That is the same relation Drive-JEPA has to
navtest through navtrain.

Read its KITScenes cell beside the zero-shot rows, never as one of them. The
zero-shot `drivejepa` row stays as it is. On nuScenes the selector is
zero-shot: no nuScenes data went into it. See [drivejepa_rerank.md](drivejepa_rerank.md).

### AutoE2E × navsim — nuPlan-trained, scored on nuPlan's test logs

The checkpoint paddock runs, `kitscenes-epoch-5`, is the release's nuPlan model
fine-tuned on KITScenes. NAVSIM's scenes are OpenScene, which is nuPlan
resampled to 2 Hz, and paddock feeds AutoE2E from nuPlan's own 10 Hz logs. So
its 0.500 EPDMS on `navtest` is not zero-shot in the way VisionPilot's or
OpenPilot's is: same rig, same cities, same camera geometry it was pre-trained
on. Whether it saw the *logs* is a separate question the release does not
answer. The `nuplan_full` stage names no split and the dataset manifest it
hashes is not published; `navtest` is drawn from nuPlan's test logs, and the
release reports its own figures on nuPlan test, which suggests those were held
out. That is argued, not verified, and `model.yaml` declares `nuplan / train`
on that reading.

Its inputs differ from training too: NAVSIM sends no route and paddock no map,
which is upstream's own KITScenes test track (`camera_only_missing_map_route`),
while every training and validation sample had both.

The fine-tune pulls the other way: the last thing the checkpoint learned is a
different rig, KIT's ring cameras seen from a reference frame 2.1 m up. The
release's `nuplan-epoch-5` would be the in-distribution choice for this column
and is not what was scored. A KITScenes number from this checkpoint would be
in-distribution while the other models there are zero-shot; paddock does not
score AutoE2E on KITScenes yet.

### Drive-JEPA retrained × AlpaSim

**The retrained checkpoint's AlpaSim row is adapted, not zero-shot, and not
held out.** Its trajectory generator was retrained on Physical AI AV's `train`
split, and AlpaSim's scenes are reconstructions of clips from that same
dataset. The training code asserts the evaluation scenes absent; that lives in
the gallops repository and is taken on its word, because `physicalai_av` is not
a key in `datasets.yaml` and no test here can check it.

Two more things separate it from the rows beside it. The ten scenes were driven
many times while the driver was developed, so choices in the driver were made
with their outcomes in view. And the two figures are far apart, 0.9956 on the first ten scenes
against 0.7697 on all 30: every failure it has is on the other twenty.

Read either figure beside the released rows, never as one of them. The
released `drivejepa` rows, 0.4209 on the 30 scenes and 0.5030 on the ten, stay
as they are.

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

### VisionPilot × everything — every figure predates a constant that moved

VisionPilot's adapter carried `CURV_SCALE = 1.0`. The stack's own
`lateral_fusion.hpp` has `ad_curvature_scale = 0.21f`, annotated "CURV_SCALE =
0.21 in training", and `auto_drive.hpp` deliberately leaves the conversion to
the caller — which is why grepping the file that documents it finds nothing. So
the planner was handed a curvature 4.8x too large, and `IDM` caps set-speed at
`sqrt(mu*g/|kappa|)` and raises the overshoot to the fourth power: on a 30 m/s
comma2k19 keyframe it demanded about -60 m/s² and reached 5 m in four seconds
instead of 120.

Fixed September 2026. On a 20-sample comma2k19 smoke the correction moved ADE@3s
from 15.09 to 5.28 and `path_length_ratio` from 0.60 to 0.84, with every lateral
number unchanged — curvature reaches the plan only through the longitudinal cap.
**Every VisionPilot number on this page and in the README was produced with 1.0**
and will not be reproduced by a re-run. Expect the open-loop rows to move most
and the closed-loop ones least: Bench2Drive and neuroNCAP are urban and slow
enough that the cap rarely binds, which is why the defect survived them.
`docs/models/VISION_PILOT_GUIDE.md` has the derivation; the re-sweep is an open
item in `docs/ROADMAP.md`.

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

ᵐ measured · ᶜ computed from parameter count · ᵖ published by the authors, not re-counted.

Everything not marked ᵐ comes from published papers, model cards and repository
inspection rather than from paddock. Paddock has measured none of it, which is
the gap [docs/design/model-card-and-latency.md](../design/model-card-and-latency.md)
proposes to close: declared numbers rot, and the fix is a fixture test asserting
the declaration against the artifact.

## Notes moved from the README support matrix

The README's notes on cells that are about a model rather than one benchmark,
moved here unchanged except for links. Each starts with the mark it carried in
the matrix; the benchmark notes are on each benchmark's page under
"Support-matrix results".

⊛ `drivejepa_rerank` is Drive-JEPA unchanged up to its 32 candidate
trajectories. A small selector then chooses among them, in place of Drive-JEPA's
NAVSIM-trained scorer. That selector was trained on KITScenes **train** (5,711
keyframes, 373 scenes, none of them in the benchmark), so its KITScenes cell is
*adapted*, not zero-shot like the rows around it.
It sees the cameras cropped onto the OpenScene lens and gets no route hint.
Through that same path, Drive-JEPA's own pick scores 2.521 and the best of its 32
scores 0.409. The selector learned closeness to the human path and nothing about
collisions, so it has no closed-loop cells until it has been through neuroNCAP.
Its nuScenes cell is zero-shot (no nuScenes data went into the selector) and was
measured with the camera-frame fix (#62). Paired over 150 scenes, it is 0.054
below constant velocity on cumulative L2 (t −2.8); its point collision rate,
1.21% against 1.66%, is not separated (t −1.5). The Drive-JEPA row is still the
camera-blind one (⊘), and this row also differs from it in the camera crop, so
the two are not a like-for-like comparison.
See `docs/models/drivejepa_rerank.md`.

⊕ The same architecture with a second checkpoint: Drive-JEPA's trajectory
generator retrained on NVIDIA's Physical AI AV dataset (`train` split). It is
driven by the native AlpaSim driver at 0.1.7
(`tools/models/drivejepa/alpasim_driver/`), not through a paddock node, and the
row was ingested from that driver's recorded runs, as the Drive-JEPA row was.
Its AlpaSim cell is *adapted*, not zero-shot: AlpaSim's scenes are
reconstructions of clips from that dataset. The training code excludes the
evaluation scenes, which is taken on that code's word, and the 30 scenes were
driven repeatedly while the driver was developed, so the number is not a
held-out one. Referee, scenes and scorer are those of the other rows (¶), one
rollout per scene. On the first ten scenes alone it scores 0.9956. Its other
cells are not run: the checkpoint has no download location yet and no paddock
node. The launch log of each of its 30 runs records the checkpoint's sha256,
the one `docs/models/drivejepa.md` gives, so the file is identified even though
another site cannot yet fetch it.
See `docs/models/drivejepa.md`.

∗ VisionPilot has no native trajectory output. For nuScenes, NAVSIM and
KITScenes, Paddock constructs sampled XY points from its projected image-space
path and speed schedule, so those three figures grade the whole synthesised
pipeline -- perception, ground projection, curve fit, cruise-speed assumption
and extrapolation past a 1 s planner horizon -- and are not comparable in kind
with a model that emits a timed trajectory. The KITScenes run additionally
needed the ego origin dropped from the roof pod to the road, without which every
projected path point lands behind the car; see `docs/benchmarks/kitscenes.md`.

The ∗ rows are also the ones **`CURV_SCALE` moved**. The adapter carried 1.0
where the stack's own `lateral_fusion.hpp` has 0.21, so the planner saw a
curvature 4.8x too large and its `sqrt(mu*g/|kappa|)` speed cap bound where it
should not have. Fixed September 2026; on a 20-sample comma2k19 smoke the
correction moved ADE@3s from 15.09 to 5.28. **Every VisionPilot figure in this
README predates it** and will not be reproduced by a re-run — the closed-loop
ones least, since urban scenes rarely reach the cap, and the comma2k19 and
KITScenes ones most. `docs/models/VISION_PILOT_GUIDE.md` §"Curvature scale" has
the derivation; re-sweeping VisionPilot is an open item in `docs/ROADMAP.md`.

◊ METEOR plans **3 s**; NAVSIM scores **4 s**. A quarter of every scored plan is
therefore not the model's: `METEOR_HORIZON_S=4.0` (on the record as `model_env`)
continues the final 0.5 s segment to 4 s, because `to_navsim_poses` refuses a
shorter horizon rather than inventing one silently. The metrics that read the far
end of the plan are where METEOR falls to or below the constant-velocity floor —
lane keeping 0.795 against the floor's 0.827, extended comfort 0.451 against
0.567, the lowest in this table. Consecutive plans disagree three times more in
the extrapolated second (1.62 m of lateral) than inside the model's own horizon
(0.52 m), measured over 10,040 consecutive pairs. Read these two cells as METEOR
plus that extension, not as a 4 s planner; its 3 s output is scored without any
extension on nuScenes and KITScenes. See `docs/models/meteor.md`.

◇ METEOR is TIER IV's 8-camera multi-task network, run from its released
camera-only ONNX. KITScenes hands it the six-camera ring (#40), so only its two
telephoto slots are absent; it has no driving-command input. Fed the front
camera alone, as the harness did before #40, it scored 4.784 -- the run is kept
as an ablation. With six cameras the residual error is longitudinal above
13 m/s, where its plans plateau near 35 m at 3 s (a prior from its training
domain). The feed matches the authors' reference loader byte for byte. On
nuScenes, with all six cameras, its L2 sits with Drive-JEPA's; its collisions
are concentrated at standstill, where the planner creeps ~1 m into the queue
ahead (22.6 % of stopped anchors), and its plans carry a left bias twice as
large in Boston as in Singapore. See `docs/models/meteor.md`.

✦ **Two models here declare training exposure to a benchmark's own dataset.**
Drive-JEPA leads that benchmark by 0.096 EPDMS on v2 and 0.085 PDMS on v1 over
OpenPilot with its 10 Hz history (0.235 and 0.216 over the 2 Hz field): it was
post-trained on NAVSIM `navtrain` and pre-trained partly on OpenScene, which
*is* the NAVSIM dataset. Qwen-Drive 1.0 reports navtest as its own headline
planning result, so its adapter declares `navtrain` too — read off the report
rather than sourced from code, and the one line to correct if the paper says
otherwise. OpenPilot, Alpamayo and VisionPilot meet `navtest` zero-shot. That is the protocol NAVSIM intends — it is not cheating —
but it means both NAVSIM columns conflate "better planner" with
"in-distribution versus zero-shot". Sourced at
the pinned sha: `navsim_v2/scripts/training/train_drive_jepa_perception_based.sh`
sets `TRAIN_TEST_SPLIT=navtrain` for the same agent whose checkpoint paddock
loads. Detail in [docs/models/comparison.md](comparison.md).

✧ The published architecture reads one front camera; the adapter declares four,
because it runs the NAVSIM perception-based variant whose BEVFormer stage is
multi-camera. Unreconciled — and since a simulator renders exactly what is
declared, it is worth measuring before the next closed-loop sweep.
