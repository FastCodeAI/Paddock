# What these models are

The support matrix says how well each model drives. This says **what each one
is** — how large, what it consumes, what it emits, and what it was trained on.

Two reasons it matters beyond curiosity. First, a model bound for an automotive
chip needs its size and latency beside its score. Second — and this is the part
the score table cannot show — **these models did not all meet these benchmarks on
equal terms.** See [what this changes](#what-this-changes-about-reading-the-support-matrix).

The machine-readable subset of this lives in each `model.yaml` under
`capabilities:` and is printed by `paddock list models`. This page is the
narrative half: provenance, licences, and the things that do not fit a schema.

---

## Inputs

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 | UniAD (`uniad`; `uniad_b2d`) | VAD (`vad`; `vad_b2d`) |
|---|---|---|---|---|---|---|---|---|---|
| **Camera views @ freq** | front, 2 frames 0.5 s apart (JEPA encoder); back, front, left, right at the current frame (BEV stage) | 4 (front_left, front_wide, front_right, front_tele), 4 frames 0.1 s apart; paddock crops the tele slot out of CAM_FRONT. Rig is 7-cam. AR1 paper used 2 | 2 (road + wide), 2 frames, 20 Hz | 1 mono front, 52–55° HFOV, 1–2 MP, 10+ Hz on Orin | 8 surround (front/back wide, front/back narrow, 4 corners), single frame; paddock fills 6 of the 8 | 6 surround (front, front-left, front-right, rear-left, rear, rear-right), 8 frames 0.5 s apart | 3 (front, front-left, front-right), 4 frames each 0.5 s apart. `DrivingScene` refuses a missing view, so on a monocular source (comma2k19) the side slots get black frames; `QWENDRIVE_VIEWS=replicate` or `crop` fill them from the road camera instead, and the policy then re-declares `cameras` to match | 6 surround (nuScenes rig), current frame; the previous BEV grid and the track queries are recurrent state, so frames arrive 0.5 s apart | 6 surround (nuScenes rig), current frame; the previous BEV grid is recurrent state, frames 0.5 s apart |
| **Per-camera input** | 512×256 | 448×280 (→160 tokens/img); triplane tokenizer fixes 288 tokens/timestep regardless of camera count | 512×256 (medmodel); 1024×512 (bigmodel) | 1024×512 (2:1) all three nets; AutoDrive adds a 50° centre crop | 768×432 uint8 RGB | 512×512 (squashed); the current front also 1024×1024 | pixel budget on a 32 px grid: current ~720p (921,600 px), history ~320p (174,080 px) | 1600×900, padded to 1600×928; BGR, caffe mean | 1600×900 scaled to 1280×720, padded to 32; torchvision mean/std |
| **Other inputs** | ego state (NAVSIM protocol) | text prompt; egomotion history (3D translation + 9D rotation, multi-timestep); optional route | desire/nav command, 512-d feature history. **No speed channel at all** | vehicle speed | ego speed; optional LiDAR raster / SD map / driving command in the checkpoint, none in the released ONNX | 256-d egomotion history (6.4 s @ 10 Hz); optional route raster + 14-channel semantic map (450×300 @ 0.4 m; paddock sends the route where the benchmark has one, never the map); per-frame camera projections | 16 ego poses @10 Hz (1.5 s) + velocity + acceleration; ego status [v, a, one-hot command]; nav command in the prompt | 18-float CAN bus (pose, accel, rotation rate, velocity, heading) into the BEV queries; lidar→global pose and time for the tracker; command (3-way nuScenes, 6-way Bench2Drive) | 18-float CAN bus into the BEV queries; command (3-way nuScenes, 6-way Bench2Drive), which selects one of its plans |

## Outputs and architecture

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 | UniAD (`uniad`; `uniad_b2d`) | VAD (`vad`; `vad_b2d`) |
|---|---|---|---|---|---|---|---|---|---|
| **Output @ freq** | trajectory from an 8192-entry anchor vocabulary | trajectory @ 6.4 s + chain-of-causation trace; 99 ms end-to-end | 33 waypoints over 10 s @ 20 Hz + lane lines/lead | AutoSteer: ego-path waypoints in camera coords, **no time parameterisation**; AutoSpeed: CIPO boxes (3 classes); AutoDrive: CIPO distance + curvature + presence prob | K=3 × 6 waypoints @ 0.5 s (3 s) + mode logits + steer/accel/brake; plus 11 perception heads | 128-d = 64 steps × (accel, curvature), 6.4 s @ 10 Hz; paddock integrates them with upstream's rollout | 50 poses (x, y, heading) @10 Hz, 5 s; optional one-sentence rationale | 6 waypoints @ 0.5 s (3 s) of the lidar's position, after a collision optimiser over predicted occupancy; plus tracks, map, motion, occupancy | one plan per command, 6 offsets @ 0.5 s (3 s) of the lidar's position; plus agent boxes with motion, map vectors |
| **Architecture** | V-JEPA-2-init ViT-L encoder + proposal-centric planner; MTD + momentum-aware selection | Cosmos-Reason2 + diffusion trajectory decoder (AR1: Cosmos-Reason VLM + traj decoder) | supercombo, a single ONNX graph | 3 separate conv nets, all YOLO-family 'n' variant | ResNet-34 + FPN per camera, depth-gated IPM lift to one 800×500 BEV @ 0.2 m, 12 task heads + refiners (`DepthSegIPMNetV52`) | ResNet-50 + BEVFormer V2 T8 (300×200 BEV, frozen for the KITScenes fine-tune), deformable map/route fusion, GRU planner | Qwen3.5-4B VLM (unchanged) + 1 B flow-matching Planning Expert reading its KV caches; separate BEV perception head | ResNet-101-DCN + FPN, BEVFormer 200×200, track/map/motion/occupancy/planning heads | ResNet-50 + FPN, BEVFormer 200×200 (30 m × 60 m), vectorised agent and map queries, ego planning query |
| **Parameters** | 307 M (encoder + planner) | **10 B** (34 B for Super) | **30.0 M** ᵐ, counted in the pinned ONNX (comma gives 26.6 M for this generation) | **28.1 M** — AutoSteer 8.22 + AutoSpeed 9.33 + AutoDrive 10.58 ᵐ | 54 M (refiner incl.) ᵖ | 79.9 M ᵖ | **5.58 B** (4.54 B VLM + 1.04 B expert, counted from the checkpoint; the 125 M perception head is not loaded for planning) | 131.8 M ᶜ | 58.4 M ᶜ |
| **Size** | 1.23 GB fp32 / 0.61 fp16 ᶜ | 20 GB bf16 ᶜ | **0.061 GB** (`driving_supercombo.onnx`); 1.76 GB big variant; 0.0075 GB dmonitoring ᵐ | 0.112 GB fp32 / 0.028 INT8 ᵐ | 0.35 GB fp32 ONNX ᵐ; 2:4 sparse trunk | 0.32 GB fp32 ᶜ | 9.1 GB VLM + 2.1 GB per planner, bf16 ᶜ | 1.00 GB fp32 ᶜ (both checkpoints) | 0.71 GB (`vad`) / 0.70 GB (`vad_b2d`) fp32 ᶜ |

Note the four-order-of-magnitude spread in the size row, and then read the
neuroNCAP column of the support matrix, where the top three are statistically
indistinguishable.

## Availability

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | METEOR | AutoE2E | Qwen-Drive 1.0 | UniAD (`uniad`; `uniad_b2d`) | VAD (`vad`; `vad_b2d`) |
|---|---|---|---|---|---|---|---|---|---|
| **Training code** | yes | yes (Apache-2.0) | **no** — comma trains offline and ships weights | yes | yes (Apache-2.0), + autolabel factory | yes | **no** — inference code only | yes (Apache-2.0); Bench2DriveZoo's yes (**CC BY-NC-ND 4.0**) | yes (Apache-2.0); Bench2DriveZoo's yes (**CC BY-NC-ND 4.0**) |
| **Inference code** | yes | yes, + AlpaSim integration | yes (`modeld`, MIT) | yes (Python + C++, ONNX Runtime/TensorRT, ROS2/Zenoh) | yes (Python + C++ TensorRT runtimes; plain ONNX runs in onnxruntime) | yes | yes (Apache-2.0) | yes (test pipeline; no closed-loop agent upstream; Bench2DriveZoo has a CARLA agent) | yes (as UniAD) |
| **Checkpoints** | yes (HF) | yes — `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo-1.5-10B`. **OpenMDW-1.1, non-commercial**; commercial on request | yes — git-LFS in repo, MIT | yes — `.pth` + ONNX fp32 + ONNX INT8, **Apache-2.0, commercial OK** | yes — `AutowareFoundation/meteor` (`.pt` + ONNX), **Apache-2.0** | yes — `AutowareFoundation/auto_e2e` v1.0 (`nuplan-epoch-5`, `kitscenes-epoch-5`); code Apache-2.0, checkpoint licence "other" (model card), BEVFormer initialisation NOASSERTION | yes — `Qwen/Qwen-Drive-1.0-4B` (VLM + planner-sft + planner-rl + perception), **Apache-2.0, commercial OK** | yes — GitHub release `uniad_base_e2e.pth`; `rethinklab/Bench2DriveZoo` on Hugging Face | yes — Google Drive (README link); `rethinklab/Bench2DriveZoo` on Hugging Face |

The licence row is the one to read before planning a product. Alpamayo's weights
are non-commercial by default; VisionPilot's and OpenPilot's are not.

## Training data — the important table

| | Pre-training | Post-training |
|---|---|---|
| **Drive-JEPA** | encoder init from V-JEPA 2 (VideoMix22M: SSv2, Kinetics, HowTo100M, YT-Temporal-1B, ImageNet); then CoVLA + DrivingDojo + **OpenScene `trainval`**, 330 h (208 h in paper v1), front camera, 8-frame clips @512×256, 2 Hz, no labels | **NAVSIM `navtrain`**, with simulator-scored vocabulary targets (+ **CARLA** for Bench2Drive: no data, split or checkpoint released; paddock runs the NAVSIM-v2 checkpoint there) |
| **Alpamayo 1.5** | VLM base Cosmos-Reason2-8B, itself post-trained from Qwen3-VL-8B-Instruct; NVIDIA internal driving: 80,000 h, >1 B images, US + EU (2,500+ cities / 25 countries per the AR1 paper); **17 public driving sets** on the model card, including **NAVSIM**-ReCogDrive, **NAVSIM**-Traj and five built on **nuScenes** (nuScenesQA, DriveLM, nuInstruct, OmniDrive, Talk2Car), splits not given | chain-of-causation traces: 700 K (AR1) / 3 M (1.5), about 10 % human, the rest VLM-labelled; RL with a reasoning reward (Cosmos-RL GRPO in AR1; whether 1.5's RL was closed loop is not stated) |
| **OpenPilot** | comma fleet, not released: the world model on 2.5 M minutes of video at 5 Hz; the vision model on fleet segments (500 k or 2.2 M: comma's posts disagree) | the policy trained entirely inside the learned simulator, a world model (openpilot 0.11's "WMI" model). The 400 k segments ≈ 6,670 h, 500 users and 87 k trips are the 2025 paper's experiment and field test, not this model |
| **VisionPilot** | AutoSpeed: OpenLane (Waymo Open images), from scratch. AutoSteer: TuSimple + CurveLanes + OpenLane images with OpenLane-V labels, from scratch | AutoDrive: from AutoSpeed's backbone (shown in the weights), then **ZOD** Sequences frame pairs labelled by AutoSpeed, ZOD's radar and its steering angle |
| **METEOR** | ResNet-34 from **ImageNet**; TIER IV Co-MLOps DRS recordings (Japan + a US subset) and a 7-camera minibus corpus @ 5 Hz, every label from CoMET (TIER IV's ensemble of 12 models, unnamed); Cosmos Transfer re-renders of those scenes train perception only. 953 GB / 12,595 scenes is the disk, not the training list | released v157: a 2:4-sparse fine-tune from v151 with the planner frozen and RL off (`rl_w 0.0`, read from the checkpoint); the GRPO stage is not shown to be in its ancestry |
| **AutoE2E** | ResNet-50 via **COCO** (FCOS); BEVFormer V2 R50 T8 on **nuScenes train**, frozen in every later stage; then nuPlan v1.1 **train** sensor logs (1,085 logs, none of `navtest`'s) | **KITScenes** train fine-tune (404 scenes), camera BEV still frozen (`kitscenes-epoch-5`, the checkpoint paddock runs) |
| **Qwen-Drive 1.0** | Qwen3.5-4B's own corpus (not published); VLM adapted with a perception head on **nuScenes** official train and **OpenScene** trainval minus 16 logs, plus 24 public driving-VQA sets (about 20 % of each; some of their training splits hold nuScenes val scenes and, in Impromptu VLA, **navtest** frames) and 30K in-house examples from China | planning expert (`planner-sft`, what paddock runs): **NAVSIM**, OpenScene, WOD-E2E, PhysicalAI-AV, no split named; `planner-rl`: 15K `navtrain`, 15K PhysicalAI-AV, 479 WOD-E2E validation scenarios |
| **UniAD** (`uniad`) | BEVFormer-Base's released nuScenes detector (stage 1's `load_from`), so ResNet-101 trained on **nuScenes** through BEVFormer | stage 1 (track + map) and stage 2 (end to end, planning) on **nuScenes `train`**, reported on `val` |
| **VAD** (`vad`) | ResNet-50 from torchvision's **ImageNet** weights | stage 1 (perception, prediction) and stage 2 (planning) on **nuScenes `train`**, reported on `val` |
| **UniAD-B2D** (`uniad_b2d`) | not stated at the pinned sha (its stage configs set no `load_from`; Bench2DriveZoo's BEVFormer starts from FCOS3D's **nuScenes** weights) | **Bench2Drive base** (1,000 CARLA clips by the Think2Drive RL expert) less its validation clips |
| **VAD-B2D** (`vad_b2d`) | ResNet-50 from torchvision's **ImageNet** weights | **Bench2Drive base** less its validation clips |

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

**Drive-JEPA trained on NAVSIM's own data, and it leads that benchmark by 0.235
EPDMS.** It was post-trained on NAVSIM `navtrain` and pre-trained partly on
OpenScene — which *is* the NAVSIM dataset. Qwen-Drive, AutoE2E and Alpamayo 1.5
also trained on NAVSIM or nuPlan data (their sections below); OpenPilot and
VisionPilot meet `navtest` zero-shot.
(OpenPilot × comma2k19 below is the undeclared case: same fleet, no way to
check.)

The pre-training used OpenScene's `trainval` split, not its `test` split, which
`navtest` is drawn from: paper v2 (arXiv 2601.22032, §3, "OpenScene(trainval
split)") and the author in [Drive-JEPA issue #10](https://github.com/linhanwang/Drive-JEPA/issues/10#issuecomment-4120088355).
Checked here too: in the scene filters vendored at the pinned sha, the 1,192
`navtrain` logs and the 136 `navtest` logs share none. So the overlap is with
NAVSIM's training data, not its test scenes.

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

**Every Drive-JEPA row on this page is the released checkpoint.** Paddock keeps
results for released checkpoints only; one we trained ourselves is kept, with
its results, beside its training code.

**Every Drive-JEPA row on this page is the perception-*based* v2 agent**,
comma2k19's included. Until November 2026 the comma2k19 cell came from
Drive-JEPA's perception-free NAVSIM v1 agent, a separate registry entry with its
own checkpoint, because the perception-based adapter then required a lidar
mounting that comma2k19 does not have. That agent's rows stay in `results/`,
retired; see [drivejepa.md](drivejepa.md#on-comma2k19).

### Drive-JEPA × nuplan — most of nuPlan mini is in its training logs

**Drive-JEPA's nuPlan cell is adapted, not zero-shot.** Paddock's nuPlan
benchmark scores nuPlan's `mini` split. NAVSIM's `navmini` scene filter names 62
mini logs (`train_test_split/scene_filter/navmini.yaml`, vendored in Drive-JEPA at
the pinned sha), and 52 of them are also `navtrain` logs. So Drive-JEPA's planner
was fine-tuned on those drives, and its encoder was pre-trained on all of OpenScene
`trainval`, which holds them too. The other 10 are `navtest` logs. Which of the 25
scored anchors fall in the 52 is not known on this machine: the anchor list lives on
the site that ran the benchmark. The nuPlan column is unranked and every run in it is
stored `valid: false` already, so no rank changes; the A says why this number is not
a zero-shot one.

### Qwen-Drive 1.0 × navsim

**Qwen-Drive's NAVSIM number is in-distribution, and it is the number the
vendor chose as its headline.** The technical report standardises trajectories
from several public driving datasets into one waypoint representation and
reports NAVSIM v1.1 navtest PDMS (87.8 sft direct, 88.2 sft reasoning, 90.7 rl)
alongside Waymo E2E and NVIDIA PhysicalAI; a model reporting navtest as its
first planning result trained on `navtrain`, and `trained_on` declares that.
The NAVSIM frames, cameras (`cam_f0/l0/r0`) and 10 Hz ego history paddock feeds
it are the ones its own scene files are built from, so on this column it plays
the benchmark as designed, like Drive-JEPA, while OpenPilot and VisionPilot
meet navtest zero-shot (Alpamayo 1.5 trained on NAVSIM data too; see *Alpamayo
1.5 × navsim*).

**What is not sourced.** The report (arXiv 2609.00111, §2.3) names splits for
the perception data and the RL stage, but not for the planning expert's NAVSIM
data; `navtrain` is the assumption there. Separately, one of the 24 driving-VQA
sets the VLM trained on, Impromptu VLA, splits its data its own way, and its
training split holds frames from 52 of `navtest`'s 136 logs (checked against
NAVSIM's `navtest` log list). Whether Qwen-Drive's 20 % sample of it included
them is not stated.

KITScenes and comma2k19 are zero-shot for this model; neuroNCAP is not (see
*Qwen-Drive 1.0 × nuscenes*). Its KITScenes cameras sit at wider bearings and
field of view than the nuPlan rig it was trained on (see [qwendrive.md](qwendrive.md)). comma2k19
is the unequal column in a different way: nothing in the corpus is a
windscreen-mounted mono camera on a Californian highway, and the model reads it
with that one frame in its front slot and two black side slots.
First place on that board is therefore a claim about a model driven outside
both its rig and its domain.

### Qwen-Drive 1.0 × nuscenes — a VLM trained on nuScenes, val scenes included

**Qwen-Drive's nuScenes and neuroNCAP cells are adapted, not zero-shot.** Its VLM
was trained with a perception head on nuScenes' official train split ("We adopt
the official nuScenes split", report §2.3.1) and on driving-VQA sets built on
nuScenes. "We use training splits for these datasets", the report says, but two
of those training splits re-split nuScenes train and val together: NuInstruct's
and Talk2Car's training splits hold 123 of the 150 val scenes between them, and
12 of neuroNCAP's 14 scenes, frontal 0103 among them (counted from their released
files). Which 20 % of each set was sampled is not stated. The planning expert
paddock runs did not train on nuScenes. Its nuScenes number, the lowest error
here and the only learned model to beat constant velocity,
should be read with this in mind.

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
distribution. Its 0.843 is third in that column, behind Qwen-Drive and Alpamayo
1.5, which meet comma2k19 outside both their rig and their domain.

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

### AutoE2E × navsim — nuPlan-trained, scored on nuPlan's test logs

The checkpoint paddock runs, `kitscenes-epoch-5`, is the release's nuPlan model
fine-tuned on KITScenes. NAVSIM's scenes are OpenScene, which is nuPlan
resampled to 2 Hz, and paddock feeds AutoE2E from nuPlan's own 10 Hz logs. So
its 0.500 EPDMS on `navtest` is not zero-shot in the way VisionPilot's or
OpenPilot's is: same rig, same cities, same camera geometry it was pre-trained
on. It did not see the *logs*: the release trains on "nuPlan v1.1 train sensor
logs" (model card at c553a2e1), the code pins that list's 1,085 logs, and none
of `navtest`'s 136 logs is among them, checked against Motional's published
train inventory. 44 of nuPlan mini's 64 logs are, which would matter if AutoE2E
were run on the nuPlan benchmark; it is not today.

Its inputs differ from training too: NAVSIM sends no route and paddock no map,
which is upstream's own KITScenes test track (`camera_only_missing_map_route`),
while every training and validation sample had both.

The fine-tune pulls the other way: the last thing the checkpoint learned is a
different rig, KIT's ring cameras seen from a reference frame 2.1 m up. The
release's `nuplan-epoch-5` would be the in-distribution choice for this column
and is not what was scored. Its KITScenes number is in-distribution while the
other models there are zero-shot; see *AutoE2E × kitscenes*, below.

### AutoE2E × kitscenes — fine-tuned on KITScenes, given the map and the logged route

**AutoE2E's KITScenes cell is adapted, not zero-shot.** The checkpoint paddock
runs, `kitscenes-epoch-5`, is the nuPlan model fine-tuned on KITScenes'
**train** split: upstream's pinned geographic split
(`Model/data_parsing/kit_scenes/pinned_splits/default_geo_split_v1_0` at
`40a75cf8`) lists 534 scenes, 533 exist in the release, and 404 had samples,
of which 40 were held out as train-dev. None of the benchmark's 140 scenes is
among them, which was checked by scene id. 117 of the 140 are that split's 117
validation scenes, which the release reports its own KITScenes figures on; the
epoch was chosen on the train-dev scenes, not on these (model card, "These sets
were used to choose the epoch"). The other 23 are `overlap_train_val` scenes, in neither
of upstream's lists but in an area that overlaps train. The 223 keyframes split
186 / 37 between the two, and the AutoE2E page reports both.

Its inputs are the ones upstream validated it with, not the other rows': the
scene's Lanelet2 map, and a route matched to the logged drive -- an oracle,
drawn from the future being scored (KITScenes' `route = "oracle"`, the
default). So the cell is in-distribution twice over: the rig, cities and
driving it was trained on, and a route no other row is given. Read it beside
the zero-shot rows, never as one of them. The AutoE2E page has the runs without
the route and without the map, which say how much of the number each carries.

### Alpamayo 1.5 × navsim — NAVSIM data in its public training sets

**Alpamayo 1.5's NAVSIM cells are adapted, not zero-shot.** Its model card
(nvidia/Alpamayo-1.5-10B, "Training Dataset") lists 17 public driving datasets
beside NVIDIA's own data, and two of them are NAVSIM data: NAVSIM-ReCogDrive and
NAVSIM-Traj. The card gives no split. ReCogDrive's published question-answer
pairs come from `navtrain`, which shares none of its logs with `navtest`, so
shared test frames are unlikely; that is argued from ReCogDrive's paper, not
stated by NVIDIA. Both NAVSIM cells already carried R for their 10 Hz history;
they now carry A as well.

### Alpamayo 1.5 × nuscenes — five nuScenes-built sets, splits not given

**Alpamayo 1.5's nuScenes and neuroNCAP cells are adapted, not zero-shot.** Five
of the public datasets on its model card are built on nuScenes images:
nuScenesQA, DriveLM, nuInstruct, OmniDrive and Talk2Car. Their targets are mostly
language (questions, answers, descriptions), not trajectories, and NVIDIA gives
no split. Some of those sets put nuScenes val scenes into their own training
splits (Talk2Car and nuInstruct re-split train and val together), so paddock's
val scenes, and neuroNCAP's 14, which are all val scenes, may have been seen.
That is argued; the card does not say.

### AutoE2E × nuscenes — a camera encoder trained on nuScenes train

**AutoE2E's nuScenes and neuroNCAP cells are adapted, though no scored scene was
in training.** Its camera encoder, ResNet-50 plus BEVFormer V2, is the official
BEVFormer V2 R50 T8 checkpoint, trained on nuScenes train only
(`nuscenes_infos_temporal_train.pkl` in BEVFormer's config; its training log runs
1,759 steps on 16 GPUs per epoch, which is train's 28,130 samples), and it stays
frozen through both of AutoE2E's own stages. So the weights that read the cameras
were fitted to the nuScenes rig and cities. Paddock's nuScenes benchmark scores
val, and all 14 neuroNCAP scenes are val scenes (checked against
`adapters/benchmarks/nuscenes/val_scenes.json`), so none was in training.

### METEOR × kitscenes — six cameras of eight, and the graph as shipped

METEOR trained on none of paddock's datasets, so it has no exposure section;
the unequal terms run the other way. It is an 8-camera network; KITScenes hands
it six, the two telephoto slots stay absent (the convention it was
trained with for one missing camera), it has no driving-command input in the
released ONNX, and that ONNX applies no ImageNet normalisation although its
training loader did (verified against the checkpoint's first conv; see
`docs/models/meteor.md`). Its 1.879 ADE is longitudinal above 13 m/s, where its
plans plateau near 35 m at 3 s, a prior from urban Japanese driving. The
front-only run made before KITScenes passed the ring scored 4.784 and is kept as an ablation: 2.9 m
of ADE was the missing cameras. Read the KITScenes cell as "the released model,
on a rig and a domain it never saw", which is the same caveat the other rows
carry in different proportions.

### METEOR × comma2k19 — one phone camera of eight, and perception that does not carry over

The released graph again, with the unequal term at its largest. comma2k19 is one
65° phone camera behind a windscreen at about 1.22 m; METEOR's rig is eight
cameras, 98° and 30° lenses at about 1.9 m, in left-hand traffic. The adapter
gives the phone frame two slots (the frame, and its centred 30° window in the
telephoto slot) and leaves six black with the calibration of METEOR's own rig,
as its training drops a camera (`METEOR_FRONT_ONLY`). That rule was chosen on
METEOR's own demo scene, where the front camera alone fed this way beats
constant velocity (1.01 against 1.37; all eight cameras 0.65), and re-measured
there through paddock's adapter. On comma2k19 its 30.167 ADE@3s is the bottom
of the column: the plan stops about 17 m ahead at every speed above 15 m/s.
An out-of-tree study of one comma2k19 segment traced that to perception, not
to the planner or the adapter: depth on this camera's road reads 3.4x too far
and the bird's-eye view holds road only about 20 m ahead. Read the cell as "the
released model's perception on a camera it never saw".
`docs/models/meteor.md` has the measurements.

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

### METEOR × bench2drive — a real-camera planner on CARLA's cameras, at its own rate

METEOR's Bench2Drive cell, 1.05 over the 50 routes, is the released graph
zero-shot: it trained on none of CARLA's towns or renders. It shares the routes,
the camera rig, the tracker and the early-stop rules with AutoE2E's cell, and is
asked at its declared 5 Hz where AutoE2E is asked at 10. Against the 20 Hz cells
(VisionPilot, constant velocity) it differs in the rate and in ending routes
early. The terms that are not its training's: six of its eight camera slots
filled, by 70 deg level cameras where it trained on 98 deg pitched ones; no
driving command, which the released graph has no input for; CARLA's vehicle
origin at the middle of the car, not the rear axle; right-hand traffic. On 43 of
the 50 routes it plans to stay where it starts. Read the cell as "the released
model on CARLA's cameras, from a standstill". Detail in
[meteor.md](meteor.md#reference-numbers) and
[docs/benchmarks/bench2drive.md](../benchmarks/bench2drive.md#notes-on-the-matrix-cells).

### METEOR × alpasim — six cameras, and a driver that is not tracked

METEOR's AlpaSim cell, 0.4232 on 30 scenes, shares the scenes, the referee
config and the scorer of the one-camera rows and nothing else. It was shown six
rendered cameras (`6cam-20Hz`); it is ranked with the one-camera rows and read
against the same constant-velocity floor. The cameras are pinhole renders
narrower than the recorded fisheyes, with the bonnet mask off. The mode is
chosen by the selector of METEOR's deployed runtime, where every other METEOR
cell takes the argmax. And the run came from a native driver that is not in
this repository, where the other METEOR cells came through the paddock node.
The graph and its pixel scaling are the same as elsewhere: the released ONNX,
as shipped. Read the cell as "the released model on AlpaSim's cameras, once".
Detail in
[meteor.md](meteor.md) and
[docs/benchmarks/alpasim.md](../benchmarks/alpasim.md).

### VisionPilot × everything — every figure but comma2k19's predates a constant that moved

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
**Every VisionPilot number on this page and in the README but comma2k19's was
produced with 1.0** and will not be reproduced by a re-run. Expect the open-loop rows to move most
and the closed-loop ones least: Bench2Drive and neuroNCAP are urban and slow
enough that the cap rarely binds, which is why the defect survived them.
`docs/models/visionpilot.md` has the derivation; the re-sweep is an open
item in `docs/ROADMAP.md`.

### UniAD × nuscenes — the benchmark is UniAD's own protocol, and its training set's twin

**UniAD's nuScenes cell is in-distribution, by design.** Both of its training stages read nuScenes `train`
(`nuscenes_infos_temporal_train.pkl` in `projects/configs/stage1_track_map/base_track_map.py` and
`stage2_e2e/base_e2e.py` at the pinned sha) and its paper reports the `val` split, which is the split
paddock's nuScenes benchmark scores. The scenes are held out; the cameras, the city streets, the rig and the
annotation conventions are the ones it trained on, and paddock's open-loop metric is UniAD's own (its L2 and
collision definitions). So this column measures it at home, the way NAVSIM measures Drive-JEPA at home,
while OpenPilot, VisionPilot and METEOR meet nuScenes zero-shot. Its stage 1 also starts from BEVFormer's
nuScenes detector, so the backbone saw nuScenes `train` twice. One protocol difference remains: UniAD's own
evaluation runs each val scene front to back, carrying its tracks and BEV state from the scene's first
frame; paddock feeds two seconds of keyframes before each scored one.

### VAD × nuscenes — trained on nuScenes train, scored on val

**VAD's nuScenes cell is in-distribution too.** Its two stages train on nuScenes `train`
(`vad_nuscenes_infos_temporal_train.pkl` in `projects/configs/VAD/VAD_base_stage_{1,2}.py` at the pinned sha)
and its paper reports `val`, the split paddock scores; the command it is given on this benchmark is derived
from the recorded future by VAD's own rule. Read the column the way the UniAD section above says: held-out
scenes from the training distribution, against other models seeing nuScenes for the first time. Like UniAD,
VAD's own evaluation runs whole scenes in order; paddock gives it two seconds of keyframes first.

### UniAD and VAD × hugsim — some of HUGSIM's nuScenes scenes are nuScenes training drives

HUGSIM reconstructs real drives as Gaussian-splatting scenes, 88 of its scenarios from nuScenes. They are not
all `val` scenes: `scene-0013` is in nuScenes `val`, but `scene-0166`, which paddock's HUGSIM notes name, is a
`train` scene, so UniAD and VAD trained on the very drive it reconstructs (the renders and the trajectories
driven through them are new). The split of the other nuScenes scenarios has not been checked here. HUGSIM
also renders the nuScenes camera rig on its KITTI-360, PandaSet and Waymo scenes, black behind the car on
KITTI-360 and Waymo, so those are out of distribution twice over. A 9 Nov 2026 A/B on three scenarios
against HUGSIM's own UniAD client gave the same outcomes as paddock's adapter (docs/models/uniad.md).

### UniAD-B2D × bench2drive — CARLA-trained, scored in CARLA

**`uniad_b2d` and `vad_b2d` are the only rows trained in the simulator Bench2Drive scores.** Bench2DriveZoo
trained them on Bench2Drive base: 1,000 CARLA clips driven by the Think2Drive reinforcement-learning expert,
less a validation split. No base clip carries an evaluation route's id, but checked on 29 Sep 2026 against
the base listing, 26 of the 220 evaluation routes have a base clip of the same scenario type passing within a
few metres of the route's scenario trigger, the same scenario instance under another id. The other rows meet
CARLA zero-shot from real data. Paddock drives them through its own camera rig (800×450 at paddock's mounts),
not the 1600×900 rig they trained on, and at their training rate of 2 Hz, where Bench2DriveZoo's own agent
asked every 20 Hz tick for its published 45.81 / 42.35 DS; see docs/models/uniad_b2d.md.

### VAD-B2D × bench2drive — the same training set and the same caveats

VAD-B2D shares UniAD-B2D's training data, its overlap with the evaluation routes and its rig difference: see
the section above. It plans one trajectory per Bench2Drive command, six in all, and the command picks the one
driven, so the command mapping (docs/models/vad_b2d.md) decides directly what it drives, more than for
UniAD-B2D, which reads the command only as an embedding.

## Provenance of these figures

ᵐ measured · ᶜ computed from parameter count · ᵖ published by the authors, not re-counted.

Everything not marked ᵐ comes from published papers, model cards and repository
inspection rather than from paddock. Paddock has measured none of it yet:
declared numbers rot, and the fix is a fixture test asserting the declaration
against the artifact.

## Notes on the models

Notes on support-matrix cells that are about a model rather than one benchmark.
Each starts with its mark. The notes about one benchmark are on that benchmark's
page, under "Support-matrix results".

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
correction moved ADE@3s from 15.09 to 5.28. **Every VisionPilot figure in the
support matrix but comma2k19's predates it** and will not be reproduced by a
re-run — the closed-loop ones least, since urban scenes rarely reach the cap, and
the KITScenes one most. The comma2k19 cell was re-run in November 2026. `docs/models/visionpilot.md` §"Curvature scale" has
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
camera-only ONNX. KITScenes hands it the six-camera ring, so only its two
telephoto slots are absent; it has no driving-command input. Fed the front
camera alone, as the harness did before 14 Sep 2026, it scored 4.784 -- the run is kept
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
otherwise. OpenPilot and VisionPilot meet `navtest` zero-shot; Alpamayo 1.5's model card lists NAVSIM data too. That is the protocol NAVSIM intends — it is not cheating —
but it means both NAVSIM columns conflate "better planner" with
"in-distribution versus zero-shot". Sourced at
the pinned sha: `navsim_v2/scripts/training/train_drive_jepa_perception_based.sh`
sets `TRAIN_TEST_SPLIT=navtrain` for the same agent whose checkpoint paddock
loads.

✧ The published architecture's encoder reads one front camera, at t and
t − 0.5 s (`drive_jepa_features.py`). The perception-based agent paddock runs also
has a BEV stage whose loader takes `cam_b0`, `cam_f0`, `cam_l0` and `cam_r0` at the
current frame (`bevformer/bev_feature_build.py`), so the adapter declares four. The
model then keeps only the front camera's projection (`drive_jepa_model.py:50`), so
the other three views' pixels never reach the plan, and the paper's "one front
camera" holds for the whole model.
