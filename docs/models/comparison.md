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

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | AutoE2E ᵘ |
|---|---|---|---|---|---|
| **Camera views @ freq** | 1 front, 2 frames (t, t−1) | 4 (front_left, front_wide, front_right, front_tele); rig is 7-cam. AR1 paper used 2 | 2 (road + wide), 2 frames, 20 Hz | 1 mono front, 52–55° HFOV, 1–2 MP, 10+ Hz on Orin | 7 surround, 10 Hz |
| **Per-camera input** | 512×256 | 448×280 (→160 tokens/img); triplane tokenizer fixes 288 tokens/timestep regardless of camera count | 512×256 (medmodel); 1024×512 (bigmodel) | 1024×512 (2:1) all three nets; AutoDrive adds a 50° centre crop | 256×256 |
| **Other inputs** | ego state (NAVSIM protocol) | text prompt; egomotion history (3D translation + 9D rotation, multi-timestep); optional route | desire/nav command, 512-d feature history. **No speed channel at all** | vehicle speed | rendered BEV nav-map tile; 256-d egomotion history (6.4 s @ 10 Hz); 896-d visual memory (64×14); camera projection operator |

## Outputs and architecture

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | AutoE2E ᵘ |
|---|---|---|---|---|---|
| **Output @ freq** | trajectory from an 8192-entry anchor vocabulary | trajectory @ 6.4 s + chain-of-causation trace; 99 ms end-to-end | 33 waypoints over 10 s @ 20 Hz + lane lines/lead | AutoSteer: ego-path waypoints in camera coords, **no time parameterisation**; AutoSpeed: CIPO boxes (3 classes); AutoDrive: CIPO distance + curvature + presence prob | 128-d = 64 steps × (accel, curvature), 6.4 s @ 10 Hz |
| **Architecture** | V-JEPA-2-init ViT-L encoder + proposal-centric planner; MTD + momentum-aware selection | Cosmos-Reason2 + diffusion trajectory decoder (AR1: Cosmos-Reason VLM + traj decoder) | supercombo, a single ONNX graph | 3 separate conv nets, all YOLO-family 'n' variant | SwinV2-T or ConvNextV2-T + fusion (concat / cross-attn / BEV) + planner GRU |
| **Parameters** | 307 M (encoder + planner) | **10 B** (34 B for Super) | not published | **28.1 M** — AutoSteer 8.22 + AutoSpeed 9.33 + AutoDrive 10.58 ᵐ | 35.3 M (concat/cross-attn); 69.7 M (BEV, default) |
| **Size** | 1.23 GB fp32 / 0.61 fp16 ᶜ | 20 GB bf16 ᶜ | **0.061 GB** (`driving_supercombo.onnx`); 1.76 GB big variant; 0.0075 GB dmonitoring ᵐ | 0.112 GB fp32 / 0.028 INT8 ᵐ | 0.141 / 0.279 GB fp32 ᶜ |

Note the four-order-of-magnitude spread in the size row, and then read the
neuroNCAP column of the support matrix, where the top three are statistically
indistinguishable.

## Availability

| | Drive-JEPA | Alpamayo 1.5 | OpenPilot | VisionPilot | AutoE2E ᵘ |
|---|---|---|---|---|---|
| **Training code** | yes | yes (Apache-2.0) | **no** — comma trains offline and ships weights | yes | yes |
| **Inference code** | yes | yes, + AlpaSim integration | yes (`modeld`, MIT) | yes (Python + C++, ONNX Runtime/TensorRT, ROS2/Zenoh) | yes |
| **Checkpoints** | yes (HF) | yes — `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo-1.5-10B`. **OpenMDW-1.1, non-commercial**; commercial on request | yes — git-LFS in repo, MIT | yes — `.pth` + ONNX fp32 + ONNX INT8, **Apache-2.0, commercial OK** | **none** |

The licence row is the one to read before planning a product. Alpamayo's weights
are non-commercial by default; VisionPilot's and OpenPilot's are not.

## Training data — the important table

| | Pre-training | Post-training |
|---|---|---|
| **Drive-JEPA** | CoVLA + DrivingDojo + **OpenScene**, 208 h, 8-frame clips @512×256, 2 Hz; encoder init from V-JEPA 2 | **NAVSIM `navtrain`** (+ **CARLA** for Bench2Drive) |
| **Alpamayo 1.5** | NVIDIA internal: 80,000 h, >1 B images, 2,500+ cities / 25 countries | chain-of-causation traces: 700 K (AR1) / 3 M (1.5); RL via Cosmos-RL / AlpaGym |
| **OpenPilot** | comma internal fleet: 400 k × 1-min segments ≈ 6,670 h @5 Hz, not released | on-policy rollouts in reprojective sim + world model; fleet deployment, 500 users, 87 k trips |
| **VisionPilot** | OpenLane (AutoSpeed CIPO); TuSimple + OpenLane + CurveLanes (AutoSteer); AutoDrive warm-starts from the AutoSpeed backbone | none — single-stage supervised |
| **AutoE2E** ᵘ | KITScenes (unconfirmed) | not documented |

---

## What this changes about reading the support matrix

**Drive-JEPA is the only model with training exposure to a paddock benchmark's
dataset, and it leads that benchmark by a wide margin.**

It was post-trained on NAVSIM `navtrain` and pre-trained partly on OpenScene —
which *is* the NAVSIM dataset. The other three met `navtest` **zero-shot**.
Drive-JEPA scores 0.843 EPDMS; the next model scores 0.608.

To be precise about what this is and is not: training on `navtrain` and testing
on `navtest` is **the protocol NAVSIM intends**. Drive-JEPA is not cheating; it
is playing the benchmark the way the benchmark is designed to be played. What
makes the column hard to read is that the other three are not playing it that
way at all. The 0.235 EPDMS gap therefore conflates *"better planner"* with
*"in-distribution versus zero-shot"*, and nothing on the page currently says so.

This compounds a caveat already recorded in
[navsim.md](../benchmarks/navsim.md): Drive-JEPA's native agent is
**scorer-aware**, reading the PDM simulator at inference time, which is why
paddock reports its proxy figure instead. Two independent advantages, both
pointing the same way.

**The same applies to Bench2Drive, before the sweep is run.** Drive-JEPA was
post-trained on CARLA. [bench2drive.md](../benchmarks/bench2drive.md) frames
that benchmark as a real→sim transfer test — *"models trained on real data
(nuScenes / NAVSIM) are out of distribution on CARLA frames, so expect them to
drive partway and slow"* — which would not be true of Drive-JEPA. Its adapter
exists and has not been run. **Worth deciding how to label that number before it
is produced, not after.**

**And AutoE2E may be trained on KITScenes.** If the unconfirmed pre-training row
is right, an AutoE2E KITScenes ADE would be in-distribution while all four
current models are zero-shot there. `docs/models/alpamayo15-eval-notes.md`
states plainly that neither Alpamayo nor Drive-JEPA was trained on KITScenes,
which is what makes that column a fair comparison today. Confirm before scoring
it.

None of this makes any recorded number wrong. It makes three of them mean
something different from what a reader assumes, which is the same class of
problem `sequences_sha256` and `valid: false` exist to prevent.

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

ᵐ measured · ᶜ computed from parameter count · ᵘ **AutoE2E is not a paddock
model** — its adapter is unmerged and no checkpoint exists publicly, so it has
no row in the support matrix. It is included here because it is under
consideration and its data belongs beside the others.

Everything not marked ᵐ comes from published papers, model cards and repository
inspection rather than from paddock. Paddock has measured none of it, which is
the gap [docs/design/model-card-and-latency.md](../design/model-card-and-latency.md)
proposes to close: declared numbers rot, and the fix is a fixture test asserting
the declaration against the artifact.
