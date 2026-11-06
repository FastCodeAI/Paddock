# Drive-JEPA, perception-free

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [linhanwang/Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) at `e21f474` |
| Adapter | [`adapters/models/drivejepa_free/`](../../src/paddock/adapters/models/drivejepa_free/) |
| Environment | shell, Python 3.9 |
| Cameras | `CAM_FRONT` |
| Plan | waypoints, 0.5 s apart, 4 s ahead |
| Paddock can run it on | [comma2k19](../benchmarks/comma2k19.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

## What it is

[Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) ships two agents. The
**perception-based** one — four cameras, BEVFormer, an anchor scorer and a lidar
mounting — is `adapters/models/drivejepa/`, and it is what paddock runs on
NAVSIM v2, nuScenes, KITScenes, neuroNCAP and Bench2Drive. This entry is the
other one:

```
front camera, 2 frames (t-0.5 s, t)          ego status, 8-D
        |                                          |
   ImageNet norm                                   |
        |                                          |
 V-JEPA-2 ViT-L  (256x512, patch 16, 2-frame temporal axis)
        |
 rearrange -> (B, D, 16, 32) -> AvgPool2d(2,2) -> 8x16 = 128 tokens
        |                                          |
        +------------ 129 key/value tokens --------+
                             |
              transformer decoder (3 layers, d_model 256)
                             |
                   TrajectoryHead -> 8 x (x, y, heading)
```

No map, no lidar, no annotations, one camera. That is why it exists here: a
dashcam dataset can feed it, and cannot feed the sibling.

Pinned in `repos.yaml` at `e21f474`, with the three patches listed there. The
checkout is imported as a library from `navsim_v1/`; the agent, its feature
builder, its model and `navsim.common.dataclasses` all run unmodified, which is
what makes the numbers a property of the released model rather than of a
reimplementation.

## Why a second registry entry and not a flag

`runtime:` is global to a model. This agent lives in `navsim_v1/`, not
`navsim_v2/`; it loads a different checkpoint; and its config path resolves
against the **process working directory**, so `cwd` differs too. One entry
serving both would need a `runtime:` block that depends on which benchmark is
running — the shape of thing that silently scores the wrong weights. Two entries
keeps every existing Drive-JEPA number bit-identical.

## Weights and environment

Two files, both required, and they are **not** interchangeable: the agent
constructor builds the V-JEPA encoder *from* the pretrain file and only then
overwrites everything with the agent checkpoint. Supplying only the `.ckpt`
fails at load.

| file | size | role |
|---|---|---|
| `vitl_merge_3dataset_e50.pt` | 4.8 GB | V-JEPA-2 ViT-L pretrain weights |
| `drive_jepa_perception_free_agent_vitl.ckpt` | 3.5 GB | the trained agent (Lightning; ~1.2 GB of it is the `state_dict`, the rest optimizer state) |

`DriveJEPAAgent.initialize()` strips the `agent.` prefix off the Lightning keys
itself, so the released checkpoint is passed as-is.

`setup.sh` builds a plain **venv**, where the perception-based sibling needs
conda. That is the practical payoff of this variant: no BEVFormer means no
mmcv-full and no mmdet, and those two — pinned against torch 2.1/cu121 with a
compiled deformable-attention kernel that has no SASS above sm_86 — are what
forced a conda toolchain over there. Everything here installs from wheels.

Two install traps worth keeping:

- `av==11.0.0` has no cp39 wheel on the default index and falls back to building
  against a system ffmpeg. With none present, `pkg-config` fails on
  `libavformat` and takes the **entire** `pip install -r requirements.txt` down
  with it. `av==12.3.0` ships prebuilt ffmpeg binaries; `setup.sh` pins it.
- `huggingface-cli download` is deprecated in current `huggingface_hub` and
  silently no-ops, exiting 0 having fetched nothing. Use `hf download`, one
  invocation per file — a second `--include` pattern is parsed as a positional
  argument and ignored.

Settings, all in `model.yaml` `runtime.env`:

| variable | default | meaning |
|---|---|---|
| `DRIVEJEPA_FREE_CHECKPOINT` | `{weights}/drive-jepa/drive_jepa_perception_free_agent_vitl.ckpt` | the trained agent |
| `DRIVEJEPA_VJEPA_WEIGHTS` | `{weights}/drive-jepa/vitl_merge_3dataset_e50.pt` | the encoder it is built on |
| `NAVSIM_DEVKIT_ROOT` | `{repo}/navsim_v1` | devkit root |

## Three repository details that are invisible from the README

Each of these cost the reference investigation real time, and none of them raise
when you get them wrong.

**1. The config path is CWD-relative.** `drive_jepa_model.py` does
`fname = "./vjepa2/configs/eval/vitl/in1k.yaml"`, resolved against the process
working directory rather than the module. `model.yaml` sets
`cwd: "{repo}/navsim_v1"` for exactly this. (A completed five-minute analysis
run was once lost writing its output inside the repo tree for the same reason.)

**2. The nuPlan crop is hard-coded in the feature builder.**
`DriveJEPAFeatureDIBuilder._get_camera_feature` does
`cameras[-1].cam_f0.image[28:-28]` unconditionally before resizing to 512x256.
An image that is already the right size silently loses 56 rows. The adapter
FOV-matches the frame and then **pre-pads** it by 28 replicated rows top and
bottom, so the builder's own slice removes exactly the padding.

**3. `cameras[-2]` is the OLDER frame.** `forward` builds its temporal axis as
`cat([camera_feature_2, camera_feature_1])`, so index −2 is t−0.5 s and −1 is
now. Reversed, nothing raises and the endpoint moves about 1.1 m — an error
that survives every smoke test.

One more, from the shipped scripts rather than the code: the released evaluation
overrides the agent yaml on the command line with `num_keyval=129`,
`freeze_encoder=false` and `double_image=true`. The yaml defaults are 257,
`true` and `false`. `num_keyval` sizes the decoder's key/value bank, so the
yaml default does not merely score worse — the checkpoint will not load into it.
`NUM_KEYVAL` in `policy.py` carries the evaluation value and a test pins it.

## Input contract and how paddock fills it

**The crop is derived, not transcribed.** nuPlan's CAM_F0 is 1920×1080 at
f ≈ 1545 px, cropped to 1920×1024 by the builder: hFOV 63.7°, vFOV 36.7°,
aspect 1.875, then squashed to 512×256. `crop_box()` stores those two angles as
tangents and asks the *actual* camera's focal length what window subtends them,
centred on the principal point and clamped to the image. On comma2k19's EON
(1164×874, f = 910, hFOV 65.2°, vFOV 51.2°) that is a 1131×603 window — aspect
1.876, so the model's training squash is reproduced rather than added to.

A camera that cannot supply the full angle is clamped, not padded, and
`diagnostics()` publishes the achieved `crop_hfov_deg` / `crop_vfov_deg` so that
becomes visible rather than merely disappointing. Padding would widen a field of
view with pixels that were never imaged.

**The pair is selected by timestamp,** not by taking the last two requests.
comma2k19 feeds 20 Hz; the last two requests are 50 ms apart, and the encoder's
temporal difference is real input — making the two frames identical cost the
reference nearly as much as blanking the camera outright.

**Acceleration is derived where the source has no channel.** `status_feature` is
`[driving_command(4), ego_velocity(2), ego_acceleration(2)]`. comma2k19 fills
speed (`canbus[13]`) and yaw rate (`canbus[12]`) but leaves the acceleration
slot at zero, and a constant zero is a different claim from "not measured" —
zeroing those fields moved the reference's endpoint by 0.63 m. So `ax` is the
least-squares slope of the recorded speed over the last 0.5 s when no channel is
present, `ay` is the centripetal term `v · yaw_rate`, and `diagnostics()` reports
`accel_source`. A source that does publish acceleration is preferred.

**No lidar.** The perception-free path never projects anything, so there is no
`require_lidar2ego()` and `lidars` are placeholders. The existing `drivejepa`
policy does not read lidar2ego either: its BEVFormer branch computes `lidar2img`,
but NAVSIM's lidar frame is the ego frame, so it places cameras by `camera2ego`.

**Ego frame** is NAVSIM/nuPlan's: x forward, y left, heading CCW — already
paddock's, so the eight output poses are returned unresampled as
`Trajectory(xy, dt=0.5)`.

## Running it

```bash
PADDOCK_REPO=<repos>/Drive-JEPA bash src/paddock/adapters/models/drivejepa_free/setup.sh
paddock doctor --model drivejepa_free
paddock verify --model drivejepa_free --bench comma2k19
paddock run --model drivejepa_free --bench comma2k19 --limit 20
paddock run --model drivejepa_free --bench comma2k19
```

Inference costs 19–31 ms a frame batched on an H200; VRAM is not the constraint.
`observe()` is record-only, so comma2k19's 160-observation window is one ViT-L
forward pass per sample, not 160.

## Reference measurements, and how to read the headline

These come from a standalone investigation under the checkout's own `.tmp/`,
**not** from a paddock run: three comma2k19 segments spanning 11–31 m/s, 119
analysis samples, all `driving_command = straight`.

**Expect a bad ADE and report it plainly.** Raw ADE was **16.68 m**. Rescaling
each predicted path so its 4 s reach matches ground truth drops it to
**2.19 m**, FDE 2.50, lateral error 1.26. The geometry transfers; the
longitudinal profile does not.

The cause is a speed prior, not a wiring error. Sweeping the input ego speed
with everything else held fixed:

| input speed | 4 s reach | implied speed |
|---|---|---|
| 5 m/s | 21.2 m | 5.3 |
| 11 | 42.2 | 10.6 |
| 14 | 52.0 | 13.0 |
| 20 | 61.7 | 15.4 |
| 25 | 63.8 | 15.9 |
| 30 | 64.1 | 16.0 |
| 40 | 63.8 | 16.0 |

The response is near-unity to ~14 m/s and completely flat above 25. The model
has no representation of "faster than about 16 m/s", which is what NAVSIM's
urban nuPlan distribution contains. comma2k19 is 33 h of Californian freeway.
Split by true speed, the error is *entirely* longitudinal at the top end — in
the 20–26 m/s bin, lateral error is 0.42 m while FDE is 34.3 m — and the model
beats constant velocity outright below 8 m/s (FDE 2.18 against 4.97), which is
the regime it was trained for:

| true speed | n | model FDE | CV FDE | lateral err | longitudinal bias @4 s |
|---|---|---|---|---|---|
| 0–8 m/s | 16 | **2.18** | 4.97 | 0.26 | −0.67 |
| 8–14 | 20 | **4.20** | 4.29 | 0.62 | −3.81 |
| 14–20 | 23 | 8.14 | 2.36 | 1.22 | −7.81 |
| 20–26 | 16 | 34.28 | 1.76 | 0.42 | −34.27 |
| 26–40 | 44 | 61.98 | 4.97 | 2.49 | −61.67 |

This is why `lateral_at_arclength_mae` and `path_length_ratio` exist on this
benchmark and why `docs/benchmarks/comma2k19.md` refuses to report ADE alone:
this model is the clearest case in the repo of a number that is simultaneously
terrible and informative. Read the comma2k19 row with those two columns or not
at all.

Ablations establishing that the inputs are live (mean shift of the 4 s endpoint
against the unperturbed plan):

| ablation | endpoint shift | ADE | reading |
|---|---|---|---|
| baseline | — | 16.68 | |
| status zeroed | 43.25 m | 41.97 | ego status dominates |
| velocity zeroed | 42.90 m | 41.82 | …and it is almost entirely velocity |
| image zeroed | 2.54 m | 18.67 | the camera is genuinely read |
| frames identical | 2.03 m | 18.46 | the *temporal* difference is used, not one frame |
| frames swapped | 1.12 m | 16.97 | order matters, and the chosen one is better |
| image mirrored | 1.04 m | 16.83 | lateral image content reaches lateral output |
| accel zeroed | 0.63 m | 16.85 | small, not nothing |
| command = left | 0.17 m | 16.63 | lateral shift **+0.078 m** (toward +y) |
| command = right | 0.26 m | 16.56 | lateral shift **−0.124 m** (toward −y) |

The last two independently confirm the `[left, straight, right, unknown]`
ordering `COMMAND_ONE_HOT` assumes. Note the caveat the reference states on
itself: comma2k19 is nearly all lane-keeping, so the command conclusions rest on
the ablation and not on real turning data.

Two differences from that investigation, both deliberate, so a disagreement in
the paddock numbers has somewhere to start:

- **The crop.** The reference cropped 1164×582 — the widest exact 2:1 window —
  giving hFOV 65.2° and vFOV 35.4°. This adapter derives 1131×603 from the
  target angles instead: hFOV 63.7°, vFOV 36.7°, both within a degree of
  nuPlan's. Closer to the training distribution, and correct on a camera that is
  not the EON.
- **Acceleration.** The reference differenced ECEF velocity centrally; this
  fits a slope to the recorded odometer over 0.5 s. Both are kinematic and
  gravity-free, which is what the model trained on.

Neither is expected to move the headline: the camera contributes about 2 m of
the 16.68 and acceleration about 0.6, against a longitudinal bias of 60 m at
freeway speed.
