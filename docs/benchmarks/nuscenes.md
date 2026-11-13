# nuScenes val, open loop

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `nuscenes` |
| Primary metric | `l2_avg` |
| Adapter | [`adapters/benchmarks/nuscenes/`](../../src/paddock/adapters/benchmarks/nuscenes/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [alpamayo2edge](../models/alpamayo2edge.md), [autoe2e](../models/autoe2e.md), [constant_velocity](../models/constant_velocity.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [uniad](../models/uniad.md), [vad](../models/vad.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

`paddock run --bench nuscenes --model <drivejepa|visionpilot|alpamayo15|meteor>`

The nuScenes planning protocol that UniAD, VAD and their successors report: on
every val keyframe, plan 3 s ahead from the real past and compare with what the
human drove. Zero-shot for every model here except three, which carry A in the
support matrix: Qwen-Drive's VLM trained on nuScenes train and on VQA sets that
hold val scenes, Alpamayo 1.5's model card lists five nuScenes-built training sets
(splits not given), and AutoE2E's frozen camera encoder was trained on nuScenes
train. See docs/models/comparison.md.

**Scoring only some samples.** `PADDOCK_NUSCENES_SAMPLES=<id>,<id>,...` (ids as
`nuscenes/val/<scene>/<index>`) scores those samples and no others. Such a run is
stored `valid: false` as a subset, with or without `--limit`. The regression
gate's nuScenes subset runs this way (docs/gate.md). The list is recorded in
`config.sample_filter` as its size and digest.

## What is scored

* **Split:** the official val split (150 scenes, vendored from the devkit's
  `create_splits_scenes` into `val_scenes.json`), 6019 keyframes at 2 Hz. The
  5119 with six future keyframes (a full 3 s) are scored; the last six of each
  scene are dropped, where the literature pads with masks.
* **Sample:** up to four preceding keyframes plus the scored one (2 s of real
  history), or as many as a model's declared `history_s` asks for when that is
  longer, NAVSIM's rule (AutoE2E: 13, 6.5 s, recorded as `history_seconds` in the
  run's config). With `camera_history` set the window stays at four. So Drive-JEPA gets its four frames 0.5 s apart and Alpamayo a pose
  history to difference. Six cameras as `image_paths` with the dataset's
  intrinsics and `cam2ego`; LIDAR_TOP ego pose; CAN-bus 18-float status
  (nearest `pose` message); `lidar2ego` for models that project through the
  lidar frame. UniAD and VAD (and their Bench2Drive checkpoints) are sent their
  own datasets' versions of these instead, below.
* **Command:** VAD's threshold on the recorded future in the ego frame: lateral offset of the 3 s
  pose ≥ 2 m → left, ≤ −2 m → right, else straight (val: 88 / 5 / 7 %). Sent as
  the 4-way `command` and as `nav_maneuver` / `nav_distance_m`, which Alpamayo
  turns into its instruction text. UniAD and VAD get their own repos' rules (below).
* **Metrics** (per sample, aggregated by the runner):
  `l2_{1,2,3}s`, `l2_avg` — L2 at exactly that horizon (VAD style) and their
  mean; `l2_uniad_{1,2,3}s`, `l2_uniad_avg` — mean over every 0.5 s step up to
  the horizon (UniAD style; lower than the point value whenever error grows with
  the horizon, as it usually does, the convention published tables
  use); `col_*` likewise for collision, where the ego footprint (UniAD's
  4.084 × 1.85 m, heading from the plan's direction of travel) is tested
  against every annotated box of the future keyframe in the current ego frame
  (separating-axis test); `ade_3s`. Primary metric: `l2_avg`.

## The datasets' own inputs (UniAD, VAD)

```yaml
# adapters/models/uniad/model.yaml (vad: `nuscenes: vad`)
inputs:
  nuscenes: uniad
```

UniAD and VAD were trained, and their authors evaluate them, on what their converters write into an info
file and their datasets build from it. Four of those inputs are not what the harness builds for every other
model, so a model that declares `inputs.nuscenes` in its model.yaml is sent its dataset's instead
(`adapters/benchmarks/nuscenes/infos.py`), in windows and in scene order alike:

| input | every other model | a model with `inputs.nuscenes` (the converter / dataset) | on val's 6,019 keyframes |
|---|---|---|---|
| canbus 3:7 | the CAN-bus message's orientation | the LIDAR_TOP ego pose's quaternion as stored, sign and all; the datasets then put its w four times (`bev_planners.nuscenes_can_bus`) | stored with w < 0 on 2,512 (41.7%) |
| `camera2ego` | the camera's calibration | the camera where it was when it fired: `obtain_sensor2top`'s camera → its own ego pose → global → the keyframe's LIDAR_TOP ego → lidar, as `lidar2ego @ sensor2lidar` | moves a camera by up to 0.57 m (median 0.09 m) |
| canbus 7:16 | the nearest `pose` message | `_get_can_bus_info`'s: the first message after the keyframe, in the file's order | another message on 2,916; up to 0.80 m/s² accel, 0.12 rad/s rotation rate, 0.085 m/s velocity |
| command | VAD's threshold, ego frame | each repo's rule on the lidar's 3 s pose in the keyframe's lidar frame: VAD's converter (float64), UniAD's `get_sdc_planning_label` (the ego box under the lidar, in float32 as its box methods compute) | another command on 62 (UniAD) / 65 (VAD) of the 5,119 scored |

canbus 0:3 and 16:18 hold the ego translation and the dataset's heading (wrapped to [0, 360)), as
`get_data_info` leaves them; ego2world, the images, intrinsics, `lidar2ego` and timestamps are the same for
every model. The run records the convention (`inputs: uniad` in its config), so these runs have ids of their
own and `--resume` will not continue a run made before them. Every other model is sent exactly what it was:
over all 5,119 samples the wire payloads of drivejepa, qwendrive, alpamayo15, visionpilot, openpilot (its
CAM_FRONT history), autoe2e (13-keyframe windows), meteor and constant_velocity (both orders) hash the same
before and after this change, as do their run configs (10 Nov 2026), and `tests/test_infos.py` pins it.

The two commands differ from each other on 5 val keyframes: UniAD's label sits 0.78 m below the lidar (the
ego box's bottom), which the pitch between two keyframes moves by up to 9 cm. No val keyframe is within
2.2 mm of a ±2 m threshold, against 0.33 mm between UniAD's float32 arithmetic and exact arithmetic, so the
order of float32 operations cannot change a command (measured over val; `infos.uniad_offset`), and UniAD's own
`get_sdc_planning_label`, run on every val keyframe, gives the same command on each (below).

These numbers come from the tables once: `sample_data`, `ego_pose` and `calibrated_sensor` reduced to the
val keyframes' LIDAR_TOP and camera records, and each val scene's CAN-bus `pose` file, cached beside the
index (`infos_cache`, Site setup). The camera geometry is built from the stored quaternions with
pyquaternion's own arithmetic (`infos.rotation_matrix`): the converter's camera-to-lidar transform runs
through global coordinates a couple of kilometres out, where a rotation equal to the last bit moved
`lidar2img` by 1.4e-9.

**Verified** against the repos' own code, on what the adapter's own `_results` hands its test pipeline:

* all 6,019 val keyframes, against the repos' converter functions run on each (`infos_reference.py`:
  `_get_can_bus_info`, `obtain_sensor2top`, UniAD's `get_sdc_planning_label`, VAD's converter lines,
  `get_data_info`'s lines): `lidar2img` within 1.3e-12, `can_bus` exact but for 16:18 (one ulp of the
  heading in degrees, 5.7e-14: the adapter takes it from `ego2world`), `l2g` and timestamp exact, the command
  the same on every keyframe, for UniAD and for VAD (10 Nov 2026; opt-in, `UNIAD_VAD_INFOS_REFERENCE`).
  The same comparison for the inputs every model gets, which UniAD and VAD were sent before: `lidar2img` off
  by up to 860 (a translation entry; cameras up to 0.57 m away), canbus 3:7 by up to 2.0 (w's sign),
  canbus 7:16 by up to 0.80, and 69 (UniAD) / 72 (VAD) commands, 62 / 65 of them scored;
* two val scenes through the repos' datasets themselves (`scene_reference.py`): scene-0103, and scene-0916,
  whose stored quaternion changes sign mid-scene and which turns both ways -- every keyframe within 1e-9
  of what the dataset handed the model (`tests/test_scene_golden.py`, in `make check`).

## Scene order (UniAD, VAD)

```toml
[benchmarks.nuscenes]
sequence = "scene"                                 # default "window"
# scene_models = "uniad, vad"  # the default: the models scene order may run
```

By default (`window`) every scored keyframe is its own episode: the node is reset and fed the keyframes
before it, then it (above), so a model steps up to five times per sample. In scene order each val scene is
one episode, which is how UniAD's and VAD's own test scripts run the val set: the node is reset at the
scene's first keyframe and plans from every keyframe in turn (`/infer_each`, `Policy.infer_each`), and each
keyframe is scored from the plan it made there. The scored keyframes, their ground truth, commands and
metrics are the window's, keyframe for keyframe; what changes is what stands behind each plan -- the scene
so far, with whatever state the model carries from keyframe to keyframe (UniAD: the previous BEV grid, its
tracks, the CAN-bus delta from the last keyframe; VAD: the grid and the delta). Each keyframe carries its
own command, as it does in the datasets. A resumed, limited or sample-listed run feeds a scene again from
its first keyframe, so a plan never depends on what else was skipped. 5,119 model steps in all, against
about 24,100 in windows.

The run config records `sequence: scene` (and drops `history_seconds`, which would be a scene's length),
so a scene-order run has its own id; a window run's config, and so its id, is unchanged. Preflight refuses
scene order for any model outside `scene_models`: it feeds a whole scene from one reset to a model that may
read a fixed window (Drive-JEPA, Alpamayo) or build a window-shaped history of its own (QwenDrive's dense
poses), and that has to be shown harmless before such a model is listed. It cannot be combined with
`camera_history`, which replaces the keyframes with camera sweeps.

**Verified** (`make check-uniad-vad`, `adapters/models/uniad/tests/test_scene_golden.py`):
`scene_reference.py` runs each repo's own evaluation on scene-0103 and scene-0916 -- its converter's info
file, the config's test dataset and pipeline, its dataloader, and the model called keyframe after keyframe
with its own state handling -- and paddock's harness builds the same scenes from nuScenes' tables, with each
model's own dataset inputs ("The datasets' own inputs", above), and sends them in scene order as the runner does, to `/infer_each`:
at every keyframe sent (34 of scene-0103's 40 and 35 of scene-0916's 41: scene order feeds a scene up to
its last scored keyframe) the plan is the repo's own loop's, a maximum difference of 0.000 m for UniAD and for
VAD (10 Nov 2026, RTX PRO 6000, the card shared). The unit tests (`tests/test_scene_order.py`, and
`test_a_model_with_no_temporal_state_scores_the_same_in_scene_order_as_in_windows` in
`tests/test_runner_scenes.py`) pin the scored set, the order, resumes and limits, the refusals, and that a
model without temporal state scores the same in either order.

**Not the repo's full val loop at scene starts (UniAD).** UniAD's `forward_test` records the new scene token
before it checks for a first frame, so in its own loop over the whole val set the first keyframe of every
scene but the first takes its CAN-bus position and heading deltas from the last keyframe of the scene before
it in time. Paddock resets at every scene start, as a freshly built model starts -- what the repo's loop
does for its first scene. On scene-0103, whose predecessor in time is scene-0102 (3.6 s earlier), the repo's
loop would plan 1.58 m differently at the first keyframe and 4.80 m at the second, under 0.15 m from the
sixth and under 0.03 m from the eleventh (the fixture's `after_previous_scene` variant, measured, not
asserted); scene-0916, whose predecessor in the loop is scene-0802, a log six days earlier, 0.035 / 0.011 m and
at most 0.047 m after. VAD checks for a first frame by its previous grid and has no
such carry.

**UniAD and VAD runs before the datasets' own inputs** (5730097 and earlier, among them the stored
window runs of 9 Nov) were sent the inputs every model gets. Fed those for scene-0103 (34 scored keyframes),
scene order was within 1.26 m of UniAD's own loop (median 0.27 m) and windows within 2.03 m (median 0.72 m);
for VAD 2.22 m (median 0.93 m) and 3.49 m (median 1.46 m). One input at a time, the quaternion's sign was
most of it (median 0.29 m UniAD, 0.89 m VAD; max 1.44 / 2.06 m), the cameras' own ego poses next (0.12 /
0.04 m) and the CAN-bus message least (0.05 / 0.06 m); scene-0103 has no keyframe where the commands
differ. Those runs are not comparable with runs made since, which record `inputs` in their config.

## Site setup

```toml
[datasets.nuscenes]
path    = "/path/to/nuscenes"
version = "v1.0-trainval"

[benchmarks.nuscenes]
index_cache = "/path/to/cache/nuscenes_val_index.pkl"   # optional; default <repo>/.cache/; or a directory
infos_cache = "/path/to/cache/nuscenes_val_infos.pkl"   # optional; default beside index_cache's file; or a directory
```

`infos_cache` holds the datasets' own inputs (above), built the first time a model that asks for them runs:
`sample_data` and `ego_pose` once more (about 15 s from a local disk, 5 GB of memory at its peak) and the
val scenes' CAN-bus `pose` files; 4 MB. Its key is its own (root, version, scene set, schema), so it is
never read as the index or the index as it.

The loader reads the `v1.0-trainval` JSON tables directly (no devkit): the
1.3 GB `sample_data`, `ego_pose` and `sample_annotation` tables are each loaded
once, reduced to the val keyframes and dropped; the 27 MB index is cached at
`index_cache` (default `<repo>/.cache/`). Building it takes ~25 s on a quiet disk; from a busy NAS it took about 20 minutes, once per checkout. An `index_cache` that does not end in `.pkl` is a directory several checkouts can share: the file in it is named by its dataset root, version and scene set, so two roots never evict each other, and it is written whole and renamed into place, so parallel runs never read half of one. The default path only needs keyframe images (`samples/`). Openpilot additionally
requires the CAM_FRONT sweeps described below.

## Cost

One node per run, images read from disk by the node. Measured on the shared
96 GB card: VisionPilot 0.45 s/sample, Drive-JEPA 0.5 s/sample (~45 min each
for the 5119), Alpamayo 2.5 s/sample (~3.5 h). Results:
`results/nuscenes/<model>/<run>/metrics.json`; per-sample plans and ground truth
in the run's `samples.jsonl` under the runs root.

## Openpilot camera history and GPU inference

Openpilot uses real approximately 12 Hz CAM_FRONT exposures between keyframes.
The scored anchors and their six future ground-truth points remain at 2 Hz.
History includes only exposures strictly before each anchor, followed by the
original anchor observation. Early scenes naturally have less than five seconds
of available history. The normal keyframe-only path remains the default.

Download the validation CAM_FRONT images under both `samples/CAM_FRONT` and
`sweeps/CAM_FRONT`; other sweep cameras are unnecessary for this model. The
metadata and CAN-bus files are still read from `[datasets.nuscenes]`.
Configure a dedicated Openpilot run:

```toml
[benchmarks.nuscenes]
camera_history = "CAM_FRONT"
history_seconds = 5.0
history_image_root = "/path/to/nuscenes/images"
history_index_cache = "/path/to/cache/nuscenes_front_exposures.pkl"

[models.openpilot.env]
PADDOCK_OPENPILOT_CUDA_MODEL = "/path/to/driving_cuda.pkl"
NVRTC_PATH = "/path/to/cuda_nvrtc/lib/libnvrtc.so.12"
DEV = "CUDA"
WARP_DEV = "CPU"
PADDOCK_OPENPILOT_NATIVE_LONG = "0"
PADDOCK_OPENPILOT_RESUME_S = "0"
```

Use model environment settings above: shell exports of adapter-specific variables
are not automatically inherited by model nodes. The history setting replaces
multi-camera keyframe history with CAM_FRONT history, so use it for Openpilot
runs only. Without it, the runner rejects Openpilot's 2 Hz input before loading
the model. `history_image_root` must contain both sample and sweep images.

To compile the CUDA artifact, run the following inside the pinned openpilot
checkout, using its virtual environment and an NVRTC library compatible with the
GPU (the completed run used NVRTC 12.9):

```bash
DEV=CUDA WARP_DEV=CPU NVRTC_PATH=/path/to/cuda_nvrtc/lib/libnvrtc.so.12 \
  .venv/bin/python openpilot/selfdrive/modeld/compile_modeld.py \
  --model-size 512x256 --camera-resolutions 1928x1208 \
  --onnx openpilot/selfdrive/modeld/models/driving_supercombo.onnx \
  --output /path/to/driving_cuda.pkl --frame-skip 4
```

Model inference and its queues use CUDA; camera warping remains on CPU. Omitting
`PADDOCK_OPENPILOT_CUDA_MODEL` retains the upstream default artifact. Construction
and per-sample reset select the same artifact with fresh temporal state.
Then run `paddock run --bench nuscenes --model openpilot`. The standard runner
uses one model node; the published four-worker run used site orchestration.

Completed predictions are also appended to `samples.partial.jsonl` in the run's
artifact directory, preserving evidence on interruption. This is not automatic
resume or a completed result. Scoring history settings are included in new run
configuration hashes. Camera-history caches assume the underlying dataset is
immutable; remove the cache after replacing metadata or image files.

**Provenance of that record.** `results/nuscenes/openpilot/20260908T110028-b71f4f`
was assembled by the site's four-worker orchestration, not written by
`paddock run` as tracked here: its `config.execution_workers` and
`dataset.n_expected` are keys no open-loop path in `src/paddock/` writes, it
retained 135 predictions from an earlier attempt, and it predates the
`camera_history` keys `run_config()` records now. Another site cannot
regenerate it with the tracked code. It stays `valid` because the number is not
in question, but landing the orchestration (or a `--workers` path in the
runner) is the open item that makes it regenerable; see `docs/ROADMAP.md`.

The completed 8 September 2026 run scored all 5,119 anchors in 150 scenes with
zero failures: point L2 average 4.057 m, cumulative L2 average 2.477 m, ADE at
3 seconds 3.535 m, point collision average 2.624%. See the
[support-matrix result](#support-matrix-results) and
[run record](../../results/nuscenes/openpilot/20260908T110028-b71f4f/run.json).
Its historical `repo_sha` remains null; publication does not retroactively assign
a code revision to that run. Saved-plan videos cover all 150 scenes.

**A second record, written by `paddock run`.**
`results/nuscenes/openpilot/20260929T174538-b479ee` scored the same 5,119
anchors (the same `sequences_sha256`) with the tracked runner, at the pinned
openpilot sha (`provenance.repo.at_pin`), on the ONNX backend, and with the
`camera_history` keys the record above predates. It is within 0.03 m of it on
every headline: point L2 average 4.029 m, cumulative L2 average 2.451 m, ADE at
3 seconds 3.508 m, point collision average 2.61%. The leaderboard lists both,
`b479ee` first. The README keeps the 8 September figures, the run this section
and the GPU validation were written from; `b479ee` is the one another site can
regenerate with the tracked code.

## Results (05 Sep 2026, n = 5119)

| model | L2 1 / 2 / 3 s (m) | L2 avg | L2 cumulative avg | collision % avg (point / cumulative) |
|---|---|---|---|---|
| Drive-JEPA | 0.54 / 1.59 / 3.52 | 1.88 | 0.91 | 1.46 / 1.01 |
| Alpamayo 1.5 (nav text) | 0.51 / 1.35 / 2.60 | 1.49 | 0.79 | 1.58 / 1.03 |
| VisionPilot | 1.03 / 2.21 / 3.84 | 2.36 | 1.44 | 3.82 / 3.35 |

Caveat: the first keyframe of each scene has one pose of history. Alpamayo,
which differences its history for speed, plans badly there (L2 avg 9.47 m on
those 150 keyframes vs 1.23 m with a full 2 s); Drive-JEPA and VisionPilot are
unaffected. The protocol keeps all 5119.

**The Drive-JEPA row above is camera-blind.** It was measured before the
camera-frame fix of 13 Oct 2026: the adapter placed the cameras relative to
LIDAR_TOP, which nuScenes yaws −90°, so none of the model's image samples landed in
the front image and it planned from ego status alone. The re-run with the fix
(`20261105T201953-308dc5`, 5 Nov 2026) sees the camera and scores worse: cumulative
L2 1.328 against 0.912, point L2 2.344 against 1.882. The support-matrix table
below uses the re-run.

Overlays for every scored keyframe (camera with the human future and each
model's plan, BEV with annotated boxes, per-scene videos) are on the NAS:
`<site video root>/Nuscenes/Simulation/<model>/<scene>/`. The root is site-specific and not a paddock setting; ask the site that made the run.

## Metrics

Every metric `benchmark.yaml` declares. All are lower is better except the two
`pdm_*` terms.

| metric | meaning |
|---|---|
| `l2_1s`, `l2_2s`, `l2_3s` | L2 to the human pose at exactly that horizon, m (VAD convention) |
| `l2_avg` | mean of `l2_1s`, `l2_2s`, `l2_3s`; the primary metric, m |
| `l2_uniad_1s`, `l2_uniad_2s`, `l2_uniad_3s` | L2 averaged over every 0.5 s step up to that horizon, m (UniAD convention) |
| `l2_uniad_avg` | mean of the three cumulative L2 values, m |
| `col_1s`, `col_2s`, `col_3s` | collision at exactly that horizon, fraction of samples (point) |
| `col_avg` | mean of `col_1s`, `col_2s`, `col_3s` |
| `col_uniad_1s`, `col_uniad_2s`, `col_uniad_3s` | collision averaged over every step up to that horizon (cumulative) |
| `col_uniad_avg` | mean of the three cumulative collision values |
| `ade_3s` | average displacement over all six 0.5 s steps, m |
| `pdm_ep`, `pdm_comfort` | NAVSIM v1's EP and C terms on the 0.5 s plan, higher is better ([results-schema](../results-schema.md)). With `col_avg` as NC they are this benchmark's PDM rows in the gate; there is no map, so no DAC |

## Reading a model's decisions back off a run

`ALPAMAYO_SAVE_REASONING=1` saves Alpamayo's reasoning text beside each
prediction; the tools that label each plan as a driving decision and scan a run
for behaviour gaps are kept outside this repo.

## Tests

`src/paddock/adapters/benchmarks/nuscenes/tests/test_harness.py`: quaternion/frame geometry, oriented-box
intersection, the command rule, plan resampling (including the origin anchor and
the mid-turn distance), the scorer's point vs cumulative conventions, that absent
box data omits the collision metrics rather than zeroing them, and that the index
cache rebuilds when the dataset/version/scene set changes.
`tests/test_infos.py`: each of the converters' computations (the CAN-bus message at and around the
keyframe, each camera at its own pose, the stored quaternion, both commands at a scene's end and where the
lever arm of the lidar decides them), the tables read into them, their cache, that only a model that
declares `inputs.nuscenes` gets them and its run records it, and that every other model's observations are
byte for byte what the harness built before them.

**Verified vs. what is not.** These are self-consistency tests: they pin the L2
and collision arithmetic and the frame conventions this adapter implements,
transcribed from the UniAD/VAD protocol. They do **not** establish numerical
parity against the original UniAD/VAD scorers on identical inputs -- there is no
golden fixture from those scorers here. Establishing that would need one recorded
(plan, future, boxes) case scored by the original code to diff against; until
then the equivalence claim rests on the transcription, not on a measured match.

## Watching a scene, every model's plan on it

`paddock render --bench nuscenes --sample nuscenes/val/scene-0103/12` draws the
scene that keyframe is in as a video: every scored keyframe in order, two a
second as recorded, each the front camera with every run's plan for it and the
recorded path (blue) drawn on the road. Beside it are the same lines from above,
with each run's error on that keyframe (`render_sample` in
[render.py](../../src/paddock/adapters/benchmarks/nuscenes/render.py), drawn by
`viz/moment.py`). It reads only what the runs keep, the plan and the score in
each record, and needs ffmpeg, from `PADDOCK_FFMPEG` or the PATH.

The website's nuScenes videos come from `python -m tools.site.videos nuscenes
--out DIR`, over the runs the results table shows. Which scenes is decided by
`showcase` in the same file, never by hand: the hardest left turn, right turn and
moment driving straight on (the largest error averaged over the models), then the
keyframe where the models' errors spread most (standard deviation), one per
scene. The kind is the command the models were given. A keyframe in a scene's
first 2 s is not picked: the models have less than their usual past there, and
AutoE2E's error on a scene's first keyframe averages 5.9 m against 0.7 m from the
fifth on. The table's floor, constant velocity, is drawn but not counted. Only
keyframes every run scored are considered; ties go to the harder keyframe, then
the lower id.

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Each note below starts with the mark it explains. Marks about a model
rather than this benchmark (∗ ◊ ◇ ✦ ✧) are explained in
[What these models are](../models/comparison.md#notes-on-the-models).

### Notes on the matrix cells

△ nuScenes uses 5,119 scoring anchors across 150 validation scenes. The headline
is the mean of cumulative L2 through 1, 2 and 3 seconds, not ADE at 3 seconds.
OpenPilot uses real approximately 12 Hz front-camera history; scoring stays at
2 Hz. See the nuScenes table below for point L2, cumulative L2 and ADE.

△ nuScenes val, open loop, zero-shot (none of these models was trained on
nuScenes): the average L2 over 0-3 s on the official 150-scene val split (5119
keyframes), UniAD/VAD protocol, `--bench nuscenes` -- the cumulative (per-step)
average, comparable to published tables (the `l2_avg` in `docs/leaderboard.md`
is the point value at 1/2/3 s). OpenPilot now uses real approximately 12 Hz
front-camera history while retaining the same 2 Hz scoring anchors; its completed
GPU run is detailed below. VisionPilot's plan here is its acceleration/steering
integrated to a path, as in NAVSIM, so unlike KITScenes it does yield an L2.
Alpamayo's figure is inflated by the first keyframe of each scene, where it has
no history to difference (9.5 m there vs 1.2 m elsewhere). A constant-velocity baseline scores 0.64 here (point-avg L2 1.29), **better than every learned model except Qwen-Drive 1.0 (0.450)** — nuScenes ego motion is largely straight, so open-loop L2 rewards a straight line; read these numbers with that in mind.

### nuScenes

Recorded validation results, 5,119 anchors across 150 scenes. Lower is better.
OpenPilot completed with zero failures using up to five seconds of real CAM_FRONT
history, CUDA inference, CPU camera warping and resume assistance disabled.

| Model | L2 at 1 s (m) | L2 at 2 s (m) | L2 at 3 s (m) | Point L2 avg (m) | Cumulative L2 avg (m) | ADE at 3 s (m) | Point collision avg (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenPilot | 1.925 | 4.054 | 6.194 | 4.057 | 2.477 | 3.535 | 2.624 |
| Alpamayo 1.5 | 0.509 | 1.351 | 2.602 | 1.487 | 0.787 | 1.249 | 1.582 |
| Alpamayo 2 Edge | 0.455 | 1.058 | 1.870 | 1.128 | 0.630 | 0.957 | 0.920 |
| Drive-JEPA | 0.957 | 2.158 | 3.917 | 2.344 | 1.328 | 1.996 | 1.628 |
| VisionPilot | 1.031 | 2.213 | 3.837 | 2.360 | 1.441 | 2.058 | 3.816 |
| METEOR | 0.616 | 1.683 | 3.356 | 1.885 | 0.977 | 1.570 | 6.152 ◇ |
| Qwen-Drive 1.0 | 0.318 | 0.722 | 1.360 | 0.800 | 0.450 | 0.678 | 0.770 |
| AutoE2E ⧗ | 0.553 | 1.444 | 2.697 | 1.565 | 0.839 | 1.319 | 1.862 |
| Baseline (Constant Velocity) | 0.372 | 1.151 | 2.340 | 1.288 | 0.641 | 1.069 | 1.661 |

Point L2 averages errors at 1, 2 and 3 seconds. Cumulative L2 averages the three
prefix means; ADE at 3 seconds averages all six half-second errors. Collision
is predicted ego-footprint overlap with annotated boxes, not an observed crash.

Alpamayo 2 Edge (run `20261111T231928-e4e2d6`) was built for frames 0.1 s apart and is
given the 2 Hz keyframes every row gets; its plan, 64 points 0.1 s apart, is read at its
own points at every 0.5 s. [Its page](../models/alpamayo2edge.md#nuscenes) says how its
input is built.

⧗ AutoE2E (`kitscenes-epoch-5`, run `20261023T004711-eedbe2`, valid) is fed the
6.5 s of past keyframes it declares, where the other rows get 2 s; its BEVFormer V2
starting weights were trained on nuScenes train, not val.

**Constant velocity was the lowest L2 on this recorded split until Qwen-Drive,
which is below it on every column**, collision included, and is the first model
here to beat the baseline outright rather than on one metric. That is worth
reading with the split's shape in mind: nuScenes val is 20-second urban clips
scored open-loop against what the log did, so a model that predicts the recorded
future well scores well whether or not it would drive. The same model places
fourth of five on neuroNCAP, which stages the collisions this split never
contains. nuScenes is also not in its declared training data (NAVSIM `navtrain`,
Waymo E2E, NVIDIA PhysicalAI), so unlike its NAVSIM columns this one is
zero-shot.

[OpenPilot full metrics](../../results/nuscenes/openpilot/20260908T110028-b71f4f/metrics.json).
Older model results predate the geometry and navigation corrections made when
this benchmark was added;
these tables report the saved runs, not new evaluations of the merged code.
