# nuScenes val, open loop

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `nuscenes` |
| Primary metric | `l2_avg` |
| Adapter | [`adapters/benchmarks/nuscenes/`](../../src/paddock/adapters/benchmarks/nuscenes/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [autoe2e](../models/autoe2e.md), [constant_velocity](../models/constant_velocity.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [uniad](../models/uniad.md), [uniad_b2d](../models/uniad_b2d.md), [vad](../models/vad.md), [vad_b2d](../models/vad_b2d.md), [visionpilot](../models/visionpilot.md) |

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
  lidar frame.
* **Command:** VAD's rule from the recorded future: lateral offset of the 3 s
  pose ≥ 2 m → left, ≤ −2 m → right, else straight (val: 88 / 5 / 7 %). Sent as
  the 4-way `command` and as `nav_maneuver` / `nav_distance_m`, which Alpamayo
  turns into its instruction text.
* **Metrics** (per sample, aggregated by the runner):
  `l2_{1,2,3}s`, `l2_avg` — L2 at exactly that horizon (VAD style) and their
  mean; `l2_uniad_{1,2,3}s`, `l2_uniad_avg` — mean over every 0.5 s step up to
  the horizon (UniAD style; lower than the point value whenever error grows with
  the horizon, as it usually does, the convention published tables
  use); `col_*` likewise for collision, where the ego footprint (UniAD's
  4.084 × 1.85 m, heading from the plan's direction of travel) is tested
  against every annotated box of the future keyframe in the current ego frame
  (separating-axis test); `ade_3s`. Primary metric: `l2_avg`.

## Site setup

```toml
[datasets.nuscenes]
path    = "/path/to/nuscenes"
version = "v1.0-trainval"

[benchmarks.nuscenes]
index_cache = "/path/to/cache/nuscenes_val_index.pkl"   # optional; default <repo>/.cache/; or a directory
```

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

## Streamed mode: Drive-JEPA's momentum

Off by default. Every sample goes to the node through `/infer_sequence`, which
resets it, so Drive-JEPA's momentum-aware selection -- declared on in its
`model.yaml` for every benchmark -- never has a previous plan here and every
pick is on the plain score. **No Drive-JEPA nuScenes number made before this
mode had momentum act**: the 05 Sep 2026 row predates the stage, and a run
since declares it on and records `fed: 0` in `provenance.policy.momentum_counts`.

`PADDOCK_NUSCENES_STREAMED=1` makes momentum act, the way upstream's own
sequential evaluation loop does: each keyframe is handed the plan the node chose
for the scene's previous keyframe.

* Each request also carries a `stream` (`protocol.StreamFrame`): the scene's
  name, the scored keyframe's own timestamp and its global pose (`ego2world`,
  x, y, heading). The observations are byte-for-byte those of a standard run.
* The runner plans on **one node, one sample at a time, in index order**
  (scene by scene, keyframe by keyframe), whatever the node count, so a
  keyframe's predecessor has been planned, on the same node, before it is sent.
* The node keeps its own previous plan per scene and feeds it only when it is
  0.45-0.55 s old -- Drive-JEPA's rule, unchanged. nuScenes keyframes are not
  exactly 0.5 s apart, so not every keyframe is fed. Of the 5,119 samples, 150
  open their scene (no predecessor) and 4,969 have their predecessor scored;
  4,485 of those are 0.45-0.55 s after it, 243 are 0.4 s and 241 0.6-0.7 s
  after it, and those are not fed. **Expected: 4,485 fed of 5,119.**
* No warm-up is planned (contrast KITScenes and comma2k19, whose keyframes
  are seconds apart).

What the run records: `config.streamed = "per_scene"` (absent from a standard
run, so a standard run's id and `--resume` check are unchanged, and a run cannot
be resumed across modes); a note naming the mode; per sample in
`samples.jsonl`, `extra.stream` and `extra.momentum_fed` (the node's own word,
from its `drivejepa_pick`); and in `metrics.json`, `by_group.streamed` with the
counts (`samples`, `streamed`, `momentum_fed`, `momentum_reported`), beside the
per-scene rows. A resumed run restarts the node, so the first keyframe after
each restart finds no plan and is recorded not fed; the counts are over what was
recorded, not over what was expected.

Every other model ignores the stream (the base `Policy.infer_sequence_streamed`
drops it) and plans exactly as in a standard run, one node at a time.

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

**The Drive-JEPA row is camera-blind.** It was measured before the camera-frame
fix: the adapter placed the cameras relative to LIDAR_TOP, which nuScenes yaws
−90°, so none of the model's image samples landed in the front image and it
planned from ego status alone. A re-measure with the fix replaces it.

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

**Verified vs. what is not.** These are self-consistency tests: they pin the L2
and collision arithmetic and the frame conventions this adapter implements,
transcribed from the UniAD/VAD protocol. They do **not** establish numerical
parity against the original UniAD/VAD scorers on identical inputs -- there is no
golden fixture from those scorers here. Establishing that would need one recorded
(plan, future, boxes) case scored by the original code to diff against; until
then the equivalence claim rests on the transcription, not on a measured match.

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

⊘ Measured camera-blind: until the camera-frame fix of 13 Oct 2026 Drive-JEPA's adapter
placed the cameras so that no sample point landed in the image on nuScenes and
neuroNCAP, and the plan came from ego status alone. Re-measure pending for
nuScenes; its other columns are not affected.

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
| Drive-JEPA ⊘ | 0.540 | 1.587 | 3.519 | 1.882 | 0.912 | 1.536 | 1.459 |
| VisionPilot | 1.031 | 2.213 | 3.837 | 2.360 | 1.441 | 2.058 | 3.816 |
| METEOR | 0.616 | 1.683 | 3.356 | 1.885 | 0.977 | 1.570 | 6.152 ◇ |
| Qwen-Drive 1.0 | 0.318 | 0.722 | 1.360 | 0.800 | 0.450 | 0.678 | 0.770 |
| AutoE2E ⧗ | 0.553 | 1.444 | 2.697 | 1.565 | 0.839 | 1.319 | 1.862 |
| Baseline (Constant Velocity) | 0.372 | 1.151 | 2.340 | 1.288 | 0.641 | 1.069 | 1.661 |

Point L2 averages errors at 1, 2 and 3 seconds. Cumulative L2 averages the three
prefix means; ADE at 3 seconds averages all six half-second errors. Collision
is predicted ego-footprint overlap with annotated boxes, not an observed crash.

⧗ AutoE2E (`kitscenes-epoch-5`, run `20261023T004711-eedbe2`, valid) is fed the
6.5 s of past keyframes it declares, where the other rows get 2 s; its BEVFormer V2
starting weights were trained on nuScenes train, not val.

⊘ Measured camera-blind; re-measure pending, see the note above.

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
