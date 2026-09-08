# nuScenes val, open loop

`paddock run --bench nuscenes --model <drivejepa|visionpilot|alpamayo15>`

The nuScenes planning protocol that UniAD, VAD and their successors report: on
every val keyframe, plan 3 s ahead from the real past and compare with what the
human drove. Zero-shot for every model here; none was trained on nuScenes.

## What is scored

* **Split:** the official val split (150 scenes, vendored from the devkit's
  `create_splits_scenes` into `val_scenes.json`), 6019 keyframes at 2 Hz. The
  5119 with six future keyframes (a full 3 s) are scored; the last six of each
  scene are dropped, where the literature pads with masks.
* **Sample:** up to four preceding keyframes plus the scored one (2 s of real
  history), so Drive-JEPA gets its four frames 0.5 s apart and Alpamayo a pose
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
index_cache = "/path/to/cache/nuscenes_val_index.pkl"   # optional; default <repo>/.cache/
```

The loader reads the `v1.0-trainval` JSON tables directly (no devkit): the
1.3 GB `sample_data`, `ego_pose` and `sample_annotation` tables are each loaded
once, reduced to the val keyframes and dropped; the 27 MB index is cached at
`index_cache` (default `<repo>/.cache/`). Building it takes ~25 s. The default path only needs keyframe images (`samples/`). Openpilot additionally
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

The completed 8 September 2026 run scored all 5,119 anchors in 150 scenes with
zero failures: point L2 average 4.057 m, cumulative L2 average 2.477 m, ADE at
3 seconds 3.535 m, point collision average 2.624%. See the
[README result](../../README.md#nuscenes--openpilot-gpu-validation) and
[run record](../../results/nuscenes/openpilot/20260908T110028-b71f4f/run.json).
Its historical `repo_sha` remains null; publication does not retroactively assign
a code revision to that run. Saved-plan videos cover all 150 scenes.

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

Overlays for every scored keyframe (camera with the human future and each
model's plan, BEV with annotated boxes, per-scene videos) are on the NAS:
`<site video root>/Nuscenes/Simulation/<model>/<scene>/` (this site's root is in the top-level repo's `docs/RUNNING_BENCH2DRIVE.md`).

## Metrics

Every metric `benchmark.yaml` declares, with its direction (all lower is better
except `inference_seconds`, which is a cost).

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
| `inference_seconds` | wall time per sample, s |

## Tests

`tests/test_nuscenes_openloop.py`: quaternion/frame geometry, oriented-box
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

## Framework changes in this PR

Adding this benchmark stayed inside its own adapter directory, with two
deliberate framework touchpoints kept minimal and called out here per
`CONTRIBUTING.md`:

* `registry.py` gained a `builtin` runtime kind (paddock's own interpreter), so a
  dependency-free baseline (`constant_velocity`) can run without a separate venv.
* `results/report.py` learned that `l2*`/`col*` are lower-is-better, so these
  metrics render with the correct arrow. This is global metric-direction
  inference; a per-metric declaration on the benchmark would be the cleaner home
  and is worth a follow-up.

Kept OUT of this PR on purpose, as separate framework follow-ups rather than more
core changes folded into an adapter:

* **Run provenance (`repo_sha`).** `run.json` carries a `repo_sha` field that the
  runner never populates, so runs do not record the code version that produced
  them. Fixing it touches `runner.py` (framework) and should be its own change.
  Until then, treat any committed result as predating later code changes —
  e.g. the results here were produced before the geometry and nav-distance fixes
  and should be re-run before they are quoted alongside the corrected code.
* **Numerical-parity fixture.** Establishing parity with the original UniAD/VAD
  scorers needs a recorded `(plan, future, boxes)` case scored by that code to
  diff against; none is vendored here (see the Tests section).
