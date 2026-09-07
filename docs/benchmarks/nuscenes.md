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
  the horizon (UniAD style, always lower, the convention published tables
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
`index_cache` (default `<repo>/.cache/`). Building it takes ~25 s. Only the
keyframe images (`samples/`) are needed; sweeps are not.

## Cost

One node per run, images read from disk by the node. Measured on the shared
96 GB card: VisionPilot 0.45 s/sample, Drive-JEPA 0.5 s/sample (~45 min each
for the 5119), Alpamayo 2.5 s/sample (~3.5 h). Results:
`results/nuscenes/<model>/<run>/metrics.json`; per-sample plans and ground truth
in the run's `samples.jsonl` under the runs root.

## Not run

openpilot: its model needs ≥ 10 Hz consecutive frames to perceive motion and
nuScenes keyframes are 2 Hz. The 12 Hz camera sweeps would make it valid; they
ship only in the `*_blobs.tgz` trainval packages (~290 GB), not in the
`*_keyframes.tgz` copy on this box.

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
intersection, the command rule, plan resampling, and the scorer's point vs
cumulative conventions on synthetic plans.
