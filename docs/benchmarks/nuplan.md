# nuPlan

Open-loop trajectory accuracy — ADE and FDE against the recorded future — on
nuPlan v1.1's original logs, over a balanced set of tagged scenario moments.

```bash
paddock run --bench nuplan --model openpilot --limit 5   # smoke
paddock run --bench nuplan --model openpilot             # full set
```

## Why nuPlan when NAVSIM already uses nuPlan

They are the same source logs and not the same data. NAVSIM scores on OpenScene,
a **2 Hz** redistribution; nuPlan's own release keeps the native **10 Hz
cameras** and **100 Hz ego poses**.

That gap decides which models can be scored at all. A model that infers ego
motion from frame-to-frame change reads a 2 Hz stream as a nearly parked car, so
`openpilot` declares `min_input_hz: 10.0` and the NAVSIM harness refuses the
pairing rather than recording the artefact. nuPlan sits exactly on that bound and
the check is `>`, so every model paddock currently runs qualifies here.

The 100 Hz pose stream is the second reason: ground truth is sampled from a
record twenty times denser than the 0.5 s spacing it is measured at, so the
target is interpolation-free.

## Data

Two separate downloads, and having the first without the second is the normal
half-way state:

| part | size (mini) | needed for |
|---|---:|---|
| `nuplan-v1.1/splits/<split>/*.db` | 13 GB | scenario selection, ground truth, calibration |
| `nuplan-v1.1/sensor_blobs/` | ~450 GB | the camera images every model actually reads |
| `maps/` | 1.4 GB | not read by this benchmark; required by the devkit |

```toml
[datasets.nuplan]
path = "/path/to/nuplan"
```

`paddock doctor` reports an empty `sensor_blobs` as its own line, because the
logs alone will happily build every sample and then fail on the first JPEG an
hour into a sweep.

Two directory layouts are in the wild — nuPlan documents
`nuplan-v1.1/splits/<split>`, while the mini archive on S3 unpacks to
`nuplan-v1.1/splits/data/cache/<split>`. Both are searched, so neither site has
to reshape its copy.

Set the split (default `mini`) with:

```toml
[benchmarks.nuplan]
split = "mini"
```

## The scored set is in the adapter, not in site config

`--limit` aside, both sites must score the same moments, so `SCENARIOS_PER_TYPE`
and `MIN_SEPARATION_S` are constants in `harness.py` rather than settings. Three
rules build the set, and each exists because of a way the count would otherwise
overstate what is measured:

1. **Cap per type** — `stationary` alone is a third of nuPlan mini's ~460 000
   tagged keyframes. Eight per type keeps the set balanced.
2. **Thin to 20 s apart** — tags sit on 20 Hz keyframes and a tagged situation
   lasts seconds, so consecutive anchors are the same moment 50 ms apart.
   Unthinned, the eight `stationary` samples were one parked car scored eight
   times.
3. **One sample per moment, under its rarest label** — nuPlan's tags overlap
   heavily. A traffic-light intersection is simultaneously `on_intersection`,
   `traversing_intersection`, `on_traffic_light_intersection` and more, all on
   one keyframe; scoring it five times makes the headline ADE a measure of
   intersection driving. The label kept is the rarest type true of that moment,
   which also spends the duplicates on the types that have fewest anchors.

Anchors are filled round-robin across logs, so a type present in twelve logs is
measured in twelve places rather than in one stretch of one drive.

A sample is dropped, not padded, when the log cannot support it: inside the
first 4.8 s there is no history to feed, and within 4 s of the end there is no
future to score. Padding the future would read as the car stopping and would
score a correct prediction as wrong.

On this site's 56-log mini copy that yields **360 anchors → 335 scored samples**,
across 53 logs and 57 scenario types. Eight types never occur without a rarer
co-tag and so contribute no sample of their own; their moments are still scored,
under the rarer label.

## Samples are sequences

Each sample is 48 preceding frames plus the anchor — 4.8 s at 10 Hz, the same
window KITScenes uses and for the same reason: it is what fills openpilot's
24-deep feature buffer at the 20 Hz it believes it is running at. Observations
carry image *paths*, not pixels; the node reads the same filesystem.

## Calibration is read without the devkit

nuPlan stores camera intrinsics and extrinsics as **pickled devkit objects**
(`nuplan.database.common.data_types.CameraIntrinsic` and friends), so a plain
`pickle.loads` raises `ModuleNotFoundError` on any machine without nuplan-devkit
— which is every machine paddock runs on, deliberately.

`loader._CalibrationUnpickler` maps those four container types onto `list` and
refuses everything else, so the numbers come out with no devkit, no torch and no
hydra in paddock's interpreter. It refuses rather than allows by default because
the blob comes out of a downloaded file, and unpickling is arbitrary code
execution by design.

This is worth a test of its own — and has one. If it silently stopped working,
every sample would lose its intrinsics and openpilot would warp to the wrong
focal length without anything failing.

## Reference: constant velocity

Computed from the logs alone (a constant-velocity plan needs only ego speed, so
it needs no camera images), over the full n=335:

| metric | mean | median |
|---|---:|---:|
| ade | 1.2538 | 0.8461 |
| fde@1s | 0.3880 | 0.2834 |
| fde@2s | 1.3502 | 0.8850 |
| fde@3s | 2.8324 | 1.8800 |
| longitudinal_mae | 1.0864 | 0.6969 |
| lateral_mae | 0.3157 | 0.0447 |

This is the floor a model has to beat, and the breakdown says where: the
baseline is near-perfect on the stationary types (`stationary_in_traffic` 0.041)
and worst exactly where speed changes (`accelerating_at_traffic_light` 3.325,
`traversing_intersection` 3.254). A model that cannot beat 1.2538 has not
learned anything a straight line does not already say.

Not a `paddock run` and not in `results/`: no model was involved, and it is
recorded here as a reference rather than as a leaderboard row.

## Camera shards and partial coverage

The nine camera zips each unpack into a directory of their own
(`sensor_blobs/nuplan-v1.1_mini_camera_8/<log>/CAM_F0/…`), while the database
only knows `<log>/CAM_F0/<token>.jpg`. Both the sharded and the flattened layout
are detected, so no site has to reshape its copy.

Holding one shard of nine is normal, not an error. Anchors whose log has no
images are **skipped**, and the run is recorded `valid: false` with the coverage
in `notes`, so it is kept and inspectable but never averaged against a
full-coverage run. `--dry-run` reports the split before anything starts.

## First result

One camera shard (8 of 9) covers 8 logs, giving **26 of 375 anchors**; 25 of
those have both the history and the future the scored moment needs.

| n=25 | visionpilot | constant velocity |
|---|---:|---:|
| ade | 3.1924 | **1.3926** |
| fde@1s | 1.7155 | 0.4624 |
| fde@2s | 3.2339 | 1.4327 |
| fde@3s | 5.8514 | 3.1994 |
| lateral_mae | 1.3283 | 0.6300 |

Both columns are over the same 25 samples. **VisionPilot is beaten by a straight
line, by more than a factor of two on every metric.** Three things to hold in
mind before reading that as a verdict on the model: n=25 is small and comes from
one shard rather than a spread of the dataset; VisionPilot's recorded strength
is closed-loop NCAP rather than open-loop imitation; and nuPlan's urban
intersection content is the regime where a constant-velocity baseline is hardest
to beat and VisionPilot is furthest from its training distribution. The number
is real, but it is a starting point, not a ranking.

Roughly 3 minutes wall-clock for the 25 samples on CPU, including model load.

## Status

**Verified**: the full path runs end to end — sample construction, selection,
ground truth, calibration decoding, camera-path resolution, inference and
scoring — against the real download, and the numbers above came out of it. The
golden tests build a synthetic log and run with no dataset, no devkit and no GPU.

**Not yet done**: only visionpilot has been scored, on 7% of the scored set.
openpilot is the interesting one here — it is the model nuPlan's 10 Hz cameras
exist to serve — and it has not been run because its repo is not cloned at this
site. A full-coverage run needs the other eight camera shards (~405 GB).
