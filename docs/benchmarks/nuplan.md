# nuPlan

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `nuplan` |
| Primary metric | `ade` |
| Adapter | [`adapters/benchmarks/nuplan/`](../../src/paddock/adapters/benchmarks/nuplan/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [constant_velocity](../models/constant_velocity.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [uniad](../models/uniad.md), [uniad_b2d](../models/uniad_b2d.md), [vad](../models/vad.md), [vad_b2d](../models/vad_b2d.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

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

`--limit` aside, every site must score the same moments, so `SCENARIOS_PER_TYPE`
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

On this site's 64-log mini copy that yields **375 anchors across 59 scenario
types and 61 logs**, of which **351 are scorable** — the rest are anchors the
drop rule above removes. `coverage()` counts the scorable ones, not the
manifest, so `--dry-run` promises what a sweep will actually deliver.

Six types never occur without a rarer co-tag and so contribute no sample of
their own; their moments are still scored, under the rarer label.

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

## Results

Five models over the same 25 samples, `valid: false` at one camera shard. Every
cell is the mean from one committed record under `results/nuplan/`, and each
model has exactly one — no column mixes two runs.

| n=25 | openpilot | meteor | **constant velocity** | drivejepa | visionpilot |
|---|---:|---:|---:|---:|---:|
| ade | **0.807** | 1.273 | 1.393 | 1.512 | 2.246 |
| fde@1s | **0.416** | 0.483 | 0.462 | 0.568 | 1.488 |
| fde@2s | **0.878** | 1.333 | 1.433 | 1.593 | 2.246 |
| fde@3s | **1.539** | 2.803 | 3.199 | 3.305 | 3.589 |
| longitudinal_mae | **0.658** | 0.831 | 1.036 | 1.429 | 1.491 |
| lateral_mae | 0.316 | 0.778 | 0.630 | **0.299** | 1.324 |

**Read these as a 25-sample subset, not as nuPlan.** They cover **7 of the 59
scenario types**, six of them stop-sign or traffic-light decelerations, because
one camera shard decides which logs are servable and the balancing rules act on
the 375-anchor manifest rather than on what a shard happens to hold. A ranking
over near-homogeneous deceleration moments is not a ranking on nuPlan. The
three sampling rules do their work; none of these numbers inherits it.

`constant_velocity` is a stored run like any other — declared in its
`model.yaml` and scored through the same harness — so the floor and the rows it
anchors are one artifact class and both regenerate from `paddock run`.

Two caveats that travel with individual rows:

- **openpilot** was run against a v0.11.1 checkout, not the `053d9c446` that
  `repos.yaml` pins, so its `run.json` carries `repo_sha: null`. The number is
  real and the adapter's arity probe is a no-op at the pin, but it is not yet
  attributable to a revision. Bumping the pin belongs in its own PR.
- **alpamayo15** has no row here. It needs `nvidia/Cosmos-Reason2-8B`, which is
  gated; only its config and tokenizer are read, and a substitute was used for
  an earlier measurement that is not committed.

## Camera shards and partial coverage

The nine camera zips each unpack into a directory of their own
(`sensor_blobs/nuplan-v1.1_mini_camera_8/<log>/CAM_F0/…`), while the database
only knows `<log>/CAM_F0/<token>.jpg`. Both the sharded and the flattened layout
are detected, so no site has to reshape its copy.

Holding one shard of nine is normal, not an error. Anchors whose log has no
images are **skipped**, and the run is recorded `valid: false` with the coverage
in `notes`, so it is kept and inspectable but never averaged against a
full-coverage run. `--dry-run` reports the split before anything starts.

## Reading VisionPilot's row

One camera shard covers 8 logs, giving **25 scorable anchors** — the arithmetic
behind the `valid: false` on every run above.

VisionPilot is beaten by a straight line on every metric — by 1.6× on ADE and
by 3.2× at the 1 s horizon — and that is worth three qualifications before it
reads as a verdict on the model: the subset is small and skewed toward
decelerations; VisionPilot's recorded strength is closed-loop NCAP rather than
open-loop imitation; and it has no goal or command interface, so at a junction
it goes straight — a limitation already visible on nuScenes and Bench2Drive.

Its row moved between the first measurement and this one, from ADE 3.192 to
2.246, and not because of anything in this adapter: merging `main` brought
`VISIONPILOT_SETPOINT`, which latches the ACC set-speed at the scored
frame instead of at the first history frame. The earlier run is deleted rather
than kept beside the new one, so `results/nuplan/` holds one record per model
and no reader has to know a commit message to tell which counts.

Roughly 3 minutes wall-clock for the 25 samples on CPU, including model load;
meteor does the same set in 27 s on GPU.

## Status

**Verified**: the full path runs end to end — sample construction, selection,
ground truth, calibration decoding, camera-path resolution, inference and
scoring — and five models are scored on one shared sample set. The golden tests
build a synthetic log and run with no dataset, no devkit and no GPU. The
adapter has been exercised at a second site, on a different nuPlan copy and
split, with no devkit present.

**Not yet done**: `Observation.speed_limit` is not populated, so an IDM-style
policy falls back to a latched cruise speed and drives away from a stopped car
— which inflates every model's error on the near-stationary samples that
dominate this subset. Reading `speed_limit_mps` from nuPlan's maps is the fix;
it is out of scope here because it adds a map dependency to a benchmark that
currently reads none. Coverage is one camera shard of nine; the rest are
~405 GB.

## Support-matrix results

nuPlan has no column in the README's support matrix yet: these runs cover 25 of the
351 anchors, and the column comes back with full runs. The per-metric results and
their notes stay here as the record. Each note below starts with the mark it explains. Marks about a model
rather than this benchmark (∗ ◊ ◇ ✦ ✧) are explained in
[What these models are](../models/comparison.md#notes-on-the-models).

### Notes on these runs

◐ **Every nuPlan figure is `valid: false` and excluded from `paddock report`.**
This site holds one of nuPlan's nine camera shards, so of the 351 scorable
anchors in the balanced manifest only **25** have images — and those 25 fall in
**7 of the 59 scenario types**, six of them stop-sign or traffic-light
decelerations. The runs are recorded and comparable *with each other* (one
shared sample set, one record per model, `sequences_sha256` pinned), but a
ranking over near-homogeneous deceleration moments is not a ranking on nuPlan,
and these are not comparable with a full-coverage run. The remaining shards are
~405 GB. OpenPilot's figure additionally comes from a v0.11.1 checkout rather
than the sha `repos.yaml` pins, so its record carries `repo_sha: null`.
See [docs/benchmarks/nuplan.md](nuplan.md).

◐◐ Alpamayo 1.5 has no nuPlan figure: it needs `nvidia/Cosmos-Reason2-8B`,
which is gated. Only its config and tokenizer are read, and the one measurement
made with a substitute is deliberately not committed, so there is no record to
cite.

### nuPlan

Open-loop displacement error against the recorded future, on nuPlan v1.1's
original logs at their native **10 Hz** cameras and 100 Hz ego poses — not
NAVSIM's 2 Hz OpenScene redistribution of the same source. That rate is what
makes the dataset scorable for a model like OpenPilot, which infers ego motion
from frame-to-frame change and declares `min_input_hz: 10.0`.

**These are a 25-sample subset, not nuPlan** (see ◐). One camera shard of nine
is present, so 25 of 351 scorable anchors have images, and they fall in 7 of 59
scenario types. Every run below is `valid: false` and excluded from the
leaderboard; they are comparable with each other and with nothing else.

| Model | ADE ↓ (m) | FDE@1s ↓ (m) | FDE@2s ↓ (m) | FDE@3s ↓ (m) | Longitudinal MAE ↓ (m) | Lateral MAE ↓ (m) |
|---|---:|---:|---:|---:|---:|---:|
| OpenPilot | **0.807** | **0.416** | **0.878** | **1.539** | **0.658** | 0.316 |
| Alpamayo 1.5 | — ◐◐ | — | — | — | — | — |
| Drive-JEPA | 1.512 | 0.568 | 1.593 | 3.305 | 1.429 | **0.299** |
| VisionPilot | 2.246 | 1.488 | 2.246 | 3.589 | 1.491 | 1.324 |
| METEOR | 1.273 | 0.483 | 1.333 | 2.803 | 0.831 | 0.778 |
| Baseline (Constant Velocity) | 1.393 | 0.462 | 1.433 | 3.199 | 1.036 | 0.630 |

Two of the four models clear the constant-velocity floor: OpenPilot, which
leads every metric but lateral error, and METEOR. Drive-JEPA and VisionPilot
do not — expected on a subset this skewed toward decelerations, where holding
the current speed is already close to right. `constant_velocity` is a stored
run like any other, declared in its `model.yaml` and scored through the same
harness, so the floor and the rows it anchors regenerate together.

Protocol, the three sampling rules that build the balanced set, and why the
calibration is decoded without nuplan-devkit are in
[docs/benchmarks/nuplan.md](nuplan.md).
