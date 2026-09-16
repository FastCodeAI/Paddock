# comma2k19

Open-loop trajectory accuracy against comma2k19's own recorded
(INS/GNSS/vision-fused) future pose. 33 hours of 20 Hz highway driving on
California's I-280, 2035 one-minute segments, two vehicles.

This page is the **protocol** — what paddock asks and how it scores the answer,
all of it paddock's choices. What the dataset *is* — its sensors and their
limits, its coordinate frames, the two errors in its own documentation, and the
derivation of the 10,165 scored samples every row below is conditioned on — is
[docs/datasets/comma2k19.md](../datasets/comma2k19.md).

```bash
paddock run --bench comma2k19 --model openpilot --limit 5      # smoke
paddock run --bench comma2k19 --model openpilot                # full, n=10165
paddock run --bench comma2k19 --model openpilot --resume <id>  # continue one in place
paddock render --bench comma2k19                               # comparison figures
```

## One sample, end to end

A **sample** is one *keyframe*: one moment in one segment at which the model is
asked to plan, and the recording says what the car did next.

| | count | span | spacing |
|---|---|---|---|
| **input** | 160 camera frames, the last of them *at* the keyframe | 7.95 s of past | 20 Hz — every recorded frame |
| **output** | 33 plan points, resampled to **20 waypoints** | 0.5 s → 10 s ahead | 0.5 s |
| **ground truth** | 200 recorded frames, read as **20 poses** | 0.5 s → 10 s ahead | 0.5 s |
| **headline** | first **6** waypoints vs the same 6 poses | 3 s ahead | 0.5 s |

So: *the model sees 160 frames of the past 7.95 s and predicts 20 waypoints out
to 10 s; `ade` is its mean error over the first 3 s, and `ade@h`/`fde@h` repeat
that at 1, 2, 3, 4, 5, 6, 8 and 10 s.*

The three counts are independent of each other and each is a separate setting:

- **160 input frames** is `history_frames`, fed one at a time through
  `Policy.infer_sequence` — real preceding frames, not a repeated still.
  supercombo's feature buffer is 96 frames deep (128 for `big`), so 160 both
  fills it and leaves the model warm before the moment that counts. At
  `input_hz = 20` that is 159 gaps × 0.05 s = **7.95 s**; lowering the rate
  keeps the 160 observations and stretches the span (see *Evaluation rate*).
- **20 output waypoints** is the harness's grid, `np.arange(1, 21) * 0.5`.
  supercombo does not predict on it: it predicts 33 points on its own
  `ModelConstants.T_IDXS`, which run 0 → 10 s and are packed tightly near the
  car. The adapter resamples those onto the 0.5 s grid, so every model is
  scored on the same clock whatever its native one is.
- **20 ground-truth poses** is `interval_s = 0.5` × `gt_horizon_s = 10`, taken
  from the segment's own fused INS/GNSS/vision pose — the next **200 recorded
  frames**, subsampled by 10.

Scoring is displacement in the ego frame, waypoint against pose. `ade` is the
mean over the first 6 pairs (`eval_horizon_s = 3.0`); `ade@h` is the mean up to
`h`, `fde@h` the error at `h` alone. A horizon the *pair* does not reach is
**`NaN`**, never a shorter horizon wearing a longer label — `scored_horizon_s`
records how far it did reach, and is 10.0 for openpilot.

**Segments are not samples.** The dataset has 2035 one-minute segments and the
default is `keyframes_per_segment = 5`, so a full sweep is **10,165 scored
samples**: 2033 segments × 5, plus two segments that yield none. Those two hold
222 and 194 frames, and by the table above a keyframe needs 160 frames behind it
and 200 ahead — 360 in all — so neither can place even one. The numbers reported
here are always samples; `n` in a run record is samples too.

## Setup

```toml
[datasets.comma2k19]
# raw_data/, not the sibling data/ (HF demo parquet) or compression_challenge/
path = "/path/to/comma2k19/raw_data"

[benchmarks.comma2k19]
# Optional. Defaults shown; all three change what is measured and are hashed
# into the run id.
history_frames = 160          # observations fed before the scored one
keyframes_per_segment = 5
input_hz = 20                 # observation rate; see "Evaluation rate" below

# These three change only how fast the sweep runs, never what it measures, so
# none of them is hashed into the run id. See "Making a sweep finish" below.
nodes = 4                     # policy nodes scored in parallel
frame_cache = "/dev/shm/paddock-comma2k19"   # where decoded frames are staged
keep_frames = false           # keep them for the next model instead of releasing
```

Frame decoding needs either **PyAV** in paddock's environment
(`pip install av`) or an ffmpeg binary via `PADDOCK_FFMPEG`. `preflight`
reports which, and refuses to run without one.

Getting the dataset: see [data setup](../data-setup.md#comma2k19). It is a
Harvard Dataverse download, `Chunk_1.zip` … `Chunk_10.zip`, ~100 GB unpacked.

### The toolkit is required, not optional

comma ships a toolkit with the dataset, and every rotation under a comma2k19
score comes from it. Clone it into `[roots] repos` at the sha pinned in
`repos.yaml`:

```bash
paddock config                      # prints the resolved [roots] repos
git clone https://github.com/commaai/comma2k19.git <repos>/comma2k19
```

The directory name must be `comma2k19`; that is the key in `repos.yaml` and how
paddock finds it.

It is not pip-installable (no `setup.py`), so paddock treats it the way it
treats every model repo: a pinned clone, put on `sys.path` for the length of one
import by [`toolkit.py`](../../src/paddock/adapters/benchmarks/comma2k19/toolkit.py)
and taken off again. Only the two numpy-only modules are imported, so it adds no
dependency to paddock's environment. `preflight` says so before anything else if
the clone is missing.

An earlier version of the adapter inlined the rotations and the EON intrinsics
and argued for it. That argument is reversed: a copied matrix is a second source
of truth with nothing watching it, on the toolkit's own dataset.

**The cost, stated plainly.** CI runs with no site `paddock.toml` and therefore
no clone, so the tests that exercise this maths **skip on a pull request** and
are run by the reviewer at a real site — the arrangement in
[reviewing](../reviewing.md) §1. That is a real loss: `camera2ego = np.eye(4)`,
the defect described below, is exactly what those tests exist to catch. Two
things partly compensate, and both do run in CI: the toolkit is pinned (checked
by a test), and `test_the_toolkit_rotations_match_an_independent_reference`
checks the toolkit against algebra written out again in the test file rather
than checking paddock against itself.

## What is scored

**Five keyframes per segment**, spread evenly across the span that has both a
full history behind it and 10 s of future ahead of it, with the real preceding
frames fed through `Policy.infer_sequence` and scored against **10 s** of
recorded future on a 0.5 s grid. 10 s is `ModelConstants.T_IDXS[-1]`, the horizon
supercombo plans over; comma2k19 is comma's dataset for comma's model and its
60 s segments put the recording nowhere near the binding constraint. Five rather
than one because the interesting splits — lead present or not, per vehicle, per
sanity gate — need enough samples of the *rarer* case to say anything, and the
decode pass over a segment is the same pass either way.

The headline is `ade` and `fde@3s` over a 3 s horizon, matching every other
open-loop benchmark here.

### Trajectory

| metric | what it is | direction |
|---|---|---|
| `ade` | mean displacement over the 3 s headline horizon, metres | lower |
| `fde@1s`, `fde@2s`, `fde@3s` | displacement at that horizon, metres | lower |
| `ade@1s`, `ade@2s`, `ade@3s`, `ade@4s`, `ade@5s`, `ade@6s`, `ade@8s`, `ade@10s` | mean displacement up to that horizon | lower |
| `fde@4s`, `fde@5s`, `fde@6s`, `fde@8s`, `fde@10s` | displacement at that horizon | lower |
| `scored_horizon_s` | how far the prediction and the recording actually overlapped, seconds | higher |
| `longitudinal_mae`, `lateral_mae` | axis-aligned error split, metres | lower |
| `along_track_mae`, `cross_track_mae` | the same split against the ground-truth *path tangent* | lower |
| `along_track_bias` | signed along-track error; positive means the plan ran ahead | toward 0 |
| `lateral_at_arclength_mae` | lateral error with speed error removed, metres | lower |
| `predicted_path_length`, `recorded_path_length` | distance covered over the horizon, metres | n/a |
| `path_length_ratio` | predicted / recorded distance — the progress component | toward 1 |

`ade@3s` is not a duplicate of `ade`: the headline averages over whatever the
plan reached, `ade@3s` is `NaN` unless the plan reached 3 s.

### Auxiliary heads

comma2k19 ships the car's own CAN speed, its radar returns and a fused
INS/GNSS/vision pose, which is enough to check most of what a supercombo-class
model predicts *besides* the path. Everything in this group is read from
`Trajectory.extras` ([the protocol's optional schema](../architecture.md)) and is
`NaN` for a model that publishes none — one declared metric list, whatever heads
the model happens to have. Scored by
[`auxiliary.py`](../../src/paddock/adapters/benchmarks/comma2k19/auxiliary.py).

| metric | what it is | direction |
|---|---|---|
| `speed_abs_err`, `speed_rel_err` | model ego speed vs CAN (pose where a segment has no CAN); relative error floors the denominator at 1 m/s | lower |
| `yaw_rate_abs_err` | model yaw rate vs the differenced recorded orientation, rad/s | lower |
| `yaw_rate_bias` | the same, signed — a mirrored convention shows here and nowhere else | toward 0 |
| `lead_prob` | the model's own confidence in its nearest lead | n/a |
| `lead_precision`, `lead_recall` | agreement with the closest in-lane radar track inside 60 m; `NaN` where the question does not arise, so the mean over a run *is* precision and recall | higher |
| `lead_range_err` | signed range error vs radar, metres; positive means the model places the lead too far | toward 0 |
| `lead_range_abs_err`, `lead_lateral_abs_err`, `lead_rel_speed_abs_err` | unsigned lead errors, metres and m/s | lower |
| `within_1sigma_x`, `within_1sigma_y`, `within_2sigma_x`, `within_2sigma_y` | fraction of waypoints whose realised error falls inside the model's own predicted sigma; a calibrated model reads ~0.68 and ~0.95 | toward 0.68 / 0.95 |
| `lane_width_m`, `lane_offset_m`, `lane_prob`, `road_width_m`, `camera_height_m` | the predicted geometry abeam the car. comma2k19 has **no lane labels**, so these are descriptive, not errors | n/a |
| `gate_speed`, `gate_lane_width`, `gate_camera_height` | 0/1 plausibility gates on the above (speed within 5 %, lane 2.5–4.5 m, camera 1.0–1.5 m); geometry gates are `NaN` below 15 mph, where the road frame is not observable | higher |
| `sanity_pass` | all applicable gates passed | higher |

A failed gate means the warp or the calibration is wrong and the trajectory
numbers for that sample mean nothing — read `sanity_pass` before reading `ade`.

### Safety and comfort

These need no extras: both are read off the *planned* trajectory, so they are
reported for every model including the constant-velocity baseline. Comfort comes
from the shared [`paddock.metrics.comfort`](../../src/paddock/metrics/comfort.py),
so a comfort number here and one on another open-loop benchmark are the same
statement; the bounds are nuPlan's `ego_is_comfortable` thresholds as NAVSIM's
PDM score uses them, cited there at the strength of published constants.

| metric | what it is | direction |
|---|---|---|
| `lead_gap_min_m` | smallest gap to the radar lead if the plan were driven and the lead held its measured closing rate, metres. `NaN` on an empty road — scoring that as a large safe gap would reward driving on deserted roads | higher |
| `lead_ttc_min_s` | when that gap first reaches zero, seconds; `NaN` when it never does | higher |
| `ttc_below_2s` | rate of plans that would have closed the gap within 2 s | lower |
| `comfort_lon_accel_max`, `comfort_lon_decel_max`, `comfort_lat_accel_max`, `comfort_lon_jerk_max`, `comfort_yaw_rate_max` | the plan's kinematic extremes, all as positive magnitudes | lower |
| `comfort_lon_accel_within_bounds`, `comfort_lon_decel_within_bounds`, `comfort_lat_accel_within_bounds`, `comfort_lon_jerk_within_bounds`, `comfort_yaw_rate_within_bounds` | fraction of plans inside each bound (2.40, 4.05, 4.89 m/s², 4.13 m/s³, 0.95 rad/s) | higher |
| `comfort_within_bounds` | fraction inside *all five* | higher |

### Miss rate, and a driving score that is sourced rather than invented

There is deliberately **no composite driving score of paddock's own**. A single
number here would have to weight an ADE against a comfort rate against a TTC, and
no such weighting is published for this dataset; inventing one would make the
benchmark's headline an opinion of paddock's rather than a measurement. The
components are reported separately and combine however a reader's application
requires, and `primary_metric` is still `ade`.

That argument forbids paddock *choosing* the weights. It does not forbid
evaluating **someone else's published formula**, which is a much weaker claim:
"here is what NAVSIM's scoring function returns when you feed it log replay". So
[`score.py`](../../src/paddock/adapters/benchmarks/comma2k19/score.py) computes
NAVSIM v1's PDMS shape with NAVSIM's weights unchanged —
`NC × DAC × (5·TTC + 5·EP + 2·C) / 12` — and prints all five sub-scores beside
it, so nobody has to trust the sum and anyone who disagrees with the weights can
reweigh them from the per-sample records. Nothing new is measured: every term is
derived from a column already in the tables above.

| metric | what it is | direction |
|---|---|---|
| `miss_rate@2s`, `miss_rate@3s`, `miss_rate@4s` | fraction of samples whose FDE at that horizon exceeded 2 m — the prediction-benchmark convention (Argoverse MR@2m). `NaN`, not a miss, where the plan never reached the horizon | lower |
| `pdm_nc_proxy` | *no-collision proxy*: 0 if the plan would have closed the radar gap to zero, else 1. 1 on an empty road | higher |
| `pdm_dac_proxy` | *drivable-area proxy*: 1 if every scored waypoint stayed within ±1.85 m (half a US lane) cross-track of the recorded path | higher |
| `pdm_ttc` | `1 − ttc_below_2s` | higher |
| `pdm_ep` | *ego progress*: `min(1, predicted / recorded)` path length, so overshooting is not extra credit. 1 where the car was stopped and there was no distance to cover | higher |
| `pdm_comfort` | `comfort_within_bounds`, unchanged | higher |
| `pdm_style_score` | the composite above. **Not** the headline | higher |

Three things about it are load-bearing, and eroding any of them turns it back
into an opinion:

* **It is named `pdm_style_score`, never "driving score", and it is not a NAVSIM
  PDMS.** Two terms are proxies for things comma2k19 cannot observe. The dataset
  has no map and no other-agent tracks, so `pdm_nc_proxy` sees only the closest
  radar lead — it cannot see a car the plan would have sideswiped, and a plan
  that leaves the road entirely is invisible to it. `pdm_dac_proxy` uses the
  human's own path as the corridor, which makes it a *lane-keeping* measure
  wearing a drivable-area name: a correctly executed lane change scores 0. A
  number from this column may not be compared with one from
  [navsim_v1.md](navsim_v1.md).
* **A term that cannot be measured is `NaN`, and `NaN` propagates** to the
  composite. A plan too short to have a jerk does not get a free comfort pass.
* **It is reported beside its components, always.** `report.md`'s "Driving score
  (proxy)" group prints all seven columns together.

**What cannot be computed here at all**, so that the absence is not mistaken for
an oversight: Bench2Drive's driving score, neuroNCAP's score and a true NAVSIM
PDMS all require a simulator whose world reacts to the plan. comma2k19 is log
replay — the human drove, the recording does not change — so no number on this
page is a closed-loop result.

**A horizon the model's plan does not reach is `NaN`, never a shorter horizon
under a longer label.** An earlier version of the shared `decompose` clamped the
end index to what was available and emitted anyway, and openpilot's adapter used
to stop at 4.0 s -- so every `fde@5s` figure recorded for this benchmark before
those two fixes is an FDE at 4.0 s and must not be read as a 5 s number.
`scored_horizon_s` is now recorded per sample so the question is answerable from
the results rather than from the changelog.

openpilot's `GRID_S` now reaches supercombo's own 10 s horizon, so all of the
above are scoreable for it. A model that plans less far still gets `NaN` past
where it stopped.

`history_frames` defaults to **160** because the queues hold
`frame_skip * feature_depth` frames of history -- 4 x 24 = 96 for the small
model but 4 x 32 = **128** for the big one -- and one default has to clear both,
with a margin.

It is tempting to justify a larger number by a settling transient after the
buffer fills, and one segment's error trace does look that way. Aggregated over
24 segments it does not: cross-track error at 3 s is 0.20 m in the first 20
frames past the 96-frame boundary and 0.27 m thereafter, already at its floor.
The single-segment impression was noise.

There is no shared external keyframe list the way KITScenes has one, so
`keyframe_indices` generates it from the segment length — deterministic, so
`--limit N` selects the same work at every site, but a property of this
benchmark's code rather than an external artefact. `--limit` counts **samples**,
so with `keyframes_per_segment = 3` a `--limit 5` run stops part-way through the
second segment.

## Evaluation rate

comma2k19 is the only dataset in this repo recorded fast enough to *have* a rate
axis — 20 Hz, against NAVSIM's one keyframe per 0.5 s — so `input_hz` exists
here and nowhere else. It answers the question no other benchmark can: how much
of a model's score is the model, and how much is the density of its input.

```bash
PADDOCK_COMMA2K19_INPUT_HZ=10 paddock run --bench comma2k19 -m openpilot
```

The environment variable wins over `[benchmarks.comma2k19] input_hz`, which wins
over 20. It is an environment variable and not only a toml key because sweeping
one model across rates is the whole point, and that should not mean editing a
site file between runs.

**Only `20 / n` rates are reachable** — 20, 10, 6.67, 5, 4, … — because anything
else would need frames the recording does not contain. `preflight` refuses the
rest and names the nearest reachable rates rather than rounding the stride
silently.

**The scored moment does not move with the rate.** The window is built backwards
from the keyframe, so lowering the rate changes only how densely the model saw
the road leading up to the scored frame.

**`history_frames` counts observations, not recorded frames**, so the model's
feature buffer fills identically at every rate. The cost is that the window's
wall-clock span is `history_frames / input_hz`:

| `input_hz` | stride | span of a 160-frame history |
|---:|---:|---:|
| 20 | 1 | 8.0 s |
| 10 | 2 | 15.9 s |
| 5 | 4 | 31.8 s |
| 4 | 5 | 39.8 s |
| 2 | 10 | 79.5 s — longer than a segment |

At 2 Hz the default window no longer fits in a 60 s segment. `preflight` says so
and names the largest `history_frames` that does, rather than yielding zero
samples with nothing to explain them. Note also that a low-rate number
confounds input rate with history span: the model is being shown far more road
than its buffer was trained to hold. That is intrinsic to the pairing, not a
choice this harness makes.

### Refusing a model the rate cannot feed

`min_input_hz` in `model.yaml` is load-bearing — openpilot declares 10.0 — and a
model that infers ego motion from frame-to-frame change reads a too-slow stream
as a parked car. So below a model's declared floor the pairing is refused before
the model is even loaded, and `--dry-run` reports it:

```bash
PADDOCK_COMMA2K19_ALLOW_SLOW=1 PADDOCK_COMMA2K19_INPUT_HZ=4 \
  paddock run --bench comma2k19 -m openpilot
```

The override exists because this is the only place in the repo where the rate
gap can be **measured** rather than merely refused. A run made with it set is
stored `valid: false` and excluded from `paddock report`, which is the point: a
measurement of the gap is worth keeping and must never enter an average.

## Making a sweep finish

10,165 samples of 160 frames each is the largest thing paddock runs, and the
first full sweep took a day on a machine with 256 cores and two H200s, both
idle. Three profiles, taken with py-spy on that sweep, say where it went:

| where | share of that process's CPU |
| --- | --- |
| node: PIL `ImageFile.load` + `Image.open` (PNG decode) | 59% |
| node: `supercombo.rgb_to_yuv420` | 23% |
| node: onnxruntime `run` — the model itself | **10%** |
| runner: PIL `_encode_tile` (PNG encode) | 39% |

The sweep was a codec with a model attached, on one core each side. Three
settings deal with it, none of which changes a number:

- **`nodes`** (default 4). Every sample is a self-contained `infer_sequence`,
  so N nodes score N samples in the time one scored one. The preparation is
  per-process CPU, so it scales until the cores run out — and the cores are
  what a machine bought for a GPU has spare. Each node still waits for the
  model's declared VRAM before it starts. Raising it also deepens the frame
  cache's retention ring: the runner holds `PIPELINE_BATCHES × nodes` samples
  at once, and staged frames have to outlive every one of them.
- **`frame_cache`**. Decoded frames are staged to disk and read back by the
  node, which runs on the same machine; the default is under `{runs}`, and a
  site whose `{runs}` is a network mount should point this at a tmpfs or a
  local NVMe. They are written raw (`.npy`), not PNG: the two PIL rows above
  are one process compressing what the other is about to decompress.
- **`keep_frames`**. Keeps the staged frames for the next model instead of
  releasing them as the sweep moves on. Raw frames are ~2.4 GB per segment, so
  say yes only with several TB free and several models to amortise the decode
  over. Otherwise the harness holds the last four segments and deletes the
  rest as it goes.

### What they are worth, measured

Same 100 samples every time, on a 256-core machine with two H200s. Rates are
steady state — from the first progress line to the last, so node startup is
excluded; the first row is the real 10,165-sample sweep rather than a slice of
one.

| staging | nodes | samples/s | 10,165 samples would take |
|---|---:|---:|---:|
| PNG under `{runs}` (NFS) — the first sweep | 1 | 0.174 | **16.2 h** (measured) |
| `.npy` under `{runs}` (NFS) | 1 | 0.201 | 14.1 h |
| `.npy` on `/dev/shm` | 1 | 0.547 | 5.2 h |
| `.npy` on `/dev/shm` | 4 (default) | 1.44 | 2.0 h |
| `.npy` on `/dev/shm` | 8 | 1.95 | 1.4 h |

**Every row scored `ade mean = 1.036782`, identical to the last digit.** That is
the claim these settings have to meet, and the reason they are not hashed into
the run id.

Two things are worth reading off it. Dropping PNG is worth little *on NFS*
(0.174 → 0.201) and a great deal on tmpfs (→ 0.547): the encode and the decode
were being paid on top of a filesystem that was itself the wall, so the two fixes
only pay together. And **8 nodes buy 1.36× over 4, not 2×** — past four nodes the
bottleneck has moved to the harness's own single decode pipeline, which is one
process reading one segment at a time.

**The GPU was never the constraint, and still is not.** onnxruntime does use
CUDA here, and across these runs the card averaged 13–24% with peaks of 86% at
eight nodes. supercombo is a small graph; a sweep of it is a data-movement
problem with a model attached, which is what the profile above says and what
an 11× speedup with the model untouched confirms.

**A 10B VLM changes none of that**, which is the sharpest evidence for it: the
alpamayo15 sweep ran the full 10,165 samples in **1.83 h at 1.54 samples/s** on
the same four nodes, marginally *faster* than supercombo's 1.44. A model two
orders of magnitude larger costs nothing extra here because the pipeline is
still waiting on the same decode. What it does need is to stop paying for
history it will not look at — alpamayo conditions on 4 frames of the 160 it is
sent, and warping all 160 on arrival cost several times the inference. That fix
belongs in the model adapter, not here; see `slots_for` and `Frame` in
`adapters/models/alpamayo15/policy.py`.

`--resume <run_id>` continues an interrupted sweep in place, keeping its id.
Scored samples are journalled to `samples.partial.jsonl` as the run goes; a
resume reads that, and the harness skips a segment without opening it when all
of its keyframes are already scored — which matters because a skipped segment
would otherwise cost a minute of HEVC decoded to produce nothing. Resuming
under different switches is refused rather than merged, and the run stays
rankable: being resumed is recorded as a note, not as a reason to exclude it.

## Ground truth needs no post-processing, unlike KITScenes

Every segment ships `global_pose/frame_{times,positions,orientations,velocities}`
— ECEF pose already fused from INS/GNSS/vision, one row per video frame. There
is no SLAM or GNSS pipeline to run.

The one thing that is easy to get backwards is the **quaternion direction**, and
the dataset's own README states it backwards. It says the quaternion rotates
ECEF into the device frame; comma2k19's own `utils/camera.py::device_from_ecef`
transposes it before use, which is the tell. `_quat2rot` returns
`ecef_from_device`, and taking a delta into device coordinates needs the
transpose. Getting it wrong does not fail — it mirrors every relative position
through the direction of travel.

## Inspecting the dataset

comma2k19 arrives as ten separately downloaded chunks, so **a partial copy is
the normal state at a new site**, not an error. Someone fetches `Chunk_1` to get
something running and the rest lands over the following week. Two depths of
answer to "what do I actually have", because the two callers have incompatible
budgets.

`paddock doctor` prints the cheap one under the benchmark's own row:

```
  [ ok ] comma2k19  preflight passed
         segments     2035 (33.9 h) in 178 routes, civic 1653, rav4 382
         chunks       10 of 10 present -- 1:188 2:194 3:202 ... 10:210
         input rate   20 Hz native, evaluating at 20 Hz (native), 160-frame history spanning 8.0 s
         decoder      ffmpeg /some/env/bin/ffmpeg
         toolkit      /some/repos/comma2k19
         calibration  5 routes sampled, pitch +2.6 to +3.6 deg, all plausible
         integrity    64/64 sampled segments hold every optional file
```

Those rows carry no `[ ok ]` glyph and no `->` fix arrow because they are
**facts, not verdicts** — `preflight` already owns the refusals, including the
calibration one this samples. They come from `Benchmark.status()`, which every
adapter may implement and which defaults to printing nothing.

`status()` must stay interactive, since `doctor` calls it on every benchmark on
every machine: `inventory.survey` walks the directory tree, stats nothing,
loads no arrays, and spot-checks completeness on a *sample* of 64 segments
rather than all 2035. It reuses the walk `preflight` just did. The walk itself
delegates to `loader.list_segments` rather than reimplementing "what counts as a
segment", because a second implementation is how a report comes to describe a
dataset the scoring path would not accept.

The expensive answer is a document to hand someone setting the dataset up:

```bash
python -m tools.benchmarks.comma2k19.metadata_report --out COMMA2K19_METADATA.md
```

That loads every array of a stratified sample of segments — two per chunk by
default, `--per-chunk` to change it — and writes twelve sections: an at-a-glance
table, the vocabulary of the directory tree, the layout, per-chunk and per-car
inventories, a storage profile, and measured video / pose / log / CAN /
geography / integrity facts. Measured, not quoted: where it repeats the
publication it says *(from README)* and where it samples it names the sample
size. Nothing on the `doctor` path may call it — it is minutes, not
milliseconds.

Its video section needs a probe, and uses the same backend split as decoding:
PyAV if the environment has it, otherwise `ffprobe` looked for beside
`PADDOCK_FFMPEG` and then on `PATH`. Without either, the section says so and the
other eleven still render.

Both depths report the directory names the walk passed over — a `Chunk_beta`, a
route directory whose name is not `dongle_id|start_time`, a segment whose
`video.hevc` never finished downloading. The harness is right to skip those
silently, since a run must not fail because a download left a partial file
behind, but a person setting the dataset up wants to be told.

## Overlay video

An ADE distribution says how often a model is wrong. Only a picture says how, so
this benchmark can draw on the frames themselves. There are two entry points,
because there are two questions.

**One scored sample**, with every model's stored plan on it:

```bash
paddock render --bench comma2k19 --sample 1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/300
```

The clip replays the history window the model was actually shown, at the rate it
was shown it, then stops on the scored keyframe and holds it with the recorded
future and each run's plan drawn together. The window and the rate come from the
sample's stored `native_id`, so what you watch is what was scored rather than a
nearby frame that looks like it. The benchmark prefix is optional, `--run-id`
picks particular runs, and `--out` chooses the directory (default
`{runs}/comma2k19/_samples`).

**One raw segment**, with no model and no run involved — for checking the
dataset itself, or a calibration, before any evaluation has happened:

```bash
python -m tools.benchmarks.comma2k19.overlay_video \
    --segment "$COMMA2K19/Chunk_1/<route>/3" --max-frames 200 --out /tmp/seg.mp4
```

Both draw the same layers from the same code
([overlay.py](../../src/paddock/adapters/benchmarks/comma2k19/overlay.py)),
adapted from the toolkit's `visualize_segment.py`: the recorded path as a
ribbon on the road plane, CAN radar returns as boxes with the in-lane lead
highlighted, a HUD of the logs at that frame, a north-up minimap of the whole
segment, and — only in the first case — the model's plan, coloured by
`viz.palette` so a model keeps its colour here as in every other figure.

Two things to know before trusting what you see:

- **A mirrored plan is a bug, not a model.** Plans arrive in paddock's ego frame
  (y **left**) and the toolkit projects comma2k19's device frame (y **right**).
  The conversion is a sign, a wrong one mirrors the drawn path about the
  direction of travel, raises nothing, and on a straight road is invisible.
  `test_a_plan_to_the_left_is_drawn_to_the_left` exists for exactly this.
- **Encoding needs a video backend**, the same one scoring needs: PyAV from the
  `comma2k19` extra, or an external ffmpeg named by `PADDOCK_FFMPEG`. Drawing
  additionally needs Pillow, which the `viz` extra declares.

On straight highway the recorded future and a working model's plan should
overlap; they separate in curves, and that separation is the thing worth looking
at.

## Calibration: the thing that was wrong before

openpilot's model never sees a raw frame. `modeld` warps every frame with
`get_warp_matrix(rpyCalib, intrinsics, …)`, where `rpyCalib` describes how the
phone is mounted, and the *calibrated* frame is defined as aligned with the car
in pitch and yaw — so the model always sees the road from the same apparent
viewpoint.

comma2k19 has no `liveCalibration`, so `calibration.py` estimates it with
calibrationd's own formula, substituting the recorded pose for the model's
`cameraOdometry` output (using the model's own output would make the
calibration depend on the model under evaluation). Estimates are **pooled over
a whole route**, because all segments of a route share one mounting and because
5 of 220 segments in a sample have *no* frame passing calibrationd's
`straight_and_fast` gate — stop-and-go traffic. Estimating per segment leaves
those at `rpy = 0`, silently; openpilot never hits this because it persists
`CalibrationParams` across drives.

Result on this dataset: 3.5–3.8° of pitch on the RAV4, 1.7–1.8° on the Civic,
consistent within each vehicle. Correcting it is worth roughly 1.7× in
cross-track error.

## Recorded results

The full dataset, 10,165 samples, every row on the identical sample set
(`sequences_sha256` `293a225a0b5cbaff` in all six run records). A model is
scored only as far as it plans; `—` is where its own horizon ends, not a
failure. Lower is better except `pdm_style_score`.

| | horizon | ADE@1s | ADE@2s | **ADE@3s** | ADE@4s | ADE@5s | ADE@10s | FDE@3s | median ADE@3s | miss@3s | PDMS-style ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo15 (`Alpamayo-1.5-10B`, 2 of 4 camera slots) | 4 s | 0.386 | 0.616 | **0.900** | 1.263 | — | — | 1.654 | 0.823 | — | — |
| qwendrive (`Qwen-Drive-1.0-4B`, side slots replicated) | 5 s | 0.352 | 0.674 | 1.016 | 1.405 | 1.847 | — | 2.075 | 1.005 | 0.501 | — |
| openpilot (`small`, split graph, `4df40d2c1`) | 10 s | 0.598 | 1.035 | 1.507 | 2.017 | 2.571 | 6.032 | 2.712 | 1.555 | 0.647 | 0.410 |
| constant velocity (registered policy, 6 waypoints) | 3 s | 0.537 | 1.017 | 1.639 | — | — | — | 3.286 | 1.475 | 0.658 | 0.417 |
| visionpilot (synthesised trajectory) | 4 s | 1.149 | 2.462 | 4.045 | 5.835 | — | — | 8.146 | 2.195 | 0.864 | 0.406 |
| drivejepa_free (perception-free agent, NAVSIM v1) | 4 s | 7.800 | 12.077 | 16.048 | 20.501 | — | — | 26.248 | 19.801 | 0.831 | 0.541 |

Two rows carry `—` in the last column and the reason differs. `alpamayo15`
predates it and is `stochastic: true`, so re-running for two proxy columns would
replace a published 0.900 with a different draw. `qwendrive` is the one model
here that plans on a 0.1 s grid: its recorded `comfort_*` values were measured
on that grid, before `metrics/comfort.py` moved to a common `REFERENCE_DT`, and
they are stale against current code — 1932 m/s³ of "jerk" that was a property of
4 cm of output quantisation amplified by `1/dt³`, not of the plan. Recomputed
from its stored predictions the figure is 11.4. Publishing either the stale
number or a recomputation that no recorded run produced would be worse than
publishing neither, so the comfort and `pdm_style_score` cells are `—` until it
is swept again. The five models at `dt = 0.5` are bit-identical under the change.

**The constant-velocity row stops at 3 s, and that is a change in producer, not
in arithmetic.** The earlier record
(`constant_velocity/20260911T231819-736a8a`) came from
`tools/benchmarks/comma2k19/constant_velocity.py`, a standalone script that
extrapolates the recorded speed across the full 10 s ground-truth horizon and is
stored under a pseudo-model, as its own `notes` field says. The row above is the
registered `constant_velocity` policy, which emits the six waypoints neuroNCAP
requires — 3 s at `waypoint_dt` 0.5. They agree to every printed digit at 1, 2
and 3 s; the tool's 4/5/10 s figures are 2.411, 3.331 and 10.048, and remain the
long-horizon reference line because nothing else here reaches 10 s except
openpilot.

Constant velocity is better at 1 s and has the better *median* at 3 s than
openpilot; openpilot wins the 3 s mean and pulls away with every horizon they
share. Both facts come from the same property of the data — I-280 is mostly
straight, so the baseline is nearly exact most of the time and wrong only on the
curves, which is exactly where a mean lives and a median does not. Reporting one
of the two alone would misdescribe the result.

Two of the four models added after the first sweep need their rows read with a
caveat attached, and in both cases the auxiliary columns are what supply it.

- **drivejepa_free's 16.048 is a reach failure, not a shape failure.** Its
  `lateral_at_arclength_mae` is 0.609 — the lowest of any model on this
  benchmark, openpilot included at 1.155 — while `path_length_ratio` is 0.672.
  It draws close to the right curve and travels two thirds of it. Four seconds
  of NAVSIM city driving saturates near 64 m and I-280 covers 120, so this is a
  training-domain speed prior meeting a highway. It is also the clearest
  demonstration in the repo of why this benchmark refuses to report ADE alone:
  ADE calls it the worst model here by an order of magnitude, and the two
  columns that separate speed from shape say the shape is the best on the board.
- **visionpilot has no native trajectory**, so its row grades a synthesised
  pipeline — perception, ground projection, curve fit, cruise-speed assumption
  and extrapolation past a 1 s planner horizon — and is not comparable in kind
  with a model that emits a timed trajectory. Its median (2.195) is barely half
  its mean (4.045), which is the signature of that extrapolation failing on a
  minority of samples rather than a uniform error.

Alpamayo is ahead of both at every horizon it reaches, on the mean *and* the
median, and it is the median that matters for the comparison above: constant
velocity beats openpilot there and does not come close here. Two qualifications
belong with that number rather than after it.

- **It sees one camera where it asks for four view slots.** Alpamayo takes three
  nominally 120° views at +55/0/−55° and a 30° telephoto; comma2k19 is a single
  forward camera, so it fills the centre slot and the telephoto cropped out of
  it and the two side slots go unsent. They are omitted rather than filled with
  the forward view, because `camera_indices` lets the model be told a view is
  absent and there is no way to tell it a view is a different view than it is.
  On a highway that costs less than it would in town, but it is a handicap, not
  a neutral difference — and the recorded `config.cameras` in `run.json` lists
  the three the *policy declares*, not the one the dataset supplied. The node's
  `camera_slots` diagnostic is the field that says what it actually got.
- **It plans 4 s, so the columns where openpilot separates from the baseline do
  not exist for it.** `scored_horizon_s` is 4.0 on every sample: 64 waypoints at
  10 Hz, of which paddock takes eight on the 0.5 s grid. Whether it would hold
  its lead at 10 s is not a question this table answers.

Its plan is also much smoother than supercombo's: `comfort_within_bounds` 0.956
against 0.302, and `comfort_lon_jerk_max` 0.64 m/s³ against 5.04 — see the jerk
result below, which is the same measurement from the other side.

Of the six, only openpilot and visionpilot publish auxiliary heads through
`Trajectory.extras`; alpamayo15, qwendrive and drivejepa_free emit a trajectory
and nothing else, so every metric in the next two paragraphs is `NaN` for them.
That is the intended behaviour of one declared metric list rather than a
per-model one: the run records what the model did not produce instead of quietly
shortening the list.

visionpilot publishes `lead` and deliberately not `lane`. The schema wants four
polylines ordered left to right; AutoSteer emits a single fitted path centreline
sampled per image row, and offsetting it by half a nominal lane to manufacture
two boundaries would make `lane_width_m` report the assumption back as a
measurement. Its lead head is the interesting comparison with supercombo's:
three times worse at range (`lead_range_abs_err` 9.10 m against 3.04) and better
at lateral placement (`lead_lateral_abs_err` 0.369 m against 0.595), which is
what a detect-and-project stack against a regression head should look like.
Neither publishes what the other lacks, so `speed_rel_err`, `yaw_rate_abs_err`
and the sanity gates stay openpilot-only.

The decomposition, for openpilot: `lateral_mae` 1.092 against `longitudinal_mae`
0.834, `along_track_bias` +0.400 (the plan runs slightly *ahead* of the car),
`path_length_ratio` 1.014, `scored_horizon_s` 10.0 on every sample. The ego-speed
head is accurate — `speed_rel_err` 0.020, `yaw_rate_abs_err` 0.0024 rad/s — so
the path error is a path error and not a misread of how fast the car was going.

Two auxiliary results are worth naming because they are not confirmations:

- **The lateral uncertainty head is under-dispersed.** `within_1sigma_y` is
  0.124 where a calibrated σ implies ≈0.68 (`within_1sigma_x` 0.559 is closer
  but still low). Read σ_y as a relative confidence, not a probability.
- **Longitudinal jerk leaves the comfort bound on 63% of samples.**
  `comfort_lon_jerk_within_bounds` 0.373, mean peak 5.04 m/s³ against NAVSIM's
  4.13. The other four comfort components are ≥0.93. This is a property of the
  plan resampled onto the 0.5 s grid and matters to anyone treating that grid
  as a controller input rather than as a score.

The lead head is the mixed one: `lead_recall` 0.945 against `lead_precision`
0.593 — radar has a lead on 46% of samples and the model claims one on 73% — with
`lead_range_abs_err` 3.04 m mean, 1.45 m median.

**Against the standalone harness's ~1.6–2.1 m "ADE over 0–10 s" below, this
`ade@10s` of 6.03 m is not a contradiction and not a reproduction.** The two
average different point sets: that harness scored supercombo's own 33 `T_IDXS`
points, which are packed near the car — over half of them fall in the first 2 s
— so its mean is dominated by short horizons, while this grid is uniform in time
out to 10 s. The comparable pair is that harness's speed error (~2% mean) against
this run's `speed_rel_err` of 0.020, which do agree. Nothing here re-derives that
number; it is quoted, and the two protocols are stated so neither is mistaken for
the other.

## Reproduction status and the previous result

**The previously published number for this pair was wrong.** It is recorded here
because it is instructive, not to be re-derived.

`results/comma2k19/openpilot/20260904T011824-9a8260` reported, on 100 segments:

| metric | mean |
|---|---:|
| `ade` | 38.84 |
| `fde@1s` | 22.29 |
| `fde@2s` | 44.43 |
| `fde@3s` | 66.36 |
| `longitudinal_mae` | 38.82 |
| `lateral_mae` | 0.96 |

At the recorded ~22 m/s, those FDEs are *exactly the distance travelled* — the
model was predicting a **stationary car**. `lateral_mae` was low not because
lane tracking worked but because the prediction sat at the origin and the
ground truth's own lateral excursion is small.

The cause was one line in the removed `loader.py`:

```python
CAMERA2EGO = np.eye(4)
```

`OpenpilotPolicy`'s `CameraAdapter` derives `rpyCalib` from `camera2ego`, and
**that derivation is correct** — it round-trips exactly
(`tests/test_harness.py::test_camera2ego_round_trips_through_openpilot`). The
defect is the input, and it is two errors in one line:

1. **The identity is not "no mounting rotation."** Paddock's camera frame is
   [right, down, forward] and its ego frame is [forward, left, up], so an
   unrotated camera is `[[0,0,1],[-1,0,0],[0,-1,0]]`. The identity says "the
   camera's axes are the ego's axes", and the correct formula faithfully turns
   that into `rpy = [-90°, 0°, +90°]`. With that warp, **every pixel the model
   sees comes from outside the camera image** and is a smeared border replicate.
2. **There is a real mount rotation and it is not zero** — see above.

The accompanying analysis attributed the result to the EON's 1164×874 sensor
being "upscaled 66 % to fill supercombo's 1928×1208 input", a cross-hardware
generalisation gap. That upscale was real and unnecessary — the ONNX backend
warps from the native resolution and resamples nothing — but it was not the
cause, and the conclusion was wrong.

**What is now in place so this cannot recur silently:**

- `CameraAdapter` raises `ImplausibleMounting` for any derived mounting beyond
  ±15°, naming the identity case explicitly in the message.
- `Comma2k19Benchmark.preflight` refuses to run when the estimated calibration
  is implausible or unavailable.
- `score` reports `along_track_mae` / `cross_track_mae` against the
  ground-truth path tangent, plus `path_length_ratio`, so "the plan went
  nowhere" is a number rather than something to be inferred from a large ADE.
- `diagnostics()` reports the measured fraction of the model's input that came
  from real pixels. For the broken configuration it is 0.000.

### Corrected reference numbers

Measured before this adapter existed, by a standalone harness that scored
**every frame** of a 10 % sample (220 segments, ~241k frames) by driving the
model directly rather than through the `Policy` protocol. That harness is not in
this repository; these figures are quoted from its report to establish the scale
of the correction, and are **not reproducible from this tree**:

| | small | big |
|---|---:|---:|
| ego-speed error vs recorded | ~2 % mean, ~0.5 % median | ~1.9 % / 0.3 % |
| ADE over 0–10 s | ~2.1 m | ~1.6 m |
| cross-track at 3 s | ~0.27 m | ~0.24 m |

They are per-frame figures over the model's full 10 s horizon and are **not**
directly comparable to this adapter's per-sample 3 s ADE.

## Which openpilot revision produced a number

openpilot ships its driving model as **two different graph layouts**, and which
one a checkout has is not something a tag ordering tells you:

| layout | files | revisions |
|---|---|---|
| split | `driving_vision.onnx` + `driving_policy.onnx` | v0.11.1 (`4df40d2c1`, 2026-05-29) |
| combined | `driving_supercombo.onnx` | `555f48c5d` (2026-08-19, the `repos.yaml` pin), `675ff5698` (2026-09-04) |

The combined graph is the *later* one. The recorded kitscenes / neuroncap /
bench2drive runs were all produced against it, which is why the pin is left
alone: moving it would silently make those numbers incomparable.

`policy._open_session` therefore selects on **what is on disk**, not on a version
string — a checkout is a sha, openpilot has shipped both, and the files are the
only thing that cannot be wrong about which. `supercombo.SupercomboSession` and
`SplitSession` each check their graph's declared inputs on load and raise with
both input sets named, so a checkout carrying the *other* layout fails at load
with a message that says so rather than on a shape mismatch mid-run.

The two are not interchangeable, and the difference is not cosmetic:

- The split layout runs vision first and pushes its feature **before** the policy
  reads the buffer, so the policy plans from the current frame. The combined
  graph reads the previous frame's hidden state — a one-tick lag that
  `ModelState.run` has and that this adapter reproduces rather than fixes.
- The feature queue is `frame_skip * (D - 1) + 1` deep in the split layout and
  `frame_skip * D` in the combined one, so `[::frame_skip]` lands on the newest
  entry in the first case and four ticks short of it in the second.
- The split policy graph has no `action_t` input and publishes no `action` head,
  so `NATIVE_LONG` takes its acceleration from the plan there.

Both are pinned by `adapters/models/openpilot/tests/test_supercombo.py`, which
drives each layout with fake graphs and asserts which frame's feature the policy
actually sees.

`diagnostics()` names every graph file loaded — `diag["onnx"]` is a dict, not a
path, for this reason — and the node serves it at `/diagnostics`. Note that
nothing in paddock currently *fetches* that endpoint, so it does not reach
`run.json`: which layout produced a recorded number is answerable by looking at
the checkout, not by reading the run. Worth closing, but it is a framework
change and not this adapter's to make unilaterally.

## Splits

`rav4` (chunks 1–2, dongle `b0c9d2329ad1606b`, 382 segments) and `civic`
(chunks 3–10, dongle `99c94dc769b5d96e`, 1653 segments) — the two vehicles the
dataset was collected with and two different EON mountings, not a train/val
split.

Note that `--limit N` draws from the front of the sorted (chunk, route,
segment) order, so a small N is **all RAV4**. Every number from a limited run
should say so; `paddock run` records `limit` in `run.json` and marks such runs
`valid: false`.

## What comma2k19 cannot score

- **No lane-line or road-edge labels.** Any lane figure is plausibility, not
  accuracy.
- **No object boxes.** CAN radar is the only object ground truth, and it is a
  stream of individual tracked points with no lead label.
- **No brake/gas/blinker ground truth** without decoding `raw_can` with a DBC.
- **No published model-evaluation protocol.** comma2k19's own benchmark scores
  GNSS positioning, so this benchmark's metric definitions are its own; see
  `metrics.py` for what each one is and why.
