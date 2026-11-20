# comma2k19

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `comma2k19` |
| Primary metric | `ade` |
| Adapter | [`adapters/benchmarks/comma2k19/`](../../src/paddock/adapters/benchmarks/comma2k19/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [alpamayo2edge](../models/alpamayo2edge.md), [autoe2e](../models/autoe2e.md), [constant_velocity](../models/constant_velocity.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [simlingo](../models/simlingo.md), [uniad](../models/uniad.md), [vad](../models/vad.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

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

Every frame also carries the car's pose relative to the first frame
(`ego2world`), its CAN speed and its yaw rate. The pose, the plan and the ground
truth are all in the car's frame: the recorded pose describes the camera, so the
route calibration that builds `camera2ego` rotates the pose and the ground truth
too. The yaw rate is the one exception: it still turns about the camera's down
axis, a scale error of at most 0.2 % (the cosine of the mount pitch).

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

## What each model sees

comma2k19 has one camera, a road-facing phone, and several models expect more.
The rule for every model is the same: a slot that means a forward view may get
a view made from the road camera (a wider or narrower crop of it); a slot that
means another direction gets nothing: black, masked out or left out, depending
on the model. A slot is never filled with the forward image at a bearing that camera
never had. This is a limitation of the dataset, and it is stated here so every
row is read with it.

| Model | What it gets |
|---|---|
| OpenPilot | the road frame in both its narrow and wide inputs, each warped with the road camera's own lens. Upstream `modeld` does the same on a one-camera device, and lists the EON that recorded comma2k19 as one |
| Alpamayo 1.5 | a front-wide crop and a 30° tele crop of the road frame; its front-left and front-right cameras are left out, which the model accepts |
| Alpamayo 2 Edge | the road frame in its front slot and a 30° tele crop of it in its telephoto slot; its four other slots are black, because the vendor's code requires all six. The run sets `ALPAMAYO2_EDGE_ABSENT_CAMERAS=black` to allow that, and its diagnostics list them as `black_slots` |
| METEOR | a front-wide crop and a 30° narrow window of the road frame; its six other slots are black, which is how its training marks a missing camera |
| AutoE2E | the road frame in its front view; its five other views are masked out |
| Qwen-Drive | the road frame in its front view; its front-left and front-right views are black, because the model has no flag for a missing camera. The run's diagnostics list them as `blank_views` |
| Drive-JEPA | the road frame in its front view; its front-left, front-right and back views are black. The model keeps only its front camera, so they never reach the plan; they are there because its loader expects four. The run's diagnostics list them as `blank_views` |
| UniAD, VAD | the road frame as CAM_FRONT, at its own intrinsics and mount; their five other slots are black at nuScenes' own calibration of them, because the spatial cross-attention projects every BEV point through all six. The run's diagnostics list them as `blank_views` |
| VisionPilot | its one camera |
| constant velocity | no image |

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
frame_cache = "/dev/shm/paddock-comma2k19-$USER"  # where decoded frames are staged
keep_frames = false           # keep them for the next model instead of releasing
```

Frame decoding needs either **PyAV** in paddock's environment or an ffmpeg
binary via `PADDOCK_FFMPEG`. `preflight` reports which, and refuses to run
without one. `make setup` installs PyAV (the `comma2k19` extra) already; on an
environment built by hand it is `pip install -e ".[comma2k19]"`.

Getting the dataset: see [data setup](../data-setup.md#comma2k19). It is ten zips on
Hugging Face, `Chunk_1.zip` … `Chunk_10.zip`, ~100 GB unpacked.

### The toolkit is required, not optional

comma ships a toolkit with the dataset, and every rotation under a comma2k19
score comes from it. This clones it into `[roots] repos` at the sha pinned in
`repos.yaml`:

```bash
paddock setup --repo comma2k19
```

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
the defect described below, is exactly what those tests exist to catch.

Two things partly compensate, and it is worth being exact about which of them a
pull request actually gets:

- The toolkit is **pinned**, checked by `test_the_toolkit_is_pinned_in_repos_yaml`.
  That one needs no clone and does run in CI.
- `test_the_toolkit_rotations_match_an_independent_reference` checks the toolkit
  against algebra written out again in the test file rather than checking
  paddock against itself. It takes the `toolkit` fixture, so it needs the clone
  and **skips in CI like the rest** — it is a check the reviewer runs at a site,
  not one the PR proves. An earlier version of this paragraph claimed both ran
  in CI; only the first does.

What the clean environment does still exercise is everything downstream of the
rotations: the metric decomposition, the evaluation grid and the horizon rules
take no fixture at all, and the staging, cache-lifetime and `--resume` tests take
`rotations`, which substitutes the reference algebra where there is no clone.

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
| `scored_horizon_s` | how far the prediction and the recording actually overlapped, seconds | n/a |
| `longitudinal_mae`, `lateral_mae` | axis-aligned error split, metres | lower |
| `along_track_mae`, `cross_track_mae` | the same split against the ground-truth *path tangent* | lower |
| `along_track_bias` | signed along-track error; positive means the plan ran ahead | toward 0 |
| `lateral_at_arclength_mae` | lateral error with speed error removed, metres | lower |
| `predicted_path_length`, `recorded_path_length` | distance covered over the horizon, metres | n/a |
| `path_length_ratio` | predicted / recorded distance — the progress component | toward 1 |

`ade@3s` is not a duplicate of `ade`: the headline averages over whatever the
plan reached, `ade@3s` is `NaN` unless the plan reached 3 s.

`scored_horizon_s` is marked n/a rather than "higher" because it describes the
model, not its quality — a 4 s planner scored over 4 s is at its own horizon,
which this doc says elsewhere is "not a failure". The leaderboard renders it
with no arrow for the same reason.

**The two path lengths in the recorded runs are one interval short.** Both were
computed by differencing the waypoints, which drops the leg from the ego origin
to waypoint 0 — the longest one. At 30 m/s over a 3 s horizon that is 75 m
reported for 90 m travelled. Fixed in the metric; the published runs were not
re-swept, so `predicted_path_length` and `recorded_path_length` in
`results/comma2k19/**` are still the short figures and a re-run will not
reproduce them.

**`path_length_ratio` moved too, slightly.** An earlier version of this note
said the ratio was unaffected because "both sides lost the same leg". They do
not: the missing leg is the prediction's own first step on one side and the
recording's on the other, and those differ by exactly the speed error the ratio
is there to measure. Dropping both pulls the quotient toward 1 — the error it
hides is largest for the samples furthest from 1, which are the ones the metric
exists to find. The second-order effect is the gate: `MIN_GT_PATH_LENGTH_M`
tests the recorded length, so a sample just under 2 m by the short measure and
just over it by the correct one is NaN in the published runs and scored in a
re-run. Both effects are small at 10 s horizons and neither is zero; treat the
published ratio column as the old definition.

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

**The comfort rates do not share a denominator across models.** A plan too
short, not finite, or too coarse to resample onto the 0.5 s reference grid is
NaN rather than a failure — "not measurable" is not "uncomfortable" — and the
aggregate drops it. On the runs recorded below, the acceleration and jerk rates
are judged on all 10,165 samples for every model, but the yaw-rate and
lateral-acceleration rates, and so `comfort_within_bounds`, on 9072 (METEOR) to
10150 (VisionPilot) of them: those two need two plan steps in a row of at least
0.25 m to have a heading to difference, so a plan that stands or creeps is not
judged on them. Read a comfort column
as *of the plans this model made that could be judged, this fraction were
comfortable*: comparable as a rate, not subtractable. `metrics.json` carries `n`
and `n_missing` per metric, so the denominator is always recoverable from the
record; it is the rendered table that shows the mean alone.

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
| `pdm_ep` | *ego progress*, NAVSIM's rule (`metrics.pdm.ego_progress`): the plan's progress along the recorded path over the recorded path's length, at most 1, so overshooting is not extra credit; 1 when neither covers 5 m | higher |
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
The single-segment impression was noise. (Both figures are against the
camera-frame ground truth of the time; the comparison between them is what
carries the argument.)

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

## Streamed mode: Drive-JEPA's momentum, with a warm-up

Off by default. Every sample goes to the node through `/infer_sequence`, which
resets it, so Drive-JEPA's momentum-aware selection -- declared on in its
`model.yaml` -- never has a previous plan here (the recorded Drive-JEPA run's
diagnostics say `fed: 0` of 1,271 replans on its node), and no order of the
samples could change that: a segment's keyframes are 210 frames (10.5 s) apart.

`PADDOCK_COMMA2K19_STREAMED=1` gives each sample a **warm-up**, as on KITScenes
(see [KITScenes](kitscenes.md#streamed-mode-drive-jepas-momentum-with-a-warm-up)):
just before the scored request, on the same node, the runner plans the window's
frame 0.5 s before the keyframe (index -10 at 20 Hz; picked by timestamp, so
another `input_hz` or a dropped frame does not move it) from the window up to
it, under a stream keyed by the sample, at that frame's time and its pose in
the window's frame -- the frame every `ego2world` of the sample is in -- and
discards the plan. The scored request follows under the same key and is handed
that plan. Both go to one node back to back inside one runner job, so several
nodes are safe. **Expected: every scored sample fed.**

**This is a Paddock choice, not upstream protocol**, for the reason given on
the KITScenes page. Recorded as `config.streamed = "warmup_0.5s"`, per sample
as `extra.warmup` and `extra.momentum_fed`, and counted in metrics.json's
`by_group.streamed`; the scored request is a standard run's plus the `stream`.

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
  `$USER` is expanded, and the recommended path keeps it: `/dev/shm` is shared
  by everyone on the machine, and a directory the first user created is one the
  second cannot write into. `preflight` probes the path for writability and
  free space, because staging starts *after* the nodes are up — without the
  check, a wrong value costs a checkpoint read and a VRAM wait before it fails.
- **`keep_frames`**. Keeps the staged frames for the next model instead of
  releasing them as the sweep moves on. Raw frames are ~2.4 GB per segment —
  about 4.9 TB for all 2033 — so say yes only with that free and several models
  to amortise the decode over; `preflight` refuses if the volume cannot hold
  it. Otherwise the harness holds the last four segments and deletes the rest
  as it goes.

  Omitting it and writing `keep_frames = false` mean the same thing. That is
  worth stating because for a while they did not: every setting in this section
  arrives as a string, so `false` reached the harness as `"False"` and read as
  **true** — the one setting whose job is to protect the disk turned retention
  on for anyone who set it explicitly. Booleans now go through
  `BenchmarkOverride.flag`, which reads `false`, `0`, `no`, `off` and an empty
  value as off.

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

One thing that is easy to get backwards is the **quaternion direction**, and
the dataset's own README states it backwards. It says the quaternion rotates
ECEF into the device frame; comma2k19's own `utils/camera.py::device_from_ecef`
transposes it before use, which is the tell. `_quat2rot` returns
`ecef_from_device`, and taking a delta into device coordinates needs the
transpose. Getting it wrong does not fail — it mirrors every relative position
through the direction of travel.

The pose describes the camera, not the car, so the recorded future is also
rotated by the route calibration before it is scored, into the car's frame that
every plan is in. That rotation's direction is the other thing easy to get
backwards: the inverse doubles the mount yaw instead of removing it.
`loader.future_in_ego_frame` and its tests pin the sign.

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

Three scenes are rendered with all six models together, chosen by querying the
recorded samples rather than by eye so that "a curve" means a measured one:

| scene | how it was selected | sample |
|---|---|---|
| straight highway | speed > 29 m/s, \|yaw rate\| < 0.002 rad/s | `3/99c94dc769b5d96e\|2018-05-03--08-13-12/23/999` |
| sustained curve | speed > 20 m/s, \|yaw rate\| > 0.06 rad/s | `4/99c94dc769b5d96e\|2018-06-27--15-03-44/18/998` |
| pull-away from low speed | lowest speeds, ranked by distance covered over the horizon — 2.65 m/s travelling 20.7 m, so a real departure and not a parked car | `8/99c94dc769b5d96e\|2018-08-17--15-51-25/6/999` |

One sample id addresses the same scene for every row, because all nine runs score
the identical sample set. The mp4s land in `{runs}/comma2k19/_samples/` and are
not tracked — they are tens of megabytes and regenerate from the stored runs.

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

Three things to know before trusting what you see:

- **A mirrored plan is a bug, not a model.** Plans arrive in paddock's ego frame
  (y **left**) and the toolkit projects comma2k19's device frame (y **right**).
  The conversion is a sign, a wrong one mirrors the drawn path about the
  direction of travel, raises nothing, and on a straight road is invisible.
  `test_a_plan_to_the_left_is_drawn_to_the_left` exists for exactly this.
- **A plan is in the car's frame, like the radar.** `camera2ego` carries the
  route calibration (see [Calibration](#calibration-the-thing-that-was-wrong-before)),
  so the overlay turns each plan into the camera's frame with the calibration its
  model was handed, `native_id.rpy_calib_deg`, exactly as it turns a radar box.
  Clips rendered before that was fixed drew every plan `focal·tan(yaw)` px
  sideways and `focal·tan(pitch)` px low — up to 37 px and 16–61 px across this
  dataset's routes — so a plan running straight down the lane sat beside the
  recorded path with its far end folded onto the road 18–67 m ahead. They show
  a model that plans short, or hooks, and it was doing neither.
  `test_the_overlay_draws_a_plan_through_the_mount_rotation` pins it.
  The scorer rotates the ground truth by the same calibration, so a plan drawn
  on the recorded path also scores as on it. A published row made before that
  still carries the mount yaw until its model is re-run (see
  [Recorded results](#recorded-results)).
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

Result on this dataset: 3.3–3.8° of pitch on the RAV4 (39 of its 42 routes) and
1.6–2.2° on the Civic (126 of 135), consistent within each vehicle and never
overlapping between them — the Civic's highest is 2.35°, the RAV4's lowest
2.57°. Correcting it was worth roughly 1.7× in cross-track error, measured
against the camera-frame ground truth used before the ground truth was rotated
into the car's frame too.

## Recorded results

The full dataset, 10,165 samples, every row on the identical sample set
(`sequences_sha256` `293a225a0b5cbaff` in all nine run records). A model is
scored only as far as it plans; `—` is where its own horizon ends, not a
failure. Lower is better except `pdm_style_score`.

**Every row was re-run in November 2026 and is scored against a ground truth in
the car's frame, the frame the plans are in.** The rows published before then
were scored against a ground truth in the camera's frame. The two differ by the
mount rotation, about 2° of yaw on the Civic; that alone moved openpilot's
ADE@3s from 1.507 to 0.843, and it reordered the table. The scorer now rotates
the ground truth by the route calibration, the same one `camera2ego` hands the
model, and the four models that read their own past path (Alpamayo 1.5,
Qwen-Drive, AutoE2E and Drive-JEPA) are shown it in the car's frame too.

**Every PDMS-style and comfort number on this page uses the rules adopted on 22
Oct 2026.** Progress follows NAVSIM's rule: a plan covering 2 m where the human
drove 4 m scores 1.0, because neither covers 5 m. And a step under 0.25 m per
0.5 s gets no heading, so a car slowing to a stop with 3 cm of wobble passes the
yaw-rate bound. Rows recorded before that date scored those two cases 0.5 and a
failure.

| | horizon | ADE@1s | ADE@2s | **ADE@3s** | ADE@4s | ADE@5s | ADE@10s | FDE@3s | median ADE@3s | miss@3s | PDMS-style ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alpamayo2edge (`Alpamayo2-Edge` f8951d0, the road camera in 2 of 6 slots, 4 black) | 6 s | 0.086 | 0.225 | **0.438** | 0.713 | 1.043 | — | 1.013 | 0.356 | 0.089 | 0.949 |
| qwendrive (`Qwen-Drive-1.0-4B`, side slots black) | 5 s | 0.154 | 0.326 | **0.589** | 0.946 | 1.411 | — | 1.316 | 0.498 | 0.195 | 0.779 |
| alpamayo15 (`Alpamayo-1.5-10B`, 2 of 4 camera slots) | 4 s | 0.152 | 0.427 | 0.819 | 1.314 | — | — | 1.864 | 0.737 | 0.395 | 0.690 |
| openpilot (`small`, combined graph, `555f48c5d`) | 10 s | 0.299 | 0.548 | 0.843 | 1.185 | 1.582 | 4.396 | 1.611 | 0.643 | 0.293 | 0.816 |
| constant velocity (registered policy, 6 waypoints) | 3 s | 0.252 | 0.611 | 1.135 | — | — | — | 2.546 | 0.816 | 0.466 | 0.650 |
| autoe2e (`kitscenes-epoch-5`, the road camera alone, five views left out) | 6 s | 0.734 | 0.977 | 1.350 | 1.949 | 2.846 | — | 2.396 | 1.339 | 0.640 | 0.766 |
| visionpilot (synthesised trajectory) | 4 s | 0.964 | 2.126 | 3.592 | 5.287 | — | — | 7.415 | 1.513 | 0.747 | 0.676 |
| vad (`VAD_base`, the road camera as CAM_FRONT, five slots black) | 3 s | 5.759 | 9.682 | 13.658 | — | — | — | 23.621 | 16.093 | 0.846 | 0.781 |
| uniad (`uniad_base_e2e`, the road camera as CAM_FRONT, five slots black) | 3 s | 6.242 | 10.538 | 14.917 | — | — | — | 25.928 | 18.008 | 0.888 | 0.679 |
| drivejepa (perception-based agent, NAVSIM v2; the road camera in 1 of 4 slots) | 4 s | 7.474 | 13.080 | 18.754 | 24.385 | — | — | 32.893 | 23.401 | 0.852 | 0.620 |
| simlingo (`RenzKa/simlingo` 26c7c89, the road camera placed in its 110 deg frame) | 2.5 s | 7.337 | 12.749 | 19.320 | — | — | — | 37.847 | 21.478 | 0.928 | 0.355 |
| meteor (`meteor_v157c3Z.onnx`, the road camera in 2 of 8 slots, `narrow_refpose`) | 3 s | 11.841 | 21.621 | 30.167 | — | — | — | 51.507 | 37.395 | 0.892 | 0.034 |

**VAD plans no faster than about 18 m/s,** the top of the nuScenes city driving it was trained on (run `20261112T213250-61c395`, all 10,165 keyframes). Where the recorded path averages 25 m/s or more over the next 3 s, its plan's median speed is 18.4 m/s. Two thirds of comma2k19's keyframes (6,689 of 10,165) average 20 m/s or more, and there its median ADE@3s is 20.27 m; below 20 m/s it is 1.36 m. UniAD has the same ceiling, a little lower (run `20261112T213250-a862cc`, all 10,165 keyframes): its plan's median speed there is 16.95 m/s, and its median ADE@3s is 22.45 m at 20 m/s or more and 1.64 m below. Its first 1,692 samples ran at paddock `0093535`; the other 8,473 ran at `79a230f`, which computes a black view's backbone features once instead of on every step (PR #217; at most 0.36 mm from the uncached plan on the same input).

The drivejepa row above is `20261106T192049-391e2d`, in which momentum was on and
could not act: each sample reset the node. It is retired by the momentum-aware run
`20261111T074533-0f26ee` (a 0.5 s warm-up, a Paddock choice; see
[Streamed mode](#streamed-mode-drive-jepas-momentum-with-a-warm-up)), 10,163 of 10,165
keyframes fed: ADE@3s 18.740, the support-matrix row. The analysis of the drivejepa
row on this page was made on the retired run; the two differ by 0.014 m at 3 s.

**The openpilot row is the run of 6 Nov 2026 at the `repos.yaml` pin,
`555f48c5d`,** on the ONNX backend; its `run.json` records both
(`provenance.repo.at_pin`, `provenance.policy.backend`). The row it replaces,
`openpilot/20260916T051616-2a91ca`, came from a clone at `4df40d2c1`, which
ships the model as two graphs where the pin ships one — see [Which openpilot
revision produced a number](#which-openpilot-revision-produced-a-number) for
what differs. On this dataset the revision changed almost nothing: that row's
own plans, scored against the ground truth in the car's frame, give ADE@3s
0.843, the same as this run, and no sample's ADE@3s differs between the two by
more than 5 cm. So the drop from 1.507 to 0.843 comes from the ground truth's
frame, not from the revision. The fix that stopped the ONNX backend feeding the
road camera's image to the wide view ([the openpilot
page](../models/openpilot.md#which-backend-ran-and-one-number-it-used-to-change))
does not touch comma2k19 either: it has one camera, so that frame feeds both of
the model's views, before the fix and after it.

`alpamayo15` is `stochastic: true`, so its re-run is a new draw as well as a new
scoring. Scored against the ground truth in the car's frame, its earlier run's
own plans give ADE@3s 0.867; the re-run, shown its past path in the car's frame
too, gives 0.819. The earlier run predated the miss-rate and PDMS-style columns,
which the re-run fills.

**The meteor row is the released graph shown one camera of the eight it was
built for**: the road camera in the wide front slot, its centred 30° window in
the narrow front slot, the other six black with the calibration of METEOR's own
rig (`METEOR_FRONT_ONLY`, chosen on METEOR's own scene; see
[docs/models/meteor.md](../models/meteor.md)). Its plans stop about 17 m ahead
at every speed above 15 m/s, where the car went 55-92 m, and bend about 3 m to
the right. It beats constant velocity on 3.5 % of samples, all below 15 m/s.

`qwendrive` was `—` here too until it was swept again, and the reason is worth
keeping because it is a property of the grid rather than of the model. It used
to emit all 50 poses of its output at `waypoint_dt` 0.1; `metrics/comfort.py`
differences jerk as `1/dt³`, so 4 cm of output quantisation became 1932 m/s³ of
"jerk" — a number about the quantiser, not about the plan. The model now reports
the ten poses on the 0.5 s grid every other row uses, and the row above has a
jerk of 11.50 m/s³. When it first moved to that grid, nothing else in its row
moved for that reason: the ADE columns changed because the grid an ADE averages
over changed, which is the `score()` alignment described under [What is
scored](#what-is-scored), not a change in what the model planned. The five
models already at `dt = 0.5` are untouched throughout.

**The constant-velocity row stops at 3 s, and that is a change in producer, not
in arithmetic.** The earlier record
(`constant_velocity/20260911T231819-736a8a`) came from
`tools/benchmarks/comma2k19/constant_velocity.py`, a standalone script that
extrapolates the recorded speed across the full 10 s ground-truth horizon and is
stored under a pseudo-model, as its own `notes` field says. The row above is the
registered `constant_velocity` policy, which emits the six waypoints neuroNCAP
requires — 3 s at `waypoint_dt` 0.5. Scored against the same ground truth, the
two agreed to every printed digit at 1, 2 and 3 s. The script has not been run
since the ground truth moved to the car's frame, so its 4/5/10 s figures (2.411,
3.331 and 10.048) are from the camera's frame, where the mount's yaw adds an
error that grows with distance. They are the only constant-velocity figures past
3 s, and nothing else here reaches 10 s except openpilot.

Constant velocity beats openpilot at 1 s (0.252 against 0.299) and loses from
2 s on, on the mean and the median alike: at 3 s, 1.135 against 0.843, with
medians of 0.816 against 0.643. Scored against the ground truth in the camera's
frame, the baseline had the better median at 3 s; the plans barely changed
between the runs, so that result came from the frame.

Two rows need a caveat, and in both cases the auxiliary columns are what supply
it.

- **drivejepa's 18.754 is a reach failure, not a shape failure.** Its
  `lateral_at_arclength_mae` is 0.150, second only to openpilot's 0.135, while
  `path_length_ratio` is 0.586. It draws close to the right curve and travels
  three fifths of it. Its 4 s plan follows the car's speed up to about 12 m/s
  and then stops growing near 53 m, while I-280 covers about 120 m in 4 s, so
  this is a training-domain speed prior meeting a highway. It is also the
  clearest demonstration in the repo of why this benchmark refuses to report ADE
  alone: ADE puts it an order of magnitude behind every model but METEOR, and
  the two columns that separate speed from shape put its shape second on the
  board. It is the perception-based model of every other benchmark, fed the one
  camera with its three other views black ([docs/models/drivejepa.md](../models/drivejepa.md#on-comma2k19)).
  Until November 2026 this row was Drive-JEPA's perception-free agent, a
  separate registry entry: 16.047, with plans that stopped near 64 m and a less
  well shaped path (0.216).
- **visionpilot has no native trajectory**, so its row grades a synthesised
  pipeline — perception, ground projection, curve fit, cruise-speed assumption
  and extrapolation past a 1 s planner horizon — and is not comparable in kind
  with a model that emits a timed trajectory. Its median (1.513) is well under
  half its mean (3.592), which is the signature of that extrapolation failing on a
  minority of samples rather than a uniform error.

Alpamayo is ahead of both at 1 and 2 s and on the 3 s mean, though only 2.4 cm
ahead of openpilot there (0.819 against 0.843); openpilot has the better 3 s
median (0.643 against 0.737) and the better ADE@4s (1.185 against 1.314). Two
qualifications belong with those numbers.

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
- **It plans 4 s, so the columns past that do not exist for it.**
  `scored_horizon_s` is 4.0 on every sample: 64 waypoints at 10 Hz, of which
  paddock takes eight on the 0.5 s grid. How it would compare with openpilot at
  5 or 10 s is not a question this table answers.

Its plan is also much smoother than supercombo's: `comfort_within_bounds` 0.991
against 0.322, and `comfort_lon_jerk_max` 0.58 m/s³ against 5.05 — see the jerk
result below, which is the same measurement from the other side.

Of the nine, only openpilot and visionpilot publish auxiliary heads through
`Trajectory.extras`; alpamayo15, alpamayo2edge and qwendrive emit a trajectory and nothing else,
and autoe2e's, meteor's and drivejepa's extras are their own diagnostics, so every
metric in the next two paragraphs is `NaN` for them.
That is the intended behaviour of one declared metric list rather than a
per-model one: the run records what the model did not produce instead of quietly
shortening the list.

visionpilot publishes `lead` and deliberately not `lane`. The schema wants four
polylines ordered left to right; AutoSteer emits a single fitted path centreline
sampled per image row, and offsetting it by half a nominal lane to manufacture
two boundaries would make `lane_width_m` report the assumption back as a
measurement. Its lead head is the interesting comparison with supercombo's:
three times worse at range (`lead_range_abs_err` 9.13 m against 3.03) and better
at lateral placement (`lead_lateral_abs_err` 0.369 m against 0.592), which is
what a detect-and-project stack against a regression head should look like.
Neither publishes what the other lacks, so `speed_rel_err`, `yaw_rate_abs_err`
and the sanity gates stay openpilot-only.

The decomposition, for openpilot: `lateral_mae` 0.137 against `longitudinal_mae`
0.807, `along_track_bias` +0.385 (the plan runs slightly *ahead* of the car),
`path_length_ratio` 1.013, `scored_horizon_s` 10.0 on every sample. Almost all
of its error is along the path. Against the ground truth in the camera's frame
`lateral_mae` read 1.092, and most of that was the mount's yaw. The ego-speed
head is accurate — `speed_rel_err` 0.020, `yaw_rate_abs_err` 0.0024 rad/s — so
the error is in the plan, not a misread of how fast the car was going.

Two auxiliary results are worth naming:

- **The uncertainty head is close to calibrated, and slightly narrow.**
  `within_1sigma_y` is 0.612 and `within_1sigma_x` 0.576, where a calibrated σ
  implies ≈0.68. Against the ground truth in the camera's frame
  `within_1sigma_y` read 0.124, which made σ_y look badly under-dispersed; that
  was the mount's yaw showing up as lateral error.
- **Longitudinal jerk leaves the comfort bound on 64% of samples.**
  `comfort_lon_jerk_within_bounds` 0.361, mean peak 5.05 m/s³ against NAVSIM's
  4.13. The other four comfort components are ≥0.97. This is a property of the
  plan resampled onto the 0.5 s grid and matters to anyone treating that grid
  as a controller input rather than as a score.

The lead head is the mixed one: `lead_recall` 0.945 against `lead_precision`
0.593 — radar has a lead on 46% of samples and the model claims one on 73% — with
`lead_range_abs_err` 3.03 m mean, 1.45 m median.

**Against the standalone harness's ~1.6–2.1 m "ADE over 0–10 s" below, this
`ade@10s` of 4.40 m is not a contradiction and not a reproduction.** The two
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

That attempt never landed on `main`, so there is no path here to open: it was
made and discarded on the branch this benchmark came in on, and its run id was
`20260904T011824-9a8260`. The figures are transcribed rather than cited, on 100
segments:

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

The cause was one line in that attempt's loader — an earlier file, not the
`loader.py` this adapter ships, which is a different implementation that landed
with the fix:

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
| split | `driving_vision.onnx` + `driving_policy.onnx` | v0.11.1 (`4df40d2c1`, 2026-05-29) — the comma2k19 row until 6 Nov 2026, now superseded |
| combined | `driving_supercombo.onnx` | `053d9c446` (the pin until this benchmark landed), `555f48c5d` (2026-08-19, the current `repos.yaml` pin — **the comma2k19 row above**), `675ff5698` (2026-09-04) |

The combined graph is the *later* one, and the pin has only ever named a
combined-graph revision: it moved `053d9c446 → 555f48c5d` when this benchmark
landed, a refresh within one layout rather than a change of layout.

What the recorded kitscenes / neuroncap / bench2drive numbers were produced
against is **not recoverable from the record** — those runs predate `repo_sha`
being written, so the honest statement is that they were made while the pin read
`053d9c446`, not that they were made from a clone at it. Both shas ship the
combined graph, so the layout those runs used is not in doubt; the exact
revision is. Runs made from here on record their clone's sha, so this is a
closed gap rather than a standing one.

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
path, for this reason — and the node serves it at `/diagnostics`. The runner
fetches it once per run, after the sweep rather than before (a policy that has
seen no frame cannot answer most of it), and stores the reply under
`provenance.policy` in `run.json` alongside `provenance.repo`. So which layout
and which backend produced a number is now answerable by reading the run, which
it was not for the comma2k19 row the re-run replaced.

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

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Each note below starts with the mark it explains. Marks about a model
rather than this benchmark (∗ ◊ ◇ ✦ ✧) are explained in
[What these models are](../models/comparison.md#notes-on-the-models).

### Notes on the matrix cells

✦ comma2k19 is the full dataset: **2035 one-minute segments, 5 keyframes each —
10,165 scored samples**, not 10,175: two truncated segments are too short to
place a keyframe at all, so it is 2033 × 5 (`{0: 2, 5: 2033}`) — derived, with
both segments named, in [docs/datasets/comma2k19.md](../datasets/comma2k19.md).
Every row is over the identical sample set (`sequences_sha256 =
293a225a0b5cbaff` in all nine `run.json`), and every row was re-run in November
2026 and scored against a ground truth in the car's frame (see [Recorded
results](comma2k19.md#recorded-results)). **Alpamayo 2 Edge leads at every horizon
from 1 s to 5 s, on the mean and the median, on one camera** (0.438 at 3 s): four of its
six slots are black, and its front slot is 65° where it was trained on 120°. Qwen-Drive is
second (0.589), on one camera too: its two side views are black, where it was trained on three
cameras. Alpamayo 1.5 and
OpenPilot follow within 2.4 cm of each other at 3 s, Alpamayo on a **shorter
horizon and one camera** (it plans 4 s where supercombo plans 10, and two of its
four view slots go unsent). Only OpenPilot plans 10 s. **Drive-JEPA's 18.740 is
the same perception-based model as the rest of its row**, momentum-aware with a
0.5 s warm-up (⟲), fed the one camera
with its three other views black, which it never reads; until November 2026
this cell was its perception-free agent, a separate registry entry (16.047). The
number is a speed prior, not a wiring fault: its plans are the second best
*shaped* here (0.150 m lateral error at matched arc length, against OpenPilot's
0.135) but reach only ~58% as far as the car travelled, because its 4 s plan
stops growing near 53 m while I-280 runs at 30 m/s. See
[docs/models/drivejepa.md](../models/drivejepa.md#on-comma2k19) and the
comma2k19 table below. **METEOR's 30.167 is a reach failure too, and a
larger one**: the released graph, shown one phone camera in two of its eight
slots, plans about 17 m ahead whatever the speed, where I-280 needs 55-92 m.

⟲ **Drive-JEPA's comma2k19 figure is momentum-aware, with a 0.5 s warm-up**
(`20261111T074533-0f26ee`). Before each scored keyframe the same node plans the
window's frame 0.5 s earlier and that plan is discarded, so the scored request can
be handed it: 10,163 of 10,165 were (the other two had no warm-up frame). That is a
Paddock choice, not upstream protocol
([Streamed mode](#streamed-mode-drive-jepas-momentum-with-a-warm-up)). It replaces
`20261106T192049-391e2d` (18.754), in which momentum was on and could not act.

### comma2k19

The full dataset: 2035 one-minute segments of 20 Hz California highway driving,
5 keyframes each — **10,165 scored samples**, ten short of 2035 × 5 because two
truncated segments (9.7 s and 11.1 s) place no keyframe at all. Every row over
the identical sample
set. Each sample feeds the model **160 real frames — 7.95 s of history ending at
the scored moment** — and scores the waypoints it plans against the 20 poses the
car actually reached, on a 0.5 s grid out to 10 s. A model is scored only as far
as it plans, so a cell is blank where its own horizon ends rather than being
padded. Lower is better throughout.

| Model | horizon | ADE@1s | ADE@2s | **ADE@3s** | ADE@4s | ADE@5s | ADE@10s | FDE@3s | median ADE@3s | PDMS-style ↑ ⁂ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo 2 Edge | 6 s | **0.086** | **0.225** | **0.438** | **0.713** | **1.043** | — | **1.013** | **0.356** | **0.949** |
| Qwen-Drive 1.0 | 5 s | 0.154 | 0.326 | 0.589 | 0.946 | 1.411 | — | 1.316 | 0.498 | 0.779 |
| Alpamayo 1.5 | 4 s | 0.152 | 0.427 | 0.819 | 1.314 | — | — | 1.864 | 0.737 | 0.690 |
| OpenPilot | 10 s | 0.299 | 0.548 | 0.843 | 1.185 | 1.582 | **4.396** | 1.611 | 0.643 | 0.816 |
| Baseline (Constant Velocity) | 3 s | 0.252 | 0.611 | 1.135 | — | — | — | 2.546 | 0.816 | 0.650 |
| AutoE2E | 6 s | 0.734 | 0.977 | 1.350 | 1.949 | 2.846 | — | 2.396 | 1.339 | 0.766 |
| VisionPilot ∗ | 4 s | 0.964 | 2.126 | 3.592 | 5.287 | — | — | 7.415 | 1.513 | 0.676 |
| VAD | 3 s | 5.759 | 9.682 | 13.658 | — | — | — | 23.621 | 16.093 | 0.781 |
| UniAD | 3 s | 6.242 | 10.538 | 14.917 | — | — | — | 25.928 | 18.008 | 0.679 |
| Drive-JEPA ⟲ | 4 s | 7.469 | 13.070 | 18.740 | 24.368 | — | — | 32.869 | 23.401 | 0.622 |
| SimLingo | 2.5 s | 7.337 | 12.749 | 19.320 | — | — | — | 37.847 | 21.478 | 0.355 |
| METEOR | 3 s | 11.841 | 21.621 | 30.167 | — | — | — | 51.507 | 37.395 | 0.034 |

**SimLingo plans far too short for the highway** (`20261119T140738-46b00e`). Trained only in CARLA, it
has never seen real camera footage or I-280's speeds: its plan covers 29.2 m on average where the
recorded drive covers 65.9 m, an along-track bias of -18.8 m at 3 s, so nearly all of its ADE@3s
of 19.320 is speed (across-track 1.310 m). Its 2.5 s plan is padded with its last position to the
3 s scored (the P mark). It gets the road camera at its true scale in SimLingo's 110 deg frame,
the rest black ([docs/models/simlingo.md](../models/simlingo.md#comma2k19)).

**VAD and UniAD plan no faster than about 18 m/s,** the top of the nuScenes city
driving their released checkpoints were trained on, while two thirds of these
keyframes (6,689 of 10,165) average 20 m/s or more. Below 20 m/s their median
ADE@3s is 1.36 m (VAD) and 1.64 m (UniAD). Each gets the road camera as
CAM_FRONT and its five other slots black.

**OpenPilot and the baseline separate after the first second.** Constant
velocity is better at 1 s (0.252 against 0.299), because I-280 is mostly
straight and a straight line at the current speed is then almost exactly right;
from 2 s on OpenPilot is better on the mean and the median, and the gap grows
with every horizon they share (1.135 against 0.843 at 3 s). The baseline's row
stops at 3 s because the registered policy emits the six waypoints neuroNCAP
requires.
[`tools/benchmarks/comma2k19/constant_velocity.py`](../../tools/benchmarks/comma2k19/constant_velocity.py)
extends the same arithmetic to 10 s, but its figures (2.411 / 3.331 / 10.048 at
4, 5 and 10 s) predate the ground truth in the car's frame, so they are not
set against OpenPilot's 4.396 at 10 s here.

**Alpamayo 2 Edge plans 64 points 0.1 s apart.** Its ADE and FDE are scored on its
own points at every 0.5 s, as every other row's are; nothing is interpolated. Its
comfort terms are on the same 0.5 s grid as every other row's: the shared comfort
code (`paddock/metrics/comfort.py`) moves any plan finer than 0.5 s onto that grid
before measuring, and for a 0.1 s plan those are its own points. Only the
lead-vehicle check reads the whole 0.1 s plan. Four of its
six slots are black (`ALPAMAYO2_EDGE_ABSENT_CAMERAS=black`, which the leaderboard
notes under the run). Run `20261112T012152-9b1e09`.

**Edge, Qwen-Drive and Alpamayo lead on less input than OpenPilot.** Both see one
camera where they expect several view slots — comma2k19 is a single forward
camera, so Alpamayo's +55°/−55° slots go unsent and Qwen-Drive's side views are
black — and both plan well short of supercombo's 10 s. Qwen-Drive is ahead of
every other model from 2 s to its 5 s horizon, on the mean and the median (0.589
and 0.498 at 3 s); Alpamayo is ahead at 1 s, by 3 mm. Alpamayo and OpenPilot are
within 2.4 cm of each other at 3 s, Alpamayo better on the mean and OpenPilot on
the median and at 4 s. What the table cannot say is how Qwen-Drive or Alpamayo
would extrapolate to 10 s, because neither goes there.

**Drive-JEPA's row is a reach failure, not a shape failure, and the auxiliary
columns are what show the difference.** It has the second-lowest lateral error
at matched arc length on this benchmark (0.150 m, against OpenPilot's 0.135)
while travelling only 58% as far as the car did (`path_length_ratio` 0.584), so
it draws close to the right curve and stops two fifths short of where the car
went. Its 4 s plan follows the car's speed up to about 12 m/s and stops growing
near 53 m; I-280 covers about 120 m in 4 s. Removing the speed error is exactly
what `lateral_at_arclength_mae` does, and it takes 18.740 to 0.150 — which is
not a number this table will headline, because the model was not asked for a
shape, it was asked for where the car would be. This is the clearest case in
the repo for why a benchmark that reported ADE alone would be describing the
wrong defect. The perception-free agent that held this row until November 2026
scored 16.047: its plans stopped further out, near 64 m, and its path was less
well shaped (0.216 m).

**METEOR's row is the released graph on a camera it never saw.** METEOR is an
eight-camera network; comma2k19 has one phone camera, so the adapter puts it
in two slots (the full frame, and its centred 30° window where METEOR's
telephoto goes) and leaves the other six black with the calibration of
METEOR's own rig, the way its training drops a camera. Fed that way on
METEOR's own recordings, the front camera alone beats constant velocity (1.01
against 1.37, [docs/models/meteor.md](../models/meteor.md)). Here it plans
about 17 m ahead at every speed above 15 m/s, where the car went 55-92 m:
`path_length_ratio` 0.321, the shortest reach on the board. Below 15 m/s its
ADE@3s is 3.84, and every sample where it beats constant velocity (3.5 % of
them) is there. An out-of-tree study of one comma2k19 segment found the model
placing the road 3.4x too far on this camera and seeing road only about 20 m
ahead in its bird's-eye view, which is where its plans end.

comma2k19 also carries the car's own CAN bus, radar and fused pose, so it scores
what a model predicts **besides** the path — one metric list, `NaN` for a model
that publishes no such head. Two of the eight do. Over the same 10,165 samples:

| what | OpenPilot | VisionPilot | reference it is scored against |
|---|---:|---:|---|
| ego speed, relative error | 0.020 | — | CAN speed |
| yaw rate, absolute error (rad/s) | 0.0024 | — | fused INS/GNSS pose |
| lead recall / precision | **0.945** / 0.593 | 0.871 / **0.752** | radar tracks |
| lead range, absolute error (m) | **3.03** (median 1.45) | 9.13 (median 6.87) | radar range |
| lead lateral, absolute error (m) | 0.592 | **0.369** | radar track position |
| lead closing speed, absolute error (m/s) | **0.938** | 1.882 | radar range rate |
| plan within 1σ, along / across track | 0.576 / 0.612 | — | the recorded future |
| longitudinal jerk within comfort bound | 0.361 | **0.894** | NAVSIM's bounds |
| sanity gates passed (lane width, camera height, speed) | 0.926 | — | plausibility ranges |

Two of those are results rather than confirmations. OpenPilot's **longitudinal
jerk exceeds the comfort bound on 64% of samples** (mean peak 5.05 m/s³ against
a 4.13 bound), which is a property of the 33-point plan resampled onto a 0.5 s
grid and is worth reading before anyone treats that grid as a controller input.
And VisionPilot's lead head, which is three times worse than supercombo's at
*range*, is the better of the two at **lateral** placement (0.369 m vs 0.592):
it is solving a detection-and-projection problem where supercombo is solving a
regression, and the two fail in different directions. OpenPilot's uncertainty
head looked badly under-dispersed against the ground truth in the camera's
frame, with 12% of outcomes inside 1σ; against the car's frame it holds 61%,
close to the 68% a calibrated 1σ holds. Alpamayo, Qwen-Drive and Drive-JEPA
publish none of these heads through paddock's extras schema, so their columns
are `NaN` throughout — which is the point of one declared metric list rather
than a per-model one. The baseline's comfort columns are all 1.0 by
construction: a straight line at constant speed has no jerk.

⁂ **PDMS-style** uses the progress and comfort rules adopted on 22 Oct 2026 (see
[Recorded results](#recorded-results)). It is NAVSIM v1's published composite,
`NC × DAC × (5·TTC + 5·EP + 2·C) / 12`, evaluated over log-replay *proxies* —
comma2k19 has no other-agent tracks and no map, so no-collision sees only the
closest radar lead and drivable-area uses the human's own path as the corridor.
It is deliberately not the primary metric and never called a driving score; the
five sub-scores are reported beside it in `report.md` so nobody has to trust the
sum. Qwen-Drive was `—` here until it was re-swept on the common 0.5 s grid:
its comfort term had been measured on a 0.1 s one, where `1/dt³` turns 4 cm of
output quantisation into 1932 m/s³ of "jerk".
See [docs/benchmarks/comma2k19.md](comma2k19.md) for the full
derivation and what it still cannot see.

[OpenPilot](../../results/comma2k19/openpilot/20261106T003719-5cde54/metrics.json)
· [Alpamayo 1.5](../../results/comma2k19/alpamayo15/20261106T005600-6c52a2/metrics.json)
· [Alpamayo 2 Edge](../../results/comma2k19/alpamayo2edge/20261112T012152-9b1e09/metrics.json)
· [Qwen-Drive 1.0](../../results/comma2k19/qwendrive/20261106T030530-7d9fd3/metrics.json)
· [Drive-JEPA](../../results/comma2k19/drivejepa/20261111T074533-0f26ee/metrics.json)
· [VisionPilot](../../results/comma2k19/visionpilot/20261106T003749-c10a7f/metrics.json)
· [METEOR](../../results/comma2k19/meteor/20261106T064120-226c00/metrics.json)
· [AutoE2E](../../results/comma2k19/autoe2e/20261106T073744-1b4fe1/metrics.json)
· [baseline](../../results/comma2k19/constant_velocity/20261106T045541-d0cdb5/metrics.json)
· [protocol, and what it cannot score](comma2k19.md)
· [the dataset itself](../datasets/comma2k19.md).
