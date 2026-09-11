# comma2k19

Open-loop trajectory accuracy against comma2k19's own recorded
(INS/GNSS/vision-fused) future pose. 33 hours of 20 Hz highway driving on
California's I-280, 2035 one-minute segments, two vehicles.

```bash
paddock run --bench comma2k19 --model openpilot --limit 5     # smoke
paddock run --bench comma2k19 --model openpilot               # full, n=2035
paddock render --bench comma2k19                              # comparison figures
```

## Setup

```toml
[datasets.comma2k19]
# raw_data/, not the sibling data/ (HF demo parquet) or compression_challenge/
path = "/path/to/comma2k19/raw_data"

[benchmarks.comma2k19]
# Optional. Defaults shown; all three change what is measured and are hashed
# into the run id.
history_frames = 160          # observations fed before the scored one
keyframes_per_segment = 1
input_hz = 20                 # observation rate; see "Evaluation rate" below
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

There is deliberately **no composite driving score**. A single number here would
have to weight an ADE against a comfort rate against a TTC, and no such weighting
is published for this dataset; inventing one would make the benchmark's headline
an opinion of paddock's rather than a measurement. The components are reported
separately and combine however a reader's application requires.

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

## A pin discrepancy to be aware of

This adapter and the openpilot ONNX backend were developed and verified against
openpilot **`675ff5698`**. `repos.yaml` pins **`555f48c5d`**, which is what the
recorded kitscenes / neuroncap / bench2drive runs were produced against.

The pin is deliberately left alone: moving it would silently make those existing
numbers incomparable, which is a worse problem than this one. Instead,
`supercombo.SupercomboSession` checks the graph's declared inputs on load and
raises with the mismatch named if the pinned revision's supercombo is not the
`{img, big_img, features_buffer, desire_pulse, traffic_convention, action_t}`
graph it was written for. Producing a comma2k19 number therefore requires either
pinning the verified revision — and re-running whatever else that pin affects —
or using the tinygrad backend, which tracks whatever the checkout ships.

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
