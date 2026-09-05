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
# Optional. Defaults shown; both change what is measured and are hashed into
# the run id.
history_frames = 160          # real preceding frames fed before the scored one
keyframes_per_segment = 1
```

Frame decoding needs either **PyAV** in paddock's environment
(`pip install av`) or an ffmpeg binary via `PADDOCK_FFMPEG`. `preflight`
reports which, and refuses to run without one.

Getting the dataset: [`scripts/download_comma2k19.py`](../../../scripts/README.md)
downloads and verifies it resumably.

## What is scored

One keyframe per segment, with the real preceding frames fed through
`Policy.infer_sequence`, scored against 5 s of recorded future on a 0.5 s grid.

`history_frames` defaults to **160** because the queues hold
`frame_skip * feature_depth` frames of history -- 4 x 24 = 96 for the small
model but 4 x 32 = **128** for the big one -- and one default has to clear both,
with a margin.

It is tempting to justify a larger number by a settling transient after the
buffer fills, and one segment's error trace does look that way. Aggregated over
24 segments it does not: cross-track error at 3 s is 0.20 m in the first 20
frames past the 96-frame boundary and 0.27 m thereafter, already at its floor.
The single-segment impression was noise, and
`experiments/op_c2k19/tests/gate_g5_g6_run.py` is what settled it.

There is no shared external keyframe list the way KITScenes has one, so
`keyframe_indices` generates it from the segment length — deterministic, so
`--limit N` selects the same work at every site, but a property of this
benchmark's code rather than an external artefact.

### For dense evaluation, use the reference implementation

`experiments/op_c2k19` scores **every frame of every segment** — 240k frames
over 10 % of the dataset in about five minutes — because it drives the model
directly instead of through a per-sample protocol. This adapter exists to put
comma2k19 on the same footing as paddock's other open-loop benchmarks: one
comparable number per segment, any model, no openpilot-specific assumptions.

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
| ade | 38.84 |
| fde@1s | 22.29 |
| fde@2s | 44.43 |
| fde@3s | 66.36 |
| longitudinal_mae | 38.82 |
| lateral_mae | 0.96 |

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

From the reference implementation, scoring **every frame** of a 10 % sample
(220 segments, ~241k frames) — see
[`experiments/op_c2k19/RESULTS.md`](../../../experiments/op_c2k19/RESULTS.md)
for the authoritative tables and the gates that back them:

| | small | big |
|---|---:|---:|
| ego-speed error vs recorded | ~2 % mean, ~0.5 % median | ~1.9 % / 0.3 % |
| ADE over 0–10 s | ~2.1 m | ~1.6 m |
| cross-track at 3 s | ~0.27 m | ~0.24 m |

These are per-frame figures over the model's full 10 s horizon and are **not**
directly comparable to this adapter's per-sample 3 s ADE; they are quoted to
establish the scale of the correction, not as this benchmark's numbers.

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
