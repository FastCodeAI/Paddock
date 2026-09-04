# comma2k19

Open-loop trajectory accuracy: ADE and FDE against comma2k19's own recorded
(INS/GNSS/vision-fused) future pose, one keyframe per raw segment.

```bash
paddock run --bench comma2k19 --model openpilot --limit 5    # smoke
paddock run --bench comma2k19 --model openpilot              # full n=2035
```

## Setup

```toml
[datasets.comma2k19]
# raw_data/, not the sibling data/ (HF demo parquet) or compression_challenge/
path = "/path/to/comma2k19/raw_data"
```

Frame decoding needs ffmpeg, which paddock's own environment does not bundle
(unlike each model's env, which is free to). Set `PADDOCK_FFMPEG` if none is on
`PATH` -- reusing the one `comma-deps-ffmpeg` installs into openpilot's own venv
is a reasonable default, since it is already there for the model's RGB->NV12
step:

```bash
export PADDOCK_FFMPEG=/path/to/repos/openpilot/.venv/bin/ffmpeg
```

## There is no shared keyframe list, unlike KITScenes

KITScenes' 223 keyframes come from a file every site downloads identically.
comma2k19 has no such published list, so this benchmark generates its own: one
keyframe per segment (`loader.choose_keyframe`, fixed at frame 600 of a nominal
~1200-frame segment, falling back toward the end only for unusually short
segments, and skipping a segment outright if even that has no room for both the
history window and the scoring horizon). Deterministic, so `--limit N` selects
the same N segments at every site -- but the set itself is a property of this
benchmark's code, not an external artifact, and a future change to
`choose_keyframe` changes what every site scores. `coverage()` reports the
count; there is no "missing" case the way KITScenes has, since every segment
`list_segments` finds under the configured root is usable by definition.

## Ground truth needs no post-processing, unlike KITScenes

Every segment ships `global_pose/frame_{positions,orientations,velocities,times}`
-- ECEF position and orientation already fused from INS/GNSS/vision, one row
per video frame. There is no SLAM or GNSS pipeline to run; `loader.ground_truth`
only has to rotate ECEF position deltas into the keyframe's own device frame.

That rotation direction is the one place this is easy to get backwards, and the
dataset's own README states it backwards (it says the quaternion rotates ECEF
into the device frame; comma2k19's own `utils/camera.py::device_from_ecef`
transposes it before use, which is the tell). `loader._quat2rot` returns
`ecef_from_device`; converting a delta into device coordinates is
`delta @ ecef_from_device`, not `ecef_from_device @ delta`. Getting this wrong
does not fail loudly -- it mirrors every relative position through the
vehicle's direction of travel, which roughly doubles ADE without an error
anywhere. `tests/test_harness.py::test_ground_truth_relative_to_a_rotated_keyframe`
exists specifically to catch a regression here: a plain identity-orientation
fixture cannot (both directions agree when there is no rotation to get wrong).

## Frames need decoding, unlike KITScenes

`video.hevc` is a raw HEVC elementary stream, not per-frame JPEGs -- there is
exactly one file for an entire ~60 s segment. `loader.ensure_frames` decodes it
once per segment into a JPEG-per-frame cache under `{runs}/comma2k19_frame_cache/`,
then observations point `image_paths` at that cache exactly the way KITScenes
points at its own per-frame JPEGs. The whole segment is decoded rather than just
the ~97-frame window one sample needs: the vendored `comma-deps-ffmpeg` build
most sites will have on hand for openpilot's own NV12 conversion is compiled
with a narrow filter allow list for that one job and does not include `select`,
which a windowed decode would need. Cached, so a second run -- or a second
model scored on the same segment -- decodes nothing.

## Rate

comma2k19's camera runs at 20 Hz natively, which is supercombo's own believed
input rate exactly (`MODEL_HZ` in the openpilot adapter). Unlike KITScenes
(10 Hz, each frame emitted twice to keep the model's sense of speed honest),
no frame-repeat trick is needed here at all -- every observation in a sample's
history is one real 1/20 s step, and the model perceives the actual recorded
motion.

## Samples are sequences

Each sample is 96 preceding source frames plus the keyframe -- enough to fill
supercombo's feature buffer with genuine motion at its own native rate, the
comma2k19 equivalent of KITScenes' 48-at-10-Hz-doubled-to-96-effective history.
Fed through `Policy.infer_sequence`, same as every other open-loop benchmark.

## Splits

`rav4` (chunks 1-2, dongle `b0c9d2329ad1606b`) and `civic` (chunks 3-10, dongle
`99c94dc769b5d96e`) -- the two vehicles the dataset was collected with, not a
train/val/test split. `samples(split=...)` supports selecting one, but the CLI
does not currently expose a `--split` flag for open-loop runs (`run_benchmark`
never passes one through), so `--limit N` today draws from the front of the
sorted (chunk, route_id, segment) order regardless -- mostly `rav4` for small
N, since chunks 1-2 sort first.

## Reproduction status

`--limit 100` (all `rav4`, chunks 1-2 -- see Splits above), `results/comma2k19/openpilot/20260904T011824-9a8260/`:

| metric | mean | median | p90 | max |
|---|---:|---:|---:|---:|
| ade | 38.84 | 47.31 | 55.77 | 60.92 |
| fde@1s | 22.29 | 27.18 | 31.93 | 34.91 |
| fde@2s | 44.43 | 54.11 | 63.76 | 69.65 |
| fde@3s | 66.36 | 80.75 | 95.49 | 104.22 |
| longitudinal_mae | 38.82 | 47.27 | 55.68 | 60.91 |
| lateral_mae | 0.96 | 0.70 | 2.66 | 5.98 |

**This is a real, large ADE, and it is almost entirely longitudinal.**
`lateral_mae` (<1 m on average) is in the same range as the KITScenes numbers
above; `longitudinal_mae` alone accounts for essentially all of `ade`. Ruled
out before concluding this is a genuine finding rather than an adapter bug:

- **Camera FOV margin** -- EON's real 65.2°x51.3° comfortably covers
  supercombo's required 58.7°x31.3° (a slightly larger margin than KITScenes'
  own working setup has), so the crop/warp is not starved of field of view.
- **Frame rate / `repeats`** -- comma2k19 is native 20 Hz with a uniform
  1/20 s step between every history frame in a sample, so `OpenpilotPolicy`'s
  `repeats` calculation comes out to exactly 1 throughout -- there is no
  hidden 4x (or any) rate mismatch being fed to the model.
- **Ground truth** -- validated directly against real segment data (11 m of
  forward travel per 0.5 s step against a recorded ~22 m/s speed, sub-metre
  lateral drift, near-zero yaw -- exactly right for a car holding lane on a
  straight highway) before any model was ever involved; see
  `tests/test_harness.py` for the synthetic version of the same check.
- **`canbus` has no path to the visual plan** -- the six ONNX inputs are
  `{img, big_img, features_buffer, desire_pulse, traffic_convention,
  action_t}`; there is no speed/`vEgo` input at all, so a difference in how
  `canbus[13]` is computed (comma2k19 uses the dataset's own recorded
  velocity; KITScenes finite-differences positions) cannot be why the plan's
  perceived speed is wrong.

What's left, and the most likely explanation: **comma2k19's EON camera
(1164x874, a 2018-era phone-grade sensor) is being upscaled 66% to fill
supercombo's 1928x1208 input**, whereas KITScenes' ring camera (3504x2272) is
being *downscaled* into the same target -- a much gentler transformation.
supercombo was trained on modern tici-camera footage; fed a materially
blurrier, lower-detail, highway-speed (~22 m/s / ~80 km/h) view through a
camera two hardware generations older than what it expects, it appears to
substantially underestimate ego speed while still tracking lane geometry
(a largely single-frame, resolution-tolerant cue) just fine. That reads as a
genuine cross-hardware generalization gap in the model, not a defect in this
adapter -- but it is only a 100-sample, single-split (`rav4`) result; a full
`civic`-inclusive run would confirm whether it holds outside chunk 1-2.
