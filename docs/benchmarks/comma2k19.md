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
