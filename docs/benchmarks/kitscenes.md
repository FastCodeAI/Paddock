# KITScenes

Open-loop trajectory accuracy: ADE and FDE against the recorded future, over a
fixed set of 223 keyframes.

```bash
paddock run --bench kitscenes --model openpilot --limit 5   # smoke
paddock run --bench kitscenes --model openpilot             # full n=223
```

## The sample list is the benchmark

All models are scored on one shared keyframe list, configured per site:

```toml
[benchmarks.kitscenes]
sample_list = "/path/to/kitscenes_e2e_final.pkl"
```

Benchmark-level, not model-level, deliberately: two models scored on different
keyframes are not comparable however close the counts look. `paddock run
--dry-run` reports how many of the list this site's tiers can serve, and
`samples()` refuses to run a partial set rather than silently scoring fewer.

That file is not a hand-written list. It is the output of the Drive-JEPA run
that went first, kept because whichever 223 keyframes it happened to score
became the set every later model had to match. Reading it as a sample list uses
`scene_id`, `keyframe_index` and `split` and ignores the predictions sitting
beside them; `paddock ingest` reads the same file for the predictions, which is
how Drive-JEPA has a row on this benchmark without a paddock run.

### The seven rows, and where each came from

| model | run | n | ADE | source |
|---|---|---:|---:|---|
| qwendrive | `20260914T220841-5a91fd` | 223 | 0.806 | a paddock run; three of the six ring cameras |
| alpamayo15 | `legacy-full-nonav` | 223 | 0.921 | `eval_kitscenes/runs/full` |
| visionpilot | `20260909T202724-153b04` | 223 | 1.534 | a paddock run; see below |
| openpilot | `legacy-full223` | 223 | 1.599 | `eval_kitscenes/runs/full223` |
| meteor | `20260914T215457-ab0586` | 223 | 1.879 | a paddock run; six of its eight cameras |
| drivejepa | `legacy-kitscenes_e2e_final` | 223 | 2.364 | `inference/kitscenes_e2e_final.pkl` |
| drivejepa_rerank | `20260918T220628-d7ed08` | 223 | 0.973 | a paddock run; **adapted**, trained on KITScenes train, see below |

All seven carry `sequences_sha256: 80665529bea7f500` — the same 223 keyframes,
so the column compares. Caveats the hash cannot express:

- **Route hint.** Drive-JEPA was run with `--command gt`, which derives a turn
  command from the recorded future on 39 of the 223 samples; openpilot and
  VisionPilot have no such input, and Alpamayo's equivalent is its `nav` arm.
  Recorded as `command_source` in that run's `config`.
- **Domain.** The checkpoint is trained on nuPlan/OpenScene — US cities, a
  windshield rig — and KITScenes is European driving from a roof pod. Its rank
  here is a zero-shot transfer result, which is why it sits last on this
  benchmark and first on NAVSIM. `adapters/benchmarks/navsim/run_kitscenes_inference.py`
  sets out what the adapter can and cannot reconcile.
- **Camera geometry, and no route hint at all.** Qwen-Drive is the only row read
  on three cameras: the ring's front, front-left and front-right, at
  +60.9 / +0.97 / −58.5 degrees and ~87° horizontally, against the ~70° nuPlan
  rig it trained on. It is zero-shot here. The benchmark supplies no navigation
  command, so every one of its samples is driven as "go straight" — a channel
  the model uses on NAVSIM is inert on this one.

- **Adapted, not zero-shot.** `drivejepa_rerank` is Drive-JEPA with a selector
  choosing among its 32 candidates. The selector was trained on 5,711 keyframes
  from 373 KITScenes **train** scenes. None of those scenes is in this list,
  which was checked by scene id. It is 1.002 on the 186 val samples and 0.828 on
  the 37 overlap_train_val ones, whose area overlaps train. It is the first
  paddock run of Drive-JEPA on this benchmark. Before it, the harness sent no
  `lidar2ego`, and Drive-JEPA refuses an observation without one. It reads the
  ring cropped onto the OpenScene lens (`DRIVEJEPA_IMAGE_GEOMETRY=openscene`),
  with no route hint. Through that path Drive-JEPA's own pick scores 2.521; the
  2.364 row's devkit loader also sent acceleration and the route hint. See
  [drivejepa_rerank.md](../models/drivejepa_rerank.md).

Nearly all of Drive-JEPA's error is longitudinal (2.317 m against 0.209 m
lateral, an 11:1 split, from the original report, which decomposed it against
Alpamayo only): it steers better than Alpamayo and predicts speed far worse,
so its last place here is a speed result rather than a path one.
## VisionPilot has no trajectory of its own

VisionPilot is an L2 assist stack. Its three networks return a path drawn in
*image* space, a distance to the closest in-path object, and boxes tagged with a
threat class. No metres, no timing. The adapter builds the trajectory by porting
the stack's own planner: project the AutoSteer points onto the road, fit a
curve, then walk an IDM speed schedule along it. So an ADE here scores
perception, projection, the curve fit, the speed assumption and extrapolation
together, and it is not comparable in kind with a model that emits a timed
trajectory directly.

Two things about this dataset broke that pipeline, and both are worth knowing
before reading the number.

**The ego frame is not on the road.** KITScenes calibrates every sensor against
`base_frame`, which is the roof pod. The front ring camera sits at z = -0.183 m
in it. Ground projection assumes z = 0 *is* the road -- true for neuroNCAP,
which zeroes the ego pose's z, and for CARLA, which mounts the camera 1.5 m up.
A negative camera height flips the sign of the ray-plane scale, so every
projected point lands *behind* the car: measured on the first three samples, 0
of 32, 0 of 29 and 0 of 28 AutoSteer points came out ahead of the ego, and since
the furthest sat at -0.09 m every waypoint fell past the fitted extent and came
from the straight-line tangent rule. The lateral path was a perfectly straight
line, identical step to step to seven decimals.

`ground_offset_m` (default 2.2, `POD_HEIGHT_M`) raises the origin to the road.
The same points then span 25-46 m ahead. Only the translation moves, so a model
reading just the rotation is untouched -- openpilot's three-sample ADE is
1.591566 against its recorded 1.591566, unchanged.

**The cruise set-speed was latched 4.8 s early.** KITScenes supplies no posted
limit, so VisionPilot falls back to `max(speed, 1.0)` at the first frame it is
shown -- which `infer_sequence` makes the *start of the history*, not the frame
being scored. That pinned 39 of 223 samples to a 1 m/s crawl while the ego was
moving. `observe()` now warms the filters without fixing the set-point, moving
the latch to the scored frame; 12 samples remain at the floor, and those are
cars genuinely stopped at the keyframe. This uses the speed the car has at
prediction time, not the recorded future -- the driver's speed changes 1.36 m/s
on average over the scored 3 s, so it stays a prediction.

Together these took the three-sample ADE from 5.486 to 1.996.

## Samples are sequences

Each sample is 48 preceding source frames plus the keyframe, not a single image.
The models need real history — openpilot fills a 24-deep feature buffer, Alpamayo
differences 16 ego poses to derive its speed — and handing them one frame would
measure a model working from a history it invented.

`Policy.infer_sequence` feeds the sequence and scores the last frame, disabling
self-priming for its duration. A model that primes *and* receives real history
sees a manufactured past followed by a real one.

## The camera ring

KITScenes records a six-camera ring and calibrates each camera against the rig
reference. Every observation the harness builds carries all six -- a frame path
plus that camera's own intrinsics and extrinsics -- and the node keeps the
cameras its policy declares (`Observation.from_wire`) and opens the ones it
reads (`Observation.image`). A model declaring four cameras is scored on four,
and a model declaring one pays nothing for the other five.

Until this change the harness handed every model `camera_ring_front` alone, as
`CAM_FRONT`. openpilot and VisionPilot read one camera, so their rows measure
what they claim; Drive-JEPA declares four, Alpamayo three and METEOR six, and a
paddock run of any of them was scored with most of its inputs absent. The
Drive-JEPA and Alpamayo rows in the table above were ingested from their own
pre-paddock loaders, which read the ring themselves, so they are unaffected.

| dataset sensor | protocol camera | looks |
|---|---|---|
| `camera_ring_front` | `CAM_FRONT` | forward |
| `camera_ring_front_left` | `CAM_FRONT_LEFT` | 60 deg left |
| `camera_ring_front_right` | `CAM_FRONT_RIGHT` | 60 deg right |
| `camera_ring_rear` | `CAM_BACK` | backward |
| `camera_ring_rear_left` | `CAM_BACK_LEFT` | 120 deg left |
| `camera_ring_rear_right` | `CAM_BACK_RIGHT` | 120 deg right |

The protocol's names are nuScenes', which every adapter's camera handling is
written against; the dataset's `rear` becomes `BACK` for that reason and no
other. `RING` in `harness.py` is the mapping, and its test pins it to
`protocol.CAMERAS` in the protocol's order.

**Calibration.** `calib.json` gives each camera its own pinhole -- on the scene
checked, focal lengths 1841-1847 px and principal points up to 29 px apart --
and a `T_to_reference` from the camera's OpenCV optical frame, x right, y down,
z forward, into the rig reference, x forward, y left, z up. That reading was
established on the front camera and then *checked* on the other five rather
than assumed: ego-frame ground points 10 m ahead, behind, left and right,
projected through each camera's K and inv(cam2ego) on scene `00fc1b9b`, land
where the ring's geometry says they must (u, v as fractions of width and
height):

| camera | 10 m ahead | 10 m behind | 10 m left | 10 m right |
|---|---|---|---|---|
| front | u 0.51, v 0.67 | behind camera | behind camera | behind camera |
| front_left | off image, u 1.49 | behind camera | u 0.20, v 0.68 | behind camera |
| front_right | off image, u -0.40 | behind camera | behind camera | u 0.83, v 0.70 |
| rear | behind camera | u 0.51, v 0.67 | behind camera | behind camera |
| rear_left | behind camera | off image, u -0.40 | u 0.83, v 0.69 | behind camera |
| rear_right | behind camera | off image, u 1.52 | behind camera | u 0.21, v 0.69 |

Front and rear put their point in the lower middle; each side camera, looking
60 deg out, sees its side's point 30 deg off-axis, a fifth of the way in from
the edge; every point that should be off the image or behind the camera is.
v is 0.67-0.70 throughout, which is what a camera 2.0 m above the road sees of
a ground point 10 m away. A wrong axis convention on any one camera would have
put its point above the horizon or in the wrong half; none did.

**The ground offset is the rig's.** `ground_offset_m` (`POD_HEIGHT_M`, 2.2) is
added to every camera's z translation, not only the front's: the six sit within
4 mm of one another in the pod frame (z from -0.1826 to -0.1864 m), so after
the shift each is 2.01-2.02 m above the road, and the z = 0 that every
ground-plane projection assumes is the same road for all six.

**Provenance.** `run.json` gains `config.cameras_supplied`: the ring cameras
the model declares, in the protocol's order, hashed into the run id alongside
`config.cameras` (what the model declares in full). The declared list did not
change with the ring and `run.json` carries no paddock revision, so without
this a ring run and a front-only run of the same model would share a run id
and differ by timestamp alone. A record without the key predates the ring and
was scored on `CAM_FRONT`.

**Tiers, per camera.** A working set is pruned by sensor -- `docs/data-setup.md`
invites exactly that -- so the harness resolves each ring camera against the
tiers independently: a camera's frames come from the first tier whose copy of
the scene has that camera's directory, while `poses.txt` and `calib.json` come
from the first tier holding the scene. At this site the NVMe working set holds
the front camera for all 127 of its listed scenes and the rear-right for 53;
the archive behind it holds all six for all 127; the 13 scenes on the local
disk hold all six. Resolving the scene once, as before, would have pointed
five cameras at files one tier further down.

**What is checked, and what is assumed.** Frame *i* of every camera is treated
as the sample taken at pose *i*. That is how the dataset is written -- one frame
index across every sensor of a scene, named `{index:010d}.jpg`, against one
reference clock -- but the scenes on disk here carry neither
`timestamp.reference.txt` nor the per-sensor `timestamp.<sensor>.txt` files the
dataset package reads for its asynchronous layout, so how far any camera's
exposure sits from its pose is not something the harness can measure. It is
assumed. What can be checked is checked: `preflight()` walks every listed
scene and refuses if a ring camera is missing from every tier or if a camera's
frame count differs from the pose count. Over the 140 scenes the sample list
names, at this site: 840 of 840 camera directories present across the union of
tiers, 840 of 840 holding exactly one frame per pose with indices contiguous
from 0. The walk takes about 0.3 s.

**An incomplete ring refuses rather than degrades.** The harness cannot see
which model is paired -- `samples()` is built before the model is -- so it
cannot tell a gap that costs nothing (openpilot, rear camera missing) from one
that silently changes what a six-camera model was scored on. Between refusing
a run that would have been fine and recording a number that is wrong with
nothing to show it, it refuses, naming the scenes and the cameras; the
sample-list hash would not have caught it, since the keyframes are unchanged.
The fix is a tier that holds the ring, declared last: the archive the working
set was pruned from. A site that genuinely holds only the front camera cannot
run this benchmark for any model until it does, and that is the intended
trade.

## Camera adaptation

openpilot's ModelState compiles its warp for fixed geometries, so a source must
become 1928x1208 exactly. Which way depends on the source's aspect, and the
adapter derives it rather than hardcoding one:

| source | aspect | strategy | why |
|---|---:|---|---|
| KITScenes 3504x2272 | 1.542 | **crop** | taller than the target, so trimming height costs no horizontal field of view |
| nuScenes render 1600x900 | 1.778 | **letterbox** | wider, so cropping would trim width — and hFOV is scarce, the wide model view wants 58.7 deg of about 64 available |

Before paddock these were two separate classes, one per loop type, each
hardcoding one strategy — which is why the open-loop path could not run a
nuScenes frame and the closed-loop path could not run a KITScenes one.

## Reproduction status

Verified against the recorded `runs/full223` results, same keyframes:

| scene / keyframe | paddock ADE | recorded ADE |
|---|---:|---:|
| 00fc1b9b / 40 | 1.0783930800368695 | 1.0783930800368695 |
| 02be7d74 / 40 | 0.8611… | identical |
| 03ed2585 / 40 | 2.8353… | identical |

The **full n=223 run reproduces every metric exactly**:

| metric | exact / 223 | paddock | recorded |
|---|---:|---:|---:|
| ade | **223/223** | 1.599424 | 1.599424 |
| fde@1s | **223/223** | 0.836421 | 0.836421 |
| fde@2s | **223/223** | 1.808482 | 1.808482 |
| fde@3s | **223/223** | 2.893613 | 2.893613 |

892 of 892 metric comparisons identical at full float precision; ADE median
1.322747 against 1.322747. Roughly 85 minutes on this hardware.


## A bug worth knowing about

Reproducing the full n=223 found something a three-sample check could not: the
openpilot adapter cached its `CameraAdapter` across samples. It is built from
the first frame's calibration, and `reset()` did not clear it, so every scene
after the first was warped with the **first scene's mounting rotation**.

KITScenes rigs are recalibrated between sessions. Across 60 `val` scenes there
are three distinct front-camera rotations, up to 1 degree apart — so roughly a
quarter of samples were fed a subtly wrong view. Intrinsics are identical
throughout, which is why this survived the obvious checks.

What made it hard to see:

- **Nothing failed.** A degree of rotation shifts the warp; it does not raise.
- **The three samples used for the initial check all shared one rotation
  group**, so they matched bit-for-bit and looked like proof.
- The error only showed as a slowly growing gap in the running mean
  (+0.021 at 25 samples, +0.067 at 100), which is easy to attribute to noise.

The lesson generalises past this adapter: **anything derived from calibration is
episode state.** If a model caches it, `reset()` must clear it, and a test should
say so. `tests/test_policy.py::test_reset_clears_the_camera_adapter` does.
