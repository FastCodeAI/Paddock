# comma2k19 — the dataset

What comma2k19 *is*, what it records, and where its numbers come from.

This is the dataset; [the benchmark](../benchmarks/comma2k19.md) is the protocol
paddock scores over it, and [data-setup](../data-setup.md) is how to get the
bytes. The split is deliberate — the benchmark's `history_frames`,
`keyframes_per_segment` and horizons are paddock's choices and can change; none
of what follows can.

Every measured claim below names the command that produced it. Where a number
could have been asserted from arithmetic, it was instead enumerated — see
*Where 10,165 comes from*, which is the one number every row of the comma2k19
leaderboard is conditioned on.

---

## What it is

comma.ai's 2019 release of **33 hours of highway driving**, almost all of it on
**California's I-280** between San Jose and San Francisco. Two vehicles, a
windscreen-mounted **EON** (a modified LeEco Le Pro3 phone) recording video at
20 Hz, and the car's own CAN bus and radar logged alongside a fused
GNSS/INS/vision ego pose.

* Paper: *comma2k19: A Driving Dataset for the Development and Validation of
  Fused Pose Estimators* — Harald Schafer, Eder Santana, Andrew Haden, Riccardo
  Biasini (comma.ai), [arXiv:1812.05752](https://arxiv.org/abs/1812.05752).
* Toolkit and dataset home: [`commaai/comma2k19`](https://github.com/commaai/comma2k19),
  pinned to a sha in [`repos.yaml`](../../repos.yaml). The toolkit
  is a hard requirement of this benchmark, not a convenience — see *The toolkit*.
* Licence: **MIT** for the toolkit code, and the dataset itself is published for
  research use under the terms on the repository. Check them before
  redistributing anything derived from the video.

What makes it unusually suitable for open-loop planning work is the second half
of the title: the pose is *already fused*. There is no SLAM run to reproduce and
no GNSS pipeline to tune before a number means anything, which is why this is
the largest open-loop board in paddock and the cheapest to reproduce exactly.

What makes it narrow is the first half of the location: it is one highway, in
one metropolitan area, in good weather, at speed. Any conclusion drawn from it
is a conclusion about steady-state highway driving. See *What it cannot answer*.

## Vocabulary and physical layout

Four levels, and paddock's sample ids are built from three of them:

```
Chunk_1 .. Chunk_10                      10 chunks — a download unit, nothing more
  <dongle_id>|<start_time>               178 routes — one continuous drive
    0, 1, 2, ...                         2035 segments — one minute each
      video.hevc                         ~1200 frames, 1164x874, 20 Hz
      global_pose/                       frame_times, frame_positions,
                                         frame_velocities, frame_orientations
      processed_log/CAN/                 speed, steering, radar tracks
      processed_log/IMU/                 accelerometer, gyro, magnetometer
      raw_log.bz2                        the unprocessed capture
```

A **chunk** is a packaging artefact — it carries no meaning, and a route does
not straddle one. A **route** is one continuous drive, named
`<dongle_id>|<start_time>`, where the dongle id identifies the vehicle. A
**segment** is one minute of that route and is the unit paddock iterates; its
identity is `chunk/route_id/segment`, which is the first three fields of every
comma2k19 `sample_id`.

The two dongle ids, and the split between them
(`loader.RAV4_DONGLE` / `CIVIC_DONGLE`, counted by the enumeration below):

| vehicle | dongle id | segments |
|---|---|---|
| Honda Civic 2016 | `99c94dc769b5d96e` | 1653 |
| Toyota RAV4 2016 | `b0c9d2329ad1606b` | 382 |

**A partial download is normal, not an error.** `loader.list_segments` is the
single definition of what counts as a segment, and it is deliberately quiet: it
skips any chunk directory whose suffix is not an integer, any route directory
without a `|` in its name, any segment directory whose name is not digits, and
— the one that matters — any directory lacking **both** `video.hevc` and
`global_pose/frame_times` (`loader._is_segment`). So an interrupted download
yields a smaller dataset rather than a crash, and the run records which samples
it actually covered (see *The fingerprint*). Results are sorted by
`(chunk, route_id, segment)` rather than by filesystem order, so `--limit N`
selects the same N samples at every site.

## The sensors, and what each is worth

Everything paddock scores a model's auxiliary heads against comes from this
list. `harness._reference()` is the code; this is what it is reading.

**Ego pose** — `global_pose/frame_positions` (ECEF), `frame_orientations`
(quaternions), `frame_velocities`, one entry per camera frame. This is the
fused product the paper is about, and it is the ground truth for every
trajectory metric. Nothing needs to be run to obtain it.

**Speed, twice, and they disagree.** The CAN bus carries the car's own
odometer; the pose stream carries `|frame_velocities|`. They differ by roughly
1 %, and **neither is privileged** — wheel-speed has a tyre-circumference
scale error, GNSS-derived speed has its own noise. So the harness prefers CAN,
falls back to the pose for a segment that logged none, and records
`speed_source` per sample rather than leaving a reader to infer which was used
from whether the error looks large.

**Yaw rate** is differenced from the recorded orientation, not read from the
gyro, and is negated on the way out because the device frame is right-positive
(see *Frames and units*).

**Radar**, from the CAN log: a set of tracks per timestamp. `loader.lead_track`
picks the closest track inside the ego lane (`HALF_LANE_M`), which is the one
definition of "the lead" in this adapter — shared by the overlay that draws the
box and the harness that scores a model's lead head against it, so the vehicle
a reviewer looks at and the number they read are the same vehicle.

**No lane labels at all.** comma2k19 publishes no lane geometry, no map, and no
other-agent tracks. This is the single most consequential gap in the dataset,
and it propagates:

* `auxiliary.py`'s lane numbers (`lane_width_m`, `lane_offset_m`) are
  **plausibility gates, not errors** — they check a model's projected lane
  against what a US highway lane can physically be (2.5–4.5 m), because there
  is nothing to compare against. A model passing them is not accurate; it is
  merely not obviously broken.
* The proxy driving score's NC and DAC terms use the radar's closest lead and
  the human's own path as a corridor, for the same reason. The benchmark doc
  says where that stops being a driving score.

**IMU** is present and paddock does not read it on the scoring path.

## Frames and units

comma2k19's **device frame is x forward, y right, z down**. Paddock's ego
contract is **x forward, y left, z up**. The conversion is
`_DEVICE_TO_EGO = diag(1, -1, -1)`, applied exactly once, on the way out of the
loader — and a y-flip applied twice, or zero times, produces a mirrored world
that raises nothing and scores plausibly.

The EON camera (`loader.EON_*`, asserted equal to the toolkit's own values by
`tests/test_harness.py`):

| | |
|---|---|
| resolution | 1164 × 874 |
| focal length | 910 px |
| horizontal FOV | ≈ 65.2° |
| height above road | **1.22 m**, *not recorded by the dataset* |
| frame rate | 19.95 Hz measured (dt = 0.05012 s ± 0.0025) |

Two of those deserve their footnotes. The camera height is openpilot's
`calibrationd.HEIGHT_INIT` prior, used because comma2k19 does not record one;
what makes it defensible is that supercombo's `road_transform` head
independently estimates 1.22 m on this data. And the **HEVC container is wrong
about the rest**: it claims 25 fps and a frame count of 0. The frame count comes
from `global_pose/frame_times`, which has one entry per frame; the rate comes
from differencing it. That 19.95 Hz happens to equal openpilot's
`ModelConstants.MODEL_RUN_FREQ`, so nothing anywhere resamples.

The camera also has a **real mount rotation and it is not zero** — 3.3–3.8° of
pitch on the RAV4 and 1.6–2.2° on the Civic for nine routes in ten, estimated
per route by `calibration.py`. Correcting it was worth about 3× in cross-track
error, measured against the camera-frame ground truth used before the ground
truth was rotated into the car's frame too. The benchmark doc's *Calibration*
section has the full story, including
the defect where passing `np.eye(4)` as "no rotation" produced ADE 38.8 m.

## Two traps that cost real time

Both are recorded here so nobody rediscovers them, and both are pinned down by
tests in `adapters/benchmarks/comma2k19/tests/test_harness.py`.

**1. The README states the quaternion direction backwards.**
`frame_orientations[i]` is `ecef_from_device`, *not* `device_from_ecef`. The
dataset's own code is right where its prose is wrong —
`comma2k19/utils/camera.py::device_from_ecef` transposes before use. Getting
this backwards raises nothing whatsoever: it mirrors every relative position
through the direction of travel, so a straight segment still looks straight.

**2. The toolkit is Python 2 in places, and its notebooks call functions that
do not exist.** At the pinned sha, `utils/orientation.py`'s `euler2quat` and
`rot2quat` use `xrange`, and `rot2quat` additionally assigns a length-1 array
into a scalar slot, which modern numpy refuses. Between them that rules out
`euler2rot`, `rot2euler`, and everything built on either. Names the upstream
notebooks call (`rotations_from_quats`, `rot_from_euler`) are not in the module
at all. So `toolkit.py` imports a deliberately small set — `quat2rot`,
`img_from_device`, `eon_intrinsics`, `rot_matrix` — and paddock keeps
`euler_from_rot` locally, which `test_harness.py` then uses to cross-check the
toolkit's forward rotation rather than trusting either alone.

## The toolkit

paddock imports the dataset's own toolkit rather than copying its matrices.
[`toolkit.py`](../../src/paddock/adapters/benchmarks/comma2k19/toolkit.py) is the
only module that knows it exists; it is not pip-installable (no `setup.py`), so
the mechanism is the one every model uses — a pinned clone whose root goes on
`sys.path` **only while the import runs**, then comes off.

That care is not fastidiousness. The toolkit's package is called `utils`, about
as collidable a name as exists, and the clone also contains a directory called
`openpilot` — a partial vendored copy of comma's stack, *not* the model paddock
evaluates. Left on `sys.path` it answers `import openpilot` with the stub, which
is exactly how this was found: openpilot's adapter tests stopped skipping and
started failing.

The clone is checked by `preflight()`, so a missing one fails a sweep in seconds
with a sentence rather than an hour in with a traceback.

## Where 10,165 comes from

Every comma2k19 row in the README is `n = 10165`, and the number is worth
deriving rather than quoting, because a row computed over a different sample set
is not comparable to the others no matter how close it looks.

A keyframe is a moment at which a model is asked to plan. It needs the whole
input window *behind* it and the whole scored future *ahead* of it, so
`harness.keyframe_indices` places the first at `history_span_frames - 1` and the
last at `n_frames - future_frames - 1`, then spaces
`keyframes_per_segment` evenly between them. At the benchmark's settings
(`history_frames = 160`, `input_hz = 20`, `gt_horizon_s = 10` at `interval_s =
0.5`, so 200 future frames) a segment needs **≥ 360 frames — 18 seconds — to
place any keyframe at all**, and gets 5 as soon as it does.

Run the enumeration rather than the arithmetic:

```python
# for the full script see the "Reproducing the counts" section below
for seg in bench._segments:
    k = len(bench.keyframe_indices(bench._n_frames(seg)))
```

which reports:

```
segments recognised          : 2035
keyframes_per_segment        : 5
minimum frames for 1 keyframe: 360

keyframes placed per segment : {0: 2, 5: 2033}
total scored samples         : 10165
2035 * 5 would be            : 10175

segments yielding fewer than 5: 2
  10/99c94dc769b5d96e|2018-11-15--01-02-09/3: 194 frames -> 0 keyframes
  7/99c94dc769b5d96e|2018-07-20--16-42-49/0: 222 frames -> 0 keyframes
```

So the shortfall is **not** a scattering of segments losing a keyframe each. It
is exactly two truncated segments — 9.7 s and 11.1 s of recording against the
60 s the rest hold — that cannot place a single one, and every other segment
contributes its full five. 2033 × 5 = 10,165.

The two are named rather than counted because "10 samples short" invites the
wrong repair. Nothing is wrong with the dataset here and nothing is missing from
the download; those two segments are short as published.

### The fingerprint

`run.json` records `dataset.sequences_sha256` — the first 16 hex digits of the
SHA-256 of the run's sorted sample ids (`results/store.py::sequence_hash`). For
a complete sweep at the settings above it is:

```
293a225a0b5cbaff
```

It is a function of `history_frames`, `keyframes_per_segment`, `input_hz` and
which segments are on disk. **A row whose hash differs is not comparable to one
whose hash matches**, whatever the two `n` values agree on — which is the single
most common reason two sites' numbers will not reconcile, and is invisible
without something like this. `paddock render` prints the hashes side by side and
warns when they diverge.

## Splits

There are none. comma2k19 publishes no train/val/test split, so paddock scores
every segment, and the benchmark doc's
[Splits](../benchmarks/comma2k19.md#splits) section records the consequence:
none of the models on this board were trained on this data, which is what makes
the comparison zero-shot for all of them and is a property worth preserving.

## What it cannot answer

Stated once, so no reader has to infer it from an absent column:

* **No closed loop.** comma2k19 is log replay. A Bench2Drive driving score, a
  neuroNCAP score or a true NAVSIM PDMS all require a simulator that reacts to
  the plan; nothing computed from this dataset is one of those. The benchmark's
  `pdm_style_score` is named for that reason.
* **No other agents, no map, no lane labels**, so nothing here measures
  interaction, right-of-way, or lane-level compliance against a reference.
* **One road type, one region, mostly clear weather, mostly daylight.** It is a
  highway dataset. A model that does well here has been shown to do well at
  steady-state highway driving, and shown nothing about intersections.
* **No night/rain stratification** is published, so a per-condition breakdown
  would have to be labelled first.

## Reproducing the counts

Everything measured above comes from one of these:

```bash
paddock doctor                                   # is the dataset visible, and how much of it
paddock list datasets                            # the configured roots
paddock run --bench comma2k19 --model X --dry-run # sample count without running anything
```

The per-segment enumeration behind *Where 10,165 comes from* is a short script
against the harness's own methods — `bench._segments`, `bench._n_frames`,
`bench.keyframe_indices` — rather than a re-implementation of the rule, so it
cannot drift from what a sweep actually does.

For a full inventory of **this machine's** copy — per-chunk segment counts,
CAN/radar coverage, disk footprint, the lot —
[`tools/benchmarks/comma2k19/metadata_report.py`](../../tools/benchmarks/comma2k19/metadata_report.py)
writes a twelve-section `COMMA2K19_METADATA.md`:

```bash
# -m from the repo root: the script imports `tools.paths`, so running it by
# path leaves `tools` off sys.path and it exits with ModuleNotFoundError.
python -m tools.benchmarks.comma2k19.metadata_report --out "$PADDOCK_RUNS/COMMA2K19_METADATA.md"
```

That report is **generated, site-specific and untracked** by design: it names
real paths, so it is written under `{roots.runs}` rather than into the repo,
where `tests/test_portability.py` would reject it. This document is the tracked
half — everything in it is true of the dataset itself, on any machine.

## See also

* [The benchmark](../benchmarks/comma2k19.md) — what is scored, the evaluation
  rate, calibration, recorded results.
* [Getting the data](../data-setup.md#comma2k19) — download and layout.
* [`loader.py`](../../src/paddock/adapters/benchmarks/comma2k19/loader.py) —
  segment discovery, pose maths, video streaming.
* [`toolkit.py`](../../src/paddock/adapters/benchmarks/comma2k19/toolkit.py) —
  the one place paddock touches comma's own code.
