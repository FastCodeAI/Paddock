# AlpaSim

Closed-loop driving on replayed real scenes. NVIDIA's AlpaSim renders a recorded
clip as a neural reconstruction, hands the policy camera frames at a fixed
control rate, and steps the world forward on what the policy commands. Errors
compound, which is the point: a policy that drifts a metre in the first second
is shown the consequence rather than being re-anchored to the log.

The primary metric is `scene_score`, 0–1, meaned over scenes.

## What you need

| | |
|---|---|
| repo | `alpasim`, pinned at `affc2ea` in `repos.yaml` |
| scenes | the `public_2601` suite that ships inside the AlpaSim repo |
| GPU | ~17 GB for the simulator alone; ~41 GB with a 10B driver resident |
| disk | ~180 MB per scene of rollout artifacts, under `{runs}` |

No dataset download, and **no `datasets.yaml` entry**: the scenes are USDZ
artifacts fetched by AlpaSim's own scene provider on first use, so there is no
root for a site to point at and `requires:` would have nothing to check.
`benchmark.yaml` therefore sets `dataset: ""`, the same as Bench2Drive. Which
scenes a run drove is pinned by `sequences_sha256` in its `run.json`, not by a
path.

## Running it

```bash
# The scene SET comes from [benchmarks.alpasim] scene_set, or --scenarios.
paddock run --bench alpasim --model alpamayo15 --scenarios public_2601_first10
paddock run --bench alpasim --model constant_velocity --scenarios public_2601_first10

# --scene takes individual scene ids, which here are clipgt-<uuid>:
paddock run --bench alpasim --model constant_velocity --scene clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
```

Roughly **5 minutes per scene** with a 10B driver at 4 cameras and 10 Hz — a
10-scene sweep is about 50 minutes, of which the first 5 are the checkpoint
read. The constant-velocity floor over the same 10 scenes takes about 8 minutes,
because the cost is rendering rather than inference.

**Two runs at once.** The bridge starts looking for a port at 9200 and the
wizard hands its services host ports from 6000, skipping ports already bound.
Two runs started together both find the same free ones before either binds
them. A run with `PADDOCK_BASE_PORT` set (the gate sets one per lane) keeps to
its own range instead: the bridge from base + 50, the wizard's services from
base + 60. Without it, the ports are 9200 and 6000 as before. Docker starts
the containers, so they would not inherit the run's `taskset`; the launcher
pins each one to the run's cores as it appears (`docker update
--cpuset-cpus`, within 2 s), and a run on every core pins nothing. Measured on
2026-10-27: one scene's 5 containers all on cores 24-27, the busiest at about
one core, the mean 17% of one.

## How the bridge steers a model

AlpaSim's controller follows a list of poses: a position and a heading per
waypoint. The bridge builds that list the same way for every model:

- **The whole plan.** It reads `plan` from the node's `/infer` answer, every
  waypoint the model returned, rather than the six points of `trajectory`
  (those six exist for neuroNCAP, which indexes them positionally).
- **Headings from the model when it has them.** A plan whose rows are
  (x, y, yaw) is driven on the model's own headings. A plan of (x, y) only gets
  headings from the path's direction: each waypoint faces along the segment
  that leads to it, and a segment shorter than 5 cm keeps the previous heading,
  so a plan that stands still does not spin. A node too old to send `plan` is
  driven on its `trajectory`, also with derived headings.
- **Recorded per scene.** Each scene's sample record says how its plans were
  steered: `headings` is `model`, `derived` or `mixed`, and `bridge_plans`
  holds the counts. `history_frames` says how a frame the planner needed no
  plan from was sent: `observe`, `infer` (a node without `/observe`, the plan
  dropped), or null when the planner sent none.
- **A clean start.** The bridge calls `/reset` at the start of every scene, for
  every model, so a model that keeps a frame history never pairs one scene's
  first frames with the previous scene's last ones.

Before this, the bridge drove six (x, y) points and told the controller to keep
the car's current heading at each of them. What the change means for the
models that declare `alpasim`:

| Model | Plan it returns | Headings now | Effect on its AlpaSim results |
|---|---|---|---|
| Drive-JEPA | 8 × (x, y, yaw) | the model's | driven on all eight poses (see below) |
| Alpamayo 1.5 | 8 × (x, y) | derived from the path | now driven on 8 waypoints with headings, where it was driven on 6 with none; and from the second scene of a sweep on, its frame and pose history starts empty, where it used to carry over from the previous scene (the node stays up for the whole sweep, and its `reset` clears that history). Its results will shift for both reasons; it has not been re-run |
| constant velocity | 6 × (x, y), straight along x | derived, and all 0 | steered exactly as before (a straight path's derived heading is 0, the "keep the current heading" it was always sent) |

The rows in [Reference numbers](#reference-numbers) were made by each model's
native AlpaSim driver, not by this bridge, so none of them changes.

## Per-model settings in the bridge

Most models are driven on one shared set of simulation settings
(`configs/driver/paddock_bridge.yaml`) and the bridge passes AlpaSim's cameras
through as they are. A model whose numbers must be compared with a native
AlpaSim driver can instead get its own settings: an entry in
`harness.DRIVER_PROFILES`, which names a driver config, extra bridge flags, a
scene list pinned to artifact uuids and whether the bridge writes a per-replan
log. Video is not among them: it stays the site's `video` setting for every
model, since AlpaSim renders it after scoring and it changes no number. A
model without an entry gets exactly what every model got before profiles existed, and `run.json` records a profile under `config`
only when one is used, so the run id changes with it.

Four models have a profile. Drive-JEPA's reproduces the settings of
Drive-JEPA's native AlpaSim driver (adapter 0.1.4b) so a run through paddock
can be checked against that driver's runs: one front camera at 20 Hz, a 3.0 s
replayed warm-up, the shared 20 Hz scoring config and the 30 scenes pinned to
their 26.01 artifacts. What the bridge does for it, and how the result compares
with the native driver, is in
[docs/models/drivejepa.md](../models/drivejepa.md#through-paddock-on-alpasim).

AutoE2E's has no native driver to match. It renders the six cameras the model
was trained on: AlpaSim's six recorded cameras re-rendered as undistorted
pinholes at the KITScenes field of view, the view straight back borrowing the
tele camera's id because AlpaSim can change a scene's camera but not add one
(`configs/driver/paddock_bridge_autoe2e.yaml`). It shares Drive-JEPA's warm-up,
control step and 30 pinned scenes. Its bridge flags select a second rhythm,
`replan_rhythm.py`, which any node can use:

| Flag | What the bridge does |
|---|---|
| `--replan-every-us 100000` | asks the node on a 100 ms grid anchored on the scene's first ask; answers the ticks between from the last plan, re-timed onto now and re-anchored to the car (the same computation as Drive-JEPA's rhythm); sends every warm-up frame to `/observe` instead of asking for a plan AlpaSim would not drive |
| `--force-gt-us`, `--frame-interval-us` | the warm-up length and control step; a test checks both equal the driver config's |
| `--pad-ego-history` | before the scene's first frame, sends ego-only frames straight back from its pose at the speed measured from poses (AlpaSim reports zero velocity while it replays the warm-up), one per control step back to 6.6 s |

The per-replan log (`<scene>.replans.jsonl`) has one line per tick: whether the
model was called or its plan reused, the motion-history rows that fall on
padding, the image frames the node repeated, and the plan's length and spacing.
What this means for AutoE2E's inputs, and what is not yet known, is in
[docs/models/autoe2e.md](../models/autoe2e.md#on-alpasim-through-the-bridge).

Qwen-Drive's reproduces the settings and inputs of its native AlpaSim driver
(adapter 0.1.4): four f-theta cameras at 20 Hz with a 30 ms shutter, the
`hyperion_8_1` bonnet mask, a 3.0 s warm-up with the driver called, 400 steps
(`configs/driver/paddock_bridge_qwendrive.yaml`), and the same 30 pinned scenes.
It uses the same rhythm with a 1.5 s grid, plus switches that change what a body
carries. Each is off unless a profile passes it, and with all of them off every
body is byte-for-byte what it was before they existed (`tests/test_bridge_switches.py`
pins that for the default, Drive-JEPA and AutoE2E profiles):

| Flag | What the bridge does |
|---|---|
| `--observe-between-replans` | after the warm-up, also sends every step that is not a replan to `/observe`, so a node that keeps its own history at the control step has every step |
| `--canbus-ego-frame` | canbus[13] and [14] carry the forward and sideways speed in the car's frame, instead of [13] carrying the speed's magnitude; [12] stays the yaw rate. While AlpaSim replays the warm-up it reports velocity and yaw rate as exactly zero; when the last two poses move faster than 1 m/s, those are replaced from the poses |
| `--canbus-acceleration` | canbus[7] and [8] carry the simulator's own forward and sideways acceleration (Qwen-Drive's profile leaves this off) |
| `--send-route` | sends the route AlpaSim holds as `route_xy`, in the car's frame at the route's own time, as AlpaSim sends it |
| `--send-scene-id` | sends the scene id as `scene_id` |
| `--named-cameras-only` | decodes and sends only the cameras named with `--camera-name` |
| `--waypoint-dt 0.1` | (an existing flag) the plan's spacing; Qwen-Drive returns all 50 of its points 0.1 s apart |

What the node does with these, and what still differs from the native runs, is
in [docs/models/qwendrive.md](../models/qwendrive.md#on-alpasim-through-the-bridge).

Alpamayo 1.5's reproduces the settings of its stored AlpaSim row, made by
AlpaSim's own Alpamayo driver: the same four f-theta cameras, warm-up and 400
steps as Qwen-Drive's (`configs/driver/paddock_bridge_alpamayo15.yaml`; a test
checks the two simulation blocks are equal), and the same 30 pinned scenes. All
four cameras are sent, the 30° telephoto included, under the node's camera
names. The node is asked every 100 ms, the model's own rate, with the step
between sent to `/observe`. It uses `--send-route` and `--named-cameras-only`
from the table above, and two more switches, also off for every other profile:

| Flag | What the bridge does |
|---|---|
| `--send-route-time` | with `--send-route`, sends the route's own timestamp as `route_timestamp_us`: the time of the car frame `route_xy` is in, so the node can move the route into its newest pose's frame as AlpaSim's Alpamayo driver does |
| `--send-plan-dt` | sends the spacing the bridge will drive the plan at (`--waypoint-dt`, here 0.1 s) as `plan_dt`, so a node that predicts more finely than its default spacing can return every point it predicts |

`tests/test_bridge_switches.py` pins what the default, Drive-JEPA, AutoE2E and
Qwen-Drive profiles send, so these two switches are checked to change none of
them. What the node does with them, and what still differs from the native
runs, is in [docs/models/alpamayo15.md](../models/alpamayo15.md#on-alpasim-through-the-bridge).

## Configuration

```toml
[benchmarks.alpasim]
repo    = "{repos}/alpasim"
compose = "docker"            # the wizard drives docker compose
drivers = "{weights}/alpasim-drivers"
```

## The tick config is the comparison group

AlpaSim runs at whatever camera set and control rate you configure, and **those
choices change what is being measured**, not merely how precisely:

| group | cameras | control | `force_gt_duration_us` |
|---|---|---|---|
| `1cam-20Hz` | 1 forward | 20 Hz | 3,000,000 |
| `4cam-20Hz` | 4 (front wide, front tele, cross L/R) | 20 Hz | 3,000,000 |
| `6cam-20Hz` | 6 (front wide, front tele, cross L/R, rear L/R) | 20 Hz | 3,000,000 |

All three groups are the **shared referee**: one control rate, one force-GT
duration and one `eval-config.yaml` across every model, so that for the first
time the rows differ by the model rather than by the harness. How much that
matters is visible in the floor itself — constant velocity scores 0.3562 at
20 Hz on the 0.1.0 driver, 0.7411 on 0.2.0 and 0.8850 on 0.2.3, a spread wider
than the gap between most of the models.

A faster control rate gives a policy more opportunities to correct, so
**numbers from different control rates are not comparable and must never be
averaged together.** Camera count is different: each model is given the
cameras it reads, and rows that differ only in camera count are ranked
together and read against the same floor. Every run records its
group in `run.json` under `config.renderer`, `config.topology` and
`config.scene_set`, all of which are hashed into the run id — so two tick
configs cannot collide on one run id. **There is no `config.group` key and no
`by_group` block on this path**: grouping the rows is done by the ingest reader
for recorded campaigns, not by the adapter, which drives one scene per run.

Compare a model against the constant-velocity floor **at its control rate and on
its scenes**.

## What the scored window actually is

Aggregates do not describe the whole rollout. AlpaSim's aggregation chain
(`eval/aggregation/processing.py`) applies, in order:

1. drop every timestep before `eval_relevant > 0` — the warm-up trim
2. derive the combined events (`offroad_or_collision`, `collision_at_fault`, …)
3. drop any trajectory with a black frame
4. **truncate after the first `offroad_or_collision`** — this keys on
   `collision_any`, so a *not-at-fault rear-end* truncates exactly like an
   at-fault one
5. **truncate again** once `dist_to_gt_trajectory` reaches the run's
   `max_dist_to_gt_trajectory` (4.0 m here)

Both truncations keep the triggering step, which is why a published maximum can
exceed the cap it was truncated on.

The consequence is the trap in this benchmark:

> **A policy that fails early is scored over a shorter window than one that
> drives the scene out.** Its trajectory-fidelity numbers are measured only over
> the easy opening seconds and look *better* as a result.

In the `1cam-20Hz` collection Drive-JEPA was scored over **46.7 %** of its ticks
and OpenPilot over **78.4 %**. On `04749bb9` Drive-JEPA was scored over 9 of 398
ticks — 2.3 % — so whatever fidelity it posted there describes less than half a
second of driving, while OpenPilot's number on the same scene covers 337 ticks.

The scored fraction is **not** recoverable from `results-summary.json` — it needs
the per-rollout `metrics.parquet` replayed through the modifier chain, which is
what `tools/benchmarks/alpasim/scored_fraction.py` does. Until you have run it,
assume nothing about comparability of fidelity numbers across runs.
**Rank on `scene_score` and the outcome rates;
treat cross-run fidelity comparisons as invalid unless the scored fractions
match.**

## `scene_score`, and why the floor beats most models on the ten scenes

```
progress_score = min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)
scene_score    = 0.0  if collision_at_fault or offroad  else progress_score
```

**The hard-failure rule zeroes a scene outright** on an at-fault collision or an
offroad event, whatever progress was made before it. There is no partial credit:
a policy that drives 95 % of a scene and then clips a car scores the same 0.0 as
one that never moved.

That rule produces the result that most needs stating plainly:

| | `scene_score` | progress | vs its floor |
|---|---|---|---|
| OpenPilot supercombo 0.2.0 | 0.9566 | 0.9010 | **+0.0716** |
| **constant-velocity floor 0.2.3, `1cam-20Hz`** | **0.8850** | **0.8290** | *(floor)* |
| VisionPilot v2 2.0.0 | 0.8488 | 0.8202 | −0.0362 |
| Drive-JEPA 0.1.4 | 0.5030 | 0.5652 | −0.3820 |

**Two of the three released models score below constant velocity.** This is not a bug in
the metric and not a claim that constant velocity drives well — it does not
steer, and it rear-ends things. But a rear-end collision is *not at fault*, so
the hard-failure rule does not zero it, while the at-fault collisions and
offroad events the real policies commit do. Driving straight at a fixed speed
down a mostly-straight recorded clip turns out to be a strong baseline against
that rule — it scores 1.0 on 8 of the 10 scenes.

**This is why the floor is recorded as a run at all** — under
`results/alpasim/constant_velocity/`, marked in its `run.json` notes as a
reference rather than a model. A `scene_score` means nothing without it. Any
model reported on this benchmark should be quoted alongside the floor for its
group, and a model that does not beat the floor has not demonstrated competence,
however respectable its absolute number looks.

## Reference numbers

`public_2601` first 10 scenes, uuid-pinned, n=10, scorer `affc2ea`. Every row is
filed under `results/alpasim/`, so `paddock report` agrees with these tables.

> **These numbers were not produced through this adapter.** They come from
> AlpaSim's own native drivers — `openpilot-supercombo-driver-0.2.0`,
> `visionpilot-v2-driver-2.0.0`, `drivejepa-v2-driver-0.1.4`,
> `alpamayo1_5-driver-0.91.0`, `constant_velocity-driver-0.2.3` — run before the
> paddock bridge existed. `alpamayo15` and `constant_velocity` declare
> `alpasim` in their `model.yaml`, and `drivejepa` and `autoe2e` do too since
> each has its own settings in the bridge (see
> [Per-model settings in the bridge](#per-model-settings-in-the-bridge));
> OpenPilot and VisionPilot do not, because nothing demonstrates that pairing.
> So these rows are **reference values for what the benchmark yields**, not
> evidence about the bridge. Drive-JEPA's run through the bridge is compared
> with its native driver in [docs/models/drivejepa.md](../models/drivejepa.md#through-paddock-on-alpasim)
> and is not one of these rows.
>
> **A later version of the Drive-JEPA driver has no row here.**
> `drivejepa-v2-driver-0.1.4b` adds momentum-aware selection to 0.1.4; its
> source is not in this repo. It was measured on 30 scenes (these ten plus
> twenty more), one rollout each, outside paddock, and that number is in
> [docs/models/drivejepa.md](../models/drivejepa.md). Paddock keeps results
> for released checkpoints only, so a driver that serves another checkpoint is
> not ingested.
>
> **METEOR's rows come from a driver that is not tracked here.** `meteor-0.1.1`
> is a native AlpaSim driver written at the Fastcode site; its source is not in
> this repository, so its two rows (`6cam-20Hz`, below) are reference values
> another site cannot yet reproduce.

### `1cam-20Hz` — the shared referee

Four rows on one referee: same control rate, same force-GT duration, same
`eval-config.yaml` (md5 `68cf4aaa89356e0f6053d76eb4ef2321`, byte-identical
across every 20 Hz row, including Alpamayo's and METEOR's).

| model | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|
| OpenPilot supercombo 0.2.0 | **0.9566** | 0.1303 | 0.9010 | 10/10 | — |
| *constant-velocity floor 0.2.3* | *0.8850* | *0.2312* | *0.8290* | *10/10* | *—* |
| VisionPilot v2 2.0.0 | 0.8488 | 0.3170 | 0.8202 | 9/10 | at-fault 1 |
| Drive-JEPA 0.1.4 | 0.5030 | 0.4306 | 0.5652 | 7/10 | at-fault 2, offroad 1 |

**Of the released checkpoints, only OpenPilot clears the floor**, by +0.0716. This is the finding, not a
measurement artefact: `scene_score` saturates at `progress_clipped_rel ≥ 0.8`
and zeroes only on `collision_at_fault` or `offroad`, so a blind constant
velocity that never commits either scores 1.0 on 8 of 10 scenes. Progress
separates what the score cannot — floor 0.8290 against OpenPilot's 0.9010.

Two rows need their caveat quoted with the number:

- **VisionPilot** finishes 0.0362 below the floor and **the entire deficit is
  one at-fault collision** on `048b974e`; the other nine scenes average 0.9431.
  Its `set_speed` is the human clip's mean speed, handed to it per scene — the
  model does not choose its own cruising speed here. Its own `smoke_gates.json`
  reads `ok: false` on 3 of 12 gates (two Prometheus ports, two that look like
  gate-side indexing); none touch the referee pins, which were read from the
  resolved config and are on-pin.
- **Drive-JEPA**'s 0.5030 is a mean over six early-terminated or hard-gated
  scenes and four full-distance runs that all scored 1.0000. It is a pattern —
  it wins 1, ties 3 and loses 6 against the floor — not one bad scene.

### `1cam-20Hz` on 30 scenes

The support matrix ranks these rows together with the 30-scene rows of the
other camera groups, all read against this floor; the ranked table is under
[Support-matrix results](#support-matrix-results). The same drivers, referee config and scorer, on 30 scenes: the ten above plus
twenty more drawn from `public_2601` (seed 20260922, stratified by speed and
turn, from the scenes of the suite available at renderer version 26.1.112 and
not already among the ten). The list is tracked as
`adapters/benchmarks/alpasim/scenes/public_2601_30scenes.yaml`, and a test
checks it against the rows' `sequences_sha256`. One rollout per scene. The rows are in
`results/alpasim/` as the `…_30scenes` runs; each merges the ten-scene run with
twenty single-scene runs, and the ingester refuses the merge if the driver or
the control rate differs between them.

| model | `scene_score` | sd | progress | at-fault | off-road | vs floor |
|---|---|---|---|---|---|---|
| OpenPilot supercombo 0.2.0 | **0.7726** | 0.3627 | 0.7238 | 0/30 | 3/30 | +0.2850 |
| VisionPilot v2 2.0.0 | 0.6289 | 0.4106 | 0.6355 | 4/30 | 1/30 | +0.1413 |
| *constant-velocity floor 0.2.3* | *0.4876* | *0.4252* | *0.5411* | *2/30* | *5/30* | *—* |
| Drive-JEPA 0.1.4 | 0.4209 | 0.4378 | 0.5091 | 7/30 | 5/30 | −0.0667 |

Three things to read with it:

- **The floor moves with the scene set**, from 0.8850 on the ten to 0.4876 on
  the 30. A mean over one set says nothing against a mean over the other, so
  quote the difference from the floor and the number of scenes with any figure.
- **Alpamayo 1.5's 30-scene row was driven through paddock's bridge.** Its
  stored native row covers the first ten scenes only (0.9000). The 30-scene run
  through the bridge, 0.7952, is in the support matrix and in the table under
  "AlpaSim (nuRec)" below; see [docs/models/alpamayo15.md](../models/alpamayo15.md#on-alpasim-through-the-bridge).
- **One rollout per scene, so the differences are loose.** Paired over the 30
  scenes, the standard error of the difference from the floor is 0.077 for
  OpenPilot, 0.080 for VisionPilot and 0.089 for Drive-JEPA 0.1.4.
  VisionPilot's +0.1413 and Drive-JEPA's −0.0667 are both under two standard
  errors.
- **`paddock run --bench alpasim` still drives the ten by default.** The
  default `scene_set` is unchanged; the twenty extra scenes were driven by each
  model's native driver and ingested. `scene_set = "public_2601_30scenes"` in
  `[benchmarks.alpasim]` selects the 30. (`--scenarios` only filters by the
  configured set's name; it does not switch sets.) The one run made that way
  is Drive-JEPA's through the bridge, compared in
  [docs/models/drivejepa.md](../models/drivejepa.md#through-paddock-on-alpasim);
  it is not filed as a row.
- **A sample is its clip.** Each scene's record has `sample_id`
  `alpasim/<clip>` whichever set drove it, so a 30-scene run and a 10-scene run
  pair on their ten shared scenes, and a run over the 30 has the same
  `sequences_sha256` as the stored 30-scene rows. The record's `split` says
  which half of the 30 the scene is in, `pilot10` (the ten above) or `ext20`
  (the twenty drawn after them), and `native_id.scene_set` names the set the
  run drove.

### `4cam-20Hz`

| model | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|
| Alpamayo 1.5 (10B), via the referee proxy | **0.9000** | 0.3000 | 0.9603 | 9/10 | at-fault 1 |

On these ten scenes the 20 Hz constant-velocity floor scores 0.8850, so
Alpamayo is +0.0150 against it. This row was made by AlpaSim's own Alpamayo
driver before paddock's bridge could drive the model; the 30-scene run through
the bridge (0.7952, the same 0.9000 on these ten) is the one the support matrix
quotes. Its environment is verified identical to the reference rows'
(`pip freeze` 300/300 packages, image `8a69f005`) and all 25 smoke gates pass on
all ten scenes. Scene-level variance on this model is large — `04749bb9` has
produced 0.0, 0.9148 and 1.0 across identical environments — so neither this
mean nor any other single-rollout mean here should be read to four figures.

### `6cam-20Hz`

| model | scenes | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|---|
| METEOR, released ONNX as shipped, driver 0.1.1 | first 10 | 0.6869 | 0.3103 | 0.6073 | 10/10 | — |
| METEOR, released ONNX as shipped, driver 0.1.1 | 30 | 0.4232 | 0.3673 | 0.4301 | 24/30 | off-road 5, at-fault 1 |
| AutoE2E `kitscenes-epoch-5`, through paddock's bridge | first 10 | 0.3175 | 0.1938 | 0.2924 | 8/10 | at-fault 2 |
| AutoE2E `kitscenes-epoch-5`, through paddock's bridge | 30 | 0.1694 | 0.2034 | 0.2867 | 14/30 | at-fault 9, off-road 7 |

One sweep of the 30 scenes of `public_2601_30scenes`, one scene per run
directory, made on 2026-10-13/14 and ingested as
`results/alpasim/meteor/legacy-meteor_011_published_30scenes`; the ten-scene
row, `…_first10`, is the first ten of those directories, not a second sweep.
Same control rate, force-GT duration, step count and `eval-config.yaml` as the
other groups, read from each run's resolved config.

- **Read against the one floor.** Constant velocity reads no camera, so the
  30-scene floor (0.4876) is these rows' floor too: METEOR is −0.0644 against
  it and AutoE2E −0.3182 ([Support-matrix results](#support-matrix-results)).
- **The cameras are not the recorded ones.** METEOR expects pinhole images, so
  the six cameras are rendered as pinholes with square pixels and no
  distortion, each at the narrower of METEOR's own camera and the recorded
  one: 97.70° for the three wide cameras (recorded about 120°), 69.66° and
  69.72° for the rear pair, 29.60° for the telephoto. The bonnet mask the
  other groups apply is off, because it was drawn for the recorded fisheye
  image and does not line up with a pinhole one. The rendered widths were
  fixed from the first ten scenes' rigs. On four of the twenty extra scenes
  (`cdd9448c`, `6725df2d`, `bd07fb91`, `b1b170d0`) the rear-right camera is
  therefore rendered wider than that scene's recorded camera: by 0.24° on
  `6725df2d`, 0.05° on `cdd9448c` and 0.006° on the other two.
- **Two of METEOR's eight camera slots are empty** (the rear wide and rear
  telephoto), fed as zeros with a neighbour's calibration, as in the model's
  own training for a missing camera.
- **The step-time check failed on 18 of 30 scenes.** The driver's own run
  checks budget 200 ms (the 95th percentile of pre-processing plus inference)
  per fresh plan, and on those scenes it was 201 to 313 ms; the host was
  shared and under load. Whether a late plan changes a score has been checked
  on one scene only: the first scene's 29 scorer metrics are identical to
  those of a run of the same scene made on 2026-09-22, and both runs were
  over the budget on that scene, by different amounts (206 and 283 ms).
- **The driver's other run checks.** Each run directory records the driver's
  own checks, and the step-time check is not the only one that failed. None
  of them is a scorer result and none marks the row invalid; the table says
  what each failure was, read from the runs' logs.

  | driver check | failed on | what the failures are |
  |---|---|---|
  | referee settings and scene list match the pinned run | 30 / 30 | every setting and the renderer version (`26.1.112-59673f88`) match; only the scene-list file differs, because the check pins the ten-scene list and this sweep used the 30-scene one |
  | step time within 200 ms | 18 / 30 | above |
  | plan shape: no heading step over 0.35 rad between waypoints | 17 / 30 | on 14, every such step is at under 1.3 m/s on a plan segment under 0.4 m long: near-stop wiggles where a heading is barely defined. On `04749bb9`, `acd22430` and `b1b170d0` they are real, at up to 25 m/s on segments up to 5.3 m, and those three scenes score 0.23, 0 and 0 |
  | consecutive plans agree within 2 m | 5 / 30 | `02eadd92`: one jump, at 0.2 s, while the simulator still drives the recorded path. `2e1ce314`, `04749bb9`, `acd22430`, `b1b170d0`: jumps of 2 to 5 m at 11 to 24 m/s in the 1.4 s after handover. These are the model's plans changing, not a harness fault |
  | warm-up guard fires on the zeroed-state ticks | 4 / 30 | `5f064b02`, `b44a4a44`, `d2715023`, `e3ed554f`: the car was stationary throughout the zeroed window, so there was nothing to guard. The guard never fired where the state was real and the starting speed was never wrong |
  | rendered camera no wider than recorded | 4 / 30 | above |

  So the failures are either the check's own scope (scene list, near-stop
  wiggles, a stationary start) or the model's plans at speed, which the
  scores on those scenes already show.
- **The first ten and the twenty extra differ sharply**: 0.6869 against
  0.2913. All five off-road events and the at-fault collision are on the
  twenty extra scenes. What the model does on them is in
  [docs/models/meteor.md](../models/meteor.md).

**AutoE2E** (`results/alpasim/autoe2e/20261022T045847-86b955`, valid) is the
first six-camera row driven through paddock's own bridge, so any site can
reproduce it with `paddock run --model autoe2e --bench alpasim` and
`scene_set = "public_2601_30scenes"`. Its cameras are not METEOR's: the
KITScenes field of view, a rear view drawn from the roof, a 1 µs global
shutter ([Per-model settings in the bridge](#per-model-settings-in-the-bridge),
[docs/models/autoe2e.md](../models/autoe2e.md#camera-setup-on-alpasim)). So it
shares the group's name, not its inputs; like every 30-scene row, it is ranked
with the rest. The score is low for one reason seen on almost every scene: the car
leaves the human's path to the left within about a second at speed, so the
scored window ends early (a median 1.55 s after handover) and 6 of its 15
at-fault collisions fall after it. Detail in
[docs/models/autoe2e.md](../models/autoe2e.md#results).

## Reproducing this

**Environment.** AlpaSim's own conda environment, plus:

- `casadi<3.8` — 3.8 changed an API the controller uses.
- On Blackwell (sm_120), the msda extension needs the deformable-attention
  patch. `torch.cuda.is_available()` returning `True` is **not** evidence of
  kernel support — compare `torch.cuda.get_arch_list()` against the device's
  compute capability. A torch built to sm_90 loads happily on sm_120 and then
  fails inside a kernel.

**Gated Hugging Face repos.** Alpamayo 1.5 resolves five, and discovers them one
failure at a time unless all are cached first:

```
nvidia/Alpamayo-1.5-10B       nvidia/Cosmos-Reason2-8B
Qwen/Qwen3-VL-8B-Instruct     Qwen/Qwen3-VL-2B-Instruct   (the processor)
```

Set `HF_HUB_OFFLINE=1` for the driver. This is load-bearing, not an
optimisation: a *cached* gated repo still triggers an unauthenticated HEAD
request that returns 401, and `huggingface_hub` does not fall back to the cache
on a 401.

**Weights are never committed.** Fetch `Alpamayo-1.5-10B` into `{weights}`; the
copy these numbers came from has a shard-manifest sha256 of `9cc6fc6ad2852e66`
(10 files, 20.64 GiB), which is what to compare against. The processor files
(`Qwen/Qwen3-VL-2B-Instruct`) may also go in `{weights}/Qwen3-VL-2B-Instruct`
(or wherever `ALPAMAYO_PROCESSOR` points); the model's server reads them from
there when that folder exists, and from the Hugging Face cache otherwise.

**Driver startup.** Set `startup_timeout_s: 900`. The default is 120 s and a cold
10B checkpoint read answers the runtime's version probe at about 305 s, so the
default kills the run before the model finishes loading.

**Scene set.** `test_suite_id: public_2601`, `limit_to_first_n: 10`, uuid-pinned
via the suite table — `sequences_sha256` in each `run.json` is the check that two
sites scored the same scenes. The Alpamayo and Drive-JEPA sweeps drove the same
ten scenes one explicit id at a time, so their directories record
`test_suite_id: null`; the split on their records comes from the campaign's
`split` key in `[ingest]`, and the matching `sequences_sha256` is what backs it.

**Driver identity.** The five `driver_version_id`s listed under Reference
numbers, as probed by the runtime at connect. Those are
**AlpaSim's own native drivers**, which is what produced the tables above — see
the note under Reference numbers. A run driven through this adapter reports
`paddock-bridge/<scenario>` instead, and the model it served is in `run.json`
under `model`, not under a `driver_version_id` key: this adapter writes none.

**Sampling, and what the recorded numbers are.** Alpamayo 1.5 generates with
`temperature=0.6`, `top_p=0.98`, and seeds only when `force_determinism` is set.
**This adapter deliberately does not set it** (`launcher.py` carries the
reasoning): `driver.model.force_determinism` seeds a model *AlpaSim* runs, and
under this adapter AlpaSim runs no model — paddock owns the process and the
bridge only relays. Setting it here would be a no-op that reads as a guarantee.
Sampling determinism for a model like Alpamayo belongs to its paddock policy.
So a run driven through paddock is **not** reproducible by this flag, and a
stochastic model's row is one draw unless its own policy seeds itself.

**The recorded Alpamayo run above is one draw per scene.** It ran through the
referee proxy with no seed recorded, and `04749bb9` has scored 0.0, 0.9148 and
1.0 across identical environments — re-running the identical command would not
reproduce it. Read it
as one sample, not as the model's score, and do not compare a fresh per-scene
result against it and conclude anything from a difference. This is the same
property `docs/benchmarks/neuroncap.md` documents for Alpamayo there, where five
seeds on one scenario spanned 0.000 to 5.000 with sd 1.817. The constant-velocity
floor has no such caveat: it is deterministic, which is what makes it the useful
reference.

**Scorer.** Pinned at alpasim `affc2ea`. Note the divergence from the
`e2e_challenge` scorer: `open_loop_collision` does not exist there, so a number
quoted from that scorer is not this one.

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Moved here from the README unchanged except for links; the symbols are
the README's former footnote marks, and each note below starts with the mark it
explains. Marks about a model rather than this benchmark (∗ ◊ ◇ ✦ ✧) are
explained in [What these models are](../models/comparison.md#notes-moved-from-the-readme-support-matrix).

### Notes on the matrix cells

¶ AlpaSim `scene_score` over **30 scenes**, one rollout per scene, n=30: the
first 10 scenes of the `public_2601` suite plus 20 more drawn from the same
suite (seeded, stratified by speed and turn). Scorer `affc2ea`, under the
**shared 20 Hz referee** — one control rate, one force-GT duration and one
`eval-config.yaml` (md5 `68cf4aaa…`) across every row, so the rows differ by
the model rather than by the harness. The files are in `results/alpasim/`, as
the `…_30scenes` runs, and `paddock report` reads them. Against the
constant-velocity floor (0.4876): Alpamayo 1.5 +0.3076 (through paddock's bridge, ¶¶), OpenPilot +0.2850, VisionPilot +0.1413,
Drive-JEPA −0.0667. With one rollout per scene these differences carry a
paired standard error of about 0.08, so VisionPilot's margin and Drive-JEPA's
deficit are both under two standard errors. The scene list is tracked as
`adapters/benchmarks/alpasim/scenes/public_2601_30scenes.yaml`. `scene_score` saturates at `progress_clipped_rel ≥ 0.8`
and zeroes a scene on an at-fault collision or an off-road event, so quote
progress beside it (floor 0.5411, OpenPilot 0.7238, VisionPilot 0.6355,
Drive-JEPA 0.5091). The floor falls from 0.8850 on the first ten scenes to
0.4876 on the 30, so a mean is only comparable to another over the same scenes.
The ten-scene figures and the per-row caveats, including the gates
VisionPilot's run did not pass, are in the AlpaSim section below and in
[docs/benchmarks/alpasim.md](alpasim.md).

¶¶ Camera count is recorded, not used to separate rows: Alpamayo ran 4 cameras
at 20 Hz through paddock's bridge (`4cam-20Hz`), METEOR 6 (⬡), AutoE2E 6 through
paddock's bridge (⬢), Qwen-Drive 3 through paddock's bridge, the other rows 1
camera at 20 Hz. Neither
the number of cameras nor the bridge keeps a row out of the ranking; every row
on the 30 scenes is ranked against the others and read against the one
constant-velocity floor. Scene-level variance on Alpamayo is large enough that
no single-rollout mean here should be read to four figures.

\* Alpamayo 1.5's cell is its 30-scene run through paddock's bridge, 0.7952
(`results/alpasim/alpamayo15/20261029T200113-9d900c`, valid). Its native row,
made by AlpaSim's own Alpamayo driver on the first 10 scenes only, scores
0.9000, and the bridge run scores the same 0.9000 on those 10, scene for scene.
On those same 10 scenes the other rows score: OpenPilot 0.9566, VisionPilot 0.8488, Drive-JEPA 0.5030,
constant-velocity floor 0.8850, and METEOR, on six cameras,
0.6869.

⬡ METEOR's AlpaSim cell is the same 30 scenes, referee config and scorer as the
¶ rows, one rollout per scene, but it was shown **six cameras** where they were
shown one (`6cam-20Hz`). It is ranked with them and read against the same
constant-velocity floor: camera count is not a reason to separate rows. It is the released ONNX fed as shipped, the convention of every
other METEOR cell (◇): raw pixels into the graph's own `/255`, no mean/std
normalisation, the two absent camera slots as zeros. The six cameras are
rendered as pinholes, narrower than the recorded fisheyes (a 120° camera is
rendered at 97.7°), with the bonnet mask off. The plan's mode is chosen by the
selector of METEOR's deployed runtime, not by the argmax the paddock node
uses. The run was made by a native AlpaSim driver that is **not tracked in
this repository**, so another site cannot reproduce the row yet. On the 30
scenes it has five off-road events and one at-fault collision, all on the
twenty extra scenes; on the first ten alone it scores 0.6869. The rows are
`results/alpasim/meteor/legacy-meteor_011_published_30scenes` and
`…_first10`. See `docs/models/meteor.md`.

⬢ AutoE2E's AlpaSim cell is the same 30 scenes, referee scorer and 20 Hz
control step, driven through **paddock's bridge** on six pinhole cameras at the
KITScenes field of view (`6cam-20Hz`, its own cameras: see the row's notes
above). Ranked with the other 30-scene rows. Run
`results/alpasim/autoe2e/20261022T045847-86b955`, valid, resumed after
start-up failures on a shared host (the reused scenes are recorded in
`run.json`; wall time does not change a drive). See `docs/models/autoe2e.md`.

Qwen-Drive's AlpaSim cell is the same 30 scenes, referee scorer and 20 Hz
control step, driven through **paddock's bridge** on its native driver's
settings: three f-theta cameras and a model call every 1.5 s. Ranked with the
other 30-scene rows. It ran on simulator image `6dad7728`, where the one-camera
rows ran on `1e0d5fea`; nobody has shown the two images drive identically. Run `results/alpasim/qwendrive/20261027T215528-d231d8`,
valid. See `docs/models/qwendrive.md#on-alpasim-through-the-bridge`.

### AlpaSim (nuRec)

Closed-loop driving on replayed real scenes: AlpaSim renders a recorded clip as
a neural reconstruction, hands the policy frames at a fixed control rate, and
steps the world forward on what the policy commands. `scene_score` is 0–1,
meaned over scenes. The set is the first 10 scenes of the `public_2601` suite,
uuid-pinned, one rollout per scene — **n = 10**. Every row is under
`results/alpasim/` and `paddock report` reads it; the protocol is in
[docs/benchmarks/alpasim.md](alpasim.md).

Drive-JEPA's row below is its released checkpoint behind the native driver at
0.1.4. What it reads and how it picks are in
[docs/models/drivejepa.md](../models/drivejepa.md).

**On 30 scenes** — these ten plus twenty more from the same suite, the set the
support matrix quotes — the same drivers under the same referee score:

| Model | Cameras | `scene_score` ↑ | Δ vs floor | Progress ↑ | `duration_frac_20s` ↑ | At-fault collisions ↓ | Any collision ↓ | Off-road ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo 1.5, through the bridge | `4cam-20Hz` | **0.7952** | **+0.3076** | 0.8434 | 0.657 | 1 / 30 | 3 / 30 | 4 / 30 |
| OpenPilot | `1cam-20Hz` | 0.7726 | +0.2850 | 0.7238 | 0.671 | 0 / 30 | 6 / 30 | 3 / 30 |
| Qwen-Drive, through the bridge | `3cam-20Hz` | 0.7001 | +0.2125 | 0.7873 | 0.631 | 5 / 30 | 10 / 30 | 2 / 30 |
| VisionPilot | `1cam-20Hz` | 0.6289 | **+0.1413** | 0.6355 | 0.570 | 4 / 30 | 11 / 30 | 1 / 30 |
| *floor* | *`1cam-20Hz`* | *0.4876* | *—* | *0.5411* | *0.447* | *2 / 30* | *8 / 30* | *5 / 30* |
| METEOR ⬡ | `6cam-20Hz` | 0.4232 | −0.0644 | 0.4301 | 0.469 | 1 / 30 | 10 / 30 | 5 / 30 |
| Drive-JEPA | `1cam-20Hz` | 0.4209 | −0.0667 | 0.5091 | 0.465 | 7 / 30 | 12 / 30 | 5 / 30 |
| AutoE2E ⬢ | `6cam-20Hz` | 0.1694 | −0.3182 | 0.2867 | 0.121 | 9 / 30 | 10 / 30 | 7 / 30 |

One rollout per scene, so read the differences with their spread: paired over
the 30 scenes, the standard error of each model's difference from the floor is
about 0.08 (OpenPilot 0.077, VisionPilot 0.080, Drive-JEPA 0.089).
Alpamayo 1.5 against OpenPilot is +0.023, well inside that spread: the two are
level. Alpamayo scores 0.9000 on the first ten scenes and 0.7427 on the twenty
extra, where its four off-road events are. METEOR's row is the
released ONNX on six cameras (⬡); it scores
0.6869 on the first ten scenes and 0.2913 on the twenty extra, where all five
of its off-road events and its one at-fault collision are. The floor is far
lower here than on the ten: on the twenty extra scenes constant velocity has two
at-fault collisions and five off-road events, and none of either on the first
ten. The rest of this section is about the first ten scenes.

**Every row ran under the shared 20 Hz referee**: one control rate, one force-GT
duration and one `eval-config.yaml` (md5 `68cf4aaa…`), so the rows differ by the
model rather than by the harness. Camera count is recorded in the Cameras
column but does not separate rows: each model is given the cameras it reads,
and every row is read against the one constant-velocity floor.

| Model | Cameras | `scene_score` ↑ | Δ vs floor | Progress ↑ | `duration_frac_20s` ↑ | At-fault collisions ↓ | Any collision ↓ | Off-road ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenPilot | `1cam-20Hz` | 0.9566 | **+0.0716** | 0.9010 | 0.777 | 0 / 10 | 1 / 10 | 0 / 10 |
| *floor* | *`1cam-20Hz`* | *0.8850* | *—* | *0.8290* | *0.654* | *0 / 10* | *2 / 10* | *0 / 10* |
| VisionPilot | `1cam-20Hz` | 0.8488 | −0.0362 | 0.8202 | 0.699 | 1 / 10 | 3 / 10 | 0 / 10 |
| Drive-JEPA | `1cam-20Hz` | 0.5030 | **−0.3820** | 0.5652 | 0.462 | 2 / 10 | 4 / 10 | 1 / 10 |
| Alpamayo 1.5 | `4cam-20Hz` | 0.9000 | +0.0150 | 0.9603 | 0.740 | 1 / 10 | 1 / 10 | 0 / 10 |
| METEOR ⬡ | `6cam-20Hz` | 0.6869 | −0.1981 | 0.6073 | 0.589 | 0 / 10 | 3 / 10 | 0 / 10 |
| AutoE2E ⬢ | `6cam-20Hz` | 0.3175 | −0.5675 | 0.2924 | 0.120 | 2 / 10 | 3 / 10 | 0 / 10 |

**An at-fault collision or an off-road event zeroes the scene outright**, whatever
progress preceded it:

```
progress_score = min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)
scene_score    = 0.0  if collision_at_fault or offroad  else progress_score
```

There is no partial credit — 95% of a scene and then a clipped car scores the
same 0.0 as never moving. That rule is why constant velocity is a serious
opponent rather than a formality: it does not steer and it rear-ends things, but
a rear-end is *not at fault*, so the hard-failure rule leaves it standing while
the at-fault collisions and off-road events a real policy commits zero it. At
20 Hz the floor scores 0.8850 and **of the released checkpoints only OpenPilot beats it**. A model that does
not clear its own floor has not demonstrated competence here, whatever the
absolute number looks like — and because the score saturates, progress is the
column that separates the floor (0.8290) from OpenPilot (0.9010).

**Scored window is the trap in this benchmark.** AlpaSim truncates a rollout at
the first collision or off-road event and again once the ego is 4 m from the
recorded trajectory, so **a policy that fails early is scored over a shorter
window than one that drives the scene out** — and its fidelity numbers, measured
only over the easy opening seconds, look *better* for it. `duration_frac_20s` is
how much of a 20 s scene survived that chain: Drive-JEPA's 0.462 against
OpenPilot's 0.777 is why its `dist_to_gt_trajectory` (2.44 m) cannot be read
against OpenPilot's (1.45 m). Do not compare fidelity across rows whose scored
windows differ.

Alpamayo's row is **one stochastic draw per scene**. No seed is recorded for
the run, and scene `04749bb9` has scored 0.0, 0.9148 and 1.0 across identical
environments — read it as one sample, not as the model's score. It is the same
property `docs/benchmarks/neuroncap.md` records for Alpamayo there, where five
seeds on one scenario spanned 0.000 to 5.000. The floor has no such caveat: it
is deterministic, which is what makes it the useful reference.
