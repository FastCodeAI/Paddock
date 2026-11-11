# NAVSIM

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `navsim` |
| Primary metric | `epdms` |
| Adapter | [`adapters/benchmarks/navsim/`](../../src/paddock/adapters/benchmarks/navsim/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [autoe2e](../models/autoe2e.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [uniad](../models/uniad.md), [uniad_b2d](../models/uniad_b2d.md), [vad](../models/vad.md), [vad_b2d](../models/vad_b2d.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

PDMS / EPDMS: NAVSIM plans a 4 s trajectory from one frame, re-simulates it
through a kinematic bicycle model with an LQR tracker, and scores the outcome
against a cached ground truth — collisions, drivable-area compliance, progress,
time-to-collision, lane keeping and comfort, combined into one composite.

It is **open loop**: the ego is re-simulated to score the plan, but the model never
sees the consequences of its own actions, so errors do not compound. Paddock records
it as `mode: open_loop`, alongside KITScenes.

## What exactly is measured

Four things a reader should not have to infer, stated once:

- **Devkit: NAVSIM v2** (`Drive-JEPA/navsim_v2`, `[benchmarks.navsim] devkit`).
  The v1 devkit beside it in the same checkout is driven by the `navsim_v1`
  benchmark ([navsim_v1.md](navsim_v1.md)), which subclasses this adapter's
  harness and proxy and scores the same scenes as v1.1 PDMS.
- **Split: `navtest`** — 136 logs, 12,146 real recorded scenes, single stage.
  Set by `[benchmarks.navsim] split`; the harness also knows
  `navtest_two_stage`, `navmini` and `navtrain`.
- **Score: EPDMS**, the v2 composite: the product of the multiplicative terms
  {no at-fault collision, drivable area, traffic-light compliance, driving
  direction} times a weighted average of {progress 5, time-to-collision 5,
  lane keeping 2, history comfort 2, two-frame extended comfort 2}. v1's PDMS
  has no lane-keeping, traffic-light or two-frame-comfort terms, so the two are
  not interchangeable, and a v1 checkpoint scored here gets a v2 number.
- **Not the v2 leaderboard protocol.** Published NAVSIM v2 figures are EPDMS
  on `navhard` *two-stage*: a synthetic second stage (`synthetic_scenes/`)
  branches off each real scene to test reactivity, and scores are far lower
  there. This fork ships `navtest_two_stage` and `navsafe_two_stage` but no
  `navhard`, and `synthetic_scenes/` is not on disk at this site, so nothing
  two-stage can run here yet. EPDMS on navtest is the same metric on
  different, easier data — do not compare it to published navhard numbers.

## OpenScene is nuPlan, not nuScenes

Worth stating plainly, because the names invite the opposite assumption and the
mistake costs a wasted download. NAVSIM runs on OpenScene, which is a compact
redistribution of **nuPlan** (Motional). A nuScenes copy cannot serve it:

| | NAVSIM / OpenScene | nuScenes |
|---|---|---|
| cameras | `CAM_F0 L0 L1 L2 R0 R1 R2 B0` (8) | `CAM_FRONT`, `CAM_FRONT_LEFT`, … (6) |
| maps | `nuplan-maps-v1.0` gpkg; Las Vegas, Pittsburgh, Boston, one-north | lanelet2 + expansion PNGs; Boston seaport, Singapore |
| logs | `2021.05.25.14.16.10_veh-35_02482_02649` | `scene-0103` |

Different collection campaign, vehicle rig, cities and map format. The nuScenes
copy is what feeds neuroNCAP and has no overlap with this.

One convenience follows from the rig: in OpenScene the **lidar frame is the ego
frame** — every log frame carries `lidar2ego_translation [0,0,0]` and an
identity rotation — so a camera's `sensor2lidar` extrinsics are already
`camera2ego`. `CAM_F0` sits at `z = 1.52 m`, a real height above the road, which
is what openpilot and alpamayo need for their ground-plane warp.

## Running it

```bash
paddock run --bench navsim --model drivejepa
paddock run --bench navsim --model drivejepa --limit 20   # smoke test
```

Paddock starts the model's policy node, then runs NAVSIM's own
`run_pdm_score_sequential.py` in the devkit's interpreter with
`agent=paddock_proxy`, which forwards each `AgentInput` to the node over HTTP.
NAVSIM stays authoritative about scoring; paddock stays authoritative about
model environments. That split is the only reason four models in four
incompatible environments can be scored by one scorer — the scorer runs on
Python 3.9 under conda, openpilot on 3.12 under uv, and they cannot share a
process.

The score CSV, hydra config and scorer log land in the run's artifact
directory, and the CSV is parsed back through the same parser `paddock ingest`
uses, so a run scored today and the August run already in the store are the
same shape.

## Configuration

```toml
[benchmarks.navsim]
python       = ".../envs/drive-jepa/bin/python"   # can import navsim and nuplan
devkit       = ".../Drive-JEPA/navsim_v2"         # the directory holding navsim/
metric_cache = ".../exp/metric_cache_v2"          # run_metric_caching.py builds one
split        = "navtest"
workers      = 4
nuplan       = ".../nuplan-v1.1"                   # optional: 10 Hz frames, see "10 Hz from nuPlan"
keyframe_cameras = false                          # optional: rear views from OpenScene's keyframes, same section
gpu_nodes    = 1                                  # policy nodes per GPU model
```

Four environment switches, all deliberate and all narrow. Every one is
written into `run.json` (`config.agent`, `config.allow_partial`,
`config.allow_slow`, `config.overrides`) and hashed into the run id, so a run
made with one can never share a config hash with a run made without it:

- `PADDOCK_NAVSIM_ALLOW_PARTIAL=1` — run despite partial sensor coverage. The
  run is stored `valid: false` with the coverage (`N of M logs`) in `notes`.
- `PADDOCK_NAVSIM_AGENT=<hydra agent group>` — run a vendor's native agent
  instead of the proxy. Used to reproduce a recorded run and to measure what the
  proxy costs; the run is stored `valid: false`, since a scorer-aware agent is
  not comparable with the proxy runs (see below).
- `PADDOCK_NAVSIM_ALLOW_SLOW=1` — score a model whose `min_input_hz` exceeds
  OpenScene's 2 Hz anyway. Stored `valid: false`: it measures the rate gap.
- `PADDOCK_NAVSIM_OVERRIDES="k=v ..."` — extra hydra overrides for the scorer,
  recorded verbatim.
- `PADDOCK_NAVSIM_MERGE_FROM=<run dir>` — take another run's completed chunk
  CSVs as this run's rows and leave their logs out of the chunks. For
  continuing a run whose chunking changed: `--resume` reuses chunks by
  index, which means nothing across orderings. Recorded as `merged_from`;
  the other run must have been made under the same measurement configuration,
  which is the caller's to check.

Logs are chunked **largest first** (`config.chunk_order: by_size`). The scorer
hands each of its workers one whole log, so a chunk costs its longest log;
chunked alphabetically, a 156-scenario log beside 22-scenario ones left seven
of eight openpilot nodes idle for two hours at a time. Grouping logs of like
size keeps the workers busy together.

`--limit N` is recorded the same way and also sets `valid: false`: it exists to
test the pipeline. A site `[models.<name>] env` block is recorded as
`config.model_env`, because an environment override can change a model's input
(`ALPAMAYO_FRAME_SELECT`, for one). None of this needs a hand edit of a results
file; the runner decides `valid` from these, and the reason is in `notes`.

## A partial copy of the sensor data scores part of navtest

navtest is 12,146 scenarios across 136 logs, and every ranked run covers all of them.
NAVSIM scores only the scenarios whose sensor blobs are on disk. With 5 logs present:

```
WARNING - Unused metric cache for 11904 tokens. Skipping these tokens.
INFO    - Starting pdm scoring of 242 scenarios...
```

That is 242 of 12,146, or 2%. The early runs on this subset are stored `valid: false`,
and `preflight` now refuses a partial split by default, so a subset cannot be reported
under the full split's name.

Closing the gap means fetching the `openscene_sensor_test_camera` shards
(`Drive-JEPA/download/download_test.sh`). Extrapolating from the 5 logs present
(3.5 GB), the navtest subset is **roughly 95 GB** — an estimate from per-log
size, not a measured figure; the script pulls the whole test split, which is
larger.

## Comparability: Drive-JEPA's native agent sees the scorer

`run_pdm_score_sequential.py` passes `metric_cache` and
`past_ego_simulated_states` into `compute_trajectory`, and Drive-JEPA's native
agent uses them: `drive_jepa_model.py:98` re-runs the PDM simulator at inference
time to re-rank its 32 proposals by predicted two-frame extended comfort — a
component of EPDMS itself.

No other model in paddock has an equivalent. **The vendor's recorded 0.90673 is
therefore not a baseline the other three can be measured against.** The proxy
agent accepts those arguments and ignores them; runs through `paddock_proxy`
are the comparable set.

What the stage reads is narrower than those argument names suggest, and it
matters outside NAVSIM. `calibrate_score` takes the previous *chosen* plan run
through the PDM simulator, the seconds since it, and `metric_cache.ego_state` —
the only attribute of the cache it touches (`drive_jepa_model.py:101-102`). No
ground truth, no map, no other agents. A node is sent the ego state with every
observation and knows its own previous plan, so it can build all three itself:
`DRIVEJEPA_MOMENTUM=on` makes the drivejepa node do so
(`adapters/models/drivejepa/momentum.py`), for closed-loop benchmarks where
consecutive calls belong to one drive.

That switch does not change this benchmark. The proxy scores each scenario
through `/infer_sequence`, which resets the node first, so no previous plan
survives from one scenario to the next and the stage never runs: proxy rows are
the raw argmax with the switch on or off. Feeding the proxy's plans back across
scenarios — which is what would close the 0.02705 below — is a separate
decision about this benchmark's comparability and is not made here.

### navhard's streamed mode (opt in)

navhard (NAVSIM v2.2, `navhard_two_stage`) has an opt-in mode that does feed
the node its own plans back, for the stage-one frames upstream's sequential
loop would: `PADDOCK_NAVHARD_STREAMED=1 paddock run --bench navhard --model drivejepa`.
Off, nothing changes for any model. On:

- For each **real** stage-one frame the proxy adds a `stream` to the
  `/infer_sequence` request: the log name, the frame's own timestamp and its
  global pose, read from the Scene (`protocol.StreamFrame`). The observations
  are byte-identical to a standard run's. The node still resets per sample, but
  Drive-JEPA keeps one previous-plan book per log across those resets and is
  fed a plan that is 0.45-0.55 s old, as in closed loop. The scorer's
  `metric_cache` and `past_ego_simulated_states` are still ignored.
- **Synthetic** stage-two scenes are sent exactly as in a standard run, with no
  stream, and never enter a log's book. The Scene tells them apart:
  `scene_metadata.corresponding_original_scene` is set only for a synthetic scene.
- The v2.2 scorer plans each worker's stage-one frames in set order (hash
  order). In one standard run's plan dump (uniad, 10 Nov 2026), only 109 of the
  225 (previous, now) pairs had the earlier frame planned first. `adapters/benchmarks/navhard/score.py`
  therefore swaps that one line of the official worker for the same tokens
  sorted by (log, timestamp), in streamed mode only. A log's frames all go to
  one worker, and each request blocks until the node has kept its plan. So
  all 225 "now" frames can be fed, unless the frame before one failed.
- The run records it. `config.streamed = "per_log"` changes the run id, and a
  standard run cannot be resumed as a streamed one or the other way round. Each
  sample's `extra.momentum_fed` and `by_group.streamed` (real and synthetic
  frames, how many were fed) come from the plan dump, and the notes say so.

Upstream's navhard protocol gives no agent a previous plan. So a streamed
navhard row is a different protocol from published navhard numbers and from
a standard paddock run. Report them side by side, not as one number.

## The fork's own baselines do not run

Both stock baseline agents in `Drive-JEPA/navsim_v2` are broken by the fork's
own edits to the scoring script, so neither has ever run in this tree:

| agent | failure |
|---|---|
| `constant_velocity_agent` | `TypeError: compute_trajectory() takes 2 positional arguments but 5 were given` — the fork widened `AbstractAgent.compute_trajectory` and updated the caller, but left the stock two-argument override |
| `human_agent` | `AttributeError: 'HumanAgent' object has no attribute '_modules'` — `__init__` never calls `super().__init__()`, and the fork added `move_all_modules_to_gpu`, which walks `dir(agent)` |

Paddock ships `PaddockConstantVelocityAgent` rather than patching vendor
source: it is a reference for the scoring path itself, and it is what proves a
failure is a model's rather than the data's.

Unrelated and cosmetic, but in the same script: line 148 builds
`StateSE2(x=poses[-1,0], y=poses[-1,0], heading=poses[-1,0])` — all three from
index 0. It pollutes only the `endpoint_*` columns, not the score.

## Trajectory conversion

NAVSIM scores 8 poses at 0.5 s as `(x, y, heading)`. What each model gives:

| model | poses × dt | heading | status |
|---|---|---|---|
| drivejepa | 8 × 0.5 s | native | ready, unchanged |
| visionpilot | 8 × 0.5 s | derived | `PLAN_HORIZON_S` 3.0 → 4.0 |
| alpamayo15 | 8 × 0.5 s of a 64 × 0.1 s raw output | derived | `GRID_INDICES` widened to 8 |
| openpilot | 8 × 0.5 s | derived | grid already matched — **blocked on input rate** |
| meteor | 6 × 0.5 s, extended to 8 | derived | 3 s planner; `METEOR_HORIZON_S=4.0` opt-in, recorded as `model_env` |

Both grid changes are additive at the tail, so no recorded score moves:
`Trajectory.to_wire` hands neuroNCAP the first six poses, and KITScenes'
`displacement_errors` truncates at its 3 s horizon. VisionPilot's speed
schedule is a fixed-step forward integration, so its first 60 steps are
identical whether the loop runs 60 or 80 times.

`convert.to_navsim_poses` resamples onto NAVSIM's grid, anchored at `t=0`
(exact — the ego is at its own origin), and **refuses a horizon shorter than 4 s
rather than extrapolating**. Padding VisionPilot's missing second would score a
plan it never made.

Heading is derived from the chord between consecutive waypoints for the three
models that do not predict yaw. PDM orients the ego footprint by heading when it
tests collisions and drivable-area compliance, so it is not a detail the scorer
shrugs off. Below a millimetre of travel the chord is numerical noise, so the
last real heading is carried forward — otherwise a stopped ego gets a random
yaw and fails checks it should pass.

The conversion lives in `convert.py`, which imports no vendor code, so
`src/paddock/adapters/benchmarks/navsim/tests/test_harness.py` pins it in paddock's own environment. Same
reasoning as `CameraFit` in the openpilot adapter.

## Reproduction status

Three runs over the 242-scenario subset above, all 242/242 with zero failures:

| run | agent | EPDMS (recomputed) | scorer reported |
|---|---|---:|---:|
| baseline | `paddock_constant_velocity` | 0.25673 | 0.25673 |
| reproduction | `drive_jepa_perception_based_agent` (native) | **0.90673** | 0.90488 |
| paddock | `paddock_proxy` → drivejepa node | **0.87968** | 0.87789 |

**The reproduction is exact.** Against the August CSV, over the same token set:

```
metric comparisons: 2379/2379 bit-identical at full float precision
mean epdms  august 0.9067303343147789
            today  0.9067303343147789
            delta  0.0
```

So paddock's command construction, environment, CSV discovery and parser are
proven, and the CSV-vs-log question is closed: both figures reproduce.

## What the proxy costs, and where

Native minus proxy is 0.02705 EPDMS. Broken down by sub-metric:

| sub-metric | native | proxy | delta |
|---|---:|---:|---:|
| two_frame_extended_comfort | 0.77114 | 0.42289 | **−0.34826** |
| drivable_area_compliance | 0.99174 | 0.98760 | −0.00413 |
| no_at_fault_collisions | 0.99380 | 0.99174 | −0.00207 |
| driving_direction_compliance | 0.99793 | 0.99793 | 0 |
| traffic_light_compliance | 1.00000 | 1.00000 | 0 |
| history_comfort | 0.99587 | 0.99587 | 0 |
| ego_progress | 0.92321 | 0.92530 | +0.00209 |
| lane_keeping | 0.98760 | 0.99174 | +0.00413 |
| time_to_collision_within_bound | 0.97934 | 0.98760 | +0.00826 |

**The entire gap is one sub-metric, and it is the one the native agent
optimises.** `calibrate_score` re-runs the PDM simulator over all 32 proposals
and re-ranks them by predicted two-frame extended comfort; the proxy takes the
raw `argmax` because it has no access to the scorer's state. Every other
metric moves by less than 0.009, and three of them move *up*.

That pattern is the evidence, not just the story: if the proxy were feeding the
model degraded inputs, collision and drivable-area compliance would degrade too
and nothing would improve. Instead time-to-collision, lane keeping and progress
all improve slightly, and the proxy scores strictly better on 42 of the 171
scenarios where the two differ.

One confound, bounded rather than dismissed: drivejepa's paddock policy
reconstructs `ego_velocity` as `[speed, 0]` from the CAN bus, so the round trip
drops lateral velocity and acceleration. On this data that is |vy| ≈ 0.09 m/s
against |vx| ≈ 4.4 m/s (max 0.40), and |ay| ≈ 0.14 m/s². Small, pre-existing —
it comes from the NeuroNCAP wire contract, not from this benchmark — but it is
not zero, and it is the first thing to check if a future proxy/native gap does
*not* localise to one sub-metric.

**Consequence for the four-model comparison.** Use 0.87968, not 0.90673. The
higher figure is a model reading the scorer's own simulator at inference time,
which none of the other three can do.


## alpamayo15: under-fed, not unmeasurable

alpamayo's first NAVSIM run scored 0.226 on **4 scenarios**, below the 0.257
constant-velocity baseline, and was gated off the benchmark on that basis. That
was wrong, and the error was sample size: the same configuration scores **0.503
over 60 scenarios**. Four scenarios is a pipeline test. It is not a measurement
of a model that samples trajectories unseeded.

The underlying observation was real, though, and worth the fix. `_image_frames`
picks four camera frames by nearest timestamp on a fixed 0.1 s grid:

| history | slots → frame index | distinct frames |
|---|---|---:|
| neuroNCAP, 10 Hz | `[6, 7, 8, 9]` | 4 of 4 |
| NAVSIM, 2 Hz | `[2, 3, 3, 3]` | **2 of 4** |

Three of four slots held the same image. A paired A/B over 60 scenarios —
identical tokens, one arm per selection strategy — measured what that cost:

| metric | grid (0.1 s) | distinct | Δ |
|---|---:|---:|---:|
| **no_at_fault_collisions** | 0.7083 | 0.9917 | **+0.2833** |
| ego_progress | 0.4282 | 0.6180 | +0.1899 |
| lane_keeping | 1.0000 | 0.9667 | −0.0333 |
| drivable_area_compliance | 0.9000 | 0.8667 | −0.0333 |
| **epdms** | **0.5032** | **0.7099** | **+0.2068** |

Better on 36 scenarios, worse on 19, tied on 5; paired sign test **p = 0.030**.

The term that moves is collisions, which is the mechanism made visible: shown
four copies of one frame, the model cannot see other vehicles *moving*, so it
drives into them. The two metrics that get slightly worse are the ones a more
confident, faster-progressing plan would be expected to cost.

`_select_frames` now defaults to `auto`: use the grid unless it degenerates —
returns the same frame for two slots while distinct frames are available — and
fall back to the most recent distinct frames when it does. Under a uniform or
jittered 10 Hz feed it never degenerates, so neuroNCAP and KITScenes take the
grid path exactly as before and no recorded score moves. That property is pinned
by test rather than re-measured. `ALPAMAYO_FRAME_SELECT=grid|distinct` forces
either arm, which is how the A/B above was run.

## openpilot cannot be measured on OpenScene

openpilot ran cleanly through the proxy and scored 0.302 against the 0.257
baseline. The recorded plans say why:

| model | ego speed | planned travel at 4 s | constant velocity |
|---|---:|---:|---:|
| openpilot | 4.76 m/s | **0.30 m** | 19.05 m |
| drivejepa | 4.76 m/s | 20.51 m | 19.05 m |

30 cm in four seconds at 4.8 m/s: it had decided the car was parked. This repo
already recorded the same fact from the other direction — `neuroncap.md`: *"0.1 s
is forced, not chosen: openpilot reads a 0.5 s step as a parked car."*

Unlike alpamayo's, this is not fixable in the adapter, and the reason is in the
model's input signature. supercombo's complete non-image input set at the pinned
commit is:

```python
dims = {'desire_pulse': DESIRE_LEN, 'traffic_convention': 2, 'action_t': 2}
```

There is no speed input. `v_ego` appears in `modeld.py` only inside
`get_action_from_model`, which converts the model's *output* plan into curvature
and acceleration for the controller. The network derives ego speed entirely from
the image sequence, and the adapter already passes everything it accepts.

Nor can the frame-repeat count be tuned around it. At `repeats=10` the model
sees nine still frames then a 2.4 m jump — the average rate is honest but it
reads the recent stillness as parked. At `repeats=1` it sees 2.4 m in a believed
0.05 s and reads 48 m/s. The information is not present at the granularity the
model needs, at any repeat count.

The remaining routes are synthesizing intermediate frames, which fabricates the
model's input, or sourcing denser sensor data — OpenScene is a 2 Hz *subsample*
of nuPlan, whose cameras ran faster, so a denser history exists upstream but in
no data held here. openpilot has coverage on KITScenes (0.1 s) and neuroNCAP
(10 Hz); NAVSIM is the outlier.

So: `capabilities.min_input_hz: 10.0`, and the stored 2 Hz runs marked invalid
with the reason. A near-baseline number that is really a rate artefact is worse
than no number, because it looks like a result. The denser data turned out to
be obtainable after all -- the next section.

## 10 Hz from nuPlan

OpenScene is a 2 Hz subsample of nuPlan, whose cameras record at 10 Hz (the
devkit's `nuplan_scenario_queries.py` says so; lidar is 20 Hz). Motional's
public bucket holds the full-rate test-split blobs and the per-log databases
that say which image was taken when and where the ego was, under the same log
names and image tokens OpenScene uses. On 2026-09-11 the three front cameras
of all 136 navtest logs (212 GB, 1,072,284 frames) and their 136 databases
were pulled onto the NAS, verified against each database's `image` table:
0 missing, 0 extra, frame interval 0.100 s.

`[benchmarks.navsim] nuplan = <root>` names that data, laid out as nuPlan's own:
`splits/<split>/<log>.db` and `sensor_blobs/<split>/<log>/<CHANNEL>/<token>.jpg`.
With it set, the harness decides per model which rate to feed:

| model declares | fed | history |
|---|---|---|
| `min_input_hz` or `native_input_hz` above 2 | nuPlan, 10 Hz | the model's `history_s`, else NAVSIM's 1.5 s |
| neither | OpenScene, 2 Hz, the scorer's own four frames | 1.5 s |

So openpilot (`min_input_hz: 10`, `history_s: 4.8` for its 96-frame buffer at
20 Hz -- what KITScenes feeds) and alpamayo (`native_input_hz: 10`, `history_s:
1.5`: four frames 0.1 s apart and 16 poses, per its vendor loader) get the
dense path; drivejepa and visionpilot keep the frames they were scored on.
qwendrive takes it too (`native_input_hz: 10`, `history_s: 1.5`), for the pose
branch rather than the image one: it reads 16 poses at 0.1 s, and its four
camera frames are 0.5 s apart either way, so the dense path changes its ego
history and not its images.
Which rate a run used is in `run.json` (`config.input_hz`, `history_source`,
`history_s`) and hashed into the run id, so a 10 Hz alpamayo run cannot share
an id with the 2 Hz one.

How it is built (`dense.py`, running inside the scorer's interpreter next to
the proxy): the keyframe's own front image names the log and the image token;
the log's database gives that image's timestamp `t0` and, for every image of
the model's channels, its timestamp and the ego pose it was taken at. Frames
at `t0 - k/10 s` are picked per camera by nearest timestamp (the rig's cameras
fire within ~8 ms of each other), poses are re-expressed in the keyframe's
frame the way NAVSIM expresses its history, and velocity, acceleration and yaw
rate are nuPlan's measured ones -- no differencing. The last frame *is*
NAVSIM's keyframe: same image, origin pose. A scenario closer to the log's
start than the history asked for repeats the log's first frame, as KITScenes
does at a scene start. nuPlan drops the odd frame -- five single 0.2 s gaps
across navtest's 408 front-camera streams -- and a neighbour stands in for one,
flagged in the plan dump (`approximate`); a hole wider than 0.5 s raises, so a
scenario with missing data fails visibly rather than being scored on a stutter. The scorer, the
metric cache and the trajectory conversion are untouched: only what the model
is shown changes.

Results, full navtest (records in `results/navsim/`):

| model | 2 Hz OpenScene | 10 Hz nuPlan | frames per plan |
|---|---:|---:|---:|
| openpilot | refused (0.302 on 4 scenarios under `ALLOW_SLOW`, a parked car) | **0.7468** | 49 |
| alpamayo15 | 0.6081 | **0.6910** | 16 |
| autoe2e | not run | **0.5002** (rear views at the 2 Hz keyframes) | 66 |

Alpamayo paired on all 12,146 scenarios: every sub-metric up except traffic
light (tied); ego progress 0.596 -> 0.681. openpilot's 12,146 plans: 49 frames
each, 94 scenarios clamped at a log start, one borrowed frame; while moving the
planned 4 s travel is a median 0.82 of constant velocity and 2 plans in 12,146
look parked (under 0.15), against every plan at 2 Hz. openpilot ran on 8 CPU
nodes at ~42 s of inference per scenario (98 supercombo passes); alpamayo on
one GPU node shared with other users' work, ~25 h each.

**Views the pull does not hold.** The pull above is the three front cameras.
`[benchmarks.navsim] keyframe_cameras = true` lets a model that also reads
L2/B0/R2 onto the dense path: those views come from OpenScene's own blobs, at
its 2 Hz keyframes only, attached to the dense frame whose image token is that
keyframe and to no frame in between. For AutoE2E, whose camera history is eight
frames 0.5 s apart, that is every image it reads, with the ego history still
at 10 Hz from the logs. A model reading a rear view at 10 Hz would get it at
2 Hz, so the switch is off by default and recorded as
`config.keyframe_cameras` in `run.json`, hashed into the run id.
`gpu_nodes = N` gives a GPU model N nodes on the shared card (default 1; `nodes`
still applies to CPU-only models): AutoE2E is ~3 GB a node and bound by its
per-request forward, so its full split ran on four. The setting reaches every
GPU model a run scores, not only the one it was set for, so it is capped by the
card's free VRAM divided by the model's `vram_gb`: under `--model all` with
`gpu_nodes = 4`, alpamayo15 (26 GB) still gets one node on a card with 30 GB free.

What it does not fix: alpamayo's field of view (nuPlan's cameras are the same
~65 deg the model gets from OpenScene) and the proxy's blindness to the
scorer's state, which every model shares.

## Sample size

Two conclusions were overturned by running more scenarios, in opposite
directions:

| model | n=4 | full | |
|---|---:|---:|---|
| visionpilot | 0.640 | 0.447 (n=242) | over-read |
| alpamayo15 | 0.226 | 0.503 (n=60) | under-read |

`--limit` exists to test the pipeline — node startup, conversion, the scorer,
the CSV round trip. The runner stores any `--limit` run, and any run made under
`PADDOCK_NAVSIM_ALLOW_PARTIAL`, with `valid: false` and the reason in `notes`,
so nothing below the full split can be averaged or ranked against a real one.

## Full navtest (12,146 scenarios)

| model | EPDMS | collisions | drivable | progress | run |
|---|---:|---:|---:|---:|---|
| drivejepa | **0.8427** | 0.984 | 0.985 | 0.889 | `20260902T173156-7e208f` |
| openpilot, 10 Hz nuPlan history | **0.7468** | 0.970 | 0.913 | 0.712 | `20260911T161513-3a4c90` |
| alpamayo15, 10 Hz nuPlan history | **0.6910** | 0.957 | 0.856 | 0.681 | `20260912T040211-a9562c` |
| alpamayo15, 2 Hz OpenScene | **0.6081** | 0.941 | 0.788 | 0.596 | `20260903T183818-5879c1` |
| visionpilot | **0.4312** | 0.727 | 0.819 | 0.404 | `20260902T175642-df5291` |
| constant_velocity | **0.2385** | 0.681 | 0.578 | 0.209 | `20260902T175923-feba73` |

## Comparison figures

```bash
paddock render --bench navsim                  # every model with a valid run
paddock render --bench navsim --model meteor,openpilot --out docs/assets
```

`render.py` draws several models' plans on the same scenario, over the map the
scorer re-simulated them against -- the drivable area, the lane centres and the
annotated agents. That context is the point: EPDMS is built from drivable-area
compliance, collisions and progress, so a plan is only readable next to the
lanes it should have stayed in.

It picks the scenarios where the models' 4 s endpoints disagree most, one per
log, because a panel where every plan lies on top of every other shows only that
the scenario was easy.

Drawing needs the devkit's `navsim.visualization` and nuPlan's map API, which
live in `[benchmarks.navsim] python` and not in paddock's interpreter -- the same
split the harness makes for scoring. So `render.py` selects and gathers, and
`draw_scenes.py` draws under the devkit's python. Colour is paddock's, keyed by
model name, so a model is the same colour here as in every other figure.

**A run can only be drawn if its plan dump carries scenario tokens.** Dumps
recorded before the proxy labelled its plans (Drive-JEPA's and VisionPilot's
full-split runs, September 2026) say what was planned but not for which
scenario; those models are named in the figure's subtitle rather than silently
missing. Re-running them is the only fix.

## Status (242-scenario subset, for reference)

| model | EPDMS | n | note |
|---|---:|---:|---|
| drivejepa | 0.87968 | 242 | via proxy; the comparable figure |
| alpamayo15 | 0.65332 | 242 | with the adaptive frame selection |
| visionpilot | 0.44694 | 242 | |
| constant velocity | 0.25673 | 242 | reference baseline |
| openpilot | — | — | not measurable at 2 Hz; 0.7468 at 10 Hz from nuPlan, see the full-navtest table |

All five are the **242-scenario subset**, not navtest, and the three model rows
are stored `valid: false` (marked retroactively on 2026-09-03: they were run
under `PADDOCK_NAVSIM_ALLOW_PARTIAL` before the runner recorded that switch;
`notes` says so). They cannot be ranked against the full-split rows above. The
full split is fetched with `paddock data fetch navsim`
([Downloading a dataset](../data-setup.md#downloading-a-dataset)).

One open wart on visionpilot, not a blocker: its fitted path does not pass
through the ego origin, so its first waypoint sits ~1 m laterally off and the
derived heading for it reads −0.39 rad before snapping back. A vehicle cannot
yaw 22° and back in 0.5 s. It is VisionPilot's own path fit rather than the
conversion, and PDM's LQR tracker smooths it, but its `no_at_fault_collisions`
of 0.640 and `time_to_collision_within_bound` of 0.628 are the weakest of any
model here and this is the first thing to check.


## Making the full split tractable

Measured on the 242-scenario subset and projected to 12,146 (x50.2), the naive
sweep was ~26 h, 17.6 h of it alpamayo. Three changes, none of which alters
what is computed:

**One forward pass per scenario, not four.** `Policy.infer_sequence` feeds every
history frame through `infer` and keeps the last result. For drivejepa and
alpamayo the model is a pure function of its recorded history, so their
policies now record the first N-1 frames and infer once on the last. drivejepa's
output is bit-identical (same history, same features, same forward); alpamayo's
is the same distribution rather than the same draw, which it never was run to
run anyway. Pinned by a test on each policy. **Not** applied to openpilot or
visionpilot: both carry per-frame state (a recurrent feature buffer; a fusion
filter) that each pass advances.

**One node per scorer worker for CPU models.** The scorer runs `workers`
processes, each a serial client of one node; a model that answers one request
at a time -- visionpilot, on one thread by necessity -- serialises them all.
`[benchmarks.navsim] nodes` starts that many nodes and each worker picks one by
PID. Any request is a self-contained `infer_sequence`, so any node can serve
any request. GPU models stay at one node: the card is shared.

**Models concurrently, not in sequence -- on an idle box.** Wall time becomes
the longest model, not the sum. Each run gets its own `PADDOCK_BASE_PORT` -- see
below for why that is not optional.

Measured on a shared host, this one backfired. With three scorer pools, six
nodes and a baseline job of my own alongside a colleague's four CARLA
instances, load reached 232 on 32 cores: four visionpilot nodes ran at exactly
the single-node rate (0.60/s vs 0.62/s) while drawing ~16 cores between them,
drivejepa at 0.38/s, and alpamayo --
whose per-scenario work is twelve CPU camera warps plus a 10B-parameter GPU
rollout -- at **0.07/s, a 50 h projection** against ~4 h alone. The scorer's
PDM simulation is CPU-bound and every pool competes for the same cores, so
concurrency only pays while the machine has idle ones. The honest sequencing
here was: stop alpamayo, let the two lighter models finish, run alpamayo last
with the box to itself.

The visionpilot half had a specific cause. Its ORT sessions were built with a
bare `SessionOptions()`, whose intra-op pool defaults to every core;
`OMP_NUM_THREADS=1` in its model.yaml bounds the MUMPS solver inside its
planner and never touched ORT. Four nodes therefore ran ~128 threads on 32
cores and thrashed. `VISIONPILOT_ORT_THREADS` now caps the pool per session --
unset leaves ORT's default so recorded single-node runs are unchanged -- and
the sweep sets it to 2 through `[models.visionpilot.env]`.

Measured effect of restarting visionpilot with the cap, nothing else changed:
its four nodes 1591% -> 151% CPU, my share of the box 2173% -> 651%, load
232 -> 79, and drivejepa -- which had not been touched -- 0.39/s -> 1.34/s.
The cores it was burning were the cores everything else was waiting for.

## Losing a run, and not losing the next one

alpamayo's first full-navtest run scored all 12,146 scenarios in 11.5 h and
produced nothing. Two scenarios had failed -- its VLM emitted no trajectory
token -- and the vendor's aggregation cannot survive a failed row:
`compute_final_scores` stacks a column mixing arrays with the scalar NaN a
failed row carries, and the failed-token report applies `not` to a Series.
The scorer keeps every result in worker memory and writes one CSV at the end,
so the crash came after the last forward pass and took everything with it.

Four changes, each of which alone would have saved that run:

- **`patches/Drive-JEPA/0002`** scores valid rows, leaves failed rows NaN with
  row alignment preserved, and negates the mask with `~`. Verified per
  `patches/README.md` and on a synthetic frame with a failed row.
- **The proxy retries a model failure** (`retries` in the agent config, default
  2, so up to three requests, with a short backoff between them; a transport
  error from a node mid-restart is retried the same way as a 5xx). For an
  unseeded sampler that is a fresh draw from the same distribution; for a
  deterministic model it costs a request. The final failure still propagates,
  so the scorer records the scenario as failed rather than paddock inventing a
  plan.
- **The plan dump carries the scenario token** (from the metric cache's path,
  read as a label, never fed to the model), so a run whose scorer dies after
  the last forward pass can be re-scored from its plans through
  `run_pdm_score_from_submission.py` in about an hour of CPU.
- **The split is scored in chunks of logs** (`[benchmarks.navsim] chunk_logs`,
  default 4): each chunk is its own scorer invocation with its own CSV,
  `navsim_scorer.log` and dump under `chunks/<i>/`; `chunks.log` at the run's
  top level is the chunk-by-chunk story across every attempt (`ok`, `reused`,
  `FAILED`), appended, never truncated. A chunk that fails is recorded with the
  number of scenarios it would have scored -- counted from the metric cache,
  which holds one directory per scenario -- and the rest continue; `run.json`
  then carries `n_expected`, `n_failed` and `n_unscored`, and `notes` says
  "N chunk(s) never scored: M scenario(s) across K log(s)" rather than counting
  a lost chunk as one scenario per log. `--limit` runs stay single-invocation,
  with the scorer's `navsim_scorer.log` at the top level.

And three levels of retry, so that surviving a crash needs nobody awake:

| what died | who retries | how |
|---|---|---|
| one scenario (the model raised, or the node did not answer) | the proxy | up to `retries` more requests (default 2) with a short backoff; an unseeded sampler gets a fresh draw |
| a scenario still failed after that | the harness | one pass re-scoring just those tokens, each with a neighbour from its log so the scorer can aggregate (it dies below two consecutive frames); the neighbours' rows are discarded, scored rows replace failed ones on merge |
| a chunk, the scorer, or a node | the runner | tears the nodes down, starts fresh ones, re-invokes `evaluate` on the **same** artifacts; completed chunks are skipped. `[benchmarks.navsim] run_retries`, default 2 |
| paddock itself, or the box | you | `paddock run --bench navsim --model X --resume <run_id>` continues that run in place, same id, nothing scored twice -- under the same configuration: the switches a run starts under are written to its `run_config.json`, and a resume with a different agent, switch or chunk size is refused rather than merged. `PADDOCK_NAVSIM_RESUME_FROM=<dir>` reuses another run's chunks instead |

A run's cost on any single failure is now one chunk of four logs, about
twenty minutes at alpamayo's rate, against the eleven and a half hours the
first run lost.

## Two runs at once

`/alive` says only that *a* paddock node is answering, so a run whose own node
was still loading could pass its health check against another run's node on
the same port and score the wrong model. Nodes therefore also answer
`/whoami` with the model they serve and the nonce the launcher gave them (the
run id), and the launcher refuses a port that answers as a different model *or
a different launch of the same model* — two concurrent A/B arms of one model
would otherwise pass each other's check. `/whoami` is retried within the
startup deadline, so a slow answer on a loaded box is not mistaken for a
foreign node, and a node that dies during the run is reported after it.
`PADDOCK_BASE_PORT` gives each concurrent run its own range. This applies to
any two concurrent paddock runs, not only NAVSIM.

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Each note below starts with the mark it explains. Marks about a model
rather than this benchmark (∗ ◊ ◇ ✦ ✧) are explained in
[What these models are](../models/comparison.md#notes-on-the-models).

### Notes on the matrix cells

† Every NAVSIM figure is the full navtest split, 12,146 real recorded
scenarios, scored through paddock's proxy agent. This is NAVSIM v2's **stage
one** only: the leaderboard's stage two, the pseudo-simulation on synthetic
scenes (navhard / navsafe two-stage), is not run -- it needs the
`synthetic_scenes/` download and a second metric cache. Alpamayo's 0.691 is
with the 10 Hz history from nuPlan (see ‡); at OpenScene's 2 Hz the same model
scores 0.608 on the same split. Its earlier 242-scenario estimate was 0.653
against that 0.608, which is why subsets are stored `valid: false`. On the subset Drive-JEPA scored 0.880
through the proxy — the vendor's own agent scores 0.907 by reading the
scorer's simulator at inference time, which no other model can do, so the
proxy figure is the comparable one. `preflight` refuses a partial split unless told
otherwise, so a subset cannot be reported as full navtest by accident, and the
subset rows in `results/` are stored `valid: false` -- they appear under
"Excluded runs" and never average against a full-split row.
The **NAVSIM v1 PDMS** column scores the same 12,146 scenarios with the v1.1
devkit -- the composite the 2024 leaderboard uses, on which the NAVSIM paper
reports constant velocity 20.6 and human 94.8 -- see
`docs/benchmarks/navsim_v1.md`. Alpamayo's and OpenPilot's v1 figures replay the
plans their 10 Hz v2 runs recorded -- the same plans, the other scorer -- so each
model's two columns describe one run; Alpamayo's 2 Hz pair (0.608 / 0.676) is in
the tables below. Drive-JEPA and VisionPilot are fresh proxy runs at 2 Hz.

‡ OpenScene is a 2 Hz dataset and OpenPilot infers motion from frame-to-frame
change, so on it OpenPilot plans a parked car — 0.30 m of travel in 4 s at
4.8 m/s. Its model has no speed input at all (`{desire_pulse, traffic_convention,
action_t}`), so this is not fixable in the adapter: it declares
`min_input_hz: 10.0` and the harness refuses the pairing on OpenScene alone. The
frames OpenScene left out exist upstream — nuPlan's cameras run at 10 Hz — and
with `[benchmarks.navsim] nuplan` naming that data the harness feeds a 10 Hz
history to OpenPilot and to Alpamayo (built for 0.1 s frames). On the full
split OpenPilot scores 0.747 at 10 Hz, its plans tracking ego speed (median
0.82 of constant-velocity travel over 4 s while moving; 2 of 12,146 plans look
parked), and Alpamayo 0.691 against 0.608 at 2 Hz, every sub-metric up but
traffic light. See "10 Hz from nuPlan" in `docs/benchmarks/navsim.md`.

☉ AutoE2E v1.0, the `kitscenes-epoch-5` checkpoint: the release's nuPlan model
fine-tuned on KITScenes. NAVSIM is fed from nuPlan at 10 Hz with the rear views
taken from OpenScene's keyframes (`keyframe_cameras`), which are exactly the
0.5 s frames its camera history reads, and no route. AutoE2E was pre-trained on
nuPlan, whose test logs `navtest` is drawn from; see *AutoE2E × navsim* in
[comparison.md](../models/comparison.md). See
[docs/models/autoe2e.md](../models/autoe2e.md).

### NAVSIM v2

Full navtest (stage one: real recorded scenes, no synthetic stage two), 12,146 scenarios, all sub-metrics 0–1. Every column except EPDMS
is the mean of a per-scenario pass/fail or ratio, and EPDMS is the weighted
product the scorer builds from them. Drive-JEPA is scored through paddock's
proxy agent (see †). Extended comfort is the mean over the scenarios where it
is defined, 10,040 of the 12,146 -- it compares consecutive frames' plans, so
it is absent for a log's first frame and for any scenario without a scorable
predecessor (`docs/results-schema.md`); every other column is over all 12,146.
The last column is the 242-scenario subset (2% of navtest) that every model
was first run on, for reference only.

| Model | EPDMS ↑ | No at-fault collisions ↑ | Drivable area ↑ | Driving direction ↑ | Traffic light ↑ | Ego progress ↑ | TTC within bound ↑ | Lane keeping ↑ | History comfort ↑ | Extended comfort ↑ | EPDMS on 242 subset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenPilot, 10 Hz nuPlan history ‡ | 0.747 | 0.970 | 0.913 | 0.954 | 0.998 | 0.712 | 0.965 | 0.947 | 0.983 | 0.653 | — |
| Alpamayo 1.5, 10 Hz nuPlan history ‡ | 0.691 | 0.957 | 0.856 | 0.933 | 0.997 | 0.681 | 0.951 | 0.963 | 0.962 | 0.576 | — |
| Alpamayo 1.5, 2 Hz OpenScene | 0.608 | 0.941 | 0.788 | 0.890 | 0.998 | 0.596 | 0.939 | 0.931 | 0.938 | 0.490 | 0.653 |
| Drive-JEPA | **0.843** | 0.984 | 0.985 | 0.991 | 0.998 | 0.889 | 0.979 | 0.974 | 0.978 | 0.477 | 0.880 |
| VisionPilot | 0.431 | 0.727 | 0.819 | 0.926 | 0.975 | 0.404 | 0.700 | 0.899 | 0.968 | 0.518 | 0.447 |
| METEOR ◊ | 0.303 | 0.694 | 0.687 | 0.805 | 0.979 | 0.317 | 0.683 | 0.795 | 0.973 | 0.451 | — |
| Qwen-Drive 1.0, 10 Hz nuPlan history ‡ | 0.840 | 0.982 | 0.964 | 0.993 | 0.998 | 0.820 | 0.977 | 0.975 | 0.982 | **0.867** | — |
| AutoE2E, 10 Hz nuPlan history ☉ | 0.500 | 0.824 | 0.777 | 0.865 | 0.990 | 0.491 | 0.814 | 0.830 | 0.984 | 0.812 | — |
| Baseline (Constant Velocity) | 0.239 | 0.681 | 0.578 | 0.805 | 0.981 | 0.209 | 0.669 | 0.827 | 0.979 | 0.567 | 0.257 |
