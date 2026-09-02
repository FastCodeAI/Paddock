# NAVSIM

PDMS / EPDMS: NAVSIM plans a 4 s trajectory from one frame, re-simulates it
through a kinematic bicycle model with an LQR tracker, and scores the outcome
against a cached ground truth — collisions, drivable-area compliance, progress,
time-to-collision, lane keeping and comfort, combined into one composite.

It is a **closed-loop proxy**: the ego is re-simulated, but it never sees the
consequences of its own actions, so errors do not compound. Paddock therefore
records it as `mode: open_loop`, alongside KITScenes.

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
```

Two escape hatches, both deliberate and both narrow:

- `PADDOCK_NAVSIM_ALLOW_PARTIAL=1` — run despite partial sensor coverage.
- `PADDOCK_NAVSIM_AGENT=<hydra agent group>` — run a vendor's native agent
  instead of the proxy. Used to reproduce a recorded run and to measure what the
  proxy costs; not a normal path.

## This site scores 242 scenarios, not navtest

navtest is 12,146 scenarios across 136 logs. The metric cache here covers all of
them. **The sensor blobs cover 5 logs**, and NAVSIM scores the intersection:

```
WARNING - Unused metric cache for 11904 tokens. Skipping these tokens.
INFO    - Starting pdm scoring of 242 scenarios...
```

242 of 12,146 — 2%. The already-stored August run is this subset, labelled
`navtest`, which is why `preflight` now refuses a partial split by default
rather than letting a subset be reported under the full split's name.

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

No model behind an HTTP node can do that, and no other model in paddock has an
equivalent. **The vendor's recorded 0.90673 is therefore not a baseline the
other three can be measured against.** The proxy agent accepts those arguments
and ignores them; runs through `paddock_proxy` are the comparable set.

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
`tests/test_navsim_adapter.py` pins it in paddock's own environment. Same
reasoning as `CameraFit` in the openpilot adapter.

## Reproduction status

Three runs over the same 242 scenarios, all 242/242 with zero failures:

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


## openpilot cannot be measured here, and the reason is the dataset

openpilot ran cleanly through the proxy — node up, `POST /infer_sequence` 200,
242 conversions with no error — and scored **EPDMS 0.302 against a
constant-velocity baseline of 0.257**. The recorded plans say why:

| model | ego speed | planned travel at 4 s | constant velocity |
|---|---:|---:|---:|
| openpilot | 4.76 m/s | **0.30 m** | 19.05 m |
| drivejepa | 4.76 m/s | 20.51 m | 19.05 m |

openpilot planned to travel 30 cm in four seconds while moving at 4.8 m/s. It
had decided the car was parked. Its sub-metrics agree: no collisions, no
direction or traffic-light faults, `ego_progress` 0.10, and
`drivable_area_compliance` 0.0 on half the scenarios.

This is a failure mode already documented in this repo from the other
direction. From `docs/benchmarks/neuroncap.md`: *"0.1 s is forced, not chosen:
openpilot reads a 0.5 s step as a parked car."* supercombo infers ego motion
from frame-to-frame change, so the observation rate is load-bearing. neuroNCAP's
scenarios were rebuilt at 10 Hz for exactly this reason and KITScenes feeds
0.1 s frames; on both, openpilot scores normally.

**NAVSIM cannot supply that.** OpenScene is 2 Hz by construction — one keyframe
every 0.500 s and exactly one image per keyframe (334 frames, 334 JPEGs per
log). More history frames would add span, not density. The only way to a denser
history is interpolating pixels that were never recorded, which would be
fabricating the model's input.

So this is recorded as a dataset/model incompatibility, not a result:

- `model.yaml` declares `capabilities.min_input_hz: 10.0`, a general statement
  about openpilot rather than a NAVSIM special case;
- the harness refuses the pairing before scoring, naming the rate gap;
- openpilot no longer declares `navsim`, so `--model all` sweeps three models
  rather than failing on one every time;
- the two runs already stored are marked `valid: false` with the reason, so they
  appear under "Excluded runs" and never enter an average.

A near-baseline number that is really a rate artefact is worse than no number,
because it looks like a result. That is what the guard exists to prevent.

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

## openpilot cannot be measured here, and that one is final

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

So: `capabilities.min_input_hz: 10.0`, `navsim` undeclared, and the stored runs
marked invalid with the reason. A near-baseline number that is really a rate
artefact is worse than no number, because it looks like a result.

## Sample size

Two conclusions were overturned by running more scenarios, in opposite
directions:

| model | n=4 | full | |
|---|---:|---:|---|
| visionpilot | 0.640 | 0.447 (n=242) | over-read |
| alpamayo15 | 0.226 | 0.503 (n=60) | under-read |

`--limit` exists to test the pipeline — node startup, conversion, the scorer,
the CSV round trip. Runs below the full split are recorded with `valid: false`
so they cannot be averaged or ranked against a real one.

## Status

| model | EPDMS | n | note |
|---|---:|---:|---|
| drivejepa | 0.87968 | 242 | via proxy; the comparable figure |
| alpamayo15 | 0.65332 | 242 | with the adaptive frame selection |
| visionpilot | 0.44694 | 242 | |
| constant velocity | 0.25673 | 242 | reference baseline |
| openpilot | — | — | not measurable at 2 Hz; see above |

All five are the **242-scenario subset**, not navtest. The full-split fetch is
`tools/benchmarks/navsim/fetch_test_blobs.py`.

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

## A launcher bug found by the parallel smoke

Running two smokes at once produced two runs reporting a node on the same
port, with **identical scores**: drivejepa's recorded plans had two columns
(visionpilot's shape; drivejepa emits x, y, heading). drivejepa's health check
had passed against visionpilot's node, up on 9001 while drivejepa's own was
still loading, and its scorer then sent every request to the wrong model.

`/alive` says only that *a* paddock node is answering. Nodes now also answer
`/whoami`, the launcher refuses a port that answers as a different model, and
`PADDOCK_BASE_PORT` gives each concurrent run its own range. The bogus run is
stored with `valid: false` and the reason. This would have applied to any two
concurrent paddock runs, not only NAVSIM.
