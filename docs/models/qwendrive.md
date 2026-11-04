# Qwen-Drive 1.0

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [QwenLM/Qwen-Drive-1.0](https://github.com/QwenLM/Qwen-Drive-1.0) at `28091c1` |
| Adapter | [`adapters/models/qwendrive/`](../../src/paddock/adapters/models/qwendrive/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` |
| Plan | waypoints, 0.5 s apart, 5 s ahead |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [neuroncap](../benchmarks/neuroncap.md), [comma2k19](../benchmarks/comma2k19.md), [alpasim](../benchmarks/alpasim.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

Qwen-Drive-1.0-4B (Qwen team + HUST, September 2026, Apache-2.0) is a Qwen3.5-4B
vision-language model with a 1 B-parameter "Planning Expert" attached: a
flow-matching transformer that reads the VLM's attention caches and regresses
a trajectory. The VLM is unchanged, so it also answers driving questions. Paddock
runs the planning path only. Vendor repo: `Qwen-Drive-1.0` in `repos.yaml`
(`28091c1`), weights `Qwen-Drive-1.0-4B/` under `[roots.weights]`.

Adapter: `src/paddock/adapters/models/qwendrive/`. Run it with

```bash
paddock doctor --model qwendrive
paddock run --model qwendrive --bench kitscenes --limit 5      # smoke
paddock run --model qwendrive --bench navsim                   # full navtest, 10 Hz history
paddock run --model qwendrive --bench neuroncap --runs 10 --scene <official set>
paddock run --model qwendrive --bench comma2k19                # 10,165 keyframes, one camera
```

## What the model consumes and emits

From the vendor's `docs/data.md`, `docs/model.md` and `src/qwen_drive/scene.py`:

| | |
|---|---|
| cameras | three: front, front-left, front-right (`CAMERA_VIEWS`); a scene missing one raises |
| frames per camera | four, 0.5 s apart: t−1.5, t−1.0, t−0.5, t−0 |
| image size | a pixel budget snapped to a 32 px grid: current frame 921,600 px (~720p), history frames 174,080 px (~320p) |
| ego history | 16 poses (x, y, heading) at 10 Hz over 1.5 s, current ego frame, x forward, y left, last row (0, 0, 0); plus velocity [16, 2] and acceleration [16, 2] |
| ego status | `[vx, vy, ax, ay, left, straight, right, unknown]` |
| navigation | integer command 0 straight / 1 left / 2 right, written into the prompt text |
| output | 50 poses of (x, y, heading) at 10 Hz, 5 s, ego frame; flow matching from seeded noise, so deterministic |

Two planners ship. `planner-sft` is imitation-trained and plans with or without
a written rationale; `planner-rl` is reward-optimised and, the vendor says, must
run in reasoning mode. Paddock's default is **`planner-sft` in direct mode**: one
VLM pass then the expert, no text generated. The adapter refuses `planner-rl`
with direct mode rather than record a number nobody's model produced. Vendor
NAVSIM v1.1 navtest PDMS: 87.8 (sft, direct), 88.2 (sft, reasoning), 90.7 (rl).

## How each benchmark's stream becomes that scene

Every benchmark hands the adapter one frame per observation. The adapter keeps
the recorded frames, poses and CAN bus, and at each plan:

- picks the four camera frames nearest by timestamp to t−1.5/−1.0/−0.5/0 s.
  A 10 Hz source contributes every fifth frame, a 2 Hz source every frame.
  Before 1.5 s of a closed-loop run has been seen, the oldest frame stands in;
  `/diagnostics` `frames_repeated_total` counts those slots.
- resamples the 16 poses onto a 0.1 s grid **by timestamp**, not by request
  count: neuroNCAP's priming steps arrive 0.5 s apart and only its loop steps at
  0.1 s. Velocity and acceleration come from the CAN bus where the source
  records them; KITScenes records speed only, so acceleration is differenced.
- maps paddock's command to both of the model's encodings: LEFT → (1, [1,0,0,0]),
  STRAIGHT → (0, [0,1,0,0]), RIGHT → (2, [0,0,1,0]), FOLLOW_REFERENCE → (0, [0,0,0,1]).
  The prompt vocabulary has no "unknown", so FOLLOW_REFERENCE reads as straight
  in the text and as unknown in the ego status, as NAVSIM's own records do.
- keeps the 0.5 s grid of the 50-pose output, indices 4, 9, … 49: ten
  waypoints over 5 s with heading — the whole plan. NAVSIM scores the first
  eight; neuroNCAP takes the first six; KITScenes truncates at 3 s; comma2k19
  scores every point it is given, against horizons it labels honestly, which is
  why the grid runs to the end rather than stopping at NAVSIM's 4 s.

| benchmark | frames | poses | command | note |
|---|---|---|---|---|
| KITScenes | 49 at 10 Hz, one `/infer_sequence`; ring cameras front, front-left, front-right | real, 4.8 s; speed only | always STRAIGHT | ring cameras sit at +60.9/−58.5° with ~87° HFOV against the ~70° nuPlan cameras the model trained on; zero-shot, not warped |
| NAVSIM v2 / v1 | `native_input_hz: 10` selects nuPlan's 10 Hz path (`[benchmarks.navsim] nuplan`): 16 observations over 1.5 s, frames picked every 0.5 s | 16 real poses with velocity, acceleration, yaw rate | NAVSIM's one-hot, round-tripped exactly | the model's own NAVSIM cameras (`cam_f0/l0/r0`) and history rate |
| neuroNCAP | one 1600×900 frame per 0.1 s tick, three cameras rendered | from `ego2world` each tick | the engine's | the adapter plans every tick, as every other model here |
| comma2k19 | 160 observations at 20 Hz per sample, one `/infer_sequence`; **one** camera, replicated into all three view slots | real, 8 s of 20 Hz pose; CAN speed, acceleration differenced | always STRAIGHT — the dataset ships no route | zero-shot on a rig nothing in the corpus resembles; see below |
| AlpaSim | every 50 ms step, three f-theta cameras; frames picked 0.5 s apart, refused if off their slot | nearest 20 Hz step per 0.1 s row; forward and sideways speed, acceleration differenced with the turning term | from the route, held over three calls | the native driver's inputs; see [On AlpaSim](#on-alpasim-through-the-bridge) |
| Bench2Drive | one frame per 20 Hz tick by default, and one per 10th tick in the native-rate campaign; the three trained views, `CAM_FRONT` / `CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` | from `ego2world` | route-derived | driven: 50 routes, DS 12.92 / RC 50.19, `results/bench2drive/qwendrive/pooled50-*`; `duration_s` averages 3457 s per route, the slowest row in that table. Re-driven at 2 Hz it scores 17.08 over the 49 routes that scored at both rates — [native-rate campaign](../benchmarks/bench2drive-native-rate.md) |

## On AlpaSim, through the bridge

`paddock run --model qwendrive --bench alpasim` drives the model on the settings
and inputs of Qwen-Drive's native AlpaSim driver (adapter 0.1.4), so a run
through paddock is the same experiment as the native runs. The settings are the
`qwendrive` entry in `alpasim/harness.py`'s `DRIVER_PROFILES` and its driver
config `paddock_bridge_qwendrive.yaml`: four f-theta cameras at 20 Hz (the
fourth, the tele camera, is rendered as natively and never sent), a 3 s replayed
warm-up with the driver called, 400 steps of 50 ms, the 30 scenes pinned to
their 26.01 artifacts.

A body carrying a scene id is how the node knows it is on AlpaSim; only that
profile's bridge sends one. There the node builds its input the way the native
driver did, not the way it does for the other benchmarks:

| Input | On AlpaSim |
|---|---|
| when the model is called | every 1.5 s (30 steps) from the end of the warm-up, on a grid; between calls the bridge re-times the last plan and re-anchors it to the car |
| what the node hears | every 50 ms step: warm-up frames and the steps between calls go to `/observe`. A frame cadence other than 50 ms is refused |
| frames | the newest, and the frame nearest each of 0.5, 1.0, 1.5 s before it; a frame more than 50 ms off its slot is refused rather than stood in for |
| ego rows | the recorded step nearest each 0.1 s slot (not interpolated), placed by the full 3-D relative pose |
| velocity | forward and sideways speed as the simulator reports them, each row in its own frame. AlpaSim reports zero velocity and yaw rate while it replays the warm-up; the bridge replaces those from the poses when they move faster than 1 m/s, so a car really standing still keeps its zero |
| acceleration | the change in velocity over 0.1 s plus the turning term: forward `dv_fwd/dt − ω·v_lat`, sideways `dv_lat/dt + ω·v_fwd`. The simulator also reports an acceleration; it is not used, because the native driver did not use it |
| command | from the route AlpaSim holds, by the native three-answer rule: the point 20 m along the route, the net heading change to it and the same change median-filtered must all lie more than 3 m to one side to say LEFT or RIGHT. No route, or answers on both sides, is "unknown" (one-hot unknown, text "straight") |
| command hold | a new command must come back on three model calls in a row before the model is told it. The hold is updated only when the model is called, as natively, so a turn reaches the model 3 s after the route first shows it, and a change that lasts less than three calls (4.5 s of driving) never reaches it. This is kept on purpose because it is what the native runs did; the native code's comment says 150 ms, which is not what it does |
| noise seed | `42 + sha256(scene id)[:8] mod 1,000,000`, per scene, as natively. `QWENDRIVE_SEED` is not read on AlpaSim |
| output | all 50 points, 0.1 s apart (the profile sets the bridge's `--waypoint-dt 0.1`, and the bridge refuses a plan spaced otherwise) |

A short, zero or backwards plan goes to the simulator as it is. The native
driver refused any plan reaching less than 1.5 m forward, to catch output left
in the model's internal 0–1 units, and that refusal ended three of its scenes
where the car was simply braking to a stop. Here the units are checked once,
when the node starts (below), and a plan reaching less than 1.5 m is logged
with the speed it was planned at and counted in `/diagnostics`
(`short_plans_total`). Each call's command, the route's own command before the
hold, the seed, the speed and the forward reach go back in the plan's `extras`
and land in the scene's per-replan log.

Differences from the native runs that remain:

- **Attention.** Paddock runs `sdpa`; the native runs used
  `flash_attention_2`. Native measured the two bit-exact on one call only.
- **Python.** Paddock's environment is Python 3.12, the vendor's; the native
  environment was 3.10.
- **The first two calls.** Natively the model was also called at warm-up steps
  0 and 30, whose plans AlpaSim does not drive; here those frames are observed.
  The command hold therefore starts at the first call after the warm-up rather
  than at step 30.

The run on the 30 scenes is `results/alpasim/qwendrive/20261027T215528-d231d8`:
valid, one rollout per scene, `scene_score` **0.7001** (progress 0.787), five
at-fault collisions and two off-road events. The native runs on the same
simulator image scored 0.6672, three of their scenes ending on the forward-reach
refusal above; with one rollout per scene the two are level. The native runs
were made by the native driver outside paddock and are not in `results/`,
so the 0.6672 cannot be re-checked from this repository.

The run's `run.json` names paddock commit `a074fe5`. That is this route's
build commit before it was rebased onto `main`, and no branch holds it. In the
commit that added this route to `main` (`caffc85`) the files this route changed
differ from it only in comments, so check out that one to reproduce the run.

## Filling three views from one camera

`DrivingScene.__post_init__` **raises** on a missing view, so `<FRONT VIEW>`,
`<FRONT LEFT VIEW>` and `<FRONT RIGHT VIEW>` all have to be filled from whatever
the source has. On comma2k19 that is one windscreen camera. `QWENDRIVE_VIEWS`
picks how:

- `ring` (default) — one real camera per slot. The model as trained, and what
  KITScenes, NAVSIM and neuroNCAP supply.
- `replicate` — the true forward frame in all three slots. Geometrically honest
  in that each view really is a forward camera, but it adds no lateral context.
- `crop` — overlapping 55 %-width thirds of the road camera, all three cut to
  one width so the model does not read one wide view beside two narrow ones.
  Real lateral content, wrong extrinsics: a crop is not a rotated camera.

`replicate` wins, measured rather than argued: 1.47 city / 2.00 highway ADE
against `crop`'s 1.78 / 2.13 on the reference investigation's segments. The
model's learned cross-view geometry is load-bearing, and violating it costs more
than honest redundancy.

This is the opposite of the choice `alpamayo15/policy.py::slots_for` makes, and
that is not an inconsistency. Alpamayo tolerates a missing slot, so dropping one
says "there is no +55° view"; repeating the centre frame there would say "here
is one" and lie. Qwen-Drive does not tolerate it, so the choice is between two
lies and the measured one wins.

The mode also decides what the policy **declares**: under `replicate` or `crop`,
`policy.cameras` is `CAM_FRONT` alone. That keeps `paddock verify`'s
`pair.declared_cameras_available` honest and stops a closed-loop simulator
rendering two cameras nobody reads. It is a property of the *source*, not of the
model, and paddock has no per-benchmark model environment — so a site that runs
both a ring source and a monocular one needs two registry entries, the way
`drivejepa` and `drivejepa_free` split.

**No route, and it costs something.** comma2k19 ships no turn-by-turn signal, so
every observation carries `Command.STRAIGHT` and the model plans every sample
under "GO STRAIGHT". Feeding the command a route planner *would* have issued —
read off the recorded future, hence diagnostic-only and never scored — improved
the reference's city ADE 1.47 → 1.28 and highway 2.00 → 1.77. `/diagnostics`
records the command per sample so a residual on curves can be attributed to a
missing input rather than to planning.

## Environment

`setup.sh` builds `{repo}/.venv` on Python 3.12 with the vendor's pins (torch
2.8.0 cu128, transformers 5.14.1). Two things are deliberate:

- **flash-attn is not installed.** It does not build for this card's sm_120
  (the alpamayo adapter hit the same wall). The model supports
  `attn_implementation="sdpa"` in both the VLM and the expert; `QWENDRIVE_ATTN`
  sets it.
- **causal-conv1d is attempted, not required.** It is a CUDA extension the
  Qwen3.5 hybrid-attention layers use; if it fails to build, transformers falls
  back to a torch implementation, slower and numerically the same. The setup log
  says which happened.

`QWENDRIVE_CHECKPOINT`, `QWENDRIVE_PLANNER`, `QWENDRIVE_MODE`,
`QWENDRIVE_VIEWS`, `QWENDRIVE_NUM_SAMPLES`, `QWENDRIVE_ATTN` and
`QWENDRIVE_SEED` are the knobs; `model.yaml` sets all but the last.
`load()` checks the planner's weights against the released sha256 for its
directory name (`planner-sft` `dcf5989e…`, `planner-rl` `74478eb7…`; the two
ship identical configs), then makes one call at the real input shape before the
node reports ready. That call pays the first call's ~14 s of kernel set-up off
the simulator's clock, and it is the units check: three grey views and a car
that has driven straight at 10 m/s must get a plan reaching at least 10 m
forward, or the node refuses to start. `num_samples > 1` executes the medoid draw;
the vendor's best-of-N figures select with the ground truth and are not
reproducible here.

## Training exposure

The vendor states the planner was trained on trajectories "standardized from
multiple public driving datasets" and reports NAVSIM navtest as its headline
number, so `trained_on` declares NAVSIM `navtrain`; see the Qwen-Drive × navsim
section of [comparison.md](comparison.md) for what that does to the NAVSIM
column. Its VLM also trained on nuScenes train and on driving-VQA sets whose
training splits hold nuScenes val scenes, 12 of neuroNCAP's 14 among them, so the
nuScenes and neuroNCAP cells are adapted too (comparison.md, Qwen-Drive 1.0 ×
nuscenes). KITScenes and comma2k19 are zero-shot — comma2k19 doubly so, since
nothing the report names is a windscreen-mounted mono camera on a Californian
highway. Bench2Drive is zero-shot as far as the report says, but one of its VQA
sets, DriveLM, has a CARLA part, and which part was used is not stated.

## Results

All seven on `planner-sft`, direct mode, one seeded sample. The first five ran
on one RTX PRO 6000; comma2k19 on an H200, seven nodes,
`QWENDRIVE_VIEWS=replicate`; the Bench2Drive row pools five source runs over
four workers.

| benchmark | n | score | wall clock | run |
|---|---:|---:|---:|---|
| KITScenes ADE@3s ↓ | 223 | 0.806 m | 6.6 min | `20260914T220841-5a91fd` |
| nuScenes cumulative L2 ↓ | 5,119 | 0.450 m | 88 min | `20260915T190350-47479e` |
| NAVSIM v2 EPDMS ↑ | 12,146 | 0.840 | 4.0 h | `20260914T221802-80457c` |
| NAVSIM v1 PDMS ↑ | 12,146 | 0.876 | 9 min, no GPU | `20260915T021715-02aab4` |
| neuroNCAP score ↑ | 200 | 2.49 | 7.7 h | `20260915T023604-12d5eb` |
| comma2k19 ADE@3s ↓ | 10,165 | 1.159 m | 2.0 h | `20260916T220846-971379` |
| Bench2Drive driving score ↑ | 50 | 12.92 | 3,457 s per route | `pooled50-40c29d24c37d542a-f250e0` |
| Bench2Drive driving score ↑, driven at 2 Hz | 49 | 17.08 | — | `native50-2hz-084418258d833a9f-9b2bc9` (`valid: false`: not the 20 Hz protocol) |

The comma2k19 row replaces an earlier one, and the earlier one is still here.
`20260915T214115-405f64` (ADE@3s 1.016) was produced by the pre-merge adapter,
which emitted all 50 poses at `waypoint_dt` 0.1 s; `score()` aligns the ground
truth onto *the prediction's* grid, so the same predictions on the merged
adapter's 0.5 s grid give 1.159, and the 0.1 s spacing also turned 4 cm of output
quantisation into 1932 m/s³ of "jerk", which is why that row carried `-` in the
comfort columns. The re-sweep was the right call; deleting the old record in the
merge that did it was not, so it is restored and marked `superseded_by` instead
— `paddock report` leaves it out, and the swap is visible in `results/` rather
than only in a merge message.

The v1 figure is worth one line on its own: the vendor reports 87.8 PDMS on this
split for exactly this configuration, and paddock measures 87.6. That agreement
covers the camera mapping, the pose history, the eight poses NAVSIM reads and the
scorer together, which is a stronger check than any one of them alone. It is
also an in-distribution number — the model declares `navtrain`.

nuScenes is the widest margin: lower than every other model on all seven
columns, collision included, and the first model here to beat the
constant-velocity baseline outright. Not zero-shot: its VLM trained on nuScenes
train, and on VQA sets whose training splits hold val scenes (Training exposure,
above).

## Why neuroNCAP is the exception

First or near-first on every open-loop column, fourth of five closed-loop. The
recorded artifacts say what the difference is, and it is not braking.

Over all 50 frontal episodes, per model, the mean peak lateral offset of the 3 s
plan against the collision rate:

| model | collisions | speed change over the episode | mean peak \|lateral\| |
|---|---:|---:|---:|
| alpamayo15 | 32% | −2.29 m/s | 10.44 m |
| drivejepa | 78% | −4.67 m/s | 1.80 m |
| openpilot | 84% | −4.99 m/s | 1.00 m |
| **qwendrive** | **86%** | **−2.81 m/s** | **1.22 m** |
| visionpilot | 90% | −2.27 m/s | 1.91 m |

Speed change is the last speed minus the first, from each episode's
`ego_poses.json`. Drive-JEPA's row is its run after the camera fix.

Qwen-Drive brakes about as hard as Alpamayo, and less hard than OpenPilot: on
`frontal-0103` seed 5 it goes from 8.1 to 2.7 m/s with its planned reach
collapsing from 21 m to under 5 m. Its planned lateral offset never passes
1.2 m. OpenPilot and Drive-JEPA brake hardest of the five and still collide in
84% and 78% of episodes. In a staged head-on the closing vehicle arrives whatever the ego's
speed, so the escape is lateral, and Alpamayo is the only model that takes it.

**This is the model, not the adapter.** Through the same code path on NAVSIM the
model produces lateral offsets up to 19.92 m at 4 s, 286 plans beyond 10 m, so
it can express evasion and does at junctions. Across all 200 neuroNCAP episodes
its peak is 6.40 m and its frontal mean is 1.22 m.

Interpretation, not measurement: neuroNCAP stages collisions that do not occur
in ordinary driving logs, and this model is trained on recordings of ordinary
driving. It answers a closing obstacle the way a careful driver answers a slow
one. Note the sample is five models with a single point at high lateral, so read
it as where to look next rather than as a fitted relationship.

Side-by-side clips of the frontal episodes, and the figure behind that table,
are reproducible from the recorded runs — the sweep already writes one mp4 per
stream per episode (`[benchmarks.neuroncap] video`).

## The 50-route Bench2Drive run

Five source runs over four workers, 15-16 Sep 2026, `planner-sft` in direct mode
with one sample, `QWENDRIVE_ATTN=sdpa`, the three trained cameras and paddock's
own dual-PID tracker — the same tracker every other model in that table drove
through, so the row compares planners and not controllers. Route 24333's first
two attempts are `Failed - Agent crashed` (a `prepare_history` edge-padding bug,
fixed before the rerun); they are not measurements and the ingest drops them, as
it does for every model.

The result is the weakest row in the table by a wide margin: 41 collisions per
progress km against 3.7 for Drive-JEPA, and 45 of 50 routes with a collision. It
is a zero-shot NAVSIM-trained planner meeting CARLA's 20 Hz closed loop, which is
what the row should be read as.

**The row is ingested, not re-drivable from this tree.** `model.yaml` declares
`kitscenes`, `nuscenes`, `navsim`, `navsim_v1` and `neuroncap`, not
`bench2drive`, so `paddock run --model qwendrive --bench bench2drive` is refused
here; `results/bench2drive/qwendrive/pooled50-*` was written by `paddock ingest`
from the per-route `results.json` trees, like every other row in that table.

## Cost

No latency is published for the model. Measured here, direct mode, one sample:

| | |
|---|---|
| checkpoint load | 121 s cold, 17 s warm |
| inference | ~1.0–1.5 s per plan |
| VRAM | ~15 GB |

`/diagnostics` reports `infer_seconds_mean` live. Reasoning mode adds a
256-token decode per plan and multiplies closed-loop cost several times over;
nothing here was run that way.
