# Qwen-Drive-1.0 in paddock

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
| Bench2Drive | one frame per 20 Hz tick by default, and one per 10th tick in the native-rate campaign; the three trained views, `CAM_FRONT` / `CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` | from `ego2world` | route-derived | driven: 50 routes, DS 12.92 / RC 50.19, `results/bench2drive/qwendrive/pooled50-*`; `duration_s` averages 3457 s per route, the slowest row in that table. Re-driven at 2 Hz it scores 17.08 over the 49 routes that scored at both rates — [native-rate campaign](../benchmarks/bench2drive-native-rate.md) |

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
`QWENDRIVE_SEED` are the knobs; `model.yaml` sets all but the last. `num_samples > 1` executes the medoid draw;
the vendor's best-of-N figures select with the ground truth and are not
reproducible here.

## Training exposure

The vendor states the planner was trained on trajectories "standardized from
multiple public driving datasets" and reports NAVSIM navtest as its headline
number, so `trained_on` declares NAVSIM `navtrain`; see the Qwen-Drive × navsim
section of [comparison.md](comparison.md) for what that does to the NAVSIM
column. Bench2Drive (CARLA), KITScenes, neuroNCAP (nuScenes, rendered) and
comma2k19 are zero-shot — and comma2k19 doubly so, since nothing the report
names is a windscreen-mounted mono camera on a Californian highway.

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
constant-velocity baseline outright. Zero-shot — nuScenes is not among the
datasets the model declares.

## Why neuroNCAP is the exception

First or near-first on every open-loop column, fourth of five closed-loop. The
recorded artifacts say what the difference is, and it is not braking.

Over all 50 frontal episodes, per model, the mean peak lateral offset of the 3 s
plan against the collision rate:

| model | collisions | speed change over the episode | mean peak \|lateral\| |
|---|---:|---:|---:|
| alpamayo15 | 32% | −2.31 m/s | 10.44 m |
| drivejepa | 78% | −5.47 m/s | 0.11 m |
| openpilot | 84% | −1.88 m/s | 1.00 m |
| **qwendrive** | **86%** | **−2.48 m/s** | **1.22 m** |
| visionpilot | 90% | −2.45 m/s | 1.91 m |

Qwen-Drive brakes about as hard as Alpamayo, and harder than OpenPilot: on
`frontal-0103` seed 5 it goes from 7.3 to 2.3 m/s with its planned reach
collapsing from 21 m to under 5 m. Its lateral offset stays near zero
throughout. Drive-JEPA brakes hardest of anyone and still collides in 78% of
episodes. In a staged head-on the closing vehicle arrives whatever the ego's
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
