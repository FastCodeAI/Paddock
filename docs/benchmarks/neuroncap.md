# neuroNCAP

Closed-loop safety scoring. A NeuRAD render of a real nuScenes sequence is
driven by the model under test while a staged actor is manoeuvred onto a
collision course; the score reflects how much of the impact the model avoided.
0 is a full-speed collision, 5 is avoidance.

## Running it

The renderer is model-independent and takes minutes to load a scene, so one
instance serves a whole sweep. Start it and leave it up:

```bash
paddock renderer --scene 0103          # foreground; use another shell below
paddock run --bench neuroncap --model openpilot --scene 0103 --scenarios stationary
paddock run --bench neuroncap --model all --scene 0103 --runs 1
```

Roughly 1.5–2.5 minutes per scenario per model on this hardware, plus node
startup — which for Alpamayo is a 21 GB checkpoint read and can take far longer
than the run itself.

## Two rates, deliberately separate

| | Value | Why |
|---|---:|---|
| simulation step | 0.1 s | openpilot needs 20 Hz input and reads a 0.5 s step as a parked car. Every model is driven at the same rate so control authority is equal. |
| waypoint grid | 0.5 s | The collision lookahead and reference-speed calculation both assume it. |

`--engine.waypoint-dt` exists to keep them apart; see `patches/neuro-ncap/0001`.
`tests/test_neuroncap_harness.py` asserts every scenario file still steps at
0.1 s, because drifting back to 0.5 s would silently compare models at a rate
one of them cannot use.

## What the simulator actually does with a plan

Answers to questions that recur, with the line that decides each. The short
version: the model is called at 10 Hz, returns six waypoints on a 0.5 s grid,
and an LQR tracker turns those into acceleration and steering for one 0.1 s
integration before it is asked again.

### How often is the model called, and is it the same for every model?

**10 Hz -- one call per 0.1 s of simulated time, identical for all four
models.** The engine's loop makes exactly one `run_model` call per iteration and
then advances the clock by `self.timestep` (`engine.py:281`), which is the
scenario file's `time-step` (`engine.py:94`).

The vendor default is 0.5 s, i.e. 2 Hz (`components/scenario.py:21`). The
tracked files under `scenarios_10hz/` override it to 0.1, and
`tests/test_neuroncap_harness.py` asserts all 20 still do.

Nothing per-model changes the rate. openpilot duplicates each frame internally
to feed supercombo at the 20 Hz it expects
(`adapters/models/openpilot/policy.py:387`), but that happens inside one call.

10 Hz is *simulated* time. Wall clock in the 2026-09 sweeps was about 1.2-1.6 s
per step, dominated by rendering, so an 8 s scenario is 80 model calls and
several minutes of real time.

### How do waypoints on a 0.5 s grid drive a 0.1 s simulation?

The six waypoints are stamped t+0.5 ... t+3.0 (`engine.py:269-270`), given a
heading by finite difference between neighbours, and wrapped in an
`InterpolatedTrajectory` (`components/vehicle_model.py:152-195`). That turns the
0.5 s grid into a continuous function of time, which the controller samples
between the model's waypoints rather than only at them.

nuPlan's `LQRTracker` then produces two commands -- longitudinal acceleration
and tire steering rate -- looking 1 s ahead in 0.1 s slices
(`discretization_time 0.1`, `tracking_horizon 10`).

**There is no brake channel.** Braking is negative acceleration, clamped to
±6.0 m/s² (`components/vehicle_model.py:199-201`). If the last waypoint is
within `STANDING_STILL_RADIUS` (1 m) the tracker is bypassed for a stopping
controller, which is why an all-zero trajectory reads as a commanded emergency
stop rather than "no plan".

### Is it one 0.1 s step before the model is called again?

Yes. `_vehicle_model_sim_steps = timestep / 0.1` (`engine.py:106`), which is
exactly 1 at 0.1 s. Each cycle is: render, one model call, one LQR solve, one
0.1 s integration.

At the vendor's 0.5 s default it would be 5, so the car would follow one stale
plan through five control updates before seeing anything new.

**The model plans 3 s ahead and only the first 0.1 s is ever executed.** The
rest exists to give LQR its lookahead: the plan is a tracking reference, not a
path that gets driven.

### How do the commands become the next frame the model sees?

They do not reach the renderer. `propagate_state` integrates a kinematic bicycle
model 0.1 s forward (`components/vehicle_model.py:203-205`) to a new
`ego2world`, and that pose becomes a camera extrinsic --
`pose = ego2world @ camera2ego` (`engine.py:165-171`) -- which NeuRAD renders
from the trained hash grids.

**The controls move a virtual camera through a fixed scene.** That is also why a
scene renders without its lidar: the geometry is in the checkpoint, not rebuilt
per frame. See [A scene needs at least one lidar keyframe](#a-scene-needs-at-least-one-lidar-keyframe).

`accel_time_constant` and `steering_angle_time_constant` are both **0**
(`components/vehicle_model.py:70-72`), so there is no actuator lag: a commanded
acceleration takes effect immediately.

### Do models output different numbers of waypoints?

Natively yes; at the interface no. neuroNCAP requires **exactly six waypoints at
0.5 s**, enforced in `Trajectory.to_wire` (`policy/protocol.py:429-441`): fewer
raises, more are dropped, because the engine builds six timestamps and indexes
them positionally.

| model | native output | adaptation |
|---|---|---|
| Drive-JEPA | 8 poses at 0.5 s, NAVSIM's grid | none; first six taken |
| openpilot | 8 points on `arange(1, 9) * 0.5` | interpolated onto the grid |
| VisionPilot | lane curvature + lead object | integrated to a path, resampled |
| Alpamayo 1.5 | its own horizon | resampled by timestamp |

### Does driving smoothness depend on every 0.1 s output?

In principle yes -- a model that commanded full throttle on one call and full
brake on the next would drive jaggedly. Three things bound it: LQR penalises
steering rate (0.05) and acceleration (1.0) against tracking error over a 1 s
horizon; acceleration is clamped to ±6 m/s²; and only 0.1 s of any plan is
executed, so a single bad frame moves the car centimetres.

**Nothing scores smoothness.** The NCAP score is a flat 5 for avoidance and
`4 · max(0, 1 − v_impact / v_reference)` otherwise, so a model could oscillate
badly and still score 5.0 if it avoids. NAVSIM has `history_comfort` and
`two_frame_extended_comfort`; neuroNCAP has no equivalent, which means
jaggedness shows up only in the per-run clips -- one reason to keep them.

### Does ego history help, and where does it come from?

`ModelInput` carries **one frame**: images, `ego2world`, canbus, timestamp,
command, calibration. No history. A model that needs it accumulates it across
calls itself, which is what `reset()` clears between runs.

* **openpilot** fills a 24-deep feature buffer, manufacturing 100 frames on the
  first call when no history has been supplied.
* **Alpamayo 1.5** differences 16 ego poses to derive its own speed.
* **Drive-JEPA** selects history *by timestamp*, not by taking the last N
  frames, so a simulator stepping faster than 0.5 s still yields ~1.5 s of
  context. Its `diagnostics()` reports the depth actually achieved, because
  otherwise that degrades silently.

History helps indirectly: successive plans agree better, so the tracker corrects
less. **No model here has an explicit temporal-consistency term** -- what
stability there is comes from the controller and the vehicle, not from the
planner being asked to be smooth.

## Configuration

```toml
[benchmarks.neuroncap]
python          = ".../envs/ncap/bin/python"   # the engine's own environment
repo            = ".../neuro-ncap"
renderer_url    = "http://127.0.0.1:8000"
renderer_python = ".../envs/neurad/bin/python" # a *different* env from the engine
renderer_cwd    = "..."
neurad_studio   = ".../neurad-studio"
checkpoints     = ".../checkpoints"            # one per scene
```

A scene needs both a scenario file (tracked, under `scenarios_10hz/`) and a
renderer checkpoint (site-local). `preflight` checks both before any model
loads — `stationary/0061.yaml` exists with no checkpoint for scene 0061, and
would otherwise fail deep into a sweep.

## Reproduction status

**Paddock's adapters reproduce the original nodes exactly.** Run head to head on
2026-09-01, scene 0103, stationary, 1 run:

| | ncap_score |
|---|---|
| paddock `visionpilot` adapter | 2.8269616956051293 |
| original `visionpilot_server.py`, invoked the original way | 2.8269616956051293 |

Identical to full float precision. That is the claim M5 set out to test, and it
holds: the conversion to the `Policy` contract changed the seams and nothing else.

`openpilot` likewise returned 5.0 on stationary-0103, matching its recorded value.

## Reproducing the recorded August runs

| model | scenario | recorded | paddock, 2026-09-01 | |
|---|---|---:|---:|---|
| openpilot | stationary | 5.0 | 5.0 | match |
| drivejepa | stationary | 5.0 | 5.0 | match |
| drivejepa | frontal | 2.0723314376255617 | 2.0723314376255617 | **exact** |
| visionpilot | stationary | 3.145 (27 Aug) / 3.388 (29 Aug) | 2.827 | **differs — explained below** |

(Both sides of the drivejepa match shared a bug, fixed since: the August node and
the adapter both placed the cameras relative to nuScenes' LIDAR_TOP, so the model
never read the image. The match still shows the pipeline was unchanged; it says
nothing about what drivejepa sees. See the adapter's *Frame* note.)

drivejepa reproduces to full float precision, which is the strongest evidence
available that the pipeline as a whole — renderer, engine, node, scoring — is
behaving as it did in August.

### Why visionpilot differs, and why it is not a drift

It initially looked like environmental drift: paddock returned 2.827 where
August recorded 3.145, and the *original* `visionpilot_server.py` invoked the
original way returned 2.827 too. Two candidate explanations survived that —
either something beneath both had changed, or the code itself had.

File timestamps settle it:

```
drivejepa_server.py    modified 2026-08-27 03:25   last recorded run 05:37   unchanged since
visionpilot_server.py  modified 2026-08-29 05:02   last recorded run 03:18   edited 1h44m after
```

The VisionPilot node was revised after the last run that scored it. **The
current code has never been scored**, and 2.827 is simply what the newer
implementation does — not a regression, and nothing environmental. drivejepa,
untouched since its run, reproduces exactly, which rules out the renderer, the
engine, the GPU and the libraries beneath them.

Applied to all four nodes, the same comparison predicts which reproduce:

| model | node modified | last scored run | prediction | outcome |
|---|---|---|---|---|
| openpilot | 08-27 03:13 | 08-27 23:28 | unchanged, should match | matched |
| drivejepa | 08-27 03:25 | 08-28 00:00 | unchanged, should match | matched exactly |
| alpamayo | 08-27 04:12 | 08-27 07:37 | unchanged, should match | **stationary matched; frontal did not — see below** |
| visionpilot | **08-29 05:02** | 08-29 03:18 | **newer, will differ** | differed |

The prediction held for the deterministic models. It does not apply to alpamayo,
for the reason in the next section — the assumption behind it is that a run is
reproducible at all.

This is the failure mode version pinning exists to prevent, and it is precisely
why this code is now in git: from here, a score change has a commit beside it.
The August visionpilot figures should be treated as belonging to a superseded
implementation, and re-measured rather than compared against.

**If a score moves unexpectedly, check this first.** Compare the adapter's git
history against the run's `started_utc` before suspecting the renderer, the GPU,
or the benchmark.


## Single-run scores are not comparable for stochastic models

Alpamayo returned 5.0 on frontal-0103 where August recorded 0.0 — the full width
of the scale. Its node was not edited, so the timestamp check predicted a match.
The prediction was wrong because the premise was: **alpamayo does not produce the
same trajectory twice.**

It samples with `temperature=1.0`, `top_p=0.95` and no seed. The five recorded
seeds for that one scenario:

| model | run_0 | run_1 | run_2 | run_3 | run_4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| alpamayo | 0.000 | 2.084 | 1.519 | 1.885 | 5.000 | 2.098 | **1.817** |
| openpilot | 1.862 | 2.610 | 2.187 | 2.004 | 2.389 | 2.210 | 0.298 |

Today's 5.0 is exactly run_4; August's 0.0 is exactly run_0. Both are single draws
from the same distribution, and neither is evidence about the model.

Consequences for anyone using this benchmark:

* **A single run is meaningless for alpamayo.** At sd 1.817, distinguishing it
  from openpilot's 2.210 to within ±0.25 would take roughly 200 runs.
* **Their reported means are statistically indistinguishable** on this scenario
  (2.098 against 2.210) despite looking a scale apart on any single run.
* Deterministic models (openpilot, drivejepa, visionpilot) reproduce exactly, so
  `--runs 1` is adequate for them and wasteful to exceed.

The fix is not to seed alpamayo into false determinism — the sampling is part of
the model — but to report a mean and spread over enough runs, and to say n.
Paddock stores per-episode records, so the spread is available rather than
collapsed into one number.

## Resolved: the two openpilot paths disagreed on traffic convention

openpilot sets `traffic_convention[int(is_rhd)] = 1`, where `is_rhd` means
right-hand *drive* — wheel on the right, i.e. left-hand traffic.

| path | value before | implies |
|---|---|---|
| open-loop KITScenes harness | `[1, 0]` | left-hand-drive / right-hand traffic |
| closed-loop node | `[0, 1]` | right-hand-drive / left-hand traffic |

They cannot both be right. KITScenes is Germany, and every neuroNCAP scene in the
official set is `boston-seaport` or `singapore-*` — the ones evaluated here are
Boston, so left-hand-drive, right-hand traffic. `[1, 0]` is correct for both, and
the closed-loop value had been telling the model it was driving a right-hand-drive
car in Boston.

**Fixed in `12f3250`**, committed 2026-09-01T00:44:56Z. Both paths now pass
`[1, 0]`, and `tests/test_openpilot_traffic_convention.py` parses the constant out
of both files and fails if they ever disagree again — they are separate constants
in separate files, because the two paths load openpilot differently, which is how
they came to disagree in the first place.

### Every recorded closed-loop openpilot score predates the fix

| run | started | vs fix |
|---|---|---|
| `20260831T235716-88a4e0` | 2026-08-31T23:57:16Z | before |
| `20260901T004005-88a4e0` | 2026-09-01T00:40:05Z | before, by 5 minutes |
| `legacy-*` (four runs) | August, undated | before |

There is no closed-loop openpilot number on record that was produced with the
correct traffic convention. The README's neuroNCAP row for openpilot — mean 3.80,
47% collision rate — is a measurement of the misconfigured model, and the size of
the error is unknown until it is re-run. It is not a small knob: it changes which
side of the road the network believes traffic is on, and the side scenario is
laterally asymmetric.

openpilot's rows are re-measured as part of the full-set sweep below; until that
lands, treat the recorded closed-loop openpilot figures as superseded.

## The evaluation set, and how many runs

### 20 scenario–scene pairs

The official set is transcribed from the vendor's own job arrays
(`scripts/arrays/ncap_slurm_array_*.txt`) into `OFFICIAL_SET` in `harness.py`:

| scenario | scenes | |
|---|---:|---|
| stationary | 10 | 0099, 0101, 0103, 0106, 0108, 0278, 0331, 0783, 0796, 0966 |
| frontal | 5 | 0103, 0106, 0110, 0346, 0923 |
| side | 5 | 0103, 0108, 0110, 0278, 0921 |

20 pairs over 14 distinct scenes. Sequence 0061 appears in the stationary array
carrying `# mini set (do not use for eval)` and is excluded; it is also the one
scene with no renderer checkpoint on this machine.

The files under `scenarios_10hz/` are the upstream scenario files with
`time-step: 0.5` changed to `0.1` and nothing else — verified one line per file
at porting time. `preflight` refuses a sweep with any of the 20 missing, and
`tests/test_neuroncap_harness.py` pins the set both ways: no file missing, and
no file present that is not in it.

Not every scene is in every category — 0099 is stationary-only — so `episodes()`
skips a (scene, scenario) pair with no file when `--scenarios` is defaulted.
When `--scenarios` names a category explicitly, a missing pair raises instead:
asking for something that does not exist should not quietly score less.

### Runs: 10, not 50

Upstream runs 50 (`RUNS=50` in `single_machine_docker_run_eval.sh`). Paddock's
sweeps use 10, which is a subsample of that protocol rather than a different one:
the engine seeds the scenario jitter with the run index (`engine.run(i)` in
`main.py`), so runs 0–9 are exactly the first ten of upstream's fifty, and a
later top-up extends rather than replaces them.

A run is **not** a repeat. Each index jitters the actor's start frame, position,
dimensions and identity, which is why deterministic models still show spread —
openpilot's sd is 0.27 on frontal-0103 and 1.00 on side-0103. Treating `--runs 1`
as sufficient for a deterministic model confuses reproducibility (same seed, same
score — true) with sufficiency (one draw characterises the model — false).

The standard error of a category mean over S scenes and R runs is
`sqrt(σ_between²/S + σ_within²/(S·R))`. Going from 1 scene to 5 attacks the
dominant term; going from 10 runs to 50 attacks a term already five times
smaller. At S=5 with openpilot's measured σ_within of 1.00 on side, the
within-scene contribution is 0.14 at R=10 and 0.06 at R=50 — both well under the
between-scene term. Five times the compute for a few percent of the error.

R=10 is also the point at which σ_between becomes measurable for the first time;
it has never been estimated here, because every recorded run is on one or two
scenes. If it turns out small, topping up runs is worth it. Alpamayo is the one
model where extra runs plausibly pay on their own: its σ_within is 1.82.

### Cost

Measured per-episode wall clock (2026-09-01 runs, `--runs 1`, including node
startup, which amortises across the runs in a sweep so these are upper bounds):

| model | s/episode | 20 pairs × 10 runs |
|---|---:|---:|
| visionpilot | 92 | ~5 h |
| openpilot | 134 | ~7.5 h |
| drivejepa | 191 | ~10.5 h |
| alpamayo15 | 375 | ~21 h |
| **total** | | **~44 h** |

The full 50-run protocol would be ~220 h with `gpu.serialize = true`.

### Per-run records reach `results/`

One subprocess drives N runs, but the store keeps them apart: the harness
overrides `ClosedLoopBenchmark.episode_samples` to emit one sample per run, so a
20-pair sweep at R=10 records `n = 200`, not `n = 20`, and the spread survives
into the only part of a run that is tracked. Runs are ordered by parsed index —
sorted as strings, `run_10` falls between `run_1` and `run_2`.

### Sweep shape

The renderer serves one scene at a time and is slow to start, so the loop is
scene-outer, model-inner — restart the renderer per scene, run all four models
against it, move on. That is 14 renderer starts rather than 56.

## Debugging video

The engine writes a frame per simulation step, per run, into the episode's
artifact directory under `roots.runs` — the raw camera as jpeg, plus two derived
streams: `FC_TRAJ` (front camera with the planned trajectory drawn on it — what
the model saw and what it decided) and `COMBINED_OUTPUTS` (the multi-panel
figure). Alongside them the engine writes `trajectories.json`, `ego_poses.json`,
`actors.json` and `reference_trajectory.json`.

Runs are the unit, because runs are what differ: `run_3/FC_TRAJ.mp4` is the clip
of the jitter draw that `run_3` scored, sitting next to the `metrics.json` that
scored it. With `--runs 1` the engine writes straight into the episode directory
and the video lands there instead.

After every episode the harness encodes each stream to H.264 at 10 fps — the
simulation steps at 0.1 s, so that is real time, which matters when the question
is whether a model reacted early enough. The last frame is held ~1.5 s so the
outcome is readable rather than a flicker.

### It has to be paid for

One run is about **83 MB** of frames, of which `FC_TRAJ` alone is 59 MB. Across
the full 20-pair set at 10 runs for four models that is **~66 GB**.

`video` in `[benchmarks.neuroncap]` decides what survives:

| mode | what is kept | full set, 4 models |
|---|---|---:|
| `encode` (default) | videos, raw `CAM_*` jpegs, all json | ~5 GB |
| `keep` | the above plus every source PNG | ~66 GB |
| `off` | no frames at all (`--engine.logger.enabled False`) | ~0 |

`encode` prunes only the derived PNG streams, and only ones that actually
encoded. Nothing is lost that cannot be rebuilt: the raw camera frames and the
trajectory json are what those streams were drawn from. Measured on real frames,
a two-run episode went 162 MB → 13 MB with both clips intact.

Encoding costs about 2.5 s per stream per run — roughly an hour spread across a
44 h sweep — and runs with `-threads 2`, because libx264 would otherwise take
every core it can see and this box fails at 90% CPU.

ffmpeg is found from `[benchmarks.neuroncap] ffmpeg`, then `$PADDOCK_FFMPEG`,
then `PATH`. **Video is never fatal**: no ffmpeg, or a failed encode, leaves the
frames in place and says so. An episode costs minutes of simulator time and a
debugging aid must not be able to lose one.

`video` is deliberately not part of `run_config()`. What is kept for debugging
does not change what was measured, and putting it there would give the same
sweep two different run ids depending on whether videos were wanted.

For all four models side by side on one clock, see
`tools/benchmarks/neuroncap/comparison_video.py` — it reads frames rather than
videos, so it needs `video = "keep"`.

## The score is not linear over 0–5

From `neuro_ncap/components/evaluator.py::_compute_ncap_score`:

```python
if not collisions:
    return 5.0                                        # a flat bonus
return 4.0 * (1 - impact_speed / reference_speed).clamp_min(0.0)
```

So a single run scores **either exactly 5.0, or at most 4.0**. There is a gap:
grazing an obstacle at walking pace scores just under 4, not just under 5. Any
figure between 4 and 5 is a mean over runs — Drive-JEPA's 4.04 is
`(5.00 + 2.13 + 5.00) / 3`, not a run that nearly avoided everything.

Two consequences worth keeping in mind when reading a table:

* **A mean hides which side of the gap its runs fell on.** 4.0 could be four
  clean avoidances and one bad collision, or five near-misses. The per-run rows
  in `results/` are what distinguish them, which is one more reason they are
  stored separately.
* **`impact_speed` and `reference_speed` are the score's own inputs.** Stored
  alongside it, any row can be checked: `4 * (1 - impact/reference)` reproduces
  the score exactly for a collision. That is how the formula above was confirmed
  against a real run rather than taken from a docstring — the README had said
  `5 · max(0, 1 − v_impact / v_reference)`, which overstates every collision
  score by 25%.


## Resuming a sweep

A full-set sweep is hours of simulator time, and until now a closed-loop run
that died took all of it with it -- only NAVSIM had `--resume`. It now works
here too:

```bash
paddock run --bench neuroncap --model openpilot --scene <the same list> --runs 10 \
  --resume 20260904T232901-abc123
```

The run id is the artifact directory's name under `roots.runs`. Episodes whose
runs are all on disk are reused; everything else is driven again. `run.json`
records `n_reused` so a resumed row says how much of it was re-driven.

**A part-finished episode is never reused.** The engine writes a run's
`metrics.json` only after scoring it, so counting the files is the check: fewer
than `--runs` asked for means the episode was interrupted, and it is driven
again from the start. A mean over however many runs happened to complete is a
wrong number rather than a slow one, which is the whole reason the check counts
rather than merely looking for the directory.

`--resume` compares the run's `run_config.json` against the current switches and
refuses if they differ, so a resume under a different scene list, run count or
camera set cannot merge two measurements under one id.

Benchmarks opt in: `ClosedLoopBenchmark.completed_metrics` returns None by
default, so a benchmark that cannot distinguish a finished episode from an
interrupted one re-runs everything rather than risking the wrong number.

**A `metrics.json` killed mid-write is skipped, not fatal.** That is the state a
SIGKILL, an OOM kill or a reboot leaves behind, and it is exactly the state
resume exists for -- so an unparseable file counts the run as missing and the
episode is driven again. It matters beyond resume: `_per_run` also backs
`_collect` and `episode_samples`, so one stale truncated file would otherwise
fail a *fresh* run into the same directory.

**A resume with nothing left to drive starts no node.** Re-issuing the command
to see where a sweep got to is a normal thing to do with a 12-hour sweep, and
loading alpamayo to reuse everything is a 21 GB checkpoint read and a VRAM wait
on a shared card. `run.json` carries `n_reused` alongside `n_expected` and
`n_failed`, and a resumed row says in `notes` that its `started_utc` and
`duration_s` cover that attempt only -- a remark, not a reason: resuming does
not make a run unrankable.


## A scene needs at least one lidar keyframe

The renderer never uses lidar points. The scene is in the trained hash grids and
the closed-loop server reads only poses, intrinsics, transforms and actor
trajectories. But NeuRAD's dataparser reads every lidar file eagerly, so
`renderer.py` substitutes an empty cloud for each missing one -- copying the
tensor width from a real cloud to do it.

That needs one real cloud. A scene with none fails at
[renderer.py:49](../../src/paddock/adapters/benchmarks/neuroncap/renderer.py#L49):

```
FileNotFoundError: no lidar files found at all -- check the nuScenes path, this is not the sweeps issue
```

Removing that guard does not help. With no lidar files the dataparser's lidar
set is empty, so `_get_world_to_mean_transform` builds the transform from the
cameras instead, and `ad_dataparser.py:320` then multiplies a Float transform
against Double lidar poses:

```
RuntimeError: expected scalar type Float but found Double
```

`preflight` now refuses such a scene before anything loads. The nuScenes
metadata is scanned once and cached, so all fourteen official scenes cost one
scan, not fourteen.

That scan reads `sample_data.json` -- 1.25 GB here -- a line at a time, keeping
only `sample_token -> filename` for LIDAR_TOP keyframes. Measured on this
machine: **peak RSS 64 MB**, against 4.6 GB for `json.loads` on the same file.
Wall clock is **6 s on an idle box and 20-30 s under load** (87% CPU, load
average 59), so it is bounded by whatever else the machine is doing. Reading the
file whole was faster when idle -- about 4 s -- and the memory is what the trade
buys: preflight is the cheap check that runs before everything else, often on a
box already busy, and it should not be the largest allocation in the run.

**This is not hypothetical -- it is why the check exists.** As extracted on
2026-09-04, the shakaal copy of nuScenes had 177 of 850 scenes with no lidar at
all, because `extract_nuscenes.sh` took `samples/CAM_*` and nothing else. Three
of those scenes -- 0796, 0921 and 0923 -- are in the official 20 pairs. The
full-set sweep spent roughly 13 GPU-hours on it: every attempt failed those
three, exited non-zero, and retried the other seventeen from the start,
producing byte-identical results each time.

Their lidar has since been fetched, so on that machine all fourteen official
scenes now pass and 174 scenes remain without lidar. The paragraph above is an
incident record, not a description of the current state -- check with
`paddock doctor` or a `--dry-run`, which is what the check is for.

Missing lidar for a scene is fixable without re-downloading the dataset. The
recipe is in [docs/data-setup.md](../data-setup.md) under "nuScenes with cameras
but no lidar for some scenes".

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Moved here from the README unchanged except for links; the symbols are
the README's former footnote marks, and each note below starts with the mark it
explains. Marks about a model rather than this benchmark (⊛ ⊕ ∗ ◊ ◇ ✦ ✧) are
explained in [What these models are](../models/comparison.md#notes-moved-from-the-readme-support-matrix).

### Notes on the matrix cells

⊙ Re-measured with the camera-frame fix (#62): 4.13, against 3.57 camera-blind.
See the neuroNCAP table.

§ Not run — zero by construction. Every neuroNCAP scenario is staged so that an
ego holding its current velocity collides at full speed. A run that avoids the
collision scores a flat **5**; a run that collides scores
`4 · max(0, 1 − v_impact / v_reference)`, which is 0 when the two speeds are
equal. The baseline therefore scores 0 in all three scenarios with a 100%
collision rate.

Note the discontinuity: **a single run scores either exactly 5, or at most 4** —
grazing an obstacle at walking pace scores 4, not 4.9. Anything between the two
is a mean over runs, which is what the model rows above are.

### neuroNCAP

The **full official set**: 20 scenario-scene pairs over 14 scenes, 10 runs each,
n = 200 per model. Mean ± std of the NCAP score; CR is the collision rate over
all 200 runs. Runs are seeded 0-9, so every model met identical scenario jitter.

| Model | stationary ↑ (0–5) | frontal ↑ (0–5) | side ↑ (0–5) | overall ↑ (0–5) | CR ↓ (of 200) |
|---|---:|---:|---:|---:|---:|
| Drive-JEPA ⊙ | 5.00 ±0.00 | 2.51 ±1.42 | 4.01 ±2.00 | **4.13** | 24.5% |
| Alpamayo 1.5 | 4.41 ±1.48 | 3.64 ±2.01 | 2.45 ±1.93 | **3.73** | 32.5% |
| OpenPilot | 4.52 ±1.34 | 2.30 ±1.24 | 3.08 ±2.25 | **3.60** | 38.0% |
| METEOR | 2.44 ±2.42 | 3.20 ±2.39 | 2.50 ±2.50 | **2.64** | 48.0% |
| Qwen-Drive 1.0 | 2.71 ±1.73 | 1.52 ±1.49 | 3.00 ±2.04 | **2.49** | 69.0% |
| VisionPilot | 2.53 ±1.60 | 1.17 ±1.38 | 0.08 ±0.18 | **1.58** | 85.5% |
| Baseline (Constant Velocity) | 0.00 | 0.00 | 0.00 | 0.00 § | 100.0% |

⊙ Measured after the camera-frame fix (#62), run `20261012T224707-f11ac0`. The
camera-blind run (`20260905T220225-f11ac0`) scored 4.23 / 2.42 / 3.39, overall 3.57,
with 45.5% collisions. Paired over the 20 scenario-scene pairs the fix is worth
+0.56 (SE 0.44, t 1.28), mostly stationary (+0.77, SE 0.30) and side.

**Qwen-Drive is the widest open-loop-to-closed-loop gap in the table.** It leads
KITScenes and ties Drive-JEPA on NAVSIM, then places fourth here, below three
models it beat on every open-loop column. The frontal column is where it goes:
1.52 with 86% of those runs ending in a collision, against 3.64 for Alpamayo.
**It brakes; it does not steer.** Over the 50 frontal episodes its planned 3 s
path reaches a mean peak lateral offset of 1.22 m, against 10.44 m for
Alpamayo — which decelerates no harder (−2.31 m/s against Qwen-Drive's −2.48)
and collides in 32% of episodes rather than 86%. Drive-JEPA brakes hard and still
collides in 78%, with or without the camera: seeing the car, it now plans a little
sideways movement (mean peak 1.80 m, against 0.11 m blind) and slows by 4.67 m/s
over the episode (5.76 blind, same measure), but not enough of either. In a staged head-on the closing
vehicle arrives whatever the ego's speed, so the escape is lateral, and Alpamayo
is the only model of the five that takes it. This is not an adapter limit: on
NAVSIM the same code path produces lateral offsets to 19.92 m. The full table,
and what is interpretation rather than measurement, are in
[docs/models/qwendrive.md](../models/qwendrive.md).

**METEOR is the mirror image: it steers, and it does not brake.** Its mean
places it fourth, level with Qwen-Drive; counted over scenario-scene pairs, it is
only marginally below Alpamayo, clearly below Drive-JEPA, and not clearly
separated from OpenPilot (paired below). Over the
50 frontal episodes its planned 3 s path reaches a mean peak lateral offset of
5.17 m, second only to Alpamayo's 10.44 m, and it collides in 36% of them, also
second best. But it speeds up through every scenario type rather than slowing:
the ego's last speed minus its first is +2.37 m/s over frontal episodes, +1.82
over side and +1.40 over stationary, against −2.81 for Qwen-Drive and −2.29 for
Alpamayo over the same frontal episodes, last speed minus first
(`docs/models/qwendrive.md` measures speed change differently, which is why its
figures for those episodes differ). So the escape that works head-on fails against an
obstacle standing in the lane, where stationary is its weakest column (2.44,
53% collisions). The same failure shows on NAVSIM, where its no-at-fault
collision rate sits at the constant-velocity floor. Detail in
[docs/models/meteor.md](../models/meteor.md).

**Scene 0103 was not representative, and the ordering it implied was wrong.**
Every model met identical scenario jitter (runs seeded 0–9), so these can be
compared paired sample for sample rather than as means:

| paired difference | mean | SE over 200 episodes | t | SE over 20 pairs | t |
|---|---:|---:|---:|---:|---:|
| Alpamayo − OpenPilot | +0.12 | 0.15 | 0.80 | 0.32 | 0.38 |
| Drive-JEPA − Alpamayo | +0.40 | 0.15 | 2.63 | 0.36 | 1.13 |
| Drive-JEPA − OpenPilot | +0.53 | 0.15 | 3.55 | 0.35 | 1.49 |
| Alpamayo − VisionPilot | **+2.15** | 0.17 | **12.89** | 0.38 | **5.73** |
| METEOR − Drive-JEPA | −1.49 | 0.21 | −7.16 | 0.48 | **−3.07** |
| METEOR − Qwen-Drive | +0.16 | 0.19 | 0.83 | 0.41 | 0.39 |

**Read the pair-level columns.** The 200 episodes are 10 seeded runs of each of
20 scenario-scene pairs, not 200 independent draws, and this benchmark's error
is dominated by the between-scene term (`docs/benchmarks/neuroncap.md`). The
pair-level standard error averages each pair's ten runs first. For the top three the two
levels disagree: over 200 episodes Drive-JEPA is ahead of Alpamayo (t 2.63) and
OpenPilot (t 3.55); over the 20 pairs it is not (t 1.13, 1.49). Over the 20 pairs
METEOR is clearly below Drive-JEPA (t −3.07), not separable from OpenPilot
(t −1.89), only marginally below Alpamayo (t −2.47) and above VisionPilot
(t +2.21), and level with Qwen-Drive either way.

The top three are **statistically indistinguishable** over the 20 pairs, though
Drive-JEPA now leads on the mean. Only VisionPilot is separated, and decisively. On scene 0103 alone the order was Drive-JEPA 4.04,
OpenPilot 3.80, Alpamayo 2.46 — which put Alpamayo last of the three and
Drive-JEPA clearly first. Neither holds up. Alpamayo moved most, from 2.46 to
3.73, because scene 0103's side scenario is one of its worst and n = 15 could
not show that.

VisionPilot's side score moved the same way in the other direction: 0.54 on
0103, 0.08 across the set.

**A single run scores either exactly 5.0 or at most 4.0** (see §), so a mean
between them is an average over runs and the standard deviation matters. The
per-run rows are in `results/`; VisionPilot's side result is 0.08 with a 100%
collision rate on all 50 runs — it hits the side obstacle at nearly full speed
every time, which the 0.54 from one scene did not show.

‖ Full official set, 20 pairs over 14 scenes, 10 runs each (n = 200), stored
`valid: true`. OpenPilot's earlier closed-loop figures were all produced before
`12f3250` fixed the `traffic_convention` one-hot — which had been telling
supercombo it was a right-hand-drive car in Boston — and are stored
`valid: false`; this is its first valid closed-loop score. Paired by seed on
frontal-0103 the fix was worth −0.16, and on side-0103 +0.36, so it moved
scenarios in both directions.

Three scenes (0796, 0921, 0923) initially failed for missing nuScenes lidar and
were completed after fetching it; both runs record `n_reused` for the episodes
carried over. See `docs/benchmarks/neuroncap.md`.
