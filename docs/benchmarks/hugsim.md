# HUGSIM

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | closed loop |
| Dataset | `hugsim` |
| Primary metric | `hd_score` |
| Adapter | [`adapters/benchmarks/hugsim/`](../../src/paddock/adapters/benchmarks/hugsim/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [alpamayo2edge](../models/alpamayo2edge.md), [autoe2e](../models/autoe2e.md), [constant_velocity](../models/constant_velocity.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [uniad](../models/uniad.md), [vad](../models/vad.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

HUGSIM ([hyzhou404/HUGSIM](https://github.com/hyzhou404/HUGSIM), data
[XDimLab/HUGSIM](https://huggingface.co/datasets/XDimLab/HUGSIM)) is a closed-loop
driving simulator built on 3D Gaussian splatting. Each scene is a reconstruction of
a real drive from KITTI-360, nuScenes, PandaSet or Waymo. The simulator renders six
cameras from it, inserts scripted vehicles, and moves the ego with its own iLQR
controller and a kinematic bicycle model, 0.25 s per step. The model plans; the
simulator decides what happens next and scores the episode, so errors compound, as
on Bench2Drive.

The release has 436 scenarios. Sixteen of them cannot run for any model (see
[The scenario set](#the-scenario-set)), so the benchmark here is **420
scenarios**. The headline is HUGSIM's own **HD-Score**.

## What you need

| | What | Where it goes |
|---|---|---|
| Simulator | the HUGSIM checkout at the sha `repos.yaml` pins, with `patches/HUGSIM/` applied | `[benchmarks.hugsim] repo` |
| Interpreter | Python with HUGSIM's stack (torch, gsplat, tiny-cuda-nn, pytorch3d, its own requirements) plus `requests` | `[benchmarks.hugsim] python` |
| Data | `scenarios/<dataset>/*.yaml`, `scenes/<dataset>/<scene>/`, `3DRealCar/<car_id>/` from XDimLab/HUGSIM, about 61 GB | `[datasets.hugsim] path` |
| HD map | the release's `nusc_map_cache`, linked as `~/.unified_data_cache` (HUGSIM hard-codes that path); only the 16 nuScenes scenarios with `load_HD_map: true` read it | your home directory |

```toml
[datasets.hugsim]
path = "/path/to/hugsim/data"

[benchmarks.hugsim]
python = "/path/to/hugsim/venv/bin/python"
repo   = "/path/to/HUGSIM"
# per_episode_timeout_s = 3600   # default
# min_free_gb = 12               # default: free VRAM to wait for before each HUGSIM process
# keep_ply = "false"             # default: the per-run scene/ground point clouds are deleted after scoring
```

HUGSIM's upstream install pins torch 2.4/cu118, which has no kernels for newer GPUs;
the site that produced the results below built torch 2.7.1/cu128 and the CUDA
extensions for sm_120, and kept every other package at the versions upstream's
`pixi.lock` pins. `paddock doctor` and every run start with `preflight`, which checks
the checkout and its three patches, the data directories, the HD-map link, and that
the interpreter can import the simulator and the bridge.

## Running it

```bash
paddock run --model drivejepa --bench hugsim                 # all 420 scenarios
paddock run --model drivejepa --bench hugsim --scene waymo   # one source dataset (a subset: valid: false)
paddock run --model drivejepa --bench hugsim --scene scene-0013-medium-00
```

`--scene` takes a scenario file name, a dataset (`kitti360`, `nuscenes`, `pandaset`,
`waymo`) or a scene name. One HUGSIM process runs per scenario and needs a few GB of
VRAM (4.4 GB peak on PandaSet). A 400-step episode of a 2 Hz model took up to about
four minutes on the site measured.

## How paddock drives it

`harness.py` starts `launcher.py` in HUGSIM's interpreter once per scenario.
`launcher.py` is upstream `closed_loop.py`'s entry point with the simulation loop
unchanged; its AD client is paddock's `bridge.py`, which turns each simulator step
into an `/infer` on the model node and the model's plan back into HUGSIM's
waypoints. The choices that fix what is measured:

- **Command only.** The bridge sends HUGSIM's route command, `0` right, `1` left,
  `2` straight (paddock's `Command` values; checked against the route geometry), and
  no route geometry. The recorded route is the human driver's exact line, so sending
  it would hand the model the answer.
- **The model's own rate.** HUGSIM steps at 4 Hz. A model that declares
  `planning_hz` is called at that rate and its last plan is held in between
  (Drive-JEPA, 2 Hz: every second step); one that declares none is called every
  step. Between calls the held plan is fixed in the world and re-sampled for where
  the car actually is, as the Bench2Drive client holds a plan.
  A model asked slower than the rate its history is built on (`native_input_hz`)
  is handed every frame between its plans (`POST /observe`, no plan asked;
  `config.observe`): Qwen-Drive plans every second step and reads poses on a 0.1 s
  grid, so it sees every 0.25 s step, the densest HUGSIM renders. A model that
  plans faster than 4 Hz (AutoE2E, Alpamayo 1.5 and METEOR declare 10, 10 and
  5 Hz) is asked every step, the most HUGSIM allows.
- **What the model is told.** HUGSIM's ego point is the front camera's calibrated
  pose, and it moves that point along the camera's heading, which on nuScenes is
  0.58° left of the calibration's vehicle axis. The vehicle frame sent to models is
  therefore the calibration's, yawed so x is HUGSIM's direction of travel; without
  that, a plan straight ahead drifted right on every replan. Cameras are sent where
  HUGSIM renders them (0.3 m below the calibrated pose on nuScenes), `lidar2ego` is
  the identity as on Bench2Drive, and speed, acceleration and yaw rate are measured
  from successive poses.
- **The controller fix.** Upstream's `traj2control` reads each waypoint the wrong
  way round, so a straight plan gets a reference heading of about +90° and the iLQR
  steers it right: 4.3 m in 14 m with a straight constant-speed client on nuScenes
  scene-0013. `patches/HUGSIM/0001` is upstream
  [PR #57](https://github.com/hyzhou404/HUGSIM/pull/57) verbatim (open, unmerged),
  and every run here uses it. With it the same client drifts 0.00 m. Quote HUGSIM
  numbers with the controller they were made with.
- **Only the model's cameras are rendered** (`patches/HUGSIM/0002`); the others are
  sent black, as HUGSIM itself sends the rear cameras on Waymo and KITTI-360, which
  have no rear data. Scoring never reads images.
- **The scripted cars' planner fix** (`patches/HUGSIM/0003`). Upstream's `IDM.update`
  indexes an empty tensor when an agent one point from its path's end has a car on that
  point (`sim/utils/agent_controller.py:170`), and the simulator dies with no score.
  That one case now takes the agent's distance to the point; an episode where it never
  happens is unchanged. It is what ended Drive-JEPA's `scene-0166-medium-01` (below).
- **The whole plan is sent**, up to 4 s (8 waypoints 0.5 s apart; NAVSIM's horizon,
  UniAD and VAD plan 3 s). HUGSIM scores every step's sent plan -- collision, drivable
  area, TTC and comfort over its length -- so a plan cut short is judged over less time
  than the model planned, and a model that plans 3 s (constant velocity, METEOR: 6
  waypoints) is judged over a shorter, easier window than one that plans 4 s: no
  collision is 0 if any sent waypoint collides. Its iLQR also tracks the whole sent
  plan, so the horizon changes how the car is driven, not only how it is scored. The bridge reads the reply's `plan`, not its `trajectory`,
  which the wire cuts at six waypoints for neuroNCAP. **The 22 Oct campaigns, now
  superseded, were sent six (3 s)** for every model, Drive-JEPA's 4 s plans included.
- **The step cap.** Upstream ends an episode after 400 steps (100 s).

### How an episode ends

`HUGSimEnv.step` prints one line when it ends an episode. The harness reads it from
the simulator's log and keeps it with each sample (`extra.ending`):

| Ending | Simulator's line |
|---|---|
| completed | `Complete` (route completion reached 1) |
| hit a vehicle | `Collision with foreground` |
| hit scenery | `Collision with background` |
| off route | `Far from preset trajectory` (more than 10 m from the recorded path) |
| step cap | none of these, after 400 steps |

The scorer also prints lower-case "collision with ..." for any *planned* trajectory
that would collide; those lines say nothing about how the episode ended and are not
read.

## Metrics

HUGSIM scores every plan the model sent (`sim/utils/score_calculator.py`). Per plan,
NAVSIM's terms without progress; per episode, the mean over its plans; and the route
completion the car reached.

| name | what it measures |
|---|---|
| `hd_score` | HUGSIM's HD-Score, 0–1, **higher is better**: `hugsim_pdms` × route completion (as a fraction). The headline |
| `hugsim_pdms` | the mean over the episode's plans of NC × DAC × (5·TTC + 2·C) / 7, 0–1, **higher is better**. NAVSIM's composite without its progress term, so not the same number as `pdms` |
| `hugsim_nc` | mean over plans of HUGSIM's no-collision term, 0–1, **higher is better** |
| `hugsim_dac` | mean over plans of HUGSIM's drivable-area term, 0–1, **higher is better** |
| `hugsim_ttc` | mean over plans of HUGSIM's time-to-collision term, 0–1, **higher is better** |
| `hugsim_comfort` | mean over plans of HUGSIM's comfort term, 0–1, **higher is better** |

`route_completion` is the canonical name: the furthest point along the route the car
reached, in percent, capped at 100. `collision_rate` is 1 for an episode that ended
in a collision with a vehicle or scenery, so a run's mean is the fraction of episodes
that did. `metrics.json` carries the per-dataset breakdown in `by_split` and the
per-difficulty one in `by_group`.

## The scenario set

| | Scenarios | Why |
|---|---:|---|
| Released | 436 | kitti360 113, nuscenes 88, pandaset 127, waymo 108; easy 80, medium 157, hard 96, extreme 103 |
| Left out for every model | 16 | fifteen PandaSet `*-medium-02` scenarios give a scripted car `ConstantPlanner(max_t=25)`, an argument the released `ConstantPlanner` does not take (`TypeError` while the environment is built); `waymo/scene-140045460035-hard-00` has `start_velo: 1R` where its siblings have `1` (`TypeError: new(): invalid data type 'str'`). Both fail before the first step, whatever drives. Listed in `harness.EXCLUDED` and in every `run.json` |
| **The benchmark** | **420** | kitti360 113, nuscenes 88, pandaset 112, waymo 107 |

Two release quirks the harness absorbs without editing the data: the scenarios name
each vehicle `<car_id>/postprocess/shadow.pth` while the released assets are
`<car_id>/{gs.pth,wlh.json}`, so the launcher writes a repaired copy of each scenario
into the episode's directory; and `kitti360/scene-1126_1326-medium-00.yaml` holds an
easy scenario (`mode: easy_00`), so difficulty is read from `mode`, not the file
name.

## Results

One campaign of ten paddock models, driven 8-10 Nov 2026 (UniAD and VAD 9-10 Nov) through paddock's own bridge and
ingested from its recordings (see [Recorded campaigns](#recorded-campaigns)). WA-JEPA was not
driven. Every run covers all 420 scenarios, is `complete` and `valid: true`. Command only,
PR #57 controller, the scripted cars' planner fix (patch 0003), the whole plan sent (up to
4 s), each model asked at the rate its `model.yaml` declares, 400-step cap. Scores ×100,
sorted by HD-Score.

| Model | Asked at | Plan sent | Run | HD-Score | PDMS | Route completion | NC | DAC | TTC | Comfort |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| [Drive-JEPA](../models/drivejepa.md) (released NAVSIM v2 checkpoint, momentum on) | 2 Hz | 4 s | [`fullplan-f7d8f9d4b9736ddd-b89341`](../../results/hugsim/drivejepa/fullplan-f7d8f9d4b9736ddd-b89341/run.json) | 37.1 | 62.1 | 47.2 | 70.3 | 96.0 | 65.0 | 87.3 |
| [Qwen-Drive](../models/qwendrive.md) | 2 Hz, every step observed | 4 s | [`fullplan-f7d8f9d4b9736ddd-1ddf85`](../../results/hugsim/qwendrive/fullplan-f7d8f9d4b9736ddd-1ddf85/run.json) | 34.8 | 52.3 | 46.7 | 61.7 | 92.7 | 55.2 | 92.8 |
| [Alpamayo 2 Edge](../models/alpamayo2edge.md) † | 4 Hz | 4 s | [`20261114T172855-b502b3`](../../results/hugsim/alpamayo2edge/20261114T172855-b502b3/run.json) | 31.2 | 53.1 | 45.3 | 63.5 | 88.7 | 57.4 | 92.1 |
| [UniAD](../models/uniad.md) (1.0 base, nuScenes checkpoint, zero-shot) | 2 Hz | 3 s | [`fullplan-f7d8f9d4b9736ddd-9d698c`](../../results/hugsim/uniad/fullplan-f7d8f9d4b9736ddd-9d698c/run.json) | 26.1 | 35.0 | 45.7 | 54.3 | 91.0 | 37.9 | 47.5 |
| [Constant velocity](../models/constant_velocity.md) (floor) | 4 Hz | 3 s | [`fullplan-f7d8f9d4b9736ddd-a8f5b7`](../../results/hugsim/constant_velocity/fullplan-f7d8f9d4b9736ddd-a8f5b7/run.json) | 24.2 | 65.6 | 28.6 | 72.9 | 93.0 | 68.6 | 100.0 |
| [Alpamayo 1.5](../models/alpamayo15.md) | 4 Hz | 4 s | [`fullplan-f7d8f9d4b9736ddd-7ea973`](../../results/hugsim/alpamayo15/fullplan-f7d8f9d4b9736ddd-7ea973/run.json) | 23.8 | 54.6 | 35.7 | 70.8 | 89.5 | 65.2 | 63.5 |
| [VisionPilot](../models/visionpilot.md) | 4 Hz | 4 s | [`fullplan-f7d8f9d4b9736ddd-92c0b7`](../../results/hugsim/visionpilot/fullplan-f7d8f9d4b9736ddd-92c0b7/run.json) | 16.3 | 35.2 | 34.5 | 67.6 | 94.3 | 48.9 | 7.8 |
| [METEOR](../models/meteor.md) | 4 Hz | 3 s | [`fullplan-f7d8f9d4b9736ddd-241d7f`](../../results/hugsim/meteor/fullplan-f7d8f9d4b9736ddd-241d7f/run.json) | 16.2 | 30.7 | 32.5 | 48.3 | 81.4 | 35.7 | 53.2 |
| [VAD](../models/vad.md) (base, nuScenes checkpoint, zero-shot) | 2 Hz | 3 s | [`fullplan-f7d8f9d4b9736ddd-9d698c`](../../results/hugsim/vad/fullplan-f7d8f9d4b9736ddd-9d698c/run.json) | 9.6 | 54.4 | 15.9 | 62.6 | 89.4 | 57.3 | 95.6 |
| [openpilot](../models/openpilot.md) | 4 Hz | 4 s | [`fullplan-f7d8f9d4b9736ddd-cac3b5`](../../results/hugsim/openpilot/fullplan-f7d8f9d4b9736ddd-cac3b5/run.json) | 6.1 | 73.5 | 7.0 | 84.1 | 88.3 | 82.6 | 100.0 |
| [AutoE2E](../models/autoe2e.md) v1.0 | 4 Hz | 4 s | [`fullplan-f7d8f9d4b9736ddd-4d9640`](../../results/hugsim/autoe2e/fullplan-f7d8f9d4b9736ddd-4d9640/run.json) | 4.3 | 27.5 | 13.7 | 44.8 | 48.9 | 37.0 | 64.9 |

HD-Score by source dataset and by difficulty:

| | kitti360 | nuscenes | pandaset | waymo | easy | medium | hard | extreme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive-JEPA | 23.8 | 41.6 | 38.3 | 46.2 | 66.4 | 50.2 | 19.8 | 12.3 |
| Qwen-Drive | 19.3 | 48.8 | 35.9 | 38.6 | 70.6 | 40.3 | 21.6 | 11.7 |
| Alpamayo 2 Edge | 10.2 | 36.7 | 47.0 | 32.3 | 47.1 | 37.4 | 24.9 | 16.1 |
| UniAD | 2.9 | 43.2 | 31.0 | 31.3 | 53.2 | 25.2 | 20.1 | 11.9 |
| Constant velocity | 7.8 | 34.6 | 30.5 | 26.2 | 49.2 | 30.4 | 14.1 | 5.5 |
| Alpamayo 1.5 | 22.4 | 19.3 | 26.3 | 26.5 | 41.8 | 27.6 | 14.8 | 12.9 |
| VisionPilot | 7.7 | 16.1 | 21.3 | 20.5 | 35.2 | 22.0 | 7.4 | 2.2 |
| METEOR | 2.1 | 31.8 | 20.8 | 13.7 | 34.9 | 15.0 | 13.8 | 5.8 |
| VAD | 3.6 | 12.3 | 11.3 | 12.1 | 12.6 | 11.7 | 7.7 | 6.4 |
| openpilot | 2.5 | 5.5 | 4.6 | 12.0 | 8.9 | 8.9 | 4.2 | 2.0 |
| AutoE2E | 0.5 | 5.0 | 6.2 | 5.6 | 4.9 | 4.5 | 4.5 | 3.3 |

How episodes ended, % of 420:

| | completed | hit a vehicle | hit scenery | off route | step cap |
|---|---:|---:|---:|---:|---:|
| Drive-JEPA | 30.7 | 38.6 | 7.1 | 11.4 | 12.1 |
| Qwen-Drive | 27.1 | 51.2 | 19.3 | 2.4 | 0.0 |
| Alpamayo 2 Edge | 25.2 | 42.1 | 24.8 | 7.9 | 0.0 |
| UniAD | 25.5 | 46.2 | 23.6 | 1.4 | 3.3 |
| Constant velocity | 6.2 | 54.8 | 25.2 | 7.6 | 6.2 |
| Alpamayo 1.5 | 16.7 | 44.0 | 18.6 | 20.7 | 0.0 |
| VisionPilot | 11.7 | 56.2 | 12.4 | 7.9 | 11.9 |
| METEOR | 11.7 | 39.0 | 42.6 | 6.7 | 0.0 |
| VAD | 0.0 | 43.3 | 35.2 | 0.5 | 21.0 |
| openpilot | 0.0 | 41.4 | 4.0 | 0.0 | 54.5 |
| AutoE2E | 0.0 | 14.8 | 67.1 | 18.1 | 0.0 |

What to read with them:

- **† Alpamayo 2 Edge was driven later, by `paddock run`, not ingested.** 14-15 Nov 2026,
  one node on a shared RTX PRO 6000, through the harness and `launcher.py` on this branch
  rather than the campaign runner, so its record's run id is a plain `paddock run` id. Same
  420 scenarios (`sequences_sha256: f7d8f9d4b9736ddd`), the same 16 left out, command only,
  PR #57 controller, patch 0003, 400-step cap. Edge declares 10 Hz, so it was asked every
  step (15,643 plans, no bridge error); its 0.1 s plan was sent as 8 waypoints 0.5 s apart.
  The run was cut off after 284 scenarios and finished with `--resume`, which reused them
  (`dataset.n_reused`); 7.0 h of episodes in all.
- **Edge's run rendered every camera, though its record lists five.** The site's HUGSIM
  interpreter has `hugsim_env` installed editable from an unpatched checkout, and the
  harness puts the checkout's root, not its `sim/`, on the path, so `hugsim_env` (and the
  `HUGSIM_RENDER_CAMS` switch patch 0002 adds to it) loaded from there; `sim.utils`, which
  patches 0001 and 0003 change, loaded from the patched checkout. `config.render_cams` is
  what the harness asked for. Scoring never reads images, and Edge's node keeps only the
  five cameras it declares, so the scores are unaffected; the run was only slower.
- **Each model at its own rate.** HUGSIM steps at 4 Hz, so a model that declares 4 Hz or more
  (Alpamayo 1.5 and AutoE2E 10 Hz, METEOR 5 Hz), or no rate (VisionPilot, openpilot, the floor),
  is asked every step. Drive-JEPA, Qwen-Drive, UniAD and VAD plan at 2 Hz, every second step; Qwen-Drive's
  2 Hz is the rate PR #167 declares in its
  `model.yaml`, with which the nodes were served, and it was handed the frame of every step between
  its plans (`config.observe`). The rate in each `run.json` is read from the episodes' calls, not
  typed.
- **Plan sent** is how many 0.5 s waypoints HUGSIM was sent per plan (`config.waypoints_sent`):
  8 for the models that plan 4 s, 6 for the four that plan 3 s (constant velocity, METEOR, UniAD, VAD). A plan is never padded.
- **Drive-JEPA ran with momentum-aware selection and the OpenScene crop**, as its `model.yaml`
  declares; momentum was fed on every call but each episode's first (23,797 of 24,217 calls).
- **VisionPilot has no trajectory generator.** The adapter synthesizes its plan (AutoSteer's lateral
  offsets fitted to a ground path, the longitudinal schedule from a ported IDM) and stretches it from
  the stack's native 1.0 s to 4.0 s, so the tail of every plan HUGSIM scores is extrapolation. Its
  comfort of 7.8 is measured on that synthesized plan.
- **openpilot ran at 4 Hz, below the 10 Hz its `model.yaml` declares as its minimum** (`min_input_hz`).
  HUGSIM renders one frame per 0.25 s step, and supercombo infers its own motion from the change between
  frames: fed frames that far apart it reads the scene as nearly still, the parked-car plan its model.yaml
  describes. That matches what it did: its plans reach a median 4.4 m in 4 s, the car's median speed over
  all steps is 0.07 m/s, 54.5% of its episodes run to the step cap and none completes, and a plan that
  barely moves collides with little, which is why its PDMS (73.5) is high. Its row is ranked, with this
  caveat: it measures openpilot at a rate it was not built for, not openpilot as it drives at 20 Hz.
  Nothing in the HUGSIM harness refuses a model below its `min_input_hz` (the open-loop harnesses do).
- **AutoE2E takes no command**, so it gets upstream's no-route input. Two thirds of its episodes end
  against scenery: it swerves out of its lane early, as it does on Bench2Drive.
- **Rear cameras on KITTI-360 and Waymo are black for every model.** HUGSIM has no rear data there.
- **One run per scenario, on a shared GPU.** No repeat was made, so no spread is known. Five attempts
  across the eight models of 8-9 Nov failed loading their scene (CUDA out of memory, other users' jobs on the
  card) before the first step and were driven again from scratch; only scored attempts count.
- **The 22 Oct campaigns are superseded.** Drive-JEPA's (momentum off, uncropped, 419 of 420 after
  the IDM crash on `nuscenes/scene-0166-medium-01`; HD-Score 33.8) and AutoE2E's (5.7) were sent six
  waypoints (3 s) of every plan. Both records are kept as they ran, `valid: false`, each pointing at
  its replacement. That campaign also drove the Final and Run 3A; neither is a paddock adapter on
  main, so neither is stored here.

## Recorded campaigns

Every stored run was driven outside the harness, by a runner that shared one scenario
queue between several lanes, each lane one model node, with the launcher logic applied at
run time (`hugsim_run.py --fix_heading --render_cams --fix_idm_lead`, the changes patches
0001-0003 make in the checkout). The 8-9 Nov campaign's AD client was paddock's own
`bridge.py`; the 22 Oct one's was the bridge `bridge.py` was made from, which sent six
waypoints. `paddock ingest --all` reads such a campaign
(`src/paddock/results/ingest/hugsim.py`):

```toml
[[ingest.hugsim]]
model = "drivejepa"
runs  = "/path/to/hugsim-eval/dj_released"   # results.jsonl, <dataset>_bridge/<scene>_<mode>/, logs/
label = "campaign"
notes = "how the node was served"
[ingest.hugsim.model_env]                     # optional: the operator's statement of the node's settings
DRIVEJEPA_MOMENTUM = "off"
```

Scores are read from each scenario's `eval.json` through the harness's own
`episode_record`, and each must agree with the campaign's `results.jsonl` line. A
scenario attempted and never scored stays expected and makes the run partial. The
controller fix and the rendered cameras are read from the runner's notes on every
scenario's log, and the call rate from each scenario's steps and model calls; a
campaign that disagrees with itself is stored and does not rank. A campaign driven through
paddock's own bridge also records the IDM guard (`idm_lead`, from the runner's note), the
waypoints sent per plan (`waypoints_sent`) and whether the frames between plans were handed
to the model (`observe`), all read from its logs; each key is absent for a campaign that
recorded none, so an older run keeps its id.

## Verified, and argued

| Claim | How | Strength |
|---|---|---|
| The stored means are the campaigns' | the ingest reproduces, from the `eval.json` files, the Drive-JEPA figures first reported from the 22 Oct campaign's `results.jsonl` (HD-Score 33.8, PDMS 71.7, RC 39.8, NC 80.8, DAC 95.2, TTC 75.5, comfort 93.6; endings 22.7 / 38.9 / 6.9 / 11.5 / 20.0), and all eight 8-9 Nov means first computed from that campaign's `results.jsonl` (every scored line must agree with its `eval.json`) | **verified** |
| Each 8-9 Nov model ran at its declared rate | `config.infer_every` read from every episode's steps and calls: 2 for Drive-JEPA and Qwen-Drive, 1 for the rest; Qwen-Drive observed on every step between plans in all 420 episodes, no other model on any | **verified** |
| `bridge.py` is the bridge that drove the campaign | camera matrices, vehicle pose and the plans sent to HUGSIM match that bridge's own output to 1e-12 on a test rig (`tests/test_bridge.py`); its scipy rotations are replaced by numpy, equal to 7e-16 on 10,000 random rotations | **verified** |
| Patch 0001 is the controller the campaign ran | the campaign applied PR #57 at run time; the patched `traj2control` gives bit-identical acceleration and steering rate on 200 random plans | **verified** |
| Patch 0002 renders what the campaign rendered | the same code as the campaign's run-time replacement, moved into `_get_obs`; scores never read images | argued (read, not re-run) |
| The three patches apply at the pinned sha | `git apply` on a clean checkout of `62c690d`, in order; the patched files equal the intended trees | **verified** |
| Patch 0003 lets scene-0166-medium-01 finish | the same guard, applied at run time, re-drove released Drive-JEPA on it: steps 0-59 identical to the crashed attempt (ego path, all 30 model calls), then on to the 400-step cap; 3,860 random IDM cases identical to upstream, finite on the 140 upstream raises on | **verified** |
| The bridge sends the model's whole plan | `plan` read before `trajectory`, the horizon counted in time, cut at 8 (`tests/test_bridge.py`); end to end, every call of the 8-9 Nov campaign logged 8 waypoints sent for the six models that plan 4 s or more (AutoE2E plans 12, openpilot 20) and 6 for the two that plan 3 s | **verified** |
| The Drive-JEPA node drove as main's adapter did on 22 Oct | the served adapter differs from `ac7788f` only by recording and a camera placement that is identical given the identity `lidar2ego`. Against a run of `ac7788f` itself on 7 scenarios: 5 identical to six decimals; the two longest (331 and 400 steps) diverged | argued, partly checked |
| `launcher.py` runs a scenario end to end, including the `--observe` launch path | not run: no HUGSIM episode was driven through the harness for this change; the records here came from the campaign runner, and `--observe` (Qwen-Drive) ran only there | **unverified** |
| The 16 left-out scenarios fail for every model | each failed the same way for three models in the campaign | **verified** |

## Support-matrix results

The README's support matrix has a HUGSIM column: each model's HD-Score (0–1) from its run in
[Results](#results), all eleven of the matrix's models.
The column's notes give the rate where a model is asked slower than every step (2 Hz) or below its
own minimum (openpilot, 4 Hz; see the caveat above).
