# HUGSIM

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
the checkout and its two patches, the data directories, the HD-map link, and that
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

Two campaigns, both driven on 22 Oct 2026, are stored, ingested from their
recordings (see [Recorded campaigns](#recorded-campaigns)). Command only, PR #57
controller, 400-step cap. Scores ×100.

| Model | Run | Scenarios | HD-Score | PDMS | Route completion | NC | DAC | TTC | Comfort |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive-JEPA (released NAVSIM v2 checkpoint), 2 Hz | [`campaign-87d4f5097a85db3c-ce3603`](../../results/hugsim/drivejepa/campaign-87d4f5097a85db3c-ce3603/run.json) | 419 of 420 | 33.8 | 71.7 | 39.8 | 80.8 | 95.2 | 75.5 | 93.6 |
| AutoE2E v1.0, 4 Hz | [`campaign-f7d8f9d4b9736ddd-d03155`](../../results/hugsim/autoe2e/campaign-f7d8f9d4b9736ddd-d03155/run.json) | 420 of 420 | 5.7 | 35.7 | 14.1 | 53.9 | 59.8 | 45.0 | 68.1 |

HD-Score by source dataset and by difficulty:

| | kitti360 | nuscenes | pandaset | waymo | easy | medium | hard | extreme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drive-JEPA | 22.8 | 45.5 | 32.7 | 37.0 | 62.2 | 41.0 | 19.3 | 15.4 |
| AutoE2E | 1.1 | 6.4 | 8.1 | 7.4 | 6.1 | 5.9 | 6.0 | 4.7 |

How episodes ended, % of each model's scored scenarios:

| | completed | hit a vehicle | hit scenery | off route | step cap |
|---|---:|---:|---:|---:|---:|
| Drive-JEPA (419) | 22.7 | 38.9 | 6.9 | 11.5 | 20.0 |
| AutoE2E (420) | 0.0 | 16.7 | 66.4 | 16.9 | 0.0 |

What to read with them:

- **Drive-JEPA's run is stored `partial`, `valid: false`.** On
  `nuscenes/scene-0166-medium-01` HUGSIM itself crashed at step 60, on both
  attempts: its scripted-vehicle controller (`sim/utils/agent_controller.py:170`)
  indexed an empty lookahead path of another car. AutoE2E finished that scenario, so
  it is a lost scenario for Drive-JEPA, not a release bug, and paddock does not rank
  a run with one. Its means are over the other 419.
- **Neither run is today's adapter.** Drive-JEPA was served by the adapter as of
  `ac7788f` (22 Oct) plus a per-call record of its 32 candidates; at that commit it
  fed cameras uncropped and picked on the plain score. Main now crops to the
  OpenScene camera (#106) and turns momentum selection on (#95). AutoE2E was served
  by `b507811` (20 Oct), unmodified; its adapter has changed since in #92, #97, #98,
  #107 and #108. A re-run on main may differ for both; each `run.json` says so.
- **Drive-JEPA was rendered the front camera only.** The released model's forward
  reads only the front camera (it encodes the two front frames and slices
  `lidar2img` to that camera), so the black side and rear frames are never read.
  Through the harness every declared camera is rendered.
- **AutoE2E takes no command**, so it gets upstream's no-route input, and it is
  called every step because it declares no planning rate. Half of its episodes end
  within 18 steps (4.5 s), mostly against scenery: it swerves out of its lane early,
  as it does on Bench2Drive.
- **Rear cameras on KITTI-360 and Waymo are black for every model.** HUGSIM has no
  rear data there.
- **One run per scenario.** No repeat was made, so no spread is known.

Two further models were driven in the same campaign, the Final (Bench2Drive-trained
Drive-JEPA) and Run 3A (WA-JEPA tokens with Drive-JEPA's planner). Neither is a
paddock adapter on main, so neither is stored here.

## Recorded campaigns

The two runs above were driven before this harness existed, by a runner that
shared one scenario queue between several lanes, each lane one model node, with the
same bridge and launcher logic as here (`bridge.py` is that bridge; `launcher.py`
that launcher, with the controller fix and the camera selection moved into the two
patches). `paddock ingest --all` reads such a campaign
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
campaign that disagrees with itself is stored and does not rank.

## Verified, and argued

| Claim | How | Strength |
|---|---|---|
| The stored means are the campaigns' | the ingest reproduces, from the `eval.json` files, the Drive-JEPA figures first reported from the campaign's `results.jsonl` (HD-Score 33.8, PDMS 71.7, RC 39.8, NC 80.8, DAC 95.2, TTC 75.5, comfort 93.6; endings 22.7 / 38.9 / 6.9 / 11.5 / 20.0) | **verified** |
| `bridge.py` is the bridge that drove the campaign | camera matrices, vehicle pose and the plans sent to HUGSIM match that bridge's own output to 1e-12 on a test rig (`tests/test_bridge.py`); its scipy rotations are replaced by numpy, equal to 7e-16 on 10,000 random rotations | **verified** |
| Patch 0001 is the controller the campaign ran | the campaign applied PR #57 at run time; the patched `traj2control` gives bit-identical acceleration and steering rate on 200 random plans | **verified** |
| Patch 0002 renders what the campaign rendered | the same code as the campaign's run-time replacement, moved into `_get_obs`; scores never read images | argued (read, not re-run) |
| Both patches apply at the pinned sha | `git apply` on a clean checkout of `62c690d`; the patched files equal the intended trees | **verified** |
| The Drive-JEPA node drove as main's adapter did on 22 Oct | the served adapter differs from `ac7788f` only by recording and a camera placement that is identical given the identity `lidar2ego`. Against a run of `ac7788f` itself on 7 scenarios: 5 identical to six decimals; the two longest (331 and 400 steps) diverged | argued, partly checked |
| `launcher.py` runs a scenario end to end | not run: no HUGSIM episode was driven through the harness for this change | **unverified** |
| The 16 left-out scenarios fail for every model | each failed the same way for three models in the campaign | **verified** |

## Support-matrix results

HUGSIM has no column in the README's support matrix yet: two of the matrix's models
have been driven here.
