# SimLingo

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [RenzKa/simlingo](https://github.com/RenzKa/simlingo) at `743b243` |
| Adapter | [`adapters/models/simlingo/`](../../src/paddock/adapters/models/simlingo/) |
| Environment | uv, Python 3.10 |
| Cameras | `CAM_SIMLINGO`, `CAM_FRONT` |
| Plan | waypoints, 0.25 s apart, 2.5 s ahead |
| Paddock can run it on | [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [navhard](../benchmarks/navhard.md) |

<!-- /summary -->

## What it is

SimLingo (Renz et al., CVPR 2025, University of Tuebingen with Wayve;
[paper](https://arxiv.org/abs/2503.09594)) is a vision-language-action model for driving:
InternVL2-1B (InternViT-300M and Qwen2-0.5B, fine-tuned with LoRA) reads one wide front camera,
the car's speed and the route, writes a short commentary on what to do, and then decodes ten
positions 0.25 s apart (2.5 s) and a twenty-point path. It is trained only in CARLA, on its own
dataset driven by the PDM-Lite expert in every town, Bench2Drive's included. On the Bench2Drive
leaderboard its authors' repository keeps (version 0.0.4) it scores 86.55, the highest
camera-only result there.

Code: `simlingo` in `repos.yaml` (RenzKa/simlingo at `743b243`). Weights:
`RenzKa/simlingo` at `26c7c89` (the epoch-13 checkpoint and its training config) and the base
model `OpenGVLab/InternVL2-1B` at `0d75ccd`, linked as `{weights}/SimLingo` and
`{weights}/InternVL2-1B`. Adapter: `src/paddock/adapters/models/simlingo/`.

```bash
paddock setup --model simlingo
paddock verify --model simlingo
paddock run --model simlingo --bench kitscenes --limit 5
```

## How paddock drives it

The adapter follows SimLingo's own CARLA agent (`team_code/agent_simlingo.py`) step for step and
imports its code rather than copying it: the model is built from the checkpoint's training config
in bfloat16; every frame goes through the agent's JPEG round trip, its bonnet cut (the bottom
4.8/16 of the image) and InternVL2's two 448-pixel tiles; the prompt is the agent's
chain-of-thought one, ending "What should the ego do next?". What paddock does differently, and why:

- **Camera.** SimLingo's one camera is 1024x512 at 110 deg, 1.5 m behind the vehicle origin and
  2.0 m up. Paddock's Bench2Drive rig renders it as `CAM_SIMLINGO`, as it renders openpilot's
  two cameras, because paddock's 70 deg `CAM_FRONT` would show it a narrower, magnified view.
  A recorded dataset has no such camera: its `CAM_FRONT` is placed at its true angular scale in
  the middle of a black 1024x512, 110 deg frame (`inputs.to_canvas`), so objects keep the size
  and bearing SimLingo learned and what the camera did not see stays black. The mount is not
  changed.
- **Route.** Where the source sends a route (`route_xy`, closed loop), it is given as the agent's
  two target points, tracked as the agent's RoutePlanner tracks them (`inputs.RouteTracker`).
  Where it does not (recorded datasets), the command is given as the agent's sentence,
  "Command: go left at the next intersection." or "Command: follow the road.", the other route
  form SimLingo was trained on.
- **Plan length.** SimLingo plans 2.5 s. Closed-loop clients need 3 s, so the plan is padded
  with its last position to `SIMLINGO_HORIZON_S` (3.0 s); NAVSIM and navhard score 4 s and set
  4.0. A padded point is a position the model did not predict, held still.
- **Control.** The agent's own steering and speed controllers do not run: paddock's client
  follows the plan, as for every model.
- **History.** SimLingo plans from the current frame alone, so frames before the scored one cost
  no forward pass (`observe`).

## Environment and weights

`setup.sh` builds a uv environment from the versions SimLingo's `environment.yaml` pins, with
three changes: torch 2.7.1 with CUDA 12.8 instead of 2.2.0 (2.2.0 has no kernels for Blackwell
GPUs), Python 3.10 instead of 3.8, and opencv-python-headless 4.8 instead of opencv-python 4.2,
which has no build for Python 3.10. It fetches both weight sets into the Hugging Face cache at
`HF_HOME`, checks every file's sha256 and links them into `{weights}`.

## Cost

Measured on an RTX PRO 6000 shared with other jobs: 2.04 GiB of GPU memory at most, 42 s to load,
and 1.3 to 2.3 s per plan, most of it the commentary SimLingo writes before it plans. In CARLA it
is asked on every 20 Hz tick, so Bench2Drive runs at about 0.02x real time.

## Results

### KITScenes

Not in this pull request.

### nuScenes

Not in this pull request.

### comma2k19

Not in this pull request.

### navhard

Not in this pull request.

### NAVSIM v1

Not in this pull request.

### NAVSIM v2

All 12,146 navtest scenarios (`navsim/simlingo/20261127T151159-59cab1`), `valid: true`: **EPDMS 0.471**, between
UniAD (0.478) and VAD (0.428). Its 2.5 s plan is padded with its last position to the 4 s scored
(`SIMLINGO_HORIZON_S = 4.0`, ranked with a note). Comfort is its weakest term (history comfort 0.346, extended
comfort 0.041), then ego progress (0.504).

### Bench2Drive

Not in this pull request.
