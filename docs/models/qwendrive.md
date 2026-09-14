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
- keeps the 0.5 s grid of the 50-pose output, indices 4, 9, … 39: eight
  waypoints over 4 s with heading. NAVSIM scores all eight; neuroNCAP takes the
  first six; KITScenes truncates at 3 s.

| benchmark | frames | poses | command | note |
|---|---|---|---|---|
| KITScenes | 49 at 10 Hz, one `/infer_sequence`; ring cameras front, front-left, front-right | real, 4.8 s; speed only | always STRAIGHT | ring cameras sit at +60.9/−58.5° with ~87° HFOV against the ~70° nuPlan cameras the model trained on; zero-shot, not warped |
| NAVSIM v2 / v1 | `native_input_hz: 10` selects nuPlan's 10 Hz path (`[benchmarks.navsim] nuplan`): 16 observations over 1.5 s, frames picked every 0.5 s | 16 real poses with velocity, acceleration, yaw rate | NAVSIM's one-hot, round-tripped exactly | the model's own NAVSIM cameras (`cam_f0/l0/r0`) and history rate |
| neuroNCAP | one 1600×900 frame per 0.1 s tick, three cameras rendered | from `ego2world` each tick | the engine's | the adapter plans every tick, as every other model here |
| Bench2Drive | one frame per 20 Hz tick | from `ego2world` | route-derived | not yet run; 20 Hz × a VLM is the cost to plan for |

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
`QWENDRIVE_NUM_SAMPLES`, `QWENDRIVE_ATTN` and `QWENDRIVE_SEED` are the knobs;
`model.yaml` sets the first five. `num_samples > 1` executes the medoid draw;
the vendor's best-of-N figures select with the ground truth and are not
reproducible here.

## Training exposure

The vendor states the planner was trained on trajectories "standardized from
multiple public driving datasets" and reports NAVSIM navtest as its headline
number, so `trained_on` declares NAVSIM `navtrain`; see the Qwen-Drive × navsim
section of [comparison.md](comparison.md) for what that does to the NAVSIM
column. KITScenes and neuroNCAP (nuScenes, rendered) are zero-shot.

## Cost

No latency is published for the model. The adapter's `/diagnostics` reports
`infer_seconds_mean`; measure it on a `--limit` smoke before projecting a full
run. Expectations before measurement, direct mode, one sample, this card:
KITScenes n=223 in well under an hour; NAVSIM navtest in 4–8 h; neuroNCAP's
200 episodes in 8–11 h. Reasoning mode adds a 256-token decode per plan and
multiplies closed-loop cost several times over.
