# Constant velocity (the floor)

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | none: paddock's own code |
| Adapter | [`adapters/models/constant_velocity/`](../../src/paddock/adapters/models/constant_velocity/) |
| Environment | uv, Python 3.11 |
| Cameras | `CAM_FRONT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [alpasim](../benchmarks/alpasim.md), [nuscenes](../benchmarks/nuscenes.md), [bench2drive](../benchmarks/bench2drive.md), [comma2k19](../benchmarks/comma2k19.md), [nuplan](../benchmarks/nuplan.md), [hugsim](../benchmarks/hugsim.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

Not a model. It holds the ego's measured speed on a straight line and never
steers, and it is here because **a closed-loop score is uninterpretable without
it**.

## Why a non-model is in the model registry

AlpaSim's `scene_score` zeroes a scene outright on an at-fault collision or an
offroad event, whatever progress preceded it. Constant velocity commits neither:
it does not steer into anything, and a rear-end collision — the failure it
*does* cause — is not at fault. On the `public_2601` first-10 set under the
shared 20 Hz referee it therefore outscores two of the three real models in its
group:

| | `scene_score`, `1cam-20Hz` | progress |
|---|---|---|
| OpenPilot supercombo 0.2.0 | 0.957 | 0.901 |
| **constant velocity 0.2.3** | **0.885** | **0.829** |
| VisionPilot v2 2.0.0 | 0.849 | 0.820 |
| Drive-JEPA 0.1.4 | 0.503 | 0.565 |

Read without the floor, that table says two driving models are worse than
doing nothing. Read with it, it says what it actually measures: the metric
punishes at-fault failure, and a policy incapable of being at fault is hard to
beat on it. Every AlpaSim number should be quoted against the floor for its
group — see [../benchmarks/alpasim.md](../benchmarks/alpasim.md).

## What it does

```
speed  = canbus[13]           # m/s, as the benchmark supplied it
plan_i = speed * dt * i       # i = 1..6, straight ahead, y = 0
```

Deterministic by construction, which is the other reason it is useful: any
difference between two runs of the floor is the harness, never the policy. That
is what makes it the right thing to verify a new benchmark adapter against.

An absent, non-finite or negative speed becomes a stationary plan rather than a
guess — a floor that invented a speed would be driving on its own invention.

## Setup

```bash
bash src/paddock/adapters/models/constant_velocity/setup.sh
```

Seconds: no repo, no weights, numpy only. A site that would rather not keep a
venv for four lines of arithmetic can point at paddock's own interpreter
instead, which already has numpy:

```toml
[models.constant_velocity]
venv = "/path/to/paddock/.venv"
```

## Limits

It reads no image, so it is blind to everything a benchmark is testing. It is a
denominator, not a baseline to beat in any interesting sense — a model that only
just clears it has not demonstrated much, and one that falls below it should be
read as failing *at fault*, not as failing to move.

## On nuScenes and Bench2Drive

The same four lines serve the open-loop protocol and CARLA:

```bash
paddock run --bench nuscenes    --model constant_velocity
paddock run --bench bench2drive --model constant_velocity
```

On nuScenes its L2 grows with the horizon exactly as the road curves away from
a straight line, and it still beats every learned model on the cumulative
average (`README.md`, nuScenes table): ego motion in the val split is mostly
straight. On Bench2Drive it drives at its entry speed until the route turns or
something stops it. NAVSIM uses its own native constant-velocity agent
(`paddock_constant_velocity`), not this adapter.
