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

## Open: the original setup is not reproducible over time

Both runs above disagree with what the same scenario recorded in August:

| run | date | ncap_score | impact speed |
|---|---|---:|---:|
| `outputs/runs1` | 2026-08-27 | 3.145 | 1.777 |
| `outputs/runs1_fixed` | 2026-08-29 | 3.388 | 1.272 |
| paddock **and** original server | 2026-09-01 | 2.827 | 2.438 |

Reference speed is 8.314 in all three, so the scenario itself is unchanged; the
ego is arriving at the actor differently. Two runs today agree exactly, so it is
deterministic *now* — something changed between August and now rather than the
run being noisy.

This is not a paddock regression: the original code reproduces paddock's number,
not August's. But it means **the August neuroNCAP figures cannot currently be
reproduced**, and any comparison against them is unsafe until the cause is found.
Candidates not yet eliminated: renderer or checkpoint state, a driver or library
update beneath onnxruntime, or a change in the neuro-ncap checkout since the runs.

Recording it rather than quietly adopting the new number: a benchmark whose
scores drift without explanation is the thing this framework exists to prevent.
