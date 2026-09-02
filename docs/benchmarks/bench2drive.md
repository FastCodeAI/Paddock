# Bench2Drive (CARLA closed-loop)

[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) is a closed-loop
end-to-end driving benchmark on **CARLA 0.9.15**. The model drives the official
leaderboard's **220 short routes** (Towns 12/13/15, 44 scenario types × 5), one
scenario per route, and is scored by the CARLA Leaderboard driving score.

Closed-loop, like neuroNCAP: the model decides what happens next and errors
compound. The model runs behind paddock's HTTP policy node; the adapter spawns the
leaderboard evaluator per route and points a model-agnostic `AutonomousAgent`
(`agent.py`) at that node.

## What it measures

| Metric | Meaning | Direction |
|---|---|---|
| `driving_score` (primary) | `route_completion × infraction_penalty` | higher better |
| `route_completion` | % of the route driven before the episode ended | higher better |
| `infraction_penalty` | product of per-infraction coefficients, starts at 1.0 | higher better |
| `collision_rate` | fraction of routes with ≥1 collision | lower better |

The infraction-penalty coefficients are CARLA Leaderboard's: collision-vehicle
×0.60, collision-pedestrian ×0.50, collision-static ×0.65, red light ×0.70, stop
sign ×0.80, plus continuous off-road / min-speed terms. An episode ends when the
car finishes, gets **blocked** (stuck too long), or **deviates** >30 m off route;
`route_completion` freezes at whatever % was reached.

## Site setup

CARLA and the Bench2Drive checkout live wherever this machine keeps them; declare
them in `paddock.toml` (never committed):

```toml
[benchmarks.bench2drive]
python      = "/path/to/conda/envs/b2d/bin/python"  # CARLA-client env (carla==0.9.15)
carla_root  = "/path/to/carla0915"                   # contains CarlaUE4.sh
repo        = "/path/to/Bench2Drive"                 # has leaderboard/ + scenario_runner/
routes_file = "bench2drive220.xml"                   # under repo/leaderboard/data/ (default)
per_route_timeout_s = 1800                            # long routes ride CARLA's 4000-frame cap
# carla_port = "2000"   # optional; tm_port, gpu_rank also overridable
```

Prerequisites (all outside this repo):

- **CARLA 0.9.15** installed at `carla_root` (the binary `CarlaUE4.sh` must exist).
  The evaluator self-launches CARLA — you do not start it yourself.
- The **Bench2Drive checkout** (`leaderboard/` + `scenario_runner/` + the routes
  XML under `leaderboard/data/`).
- A **CARLA-client conda env** (`carla==0.9.15` + leaderboard/scenario_runner deps).

Check readiness with `paddock doctor`; the adapter's `preflight()` reports any
missing piece in seconds rather than mid-sweep.

## Running

```bash
paddock run --bench bench2drive --model drivejepa                 # all routes
paddock run --bench bench2drive --model drivejepa --scene 1773    # one route id
paddock run --bench bench2drive --model all --scenarios ParkedObstacle,AccidentTwoWays
paddock run --bench bench2drive --model drivejepa --dry-run       # enumerate, start nothing
```

`--scene` selects a route id or a town (default "0103" from the shared CLI means
*no filter* — all routes); `--scenarios` filters by scenario type. Each route
records its images, a per-tick `log.csv`, front/BEV frames, `results.json`, and
the evaluator logs under the run's artifact dir, alongside the canonical
`run.json` / `metrics.json` paddock writes.

## Notes

- One model-agnostic agent (`agent.py`) serves every model; it renders exactly the
  cameras the model declares (`PADDOCK_CAMERAS`) and calls the node
  (`PADDOCK_NODE_URL`) — both set by the harness. This replaces the per-model
  CARLA agents used before paddock integration.
- Bench2Drive is a real→sim transfer test: models trained on real data (nuScenes /
  NAVSIM) are out of distribution on CARLA frames, so expect them to drive partway
  and slow. A slow model (Alpamayo ~0.05× real-time) rides the per-route timeout.
- The golden test (`tests/test_harness.py`) pins the `results.json → metrics`
  parse against a recorded route, so scoring changes fail loudly.
