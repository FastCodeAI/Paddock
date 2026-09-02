# Bench2Drive (CARLA closed-loop)

[Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) is a closed-loop
end-to-end driving benchmark on **CARLA 0.9.15**. The model drives the official
leaderboard's **220 short routes** (Towns 12/13/15, 44 scenario types × 5), one
scenario per route, and is scored by the CARLA Leaderboard driving score.

Closed-loop, like neuroNCAP: the model decides what happens next and errors
compound. The model runs behind paddock's HTTP policy node; the adapter spawns the
leaderboard evaluator per route and points a model-agnostic `AutonomousAgent`
(`agent.py`) at that node. Waypoints come back on paddock's 0.5 s grid and are
tracked by the Bench2Drive dual-PID controller, vendored in `pid.py` (see
*Notes*).

## What it measures

| Metric | Meaning | Direction |
|---|---|---|
| `driving_score` (primary) | `route_completion × infraction_penalty`, 0–100 | higher better |
| `route_completion` | % of the route driven before the episode ended | higher better |
| `infraction_penalty` | product of per-infraction coefficients, starts at 1.0 — a 0–1 **multiplier**, not a count | higher better |
| `collision_rate` | fraction of routes with ≥1 collision | lower better |

The infraction-penalty coefficients are CARLA Leaderboard's: collision-vehicle
×0.60, collision-pedestrian ×0.50, collision-static ×0.65, red light ×0.70, stop
sign ×0.80, plus continuous off-road / min-speed terms. An episode ends when the
car finishes, gets **blocked** (stuck too long), **deviates** >30 m off route,
**times out**, or hits the evaluator's 4000-frame cap; `route_completion`
freezes at whatever % was reached. All of those are the benchmark working and
they score.

### What does *not* score

The evaluator writes a numeric record for a route that never really ran and
marks it only in the record's `status`. The adapter reads that field and fails
the episode (`RuntimeError`, so the run is stored `partial` / `valid: false`)
for every failure the evaluator distinguishes:

| `status` | Cause | Evaluator exit code |
|---|---|---|
| `Failed - Agent crashed` | exception in `agent.py` — a node HTTP 500, a NaN plan, a missing sensor | **0** |
| `Failed - Agent couldn't be set up` | `setup()` raised | −1 |
| `Failed - Agent's sensors were invalid` | sensor layout rejected | −1 |
| `Failed - Simulation crashed` | CARLA died or the scenario failed to load | −1 |

A non-zero exit or a timeout fails the episode before `results.json` is read at
all. `results.json` existing is not a signal: the evaluator writes it, empty, at
startup. The distinction is tested in `tests/test_harness.py` against the
status strings in `statistics_manager.py` at the pinned revision.

## Site setup

CARLA and the Bench2Drive checkout live wherever this machine keeps them; declare
them in `paddock.toml` (never committed):

```toml
[benchmarks.bench2drive]
python      = "/path/to/conda/envs/b2d/bin/python"  # CARLA-client env, see prerequisites
carla_root  = "/path/to/carla0915"                   # contains CarlaUE4.sh
repo        = "/path/to/Bench2Drive"                 # has leaderboard/ + scenario_runner/
routes_file = "bench2drive220.xml"                   # under repo/leaderboard/data/ (default)
# routes_subset = "1711,1773,1790"   # pin the run to these route ids; recorded in run.json
# per_route_timeout_s = 1800         # default; long routes ride CARLA's 4000-frame cap
# carla_port = "2000"                # the evaluator moves up to the first free port from here
# tm_port    = "8000"                # traffic manager port, likewise
# gpu_rank   = "0"                   # CARLA's -graphicsadapter
# record     = "false"               # per-tick front/BEV JPEGs + log.csv; ~40 GB per full sweep
```

Every key above is read by the adapter; nothing else is. `routes_subset` changes
*what is measured*, so it is written to `run.json` and hashed into the run id.

Prerequisites (all outside this repo):

- **CARLA 0.9.15** installed at `carla_root` (`CarlaUE4.sh` must exist, and
  `PythonAPI/carla/agents/` — the agent imports `RoadOption` from it). The
  evaluator self-launches CARLA; you do not start it yourself.
- The **Bench2Drive checkout** at the revision pinned in `repos.yaml`
  (`leaderboard/` + `scenario_runner/` + the routes XML under `leaderboard/data/`).
  Nothing from Bench2DriveZoo is needed: the PID tracker its baselines use is
  vendored here.
- A **CARLA-client interpreter** (`python` above) that can import `carla`,
  `cv2`, `numpy` and `requests`. `carla` may be the pip package or the egg the
  0.9.15 release ships under `PythonAPI/carla/dist/`, which the adapter puts on
  the path itself. Python 3.8 is fine; `agent.py` and `pid.py` are kept 3.8-compatible.

`paddock doctor` runs the adapter's `preflight()`, which checks all of the above
— including a real import of everything `agent.py` needs, inside that
interpreter — so a missing piece is reported in seconds, not after CARLA has
booted on the first route.

## Running

```bash
paddock run --bench bench2drive --model drivejepa                 # all routes
paddock run --bench bench2drive --model drivejepa --scene 1773    # one route id
paddock run --bench bench2drive --model drivejepa --scene 1711,1773,Town13
paddock run --bench bench2drive --model all --scenarios ParkedObstacle,AccidentTwoWays
paddock run --bench bench2drive --model drivejepa --runs 3        # 3 repetitions per route, averaged
paddock run --bench bench2drive --model drivejepa --dry-run       # enumerate, start nothing
```

`--scene` selects route ids or towns (comma-separated; omit it for every route),
`--scenarios` filters by scenario type. An unknown id, town or type is reported
before the model loads. `--runs N` is forwarded as `--repetitions N`; the
per-route metrics are the mean over repetitions and `n_runs` records how many.

Each route leaves `results.json` and `evaluator.log` (the evaluator's stdout and
stderr, streamed as it runs, so a hung route can be read while it hangs) under
the run's artifact dir, alongside the canonical `run.json` / `metrics.json`
paddock writes. With `record = "true"` it also leaves front/BEV frames and a
per-tick `log.csv`.

## Shared-machine behaviour

- The evaluator is started in its own session and, on a timeout, that session
  is killed. CARLA runs in *another* session (the evaluator's doing) and is
  found by the `-carla-rpc-port=<port>` token in its command line — the port the
  evaluator printed, not the one requested — and killed; the port is then
  polled free before the next route. Nothing is killed by process name.
- Upstream caveat: on `Simulation crashed` the evaluator itself runs
  `ps | grep -graphicsadapter=<gpu_rank> | kill -9`, which takes down every
  CARLA on that GPU index, including other people's. That is the checkout's
  behaviour, not paddock's; on a shared box, use a `gpu_rank` nobody else does.
- Recording is off by default: a JPEG per camera per tick, a 512² BEV render per
  tick and a CSV flush, inside the synchronous tick, come to roughly 40 GB for a
  220-route sweep.

## Notes

- One model-agnostic agent (`agent.py`) serves every model. It renders exactly
  the cameras the model declares and calls the node; both come from `PADDOCK_*`
  variables the harness sets. It runs inside the CARLA-client interpreter with
  paddock's source on `PYTHONPATH`, and imports the wire contract (`Command`,
  canbus offsets, image encoding, `Trajectory` validation) from
  `paddock.policy.protocol` rather than restating it — so a NaN from a mis-warmed
  model is rejected in the agent, not handed to CARLA as a steer value.
- `pid.py` is the TCP dual-PID (lateral PID on the heading to a 4 m aim point,
  longitudinal PID on the speed error, gains 1.25/0.75/0.3 and 5.0/0.5/1.0),
  the same law Bench2DriveZoo's UniAD/VAD agents drive with. It lives here
  because the Bench2Drive checkout does not ship it and a control law under a
  score should be pinned where a change to it shows up in a diff.
- Bench2Drive is a real→sim transfer test: models trained on real data (nuScenes /
  NAVSIM) are out of distribution on CARLA frames, so expect them to drive partway
  and slow. A slow model (Alpamayo ~0.05× real-time) rides the per-route timeout.

## Verification status

| Claim | Evidence | Strength |
|---|---|---|
| `results.json → metrics` reproduces what CARLA recorded | golden test on a real 0.0.4 record (route 1773: 36.0 / 100 / 0.36) | **strong** |
| crash statuses fail the episode, poor outcomes score | status strings read from `statistics_manager.py` at 7ec25d1; parametrised test over all of them | **strong** |
| evaluator flags, exit codes, checkpoint-at-startup, CARLA port and session | read from `leaderboard_evaluator.py` at 7ec25d1; launch path tested against a stand-in evaluator (`tests/test_harness.py`) | **strong** for the harness, argued for the real evaluator |
| `agent.py` imports and runs its control path in the CARLA-client env | imported and exercised in the site's `b2d` env (Python 3.8, 0.9.15 egg) without a server | moderate |
| a model drives a route end to end through paddock | **not yet run** | unverified |
