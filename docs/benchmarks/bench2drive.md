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
| `progress_km` | `Σ route_length_m × route_completion/100 / 1000` — distance actually exposed, and the denominator upstream's aggregate uses | context, not a score |
| collisions per progress km | a collision count divided by `progress_km`; the form upstream publishes (`statistics_manager.py:489-501`) | lower better |
| `min_speed_entries` | number of ego-to-traffic speed ratios `MinimumSpeedRouteTest` recorded — about 20 per route by construction, so a **count of checkpoints reached**, not of violations | context, not a score |
| `below_100_percent` | share of those recorded ratios below background-traffic speed | lower better |

The infraction-penalty coefficients are CARLA Leaderboard's: collision-vehicle
×0.60, collision-pedestrian ×0.50, collision-static ×0.65, red light ×0.70, stop
sign ×0.80, scenario timeout ×0.70, yield-to-emergency ×0.70. **Two terms are
switched off at the pinned revision** (`statistics_manager.py`): off-road is
weighted 0 and `MIN_SPEED_INFRACTION` is tagged `'unused'`. There is no direct
minimum-speed multiplier. Slow driving can still reduce
completion through blocking or timeout. Recorded minimum-speed entries include
checkpoint percentages above 100%, so their count alone is not speed compliance;
see the [harness audit and normalized tables](bench2drive-speed-normalization.md). An episode ends when the
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
# per_route_timeout_s = 5400         # default; must cover CARLA's boot + the route, see Cost
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

## Render quality

The pinned evaluator launches CARLA at its default (Epic) quality. A one-line
local edit, `-quality-level=Low` on the launch line in `leaderboard_evaluator.py`,
renders the sky black at midday and the wet-road decal layer as white speckle —
frames that exist in no training distribution and are not the benchmark's. It
leaves no trace in `run.json`, and it produced a full set of results at one site
before anyone looked at a frame. `preflight()` now refuses an evaluator that
contains the flag. If a site genuinely needs Low, make it a patch under
`patches/Bench2Drive/` so the change is on the record and travels with the pin.

## Cost

**The evaluator boots CARLA afresh for every route.** On the site measured that
is about ten minutes of Town12 map load per route (a little more at default
quality), before any driving. Wall clock per route is therefore roughly

    boot (~600 s)  +  game_time / realtime_factor

where a long route rides the 4000-frame cap (200 s of game time at 20 Hz), and
the real-time factor is the model's, not CARLA's. Measured on one machine, at
Low quality, single GPU:

| Model | real-time factor | mean wall / route | 220 routes |
|---|---:|---:|---:|
| Alpamayo 1.5 | 0.04–0.06× | ~2500 s | ~150 GPU-h |
| openpilot | ~0.13× | ~840 s | ~50 GPU-h |
| Drive-JEPA | 0.2–0.3× | ~470 s | ~30 GPU-h |
| VisionPilot | ~0.45× | ~370 s | ~25 GPU-h |

Two things follow. `per_route_timeout_s` has to cover boot **plus** the route:
the old default of 1800 s timed out every long Alpamayo route and recorded
nothing for it, which is why the default is now 5400 s. And a full sweep of all
four models at three repetitions is on the order of **750 GPU-hours** on one
card — run the slow model last, and gate on a small subset first (`--scene`
with a handful of route ids, or `routes_subset`) until every model completes
routes cleanly. The evaluator's own start-up (`time.sleep(30)` then a 600 s
client timeout, retried) does not add to this: the first connection attempt
returns as soon as CARLA answers, so the wait *is* the map load, not extra.

## Repetitions

`--runs N` reaches the evaluator as `--repetitions N`; the route's metrics are
the mean over its repetitions. A repetition whose record carries a crash status
(§ *What does not score*) is **dropped from that mean, not averaged and not
fatal**: with `--runs 3`, one crashed repetition must not throw away two that
drove, and its filled-in score must not count either. `n_runs` says how many
repetitions scored and `n_crashed` how many did not, so a thin average is
visible as thin. Only when every repetition crashed does the episode fail.

Run-to-run variance is real here — the same model on the same route has scored
0 and 100 on consecutive runs when its node did not engage — so treat a single
repetition as a smoke test, not a measurement.

## Efficiency and comfort

Every route writes `metric_info/<save_name>/metric_info.json` under the run's
artifact directory: per-tick ego acceleration, angular velocity, heading
vectors, position and rotation — the six fields Bench2Drive's
`tools/efficiency_smoothness_benchmark.py` reads (`AutonomousAgent.get_metric_info`
at the pinned revision), keyed by the `save_name` the evaluator hands the agent.
It is written whether or not `record` is on: these are metrics, not debugging
frames, and at a few hundred kilobytes a route they cost nothing.

Bench2Drive's own tools then score a sweep. They live in the checkout and expect
the leaderboard's layout — a folder of `results.json` files and one directory of
`metric_info` — which a paddock run provides:

```bash
RUN=/path/to/runs/bench2drive/<model>/<run_id>       # the run's artifact dir
mkdir -p $RUN/leaderboard && for d in $RUN/*-Town*/; do
  cp "$d/results.json" "$RUN/leaderboard/$(basename $d).json"; done
cd /path/to/Bench2Drive
python tools/merge_route_json.py -f $RUN/leaderboard/            # -> merged.json
python tools/ability_benchmark.py -r $RUN/leaderboard/merged.json
python tools/efficiency_smoothness_benchmark.py \
  -f $RUN/leaderboard/merged.json -m $RUN/metric_info/
```

`merge_route_json.py` assumes **exactly 220 routes** and silently scores a
missing one as 0, and it does not understand repetitions (it counts records):
run each repetition into its own run (three `paddock run … --runs 1`) if you
intend to feed the upstream tools, and check `n_expected` against `n_samples`
in `run.json` before believing a merged number.

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

`paddock run ... --resume <run_id>` continues a run in place: a route whose
`results.json` holds a record for every repetition is reused as it was read
(`harness.completed_metrics`), and every other route -- killed mid-drive,
crashed, never started -- is driven. A reused route carries no `duration_s`;
that is the harness's own clock and is not on disk.

## Running a sweep in parallel, and after a reboot

A full sweep is 220 routes of up to ~40 minutes each at the real-time factors in
*Cost*, so it runs as several `paddock run` workers side by side, one CARLA each.
`campaign.py` in this adapter keeps such a sweep moving, and
`tools/benchmarks/bench2drive/restart_campaign.sh` wraps it in one idempotent
command for after a reboot, a shutdown, or when finished workers leave the rest
of the routes unevenly spread:

```bash
export PADDOCK_CONFIG=/path/to/paddock.toml PADDOCK_PYTHON=/path/to/paddock/venv/bin/python
tools/benchmarks/bench2drive/restart_campaign.sh --model openpilot --workers 5 --held 11715
tools/benchmarks/bench2drive/restart_campaign.sh --model openpilot --workers 5 --held 11715 --rebalance
tools/benchmarks/bench2drive/restart_campaign.sh --model openpilot --status
# also serve the dashboard (tools/benchmarks/bench2drive/dashboard.py); --dashboard-root must
# contain {runs}, because live frames are served by path relative to it
tools/benchmarks/bench2drive/restart_campaign.sh --model openpilot --workers 5 \
    --dashboard-root /path/to/experiment --expert /path/to/expert [--features analysis.csv]
```

It starts, each in its own session so it outlives the shell, only what is not
already running:

| Piece | What it does |
|---|---|
| `campaign guard` | kills an evaluator that can no longer finish -- its log silent 15 min (a driving route logs every ~0.3 s), or its CARLA gone for 2 min -- so the worker records the route FAILED and moves on instead of waiting out `per_route_timeout_s` |
| `campaign launch` | every route with no scored attempt under `{runs}/bench2drive/<model>/` (any run id) and not queued in an alive worker, sorted and dealt longest-expected-first to the free worker slots; each worker is a fresh `paddock run --scene <its routes>` with a copy of the site config whose CARLA / traffic-manager ports sit 150 above the previous slot's, and its own node port base |
| `--held` | routes driven last, by one worker, once everything else has scored and no worker is alive |
| `--rebalance` | stops the alive workers first (their current routes are driven again) and deals everything left anew |
| `--dashboard-root` | the per-route video dashboard, on 127.0.0.1 |

The expected cost of a route is the mean wall time of its scenario's finished
routes in the same sweep, so a resumed sweep is dealt by what its scenarios
actually cost: a car that stops behind an obstacle rides the scenario timeout or
the tick cap (the blocked test is switched off while it "waits for a gap"),
which is ~30 minutes against ~5 for a route it completes.

Two failure modes this exists for, both met on a 220-route sweep at one site:

* **A CARLA that dies mid-route does not end it.** CARLA stopped delivering
  sensor data ("A sensor took too long to send their data"), or died inside
  `route_scenario.spawn_parked_vehicles`; the evaluator then deadlocked in its
  own shutdown, logging a "failed to destroy actor" line every 600 s, so the
  worker sat idle for the full 90-minute route timeout. The guard's second test
  -- the CARLA on the port the evaluator logged is gone -- catches the case
  silence alone never would.
* **A CARLA crash can take every worker down.** On a crash the evaluator runs
  `ps | grep -graphicsadapter=<rank> | kill -9` (see *Shared-machine
  behaviour*), which is every worker's CARLA on that GPU. Routes known to crash
  CARLA belong in `--held`: 11715 at the pin, and at that site the Town12/13
  `InterurbanAdvancedActorFlow` routes 23695, 23708 and 24078.

Worker configs, logs (`logs/worker<slot>.log`) and a launch history
(`workers.json`) live in `{runs}/bench2drive/_campaign/<model>/`. Every attempt
lands under `{runs}/bench2drive/<model>/<run_id>/` as usual, so the dashboard,
the analysis tools and `paddock ingest` (below, which takes the latest scored
attempt per route) see one sweep however many launches it took.

## Pooling recorded routes into a run (`paddock ingest`)

A Bench2Drive campaign is rarely one process: a route is re-run after a crash,
an expansion set is driven a week later, half a sweep runs on a second box. The
recordings are per route, so the pooled run that a leaderboard row needs is
assembled afterwards, by `paddock ingest --all` from `[[ingest.bench2drive]]`:

```toml
[[ingest.bench2drive]]                      # an array of tables: one block per run
model  = "drivejepa"
runs   = "/path/to/paddock-runs/bench2drive/drivejepa"   # or a list of roots
label  = "pooled50"
routes = ["1711", "1852"]                   # optional; default: every route found
notes  = "24 routes from the 03-04 Sep campaign, 26 from the 09 Sep expansion"

[ingest.bench2drive.model_env]              # recorded in config, over what ran
PADDOCK_CV_SPEED_M_S = "6"

[ingest.bench2drive.setups]                 # recording setup -> route ids
assisted = ["1711"]
```

What it guarantees, and what each key is for:

| | |
|---|---|
| `model` | checked against the `model` in each source run's own `run.json`. A route recorded for another model is refused, so `runs` pointed one component too high cannot pool two models into one row |
| `runs` | one root or several. Everything below them is walked: `<run_id>/<route>-<Town>-<Scenario>/results.json` |
| `routes` | the route subset to score. It selects what is scored, not what is seen: a route attempted and never scored is recorded in `config.routes_never_scored` either way |
| `label` | the run id is `<label>-<route hash>-<config hash>`. The config hash is why five constant-velocity speed bands over the same 50 routes are five directories and not one overwritten one |
| `model_env` | recorded in `config.model_env`, layered over what the source runs recorded. For the CV baseline it is the only machine-readable statement of which speed band a row is |
| `setups` | the configuration each route was actually driven under, aggregated into `metrics.json`'s `by_group`. A run pooling more than one is written `valid: false` — its mean is not a measurement — and `paddock report` lists it as excluded instead of ranking it |

Per route the ingest keeps the latest run id whose `results.json` *scores*: a
crashed attempt never shadows its rerun, because `parse_results` refuses a route
that did not run (see *What does not score*). `routes_file`, `waypoint_dt`,
`model_env` and the contributing sites are read from the harness's own `run.json`
(or the older `run_config.json`) beside the route folders rather than typed into
the ingest, and `config.source_config_read` says how many source runs kept one.
Aggregates are recomputed by the store from the per-route metrics; nothing is
transcribed from a report.

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
- The node log (`<model>_node.log` in the run dir) now carries the model's
  load/ready lines and a heartbeat every 100 `/infer` calls with mean latency
  and how far the returned plan reaches. A node that answers but never plans
  motion — a plan whose first waypoint stays at the origin for a whole route —
  is what "blocked at exactly 60 s with brake held" looks like from CARLA's
  side, and it is now visible from the node's.
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
| a model drives a route end to end through paddock | openpilot and VisionPilot completed routes (DS 100) through this adapter at one site; frames were at Low quality (see *Render quality*) and the run predates the crash-status and repetition handling above | moderate |
| `metric_info.json` matches what the upstream tool reads | field names and layout taken from `get_metric_info` and `efficiency_smoothness_benchmark.read_from_json` at 7ec25d1; not yet fed through the tool end to end | argued |
| crashed repetitions are dropped, not averaged | unit-tested over crashed, blocked and all-crashed triples | **strong** |
| an edited CARLA launch line is refused | unit-tested against a stub evaluator with and without the flag | **strong** |
| `--resume` reuses scored routes and drives the rest | unit-tested (scored, killed, torn, all-crashed, short of repetitions); on a real interrupted sweep it reused the 34 scored routes and drove the other 185 | **strong** |
| `campaign` deals what is left and the guard frees hung workers | selection, dealing, per-slot config and `--held` unit-tested (`tests/test_campaign.py`); the process side used on one 220-route sweep, where the guard freed two evaluators whose CARLA had died | moderate |

## September 2026 runs: what changed and what the numbers are

The first sweeps (2–3 Sep) were harness artifacts for two models; the fixes
below made the 24-route set (5 abilities × 5 routes minus 11715) the reference.

* **Route 11715 (Town12 EnterActorFlow) is excluded for every model.** CARLA
  segfaults inside `route_scenario.spawn_parked_vehicles` ~54 s in, on all nine
  attempts across three stacks, regardless of ego speed.
* **Wire format:** the agent now sends `speed_limit` (m/s, from
  `hero.get_speed_limit()`) and the next junction maneuver as `nav_maneuver` /
  `nav_distance_m` (walked forward on the 1 m world plan). Both optional; only
  VisionPilot (set-speed) and Alpamayo (navigation text) read them.
* **Timeouts:** `PADDOCK_INFER_TIMEOUT_S` (default 120) raises the per-tick
  HTTP timeout; heavy nodes under GPU contention need 600. Use
  `per_route_timeout_s = 14400` or more for slow models.
* **openpilot** runs with its comma-three rig (`CAM_OP_ROAD` 1928×1208 fov 40
  → `img`, `CAM_OP_WIDE` fov 119 → `big_img`; warp verified pixel-exact) and a
  reproduction of openpilot 0.11.2's longitudinal planner
  (`PADDOCK_OPENPILOT_NATIVE_LONG=1`, `PADDOCK_OPENPILOT_MODE=experimental`).
  Its model does not launch from standstill without a driver; the
  driver-resume rule (`PADDOCK_OPENPILOT_RESUME_S=3.0`: stopped 3 s, no lead →
  the e2e candidate is dropped until 1 m/s) is logged as `resume` in log.csv
  and every route is scored twice: **unassisted** (frozen at the first nudge,
  as if blocked there) and **assisted \*** (the full run). Tally the two from a
  finished run's own `results.json` + `log.csv`:
  `python -m paddock.adapters.benchmarks.bench2drive.tally <run_dir> --routes-xml <repo>/leaderboard/data/bench2drive220.xml`.
* **VisionPilot** cruises at the posted limit (its NeuroNCAP latch pinned it at
  1 m/s) and runs on the CUDA provider with two threads.
* **Alpamayo** receives the route as its navigation text ("Turn left in 37m")
  and samples with the reference notebook's settings.
* **Drive-JEPA** is the released NAVSIM-v2 checkpoint transferred zero-shot to
  CARLA. The paper's Bench2Drive result (DS 64.52) comes from a
  Bench2Drive-trained variant whose configs are in the repo but whose data
  pipeline, agent and checkpoint are unreleased; do not compare.
* **Official metrics beyond DS:** `tools/statistic_route_json.is_success`
  (Completed and no infraction except min-speed), and
  `tools/efficiency_smoothness_benchmark.py` on the per-tick `metric_info`
  the agent writes (needs scipy; the b2d conda env lacks it). Wrapper:
  `paddock-ops/bench2drive_official_metrics.py`.

Results on the 24 (04–06 Sep 2026), one run per route:

| model | DS | RC | Success | Efficiency | Smoothness | penalised events |
|---|---|---|---|---|---|---|
| Drive-JEPA (NAVSIM ckpt, zero-shot) | 31.11 | 40.4 | 0.0 % | 98.4 | 0.653 | 19 |
| VisionPilot | 49.85 | 75.5 | 25.0 % | 175.5 | 0.310 | 34 |
| Alpamayo 1.5 (nav) | 44.58 | 73.4 | 0.0 % | 180.7 | 0.598 | 34 |
| openpilot, no nudge | 30.09 | 30.8 | 12.5 % | — | — | 1 (+17 blocked) |
| openpilot, with nudge \* | 53.62 | 66.8 | 29.2 % | 171.6 | 0.410 | 21 |

Videos with the driven trajectory overlaid, every route, every model:
`<site video root>/Bench2drive/Simulation/<model>/<route>/{front.cam.mp4,bev.mp4}`
plus `route_table.md`. Raw runs: `{runs}/bench2drive/<model>/<run>/<route>/`. Machine-specific locations for this site are kept in the top-level repo's `docs/RUNNING_BENCH2DRIVE.md`, not here.
Per-scenario infractions, the nudge, and the metric definitions are on the
"Paddock Benchmark Metrics" artifact.
