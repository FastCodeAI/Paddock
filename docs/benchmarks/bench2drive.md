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
| `ended_stuck`, `ended_over_time`, `ended_collision` | with the early stop on (*Ending routes early*): share of routes that rule ended, 0/1 per route | lower better |
| `ended_manual` | share of routes a person ended by hand (a run with one is `valid: false`) | context, not a score |

The infraction-penalty coefficients are CARLA Leaderboard's: collision-vehicle
×0.60, collision-pedestrian ×0.50, collision-static ×0.65, red light ×0.70, stop
sign ×0.80, scenario timeout ×0.70, yield-to-emergency ×0.70. **Two terms are
switched off at the pinned revision** (`statistics_manager.py`): off-road is
weighted 0 and `MIN_SPEED_INFRACTION` is tagged `'unused'`. There is no direct
minimum-speed multiplier. Slow driving can still reduce
completion through blocking or timeout. Recorded minimum-speed entries include
checkpoint percentages above 100%, so their count alone is not speed compliance;
see the [harness audit and normalized tables](bench2drive-speed-normalization.md).
The published campaign calls every model once per simulator tick (20 Hz); what the
same routes score with each model driven at its own replan rate instead is in
[native inference rate](bench2drive-native-rate.md). An episode ends when the
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
# connect_timeout_s   = 60           # default; one attempt to reach a booting CARLA, see Cost
# carla_port = "2000"                # the evaluator moves up to the first free port from here
# tm_port    = "8000"                # traffic manager port, likewise
# gpu_rank   = "0"                   # CARLA's -graphicsadapter
# record     = "false"               # the per-tick record; ~100 KB a tick on paddock's rig
# record_inputs = "false"            # with record: each tick's whole request, for offline replay; ~2.5 MB a tick
# stop_stuck_s       = "30"          # end a route 30 s after its last 3 m of progress; see Ending routes early
# stop_expert_factor = "4"           # end a route at 4x the expert's time on it; needs `expert`
# stop_on_collision  = "true"        # end a route at the first collision the leaderboard records
# expert = "/path/to/expert"         # what `python -m tools.benchmarks.bench2drive.expert summarize` wrote
# camera_rig = "paddock"             # or "bench2drive" / "kitscenes": see Camera rig; any other is valid: false
```

Every key above is read by the adapter; nothing else is. `routes_subset` changes
*what is measured*, so it is written to `run.json` and hashed into the run id.
So do the `stop_*` settings, whose rules the run's notes also state, and a
`camera_rig` other than paddock's.

**How often the model is asked for a plan** is the model's own `planning_hz`
(`model.yaml`, `capabilities`). The client asks the node once per
1/`planning_hz` seconds and follows the last plan unchanged in between; a model
without it is asked every 0.05 s tick. Drive-JEPA declares 2 Hz, AutoE2E 10 Hz, METEOR 5 Hz. The rate is
recorded as `config.infer_hz` and hashed into the run id, and the host's
`PADDOCK_INFER_HZ` overrides it for every model. Where that replaces the model's
own rate, the run records the declared rate beside it (`config.planning_hz`) and
is stored `valid: false`: an operator's rate. A pooled record carries the rate
too, and ranks at the model's own (*Pooling recorded routes*, below).

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

## Camera rig

Every model is driven through the same cameras: paddock's rig, nuScenes' six
views at half nuScenes' 1600x900, plus openpilot's pair (`rig.py`). A site can
name another rig to ask how much a model's drive depends on its cameras:

| `camera_rig` | what it renders |
|---|---|
| `paddock` (default) | six 800x450 views, 70 deg (back 110), 1.5 m up |
| `bench2drive` | Bench2Drive's own rig, which its dataset was recorded with (`tools/data_collect.py` at 7ec25d1): the same six at 1600x900, 1.6 m up |
| `kitscenes` | KITScenes' roof pod, approximated from the calibration in `kitscenes.md`: an 88 x 54 deg front 1.8 m up and 87 x 63 deg side and rear views 2.0 m up |

Another rig changes what the model is shown, so it is in `run.json` and the run
id, and the run is `valid: false` with the rig named in its notes. It also
changes how the frames travel: a frame of a megapixel or more goes to the node
as JPEG (q95), the rest raw (`recorder.wire_codec`), so both other rigs' views
arrive JPEG-compressed where paddock's arrive raw, and a comparison between
rigs changes the two at once. The agent
writes the rig it rendered beside each recorded route as `rig.json`, and
`route_video` draws plans onto the front camera through it. The `kitscenes`
front differs in size from its ring: it suits a model that packs each view on its
own (AutoE2E), not one that stacks the raw frames.

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
routes cleanly.

The evaluator's own start-up is `time.sleep(30)`, then up to 20 connection
attempts. Upstream, each attempt waits the full client timeout (`--timeout`,
600 s), so a CARLA that needed more than 30 s to come up cost ten minutes before
the retry that connected at once. That happened on 3 of 10 routes of the
2026-10-16 gate smoke (`load_world failed , attempts=0` / `time-out of 600000ms`
/ `load_world success , attempts=1` in `evaluator.log`), about 30 minutes in
all. Patch `0002` gives the attempts a timeout of their own,
`--connect-timeout`, which the adapter passes as `connect_timeout_s` (60 s by
default); the client goes back to `--timeout` once connected, because that same
value bounds the large towns' world load and the agent and tick watchdogs.

## Ending routes early

Off unless the site sets it. A route the car will not finish still drives until
the leaderboard ends it, and the leaderboard is patient: its blocked test waits
60 s under 0.1 m/s, and the scenarios that make the ego wait for a gap
(AccidentTwoWays, ConstructionObstacleTwoWays, HazardAtSideLane, ...) switch that
test off, which leaves the 4000-tick cap. In the first 25 routes of the October
openpilot sweep, three stood still for 130–195 s of game time each, 27–42 min
of wall clock apiece.

Three settings end a route sooner (`early_stop.py`):

| setting | ends the route when |
|---|---|
| `stop_stuck_s` | the ego has gone this many game seconds without getting 3 m further along the route (the blocked test's 0.1 m/s over 30 s). Progress, not speed, so a car that a driver-resume nudge keeps creeping counts too. The clock pauses while the ego waits at a red or amber light -- inside the light's trigger box, or up to 50 m short of the next light on the route, which is where a car queued behind others stands -- so a short window does not end every route that meets a red signal, and the clock runs on from where it was when the light turns; it does run while the car yields to a pedestrian or waits for a gap, so a window of a few seconds cuts those scenes short too |
| `stop_expert_factor` + `expert` | the game clock reaches this multiple of Think2Drive's time on the route (`duration_s` of `<expert>/<route>/expert_traj.json`). The nine routes with no expert clip get the stuck rule alone; the run's config names the site's routes with none (`early_stop.no_expert`), so they are on the record and in the run id. Preflight refuses an `expert` directory with no `expert_traj.json` in it (drives fetched, never summarized) |
| `stop_on_collision` | the leaderboard's own `CollisionTest` records a collision with a static object, a vehicle or a pedestrian: the events the infraction penalty is computed from, read by the agent through the ScenarioManager, one tick after the contact. Not its `actual_value`, which also counts a brush with the sidewalk that is neither recorded nor penalised. A run with it is `valid: false`: Bench2Drive charges a collision as a penalty (×0.50–0.65) and lets the car drive on, so a route ended at one scores something else |

And one stop that is not a rule: a route ends **by hand** when its directory
grows a `stop_request` file while it drives, which the dashboard's "Stop route"
button writes (`POST /api/stop?route=<id>`, 127.0.0.1 only). The route is scored
as it stood, and the run is `valid: false`, its notes counting the routes ended
by hand: a moment a person chose is not one any rule reproduces. With `--runs N`
the request stops the route, every repetition left of it too: each of those ends
on its first tick, so the worker moves on.

The agent ends the route through the leaderboard's own tick cap, which `preflight`
checks the checkout still has, so the route is scored as it stands: the route
completion and infractions of that moment, status `Failed - TickRuntime`. Beside
`results.json` it writes `early_stop.json` with the rule that fired (`null` if
none did), the game time, and the route's own budget, and with `--runs N` one
such record per repetition under `repetitions`; the metrics carry it per route
as `ended_stuck` / `ended_over_time` / `ended_collision` / `ended_manual` (the
share of the route's repetitions, so 0 or 1 with one). The dashboard lists such a route as "Stopped: <reason>", and its video
says why it ended.

A route the stuck or over-time rule ends is a **failed drive**, and the run ranks. A car in a
driving scene is expected to move; one that gets no further along its route for
`stop_stuck_s`, or needs `stop_expert_factor` times the expert's time, has failed
the scene, so the route is scored as it stood when it ended. The rules are in the
run's config, and so in its run id, and its notes say which rules it ran under
and why its scores stand; the ingest carries both into a pooled run. What it
changes against a run to Bench2Drive's own end: a car that would have crept on
after the window, or finished just after its budget, scores what it had at the
cut. 1825 crept from 14 m to 20 m between 34 s and the 4000-tick cap (RC 15.19
without the rule, 10.65 with it), and 2802 was cut at 4x the expert's 19.1 s,
104 m along a 112 m route (RC 92.86). A pool of runs that stopped routes under
*different* rules, or of some runs with the rules and some without, does not rank:
one mean describes none of them.

Choose the factor from what the model needs, not from the expert: on the five
routes openpilot completed in the sweep above it took 1.6–2.9× Think2Drive's
time, so a factor of 1 would have cut every one of them short. Of Think2Drive's
own 211 drives, 3 stand still for longer than 30 s.

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
paddock writes. With `record = "true"` it also leaves front/BEV frames, a
per-tick `log.csv`, and the analysis record (`recorder.py`):

| file | what |
|---|---|
| `ticks.jsonl` | one line per tick: the node's whole reply (every waypoint, every `extras` field), what it was sent besides pixels (command, navigation, the `route_xy` polyline, speed), the actuation and `/infer` latency, and state the model never sees: actors within 60 m, the lane under the ego, progress along the route, the next light and stop sign |
| `route_plan.json` | the leaderboard's route at its own 1 m spacing, with the lights and stop signs on it |
| `actors.jsonl` | blueprint, role and extent of each actor, once |
| `model_<camera>/` | quarter-size thumbnails of every camera the model reads, at 10 Hz (openpilot's pair as `model_road` / `model_wide`) |
| `inputs/<tick>.npz` | only with `record_inputs = "true"` as well: the tick's whole request -- every model camera's frame exactly as sent (uint8, lossless) and the rest of the body -- so the route can be fed to a model again offline (`tools/benchmarks/bench2drive/replay.py`). ~2.5 MB a tick for six 800×450 views: a few routes' worth for a deep dive, not a sweep's |

None of it reaches the model, and a failure in it disables the record without
stopping the route.

### Seeing a route

Three tools under `tools/benchmarks/bench2drive/` read those recordings, for any
model; the driver's name comes from `{runs}/bench2drive/<model>/`:

```bash
# Think2Drive's own drives of the routes (Bench2Drive's expert), anno/ only, SHA-256 checked
python -m tools.benchmarks.bench2drive.expert fetch --out <expert> --routes 1711,1852
python -m tools.benchmarks.bench2drive.expert summarize --out <expert>
# a video and a summary still per route: front camera, BEV with plan / expert / route,
# the model's cameras and what it reported, route map, speed / progress / controls, score card
python -m tools.benchmarks.bench2drive.route_video --runs {runs}/bench2drive/<model> \
    --expert <expert> --routes-xml <repo>/leaderboard/data/bench2drive220.xml --out <dir>
# a dashboard on 127.0.0.1 that renders each route as it finishes; --root must contain {runs},
# because live frames are served by their path below it
python -m tools.benchmarks.bench2drive.dashboard serve --root <dir> --runs {runs}/bench2drive/<model> \
    --expert <expert> --routes-xml <repo>/leaderboard/data/bench2drive220.xml \
    [--routes 1711,1852] [--title "checkpoint, rate"] [--port 8765]
```

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
# a copy of the site config whose [roots] runs is this sweep's own (below)
export PADDOCK_CONFIG=/path/to/sweep/paddock.toml PADDOCK_PYTHON=/path/to/paddock/venv/bin/python
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

**Give each sweep its own runs root** (`[roots] runs` in a copy of the site
config). `launch` counts a route as done when any run under
`{runs}/bench2drive/<model>/` scored it, whatever settings that run was made
under, and `paddock ingest` pools the latest scored attempt per route: on a
root shared with runs started by hand, or with the gate's Bench2Drive lane, a
route one of those scored is never driven by the sweep, and its score is pooled
in with the sweep's.

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
* **A route that crashes CARLA crashes it every time.** Upstream's evaluator
  then ran `ps | grep -graphicsadapter=<rank> | kill -9`, every worker's CARLA
  on that GPU; with patch `0001`, which preflight requires (see *Shared-machine
  behaviour*), it kills only its own. Such a route is still dealt again on every
  launch while it has no score, so routes known to crash CARLA belong in
  `--held`, driven once at the end: 11715 at the pin, and at that site the
  Town12/13 `InterurbanAdvancedActorFlow` routes 23695, 23708 and 24078.

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
So is `declared_env`, the model's own `model.yaml` env. Source runs that recorded
different ones, or a mix of runs that recorded one and runs from before it was
recorded (#99), are pooled `valid: false` with a note: two settings of the model
behind one mean, the same reason as two recording setups. So is `infer_hz`, how
often the node was asked for a plan, judged from what the runs recorded: the pool
ranks at the rate the model's `model.yaml` declared for them (`planning_hz`). It
does not rank at the host's `PADDOCK_INFER_HZ` in its place, at two rates, or at a
rate not known because a run kept no config. A run that recorded no rate was asked
every tick; before #106, a rate set by hand reached the record only as
`PADDOCK_INFER_HZ` in `model_env`, and such a pool does not rank either.
Aggregates are recomputed by the store from the per-route metrics; nothing is
transcribed from a report.

## Shared-machine behaviour

- The evaluator is started in its own session and, on a timeout, that session
  is killed. CARLA runs in *another* session (the evaluator's doing) and is
  found by the `-carla-rpc-port=<port>` token in its command line — the port the
  evaluator printed, not the one requested — and killed; the port is then
  polled free before the next route. Nothing is killed by process name.
- Upstream, on `Simulation crashed` the evaluator runs
  `ps | grep -graphicsadapter=<gpu_rank> | kill -9`, which kills every CARLA on
  that GPU index that its user can signal: on a one-GPU box, every parallel
  sweep worker and gate lane of that user, and every job under a shared account.
  Patch `0001` makes it kill only its own server's process group. Preflight
  refuses an evaluator without both patches (`patches/README.md`).
- The node log (`<model>_node.log` in the run dir) now carries the model's
  load/ready lines and a heartbeat every 100 `/infer` calls with mean latency
  and how far the returned plan reaches. A node that answers but never plans
  motion — a plan whose first waypoint stays at the origin for a whole route —
  is what "blocked at exactly 60 s with brake held" looks like from CARLA's
  side, and it is now visible from the node's.
- Recording is off by default: a JPEG per camera per tick, a 512² BEV render per
  tick and a CSV flush, inside the synchronous tick. Measured ~100 KB a tick on
  paddock's rig (59 routes, 8,077 ticks): ~22 GB for 220 routes of 1,000 ticks,
  ~90 GB if every route rode the 4000-tick cap. The other rigs' larger frames
  make it ~280 KB a tick (`bench2drive`) and ~390 KB (`kitscenes`), one route
  each. `record_inputs` adds ~2.5–3 MB a tick on paddock's rig, for a few routes.

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
| the calibration the agent sends is each camera's CARLA mount in paddock's frames | `tests/test_calibration.py` runs the agent's own `setup()` for every camera of every rig, against the conversion written out one frame change at a time, the pinhole of the rendered view, a road point on the ego's left landing left of centre, and one 5 m to its left in front of the left cameras and behind the right ones; carla and the HTTP session are stand-ins, and 6 of its 10 tests fail with the agent's y-flip removed, the side-camera ones among them | **strong** for the formula; not compared with a live CARLA's matrices |

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

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Moved here from the README unchanged except for links; the symbols are
the README's former footnote marks, and each note below starts with the mark it
explains. Marks about a model rather than this benchmark (⊛ ⊕ ∗ ◊ ◇ ✦ ✧) are
explained in [What these models are](../models/comparison.md#notes-moved-from-the-readme-support-matrix).

### Notes on the matrix cells

◆ Historical Bench2Drive mean driving score on 24 routes (04–06 Sep 2026),
one run per route; route 11715 was excluded after CARLA crashes. These are
driving scores, not per-ability scores. OpenPilot ≈ is the **unassisted
pre-resume estimate**, reconstructed from the assisted recordings: 16 routes
are censored before intervention and eight had no intervention. It is not a
separate unassisted sweep. The historical OpenPilot scores predate later
controller fixes. The assisted and expansion recordings are reported
separately in [OpenPilot, by recording set](#openpilot-by-recording-set); they are
never pooled into one mean. See [Bench2Drive protocol](bench2drive.md).

VisionPilot's cell is its 20 Hz run over **all 50 routes**, 47.82
([run](../../results/bench2drive/visionpilot/pooled50-40c29d24c37d542a-d94009/run.json)):
the historical 24 (49.85, 03–04 Sep) and the 26 expansion routes (45.95, 09 Sep),
with the set-speed latch fix in place for both.

※ Bench2Drive driving score at the model's **native inference rate** — Drive-JEPA
and Qwen-Drive replan at 2 Hz, Alpamayo 1.5 at 10 Hz — over the 50-route set,
18–20 Sep 2026. Alpamayo 1.5 and Qwen-Drive are means over 49 routes: route 3800
crashes CARLA for both at their native rate and is reported as never scored, not
as a zero. Every other cell in this column but AutoE2E's (☉) and METEOR's (below) is driven at 20 Hz, one inference per
simulator tick; OpenPilot's native rate *is* 20 Hz, and VisionPilot was not
re-run. So the ※ cells are **not comparable with the ◆ cells**: a different rate
and a different route set. At 20 Hz the same three scored Alpamayo 1.5 44.58 and
Drive-JEPA 31.11 (24 routes, ◆) and Qwen-Drive 12.92 (50 routes). Drive-JEPA uses
its released NAVSIM-v2 checkpoint zero-shot in CARLA. The runs stay
`valid: false` and off the ranked [leaderboard](../leaderboard.md): each one's rate
was set by hand, and the leaderboard ranks a model at 20 Hz or at the rate its
`model.yaml` declares. Per-route detail and the 20 Hz comparison are in
[Native inference rate](#native-inference-rate--three-models-re-driven).

♣ The Bench2Drive baseline uses a **fixed 3 m/s target**, with straight ego-frame
waypoints and the shared controller; actual vehicle speed can differ. nuScenes
constant velocity instead extrapolates the recorded ego velocity. 48.31 is that
3 m/s target over all 50 routes (47.94 on the historical 24); the whole speed sweep, and why the
50-route table shows 40.39, is in
[Constant velocity — the speed sweep](#constant-velocity--the-speed-sweep).

✦ Bench2Drive constant velocity is a do-nothing floor: "constant velocity" means holding the current speed, and every route starts from a standstill, so the baseline holds zero and never departs -- blocked after 60 s, driving score 0 on every route (confirmed on 3 routes, structural for all 24). Unlike nuScenes, where the logged ego is already moving, it is not an informative baseline in closed loop.

☉ AutoE2E's cell, 15.30, is its mean driving score over the 50 routes with its
own inputs, the measured ego history and the route raster, asked for a plan at the
10 Hz its `model.yaml` declares
([run](../../results/bench2drive/autoe2e/pooled50-40c29d24c37d542a-cc7a9c/run.json),
`valid: true`): route completion 26.37 %, infraction penalty 0.60. A route ends
after 30 s without 3 m of progress or at 4x the expert's time, and is scored as a
failed drive: 42 of the 50 ended on the first rule, 1 on the second. The other rows
drove every route to Bench2Drive's own end, so the cell is marked R, and AutoE2E is
left out of the 50-route table below, whose infraction counts depend on how long
each route ran. Fed a
steady straight past and no route instead, which are not upstream's inputs, and
stopped at its first collision, it scored 37.74 on the same routes
([run](../../results/bench2drive/autoe2e/pooled50-steady-noroute-40c29d24c37d542a-8d26e9/run.json),
`valid: false`); why that past helps is in [the model's deep dive](../models/autoe2e.md).

METEOR's cell, 1.05, is its mean driving score over the 50 routes: TIER IV's
released ONNX as shipped, through paddock's six cameras in six of its eight slots
and with no driving command, which the released graph has no input for, asked for
a plan at the 5 Hz its `model.yaml` declares
([run](../../results/bench2drive/meteor/pooled50-40c29d24c37d542a-867d57/run.json),
`valid: true`): route completion 6.30 %, infraction penalty 0.90, under AutoE2E's
early-stop rules. The other rows drove every route to Bench2Drive's own end, so,
like AutoE2E's, the cell is marked R. On 43 of the 50 routes the car never leaves
the start. Standing still, METEOR plans a 3 s path that ends a median 0.17 m ahead, and its own
acceleration output is negative, so the shared controller, which brakes when a
plan asks for less than 0.4 m/s, holds the brake until a rule ends the route: the
stuck rule at 30 s on 30 of them, the over-time rule on 1, and Bench2Drive's own
blocked test at 60 s on the 12 that start about 30 m short of a red light, which
pauses the stuck clock. On the 7 routes it drives, it collides on every one. Like
AutoE2E, it is left out of the 50-route table below. Detail in
[meteor.md](../models/meteor.md#reference-numbers).

### Bench2Drive — 50 routes

Five models and constant velocity at five fixed target speeds, evaluated on the
same 50 routes. DS = mean driving score; RC = mean route completion.
**Progress km** is `Σ route_length_m × RC/100 / 1000` — how far each system was
actually exposed, and the denominator Bench2Drive's own aggregate uses. The
infraction columns are **recorded counts, not rates**, and the exposures differ
by nearly 2× (DriveJEPA 2.14 km against CV 3 m/s's 4.09), so compare them through
the per-km table below rather than across this one.

| Model/system | Routes | DS ↑ | RC ↑ (%) | Progress km | Ped. collision ↓ | Vehicle collision ↓ | Layout collision ↓ | Red light ↓ | Stop sign ↓ | Outside lanes ↓ | Deviation ↓ | Blocked ↓ | Emergency yield ↓ | Min-speed entries | Route timeout ↓ | Scenario timeout ↓ | Collisions total ↓ | Routes with collision ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DriveJEPA | 50/50 | 31.71 | 39.10 | 2.14 | 2 | 3 | 3 | 1 | 1 | 7 | 6 | 17 | 3 | 338 | 0 | 0 | 8 | 8/50 |
| VisionPilot | 50/50 | 47.82 | 71.19 | 3.92 | 6 | 23 | 13 | 11 | 3 | 11 | 10 | 2 | 4 | 655 | 0 | 0 | 42 | 29/50 |
| Alpamayo 1.5 (nav) | 50/50 | 43.04 | 72.32 | 3.98 | 3 | 42 | 1 | 0 | 0 | 22 | 3 | 9 | 1 | 660 | 0 | 0 | 46 | 27/50 |
| OpenPilot † | 50/50 | — | — | 3.46 | 3 | 10 | 5 | 5 | 4 | 4 | 9 | 10 | 4 | 587 | 0 | 0 | 18 | 14/50 |
| QwenDrive ‡ | 50/50 | 12.92 | 50.19 | 2.68 | 0 | 50 | 60 | 1 | 0 | 46 | 9 | 16 | 0 | 437 | 0 | 0 | 110 | 45/50 |
| CV (fixed 3 m/s) | 50/50 | 48.31 | 74.65 | 4.09 | 8 | 27 | 14 | 5 | 4 | 11 | 11 | 1 | 5 | 688 | 0 | 0 | 49 | 27/50 |
| CV (fixed 6 m/s) | 50/50 | 40.39 | 73.08 | 3.98 | 9 | 24 | 26 | 11 | 4 | 14 | 14 | 4 | 5 | 673 | 0 | 0 | 59 | 31/50 |
| CV (fixed 9 m/s) | 50/50 | 43.57 | 72.70 | 3.97 | 6 | 24 | 14 | 11 | 4 | 13 | 15 | 3 | 4 | 667 | 0 | 0 | 44 | 29/50 |
| CV (fixed 12 m/s) | 50/50 | 42.19 | 72.12 | 3.92 | 4 | 35 | 16 | 11 | 4 | 14 | 13 | 3 | 4 | 659 | 0 | 0 | 55 | 28/50 |
| CV (fixed 15 m/s) | 50/50 | 41.45 | 73.68 | 4.02 | 4 | 37 | 17 | 11 | 4 | 15 | 15 | 4 | 4 | 676 | 0 | 0 | 58 | 28/50 |

† **OpenPilot has no DS or RC cell here, deliberately.** Its 24 historical routes
were driven with human resume assistance on 16 of them, and its 26 expansion
routes use a different upstream longitudinal setup; no single mean describes both,
so none is published and nothing ranks OpenPilot against the other systems. The
counts in its row are the recorded totals over all 50 routes. The scores are in
[OpenPilot, by recording set](#openpilot-by-recording-set), and its tracked run
carries the same split in `by_group` with `valid: false` on the pooled mean.
DriveJEPA uses a NAVSIM-v2 checkpoint zero-shot; VisionPilot uses a synthesized trajectory with an extrapolated tail.
‡ QwenDrive is zero-shot, on the three cameras it was trained for (front,
front-left, front-right), through the same tracker as every other row; its route
24333 entry is the valid rerun after a startup-guard fix. Its adapter is in this
tree, but its `model.yaml` does not declare `bench2drive`, so this row is
ingested from the recordings rather than re-drivable with `paddock run` —
[`docs/models/qwendrive.md`](../models/qwendrive.md).
(nav) marks the Alpamayo run that receives the route's navigation instruction (manoeuvre and distance) on the wire each tick.
CV rows hold the stated fixed target speed; the score depends strongly on the chosen speed, so all five bands are shown.

#### Collisions per progress km

Upstream divides every infraction except `outside_route_lanes` by the same
`km_driven`, so this is the form the counts above are comparable in. Raw counts
are the one form Bench2Drive itself never publishes.

| Model/system | Progress km | Ped. ↓ | Vehicle ↓ | Layout ↓ | Total ↓ |
|---|---:|---:|---:|---:|---:|
| DriveJEPA | 2.14 | 0.94 | 1.40 | 1.40 | 3.74 |
| VisionPilot | 3.92 | 1.53 | 5.87 | 3.32 | 10.72 |
| Alpamayo 1.5 (nav) | 3.98 | 0.75 | 10.56 | 0.25 | 11.56 |
| QwenDrive ‡ | 2.68 | 0.00 | 18.66 | 22.40 | 41.06 |
| CV (fixed 3 m/s) | 4.09 | 1.95 | 6.60 | 3.42 | 11.97 |
| CV (fixed 6 m/s) | 3.98 | 2.26 | 6.02 | 6.53 | 14.81 |
| CV (fixed 9 m/s) | 3.97 | 1.51 | 6.05 | 3.53 | 11.09 |
| CV (fixed 12 m/s) | 3.92 | 1.02 | 8.93 | 4.08 | 14.03 |
| CV (fixed 15 m/s) | 4.02 | 1.00 | 9.21 | 4.23 | 14.44 |

DriveJEPA's 3 vehicle collisions against VisionPilot's 23 is 1.40/km against
5.87/km once exposure is divided out, and part of even that is DriveJEPA having
the table's highest blocked count (17) — stationary, not safe. QwenDrive is the
other direction: 41.06 collisions per progress km, an order of magnitude above
every other row. OpenPilot is in its own block below; its two recording sets are
not pooled into a rate either.

#### Minimum-speed normalization

| Model/system | Recorded entries | Below 100% ↓ (%) |
| --- | ---: | ---: |
| DriveJEPA | 338 | 59.17 |
| VisionPilot | 655 | 21.22 |
| Alpamayo 1.5 (nav) | 660 | 44.09 |
| QwenDrive ‡ | 437 | 53.32 |
| CV (fixed 3 m/s) | 688 | 67.73 |
| CV (fixed 6 m/s) | 673 | 46.66 |
| CV (fixed 9 m/s) | 667 | 13.04 |
| CV (fixed 12 m/s) | 659 | 9.71 |
| CV (fixed 15 m/s) | 676 | 7.99 |

“Below 100%” is the share of recorded ratios below background-traffic speed, not
a safety violation rate: B2D records a ratio at each distance checkpoint even
when the ego is faster than traffic, and lawful stops produce ratios below 100%.
**Entries per progress km is not published**, because it cannot separate these
systems: the checkpoint grid splits *each* route into 20 parts of its own length,
so a route contributes ~20 entries however long it is, and the quantity reduces
to 20 / (completion-weighted mean route length) — 181.7 to 187.3 across all ten
systems, a 3.1% spread, against the 8.5× spread the speed column shows over the
same rows. What is left is the route mix and how far each system got, not how
fast it drove. `below_100_percent` is the column carrying the speed signal;
`entries_per_progress_km` stays in the JSON with that caveat attached.
OpenPilot's entries are in its own block; the two recording sets are not pooled here either.

#### OpenPilot, by recording set

Three different experiments, kept apart. They are never averaged together, and the
pooled 50-route mean they would produce is withheld from
`bench2drive-speed-normalization.json` for the same reason.

| Recording set | Routes | DS ↑ | RC ↑ (%) | Min-speed entries | Below 100% ↓ (%) |
|---|---:|---:|---:|---:|---:|
| Historical 24 — **assisted**, human resume on 16 of 24 | 24 | 53.62 | 66.81 | 300 | 36.00 |
| Expansion 26 — unassisted, different upstream longitudinal setup | 26 | 43.01 | 60.45 | 287 | 58.54 |
| Unassisted 24 estimate ◆ — assisted runs censored at the first resume | 24 | 30.09 | 30.76 | — | — |

The third row is the reconstruction behind ◆ and the 30.09 in the support matrix,
not a measurement: 16 routes are censored before intervention and eight had no
intervention. Its per-route values are
[`openpilot-24-infractions.csv`](openpilot-24-infractions.csv), and
every record in the pooled JSON now carries an `assistance` block, so the assisted
and unassisted subsets can be recomputed from the linked file alone.

#### Constant velocity — the speed sweep

The baseline is not one experiment. All five targets were driven over the same
50 routes; the 6 m/s row is the one carried into the tables above.

| Target speed (m/s) | Routes | DS ↑ | RC ↑ (%) |
|---:|---:|---:|---:|
| 3 | 50/50 | 48.31 | 74.65 |
| 6 | 50/50 | 40.39 | 73.08 |
| 9 | 50/50 | 43.57 | 72.70 |
| 12 | 50/50 | 42.19 | 72.12 |
| 15 | 50/50 | 41.45 | 73.68 |

This fixes the target speed rather than holding the zero speed at route start;
the path is straight in the ego frame, not a route-following reference. The
**48.31 ♣** in the support matrix is the 3 m/s target over all 50 routes
(RC 74.65); the same setting scores 47.94 over the *historical 24* (RC 74.68), and
the two are separate runs in the store:
[all 50](../../results/bench2drive/constant_velocity/pooled50-3mps-40c29d24c37d542a-02b6ac/run.json)
· [historical 24](../../results/bench2drive/constant_velocity/20260908T084921-804913/run.json).

[Calculation](bench2drive-speed-normalization.md) · [Recorded data](bench2drive-speed-normalization.json)

#### Native inference rate — three models re-driven

Every row above is driven at **20 Hz**: one inference per simulator tick, which is
a property of the protocol, not of the models. Three of them were built to replan
far more slowly. Re-driving the same 50 routes at that slower rate — 18–20 Sep 2026,
same routes, same `waypoint_dt`, same tracker, `PADDOCK_INFER_HZ` the only knob
turned deliberately — scores all three higher, so **the 20 Hz protocol understates
them**.

| Model | Native rate | Routes | DS 20 Hz | DS native ↑ | RC 20 Hz | RC native ↑ | Routes completed |
|---|---:|---:|---:|---:|---:|---:|---:|
| DriveJEPA | 2 Hz | 50/50 | 31.71 | **57.60** | 39.10 | 70.97 | 5 → 23 |
| Alpamayo 1.5 (nav) | 10 Hz | 49/50 | 43.20 | **55.82** | 73.07 | 76.52 | 24 → 27 |
| QwenDrive ‡ | 2 Hz | 49/50 | 12.94 | **17.08** | 50.64 | 64.30 | 6 → 14 |

The 20 Hz columns are the published recordings **restricted to the routes scored at
both rates**, so for Alpamayo 1.5 and QwenDrive they are not the all-50 means in the
table above. DriveJEPA and QwenDrive gain by driving further instead of stalling
(blocked 17 → 7 and 16 → 10), which also raises their collision counts; Alpamayo 1.5
is the only one that gets safer — infraction penalty 0.648 → 0.725, vehicle
collisions 42 → 23. **A higher score here is partly more exposure, not only better
driving** — though normalized for that exposure all three collide *less* often per
progress-km (DriveJEPA 3.74 → 3.05, Alpamayo 1.5 11.76 → 6.50, QwenDrive 41.48 →
31.25), so the extra distance is not bought with a higher collision rate. Route 3800 is unscorable for Alpamayo 1.5 and QwenDrive (CARLA segfaults;
DriveJEPA completes it and scores 100) and is recorded as requested-and-never-scored
— `n_failed: 1`, `status: partial` — never as a zero. **OpenPilot's native rate is
20 Hz**, so its published run is already its native-rate run, and VisionPilot was not
re-run; nothing here restates either. The rate is recorded per run in
`config.model_env`, not derived from `model.yaml`. They were driven before the
adapter honoured a rate on `main`; it now does (a model's `planning_hz`, above),
and Drive-JEPA declares 2 Hz, so a clean checkout drives it at that rate by
default. CARLA re-runs are not bit-reproducible, so these stay landed results. The three runs are `valid: false` and **do not rank**
on the board: each rate was set by hand, in the runs' `model_env`, and the board
ranks a model only at 20 Hz or at the rate its `model.yaml` declares. Their native scores are the Bench2Drive cells of the support matrix for
these three models (※). `model_env` also differs in more than the rate — thread counts, a trace
sink, `ALPAMAYO_NAV` — each argued immaterial and none measured to be; the
[page](bench2drive-native-rate.md) books that as argued, not
verified.

Tracked runs, one per model:
[DriveJEPA](../../results/bench2drive/drivejepa/native50-2hz-40c29d24c37d542a-447e23/run.json) ·
[Alpamayo 1.5](../../results/bench2drive/alpamayo15/native50-10hz-084418258d833a9f-c4d076/run.json) ·
[QwenDrive](../../results/bench2drive/qwendrive/native50-2hz-084418258d833a9f-9b2bc9/run.json)

[Protocol, full infraction breakdown, and what is argued rather than measured](bench2drive-native-rate.md) · [Per-route records](bench2drive-native-rate.json)

Tracked run records for every row above, recomputed from the per-route `results.json`
files by `paddock ingest` (`[[ingest.bench2drive]]`, one pooled closed-loop run per
model or speed band): `results/bench2drive/<model>/pooled50-*`. `paddock report` reads
them like any other run.

[VisionPilot and OpenPilot failure videos](../evidence/driving-videos/README.md) — manifest and hashes tracked; the MP4s live under `{runs}`, not in the repo.
