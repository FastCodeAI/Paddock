# openpilot's driving model on Bench2Drive-220: why it fails

This note explains why openpilot's driving model (supercombo) scores what it does on Bench2Drive's 220 CARLA
routes, driven closed loop through paddock at the adapter's defaults. Every number is recomputed from the per-tick
record by [`openpilot_drv_supcombo_bench2drive_lib.py`](openpilot_drv_supcombo_bench2drive_lib.py) (the failure
catalogue) and [`openpilot_drv_supcombo_bench2drive_report.py`](openpilot_drv_supcombo_bench2drive_report.py) (the
generated report, whose tables are not repeated here in full). How the record is made and read is in
[docs/design/driving-failure-analysis.md](../../../../docs/design/driving-failure-analysis.md).

## The short answers

**Headline** (measured). 218 of 220 routes scored; the other two crash CARLA on every attempt (§7).

| Metric | Value |
|---|---|
| Driving score, mean over the scored routes | **33.2** |
| Driving score, the two unscored routes as 0 (how `merge_route_json.py` divides) | **32.9** |
| Route completion | 40.3 % |
| Infraction penalty | 0.884 |
| Success (completed, no penalised infraction) | 8 routes, 3.7 % |
| Routes driven to 100 % completion | 28 |

For scale, Bench2Drive's published results on these routes come from models trained on CARLA data: TCP 51.4, UniAD
38.7, VAD 38.7 and ADMLP 35.4. openpilot has seen no CARLA frame.

**Where the score went** (measured: driving-score points lost against 100, by each route's primary cause):

| Failure family | Routes | Mean DS | Share of all lost points |
|---|---:|---:|---:|
| **Standstill trap**: stops, and does not start again | 93 | 16.5 | **53 %** |
| **Route following**: no route input, misses turns and exits | 50 | 45.4 | 19 % |
| **Blocked, no overtake**: stops behind an obstacle it would have to pass | 31 | 30.4 | 15 % |
| **Collision**: then stands until the end | 16 | 18.2 | 9 % |
| Completed, with an infraction | 20 | 69.8 | 4 % |
| Success | 8 | 100 | 0 |

1. **Half the lost score is one behaviour: the car stops and does not start again** (measured; mechanism argued, §1).
   - **44 routes never pulled away from the start.** 49 more stopped later and stood until the route ended, 41 of
     them with a clear lane ahead.
   - Standing still, supercombo plans a median 0.14 m of travel over its whole 3 s output horizon.
   - **Of 126 stops after the start, 96 never ended.**
2. **Without a route, it drives the lane, so turns become deviations** (measured, mechanism verified in code, §2).
   - The adapter's default sends no desire, so the turn never reaches the model.
   - **It completed 0 of 31 left turns** it reached, and 5 of 22 right turns, each by following a right-turn lane.
   - 41 routes ended there; 9 more ended at a lane change or exit.
3. **It stops correctly for obstacles, then never goes around them** (measured, §3).
   - 31 routes stood behind something in the lane until the route ended: a stationary car (22), cones or a parked
     mesh (8), or queued traffic (1).
   - 21 of the 30 behind a stationary car or cones had stopped more than 15 m short of it, a median 33 m.
   - Overtaking is its weakest ability: **DS 20.6**.
4. **Red lights: it usually stops, but often goes again on red** (measured, §4).
   - It stopped at 40 of the 52 lights where it saw red on the approach.
   - Of its **22 red-light infractions, 13 came after it had stopped for that red**.
5. **Collisions are the smallest cost**, and in many it was already braking (measured, §5).
   - 38 collisions on 27 routes: 22 vehicles, 14 static objects, 2 pedestrians.
   - In 21 of the 38 it was braking hard in the 2 s before.
6. **Its pace is fine; its time stopped is not** (measured, §6).
   - While moving it averages 6.43 m/s against the expert's 6.44 m/s, on 165 routes with both.
   - It stands still 61 % of the time; the expert stands still 23 %.
7. **No sign that night or fog hurt it** (measured; the reading is argued, §6).
   - Night routes score 36.1 against 31.3 for day.
   - Rain correlates with *better* driving: DS 37.7 vs 28.9, and 88 % launched vs 72 %. Weather is confounded with
     town and scenario here.

## How to read this note

**Labels.**
- **verified**: checked in the code or a log;
- **measured**: computed from the recorded ticks or the leaderboard's own `results.json`;
- **argued**: a reading of the measurements, not isolated by an experiment.

**Configuration.**
- Paddock's openpilot adapter at its defaults: the TCP dual-PID tracks supercombo's plan, with no route desire, no
  native longitudinal planner and no driver-resume.
- The ONNX backend on the CUDA provider; openpilot at the `repos.yaml` pin `555f48c5d`.
- Bench2Drive `7ec25d1`; CARLA 0.9.15 at its default quality; one repetition per route.

**The record.** With `record = "true"`, the agent writes `ticks.jsonl` for every route. One line per 20 Hz tick holds:
- the node's whole reply: plan, lead, lanes, pose, sigma, desire;
- what the agent sent and actuated;
- privileged state the model never sees: actors within 60 m, the lane, route progress, the next light.

The expert is Think2Drive: its drive of the same route from Bench2Drive's public dataset, available for 209 of the
218 scored routes.

**Terms.**
- **standstill**: speed below 0.3 m/s for at least 1 s.
- **ahead**: what was in the ego's lane, in this order: a red light on the route within 30 m; a stationary vehicle,
  prop or pedestrian within 40 m (privileged); a moving vehicle within 15 m; the model's own lead within 40 m; else
  *nothing*.
- **plan reach**: |last waypoint| of the plan the node returned. The wire carries 6 waypoints, so 3 s.
- **primary cause**: the highest-priority pattern of [the catalogue](openpilot_drv_supcombo_bench2drive_lib.py)
  (`PATTERNS`) that holds for the route. `patterns.csv` lists all of them, with evidence.

**The runs.** All in `results/bench2drive/openpilot/` and `{runs}/bench2drive/openpilot/`; the attempt that counts
per route is the latest that scored.

| runs | routes counted | what |
|---|---:|---|
| `20261006T195000-1799d7`, `…195021-f4133b`, `…195041-0f2ae7`, `…195101-62634b`, `…195121-a31290` | 202 | the five original shards, resumed in place after reboots |
| `20261007T180507-0b0b57` … `20261007T205535-69051e` (9 runs) | 16 | the rebalanced remainder and the routes held back for crashing CARLA |

## 1. The standstill trap (93 routes, 53 % of lost points)

**What happens** (measured).
- Every Bench2Drive route starts from standstill.
- **On 44 routes the car never exceeded 1 m/s.** Their plans in the first 3 s reached a median 0.6 m; on routes that
  did pull away, 8.9 m. The car that will not launch has already decided in its first plans.
- **After the start, the car stopped 126 times; 96 of those stops never ended**: the route timed out around a
  standing car.
- Standing in a stop that never ended, the plan reached a median **0.14 m**. In stops it left, it reached 1.62 m.

**What was in the lane when it stopped for good**, for the 93 routes:

| Primary cause | Routes |
|---|---:|
| never pulled away | 44 |
| stopped with a clear road and never restarted | 41 |
| still at a red light when the route ended | 4 |
| stopped behind traffic, never restarted after it left | 3 |
| stopped at a red light, never restarted after green | 1 |

"Clear road" is privileged state, not a guess: no vehicle, prop or pedestrian in its lane within 40 m, and no red
light. The 40 m is a horizon, not a gap (measured). On 10 of the 41 routes, a stationary vehicle or prop stands in
the lane 43–60 m ahead. Their frames differ (verified, four of the ten):
- 2847: an empty junction on a green light;
- 2709 and 28154: stopped traffic;
- 2539: the scenario's accident, in fog.

So the 10 stay counted here. The check, and the signal that would separate them, are in
[the design note](../../../../docs/design/driving-failure-analysis.md), §3.3.

**What releases it** (measured).
- Motion in view helps a little, but is not enough. Something moved ahead in the last 2 s of 101 later stops; 28
  of those ended. With nothing moving, 2 of 25 ended.
- At launch, something was moving ahead in the first 3 s on 212 of 218 routes, including 41 of the 44 that never
  launched.
- The stops it did leave were mostly red lights (13 of 30), after a median 9.5 s.

**Why** (argued).
- supercombo has no speed input. Its non-image inputs are `desire_pulse`, `traffic_convention` and `action_t`
  (verified, `model.yaml`), so ego motion is read from frame-to-frame change.
- A stationary camera sees a static world, and the model plans to stay in it. In openpilot's runtime a driver
  answers this by tapping resume or the gas (as the adapter's own comments on its driver-resume emulation note).
- Paddock's adapter has an emulation of that, off by default (`PADDOCK_OPENPILOT_NATIVE_LONG=1`,
  `PADDOCK_OPENPILOT_RESUME_S`). A run with it on is the experiment that would isolate this cause.

**Where it hits** (measured).
- Never-pulled-away routes are concentrated in scenarios that start beside or behind something: ParkingExit 5 of 5,
  AccidentTwoWays 4, ConstructionObstacleTwoWays 3, ParkingCrossingPedestrian 3.
- Examples on the dashboard's route pages: 1833 and 1852 (never pulled away, DS 0); 2084 and 2086 (stopped with a
  clear lane at ~5 s and stood 194 s).

## 2. Route following (50 routes, 19 %)

**What happens** (measured).
- On scored routes the car reached 53 left/right junction manoeuvres. It **completed 5 (all right turns) and 0 of 31
  left turns**.
- 41 routes ended in a deviation at the turn: the car went straight on.
- 9 more left the route away from any junction, at a lane change or exit: MergerIntoSlowTraffic(V2), HighwayExit,
  InterurbanActorFlow.

**Why** (verified for the input, argued for the rest).
- At the adapter's default `PADDOCK_OPENPILOT_DESIRE=off`, the route command never reaches the model.
- The model's own `desire_state` output inside the turns averaged 0.01 for left and 0.09 for right. It does not plan
  a turn the lane does not lead into. The 5 right turns it took were lane-following through a right-turn lane (route
  2050).
- `PADDOCK_OPENPILOT_DESIRE=turn` would send the turn. On KITScenes that input made the model turn more, but later
  (see the KITScenes four-questions note), so it is a measurement, not a fix.

## 3. Obstacles it does not pass (31 routes, 15 %)

**What happens** (measured).
- It stops behind stationary obstacles: a stopped car or accident 22 times, construction cones or a parked mesh 8
  times. It stays there until the scenario timeout or the 200 s tick cap.
- Lane changes happened on only 6 of the 45 Overtaking routes, 13 in all, and **5 of those 6 routes had a
  collision**. One, 25318 (ParkedObstacle), passed its obstacle by changing lanes and completed, with a collision.
  The lane changes it makes look like swerves into contact, not overtakes.
- Overtaking scores **DS 20.6**, its lowest ability. The two clean overtaking successes (1773, 1790) changed lanes 0
  times: the obstacle left enough of the lane to squeeze past.

**Why** (argued). openpilot's planner keeps its lane; lane changes come only from a driver's blinker (desire), which
Bench2Drive does not provide. These scenarios switch the leaderboard's blocked test off while the ego "waits for a
gap", so a car that never pulls out keeps the completion it had at the obstacle.

This is not a perception failure (measured). On 28 of the 31 routes, the model's lead head reported the obstacle
(lead probability > 0.5) when the car stopped. But the car stopped a **median 31 m short of it** (range 0.7–38 m) and
never crept closer, so even this family carries the standstill behaviour of §1.

21 of the 30 cars blocked by a stationary car or cones stopped more than 15 m short. Each had first driven 10–90 m
toward the obstacle, then stood a median 172 s. This is the `stopped_short_of_obstacle` label in `patterns.csv`. A
driver would have rolled up to the obstacle before deciding how to pass it.

## 4. Traffic lights

**What happens** (measured).
- It saw red on the approach to 52 lights and stopped at 40 of them.
- The leaderboard recorded **22 red-light infractions; in 13 the car had first stopped for that red**, then crossed
  while it was still red.
- Restarts after green are bimodal. Of the 24 red-light stops during which the light turned green, half moved within
  0.9 s, and a quarter stood 38 s or more.

**Why** (argued). Going on red after stopping is the release mechanism of §1 working in the wrong direction: what
starts the car is change in the scene, such as cross traffic clearing, not the light. The model has no input that
says "the light is red". Its longitudinal plan is learned from human drivers, who mostly stop for reds.

## 5. Collisions

**What happens** (measured).
- 38 collisions on 27 routes: 22 with vehicles (mean 7.6 m/s), 14 with static objects (guardrails 5, buildings 3),
  2 with pedestrians.
- Before vehicle collisions the model's lead probability averaged 0.68 in the 2 s before, and in 21 of 38 it was
  already braking hard. Static collisions happen at a mean 8.6 m/s, 5 of the 14 in junctions. The lead head
  averaged 0.48 before them; a guardrail or wall is not a lead, so nothing in its outputs warns of one.
- 16 routes ended with a collision followed by a standstill; their completion averaged 44 %.

## 6. What is *not* the problem

- **Speed while moving** (measured). 6.43 vs 6.44 m/s against the expert on the 165 routes both drove. Its maximum
  exceeds the 30 km/h posted limit on 104 routes; the expert's median maximum, 10.0 m/s, is above it too. The speed
  limit is not a model input and carries no penalty.
- **Inference** (measured). A mean 143 ms per `/infer` round trip, and every route ran at the simulator's fixed step,
  so latency does not change what the model sees.
- **Night and fog** (measured, read as argued). Night routes score 36.1 against 31.3 for day, and fog 29.1 against
  35.1 for clear. The model's ego lane-line probability is low everywhere (0.32–0.47, the mean of its two ego lines),
  which says CARLA's lane markings read as weak lines to it in any weather. Rain scores higher (37.7 vs 28.9) and
  launches more often (88 % vs 72 %). A plausible mechanism is that rain and reflections add frame-to-frame change
  for a model that infers motion from it. Weather is assigned per route and is confounded with town and scenario,
  so this is a correlation only.

## 7. Routes that did not score

Two routes crash CARLA itself (`Signal 11`) on every attempt:
- **11715** (EnterActorFlow) at 53.65 s, three times here. Paddock's Bench2Drive notes record nine more attempts on
  three setups.
- **23695** (InterurbanAdvancedActorFlow) at ~36.5 s, three times.

Both are ActorFlow scenarios. In each crash the ego was standing at the merge, and a fresh flow vehicle had spawned
in the second before. That is a correlation, not a diagnosis. Two routes in the same family scored after retries:
24071 on its 4th attempt, 23708 on its 5th. Their scores are conditional on not hitting the crash.

## What would test each cause

| Cause | Run | Expected if the reading is right |
|---|---|---|
| standstill trap | `PADDOCK_OPENPILOT_NATIVE_LONG=1`, `PADDOCK_OPENPILOT_RESUME_S=3` (driver-resume emulation) | most of the 93 standstill routes move; DS gains up to ~half the lost points |
| route following | `PADDOCK_OPENPILOT_DESIRE=turn` | turns taken at more of the 53 junctions, but late; deviations fall |
| overtaking | none in the adapter: openpilot has no lane-change-for-obstacle behaviour | the 31 stay blocked in any configuration; a planner outside the model would be needed |
| red-light restarts | the driver-resume run, read per light | the 13 go-on-red cases are what an assisted start does at a red, so they bound the risk |

## Reproducing

```bash
python -m tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_lib \
    --runs {runs}/bench2drive/openpilot --expert <expert dir> --routes-xml <Bench2Drive>/leaderboard/data/bench2drive220.xml \
    --out <analysis dir>
python -m tools.models.openpilot.analysis.openpilot_drv_supcombo_bench2drive_report --analysis <analysis dir> \
    --routes-xml <Bench2Drive>/leaderboard/data/bench2drive220.xml
```

The analysis dir then holds `route_features.csv`, `patterns.csv`, `pattern_catalogue.csv`, `standstills.csv`,
`lights.csv`, `turns.csv`, `collisions.csv`, and `REPORT.md` with its figures.
