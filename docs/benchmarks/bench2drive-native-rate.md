# Bench2Drive at each model's native inference rate

The published 50-route campaign drives every model at **20 Hz**: one inference
per simulator tick. That is a property of the protocol, not of the models. Three
of them were built to replan far more slowly, and driving them faster does not
give them more information — it asks them for a plan at a cadence they were not
built to produce.

This page is the re-run that measures the difference. 18-20 Sep 2026, on a
RunPod RTX PRO 6000, CARLA 0.9.15 at default Epic quality, the Bench2Drive
checkout pinned in [`repos.yaml`](../../repos.yaml), and the **same 50 routes**
as the 20 Hz campaign.

## What was varied, and what was held

| | 20 Hz campaign | Native-rate campaign |
|---|---|---|
| Inference rate | 20 Hz for every model | `PADDOCK_INFER_HZ` = the model's own rate |
| DriveJEPA | 20 Hz | **2 Hz** |
| QwenDrive | 20 Hz | **2 Hz** |
| Alpamayo 1.5 (nav) | 20 Hz | **10 Hz** |
| Routes | `bench2drive220.xml`, the 50-route subset | identical, same subset |
| `waypoint_dt` | 0.5 s | 0.5 s |
| Tracker / controller | shared dual-PID tracker | identical |
| Simulator | CARLA 0.9.15, Epic | CARLA 0.9.15, Epic |

`PADDOCK_INFER_HZ` is the only knob turned **deliberately**, and it is not the
only thing that differs. Diffing `config.model_env` of the two records shows the
rest: `MKL/OMP_NUM_THREADS` went 2 → 4 for DriveJEPA and QwenDrive and was left
unset for Alpamayo, the 20 Hz QwenDrive run wrote a `QWEN_DRIVE_TRACE` sink this
one did not, the 20 Hz Alpamayo run set `ALPAMAYO_NAV=1` and
`ALPAMAYO_RESERVE_GB=0` where this campaign set neither, and this campaign records
`ALPAMAYO_CHECKPOINT` and `ALPAMAYO_COSMOS` — the same two weights, named in the
record where the 20 Hz run named them nowhere. None of it should move a
trajectory — thread counts are throughput, the trace sink is write-only, the
checkpoints are the same model, and `ALPAMAYO_NAV` defaults to 1 at
[`policy.py:239`](../../src/paddock/adapters/models/alpamayo15/policy.py), so both
Alpamayo campaigns drove with the nav instruction on — but that is an argument,
not a reading, and the table at the end of this page books it as one.

The rate is recorded in `config.model_env` of each tracked run, and that is the
only place it is recorded: the harness did not write it into
the source runs' `run_config.json`, so it reaches the record through the ingest
config and each `run.json` note now says so. It is still checkable from the
recordings without trusting that note — each route's `log.csv` carries the planned
waypoints per frame, and counting runs of consecutive frames holding an identical
`wp0x…wp5y` tuple recovers the decimation directly: 10-frame holds against a
20 Hz sim for the 2 Hz models, 2-frame holds for Alpamayo at 10 Hz, which never
once holds a plan for five frames.

**OpenPilot and VisionPilot were not re-run.** OpenPilot's native rate *is* 20 Hz,
so its runs are already native-rate runs; VisionPilot was left alone.
Nothing on this page says anything about either. (OpenPilot's 31 Oct 50-route run,
`pooled50-40c29d24c37d542a-c7d2b2`, which the support matrix ranks, is one 20 Hz
configuration and is not part of this comparison; see [bench2drive.md](bench2drive.md).)

### Two things this page does not get to assume

**The rate is recorded, not derived.** Which rate is "native" for a model was the
operator's call per run, and `model.yaml` does not yet state it in a form the
harness could read back. What the capability blocks say today:

| Model | Rate driven | `model.yaml` says | |
|---|---:|---|---|
| Alpamayo 1.5 | 10 Hz | `native_input_hz: 10.0`, `waypoint_dt: 0.5`; since Nov 2026 `planning_hz: 10.0` | matches |
| QwenDrive | 2 Hz | `native_input_hz: 10.0`, `waypoint_dt: 0.5` | `native_input_hz` describes its 10 Hz *ego-pose history*; its camera frames are picked 0.5 s apart, which is the 2 Hz cadence it was driven at |
| DriveJEPA | 2 Hz | no `native_input_hz`; `waypoint_dt: 0.5` | 2 Hz is its planning cadence, not a declared field |

So `PADDOCK_INFER_HZ` in `config.model_env` is the authoritative statement of what
each run did. Making the rate derivable from `model.yaml` — one field, checked
against the run — is follow-up work, not something this campaign settles.

**The harness support is on `main` since 2026-10-22.** A model declares
`planning_hz` in `model.yaml`; the Bench2Drive client asks the node once per
1/`planning_hz` seconds and follows the last plan unchanged in between, which is
what this campaign did (the 10-frame holds above). The rate is recorded as
`config.infer_hz`, and the host's `PADDOCK_INFER_HZ` still overrides it.
Drive-JEPA declares 2 Hz and Alpamayo 1.5 10 Hz (since Nov 2026); QwenDrive declares
none yet. The runs on this page were driven before that, on another branch, and were
ingested.

## Tracked runs

| Model | Native rate | Tracked run | 20 Hz run |
|---|---:|---|---|
| DriveJEPA | 2 Hz | [`native50-2hz-…447e23`](../../results/bench2drive/drivejepa/native50-2hz-40c29d24c37d542a-447e23/run.json) | `pooled50-40c29d24c37d542a-1de367` |
| Alpamayo 1.5 (nav) | 10 Hz | [`native50-10hz-…c4d076`](../../results/bench2drive/alpamayo15/native50-10hz-084418258d833a9f-c4d076/run.json) | `pooled50-40c29d24c37d542a-36d2d0` |
| QwenDrive | 2 Hz | [`native50-2hz-…9b2bc9`](../../results/bench2drive/qwendrive/native50-2hz-084418258d833a9f-9b2bc9/run.json) | `pooled50-40c29d24c37d542a-f250e0` |

`paddock ingest` writes all three `valid: true`. They rank in `docs/leaderboard.md` and in the support matrix at the native rate. The comparison they exist for is this page, where both
columns are stated.

All three were written by `paddock ingest` from the per-route `results.json`
trees, not typed here; every source run id is in the record (`config.source_runs`), and
the per-route file digests are in the JSON beside this page rather than in the
record — `store.write_run` keeps `samples.jsonl` outside the repo. The
per-route detail the ingest does not keep — status and the infraction breakdown —
is in [`bench2drive-native-rate.json`](bench2drive-native-rate.json), one record
per route per model, carrying the sha256 of the `results.json` it came from.

## Results

Every comparison below is over the routes **scored at both rates**: 50 for
DriveJEPA, 49 for Alpamayo 1.5 and QwenDrive (see [route 3800](#route-3800-is-unscorable-for-two-of-the-three-models)).
The 20 Hz column is therefore not always the published all-50 mean — for
Alpamayo 1.5 and QwenDrive it is the same recordings restricted to 49 routes.

"Routes completed" counts routes the evaluator ended with status `Completed`.
Infractions are **recorded counts, not rates**; the totals exclude
`min_speed_infractions`, which is a per-checkpoint speed-ratio log rather than a
violation (see [minimum-speed normalization](bench2drive-speed-normalization.md)),
and that count is listed separately at the foot of each table.

### DriveJEPA — 20 Hz → 2 Hz, 50/50 routes

| | 20 Hz | 2 Hz (native) | Δ |
|---|---:|---:|---:|
| Driving score ↑ | 31.71 | **57.60** | +25.89 |
| Route completion ↑ (%) | 39.10 | **70.97** | +31.87 |
| Infraction penalty ↑ | 0.862 | 0.848 | −0.014 |
| Routes completed ↑ | 5 | **23** | +18 |
| Total infractions ↓ | 43 | 39 | −4 |
| — Pedestrian collision ↓ | 2 | 4 | +2 |
| — Vehicle collision ↓ | 3 | 6 | +3 |
| — Layout collision ↓ | 3 | 2 | −1 |
| — Red light ↓ | 1 | 4 | +3 |
| — Stop sign ↓ | 1 | 2 | +1 |
| — Outside lanes ↓ | 7 | 3 | −4 |
| — Route deviation ↓ | 6 | 9 | +3 |
| — Blocked / stalled ↓ | 17 | **7** | −10 |
| — Emergency yield ↓ | 3 | 2 | −1 |
| Min-speed entries (not a violation) | 338 | 653 | |

### Alpamayo 1.5 (nav) — 20 Hz → 10 Hz, 49/50 routes

| | 20 Hz | 10 Hz (native) | Δ |
|---|---:|---:|---:|
| Driving score ↑ | 43.20 | **55.82** | +12.62 |
| Route completion ↑ (%) | 73.07 | **76.52** | +3.45 |
| Infraction penalty ↑ | 0.648 | **0.725** | +0.077 |
| Routes completed ↑ | 24 | **27** | +3 |
| Total infractions ↓ | 81 | **73** | −8 |
| — Pedestrian collision ↓ | 3 | 2 | −1 |
| — Vehicle collision ↓ | 42 | **23** | −19 |
| — Layout collision ↓ | 1 | 2 | +1 |
| — Stop sign ↓ | 0 | 1 | +1 |
| — Outside lanes ↓ | 22 | 28 | +6 |
| — Route deviation ↓ | 3 | 9 | +6 |
| — Blocked / stalled ↓ | 9 | 7 | −2 |
| — Emergency yield ↓ | 1 | 1 | 0 |
| Min-speed entries (not a violation) | 654 | 694 | |

### QwenDrive — 20 Hz → 2 Hz, 49/50 routes

| | 20 Hz | 2 Hz (native) | Δ |
|---|---:|---:|---:|
| Driving score ↑ | 12.94 | **17.08** | +4.14 |
| Route completion ↑ (%) | 50.64 | **64.30** | +13.66 |
| Infraction penalty ↑ | 0.297 | 0.291 | −0.006 |
| Routes completed ↑ | 6 | **14** | +8 |
| Total infractions ↓ | 180 | 165 | −15 |
| — Pedestrian collision ↓ | 0 | 3 | +3 |
| — Vehicle collision ↓ | 50 | 60 | +10 |
| — Layout collision ↓ | 59 | **42** | −17 |
| — Red light ↓ | 1 | 3 | +2 |
| — Stop sign ↓ | 0 | 2 | +2 |
| — Outside lanes ↓ | 45 | 39 | −6 |
| — Route deviation ↓ | 9 | 5 | −4 |
| — Blocked / stalled ↓ | 16 | 10 | −6 |
| — Emergency yield ↓ | 0 | 1 | +1 |
| Min-speed entries (not a violation) | 432 | 560 | |

## Reading

**All three score higher at their native rate, so the 20 Hz protocol understates
them.** That is the finding, and it is a measurement: the driving score is the
evaluator's own, over the same routes, with one environment variable changed.

Past that, the three move for different reasons, and the reasons are argued from
the breakdown rather than measured directly:

- **DriveJEPA and QwenDrive gain by driving further instead of stalling.**
  DriveJEPA's blocked/stalled count falls 17 → 7 and its completed routes go
  5 → 23; QwenDrive's blocked falls 16 → 10 and its completed routes 6 → 14. In
  both cases the car that used to stop now keeps going, which is also why their
  **collision exposure rises**: DriveJEPA's vehicle collisions 3 → 6 and
  pedestrian 2 → 4, QwenDrive's vehicle 50 → 60 and pedestrian 0 → 3. Their
  infraction penalty is flat to slightly worse (0.862 → 0.848, 0.297 → 0.291)
  while route completion nearly doubles. The score goes up because the route
  term goes up, not because the driving got safer.
- **Alpamayo 1.5 is the only one that gets safer.** Its vehicle collisions
  nearly halve, 42 → 23, and its infraction penalty rises 0.648 → 0.725 — the
  only penalty improvement in the set — on route completion that barely moves
  (73.07 → 76.52). This is the one model whose gain is not mostly extra
  distance.

### The caveat that applies to all three

**A higher driving score here comes partly from more exposure, not only from
better driving.** Bench2Drive's composite is route completion multiplied by an
infraction penalty, so a model that stalls early is scored on a short route it
never had the chance to fail on. Moving from 39% to 71% route completion means
roughly twice the opportunity to collide — and DriveJEPA and QwenDrive duly
collide more in absolute terms. The counts in these tables are **raw counts over
unequal exposure** and must not be compared across the two columns as if they
were rates. The README publishes per-progress-km collision rates for the 20 Hz
campaign for exactly this reason. It is computable for these runs too — route
length is a property of the route, not of the run, and
[`bench2drive-speed-normalization.json`](bench2drive-speed-normalization.json)
carries `route_length_m` for all 50 — so here it is, joined on `route_id` over
the routes scored at both rates, progress km being route length × route
completion:

| Model | Progress km 20 Hz | Progress km native | Collisions 20 Hz | Collisions native | **Per progress-km 20 Hz → native** |
|---|---:|---:|---:|---:|---:|
| DriveJEPA (50) | 2.14 | 3.93 | 8 | 12 | 3.74 → **3.05** |
| Alpamayo 1.5 (49) | 3.91 | 4.15 | 46 | 27 | 11.76 → **6.50** |
| QwenDrive (49) | 2.63 | 3.36 | 109 | 105 | 41.48 → **31.25** |

Collisions are the three collision keys summed. **Normalized for exposure, all
three collide less often at their native rate** — including DriveJEPA, whose raw
count rises 8 → 12 while its rate falls, because it drove 1.8× as far. That is
the comparison the raw counts above cannot support, and it is stronger than the
headline: the two models that gain mostly by driving further are not buying that
distance with a higher collision rate.

## Route 3800 is unscorable for two of the three models

Route **3800 (Town13, MergerIntoSlowTraffic)** never scored for `alpamayo15` or
`qwendrive` at their native rates. Six attempts across the two models, including
runs alone on an idle 96 GB card, produced no scoring checkpoint. **What is
recorded differs by model, and the two halves are not equally strong:**

- **QwenDrive — recorded.** `Signal=11`, `CommonUnixCrashHandler: Signal=11` and
  `Segmentation fault (core dumped)` are in the evaluator log of attempt
  `20260918T091245-f19361`, at roughly 1300 frames, after which the leaderboard's
  own signal handler raises `The simulation took longer than 600.0s to update`.
- **Alpamayo 1.5 — inferred.** Neither attempt wrote an evaluator log — the tree
  holds 49 of them, one per scored route, and none for 3800 — so there is no
  CARLA crash output for this model at all. What there is: two node logs that stop
  mid-inference, `20260919T004052-2ce451` after `infer #601` and
  `20260919T062659-2ce451` after `infer #701`, which at 10 Hz against a 20 Hz sim
  is about 1200 and 1400 frames. That is where the 1300-1411 range comes from. The
  `UNSCORABLE.txt` markers in both trees were written by hand a day later and
  restate the conclusion, so they are not evidence for it.

**It is a model/route pairing, not a broken route:** DriveJEPA drives the same
route at 2 Hz in the same campaign and scores **100** (source run
`20260918T050144-892d9d`). That distinguishes it from route 11715, which crashes
CARLA for every model and is excluded from the route set entirely.
Since 9 Nov 2026 3800 is among the routes set aside for every model as well: it
crashed CARLA on all three attempts by Alpamayo 1.5 at 10 Hz in the 220-route runs
([Routes set aside](bench2drive.md#routes-set-aside)). The records on this page were
pooled before that, and keep it as a route lost.

**But it is not independent of the variable under test.** Both models drove 3800
to a score at 20 Hz — `alpamayo15` **35.33**, `qwendrive` **12.18**, in the
tracked `pooled50` records — and it became unscorable only at the native rate. So
dropping it removes the one route where the native rate did worst, and the
49-route denominator flatters this page. The effect is bounded: scoring 3800 as a
**0** natively, which is the worst it could have been, still leaves Alpamayo 1.5
at 54.70 against 43.04 and QwenDrive at 16.73 against 12.92 over all 50. Both are
still up, so the direction survives the objection even at its strongest.

How it is recorded — this is the part that matters for the leaderboard:

- Both runs list 3800 in `config.routes_subset`: it was **requested**.
- Both carry `dataset.n_failed: 1`, `dataset.n_expected: 50`,
  `dataset.n_samples: 49` and `status: "partial"`. Both are `valid: true`
  (`ranked = true` in their ingest entries).
- Both name it in `notes` — "routes expected but never scored: 3800". The
  QwenDrive record additionally carries
  `config.routes_never_scored: {"3800": "…no records in _checkpoint (the route
  did not score)"}`, because its one attempt did write a `results.json` with an
  empty `_checkpoint`; the Alpamayo attempts wrote no `results.json` at all, so
  its note reads "no attempt found" and the evidence lives in the
  `UNSCORABLE.txt` marker beside the empty route directory and in this page.
- **It is never scored as a zero.** A zero from a segfault is indistinguishable
  from a model that drove into a wall and would drag the mean down by about one
  point per model; the means above are over 49 routes, with the denominator
  stated.

## Verified, and argued

Following [docs/reviewing.md](../reviewing.md), kept apart:

| Claim | How | Strength |
|---|---|---|
| The driving scores, completions, penalties and infraction counts above | evaluator output, ingested by `paddock ingest`; every table cell recomputed from the tracked records by `tests/test_bench2drive_native_rate_docs.py` | **verified** |
| The native-rate and 20 Hz records describe the same 50 routes | same `routes_file`, same `routes_subset`, same `waypoint_dt` in both `run.json`s | **verified** |
| The rate is the only difference that could move a trajectory | `model_env` differs in more than the rate: `MKL/OMP_NUM_THREADS` (2 → 4 for DriveJEPA and QwenDrive, unset for Alpamayo), `QWEN_DRIVE_TRACE` at 20 Hz only, `ALPAMAYO_NAV=1` and `ALPAMAYO_RESERVE_GB=0` at 20 Hz only, `ALPAMAYO_CHECKPOINT` and `ALPAMAYO_COSMOS` recorded here only. Thread counts are throughput, the trace sink is write-only, the checkpoints are the same weights, and `ALPAMAYO_NAV` defaults to 1, so both Alpamayo campaigns drove the nav instruction on. `tests/test_bench2drive_native_rate_docs.py` pins this exact set | **argued.** Each difference is argued immaterial; none was measured to be |
| The rate each run was driven at | not recorded by the harness: it reaches `config.model_env` through the ingest config, and each `run.json` note says so | **argued** from the operator's config — but *checkable*: plan-hold lengths in the per-route `log.csv` recover 10-frame decimation for the 2 Hz models and 2-frame for Alpamayo |
| The simulator build (CARLA 0.9.15, Epic) and the shared tracker and controller were the same in both campaigns | nothing records them — not `run_config.json`, not the node logs, not the evaluator logs | **argued — operator testimony.** Believed, and unfalsifiable from the recordings |
| Route 3800 segfaults CARLA for QwenDrive | `Signal=11`, `CommonUnixCrashHandler` and `Segmentation fault` in the evaluator log of `20260918T091245-f19361` | **verified** |
| Route 3800 segfaults CARLA for Alpamayo 1.5 | no evaluator log and no CARLA crash output exists for this model on this route; two node logs stop mid-inference at `infer #601` and `infer #701` | **argued.** The node logs show the run dying where a segfault would put it; the `UNSCORABLE.txt` markers restate the claim and are not evidence for it |
| Dropping route 3800 does not manufacture the result | both models scored it at 20 Hz (35.33, 12.18), so its loss is not independent of the rate; scoring it 0 natively still leaves Alpamayo 54.70 vs 43.04 and QwenDrive 16.73 vs 12.92 | **verified** — the bound is computed from the tracked records |
| Route 3800 is a model/route pairing, not a broken route | DriveJEPA scores 100 on it in the same campaign | **strong** — one model completing it rules out the route alone, but not every interaction |
| DriveJEPA and QwenDrive gain by driving further rather than driving better | blocked counts fall, completion rises, penalty flat-to-worse, collisions rise | **argued** from the breakdown; no per-tick analysis was done |
| Alpamayo 1.5 gains by driving better | penalty rises, vehicle collisions halve, completion nearly flat | **argued**, same basis |
| 20 Hz is the wrong protocol for these models generally | — | **not claimed.** One campaign, one route set, one simulator build, one rate each. No rate sweep was run, so nothing here locates an optimum or shows the relation is monotonic |
| The same holds for OpenPilot and VisionPilot | — | **not measured.** Neither was re-run |
| The rate driven is each model's declared native rate | — | **not derivable here.** `model.yaml` declares `native_input_hz` for two of the three and it disagrees with the rate driven for QwenDrive; the rate is recorded per run in `config.model_env`, not read from the registry (see above) |
| Another site could re-drive these runs | — | **unverified**: the rate is read on `main` now (`planning_hz`, or `PADDOCK_INFER_HZ`), but CARLA re-runs are not bit-reproducible. The *ingest* reproduces on `main` from the trees |

## Where the recordings live

Outside the repo, and they are not committed:

- Per-route Bench2Drive results trees, one run id per route:
  `{runs}/bench2drive-diagnostics/native-50/<model>/<run_id>/<route>-<Town>-<Scenario>/`
  — on the NAS at `paddock-runs/bench2drive-diagnostics/native-50`.
- Per-route front and BEV clips: `Bench2drive/Simulation/native50` on the NAS.

No video or frame from this campaign is tracked; `tests/test_bench2drive_docs.py`
gates that for the repo as a whole. What is tracked is the small stuff: the two
`results/` records per model and the per-route JSON beside this page.

## Reproducing the ingest

The `[[ingest.bench2drive]]` blocks are in
[`paddock.toml.example`](../../paddock.toml.example). Point `runs` at a copy of
the trees, then:

```bash
paddock ingest --all
```

The run ids above fall out of the config; they are not chosen. Never hand-edit a
`run.json` or `metrics.json` — regenerate.

[50-route tables](bench2drive.md#bench2drive--50-routes) ·
[Bench2Drive protocol](bench2drive.md) ·
[Per-route records](bench2drive-native-rate.json)
