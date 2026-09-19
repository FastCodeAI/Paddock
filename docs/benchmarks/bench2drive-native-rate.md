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

`PADDOCK_INFER_HZ` is the only knob turned. It is recorded in `config.model_env`
of each tracked run, which is what keeps a native-rate run and a 20 Hz run of the
same model from hashing to the same run id — the ingest's run id is
`<label>-<route-set hash>-<config hash>`, and without the rate in the config the
two campaigns differ only by a label.

**OpenPilot and VisionPilot were not re-run.** OpenPilot's native rate *is* 20 Hz,
so the published run is already its native-rate run; VisionPilot was left alone.
Nothing on this page says anything about either.

### Two things this page does not get to assume

**The rate is recorded, not derived.** Which rate is "native" for a model was the
operator's call per run, and `model.yaml` does not yet state it in a form the
harness could read back. What the capability blocks say today:

| Model | Rate driven | `model.yaml` says | |
|---|---:|---|---|
| Alpamayo 1.5 | 10 Hz | `native_input_hz: 10.0`, `waypoint_dt: 0.5` | matches |
| QwenDrive | 2 Hz | `native_input_hz: 10.0`, `waypoint_dt: 0.5` | `native_input_hz` describes its 10 Hz *ego-pose history*; its camera frames are picked 0.5 s apart, which is the 2 Hz cadence it was driven at |
| DriveJEPA | 2 Hz | no `native_input_hz`; `waypoint_dt: 0.5` | 2 Hz is its planning cadence, not a declared field |

So `PADDOCK_INFER_HZ` in `config.model_env` is the authoritative statement of what
each run did. Making the rate derivable from `model.yaml` — one field, checked
against the run — is follow-up work, not something this campaign settles.

**The harness support is not on `main`.** `PADDOCK_INFER_HZ` is read by the
Bench2Drive `agent.py` / `harness.py` on the branch that drove this campaign; it
is not in `src/paddock/adapters/benchmarks/bench2drive` on `main` yet. This PR
lands the **results and their records**, not the knob. Until that branch merges,
`paddock run` on `main` drives at 20 Hz and these runs cannot be re-driven from a
clean checkout — the ingest, which is what produced the tracked records, does run
on `main`.

## Tracked runs

| Model | Native rate | Tracked run | 20 Hz run |
|---|---:|---|---|
| DriveJEPA | 2 Hz | [`native50-2hz-…447e23`](../../results/bench2drive/drivejepa/native50-2hz-40c29d24c37d542a-447e23/run.json) | `pooled50-40c29d24c37d542a-1de367` |
| Alpamayo 1.5 (nav) | 10 Hz | [`native50-10hz-…c4d076`](../../results/bench2drive/alpamayo15/native50-10hz-084418258d833a9f-c4d076/run.json) | `pooled50-40c29d24c37d542a-36d2d0` |
| QwenDrive | 2 Hz | [`native50-2hz-…9b2bc9`](../../results/bench2drive/qwendrive/native50-2hz-084418258d833a9f-9b2bc9/run.json) | `pooled50-40c29d24c37d542a-f250e0` |

All three were written by `paddock ingest` from the per-route `results.json`
trees, not typed here; every source run id and file digest is in the record. The
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
campaign for exactly this reason; the same normalization has **not** been
computed for the native-rate runs, because `route_length_m` per route is not
carried in the ingest's per-route metrics and recomputing it was out of scope
here.

## Route 3800 is unscorable for two of the three models

Route **3800 (Town13, MergerIntoSlowTraffic)** never scored for `alpamayo15` or
`qwendrive`. CARLA segfaults — `Signal=11`, `CommonUnixCrashHandler: Signal=11`,
`Segmentation fault (core dumped)` in the evaluator log — at roughly 1300-1411
frames, after which the leaderboard's own signal handler raises
`The simulation took longer than 600.0s to update`. Six attempts across the two
models, including runs alone on an idle 96 GB card, produced no scoring
checkpoint.

**It is a model/route pairing, not a broken route:** DriveJEPA drives the same
route at 2 Hz in the same campaign and scores **100** (source run
`20260918T050144-892d9d`). That distinguishes it from route 11715, which crashes
CARLA for every model and is excluded from the route set entirely.

How it is recorded — this is the part that matters for the leaderboard:

- Both runs list 3800 in `config.routes_subset`: it was **requested**.
- Both carry `dataset.n_failed: 1`, `dataset.n_expected: 50`,
  `dataset.n_samples: 49`, `status: "partial"` and `valid: false`.
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
| The native-rate and 20 Hz records describe the same routes at different rates | same `routes_file`, same `routes_subset`, same `waypoint_dt` in both `run.json`s; `PADDOCK_INFER_HZ` is the only differing `model_env` key | **verified** |
| Route 3800 segfaults CARLA for these two models | `Signal=11` in the evaluator log of the qwendrive attempt; `UNSCORABLE.txt` markers; six attempts | **verified** |
| Route 3800 is a model/route pairing, not a broken route | DriveJEPA scores 100 on it in the same campaign | **strong** — one model completing it rules out the route alone, but not every interaction |
| DriveJEPA and QwenDrive gain by driving further rather than driving better | blocked counts fall, completion rises, penalty flat-to-worse, collisions rise | **argued** from the breakdown; no per-tick analysis was done |
| Alpamayo 1.5 gains by driving better | penalty rises, vehicle collisions halve, completion nearly flat | **argued**, same basis |
| 20 Hz is the wrong protocol for these models generally | — | **not claimed.** One campaign, one route set, one simulator build, one rate each. No rate sweep was run, so nothing here locates an optimum or shows the relation is monotonic |
| The same holds for OpenPilot and VisionPilot | — | **not measured.** Neither was re-run |
| The rate driven is each model's declared native rate | — | **not derivable here.** `model.yaml` declares `native_input_hz` for two of the three and it disagrees with the rate driven for QwenDrive; the rate is recorded per run in `config.model_env`, not read from the registry (see above) |
| Another site could re-drive these runs | — | **unverified**, and blocked twice over: `PADDOCK_INFER_HZ` is not read by the Bench2Drive adapter on `main`, and CARLA re-runs are not bit-reproducible anyway. The *ingest* reproduces on `main` from the trees |

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

[50-route tables](../../README.md#bench2drive--50-routes) ·
[Bench2Drive protocol](bench2drive.md) ·
[Per-route records](bench2drive-native-rate.json)
