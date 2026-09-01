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

## Reproducing the recorded August runs

| model | scenario | recorded | paddock, 2026-09-01 | |
|---|---|---:|---:|---|
| openpilot | stationary | 5.0 | 5.0 | match |
| drivejepa | stationary | 5.0 | 5.0 | match |
| drivejepa | frontal | 2.0723314376255617 | 2.0723314376255617 | **exact** |
| visionpilot | stationary | 3.145 (27 Aug) / 3.388 (29 Aug) | 2.827 | **differs — explained below** |

drivejepa reproduces to full float precision, which is the strongest evidence
available that the pipeline as a whole — renderer, engine, node, scoring — is
behaving as it did in August.

### Why visionpilot differs, and why it is not a drift

It initially looked like environmental drift: paddock returned 2.827 where
August recorded 3.145, and the *original* `visionpilot_server.py` invoked the
original way returned 2.827 too. Two candidate explanations survived that —
either something beneath both had changed, or the code itself had.

File timestamps settle it:

```
drivejepa_server.py    modified 2026-08-27 03:25   last recorded run 05:37   unchanged since
visionpilot_server.py  modified 2026-08-29 05:02   last recorded run 03:18   edited 1h44m after
```

The VisionPilot node was revised after the last run that scored it. **The
current code has never been scored**, and 2.827 is simply what the newer
implementation does — not a regression, and nothing environmental. drivejepa,
untouched since its run, reproduces exactly, which rules out the renderer, the
engine, the GPU and the libraries beneath them.

Applied to all four nodes, the same comparison predicts which reproduce:

| model | node modified | last scored run | prediction | outcome |
|---|---|---|---|---|
| openpilot | 08-27 03:13 | 08-27 23:28 | unchanged, should match | matched |
| drivejepa | 08-27 03:25 | 08-28 00:00 | unchanged, should match | matched exactly |
| alpamayo | 08-27 04:12 | 08-27 07:37 | unchanged, should match | **stationary matched; frontal did not — see below** |
| visionpilot | **08-29 05:02** | 08-29 03:18 | **newer, will differ** | differed |

The prediction held for the deterministic models. It does not apply to alpamayo,
for the reason in the next section — the assumption behind it is that a run is
reproducible at all.

This is the failure mode version pinning exists to prevent, and it is precisely
why this code is now in git: from here, a score change has a commit beside it.
The August visionpilot figures should be treated as belonging to a superseded
implementation, and re-measured rather than compared against.

**If a score moves unexpectedly, check this first.** Compare the adapter's git
history against the run's `started_utc` before suspecting the renderer, the GPU,
or the benchmark.


## Single-run scores are not comparable for stochastic models

Alpamayo returned 5.0 on frontal-0103 where August recorded 0.0 — the full width
of the scale. Its node was not edited, so the timestamp check predicted a match.
The prediction was wrong because the premise was: **alpamayo does not produce the
same trajectory twice.**

It samples with `temperature=1.0`, `top_p=0.95` and no seed. The five recorded
seeds for that one scenario:

| model | run_0 | run_1 | run_2 | run_3 | run_4 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| alpamayo | 0.000 | 2.084 | 1.519 | 1.885 | 5.000 | 2.098 | **1.817** |
| openpilot | 1.862 | 2.610 | 2.187 | 2.004 | 2.389 | 2.210 | 0.298 |

Today's 5.0 is exactly run_4; August's 0.0 is exactly run_0. Both are single draws
from the same distribution, and neither is evidence about the model.

Consequences for anyone using this benchmark:

* **A single run is meaningless for alpamayo.** At sd 1.817, distinguishing it
  from openpilot's 2.210 to within ±0.25 would take roughly 200 runs.
* **Their reported means are statistically indistinguishable** on this scenario
  (2.098 against 2.210) despite looking a scale apart on any single run.
* Deterministic models (openpilot, drivejepa, visionpilot) reproduce exactly, so
  `--runs 1` is adequate for them and wasteful to exceed.

The fix is not to seed alpamayo into false determinism — the sampling is part of
the model — but to report a mean and spread over enough runs, and to say n.
Paddock stores per-episode records, so the spread is available rather than
collapsed into one number.

## Open: the two openpilot paths disagree on traffic convention

openpilot sets `traffic_convention[int(is_rhd)] = 1`, where `is_rhd` means
right-hand *drive* — wheel on the right, i.e. left-hand traffic.

| path | value | implies |
|---|---|---|
| open-loop KITScenes harness | `[1, 0]` | left-hand-drive / right-hand traffic |
| closed-loop node | `[0, 1]` | right-hand-drive / left-hand traffic |

They cannot both be right. KITScenes is Germany, and the neuroNCAP scenes used so
far (0103, 0108) are `boston-seaport` — both left-hand-drive, right-hand traffic.
`[1, 0]` looks correct for both, making the closed-loop value a bug that has been
telling the model it is driving a right-hand-drive car in Boston.

**Not changed.** Correcting it changes every recorded closed-loop openpilot score,
and that re-measurement deserves its own decision and its own commit rather than
arriving as a side effect of a migration. Flagged at the constant in
`adapters/models/openpilot/policy.py`.
