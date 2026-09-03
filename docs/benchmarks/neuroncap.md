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

## Resolved: the two openpilot paths disagreed on traffic convention

openpilot sets `traffic_convention[int(is_rhd)] = 1`, where `is_rhd` means
right-hand *drive* — wheel on the right, i.e. left-hand traffic.

| path | value before | implies |
|---|---|---|
| open-loop KITScenes harness | `[1, 0]` | left-hand-drive / right-hand traffic |
| closed-loop node | `[0, 1]` | right-hand-drive / left-hand traffic |

They cannot both be right. KITScenes is Germany, and every neuroNCAP scene in the
official set is `boston-seaport` or `singapore-*` — the ones evaluated here are
Boston, so left-hand-drive, right-hand traffic. `[1, 0]` is correct for both, and
the closed-loop value had been telling the model it was driving a right-hand-drive
car in Boston.

**Fixed in `12f3250`**, committed 2026-09-01T00:44:56Z. Both paths now pass
`[1, 0]`, and `tests/test_openpilot_traffic_convention.py` parses the constant out
of both files and fails if they ever disagree again — they are separate constants
in separate files, because the two paths load openpilot differently, which is how
they came to disagree in the first place.

### Every recorded closed-loop openpilot score predates the fix

| run | started | vs fix |
|---|---|---|
| `20260831T235716-88a4e0` | 2026-08-31T23:57:16Z | before |
| `20260901T004005-88a4e0` | 2026-09-01T00:40:05Z | before, by 5 minutes |
| `legacy-*` (four runs) | August, undated | before |

There is no closed-loop openpilot number on record that was produced with the
correct traffic convention. The README's neuroNCAP row for openpilot — mean 3.80,
47% collision rate — is a measurement of the misconfigured model, and the size of
the error is unknown until it is re-run. It is not a small knob: it changes which
side of the road the network believes traffic is on, and the side scenario is
laterally asymmetric.

openpilot's rows are re-measured as part of the full-set sweep below; until that
lands, treat the recorded closed-loop openpilot figures as superseded.

## The evaluation set, and how many runs

### 20 scenario–scene pairs

The official set is transcribed from the vendor's own job arrays
(`scripts/arrays/ncap_slurm_array_*.txt`) into `OFFICIAL_SET` in `harness.py`:

| scenario | scenes | |
|---|---:|---|
| stationary | 10 | 0099, 0101, 0103, 0106, 0108, 0278, 0331, 0783, 0796, 0966 |
| frontal | 5 | 0103, 0106, 0110, 0346, 0923 |
| side | 5 | 0103, 0108, 0110, 0278, 0921 |

20 pairs over 14 distinct scenes. Sequence 0061 appears in the stationary array
carrying `# mini set (do not use for eval)` and is excluded; it is also the one
scene with no renderer checkpoint on this machine.

The files under `scenarios_10hz/` are the upstream scenario files with
`time-step: 0.5` changed to `0.1` and nothing else — verified one line per file
at porting time. `preflight` refuses a sweep with any of the 20 missing, and
`tests/test_neuroncap_harness.py` pins the set both ways: no file missing, and
no file present that is not in it.

Not every scene is in every category — 0099 is stationary-only — so `episodes()`
skips a (scene, scenario) pair with no file when `--scenarios` is defaulted.
When `--scenarios` names a category explicitly, a missing pair raises instead:
asking for something that does not exist should not quietly score less.

### Runs: 10, not 50

Upstream runs 50 (`RUNS=50` in `single_machine_docker_run_eval.sh`). Paddock's
sweeps use 10, which is a subsample of that protocol rather than a different one:
the engine seeds the scenario jitter with the run index (`engine.run(i)` in
`main.py`), so runs 0–9 are exactly the first ten of upstream's fifty, and a
later top-up extends rather than replaces them.

A run is **not** a repeat. Each index jitters the actor's start frame, position,
dimensions and identity, which is why deterministic models still show spread —
openpilot's sd is 0.27 on frontal-0103 and 1.00 on side-0103. Treating `--runs 1`
as sufficient for a deterministic model confuses reproducibility (same seed, same
score — true) with sufficiency (one draw characterises the model — false).

The standard error of a category mean over S scenes and R runs is
`sqrt(σ_between²/S + σ_within²/(S·R))`. Going from 1 scene to 5 attacks the
dominant term; going from 10 runs to 50 attacks a term already five times
smaller. At S=5 with openpilot's measured σ_within of 1.00 on side, the
within-scene contribution is 0.14 at R=10 and 0.06 at R=50 — both well under the
between-scene term. Five times the compute for a few percent of the error.

R=10 is also the point at which σ_between becomes measurable for the first time;
it has never been estimated here, because every recorded run is on one or two
scenes. If it turns out small, topping up runs is worth it. Alpamayo is the one
model where extra runs plausibly pay on their own: its σ_within is 1.82.

### Cost

Measured per-episode wall clock (2026-09-01 runs, `--runs 1`, including node
startup, which amortises across the runs in a sweep so these are upper bounds):

| model | s/episode | 20 pairs × 10 runs |
|---|---:|---:|
| visionpilot | 92 | ~5 h |
| openpilot | 134 | ~7.5 h |
| drivejepa | 191 | ~10.5 h |
| alpamayo15 | 375 | ~21 h |
| **total** | | **~44 h** |

The full 50-run protocol would be ~220 h with `gpu.serialize = true`.

### Per-run records reach `results/`

One subprocess drives N runs, but the store keeps them apart: the harness
overrides `ClosedLoopBenchmark.episode_samples` to emit one sample per run, so a
20-pair sweep at R=10 records `n = 200`, not `n = 20`, and the spread survives
into the only part of a run that is tracked. Runs are ordered by parsed index —
sorted as strings, `run_10` falls between `run_1` and `run_2`.

### Sweep shape

The renderer serves one scene at a time and is slow to start, so the loop is
scene-outer, model-inner — restart the renderer per scene, run all four models
against it, move on. That is 14 renderer starts rather than 56.
