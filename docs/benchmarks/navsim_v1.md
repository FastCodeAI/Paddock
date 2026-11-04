# NAVSIM v1 (PDMS)

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `navsim` |
| Primary metric | `pdms` |
| Adapter | [`adapters/benchmarks/navsim_v1/`](../../src/paddock/adapters/benchmarks/navsim_v1/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [drivejepa](../models/drivejepa.md), [qwendrive](../models/qwendrive.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

The same OpenScene scenes as [`navsim`](navsim.md), scored by the **v1.1**
devkit as **PDMS**: the composite the 2024 NAVSIM leaderboard and most published
navtest figures use (TransFuser 84.0, Human 94.8, constant velocity 20.6 in the
NAVSIM paper). `navsim` scores v2 EPDMS on the same data; the two are different
composites and are never ranked against each other.

Read [navsim.md](navsim.md) first. Everything about the data, the proxy, the
switches, chunking, resume and retries is shared -- `navsim_v1` is a subclass of
that adapter's harness and proxy agent, and this page covers only what differs.

## What exactly is measured

- **Devkit: NAVSIM v1.1** (`Drive-JEPA/navsim_v1`, `[benchmarks.navsim_v1] devkit`).
  It sits beside the v2 devkit in the same pinned clone, and its scoring code is
  byte-identical to upstream navsim v1.1. This is v1.1, not v1.0: v1.0 multiplied
  driving direction compliance into the score; v1.1 computes it with weight zero,
  which is the paper's formula and what published numbers use.
- **Split: `navtest`**, 136 logs, 12,146 real recorded scenes, single stage.
- **Score: PDMS** = NC × DAC × (5·TTC + 5·EP + 2·comfort) / 12, where NC is no
  at-fault collision and DAC drivable-area compliance (multiplicative), TTC time
  to collision within bound, EP ego progress and comfort the acceleration, jerk
  and yaw-rate bounds over the re-simulated 4 s plan. Driving direction
  compliance is recorded per scenario and does not enter the score.
- **Not EPDMS.** v2 adds traffic-light compliance as a multiplier and lane
  keeping, history comfort and two-frame extended comfort as weighted terms, and
  re-scores zero-valued terms against the human's own (`human_penalty_filter`).
  `comfort` here is v1's single term and is not the same number as v2's
  `history_comfort`; it keeps the scorer's own name for that reason.

## How it differs from `navsim` in the adapter

| | `navsim` (v2) | `navsim_v1` |
|---|---|---|
| scorer script | `run_pdm_score_sequential.py` | `run_pdm_score.py` (the v2 tree also has a file of that name; preflight checks the devkit's `setup.py` version) |
| agent call | `compute_trajectory(input, past_states, metric_cache, interval)` | `compute_trajectory(input, scene)` -- the v1 proxy asks for the Scene to label its plan dump, since v1 never hands the agent the metric cache |
| `Camera.camera_path` | present | added by `patches/Drive-JEPA/0003-navsim-v1-camera-path.patch`, so the proxy ships image paths, not pixels; preflight refuses an unpatched tree |
| sub-metrics | nine | six: NC, DAC, EP, TTC, `comfort`, DDC |
| aggregate rows in the CSV | `average_all_frames`, `pseudo_closed_loop`, `average_expert_frames` | one `average` row, whose `valid` is the AND of every row's |
| smallest scorable run | 2 scenarios (two-frame comfort pairs frames) | 1 |
| metric cache | `metric_cache_v2` | its own; see below |

The switches (`PADDOCK_NAVSIM_AGENT`, `PADDOCK_NAVSIM_ALLOW_PARTIAL`,
`PADDOCK_NAVSIM_ALLOW_SLOW`, `PADDOCK_NAVSIM_OVERRIDES`) are shared by both
benchmarks. An `OVERRIDES` string written for v2 (`traffic_agents_policy=...`)
breaks a v1 run; unset it between them.

## Configuration

```toml
[benchmarks.navsim_v1]
python       = "/path/to/envs/drive-jepa/bin/python"   # the same interpreter as [benchmarks.navsim]
devkit       = "/path/to/Drive-JEPA/navsim_v1"
metric_cache = "/path/to/exp/metric_cache_v1"
split        = "navtest"
workers      = 4
chunk_logs   = 4
run_retries  = 2
nodes        = 4
```

The interpreter's `navsim` is an editable install of the v2 tree. The harness
puts `devkit` first on `PYTHONPATH` for the scorer, so `import navsim` -- and
hydra's `pkg://navsim` config search path -- resolve to the tree being scored.

## The metric cache is per devkit generation

The v2 cache's pickles do not load under v1 (`MetricCache` gained fields, and an
enum v1 does not have). Build a v1 cache from the navtest logs and maps -- no
sensor blobs needed -- with the devkit's own caching script, run the way the
harness runs the scorer:

```bash
python -m tools.benchmarks.navsim.build_metric_cache --bench navsim_v1 --logs 2021.06.03.12.02.06_veh-35_01100_01227   # timing smoke
python -m tools.benchmarks.navsim.build_metric_cache --bench navsim_v1                                                # the split
```

Run the full build detached: about 1.5 s per scenario per worker, so roughly
two hours for navtest at four workers on a shared box (this site measured 62
scenarios in 111 s including startup, then the full split). It skips scenarios
already cached, so a killed build is resumed by running it again. The scorer
reads the cache through `metadata/*.csv`, which the build writes **last**, so
preflight refuses a cache with no csv (an interrupted build) or whose csv lists
fewer scenarios than the split declares (a csv from an earlier, smaller build)
-- either would silently score a subset under the split's name.

The tool exists because the same script started bare from the `navsim_v1`
directory imports the v2 package and builds a v2 cache under a v1 name.
The vendor also publishes a prebuilt v1 cache (`metric_cache.tar` in the
Drive-JEPA Hugging Face dataset); its metadata csv carries the vendor's paths
and its provenance is theirs. Building it is preferred.

## Running it

```bash
paddock run --bench navsim_v1 --model drivejepa --limit 12    # smoke
paddock run --bench navsim_v1 --model drivejepa               # full navtest
paddock run --bench navsim_v1 --model drivejepa --resume <run_id>
```

openpilot is refused on both NAVSIM benchmarks for the same reason: OpenScene is
2 Hz and it needs 10 Hz to perceive motion (see navsim.md).

### Runs with no model process

```bash
python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --agent constant_velocity
python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --agent human
python -m paddock.adapters.benchmarks.navsim.nodeless --bench navsim_v1 --replay navsim/alpamayo15/<run_id>
```

`nodeless.py` runs the scorer without starting a node and stores the result
through the runner's own store path. Three uses:

- **Constant velocity**, stored under the pseudo-model `constant_velocity`,
  `valid: true`: the floor every row is read against.
- **Human**, the recorded future handed back, stored under `human`,
  `valid: false` (privileged): the ceiling. Together the two anchor the whole
  scoring path -- cache, maps, scorer -- against the devkit's published figures
  before any model runs. The NAVSIM paper reports 20.6 and 94.8 on navtest.
- **Replay** of the plans an earlier run recorded, stored under that run's model
  with `config.replayed_from` and a note saying so. Every proxy run since the
  token-labelling fix dumps its plans with scenario tokens; replaying them under
  this devkit scores exactly the plans that produced the v2 EPDMS, with no GPU.
  Older dumps without tokens are refused, and a replay of a run stored
  `valid: false` is stored `valid: false` too, with the source's reason. A
  run that resumed from an earlier run's chunks (`merged_from`) keeps those
  plans in the earlier run's directory; the replay follows that chain, oldest
  first, so the plan that was scored wins.

## Metrics

| metric | direction | meaning |
|---|---|---|
| `pdms` | higher | the composite above, 0–1 |
| `no_at_fault_collisions` | higher | 1 unless the re-simulated ego caused a collision (½ for a stationary one) |
| `drivable_area_compliance` | higher | 1 unless the ego left the drivable area |
| `ego_progress` | higher | progress along the route relative to the PDM-Closed reference |
| `time_to_collision_within_bound` | higher | 1 unless projected time to collision fell below the bound |
| `comfort` | higher | 1 if acceleration, jerk and yaw rate stayed within bounds over the 4 s plan |
| `driving_direction_compliance` | higher | 1 unless the ego drove against traffic; recorded, weight zero in `pdms` |

## Anchors: the scoring path against the paper

Both anchors were run over the full navtest split (12,146 scenarios, zero
failures) on a metric cache built at this site with the v1 devkit, before any
model. The NAVSIM paper's navtest table is the reference.

| term | constant velocity, paddock | paper | human, paddock | paper |
|---|---:|---:|---:|---:|
| no at-fault collisions | 68.02 | 68.0 | 100.00 | 100 |
| drivable area compliance | 57.84 | 57.8 | 100.00 | 100 |
| time to collision within bound | 50.03 | 50.0 | 100.00 | 100 |
| comfort | 100.00 | 100 | 99.90 | 99.9 |
| ego progress | 19.44 | 19.4 | 86.96 | 87.5 |
| **PDMS** | **20.65** | **20.6** | **94.55** | **94.8** |

Runs `20260911T034553-8f28ac` (`constant_velocity`, valid) and
`20260911T041230-67c4f3` (`human`, stored `valid: false` as privileged).

**Verified:** every term of the constant-velocity baseline, and every term but
one of the human's, match the paper to the precision it prints. The cache, the
maps, the v1.1 scorer and the PDMS vocabulary are therefore the paper's.

**Argued:** the human's 0.25-point PDMS gap sits entirely in ego progress
(86.96 against 87.5). Progress is normalised against the PDM-Closed reference
plan stored in the metric cache, and a human plan runs close to that reference,
so it is the term most sensitive to how the cache was built (this site's
nuplan-devkit 1.2.0 against whatever produced the paper's cache). The
constant-velocity progress, far from the reference, matches. Not investigated
further: the human is an anchor, not a result.

## v2 is unchanged by the shared-code changes

The v2 harness, proxy and parser were generalised for this benchmark. To show
the v2 numbers did not move, the constant-velocity baseline was re-run over the
full navtest through the refactored path (`nodeless --bench navsim --agent
constant_velocity`, run `20260911T051742-c99808`) and compared per scenario with
the stored record `20260902T175923-feba73`:

- mean EPDMS 0.238544 in both;
- 12,139 of 12,146 scenarios identical in every metric;
- the other 7 differ only in `ego_progress` (and so in `epdms`), by at most
  1e-11 -- last-bit floating-point noise in the progress normaliser.

The re-run is not committed, so the leaderboard does not carry two identical
baseline rows. Its record is kept at
`{runs}/ops/navsim_v1/v2_regression_record/20260911T051742-c99808/`, its
artifacts (samples, per-chunk CSVs, scorer logs) under
`{runs}/navsim/constant_velocity/20260911T051742-c99808/`, and the comparison
script beside the record as `compare_cv.py`. The comparison's output, verbatim:

```
compare_cv: old 20260902T175923-feba73 n=12146 mean=0.238544; new 20260911T051742-c99808 n=12146 mean=0.238544
compare_cv: 12146 common tokens, 12139 identical across every metric; worst |delta| 9.78e-12 at navsim/navtest/29be20efdac25f74 ego_progress
compare_cv: DIFFERENCES FOUND
```

The last line is the script demanding exact equality; the seven cells it
counts as different are the 1e-11 ones above.

## Full navtest (12,146 scenarios)

All rows are the full split with zero failed scenarios and no unscored chunks;
records under `results/navsim_v1/`. The v2 column is the same model's EPDMS
from `results/navsim/` for reference: a different composite, not a ranking.

| model | PDMS | NC | DAC | EP | TTC | comfort | DDC (weight 0) | v2 EPDMS | run | how |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| drivejepa | **0.892** | 0.984 | 0.985 | 0.887 | 0.881 | 0.999 | 0.988 | 0.843 | `20260911T082329-575df6` | proxy, 7.5 h |
| alpamayo15 | **0.676** | 0.941 | 0.788 | 0.604 | 0.878 | 0.979 | 0.885 | 0.608 | `20260911T044805-0aea35` | replay of `navsim/alpamayo15/20260903T183818-5879c1`'s 12,146 plans, 30 min, no GPU |
| visionpilot | **0.488** | 0.771 | 0.818 | 0.451 | 0.617 | 0.996 | 0.920 | 0.431 | `20260911T083323-c07583` | proxy, four nodes, 8.2 h |
| meteor | **0.366** | 0.693 | 0.687 | 0.340 | 0.582 | 0.997 | 0.800 | 0.303 | `20260915T005223-071c0b` | replay of `navsim/meteor/20260914T234307-e56cfc`, 12 min. Its plans are 3 s extended to 4 s (`METEOR_HORIZON_S`); see `docs/models/meteor.md` |
| constant_velocity | **0.207** | 0.680 | 0.578 | 0.194 | 0.500 | 1.000 | 0.799 | 0.239 | `20260911T034553-8f28ac` | node-less baseline, 27 min |
| human (`valid: false`) | 0.946 | 1.000 | 1.000 | 0.870 | 1.000 | 0.999 | 0.988 | — | `20260911T041230-67c4f3` | node-less anchor, 35 min |

With the **10 Hz history from nuPlan** (the v2 runs of 14 Sep 2026, whose plans
were replayed here once those runs were in the store -- a store test requires
every `replayed_from` to resolve; both sources are merges of an earlier and a
completing run, and the replay read both directories):

| model | PDMS | NC | DAC | EP | TTC | comfort | DDC (weight 0) | v2 EPDMS | run | how |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| openpilot, 10 Hz | **0.807** | 0.969 | 0.913 | 0.719 | 0.936 | 1.000 | 0.948 | 0.747 | `20260914T181009-dbcef4` | replay of `navsim/openpilot/20260911T161513-3a4c90`, 13 min |
| alpamayo15, 10 Hz | **0.753** | 0.957 | 0.856 | 0.690 | 0.896 | 0.991 | 0.927 | 0.691 | `20260914T175447-89a9ca` | replay of `navsim/alpamayo15/20260912T040211-a9562c`, 15 min |
| autoe2e, 10 Hz | **0.538** | 0.824 | 0.777 | 0.514 | 0.700 | 1.000 | 0.859 | 0.500 | `20261104T210206-3022fb` | replay of `navsim/autoe2e/20261009T055926-3ceb18`, 95 min with the box loaded |

The 10 Hz history changes what the model sees, not what is scored: the same
navtest scenes, the same v1 cache. For Alpamayo it is worth +0.077 PDMS over
its 2 Hz plans; OpenPilot, refused at 2 Hz, scores above Alpamayo at 10 Hz.
The README reports the 10 Hz pair for both models, so each model's two columns
describe one run.

Two consistency checks across the scorer generations, for the two models whose
plans are the same under both: Drive-JEPA is deterministic and Alpamayo's plans
were replayed, and for both the no-at-fault-collision and drivable-area terms
agree with the v2 run to three or four decimals (0.984 / 0.985 and 0.941 /
0.788). The terms that move are the ones v2 redefined: time to collision (v2
applies its human-penalty filter and a collision horizon), ego progress, and
comfort, which is a different metric in each generation.

The wall-clock figures were measured with the box near its CPU limit and the
guard holding paddock's pools to two cores for long stretches; the v2 runs of
the same models took about half as long.

Drive-JEPA's paper reports 93.3 PDMS. That figure is its v1-trained checkpoint
through its own scorer-aware agent; paddock scores the released v2 checkpoint
through the proxy, so the two are not the same measurement and 0.892 is the
comparable one here.

## Support-matrix results

The per-metric results behind this benchmark's column in the
[README's support matrix](../../README.md#support-matrix), and the notes on its
cells. Moved here from the README unchanged except for links; the symbols are
the README's former footnote marks, and each note below starts with the mark it
explains. Marks about a model rather than this benchmark (∗ ◊ ◇ ✦ ✧) are
explained in [What these models are](../models/comparison.md#notes-moved-from-the-readme-support-matrix).

### NAVSIM v1

The same 12,146 navtest scenarios, scored by the v1.1 devkit as PDMS =
NC × DAC × (5·TTC + 5·EP + 2·comfort) / 12 (`docs/benchmarks/navsim_v1.md`).
Driving direction compliance is recorded with weight zero. Full navtest,
12,146 scenarios each, zero failures. Drive-JEPA and VisionPilot are fresh
proxy runs; the Alpamayo, OpenPilot and AutoE2E rows replay the plans their v2 runs
recorded (the 10 Hz nuPlan history and, for Alpamayo, the 2 Hz OpenScene run
too), so each scores the very plans behind the v2 figure above. The baseline
reproduces the NAVSIM
paper's constant-velocity row (20.6) term by term, and the recorded human
future scores 0.946 against the paper's 94.8 (privileged; stored `valid: false`).

| Model | PDMS ↑ | No at-fault collisions ↑ | Drivable area ↑ | Ego progress ↑ | TTC within bound ↑ | Comfort ↑ | Driving direction (weight 0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenPilot, 10 Hz nuPlan history ‡ | 0.807 | 0.969 | 0.913 | 0.719 | 0.936 | 1.000 | 0.948 |
| Alpamayo 1.5, 10 Hz nuPlan history ‡ | 0.753 | 0.957 | 0.856 | 0.690 | 0.896 | 0.991 | 0.927 |
| Alpamayo 1.5, 2 Hz OpenScene | 0.676 | 0.941 | 0.788 | 0.604 | 0.878 | 0.979 | 0.885 |
| Drive-JEPA | **0.892** | 0.984 | 0.985 | 0.887 | 0.881 | 0.999 | 0.988 |
| VisionPilot | 0.488 | 0.771 | 0.818 | 0.451 | 0.617 | 0.996 | 0.920 |
| METEOR ◊ | 0.366 | 0.693 | 0.687 | 0.340 | 0.582 | 0.997 | 0.800 |
| Qwen-Drive 1.0, 10 Hz nuPlan history ‡ | 0.876 | 0.982 | 0.964 | 0.817 | 0.941 | 1.000 | 0.987 |
| AutoE2E v1.0, 10 Hz nuPlan history ‡ | 0.538 | 0.824 | 0.777 | 0.514 | 0.700 | 1.000 | 0.859 |
| Baseline (Constant Velocity) | 0.207 | 0.680 | 0.578 | 0.194 | 0.500 | 1.000 | 0.799 |

**Qwen-Drive reproduces its own published row here, term by term.** Its
technical report gives the `planner-sft`-without-reasoning navtest row — the
exact configuration paddock runs — and every term agrees:

| | NC | DAC | EP | TTC | Comfort | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| vendor, SFT w/o reasoning | 98.2 | 96.4 | 82.0 | 94.4 | 100.0 | 87.8 |
| paddock | 98.19 | 96.42 | 81.73 | 94.06 | 99.99 | **87.58** |

Every term is within 0.34 and five of the six within 0.27. This is the only row
in the table with a vendor figure to check against, and matching the *terms*
rather than just the composite is evidence about the whole path — the camera
mapping, the 10 Hz history, the 8-pose conversion and the v1 scorer — not only
about the model. Read it as an in-distribution result all the same: Qwen-Drive
declares `navtrain` (see ✦).
