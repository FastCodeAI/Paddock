# NAVSIM v1 (PDMS)

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
  Older dumps without tokens are refused.

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

## Status

The adapter, the v1 metric cache and the anchors are being run on this site;
figures land here, in the README column and in `results/navsim_v1/` as they
complete, with what was verified separated from what was argued.
