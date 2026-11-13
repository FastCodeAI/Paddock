# navhard (NAVSIM v2.2, two stages)

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Loop | open loop |
| Dataset | `navhard` |
| Primary metric | `epdms` |
| Adapter | [`adapters/benchmarks/navhard/`](../../src/paddock/adapters/benchmarks/navhard/) |
| Models paddock can run on it | [alpamayo15](../models/alpamayo15.md), [autoe2e](../models/autoe2e.md), [drivejepa](../models/drivejepa.md), [meteor](../models/meteor.md), [openpilot](../models/openpilot.md), [qwendrive](../models/qwendrive.md), [uniad](../models/uniad.md), [vad](../models/vad.md), [visionpilot](../models/visionpilot.md) |

<!-- /summary -->

NAVSIM v2's leaderboard protocol: EPDMS on `navhard_two_stage`, scored by the
**v2.2** devkit as *pseudo closed loop*. The model plans on a real recorded
scene (stage one), then again on synthetic follow-up scenes that start near
where a 4 s plan could have taken it (stage two). Each follow-up is weighted by
how close its start is to where the model's own stage-one plan ended, so a
model is scored on the situations its own plan leads into.

Read [navsim.md](navsim.md) first. The proxy agent, the node pool, preflight
and the site switches are shared: `navhard` is a subclass of that adapter's
harness. This page covers what differs. [`navsim`](navsim.md) scores
`navtest`, stage one only; the two numbers are the same metric on different
data and are never ranked against each other.

## What exactly is measured

- **Scorer: NAVSIM v2.2's own `run_pdm_score.py`**, run through
  `adapters/benchmarks/navhard/score.py`. That file imports the official
  module, wraps one function to record each frame's score, and calls the
  official `main()`; nothing it scores is computed in paddock. The devkit is a
  site setting (`[benchmarks.navhard] devkit`), a checkout of
  `autonomousvision/navsim` at v2.2. The stored runs below used commit `0a380a9`;
  `run.json` does not record it.
- **Split: `navhard_two_stage`.** 450 real stage-one frames, in 225 pairs of a
  frame and the one 0.5 s before it in the same log. Each stage-one frame has
  pre-rendered synthetic follow-ups (Drive-JEPA's run records 5,462 planned).
- **Score per stage-one frame:** its stage-one EPDMS × the weighted mean of its
  follow-ups' EPDMS. The weights are the devkit's gaussian of the distance
  between a follow-up's start and the model's own stage-one endpoint.
- **Headline: `epdms`**, the mean of that product over the frames. It equals
  the devkit's `extended_pdm_score_combined` (the mean over a pair's two frames,
  then over pairs); the harness checks the two agree to 1e-6 and refuses the run
  if they do not. A frame whose pair partner has no score is left out, as the
  devkit's pair mean does.
- **One scorer invocation, no chunks, no by-token retry.** A frame's follow-ups
  and its pair must be scored together, so `chunk_logs` is 0 whatever the site
  says.
- **2 Hz only.** The synthetic scenes exist only at OpenScene's 2 Hz, so there
  is no denser history to give a model: every model here gets a 2 Hz history,
  including Qwen-Drive and AutoE2E, whose navtest cells use nuPlan's 10 Hz.
  OpenPilot, whose `min_input_hz` is 10, is scored at it too and ranked: 2 Hz
  is this benchmark's rate for every model, not a shortfall of the site. Its
  adapter shows each 0.5 s frame 10 times to its 20 Hz model, and its run
  config records `below_min_input_hz`. supercombo infers its own speed from
  the change between frames, so it reads held frames as near standstill (as
  on HUGSIM at 4 Hz); its score measures it at that rate.

### Metrics

| metric | direction | what it is |
|---|---|---|
| `epdms` | ↑ higher is better | stage-one EPDMS × weighted mean of the follow-ups' EPDMS, per stage-one frame; the primary metric |
| `epdms_stage_one` | ↑ higher is better | the frame's own EPDMS on the real scene |
| `epdms_stage_two` | ↑ higher is better | the weighted mean of its follow-ups' EPDMS |
| `no_at_fault_collisions_stage_one`, `no_at_fault_collisions_stage_two` | ↑ higher is better | no at-fault collision, per stage (multiplier in EPDMS) |
| `drivable_area_compliance_stage_one`, `drivable_area_compliance_stage_two` | ↑ higher is better | stays in the drivable area (multiplier) |
| `driving_direction_compliance_stage_one`, `driving_direction_compliance_stage_two` | ↑ higher is better | does not drive against traffic (multiplier) |
| `traffic_light_compliance_stage_one`, `traffic_light_compliance_stage_two` | ↑ higher is better | does not run a red light (multiplier) |
| `ego_progress_stage_one`, `ego_progress_stage_two` | ↑ higher is better | progress along the route against the PDM planner's (weighted term) |
| `time_to_collision_within_bound_stage_one`, `time_to_collision_within_bound_stage_two` | ↑ higher is better | time to collision stays above the bound (weighted term) |
| `lane_keeping_stage_one`, `lane_keeping_stage_two` | ↑ higher is better | stays near the lane centre (weighted term) |
| `history_comfort_stage_one`, `history_comfort_stage_two` | ↑ higher is better | acceleration, jerk and yaw-rate bounds over history and plan (weighted term) |
| `two_frame_extended_comfort_stage_one`, `two_frame_extended_comfort_stage_two` | ↑ higher is better | consistency between consecutive frames' plans (weighted term) |

Each stage's sub-metric is the devkit's own value: the frame's on stage one, the
weighted mean over its follow-ups on stage two. EPDMS itself is defined in
[navsim.md](navsim.md).

## Drive-JEPA's streamed mode

Upstream's v2.2 scorer hands no agent a previous plan, and each scene reaches
the node through `/infer_sequence`, which resets it, so Drive-JEPA's momentum
stage never acts in a standard run. `PADDOCK_NAVHARD_STREAMED=1` (off by
default) makes it act on the 225 stage-one frames that have a frame 0.5 s
earlier in the same log: the proxy names each real frame's log, timestamp and
pose, the node keeps its own previous plan per log, and the scorer plans each
log's stage-one frames in time order. Synthetic follow-ups are sent as in a
standard run. The run records `config.streamed = "per_log"` (a new run id) and
counts in `by_group.streamed`. The details are in
[navsim.md, "navhard's streamed mode"](navsim.md#navhards-streamed-mode-opt-in).
A streamed row is a different protocol from upstream's navhard and from a
standard paddock run.

## Configuration

```toml
[datasets.navhard]
path = "/path/to/navhard_root"     # navsim_logs/, sensor_blobs/, maps/ and navhard_two_stage/
exp  = "/path/to/navhard_exp"      # the scorer's NAVSIM_EXP_ROOT; preflight requires it

[benchmarks.navhard]
python       = "/path/to/envs/drive-jepa/bin/python"   # an interpreter that imports the v2.2 devkit
devkit       = "/path/to/navsim"                        # autonomousvision/navsim at v2.2
metric_cache = "/path/to/metric_cache_navhard_v22"      # v2.2's run_metric_caching.py, train_test_split=navhard_two_stage
split        = "navhard_two_stage"
workers      = 4
gpu_nodes    = 1
```

Every other key is read as `[benchmarks.navsim]` reads it (see
[navsim.md](navsim.md)), except `chunk_logs`, which is always 0 here.
`paddock data fetch navhard` downloads the data, pinned: from OpenScene's Hugging
Face repository at a fixed commit, the navtest logs and cameras that stage one reads
and the three `navsim_v2.2_navhard_two_stage_*` archives; from nuPlan's own host, the
maps, checked by size and sha256.

## Results

All nine runs are the full split, 450 stage-one frames, all 450 scored, and carry
`sequences_sha256: 3af39efa412cc8df`, so they are the same frames. In each, the
per-frame mean matches the devkit's combined row (`by_group.source_aggregates.paddock_check`).
NC is no at-fault collisions and DAC drivable-area compliance.

| Model | EPDMS ↑ | stage one ↑ | stage two ↑ | NC, stage one | DAC, stage one | NC, stage two | DAC, stage two | run | stored `valid` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Drive-JEPA ⟲ | 0.4277 | 0.818 | 0.529 | 0.956 | 0.960 | 0.884 | 0.852 | `20261110T165434-167fd4` | true |
| Qwen-Drive 1.0 | 0.3124 | 0.763 | 0.427 | 0.969 | 0.880 | 0.812 | 0.745 | `20261111T060112-5231af` | true |
| Alpamayo 1.5 | 0.1928 | 0.507 | 0.366 | 0.924 | 0.664 | 0.841 | 0.650 | `20261111T090014-04904f` | true |
| VisionPilot ∗ | 0.1921 | 0.453 | 0.390 | 0.921 | 0.618 | 0.847 | 0.672 | `20261110T165413-30c0d8` | true |
| UniAD ⊟ | 0.1234 | 0.386 | 0.300 | 0.890 | 0.527 | 0.806 | 0.535 | `20261110T092734-19b732` | true |
| Baseline (Constant Velocity) | 0.1148 | 0.290 | 0.342 | 0.889 | 0.429 | 0.832 | 0.591 | `20261110T085953-e122fd` | true |
| VAD ⊟ | 0.0990 | 0.343 | 0.250 | 0.860 | 0.522 | 0.769 | 0.491 | `20261110T185531-0cecc0` | true |
| METEOR ◊ | 0.0878 | 0.315 | 0.261 | 0.868 | 0.507 | 0.779 | 0.545 | `20261110T181854-9b4014` | true |
| AutoE2E | 0.0822 | 0.330 | 0.240 | 0.858 | 0.520 | 0.764 | 0.436 | `20261110T192825-24d0b3` | true |

`epdms` is the mean of a per-frame product, so it is not the product of the two
stage columns.

⟲ **Drive-JEPA is momentum-aware, in streamed mode** (`config.streamed =
"per_log"`): 225 of the 450 stage-one frames were handed the plan the node made
0.5 s earlier in the same log, and none of the synthetic follow-ups. The site
pointed `DRIVEJEPA_CHECKPOINT` at a copy of the released checkpoint on a faster
disk; the copy's sha256 is
`740335c800e248598949c2f3c70700da79fe52f5fdc282b887d08fe112d30d05`, the same as
recorded for the file model.yaml declares, and both are 3,735,352,913 bytes. A
byte-identical copy is not an override, so the record, written again from its
stored chunks (`--resume`, nothing scored again), is ranked.

∗ **VisionPilot's plan is built by paddock around its outputs, and most of it
is extrapolation.** The stack plans 1.0 s ahead; the adapter extends that to
the 4 s NAVSIM scores (`PLAN_HORIZON_S`), so every point after the first second
continues a controller that never plans that far. See
[docs/models/visionpilot.md](../models/visionpilot.md).

⊟ **UniAD and VAD plan 3 s; their plan is held to 4 s by paddock**
(`UNIAD_HORIZON_S` / `VAD_HORIZON_S` = 4.0), a second the model never planned.
Their model.yaml lists that setting as ranked with a note, as METEOR's is; the
records, written again from their stored chunks (`--resume`, nothing scored
again), are ranked with that note (see ⊟ in
[navsim.md](navsim.md#notes-on-the-matrix-cells)).

◊ METEOR plans 3 s and is continued to 4 s by `METEOR_HORIZON_S`, which its
model.yaml lists as ranked with a note.

The constant-velocity row is paddock's own agent, run with no model process.
Neither it nor any row here is comparable with navtest EPDMS or with published
navhard figures without checking that the scorer, split and stage weights
match.
