# AlpaSim

Closed-loop driving on replayed real scenes. NVIDIA's AlpaSim renders a recorded
clip as a neural reconstruction, hands the policy camera frames at a fixed
control rate, and steps the world forward on what the policy commands. Errors
compound, which is the point: a policy that drifts a metre in the first second
is shown the consequence rather than being re-anchored to the log.

The primary metric is `scene_score`, 0–1, meaned over scenes.

## What you need

| | |
|---|---|
| repo | `alpasim`, pinned at `affc2ea` in `repos.yaml` |
| scenes | the `public_2601` suite that ships inside the AlpaSim repo |
| GPU | ~17 GB for the simulator alone; ~41 GB with a 10B driver resident |
| disk | ~180 MB per scene of rollout artifacts, under `{runs}` |

No dataset download, and **no `datasets.yaml` entry**: the scenes are USDZ
artifacts fetched by AlpaSim's own scene provider on first use, so there is no
root for a site to point at and `requires:` would have nothing to check.
`benchmark.yaml` therefore sets `dataset: ""`, the same as Bench2Drive. Which
scenes a run drove is pinned by `sequences_sha256` in its `run.json`, not by a
path.

## Running it

```bash
# The scene SET comes from [benchmarks.alpasim] scene_set, or --scenarios.
paddock run --bench alpasim --model alpamayo15 --scenarios public_2601_first10
paddock run --bench alpasim --model constant_velocity --scenarios public_2601_first10

# --scene takes individual scene ids, which here are clipgt-<uuid>:
paddock run --bench alpasim --model constant_velocity --scene clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
```

Roughly **5 minutes per scene** with a 10B driver at 4 cameras and 10 Hz — a
10-scene sweep is about 50 minutes, of which the first 5 are the checkpoint
read. The constant-velocity floor over the same 10 scenes takes about 8 minutes,
because the cost is rendering rather than inference.

## Configuration

```toml
[benchmarks.alpasim]
repo    = "{repos}/alpasim"
compose = "docker"            # the wizard drives docker compose
drivers = "{weights}/alpasim-drivers"
```

## The tick config is the comparison group

AlpaSim runs at whatever camera set and control rate you configure, and **those
choices change what is being measured**, not merely how precisely:

| group | cameras | control | `force_gt_duration_us` |
|---|---|---|---|
| `1cam-20Hz` | 1 forward | 20 Hz | 3,000,000 |
| `4cam-20Hz` | 4 (front wide, front tele, cross L/R) | 20 Hz | 3,000,000 |
| `6cam-20Hz` | 6 (front wide, front tele, cross L/R, rear L/R) | 20 Hz | 3,000,000 |

All three groups are the **shared referee**: one control rate, one force-GT
duration and one `eval-config.yaml` across every model, so that for the first
time the rows differ by the model rather than by the harness. How much that
matters is visible in the floor itself — constant velocity scores 0.3562 at
20 Hz on the 0.1.0 driver, 0.7411 on 0.2.0 and 0.8850 on 0.2.3, a spread wider
than the gap between most of the models.

A faster control rate gives a policy more opportunities to correct, and four
cameras see what one cannot. **Numbers from different groups are
not comparable and must never be averaged together.** Every run records its
group in `run.json` under `config.renderer`, `config.topology` and
`config.scene_set`, all of which are hashed into the run id — so two tick
configs cannot collide on one run id. **There is no `config.group` key and no
`by_group` block on this path**: grouping the rows is done by the ingest reader
for recorded campaigns, not by the adapter, which drives one scene per run.

Compare a model against the constant-velocity floor **of its own group**.

## What the scored window actually is

Aggregates do not describe the whole rollout. AlpaSim's aggregation chain
(`eval/aggregation/processing.py`) applies, in order:

1. drop every timestep before `eval_relevant > 0` — the warm-up trim
2. derive the combined events (`offroad_or_collision`, `collision_at_fault`, …)
3. drop any trajectory with a black frame
4. **truncate after the first `offroad_or_collision`** — this keys on
   `collision_any`, so a *not-at-fault rear-end* truncates exactly like an
   at-fault one
5. **truncate again** once `dist_to_gt_trajectory` reaches the run's
   `max_dist_to_gt_trajectory` (4.0 m here)

Both truncations keep the triggering step, which is why a published maximum can
exceed the cap it was truncated on.

The consequence is the trap in this benchmark:

> **A policy that fails early is scored over a shorter window than one that
> drives the scene out.** Its trajectory-fidelity numbers are measured only over
> the easy opening seconds and look *better* as a result.

In the `1cam-20Hz` collection Drive-JEPA was scored over **46.7 %** of its ticks
and OpenPilot over **78.4 %**. On `04749bb9` Drive-JEPA was scored over 9 of 398
ticks — 2.3 % — so whatever fidelity it posted there describes less than half a
second of driving, while OpenPilot's number on the same scene covers 337 ticks.

The scored fraction is **not** recoverable from `results-summary.json` — it needs
the per-rollout `metrics.parquet` replayed through the modifier chain, which is
what `tools/benchmarks/alpasim/scored_fraction.py` does. Until you have run it,
assume nothing about comparability of fidelity numbers across runs.
**Rank on `scene_score` and the outcome rates;
treat cross-run fidelity comparisons as invalid unless the scored fractions
match.**

## `scene_score`, and why the floor beats most models on the ten scenes

```
progress_score = min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)
scene_score    = 0.0  if collision_at_fault or offroad  else progress_score
```

**The hard-failure rule zeroes a scene outright** on an at-fault collision or an
offroad event, whatever progress was made before it. There is no partial credit:
a policy that drives 95 % of a scene and then clips a car scores the same 0.0 as
one that never moved.

That rule produces the result that most needs stating plainly:

| | `scene_score` | progress | vs its floor |
|---|---|---|---|
| OpenPilot supercombo 0.2.0 | 0.9566 | 0.9010 | **+0.0716** |
| **constant-velocity floor 0.2.3, `1cam-20Hz`** | **0.8850** | **0.8290** | *(floor)* |
| VisionPilot v2 2.0.0 | 0.8488 | 0.8202 | −0.0362 |
| Drive-JEPA 0.1.4 | 0.5030 | 0.5652 | −0.3820 |

**Two of the three released models score below constant velocity.** (The
retrained Drive-JEPA checkpoint, an adapted result, is in the table further down.) This is not a bug in
the metric and not a claim that constant velocity drives well — it does not
steer, and it rear-ends things. But a rear-end collision is *not at fault*, so
the hard-failure rule does not zero it, while the at-fault collisions and
offroad events the real policies commit do. Driving straight at a fixed speed
down a mostly-straight recorded clip turns out to be a strong baseline against
that rule — it scores 1.0 on 8 of the 10 scenes.

**This is why the floor is recorded as a run at all** — under
`results/alpasim/constant_velocity/`, marked in its `run.json` notes as a
reference rather than a model. A `scene_score` means nothing without it. Any
model reported on this benchmark should be quoted alongside the floor for its
group, and a model that does not beat the floor has not demonstrated competence,
however respectable its absolute number looks.

## Reference numbers

`public_2601` first 10 scenes, uuid-pinned, n=10, scorer `affc2ea`. Every row is
filed under `results/alpasim/`, so `paddock report` agrees with these tables.

> **These numbers were not produced through this adapter.** They come from
> AlpaSim's own native drivers — `openpilot-supercombo-driver-0.2.0`,
> `visionpilot-v2-driver-2.0.0`, `drivejepa-v2-driver-0.1.4`,
> `alpamayo1_5-driver-0.91.0`, `constant_velocity-driver-0.2.3` — run before the
> paddock bridge existed. That is also why only `alpamayo15` and
> `constant_velocity` declare `alpasim` in their `model.yaml`: declaring the
> other three would claim a pairing runs through the bridge, and nothing here
> demonstrates that. So these rows are **reference values for what the
> benchmark yields**, not evidence about this PR's code path. The live path
> through the bridge is unverified, as the PR says.
>
> **Two later versions of the Drive-JEPA driver exist; one has a row here.**
> `drivejepa-v2-driver-0.1.4b` adds momentum-aware selection to 0.1.4, and
> `drivejepa-v2-driver-0.1.7` drives a retrained checkpoint; 0.1.7's source is
> tracked at `tools/models/drivejepa/alpasim_driver/`. Both were measured on
> 30 scenes (these ten plus twenty more), one rollout each, outside paddock,
> and those numbers are in [docs/models/drivejepa.md](../models/drivejepa.md).
> The ten of 0.1.7's runs that are this benchmark's scenes are ingested as
> `results/alpasim/drivejepa/legacy-drivejepa_017_20hz`, the same way the
> 0.1.4 row was. 0.1.4b has no row.
>
> **METEOR's rows come from a driver that is not tracked here.** `meteor-0.1.1`
> is a native AlpaSim driver written at the Fastcode site; its source is not in
> this repository, so its two rows (`6cam-20Hz`, below) are reference values
> another site cannot yet reproduce.

### `1cam-20Hz` — the shared referee

Five rows on one referee: same control rate, same force-GT duration, same
`eval-config.yaml` (md5 `68cf4aaa89356e0f6053d76eb4ef2321`, byte-identical
across every 20 Hz row, including Alpamayo's and METEOR's).

| model | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|
| OpenPilot supercombo 0.2.0 | **0.9566** | 0.1303 | 0.9010 | 10/10 | — |
| *constant-velocity floor 0.2.3* | *0.8850* | *0.2312* | *0.8290* | *10/10* | *—* |
| VisionPilot v2 2.0.0 | 0.8488 | 0.3170 | 0.8202 | 9/10 | at-fault 1 |
| Drive-JEPA 0.1.4 | 0.5030 | 0.4306 | 0.5652 | 7/10 | at-fault 2, offroad 1 |
| Drive-JEPA 0.1.7, retrained checkpoint (adapted) | 0.9956 | 0.0131 | 0.9765 | 10/10 | — |

**Of the released checkpoints, only OpenPilot clears the floor**, by +0.0716. This is the finding, not a
measurement artefact: `scene_score` saturates at `progress_clipped_rel ≥ 0.8`
and zeroes only on `collision_at_fault` or `offroad`, so a blind constant
velocity that never commits either scores 1.0 on 8 of 10 scenes. Progress
separates what the score cannot — floor 0.8290 against OpenPilot's 0.9010.

Two rows need their caveat quoted with the number:

- **VisionPilot** finishes 0.0362 below the floor and **the entire deficit is
  one at-fault collision** on `048b974e`; the other nine scenes average 0.9431.
  Its `set_speed` is the human clip's mean speed, handed to it per scene — the
  model does not choose its own cruising speed here. Its own `smoke_gates.json`
  reads `ok: false` on 3 of 12 gates (two Prometheus ports, two that look like
  gate-side indexing); none touch the referee pins, which were read from the
  resolved config and are on-pin.
- **Drive-JEPA 0.1.7** is not the released model and not a zero-shot result.
  Its trajectory generator was retrained on Physical AI AV (`train` split), and
  AlpaSim's scenes are reconstructions of clips from that dataset. The training
  code excludes the evaluation scenes, which is taken on that code's word. These
  ten scenes were also driven repeatedly while the driver was developed, so
  0.9956 is not a held-out number. The ten runs are the first ten of a
  30-scene sweep made on 2026-10-10; on all 30 the mean is 0.7697, with three
  at-fault collisions and two off-road events, all on the other twenty scenes.
  Nine of the ten scenes score 1.0000 and the tenth 0.9563.
- **Drive-JEPA**'s 0.5030 is a mean over six early-terminated or hard-gated
  scenes and four full-distance runs that all scored 1.0000. It is a pattern —
  it wins 1, ties 3 and loses 6 against the floor — not one bad scene.

### `1cam-20Hz` on 30 scenes — what the README's support matrix quotes

The same drivers, referee config and scorer, on 30 scenes: the ten above plus
twenty more drawn from `public_2601` (seed 20260922, stratified by speed and
turn, from the scenes of the suite available at renderer version 26.1.112 and
not already among the ten). The list is tracked as
`adapters/benchmarks/alpasim/scenes/public_2601_30scenes.yaml`, and a test
checks it against the rows' `sequences_sha256`. One rollout per scene. The rows are in
`results/alpasim/` as the `…_30scenes` runs; each merges the ten-scene run with
twenty single-scene runs, and the ingester refuses the merge if the driver or
the control rate differs between them.

| model | `scene_score` | sd | progress | at-fault | off-road | vs floor |
|---|---|---|---|---|---|---|
| OpenPilot supercombo 0.2.0 | **0.7726** | 0.3627 | 0.7238 | 0/30 | 3/30 | +0.2850 |
| Drive-JEPA 0.1.7, retrained checkpoint (adapted) | 0.7697 | 0.4022 | 0.8284 | 3/30 | 2/30 | +0.2821 |
| VisionPilot v2 2.0.0 | 0.6289 | 0.4106 | 0.6355 | 4/30 | 1/30 | +0.1413 |
| *constant-velocity floor 0.2.3* | *0.4876* | *0.4252* | *0.5411* | *2/30* | *5/30* | *—* |
| Drive-JEPA 0.1.4 | 0.4209 | 0.4378 | 0.5091 | 7/30 | 5/30 | −0.0667 |

Three things to read with it:

- **The floor moves with the scene set**, from 0.8850 on the ten to 0.4876 on
  the 30. A mean over one set says nothing against a mean over the other, so
  quote the difference from the floor and the number of scenes with any figure.
- **Alpamayo 1.5 has no 30-scene row.** It was run on the first ten scenes
  only, and its 0.9000 is a ten-scene mean in a different camera group.
- **One rollout per scene, so the differences are loose.** Paired over the 30
  scenes, the standard error of the difference from the floor is 0.077 for
  OpenPilot, 0.076 for the retrained Drive-JEPA, 0.080 for VisionPilot and
  0.089 for Drive-JEPA 0.1.4. VisionPilot's +0.1413 and Drive-JEPA's −0.0667
  are both under two standard errors. OpenPilot against the retrained row is
  +0.003 ± 0.045: level, and the bold marks the best released checkpoint, not
  a win over the retrained one.
- **`paddock run --bench alpasim` still drives the ten by default.** The
  default `scene_set` is unchanged; the twenty extra scenes were driven by each
  model's native driver and ingested. `--scenarios public_2601_30scenes` (or
  `scene_set` in `[benchmarks.alpasim]`) selects the 30, and no row has been
  produced that way yet.

### `4cam-20Hz`

| model | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|
| Alpamayo 1.5 (10B), via the referee proxy | **0.9000** | 0.3000 | 0.9603 | 9/10 | at-fault 1 |

Alpamayo has no constant-velocity floor in its own group at 20 Hz, so it cannot
be quoted against one; the `1cam-20Hz` floor is a different measurement and is
not a substitute. Its environment is verified identical to the reference rows'
(`pip freeze` 300/300 packages, image `8a69f005`) and all 25 smoke gates pass on
all ten scenes. Scene-level variance on this model is large — `04749bb9` has
produced 0.0, 0.9148 and 1.0 across identical environments — so neither this
mean nor any other single-rollout mean here should be read to four figures.

### `6cam-20Hz`

| model | scenes | `scene_score` | sd | progress | passed | failures |
|---|---|---|---|---|---|---|
| METEOR, released ONNX as shipped, driver 0.1.1 | first 10 | 0.6869 | 0.3103 | 0.6073 | 10/10 | — |
| METEOR, released ONNX as shipped, driver 0.1.1 | 30 | 0.4232 | 0.3673 | 0.4301 | 24/30 | off-road 5, at-fault 1 |

One sweep of the 30 scenes of `public_2601_30scenes`, one scene per run
directory, made on 2026-10-13/14 and ingested as
`results/alpasim/meteor/legacy-meteor_011_published_30scenes`; the ten-scene
row, `…_first10`, is the first ten of those directories, not a second sweep.
Same control rate, force-GT duration, step count and `eval-config.yaml` as the
other groups, read from each run's resolved config.

- **No floor, so no "vs floor" column.** Constant velocity has not been run on
  six cameras. It reads no camera, so its score would probably be the
  `1cam-20Hz` floor's, but that has not been measured and the one-camera floor
  is not quoted against this row.
- **The cameras are not the recorded ones.** METEOR expects pinhole images, so
  the six cameras are rendered as pinholes with square pixels and no
  distortion, each at the narrower of METEOR's own camera and the recorded
  one: 97.70° for the three wide cameras (recorded about 120°), 69.66° and
  69.72° for the rear pair, 29.60° for the telephoto. The bonnet mask the
  other groups apply is off, because it was drawn for the recorded fisheye
  image and does not line up with a pinhole one. The rendered widths were
  fixed from the first ten scenes' rigs. On four of the twenty extra scenes
  (`cdd9448c`, `6725df2d`, `bd07fb91`, `b1b170d0`) the rear-right camera is
  therefore rendered wider than that scene's recorded camera: by 0.24° on
  `6725df2d`, 0.05° on `cdd9448c` and 0.006° on the other two.
- **Two of METEOR's eight camera slots are empty** (the rear wide and rear
  telephoto), fed as zeros with a neighbour's calibration, as in the model's
  own training for a missing camera.
- **The step-time check failed on 18 of 30 scenes.** The driver's own run
  checks budget 200 ms (the 95th percentile of pre-processing plus inference)
  per fresh plan, and on those scenes it was 201 to 313 ms; the host was
  shared and under load. Whether a late plan changes a score has been checked
  on one scene only: the first scene's 29 scorer metrics are identical to
  those of a run of the same scene made on 2026-09-22, and both runs were
  over the budget on that scene, by different amounts (206 and 283 ms).
- **The driver's other run checks.** Each run directory records the driver's
  own checks, and the step-time check is not the only one that failed. None
  of them is a scorer result and none marks the row invalid; the table says
  what each failure was, read from the runs' logs.

  | driver check | failed on | what the failures are |
  |---|---|---|
  | referee settings and scene list match the pinned run | 30 / 30 | every setting and the renderer version (`26.1.112-59673f88`) match; only the scene-list file differs, because the check pins the ten-scene list and this sweep used the 30-scene one |
  | step time within 200 ms | 18 / 30 | above |
  | plan shape: no heading step over 0.35 rad between waypoints | 17 / 30 | on 14, every such step is at under 1.3 m/s on a plan segment under 0.4 m long: near-stop wiggles where a heading is barely defined. On `04749bb9`, `acd22430` and `b1b170d0` they are real, at up to 25 m/s on segments up to 5.3 m, and those three scenes score 0.23, 0 and 0 |
  | consecutive plans agree within 2 m | 5 / 30 | `02eadd92`: one jump, at 0.2 s, while the simulator still drives the recorded path. `2e1ce314`, `04749bb9`, `acd22430`, `b1b170d0`: jumps of 2 to 5 m at 11 to 24 m/s in the 1.4 s after handover. These are the model's plans changing, not a harness fault |
  | warm-up guard fires on the zeroed-state ticks | 4 / 30 | `5f064b02`, `b44a4a44`, `d2715023`, `e3ed554f`: the car was stationary throughout the zeroed window, so there was nothing to guard. The guard never fired where the state was real and the starting speed was never wrong |
  | rendered camera no wider than recorded | 4 / 30 | above |

  So the failures are either the check's own scope (scene list, near-stop
  wiggles, a stationary start) or the model's plans at speed, which the
  scores on those scenes already show.
- **The first ten and the twenty extra differ sharply**: 0.6869 against
  0.2913. All five off-road events and the at-fault collision are on the
  twenty extra scenes. What the model does on them is in
  [docs/models/meteor.md](../models/meteor.md).

## Reproducing this

**Environment.** AlpaSim's own conda environment, plus:

- `casadi<3.8` — 3.8 changed an API the controller uses.
- On Blackwell (sm_120), the msda extension needs the deformable-attention
  patch. `torch.cuda.is_available()` returning `True` is **not** evidence of
  kernel support — compare `torch.cuda.get_arch_list()` against the device's
  compute capability. A torch built to sm_90 loads happily on sm_120 and then
  fails inside a kernel.

**Gated Hugging Face repos.** Alpamayo 1.5 resolves five, and discovers them one
failure at a time unless all are cached first:

```
nvidia/Alpamayo-1.5-10B       nvidia/Cosmos-Reason2-8B
Qwen/Qwen3-VL-8B-Instruct     Qwen/Qwen3-VL-2B-Instruct   (the processor)
```

Set `HF_HUB_OFFLINE=1` for the driver. This is load-bearing, not an
optimisation: a *cached* gated repo still triggers an unauthenticated HEAD
request that returns 401, and `huggingface_hub` does not fall back to the cache
on a 401.

**Weights are never committed.** Fetch `Alpamayo-1.5-10B` into `{weights}`; the
copy these numbers came from has a shard-manifest sha256 of `9cc6fc6ad2852e66`
(10 files, 20.64 GiB), which is what to compare against.

**Driver startup.** Set `startup_timeout_s: 900`. The default is 120 s and a cold
10B checkpoint read answers the runtime's version probe at about 305 s, so the
default kills the run before the model finishes loading.

**Scene set.** `test_suite_id: public_2601`, `limit_to_first_n: 10`, uuid-pinned
via the suite table — `sequences_sha256` in each `run.json` is the check that two
sites scored the same scenes. The Alpamayo and Drive-JEPA sweeps drove the same
ten scenes one explicit id at a time, so their directories record
`test_suite_id: null`; the split on their records comes from the campaign's
`split` key in `[ingest]`, and the matching `sequences_sha256` is what backs it.

**Driver identity.** The five `driver_version_id`s listed under Reference
numbers, as probed by the runtime at connect. Those are
**AlpaSim's own native drivers**, which is what produced the tables above — see
the note under Reference numbers. A run driven through this adapter reports
`paddock-bridge/<scenario>` instead, and the model it served is in `run.json`
under `model`, not under a `driver_version_id` key: this adapter writes none.

**Sampling, and what the recorded numbers are.** Alpamayo 1.5 generates with
`temperature=0.6`, `top_p=0.98`, and seeds only when `force_determinism` is set.
**This adapter deliberately does not set it** (`launcher.py` carries the
reasoning): `driver.model.force_determinism` seeds a model *AlpaSim* runs, and
under this adapter AlpaSim runs no model — paddock owns the process and the
bridge only relays. Setting it here would be a no-op that reads as a guarantee.
Sampling determinism for a model like Alpamayo belongs to its paddock policy.
So a run driven through paddock is **not** reproducible by this flag, and a
stochastic model's row is one draw unless its own policy seeds itself.

**The recorded Alpamayo run above is one draw per scene.** It ran through the
referee proxy with no seed recorded, and `04749bb9` has scored 0.0, 0.9148 and
1.0 across identical environments — re-running the identical command would not
reproduce it. Read it
as one sample, not as the model's score, and do not compare a fresh per-scene
result against it and conclude anything from a difference. This is the same
property `docs/benchmarks/neuroncap.md` documents for Alpamayo there, where five
seeds on one scenario spanned 0.000 to 5.000 with sd 1.817. The constant-velocity
floor has no such caveat: it is deterministic, which is what makes it the useful
reference.

**Scorer.** Pinned at alpasim `affc2ea`. Note the divergence from the
`e2e_challenge` scorer: `open_loop_collision` does not exist there, so a number
quoted from that scorer is not this one.
