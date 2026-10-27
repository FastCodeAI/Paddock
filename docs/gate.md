# The regression gate

`tools/gate.py` answers one question before a new checkpoint (or a code change)
replaces an old one: **does the candidate lose to its baseline anywhere, by more
than chance?** It never runs a model itself except through `paddock run`, and it
never edits a result; it reads the per-sample records two runs already wrote and
compares them pair by pair.

```bash
python -m tools.gate compare --bench neuroncap \
    --baseline drivejepa:20260905T220225-f11ac0 --candidate drivejepa:20261012T224707-f11ac0
python -m tools.gate compare --bench nuscenes --results results \
    --baseline drivejepa:<run> --baseline-results ../other-worktree/results --candidate drivejepa:<run>
python -m tools.gate check manifest.toml            # every [[pair]] in it, one overall verdict
python -m tools.gate smoke --model drivejepa \
    --env DRIVEJEPA_CHECKPOINT=/path/to/candidate.ckpt \
    --baseline /path/to/baselines.toml               # prints the plan; add --execute to run it
python -m tools.gate subset neuroncap                # one smoke subset, one id per line
```

Run it with paddock's own interpreter. `compare` and `check` need the site
config only to find `{roots.runs}`, where `samples.jsonl` lives.

## How a pair is scored

Both runs' `samples.jsonl` are joined on `sample_id`; only shared samples count.
The difference is candidate minus baseline, **signed so that positive is
better** (`paddock.results.report.lower_is_better` decides the direction), so it
equals the gap between the two means over the shared samples.

Its standard error is **clustered on the benchmark's independent unit**: the
scenario-scene pair on neuroNCAP, the scene on nuScenes and KITScenes, the log on
NAVSIM, the route on Bench2Drive, the clip on AlpaSim. Samples from one scene
move together; counting them as independent once turned a neuroNCAP "not
separable" into t = -4.25. NAVSIM samples name only their token, so the log
comes from the runs' plan dumps (`trajectories.jsonl`, following
`config.merged_from`); a pair whose dumps do not name a sample's log gets NO
UNIT, never a verdict over tokens.

The verdict compares t with **t crit**, the two-sided 5% critical value of
Student's t on (units - 1) degrees of freedom. A clustered standard error from G
units has about G - 1 degrees of freedom, and the smoke has few units: on 3
scenario pairs a fixed |t| >= 2 fires 18% of the time when nothing changed, and
the 5% bar is 4.30. Paddock's env has no scipy, so the values come from a table
in `tools/gate.py` (checked against scipy; `tests/test_gate.py` pins it).

**Several pairs at once.** Each pair is tested at 5%, 2.5% per direction, so an
unchanged model is called a REGRESSION on at least one pair more often than
that: about 10% of the time on the quick tier's 4 pairs and about 16% on
`--tier all`'s 7, if the pairs were independent. (They are not quite: the two
neuroNCAP subsets share two pairs.) A FAIL on one pair, alone, is a reason to look
at that pair's numbers, not proof the candidate is worse.

## Reading a verdict

```
bench        metric          baseline candidate better by      SE      t t crit     MDE   units  verdict
neuroncap    ncap_score         3.568     4.130     0.562   0.439   1.28   2.09   1.297      20  NEUTRAL
             model_env: baseline {}, candidate {}
             note: neither run overrides drivejepa's env, so both ran model.yaml's own weights: ...
```

**MDE** is the smallest true difference the pair detects 80% of the time,
(t crit + t80) x SE, in the metric's units. It is how a NEUTRAL is read: a
NEUTRAL with an MDE of 1.3 points says the candidate is not worse by 1.3 or more;
it says nothing about a 0.5-point drop. It comes from the pair's own SE, so it
moves with how different the candidate is: on neuroNCAP 20 pairs, a small
finetune of the same model gives about 0.55, a change of what the model sees
about 1.5 (measured from the 200-episode sweeps, 2026-10-21).

| verdict | means |
|---|---|
| GAIN | t >= t crit: better, beyond chance at 5% |
| REGRESSION | t <= -t crit: worse, beyond chance at 5% |
| NEUTRAL | not separable at this many units. Not "the same": read the SE |
| TOO FEW UNITS | fewer than 2 units, so no standard error and no verdict |
| NO RECORDS | one side has no `samples.jsonl` on this site; means are shown, nothing is claimed |
| NO UNIT | some paired samples have no independent unit (NAVSIM: no plan dump names their log), so the standard error would be computed over samples and come out too small |
| CONFIG DIFFERS | the two runs' `config` differs beyond `model_env`, `declared_env` and the subset keys (`limit`, `scene`, `runs`, NAVSIM's token filter, ...), so a difference could be the setup's, not the candidate's. The keys are in a note; `allow_config = ["key"]` on the pair (`--allow-config key` on `compare`) accepts one the candidate is meant to change |
| INCOMPLETE | the candidate lost samples to a failure (`dataset.n_failed`, or `status: partial`). Pairing on shared samples would drop exactly those, usually the hard cases, so the rest is shown and nothing is claimed |
| SUBSET DIFFERS | the pair names its subset (`subset = "<name>"`, which the smoke writes) and one of the runs does not hold all of it, most often a baseline made on an older draw of the subset. The shared samples are compared and shown, but they are not the measurement the subset defines, so nothing is claimed. The note says which run holds how many; run that one again on this subset |

`check` adds one overall line: **FAIL** if any pair is a REGRESSION (exit 1);
**SHIP** only if every pair was decided (GAIN, NEUTRAL or REGRESSION, so no NO
RECORDS, TOO FEW UNITS or INCOMPLETE), none regressed, and at least one
closed-loop pair that may ship is a GAIN; **HOLD** otherwise. The quick tier's
two crash checks (`crash_check = true`: one Bench2Drive route, one AlpaSim
scene) are always TOO FEW UNITS, so one that ran cleanly is left out of that
rule; one that failed is INCOMPLETE, which holds, and stops the gate. A pair with
`can_ship = false` (the Bench2Drive subset: seven routes detect only a 17-38
point drop; the neuroNCAP quick subset: three episodes) still fails the gate on a REGRESSION, and its GAIN never ships a
checkpoint; the line says `(cannot ship)`. The smoke also holds while any of
its subsets is unfinished, and names them beside the verdict. A `threshold` on a pair also reports whether the point estimate
dropped by more than that many metric units.

Read the notes; they are where the gate says what it cannot check:

- **model_env.** Both runs' `config.model_env` is printed. The runner records a
  `[models.X] env` override such as `DRIVEJEPA_CHECKPOINT` (paths as
  `<site path:digest>`) and, since #82, marks the run `valid: false` unless the
  key is tuning-only. A "candidate" launched without its override is just the
  baseline again. The gate says so
  when the two are identical, and when one side sets a key the other does not.
  Identical is legitimate for a code change on the same weights.
- **declared_env.** The env a model's own `model.yaml` sets is recorded as
  `config.declared_env` (#99). A change to it is something a candidate is meant
  to test, like a code change, so it is not CONFIG DIFFERS: the gate names each
  key that differs, and says when one run has no such record.
- **invalid.** A smoke run of a subset is `valid: false`; the note is expected
  there and a warning anywhere else.
- **code.** `run.json` records the paddock commit a run was made with
  (`provenance.paddock`, since 2026-10-21) and the model's own repo revision.
  When the two runs differ, or one ran from a checkout with uncommitted edits,
  the gate says so: an adapter change between baseline and candidate is part of
  the difference. Older runs have no paddock commit, and the note says that too.
- **unpaired / different sample sets.** Only shared samples are compared.

### PDM terms beside the score

Each benchmark has its own headline, and a headline can stay flat while what
drives it moves. A candidate that collides more and makes more progress can keep
the same EPDMS or driving score. So each pair also shows NAVSIM v1's PDM score
terms, as far as its benchmark can measure them:

    PDM score = NC x DAC x (5*TTC + 5*EP + 2*C) / 12

`pdm:` in each `benchmark.yaml` says which of its metrics fills which term. The
formula, NAVSIM's weights and the open-loop EP and C live in
`paddock.metrics.pdm`. The terms always print in the same order (NC, DAC, TTC,
EP, C, then the composite), labelled by term, under the primary. They use the
same pairing and the same unit as the primary, and they get the same refusals:
an INCOMPLETE pair has INCOMPLETE terms. The terms a benchmark cannot measure
are named in a `not measured on <bench>` line, never filled with a pass.

```
bench        metric                  baseline candidate better by      SE      t t crit     MDE   units  verdict
bench2drive  driving_score             53.500    53.500     0.000   0.378   0.00   2.37   1.233       8  NEUTRAL
               NC  collision_rate       0.000     0.687    -0.687   0.030 -23.30   2.37   0.096       8  REGRESSION
               EP  route_completion    63.500    95.500    32.000   0.845  37.86   2.37   2.756       8  GAIN
               not measured on bench2drive: DAC, TTC, C
```

(That table comes from a test fixture, not a real run.)

| benchmark | NC | DAC | TTC | EP | C | composite |
|---|---|---|---|---|---|---|
| navsim_v1 | NAVSIM | NAVSIM | NAVSIM | NAVSIM | NAVSIM | `pdms`, the real one (it is the headline) |
| navsim | NAVSIM | NAVSIM | NAVSIM | NAVSIM | `history_comfort` | `pdm_style_score`, composed by the gate |
| comma2k19 | radar-lead proxy | own-lane proxy | radar lead | path length | plan | `pdm_style_score` (its `score.py`) |
| alpasim | `collision_at_fault` | `offroad` | - | `progress` | - | - |
| nuscenes | `col_avg`: box overlap, any category, no at-fault call | - (no map loaded) | - | `pdm_ep` | `pdm_comfort` | - |
| bench2drive | `collision_rate` | - | - | `route_completion` | - | - |
| kitscenes, nuplan | - | - | - | `pdm_ep` | `pdm_comfort` | - |
| neuroncap | `pdm_nc`: NAVSIM's at-fault rules, every plan against the actors' real motion | `pdm_dac`: nuScenes' drivable area | `pdm_ttc` | `pdm_ep` | `pdm_comfort` | `pdm_style_score`, the mean of every decision's composite |

A slot says which question a metric answers, not that two benchmarks answer it
the same way: NAVSIM's NC re-simulates the agents, nuScenes' NC is an overlap
with recorded boxes. Read the rows within a benchmark, candidate against
baseline. Do not compare them across benchmarks.

**The composite.** It appears only where all five terms are measured. When a
benchmark has all five and no composite of its own (NAVSIM v2), the gate builds
`pdm_style_score` per sample from the five terms, so runs already on disk have
it too. A sample missing any term gets no composite, which is never a pass.

**PDM rows never change a verdict or the overall line.** Each row is a test of
its own at 5%. If the rows were independent, an unchanged model would show at
least one row beyond t crit in the wrong direction about 35% of the time over
the quick tier's 17 rows, and about 51% over `--tier all`'s 28 (KITScenes 2,
nuScenes 3, NAVSIM 6, each neuroNCAP subset 6, Bench2Drive 2, AlpaSim 3; a
pass counts the composite the gate builds for NAVSIM v2). They are not
independent, so the real rate is lower, by an amount nobody has measured. That
rate is too high for a FAIL to rest on. A row that is worse beyond t crit is
named under the overall line (`PDM terms worse beyond t crit, not part of the
verdict: ...`), and in the smoke's per-subset line, as a pointer for whoever
reads the pair.

PDM rows apply only when the pair is scored on the benchmark's primary metric:
`--metric` with another metric pairs that metric alone. A run with per-sample
records but no values for a term (a run scored before `pdm_ep` existed, say) is
not compared on that term, and one note on the primary names it.

**neuroNCAP has all five, by NAVSIM's rules** (`paddock.metrics.pdm.score_plan`,
fed by `adapters/benchmarks/neuroncap/pdm_terms.py`): at every decision, the
model's 3 s plan against the actors' real motion and nuScenes' drivable area, the
human's logged path as EP's centreline and reference. Its boxes reproduce the
engine's own collision verdict on all 600 runs of three stored sweeps
(`tools/benchmarks/neuroncap/check_pdm_geometry.py`, 223 of them collided). A
sweep made before the terms existed gets them from its stored episodes, in about
2 minutes, with `tools/benchmarks/neuroncap/pdm_rescore.py`: a new run of the
same model, `replayed_from` its source (a subset key here), stored `valid: false`.
Every plan is first driven by NAVSIM's own vehicle simulator, ported in
`paddock.metrics.pdm_sim`, as NAVSIM does before it scores. Re-scored, the camera
fix (Drive-JEPA blind against camera-fixed, 20 pairs x 10) leaves the NCAP score
NEUTRAL (+0.56, MDE 1.30) and the PDM composite NEUTRAL (0.621 to 0.661, MDE
0.21), and moves comfort the wrong way beyond t crit (0.998 to 0.935). Details
in docs/benchmarks/neuroncap.md.

What the other empty cells would take: for AlpaSim, the ego and actor poses each
run already writes; for Bench2Drive, `outside_route_lanes` and the 20 Hz
`metric_info.json`, both written per route and not yet read, or PR #79's recorder.
Those need fresh baselines. For nuScenes and nuPlan, loading their maps and agent
tracks.

## Full runs and the smoke

**Full runs**: the full-benchmark runs already in `results/`, compared with
`compare` or a hand-written `check` manifest. Most power, most GPU time; the
check before a release.

**The smoke**: fixed subsets, tracked in `tools/gate_smoke/<name>.toml`, each
drawn once and headed by how it was drawn. They must never be re-picked to make a
result look better; a subset that turns out wrong is replaced in its own commit,
and its baselines re-run.

**How the subsets are drawn** (since 2026-10-22, `python -m tools.gate_subsets
<benchmark>`, seed 20261022). Each benchmark is split into groups that change how
a model has to drive -- the maneuver (left, right, straight), the speed band
(slow under 3 m/s, fast from 8 m/s), the city or town, the crash type, the
Bench2Drive scenario family -- as far as the benchmark records them. **Every group
gets the same share** (`tools.gate.balanced_pick`), so a right turn at speed is as
present as straight driving, and the samples within a group are drawn at random.
The gate's mean therefore weighs rare driving more than the full benchmark does:
it is a test for what broke, not an estimate of the full score.

The smoke has two tiers of its own, because one checkpoint at a time cannot wait
hours for every answer:

- **quick** (`--tier quick`, the default): every checkpoint. The three open-loop
  benchmarks, three neuroNCAP pairs (one per crash type), and one Bench2Drive route
  and one AlpaSim scene as crash checks. It catches a pipeline break or a large
  regression in about 40 minutes. The two crash checks are one sample each, too
  few for a verdict (TOO FEW UNITS): they show the score, and a checkpoint that
  cannot drive in CARLA or AlpaSim at all is INCOMPLETE and stops the gate
  before the closed tier.
- **closed** (`--tier closed`): a checkpoint worth keeping. The finer neuroNCAP
  subset, Bench2Drive and AlpaSim; a few hours.
- `--tier all` runs quick, then closed only if every quick subset finished and
  none regressed.

| subset | tier | what | groups | units (t crit) | how `paddock run` is told | pairs with |
|---|---|---|---|---|---|---|
| kitscenes | quick | all 223 keyframes; the verdict reads the 186 `val` ones | KITScenes' split: `val` 186, `overlap_train_val` 37 (places also in its training data, compared alone in a note) | 117 scenes (1.98) | nothing: the full sample list | any full run, or a smoke |
| nuscenes | quick | 1,000 of the 5,119 val keyframes, from 143 of the 150 scenes | city area x maneuver x speed: 34 | 143 scenes (1.98) | `--limit 1000` and `PADDOCK_NUSCENES_SAMPLES=<the ids>` | any run of the same config, a full one included |
| navsim (v2) | quick | 1,018 navtest tokens from 126 of its 136 logs: 1,000 drawn, then every log touched once topped up to 2 | city x driving command x speed: 35 | 126 logs (1.98) | `--limit 1018` and `PADDOCK_NAVSIM_OVERRIDES=train_test_split.scene_filter.tokens=[...]` | a smoke of the same list only |
| neuroncap_quick | quick | 3 pairs, 1 run (seed 0); cannot ship | crash type: 1 each | 3 pairs (4.30) | `--scene <its scenes> --runs 1`, `PADDOCK_NCAP_PAIRS=<the pairs>`, `PADDOCK_NCAP_MANAGE_RENDERER=true` | a full sweep (its run 0) |
| bench2drive_quick | quick | route 3717, the shortest (72 m) of the closed tier's 7; a crash check, cannot ship | none | 1 route (no verdict) | `--scene 3717` | the closed tier's baseline |
| alpasim_quick | quick | one of the closed tier's 10 scenes, at random (seed 20261027); a crash check, cannot ship | none | 1 scene (no verdict) | `--scene <its clip>` within `public_2601_gate10` | the closed tier's baseline |
| neuroncap | closed | 15 of the 20 pairs over 13 scenes, 1 run (seed 0) | crash type: 5 each (all frontal and side, 5 of 10 stationary) | 15 pairs (2.14) | as above | a full sweep (its run 0) |
| bench2drive | closed | 7 of the 220 routes; cannot ship | left turn, right turn, and straight by hazard (junction, highway or high speed, obstacle, pedestrian or cyclist, cut-in or hard brake): 1 each | 7 routes (2.45) | `--scene <7 route ids>` | a smoke baseline |
| alpasim | closed | 10 of the 50 `public_2601` scenes downloaded on the site (scene set `public_2601_gate10`), not of all 913 | maneuver x speed, read from each scene's recorded drive: 7 | 10 scenes (2.26) | `scene_set` pinned in the derived config | a smoke baseline |

A subset's baseline is `[baseline.<subset>]` in the baseline manifest, else
`[baseline.<benchmark>]`: one full neuroNCAP sweep serves both neuroNCAP subsets.

A smoke run of a subset is stored `valid: false` with the reason in `notes`
(`--limit`, a few of 20 pairs, a few of the site's routes, a few clips of the
scene set), so it never reaches the leaderboard. Two smokes are their whole
benchmark and are ordinary valid runs: KITScenes (all 223 keyframes) and
AlpaSim (the whole pinned `public_2601_gate10`).

A pair in a `check` manifest may name its subset (`subset = "<name>"`; the smoke
writes it). A subset with `decide_on` has groups that run but do not decide:
KITScenes decides on `val`, and the 37 `overlap_train_val` keyframes are left out
of the verdict and compared alone in a note on the pair.

## Running the smoke

To have the site's shared worker run it for you, one job at a time, submit it to
the queue instead: `python -m tools.gate_queue submit --model drivejepa --env
DRIVEJEPA_CHECKPOINT=/path/to/ckpt` ([queue.md](queue.md)). By hand:

```bash
python -m tools.gate smoke --model drivejepa --env DRIVEJEPA_CHECKPOINT=/path/to/ckpt \
    --baseline /path/to/baselines.toml                 # plan only, quick tier
setsid nohup python -m tools.gate smoke ... --tier all --execute \
    --cores 4-7 --cores 8-11 --cores 12-15 \
    --stage-dir /fast/local/disk/gate_cache > smoke.out 2>&1 < /dev/null &
```

- **Lanes.** Each `--cores` is one lane: a `taskset` core list and its own port
  range (`--base-port`, then +100 per lane). Subsets of the same tier run side by
  side, one per lane, never two of one benchmark at once. Each launch waits for
  the GPU memory below, and the next one waits until this one's model has said
  it is ready, so every GPU check sees the memory already taken. A subset whose
  benchmark runs a simulator (neuroNCAP, Bench2Drive, AlpaSim) starts it inside
  its first episode, after the model is ready, and neuroNCAP restarts its
  renderer at every scene, so its memory may not be taken yet when the next
  check reads the card. Each running simulator's share (8 GB for neuroNCAP and
  Bench2Drive, 17 GB for AlpaSim) therefore counts as taken until its lane ends.
  With enough lanes and memory, neuroNCAP, Bench2Drive and AlpaSim run side by
  side; until 2026-10-27 the closed tier ran one subset at a time. Each keeps its
  lane's cores: neuroNCAP's renderer and CARLA as the lane's children, AlpaSim's
  Docker containers because its launcher pins them to the cores its run was
  given (shakaal fails above about 90% CPU). Two subsets of one benchmark never
  run at once. With one `--cores` (the default, `4-7`)
  everything runs one after another.
- **NAVSIM gets two model nodes.** The derived config sets
  `[benchmarks.navsim] gpu_nodes = 2` unless the site sets its own: the scorer's
  four workers shared one node, which answers one scenario at a time. Each
  scenario is still one request to one node, and `gpu_nodes` is not in the run
  config, so the runs pair with baselines made on one node. The GPU wait counts
  both nodes.
- **Stop at the first regression.** When a finished subset is a REGRESSION
  against its baseline, nothing else is launched; running ones finish. The
  closed tier never starts after a quick-tier regression or an unfinished quick
  subset. `--keep-going` turns both off.
- **Stopping it.** `kill <gate pid>` (SIGTERM) or Ctrl-C. Every running lane
  gets SIGINT, so `paddock run` tears down its model node, renderer or CARLA,
  and is killed if it has not exited within 2 min; then the gate exits 130. Each
  lane's PID is in `state.json` while it runs, and a re-run refuses to start
  while a recorded run is still alive (after a `kill -9` of the gate, say). The
  same command resumes what was stopped.
- **`--stage-dir`.** Every `--env` value that names a file, typically the
  candidate's checkpoint, is copied there first (with any file under 1 MB beside
  it, such as gallops' `checkpoint.json`), and the run reads the copy. Point it
  at a fast local disk: a checkpoint read from a slow network share was enough to
  make node starts time out.

Without `--execute` it prints, per benchmark, the exact `paddock run` line
(`taskset -c 4-7 nice`, `PADDOCK_BASE_PORT`, `PYTHONPATH` on this checkout's
`src`, the subset's flags) and the manifest it will write. With `--execute` it:

1. writes `PADDOCK_CONFIG` for the candidate: the site `paddock.toml` with the
   `--env` values merged into `[models.<model>.env]`, so the override reaches the
   node and is recorded as `model_env`; the neuroNCAP `renderer_url` moved to
   `--renderer-port` (default 8700), so the renderers the smoke starts do not
   land on the site's port, where a long-lived `paddock renderer` usually is;
   and nuScenes' `index_cache` in `{roots.runs}/gate/cache/nuscenes`, one index
   every checkout shares, unless the site names its own. Neither is in a run
   config, so neither changes what is measured. neuroNCAP's `video` stays the
   site's: with the default `encode`, every episode leaves its mp4s, which is how
   a regression is looked at;
2. launches one benchmark at a time, each after `nvidia-smi` shows enough free
   memory on the card the model will use: the model's declared `vram_gb`, plus
   its simulator's share (`SIMULATOR_GB`: 8 GB for neuroNCAP's renderer, as
   measured; 8 GB assumed for CARLA, not measured), plus 2 GB. For Drive-JEPA
   that is 14 GB open loop and 22 GB closed loop. It gives up after 12 h;
3. streams each run's output to `<state>/<bench>.log`, and its own progress,
   flushed, to stdout;
4. writes `<state>/manifest.toml` pairing each finished run with its baseline, and
   runs `check` on it.

`<state>` is `{roots.runs}/gate/smoke/<label>`, the label being the model and a
digest of `--env` (`drivejepa-released` with none). **It is safe to re-run the
same command after an interruption**: a benchmark whose run has a `run.json` is
skipped, one that was started and not finished continues under `--resume`, and
the rest start fresh. The run id is found as the one new artifacts directory
since launch; if two appeared (someone ran the same model on the same benchmark
meanwhile) it stops and asks rather than resume the wrong one. A benchmark the
model's `model.yaml` does not declare is skipped and said so.

### How long it takes

The first full smoke, the released Drive-JEPA on 2026-10-15/16, ran every subset
one after another, before the tiers existed, on a day when the NAS read at
2.8 MB/s and the GPU was shared:

| bench | wall clock | what dominated |
|---|---|---|
| kitscenes | 23 min | image reads from the NAS |
| nuscenes | 35 min | image reads from the NAS |
| navsim | 30 min | the scorer |
| neuroncap | 132 min (second attempt) | 18 episodes, ~6 min each, rendering 4 cameras |
| bench2drive | 240 min | CARLA at 0.06–0.17x real time, one inference per 0.05 s tick; one route hit its 90 min timeout |
| **total** | **~7.7 h** | |

The same run checked the gate against itself: nuScenes 2.497 against the
released run's 2.497 and neuroNCAP 3.175 against 3.175 on the shared seeds, both
NEUTRAL to the last decimal.

**The tiered smoke, timed** (released Drive-JEPA, `--tier all`, three lanes for
the quick tier, 2026-10-20 18:49-21:30 PDT, state dir
`{roots.runs}/gate/smoke/drivejepa-released-timing-20261021`). A quiet evening:
box CPU 14%, the NAS reading 65 MB/s, the GPU shared with one other job.

| subset | tier | wall clock | against its baseline |
|---|---|---|---|
| kitscenes | quick | 5.7 min | identical (ADE 3.222) |
| nuscenes | quick | 7.8 min, the shared index built once | identical (L2 2.497) |
| navsim | quick | 13.4 min | identical (EPDMS 0.875) |
| neuroncap_quick | quick | 12.8 min | identical (4.000, 3 episodes) |
| neuroncap | closed | 45.0 min | identical (3.175, 18 episodes) |
| bench2drive | closed | 95.7 min, 7.4-23.0 min a route | +7.0 driving score, t 0.82, MDE 28.7: NEUTRAL, cannot ship |
| **total** | | **2 h 40 min**: quick tier 19 min, closed tier 2 h 21 min | overall HOLD |

Most of the difference from the 7.7 h above is the evening, not the code: the
NAS was about 20x faster, and neuroNCAP's 18 episodes took 45 min in all,
where on 2026-10-16 their simulation alone took 104 min on a busy GPU. One run
cannot split the two.
What the code did, seen in the logs: the nuScenes index was built once into the
shared cache and is reused by every checkout from now on; one Bench2Drive route
(3364) found CARLA not yet up and lost 60 s to it instead of 600 s; three lanes
ran the quick tier side by side. Bench2Drive is now the longest part of the
closed tier and the least sensitive one.

The baseline manifest is site-local TOML:

```toml
model = "drivejepa"

[baseline.neuroncap]
run = "drivejepa:20261012T224707-f11ac0"

[baseline.nuscenes]
run = "drivejepa:20261012T215049-798a0d"
results = "/path/to/other/worktree/results"   # where its run.json is, if not this checkout

[baseline.navsim]
note = "missing: why"                           # no `run`: the pair is written as a comment
```

A benchmark with no baseline gets its candidate run and a comment in the
manifest, not a pair.

**Baselines are made per site, not shipped.** The smoke's subsets are tracked;
the runs it pairs against are not. Their per-sample records live under the
site's `{roots.runs}` by design, and a smoke run is a subset, not a result for
`results/`. (The KITScenes smoke is the exception that proves the point: it is
the whole sample list, so its `run.json` is `valid: true`, and committing it
would put a second Drive-JEPA KITScenes row on the leaderboard.) So a site
that has never run the gate makes its own baselines first:

1. Smoke the released model with no `--env`, every tier:
   `python -m tools.gate smoke --model <m> --baseline <empty or partial manifest> --tier all --execute`.
2. Put each finished run in the manifest as `[baseline.<subset>] run = "<m>:<run id>"`,
   with `results` naming the checkout whose `results/` holds its `run.json` if
   that is not the one the gate runs from.
3. Where a full run already exists on the site (a neuroNCAP sweep, full nuScenes
   val), it can serve instead: the smoke's samples are a subset of it. The gate
   refuses a pairing whose run configs differ (CONFIG DIFFERS), so a full run
   made with other settings cannot stand in by accident.

The released model smoked against itself should come out NEUTRAL everywhere; on
shakaal it did (nuScenes and neuroNCAP identical to three decimals), which is
the check that the site's baselines are sound.

## Limits

- **Bench2Drive and AlpaSim report NO RECORDS** against runs ingested from
  campaigns that kept no per-route or per-clip records. A smoke baseline made
  through `paddock run` has them. The released Drive-JEPA's Bench2Drive smoke
  baseline holds all 7 routes of the subset, but most ended "Agent got blocked":
  check it against the 50-route results before trusting it.
- **NAVSIM smoke pairs only with a NAVSIM smoke.** Two-frame extended comfort
  compares each scenario with the next *scored* one in its log, so a token's
  EPDMS depends on which others were scored.
- **The groups need recorded data.** nuScenes' were read from a full run's
  stored samples (its command and future path) and nuScenes' log table; NAVSIM's
  from navtest's own log pickles; Bench2Drive's from its route file; AlpaSim's
  from the recorded drive inside each scene file.
- **AlpaSim draws only from scenes already downloaded.** AlpaSim fetches a
  missing scene from a gated Hugging Face dataset when a run starts, and the
  gate queue's account has no token for it. So the 10 are drawn from the 50
  `public_2601` scenes on shakaal, not from all 913. That pool is not a random
  sample of the suite (the first ten of its list, the twenty drawn for
  `public_2601_30scenes`, twenty more), and thirty of its scenes are the ones
  behind the published 30-scene AlpaSim results. The first draw (2026-10-22, 10
  of all 913 at random) never ran: none of its scenes was on disk.
  **Another site must download these 10 once** before its gate can run AlpaSim:
  the scenes listed in `public_2601_gate10.yaml`, into its
  `[benchmarks.alpasim] scene_cache`, with a Hugging Face token that has access
  to NVIDIA's gated dataset. AlpaSim fetches a missing scene itself when a run
  starts, so one run of this scene set with `HF_TOKEN` set does it.
- **The AlpaSim smoke runs its pinned scene set as the default set**: the
  derived config sets `[benchmarks.alpasim] scene_set = "public_2601_gate10"`,
  so a site configured for `public_2601_30scenes` still runs the subset's 10.
- **Drive-JEPA on AlpaSim runs through the bridge on its native driver's
  settings.** Its `model.yaml` declares alpasim, so the smoke drives it. With
  one rollout per scene, a single scene can move by up to about 0.55 between
  two runs of the same code, from near-ties between two proposals after the
  handover (71d7d03b, bae463d3 and 5f064b02 in
  [the 30-scene comparison](models/drivejepa.md#through-paddock-on-alpasim)),
  while the 30-scene mean moved by 0.013. For this model, compare means, or
  flag only when several scenes move.
- **A checkpoint override invalidates a run** (#82: `src/paddock/runner.py:169`
  `_env_override_reasons`), so a candidate run is never ranked as the released
  model. NAVSIM's `PADDOCK_NAVSIM_OVERRIDES` is recorded in `config.overrides`
  but is not one of its `invalid_reasons`
  (`src/paddock/adapters/benchmarks/navsim/harness.py:741`); the NAVSIM smoke is
  invalid through `--limit` regardless.
- **AlpaSim is not CPU-capped.** Its simulator runs in `docker compose`
  containers, which inherit nothing from the lane's `taskset`/`nice`. `--cores`
  lists that overlap are refused, but the containers' CPU is outside them.
- **Bench2Drive needs `patches/Bench2Drive/` applied** to the site's checkout;
  preflight refuses an evaluator without them. Unpatched, a crashed route kills
  every CARLA its user runs on the GPU, other lanes included.
- **Launches onto one card take turns** (`runtime/lease.py`): the last memory
  check and the model's load hold that card's lock, shared by every user, so two
  lanes or two people never start into the same free memory. A simulator that
  starts inside the first episode loads after the lock is released; the wait
  above counts its memory.
- **The neuroNCAP smoke starts its own renderers** on `--renderer-port`, one
  scene at a time. Two smokes at once need different `--renderer-port` (and
  `--base-port`) values; if the port is taken, neuroNCAP's preflight says so and
  that benchmark does not start.
- Smoke runs write their `run.json` into this checkout's `results/` like any
  run. They are evidence for a decision, not results to commit: the subset ones
  are `valid: false`, and the KITScenes and AlpaSim ones, valid, would add a
  second row for the model.
