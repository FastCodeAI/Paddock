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

## Reading a verdict

```
bench        metric          baseline candidate better by      SE      t t crit   units  verdict
neuroncap    ncap_score         3.568     4.130     0.562   0.439   1.28   2.09      20  NEUTRAL
             model_env: baseline {}, candidate {}
             note: neither run overrides drivejepa's env, so both ran model.yaml's own weights: ...
```

| verdict | means |
|---|---|
| GAIN | t >= t crit: better, beyond chance at 5% |
| REGRESSION | t <= -t crit: worse, beyond chance at 5% |
| NEUTRAL | not separable at this many units. Not "the same": read the SE |
| TOO FEW UNITS | fewer than 2 units, so no standard error and no verdict |
| NO RECORDS | one side has no `samples.jsonl` on this site; means are shown, nothing is claimed |
| NO UNIT | some paired samples have no independent unit (NAVSIM: no plan dump names their log), so the standard error would be computed over samples and come out too small |
| CONFIG DIFFERS | the two runs' `config` differs beyond `model_env` and the subset keys (`limit`, `scene`, `runs`, NAVSIM's token filter, ...), so a difference could be the setup's, not the candidate's. The keys are in a note; `allow_config = ["key"]` on the pair (`--allow-config key` on `compare`) accepts one the candidate is meant to change |
| INCOMPLETE | the candidate lost samples to a failure (`dataset.n_failed`, or `status: partial`). Pairing on shared samples would drop exactly those, usually the hard cases, so the rest is shown and nothing is claimed |

`check` adds one overall line: **FAIL** if any pair is a REGRESSION (exit 1);
**SHIP** only if every pair was decided (GAIN, NEUTRAL or REGRESSION, so no NO
RECORDS, TOO FEW UNITS or INCOMPLETE), none regressed, and at least one
closed-loop pair is a GAIN; **HOLD** otherwise. The smoke also holds while any of
its subsets is unfinished, and names them beside the verdict. A `threshold` on a pair also reports whether the point estimate
dropped by more than that many metric units.

Read the notes; they are where the gate says what it cannot check:

- **model_env.** Both runs' `config.model_env` is printed. The runner records a
  `[models.X] env` override such as `DRIVEJEPA_CHECKPOINT` (paths as
  `<site path:digest>`) but does not make a run invalid on it, so a "candidate"
  launched without its override is just the baseline again. The gate says so
  when the two are identical, and when one side sets a key the other does not.
  Identical is legitimate for a code change on the same weights.
- **invalid.** A smoke run of a subset is `valid: false`; the note is expected
  there and a warning anywhere else.
- **unpaired / different sample sets.** Only shared samples are compared.

## Full runs and the smoke

**Full runs**: the full-benchmark runs already in `results/`, compared with
`compare` or a hand-written `check` manifest. Most power, most GPU time; the
check before a release.

**The smoke**: fixed subsets, tracked in `tools/gate_smoke/<name>.toml`, each
chosen once by a documented rule and headed by how it was chosen. They must never
be re-picked to make a result look better; a subset that turns out wrong is
replaced in its own commit, and its baselines re-run. The drawing rule is
`tools.gate.stratified_pick` where a draw was seeded.

The smoke has two tiers of its own, because one checkpoint at a time cannot wait
hours for every answer:

- **quick** (`--tier quick`, the default): every checkpoint. The three open-loop
  benchmarks and neuroNCAP's three pairs at 0103. It catches a pipeline break or
  a large regression in about 40 minutes.
- **closed** (`--tier closed`): a checkpoint worth keeping. The finer neuroNCAP
  subset, Bench2Drive and AlpaSim; a few hours.
- `--tier all` runs quick, then closed only if every quick subset finished and
  none regressed.

| subset | tier | what | units (t crit) | how `paddock run` is told | pairs with |
|---|---|---|---|---|---|
| kitscenes | quick | all 223 keyframes | 140 scenes (1.98) | nothing: the full sample list | any full run, or a smoke |
| nuscenes | quick | first 30 val scenes, all 1,011 anchors | 30 scenes (2.05) | `--limit 1011` | a full run (a prefix of it) |
| navsim (v2) | quick | 1,000 navtest tokens, every one of its 136 logs, at least 2 per log | 136 logs (1.98) | `--limit 1000` and `PADDOCK_NAVSIM_OVERRIDES=train_test_split.scene_filter.tokens=[...]` | a smoke of the same list only |
| neuroncap_quick | quick | stationary, frontal and side at 0103; 1 run (seed 0) | 3 pairs (4.30) | `--scene 0103 --runs 1`, `PADDOCK_NCAP_MANAGE_RENDERER=true` | a full sweep (its run 0) |
| neuroncap | closed | stationary/frontal/side at 0103, frontal+side at 0110, stationary at 0796; 3 runs | 6 pairs (2.57) | `--scene 0103,0110,0796 --runs 3`, `PADDOCK_NCAP_MANAGE_RENDERER=true` | a full sweep (its runs 0-2) |
| bench2drive | closed | 7 routes where the four unassisted systems' scores spread by sd >= 12 | 7 routes (2.45) | `--scene <7 route ids>` | a smoke baseline |
| alpasim | closed | `public_2601_first10` | 10 clips (2.26) | `scene_set` pinned in the derived config | a smoke baseline |

A subset's baseline is `[baseline.<subset>]` in the baseline manifest, else
`[baseline.<benchmark>]`: one full neuroNCAP sweep serves both neuroNCAP subsets.

A smoke run of a subset is stored `valid: false` with the reason in `notes`
(`--limit`, a few of 20 pairs, a few of the site's routes, a few clips of the
scene set), so it never reaches the leaderboard. Two smokes are their whole
benchmark and are ordinary valid runs: KITScenes (all 223 keyframes) and
AlpaSim (the whole pinned `public_2601_first10`).

## Running the smoke

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
  its first episode, after the model is ready, so nothing else launches while
  one is running: the closed tier runs one subset at a time. With one
  `--cores` (the default, `4-7`) everything runs one after another.
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
   node and is recorded as `model_env`, and the neuroNCAP `renderer_url` moved to
   `--renderer-port` (default 8700), so the renderers the smoke starts do not
   land on the site's port, where a long-lived `paddock renderer` usually is;
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

A day with a healthy NAS saves most of the open-loop time; the closed-loop
two are most of the cost either way. From those measurements, and not yet timed
as such: the **quick tier** in three lanes is ~40-45 min on that day, the three
open-loop subsets side by side (the longest took 35 min) and neuroNCAP's three
quick episodes (~20 min) in the first lane to come free; the **closed tier**, one subset at a time, is ~132 min of
neuroNCAP plus ~87 min of Bench2Drive on its 7 routes (they took 87 of the
240 min). The same run checked the gate against
itself: nuScenes 2.497 against the released run's 2.497 and neuroNCAP 3.175
against 3.175 on the shared seeds, both NEUTRAL to the last decimal.

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
- **nuScenes smoke is a prefix, not a stratified draw.** Open-loop benchmarks
  take no scene filter: `paddock run` hands `--scene` to closed-loop harnesses
  only (`src/paddock/runner.py:461` calls `_run_open_loop` with `limit` and no
  scenes), and `NuScenesBenchmark.samples`
  (`src/paddock/adapters/benchmarks/nuscenes/harness.py:223`) has no scene
  argument. `--limit` takes the first N samples in `val_scenes.json` order, so
  the 30 scenes are the first 30: boston-seaport and singapore-onenorth, no
  queenstown or hollandvillage. A stratified choice needs a scene filter there.
- **The AlpaSim smoke runs its pinned scene set as the default set**: the
  derived config sets `[benchmarks.alpasim] scene_set = "public_2601_first10"`,
  so a site configured for `public_2601_30scenes` still runs the subset's 10.
- **Drive-JEPA's AlpaSim numbers do not come from `paddock run`.** Its
  `model.yaml` does not declare alpasim; they come from
  `tools/models/drivejepa/alpasim_driver/`. The smoke skips AlpaSim for it
  until the bridge can feed it the inputs that driver does.
- **A checkpoint override does not invalidate a run on its own** until #82
  (`src/paddock/runner.py:91` records `model_env`, `_validity` does not read
  it). NAVSIM's `PADDOCK_NAVSIM_OVERRIDES` is recorded in `config.overrides` but
  is not one of its `invalid_reasons`
  (`src/paddock/adapters/benchmarks/navsim/harness.py:710`). Every smoke run is
  invalid through its subset regardless.
- **The neuroNCAP smoke starts its own renderers** on `--renderer-port`, one
  scene at a time. Two smokes at once need different `--renderer-port` (and
  `--base-port`) values; if the port is taken, neuroNCAP's preflight says so and
  that benchmark does not start.
- Smoke runs write their `run.json` into this checkout's `results/` like any
  run. They are evidence for a decision, not results to commit: the subset ones
  are `valid: false`, and the KITScenes and AlpaSim ones, valid, would add a
  second row for the model.
