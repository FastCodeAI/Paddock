# Adding a model

> **Start with `paddock new model <name>`** — it scaffolds the directory below
> and renames the template consistently. Then `paddock setup --model <name>` builds
> its environment, and `paddock verify --model <name>` checks the contract before
> you spend a sweep on it. The shared mechanics and the
> full contract are in [adding-an-adapter.md](adding-an-adapter.md); this page is the
> model-specific detail.

**You touch one registry file and one new directory, then add a docs page and a
README row. You do not edit anything under `src/paddock/` outside your own adapter.** If you cannot add a model
without changing the core, that is a framework bug — raise it, and fix it *as* a
framework change with its own reasoning, rather than working around it inside
the adapter. That has happened and is the intended outcome, not a violation of
the rule above: comma2k19 needed parallel open-loop nodes and an open-loop
`--resume` in `runner.py`, and a declared `ground_truth_dt` on
`OpenLoopBenchmark`, because a per-adapter copy of either would have been a
second scheduler and a second definition of the ground-truth grid. The rule the
sentence is defending is that the *core must not learn your adapter's name*.

Budget roughly half a day for a model whose repo already runs, most of it spent
on coordinate frames.

---

## What you will create and modify

| | Path | Action |
|---|---|---|
| 1 | `repos.yaml` | **modify** — add your upstream repo + pinned sha |
| 2 | `src/paddock/adapters/models/<name>/model.yaml` | **create** — from `_template` |
| 3 | `src/paddock/adapters/models/<name>/policy.py` | **create** — three methods |
| 4 | `src/paddock/adapters/models/<name>/setup.sh` | **create** — build the env |
| 5 | `src/paddock/adapters/models/<name>/tests/test_policy.py` | **create** — fixture test |
| 6 | `docs/models/<name>.md` | **create** — quirks and reference numbers; `python -m tools.doc_summaries` writes its summary table |
| 7 | `README.md` | **modify** — add a row to the models table, linking to the page in step 6; `make check` fails without it |
| 8 | `src/paddock/adapters/models/<name>/alpasim.yaml` | **AlpaSim only** — the settings AlpaSim drives the model on; see [AlpaSim settings](#alpasim-settings); needed by any model that reads a camera there |
| 9 | `paddock.toml` | **your machine only** — never committed |

Nothing else. In particular you do **not** register your model in any central
list: `paddock list` discovers it by finding `model.yaml`.

---

## Step 1 — register the repository

Add to [`repos.yaml`](../repos.yaml):

```yaml
repos:
  mymodel:
    url: https://github.com/org/mymodel.git
    sha: 1a2b3c4          # pin it — see below
    description: One line.
    patches: []           # only if the repo cannot be configured externally
```

**Pin the sha.** A floating `main` means two sites run the same command and get
different numbers, and nothing tells them. If you later need a newer upstream,
bump the sha in a PR, so the change to the numbers has a commit that explains it.

## Step 2 — clone it

```bash
paddock setup --repo mymodel
```

Clones into `[roots.repos]/mymodel`, checks out the pinned sha and applies any
listed patches. If a clone is already there it is **adopted, not re-cloned** —
your existing checkout and any local work survive. If it sits at another sha,
setup changes nothing, prints the checkout that would fix it, and exits 1.

## Step 3 — copy the template

```bash
cp -r src/paddock/adapters/models/_template src/paddock/adapters/models/mymodel
```

## Step 4 — declare the runtime

Edit `model.yaml`. This is where the model's environment stops being tribal
knowledge.

Use whatever the upstream repo actually needs -- `runtime.kind` may be `uv`,
`conda` or `shell`, and models genuinely differ (Drive-JEPA is conda on Python
3.9; openpilot is uv on 3.12). This is independent of how paddock's own
environment was built.

```yaml
name: mymodel
description: What it is and who publishes it.
repo: mymodel

runtime:
  kind: uv                       # uv | conda | shell -- whatever the repo needs
  python: "3.12"
  venv: "{repo}/.venv"           # the interpreter paddock launches
  pythonpath: ["{repo}"]
  cwd: "{repo}"
  env:
    MYMODEL_WEIGHTS: "{weights}/mymodel"

capabilities:
  cameras: [CAM_FRONT]           # only what infer() reads
  vram_gb: 24
  startup_timeout_s: 600
  waypoint_dt: 0.5
  planning_hz: 2.0               # closed loop: replan this often; omit to be asked every tick

benchmarks: [kitscenes, neuroncap]
```

**Use template variables, never real paths.** `{repo}`, `{repos}`, `{weights}`,
`{runs}`, `{data.NAME}`, `{data.NAME.tiers}`, `{site}` are expanded from the
site's `paddock.toml` at launch. `tests/test_portability.py` fails the build if
you write a home directory or a mount point here — that test is why this repo
still works at another site.

Four fields are worth more thought than they look:

- **`cameras`** — closed-loop wall-clock is dominated by rendering, and the
  simulator renders exactly what you declare. Declaring six when you read one
  makes every run several times slower.
- **`vram_gb`** — paddock waits for this much headroom before starting your
  model. Getting it right turns "CUDA OOM eight minutes into a checkpoint read"
  into "waiting for 24 GB (12 GB free)".
- **`env`** — the preferred way to point a repo at data. Most research repos
  already read an environment variable for their data root; find it and use it
  rather than patching source. Every key except the tuning ones below is
  recorded, as written, in `run.json` as `declared_env` and hashed into the run
  id, so changing one (a planner head, a switch) gives new runs a new id and
  `--resume` refuses to continue an old run under it.
- **`tuning_env`** — the `env` keys a site may override in `[models.<name>.env]`
  without changing what the model computes: thread counts, memory caps. A site
  override of any other key (a checkpoint, a horizon, a mode) is recorded in
  `run.json` as `model_env` and makes the run `valid: false`, so a finetune
  swapped in through a site config is never ranked under the model's name.
  The one exception is a key listed in `ranked_with_note`, below.
  OpenMP and BLAS thread counts, and keys that only say where a library or
  cache lives (`LD_LIBRARY_PATH`, `HF_HOME`, ...; `runner.LOCATION_ENV`), are
  always allowed. A checkpoint path is not: a moved checkpoint and a finetune
  look the same to the runner.
- **`env_passthrough`** — a list of var *names* (not values) to forward from
  whoever's shell launched `paddock run`, for ablation/debugging knobs your
  own code reads (e.g. `MYMODEL_TEMPERATURE=0.5 paddock run --model mymodel ...`).
  Without declaring a name here, `runtime/env.py` builds the subprocess's
  environment from a strict allowlist plus `env` above, so a var not in either
  place silently never reaches the model — the shell invocation looks like it
  worked and does nothing. A name set in the shell is held to the same rule
  as a site `env` override: unless it is in `tuning_env` or `ranked_with_note`,
  or restates the declared default, the run is recorded `valid: false`.
- **`score_notes`** (top level, beside `benchmarks`) — env keys that change what a
  score means, each with one line. `paddock report` prints the line under any
  ranked run whose `model_env` set the key to something other than the declared
  default, and `{value}` in the line becomes what the run set. Use it where a
  row would otherwise read as comparable when it is not: a plan stretched to fit
  a benchmark's horizon, one camera repeated into three slots. Leave out keys
  that change no number on the benchmarks you ran.
- **`ranked_with_note`** (top level) — the `score_notes` keys a site may set and
  stay ranked, with the note beside the row. List a fixed choice your model needs
  to run on a benchmark at all, such as METEOR's horizon on NAVSIM. Leave out any
  number that could be tuned to a benchmark's own errors, and never a checkpoint.
  A run that sets a noted key not listed here is stored `valid: false`, and its
  note prints only with `paddock report --include-invalid`.
- **`inputs`** (top level, optional) — `{benchmark: convention}`: inputs a benchmark
  builds your model's own way instead of the way it builds them for every model.
  Only a benchmark that offers conventions reads it, refuses one it does not know,
  and records the one used in the run's config. Today nuScenes offers `uniad` and
  `vad`: each keyframe as that repo's converter and dataset give it (the stored ego
  quaternion, each camera at its own ego pose, the CAN-bus message after the
  keyframe, the repo's command rule; docs/benchmarks/nuscenes.md). Leave it out
  unless your model was trained on one of those datasets' own inputs.

## Step 5 — write `setup.sh`

Build the environment from nothing, on a machine that has never run this model.
`paddock setup --model mymodel` runs it with `PADDOCK_REPO`, `PADDOCK_WEIGHTS`
and `PADDOCK_VENV` set from `paddock.toml`, so the script contains no absolute
paths.

Record anything non-obvious as a comment **and** in `docs/models/<name>.md`: a
pinned CUDA version, a package that must *not* be installed, a build needing a
compiler the OS does not ship. See [troubleshooting.md](troubleshooting.md) for
the ones that have already bitten us.

## Step 6 — implement the policy

```python
class MyModelPolicy(Policy):
    cameras = ("CAM_FRONT",)
    waypoint_dt = 0.5

    def load(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, obs: Observation) -> Trajectory: ...

POLICY = MyModelPolicy
```

The same class serves **both** loop types — open-loop harnesses call `infer`
directly, and `paddock.policy.server` wraps it in the HTTP node closed-loop
simulators drive. Writing it once is the point: the duplicated path is exactly
where camera handling silently diverges between the two.

Four rules that cause real bugs when broken:

1. **Ego frame, x forward, y LEFT, metres.** Not y-right. Not world frame. If
   your model outputs anything else, convert here, once.
2. **Waypoint k is at `(k+1) * waypoint_dt`.** No waypoint for the current
   position.
3. **Return at least 6 waypoints.** Closed-loop uses exactly the first 6 and
   indexes them positionally, so a 7th is dropped and a 5th raises.
4. **Never return zeros to mean "no plan".** An all-zero trajectory is a
   commanded emergency stop and scores as one. Use `constant_velocity(obs.speed)`.

Import the model's own package **inside `load()`**, not at module scope. This
file is imported by `paddock list` and by the test suite, which run in paddock's
environment where your model's dependencies do not exist.

## Step 7 — the fixture test

`tests/test_policy.py` must run in paddock's own environment: no GPU, no
dataset, no weights. It is what a reviewer can actually execute on your PR.

It cannot check that the model is good. It checks the contract — frame, shape,
finiteness, and that a moving ego plans ahead of itself rather than emergency
braking. The template's version is a working starting point.

## Step 8 — verify

```bash
paddock doctor --model mymodel        # interpreter present? repo at the pinned sha?
paddock run --model mymodel --bench kitscenes --limit 5
```

Run the constant-velocity template **before** wiring in your model. Getting the
plumbing green first makes coordinate-frame bugs obvious, because nothing else
is moving.

## AlpaSim settings

Only if your model lists `alpasim` in `benchmarks`. AlpaSim's bridge sends each
camera under AlpaSim's own id (`camera_front_wide_120fov`, ...), and a model node
keeps only the cameras its model declares, by name. So a model that reads any
camera needs a file `alpasim.yaml` beside `model.yaml` that renames AlpaSim's
cameras to its names. Without one, AlpaSim refuses the model before a dry run or
any model starts, naming the cameras that would not arrive. (Before this check
the model got no images and no error, and planned from poses alone.)

**What goes where.** A fact any benchmark could use stays in `model.yaml`: the
cameras the model reads, its replan rate (`planning_hz`), its waypoint spacing.
Only settings that mean something to AlpaSim alone go in `alpasim.yaml`. Do
**not** edit `adapters/benchmarks/alpasim/harness.py`: the harness finds the
file by the model's name, so the folder must be named after the model.

Four keys are required and one is optional:

| Key | What it is |
|---|---|
| `driver` | the driver config passed to AlpaSim as `driver=`: `paddock_bridge` is the shared one; your own goes in `<model>/driver/<driver>.yaml` |
| `bridge_args` | a list of extra `bridge.py` flags, as strings (quote numbers: `"50000"`); `--camera-name ALPASIM_ID=NAME` renames a camera, `--placeholder NAME` fills one AlpaSim does not render |
| `scenes_csv` | a file under `adapters/benchmarks/alpasim/scenes/` pinning each scene to one artifact, or `null` to name scenes by id |
| `replan_log` | `true` to have the bridge write a per-replan log beside the run's samples |
| `cameras` | optional: the cameras the model reads on AlpaSim, when fewer than `model.yaml` declares. `[]` for a model that reads no image there (constant velocity) |

A missing required key, an unknown key, or a value of the wrong type fails with
the file's path in the error; it never falls back to the default. AlpaSim also
refuses, before anything starts, a driver config that is neither in
`adapters/benchmarks/alpasim/configs/driver/` nor in `<model>/driver/` (or is in
both), a scene list that does not exist, and a camera the model reads that the
flags never rename or fill. A model that reads no camera and needs no other
setting can go without the file.

`adapters/models/drivejepa/alpasim.yaml` is a worked example, with a comment on
every flag. A driver config in `<model>/driver/` is found because AlpaSim's
config loader searches the model's folder after the benchmark's own configs; it
can extend the shared one with `defaults: [paddock_bridge, _self_]`. Its first
line must be `# @package _global_`, as in every driver config in the benchmark's
folder. Without it AlpaSim still loads the file, but puts its settings under
`driver.` where nothing reads them, so the run keeps the shared values with no
error. AlpaSim refuses a driver config in a model's folder without that line. The four
driver configs that existed before this convention stay in the benchmark's
folder until #94 replaces them with one shared setting.

This file is a convention: one `<benchmark>.yaml` in a model's folder, for a
benchmark that needs per-model settings. AlpaSim is the only benchmark that
reads one so far.

## Step 9 (optional) — probes

Diagnostics that need this model's internals belong in your own tooling, outside
paddock. Anything cheap enough to serve on request goes in `diagnostics()`, which
`paddock` exposes at `/diagnostics` on a running node.

What does **not** go here is a figure comparing this model with others.
Rendering is scoped to the benchmark, because debugging means putting several
models side by side and a model cannot draw that — it does not know about its
rivals. See [architecture.md](architecture.md#where-does-a-file-belong).

## Step 10 — document

Write `docs/models/<name>.md`: environment, weights and where they come from,
quirks, and the reference numbers your run produced. The next person to touch
this model is probably at another site.

To put the model in the README's support matrix, add it to
[docs/support-matrix.toml](support-matrix.toml): a `[[model]]` entry and a
`[cells.<name>]` line per benchmark naming your run. Then run
`python -m tools.readme_figures` and commit the redrawn picture. `make check`
fails if a named run is missing or the picture is out of date.

## Step 11 — open the PR

```bash
make check          # lint + tests + portability gate
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for what a reviewer looks at.

---

## When a repo resists configuration

Ranked by preference. Exhaust each before the next.

| | Mechanism | Use when |
|---|---|---|
| 1 | **Environment variable** | the repo already reads one — most do |
| 2 | **CLI flag** | the repo takes a `--data-root` or Hydra-style override |
| 3 | **Symlink** | a relative path is baked in, e.g. `<repo>/data/nuscenes` |
| 4 | **Source patch** | genuinely nothing else works |

A patch is a maintenance cost paid on every upstream bump, so it needs to earn
its place. When one is unavoidable:

```bash
cd $(paddock config | awk '/^repos/{print $2}')/mymodel
git diff > /path/to/paddock/patches/mymodel/0001-short-description.patch
```

Add it to the `patches:` list in `repos.yaml` and explain in the patch header
**why** external configuration was not possible. `paddock setup` applies patches
in order after checkout.

---

## When the environment resists you instead

This checklist covers what to write. [Troubleshooting](troubleshooting.md#bringing-up-a-new-model-or-benchmark) covers what goes
wrong doing it — `uv sync` failing on an extra you do not need, a `setup.sh`
missing a dependency the shared machinery assumes, a model that needs a dataset
your benchmark does not use — the failures that look nothing like a paddock bug
because they happen inside a model's own interpreter or a site's config.
