# Adding a model

**You touch one registry file and one new directory. You do not edit anything
under `src/paddock/` outside your own adapter.** If you cannot add a model
without changing the core, that is a framework bug — please open an issue.

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
| 6 | `docs/models/<name>.md` | **create** — quirks and reference numbers |
| 7 | `paddock.toml` | **your machine only** — never committed |

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

Clones into `[roots.repos]/mymodel` and symlinks it into `external/`. If a clone
is already there it is **adopted, not re-cloned** — your existing checkout and
any local work survive.

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

benchmarks: [kitscenes, neuroncap]
```

**Use template variables, never real paths.** `{repo}`, `{repos}`, `{weights}`,
`{runs}`, `{data.NAME}`, `{data.NAME.tiers}`, `{site}` are expanded from the
site's `paddock.toml` at launch. `tests/test_portability.py` fails the build if
you write a home directory or a mount point here — that test is why this repo
still works at the other site.

Three fields are worth more thought than they look:

- **`cameras`** — closed-loop wall-clock is dominated by rendering, and the
  simulator renders exactly what you declare. Declaring six when you read one
  makes every run several times slower.
- **`vram_gb`** — paddock waits for this much headroom before starting your
  model. Getting it right turns "CUDA OOM eight minutes into a checkpoint read"
  into "waiting for 24 GB (12 GB free)".
- **`env`** — the preferred way to point a repo at data. Most research repos
  already read an environment variable for their data root; find it and use it
  rather than patching source.

## Step 5 — write `setup.sh`

Build the environment from nothing, on a machine that has never run this model.
Paddock passes `PADDOCK_REPO`, `PADDOCK_WEIGHTS` and `PADDOCK_VENV` in, so the
script contains no absolute paths.

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

## Step 9 — document

Write `docs/models/<name>.md`: environment, weights and where they come from,
quirks, and the reference numbers your run produced. The next person to touch
this model is probably at the other site.

## Step 10 — open the PR

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
