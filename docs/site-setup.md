# Setting up paddock on a new machine

Everything paddock needs to know about *your* machine lives in one gitignored
file: `paddock.toml`. Nothing else is site-specific, and nothing you configure
here will conflict with anyone else's setup.

Expect 15 minutes if the data and clones are already on the box, plus however
long each model's environment takes to build.

---

## 1. Install

Paddock is pure Python with no compiled dependencies, so venv and conda work
equally well. Pick whichever your team already uses.

**venv / uv** (the default):

```bash
git clone <paddock> && cd paddock
make setup
source .venv/bin/activate      # puts `paddock` on PATH for this shell
```

**conda**, if that is your house standard:

```bash
git clone <paddock> && cd paddock
make setup-conda
conda activate paddock
```

Either way, `make check`, `make test` and `make doctor` run against whichever
environment is active, so the rest of this repo does not care which you chose.
`make which` shows the interpreter they will use. (Conda `base` is deliberately
ignored -- installing a project into base is how base stops working.)

**`paddock` is not installed system-wide** -- it lives in that venv, so a shell that has not
activated it will report `command not found`. Three ways to run it:

| | When |
|---|---|
| activate the environment, then `paddock ...` | normal interactive use |
| `.venv/bin/paddock ...` | one-off, or from a cold shell |
| `make doctor` | no activation needed |

If you want it permanently on PATH without activating, `uv tool install -e .`
puts a `paddock` shim in `~/.local/bin`.

Paddock's own dependencies are deliberately thin — httpx, numpy, pydantic,
PyYAML, typer — and it never shares an environment with a model.

## 2. Configure

```bash
cp paddock.toml.example paddock.toml
$EDITOR paddock.toml
```

Three roots are required:

```toml
site = "myorg-myhost"       # recorded in every run.json; keep it stable

[roots]
repos   = "..."   # where vendor clones live, or should be cloned to
runs    = "..."   # run artifacts: predictions, frames, videos. Tens of GB.
weights = "..."   # model checkpoints
```

Then a `[datasets.*]` section for each dataset you have — see
[data-setup.md](data-setup.md). Leave out what you do not have; being incomplete
is not an error, and `doctor` will tell you what each gap blocks.

Two things worth getting right now rather than later:

- **`site`** distinguishes your numbers from the other organisation's in a shared
  results history. Do not leave it as `CHANGEME`.
- **`runs`** must not be on the root filesystem. A neuroNCAP sweep produces tens
  of gigabytes of frames, and filling `/` takes the machine down with it.
  `doctor` warns about this.

A setting meant for one model on one benchmark only belongs in a pairing table,
`[models.<model>.benchmarks.<benchmark>]`. Its `env` table applies over the
model's `[models.<model>.env]` and its `settings` table over
`[benchmarks.<benchmark>]`, for that pairing and no other. `run.json` records it
as it would the same values set globally, and `paddock run` names it before
anything starts, so `--dry-run` answers "is this site running stock?". A
misspelt key in the table, or a benchmark name that does not exist, stops the
run. Delete the tables to reset. `paddock.toml.example` shows one.

Set `PADDOCK_CONFIG=/path/to/other.toml` to use a different file, so one checkout
can serve several layouts.

## 3. Fetch the repositories

```bash
paddock setup
```

For every entry in [`repos.yaml`](../repos.yaml) this clones it into
`[roots.repos]` at the pinned sha, symlinks it into `external/`, and applies any
patches listed for it.

**Existing clones are adopted, not re-cloned.** If you already have these repos
checked out under `[roots.repos]`, paddock uses them where they are, and your
local work is left alone. It will warn if a clone sits at a different revision
than the pin, because that will change your numbers.

## 4. Check

```bash
paddock doctor
```

This is the command to run whenever anything is confusing. It reports:

| Section | What it checks |
|---|---|
| configuration | `paddock.toml` parses; `site` has been set |
| roots | the three roots exist, `runs` is writable and not on `/` |
| repos | each is cloned, and sits at the sha `repos.yaml` pins |
| datasets | each configured dataset resolves, with its required subdirs |
| models | each model's interpreter exists — i.e. `setup.sh` has been run |
| hardware | GPU headroom against the largest model's declared need |
| environment | `HF_HOME` is set and not pointing at the root filesystem |

Warnings are fine — they mean "this specific thing will not run". Failures mean
paddock cannot proceed at all.

## 5. Build the environments you need

Each model builds its own, and none of them are compatible with each other. Only
build the ones you intend to run:

```bash
bash src/paddock/adapters/models/<name>/setup.sh
```

`doctor` reports which interpreters are still missing. Per-model quirks — pinned
CUDA versions, packages that must *not* be installed — are in
`docs/models/<name>.md` and [troubleshooting.md](troubleshooting.md).

## 6. Smoke test

```bash
paddock list                                          # what is registered
paddock run --model <m> --bench kitscenes --limit 5   # a few samples
paddock report                                        # leaderboard
```

---

## Common problems

**`no site configuration at .../paddock.toml`** — step 2 was skipped.

**`unknown template variable {data.foo}`** — a tracked config refers to a dataset
you have not configured. Add `[datasets.foo]` to `paddock.toml`, or run a
benchmark that does not need it.

**A repo is at a different sha than the pin.** Your numbers will not match
anyone else's. Either check out the pinned revision, or — if the move is
deliberate — bump `repos.yaml` in a PR so the change is recorded.

**`interpreter missing`** — that model's `setup.sh` has not been run. It is a
warning, not a failure: every other model still works.

**GPU shows less free memory than expected.** Someone else is using the card.
Paddock waits for the headroom a model declares rather than failing partway
through a checkpoint read, so a run may sit waiting rather than erroring.
