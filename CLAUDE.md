# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make setup          # uv venv at .venv + editable install of [dev,serve,viz,comma2k19]
make setup-conda    # alternative: conda env named 'paddock'
make check          # lint + tests. The pre-PR gate — and exactly what CI runs.
make test           # pytest only
make lint           # ruff check + ruff format --check
make fmt            # ruff check --fix + ruff format
make typecheck      # mypy. ADVISORY: not in `make check`, never blocks a PR.
make which          # which interpreter the targets will use
```

Every target runs against the active venv/conda env, falling back to `.venv`.
Override with `make test PY=/path/to/python`.

A single test: `.venv/bin/python -m pytest src/paddock/adapters/benchmarks/neuroncap/tests/test_harness.py::test_scenarios_step_at_ten_hertz -q`.
`testpaths` is `tests` **and** `src/paddock/adapters`, so adapter tests under
`adapters/**/tests/` are collected by `make check` too. `tools/` and
`*/scenarios_10hz` are excluded from collection because older modules there
import torch/openpilot/mmcv, which paddock's own env deliberately lacks.

CLI (after `source .venv/bin/activate`, or as `.venv/bin/paddock`):

```bash
paddock doctor                                    # what this machine can actually run — start here
paddock config                                    # resolved site config
paddock list [models|benchmarks|datasets|repos]
paddock run --model openpilot --bench kitscenes --limit 5
paddock run --model all --bench neuroncap --runs 1 --scene 0103
paddock run --bench navsim --model alpamayo15 --resume <run_id>   # continue a run in place, keeping its id
paddock run ... --dry-run                         # preflight only
paddock renderer --scene 0103                     # foreground NeuRAD renderer; one serves a whole sweep
paddock render --bench kitscenes                  # cross-model comparison figures
paddock report [--include-invalid]                # leaderboard from results/
paddock ingest --all                              # import pre-paddock results
```

`paddock setup [--repo NAME | --model NAME] [--dry-run]` clones each `repos.yaml`
entry into `[roots.repos]` at its pinned sha (or adopts a clone already there),
then applies its patches; `--model` also runs that model's `setup.sh`. A clone at
another sha is reported and left alone, and the exit code is 1.

Set `PADDOCK_CONFIG` to use a config file other than `./paddock.toml`, and
`PADDOCK_BASE_PORT` to give concurrent runs disjoint port ranges (two runs
scanning from 9000 will collide).

## The invariant that shapes everything

**Tracked files say what a thing is; `paddock.toml` (gitignored) says where it
lives.** Tracked files name locations only through template variables — `{repo}`,
`{repos}`, `{runs}`, `{weights}`, `{data.NAME}`, `{data.NAME.tiers}`, `{site}` —
expanded at launch by `config.py`.

[tests/test_portability.py](tests/test_portability.py) fails the build if a
tracked file contains a real machine root — a home directory, a mount
point, or one of this workspace's own disks. (That test necessarily contains
the very patterns it bans, and exempts itself; do not restate them here or
this file joins the failure list.)
When it fails, the fix is almost always a template variable; the test has no
exemption list to add to. The `[tool.ruff.lint.per-file-ignores]` block in
`pyproject.toml` covers older code not yet brought up to the lint rules; it may
only shrink, so do not add to it.

## Architecture

Two axes meeting at one interface. Read [docs/architecture.md](docs/architecture.md)
for the full picture; the load-bearing parts:

**One `Policy` per model** ([policy/protocol.py](src/paddock/policy/protocol.py)):
`load()`, `reset()`, `infer(Observation) -> Trajectory`. The same object serves
open-loop scoring (called in-process) and closed-loop simulation (wrapped as an
HTTP node by [policy/server.py](src/paddock/policy/server.py), endpoints
`/alive`, `/infer`, `/reset`). This is why an ADE and an NCAP score are
statements about the same system — before paddock each model had two divergent
camera paths.

**Paddock never imports a model.** It is one Python 3.11 process that launches
each model's `Policy` in that model's *own* interpreter (uv, conda or shell,
per `model.yaml`), with `REPO_ROOT/src` on `PYTHONPATH` rather than installed.
Everything after that is localhost HTTP. Model deps (torch, tinygrad,
onnxruntime) belong in the model's env and must be imported inside `load()`.

**Open-loop harnesses send image *paths*, closed-loop simulators send pixels.**
The node reads the same disk; one KITScenes frame is 24 MB and a full run would
otherwise push 348 GB through a socket.

**Benchmark base classes** ([benchmark.py](src/paddock/benchmark.py)) split
three ways, all implementing `preflight()`, `run_config()` and
`invalid_reasons()`:
- `OpenLoopBenchmark` — `samples()`, `score()`
- `ClosedLoopBenchmark` — `episodes()`, `run_episode()`, plus
  `completed_metrics()` for `--resume`
- `ExternallyScoredBenchmark` — the external scorer owns the iteration (NAVSIM)

**Registry is discovery, not a list** ([registry.py](src/paddock/registry.py)):
models and benchmarks are found by walking `adapters/models/*/model.yaml` and
`adapters/benchmarks/*/benchmark.yaml`. Adding either touches **one registry
file and one new directory — never `src/paddock/` outside the new adapter** — plus
its docs page and a README row, which `make check` requires. If
you cannot, that is a framework bug worth raising rather than working around.
Copy from the `_template/` directory in each.

**Runtime** ([runtime/](src/paddock/runtime/)): `env.py` builds the per-model
environment (`PASSTHROUGH` is the one definition of what a subprocess inherits),
`gpu.py` waits for declared VRAM before launch instead of failing eight minutes
into a checkpoint read, `lease.py` makes the last check and the launch one step
per card across processes and users, `proc.py` starts/health-checks/tears down nodes. With
`gpu.serialize = true` exactly one model is resident at a time; the simulator's
renderer stays up across models.

**Results** ([results/](src/paddock/results/)): `results/<bench>/<model>/<run_id>/{run,metrics}.json`
is tracked (kilobytes, reviewable in a diff); per-sample records and artifacts go
to `{roots.runs}` outside the repo. `run_id` is `<UTC timestamp>-<short config hash>`.
`_validity()` in [runner.py](src/paddock/runner.py) is the single place that
decides `valid` — a subset, an overriding env, or a failure makes a run
`valid: false` with the reason in `notes`, and `paddock report` excludes it.
**Never hand-edit a metrics or run file; regenerate it** so provenance stays true.

**Rendering is benchmark-scoped, not model-scoped** — debugging means putting
models side by side, and a model cannot draw that. `viz/` knows neither models
nor benchmarks; `adapters/benchmarks/<b>/render.py` takes runs from any model;
`adapters/models/<m>/` knows only its own internals. `tools/` mirrors the same
split. `viz/palette.py` fixes colour per **model name** — folding a metric into
a dict key silently makes colour a function of the number.

## Conventions

- `Trajectory` is ego-frame, **x forward, y left**, metres, evenly spaced by
  `waypoint_dt` starting at `t = waypoint_dt`.
- neuroNCAP wants **exactly six waypoints** — a seventh is dropped, a fifth
  raises `IndexError`.
- An **all-zero trajectory is a commanded emergency stop**, not "no plan". A
  model that has not warmed up returns `constant_velocity(obs.speed)`; zeros
  score as a deliberate brake, which looks like a good result for the wrong reason.
- `min_input_hz` in `model.yaml` is load-bearing: models that infer ego motion
  from frame-to-frame change (openpilot) read a 2 Hz dataset as a parked car,
  so the harness refuses the pairing rather than recording the artefact.
- Declare only the `cameras` that `infer()` reads — closed-loop wall clock is
  dominated by rendering exactly what you declare.
- Contracts are imported, never restated: `Command`/`CANBUS_*` and
  `decode_image` from `policy.protocol`, `Trajectory` validation, `runtime.env.PASSTHROUGH`,
  `BenchmarkOverride.get()`. A client running inside a foreign interpreter can
  still `import paddock.*`; if it cannot, fix the env, not by copying constants.
- Every entry in `repos.yaml` pins a sha. Bump it in its own PR so a change in
  numbers has a commit that explains it.
- Prefer failing loudly in `preflight()`/`doctor` over failing subtly mid-run.
- Comments explain **why**, especially where the reason is expensive to
  rediscover (`OMP_NUM_THREADS=1` is a deadlock fix, not a perf tweak).
- Match the surrounding file's voice over any global preference.

## Before opening a PR

`make check` must pass, portability gate included; check `$?` directly rather
than piping to `tail`. CI (`.github/workflows/check.yml`) runs that same target
on every PR, in a clean environment with no site `paddock.toml`, alongside a job
proving every patch still applies at its pinned sha. So a green `make check`
locally and a green PR are the same claim by construction — keep it that way:
**anything that can reject work belongs in `make check`**, never in the
workflow's own command list.

`make typecheck` (mypy) is the deliberate exception: it is advisory, absent from
`make check`, and its CI job is `continue-on-error`, because older code was brought in
without type annotations, and a check that reddens unrelated PRs is one people
learn to ignore. Do not add it to the required checks; see `[tool.mypy]` in
`pyproject.toml` for the full reasoning and what would have to happen first. The reviewer checklist is in
[CONTRIBUTING.md](CONTRIBUTING.md) and how to actually verify each item is in
[docs/reviewing.md](docs/reviewing.md) — including the standard that reviews and
`docs/ROADMAP.md` share: **separate what you verified from what you argued**,
with the strength of each claim stated.

**Write the PR title and description so the user can read them.** Say what the
PR does in one plain sentence first. Use plain words and short sentences, define
every term the first time it appears, use no labels made up during your session,
and say what the reader has to do. The same rules cover reviews and replies; see
"Write it so the user can read it" in
[.claude/commands/pr-review.md](.claude/commands/pr-review.md).
