# Contributing to Paddock

Most people reading your code will be on a machine you have never seen, with
data laid out differently from yours, some months from now. Nearly every
convention here follows from that.

---

## Before you push

```bash
make check          # lint + tests + the portability gate
```

The same command runs on every PR in GitHub Actions
([`.github/workflows/check.yml`](.github/workflows/check.yml)), in a clean
environment with **no site `paddock.toml`** -- so a test that quietly needs the
author's site config fails there rather than at another site. Tests that
genuinely need a site skip in CI; the reviewer runs `make check` with the site
config to cover them. A second job checks that every patch under `patches/`
still applies at the sha `repos.yaml` pins (`python tools/ci/check_patches.py`
runs the same check locally).

CI runs no model, simulator or dataset. A number in `results/` is produced at a
site and reviewed by diff, and the mechanical part of that review is
`tests/test_results_store.py` and `tests/test_contracts.py`, both part of
`make check`.

## The one rule that is not negotiable

**No tracked file may contain a path that exists on only one machine.**

Where something lives goes in `paddock.toml`, which is gitignored. Tracked files
name locations through template variables — `{repo}`, `{repos}`, `{runs}`,
`{weights}`, `{data.NAME}`, `{data.NAME.tiers}`, `{site}` — expanded at launch by
`paddock.config`.

`tests/test_portability.py` enforces this. If it fails on your change, the fix is
almost always a template variable; the test has no exemption list. Placeholders
in prose (`/path/to/thing`) are fine; a real root from a real box is not.

Why so strict: a single leaked path works perfectly for whoever wrote it and
fails for everyone else, with an error naming a directory they have never heard
of. That failure is expensive and lands on the wrong person.

## Adding things

| I want to... | Read | Core changes needed |
|---|---|---|
| add a model | [docs/adding-a-model.md](docs/adding-a-model.md) | none |
| add a benchmark | [docs/adding-a-benchmark.md](docs/adding-a-benchmark.md) | none |
| add a dataset | [docs/data-setup.md](docs/data-setup.md) | none |

Adding any of these should touch one registry file and one new directory. **If
you find yourself editing `src/paddock/` to add a model or a benchmark, stop and
open an issue** — that is a gap in the framework, and patching around it in your
adapter hides the gap from the next person.

## Pinning

Every entry in `repos.yaml` pins a commit sha. A floating ref means two sites run
the same command and get different numbers with nothing to tell them apart. To
move to newer upstream, bump the sha in its own PR, so a change in results has a
commit that explains it.

## Results

`results/` is tracked, because a shared leaderboard is worth reviewing in a diff.
Only metrics and provenance go there — kilobytes. Predictions, frames and videos
go to `[roots.runs]`, outside the repo.

Every `run.json` records `site`, so numbers from different machines stay
distinguishable. Never hand-edit a metrics file: regenerate it, so its
provenance stays true.

## Code

- Match the surrounding code. It is more important that the file reads as one
  voice than that it matches any particular preference.
- Comments should explain **why**, especially where the reason is non-obvious and
  expensive to rediscover. `OMP_NUM_THREADS=1` looks like a performance tweak and
  is actually a deadlock fix; the comment saying so is worth more than the line.
- Keep model-specific dependencies out of paddock's own environment. Import them
  inside `load()`.
- Prefer failing loudly at setup over failing subtly at run time. `preflight()`
  and `doctor` exist for this.

## Review checklist

How to actually check these -- and what has slipped past a green `make check`
before -- is in [docs/reviewing.md](docs/reviewing.md). A reviewer will look for:

- [ ] `make check` passes, portability gate included
- [ ] no real machine paths in tracked files
- [ ] new repos pin a sha
- [ ] a fixture or golden test that runs with no GPU and no dataset
- [ ] `docs/models/<name>.md` or `docs/benchmarks/<name>.md` exists, its summary
  table written by `python -m tools.doc_summaries` (`make check` fails otherwise)
- [ ] canonical metric names reused, or a new one documented
- [ ] non-obvious environment requirements written down, not just encoded
- [ ] no changes to `src/paddock/` for a pure model/benchmark addition
