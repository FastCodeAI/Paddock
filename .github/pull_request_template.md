## What

<!-- What this changes and why, written for someone who was not there: one plain sentence first, short sentences, every term defined, and what the reader has to do (if anything). Name the files a reviewer should open first. -->

## Verified

<!-- Claim, evidence, strength -- the way docs/ROADMAP.md records status. "Nothing beyond make check" is an honest row. -->

| Claim | Evidence | Strength |
|---|---|---|
| `make check` passes | exit 0, N passed / M skipped, with / without a site config | |
| | | |

## Not verified

<!-- What this PR asserts but did not run: an end-to-end route, a re-measured score, a flag checked against an upstream checkout. Leaving this empty is a claim. -->

## Checklist (CONTRIBUTING.md)

- [ ] no real machine paths in tracked files (the portability gate is part of `make check`)
- [ ] new upstream repos pin a sha in `repos.yaml`; new patches are listed there and apply at that sha
- [ ] a fixture or golden test that runs with no GPU and no dataset
- [ ] `docs/benchmarks/<name>.md` or `docs/models/<name>.md` exists for anything added
- [ ] every `[benchmarks.<name>]` key the code reads is in `paddock.toml.example` and the doc
- [ ] canonical metric names reused, or a new one documented with its direction
- [ ] ROADMAP, README and docs say what the diff does, not what it was going to do
