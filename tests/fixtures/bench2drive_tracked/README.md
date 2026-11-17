# Tracked Bench2Drive records, and one made from raw routes in this repo

`tests/test_ingest_bench2drive_tracked.py` pools routes from a tracked record's
`upstream_scores.json` (`[[ingest.bench2drive]] tracked`). Every file here is real tool output,
copied without edits.

## `bench2drive/<model>/<run_id>/`: four records made at another site

These are the four 216-route Bench2Drive records pooled at site `a100-gpu-vm-002`. The raw
`results.json` behind them stayed at that site; only these files reached the repo. They are
copied byte for byte from `results/bench2drive/` at commit 9ba4b7e on `main` (checked with
`cmp`, 12 Nov 2026). The commit that added each one:

| Record | Added by |
|---|---|
| `alpamayo15/full220-836dda734659a86c-f81033` | eec3c4a (#172), 9 Nov 2026 |
| `autoe2e/full220-836dda734659a86c-63902e` | 381bde6 (#166), 9 Nov 2026 |
| `meteor/full220-836dda734659a86c-dda00e` | 5fc2aa2 (#164), 9 Nov 2026 |
| `openpilot/full220-836dda734659a86c-2803bd` | 6a6cfb7 (#163), 9 Nov 2026 |

Each holds `run.json`, `metrics.json`, and the `upstream_scores.json` that
`tools/benchmarks/bench2drive/upstream_scores.py` wrote at that site from the raw routes. The
tests check that each record's rows reproduce its own `metrics.json`, every block except
`duration_s`, which the rows do not hold. They are copies because this branch is based on
PR #167, which predates them on `main`. Once both are on one branch, the copies and the
originals must hold the same `upstream_scores.json` and `metrics.json`. Their `run.json` may
differ in what `paddock supersede` changes (`valid`, `superseded_by`, `notes`): #167
supersedes the originals by their 220-route pools, and the copies stay as the records were made.

## `five_routes_upstream_scores.json`: the five routes in `../bench2drive_upstream`

This file was written by `python -m tools.benchmarks.bench2drive.upstream_scores`'s
`scores()` and `dumps()` on 12 Nov 2026, at Bench2Drive 7ec25d1, on the record that
`bench2drive.ingest(<copy of ../bench2drive_upstream>, model="openpilot", site="s")` pools.
That pool is the same one `tests/test_bench2drive_upstream_scores.py` makes, with run id
`pooled-e8997da7f5e56aad-0b7a71`. The scorers ran in the site's CARLA-client interpreter,
with scipy 1.10.1 and networkx 3.1 added. Its pooled figures are the ones that
directory's README gives for Bench2Drive's own scripts: 2 successes and multi-ability mean
0.5. Its comfort figures are not the routes' own, because those `metric_info.json` files are
cut to 100 ticks.

Because its raw routes are in the repo, the test can pool the same five routes both ways and
compare. If a change to the ingest moves that pool's run id, the test says so; the
`record` field must then be regenerated the same way, since it can no longer be checked.
