# Adding a benchmark

Same shape as [adding a model](adding-a-model.md): one registry file, one new
directory, no changes to `src/paddock/`.

The one extra step is that a benchmark needs data, and data lives in a different
place at every site. So a benchmark is declared in two halves — **what it needs**
(tracked, identical everywhere) and **where that is** (each site's own
`paddock.toml`).

---

## What you will create and modify

| | Path | Action |
|---|---|---|
| 1 | `datasets.yaml` | **modify** — declare the dataset, with no paths |
| 2 | `paddock.toml` | **your machine only** — where it lives here. Never committed |
| 3 | `src/paddock/adapters/benchmarks/<name>/benchmark.yaml` | **create** — from `_template` |
| 4 | `src/paddock/adapters/benchmarks/<name>/harness.py` | **create** |
| 5 | `src/paddock/adapters/benchmarks/<name>/tests/test_harness.py` | **create** — golden test |
| 6 | `docs/benchmarks/<name>.md` | **create** |

If the benchmark uses a dataset that is already declared, skip step 1.

---

## Step 1 — declare the dataset

In [`datasets.yaml`](../datasets.yaml). This says what the dataset *is*, so
`paddock doctor` can tell a colleague what they are missing and what it blocks:

```yaml
datasets:
  mydataset:
    description: One line.
    requires: [images, annotations]     # subdirs that must exist under the root
    splits: [train, val, test]
    notes: >
      What a partial copy still supports, and what it does not. Written for
      someone who has a subset and needs to know whether their number is
      comparable to yours.
```

`requires` is checked by `doctor`; `notes` is shown when something is missing.
Both are worth real effort — they are the difference between "dataset
incomplete" and knowing which benchmarks that actually stops.

## Step 2 — point your machine at it

In your `paddock.toml` (gitignored):

```toml
[datasets.mydataset]
path = "/wherever/you/keep/it"
```

Or, if the data is spread across tiers of different speed:

```toml
[datasets.mydataset]
tiers = ["/fast/nvme/mydataset", "/bulk/archive/mydataset"]
```

Paddock resolves each entry against the first tier that has it, so a small
working set can front a large archive without duplicating it. See
[data-setup.md](data-setup.md).

## Step 3 — copy the template

```bash
cp -r src/paddock/adapters/benchmarks/_template src/paddock/adapters/benchmarks/mybench
```

## Step 4 — declare the benchmark

```yaml
name: mybench
description: What it measures, on what data.
mode: open_loop            # open_loop | closed_loop
dataset: mydataset
metrics: [ade, fde@3s]
primary_metric: ade
```

**Reuse canonical metric names.** `ade`, `fde@1s`, `fde@2s`, `fde@3s`,
`min_ade@6`, `ncap_score`, `collision_rate`, `pdms`, `epdms`. A new name should
mean a genuinely new quantity — otherwise the leaderboard grows two columns for
one thing. If you do add one, document it in
[results-schema.md](results-schema.md).

## Step 5 — implement the harness

### Which mode?

**Open loop** — the dataset decides what happens next. Paddock iterates samples,
calls `infer` on each, and scores against the recorded future. Errors do not
compound. Cheap, parallel, reproducible; measures imitation, not competence.

**Closed loop** — the model decides what happens next. A simulator drives the
policy, feeds back its own consequences, and scores the outcome. Errors compound,
which is the point. Expensive and serial; measures whether the model can
actually drive.

### Open loop

```python
class MyBenchmark(OpenLoopBenchmark):
    def preflight(self) -> list[str]: ...
    def samples(self, limit=None, split=None) -> Iterator[Sample]: ...
    def score(self, sample, prediction) -> dict[str, float]: ...

BENCHMARK = MyBenchmark
```

- **`samples` must be deterministic.** `--limit 5` has to select the same five
  samples at both sites, or two people comparing smoke tests reach different
  conclusions and neither is wrong. Sort explicitly; never rely on the order the
  filesystem returns.
- **`score` returns per-sample metrics, not aggregates.** Paddock aggregates, so
  every benchmark aggregates the same way and per-sample records stay
  inspectable.
- **Use `paddock.metrics`.** The shared ADE handles the alignment rule that
  catches people out: interpolate ground truth onto the prediction's timestamps,
  and never extrapolate past the end of the recorded future — trim instead. A
  hand-rolled ADE that extrapolates flatters long-horizon predictions.

### Closed loop

```python
class MyBenchmark(ClosedLoopBenchmark):
    def preflight(self) -> list[str]: ...
    def episodes(self, scenes=None, runs=1) -> Iterator[Episode]: ...
    def run_episode(self, episode, policy_url, out_dir) -> dict[str, float]: ...

BENCHMARK = MyBenchmark
```

Paddock owns the model process: it gates on VRAM, starts the node, health-checks
it, and tears it down. You own the simulator. `policy_url` is a node that is
already up and answering.

**Every model must see identical episodes and identical seeds.** Derive per-run
variation from the run index, not from a clock or an unseeded RNG. Paired
comparison is worth far more than averaging, and it is free if you plan for it.

## Step 6 — write `preflight()` first

It returns the reasons this benchmark cannot run here, cheaply, before any model
loads. Get it right before anything else: it is the difference between

```
the sweep died after fifty minutes
```

and

```
scenario stationary/0061 needs renderer checkpoint 0061, which is not present
```

Check the data root, the splits, and any per-scene assets the run will reach for
an hour in.

## Step 7 — golden test

Pin a handful of samples and their expected metric values, so a refactor that
changes the numbers fails loudly rather than quietly rewriting history. It must
run in paddock's environment with no GPU and no full dataset — commit a tiny
fixture if you need one.

## Step 8 — verify and document

```bash
paddock doctor
paddock run --bench mybench --model <any model> --limit 5
make check
```

Then write `docs/benchmarks/<name>.md`: what data is required, what the metrics
mean, roughly what a run costs in wall-clock, and reference numbers.
