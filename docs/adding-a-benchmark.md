# Adding a benchmark

> **Start with `paddock new benchmark <name>`** — it scaffolds the directory below
> and renames the template consistently. Then `paddock verify --bench <name>`
> checks the contract before you spend a sweep on it. The shared mechanics and the
> full contract are in [adding-an-adapter.md](adding-an-adapter.md); this page is the
> benchmark-specific detail.

Same shape as [adding a model](adding-a-model.md): one registry file, one new
directory, no changes to `src/paddock/`.

Two base classes cover most benchmarks (`OpenLoopBenchmark`, `ClosedLoopBenchmark`);
a third, `ExternallyScoredBenchmark`, is for a scorer that owns its own iteration
and only needs a running node from paddock -- NAVSIM is the example. A *variant*
of an existing benchmark, the same scorer family on another devkit generation,
is still its own directory and name (its own results tree, site block and doc),
and subclasses the sibling's harness rather than copying it: `navsim_v1` next to
`navsim` shows the shape.

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
| 7 | `docs/datasets/<dataset>.md` | **create**, if the dataset is new to paddock |

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
pdm:                       # optional: which metric fills which PDM term
  ep: pdm_ep
  comfort: pdm_comfort
```

**`pdm`** maps NAVSIM v1's PDM terms (`nc`, `dac`, `ttc`, `ep`, `comfort`,
and `score` for a composite) to the declared metrics that measure them here. The
regression gate shows them in the same rows on every benchmark
([gate.md](gate.md#pdm-terms-beside-the-score)). Any open-loop benchmark can
fill EP and C: `paddock.metrics.pdm.plan_terms` gives `pdm_ep` and
`pdm_comfort` from the plan and its recorded future. Leave out a term the
benchmark cannot measure; never fill one with a constant. A term may be missing
on some samples, as long as which ones does not depend on what the model did
(NAVSIM's two-frame comfort has no value on a log's first frame). Do not use a
metric that is recorded only on some outcomes: neuroNCAP's `impact_speed`
exists only for runs that collided, so pairing it compares a subset chosen by
the result.

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
    def samples(self, limit=None, split=None, skip=()) -> Iterator[Sample]: ...
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
- **`skip` is what makes `--resume` worth having.** It holds the ids an
  interrupted run already scored. Dropping them from what you yield is correct
  but pointless; the win is not *building* them, as early as your data lets you
  decide — comma2k19 rebuilds a segment's ids from its frame count and never
  opens the video. The runner drops a repeat either way, so a benchmark whose
  samples are cheap may accept the parameter and ignore it.
- **Override `node_count` if a sample costs more to prepare than to infer.**
  Open-loop requests are self-contained, so the runner will score N at a time
  across N nodes, and decode-bound work scales with them. The default is one.

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

### `status()` — the other half of the same question

`preflight` says what is wrong. It cannot say what is *right*, and at a new site
that is the more useful half: "2035 segments, no decoder" tells someone far more
than the refusal alone. So `Benchmark` has a second, optional hook:

```python
def status(self) -> list[tuple[str, str]]:
    """Informational rows about what this benchmark's data actually holds."""
```

`paddock doctor` prints the pairs indented under the benchmark's row with no
status glyph and no fix arrow, because they are **facts, not verdicts** — the
moment a row implies a judgement it belongs in `preflight` instead. It defaults
to returning nothing, so this costs other adapters nothing.

Two rules make it safe to leave on. `doctor` calls it for every benchmark on
every machine, so it must stay interactive: walk directories if you must, but do
not read arrays, and sample rather than sweep. And it must not be the reason a
diagnosis fails — `doctor` catches what it raises and turns that into a row, but
a benchmark whose description throws has lost the point of the exercise. Reuse
whatever `preflight` already computed rather than paying for it twice.
`adapters/benchmarks/comma2k19/inventory.py` is the worked example.

### `close()` — let go of what the harness started

A harness that launches a process of its own (neuroNCAP's managed renderer)
stops it in `close()`. The runner calls it once the harness is done with,
whatever the outcome, and reports rather than raises anything it raises. It
matters when a pairing table splits a sweep into one harness per model: a
renderer the last harness left running is what the next one's `preflight`
finds answering, and refuses. A harness that starts nothing needs nothing here;
the default does nothing.

## Step 7 — golden test

Pin a handful of samples and their expected metric values, so a refactor that
changes the numbers fails loudly rather than quietly rewriting history. It must
run in paddock's environment with no GPU and no full dataset — commit a tiny
fixture if you need one.

## Step 8 (optional) — a renderer

If comparing models on this benchmark would be helped by a figure, add
`render.py` beside `harness.py`:

```python
def render(runs: list[LoadedRun], out_dir: Path, cfg) -> list[Path]:
    """Draw runs from any models together; return what was written."""
```

`paddock render --bench <name>` finds it. Each `LoadedRun` carries the run
record, its metrics, and its per-sample records including predictions.

Two things to get right:

- **Colour comes from `paddock.viz.palette`, keyed on the model name.** Never
  from list position, and never from a dict key you have folded a metric into --
  both make a model change colour between figures.
- **Check `sequences_sha256` before implying a comparison.** Runs over different
  sample sets are not comparable however close the counts look; `paddock render`
  warns, and a renderer should not paper over it.

Rendering belongs to the benchmark rather than to a model because debugging
means putting several models side by side. See
[architecture.md](architecture.md#where-does-a-file-belong).

### One sample, drawn

`render.py` may also define a second, independent hook:

```python
def render_sample(runs: list[LoadedRun], sample_id: str, out_dir: Path, cfg) -> list[Path]:
    """Draw the one sample `sample_id`; return what was written."""
```

`paddock render --bench <name> --sample <sample_id>` finds it, and says so
plainly if the benchmark does not define it. The two hooks answer different
questions and neither implies the other: `render` says *how often* a model is
wrong across a run, `render_sample` says *how* it was wrong on one — which is
the question a long tail in the first figure always raises next.

What makes this implementable is `native_id`. The runner stores it with every
per-sample record, and it is the benchmark's own handle on the sample — scene
id, frame index, whatever is needed to find the raw data again. Put enough in
it to re-decode the sample without re-deriving anything, and a stored run stays
inspectable months later. comma2k19 stores the route, the keyframe, the window
and the input rate, and re-decodes the video from that
([comma2k19/render.py](../src/paddock/adapters/benchmarks/comma2k19/render.py)).

Raise `LookupError` when the sample is not in the loaded runs, or when the raw
data behind it is not present at this site. The message is shown verbatim and
the CLI exits non-zero, so name what *is* available rather than only what is
not — the usual cause is a run whose artifacts under `[roots] runs` were cleaned
up, and a bare "not found" sends the reader looking in the wrong place.

## Step 9 — verify and document

```bash
paddock doctor
paddock run --bench mybench --model <any model> --limit 5
make check
```

Then write `docs/benchmarks/<name>.md`: what data is required, what the metrics
mean, roughly what a run costs in wall-clock, and reference numbers.

If the dataset itself is new to paddock, write `docs/datasets/<dataset>.md` as
well, and keep the line between them: the **benchmark** page documents paddock's
protocol — history length, horizons, what is scored — all of which are choices
that can change; the **dataset** page documents what is true of the data on any
machine — sensors and their limits, coordinate frames and units, coverage, the
traps in its own documentation, and what it cannot answer. Two rules make the
dataset page worth having rather than a second copy of the first:

* **Derive counts, do not assert them.** Any sample count a leaderboard row is
  conditioned on should be produced by a script against the harness's own
  methods, with the output pasted in, so it cannot drift from what a sweep does.
  [comma2k19](datasets/comma2k19.md#where-10165-comes-from) is the worked example.
* **Tracked means portable.** A machine-specific inventory belongs under
  `{roots.runs}` with the command that regenerates it, not in the repo;
  `tests/test_portability.py` will reject the paths otherwise.

[SKILLS.md](../SKILLS.md) is the companion to this page: not what to write, but
what went wrong writing it — vendored tools missing the flags you would reach
for, quaternion conventions that need verifying against the data rather than the
docs, and a `--dry-run` sample count that reads 0 for a new benchmark.
