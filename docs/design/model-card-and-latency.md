# Design: the model card, and latency

**Status:** proposed. Nothing here is implemented.

Paddock can say how well a model drives. It cannot say **what the model is** —
how large, what it consumes, what it emits, what it was trained on — or **what
it costs to run**. Neither is recorded for any of the four model adapters in
the registry.

That was tolerable while every model was somebody else's frozen release. It
stops being tolerable for two reasons arriving at once: we are about to train
models of our own, and the destination for them is an automotive SoC, where size
and latency are not footnotes but the whole constraint.

This document proposes the smallest set of changes that closes both gaps. It is
one design because they share a mechanism: a model declaring facts about itself
in `model.yaml`, and paddock measuring the one fact a model cannot self-report.

---

## 1. What is missing today

**Static facts about a model live in prose, or nowhere.** Reconstructing the
table below took four `docs/models/*.md` files, and two cells are still
unknown:

| | params | MACs | precision | latency |
|---|---|---|---|---|
| Alpamayo 1.5 | 10 B | — | — | — |
| VisionPilot | 28.1 M | — | fp32 / int8 | 21 fps end-to-end (hand-run) |
| OpenPilot | — (60.9 MB ONNX) | — | — | — |
| Drive-JEPA | — (12 GB VRAM) | — | — | — |

None of it is in `results/`, so none of it can be compared, regressed against,
or diffed in a PR.

**The only timing paddock records is not model latency.** `inference_seconds`
is produced by NAVSIM's own inference loop
(`adapters/benchmarks/navsim/run_inference.py`). It is host wall-clock per
sample, including HTTP transport and image decode, and it exists for exactly one
of the five benchmarks.

**Nothing declares what a model was trained on.** Every number in the matrix is
zero-shot — `docs/benchmarks/nuscenes.md` says so explicitly — but that claim
lives in prose, and the moment we train on nuScenes it becomes false for one row
with nothing on the page to say which.

**And there is no provenance for the weights at all.** `RunRecord` carries
`repo_sha` ([store.py:65](../../src/paddock/results/store.py#L65)) with the
comment "Upstream revision the model was run from, where known" — and **no code
path sets it.** All 70 tracked `run.json` files carry `null`.
`docs/results-schema.md` documents a `model: {name, repo_sha, adapter_sha}`
shape that paddock has never emitted.

---

## 2. The model card

Extend `capabilities:` in `model.yaml`. The existing four keys each drive a real
decision — `cameras` decides what a simulator renders (the largest closed-loop
cost), `vram_gb` gates a launch, `min_input_hz` refuses a rate-mismatched
pairing, `waypoint_dt` sets the grid. **Every new key should meet that bar too**,
or it is documentation wearing a schema's clothes.

| field | what it drives |
|---|---|
| `params_m`, `macs_g` | the deployment column; a fixture test asserting declared against actual |
| `precision` | pairs an fp32 and an int8 adapter into a paired delta (§6) |
| `stochastic` | warns at `--runs 1`. This is the roadmap's own open item: alpamayo's frontal score has sd 1.817 on a single scenario, and a single draw from that is not a measurement |
| `output_kind` | `waypoints \| controls \| curvature_accel` — explains the `n/a` cells rather than leaving a reader to find the footnote |
| `horizon_s` | catches horizon mismatches. VisionPilot's `PLAN_HORIZON_S` 3.0 → 4.0 was worth real EPDMS, and nothing would have caught it |
| `history_frames` | KITScenes feeds 48 preceding frames and models consume different amounts of it; today that is invisible |
| `needs_speed` | records openpilot's hard blocker — its complete non-image input set is `{desire_pulse, traffic_convention, action_t}`, with no speed channel at all |
| `architecture` | one line, e.g. `"V-JEPA ViT-L encoder + BEVFormer"` |
| `trained_on` | list of `{dataset, split}` — §3 |
| `checkpoint` | path template to the weights directory — §4 |

Add the fields to `ModelSpec`
([registry.py:90](../../src/paddock/registry.py#L90)) and its loader, surface
them in `paddock list models`
([cli.py:60](../../src/paddock/cli.py#L60), which already prints
`runtime.kind`, `cameras` and `vram_gb`), and write them into `run.json`.

**Declared numbers rot.** Each adapter's own fixture test — already collected by
`make check`, since `testpaths` includes `src/paddock/adapters` — should assert
the declared parameter count against the artifact when one is present and skip
when it is not. That is the difference between a model card and a comment.

---

## 3. Contamination: a remark, not an invalidity

`invalid_reasons()` is the wrong home, for two reasons.

**Mechanically**, it is declared on `Benchmark`
([benchmark.py:103](../../src/paddock/benchmark.py#L103)), **takes no model
argument**, and the runner calls it bare. Contamination is a property of the
(model, dataset) *pair* — which the runner already holds. Expressing it there
means changing a base-class signature and every harness, for something the
runner can compute directly.

**Semantically**, and this matters more: `paddock report` drops `valid: false`
rows into a footer. Marking a contaminated run invalid would **delete our
headline number from the leaderboard entirely** — which is the opposite of
honest. A contaminated run is rankable and *differently interpretable*; those
are different axes.

Paddock already has vocabulary for the second: `remarks`, "things a reader of
the row should know that do not make it unrankable". One gap — `notes` is
rendered only for excluded runs, so a remark on a valid run is currently stored
and never shown. The report needs a small change to display it.

So:

- **same dataset, different split** (trained on navtrain, scored on navtest) —
  standard protocol. A remark, shown in the row.
- **same dataset and same split** — genuinely unrankable. `valid: false`, which
  keeps "one place decides valid" intact.

`trained_on` is a **self-declaration, not a measurement**, the same class of
claim as a pinned sha in `repos.yaml`. Nothing verifies it. Real verification is
sample-id overlap between a training manifest and what `sequences_sha256`
hashed, and it belongs in the training repo.

---

## 4. Latency, measured in the node

**Measure it in `policy/server.py`, not in a benchmark.** The node is the one
place that sees the true time inside `Policy.infer()` for every model and every
benchmark. Both endpoints already funnel through it —
`/infer` at [server.py:69](../../src/paddock/policy/server.py#L69) and
`/infer_sequence` at [server.py:93](../../src/paddock/policy/server.py#L93) —
and each does `Observation.from_wire(...)` then `policy.infer(...)`, so the
decode and the model are already separated by a line break.

Time three things and return them beside the trajectory:

| metric | what it is |
|---|---|
| `infer_seconds` | inside `Policy.infer()`, excluding transport and decode |
| `decode_seconds` | `Observation.from_wire` — image decode |
| `wire_seconds` | HTTP round trip, measured by the caller |

Because this lives in the node, **every existing benchmark gains latency with no
benchmark-adapter change at all.** It aggregates through
`MetricsRecord.from_samples`, whose `summarise()`
([store.py:93](../../src/paddock/results/store.py#L93)) already computes p90 —
the right statistic for latency.

### Two naming traps, both already recorded in the code

**A metric called `latency_ms` would render backwards.**
`_LOWER_IS_BETTER_MARKERS` in
[report.py:17](../../src/paddock/results/report.py#L17) matches by substring:
`("ade", "fde", "_mae", "_error", "_err", "_loss", "seconds")`. Neither
"latency" nor "ms" is in it, so a lower-is-better metric would sort and read as
higher-is-better. The comment two lines above records this exact bug happening
before — *"an earlier `_s` here matched ncap_score and rendered the headline
metric backwards"*. Naming the metrics with a `seconds` suffix makes them
correct with **no change to `report.py`**, which is why it is worth the slightly
clumsier unit.

**`infer_seconds` sits uncomfortably close to the existing
`inference_seconds`.** The schema doc says a new name should mean a genuinely
new quantity, not a second spelling — and these are genuinely different (one is
host wall-clock including transport, one is time in the model). Proposal: add
the three, and mark `inference_seconds` deprecated in
`docs/results-schema.md` once all three exist, since it is approximately their
sum. **This is the decision in this document most worth pushing back on.**

### State the caveat honestly

This box runs at 100% SM utilisation with ten resident processes and ~7.5 GB of
free VRAM. Host-side latency measured here is a **relative** signal — fp32
against int8, checkpoint against checkpoint, on the same box at the same time —
and not an absolute claim about any hardware.

That means `run.json`'s `hardware` block, currently free-form and holding only
`{"gpu_free_gb_at_start": ...}`, must carry the actual device and the contention
at start. Without it, two latency rows are not comparable, and a leaderboard
that sorts them is inviting a false conclusion.

---

## 5. On-target latency: ingest, not a fourth benchmark shape

Driving an R-Car board is not a benchmark, and trying to make it one would be
expensive.

Every structural assumption in the run path is localhost: `find_free` port,
`httpx.get("http://127.0.0.1:<port>/alive")`, `subprocess.Popen` of a local
interpreter, environment built from this machine's `PASSTHROUGH`, VRAM gating
via `nvidia-smi`. A board means a new transport, a deploy step, and a second
definition of "node" — a large, invasive change to the one component that is
currently small and correct, for a measurement whose units are comparable to
nothing else in `results/`.

It also would not fit the abstraction. The three benchmark shapes differ on *who
decides what happens next*, and all three produce per-sample driving-quality
scores aggregated over a sample set whose identity is `sequences_sha256`. A
board profile has no samples and no dataset; what varies between two of its rows
is the **hardware**, which is not a dimension the report has.

And `registry.load_benchmarks` hard-refuses any `mode` outside
`{open_loop, closed_loop}`, so a fourth shape starts by editing core anyway.

**Instead:** a board harness in the deployment repo emits a
`metrics.json`-shaped file, and `paddock ingest` reads it — which is exactly
what ingest exists for, "results produced elsewhere, normalised into the
canonical store". One reader, no new execution model.

If it ever must live inside paddock, it wants a separate verb (`paddock
profile`) writing a separate tree, because its rows are indexed by hardware
rather than by dataset.

---

## 6. Quantized accuracy needs no framework change at all

An int8 export scored on NAVSIM is an ordinary run of a **different model
adapter** — `<model>_int8` — on the same benchmark, over the same samples, with
the same `sequences_sha256`. Copy the adapter directory, point it at the
quantized artifact, declare `precision: int8`.

Paddock's paired-episode guarantee makes fp32 against int8 a **paired delta**
rather than two averages, which is far more sensitive. `viz/palette.py` colours
by model name, so the pair gets distinct colours in every comparison figure for
free.

This is the number an OEM argues about, and it is available today.

---

## 7. PR breakdown

| PR | Contents | Rough size |
|---|---|---|
| 1 | Card fields in `ModelSpec` + `_template/model.yaml` + `paddock list models` + docs. Backfill all four adapters from the numbers already in `docs/models/`. No behaviour change. | ~150 lines |
| 2 | Node-side timing, the three `*_seconds` metrics, `hardware` block populated, report column. | ~120 lines |
| 3 | `trained_on` and checkpoint identity into `_measurement_config`; remark display in `report.py`. | ~150 lines |
| 4 | Declared-vs-actual fixture test hook. | ~80 lines |
| later | Board-harness ingest reader. | ~120 lines |

### A live bug PR 3 also fixes

**`--resume` can silently merge two different checkpoints into one run.**
`_config_marker_problems` ([runner.py:178](../../src/paddock/runner.py#L178))
exists to refuse a resume whose configuration changed — its stated purpose is
that "a changed agent, switch or chunk size is a refusal rather than two
measurements merged under one id". But `_measurement_config`
([runner.py:63](../../src/paddock/runner.py#L63)) includes the caller's
switches, `model_spec.cameras`, `harness.run_config()` and the site
`[models.<name>] env` block — and **not** `model.yaml`'s own `runtime.env`. So
`ALPAMAYO_CHECKPOINT` can be repointed and a 12-hour NAVSIM resume will accept
it, producing one `run.json` whose scenarios came from two different models.

Putting checkpoint identity into `_measurement_config` closes it, and that one
dict feeds the run-id hash, `run.json["config"]` and the resume marker at once.

### Checkpoint identity, cheaply

Identity must be knowable **before the node starts** — the run id is computed
ahead of `proc.model_node` and names both the artifacts directory and the
`/whoami` nonce — so the node cannot supply it.

1. **A trainer-written sidecar** is the honest identity: only the trainer knows
   its step and its own commit, and a sidecar survives `rsync` where
   `(path, size, mtime)` does not. The gallops repo writes exactly this;
   see its `docs/checkpoints.md`.
2. **`(basename, size, mtime_ns)`** as a fallback when there is no sidecar.
   Record *which kind* of identity it is, so a reader knows what it is trusting.
3. **Never hash 20 GB per run.** Make a digest opt-in for published runs and
   cache it in the sidecar, so it is computed once per checkpoint.

---

## 8. Blast radius, and two traps

Touched: `registry.py` (fields), `runner.py` (`_measurement_config`,
`_store_run`, `_validity`), `policy/server.py` (timing), `report.py` (display),
`cli.py` (list), plus `_template/model.yaml` and docs.

**`RunRecord` does not change.** The new identity rides inside the existing
free-form `config` dict, which matters because `RunRecord(**run_raw)` re-parses
every historical file. No existing result is invalidated, no existing run id
moves, and a model declaring none of the new fields behaves byte-identically.

Two traps:

- **Put checkpoint *identity* into `run.json`, never the path.** `store.scrub()`
  and `tests/test_portability.py` ban two *different* lists of machine roots,
  and neither covers an arbitrary one — a weights root at, say, `/data2/...`
  would pass `make check` and ship a machine path inside a tracked result.
- **Latency instrumentation must not move a single score.** The verification
  below makes that explicit, because timing code that perturbs a measurement is
  worse than no timing code.

---

## 9. What this deliberately does not do

- **It does not make paddock train anything.** No dataloader, no `paddock
  train`. Training lives in the gallops repo; paddock only ever sees a
  checkpoint and an adapter. The temptation with an in-house model is to skip
  the subprocess and import directly — that would put torch in paddock's
  environment, break `make check` on a GPU-less machine, and destroy the
  property that makes an ADE and an NCAP score statements about the same system.
  **Our own model should be held at arm's length more strictly than the vendors',
  because it is the one we will be tempted to advantage.**
- **It does not verify `trained_on`.** Self-declaration only, and the doc should
  say so where a reader will see it.
- **It does not add a hardware axis to the report.** §5 keeps board numbers out
  of the dataset-indexed tree entirely.

---

## Verification

- `make check` green, portability gate included; check `$?` directly rather than
  piping to `tail`.
- `paddock list models` shows the card for all four models, backfilled from
  `docs/models/`.
- `paddock run --model visionpilot --bench kitscenes --limit 5` emits
  `infer_seconds` with a plausible p90, **and its ADE is byte-identical to the
  recorded value**. Timing must not perturb a score.
- Re-run one stored configuration: its `run_id` is unchanged for a model that
  declares no checkpoint, and changes when one is declared.
- Attempt a `--resume` with a repointed checkpoint and confirm it is now
  refused rather than accepted.
- Confirm a latency metric sorts the right way in `paddock report` — the trap in
  §4 is a rendering bug that no unit test on the metric itself would catch.
