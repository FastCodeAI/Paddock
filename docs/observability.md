# Observability: dissecting what a model actually did

A leaderboard row says *which* model is wrong. The comma2k19 viewer says *where*.
Neither says **why**, and the difference matters because the three explanations
for a bad number need three different fixes.

The four worst defects this repo has found were all silent contract violations
inside an adapter — `camera2ego = eye(4)` (a 90° frame error that published
ADE 38.8 m as a modelling result), a camera adapter leaking across scenes,
`traffic_convention` differing between loop types, and an `fde@5s` that was FDE
at 4 s. Every one ran to completion and produced a plausible number. A plan that
sits at the origin, a plan built by subsampling a 50-point output onto a 0.5 s
grid, and a plan warped from a badly-derived mounting all look the same in a BEV
panel.

So the instrument is not a UI. It is a per-step record of what each call was
handed, what it returned, the assumption it encodes, and the invariants it
asserts — from the model's own interpreter and from paddock's, joined under one
sample id.

```bash
paddock explain --model openpilot --bench comma2k19 --sample comma2k19/3/99c9.../16/519
paddock explain --model all --bench comma2k19 --scene 3/99c9.../16
paddock explain ... --capture full --format json
```

---

## The verdict: adapter, harness, or model?

Every check and every assumption carries an **attribution**, and
`paddock explain` closes its report with the split. This is the output the whole
exercise exists to produce.

| attribution | what it means | what to change |
|---|---|---|
| `adapter` | paddock got this wrong and paddock can fix it. Frame flips, mounting derivation, warp margins, history cadence, resampling onto the 0.5 s grid, the medoid/subsample choice, the alignment in `score`. | paddock |
| `harness` | the benchmark is asking something the pairing cannot support. A horizon past the model's native support, a rate below `min_input_hz`, a keyframe whose recorded future is 1 m long. | the question |
| `model` | the input was faithful, the output was read in the model's own frame, grid and units, and the plan is still wrong. | the model |

Only the last licenses a statement about model quality, and it is **the weakest
of the three claims, deliberately so**. A trace proves what the adapter did; of
the model it can only ever say that no adapter- or harness-side explanation was
found. The report prints that caveat rather than implying a conclusion:

```
verdict
  adapter  clean
  harness  clean
  model    clean
  Every invariant this trace could assert holds, so nothing here explains the
  number on the adapter or harness side. That is the weakest of the three claims:
  it says no other explanation was found, not that the model is at fault.
```

`docs/reviewing.md`'s standard applies: **separate what you verified from what
you argued.** An attribution is written by whoever annotated the step, and it is
reviewable precisely because it sits next to the check that produced it.

---

## The mechanism

Three pieces in `src/paddock/policy/trace.py`, which imports nothing from paddock
above its foot (see *Import rules* below).

### `@observed` — decorate what already exists

The adapters' private methods are **already** the right step boundaries:
`_bind_cameras`, `_repeats`, `plan_to_grid`, `build_extras`, `_letterbox`,
`_select_frames`, `resample_history`. Annotating them adds nothing inline and
restructures nothing — which is the whole reason for the decorator form. An
explainability layer that made each adapter longer would be paid for on every
read of a file that is already 1,000 lines.

```python
@observed(
    "plan_to_grid",
    why="supercombo emits 33 points on its own T_IDXS out to 10 s; paddock wants "
        "20 points 0.5 s apart. Interpolation, never extrapolation.",
    checks=("resample.is_interpolation_not_extrapolation",),
)
def plan_to_grid(plan, grid_s=GRID_S): ...
```

- `why` — the assumption this step encodes, in one sentence. Usually a frame
  convention, a unit, or what the model's native grid is.
- `checks` — names in `trace.CHECKS`.
- `capture` — parameter names (or `"return"`) to keep verbatim past
  `MAX_ELEMENTS`.
- `skip` — parameters to drop entirely, for a weights object or session handle
  whose `repr` is noise.

Apply it **under** `@staticmethod`/`@classmethod`, never over them: a
staticmethod object is not callable on 3.9, which is a live version here.

`trace.note(**facts)` records what a step knows and its signature does not show —
how many waypoints an alignment trimmed, which branch was taken, how many history
slots a stand-in filled. It is a no-op when nothing is listening.

### Off, it costs a context lookup

When no recorder is active the wrapper is one `ContextVar.get()` and a direct
call. Nothing is bound, summarised, checked or retained; `inspect.signature` is
built on the first *recorded* call, not at import.

**Measured at 57 ns/call** on this machine by
`tests/test_trace.py::test_an_untraced_call_costs_a_context_lookup`, which
prints the number and fails past 5 µs. That is what keeps annotation safe on the
10,165-sample sweeps: nothing about a published run changes, including its cost.

### What a step records

Summaries, never payloads.

| input | recorded |
|---|---|
| `np.ndarray` | shape, dtype, min/max/mean, count of non-finite, plus the values themselves under `MAX_ELEMENTS` (512) |
| image (HxWx3 uint8) | a base64 PNG thumbnail, long edge 256 px |
| `Trajectory` | n, dt, first/last waypoint, path length, which extras are present |
| `Observation` | cameras, timestamp, speed, yaw rate, **and the mounting angle off level per camera** |
| `Enum` | `Command.STRAIGHT`, not `2` |
| non-finite float | `"nan"` / `"inf"` — JSON has no spelling for these and a document only Python can read is not a record |
| anything else | `repr`, truncated to 200 chars |

A trace is capped at `BUDGET_BYTES` (1 MB). Past the cap, arrays and thumbnails
stop being captured and `truncated` is set — step names, timings and findings
keep accruing, because those are what the report is built from. Eviction is
first-come-first-served, so a truncated trace loses its *late* arrays; the scored
plan survives regardless, because `explain` writes `prediction` and
`ground_truth` at the document's top level, outside the step tree and outside the
budget.

**The budget counts the whole document, not just the arrays in it.** A comma2k19
keyframe replays a 160-frame history window: 321 steps and 672 findings, which is
300 KB of scaffolding before a single number. An estimate that charged only
captured arrays put that at 77 KB against a real megabyte, so the cap never
engaged — a budget that undercounts by thirteen times is not a budget.
`test_the_budget_counts_the_bytes_the_document_actually_costs` pins the estimate
against `json.dumps` of the real thing.

Two string tables keep the same document honest about repetition. A decorator's
`why` and a check's rationale are each a paragraph attached to something that
happens once per history frame, and written inline they were 40% of the file. So
a step carries `why_id` into the document's `whys` list — read it through
`trace.why_of(document, step)` — and a finding carries only its `check`, which
keys `rationales`. Findings are stored once, flattened at the top level with the
`uid` of the step that produced them, because 160 sibling `infer` steps all share
the path `infer_sequence/observe/infer` and only a uid tells them apart.
Together these took a comma2k19 openpilot document from 1.0 MB to 440 KB with
nothing lost.

### Checks

`trace.CHECKS` maps a name to a predicate, the reason it exists, and an
attribution. A predicate is handed the finished `Step` — raw inputs and output,
before summarising — and returns `(ok, detail)`, or **`None` to abstain**.

Abstaining is not a pass. A check that reports green on a step that does not
carry its subject is how a check quietly stops checking, so
`trajectory.turn_direction_matches_yaw_rate` says nothing at all when the car is
going straight, and `observation.mounting_is_level` *passes* rather than
abstaining when every camera is exactly level.

**Calibrate a new check against the recorded expert before trusting it.** These
checks issue an `adapter` verdict, and a false one sends someone hunting for a
bug that is not there. `trajectory.turn_direction_matches_yaw_rate` first
compared the measured yaw rate to the plan's *endpoint*, and on viewer10's 850
keyframes that failed the **human driver** on 16 of the 240 it fired on — every
one an S-bend, where a car finishing a left turn correctly plans a future that
ends far to the right. It duly returned a confident `adapter` verdict against
openpilot on a curve segment where openpilot was right. Comparing rate against
rate over the plan's first 1.5 s instead fails the human on none of 204 while
still catching a negated y on all 204. Running a candidate check over ground
truth costs one script and is the difference between evidence and noise.

The protocol-level set lives in `policy/checks.py` and needs nothing but the
protocol:

| check | attribution | the defect it names |
|---|---|---|
| `trajectory.not_a_commanded_stop` | adapter | zeros are a commanded brake, not "no plan" |
| `trajectory.moves_forward` | adapter | a frame error, or a model shown a static world |
| `trajectory.spacing_matches_speed` | adapter | a native grid under someone else's label — the `fde@5s`-that-was-4 s class |
| `trajectory.turn_direction_matches_yaw_rate` | adapter | a y-sign flip, which ADE cannot see because ADE is a magnitude |
| `observation.mounting_is_level` | harness | `camera2ego = eye(4)`, named as the convention error when the matrix is the identity |
| `observation.history_is_ordered` | harness | a mis-ordered window inside an accepted `min_input_hz` pairing |
| `image.is_not_blank` (warning) | adapter | a decode that returned nothing, or a warp sampling outside the source |
| `resample.is_interpolation_not_extrapolation` | harness | reading a model past its native horizon, then reporting the invention as its 3 s accuracy |

Model-specific checks (openpilot's warp margin, qwendrive's history cadence)
register beside the adapter that knows what they mean.

### Protocol steps, free for every model

`Policy.__init_subclass__` wraps each subclass's own `load`, `reset`, `observe`,
`infer` and `infer_sequence`. **A model with zero annotations still produces a
useful trace**: the per-frame skeleton, the timings, the observation summaries
and the protocol invariants on the plan it returned.

```
infer_sequence  0.79 ms
  reset     0.00 ms
  observe   0.47 ms          <- the default `observe` is a full forward pass,
    infer   0.07 ms             discarded; this is the cost an adapter overrides
  observe   0.14 ms             `observe` to escape
    infer   0.01 ms
  infer     0.01 ms   findings=4
```

An abstract method is left abstract, and a hand annotation wins: a method already
carrying `__paddock_step__` is not wrapped again, so an adapter that has
something to say about its own `infer` keeps it.

```python
from paddock.policy.protocol import INFER_CHECKS

@observed("infer", why="...", checks=(*INFER_CHECKS, "openpilot.warp_in_bounds"))
def infer(self, obs): ...
```

### `ASSUMPTIONS` — what an adapter takes on faith

"The model was never trained to output this format" is a sentence about a model
that no metric can produce. `Policy.ASSUMPTIONS` is where it is said, as data
rather than prose buried in a docstring:

```python
ASSUMPTIONS = (
    Assumption(
        what="the network emits 33 points on its own T_IDXS out to 10 s",
        source="supercombo model card, T_IDXS in common/transformations",
        step="plan_to_grid",
        attribution="adapter",
    ),
)
```

Empty by default, so an adapter that has declared nothing reads as **thin rather
than as complete**. The block rides back on `/infer_sequence` beside the trace
and is printed above the findings, which puts the argument beside the number it
explains.

---

## How it is turned on

`/infer_sequence` reads `body["trace"]`. When set, the **same**
`policy.infer_sequence(observations)` call is wrapped in a recorder and the
document rides back beside `trajectory`/`extras`. Same call, same code path, same
result — the flag only decides whether anyone is listening, and both branches
build their response through one function so that cannot drift.

This is not a second endpoint and not a side-channel file, because **the traced
run must be the scored run**. openpilot's `_bind_cameras` records what happens
when two paths exist for one decision: the ONNX branch had no two-camera case and
silently scored the narrow frame in the wide view.

It is also deliberately **not** a flag on `paddock run`. Published numbers stay
untouched; you point `paddock explain` at the handful of samples something else
raised a question about. Nothing here writes into `results/`.

---

## Reading a report

```
openpilot x comma2k19/3/99c9.../16/519
  ade=1.663, fde=3.201
  assumes [plan_to_grid] the network emits 33 points on its own T_IDXS out to 10 s
           source: supercombo model card, T_IDXS in common/transformations
  FAIL observation.mounting_is_level (harness) [x160, e.g. infer_sequence/observe]
       CAM_FRONT is 120.00 deg off level (limit 15); the matrix is the identity,
       so this is the convention error, not a mounting
       why: camera2ego is the single easiest thing to get wrong and the hardest
            to notice...
  12 inference step(s), 4 scoring step(s)
```

- **Failures first.** A report is read from the top and the top should be what is
  wrong.
- **`[x160, e.g. <step>]`** means the check fired identically on every frame of
  the history window. "On every frame" and "on one frame" are different defects,
  so the count is kept and one example step is named.
- **The step count is stated, not implied.** `0 inference step(s)` means the
  adapter is un-annotated, not that it did nothing.

Each dissection is also written to
`{roots.runs}/<bench>/_explain/<sample_slug>/<model>.json`, beside `_viewer/` and
`_compare/` — outside the repo, because a document with thumbnails in it is
hundreds of kilobytes and `results/` is the part that stays reviewable in a diff.

### Settings must match the run

`paddock explain` reads benchmark settings from the environment exactly as
`paddock run` does. For comma2k19 that means `PADDOCK_COMMA2K19_SCENE_SET`: the
active scene set decides which segments pool into a route's calibration, so
reaching outside it would build a differently-calibrated observation under the id
of a sample that was scored under another. The harness refuses rather than
answering, and names the variable.

---

## Annotating a new adapter

1. Nothing is required. The protocol steps already trace.
2. Add `ASSUMPTIONS` first — the native output grid, horizon and rate, the
   training frame and units, and what paddock does to reach the 0.5 s × 20 grid.
   That is the highest-value paragraph in the file and it usually already exists,
   scattered across docstrings and unreachable from the number it explains.
3. Decorate the seams that already exist, one model per commit. Do not create
   functions to decorate; if there is no natural boundary, that is worth noting
   rather than manufacturing.
4. Register model-specific checks beside the adapter, with an attribution and a
   `why` that names the defect rather than restating the rule.
5. Use `trace.note()` for the branch taken and the counts a signature hides.

## Import rules

`protocol.py` imports `trace.py`, so **`trace.py` must not import `protocol.py`**.
`Trajectory` and `Observation` are recognised structurally (`hasattr(value,
"waypoints") and hasattr(value, "dt")`) and findings are plain dicts shaped like
`conformance.Finding`. `trace.py` also has to import on Python 3.8/3.9, because
vendor interpreters load it — Drive-JEPA's is 3.9.

`checks.py` is imported from `trace.py`'s **foot**, so `CHECKS` can never be
empty: a registry that is only populated if somebody remembered to import the
module that fills it is a check that silently stops checking.
`tests/test_trace.py` parses the module and enforces both rules.

`trace.UNROTATED_CAMERA2EGO` restates `conformance.UNROTATED_CAMERA2EGO` for the
same reason, and `test_the_unrotated_frame_is_the_one_conformance_uses` pins the
two against each other so they cannot drift.

## Known limitations, stated rather than hidden

- **The `model` attribution is the weakest of the three.** `ASSUMPTIONS` is
  written by a human reading the upstream repo, so
  `resample.is_interpolation_not_extrapolation` checks a *claim* about the native
  grid rather than deriving one. Auditable, not derived.
- **Only annotated steps appear.** An un-annotated adapter's trace is honestly
  thin rather than misleadingly complete.
- **Thumbnails are lossy by design.** For a suspected sub-pixel warp defect use
  `--capture full`, which raises the budget to `FULL_BUDGET_BYTES` (16 MB) and
  keeps every array verbatim as `values` in the same JSON. The plan for this work
  sketched a side-car `.npz`; it was dropped because the numbers travel either
  way and a second file would add a path-passing contract for no gain. It is
  still a cap — a trace that exhausts a machine's memory is a worse outcome than
  one that says it was truncated.
- **Threads are not followed.** A `ContextVar` follows an async task and a plain
  call chain. A policy that farms work out to a thread pool loses the steps inside
  those threads — they do not appear, rather than appearing wrong.
