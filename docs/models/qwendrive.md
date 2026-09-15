# Qwen-Drive 1.0 — Adapter Notes

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

## What it is

[Qwen-Drive 1.0](https://github.com/QwenLM/Qwen-Drive-1.0) is a Qwen3.5-4B
vision-language model with a 1 B flow-matching **planning expert** attached to
it, 5.58 B parameters loaded in bf16. The expert is not a head on top of a
pooled embedding: it cross-attends to the post-rotary K/V of the VLM's eight
`full_attention` layers. In `REASONING_PLANNING` the VLM first writes a
one-sentence rationale and the expert plans from *that* cache, so the text is
upstream of the trajectory rather than commentary on it. The rationale is
published per sample in `diagnostics()`.

Three inference modes ship: `VQA` (text only), `DIRECT_PLANNING` and
`REASONING_PLANNING`. Two planner heads ship: `planner-sft`, which serves both
planning modes, and `planner-rl`, which is reasoning-only. The adapter defaults
to `REASONING_PLANNING` + `planner-sft`.

Pinned in `repos.yaml` at `28091c1`. The checkout is imported as a library from
`<repos>/Qwen-Drive-1.0/src`; nothing is vendored and nothing is patched.
`scripts/demo.py` is *not* used — it hard-codes `attn_implementation="flash_attention_2"`,
which cannot be built here (below).

**What it was trained on is not stated.** The released card names no public
driving corpus and the checkout carries no training manifest, so
`model.yaml` records `trained_on: [{dataset: not-stated}]`. The useful claim
would be "none of paddock's benchmarks", and nothing in the checkout supports
it. The bundled demo scenes are in Waymo Open Dataset format, which is evidence
about the input format, not about the training set.

## Weights and environment

The snapshot directory must hold `config.json`, `model.safetensors` and the
`planner-sft/` head. Point `QWENDRIVE_WEIGHTS` at it. A site whose weights live
in a HuggingFace cache should name the revision outright in `paddock.toml`, so
a later download cannot quietly move what a recorded run was scored against.

`setup.sh` builds the venv from the repo's `requirements.txt` plus two things
that repository does not get right on its own:

**`causal-conv1d` must be compiled, with C++20 forced.** Qwen3.5's attention is
hybrid: `layer_types` is 3× `linear_attention` then 1× `full_attention`,
repeated 8× for 32 layers. The 24 linear layers are Gated DeltaNet and fall back
to a slow reference path unless `causal-conv1d` and `flash-linear-attention`
both resolve. torch 2.14's ATen headers require C++20 while the system gcc
defaults to C++17, so the standard has to be forced through `CXXFLAGS`. Verify
afterwards:

```python
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
m.is_fast_path_available   # must be True
```

Probing that by importing the submodule directly raises
`Could not import module 'modeling_qwen3_5'`. That is transformers' lazy-module
accessor, not a real failure.

**`flash-linear-attention` ships half a package.** As of 0.3.2 the project split
in two: `fla.layers` and `fla.models` stayed in `flash-linear-attention` while
`fla.modules` and `fla.ops` moved to a separate `fla-core` distribution. The
repo's `requirements.txt` names only the first, so a `--no-deps` install yields
a package that imports and then fails at
`from fla.modules import FusedRMSNormGated`, surfacing as
`ModuleNotFoundError: Could not import module 'Qwen3_5ForConditionalGeneration'`
— which looks exactly like a truncated download and is not one. The wheel sizes
give it away: 0.3.1 is 656 KB, 0.3.2 is 268 KB. Install **both**, both under
`--no-deps`.

**flash-attn is deliberately absent.** Two independent reasons, and the adapter
asks for `sdpa` instead:

1. The C++20 trick above does not work for it. flash-attn's `setup.py` puts
   `-std=c++17` in `extra_compile_args`, which distutils emits *after* the
   environment's `CXXFLAGS`, so the variable is silently overridden and the
   ATen C++20 errors come straight back.
2. flash-attn 2.8.3 does not compile against CUDA 13 regardless — its vector
   types hit the `double4`/`double4_16a` deprecations.

It is a speed optimisation for the eight full-attention layers. It changes no
result, so "fixing" the missing dependency buys nothing and breaks the install.

Settings, all in `model.yaml` `runtime.env`:

| variable | default | meaning |
|---|---|---|
| `QWENDRIVE_WEIGHTS` | `{weights}/Qwen-Drive-1.0-4B` | snapshot directory |
| `HF_HOME` | `{weights}` | keeps tokenizer cache lookups off the root filesystem |
| `QWENDRIVE_PLANNER` | `planner-sft` | a bare name resolves inside the snapshot; an absolute path is taken as-is |
| `QWENDRIVE_MODE` | `reasoning_planning` | `direct_planning` is the ablation |
| `QWENDRIVE_VIEWS` | `replicate` | `crop` is the ablation; see below |
| `QWENDRIVE_SEED` | `42` | passed on every call, so `stochastic: false` is a property of this file |
| `QWENDRIVE_SAMPLES` | `1` | flow-sampler draws; sample 0 is scored |
| `QWENDRIVE_ATTN` | `sdpa` | see above |

## Input contract and how paddock fills it

`DrivingScene` wants, per scored frame: three camera views at four timestamps,
16 ego poses at 10 Hz, an ego-status vector, and a navigation command.

**Conventions already match.** x forward, y **left**, heading positive for a
left turn, metres and radians. comma2k19's device frame is y-right, but
`adapters/benchmarks/comma2k19/loader.py` converts to the ego convention before
the `Observation` is built, so **the policy flips no signs**. A flip here would
mirror every plan and still look perfectly smooth — see
`docs/datasets/comma2k19.md` for why that is the easiest thing on this dataset
to get silently wrong. `tests/test_policy.py::test_a_left_turn_is_left_in_every_channel`
pins it: a left arc must show positive lateral position, negative past heading
and positive lateral acceleration, and the mirrored right arc the reverse.

**Three views from one camera.** `DrivingScene.__post_init__` **raises** on a
missing view, so `<FRONT VIEW>`, `<FRONT LEFT VIEW>` and `<FRONT RIGHT VIEW>`
must all be filled from whatever the source has. Two modes:

- `replicate` (default) — the true forward frame in all three slots.
  Geometrically honest, each view really is a forward camera, but it adds no
  lateral context.
- `crop` — overlapping 55 %-width thirds. Real lateral content, wrong
  extrinsics: a crop is not a rotated camera.

`replicate` wins, measured rather than argued: 1.47 city / 2.00 highway ADE
against `crop`'s 1.78 / 2.13 on the reference investigation's segments. The
model's learned cross-view geometry is load-bearing, and violating it costs more
than honest redundancy.

This is the opposite of the choice `alpamayo15/policy.py::slots_for` makes, and
that is not an inconsistency. Alpamayo tolerates a missing slot, so dropping one
says "there is no +55° view"; repeating the centre frame there would say "here
is one" and lie. Qwen-Drive does not tolerate it, so the choice is between two
lies and the measured one wins.

**History is resampled by timestamp, never by request count.** 16 poses at
10 Hz over 1.5 s, ego frame of the scored frame, last row exactly `(0, 0, 0)`;
camera timestamps at −1.5, −1.0, −0.5 and 0 s. comma2k19 feeds 20 Hz, and
differencing a 20 Hz history labelled 10 Hz would tell the model it is going
half speed. Ego state is by far this model's strongest input — the reference
ablation put a stationary history at ADE 15.79 city / 70.74 highway against a
1.47 / 2.00 baseline — so `diagnostics()` publishes the pose-derived speed
beside the recorded one, and a disagreement there is the first thing to look at
when a number will not line up.

**Ego status** is `[vx, vy, ax, ay]` plus a one-hot `[left, straight, right,
unknown]`. `vx` comes from the recorded odometer (`obs.speed`) resampled onto
the grid rather than from differentiated poses, so there is no differentiation
lag at `t = 0` — the frame the expert plans from. `vy` is the sideslip implied
by the pose history, `ax = d/dt smooth(vx)`, and `ay = vx · yaw_rate`, taken
from the recorded yaw rate because differentiating a 10 Hz heading manufactures
about 3 m/s² of lateral acceleration at 30 m/s. A source with no odometer falls
back to pose-derived speed and records `speed_source` in `diagnostics()`.

The smoother's edge handling is deliberate and is the one place a plausible
choice is wrong: replicating the endpoint flattens the filter's last samples
towards the last value, and here the last sample is `t = 0`. A car braking at a
steady 3 m/s² came out at 2. `smooth()` pads by **odd reflection**, which
continues the local slope through the boundary.

**No route.** comma2k19 ships no turn-by-turn signal, so every observation
carries `Command.STRAIGHT` and the model plans every sample under "GO STRAIGHT".
That is a real cost, not a formality: feeding the command a route planner *would*
have issued — read off the recorded future, hence diagnostic-only and never
scored — improved the reference's city ADE 1.47 → 1.28 and highway 2.00 → 1.77.
`diagnostics()` records `nav_command` so a residual on turns can be attributed
to a missing input rather than to planning. The mapping was cross-checked
against the released WOD scene files, where a left turn has future y = +11.98 m
and a right y = −11.0 m.

**Determinism.** The flow sampler draws sample *k* from a `torch.Generator`
seeded `seed + k`, and the reasoning turn is generated with `do_sample=False,
num_beams=1`. With `QWENDRIVE_SEED` fixed the model is reproducible, which is
why `model.yaml` declares `stochastic: false`. The seed is passed explicitly on
every call rather than inherited from whatever `noise_seed` the checkpoint
happens to ship.

**Why `num_samples: 1`.** Three flow-matching samples differed by a standard
deviation of only 0.09–0.14 m at t = 5 s in the reference measurements, so
best-of-3 barely improved on the first sample (1.75 vs 1.78 city) — and
best-of-N needs ground truth a deployed planner does not have. Raising
`QWENDRIVE_SAMPLES` costs latency roughly linearly and buys almost nothing;
any diversity-based use would need a different noise scale or step count.

## Output

50 waypoints `(x, y, heading)` — 5 s at 10 Hz in the scored frame's ego frame.
The adapter takes `(x, y)` of sample 0 and returns `Trajectory(xy, dt=0.1)`, so
`waypoint_dt: 0.1` and `horizon_s: 5.0` in `model.yaml`; the dashboard reads the
time axis from the registry, not from the array.

Everything else the model produces — the rationale, the nav command actually
used, the resolved view mode, the speed source and both speeds — rides in
`diagnostics()`. No benchmark reads them; they are there so a surprising number
has somewhere to be explained from.

## Running it

```bash
PADDOCK_REPO=<repos>/Qwen-Drive-1.0 bash src/paddock/adapters/models/qwendrive/setup.sh
paddock doctor --model qwendrive
paddock verify --model qwendrive --bench comma2k19
paddock run --model qwendrive --bench comma2k19 --limit 20
paddock run --model qwendrive --bench comma2k19
```

The card must hold `vram_gb: 18` per node — 11.2 GB of bf16 weights plus the
vision tower's activations for 12 images and the reasoning turn's KV cache.
`comma2k19/harness.py::node_count` caps parallel nodes by what the selected card
will actually hold, so pick the card first:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=1
```

`startup_timeout_s` is 1800: a 13 GB checkpoint read on a contended card is
slow, and failing at eight minutes into it teaches nothing.

## Reference measurements

These come from a standalone investigation under the checkout's own `.tmp/`,
**not** from a paddock run: three 60 s comma2k19 segments, 107 keyframes each at
2 Hz, `REASONING_PLANNING` + `planner-sft`, three samples, `replicate` views,
`nav = GO STRAIGHT`. They are recorded here as what the paddock sweep should be
checked against, and a large disagreement means paddock's feed differs from that
one.

| segment | v (m/s) | ADE | median | FDE@5s | heading MAE | ADE@1s | ADE@3s | ADE/distance |
|---|---|---|---|---|---|---|---|---|
| city, signals + left turn | 6.7 | 1.78 | 1.27 | 5.41 | 0.056 rad | 0.20 | 1.87 | 5.3 % |
| arterial, hard braking | 12.3 | 2.38 | 1.71 | 6.81 | 0.050 rad | 0.29 | 2.54 | 4.3 % |
| highway at night | 27.9 | 1.95 | 1.82 | 5.53 | 0.026 rad | 0.37 | 2.03 | 1.4 % |

Absolute error is roughly flat across a 4× speed range, so relative error falls
sharply with speed. The mean is well above the median in every segment: error is
concentrated in a few manoeuvres, and the six worst city keyframes are all the
same left turn, where the model's own rationale is correct ("turn left at the
intersection following the lead white SUV under the green traffic light") and
the plan under-turns because it was told to go straight.

Input ablations, same keyframes, one input perturbed at a time. "Shift" is the
mean distance from the baseline plan — it shows an input reaches the model at
all, independently of whether it helps:

| condition | city ADE | shift | highway ADE | shift |
|---|---|---|---|---|
| baseline | 1.47 | — | 2.00 | — |
| `direct` (no reasoning) | 1.55 | 0.21 | 2.04 | 0.19 |
| `nav_oracle` *(diagnostic, leaks GT)* | 1.28 | 0.53 | 1.77 | 0.30 |
| `crop_views` | 1.78 | 0.81 | 2.13 | 1.18 |
| `blank_cameras` | 1.97 | 1.74 | 3.07 | 1.94 |
| `still_history` | 15.79 | 15.18 | 70.74 | 69.93 |

The ranking is ego-state ≫ vision ≈ nav-command ≫ reasoning-mode. Grey images
cost +34 % ADE in the city and +54 % on the highway, which is the right
direction — at 28 m/s the lane geometry 140 m ahead can only come from the
image. On the four scenes bundled with the checkout, SDPA gave ADE 0.23–0.58 m,
which is what establishes that the environment is sound independently of any
comma2k19 code.

Timing: 1.43–1.49 s per scene on one H200 with SDPA, for 12 images plus the
reasoning turn plus 10 sampler steps × 3 samples.

One limitation worth carrying forward: VQA probes on real comma2k19 frames
showed solid scene *content* recognition (lead-vehicle identity, traffic-light
state, lane count, weather) but unreliable scene *handedness* — the model called
a green left-arrow signal a mandatory right turn. That reinforces the
missing-route reading above: the navigation command is doing work the vision
does not reliably supply on this imagery.
