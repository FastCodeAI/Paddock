# Drive-JEPA — Adapter Notes

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

[Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) (perception-based, NAVSIM
v2) reaches paddock two ways, and this page covers both because they pick a
plan the same way and differ in everything around it.

| | The paddock node | The AlpaSim driver |
|---|---|---|
| Where | `src/paddock/adapters/models/drivejepa/` | `tools/models/drivejepa/alpasim_driver/` |
| What it is | a `Policy` behind paddock's HTTP node | a native gRPC driver AlpaSim's runtime calls directly |
| Benchmarks | NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive | AlpaSim only, and **not through paddock** |
| Weights | the released checkpoint | **retrained** on Physical AI AV (adapter 0.1.7) |
| Cameras | four | one, front |
| Momentum-aware selection | off by default (`DRIVEJEPA_MOMENTUM`) | on by default |
| Rows in `results/` | every Drive-JEPA row | none |

The perception-free NAVSIM v1 agent is a third thing with its own page,
[drivejepa_free.md](drivejepa_free.md).

The rest of this page is about the AlpaSim driver at version 0.1.7, which is
the only place the retrained weights run. AlpaSim is a closed-loop test: the
model's plan steers a simulated car, and the next camera frame it sees depends
on where that plan took it. The first 3.0 s of each scene replay the recorded
drive (the warm-up); at the handover the model takes control. How to launch the
driver is in
[its README](../../tools/models/drivejepa/alpasim_driver/README.md).

---

## What the model reads

**One camera.** AlpaSim's `camera_front_wide_120fov`, pinhole intrinsics only;
an f-theta camera is refused rather than half-rectified. Two frames, 0.5 s
apart (the model's trained spacing; a pair outside ±10 % of that is a hard
failure, not a silent accept).

**Crop, resize, mask.** Each frame is cropped 28 rows top and bottom, resized to
512 × 256, and then **the bottom 37 rows are set to zero**. This is the bonnet
mask: the recording cars' own bonnet fills the bottom of every training frame
and the simulator never renders one, so those rows are blacked out on both
sides. The retrained weights were trained with the same 37 rows black on every
sample (37 is the highest bonnet edge measured across the training vehicles,
35, plus 2), so the mask is part of what 0.1.7 is, not a preference. The
released checkpoint never saw a masked row; do not run it behind this driver.

**`ego_status`, 11 numbers per frame:**

| Index | Content | 0.1.7 |
|---|---|---|
| 0–2 | pose relative to the current frame (x, y, heading) | as reported |
| 3–4 | velocity, in the car's own frame (x forward, y left) | as reported, except the handover sentinel below |
| 5–6 | acceleration | **zeroed** |
| 7–10 | driving command, one-hot (left, straight, right, unknown) | from the route |

Acceleration is **zeroed, not removed**: the two slots are still in the vector
and the model still reads them; they hold 0.0. The retraining set every sample's
acceleration to zero, so a non-zero value there is an input these weights have
never seen.

The same zero is used a second time, in the ego state the momentum re-score
simulates from (next section), so the model and the re-score agree about the
car's state.

**Batch fix.** Two lines of an upstream layer that builds a top-down map from
the camera features mix rows of a batch
(reference points laid out per batch against values laid out per example; one
row's camera mask used for all). `batch_fix.py` corrects both, plus a third
pairing found beside them, and is switched on for every forward. At batch 1 —
which is what a driver runs — it changes no tensor; it is on so that the car
runs the code path the generator was trained under. There is no switch to turn
it off.

## How a plan is picked

Drive-JEPA has two parts: a generator that proposes 32 candidate paths, and a
scorer that rates them and picks one.

1. The generator proposes **32 trajectories**, eight poses at 0.5 s (4 s), from
   a vocabulary of anchors — 10,240 for these weights (the released 8,192 plus
   2,048 faster shapes).
2. The scorer gives each proposal a score *S*.
3. **Momentum-aware selection** (momentum, for short: a step from the
   Drive-JEPA paper that favours proposals consistent with the plan already
   being driven). The driver hands the model the trajectory it chose at the
   previous replan. Upstream's own `calibrate_score` then simulates each
   proposal and the previous plan forward and checks two-frame comfort between
   them; the score becomes `(14·S + 2·S_c) / 16` with `S_c ∈ {0, 1}`. The
   argmax of that is driven.
4. **The 0.55 s rule.** A previous plan older than 0.55 s is dropped and the
   plain *S* is used, as upstream's evaluation loop does. The first replan of a
   rollout has no previous plan and also uses plain *S*.

This is the step [PR #59](https://github.com/FastCodeAI/Paddock/pull/59) put
into the paddock node; here it has been on since 0.1.4b. `DRIVEJEPA_MOMENTUM=off`
selects on plain *S*; an unknown value refuses to start.

## The 20 Hz divider and the re-anchor

AlpaSim's simulation clock runs at 20 Hz and asks the driver for a plan every
50 ms. Drive-JEPA is a 2 Hz model, so `divider.py` sits between them:

- **Replan every 0.5 s.** Real inference runs on a grid keyed on elapsed
  simulation time from the rollout's first request — not on a call counter,
  which drifts if a tick is skipped or repeated. Nine ticks in ten answer from
  the cached plan. With the 3.0 s warm-up the grid lands exactly on the
  handover tick.
- **Re-anchor on every tick.** A cached plan is never returned verbatim. The
  runtime and the controller each compose the plan with the car's pose at the
  plan's stated start time, and those agree only if that time is *now*; a
  450 ms-old plan handed back unchanged is displaced by the car's own motion
  over those 450 ms (about 11 m at highway speed). So every tick reports
  `model_t0 = now` and waypoints re-expressed in the car's own frame (x forward,
  y left) at that instant.
- **Handover sentinel.** During warm-up AlpaSim reports velocity as exactly
  zero while the poses are moving. If the reported velocity is zero and the
  last two poses imply more than 1.0 m/s, the planner substitutes the speed
  derived from the poses and logs a `handover sentinel` line. A genuine
  standstill keeps its zero.

## The provenance lines

The launcher prints two lines before it binds a port, so they open every run's
driver log:

```
provenance: adapter_version=0.1.7 version_id=drivejepa-v2-driver-0.1.7 planner_module=<path> imported_via=<path> divider_module=<path> replan_period_us=500000
drivejepa-0.1.7 | <weights label> | <selection label> | switch=on | previous plan dropped above 0.55 s | drive_jepa_model.py sha256=<hash> | momentum_module=<path>
```

| Field | Read it as |
|---|---|
| `adapter_version`, `version_id` | read from the imported module, not typed into the launcher; `version_id` is what AlpaSim's runtime records when it connects |
| `planner_module`, `divider_module`, `momentum_module` | the source files that actually ran. The launcher refuses to start if any resolves outside the driver's directory |
| `<weights label>` | a fixed string naming the weights and the three input rules: second-run generator with the first run's scorer, batch fix on, bonnet mask 37 rows, acceleration zeroed in the model input and in the re-score |
| `<selection label>` | a fixed string naming momentum-aware selection and the upstream function that does it, `calibrate_score` |
| `switch=on` | momentum-aware selection is active. `OFF (exact 0.1.4 selection)` otherwise |
| `drive_jepa_model.py sha256` | the upstream file holding `calibrate_score`, hashed from the checkout the worker imports |

The two labels are constants in `constants.py`, kept character for character as
the recorded runs printed them, because the scripts that checked those runs
match on them. They are written in the training notes' shorthand; the table
above is what they mean.

The worker adds one more line to its own log once the model is loaded:
`drivejepa-0.1.7 batch_fix: installed on 1 self-attention and 1 cross-attention modules; state {'fix1': True, 'fix2': True, 'fix3': True}`.

## Weights

| | |
|---|---|
| Location | `<weights>/drivejepa_physicalai/second-run-generator/step_000000002/drive_jepa_physicalai_generator.ckpt`, with its `checkpoint.json` beside it |
| Bytes | 1,273,002,536 |
| sha256 | `29d5f9517c2ea8789f35c51f0934493e0727bde2d40b573ecc3eb33f5922d916` |
| Layout | upstream's state dict (`agent._pad_model.*`), 495 tensors |

The worker checks size and sha256 before the 90-second load and refuses any
other file. Weights are never committed. The location above is the layout
gallops and paddock share; at the site that trained it the file still sits in
the training run's own output directory, with the sidecar beside it, so lay it
out as above when copying it.

Drive-JEPA has two parts: a generator that proposes 32 candidate paths, and a
scorer that rates them and picks one. Both were trained together in the first
training run. The scorer did not pass that run's own quality check, so it was
frozen for the second training run, which retrained only the generator. The
shipped checkpoint therefore contains the first run's scorer unchanged.

What is in the file, by part:

| Part | Tensors | Where it comes from |
|---|---|---|
| Encoder (V-JEPA 2 ViT-L) | 294 | the released checkpoint, bit for bit. Frozen in both training runs |
| Scorer | 30 | the first training run, bit for bit. Frozen in the second |
| Generator | 171 | the second training run, which started from the first run's generator |

Training, in the three sentences a reader of a score needs:

- **Data:** NVIDIA Physical AI AV, `train` split only — 126,981 samples from
  4,018 clips. The 30 AlpaSim evaluation scenes and NVIDIA's `test` split were
  excluded, and every script asserts it.
- **Vocabulary:** extended from 8,192 to 10,240 anchors, because the released
  vocabulary tops out near 15 m/s and highway scenes need more.
- **Spacing term:** the second run's code has a loss term that penalises
  unevenly spaced waypoints. It exists in the code but **was switched off
  (weight 0) for the shipped checkpoint**, so it did not shape these weights.

The training code, its step-by-step reports and the checkpoint sidecar are in
the gallops repository, under `experiments/drivejepa_retrain/`. Paddock holds
no training code.

`trained_on` for these weights is `physicalai_av` / `train`. `datasets.yaml`
has no entry for that dataset, so paddock cannot check the declaration against
a benchmark's scenes; the exclusion of the evaluation scenes is asserted in the
training code, not here.

## Version history

AlpaSim, `1cam-20Hz`, 30 scenes, one rollout per scene, renderer
`26.1.112-59673f88`. Numbers are read from the recorded runs' `totals.json` and
`models_compare.json`, at four decimals. An at-fault collision is one the
scoring attributes to the driven car: it ran into something, as opposed to
being hit from behind while driving normally.

| Version | `scene_score` ↑, 30 scenes | At-fault collisions ↓ | Any collision ↓ | Off-road ↓ |
|---|---:|---:|---:|---:|
| 0.1.4 | 0.4209 | 7 / 30 | 12 / 30 | — |
| 0.1.4b | 0.5433 | 4 / 30 | 10 / 30 | 4 / 30 |
| 0.1.7 | 0.7697 | 3 / 30 | 3 / 30 | 2 / 30 |

| Version | What it is | Weights |
|---|---|---|
| 0.1.4 | the first 20 Hz driver: it replans every 0.5 s and answers the 50 ms clock from its last plan in between | released |
| 0.1.4b | 0.1.4 plus momentum-aware selection, from the driver's own previous plan | released |
| 0.1.7 | 0.1.4b driving the retrained checkpoint, with what that checkpoint needs: bonnet mask (37 rows), acceleration zeroed in the model input and the re-score, batch fix | retrained, above |

For scale: the constant-velocity baseline (a car that keeps its current speed
and heading, with no model) scores 0.4876 on the same 30 scenes, and OpenPilot
scores 0.7726.

**0.1.7's three at-fault collisions are not a subset of 0.1.4b's four.** One
scene is in both lists; two of 0.1.7's are new. The mean rose and the collision
count fell by one, but a scene 0.1.4b drove cleanly can be one 0.1.7 does not.

*Footnote.* The README's AlpaSim Drive-JEPA row, 0.5030, is driver 0.1.4 on the
first 10 of these 30 scenes. Its 30-scene figure is 0.4209, in the table above.

## Not verified

- **One rollout per scene.** No repeat, no seed sweep, no interval on any
  number above.
- **The promoted code has not driven a scene since promotion.** The numbers
  were produced by this driver before its paths were moved to environment
  variables and its imports to paddock's scaffold. Since then it has passed its tests and a
  `--check-only` start against the real checkpoint (size, sha256, batch fix
  installed, both provenance lines) — and nothing more.
- **The 30 scenes are not the benchmark's set.** The benchmark's set is the
  first 10 of these 30, so a 30-scene mean is not comparable to any row in
  `docs/benchmarks/alpasim.md`; for driver 0.1.4 the two figures are 0.5030 and
  0.4209.
- **No `results/` row was added**, for the retrained weights or for 0.1.4b. The
  recorded runs were made by a native driver, not through `paddock run`, and
  this page is the only place their numbers appear.
- **Nothing here runs through the paddock bridge.** `drivejepa` still does not
  declare `alpasim` in its `model.yaml`.
- **The scorer's own check failed in training.** On held-out clips the
  scorer picks the truly best proposal 37.9 % of the time (released: 34.2 %),
  against a target of 50 %.
