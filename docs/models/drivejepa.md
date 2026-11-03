# Drive-JEPA — Adapter Notes

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<weights>` is `[roots.weights]`.

[Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) (perception-based, NAVSIM
v2) reaches paddock two ways, and this page covers both because they pick a
plan the same way and differ in everything around it.

| | The paddock node | The AlpaSim driver |
|---|---|---|
| Where | `src/paddock/adapters/models/drivejepa/` | outside this repo |
| What it is | a `Policy` behind paddock's HTTP node | a native gRPC driver AlpaSim's runtime calls directly |
| Benchmarks | NAVSIM, KITScenes, nuScenes, neuroNCAP, Bench2Drive, and AlpaSim through paddock's bridge | AlpaSim only, and **not through paddock** |
| Weights | the released checkpoint | the released checkpoint (versions 0.1.4 and 0.1.4b) |
| Cameras | four (on AlpaSim: one rendered, three placeholders) | one, front |
| Momentum-aware selection | on, declared in `model.yaml` (`DRIVEJEPA_MOMENTUM`); acts only in closed loop | on from 0.1.4b |
| Rows in `results/` | every Drive-JEPA row that was run through paddock | none (its AlpaSim rows are ingested) |

In the paddock node, momentum is declared on in `model.yaml`. The run id does not
yet record a value a `model.yaml` declares
([issue #99](https://github.com/FastCodeAI/Paddock/issues/99)), so a
closed-loop run started before momentum was declared on must not be continued
with `--resume` after it: the check `--resume` makes would pass, and the run
would mix episodes driven both ways under one id.

The perception-free NAVSIM v1 agent is a third thing with its own page,
[drivejepa_free.md](drivejepa_free.md).

How the paddock node drives AlpaSim, and how that compares with the native
driver it copies, is in [Through paddock on AlpaSim](#through-paddock-on-alpasim)
near the end. Apart from that section, this page is about the AlpaSim driver.
AlpaSim is a closed-loop test: the
model's plan steers a simulated car, and the next camera frame it sees depends
on where that plan took it. The first 3.0 s of each scene replay the recorded
drive (the warm-up); at the handover the model takes control.

---

## How a plan is picked

Drive-JEPA has two parts: a generator that proposes 32 candidate paths, and a
scorer that rates them and picks one.

1. The generator proposes **32 trajectories**, eight poses at 0.5 s (4 s), from
   a vocabulary of 8,192 anchors.
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
into the paddock node; here it has been on since 0.1.4b. The effect #59
measured for it on neuroNCAP was measured on a paddock node that could not see
the camera (next paragraph) and is to be re-measured; the 0.1.4b figure on this
page comes from this driver and does not depend on that.

**The paddock node was camera-blind on nuScenes-based benchmarks; this driver
is not.** The model reads the image at points written in the car's own frame
and projects them with each camera's matrix as if they were lidar coordinates.
The paddock node composed nuScenes' real lidar pose into that matrix, which put
every such point off the front image, so on neuroNCAP and nuScenes open-loop
the model planned from ego status alone
([PR #62](https://github.com/FastCodeAI/Paddock/pull/62) corrects it). This
driver has no lidar in the chain: AlpaSim's `rig_to_camera` is used as the
camera's pose in the car's frame, unchanged. With the front camera AlpaSim
declares, a point 10 m ahead projects to the image centre, a point to the left
lands left, and a point behind has negative depth. `DRIVEJEPA_MOMENTUM=off`
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

| Version | What it is | Weights |
|---|---|---|
| 0.1.4 | the first 20 Hz driver: it replans every 0.5 s and answers the 50 ms clock from its last plan in between | released |
| 0.1.4b | 0.1.4 plus momentum-aware selection, from the driver's own previous plan | released |

For scale: the constant-velocity baseline (a car that keeps its current speed
and heading, with no model) scores 0.4876 on the same 30 scenes, and OpenPilot
scores 0.7726.

**On the benchmark's own ten scenes.** The AlpaSim benchmark is the first 10
of these 30 scenes, and the README's rows are on those ten:

| Version | `scene_score` ↑, 10 scenes | `scene_score` ↑, 30 scenes | `results/alpasim/drivejepa/` |
|---|---:|---:|---|
| 0.1.4 | 0.5030 | 0.4209 | `legacy-drivejepa_014_20hz` |

## Through paddock on AlpaSim

`paddock run --bench alpasim --model drivejepa` drives the paddock node, with
the released checkpoint, through paddock's AlpaSim bridge. The bridge gives
Drive-JEPA its own settings, copied from the native driver at 0.1.4b, so the
two can be compared scene by scene: one front camera at 20 Hz (the node's
other three cameras get copies of it), a 3.0 s replayed warm-up, a replan every
0.5 s answered from the last plan in between, the same frame pair, speed,
acceleration and driving command, and the 30 scenes pinned to the same
artifacts. The settings live in the bridge (`harness.DRIVER_PROFILES`), not in
the model; [docs/benchmarks/alpasim.md](../benchmarks/alpasim.md#per-model-settings-in-the-bridge)
says how.

**What the bridge drives.** The node's `/infer` answer carries `plan`: all
eight (x, y, heading) poses the model returned, beside the six (x, y) points of
`trajectory` that neuroNCAP reads. The bridge drives `plan` on the model's own
headings, the way it reads every model's plan (a model that returns only x and
y gets headings from its path). Drive-JEPA's answer also reports its pick in
`extras`: which of the 32 proposals it chose, that proposal's score, and
whether momentum had a previous plan to compare with. The bridge logs these
per replan without interpreting them.

**A clean start per scene.** The bridge calls `/reset` at the start of every
scene, so the node's frame history and momentum's previous plan never carry
over from the previous scene.

**Result.** All 30 scenes in one run (the 22 driven first and the 8 that failed
to start while the disk was busy, re-driven with `--resume`; no scene was
driven twice), momentum on, made before the bridge sent the first frame to
`/observe` (see the first-plan item below), against the native 0.1.4b runs on the same scenes:

| Scenes | Through paddock | Native 0.1.4b | Difference | At-fault collisions | Off-road |
|---|---:|---:|---:|---|---|
| first 10 (the benchmark's default set) | 0.7573 | 0.7271 | +0.0302 | 0 / 1 | 1 / 1 |
| other 20 | 0.4166 | 0.4514 | −0.0348 | 3 / 3 | 5 / 3 |
| all 30 | 0.5302 | 0.5433 | −0.0132 | 3 / 4 | 6 / 4 |

Collisions and off-road are counts of scenes, through paddock / native.
Sixteen scenes score the same, eight higher and six lower. In every scene the
model's inputs (frame times, speed, acceleration, command) are identical to the
native driver's up to the handover; after it the two cars drift apart by a few
centimetres per second. The first pick that differs comes between 1.0 s and
16.5 s into the scene, and from there each run drives its own scene. Most of the large differences start at a
near-tie between two proposals, and a near-tie can go either way: 71d7d03b
(0.55 → 0, leaves the road) and 5f064b02 (0.65 → 1.00) both split on score
gaps of under 0.01.

**Known differences from the native driver**, none of them a difference in
what the model is given before the handover:

- **Runtime.** The node is a different process, built differently, on the
  same GPU. Re-run on recorded inputs, the same pick's score differs by up to
  0.004, so proposals that close can be picked either way.
- **The first plan's momentum: fixed by `/observe`, after the run in the
  table.** In that run, the bridge's first request of a scene only delivered a
  frame and its plan was thrown away, but momentum kept that plan, so the first
  real replan was compared with it where the native driver had no previous
  plan. In 8 of the 30 scenes the warm-up picks differed from then on (seven of
  them for this reason, one at a near-tie), and the score changed on two
  (cdd9448c, 0245ff75). The bridge now sends that frame to `/observe`, which
  records it and plans nothing. Replayed offline on three recorded scenes, 114
  of the 117 replans now match the native driver; the three that do not are on
  04394343, from the sideways difference below. The table has not been re-run
  since.
- **Sideways speed and acceleration are not passed.** The native driver gave
  the model and momentum the car's sideways speed and acceleration as well as
  forward ones. The node's input has forward only, so the model gets 0 for
  both. This matters only where the car moves sideways: on 04394343 the native
  driver was sent 0.78 m/s and 4.0 m/s² at the handover, the plans differ from
  the next replan and the picks a second after that; the native driver's car
  collided there (0) and this one did not (0.24). Passing them is a separate
  change.

## Not verified

- **One rollout per scene.** No repeat, no seed sweep, no interval on any
  number above.
- **A 30-scene mean and a ten-scene mean are not comparable.** `paddock run`
  drives the first 10 of these 30, and `docs/benchmarks/alpasim.md` has a table
  for each set; for driver 0.1.4 the two figures are 0.5030 and 0.4209.
- **The `results/` rows for 0.1.4 are ingested, not run through paddock.**
  Their runs were made by the native driver and read by paddock's AlpaSim
  ingester; the 30-scene figure is a row too, as the `…_30scenes` run. 0.1.4b
  has no row.
- **The run through paddock is not a row either, and is stored `valid: false`.**
  Momentum is switched on by a site setting (`DRIVEJEPA_MOMENTUM = "on"`), and a
  site setting that can change the plan makes a run invalid by rule. Its
  numbers are in [Through paddock on AlpaSim](#through-paddock-on-alpasim) as a
  comparison with the native driver, not as a ranked result.
- **One run through paddock.** As for the native rows, no repeat; given the
  near-ties above, a second run would not give the same per-scene scores.
