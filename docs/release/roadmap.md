# Paddock release roadmap

*Status 29 October 2026. Three releases; the first two are dated.*

| Release | Date | What ships | Why it matters |
|---|---|---|---|
| **v1.0 · Clean release** | **Fri 20 Nov 2026** | the codebase cleaned for a public repo; every adapter behind the support matrix tested; the matrix regenerated from those runs | a release anyone can install, run and reproduce |
| **v1.1 · Failure-mode analysis** | **Fri 11 Dec 2026** | one standard failure analysis for every model × dataset | Paddock's main value: *why* a model fails and what to change, not only a score |
| **v1.2 · Regression gate for fine-tuned models** | later (proposed: Q1 2027) | a supported verdict on whether a fine-tuned checkpoint beats its base | "did training help?" answered paired and honestly |

Before v1.0, on **Mon 3 Nov**, the repository goes public. The cleanup in #121 lands
first.

---

## v1.0 · Clean release, Friday 20 November

**What it is:**
- the Paddock codebase, cleaned up;
- every model and benchmark adapter in the support matrix, tested;
- every matrix cell regenerated from those tested adapters.

No new features.

**Done when:**
1. **Clean.** Three release PRs land:
   - #121 moves research scripts, notebooks and dead code to the private `paddock-lab`;
   - a second PR gives every model and benchmark page one layout;
   - a third rewrites the README and makes `paddock setup` work.

   CI is green (#125).
2. **Tested.** At the release commit, each model adapter passes `paddock verify` and a
   smoke run on every benchmark in its matrix row.
3. **Current.** Every matrix cell is a valid run of the released adapter code:
   - no stale (S) cells;
   - no typed-in value where a run can stand in;
   - the pending result PRs landed.
4. **Released.** Version 1.0.0 tagged (`pyproject.toml` says 0.1.0 today), with release
   notes.

**To wrap up by 20 November:**

| Item | Today | To do | PR |
|---|---|---|---|
| Repo cleanup | open | review; create the private `paddock-lab`; land before 3 Nov | #121 |
| One page layout; README and `paddock setup` | not opened | the two follow-up release PRs | — |
| CI on Ubuntu 24.04 | open | land | #125 |
| METEOR on Bench2Drive | 1.05, valid, R | land | #126 |
| Qwen-Drive on AlpaSim | 0.7001, valid | land; adds `Observation.scene_id` | #123 |
| AutoE2E on NAVSIM v1 | 0.538, valid, A | land | #122 |
| AutoE2E on Bench2Drive | 15.30, valid | land | #96 |
| Bench2Drive's native-rate cells | 3 cells unranked (R) | land after review: it sets `valid: true` by hand, where CLAUDE.md asks for a regenerated record | #124 |
| Stale cells (S) | openpilot comma2k19, Drive-JEPA nuScenes, the VisionPilot row | re-run at the release commit; VisionPilot after #57 is re-measured | #57 |
| Typed values | openpilot Bench2Drive 30.09; constant velocity KITScenes 1.381 | replace with runs; openpilot's own 50-route run is in #124 | #124 |
| Empty cells | METEOR comma2k19; nuPlan for Alpamayo, Qwen-Drive, AutoE2E | run them, or mark "not supported" with the reason | — |

**Decisions needed:**
- **HUGSIM.** Give it a column in v1.0 (2 of 8 models have driven it), or keep it out of
  the matrix until v1.1?
- **nuPlan.** It is unranked: 25 anchors from 1 of its 9 camera shards. Keep the column,
  or drop it from v1.0?
- **WA-JEPA.** Merged in #83, smoke run only. Add a row, or leave it out of v1.0?

## Support matrix: current status, 29 October

Read from `docs/support-matrix.toml` and the runs it names, ranked by the same code that
draws the README picture.

**How to read the tables**
- **Bold** is the best ranked cell in its column.
- *Italic · #n* is a result waiting in pull request #n.
- **Marks:**
  - **A** adapted to the benchmark;
  - **P** a pipeline around the model is graded;
  - **S** stale, owed a re-run;
  - **R** run differently, not ranked.
- A mark on a model's name applies to its whole row.

**Open loop and NAVSIM**

| Model | nuScenes L2 ↓ | KITScenes ADE ↓ | comma2k19 ADE@3s ↓ | nuPlan ADE ↓ ¹ | NAVSIM v2 ↑ | NAVSIM v1 ↑ |
|---|---|---|---|---|---|---|
| OpenPilot | 2.477 | 1.248 <sup>A</sup> | 1.507 <sup>S</sup> | 0.807 | 0.747 ² | 0.807 ² |
| Alpamayo 1.5 | 0.787 <sup>A</sup> | 1.002 | **0.900** | — | 0.691 <sup>A</sup> ² | 0.753 <sup>A</sup> ² |
| Drive-JEPA | 0.912 <sup>S</sup> | 2.364 | 16.048 ³ | 1.512 <sup>A</sup> | **0.843** <sup>A</sup> | **0.892** <sup>A</sup> |
| VisionPilot <sup>P S</sup> | 1.441 | 1.534 | 4.045 | 2.246 | 0.431 | 0.488 |
| METEOR | 0.977 | 1.879 | — | 1.273 | 0.303 <sup>P</sup> | 0.366 <sup>P</sup> |
| Qwen-Drive 1.0 | **0.450** <sup>A</sup> | **0.806** | 1.159 | — | 0.840 <sup>A</sup> | 0.876 <sup>A</sup> |
| AutoE2E | 0.839 <sup>A</sup> | 0.847 <sup>A</sup> | 2.016 | — | 0.500 <sup>A</sup> ² | *0.538 <sup>A</sup> · #122* |
| Constant velocity | 0.641 | 1.381 ⁴ | 1.639 | 1.393 | 0.239 | 0.207 |

**Closed loop**

| Model | Bench2Drive ↑ | neuroNCAP ↑ | AlpaSim ↑ |
|---|---|---|---|
| OpenPilot | 30.09 <sup>R</sup> ⁴ | 3.60 | **0.7726** |
| Alpamayo 1.5 | 55.82 <sup>R</sup> (10 Hz) | 3.73 <sup>A</sup> | 0.9000 <sup>R</sup> |
| Drive-JEPA | 57.60 <sup>R</sup> (2 Hz) | **4.13** | 0.4209 |
| VisionPilot <sup>P S</sup> | 47.82 | 1.58 | 0.6289 |
| METEOR | *1.05 <sup>R</sup> · #126* | 2.64 | 0.4232 <sup>R</sup> |
| Qwen-Drive 1.0 | 17.08 <sup>R</sup> (2 Hz) | 2.49 <sup>A</sup> | *0.7001 · #123* |
| AutoE2E | *15.30 · #96* | 3.16 <sup>A</sup> | 0.1694 <sup>R</sup> |
| Constant velocity | **48.31** | 0 (by construction) | 0.4876 |

**Notes**
1. nuPlan is not ranked.
2. Run at 10 Hz.
3. Drive-JEPA's front-camera variant.
4. Typed from a docs page rather than read from a run.

Only two Bench2Drive cells are ranked today: VisionPilot and constant velocity. #124
ranks the native-rate cells.

## v1.1 · Failure-mode analysis, Friday 11 December

**Why.** A score says how good a model is; failure analysis says why, and what to
change. That is the value no single benchmark gives, and it produced Paddock's biggest
wins:
- openpilot −21.8% on KITScenes with no retraining;
- Drive-JEPA made to see its camera, neuroNCAP 3.57 → 4.13.

Today that analysis is a set of per-model research scripts, and most of them leave the
public repo with #121.

**What exists today**

| Tool | What it answers | Covers | After 3 Nov |
|---|---|---|---|
| `paddock render` | how models differ on the same sample, side by side | 5 of 9 benchmarks, every model | stays |
| comma2k19 report and overlay video | every declared metric, mean and median; the plan drawn on the drive | comma2k19 | stays |
| NAVSIM's safety terms on other benchmarks (#102) | collision, drivable area, time to collision, progress, comfort | neuroNCAP in full; comma2k19, nuScenes, KITScenes and nuPlan in part | stays |
| neuroNCAP comparison video | what happened in closed loop | neuroNCAP | stays |
| Regression gate | did a change help, beyond noise? | smoke subsets of 6 benchmarks | stays |
| openpilot notebooks, failure tags, "four questions" | start-speed share, camera calibration, scene sweeps, failure classes | openpilot on KITScenes, comma2k19, Bench2Drive | moves to `paddock-lab` |
| Drive-JEPA candidate viewer | could it not make a good plan, or did it pick the wrong one? | Drive-JEPA on KITScenes, nuScenes | moves |
| VisionPilot diagnostics | what each of its three networks saw and decided | VisionPilot | moves |
| neuroNCAP render and projection checks | was the hazard rendered where the model could see it? | neuroNCAP | moves |
| nuScenes collision margin | does an open-loop near miss predict a closed-loop crash? | nuScenes | moves |
| openpilot Bench2Drive analysis (#79); Alpamayo decision analysis (#86) | standstill trap, untimed turns; reasoning against action | one model each | open PRs |

**The pattern:** deep for one model at a time, and nothing runs for every model and
dataset.

**The standardization plan: five ideas**

| Idea | What | Why |
|---|---|---|
| 1. One failure report per run | `paddock analyze <run>` writes a report with figures, for any model on any benchmark | the same diagnosis for every model, a customer's included, with one command |
| 2. Error broken down the same way everywhere | open loop and NAVSIM: lateral and longitudinal error, start-speed share, error by horizon, speed band and manoeuvre | one ADE hides what failed. 64% of openpilot's KITScenes error is the speed it starts at; Drive-JEPA's error is 11:1 longitudinal |
| 3. One failure vocabulary | scenario tags computed from the ground truth, not per model: turn left or right, straight, stop, start from rest, lead vehicle, junction, speed band | compare failure profiles across models: who fails left turns, who cannot start. openpilot's 769-line tag file proved the idea for one model |
| 4. Closed-loop outcome classes, the same across simulators | completed, at-fault collision (front, side, rear), off road, stuck at standstill, timeout; time to the event; NAVSIM's safety terms | the standstill trap (53% of openpilot's lost Bench2Drive points) and METEOR not pulling away on 43 of 50 routes (#126) were found by hand; each simulator reports failure its own way |
| 5. Input checks for every model | blank the camera, freeze the history, change the speed: does the plan move? | Drive-JEPA was camera-blind for weeks and still scored; AutoE2E's plans follow its own past motion |

**Expected on 11 December:**
- `paddock analyze` runs on every valid run behind the v1.0 matrix, and writes one
  report per run.
- Ideas 2 and 3 cover the four open-loop benchmarks and NAVSIM.
- Idea 4 covers neuroNCAP and Bench2Drive. AlpaSim and HUGSIM may follow.
- Idea 5's results are recorded for all 8 models, on each model's page.
- Each benchmark page has a model × failure-type table, regenerated from the reports
  the way the support matrix is.
- Tested without a GPU in `make check`, on stored fixtures.
- Out of scope: model-specific probes, such as VisionPilot's networks and Alpamayo's
  reasoning traces. They stay in `paddock-lab`.

## v1.2 · Regression gate for fine-tuned models, later

**What.** `tools/gate.py`, an internal tool today, becomes a supported feature. Given a
fine-tuned checkpoint and its base model, it:
- runs both on the same held-out subsets;
- answers SHIP, HOLD or FAIL;
- states the smallest change each benchmark could have detected.

**Why.** Open-loop gains have reversed in closed loop before. Drive-JEPA's re-ranker went
2.52 → 0.97 m on KITScenes, but 3.01 → 1.85 on neuroNCAP.

**Exists today:**
- pairing, sample by sample;
- errors clustered by scene, log or route;
- the smallest detectable difference;
- tiered smoke subsets: a quick tier in about 40 minutes, a closed-loop tier in a few
  hours;
- a shared run queue.

**Still to do:**
- a user-facing command and its docs;
- checkpoints read from Gallops' sidecars;
- predictable tier run times on shared machines.

The date is set after v1.1.

## Open pull requests, 29 October

| PR | What | Lands in |
|---|---|---|
| #121 | cleanup before the repo goes public | v1.0, before 3 Nov |
| #125 | CI on Ubuntu 24.04 | v1.0 |
| #126 | METEOR on Bench2Drive, 1.05 | v1.0 |
| #124 | Bench2Drive's native-rate cells ranked | v1.0, after review |
| #123 | Qwen-Drive on AlpaSim, 0.7001 | v1.0 |
| #122 | AutoE2E on NAVSIM v1, 0.538 | v1.0 |
| #96 | AutoE2E on Bench2Drive, 15.30, with its recorder and early stop | v1.0 |
| #57 | VisionPilot synced to upstream | v1.0 if re-measured, since it un-stales the row; otherwise after |
| #91 | VisionPilot improvements, no measurements yet | after v1.0 |
| #79 | openpilot Bench2Drive failure analysis | input to v1.1 (idea 4), or `paddock-lab` |
| #86 | Alpamayo decision analysis (conflicts) | input to v1.1, or `paddock-lab` |
| #53 | sim-to-real study, no push since 13 Oct | `paddock-lab` |
