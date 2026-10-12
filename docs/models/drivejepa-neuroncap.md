# Why Drive-JEPA crashes on neuroNCAP

> **Correction (2026-10-12): everything on this page was measured on a camera-blind
> model.** Until #62 the adapter placed Drive-JEPA's cameras relative to nuScenes'
> LIDAR_TOP, yawed −90°, so none of the points the model reads the image at landed
> in the front image and it planned from ego status alone (a black frame gave the
> identical plan). The "fixed habits" below — the speed decay, the creep, no swerve
> among the 32 — are that blindness. With the fix it brakes hard for an oncoming
> car on 0103 and drives at a normal pace; see #62 and
> `docs/models/drivejepa-camera-blind.md` on `record/drivejepa-camera-blind`.

Drive-JEPA scores 3.57 on the full neuroNCAP set, which is not separable from the
best model there (Alpamayo 3.73). That number hides three fixed habits, and this
page records what each of them is, how it was measured, and which of them a
better *choice* among the model's own candidates could fix. The short answer:
**none of them**. Everything Drive-JEPA computes was tested as a selection
signal, and the remaining lever is training.

Every measurement here comes from artifacts already on disk. Nothing was
re-simulated except where a run id is given, and each of those runs reproduced
the September sweep's plans and scores exactly before anything was read from it.

## Where the score is lost

Paired at the scenario-pair level (n = 20 pairs; the ten seeds of a pair share a
scene, so episode-level standard errors are 2-3x too small -- see
[neuroncap.md](../benchmarks/neuroncap.md)):

| split | Drive-JEPA | Alpamayo 1.5 | difference | t |
|---|---:|---:|---:|---:|
| all | 3.57 | 3.73 | -0.16 (SE 0.32) | -0.50 |
| frontal | 2.42 | 3.64 | -1.22 (SE 0.66) | -1.85 |
| side | **3.39** | 2.45 | +0.93 (SE 0.85) | +1.10 |
| stationary | 4.23 | 4.41 | -0.17 (SE 0.18) | -0.95 |

## Three habits, all measured over the 200-episode sweep

**It never plans sideways.** Over the 50 frontal episodes the largest sideways
offset in any 3 s plan averages 0.11 m, and the car never gets more than 0.3 m
off the logged path. Alpamayo plans 10.4 m and METEOR 5.2 m, and those two avoid
most frontal collisions.

**It brakes the same way in every episode.** Speed at 2 s over starting speed is
0.547 +/- 0.044 across all 200 episodes, with a correlation of 0.15 against the
distance to the target at the start. It slows the same way whether the hazard is
18 m or 90 m away, and where the logged human held 10 m/s. Every other model
varies with the scene (sd 0.19-0.36). Each plan's first waypoint asks for 0.89x
the current speed.

**It never plans a stop.** No plan in 200 episodes is slower than 1.29 m/s, and
the car never dropped below 0.3 m/s. In stationary scenarios it creeps into the
parked car at about 1.7 m/s -- 32 of 100 episodes.

The braking habit is also why it leads on side scenarios: it arrives late and the
crossing car has gone. Where the crossing car has not gone, it creeps into it.

## What the benchmark does and does not explain

**The engine's route command is not the cause.** The vendor derives a
LEFT/RIGHT/STRAIGHT command from the logged path, and in a scenario whose path
runs through the obstacle it can point a swerving car back into it. Recomputed
with the vendor's own rule for every frontal tick (`kinematics.py`), the command
Drive-JEPA received was STRAIGHT on 100% of them: it never gets the 2 m off the
path that would change it.

**Braking alone cannot score well in frontal.** The impact speed the score uses
is the *relative* speed of the two cars
(`neuro_ncap/evaluation/collision.py`), and the target drives toward the ego, so
a full stop still scores at most 4 * (1 - v_actor / v_reference), about 2.0 on
these pairs. Drive-JEPA's frontal collisions already average 1.69.

**Its own braking is fed back to it.** In closed loop the acceleration input is
the car's real acceleration, which is the result of Drive-JEPA's previous plan
(`neuro_ncap/engine.py`, `_simulate_canbus`); in NAVSIM training it is always a
human's. Feeding 0 instead lifts the plans from 0.89x to 0.97-0.98x of current
speed and the 2 s ratio from 0.53-0.60 to 0.62-0.80, so the feedback drives much
of the habit. It is not a bug in the benchmark -- a real car reads its own
deceleration too -- and removing the input costs score (-0.23 at pair level,
SE 0.16, n = 3 pairs), because the slowdown was protecting it. The fix belongs in
training, by dropping or noising that input.

## Could a better choice among the 32 candidates fix it?

Drive-JEPA proposes 32 trajectories a tick and its NAVSIM-trained scorer drives
one. Dumping all 32 over 7 pairs x 3 seeds and checking each against the target
(`candidate_check.py`, `candidate_report.py`) gives the shape of the problem:

| split | what the 32 hold before a crash | fixable by choosing better? |
|---|---|---|
| frontal | no swerve anywhere: the widest candidate moves 0.8-1.2 m sideways. Only early hard stops stay clear, and they are gone 1-3 s before impact | **no** -- at best an earlier stop, capped near 2.0 |
| stationary | a safe stop on 97-98% of ticks, up to 0.1 s before impact; the scorer ranks the best one about 9th of 32 | yes, in principle |
| side | a safe option on 79-98% of ticks until about 1 s before impact, ranked 4th-8th | yes, in principle |

So the frontal gap needs a model that proposes swerves. For the other two, every
signal the model produces was then tested as the picker:

| signal | ranks hitting candidates as riskier? | picks a safe one when it matters? |
|---|---|---|
| the scorer's final score | partly | no -- it drives the colliding one |
| the KITScenes re-ranker (`drivejepa_rerank`) | no | no: 0 of 322 ticks; it never picks a stop |
| the scorer's TTC logit | yes, within a tick | only by braking everywhere: rescuing 17 of 322 costs KITScenes ADE 0.97 -> 3.96 |
| `pred_col_agent`, the training-only collision head | yes: 85-94% (stationary), 77-91% (frontal), blind on side | no: 0 of 322 |
| a linear probe on the features every scorer head reads | yes, and no better than the head once plan length is matched: 94% (stationary), 95% (frontal), 41% (side) | no: 0 of 322 |

"When it matters" is the 322 ticks where the plan Drive-JEPA drove hits although
one of the 32 was clear. The collision head is the sharpest of these signals and
still fails there: it gives the colliding pick 3-7% and the real escapes 2-5%, and
its own safest candidate is clear on 0-4% of those ticks. It flags candidates that
drive far into an obstacle, which is where its good ranking comes from, and misses
the slow creep that makes contact two seconds later -- which is how Drive-JEPA
actually crashes in stationary scenarios.

The probe is the one that settles whether the heads are simply failing to express
something the model knows. It is a leave-one-pair-out ridge fit on
`proposal_feature`, the 256 numbers every scorer head reads, over 38,080 candidate
rows, and it is judged by the same two tests. Its ranking looks strong until the
comparison is length-matched, at which point it lands where the collision head
already is -- so the head is close to the linear ceiling of those features and a
head-only finetune has little headroom. Two things it does establish: the features
carry nothing linear about a car crossing our flank (41%, against 33% for plan
length alone), so side is not a head problem; and every signal found here is
dominated by plan length (the probe's risk correlates 0.97 with it, and its safest
candidate is the shortest one on all 322 ticks), which is why nothing turns a good
ranking into "stop now" without also braking everywhere.

Two cautions on those numbers. "Clear" is an open-loop verdict: one candidate
followed for 4 s with no replanning, so it shows an option existed, not that
choosing it would have saved the episode. And the collision head's own label is
NAVSIM's at-fault collision, judged by NAVSIM's simulation, which is not the same
as the box overlap checked here -- the side blindness may simply be that a car
driving into our flank is not our fault by that definition.

## Would a wider set of candidates fix it?

Selection is exhausted, but that is a statement about a set of 32 candidates that
never moves sideways. The obvious cheaper repair is to leave the scorer alone and
widen what it chooses from -- proposal diversity is trainable on data already held,
and needs no rendered collision scenes at all. So the swerves were handed to it
directly.

Six were injected per tick across the three frontal pairs, at +/-1.5, 2.5 and 3.5 m,
each with the same forward travel as the plan the model picked and a heading
recomputed to follow the displaced path. They enter at the last refinement, so
`Bev_refiner` samples the image along the swerve and the scorer reads real features
for it rather than a pasted trajectory. Driving is untouched: the plan comes from a
clean first pass, so the three episodes reproduce their recorded scores exactly
(2.171, 1.865, 1.747) and every candidate slot left alone scores bit-identically --
`0.00e+00` drift across all 509 ticks, which is what makes the comparison a
measurement rather than a demonstration.

| | clear of the target | mean `pdm_score` | preferred over the driven plan |
|---|---:|---:|---:|
| the plan it drove | 27.6% | 0.805 | -- |
| swerve +3.5 m | 69.4% | 0.163 | 0 of 434 |
| swerve -3.5 m | 77.4% | 0.039 | 0 of 434 |
| best of the six | 84.6% | 0.168 | 0 of 434 |

On the 248 of 314 hitting ticks where a swerve would have been clear, the best clear
swerve scores 0.151 against 0.808 for the plan that drives into the target, and is
never preferred. **So the missing swerve is not only absent from what the model
proposes; it is refused when supplied.** Widening the proposal set alone would
change no decision, because the half that chooses discards the wider options.

One caveat the result carries: these candidates are far outside anything the scorer
was trained on, so a low score may be unfamiliarity rather than a judgement about
danger. That does not reopen the cheap path -- an unfamiliar option is refused as
firmly as a disliked one -- but it does suggest the scorer may retrain more easily
than a five-fold gap implies.

## Reproducing this

The cross-model parts need only recorded runs:

```bash
python -m tools.benchmarks.neuroncap.kinematics \
    --run drivejepa/<run id> --run alpamayo15/<run id> --out ticks.json
python -m tools.benchmarks.neuroncap.summarise --ticks ticks.json \
    --run drivejepa/<run id> --run alpamayo15/<run id> --against alpamayo15
```

The candidate parts need a run that dumped the model's full output per tick.
`DriveJepaPolicy.infer` keeps only the chosen trajectory, so the dump came from
an analysis-only adapter kept outside the repo: it subclasses the policy, wraps
`agent.forward` to keep `proposals`, `pred_logit` and `pdm_score`, and writes one
JSONL per episode. For the collision head it also wraps the scorer's `forward`
and runs `pred_col_agent` and `pred_area` on its inputs under `no_grad`, because
those heads are evaluated only in training mode (`score_module/scorer.py`) and
return None while driving. None of it touches the pick, which is what lets the
run be checked against the recorded sweep: identical plans, identical scores.

```bash
python -m tools.models.drivejepa.neuroncap.candidate_check \
    --run <model>/<run id> --dumps <dump dir> --out check.json
python -m tools.models.drivejepa.neuroncap.candidate_report check.json
python -m tools.models.drivejepa.neuroncap.aux_heads --check check.json --dumps <dump dir>
```

Dumps and figures live under `{runs}`, not in the repo: one run of 21 episodes is
about 200 MB of candidates and another 180 MB with the heads.

## What is left

Fine-tuning **both halves**, and the injection is what makes that "both" rather
than "one":

* **propose evasive swerves** -- no candidate ever does, and forcing the command
  to LEFT or RIGHT for a whole episode moves the plan only 0.2-0.5 m, so it is
  not the command;
* **choose one when it is there** -- handed six, the scorer preferred none of
  them on any of 434 frontal ticks, so widening the proposals without retraining
  the scorer would move no decision;
* **score an imminent slow collision sharply** -- the collision head exists and
  is weak exactly where the crashes are, though the probe puts it near the linear
  ceiling of the features it reads, so this is unlikely to be a head-only repair.

Training is also where the acceleration feedback should be addressed. No
benchmark scene may enter that training: neuroNCAP's 20 pairs over 14 nuScenes
val scenes are the test set, so collision data has to come from elsewhere.
Two sources were weighed and only one survives. **nuScenes train boxes are
nearly free and nearly empty**: over its 700 train scenes there are 7.7 s of
braking at or below -4 m/s2 and 0.1 s of lateral acceleration at or above
4 m/s2, and NAVSIM's own trainval -- the corpus this model was trained on --
holds 1.5 s and 2.4 s of the same in 20.1 h. The habits are a faithful
reflection of what it was shown, so more of it teaches nothing. That leaves
**neuroNCAP-style actor insertion on non-benchmark scenes**, which is the
expensive answer: a NeuRAD reconstruction costs about 6-8 h of GPU per scene
(0.137-0.181 s/step over the 150,001 steps the benchmark's own reconstructions
used), and the target trajectories have to come from a planner that can see the
inserted actor's future, since no human in any log ever swerved around it.
