# alpamayo15_teacher / alpamayo15_teacher_sft: the planner used to label Alpamayo 1.5's training data

Not a scored baseline and not a real driving system. This is a hand-written,
privileged planner -- exact world state in, no learned components, no image
input -- driven through paddock's own closed loop purely to produce the
labelled episodes Alpamayo 1.5's SFT fine-tuning trains on. It appears on
`paddock run`/`paddock report` like any other model because that is the
mechanism paddock already has for driving something through neuroNCAP's
renderer and recording what happened; it should never be read as a
leaderboard entrant, since it has information (exact actor pose and velocity)
no vision model is given.

Adapter: `src/paddock/adapters/models/alpamayo15_teacher/`. It is a thin
translation layer -- state bookkeeping between calls, SFT-record buffering --
not a reimplementation of the planner itself. All planning logic (Frenet-frame
candidate generation, speed profiles, go-around laterals, hard-constraint +
anti-freeze scoring) lives in `alpamayo-scripts`' `rl_training/expert/`, a
pinned vendor repo (see `repos.yaml`), imported the same way every other
model's own code is: in that repo's own interpreter, over its own
`pythonpath`, never inside paddock's process.

```bash
paddock run --model alpamayo15_teacher --bench neuroncap --limit 1
```

`alpamayo15_teacher_sft` is the same planner, re-registered under its own
name so its `model.yaml` can declare Alpamayo 1.5's full three-camera rig
(`CAM_FRONT_LEFT`/`CAM_FRONT`/`CAM_FRONT_RIGHT`) instead of
`alpamayo15_teacher`'s single `CAM_FRONT` (kept there for wall-clock parity
with the vision models it is occasionally compared against on render cost,
not for scoring). Collection for SFT runs against `_sft`, so the engine
renders every camera the model being trained actually conditions on.

## Why it's privileged, and how that reaches it

The planner needs exact current-step actor state (position, heading,
velocity) and which scene is being replayed -- neither of which a normal
`Policy` receives. Both arrive on the wire via `Observation.actors` /
`Observation.scene_name`, populated only when neuroNCAP's engine carries
`patches/neuro-ncap/0003-expose-current-actor-state-to-model.patch` (pinned in
`repos.yaml`). Every other model ignores both fields, so this patch cannot
change what anything else is scored on.

## Collecting SFT data

Set `EXPERT_RECORD_SFT_DIR` to a directory before the run; the policy buffers
one JSONL file per episode there (ego state, actors, scene identity, the
planner's native 64-waypoint/10 Hz plan), flushed on the *next* episode's
first `reset()` or scene change -- there is no explicit "episode ended" signal
on the wire, so that is the first point the previous one is provably over.
Camera frames are not written by this policy; the vendor engine already
writes raw per-step jpegs under the run directory, and each record's
`timestamp_us` is the join key a downstream script uses to find them.

```bash
EXPERT_RECORD_SFT_DIR=/path/to/sft-data paddock run --model alpamayo15_teacher_sft --bench neuroncap
```

## Limits

- **Not comparable to any scored model.** It reads ground truth no vision
  model gets; a collision rate or ADE from it means something different in
  kind, not just in number, from every other row `paddock report` prints.
- **Only neuroNCAP.** Open-loop benchmarks have no actor ground-truth channel
  for this adapter to read -- extending to one needs that benchmark's own
  ground-truth side channel first, not a change here.
- **The data it produces has not been independently re-verified against the
  rendered scene** beyond what `rl_training/expert/`'s own tests check. If a
  training run downstream looks wrong, check the labels before the trainer.
