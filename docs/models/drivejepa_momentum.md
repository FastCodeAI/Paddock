# drivejepa_momentum — Drive-JEPA with momentum-aware selection

Drive-JEPA's paper ends with a selection stage, *Momentum-aware Trajectory
Selection*, that "picks the final trajectory by accounting for cross-frame
comfort". Every `drivejepa` result in paddock leaves it off on purpose. This
entry turns it on so its effect can be measured, and so far that is on
neuroNCAP only.

Everything up to the 32 scored candidates is `drivejepa` unchanged: same
checkpoint, environment, inputs and forward pass. Only the pick differs.

## What the step does

At the pinned sha the stage is `DriveJEPAModel.calibrate_score`
(`navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py`).

1. Roll out each of the 32 candidates through NAVSIM's PDM simulator (LQR
   tracker and kinematic bicycle, 40 steps at 0.1 s).
2. Compare each rollout with the rollout of the plan driven on the previous
   frame, over the time the two overlap. A candidate passes if the RMS
   differences in acceleration, jerk, yaw rate and yaw acceleration are all
   within NAVSIM v2's two-frame Extended Comfort thresholds (0.7 m/s², 0.5 m/s³,
   0.1 rad/s, 0.1 rad/s²).
3. Score = `(14 * pdm_score + 2 * passed) / 16`, then take the argmax. Passing
   adds a flat 0.125.

The vendor only runs this when the caller passes `past_ego_simulated_states`,
and only NAVSIM v2's sequential scorer does that. Every paddock benchmark calls
the agent without it, so every `drivejepa` number is the plain argmax.

## What had to change for closed loop

- **The starting state is built from the observation.** NAVSIM reads it from
  its metric cache, which closed loop does not have. This adapter builds it
  from the observation: rear axle at the origin, measured speed, acceleration
  and yaw rate, and the tire angle that yaw rate implies for the simulator's
  wheelbase. The comfort features do not change when the frame moves or
  rotates, so each frame's rollouts stay in that frame's own coordinates.
- **The previous frame is the previous tick, 0.1 s back.** NAVSIM steps at
  0.5 s and neuroNCAP at 0.1 s. Like the vendor, the adapter drops the previous
  plan when it is more than 0.55 s old.
- **The overlap is rounded.** The vendor computes `int(interval / 0.1)`, which
  gives 2 for 0.3 s, and 0 (an empty overlap) for an interval just under 0.1 s.

## Verified

- **Same scores as the vendor.** Replayed through both this adapter and the
  vendor's own `calibrate_score`: 1,242 recorded neuroNCAP ticks (21 episodes
  from the Drive-JEPA candidate dumps), with each tick's pick fed into the next
  as it would be in closed loop. Scores were bit-identical on every tick (max
  |diff| 0) and the picks matched. This tests the scoring code, not how the
  starting state is built: the dumps have no yaw rate, so the replay used 0.
- **Momentum rarely changes the pick along the baseline's own path.** In that
  replay it moved the pick on 24 of 1,242 ticks (1.9%). In closed loop each
  changed pick moves the car, so this is a preview, not the effect.
- **The real model, end to end, on real frames.** 20 consecutive nuScenes val
  keyframes (scene-0003), run through plain `drivejepa` and through this model
  with the real checkpoint and the same loaded weights. Momentum applied on 16
  ticks; the other 4 had no previous plan within 0.55 s. It changed the pick
  on 1 tick. That tick is the only one whose plan differs from `drivejepa`;
  the other 19 plans are bit-identical. That shows the only thing this model
  changes is the pick, and its own `pick_changed` count matches what it
  actually drove. Along the way, 46% of candidates passed the comfort check.
- **Yaw rate is not taken from neuroNCAP's CAN bus.** The engine differences
  raw headings without wrapping, so a heading crossing ±180° would read as
  about 63 rad/s and become full steering lock in the starting state. None of
  the September sweep's 200 episodes crossed it (largest heading change
  0.03 rad per tick), but this model can drive a different path, so it takes
  the wrapped heading change between the last two frames instead.
- **Cost:** about 20 ms of CPU per tick for the 32 rollouts.

## Results

neuroNCAP, the full official set (20 scenario pairs × 10 seeds, n=200, 0
failures, `valid: true`), run `20261002T065426-a87115`. Paired seed for seed
with `drivejepa` run `20260905T220225-f11ac0`. The SE and t columns are at the
pair level (n = number of pairs), as neuroNCAP comparisons must be.

| split | drivejepa | momentum | diff | pair SE | t |
|---|---:|---:|---:|---:|---:|
| all (n=20) | 3.568 | 3.618 | +0.050 | 0.050 | +0.99 |
| stationary (10) | 4.232 | 4.232 | +0.000 | 0.000 | — |
| frontal (5) | 2.421 | 2.420 | −0.000 | 0.001 | −0.06 |
| side (5) | 3.388 | 3.586 | +0.198 | 0.201 | +0.99 |

Collision rate fell from 45.5% to 44.5%: 2 of 200 episodes.

**Read it as no measurable effect.** The whole +0.050 comes from a single
pair, side-0108. There, seeds 1 and 8 went from a collision to a clean pass
(0 → 5 each), and the other 8 seeds still collided, at 6.16 m/s instead of
6.22. No other pair moved by more than 0.01. A one-pair gain is exactly what
t = 0.99 says: a near miss tipping over in 2 of 10 runs, not a change in how
the model drives.

Momentum applied on 13,743 of 13,977 ticks and changed the pick on 324 of
them (2.4%). On 3.1% of ticks the model's own top choice failed the comfort
check; 39% of candidates passed it on an average tick. It cannot help the
frontal split, where the crashes need a swerve that none of the 32 candidates
contains (see `docs/models/drivejepa-neuroncap.md` on the analysis branch).
