# Drive-JEPA adapter fixtures

## `momentum_replay_made_up.npz`

A made-up drive's menus and what the momentum stage picks from them: 27
consecutive replans 0.5 s apart, 101 kB. Used by `../test_momentum.py`, and
written by `make_momentum_replay.py` beside it.

| array | shape | what |
|---|---|---|
| `time_us` | (27,) | timestamp of each replan |
| `proposals` | (27, 32, 8, 3) | 32 proposals, (x, y, heading) at 0.5 s, ego frame |
| `pred_logit` | (27, 32, 6) | six logits per proposal; the last is the learned score |
| `ego_pose` | (27, 3) | (x, y, heading) in a frame fixed for the drive |
| `ego_velocity`, `ego_acceleration` | (27, 2) | the ego-status row of each replan |
| `pick` | (27,) | the proposal picked with the stage on |
| `calibrated` | (27, 32) | the score after `calibrate_score`, the one the argmax was taken over |

**Where it came from.** Every number is invented, so the repository holds no
model output recorded on a licensed scene. The drive is about 8 m/s with a slow
left bend. Each replan has 32 arcs: three quarters close to the current motion,
as a trained planner's are, and a quarter outliers. The learned score sits in a
band narrow enough for the stage's 2/16 comfort term to matter.

`pick` and `calibrated` are what upstream's own chain returns on those menus:
the vendored `DriveJEPAModel.calibrate_score` and the argmax after it, fed by a
loop that follows upstream's evaluation loop (`run_pdm_score_sequential.py`)
and simulates each chosen plan the way `pdm_score.py` does, each replan chained
from the previous pick. That loop calls upstream's functions directly and
shares no code with `momentum.py`, so the values are not a previous run of the
code under test. It also fills the simulator's first slot with a different plan
from the second, so a simulator that coupled its batch entries would make the
two disagree. Re-running the script in Drive-JEPA's environment writes the same
arrays.

**What it exercises.** The stage changes the pick on 13 of the 26 replans it is
fed, so both outcomes, overturning the plain argmax and keeping it, are pinned.

**What it does not cover.** Every interval is exactly 0.5 s; the ego state is
given, not built from a paddock `Observation`; and it is a replay of fixed
menus, not a drive in which the stage's picks moved the car.
