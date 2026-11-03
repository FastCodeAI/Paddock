# Drive-JEPA adapter fixtures

## `momentum_replay_2e1ce314.npz`

One drive's menus and what the momentum stage picked from them: 27 consecutive
replans 0.5 s apart, 102 kB. Used by `../test_momentum.py`.

| array | shape | what |
|---|---|---|
| `time_us` | (27,) | timestamp of each replan |
| `proposals` | (27, 32, 8, 3) | the model's 32 proposals, (x, y, heading) at 0.5 s, ego frame |
| `pred_logit` | (27, 32, 6) | the scorer's six logits per proposal; the last is the learned score |
| `ego_pose` | (27, 3) | (x, y, heading) in the simulator's local frame, fixed for the drive |
| `ego_velocity`, `ego_acceleration` | (27, 2) | the newest ego-status row the model was given |
| `pick` | (27,) | the proposal picked with the stage on |
| `calibrated` | (27, 32) | the score after `calibrate_score`, the one the argmax was taken over |

**Where it came from.** AlpaSim scene `2e1ce314` (NuRec 26.01).
The observations are those of a closed-loop drive of that scene under
`drivejepa-v2-driver-0.1.4`, which did not feed the stage. On 2026-10-01 they
were replayed in order through the planner of `drivejepa-v2-driver-0.1.4b`, the
first driver to hand Drive-JEPA its previous plan, with the real checkpoint:
Drive-JEPA at e21f474, the sha `repos.yaml` pins. The menus -- `proposals`,
`pred_logit` and the ego state beside them -- are that pass's model outputs,
and each replan was chained from the planner's own previous pick. `pick` and
`calibrated` are what the vendored `DriveJEPAModel.calibrate_score` and the
argmax after it returned there. Both drivers are AlpaSim drivers kept outside
paddock. A standalone script sharing no code with either recomputed all 27 picks
from the same menus and agreed.

The values are therefore not a previous run of the code under test: they predate
`momentum.py` and were produced by a different adapter around the same upstream
stage.

**Why this drive.** The stage changes the pick on 19 of its 27 replans, the most
of the 30 drives recorded, so a bookkeeping error has the most chances to show.

**What it does not cover.** Every interval is exactly 0.5 s; the ego state is
the recorded driver's, not one built from a paddock `Observation`; and it is a
replay of fixed menus, not a drive in which the stage's picks moved the car.
