# openpilot's driving model on KITScenes: four questions

This note answers four questions about openpilot's driving model (supercombo) on KITScenes, run open loop through paddock, and records what each answer assumes. Every measured number comes from [the analysis notebook](openpilot_drv_supcombo_kitscenes_openloop_analysis.ipynb) beside it, which recomputes them from the runs below each time it executes. The § numbers are the notebook's sections, and the figures are there, not here. The section after Q4 takes the four as one problem, with numbers from [the image calibration notebook](openpilot_drv_supcombo_kitscenes_image_calibration.ipynb), and there the § numbers are that notebook's.

## The short answers

1. **Can openpilot take driving intent, how can paddock supply it, and how does the model do with it and without it?** (§17)
    - It has one input for intent, `desire_pulse`, and paddock can send the route's turn there.
    - The model reads it, but the score gets worse: on the 39 turn keyframes, ADE goes from 1.72 to 2.03 m.
    - All of that cost is on the 9 turns 40 m or more ahead, which the model does not yet see. On the nearer 30, ADE does not move, and the plan turns later.
2. **Why does the camera height vary across val?** (§14)
    - The camera hardly varies: three rigs, whose heights differ by 2 cm.
    - What varies is the height the model reads from the image. It takes one of two values, near 1.39 or near 1.62 m, and which one follows the lane lines the model sees, not the rig or the city.
3. **What do the predicted, calibrated and GT speed profiles look like, scene by scene, and does ADE capture speed-profile errors?** (§16)
    - Mostly it does not. 64% of ADE is the error in the start speed, and ADE hardly ranks keyframes by the profile (ρ 0.06).
    - On 21 keyframes, the misread start speed hides an error in the profile, which shows only once the start is set right.
4. **Which image calibration makes the model read the right speed at t = 0?** (§15)
    - Showing the road as from about 1.55 m instead of the camera's 1.88 m: a height scale s\* = 1.22 (CI 1.17–1.26), measured on the 54 moving keyframes of a 60-keyframe sweep.
    - At the sweep points either side, 1.15 and 1.30, ADE falls from 1.44 to 0.97 and 1.03 m. Rescaling the plan by the speed from odometry does as well (0.97 m), without touching the image.

**Taken together** ([the four questions as one problem](#the-four-questions-as-one-problem), on the 39 turn keyframes): the plan's error has two parts, with different causes.
- **Its speed starts in the image.** A height scale near 1.2 fixes part of it. Rescaling the plan by the car's speed does nearly as well on the ten turns the image settings were searched on, and better on the other 29: there, ADE goes from 1.77 to 1.44 m with the image, and to 0.83 m with the rescale.
- **Its turn does not start in the image.** Over 95 image settings, at most one more of ten plans turns toward the route. The route as a desire makes more of them turn, but later than the driver did, which costs ADE once the speed is right.

## How to read this note

**Labels.** Each claim carries one:
- **verified**: checked in the code or the data;
- **measured**: computed by the notebook from the runs, and re-checked each time it executes;
- **argued**: a reading of the measurements or of the code, not itself measured;
- **OPI**: measured by the `openpilot_inference` repo under `{repos}`, on the same pinned ONNX with its own pipeline, and not re-checked here;
- **memory**: recalled, not checked.

**CIs** are 95% bootstrap intervals over scenes, since keyframes of one scene are not independent.

**Terms.**
- **Keyframes.** A keyframe is one scored sample. There are 223, over 140 scenes: 186 in val and 37 in overlap_train_val.
    - They are named here by their scene id's first 8 characters and their frame index, as in `b8002a79/40`.
    - `show_ab("kitscenes/val/<scene id>/<keyframe>")` draws one.
- **H = 3 s** is the scoring horizon. The GT covers 4 s.
- **The pose speed** is the speed from the dataset's poses: base_frame poses from the INS, 10 Hz, as a central difference over the frames either side of the keyframe. It is "the right speed".
- **The pose head** is the model's own speed output, and **`plan_v0`** is the speed at which its plan starts.
- **v_go = 2 m/s.** Speed ratios below it are noise, so they are left out.
- **The low and high modes** are the road head's two readings of the camera height, near 1.39 and 1.62 m. The notebook's tag is `cam_h_read`, and Q2 explains them.

**The runs.** Each is in `results/kitscenes/openpilot/<run_id>/`.

| run | what it sets | keyframes | used for |
|---|---|---|---|
| `20260930T154018-242869` | nothing: no desire, command STRAIGHT, height scale 1 | 223 | the baseline, `RUN_ID` |
| `20260930T014333-d329a6` | `command_source = "gt"`, `PADDOCK_OPENPILOT_DESIRE = "turn"` | 223 | the intent arm (Q1) |
| `20260930T014333-5811fc`, `20260930T160509-57c362`, `20260930T160954-02dd77`, `20260930T161429-4576f3` | `PADDOCK_OPENPILOT_HEIGHT_SCALE` = 1, 1.15, 1.30 and 1.53 | the first 60 by (scene, keyframe) | the height sweep (Q4) |

- **The baseline replaces `20260929T043619-242869`** (`paddock supersede`). The adapter cached its warps, and that cache gave 14 of its keyframes another rig's warp.
- **The arm and the s = 1 point ran before that fix, and neither was hit** (measured). The notebook checks both on every execution:
    - the arm's outputs, other than the plan, equal the fixed baseline's on all 223 keyframes, which a stale warp would break;
    - the s = 1 point equals the baseline on all 60.
- **The sweep points are subsets**, so they are `valid: false` and `paddock report` leaves them out.
- **The arm was stored valid,** so `paddock report` listed it as a second openpilot row. That is why the notebook pins `RUN_ID`. Since the review on #60 it is stored `valid: false`, as every `command_source = "gt"` run now is, because its route is read from the recorded future.

---

## Q1 Driving intent (§17)

### What openpilot can take (verified in the adapter and the pinned openpilot)

supercombo has six inputs, and none of them carries a route, a map or a speed:

| input | shape | what paddock sends |
|---|---|---|
| `img`, `big_img` | (1, 12, 128, 256) | the front camera through the narrow and the wide warp, as two frames 0.2 s apart |
| `features_buffer` | (1, 24, 512) | the model's own hidden state from earlier ticks |
| `desire_pulse` | (1, 25, 8) | zeros, unless `PADDOCK_OPENPILOT_DESIRE` is set |
| `traffic_convention` | (1, 2) | right-hand traffic |
| `action_t` | (1, 2) | the actuator delays, as openpilot's `modeld` computes them |

**`desire_pulse` is the one way in for intent.**
- It is a one-hot over openpilot's `log.Desire`, with 8 slots:
    - 0 none;
    - 1 turnLeft, 2 turnRight;
    - 3 laneChangeLeft, 4 laneChangeRight;
    - 5 keepLeft, 6 keepRight;
    - 7 unused.
- **It is an event, not a state.**
    - openpilot's `modeld` sends a desire once, on the tick it begins: the rising edge. The model decides when the manoeuvre is over.
    - A queue holds the pulse for 5 s: 100 ticks at 20 Hz, max-pooled into 25 bins of 0.2 s, so a one-tick pulse cannot fall between them.
- **It carries no distance and no timing.** Nothing in the input says where the turn is.
- **The model reports how it reads the desire** in its 8-way `desire_state` output. The adapter records that, with the slot it sent, as `extras.desire`.
- **Upstream, `modeld` sends it one tick late.** It reads the desire helper's output before updating it, so openpilot's own pulse lags a tick. That does not apply here, since paddock sets the desire itself.

### openpilot never sends a turn desire (verified)
- **`DesireHelper` sets only none and the lane-change pair.** A lane change needs:
    - the blinker;
    - 20 mph or more;
    - lateral control active;
    - torque toward the blinker.
- **No Python or C++ file in openpilot's history names turnLeft or turnRight.** That is `git log -G` over the pinned checkout's 17,604 commits since 2016.
- **So turn desires are off openpilot's deployment distribution.**
    - Whether training covered them cannot be read from the code. It can only be measured, and the model does read them (below).
- **Forks send them** (memory, not checked): FrogPilot and sunnypilot send turnLeft/turnRight when the blinker is on below lane-change speed.

### How paddock supplies it (verified in code)

There are two settings, both off by default, so no earlier run changes. The [openpilot guide](../../../../docs/models/openpilot-guide.md) and the [KITScenes doc](../../../../docs/benchmarks/kitscenes.md#the-route-hint) document both.

1. **`[benchmarks.kitscenes] command_source = "gt"`** gives the scored keyframe the sample list's `driving_command`, as LEFT, STRAIGHT or RIGHT.
    - The history frames stay STRAIGHT.
    - `unknown`, or anything that is not one-hot, reads as STRAIGHT.
    - It is benchmark-wide: alpamayo15, qwendrive, drivejepa and drivejepa_free read `Command` too.
    - It enters `run_config()` only when it is not `straight`, so the ids of older runs are unchanged.
2. **`PADDOCK_OPENPILOT_DESIRE`, in `[models.openpilot.env]`,** sends the command as a desire, through the adapter's table `DESIRE_MODES`:

    | mode | LEFT | RIGHT | STRAIGHT, FOLLOW_REFERENCE |
    |---|---|---|---|
    | `off` (default) | none | none | none |
    | `turn` | turnLeft | turnRight | none |
    | `lane_change` | laneChangeLeft | laneChangeRight | none |

    - **The table is explicit because the two enums do not line up.** `Command.RIGHT` is 0, which is the slot for none, and `Command.STRAIGHT` is 2, which is turnRight. Passed straight through, a command would drop every right turn and send every straight keyframe as a right turn.
    - **The adapter holds the desire level** for the frame that carries it. Both backends turn it into the rising edge, as `modeld` does.
    - **The setting is recorded** in the run's `model_env`, which is hashed into its id, and in the adapter's provenance.

**Only the keyframe carries the command** (argued from the code). Each keyframe is predicted after 49 frames, 4.8 s, each fed as two 20 Hz ticks.
- **Stamped on every frame,** as the nuScenes harness does, the rising edge would fall on the first frame. By the scored frame it would be 4.8 s old, in the queue's oldest bin.
- **Stamped on the keyframe alone,** it arrives on the keyframe's first tick, in the newest bin. That is when `modeld` would send it: as the manoeuvre starts.

**KITScenes has no honest source of intent** (verified).
- There is no CAN, so no blinker, steering or pedals.
- There is no route. The toolkit's one route API, `SceneMap.get_route`, needs `lanelet2`, which does not install under numpy 2.
- **Both hints in the sample list come from the recorded future:**
    - **`driving_command`**, by Drive-JEPA's rule: a turn exactly when the 4 s GT ends 5 m or more to one side. The notebook checks this: every turn GT ends 5.07 m or more to its side, and every straight one within 4.46 m.
    - **`nav_text`** ("Turn right in 21m"), from further along the recorded path. It disagrees with `driving_command` on 8 keyframes.

### What the intent arm assumes

Each of these limits what the arm can show.

1. **The route's turn is sent as a turn desire.**
    - That is what a route command means. But openpilot never sends turn desires, so the arm tests slots its runtime does not use (verified).
    - The model reads them all the same (measured, below).
2. **The command is the GT's own end.**
    - The arm is told which side the scored answer ends on. That is more than a car on a route knows: a route says where the car is going, not where it will be in 4 s.
    - So the arm is an oracle for the side, and it still scores worse (argued). That is evidence against handing this model a route through the desire, as it is sent here.
3. **The desire is sent once, at the keyframe.**
    - Keyframes fall every 9 s, wherever the turn is. The 39 turns are 9 to 79 m ahead by `nav_text`, and half are already underway: the GT has turned 9.7° or more within 1 s.
    - openpilot would send a desire as the manoeuvre starts, and the keyframe is not chosen that way.
4. **Straight is none.**
    - On the 184 straight keyframes the arm is the baseline, bit for bit.
    - So it cannot help the 7 keyframes where the model turns off a straight route (`turns_off_route`).
5. **There is one arm, driven by `driving_command`.**
    - The 8 keyframes where `nav_text` disagrees are all straight by `driving_command`, and 4 of them turn by 15° or more within the GT's 4 s.
    - The arm sent nothing on them, so an arm driven by `nav_text` is not measured.
6. **The test is open loop.** The plan is scored against one driver's 3 s. A plan that turns later but still makes the turn might do well closed loop, yet it scores worse here (argued).
7. **There is no lane-change control.**
    - It was planned to show whether any desire moves a turn plan. The turn desire moved every one, so it was not needed for that.
    - It would still show whether the unused turn slots behave differently from the lane-change pair.

### What the desire does (measured, the 39 turn keyframes over 33 scenes)

**The model reads the desire, and it changes nothing but the plan.**
- **With the desire:** sent the route's turn, `desire_state` puts 0.991 or more on the route's side at all 39.
- **Without it:** the baseline already leans to the route's side (its argmax) on 9 of the 39 turns, and takes the routed turn on all 9. On the 184 straight keyframes, its argmax is none on 180.
- **Only the plan moves.** Every other output is the baseline's to the bit, on all 223 keyframes: the lane lines, road edges, lead, pose and road heads.
    - The plan, its σ and `plan_v0` differ, and only on the 39 turns.
    - So the desire changes what the model plans, not what it sees.

**The score gets worse.**

| on the 39 turns | baseline | arm | change (CI) | arm better on |
|---|---|---|---|---|
| ADE over 3 s (m) | 1.72 | 2.03 | +0.30 (−0.04 to +0.72) | 21 |
| FDE at 3 s (m) | 3.30 | 4.25 | +0.95 (+0.17 to +1.85) | 17 |
| lateral MAE over 3 s (m) | 0.58 | 0.90 | +0.32 (+0.18 to +0.47) | 7 |
| longitudinal MAE over 3 s (m) | 1.50 | 1.56 | +0.06 (−0.31 to +0.51) | 22 |
| heading error over H (°) | 10.6 | 16.3 | +5.7 (+3.0 to +8.5) | 8 |

- **The mean rises through the tail.** The median keyframe's ADE hardly moves (−0.05 m).
- **Over all 223 keyframes,** ADE goes from 1.595 to 1.649 m.
- **Lateral MAE rises on both sides:** +0.43 m on left turns, and +0.19 m on right ones.

**The plan commits to the route, but it turns later, and it brakes for the turn.**
- **It commits.**
    - Over its 10 s, the plan turns a median 11.3° more, and more on 32 of 39.
    - It takes the routed turn on 35 turns instead of 31. turn_not_planned and turn_wrong_way each fall from 4 to 2.
- **It turns less within the 3 s that are scored:** a median 16.0°, against the baseline's 19.0° and the GT's 29.4°.
    - understeer goes from 10 to 17.
    - At H, the arm ends a median 0.74 m outside the GT's turn, where the baseline ended 0.35 m inside it.
- **It brakes.**
    - Acceleration over H falls by a median 0.33 m/s², and is lower on 30 of 39.
    - The plan brakes on 11 turns instead of 3, and phantom_brake goes from 3 to 10.
    - The GT brakes on 1 of the 39, and accelerates on 11.

### Where the cost comes from: turns the model cannot see yet (measured, `turn_change_by()`)

Spearman's ρ of the change in ADE (arm − baseline) on the 39 turns, with:

| | ρ (CI) |
|---|---|
| the distance to the turn, per `nav_text` | +0.46 (+0.10 to +0.72) |
| the pose speed at t = 0 | +0.48 (+0.14 to +0.74) |
| the time to the turn, their ratio | −0.06 (−0.37 to +0.27) |
| the baseline's own p of the turn | −0.43 (−0.70 to −0.10) |
| how far the GT turns over H | −0.18 (−0.54 to +0.20) |

**Distance and speed cannot be told apart.** They move together on these turns (ρ 0.91).

Split at 40 m to the turn (baseline → arm; the GT in brackets):

| | 9 turns 40 m or more ahead | 30 turns under 40 m ahead |
|---|---|---|
| distance to the turn | 40–79 m, median 54 | 9–38 m, median 24 |
| pose speed at t = 0, median | 10.0 m/s | 4.3 m/s |
| time to the turn at that speed, median | 5.3 s | 5.7 s |
| the baseline's p of the turn, median | 0.0003 | 0.34 |
| ADE, mean | 1.95 → 3.49 m | 1.66 → 1.59 m |
| the change in ADE: mean, median; rises on | +1.54, +1.56; 7 of 9 | −0.07, −0.24; 11 of 30 |
| along-track MAE, mean | 1.92 → 3.23 m | 1.38 → 1.06 m |
| cross-track MAE, mean | 0.28 → 0.70 m | 0.67 → 0.96 m |
| acceleration over H, mean | +0.25 → −0.56 m/s² (+0.44) | +0.08 → −0.15 m/s² (+0.28) |
| heading change over H toward the route, mean | 15.3° → 16.5° (16.5°) | 24.7° → 15.7° (34.9°) |
| heading change over the plan's 10 s, mean | 22° → 42° | 33° → 49° |
| first heads 15° toward the route, median\* | 3.0 → 3.0 s (2.25 s) | 1.25 → 2.5 s (1.5 s) |
| paths that head 15° toward it at all | 7 → 9 (6) | 26 → 28 (30) |
| phantom_brake | 0 → 6 | 3 → 4 |

\* Over the paths that reach 15°, so each median is over a different set. The GT covers 4 s.

- **Far turns: the model does not see them, and the desire makes it brake.**
    - Unaided, the model puts a median p of 0.0003 on the turn.
    - With the desire, it brakes where the GT accelerates, and plans nearly twice the turn over its 10 s.
    - Most of the ADE rise is along the path.
- **Near turns: the model sees them, and the desire makes it turn later.**
    - Unaided, the model already puts a median p of 0.34 on the turn.
    - With the desire, it first heads 15° toward the route at 2.5 s instead of 1.25 s, where the GT does at 1.5 s. It turns further over its 10 s.
    - Error moves from along the path to across it, and ADE stays flat.
- **Both groups are about 5.5 s from their turn.** So what separates them is the distance, or the speed, not how early in time the desire comes.

### The side by side (the figures in §17)

`show_ab()` draws a keyframe as both runs planned it:
- the camera twice, with the baseline's plan left and the arm's right, drawn over a thin copy of the baseline's;
- a BEV with the GT and both plans;
- the speed per 0.5 s with the t = 0 markers of Q4;
- a text panel of what changed.

**§13's keyframes, drawn again.**
- **The 13 straight keyframes are unchanged.** Of the 20 keyframes, 13 are straight, and the arm plans them bit for bit as the baseline does.
- **All 7 turns change.**
    - Both turn_not_planned keyframes, `dbba76d9/130` and `67abb4e2/220`, plan the turn with the desire.
    - Both turn_wrong_way keyframes, `5bf1c78c/40` and `d96782f8/220`, now turn the right way. `d96782f8/220` turns too little instead, and enters turn_not_planned.
    - ADE falls on 4 of the 7 and rises on 3.
- **`dbba76d9/130` shows the timing.** It is "Turn left in 18m", at 3.9 m/s, where the GT turns 61° within 3 s and first heads 15° left at 1.5 s.
    - The baseline plans straight on.
    - The arm plans the left turn, but goes straight for about 10 m first. It first heads 15° left at 5.0 s, well after H.
    - It also slows, from 3.5 to 1.5 m/s by 3.5 s, where the GT holds about 3.5 m/s and then speeds up to 4.6.
    - ADE barely moves: 2.32 → 2.27 m.

**Every turn keyframe** (`turn_gallery()`, the largest rise in ADE first): ADE rises on 18 and falls on 21. Three show the pattern:
- **`b8002a79/40`: a far turn the model does not see.** It is 64 m ahead at 12.8 m/s, on a road that already curves right.
    - Unaided, the model follows the curve: ADE 2.14 m.
    - With the desire, it brakes from 11 to 5 m/s over 10 s and turns right more sharply than the road: ADE 5.12 m, almost all of it along the path.
- **`8c501afb/40`: a near turn made worse.** It is 17 m ahead at 4.2 m/s.
    - Unaided, the model turns left with the GT: ADE 0.74 m.
    - With the desire, it goes straight for about 10 m before turning: ADE 2.30 m.
- **`6848358c/40`: a near turn made better.** It is 22 m ahead at 3.8 m/s.
    - Unaided, the model turns at once and slows to 2 m/s.
    - With the desire, it turns later and keeps more of its speed, nearer the GT, which speeds up: ADE 2.97 → 1.55 m.

### What it means (argued)

- **The model reads a turn desire as "a turn starts here", and the desire says nothing about where the turn is.**
    - openpilot sends a desire when the manoeuvre starts, and lets the model decide when it is over.
    - KITScenes' turn keyframes are not chosen that way. Half are already turning, and the car does not slow through them.
- **Where the image shows no turn, the model trusts the desire.**
    - 40 m or more ahead at 10 m/s, it slows for a turn it cannot see. The GT does not slow within its 4 s.
    - Those 9 keyframes carry the whole of the ADE cost.
- **Where the model already sees the turn, the desire changes when it turns more than whether.**
    - It turns later, and further over its 10 s.
    - That moves error from along the path to across it, and leaves ADE flat.
- **The turn slots respond, although openpilot's runtime never sends them.** So either training covered them, or the head generalises from the lane-change pair. The code cannot say which.
- **A hint from the future does the same to another model** (verified in `results/`). alpamayo15's nav hint, on the same 223 keyframes, helped the typical keyframe and hurt the tail:

    | alpamayo15 | with the hint | without |
    |---|---|---|
    | median ADE (m) | 0.661 | 0.696 |
    | mean ADE (m) | 1.00 | 0.92 |
    | p90 ADE (m) | 2.17 | 1.93 |

### What a test of intent that could help would need

- **A command from where the car is going, not where it ends up: a route.**
    - On KITScenes that needs `lanelet2` under numpy 2.
    - Short of that, `nav_text` could drive the arm; that arm was not run.
- **Timing.** The desire should come when the turn becomes relevant, gated by the distance to it or by speed. These turns cannot say which.
    - Composed from the two runs, not run: sending the desire only where `nav_text` puts the turn under 40 m would have moved the turns' mean ADE from 1.72 to about 1.67 m.
    - The 40 m cut was chosen after seeing these data. So that is a hypothesis for a run, not a result.
- **Closed loop, where a route exists** (argued). Bench2Drive sends CARLA's route command. Whether turning later matters is a closed-loop question.
- **The blinker, where a dataset records it.** comma2k19 has CAN, but its blinker needs the raw CAN decoded.
- **The `lane_change` control,** to compare the turn slots with the ones openpilot actually uses.

---

## Q2 The camera height across val (§14)

**The camera hardly varies across val; the model's reading of its height does** (verified on the 117 val scenes and the run's 23 overlap_train_val ones).

**Three calibrations, one per rig, cover every scene.** Only `T_to_reference` differs between them; the intrinsics are identical.

| rig | 53df1231 | 901dfe25 | dbb25dcf |
|---|---|---|---|
| val scenes | 89 | 25 | 3 |
| val cities | 84 Frankfurt, 5 Karlsruhe | 25 Karlsruhe | 3 Karlsruhe |
| front camera (x, y, z) in base_frame (m) | (0.207, 0.005, −0.183) | (0.140, 0.008, −0.203) | (0.139, 0.010, −0.201) |
| optical axis yaw (+ left) | 0.97° | 0.04° | 0.02° |
| pitched up | 0.11° | 0.04° | 0.37° |
| the harness's height (m) | 2.017 | 1.997 | 1.999 |
| the map's height, val scene medians (m): min / median / max | 1.825 / 1.868 / 1.914 | 1.826 / 1.880 / 1.976 | 1.862 / 1.897 / 1.920 |
| run keyframes, low / high mode | 88 / 73 | 31 / 19 | 11 / 1 |

Rig and city are confounded on val: Frankfurt is only on 53df1231, and dbb25dcf has 3 val scenes.

**The camera has four "heights"** (measured):
1. **The harness's: 2.00–2.02 m, fixed per rig.**
    - It is calib z plus a fixed guess of 2.2 m for base_frame.
    - The map puts base_frame 2.05–2.10 m up, so this height is 0.10–0.15 m too high.
    - openpilot never uses it: its warp is a rotation.
2. **The map's: 1.87–1.90 m** in the middle of each rig's scenes, with scene medians from 1.83 to 1.98 m.
    - It is measured per frame against the Lanelet2 map, so it moves with the road: 1.83–1.93 m within the scene §14 plots.
    - It is missing at the 3 keyframes that sit on no lanelet.
3. **The model's: near 1.39 m on 130 keyframes, and near 1.62 m on 93.** Its road head infers this from the image, and it is the one that jumps.
    - In the scene §14 plots, on one rig and a flat map, it reads 1.63 m at the first keyframe and 1.35–1.38 m at the other three.
    - There, the reading drops as the lane-line probability falls from 0.47 to 0.01–0.06.
4. **openpilot's own: 1.22 m,** the camera its training rig carries. No KITScenes camera is near it.

**What sets the model's reading** (measured, `mode_drivers()`).
- **Lane lines, most of all.**
    - The low mode's median lane-line probability is 0.11, against 0.47: AUC 0.18 (CI 0.12–0.24), and 0.23 within speed terciles.
    - 70% of the 166 keyframes where the model sees no lane lines are in the low mode, against 22% of the 49 where it sees weak ones.
- **Then speed and a lead car.**
    - The low mode's median speed is 5.5 m/s, against 8.9: AUC 0.29, and 0.40 within lane-line terciles.
    - The lead probability's AUC is 0.28, and 0.34 within.
- **Not the rig, the city, the road type or the map.**
    - Rig p 0.10 and city p 0.31, or 0.12 and 0.24 holding lane lines and speed. Road type p 0.67.
    - The map's height at the keyframe: AUC 0.59 (0.49–0.68).
    - The map's lane width 10 m ahead does separate the modes, AUC 0.68 (0.61–0.75), but hardly beyond the lane lines: 0.55 within their terciles.
    - The keyframe's position in its scene is borderline: AUC 0.58 (0.50–0.65).
- **The split, weakly.**
    - 38% of overlap_train_val keyframes are in the low mode, against 62% of val's: p 0.031, one of four scene-level tests.
    - Both modes occur in both splits. So OPI's "follows the split, not the rig", read off a few scenes, does not hold on these 140.
- **The mode is steadier within a scene than chance, but not fixed.**
    - Of the 51 scenes with 2 or more keyframes, 22 are all low, 12 all high and 17 mixed.
    - A shuffle mixes a median of 30 (p 0.0005).

**What it does to the speed** (measured): the pose head reads slower in the low mode, on every rig.
- On 53df1231 it reads 0.81 of the pose speed in the low mode, against 0.91 in the high one.
- On 901dfe25 it reads 0.84 against 0.92.
- Within a mode, those two rigs differ by 0.01–0.03, so their 0.9° difference in yaw is not what sets the misread.

**Argued.**
- **The road head infers the camera's height from what the road shows,** lane lines above all.
    - With them it reads near 1.62 m, still 0.26 m under the map's height.
    - Without them it falls to near 1.39 m.
- **The low mode is a floor, not a reading.**
    - Q4's sweep changes the image, and the low mode barely moves: 1.39 m at s = 1, 1.36 m at s = 1.53, where the image shows the road as from 1.23 m.
    - OPI's E5b fits, on one scene of 34 frames (OPI). Through a camera 0.20 m lower, a scene already reading 1.40 m went only to 1.38 m, while two scenes reading 1.61–1.62 m dropped by 0.21–0.22 m.
- **The mode follows the road's content, which persists but can change within a scene.** Keyframes are 9 s apart, in scenes of 10–60 s. That fits the consistency above better than a mode fixed per scene, or one set frame by frame.
- **So the height varies across val because the roads do,** in their markings, speed and traffic. The calibration contributes 2 cm and a 0.9° yaw, and the mode follows neither.

---

## Q3 Speed profiles, and what ADE sees (§16)

**The plots.** `speed_profile(key)` draws one keyframe in four panels over the GT's 4 s, with ADE's 3 s horizon dashed:
1. **The BEV,** with grey connectors from each planned point to the GT's point at the same time. ADE is the mean of their lengths.
2. **The distance travelled against time.** The gap between two curves is the along-track error.
3. **The speed per 0.5 s step,** the sampled waypoints ADE compares, with Q4's t = 0 markers.
4. **The along- and cross-track error against time,** with each plan's ADE.

**What "calibrated" means.** A calibrated plan is the recorded plan with its start speed corrected:
- **`pose`:** scaled by the pose speed over the pose head's speed. A car could do this, since its odometry knows the speed.
- **`oracle`:** scaled so its first step is the GT's. This is an upper bound, and the audit below uses it.
- **`image`:** the plan from the Q4 sweep run nearest the keyframe's s\*. Here the model replans from a start speed that is fixed in the image. Only the sweep's 60 keyframes have one.

**Getting the plots.**
- **In the notebook:** `speed_profile_sheet(mode)` draws a failure mode's keyframes. The committed notebook shows the worst and the two most typical keyframes of each of these four modes:
    - too_slow_cruise;
    - under_brakes;
    - under_accelerates;
    - phantom_brake.
- **For every scene:** set `EXPORT_PROFILES = True` in §1. `export_profiles()` then writes every keyframe's profile, a page each, to `{runs}/kitscenes/_analysis/openpilot/<run_id>/profiles.pdf`.

**What ADE should see, analytically.** Say a plan misreads the start speed by Δv and then holds it. It is Δv · t out at time t. Over ADE's six points (0.5 to 3 s), that averages to 1.75 s × |Δv|. The pose head reads 0.877 of the speed on this run, so ADE should carry a large start-speed term.

**The answer** (measured, on the baseline's 223 keyframes over 140 scenes).
- **ADE is mostly the start speed.**
    - The start speed explains 64% of the summed ADE (CI 59–68%). That is ADE minus the ADE of the oracle plan (`ade_start_H`).
    - ADE ranks keyframes much as |start-speed error| × 1.75 s does: ρ 0.84 (0.78–0.89).
- **ADE hardly ranks the speed profile.**
    - Its ρ with the profile's own error (`vprof_rmse`, after the start is set right) is 0.06 (−0.07 to 0.20).
    - The profile error is in ADE all the same. With the start speed set right, the rest of ADE tracks it at ρ 0.81. It is the start speed's term that covers it.
- **The three groups `ade_blind_spots()` lists:**
    - **inflated: 20 keyframes,** at a median 10.7 m/s. They rank high for their start speed alone. The worst has 5.02 m of ADE, and 0.27 m once the start speed is right.
    - **missed: 36 keyframes,** at a median 2.65 m/s. That is more than chance over all keyframes, but not within speed thirds (p 0.78). So speed alone explains it: ADE is in metres, and a slow keyframe's profile error is small in metres.
    - **masked: 21 keyframes, ADE's real blind spot.** Setting the start speed right raises their ADE by 0.1 m or more.
        - 19 read slow and plan more acceleration than the GT, and the two cancel in position.
        - In 5, the GT brakes and the plan does not. `a5ef9e38/40` scores 0.72 m, lower than 78% of keyframes, while the GT slows from 10.5 to 5.2 m/s and the plan only from 9.6 to 9.3. With the start speed set right, its ADE is 1.45 m.
        - Only 6 of the 21 carry a profile tag.

**What to do about it.**
- **No new tag.** The missed group is explained by speed. The masked one is a property of the score, not of the plan: the profile tags classify the plan's own acceleration, whatever speed it starts at.
- **To score profiles, report `ade_net_H` (ADE with the start speed set right) or `vprof_rmse` beside ADE.** Both are in the notebook's table and its export.

---

## Q4 The speed at t = 0, and the image calibration that reads it right (§15)

### What "the right speed at t = 0" means here

The speed panels in §13, §16 and §17 mark three speeds at t = 0. §13–§16's text calls the first two `>` and `<`, matplotlib's names for the same triangles.
- **▶, the pose speed:** the right speed. The GT's first 0.5 s step agrees with it to 0.1% (median).
- **◀, the pose head's speed:** what the model believes its speed is.
- **◆ (§17) or ◇ (§16), `plan_v0`:** the speed at which the plan starts. The adapter has recorded it since the desire setting was added.

The first plotted points, at t = 0.25 s, are each path's mean over its first 0.5 s step. They differ from the markers by the acceleration within that step, and by being a mean.

"The right speed at t = 0" means ◀ meets ▶: the model reads its own speed right. The plan's first point meeting the GT's is a second target, and zero along-track error at 3 s is a third.

### What the baseline's records say (measured, part A: 205 keyframes at v_go or faster)

Medians of each ratio:

| ratio | all | low mode (113) | high mode (92) |
|---|---|---|---|
| pose head / pose speed | 0.877 | 0.817 | 0.915 |
| pose head / pose speed over the frames the model saw (k−2..k) | 0.882 | 0.822 | 0.918 |
| `plan_v0` / pose head | 1.024 | 1.053 | 0.997 |
| plan's first step / `plan_v0` | 1.010 | 1.010 | 1.011 |
| plan's first step / GT's first step | 0.909 | 0.877 | 0.921 |
| GT's first step / pose speed | 1.001 | 0.999 | 1.002 |

- **The pose speed is right.** The GT's first step matches it.
- **It is not a lag.** The speed over the two frames the model actually saw gives the same ratio (0.882 against 0.877).
- **The pose head reads 12% slow,** 18% in the low mode.
- **The plan starts where the head reads.** In the low mode it starts 5% faster than the head, which makes up part of the gap.
- **The misread follows the height the model reads** (a cross-section, not a counterfactual).
    - Across keyframes, the speed ratio goes as (`op_cam_h` / the map's height)^0.68 (CI 0.62–0.75).
    - Pure geometry would give 1. OPI measured 0.64 with its own height knob.
    - So the speed is misread less than the height. Read through pure geometry, the speed's misread implies a camera at 1.72 m in the low mode and 1.76 m in the high one, where the map says 1.88 m.
- **The misread reaches across the road too.**
    - On the 54 keyframes where the model sees lane lines, its lane width 10 m ahead reads 0.73 of the map's in the low mode (13 keyframes) and 0.81 in the high one (41).
    - It goes as the height ratio^1.33 (1.16–1.54), more steeply than the speed.

### The height knob (verified in the adapter)

**The warp is a rotation only.** It takes the mount's angles from calib.json and nothing tells the model how high the camera sits. So a KITScenes camera 1.88 m up shows the road nearer and narrower than openpilot's own 1.22 m camera would see it.

**`PADDOCK_OPENPILOT_HEIGHT_SCALE = s` puts `diag(1, 1, s)` between the model ray and the mount's rotation:**

```
K @ view_frame_from_device_frame @ rot_from_euler(rpy) @ diag(1, 1, s) @ calib_from_model
```

- **The road plane shows as from a camera at h / s.** At s = h / 1.22, it shows as openpilot's own camera would see it.
- **The horizon stays where it is,** and every pixel keeps its bearing.
- **A vertical object looks 1/s as tall.**
- **So the knob measures what the model makes of road geometry. It does not correct the camera.**
- **At s = 1 the warp is untouched, bit for bit.**
- **Two scales are refused:** a scale that is not positive, and one that pushes a view further past the sensor's edge than the unscaled warp does. KITScenes' three rigs stay entirely on the sensor up to 1.53.

**The sweep points.**
- **1.15 and 1.30 bracket OPI's two crossings** (OPI): a speed ratio of 1 near s ≈ 1.27, and zero along-track error at 3 s near 1.2.
- **1.53 shows the road as from 1.22 m,** openpilot's own height. This reproduces OPI's E5c, which OPI rejected: speed ratio 1.134, ADE 1.885 m (OPI).
- **Each point is the first 60 keyframes by (scene, keyframe)**: 37 scenes, of which 54 keyframes over 35 scenes are moving at v_go or faster.

### What the sweep measures (measured, part B)

**s\*, the scale at which each reading is right**, interpolated in log s between the two sweep points that bracket it, with its median per group:

| target | all (54 keyframes) | low mode (28) | high mode (26) | rig 53df1231 (41) | rig 901dfe25 (9) |
|---|---|---|---|---|---|
| ◀ meets ▶: the head reads the pose speed | **1.22** (1.17–1.26) | 1.25 (1.22–1.30) | 1.14 (1.12–1.21) | 1.24 (1.17–1.30) | 1.20 (1.16–1.25) |
| the plan's first point is the GT's | 1.18 (1.12–1.22) | 1.16 (1.07–1.21) | 1.19 (1.09–1.24) | 1.20 (1.15–1.25) | 1.17 (1.03–1.20) |
| the plan is neither ahead nor behind at 3 s | 1.19 (1.14–1.28) | 1.15 (1.12–1.28) | 1.24 (1.14–1.33) | 1.24 (1.14–1.33) | 1.19 (1.15–1.29) |

- **At s\* the image shows the road as from a median 1.55 m.** That is each keyframe's map height, near 1.88 m, divided by its own s\*. openpilot's own camera sits at 1.22 m.
- **The low mode needs more to read the speed right** (1.25 against 1.14), but not to start the plan right (1.16 against 1.19). In the low mode the plan already starts 5% faster than the head reads.
- **The rigs agree.** dbb25dcf has only one scene in the sweep.

**How the model responds** (median exponents a in reading ∝ s^a, per keyframe):

| reading | all | low mode | high mode |
|---|---|---|---|
| the pose head's speed | 0.54 | 0.71 | 0.43 |
| the plan's first step | 0.56 | 0.65 | 0.43 |
| the model's lane width | 0.41 | 0.53 | 0.32 |
| the road head's height (`op_cam_h`) | −0.23 | −0.06 | −0.40 |
| geometry through the height the road head reads | 0.77 | 0.95 | 0.60 |

- **The speed responds less than the image changes.** Pure geometry would give 1.
- **The road head hardly follows the image down.** At s = 1.53 the image shows the road as from 1.23 m. The road head reads a median 1.38 m, and 1.36 m in the low mode, which reads 1.39 m at s = 1.
- **The height the road head reads does not explain the speed alone.** Geometry through it predicts 0.77, and the speed responds 0.17 less than that: 0.19 less in the low mode, 0.16 in the high one.

### Does reading the speed right help the trajectory? (measured, the sweep's 60 keyframes)

| s | ADE, mean (m) | change from s = 1 (CI) | better on | along-track at 3 s, mean (m) | lateral MAE (m) |
|---|---|---|---|---|---|
| 1 | 1.44 | | | −1.83 (behind) | 0.30 |
| 1.15 | **0.97** | −0.47 (−0.76 to −0.19) | 45 of 60 | −0.35 | 0.31 |
| 1.30 | 1.03 | −0.41 (−0.89 to 0.00) | 41 of 60 | +0.88 (ahead) | 0.32 |
| 1.53 | 1.95 | +0.50 (+0.01 to +0.98) | 23 of 60 | +3.03 | 0.32 |

**Fixing the output does as well as fixing the image.** ADE on the same 60 keyframes:

| image | as the model planned | rescaled by pose speed / head speed |
|---|---|---|
| s = 1 | 1.44 | 0.97 |
| s = 1.15 | 0.97 | 0.88 |
| s = 1.30 | 1.03 | 0.78 |
| s = 1.53 | 1.95 | 0.82 |

The bound, the plan rescaled so its first step is the GT's, is 0.57 m.

- **The rescale is §4's `k_speed`.** It needs only the speed, which a car's odometry knows.
- **The lateral MAE does not move with s.** The misread is in scale along the path, not in the path's shape.

### What it means (argued)

- **To make the model read the speed right, show the road as from about 1.55 m:** the camera's height divided by 1.22.
    - That corrects what the model makes of the road, not the camera, which is 1.88 m up.
    - openpilot's own 1.22 m overshoots. There, the head reads 12–15% fast, and the plan runs 3 m ahead of the GT at 3 s.
- **The misread is the road head's.** It reads the camera lower than it is, 1.39 or 1.62 m against 1.88, and the speed follows that reading, weakly.
    - The low mode is the road head at a floor near 1.38 m, not a reading of the image: it hardly moves when the image does.
    - So the low mode needs the larger correction.
- **For scoring, the output rescale is the simpler fix.** It needs no image model, matches the knob at s = 1.15, and combines with it.

### Assumptions and limits

- **"Right" is the pose speed.** It comes from INS-derived poses at 10 Hz, and has no sign, so a reversing car counts as a misread (`af665a7a/400` is one).
- **s\* is measured on the scored keyframes.** It uses 60 keyframes of the scored set, so a full run at s\* would be tuned on the set it is scored on. None was made. A fair number would take s from scenes outside the scored set, such as KITScenes' train or test split, and then run val.
- **The 60 are the first by (scene, keyframe), not a random draw.** They cover 37 of the 140 scenes, and their ADE at s = 1 (1.44 m) is lower than the full set's (1.60 m).
- **Four points.** s\* is interpolated linearly in log s between them.
    - A keyframe the sweep does not bracket reads `< 1` or `> 1.53`. It still counts in the median, at the end it falls beyond, so the medians are order statistics.
    - That matters little for the speed, bracketed on 51 of 54 keyframes, and most for the along-track target, bracketed on 37.
- **The knob changes only the image's geometry.** The mount's rotation comes from calib.json, as in every run. A lane-line pitch correction barely moves the speed (OPI's R1: ratio 0.861 → 0.862).
- **A focal zoom was tested later,** as the image calibration notebook's `zoom`. Zoomed out to 0.9, it cuts ADE on that notebook's ten turns nearly as much as the height scale does: 0.99 m, against 0.88 m at 1.22 ([the four questions as one problem](#the-four-questions-as-one-problem)).

---

## The four questions as one problem

**The four questions are links in one chain,** from the image the model is handed to the score its plan gets:

| link | question | what goes wrong, as that question found it |
|---|---|---|
| the image → what the model reads | Q2, Q4 | The warp only rotates, so the road shows as from the camera's 1.88 m, where openpilot's own camera sits at 1.22 m. The road head reads the camera at 1.39 or 1.62 m, and the pose head reads the speed 12% slow, 18% in the low mode. |
| what it reads → the plan's speed | Q4 | The plan starts where the pose head reads: 1.02 of it (median). |
| what it sees and is told → the plan's turn | Q1 | Unaided, the model sees the near turns and not the far ones. Sent the route as a desire, it turns later on the near ones, and brakes for the far ones. |
| the plan → the score | Q3 | Over all keyframes, 64% of ADE is the start speed. So a fix to the speed alone can cut ADE a lot, whether or not the plan turns. |

So the plan's error has two parts with different causes (argued): its **scale along the path**, which starts in the image, and its **turn**, which starts in what the model sees and is told.

**[The image calibration notebook](openpilot_drv_supcombo_kitscenes_image_calibration.ipynb) tests both from the image end.**
- **Ten turn keyframes,** five left and five right, chosen to cover every way stock gets a turn wrong, plus two controls.
- **About a hundred image settings:** the height scale, a focal zoom, a horizontal stretch, yaw, pitch and roll, a shift over the road, sampling, tone, model ticks per frame, and blanking a view. Each is a field of one `Setting`, drawn as the model sees it, and can be tweaked by hand.
- **Three other fixes to beat:** the route as a desire (Q1's arm), scales a car could choose from the model's own readings, and rescaling the plan so its first step runs at the car's speed. That rescale is Q3's `pose` calibration, divided by the plan's first step instead of the pose head.
- **The other 29 turn keyframes,** which nothing was chosen on, for every setting a car could use.

**In this section, § numbers and measured numbers are that notebook's,** except where a Q is cited.
- Its node reproduces the recorded runs within 0.03 m over H: stock and the desire to 0.016 m, and the sweep's 1.15 and 1.30 to 0.018 and 0.029 m (measured, its §4 and §9).
- Its CIs resample scenes, as here. The ten are ten scenes. The 29 span 25, and two of those also hold one of the ten.

### The image fixes the speed (measured)

On the ten turns, against stock's 1.60 m:
- **The height scale is the lever.** At Q4's s\* = 1.22, ADE falls to 0.88 m (−0.72, CI −1.11 to −0.31), better on 9 of 10. The plan's start speed goes from 0.87 to 1.01 of the GT's.
- **A focal zoom pulls the same lever.** The model reads distance along the road as height_scale / zoom times stock's (§3). Zoomed out to 0.9, as at a height scale of 1.11, ADE is 0.99 m.
- **Little adds to it.** §8's best cell, height 1.15 with the view pitched 1° down, gives 0.82 m. Choosing on nine turns and scoring the tenth gives 0.95 m (§10), so 0.13 m of that gain came from choosing on these turns.
- **The model's own reading can set the scale.** s = h / `cam_h`, the camera's real height over the height the road head reads, chosen per turn with no GT, gives 0.94 m.

On the other 29, against stock's 1.77 m, the gain is under half as large: 1.44 m at 1.22 (−0.33, CI −0.73 to +0.07), better on 18 of 29.
- **Stock's plans there start nearer the right speed:** at 0.96 of the GT's, against 0.87 on the ten. The pose head reads 0.82 of the car's speed on both.
- **So one scale overshoots there,** to 1.07.
- **Scales chosen per turn from the model's own readings do no better:** −0.20 to −0.32 m.

### The image does not fix the turn (measured)

- **The heading hardly moves.** Over 95 image settings, the heading error over H stays between −21.4° and −15.8°, against stock's −19.4°. At most 6 of the 10 plans turn toward the route, against stock's 5.
- **So ADE nearly halves while the plans turn little more.** What moves is the speed.
    - Setting only the plan's start speed right, the oracle row below, takes 0.62 m off stock's 1.60 m.
    - The image takes off more, 0.72 m at 1.22, so it changes more of the plan than its start.
- **More of the turn in view does not help.** Zooming the wide view out, or turning it toward the route, puts up to 48% of the GT's waypoints inside it, against 30%. Neither ADE nor the heading gets better by more than §6's noise floor, what a half-pixel shift does.
- **The plan comes from the narrow view.** Blanking it adds 9.0 m to ADE. Blanking the wide view changes ADE by −0.14 m.

### On new turns, the output fixes the speed better (measured)

| ADE (m) | ten turns (§12) | 29 turns (§13) |
|---|---|---|
| stock | 1.60 | 1.77 |
| image, height 1.22 | 0.88 | 1.44 |
| image, chosen on nine turns and scored on the tenth | 0.95 | |
| image, scaled per turn by the road head's height (`height from cam_h`) | 0.94 | 1.57 |
| **output: the plan × v / v_plan,** the car's speed over the plan's first step | 1.02 | **0.83** |
| image chosen on the ten, then the output rescale | 0.88 | 0.84 |
| oracle: the plan × the GT's first step / v_plan | 0.98 | 0.74 |

- **On the 29, the rescale gains about three times what the image does:** −0.94 m (CI −1.35 to −0.57) against −0.33, better on 24 of 29. It gains both on the turns stock matched (−0.79 m) and on the rest (−1.33 m).
- **With the rescale, the image adds nothing there:** 0.84 m against 0.83. So on turns it was not chosen on, the image's whole gain in ADE was the start speed (argued).
- **On the ten, the image is a little ahead,** 0.95 m held out against 1.02, and ahead even of the oracle's 0.98. Its plans turn 1–2° more within H than stock's, which a rescale cannot do: it scales the path's length, not its shape. That edge does not carry over to the 29.

### The route turns the plan, later (measured)

- **Sent as a desire, the route makes more plans take the turn:** 8 of 10, instead of 5. The plan's turn over the whole GT goes from 0.41 to 0.81 of the GT's (median).
- **Within H, those plans turn less** (−25.8°, against −19.4°), and ADE rises to 2.11 m.
    - Nearly all of the rise is on the two turns 64–66 m ahead, +2.43 m on average. On the other eight it is +0.04 m.
    - That is Q1's split at 40 m, on these ten.
- **Sent only on turns under 40 m ahead, the far turns' cost goes, and the later turn stays.** These rows are composed from the runs, not run, and the 40 m is Q1's, chosen after seeing all 39 turns:

    | | ten: ADE (m) | ten: plans toward the route | 29: ADE (m) |
    |---|---|---|---|
    | stock | 1.60 | 5 | 1.77 |
    | the desire under 40 m | 1.63 | 7 | 1.69 |
    | image chosen on the ten | 0.82 | 5 | 1.44 |
    | the same, with the desire under 40 m | 1.19 | 8 | 1.41 |
    | output rescale | 1.02 | 5 | 0.83 |
    | the same, with the desire under 40 m | 1.31 | 7 | 0.92 |

- **With the speed fixed, the gated desire costs more on the ten than it did alone:** +0.37 m with the image and +0.29 m with the rescale, against +0.03 m. On the 29 it gains nothing: −0.03 and +0.09 m, against −0.08 m alone.
- **Within H, the gated plans turn less than stock's:** −24.6° to −26.8°, against −19.4°, on the ten, and −9.3° to −11.0°, against −4.3°, on the 29.

### What it means (argued)

1. **The speed and the turn are two problems.** On the ten, each fix moves one and leaves the other.
    - The image and the rescale bring the plan's start speed from 0.87 of the GT's to 1.01 at a height scale of 1.22, or 1.02 rescaled. At most one more plan turns toward the route, and in §12's rows the plans make 0.40–0.46 of the GT's turn, against stock's 0.41.
    - The desire brings 8 of 10 plans to turn toward the route, making 0.81 of the GT's turn, and moves the start speed only to 0.93.
    - So no one knob solves both, and a score has to show both (point 5).
2. **Fix the speed at the output.**
    - The rescale reads the speed per keyframe, from odometry, as a car could. It is chosen on nothing, and on the turns nothing was chosen on, it gained about three times what the image did.
    - The rescale sets the plan's own start. The image can only set what the pose head reads, and the plan does not start exactly there: 5% above it in the low mode, level with it in the high one (Q4). Searched per turn until the head read the car's speed, the 29's plans started 11% fast.
3. **Keep the image fix for what the output cannot reach.**
    - The rescale corrects one number, the plan's scale. The height scale corrects what the model reads from the road: the pose head, from 0.82 to 0.95 of the car's speed at 1.22 on the ten, and, by Q4's exponents, the lane width as well.
    - So the image is the fix to test where the model's other outputs are used, or where the plan's speed changes what the model sees next, as in closed loop.
    - **Use a scale near 1.15–1.22, chosen outside the scored set.** Q4's s\* and §8's best cell were each chosen on what they score, and §10 puts that optimism at 0.13 m on ten turns. A fair number takes s from KITScenes' train or test split, then runs val.
4. **The turn is not an image problem. It is what the model is told, and when.**
    - Over 95 settings, at most one more plan of the ten turns toward the route. The plan comes from the narrow view, and more of the turn in the wide view changes nothing.
    - Told the route at the keyframe, the model reads "a turn starts here" (Q1), and Q1 found half of the 39 turns already underway there. So the plan turns late, and ADE within H charges for that once the speed is right.
    - Gating by distance removes the far turns' cost, not the lateness.
    - So the next test of intent is timing: the desire's rising edge where the driver's turn begins, which for a turn already underway is on the history frames. Those frames would have to carry the command, which `command_source = "gt"` does not do (Q1). Whether turning later matters at all is a closed-loop question (Q1).
5. **Score turns by more than ADE.** On the ten, ADE nearly halved while no more plans turned, and the desire raised ADE while more did. Beside ADE, report:
    - `planned`: whether the plan turns toward the route at all;
    - the heading error over H (`dpsi_err`), negative for too little turn;
    - `turn_frac`: how much of the whole turn the plan makes;
    - ADE with the start speed set right (the oracle row here, Q3's `ade_net_H`), so the speed does not cover the turn.

---

## Reproducing

**The notebook.** `.venv/bin/jupyter execute --inplace` on it re-runs everything from the recorded runs. It executes in about a minute here, without a GPU.
- §1 pins `RUN_ID`. `INTENT_RUN` and `SWEEP` find the other runs by the settings they recorded.
- `FOCUS` restricts every section to one scene or a list of keyframes, and the §17 side by side draws each of them.

**The image calibration notebook.** It needs [`openpilot_drv_supcombo_image_node.py`](openpilot_drv_supcombo_image_node.py) beside it: the adapter, subclassed, which the notebook launches in openpilot's own interpreter as `paddock run` would.
- Every result is cached under `{runs}/kitscenes/_analysis/openpilot/image_calibration/`, so a re-execution takes about a minute on CPU.
- From cold, it runs up to about 1,500 keyframes, at about 4 s each on CPU over §1's five `NODES`: about 20 minutes.

**The runs.** A run's model env comes only from `[models.openpilot.env]`, and exports in the shell never reach the node. So each run uses a copy of `paddock.toml`, kept outside the repo and passed with `PADDOCK_CONFIG`.

The intent arm:

```toml
[benchmarks.kitscenes]
command_source = "gt"

[models.openpilot.env]
PADDOCK_OPENPILOT_DESIRE = "turn"
```

```sh
paddock run --model openpilot --bench kitscenes
```

One sweep point:

```toml
[models.openpilot.env]
PADDOCK_OPENPILOT_HEIGHT_SCALE = "1.15"
```

```sh
paddock run --model openpilot --bench kitscenes --limit 60
```

## Open

- **A held-out s\*.** Choose the scale on scenes outside the scored set, then run val at it.
- **A desire timed to the turn:** the command stamped on the history frames from where the driver's turn begins, so the pulse arrives as the manoeuvre does.
    - Gating by distance alone, composed in [the four questions as one problem](#the-four-questions-as-one-problem), removed the far turns' cost but not the late turn.
    - An arm driven by `nav_text` is unmeasured too.
- **The image against the output, closed loop.** There the plan's speed changes what the model sees next, so the height scale and the rescale may part ways.
- **The `lane_change` control.**
- **The 2.2 m ground offset** is 0.10–0.15 m above what the map implies. openpilot does not use it, but models that read `camera_height()` do.
