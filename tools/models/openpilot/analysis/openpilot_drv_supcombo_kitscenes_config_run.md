# openpilot's driving model on KITScenes: the scene sweep's config, run by paddock

This note records what happened when paddock itself ran the scene-sweep notebook's `CONFIG` on all 223 KITScenes keyframes, and why that run is now this site's default for openpilot on KITScenes. Every measured number comes from [the config-run notebook](openpilot_drv_supcombo_kitscenes_config_run.ipynb) beside it, which recomputes them from the recorded runs each time it executes. The part numbers are its parts, and the figures are there, not here. Keyframes, H, the pose speed, the pose head and the low and high modes mean what they mean in [the four-questions note](openpilot_drv_supcombo_kitscenes_four_questions.md#how-to-read-this-note).

## The short answer

- **Better overall, and adopted.** ADE fell from 1.595 to 1.248 m on the 223 keyframes: −0.347 m (CI −0.551 to −0.136), or −21.8%.
    - That clears the bar fixed before the run, so on this site a plain `paddock run --model openpilot --bench kitscenes` now runs this configuration.
    - Deleting two tables from the site config resets it.
- **The gain is the speed, on fast straights.** Stock's plan started about 9% slow; the config's starts at about the right speed, or slightly above it.
    - Above 6 m/s, ADE falls in every speed band, by half at 12 m/s and over.
    - Below 6 m/s it rises a little.
- **Turns and lateral error got worse.**
    - On the 39 turns, ADE rose by 0.234 m (CI −0.222 to +0.681).
    - On the 14 whose GT is a left turn, it rose by 0.675 m.
    - Lateral error rose by 19% overall.
- **Most of the gain is on keyframes nothing was chosen on, but all of those are straight.** It is −0.540 m on the 133 held out. There is no held-out estimate for turns.

## How to read this note

**Labels.** Each claim carries one, as in the four-questions note:
- **verified**: checked in the code, the data or a run's record;
- **measured**: computed by the notebook from the runs, and re-checked each time it executes;
- **argued**: a reading of the measurements, not itself measured.

**CIs** are 95% bootstrap intervals over scenes (2,000 draws), since keyframes of one scene are not independent. **Changes are config − stock**, in metres of ADE unless named. A keyframe that moves less than 0.05 m is a tie, neither better nor worse.

**Tuned and held out.** A keyframe is tuned if any knob of the config was chosen or scored on it, and held out otherwise. The sets are restated in the notebook's lib (`CHOSEN`, `SCORED`), each from the cell that defines it:
- **sweep60**: the first 60 keyframes by (scene, keyframe), the open-loop notebook's height sweep, where s\* = 1.22 was measured;
- **ten**: the image-calibration notebook's ten turns, about 114 settings each, the height-by-pitch grids among them;
- **five**: five of the ten, where the image-tweak notebook set the pitch, camera, cut and desire by hand;
- **the scene sweep**: dbba76d9, where the config was checked on frames 40 to 159;
- **every turn**: the desire was scored on all 39 turns, by the intent run and the calibration notebook, before the config was fixed.

That makes 90 tuned keyframes over 61 scenes. The other 133 are held out, and all of them are straight (measured, part 3).

## What ran (verified)

| arm | run | front camera | what it sets | ADE (m) |
|---|---|---|---|---|
| stock | `20260930T154018-242869` | ring | nothing | 1.595 |
| intent | `20260930T014333-d329a6` | ring | `command_source = "gt"`, `PADDOCK_OPENPILOT_DESIRE = "turn"` | 1.649 |
| config | `20261005T205400-a369d1` | base | the scene sweep's `CONFIG`, below | 1.248 |

Each is in `results/kitscenes/openpilot/<run_id>/`, with its per-keyframe records under `{runs}`. All three score the same 223 keyframes.

Only stock is valid. Intent and the config read the route from the recorded future (`command_source = "gt"`). Kitscenes stores every such run `valid: false`, and these two were re-stored that way by `paddock run --resume`, after the review on #60. Their samples and metrics did not change.

**The config is the scene sweep's `CONFIG`, knob for knob:**

| `CONFIG` | paddock |
|---|---|
| `camera="base"` | `front_camera = "base"`: the base front camera, 5856x3104, in CAM_FRONT's place |
| `fov_deg=55.0`, at comma's aspect and width | `PADDOCK_OPENPILOT_FOV_DEG = "55"`: a 55° cut, area-resampled to comma's 1164x874 |
| `height_scale=1.22` | `PADDOCK_OPENPILOT_HEIGHT_SCALE = "1.22"`: the road shown as from a camera 1.22 times lower |
| `pitch_deg=-1.7` | `PADDOCK_OPENPILOT_PITCH_DEG = "-1.7"`: the view tilted 1.7° down |
| `desire="turn"` | `PADDOCK_OPENPILOT_DESIRE = "turn"` with `command_source = "gt"`: the route as a turn desire |
| every other knob | its identity, which is paddock's default |

- **ONNX, on CUDA.** `PADDOCK_OPENPILOT_BACKEND = "onnx"` is pinned because the tinygrad backend's warp is compiled for 1928x1208, so it refuses the cut.
- **Paddock's run is the node's** (measured, part 2). On dbba76d9/40 and /130, the two keyframes both ran, paddock's ADE equals the scene-sweep notebook's to the last bit:
    - for the config, 0.5895184985529843 and 2.061547588237582;
    - for stock, 1.2108458054936533 and 2.3175400615269024.
- **The warp stays in the frame, except big_img's edges.** `img` is fully in bounds. `big_img` is 0.926 in bounds: its 58.7° is wider than the 55° cut, so its outer 7.4% is smeared from the cut's border.
- **It took 29 minutes:** 1,746 s in the run's first record. The record now in `results/` is the `--resume` that re-stored it, so its `duration_s` and `started_utc` are that attempt's.

## The gate (verified)

The bar was fixed before the run: the config becomes this site's default only if its `metrics.ade.mean` is below stock's 1.595359 on the same 223 keyframes. It scored 1.248155.

- **This site's `paddock.toml` carries the two pairing tables,** with a comment naming this run and this note. A plain run of openpilot on KITScenes measures the config here, and `--dry-run` names the pairing.
- **Nothing else changed.** The tables reach only openpilot on KITScenes. The code's defaults are stock, so CI and other sites run stock.
- **The default is an analysis run, not a leaderboard row.** The config reads the route from the recorded future, so this run, and every plain run here while the tables are in place, is stored `valid: false`. Stock stays valid, the untuned baseline. Of the three, it is the only row `paddock report` ranks.
- **`paddock render --bench kitscenes` draws stock by default,** since it takes each model's newest valid run. Pass `--run-id` to draw the config.

## What changed (measured)

### Overall, and on what the config never saw (part 3)

| keyframes | n | scenes | stock | config | change (CI) | better / worse |
|---|---|---|---|---|---|---|
| all | 223 | 140 | 1.595 | 1.248 | −0.347 (−0.551 to −0.136) | 129 / 83 |
| tuned | 90 | 61 | 1.558 | 1.495 | −0.063 (−0.404 to +0.259) | 49 / 39 |
| held out | 133 | 94 | 1.620 | 1.081 | −0.540 (−0.810 to −0.287) | 80 / 44 |
| held out, in scenes with no tuned keyframe | 110 | 79 | 1.620 | 1.053 | −0.568 (−0.819 to −0.315) | 67 / 35 |
| turns | 39 | 33 | 1.724 | 1.958 | +0.234 (−0.222 to +0.681) | 19 / 19 |
| turns not chosen on | 23 | 20 | 1.865 | 2.061 | +0.196 (−0.321 to +0.806) | 11 / 11 |
| straight | 184 | 128 | 1.568 | 1.098 | −0.470 (−0.693 to −0.255) | 110 / 64 |

- **The sets the knobs were chosen on did not all improve.** Sweep60 went −0.213 (CI −0.646 to +0.188), ten +0.280 (CI −0.276 to +0.831), and five +0.260.
- **FDE improved at every horizon:** from 0.828 to 0.618 m at 1 s, and from 2.904 to 2.355 m at 3 s.
- **The gain is along the path.** Longitudinal MAE fell from 1.526 to 1.136 m (−25.5%).
- **Across the path, the config is worse.** Lateral MAE rose from 0.286 to 0.339 m: +0.053 (CI +0.028 to +0.082), or +18.6%. It got worse on 81 keyframes and better on 42.

### By speed (part 6)

The bands are of the pose speed at the keyframe. "Start / GT" is the median of the plan's first step over the GT's, on keyframes where the GT starts at v_go or faster.

| pose speed | keyframes | stock | config | change (CI) | start / GT, stock → config |
|---|---|---|---|---|---|
| 0–2 m/s | 18 | 0.841 | 0.898 | +0.057 (−0.205 to +0.247) | 0.647 → 0.859 |
| 2–4 m/s | 37 | 1.347 | 1.569 | +0.222 (−0.110 to +0.536) | 0.930 → 1.153 |
| 4–6 m/s | 39 | 1.126 | 1.450 | +0.324 (+0.016 to +0.625) | 0.916 → 1.137 |
| 6–8 m/s | 40 | 1.455 | 1.080 | −0.375 (−0.693 to −0.045) | 0.908 → 1.062 |
| 8–10 m/s | 36 | 1.780 | 1.037 | −0.743 (−1.275 to −0.288) | 0.908 → 1.028 |
| 10–12 m/s | 30 | 1.903 | 1.097 | −0.806 (−1.185 to −0.406) | 0.895 → 1.025 |
| 12+ m/s | 23 | 2.934 | 1.485 | −1.449 (−2.312 to −0.571) | 0.913 → 1.027 |

- **The crossing is at 6 m/s.**
    - Below it, the config's start overshoots by 14–15% at 2–6 m/s, and ADE rises.
    - Above it, the start is within 3–6% of the GT's, and ADE falls in every band, by half at 12 m/s and over.
- **The tags agree** (part 6).
    - Where the ego is already yawing (`ego_turning`, 56 keyframes), +0.140; where it is steady (167), −0.511.
    - On the 8 keyframes with clear lane lines, −2.022, better on all 8.

### Turns (parts 6 and 7)

- **Left turns got worst.** By the GT's lateral shape, `turn_L` (14 keyframes) went +0.675 (CI +0.014 to +1.232), with 5 better and 9 worse. It is the only shape whose CI lies above zero. `straight` went −0.939 (−49%) and `curve_L` −0.454.
- **Near turns lose to the image, far turns to the desire.**
    - Intent equals stock to the last bit on all 184 straight keyframes, so on straights the whole change is the image.
    - On turns, intent − stock is the desire alone, on the ring camera. Config − intent is the image, given the desire.

| turns | the desire alone | the image, given the desire | both |
|---|---|---|---|
| all 39 | +0.305 (−0.043 to +0.717) | −0.070 (−0.502 to +0.343) | +0.234 |
| the 30 within 40 m | −0.066 (−0.299 to +0.156) | +0.289 (−0.163 to +0.694); 8 better, 20 worse | +0.223 |
| the 9 beyond 40 m | +1.540 (+0.585 to +2.418); 7 worse | −1.266 (−1.974 to −0.622); 8 better | +0.274 |

### Where it helped and where it hurt (parts 4, 5 and 9)

- **Scenes.** Of 140, 86 got better, 45 worse, and 9 moved less than 0.05 m.
    - The 12 most improved average 7.6–14.6 m/s, and all are straights apart from one left turn in 873eebf4. The top three are 2381584f (5.058 → 1.405 m), 97a94a07 (3.827 → 0.693) and c71eb5a2 (3.063 → 0.462).
    - Seven of the 12 most worsened average under 5 m/s. The top three are b8002a79 (two right turns, 1.427 → 3.423), fdcc7efa (a left turn, 0.710 → 2.645) and ed483615 (a straight at 3.8 m/s, 0.857 → 2.683).
- **The largest gain** is 93bcdd59/40, a straight at 13.7 m/s, where ADE fell from 7.854 to 2.491 m. Stock's plan started at 9.7 m/s, the config's at 13.0, and the GT at 13.9. It is the notebook's `FOCUS`.
- **The especially bad keyframes:**
    - **873eebf4/130** is the config's worst plan, at 6.077 m: a left turn 9 m ahead at 2.9 m/s. Stock planned it worse, at 6.578 m.
    - **9ffae20c/310** is the largest loss: +3.951 m (1.120 → 5.071), on a held-out straight at 8.9 m/s. Read off its figure, the GT slows and covers about 24 m in 4 s, where the config plans about 31 m in 3 s.
    - **b8002a79/130** is a right turn 79 m ahead at 14.6 m/s: 0.716 → 3.775 m, with intent at 4.539. Told of the turn, the model turns its plan long before the road does.
    - **3a616875** has all five keyframes tuned (in sweep60), and three of them are among the 15 largest losses. At /40, a straight at 4.75 m/s, ADE went 0.644 → 2.980 m: the config plans about 17 m in 3 s where the GT covers 16 m in 4 s.
- **Stock was already too fast on 5 of the 15 largest losses,** by its own `too_fast_cruise` tag. There the config's faster start adds to an error stock already had.

### What the model read (part 8)

| reading | stock | config |
|---|---|---|
| the road head's camera height, median (m) | 1.423 | 1.372 |
| keyframes read under 1.5 m (the low mode) | 130 of 223 | 223 of 223 |
| pose head / pose speed, median | 0.877 | 1.033 |
| plan's first step / the GT's, median | 0.909 | 1.051 |
| its mean absolute error (m/s) | 0.845 | 0.558 |
| on the 39 turns: first step / the GT's | 0.921 | 1.140 |
| on the 39 turns: its mean absolute error (m/s) | 0.743 | 0.877 |

- **The speed is no longer read slow; it is read slightly fast.** On the turns, which are slow, it is read 14% fast.
- **The camera-height reading is no longer bimodal.** Every keyframe now reads in the low mode.
    - The four-questions note measured that the low mode hardly follows the image, and argued that it is a floor near 1.38 m (its Q4).
    - The base camera's map height over 1.22 is also 1.372 m, so this run cannot tell the floor from a true reading.

**The two cameras' heights** (measured, part 8), over the 223 keyframes:

| camera | calibration (m) | map (m) | map / 1.22 (m) |
|---|---|---|---|
| ring front | 2.017 | 1.875 | 1.537 |
| base front | 1.816 | 1.674 | 1.372 |

The base camera sits 0.20 m below the ring's, by either measure.

## What it means (argued)

- **The config fixes the speed it was built to fix, then overshoots.**
    - The four-questions note's rule is to show the road as from about 1.55 m. s\* = 1.22 was fitted on ring frames, where it does that.
    - On the base camera, 1.22 shows the road from 1.37 m, and a road seen from lower reads as faster. That fits the sign of the over-read: pose head 1.033, plan start 1.051.
    - It does not obviously explain the over-read's pattern: 1.14–1.15 at 2–6 m/s, but about 1.03 above 8 m/s.
    - By the same rule, the base camera wants s ≈ 1.08 (1.674 / 1.55). That is untested, and it would take one more run.
- **ADE rewards the config because ADE is mostly the start speed:** 64% of it, by the four-questions note's Q3.
    - Fast straights carry stock's largest start-speed errors, so fixing the speed there moves ADE most.
    - The lateral error, a smaller share, got worse.
- **The turn results say little about a car.**
    - The desire comes from the GT path (`command_source = "gt"`), an oracle's hint a car would not have.
    - On far turns that hint arrives 40 m or more ahead, which is where it costs most.
    - Every turn was used to choose it, so no turn is held out.
- **The held-out gain is a straight-road gain.** The 133 held-out keyframes are all straight, and straights are where the config helps.
    - The nearest thing to held-out turns is the 23 the image knobs were not chosen on: +0.196 (CI −0.321 to +0.806).
    - Even those had their desire scored.

## Caveats

- **Tuned on the test set.** 90 of the 223 keyframes were used to choose or score a knob. The run was first stored valid anyway, by decision. It is now invalid for its route hint, not for the tuning: a run with the camera knobs alone would still be ranked. For what the config would do on new straights, read the held-out rows, not the headline.
- **Not all of the four-questions note's "other 29" turns are untouched** (verified, against the lib's `CHOSEN`).
    - That note calls them the turn keyframes "which nothing was chosen on". But 6 of them are in sweep60, the 60 keyframes s\* was measured on.
    - Here those 6 count as tuned, which leaves 23 turns not chosen on.
- **big_img is partly filler.** 7.4% of it lies outside the 55° cut and is smeared from the cut's border. Whether the model reads that margin was not tested.

## Reproducing and resetting

**The notebook.** Running `.venv/bin/jupyter execute --inplace` on it reads the three runs' records. It re-executes in seconds, with no GPU and no node.
- Part 1's `RUNS` pins the runs.
- `ON`, `FOCUS` and `OVERRIDES` move every part to another subset, keyframe or threshold.

**The run.** The config comes from two pairing tables in the site config. They reach openpilot on KITScenes and nothing else:

```toml
[models.openpilot.benchmarks.kitscenes.env]
PADDOCK_OPENPILOT_BACKEND = "onnx"
PADDOCK_OPENPILOT_HEIGHT_SCALE = "1.22"
PADDOCK_OPENPILOT_PITCH_DEG = "-1.7"
PADDOCK_OPENPILOT_FOV_DEG = "55"
PADDOCK_OPENPILOT_DESIRE = "turn"

[models.openpilot.benchmarks.kitscenes.settings]
command_source = "gt"
front_camera = "base"
```

```sh
paddock run --model openpilot --bench kitscenes
```

- `paddock.toml.example` carries the same tables, commented out.
- The run was made from a copy of this site's `paddock.toml` with the tables added. The copy is kept at `{runs}/kitscenes/_analysis/openpilot/configs/config_base_turn.toml` and was passed with `PADDOCK_CONFIG`.

**To reset to stock, delete the two tables.** The next run then records stock's config and hashes back to `242869`. That was verified before adoption: with the tables removed, the config paddock would record equals stock's recorded config, hash included.

## Open

- **A height scale for the base camera,** chosen on scenes outside the scored set. By the four-questions note's rule it is about 1.08.
- **The near turns.** Given the desire, the image costs 0.29 m on the 30 near turns, and is worse on 20 of them. Which knob of the image does that is not split out.
- **A desire timed to the turn,** from the four-questions note's Open list, so that far turns stop paying for an early hint.
