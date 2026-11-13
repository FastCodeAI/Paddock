# OpenPilot

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [commaai/openpilot](https://github.com/commaai/openpilot) at `555f48c5d` |
| Adapter | [`adapters/models/openpilot/`](../../src/paddock/adapters/models/openpilot/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_OP_ROAD`, `CAM_OP_WIDE`, `CAM_FRONT` |
| Plan | waypoints, 0.5 s apart, 10 s ahead |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [neuroncap](../benchmarks/neuroncap.md), [bench2drive](../benchmarks/bench2drive.md), [navsim](../benchmarks/navsim.md), [navhard](../benchmarks/navhard.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [nuplan](../benchmarks/nuplan.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

comma.ai's openpilot driving model, `supercombo`. This page says what the model
reads and returns, and then what paddock's adapter does around it in closed
loop. Building openpilot itself is covered by
[openpilot's own documentation](https://github.com/commaai/openpilot/tree/555f48c5d/docs).

## What the model reads and returns

Run by [modeld.py](https://github.com/commaai/openpilot/blob/555f48c5d/selfdrive/modeld/modeld.py) at **20 Hz**
(`MODEL_RUN_FREQ`), with a temporal context sampled at **5 Hz**
(`MODEL_CONTEXT_FREQ`) → `frame_skip = 4`.

**Inputs** (shapes from ONNX metadata; comments in
[compile_modeld.py](https://github.com/commaai/openpilot/blob/555f48c5d/selfdrive/modeld/compile_modeld.py) give the canonical values):

| Input | Shape | Meaning |
|---|---|---|
| `img` | `(1, 12, 128, 256)` | narrow/road camera: 2 frames × 6 planes, YUV420 |
| `big_img` | `(1, 12, 128, 256)` | wide camera, same packing |
| `features_buffer` | `(1, 24, 512)` | past 24 hidden states (recurrent context) |
| `desire_pulse` | `(1, 25, 8)` | one-hot desire history, one-shot rising-edge pulse |
| `traffic_convention` | `(1, 2)` | one-hot LHD / RHD |
| `action_t` | `(1, 2)` | `[lat_action_t, long_action_t]` — the horizon the model should aim at, computed from actuator delay + frame delay |

Image preprocessing is *not* a separate CV step — it's fused into a tinygrad JIT
(`make_warp` / `make_frame_prepare`):

1. Take the raw NV12 buffer pointer directly from VisionIPC (zero copy,
   `Tensor.from_blob`).
2. Apply a **perspective warp** with a calibration-derived 3×3 matrix
   (`get_warp_matrix` from `extrinsicsCalibration` rpy + camera intrinsics), which
   removes device mounting error. Warps into the "med model frame":
   `MEDMODEL_INPUT_SIZE = (512, 256)`, focal 910.
   Y at full res, U/V at half res, nearest-neighbour sampling.
3. `frames_to_tensor` re-packs the 512×256 YUV into 6 planes at 256×128
   (4 interleaved Y subsamples + U + V), then stacks 2 timesteps → 12 channels.

The 8 `desire` slots map to the `Desire` enum in log.capnp:
`none, turnLeft, turnRight, laneChangeLeft, laneChangeRight, keepLeft, keepRight`
(+1 padding). Only a *rising edge* is fed — the model decides when the maneuver is
done.

**Outputs.** The ONNX emits one flat vector; `output_slices` (pickled into the ONNX
`metadata_props`) names each span, and
[parse_model_outputs.py](https://github.com/commaai/openpilot/blob/555f48c5d/selfdrive/modeld/parse_model_outputs.py)
decodes them:

| Head | Shape | Decoding | Contents |
|---|---|---|---|
| `plan` | `(33, 15)` | MDN (μ, σ) | ego trajectory over 33 time indices: position(3), velocity(3), acceleration(3), euler-from-current(3), orientation-rate(3) |
| `lane_lines` | `(4, 33, 2)` | MDN | 4 lane lines |
| `lane_lines_prob` | `(8,)` | sigmoid | existence probability |
| `road_edges` | `(2, 33, 2)` | MDN | left/right road edge |
| `lead` | `(2 hyp → 3 sel, 6, 4)` | multi-hypothesis MDN | up to 3 leads × 6 future times × (x, y, v, a) |
| `lead_prob` | | sigmoid | lead existence |
| `desire_state` | `(8,)` | softmax | current desire (drives the lane-change state machine) |
| `desire_pred` | `(4, 8)` | softmax | predicted future desire |
| `meta` | `(~55,)` | sigmoid | engaged prob; gas/brake disengage + steer-override probs at 2/4/6/8/10 s; hard-brake 3/4/5 m/s² probs; gas-press/brake-press/blinker probs at 0/2/4/6/8/10 s |
| `pose` | `(6,)` | MDN | ego velocity + rotation rate (visual odometry) |
| `wide_from_device_euler` | `(3,)` | MDN | wide-cam extrinsics estimate |
| `road_transform` | `(6,)` | MDN | road-plane transform |
| `action` | `(2,)` | raw | **direct control output**: `[curvature·v², accel]` |
| `hidden_state` | `(512,)` | — | fed back into `features_buffer` next tick |

Time indices are quadratically spaced: `T_IDXS = 10·(i/32)²` seconds and
`X_IDXS = 192·(i/32)²` metres, i ∈ [0, 33).

Note the model is *end-to-end for control*: when the `action` head is present,
`modeld` uses `desired_curvature` and `desired_acceleration` straight from the net
(smoothed by `LONG_SMOOTH_SECONDS = 0.3`); otherwise it falls back to deriving them
from the `plan` trajectory. See `get_action_from_model` in
[modeld.py](https://github.com/commaai/openpilot/blob/555f48c5d/selfdrive/modeld/modeld.py).

**Published messages** (`modeld` → 20 Hz):
- `modelV2` — full `ModelDataV2` ([log.capnp](https://github.com/commaai/openpilot/blob/555f48c5d/cereal/log.capnp)):
  trajectories, lane lines, `leadsV3`, `meta.disengagePredictions`, `action`,
  `confidence` (red/yellow/green).
- `drivingModelData` — a compact subset for the UI/downstream (polyfit paths).
- `cameraOdometry` — the `pose` head as a velocity/rotation measurement, consumed
  by `locationd` and `calibrationd`.

**Two layouts.** Everything above describes the **combined** graph
(`driving_supercombo.onnx`), which is what `repos.yaml`'s pin `555f48c5d`
(2026-08-19) ships. v0.11.1 (`4df40d2c1`, 2026-05-29) ships the same model
**split** in two, and the differences matter to anyone driving it directly:

| | combined | split |
|---|---|---|
| graphs | `driving_supercombo.onnx` | `driving_vision.onnx` → `driving_policy.onnx` |
| vision inputs | — | `{img, big_img}`, uint8 |
| policy inputs | all six above | `{features_buffer, desire_pulse, traffic_convention}` — **no `action_t`** |
| output dtype | float32 | float16 |
| `action` head | yes | no — control comes from the `plan` |
| feature the policy sees | the **previous** frame's `hidden_state` | the **current** frame's |

That last row is the one that bites. `ModelState.run` writes `prev_feat` *after*
the call and `run_policy` pushes it *before*, so the combined graph plans from a
one-tick-stale hidden state; the split layout runs vision first and pushes its
feature between the two calls, so there is no lag to reproduce. The queue depths
differ to match: `frame_skip * D` combined (so `[::frame_skip]` never reaches the
newest entry) versus `frame_skip * (D - 1) + 1` split (so it lands exactly on it).

Paddock's `supercombo.SupercomboSession` / `SplitSession` own one layout each and
`policy._open_session` picks by what is on disk. Both refuse a graph from the
other layout at load time, with both input sets named.

## How paddock drives it in closed loop

What `adapters/models/openpilot/policy.py` does in closed loop, and why.

* **Cameras.** `model.yaml` declares `CAM_OP_ROAD` (1928×1208, fov 40 → the
  `img` slot) and `CAM_OP_WIDE` (fov 119 → `big_img`), the comma-three rig, so
  the driving model sees the geometry it was trained on; `CAM_FRONT` stays
  declared as the single-camera fallback for open-loop sources. The warp was
  verified pixel-exact (horizon rows 47.6 / 151.8 = `MEDMODEL_CY`,
  `(256+47.6)/2`). With the reference 800×450 fov-70 front camera the model
  froze for adjacent parked cars (1711: DS 1.6 → 40.7 after the change).
* **Longitudinal control.** `model.yaml` declares
  `PADDOCK_OPENPILOT_NATIVE_LONG=1`, which makes the node
  return `extras.native_long` — a reproduction of openpilot 0.11.2's
  longitudinal planner: candidates from cruise-to-limit, lead-follow (the
  model's `lead` / `lead_prob` on the MPC's safe-distance model) and, in
  `PADDOCK_OPENPILOT_MODE=experimental`, the e2e plan; `accel = min`,
  `should_stop = any`. The Bench2Drive agent then drives throttle from a
  closed-loop speed PID on the planner's target speed instead of the shared TCP
  tracker, whose 0.4 m/s brake rule braked on 92 % of ticks. No other
  benchmark reads it. It is declared because it is part of how openpilot
  drives, and it is not tuning because it moves the score: a site `"0"` puts
  the shared tracker back and makes the run `valid: false`. It needs the onnx
  backend, which `"auto"` resolves to at the pinned sha; `load()` refuses
  tinygrad while it is on.
* **The nudge.** The 0.11.2 model has no launch-from-standstill without a
  driver: once stationary on a clear road its plan stays at ~0 m (verified by
  offline replay with a fresh state). `PADDOCK_OPENPILOT_RESUME_S` (3.0 unless
  set, so on with the declared planner) emulates
  the driver's resume: stopped ≥ 3 s with `lead_prob < 0.5` → the e2e candidate
  is dropped and the cruise/lead candidates accelerate until 1 m/s. It never
  steers and never fires with a lead present. Each firing is a rising edge of
  the `resume` column in log.csv; routes are scored unassisted (frozen at the
  first nudge) and assisted \* (full run). See `docs/benchmarks/bench2drive.md`.
* **The nudge on the default tracker.** `NATIVE_LONG` with `RESUME_S` also
  swaps the whole longitudinal law for openpilot's planner, so a score from it
  cannot say which of the two moved it. `PADDOCK_OPENPILOT_RESUME_PLAN=1` gives
  the nudge alone, with `NATIVE_LONG` set to `"0"`: after `RESUME_S` stopped with
  `lead_prob < 0.5`, the node returns a launch straight ahead at 2 m/s until the
  car rolls at 1 m/s, and the shared tracker drives it like any plan. Every
  other frame is the model's own plan, untouched. Straight, because at
  standstill the plan reaches ~0.1 m, too short to steer by. Each frame's state
  is in `extras.resume` (`active`, `count`, `still_s`, `lead_prob`). Off by
  default, and ignored under `NATIVE_LONG`, which has its own nudge.
* **Not a route follower.** By default no navigation input reaches the model
  (its desire pulses are zeros), so junction turns are missed by construction.
  `PADDOCK_OPENPILOT_DESIRE` changes that; see the next section.

### A route command reaches the desire input only when asked

`desire_pulse` is supercombo's one channel for intent (see the inputs above): a one-hot over
`log.Desire`, fed as a rising-edge pulse and max-pooled into 25 bins of 0.2 s,
so a pulse stays in view for 5 s. By default the adapter sends none, and every
run recorded before this setting fed zeros. `PADDOCK_OPENPILOT_DESIRE`, under
`[models.openpilot.env]`, sends the observation's route command instead:

| mode | LEFT | RIGHT | STRAIGHT, FOLLOW_REFERENCE |
|---|---|---|---|
| `off` (default) | none | none | none |
| `turn` | turnLeft | turnRight | none |
| `lane_change` | laneChangeLeft | laneChangeRight | none |
| `route` | turnLeft | turnRight | from the route, below |

`route` is the one mode that is not a table. It reads the route frame by frame
and sends each manoeuvre when it is near, for as long as it applies:

- **A lane change** comes from `nav_lane_change` / `nav_lane_change_m`, the
  next lane change on the route and the metres to it, which the Bench2Drive
  agent sends (`adapters/benchmarks/bench2drive/navigation.py`). It is sent
  from 2.5 s of travel ahead at the current speed, or 10 m, whichever is
  further, and held until the car has passed it. So it arrives as exactly one
  edge, as `DesireHelper` sends one per lane change; a second edge would ask for
  a second lane change.
- **A junction turn** is the command where the car is now, or a `nav_maneuver`
  of `TURN_LEFT` / `TURN_RIGHT` within 40 m. Further ahead, the turn desire cost
  ADE on KITScenes: the model cannot see the turn yet. It is held while the
  turn is near or under way, and dropped for one frame every 4 s, so a fresh
  edge is always inside the 5 s the model keeps one. A car that waits at a red
  light before its turn (Bench2Drive route 2215 waited 10 s) would otherwise
  enter the junction with the desire long gone.
- A lane change in reach takes precedence over a turn. Everything else is none.

`turn` is what a route command means, and it is off openpilot's deployment
distribution. `DesireHelper` sends only the lane-change pair, on the blinker
above 20 mph, and no Python or C++ file in openpilot's history names a turn
desire. The code cannot say whether training covered the turn slots; only a run
can. `lane_change` is in distribution but a different manoeuvre, which makes it
the control arm. The table is explicit because the two enums do not line up:
`Command.RIGHT` is 0 and slot 0 is `none`, so a command passed straight through
as a slot would quietly drop every right turn.

The adapter holds the desire level for the frame that carries it, and both
backends turn it into the edge, as openpilot's own `modeld` does. So a
benchmark that stamps a command on one frame delivers it once, on that frame,
and one that stamps every frame delivers it on the first. KITScenes'
`command_source = "gt"` stamps the scored keyframe alone
([The route hint](../benchmarks/kitscenes.md#the-route-hint)).

Every record's extras carry:
- `desire`: `input`, the slot this frame sent, and `state`, the model's own
  8-way `desire_state`;
- `plan_v0`: the plan's speed at t = 0, beside the pose head's.

The mode is recorded in `provenance.policy.desire_mode`. Set in a site's env
block, it is also in `config.model_env`, so a desire run gets its own id. It
stays ranked, with a note on the leaderboard (`ranked_with_note` in `model.yaml`). On KITScenes the commands come from
`command_source = "gt"`, the route signal the benchmark's other route-conditioned
rows had ([The route hint](../benchmarks/kitscenes.md#the-route-hint)).

### A camera higher than openpilot's, shown from openpilot's height

The warp is a pure rotation: `get_warp_matrix` takes the mounting angles and
the intrinsics, and nothing tells the model how high the camera sits. openpilot's
sits 1.22 m over the road (`calibrationd.HEIGHT_INIT`, which the
`road_transform` head reads back on
[comma2k19](../datasets/comma2k19.md)). From a camera h metres up, a road point
lands on the model pixel where openpilot's camera would see the road at 1.22/h
of the distance, so the road looks nearer and narrower than it is.

`PADDOCK_OPENPILOT_HEIGHT_SCALE`, under `[models.openpilot.env]`, takes
s = h / 1.22 and puts `diag(1, 1, s)` between the model ray and the mounting
rotation:

```
K @ view_frame_from_device_frame @ rot_from_euler(rpy) @ diag(1, 1, s) @ calib_from_model
```

A camera 1.87 m over the road is s = 1.87 / 1.22 = 1.53. In the calibrated
frame z points down and the road is the plane z = h, so:

- the road plane comes out as openpilot's camera would have seen it: the model
  pixel that shows the road X m ahead and Y m across from openpilot's camera
  shows this camera's road point at the same X and Y;
- rays level with the camera do not move, so the horizon row stays put and
  every pixel keeps its bearing;
- nothing off the road plane is right: a vertical object looks 1/s as tall.

So the scale measures what the model makes of the road geometry it was trained
on. It is not a correction for the camera. At 1.0, the default, the warp is
`get_warp_matrix`'s, bit for bit. `extras.road.camera_height`, the
`road_transform` head's reading of its own height, shows what the model made of
the scaled view.

There are two refusals:
- a value that is not a positive number stops the node at startup;
- for s > 1 each view is taller and needs more of the sensor. A scale that
  pushes a view further past the sensor's edge than the unscaled warp does is
  refused at the first frame, with the fraction of real pixels before and
  after. Past the edge the warp smears border pixels into the input, and the
  run would measure the smear. On nuScenes' 1600×900 front camera, 1.53 is
  refused for `big_img`. KITScenes' three rigs (3504×2272) stay entirely on the
  sensor up to 1.53.

The margins in `diagnostics()` are the scaled views'. The scale is recorded in
`provenance.policy.height_scale`. Set in a site's env block, it is also in
`config.model_env`, so a scaled run gets its own id and is stored
`valid: false`, a site env override. Nothing in the record tells a scale measured on the rig
from one fitted to the benchmark's own errors, and the second is tuned on the
test set.

### A view tilted over the camera's, and cut to comma's size

Two more knobs go with the scale. The values the config run uses were chosen
in KITScenes analysis notebooks kept outside this repo, by
inspecting one KITScenes scene, dbba76d9, and the patterns the earlier analysis
found, not fitted. That scene is in the scored set, so a run with them is not a
zero-shot measurement.

`PADDOCK_OPENPILOT_PITCH_DEG` tilts the model's view over the camera's,
positive up. It is a rotation about y in the calibrated frame, ahead of the
scale:

```
K @ view_frame_from_device_frame @ rot_from_euler(rpy) @ R_y(p) @ diag(1, 1, s) @ calib_from_model
```

The scale keeps the horizon row; the tilt moves it. At p = -1.7 the model's
horizon row shows a level camera's row `cy + f tan 1.7°`, as if the camera sat
1.7° more nose-down than its calibration says. `R_y` is built in numpy,
not with `rot_from_euler`, so the analysis runs it came from are reproduced bit
for bit.

`PADDOCK_OPENPILOT_FOV_DEG` warps the views from a *cut*: F degrees across
about the principal point, at comma's aspect, resampled by area to 1164×874.
That is the frame comma's EON road camera would have recorded at that field of
view. Without it, the ONNX warp gathers the nearest pixel straight from the
source. On KITScenes' 5856 px base front camera that keeps about one pixel in
six across `big_img` and aliases the rest. With s source pixels per cut
pixel and the box's edge x0 kept in floats, the cut's focal length is f / s and
its principal point (c + 0.5 - x0) / s - 0.5. A box rounded to whole pixels
would move the frame off its K.

A cut narrower than `big_img`'s 58.7° leaves that view's sides past the cut on
purpose. At 55° about 7.6 % of `big_img` is smeared border on a level mount, and
7.4 % with the config run's height and tilt on the base camera. `warp_in_bounds`
says so. `img` (31.4°) stays whole, and a cut that does not hold it whole is
refused when the adapter is built: `5.5` typed for `55` would leave `img` 4 %
real pixels on the base camera, and the run would measure the smear.

The refusals:
- a tilt beyond ±15° or a field of view outside [0, 180) stops the node at
  startup, naming the variable;
- a cut stops the node at load on the tinygrad backend, whose warp is compiled
  for 1928×1208. Set `PADDOCK_OPENPILOT_BACKEND = "onnx"` beside it;
- at the first frame, a camera without square pixels, or a cut wider than the
  camera recorded about its principal point, is refused;
- the scale's guard covers the tilt too, against the cut's edge when there is
  a cut: the composed warp is judged against the warp without the knobs. On
  KITScenes' base camera with a 55° cut, s = 1.22 alone takes `big_img`'s top
  off the cut (89.9 % real pixels against 92.4 %) and is refused. Tilted by
  p = -1.7 as well, it fits (92.7 %).

Both are recorded in `provenance.policy` as `pitch_deg` and `fov_deg`. A cut
also sets `resample_mode` to `"cut"`, adds a `cut` block (size, box, hfov,
vfov), and has its margins measured against the cut. At 0, the defaults,
nothing is composed and nothing is cut: the warp is `get_warp_matrix`'s, bit
for bit.

Like the scale, both knobs describe one camera rig. A site that adopts them
sets them in a pairing table, `[models.openpilot.benchmarks.kitscenes.env]`,
not in `[models.openpilot.env]`, where they would reach comma2k19's camera too.
See [site-setup.md](../site-setup.md#2-configure).

### Which backend ran, and one number it used to change

`_resolve_backend` picks between tinygrad and onnxruntime by **trying an
import**, and `setup.sh` makes a failed `scons` non-fatal — so whether a site
gets the compiled JIT pickles or falls back to the raw ONNX graphs is a
property of that site's build, not of anything in `model.yaml`. That is
deliberate (a box without a working clang should still be able to score
openpilot), but it means the backend is part of a number's provenance.

Until this was fixed, it was part of a number's *value* as well. The tinygrad
path branched on the two-camera comma rig and the ONNX path did not: it built
one camera from the road view and passed that same frame as both `img` and
`big_img`, so on the ONNX backend the wide view — the one the model uses to see
what is beside the car — received a crop of the road camera. Both paths now bind
cameras through one function, and a test asserts the two views receive different
arrays when the rig supplies two.

**What this invalidates: no cell of the support matrix.** The defect needed two
things at once: the ONNX backend, and a rig that supplies two cameras. With one
camera, that frame fed both views on either backend, before the fix and after it.
Today's ONNX path is tested (`test_one_camera_still_feeds_both_views`). The rest
was read, not run: the tinygrad path in `_bind_cameras`, and both paths before
the fix in the fix's diff, `cc57005`. Only Bench2Drive supplies two:
paddock's CARLA rig renders `CAM_OP_ROAD` and `CAM_OP_WIDE`
([rig.py](../../src/paddock/adapters/benchmarks/bench2drive/rig.py)), and no
other benchmark adapter has supplied the pair in any commit. The ONNX path
itself reached `main` together with its fix, in #44 (merged 19 Sep 2026, UTC);
until then `main` ran openpilot through tinygrad alone. Each cell is below,
checked against its own `run.json` for issue #194. In four of those records,
nuScenes', KITScenes', comma2k19's and NAVSIM v2's, `config.cameras` names
`CAM_OP_WIDE`: it is the list in OpenPilot's `model.yaml`, not what the
benchmark gave it. KITScenes' record also says what it gave
(`config.cameras_supplied`); the other three rows say where the count comes
from:

| Cell | Run | Cameras | Why the defect does not reach it |
|---|---|---|---|
| nuScenes | [`20260908T110028-b71f4f`](../../results/nuscenes/openpilot/20260908T110028-b71f4f/run.json) | 1 | nuScenes' camera list ([`CAMERAS`](../../src/paddock/adapters/benchmarks/nuscenes/loader.py)) has never held a comma camera, so of the three OpenPilot asks for only `CAM_FRONT` is there. It also ran on tinygrad's CUDA build, not ONNX (`config.model_env`) |
| KITScenes | [`20261008T004723-a369d1`](../../results/kitscenes/openpilot/20261008T004723-a369d1/run.json) | 1 | `config.cameras_supplied` is `CAM_FRONT` alone |
| comma2k19 | [`20261106T003719-5cde54`](../../results/comma2k19/openpilot/20261106T003719-5cde54/run.json) | 1 | One camera ([comma2k19.md](../benchmarks/comma2k19.md#recorded-results)), and driven after the fix |
| NAVSIM v2 | [`20260911T161513-3a4c90`](../../results/navsim/openpilot/20260911T161513-3a4c90/run.json) | 1 | NAVSIM's camera map (`CAMERA_SLOTS`) has never held a comma camera, so it supplies `CAM_FRONT` alone |
| NAVSIM v1 | [`20260914T181009-dbcef4`](../../results/navsim_v1/openpilot/20260914T181009-dbcef4/run.json) | — | No model ran: it rescores NAVSIM v2's plans |
| Bench2Drive, 50 routes | [`pooled50-40c29d24c37d542a-c7d2b2`](../../results/bench2drive/openpilot/pooled50-40c29d24c37d542a-c7d2b2/run.json) | 2 | Driven 31 Oct 2026 on Paddock `9ae08a7`, which contains the fix |
| Bench2Drive, 220 routes | [`full220-836dda734659a86c-2803bd`](../../results/bench2drive/openpilot/full220-836dda734659a86c-2803bd/run.json) | 2 | Driven 6–7 Nov 2026 on Paddock `0f0b9ef`, which contains the fix |
| neuroNCAP | [`20260905T055713-11ac59`](../../results/neuroncap/openpilot/20260905T055713-11ac59/run.json) | 1 | `config.cameras` is `CAM_FRONT` alone |
| AlpaSim | [`legacy-openpilot_020_30scenes`](../../results/alpasim/openpilot/legacy-openpilot_020_30scenes/run.json) | 1 | `config.cameras` is 1 |

So no cell needs a re-run. Two limits on what the records show:

- The two Bench2Drive records give their Paddock commit and their backend only
  in `notes`, because they were pooled from source runs that kept no
  `run.json`. A run that `paddock run` writes records its backend in
  `provenance.policy.backend`.
- The historical 24- and 26-route Bench2Drive figures on
  [bench2drive.md](../benchmarks/bench2drive.md) are not in `results/`, so they
  cannot be checked this way. No cell shows them.
