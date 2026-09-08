# AutoE2E

Autoware's open end-to-end planner. Camera-only, no HD map at inference: seven
views plus a rendered navigation tile go in, and **64 timesteps x 2 control
signals** come out — longitudinal acceleration and path curvature at 10 Hz, a
6.4 s horizon.

| | |
|---|---|
| Upstream | [autowarefoundation/auto_e2e](https://github.com/autowarefoundation/auto_e2e) |
| Pinned sha | `b6b83fd0` |
| Backbone | SwinV2-Tiny (timm), 7-view feature fusion |
| Views | 7 — hi-res centre + 6 ring. See [views](#views) |
| Output | `[64, 2]` of `[accel m/s², curvature 1/m]` at 10 Hz |
| Benchmarks | `kitscenes` |
| Environment | uv, Python 3.12, torch |

---

## Read this first: there is no public checkpoint

The backbone is timm-pretrained, but **the planner head that turns features into
controls is randomly initialised until a checkpoint is loaded.** A random head
still returns a smooth, finite, plausible trajectory, and every metric
downstream will happily average it into an ADE that looks like a result.

So `policy.load()` refuses to start without `AUTOE2E_CHECKPOINT`:

```
AUTOE2E_CHECKPOINT is unset. AutoE2E's planner head is randomly initialised
without a checkpoint and would still return a smooth, finite trajectory -- an
ADE that looks like a result and is not.
```

Upstream trains and serves through AWS/Flyte and publishes results to its
[DataModelConsole dashboard](https://d2itskdqq39tx1.cloudfront.net/); the
weights are not in the repo and are not on Hugging Face. To run this adapter you
need a checkpoint from that pipeline, in the shape
`Platform/pipelines/inference.py:load_policy` expects:

```python
{"config": {...}, "model_state_dict": {...}}
```

`config` is the `AutoE2E.__init__` kwargs the checkpoint was trained with.
Upstream's `model_kwargs()` filters it against the current signature, so a
checkpoint from before an ABI change fails there with upstream's own error
rather than silently constructing a different model here.

Point `paddock.toml` at it:

```toml
[models.autoe2e.env]
AUTOE2E_CHECKPOINT = "/your/path/autoe2e.pt"
```

`diagnostics()` reports the checkpoint's SHA-256, so a recorded run says which
weights produced it.

**Status: adapter written and contract-tested; never run.** No number in
`results/` comes from this model. That matches how `docs/ROADMAP.md` uses
"done" for M1–M4 — the code does what it claims about data that exists, and
inference has not happened.

---

## Views

AutoE2E reads seven views. The KITScenes harness emits one (`CAM_FRONT`, from
`camera_ring_front`), because the four models that came before this one all read
a single forward camera. So `kitscenes_autoe2e_loader.py` resolves the other six
from the front camera's path and reads the scene's own `calib.json` for all
seven — the same thing `kitscenes_alpamayo_loader.py` does, and for the same
reason. **The harness is not modified, so every other model's recorded numbers
are byte-identical.**

Slot order is upstream's `CAMERA_NAMES`:

| Slot | paddock name | KITScenes sensor |
|---:|---|---|
| 0 | `CAM_FRONT_TELE` | `camera_base_front_center` |
| 1 | `CAM_FRONT` | `camera_ring_front` |
| 2 | `CAM_FRONT_LEFT` | `camera_ring_front_left` |
| 3 | `CAM_FRONT_RIGHT` | `camera_ring_front_right` |
| 4 | `CAM_BACK` | `camera_ring_rear` |
| 5 | `CAM_BACK_LEFT` | `camera_ring_rear_left` |
| 6 | `CAM_BACK_RIGHT` | `camera_ring_rear_right` |

`CAM_FRONT_TELE` has no nuScenes equivalent — it is KITScenes' hi-res centre
camera, and the model was built expecting it, so it is named rather than
dropped. `CAM_FRONT` deliberately maps to `camera_ring_front`, the same forward
camera every other model in the leaderboard is scored on. The stereo front pair
is dropped, as upstream drops it: duplicate forward coverage.

**Slot order is load-bearing.** BEV fusion indexes `camera_tiles` against
`projection` positionally, so slot *i* must be the camera whose matrix is row
*i*. Shuffling one without the other projects each camera through its
neighbour's geometry — it does not raise, and the trajectory still looks
reasonable. `test_view_order_matches_the_declared_cameras` pins `model.yaml`,
`POLICY.cameras` and `VIEW_ORDER` to one sequence.

---

## Controls, not waypoints

AutoE2E does not emit a trajectory. It emits `[64, 2]` of acceleration and
curvature, which `infer` integrates through **upstream's own**
`evaluation.metrics.integrate_trajectory`:

```
v     += a·dt              (floored at 0)
theta += kappa·v·dt
x, y  += v·(cos, sin)(theta)·dt
```

Forward Euler at `dt = 0.1`, `theta0 = 0`, giving ego-centric positions with x
forward and y left — paddock's convention already, so there is no frame
conversion and no place for one to be applied twice.

Reusing upstream's integrator is what makes a paddock ADE comparable to theirs:
their `kitscenes_benchmark.compute_displacement_metrics` scores through the same
function. Fitting a curve to the integrated points, or integrating with a
different `v0`, computes a different metric from identical model output.

**Resampling.** The rollout gives a position every 0.1 s; paddock's grid is
0.5 s, so `resample` takes every fifth point starting at the fifth — exactly on
grid, no interpolation, so nothing smooths the model's own dynamics. 64 steps
yield **12 waypoints out to 6.0 s**; KITScenes scores the first 6 (3 s). An
off-grid `waypoint_dt` raises rather than rounding to a neighbour and reporting a
waypoint as describing an instant half a step from the one it does.

---

## Egomotion history

The model's only motion input: `[64, 4]` of `[speed, accel, yaw_rate,
curvature]`, flattened to the 256-wide ABI.

KITScenes stores **no** velocity, acceleration or curvature columns — upstream
derives all four from `poses.txt` by finite differencing, so this adapter does
too, on the recorded timestamps rather than an assumed 10 Hz.

> **Why not `canbus`?** The KITScenes harness fills only the velocity slot
> (`canbus[13]`). Acceleration (`[7]`) and yaw rate (`[12]`) are left at their
> initialised **zero**, which a model reading all four cannot distinguish from a
> stationary car. Reading `canbus` here would feed the planner three zeroed
> signals and nothing would fail.

Curvature is `yaw_rate / max(speed, 0.1)`, matching upstream's floor. A smaller
floor turns a stopped car's curvature into a spike the planner reads as a hard
turn.

**Padding.** The protocol's observation window is 4 s — 40 rows at 10 Hz — and
the ABI is 64 rows, so rows 0–23 stay zero and the newest step is the last row.
That is upstream's `left_zero_pad_to_64`, and its benchmark manifest parser
rejects any other value, so the padding is part of the protocol rather than a
convenience.

`min_input_hz: 10.0` is declared for the same reason openpilot declares it:
below 10 Hz the differencing measures the sampling gap rather than the motion.
That correctly refuses NAVSIM's 2 Hz OpenScene rather than recording the
artefact.

**History is poses, not forward passes.** `observe` records the pose and
returns — it does not run the backbone, because AutoE2E is a pure function of
its recorded history. A 49-frame KITScenes sample therefore costs **one** forward
pass and seven JPEG decodes, not 49 passes and 343 decodes. The default
`observe` would have computed the same tensor 49 times.

---

## The map tile is zeroed

`map_context` is a `(3, H, W)` raster of the Lanelet2 HD map, rotated so ego
heading points up. Rendering it needs `lanelet2`, which pip cannot supply on
many platforms.

**Without it this adapter sends a zero tile with the validity gate off**
(`map_valid=False`). That is upstream's own no-map path — their dataset does the
same thing under `rasterize_map_at_runtime=False`, and their README says "without
Lanelet2, map tiles fall back to zero tensors and the dataset still functions".

It is not a crash. It is **a different measurement**: the planner runs without
its navigation input, so any number produced this way is AutoE2E-without-a-map
and must be reported as such. `diagnostics()["last_inference"]["map_enabled"]`
records which way a run went.

To enable it:

```bash
conda install -c robostack-staging ros-humble-lanelet2-python   # pip often has no wheel
```

then set `AUTOE2E_ENABLE_MAP=1`. Route conditioning (`route_mask`) is left off
and gated off — paddock has no route source for KITScenes.

---

## Setup

```bash
paddock setup --repo auto_e2e                                  # not yet implemented; clone by hand
bash src/paddock/adapters/models/autoe2e/setup.sh
paddock doctor --model autoe2e
paddock run --model autoe2e --bench kitscenes --limit 5
```

Non-obvious, and repeated in `setup.sh`:

- **Python 3.12 is a ceiling, not a preference.** Upstream's pinned torch build
  publishes no wheels for 3.13+, and pip resolves to a much older torch instead
  of failing, which then breaks on the model's own ops.
- **The KITScenes SDK installs with `--no-deps`**, upstream's instruction: it
  pins versions that conflict with the model's torch.
- `TORCH_CHANNEL` defaults to `cu118`; CPU wheels work for the fixture test and
  are far too slow for a 223-sample sweep.

The sha was bumped from `16d60f0a`. The geometry ABI changed in upstream #77/#107:
the `[B,V,3,4]` `camera_params` kwarg became a `projection` operator, and passing
the old name now raises `TypeError` rather than silently falling back to
`geometry_type="pseudo"` — a learned prior, not calibration. This adapter is
written against the new signature.

> **Upstream's `kit_scenes/README.md` is stale.** Its usage block still passes
> `camera_params=` and unpacks a 3-tuple return. Both are wrong against
> `b6b83fd0`: `camera_params` raises, and infer mode returns a single tensor.
> Follow `model_components/auto_e2e.py`, not that README.

---

## What is tested, and what is not

`tests/test_policy.py` runs in paddock's own environment — no GPU, no dataset, no
weights, none of the model's packages — because that is what a reviewer can
execute on the PR. It covers the pure parts that decide the number: the
resampling grid, the egomotion derivation and its padding, view-slot ordering,
and the refusal to run without a checkpoint.

Not covered, and needing a machine with the model installed:

| | Why not |
|---|---|
| the forward pass | needs torch, timm, weights |
| projection matrices | call upstream's `scale_intrinsic`, which needs torchvision |
| the kinematic rollout | upstream's `integrate_trajectory` |
| whether the ADE is any good | needs a checkpoint |

The first real run should check `diagnostics()`: `v0_from_poses` against
`v0_from_canbus`. The integration's initial speed comes from the differenced
poses, and a gap between the two means the harness and this adapter disagree
about the pose stream — worth seeing rather than averaging away.

---

## Closed loop

`benchmarks: [kitscenes]`, deliberately. Closed loop needs a simulator that
renders all seven views; neuroNCAP's NeuRAD renders the six nuScenes cameras and
has no hi-res centre camera to give, so the pairing is not yet meaningful.

`infer` on a single observation with no history returns
`constant_velocity(obs.speed)` rather than zeros — an all-zero trajectory is a
commanded emergency stop and scores as one.
