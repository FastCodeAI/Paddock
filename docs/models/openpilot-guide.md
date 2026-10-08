# openpilot — Repo Guide

> Paths in this guide are written as placeholders: `<repos>` is
> `[roots.repos]` from your `paddock.toml`, `<data>` a dataset root, and
> `<scratch>` a working area. See [docs/site-setup.md](../site-setup.md).

A working reference for this checkout: what the system is, how it's wired, what the
neural nets consume and emit, and how to run it on sample data.

Snapshot: branch `master`, commit `053d9c446`, working tree clean. `repos.yaml`
now pins `555f48c5d`; both ship the **combined** graph, so §§1–7 below still
describe what a clone at the pin contains. Where the two layouts differ, §6 says
so explicitly.

---

## 1. What this is

openpilot is comma.ai's open-source driver assistance system — an end-to-end
vision stack that drives a car's existing steering/throttle/brake APIs. Feature set:

| Feature | Implemented by |
|---|---|
| **ALC** (Automated Lane Centering) | `modeld` → `controlsd` lateral control |
| **ACC** (Adaptive Cruise Control) | `modeld` + `radard` → `plannerd` long MPC → `controlsd` |
| **FCW** (Forward Collision Warning) | `modeld` hard-brake probability heads |
| **LDW** (Lane Departure Warning) | `plannerd` / `lib/ldw.py` off lane-line predictions |
| **DM** (Driver Monitoring) | `dmonitoringmodeld` → `dmonitoringd` policy |
| **Lane changes / turns** | `desire` input loop, `DesireHelper` state machine |

It targets the comma four / comma 3X / chestnut hardware running AGNOS, but the
whole stack runs on a Linux PC against recorded logs or a simulator.

### 1.1 Repo layout — read this first

The layout is unusual: the git root is a thin shell, and **the actual Python package
lives in the nested `openpilot/` directory**.

```
.                          git root — SConstruct, pyproject.toml, launch scripts
├── openpilot/             the importable `openpilot` package
│   ├── cereal/            capnp message schema + pub/sub messaging + VisionIPC
│   ├── common/            params, transformations, realtime, filters, hardware
│   ├── selfdrive/
│   │   ├── modeld/        ★ the neural networks and their runtime
│   │   ├── controls/      controlsd, plannerd, radard, lat/long controllers
│   │   ├── locationd/     calibrationd, paramsd, torqued, lagd, locationd (EKFs)
│   │   ├── monitoring/    driver-monitoring policy
│   │   ├── selfdrived/    state machine, events/alerts
│   │   ├── car/           card — car abstraction process
│   │   ├── pandad/        panda (CAN interface board) driver
│   │   ├── ui/            raylib UI + soundd
│   │   └── test/          ★ process_replay — offline replay harness
│   ├── system/            camerad, loggerd, sensord, manager, athena, updated
│   └── tools/             replay, sim (MetaDrive), plotjuggler, cabana, lib/
├── panda/                 submodule — CAN interface firmware + safety
├── opendbc_repo/          submodule — car models, DBC files, CarParams/CarState
├── msgq_repo/             submodule — shared-memory IPC + VisionIPC
├── rednose_repo/          submodule — EKF codegen (locationd/paramsd)
├── tinygrad_repo/         submodule — ★ the ML inference framework
└── teleoprtc_repo/        submodule — WebRTC
```

All six submodules are checked out and git-LFS weights are pulled; the tree is
built and both models have been run on real data (§6.1). The one exception is
`big_driving_supercombo.onnx`, deliberately left as a 135-byte LFS pointer since it
needs the chestnut USB GPU.

Generated artifacts live in [openpilot/model_output/](openpilot/model_output/)
(videos, snapshots, cached inference) with the scripts in [tools_local/](tools_local/).
Neither is tracked by git.

---

## 2. The framework

### 2.1 Messaging — cereal / msgq

Every process is an independent OS process. They communicate only through
**cereal**, a Cap'n Proto pub/sub layer over **msgq** (lock-free shared memory) with
a ZMQ fallback (`ZMQ=1`).

- **Schema**: [openpilot/cereal/log.capnp](openpilot/cereal/log.capnp) — one giant
  `Event` union; every service is a member. This file is the source of truth for
  all inputs and outputs in the system.
- **Registry**: [openpilot/cereal/services.py](openpilot/cereal/services.py) — per
  service: publish frequency, whether it's logged, qlog decimation, queue size.
- **API**: `PubMaster(['modelV2'])` / `SubMaster(['carState', ...], poll='modelV2')`.
  `SubMaster` tracks freshness/validity so a process can detect a dead upstream.

Large image buffers do **not** go through cereal. They go through **VisionIPC**
(`msgq_repo`, wrapped by [openpilot/cereal/visionipc.py](openpilot/cereal/visionipc.py)):
`camerad` owns ring buffers of NV12 frames in shared memory, consumers get a raw
pointer. Streams: `VISION_STREAM_NARROW_ROAD`, `VISION_STREAM_WIDE_ROAD`,
`VISION_STREAM_CABIN`, plus encode/driver-view variants.

### 2.2 Process supervision

[openpilot/system/manager/process_config.py](openpilot/system/manager/process_config.py)
declares every process and a predicate for when it should run (`only_onroad`,
`driverview`, `always_run`, …). `manager.py` starts/stops them accordingly.
Entry point: [launch_openpilot.sh](launch_openpilot.sh).

### 2.3 Persistent state — Params

[openpilot/common/params.py](openpilot/common/params.py) is a filesystem-backed
key-value store (`/data/params`, or a tmp dir on PC) used for cross-process and
cross-boot state: `CarParams`, `CalibrationParams`, `LiveParametersV2`, toggles.
`OPENPILOT_PREFIX` namespaces it so multiple instances can coexist.

### 2.4 Build

**SCons** ([SConstruct](SConstruct)). It compiles the C++ (camerad, loggerd,
replay, pandad, msgq) **and** compiles the ONNX models into tinygrad JIT pickles —
see §4.3. Python env is managed by **uv** ([pyproject.toml](pyproject.toml),
Python 3.12.x pinned). Native deps ship as `comma-deps-*` wheels (capnproto,
acados, ffmpeg, zstd, zeromq, raylib, git-lfs) so there is no system package step.

---

## 3. Dataflow

```
                 ┌──────────┐   VisionIPC (NV12 frames)
   3× cameras ──►│ camerad  │──────────┬──────────────────────┐
                 └──────────┘          │                      │
                                       ▼                      ▼
  road + wide road frames        ┌──────────┐          ┌─────────────────┐
                                 │  modeld  │          │dmonitoringmodeld│  cabin frames
                                 └────┬─────┘          └────────┬────────┘
                    modelV2 / drivingModelData /                │ driverStateV2
                    cameraOdometry                              ▼
                                      │                  ┌──────────────┐
                    ┌─────────────────┼──────────┐       │dmonitoringd  │
                    ▼                 ▼          ▼       └──────┬───────┘
              ┌──────────┐     ┌────────────┐  ┌─────────────┐  │
              │ plannerd │     │ locationd  │  │calibrationd │  │ driverMonitoringState
              │ (long MPC│     │ paramsd    │  │ torqued     │  │
              │  + LDW)  │     │ lagd       │  │             │  │
              └────┬─────┘     └─────┬──────┘  └──────┬──────┘  │
       longitudinalPlan │            │ live params    │ extrinsicsCalibration
                        ▼            ▼                ▼         ▼
                      ┌───────────────────────────────────────────┐
                      │              selfdrived                   │  state machine,
                      │           (engage/disengage)              │  events, alerts
                      └────────────────┬──────────────────────────┘
                                       │ selfdriveState
                                       ▼
                                 ┌──────────┐
       radarTracks ──► radard ──►│controlsd │ lat: torque/PID/angle/curvature
                    radarState   │          │ long: PID + MPC
                                 └────┬─────┘
                                      │ carControl
                                      ▼
                          card ──► pandad ──► panda ──► CAN bus ──► car
                            ▲                              │
                            └────── carState ◄─────────────┘
```

Everything is logged by `loggerd` into segments (`rlog.zst` + `*.hevc`), which is
exactly what the replay tools consume.

---

## 4. The models

ONNX files in [openpilot/selfdrive/modeld/models/](openpilot/selfdrive/modeld/models/).
**Which ones you have depends on the revision** — comma has shipped the driving
model both as one combined graph and as a vision/policy pair, and the pair is the
*earlier* of the two (see "Two layouts" at the end of §4.1):

| File | LFS size | Role |
|---|---:|---|
| `driving_supercombo.onnx` | 60.9 MB | combined layout: the whole driving policy in one graph |
| `driving_vision.onnx` + `driving_policy.onnx` | 44.7 MB + 13.4 MB | split layout: the same model as two graphs |
| `big_*` variants | up to 1.76 GB | high-capacity, runs on the "chestnut" external AMD GPU |
| `dmonitoring_model.onnx` | 7.5 MB | driver monitoring |

There is no training code in this repo — comma trains offline and ships weights.

### 4.1 supercombo — the driving model

Run by [modeld.py](openpilot/selfdrive/modeld/modeld.py) at **20 Hz**
(`MODEL_RUN_FREQ`), with a temporal context sampled at **5 Hz**
(`MODEL_CONTEXT_FREQ`) → `frame_skip = 4`.

**Inputs** (shapes from ONNX metadata; comments in
[compile_modeld.py](openpilot/selfdrive/modeld/compile_modeld.py) give the canonical values):

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
[parse_model_outputs.py](openpilot/selfdrive/modeld/parse_model_outputs.py)
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
[modeld.py:44](openpilot/selfdrive/modeld/modeld.py#L44).

**Published messages** (`modeld` → 20 Hz):
- `modelV2` — full `ModelDataV2` ([log.capnp:1045](openpilot/cereal/log.capnp#L1045)):
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

### 4.2 dmonitoring — the driver monitoring model

Run by [dmonitoringmodeld.py](openpilot/selfdrive/modeld/dmonitoringmodeld.py) on
the cabin (driver-facing fisheye) stream.

- **Inputs**: `input_img` — cabin frame warped to `DM_INPUT_SIZE = (1440, 960)`;
  `calib` — the rpy calibration vector.
- **Outputs** — computed twice, once for a left-hand-drive crop and once for RHD
  (`_lhd` / `_rhd` suffixes), plus a `wheel_on_right` probability to pick between
  them. Per side: `face_descs` (face orientation 3 + position 2, with log-σ),
  `face_prob`, `left/right_eye_prob`, `left/right_blink_prob`, `sunglasses_prob`,
  `using_phone_prob`, `sleep_prob` — all sigmoid.
- **Published**: `driverStateV2` ([log.capnp:2122](openpilot/cereal/log.capnp#L2122)).
  [monitoring/policy.py](openpilot/selfdrive/monitoring/policy.py) turns this into
  distraction/awareness state and escalating alerts.

### 4.3 The inference stack — tinygrad, not ONNXRuntime

openpilot does **not** ship an ONNX runtime at drive time. At build time
([modeld/SConscript](openpilot/selfdrive/modeld/SConscript)):

1. `compile_modeld.py` loads the ONNX with tinygrad's `OnnxRunner`.
2. It wraps **two** functions in `TinyJit(..., prune=True)`:
   - `warp` — the NV12 → model-tensor preprocessing, compiled per camera resolution
     (1928×1208 for tici, 1344×760 for mici).
   - `run_policy` — ring-buffer shifting (`img_q`, `big_img_q`, `feat_q`,
     `desire_q`) + the network itself.
3. The traced/pruned JIT graphs are pickled out-of-band (`dump_oob`) into
   `driving_tinygrad.pkl`, then split into 45 MB chunks
   ([common/file_chunker.py](openpilot/common/file_chunker.py)) to stay under
   GitHub's file limit.
4. `modeld` at runtime just unpickles and calls — no graph construction, no ONNX.

Backends are selected by arch:

| Target | Flags |
|---|---|
| comma device (`comma_arm64`) | `DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 OPENPILOT_HACKS=1` — Adreno GPU |
| PC (Linux) | `DEV=CPU:LLVM` |
| macOS | `DEV=CPU` |
| chestnut / big model | `DEV=USB+AMD:LLVM WARP_DEV=QCOM FLOAT16=1 TC_OPT=2` — external AMD GPU over USB |

The big-model path is genuinely exotic: an AMD GPU attached over USB-C, driven
directly by tinygrad (SMU register pokes for temperature/power telemetry live in
`ChestnutState`). `modeld` loads it in a thread with a 60 s timeout and falls back
to the small model if it fails or produces non-finite output.

### 4.4 Non-neural models

Not everything is a net, and it matters for understanding the outputs:

- **`locationd`** — EKF fusing IMU + `cameraOdometry` (+ GNSS), generated by
  **rednose**. Publishes `livePose` / `deviceMotion`.
- **`paramsd`** — EKF learning vehicle steer ratio and stiffness → `vehicleParameters`.
- **`calibrationd`** — estimates device mounting rpy → `extrinsicsCalibration`,
  which is what builds the model warp matrix. Without it the model sees an
  uncorrected view.
- **`torqued`** — online learner for the lateral torque response curve.
- **`lagd`** — estimates actuator lag → `lateralDelay`, which feeds back into the
  model's `action_t` input.
- **`plannerd`** — longitudinal MPC (acados) over the model's plan + radar leads.
- **`radard`** — fuses radar tracks with the model's vision leads → `radarState`.

---

## 5. Tooling

| Tool | Purpose |
|---|---|
| [tools/replay/](openpilot/tools/replay/) | C++ replay: publishes a recorded route's messages + decoded video onto the real msgq bus, so live processes think they're driving |
| [selfdrive/test/process_replay/](openpilot/selfdrive/test/process_replay/) | **Deterministic offline replay of individual processes.** The main way to run models on data without a device |
| [process_replay/model_replay.py](openpilot/selfdrive/test/process_replay/model_replay.py) | Runs `modeld` + `dmonitoringmodeld` over a fixed route, checks timings, plots diffs vs master |
| [tools/sim/](openpilot/tools/sim/) | MetaDrive simulator bridge — closed-loop driving without a car |
| [tools/lib/logreader.py](openpilot/tools/lib/logreader.py) | Read `rlog.zst` segments locally or from the cloud |
| [tools/lib/framereader.py](openpilot/tools/lib/framereader.py) | Decode `*.hevc` to NV12 frames |
| [tools/plotjuggler/](openpilot/tools/plotjuggler/), [tools/jotpluggler/](openpilot/tools/jotpluggler/) | Time-series plotting of logs |
| [tools/cabana/](openpilot/tools/cabana/) | CAN reverse-engineering GUI |
| [tools/clip/](openpilot/tools/clip/) | Render a route segment to video |

**Data format.** A drive = a *route* (`dongleid|timestamp`), split into 1-minute
*segments*. Each segment directory holds `rlog.zst` (all messages), `qlog.zst`
(decimated), and `fcamera.hevc` (road), `ecamera.hevc` (wide), `dcamera.hevc`
(cabin), `qcamera.ts` (low-res). Public CI data is at
`https://commadataci.blob.core.windows.net/openpilotci/` — no auth needed.

---

## 6. Running it on sample data

### 6.1 Status — this checkout is now built and verified

Setup was run on 2026-08-18 and `model_replay.py` completed successfully.

| | |
|---|---|
| submodules (all 6) | ✅ checked out |
| git-LFS weights | ✅ `driving_supercombo.onnx` 60.9 MB, `dmonitoring_model.onnx` 7.5 MB |
| big model | ⏭ deliberately skipped (1.8 GB, needs chestnut USB GPU) |
| venv | ✅ `.venv`, CPython 3.12.13, tinygrad 0.13.0 |
| build | ✅ `scons: done building targets` |
| compiled JITs | ✅ `driving_tinygrad.pkl` 71.8 MB (2 chunks), `dmonitoring_model_tinygrad.pkl` 12.5 MB, `dm_warp_{1928x1208,1344x760}.pkl` |
| model replay | ✅ 60 `modelV2` + 61 `driverStateV2` messages produced |
| hardware | 32 cores, 123 GB RAM, RTX PRO 6000, Ubuntu 24.04 |

The GPU is not used: modeld's PC backend is `CPU:LLVM`. tinygrad has CUDA support
but the SConscript doesn't wire it up for this target, and the JIT pickles are
backend-specific.

### 6.2 Setup (as actually performed)

```bash
cd <repos>/openpilot

# 1. system deps. On this box gcc/g++/make/curl/git were already present, so
#    tools/setup_dependencies.sh was skipped (it otherwise wants sudo for udev rules).
#    But clang IS required and is NOT installed by that script — see §6.8:
sudo apt-get install -y clang

# 2. submodules — required, pyproject path-sources point into them
git submodule update --init --recursive --jobs 6

# 3. python env (uv fetches CPython 3.12.13 + all deps into .venv)
uv sync --frozen --all-extras
export PATH="$PWD/.venv/bin:$PATH" && source .venv/bin/activate

# 4. ★ initialize LFS filters for this repo FIRST — without this the pull is a
#    silent no-op that still exits 0 (see §6.8)
git lfs install --local

# 5. pull weights, skipping the 1.8 GB big model that needs a chestnut USB GPU
git lfs pull --exclude="openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"

# 6. build: C++ targets + compile ONNX → tinygrad JIT pickles
scons -j$(nproc)
```

The whole flow is also wrapped by [tools/op.sh](tools/op.sh) (`tools/op.sh setup`,
then `op build`, `op check`) — but note it runs a plain `git lfs pull` and does not
install clang.

Step 6 is the long one — `compile_modeld.py` traces and compiles the model plus a
warp kernel for each of the two camera resolutions, and validates each JIT by
running it three times, pickle-round-tripping it, and asserting bit-exact replay.
Measured here: **~9 minutes** wall clock on 32 cores. First JIT enqueue is 13.0 s,
dropping to 173 ms once warm.

### 6.3 Option A — run the models on a public route (recommended first run)

This is the shortest path to "models producing outputs on real data", and it needs
no comma account and no hardware.

```bash
python openpilot/selfdrive/test/process_replay/model_replay.py
```

What it does: downloads route `8494c69d3c710e81|000001d4--2648a9a404` segment 4
from public CI storage, decodes frames 0–60 of the road/wide/cabin cameras to NV12,
spins up `modeld` and `dmonitoringmodeld` as real processes against a fake VisionIPC
server, feeds them the logged messages, collects `modelV2`, `drivingModelData`,
`cameraOdometry`, `driverStateV2`, prints a timing table, and diffs against the
master reference logs.

Frames are cached in `/tmp/model_replay_cache` so reruns are fast.

**Verified result on this machine.** The segment is a 2023 Toyota RAV4 TSS2
(fingerprinted from the logged CAN), highway cruise at ~25.6 m/s. Timing:

| model | max instant | allowed | average | allowed |
|---|---:|---:|---:|---:|
| `modelV2` | 0.1765 s | 0.05 | 0.1714 s | 0.028 |
| `driverStateV2` | 0.1232 s | 0.05 | 0.1180 s | 0.018 |

**The ❌ timing failures are expected on PC and do not fail the run** — the check is
`assert timings_ok or PC` ([model_replay.py:196](openpilot/selfdrive/test/process_replay/model_replay.py#L196)),
so it exits 0. ~171 ms/frame is ~5.8 Hz against the 20 Hz onroad budget; the CPU
backend is roughly 6× too slow for real driving, which is exactly why the device
uses the QCOM GPU. Irrelevant for offline replay, which is not wall-clock bound.

Two caveats on interpreting the output:

- `Missing encoded frame for camera feed cabinCameraState with frameId: 5972` is a
  benign boundary artifact of trimming to a 60-frame window.
- The master-reference diff printed **empty**. That comparison pulls reference logs
  via `GithubUtils` using `CI_ARTIFACTS_TOKEN` / `GITHUB_COMMENTS_TOKEN`, which are
  unset here — so treat it as *not run*, not as "no differences found".

### 6.3a Sanity-checking the outputs

Timing tables don't prove the model is producing meaningful predictions. Extracting
actual values (script in the scratchpad) gives:

```
 frame      curv    accel  stop    v@1s    x@5s  lead_d lead_p llProb    conf
     1  -0.00001    0.075 False   12.39   48.79   15.62   0.06   0.05   green   <- warmup
     7   0.00016    0.149 False   26.08  100.93  112.62   0.35   0.69   green
    31   0.00029    0.046 False   25.59   99.86  105.38   0.29   0.74   green
    55   0.00069   -0.033 False   25.43   99.12  113.00   0.36   0.81     red

--- detail @ frame 31 ---
position.x[:8]    [ 0.  0.25  1.  2.25  4.  6.26  9.01 12.27]
position.y[:8]    [0. 0. 0. 0. 0. 0. 0.01 0.02]
velocity.x[:8]    [25.63 25.65 25.64 25.64 25.64 25.63 25.64 25.62]
laneLineProbs     [0.704 0.988 0.974 0.299]
leads             [d=105.4m v=26.1m/s p=0.29], [d=104.5m p=0.23], [d=103.9m p=0.22]
desireState       [0.999 0. 0. 0. 0. 0. 0. 0.]
gasPressProbs     [0.951 0.973 0.981 0.985 0.986 0.987]
brake3ms2Probs    [0. 0. 0.001 0.002 0.003]
cameraOdometry    trans=[25.38 0.00 0.03] m/s  rot=[0.002 -0.0001 0.0032] rad/s
```

These are internally consistent, which is the real check:

- `position.x` matches `v · T_IDXS` exactly — at i=1, `T_IDXS[1]=0.0098 s`,
  `25.6 × 0.0098 = 0.25` ✓, at i=3, `0.088 s → 2.25` ✓. The quadratic time spacing
  from §4.1 is visible directly in the trajectory.
- `cameraOdometry.trans[0] ≈ 25.4 m/s` independently agrees with `velocity.x ≈ 25.6`
  — two separate heads (pose vs. plan) converging on the same speed.
- `position.y ≈ 0`, `orientation.z ≈ 0`, curvature ~1e-4 → straight road ✓
- `laneLineProbs [0.70, 0.99, 0.97, 0.30]` → the two ego-lane lines are confident,
  the outer two much less ✓
- `desireState` is 0.999 on `none` — no lane change pending ✓
- High `gasPressProbs` with near-zero hard-brake probs: the human is driving on the
  accelerator, no braking event predicted ✓

Frame 1 is the warmup artifact — `features_buffer` is still zeros, so the velocity
(12.4 vs 25.6) and lead distance (15.6 m vs ~105 m) are meaningless. This is why
`model_replay.py` drops the first sample (`ts = ts[1:]`).

The `confidence` flip to `red` around frame 43 is a normal function of the
disengage-prediction heads, not an error.

### 6.3b Visualizing predictions on video

Scripts in [tools_local/](tools_local/), outputs in [openpilot/model_output/](openpilot/model_output/).
Both models were run over 600 frames (30 s) of the same segment and their outputs
drawn back onto the source video.

```bash
python tools_local/run_and_render.py              # modeld  -> openpilot/model_output/model_overlay.mp4
python tools_local/run_and_render.py --rerender   # reuse cached inference, redraw only
python tools_local/run_dm.py && python tools_local/render_dm.py   # DM -> dm_overlay.mp4
```

| output | contents |
|---|---|
| `model_overlay.mp4` | 30 s, path + lane lines + road edges + lead markers + HUD |
| `dm_overlay.mp4` | 30 s, face box + gaze vector + attention probability bars |
| `modelv2_0_600.pkl` | 600 `modelV2` messages flattened to numpy |
| `driverstate_0_600.pkl` | 600 `driverStateV2` messages |

**Projecting model output into the image.** Model trajectories are in the
calibrated device frame; the mapping to road-camera pixels is the same one the UI
builds in [augmented_road_view.py:144](openpilot/selfdrive/ui/onroad/augmented_road_view.py#L144):

```python
device_from_calib = rot_from_euler(calib.rpyCalib)
view_from_calib   = view_frame_from_device_frame @ device_from_calib
calib_transform   = device_camera.narrow_road.intrinsics @ view_from_calib
u, v = (calib_transform @ [x, y, z])[:2] / (calib_transform @ [x, y, z])[2]
```

Two geometry traps worth knowing, both hit here:

- **The ribbon z-offset is not mirrored.** To draw a line as a filled band, the UI
  offsets `±y_off` but applies `z_off` *equally* to both edges
  ([model_renderer.py:360](openpilot/selfdrive/ui/onroad/model_renderer.py#L360)).
  Mirroring z as well turns the path into a vertical wall climbing into the sky.
  Lane lines hide the bug because they use `z_off = 0`; only the path
  (`z_off = calibration height`, 1.38 m on this route) exposes it.
- **`facePosition` cannot be projected with a pinhole matrix.** It is in undistorted
  coordinates while the cabin camera is a fisheye. `cam_K @ inv(dm_intrinsics)`
  reduces to a pure translation of `+(244, 248)` — the centered 1440×960 crop of the
  1928×1208 frame — which is correct for the *crop* but ignores distortion entirely,
  putting the face box on the passenger headrest. openpilot itself carries only a
  hardcoded approximation for its own 2160×1080 layout
  ([cabin_camera_dialog.py:66](openpilot/selfdrive/ui/onroad/cabin_camera_dialog.py#L66),
  marked `# TODO: replace with distort_points`). Reusing it rescaled works, but note
  the **x sign flips**: the UI renders the cabin view mirrored, the raw frame is not.

**Measured over the 30 s window** (`tools_local/inspect_model.py` prints this):

| quantity | min | mean | max |
|---|---:|---:|---:|
| speed | 26.3 | 65.4 | 71.3 mph |
| desiredCurvature | -0.000 | 0.001 | 0.003 1/m |
| desiredAcceleration | -0.053 | 0.128 | 0.658 m/s² |
| lead distance | 11.0 | 105.1 | 123.6 m |
| modeld inference | 168 | 170 | 374 ms |

Lane-line probs average `[0.56, 0.86, 0.86, 0.40]` — ego-lane lines confident, outer
two much less. `shouldStop` and `hardBrakePredicted` are 0 for all 600 frames, as
expected for uneventful highway cruising. Driver monitoring: `faceProb` 0.96 mean,
`wheelOnRightProb` 0.000 (LHD correctly identified on all 600 frames), 117 ms/frame.

Note `confidence` is `red` on 520/600 frames. That is not an error — it is derived
from the disengage-prediction heads, and this clip is a human driving manually with
`gasPressProbs ≈ 0.98`, which is exactly the condition those heads fire on.

### 6.4 Option B — replay on your own log, programmatically

Best when you want the parsed outputs as data rather than a pass/fail report.

```python
from openpilot.selfdrive.test.process_replay.process_replay import (
    get_process_config, replay_process)
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.openpilotci import get_url

ROUTE, SEG = "8494c69d3c710e81|000001d4--2648a9a404", 4

lr = list(LogReader(get_url(ROUTE, SEG, "rlog.zst")))
frs = {
    'narrowRoadCameraState': FrameReader(get_url(ROUTE, SEG, "fcamera.hevc"), pix_fmt='nv12'),
    'wideRoadCameraState':   FrameReader(get_url(ROUTE, SEG, "ecamera.hevc"), pix_fmt='nv12'),
    'cabinCameraState':      FrameReader(get_url(ROUTE, SEG, "dcamera.hevc"), pix_fmt='nv12'),
}

msgs = replay_process(get_process_config("modeld"), lr, frs)

for m in msgs:
    if m.which() == 'modelV2':
        md = m.modelV2
        print(md.frameId,
              "curv=%.4f" % md.action.desiredCurvature,
              "accel=%.2f" % md.action.desiredAcceleration,
              "lead_x=", list(md.leadsV3[0].x)[:3] if len(md.leadsV3) else None)
```

Swap the URLs for local paths to run on your own recorded segments. To also get the
raw pre-parse network output vector, set `SEND_RAW_PRED=1` — it lands in
`modelV2.rawPredictions`.

The same `replay_process` works for the non-model processes (`controlsd`,
`plannerd`, `locationd`, `radard`, `paramsd`, `torqued`, `calibrationd`,
`dmonitoringd`, `ubloxd`), so you can replay a whole slice of the stack. Some need
seeded Params — use `get_custom_params_from_lr(previous_segment_lr, 'last')`.
See [process_replay/README.md](openpilot/selfdrive/test/process_replay/README.md).

### 6.5 Option C — full-stack live replay

Runs the real processes on the real message bus, with the UI.

```bash
# terminal 1: publish a recorded route onto msgq (incl. decoded video)
openpilot/tools/replay/replay --demo

# terminal 2: watch it
cd openpilot/selfdrive/ui && ./ui.py
```

`--demo` uses a public route; your own routes need
`python openpilot/tools/lib/auth.py` first, or `--data_dir /path/to/routes` for
local segments. Add `--cabin --wide-road` plus `./watch3.py` to see all three
cameras.

To run the actual openpilot processes (not just visualize):
`./tools/sim/launch_openpilot.sh` in another terminal.

### 6.6 Option D — closed-loop simulation

```bash
./openpilot/tools/sim/launch_openpilot.sh   # terminal 1: the stack
./openpilot/tools/sim/run_bridge.py         # terminal 2: MetaDrive bridge
```

Press `2` to engage, `1`/`2` to change set speed, `s` to disengage. This is the only
option that closes the loop — the model's actions actually move the vehicle.

### 6.7 Regression tests

```bash
openpilot/selfdrive/test/process_replay/test_processes.py            # all procs, all cars
openpilot/selfdrive/test/process_replay/test_processes.py --whitelist-procs controlsd
DISABLE_FILEREADER_CACHE=1 ...test_processes.py                      # bypass log cache
```

Paddock's own adapter tests skip whatever needs openpilot in `make check` and
CI: the warp geometry, the cut, and the guards. Those tests have no automated
gate. To run them, put the pinned checkout on `PYTHONPATH` in paddock's env:

```bash
env PYTHONPATH=<openpilot checkout>:<openpilot checkout>/opendbc_repo \
  .venv/bin/python -m pytest src/paddock/adapters/models/openpilot/tests/test_policy.py
```

The one that reads cereal's capnp schemas, `test_desire_slots_are_openpilots_own`,
still skips there. It runs in openpilot's own interpreter.

### 6.8 Snags — the two that actually bit, plus the usual suspects

**1. `git lfs pull` silently does nothing (hit here).** If `git lfs install` was
never run for the repo, the pull prints

```
Skipping object checkout, Git LFS is not installed for this repository.
```

…and **still exits 0**, leaving every `.onnx` as a 130-byte pointer. A zero exit
code means nothing here. Fix with `git lfs install --local`, then re-pull. Always
verify by size:

```bash
ls -la openpilot/selfdrive/modeld/models/*.onnx   # want MB, not ~130 bytes
```

**2. `clang` is required but undocumented (hit here).** The build dies with:

```
FileNotFoundError: [Errno 2] No such file or directory: 'clang'
scons: *** [.../driving_tinygrad.pkl.chunkmanifest] Error 1
```

`tools/setup_dependencies.sh` installs `build-essential` (gcc) and never mentions
clang, but tinygrad's `CPUDevice.__init__` unconditionally compiles four worker
programs (`signal_prog`, `wait_prog`, `timestamp_prog`, `worker_prog`) through
`ClangRenderer` → `ClangCompiler`, which shells out to the `clang` binary
([compiler_cpu.py:22](tinygrad_repo/tinygrad/runtime/support/compiler_cpu.py#L22)).

Two things that look like workarounds but aren't:
- **`CC=gcc` does not work.** The invocation uses clang-only flags —
  `--target=x86_64-none-unknown-elf`, `-ffreestanding -nostdlib`, and writing the
  object to stdout via `-o -`.
- **`DEV=CPU:LLVM` does not avoid it.** There *is* a ctypes-based `LLVMCompiler`
  that uses `libLLVM.so` directly, and libLLVM 17–20 were already present on this
  box — but the `:LLVM` suffix only selects the *kernel renderer*. The device
  bootstrap above still goes through clang.

Fix: `sudo apt-get install -y clang` (18.1.3 here). Also note `scons` exits 0 when
piped into `tail`, so check for `scons: *** ` in the output rather than trusting `$?`.

**Other likely snags:**

- **`uv sync` fails resolving `pandacan`/`opendbc`/`msgq`** — submodules not
  initialized; pyproject path-sources point into them.
- **`ImportError: openpilot.cereal`** — run from the git root with `.venv` active;
  the package root is the nested `openpilot/` dir.
- **`dmonitoringmodeld` won't start via manager on PC** — by design
  (`enabled=(WEBCAM or not PC)`). `process_replay` invokes it directly, which is why
  Option A works regardless.
- **capnp lists don't support slicing.** `md.position.x[:8]` raises
  `TypeError: an integer is required`. Use `np.array(list(md.position.x))[:8]`.
- **A partial build is easy to resume** — `scons -k` (keep going) builds everything
  that isn't blocked, so you can fix one missing tool and only pay for the
  remaining targets.

---

## 7. Where to look next

| Question | File |
|---|---|
| What does message X contain? | [openpilot/cereal/log.capnp](openpilot/cereal/log.capnp) |
| How often is X published? | [openpilot/cereal/services.py](openpilot/cereal/services.py) |
| What processes exist? | [openpilot/system/manager/process_config.py](openpilot/system/manager/process_config.py) |
| Model input/output shapes & indices | [modeld/constants.py](openpilot/selfdrive/modeld/constants.py) |
| Model output decoding | [modeld/parse_model_outputs.py](openpilot/selfdrive/modeld/parse_model_outputs.py) |
| capnp message population | [modeld/fill_model_msg.py](openpilot/selfdrive/modeld/fill_model_msg.py) |
| Preprocessing / JIT compile | [modeld/compile_modeld.py](openpilot/selfdrive/modeld/compile_modeld.py) |
| Camera geometry & warps | [common/transformations/](openpilot/common/transformations/) |
| Control laws | [selfdrive/controls/lib/](openpilot/selfdrive/controls/lib/) |
| Engage/disengage logic, alerts | [selfdrive/selfdrived/](openpilot/selfdrive/selfdrived/) |
| Car-specific interfaces | `opendbc_repo/opendbc/car/` (submodule) |
| Safety model | [docs/SAFETY.md](docs/SAFETY.md), `panda/` (submodule) |

## 5. Paddock adapter: Bench2Drive configuration (September 2026)

What `adapters/models/openpilot/policy.py` does in closed loop, and why.

* **Cameras.** `model.yaml` declares `CAM_OP_ROAD` (1928×1208, fov 40 → the
  `img` slot) and `CAM_OP_WIDE` (fov 119 → `big_img`), the comma-three rig, so
  the driving model sees the geometry it was trained on; `CAM_FRONT` stays
  declared as the single-camera fallback for open-loop sources. The warp was
  verified pixel-exact (horizon rows 47.6 / 151.8 = `MEDMODEL_CY`,
  `(256+47.6)/2`). With the reference 800×450 fov-70 front camera the model
  froze for adjacent parked cars (1711: DS 1.6 → 40.7 after the change).
* **Longitudinal control.** `PADDOCK_OPENPILOT_NATIVE_LONG=1` makes the node
  return `extras.native_long` — a reproduction of openpilot 0.11.2's
  longitudinal planner: candidates from cruise-to-limit, lead-follow (the
  model's `lead` / `lead_prob` on the MPC's safe-distance model) and, in
  `PADDOCK_OPENPILOT_MODE=experimental`, the e2e plan; `accel = min`,
  `should_stop = any`. The Bench2Drive agent then drives throttle from a
  closed-loop speed PID on the planner's target speed instead of the shared TCP
  tracker, whose 0.4 m/s brake rule braked on 92 % of ticks. Without
  `NATIVE_LONG` the node behaves as before.
* **The nudge.** The 0.11.2 model has no launch-from-standstill without a
  driver: once stationary on a clear road its plan stays at ~0 m (verified by
  offline replay with a fresh state). `PADDOCK_OPENPILOT_RESUME_S=3.0` emulates
  the driver's resume: stopped ≥ 3 s with `lead_prob < 0.5` → the e2e candidate
  is dropped and the cruise/lead candidates accelerate until 1 m/s. It never
  steers and never fires with a lead present. Each firing is a rising edge of
  the `resume` column in log.csv; routes are scored unassisted (frozen at the
  first nudge) and assisted \* (full run). See `docs/benchmarks/bench2drive.md`.
* **Not a route follower.** By default no navigation input reaches the model
  (its desire pulses are zeros), so junction turns are missed by construction.
  `PADDOCK_OPENPILOT_DESIRE` changes that; see the next section.

### A route command reaches the desire input only when asked

`desire_pulse` is supercombo's one channel for intent (§4.1): a one-hot over
`log.Desire`, fed as a rising-edge pulse and max-pooled into 25 bins of 0.2 s,
so a pulse stays in view for 5 s. By default the adapter sends none, and every
run recorded before this setting fed zeros. `PADDOCK_OPENPILOT_DESIRE`, under
`[models.openpilot.env]`, sends the observation's route command instead:

| mode | LEFT | RIGHT | STRAIGHT, FOLLOW_REFERENCE |
|---|---|---|---|
| `off` (default) | none | none | none |
| `turn` | turnLeft | turnRight | none |
| `lane_change` | laneChangeLeft | laneChangeRight | none |

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
block, it is also in `config.model_env`, so a desire run gets its own id and
its own leaderboard row. On KITScenes the commands come from
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
`config.model_env`, so a scaled run gets its own id and, over a full set, its
own leaderboard row. Nothing in the record tells a scale measured on the rig
from one fitted to the benchmark's own errors, and the second is tuned on the
test set.

### A view tilted over the camera's, and cut to comma's size

Two more knobs go with the scale. Both were fitted in the KITScenes analysis
notebooks (`tools/models/openpilot/analysis/`) to KITScenes' own keyframes, so
a run with them is tuned on the test set.

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

**What this invalidates.** Any recorded openpilot row produced on the ONNX
backend is no longer bit-reproducible from this tree; the same run today will
plan from a different wide input. Which rows those are is not fully answerable
from the record, because `/diagnostics` was not being fetched when they were
made — that is the same gap as the revision one above, and it is closed the same
way: `run.json` now carries `provenance.policy.backend`. The comma2k19 openpilot
row is the known case, and it is flagged in
[docs/benchmarks/comma2k19.md](../benchmarks/comma2k19.md#recorded-results); the
re-sweep is in the [ROADMAP](../ROADMAP.md).
