<!--
Vendored into paddock so that scaffold.py's contract citations resolve.

Written against NVlabs/alpasim at the sha repos.yaml pins (`alpasim`). It was
authored outside this repo, in a working tree that is not tracked here -- which
is why the original pointer, `paddock-sid/Alpasim/adapters/INTERFACE.md`,
resolved on one machine and nowhere else. `scaffold.py` cites this file
twenty-nine times as the authority for the things most easily got wrong: rig
frame composition, quaternion field order, the empty-trajectory "not ready yet"
semantics, and session lifecycle. A pointer into one person's scratch directory
is how 840 lines end up being re-derived from the proto by the next person.

Paths inside are relative to the AlpaSim clone root unless they start with
`paddock-sid/`. Line numbers refer to the pinned sha; if upstream moves, the
proto is the authority and this document is the explanation.
-->

# The AlpaSim Driver Interface

**Audience:** someone who has never written an autonomous-driving policy and has
never used gRPC. Everything you need to write a custom driver adapter is here.

**How to verify every claim.** Unless a path starts with `paddock-sid/`, all file
paths in this document are **relative to the AlpaSim clone root** (wherever
`[roots.repos]` puts it at your site):

```
paddock-sid/Alpasim/alpasim/
```

So `src/grpc/alpasim_grpc/v0/egodriver.proto` means
`paddock-sid/Alpasim/alpasim/src/grpc/alpasim_grpc/v0/egodriver.proto`.
Line numbers are given as `path:line` and refer to the clone as it exists today.

---

## 0. Two ideas you need first

**What a "driver" is.** AlpaSim replays a recorded driving scene and re-renders
the camera images from wherever *your* car has drifted to. Your driver is the
brain: it is handed camera frames and where the car currently is, and it answers
with a short list of future positions the car should pass through — a
*trajectory*. A separate controller service turns that trajectory into steering
and pedals. You never touch steering directly.

**What gRPC is.** gRPC is a way for two programs to call each other's functions
over the network. The function signatures are written once in a `.proto` file;
a code generator turns that into Python classes. Your driver is the **server**
(it waits and answers); the AlpaSim runtime is the **client** (it calls). Every
call here is *unary* — one request message in, one response message out. There
are no streams.

Two generated Python modules matter:

- `*_pb2.py` — the **message** classes (the data: `DriveRequest`, `Trajectory`, …)
- `*_pb2_grpc.py` — the **service** classes: `EgodriverServiceServicer` (the base
  class you subclass to build a server) and `EgodriverServiceStub` (what the
  runtime uses to call you).

---

## 1. Where the interface is defined

All protos live in `src/grpc/alpasim_grpc/v0/`. There are nine; three concern a
driver.

| `.proto` | Compiled modules | Relevance to a driver |
|---|---|---|
| `egodriver.proto` | `egodriver_pb2.py`, `egodriver_pb2.pyi`, `egodriver_pb2_grpc.py` | **The driver contract.** Defines `EgodriverService` and every request/response. |
| `common.proto` | `common_pb2.py`, `common_pb2.pyi`, `common_pb2_grpc.py` | Shared value types: `Vec3`, `Quat`, `Pose`, `PoseAtTime`, `Trajectory`, `DynamicState`, `VersionId`, `Empty`. |
| `sensorsim.proto` | `sensorsim_pb2.py`, `sensorsim_pb2.pyi`, `sensorsim_pb2_grpc.py` | Camera descriptions (`AvailableCamerasReturn.AvailableCamera`, `CameraSpec`, `ImageFormat`) that arrive inside `DriveSessionRequest`. |

The remaining six — `runtime.proto`, `controller.proto`, `physics.proto`,
`traffic.proto`, `video_model.proto`, `logging.proto` (each with the same
`_pb2.py` / `_pb2_grpc.py` pair) — belong to other services. A driver never
implements or calls them. (`logging.proto` is worth knowing about: the runtime
copies every driver request and response into a `LogEntry` for the run log — see
`src/runtime/alpasim_runtime/services/driver_service.py:85`.)

The package name is `egodriver`, so the fully-qualified service is
`egodriver.EgodriverService` (`src/grpc/alpasim_grpc/v0/egodriver.proto:6,11`).

---

## 2. The contract at a glance

Lifecycle order. Left is the AlpaSim **runtime** (client), right is **your
driver** (server). Arrows point in the direction of the call.

```
   RUNTIME (client)                                  YOUR DRIVER (server)
        |                                                    |
        |                                    [ process starts, binds host:port,
        |                                      registers servicer, loads model ]
        |                                                    |
  ======| STARTUP PROBE — once per process, before any session |=============
        |                                                    |
        |------------- get_version(Empty) ------------------>|
        |<------------ VersionId ----------------------------|
        |                                                    |
  ======| PER ROLLOUT (one scene) =================================
        |                                                    |
        |--- start_session(DriveSessionRequest) ------------>|  session_uuid,
        |      { uuid, seed, scene_id, camera list }         |  seed, cameras
        |<------------ SessionRequestStatus (empty) ---------|  -> build Session
        |                                                    |
        |   .------------------------------------------.     |
        |   |  PER TICK  (every control_timestep_us)   |     |
        |   |                                          |     |
        |   |  (a) camera events fire first, one call  |     |
        |   |      per camera per frame:               |     |
        |------- submit_image_observation(RolloutCameraImage) ->|  decode + cache
        |   |<--------------- Empty --------------------------|
        |   |                                          |     |
        |   |  (b) then the policy event:              |     |
        |------- submit_egomotion_observation(RolloutEgoTrajectory) ->| poses +
        |   |<--------------- Empty --------------------------|  velocities
        |   |                                          |     |
        |------- submit_route(RouteRequest)  [if a route generator] ->| nav goal
        |   |<--------------- Empty --------------------------|
        |   |                                          |     |
        |------- submit_recording_ground_truth(GroundTruthRequest) ->| (optional,
        |   |<--------------- Empty --------------------------|   usually ignored)
        |   |                                          |     |
        |   |  === BARRIER: runtime waits for every    |     |
        |   |      observation above to be ACKed ===   |     |
        |   |                                          |     |
        |------- drive(DriveRequest) --------------------->|  RUN THE MODEL
        |   |          { time_now_us, time_query_us }   |     |
        |   |<--- DriveResponse { trajectory, debug, ----|  waypoints
        |   |                     terminate_session }    |     |
        |   `------------------------------------------'     |
        |                                                    |
        |--- close_session(DriveSessionCloseRequest) ------->|  drop Session
        |<------------ Empty --------------------------------|
        |                                                    |
```

Evidence for this ordering:

- Startup version probe: `src/runtime/alpasim_runtime/validation.py:103`
  (`stub.get_version(Empty(), wait_for_ready=True, ...)`).
- Session open/close: `src/runtime/alpasim_runtime/services/driver_service.py:91`
  and `:101`.
- Images are submitted by the camera event, which runs at priority 10;
  the policy event runs at priority 20 — lower runs first
  (`src/runtime/alpasim_runtime/events/base.py:27-39`,
  `src/runtime/alpasim_runtime/events/camera.py:151`).
- Egomotion → route → ground truth → **barrier** → `drive`:
  `src/runtime/alpasim_runtime/events/policy.py:113-161`. The barrier is the
  explicit comment *"all observations (images + egopose + route + GT) must reach
  the driver before we call drive()"* at `policy.py:142-143`, implemented by
  `await ctx.drain_outstanding_tasks()`.

**Important consequence:** the observation RPCs are fired as *concurrent tasks*
and only joined at the barrier. Your servicer must be thread-safe — the reference
driver protects its session map and frame caches with locks
(`src/driver/src/alpasim_driver/main.py:524`,
`src/driver/src/alpasim_driver/frame_cache.py:55`).

---

## 3. The RPC methods

Service `EgodriverService`, defined at
`src/grpc/alpasim_grpc/v0/egodriver.proto:11-20`. All eight are unary-unary.
**A driver must implement all eight** — the generated `EgodriverServiceServicer`
base class returns `UNIMPLEMENTED` for any you skip, and the runtime calls all
except the optional ones noted below.

| Method | Request type | Response type | When the runtime calls it |
|---|---|---|---|
| `get_version` | `common.Empty` | `common.VersionId` | Once at startup, before any session, with `wait_for_ready=True`. Doubles as a readiness probe. |
| `start_session` | `DriveSessionRequest` | `common.SessionRequestStatus` | Once per rollout, immediately after the gRPC channel is up. |
| `submit_image_observation` | `RolloutCameraImage` | `common.Empty` | Once per camera per rendered frame, at the frame's shutter-close time. |
| `submit_egomotion_observation` | `RolloutEgoTrajectory` | `common.Empty` | Once per tick, carrying *every* ego pose since the previous tick (not just the newest). |
| `submit_route` | `RouteRequest` | `common.Empty` | Once per tick — **only if** a route generator is configured. |
| `submit_recording_ground_truth` | `GroundTruthRequest` | `common.Empty` | Once per tick — **only if** `send_recording_ground_truth` is on. |
| `drive` | `DriveRequest` | `DriveResponse` | Once per tick, after the observation barrier. This is the only call that must return real content. |
| `close_session` | `DriveSessionCloseRequest` | `common.Empty` | Once at the end of the rollout. |

`SessionRequestStatus` is an intentionally empty message, reserved for future
extension (`common.proto:80`). Returning a bare `SessionRequestStatus()` is the
correct "OK".

---

## 4. Message reference

Types prefixed `common.` come from `common.proto`; unprefixed ones from
`egodriver.proto`. Field numbers are shown because they are the wire identity —
they must never be reused or renumbered.

### 4.1 Primitive value types (`src/grpc/alpasim_grpc/v0/common.proto`)

| Message | Field | # | Type | Meaning | Example |
|---|---|---|---|---|---|
| `Vec3` | `x`, `y`, `z` | 1–3 | `float` | 3-D vector, **metres** (or m/s, m/s² in `DynamicState`). | `x=12.4, y=-0.3, z=0.0` |
| `Quat` | `w`, `x`, `y`, `z` | 1–4 | `float` | Rotation quaternion. **Note the `w`-first field order.** | `w=1.0, x=0, y=0, z=0` (no rotation) |
| `Pose` | `vec` | 1 | `Vec3` | Translation. Applied **first**, then the rotation (`common.proto:31`). | `(12.4, -0.3, 0.0)` |
| | `quat` | 2 | `Quat` | Rotation. | identity |
| `PoseAtTime` | `pose` | 1 | `Pose` | A pose … | |
| | `timestamp_us` | 2 | `fixed64` | … at this simulation time, in **microseconds**. | `1_500_000` |
| `Trajectory` | `poses` | 1 | `repeated PoseAtTime` | An ordered list of timestamped poses. The single container used for ego history, ground truth *and* your answer. | 7 poses |
| `DynamicState` | `angular_velocity` | 1 | `Vec3` | rad/s, rig frame. | `(0,0,0.05)` = turning left |
| | `linear_velocity` | 2 | `Vec3` | m/s, rig frame. `x` is forward speed. | `(8.3, 0.0, 0.0)` |
| | `linear_acceleration` | 3 | `Vec3` | m/s², rig frame. | `(0.4, 0.0, 0.0)` |
| | `angular_acceleration` | 4 | `Vec3` | rad/s², rig frame. | `(0,0,0)` |
| `VersionId` | `version_id` | 1 | `string` | Free-form build identity. | `"vam-driver-0.3.1"` |
| | `git_hash` | 2 | `string` | Source revision, or `"unknown"`. | `"unknown"` |
| | `grpc_api_version` | 3 | `APIVersion` | `{major, minor, patch}` — set it to `alpasim_grpc.API_VERSION_MESSAGE` (`common.proto:74`). | `0.54.0` |
| `Empty` | — | — | — | No fields. The ACK for every observation RPC. | `Empty()` |

There is **no dedicated speed, heading or yaw field anywhere in the protos.**
Speed comes from `DynamicState.linear_velocity`; heading comes out of
`Pose.quat`. See §6.

### 4.2 `DriveSessionRequest` — sent once, opens a session

`egodriver.proto:37-57`

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Identifies this rollout. **Every later RPC repeats it** — a driver process can host several concurrent sessions. | `"a3f1…"` |
| `random_seed` | 2 | `fixed64` | Seed for reproducible inference. | `4021995312` |
| `debug_info` | 4 | `optional DebugInfo` | `{ scene_id: string }`. Deliberately **absent in benchmark runs to prevent data leakage** (`egodriver.proto:53`). Never make behaviour depend on it. | `scene_id="scene_0042"` |
| `rollout_spec` | 5 | `RolloutSpec` | `{ vehicle: VehicleDefinition }`. | |
| `rollout_spec.vehicle.available_cameras` | 3 | `repeated sensorsim.AvailableCamerasReturn.AvailableCamera` | **The only place you learn what cameras exist.** | see below |

Each `AvailableCamera` (`sensorsim.proto:220-229`):

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `intrinsics` | 1 | `CameraSpec` | Lens model + resolution (below). | |
| `rig_to_camera` | 2 | `common.Pose` | Camera mounting pose. ⚠️ **The proto comment at `sensorsim.proto:223` says the name is wrong and it is really `camera_to_rig`** — "should be camera_to_rig -> silently breaks things". Treat the direction as unverified. | translation `(1.659, -0.014, 1.515)` |
| `logical_id` | 3 | `string` | The camera's name. This string is the key you match on. | `"camera_front_wide_120fov"` |

`CameraSpec` (`sensorsim.proto:143-166`) — the fields a driver actually needs:

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `camera_param` | oneof 2/3/4 | `FthetaCameraParam` \| `OpenCVPinholeCameraParam` \| `OpenCVFisheyeCameraParam` | Lens model. Exactly one is set. Fisheye/f-theta images may need rectifying before a pinhole-trained model sees them. | `opencv_pinhole_param` |
| `logical_id` | 5 | `string` | Camera name again (bookkeeping). | `"camera_front_wide_120fov"` |
| `resolution_h` | 7 | `uint32` | Image height, pixels. | `1080` |
| `resolution_w` | 8 | `uint32` | Image width, pixels. | `1920` |
| `shutter_type` | 9 | `ShutterType` | `GLOBAL`, `ROLLING_TOP_TO_BOTTOM`, … (`sensorsim.proto:56-63`). | `ROLLING_TOP_TO_BOTTOM` |
| `external_distortion` | oneof 10 | `BivariateWindshieldModelParameters` | Optional windshield model. | unset |

Example values above come from the shipped VaVAM camera preset,
`src/wizard/configs/driver/vavam_configs.yaml:19-33`.

### 4.3 `RolloutCameraImage` — one camera frame

`egodriver.proto:63-73`

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Which session. | `"a3f1…"` |
| `camera_image` | 3 | `CameraImage` | The frame. | |
| `camera_image.frame_start_us` | 2 | `fixed64` | Shutter **open** time, µs. | `1_499_970` |
| `camera_image.frame_end_us` | 3 | `fixed64` | Shutter **close** time, µs. **This is the timestamp the frame is filed under** — the reference driver keys its cache on `frame_end_us` (`src/driver/src/alpasim_driver/main.py:860-864`). | `1_500_000` |
| `camera_image.image_bytes` | 4 | `bytes` | The **encoded** image file, not raw pixels. | JPEG byte string |
| `camera_image.logical_id` | 5 | `string` | Which camera. Matches an `AvailableCamera.logical_id`. | `"camera_front_wide_120fov"` |

**Encoding is not in this message.** There is no format field on
`CameraImage`. The encoding is a runtime-side setting,
`image_format: str = "jpeg"  # Literal["jpeg", "png"]`
(`src/runtime/alpasim_runtime/config.py:337`), applied at
`src/runtime/alpasim_runtime/events/camera.py:84,101`. JPEG is the default. A
robust adapter should sniff the bytes (PIL's `Image.open` does this for you)
rather than assume — the reference driver's CPU path does exactly that, while its
optional CUDA path hard-assumes JPEG (§6).

Resolution is **not** in this message either; it comes from
`AvailableCamera.intrinsics` at session start, and is confirmed by the decoded
image itself.

### 4.4 `RolloutEgoTrajectory` — where the car is and how it is moving

`egodriver.proto:75-84`

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Which session. | `"a3f1…"` |
| `trajectory` | 3 | `common.Trajectory` | **Ego pose history.** Each `PoseAtTime` is the active transform `local -> rig_est` — i.e. where the car is in the world frame, as *estimated*. May diverge from truth when error models are on (`egodriver.proto:77-79`). | 1–N poses |
| `dynamic_states` | 5 | `repeated common.DynamicState` | Velocities and accelerations **in the rig frame**. Entries correspond **1:1 with `trajectory.poses`** (`egodriver.proto:80-83`) — the reference driver zips them with `strict=True` and will raise on a length mismatch (`main.py:877-882`). | 1–N states |

**History is real.** The runtime sends *every* pose accumulated since the last
tick, not just the newest one — see the comment and mask at
`src/runtime/alpasim_runtime/events/policy.py:76-89`. With
`pose_reporting_interval_us > 0` the controller emits intermediate poses and they
all arrive here. Your driver accumulates them; the reference driver appends and
re-sorts by timestamp (`main.py:335-340`).

### 4.5 `RouteRequest` — the navigation goal

`egodriver.proto:86-89`, wrapping `Route` (`egodriver.proto:22-27`)

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Which session. | `"a3f1…"` |
| `route.timestamp_us` | 2 | `fixed64` | Time at which the waypoints are valid. | `1_500_000` |
| `route.waypoints` | 1 | `repeated common.Vec3` | Route points **in the rig frame at that timestamp** — i.e. relative to the car, x forward, y left. | `(20.0, 3.5, 0.0)` = 20 m ahead, 3.5 m to the left |

This is a coarse map route, not a plan: typically the driver reduces it to a
single discrete command (left / straight / right).

### 4.6 `GroundTruthRequest` — what the real car did

`egodriver.proto:91-94`, wrapping `GroundTruth` (`egodriver.proto:29-35`)

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Which session. | `"a3f1…"` |
| `ground_truth.timestamp_us` | 2 | `fixed64` | Validity time. | `1_500_000` |
| `ground_truth.trajectory` | 1 | `common.Trajectory` | The path the **real recorded car** followed, in the **rig frame** at that timestamp. | 20 poses |

This is the recorded future. Using it to drive is cheating — the reference driver
logs it and discards it (`main.py:907-912`).

### 4.7 `DriveRequest` — "what should I do now?"

`egodriver.proto:96-107`

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `session_uuid` | 1 | `string` | Which session. | `"a3f1…"` |
| `time_now_us` | 3 | `fixed64` | Current simulation time, µs. The tick's `step_start_us`. | `1_500_000` |
| `time_query_us` | 4 | `fixed64` | The time the controller will next need a pose for — `time_now_us + control_timestep_us` (`policy.py:60,161`). | `2_000_000` |
| `renderer_data` | 5 | `bytes` | Optional opaque blob the renderer produced this tick. Free-form; the reference driver ignores it. | `b""` |

There is a **nested, unused** `RolloutDriveRequest` message at
`egodriver.proto:97-100` with its own `time_now_us`/`time_query_us`. No field
references it. The top-level fields 3 and 4 are the live ones — the runtime
comment calls them "old and new fields for backward compatibility"
(`driver_service.py:221`).

### 4.8 `DriveResponse` — your answer

`egodriver.proto:109-129`

| Field | # | Type | Meaning | Example |
|---|---|---|---|---|
| `trajectory` | 2 | `common.Trajectory` | **The plan.** Poses are the active transform `local -> rig_est` produced by the driver (`egodriver.proto:110`). See §5. | 7 poses |
| `debug_info` | 3 | `DebugInfo` | Optional diagnostics. | |
| `debug_info.unstructured_debug_info` | 1 | `bytes` | Anything you like, with your own (de)serialisation (`egodriver.proto:113-115`). The reference driver `pickle`s a dict here. | pickled dict |
| `debug_info.sampled_trajectories` | 2 | `repeated common.Trajectory` | Every candidate your model sampled, so scorers can measure sample spread (e.g. minADE) (`main.py:1002-1006`). | 1 trajectory |
| `terminate_session` | 4 | `bool` | `true` ends the rollout **immediately**, without finishing the current step (`egodriver.proto:123-128`). The runtime raises `EndSimulationException` (`policy.py:167-171`). | `false` |

**Returning an empty `Trajectory()` is legal and is the correct "not ready yet"
answer.** The reference driver does this before it has enough camera frames or
any ego pose (`main.py:930-955`).

---

## 5. Trajectory format

### Coordinate frame

AlpaSim defines four frames (`CONTRIBUTING.md:74-92`, in the AlpaSim clone):

- **`local`** — an inertial ENU frame, fixed per scenario. The "world".
- **`rig`** — body-fixed: **x forward, y left, z up, origin at the centre of the
  rear axle projected onto the ground** (`CONTRIBUTING.md:80-84`).
- **`aabb`** — bounding-box frame; irrelevant to drivers.
- **`ecef`** — global WGS84; irrelevant to drivers.

Because proprioception is noisy, the ego pose you receive is in an *estimated*
frame, written `rig_est` — equivalently `local -> rig_est`. **The runtime, not
you, converts your answer back to the true frame**
(`CONTRIBUTING.md:89-92`; implemented at `policy.py:208-226`,
`transform_trajectory_from_noisy_to_true_local_frame`). You work entirely in the
estimated world and can ignore that noise exists.

So there are two frames in play in one tick:

| | Frame | Read it as |
|---|---|---|
| Model *thinks* in | `rig_est` | offsets from the car: +x ahead, +y left |
| `DriveResponse.trajectory` must be in | `local` (as `local -> rig_est` poses) | absolute positions in the scene |

The bridge is one line: take the ego pose at your planning instant `t0`, and
left-multiply each rig-frame waypoint by it. `main.py:102-147`
(`_rig_est_waypoints_to_local_trajectory`) is exactly that, and it is the single
function every adapter has to get right.

### Units and conventions

| Quantity | Unit |
|---|---|
| Position (`Vec3`) | metres |
| Timestamp (`timestamp_us`) | microseconds, `fixed64`, simulation clock |
| Velocity / acceleration | m/s, m/s², rig frame |
| Rotation | quaternion, **`w` first** in the proto (`common.proto:17-22`) |
| Angles (internal) | radians |

### Shape of the returned trajectory

```
DriveResponse.trajectory.poses
  = [ anchor_pose_at_t0 ] + [ waypoint_1, waypoint_2, ..., waypoint_T ]
```

- **The first pose is the anchor**: your ego pose at the planning instant `t0`,
  timestamped `t0` — not a future waypoint (`main.py:133-135`).
- The remaining `T` poses are the plan, spaced `1 / output_frequency_hz`
  apart, timestamps `t0 + k · step_us` (`main.py:1085-1093`).
- Total length is therefore `T + 1`.
- `T` and the spacing are **your model's choice** — nothing in the proto fixes
  them. VaVAM emits `T = 6` at 2 Hz.

Two ways to set the timestamps (`main.py:1069-1110`):

1. **Fallback** — leave `model_t0_us`, `pose_local_to_rig_t0`, and
   `waypoint_timestamps_us` unset; the driver anchors at `DriveRequest.time_now_us`
   and spaces waypoints uniformly.
2. **Explicit** — supply all three together (a partial set raises), anchoring at
   the exact camera frame the model actually looked at. Alpamayo models do this;
   VaVAM uses the fallback.

### Worked example

VaVAM, 2 Hz, six waypoints. Suppose at `t0 = 1_500_000 µs` the car sits at
`local` position `(100.0, 50.0, 0.0)` facing +x (identity rotation), and the
model predicts these rig-frame offsets — straight ahead at ~8 m/s, drifting
slightly left:

| k | rig-frame `(x, y)` m | timestamp µs | → `local` position |
|---|---|---|---|
| — (anchor) | — | 1 500 000 | (100.0, 50.0, 0.0) |
| 1 | (4.0, 0.0) | 2 000 000 | (104.0, 50.0, 0.0) |
| 2 | (8.0, 0.1) | 2 500 000 | (108.0, 50.1, 0.0) |
| 3 | (12.0, 0.3) | 3 000 000 | (112.0, 50.3, 0.0) |
| 4 | (16.0, 0.6) | 3 500 000 | (116.0, 50.6, 0.0) |
| 5 | (20.0, 1.0) | 4 000 000 | (120.0, 51.0, 0.0) |
| 6 | (24.0, 1.5) | 4 500 000 | (124.0, 51.5, 0.0) |

Seven poses, 3 s of horizon, 500 000 µs apart. The `local` column is trivial here
only because the anchor rotation is identity; in general it is a full
`pose_local_to_rig_t0 @ waypoint_in_rig` composition.

Each waypoint also needs an orientation. Models that only predict a ground path
derive a heading from the direction of travel — for waypoint `k`, `atan2` of the
delta from waypoint `k-1` (with the origin standing in for `k=0`), then rotate
about z only (`models/base.py:157-192`, `models/base.py:105-134`). The z
coordinate stays 0 for such models.

The unit test that pins this shape down is
`src/driver/src/alpasim_driver/tests/test_drive_response_trajectory.py:60-73`.

---

## 6. VaVAM walkthrough

VaVAM is the bundled reference driver. It is *not* a separate program: it is one
model plugin behind a single generic servicer.

- **Servicer (shared by every model):** `src/driver/src/alpasim_driver/main.py`
  — `class EgoDriverService(EgodriverServiceServicer)` at `main.py:456`.
- **VaVAM model:** `src/driver/src/alpasim_driver/models/vam_model.py` —
  `class VAMModel(BaseTrajectoryModel)` at `vam_model.py:78`.
- **Model interface it satisfies:** `src/driver/src/alpasim_driver/models/base.py`.
- **Config:** `src/wizard/configs/driver/vavam.yaml` (+ `vavam_configs.yaml`).
- **Plugin registration:** `src/driver/pyproject.toml:47` —
  `vam = "alpasim_driver.models.vam_model:VAMModel"` under the
  `alpasim.models` entry-point group. The upstream model comes from
  `https://github.com/valeoai/VideoActionModel.git` (`pyproject.toml:51`).

So: **to write an adapter you can either implement the eight RPCs yourself, or
register a plugin under `alpasim.models` and let this servicer do the plumbing.**

### 6.1 Server startup

`main.py:1145-1177` — the entire server is ten lines.

```python
def serve(cfg: DriverConfig, ready_event: threading.Event | None = None) -> None:
    server = grpc.server(ThreadPoolExecutor())          # thread pool => concurrent RPCs

    service = EgoDriverService(cfg=cfg)                 # loads the model (slow)
    add_EgodriverServiceServicer_to_server(service, server)   # from egodriver_pb2_grpc

    address = f"{cfg.host}:{cfg.port}"
    server.add_insecure_port(address)                   # plaintext; no TLS, no auth

    server.start()
    ...
    server.wait_for_termination()
```

Points worth noting:

- **No handshake, no registration, no service discovery.** The driver binds a
  port and waits. The runtime is told the address by configuration and connects
  with `wait_for_ready=True`, so it simply blocks until the port answers
  (`validation.py:100-106`).
- **`get_version` is the de-facto readiness signal.** It is the first call, and
  it is what the runtime blocks on while the model loads.
- **Port is assigned by the orchestrator, not fixed.** `vavam.yaml:23` sets
  `port: ???` (an OmegaConf mandatory value); the wizard fills `{port}` from
  `baseport: 6000` upward (`src/wizard/configs/base_config.yaml:74-77,182`). Host
  defaults to `0.0.0.0` (`vavam.yaml:22`).
- **The model loads before the port opens** — `EgoDriverService.__init__`
  constructs the model (`main.py:481-487`) and starts a background inference
  thread (`main.py:518-522`) before `serve` calls `server.start()`.

### 6.2 Session start

`main.py:776-800`, delegating to `Session.create` (`main.py:196-286`):

```python
def start_session(self, request, context) -> SessionRequestStatus:
    with self._sessions_lock:
        if request.session_uuid in self._sessions:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, ...)
        session = Session.create(request, self._cfg, self._context_length, ...)
        self._sessions[request.session_uuid] = session
    return SessionRequestStatus()
```

`Session.create` reads `request.rollout_spec.vehicle.available_cameras`, indexes
them by `logical_id`, and **raises if a camera the config asks for is missing**
(`main.py:247-251`). It then allocates one `FrameCache` per camera. VaVAM demands
exactly one camera, `camera_front_wide_120fov` (`vavam.yaml:27`,
`vam_model.py:85,127-131`).

### 6.3 One tick, in order

**(a) Camera frame arrives** — `submit_image_observation`, `main.py:822-867`:

```python
grpc_image = request.camera_image
session = self._sessions[request.session_uuid]
...
image = Image.open(BytesIO(grpc_image.image_bytes))          # PIL sniffs JPEG/PNG
frame = np.array(session.rectify_image(grpc_image.logical_id, image))   # HWC uint8
...
session.add_image(grpc_image.logical_id, frame, grpc_image.frame_end_us)
return Empty()
```

Decoding in three steps: bytes → PIL (format auto-detected) → optional f-theta→
pinhole rectification → HWC `uint8` numpy, cached under `frame_end_us`. There is
an alternative CUDA path (`main.py:830-837`) that calls
`torchvision.io.decode_jpeg` directly onto the GPU — **that path is JPEG-only and
cannot be combined with rectification** (`main.py:494-498`). A minimum-resolution
check rejects renders too small for the model (`main.py:843-853`).

**(b) Ego motion arrives** — `submit_egomotion_observation`, `main.py:869-884`:

```python
session.add_egoposes(request.trajectory)
for pose, dynamic_state in zip(request.trajectory.poses,
                               request.dynamic_states, strict=True):
    session.add_dynamic_state(pose.timestamp_us, dynamic_state)
```

Poses are appended and re-sorted by timestamp. `strict=True` enforces the 1:1
correspondence the proto promises.

**(c) Route arrives** — `submit_route`, `main.py:886-905`: the route is stored,
then collapsed into one discrete command. The rule (`navigation.py:40-69`): take
the first waypoint at least `min_lookahead_distance` (20 m) away; if its rig-frame
`y` exceeds `command_distance_threshold` (3 m) it is LEFT, below `-3 m` RIGHT,
otherwise STRAIGHT.

**(d) Ground truth arrives** — logged and dropped (`main.py:907-912`).

**(e) `drive` is called** — `main.py:918-1009`:

```python
session = self._sessions[request.session_uuid]

if not self._check_frames_ready(session):        # not enough frames yet
    return DriveResponse(trajectory=Trajectory())        # empty == "no plan"

pose_snapshot = session.poses[-1] if session.poses else None
if pose_snapshot is None:
    return DriveResponse(trajectory=Trajectory())

future = Future()
job = DriveJob(session_id=request.session_uuid, session=session,
               command=session.current_command, pose=pose_snapshot,
               timestamp_us=request.time_now_us, result=future)
self._job_queue.put_nowait(job)                  # hand to the worker thread
prediction = future.result()                     # block this RPC until done

alpasim_traj = self._convert_prediction_to_alpasim_trajectory(
    prediction, job.pose, job.timestamp_us)
return DriveResponse(trajectory=alpasim_traj, debug_info=debug_info)
```

The gRPC handler does not run the model itself. It queues a job and blocks on a
`Future`; a single background worker (`main.py:569-624`) drains up to
`max_batch_size` jobs and runs **one** forward pass for all of them. This is how
several concurrent rollouts share one GPU (`max_batch_size: 32` in
`vavam.yaml:28`).

Speed and acceleration are derived, not received:

```python
_, state = session.dynamic_states[-1]
speed = np.sqrt(state.linear_velocity.x**2 + state.linear_velocity.y**2)
acceleration = state.linear_acceleration.x
```
(`main.py:658-660` — planar speed magnitude; longitudinal acceleration only.)

**(f) The model runs** — `vam_model.py:214-280`:

```python
frames = inp.camera_images[cam_id]                      # list[(ts, HWC uint8)]
if len(frames) != self._context_length: raise ValueError(...)
...
all_tokens = self._tokenize_frames(all_images)          # resize→900x1600, VQ-encode
batched_tokens = all_tokens.reshape(batch_size, self._context_length, h, w)
batched_commands = torch.tensor(all_commands, ...).unsqueeze(1)

trajectories = self._vam(batched_tokens, batched_commands, self.DTYPE)

trajectory_xy = _format_trajs(trajectories[i:i+1])      # (T, 2) in the rig frame
headings = self._compute_headings_from_trajectory(trajectory_xy)
results.append(ModelPrediction.from_planar(trajectory_xy, headings))
```

Images are resized-and-centre-cropped to 900×1600 for `NeuroNCAPTransform`
(`vam_model.py:87-88,168-173`), VQ-tokenised, and fed with a command token. The
command is **re-encoded** because VaVAM's enum differs from AlpaSim's:
AlpaSim `LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3` vs VaVAM `RIGHT=0, LEFT=1,
STRAIGHT=2` (`vam_model.py:154-166`, `models/base.py:19-30`). Output is planar
`(T, 2)` — headings are derived, z is 0.

**(g) Response packing** — `main.py:1011-1076` then `main.py:102-147`:

```python
positions = prediction.selected_positions                       # (T, 3) rig frame
# optional smoothing/retiming in the rig frame (VaVam-Eco); x,y only
pose_local_to_rig_t0, model_t0_us, waypoint_timestamps_us = \
    self._prediction_reference(prediction, current_pose, time_now_us)
return _rig_est_waypoints_to_local_trajectory(
    positions, prediction.selected_rotations,
    pose_local_to_rig_t0, model_t0_us, waypoint_timestamps_us)
```

and inside that function:

```python
trajectory = Trajectory(poses=[pose_to_grpc_at_time(pose_local_to_rig_t0, model_t0_us)])
for waypoint_in_rig, timestamp_us in zip(waypoints_in_rig, waypoint_timestamps_us,
                                         strict=True):
    waypoint_in_local = pose_local_to_rig_t0 @ GeometryPose.from_se3(waypoint_in_rig)
    trajectory.poses.append(pose_to_grpc_at_time(waypoint_in_local, int(timestamp_us)))
```

Anchor first, then each rig waypoint composed into `local`. VaVAM's
`output_frequency_hz` is a hardcoded `2` (`vam_model.py:151-152`), so with
`T = 6` (asserted at
`src/driver/src/alpasim_driver/tests/test_vam_batched.py:147`) the response is
seven poses covering 3 s.

### 6.4 The minimal driver skeleton

Stripped to essentials, a working driver is:

```python
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import Empty, SessionRequestStatus, Trajectory, VersionId
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer, add_EgodriverServiceServicer_to_server)
import grpc
from concurrent.futures import ThreadPoolExecutor

class MyDriver(EgodriverServiceServicer):
    def get_version(self, request, context):
        return VersionId(version_id="my-driver-0.1", git_hash="unknown",
                         grpc_api_version=API_VERSION_MESSAGE)
    def start_session(self, request, context):
        ...                       # read request.rollout_spec.vehicle.available_cameras
        return SessionRequestStatus()
    def submit_image_observation(self, request, context):  ...; return Empty()
    def submit_egomotion_observation(self, request, context): ...; return Empty()
    def submit_route(self, request, context):              ...; return Empty()
    def submit_recording_ground_truth(self, request, context): return Empty()
    def drive(self, request, context):
        return DriveResponse(trajectory=Trajectory())   # legal "not ready"
    def close_session(self, request, context):         ...; return Empty()

server = grpc.server(ThreadPoolExecutor())
add_EgodriverServiceServicer_to_server(MyDriver(), server)
server.add_insecure_port("0.0.0.0:6001")
server.start(); server.wait_for_termination()
```

---

## 7. Things the protos do not tell you

Behaviours you can only learn from the code:

1. **The observation barrier.** Observations are dispatched concurrently and only
   joined before `drive` (`policy.py:142-143`). Your servicer *must* be
   thread-safe.
2. **`get_version` is the readiness gate.** Probed with `wait_for_ready=True`
   before any session (`validation.py:100-106`).
3. **Version numbers are recorded, not enforced.** `_assert_consistent_versions`
   (`validation.py:112-149`) only checks that *multiple addresses of the same
   service* agree. Nothing compares your `grpc_api_version` against the
   runtime's. A version mismatch fails later as a wire error, not a clear message.
4. **No health-check service, no TLS, no auth.** `add_insecure_port` only
   (`main.py:1159`). The grpc health-checking protocol is not used.
5. **Empty trajectories are the warm-up protocol.** Return `Trajectory()` while
   you lack context frames or poses (`main.py:930-955`).
6. **`session_uuid` really is multi-session.** One process, many concurrent
   rollouts, batched onto one GPU.
7. **`inference_seed = session.seed + session.inference_count`** — a fresh seed
   per tick, derived from `DriveSessionRequest.random_seed` (`main.py:713`).
8. **`debug_info.scene_id` is withheld in benchmark runs** to prevent leakage
   (`egodriver.proto:53`). Do not depend on it.
9. **Frame timestamps are keyed on `frame_end_us`**, not `frame_start_us`
   (`main.py:860-864`). Submitting the same timestamp twice for one camera raises
   (`frame_cache.py:63-64`).
10. **Camera frames can lag the drive request.** The driver warns once per session
    when the newest frame's timestamp differs from `time_now_us`
    (`main.py:739-771`).
11. **Command enums differ between AlpaSim and models** — each model implements
    `_encode_command` to translate (`vam_model.py:154-166`).
12. **`rig_to_camera` may be misnamed.** The proto itself flags it
    (`sensorsim.proto:223`).
13. **`time_query_us` and `renderer_data` are ignored** by the reference driver;
    it plans from `time_now_us` alone.
14. **`subsample_factor` lets you tick faster than your model's frame spacing** —
    the cache holds `context_length × subsample_factor` frames and picks every
    Nth (`frame_cache.py:36-49`, `vavam_4hz.yaml`).
15. **Coordinate noise is the runtime's problem, not yours**
    (`CONTRIBUTING.md:89-92`, `policy.py:208-226`).

---

## 8. Open questions

Things I could not settle from the code alone:

1. **Direction of `AvailableCamera.rig_to_camera`.** The proto comment says the
   name is wrong and it is really `camera_to_rig` (`sensorsim.proto:223`), but the
   field was never renamed. The reference driver's rectification path consumes it
   (`rectification.py`) — I did not trace the arithmetic far enough to confirm
   which direction is actually stored. **Verify empirically before using it.**
2. **Is `submit_recording_ground_truth` ever populated in benchmark runs?** It is
   gated on `send_recording_ground_truth` (`policy.py:134-140`); I did not find
   which presets enable it. It is unused by every bundled driver.
3. **Is `terminate_session` used by anything shipped?** The proto documents it for
   RL episode termination; no bundled driver sets it.
4. **Maximum trajectory horizon.** Nothing in the protos or the driver bounds `T`.
   Whether the controller or the scorers assume a minimum horizon (VaVAM's 3 s,
   Alpamayo's longer) is decided downstream, in code I did not read
   (`src/controller/`, `src/eval/`).
5. **What `renderer_data` ever contains.** Declared as free-form bytes; produced
   at `camera.py` and passed through at `policy.py:161`. No bundled driver reads
   it, so its schema is undefined.
6. **Whether PNG actually works end-to-end.** `image_format` accepts `"png"`
   (`config.py:337`) and PIL would decode it, but the CUDA decode path is
   JPEG-only (`main.py:837`). I did not find a test covering PNG.
7. **`DriveRequest.RolloutDriveRequest` (`egodriver.proto:97-100`)** is declared
   but never referenced. Presumably vestigial; not confirmed.
8. **Timeouts and retries.** I found no deadline set on the driver RPCs. Whether a
   slow `drive` stalls the rollout forever or is cut off elsewhere is undetermined.
