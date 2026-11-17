# VisionPilot

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [autowarefoundation/vision_pilot](https://github.com/autowarefoundation/vision_pilot) at `e93eceaa` |
| Adapter | [`adapters/models/visionpilot/`](../../src/paddock/adapters/models/visionpilot/) |
| Environment | conda, Python 3.10 |
| Cameras | `CAM_FRONT` |
| Plan | controls, 0.5 s apart |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [navhard](../benchmarks/navhard.md), [bench2drive](../benchmarks/bench2drive.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [nuplan](../benchmarks/nuplan.md), [hugsim](../benchmarks/hugsim.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

Autoware's VisionPilot: three small camera networks and a planner, built for one
forward camera. This page says what the networks are and what paddock's adapter
does around them. Building and running VisionPilot on its own is covered by
[its repository](https://github.com/autowarefoundation/vision_pilot/tree/e93eceaa).

## What it is

Vision Pilot is driver-assistance software: the code that sits between a **single forward-facing
camera** and the car's steering and pedals. It is the kind of system that keeps a car centred in
its lane and a safe distance behind the car in front.

It can do these things:

| Name | In plain terms |
| --- | --- |
| Adaptive cruise control | Hold a set speed, but slow down for the car in front |
| Forward collision warning | Warn the driver about an imminent crash |
| Emergency braking | Brake automatically if the driver does not |
| Lane keep assist | Steer to stay centred in the lane |
| Lane departure warning | Warn when drifting out of the lane |
| Speed assist | Know and respect the speed limit |
| Highway autopilot | Hands-free driving within one lane |

Two things make it unusual:

- **One ordinary camera.** No radar, no lidar. A 1–2 megapixel camera with a fairly narrow view
  (about 52–55 degrees wide).
- **No maps.** It does not look up a pre-built 3D map of the road. It reads the road live, every
  frame. This is often called "mapless".

**This repo is the plumbing, not the brains.** It does not train anything. It loads three
ready-made neural networks, runs them on each camera frame, combines their answers, and turns the
result into two numbers: *how much to turn the wheel* and *how hard to press the pedal*.

---

## The three networks

All three look at the same camera picture, shrunk to 1024×512 pixels. Each answers a different
question.

### AutoSpeed — "which car ahead could I hit?"

- **Gets:** one picture.
- **Gives back:** boxes drawn around vehicles, each tagged with how much of a threat it is:
  *directly in my lane*, *cutting in or out of my lane*, *off to the side*, or *not relevant*.
- **Used for:** finding the vehicle we might rear-end. Its box is converted into a real distance
  in metres using the camera calibration.

### AutoSteer — "where does the road go?"

- **Gets:** one picture.
- **Gives back:** 64 points tracing the path ahead, plus a confidence score for each point.
- **Used for:** working out how far off-centre we are and how sharply the road bends.
- **Why it is clever:** it never looks for painted lane lines. It predicts the *drivable path*
  directly, so it still works where the paint is faded, worn away, or covered.

### AutoDrive — "the shortcut"

- **Gets:** **two** pictures — the current frame and the one before it.
- **Gives back:** three numbers — how far away the car in front is, how sharply the road bends,
  and how confident it is that there is a car in front at all.
- **Why two pictures:** comparing consecutive frames reveals motion. That is how a single camera
  can estimate distance and closing speed, which one still image cannot give you.

### Why three networks instead of one

This is the part worth understanding. Two of the networks answer each important question, by
completely different methods:

| Question | Answered by measuring | Answered by predicting |
| --- | --- | --- |
| How far is the car ahead? | AutoSpeed's box, converted through geometry | AutoDrive, directly |
| How sharply does the road bend? | AutoSteer's path, fitted to a curve | AutoDrive, directly |
| Is there a car ahead at all? | AutoSpeed's lane tag | AutoDrive's confidence score |

Every safety-critical number has two independent sources. A combining step weighs them against
each other. If one goes wrong, it does not quietly become a bad steering command. That redundancy
is the safety argument for the whole design.

---

## Architecture and size

Everything in this section was read out of the shipped model files, not from documentation.

### Size on disk and parameter counts

| Model | fp32 file | int8 file | Parameters | Graph nodes | Conv layers |
| --- | --- | --- | --- | --- | --- |
| AutoSpeed | 38.0 MB | 11.1 MB | 9.33 M | 315 | 78 |
| AutoSteer | 33.3 MB | 9.1 MB | 8.22 M | 201 | 44 |
| AutoDrive | 43.0 MB | 11.8 MB | 10.58 M | 316 | 63 |
| **All three** | **114 MB** | **32 MB** | **28.1 M** | | |

These are small networks by modern standards — 28 million parameters in total, about the size of a
single mid-range image classifier. That is deliberate: the whole set has to run several times a
second inside a car.

### A shared backbone

All three are built on the same feature extractor, which is why the AutoDrive repository offers the
option of starting training from AutoSpeed's weights. In every model the parameters are grouped into
five stages named `p1` through `p5`, each halving the image size and widening the channels:

- first layer is a 3×3 convolution over the 3 colour channels, producing 16 channels
- widest layer produces 256 channels
- activations appear as `Mul` + `Sigmoid` pairs, which is SiLU (also called Swish)
- all three export at ONNX opset 18

### What each one adds on top

**AutoSpeed** — backbone → a feature pyramid neck (`fpn.h1`…`h6`) → a split detection head
(`head.box`, `head.cls`, `head.dfl`). The `dfl` part means box edges are predicted as small
probability distributions rather than single numbers, which is a YOLO-family technique for
sharper boxes.

Its single output tensor is `[1, 8, 10752]`, and both numbers decompose exactly:

```
10752 anchors = 128x64  (stride 8)  = 8192
              +  64x32  (stride 16) = 2048
              +  32x16  (stride 32) =  512
8 channels    = 4 box coordinates + 4 threat classes
```

So it detects at three scales simultaneously — small distant vehicles on the fine grid, large near
ones on the coarse grid.

**AutoSteer** — the same backbone → a much smaller neck (`fpn.h1`, `h2`) → three short head
branches, two of which produce the outputs: `lane_value` (where the path is) and `height` (how much
to trust each point). Fewer convolution layers than the other two (44), because it only has to
trace a line.

**AutoDrive** — backbone → three convolutions → two fully-connected layers → three tiny separate
heads, one per answer: `distance_head`, `curvature_head`, `flag_head`. It is the only one of the
three whose batch-normalisation layers were left unfused in the export (28 of them), and the only
one taking two images at once.

### Measured speed

Full clip on the graphics card, mean over 197 frames, all three models running concurrently:

| | Preprocess | AutoDrive | AutoSteer | AutoSpeed | All three in parallel | End to end |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 6.1 ms | 39.5 ms | 38.6 ms | 40.5 ms | 40.7 ms | 21 fps |
| int8 | 5.4 ms | 53.2 ms | 48.8 ms | 52.5 ms | 54.1 ms | 17 fps |

The parallel figure being roughly equal to the slowest single model shows the three really do
overlap on the graphics card rather than queueing up.

### int8 is smaller but *slower* — do not use it for speed

This is counter-intuitive and worth knowing before you reach for it. The int8 files contain **no
integer convolution operations at all** — instead they hold 163 to 456 `DequantizeLinear` nodes
feeding ordinary floating-point convolutions. That is *weight-only* quantisation: the weights are
stored compactly, then converted back to floating point before every layer runs.

So int8 buys you a 3.5× reduction on disk and in memory, and costs you about 30% in speed on the
graphics-card path, because the conversion is extra work that fp32 simply does not do. Reach for it
when memory is the constraint, not when throughput is.

*(Timings come from a graphics card shared with other jobs, so absolute numbers move around between
runs; the relative ordering has been stable. The 18.5 fps quoted elsewhere in this guide is a
different measurement — the whole application, including the planner solving on the processor every
frame.)*

---

## How paddock drives it

* **Set-speed.** The IDM cruise speed used to be latched at run start as
  `max(start speed, 1.0)`, a NeuroNCAP assumption (ego already at cruise). On
  Bench2Drive every route starts from standstill, so VisionPilot crawled at
  1 m/s for whole routes and its "completions" were completions by crawl. The
  policy now takes `Observation.speed_limit` (the simulator's posted limit) each
  tick and falls back to the latch only when no limit is sent.
* **GPU.** A site that sets `VISIONPILOT_PROVIDERS="cuda"` now gets its runs
  stored `valid: false`. The provider is not one of VisionPilot's speed-only
  settings, because the two providers do different arithmetic and nobody has
  shown that never moves a plan. Run on the default CPU provider for a ranked
  row, with `VISIONPILOT_ORT_THREADS` and the benchmark's node count, both
  speed-only, to share the cores. The Bench2Drive and nuScenes rows ran on CUDA
  before that rule and stay ranked. So did the first comma2k19 row; its re-run
  of November 2026 used the CPU provider.

  Those rows ran with this setup. The ONNX stack defaulted to the CPU provider
  with ~94 threads. Site
  env: `VISIONPILOT_PROVIDERS="cuda"`, `VISIONPILOT_ORT_THREADS="2"`, and
  `LD_LIBRARY_PATH` pointing at the venv's nvidia wheel libs
  (`site-packages/nvidia/{cudnn,cublas,cuda_runtime,cuda_nvrtc}/lib`); without
  it ORT 1.23 fails "libcudnn.so.9: cannot open" and silently falls back to
  CPU. Under shared-GPU pressure the CUDA provider is built with
  `arena_extend_strategy=kSameAsRequested`, a 1.5 GB limit and heuristic conv
  search (`VISIONPILOT_GPU_MEM_MB`).
* **No route input.** VisionPilot has no goal or command interface, so on
  junction routes it goes straight; nuScenes and Bench2Drive both show it.
* **Curvature scale (fixed September 2026; earlier runs are stale).**
  `auto_drive.hpp` documents `curvature_1pm = curvature_raw * CURV_SCALE` and
  then deliberately does not apply it — "Domain conversion is the caller's job"
  — so the constant is not in `auto_drive.cpp`, and grepping for it there finds
  nothing. It lives in the caller:
  `lateral_fusion.hpp::Config::ad_curvature_scale = 0.21f`, annotated "CURV_SCALE
  = 0.21 in training", with `debug_draw.hpp` carrying the same 0.21 sourced to
  `load_data_auto_drive.py`. The adapter had 1.0. That is not a harmless
  default: `IDM` caps the set-speed at `sqrt(mu*g/|kappa|)` and raises the
  overshoot to the fourth power, so a curvature 4.8× too large made the planner
  demand ~-60 m/s² on a 30 m/s comma2k19 keyframe and reach 5 m in four seconds
  instead of 120. On comma2k19 the correction moved ADE@3s from 15.09 to 5.28
  over a 20-sample smoke and `path_length_ratio` from 0.60 to 0.84, leaving
  every lateral number unchanged — curvature reaches the plan only through the
  longitudinal cap. It surfaced there rather than on the closed-loop benchmarks
  because those are urban and slow enough that the cap rarely binds, but
  **every VisionPilot result recorded before this change used 1.0** and will not
  be reproduced by a re-run. Re-run since at the pinned `e93eceaa` with 0.21:
  KITScenes and comma2k19 (5-6 Nov 2026), nuScenes, NAVSIM v2 and v1, and
  neuroNCAP (11-12 Nov 2026), Bench2Drive's 220 routes (6-8 Nov 2026), and navhard.
  neuroNCAP ran on onnxruntime's CUDA provider (ranked with its note, `score_notes`), driven
  at main 9ba4b7e from 11 Nov 20:42 to 12 Nov 10:45 UTC with a pause; its record was written
  by a `--resume` that reused all 200 runs, so its `started_utc` and `duration_s` are that
  resume's.
  AlpaSim's cell needs none: it was driven by AlpaSim's own VisionPilot driver,
  a separate port that applies `AD_CURVATURE_SCALE = 0.21` itself.
