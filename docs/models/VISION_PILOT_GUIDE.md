# Vision Pilot — a plain-language guide

What this repo is, what the three model repos do, where the test data lives, and how to build and
run it.

---

## 1. What Vision Pilot is

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

## 2. Do I need to clone the other three repos?

**No.** The trained networks are already committed here as files you can use directly:

```
VisionPilot/modules/models/weights/
  autodrive_fp32.onnx   43 MB      autodrive_int8.onnx   12 MB
  autospeed_fp32.onnx   38 MB      autospeed_int8.onnx   11 MB
  autosteer_fp32.onnx   33 MB      autosteer_int8.onnx    9 MB
```

("`.onnx`" is just a portable file format for a trained neural network. "`fp32`" is the full-size
version; "`int8`" is a shrunken, faster, slightly less accurate one. Pick which with the
`model.precision` setting.)

The code loads them by filename in
[inference.cpp:83-86](../VisionPilot/modules/models/src/inference.cpp#L83-L86). The build copies
the whole `weights/` folder next to the program, which is why you must run the program **from the
`build` folder** — it looks for the files relative to wherever you started it.

Clone [auto_speed](https://github.com/autowarefoundation/auto_speed),
[auto_steer](https://github.com/autowarefoundation/auto_steer), or
[auto_drive](https://github.com/autowarefoundation/auto_drive) only if you want to **retrain** the
networks on your own data or export new files. To run Vision Pilot, you do not need them.

There is also one Git submodule (`zenoh_carla_bridge`) that is not set up and not needed — it only
matters for one particular simulator setup.

---

## 3. The three networks, in plain terms

All three look at the same camera picture, shrunk to 1024×512 pixels. Each answers a different
question.

### AutoSpeed — "which car ahead could I hit?"

- **Gets:** one picture.
- **Gives back:** boxes drawn around vehicles, each tagged with how much of a threat it is:
  *directly in my lane*, *cutting in or out of my lane*, *off to the side*, or *not relevant*.
- **Used for:** finding the vehicle we might rear-end. Its box is converted into a real distance
  in metres using the camera calibration (section 6).

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

## 4. Architecture and size

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

## 5. How the pieces fit together

[`app/vision_pilot.cpp`](../VisionPilot/app/vision_pilot.cpp) is the whole story in about 200
lines — one loop, worth reading start to finish.

```
 camera, video file, or simulator
              |
              v
     Image preparation           makes TWO versions of each frame
              |--> a "flattened" top-down view  --> AutoDrive (needs this frame + the last)
              |--> a plain resized picture      --> AutoSteer
                                                --> AutoSpeed
              |
              v
     Combining step
       |
       |-- Car-in-front tracker
       |     takes AutoDrive's distance guess + AutoSpeed's box
       |     gives  distance in metres, and whether the gap is opening or closing
       |
       |-- Lane tracker
             takes AutoSteer's 64 path points, fits a curve through them
             gives  how far off-centre we are, which way we point, how sharp the bend is
              |
              v
        The planner                an optimiser that looks a few seconds ahead
             gives  a steering angle and an acceleration
              |
              |--> the car   (over the vehicle's data bus, or to a simulator)
              +--> the screen (heads-up display, browser stream, or a recording file)
```

### Where things live

| Folder | What it does |
| --- | --- |
| [`modules/sensing/`](../VisionPilot/modules/sensing) | Reads camera or video frames; reads vehicle speed; sends commands back |
| [`modules/engine/`](../VisionPilot/modules/engine) | Loads the networks and decides whether they run on the graphics card or the processor |
| [`modules/models/`](../VisionPilot/modules/models) | The three networks and the code that runs them |
| [`modules/safety_guardian/`](../VisionPilot/modules/safety_guardian) | The combining step and the planner |
| [`modules/visualization/`](../VisionPilot/modules/visualization) | The on-screen display and browser streaming |
| [`modules/debug/`](../VisionPilot/modules/debug) | The `--debug-viz` diagnostic overlay |
| [`Calibration/`](../Calibration/) | Measuring your own camera (section 6) |
| [`Simulation/`](../Simulation/) | Running against driving simulators instead of a real car |
| [`Functional_Safety/`](../Functional_Safety/) | Safety documentation |

---

## 6. The camera calibration, and why it matters

[`config/H.yaml`](../VisionPilot/config/H.yaml) holds nine numbers that convert **pixels into
metres** on the road surface.

Everything downstream depends on it:

- AutoSpeed gives you a *box in pixels*. These numbers turn it into "that car is 42 metres ahead."
- AutoSteer gives you *points in pixels*. These numbers turn them into a path measured in metres,
  which is what the off-centre distance and the bend are calculated from.

So if this file is wrong, **the distances and lane position come out wrong even when the networks
are working perfectly.** Nothing will look broken; the car will just behave badly.

The version in the repo matches the supplied sample data, which is 1920×1080. For your own camera,
follow the [calibration guide](../Calibration/) — and keep a copy of the original file first.

---

## 7. Where the test data is

Vision Pilot in video mode needs **two files**: a video, and a matching list of vehicle speeds.
The speed file is plain text — **one speed in metres per second per line, one line per video
frame**. If it runs out before the video does, the program stops with an error.

### The official sample clips (already downloaded)

The [main README](../README.md) links to a Google Drive folder. It can be fetched from the command
line:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1-Sxgz3XHzFD6XtETz1sVFRtDKY3W57QB" \
      -O openlane_sample
```

On this machine they live at:

```
/shakaal_hdd_1/workspace/naren/renesas/testdata/openlane_sample/
    test_open_lane_2/   input.mp4  frame_speed.txt  H.yaml
    test_open_lane_5/   input.mp4  frame_speed.txt
    test_open_lane_6/   ...            (7 clips in total)
```

Real road footage from the OpenLane dataset — seven clips, about 198 frames each at 10 frames per
second, covering roughly 23 to 106 km/h.

**They are 1920×1280, not 1920×1080.** The comment inside
[`config/H.yaml`](../VisionPilot/config/H.yaml) says 1920×1080, and that comment is wrong. Clip 2
ships its own copy of `H.yaml` and it is byte-for-byte identical to the one in the repo, which
confirms the shipped calibration is the right one for this footage. You can check it is being
applied correctly from the startup line:

```
[Pipeline] H_resized set — raw=1920x1280  top_crop=320  sx=1.8750 sy=1.8750
```

1280 − 320 = 960, and 1920×960 is exactly the 2:1 shape the networks expect. If you feed in
footage of a different size, this crop no longer lines up with the calibration and every distance
will be wrong.

### What a good run looks like

Running all seven clips end to end:

| Clip | Speed | Lead vehicle | Car ahead | Max drift | Mean pedal |
| --- | --- | --- | --- | --- | --- |
| 2 | 46–73 km/h | 34–150 m | 85% of frames | 1.43 m | +1.29 m/s² |
| 5 | 102–105 km/h | 53–150 m | 36% | 0.52 m | +0.37 |
| 6 | 100–106 km/h | 44–92 m | 100% | 0.17 m | −0.80 |
| 7 | 43–85 km/h | 22–69 m | 100% | 1.64 m | +0.44 |
| 8 | 57–73 km/h | 22–150 m | 53% | 1.04 m | +1.03 |
| 9 | 63–72 km/h | 27–37 m | 100% | 0.21 m | −0.10 |
| 10 | 23–48 km/h | 16–51 m | 100% | 0.36 m | +0.91 |

A distance of exactly 150 m means "nothing found" — that is the cut-off, not a measurement. Pedal
commands sit inside roughly ±2 m/s², which is a normal comfort range; lane drift is mostly well
under half a metre. Clips 7 and 2 show larger drift because the road curves and the car changes
lane.

There is a visual walkthrough of these runs in
[`telemetry_report.html`](telemetry_report.html) next to this file.

### If you make your own test clip

The shipped calibration only fits OpenLane footage. Anything else — a dashcam video, a synthetic
clip — needs its own calibration (section 6), otherwise the networks will look fine while every
distance and lane position comes out wrong.

## 8. Running it, and reading the output

Set the input in [`config/vision_pilot.conf`](../VisionPilot/config/vision_pilot.conf):

| `source.mode` | Where frames come from |
| --- | --- |
| `video` | A video file plus a speed file — replaying recorded data |
| `v4l2` | A camera plugged into this machine, with a live vehicle bus |
| `ros2` | Another program feeding frames in, e.g. a driving simulator |

Then, from the `build` folder:

```bash
./VisionPilot                 # with a display attached
./VisionPilot --no-window     # no display — prints numbers only
./VisionPilot --debug-viz     # diagnostic overlay instead of the clean display
```

**Edit the settings files before building.** The build copies `config/` next to the program, so
edits made afterwards are ignored until you build again.

### What the log line means

Each line is one camera frame:

```
plan: tyre=-0.0043 rad  accel=-45.244 m/s²  |  cte=0.02m(raw=0.06m) cte_dot=-0.05m/s
      epsi=0.012 epsi_dot=+0.014rad/s  kappa=-0.0005  |  cipo=true  dist=5.4 m  vel=-1.46 m/s
```

| Field | Meaning |
| --- | --- |
| `tyre` | How far to turn the front wheels, in radians. Multiply by 57 for degrees |
| `accel` | How hard to accelerate. Negative means brake |
| `cte` | How far sideways the car is from the middle of its lane, in metres |
| `raw` | The same thing before smoothing — the gap shows the filter working |
| `cte_dot` | How fast that sideways offset is changing |
| `epsi` | The angle between where the car points and where the road goes, in radians |
| `epsi_dot` | How fast that angle is changing |
| `kappa` | How sharply the road bends. 1 divided by this is the turn radius in metres |
| `cipo` | Whether a vehicle was found ahead in our lane |
| `dist` | How far ahead that vehicle is, in metres |
| `vel` | Its speed relative to us. Negative means the gap is closing |

Every one of these is a blend of several networks — see the table in section 3.

There is a visual walkthrough of a real run, with charts and annotated frames, in
[`telemetry_report.html`](telemetry_report.html) next to this file.

### Seeing the overlays without a display

Vision Pilot draws its overlays into a window, which is useless on a machine with no display. The
[`render_tool/`](../render_tool/) folder holds a small companion program that replays a clip
through the real pipeline, calls the project's **own** drawing code, and writes MP4s and PNGs
instead of opening a window. Nothing in `VisionPilot/` is modified. See
[`render_tool/README.md`](../render_tool/README.md); pre-rendered results for four clips are in
`render_tool/out/`.

---

## 9. Building it

Two routes. The first needs an administrator password; the second does not.

### Route A — system packages

```bash
sudo apt-get install -y --no-install-recommends \
    cmake build-essential libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-nice libnice-dev libsrtp2-dev \
    libboost-system-dev nlohmann-json3-dev \
    coinor-libipopt-dev libcppad-dev liblapack-dev libblas-dev

# one header lives in a differently-named folder on Ubuntu 24.04
hdr="$(dirname "$(find /usr/include -name IpIpoptApplication.hpp | head -1)")"
sudo ln -sfn "$hdr" /usr/include/coin-or
```

Then fetch the neural-network runtime. **Match it to your machine's CUDA version** — run
`nvcc --version` to check. On a CUDA 12 machine use the `gpu` file below; the project's Dockerfile
uses a CUDA 13 file instead, which will not work on a CUDA 12 machine.

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-gpu-1.26.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.26.0.tgz
python3 -m pip install --target ./cudnn nvidia-cudnn-cu12     # the graphics-card maths library
```

### Route B — no administrator password (verified working)

This is how the machine this guide was written on actually runs it.

```bash
conda create -y -n visionpilot2 -c conda-forge \
  cmake make pkg-config gcc_linux-64 gxx_linux-64 \
  "libopencv=4.10" gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad \
  libsrtp2 zlib "libboost-devel=1.84" nlohmann_json ipopt cppad lapack blas
```

Four details matter, each found the hard way:

| Detail | Why |
| --- | --- |
| `libopencv=4.10` | The default is now version 5; this project needs version 4 |
| `libboost-devel=1.84` | Version 1.91 deleted a component the project asks for by name |
| `zlib` | Without it the video-streaming libraries are not detected, and the build then fails to link |
| No `opencv` package | The Python version of it forces a newer Python and blocks the version 4 pin |

### Building and running, either route

```bash
cd VisionPilot && mkdir -p build && cd build
cmake -DONNXRUNTIME_ROOT=<runtime folder> -DGPU=ON \
      -DCMAKE_CXX_STANDARD_LIBRARIES="-lcppad_lib" ..
make -j"$(nproc)" VisionPilot
./VisionPilot --no-window
```

`-lcppad_lib` is required — the planner will not link without it.

### One setting you must not miss on Route B

```bash
export OMP_NUM_THREADS=1
```

Without it the program runs a burst of frames at full speed and then **freezes forever**, at a
different frame each time. The cause is a clash between two versions of a parallel-computing
library inside the planner's equation solver. It is a packaging quirk, not a bug in Vision Pilot —
Route A is unaffected.

### Measured performance

Full 150-frame clip at 1920×1080, on the graphics card, full-size networks:

```
149 of 149 frames    18.5 frames per second    3.1 GB of graphics memory
```

The processor-only mode also works but is far slower — roughly 35–40 seconds per frame, because
that code path deliberately turns off all speed optimisations.

---

## 10. Two rough edges in the project itself

Both are hidden when you build with Ubuntu's system packages, and both appear on a build without
on-screen display support:

1. **Building without video streaming does not link.** When the streaming libraries are missing,
   the build system prints a warning and drops the streaming source file — but
   `visualization.cpp:638` still calls into it, so linking fails. Installing `zlib` avoids this by
   making the libraries detectable.
2. **Shutting down cleanly can crash.** The display class calls "close all windows" whether or not
   a window was ever opened. On a build with no on-screen display support that raises an error
   from a destructor, which aborts the program **after all frames are processed** — so you get
   correct results and then a crash message. Checking the "is there a window" flag before those
   two calls would fix it.

---

## 11. When something goes wrong

| What you see | What it usually is |
| --- | --- |
| "Config file not found", or the networks are not found | You started the program from the wrong folder — `cd build` first |
| Distances or lane position are nonsense, but the boxes look right | The camera calibration in `config/H.yaml` does not match your footage |
| Runs a few frames fast, then freezes with the graphics card idle | Set `OMP_NUM_THREADS=1` (section 9) |
| Crash right after the clip finishes | The shutdown bug in section 10 — your results are still valid |
| "undefined reference to WebRTCStreamer" while building | Streaming libraries not detected — install `zlib` |
| "read() called past end of speeds data" | The speed file has fewer lines than the video has frames |
| Nothing appears in the browser | Set `webrtc_on = true`, and make sure the streaming libraries were present when you built |
