# Alpamayo 2 Edge

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge) at `02f55a7` |
| Adapter | [`adapters/models/alpamayo2edge/`](../../src/paddock/adapters/models/alpamayo2edge/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `CAM_FRONT_TELE` |
| Plan | waypoints, 0.1 s apart, 6.4 s ahead |
| Paddock can run it on | [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [navhard](../benchmarks/navhard.md), [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [neuroncap](../benchmarks/neuroncap.md), [alpasim](../benchmarks/alpasim.md), [hugsim](../benchmarks/hugsim.md), [nuplan](../benchmarks/nuplan.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

NVIDIA's Alpamayo 2 Edge (November 2026) is a 2.65 B-parameter vision-language-action
model. A Cosmos3-Edge vision-language model reads six cameras, 2 s of the car's
past poses and one route sentence. It writes a one-sentence rationale, or
`<NOMINAL>` when nothing calls for one. A flow-matching expert, which reads the
language model's cache, then plans 6.4 s ahead. The code is Apache-2.0; the Hub
lists the weights as OpenMDW-1.1, and the card in the pinned snapshot says the
licence is still to be confirmed.

Vendor repo: `alpamayo2-edge` in `repos.yaml` (`02f55a7`). Weights:
`nvidia/Alpamayo2-Edge` at revision `f8951d0`, linked as `{weights}/Alpamayo2-Edge`.
Adapter: `src/paddock/adapters/models/alpamayo2edge/`. Run it with

```bash
HF_HOME=<the site's Hugging Face cache> paddock setup --model alpamayo2edge
paddock doctor --model alpamayo2edge
paddock verify --model alpamayo2edge                          # the real model, on a GPU
paddock run --model alpamayo2edge --bench navsim --limit 12   # smoke, stored valid: false
paddock run --model alpamayo2edge --bench navsim              # navtest
```

## What the model consumes and emits

From the vendor's `load_physical_aiavdataset.py`, `sample.py` and `inference.py`:

| | |
|---|---|
| cameras | six: cross left (ID 0), front wide (1) and cross right (2), 120°; rear left (3) and rear right (5), 70°; front telephoto (6), 30° |
| frames | front wide and telephoto at −2, −1, −0.5, −0.1 and 0 s; the other four at −1, −0.1 and 0 s. 22 in all, each labelled in the prompt with the time it was taken (`<t=-0.5s>`) |
| image size | the vendor resizes the front and telephoto frames to 576×320 and the rest to 416×224 |
| ego history | 20 poses 0.1 s apart, −1.9 s to now: position and full rotation, in the current frame |
| route | one sentence, such as "Turn left in 25m" or "Continue straight". It is required |
| output | 64 poses 0.1 s apart (6.4 s) with heading, from 10 flow-matching steps, and the rationale |
| sampling | seed 43, top-p 0.98, temperature 0.6, at most 256 generated tokens, one plan |

The adapter runs the vendor's own `inference.predict` with these settings, on the
input built as below.

## How a NAVSIM scenario becomes that input

NAVSIM's 2 Hz path ([navsim.md](../benchmarks/navsim.md)) sends four frames, 0.5 s
apart, of the five cameras Edge's slots use. Each frame carries the car's pose
relative to the current frame, its measured velocity and the 4-way driving command.

**Cameras.** Each slot is filled from the paddock camera that looks the same way.
Every frame is the widest 16:9 window around the camera's principal point, resized
to the vendor's 1920×1080.

| Edge slot | Trained on | NAVSIM gives |
|---|---|---|
| 0 cross left | 120° | `CAM_FRONT_LEFT` (L0, 55° left), 62° after the crop |
| 1 front wide | 120° | `CAM_FRONT` (F0), 62° |
| 2 cross right | 120° | `CAM_FRONT_RIGHT` (R0, 55° right), 62° |
| 3 rear left | 70° | `CAM_BACK_LEFT` (L2, 141° left), 62° |
| 5 rear right | 70° | `CAM_BACK_RIGHT` (R2, 141° right), 62° |
| 6 front telephoto | 30° | a 30° crop of `CAM_FRONT`: exact |

In the three 120° slots everything looks larger, so nearer, than the model learned to
read it: 1.8 times at the centre of the frame if the training cameras are fisheye
lenses, 2.9 times if they are pinhole, and NVIDIA does not publish which. Alpamayo 1.5
has the same mismatch on NAVSIM.

**Frames.** For each slot, the newest frame at or before the slot's time, which is
the vendor's own rule. At 2 Hz the front and telephoto cameras get the frames at
−1.5, −1.0, −0.5, −0.5 and 0 s, and the others −1.0, −0.5 and 0 s. The prompt
labels each frame with the time it was taken, so the model is told which frame is
repeated and how old the oldest is. On a 10 Hz source the same rule returns
exactly the native frames.

**History.** The model reads 20 poses 0.1 s apart, and NAVSIM sends four, 0.5 s
apart. With straight lines between them, every 0.1 s step of a segment would carry
the segment's average speed. The newest steps would then show the speed of 0.25 s
ago. The vendor's action space fits the plan's starting speed to those steps
(`estimate_t0_states`), so the plan would start at the wrong speed. The adapter
joins the poses with cubics that also pass through each frame's measured velocity,
which is exact for steady acceleration. The vendor's fit then reads the car's own
speed:

| Car at 8 m/s now | Fit from straight lines | Fit from the cubic history |
|---|---|---|
| steady speed | 8.000 m/s | 8.000 m/s |
| braking at 2 m/s² | 8.551 m/s | 8.000 m/s |
| braking at 4 m/s² | 9.103 m/s | 7.999 m/s |
| accelerating at 2 m/s² | 7.448 m/s | 8.000 m/s |

The oldest 0.4 s of the history lies before NAVSIM's first frame. It is continued
at that frame's velocity and turn rate. Repeating the first pose there would write
four parked steps into the history the model reads.

**Route.** NAVSIM's command LEFT becomes "Turn left", RIGHT "Turn right", and the
rest "Continue straight". NAVSIM gives no distance, so the sentence has none;
Alpamayo 1.5 gets the same sentence there. A source that sends its next maneuver and
the distance to it (`nav_maneuver`, `nav_distance_m`) gets "Turn left in 25m".

**Plan.** All 64 poses go back, 0.1 s apart, with the model's own heading. NAVSIM
reads its eight poses at 0.5 s to 4 s from them.

## How a Bench2Drive route becomes that input

Bench2Drive ([bench2drive.md](../benchmarks/bench2drive.md)) is closed loop. CARLA
renders paddock's camera rig, the same for every model, at 20 Hz, and Edge's plan
drives the car. `model.yaml` declares `planning_hz: 10.0`, so the client asks for a
plan every second tick, once every 0.1 s, and follows the last plan in between. Each
request carries the newest frame of the five cameras Edge's slots read, the car's
pose and speed, and the next junction maneuver on the route with the distance to it.

**Cameras.** Every view is 800×450 and 70° wide. The widest 16:9 window around its
centre is the whole frame, resized to the vendor's 1920×1080 as on NAVSIM.

| Edge slot | Trained on | Bench2Drive gives |
|---|---|---|
| 0 cross left | 120° | `CAM_FRONT_LEFT` (55° left), 70° |
| 1 front wide | 120° | `CAM_FRONT`, 70° |
| 2 cross right | 120° | `CAM_FRONT_RIGHT` (55° right), 70° |
| 3 rear left | 70° | `CAM_BACK_LEFT` (110° left), 70° |
| 5 rear right | 70° | `CAM_BACK_RIGHT` (110° right), 70° |
| 6 front telephoto | 30° | a 30° crop of `CAM_FRONT`, 306×172 pixels before the resize |

The front and cross slots are narrower than the model saw, by less than on NAVSIM
(62°). The rear slots match.

**Frames.** Asked every 0.1 s, the client hands the node a frame every 0.1 s, so
once the route has run 2 s every slot gets exactly the frame its time asks for: −2,
−1, −0.5, −0.1 and 0 s for the front and telephoto, −1, −0.1 and 0 s for the
others. Before that, the oldest frame fills the slots that reach further back, and
the prompt labels it with its real time.

**History.** The 20 poses are the client's own, 0.1 s apart, read from CARLA, with
the speedometer's speed along the car's heading. In a route's first 1.9 s the part
before the first frame is continued at that frame's velocity. Every route starts
from rest, so that part reads as a car standing where it is.

**Route.** The client sends the next junction maneuver on the route and its
distance along it, so the sentence is "Turn left in 37m", "Turn right in 9m" or
"Continue straight", the one Alpamayo 1.5 gets. A route with no junction maneuver
left gives "Continue straight". The distance is not capped, so a turn 400 m ahead
reads "Turn left in 400m".

**Plan.** All 64 poses go back, 0.1 s apart. The controller, the same for every
model, follows the plan's own points at 0.5, 1.0, … 3.0 s: every fifth point. Other
models plan 0.5 s apart, and their first six points are the same six times.

**Each frame is resized once.** A plan reads 22 frames, 16 of which an earlier plan
already read, and resizing one 800×450 frame to 1920×1080 takes 30 to 40 ms. The
adapter keeps a closed-loop client's frames resized, by the time they were taken,
for the 2 s the slots look back, and forgets them at the next route's reset. A plan
then resizes its 6 new frames, not all 22.

## On AlpaSim through the bridge

AlpaSim ([alpasim.md](../benchmarks/alpasim.md)) is closed loop on replayed real
drives. Its scenes are NuRec reconstructions of PhysicalAI-AV recordings, made with
the camera rig Edge was trained on, so here every slot gets its own camera through
its own lens. AlpaSim ships no Edge driver, so there is no native row to reproduce.
The settings are the ones every Alpamayo row on AlpaSim uses: the shared 20 Hz
control step, 3.0 s of replayed warm-up with the node called throughout, 400 steps
and the same 30 pinned scenes, with Edge's six cameras rendered instead of Alpamayo
1.5's four (`alpasim.yaml` and `driver/paddock_bridge_alpamayo2edge.yaml` in the
adapter's folder).

**Cameras.** AlpaSim renders six cameras at 1920×1080, 20 Hz, 30 ms shutter, and
the bridge renames them to the node's names:

| Edge slot | Trained on | AlpaSim renders |
|---|---|---|
| 0 cross left | 120° | `camera_cross_left_120fov` as `CAM_FRONT_LEFT` |
| 1 front wide | 120° | `camera_front_wide_120fov` as `CAM_FRONT` |
| 2 cross right | 120° | `camera_cross_right_120fov` as `CAM_FRONT_RIGHT` |
| 3 rear left | 70° | `camera_rear_left_70fov` as `CAM_BACK_LEFT` |
| 5 rear right | 70° | `camera_rear_right_70fov` as `CAM_BACK_RIGHT` |
| 6 front telephoto | 30° | `camera_front_tele_30fov` as `CAM_FRONT_TELE` |

The lenses are f-theta, and the bridge sends no pinhole intrinsics for them. A frame
that comes without intrinsics is not cropped: it is resized whole to 1920×1080, which
for a frame already that size is nothing, and then the vendor's own resize runs as on
every source. A slot that would need a crop, the telephoto cut out of `CAM_FRONT`, is
refused for such a frame by name instead of approximated. On AlpaSim it never needs
one: a `CAM_FRONT_TELE` in the request fills slot 6 itself. `CAM_FRONT_TELE` is not
in `model.yaml`'s `cameras`, which is what a simulator is asked to render, so neuroNCAP
and Bench2Drive are never asked for it; the node decodes it whenever it is sent.

**Frames and history.** The node is asked for a plan every 100 ms and hears the
step between through `/observe`, so it holds a frame of every camera and a pose
every 50 ms. Every slot then gets exactly the frame its time asks for, and every one
of the 20 history poses is a recorded one. The warm-up is 3.0 s, longer than the
2.0 s the oldest slot looks back, so this holds from the first plan AlpaSim drives.
`--canbus-ego-frame` sends the car's forward and sideways speed in its own frame, and
replaces the zeros AlpaSim reports during the warm-up with the speed from its poses.

**Route.** AlpaSim's command is `FOLLOW_REFERENCE` at every step, which reads as
"Continue straight". What it holds is the route polyline, and the bridge sends it
with the time of the car frame it is in. The node moves the route into the newest
pose's frame through the pose at that time and classifies the bend ahead by heading
change, as AlpaSim's own Alpamayo driver does (`inputs.route_to_nav_text`, ported
from AlpaSim's `nav_text.py`): "Turn left in 20m", "Turn right in 9m" or "Continue
straight", with turns nearer than 5 m or beyond 40 m ignored. A route with no usable
point falls back to the command, so "Continue straight": Edge refuses an input with
no sentence, where AlpaSim's driver gives Alpamayo 1.5 none.

**Plan.** All 64 poses go back, 0.1 s apart, with the model's own headings, and the
bridge drives them all.

**Checked.** `tests/test_alpasim_inputs.py` covers the slots, the uncropped frames,
the route and its move into the newest frame, and that the route port agrees with
alpamayo15's port of the same function. `tests/test_alpasim_vendor.py`, in the model's
environment, puts a full-size 1920×1080 AlpaSim sample with the telephoto through
NVIDIA's `validate_sample` and `prepare_model_inputs`. Run on the real checkpoint on an H200 with
AlpaSim-shaped requests (20 Hz pixels at 1916×1080, no intrinsics, the telephoto, a
route 0.5 s old), one plan took 0.5 s after the first, every slot's frame times were
the vendor's, and a turn 24 m ahead on a route 0.5 s old at 8 m/s read "Turn left
in 20m". The 30-scene result is under [Results, AlpaSim](#alpasim).

## Settings

Each is read from the environment of the node; `model.yaml` declares the defaults.

| Setting | Default | What it does | Run stays ranked |
|---|---|---|---|
| `ALPAMAYO2_EDGE_FRAMES` | `nearest` | `distinct`: the most recent frames whatever their spacing, Alpamayo 1.5's rule | no |
| `ALPAMAYO2_EDGE_HISTORY` | `hermite` | `linear`: straight lines between the source's poses | no |
| `ALPAMAYO2_EDGE_SAMPLES` | `1` | plans drawn per scenario; the medoid, the plan nearest the others, is returned | no |
| `ALPAMAYO2_EDGE_SEED` | `43` | set before every plan | no |
| `ALPAMAYO2_EDGE_NAV_DISTANCE_M` | unset | a distance for turns the source gives none ("Turn left in 20m") | no |
| `ALPAMAYO2_EDGE_SAVE_REASONING` | `0` | `1` keeps the rationale's first 400 characters in each plan's extras | yes |
| `ALPAMAYO2_EDGE_ABSENT_CAMERAS` | `refuse` | `black`: a slot whose camera the source never sends gets a black frame at each time it asks for, and is named in `black_slots`. comma2k19 needs it | yes, with a note |

No default was chosen by scoring navtest. They are the vendor's settings, plus the
reasons above.

`ALPAMAYO2_EDGE_ABSENT_CAMERAS` stays `refuse` everywhere but comma2k19. A black frame
is a claim the source never made, so a source missing a camera stops the plan unless
the run says otherwise. comma2k19 records one camera, and the vendor requires all six
slots: there the front and telephoto slots are real and the other four black. The
leaderboard prints that under the run, as it prints METEOR's horizon note. To score
Edge on comma2k19, set it for that pairing alone in `paddock.toml`:

```toml
[models.alpamayo2edge.benchmarks.comma2k19.env]
ALPAMAYO2_EDGE_ABSENT_CAMERAS = "black"
```

`model.yaml` also sets `OMP_NUM_THREADS` to 4, the CPU threads torch and numpy use in
the node, because torch takes every core by default. On a 48-core machine shared with
CARLA, the vendor's processor then took 1.4 s to pack one plan's 22 frames, where 4
threads take 0.18 s and 16 take 0.07 s. The packed pixels and the plan were
bit-identical at 2, 4, 8, 16 and 48 threads, on one input. The runner counts
`OMP_NUM_THREADS` as a speed setting for every model, so a site that sets its own in
`[models.alpamayo2edge.env]` in `paddock.toml` keeps its run ranked. Edge's Bench2Drive
record was driven when the 4 threads came from a setting of Edge's own,
`ALPAMAYO2_EDGE_THREADS`, which its notes name: it set torch's threads alone and left
numpy's at every core.

The seed is set before every plan. A plan therefore depends on its input alone, not
on the node, the order, a retry or a resumed run. It is still one draw: on one
KITScenes frame, plans drawn with different seeds ended metres apart at 6.4 s.

The vendor's code refuses a plan whose rationale has not reached the trajectory within
256 generated tokens. With the seed fixed, asking again repeats the same draw, so the
adapter draws again with the next seed, up to four draws, and the scenario fails only
if all four are refused. On navtest this happened in 2 of the 12,146 scenarios at
seed 43, and seed 44 planned both. The node logs each such plan, and `/diagnostics`
counts them (`drawn_again`).

## Environment and weights

`setup.sh` builds the vendor's own lock (`uv sync --locked`: Python 3.12, torch
2.8.0 with CUDA 12.8, transformers 5.17.0) into `PADDOCK_VENV`, with SDPA attention;
flash-attn is not installed. It adds fastapi and uvicorn for the node.

The checkpoint is 5.3 GB and not gated. `setup.sh` fetches it into the Hugging Face
cache at `HF_HOME` (default `{weights}/hf_cache`) and checks the sha256 of every file
the model reads. It then links `{weights}/Alpamayo2-Edge` to the pinned snapshot, by a
relative path. A later download of another revision cannot change what a run is
scored against.

The vendor's `prepare_model_inputs` reads the processor from the checkpoint on every
call, 0.84 s each on a network disk. The adapter reads it once, at load, and runs
the same steps (`prepare_inputs`). A test checks that both give the same tensors.

## Tests

`make check` runs `tests/test_inputs.py` and `tests/test_policy.py` in paddock's own
environment, with the model stubbed. `tests/test_vendor_inputs.py` and
`tests/test_alpasim_vendor.py` need the model's
environment and the checkpoint's config, tokenizer and processor. It runs on the CPU
and loads no weights. `make check-alpamayo2edge` runs the whole folder there, and
fails rather than skips when the environment or the checkpoint is missing:

```bash
make check-alpamayo2edge ALPAMAYO2_EDGE_PY=<model venv>/bin/python \
  ALPAMAYO2_EDGE_CHECKPOINT=<{weights}/Alpamayo2-Edge>
```

It checks the restated constants against the vendor's. Among them is the start of the
message the vendor refuses an incomplete rationale with, which is how the adapter
knows to draw again: the test has the vendor's own code refuse a draw and checks that
the next seed is drawn. It also checks that a NAVSIM sequence passes the vendor's
`validate_sample`, that the prompt labels its frames as above, that the processor
read once gives the vendor's tensors (2,892 image tokens), and the speed fits in the
table above.

## Cost

On NAVSIM, measured on an H200 shared with other jobs, with one plan per scenario:

- **GPU memory:** 6.9 GiB per node as `nvidia-smi` counts it, of which 5.8 GiB is
  torch's peak. Six plans per scenario need 10.6 GiB.
- **Time per plan,** on NAVSIM's 1920×1080 frames: 0.14 s to build the input when
  three of the four frames are already resized, 0.45 s for a log's first scenario
  (16 files read), and 0.5 to 1.2 s in the model.
- **Disk reads:** resized frames are kept by file across scenarios. NAVSIM's next
  scenario in a log is 0.5 s later and shares three of its four frames, so it reads
  5 files instead of 16.

On Bench2Drive, measured on an A100 80GB with one node and one CARLA server, on
route 2802:

- **GPU memory:** 6.5 GiB for the node (6,610 MiB as `nvidia-smi` counts it), beside
  CARLA's 7.1 to 7.4 GiB.
- **Time per plan,** over the route's 438 plans, with 4 CPU threads: 0.19 s to build
  the input, which resizes the plan's 6 new frames, and 1.0 s in the model (median;
  0.85 to 1.12 s for 90 % of plans). The model's time includes 0.18 s in which
  NVIDIA's processor packs the 22 frames on the CPU. With torch's default of every
  core, plans on the same route took 1.5 to 2.4 s in the model.
- **Wall clock:** a plan every 0.1 s of game time, so the route ran at 0.064 times
  real time, against 0.04 with torch's default threads.

## Training exposure

The card on the Hub's main branch lists about 115,000 hours of NVIDIA's own
multi-camera driving with egomotion. It also lists about 3.7 M chain-of-causation
rationales written by Alpamayo 2 Super, navigation instructions and VQA data. The
card in the pinned snapshot is a template that leaves training data "TBD".
NVIDIA's fine-tuning recipe lists the public PhysicalAI-AV chunks the model has not
seen, which implies it saw others. No public driving set is named, and no NAVSIM or
nuPlan data is claimed.

## Known limits

- **NAVSIM is 2 Hz; the model was built for 10 Hz.** Three of its five front
  frames and the −0.1 s frame of every camera do not exist at 2 Hz, and 0.4 s of its
  history is continued, not recorded. With `[benchmarks.navsim] nuplan` set, the
  harness feeds nuPlan's 10 Hz frames instead; nobody has run Edge on them yet.
- **Field of view.** The front and cross slots get 62° on NAVSIM and 70° on
  Bench2Drive, where the model saw 120°. On KITScenes every ring camera is about 87°:
  narrower than 120° in the front and cross slots, wider than 70° in the rear ones
  (`slot_geometry` in the run record). On nuScenes every ring slot is about 60°: half
  the front and cross slots' 120°, and narrower than the rear slots' 70°.
- **Lens.** The crop treats every camera as a pinhole; the training rig's lens model
  is not taken into account.
- **No route distance on NAVSIM**, as above.
- **No lane change in the route sentence.** Bench2Drive's routes include lane
  changes, and the sentence names only turns, so Edge is told "Continue straight"
  through one. Alpamayo 1.5 gets the same sentence.
- **A blurred telephoto on Bench2Drive.** Its slot is a crop 306 pixels wide of the
  800-pixel front view, which the vendor's processor then reads at 576×320.
- **CARLA is not what Edge was trained on.** Its training drives are real camera
  video; Bench2Drive's frames are rendered.
- **Closed loop on Bench2Drive, neuroNCAP and HUGSIM.** Bench2Drive and neuroNCAP drive
  Edge's own points at 0.5, 1.0, … 3.0 s: neuroNCAP's engine from the wire's `trajectory`
  field (`Trajectory.to_wire`), Bench2Drive's client from the whole plan. HUGSIM's bridge
  reads the whole plan too and sends its first 4 s as 8 waypoints 0.5 s apart.

## Results

### NAVSIM

NAVSIM navtest, all 12,146 scenarios, on OpenScene's 2 Hz frames, with no failed
scenario. The v1 score replays the v2 run's plans through the v1 devkit, so the two
numbers describe one set of plans.

| Benchmark | Score | Run | How |
|---|---:|---|---|
| NAVSIM v2 | **0.626** EPDMS | `navsim/alpamayo2edge/20261111T052918-f9f90a` | four nodes on one H200, 92 min |
| NAVSIM v1 | **0.699** PDMS | `navsim_v1/alpamayo2edge/20261111T070439-f19e8f` | replay of the v2 run's plans, 9 min, no GPU |

Every sub-metric is in [navsim.md](../benchmarks/navsim.md#navsim-v2) and
[navsim_v1.md](../benchmarks/navsim_v1.md#navsim-v1). Alpamayo 1.5 scores 0.608 and
0.676 on the same 2 Hz frames, and 0.691 and 0.753 with nuPlan's 10 Hz frames. Edge has
not been run at 10 Hz, so compare it with the 2 Hz pair.

**The scenarios and ground truth are the ones the other rows used.** Edge's runs were
made at a different site from the earlier NAVSIM rows, with metric caches built there.
The constant-velocity floor, scored on those caches, reproduces its stored rows exactly:
0.238544 EPDMS and 0.206517 PDMS, every sub-metric to six decimals, with the same
`sequences_sha256` (`82e33c6837552a03` for v2, `013e89176339021d` for v1). The two
records reported here carry the same hashes.

**The plans are reproducible.** The run reported here is the second full run. The first,
`navsim/alpamayo2edge/20261111T031938-f9f90a`, was made at `02b26a8`, before the redraw
was added. It scored 12,144 scenarios and lost the two the vendor refused, so it was
stored partial and `valid: false`; its record is kept in `results/`, and the leaderboard
lists it under "Excluded runs". Every one of those 12,144 plans is identical in the two
runs, to the 0.1 mm the plan dump records, although different nodes served them.

What the plans look like, from the run's plan dump and node logs:

- **The route sentence is followed.** Of the 2,501 "Turn left" scenarios, 86 % of
  plans end more than 1 m left at 4 s and 0.4 % more than 1 m right. Of the 1,575
  "Turn right" ones, 84 % end right and 0.6 % left.
- **The plan starts at the car's speed.** In the 10,707 scenarios where the car moves
  faster than 1 m/s, the plan's first 0.1 s step is within 0.2 m/s of the measured
  speed in 95 % of them; the median difference is +0.1 m/s.
- **The model wrote `<NOMINAL>` in 37 % of scenarios** (4,488 of 12,146) and a
  rationale in the rest.
- **Two plans were drawn again**, the two the first run lost, both at seed 44.

### navhard

NAVSIM v2.2's two-stage protocol on `navhard_two_stage`
([navhard.md](../benchmarks/navhard.md#results)): all 450 stage-one frames scored,
none failed, on OpenScene's 2 Hz frames, the only rate navhard's synthetic follow-ups
exist at. It ran as one node on the site's shared RTX PRO 6000 in 3.5 h.

| Benchmark | Score | stage one | stage two | Run |
|---|---:|---:|---:|---|
| navhard | **0.180** EPDMS | 0.501 | 0.340 | `navhard/alpamayo2edge/20261113T111924-f36c51` |

**The frames are the ones the other rows used.** The record carries
`sequences_sha256: 3af39efa412cc8df`, as all nine earlier navhard records do, and its
per-frame mean (0.179572) equals the devkit's combined row
(`by_group.source_aggregates.paddock_check`).

Alpamayo 1.5 scores 0.193 on the same frames (0.507 and 0.366 by stage), so Edge sits
just below it, fifth of ten. What pulls the score down, from the record's sub-metrics:
drivable-area compliance (0.636 in stage one, 0.647 in stage two), lane keeping in
stage two (0.479) and two-frame extended comfort (0.409 and 0.444). Traffic-light
compliance (0.996) and no at-fault collisions in stage one (0.912) are high. EPDMS is a
per-frame product, so its median frame scores 0.

### KITScenes

All 223 keyframes of the list every KITScenes row uses, at its recorded 10 Hz, so each
frame is the one at the time Edge asks for and no pose history is extrapolated. Edge's
64 points, 0.1 s apart, are scored on their own points at 0.5, 1.0, ... s.

| Route | ADE@3s ↓ (m) | FDE@3s ↓ (m) | Run |
|---|---:|---:|---|
| as text, `command_source = "gt"` | **0.752** | 1.393 | `kitscenes/alpamayo2edge/20261111T230906-22b8a7` |
| "Continue straight" on every keyframe | 0.762 | 1.423 | `kitscenes/alpamayo2edge/20261111T223048-778679` |

The first is the support-matrix cell. It is given the route the way Alpamayo 1.5's
cell (`legacy-full-nav`, 1.002 m) is: a turn on 39 of the 223 keyframes, from the
recorded future ([kitscenes.md](../benchmarks/kitscenes.md#the-rows-and-where-each-came-from)
says how that column was made here). The second shows what the route is worth. The 184
keyframes told to go straight give identical plans in both runs; on the 39 turns ADE
moves from 1.068 to 1.012 m, a paired bootstrap 95% interval of −0.172 to +0.014 m.

Both runs were made with no local changes, one per H200, about 13 minutes each: the
second at `f8a51e6`, the first at `30b616e`. The first was made again at that later
commit so that its record lists the turn commands it was given, and every number came
out the same as the first time. In each, the model wrote `<NOMINAL>` on 74 of the 223
keyframes, and no plan was drawn again.

### nuScenes

All 5,119 keyframes of the 150 validation scenes, at nuScenes' 2 Hz. A frame 0.1 s old
does not exist at 2 Hz, so each slot's −0.1 s frame is the −0.5 s one (`ALPAMAYO2_EDGE_FRAMES
= nearest`), and the pose history between keyframes is rebuilt from their measured speeds.
Edge's 64 points, 0.1 s apart, are read at every 0.5 s, which lands on its own points. The
route is a sentence built from the recorded future, as for every model here: "Turn left in
25m" on 257 keyframes, "Turn right in …" on 361, "Continue straight" on the rest.

| L2 at 1 / 2 / 3 s (m) ↓ | Point L2 avg ↓ | Cumulative L2 avg ↓ | ADE at 3 s ↓ | Point collision avg ↓ | Run |
|---|---:|---:|---:|---:|---|
| 0.455 / 1.058 / 1.870 | **1.128** | 0.630 | 0.957 | 0.92 % | `nuscenes/alpamayo2edge/20261111T231928-e4e2d6` |

Second on point L2 of the full-split rows, behind Qwen-Drive 1.0 (1.128 against 0.800), and
below the constant-velocity baseline (1.288), which only Qwen-Drive had beaten. Alpamayo
1.5's row is 1.487. Its point collision rate, 0.92 %, is the lowest after Qwen-Drive's
0.77 %. nuScenes is not in Edge's claimed training data.

Made at `e931fbf` with no local changes on one H200, 64 minutes. That commit is on the
branch `alpamayo2edge/nuscenes-neuroncap`; its Edge adapter and nuScenes code are this
branch's, and the one other file it differs in, `policy/protocol.py`, changes only what a
closed-loop simulator is sent. The model wrote
`<NOMINAL>` on 1,967 of the 5,119 keyframes (38 %), and no plan was drawn again.

### comma2k19

All 10,165 samples, five keyframes in each of 2,033 one-minute segments of I-280 highway
driving, at the recorded 20 Hz, so every frame is the one at the time Edge asks for and
no pose history is extrapolated. comma2k19 has one camera, so the front slot gets it (65°,
where Edge was trained on 120°), the telephoto slot a 30° crop of it, and the other four
slots black frames (`ALPAMAYO2_EDGE_ABSENT_CAMERAS=black`). Edge's 64 points, 0.1 s apart,
are scored on their own points at every 0.5 s, and so are its comfort terms.

| ADE at 1 / 2 / 3 / 4 / 5 s (m) ↓ | FDE at 3 s ↓ | Median ADE at 3 s ↓ | Miss rate at 3 s ↓ | PDMS-style ↑ | Run |
|---|---:|---:|---:|---:|---|
| 0.086 / 0.225 / **0.438** / 0.713 / 1.043 | 1.013 | 0.356 | 8.9 % | 0.949 | `comma2k19/alpamayo2edge/20261112T012152-9b1e09` |

The lowest ADE of every comma2k19 row at every horizon from 1 s to 5 s, on the mean
and the median: Qwen-Drive 1.0 is next, at 0.589 m at 3 s, and Alpamayo 1.5 at 0.819.
Its plans reach as far as the car went (path length 99.7 % of the recorded), and its
PDMS-style score is the highest here (0.949; OpenPilot's 0.816). comma2k19 is not in
its claimed training data.

Made at `8c5ed2f` with no local changes: four nodes, while a neuroNCAP sweep shared
the site's two H200s, 5.6 hours. No plan was drawn again. That commit is on the branch
`alpamayo2edge/nuscenes-neuroncap`; its Edge adapter and comma2k19 code are this
branch's, except that its harness imported the same 0.5 s stride rule from
`policy/protocol.py`, which there also changes what a closed-loop simulator is sent.

### HUGSIM

All 420 scenarios ([hugsim.md](../benchmarks/hugsim.md#results)), command only, asked
every 0.25 s step (Edge declares 10 Hz, HUGSIM steps at 4 Hz), the first 4 s of each plan
sent. Scores ×100.

| Benchmark | HD-Score | PDMS | Route completion | NC | DAC | TTC | Comfort | Run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HUGSIM | **31.2** | 53.1 | 45.3 | 63.5 | 88.7 | 57.4 | 92.1 | `hugsim/alpamayo2edge/20261114T172855-b502b3` |

Third of eleven, below Drive-JEPA (37.1) and Qwen-Drive (34.8), above UniAD (26.1), the
constant-velocity floor (24.2) and Alpamayo 1.5 (23.8). By source dataset: PandaSet 47.0,
nuScenes 36.7, Waymo 32.3, KITTI-360 10.2. By difficulty: easy 47.1, medium 37.4, hard
24.9, extreme 16.1. 25.2 % of episodes completed, 42.1 % hit a vehicle, 24.8 % hit
scenery, 7.9 % left the route, none ran to the step cap.

Driven 14-15 Nov 2026 by `paddock run` on one node of the site's shared RTX PRO 6000:
15,643 plans, no bridge error. It was cut off after 284 scenarios and finished with
`--resume`. Every camera was rendered, though the record lists Edge's five; why, and why
the scores do not depend on it, is in [hugsim.md](../benchmarks/hugsim.md#results). Building
the inputs, not the model, set the pace on PandaSet: 9-14 s per plan to resize the frames
on the shared CPU, against about 1 s in the model.

### neuroNCAP

The full official set: 20 scenario–scene pairs over 14 scenes, 10 runs each, 200 runs. The
simulator renders at 10 Hz, so every frame is the one at the time Edge asks for, once a run
is 2 s old; before that, its earliest frames repeat and its past path is continued, as for
every model at the start of a run. The engine drives six points 0.5 s apart, and is sent
Edge's own points at 0.5 ... 3.0 s, every fifth of its 64 (`Trajectory.to_wire`). The route
is the scenario's command, so a turn has no distance ("Turn right").

| Stationary ↑ | Frontal ↑ | Side ↑ | Overall ↑ (0–5) | Collision rate ↓ | Run |
|---:|---:|---:|---:|---:|---|
| 3.45 ±1.71 | 3.22 ±2.19 | 1.56 ±1.99 | **2.92** | 54 % | `neuroncap/alpamayo2edge/20261111T231934-33adca` |

Fifth of the eight model rows on the full set: below Drive-JEPA (4.13), Alpamayo 1.5 (3.73), OpenPilot
(3.60) and AutoE2E (3.16), above METEOR (2.64), Qwen-Drive 1.0 (2.49) and VisionPilot (1.49). Side impacts are
its weak column: 38 of 50 side runs collided, every run in `side-0103` and `side-0921`. It
avoided every collision in four pairs: `stationary-0106`, `stationary-0108`, `side-0108` and
`frontal-0346`.

Made at `e931fbf` with no local changes on the site's two H200s, 10.3 hours. The model
wrote `<NOMINAL>` on 4,559 of its 12,917 plans (35 %), and no plan was drawn again.
That commit is on the branch `alpamayo2edge/nuscenes-neuroncap`. Its Edge adapter,
closed-loop wire and neuroNCAP code are this branch's, but for two things that change
nothing here: the stride helper was named `wire_stride`, and the adapter did not yet have
`ALPAMAYO2_EDGE_ABSENT_CAMERAS`, whose default refuses a missing camera as it did then;
neuroNCAP sends all five.

### AlpaSim

The 30 pinned `public_2601` scenes (26.01 artifacts), the set every AlpaSim row since the
scene set was recorded is on, at the shared 20 Hz control step. Edge sees its own six
cameras through their own lenses, the real telephoto among them, and its route sentence
comes from AlpaSim's route ([On AlpaSim through the bridge](#on-alpasim-through-the-bridge)).

| scene_score ↑ | Progress ↑ | At-fault collision ↓ | Offroad ↓ | Distance to the human's path ↓ | Run |
|---:|---:|---:|---:|---:|---|
| **0.437** | 0.628 | 23 % (7 of 30) | 20 % (6 of 30) | 2.94 m | `alpasim/alpamayo2edge/20261112T220852-f6d995` |

Sixth of nine rows on these 30 scenes: below Alpamayo 1.5 (0.795), OpenPilot (0.773),
Qwen-Drive (0.700), VisionPilot (0.629) and the constant-velocity floor (0.488), above
METEOR (0.423), Drive-JEPA (0.421) and AutoE2E (0.169). A scene that ends in an at-fault
collision or off the road scores 0, and 13 of the 30 did; the other 17 average 0.77, and six
of them scored 1.0. Where it collided or
left the road it had decided to: the last rationale before the end is a deliberate move, as
in "Steer right into the open roadway and accelerate to bypass the road debris ahead" or a
left turn on a green arrow, not a plan that stopped making sense. It also leaves the human's
line on purpose elsewhere ("Change lanes to the left to overtake the slower lead vehicle"),
which `dist_to_gt_trajectory` counts against it up to its 4 m cap. Alpamayo 1.5's row is on
the same settings but four cameras, so the gap is the two models, not the harness; Edge also
sits below Alpamayo 1.5 on neuroNCAP (2.92 against 3.73).

What was checked on this run, rather than argued: every scene ran its full length (168 or 169
plans each, 5,048 in all); every plan came back whole and was driven on the model's own
headings (`bridge_plans`: 5,048 `model`, 0 `derived`); every route sentence came from
AlpaSim's route (`from route` on all 5,048), 520 of them a turn; the first scene, driven twice,
scored the same to the last digit. Edge wrote `<NOMINAL>` on 2,216 plans (44 %) and no plan was
drawn again.

Made at `c372441` with no local changes, on one of the site's H200s shared with the
simulator, 3.3 hours (about 6.6 minutes a scene). On this site Docker cannot mount either
NFS share, so the AlpaSim checkout, the scene cache and the run directory were on local
`/tmp`; that changes where files are, not what is computed.

### Bench2Drive

Every route of `bench2drive220.xml`, driven under the
[220-route protocol](../benchmarks/bench2drive.md#the-220-route-protocol): asked at its
declared 10 Hz, routes ended after 30 s without progress or at four times the expert's
time, up to three attempts a route. 216 routes were driven on two A100s; the four that
site set aside for every model (3800, 11715, 23695, 24071) were driven here later under
patch 0003, from the same commit (`f873bc2`), and pooled with them.

| Routes | Driving score ↑ | Route completion ↑ | Infraction penalty ↑ | Run | How |
|---|---:|---:|---:|---|---|
| 220 of 220 | **13.87** | 38.78 % | 0.483 | `bench2drive/alpamayo2edge/full220-86aa45e768a29a64-ffe60a` | 216 routes: 4 workers on each of two A100s, 11.6 h; 4 routes: one RTX PRO 6000 Blackwell, 1 h |

It replaces the 216-route record `full220-836dda734659a86c-56db50` (13.74). It ranks
tenth of the eleven models in that column, just below AutoE2E (14.29) and above METEOR
(1.89). The tables, and Bench2Drive's own scores for the 216 routes, are in
[bench2drive.md](../benchmarks/bench2drive.md#bench2drive--220-routes).

- **The four routes driven here** scored 16.55 (3800), 41.46 (11715), 21.11 (23695)
  and 4.77 (24071). The 30 s rule ended three of them and the over-time rule 11715;
  23695 and 24071 have a collision. Each scored on its first attempt, and patch 0003
  acted on 11715, 23695 and 24071.

The rest of this part is over the 216 routes driven on the A100s.

- **It completes 16 routes, and none is a success.** Each of the 16 has an
  infraction, and 142 of the 216 routes have a collision.
- **Its one ability is Traffic signs:** 52 of 94 routes got past their first junction
  with no red light or stop sign run, the most of any model in the column.
- **Most routes end on the 30 s rule:** 166 of 216, and 14 on the over-time rule.
- **The model wrote `<NOMINAL>` in 18 % of plans** (18,069 of 99,399), against 37 %
  of NAVSIM scenarios, and a rationale in the rest.
- **One plan was drawn again.**
- **With eight nodes sharing the two cards,** a plan took 2.1 s in the model at the
  median, 14.2 s at most.
