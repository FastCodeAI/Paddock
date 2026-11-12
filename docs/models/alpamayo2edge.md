# Alpamayo 2 Edge

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge) at `02f55a7` |
| Adapter | [`adapters/models/alpamayo2edge/`](../../src/paddock/adapters/models/alpamayo2edge/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `CAM_FRONT_TELE` |
| Plan | waypoints, 0.1 s apart, 6.4 s ahead |
| Paddock can run it on | [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [kitscenes](../benchmarks/kitscenes.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [neuroncap](../benchmarks/neuroncap.md) |
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
leaderboard prints that under the run, as it prints METEOR's horizon note.

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
environment, with the model stubbed. `tests/test_vendor_inputs.py` needs the model's
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

Measured on an H200 shared with other jobs, with one plan per scenario:

- **GPU memory:** 6.9 GiB per node as `nvidia-smi` counts it, of which 5.8 GiB is
  torch's peak. Six plans per scenario need 10.6 GiB.
- **Time per plan,** on NAVSIM's 1920×1080 frames: 0.14 s to build the input when
  three of the four frames are already resized, 0.45 s for a log's first scenario
  (16 files read), and 0.5 to 1.2 s in the model.
- **Disk reads:** resized frames are kept by file across scenarios. NAVSIM's next
  scenario in a log is 0.5 s later and shares three of its four frames, so it reads
  5 files instead of 16.

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
- **Field of view.** On NAVSIM the front and cross slots get 62° where the model saw
  120°. On KITScenes every ring camera is about 87°: narrower than 120° in the front
  and cross slots, wider than 70° in the rear ones (`slot_geometry` in the run record).
  On nuScenes every ring slot is about 60°: half the front and cross slots' 120°, and
  narrower than the rear slots' 70°.
- **Lens.** The crop treats every camera as a pinhole; the training rig's lens model
  is not taken into account.
- **No route distance on NAVSIM**, as above.
- **Open loop only.** The plan's first six poses span 0.6 s, so a closed-loop client
  reading six waypoints would get the wrong horizon. `benchmarks` lists only the
  open-loop NAVSIM and KITScenes.

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
recorded future ([kitscenes.md](../benchmarks/kitscenes.md#the-eight-rows-and-where-each-came-from)
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
are scored on their own points at every 0.5 s; its comfort terms read the whole plan.

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
