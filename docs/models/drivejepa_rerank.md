# drivejepa_rerank: Drive-JEPA with a learned choice among its candidates

Drive-JEPA builds 32 candidate trajectories. A scorer head then rates each one,
and the top-rated candidate is driven. That scorer was trained on one target
only, NAVSIM's PDM score. Off NAVSIM it ranks the 32 close to random.

On KITScenes the best of the 32 lands within 0.37 m ADE of the human driver.
The one it picks lands 2.36 m away, ranked about 15th of 32 on average. That
is why Drive-JEPA is last on KITScenes and first on NAVSIM.

`drivejepa_rerank` keeps everything up to the 32 candidates and replaces only
the choice. A small selector, trained on KITScenes **train**, scores every
candidate, and the top one is driven.

Adapter: `src/paddock/adapters/models/drivejepa_rerank/`. It runs in
Drive-JEPA's own env with the same checkpoint. Weights are a gallops checkpoint
under `{weights}/drivejepa_rerank/`.

```bash
paddock run --model drivejepa_rerank --bench kitscenes
paddock run --model drivejepa_rerank --bench nuscenes
```

## Getting the weights

`model.yaml` loads `{weights}/drivejepa_rerank/kitscenes-train-h128-t1/step_000000325/`.
That directory holds two files, and both are required:
- `reranker.npz`, 43 KB, sha256 `2050b799e8e88d4f…6b91fa`
- `checkpoint.json`, which records what the weights were trained on

Neither is in git. A run stops at `no re-ranker weights at …` until the
directory exists. There are two ways to get it.

**Copy it.** It exists at the `fastcode-shakaal` site under that site's
`[roots.weights]`. Copy the whole directory. `ranker.load` refuses
`reranker.npz` without its `checkpoint.json`, because the report's train/test
guard reads it.

**Rebuild it.** You need KITScenes train, Drive-JEPA's env, and about 3 GPU
hours at this site's speed. Every step writes files outside the repo.

```bash
# 1. the 5,711 train keyframes (gallops): the scenes in its rerank_train_scenes.txt, keyframes by rule
python -m gallops.data.rerank keyframes -o train_samples.pkl

# 2. Drive-JEPA's 32 candidates for each, through paddock's own path, with the crop the weights need
#    (Drive-JEPA env; set model.yaml's env, including DRIVEJEPA_IMAGE_GEOMETRY=openscene)
python tools/models/drivejepa/paddock_candidates.py --bench kitscenes --rows train_samples.pkl --output train.jsonl

# 3. features and per-candidate ADE labels (gallops)
python -m gallops.data.rerank dataset train.jsonl --dataset kitscenes --split train -o train.npz

# 4. train (gallops), with the settings recorded in checkpoint.json
gallops train-rerank train.npz --train-run kitscenes-train-h128-t1 --hidden 128 --temperature 1.0
```

Step 2 runs the model, so it is paddock's. Steps 1, 3 and 4 make and use
training data, so they are gallops'. Checked against this checkpoint's own
files:
- step 1 lists the same 5,711 keyframes it was trained on
- step 3 on the original dump gives every array of the original `train.npz`
  unchanged (its sha256 is checkpoint.json's `dataset_sha256`), plus an
  `image_geometry` key

The scene set is not a rule. It is the first 373 of the split's 533 scenes to
finish downloading, at dataset revision `213352cf`, so gallops tracks it as
`src/gallops/data/rerank_train_scenes.txt`.

Step 4 picks its epoch on held-out scenes. Training is seeded, so the same
dataset should give the same step, but that has not been checked at a second
site. If the step differs, point `model.yaml` at the new directory. That diff is
the record that the weights changed.

## Why the choice, and not the inputs

Every input-side explanation was tested first, paired on the benchmark's 223
samples through the devkit loader that produced the 2.364 row. None moved the
pick by more than noise:

| change | effect on the pick's ADE | scene-level t |
|---|---:|---:|
| acceleration zeroed, or smoothed over ±0.5 s | ≤ 0.07 m | ≤ 1.75 |
| route command straight or unknown, in place of the recorded one | ≤ 0.07 m | ≤ 1.75 |
| ego origin moved to the rear axle (1.0 / 1.461 / 2.0 m) | −0.09 to +0.02 m | ≤ 0.7 |
| OpenScene's lens distortion re-applied | −0.14 m | −1.27 |

Across all of these, best-of-32 stayed within 0.35–0.39 m and the pick's mean
rank within 14.6–15.3. So the model sees the road well enough, and the
misranking is in the scorer. A Lanelet2 check pointed the same way: 90% of the
candidates the scorer passed over are map-legal.

The scripts behind these checks dump the candidates through the devkit loader
and score them against the Lanelet2 maps. They are not merged; they live on the
branch `drivejepa/selection-diagnostics`.

Retraining that scorer means simulating every candidate against a PDM metric
cache. That cache exists only for NAVSIM. A selector trained on imitation (how
close each candidate comes to the human) needs only candidates and logged
futures, and KITScenes train has both.

## How it is built

**Features.** 18 per candidate, all taken from what the model already output
([ranker.py](../../src/paddock/adapters/models/drivejepa_rerank/ranker.py)):
- the scorer's six raw logits
- the trajectory's shape: travel at 1/2/3 s, lateral offset and heading at 3 s,
  step statistics, total turning
- ego speed

Nothing in them looks at the future. The future is the label.

**Label and model.** The label for each candidate is its ADE@3s to the logged
future, turned into a soft target over the 32 (softmax of −ADE). The selector
is a one-hidden-layer MLP (128 units), trained with listwise cross-entropy in
numpy.

**Epoch selection.** The epoch is chosen on a held-out 15% of *train* scenes,
by the ADE of the pick. A lower loss that drives a worse trajectory is not an
improvement.

**Hidden size and label temperature.** 128 units and temperature 1.0 won a
sweep over {16, 32, 64, 128} × {0.3, 0.5, 1.0}. The sweep ran on the devkit
loader's candidates and ranked configurations by held-out train scenes only.
It also computed the benchmark score of every configuration. The held-out
winner was the benchmark's best as well, so the choice did not depend on the
benchmark. The benchmark was not unseen, though, and this row should be read
with that in mind.

**Training data.** It comes from this path. Candidates were dumped through
paddock's own KITScenes harness and this policy
([paddock_candidates.py](../../tools/models/drivejepa/paddock_candidates.py)),
for the same 5,711 train keyframes. So the selector was fitted to exactly the
candidates a `paddock run` produces.

**Who owns what.** Paddock runs the model and dumps its candidates. Gallops
does the rest of training: `python -m gallops.data.rerank` lists the train
keyframes and turns a dump into features and ADE labels, and `gallops
train-rerank` writes `reranker.npz` and a `checkpoint.json` sidecar. Neither
repo imports the other, so the npz carries the contract:
- the `feature_names` it was fitted to, which must equal `FEATURE_NAMES`
- a probe of inputs with gallops' own scores for them, which `ranker.load`
  recomputes

A file that fails either check is refused at load, before the 1.2 GB model is
read. The probe replays stored features, so neither check sees gallops' copy of
`candidate_features` drift from this one. A test on each side holds them equal
against the same input and expected values (`test_features_match_gallops`).

## Two inputs differ from `drivejepa`

**Cameras cropped onto the OpenScene pinhole** (`DRIVEJEPA_IMAGE_GEOMETRY=openscene`).
- The KITScenes ring is 87° wide; the OpenScene cameras the checkpoint trained
  on are 64°.
- The devkit loader behind the 2.364 row crops each camera about its principal
  point and rescales it to OpenScene's intrinsics. This option reuses that
  loader's own crop window.
- `drivejepa` itself keeps `native`, so its published rows do not change.
- With the crop, paddock's path gives a pick ADE of 2.521 and best-of-32 of
  0.409 on the 223 samples. Without it, best-of-32 was worse on every one of
  the first six samples checked.
- `model.yaml` sets it for every benchmark, so it changes nuScenes too. Computed
  from the `v1.0-trainval` calibrations:
  - `CAM_BACK` narrows from 89–90° to 64°.
  - The front and side cameras (64–65°) are rescaled, and the crop window runs
    past the 1600×900 image. That leaves up to 53 rows and 29 columns of black
    padding in the 1920×1080 output, of which the pipeline's `[28:-28]` row
    crop removes part.

  So a nuScenes run of this model differs from `drivejepa`'s in the pixels as
  well as the selector.

**Route command.** Paddock's KITScenes harness sends STRAIGHT for every sample,
as it does for every model. The devkit run derived a turn command from the
recorded future on 39 samples. Measured above, this is worth ≤ 0.07 m.

## Results

Weights: `kitscenes-train-h128-t1/step_000000325`, from gallops commit `dfa7459`.
They were trained on 4,858 keyframes from 318 train scenes; the epoch was chosen
on the other 853 keyframes from 55 train scenes. On those held-out scenes the
pick's ADE went from 2.515 to 1.050, against a best-of-32 of 0.502.

**KITScenes**, run `20260921T210621-d7ed08`, all 223 samples, valid:

| | ADE@3s (m) | FDE@3s (m) |
|---|---:|---:|
| drivejepa_rerank | **0.973** | **2.082** |
| Drive-JEPA's own pick, same path and inputs | 2.521 | — |
| best of the 32 (the ceiling for any selector) | 0.409 | — |
| candidate closest to constant velocity | 1.159 | — |
| `drivejepa` recorded row (devkit loader) | 2.364 | 4.199 |
| constant-velocity baseline row | 1.381 | 3.091 |

Paired on the same 223 samples, with scene-clustered SEs over 140 scenes:
- against Drive-JEPA's own pick: −1.548 m, SE 0.174, t −8.0
- against the candidate closest to constant velocity: −0.186 m, SE 0.050, t −4.9

So it is not just picking "keep your speed". By split, it scores 1.002 on the
186 val samples and 0.828 on the 37 overlap_train_val samples, whose area
overlaps train. The offline report on the dumped candidates
(`tools/models/drivejepa/rerank.py report`) gives the same 0.973. A 5-sample
smoke matched the paddock run to within the dump's 4-decimal rounding.

The re-ranker's pick is better than the model's on 155 samples, worse on 46 and
about the same on 22.
- **Where it wins,** the model picks a plan at roughly half the current speed
  on an open road. At 7–14 m/s, that is ADE 10–13 m.
- **Where it loses,** the human slowed for something ahead (a red light, a
  crossing, a turning car) and the re-ranker kept the current speed. Its only
  view of the scene is the scorer's logits, so it cannot see the reason to slow.

Figures: `paddock-runs/analysis/drivejepa_kitscenes_selection/plots/` at this
site. They are not tracked.

**nuScenes** val, run `20261012T220930-6194bd`, all 5,119 samples over 150
scenes, valid, made with the camera-frame fix (#62): its `run.json` records
`camera_frame: "ego"`. No nuScenes data went into the selector, so this is
zero-shot.

| | cumulative L2 (m) | ADE@3s (m) | L2 at 1 / 2 / 3 s (m) | point collision (%) |
|---|---:|---:|---|---:|
| drivejepa_rerank | **0.587** | **0.949** | 0.373 / 0.953 / 2.126 | 1.21 |
| `drivejepa` with the fix, native cameras (run `20261012T215049-798a0d`, not yet tracked) | 1.360 | 2.031 | 0.983 / 2.201 / 3.957 | 1.35 |
| `drivejepa` recorded row, camera-blind | 0.912 | 1.536 | 0.540 / 1.587 / 3.519 | 1.46 |
| this re-ranker before the fix (run `20260918T221353-6194bd`, withdrawn) | 0.620 | 0.985 | 0.394 / 1.051 / 2.075 | 1.82 |
| constant velocity | 0.641 | 1.069 | 0.372 / 1.151 / 2.340 | 1.66 |

Paired on the same 5,119 samples, with SEs over the 150 scenes:
- against constant velocity:
  - cumulative L2 −0.054 m, t −2.8
  - ADE@3s −0.120 m, t −3.4
  - point collisions −0.45 points, t −1.5, not separated
  - UniAD's cumulative collisions (`col_uniad_avg`) 1.37% against 1.07%,
    +0.31 points, t +0.7, not separated
- against the same selector before the fix: cumulative L2 −0.032 m (t −1.6),
  point collisions −0.61 points (t −2.3)
- against `drivejepa` with the fix: cumulative L2 −0.772 m (t −16)

**Seeing the camera made Drive-JEPA's own pick worse on nuScenes** (1.360
against 0.912 blind). Under the crop this model uses, its 32 candidates still
hold good plans, and the selector finds them. That last comparison mixes two
changes: `drivejepa` runs on native cameras, and this model on the OpenScene
crop (see above). The selector's own effect at the same pixels has not been
measured on nuScenes, because the run records only the driven trajectory, not
Drive-JEPA's own pick.

## Limits

- **Open-loop only.** The selector learned closeness to a human path and nothing
  about collisions. `model.yaml` declares only KITScenes and nuScenes until it
  has been through neuroNCAP, where it must be compared pair by pair against
  `drivejepa`.
- **On neuroNCAP it is measured worse, not merely unmeasured.** A smoke on
  scene 0103 with the camera-frame fix gave 1.85 against `drivejepa`'s 3.01 on
  the same nine episodes (three scenario pairs, seeds 0–2), and it was hit in 6
  of them against 5. That is a subset, not a neuroNCAP score. Runs:
  `neuroncap/drivejepa_rerank/20261012T231620-01570a` (not tracked) and
  `neuroncap/drivejepa/20261012T205915-7153f0`. The benchmark is a gate this
  selector is not expected to pass as it stands. An earlier offline replay over
  neuroNCAP candidate dumps pointed the same way, but those dumps were made
  camera-blind, so its numbers are not quoted here.
- **NAVSIM is expected to get worse, not better.** NAVSIM scores PDM, which is
  exactly what the original scorer was trained for. A selector trained on
  imitation gives that up. It has not been measured.
- **Adapted, not zero-shot, on KITScenes.** See
  [comparison.md](comparison.md#drive-jepa-rerank--kitscenes).
