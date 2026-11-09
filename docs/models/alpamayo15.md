# Alpamayo 1.5

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) at `7a8f1c7` |
| Adapter | [`adapters/models/alpamayo15/`](../../src/paddock/adapters/models/alpamayo15/) |
| Environment | uv, Python 3.12 |
| Cameras | `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_TELE` |
| Plan | waypoints, 0.5 s apart, 6.4 s ahead |
| Paddock can run it on | [kitscenes](../benchmarks/kitscenes.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [bench2drive](../benchmarks/bench2drive.md), [alpasim](../benchmarks/alpasim.md), [nuscenes](../benchmarks/nuscenes.md), [comma2k19](../benchmarks/comma2k19.md), [nuplan](../benchmarks/nuplan.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

NVIDIA's Alpamayo 1.5 (10B), a vision-language driving model: four cameras, four
frames each, 1.6 s of ego history and an optional navigation sentence in; a
reasoning trace and 64 poses 0.1 s apart (6.4 s) out. The adapter is
`src/paddock/adapters/models/alpamayo15/`; the model card and training data are
in [comparison.md](comparison.md), and its KITScenes protocol is
[below](#on-kitscenes).

## On AlpaSim, through the bridge

`paddock run --model alpamayo15 --bench alpasim` drives the model on the settings
of its stored AlpaSim result (`results/alpasim/alpamayo15/legacy-alpamayo_proxy_referee_20hz`,
made by AlpaSim's own Alpamayo driver) and builds its input the way that driver
does. The settings are the model's own
`adapters/models/alpamayo15/alpasim.yaml` and its driver config `paddock_bridge_alpamayo15.yaml`: four
f-theta cameras at 20 Hz (front wide, both cross cameras and the 30° telephoto),
a 3 s replayed warm-up with the driver called, 400 steps of 50 ms, and the 30
scenes pinned to their 26.01 artifacts. With `scene_set = "public_2601_30scenes"`
in `[benchmarks.alpasim]` it drives all 30.

The node switches to AlpaSim's inputs only on fields that profile's bridge
sends and no other client does, so every other benchmark gets exactly what it
got before (`tests/test_unchanged_inputs.py` pins seven of them, recorded from
the adapter before the change):

| Input | On AlpaSim | Switched on by |
|---|---|---|
| when the model is called | every 100 ms from the end of the warm-up, the model's own rate; the step between goes to `/observe`, and the bridge answers it from the last plan, re-timed and re-anchored to the car | `--replan-every-us 100000`, `--observe-between-replans` |
| cameras | all four, the telephoto as rendered at 30° where other benchmarks fill that slot with a crop of the front camera; f-theta frames go in as rendered, with no crop, resize or undistortion, as AlpaSim's driver does | a camera named `CAM_FRONT_TELE`; a camera with no pinhole entry |
| frames | the four newest per camera, 0.1 s apart | — |
| ego history | 16 poses 0.1 s apart from the 20 Hz pose stream | — |
| navigation text | AlpaSim's own rule (`route_nav.py`, ported from AlpaSim's `nav_text.py`): the route is moved into the newest pose's frame, and the first turn within 40 m becomes "Turn left in 20m" or similar; otherwise "Continue straight", and no text when AlpaSim has no route or it is unusable | `route_xy` with `route_timestamp_us` (`--send-route`, `--send-route-time`) |
| output | all 64 points 0.1 s apart, with the model's own headings, and the first 400 characters of its reasoning in `extras` (so in the per-replan log). Any spacing other than 0.1 s is refused | `plan_dt` (`--send-plan-dt`) |

The model reads poses and pixels only, so no speed or acceleration switch is
on. Sampling is the model's own (`top_p 0.98`, temperature 0.6, one sample), so
a scene's drive is one draw.

**Inputs checked against AlpaSim's own driver.** Ten scenes' recorded simulator
messages were replayed offline through both: AlpaSim's Alpamayo driver code, and
the bridge with this profile into the paddock node, the model replaced by a
recorder on both sides. On 1,681 model calls after the warm-up, the call times,
camera indices, all 16 frame times, pixels, prompt text, navigation text and
sampling settings were identical, and the ego history agreed to 3.8e-6 m and
8.1e-6°. The replay was run at the author's site and is not committed.

The 30-scene run is `results/alpasim/alpamayo15/20261029T200113-9d900c`: valid,
one rollout per scene, `scene_score` **0.7952** (progress 0.843), one at-fault
collision and four off-road events. On the first ten scenes it scores 0.9000,
the same as the stored native row on every scene, including the at-fault
collision on `04749bb9`. The stored row ran on simulator image `8a69f005`, this
one on `6dad7728`.

On the seven scenes whose route turns, the node logged the navigation text it
gave the model on every call. It equals, call for call, the text AlpaSim's own
driver code computes from the same run's recorded messages.

What remains different from the native runs:

- **JPEG decoding.** AlpaSim's driver decoded frames on the GPU (nvJPEG); the
  bridge decodes them with PIL. The two can differ by rounding; not measured.
- **The plan's height and tilt.** The bridge returns every plan flat (x, y and
  heading); the native driver kept the model's full poses on model calls.
  AlpaSim's controller is planar, so this is not expected to matter; not
  measured.
- **Warm-up calls.** Natively the model was also called during the warm-up,
  whose plans AlpaSim does not drive; here those frames are observed only.
- **Speed.** A call takes 2.75 s at the median through paddock, about 3–4 times
  the native runs' 0.7 s, varying with other load on a shared GPU. AlpaSim waits
  for every plan, so this changes wall time, not the drive.

`make check-alpamayo ALPAMAYO_PY=<repos>/alpamayo1.5/a1_5_venv/bin/python` runs
the adapter's input tests on the CPU without weights; `make check` skips them,
because they need torch and `alpamayo1_5`.

## On KITScenes

Scores NVIDIA Alpamayo-1.5-10B zero-shot on KITScenes-Multimodal, on **the same 223
keyframes** a previous Drive-JEPA run scored, so the two are directly comparable.

Nothing under `src/alpamayo1_5/` is modified — that is an upstream NVlabs clone.

### Quick start

```bash
export UV_PROJECT_ENVIRONMENT=$PWD/a1_5_venv
uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.12
uv sync --no-install-package flash-attn          # see "Attention backend" below

./a1_5_venv/bin/python eval_kitscenes/selftest.py            # CPU only, no GPU needed
./a1_5_venv/bin/python eval_kitscenes/run_kitscenes_alpamayo.py \
    --out runs/full --stream-download --prefetch --controls
./a1_5_venv/bin/python eval_kitscenes/report.py \
    --run runs/full/results.jsonl --out runs/full/report.md
```

No conda environment is involved. The `drive-jepa` conda env exists only to satisfy
NAVSIM; this harness reuses that run's *conventions* and its results pickle, nothing else.

### Protocol

Reproduced exactly from `Drive-JEPA/navsim_v2/scripts/inference/`:

| | |
|---|---|
| Samples | the 223 `(scene_id, keyframe)` rows in `kitscenes_e2e_final.pkl`, 140 scenes |
| Keyframes | `first=40, stride=90, last=N-51` (the paper's e2e protocol: non-overlapping 9 s windows) |
| Scoring grid | 0.5 s, subsampled from Alpamayo's 10 Hz output at indices `[4,9,14,19,24,29,…]` |
| ADE | mean L2 over **0–3 s** (6 grid points) |
| FDE@Ns | L2 at grid index `2N-1` |
| Ground truth | `to_relative_poses(poses[k], poses[k+5(i+1)])` from `poses.txt` |

`metrics.displacement_errors` is a byte-for-byte port; feeding the reference run's own
predictions back through it reproduces all 223 recorded values to **0.0e+00**.

Alpamayo natively predicts 64 waypoints at 10 Hz (6.4 s). Dense 10 Hz metrics are
reported as `ade_10hz_*` but are **not** comparable to the headline — averaging over
0–3 s at 10 Hz includes the near-zero 0.1–0.4 s region and lowers ADE systematically.

### Adaptation

**Ego frame.** `poses.txt` is TUM `x y z qx qy qz qw` in `base_frame` (== `lidar_top`, the
roof pod). History and future are transformed into the t0 frame with full SO(3), mirroring
`load_physical_aiavdataset.py`. KITScenes is natively 10 Hz, so the 16-step history and
4 camera frames are taken directly with no resampling — a better fit than Drive-JEPA got,
which NAVSIM forced to 2 Hz.

**Cameras.** KITScenes ring cameras are 3504×2272 pinholes at ~87° hFOV. Slots are matched
by yaw (KITScenes `+60.9 / +0.97 / −58.5` vs Alpamayo's canonical `+55 / 0 / −55`):

| slot | sensor | warp |
|---|---|---|
| 0 `cross_left_120fov` | `camera_ring_front_left` | fill |
| 1 `front_wide_120fov` | `camera_ring_front` | fill |
| 2 `cross_right_120fov` | `camera_ring_front_right` | fill |
| 6 `front_tele_30fov` | `camera_ring_front` | **zoom** (exact 30° digital telephoto) |

All crops are 16:9 and centred on the **principal point**, not the image centre (they
differ by 11.5 px on `camera_ring_front`). Only the aspect ratio is load-bearing — the
Qwen3-VL processor resizes to 320×576 regardless.

The telephoto slot is geometrically exact, because 30° is *narrower* than the 87° source.
The three 120° slots cannot be: no crop invents field of view, so objects subtend **1.82×**
the expected angle (focal/width 0.526 vs 0.289), biasing depth. `--fov-mode letterbox`
trades that for projective fidelity at the cost of a 70% black border; over the full 223
samples it is worse (medoid ADE 1.009 vs 0.875), so `fill` is the default.

### Arms

Every sample is scored under each arm in a single pass, because scenes are deleted after
scoring and a second pass costs another 594 GB.

| arm | fov | nav | K |
|---|---|---|---|
| `nonav` | fill | — | 6 |
| `nav` | fill | GT-derived route text | 6 |
| `letterbox` | letterbox | — | 6 |

K=6 rather than 16 because the GPU is shared and 6 is the number NVIDIA publishes
(minADE₆); with K=6 the `min_ade@6` column equals `min_ade`.

**Why `nav` exists.** The reference run used `--command gt`, which reads `poses[k+50]` and
hands the planner a left/straight/right one-hot on 39 of 223 samples. Withholding the
equivalent from Alpamayo would lose the comparison on a technicality, so `nav` renders the
same hint (same 5.0 m lateral threshold) as `<|route_start|>…<|route_end|>` text. The two
prompts differ *only* in that span.

One K-draw pass yields every sampling statistic: single-draw ADE (strictly comparable to
a deterministic planner), medoid (the fair consensus number), and minADE@6. When K>6 the
minADE@6 figure is estimated by averaging the minimum over random 6-subsets.

**Which number to quote.** The medoid. minADE is oracle-selected and flatters the sampler;
mean-over-K penalises multimodality the metric cannot reward. Note the medoid is not always
the best draw — on one sample it scored 3.59 against a first draw of 1.30.

### Operational notes

**Shared GPU.** This box's GPU is shared. `model_loader.acquire_vram` reserves the memory
*before* the 12-minute checkpoint read and releases it into our own caching-allocator pool
just before `.to("cuda")`; PyTorch keeps freed blocks in-process, so the model's allocations
are served from what we already hold. Checking free memory and then loading is a race that
loses — the first attempt did exactly that. During scoring, OOM triggers retry-with-backoff;
anything that still fails is written to `skipped.json` rather than silently dropped, since a
shrunken sample set would quietly invalidate the paired comparison.

**Attention backend.** `flash-attn` must not be installed: the local CUDA toolkit is 12.0,
too old to emit sm_120 for this Blackwell card. `config.json` requests `flash_attention_2`
and `ReasoningVLAConfig.__init__` defaults to it, so `attn="sdpa"` is passed explicitly and
asserted after load.

**Gated weights.** `vlm_name_or_path` points at the gated `nvidia/Cosmos-Reason2-8B` and is
consumed *during* config construction, so the config is built by hand with the local path.
Only the Qwen3VL config and tokenizer come from there; the VLM weights are in the Alpamayo
checkpoint.

**Streaming.** 594 GB of scene tars, fetched one at a time and deleted after scoring — peak
disk ~25 GB (largest single tar is 13.6 GB). Sustained ~75 MB/s over the full run.
`--prefetch` overlaps the next download with the current scene's inference, and declines to
prefetch when the next tar is large or disk is tight.

The dataset is **gated**, and `kitscenes.download` exits 0 even when every scene fails, so
an unauthenticated run looks like a very fast successful one. `check_hf_auth` refuses to
start without a working token, and every fetch is verified by checking the extracted
directory exists — exit status alone is not trusted.

### Verification

`selftest.py` (CPU only) — the load-bearing check is **GT parity**: ground truth recomputed
from `poses.txt` must reproduce the reference pickle's `human_poses`. It agrees to
**1.4e-14 m**, and `ego_speed` to 5.7e-07, which proves the frame indexing, pose parsing and
keyframe arithmetic all line up with the run being compared against.

`--controls` re-runs one sample with images zeroed and with the left/right cross cameras
swapped. The **blind** control is decisive and passes: ADE degrades 2.8–4.1× and the
chain-of-causation collapses from a specific observation ("cyclists crossing our right-turn
path") to a fabrication ("upcoming speed bump"). The **swap** control is *inconclusive* —
both tested scenarios turned out to be front-camera-dominated, and a single K=1 draw cannot
resolve the difference against sampler variance. Left/right correctness therefore rests on
the calibration yaws and on visual inspection of the contact sheet
(`selftest.py --contact-sheet`), not on an ablation.

The SO(3)-vs-yaw-only frame correction is applied to every prediction; its magnitude
reached **8.8 cm** over the full 669-row run, larger than the 3.5 cm seen on the six
smoke-test samples.  It is a diagnostic, not an error -- but the self-test's 5 cm
assertion is calibrated to the smoke-test scenes and is not a bound on the dataset.

### Results (full run, 223 samples)

| system | ADE | FDE@3s | longitudinal | lateral |
|---|---:|---:|---:|---:|
| Constant velocity | 1.381 | 3.091 | — | — |
| Drive-JEPA (zero-shot) | 2.364 | 4.199 | 2.317 | 0.209 |
| **Alpamayo-1.5 `nav` medoid** | **0.875** | **1.862** | 0.718 | 0.331 |
| Alpamayo-1.5 `nav` minADE@6 | 0.543 | 1.035 | — | — |

Paired over all 223 samples: mean difference −1.488 m, Alpamayo wins 168/223 (75%),
Wilcoxon signed-rank z = −9.19, p = 4.05e-20. Ground-truth agreement 2.22e-14 m.

The route hint barely matters (`nav` 0.875 vs `nonav` 0.892), so Alpamayo's advantage is
not an artefact of the hint Drive-JEPA received. The decomposition is the substantive
finding: Drive-JEPA's *path* is more accurate (lateral 0.209 vs 0.331), and its entire
deficit is longitudinal — it does not predict speed well.

Wall-clock ~2 h for the full run, of which only 34 min was GPU; the rest is download.

### Known limitations

- **Field of view** — see above; the 1.82× scale error on the 120° slots is unavoidable.
- **Roof-pod origin** — `poses.txt` is in the pod frame, not a rear-axle ego frame. Both
  models consume the same stream so the comparison is unaffected, but both are penalised on
  turns relative to a rear-axle ground truth. 18% of samples exceed 15° of yaw change.
- **Zero-shot** — neither model was trained on KITScenes; these are not comparable to
  KITScenes-trained baselines.
- **Sampling** — Alpamayo samples (T=0.6, top-p=0.98); seeds are fixed per (sample, arm).

### Closed loop and navigation (September 2026)

* **Navigation text.** The adapter used to call `helper.create_message(...,
  nav_text=None)`. Bench2Drive supplies a route command every tick and the
  model's own interface takes an instruction, so the agent now sends the next
  junction maneuver and the metres to it (`nav_maneuver`, `nav_distance_m`) and
  `nav_text_for()` phrases it the way `notebooks/inference_nav.ipynb` and
  `nav_utils` do: "Turn left in 37m", "Turn right in 9m", "Continue straight".
  Sources without the fields (NAVSIM) fall back to the 4-way command ("Turn
  left"). `ALPAMAYO_NAV=0` gives the no-nav baseline.
* **Sampling.** Defaults follow the reference inference notebook: top-p 0.98,
  temperature 0.6, one sample, 256 generation tokens
  (`ALPAMAYO_NUM_TRAJ_SAMPLES`, `ALPAMAYO_TEMPERATURE`, `ALPAMAYO_TOP_P`,
  `ALPAMAYO_MAX_GEN`). Six samples cost 3.7 s per call against 1.0 s, because
  the VLM writes one chain-of-thought per sample (`num_return_sequences`).
  Classifier-free guidance (`..._cfg_nav`, 60 GB+) is optional in the notebook
  and not used. `ALPAMAYO_SAVE_REASONING=1` writes the model's chain-of-thought
  reasoning alongside each prediction, for later analysis.
* **History.** Frames stamped no later than the previous one are dropped
  (KITScenes' clamped warm-up repeats frame 0). The chain-of-causation is read
  from `extra["cot"]`, kept in `diagnostics()` and logged every 100 calls.
* **Planning rate.** `model.yaml` declares `planning_hz: 10.0`, the 0.1 s grid
  its frames and poses are on, so Bench2Drive asks it once every 0.1 s and
  follows the last plan in between (since November 2026). Before that it was
  asked every 20 Hz tick unless the host set `PADDOCK_INFER_HZ`, which is how its
  10 Hz Bench2Drive cell was driven.
* **Behaviour worth knowing.** From standstill the model derives zero speed
  from its own pose history and plans near-zero motion until a sample commits;
  in CARLA that is 20+ s of waiting followed by a hard launch, and at junctions
  it stops short of the turn although the instruction says "Turn left in 4m"
  (its own reasoning: "stop for the red light", "keep lane"). On nuScenes the
  same mechanism inflates its error on the first keyframe of each scene (L2
  avg 9.5 m there, 1.2 m elsewhere). Results: Bench2Drive 24 routes DS 44.6
  (best Overtaking of the four stacks); nuScenes val L2 avg 1.49 m (best of
  three). On the 220-route protocol, over 216 routes
  (`results/bench2drive/alpamayo15/full220-836dda734659a86c-f81033`), DS 37.66,
  completing 53 routes. See `docs/benchmarks/bench2drive.md` and `nuscenes.md`.
