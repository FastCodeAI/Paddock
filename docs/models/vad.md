# VAD

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [hustvl/VAD](https://github.com/hustvl/VAD) at `1688c4b` |
| Adapter | [`adapters/models/vad/`](../../src/paddock/adapters/models/vad/) |
| Environment | conda, Python 3.9 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [hugsim](../benchmarks/hugsim.md), [nuscenes](../benchmarks/nuscenes.md), [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuplan](../benchmarks/nuplan.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md), [navhard](../benchmarks/navhard.md), [alpasim](../benchmarks/alpasim.md), [comma2k19](../benchmarks/comma2k19.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is `[roots.repos]` from your `paddock.toml`,
> `<weights>` is `[roots.weights]`.

## What it is

[VAD](https://github.com/hustvl/VAD) (HUST Vision Lab, ICCV 2023) plans from a vectorised scene: a BEVFormer
grid (200 × 200 cells over 30 m across and 60 m along; ResNet-50; six cameras scaled to 1280×720) feeds agent
queries, each with its own motion forecast, and map queries for lane dividers, crossings and road
boundaries. One ego query attends to both and plans three trajectories, one per command (right, left,
straight), each six offsets 0.5 s apart (3 s); the command picks the one driven.

Paddock runs the released VAD-Base checkpoint, trained on nuScenes `train`, on the upstream code. What VAD
shares with UniAD -- the environment, the lidar frame, the CAN bus, the frame of the plan, the history grid,
the verification -- is written up once in [uniad.md](uniad.md); this page is what is VAD's own.

## Code and weights

`hustvl/VAD` at `1688c4b`, unpatched. The config is `projects/configs/VAD/VAD_base_stage_2.py`, the one the
README links to the checkpoint; it differs from `VAD_base_e2e.py` only in loss weights, epochs and `load_from`.

| file | source | sha256 |
|---|---|---|
| `<weights>/vad/VAD_base.pth` (707,221,363 bytes) | Google Drive id `1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS` (VAD-Base in the README; `gdown` fetches it) | `4086547b921fee23139b09f0b79fd53a8b66dd02dd791ea80454d8237cefe4c4` |

The environment is UniAD's, built in the order [uniad.md](uniad.md) gives: Bench2DriveZoo's checkout first, whose two
`roiaware_pool3d` kernels the build copies (CC BY-NC-ND 4.0), then `paddock setup --model uniad`. `[models.vad] venv`
points at the same env.

## What is VAD's own

- **Normalisation**: the one the released weights were trained with, caffe-style (mean 103.530, 116.280,
  123.675; std 1; `to_rgb=False`). The frames go in BGR, as `mmcv.imread` reads them, and stay BGR. This is
  not the config's: see [the normalisation](#the-normalisation-and-every-vad-result-before-20-nov-2026) below.
  Then the test pipeline scales every camera by 0.8 to 1280×720, `lidar2img` with them, and pads to 32.
- **Command**: `ego_fut_cmd`, an integer one-hot of [right, left, straight] (`vad_nuscenes_converter.py`);
  FOLLOW_REFERENCE is straight. It selects one of the three plans: `ego_fut_preds[command].cumsum()`, as VAD's
  own `simple_test_pts` does.
- **No ego history**: VAD-Base sets `ego_his_encoder=None` and `ego_lcf_feat_idx=None`, so neither the past
  trajectory nor the ego features are read.
- **The plan** is the future position of the LIDAR_TOP origin in the current lidar frame
  (`vad_nuscenes_converter.py:436` takes the LIDAR_TOP sensor's global pose, 443-448 bring it into the current
  lidar frame), handled as for UniAD.
- **Ground truth the forward reads**: VAD's `forward_test` scores its own detections, motion and plan, and
  indexes ground truth to do it. The adapter hands it no boxes and a future marked invalid; the metrics work on
  a copy of the predictions.
- **The first plan** of an episode has no previous BEV grid. On scene-0103 it comes out short (7.5 m in 3 s
  at 8.4 m/s) on VAD's own path as on paddock's; the next one, 0.5 s later, is normal.

**HUGSIM's client** (`hyzhou404/VAD_SIM`) is not used: it never calls `model.eval()`, so dropout stays on and
a plan is a draw; it normalises HUGSIM's RGB renders with the caffe mean meant for BGR, so red and blue are
swapped; and it sets `use_can_bus=False`. Its normalisation's scale is right: the caffe values, with no division
by a std, are the ones the released weights were trained with (an earlier version of this page said the
opposite, wrongly). The adapter feeds BGR with those values, in eval mode, with the CAN bus on.

## The normalisation, and every VAD result before 20 Nov 2026

The config the adapter loads, `VAD_base_stage_2.py` at `1688c4b`, holds torchvision's `img_norm_cfg` (mean
123.675, 116.28, 103.53; std 58.395, 57.12, 57.375; `to_rgb=True`). The released checkpoint was not trained with
it. VAD's own docs at the same sha (`docs/train_eval.md`, "Reproduce results with pre-trained weights") say to
evaluate the released weights with

```python
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
```

"this is the original setting when we train the model", changed in the configs later on the advice of
[hustvl/VAD issue #9](https://github.com/hustvl/VAD/issues/9): "If you use the new `img_norm_cfg` config, you
will get wrong metric results and visualizations." So the adapter changes the config before it builds the
pipeline (`VADPolicy._configure`, `with_normalisation` in [`policy.py`](../../src/paddock/adapters/models/vad/policy.py)):
every `NormalizeMultiviewImage` step of the test pipeline gets the values above. Fed the config's values
instead, the model reads red and blue swapped and about 1/58 of its training scale.

**Every VAD result paddock made before this fix (20 Nov 2026) used the config's normalisation, so every one is
wrong in that way**, on every benchmark: the stored runs under `results/<bench>/vad/` (nuScenes, KITScenes,
comma2k19, NAVSIM v1, NAVSIM v2, navhard, neuroNCAP, AlpaSim, Bench2Drive, HUGSIM), and the VAD cells of the
support matrix that name them, which the README marks **S** (stale). Re-running them gives numbers for the model
as it was trained. On the golden keyframes the change moves VAD's plan by up to 5.5 m at 3 s on four scene-0103
keyframes, and by a median 3.5 m (max 8.2 m) over scene-0103 and 2.4 m (max 5.3 m) over scene-0916, with every
other input the same. Bench2DriveZoo's VAD checkpoint is a different model, trained with Bench2DriveZoo's own
config (which holds torchvision's values); nothing here changes how it is fed.

## Benchmarks

As [UniAD's](uniad.md#benchmarks): nuscenes, hugsim, bench2drive, kitscenes, nuplan, neuroncap and alpasim run
(on neuroncap the plan is held between 0.5 s steps; on alpasim, UniAD's profile, flag for flag);
navsim, navsim_v1 and navhard need `VAD_HORIZON_S = "4.0"`, set in a pairing for each of the three
(`[models.vad.benchmarks.navsim.env]` and likewise `navsim_v1` and `navhard`: a pairing matches the benchmark
name exactly; `paddock.toml.example`); without it every token is refused. model.yaml lists the setting in
`score_notes` and `ranked_with_note`, as UniAD's horizon is, so such a run is ranked with a note. comma2k19
sends one camera: it is CAM_FRONT, and the other five slots are black at nuScenes' own
calibration, as for UniAD ([uniad.md](uniad.md)).

## Verification

Golden test, as for UniAD (`tests/fixtures/golden_vad.json`, written from VAD's own converter, dataset and
`test_pipeline`, with the `img_norm_cfg` VAD's docs give for the released weights): **0.000 m maximum
difference** on four scene-0103 keyframes, for the stored and the canonical quaternion, and an exact repeat
after reset (20 Nov 2026; the fixture of 9 Nov was made with the config's normalisation, and the adapter as it
was then misses this one by 5.5 m). The quaternion's sign moves VAD's 3 s waypoint by up to 0.37 m; placing
each camera at its own ego pose, by 0.58 m (at the first keyframe, which has no previous grid; 0.13 m at
most after). On nuScenes VAD declares `inputs: {nuscenes:
vad}`, so the harness sends those as VAD's dataset has them (docs/benchmarks/nuscenes.md, "The datasets' own
inputs"): on scene-0103 and scene-0916 every input the adapter hands its pipeline is the dataset's within
1e-9, the command exactly (`test_scene_golden.py`, in `make check`), and over all 6,019 val keyframes against
VAD's converter `lidar2img` is within 1.3e-12 and every command the same (10 Nov 2026). In scene order, VAD's
own evaluation of every keyframe of both scenes (converter, `VADCustomNuScenesDataset`, dataloader, test loop)
and the harness's scene-order request, sent as the runner sends it, agree: **0.000 m** at every keyframe sent
(20 Nov 2026, both fixtures written again with the released weights' normalisation; their recorded inputs are
unchanged). Node: 0.29-0.30 s per `/infer` (0.64 s first),
about 3.7 GB of GPU memory.

A step on nuScenes (six 1600×900 JPEGs, the card shared; medians, ms) spends more on preparing than on the
model: read + decode 21, BGR + float32 27, the rest of `_results` 29, the test pipeline (normalise, scale to
1280×720, pad, format) 126, collate 20, to GPU 6, forward 134. `UNIAD_VAD_THREADS` ([UniAD's
section](uniad.md#cost-of-a-step-and-the-threads-that-prepare-frames)) prepares the next frame while the model
runs and holds OpenCV to four threads: same plans (golden 0.000 m; identical in an A/B on twelve keyframes),
a five-keyframe window 2.05 / 1.96 → 1.06 / 1.12 s, an in-order keyframe 409 / 396 → 211 / 228 ms, CPU per
window 5.0 → 2.5 s (OpenCV had been using every core). A single `/infer` has nothing to overlap: 0.35 s either
way.
