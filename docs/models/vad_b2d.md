# VAD-B2D

<!-- summary: written by `python -m tools.doc_summaries` from model.yaml, benchmark.yaml and repos.yaml -->

| | |
|---|---|
| Upstream | [Thinklab-SJTU/Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) at `498c1f7` |
| Adapter | [`adapters/models/vad_b2d/`](../../src/paddock/adapters/models/vad_b2d/) |
| Environment | conda, Python 3.9 |
| Cameras | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |
| Plan | waypoints, 0.5 s apart, 3 s ahead |
| Paddock can run it on | [hugsim](../benchmarks/hugsim.md), [nuscenes](../benchmarks/nuscenes.md), [bench2drive](../benchmarks/bench2drive.md), [kitscenes](../benchmarks/kitscenes.md), [nuplan](../benchmarks/nuplan.md), [neuroncap](../benchmarks/neuroncap.md), [navsim](../benchmarks/navsim.md), [navsim_v1](../benchmarks/navsim_v1.md) |
| Size, licence, training data | [comparison.md](comparison.md) |

<!-- /summary -->

> Paths in this guide are written as placeholders: `<repos>` is `[roots.repos]` from your `paddock.toml`,
> `<weights>` is `[roots.weights]`.

## What it is

VAD-Base trained by the Bench2Drive authors on their CARLA dataset (Bench2Drive base less a validation split),
released in [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo). The architecture is
[`vad`](vad.md)'s, but it plans six trajectories, one per Bench2Drive command, and the command picks the one
driven. Everything it shares with [`uniad_b2d`](uniad_b2d.md) -- code, environment, the virtual lidar, the
CAN-bus layout, the colour order, the command mapping, the rate and rig choices -- is written up there.

## Code and weights

Bench2DriveZoo at `498c1f7` (**CC BY-NC-ND 4.0**), config `adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py`.

| file | source | sha256 |
|---|---|---|
| `<weights>/bench2drivezoo/vad_b2d_base.pth` (699,792,372 bytes) | `huggingface.co/rethinklab/Bench2DriveZoo` | `0f2d48b5c6cdb06d0dffb21612791fe740f3f5c7273b3b5e78e1f19e4a527eab` |

`paddock setup --model vad_b2d` runs `uniad_b2d`'s setup (Bench2DriveZoo's ops, in place).

## What is its own

- **Normalisation** as VAD's (torchvision mean and std, `to_rgb=True`), on BGR frames after the JPEG
  quality-20 round trip.
- **Pipeline**: the config's `inference_only_pipeline`, as the agent runs it: normalise, pad to 32, scale by
  0.8, pad again. The training pipeline scales 1600×900 straight to 1280×720; the agent's first pad adds 28
  rows of zeros below the image before the scale, so the image content is the same to within a pixel row and
  the padded area is larger. The adapter follows the agent, since it is what the config names for inference.
- **Command**: `ego_fut_cmd`, a one-hot of the six commands; the plan driven is
  `ego_fut_preds[command].cumsum()`, as the agent takes it. Benchmarks as `uniad_b2d`'s; navsim needs
  `VAD_B2D_HORIZON_S = "4.0"`. The agent also builds `ego_lcf_feat`; this
  checkpoint never reads it (`ego_lcf_feat_idx=None`).
- **The plan** is the future position of Bench2Drive's lidar origin in the current lidar frame
  (`B2D_vad_dataset.py:373-383`), handled as for the others.

The agent's RGB moves this model's 3 s waypoint by up to 1.2 m on the golden frames. Bench2DriveZoo published
42.35 DS for it, driven every 20 Hz tick with RGB frames.

## Verification

Golden test against Bench2DriveZoo's own path (`tests/fixtures/golden_vad_b2d.json`): **0.000 m maximum
difference** on four keyframes, exact after reset (9 Nov 2026). Node: 0.29-0.31 s per `/infer` (0.59 s first),
about 2.8 GB of GPU memory.
