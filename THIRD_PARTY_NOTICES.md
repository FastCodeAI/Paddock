# Third-party notices

Paddock's own code is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). A few
files in this repository hold material from other projects. They are listed below with
their source and licence, and that material keeps its own licence.

Paddock contains no upstream project's source tree, no dataset and no model weights.
`paddock setup` downloads each upstream project from its own repository onto your
machine, and `paddock data fetch` downloads each dataset from its owner's host. Each
comes under its owner's licence and terms, which you accept when you download it.

## Code adapted from other projects

| File | Source | Licence |
|---|---|---|
| `src/paddock/adapters/models/alpamayo15/route_nav.py`, in part | [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim): `nav_text.py` and `alpamayo_base.py` | Apache-2.0, Copyright (c) 2026 NVIDIA Corporation |
| `src/paddock/adapters/models/visionpilot/policy.py`, in part, ported from C++ | [Autoware Foundation VisionPilot](https://github.com/autowarefoundation/vision_pilot): `longitudinal_fusion.cpp`, `longitudinal_planning.cpp`, `planning.cpp` and `auto_speed.cpp` | Apache-2.0 |
| `src/paddock/adapters/benchmarks/bench2drive/pid.py`, reimplemented | The control law of the PID controller in [TCP](https://github.com/OpenDriveLab/TCP) (`leaderboard/team_code/pid_controller.py`), which the UniAD and VAD agents of Bench2Drive's baselines also use | Apache-2.0 |
| `src/paddock/adapters/models/meteor/policy.py`: METEOR's camera rig, as numbers (intrinsics, and three rows of each camera's transform) | [TIER IV METEOR](https://github.com/tier4/METEOR): `bevlane/ref_rig_drs.npz` | Apache-2.0 |

Each file names its source, at the top or beside the copied lines.

## Data taken from other projects

| Files | Source | Licence |
|---|---|---|
| `src/paddock/adapters/benchmarks/nuscenes/val_scenes.json` | The nuScenes validation split, as listed in Motional's [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (`python-sdk/nuscenes/utils/splits.py`) | Apache-2.0, Copyright 2021 Motional |
| `src/paddock/adapters/benchmarks/neuroncap/scenarios_10hz/` | [neuro-ncap](https://github.com/atonderski/neuro-ncap)'s `scenarios/`, with the time step changed from 0.5 s to 0.1 s | MIT, Copyright (c) 2024 William Ljungbergh & Adam Tonderski |
| `src/paddock/adapters/benchmarks/alpasim/scenes/` | Scene IDs and file paths as listed in [NVIDIA AlpaSim](https://github.com/NVlabs/alpasim)'s `data/scenes/` | Apache-2.0, Copyright 2025 NVIDIA Corporation |
| `src/paddock/adapters/benchmarks/bench2drive/tests/fixtures/routes_sample.xml` | The id, town and scenario type of two routes in [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)'s `leaderboard/data/bench2drive220.xml`; the rest of each entry is left out | CC BY-NC-ND 4.0 |

The neuroNCAP scenarios refer to objects in nuScenes scenes by their nuScenes IDs.
nuScenes itself is under its own terms (see the last table).

## Patches to other projects

`patches/` holds edits that `paddock setup` applies to a project after downloading it.
Each patch quotes a few lines of that project's code, and those lines keep that
project's licence.

| Directory | Project | Licence of the code the patches edit |
|---|---|---|
| `patches/neuro-ncap/` | [neuro-ncap](https://github.com/atonderski/neuro-ncap) | MIT, Copyright (c) 2024 William Ljungbergh & Adam Tonderski |
| `patches/Drive-JEPA/` | [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA) | Apache-2.0 |
| `patches/HUGSIM/` | [HUGSIM](https://github.com/hyzhou404/HUGSIM) | MIT, Copyright (c) 2024 X-Dimensional Representations Lab (X-D Lab) of Zhejiang University |
| `patches/Bench2Drive/` | [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | The file they edit, `leaderboard/leaderboard/leaderboard_evaluator.py`, is headed "Copyright (c) 2018-2019 Intel Corporation" and "licensed under the terms of the MIT license". Bench2Drive's README says: "All assets and code are under the CC-BY-NC-ND unless specified otherwise." |

## Results, and the datasets behind them

The scores in `results/`, and the tables and charts drawn from them in this repository,
are measurements of models on the benchmarks below. They are scores and run settings,
with no images, sensor readings or labels from the datasets. Some datasets' terms also
cover results derived from the data, so those terms may apply to the scores.

| Benchmarks | Data | Its licence |
|---|---|---|
| nuScenes, neuroNCAP | [nuScenes](https://www.nuscenes.org/nuscenes) (Motional) | [CC BY-NC-SA 4.0 and Motional's terms](https://www.nuscenes.org/terms-of-use) |
| nuPlan, NAVSIM v1 and v2 | [nuPlan](https://www.nuscenes.org/nuplan) (Motional), and [OpenScene](https://github.com/OpenDriveLab/OpenScene), a re-release of nuPlan | [CC BY-NC-SA 4.0 and Motional's terms](https://www.nuscenes.org/terms-of-use) |
| KITScenes | [KITScenes Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) (KIT) | CC BY-NC 4.0 and KIT's terms |
| comma2k19 | [comma2k19](https://github.com/commaai/comma2k19) (comma.ai) | MIT |
| Bench2Drive | [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)'s routes, in the [CARLA](https://github.com/carla-simulator/carla) simulator | CC BY-NC-ND 4.0 (Bench2Drive); MIT code and CC BY assets (CARLA) |
| AlpaSim | [PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) (NVIDIA) | NVIDIA Autonomous Vehicles NuRec Dataset License Agreement |
| HUGSIM | [HUGSIM's scenes](https://huggingface.co/datasets/XDimLab/HUGSIM), built from nuScenes, KITTI-360, Waymo Open Dataset and PandaSet drives | Tagged MIT; the source datasets have their own terms |

Please cite each model and dataset you use; their papers are linked from the
[README](README.md). Names of models, datasets and organisations identify the projects
Paddock works with. Being listed here does not mean a project endorses Paddock.

## MIT licence text

The MIT-licensed material above is under the following copyright notices:

- Copyright (c) 2024 William Ljungbergh & Adam Tonderski (neuro-ncap)
- Copyright (c) 2024 X-Dimensional Representations Lab (X-D Lab) of Zhejiang University (HUGSIM)
- Copyright (c) 2018-2019 Intel Corporation (the Bench2Drive file the patches edit), and
  Copyright (c) 2019 CARLA (its `leaderboard/` folder)

each with this permission notice:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The Apache License 2.0 text, which covers the Apache-2.0 material above, is in
[LICENSE](LICENSE).
