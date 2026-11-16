# Paddock public release: licence terms of the models, benchmarks and datasets it uses

*Prepared by the Paddock team for Renesas Legal on 10 November 2026. This is a fact sheet
to support a legal review, not legal advice. All terms were read on that date. Where this
document interprets a term, it says so; the decisions are Legal's.*

## 1. Summary

Paddock is Renesas's open-source tool for testing self-driving models on public
benchmarks. We plan to publish its source code and the scores it has measured. We will not
host or pass on any dataset or model weights: each user downloads those from the original
publisher and accepts that publisher's terms. The repository holds no dataset files, camera
images, videos or model weights. The website built from it, though, shows four videos of
nuScenes scenes, which are already public on Hugging Face (decision 9).

Terms used in this document:

- **Model**: software that takes a car's camera images and outputs the path the car
  should drive. A model has **code** and **weights**, the file of trained numbers that
  makes it work. Code and weights often have different licences.
- **Benchmark**: a fixed test that gives a model a score. It runs either on a **dataset**
  of recorded drives or inside a **simulator**, a program that drives a virtual car
  through a recorded or synthetic scene.
- **Upstream**: the original publisher of a model, benchmark or dataset.
- **Patch**: a file that lists line-by-line edits to someone else's code.

### 1.1 Recommendation

- **Framework (Paddock's code): yes, after seven small clean-ups** (section 4).
- **Results: yes for nine of the ten benchmarks.** comma2k19's results need only credit.
  The other eight need one condition: Legal accepts that publishing this open, free
  evaluation is non-commercial research use. Credits and licence notices are then enough.
  Written confirmation from Motional would make it firm for five of those eight.
- **AlpaSim results: no**, unless NVIDIA agrees in writing. Its scene licence makes "any
  results of benchmarking" NVIDIA's confidential information.

| Part of the release | Release? | Why | How to make it a clear yes |
|---|---|---|---|
| Paddock's code | Yes, after clean-ups | Upstream code is linked, not copied. No upstream code licence is copyleft. The few copied lines are under MIT or Apache-2.0, except in two Bench2Drive patches. | The clean-ups in section 4 |
| Results on comma2k19 | Yes | The data is under MIT. | Credit comma.ai |
| Results on nuScenes, nuPlan, NAVSIM v1 and v2, neuroNCAP | Yes, if non-commercial | Motional's data is for non-commercial use, and its terms cover derived "tables and charts". Its FAQ names "educational use and some research use" as free, and prices commercial use. | Credit and a CC BY-NC-SA 4.0 notice on these results. For a firm yes, Motional's written confirmation. |
| Results on KITScenes | Yes, if non-commercial | CC BY-NC 4.0, and the data is "intended for academic research". KIT also asks users to wait for a more stable release before final benchmark reporting. | Credit and notice. For a firm yes, KIT's confirmation. |
| Results on HUGSIM | Yes, if non-commercial | Its scenes are built from nuScenes, KITTI-360 and Waymo data, all non-commercial. | Credit, including Waymo's attribution sentence |
| Results on Bench2Drive | Yes, if non-commercial | The benchmark's code is CC BY-NC-ND 4.0. | Credit. For a firm yes, Thinklab's confirmation. |
| Scores of AutoE2E, Alpamayo 1.5, Qwen-Drive and VisionPilot | Yes, if non-commercial | AutoE2E's weights are non-commercial, and Alpamayo's model card says the same. Waymo's licence reaches models trained on its data, which Qwen-Drive and VisionPilot partly were. We publish scores only, and Alpamayo's licence puts no limit on outputs. | Credit. Ask NVIDIA which Alpamayo statement governs. |
| Results on AlpaSim | No | NVIDIA's agreement makes "any results of benchmarking" confidential, and limits use to internal development with NVIDIA technology. | NVIDIA's written permission. Until then, release without AlpaSim results. |

"Yes, if non-commercial" means: if Legal accepts the release as non-commercial research
use (decision 1 below).

### 1.2 The licences, grouped

| Group | What it allows | Where it applies |
|---|---|---|
| **A. Permissive**: MIT, Apache-2.0, BSD-3-Clause, CC0, CC BY, OpenMDW-1.1 | Any use, commercial included. Keep the licence and copyright notices. | Paddock itself. The code of all eleven models, and of NAVSIM, KITScenes' library, comma2k19, neuroNCAP, AlpaSim, HUGSIM, neurad-studio and CARLA. The weights of METEOR, OpenPilot, Qwen-Drive, UniAD, VisionPilot, Alpamayo 2 Edge, WA-JEPA, Drive-JEPA and Alpamayo 1.5 (the last three also in G). The data of comma2k19, PandaSet, 3DRealCar and CARLA's assets. |
| **B. Commercial use, with conditions**: NVIDIA Open Model License | Commercial use, if NVIDIA's Trustworthy AI terms are followed and its safety guardrails are kept. Attribution when redistributing. | Cosmos-Reason2-8B, the model Alpamayo 1.5 is built on |
| **C. Non-commercial**: CC BY-NC 4.0, CC BY-NC-SA 3.0 and 4.0, Motional's Dataset Terms, Waymo's licence | Non-commercial use only, with credit. ShareAlike (SA) licences also require adapted material to carry the same licence. | Data: nuScenes, nuPlan, OpenScene (NAVSIM), KITScenes, KITTI-360, Waymo, OpenLane. Weights: AutoE2E. |
| **D. Non-commercial, no derivatives**: CC BY-NC-ND 4.0 | Non-commercial use, with credit. Exact copies may be shared; changed versions may not. | Bench2Drive's code and route lists; Bench2DriveZoo, which UniAD's and VAD's environment is built from |
| **E. Internal use only, results confidential**: NVIDIA AV NuRec Dataset License Agreement | Internal development of autonomous vehicles using NVIDIA technology. No disclosure of benchmark results. Expires 12 months after download. | AlpaSim's scenes |
| **F. Permission needed for commercial use**: sunnypilot's custom licence | Viewing and modifying. Commercial, for-profit or closed-source use needs written permission. | sunnypilot, listed in Paddock but never used |
| **G. Unclear or conflicting** | Needs the publisher's confirmation | Alpamayo 1.5 (permissive licence, but its card says non-commercial). HUGSIM's scenes (MIT tag over non-commercial sources). neuroNCAP's renderer files (no licence stated). Drive-JEPA's and WA-JEPA's weights (licence tag only, no text). VAD's weights (no licence stated). |

### 1.3 Decisions for Legal

1. **Non-commercial use.** Is publishing this open, free evaluation, by Renesas,
   non-commercial research use? This decides the results on eight benchmarks and the
   scores of four models. If not, commercial licences are needed; Motional sells one. See
   section 3.
2. **Licence on results from Motional's data.** Must the results for nuScenes, nuPlan,
   NAVSIM and neuroNCAP carry CC BY-NC-SA 4.0 rather than Paddock's Apache-2.0? See
   section 3.
3. **AlpaSim.** Ask NVIDIA for written permission, or release without AlpaSim results?
   Also, who accepted NVIDIA's agreement, and when? It expires 12 months after the first
   download, which was no later than 7 September 2026. See section 7.
4. **Bench2Drive patches.** Does the MIT header of the file they edit cover them, or does
   Bench2Drive's CC BY-NC-ND licence? See sections 4 and 7.
5. **HUGSIM.** Rely on the scenes' MIT tag, or treat them as non-commercial like their
   sources? See section 7.
6. **Alpamayo 1.5.** Which governs: the OpenMDW-1.1 licence, or the model card's "ready
   for non-commercial use"? See section 6.
7. **Training data behind permissive weights.** Does a training set's licence reach the
   weights trained on it? Waymo's says so for its data, which brings Qwen-Drive and
   VisionPilot under decision 1. For the other models, does it matter only for future
   product use? See section 6.
8. **OpenPilot's indemnity.** Its README asks every user to indemnify comma.ai. Is that
   acceptable? See section 6.
9. **Videos of nuScenes scenes.** The website shows four neuroNCAP clips: renderings of
   nuScenes scenes, so nuScenes' terms apply to them. They are already public, in a team
   member's Hugging Face dataset (`naren-fastcode/paddock-videos`), shared under
   CC BY-NC-SA 4.0 with credit to nuScenes. May Renesas publish them, and should the
   dataset stay public until Legal decides? See section 4.

## 2. At a glance

### Models

| Model (publisher) | Code | Weights | Commercial use of the weights |
|---|---|---|---|
| Alpamayo 1.5 (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpamayo1.5/blob/7a8f1c781a826f09be53e1e211f26e947ec18019/LICENSE) | [OpenMDW-1.1](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | licence allows it; card says non-commercial |
| Alpamayo 2 Edge (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpamayo2-edge) | [OpenMDW-1.1](https://huggingface.co/nvidia/Alpamayo2-Edge) | licence allows it; the card version Paddock pinned said "to be confirmed" |
| AutoE2E (Autoware Foundation) | [Apache-2.0](https://github.com/autowarefoundation/auto_e2e/blob/40a75cf8a26b60f6b094125e3924fe0bdb427b02/LICENSE) | [CC BY-NC-SA 4.0](https://huggingface.co/AutowareFoundation/auto_e2e#license) | no |
| Drive-JEPA (Linhan Wang et al.) | [Apache-2.0](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/LICENSE) | [CC0-1.0](https://huggingface.co/datasets/LinhanWang/Drive-JEPA) | tag allows it; trained on non-commercial data |
| METEOR (TIER IV) | [Apache-2.0](https://github.com/tier4/METEOR/blob/dc193a81e9959de57b751e269127ae9b67e3f300/LICENSE) | [Apache-2.0](https://huggingface.co/AutowareFoundation/meteor) | yes |
| OpenPilot (comma.ai) | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE) | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE), in the code repository | yes; README adds an indemnity |
| Qwen-Drive 1.0 (Qwen team, Alibaba) | [Apache-2.0](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/LICENSE) | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) | licence allows it; trained partly on Waymo data |
| UniAD (OpenDriveLab) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/LICENSE) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/README.md#license), as UniAD's "assets" | licence allows it; trained on nuScenes, non-commercial data |
| VAD (HUST Vision Lab) | [Apache-2.0](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/LICENSE) | not stated; a Google Drive link in [VAD's README](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/README.md) | not stated; trained on nuScenes, non-commercial data |
| VisionPilot (Autoware Foundation) | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE) | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE), in the code repository | licence allows it; trained partly on Waymo-derived data |
| WA-JEPA (AFARI Research) | [Apache-2.0](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/LICENSE) | [Apache-2.0](https://huggingface.co/AFARI-Research/WA-JEPA) | tag allows it; trained on non-commercial data |
| Constant velocity (Renesas) | Apache-2.0 (Paddock) | none | not applicable |

### Benchmarks

| Benchmark (publisher) | Code | Data or scenes | Commercial use of the data |
|---|---|---|---|
| nuScenes (Motional) | Paddock's own | [CC BY-NC-SA 4.0 + Motional terms](https://www.nuscenes.org/terms-of-use) | no; [paid licence](https://www.nuscenes.org/terms-of-use-commercial) |
| nuPlan (Motional) | Paddock's own | [CC BY-NC-SA 4.0 + Motional terms](https://www.nuscenes.org/terms-of-use) | no; [paid licence](https://www.nuscenes.org/terms-of-use-commercial) |
| NAVSIM v1 and v2 (autonomousvision) | [Apache-2.0](https://github.com/autonomousvision/navsim) | OpenScene: [CC BY-NC-SA 4.0 + nuPlan terms](https://github.com/OpenDriveLab/OpenScene) | no |
| KITScenes (KIT) | [Apache-2.0](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/LICENSE) | [CC BY-NC 4.0 + KIT terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) | no |
| comma2k19 (comma.ai) | [MIT](https://github.com/commaai/comma2k19/blob/4c7f1a6e1957745beadc1def0e7225f559b09a2a/LICENSE) | [MIT](https://huggingface.co/datasets/commaai/comma2k19) | yes |
| Bench2Drive (Thinklab, SJTU) | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE), with MIT parts | CARLA: [MIT code, CC-BY assets](https://github.com/carla-simulator/carla/blob/0.9.15/README.md#licenses) | no, for the benchmark code |
| neuroNCAP (Ljungbergh, Tonderski et al.) | [MIT](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/LICENSE) | nuScenes, as above; renderer files: none stated | no |
| AlpaSim (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/LICENSE) | [NVIDIA AV NuRec agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | internal development only; results confidential |
| HUGSIM (X-D Lab, Zhejiang University) | [MIT](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/LICENSE) | [tagged MIT](https://huggingface.co/datasets/XDimLab/HUGSIM), built from nuScenes, KITTI-360, Waymo and PandaSet | mostly no |

## 3. Is credit enough, or is permission needed?

Every licence here requires credit. For permissive licences, credit is all that is
needed. For non-commercial licences, credit is required but not sufficient: the use itself
must also be non-commercial, and no amount of credit turns a commercial use into a
non-commercial one. Two items need permission however they are credited: the AlpaSim
results, and, on one reading, the Bench2Drive patches.

| Licence group (section 1.2) | Is credit enough? | When permission is needed |
|---|---|---|
| A. Permissive | Yes. Keep the licence and copyright notices. | Never |
| B. NVIDIA Open Model License | Yes, for running the model | Only to redistribute it, which we do not do |
| C. Non-commercial | Only if the use is non-commercial (decision 1) | If Legal judges the use commercial, or wants certainty |
| D. Non-commercial, no derivatives | For running Bench2Drive and publishing scores: as for C. For the two patches: possibly not (decision 4). | To share the patches, unless the file's MIT header covers them |
| E. NVIDIA NuRec agreement | No | Always, to publish AlpaSim results |
| F. sunnypilot | No | For any commercial use. We do not use it, so we remove it. |
| G. Unclear or conflicting | Probably, but not confirmed | A short confirmation from the publisher removes the doubt |

Three points behind this table:

- **A score is a fact, but two contracts reach it anyway.** Creative Commons says "CC
  licenses do not contractually impose restrictions on uses of a work where there is no
  underlying copyright" ([CC FAQ](https://creativecommons.org/faq/)), so a CC licence
  alone may not reach a published number. But Motional's terms ("Use of any data derived
  from the Datasets, which may appear in any format such as tables and charts, is also
  subject to these Dataset Terms") and NVIDIA's ("any results of benchmarking") are
  contracts accepted at download, and they reach results directly.
- **The scores came from using the data, and the licences do govern that use.** If the
  use was commercial, credit on the published numbers does not change that.
- **A company can make non-commercial use, but the purpose decides.** Creative Commons:
  "NonCommercial turns on the use, not the identity of the reuser"
  ([CC wiki](https://wiki.creativecommons.org/wiki/NonCommercial_interpretation)). Its
  licences define NonCommercial as "not primarily intended for or directed towards
  commercial advantage or monetary compensation". Against that, Motional's FAQ names
  "educational use and some research use" as free and prices commercial use, KITScenes is
  "intended for academic research", and Paddock's release also showcases Renesas's work.
  Weighing these is decision 1. Creative Commons adds that "licensees are always free to
  contact licensors to ask permission".

## 4. What the licences allow in the Paddock code release

| What the release does | Allowed? | Condition or action |
|---|---|---|
| Names each upstream project and links to the exact version; `paddock setup` downloads it on the user's machine | Yes, for all | Nothing is redistributed, and each user accepts the upstream terms. sunnypilot is removed (pull request #181, merged on 12 November). |
| Paddock's own code, which starts upstream code and calls it while running | Yes | It is Renesas's own work. No upstream code licence is copyleft (none requires code that uses it to take the same licence), so Paddock can stay Apache-2.0. |
| Patches to neuro-ncap (MIT), Drive-JEPA (Apache-2.0) and HUGSIM (MIT) | Yes | Keep the upstream notices and say what changed; each patch's header already explains its change. |
| Two patches to Bench2Drive | Unclear (decision 4) | They change lines that Bench2Drive wrote; those lines are not in [CARLA's original](https://github.com/carla-simulator/leaderboard/blob/leaderboard-2.0/leaderboard/leaderboard_evaluator.py). But the file is headed "Copyright (c) 2018-2019 Intel Corporation … This work is licensed under the terms of the MIT license." To make it a yes: offer both fixes to Bench2Drive, or get Thinklab's permission. Holding the patches back needs a small code change, because Paddock's Bench2Drive runner expects the timeout option that one of them adds. |
| Code adapted from AlpaSim and ported from VisionPilot (both Apache-2.0), and two reimplementations: TCP's PID controller (Apache-2.0) and openpilot's camera calibration (MIT) | Yes | Each file names its source, and THIRD_PARTY_NOTICES.md lists them (pull request #182, merged on 12 November). The PID controller's own description called it "vendored" from Bench2DriveZoo, which is CC BY-NC-ND 4.0; the code is a rewrite, and now says so. |
| nuScenes' validation scene list, and 20 neuroNCAP test scenarios with nuScenes object IDs and positions | Yes | Both copy permissively licensed upstream files. The scene list is exactly the one Motional publishes in its nuScenes devkit (Apache-2.0), and the scenarios are neuro-ncap's own (MIT) with only the time step changed. THIRD_PARTY_NOTICES.md credits both (pull request #182). |
| AlpaSim scene list (IDs and file paths) | Yes | The same IDs are published in AlpaSim's own Apache-2.0 repository. |
| Test data for UniAD and VAD, added on 12 November: a verbatim subset of nuScenes' tables for scenes 0103 and 0916 (sample records, calibration, ego poses and CAN-bus messages; no images and no annotations), and files holding both models' inputs and plans on those scenes | Yes, for non-commercial use, with credit, under CC BY-NC-SA 4.0 | THIRD_PARTY_NOTICES.md lists the scene-0103 data, but neither the table excerpt (`uniad/tests/fixtures/nuscenes_scene_tables.json.gz`) nor scene 0916; add both. If publishing nuScenes excerpts is not allowed, replace them with made-up data, as for AlpaSim. |
| Code repeated verbatim from UniAD and VAD in their tests (`uniad/tests/infos_reference.py`) | Yes, with notice | Apache-2.0. Not yet in THIRD_PARTY_NOTICES.md; add it. |
| The list of the 223 KITScenes keyframes Paddock scores (`kitscenes/sample_list.csv`, added on 12 November): scene id, keyframe index, split, and a driving command Paddock derives from each recorded drive | Yes, for non-commercial use, with credit | Not yet in THIRD_PARTY_NOTICES.md; add it, with KITScenes' licence. |
| One patch to UniAD (Apache-2.0), and two to OpenMMLab code that Bench2DriveZoo carries | Yes | Each Bench2DriveZoo patch changes one line of OpenMMLab's own code, which is Apache-2.0; both lines are in mmdet3d 0.17.1 and mmcv 1.4.0 as OpenMMLab published them, so neither is Bench2DriveZoo's own. THIRD_PARTY_NOTICES.md lists both. |
| Bench2DriveZoo (CC BY-NC-ND 4.0), which `paddock setup` downloads to build UniAD's and VAD's environment | Yes, if running it is non-commercial use | Nothing of it is in the repository. The build copies two of its files, which per UniAD's page are OpenMMLab's own mmdet3d 0.17.1 files (Apache-2.0). |
| Test files cut from two real AlpaSim runs: scores, and one model's outputs on one scene | No | Replaced with made-up data in pull request #183, including which scene passes, and the model's outputs by a made-up drive. That pull request is not merged yet, so `main` still holds the real files. |
| Two route names from Bench2Drive's route list: each route's id and town, and its scenario's name and type, 3% of each entry, with no waypoints or weather | Yes, for non-commercial use, with credit and a licence notice | Kept, because Paddock's results use the same names throughout. The test's README and THIRD_PARTY_NOTICES.md now say what the file holds and Bench2Drive's licence (pull requests #182 and #183). |
| Results: scores, tables and charts | See section 1.1 | |
| A website built from the results on `main` (GitHub Pages), added after this review began | See section 1.1 | It is built only once the repository is public, and shows the results `main` holds, so the AlpaSim decision applies to it too. It would show eight videos: four neuroNCAP clips, renderings of nuScenes scenes, and, since 12 November, four AlpaSim clips, renderings of NVIDIA's scenes, whose captions give each scene's scores. They are hosted on a team member's Hugging Face dataset, `naren-fastcode/paddock-videos`, which was public on 11 November and is not publicly reachable on 16 November. The page also offers every result as a `data.json` download. The repository's own note says whether Renesas may show the clips "is still to be confirmed with Renesas legal" (decision 9). |
| `paddock data fetch`, which downloads the datasets and CARLA, added after this review began | Yes | It downloads from each owner's own host, names each dataset's licence, and refuses a dataset that has terms until the user passes `--accept-terms`. HUGSIM's entry gives its licence as MIT; whether to add its source datasets' terms waits on decision 5. |
| Names of models, datasets and publishers in documents and charts | Yes, to describe them | Do not imply endorsement. Motional's terms forbid it, and Paddock's README already says that being listed does not mean a project endorses Paddock. |

**Clean-ups before release.** Four are merged (pull requests #181 and #182, on 12 November),
one is in pull request #183, which is not merged yet, and two wait on Legal:

1. **AlpaSim.** The test files cut from AlpaSim runs are replaced with made-up data in
   pull request #183, not merged yet. Removing the AlpaSim results, page and videos
   waits on decision 3.
2. **Bench2Drive patches.** Offer the two fixes upstream, or ask Thinklab for
   permission; otherwise hold the patches back and make the timeout option optional.
   Waits on decision 4.
3. **Bench2Drive route names.** Kept, with a licence note (pull request #182, merged, and #183, not):
   the file holds only names, which Paddock's results use everywhere.
4. **THIRD_PARTY_NOTICES.md**, listing every file that holds other projects' material,
   with the licences of the datasets behind the results (pull request #182, merged).
5. **The VisionPilot port** names its source and licence (pull request #182, merged).
6. **sunnypilot** is removed from the list of upstream repositories (pull request #181, merged).
7. **Paddock's model comparison page** gives each model's weights licence as its
   publisher states it (pull request #182, merged).

## 5. How others report results on these benchmarks

- **Companies publish scores on these non-commercial datasets, with credit.** NVIDIA
  Research won the CVPR 2024 NAVSIM challenge, which runs on nuPlan data, and announced it
  publicly ([NVIDIA blog](https://blogs.nvidia.com/blog/auto-research-cvpr-2024/)).
  Alibaba's Qwen team publishes Qwen-Drive's scores on NAVSIM and on Waymo's WOD-E2E
  ([model card](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B)).
- **Publishing other parties' models' scores is normal.** Qwen-Drive's card lists other
  publishers' models beside its own, Alpamayo 1.5 among them. neuroNCAP's authors, whose
  project page is on [Zenseact's research site](https://research.zenseact.com/publications/neuro-ncap/),
  publish the score of UniAD, a third-party model, on scenes from nuScenes, and invite
  others to add theirs ([README](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/README.md)).
  Bench2Drive publishes the scores of third-party models such as UniAD and VAD
  ([README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md)).
- **Dataset publishers invite it.** Motional runs public nuScenes challenges
  ([FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md)). The NAVSIM
  paper reports: "NAVSIM enabled a new competition held at CVPR 2024, where 143 teams
  submitted 463 entries" ([paper](https://arxiv.org/abs/2406.15349)). Waymo's licence
  counts benchmarking for research publication as non-commercial use
  ([terms](https://waymo.com/open/terms/)).
- **What projects hold back is weights trained on restricted data.** OpenMMLab's
  MMDetection3D, an open-source 3D detection toolkit, records: "Delete checkpoints on
  Waymo to comply its specific license agreement"
  ([changelog](https://mmdetection3d.readthedocs.io/en/latest/notes/changelog_v1.0.x.html)).
  Its Waymo code and instructions stayed. Paddock already hosts no weights.
- **Where a licence forbids publishing results, permission comes first.** Licence terms
  that forbid publishing benchmark results without the vendor's approval are known as
  "DeWitt clauses", after a database benchmark published in the early 1980s. They have
  long been common in commercial database licences, and some vendors have dropped them,
  Databricks in 2021 ([essay](https://dwheeler.com/essays/dewitt-clause.html),
  [Databricks](https://www.databricks.com/blog/2021/11/08/eliminating-the-dewitt-clause-for-database-benchmarking.html)).
  NVIDIA's NuRec agreement contains such a clause.
- **Practice does not settle whether a company's publication is non-commercial.** The
  dataset terms here do not define it beyond Motional's FAQ and KITScenes' statement of
  intended use. The firm route is the publisher's written confirmation.

## 6. Models: details

Each LICENSE link points to the exact version of the code that Paddock uses.

#### Alpamayo 1.5 (NVIDIA)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpamayo1.5/blob/7a8f1c781a826f09be53e1e211f26e947ec18019/LICENSE) | |
| Weights: [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | Grants rights "without restriction", and "does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs". Also: "YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF". |
| Model card | | "This model is ready for non-commercial use. Commercial licensing available upon request." (decision 6) |
| Built on | [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B): [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). That model is built on [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): Apache-2.0. | Commercial use allowed. Use must follow NVIDIA's Trustworthy AI terms. Attribution duties apply only to sharing Cosmos models or building new ones from them. Paddock loads only Cosmos-Reason2-8B's configuration files. |
| Trained on | NVIDIA's own driving data, plus public sets built on nuScenes and NAVSIM | Decision 7 |
| In the release | Paddock's adapter, including one file adapted from AlpaSim that keeps NVIDIA's notice. Older neuroNCAP runs, imported from before Paddock existed, are stored under the name `alpamayo` and do not record which checkpoint ran; NVIDIA's earlier [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) has the same OpenMDW-1.1 tag. | |

#### Alpamayo 2 Edge (NVIDIA)

Added to Paddock after this review began.

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpamayo2-edge) | |
| Weights: [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge), not gated | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | The current model card states OpenMDW-1.1 and, unlike Alpamayo 1.5's, does not call the weights non-commercial. The earlier card version Paddock pins said the licence was still to be confirmed. |
| In the release | Paddock's adapter; results on eight benchmarks: NAVSIM v1 and v2, nuScenes, KITScenes, comma2k19, neuroNCAP, Bench2Drive and AlpaSim | |

#### AutoE2E (Autoware Foundation)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/autowarefoundation/auto_e2e/blob/40a75cf8a26b60f6b094125e3924fe0bdb427b02/LICENSE) | |
| Weights: [AutowareFoundation/auto_e2e](https://huggingface.co/AutowareFoundation/auto_e2e#license), checkpoint `kitscenes-epoch-5` | CC BY-NC-SA 4.0 | "commercial use is not covered. It needs permission from the upstream rights holders, including a commercial license from Motional for nuPlan and nuScenes". The card overrides the licence "other" recorded in the files of the v1.0 release. |
| Trained on | nuPlan, then fine-tuned on KITScenes, with a camera encoder trained on nuScenes | The reason the card chose CC BY-NC-SA 4.0 |
| In the release | Paddock's adapter; scores only | |

#### Drive-JEPA (Linhan Wang et al.)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/LICENSE) | Includes two copies of NAVSIM's scoring code (Apache-2.0), which Paddock uses to run NAVSIM, and Meta's V-JEPA 2 code (MIT). |
| Weights: [LinhanWang/Drive-JEPA](https://huggingface.co/datasets/LinhanWang/Drive-JEPA) | CC0-1.0, tag only | A public-domain dedication, with no licence text. The same repository holds scoring caches built from nuPlan logs. |
| Trained on | NAVSIM and OpenScene, both from nuPlan | Decision 7 |
| In the release | Paddock's adapter; three patches (Apache-2.0 code) | The repository also carries a [copy of Motional's terms](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/download/LICENSE), which this document quotes. |

#### METEOR (TIER IV)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/tier4/METEOR/blob/dc193a81e9959de57b751e269127ae9b67e3f300/LICENSE) | |
| Weights: [AutowareFoundation/meteor](https://huggingface.co/AutowareFoundation/meteor) | Apache-2.0 | |
| Trained on | TIER IV's own recordings in Japan, plus re-renderings made with NVIDIA Cosmos Transfer | |
| In the release | Paddock's adapter | |

#### OpenPilot (comma.ai)

| Part | Licence | Terms that matter |
|---|---|---|
| Code and weights | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE) | The weights are files inside the code repository. |
| [README](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/README.md) | | "Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys' fees and costs) which arise out of, relate to or result from any use of this software by user." Also: "THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT." (decision 8) |
| Trained on | comma.ai's own fleet data, which is not public | |
| In the release | Paddock's adapter | |

#### Qwen-Drive 1.0 (Qwen team, Alibaba)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/LICENSE) | "Copyright 2026 Alibaba Group Holding Limited" |
| Weights: [Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) | Apache-2.0 | Built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0 |
| Trained on | nuScenes, OpenScene, Waymo's WOD-E2E, NVIDIA's PhysicalAI-AV, and other public driving sets | Decision 7. Waymo's licence names models trained on its data. |
| In the release | Paddock's adapter | |

#### UniAD (OpenDriveLab)

Added to Paddock on 12 November. No results yet.

| Part | Licence | Terms that matter |
|---|---|---|
| Code: [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/LICENSE) | Its README: "All assets and code are under the Apache 2.0 license unless specified otherwise." |
| Weights: `uniad_base_e2e.pth`, from UniAD's own GitHub release v1.0.1 | Apache-2.0, as UniAD's "assets" | Trained on nuScenes `train`, non-commercial data; its first stage starts from BEVFormer's nuScenes weights. |
| Build dependency: [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db) | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db/LICENSE) | `paddock setup` downloads it to build the environment UniAD and VAD share; the build copies two of its files, OpenMMLab's own. |
| In the release | Paddock's adapter; one patch to UniAD and two to Bench2DriveZoo; test data from two nuScenes scenes (section 4) | |

#### VAD (HUST Vision Lab)

Added to Paddock on 12 November. No results yet.

| Part | Licence | Terms that matter |
|---|---|---|
| Code: [hustvl/VAD](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8) | [Apache-2.0](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/LICENSE) | Its README: "All code in this repository is under the Apache License 2.0." It says nothing about the weights. |
| Weights: VAD-Base, from a Google Drive link in [VAD's README](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/README.md) | Not stated | Trained on nuScenes `train`, non-commercial data. |
| In the release | Paddock's adapter, which shares UniAD's environment and test data | |

#### VisionPilot (Autoware Foundation)

| Part | Licence | Terms that matter |
|---|---|---|
| Code and weights | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE) | The weights are files inside the code repository, whose README offers "the complete codebase as free and fully open-source, including AI model weights". |
| Trained on | Partly [OpenLane](https://github.com/OpenDriveLab/OpenLane): CC BY-NC-SA 4.0 plus Waymo's non-commercial licence | Decision 7 |
| In the release | Paddock's adapter, which ports part of VisionPilot's C++ planner (clean-up 5 in section 4). | |

#### WA-JEPA (AFARI Research)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/LICENSE) | Includes Meta's V-JEPA 2 code (MIT, with an Apache-2.0 part). |
| Weights: [AFARI-Research/WA-JEPA](https://huggingface.co/AFARI-Research/WA-JEPA) | Apache-2.0, tag only | The model card is empty. |
| Trained on | nuPlan video, then NAVSIM, according to Paddock's own notes | Decision 7 |
| In the release | Paddock's adapter | |

#### Constant velocity (Renesas)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | Apache-2.0 (Paddock) | Paddock's own baseline: a few lines of arithmetic, with no weights and no third-party code. |

## 7. Benchmarks and their data: details

#### nuScenes (Motional): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Data: [nuScenes](https://www.nuscenes.org/nuscenes) | [CC BY-NC-SA 4.0 + Motional's Dataset Terms](https://www.nuscenes.org/terms-of-use) | "available for non-commercial use subject to the Terms of Use below" |
| Results derived from the data | Dataset Terms | "Use of any data derived from the Datasets, which may appear in any format such as tables and charts, is also subject to these Dataset Terms." |
| Precedence | Dataset Terms | "Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail." |
| Endorsement and names | Dataset Terms | No implying that Motional endorses the use, and no use of its name, trademark or logo, without its written consent. |
| Other clauses | Dataset Terms | The user indemnifies Motional. Massachusetts law. Motional may amend the terms at any time. |
| Commercial use | [Commercial licence](https://www.nuscenes.org/terms-of-use-commercial); [FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md) | Free "for educational use and some research use"; for commercial use, "we adjust the pricing terms to the use case and company size". |
| Code | Paddock's own | Follows a published scoring method. |
| In the release | Results; the validation scene list as Motional's nuScenes devkit publishes it (Apache-2.0); a verbatim subset of nuScenes' tables for two scenes, with UniAD's and VAD's inputs and plans on them, as test data | Section 4. |

#### nuPlan (Motional): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Data: [nuPlan](https://www.nuscenes.org/nuplan) | Same as nuScenes | One document, titled "nuScenes, nuPlan, nuImages, and nuReality Terms of Use", covers both. |
| In the release | 5 result runs | Kept but not ranked, because they used part of the data. |

#### NAVSIM v1 and v2 (autonomousvision): recorded drives; the model's plan is scored in a short simulation

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: [autonomousvision/navsim](https://github.com/autonomousvision/navsim) | Apache-2.0 | "The datasets (including nuPlan and OpenScene) inherit their own distribution licenses." Paddock runs the copies inside Drive-JEPA. |
| Data: [OpenScene](https://github.com/OpenDriveLab/OpenScene) (OpenDriveLab), a smaller re-release of nuPlan | CC BY-NC-SA 4.0 + nuPlan's terms | Users "may not use the work for commercial purposes". [Copy on Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene). |
| In the release | Results | |

#### KITScenes (Karlsruhe Institute of Technology): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: the library that reads the data | [Apache-2.0](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/LICENSE), [NOTICE](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/NOTICE) | Bundles Lanelet2 (BSD-3-Clause) and CC0 icons. "Using this Python API does not grant rights to use the dataset beyond those terms." |
| Data: [KIT-MRT/KITScenes-Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal), access on request | CC BY-NC 4.0 + KIT's terms ([summary](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/README.md#license)) | KIT's terms prevail where they conflict. Research papers must cite the KITScenes paper, and media use must link to kitscenes.com. "We reserve all rights that are not explicitly granted to you." |
| Intended use | Dataset card | "intended for academic research". Also: "For final benchmark reporting, please wait for a more stable public release." |
| Term added on 13 November 2026 | KIT's terms | Aerial feature tiles: users "must not reconstruct, recover or approximate the source imagery from them", nor publish such reconstructions. Paddock does not use the aerial tiles. |
| In the release | Results; the list of the 223 keyframes Paddock scores | Section 4. |

#### comma2k19 (comma.ai): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: the toolkit | [MIT](https://github.com/commaai/comma2k19/blob/4c7f1a6e1957745beadc1def0e7225f559b09a2a/LICENSE) | |
| Data: [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19) | MIT (tag) | Also distributed through Academic Torrents, as the toolkit's README says. |
| In the release | Results | |

#### Bench2Drive (Thinklab, Shanghai Jiao Tong University): CARLA simulator

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code and routes | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE) | "All assets and code are under the CC-BY-NC-ND unless specified otherwise" ([README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md#license)). The licence lets you "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only", and "produce and reproduce, but not Share, Adapted Material". |
| Folders `leaderboard/` and `scenario_runner/` | [MIT](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/LICENSE), copyright CARLA | The [file the patches edit](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/leaderboard/leaderboard_evaluator.py) is headed with Intel's copyright and the MIT licence. The lines the patches change were added by Bench2Drive; they are not in [CARLA's original](https://github.com/carla-simulator/leaderboard/blob/leaderboard-2.0/leaderboard/leaderboard_evaluator.py). (decision 4) |
| Simulator: CARLA 0.9.15, which each user installs | [MIT code, CC-BY assets](https://github.com/carla-simulator/carla/blob/0.9.15/README.md#licenses) | "CARLA specific code is distributed under MIT License. CARLA specific assets are distributed under CC-BY License." CARLA is built on Unreal Engine 4, which "follows its own license terms" ([Unreal Engine FAQ](https://www.unrealengine.com/en-US/faq)). |
| In the release | Results; two patches; the names of two routes; outputs of a few of Paddock's own runs, used as test data | Section 4. Bench2Drive's training dataset (Apache-2.0, on Hugging Face) is not used. |

#### neuroNCAP (William Ljungbergh, Adam Tonderski et al.): nuScenes scenes re-rendered so that a model can drive in them

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [MIT](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/LICENSE) | |
| Renderer: [neurad-studio](https://github.com/georghess/neurad-studio) | Apache-2.0 | |
| Renderer files, one per scene | None stated | Downloaded from a Dropbox link in neuro-ncap's [download script](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/scripts/download/download_neurad_weights.sh). They are reconstructions of nuScenes scenes. |
| Data | nuScenes, as above | |
| In the release | Results; three patches; 20 test scenarios, neuro-ncap's own (MIT) with the time step changed, holding nuScenes object IDs and positions; four website videos of its scenes | Section 4. |

#### AlpaSim (NVIDIA): reconstructed real drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/LICENSE) | "Copyright 2025 NVIDIA Corporation" |
| Scenes: [nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec), release 26.01 | NVIDIA Autonomous Vehicles NuRec Dataset License Agreement, accepted by click-through ([PDF](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/blob/main/LICENSE.pdf), visible after accepting) | The text is identical in release 26.01 and in the current version. |
| §1, purpose | Agreement | "solely for your internal development of autonomous vehicles and automated driving assisted systems using NVIDIA technology" |
| §3, confidentiality | Agreement | "'Confidential Information' means the Dataset including its features and functionality, output, and any results of benchmarking or other competitive analysis or regression or performance data relating to the Dataset." The user "will not disclose NVIDIA Confidential Information to any third party", except to authorised users bound by a written confidentiality agreement. |
| §4.4 and §4.6, limits | Agreement | No identifying people, including through licence plates. No derivative works, and no distributing or hosting the data. |
| §8, term | Agreement | "This Agreement expires twelve (12) months after the date of initial delivery or download of the Dataset." On termination the user "must stop using and destroy all copies of the Dataset". |
| §11, law | Agreement | United States and Delaware law; the courts of Santa Clara County, California. |
| In the release | 15 result runs over nine models, Alpamayo 2 Edge's added on 16 November; an AlpaSim column in the README's results chart; an AlpaSim page; test files cut from two real runs, until pull request #183 merges; four website videos of its scenes, whose captions give per-scene scores; the scene list, which comes from AlpaSim's own repository | Section 4 |

#### HUGSIM (X-D Lab, Zhejiang University): reconstructed real drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [MIT](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/LICENSE) | |
| Scenes: [XDimLab/HUGSIM](https://huggingface.co/datasets/XDimLab/HUGSIM) | MIT, tag only, no licence text | Built from the four sources below (decision 5). |
| Source: [nuScenes](https://www.nuscenes.org/terms-of-use) | CC BY-NC-SA 4.0 + Motional's terms | As above |
| Source: [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) | CC BY-NC-SA 3.0 | "you may not use this work for commercial purposes" |
| Source: [Waymo Open Dataset](https://waymo.com/open/terms/) | "Waymo Dataset License Agreement for Non-Commercial Use (March 2025)" | Non-commercial use "means research, teaching, scientific publication and personal experimentation", and expressly includes benchmarking for research publication. Work made with the data must carry an attribution. §4(a) begins "You must not use or deploy the Dataset, any models trained on or refined using the Dataset, or the weights or biases", and bars their use in vehicles, in production systems, or for other primarily commercial purposes. Commercial licences: open-dataset@waymo.com. |
| Source: PandaSet (Scale AI and Hesai) | "a no-cost commercial license", according to [its paper](https://arxiv.org/abs/2112.12610) | Its website, pandaset.org, no longer shows PandaSet: on 10 November 2026 it redirected to an unrelated site, so the licence text itself could not be read. |
| Vehicle models: [3DRealCar](https://github.com/xiaobiaodu/3DRealCar_Toolkit) | Apache-2.0 | "3DRealCar dataset is allowed commercial usage for free." |
| In the release | Results for two models; two patches to MIT code, one of them HUGSIM's own proposed fix ([pull request #57](https://github.com/hyzhou404/HUGSIM/pull/57)) | |

## 8. Listed in Paddock but not used

| Project | Licence | Note |
|---|---|---|
| sunnypilot | Custom: [LICENSE.md](https://github.com/sunnypilot/sunnypilot/blob/0b1ed0d047aca5bed45f20323823c5d5e9cb6bc1/LICENSE.md) | Use "for any commercial purposes, for-profit projects, or in closed source projects requires explicit written permission from the original author(s)", and any project using it must show an acknowledgement. Removed from the list in pull request #181, merged on 12 November. |
| Alpamayo 2 (NVIDIA) | Code: [Apache-2.0](https://github.com/NVlabs/alpamayo2/blob/beb2977d9a7e9d66837d4a3ad5144ff59de37519/LICENSE). Weights: [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super), OpenMDW-1.1. | Not tested yet |
| V-JEPA 2 (Meta) | [MIT](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE), with three files under Apache-2.0 | Listed for reference. Paddock uses only the copies inside Drive-JEPA and WA-JEPA. |

## 9. How this was checked

- **Code licences** were read from each upstream repository's licence files at the
  version Paddock uses, together with the licence section of its README. GitHub's own
  licence detection agrees wherever it recognises the licence.
- **Bench2Drive's patched file** was compared with CARLA's original evaluator, on both of
  CARLA's branches, to find which lines Bench2Drive added.
- **Weights and datasets on Hugging Face**: licence tags were read from Hugging Face's
  public interface, and model and dataset cards were read in full where they are public.
- **AlpaSim's agreement** was read from the access-page text that Hugging Face publishes
  for release 26.01 and for the current version. The two are identical. The PDF opens only
  after accepting, and was not read.
- **KITScenes' terms** were read from the dataset card downloaded with the data. They
  match the live page.
- **Motional's terms**: the nuScenes website needs a browser to display them, so this
  document quotes the copy in Drive-JEPA's repository, "last updated on Nov 16, 2021".
  Please confirm against the live page. Motional's FAQ was read in its nuScenes devkit.
- **Other public pages** (OpenScene, NAVSIM, CARLA, KITTI-360, Waymo, 3DRealCar,
  OpenLane, the NVIDIA Open Model License, Creative Commons' FAQ and wiki, and the
  sources in section 5) were read on 10 November 2026.
- **Links**: every link in this document opened on 10 November 2026 except two. The
  AlpaSim PDF opens only after accepting the agreement, and the Unreal Engine site
  refuses automated access; that link is the one CARLA's own README gives.
- **The clean-ups** were made on 10 November 2026 in pull requests #181 to #183. While making
  them, the nuScenes scene list, the 20 neuroNCAP scenarios, the AlpaSim scene list and the
  Bench2Drive route file were each compared with their upstream source; section 4 gives
  what each turned out to be.
- **Re-checked on 16 November 2026 against `main` at `73e9ac6`.** Every licence tag and
  model card above is unchanged, and NVIDIA's AlpaSim agreement is word for word the same.
  KITScenes' terms gained one term, on aerial features, which Paddock does not use.
- **Not checked**: PandaSet's licence text; WA-JEPA's training data beyond Paddock's
  notes; and the licences of the software libraries each model installs, such as
  PyTorch, which Paddock does not ship.
