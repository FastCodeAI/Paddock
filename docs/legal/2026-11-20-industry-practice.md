# Paddock release: how other companies publish results on non-commercial datasets

*Prepared by the model development team for Renesas Legal, and for the business decision
that follows Legal's risk analysis. 20 November 2026. It answers an action from the
meeting of 17 November: "Check whether other companies, such as NXP and Infineon, publish
benchmark results on non-commercial datasets." It extends section 7 of the licence review
(`paddock-third-party-licences.md`). A fact sheet, not legal advice.*

**How to read it.** Every finding links to its source and quotes a few words from it.
The full list of sources is at the end, under References. Each finding carries one of
three labels:

- **Verified**: we opened the source on 20 November 2026 and quote it.
- **Reported**: a source we could not open ourselves, or saw only in a search, says so.
- **Our reading**: our inference, for Legal to weigh.

Terms used below:

- **Non-commercial dataset**: a dataset whose licence allows only non-commercial use, such
  as nuScenes and nuPlan (Motional's: CC BY-NC-SA 4.0 plus Motional's own terms).
- **Challenge**: a contest on a dataset's hidden test data, run by the dataset's owner or
  a research workshop, with a public ranking (a **leaderboard**) and often prize money.
- **Model library**: a public collection of trained models that a company publishes, with
  each model's accuracy, and often its speed on the company's own chips.
- **MLPerf**: the industry's standard benchmarks for AI computers, run by MLCommons, an
  industry consortium. **AVCC**: the Autonomous Vehicle Computing Consortium, a
  non-profit group of carmakers, suppliers and chip makers, which Renesas joined in 2019.

## Summary

- **Yes, commercial companies publish scores on these datasets, chip makers among them.**
  We found 14 research papers by company researchers (NVIDIA, Horizon Robotics,
  Qualcomm, Infineon, NXP, Xiaomi, Bosch, BMW, Wayve and others); model libraries from
  four chip makers with nuScenes accuracy (NVIDIA, AMD, Texas Instruments, Hailo), three
  of them beside their own chips' speed; and company teams on the owners' leaderboards.
- **NXP and Infineon publish research, not marketing.** Their researchers co-write
  nuScenes papers with TU Delft and TU Munich. We found neither using such scores in a
  model library or in product material.
- **Most of it is research done with universities.** 12 of the 14 papers have university
  co-authors; the model libraries and marketing pages, the commercial-looking channels,
  have none. The licences, though, test the purpose of the use, not who the user is
  (section 2.1).
- **Almost nobody says it has permission. The one case that did is the closest to
  Renesas.** MLPerf Automotive, an industry benchmark that chip makers use to sell to
  carmakers, called the use of non-commercial data a "legal ambiguity" and asked Motional
  first. Motional agreed, and supplied nuScenes; the agreement "required lawyers and
  time", over "months".
- **Motional's own terms are stricter than its FAQ.** Its commercial-terms page asks for
  a licence for any use "with the expectation of generating revenue either at present or
  in the future, such as industrial research and development", "regardless of whether
  you work in academia or industry".
- **Owners welcome companies inside their own challenges.** Motional's nuScenes
  leaderboard ranks SenseTime 2nd and Hesai 7th; NVIDIA won NAVSIM awards in 2024 and
  2025, the second on a test set Motional provided. Those rights come with each
  challenge's terms.
- **We found no dispute over published scores.** None of the 22,457 notices in GitHub's
  public archive of copyright takedowns concerns these datasets, and projects that ran
  into Waymo's licence withdrew trained weights but kept their scores. The one live suit
  by a dataset owner against a chip maker, Jamendo v. NVIDIA (2026), is about building
  products from non-commercial data.
- **Our reading:** Legal's rating of the risk as very low fits the evidence. But what
  companies do shows tolerance, not permission; and the one industry body that weighed
  the question for commercial use asked first.

## What to focus on

1. **Ask Motional, as MLPerf did** (section 3). It is the only route that turns
   "tolerated" into "allowed". Motional has said yes to an industry benchmark before, and
   one letter covers five of Paddock's eight non-commercial benchmarks: nuScenes, nuPlan,
   NAVSIM v1 and v2, and neuroNCAP. Allow months, as MLPerf did.
2. **Read Motional's commercial-terms sentence beside its "tables and charts" clause**
   (section 4.1). Together they are the strongest text for Legal's conservative reading;
   the FAQ's "some research use" is the strongest against it.
3. **If Renesas publishes without asking, publish the way company researchers do**
   (sections 2 and 7.3): as research, with no Renesas product names or chip speeds beside
   scores on non-commercial data, each result marked with its data's licence (Q4), the
   disclaimer of Q14, and a way to take results down on request. Chip makers' libraries
   that pair nuScenes accuracy with their chips' speed are common, but are the
   commercial end of practice; Paddock does not need that pattern.
4. **A university partner helps only if it is real** (section 2.1): a university that
   runs and publishes the evaluation under its own registration changes the facts; one
   used as a channel does not, as Legal said on 17 November.
5. **Treat Waymo-based results differently** (section 4.2). Waymo's licence counts
   "benchmarking for purposes of academic or applied research publication" as
   non-commercial; this covers HUGSIM's Waymo-derived scenes.
6. **Keep AlpaSim out** (section 4.7). NVIDIA lets AlpaSim scores be published only
   inside its own 2026 challenge, under that challenge's terms.
7. **Get every credit line right** (section 6.4). People who sue over Creative Commons
   licences for money usually claim a missing credit. Under version 3.0 licences, such as
   that of KITTI-360, one of HUGSIM's sources, a breach ends the licence with no time to
   fix it. This makes Q11 (credit lines) worth doing before release whatever is decided.

## 1. Chip makers: NXP, Infineon and others

**The two companies the meeting named publish research on nuScenes, not marketing.**
Infineon's and NXP's researchers co-write papers with universities that report results on
nuScenes. We found neither company using such results in a model library, on a product
page or in a press release. NXP's one product use is a 2019 presentation that shows an NXP
chip running on KITTI data, with no score.

**Other chip makers do publish such scores**, mostly in model libraries, and there on
their own, without a university. Their marketing names rankings, not numbers. None of
them states a licence or permission for the data.

| Company | Where it publishes scores | With a university? | Licence or permission stated |
|---|---|---|---|
| Infineon | Research papers | Yes: TU Munich | None |
| NXP | A research paper; 2019 product slides, on KITTI data, with no score | Paper: yes, TU Delft. Slides: no | None |
| NVIDIA | A model library, beside its own chip's speed; its blog; challenges; papers | Library and blog: no. Papers: yes | None. A nuScenes-trained model's card says "ready for commercial/non-commercial use". |
| Horizon Robotics | Its corporate technology page, as rankings; open code; papers; challenges | Page and code: no. Papers: some | None on the page. A model it co-developed says "Non-commercial license". |
| AMD (Xilinx) | A model library, beside its own board's speed | No | Apache-2.0 for the library; nothing on nuScenes |
| Texas Instruments | A model library | No | "not meant for deployment in production" |
| Hailo | A model library, beside its own chip's speed | No | MIT for the library; nothing on nuScenes |
| Qualcomm | Papers; a challenge; its chip's speed, without naming a dataset | No | None |
| Ambarella, Mobileye and Intel, Black Sesame, Axera, SiMa.ai | Nothing found, in one or two searches each: weak negatives | | |

**The evidence, all Verified:**

- **Infineon.** Authors at "Infineon Technologies AG" and TU Munich. A radar-camera
  detector that "outperforms current state-of-the-art radar-camera fusion-based object
  detectors in the nuScenes dataset" ([MCAF-Net](https://arxiv.org/abs/2307.08339), July
  2023), and a depth model "benchmarked on the nuScenes dataset"
  ([TRIDE](https://arxiv.org/abs/2508.08038), August 2025). Neither paper mentions the
  data's licence; we searched their full text.
- **NXP.** Two authors at "NXP Semiconductors", with TU Delft, report better results
  "through testing on scenes from the nuScenes dataset" (IEEE SENSORS 2023,
  [record](https://repository.tudelft.nl/record/uuid:25d7b30d-2303-4a04-9e5b-af7a2ce8414e)).
  A 2019 product presentation, marked "COMPANY PUBLIC", runs a lidar network on NXP's S32V
  chip: "Input image is from Velodyne HDL-64E LiDAR as part of KITTI dataset"
  ([slides](https://www.nxp.com/docs/en/training-reference-material/ADAS-&-DRIVER-REPLACEMENT-VISION-OVERVIEW-FOR-ADAS-SYSTEMS.pdf),
  slide 15). The slides carry only "© 2019 NXP B.V.".
- **NVIDIA.** Its CUDA-BEVFusion library reports, on nuScenes, "67.66 | 70.81 | 25(on
  ORIN)": accuracy (mAP and NDS) and frames per second on NVIDIA's Orin chip
  ([library](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)).
  Its BEVFormer card: "nuScenes dataset was used for training, testing and evaluation";
  "This model is ready for commercial/non-commercial use"; "This model is not owned or
  developed by NVIDIA" ([card](https://docs.api.nvidia.com/nim/reference/nvidia-bevformer)).
  Its blog: "NVIDIA Research topped the leaderboard this year in the End-to-End Driving at
  Scale category", in a post that also announces a product, Omniverse Cloud Sensor RTX
  ([17 June 2024](https://blogs.nvidia.com/blog/auto-research-cvpr-2024/)); and "the
  second consecutive year that NVIDIA's topped the leaderboard"
  ([11 June 2025](https://blogs.nvidia.com/blog/auto-research-cvpr-2025/)).
- **Horizon Robotics.** Its corporate technology page: "Sparse4D—Horizon's end-to-end
  perception algorithm—ranks No.1 on the nuScenes public dataset", and "Horizon Robotics
  has won five world titles in the Google Waymo Open Dataset Challenge"
  ([technology page](https://www.horizon.auto/en/technology)). DiffusionDrive, developed
  with HUST, reports "88.1 PDMS on NAVSIM" and says "License: Non-commercial license",
  though its tag says Apache-2.0 ([card](https://huggingface.co/hustvl/DiffusionDrive)).
- **AMD.** Its Vitis AI 1.4 model library: PointPillars on nuScenes, "mAP: 42.2 NDS:
  55.1", and "mAP: 40.5 NDS: 53.0" once adapted to its chips; 2.3 frames per second on
  its ZCU102 board. The library is "licensed under Apache License Version 2.0"
  ([library](https://github.com/Xilinx/Vitis-AI/tree/v1.4/models/AI-Model-Zoo)).
- **Texas Instruments.** Sparse4Dv3 on nuScenes: mAP 44.66, NDS 55.94
  ([edgeai-mmdetection3d](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d)).
  Its model library says the models are "made available for experimentation and
  development - they are not meant for deployment in production"
  ([edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo)).
- **Hailo.** Under "Nuscenes 2019": PETRv2 at 25.9, 23.3 on its chip, and 26.4 frames
  per second on Hailo-8. "The Hailo Model Zoo is released under the MIT license"
  ([library](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection_3d.rst)).
- **Qualcomm.** "X-Align achieves a new record mIoU of 65.7% on nuScenes BEV
  segmentation", from Qualcomm AI Research alone
  ([X-Align](https://arxiv.org/abs/2210.06778)). Second in HUGSIM's 2025 rendering
  challenge (section 4.6). Its BEVFormer page gives the model's speed on its SA8295P car
  chip, "1841.885 ms", and names no dataset ([card](https://huggingface.co/qualcomm/BEVFormer)).

**A supplier, for contrast.** Valeo, a tier-1 supplier, is the only company we found that
attaches the data's terms: "The nuScenes models are released under the following terms",
linking to Motional's terms ([VaViT](https://github.com/valeoai/VaViT)), and, for data
derived from nuScenes, "Those data can be used for non-commercial purposes only"
([VideoActionModel](https://github.com/valeoai/VideoActionModel)). Verified.

**Our reading:** chip makers commonly show nuScenes accuracy beside the speed of their
own chips, which is closer to promoting a product than anything Paddock plans. None says
it holds a licence, and none has been challenged in public (section 6). NVIDIA's card
calls a nuScenes-trained model "ready for commercial" use without mentioning Motional's
terms: either NVIDIA holds a licence, or it reads the terms as not reaching a trained
model. We cannot tell which.

## 2. How companies publish: five channels

| Channel | Examples | With a university? | Permission stated? |
|---|---|---|---|
| Entries in the owner's own challenge | nuScenes: SenseTime, Hesai, Bosch. NAVSIM: NVIDIA. Waymo: Horizon Robotics, TuSimple. HUGSIM: NVIDIA, Xiaomi, Qualcomm. | Mostly company teams; some joint | The challenge's own terms |
| Research papers by company researchers | 14 papers, below: Bench2Drive, neuroNCAP, NAVSIM, nuScenes, Waymo | 12 of 14 | None found |
| Model libraries for the company's own chips | NVIDIA, AMD, Texas Instruments, Hailo (section 1) | No | None |
| Product marketing | Rankings, not numbers: Horizon's technology page, NVIDIA's blog. NXP's 2019 slides use KITTI data, with no score. | No | None |
| An industry benchmark | MLPerf Automotive: NVIDIA, GateOverflow (section 3) | Run by industry groups, for companies | Yes: Motional agreed |

Challenge entries are in section 4. **Research papers**, each Verified on arXiv or its
publisher's record, with the company among its authors' affiliations:

| Company, with | Paper | Benchmark | Quote |
|---|---|---|---|
| Xiaomi EV, with Huazhong University of Science and Technology | [ORION](https://arxiv.org/abs/2503.19755), March 2025 | Bench2Drive | "77.74 Driving Score (DS) and 54.62% Success Rate (SR) on the challenge Bench2Drive datasets" |
| NVIDIA, with Fudan University | [Hydra-NeXt](https://arxiv.org/abs/2503.12030), March 2025 | Bench2Drive | "65.89 Driving Score (DS) and 48.20% Success Rate (SR) on the Bench2Drive dataset" |
| Wayve, with the University of Tübingen | [SimLingo](https://arxiv.org/abs/2503.09594), March 2025 | Bench2Drive | "state-of-the-art performance on the widely used CARLA simulator on the Bench2Drive benchmark" |
| Horizon Robotics, with Xi'an Jiaotong University | [AlignDrive](https://arxiv.org/abs/2601.01762), January 2026 | Bench2Drive | "achieves SOTA performance with a driving score of 89.07" |
| BMW Group, with TU Munich, Glasgow and TU Wien | [DRIFT](https://arxiv.org/abs/2611.24719), November 2026 | Bench2Drive | "reaches up to a Driving Score (DS) of 97.03" |
| Bosch Research, with Tsinghua University | [Impromptu VLA](https://arxiv.org/abs/2505.23757), May 2025 | neuroNCAP | "improving closed-loop NeuroNCAP scores and collision rates" |
| Zenseact, with Linköping, Lund and TU Delft | [neuroNCAP](https://arxiv.org/abs/2404.07762), April 2024 | neuroNCAP itself, on nuScenes | "we invite the research community to explore, refine, and validate their AD models" |
| NVIDIA, with Fudan University and others | [Hydra-MDP](https://arxiv.org/abs/2406.06978), June 2024 | NAVSIM | Winner of the NAVSIM challenge at CVPR 2024 (section 4.3) |
| Infineon, with TU Munich | [MCAF-Net](https://arxiv.org/abs/2307.08339), July 2023 | nuScenes | "outperforms current state-of-the-art radar-camera fusion-based object detectors in the nuScenes dataset" |
| Infineon, with TU Munich | [TRIDE](https://arxiv.org/abs/2508.08038), August 2025 | nuScenes | "benchmarked on the nuScenes dataset" |
| NXP, with TU Delft | [Occupancy grid paper](https://repository.tudelft.nl/record/uuid:25d7b30d-2303-4a04-9e5b-af7a2ce8414e), IEEE SENSORS 2023 | nuScenes | "testing on scenes from the nuScenes dataset" |
| TuSimple, with the University of Central Florida | [LidarMultiNet](https://arxiv.org/abs/2206.11428), June 2022 | Waymo | "1st place winning solution for the Waymo Open Dataset 3D semantic segmentation challenge 2022" |
| Qualcomm, alone (its one university author did the work at Qualcomm) | [X-Align](https://arxiv.org/abs/2210.06778), October 2022 | nuScenes | "a new record mIoU of 65.7% on nuScenes BEV segmentation" |
| Horizon Robotics, alone | [AFDetV2](https://arxiv.org/abs/2112.09205), December 2021 | Waymo | "won the 1st place in the Real-Time 3D Detection of the Waymo Open Dataset Challenge 2021" |

None of the 13 on arXiv mentions a licence or permission for the benchmark's data; we
searched their full text, acknowledgements included (section 5). Five of them, ORION,
Impromptu VLA, neuroNCAP, LidarMultiNet and X-Align, are themselves published on arXiv
under CC BY 4.0, which lets anyone reuse their tables commercially. **Our reading:** at
the least, their authors did not treat the scores as bound by the data's non-commercial
terms.

### 2.1 Company alone, or with a university or non-profit: does it make a difference?

**What we found.** 12 of the 14 company papers have university co-authors. The company
channels without one are the commercial-looking ones: model libraries, marketing pages,
and most company teams in challenges. Industry groups and non-profits appear in three
roles: MLCommons, an industry consortium, and AVCC, a non-profit, ran MLPerf Automotive;
the Autoware Foundation, a non-profit, publishes AutoE2E; and OpenDriveLab, a research
lab, runs the NAVSIM challenges. neuroNCAP, which Paddock runs, was written by
Zenseact with three universities; one co-author is Holger Caesar, now at TU Delft, the
first author of the [nuScenes paper](https://arxiv.org/abs/1903.11027). All Verified
from the papers' author lists.

**What the terms say: the user's purpose decides, not who they are.**

| Source | Quote |
|---|---|
| Creative Commons | "NonCommercial turns on the use, not the identity of the reuser" ([wiki](https://wiki.creativecommons.org/wiki/NonCommercial_interpretation)) |
| Motional | "This is regardless of whether you work in academia or industry." |
| Cityscapes | Free "to academic and non-academic entities for non-commercial purposes" |
| Waymo | Excludes purposes "primarily intended for or directed towards commercial advantage", whoever has them |

**What a partner does change. Our reading:**

- **Who uses the data.** When a university's researchers run the experiments under the
  university's own registration and publish them, the use is the university's research,
  and the company co-author contributes ideas. That is the strongest form of the
  "scientific publication" reading. We cannot see, from a paper, who ran what or whose
  registration was used.
- **How the publication looks.** A peer-reviewed paper with university co-authors reads
  as research; a model library or a product page reads as the company's own.
- **Not the purpose.** Being a non-profit or an industry group did not settle it either:
  AVCC and MLCommons still asked Motional, because their members would use the results to
  sell hardware. The Autoware Foundation, a non-profit, released AutoE2E's weights as
  non-commercial and tells commercial users to get Motional's licence.
- **This matches Legal's view of 17 November**, which the meeting summary records as "No
  way around it if Renesas funds the work". A genuine collaboration, in which a
  university runs and publishes the evaluation under its own registration, would change
  the facts; a university used as a channel for Renesas's own release would not.
- **Paddock today** is built by Renesas and its contractors, with no university partner.

## 3. MLPerf Automotive: the one case of explicit permission

MLPerf Automotive measures how fast in-car computers run AI models. MLCommons built it
with AVCC, which Renesas joined as a core member in 2019; AVCC closed on 31 December 2025,
and MLCommons carries the benchmark on. Its 3D-detection test runs the BEVFormer model on
nuScenes, and its second round added UniAD, also on nuScenes.

| Point | Quote | Source |
|---|---|---|
| Results help sell hardware | "benchmarks in the MLPerf Automotive benchmark suite are being adopted into the Request for Information/Quotation (RFI/RFQ) process" | [Paper v2](https://arxiv.org/html/2510.27065v2), §1 |
| The licence problem | "Benchmark submissions are not a direct commercial activity; however, submitters might use benchmark results as part of a customer request for quotation. This legal ambiguity limits using any public dataset." | [Paper v1](https://arxiv.org/html/2510.27065v1), §2.7 |
| Their answer | "Instead, we need to obtain explicit permission to use datasets for the benchmark." | Paper v1, §2.7 |
| Motional agreed | "In the case of nuScenes, both parties were in agreement to allow the usage for the benchmark, but legal details of the license required lawyers and time." | Paper v1, §2.7 |
| How long | "the entire process required months" | Paper v1, §2.7 |
| Motional's part | "Motional provided its nuScenes dataset for the 3-D object recognition test." | [MLCommons, 27 August 2025](https://mlcommons.org/2025/08/mlperf-auto-v0-5-results/) |
| Access is gated | "Contact MLCommons support for accessing the NuScenes dataset." | [Benchmark README](https://github.com/mlcommons/mlperf_automotive/blob/master/automotive/camera-3d-detection/README.md) |

- **Verified:** the working group that designed the benchmark included "Ambarella, ARM,
  Bosch, C-Tuning Foundation, CeCaS, Cognata, Motional, NVIDIA, Qualcomm" and others
  (Samsung, ZF among them). The first round's submitters were "GateOverflow and Nvidia".
  Motional's own speaker: "Motional is proud to continue supporting the development of
  safe autonomous systems with the use of nuScenes" (MLCommons, 27 August 2025).
- **Verified:** the paper's later version (v2, 23 September 2026) keeps the sentences on
  licensing, but no longer has the two sentences on the nuScenes agreement. Version 1
  remains on arXiv.
- **Verified:** Renesas joined AVCC "as a core member" on 30 October 2019
  ([Renesas](https://www.renesas.com/en/about/press-room/renesas-electronics-joins-autonomous-vehicle-computing-consortium-speed-autonomous-driving-vehicle)).
  AVCC "concluded operations on December 31, 2025" ([AVCC](https://avcc.org/)).
- **Our reading:** this is the closest precedent to Paddock. Its scores are chip makers'
  scores, and customers use them to choose suppliers: the concern Legal raised. Its
  organisers did not rely on calling the work research; they asked Motional, and Motional
  agreed. The terms of that agreement are not public.

## 4. What the dataset owners say

### 4.1 Motional: nuScenes and nuPlan, and through them NAVSIM and neuroNCAP

| Point | Quote | Source |
|---|---|---|
| What is free | "For non-commercial use nuScenes and nuImages are free, e.g. for educational use and some research use." | [FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md) |
| When a licence is needed | "for any activities with the expectation of generating revenue either at present or in the future, such as industrial research and development (“R&D”)" | [Commercial terms](https://www.nuscenes.org/terms-of-use-commercial) |
| For whom | "This is regardless of whether you work in academia or industry." | Commercial terms |
| Results | "Use of any data derived from the Datasets, which may appear in any format such as tables and charts, is also subject to these Dataset Terms." | [Terms of use](https://www.nuscenes.org/terms-of-use) |
| Price | "commercial license pricing varies and is customized for your organization and use case" | Commercial terms |
| In 2020 | "nuScenes is free of charge for academic use, and licensing is available for commercial purposes." | [Press release, 2 September 2020](https://www.prnewswire.com/news-releases/motional-expands-groundbreaking-nuscenes-datasets-to-enable-a-safer-smarter-autonomous-driving-industry-301122455.html) |

Both terms pages say they were "last updated on Nov 16, 2021". The licence review quoted
only the FAQ; the commercial-terms sentence is new to this review, and is the clearest
statement of where Motional draws the line.

**Challenges that Motional runs or supplies, all Verified:**

- **nuScenes detection challenge**, hosted on EvalAI by the team "Motional": 781 entries
  on 20 November 2026. Among the first 100, teams named after companies: SenseTime (2nd),
  Leapmotor (3rd), Hesai (7th), Meituan (45th), FiveAI (54th) and Bosch (86th). Teams
  choose their own names ([leaderboard](https://eval.ai/web/challenges/challenge-page/356/leaderboard/1012)).
- **nuPlan planning challenge**, 2023: the organiser collects a "school or company name",
  posts "a list of the contestants' names, or team names in ranked order", and allows the
  data "solely to prepare for and compete in such Competition"
  ([EvalAI](https://eval.ai/web/challenges/challenge-page/1856/overview)).
- **nuReasoning challenge**, 2026: "The Organizer may publish leaderboard results, team
  names, scores, rankings, method names, affiliations, and summaries of submitted
  methods." Also: "Continued use of the datasets for commercial purposes is governed by
  Motional's Commercial Terms." ([terms](https://www.nuscenes.org/nureasoning/terms))
- **NAVSIM challenges**, run by OpenDriveLab: "The private test set is provided by nuPlan
  from Motional." NVIDIA took the Innovation Award in 2024 and was champion in 2025
  (section 4.3).
- **MLPerf Automotive**: Motional supplied nuScenes by agreement (section 3).

**Not found:** any list or count of Motional's commercial licensees, and any statement by
Motional about companies publishing scores outside its own challenges.

**Our reading:** Motional welcomes companies in its challenges and names them, and it
agreed to nuScenes' use in an industry benchmark. Outside those, its terms treat research
done with "the expectation of generating revenue" as commercial, whoever does it.

### 4.2 Waymo: the Waymo Open Dataset, behind HUGSIM's Waymo scenes

| Point | Quote |
|---|---|
| Non-commercial means | "research, teaching, scientific publication and personal experimentation" |
| Benchmarking | "Non-commercial Purposes include use of the Dataset to perform benchmarking for purposes of academic or applied research publication." |
| But not | "purposes primarily intended for or directed towards commercial advantage or monetary compensation" |
| Credit | Work made with the data must say: "This [type of work, e.g., publication, software, model] was made using the Waymo Open Dataset" |
| Commercial use | "please contact open-dataset@waymo.com" |

Source: [Waymo's licence](https://waymo.com/open/terms/), dated March 2025, Verified.

- **Verified:** its challenges accept "an individual employed by or otherwise affiliated
  with an entity", and pay no award "to any employer". Companies have won: Horizon
  Robotics' AFDetV2 "won the 1st place in the Real-Time 3D Detection of the Waymo Open
  Dataset Challenge 2021" ([paper](https://arxiv.org/abs/2112.09205)), and TuSimple's
  LidarMultiNet was the "1st place winning solution" in 2022
  ([paper](https://arxiv.org/abs/2206.11428)).
- **Verified:** "While we are not hosting formal Challenges in 2026, the Waymo Open
  Dataset leaderboards remain fully active. We encourage researchers to continue
  submitting their work to benchmark models, share findings, and showcase new
  breakthroughs" ([challenges](https://waymo.com/open/challenges/)).
- **Our reading:** Waymo's licence is the only one here that names publishing benchmark
  results as non-commercial, and "applied research publication" covers a company's
  research paper. It still excludes a purpose "primarily" commercial.

### 4.3 OpenDriveLab: OpenScene and the NAVSIM challenges

- **Verified:** OpenScene, NAVSIM's data, is distributed under "Creative Commons
  Attribution-NonCommercial-ShareAlike license and nuPlan Dataset License Agreement for
  Non-Commercial Use" ([OpenScene](https://github.com/OpenDriveLab/OpenScene)).
- **Verified, 2024 challenge:** "An entity can have multiple teams." 143 teams made 463
  submissions to the NAVSIM track, and "Innovation Award goes to Team NVIDIA", worth USD
  9,000 ([2024](https://opendrivelab.com/challenge2024/)). NVIDIA announced it
  ([NVIDIA blog](https://blogs.nvidia.com/blog/auto-research-cvpr-2024/)).
- **Verified, 2025 challenge:** "The private test set is provided by nuPlan from
  Motional." At CVPR, "The outstanding Champion goes to team NVIDIA." At ICCV, an
  Innovation Award went to "bjtu_jia_team&qcraft", QCraft being a self-driving company
  ([2025](https://opendrivelab.com/challenge2025/)).

### 4.4 KIT: KITScenes

- **Verified:** CC BY-NC 4.0 with KIT's terms, "intended for academic research" (licence
  review, section 9).
- **Verified:** KIT's own site publishes other parties' scores, as Paddock does: "Seven
  open-source models, evaluated zero-shot from their public checkpoints"
  ([KITScenes benchmark](https://kitscenes.com/benchmarks/multimodal-e2e-driving)).
- **Not found:** a challenge, company entries, a commercial licence offer, or any
  statement about who may publish results.

### 4.5 Thinklab, Shanghai Jiao Tong University: Bench2Drive

- **Verified:** "All assets and code are under the CC-BY-NC-ND unless specified
  otherwise" ([README](https://github.com/Thinklab-SJTU/Bench2Drive)). There is no
  hosted leaderboard; the README lists published scores, including a model with Wayve
  co-authors.
- **Verified:** company research labs publish Bench2Drive scores in papers (section 2).
- **Not found:** any statement by Thinklab on company use, permission or a commercial
  licence.

### 4.6 neuroNCAP and HUGSIM

- **neuroNCAP**, written by Zenseact with three universities (section 2.1). **Verified:**
  Bosch Research co-wrote Impromptu VLA, which reports "improving closed-loop NeuroNCAP
  scores and collision rates" ([paper](https://arxiv.org/abs/2505.23757)).
- **HUGSIM.** **Verified:** its challenge at the ICCV 2025 RealADSim workshop paid
  "Innovation Award: $9,000 Outstanding Champion: $9,000". Its closed-loop track, run on
  HUGSIM, ranked "UT/NV" and "NVIDIA/FDU" first and second; its image track ranked
  "XiaomiEV Team" and "Qualcomm AI Research" first and second
  ([RealADSim 2025](https://realadsim.github.io/2025/)).

### 4.7 NVIDIA: AlpaSim

- **Verified:** NVIDIA's NuRec licence treats "any results of benchmarking" as NVIDIA's
  confidential information (licence review, Q1).
- **Verified:** NVIDIA nevertheless ran a public AlpaSim challenge in 2026, under its own
  terms. One paper reports that its model "ranks third in the 2026 NVIDIA AlpaSim E2E
  Driving Challenge" ([paper](https://arxiv.org/abs/2611.19922)).
- **Verified:** a participant objected that the challenge's late terms "can make
  participation impossible or unacceptable for industrial teams or startups". NVIDIA
  replied: "it is more about preventing exposure of the company rather than trying to
  expose teams' solutions" ([issue #188](https://github.com/NVlabs/alpasim/issues/188),
  22 September 2026).
- **Our reading:** AlpaSim scores become public only through NVIDIA's own challenge, on
  NVIDIA's terms. This does not change Legal's "No" for Paddock (Q1).

### 4.8 Two older datasets, for comparison

- **Cityscapes** (Daimler, MPI Informatics and TU Darmstadt). **Verified:** "made freely
  available to academic and non-academic entities for non-commercial purposes such as
  academic research, teaching, scientific publications, or personal experimentation",
  but not "using the data with a purpose to procure a commercial gain"
  ([licence](https://www.cityscapes-dataset.com/license/)).
- **KITTI** (Karlsruhe Institute of Technology and Toyota Technological Institute at
  Chicago). **Verified:** published under CC BY-NC-SA 3.0, so "you may not use this work
  for commercial purposes" ([KITTI](https://www.cvlibs.net/datasets/kitti/)). Its
  leaderboard accepts "only submissions with significant novelty that are leading to a
  peer-reviewed paper", and ranks entries from companies, such as Uber's former
  self-driving unit, "UberATG-MMF"
  ([3D detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)).
- **Our reading:** both let a company publish research, and both draw the line at
  commercial gain. They are the norm that Waymo's licence writes down and that Motional's
  commercial terms narrow.

## 5. Did they get permission?

Almost nobody says. Two arrangements with Motional are public: an industry benchmark's
and a start-up's partnership. One open-model publisher, a non-profit, tells its users to
get Motional's licence, and one supplier follows the non-commercial terms. Everyone else
is silent.

| Who | Dataset | What is stated | Source |
|---|---|---|---|
| MLCommons, with AVCC | nuScenes | Motional agreed to the benchmark's use, after months of legal work | Section 3 |
| SiaSearch, a Berlin start-up, 2021 | nuScenes | "Motional recently partnered with Berlin-based startup SiaSearch"; "nuScenes is available for commercial use under our commercial license agreement" | [SiaSearch](https://dev.to/siasearch/effortlessly-explore-the-nuscenes-dataset-with-siasearch-275) |
| Autoware Foundation, AutoE2E's weights | nuPlan, nuScenes | Commercial use "needs permission from the upstream rights holders, including a commercial license from Motional"; the weights are released non-commercial | [Model card](https://huggingface.co/AutowareFoundation/auto_e2e) |
| Challenge entrants | nuScenes, NAVSIM, Waymo, HUGSIM | Each challenge's own terms | Section 4 |
| Company research papers | Bench2Drive, neuroNCAP, NAVSIM, nuScenes, Waymo | Nothing found | Section 2 |
| Chip makers' model libraries | nuScenes | Code licences only (MIT, Apache-2.0); Texas Instruments: "not meant for deployment in production" | Section 1 |
| NVIDIA, a model card | nuScenes | A nuScenes-trained model "is ready for commercial/non-commercial use"; Motional not mentioned | Section 1 |
| Valeo, a supplier | nuScenes, Waymo | Links its models to Motional's and Waymo's terms; nuScenes-derived data "for non-commercial purposes only" | Section 1 |
| Motional | | Names no licensees; prices each licence for "your organization and use case" | Section 4.1 |

All Verified. For the papers, we searched the full text of the 13 on arXiv in section
2, acknowledgements included, for "licence", "permission", "non-commercial" and
"Motional", and found no statement about the data's licence. The NXP paper's full text
is not open, and was not read.

**Our reading:** silence does not mean no licence. Motional sells licences, treats
industrial R&D as needing one, and does not publish who holds them. A company that uses
nuScenes in its own development may well hold one that also covers its papers. So
"others publish" cannot tell Renesas whether they publish with permission.

## 6. Issues and disputes

**In short:** we found no case of a dataset owner objecting to a company's published
scores. What owners and projects police is copies of the data and trained weights. Courts
judge "non-commercial" by the user's purpose, differently from country to country, and
none has ruled on a company promoting its own products. The one live lawsuit by a dataset
owner against a chip maker is about training products.

### 6.1 Takedowns and withdrawals

- **No takedown notices. Verified.** GitHub publishes every copyright takedown notice it
  receives (notices under the US DMCA) in a public archive. We searched all 22,457
  notices as of 20 November 2026: none comes from or concerns Motional, nuScenes, nuPlan,
  Waymo's dataset, KITTI, Cityscapes, Argoverse, ImageNet, OpenScene, NAVSIM, Bench2Drive,
  KITScenes or comma2k19 ([archive](https://github.com/github/dmca)). Takedowns sent to
  other sites, such as Hugging Face, are not public and were not checked.
- **Weights withdrawn, scores kept. Verified.** OpenMMLab's MMDetection3D recorded "Delete
  checkpoints on Waymo to comply its specific license agreement", and later "Fix the
  SECOND results on Waymo in the corresponding README"
  ([changelog](https://mmdetection3d.readthedocs.io/en/latest/notes/changelog_v1.0.x.html)).
  OpenPCDet, directly below its table of Waymo scores: "We could not provide the above
  pretrained models due to Waymo Dataset License Agreement, but you could easily achieve
  similar performance by training with the default configs"
  ([README](https://github.com/open-mmlab/OpenPCDet/blob/master/README.md)).
- **Our reading:** both projects judged that Waymo's licence reaches trained weights but
  not published scores. Paddock already publishes no weights.

### 6.2 A dataset owner suing a chip maker

**Verified, as allegations:** "On June 22, 2026, Jamendo, a music licensing platform
owned by Belgium-based Winamp Group, sued NVIDIA ... accusing the chipmaker of training
its Fugatto and Audio Flamingo audio models on Jamendo's music without authorization."
"The dataset was released under a non-commercial Creative Commons license and an express
research-use notice." The claims include "breach of contract (the Creative Commons BY-NC
license and Jamendo's terms of use)"
([Kronenberger Rosenfeld, 22 July 2026](https://kr.law/news/article-detail/jamendos-copyright-suit-against-nvidia-and-suno)).
**Reported:** the suit was withdrawn and filed again in October 2026. Nothing is decided.

**Our reading:** this is the first case we found of an owner taking a non-commercial
licence to court against a chip maker. It is about building products from the data,
which Paddock does not do; but it shows that a request to take things down is not always
the worst case once a product is involved.

### 6.3 What courts have said "non-commercial" means

| Court | What it held | Quote |
|---|---|---|
| US Court of Appeals, 2nd Circuit: Great Minds v. FedEx Office, 21 March 2018 | A non-commercial licensee, a school district, may have a commercial copy shop do its copying | "licensees may use third-party assistance in exercising their rights under non-exclusive copyright licenses" ([opinion](https://caselaw.findlaw.com/court/us-2nd-circuit/1922945.html)) |
| US Court of Appeals, 9th Circuit: Great Minds v. Office Depot, 27 December 2019 | The same | "a non-commercial licensee may hire a third-party contractor, including those working for commercial gain, to help implement the License" ([opinion](https://cdn.ca9.uscourts.gov/datastore/opinions/2019/12/27/18-55331.pdf)) |
| Cologne Regional Court, 5 March 2014 | Non-commercial means purely private use, so a public broadcaster's use breached CC BY-NC | "allein eine rein private Nutzung" ("purely private use only"), as the appeal court restates it |
| Cologne Higher Regional Court, 31 October 2014 (6 U 60/14) | Reversed on that point: doubts count against the licensor who wrote the terms, so a public body's use with no "direct financial advantage" is non-commercial. Still liable for cropping out the photographer's credit. | "zu Lasten des Verwenders" ("against the party that drafted the terms"); damages for non-commercial use "mit Null angesetzt" ("set at zero"); EUR 482.13 in costs ([judgment](https://nrwe.justiz.nrw.de/olgs/koeln/j2014/6_U_60_14_Urteil_20141031.html)) |

All Verified. Creative Commons' own study, "Defining Noncommercial" (2009), found that
"uses by for-profit companies are typically considered more commercial", and that "there
is more uncertainty than clarity around whether specific uses of online content are
commercial or noncommercial"
([report](https://mirrors.creativecommons.org/defining-noncommercial/Defining_Noncommercial_fullreport.pdf)).

**Our reading:** the US cases look at the licensee's own purpose; the German cases split;
none involves a company using material to promote its products. Courts give no answer to
Q2 either way.

### 6.4 Other risks seen in practice

- **People who sue over licence slips for money. Verified.** Creative Commons, 8 February
  2022: "license enforcement has even become a business model", which it calls "a
  perversion of the founding ideals of Creative Commons"
  ([Creative Commons](https://creativecommons.org/2022/02/08/copyleft-trolls/)). Such
  claims usually turn on a missing credit. Under version 4.0 licences, a breach fixed
  "within 30 days of Your discovery of the violation" restores the rights
  ([4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)); under version
  3.0, the rights "terminate automatically upon any breach", with no such cure
  ([3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode)). KITTI-360, one
  of HUGSIM's sources, is under version 3.0.
- **MLCommons' own rules. Verified.** For its YOLO benchmark: "The full coco dataset has
  images that are not compliant with MLC legal rules", so it filters them out
  ([README](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/yolo/README.md)).
  For a benchmark on Waymo's data: "One has to accept the MLCommons Waymo Open Dataset
  EULA", and the data is hosted "exclusively by MLCommons Members"
  ([README](https://github.com/mlcommons/inference/blob/master/automotive/3d-object-detection/README.md)).
  Yet an older benchmark still runs on Criteo's data, whose "CRITEO DATA TERMS OF USE FOR
  MLPerf Users" are CC BY-NC-SA 4.0
  ([Criteo](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/)),
  and vendors publish those results.
- **The research exception and company influence. Verified.** EU law counts a body as a
  research organisation only if its results "cannot be enjoyed on a preferential basis by
  an undertaking that exercises a decisive influence upon such organisation" (Directive
  2019/790, Art. 2(1), [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019L0790)).
  In Kneschke v. LAION, the Hamburg Higher Regional Court held on 10 December 2025 that
  LAION, a non-profit, kept its research status because a private company's "bestimmenden
  Einfluss" ("decisive influence") was missing
  ([summary](https://medien-internet-und-recht.de/volltext.php?mir_dok_id=3526)). The
  Federal Court of Justice heard the appeal on 3 September 2026 (I ZR 281/25,
  [press release](https://www.bundesgerichtshof.de/SharedDocs/Pressemitteilungen/DE/2026/2026085.html)).
  **Our reading:** this is copyright law's research exception, not a dataset licence;
  but it uses the same test as Legal's view on publishing through a university.

## 7. Why companies can do it, and what is acceptable

### 7.1 Six reasons companies publish, and how far each goes

| Reason | How far it goes |
|---|---|
| 1. The owner invites it | Far, but only inside the owner's own challenge, under that challenge's terms (section 4). |
| 2. Research publication reads as non-commercial | Waymo writes this down, and Cityscapes and KITTI allow companies' research. Motional's commercial terms narrow it: research done with "the expectation of generating revenue" needs a licence. |
| 3. A score is a fact | Creative Commons licences do not reach facts. Motional's terms reach "tables and charts" by contract (licence review, section 5). |
| 4. Owners gain from it | Each company score that cites a dataset makes it more of a standard. Motional says nuScenes "pioneered an industry-wide culture of data-sharing" (MLCommons, 27 August 2025). Our reading: this explains why owners rarely object; it does not give a right to publish. |
| 5. Some hold licences | Possible, and not public (section 5). |
| 6. Nobody has been pursued | For published scores, we found no case (section 6). For building products from non-commercial data, one suit against a chip maker is pending: Jamendo v. NVIDIA, 2026. |

### 7.2 Patterns, from invited to licensed

| Pattern | How common | Owner's position | Paddock |
|---|---|---|---|
| Entering the owner's challenge | Very common | Invited, under the challenge's terms | Not Paddock |
| A company lab's research paper | Very common | Waymo, Cityscapes, KITTI: allowed. Motional: a licence for industrial R&D. | Closest, if released as research |
| A public benchmark of others' models | Usual for universities: KIT, Thinklab, neuroNCAP's authors | Not addressed | This is Paddock |
| A model library with accuracy on the company's chips | Common: NVIDIA, AMD, Texas Instruments, Hailo | Not addressed | Not Paddock |
| Product marketing that quotes scores | Rankings only: Horizon Robotics, NVIDIA. No numbers found. | Our reading: "generating revenue", in Motional's terms | Keep apart |
| An industry benchmark used to sell hardware | MLPerf Automotive | Permission obtained | The model for asking |
| Products built with the data | Not checked here | Motional: a licence. Waymo: barred from vehicles. | Not Paddock |

### 7.3 Where Paddock sits

**Our reading:**

- **In form, Paddock's release follows the research pattern.** Its code is free and open.
  It scores other parties' models, publishes its method, credits every source, and hosts
  no data, weights or videos.
- **In purpose, Legal sees promotion of Renesas.** Purpose is what Motional's commercial
  terms and Waymo's "primarily" clause look at. It is also what made MLPerf's organisers
  ask Motional first.
- **What keeps it on the research side** is what companies' papers do: no Renesas
  product names or chip speeds beside scores on non-commercial data, scores presented as
  a technical report, each result marked with its data's licence (Q4), the disclaimer of
  Q14, and a way to take results down on request.
- **The only route to certainty is permission.** One letter from Motional covers five of
  the eight non-commercial benchmarks; KIT and Thinklab cover two more.

## 8. How this was checked

- **When:** on 20 November 2026. Web pages change; each quote is as read that day.
- **Motional's terms:** nuscenes.org needs a browser to show its pages, so we read their
  text from the site's own script files, the files a browser runs to draw them: the terms
  of use, the commercial terms and the nuReasoning terms. Both terms pages say "last
  updated on Nov 16, 2021", and the "tables and charts" sentence is on both. This settles
  the licence review's request (its section 12) to confirm its copy of Motional's terms
  against the live page: they match.
- **Leaderboards:** EvalAI's public interface, for Motional's nuScenes detection challenge
  and its nuPlan challenge; the challenge pages themselves for OpenDriveLab, Waymo and
  RealADSim.
- **The MLPerf Automotive paper:** both versions on arXiv were downloaded and compared.
- **Papers:** each paper's abstract on arXiv for its score, its first page for its
  authors' companies, and its full text for any statement about the data's licence.
- **Takedowns:** GitHub's archive was downloaded whole on 20 November 2026, and every
  notice's text searched for each dataset's and owner's name.
- **Links:** all 75 open. Three sites refuse automated downloads (NXP's, MLCommons' news
  page and FindLaw's), so those pages were read through a web-reading tool instead.
- **Searching:** AI research assistants ran the first searches. Every quote marked
  Verified was then opened again by us at its source; the rest is marked Reported.
- **Not checked:** private licences, which are not public; the NXP paper's full text;
  papers beyond the 14 listed; takedowns sent to sites other than GitHub; pages in
  Chinese, such as those of Chinese chip makers; and Samsung, MediaTek, Continental, ZF,
  Aptiv and Denso, which the search did not reach.

## References

All opened on 20 November 2026; section 8 notes the few that refuse automated access.
Each source is listed once, under the first section that uses it.

**Chip makers (section 1)**

- Infineon, with TU Munich: [MCAF-Net](https://arxiv.org/abs/2307.08339), July 2023;
  [TRIDE](https://arxiv.org/abs/2508.08038), August 2025.
- NXP, with TU Delft: [occupancy grid paper](https://repository.tudelft.nl/record/uuid:25d7b30d-2303-4a04-9e5b-af7a2ce8414e),
  IEEE SENSORS 2023. NXP: [ADAS vision overview slides](https://www.nxp.com/docs/en/training-reference-material/ADAS-&-DRIVER-REPLACEMENT-VISION-OVERVIEW-FOR-ADAS-SYSTEMS.pdf),
  June 2019, slide 15.
- NVIDIA: [CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion);
  [BEVFormer model card](https://docs.api.nvidia.com/nim/reference/nvidia-bevformer);
  [blog, 17 June 2024](https://blogs.nvidia.com/blog/auto-research-cvpr-2024/);
  [blog, 11 June 2025](https://blogs.nvidia.com/blog/auto-research-cvpr-2025/).
- Horizon Robotics: [technology page](https://www.horizon.auto/en/technology);
  [DiffusionDrive model card](https://huggingface.co/hustvl/DiffusionDrive).
- AMD: [Vitis AI 1.4 model library](https://github.com/Xilinx/Vitis-AI/tree/v1.4/models/AI-Model-Zoo).
- Texas Instruments: [edgeai-mmdetection3d](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-mmdetection3d);
  [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo).
- Hailo: [3D detection models for Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_object_detection_3d.rst).
- Qualcomm: [X-Align](https://arxiv.org/abs/2210.06778), October 2022;
  [BEVFormer on Qualcomm AI Hub](https://huggingface.co/qualcomm/BEVFormer).
- Valeo: [VaViT](https://github.com/valeoai/VaViT);
  [VideoActionModel](https://github.com/valeoai/VideoActionModel).

**Company papers (section 2)**

- Bench2Drive: [ORION](https://arxiv.org/abs/2503.19755), [Hydra-NeXt](https://arxiv.org/abs/2503.12030),
  [SimLingo](https://arxiv.org/abs/2503.09594), [AlignDrive](https://arxiv.org/abs/2601.01762),
  [DRIFT](https://arxiv.org/abs/2611.24719).
- neuroNCAP: [neuroNCAP](https://arxiv.org/abs/2404.07762), [Impromptu VLA](https://arxiv.org/abs/2505.23757).
- NAVSIM: [Hydra-MDP](https://arxiv.org/abs/2406.06978).
- Waymo: [LidarMultiNet](https://arxiv.org/abs/2206.11428), [AFDetV2](https://arxiv.org/abs/2112.09205).
- nuScenes itself: [the nuScenes paper](https://arxiv.org/abs/1903.11027).
- Creative Commons: [NonCommercial interpretation](https://wiki.creativecommons.org/wiki/NonCommercial_interpretation).

**MLPerf Automotive (section 3)**

- The paper: [version 1, 31 October 2025](https://arxiv.org/html/2510.27065v1), §2.7;
  [version 2, 23 September 2026](https://arxiv.org/html/2510.27065v2), §1 and §2.6.
- [MLCommons' announcement, 27 August 2025](https://mlcommons.org/2025/08/mlperf-auto-v0-5-results/).
- [The benchmark's README on nuScenes access](https://github.com/mlcommons/mlperf_automotive/blob/master/automotive/camera-3d-detection/README.md).
- [Renesas joins AVCC, 30 October 2019](https://www.renesas.com/en/about/press-room/renesas-electronics-joins-autonomous-vehicle-computing-consortium-speed-autonomous-driving-vehicle);
  [AVCC's closing notice](https://avcc.org/).

**Dataset owners (section 4)**

- Motional: [terms of use](https://www.nuscenes.org/terms-of-use);
  [commercial terms](https://www.nuscenes.org/terms-of-use-commercial);
  [FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md);
  [press release, 2 September 2020](https://www.prnewswire.com/news-releases/motional-expands-groundbreaking-nuscenes-datasets-to-enable-a-safer-smarter-autonomous-driving-industry-301122455.html);
  [nuScenes detection leaderboard](https://eval.ai/web/challenges/challenge-page/356/leaderboard/1012);
  [nuPlan challenge](https://eval.ai/web/challenges/challenge-page/1856/overview);
  [nuReasoning challenge terms](https://www.nuscenes.org/nureasoning/terms).
- Waymo: [licence and 2025 challenge rules](https://waymo.com/open/terms/);
  [challenges](https://waymo.com/open/challenges/).
- OpenDriveLab: [OpenScene](https://github.com/OpenDriveLab/OpenScene);
  [challenge 2024](https://opendrivelab.com/challenge2024/);
  [challenge 2025](https://opendrivelab.com/challenge2025/).
- KIT: [KITScenes benchmark](https://kitscenes.com/benchmarks/multimodal-e2e-driving).
- Thinklab: [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive).
- HUGSIM: [RealADSim 2025 challenge](https://realadsim.github.io/2025/).
- NVIDIA AlpaSim: [issue #188](https://github.com/NVlabs/alpasim/issues/188);
  [Lao Siji](https://arxiv.org/abs/2611.19922).
- Cityscapes: [licence](https://www.cityscapes-dataset.com/license/). KITTI:
  [home page](https://www.cvlibs.net/datasets/kitti/);
  [3D detection leaderboard](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

**Permission (section 5)**

- [SiaSearch, 12 April 2021](https://dev.to/siasearch/effortlessly-explore-the-nuscenes-dataset-with-siasearch-275).
- [AutoE2E model card](https://huggingface.co/AutowareFoundation/auto_e2e).

**Issues and disputes (section 6)**

- Takedowns: [GitHub's takedown archive](https://github.com/github/dmca);
  [MMDetection3D changelog](https://mmdetection3d.readthedocs.io/en/latest/notes/changelog_v1.0.x.html);
  [OpenPCDet README](https://github.com/open-mmlab/OpenPCDet/blob/master/README.md).
- Lawsuit: [Jamendo v. NVIDIA, as described by Kronenberger Rosenfeld](https://kr.law/news/article-detail/jamendos-copyright-suit-against-nvidia-and-suno).
- Courts: [Great Minds v. FedEx Office](https://caselaw.findlaw.com/court/us-2nd-circuit/1922945.html);
  [Great Minds v. Office Depot](https://cdn.ca9.uscourts.gov/datastore/opinions/2019/12/27/18-55331.pdf);
  [Cologne Higher Regional Court, 6 U 60/14](https://nrwe.justiz.nrw.de/olgs/koeln/j2014/6_U_60_14_Urteil_20141031.html).
- Creative Commons: [Defining Noncommercial, 2009](https://mirrors.creativecommons.org/defining-noncommercial/Defining_Noncommercial_fullreport.pdf);
  [on copyleft trolls, 2022](https://creativecommons.org/2022/02/08/copyleft-trolls/);
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en);
  [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode).
- MLCommons: [YOLO benchmark README](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/yolo/README.md);
  [Waymo-based benchmark README](https://github.com/mlcommons/inference/blob/master/automotive/3d-object-detection/README.md);
  [Criteo's terms for MLPerf users](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/).
- Research exception: [Directive (EU) 2019/790](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019L0790);
  [Kneschke v. LAION, appeal summary](https://medien-internet-und-recht.de/volltext.php?mir_dok_id=3526);
  [Federal Court of Justice, hearing notice](https://www.bundesgerichtshof.de/SharedDocs/Pressemitteilungen/DE/2026/2026085.html).
