# Paddock public release: licence review and decisions for Renesas Legal

*Prepared by the Paddock team for Renesas Legal. First issued on 10 November 2026, and
checked again against Paddock's `main` at `73e9ac6` on 16 November 2026. A fact sheet to
support a legal review, not legal advice: where it interprets a term it says so, and the
decisions are Legal's. Part A is the decision list for the release meeting; Part B holds
the evidence.*

# Part A. Decisions for the release meeting

Paddock is Renesas's open-source tool for testing self-driving models on public
benchmarks. Renesas plans to publish its code, and the scores it has measured, on GitHub
and on a website built from that code. Paddock hosts no dataset and no model weights: each
user downloads those from their owners.

**Our recommendation: release once four conditions are met.**

1. 🔴 Nothing from NVIDIA's AlpaSim is published: no scores, page or videos (Q1).
2. 🔴 Motional confirms in writing that this release is non-commercial use (Q2).
3. 🔴 Legal confirms that Renesas owns all of Paddock's code (Q3).
4. The changes that the answers call for are made (listed in section 6, each marked
   with the colour of its question).

Terms used below:

- **Model**: software that turns a car's camera images into the path the car should
  drive. Its **weights** are the file of trained numbers that makes it work. Code and
  weights often have different licences.
- **Benchmark**: a fixed test that gives a model a score, on a **dataset** of recorded
  drives or in a **simulator**, a program that drives a virtual car.
- **Non-commercial**: use "not primarily intended for or directed towards commercial
  advantage or monetary compensation", in Creative Commons' words.
- **Patch**: a file of line-by-line edits to someone else's code.
- **Run record**: the small file Paddock keeps for each test run, with its scores and
  settings.
- **Hugging Face**: a public website that hosts models, datasets and videos.

**Colours.** Every question, and every part of the release in Part B, carries one of
three colours:

| Colour | On a question | On a part of the release |
|---|---|---|
| 🔴 Red: critical | It can block the release or expose Renesas to a claim. Decide it in the meeting. | Left out of the release, unless its owner gives permission |
| 🟡 Amber: non-critical | Low risk, or a cheap fix. Accept the recommendation unless someone objects. | Released after a change, or on a condition such as Q2 |
| 🟢 Green: clear decision | Nothing left to weigh. Confirm it for the record. | Released as it is, or nothing left to do |

**How to use the list.** Each question takes a yes or a no. The 7 red questions in Tier
1 need discussion. The 9 amber questions in Tier 2, and then the 10 green ones in Tier
3, can each be settled in one vote, accepting every recommendation unless someone
objects. Part B, section 1, gives each question's background and what follows from
each answer, under the same number. Write the answers in the last column; the Paddock
team then commits them to the git branch `legal/release-licence-review`, which keeps a
record of the decisions.

### 🔴 Tier 1. Critical: decide in the meeting

| # | Question | We recommend | Answer |
|---|---|---|---|
| 🔴 Q1 | May Paddock publish anything from AlpaSim: its scores, page, chart column, videos or captions? | **No**, unless NVIDIA agrees in writing | |
| 🔴 Q2 | Is Renesas publishing this free evaluation a non-commercial use, as most of the datasets require? | **Yes**, confirmed in writing by Motional | |
| 🔴 Q3 | Does Renesas own all of Paddock's code, including work by people outside Renesas? | **Yes**, once the contracts confirm it | |
| 🔴 Q4 | Must results from Motional's data carry Motional's licence (CC BY-NC-SA 4.0), not Paddock's Apache-2.0? | **Yes** | |
| 🔴 Q5 | May the repository keep small, credited excerpts of nuScenes, KITScenes and Bench2Drive data? | **Yes**, if Q2 is yes | |
| 🔴 Q6 | May the website show four videos rendered from nuScenes scenes? | **Yes**, once checked for faces and licence plates, moved to a Renesas account and credited | |
| 🔴 Q7 | May Paddock publish two patches to Bench2Drive's code, whose licence forbids sharing changed versions? | **No**, until Bench2Drive accepts the fixes or allows it | |

### 🟡 Tier 2. Non-critical: accept the recommendations unless someone objects

| # | Question | We recommend | Answer |
|---|---|---|---|
| 🟡 Q8 | May Paddock publish scores of models whose weights or training data are non-commercial? | **Yes**, under Q2 | |
| 🟡 Q9 | May the repository publish nuScenes download addresses that nuscenes.org gives only after sign-in? | **No**: users sign in and download themselves | |
| 🟡 Q10 | Should Paddock treat HUGSIM's scenes as non-commercial and Waymo-restricted, though they are tagged MIT? | **Yes** | |
| 🟡 Q11 | Should the results carry the credit lines that Waymo and KIT ask for? | **Yes** | |
| 🟡 Q12 | Should `paddock setup` show each model's weights licence, and ask users to accept non-commercial ones? | **Yes** | |
| 🟡 Q13 | May the run records be published as they are, although they name contributors, FastCode and Renesas hosts? | **No**: use neutral names in the release | |
| 🟡 Q14 | Should the README and website say that the scores are research measurements, not safety assessments? | **Yes** | |
| 🟡 Q15 | Is openpilot's request that every user indemnify comma.ai acceptable? | **Yes**: Paddock does not redistribute openpilot | |
| 🟡 Q16 | May Renesas publish code written mostly with an AI coding assistant? | **Yes**, subject to Renesas's AI policy | |

### 🟢 Tier 3. Clear decisions: confirm for the record

| # | Question | We recommend | Answer |
|---|---|---|---|
| 🟢 Q17 | Is it acceptable to point users to each owner's downloads and host nothing, for all upstream code, weights and data? | **Yes** | |
| 🟢 Q18 | Publish Paddock as a new repository, without its history, under a GitHub organisation Renesas controls? | **Yes** | |
| 🟢 Q19 | Is THIRD_PARTY_NOTICES.md enough, once the four items it misses are added? | **Yes** | |
| 🟢 Q20 | Is it acceptable to name models, datasets and companies to describe them, with no logos? | **Yes** | |
| 🟢 Q21 | Should the name "Paddock" be searched for trademark conflicts before release? | **Yes** | |
| 🟢 Q22 | Is Apache-2.0's default for contributions enough, without a contributor agreement? | **Yes** | |
| 🟢 Q23 | Are Paddock's own dependencies acceptable: all permissive, and none shipped? | **Yes** | |
| 🟢 Q24 | Publish KITScenes' scores now, labelled preliminary, though KIT asks users to wait for a stable release? | **Yes** | |
| 🟢 Q25 | Can Paddock be published as open source without an export-control filing? | **Yes**, once Renesas's export-control office confirms | |
| 🟢 Q26 | Should the website link to Renesas's privacy and legal pages? | **Yes** | |

**With the recommended answers, the release contains** Paddock's code under Apache-2.0,
with complete third-party notices; scores on nine benchmarks, credited and marked with
their data's licences; nothing from AlpaSim; the four nuScenes videos, checked, credited
and hosted by Renesas; no Bench2Drive patches until Bench2Drive allows them; and
downloads that show each licence and ask users to accept terms where the owner has them.

**Decided by** ____________ **on** ____________.

# Part B. The evidence

## 1. The questions in detail

Each question's background, with the section that holds the evidence, and what follows
from each answer. Each opens with its colour, and the recommended answer is marked.

#### Q1. AlpaSim

🔴 **Critical.** AlpaSim's scenes come under NVIDIA's AV NuRec Dataset License Agreement, accepted by
click-through. It treats "any results of benchmarking" as NVIDIA's confidential
information (its §3), forbids derivative works and distributing the data (§4.6), and
allows use only for internal development "using NVIDIA technology" (§1). Paddock holds 15
AlpaSim runs, an AlpaSim column in the README's chart, an AlpaSim page, and four website
videos rendered from NVIDIA's scenes, whose captions give each scene's scores (sections
6.2 and 9).

- **No (recommended):** remove all of these before release, and merge pull request #183,
  which replaces the test files cut from real AlpaSim runs with made-up ones.
- **Yes:** only with NVIDIA's written permission, given before release.
- **Either way:** find out who accepted the agreement, and for which company. It ends 12
  months after the first download, so no later than 7 September 2027, and every copy must
  then be destroyed. Also ask the owner of the Hugging Face video dataset whether the
  AlpaSim videos were ever public; if they were, Legal decides whether to tell NVIDIA.

#### Q2. Non-commercial use

🔴 **Critical.** Eight of the ten benchmarks may be used only non-commercially: through Motional's data
(nuScenes, nuPlan, NAVSIM v1 and v2, and neuroNCAP), KIT's data (KITScenes), HUGSIM's
source data (Waymo's among it), and Bench2Drive's own code. Creative Commons says the
test "turns on the use, not the identity of the reuser". Motional's FAQ names
"educational use and some research use" as free and prices commercial use; KITScenes is
"intended for academic research"; Waymo counts benchmarking for research publication as
non-commercial. Against that, the release also showcases Renesas's work (section 5).

- **Yes (recommended):** the results on those eight benchmarks, and the scores in Q8, can
  be published with credit. Ask Motional to confirm in writing, which covers five of the
  eight; confirmations from KIT and from Thinklab (Bench2Drive) would firm up two more.
- **No:** buy commercial licences (Motional sells one), or publish only comma2k19's
  results, whose data is under MIT.

#### Q3. Who owns the code

🔴 **Critical.** `LICENSE`, the README and `CITATION.cff` name Renesas Electronics Corporation as the
copyright holder. The commits come from 11 authors, writing from `renesas.com`,
`fastcode.ai`, Gmail and GitHub's private addresses, and the repository sits in the
`FastCodeAI` GitHub organisation (section 6.4).

- **Yes (recommended), once the contracts confirm** that every author's work belongs to
  Renesas: employment contracts for Renesas staff, and the contracts that cover everyone
  else, such as the authors writing from `fastcode.ai`.
- **No:** get the rights assigned in writing, or take that work out before release.

#### Q4. The results' licence

🔴 **Critical.** Motional's terms say: "Use of any data derived from the Datasets, which may appear in any
format such as tables and charts, is also subject to these Dataset Terms." The scores on
nuScenes, nuPlan, NAVSIM and neuroNCAP are such derived data. Paddock's own licence,
Apache-2.0, allows commercial use, which Motional's terms do not. THIRD_PARTY_NOTICES.md
already lists the dataset behind each benchmark's results, but nothing marks the results
themselves, and the website's footer names only Apache-2.0 (sections 5 and 9).

- **Yes (recommended):** mark these results CC BY-NC-SA 4.0 under Motional's terms, apart
  from the code, in the repository and on the website, including its `data.json`
  download. Mark the other benchmarks' results with their own data's licences the same
  way.
- **No:** no change.

#### Q5. Dataset excerpts

🔴 **Critical.** The repository holds three small excerpts of non-commercial data: a verbatim subset of
nuScenes' tables for two scenes, with no images and no annotations (UniAD's and VAD's test
data); the list of the 223 KITScenes keyframes Paddock scores; and the names of two
Bench2Drive routes. Each licence allows sharing such parts for non-commercial use with
credit; Bench2Drive's allows exact copies, not changed ones (section 6).

- **Yes (recommended), if Q2 is yes:** keep them, and add to THIRD_PARTY_NOTICES.md the
  three parts it misses: the nuScenes table excerpt, scene 0916, and the KITScenes list
  (Q19).
- **No:** replace the nuScenes excerpt with made-up data, as was done for AlpaSim; have
  each user rebuild the KITScenes list from the data; and drop the route names, which
  Paddock's Bench2Drive results use throughout.

#### Q6. Videos of nuScenes scenes

🔴 **Critical.** The website would show four neuroNCAP videos: renderings of nuScenes scenes 0923, 0278,
0966 and 0108 with another car added, so nuScenes' licence applies (non-commercial, with
credit, CC BY-NC-SA 4.0). They sit in a team member's personal Hugging Face dataset, which
was public on 11 November and is not publicly reachable on 16 November. Nobody has
checked them for identifiable faces or licence plates, and the website credits no dataset
beside them (section 6.2).

- **Yes (recommended), on three conditions:** check them for faces and licence plates and
  blur where needed; move them to a Hugging Face account that Renesas controls; and credit
  nuScenes and its licence beside them.
- **No:** publish the website without its video section.

#### Q7. Bench2Drive patches

🔴 **Critical.** Paddock carries two patches to Bench2Drive's evaluator. Bench2Drive is licensed
CC BY-NC-ND 4.0, which allows sharing exact copies but not changed versions. The edited
file carries Intel's MIT header from CARLA, but the lines the patches change were written
by Bench2Drive, not CARLA (sections 6 and 9).

- **No (recommended), until Bench2Drive takes the fixes or allows them in writing:** hold
  the patches back. Paddock's Bench2Drive runner then needs a small change, because it
  expects a timeout option that one of the patches adds.
- **Yes:** publish them, relying on the file's MIT header.

#### Q8. Models under non-commercial terms

🟡 **Non-critical.** Some models are non-commercial through their weights or their training data. AutoE2E's
weights are CC BY-NC-SA 4.0. Alpamayo 1.5's licence is permissive, but its model card says
"ready for non-commercial use". Qwen-Drive and VisionPilot were trained partly on Waymo
data, and Waymo's licence covers models trained on it. Drive-JEPA and WA-JEPA, and UniAD
and VAD (no results yet), were trained on nuScenes or nuPlan data. Paddock publishes
scores only, never weights (section 8).

- **Yes (recommended), under Q2:** publish their scores. Ask NVIDIA which statement
  governs Alpamayo 1.5, and check these licences again before any product use.
- **No:** drop their scores.

#### Q9. nuScenes sign-in

🟡 **Non-critical.** nuScenes asks every user to register at nuscenes.org before downloading. Paddock's
download recipe uses the addresses that nuscenes.org gives after sign-in, and Motional's
host does not check registration, so a public recipe lets anyone skip the sign-in.
Paddock shows the terms and requires `--accept-terms`, but cannot check registration.
Motional's own tutorials link directly only to nuScenes' small "mini" sample. nuPlan's
addresses are different: NAVSIM's own download scripts publish them (section 6.1).

- **No (recommended):** take the nuScenes addresses out. Users sign in and download from
  nuscenes.org, and Paddock checks the files they downloaded. nuPlan's addresses stay, as
  NAVSIM publishes them.
- **Yes:** keep the recipe as it is.

#### Q10. HUGSIM

🟡 **Non-critical.** HUGSIM's scenes are tagged MIT on Hugging Face, with no licence text, but are rebuilt from
nuScenes, KITTI-360 and Waymo data, all non-commercial. Waymo's licence also lets its data
be shared only with people registered with Waymo who accepted its terms. Paddock's
download recipe shows "MIT" and asks for nothing (sections 6.1 and 9).

- **Yes (recommended):** show the sources' terms in the recipe, require `--accept-terms`,
  and tell users to register with Waymo. Check that whoever made Paddock's two HUGSIM runs
  is registered with Waymo.
- **No:** rely on the MIT tag. Paddock keeps handing Waymo-derived scenes to users who
  never accepted Waymo's terms.

#### Q11. Credit lines

🟡 **Non-critical.** Waymo's licence asks work made with its data to carry an attribution sentence; HUGSIM's
results include scenes rebuilt from Waymo data and carry none. KIT's terms say: "If the
dataset is used in media, a link to our websites (kitscenes.com) is included"; no page
links there (section 6.3).

- **Yes (recommended):** add Waymo's sentence beside HUGSIM's results, and a link to
  kitscenes.com beside KITScenes' results, in the README and on the website.
- **No:** leave them out.

#### Q12. Weights notices

🟡 **Non-critical.** `paddock setup` downloads several models' weights from their publishers' own repositories
without showing their licence. AutoE2E's are non-commercial (CC BY-NC-SA 4.0). Datasets
with terms already need `--accept-terms` (section 6.1).

- **Yes (recommended):** show each model's weights licence when downloading, and require
  `--accept-terms` for non-commercial weights.
- **No:** users keep receiving non-commercial weights without a notice.

#### Q13. Names in the run records

🟡 **Non-critical.** Each run record names the "site" that produced it, and would be published with the
results. Of the 163 records, 55 contain a contributor's user name or first name, 109
contain FastCode's name, and 12 a Renesas host name (section 6.4).

- **No (recommended):** replace the site names with neutral labels in the release, by a
  script, so the change itself is on record.
- **Yes:** publish them as they are, with the named contributors' consent.

#### Q14. Disclaimer

🟡 **Non-critical.** Apache-2.0 disclaims any warranty for the code, but nothing says what the scores are. They
compare other companies' models, NVIDIA's among them, under Paddock's own settings, and
can differ from the publishers' own figures (section 6.4).

- **Yes (recommended):** say in the README and on the website that the scores are
  Paddock's own research measurements, made with Paddock's settings; that they are not
  safety assessments; and that they can differ from the publishers' own figures.
- **No:** no statement.

#### Q15. openpilot's indemnity

🟡 **Non-critical.** openpilot's README says "Any user of this software shall indemnify and hold harmless
Comma.ai, Inc." against claims arising from its use. Paddock does not redistribute
openpilot: each user downloads it, so the indemnity is between that user and comma.ai.
Renesas's own test runs made Renesas such a user (section 8).

- **Yes (recommended):** accept it.
- **No:** drop openpilot and its scores.

#### Q16. Code written with an AI assistant

🟡 **Non-critical.** 407 of the 413 commits on `main` that are not merges carry a "Co-Authored-By: Claude"
line: they were written with Anthropic's AI coding assistant. Whether this affects
copyright, or Renesas's policy, is for Legal (section 6.4).

- **Yes (recommended), subject to Renesas's AI policy:** publish as it is.
- **No:** the release waits for a policy decision.

#### Q17. Linking, not hosting

🟢 **Clear decision.** Paddock names each upstream project and links to the exact version. `paddock setup` and
`paddock data fetch` download code, weights and data on the user's own machine, from the
owner's own host, when the user asks. The repository holds none of them, apart from the
excerpts in Q5, the patches, and the copied lines that THIRD_PARTY_NOTICES.md lists
(sections 6 and 6.1).

- **Yes (recommended).**
- **No:** say which links to remove.

#### Q18. A new repository

🟢 **Clear decision.** The git history holds the real AlpaSim test files and results, internal analysis, and
every contributor's name and e-mail address. The plan is to publish Paddock as a new
repository with no history. Today it is private, in the `FastCodeAI` GitHub organisation
(section 6.4).

- **Yes (recommended):** publish it as a new repository, in a GitHub organisation that
  Renesas controls.
- **No:** the history, with all of the above, becomes public.

#### Q19. Third-party notices

🟢 **Clear decision.** THIRD_PARTY_NOTICES.md lists the files that hold other projects' material (copied and
adapted code, patches, test data and excerpts), each with its source and licence, and the
dataset behind each benchmark's results. It misses four items: the nuScenes table excerpt,
scene 0916 in UniAD's and VAD's test data, the KITScenes keyframe list, and the lines
UniAD's and VAD's tests repeat verbatim (section 6).

- **Yes (recommended):** add the four.
- **No:** say what else it needs.

#### Q20. Names and logos

🟢 **Clear decision.** Models, datasets and companies are named only to describe them, and there is no
third-party logo anywhere. The README and THIRD_PARTY_NOTICES.md say that being listed
does not mean a project endorses Paddock; Motional's terms forbid implying its
endorsement. A model's publisher may embed a "Paddock" badge showing its model's score
(section 6.3).

- **Yes (recommended).**
- **No:** say which names to change.

#### Q21. The name "Paddock"

🟢 **Clear decision.** Nobody has checked "Paddock" for trademark conflicts in software (section 6.4).

- **Yes (recommended):** search before release.
- **No:** release under the name as it is.

#### Q22. Contributions

🟢 **Clear decision.** Under Apache-2.0, contributions are licensed under Apache-2.0 unless the contributor says
otherwise. Paddock has no contributor agreement and no sign-off requirement (section 6.4).

- **Yes (recommended).**
- **No:** add a sign-off requirement or a contributor agreement.

#### Q23. Dependencies

🟢 **Clear decision.** Paddock's own dependencies are all under permissive licences (BSD, MIT and matplotlib's
own); none is copyleft, and Paddock ships none of them. Each model's environment installs
its own packages, such as PyTorch, which this review did not cover (section 6.4).

- **Yes (recommended).**
- **No:** say what to check further.

#### Q24. KITScenes' scores now

🟢 **Clear decision.** KITScenes' dataset card says: "For final benchmark reporting, please wait for a more
stable public release." We read this as a request, not one of KIT's terms. Nothing in
Paddock labels its KITScenes scores as preliminary (section 9).

- **Yes (recommended):** publish them now, labelled preliminary.
- **No:** hold KITScenes' results until KIT's stable release.

#### Q25. Export control

🟢 **Clear decision.** Not checked by us. Paddock is evaluation software, implements no encryption, and would be
published as open source (section 6.4).

- **Yes (recommended), once Renesas's export-control office confirms.**
- **No:** classify it before release.

#### Q26. Website legal notices

🟢 **Clear decision.** The website sets no cookies, runs no analytics, and loads only the videos and their
stills from another host, Hugging Face. Its footer gives Paddock's licence, but it has no
privacy notice or terms of use (sections 6.2 and 6.3).

- **Yes (recommended):** add links to Renesas's privacy and legal pages to the footer.
- **No:** publish it as it is.

## 2. What is released, and our recommendation for each part

| Part of the release | Release? | Why | How to make it a clear yes |
|---|---|---|---|
| Paddock's code | 🟡 Yes, after the clean-ups in section 6 | Upstream code is linked, not copied. No upstream code licence is copyleft. The few copied lines are under MIT or Apache-2.0, except in two Bench2Drive patches. | Q3, Q7, Q19 |
| Download recipes and `paddock setup` | 🟡 Yes, after three changes | Each download comes from its owner's own host. But nuScenes' recipe skips Motional's sign-in, HUGSIM's shows only "MIT", and model weights download without their licence. | Q9, Q10, Q12 |
| Results on comma2k19 | 🟢 Yes | The data is MIT. | Credit comma.ai |
| Results on nuScenes, nuPlan, NAVSIM v1 and v2, and neuroNCAP | 🟡 Yes, if non-commercial (Q2) | Motional's data is for non-commercial use, and its terms cover derived "tables and charts". Its FAQ names "educational use and some research use" as free, and prices commercial use. | Credit, and a CC BY-NC-SA 4.0 notice on these results (Q4). For a firm yes, Motional's written confirmation. |
| Results on KITScenes | 🟡 Yes, if non-commercial (Q2) | CC BY-NC 4.0; the data is "intended for academic research". KIT asks users to wait for a more stable release before final benchmark reporting. | Credit with a link to kitscenes.com (Q11), labelled preliminary (Q24). For a firm yes, KIT's confirmation. |
| Results on HUGSIM | 🟡 Yes, if non-commercial (Q2) | Its scenes are rebuilt from nuScenes, KITTI-360 and Waymo data, all non-commercial. | Credit, including Waymo's sentence (Q10, Q11) |
| Results on Bench2Drive | 🟡 Yes, if non-commercial (Q2) | The benchmark's code is CC BY-NC-ND 4.0. | Credit. For a firm yes, Thinklab's confirmation. |
| Scores of models under non-commercial terms | 🟡 Yes, if non-commercial (Q2) | AutoE2E's weights are non-commercial, and Alpamayo 1.5's card says the same. Waymo's licence covers models trained on its data, which Qwen-Drive and VisionPilot partly were. Paddock publishes scores only. | Q8 |
| Results, page, chart column, videos and captions on AlpaSim | 🔴 No | NVIDIA's agreement treats "any results of benchmarking" as confidential, forbids sharing derived works, and limits use to internal development with NVIDIA technology. | NVIDIA's written permission (Q1) |
| Run records | 🟡 Yes, with neutral site names | Their site names identify contributors, FastCode and Renesas hosts. | Q13 |
| Videos of nuScenes scenes on the website | 🟡 Yes, if non-commercial (Q2) | Renderings of nuScenes scenes: non-commercial, with credit, under nuScenes' licence. | Q6 |
| Excerpts of nuScenes, KITScenes and Bench2Drive data | 🟡 Yes, if non-commercial (Q2) | Small and credited, under their datasets' licences. | Q5, Q19 |

## 3. The licences, grouped

Each group carries the colour of what it means for material under it in this release.

| Group | What it allows | Where it applies |
|---|---|---|
| 🟢 **A. Permissive**: MIT, Apache-2.0, BSD-3-Clause, CC0, CC BY, OpenMDW-1.1 | Any use, commercial included. Keep the licence and copyright notices. | Paddock itself. The code of all eleven models, and of NAVSIM, KITScenes' library, comma2k19, neuroNCAP, AlpaSim, HUGSIM, neurad-studio and CARLA. The weights of METEOR, OpenPilot, Qwen-Drive, UniAD, VisionPilot, Alpamayo 2 Edge, WA-JEPA, Drive-JEPA and Alpamayo 1.5 (the last three also in G). The data of comma2k19, PandaSet, 3DRealCar and CARLA's assets. |
| 🟢 **B. Commercial use, with conditions**: NVIDIA Open Model License | Commercial use, if NVIDIA's Trustworthy AI terms are followed and its safety guardrails are kept. Attribution when redistributing. | Cosmos-Reason2-8B, the model Alpamayo 1.5 is built on |
| 🟡 **C. Non-commercial**: CC BY-NC 4.0, CC BY-NC-SA 3.0 and 4.0, Motional's Dataset Terms, Waymo's licence | Non-commercial use only, with credit. ShareAlike (SA) licences also require adapted material to carry the same licence. | Data: nuScenes, nuPlan, OpenScene (NAVSIM), KITScenes, KITTI-360, Waymo, OpenLane. Weights: AutoE2E. |
| 🟡 **D. Non-commercial, no derivatives**: CC BY-NC-ND 4.0 | Non-commercial use, with credit. Exact copies may be shared; changed versions may not. | Bench2Drive's code and route lists; Bench2DriveZoo, which UniAD's and VAD's environment is built from |
| 🔴 **E. Internal use only, results confidential**: NVIDIA AV NuRec Dataset License Agreement | Internal development of autonomous vehicles using NVIDIA technology. No disclosure of benchmark results. Expires 12 months after download. | AlpaSim's scenes |
| 🟢 **F. Permission needed for commercial use**: sunnypilot's custom licence | Viewing and modifying. Commercial, for-profit or closed-source use needs written permission. | sunnypilot, which Paddock never used, and removed on 12 November (pull request #181) |
| 🟡 **G. Unclear or conflicting** | Needs the publisher's confirmation | Alpamayo 1.5 (permissive licence, but its card says non-commercial). HUGSIM's scenes (MIT tag over non-commercial sources). neuroNCAP's renderer files (no licence stated). Drive-JEPA's and WA-JEPA's weights (licence tag only, no text). VAD's weights (no licence stated). |

## 4. At a glance

Each row carries the colour of that model's or benchmark's results in the release.

### Models

| Model (publisher) | Code | Weights | Commercial use of the weights |
|---|---|---|---|
| 🟡 Alpamayo 1.5 (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpamayo1.5/blob/7a8f1c781a826f09be53e1e211f26e947ec18019/LICENSE) | [OpenMDW-1.1](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | licence allows it; card says non-commercial |
| 🟢 Alpamayo 2 Edge (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpamayo2-edge) | [OpenMDW-1.1](https://huggingface.co/nvidia/Alpamayo2-Edge) | licence allows it; the card version Paddock pinned said "to be confirmed" |
| 🟡 AutoE2E (Autoware Foundation) | [Apache-2.0](https://github.com/autowarefoundation/auto_e2e/blob/40a75cf8a26b60f6b094125e3924fe0bdb427b02/LICENSE) | [CC BY-NC-SA 4.0](https://huggingface.co/AutowareFoundation/auto_e2e#license) | no |
| 🟡 Drive-JEPA (Linhan Wang et al.) | [Apache-2.0](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/LICENSE) | [CC0-1.0](https://huggingface.co/datasets/LinhanWang/Drive-JEPA) | tag allows it; trained on non-commercial data |
| 🟢 METEOR (TIER IV) | [Apache-2.0](https://github.com/tier4/METEOR/blob/dc193a81e9959de57b751e269127ae9b67e3f300/LICENSE) | [Apache-2.0](https://huggingface.co/AutowareFoundation/meteor) | yes |
| 🟡 OpenPilot (comma.ai) | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE) | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE), in the code repository | yes; README adds an indemnity |
| 🟡 Qwen-Drive 1.0 (Qwen team, Alibaba) | [Apache-2.0](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/LICENSE) | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) | licence allows it; trained partly on Waymo data |
| 🟡 UniAD (OpenDriveLab) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/LICENSE) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/README.md#license), as UniAD's "assets" | licence allows it; trained on nuScenes, non-commercial data |
| 🟡 VAD (HUST Vision Lab) | [Apache-2.0](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/LICENSE) | not stated; a Google Drive link in [VAD's README](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/README.md) | not stated; trained on nuScenes, non-commercial data |
| 🟡 VisionPilot (Autoware Foundation) | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE) | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE), in the code repository | licence allows it; trained partly on Waymo-derived data |
| 🟡 WA-JEPA (AFARI Research) | [Apache-2.0](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/LICENSE) | [Apache-2.0](https://huggingface.co/AFARI-Research/WA-JEPA) | tag allows it; trained on non-commercial data |
| 🟢 Constant velocity (Renesas) | Apache-2.0 (Paddock) | none | not applicable |

### Benchmarks

| Benchmark (publisher) | Code | Data or scenes | Commercial use of the data |
|---|---|---|---|
| 🟡 nuScenes (Motional) | Paddock's own | [CC BY-NC-SA 4.0 + Motional terms](https://www.nuscenes.org/terms-of-use) | no; [paid licence](https://www.nuscenes.org/terms-of-use-commercial) |
| 🟡 nuPlan (Motional) | Paddock's own | [CC BY-NC-SA 4.0 + Motional terms](https://www.nuscenes.org/terms-of-use) | no; [paid licence](https://www.nuscenes.org/terms-of-use-commercial) |
| 🟡 NAVSIM v1 and v2 (autonomousvision) | [Apache-2.0](https://github.com/autonomousvision/navsim) | OpenScene: [CC BY-NC-SA 4.0 + nuPlan terms](https://github.com/OpenDriveLab/OpenScene) | no |
| 🟡 KITScenes (KIT) | [Apache-2.0](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/LICENSE) | [CC BY-NC 4.0 + KIT terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal) | no |
| 🟢 comma2k19 (comma.ai) | [MIT](https://github.com/commaai/comma2k19/blob/4c7f1a6e1957745beadc1def0e7225f559b09a2a/LICENSE) | [MIT](https://huggingface.co/datasets/commaai/comma2k19) | yes |
| 🟡 Bench2Drive (Thinklab, SJTU) | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE), with MIT parts | CARLA: [MIT code, CC-BY assets](https://github.com/carla-simulator/carla/blob/0.9.15/README.md#licenses) | no, for the benchmark code |
| 🟡 neuroNCAP (Ljungbergh, Tonderski et al.) | [MIT](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/LICENSE) | nuScenes, as above; renderer files: none stated | no |
| 🔴 AlpaSim (NVIDIA) | [Apache-2.0](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/LICENSE) | [NVIDIA AV NuRec agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) | internal development only; results confidential |
| 🟡 HUGSIM (X-D Lab, Zhejiang University) | [MIT](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/LICENSE) | [tagged MIT](https://huggingface.co/datasets/XDimLab/HUGSIM), built from nuScenes, KITTI-360, Waymo and PandaSet | mostly no |

## 5. Is credit enough, or is permission needed?

Every licence here requires credit. For permissive licences, credit is all that is
needed. For non-commercial licences, credit is required but not sufficient: the use itself
must also be non-commercial, and no amount of credit turns a commercial use into a
non-commercial one. Two items need permission however they are credited: the AlpaSim
results, and, on one reading, the Bench2Drive patches.

| Licence group (section 3) | Is credit enough? | When permission is needed |
|---|---|---|
| 🟢 A. Permissive | Yes. Keep the licence and copyright notices. | Never |
| 🟢 B. NVIDIA Open Model License | Yes, for running the model | Only to redistribute it, which we do not do |
| 🟡 C. Non-commercial | Only if the use is non-commercial (Q2) | If Legal judges the use commercial, or wants certainty |
| 🟡 D. Non-commercial, no derivatives | For running Bench2Drive and publishing scores: as for C. For the two patches: possibly not (Q7). | To share the patches, unless the file's MIT header covers them |
| 🔴 E. NVIDIA NuRec agreement | No | Always, to publish AlpaSim results |
| 🟢 F. sunnypilot | No | For any commercial use. Paddock never used it, and removed it on 12 November (pull request #181). |
| 🟡 G. Unclear or conflicting | Probably, but not confirmed | A short confirmation from the publisher removes the doubt |

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
  Weighing these is Q2. Creative Commons adds that "licensees are always free to
  contact licensors to ask permission".

## 6. What the licences allow in the Paddock code release

The colour in the second column is each item's place in the release.

| What the release does | Allowed? | Condition or action |
|---|---|---|
| Names each upstream project and links to the exact version; `paddock setup` downloads it on the user's machine | 🟢 Yes, for all | Nothing is redistributed, and each user accepts the upstream terms. sunnypilot is removed (pull request #181, merged on 12 November). |
| Paddock's own code, which starts upstream code and calls it while running | 🟢 Yes | It is Renesas's own work. No upstream code licence is copyleft (none requires code that uses it to take the same licence), so Paddock can stay Apache-2.0. |
| Patches to neuro-ncap (MIT), Drive-JEPA (Apache-2.0) and HUGSIM (MIT) | 🟢 Yes | Keep the upstream notices and say what changed; each patch's header already explains its change. |
| Two patches to Bench2Drive | 🔴 Unclear (Q7) | They change lines that Bench2Drive wrote; those lines are not in [CARLA's original](https://github.com/carla-simulator/leaderboard/blob/leaderboard-2.0/leaderboard/leaderboard_evaluator.py). But the file is headed "Copyright (c) 2018-2019 Intel Corporation … This work is licensed under the terms of the MIT license." To make it a yes: offer both fixes to Bench2Drive, or get Thinklab's permission. Holding the patches back needs a small code change, because Paddock's Bench2Drive runner expects the timeout option that one of them adds. |
| Code adapted from AlpaSim and ported from VisionPilot (both Apache-2.0), and two reimplementations: TCP's PID controller (Apache-2.0) and openpilot's camera calibration (MIT) | 🟢 Yes | Each file names its source, and THIRD_PARTY_NOTICES.md lists them (pull request #182, merged on 12 November). The PID controller's own description called it "vendored" from Bench2DriveZoo, which is CC BY-NC-ND 4.0; the code is a rewrite, and now says so. |
| nuScenes' validation scene list, and 20 neuroNCAP test scenarios with nuScenes object IDs and positions | 🟢 Yes | Both copy permissively licensed upstream files. The scene list is exactly the one Motional publishes in its nuScenes devkit (Apache-2.0), and the scenarios are neuro-ncap's own (MIT) with only the time step changed. THIRD_PARTY_NOTICES.md credits both (pull request #182). |
| AlpaSim scene list (IDs and file paths) | 🟢 Yes | The same IDs are published in AlpaSim's own Apache-2.0 repository. |
| Test data for UniAD and VAD, added on 12 November: a verbatim subset of nuScenes' tables for scenes 0103 and 0916 (sample records, calibration, ego poses and CAN-bus messages; no images and no annotations), and files holding both models' inputs and plans on those scenes | 🟡 Yes, for non-commercial use, with credit, under CC BY-NC-SA 4.0 | THIRD_PARTY_NOTICES.md lists the scene-0103 data, but neither the table excerpt (`uniad/tests/fixtures/nuscenes_scene_tables.json.gz`) nor scene 0916; add both. If publishing nuScenes excerpts is not allowed, replace them with made-up data, as for AlpaSim. |
| Code repeated verbatim from UniAD and VAD in their tests (`uniad/tests/infos_reference.py`) | 🟡 Yes, with notice | Apache-2.0. Not yet in THIRD_PARTY_NOTICES.md; add it. |
| The list of the 223 KITScenes keyframes Paddock scores (`kitscenes/sample_list.csv`, added on 12 November): scene id, keyframe index, split, and a driving command Paddock derives from each recorded drive | 🟡 Yes, for non-commercial use, with credit | Not yet in THIRD_PARTY_NOTICES.md; add it, with KITScenes' licence. |
| One patch to UniAD (Apache-2.0), and two to OpenMMLab code that Bench2DriveZoo carries | 🟢 Yes | Each Bench2DriveZoo patch changes one line of OpenMMLab's own code, which is Apache-2.0; both lines are in mmdet3d 0.17.1 and mmcv 1.4.0 as OpenMMLab published them, so neither is Bench2DriveZoo's own. THIRD_PARTY_NOTICES.md lists both. |
| Bench2DriveZoo (CC BY-NC-ND 4.0), which `paddock setup` downloads to build UniAD's and VAD's environment | 🟡 Yes, if running it is non-commercial use | Nothing of it is in the repository. The build copies two of its files, which per UniAD's page are OpenMMLab's own mmdet3d 0.17.1 files (Apache-2.0). |
| Test files cut from two real AlpaSim runs: scores, and one model's outputs on one scene | 🔴 No | Replaced with made-up data in pull request #183, including which scene passes, and the model's outputs by a made-up drive. That pull request is not merged yet, so `main` still holds the real files. |
| Two route names from Bench2Drive's route list: each route's id and town, and its scenario's name and type, 3% of each entry, with no waypoints or weather | 🟡 Yes, for non-commercial use, with credit and a licence notice | Kept, because Paddock's results use the same names throughout. The test's README and THIRD_PARTY_NOTICES.md now say what the file holds and Bench2Drive's licence (pull requests #182 and #183). |
| Results: scores, tables and charts | 🟡 See section 2 | |
| A website built from the results on `main` (GitHub Pages), added after this review began | 🟡 See section 2 | It is built only once the repository is public, and shows the results `main` holds, so the AlpaSim decision applies to it too. It would show eight videos: four neuroNCAP clips, renderings of nuScenes scenes, and, since 12 November, four AlpaSim clips, renderings of NVIDIA's scenes, whose captions give each scene's scores. They are hosted on a team member's Hugging Face dataset, `naren-fastcode/paddock-videos`, which was public on 11 November and is not publicly reachable on 16 November. The page also offers every result as a `data.json` download. The repository's own note says whether Renesas may show the clips "is still to be confirmed with Renesas legal" (Q6). |
| `paddock data fetch`, which downloads the datasets and CARLA, added after this review began | 🟡 Yes, after two changes | It downloads from each owner's own host, names each dataset's licence, and refuses a dataset that has terms until the user passes `--accept-terms`. But nuScenes' entry uses the addresses nuscenes.org gives only after sign-in (Q9), and HUGSIM's gives its licence as MIT, without its source datasets' terms (Q10). |
| Names of models, datasets and publishers in documents and charts | 🟢 Yes, to describe them | Do not imply endorsement. Motional's terms forbid it, and Paddock's README already says that being listed does not mean a project endorses Paddock. |

**Clean-ups.** What is done, what is waiting, and what the recommended answers call for.

- **Done**, merged on 12 November. sunnypilot is removed from the upstream list (pull
  request #181). THIRD_PARTY_NOTICES.md lists the files that hold other projects'
  material and the dataset behind each benchmark's results; the VisionPilot port names
  its source; and the model comparison page gives each weights licence as its publisher
  states it (pull request #182).
- **Waiting to merge:** the made-up AlpaSim test files (pull request #183).
- **To do, with the recommended answers**, each marked with its question's colour:
  1. 🔴 Remove everything from AlpaSim: its runs, chart column, page, videos and captions
     (Q1).
  2. 🔴 Mark the results with their data's licences, in the repository and on the website
     (Q4).
  3. 🔴 Add the four missing items to THIRD_PARTY_NOTICES.md (Q5, Q19).
  4. 🔴 Check the nuScenes videos for faces and licence plates, move them to an account
     Renesas controls, and credit nuScenes beside them (Q6).
  5. 🔴 Hold back the two Bench2Drive patches, and let the runner work without them,
     unless Bench2Drive takes the fixes (Q7).
  6. 🟡 Take the nuScenes sign-in addresses out of the download recipe (Q9).
  7. 🟡 Show the terms of HUGSIM's sources in its download recipe, with `--accept-terms`
     (Q10).
  8. 🟡 Add Waymo's attribution sentence and a link to kitscenes.com (Q11).
  9. 🟡 Show each model's weights licence in `paddock setup`, with `--accept-terms` for
     non-commercial weights (Q12).
  10. 🟡 Replace the run records' site names with neutral labels in the release (Q13).
  11. 🟡 Say what the scores are, and are not, in the README and on the website (Q14).
  12. 🟢 Publish as a new repository, without history, in a GitHub organisation Renesas
      controls (Q18).
  13. 🟢 Search the name "Paddock" for trademark conflicts (Q21).
  14. 🟢 Label KITScenes' scores as preliminary (Q24).
  15. 🟢 Link Renesas's privacy and legal pages from the website's footer (Q26).

### 6.1 Downloads

Paddock hosts nothing. Every download runs on the user's own machine, from the owner's own
host, when the user asks for it. Checked on 16 November against `main`.

| What | From | Licence shown to the user | Terms check |
|---|---|---|---|
| 🟢 Upstream code, 20 repositories, each at a pinned commit (`paddock setup`) | GitHub | No | None; each repository's own licence applies |
| 🟡 nuScenes (`paddock data fetch`) | Motional's own download host, at the addresses nuscenes.org gives only after sign-in | Yes: "CC BY-NC-SA 4.0, non-commercial use only; nuScenes also asks every user to register at nuscenes.org" | `--accept-terms`. The host does not check registration, so publishing these addresses lets a user skip Motional's sign-in. Motional's own tutorials link directly only to nuScenes' small "mini" sample (Q9). |
| 🟢 nuPlan, and NAVSIM's maps | Motional's own download bucket, at the addresses NAVSIM's own download scripts publish | Yes: CC BY-NC-SA 4.0, non-commercial use only | `--accept-terms` |
| 🟢 NAVSIM's scenes (OpenScene) | Hugging Face, `OpenDriveLab/OpenScene` | Yes: CC BY-NC-SA 4.0, non-commercial use only | `--accept-terms` |
| 🟢 KITScenes | Hugging Face, gated: the user's own login, which must have accepted KIT's terms | Yes: CC BY-NC 4.0, non-commercial use only | KIT's gate, and `--accept-terms` |
| 🟢 neuroNCAP's renderer files | Dropbox, the link in neuro-ncap's own download script | Yes: nuScenes' terms, which the renderers inherit | `--accept-terms` |
| 🟡 HUGSIM's scenes | Hugging Face, `XDimLab/HUGSIM` | Yes, as "MIT" | None. The scenes include ones rebuilt from Waymo data, and Waymo's licence lets data be shared only with people registered with Waymo who agreed to its terms (Q10). |
| 🟢 comma2k19 | Hugging Face, `commaai/comma2k19` | Yes: MIT | None needed |
| 🟢 CARLA 0.9.15, for Bench2Drive | CARLA's own release bucket | Yes: MIT code, CC BY assets, Unreal Engine's own licence | None |
| 🟡 Weights of Alpamayo 2 Edge, AutoE2E, METEOR and WA-JEPA (`paddock setup --model`) | Hugging Face, the publishers' own repositories | No | None. AutoE2E's weights are non-commercial (CC BY-NC-SA 4.0) and download without a notice (Q12). |
| 🟢 Weights of openpilot and VisionPilot | Inside their code repositories (git LFS for openpilot) | No | None; MIT and Apache-2.0 |
| 🟢 Weights of Alpamayo 1.5, Drive-JEPA, Qwen-Drive, UniAD and VAD | The publisher's own link, which each model's page gives; the user fetches them | On the model pages that link to them | Alpamayo 1.5's model card says non-commercial; VAD's weights state no licence |
| 🟢 Python packages | PyPI and PyTorch's package index | No | Paddock's own dependencies are all permissive (section 6.4) |

### 6.2 Videos and images

**In the repository**, every image is Paddock's own: the logo (`paddock.svg`), the social
preview card (`social-preview.png`, text only), and two charts drawn from the results
(`models.svg`, and `support-matrix.svg`, whose AlpaSim column shows AlpaSim scores). There
are no dataset images and no videos.

**On the website**, which is built only once the repository is public, `tools/site/media.toml`
lists eight videos with a still image each. They are hosted outside git, on a team member's
personal Hugging Face account, as `naren-fastcode/paddock-videos`:

| Videos | What they show | Licence position | What to know |
|---|---|---|---|
| 🟡 Four neuroNCAP clips | Renderings of nuScenes scenes 0923, 0278, 0966 and 0108, rebuilt by neuroNCAP's renderers, with another car added | nuScenes' terms: CC BY-NC-SA 4.0, non-commercial, with credit. The dataset's card was tagged CC BY-NC-SA 4.0 and credited nuScenes on 11 November. | The website page itself credits no dataset beside them. Whether they show identifiable faces or licence plates has not been checked. |
| 🔴 Four AlpaSim clips, added on 12 November | Renderings of NVIDIA's scenes bae463d3, 5f064b02, 8023dce5 and 04749bb9; each caption gives the scene's scores | NVIDIA's agreement forbids distributing derived works of the scenes (§4.6) and treats benchmark results as confidential (§3) | Not publishable without NVIDIA's written permission. Whether they were ever publicly reachable has not been checked: ask the dataset's owner. |

The dataset was public and ungated on 11 November. On 16 November Hugging Face no longer
serves it to anyone not logged in, so it is private or deleted; `main`'s website list still
points to it. The website also offers every result, AlpaSim's included, as a `data.json`
download.

### 6.3 References and names

- 🟢 **References.** The README's tables link each model's code, with its licence as GitHub
  reports it, and its paper; and each benchmark's own site and paper. All eleven code
  licences in the README match the projects' own LICENSE files.
- 🟢 **Citations.** The README asks users to cite each model and benchmark they run, and
  `CITATION.cff` lets them cite Paddock as Renesas's work.
- 🟢 **Names, no logos (Q20).** Models, datasets and companies are named only to describe them.
  There is no third-party logo in the README, the documents or the website. The website
  loads no outside fonts or scripts; the only things it loads from elsewhere are the videos
  and their stills (section 6.2). The README and THIRD_PARTY_NOTICES.md say that being
  listed does not mean a project endorses Paddock; Motional's terms forbid implying its
  endorsement or using its name or logo without consent.
- 🟢 **Badges.** The website lets a model's publisher embed a "Paddock" badge showing the
  model's score: Paddock's own name, beside Paddock's own measurement.
- 🟡 **Attribution lines the datasets ask for (Q11).** KITScenes' paper is cited in the README, but
  KIT's terms also say: "If the dataset is used in media, a link to our websites
  (kitscenes.com) is included." No page in the repository links to kitscenes.com. Waymo's
  licence asks work made with its data to carry an attribution sentence; none appears,
  and the two HUGSIM runs include scenes rebuilt from Waymo data.
- 🟢 **"Paddock ships no weights and no datasets."** The README says this, and the website
  says Paddock ships none of the models, benchmarks and datasets. It is true of weights and
  of whole datasets; the repository does hold the small dataset excerpts in section 6.

### 6.4 Release hygiene

- 🔴 **Who owns the code (Q3).** `LICENSE` and the README name Renesas Electronics Corporation as
  the copyright holder. The commits on `main` come from 11 authors, writing from
  `renesas.com`, `fastcode.ai`, Gmail and GitHub's private no-reply addresses.
- 🟡 **Names in the run records (Q13).** Every run record names the "site" that produced it, and
  these names would be published with the results. Of the 163 records, 55 contain a
  contributor's user name or first name (such as `hpcai-skodate` and
  `fastcode-navvrat-b2d-pooled`), 109 contain the name of FastCode, whose staff
  contributed to Paddock, and 12 a Renesas host name (such as `renesas-sjc`).
- 🟡 **Disclaimers (Q14).** Apache-2.0 disclaims any warranty for the code. Nothing says that the
  scores are Paddock's own research measurements, not safety assessments, or that they can
  differ from the publishers' own figures. The README's only statement of this kind is
  that being listed does not mean a project endorses Paddock.
- 🟢 **Export control (Q25)** has not been checked. Paddock implements no encryption.
- 🟡 **Code written with an AI assistant (Q16).** 407 of the 413 commits on `main` that are not
  merges carry a "Co-Authored-By: Claude" line: they were written with Anthropic's AI
  coding assistant.
- 🟢 **Secrets.** No access token or private key is in the files or the history (searched for
  Hugging Face, GitHub, AWS, Slack, OpenAI and Google keys, and private-key blocks).
- 🟢 **Paddock's own dependencies (Q23)** are all under permissive licences: BSD, MIT, and
  matplotlib's own. None is copyleft, and Paddock ships none of them; users install them.
  Each model's environment installs its own packages, such as PyTorch, which this review
  did not cover.
- 🟢 **Git history (Q18).** It holds the real AlpaSim test files and results, and internal
  analysis. The plan is to publish Paddock as a new repository without history.
- 🟢 **Contributions (Q22).** Apache-2.0 makes contributions Apache-2.0 unless stated otherwise.
  There is no contributor agreement and no sign-off requirement.
- 🟢 **The name "Paddock" (Q21)** has not been checked for trademark conflicts.

## 7. How others report results on these benchmarks

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

## 8. Models: details

Each LICENSE link points to the exact version of the code that Paddock uses. Each
heading carries the colour of the model's row in section 4.

#### 🟡 Alpamayo 1.5 (NVIDIA)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpamayo1.5/blob/7a8f1c781a826f09be53e1e211f26e947ec18019/LICENSE) | |
| Weights: [nvidia/Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | Grants rights "without restriction", and "does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs". Also: "YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF". |
| Model card | | "This model is ready for non-commercial use. Commercial licensing available upon request." (Q8) |
| Built on | [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B): [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). That model is built on [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): Apache-2.0. | Commercial use allowed. Use must follow NVIDIA's Trustworthy AI terms. Attribution duties apply only to sharing Cosmos models or building new ones from them. Paddock loads only Cosmos-Reason2-8B's configuration files. |
| Trained on | NVIDIA's own driving data, plus public sets built on nuScenes and NAVSIM | Q8 |
| In the release | Paddock's adapter, including one file adapted from AlpaSim that keeps NVIDIA's notice. Older neuroNCAP runs, imported from before Paddock existed, are stored under the name `alpamayo` and do not record which checkpoint ran; NVIDIA's earlier [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) has the same OpenMDW-1.1 tag. | |

#### 🟢 Alpamayo 2 Edge (NVIDIA)

Added to Paddock after this review began.

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpamayo2-edge) | |
| Weights: [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge), not gated | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | The current model card states OpenMDW-1.1 and, unlike Alpamayo 1.5's, does not call the weights non-commercial. The earlier card version Paddock pins said the licence was still to be confirmed. |
| In the release | Paddock's adapter; results on eight benchmarks: NAVSIM v1 and v2, nuScenes, KITScenes, comma2k19, neuroNCAP, Bench2Drive and AlpaSim | |

#### 🟡 AutoE2E (Autoware Foundation)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/autowarefoundation/auto_e2e/blob/40a75cf8a26b60f6b094125e3924fe0bdb427b02/LICENSE) | |
| Weights: [AutowareFoundation/auto_e2e](https://huggingface.co/AutowareFoundation/auto_e2e#license), checkpoint `kitscenes-epoch-5` | CC BY-NC-SA 4.0 | "commercial use is not covered. It needs permission from the upstream rights holders, including a commercial license from Motional for nuPlan and nuScenes". The card overrides the licence "other" recorded in the files of the v1.0 release. |
| Trained on | nuPlan, then fine-tuned on KITScenes, with a camera encoder trained on nuScenes | The reason the card chose CC BY-NC-SA 4.0 |
| In the release | Paddock's adapter; scores only | |

#### 🟡 Drive-JEPA (Linhan Wang et al.)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/LICENSE) | Includes two copies of NAVSIM's scoring code (Apache-2.0), which Paddock uses to run NAVSIM, and Meta's V-JEPA 2 code (MIT). |
| Weights: [LinhanWang/Drive-JEPA](https://huggingface.co/datasets/LinhanWang/Drive-JEPA) | CC0-1.0, tag only | A public-domain dedication, with no licence text. The same repository holds scoring caches built from nuPlan logs. |
| Trained on | NAVSIM and OpenScene, both from nuPlan | Q8 |
| In the release | Paddock's adapter; three patches (Apache-2.0 code) | The repository also carries a [copy of Motional's terms](https://github.com/linhanwang/Drive-JEPA/blob/e21f47410b4d26b61f05f9bd23e169c0390cae2a/download/LICENSE), which this document quotes. |

#### 🟢 METEOR (TIER IV)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/tier4/METEOR/blob/dc193a81e9959de57b751e269127ae9b67e3f300/LICENSE) | |
| Weights: [AutowareFoundation/meteor](https://huggingface.co/AutowareFoundation/meteor) | Apache-2.0 | |
| Trained on | TIER IV's own recordings in Japan, plus re-renderings made with NVIDIA Cosmos Transfer | |
| In the release | Paddock's adapter | |

#### 🟡 OpenPilot (comma.ai)

| Part | Licence | Terms that matter |
|---|---|---|
| Code and weights | [MIT](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/LICENSE) | The weights are files inside the code repository. |
| [README](https://github.com/commaai/openpilot/blob/555f48c5d28709f039b79f3f6105e51305edd4b5/README.md) | | "Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys' fees and costs) which arise out of, relate to or result from any use of this software by user." Also: "THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT." (Q15) |
| Trained on | comma.ai's own fleet data, which is not public | |
| In the release | Paddock's adapter | |

#### 🟡 Qwen-Drive 1.0 (Qwen team, Alibaba)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/QwenLM/Qwen-Drive-1.0/blob/28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0/LICENSE) | "Copyright 2026 Alibaba Group Holding Limited" |
| Weights: [Qwen/Qwen-Drive-1.0-4B](https://huggingface.co/Qwen/Qwen-Drive-1.0-4B) | Apache-2.0 | Built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0 |
| Trained on | nuScenes, OpenScene, Waymo's WOD-E2E, NVIDIA's PhysicalAI-AV, and other public driving sets | Q8. Waymo's licence names models trained on its data. |
| In the release | Paddock's adapter | |

#### 🟡 UniAD (OpenDriveLab)

Added to Paddock on 12 November. No results yet.

| Part | Licence | Terms that matter |
|---|---|---|
| Code: [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab) | [Apache-2.0](https://github.com/OpenDriveLab/UniAD/blob/532fc330151758c5e345aef74d2a1bf1042e50ab/LICENSE) | Its README: "All assets and code are under the Apache 2.0 license unless specified otherwise." |
| Weights: `uniad_base_e2e.pth`, from UniAD's own GitHub release v1.0.1 | Apache-2.0, as UniAD's "assets" | Trained on nuScenes `train`, non-commercial data; its first stage starts from BEVFormer's nuScenes weights. |
| Build dependency: [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db) | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2DriveZoo/blob/498c1f799dd90faf840dedb3f0d3234ec2e567db/LICENSE) | `paddock setup` downloads it to build the environment UniAD and VAD share; the build copies two of its files, OpenMMLab's own. |
| In the release | Paddock's adapter; one patch to UniAD and two to Bench2DriveZoo; test data from two nuScenes scenes (section 6) | |

#### 🟡 VAD (HUST Vision Lab)

Added to Paddock on 12 November. No results yet.

| Part | Licence | Terms that matter |
|---|---|---|
| Code: [hustvl/VAD](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8) | [Apache-2.0](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/LICENSE) | Its README: "All code in this repository is under the Apache License 2.0." It says nothing about the weights. |
| Weights: VAD-Base, from a Google Drive link in [VAD's README](https://github.com/hustvl/VAD/blob/1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8/README.md) | Not stated | Trained on nuScenes `train`, non-commercial data. |
| In the release | Paddock's adapter, which shares UniAD's environment and test data | |

#### 🟡 VisionPilot (Autoware Foundation)

| Part | Licence | Terms that matter |
|---|---|---|
| Code and weights | [Apache-2.0](https://github.com/autowarefoundation/vision_pilot/blob/e93eceaaa70ee4e7b55fd4ecc04c023c163c8098/LICENSE) | The weights are files inside the code repository, whose README offers "the complete codebase as free and fully open-source, including AI model weights". |
| Trained on | Partly [OpenLane](https://github.com/OpenDriveLab/OpenLane): CC BY-NC-SA 4.0 plus Waymo's non-commercial licence | Q8 |
| In the release | Paddock's adapter, which ports part of VisionPilot's C++ planner and names its source (pull request #182). | |

#### 🟡 WA-JEPA (AFARI Research)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/AFARI-Research/WA-JEPA/blob/bec29660f5ea46ac73db8e2d0c33c8d1a72c23ad/LICENSE) | Includes Meta's V-JEPA 2 code (MIT, with an Apache-2.0 part). |
| Weights: [AFARI-Research/WA-JEPA](https://huggingface.co/AFARI-Research/WA-JEPA) | Apache-2.0, tag only | The model card is empty. |
| Trained on | nuPlan video, then NAVSIM, according to Paddock's own notes | Q8 |
| In the release | Paddock's adapter | |

#### 🟢 Constant velocity (Renesas)

| Part | Licence | Terms that matter |
|---|---|---|
| Code | Apache-2.0 (Paddock) | Paddock's own baseline: a few lines of arithmetic, with no weights and no third-party code. |

## 9. Benchmarks and their data: details

Each heading carries the colour of the benchmark's row in section 4.

#### 🟡 nuScenes (Motional): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Data: [nuScenes](https://www.nuscenes.org/nuscenes) | [CC BY-NC-SA 4.0 + Motional's Dataset Terms](https://www.nuscenes.org/terms-of-use) | "available for non-commercial use subject to the Terms of Use below" |
| Results derived from the data | Dataset Terms | "Use of any data derived from the Datasets, which may appear in any format such as tables and charts, is also subject to these Dataset Terms." |
| Precedence | Dataset Terms | "Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, these Dataset Terms shall prevail." |
| Endorsement and names | Dataset Terms | No implying that Motional endorses the use, and no use of its name, trademark or logo, without its written consent. |
| Other clauses | Dataset Terms | The user indemnifies Motional. Massachusetts law. Motional may amend the terms at any time. |
| Commercial use | [Commercial licence](https://www.nuscenes.org/terms-of-use-commercial); [FAQ](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/faqs.md) | Free "for educational use and some research use"; for commercial use, "we adjust the pricing terms to the use case and company size". |
| Code | Paddock's own | Follows a published scoring method. |
| In the release | Results; the validation scene list as Motional's nuScenes devkit publishes it (Apache-2.0); a verbatim subset of nuScenes' tables for two scenes, with UniAD's and VAD's inputs and plans on them, as test data | Section 6. |

#### 🟡 nuPlan (Motional): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Data: [nuPlan](https://www.nuscenes.org/nuplan) | Same as nuScenes | One document, titled "nuScenes, nuPlan, nuImages, and nuReality Terms of Use", covers both. |
| In the release | 5 result runs | Kept but not ranked, because they used part of the data. |

#### 🟡 NAVSIM v1 and v2 (autonomousvision): recorded drives; the model's plan is scored in a short simulation

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: [autonomousvision/navsim](https://github.com/autonomousvision/navsim) | Apache-2.0 | "The datasets (including nuPlan and OpenScene) inherit their own distribution licenses." Paddock runs the copies inside Drive-JEPA. |
| Data: [OpenScene](https://github.com/OpenDriveLab/OpenScene) (OpenDriveLab), a smaller re-release of nuPlan | CC BY-NC-SA 4.0 + nuPlan's terms | Users "may not use the work for commercial purposes". [Copy on Hugging Face](https://huggingface.co/datasets/OpenDriveLab/OpenScene). |
| In the release | Results | |

#### 🟡 KITScenes (Karlsruhe Institute of Technology): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: the library that reads the data | [Apache-2.0](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/LICENSE), [NOTICE](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/NOTICE) | Bundles Lanelet2 (BSD-3-Clause) and CC0 icons. "Using this Python API does not grant rights to use the dataset beyond those terms." |
| Data: [KIT-MRT/KITScenes-Multimodal](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal), access on request | CC BY-NC 4.0 + KIT's terms ([summary](https://github.com/KIT-MRT/kitscenes/blob/7765cdec5490894266070ab46e23724b58b3da42/README.md#license)) | KIT's terms prevail where they conflict. Research papers must cite the KITScenes paper, and media use must link to kitscenes.com. "We reserve all rights that are not explicitly granted to you." |
| Intended use | Dataset card | "intended for academic research". Also: "For final benchmark reporting, please wait for a more stable public release." |
| Term added on 13 November 2026 | KIT's terms | Aerial feature tiles: users "must not reconstruct, recover or approximate the source imagery from them", nor publish such reconstructions. Paddock does not use the aerial tiles. |
| In the release | Results; the list of the 223 keyframes Paddock scores | Section 6. |

#### 🟢 comma2k19 (comma.ai): recorded drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code: the toolkit | [MIT](https://github.com/commaai/comma2k19/blob/4c7f1a6e1957745beadc1def0e7225f559b09a2a/LICENSE) | |
| Data: [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19) | MIT (tag) | Also distributed through Academic Torrents, as the toolkit's README says. |
| In the release | Results | |

#### 🟡 Bench2Drive (Thinklab, Shanghai Jiao Tong University): CARLA simulator

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code and routes | [CC BY-NC-ND 4.0](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE) | "All assets and code are under the CC-BY-NC-ND unless specified otherwise" ([README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md#license)). The licence lets you "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only", and "produce and reproduce, but not Share, Adapted Material". |
| Folders `leaderboard/` and `scenario_runner/` | [MIT](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/LICENSE), copyright CARLA | The [file the patches edit](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/leaderboard/leaderboard/leaderboard_evaluator.py) is headed with Intel's copyright and the MIT licence. The lines the patches change were added by Bench2Drive; they are not in [CARLA's original](https://github.com/carla-simulator/leaderboard/blob/leaderboard-2.0/leaderboard/leaderboard_evaluator.py). (Q7) |
| Simulator: CARLA 0.9.15, which each user installs | [MIT code, CC-BY assets](https://github.com/carla-simulator/carla/blob/0.9.15/README.md#licenses) | "CARLA specific code is distributed under MIT License. CARLA specific assets are distributed under CC-BY License." CARLA is built on Unreal Engine 4, which "follows its own license terms" ([Unreal Engine FAQ](https://www.unrealengine.com/en-US/faq)). |
| In the release | Results; two patches; the names of two routes; outputs of a few of Paddock's own runs, used as test data | Section 6. Bench2Drive's training dataset (Apache-2.0, on Hugging Face) is not used. |

#### 🟡 neuroNCAP (William Ljungbergh, Adam Tonderski et al.): nuScenes scenes re-rendered so that a model can drive in them

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [MIT](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/LICENSE) | |
| Renderer: [neurad-studio](https://github.com/georghess/neurad-studio) | Apache-2.0 | |
| Renderer files, one per scene | None stated | Downloaded from a Dropbox link in neuro-ncap's [download script](https://github.com/atonderski/neuro-ncap/blob/ecdcf284e2b7b83c537f3292a06c0adddff55811/scripts/download/download_neurad_weights.sh). They are reconstructions of nuScenes scenes. |
| Data | nuScenes, as above | |
| In the release | Results; three patches; 20 test scenarios, neuro-ncap's own (MIT) with the time step changed, holding nuScenes object IDs and positions; four website videos of its scenes | Section 6. |

#### 🔴 AlpaSim (NVIDIA): reconstructed real drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [Apache-2.0](https://github.com/NVlabs/alpasim/blob/affc2eab209fa43bdfa2f26c0f8d437922d78a68/LICENSE) | "Copyright 2025 NVIDIA Corporation" |
| Scenes: [nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec), release 26.01 | NVIDIA Autonomous Vehicles NuRec Dataset License Agreement, accepted by click-through ([PDF](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/blob/main/LICENSE.pdf), visible after accepting) | The text is identical in release 26.01 and in the current version. |
| §1, purpose | Agreement | "solely for your internal development of autonomous vehicles and automated driving assisted systems using NVIDIA technology" |
| §3, confidentiality | Agreement | "'Confidential Information' means the Dataset including its features and functionality, output, and any results of benchmarking or other competitive analysis or regression or performance data relating to the Dataset." The user "will not disclose NVIDIA Confidential Information to any third party", except to authorised users bound by a written confidentiality agreement. |
| §4.4 and §4.6, limits | Agreement | No identifying people, including through licence plates. No derivative works, and no distributing or hosting the data. |
| §8, term | Agreement | "This Agreement expires twelve (12) months after the date of initial delivery or download of the Dataset." On termination the user "must stop using and destroy all copies of the Dataset". |
| §11, law | Agreement | United States and Delaware law; the courts of Santa Clara County, California. |
| In the release | 15 result runs over nine models, Alpamayo 2 Edge's added on 16 November; an AlpaSim column in the README's results chart; an AlpaSim page; test files cut from two real runs, until pull request #183 merges; four website videos of its scenes, whose captions give per-scene scores; the scene list, which comes from AlpaSim's own repository | Section 6 |

#### 🟡 HUGSIM (X-D Lab, Zhejiang University): reconstructed real drives

| Part | Licence or source | Terms that matter |
|---|---|---|
| Code | [MIT](https://github.com/hyzhou404/HUGSIM/blob/62c690d39fd90020e68a196bd8bcc1c4d4191f2e/LICENSE) | |
| Scenes: [XDimLab/HUGSIM](https://huggingface.co/datasets/XDimLab/HUGSIM) | MIT, tag only, no licence text | Built from the four sources below (Q10). |
| Source: [nuScenes](https://www.nuscenes.org/terms-of-use) | CC BY-NC-SA 4.0 + Motional's terms | As above |
| Source: [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) | CC BY-NC-SA 3.0 | "you may not use this work for commercial purposes" |
| Source: [Waymo Open Dataset](https://waymo.com/open/terms/) | "Waymo Dataset License Agreement for Non-Commercial Use (March 2025)" | Non-commercial use "means research, teaching, scientific publication and personal experimentation", and expressly includes benchmarking for research publication. Work made with the data must carry an attribution. §4(a) begins "You must not use or deploy the Dataset, any models trained on or refined using the Dataset, or the weights or biases", and bars their use in vehicles, in production systems, or for other primarily commercial purposes. Commercial licences: open-dataset@waymo.com. |
| Source: PandaSet (Scale AI and Hesai) | "a no-cost commercial license", according to [its paper](https://arxiv.org/abs/2112.12610) | Its website, pandaset.org, no longer shows PandaSet: on 10 November 2026 it redirected to an unrelated site, so the licence text itself could not be read. |
| Vehicle models: [3DRealCar](https://github.com/xiaobiaodu/3DRealCar_Toolkit) | Apache-2.0 | "3DRealCar dataset is allowed commercial usage for free." |
| In the release | Results for two models; two patches to MIT code, one of them HUGSIM's own proposed fix ([pull request #57](https://github.com/hyzhou404/HUGSIM/pull/57)) | |

## 10. Listed in Paddock but not used

| Project | Licence | Note |
|---|---|---|
| sunnypilot | Custom: [LICENSE.md](https://github.com/sunnypilot/sunnypilot/blob/0b1ed0d047aca5bed45f20323823c5d5e9cb6bc1/LICENSE.md) | Use "for any commercial purposes, for-profit projects, or in closed source projects requires explicit written permission from the original author(s)", and any project using it must show an acknowledgement. Removed from the list in pull request #181, merged on 12 November. |
| Alpamayo 2 (NVIDIA) | Code: [Apache-2.0](https://github.com/NVlabs/alpamayo2/blob/beb2977d9a7e9d66837d4a3ad5144ff59de37519/LICENSE). Weights: [nvidia/Alpamayo2-Super](https://huggingface.co/nvidia/Alpamayo2-Super), OpenMDW-1.1. | Not tested yet |
| V-JEPA 2 (Meta) | [MIT](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE), with three files under Apache-2.0 | Listed for reference. Paddock uses only the copies inside Drive-JEPA and WA-JEPA. |

## 11. What changed since the review of 11 November

- **Merged:** pull requests #181 (sunnypilot removed) and #182 (THIRD_PARTY_NOTICES.md
  and corrected weights licences), both on 12 November. #183 (made-up AlpaSim test files)
  is still open.
- **New models:** UniAD and VAD, on 12 November. Their code is Apache-2.0, and both were
  trained on nuScenes. UniAD's weights are Apache-2.0; VAD's state no licence. Their
  environment is built from Bench2DriveZoo, which is CC BY-NC-ND 4.0. Alpamayo 2 Edge now
  has results on eight benchmarks.
- **Dataset excerpts in the repository:** a verbatim subset of nuScenes' tables for two
  scenes, in UniAD's and VAD's test data, and the list of 223 KITScenes keyframes that
  Paddock scores (Q5).
- **Website videos:** four AlpaSim videos, whose captions give per-scene scores, joined
  the four neuroNCAP videos on 12 November (Q1, Q6). The Hugging Face dataset that holds
  them was public on 11 November and is not publicly reachable on 16 November.
- **AlpaSim:** 15 runs; Alpamayo 2 Edge's was added on 16 November.
- **Terms:** KITScenes added a term on 13 November about aerial feature tiles, which
  Paddock does not use. Every other licence and model card is unchanged, and NVIDIA's
  agreement is word for word the same.
- **New in this review:** the decision list (Part A) and its background (section 1);
  downloads (section 6.1), videos and images (6.2), references and names (6.3), and
  release hygiene (6.4). Its new findings: nuScenes' download addresses skip Motional's
  sign-in (Q9); the run records name contributors, FastCode and Renesas hosts (Q13);
  KITScenes' link and Waymo's sentence are missing (Q11); nothing says what the scores
  are (Q14); and weights download without their licence (Q12).

## 12. How this was checked

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
  sources in section 7) were read on 10 November 2026.
- **Links**: all 81 links in this document were opened again on 16 November 2026, and
  every one opened except two, as on 10 November. The AlpaSim PDF opens only after
  accepting the agreement, and the Unreal Engine site refuses automated access; that link
  is the one CARLA's own README gives.
- **The clean-ups** were made on 10 November 2026 in pull requests #181 to #183. While making
  them, the nuScenes scene list, the 20 neuroNCAP scenarios, the AlpaSim scene list and the
  Bench2Drive route file were each compared with their upstream source; section 6 gives
  what each turned out to be.
- **Re-checked on 16 November 2026 against `main` at `73e9ac6`.** Every licence tag and
  model card above is unchanged, and NVIDIA's AlpaSim agreement is word for word the same.
  KITScenes' terms gained one term, on aerial features, which Paddock does not use.
- **Also checked on 16 November**: every binary and large file in the repository (none
  holds dataset images or video; the large JSON files are Paddock's own results and test
  data, one of them NAVSIM's own simulator run on made-up plans); the fields of all 163
  run records; the download addresses in `datasets.yaml`, against those Motional's
  nuScenes tutorials and NAVSIM's download scripts publish; and what the website loads
  from other hosts (only the videos and their stills; it sets no cookies and has no
  analytics).
- **Not checked**: PandaSet's licence text; WA-JEPA's training data beyond Paddock's
  notes; and the licences of the software libraries each model installs, such as
  PyTorch, which Paddock does not ship.
