# Paddock release: decision matrix for scores on non-commercial datasets

*20 November 2026. For the business decision that follows Legal's risk analysis, which
the meeting of 17 November gave to Tarkesh Pande, and for Renesas Legal. One page, built
from the licence review (`paddock-third-party-licences.md`), the meeting's list of
actions (`2026-11-17-release-actions.md`) and the industry check
(`2026-11-20-industry-practice.md`), whose sections are cited below. A summary for a
decision, not legal advice.*

**The decision.** Whether, and how, Paddock publishes its scores on the eight benchmarks
whose data or code is licensed for non-commercial use only: Motional's data (nuScenes,
nuPlan, NAVSIM v1 and v2, and neuroNCAP), KITScenes, Bench2Drive and HUGSIM. Everything
else is settled: the red items in the list of actions change whatever is decided, and its
green items go ahead. Legal's advice from three external law firms is due within about
two weeks of 17 November, and can arrive while any option below starts.

**Colours.** 🟢 can start now, with little or no risk left; 🟡 possible, with a cost or a
risk the business accepts in writing; 🔴 not recommended.

## 1. The options

| Option | What it allows | Time and cost | Risk left | Precedent |
|---|---|---|---|---|
| 🟢 **A. Release what is clear now** | Paddock's code, and comma2k19's results (MIT data) | Days | None from the datasets | comma2k19: Legal, 17 November. Code: once Q3 and Q19 are done. |
| 🟢 **B. Ask the owners in writing** | All eight, as each owner agrees | Weeks to months; terms unknown | Low, once agreed | MLPerf Automotive asked Motional, and Motional agreed after months (section 3) |
| 🟡 **C. Buy Motional's commercial licence** | Motional's five | Time unknown; a price "customized for your organization and use case" | Lowest, for those five | Motional sells licences; it names no licensees (section 4.1) |
| 🟡 **D. Publish as research, and accept the risk in writing** | All eight, now | Days, once the safeguards below are in | Low in practice; Motional's terms still ask for a licence for "industrial research and development" | 14 company papers and four chip makers' model libraries; no dispute found (sections 1, 2 and 6) |
| 🔴 **E. Publish beside product claims** | Scores next to Renesas chip names or speeds, in marketing | Days | Highest: what Motional's terms call use with "the expectation of generating revenue" | Other chip makers' marketing names rankings only; MLPerf asked first because scores may reach customers' quotations (sections 2 and 3) |
| 🔴 **F. Publish through a university Renesas funds** | Nothing that B or D does not | Months | Unchanged | Legal, 17 November: "No way around it if Renesas funds the work" (section 2.1) |

## 2. Each benchmark: who can say yes

| Benchmark | Data owner | Licence | Who to ask | Note |
|---|---|---|---|---|
| 🟡 nuScenes, nuPlan, NAVSIM v1 and v2, neuroNCAP | Motional | CC BY-NC-SA 4.0, plus Motional's terms, which cover "tables and charts" | Motional | One letter covers five |
| 🟡 KITScenes | KIT | CC BY-NC 4.0, plus KIT's terms | KIT | Label the scores preliminary (Q24) |
| 🟡 Bench2Drive | Thinklab, Shanghai Jiao Tong University | CC BY-NC-ND 4.0, for its code | Thinklab | Rewrite the two patches first (Q7) |
| 🟡 HUGSIM | Zhejiang University; scenes from nuScenes, KITTI-360, Waymo and PandaSet | Mostly non-commercial | Motional and KITTI-360's owners; Waymo, if its wording below is not enough | Waymo counts "applied research publication" as non-commercial (section 4.2) |
| 🟢 comma2k19 | comma.ai | MIT | No one | Credit comma.ai |
| 🔴 AlpaSim | NVIDIA | NVIDIA's NuRec agreement | Not asked | Out, by Legal's decision (Q1) |

## 3. Our recommended path, for the business to weigh

1. **Now, whatever is decided:** make the red changes in the list of actions, and add the
   safeguards that every option needs: each result marked with its data's licence (Q4),
   the credit lines (Q11), the disclaimer (Q14), no Renesas product claims beside these
   scores, and a way to take results down on request.
2. **Now:** write to Motional, KIT and Thinklab (option B), with MLPerf Automotive as the
   precedent, and set a date by which to have their answers.
3. **If the release cannot wait for the answers:** release option A first, and add each
   benchmark's results as its owner agrees.
4. **If an owner declines, or has not answered by that date:** choose, with Legal's risk
   analysis and the law firms' advice, between D (accept the risk in writing), C (buy
   Motional's licence), and leaving that benchmark's results out.

## 4. What the industry check adds

| Finding | What it means for the decision |
|---|---|
| MLPerf Automotive called non-commercial data a "legal ambiguity", asked Motional first, and Motional agreed | B is realistic; allow months (section 3) |
| Motional's commercial terms require a licence for "industrial research and development", "regardless of whether you work in academia or industry" | Under D, Renesas relies on Motional's tolerance rather than on its terms (section 4.1) |
| Companies, chip makers included, publish such scores widely, and no dispute over published scores was found | D's risk is low in practice (sections 1, 2 and 6) |
| Most company papers have university co-authors, but the licences judge purpose, not identity | No shortcut through a university (section 2.1) |
| Jamendo, a dataset owner, sued NVIDIA in 2026 under a non-commercial licence, over building products | Keep scores apart from products (section 6.2) |
| Lawsuits over Creative Commons licences usually claim a missing credit, and version 3.0 licences allow no cure | Get every credit line right before release (section 6.4) |

## 5. Decision

| Question | Answer |
|---|---|
| Which options: A, B, C, D? | |
| Write to Motional, KIT and Thinklab? | |
| Date by which to have their answers | |
| If no answer by then: C, D, or leave those results out? | |

**Decided by** ____________ **on** ____________.
