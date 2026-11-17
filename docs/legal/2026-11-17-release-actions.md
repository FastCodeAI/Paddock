# Paddock release: what it may and may not do, after Legal's review

*17 November 2026. Combines Renesas Legal's views from the meeting that day (see
`2026-11-17-legal-meeting-summary.md`) with the licence review
(`paddock-third-party-licences.md`, whose questions Q1 to Q26 are cited below).*

The release is paused. Legal cannot clear it until three external law firms advise on
whether publishing benchmark results on non-commercial datasets is a commercial use,
which is expected within about two weeks. The red items must change whatever is decided.
The amber items wait for the business's decision. The green items can go ahead.

Each item's basis is one of:

- **Legal**: Legal's view in the meeting.
- **Follows Legal**: our reading of Legal's reasoning, applied to a case the meeting did
  not discuss. Legal should confirm it.
- **Review**: the licence review alone; the meeting did not discuss it.

## 🔴 Not acceptable: change before any release

| Action | Basis | Instead |
|---|---|---|
| Publish anything from AlpaSim: its results, chart column, page, videos and captions (Q1) | Legal | Remove all of it, and merge pull request #183, which replaces the test files cut from real AlpaSim runs with made-up ones. Also ask Legal whether Renesas may keep using AlpaSim internally, since its use is outside the licence's scope. |
| Show the website's four videos of nuScenes scenes (Q6) | Legal | Publish the website without its video section. |
| Publish the nuScenes download addresses that skip nuscenes.org's sign-in (Q9) | Legal | Point users to nuscenes.org, where they sign in and accept the terms; Paddock then checks the files they downloaded. |
| Publish nuPlan's direct download addresses, and download HUGSIM's scenes rebuilt from Waymo data for users who never registered with Waymo (Q9, Q10) | Follows Legal | The same: each user signs up with the owner first. |
| Publish the two Bench2Drive patches as they are (Q7) | Legal | Checked on 17 November: they hold 28 lines of the file they edit. 17 are CARLA's original code (MIT) and 5 are blank, but 6 are Bench2Drive's own: 3 lines that one patch replaces, and 3 unchanged lines around the other's edits. Rewrite them to hold only Paddock's own lines, make both fixes from Paddock's own code, or offer them to Bench2Drive as pull requests. |
| Keep verbatim nuScenes records in UniAD's and VAD's test data (Q5) | Follows Legal | Legal treats publishing dataset content like publishing the dataset. Replace them with made-up data, as was done for AlpaSim. |
| Publish through a university that Renesas funds, to use its research exception | Legal | None: Legal does not see this as a way around the licences. |

## 🟡 Needs the business's decision, after Legal's risk analysis

| Action | Basis | Risk and options |
|---|---|---|
| Publish results on non-commercial datasets: nuScenes, nuPlan, NAVSIM v1 and v2, neuroNCAP, KITScenes, HUGSIM, and Bench2Drive, whose code is non-commercial (Q2, Q4) | Legal | Renesas accepted these licences itself, and publishing to promote Renesas products is probably commercial use. Legal rates the risk very low: the worst case is likely a request to take the results down. The options are to wait for the external advice, buy commercial licences (Motional sells one for nuScenes and nuPlan), or publish and accept the risk in writing. |
| Publish the scores of AutoE2E and Alpamayo 1.5, whose own weights are non-commercial (Q8) | Follows Legal | Renesas accepted these weights' terms itself, as with the datasets. Decide together with the row above. |
| Keep the KITScenes keyframe list and the two Bench2Drive route names, small lists drawn from non-commercial material (Q5) | Follows Legal | Decide together with the results. |

## 🟢 Acceptable

| Action | Basis | Condition |
|---|---|---|
| Publish results on comma2k19, whose data is MIT | Legal | Credit comma.ai. |
| Run other parties' pre-trained models, including ones they trained on non-commercial data, and publish their scores (Q8) | Legal | Their scores on non-commercial datasets still wait for the amber decision. |
| Publish patches that hold only Paddock's own changes, for users to apply themselves (Q7) | Legal | The patches to MIT and Apache-2.0 code (neuro-ncap, Drive-JEPA, HUGSIM, UniAD, and OpenMMLab's code in Bench2DriveZoo) are fine as they are (Review). |
| Point users to each owner's own download host, where they accept the owner's terms themselves: a sign-in page, a gated host such as KITScenes', or an owner's open host such as OpenScene's | Legal; open hosts follow Legal | None. |
| Download freely licensed code and data directly: upstream code at its pinned version, comma2k19 and CARLA | Review | None. |
| Publish Paddock's own code under Apache-2.0 | Review | Once Renesas's ownership of every contribution is confirmed (Q3), and with complete third-party notices (Q19). |
| Name models, datasets and companies to describe them, without logos (Q20) | Review | None. |
| Publish Paddock as a new repository, without its git history (Q18) | Review | None. |

## Not discussed on 17 November

The review's recommendations stand until Legal says otherwise.

- **Needed before release:** confirm that Renesas owns all of the code (Q3); mark the
  results with their data's licences (Q4); show HUGSIM's source terms (Q10); add Waymo's
  credit sentence and a link to kitscenes.com (Q11); show each model's weights licence in
  `paddock setup` (Q12); replace the people's and hosts' names in the run records (Q13);
  say that the scores are research measurements, not safety assessments (Q14); and
  complete THIRD_PARTY_NOTICES.md (Q19).
- **To confirm:** openpilot's indemnity (Q15); code written with an AI assistant (Q16); a
  trademark search for "Paddock" (Q21); contributions (Q22); dependencies (Q23);
  reporting KITScenes' scores as preliminary (Q24); export control (Q25); and the
  website's links to Renesas's legal pages (Q26).

## Follow-ups

| Who | What | When |
|---|---|---|
| Model team | Send Legal the meeting summary and this list | Now |
| Model team | Make the red changes, which are needed whatever is decided | Now |
| Model team | Check whether other companies, not universities, publish benchmark results on non-commercial datasets. The review (section 7) already found three: NVIDIA's research team (a NAVSIM challenge, on nuPlan data), Alibaba's Qwen team (NAVSIM and Waymo scores) and Zenseact (neuroNCAP, on nuScenes scenes). Still to check: NXP, Infineon and other chip makers. | Before the business decides |
| Carl Mair | Send the risk analysis to the business, and look again at NVIDIA's licence | After receiving this list |
| Carl Mair | Share the external law firms' advice | Within about two weeks |
| Tarkesh Pande | Decide the amber items, meeting Carl if useful | After Legal's risk analysis |
