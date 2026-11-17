# Paddock release: meeting with Renesas Legal, 17 November 2026

*57 minutes. Renesas Legal: Carl Mair. Model development team: Shreedhar Kodate and
Gayatri Banda. Reviewed: the licence review, `paddock-third-party-licences.md`. The full
list of what the release may and may not do is in `2026-11-17-release-actions.md`.*

## Outcome

- **Legal cannot clear the release yet.** The open question is whether publishing
  benchmark results on non-commercial datasets is a commercial use. Three external law
  firms are advising on it, because Renesas's Edge AI group asked the same question; that
  group has paused publishing models it retrained on such datasets. The advice is
  expected within about two weeks.
- **Legal will send a risk analysis, and the business decides.** The decision is Tarkesh
  Pande's, as Ash is away until 30 November.
- **The release is paused** until then.

## Legal's views (preliminary)

| Topic | Legal's view |
|---|---|
| Results on non-commercial datasets (nuScenes, NAVSIM, KITScenes and others) | Renesas accepted these licences itself, and publishing the results promotes Renesas products. Conservatively, that is commercial use and a breach. Legal sees the risk as very low (the worst case is likely a request to take the results down) and will not block the release if the business accepts the risk. |
| Results on MIT-licensed data (comma2k19) | Fine. |
| AlpaSim (NVIDIA) | No. NVIDIA's licence covers internal development for NVIDIA-based vehicle platforms; Renesas's are not, so Renesas's use is outside its scope. NVIDIA is unlikely to grant permission. |
| Running other parties' pre-trained models | Fine: they trained the models, so Renesas never accepted the training data's terms. |
| Bench2Drive patches | Fine, if each patch holds only Paddock's own changes, with none of Bench2Drive's code, and users apply it themselves. |
| Download links | Remove any that let users skip a dataset's sign-in. Users must accept the owner's terms themselves; otherwise Renesas would in effect be sub-licensing the data. |
| Website videos of nuScenes scenes | No: the same as publishing the dataset. |
| Publishing through a university | No way around it if Renesas funds the work. |

## Actions

| Who | Action | Status |
|---|---|---|
| Model team | Send Carl this summary and the list of actions | This document |
| Model team | Check whether other companies, such as NXP and Infineon, publish benchmark results on non-commercial datasets | To do |
| Model team | Check whether the Bench2Drive patches hold any of Bench2Drive's code | Done: they hold 6 of its lines, so they need rewriting |
| Carl Mair | Send the risk analysis to the business | Waiting |
| Carl Mair | Share the external law firms' advice | Within about two weeks |
| Tarkesh Pande, with Carl Mair | Decide whether to wait, buy licences or accept the risk; meet if useful | Waiting |

**Also raised:** Renesas's AI dashboard on Hugging Face publishes optimized models, such
as Alpamayo 1.5. Carl asked who is responsible for it and whether it had a legal review.

**One clarification:** in the review's "At a glance" tables, "Commercial use: no" meant
that the licence does not allow commercial use. It did not claim that Renesas's use is
non-commercial; that is the open question. The column has been renamed to say so.
