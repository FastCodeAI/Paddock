# Release v1.0.0: the plan, from executive summary to evidence

The documents behind Paddock 1.0, and the RAIL programme around it. They are organised
from high level to low level, so each reader can stop at the depth they need.

| Level | What | Start with |
|---|---|---|
| **0 · Executive** | the decisions, the story, the date | [plan-v1.0.0.md](plan-v1.0.0.md), or present [briefing.html](briefing.html) |
| **1 · Strategy** | one document per executive question, with its evidence | [1-strategy/](1-strategy/) |
| **2 · Design** | how each piece would work: decisions, not implementations | [2-design/](2-design/) |
| **3 · Reference** | every number's source, open work, a glossary | [3-reference/](3-reference/) |

**Reading paths**
- **Leadership:** the executive plan, or the briefing (about 20 minutes). Then
  [8 · Release](1-strategy/08-release-v1.0.0.md) for the decisions.
- **Renesas inference teams:** [7 · Jockey](1-strategy/07-jockey.md) and
  [model to car](2-design/model-to-car.md). The six questions for you are in 7.
- **Engineers on the release:** [8 · Release](1-strategy/08-release-v1.0.0.md),
  [packaging and setup](2-design/packaging-and-setup.md) and
  [open work](3-reference/open-work.md).

## Opening the briefing

`briefing.html` is one self-contained page. It needs no network: it loads only the
figures beside it. GitHub shows HTML as source, so open it from a checkout:

```bash
git clone git@github.com:FastCodeAI/Paddock.git && cd Paddock
git checkout plan/release-v1.0.0
# then open docs/release/briefing.html in any browser
```

**Navigating it**
- The arrow keys, Page Up and Page Down move between topics.
- The figures on the map are links.
- Printing gives one topic per landscape page, which makes a PDF handout.

It is laid out for a 1680×1050 screen, and nothing scrolls sideways at that size.

## Conventions

- **Snapshot.** Facts are as of 22 October 2026: Paddock `main` at `3b0b42a`, Gallops
  `main`, and GitHub's open pull requests.
- **Verified against argued.** Every document ends with its sources, keeping what was
  verified apart from what is argued, as `docs/reviewing.md` asks of reviews.
  [Evidence](3-reference/evidence.md) collects the headline numbers in one place.
- **Width.** Figures are fixed 1280-pixel canvases and tables have at most five short
  columns, so nothing scrolls sideways on GitHub or at 1680×1050.
  [figures/README.md](figures/README.md) explains how to re-render the figures.
- **Keep the pair in step.** The briefing mirrors the executive plan: change one, change
  the other.
- **Supersedes** the OEM deployment plan of 15 October, which was never committed. Its
  decisions are carried into [2-design/](2-design/).
