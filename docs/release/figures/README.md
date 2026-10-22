# Release figures

Every figure in `docs/release/` is a PNG rendered from one HTML/CSS source in
[`src/`](src/). Edit the source, then re-render. Never edit a PNG by hand.

## The budget: why each figure is a fixed 1280 px canvas

The release docs have to be readable on a 1680×1050 screen without horizontal
scrolling. Each figure is therefore a canvas:
- exactly **1280 CSS px wide**;
- at most **860 px tall**;
- with no text smaller than **16 px**.

What that buys:
- At 1680×1050 every figure shows 1:1, whole, with room for the browser's own chrome.
- GitHub scales an image into its ~1,000 px column. There, 16 px text lands near
  12.5 px, the smallest that still reads comfortably. That is the reason for the
  floor.
- Each figure is a ready-made 16:9-ish slide.

The PNG is rendered at 2× (2,560 px wide), so it stays sharp when scaled.

**Why not Mermaid,** which the rest of `docs/` uses:
- Mermaid lays a diagram out by itself, and nothing holds that layout to a width.
- The deployment figures of 15 October were Mermaid, about 3,000 px of layout
  squeezed into 784. Their labels came out near 8 px.
- An HTML/CSS canvas cannot grow past its width. When text does not fit, the
  renderer fails rather than shrinking it.

## Conventions

Colour comes from four hues: a colour-blind-safe set that passed an all-pairs check
in light and dark modes. A six-hue set failed (green/orange and pink/orange could
not be told apart), so do not add a fifth hue. **Colour is never the only cue:**
everything coloured also carries a label, and usually a shape or an outline.

| In the strategy figures | Hue | Class |
|---|---|---|
| Paddock: benchmark and analysis | blue `#2a78d6` | `p-paddock` |
| Gallops: fine-tuning | green `#008300` | `p-gallops` |
| Jockey: the in-house model | pink `#e87ba4` | `p-jockey` |
| What customers receive | yellow `#eda100` | `p-out` |
| Anything outside RAIL | grey | `p-ext` |

The design figures colour artifacts by type instead. Shape repeats the type, so the
figure still reads in greyscale:

| Artifact | Look | Class |
|---|---|---|
| Licensed data | blue fill | `t-data` |
| Model files | pink pill | `t-model` |
| Code and environments | green, double outline | `t-code` |
| Outputs: runs, videos, reports, checkpoints | yellow fill | `t-out` |
| Licence or data contract | dashed outline | `t-contract` |
| Customer IP | thick black outline | `t-ip` |
| Service or compute | white | `t-svc` |
| People and agents | light grey | `t-actor` |

Other shared pieces, all in [`src/figures.css`](src/figures.css):
- **Status chips:** `done` (green), `wip` (yellow), `plan` (dashed).
- **Arrows:** `arrow`, with `.down`, `.left` and `.dashed` variants. They are
  drawn in CSS, so they scale with the canvas.
- **Rating dots:** `dots` with `<i class="on">`. These are CSS circles, because the
  half-dot glyph falls back to a font of a different size.

## Rendering

[`render.cjs`](render.cjs) screenshots the `.figure` element of each source at 2×.
It refuses to write a PNG when the figure breaks its budget: the wrong width, too
tall, or any text overflowing its box.

Puppeteer is not a paddock dependency. Install it once, outside the repo:

```bash
npm install --prefix <scratch> puppeteer@19
NODE_PATH=<scratch>/node_modules node docs/release/figures/render.cjs            # all
NODE_PATH=<scratch>/node_modules node docs/release/figures/render.cjs rail-map   # one
```

- Set `CHROME` to use a Chrome or Chromium other than the one puppeteer downloaded.
- Set `FIGURE_DRAFT=1` to write the PNG even when the budget check fails, so you can
  see what broke.
- Fonts: the canvas asks for Carlito, which is metric-compatible with Calibri, then
  Liberation Sans. Render on a machine that has one of them, or the line breaks
  move.
- After rendering, open the PNG at 100% and read it. A figure that renders without
  error can still have a label in the wrong place.

## Index

| Figure | Shows | Used in |
|---|---|---|
| [rail-map](rail-map.png) | Paddock, Gallops and Jockey, the loop between them, inputs and outputs | executive plan §1, strategy 1, briefing home |
| [score-vs-driving](score-vs-driving.png) | six models ranked open loop (nuScenes) against closed loop (neuroNCAP): the ranks flip | executive plan §2, strategy 2 |
| [trial-ladder](trial-ladder.png) | five rungs from an evidence pack to the customer's own model, and which release delivers each | executive plan §3, strategy 3, hands-on access |
| [using-paddock](using-paddock.png) | setup, orchestration, compute, storage and results: today, in v1.0.0, later | executive plan §4, strategy 4 |
| [analysis-loop](analysis-loop.png) | evaluate, diagnose, choose the cheapest remedy (input, choice, fine-tune, new model), re-score with the gate | executive plan §5, strategy 5 |
| [gallops-results](gallops-results.png) | six fine-tuning results, before and after, with how far each gain holds | executive plan §6, strategy 6 |
| [jockey-candidates](jockey-candidates.png) | four model families rated on six criteria, and what Jockey takes from each | executive plan §7, strategy 7 |
| [model-to-car](model-to-car.png) | five stages from training to a car, who owns each, and what exists | strategy 7, design: model to car |
| [release-timeline](release-timeline.png) | v1.0.0 worked back from 25 November: workstreams, milestones, holidays, what follows | executive plan §8, strategy 8 |
