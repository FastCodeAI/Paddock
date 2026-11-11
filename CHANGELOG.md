# Changelog

Notable changes to Paddock. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (unreleased)

First public release.

- One `Policy` interface for open-loop scoring and closed-loop simulation, with each
  model in its own environment behind a local HTTP server.
- Adapters for Alpamayo 1.5, Alpamayo 2 Edge, AutoE2E, Drive-JEPA, METEOR, OpenPilot,
  Qwen-Drive 1.0, VisionPilot and WA-JEPA, and a constant-velocity floor.
- Benchmarks: nuScenes, KITScenes, comma2k19, nuPlan, NAVSIM v1 and v2, Bench2Drive,
  neuroNCAP, AlpaSim and HUGSIM.
- A command line: `paddock doctor`, `setup`, `new`, `verify`, `run`, `render` and `report`.
- `paddock data fetch` downloads every dataset Paddock reads (comma2k19, HUGSIM, NAVSIM,
  nuScenes, nuPlan, KITScenes and neuroNCAP's renderers), and CARLA for Bench2Drive, from
  their official sources, pinned and checked file by file, and unpacks them where paddock
  reads them. `paddock setup --bench bench2drive` builds the Python that drives CARLA.
- A results store that records each run's settings and upstream commits beside its
  scores, and a leaderboard built from it.
- A regression gate that compares a change against earlier runs of the released models.
