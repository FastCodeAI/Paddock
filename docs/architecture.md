# Architecture

Two axes — models and benchmarks — meeting at one interface.

```
                    ┌───────────── Policy ─────────────┐
                    │  load() · reset() · infer(obs)   │
   Benchmarks       └──────────────────────────────────┘        Models
   ──────────                      ▲                            ──────
   kitscenes   ──┐                 │                ┌── openpilot    (uv, py3.12)
   nuscenes    ──┤  open loop:     │  closed loop:  ├── alpamayo1.5  (uv, py3.12)
   navsim      ──┼─ iterate        ┼─ simulator     ┼── drivejepa    (conda, py3.9)
   neuroncap   ──┤  samples        │  drives over   ├── visionpilot  (C++)
   bench2drive ──┘  in process     │  HTTP          └── ...
                                   ▼
        results/<bench>/<model>/<run>/{run,metrics}.json    tracked, KB
        {runs}/<bench>/<model>/<run>/{samples.jsonl,...}    external, GB
```

---

## The Policy protocol

Three methods, defined in `paddock/policy/protocol.py`:

```python
class Policy(ABC):
    cameras: tuple[str, ...] = ("CAM_FRONT",)
    waypoint_dt: float = 0.5

    def load(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, obs: Observation) -> Trajectory: ...
```

`Observation` carries one timestep: images per camera, `ego2world`, an 18-float
nuScenes-layout CAN bus vector, timestamp, a navigation command, and calibration.
`Trajectory` is a plan in the ego frame — x forward, y **left**, metres — evenly
spaced by `waypoint_dt` starting at `t = waypoint_dt`.

### Why one protocol for both loop types

The same `Policy` serves open-loop scoring (called in-process) and closed-loop
simulation (wrapped in an HTTP node by `policy/server.py`).

This is not tidiness. Before paddock, each model had *two* implementations of its
camera handling — one in its open-loop eval script and one in its simulator
node — and they diverged. In one case the two paths cropped the image
differently, which means an ADE and an NCAP score for that model were statements
about two slightly different systems. One `Policy` per model, owning its warp
once, is what makes cross-benchmark comparison mean anything.

### The wire contract

Closed-loop benchmarks talk to a model node over three endpoints: `/alive`,
`/infer`, `/reset`. Ported unchanged from the NeuroNCAP model nodes.

Two details that look like off-by-ones and are not:

- **Six waypoints is exact, not a floor.** NeuroNCAP builds six waypoint
  timestamps per step and indexes them positionally; a seventh is dropped, a
  fifth raises `IndexError`.
- **An all-zero trajectory is a commanded emergency stop**, not "no plan". A
  model that has not warmed up must return `constant_velocity(obs.speed)`.
  Returning zeros scores as a deliberate brake, which for a collision-avoidance
  benchmark looks like a *good* result for the wrong reason.

`Observation.from_wire` decodes images that arrive as base64
`torch.save`d uint8 tensors. It includes a torch-free reader, because openpilot's
environment runs tinygrad and has no torch at all — installing torch there just
to unpickle an image would be absurd, and would change what is being evaluated.

## Open loop vs closed loop

| | Open loop | Closed loop |
|---|---|---|
| Who decides what happens next | the dataset | the model |
| Errors | do not compound | compound — the point |
| Cost | cheap, parallel | expensive, serial |
| Measures | imitation of a recorded expert | whether it can actually drive |
| Examples | KITScenes, nuScenes, NAVSIM | neuroNCAP, Bench2Drive |

Both matter, and neither substitutes for the other. Open-loop metrics are
dominated by ego status — a model that just continues the current motion scores
respectably — so a good ADE is weak evidence of competence. Closed-loop is where
a policy has to recover from its own mistakes, but it is orders of magnitude more
expensive and has far higher variance, so it cannot be the fast feedback signal.

The `Benchmark` base classes in `paddock/benchmark.py` reflect this split:
`OpenLoopBenchmark` implements `samples()` and `score()`; `ClosedLoopBenchmark`
implements `episodes()` and `run_episode()`. Both implement `preflight()`.

## Process and GPU model

Every model runs in its **own** interpreter, in its own environment, as a
subprocess. Five Python versions across three managers are in play — this is what
evaluating four independent research codebases costs, and the registry makes it
declarative rather than tribal.

Paddock owns model processes; benchmarks own simulators. For closed loop:

1. `runtime/gpu.py` waits for the VRAM the model declares, rather than failing
   eight minutes into a checkpoint read with a CUDA OOM traceback that says
   nothing about the real cause (someone else has the card).
2. `runtime/proc.py` starts the node with the env built by `runtime/env.py`,
   polls `/alive` until it answers, runs every scenario, then tears it down and
   waits for the driver to reclaim the memory.
3. The next model starts only then. With `gpu.serialize = true` exactly one model
   is resident at a time. The simulator's renderer stays up throughout: it is
   model-independent and expensive to start.

Every model sees identical episodes and identical seeds, derived from the run
index. The comparison is paired, not merely averaged — which is worth far more
and costs nothing if planned for.

## Configuration

Covered in [site-setup.md](site-setup.md); the short version is that tracked
files declare what a thing is and `paddock.toml` declares where it lives.
`config.py` expands the template variables; `tests/test_portability.py` fails the
build if a real machine path reaches a tracked file.

## Where does a file belong?

One question settles it:

> **Does this need to know which model produced the result?**

If no, it belongs to the benchmark or the shared layer. Three tiers:

| Tier | Knows about | Examples |
|---|---|---|
| `paddock/viz/` | neither models nor benchmarks | BEV drawing, the model colour palette, contact sheets |
| `adapters/benchmarks/<b>/render.py` | this benchmark's ground truth and what a comparison means; takes runs from **any** models | KITScenes error distribution and worst-disagreement grid |
| `adapters/models/<m>/` | this model's internals only | openpilot's field-of-view margin, VisionPilot's planner trace |

**Rendering is benchmark-scoped, not model-scoped**, because debugging means
putting several models side by side on one benchmark. A model cannot draw that
comparison: it does not know about its rivals, and should not.

Before paddock this was inverted. A three-way KITScenes plot lived under
openpilot, a trajectory plot under Drive-JEPA, a per-sample sheet under
Drive-JEPA again -- each able to draw only the model it sat beside, and each
re-deriving the same axes. `tools/` now mirrors the same split:

```
tools/benchmarks/<b>/   harness checks and one-off figures for that benchmark
tools/models/<m>/       probes into that model's internals
```

### The colour rule

`paddock/viz/palette.py` assigns a colour per **model name**, fixed once.
Debugging produces several figures at a time; if colour came from position in
whatever list a script built, a model would change colour between figures and
the reader would have to re-check the legend each time -- or not notice.

This is easy to break by accident. Folding a metric into a dict key
(`"openpilot 2.84 m"`) makes the colour a function of the number, and the same
model comes out a different colour in every panel. Pass legend text separately;
`bev.draw` takes `labels=` for exactly this reason.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | site config, template expansion |
| `registry.py` | load repos / datasets / models / benchmarks; adapters are discovered, not listed |
| `doctor.py` | readiness reporting |
| `policy/protocol.py` | `Policy`, `Observation`, `Trajectory`, the wire codec |
| `policy/server.py` | wrap any `Policy` into an HTTP node |
| `benchmark.py` | `OpenLoopBenchmark`, `ClosedLoopBenchmark`, `Sample`, `Episode` |
| `runtime/env.py` | per-model environment construction |
| `runtime/proc.py` | launch, health-check, tear down model nodes |
| `runtime/gpu.py` | VRAM gating |
| `metrics/` | trajectory (ADE/FDE), ncap, pdms |
| `results/` | canonical store, ingest of legacy formats, reporting |
| `adapters/models/<n>/` | one model |
| `adapters/benchmarks/<n>/` | one benchmark: `harness.py` to run it, optional `render.py` to compare models on it |
| `viz/` | drawing primitives shared by every benchmark |
| `render.py` | loads runs and dispatches to a benchmark's renderer |
