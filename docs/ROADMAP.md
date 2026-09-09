# Roadmap

Milestone status for paddock. Kept in the repo rather than in anyone's notes,
because both organisations need to see it.

| | Milestone | Status | Evidence |
|---|---|---|---|
| **M1** | Foundation and contribution contract | **done** | `make check`; `paddock doctor`; `tests/test_alt_site.py` |
| **M2** | Rescue untracked work into adapters and patches | **done (models)** | patches verified byte-identical; 4 model adapters converted |
| **M3** | Registries, `paddock setup`, data projection | partial | registries, projection and full de-hardcoding done; `paddock setup` not implemented |
| **M4** | Ingest existing results | **done** | 36 runs, 77 tests, golden values from the original reports |
| **M5** | Reproduce closed-loop (neuroNCAP) | **partial** | adapters proven identical to the original nodes; see `docs/benchmarks/neuroncap.md` |
| **M6** | Reproduce open-loop (KITScenes) | **done** | full n=223: 892/892 metric comparisons identical at full float precision |
| **M7** | NAVSIM benchmark adapter | **done** | reproduction exact (2379/2379 metrics identical, delta 0.0); 3 of 4 models scored, openpilot blocked by a 2 Hz dataset |

## What "done" means here, and what it does not

M1–M4 are verified in the sense that **the code does what it claims about data
that already exists**. No model has been run through paddock yet. The adapters
import cleanly, their contracts hold under test, and the ingest reproduces
published numbers — but inference has not happened.

That distinction matters, so it is spelled out:

| Claim | Verified how | Strength |
|---|---|---|
| Patches reproduce the working trees | clean checkout at pinned sha + apply → `diff` clean on every patched file | **proof** |
| Ingest matches published results | 25 golden assertions against the numbers in the original `report.md` files, not against paddock's own output | **strong** |
| `crop_box` unification is behaviour-preserving | KITScenes crop is byte-identical: `(1, 151, 3503, 2121)` | **strong** |
| Repo works at another site | `tests/test_alt_site.py` resolves a foreign layout; `test_portability.py` bans machine paths | **strong** |
| Adapters obey the Policy contract | per-model contract tests: frame convention, shapes, guards, reset semantics | **moderate** — no GPU, no weights |
| Adapters produce the same output as the original nodes | head-to-head on neuroNCAP (both return `2.8269616956051293`), and KITScenes n=223 with 892/892 metrics identical | **proof** |
| Adapters reproduce the **August** recorded numbers | drivejepa matches to full float precision; visionpilot differs because its node was edited after its last scored run | **strong** |

M5 closed the first of those rows. Running paddock's adapters against the same
renderer as the recorded August runs:

- **drivejepa reproduces exactly** — `2.0723314376255617` on frontal-0103,
  identical to what was recorded on 27 Aug. That exercises the whole pipeline:
  renderer, engine, node, scoring.
- **paddock's adapter and the original node agree to full float precision**, run
  head to head, so the conversion to the `Policy` contract is faithful.
- **alpamayo cannot be compared on a single run.** Its frontal score spans the
  full 0-5 scale across seeds (sd 1.817 against openpilot's 0.298); today's 5.0
  and August's 0.0 are both single draws from the same distribution. It samples
  unseeded by design.
- **visionpilot differs (2.827 against 3.145)** — because its node was edited
  1h44m *after* the last run that scored it. The current code had never been
  measured. drivejepa being untouched and reproducing exactly rules out any
  environmental cause. Detail in `docs/benchmarks/neuroncap.md`.

That last point is the failure mode version pinning exists to prevent, and the
reason this code now lives in git.

Two open items came out of M5, both in `docs/benchmarks/neuroncap.md`:

- the two openpilot paths disagree on `traffic_convention`, and the closed-loop
  value looks wrong for boston-seaport. Not changed, because fixing it
  invalidates every recorded closed-loop score.
- stochastic models need a run count chosen from their variance, not `--runs 1`.

Still unrun: the third scenario (`side`), scene 0108, and multi-run seeds.

## Remaining work

**M3 — `paddock setup`.** Clone or adopt vendor repos, build the `external/`
symlink farm, apply patches, run per-model `setup.sh`. Today `doctor` reports
what is missing; `setup` does not yet fix it.

**M5 — closed loop.** `paddock run --bench neuroncap --model all --runs 1
--scene 0103` must reproduce `ladder_current.log`:

| model | stationary | frontal | side |
|---|---:|---:|---:|
| openpilot | 5.0 | 1.862 | 5.0 |
| drivejepa | 5.0 | 2.072 | 5.0 |
| alpamayo | 5.0 | 0.000 | 1.176 |
| visionpilot | 3.145 | 1.684 | 1.074 |

Needs: the neuroNCAP benchmark adapter (rescued but not converted), the renderer
launcher, `runtime/proc.py` and `runtime/gpu.py`.

**M6 — done.** `paddock run --bench kitscenes --model openpilot` reproduces the
recorded `runs/full223` exactly: ADE 1.599424, median 1.322747, FDE@3s 2.893613,
with all 892 metric comparisons identical at full float precision.

Getting there found a real bug — the adapter was reusing the first scene's
camera calibration for all 223 samples — which a three-sample check had missed
because those three shared a rotation group. See `docs/benchmarks/kitscenes.md`.

The sample-list question is settled: all 223 keyframes are servable from the
configured tiers, 0 missing, and `samples()` refuses a partial set rather than
silently scoring fewer.

**M7 — NAVSIM.** `paddock run --bench navsim --model drivejepa` reproduces the
recorded run exactly: 2379/2379 metric comparisons bit-identical, mean EPDMS
delta 0.0. NAVSIM is now a registered benchmark with a third execution shape,
`ExternallyScoredBenchmark`, for benchmarks whose scorer owns the iteration.

Getting there found that **both of the vendor fork's baseline agents are broken
by the fork's own edits** and have never run in this tree — see
`docs/benchmarks/navsim.md`. Paddock ships a working constant-velocity baseline
rather than patching vendor source.

All three trajectory conversions were done and tested. The horizon was never
the hard part; the **observation rate** was, and the first read of it was wrong
in one of the two cases.

| model | NAVSIM | |
|---|---:|---|
| drivejepa | **0.84272** (n=12146) | via proxy; the comparable figure. 0.880 on the 242 subset |
| alpamayo15 | **0.60813** (n=12146) | `GRID_INDICES` widened to 8; frame selection fixed |
| visionpilot | **0.43122** (n=12146) | `PLAN_HORIZON_S` 3.0 -> 4.0. 0.447 on the 242 subset |
| openpilot | not measurable | no speed input; infers motion from 10 Hz frames |
| constant velocity | **0.23854** (n=12146) | reference baseline. 0.257 on the 242 subset |

Both grid changes are additive at the tail, so no recorded neuroNCAP or
KITScenes score moves: `to_wire` takes the first six poses and
`displacement_errors` truncates at 3 s. Pinned by tests.

**openpilot is genuinely blocked.** OpenScene is 2 Hz by construction and
supercombo's complete non-image input set is `{desire_pulse,
traffic_convention, action_t}` -- there is no channel to tell it its speed, so
it reads the sequence as a parked car and plans 0.30 m of travel in 4 s at
4.8 m/s. `neuroncap.md` recorded the same fact years of debugging earlier
("0.1 s is forced, not chosen"). Models now declare
`capabilities.min_input_hz` and benchmarks refuse the pairing, so a rate
artefact is never recorded as a result.

**alpamayo was gated on a mistake and is fine.** Its 4-scenario score of 0.226
-- below the 0.257 baseline -- was read as the same failure; 60 scenarios put
the identical configuration at 0.503. Sample size, not capability. But the
observation behind it was real: `_image_frames` resampled onto a 0.1 s grid
that 2 Hz data cannot support, returning two distinct images across four slots.
A paired A/B over 60 scenarios measured the cost at **+0.207 EPDMS**
(p = 0.030), almost all of it collisions avoided --
`no_at_fault_collisions` 0.708 -> 0.992. Shown four copies of one frame the
model cannot see other vehicles moving. Frame selection is now adaptive and
falls back only where the grid degenerates, which never happens at 10 Hz.

**Sample size is the lesson.** Two conclusions were overturned by running more
scenarios, in opposite directions: visionpilot 0.640 -> 0.447, alpamayo
0.226 -> 0.503. Sub-split runs are now stored with `valid: false` so they cannot
be ranked against a real one.

Three things found on the way, none NAVSIM-specific:

- **`[models.<name>] env` in paddock.toml was parsed and never applied**, so a
  site could write one, see no error, and get none of the effect. Fixed, with
  tests.
- **`gpu.serialize` is parsed and never enforced.** Only real free VRAM gates a
  node start. Left as-is -- it is what allowed the concurrent sweep -- but the
  key should either do something or go.
- **Two concurrent runs could score each other's model.** `/alive` only proves
  *a* node is answering; a run whose node was still loading passed its health
  check against a sibling's node on the same port and recorded the sibling's
  plans under its own name, with a matching score. Nodes now answer `/whoami`,
  the launcher verifies identity and refuses, and `PADDOCK_BASE_PORT` isolates
  concurrent runs. The bogus run is stored `valid: false`. Found only because
  the two smokes reported identical scores.

The full-split sweep runs three models concurrently on their own port ranges,
with `infer_sequence` overrides for drivejepa and alpamayo (one forward pass
per scenario, not four) and four nodes for visionpilot. Projected ~5 h wall
against ~26 h sequential.

## Open questions

**The NAVSIM headline — settled.** The CSV and its run log disagree (0.90673 vs
0.90488) over the same 242 scenarios, and re-running the vendor's own agent
through paddock reproduced *both* figures exactly. So this was never a
discrepancy to resolve: the two answer different questions, as
`src/paddock/results/ingest/navsim.py` argued. Paddock reports the recomputed
0.90673 and keeps 0.90488 as `source_reported`.

**What "navtest" means here — open, and a labelling error.** The stored August
run is labelled `navtest` but scored 242 of navtest's 12,146 scenarios: the
metric cache covers all 136 logs, sensor blobs cover 5. `preflight` now refuses
a partial split unless `PADDOCK_NAVSIM_ALLOW_PARTIAL=1` is set, so the label
cannot be produced accidentally again — but the existing row and the README
support matrix still carry the unqualified name. Closing it means ~95 GB of
`openscene_sensor_test_camera` shards. See `docs/benchmarks/navsim.md`.

**Drive-JEPA's NAVSIM agent is scorer-aware.** It receives `metric_cache` and
`past_ego_simulated_states` from the scoring loop and re-runs the PDM simulator
at inference time to re-rank its proposals by predicted two-frame comfort — a
component of EPDMS. No model behind an HTTP node can do that. Its recorded
0.90673 is therefore not a baseline the other three can be measured against;
the comparable set is runs through `paddock_proxy`.

## Bench2Drive / CARLA — adapter landed, not yet run

`adapters/benchmarks/bench2drive`: a model-agnostic CARLA `agent.py`, the
vendored dual-PID tracker, and a harness that fails a route on every failure
the evaluator distinguishes rather than scoring it. All four models
(openpilot, visionpilot, alpamayo15, drivejepa) declare it.

| Claim | Verified how | Strength |
|---|---|---|
| parse reproduces a recorded score | golden test on a real 0.0.4 `results.json` | **strong** |
| crashes fail, poor outcomes score | parametrised over the evaluator's status strings at 7ec25d1 | **strong** |
| launch, timeout, CARLA cleanup | stand-in evaluator under test; real flags read from the checkout | **strong** / argued |
| agent imports in the CARLA-client env | exercised in a Python 3.8 env with the 0.9.15 egg, no server | moderate |
| a model drives a route through paddock | nothing yet | **unverified** |

Needs a site with CARLA 0.9.15 and the pinned checkout wired in
`[benchmarks.bench2drive]`. Detail in `docs/benchmarks/bench2drive.md`.

## Beyond reproduction

Not started, and larger than everything above:

- **nuScenes open-loop** — needs `nuscenes_infos*.pkl`, which do not exist, and the
  local copy has no `sweeps/` and empty radar. Camera-only is viable; fusion is not.
- **alpamayo2, vjepa2, sunnypilot** — never evaluated.
- **Training** — the framework's eventual purpose; out of scope until evaluation is trustworthy.
