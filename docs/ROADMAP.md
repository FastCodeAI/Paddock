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
| **M7b** | NAVSIM v1 PDMS beside v2 EPDMS | **done** | constant velocity reproduces the paper's navtest row term by term (20.65 vs 20.6); v2 baseline re-run identical to 1e-11; drivejepa 0.892, alpamayo15 0.676 (replayed), visionpilot 0.488, n=12,146 each |

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

**M7b — NAVSIM v1 PDMS.** The same scenes scored by the v1.1 devkit vendored
beside the v2 one, as its own benchmark `navsim_v1` (own metric name, results
tree and site block), implemented as a subclass of the v2 harness and proxy.
Doing it found that the shared parser had one devkit generation's vocabulary
baked in, that the v2 harness kept its variant knobs as module constants, and
that the runner could not run an external scorer without a policy node -- the
constant-velocity record for v2 was an untracked one-off. All three are now
general: `ScoreVocabulary`, class attributes, and `nodeless.py`, which also
replays a run's recorded plans under another devkit (alpamayo15's v1 number
comes from the plans that produced its v2 number, no GPU).

| model | NAVSIM v1 PDMS | v2 EPDMS | |
|---|---:|---:|---|
| drivejepa | **0.892** (n=12146) | 0.843 | proxy; the paper's 93.3 is a v1 checkpoint through a scorer-aware agent |
| alpamayo15 | **0.676** (n=12146) | 0.608 | the v2 run's own plans, replayed |
| visionpilot | **0.488** (n=12146) | 0.431 | proxy, four nodes |
| constant velocity | **0.207** (n=12146) | 0.239 | paper: 20.6, every term matching |
| human | 0.946, `valid: false` | — | paper: 94.8; the gap is ego progress only |
| openpilot, 10 Hz history | **0.807** (n=12146) | 0.747 | PR #33's plans, replayed |
| alpamayo15, 10 Hz history | **0.753** (n=12146) | 0.691 | PR #33's plans, replayed |

**Verified:** the baseline row against the paper, term by term; the v2 path
unchanged (12,139/12,146 scenarios identical, the rest within 1e-11); n and
zero failures in every record. **Argued:** the human's 0.25-point gap is the
cache's reference plan; the fork's v1 tree is upstream v1.1 (byte-identical
scoring files; its occupancy-map edit is a no-op under shapely 2).

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
- **A run whose inputs were altered to fit a benchmark is ranked as if they
  were not** (#48). METEOR plans 3 s where NAVSIM scores 4 s, so its run set
  `METEOR_HORIZON_S=4.0` and a quarter of every scored trajectory is paddock's
  continuation of the model's last segment. `run.json` records it and the run
  id hashes it, but `paddock report` prints the row beside Drive-JEPA's with no
  marker: `notes` renders only for excluded runs, and `invalid_reasons()` is a
  benchmark hook that cannot see a model-side env var. `valid: false` is the
  wrong answer -- the measurement is sound, it is not comparable in kind.
  **Open.**

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

**M8 — comma2k19.** `paddock run --bench comma2k19 --model <m>` scores the full
released dataset: 2035 one-minute segments of 20 Hz California highway driving,
10,165 samples, six models on the board. It is the first benchmark here whose
sensor rate is high enough for openpilot to perceive its own motion, which is
what makes an openpilot open-loop number possible at all — NAVSIM's 2 Hz is
exactly the pairing `min_input_hz` refuses.

| model | horizon | ADE@3s | |
|---|---:|---:|---|
| alpamayo15 | 4 s | **0.900** | 2 of 4 camera slots filled; a mono dataset |
| qwendrive | 5 s | 1.159 | side slots replicated from the road camera |
| openpilot | 10 s | 1.507 | see the two re-sweeps below |
| constant velocity | 3 s | 1.639 | registered policy, 6 waypoints |
| visionpilot | 4 s | 4.045 | synthesised trajectory |
| drivejepa_free | 4 s | 16.048 | perception-free agent, NAVSIM v1 |

**Verified:** every row is over the *identical* sample set — `sequences_sha256`
`293a225a0b5cbaff` in all six run records, which is a stronger statement than
six matching values of `n` and is the evidence that makes the column
comparable. The 2033 × 5 = 10,165 derivation names the two truncated segments
rather than counting them. Pose maths is delegated to a pinned
`commaai/comma2k19` clone rather than reimplemented, and a reference
implementation in the test suite is asserted against it wherever that clone
exists. **Argued:** the calibration estimate (3.5–3.8° pitch on the RAV4,
1.7–1.8° on the Civic) uses calibrationd's own formula with recorded pose
substituted for the model's `cameraOdometry`; that substitution is reasoned
from, not measured against, openpilot's own calibration.

Two re-sweeps are owed on the openpilot row, and it carries a footnote in the
README and the benchmark doc until they land:

- **Re-sweep openpilot on comma2k19 at the `repos.yaml` pin.** The recorded
  1.507 was produced from a clone at `4df40d2c1` (split graph); the pin is
  `555f48c5d` (combined graph), and the two plan from different hidden states.
  The pin was left alone rather than moved to match, because moving it to a
  revision chosen to fit one row is how a pin stops meaning anything. Runs now
  record `repo_sha` and `provenance.repo.at_pin`, so this cannot recur unseen.
- **Re-sweep openpilot wherever the ONNX backend was used.** `_infer_onnx` fed
  the road camera's image to the wide view; the tinygrad path did not. Fixed,
  and the backend is now on the record as `provenance.policy.backend`, but rows
  produced before the fix are not bit-reproducible from this tree.

A third re-sweep is owed across every benchmark, and it is not openpilot's:

- **Re-sweep VisionPilot everywhere.** Its adapter carried `CURV_SCALE = 1.0`
  where the stack has 0.21, so `IDM`'s `sqrt(mu*g/|kappa|)` speed cap bound on a
  curvature 4.8x too large. Fixed September 2026; on a 20-sample comma2k19 smoke
  ADE@3s moved 15.09 -> 5.28. **Every VisionPilot row in the README and in
  `docs/models/comparison.md` predates it.** Expect the open-loop rows to move
  most: Bench2Drive and neuroNCAP are urban and slow enough that the cap rarely
  binds, which is how this survived them. Marked `∗` in the README until it
  lands.

Landed alongside it, none comma2k19-specific:

- **`drivejepa_free`, the eighth model adapter** — Drive-JEPA's perception-free
  agent as its own registry entry, because it is a different model from the
  perception-based v2 every other Drive-JEPA column refers to and sharing a row
  with it would misattribute both.
- **`paddock new` and `paddock verify` (M3).** Not `paddock setup` — no cloning,
  no patching — but the scaffolding half: `new` copies the `_template/` adapter
  and rewrites its names, `verify` checks an adapter against the contracts
  before a run rather than eight minutes into one. `setup` itself is still
  unimplemented and still exits 2 saying so.
- **Parallel open-loop nodes and open-loop `--resume`.** comma2k19 is the first
  open-loop benchmark large enough to need either: `node_count` runs N policy
  nodes against one harness, and a journal lets an interrupted sweep continue in
  place keeping its run id. This is the orchestration the nuScenes-openpilot
  item under "Beyond reproduction" asks for; what remains there is re-making
  that particular record with it.

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
the evaluator distinguishes rather than scoring it. All five models
(openpilot, visionpilot, alpamayo15, drivejepa, meteor) declare it.

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

- **nuScenes open-loop** — **done (camera-only); protocol transcribed and pinned by
  tests, parity with the UniAD/VAD scorers unverified (no fixture).** `paddock run --bench nuscenes`
  scores the UniAD/VAD L2 + collision protocol on the 150 val scenes, reading the
  `v1.0-trainval` JSON tables directly (no `nuscenes_infos*.pkl`, which do not
  exist here). Detail in `docs/benchmarks/nuscenes.md`. Fusion is still not viable
  (the local copy has no `sweeps/` and empty radar); camera-only is what runs.

Not started, and larger than everything above:

- **alpamayo2, vjepa2, sunnypilot** — never evaluated.
- **Drive-JEPA and VisionPilot cannot be drawn in a NAVSIM comparison.** Their
  full-split runs predate the proxy labelling its plan dumps with scenario
  tokens, so no plan can be tied to a scenario and `paddock render --bench
  navsim` names them as omitted. They are the two most useful rows to compare
  against (one leads the benchmark, one sits mid-field). Re-running them is the
  only fix: about 4 h and 3.5 h of proxy time each.
- **nuScenes openpilot record is not regenerable with tracked code** — the
  tracked run (`20260908T110028-b71f4f`) was assembled by four-worker site
  orchestration. The orchestration itself has now landed with M8 (`node_count`
  plus open-loop `--resume`), so what is left is narrower than it was: point it
  at nuScenes and re-make the record. Number not in question; provenance is.
- **Training** — the framework's eventual purpose; out of scope until evaluation is trustworthy.
